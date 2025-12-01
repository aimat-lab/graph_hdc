"""
HDC-based Bayesian Optimization Molecular Search Experiment

This experiment extends the base optimize_molecule_bo.py experiment to use
hyperdimensional computing (HDC) encodings for molecular representations.

The experiment uses the HyperNet encoder from the graph_hdc package to convert
molecular graphs into high-dimensional hypervectors, which are then used as
inputs to the Gaussian Process models in the Bayesian Optimization loop.

Key Features:
    - Continuous FHRR encodings for better regression performance
    - Message passing with configurable depth
    - High-dimensional representations (default 2048D)
    - Dataset statistics computation for proper scaling
    - Works seamlessly with BotTorch's GP models

Usage:
    Run directly or create configuration YAML files:

    .. code-block:: yaml

        extend: optimize_molecule_bo__hdc.py
        parameters:
            DATASET_NAME: "aqsoldb"
            TARGET_INDEX: 0
            TARGET_VALUE: 5.0
            EMBEDDING_SIZE: 2048
            NUM_LAYERS: 2
            NUM_INITIAL_SAMPLES: 10
            NUM_BO_ROUNDS: 20
            ACQUISITION_FUNCTION: "EI"

Example:
    .. code-block:: bash

        # Run with debug mode
        python optimize_molecule_bo__hdc.py

        # Run with configuration
        python -m pycomex run optimize_molecule_bo__hdc__aqsoldb.yml
"""
import os
import time
from typing import List, Literal

import numpy as np
import networkx as nx
from rdkit import Chem
from rdkit.Chem import rdmolops

from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace
from chem_mat_data._typing import GraphDict

from graph_hdc.models import HyperNet
from graph_hdc.special.molecules import graph_dict_from_mol
from graph_hdc.special.molecules import (
    make_molecule_node_encoder_map,
    make_molecule_node_encoder_map_cont,
    make_molecule_graph_encoder_map_cont,
)

# == HDC PARAMETERS ==

# :param EMBEDDING_SIZE:
#       The size of the hypervector embeddings. This determines the dimensionality
#       of the HDC representation. Higher dimensions can encode more information
#       but increase computational cost. For GP fitting, moderate sizes (2048-4096)
#       typically provide good performance.
EMBEDDING_SIZE: int = 2048

# :param NUM_LAYERS:
#       The number of message passing layers in the HyperNet encoder. This controls
#       the depth of information aggregation in the molecular graph. More layers
#       allow information to propagate further across the molecular structure.
NUM_LAYERS: int = 2

# :param ENCODING_MODE:
#       The encoding mode for the HyperNet. 'categorical' uses only discrete encodings,
#       while 'continuous' uses FHRR encodings for continuous features which often
#       performs better for regression tasks and Gaussian Process modeling.
ENCODING_MODE: Literal['categorical', 'continuous'] = 'continuous'

# :param DEVICE:
#       The device to use for computation ('cpu' or 'cuda:0'). If CUDA is available,
#       using GPU can significantly speed up the encoding process.
DEVICE: str = 'cpu'

# :param BATCH_SIZE:
#       The batch size for HyperNet forward pass. Larger batches are more efficient
#       but require more memory.
BATCH_SIZE: int = 600

# == EXPERIMENT ==

experiment = Experiment.extend(
    'optimize_molecule_bo.py',
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals()
)


@experiment.hook('process_dataset', replace=True, default=False)
def process_dataset(e: Experiment,
                    index_data_map: dict[int, GraphDict]
                    ) -> None:
    """
    Process molecules into HDC hypervector representations using HyperNet.

    This hook converts SMILES strings into graph representations, computes dataset
    statistics needed for continuous encodings, and then uses the HyperNet encoder
    to generate hypervector embeddings for all molecules.

    The HDC representation provides a high-dimensional continuous space that is
    particularly well-suited for Gaussian Process modeling due to:
    1. Smooth representation space (no discrete jumps)
    2. Distributed encoding (information spread across all dimensions)
    3. Compositionality (similar structures have similar representations)
    4. Natural uncertainty quantification through GP posterior

    The process involves:
    1. Computing dataset statistics (max graph size, max diameter) for FHRR encodings
    2. Constructing the HyperNet encoder with appropriate node and graph encoders
    3. Converting SMILES to graph dictionaries
    4. Running forward pass to generate hypervectors
    5. Storing hypervectors in 'graph_features' field

    :param e: The experiment instance providing access to parameters and logging.
    :param index_data_map: Dictionary mapping indices to graph dictionaries. This
        dictionary is modified in-place to add 'graph_features' to each entry.

    :return: None. Modifies index_data_map in-place by adding 'graph_features' key.
    """
    e.log(f'HDC process_dataset called with {len(index_data_map)} molecules')

    # === DATASET STATISTICS ===
    # For the continuous encoding mode, we need to know the maximum graph size
    # and diameter in the dataset to properly scale the FHRR encodings.

    @experiment.cache.cached(
        name=f'hdc_stats_{e.DATASET_NAME}__'
             f'numdata_{e.NUM_DATA}__'
             f'seed_{e.SEED}'
    )
    def dataset_statistics() -> dict:
        """
        Compute dataset statistics needed for FHRR encodings.

        The statistics are used to properly scale continuous encodings such as
        graph size and diameter, ensuring that the FHRR representations are
        well-calibrated to the dataset.

        :return: Dictionary with 'size' and 'diameter' statistics including
            min, max, mean, and median values.
        """
        e.log('computing dataset statistics for FHRR encodings...')
        sizes: List[int] = []
        diameters: List[int] = []

        for index, graph in index_data_map.items():
            smiles: str = graph['graph_repr']
            mol: Chem.Mol = Chem.MolFromSmiles(smiles)
            adj = rdmolops.GetAdjacencyMatrix(mol)
            graph_nx = nx.from_numpy_array(adj)

            sizes.append(len(graph_nx.nodes))
            diameters.append(nx.diameter(graph_nx))

        stats = {
            'size': {
                'min': min(sizes),
                'max': max(sizes),
                'mean': sum(sizes) / len(sizes),
                'median': sorted(sizes)[len(sizes) // 2],
            },
            'diameter': {
                'min': min(diameters),
                'max': max(diameters),
                'mean': sum(diameters) / len(diameters),
                'median': sorted(diameters)[len(diameters) // 2],
            }
        }
        e.log(f'dataset statistics: max_size={stats["size"]["max"]}, '
              f'max_diameter={stats["diameter"]["max"]}')
        return stats

    stats: dict = dataset_statistics()

    # === CONSTRUCT HYPERNET ENCODER ===

    if e.ENCODING_MODE == 'continuous':
        # Continuous mode with FHRR encodings for better regression performance
        # and smoother GP modeling
        node_encoder_map = make_molecule_node_encoder_map_cont(
            dim=e.EMBEDDING_SIZE,
            seed=e.SEED,
        )
        graph_encoder_map = make_molecule_graph_encoder_map_cont(
            dim=e.EMBEDDING_SIZE,
            seed=e.SEED,
            max_graph_size=stats['size']['max'],
            max_graph_diameter=stats['diameter']['max'],
        )

    elif e.ENCODING_MODE == 'categorical':
        # Categorical mode with only discrete encodings
        # May result in less smooth GP posteriors
        node_encoder_map = make_molecule_node_encoder_map(
            dim=e.EMBEDDING_SIZE,
            seed=e.SEED,
        )
        graph_encoder_map = {}

    e.log('creating HyperNet encoder...')
    e.log(f' * DEVICE: {e.DEVICE}')
    e.log(f' * EMBEDDING_SIZE: {e.EMBEDDING_SIZE}')
    e.log(f' * NUM_LAYERS: {e.NUM_LAYERS}')
    e.log(f' * ENCODING_MODE: {e.ENCODING_MODE}')

    hyper_net = HyperNet(
        hidden_dim=e.EMBEDDING_SIZE,
        depth=e.NUM_LAYERS,
        device=e.DEVICE,
        node_encoder_map=node_encoder_map,
        graph_encoder_map=graph_encoder_map,
        seed=e.SEED,
        normalize_all=True,
    )

    # Save the encoder for potential future use
    model_path = os.path.join(e.path, 'hyper_net.pth')
    hyper_net.save_to_path(model_path)
    e.log(f'saved HyperNet encoder to {model_path}')

    # === PROCESS DATASET ===
    # Cache the entire processing step since it can be time-consuming

    @experiment.cache.cached(
        name=f'hdc_embeddings_{e.DATASET_NAME}__'
             f'numdata_{e.NUM_DATA}__'
             f'seed_{e.SEED}__size_{e.EMBEDDING_SIZE}__depth_{e.NUM_LAYERS}__'
             f'mode_{e.ENCODING_MODE}'
    )
    def process_dataset_cached():
        """
        Convert molecules to HDC embeddings and cache the results.

        This caching strategy avoids recomputing embeddings when running
        multiple BO experiments on the same dataset with the same
        encoding parameters.

        :return: Dictionary mapping indices to embeddings.
        """
        # Convert SMILES to graph representations
        e.log('converting SMILES to graph representations...')
        time_start_convert = time.time()

        graphs: List[dict] = []
        for c, (index, data) in enumerate(index_data_map.items()):
            smiles: str = data['graph_repr']
            mol: Chem.Mol = Chem.MolFromSmiles(smiles)

            graph = graph_dict_from_mol(mol)
            del graph['graph_labels']  # Remove labels since we don't need them

            # Update the index_data_map with graph structure info
            index_data_map[index].update(graph)
            graphs.append(graph)

            if c % 1000 == 0:
                e.log(f' * converted {c} molecules')

        time_end_convert = time.time()
        e.log(f'converted molecules after {time_end_convert - time_start_convert:.2f} seconds')

        # Run HyperNet forward pass
        e.log(f'running HyperNet forward pass on {len(graphs)} graphs...')
        e.log(f'batch_size: {e.BATCH_SIZE}')
        time_start_forward = time.time()

        results = hyper_net.forward_graphs(graphs, batch_size=e.BATCH_SIZE)

        e.log(f'forward_graphs returned {len(results)} results')
        if len(results) != len(graphs):
            e.log(f'WARNING: Result count mismatch! graphs={len(graphs)}, results={len(results)}')

        embeddings_dict = {}
        for (index, graph), result in zip(index_data_map.items(), results):
            embeddings_dict[index] = result['graph_embedding']

        e.log(f'embeddings_dict contains {len(embeddings_dict)} entries')

        # Check for index mismatches
        expected_indices = set(index_data_map.keys())
        actual_indices = set(embeddings_dict.keys())
        missing = expected_indices - actual_indices
        extra = actual_indices - expected_indices

        if missing:
            e.log(f'WARNING: {len(missing)} indices missing from embeddings_dict: {list(missing)[:10]}...')
        if extra:
            e.log(f'WARNING: {len(extra)} unexpected indices in embeddings_dict: {list(extra)[:10]}...')

        time_end_forward = time.time()
        e.log(f'completed forward pass after {time_end_forward - time_start_forward:.2f} seconds')

        return embeddings_dict

    # Get the embeddings (from cache if available)
    embeddings_dict = process_dataset_cached()

    # Store embeddings in the graph dictionaries
    for index in index_data_map:
        index_data_map[index]['graph_features'] = embeddings_dict[index]

    stored_count = sum(1 for idx in index_data_map if 'graph_features' in index_data_map[idx])
    e.log(f'successfully stored graph_features for {stored_count}/{len(index_data_map)} molecules')


experiment.run_if_main()
