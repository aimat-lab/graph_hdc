"""
Similarity-Based Bioactivity Prediction with HDC Encoding

This module extends the predict_bioactivity.py base experiment to use hyperdimensional
computing (HDC) for molecular encoding. Molecules are encoded into high-dimensional
hypervectors using a HyperNet message passing network, and similarity is computed
using cosine distance.

This implementation enables evaluation of HDC representations for virtual screening
tasks, comparing their performance against traditional molecular fingerprints for
identifying bioactive compounds through similarity-based ranking.

Key Features:
    - HyperNet encoding with configurable dimensionality and depth
    - Support for both categorical and continuous encoding modes
    - Cosine distance metric for similarity computation
    - Caching of expensive encoding operations
    - Standard virtual screening evaluation protocol

Usage:
    Create a YAML configuration file to run this experiment:

    .. code-block:: yaml

        extend: predict_bioactivity__hdc.py
        parameters:
            DATASET_NAME: "bl_chembl_reg"
            NUM_QUERY_ACTIVES: 5
            NUM_REPETITIONS: 50
            EMBEDDING_SIZE: 2048
            NUM_LAYERS: 2
            ENCODING_MODE: "continuous"
            SEED: 1

Expected Performance:
    - ECFP4 baseline typically achieves AUC ~0.75-0.85
    - HDC performance depends on embedding size and depth
    - BEDROC(α=20) emphasizes early recognition crucial for virtual screening
"""
import os
import time
from typing import List, Literal

import numpy as np
import networkx as nx
from rdkit import Chem
from rdkit.Chem import rdmolops
from rich.pretty import pprint

from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

from graph_hdc.models import HyperNet
from graph_hdc.special.molecules import graph_dict_from_mol
from graph_hdc.special.molecules import (
    make_molecule_node_encoder_map,
    make_molecule_node_encoder_map_cont,
    make_molecule_graph_encoder_map_cont,
)

# == DATASET PARAMETERS ==
# These are inherited from the base experiment but can be overridden

DATASET_NAME: str = 'bl_chembl_reg'
DATASET_NAME_ID: str = DATASET_NAME

# == EMBEDDING PARAMETERS ==

# :param EMBEDDING_SIZE:
#       The size of the graph embedding vectors. This will be the number of elements
#       in each of the hypervectors that represent the individual molecular graphs.
#       Larger sizes can capture more information but increase computational cost.
#       Typical values: 1024, 2048, 4096.
EMBEDDING_SIZE: int = 2048

# :param NUM_LAYERS:
#       The number of layers in the hypernetwork. This parameter determines the depth
#       of the hypernetwork which is used to generate the graph embeddings. This means
#       it is the number of message passing steps applied in the encoder. More layers
#       allow information to propagate further through the graph structure.
#       Typical values: 2-4.
NUM_LAYERS: int = 2

# :param BATCH_SIZE:
#       The size of the batches to be used during encoding. This parameter determines
#       the number of samples that are processed in parallel during the forward pass
#       of the HyperNet. Larger batches are more efficient but require more memory.
BATCH_SIZE: int = 600

# :param DEVICE:
#       The device to be used for encoding. This parameter can be set to 'cuda:0' to
#       use the GPU for encoding, or to 'cpu' to use the CPU.
DEVICE: str = "cpu"

# :param ENCODING_MODE:
#       This string determines the mode in which the HyperNet encoder operates. The
#       categorical mode only uses categorical encodings for the node and graph features.
#       The continuous mode is the newer version of the encoder that encodes certain
#       features with the FHRR continuous encodings, which typically performs better
#       for similarity tasks.
ENCODING_MODE: Literal['categorical', 'continuous'] = 'continuous'

# == EXPERIMENT PARAMETERS ==

experiment = Experiment.extend(
    'predict_bioactivity.py',
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals()
)


@experiment.hook('process_dataset', replace=True, default=False)
def process_dataset(
    e: Experiment,
    index_data_map: dict
) -> None:
    """
    Process the bioactivity dataset using HyperNet to generate HDC embeddings.

    This hook replaces the base implementation and uses a HyperNet encoder to
    convert molecular graphs into high-dimensional hypervectors. The process
    involves:
        1. Computing dataset statistics (max graph size, max diameter)
        2. Constructing node and graph encoders based on ENCODING_MODE
        3. Creating a HyperNet model
        4. Processing all molecules through the HyperNet (with caching)
        5. Storing the resulting embeddings as 'graph_features'

    The encoding is cached to avoid recomputation across multiple runs with the
    same dataset and hyperparameters.

    :param e: The experiment instance.
    :param index_data_map: Dictionary to be modified in-place with 'graph_features'.

    :return: None. Modifies index_data_map in-place.
    """
    # --- Dataset Statistics ---
    # We need to determine the maximum graph size and diameter for the dataset
    # because the continuous encoding requires these values. This information is
    # cached to avoid recomputation.
    @experiment.cache.cached(name=f'stats_{e.DATASET_NAME}')
    def dataset_statistics() -> dict:

        e.log('computing dataset statistics for HDC encoding...')
        sizes: List[int] = []
        diameters: List[int] = []

        for index, graph in index_data_map.items():
            smiles: str = graph['graph_repr']
            mol: Chem.Mol = Chem.MolFromSmiles(smiles)
            adj = rdmolops.GetAdjacencyMatrix(mol)
            graph_nx = nx.from_numpy_array(adj)

            sizes.append(len(graph_nx.nodes))

            # Handle disconnected graphs (shouldn't happen after filtering)
            if nx.is_connected(graph_nx):
                diameters.append(nx.diameter(graph_nx))
            else:
                # For disconnected graphs, use max diameter of components
                components = nx.connected_components(graph_nx)
                max_diameter = max(
                    nx.diameter(graph_nx.subgraph(comp))
                    for comp in components
                )
                diameters.append(max_diameter)

        return {
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

    stats: dict = dataset_statistics()
    e.log('dataset statistics:')
    pprint(stats)

    # --- Constructing HyperNet Encoder ---
    # We differentiate two modes: categorical (required for decoding) and
    # continuous (better performance for similarity tasks).

    if e.ENCODING_MODE == 'continuous':
        # Continuous mode with FHRR encodings (recommended for bioactivity prediction)
        e.log('using continuous encoding mode with FHRR')
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
        # Categorical mode (previous implementation)
        e.log('using categorical encoding mode')
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

    e.log('saving HyperNet encoder to disk...')
    model_path = os.path.join(e.path, 'hyper_net.pth')
    hyper_net.save_to_path(model_path)

    # --- Processing Dataset ---
    # After constructing the HyperNet encoder, we use it to process the entire
    # dataset and generate the HDC vectors for each molecular graph. This is
    # cached to avoid recomputation across runs with the same parameters.
    @experiment.cache.cached(
        name=f'hdc_{e.DATASET_NAME}__seed_{e.SEED}__size_{e.EMBEDDING_SIZE}__depth_{e.NUM_LAYERS}__mode_{e.ENCODING_MODE}'
    )
    def process_dataset_cached():

        # Convert molecules to graph representations
        e.log('processing molecules into graph dictionaries...')
        time_start_process = time.time()
        graphs: List[dict] = []

        for c, (index, data) in enumerate(index_data_map.items()):
            smiles: str = data['graph_repr']
            mol: Chem.Mol = Chem.MolFromSmiles(smiles)

            graph = graph_dict_from_mol(mol)

            # Preserve multi-label classification labels from bl_chembl_cls
            # (graph_labels contains 36 binary values: 35 targets + 1 decoy indicator)
            if 'graph_labels' in data:
                graph['graph_labels'] = data['graph_labels']

            index_data_map[index].update(graph)
            graphs.append(graph)

            if c % 1000 == 0 and c > 0:
                e.log(f' * {c} molecules processed')

        time_end_process = time.time()
        e.log(f'processed {len(graphs)} molecules into graph dicts after '
              f'{time_end_process - time_start_process:.2f} seconds')

        # Forward pass through HyperNet
        e.log('encoding molecules with HyperNet...')
        time_start_forward = time.time()

        results = hyper_net.forward_graphs(graphs, batch_size=e.BATCH_SIZE)

        for (index, graph), result in zip(index_data_map.items(), results):
            # Store HDC embedding as graph_features
            index_data_map[index]['graph_features'] = result['graph_embedding']

        time_end_forward = time.time()
        e.log(f'encoded {len(graphs)} molecules with HyperNet after '
              f'{time_end_forward - time_start_forward:.2f} seconds')

        return index_data_map

    index_data_map_processed = process_dataset_cached()

    # Update the graph_features and graph_labels in the original index_data_map
    for index in index_data_map:
        index_data_map[index]['graph_features'] = index_data_map_processed[index]['graph_features']
        # Preserve graph_labels for bl_chembl_cls dataset
        if 'graph_labels' in index_data_map_processed[index]:
            index_data_map[index]['graph_labels'] = index_data_map_processed[index]['graph_labels']

    e.log(f'all {len(index_data_map)} molecules now have HDC embeddings and labels')


@experiment.hook('compute_distance', replace=True, default=False)
def compute_distance(e: Experiment,
                    features1: np.ndarray,
                    features2: np.ndarray
                    ) -> float:
    """
    Compute cosine distance between two HDC vectors.

    Cosine distance is the standard similarity metric for hyperdimensional
    computing representations. It measures the angle between two vectors in
    high-dimensional space, making it robust to vector magnitude differences.

    Cosine distance is defined as:
        cosine_distance = 1 - cosine_similarity

    Where cosine_similarity is:
        cosine_similarity = (a · b) / (||a|| * ||b||)

    This metric ranges from 0 (identical direction) to 1 (orthogonal vectors)
    for normalized vectors, with lower values indicating greater similarity.

    :param e: The experiment instance.
    :param features1: First HDC vector (typically D-dimensional where D=EMBEDDING_SIZE).
    :param features2: Second HDC vector.

    :return: Cosine distance in [0, 1]. Lower values indicate more similar molecules.

    Example:

    .. code-block:: python

        # Two identical vectors have distance 0
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([1.0, 0.0, 0.0])
        dist = compute_distance(e, vec1, vec2)  # Returns 0.0

        # Two orthogonal vectors have distance 1
        vec1 = np.array([1.0, 0.0])
        vec2 = np.array([0.0, 1.0])
        dist = compute_distance(e, vec1, vec2)  # Returns 1.0
    """
    # Normalize vectors to unit length
    norm1 = np.linalg.norm(features1)
    norm2 = np.linalg.norm(features2)

    # Avoid division by zero
    if norm1 == 0 or norm2 == 0:
        return 1.0  # Maximum distance for zero vectors

    # Compute cosine similarity
    cosine_similarity = np.dot(features1, features2) / (norm1 * norm2)

    # Convert to distance (0 = identical, 1 = orthogonal)
    # Clip to [0, 1] to handle numerical precision issues
    cosine_distance = np.clip(1 - cosine_similarity, 0.0, 1.0)

    return float(cosine_distance)


experiment.run_if_main()
