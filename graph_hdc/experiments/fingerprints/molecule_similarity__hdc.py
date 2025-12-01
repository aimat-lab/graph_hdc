"""
Molecular Similarity Experiment with HDC Encoding

This module extends the molecule_similarity.py base experiment to use hyperdimensional
computing (HDC) for molecular encoding. Molecules are encoded into high-dimensional
hypervectors using a HyperNet message passing network, and similarity is computed
using cosine distance.

Key Features:
    - HyperNet encoding with configurable dimensionality and depth
    - Support for both categorical and continuous encoding modes
    - Cosine distance metric for similarity computation
    - Caching of expensive encoding operations

Usage:
    Create a YAML configuration file to run this experiment:

    .. code-block:: yaml

        extend: molecule_similarity__hdc.py
        parameters:
            DATASET_NAME: "qm9_smiles"
            NUM_SAMPLES: 10
            NUM_NEIGHBORS: 5
            EMBEDDING_SIZE: 2048
            NUM_LAYERS: 2
            ENCODING_MODE: "continuous"
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

DATASET_NAME: str = 'aqsoldb'
DATASET_NAME_ID: str = DATASET_NAME

# == EMBEDDING PARAMETERS ==

# :param EMBEDDING_SIZE:
#       The size of the graph embedding vectors. This will be the number of elements
#       in each of the hypervectors that represent the individual molecular graphs.
#       Larger sizes can capture more information but increase computational cost.
EMBEDDING_SIZE: int = 2048

# :param NUM_LAYERS:
#       The number of layers in the hypernetwork. This parameter determines the depth
#       of the hypernetwork which is used to generate the graph embeddings. This means
#       it is the number of message passing steps applied in the encoder. More layers
#       allow information to propagate further through the graph structure.
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
    'molecule_similarity.py',
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
    Process the dataset using HyperNet to generate HDC embeddings.

    This hook replaces the base implementation and uses a HyperNet encoder to
    convert molecular graphs into high-dimensional hypervectors. The process
    involves:
        1. Computing dataset statistics (max graph size, max diameter)
        2. Constructing node and graph encoders based on ENCODING_MODE
        3. Creating a HyperNet model
        4. Processing all molecules through the HyperNet (with caching)
        5. Storing the resulting embeddings as 'graph_features'

    The encoding is cached to avoid recomputation across multiple runs.

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

        e.log('computing dataset statistics...')
        sizes: List[int] = []
        diameters: List[int] = []

        for index, graph in index_data_map.items():
            smiles: str = graph['graph_repr']
            mol: Chem.Mol = Chem.MolFromSmiles(smiles)
            adj = rdmolops.GetAdjacencyMatrix(mol)
            graph_nx = nx.from_numpy_array(adj)

            sizes.append(len(graph_nx.nodes))
            diameters.append(nx.diameter(graph_nx))

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
        # Continuous mode with FHRR encodings
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

    # Store encoder as private instance attribute (NOT in serializable storage)
    # This prevents JSON serialization errors and ensures proper access
    e._hyper_net_encoder = hyper_net
    e._dataset_stats = stats

    # --- Processing Dataset ---
    # After constructing the HyperNet encoder, we use it to process the entire
    # dataset and generate the HDC vectors for each molecular graph. This is
    # cached to avoid recomputation.
    @experiment.cache.cached(
        name=f'hdc_{e.DATASET_NAME}__seed_{e.SEED}__size_{e.EMBEDDING_SIZE}__depth_{e.NUM_LAYERS}__mode_{e.ENCODING_MODE}'
    )
    def process_dataset_cached():

        # Convert molecules to graph representations
        e.log('processing molecules into graphs...')
        time_start_process = time.time()
        graphs: List[dict] = []

        for c, (index, data) in enumerate(index_data_map.items()):
            smiles: str = data['graph_repr']
            mol: Chem.Mol = Chem.MolFromSmiles(smiles)

            graph = graph_dict_from_mol(mol)

            # Remove graph_labels if present (not needed for similarity)
            if 'graph_labels' in graph:
                del graph['graph_labels']

            index_data_map[index].update(graph)
            graphs.append(graph)

            if c % 1000 == 0 and c > 0:
                e.log(f' * {c} molecules processed')

        time_end_process = time.time()
        e.log(f'processed {len(graphs)} molecules after '
              f'{time_end_process - time_start_process:.2f} seconds')

        # Forward pass through HyperNet
        e.log('encoding molecules with HyperNet...')
        time_start_forward = time.time()

        results = hyper_net.forward_graphs(graphs, batch_size=e.BATCH_SIZE)

        for (index, graph), result in zip(index_data_map.items(), results):
            index_data_map[index]['graph_features'] = result['graph_embedding']

        time_end_forward = time.time()
        e.log(f'encoded {len(graphs)} molecules after '
              f'{time_end_forward - time_start_forward:.2f} seconds')

        return index_data_map

    index_data_map_processed = process_dataset_cached()

    # Update the graph_features in the original index_data_map
    for index in index_data_map:
        index_data_map[index]['graph_features'] = index_data_map_processed[index]['graph_features']


@experiment.hook('encode_molecule', replace=True, default=False)
def encode_molecule(e: Experiment,
                   smiles: str
                   ) -> np.ndarray:
    """
    Encode a single molecule using the existing HyperNet encoder.

    This hook reuses the HyperNet encoder that was created during the initial
    process_dataset call, avoiding the expensive re-initialization of the encoder.
    The encoder is retrieved from e._hyper_net_encoder (private instance attribute).

    :param e: The experiment instance.
    :param smiles: SMILES string of the molecule to encode.

    :return: HDC vector as numpy array, or None if encoding fails.
    """
    # Retrieve the stored HyperNet encoder from private instance attribute
    hyper_net = getattr(e, '_hyper_net_encoder', None)
    if hyper_net is None:
        raise RuntimeError(
            "HyperNet encoder not found. Make sure process_dataset was called first."
        )

    try:
        # Convert SMILES to RDKit mol
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Convert to graph dict
        graph = graph_dict_from_mol(mol)

        # Remove graph_labels if present
        if 'graph_labels' in graph:
            del graph['graph_labels']

        # Encode using the existing HyperNet encoder
        results = hyper_net.forward_graphs([graph], batch_size=1)

        # Extract the graph embedding
        if len(results) > 0 and 'graph_embedding' in results[0]:
            return results[0]['graph_embedding']
        else:
            return None

    except Exception as ex:
        # Log error but don't crash - return None to indicate failure
        return None


@experiment.hook('compute_distance', replace=True, default=False)
def compute_distance(e: Experiment,
                    features1: np.ndarray,
                    features2: np.ndarray
                    ) -> float:
    """
    Compute cosine distance between two HDC vectors.

    Cosine distance is defined as:
        cosine_distance = 1 - cosine_similarity

    Where cosine_similarity is:
        cosine_similarity = (a · b) / (||a|| * ||b||)

    This metric measures the angle between two vectors in high-dimensional space.
    It ranges from 0 (identical direction) to 2 (opposite directions), though in
    practice with normalized vectors it ranges from 0 to 1.

    :param e: The experiment instance.
    :param features1: First HDC vector.
    :param features2: Second HDC vector.

    :return: Cosine distance (lower = more similar).
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
    cosine_distance = 1 - cosine_similarity

    return float(cosine_distance)


experiment.run_if_main()
