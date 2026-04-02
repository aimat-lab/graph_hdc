"""
HDC-based property prediction for transition metal complexes.

Extends :mod:`predict_tmcs` by replacing ``process_dataset`` with a
:class:`~graph_hdc.models.HyperNet` encoder that maps TMC graphs to
high-dimensional hypervectors.

The encoding decomposes the dense 91-dim ``node_attributes`` (from
``MetalOrganicProcessing``) back into separate symbolic properties
(atoms, degrees, is_metal, formal_charge, d_electrons) and binds their
hypervector representations via circular convolution.

Graph-level properties (size, diameter, total_charge) are encoded with
continuous FHRR encoders and added to the embedding.
"""
import os
import time
from typing import List, Literal

import numpy as np
import networkx as nx
import torch
import umap
import matplotlib.pyplot as plt
from rich.pretty import pprint
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

from graph_hdc.models import HyperNet
from graph_hdc.special.tmcs import (
    graph_dict_from_tmc_graph,
    make_tmc_node_encoder_map,
    make_tmc_node_encoder_map_cont,
    make_tmc_graph_encoder_map_cont,
    TMC_ATOMIC_NUMBERS,
)

# == DEFAULT PARAMETERS ==

DATASET_NAME: str = 'tmqmg'
DATASET_NAME_ID: str = DATASET_NAME
DATASET_TYPE: str = 'regression'

# == EMBEDDING PARAMETERS ==

# :param EMBEDDING_SIZE:
#       Dimensionality of the hypervectors representing each TMC graph.
EMBEDDING_SIZE: int = 2048
# :param NUM_LAYERS:
#       Number of message-passing iterations in the HyperNet encoder.
NUM_LAYERS: int = 2
# :param BATCH_SIZE:
#       Batch size for the HyperNet forward pass.
BATCH_SIZE: int = 8
# :param DEVICE:
#       Device for HyperNet computation.
DEVICE: str = "cpu"
# :param ENCODING_MODE:
#       ``'categorical'`` uses purely categorical encoders (required for
#       decoding).  ``'continuous'`` uses FHRR for ``node_degrees``.
ENCODING_MODE: Literal['categorical', 'continuous'] = 'continuous'
# :param HYPER_NET_LAYER_NORMALIZE:
#       Whether to L2-normalise node embeddings after each message-passing
#       layer.
HYPER_NET_LAYER_NORMALIZE: bool = True

# == VISUALIZATION ==

# :param PLOT_UMAP:
#       Whether to produce a UMAP scatter plot of the HDC embeddings.
PLOT_UMAP: bool = False

# == EXPERIMENT SETUP ==

experiment = Experiment.extend(
    'predict_tmcs.py',
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)


@experiment.hook('process_dataset', replace=True, default=False)
def process_dataset(e: Experiment, index_data_map: dict) -> None:
    """
    Encode TMC graphs into hypervectors using :class:`HyperNet`.

    Steps:

    1. Compute dataset statistics (graph sizes, diameters, charges) — cached.
    2. Build node and graph encoder maps for the TMC feature set.
    3. Decompose each graph's dense ``node_attributes`` into separate
       symbolic property arrays via :func:`graph_dict_from_tmc_graph`.
    4. Run the HyperNet forward pass to obtain graph embeddings.
    5. Store the embedding as ``graph_features`` on each graph dict.
    """
    # --- dataset statistics (cached) ---
    @experiment.cache.cached(name=f'tmc_stats_{e.DATASET_NAME}')
    def dataset_statistics() -> dict:
        sizes: List[int] = []
        diameters: List[int] = []
        total_charges: List[float] = []

        for index, graph in index_data_map.items():
            num_nodes = len(graph['node_indices'])
            sizes.append(num_nodes)

            edge_indices = graph['edge_indices']
            G = nx.Graph()
            G.add_nodes_from(range(num_nodes))
            for i, j in edge_indices:
                G.add_edge(int(i), int(j))

            if num_nodes <= 1 or G.number_of_edges() == 0:
                diameters.append(0)
            elif nx.is_connected(G):
                diameters.append(nx.diameter(G))
            else:
                largest_cc = max(nx.connected_components(G), key=len)
                diameters.append(nx.diameter(G.subgraph(largest_cc)))

            total_charges.append(abs(float(graph['graph_attributes'][1])))

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
            },
            'total_charge': {
                'min': min(total_charges),
                'max': max(total_charges),
            },
        }

    stats: dict = dataset_statistics()
    pprint(stats)

    # --- encoder maps ---
    if e.ENCODING_MODE == 'continuous':
        node_encoder_map = make_tmc_node_encoder_map_cont(
            dim=e.EMBEDDING_SIZE,
            atoms=TMC_ATOMIC_NUMBERS,
            seed=e.SEED,
        )
    elif e.ENCODING_MODE == 'categorical':
        node_encoder_map = make_tmc_node_encoder_map(
            dim=e.EMBEDDING_SIZE,
            atoms=TMC_ATOMIC_NUMBERS,
            seed=e.SEED,
        )

    graph_encoder_map = make_tmc_graph_encoder_map_cont(
        dim=e.EMBEDDING_SIZE,
        max_graph_size=stats['size']['max'],
        max_graph_diameter=stats['diameter']['max'],
        max_total_charge=max(stats['total_charge']['max'], 6.0),
        seed=e.SEED,
    )

    # --- HyperNet construction ---
    e.log('creating HyperNet encoder...')
    e.log(f' * DEVICE: {e.DEVICE}')
    e.log(f' * EMBEDDING_SIZE: {e.EMBEDDING_SIZE}')
    e.log(f' * NUM_LAYERS: {e.NUM_LAYERS}')
    e.log(f' * ENCODING_MODE: {e.ENCODING_MODE}')
    e.log(f' * HYPER_NET_LAYER_NORMALIZE: {e.HYPER_NET_LAYER_NORMALIZE}')

    hyper_net = HyperNet(
        hidden_dim=e.EMBEDDING_SIZE,
        depth=e.NUM_LAYERS,
        device=e.DEVICE,
        node_encoder_map=node_encoder_map,
        graph_encoder_map=graph_encoder_map,
        seed=e.SEED,
        normalize_all=True,
        layer_normalize=e.HYPER_NET_LAYER_NORMALIZE,
    )

    e.log('saving HyperNet encoder to disk...')
    model_path = os.path.join(e.path, 'hyper_net.pth')
    hyper_net.save_to_path(model_path)

    # --- process dataset (cached) ---
    cache_key = (
        f'tmc_hdc_{e.DATASET_NAME}'
        f'__seed_{e.SEED}'
        f'__size_{e.EMBEDDING_SIZE}'
        f'__depth_{e.NUM_LAYERS}'
        f'__enc_{e.ENCODING_MODE}'
        f'__lnorm_{e.HYPER_NET_LAYER_NORMALIZE}'
    )

    @experiment.cache.cached(name=cache_key)
    def process():
        e.log('decomposing TMC graphs into symbolic property arrays...')
        time_start = time.time()
        graphs: List[dict] = []
        for c, (index, data) in enumerate(index_data_map.items()):
            graph = graph_dict_from_tmc_graph(data)
            del graph['graph_labels']
            index_data_map[index].update(graph)
            graphs.append(graph)

            if c % 1000 == 0:
                e.log(f' * {c} TMCs done')

        e.log(f'decomposed in {time.time() - time_start:.2f}s')

        e.log('running HyperNet forward pass...')
        time_start = time.time()
        results = hyper_net.forward_graphs(graphs, batch_size=600)
        for (index, _), result in zip(index_data_map.items(), results):
            index_data_map[index]['graph_features'] = result['graph_embedding']

        e.log(f'forward pass done in {time.time() - time_start:.2f}s')
        return index_data_map

    index_data_map_processed = process()
    for index in index_data_map:
        index_data_map[index]['graph_features'] = (
            index_data_map_processed[index]['graph_features']
        )


@experiment.hook('after_dataset', replace=False, default=False)
def after_dataset(e: Experiment, index_data_map: dict, **kwargs) -> None:
    """
    Optional UMAP visualisation of the HDC embeddings.
    """
    if e.PLOT_UMAP:
        e.log('plotting UMAP dimensionality reduction...')

        hvs = [data['graph_features'] for data in index_data_map.values()]

        reducer = umap.UMAP(
            n_components=2,
            random_state=e.SEED,
            metric='cosine',
            min_dist=0.0,
            n_neighbors=100,
        )
        reduced = reducer.fit_transform(hvs)

        if e.DATASET_TYPE == 'regression':
            labels = [data['graph_labels'][0] for data in index_data_map.values()]
        elif e.DATASET_TYPE == 'classification':
            labels = [np.argmax(data['graph_labels']) for data in index_data_map.values()]

        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(8, 6))
        ax.set_title('UMAP reduction of TMC HDC vectors')
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')

        vmin, vmax = np.percentile(labels, [2, 98])
        clipped_labels = np.clip(labels, vmin, vmax)

        scatter = ax.scatter(
            reduced[:, 0], reduced[:, 1],
            c=clipped_labels,
            marker='.',
            cmap='bwr',
            alpha=0.5,
            edgecolors='none',
            s=10,
        )
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('target')

        fig_path = os.path.join(e.path, 'umap_reduction.png')
        fig.savefig(fig_path, dpi=600)


experiment.run_if_main()
