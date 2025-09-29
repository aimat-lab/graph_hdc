import os
import time
import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Literal, Dict

import umap
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from rich.pretty import pprint
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace
from chem_mat_data.processing import MoleculeProcessing, OneHotEncoder
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import rdmolops

# from visual_graph_datasets.data import nx_from_graph
from graph_hdc.models import HyperNet
from graph_hdc.special.molecules import graph_dict_from_mol
from graph_hdc.special.molecules import (
    make_molecule_node_encoder_map,
    make_molecule_node_encoder_map_cont,
)
from graph_hdc.special.molecules import (
    make_molecule_graph_encoder_map_cont,
)
import pandas as pd

DATASET_NAME: str = 'aqsoldb'
# :param DATASET_NAME_ID:
#       The name of the dataset to be used later on for the identification of the dataset. This name will NOT be used 
#       for the downloading of the dataset but only later on for identification. In most cases these will be the same 
#       but in cases for example one dataset is used as the basis of some deterministic calculation of the target values 
#       and in this case the name should identify it as such.
DATASET_NAME_ID: str = DATASET_NAME
DATASET_TYPE: str = 'regression'

# == EMBEDDING PARAMETERS ==

# :param EMBEDDING_SIZE:
#       The size of the graph embedding vectors. This will be the number of elements in each of the 
#       hypervectors that represent the individual molecular graphs.
EMBEDDING_SIZE: int = 2048
# :param NUM_LAYERS:
#       The number of layers in the hypernetwork. This parameter determines the depth of the hypernetwork
#       which is used to generate the graph embeddings. This means it is the number of message passing 
#       steps applied in the encoder.
NUM_LAYERS: int = 2
# :param BATCH_SIZE:
#       The size of the batches to be used during training. This parameter determines the number of samples
#       that are processed in parallel during the training of the model.
BATCH_SIZE: int = 8
# :param DEVICE:
#       The device to be used for the training of the model. This parameter can be set to 'cuda:0' to use the
#       GPU for training, or to 'cpu' to use the CPU.
#DEVICE: str = "cuda:0" if torch.cuda.is_available() else "cpu"
DEVICE: str = "cpu"
# :param ECODING_MODE:
#       This string determines the mode in which the HyperNet encoder operates in. The categorical mode
#       only uses categorical encodings for the node and graph features. The continuous mode is the newer 
#       version of the encoder that encodes certain features with the FHRR continuous encodings. 
ENCODING_MODE: Literal['categorical', 'continuous'] = 'continuous'

# == VISUALIZATION PARAMETERS ==

# :param PLOT_UMAP:
#       A boolean flag that determines whether to plot the UMAP dimensionality reduction of the HDC vectors
#       for the molecular graphs in the dataset.
PLOT_UMAP: bool = False

# == EXPERIMENT PARAMETERS ==

experiment = Experiment.extend(
    'predict_molecules.py',
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
    This hook is supposed to implement the processing of the dataset into a suitable vector representation
    that can then be used to train the prediction models on. It gives the dataset in the format of the 
    `index_data_map: Dict[int, dict]` and expects that each dict element in that dataset is updated with 
    a "graph_features" entry that contains the vector representation of the graph.
    """
    
    # --- dataset statistics ---
    # In the first step we need to determine the datasets statistics. Primarily we need the information
    # about the maximum graph size and the maximum graph diameter to be known for the dataset because
    # the encoding of the continuous graph features needs to know these values.
    # Since this is information for which we need to loop through the entire dataset but does not change 
    # for each dataset we obviously want to cache that.
    @experiment.cache.cached(name=f'stats_{e.DATASET_NAME}')
    def dataset_statistics() -> dict:
        
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
    pprint(stats)
    
    # --- Constructing HyperNet encoder ---
    # We want to differentiate two modes here: The categorical only mode which is the one that is required 
    # for the decoding and the continuous mode which is the one that performs better for the regression 
    # tasks.
    
    # updated mode with better regression performance.
    if e.ENCODING_MODE == 'continuous':
        
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
        
    # The previous mode.
    elif e.ENCODING_MODE == 'categorical':
        
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

    # --- processing dataset ---
    # After having constructed the HyperNet encoder, we can now use it to process the entire dataset
    # and generate the HDC vectors for each of the molecular graphs in the dataset.
    @experiment.cache.cached(name=f'hdc_{e.DATASET_NAME}__seed_{e.SEED}__size_{e.EMBEDDING_SIZE}__depth_{e.NUM_LAYERS}')
    def process_dataset():
        
        # --- graph representations ---
        # Before we can do the actual forward pass of the encoder, we need to convert the individual
        # elements of the dataset into the graph representations. Only on those graph 
        # representations, the encoder can operate.
        e.log('processing molecules into graphs...')
        time_start_process = time.time()
        graphs: List[dict] = []
        for c, (index, data) in enumerate(index_data_map.items()):
            
            smiles: str = data['graph_repr']
            mol: Chem.Mol = Chem.MolFromSmiles(smiles)
            
            graph = graph_dict_from_mol(mol)
            
            del graph['graph_labels']
            index_data_map[index].update(graph)
            
            graphs.append(graph)
            
            if c % 1000 == 0:
                e.log(f' * {c} molecules done')
                
        time_end_process = time.time()
        e.log(f'processed the dataset after {time_end_process - time_start_process:.2f} seconds')
            
        # --- forward pass ---
        # Finally, the forward pass of the HyperNet encoder can be done on the graph representations
        # to generate the HDC vectors for each of the molecular graphs in the dataset. We then save 
        # that vector representation as the additonal "graph_features" entry in the graph dict, as 
        # this is expected as the outcome of this hook implementation.
        e.log('doing the model forward pass on all the graphs...')
        
        time_start_forward = time.time()
        results = hyper_net.forward_graphs(graphs, batch_size=600)
        for (index, graph), result in zip(index_data_map.items(), results):
            index_data_map[index]['graph_features'] = result['graph_embedding']
            
        time_end_forward = time.time()
        e.log(f'done the model forward pass after {time_end_forward - time_start_forward:.2f} seconds')
            
        return index_data_map
    
    index_data_map_processed = process_dataset()
    for index in index_data_map:
        index_data_map[index]['graph_features'] = index_data_map_processed[index]['graph_features']


@experiment.hook('after_dataset', replace=False, default=False)
def after_dataset(e: Experiment,
                  index_data_map: dict,
                  **kwargs
                  ) -> None:
    
    if e.PLOT_UMAP:
        
        e.log('plotting UMAP dimensionality reduction...')
        
        # First of all we need to collect all the HDC vectors for the various graphs in the dataset
        hvs = [data['graph_features'] for data in index_data_map.values()]
        
        reducer = umap.UMAP(
            n_components=2, 
            random_state=e.SEED,
            metric='cosine',
            min_dist=0.0,
            n_neighbors=100,
        )
        reduced = reducer.fit_transform(hvs)
        
        # Extract the class labels from the graph dicts
        if e.DATASET_TYPE == 'regression':
            labels = [data['graph_labels'][0] for data in index_data_map.values()]
            
        if e.DATASET_TYPE == 'classification':
            labels = [np.argmax(data['graph_labels']) for data in index_data_map.values()]
                    
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(8, 6))
        ax.set_title('UMAP reduction of HDC vectors\n'
                     '')
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        
        # Calculate the 0.05 and 0.95 percentiles
        vmin, vmax = np.percentile(labels, [2, 98])
        
        # Clip the labels to the 0.05 and 0.95 percentiles
        clipped_labels = np.clip(labels, vmin, vmax)
        
        scatter = ax.scatter(
            reduced[:, 0], reduced[:, 1], 
            c=clipped_labels, 
            marker='.',
            cmap='bwr', 
            alpha=0.5,
            edgecolors='none',
            s=10  # Adjust the size of the scatter points
        )
        
        # # Add a color bar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('target')
        
        fig_path = os.path.join(e.path, 'umap_reduction.png')
        fig.savefig(fig_path, dpi=600)
    

experiment.run_if_main()