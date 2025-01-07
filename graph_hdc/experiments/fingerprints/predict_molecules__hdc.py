import os
import torch
import torch.nn as nn
from torch import Tensor
from typing import List

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

# from visual_graph_datasets.data import nx_from_graph
from graph_hdc.models import HyperNet
from graph_hdc.special.molecules import graph_dict_from_mol
from graph_hdc.special.molecules import make_molecule_node_encoder_map

DATASET_NAME: str = 'aqsoldb'
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
BATCH_SIZE: int = 50
# :param DEVICE:
#       The device to be used for the training of the model. This parameter can be set to 'cuda:0' to use the
#       GPU for training, or to 'cpu' to use the CPU.
#DEVICE: str = "cuda:0" if torch.cuda.is_available() else "cpu"
DEVICE: str = "cpu"

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
def process_dataset(e: Experiment,
                    index_data_map: dict
                    ) -> None:
    
    node_encoder_map = make_molecule_node_encoder_map(
        dim=e.EMBEDDING_SIZE,
    )
    
    hyper_net = HyperNet(
        hidden_dim=e.EMBEDDING_SIZE,
        depth=e.NUM_LAYERS,
        device=e.DEVICE,
        node_encoder_map=node_encoder_map,
    )    

    e.log('encoding molecular graphs as hyperdimensional vectors...')
    graphs: List[dict] = []
    for c, (index, data) in enumerate(index_data_map.items()):
        smiles: str = data['graph_repr']
        mol: Chem.Mol = Chem.MolFromSmiles(smiles)
        
        graph = graph_dict_from_mol(mol)
        del graph['graph_labels']
        index_data_map[index].update(graph)
        
        graphs.append(graph)
        
    results = hyper_net.forward_graphs(graphs)
    for (index, graph), result in zip(index_data_map.items(), results):
        index_data_map[index]['graph_features'] = result['graph_embedding']
        
        
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