"""
This experiment will use a certain molecular dataset as the basis for the reconstruction 
of the molecular formula based on the hdc fingerprints and will evaluate the accuracy 
of this reconstruction.
"""
import os
import csv
import time
import random
from typing import Union
from collections import Counter

import torch
import numpy as np
import rdkit.Chem as Chem
from chem_mat_data import load_graph_dataset
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

from graph_hdc.models import HyperNet
from graph_hdc.special.molecules import graph_dict_from_mol, make_molecule_node_encoder_map


# == DATASET PARAMETERS ==

SEED: int = 0
DATASET_NAME: str = 'aqsoldb'
NUM_DATA: Union[int, float] = 100

# == ENCODING PARAMETERS ==

EMBEDDING_SIZE: int = 25_000
NUM_LAYERS: int = 2

# == EXPERIMENT PARAMETERS ==

PERIODIC_TABLE = Chem.GetPeriodicTable()
__DEBUG__: bool = False

experiment = Experiment(
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)

def atom_composition_from_smiles(smiles: str) -> dict[str, int]:
    mol: Chem.Mol = Chem.MolFromSmiles(smiles)
    atom_counts = Counter(atom.GetSymbol() for atom in mol.GetAtoms())
    return dict(atom_counts)

def atom_composition_from_constraints(constraints: list[dict]) -> dict[str, int]:
    atom_counts = Counter()
    for constraint in constraints:
        atom_symbol = PERIODIC_TABLE.GetElementSymbol(constraint['src']['node_atoms'])
        atom_counts[atom_symbol] += constraint['num']
        
    return dict(atom_counts)


@experiment.hook('load_dataset', replace=False, default=True)
def load_dataset(e: Experiment) -> dict[int, dict]:
    
    ## --- Dataset Loading ---
    # This function will download the dataset from the ChemMatData file share and return the already pre-processed 
    # list of graph dict representations.
    graphs: list[dict] = load_graph_dataset(
        e.DATASET_NAME,
        folder_path='/tmp'
    )
    
    # We associate each graph element in the dataset with a unique index so that they will be recognizable 
    # throughout the experiment.
    index_data_map: dict[int, dict] = dict(enumerate(graphs))
    
    ## --- Filtering ---
    # We want to make sure that all of the elements in the dataset are actually valid molecules with 
    # this filtering step. Essentially we want to remove all single atoms, disconnected graphs etc. 
    # just to be sure.
    
    e.log('filtering dataset to remove invalid SMILES and unconnected graphs...')
    indices = list(index_data_map.keys())
    for index in indices:
        graph = index_data_map[index]
        smiles = graph['graph_repr']
        
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            del index_data_map[index]
            
        elif len(mol.GetAtoms()) < 2:
            del index_data_map[index]
            
        elif len(mol.GetBonds()) < 1:
            del index_data_map[index]
            
        # disconnected graphs
        elif '.' in smiles:
            del index_data_map[index]
            
        del graph['node_indices']
        del graph['node_attributes']
        del graph['edge_indices']
        del graph['edge_attributes']
    
    return index_data_map


@experiment.hook('get_graph_labels')
def get_graph_labels(e: Experiment,
                     index: int,
                     graph: dict
                     ) -> np.ndarray:
    return graph['graph_labels']


@experiment
def experiment(e: Experiment):
    
    e.log('starting experiment...')
    e.log(f'using the seed: {e.SEED}')
    
    ## --- loading dataset ---
    # At first we need to load the dataset on which to base the reconstructions on from the 
    # ChemMatData file share.
    e.log(f'loading the dataset with the name {e.DATASET_NAME} ...')
    index_data_map_all: dict[int, dict] = e.apply_hook(
        'load_dataset',
    )
    e.log(f'✅ loaded dataset with {len(index_data_map_all)} samples')

    # After loading all of the elements of the dataset we do the sub-sampling only to 
    # a subset of the dataset to speed up the experiment.
    num_data = e.NUM_DATA
    if isinstance(num_data, float):
        num_data = int(len(index_data_map_all) * e.NUM_DATA)

    num_data = min(num_data, len(index_data_map_all))
    e.log(f'sampling {num_data} samples from the dataset...')
    
    random.seed(e.SEED)
    indices_sampled: int = random.sample(list(index_data_map_all.keys()), k=num_data)
    index_data_map: dict[int, dict] = {
        index: index_data_map_all[index] 
        for index in indices_sampled
    }

    ## --- encoding dataset ---
    # In the next step we need to go through the various samples of the dataset and 
    # encode them into an HDC vector representation. This is done by using the HyperNet 
    # model.
    node_encoder_map = make_molecule_node_encoder_map(
        dim=e.EMBEDDING_SIZE,
    )
    
    e.log('creating HyperNet encoder...')
    e.log(f' * EMBEDDING_SIZE: {e.EMBEDDING_SIZE}')
    e.log(f' * NUM_LAYERS: {e.NUM_LAYERS}')
    hyper_net = HyperNet(
        hidden_dim=e.EMBEDDING_SIZE,
        depth=e.NUM_LAYERS,
        node_encoder_map=node_encoder_map,
    )
    
    e.log('💾 saving HyperNet encoder to disk...')
    model_path = os.path.join(e.path, 'hyper_net.pth')
    hyper_net.save_to_path(model_path)

    e.log('processing molecules into graphs...')
    graphs: list[dict] = []
    for c, (index, data) in enumerate(index_data_map.items()):
        
        smiles: str = data['graph_repr']
        mol: Chem.Mol = Chem.MolFromSmiles(smiles)
        
        graph = graph_dict_from_mol(mol)
        del graph['graph_labels']
        index_data_map[index].update(graph)
        
        graphs.append(graph)
        
        if c % 1000 == 0:
            e.log(f' * {c} molecules done')
        
    e.log(f'doing the model forward pass on {len(graphs)} graphs...')
    time_start_encoding = time.time()
    results = hyper_net.forward_graphs(graphs, batch_size=600)
    for (index, graph), result in zip(index_data_map.items(), results):
        index_data_map[index].update(result)
        
    duration_encoding = time.time() - time_start_encoding
    e['duration/encoding'] = duration_encoding
    e.log(f'✅ done encoding the dataset in {duration_encoding:.2f} seconds')

    ## --- reconstructing molecules ---
    # After encoding the dataset into the hdc vectors we can now use the decoding capabilities 
    # of the HyperNet model to reconstruct all of the nodes in the graph dicts.

    e.log('reconstructing the graphs from the hypervectors...')
    time_start_reconstruction = time.time()
    for index, graph in index_data_map.items():
        graph_hv_stack = graph['graph_hv_stack']
        
        # This method will take either the final graph embedding or the whole graph hv stack 
        # across the different message passing depths as an input and return a list of 
        # dictionaries which represent the nodes of the reconstructed graph.
        order_zero_constraints: list[dict] = hyper_net.decode_order_zero(
            torch.tensor(graph_hv_stack)
        )
        graph['order_zero_constraints'] = order_zero_constraints

    duration_reconstruction = time.time() - time_start_reconstruction
    e['duration/reconstruction'] = duration_reconstruction
    e.log(f'✅ done reconstructing the graphs in {duration_reconstruction:.2f} seconds')

    ## --- evaluating composition reconstruction ---
    # the easiest thing to compare is the composition of the molecule, meaning which atoms are 
    # present and also the number of atoms of each type.
    
    for index, graph in index_data_map.items():
        
        # First we need to compute the true composition of the molecule from the graph_repr aka 
        # the SMILES string of the molecule which this function does for us.
        atom_counts_true: dict[str, int] = atom_composition_from_smiles(graph['graph_repr'])
        graph['atom_counts_true'] = atom_counts_true
        
        # Then we can get the recovered composition (from the reconstruction) of the molecule 
        # from the order zero constraints which were computed in the previous step.
        atom_counts_pred: dict[str, int] = atom_composition_from_constraints(graph['order_zero_constraints'])
        graph['atom_counts_pred'] = atom_counts_pred

    # Finally, the accuracy can be calculated as the number of times that the true and predicted 
    # compositions are exactly the same.
    acc_composition: float = sum(
        int(set(graph['atom_counts_true']) == set(graph['atom_counts_pred']))
        for index, graph in index_data_map.items()
    ) / len(index_data_map)
    print(f'💡 composition reconstruction accuracy: {acc_composition:.2%}')
    e['metrics/acc/composition'] = acc_composition
    
    ## --- saving results ---
    # In the end we want to save all the reconstructions to the disk as a JSON file so that 
    # they can be analyzed later on.
    
    e.log('saving the results to the disk...')
    results = []
    for index, graph in index_data_map.items():
        results.append({
            'index': index,
            'smiles': graph['graph_repr'],
            'composition_true': list(set(graph['atom_counts_true'].items())),
            'composition_pred': list(set(graph['atom_counts_pred'].items())),
        })
    
    e.commit_json('result.json', results)

experiment.run_if_main()