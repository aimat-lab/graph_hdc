import os
import tempfile
from typing import List

import torch
import numpy as np
import rdkit.Chem as Chem
from rich.pretty import pprint
from graph_hdc.special.molecules import graph_dict_from_mol
from graph_hdc.special.molecules import AtomEncoder
from graph_hdc.special.molecules import make_molecule_node_encoder_map
from graph_hdc.models import HyperNet
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from graph_hdc.graph import data_list_from_graph_dicts
from .utils import ARTIFACTS_PATH



def test_graph_dict_from_mol_basically_works():
    """
    The graph_dict_from_mol function should return a dictionary with the expected keys for a viable 
    graph dict representation when given a rdkit.Mol instance.
    """
    mol = Chem.MolFromSmiles('CCO')
    graph = graph_dict_from_mol(mol)
    
    pprint(graph)
    assert isinstance(graph, dict)
    assert 'node_atoms' in graph
    assert 'node_degrees' in graph
    assert 'node_valences' in graph
    
    
class TestAtomEncoder():
    """
    Test cases for the AtomEncoder class.
    """
    
    def test_construction_basically_works(self):
        """
        object instance should be able to be constructed without any errors.
        """
        encoder = AtomEncoder(dim=100, atoms=['C', 'N', 'O'])
        assert encoder.dim == 100
        assert encoder.num_categories == 4 # 3 including the "unknown" case
        assert len(encoder.embeddings) == 4
        assert encoder.embeddings.shape[1] == 100
        
        pprint(encoder.atom_index_map)

    def test_encode_basically_works(self):
        """
        Encoding a string atom symbol should return a tensor of the correct shape.
        """
        dim = 100
        encoder = AtomEncoder(dim=dim, atoms=['C', 'N', 'O'])
        hv = encoder.encode('C')
        
        assert isinstance(hv, torch.Tensor)
        assert hv.shape[0] == dim
        
    def test_encode_with_numpy_array_works(self):
        """
        Encoding should also work when fetching a string element from a numpy array of strings 
        as it will later be in the actual use case.
        """
        dim = 100
        encoder = AtomEncoder(dim=dim, atoms=['C', 'N', 'O'])
        atoms: np.ndarray = np.array(['C', 'N', 'O'], dtype=str)
        atom: str = atoms[0]
        hv = encoder.encode(atom)
        
        assert isinstance(hv, torch.Tensor)
        assert hv.shape[0] == dim
        
    def test_decode_basically_works(self):
        """
        Decoding should return a atomic number integer when given a tensor.
        """
        dim = 100
        encoder = AtomEncoder(dim=dim, atoms=['C', 'N', 'O'])
        hv = encoder.encode('C')
        atom = encoder.decode(hv)
        
        assert isinstance(atom, int)
        assert atom == 6
        

class TestMoleculeEncoding():
    """
    Test cases not for any specific class but instead for the encoding of molecules in general.
    """
    
    def test_make_molecule_node_encoder_map_basically_works(self):
        """
        The make_molecule_node_encoder_map function should return a dictionary with the expected keys 
        for a viable node_encoder_map when given a list of atom symbols.
        """
        dim = 100
        node_encoder_map = make_molecule_node_encoder_map(dim=dim, atoms=['C', 'N', 'O'])
        
        assert isinstance(node_encoder_map, dict)
        # We generally want to encode these three node properties
        assert 'node_atoms' in node_encoder_map
        assert 'node_degrees' in node_encoder_map
        assert 'node_valences' in node_encoder_map
        
    def test_encode_molecule_to_hypervector(self):
        """
        If it is possible to encode a mol object constructed from a SMILES all the way into a graph hyper 
        vector using the special molecule processing pipeline.
        """
        dim = 100
        node_encoder_map = make_molecule_node_encoder_map(dim=dim)
        
        # We can construct the HyperNet encoder with the special molecule node encoder map
        hyper_net = HyperNet(
            hidden_dim=dim,
            depth=3,
            node_encoder_map=node_encoder_map,
        )
        
        # We can construct a graph dict from a SMILES string using the special 
        # graph_dict_from_mol function
        graph: dict = graph_dict_from_mol(Chem.MolFromSmiles('CCO'))
        
        # Finally, the encoder net supports the direct conversion of a graph dict to a hyper vector with 
        # the forward_graphs method.
        results: List[dict] = hyper_net.forward_graphs([graph]) 
        result: dict = results[0]
        pprint(result)
        
        assert isinstance(result, dict)
        assert isinstance(result['graph_embedding'], np.ndarray)
        graph_embedding = result['graph_embedding']
        assert graph_embedding.shape == (dim, )
        
    def test_saving_loading_hyper_net_works(self):
        """
        If it is possible to save and load a HyperNet instance to and from a file when using the molecule 
        specific node encoder map.
        """
        dim = 100
        node_encoder_map: dict = make_molecule_node_encoder_map(dim=dim)
        hyper_net = HyperNet(
            hidden_dim=dim,
            depth=3,
            node_encoder_map=node_encoder_map,
        )
        
        with tempfile.TemporaryDirectory() as path:
            file_path = os.path.join(path, 'hyper_net.pth')
            hyper_net.save_to_path(file_path)
            
            hyper_net_loaded = HyperNet.load(file_path)
            assert isinstance(hyper_net_loaded, HyperNet)
            
    def test_reconstruct_molecule(self):
        """
        Test if a molecule graph can be reconstructed from its hypervector.
        """
        # Create molecule and obtain its graph dict representation
        mol = Chem.MolFromSmiles('CCC(N)CCO')
        graph_dict = graph_dict_from_mol(mol)
        
        # Setup HyperNet with molecule-specific node encoder map (using atoms available in the molecule)
        dim = 50_000
        node_encoder_map = make_molecule_node_encoder_map(dim=dim, atoms=['C', 'N', 'O'])
        hyper_net = HyperNet(
            hidden_dim=dim,
            depth=3,
            node_encoder_map=node_encoder_map,
        )
        
        # Convert graph dict to PyG data object and compute the graph embedding
        data_list = data_list_from_graph_dicts([graph_dict])
        data = next(iter(DataLoader(data_list, batch_size=1)))
        result = hyper_net.forward(data)
        graph_embedding = result['graph_embedding']
        
        # Reconstruct graph dict from the graph hypervector
        rec_dict = hyper_net.reconstruct(
            graph_embedding, 
            learning_rate=1.0,
            num_iterations=25,
            batch_size=10,
            low=0.0,
            high=1.0,
        )
        
        # Convert reconstructed graph dict to a networkx graph
        rec_g = nx.Graph()
        for node in rec_dict['node_indices']:
            # Use reconstructed atom info if available, otherwise use a placeholder
            atom = rec_dict.get('node_atoms', ['?'] * len(rec_dict['node_indices']))[node]
            rec_g.add_node(node, atom=atom)
        for edge in rec_dict['edge_indices']:
            rec_g.add_edge(int(edge[0]), int(edge[1]))
        
        # Convert the original graph dict to a networkx graph for visualization
        orig_g = nx.Graph()
        for idx, atom in enumerate(graph_dict.get('node_atoms', ['?'] * len(graph_dict.get('node_atoms', [])))):
            orig_g.add_node(idx, atom=atom)
        for edge in graph_dict.get('edge_indices', []):
            orig_g.add_edge(int(edge[0]), int(edge[1]))
        
        # Plot original and reconstructed graphs side by side
        
        atom_color_map = {
            6: 'gray',  # Carbon
            7: 'blue',  # Nitrogen
            8: 'red',   # Oxygen
            17: 'green', # Chlorine
            16: 'yellow', # Sulfur
            15: 'orange', # Phosphorus
        }
        
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        pos_orig = nx.spring_layout(orig_g, seed=42)
        nx.draw(
            orig_g, pos_orig, 
            ax=axs[0], 
            with_labels=True, 
            labels={i: atom for i, atom in enumerate(graph_dict['node_atoms'])},
            node_color=[atom_color_map.get(atom, 'black') for atom in graph_dict['node_atoms']]
        )
        axs[0].set_title('Original Molecule Graph')
        
        pos_rec = nx.spring_layout(rec_g, seed=42)
        nx.draw(
            rec_g, pos_rec, 
            ax=axs[1], 
            with_labels=True, 
            labels={i: atom for i, atom in enumerate(rec_dict['node_atoms'])},
            node_color=[atom_color_map.get(atom, 'black') for atom in rec_dict['node_atoms']]
        )
        axs[1].set_title('Reconstructed Molecule Graph')
        plt.tight_layout()
        # Optionally, save figure to a desired artifact path
        fig_path = os.path.join(ARTIFACTS_PATH, 'reconstructed_molecule_graph.png')
        plt.savefig(fig_path)
        plt.close()
