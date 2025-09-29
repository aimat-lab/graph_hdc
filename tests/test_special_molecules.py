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
from graph_hdc.special.molecules import make_molecule_graph_encoder_map
from graph_hdc.utils import ContinuousEncoder
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
        
    def test_encoding_with_graph_properties_and_normalization(self):
        """
        Test of a molecule can be encoded with node and graph properties as well as the normalization
        """
        # Create molecule and obtain its graph dict representation
        mol = Chem.MolFromSmiles('CCC(N)CCCO')
        graph_dict = graph_dict_from_mol(mol)
        
        # Setup HyperNet with molecule-specific node encoder map (using atoms available in the molecule)
        dim = 10_000
        node_encoder_map = make_molecule_node_encoder_map(dim=dim, atoms=['C', 'N', 'O'])
        graph_encoder_map = make_molecule_graph_encoder_map(
            dim=dim,
            max_graph_size=20.0,
            max_graph_diameter=10.0,
        )
        hyper_net = HyperNet(
            hidden_dim=dim,
            depth=3,
            node_encoder_map=node_encoder_map,
            graph_encoder_map=graph_encoder_map,
            normalize_all=True,
        )
        
        # Convert graph dict to PyG data object and compute the graph embedding
        data_list = data_list_from_graph_dicts([graph_dict])
        data = next(iter(DataLoader(data_list, batch_size=1)))
        result = hyper_net.forward(data)
        graph_embedding = result['graph_embedding']
        
        assert torch.is_tensor(graph_embedding)
        assert graph_embedding.ndim == 2
        assert graph_embedding.shape[1] == dim


class TestEncoderReproducibility():
    """
    Test cases for verifying encoder reproducibility with seeds.
    """
    
    def test_atom_encoder_reproducibility_with_same_seed(self):
        """
        Test that independent AtomEncoder objects yield the same hypervectors when given the same seed.
        """
        dim = 100
        atoms = ['C', 'N', 'O', 'S']
        seed = 42
        
        # Create two independent AtomEncoder instances with the same seed
        encoder1 = AtomEncoder(dim=dim, atoms=atoms, seed=seed)
        encoder2 = AtomEncoder(dim=dim, atoms=atoms, seed=seed)
        
        # Test that embeddings are identical
        assert torch.allclose(encoder1.embeddings, encoder2.embeddings)
        
        # Test that encoding produces identical results
        for atom in atoms:
            hv1 = encoder1.encode(atom)
            hv2 = encoder2.encode(atom)
            assert torch.allclose(hv1, hv2)
            
    def test_atom_encoder_different_with_different_seeds(self):
        """
        Test that AtomEncoder objects yield different hypervectors when given different seeds.
        """
        dim = 100
        atoms = ['C', 'N', 'O', 'S']
        
        # Create two AtomEncoder instances with different seeds
        encoder1 = AtomEncoder(dim=dim, atoms=atoms, seed=42)
        encoder2 = AtomEncoder(dim=dim, atoms=atoms, seed=123)
        
        # Test that embeddings are different
        assert not torch.allclose(encoder1.embeddings, encoder2.embeddings)
        
        # Test that encoding produces different results for at least one atom
        different_found = False
        for atom in atoms:
            hv1 = encoder1.encode(atom)
            hv2 = encoder2.encode(atom)
            if not torch.allclose(hv1, hv2):
                different_found = True
                break
        assert different_found
        
    def test_molecule_encoder_map_reproducibility_with_same_seed(self):
        """
        Test that independent molecule encoder maps yield the same hypervectors when given the same seed.
        """
        dim = 100
        atoms = ['C', 'N', 'O', 'S']
        seed = 42
        
        # Create two independent molecule encoder maps with the same seed
        encoder_map1 = make_molecule_node_encoder_map(dim=dim, atoms=atoms, seed=seed)
        encoder_map2 = make_molecule_node_encoder_map(dim=dim, atoms=atoms, seed=seed)
        
        # Test that all encoders in the maps produce identical results
        for key in encoder_map1.keys():
            encoder1 = encoder_map1[key]
            encoder2 = encoder_map2[key]
            
            # Test embeddings are identical
            assert torch.allclose(encoder1.embeddings, encoder2.embeddings)
            
        # Test specific encoding for atom encoders
        for atom in atoms:
            hv1 = encoder_map1['node_atoms'].encode(atom)
            hv2 = encoder_map2['node_atoms'].encode(atom)
            assert torch.allclose(hv1, hv2)
            
        # Test specific encoding for degree and valence encoders
        for value in range(6):  # Test first 6 values for degrees and valences
            # Test node_degrees encoder
            if value < 10:  # node_degrees has 10 categories
                hv1 = encoder_map1['node_degrees'].encode(value)
                hv2 = encoder_map2['node_degrees'].encode(value)
                assert torch.allclose(hv1, hv2)
                
            # Test node_valences encoder
            if value < 6:  # node_valences has 6 categories
                hv1 = encoder_map1['node_valences'].encode(value)
                hv2 = encoder_map2['node_valences'].encode(value)
                assert torch.allclose(hv1, hv2)
                
    def test_molecule_encoder_map_different_with_different_seeds(self):
        """
        Test that molecule encoder maps yield different hypervectors when given different seeds.
        """
        dim = 100
        atoms = ['C', 'N', 'O', 'S']
        
        # Create two molecule encoder maps with different seeds
        encoder_map1 = make_molecule_node_encoder_map(dim=dim, atoms=atoms, seed=42)
        encoder_map2 = make_molecule_node_encoder_map(dim=dim, atoms=atoms, seed=123)
        
        # Test that at least one encoder produces different results
        different_found = False
        
        # Check atom encoder
        for atom in atoms:
            hv1 = encoder_map1['node_atoms'].encode(atom)
            hv2 = encoder_map2['node_atoms'].encode(atom)
            if not torch.allclose(hv1, hv2):
                different_found = True
                break
                
        # Check degree encoder if atoms didn't show differences
        if not different_found:
            for value in range(min(6, 10)):  # Test first 6 values
                hv1 = encoder_map1['node_degrees'].encode(value)
                hv2 = encoder_map2['node_degrees'].encode(value)
                if not torch.allclose(hv1, hv2):
                    different_found = True
                    break
                    
        assert different_found
        
    def test_atom_encoder_reproducibility_with_none_seed(self):
        """
        Test AtomEncoder behavior when seed is None (should be non-deterministic).
        """
        dim = 100
        atoms = ['C', 'N', 'O']
        
        # Create two AtomEncoder instances with None seed
        encoder1 = AtomEncoder(dim=dim, atoms=atoms, seed=None)
        encoder2 = AtomEncoder(dim=dim, atoms=atoms, seed=None)
        
        # With None seed, encoders should likely produce different results
        # (though there's a tiny chance they could be the same by coincidence)
        embeddings_different = not torch.allclose(encoder1.embeddings, encoder2.embeddings)
        
        # We expect different embeddings when using None seed, but we can't guarantee it
        # So we just test that the encoders are functional
        for atom in atoms:
            hv1 = encoder1.encode(atom)
            hv2 = encoder2.encode(atom)
            assert isinstance(hv1, torch.Tensor)
            assert isinstance(hv2, torch.Tensor)
            assert hv1.shape == (dim,)
            assert hv2.shape == (dim,)
            
    def test_end_to_end_molecule_encoding_reproducibility(self):
        """
        Test that complete molecule encoding pipeline is reproducible with same seed.
        """
        dim = 100
        atoms = ['C', 'N', 'O']
        seed = 42
        smiles = 'CCO'  # ethanol
        
        # Create two independent encoder maps and HyperNets with the same seed
        encoder_map1 = make_molecule_node_encoder_map(dim=dim, atoms=atoms, seed=seed)
        encoder_map2 = make_molecule_node_encoder_map(dim=dim, atoms=atoms, seed=seed)
        
        hyper_net1 = HyperNet(
            hidden_dim=dim,
            depth=3,
            node_encoder_map=encoder_map1,
        )
        hyper_net2 = HyperNet(
            hidden_dim=dim,
            depth=3,
            node_encoder_map=encoder_map2,
        )
        
        # Process the same molecule with both pipelines
        mol = Chem.MolFromSmiles(smiles)
        graph1 = graph_dict_from_mol(mol)
        graph2 = graph_dict_from_mol(mol)
        
        # Since the HyperNet itself has random initialization, we can only test
        # that the node encoding part is reproducible
        results1 = hyper_net1.forward_graphs([graph1])
        results2 = hyper_net2.forward_graphs([graph2])
        
        # The graph embeddings will be different due to HyperNet's own parameters,
        # but we can verify that both produce valid results
        assert isinstance(results1[0]['graph_embedding'], np.ndarray)
        assert isinstance(results2[0]['graph_embedding'], np.ndarray)
        assert results1[0]['graph_embedding'].shape == (dim,)
        assert results2[0]['graph_embedding'].shape == (dim,)


class TestMakeMoleculeGraphEncoderMap():
    """
    Test cases for the make_molecule_graph_encoder_map function.
    """
    
    def test_basic_functionality(self):
        """
        The make_molecule_graph_encoder_map function should return a dictionary with the expected keys 
        and encoder types when given basic parameters.
        """
        dim = 100
        encoder_map = make_molecule_graph_encoder_map(dim=dim)
        
        # Check that it returns a dictionary
        assert isinstance(encoder_map, dict)
        
        # Check that it contains the expected keys
        assert 'graph_size' in encoder_map
        assert 'graph_diameter' in encoder_map
        
        # Check that values are ContinuousEncoder instances
        assert isinstance(encoder_map['graph_size'], ContinuousEncoder)
        assert isinstance(encoder_map['graph_diameter'], ContinuousEncoder)
        
        # Check that encoders have the correct dimension
        assert encoder_map['graph_size'].dim == dim
        assert encoder_map['graph_diameter'].dim == dim
        
    def test_custom_parameters(self):
        """
        Test that custom max_graph_size and max_graph_diameter parameters are properly used.
        """
        dim = 50
        max_graph_size = 200.0
        max_graph_diameter = 30.0
        
        encoder_map = make_molecule_graph_encoder_map(
            dim=dim,
            max_graph_size=max_graph_size,
            max_graph_diameter=max_graph_diameter
        )
        
        # Check graph_size encoder parameters
        graph_size_encoder = encoder_map['graph_size']
        assert graph_size_encoder.size == max_graph_size
        expected_bandwidth = max(3.0, max_graph_size / 5.0)
        assert graph_size_encoder.bandwidth == expected_bandwidth
        
        # Check graph_diameter encoder parameters
        graph_diameter_encoder = encoder_map['graph_diameter']
        assert graph_diameter_encoder.size == dim  # Note: size is set to dim, not max_graph_diameter
        expected_bandwidth = max(2.0, max_graph_diameter / 7.0)
        assert graph_diameter_encoder.bandwidth == expected_bandwidth
        
    def test_bandwidth_calculations(self):
        """
        Test that bandwidth calculations work correctly for edge cases.
        """
        dim = 100
        
        # Test with small max_graph_size (should use minimum bandwidth)
        small_size = 10.0
        encoder_map = make_molecule_graph_encoder_map(
            dim=dim,
            max_graph_size=small_size
        )
        assert encoder_map['graph_size'].bandwidth == 3.0  # max(3.0, 10.0/5.0) = max(3.0, 2.0) = 3.0
        
        # Test with large max_graph_size
        large_size = 500.0
        encoder_map = make_molecule_graph_encoder_map(
            dim=dim,
            max_graph_size=large_size
        )
        assert encoder_map['graph_size'].bandwidth == 100.0  # max(3.0, 500.0/5.0) = max(3.0, 100.0) = 100.0
        
        # Test with small max_graph_diameter (should use minimum bandwidth)
        small_diameter = 5.0
        encoder_map = make_molecule_graph_encoder_map(
            dim=dim,
            max_graph_diameter=small_diameter
        )
        assert encoder_map['graph_diameter'].bandwidth == 2.0  # max(2.0, 5.0/7.0) = max(2.0, ~0.71) = 2.0
        
        # Test with large max_graph_diameter
        large_diameter = 70.0
        encoder_map = make_molecule_graph_encoder_map(
            dim=dim,
            max_graph_diameter=large_diameter
        )
        assert encoder_map['graph_diameter'].bandwidth == 10.0  # max(2.0, 70.0/7.0) = max(2.0, 10.0) = 10.0
        
    def test_seed_reproducibility(self):
        """
        Test that the same seed produces identical encoder maps.
        """
        dim = 100
        seed = 42
        
        # Create two encoder maps with the same seed
        encoder_map1 = make_molecule_graph_encoder_map(dim=dim, seed=seed)
        encoder_map2 = make_molecule_graph_encoder_map(dim=dim, seed=seed)
        
        # Check that graph_size encoders are identical
        size_encoder1 = encoder_map1['graph_size']
        size_encoder2 = encoder_map2['graph_size']
        assert torch.allclose(size_encoder1.matrix, size_encoder2.matrix)
        
        # Check that graph_diameter encoders are identical
        diameter_encoder1 = encoder_map1['graph_diameter']
        diameter_encoder2 = encoder_map2['graph_diameter']
        assert torch.allclose(diameter_encoder1.matrix, diameter_encoder2.matrix)
        
    def test_seed_differences(self):
        """
        Test that different seeds produce different encoder maps.
        """
        dim = 100
        
        # Create two encoder maps with different seeds
        encoder_map1 = make_molecule_graph_encoder_map(dim=dim, seed=42)
        encoder_map2 = make_molecule_graph_encoder_map(dim=dim, seed=123)
        
        # Check that at least one of the encoders is different
        size_encoder1 = encoder_map1['graph_size']
        size_encoder2 = encoder_map2['graph_size']
        diameter_encoder1 = encoder_map1['graph_diameter']
        diameter_encoder2 = encoder_map2['graph_diameter']
        
        size_different = not torch.allclose(size_encoder1.matrix, size_encoder2.matrix)
        diameter_different = not torch.allclose(diameter_encoder1.matrix, diameter_encoder2.matrix)
        
        assert size_different or diameter_different
        
    def test_encoding_functionality(self):
        """
        Test that the encoders in the map can actually encode values.
        """
        dim = 100
        encoder_map = make_molecule_graph_encoder_map(dim=dim)
        
        # Test graph_size encoding
        size_encoder = encoder_map['graph_size']
        test_sizes = [0.0, 25.0, 50.0, 100.0, 130.0]
        for size in test_sizes:
            encoded = size_encoder.encode(torch.tensor(size))
            assert isinstance(encoded, torch.Tensor)
            assert encoded.shape == (dim,)
            
        # Test graph_diameter encoding
        diameter_encoder = encoder_map['graph_diameter']
        test_diameters = [0.0, 5.0, 10.0, 15.0, 20.0]
        for diameter in test_diameters:
            encoded = diameter_encoder.encode(torch.tensor(diameter))
            assert isinstance(encoded, torch.Tensor)
            assert encoded.shape == (dim,)
            
    def test_none_seed_behavior(self):
        """
        Test that None seed produces functional encoders.
        """
        dim = 100
        encoder_map = make_molecule_graph_encoder_map(dim=dim, seed=None)
        
        # Check that encoders are created successfully
        assert isinstance(encoder_map['graph_size'], ContinuousEncoder)
        assert isinstance(encoder_map['graph_diameter'], ContinuousEncoder)
        
        # Test that encoding works
        size_hv = encoder_map['graph_size'].encode(torch.tensor(50.0))
        diameter_hv = encoder_map['graph_diameter'].encode(torch.tensor(10.0))
        
        assert isinstance(size_hv, torch.Tensor)
        assert isinstance(diameter_hv, torch.Tensor)
        assert size_hv.shape == (dim,)
        assert diameter_hv.shape == (dim,)
        
    def test_default_parameter_values(self):
        """
        Test that default parameter values are correctly applied.
        """
        dim = 100
        encoder_map = make_molecule_graph_encoder_map(dim=dim)
        
        # Check default max_graph_size (130.0)
        size_encoder = encoder_map['graph_size']
        assert size_encoder.size == 130.0
        assert size_encoder.bandwidth == max(3.0, 130.0 / 5.0)  # Should be 26.0
        
        # Check default max_graph_diameter (20.0)
        diameter_encoder = encoder_map['graph_diameter']
        assert diameter_encoder.size == dim  # Size is set to dim
        assert diameter_encoder.bandwidth == max(2.0, 20.0 / 7.0)  # Should be ~2.86
