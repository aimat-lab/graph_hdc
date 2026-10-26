"""
Tests for the training-free decoding of the atom composition from HDF vectors (graph_hdc.special.composition).
"""
import numpy as np
import pytest
from rdkit import Chem

from graph_hdc.models import HyperNet
from graph_hdc.special.composition import (atom_composition, atom_type_codebook, composition_metrics,
                                           decode_composition, decode_size_and_diameter, integer_counts)
from graph_hdc.special.molecules import (graph_dict_from_mol, make_molecule_graph_encoder_map_cont,
                                         make_molecule_node_encoder_map_cont)

SMILES = ['CC(=O)Oc1ccccc1C(=O)O', 'c1ccncc1', 'CC(C)(C)N', 'OCC(O)CO', 'C[NH3+]', 'FC(F)(F)c1ccc(Cl)cc1']


def make_net(dim: int, seed: int = 1, max_size: int = 20, max_diameter: int = 10):
    node_map = make_molecule_node_encoder_map_cont(dim=dim, seed=seed)
    graph_map = make_molecule_graph_encoder_map_cont(dim=dim, seed=seed, max_graph_size=max_size,
                                                     max_graph_diameter=max_diameter)
    net = HyperNet(hidden_dim=dim, depth=2, device='cpu', node_encoder_map=node_map, graph_encoder_map=graph_map,
                   seed=seed, normalize_all=True, bidirectional=True)
    return net, node_map, graph_map


def graphs():
    return [graph_dict_from_mol(Chem.MolFromSmiles(s), hydrogens='total') for s in SMILES]


def test_codebook_reproduces_the_layer0_readout_of_hypernet():
    """The sum of the codebook vectors of all atoms equals HyperNet's layer-0 readout of the molecule."""
    net, node_map, _ = make_net(dim=512)
    types, codebook = atom_type_codebook(node_map)
    assert len(types) == len(node_map['node_atoms'].atoms) * 25
    assert codebook.shape == (len(types), 512)
    for graph, result in zip(graphs(), net.forward_graphs(graphs(), batch_size=len(SMILES))):
        expected = atom_composition(graph, types) @ codebook
        np.testing.assert_allclose(np.asarray(result['graph_hv_stack'])[0], expected, atol=1e-8)


def test_codebook_vectors_have_unit_norm_with_unit_modulus_codebooks():
    """Up to the real DC/Nyquist coefficients, whose fractional powers lose a little norm (about 1/D)."""
    _, node_map, _ = make_net(dim=256)
    _, codebook = atom_type_codebook(node_map)
    np.testing.assert_allclose(np.linalg.norm(codebook, axis=1), 1.0, atol=1e-2)


def test_layer0_readout_decodes_exactly():
    net, node_map, _ = make_net(dim=2048)
    types, codebook = atom_type_codebook(node_map)
    for graph, result in zip(graphs(), net.forward_graphs(graphs(), batch_size=len(SMILES))):
        decoded = decode_composition(np.asarray(result['graph_hv_stack'])[0], codebook)
        np.testing.assert_array_equal(decoded, atom_composition(graph, types))


def test_size_and_diameter_are_decoded_from_the_final_vector():
    """Including the two-atom molecule, for which a query with the size encodings alone fails."""
    net, _, graph_map = make_net(dim=2048)
    for graph, result in zip(graphs(), net.forward_graphs(graphs(), batch_size=len(SMILES))):
        size, diameter = decode_size_and_diameter(result['graph_embedding'], graph_map['graph_size'],
                                                  graph_map['graph_diameter'], max_size=20, max_diameter=10)
        assert size == len(graph['node_atoms'])
        assert diameter == graph['graph_diameter']


def test_final_vector_decoding_returns_counts_that_sum_to_the_atom_count():
    net, node_map, graph_map = make_net(dim=2048)
    types, codebook = atom_type_codebook(node_map)
    for graph, result in zip(graphs(), net.forward_graphs(graphs(), batch_size=len(SMILES))):
        n = len(graph['node_atoms'])
        decoded = decode_composition(result['graph_embedding'], codebook, num_atoms=n)
        assert decoded.min() >= 0
        assert decoded.sum() <= n


def test_integer_counts():
    np.testing.assert_array_equal(integer_counts(np.array([0.31, 0.15, 0.02, 0.0]), 3), [2, 1, 0, 0])
    np.testing.assert_array_equal(integer_counts(np.array([0.2, 0.2, 0.2]), 3), [1, 1, 1])
    np.testing.assert_array_equal(integer_counts(np.zeros(3), 3), [0, 0, 0])


def test_composition_metrics():
    true = np.array([2, 1, 0, 0])
    assert composition_metrics(true, true) == {'exact_counts': 1.0, 'exact_set': 1.0, 'atoms_correct': 1.0}
    metrics = composition_metrics(np.array([1, 1, 1, 0]), true)
    assert metrics['exact_counts'] == 0.0 and metrics['exact_set'] == 0.0
    assert metrics['atoms_correct'] == pytest.approx(2 / 3)
    metrics = composition_metrics(np.array([1, 2, 0, 0]), true)
    assert metrics['exact_set'] == 1.0 and metrics['exact_counts'] == 0.0


def test_atom_composition_rejects_types_outside_the_codebook():
    _, node_map, _ = make_net(dim=128)
    types, _ = atom_type_codebook(node_map, degrees=range(2), hydrogens=range(2))
    graph = graph_dict_from_mol(Chem.MolFromSmiles('CC(C)C'), hydrogens='total')   # central C has degree 3
    with pytest.raises(KeyError):
        atom_composition(graph, types)


def test_size_and_diameter_batch_mode_matches_single_vectors():
    net, _, graph_map = make_net(dim=1024)
    vectors = np.stack([r['graph_embedding'] for r in net.forward_graphs(graphs(), batch_size=len(SMILES))])
    sizes, diameters = decode_size_and_diameter(vectors, graph_map['graph_size'], graph_map['graph_diameter'], 20, 10)
    for vector, size, diameter in zip(vectors, sizes, diameters):
        assert decode_size_and_diameter(vector, graph_map['graph_size'], graph_map['graph_diameter'], 20, 10) \
            == (size, diameter)
