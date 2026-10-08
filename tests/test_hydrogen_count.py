"""
Hydrogen counting in graph_dict_from_mol: "implicit" (the original behavior, RDKit GetNumImplicitHs) is 0 for every
atom written in brackets, "total" (RDKit GetTotalNumHs) counts all bonded hydrogens.
"""
import numpy as np
import pytest
from rdkit import Chem

from graph_hdc.models import HyperNet
from graph_hdc.special.molecules import graph_dict_from_mol, make_molecule_node_encoder_map_cont


@pytest.mark.parametrize('smiles, index, implicit, total', [
    ('c1cc[nH]c1', 3, 0, 1),        # pyrrole N-H, a bracket atom
    ('C[NH+](C)C', 1, 0, 1),        # protonated tertiary amine
    ('C[C@@H](N)O', 1, 0, 1),       # stereocentre written in brackets
    ('CC(N)O', 1, 1, 1),            # atoms without brackets are unaffected
])
def test_hydrogen_count_modes(smiles, index, implicit, total):
    mol = Chem.MolFromSmiles(smiles)
    assert graph_dict_from_mol(mol)['node_valences'][index] == implicit
    assert graph_dict_from_mol(mol, hydrogens='implicit')['node_valences'][index] == implicit
    assert graph_dict_from_mol(mol, hydrogens='total')['node_valences'][index] == total


def test_unknown_hydrogens_mode_raises():
    with pytest.raises(ValueError):
        graph_dict_from_mol(Chem.MolFromSmiles('CC'), hydrogens='all')


def encode(smiles_list: list, hydrogens: str) -> np.ndarray:
    net = HyperNet(hidden_dim=256, depth=2, node_encoder_map=make_molecule_node_encoder_map_cont(dim=256, seed=0),
                   seed=0, normalize_all=True, bidirectional=True)
    graphs = []
    for smiles in smiles_list:
        graph = graph_dict_from_mol(Chem.MolFromSmiles(smiles), hydrogens=hydrogens)
        graph.pop('graph_labels', None)
        graphs.append(graph)
    x = np.stack([np.asarray(r['graph_embedding'], dtype=np.float64) for r in net.forward_graphs(graphs)])
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_tautomers_are_separated_only_with_total_hydrogens():
    # a QM9 pair that collided exactly with implicit hydrogen counts
    pair = ['CCc1[nH]nnc1C=O', 'CCc1n[nH]nc1C=O']
    implicit = encode(pair, 'implicit')
    total = encode(pair, 'total')
    assert implicit[0] @ implicit[1] > 1 - 1e-12
    assert total[0] @ total[1] < 1 - 1e-6


def test_stereo_notation_does_not_change_hdf_with_total_hydrogens():
    pair = ['C[C@@H](N)O', 'CC(N)O']
    assert encode(pair, 'implicit')[0] @ encode(pair, 'implicit')[1] < 1 - 1e-6
    total = encode(pair, 'total')
    assert total[0] @ total[1] > 1 - 1e-12
