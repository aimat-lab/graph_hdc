"""
Optional stereo attribute: graph_dict_from_mol(stereo=True) adds the CIP descriptor of every atom as "node_stereo",
and the encoder maps created with stereo=True encode it. Without stereo, stereoisomers receive identical fingerprints;
with stereo, they are separated, while molecules without specified stereo elements keep exactly the same fingerprint.
"""
import numpy as np
import pytest
from rdkit import Chem

from graph_hdc.models import HyperNet
from graph_hdc.special.molecules import (
    STEREO_LABELS,
    StereoEncoder,
    graph_dict_from_mol,
    make_molecule_node_encoder_map,
    make_molecule_node_encoder_map_cont,
    make_molecule_graph_encoder_map_cont,
    stereo_labels,
)

PAIRS = [
    ('N[C@@H](C)C(=O)O', 'N[C@H](C)C(=O)O'),                      # enantiomers
    ('C/C=C/C', 'C/C=C\\C'),                                        # E/Z isomers
    ('C[C@H]1CC[C@@H](C)CC1', 'C[C@H]1CC[C@H](C)CC1'),              # cis/trans (pseudoasymmetric r/s centres)
    ('C[C@@H](O)[C@H](N)C(=O)O', 'C[C@H](O)[C@H](N)C(=O)O'),        # diastereomers (threonine, allo-threonine)
]
NO_STEREO = ['CCO', 'c1ccccc1O', 'CC(C)Cc1ccc(cc1)C(C)C(=O)O', 'C1CCC2CCCCC2C1', 'C[NH+](C)C', 'c1cc[nH]c1']


def labels(smiles: str) -> list:
    return [STEREO_LABELS[int(i)] for i in stereo_labels(Chem.MolFromSmiles(smiles))]


@pytest.mark.parametrize('smiles, expected', [
    ('N[C@@H](C)C(=O)O', ['none', 'S', 'none', 'none', 'none', 'none']),
    ('N[C@H](C)C(=O)O', ['none', 'R', 'none', 'none', 'none', 'none']),
    ('C/C=C/C', ['none', 'E', 'E', 'none']),
    ('C/C=C\\C', ['none', 'Z', 'Z', 'none']),
    ('C[C@H]1CC[C@@H](C)CC1', ['none', 's', 'none', 'none', 's', 'none', 'none', 'none']),
    ('NC(C)C(=O)O', ['none'] * 6),                                  # unspecified stereocentre
    ('CC=CC', ['none'] * 4),                                        # unspecified double bond
])
def test_stereo_labels(smiles, expected):
    assert labels(smiles) == expected


def test_stereo_labels_do_not_depend_on_atom_order():
    # RDKit's CIP labeler assigns E or Z to this double bond depending on the atom order (a ZINC250k molecule)
    mol = Chem.MolFromSmiles('Cn1cc(/C=C(/C#N)c2nc3cccnc3[n-]2)c(-c2ccncc2)n1')
    reference = sorted(STEREO_LABELS[int(i)] for i in stereo_labels(mol))
    assert reference.count('E') + reference.count('Z') == 2
    rng = np.random.default_rng(0)
    for _ in range(30):
        order = [int(i) for i in rng.permutation(mol.GetNumAtoms())]
        renumbered = Chem.RenumberAtoms(mol, order)
        labels = stereo_labels(renumbered)
        # the descriptor of every atom follows the atom through the renumbering
        assert [labels[order.index(i)] for i in range(mol.GetNumAtoms())] == list(stereo_labels(mol))
        assert sorted(STEREO_LABELS[int(i)] for i in labels) == reference


def test_stereo_labels_do_not_modify_the_molecule():
    mol = Chem.MolFromSmiles('N[C@@H](C)C(=O)O')
    before = {a.GetIdx(): a.GetPropsAsDict(includePrivate=True).get('_CIPCode') for a in mol.GetAtoms()}
    stereo_labels(mol)
    after = {a.GetIdx(): a.GetPropsAsDict(includePrivate=True).get('_CIPCode') for a in mol.GetAtoms()}
    assert before == after


def test_graph_dict_has_stereo_only_on_request():
    mol = Chem.MolFromSmiles('N[C@@H](C)C(=O)O')
    assert 'node_stereo' not in graph_dict_from_mol(mol)
    assert list(graph_dict_from_mol(mol, stereo=True)['node_stereo']) == [0, 2, 0, 0, 0, 0]


def test_stereo_encoder_none_is_binding_identity():
    encoder = StereoEncoder(dim=64, seed=3)
    identity = np.zeros(64)
    identity[0] = 1.0
    assert np.array_equal(encoder.encode(0).numpy(), identity)
    assert all(abs(float(encoder.encode(i).norm()) - 1) < 0.5 for i in range(1, len(STEREO_LABELS)))
    assert encoder.decode(encoder.encode(4)) == 4


def test_stereo_encoder_unit_modulus():
    import torch
    unit = StereoEncoder(dim=64, seed=3, unit_modulus=True)
    magnitudes = torch.fft.fft(unit.embeddings, dim=-1).abs()
    assert torch.allclose(magnitudes, torch.ones_like(magnitudes), atol=1e-10)
    identity = np.zeros(64)
    identity[0] = 1.0
    assert np.array_equal(unit.encode(0).numpy(), identity)
    # the same random draw: the phases agree with the Gaussian codebook
    gaussian = StereoEncoder(dim=64, seed=3, unit_modulus=False)
    phases = lambda e: torch.angle(torch.fft.fft(e.embeddings[1:], dim=-1))
    assert torch.allclose(torch.cos(phases(unit)), torch.cos(phases(gaussian)), atol=1e-8)


def test_stereo_encoder_does_not_touch_global_rng():
    import torch
    torch.manual_seed(123)
    expected = torch.rand(3)
    torch.manual_seed(123)
    StereoEncoder(dim=64, seed=3)
    assert torch.equal(torch.rand(3), expected)


def test_categorical_map_gets_stereo_encoder_last():
    assert list(make_molecule_node_encoder_map(dim=64, seed=0)) == ['node_atoms', 'node_degrees', 'node_valences']
    assert list(make_molecule_node_encoder_map(dim=64, seed=0, stereo=True))[-1] == 'node_stereo'
    assert list(make_molecule_node_encoder_map_cont(dim=64, seed=0, stereo=True))[-1] == 'node_stereo'


def encode(smiles_list: list, stereo: bool, dim: int = 512) -> np.ndarray:
    net = HyperNet(hidden_dim=dim, depth=2, device='cpu',
                   node_encoder_map=make_molecule_node_encoder_map_cont(dim=dim, seed=0, stereo=stereo),
                   graph_encoder_map=make_molecule_graph_encoder_map_cont(dim=dim, seed=0, max_graph_size=60.0,
                                                                          max_graph_diameter=20.0),
                   seed=0, normalize_all=True, bidirectional=True)
    graphs = []
    for smiles in smiles_list:
        graph = graph_dict_from_mol(Chem.MolFromSmiles(smiles), hydrogens='total', stereo=stereo)
        graph.pop('graph_labels', None)
        graphs.append(graph)
    return np.stack([np.asarray(r['graph_embedding'], dtype=np.float64) for r in net.forward_graphs(graphs)])


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))


@pytest.mark.parametrize('pair', PAIRS)
def test_stereoisomers_are_separated_only_with_stereo(pair):
    plain = encode(list(pair), stereo=False)
    assert np.array_equal(plain[0], plain[1])
    stereo = encode(list(pair), stereo=True)
    assert cosine(stereo[0], stereo[1]) < 0.99


def test_molecules_without_stereo_keep_their_fingerprint():
    assert np.array_equal(encode(NO_STEREO, stereo=False), encode(NO_STEREO, stereo=True))


def test_stereo_fingerprint_does_not_depend_on_atom_order():
    mol = Chem.MolFromSmiles('C[C@@H](O)[C@H](N)C(=O)O')
    rng = np.random.default_rng(0)
    variants = [Chem.MolToSmiles(Chem.RenumberAtoms(mol, [int(i) for i in rng.permutation(mol.GetNumAtoms())]),
                                 canonical=False) for _ in range(5)]
    x = encode(variants, stereo=True)
    assert np.abs(x - x[0]).max() < 1e-12
