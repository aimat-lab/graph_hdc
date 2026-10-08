"""
Regression tests for the edge direction of the HyperNet message passing.

The graph dicts created by ``graph_dict_from_mol`` store every bond only once. Until 2026-10-08, HyperNet
passed messages only along that one direction by default, so every atom only aggregated the neighbors with a
higher atom index and the fingerprint of a molecule depended on the atom order of its SMILES. HyperNet is now
bidirectional by default; these tests make sure it stays that way.
"""
import os
import tempfile

import jsonpickle
import numpy as np
import pytest
from rdkit import Chem

from graph_hdc.models import HyperNet
from graph_hdc.special.molecules import graph_dict_from_mol, make_molecule_node_encoder_map_cont

SMILES = ['CCO', 'c1ccccc1O', 'CC(=O)Nc1ccc(O)cc1', 'C1CCC2CCCCC2C1', 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C']


def _encode(hyper_net: HyperNet, smiles: list) -> np.ndarray:
    graphs = []
    for smi in smiles:
        graph = graph_dict_from_mol(Chem.MolFromSmiles(smi))
        graph.pop('graph_labels', None)
        graphs.append(graph)
    results = hyper_net.forward_graphs(graphs, batch_size=100)
    embeddings = np.stack([np.asarray(r['graph_embedding'], dtype=np.float64) for r in results])
    return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)


def _random_orderings(smiles: str, num: int = 4) -> list:
    mol = Chem.MolFromSmiles(smiles)
    return [Chem.MolToSmiles(mol)] + [Chem.MolToSmiles(mol, doRandom=True) for _ in range(num)]


@pytest.fixture(scope='module')
def node_encoder_map():
    return make_molecule_node_encoder_map_cont(dim=1024, seed=0)


def test_bidirectional_is_default(node_encoder_map):
    assert HyperNet(hidden_dim=1024, node_encoder_map=node_encoder_map).bidirectional is True


@pytest.mark.parametrize('smiles', SMILES)
def test_default_embedding_invariant_to_atom_order(node_encoder_map, smiles):
    hyper_net = HyperNet(hidden_dim=1024, depth=2, node_encoder_map=node_encoder_map, normalize_all=True, seed=0)
    embeddings = _encode(hyper_net, _random_orderings(smiles))
    assert np.allclose(embeddings @ embeddings[0], 1.0, atol=1e-9)


def test_one_directional_embedding_depends_on_atom_order(node_encoder_map):
    # documents why the default was changed: with one-directional messages the atom order matters
    hyper_net = HyperNet(hidden_dim=1024, depth=2, node_encoder_map=node_encoder_map, normalize_all=True,
                         seed=0, bidirectional=False)
    sims = []
    for smiles in SMILES[1:]:
        embeddings = _encode(hyper_net, _random_orderings(smiles))
        sims.append((embeddings @ embeddings[0]).min())
    assert min(sims) < 0.999


def test_load_without_bidirectional_attribute_is_one_directional(node_encoder_map):
    # files saved before the attribute existed were one-directional and must load as such
    hyper_net = HyperNet(hidden_dim=1024, depth=2, node_encoder_map=node_encoder_map)
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, 'hyper_net.pth')
        hyper_net.save_to_path(path)
        assert HyperNet.load(path).bidirectional is True

        with open(path) as file:
            data = jsonpickle.loads(file.read())
        del data['attributes']['bidirectional']
        with open(path, 'w') as file:
            file.write(jsonpickle.dumps(data))
        assert HyperNet.load(path).bidirectional is False
