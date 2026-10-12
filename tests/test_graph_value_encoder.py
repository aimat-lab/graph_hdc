"""
Tests for a further continuous global attribute of HDF (``make_value_graph_encoder``, used for Crippen's logP in
ex_26): the bandwidth rule, and that HyperNet sums the attribute's hypervector with those of the graph size and
diameter, such that the embedding depends on the attribute value and nothing changes without it.
"""
import numpy as np
import pytest
import torch
from rdkit import Chem
from rdkit.Chem import Crippen

from graph_hdc.models import HyperNet
from graph_hdc.special.molecules import (
    graph_dict_from_mol,
    make_molecule_graph_encoder_map_cont,
    make_molecule_node_encoder_map_cont,
    make_value_graph_encoder,
)

DIM = 1024
SMILES = ['CCO', 'c1ccccc1O', 'CC(=O)Nc1ccc(O)cc1', 'CCCCCCCCCC(=O)O']


def test_bandwidth_is_a_seventh_of_the_range_with_a_floor():
    assert make_value_graph_encoder([-3.5, 0.0, 10.5], dim=DIM, seed=50, quantile=0.0).bandwidth == pytest.approx(2.0)
    assert make_value_graph_encoder([0.2, 1.1, 0.9], dim=DIM, seed=50).bandwidth == pytest.approx(1.0)
    assert make_value_graph_encoder([0.2, 1.1], dim=DIM, seed=50, min_bandwidth=0.5,
                                    quantile=0.0).bandwidth == pytest.approx(0.5)


def test_bandwidth_ignores_a_few_extreme_values():
    # the default range runs from the 1st to the 99th percentile
    values = list(np.linspace(0.0, 14.0, 1000)) + [-200.0, 300.0]
    assert make_value_graph_encoder(values, dim=DIM, seed=50).bandwidth == pytest.approx(2.0, rel=0.05)
    assert make_value_graph_encoder(values, dim=DIM, seed=50, quantile=0.0).bandwidth == pytest.approx(500.0 / 7)


def test_similarity_decreases_within_one_bandwidth():
    # unit-modulus fractional power encoding: sinc-shaped similarity that reaches about zero at one bandwidth
    encoder = make_value_graph_encoder([0.0, 14.0], dim=4096, seed=50, quantile=0.0)  # bandwidth 2
    codes = encoder.encode_batch(torch.tensor([1.0, 1.25, 1.5, 2.0, 3.0, 9.0], dtype=torch.float64)).numpy()
    codes /= np.linalg.norm(codes, axis=1, keepdims=True)
    sims = codes[1:] @ codes[0]
    assert sims[0] > sims[1] > sims[2] > 0.5
    assert np.all(np.abs(sims[3:]) < 0.1)


def _hyper_net(with_value: bool, seed: int = 0) -> HyperNet:
    graph_map = make_molecule_graph_encoder_map_cont(dim=DIM, seed=seed, max_graph_size=20, max_graph_diameter=10)
    if with_value:
        graph_map['graph_clogp'] = make_value_graph_encoder([-2.0, 5.0], dim=DIM, seed=seed + 50)
    return HyperNet(hidden_dim=DIM, depth=2, node_encoder_map=make_molecule_node_encoder_map_cont(dim=DIM, seed=seed),
                    graph_encoder_map=graph_map, normalize_all=True, seed=seed)


def _encode(hyper_net: HyperNet, smiles: list, clogp: list = None) -> np.ndarray:
    graphs = []
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        graph = graph_dict_from_mol(mol, hydrogens='total')
        graph.pop('graph_labels', None)
        if clogp is not None:
            graph['graph_clogp'] = clogp[i]
        graphs.append(graph)
    results = hyper_net.forward_graphs(graphs, batch_size=100)
    return np.stack([np.asarray(r['graph_embedding'], dtype=np.float64) for r in results])


def test_embedding_depends_on_the_attribute_value():
    hyper_net = _hyper_net(with_value=True)
    a = _encode(hyper_net, ['CCO', 'CCO'], clogp=[0.0, 3.0])
    assert not np.allclose(a[0], a[1])
    b = _encode(hyper_net, ['CCO', 'CCO'], clogp=[0.5, 0.5])
    assert np.allclose(b[0], b[1])


def test_batched_values_are_assigned_to_their_molecules():
    hyper_net = _hyper_net(with_value=True)
    clogp = [Crippen.MolLogP(Chem.MolFromSmiles(s)) for s in SMILES]
    batch = _encode(hyper_net, SMILES, clogp)
    single = np.concatenate([_encode(hyper_net, [s], [v]) for s, v in zip(SMILES, clogp)])
    assert np.allclose(batch, single, atol=1e-10)


def test_without_the_attribute_the_standard_embedding_is_unchanged():
    # an extra graph dict entry is ignored by a HyperNet whose map does not encode it
    plain = _hyper_net(with_value=False)
    assert np.allclose(_encode(plain, SMILES), _encode(plain, SMILES, clogp=[1.0] * len(SMILES)))
