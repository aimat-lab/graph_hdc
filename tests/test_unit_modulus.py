"""
Regression tests for the unit-modulus codebooks of the HDF encoders.

Until 2026-10-09, the random codebook vectors of HDF had random Fourier magnitudes: the element vectors of
``AtomEncoder`` were Gaussian and the base vectors of ``ContinuousEncoder`` had a complex Gaussian spectrum that
the fractional power encoding raises to the power value / bandwidth. Because binding (circular convolution)
multiplies spectra, the embeddings then concentrated on a few Fourier components. Both encoders now draw
unit-modulus codebooks by default (``unit_modulus=True``): the same random phases, all magnitudes one. Experiment
ex_22 compared both versions; ``unit_modulus=False`` reproduces the original encoder.
"""
import os
import tempfile

import numpy as np
import pytest
import torch
from rdkit import Chem

from graph_hdc.models import HyperNet
from graph_hdc.special.molecules import (
    AtomEncoder,
    graph_dict_from_mol,
    make_molecule_graph_encoder_map_cont,
    make_molecule_node_encoder_map_cont,
)
from graph_hdc.special.spectrum import effective_components
from graph_hdc.utils import ContinuousEncoder

DIM = 1024
SMILES = ['CCO', 'c1ccccc1O', 'CC(=O)Nc1ccc(O)cc1', 'C1CCC2CCCCC2C1', 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C', 'C[NH3+]']


def _graphs() -> list:
    graphs = []
    for smiles in SMILES:
        graph = graph_dict_from_mol(Chem.MolFromSmiles(smiles), hydrogens='total')
        graph.pop('graph_labels', None)
        graphs.append(graph)
    return graphs


def _embed(node_map: dict, graph_map: dict) -> np.ndarray:
    hyper_net = HyperNet(hidden_dim=DIM, depth=2, node_encoder_map=node_map, graph_encoder_map=graph_map,
                         normalize_all=True, seed=0)
    return np.stack([np.asarray(r['graph_embedding'], dtype=np.float64)
                     for r in hyper_net.forward_graphs(_graphs(), batch_size=100)])


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.dot(a, b) / (a.norm() * b.norm()))


def test_unit_modulus_is_default():
    assert ContinuousEncoder(dim=DIM, size=10.0, bandwidth=2.0, seed=0).unit_modulus is True
    assert AtomEncoder(dim=DIM, atoms=['C', 'N', 'O'], seed=0).unit_modulus is True
    for encoder in make_molecule_node_encoder_map_cont(dim=DIM, seed=0).values():
        assert encoder.unit_modulus is True
    for encoder in make_molecule_graph_encoder_map_cont(dim=DIM, seed=0, max_graph_size=9,
                                                        max_graph_diameter=8).values():
        assert encoder.unit_modulus is True


def test_factories_pass_unit_modulus_false():
    for encoder in make_molecule_node_encoder_map_cont(dim=DIM, seed=0, unit_modulus=False).values():
        assert encoder.unit_modulus is False
    for encoder in make_molecule_graph_encoder_map_cont(dim=DIM, seed=0, max_graph_size=9, max_graph_diameter=8,
                                                        unit_modulus=False).values():
        assert encoder.unit_modulus is False


def test_continuous_encoder_unit_modulus():
    unit = ContinuousEncoder(dim=DIM, size=10.0, bandwidth=2.0, seed=3)
    gaussian = ContinuousEncoder(dim=DIM, size=10.0, bandwidth=2.0, seed=3, unit_modulus=False)
    assert torch.allclose(unit.matrix.abs(), torch.ones(DIM, dtype=torch.float64))
    # same random draw: the phases are those of the original spectrum
    assert torch.allclose(torch.angle(unit.matrix), torch.angle(gaussian.matrix))
    assert gaussian.matrix.abs().std() > 0.2
    codes = unit.encode_batch(torch.arange(0, 7, dtype=torch.float64))
    # every value has (nearly) the same norm; only the real DC and Nyquist terms deviate
    assert torch.allclose(codes.norm(dim=1), torch.ones(7, dtype=torch.float64), atol=1e-2)
    # the similarity depends only on the difference of the values
    assert abs(_cosine(codes[1], codes[2]) - _cosine(codes[4], codes[5])) < 2e-2
    assert abs(_cosine(codes[1], codes[3]) - _cosine(codes[3], codes[5])) < 2e-2


def test_atom_encoder_unit_modulus():
    unit = AtomEncoder(dim=DIM, atoms=['C', 'N', 'O', 'S', 'Cl'], seed=11)
    gaussian = AtomEncoder(dim=DIM, atoms=['C', 'N', 'O', 'S', 'Cl'], seed=11, unit_modulus=False)
    assert unit.embeddings.dtype == torch.float64
    spectrum = torch.fft.fft(unit.embeddings, dim=-1)
    assert torch.allclose(spectrum.abs(), torch.ones_like(spectrum.abs()), atol=1e-9)
    # still one independent random vector per element (incl. the unknown element), nearly orthogonal
    assert unit.embeddings.shape == gaussian.embeddings.shape == (6, DIM)
    assert torch.allclose(unit.embeddings.norm(dim=1), torch.ones(6, dtype=torch.float64))
    similarities = unit.embeddings @ unit.embeddings.T
    off_diagonal = similarities[~torch.eye(6, dtype=torch.bool)]
    assert off_diagonal.abs().max() < 6.0 / np.sqrt(DIM)
    # same random draw as the original Gaussian vectors (only the Fourier magnitudes differ)
    gaussian_spectrum = torch.fft.fft(gaussian.embeddings, dim=-1)
    assert torch.allclose(torch.angle(spectrum)[:, 1:DIM // 2], torch.angle(gaussian_spectrum)[:, 1:DIM // 2])
    assert unit.encode('N') is not None and torch.equal(unit.encode('N'), unit.embeddings[1])


def test_unit_modulus_equals_post_hoc_normalization():
    """The ex_22 runs normalized the spectra of encoders built the original way; the default must equal that."""
    default = make_molecule_node_encoder_map_cont(dim=DIM, seed=5)
    original = make_molecule_node_encoder_map_cont(dim=DIM, seed=5, unit_modulus=False)
    spectrum = torch.fft.fft(original['node_atoms'].embeddings, dim=-1)
    assert torch.allclose(default['node_atoms'].embeddings, torch.fft.ifft(spectrum / spectrum.abs(), dim=-1).real)
    for name in ('node_degrees', 'node_valences'):
        assert torch.allclose(default[name].matrix, original[name].matrix / original[name].matrix.abs())


def test_unit_modulus_spreads_the_embedding_spectrum():
    unit = _embed(make_molecule_node_encoder_map_cont(dim=DIM, seed=0),
                  make_molecule_graph_encoder_map_cont(dim=DIM, seed=0, max_graph_size=12, max_graph_diameter=8))
    gaussian = _embed(make_molecule_node_encoder_map_cont(dim=DIM, seed=0, unit_modulus=False),
                      make_molecule_graph_encoder_map_cont(dim=DIM, seed=0, max_graph_size=12, max_graph_diameter=8,
                                                           unit_modulus=False))
    assert np.isfinite(unit).all()
    assert effective_components(unit) > 3 * effective_components(gaussian)


def test_saved_original_model_keeps_its_codebooks():
    """Saved models store their codebooks: a model saved with the original encoder must not load as unit-modulus."""
    node_map = make_molecule_node_encoder_map_cont(dim=DIM, seed=2, unit_modulus=False)
    graph_map = make_molecule_graph_encoder_map_cont(dim=DIM, seed=2, max_graph_size=12, max_graph_diameter=8,
                                                     unit_modulus=False)
    hyper_net = HyperNet(hidden_dim=DIM, depth=2, node_encoder_map=node_map, graph_encoder_map=graph_map,
                         normalize_all=True, seed=2)
    with tempfile.TemporaryDirectory() as path:
        model_path = os.path.join(path, 'hyper_net.pth')
        hyper_net.save_to_path(model_path)
        loaded = HyperNet.load(model_path)
    assert torch.equal(loaded.node_encoder_map['node_atoms'].embeddings, node_map['node_atoms'].embeddings)
    for name in ('node_degrees', 'node_valences'):
        assert torch.equal(loaded.node_encoder_map[name].matrix, node_map[name].matrix)
    for name in ('graph_size', 'graph_diameter'):
        assert torch.equal(loaded.graph_encoder_map[name].matrix, graph_map[name].matrix)
    assert loaded.node_encoder_map['node_degrees'].matrix.abs().std() > 0.2
