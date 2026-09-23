"""
Unit tests for the Sherlock Fingerprint baseline
(:mod:`graph_hdc.baselines.sherlock`).
"""

import os
import tempfile

import numpy as np
import pytest

from graph_hdc.baselines.sherlock import (
    SherlockFingerprint,
    enumerate_bit_infos,
    bernoulli_entropy,
)

# A small but chemically diverse corpus of SMILES, enough for the entropy ranking
# to actually discard descriptors (more distinct descriptors than the output size).
CORPUS = [
    'CCO', 'CC(=O)O', 'c1ccccc1', 'CC(=O)Oc1ccccc1C(=O)O', 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',
    'C1CCCCC1', 'CCN(CC)CC', 'CC(C)Cc1ccc(cc1)C(C)C(=O)O', 'O=C(O)c1ccccc1O', 'Clc1ccccc1',
    'CC(C)(C)c1ccc(O)cc1', 'c1ccc2ccccc2c1', 'OCC1OC(O)C(O)C(O)C1O', 'CC(N)C(=O)O',
    'c1ccncc1', 'CCOC(=O)c1ccccc1', 'NCCc1ccc(O)cc1', 'CC(=O)Nc1ccc(O)cc1',
    'COc1ccccc1', 'CCCCCCCC(=O)O', 'c1ccc(cc1)S(=O)(=O)N', 'C1=CC=C(C=C1)N',
    'O=C1CCCCC1', 'Cc1ccccc1C', 'Nc1ccccc1', 'c1ccc(cc1)C(F)(F)F', 'CSc1ccccc1',
]


def test_enumerate_bit_infos_are_collision_free_tuples():
    """Descriptors are (bit_id, atom_symbol, frag_smiles, radius) tuples."""
    bit_infos = enumerate_bit_infos('CCO', radius=3)
    assert len(bit_infos) > 0
    for bi in bit_infos:
        assert isinstance(bi, tuple) and len(bi) == 4
        bit_id, atom_symbol, frag_smiles, radius = bi
        assert isinstance(bit_id, int)
        assert isinstance(atom_symbol, str)
        assert isinstance(frag_smiles, str)
        assert 0 <= radius <= 3


def test_invalid_smiles_yields_empty_and_zero_vector():
    assert enumerate_bit_infos('not_a_smiles', radius=3) == set()
    sf = SherlockFingerprint(radius=3, size=32).fit(CORPUS)
    vec = sf.transform_smiles('not_a_smiles')
    assert vec.shape == (32,)
    assert vec.sum() == 0.0


def test_fit_and_transform_shapes_and_binary():
    sf = SherlockFingerprint(radius=3, size=64).fit(CORPUS)
    # the corpus must have produced more descriptors than we keep
    assert len(sf.counts_) > 64
    assert len(sf.bitinfo_to_index_) == 64

    vec = sf.transform_smiles('CC(=O)Oc1ccccc1C(=O)O')
    assert vec.shape == (64,)
    assert vec.dtype == np.float64
    assert set(np.unique(vec)).issubset({0.0, 1.0})
    assert vec.sum() > 0

    mat = sf.transform(['CCO', 'c1ccccc1', 'CC(=O)O'])
    assert mat.shape == (3, 64)


def test_descriptor_index_map_is_one_to_one():
    """Collision-free: every retained descriptor maps to a distinct output bit."""
    sf = SherlockFingerprint(radius=3, size=64).fit(CORPUS)
    indices = list(sf.bitinfo_to_index_.values())
    assert len(indices) == len(set(indices))
    assert sorted(indices) == list(range(len(indices)))


def test_size_matched_is_prefix_of_native_ranking():
    """A smaller size keeps exactly the top-N highest-entropy descriptors."""
    sf = SherlockFingerprint(radius=3, size=64).fit(CORPUS)
    native_order = [sf.index_to_bitinfo_[i] for i in range(len(sf.index_to_bitinfo_))]

    sf.set_output(size=32)
    matched_order = [sf.index_to_bitinfo_[i] for i in range(len(sf.index_to_bitinfo_))]
    assert matched_order == native_order[:32]
    assert sf.transform_smiles('CCO').shape == (32,)


def test_save_load_roundtrip(tmp_path):
    sf = SherlockFingerprint(radius=3, size=64).fit(CORPUS)
    path = os.path.join(tmp_path, 'sherlock.pkl')
    sf.save(path)

    loaded = SherlockFingerprint.load(path)
    smiles = ['CCO', 'CC(=O)O', 'c1ccccc1']
    assert np.array_equal(sf.transform(smiles), loaded.transform(smiles))
    # the full counts table is preserved so re-sizing still works after load
    loaded.set_output(size=16)
    assert loaded.transform_smiles('CCO').shape == (16,)


def test_lightweight_dictionary_roundtrip(tmp_path):
    """save_dictionary drops the count table but preserves transform + size-matching."""
    sf = SherlockFingerprint(radius=3, size=64).fit(CORPUS)
    path = os.path.join(tmp_path, 'sherlock_dict.pkl')
    sf.save_dictionary(path)

    light = SherlockFingerprint.load(path)
    assert light.counts_ is None                      # count table dropped
    assert light.ranked_bitinfos_ is not None

    smiles = ['CCO', 'CC(=O)O', 'c1ccccc1', 'CC(=O)Oc1ccccc1C(=O)O']
    assert np.array_equal(sf.transform(smiles), light.transform(smiles))

    # size-matching still works via prefix slicing
    light.set_output(size=16)
    assert light.transform_smiles('CCO').shape == (16,)
    # asking for a larger size than stored, or a different radius, is a clear error
    with pytest.raises(ValueError):
        light.set_output(size=1000)
    with pytest.raises(ValueError):
        light.set_output(size=16, max_radius=2)


def test_parallel_fit_matches_serial():
    serial = SherlockFingerprint(radius=3, size=64).fit(CORPUS, n_jobs=1)
    parallel = SherlockFingerprint(radius=3, size=64).fit(CORPUS, n_jobs=2)
    assert serial.counts_ == parallel.counts_


def test_bernoulli_entropy_sign_convention():
    """Most-informative (p=0.5) descriptors are the most negative -> sorted first."""
    counts = np.array([50, 1, 99])  # corpus of 100 -> p = 0.5, 0.01, 0.99
    ent = bernoulli_entropy(counts, corpus_size=100)
    # p=0.5 has the largest Shannon entropy -> most negative signed value here
    assert np.argmin(ent) == 0
    assert ent[0] < ent[1]
    assert ent[0] < ent[2]
