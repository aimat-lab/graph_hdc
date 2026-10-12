"""
Tests for GRAPH_DESCRIPTORS of the fingerprint experiment module (ex_25, reviewer comment R3.3): the graph size and the
graph diameter appended to a baseline fingerprint must be exactly the global descriptors that HDF encodes.
"""
import os
import sys
import importlib.util

import numpy as np
import pytest
from rdkit import Chem

from graph_hdc.special.molecules import graph_dict_from_mol

FINGERPRINTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'graph_hdc', 'experiments', 'fingerprints')

SMILES = [
    'CCO', 'c1ccccc1', 'c1ccc2ccccc2c1', 'C1CCC2CCCCC2C1', 'C1CCCC1', 'c1cc[nH]c1', '[NH3+]CC(=O)[O-]',
    'C[C@@H](O)CC', 'CC(C)(C)c1ccc(O)cc1', 'O=C(O)c1ccccc1OC(C)=O', 'FC(F)(F)C(Cl)Br', 'N#CC#N',
    'CCCCCCCCCCCCCCCC', 'O=C1c2ccccc2C(=O)c2ccccc21',
]


@pytest.fixture(scope='module')
def fp_module():
    # the experiment modules are scripts, not a package; pycomex needs the module registered in sys.modules
    spec = importlib.util.spec_from_file_location('predict_molecules__fp',
                                                  os.path.join(FINGERPRINTS, 'predict_molecules__fp.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class StubExperiment(dict):
    """The parts of a pycomex Experiment that append_graph_descriptors uses: log() and item assignment."""

    def log(self, message: str):
        pass


def test_graph_descriptors_match_hdf(fp_module):
    """The size and diameter equal those of the graph dicts from which HDF encodes its global descriptors."""
    for smiles in SMILES:
        mol = Chem.MolFromSmiles(smiles)
        graph = graph_dict_from_mol(mol, hydrogens='total')
        assert fp_module.graph_descriptors(mol) == (graph['graph_size'], graph['graph_diameter']), smiles


def test_append_graph_descriptors(fp_module):
    """Two features are appended, divided by the dataset maxima; the fingerprint itself is unchanged."""
    rng = np.random.default_rng(0)
    index_data_map = {i: {'graph_repr': smiles, 'graph_features': rng.integers(0, 2, 16).astype(float)}
                      for i, smiles in enumerate(SMILES)}
    before = {i: graph['graph_features'].copy() for i, graph in index_data_map.items()}
    e = StubExperiment()
    fp_module.append_graph_descriptors(e, index_data_map)

    sizes = {i: fp_module.graph_descriptors(Chem.MolFromSmiles(s)) for i, s in enumerate(SMILES)}
    max_size = max(size for size, _ in sizes.values())
    max_diameter = max(diameter for _, diameter in sizes.values())
    assert e['graph_descriptors/max_size'] == max_size == 16
    assert e['graph_descriptors/max_diameter'] == max_diameter == 15
    for i, graph in index_data_map.items():
        features = graph['graph_features']
        assert features.shape == (18,)
        np.testing.assert_array_equal(features[:16], before[i])
        size, diameter = sizes[i]
        assert features[16] == size / max_size and features[17] == diameter / max_diameter
        assert 0.0 < features[16] <= 1.0 and 0.0 < features[17] <= 1.0


def test_default_is_off(fp_module):
    """Without the parameter the fingerprint experiments are unchanged."""
    assert fp_module.GRAPH_DESCRIPTORS is False
