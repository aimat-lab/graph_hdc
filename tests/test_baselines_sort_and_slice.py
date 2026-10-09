"""
Unit tests for the Sort & Slice ECFP baseline (:mod:`graph_hdc.baselines.sort_and_slice`). The central check is that
the fingerprints equal those of the authors' reference implementation (verbatim copy in tests/reference/) for
training molecules, unseen molecules, both binary and count vectors and several ECFP settings.
"""
import numpy as np
import pytest
from rdkit import Chem

from graph_hdc.baselines.sort_and_slice import SortAndSliceFingerprint, fit_transform_split
from .reference.sort_and_slice_ecfp_featuriser import create_sort_and_slice_ecfp_featuriser

# Training molecules: drug-like compounds, charged and stereo atoms, aromatic heterocycles, small molecules.
TRAIN = [
    'CCO', 'CC(=O)O', 'c1ccccc1', 'CC(=O)Oc1ccccc1C(=O)O', 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C', 'C1CCCCC1',
    'CCN(CC)CC', 'CC(C)Cc1ccc(cc1)C(C)C(=O)O', 'O=C(O)c1ccccc1O', 'Clc1ccccc1', 'CC(C)(C)c1ccc(O)cc1',
    'c1ccc2ccccc2c1', 'OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O', 'C[C@@H](N)C(=O)O', 'c1ccncc1',
    'CCOC(=O)c1ccccc1', 'NCCc1ccc(O)cc1', 'CC(=O)Nc1ccc(O)cc1', 'COc1ccccc1', 'CCCCCCCC(=O)O',
    'NS(=O)(=O)c1ccccc1', 'C[NH3+]', 'O=C([O-])c1ccc[nH]1', 'c1cc[nH]c1', 'C/C=C/C(=O)O', 'FC(F)(F)c1ccccc1',
    'CSc1ccccc1', 'O=C1CCCCC1', 'Cc1ccccc1C', 'C', 'N#CC(C)(C)O', 'OC1CC2CCC1C2', 'c1ccc2c(c1)ccc1ccccc12',
]
# Molecules that are not part of the training set, some with substructures that do not occur in it.
TEST = ['CCCl', 'Brc1ccccc1', 'c1ccc(cc1)P(c1ccccc1)c1ccccc1', 'CC(C)CO', 'O=[N+]([O-])c1ccccc1', 'C1CC1',
        '[Na+].[Cl-]', 'CCO']


def mols(smiles_list):
    return [Chem.MolFromSmiles(s) for s in smiles_list]


@pytest.mark.parametrize('settings', [
    dict(max_radius=2, vec_dimension=1024, sub_counts=False),           # more positions than substructures: padding
    dict(max_radius=2, vec_dimension=1024, sub_counts=True),
    dict(max_radius=2, vec_dimension=37, sub_counts=False),             # slicing, with ties at the cut-off
    dict(max_radius=2, vec_dimension=37, sub_counts=True),
    dict(max_radius=1, vec_dimension=64, sub_counts=False),
    dict(max_radius=3, vec_dimension=128, sub_counts=True),
    dict(max_radius=2, vec_dimension=64, sub_counts=False, pharm_atom_invs=True),
    dict(max_radius=2, vec_dimension=64, sub_counts=True, chirality=True),
    dict(max_radius=2, vec_dimension=64, sub_counts=False, bond_invs=False),
])
def test_equals_reference_implementation(settings):
    reference = create_sort_and_slice_ecfp_featuriser(mols(TRAIN), print_train_set_info=False, **settings)
    fp = SortAndSliceFingerprint(size=settings['vec_dimension'], radius=settings['max_radius'],
                                 counts=settings['sub_counts'],
                                 pharm_atom_invs=settings.get('pharm_atom_invs', False),
                                 bond_invs=settings.get('bond_invs', True),
                                 chirality=settings.get('chirality', False)).fit(mols(TRAIN))
    for mol in mols(TRAIN + TEST):
        expected = np.asarray(reference(mol), dtype=float)
        actual = fp.transform_mol(mol)
        assert actual.shape == (settings['vec_dimension'],)
        assert np.array_equal(actual, expected), Chem.MolToSmiles(mol)


def test_vocabulary_is_the_most_prevalent_substructures():
    fp = SortAndSliceFingerprint(size=10, radius=2).fit(mols(TRAIN))
    prevalence = {}
    for mol in mols(TRAIN):
        for sub_id in fp.substructures(mol):
            prevalence[sub_id] = prevalence.get(sub_id, 0) + 1
    selected = sorted(fp.vocabulary, key=fp.vocabulary.get)
    # every selected substructure is at least as prevalent as every substructure that was left out
    assert min(prevalence[s] for s in selected) >= max(prevalence[s] for s in prevalence if s not in fp.vocabulary)
    # positions follow decreasing prevalence, ties by the larger identifier first
    keys = [(prevalence[s], s) for s in selected]
    assert keys == sorted(keys, reverse=True)
    assert fp.num_reference_molecules == len(TRAIN) and fp.num_reference_substructures == len(prevalence)


def test_binary_and_counts():
    train = mols(['CCO', 'CCCO', 'c1ccccc1'])
    binary = SortAndSliceFingerprint(size=64, radius=1, counts=False).fit(train)
    counts = SortAndSliceFingerprint(size=64, radius=1, counts=True).fit(train)
    mol = Chem.MolFromSmiles('CCCCO')
    assert set(np.unique(binary.transform_mol(mol))) <= {0.0, 1.0}
    assert counts.transform_mol(mol).max() > 1               # several CH2 groups
    assert np.array_equal(counts.transform_mol(mol) > 0, binary.transform_mol(mol) > 0)


def test_independent_of_training_order():
    a = SortAndSliceFingerprint(size=50, radius=2).fit(mols(TRAIN))
    b = SortAndSliceFingerprint(size=50, radius=2).fit(mols(TRAIN[::-1]))
    assert a.vocabulary == b.vocabulary


def test_unseen_substructures_are_ignored():
    fp = SortAndSliceFingerprint(size=2048, radius=2).fit(mols(['CCO', 'CCC']))
    assert len(fp.vocabulary) < 2048                         # fewer substructures than positions
    assert not fp.transform_mol(Chem.MolFromSmiles('Brc1ccccc1')).any()   # nothing in common with the training set
    assert fp.transform_mol(Chem.MolFromSmiles('CCO'))[len(fp.vocabulary):].sum() == 0


def test_fit_transform_split_uses_only_the_fit_indices():
    molecules = dict(enumerate(mols(TRAIN + TEST)))
    train_indices = list(range(len(TRAIN)))
    features = fit_transform_split(SortAndSliceFingerprint(size=256, radius=2), molecules, train_indices)
    assert set(features) == set(molecules) and all(v.shape == (256,) for v in features.values())
    # the vocabulary only depends on the training molecules: replacing a test molecule changes nothing else
    modified = dict(molecules)
    modified[len(TRAIN)] = Chem.MolFromSmiles('IC(I)(I)I')
    features_modified = fit_transform_split(SortAndSliceFingerprint(size=256, radius=2), modified, train_indices)
    assert all(np.array_equal(features[i], features_modified[i]) for i in molecules if i != len(TRAIN))
    # equivalent to fitting on the training molecules directly
    fp = SortAndSliceFingerprint(size=256, radius=2).fit(mols(TRAIN))
    assert all(np.array_equal(features[i], fp.transform_mol(m)) for i, m in molecules.items())


def test_errors():
    with pytest.raises(RuntimeError):
        SortAndSliceFingerprint().transform_mol(Chem.MolFromSmiles('CCO'))
    with pytest.raises(ValueError):
        SortAndSliceFingerprint().fit([])
