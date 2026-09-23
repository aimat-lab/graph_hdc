"""
Unit tests for the folded MHFP (SECFP) and MAP4 fingerprints
(:mod:`graph_hdc.baselines.minhash_fps`).

The reference implementations (the ``mhfp`` and ``map4`` packages of the Reymond group) cannot be
installed next to this project (``map4`` requires ``tmap``, which has no Python 3.11 wheels), so the
expected set bits below were generated once with the reference code (mhfp 1.9.6
``MHFPEncoder.secfp_from_mol(mol, length=1024, radius=3)`` and map4 1.0
``MAP4Calculator(dimensions=1024, radius=2, is_folded=True)``, rdkit 2025.9.1) and are pinned here.
The re-implementation was verified to be bit-identical to the reference on 1200 molecules of
FreeSolv, Lipophilicity and AqSolDB for sizes 8 to 16384 and SECFP radii 1 to 3.
"""
import numpy as np
import pytest
from rdkit import Chem

from graph_hdc.baselines.minhash_fps import secfp_fingerprint, map4_fingerprint, map4_shingles

REFERENCE_BITS = {
    'CC(=O)Nc1ccc(O)cc1': {
        'secfp6_1024': [25, 55, 69, 125, 132, 144, 272, 284, 413, 417, 453, 514, 518, 541, 542, 556, 582, 637, 792, 796, 817, 822, 855, 909, 977, 1005],
        'map4_1024': [2, 18, 19, 23, 28, 34, 50, 67, 72, 79, 81, 82, 93, 98, 114, 118, 121, 157, 181, 194, 216, 220, 222, 263, 306, 314, 327, 333, 347, 357, 367, 369, 392, 395, 401, 443, 451, 468, 475, 491, 499, 517, 572, 611, 612, 621, 667, 677, 684, 688, 689, 697, 698, 714, 738, 744, 749, 760, 778, 803, 814, 833, 845, 874, 875, 884, 934, 943, 958, 964, 992],
    },
    'C[C@H](N)C(=O)O': {
        'secfp6_1024': [88, 119, 238, 319, 453, 471, 473, 676, 687, 743, 750, 770, 781, 787, 855, 877],
        'map4_1024': [3, 18, 21, 27, 35, 46, 121, 133, 170, 247, 258, 269, 271, 325, 328, 339, 347, 395, 428, 455, 472, 562, 598, 604, 708, 731, 733, 757, 800, 965],
    },
    'c1ccc2ccccc2c1': {
        'secfp6_1024': [132, 137, 149, 372, 822, 841, 951, 957, 1008],
        'map4_1024': [18, 80, 222, 225, 267, 306, 396, 407, 414, 447, 450, 476, 491, 518, 666, 697, 783, 875, 917, 976, 981, 1011, 1017],
    },
    'CCN(CC)CCOC(=O)c1ccc(N)cc1': {
        'secfp6_1024': [31, 38, 56, 68, 128, 132, 177, 184, 186, 233, 255, 272, 281, 316, 321, 349, 425, 453, 484, 488, 489, 538, 556, 680, 742, 779, 796, 822, 827, 838, 841, 855, 859, 860, 873, 926, 972, 995],
        'map4_1024': [9, 10, 13, 14, 16, 18, 21, 23, 26, 34, 35, 38, 44, 48, 55, 62, 67, 79, 85, 97, 109, 127, 130, 133, 137, 140, 143, 184, 215, 222, 223, 242, 247, 248, 253, 281, 284, 291, 316, 329, 334, 338, 342, 344, 364, 370, 389, 390, 422, 423, 468, 475, 489, 490, 496, 501, 504, 509, 514, 519, 532, 544, 552, 553, 564, 578, 587, 592, 596, 603, 611, 613, 627, 634, 644, 652, 671, 677, 683, 688, 697, 700, 702, 714, 718, 719, 723, 739, 744, 746, 747, 748, 749, 755, 763, 764, 770, 772, 777, 778, 781, 796, 800, 818, 824, 831, 832, 833, 835, 836, 837, 841, 846, 848, 856, 859, 866, 875, 880, 889, 894, 899, 903, 911, 931, 933, 936, 952, 966, 967, 968, 971, 976, 981, 982, 987, 989, 991, 993, 994, 1012, 1015, 1019, 1021, 1022],
    },
}


@pytest.mark.parametrize('smiles', list(REFERENCE_BITS))
def test_secfp_matches_reference(smiles):
    fp = secfp_fingerprint(Chem.MolFromSmiles(smiles), length=1024, radius=3)
    assert np.flatnonzero(fp).tolist() == REFERENCE_BITS[smiles]['secfp6_1024']


@pytest.mark.parametrize('smiles', list(REFERENCE_BITS))
def test_map4_matches_reference(smiles):
    fp = map4_fingerprint(Chem.MolFromSmiles(smiles), length=1024, radius=2)
    assert np.flatnonzero(fp).tolist() == REFERENCE_BITS[smiles]['map4_1024']


@pytest.mark.parametrize('encode', [secfp_fingerprint, map4_fingerprint])
@pytest.mark.parametrize('length', [8, 100, 2048, 16384])
def test_fingerprint_shape_and_binary(encode, length):
    fp = encode(Chem.MolFromSmiles('CC(=O)Nc1ccc(O)cc1'), length=length)
    assert fp.shape == (length,)
    assert fp.dtype == np.uint8
    assert set(np.unique(fp)) <= {0, 1}
    assert fp.sum() > 0


def test_folding_is_consistent_across_lengths():
    """A bit set at length 2048 must map onto the same bit modulo 1024 (hashes are folded modulo)."""
    mol = Chem.MolFromSmiles('CCN(CC)CCOC(=O)c1ccc(N)cc1')
    for encode in (secfp_fingerprint, map4_fingerprint):
        large, small = encode(mol, length=2048), encode(mol, length=1024)
        assert set((np.flatnonzero(large) % 1024).tolist()) == set(np.flatnonzero(small).tolist())


def test_map4_shingles_format():
    shingles = map4_shingles(Chem.MolFromSmiles('CCO'), radius=2)
    # 3 atom pairs x 2 radii, shingles are unique "env|distance|env" strings
    assert len(shingles) == len(set(shingles)) <= 6
    assert all(s.decode().count('|') == 2 for s in shingles)
