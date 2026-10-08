"""
Folded variants of the Reymond-group "MinHash" fingerprints MHFP and MAP4.

Both fingerprints were proposed as MinHash sketches of a molecule's set of string "shingles", compared
via the MinHash estimate of the Jaccard distance. The same shingle sets can instead be hashed and
folded into a binary vector of arbitrary length, which is the form that plugs into the same models,
distances and kernels as the other binary fingerprints (Morgan, RDKit, atom pair, ...):

- **SECFP** (``secfp_fingerprint``) -- the folded variant of MHFP, proposed in the MHFP paper as a
  drop-in replacement for ECFP. Shingles are the canonical SMILES of the circular substructures of
  radius 1..r around every atom plus the SMILES of the SSSR rings.

      Probst, Reymond. "A probabilistic molecular fingerprint for big data settings."
      J. Cheminform. 10, 66 (2018). doi:10.1186/s13321-018-0321-8

  Computed with the reference implementation, ``mhfp.encoder.MHFPEncoder.secfp_from_mol`` of the
  ``mhfp`` package (github.com/reymond-group/mhfp, MIT license).

- **MAP4** (``map4_fingerprint``) -- the MinHashed atom-pair fingerprint. Shingles are the strings
  ``CS_i(j) | d(j, k) | CS_i(k)`` for every atom pair (j, k), every radius i = 1..r (r = 2 for MAP4),
  where CS_i is the canonical SMILES of the circular substructure of radius i and d the topological
  distance. Here in its folded form (``MAP4Calculator(is_folded=True)`` of the reference code).

      Capecchi, Probst, Reymond. "One molecular fingerprint to rule them all: drugs, biomolecules,
      and the metabolome." J. Cheminform. 12, 43 (2020). doi:10.1186/s13321-020-00445-4

  The reference package (github.com/reymond-group/map4, MIT license, Copyright (c) 2017 GDB /
  Reymond Research Group) cannot be installed on Python >= 3.11 because it unconditionally imports
  ``tmap``, which the folded variant does not need. ``map4_shingles`` therefore reproduces the
  shingling of the reference code (``MAP4Calculator._calculate``), while the hashing and folding use
  ``mhfp.encoder.MHFPEncoder.hash`` and ``MHFPEncoder.fold`` exactly as the reference does for the
  folded variant (``MAP4Calculator._fold``). The result was verified to be bit-identical to map4 1.0.
"""
import warnings
import itertools
from typing import Dict, List

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdmolops
from mhfp.encoder import MHFPEncoder


def secfp_fingerprint(mol: Chem.Mol, length: int = 2048, radius: int = 3) -> np.ndarray:
    """
    Folded MHFP (SECFP) binary fingerprint of ``mol`` with ``length`` bits, computed with the reference
    implementation (``mhfp.encoder.MHFPEncoder.secfp_from_mol`` with its default shingling settings).

    .. code-block:: python

        fp = secfp_fingerprint(Chem.MolFromSmiles('CCO'), length=1024, radius=3)  # SECFP6

    :param mol: The RDKit molecule.
    :param length: The number of bits of the folded fingerprint.
    :param radius: The maximum radius of the circular substructures (3 = SECFP6, the MHFP default).

    :returns: A numpy uint8 array of shape (length,).
    """
    with warnings.catch_warnings():
        # Molecules with a single heavy atom (e.g. methane in QM9) have no circular substructures of
        # radius >= 1 and no rings; the reference then warns and returns the all-zero fingerprint.
        warnings.filterwarnings('ignore', message='The length of the shingling is 0')
        return MHFPEncoder.secfp_from_mol(mol, length=length, radius=radius)


def _find_env(mol: Chem.Mol, idx: int, radius: int) -> str:
    """Canonical SMILES of the circular substructure of ``radius`` bonds rooted at atom ``idx``."""
    env = rdmolops.FindAtomEnvironmentOfRadiusN(mol, radius, idx)
    atom_map = {}
    submol = Chem.PathToSubmol(mol, env, atomMap=atom_map)
    if idx in atom_map:
        return Chem.MolToSmiles(submol, rootedAtAtom=atom_map[idx], canonical=True, isomericSmiles=False)
    return ''


def map4_shingles(mol: Chem.Mol, radius: int = 2) -> List[bytes]:
    """
    The set of MAP4 atom-pair shingles of ``mol`` (as in ``MAP4Calculator._calculate`` of the reference
    code, without counts).

    :param mol: The RDKit molecule.
    :param radius: The maximum radius of the circular substructures (2 = MAP4).

    :returns: A list of unique utf-8 encoded shingle strings.
    """
    atom_envs: Dict[int, List[str]] = {
        atom.GetIdx(): [_find_env(mol, atom.GetIdx(), r) for r in range(1, radius + 1)]
        for atom in mol.GetAtoms()
    }
    distance_matrix = rdmolops.GetDistanceMatrix(mol)
    shingles = set()
    for idx1, idx2 in itertools.combinations(range(mol.GetNumAtoms()), 2):
        dist = str(int(distance_matrix[idx1][idx2]))
        for i in range(radius):
            env_a, env_b = sorted([atom_envs[idx1][i], atom_envs[idx2][i]])
            shingles.add(f'{env_a}|{dist}|{env_b}'.encode('utf-8'))
    return list(shingles)


def map4_fingerprint(mol: Chem.Mol, length: int = 2048, radius: int = 2) -> np.ndarray:
    """
    Folded MAP4 binary fingerprint of ``mol`` with ``length`` bits: the MAP4 shingles hashed and folded
    with the reference ``mhfp`` functions, as in ``MAP4Calculator._fold`` of the reference code.

    .. code-block:: python

        fp = map4_fingerprint(Chem.MolFromSmiles('CCO'), length=1024)  # MAP4, folded

    :param mol: The RDKit molecule.
    :param length: The number of bits of the folded fingerprint.
    :param radius: The maximum radius of the circular substructures (2 = MAP4).

    :returns: A numpy uint8 array of shape (length,).
    """
    shingles = map4_shingles(mol, radius=radius)
    if not shingles:
        # molecules with a single heavy atom have no atom pairs (the reference returns all zeros as well)
        return np.zeros(length, dtype=np.uint8)
    return MHFPEncoder.fold(MHFPEncoder.hash(shingles), length=length)
