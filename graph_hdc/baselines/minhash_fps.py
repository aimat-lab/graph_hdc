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

  Bit-identical to ``mhfp.encoder.MHFPEncoder.secfp_from_mol`` of the reference package
  (github.com/reymond-group/mhfp, MIT): the shingling is taken from RDKit's built-in implementation
  (``rdkit.Chem.rdMHFPFingerprint``; with ``isomeric=True, kekulize=False`` its shingles equal those of
  the reference package, which writes isomeric SMILES and whose kekulize option has no effect with
  current RDKit versions; verified on 1200 molecules of FreeSolv/Lipophilicity/AqSolDB), but RDKit's
  ``EncodeSECFPMol`` uses a different hash function than the reference, so hashing and folding follow
  the reference (first 4 bytes of the SHA-1 digest, modulo the vector length).

- **MAP4** (``map4_fingerprint``) -- the MinHashed atom-pair fingerprint. Shingles are the strings
  ``CS_i(j) | d(j, k) | CS_i(k)`` for every atom pair (j, k), every radius i = 1..r (r = 2 for MAP4),
  where CS_i is the canonical SMILES of the circular substructure of radius i and d the topological
  distance. Here in its folded form (``MAP4Calculator(is_folded=True)`` of the reference code).

      Capecchi, Probst, Reymond. "One molecular fingerprint to rule them all: drugs, biomolecules,
      and the metabolome." J. Cheminform. 12, 43 (2020). doi:10.1186/s13321-020-00445-4

  Re-implementation of the reference code (github.com/reymond-group/map4, MIT license, Copyright (c)
  2017 GDB / Reymond Research Group). The reference package cannot be installed on Python >= 3.11
  because it unconditionally imports ``tmap``; the folded path does not need ``tmap`` and is
  reproduced here bit-for-bit (SHA-1 shingle hashes as in ``mhfp.encoder.MHFPEncoder.hash``, folded
  modulo the vector length as in ``MHFPEncoder.fold``).
"""
import struct
import itertools
from hashlib import sha1
from typing import Dict, List

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdmolops
from rdkit.Chem.rdMHFPFingerprint import MHFPEncoder


# The shingling does not use the MinHash permutations, so a single encoder instance is enough.
_SHINGLING_ENCODER = MHFPEncoder(1, 42)


def _fold_shingles(shingles: List[bytes], length: int) -> np.ndarray:
    """Hash shingles as ``mhfp.encoder.MHFPEncoder.hash`` and fold them as ``MHFPEncoder.fold``."""
    array = np.zeros(length, dtype=np.uint8)
    if shingles:
        hashes = np.array([struct.unpack('<I', sha1(s).digest()[:4])[0] for s in shingles], dtype=np.uint64)
        array[hashes % length] = 1
    return array


def secfp_fingerprint(mol: Chem.Mol, length: int = 2048, radius: int = 3) -> np.ndarray:
    """
    Folded MHFP (SECFP) binary fingerprint of ``mol`` with ``length`` bits.

    .. code-block:: python

        fp = secfp_fingerprint(Chem.MolFromSmiles('CCO'), length=1024, radius=3)  # SECFP6

    :param mol: The RDKit molecule.
    :param length: The number of bits of the folded fingerprint.
    :param radius: The maximum radius of the circular substructures (3 = SECFP6, the MHFP default).

    :returns: A numpy uint8 array of shape (length,).
    """
    shingles = _SHINGLING_ENCODER.CreateShinglingFromMol(
        mol,
        radius=radius,
        rings=True,
        isomeric=True,
        kekulize=False,
        min_radius=1,
    )
    return _fold_shingles([s.encode('utf-8') for s in shingles], length)


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
    The set of MAP4 atom-pair shingles of ``mol`` (as in ``MAP4Calculator._calculate``).

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
    Folded MAP4 binary fingerprint of ``mol`` with ``length`` bits.

    .. code-block:: python

        fp = map4_fingerprint(Chem.MolFromSmiles('CCO'), length=1024)  # MAP4, folded

    :param mol: The RDKit molecule.
    :param length: The number of bits of the folded fingerprint.
    :param radius: The maximum radius of the circular substructures (2 = MAP4).

    :returns: A numpy uint8 array of shape (length,).
    """
    return _fold_shingles(map4_shingles(mol, radius=radius), length)
