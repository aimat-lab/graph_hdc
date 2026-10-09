"""
Sort & Slice ECFP: an extended-connectivity fingerprint vectorised without hash-based folding.

A hashed ECFP folds an unbounded set of integer substructure identifiers into a fixed number of bits, so that
different substructures can share a bit. Sort & Slice instead ranks the identifiers by the number of reference
molecules that contain them (no labels involved; usually the training molecules) and gives each of the
``size`` most prevalent identifiers its own position. Substructures outside this vocabulary are ignored.

    Dablander, Hanser, Lambiotte, Morris. "Sort & Slice: a simple and superior alternative to hash-based
    folding for extended-connectivity fingerprints." J. Cheminform. 16, 135 (2024).
    doi:10.1186/s13321-024-00932-y

The reference implementation is ``create_sort_and_slice_ecfp_featuriser`` in github.com/oxpig/ECFP-Sort-and-Slice
(MIT license). It looks up the position of every substructure in a Python list, which is too slow for datasets of
the size of QM9, so the same procedure is implemented here with a dictionary. The tests
(tests/test_baselines_sort_and_slice.py) check that the output equals that of the reference function, of which
tests/reference/ holds a verbatim copy.

.. code-block:: python

    fp = SortAndSliceFingerprint(size=2048, radius=2)
    fp.fit(train_mols)
    x = fp.transform_mol(mol)   # float array of shape (2048,)
"""
from collections import Counter
from typing import Dict, Hashable, Iterable, Sequence

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator


class SortAndSliceFingerprint:
    """
    ECFP vectorised via Sort & Slice. The parameters correspond to those of the reference
    ``create_sort_and_slice_ecfp_featuriser``: ``radius`` = ``max_radius``, ``size`` = ``vec_dimension`` and
    ``counts`` = ``sub_counts``, while the remaining ones have the same names. Ties (identifiers contained in
    equally many reference molecules) are broken by the larger identifier first, as with the default
    ``break_ties_with`` of the reference.

    :param size: The fingerprint length, i.e. the maximum number of substructures in the vocabulary. If the
        reference molecules contain fewer distinct substructures, the remaining positions are always zero.
    :param radius: The maximum radius of the ECFP substructures (2 = ECFP4).
    :param counts: Whether a position holds the number of occurrences of its substructure (True, the default
        of the reference) or only its presence as 0/1 (False).
    :param pharm_atom_invs: Pharmacophoric atom invariants (FCFP) instead of the standard ECFP ones.
    :param bond_invs: Whether the bond types enter the substructure identifiers.
    :param chirality: Whether chirality enters the substructure identifiers.
    """

    def __init__(self, size: int = 2048, radius: int = 2, counts: bool = False, pharm_atom_invs: bool = False,
                 bond_invs: bool = True, chirality: bool = False):
        self.size = size
        self.radius = radius
        self.counts = counts
        # the same generator settings as the reference implementation
        self.generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=radius,
            atomInvariantsGenerator=(rdFingerprintGenerator.GetMorganFeatureAtomInvGen() if pharm_atom_invs
                                     else rdFingerprintGenerator.GetMorganAtomInvGen(includeRingMembership=True)),
            useBondTypes=bond_invs,
            includeChirality=chirality,
        )
        # substructure identifier -> position in the fingerprint; None until fitted
        self.vocabulary: Dict[int, int] = None

    def substructures(self, mol: Chem.Mol) -> Dict[int, int]:
        """The integer ECFP substructure identifiers of ``mol``, mapped to their number of occurrences."""
        return dict(self.generator.GetSparseCountFingerprint(mol).GetNonzeroElements())

    def fit_substructures(self, substructure_dicts: Iterable[Dict[int, int]]) -> 'SortAndSliceFingerprint':
        """Fit the vocabulary on the substructures of the reference molecules (see ``substructures``)."""
        prevalence, num_molecules = Counter(), 0
        for substructures in substructure_dicts:
            prevalence.update(substructures.keys())
            num_molecules += 1
        if num_molecules == 0:
            raise ValueError('cannot fit the Sort & Slice vocabulary on zero reference molecules')
        ranked = sorted(prevalence, key=lambda sub_id: (prevalence[sub_id], sub_id), reverse=True)
        self.vocabulary = {sub_id: position for position, sub_id in enumerate(ranked[:self.size])}
        self.num_reference_molecules = num_molecules
        self.num_reference_substructures = len(prevalence)
        return self

    def fit(self, mols: Iterable[Chem.Mol]) -> 'SortAndSliceFingerprint':
        """Fit the vocabulary on the given reference molecules."""
        return self.fit_substructures(self.substructures(mol) for mol in mols)

    def transform_substructures(self, substructures: Dict[int, int]) -> np.ndarray:
        """The fingerprint (float array of shape (size,)) of a molecule with the given substructures."""
        if self.vocabulary is None:
            raise RuntimeError('SortAndSliceFingerprint is not fitted; call fit() or fit_substructures() first.')
        vector = np.zeros(self.size)
        for sub_id, count in substructures.items():
            position = self.vocabulary.get(sub_id)
            if position is not None:
                vector[position] = count if self.counts else 1
        return vector

    def transform_mol(self, mol: Chem.Mol) -> np.ndarray:
        """The fingerprint (float array of shape (size,)) of ``mol``."""
        return self.transform_substructures(self.substructures(mol))

    __call__ = transform_mol


def fit_transform_split(fingerprint: SortAndSliceFingerprint,
                        mols: Dict[Hashable, Chem.Mol],
                        fit_indices: Sequence[Hashable],
                        ) -> Dict[Hashable, np.ndarray]:
    """
    Fit ``fingerprint`` on the molecules ``fit_indices`` of ``mols`` (e.g. the training split; no labels are
    involved) and encode all molecules of ``mols`` with it, computing the substructures of each molecule once.

    :returns: A dict mapping every index of ``mols`` to its fingerprint (float array of shape (size,)).
    """
    substructures = {index: fingerprint.substructures(mol) for index, mol in mols.items()}
    fingerprint.fit_substructures(substructures[index] for index in fit_indices)
    return {index: fingerprint.transform_substructures(s) for index, s in substructures.items()}
