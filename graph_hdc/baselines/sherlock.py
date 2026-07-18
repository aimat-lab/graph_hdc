"""
Sherlock Fingerprint -- entropy-ranked, collision-free Morgan fingerprint.

This module re-implements the *Sherlock Fingerprint* (Structural Hash-based
Entropy-Ranked Local Organic Chemical Keys, "SF") introduced by

    Xu, Ryu, Tong, ... Gerwick, Cottrell.
    "SPECTRE: A Multimodal Spectral Transformer for Small Molecule Annotation."
    J. Chem. Inf. Model. 2026, 66, 2501-2512. DOI: 10.1021/acs.jcim.5c02444

so that it can be used as a *baseline* molecular representation alongside the
hyperdimensional fingerprints (HDF) of this project. The re-implementation
follows the reference code published by the authors at
``github.com/xwd0418/Spectre`` (``datasets/fp_loader_utils.py`` ::
``Hash_Entropy_FP_loader`` and ``notebook_and_scripts/.../find_frags.py``).

Design rationale
----------------
The Sherlock fingerprint is an ECFP/Morgan descendant that fixes two well-known
shortcomings of the standard Morgan fingerprint -- *hash collisions* (distinct
substructures folded onto the same bit via ``hash mod L``) and the resulting
*redundancy*. It does so in two steps:

1. **Collision-free descriptors.** Every circular substructure is identified not
   by its raw 32-bit Morgan hash but by the tuple
   ``(bit_id, central_atom_symbol, canonical_fragment_smiles, radius)``. Because
   the canonical SMILES of the fragment is part of the key, two substructures
   that happen to collide on ``bit_id`` remain *distinct* descriptors. This is
   the key that makes the fingerprint collision-free.

2. **Entropy ranking.** Over a large reference corpus, the per-descriptor
   *presence* frequency ``p = count / corpus_size`` is turned into a Bernoulli
   entropy and the ``size`` most-informative descriptors (``p`` closest to 0.5)
   are retained, each mapped one-to-one to a single output bit. The paper uses
   ``size = 16384`` and ``radius = 6`` over a ~526k-molecule corpus
   (COCONUT + LOTUS + DeepSAT training).

Because the retained descriptors are produced by ranking, a *size-matched*
fingerprint of dimension ``k < 16384`` is simply the ``k`` highest-entropy
descriptors -- i.e. a prefix of the full ranking. :meth:`SherlockFingerprint.set_output`
exposes exactly this, so a single corpus fit supports both the native 16384-bit
fingerprint and any smaller, dimension-matched variant.

Example
-------

.. code-block:: python

    from graph_hdc.baselines import SherlockFingerprint

    # fit the descriptor dictionary on a reference corpus of SMILES (once)
    sf = SherlockFingerprint(radius=6, size=16384)
    sf.fit(corpus_smiles, n_jobs=8)
    sf.save('sherlock_r6.pkl')

    # ... later, in an experiment ...
    sf = SherlockFingerprint.load('sherlock_r6.pkl')
    vec = sf.transform_smiles('CCO')            # native 16384-bit vector

    sf.set_output(size=2048)                    # size-matched variant
    vec_small = sf.transform_smiles('CCO')      # 2048-bit vector
"""

from __future__ import annotations

import pickle
from collections import Counter
from typing import Iterable, Optional, Sequence

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

# A single circular-substructure descriptor. Mirrors the "bit_info" tuple used by
# the reference implementation: (morgan_bit_id, central_atom_symbol,
# canonical_fragment_smiles, radius). Including the fragment SMILES is what makes
# the descriptor collision-free.
BitInfo = tuple

#: Radius of the native Sherlock fingerprint from the SPECTRE paper.
DEFAULT_RADIUS: int = 6
#: Dimensionality of the native Sherlock fingerprint from the SPECTRE paper.
DEFAULT_SIZE: int = 16384


# -- generator cache ---------------------------------------------------------
# rdFingerprintGenerator objects are stateless and reusable; cache one per radius
# so that we do not rebuild it for every molecule during a large corpus fit.
_GENERATOR_CACHE: dict = {}


def _get_generator(radius: int):
    gen = _GENERATOR_CACHE.get(radius)
    if gen is None:
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius)
        _GENERATOR_CACHE[radius] = gen
    return gen


def enumerate_bit_infos(
    smiles: str,
    radius: int = DEFAULT_RADIUS,
    ignore_atoms: Sequence[int] = (),
) -> set:
    """
    Enumerate the set of collision-free Sherlock descriptors present in a molecule.

    For the molecule given by ``smiles`` this computes the Morgan bit-info map up
    to ``radius`` and, for every circular atom environment, builds the descriptor
    tuple ``(bit_id, central_atom_symbol, canonical_fragment_smiles, radius)``.
    Each descriptor is reported *once* (presence, not count) -- matching the
    reference ``count_circular_substructures`` behaviour where, within a single
    molecule, each fragment is simply on or off.

    :param smiles: SMILES string of the molecule.
    :param radius: Maximum circular radius to enumerate (Morgan generator radius).
    :param ignore_atoms: Optional atom indices to exclude from the enumeration.
        Descriptors centered on these atoms are not reported. Used e.g. to blank
        out parts of a molecule.

    :return: A ``set`` of :data:`BitInfo` tuples present in the molecule. Returns
        an empty set for an invalid SMILES.
    """
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else smiles
    if mol is None:
        return set()

    gen = _get_generator(radius)
    ao = rdFingerprintGenerator.AdditionalOutput()
    ao.AllocateBitInfoMap()
    gen.GetSparseFingerprint(mol, additionalOutput=ao)
    info = ao.GetBitInfoMap()

    ignore = set(ignore_atoms)
    bit_infos: set = set()
    for bit_id, atom_envs in info.items():
        for atom_idx, env_radius in atom_envs:
            if atom_idx in ignore:
                continue
            env = Chem.FindAtomEnvironmentOfRadiusN(mol, env_radius, atom_idx)
            submol = Chem.PathToSubmol(mol, env)
            frag_smiles = Chem.MolToSmiles(submol, canonical=True)
            atom_symbol = mol.GetAtomWithIdx(atom_idx).GetSymbol()
            bit_infos.add((bit_id, atom_symbol, frag_smiles, env_radius))

    return bit_infos


def bernoulli_entropy(counts: np.ndarray, corpus_size: int) -> np.ndarray:
    """
    Compute the (signed) Bernoulli entropy used for descriptor ranking.

    This reproduces the reference implementation exactly, including its sign
    convention: it returns ``p*log2(p) + (1-p)*log2(1-p)`` (i.e. the *negative*
    of the Shannon entropy) so that the *smallest* (most negative) values
    correspond to the *most* informative descriptors (presence probability
    ``p`` closest to 0.5). Sorting ascending and keeping the head therefore keeps
    the highest-entropy descriptors.

    :param counts: Array of per-descriptor document counts (number of corpus
        molecules containing the descriptor).
    :param corpus_size: Total number of molecules in the reference corpus.

    :return: Array of signed entropy values, same shape as ``counts``.
    """
    p = counts / corpus_size
    return (
        p * np.log2(np.clip(p, 1e-7, 1))
        + (1 - p) * np.log2(np.clip(1 - p, 1e-7, 1))
    )


# -- multiprocessing worker --------------------------------------------------
_WORKER_RADIUS: int = DEFAULT_RADIUS


def _init_worker(radius: int) -> None:
    global _WORKER_RADIUS
    _WORKER_RADIUS = radius


def _count_worker(smiles: str) -> frozenset:
    return frozenset(enumerate_bit_infos(smiles, _WORKER_RADIUS))


class SherlockFingerprint:
    """
    Entropy-ranked, collision-free Morgan fingerprint (Sherlock Fingerprint).

    The fingerprint is a two-stage object: it is first :meth:`fit` on a reference
    corpus of SMILES (which accumulates per-descriptor document counts and selects
    the highest-entropy descriptors), and thereafter :meth:`transform_smiles`
    converts an arbitrary molecule into a binary presence/absence vector.

    The corpus fit is *task independent* -- exactly as in the paper, the descriptor
    dictionary is derived from a large generic molecule collection, not from the
    downstream prediction dataset -- so a fitted instance can be saved once and
    reused as a stateless featurizer across datasets without any data leakage.

    :param radius: Circular radius of the fingerprint (``6`` in the paper). This is
        both the enumeration radius and the maximum descriptor radius kept.
    :param size: Output dimensionality, i.e. number of descriptors kept (``16384``
        in the paper). Can be changed after fitting via :meth:`set_output` to obtain
        a size-matched variant.

    :ivar counts_: ``dict`` mapping every observed :data:`BitInfo` descriptor to
        the number of corpus molecules it appeared in (the fit artifact).
    :ivar corpus_size_: Number of molecules the dictionary was fit on.
    :ivar bitinfo_to_index_: ``dict`` mapping each *retained* descriptor to its
        output bit index; ``None`` until :meth:`fit`/:meth:`set_output` runs.
    """

    def __init__(self, radius: int = DEFAULT_RADIUS, size: int = DEFAULT_SIZE) -> None:
        self.radius: int = radius
        self.size: int = size

        # populated by fit()
        self.counts_: Optional[dict] = None
        self.corpus_size_: Optional[int] = None
        self.fit_radius_: Optional[int] = None
        # number of distinct descriptors seen in the corpus; kept even for a lightweight
        # dictionary (where the full ``counts_`` table is dropped) for reporting.
        self.n_descriptors_: Optional[int] = None

        # populated by _select() / set_output()
        self.bitinfo_to_index_: Optional[dict] = None
        self.index_to_bitinfo_: Optional[dict] = None
        # the retained descriptors in entropy-rank order (index 0..size-1); this is all
        # that a lightweight, count-free dictionary needs, and it supports size-matching
        # by prefix slicing.
        self.ranked_bitinfos_: Optional[list] = None

    # -- fitting -------------------------------------------------------------

    def fit(
        self,
        smiles_iterable: Iterable[str],
        n_jobs: int = 1,
        chunksize: int = 64,
        log=None,
    ) -> 'SherlockFingerprint':
        """
        Fit the descriptor dictionary on a reference corpus of SMILES.

        Every molecule is decomposed into its collision-free circular descriptors
        (up to :attr:`radius`) and a global document-count per descriptor is
        accumulated. The highest-entropy ``size`` descriptors are then selected and
        assigned output bit indices.

        The full ``counts_`` table is retained (not just the selected descriptors)
        so that :meth:`set_output` can later re-select a different ``size`` (or a
        smaller ``max_radius``) without re-scanning the corpus.

        :param smiles_iterable: Iterable of SMILES strings forming the corpus.
        :param n_jobs: Number of worker processes for the (embarrassingly parallel)
            enumeration. ``1`` runs serially; ``>1`` uses a ``multiprocessing.Pool``.
        :param chunksize: Task chunk size for the process pool.
        :param log: Optional callable ``log(msg)`` for progress reporting (e.g. an
            experiment logger). Progress is emitted every 20000 molecules.

        :return: ``self`` (fitted).
        """
        counts: Counter = Counter()
        corpus_size = 0

        def _report(n):
            if log is not None and n % 20000 == 0:
                log(f' * fit: processed {n} corpus molecules, '
                    f'{len(counts)} distinct descriptors so far')

        smiles_list = list(smiles_iterable)

        if n_jobs is not None and n_jobs > 1:
            import multiprocessing as mp
            with mp.Pool(
                processes=n_jobs,
                initializer=_init_worker,
                initargs=(self.radius,),
            ) as pool:
                for bit_infos in pool.imap_unordered(
                    _count_worker, smiles_list, chunksize=chunksize
                ):
                    counts.update(bit_infos)
                    corpus_size += 1
                    _report(corpus_size)
        else:
            for smiles in smiles_list:
                counts.update(enumerate_bit_infos(smiles, self.radius))
                corpus_size += 1
                _report(corpus_size)

        self.counts_ = dict(counts)
        self.corpus_size_ = corpus_size
        self.fit_radius_ = self.radius
        self.n_descriptors_ = len(self.counts_)
        if log is not None:
            log(f' * fit done: {corpus_size} molecules, '
                f'{len(self.counts_)} distinct descriptors')

        self._select(size=self.size, max_radius=self.radius)
        return self

    # -- descriptor selection ------------------------------------------------

    def _select(self, size: int, max_radius: int) -> None:
        """
        Select the ``size`` highest-entropy descriptors with radius ``<= max_radius``.

        This is the step that turns the raw ``counts_`` table into a concrete
        ``descriptor -> bit index`` mapping. It reproduces the reference
        ``setup``/``keep_smallest_entropy`` logic: filter by radius, compute the
        signed Bernoulli entropy, ``argsort`` ascending with a *stable* sort, and
        keep the first ``size`` descriptors.

        :param size: Number of descriptors (output bits) to keep.
        :param max_radius: Maximum descriptor radius to consider.
        """
        if self.counts_ is None:
            raise RuntimeError('SherlockFingerprint is not fitted; call fit() or load() first.')

        items = [(bi, c) for bi, c in self.counts_.items() if bi[3] <= max_radius]
        if not items:
            raise ValueError(f'No descriptors with radius <= {max_radius} in the fitted corpus.')

        bitinfos, counts = zip(*items)
        counts = np.asarray(counts, dtype=np.float64)
        entropy = bernoulli_entropy(counts, self.corpus_size_)

        n_keep = min(size, len(bitinfos))
        if n_keep < size:
            # Not enough distinct descriptors in the corpus for the requested size.
            # Keep everything available; the remaining bits stay all-zero.
            pass
        order = np.argsort(entropy, kind='stable')[:n_keep]

        self.bitinfo_to_index_ = {bitinfos[idx]: bit for bit, idx in enumerate(order)}
        self.index_to_bitinfo_ = {bit: bitinfos[idx] for bit, idx in enumerate(order)}
        self.ranked_bitinfos_ = [self.index_to_bitinfo_[bit] for bit in range(len(order))]
        self.size = size
        self.radius = max_radius

    def _select_from_ranked(self, size: int) -> None:
        """
        Select the top-``size`` descriptors from a pre-ranked list (lightweight path).

        Used when only the ranked descriptor list is available (a dictionary loaded via
        :meth:`load` from :meth:`save_dictionary`, without the full count table). Because
        the list is already entropy-ordered, size-matching is a simple prefix slice.

        :param size: Number of descriptors (output bits) to keep; must be ``<=`` the number
            of ranked descriptors available.
        """
        if self.ranked_bitinfos_ is None:
            raise RuntimeError('SherlockFingerprint has no descriptors; call fit() or load() first.')
        if size > len(self.ranked_bitinfos_):
            raise ValueError(
                f'requested size {size} exceeds the {len(self.ranked_bitinfos_)} descriptors in this '
                f'(lightweight) dictionary; re-fit or load the full artifact for a larger size.'
            )
        kept = self.ranked_bitinfos_[:size]
        self.bitinfo_to_index_ = {bi: i for i, bi in enumerate(kept)}
        self.index_to_bitinfo_ = {i: bi for i, bi in enumerate(kept)}
        self.size = size

    def set_output(self, size: Optional[int] = None, max_radius: Optional[int] = None) -> 'SherlockFingerprint':
        """
        Reconfigure the output dimensionality (and/or radius) after fitting.

        This is how a *size-matched* fingerprint is produced: because descriptors
        are entropy-ranked, requesting a smaller ``size`` simply keeps the top-``size``
        highest-entropy descriptors -- a prefix of the native ranking. No re-fit is
        required.

        When the full count table is available (fitted, or loaded via :meth:`save`) any
        ``size`` up to the number of distinct descriptors and any ``max_radius <=`` the
        fitted radius can be selected. When only a lightweight dictionary is loaded (via
        :meth:`save_dictionary`), ``max_radius`` is fixed and ``size`` may only be reduced
        (a prefix of the stored ranking).

        :param size: New output dimensionality. Defaults to the current :attr:`size`.
        :param max_radius: New maximum descriptor radius. Must be ``<=`` the radius
            the corpus was fit with. Defaults to the current :attr:`radius`.

        :return: ``self``.
        """
        size = self.size if size is None else size
        max_radius = self.radius if max_radius is None else max_radius

        if self.counts_ is not None:
            if self.fit_radius_ is not None and max_radius > self.fit_radius_:
                raise ValueError(
                    f'max_radius={max_radius} exceeds the fitted radius {self.fit_radius_}; '
                    f're-fit with a larger radius to use it.'
                )
            self._select(size=size, max_radius=max_radius)
        else:
            # lightweight dictionary: no counts, only the ranked list at the fitted radius
            if max_radius != self.radius:
                raise ValueError(
                    f'this is a lightweight dictionary (no count table); max_radius is fixed at '
                    f'{self.radius}. Load the full artifact (save()) to re-select by radius.'
                )
            self._select_from_ranked(size=size)
        return self

    # -- transform -----------------------------------------------------------

    def transform_smiles(self, smiles: str, ignore_atoms: Sequence[int] = ()) -> np.ndarray:
        """
        Convert a single molecule into its Sherlock fingerprint vector.

        :param smiles: SMILES string of the molecule.
        :param ignore_atoms: Optional atom indices to exclude (see
            :func:`enumerate_bit_infos`).

        :return: A ``float64`` numpy array of length :attr:`size`. Bits corresponding
            to retained descriptors present in the molecule are ``1.0``, the rest
            ``0.0``. An invalid SMILES yields an all-zero vector.
        """
        if self.bitinfo_to_index_ is None:
            raise RuntimeError('SherlockFingerprint is not fitted; call fit() or load() first.')

        vec = np.zeros(self.size, dtype=np.float64)
        for bit_info in enumerate_bit_infos(smiles, self.radius, ignore_atoms=ignore_atoms):
            j = self.bitinfo_to_index_.get(bit_info)
            if j is not None:
                vec[j] = 1.0
        return vec

    def transform(self, smiles_list: Sequence[str]) -> np.ndarray:
        """
        Convert a list of molecules into a stacked Sherlock fingerprint matrix.

        :param smiles_list: Sequence of SMILES strings.

        :return: A ``(len(smiles_list), size)`` ``float64`` array.
        """
        return np.stack([self.transform_smiles(s) for s in smiles_list])

    # -- persistence ---------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Persist the full fit artifact (descriptor counts + configuration) to disk.

        The saved object contains the complete ``counts_`` table, so a reloaded
        instance supports :meth:`set_output` for any ``size``/``max_radius`` without
        re-scanning the corpus.

        :param path: Destination file path (a pickle).
        """
        state = {
            'format': 'full',
            'radius': self.radius,
            'size': self.size,
            'counts_': self.counts_,
            'corpus_size_': self.corpus_size_,
            'fit_radius_': self.fit_radius_,
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

    def save_dictionary(self, path: str) -> None:
        """
        Persist a *lightweight* dictionary: only the retained, entropy-ranked descriptors.

        Unlike :meth:`save`, this drops the (potentially multi-gigabyte) full count table
        and stores only the ``size`` selected descriptors in rank order. The resulting file
        is a few MB and loads quickly, which is what the experiments use. A dictionary loaded
        from this file supports the native size and any *smaller* size-matched variant (a
        prefix of the ranking) at the fitted radius, but cannot re-select a different radius
        (use :meth:`save`/:meth:`load` for that).

        :param path: Destination file path (a pickle).
        """
        if self.ranked_bitinfos_ is None:
            raise RuntimeError('SherlockFingerprint is not fitted; call fit() or load() first.')
        state = {
            'format': 'light',
            'radius': self.radius,
            'size': self.size,
            'fit_radius_': self.fit_radius_,
            'corpus_size_': self.corpus_size_,
            'n_descriptors': len(self.counts_) if self.counts_ is not None else None,
            'ranked_bitinfos_': self.ranked_bitinfos_,
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str) -> 'SherlockFingerprint':
        """
        Load a fingerprint previously saved with :meth:`save` or :meth:`save_dictionary`.

        The file format (full count table vs. lightweight ranked dictionary) is detected
        automatically.

        :param path: Path to the pickle written by :meth:`save`/:meth:`save_dictionary`.

        :return: A fitted :class:`SherlockFingerprint` with its output configured to the
            saved ``size``/``radius``.
        """
        with open(path, 'rb') as f:
            state = pickle.load(f)

        if state.get('format') == 'light':
            obj = cls(radius=state['radius'], size=state['size'])
            obj.corpus_size_ = state['corpus_size_']
            obj.fit_radius_ = state['fit_radius_']
            obj.n_descriptors_ = state.get('n_descriptors')
            obj.ranked_bitinfos_ = state['ranked_bitinfos_']
            obj._select_from_ranked(size=state['size'])
            return obj

        # full artifact (default / legacy)
        obj = cls(radius=state['fit_radius_'] or state['radius'], size=state['size'])
        obj.counts_ = state['counts_']
        obj.corpus_size_ = state['corpus_size_']
        obj.fit_radius_ = state['fit_radius_']
        obj.n_descriptors_ = len(state['counts_']) if state['counts_'] is not None else None
        obj._select(size=state['size'], max_radius=state['radius'])
        return obj
