"""
Command-line utility to fit a :class:`~graph_hdc.baselines.sherlock.SherlockFingerprint`
descriptor dictionary on a reference corpus of molecules.

The Sherlock fingerprint (SPECTRE, Xu et al. 2026) selects its descriptors by
entropy over a large, *task-independent* corpus. In the paper that corpus is
``COCONUT + LOTUS + DeepSAT training`` (~526k molecules). This script reads SMILES
from one or more corpus files, fits the dictionary once, and saves the fit artifact
so that the experiments can load it as a stateless featurizer.

Supported corpus file formats (auto-detected by extension, override with ``--format``):

- ``.pkl`` -- a pickled ``list``. Either a list of SMILES strings, or a list of
  records whose first element is the SMILES (this matches the authors'
  ``inference_metadata_latest_RDkit.pkl`` layout).
- ``.csv`` / ``.tsv`` -- a table with a SMILES column (name auto-detected among
  ``smiles``/``canonical_smiles``/``SMILES``/``Canonical_SMILES``, or set
  ``--smiles-col``).
- ``.smi`` / ``.txt`` -- one SMILES per line (first whitespace-separated token).

Example
-------

.. code-block:: bash

    python -m graph_hdc.baselines.fit_sherlock \\
        --corpus coconut.csv --corpus lotus.csv \\
        --radius 6 --size 16384 --jobs 16 \\
        --output sherlock_r6_coconut_lotus.pkl
"""

import argparse
import csv
import gzip
import os
import pickle
import sys
from typing import Iterator, List

from graph_hdc.baselines.sherlock import SherlockFingerprint

_SMILES_COLUMN_CANDIDATES = [
    'smiles', 'canonical_smiles', 'SMILES', 'Canonical_SMILES',
    'canonicalsmiles', 'CanonicalSMILES', 'structure_smiles',
]


def _open(path: str):
    """Open a possibly gzip-compressed text file transparently."""
    if path.endswith('.gz'):
        return gzip.open(path, 'rt')
    return open(path, 'r')


def _iter_smiles_from_pickle(path: str) -> Iterator[str]:
    with open(path, 'rb') as f:
        data = pickle.load(f)
    for record in data:
        if isinstance(record, str):
            yield record
        elif isinstance(record, (list, tuple)) and len(record) > 0:
            yield record[0]


def _iter_smiles_from_table(path: str, delimiter: str, smiles_col: str = None) -> Iterator[str]:
    with _open(path) as f:
        reader = csv.reader(f, delimiter=delimiter)
        header = next(reader)
        if smiles_col is not None:
            col = header.index(smiles_col)
        else:
            col = None
            lowered = [h.strip().lower() for h in header]
            for cand in _SMILES_COLUMN_CANDIDATES:
                if cand.lower() in lowered:
                    col = lowered.index(cand.lower())
                    break
            if col is None:
                raise ValueError(
                    f'Could not auto-detect a SMILES column in {path}. Header: {header}. '
                    f'Pass --smiles-col explicitly.'
                )
        for row in reader:
            if col < len(row) and row[col]:
                yield row[col]


def _iter_smiles_from_lines(path: str) -> Iterator[str]:
    with _open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield line.split()[0]


def iter_corpus_smiles(path: str, fmt: str = None, smiles_col: str = None) -> Iterator[str]:
    """
    Yield SMILES strings from a single corpus file, dispatching on its format.

    :param path: Path to the corpus file (optionally ``.gz`` compressed).
    :param fmt: Explicit format override (``pkl``/``csv``/``tsv``/``smi``); auto-detected
        from the extension when ``None``.
    :param smiles_col: Explicit SMILES column name for tabular formats.

    :return: Iterator over SMILES strings.
    """
    base = path[:-3] if path.endswith('.gz') else path
    ext = os.path.splitext(base)[1].lower().lstrip('.')
    fmt = fmt or ext

    if fmt in ('pkl', 'pickle'):
        yield from _iter_smiles_from_pickle(path)
    elif fmt in ('csv',):
        yield from _iter_smiles_from_table(path, ',', smiles_col)
    elif fmt in ('tsv', 'tab'):
        yield from _iter_smiles_from_table(path, '\t', smiles_col)
    elif fmt in ('smi', 'txt'):
        yield from _iter_smiles_from_lines(path)
    else:
        raise ValueError(f'Unsupported corpus format {fmt!r} for {path}.')


def main(argv: List[str] = None) -> int:
    parser = argparse.ArgumentParser(description='Fit a Sherlock Fingerprint descriptor dictionary on a corpus.')
    parser.add_argument('--corpus', action='append', required=True,
                        help='Corpus file with SMILES (repeat for multiple files, e.g. COCONUT and LOTUS).')
    parser.add_argument('--output', '-o', required=True,
                        help='Output path for the full fit artifact (pickle with the complete count table).')
    parser.add_argument('--dictionary-output', default=None,
                        help='Output path for the lightweight dictionary (only the ranked, retained '
                             'descriptors; a few MB, fast to load -- this is what the experiments use). '
                             'Defaults to the --output path with a "_dict" suffix. Pass "none" to skip.')
    parser.add_argument('--radius', type=int, default=6, help='Circular radius (paper: 6).')
    parser.add_argument('--size', type=int, default=16384, help='Number of descriptors to retain (paper: 16384).')
    parser.add_argument('--jobs', '-j', type=int, default=1, help='Worker processes for enumeration.')
    parser.add_argument('--format', default=None, help='Force corpus format (pkl/csv/tsv/smi).')
    parser.add_argument('--smiles-col', default=None, help='SMILES column name for tabular corpora.')
    parser.add_argument('--dedup', action=argparse.BooleanOptionalAction, default=True,
                        help='Deduplicate SMILES (by string) before fitting so that each distinct '
                             'molecule contributes one count. On by default; important for corpora '
                             'like LOTUS that list structure-organism pairs (repeated structures).')
    parser.add_argument('--limit', type=int, default=None, help='Optional cap on number of corpus molecules (debug).')
    args = parser.parse_args(argv)

    def log(msg):
        print(msg, flush=True)

    log(f'Fitting Sherlock Fingerprint: radius={args.radius}, size={args.size}, jobs={args.jobs}')

    smiles: List[str] = []
    for corpus_path in args.corpus:
        log(f'reading corpus file: {corpus_path}')
        n_before = len(smiles)
        for s in iter_corpus_smiles(corpus_path, fmt=args.format, smiles_col=args.smiles_col):
            smiles.append(s)
            if args.limit is not None and len(smiles) >= args.limit:
                break
        log(f' * {len(smiles) - n_before} SMILES read from {corpus_path}')
        if args.limit is not None and len(smiles) >= args.limit:
            log(f' * reached --limit {args.limit}, stopping')
            break

    log(f'total corpus size: {len(smiles)} molecules (raw)')

    if args.dedup:
        n_raw = len(smiles)
        smiles = list(dict.fromkeys(smiles))  # order-preserving string dedup
        log(f'deduplicated: {n_raw} -> {len(smiles)} distinct SMILES')

    sherlock = SherlockFingerprint(radius=args.radius, size=args.size)
    sherlock.fit(smiles, n_jobs=args.jobs, log=log)

    sherlock.save(args.output)
    log(f'saved full fit artifact to: {args.output} '
        f'({len(sherlock.counts_)} distinct descriptors, {len(sherlock.bitinfo_to_index_)} retained)')

    dict_output = args.dictionary_output
    if dict_output is None:
        root, ext = os.path.splitext(args.output)
        dict_output = f'{root}_dict{ext or ".pkl"}'
    if str(dict_output).lower() != 'none':
        sherlock.save_dictionary(dict_output)
        log(f'saved lightweight dictionary to: {dict_output} '
            f'(the SHERLOCK_DICTIONARY_PATH the experiments should use)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
