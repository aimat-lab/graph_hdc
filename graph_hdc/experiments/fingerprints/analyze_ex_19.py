"""
Analysis for Experiment 19 (HDF vs. Morgan, SECFP, MAP4, Sort & Slice and Sherlock with the MLP, on the splits of
the GNN comparison; reviewer comment R2.1). See ``_slurm_ex_19.py`` for the setup.

The fingerprint results come from the archives of the given prefix (modules predict_molecules__fp and
predict_molecules__sherlock), the HDF results (MLP) from the current round of ex_14 (``analyze_ex_14.collect``).
Every fingerprint run must have used exactly the split (train, validation and test indices, in order) of the HDF
run of its target and seed, which the paired tests rely on; mismatches are printed.

Writes to ``_ex19/``:

* ``results_<prefix>.csv``             one row per (representation, dataset, seed) with the test MAE / R2
* ``fp_comparison_mlp_<prefix>.tex``   SI-style table, built with the table code of ``analyze_ex_14.py``: mean
                                       (std below), best mean bold, triangles vs. HDF (paired Wilcoxon over the
                                       seeds, Holm-corrected within each target)

Usage:
    python analyze_ex_19.py                                       # prefix ex_19_fp, HDF from prefix ex_14_gnn
    python analyze_ex_19.py ex_19_smoke --hdf-prefix ex_14_smoke  # e.g. a local smoke run
    python analyze_ex_19.py ex_19_fp --csv _ex19/results_ex_19_fp.csv   # rebuild the table from a results CSV
"""
import os
import sys
import csv
import json
import glob
import argparse

import numpy as np

import analyze_ex_14
from analyze_ex_14 import table

PATH = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(PATH, 'results')
OUT = os.path.join(PATH, '_ex19')

REPRESENTATIONS = [('hdf', 'HDF'), ('morgan', 'Morgan'), ('secfp', 'SECFP'), ('map4', 'MAP4'),
                   ('sort_slice', 'Sort \\& Slice'), ('sherlock', 'Sherlock')]
FIELDS = ['variant', 'arch', 'model', 'dataset', 'seed', 'mae', 'r2']


def collect_fingerprints(prefix: str):
    """
    Records ``{variant, arch, model, dataset, seed, mae, r2}`` (the record format of analyze_ex_14) of the completed
    fingerprint archives of the prefix, and their split indices per (variant, dataset, seed). A newer archive of
    the same run (e.g. a retry) replaces an older one. Unreadable archives are reported and skipped.
    """
    archives = []
    for module in ('predict_molecules__fp', 'predict_molecules__sherlock'):
        for meta_path in glob.glob(os.path.join(RESULTS, module, '*', 'experiment_meta.json')):
            try:
                meta = json.load(open(meta_path))
            except Exception:
                continue
            params = {k: v.get('value') for k, v in meta.get('parameters', {}).items() if isinstance(v, dict)}
            # a killed or timed-out run keeps status 'running' (and has_error False), so require 'done'
            if params.get('__PREFIX__') != prefix or meta.get('status') != 'done' or meta.get('has_error'):
                continue
            archives.append((meta.get('start_time') or 0, module, params, os.path.dirname(meta_path)))

    records, splits = {}, {}
    for _, module, params, folder in sorted(archives, key=lambda a: a[0]):
        try:
            data = json.load(open(os.path.join(folder, 'experiment_data.json')))
        except Exception as error:
            print(f'skipping unreadable archive {folder}: {error}')
            continue
        metric = data.get('metrics', {}).get('test_neural_net2')
        if metric is None:
            continue
        variant = 'sherlock' if module == 'predict_molecules__sherlock' else params['FINGERPRINT_TYPE']
        key = (variant, params['NOTE'], params['SEED'])
        records[key] = {'variant': variant, 'arch': '', 'model': 'neural_net2', 'dataset': params['NOTE'],
                        'seed': params['SEED'], 'mae': metric['mae'], 'r2': metric['r2']}
        indices = data.get('indices', {})
        splits[key] = tuple(tuple(indices.get(part) or ()) for part in ('train', 'val', 'test'))
    return list(records.values()), splits


def collect_hdf(prefix: str):
    """The HDF + MLP records of the current ex_14 round of the prefix and their splits per (dataset, seed)."""
    splits14 = {}
    records = [r for r in analyze_ex_14.collect(prefix, splits14) if r['variant'] == 'hdf' and r['model'] == 'neural_net2']
    splits = {(dataset, seed): split for (variant, _, dataset, seed), split in splits14.items() if variant == 'hdf'}
    return [{k: r[k] for k in FIELDS} for r in records], splits


def check_splits(fp_splits: dict, hdf_splits: dict) -> set:
    """The (variant, dataset, seed) of the fingerprint runs whose split differs from that of their HDF run."""
    mismatches = set()
    for (variant, dataset, seed), split in sorted(fp_splits.items()):
        reference = hdf_splits.get((dataset, seed))
        if reference is None or not split[2] or split != reference:
            mismatches.add((variant, dataset, seed))
            reason = 'no HDF run' if reference is None else ('no recorded test indices' if not split[2] else 'differs')
            print(f'SPLIT MISMATCH: {variant} {dataset} seed {seed} ({reason})')
    return mismatches


def main(prefix: str, hdf_prefix: str, csv_path: str = None):
    os.makedirs(OUT, exist_ok=True)
    if csv_path:
        with open(csv_path) as f:
            records = [{**row, 'seed': int(row['seed']), 'mae': float(row['mae']),
                        'r2': float(row['r2']) if row['r2'] else None} for row in csv.DictReader(f)]
        print(f'loaded {len(records)} records from {csv_path}')
    else:
        fp_records, fp_splits = collect_fingerprints(prefix)
        hdf_records, hdf_splits = collect_hdf(hdf_prefix)
        print(f'collected {len(fp_records)} fingerprint records (prefix "{prefix}") and {len(hdf_records)} HDF records '
              f'(prefix "{hdf_prefix}")')
        mismatches = check_splits(fp_splits, hdf_splits)
        print(f'split check: {len(mismatches)} fingerprint runs without the split of their HDF run'
              + (' -- EXCLUDED from the results' if mismatches else ''))
        fp_records = [r for r in fp_records if (r['variant'], r['dataset'], r['seed']) not in mismatches]
        # HDF only for the (dataset, seed) pairs that have fingerprint runs, so that a partial run is not compared
        # with the full HDF set
        pairs = {(r['dataset'], r['seed']) for r in fp_records}
        records = [r for r in hdf_records if (r['dataset'], r['seed']) in pairs] + fp_records
        with open(os.path.join(OUT, f'results_{prefix}.csv'), 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            for r in sorted(records, key=lambda r: (r['dataset'], r['variant'], r['seed'])):
                writer.writerow(r)
    if not records:
        return

    # analyze_ex_14.table leaves out every target for which a column has no results at all; say so
    for dataset in sorted({r['dataset'] for r in records}):
        missing = [label for variant, label in REPRESENTATIONS
                   if not any(r['dataset'] == dataset and r['variant'] == variant for r in records)]
        if missing:
            print(f'WARNING: {dataset} is left out of the table, no results for: {", ".join(missing)}')

    columns = [(label, variant, '', 'neural_net2') for variant, label in REPRESENTATIONS]
    header =['Dataset & Quantity & ' + ' & '.join(label for _, label in REPRESENTATIONS) + ' \\\\']
    note = ('generated by analyze_ex_19.py; test MAE of the MLP, mean (std below) over the seeds; bold = best mean '
            'per target; triangle down/up = lower/higher MAE than HDF, filled if significant (paired Wilcoxon, '
            'Holm-corrected within each target, p < 0.05), hollow otherwise')
    content = table(records, columns, header, note)
    name = f'fp_comparison_mlp_{prefix}.tex'
    with open(os.path.join(OUT, name), 'w') as f:
        f.write(content)
    print(f'--- {name}\n{content}')

    print('mean test MAE (raw units) per target:')
    for dataset in analyze_ex_14.DATASET_ORDER:
        cells = []
        for variant, label in REPRESENTATIONS:
            values = [r['mae'] for r in records if r['dataset'] == dataset and r['variant'] == variant]
            cells.append(f'{label} {np.mean(values):.4g} (n={len(values)})' if values else f'{label} --')
        if any('(n=' in c for c in cells):
            print(f'  {dataset:20s} ' + '  '.join(cells))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analysis of ex_19 (HDF vs. modern fingerprints).')
    parser.add_argument('prefix', nargs='?', default='ex_19_fp')
    parser.add_argument('--hdf-prefix', default='ex_14_gnn')
    parser.add_argument('--csv', default=None, help='rebuild the table from this results CSV instead of the archives')
    args = parser.parse_args()
    main(args.prefix, args.hdf_prefix, args.csv)
