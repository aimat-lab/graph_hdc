"""
Analysis for Experiment 18 (HDF vs. Morgan, SECFP (folded MHFP) and folded MAP4 with the MLP; reviewer
comment R2.1). See ``_slurm_ex_18.py`` for the setup.

Writes to ``_ex18/``:

* ``results_<prefix>.csv``               one row per (representation, dataset, seed) with the test MAE / R2
* ``fp_comparison_mlp_<prefix>.tex``     SI-style table, built with the table code of ``analyze_ex_14.py``:
                                         mean (std below), best mean bold, triangles vs. HDF (paired Wilcoxon
                                         over the seeds, Holm-corrected within each target)

It also checks that all representations of a (dataset, seed) used the same split (train, validation and test
indices, in order), which the paired tests rely on.

Usage:
    python analyze_ex_18.py                 # prefix ex_18_fp
    python analyze_ex_18.py ex_18_smoke     # e.g. the smoke run
"""
import os
import sys
import csv
import json
import glob
from collections import defaultdict

import numpy as np

from analyze_ex_14 import table

PATH = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(PATH, 'results')
OUT = os.path.join(PATH, '_ex18')

REPRESENTATIONS = [('hdf', 'HDF'), ('morgan', 'Morgan'), ('secfp', 'SECFP'), ('map4', 'MAP4')]


def collect(prefix: str):
    """
    Records ``{variant, arch, model, dataset, seed, mae, r2}`` (the record format of analyze_ex_14) from the
    completed archives of the prefix, plus the split indices per (variant, dataset, seed). A newer archive of
    the same run (e.g. a retry) replaces an older one. Unreadable archives are reported and skipped.
    """
    archives = []
    for module in ('predict_molecules__hdc', 'predict_molecules__fp'):
        for meta_path in glob.glob(os.path.join(RESULTS, module, '*', 'experiment_meta.json')):
            try:
                meta = json.load(open(meta_path))
            except Exception:
                continue
            params = {k: v.get('value') for k, v in meta.get('parameters', {}).items() if isinstance(v, dict)}
            # a killed or timed-out run keeps status 'running' (and has_error False), so require 'done'
            if params.get('__PREFIX__') != prefix or meta.get('status') != 'done' or meta.get('has_error'):
                continue
            if module == 'predict_molecules__hdc' and params.get('BIDIRECTIONAL') is not True:
                print(f'skipping one-directional HDF archive {os.path.dirname(meta_path)}')
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
        variant = 'hdf' if module == 'predict_molecules__hdc' else params['FINGERPRINT_TYPE']
        key = (variant, params['NOTE'], params['SEED'])
        records[key] = {'variant': variant, 'arch': '', 'model': 'neural_net2', 'dataset': params['NOTE'],
                        'seed': params['SEED'], 'mae': metric['mae'], 'r2': metric['r2']}
        indices = data.get('indices', {})
        # the order matters as well: it also drives the MLP's internal validation split
        splits[key] = tuple(tuple(indices.get(part) or ()) for part in ('train', 'val', 'test'))
    return list(records.values()), splits


def check_splits(splits: dict) -> int:
    """Number of (dataset, seed) pairs whose representations used different (or unrecorded) splits."""
    by_pair = defaultdict(dict)
    for (variant, dataset, seed), split in splits.items():
        by_pair[(dataset, seed)][variant] = split
    mismatches = 0
    for (dataset, seed), variants in sorted(by_pair.items()):
        missing = sorted(v for v, split in variants.items() if not split[2])
        if missing or len(set(variants.values())) > 1:
            mismatches += 1
            print(f'SPLIT MISMATCH: {dataset} seed {seed}: {sorted(variants)}'
                  + (f' (no recorded test indices: {missing})' if missing else ''))
    return mismatches


def main(prefix: str):
    os.makedirs(OUT, exist_ok=True)
    records, splits = collect(prefix)
    print(f'collected {len(records)} records for prefix "{prefix}"')
    if not records:
        return
    mismatches = check_splits(splits)
    print(f'split check: {mismatches} mismatching (dataset, seed) pairs')

    with open(os.path.join(OUT, f'results_{prefix}.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['variant', 'dataset', 'seed', 'mae', 'r2'])
        writer.writeheader()
        for r in sorted(records, key=lambda r: (r['dataset'], r['variant'], r['seed'])):
            writer.writerow({k: r[k] for k in writer.fieldnames})

    columns = [(label, variant, '', 'neural_net2') for variant, label in REPRESENTATIONS]
    header = ['Dataset & Quantity & ' + ' & '.join(label for _, label in REPRESENTATIONS) + ' \\\\']
    note = ('generated by analyze_ex_18.py; test MAE of the MLP, mean (std below) over the seeds; bold = best mean '
            'per target; triangle down/up = lower/higher MAE than HDF, filled if significant (paired Wilcoxon, '
            'Holm-corrected within each target, p < 0.05), hollow otherwise')
    content = table(records, columns, header, note)
    name = f'fp_comparison_mlp_{prefix}.tex'
    with open(os.path.join(OUT, name), 'w') as f:
        f.write(content)
    print(f'--- {name}\n{content}')

    # plain-text overview: mean MAE per target and representation
    print('mean test MAE (raw units) per target:')
    for dataset in sorted({r['dataset'] for r in records}):
        cells = []
        for variant, label in REPRESENTATIONS:
            values = [r['mae'] for r in records if r['dataset'] == dataset and r['variant'] == variant]
            cells.append(f'{label} {np.mean(values):.4g} (n={len(values)})' if values else f'{label} --')
        print(f'  {dataset:20s} ' + '  '.join(cells))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'ex_18_fp')
