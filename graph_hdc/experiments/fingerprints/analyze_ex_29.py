"""
Analysis of Experiment 29 (see _slurm_ex_29.py): the data of Figure 4, Figure 3b and SI Figure S1, re-run with the
standard HDF encoder.

    python analyze_ex_29.py                                   # the ex_29 prefixes, output _ex27/ex29/
    python analyze_ex_29.py <prefix_size> <prefix_knn> <prefix_train> <output subfolder of _ex27>   (smoke test)

Writes to the output folder (the plots are made locally from these files):

* fig4_runs.csv:  target, method, depth, dimension, seed, mae, r2 (MLP of Table 1)
* fig3b_runs.csv: target, distance (native / euclidean), method, dimension, seed, mae (KNN)
* s1_runs.csv:    target, method, depth, num_train, seed, mae, r2 (scikit-learn MLP)
* ex29_report.txt: per figure the completed runs, the configurations of the expected grid (_slurm_ex_29) without a
  result and the configurations with more than one archive (the latest is used), results without a finite MAE, the
  encoder check (every HDF run with the standard encoder and a new codebook per seed) and the split check (all runs of
  a target and seed share their test molecules; S1: all methods, depths and training sizes of a seed)
"""
import os
import csv
import sys
import math
from collections import defaultdict

from analyze_ex_27 import load_runs, ok
from _slurm_ex_27 import OUT, GROUP, TARGET_INDEX, SEEDS, method_of_module
import _slurm_ex_29 as s29

STANDARD = {'SPECTRUM': 'unit', 'BIDIRECTIONAL': True, 'HYDROGEN_COUNT': 'total', 'GRAPH_ATTRIBUTES': True,
            'ENCODING_MODE': 'continuous', 'CODEBOOK_SEED': None}


def representation(run: dict) -> tuple:
    """(method, dimension, depth) of a run."""
    p = run['params']
    if run['module'] == 'hdc':
        return 'hdf', p['EMBEDDING_SIZE'], p['NUM_LAYERS']
    return method_of_module('fp', p), p['FINGERPRINT_SIZE'], p['FINGERPRINT_RADIUS']


def metric(run: dict, model: str, suffix: str = '') -> dict:
    return run['data'].get('metrics', {}).get(f'test_{model}{suffix}', {})


def test_split(run: dict) -> tuple:
    return tuple(sorted(run['data'].get('indices', {}).get('test', [])))


def latest(keyed: list) -> tuple:
    """keyed: [(key, run, row)]. The row of the latest run per key, and the keys with more than one run."""
    best, count = {}, defaultdict(int)
    for key, run, row in keyed:
        count[key] += 1
        start = run['meta'].get('start_time', 0)
        if key not in best or start > best[key][0]:
            best[key] = (start, row)
    return {key: row for key, (_, row) in best.items()}, sorted(k for k, n in count.items() if n > 1)


def encoder_problems(runs: list) -> list:
    """The HDF runs whose encoder settings are not the standard ones with a new codebook per seed."""
    return [run['path'] for run in runs if run['module'] == 'hdc'
            and any(run['params'].get(k) != v for k, v in STANDARD.items())]


def split_problems(runs: list, key) -> list:
    """The groups of runs (by key(run)) whose test splits differ or are not stored."""
    splits = defaultdict(set)
    for run in runs:
        splits[key(run)].add(test_split(run))
    return [k for k, s in splits.items() if len(s) != 1 or not next(iter(s))]


def finite(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def write(folder: str, name: str, header: list, rows: list):
    with open(os.path.join(folder, name), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def summary(name: str, runs: list, rows: dict, duplicates: list, expected: set, mae_column: int) -> list:
    missing = sorted(expected - set(rows)) if expected else []
    lines = [f'{name}: {len(runs)} completed runs, {len(rows)} results'
             + (f' of {len(expected)} expected' if expected else '')]
    lines += [f'   missing ({len(missing)}): {missing[:12]}{" ..." if len(missing) > 12 else ""}'] if missing else []
    lines += [f'   more than one archive ({len(duplicates)}, the latest is used): {duplicates[:6]}'] if duplicates else []
    bad = [key for key, row in rows.items() if not finite(row[mae_column])]
    lines += [f'   results without a finite MAE ({len(bad)}): {bad[:6]}'] if bad else []
    lines += [f'   encoder: {encoder_problems(runs) or "all HDF runs standard, new codebook per seed"}']
    return lines


def main(prefix_size: str, prefix_knn: str, prefix_train: str, folder: str, grid: bool):
    os.makedirs(folder, exist_ok=True)
    lines = []

    # --- Figure 4: MLP over the dimension; key (target, method, depth, dimension, seed)
    runs = [run for run in load_runs(prefix_size, any_config=True) if ok(run)]
    keyed = []
    for run in runs:
        method, dimension, depth = representation(run)
        m, p = metric(run, 'neural_net2'), run['params']
        key = (p['NOTE'], method, depth, dimension, p['SEED'])
        keyed.append((key, run, [*key, m.get('mae'), m.get('r2')]))
    rows, duplicates = latest(keyed)
    write(folder, 'fig4_runs.csv', ['target', 'method', 'depth', 'dimension', 'seed', 'mae', 'r2'],
          [rows[k] for k in sorted(rows)])
    expected = {(note, method, depth, dim, seed) for note in s29.SIZE_NOTES for method in s29.METHODS_29
                for depth in s29.SIZE_DEPTHS for dim in s29.SIZE_DIMENSIONS for seed in SEEDS} if grid else set()
    lines += summary(f'Figure 4 ({prefix_size})', runs, rows, duplicates, expected, 5)
    lines += [f'   splits (target, seed; both methods, all dimensions and depths): '
              f'{split_problems(runs, lambda r: (r["params"]["NOTE"], r["params"]["SEED"])) or "identical"}']

    # --- Figure 3b: KNN over the dimension, both distances; key (target, distance, method, dimension, seed)
    runs = [run for run in load_runs(prefix_knn, any_config=True) if ok(run)]
    keyed = []
    for run in runs:
        method, dimension, _depth = representation(run)
        p = run['params']
        distance = 'euclidean' if p['KN_METRIC'] in ('minkowski', 'euclidean') else 'native'
        key = (p['NOTE'], distance, method, dimension, p['SEED'])
        keyed.append((key, run, [*key, metric(run, 'k_neighbors').get('mae')]))
    rows, duplicates = latest(keyed)
    write(folder, 'fig3b_runs.csv', ['target', 'distance', 'method', 'dimension', 'seed', 'mae'],
          [rows[k] for k in sorted(rows)])
    expected = {(note, variant, method, dim, seed) for variant, metrics in s29.KNN_VARIANTS.items()
                for note in s29.KNN_NOTES for method in metrics for dim in s29.KNN_DIMENSIONS
                for seed in SEEDS} if grid else set()
    lines += summary(f'Figure 3b ({prefix_knn})', runs, rows, duplicates, expected, 5)
    lines += [f'   splits (target, seed; both methods, distances and dimensions): '
              f'{split_problems(runs, lambda r: (r["params"]["NOTE"], r["params"]["SEED"])) or "identical"}']

    # --- SI Figure S1: scikit-learn MLP over the number of training molecules, the targets of a run (QM9: four)
    runs = [run for run in load_runs(prefix_train, any_config=True) if ok(run)]
    keyed = []
    for run in runs:
        method, _dimension, depth = representation(run)
        p = run['params']
        # the targets of the run: those of its group (NOTE) that are in its TARGET_INDICES
        for note in [n for n in GROUP[p['NOTE']][1] if TARGET_INDEX[n] in p['TARGET_INDICES']]:
            m = metric(run, 'neural_net', f'__t{TARGET_INDEX[note]}')
            key = (note, method, depth, p['NUM_TRAIN'], p['SEED'])
            keyed.append((key, run, [*key, m.get('mae'), m.get('r2')]))
    rows, duplicates = latest(keyed)
    write(folder, 's1_runs.csv', ['target', 'method', 'depth', 'num_train', 'seed', 'mae', 'r2'],
          [rows[k] for k in sorted(rows)])
    expected = {(note, method, depth, n, seed) for note in s29.TRAIN_NOTES for method in s29.METHODS_29
                for depth in s29.TRAIN_DEPTHS for n in s29.TRAIN_SIZES for seed in SEEDS} if grid else set()
    lines += summary(f'SI Figure S1 ({prefix_train})', runs, rows, duplicates, expected, 5)
    lines += [f'   test splits (seed; both methods, depths and training sizes): '
              f'{split_problems(runs, lambda r: r["params"]["SEED"]) or "identical"}']

    with open(os.path.join(folder, 'ex29_report.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    if len(sys.argv) >= 5:
        main(*sys.argv[1:4], os.path.join(OUT, sys.argv[4]), grid=False)
    else:
        main(s29.PREFIX_SIZE, s29.PREFIX_KNN, s29.PREFIX_TRAIN, os.path.join(OUT, 'ex29'), grid=True)
