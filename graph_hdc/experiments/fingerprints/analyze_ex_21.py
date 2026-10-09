"""
Analysis of Experiment 21 (ex_21): HDF without and with the optional stereo attribute.

Collects the archives of the three parts of ``_slurm_ex_21.py`` and writes ``_ex21/summary_<prefix>.md``:

* collisions (ZINC250k, stereoisomers counted as distinct molecules): fraction of molecules whose HDF vector is
  identical to that of another molecule, and how many of them differ only in their stereochemistry;
* BACE pIC50 (ex_14 configuration, 10 seeds): test MAE / RMSE / R2 of MLP and KNN, the paired difference between
  the arms of a seed, and as a reference the difference that another MLP seed alone causes (re-seeded run without
  stereo);
* enantiomer docking ranking (Adams et al., 3 seeds): test ranking accuracy and MAE of MLP and KNN, next to the values
  that Adams et al. report for their models.

    python analyze_ex_21.py [prefix]      # default prefix: ex_21
"""
import os
import sys
import json
import glob
from collections import defaultdict

import numpy as np
from scipy.stats import wilcoxon

PATH = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(PATH, 'results')
OUT = os.path.join(PATH, '_ex21')
MODELS = {'neural_net2': 'MLP', 'k_neighbors': 'KNN'}

# Table 2 of Adams et al. (ICLR 2022): enantiomer ranking accuracy (%), mean and standard deviation over three trials
CHIRO_REFERENCE = [
    ('ChIRo', 72.0, 0.5),
    ('SphereNet', 68.6, 0.3),
    ('DMPNN with chiral tags', 65.9, 0.6),
    ('Tetra-DMPNN (concatenate) with chiral tags', 70.1, 0.5),
    ('2D model without stereo information (ablation, Table 3)', 49.6, 0.3),
]


def archives(module: str, prefix: str) -> list:
    """(parameters, data) of every finished archive of ``module`` with the given prefix, oldest first."""
    found = []
    for meta_path in glob.glob(os.path.join(RESULTS, module, '*', 'experiment_meta.json')):
        meta = json.load(open(meta_path))
        params = {k: v.get('value') for k, v in meta.get('parameters', {}).items() if isinstance(v, dict)}
        if params.get('__PREFIX__') != prefix or meta.get('status') != 'done' or meta.get('has_error'):
            continue
        data = json.load(open(os.path.join(os.path.dirname(meta_path), 'experiment_data.json')))
        found.append((meta.get('start_time') or 0, params, data))
    return [(params, data) for _, params, data in sorted(found, key=lambda t: t[0])]


def mean_sd(values: list, scale: float = 1.0, digits: int = 3) -> str:
    if not values:
        return '-'
    values = np.asarray(values, dtype=float) * scale
    sd = values.std(ddof=1) if len(values) > 1 else 0.0
    return f'{values.mean():.{digits}f} ± {sd:.{digits}f} (n={len(values)})'


def collisions(prefix: str) -> str:
    rows = {}
    for params, data in archives('molecule_collisions', f'{prefix}_collisions'):
        for dim, record in data.get('hdf', {}).items():
            rows[(params['DATASET_NAME'], int(dim), bool(params['STEREO']))] = (params, data, record)
    lines = ['## Collisions (stereoisomers counted as distinct molecules)\n',
             'The Morgan fingerprints are the unchanged ones of the paper (without chirality), so every pair of '
             'molecules that differ only in their stereochemistry collides for them by construction.\n',
             '| dataset | D | HDF | molecules | differ only in stereo | molecules in HDF collisions | of which stereo-only '
             '| HDF collisions split by 1-WL | noise max | Morgan (unchanged) |',
             '|---|---|---|---|---|---|---|---|---|---|']
    for (dataset, dim, stereo), (params, data, record) in sorted(rows.items()):
        hdf = record['collisions']
        morgan = data.get('morgan', {}).get(str(dim), {}).get('collisions', {})
        lines.append(
            f'| {dataset} | {dim} | {"with stereo" if stereo else "without stereo"} '
            f'| {data["dataset"]["num_molecules"]} | {data["dataset"]["stereo_groups"]["num_in_collisions"]} '
            f'| {hdf["num_in_collisions"]} ({hdf["collision_rate"]:.4%}) | {record["num_in_stereo_only_collisions"]} '
            f'| {record["num_in_hdf_collisions_split_by_wl"]} | {record["noise_max"]:.1e} '
            f'| {morgan.get("collision_rate", float("nan")):.4%} |')
    return '\n'.join(lines) + '\n'


def paired_table(runs: dict, metrics: list, arm: str = 'stereo', reference: str = 'plain') -> list:
    """runs: {(seed, arm): data}; one row per (model, metric) with both arms and their paired difference over the
    seeds that have both."""
    lines = [f'| model | metric | {reference} | {arm} | difference ({arm} - {reference}) | seeds better ({arm}) '
             f'| Wilcoxon p |', '|---|---|---|---|---|---|---|']
    seeds = sorted({seed for seed, a in runs if a == arm and (seed, reference) in runs})
    for model, label in MODELS.items():
        for metric, scale, better in metrics:
            get = lambda seed, a: runs[(seed, a)].get('metrics', {}).get(f'test_{model}', {}).get(metric)
            both = [s for s in seeds if get(s, reference) is not None and get(s, arm) is not None]
            if not both:
                continue
            first, second = [get(s, reference) for s in both], [get(s, arm) for s in both]
            diff = [b - a for a, b in zip(first, second)]
            wins = sum((d < 0) if better == 'lower' else (d > 0) for d in diff)
            # two-sided Wilcoxon signed-rank test paired by repetition (as in the GNN comparison of the SI); with
            # fewer than 5 repetitions no test can reach p < 0.05
            p = (f'{wilcoxon(second, first).pvalue:.3g}' if len(diff) >= 5 and any(d != 0 for d in diff)
                 else '-')
            lines.append(f'| {label} | {metric} | {mean_sd(first, scale)} | {mean_sd(second, scale)} '
                         f'| {mean_sd(diff, scale)} | {wins} of {len(diff)} | {p} |')
    return lines


def arm(params: dict) -> str:
    if params.get('STEREO'):
        return 'stereo'
    return 'plain' if params.get('NN_SEED') is None else 'plain_reseeded'


def with_rmse(data: dict) -> dict:
    for key, values in data.get('metrics', {}).items():
        if 'mse' in values:
            values['rmse'] = float(np.sqrt(values['mse']))
    return data


def bace(prefix: str) -> str:
    runs = {(p['SEED'], arm(p)): with_rmse(d) for p, d in archives('predict_molecules__hdc', f'{prefix}_bace')}
    metrics = [('mae', 1.0, 'lower'), ('rmse', 1.0, 'lower'), ('r2', 1.0, 'higher')]
    lines = ['## BACE pIC50 (ex_14 configuration: D = 2048, L = 2)\n']
    lines += paired_table(runs, metrics)
    lines += ['\nReference: the same representation without stereo, trained with another network seed (NN_SEED = '
              '1000 + seed). Only the MLP depends on the network seed; KNN rows are identical by construction.\n']
    lines += paired_table(runs, metrics, arm='plain_reseeded')
    return '\n'.join(lines) + '\n'


def chiro(prefix: str) -> str:
    runs = {(p['SEED'], arm(p)): with_rmse(d) for p, d in archives('predict_enantiomers__hdc', f'{prefix}_chiro')}
    lines = ['## Enantiomer docking ranking (Adams et al., original splits)\n']
    lines += paired_table(runs, [('ranking_accuracy', 100.0, 'higher'), ('mae', 1.0, 'lower')])
    ties = defaultdict(list)
    for (seed, a), data in runs.items():
        for model in MODELS:
            record = data.get('metrics', {}).get(f'test_{model}', {})
            if 'num_pairs' in record:
                ties[(model, a)].append(f'{record["num_tied_pairs"]}/{record["num_pairs"]}')
    lines.append('\nTied test pairs (predictions closer than TIE_TOLERANCE): '
                 + '; '.join(f'{MODELS[m]} {a}: {", ".join(v)}' for (m, a), v in sorted(ties.items())))
    lines.append('\nReference (Adams et al., Table 2 and 3), ranking accuracy in %. Their protocol differs from ours: '
                 'trained graph neural networks, the epoch selected by the validation ranking accuracy, both '
                 'enantiomers of a pair in every minibatch, and predictions averaged over the conformers.\n')
    lines += [f'- {name}: {value:.1f} ± {sd:.1f}' for name, value, sd in CHIRO_REFERENCE]
    return '\n'.join(lines) + '\n'


def main(prefix: str):
    text = '\n'.join([f'# ex_21: HDF without and with stereo descriptors (prefix {prefix})\n',
                      collisions(prefix), bace(prefix), chiro(prefix)])
    print(text)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f'summary_{prefix}.md'), 'w') as f:
        f.write(text)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'ex_21')
