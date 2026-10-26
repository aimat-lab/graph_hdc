"""
Analysis for Experiment 26 (Crippen's logP as a further global attribute of HDF). See ``_slurm_ex_26.py``.

Pairs every run of the given prefix (GRAPH_CLOGP = True, arm "hdf_clogp") with the HDF run of ex_14 round 4 (prefix
ex_14_gnn, arm "hdf") of the same target (NOTE) and seed. Both must use the ex_14 configuration (D = 2048, L = 2,
continuous encoding, the current ex_14 round: bidirectional, total H, unit-modulus codebooks; the ex_14 MLP) and the
same split (train, validation and test indices, in order); pairs with another split are printed and left out. Of
several runs of an (arm, target, seed), the newest is used.

Per target and model (MLP, KNN): mean and standard deviation of the test MAE of both arms, the mean relative change
(clogp - hdf) / hdf over the seeds, the number of seeds on which logP lowers the MAE, and a two-sided Wilcoxon
signed-rank test, paired by seed, with the p-values also Holm-corrected over the nine targets.

Writes ``_ex26/results_<prefix>.csv`` (one row per arm, target, seed and model) and ``_ex26/summary_<prefix>.md``.

    python analyze_ex_26.py [prefix] [--ex14-results DIR]
        prefix          default ex_26_clogp (ex_26_smoke for the smoke test)
        --ex14-results  results folder that holds the ex_14 round 4 archives (default: this folder's results; on
                        KCIST ~/Programming/graph_hdc_ex14/graph_hdc/experiments/fingerprints/results)
"""
import os
import csv
import json
import glob
import argparse

import numpy as np
from scipy.stats import wilcoxon

from _slurm_ex_14 import DATASETS, DOWNSTREAM, is_current

PATH = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(PATH, '_ex26')
HDF_PREFIX = 'ex_14_gnn'
MODELS = [('neural_net2', 'MLP'), ('k_neighbors', 'KNN')]
NOTES = [ds[0] for ds in DATASETS]


def holm(pvalues: list) -> list:
    """Holm-corrected p-values (same order as the input; None stays None)."""
    present = sorted((p, i) for i, p in enumerate(pvalues) if p is not None)
    out, running = [None] * len(pvalues), 0.0
    for rank, (p, i) in enumerate(present):
        running = max(running, min(1.0, (len(present) - rank) * p))
        out[i] = running
    return out


def arm_of(params: dict, prefix: str):
    """'hdf_clogp' / 'hdf' for runs of the ex_14 round 4 HDF configuration, otherwise None."""
    if not is_current('hdc', params):
        return None
    hidden = params.get('NN_HIDDEN_LAYER_SIZES')
    if not (params.get('EMBEDDING_SIZE') == 2048 and params.get('NUM_LAYERS') == 2
            and params.get('ENCODING_MODE') == 'continuous' and params.get('GRAPH_ATTRIBUTES', True) is True
            and params.get('QUANTIZE_BITS') is None and params.get('NUM_DATA') == 1.0
            and params.get('NUM_TEST') == 0.1 and params.get('NN_SEED') is None
            and hidden is not None and tuple(hidden) == tuple(DOWNSTREAM['NN_HIDDEN_LAYER_SIZES'])
            and params.get('NN_LEARNING_RATE_INIT') == DOWNSTREAM['NN_LEARNING_RATE_INIT']):
        return None
    if params.get('__PREFIX__') == prefix and params.get('GRAPH_CLOGP') is True:
        return 'hdf_clogp'
    # the control of the smoke test: the ex_14 command re-run under the ex_26 prefix, without logP
    if params.get('__PREFIX__') == prefix and params.get('GRAPH_CLOGP') is False:
        return 'control'
    if params.get('__PREFIX__') == HDF_PREFIX and not params.get('GRAPH_CLOGP'):
        return 'hdf'
    return None


def collect(prefix: str, folders: list) -> dict:
    """Newest run per (arm, target, seed): {key: (start, record, split)} with record {model: (mae, r2)}."""
    runs = {}
    for folder in folders:
        for meta_path in glob.glob(os.path.join(folder, 'predict_molecules__hdc', '*', 'experiment_meta.json')):
            try:
                meta = json.load(open(meta_path))
            except Exception:
                continue
            params = {k: v.get('value') for k, v in meta.get('parameters', {}).items() if isinstance(v, dict)}
            # a killed or timed-out run keeps status 'running', so require 'done'
            if meta.get('status') != 'done' or meta.get('has_error'):
                continue
            arm = arm_of(params, prefix)
            if arm is None or params.get('NOTE') not in NOTES:
                continue
            try:
                data = json.load(open(os.path.join(os.path.dirname(meta_path), 'experiment_data.json')))
            except Exception as error:
                print(f'skipping unreadable archive {os.path.dirname(meta_path)}: {error}')
                continue
            metrics = data.get('metrics', {})
            record = {model: (metrics[f'test_{model}']['mae'], metrics[f'test_{model}']['r2'])
                      for model, _ in MODELS if f'test_{model}' in metrics}
            indices = data.get('indices', {})
            split = tuple(tuple(indices.get(part) or ()) for part in ('train', 'val', 'test'))
            key = (arm, params['NOTE'], params['SEED'])
            start = meta.get('start_time') or 0
            if key not in runs or start > runs[key][0]:
                runs[key] = (start, record, split)
    return runs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('prefix', nargs='?', default='ex_26_clogp')
    parser.add_argument('--ex14-results', default=os.path.join(PATH, 'results'))
    args = parser.parse_args()
    folders = sorted({os.path.realpath(os.path.join(PATH, 'results')), os.path.realpath(args.ex14_results)})
    runs = collect(args.prefix, folders)
    print(f'{sum(k[0] == "hdf_clogp" for k in runs)} logP runs ({args.prefix}), '
          f'{sum(k[0] == "hdf" for k in runs)} HDF runs ({HDF_PREFIX}) in {folders}')

    # control (smoke test): the ex_14 command re-run without logP must reproduce the ex_14 archive
    for arm, note, seed in sorted(k for k in runs if k[0] == 'control'):
        _, control, split_c = runs[('control', note, seed)]
        if ('hdf', note, seed) not in runs:
            print(f'control {note} seed {seed}: no ex_14 run to compare with')
            continue
        _, hdf, split_h = runs[('hdf', note, seed)]
        diffs = {label: control[model][0] - hdf[model][0] for model, label in MODELS
                 if model in control and model in hdf}
        print(f'control {note} seed {seed}: split {"identical" if split_c == split_h else "DIFFERENT"}, MAE minus '
              f'ex_14: ' + ', '.join(f'{label} {d:+.3g}' for label, d in diffs.items()))

    rows, lines, mismatches = [], [], []
    summary = {model: [] for model, _ in MODELS}
    for note in NOTES:
        seeds = sorted(seed for arm, n, seed in runs if arm == 'hdf_clogp' and n == note)
        if 0 < len(seeds) < 10:
            print(f'{note}: logP runs missing for seeds {sorted(set(range(10)) - set(seeds))}')
        pairs = []
        for seed in seeds:
            if ('hdf', note, seed) not in runs:
                print(f'{note} seed {seed}: no ex_14 run')
                continue
            _, clogp, split_c = runs[('hdf_clogp', note, seed)]
            _, hdf, split_h = runs[('hdf', note, seed)]
            if split_c != split_h:
                mismatches.append((note, seed))
                print(f'!!! {note} seed {seed}: the logP run used another split than ex_14, left out')
                continue
            pairs.append((seed, hdf, clogp))
            for arm, record in (('hdf', hdf), ('hdf_clogp', clogp)):
                for model, (mae, r2) in record.items():
                    rows.append({'arm': arm, 'target': note, 'seed': seed, 'model': model, 'mae': mae, 'r2': r2})
        for model, label in MODELS:
            a = np.array([hdf[model][0] for _, hdf, clogp in pairs if model in hdf and model in clogp])
            b = np.array([clogp[model][0] for _, hdf, clogp in pairs if model in hdf and model in clogp])
            if len(a) == 0:
                continue
            rel = (b - a) / a
            p = wilcoxon(b, a).pvalue if len(a) >= 2 and np.any(a != b) else None
            summary[model].append({'target': note, 'n': len(a), 'hdf': (a.mean(), a.std()),
                                   'clogp': (b.mean(), b.std()), 'rel': rel.mean(), 'better': int((b < a).sum()),
                                   'p': p})
    for model, label in MODELS:
        entries = summary[model]
        for entry, p_holm in zip(entries, holm([e['p'] for e in entries])):
            entry['p_holm'] = p_holm
        lines += [f'## {label}', '',
                  '| target | n | HDF MAE | HDF + logP MAE | change | seeds better | Wilcoxon p | Holm p (9 targets) |',
                  '|---|---|---|---|---|---|---|---|']
        for e in entries:
            fmt = lambda p: '-' if p is None else f'{p:.3g}'
            lines.append(f"| {e['target']} | {e['n']} | {e['hdf'][0]:.4g} ± {e['hdf'][1]:.2g} | "
                         f"{e['clogp'][0]:.4g} ± {e['clogp'][1]:.2g} | {100 * e['rel']:+.1f}% | "
                         f"{e['better']} of {e['n']} | {fmt(e['p'])} | {fmt(e['p_holm'])} |")
        lines.append('')
    lines.append(f'Pairs left out because of a different split: {mismatches or "none"}')

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f'results_{args.prefix}.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['arm', 'target', 'seed', 'model', 'mae', 'r2'])
        writer.writeheader()
        writer.writerows(rows)
    text = '\n'.join([f'# ex_26: HDF with logP as a further global attribute ({args.prefix})', ''] + lines)
    with open(os.path.join(OUT, f'summary_{args.prefix}.md'), 'w') as f:
        f.write(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
