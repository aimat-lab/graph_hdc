"""
Analysis of Experiment 28 (see _slurm_ex_28.py): the data of Figure 2 of the main text, the model comparison.

    python analyze_ex_28.py

Collects the test R2 and MAE of every bar of the figure, i.e. per target (NOTES_28), method (HDF, Morgan, RDKit),
model and setting (fixed / tuned), over the seeds 0-9, from the archives on this machine:

    model   fixed                                   tuned
    NN      ex_27_fixed (neural_net2)               ex_27_opt, best configuration of the ex_27 study
    KNN     ex_27_fixed (k_neighbors)               ex_28_opt, best configuration of the ex_28 study
    GB      ex_28_fixed (grad_boost)                ex_28_opt
    RF      ex_28_fixed (random_forest)             ex_28_opt

Only runs that are done without error count, and for the tuned setting only runs with the best configuration of their
study (the latest per seed). Writes to _ex27/ex28/:

* fig2_runs.csv: target, method, model, setting, seed, r2, mae
* fig2_summary.csv: the mean, standard deviation and number of seeds of R2 and MAE per bar (only once every bar has all
  seeds; a stale file is removed otherwise)
* ex28_configs.csv: the best trial of every ex_28 study (validation MAE and parameters)
* fig2_report.txt: counts, missing bars, a compact table of the mean R2 and the check that every ex_28 (and ex_27_opt)
  run used the test split of the ex_27 fixed runs of its dataset and seed
"""
import os
import csv

import numpy as np

from analyze_ex_27 import load_runs, ok
from _slurm_ex_27 import (OUT, GROUP, TARGET_INDEX, SEEDS, PREFIX_FIXED as PREFIX_27_FIXED,
                          PREFIX_OPT as PREFIX_27_OPT, best_params as best_params_27, matches_params)
from _slurm_ex_28 import (PREFIX_FIXED, PREFIX_OPT, METHODS_28, NOTES_28, MODELS_28, FIXED, best_params, best_trial)

OUT_28 = os.path.join(OUT, 'ex28')
LABEL = {'grad_boost': 'GB', 'k_neighbors': 'KNN', 'random_forest': 'RF', 'neural_net2': 'NN'}
MODELS = ['GB', 'KNN', 'RF', 'NN']   # the order of the groups in Figure 2


def run_metrics(run: dict):
    """(note, model label, r2, mae) for every target and model of a run."""
    params, metrics = run['params'], run['data'].get('metrics', {})
    if params.get('TARGET_INDICES'):
        pairs = [(note, f'__t{TARGET_INDEX[note]}') for note in GROUP[params['NOTE']][1]
                 if TARGET_INDEX[note] in params['TARGET_INDICES']]
    else:
        pairs = [(params['NOTE'], '')]
    for note, suffix in pairs:
        for model in params.get('MODELS', []):
            values = metrics.get(f'test_{model}{suffix}', {})
            yield note, LABEL.get(model, model), values.get('r2'), values.get('mae')


def collect(lines: list) -> dict:
    """{(note, method, model, setting, seed): (r2, mae)} of the latest qualifying run of every bar and seed."""
    say = lines.append
    out = {}

    def add(runs, setting, keep_models, best_of=None, required=None):
        used = 0
        for run in sorted(filter(ok, runs), key=lambda r: r['meta'].get('start_time', 0)):
            if run['method'] not in METHODS_28:
                continue
            if required is not None and not matches_params(run['params'], required):
                continue
            for note, model, r2, mae in run_metrics(run):
                if note not in NOTES_28 or model not in keep_models:
                    continue
                if best_of is not None and not matches_params(run['params'], best_of(model, run['method'], note)):
                    continue
                out[(note, run['method'], model, setting, run['params']['SEED'])] = (r2, mae)
                used += 1
        return used

    cache = {}

    def best_27(model, method, note):
        key = ('27', method, note)
        if key not in cache:
            try:
                cache[key] = best_params_27(method, note)
            except Exception as error:
                say(f'no best ex_27 configuration for {method} {note}: {error}')
                cache[key] = {'__unmatched__': True}
        return cache[key]

    inverse = {label: model for model, label in LABEL.items()}

    def best_28(model, method, note):
        key = ('28', model, method, note)
        if key not in cache:
            try:
                cache[key] = best_params(inverse[model], method, note)
            except Exception as error:
                say(f'no best ex_28 configuration for {model} {method} {note}: {error}')
                cache[key] = {'__unmatched__': True}
        return cache[key]

    say(f'{PREFIX_27_FIXED}: {add(load_runs(PREFIX_27_FIXED), "fixed", {"NN", "KNN"})} target results used')
    # (load_runs recognizes the standard representations; the model settings must be the fixed ones as well)
    say(f'{PREFIX_FIXED}: {add(load_runs(PREFIX_FIXED), "fixed", {"GB", "RF"}, required=FIXED)} target results used')
    say(f'{PREFIX_27_OPT}: {add(load_runs(PREFIX_27_OPT, any_config=True), "tuned", {"NN"}, best_27)} '
        f'target results used')
    say(f'{PREFIX_OPT}: {add(load_runs(PREFIX_OPT, any_config=True), "tuned", {"GB", "RF", "KNN"}, best_28)} '
        f'target results used')
    return out


def check_splits(lines: list):
    """
    Whether the ex_28 runs used exactly the data splits of ex_27: the test indices of every ex_28 run (fixed and tuned,
    seeds 0-9) against those of the ex_27 fixed runs of the same dataset and seed (ClogP uses the AqSolDB molecules).
    """
    def test_split(run):
        return tuple(sorted(run['data'].get('indices', {}).get('test', [])))

    reference = {}
    for run in filter(ok, load_runs(PREFIX_27_FIXED)):
        reference.setdefault((run['params']['DATASET_NAME'], run['params']['SEED']), test_split(run))
    counts = {'same': 0, 'different': 0, 'no reference': 0}
    for prefix in (PREFIX_FIXED, PREFIX_OPT, PREFIX_27_OPT):
        for run in filter(ok, load_runs(prefix, any_config=prefix != PREFIX_FIXED)):
            key = (run['params']['DATASET_NAME'], run['params']['SEED'])
            if key not in reference:
                counts['no reference'] += 1
            elif test_split(run) == reference[key] and reference[key]:
                counts['same'] += 1
            else:
                counts['different'] += 1
                lines.append(f'   DIFFERENT test split: {prefix} {key}')
    lines.append(f'test splits against ex_27_fixed: {counts}')


def configs(lines: list):
    rows = []
    for model in MODELS_28:
        for method in METHODS_28:
            for note in NOTES_28:
                try:
                    trial = best_trial(model, method, note)
                    rows.append({'model': LABEL[model], 'method': method, 'target': note, 'val_mae': trial.value,
                                 'trial': trial.number, 'params': trial.user_attrs['params']})
                except Exception as error:
                    lines.append(f'study {model} {method} {note}: {error}')
    if rows:
        with open(os.path.join(OUT_28, 'ex28_configs.csv'), 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines.append(f'{len(rows)} of {len(MODELS_28) * len(METHODS_28) * len(NOTES_28)} studies complete')


def main():
    os.makedirs(OUT_28, exist_ok=True)
    lines = []
    results = collect(lines)
    with open(os.path.join(OUT_28, 'fig2_runs.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['target', 'method', 'model', 'setting', 'seed', 'r2', 'mae'])
        for key in sorted(results, key=str):
            writer.writerow([*key, *results[key]])

    summary, missing = [], []
    for note in NOTES_28:
        for method in METHODS_28:
            for model in MODELS:
                for setting in ('fixed', 'tuned'):
                    pairs = [results[(note, method, model, setting, s)] for s in SEEDS
                             if (note, method, model, setting, s) in results]
                    # a run without a metric (should not happen) counts as missing
                    pairs = [(r2, mae) for r2, mae in pairs if r2 is not None and mae is not None]
                    r2, mae = [p[0] for p in pairs], [p[1] for p in pairs]
                    if len(r2) < len(SEEDS):
                        missing.append(f'{note} {method} {model} {setting}: {len(r2)} of {len(SEEDS)} seeds')
                    summary.append({'target': note, 'method': method, 'model': model, 'setting': setting,
                                    'r2_mean': np.mean(r2) if r2 else np.nan,
                                    'r2_std': np.std(r2, ddof=1) if len(r2) > 1 else np.nan,
                                    'mae_mean': np.mean(mae) if mae else np.nan,
                                    'mae_std': np.std(mae, ddof=1) if len(mae) > 1 else np.nan, 'n': len(r2)})
    path = os.path.join(OUT_28, 'fig2_summary.csv')
    if missing:
        lines.append(f'{len(missing)} of {len(summary)} bars incomplete:')
        lines += [f'   {m}' for m in missing]
        if os.path.exists(path):
            os.remove(path)
    else:
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)
        lines.append(f'all {len(summary)} bars complete: {os.path.relpath(path, OUT)}')

    lines.append('\nmean test R2 (fixed / tuned):')
    for note in NOTES_28:
        for method in METHODS_28:
            cells = []
            for model in MODELS:
                pair = [next((s['r2_mean'] for s in summary if (s['target'], s['method'], s['model'], s['setting'])
                              == (note, method, model, setting)), np.nan) for setting in ('fixed', 'tuned')]
                cells.append(f'{model} {pair[0]:.3f}/{pair[1]:.3f}')
            lines.append(f'   {note:14s} {method:7s} ' + '  '.join(cells))
    configs(lines)
    check_splits(lines)
    with open(os.path.join(OUT_28, 'fig2_report.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
