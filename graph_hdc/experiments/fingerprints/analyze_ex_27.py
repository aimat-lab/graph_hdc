"""
Analysis of Experiment 27 (see _slurm_ex_27.py): the re-run of Table 1 and of the KNN table on JUPITER.

    python analyze_ex_27.py check     # the GPU equivalence check and the smoke test of the fixed stage

``check`` prints (and writes to _ex27/check_report.txt):

* the status of every run of the check and the smoke test, and the device the MLPs trained on, per variant
* per target, method and seed the test MAE of the MLP in the three variants: cpu (the data path of the earlier tables),
  gpu (the production configuration) and cpu_nnseed (the cpu variant with another network seed), and per target and
  method the mean absolute differences gpu - cpu and cpu_nnseed - cpu over the seeds: a GPU that changes nothing but
  the floating-point arithmetic gives differences no larger than those from a new network seed, without a sign
* the KNN test MAE on the CPU and on the GPU (the same exact search; only the choice among neighbors at the same
  distance could differ)
* the training times of the MLP and the run times, and the MAE of all seven QM9 targets of the gpu runs
* the MAE of every smoke run, per target and method
"""
import os
import sys
import glob
import json

import numpy as np

from _slurm_ex_27 import (PATH, OUT, PREFIX_CHECK, PREFIX_SMOKE, GROUP, TARGET_INDEX, METHODS, CHECK_TARGETS,
                          CHECK_METHODS, CHECK_SEEDS, SMOKE_GROUPS, method_of_params, check_variant, group_of_note)

VARIANTS = ['cpu', 'gpu', 'cpu_nnseed']


def load_runs(prefix: str) -> list:
    """Every archive of the prefix as a dict: module, method, params, meta, data, path."""
    runs = []
    for module in ('hdc', 'fp'):
        pattern = os.path.join(PATH, 'results', f'predict_molecules__{module}', f'{prefix}__*', 'experiment_meta.json')
        for meta_path in sorted(glob.glob(pattern)):
            try:
                meta = json.load(open(meta_path))
            except Exception:
                continue
            params = {k: v.get('value') for k, v in meta.get('parameters', {}).items() if isinstance(v, dict)}
            if params.get('__PREFIX__') != prefix:
                continue
            data_path = meta_path.replace('experiment_meta.json', 'experiment_data.json')
            try:
                data = json.load(open(data_path))
            except Exception:
                data = {}
            runs.append({'module': module, 'method': method_of_params(module, params), 'params': params,
                         'meta': meta, 'data': data, 'path': os.path.dirname(meta_path)})
    return runs


def ok(run: dict) -> bool:
    return run['meta'].get('status') == 'done' and not run['meta'].get('has_error')


def target_results(run: dict) -> dict:
    """{note: {'mlp', 'knn' (test MAE), 'mlp_time', 'knn_time' (s)}} of the target(s) of a run."""
    params, data = run['params'], run['data']
    metrics, fit = data.get('metrics', {}), data.get('fit_time', {})
    if params.get('TARGET_INDICES'):
        notes = GROUP[params['NOTE']][1]
        pairs = [(note, f'__t{TARGET_INDEX[note]}') for note in notes if TARGET_INDEX[note] in params['TARGET_INDICES']]
    else:
        pairs = [(params['NOTE'], '')]
    return {note: {'mlp': metrics.get(f'test_neural_net2{suffix}', {}).get('mae'),
                   'knn': metrics.get(f'test_k_neighbors{suffix}', {}).get('mae'),
                   'mlp_time': fit.get(f'neural_net2{suffix}'),
                   'knn_time': fit.get(f'k_neighbors{suffix}')}
            for note, suffix in pairs}


def fmt(value, digits: int = 4) -> str:
    return '-' if value is None else f'{value:.{digits}f}'


def report_check(lines: list):
    runs = load_runs(PREFIX_CHECK)
    smoke = load_runs(PREFIX_SMOKE)
    say = lines.append

    say('== status ==')
    for label, group in [('check', runs), ('smoke', smoke)]:
        failed = [run for run in group if not ok(run)]
        say(f'{label}: {len(group)} archives, {len(group) - len(failed)} done, {len(failed)} not done')
        for run in failed:
            say(f'   NOT DONE {os.path.basename(run["path"])}: status={run["meta"].get("status")} '
                f'error={run["meta"].get("error_type")}')
    devices = {}
    for run in runs:
        variant = check_variant(run['params'])
        devices.setdefault(variant, set()).add(run['data'].get('nn', {}).get('device'))
    devices['smoke'] = {run['data'].get('nn', {}).get('device') for run in smoke}
    say('MLP devices: ' + ', '.join(f'{v}: {sorted(map(str, d))}' for v, d in devices.items()))

    # results[(note, method, seed, variant)] = target results
    results = {}
    for run in filter(ok, runs):
        for note, values in target_results(run).items():
            results[(note, run['method'], run['params']['SEED'], check_variant(run['params']))] = values

    say('\n== MLP test MAE: cpu (earlier data path) / gpu (production) / cpu_nnseed (another network seed) ==')
    for note in CHECK_TARGETS:
        for method in CHECK_METHODS:
            say(f'{note} {method}')
            rows = []
            for seed in CHECK_SEEDS:
                values = [results.get((note, method, seed, v), {}).get('mlp') for v in VARIANTS]
                rows.append(values)
                say(f'   seed {seed}: ' + '  '.join(f'{v} {fmt(x)}' for v, x in zip(VARIANTS, values)))
            complete = [r for r in rows if None not in r]
            if complete:
                arr = np.array(complete)
                say(f'   mean:   ' + '  '.join(f'{v} {fmt(x)}' for v, x in zip(VARIANTS, arr.mean(axis=0))))
                gpu_diff = arr[:, 1] - arr[:, 0]
                seed_diff = arr[:, 2] - arr[:, 0]
                say(f'   gpu - cpu: mean {fmt(gpu_diff.mean())}, mean |.| {fmt(np.abs(gpu_diff).mean())}'
                    f' ({100 * np.abs(gpu_diff).mean() / arr[:, 0].mean():.1f}% of the cpu MAE)'
                    f' | nnseed - cpu: mean {fmt(seed_diff.mean())}, mean |.| {fmt(np.abs(seed_diff).mean())}'
                    f' ({100 * np.abs(seed_diff).mean() / arr[:, 0].mean():.1f}%) [{len(complete)} seeds]')

    say('\n== KNN test MAE: torch backend on the CPU / on the GPU ==')
    for note in CHECK_TARGETS:
        for method in CHECK_METHODS:
            pairs = [(results.get((note, method, seed, 'cpu'), {}).get('knn'),
                      results.get((note, method, seed, 'gpu'), {}).get('knn')) for seed in CHECK_SEEDS]
            say(f'{note:14s} {method:7s} ' + '  '.join(f'{fmt(c)}/{fmt(g)}' for c, g in pairs)
                + ('   max |diff| ' + fmt(max(abs(c - g) for c, g in pairs if None not in (c, g)), 8)
                   if any(None not in p for p in pairs) else ''))

    say('\n== MLP training time (s, mean over seeds): cpu / gpu ==')
    for note in CHECK_TARGETS:
        for method in CHECK_METHODS:
            times = {v: [results.get((note, method, seed, v), {}).get('mlp_time') for seed in CHECK_SEEDS]
                     for v in ('cpu', 'gpu')}
            means = {v: np.mean([t for t in ts if t is not None]) if any(t is not None for t in ts) else None
                     for v, ts in times.items()}
            say(f'{note:14s} {method:7s} cpu {fmt(means["cpu"], 1)}  gpu {fmt(means["gpu"], 1)}')
    say('run durations (s, mean over seeds; the gpu runs of QM9 train all seven targets):')
    for note in CHECK_TARGETS:
        group = group_of_note(note)
        for method in CHECK_METHODS:
            durations = {}
            for run in filter(ok, runs):
                params = run['params']
                if run['method'] != method or params['NOTE'] not in (note, group):
                    continue
                durations.setdefault(check_variant(params), []).append(run['meta'].get('duration', np.nan))
            say(f'{note:14s} {method:7s} ' + '  '.join(f'{v} {np.mean(durations[v]):.0f} (n={len(durations[v])})'
                                                       for v in VARIANTS if v in durations))

    say('\n== QM9, gpu runs: test MAE of all seven targets (mean over seeds) ==')
    for method in CHECK_METHODS:
        values = {}
        for run in filter(ok, runs):
            if run['method'] == method and run['params'].get('TARGET_INDICES') and check_variant(run['params']) == 'gpu':
                for note, result in target_results(run).items():
                    values.setdefault(note, []).append((result['mlp'], result['knn']))
        say(f'{method}: ' + ', '.join(f'{note} MLP {fmt(np.mean([m for m, k in v]))} KNN {fmt(np.mean([k for m, k in v]))}'
                                      f' (n={len(v)})' for note, v in values.items()))

    say('\n== smoke (fixed-stage commands, seed 0): test MAE MLP / KNN ==')
    for group in SMOKE_GROUPS:
        for note in GROUP[group][1]:
            cells = []
            for method, *_ in METHODS:
                found = [target_results(run).get(note) for run in smoke
                         if ok(run) and run['method'] == method and note in target_results(run)]
                cells.append(f'{method} ' + (f'{fmt(found[0]["mlp"])}/{fmt(found[0]["knn"])}' if found else 'MISSING'))
            say(f'{note:14s} ' + '  '.join(cells))


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'check'
    lines = []
    if mode == 'check':
        report_check(lines)
    else:
        sys.exit(f'unknown mode {mode}')
    text = '\n'.join(lines)
    print(text)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f'{mode}_report.txt'), 'w') as f:
        f.write(text + '\n')
