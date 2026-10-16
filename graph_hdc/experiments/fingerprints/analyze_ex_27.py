"""
Analysis of Experiment 27 (see _slurm_ex_27.py): the re-run of Table 1 and of the KNN table on JUPITER.

    python analyze_ex_27.py check     # the GPU equivalence check and the smoke test of the fixed stage
    python analyze_ex_27.py fixed     # the fixed MLP and KNN tables
    python analyze_ex_27.py opt       # the optimized MLP table and the chosen configurations

``opt`` writes opt_runs.csv, opt_configs.csv (the best trial of every study: its validation MAE and parameters) and,
once every cell has results, dataset_comparison_ex27_opt.tex (the format of the fixed tables).

``fixed`` writes to _ex27/: fixed_runs.csv (one row per target, method and seed: test MAE of the MLP and the KNN),
fixed_summary.csv (mean, standard deviation and number of seeds per target, method and model) and the two tables in
the format of the paper's tabulars (dataset_comparison_ex27_mlp.tex, dataset_comparison_ex27_knn.tex: mean over the
seeds with the standard deviation underneath, the best mean of a row in bold, the median rank and the median and
maximum relative deviation from the best mean on top, computed from the printed means as scripts/table_stats.py of
the paper does). The means are printed with three significant digits in the row's smallest mean (the paper's three
decimals do not resolve the QM9 energies in Hartree); the units are those of the datasets.

``check`` prints (and writes to _ex27/check_report.txt):

* the status of every run of the check and the smoke test, and the device the MLPs trained on, per variant
* whether the production code path on the CPU (cpu_prod) reproduces the data path of the earlier tables (cpu) exactly,
  for the MLP and the KNN
* per target, method and seed the test MAE of the MLP in the variants cpu (the data path of the earlier tables),
  cpu_prod, gpu (the production configuration on a GPU) and cpu_nnseed (the cpu variant with another network seed),
  and per target and method the mean absolute differences gpu - cpu and cpu_nnseed - cpu over the seeds: a GPU that
  changes nothing but the floating-point arithmetic gives differences no larger than those from a new network seed,
  without a sign
* the KNN test MAE on the CPU and on the GPU (the same exact search: tied neighbors are chosen by the training order
  on both, so only rounding of the similarities could differ)
* the training times of the MLP, the run times and the peak GPU memory, and the MAE of all seven QM9 targets of the
  gpu runs
* the MAE of every smoke run, per target and method
"""
import os
import sys
import glob
import json

import numpy as np

from _slurm_ex_27 import (PATH, OUT, PREFIX_CHECK, PREFIX_CPUPROD, PREFIX_SMOKE, PREFIX_FIXED, PREFIX_OPT, GROUP,
                          TARGET_INDEX, METHODS, SEEDS, CHECK_TARGETS, CHECK_METHODS, CHECK_SEEDS, SMOKE_GROUPS,
                          method_of_params, method_of_module, check_variant, group_of_note, best_trial, best_params,
                          matches_params)

VARIANTS = ['cpu', 'cpu_prod', 'gpu', 'cpu_nnseed']


def load_runs(prefix: str, any_config: bool = False) -> list:
    """
    Every archive of the prefix as a dict: module, method, params, meta, data, path. The method is recognized by the
    fixed configurations of _slurm_ex_27 (None for other configurations), or with ``any_config`` by the module and the
    fingerprint type alone (the optimized runs).
    """
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
            method = method_of_module(module, params) if any_config else method_of_params(module, params)
            runs.append({'module': module, 'method': method, 'params': params,
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
    runs = load_runs(PREFIX_CHECK) + load_runs(PREFIX_CPUPROD)
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
        devices.setdefault(check_variant(run['params']), set()).add(run['data'].get('nn', {}).get('device'))
    devices['smoke'] = {run['data'].get('nn', {}).get('device') for run in smoke}
    say('MLP devices: ' + ', '.join(f'{v}: {sorted(map(str, d))}' for v, d in devices.items()))

    # results[(note, method, seed, variant)] = target results
    results = {}
    for run in filter(ok, runs):
        for note, values in target_results(run).items():
            results[(note, run['method'], run['params']['SEED'], check_variant(run['params']))] = values

    say('\n== production code path on the CPU (cpu_prod) vs the earlier data path (cpu): must be identical ==')
    pairs = [(results[(n, m, s, 'cpu')], results[(n, m, s, 'cpu_prod')])
             for n in CHECK_TARGETS for m in CHECK_METHODS for s in CHECK_SEEDS
             if (n, m, s, 'cpu') in results and (n, m, s, 'cpu_prod') in results]
    for model in ('mlp', 'knn'):
        differences = [abs(a[model] - b[model]) for a, b in pairs if a[model] is not None and b[model] is not None]
        say(f'{model}: {len(differences)} pairs, identical {sum(d == 0 for d in differences)}, '
            f'max |difference| {max(differences) if differences else float("nan"):.3g}')

    say('\n== MLP test MAE per seed: ' + ' / '.join(VARIANTS) + ' ==')
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
                cpu, gpu, nnseed = arr[:, VARIANTS.index('cpu')], arr[:, VARIANTS.index('gpu')], \
                    arr[:, VARIANTS.index('cpu_nnseed')]
                say(f'   mean:   ' + '  '.join(f'{v} {fmt(x)}' for v, x in zip(VARIANTS, arr.mean(axis=0))))
                say(f'   gpu - cpu: mean {fmt((gpu - cpu).mean())}, mean |.| {fmt(np.abs(gpu - cpu).mean())}'
                    f' ({100 * np.abs(gpu - cpu).mean() / cpu.mean():.1f}% of the cpu MAE)'
                    f' | nnseed - cpu: mean {fmt((nnseed - cpu).mean())}, mean |.| {fmt(np.abs(nnseed - cpu).mean())}'
                    f' ({100 * np.abs(nnseed - cpu).mean() / cpu.mean():.1f}%) [{len(complete)} seeds]')
    # pooled over targets and methods: relative differences (in % of the cpu MAE)
    relative = {'gpu': [], 'cpu_nnseed': []}
    for (note, method, seed, variant), values in results.items():
        base = results.get((note, method, seed, 'cpu'), {}).get('mlp')
        if variant in relative and base and values['mlp'] is not None:
            relative[variant].append(100 * (values['mlp'] - base) / base)
    for variant, values in relative.items():
        if values:
            say(f'pooled {variant} - cpu: {len(values)} pairs, mean {np.mean(values):+.2f}%, '
                f'mean |.| {np.mean(np.abs(values)):.2f}%, larger than cpu in {sum(v > 0 for v in values)}')

    say('\n== KNN test MAE per seed: cpu / gpu (torch backend) ==')
    for note in CHECK_TARGETS:
        for method in CHECK_METHODS:
            pairs = [(results.get((note, method, seed, 'cpu'), {}).get('knn'),
                      results.get((note, method, seed, 'gpu'), {}).get('knn')) for seed in CHECK_SEEDS]
            known = [abs(c - g) for c, g in pairs if None not in (c, g)]
            say(f'{note:14s} {method:7s} ' + '  '.join(f'{fmt(c)}/{fmt(g)}' for c, g in pairs)
                + (f'   max |diff| {max(known):.3g}' if known else ''))

    say('\n== MLP training time per target (s, mean over seeds): cpu / gpu ==')
    for note in CHECK_TARGETS:
        for method in CHECK_METHODS:
            means = {}
            for v in ('cpu', 'gpu'):
                times = [results.get((note, method, seed, v), {}).get('mlp_time') for seed in CHECK_SEEDS]
                times = [t for t in times if t is not None]
                means[v] = float(np.mean(times)) if times else None
            say(f'{note:14s} {method:7s} cpu {fmt(means["cpu"], 1)}  gpu {fmt(means["gpu"], 1)}')
    say('run durations (s, mean over seeds; the gpu runs of QM9 train all seven targets) and peak GPU memory (GB):')
    for note in CHECK_TARGETS:
        group = group_of_note(note)
        for method in CHECK_METHODS:
            durations, memory = {}, []
            for run in filter(ok, runs):
                params = run['params']
                if run['method'] != method or params['NOTE'] not in (note, group):
                    continue
                variant = check_variant(params)
                durations.setdefault(variant, []).append(run['meta'].get('duration', np.nan))
                if variant == 'gpu' and 'gpu' in run['data']:
                    memory.append(run['data']['gpu'].get('max_memory_reserved', np.nan) / 1e9)
            say(f'{note:14s} {method:7s} ' + '  '.join(f'{v} {np.mean(durations[v]):.0f} (n={len(durations[v])})'
                                                       for v in VARIANTS if v in durations)
                + (f'  | GPU reserved max {max(memory):.2f}' if memory else ''))
    smoke_memory = [run['data']['gpu'].get('max_memory_reserved', np.nan) / 1e9 for run in smoke
                    if ok(run) and 'gpu' in run['data']]
    if smoke_memory:
        say(f'smoke runs: peak GPU memory reserved max {max(smoke_memory):.2f} GB')

    say('\n== QM9, gpu runs: test MAE of all seven targets (mean over seeds) ==')
    for method in CHECK_METHODS:
        values = {}
        for run in filter(ok, runs):
            if run['method'] == method and check_variant(run['params']) == 'gpu' \
                    and len(run['params'].get('TARGET_INDICES') or []) > 1:
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


# The rows of Table 1 in the paper's order: (note, dataset label, quantity label). QM9 energies are in Hartree (the
# unit of the dataset), as in the revised table.
ROWS = [
    ('hopv15_pce', 'HOPV15', r'PCE \qunit{\%}'),
    ('hopv15_voc', '', r'$V_{oc}$ \qunit{V}'),
    ('freesolv_hfe', 'FreeSolv', r'$\Delta G$ \qunit{kcal/mol}'),
    ('bace_ic50', 'BACE', r'IC50 \qunit{pIC50}'),
    ('lipophilicity_logD', 'LIPOP', 'LogD'),
    ('aqsoldb_logs', 'AqSolDB', r'logS \qunit{log\,M}'),
    ('clogp', 'clogp', 'ClogP'),
    ('compas_energy', 'COMPAS-3X', r'$U_0$ \qunit{eV}'),
    ('compas_gap', '', r'Gap \qunit{eV}'),
    ('compas_dipole', '', r'$\mu$ \qunit{D}'),
    ('qm9_dipole', 'QM9', r'$\mu$ \qunit{D}'),
    ('qm9_cv', '', r'$C_v$ \qunit{cal/(mol\,K)}'),
    ('qm9_zpve', '', r'$ZPVE$ \qunit{Ha}'),
    ('qm9_alpha', '', r'$\alpha$ \qunit{$a_0^3$}'),
    ('qm9_enthalpy', '', r'$\Delta H$ \qunit{Ha}'),
    ('qm9_energy', '', r'$U_0$ \qunit{Ha}'),
    ('qm9_gap', '', r'Gap \qunit{Ha}'),
]
COLUMN_LABELS = {'hdf': 'HDF', 'morgan': 'Morgan', 'rdkit': 'RDKit', 'torsion': 'Torsion', 'atompair': 'AtomPair'}


def decimals(values) -> int:
    """Decimals that give the smallest value three significant digits, at least three (the paper's convention)."""
    smallest = min(v for v in values if v > 0)
    return max(3, 2 - int(np.floor(np.log10(smallest))))


def tabular(summary: dict, model: str, datapoints: dict) -> str:
    """The table of one model in the format of the paper's tabulars/dataset_comparison*.tex."""
    methods = [method for method, *_ in METHODS]
    rounded, cells = [], []
    for note, _, _ in ROWS:
        means = [summary[(note, method, model)][0] for method in methods]
        d = decimals(means)
        row_means = [round(m, d) for m in means]
        best = min(row_means)
        row_cells = []
        for method, mean in zip(methods, row_means):
            std = summary[(note, method, model)][1]
            value = f'{mean:.{d}f}'
            value = rf'\mathbf{{ {value} }}' if mean == best else f' {value} '
            row_cells.append(rf'$\underset{{\color{{darkgray}} \pm{std:.{max(2, d - 1)}f}}}{{{value}}}$')
        rounded.append(row_means)
        cells.append(row_cells)
    X = np.array(rounded)
    best = X.min(axis=1, keepdims=True)
    deviation = 100 * (X - best) / best
    ranks = np.array([np.argsort(np.argsort(row)) + 1.0 for row in X])
    # average ranks for ties (as scipy.stats.rankdata, used by the paper's table_stats.py)
    for i, row in enumerate(X):
        for value in set(row):
            tied = row == value
            ranks[i, tied] = ranks[i, tied].mean()
    out = [r'\begin{tabular}{llrccccc}', r'\toprule',
           ' &  &  & ' + ' & '.join(COLUMN_LABELS[m] for m in methods) + r' \\', r'\midrule', '']
    for title, values in [('Median Rank', [f'${v:g}$' for v in np.median(ranks, axis=0)]),
                          ('Median rel. deviation', [f'${v:.1f}\\%$' for v in np.median(deviation, axis=0)]),
                          ('Max. rel. deviation', [f'${v:,.0f}\\%$'.replace(',', '{,}') for v in deviation.max(axis=0)])]:
        out += [rf'\multicolumn{{3}}{{l}}{{{title}}} &'] + [f'{v} &' for v in values[:-1]] + [f'{values[-1]} \\\\', '']
    out += [r'\arrayrulecolor{gray}\midrule\arrayrulecolor{black}', '',
            'Dataset & Quantity & Datapoints & ' + ' & '.join(COLUMN_LABELS[m] for m in methods) + r' \\', '',
            r'\arrayrulecolor{gray}\midrule\arrayrulecolor{black}', '']
    for (note, dataset, quantity), row_cells in zip(ROWS, cells):
        out += [f'{dataset} &', f'{quantity} &', f'{datapoints.get(note, "")} &']
        out += [f'{c} &' for c in row_cells[:-1]] + [f'{row_cells[-1]} \\\\', '']
    out += [r'\bottomrule', r'\end{tabular}']
    return '\n'.join(out) + '\n'


def report_fixed(lines: list):
    say = lines.append
    runs = load_runs(PREFIX_FIXED)
    failed = [run for run in runs if not ok(run)]
    say(f'{len(runs)} archives of {PREFIX_FIXED}, {len(runs) - len(failed)} done')
    for run in failed:
        say(f'   NOT DONE {os.path.basename(run["path"])}: status={run["meta"].get("status")} '
            f'error={run["meta"].get("error_type")}')
    devices = {run['data'].get('nn', {}).get('device') for run in runs if ok(run)}
    say(f'MLP devices: {sorted(map(str, devices))}')

    records, datapoints = [], {}
    for run in filter(ok, runs):
        indices = run['data'].get('indices', {})
        size = sum(len(indices.get(split, [])) for split in ('train', 'val', 'test'))
        for note, values in target_results(run).items():
            records.append((note, run['method'], run['params']['SEED'], values['mlp'], values['knn']))
            datapoints[note] = size
    seen = {}
    for note, method, seed, mlp, knn in records:
        seen.setdefault((note, method, seed), 0)
        seen[(note, method, seed)] += 1
    duplicates = [key for key, count in seen.items() if count > 1]
    if duplicates:
        say(f'WARNING: {len(duplicates)} (target, method, seed) with several archives, e.g. {duplicates[:3]}; '
            f'all are averaged')

    summary = {}
    say('\ntest MAE, mean +- std over the seeds (n):')
    for note, _, _ in ROWS:
        for model, column in (('mlp', 3), ('knn', 4)):
            cells = []
            for method, *_ in METHODS:
                values = [r[column] for r in records if r[0] == note and r[1] == method and r[column] is not None]
                summary[(note, method, model)] = (float(np.mean(values)) if values else np.nan,
                                                  float(np.std(values, ddof=1)) if len(values) > 1 else np.nan,
                                                  len(values))
                mean, std, n = summary[(note, method, model)]
                cells.append(f'{method} {mean:.5g}+-{std:.2g} ({n})')
            say(f'{note:20s} {model}: ' + '  '.join(cells))

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'fixed_runs.csv'), 'w') as f:
        f.write('target,method,seed,mlp_test_mae,knn_test_mae\n')
        for note, method, seed, mlp, knn in sorted(records, key=lambda r: (r[0], r[1], r[2])):
            f.write(f'{note},{method},{seed},{mlp!r},{knn!r}\n')
    with open(os.path.join(OUT, 'fixed_summary.csv'), 'w') as f:
        f.write('target,method,model,mean,std,n\n')
        for (note, method, model), (mean, std, n) in summary.items():
            f.write(f'{note},{method},{model},{mean!r},{std!r},{n}\n')
    # the tables only with every seed in every cell; an older table of an incomplete state is removed
    complete = all(summary[(note, method, model)][2] == len(SEEDS) for note, *_ in ROWS for method, *_ in METHODS
                   for model in ('mlp', 'knn'))
    for model in ('mlp', 'knn'):
        path = os.path.join(OUT, f'dataset_comparison_ex27_{model}.tex')
        if complete:
            with open(path, 'w') as f:
                f.write(tabular(summary, model, datapoints))
        elif os.path.exists(path):
            os.remove(path)
    if complete:
        say('\nwrote _ex27/fixed_runs.csv, fixed_summary.csv, dataset_comparison_ex27_mlp.tex, '
            'dataset_comparison_ex27_knn.tex')
    else:
        say(f'\nnot every cell has all {len(SEEDS)} seeds: wrote only fixed_runs.csv and fixed_summary.csv')


# the parameters of a chosen configuration in the table of the optimized configurations (opt_configs.csv)
CONFIG_KEYS = ['EMBEDDING_SIZE', 'NUM_LAYERS', 'GRAPH_ATTRIBUTES', 'FINGERPRINT_SIZE', 'FINGERPRINT_RADIUS',
               'FINGERPRINT_MAX_DISTANCE', 'FINGERPRINT_TORSION_ATOMS', 'NN_HIDDEN_LAYER_SIZES', 'NN_LEARNING_RATE_INIT']


def report_opt(lines: list):
    """The optimized MLP table (best configuration of every study over the seeds) and the chosen configurations."""
    say = lines.append
    runs = load_runs(PREFIX_OPT, any_config=True)
    failed = [run for run in runs if not ok(run)]
    say(f'{len(runs)} archives of {PREFIX_OPT}, {len(runs) - len(failed)} done')
    for run in failed:
        say(f'   NOT DONE {os.path.basename(run["path"])}: error={run["meta"].get("error_type")}')

    # only the runs with the best configuration of their study, the latest one per target, method and seed
    latest, best, stale = {}, {}, 0
    for run in sorted(filter(ok, runs), key=lambda r: r['meta'].get('start_time', 0)):
        key = (run['params']['NOTE'], run['method'])
        if key not in best:
            try:
                best[key] = best_params(*key[::-1])
            except Exception as error:
                say(f'no best configuration for {key}: {error}')
                best[key] = None
        if best[key] is None or not matches_params(run['params'], best[key]):
            stale += 1
            continue
        latest[(*key, run['params']['SEED'])] = run
    if stale:
        say(f'skipped {stale} archives whose configuration is not (or no longer) the best of their study')
    records, datapoints = [], {}
    for run in latest.values():
        indices = run['data'].get('indices', {})
        for note, values in target_results(run).items():
            records.append((note, run['method'], run['params']['SEED'], values['mlp']))
            datapoints[note] = sum(len(indices.get(split, [])) for split in ('train', 'val', 'test'))
    summary = {}
    for note, _, _ in ROWS:
        for method, *_ in METHODS:
            values = [r[3] for r in records if r[0] == note and r[1] == method and r[3] is not None]
            summary[(note, method, 'mlp')] = (float(np.mean(values)) if values else np.nan,
                                              float(np.std(values, ddof=1)) if len(values) > 1 else np.nan,
                                              len(values))
        say(f'{note:20s} ' + '  '.join(f'{m} {summary[(note, m, "mlp")][0]:.5g}+-{summary[(note, m, "mlp")][1]:.2g}'
                                       f' ({summary[(note, m, "mlp")][2]})' for m, *_ in METHODS))

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'opt_runs.csv'), 'w') as f:
        f.write('target,method,seed,mlp_test_mae\n')
        for note, method, seed, mlp in sorted(records, key=lambda r: (r[0], r[1], r[2])):
            f.write(f'{note},{method},{seed},{mlp!r}\n')
    with open(os.path.join(OUT, 'opt_configs.csv'), 'w') as f:
        f.write('target,method,validation_mae,' + ','.join(CONFIG_KEYS) + '\n')
        for note, _, _ in ROWS:
            for method, *_ in METHODS:
                try:
                    trial = best_trial(method, note)
                except Exception as error:
                    say(f'no complete study {method}__{note}: {error}')
                    continue
                params = json.loads(trial.user_attrs['params'])
                cells = [str(params.get(k, '')).replace(',', ';') for k in CONFIG_KEYS]
                f.write(f'{note},{method},{trial.value!r},' + ','.join(cells) + '\n')
    # the table only with every seed of the current best configuration in every cell; an older table is removed
    path = os.path.join(OUT, 'dataset_comparison_ex27_opt.tex')
    if all(summary[(note, method, 'mlp')][2] == len(SEEDS) for note, *_ in ROWS for method, *_ in METHODS):
        with open(path, 'w') as f:
            f.write(tabular(summary, 'mlp', datapoints))
        say('\nwrote _ex27/opt_runs.csv, opt_configs.csv, dataset_comparison_ex27_opt.tex')
    else:
        if os.path.exists(path):
            os.remove(path)
        say(f'\nnot every cell has all {len(SEEDS)} seeds: wrote only opt_runs.csv and opt_configs.csv')


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'check'
    lines = []
    if mode == 'check':
        report_check(lines)
    elif mode == 'fixed':
        report_fixed(lines)
    elif mode == 'opt':
        report_opt(lines)
    else:
        sys.exit(f'unknown mode {mode}')
    text = '\n'.join(lines)
    print(text)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f'{mode}_report.txt'), 'w') as f:
        f.write(text + '\n')
