"""
Analysis for Experiment 13 (the HPO-first main comparison table). Runs headless so it can sit
between the HPO and table SLURM stages.

    python analyze_ex_13.py select   # ex_13_hpo archives -> best config per (rep, dataset) by
                                      # VALIDATION r2 -> experiment_best_parameters_map__ex13.json
    python analyze_ex_13.py table    # ex_13_table archives -> MAE table (mean +/- std over seeds),
                                      # median rank + rel. deviation, LaTeX, and paired Wilcoxon vs HDC

Selection is on the held-out *validation* metric (val_neural_net2), never test -- that is the whole
point of the HPO-first protocol (Design A): tune on validation, freeze, report test over seeds 0-4.
"""
import os
import sys
import json
import glob
from collections import defaultdict

import numpy as np

PATH = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(PATH, 'results')
JSON_OUT = os.path.join(PATH, 'experiment_best_parameters_map__ex13.json')

# featurization + model params that define a configuration
FP_KEYS = ('FINGERPRINT_TYPE', 'FINGERPRINT_SIZE', 'FINGERPRINT_RADIUS')
HDC_KEYS = ('EMBEDDING_SIZE', 'NUM_LAYERS')
SHERLOCK_KEYS = ('FINGERPRINT_SIZE', 'SHERLOCK_RADIUS', 'SHERLOCK_DICTIONARY_PATH')
MLP_KEYS = ('NN_HIDDEN_LAYER_SIZES', 'NN_LEARNING_RATE_INIT')

# table layout
REP_ORDER = ['hdc', 'morgan', 'count_morgan', 'rdkit', 'torsion', 'atom', 'sherlock']
REP_LABEL = {'hdc': 'HDC', 'morgan': 'Morgan', 'count_morgan': 'CountMorgan', 'rdkit': 'RDKit',
             'torsion': 'Torsion', 'atom': 'AtomPair', 'sherlock': 'Sherlock'}
DATASET_ORDER = [
    'aqsoldb_logs', 'clogp', 'freesolv_hfe', 'lipophilicity_logD', 'bace_ic50',
    'hopv15_gap', 'hopv15_jsc', 'hopv15_voc', 'hopv15_pce',
    'compas_dipole', 'compas_gap', 'compas_energy',
    'qm9_dipole', 'qm9_alpha', 'qm9_gap', 'qm9_energy', 'qm9_zpve', 'qm9_enthalpy', 'qm9_cv',
]


def _iter_archives(prefix: str):
    """Yield (params, data) for every completed archive whose __PREFIX__ matches."""
    for meta_path in glob.glob(os.path.join(RESULTS, 'predict_molecules__*', '*', 'experiment_meta.json')):
        try:
            meta = json.load(open(meta_path))
        except Exception:
            continue
        params = {k: v.get('value') for k, v in meta.get('parameters', {}).items() if isinstance(v, dict)}
        if prefix not in str(params.get('__PREFIX__', '')):
            continue
        if meta.get('has_error'):
            continue
        data_path = os.path.join(os.path.dirname(meta_path), 'experiment_data.json')
        if not os.path.exists(data_path):
            continue
        try:
            data = json.load(open(data_path))
        except Exception:
            continue
        yield params, data


def _rep_of(params: dict) -> str:
    if 'SHERLOCK_RADIUS' in params:
        return 'sherlock'
    return params['FINGERPRINT_TYPE'] if params.get('FINGERPRINT_TYPE') else 'hdc'


def _config_of(params: dict) -> dict:
    if 'SHERLOCK_RADIUS' in params:
        keys = SHERLOCK_KEYS + MLP_KEYS
    elif params.get('FINGERPRINT_TYPE'):
        keys = FP_KEYS + MLP_KEYS
    else:
        keys = HDC_KEYS + MLP_KEYS
    return {k: params[k] for k in keys if k in params}


def select():
    # (rep, note) -> list of (val_r2, config)
    groups = defaultdict(list)
    n = 0
    for params, data in _iter_archives('ex_13_hpo'):
        metrics = data.get('metrics', {})
        val = metrics.get('val_neural_net2', {})
        if 'r2' not in val:
            continue
        groups[(_rep_of(params), params['NOTE'])].append((val['r2'], _config_of(params)))
        n += 1
    best = {}
    for key, cands in groups.items():
        val_r2, cfg = max(cands, key=lambda t: t[0])
        best[key] = cfg
        print(f'  {key[0]:>8} / {key[1]:<20} best val_r2={val_r2:.3f} over {len(cands):>2} configs -> {cfg}')
    json.dump([[list(k), v] for k, v in best.items()], open(JSON_OUT, 'w'), indent=2)
    print(f'\nscanned {n} ex_13_hpo archives, selected {len(best)} (rep,dataset) cells -> {JSON_OUT}')


def table():
    # Dedup by (rep, dataset, seed): re-runs (e.g. OOM retries) can leave more than one archive for
    # the same cell+seed; keep a single MAE per seed so a duplicate can't skew the mean.
    by_seed = {}
    for params, data in _iter_archives('ex_13_table'):
        test = data.get('metrics', {}).get('test_neural_net2', {})
        if 'mae' in test:
            by_seed[(_rep_of(params), params['NOTE'], params.get('SEED'))] = test['mae']
    # (rep, note) -> list of test MAE (one per seed)
    maes = defaultdict(list)
    for (rep, note, _seed), mae in by_seed.items():
        maes[(rep, note)].append(mae)

    # numeric mean-MAE matrix [dataset][rep]
    mean = {d: {} for d in DATASET_ORDER}
    std = {d: {} for d in DATASET_ORDER}
    for (rep, note), vals in maes.items():
        if note in mean:
            mean[note][rep] = float(np.mean(vals))
            std[note][rep] = float(np.std(vals))

    # per-dataset ranks + relative deviation from the best
    ranks = {rep: [] for rep in REP_ORDER}
    reldev = {rep: [] for rep in REP_ORDER}
    for d in DATASET_ORDER:
        present = {r: mean[d][r] for r in REP_ORDER if r in mean[d]}
        if not present:
            continue
        best_val = min(present.values())
        order = sorted(present, key=lambda r: present[r])
        for i, r in enumerate(order):
            ranks[r].append(i + 1)
            reldev[r].append((present[r] - best_val) / best_val * 100.0 if best_val else 0.0)

    # --- text table ---
    header = ['Dataset'] + [REP_LABEL[r] for r in REP_ORDER]
    print('\t'.join(header))
    for d in DATASET_ORDER:
        row = [d]
        best_val = min([mean[d][r] for r in REP_ORDER if r in mean[d]], default=None)
        for r in REP_ORDER:
            if r in mean[d]:
                mark = '*' if mean[d][r] == best_val else ' '
                row.append(f'{mean[d][r]:.3f}±{std[d][r]:.2f}{mark}')
            else:
                row.append('—')
        print('\t'.join(row))
    print('MedianRank\t' + '\t'.join(f'{np.median(ranks[r]):.1f}' if ranks[r] else '—' for r in REP_ORDER))
    print('RelDev%\t' + '\t'.join(f'{np.mean(reldev[r]):.1f}' if reldev[r] else '—' for r in REP_ORDER))

    # --- paired Wilcoxon: HDC vs each baseline over per-dataset mean MAE ---
    try:
        from scipy.stats import wilcoxon
        print('\nPaired Wilcoxon (HDC vs baseline, per-dataset mean MAE):')
        for r in REP_ORDER:
            if r == 'hdc':
                continue
            pairs = [(mean[d]['hdc'], mean[d][r]) for d in DATASET_ORDER if 'hdc' in mean[d] and r in mean[d]]
            if len(pairs) >= 6:
                h = [p[0] for p in pairs]; b = [p[1] for p in pairs]
                stat, p = wilcoxon(h, b)
                wins = sum(1 for x, y in pairs if x < y)
                print(f'  HDC vs {REP_LABEL[r]:<8}: p={p:.4f}  (HDC lower MAE on {wins}/{len(pairs)})')
    except Exception as e:
        print('  (wilcoxon skipped:', e, ')')

    # --- LaTeX (underset std, bold best per row) ---
    lines = [r'\begin{tabular}{ll' + 'c' * len(REP_ORDER) + '}', r'\toprule',
             ' & '.join(['Dataset', 'Quantity'] + [REP_LABEL[r] for r in REP_ORDER]) + r' \\', r'\midrule']
    for d in DATASET_ORDER:
        if not mean[d]:
            continue
        best_val = min(mean[d][r] for r in REP_ORDER if r in mean[d])
        cells = []
        for r in REP_ORDER:
            if r in mean[d]:
                val = f'{mean[d][r]:.3f}'
                val = r'\mathbf{' + val + '}' if mean[d][r] == best_val else val
                cells.append(r'$\underset{\pm' + f'{std[d][r]:.2f}' + '}{' + val + '}$')
            else:
                cells.append('--')
        lines.append(f'{d} & MAE & ' + ' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    tex = os.path.join(PATH, '_ex13', 'dataset_comparison_ex13.tex')
    open(tex, 'w').write('\n'.join(lines) + '\n')
    print(f'\nLaTeX table -> {tex}')


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'select'
    if mode == 'select':
        select()
    elif mode == 'table':
        table()
    else:
        sys.exit('usage: python analyze_ex_13.py {select|table}')
