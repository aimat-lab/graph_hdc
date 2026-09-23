"""
Analysis for Experiment 14 (HDF vs. graph neural networks, reviewer comment R2.1).

Collects the ex_14 archives of the three experiment modules and writes to ``_ex14/``:

* ``results_ex14.csv``           one row per (variant, model, dataset, seed) with test MAE / R2 and timings
* ``gnn_comparison_mlp.tex``     MAE table: HDF + MLP vs. random-init GNN + MLP vs. trained GNN (end-to-end)
* ``gnn_comparison_knn.tex``     MAE table: HDF + KNN vs. random-init GNN + KNN
* ``gnn_comparison_cost.tex``    median featurization / training wall time per variant and dataset
* ``gnn_convergence.txt``        best epoch vs. stop epoch of the trained GNNs (early-stopping sanity check)

Significance: paired two-sided Wilcoxon signed-rank tests over the seeds (HDF vs. each other column of the
same table, paired by SEED so that both sides share the identical train/val/test split), Holm-corrected
over all tests of a table. Marks: * p<0.05, ** p<0.01 (corrected).

Usage:
    python analyze_ex_14.py                 # prefix ex_14_gnn
    python analyze_ex_14.py ex_14_smoke     # any other archive prefix, e.g. the smoke runs
"""
import os
import sys
import csv
import json
import glob
from collections import defaultdict

import numpy as np
from scipy.stats import wilcoxon

PATH = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(PATH, 'results')
OUT = os.path.join(PATH, '_ex14')

ARCHS = ['gcn', 'gin', 'gatv2']
ARCH_LABEL = {'gcn': 'GCN', 'gin': 'GIN', 'gatv2': 'GATv2'}
DATASET_ORDER = [
    'freesolv_hfe', 'aqsoldb_logs', 'lipophilicity_logD', 'bace_ic50', 'hopv15_pce',
    'compas_gap', 'qm9_gap', 'qm9_energy', 'qm9_zpve',
]
DATASET_LABEL = {
    'freesolv_hfe': 'FreeSolv', 'aqsoldb_logs': 'AqSolDB', 'lipophilicity_logD': 'Lipophilicity',
    'bace_ic50': 'BACE', 'hopv15_pce': 'HOPV15 (PCE)', 'compas_gap': 'COMPAS-3X (gap)',
    'qm9_gap': 'QM9 (gap)', 'qm9_energy': r'QM9 ($U_0$)', 'qm9_zpve': 'QM9 (ZPVE)',
}


def iter_archives(prefix: str):
    """Yield (module, meta, params, data) for every completed archive whose __PREFIX__ equals ``prefix``."""
    pattern = os.path.join(RESULTS, 'predict_molecules__*', '*', 'experiment_meta.json')
    for meta_path in glob.glob(pattern):
        module = os.path.basename(os.path.dirname(os.path.dirname(meta_path)))
        try:
            meta = json.load(open(meta_path))
        except Exception:
            continue
        params = {k: v.get('value') for k, v in meta.get('parameters', {}).items() if isinstance(v, dict)}
        # a killed or timed-out run keeps status 'running' (and has_error False), so require 'done'
        if params.get('__PREFIX__') != prefix or meta.get('status') != 'done' or meta.get('has_error'):
            continue
        # the first round of trained GNNs used early stopping and was replaced by full-length runs
        if module == 'predict_molecules__gnn' and params.get('EARLY_STOPPING_PATIENCE') is not None:
            continue
        data_path = os.path.join(os.path.dirname(meta_path), 'experiment_data.json')
        if not os.path.exists(data_path):
            continue
        try:
            data = json.load(open(data_path))
        except Exception:
            continue
        yield module, meta, params, data


def collect(prefix: str) -> list:
    """
    Flatten the archives into records ``{variant, arch, model, dataset, seed, mae, r2, ...}``.

    variant is 'hdf', 'random' or 'trained'; model is the downstream model ('neural_net2', 'k_neighbors')
    or 'end2end' for the trained GNN. Duplicates of (variant, arch, model, dataset, seed), e.g. from a
    retried run, keep the newest archive.
    """
    records = {}
    # oldest first, so that a newer archive of the same run (e.g. a retry) overwrites an older one
    archives = sorted(iter_archives(prefix), key=lambda t: t[1].get('start_time') or 0)
    for module, meta, params, data in archives:
        metrics = data.get('metrics', {})
        if module == 'predict_molecules__hdc':
            variant, arch, models = 'hdf', '', ['neural_net2', 'k_neighbors']
        elif module == 'predict_molecules__gnn_random':
            variant, arch, models = 'random', params['GNN_ARCH'], ['neural_net2', 'k_neighbors']
        elif module == 'predict_molecules__gnn':
            variant, arch = 'trained', params['MODELS'][0]
            models = [arch]
        else:
            continue
        for model in models:
            key = f'test_{model}'
            if key not in metrics:
                continue
            rec = {
                'variant': variant, 'arch': arch,
                'model': 'end2end' if variant == 'trained' else model,
                'dataset': params['NOTE'], 'seed': params['SEED'],
                'mae': metrics[key]['mae'], 'r2': metrics[key]['r2'],
                'process_time': data.get('process_time'),
                # pure encoding time (HDF: only set on the cache-building run of a dataset/seed)
                'encode_time': data.get('encode_time'),
                'train_time': data.get('train_time', {}).get(model),
                'fit_time': data.get('fit_time', {}).get(model),
                'best_epoch': data.get('best_epoch', {}).get(model),
                'epochs': data.get('epochs', {}).get(model),
            }
            k = (rec['variant'], rec['arch'], rec['model'], rec['dataset'], rec['seed'])
            records[k] = rec
    return list(records.values())


def by_seed(records: list, variant: str, arch: str, model: str, dataset: str) -> dict:
    return {r['seed']: r['mae'] for r in records
            if (r['variant'], r['arch'], r['model'], r['dataset']) == (variant, arch, model, dataset)}


def holm(pvalues: list) -> list:
    """Holm-Bonferroni adjusted p-values (None entries are passed through)."""
    idx = [i for i, p in enumerate(pvalues) if p is not None]
    order = sorted(idx, key=lambda i: pvalues[i])
    adjusted = list(pvalues)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(order) - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def fmt(values: list) -> str:
    mean, std = np.mean(values), np.std(values)
    digits = max(0, 2 - int(np.floor(np.log10(abs(mean))))) if mean != 0 else 2
    digits = min(digits, 4)
    return f'{mean:.{digits}f} $\\pm$ {std:.{digits}f}'


def table(records: list, columns: list, caption_note: str) -> str:
    """
    LaTeX tabular with datasets as rows and ``columns`` = [(label, variant, arch, model), ...]. The first
    column is the reference (HDF) against which all others are tested. The best mean MAE per row is bold.
    """
    rows, tests = [], []
    for dataset in DATASET_ORDER:
        cells = [by_seed(records, v, a, m, dataset) for _, v, a, m in columns]
        if not any(cells):
            continue
        ref = cells[0]
        pvals = []
        for cell in cells[1:]:
            seeds = sorted(set(ref) & set(cell))
            diffs = [ref[s] - cell[s] for s in seeds]
            if len(seeds) >= 5 and any(d != 0 for d in diffs):
                pvals.append(wilcoxon([ref[s] for s in seeds], [cell[s] for s in seeds]).pvalue)
            else:
                pvals.append(None)
        rows.append((dataset, cells))
        tests.append(pvals)

    flat = holm([p for pv in tests for p in pv])
    it = iter(flat)
    tests = [[next(it) for _ in pv] for pv in tests]

    lines = [
        '\\begin{tabular}{l' + 'c' * len(columns) + '}',
        '\\toprule',
        'Dataset & ' + ' & '.join(label for label, *_ in columns) + ' \\\\',
        '\\midrule',
    ]
    for (dataset, cells), pvals in zip(rows, tests):
        means = [np.mean(list(c.values())) if c else np.inf for c in cells]
        best = int(np.argmin(means))
        out = [DATASET_LABEL.get(dataset, dataset)]
        for i, cell in enumerate(cells):
            if not cell:
                out.append('--')
                continue
            text = fmt(list(cell.values()))
            if i == best:
                text = f'\\textbf{{{text}}}'
            if i > 0 and pvals[i - 1] is not None:
                text += '$^{**}$' if pvals[i - 1] < 0.01 else ('$^{*}$' if pvals[i - 1] < 0.05 else '')
            if len(cell) < 10:
                text += f' ({len(cell)})'   # flags incomplete cells (fewer seeds than planned)
            out.append(text)
        lines.append(' & '.join(out) + ' \\\\')
    lines += ['\\bottomrule', '\\end{tabular}', f'% {caption_note}']
    return '\n'.join(lines) + '\n'


def cost_table(records: list) -> str:
    columns = [('HDF', 'hdf', '', 'neural_net2')]
    columns += [(f'Random {ARCH_LABEL[a]}', 'random', a, 'neural_net2') for a in ARCHS]
    columns += [(f'Trained {ARCH_LABEL[a]}', 'trained', a, 'end2end') for a in ARCHS]
    lines = [
        '\\begin{tabular}{l' + 'c' * len(columns) + '}',
        '\\toprule',
        'Dataset & ' + ' & '.join(label for label, *_ in columns) + ' \\\\',
        '\\midrule',
    ]
    for dataset in DATASET_ORDER:
        out = [DATASET_LABEL.get(dataset, dataset)]
        found = False
        for _, v, a, m in columns:
            recs = [r for r in records if (r['variant'], r['arch'], r['model'], r['dataset']) == (v, a, m, dataset)]
            # encoding time for the fixed representations, training time for the end-to-end GNNs
            times = [r['train_time'] if v == 'trained' else r['encode_time'] for r in recs]
            times = [t for t in times if t is not None]
            found |= bool(times)
            out.append(f'{np.median(times):.0f}' if times else '--')
        if found:
            lines.append(' & '.join(out) + ' \\\\')
    lines += ['\\bottomrule', '\\end{tabular}',
              '% median wall time in seconds: featurization + encoding of the whole dataset (HDF on CPU, '
              'random GNN on GPU; HDF only from the cache-building run of a dataset/seed, so QM9 is '
              'reported under the gap target) / training until the best epoch (trained GNN, GPU)']
    return '\n'.join(lines) + '\n'


def convergence(records: list) -> str:
    lines = ['dataset\tarch\tn\tmedian_best_epoch\tmax_best_epoch\tmedian_stop_epoch\tmax_stop_epoch\tbest_after_900']
    for dataset in DATASET_ORDER:
        for a in ARCHS:
            recs = [r for r in records if r['variant'] == 'trained' and r['arch'] == a and r['dataset'] == dataset
                    and r['best_epoch'] is not None]
            if not recs:
                continue
            best = [r['best_epoch'] for r in recs]
            stop = [r['epochs'] for r in recs]
            # best epoch in the last 10% of training: the run might still have been improving
            hit_cap = sum(1 for b in best if b >= 900)
            lines.append(f'{dataset}\t{a}\t{len(recs)}\t{np.median(best):.0f}\t{max(best)}\t'
                         f'{np.median(stop):.0f}\t{max(stop)}\t{hit_cap}')
    return '\n'.join(lines) + '\n'


def main(prefix: str):
    os.makedirs(OUT, exist_ok=True)
    records = collect(prefix)
    print(f'collected {len(records)} records for prefix "{prefix}"')
    if not records:
        return

    with open(os.path.join(OUT, f'results_{prefix}.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(sorted(records, key=lambda r: (r['dataset'], r['variant'], r['arch'], r['model'], r['seed'])))

    mlp_columns = [('HDF + MLP', 'hdf', '', 'neural_net2')]
    mlp_columns += [(f'Rand. {ARCH_LABEL[a]} + MLP', 'random', a, 'neural_net2') for a in ARCHS]
    mlp_columns += [(f'Trained {ARCH_LABEL[a]}', 'trained', a, 'end2end') for a in ARCHS]
    knn_columns = [('HDF + KNN', 'hdf', '', 'k_neighbors')]
    knn_columns += [(f'Rand. {ARCH_LABEL[a]} + KNN', 'random', a, 'k_neighbors') for a in ARCHS]

    outputs = {
        f'gnn_comparison_mlp_{prefix}.tex': table(records, mlp_columns, 'test MAE, mean +/- std over seeds; '
                                                  'Wilcoxon vs. HDF + MLP, Holm-corrected'),
        f'gnn_comparison_knn_{prefix}.tex': table(records, knn_columns, 'test MAE, mean +/- std over seeds; '
                                                  'Wilcoxon vs. HDF + KNN, Holm-corrected'),
        f'gnn_comparison_cost_{prefix}.tex': cost_table(records),
        f'gnn_convergence_{prefix}.txt': convergence(records),
    }
    for name, content in outputs.items():
        with open(os.path.join(OUT, name), 'w') as f:
            f.write(content)
        print(f'--- {name}\n{content}')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'ex_14_gnn')
