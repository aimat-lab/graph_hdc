"""
Analysis of Experiment 22 (ex_22): unit-modulus spectra and the graph size/diameter encodings of HDF.

Pairs the runs of ``_slurm_ex_22.py`` by (dataset, seed) and compares the MLP test MAE of the three changed arms with
the original HDF (SPECTRUM="gaussian", GRAPH_ATTRIBUTES=True). All arms of a seed use the same split, the same
random draws for the codebooks and the same MLP seed, so the paired difference isolates the change.

Per dataset and contrast: mean test MAE ± SD over the seeds per arm, the mean paired change relative to the
original arm with a t-based 95% confidence interval, the number of seeds in which the arm is better, and the
Wilcoxon signed-rank p-value, Holm-corrected per contrast over the datasets (family fixed before the runs: each
contrast is its own question; with 10 seeds and one family of all 15 cells only a 10/10 sweep could reach
p < 0.05). Also the median number of
effective Fourier components of the embeddings per arm (diagnostic logged by predict_molecules__hdc.py).

    python analyze_ex_22.py [prefix]      # default prefix: ex_22_unitspec

Writes ``_ex22/unitspec_comparison_<prefix>.md`` (tables) and ``.csv`` (all runs).
"""
import os
import sys
import csv
import json
import glob
from collections import defaultdict

import numpy as np
from scipy.stats import t as t_dist, wilcoxon

from analyze_ex_14 import DATASET_ORDER, DATASET_LABEL

PATH = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(PATH, 'results')
OUT = os.path.join(PATH, '_ex22')
MODEL = 'neural_net2'
ORIGINAL = ('gaussian', True)
ARMS = [('gaussian', True), ('unit', True), ('gaussian', False), ('unit', False)]
ARM_LABEL = {
    ('gaussian', True): 'original',
    ('unit', True): 'unit spectra',
    ('gaussian', False): 'no size/diameter',
    ('unit', False): 'unit spectra, no size/diameter',
}
# settings every ex_22 archive must have (anything else is not counted)
REQUIRED = {'BIDIRECTIONAL': True, 'HYDROGEN_COUNT': 'total', 'EMBEDDING_SIZE': 2048, 'NUM_LAYERS': 2,
            'ENCODING_MODE': 'continuous', 'QUANTIZE_BITS': None}


def iter_archives(prefix: str):
    """Yield (meta, params, data) for every completed predict_molecules__hdc archive with the given prefix."""
    pattern = os.path.join(RESULTS, 'predict_molecules__hdc', '*', 'experiment_meta.json')
    for meta_path in glob.glob(pattern):
        try:
            meta = json.load(open(meta_path))
        except Exception:
            continue
        params = {k: v.get('value') for k, v in meta.get('parameters', {}).items() if isinstance(v, dict)}
        # a killed or timed-out run keeps status 'running' (and has_error False), so require 'done'
        if params.get('__PREFIX__') != prefix or meta.get('status') != 'done' or meta.get('has_error'):
            continue
        data_path = os.path.join(os.path.dirname(meta_path), 'experiment_data.json')
        try:
            data = json.load(open(data_path))
        except Exception:
            continue
        yield meta, params, data


def collect(prefix: str) -> tuple:
    """{(dataset, seed): {arm: record}}; a retried run keeps the newest archive. Also the skipped archives."""
    runs, skipped = defaultdict(dict), []
    archives = sorted(iter_archives(prefix), key=lambda t: t[0].get('start_time') or 0)
    for meta, params, data in archives:
        wrong = {k: params.get(k) for k, v in REQUIRED.items() if params.get(k) != v}
        if wrong or 'SPECTRUM' not in params or 'GRAPH_ATTRIBUTES' not in params:
            skipped.append((meta.get('name'), wrong))
            continue
        metrics = data.get('metrics', {}).get(f'test_{MODEL}')
        if metrics is None:
            skipped.append((meta.get('name'), 'no MLP test metrics'))
            continue
        arm = (params['SPECTRUM'], bool(params['GRAPH_ATTRIBUTES']))
        runs[(params['NOTE'], params['SEED'])][arm] = {
            'mae': metrics['mae'], 'r2': metrics['r2'],
            'effective_components': data.get('embedding', {}).get('effective_components', float('nan')),
        }
    return runs, skipped


def holm(pvalues: list) -> list:
    """Holm-adjusted p-values (NaN entries stay NaN and do not count)."""
    p = np.array(pvalues, dtype=float)
    valid = np.flatnonzero(~np.isnan(p))
    order = valid[np.argsort(p[valid])]
    adjusted = np.full_like(p, np.nan)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(order) - rank) * p[i]))
        adjusted[i] = running
    return adjusted.tolist()


def main(prefix: str):
    runs, skipped = collect(prefix)
    datasets = [d for d in DATASET_ORDER if any(k[0] == d for k in runs)]
    print(f'prefix {prefix}: {sum(len(v) for v in runs.values())} runs, {len(skipped)} archives skipped')
    for name, reason in skipped[:10]:
        print(f'  skipped {name}: {reason}')

    # --- per-arm summary ---
    arm_lines = ['| Dataset | n | ' + ' | '.join(f'MAE {ARM_LABEL[a]}' for a in ARMS) + ' | eff. components |',
                 '|' + '---|' * (len(ARMS) + 3)]
    cells, records = [], []
    for dataset in datasets:
        seeds = sorted(s for (d, s), v in runs.items() if d == dataset and all(a in v for a in ARMS))
        if not seeds:
            continue
        mae = {a: np.array([runs[(dataset, s)][a]['mae'] for s in seeds]) for a in ARMS}
        components = {a: np.nanmedian([runs[(dataset, s)][a]['effective_components'] for s in seeds]) for a in ARMS}
        arm_lines.append(f'| {DATASET_LABEL.get(dataset, dataset)} | {len(seeds)} | '
                         + ' | '.join(f'{mae[a].mean():.4g} ± {mae[a].std(ddof=1):.2g}' for a in ARMS) + ' | '
                         + ' / '.join(f'{components[a]:.0f}' for a in ARMS) + ' |')
        for arm in ARMS[1:]:
            diff = mae[arm] - mae[ORIGINAL]
            rel = 100 * diff / mae[ORIGINAL].mean()
            half = t_dist.ppf(0.975, len(seeds) - 1) * rel.std(ddof=1) / np.sqrt(len(seeds))
            p = wilcoxon(mae[arm], mae[ORIGINAL]).pvalue if len(seeds) >= 5 and np.any(diff != 0) else float('nan')
            cells.append({'dataset': dataset, 'arm': arm, 'n': len(seeds), 'rel': rel.mean(), 'half': half,
                          'better': int((diff < 0).sum()), 'p': p})
        for s in seeds:
            for a in ARMS:
                records.append([dataset, s, a[0], a[1], runs[(dataset, s)][a]['mae'], runs[(dataset, s)][a]['r2'],
                                runs[(dataset, s)][a]['effective_components']])

    for arm in ARMS[1:]:
        family = [c for c in cells if c['arm'] == arm]
        for cell, adjusted in zip(family, holm([c['p'] for c in family])):
            cell['p_holm'] = adjusted
    diff_lines = ['| Dataset | Arm vs. original | ΔMAE (mean, 95% CI) | better in | Wilcoxon p | Holm p |',
                  '|' + '---|' * 6]
    for c in cells:
        diff_lines.append(f'| {DATASET_LABEL.get(c["dataset"], c["dataset"])} | {ARM_LABEL[c["arm"]]} | '
                          f'{c["rel"]:+.1f}% [{c["rel"] - c["half"]:+.1f}, {c["rel"] + c["half"]:+.1f}] | '
                          f'{c["better"]}/{c["n"]} | {c["p"]:.3f} | {c["p_holm"]:.3f} |')

    text = ('MLP test MAE (mean ± SD over seeds) per arm; effective Fourier components of the embeddings '
            '(median over seeds) in the arm order of the MAE columns\n\n' + '\n'.join(arm_lines)
            + '\n\nPaired change of the test MAE relative to the original arm (negative = lower error); '
            'Holm per contrast over the datasets\n\n' + '\n'.join(diff_lines))
    print()
    print(text)
    if not records:
        print('no complete seeds found, existing outputs are left unchanged')
        return
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f'unitspec_comparison_{prefix}.md'), 'w') as f:
        f.write(text + '\n')
    with open(os.path.join(OUT, f'unitspec_comparison_{prefix}.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['dataset', 'seed', 'spectrum', 'graph_attributes', 'mae', 'r2', 'effective_components'])
        writer.writerows(records)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'ex_22_unitspec')
