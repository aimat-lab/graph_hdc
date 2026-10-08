"""
Analysis of Experiment 15 (ex_15): HDF with one-directional vs. bidirectional message passing.

Pairs the runs of ``_slurm_ex_15.py`` by (dataset, downstream model, seed) and compares the test MAE of
``BIDIRECTIONAL=False`` (old behavior) with ``BIDIRECTIONAL=True`` (fixed). Both arms use the same split per
seed, so the paired difference isolates the effect of the edge direction.

    python analyze_ex_15.py [prefix]      # default prefix: ex_15_bidir_t2 (ex_15_bidir: cancelled 8-thread runs)

Writes ``_ex15/bidir_comparison_<prefix>.md`` (table) and ``.csv`` (all paired records).
"""
import os
import sys
import csv
from collections import defaultdict

import numpy as np
from scipy.stats import wilcoxon

from analyze_ex_14 import iter_archives, DATASET_ORDER, DATASET_LABEL

PATH = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(PATH, '_ex15')
MODELS = ['neural_net2', 'k_neighbors']
MODEL_LABEL = {'neural_net2': 'MLP', 'k_neighbors': 'KNN'}


def collect(prefix: str) -> dict:
    """{(dataset, model, seed): {False: (mae, r2), True: (mae, r2)}}; a retried run keeps the newest archive."""
    runs = defaultdict(dict)
    archives = sorted(iter_archives(prefix), key=lambda t: t[1].get('start_time') or 0)
    for module, meta, params, data in archives:
        if module != 'predict_molecules__hdc' or 'BIDIRECTIONAL' not in params:
            continue
        metrics = data.get('metrics', {})
        for model in MODELS:
            key = f'test_{model}'
            if key in metrics:
                runs[(params['NOTE'], model, params['SEED'])][bool(params['BIDIRECTIONAL'])] = (
                    metrics[key]['mae'], metrics[key]['r2'])
    return runs


def main(prefix: str):
    runs = collect(prefix)
    os.makedirs(OUT, exist_ok=True)
    datasets = [d for d in DATASET_ORDER if any(k[0] == d for k in runs)]
    rows, records = [], []
    for dataset in datasets:
        for model in MODELS:
            seeds = sorted(s for (d, m, s), v in runs.items() if d == dataset and m == model and len(v) == 2)
            if not seeds:
                continue
            old = np.array([runs[(dataset, model, s)][False][0] for s in seeds])
            new = np.array([runs[(dataset, model, s)][True][0] for s in seeds])
            r2_old = np.array([runs[(dataset, model, s)][False][1] for s in seeds])
            r2_new = np.array([runs[(dataset, model, s)][True][1] for s in seeds])
            diff = new - old
            p = wilcoxon(new, old).pvalue if len(seeds) >= 5 and np.any(diff != 0) else float('nan')
            rows.append([
                DATASET_LABEL.get(dataset, dataset), MODEL_LABEL[model], len(seeds),
                f'{old.mean():.4g} ± {old.std():.2g}', f'{new.mean():.4g} ± {new.std():.2g}',
                f'{100 * diff.mean() / old.mean():+.1f}%', f'{int((diff < 0).sum())}/{len(seeds)}',
                f'{p:.3f}', f'{r2_old.mean():.3f} → {r2_new.mean():.3f}',
            ])
            for s, a, b in zip(seeds, old, new):
                records.append([dataset, model, s, a, b])

    header = ['Dataset', 'Model', 'n', 'MAE one-directional', 'MAE bidirectional', 'ΔMAE (mean)',
              'bidir. better', 'Wilcoxon p', 'R² old → new']
    lines = ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
    lines += ['| ' + ' | '.join(str(c) for c in row) + ' |' for row in rows]
    table = '\n'.join(lines)
    print(f'prefix {prefix}: {len(records)} paired runs\n')
    print(table)

    with open(os.path.join(OUT, f'bidir_comparison_{prefix}.md'), 'w') as f:
        f.write(table + '\n')
    with open(os.path.join(OUT, f'bidir_comparison_{prefix}.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['dataset', 'model', 'seed', 'mae_one_directional', 'mae_bidirectional'])
        writer.writerows(records)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'ex_15_bidir_t2')
