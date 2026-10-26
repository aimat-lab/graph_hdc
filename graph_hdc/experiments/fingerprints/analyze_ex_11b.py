"""
Analysis of ex_11_b, the re-run of the similarity-based virtual screening benchmark (``_slurm_ex_11b.py``), next to
the original ex_11 runs (prefix ex_11_bioact).

Arms: Morgan (ECFP4) and HDF without (main arm) and with (reference) the graph size and diameter encodings, all with
the standard HDF configuration of the revision; the ex_11 runs give the original Morgan and HDF results.

Outputs in ``_ex11b/analysis/`` (``_ex11b/analysis_smoke/`` with --smoke):
- ``summary.csv`` / ``summary.md``: mean over targets of AUC, BEDROC, EF1% and EF5% per dataset and arm.
- ``paired.md``: per dataset, HDF arm vs. Morgan over the targets (number of targets where HDF is better, two-sided
  Wilcoxon signed-rank test), since all runs evaluate the same targets with the same query actives (seed 1).
- ``morgan_reproduction.md``: largest per-target difference between the Morgan runs of ex_11_b and ex_11.

    python analyze_ex_11b.py [--smoke]
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

PATH = os.path.dirname(os.path.abspath(__file__))
DATASETS = ['riniker_1', 'riniker_2', 'muv', 'dud_e']
METRICS = ['auc', 'bedroc', 'ef1', 'ef5']
ARMS = ['fp_ex11', 'fp', 'hdc_ex11', 'hdc_global', 'hdc_noglobal']
LABELS = {
    'fp_ex11': 'Morgan (ex_11)', 'fp': 'Morgan', 'hdc_ex11': 'HDF (ex_11, original encoder)',
    'hdc_global': 'HDF with size/diameter', 'hdc_noglobal': 'HDF without size/diameter',
}


def archives(prefix: str) -> pd.DataFrame:
    """One row per finished archive of the given prefix: dataset, arm, path, end time."""
    rows = []
    for module in ('fp', 'hdc'):
        for path in glob.glob(os.path.join(PATH, 'results', f'predict_bioactivity__{module}', f'{prefix}__*')):
            meta_path = os.path.join(path, 'experiment_meta.json')
            if not os.path.exists(meta_path) or not os.path.exists(os.path.join(path, 'per_target_results.csv')):
                continue
            meta = json.load(open(meta_path))
            params = {k: v.get('value') for k, v in meta['parameters'].items() if isinstance(v, dict)}
            if params.get('__PREFIX__') != prefix:
                continue
            if prefix.startswith('ex_11_bioact'):
                arm = f'{module}_ex11'
            elif module == 'fp':
                arm = 'fp'
            else:
                arm = 'hdc_global' if params.get('GRAPH_ATTRIBUTES', True) else 'hdc_noglobal'
            rows.append({'dataset': params['DATASET_NAME'], 'arm': arm, 'path': path,
                         'end': meta.get('end_time') or 0, 'status': meta.get('status')})
    return pd.DataFrame(rows)


def main(smoke: bool = False):
    out = os.path.join(PATH, '_ex11b', 'analysis_smoke' if smoke else 'analysis')
    os.makedirs(out, exist_ok=True)
    found = pd.concat([archives('ex_11_bioact'), archives('ex_11_b_smoke' if smoke else 'ex_11_b')])
    duplicates = found.groupby(['dataset', 'arm']).size()
    for (dataset, arm), n in duplicates[duplicates > 1].items():
        print(f'WARNING: {n} archives for {dataset}/{arm}; using the latest')
    found = found.sort_values('end').groupby(['dataset', 'arm']).tail(1)

    per_target, summary = {}, []
    for row in found.itertuples():
        frame = pd.read_csv(os.path.join(row.path, 'per_target_results.csv'))
        per_target[(row.dataset, row.arm)] = frame.set_index('target_id')
        summary.append({'dataset': row.dataset, 'arm': row.arm, 'targets': len(frame),
                        **{m: frame[f'{m}_mean'].mean() for m in METRICS}, 'archive': os.path.basename(row.path)})
    summary = pd.DataFrame(summary)
    summary['arm'] = pd.Categorical(summary['arm'], ARMS, ordered=True)
    summary = summary.sort_values(['dataset', 'arm'])
    summary.to_csv(os.path.join(out, 'summary.csv'), index=False)

    lines = ['Mean over targets (each target: mean over 50 repetitions x 5 query actives).', '']
    for dataset in [d for d in DATASETS if d in set(summary['dataset'])]:
        part = summary[summary['dataset'] == dataset]
        lines += [f'**{dataset}** ({int(part["targets"].max())} targets)', '',
                  '| Arm | AUC | BEDROC | EF1% | EF5% |', '|---|---|---|---|---|']
        for r in part.itertuples():
            lines.append(f'| {LABELS[r.arm]} | {r.auc:.3f} | {r.bedroc:.3f} | {r.ef1:.2f} | {r.ef5:.2f} |')
        lines.append('')
    open(os.path.join(out, 'summary.md'), 'w').write('\n'.join(lines))
    print('\n'.join(lines))

    lines = ['HDF arm vs. Morgan of the same run over the targets: targets where HDF is better / all, mean difference '
             '(HDF - Morgan) and two-sided Wilcoxon signed-rank p.', '']
    for dataset in DATASETS:
        for arm, reference in (('hdc_noglobal', 'fp'), ('hdc_global', 'fp'), ('hdc_noglobal', 'hdc_global'),
                               ('hdc_ex11', 'fp_ex11')):
            if (dataset, arm) not in per_target or (dataset, reference) not in per_target:
                continue
            a, b = per_target[(dataset, arm)], per_target[(dataset, reference)]
            common = a.index.intersection(b.index)
            cells = []
            for m in METRICS:
                x, y = a.loc[common, f'{m}_mean'], b.loc[common, f'{m}_mean']
                diff = x - y
                p = wilcoxon(x, y).pvalue if len(common) > 1 and np.any(diff != 0) else float('nan')
                cells.append(f'{m}: {int((diff > 0).sum())}/{len(common)}, {diff.mean():+.3f}, p={p:.2g}')
            lines.append(f'- {dataset}, {LABELS[arm]} vs. {LABELS[reference]}: ' + '; '.join(cells))
    open(os.path.join(out, 'paired.md'), 'w').write('\n'.join(lines))
    print('\n'.join(lines))

    lines = ['Morgan, ex_11_b vs. ex_11: largest absolute per-target difference of each metric (0 = reproduced).', '']
    for dataset in DATASETS:
        if (dataset, 'fp') in per_target and (dataset, 'fp_ex11') in per_target:
            a, b = per_target[(dataset, 'fp')], per_target[(dataset, 'fp_ex11')]
            same_targets = a.index.equals(b.index)
            diffs = {m: float(np.max(np.abs(a[f'{m}_mean'] - b.loc[a.index, f'{m}_mean']))) for m in METRICS}
            lines.append(f'- {dataset}: same targets {same_targets}; '
                         + ', '.join(f'{m} {d:.2e}' for m, d in diffs.items()))
    open(os.path.join(out, 'morgan_reproduction.md'), 'w').write('\n'.join(lines))
    print('\n'.join(lines))


if __name__ == '__main__':
    main(smoke='--smoke' in sys.argv)
