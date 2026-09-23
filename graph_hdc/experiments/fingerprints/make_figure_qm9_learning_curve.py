"""
Aggregate the HDC + neural-network QM9 atomization-energy runs into a learning
curve and plot it on the same log-log axes as the classic QM9 figure
(out-of-sample MAE vs. number of training molecules).

Each experiment archive stores the test metrics in ``experiment_data.json`` and
prints the actual training-set size in its log (``train: N, val: ..., test: ...``).
We key the aggregation on that logged N because the pycomex parameter snapshot
does not persist the concrete parameter values. Results are grouped by training
size and reduced to mean +/- std over the random seeds.

Usage::

    python make_figure_qm9_learning_curve.py
"""
import os
import re
import json
import glob
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

# 1 kcal/mol in eV; the qm9_smiles atomization-energy target (idx 15) is in kcal/mol.
KCAL2EV = 0.0433641

# Notable QM9 methods hand-digitized from the provided Fig. 2 (Ramakrishnan/Faber
# et al. learning curves). These are eyeballed off the log-log plot against its
# gridlines (0.005/0.025/0.06/0.15/0.4 eV; 100/1k/10k/100k) and are approximate
# (~10-20%) -- intended as visual context, not exact reference values. Each entry
# maps method -> (dict(style), [(N_train, MAE_eV), ...]).
FIG2_METHODS = {
    'CM  (Coulomb matrix)': (dict(color='0.55', marker='*', ms=7), [
        (12000, 0.28), (20000, 0.20), (50000, 0.16), (100000, 0.135), (130000, 0.12)]),
    'DTNN': (dict(color='tab:blue', marker='x', ms=6), [
        (30000, 0.043), (60000, 0.040), (100000, 0.037)]),
    'SchNet': (dict(color='0.35', marker='h', ms=6), [
        (10000, 0.055), (20000, 0.037), (50000, 0.022), (100000, 0.016), (130000, 0.015)]),
    'FCHL': (dict(color='black', marker='o', ms=5), [
        (100, 0.30), (1000, 0.062), (2000, 0.043), (5000, 0.026), (10000, 0.019), (16000, 0.016)]),
    'SOAP': (dict(color='goldenrod', marker='D', ms=5), [
        (100, 0.38), (1000, 0.12), (5000, 0.050), (10000, 0.030), (30000, 0.016), (70000, 0.0075)]),
}

_HERE = os.path.dirname(os.path.abspath(__file__))

# Our two same-pipeline representations: (label, results glob, plot style).
THIS_WORK = [
    ('HDC (2D graph) + NN  [this work]',
     os.path.join(_HERE, 'results', 'predict_molecules__hdc__qm9_atomization', 'ex_12_qm9_ae_*'),
     dict(color='tab:red', marker='o')),
    ('Morgan FP + NN  [this work]',
     os.path.join(_HERE, 'results', 'predict_molecules__fp__qm9_atomization', 'ex_12_qm9_ae_fp_*'),
     dict(color='tab:orange', marker='s')),
]
FIG_DIR = os.path.join(_HERE, 'figures')

_TRAIN_RE = re.compile(r'train:\s*(\d+),\s*val:')


def train_size_from_log(archive: str) -> int | None:
    """Read the actual training-set size that a run used from its log."""
    log_path = os.path.join(archive, 'experiment_out.log')
    if not os.path.exists(log_path):
        return None
    with open(log_path) as f:
        for line in f:
            m = _TRAIN_RE.search(line)
            if m:
                return int(m.group(1))
    return None


def mae_from_archive(archive: str) -> float | None:
    """Read the test-set MAE (native units, kcal/mol) from an archive."""
    dj = os.path.join(archive, 'experiment_data.json')
    if not os.path.exists(dj):
        return None
    data = json.load(open(dj))
    metrics = data.get('metrics', {})
    key = next((k for k in metrics if k.startswith('test_')), None)
    if key is None or 'mae' not in metrics[key]:
        return None
    return float(metrics[key]['mae'])


def collect(results_glob: str) -> dict[int, list[float]]:
    by_size: dict[int, list[float]] = defaultdict(list)
    for archive in glob.glob(results_glob):
        n = train_size_from_log(archive)
        mae = mae_from_archive(archive)
        if n is None or mae is None:
            print(f'  ! skipping (incomplete): {os.path.basename(archive)}')
            continue
        by_size[n].append(mae)
    return dict(sorted(by_size.items()))


def main() -> None:
    # Aggregate every "this work" representation and print a comparison table.
    curves = []  # (label, style, sizes, means_ev, stds_ev)
    print('\nQM9 atomization-energy learning curves  [this work]')
    for label, results_glob, style in THIS_WORK:
        by_size = collect(results_glob)
        if not by_size:
            print(f'  (no archives yet for: {label})')
            continue
        sizes = np.array(sorted(by_size))
        means = np.array([np.mean(by_size[n]) for n in sizes])            # kcal/mol
        stds = np.array([np.std(by_size[n], ddof=1) if len(by_size[n]) > 1 else 0.0
                         for n in sizes])
        counts = [len(by_size[n]) for n in sizes]
        print(f'\n  {label}')
        print(f"  {'N_train':>8} {'seeds':>6} {'MAE kcal/mol':>14} {'MAE eV':>14}")
        for n, m, s, c in zip(sizes, means, stds, counts):
            print(f'  {n:>8} {c:>6} {m:>8.2f}±{s:<5.2f} {m*KCAL2EV:>7.3f}±{s*KCAL2EV:<6.3f}')
        curves.append((label, style, sizes, means * KCAL2EV, stds * KCAL2EV))

    if not curves:
        raise SystemExit('No completed archives found for any representation.')

    fig, ax = plt.subplots(figsize=(9.5, 6))

    # Notable published methods, hand-digitized from the provided Fig. 2 (thin
    # reference lines). Approximate -- see FIG2_METHODS note.
    for label, (style, pts) in FIG2_METHODS.items():
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, lw=1.2, alpha=0.75, ls='-', **style, label=label)

    # 1 kcal/mol chemical-accuracy line.
    ax.axhline(KCAL2EV, color='0.4', ls=':', lw=1.2,
               label='1 kcal/mol (chem. accuracy)')

    # This work: our same-pipeline representations (bold).
    for label, style, sizes, means_ev, stds_ev in curves:
        ax.errorbar(sizes, means_ev, yerr=stds_ev, ms=8, lw=2.6,
                    capsize=4, zorder=10, label=label, **style)
        for x, y in zip(sizes, means_ev):
            ax.annotate(f'{y:.2f}', (x, y), textcoords='offset points',
                        xytext=(7, 7), fontsize=7.5, color=style['color'])

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(80, 1.5e5)
    ax.set_ylim(5e-3, 9.0)
    ax.set_xlabel('Number of training molecules')
    ax.set_ylabel('Test MAE of atomization energy (eV)')
    ax.set_title('QM9 atomization energy learning curves\n'
                 'HDC vs. Morgan-FP (same NN pipeline) vs. notable methods (Fig. 2, digitized)',
                 fontsize=12)
    ax.grid(True, which='both', ls='--', alpha=0.3)
    ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=8,
              title='Method', title_fontsize=9)

    # Footnote on provenance of the reference curves.
    fig.text(0.01, 0.005,
             'Reference curves hand-digitized from the provided Fig. 2 (approximate, '
             '~10-20%); every reference method uses 3D geometry.',
             fontsize=7, color='0.35')
    fig.tight_layout()

    os.makedirs(FIG_DIR, exist_ok=True)
    for ext in ('svg', 'png'):
        out = os.path.join(FIG_DIR, f'qm9_atomization_learning_curve_hdc.{ext}')
        fig.savefig(out, dpi=200)
        print(f'saved {out}')


if __name__ == '__main__':
    main()
