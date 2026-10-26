"""
SI Figure S1 (ex_29): the test MAE of the scikit-learn MLP over the number of training molecules for HDF (depth 1-3)
and Morgan fingerprints (radius 1-3) at 2048 dimensions on four QM9 targets (U0, HOMO-LUMO gap, heat capacity Cv,
dipole moment): the mean over the 10 repetitions with a band of one standard deviation, log-log axes. Paper style
(Roboto Condensed, panel letters, dashed grid for line plots, markers with black edges, unlabeled minor ticks).

    python make_figure_training_size.py <s1_runs.csv of analyze_ex_29.py> <output .pdf> [--png]
"""
import sys
import csv
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import LogLocator, MaxNLocator, NullFormatter, FuncFormatter  # noqa: E402

FONT_SIZE = 11
TARGETS = [('qm9_energy', 'QM9 $U_0$'), ('qm9_gap', 'QM9 HOMO–LUMO gap'), ('qm9_cv', 'QM9 $C_v$'),
           ('qm9_dipole', 'QM9 dipole moment $\\mu$')]
# (method, depth, label, colour, line style, marker): light to dark with the depth / radius
SERIES = [('morgan', 1, 'Morgan (radius 1)', '#A9B8FF', ':', 'o'),
          ('morgan', 2, 'Morgan (radius 2)', '#7C92FF', '--', 's'),
          ('morgan', 3, 'Morgan (radius 3)', '#4C64EB', '-', 'D'),
          ('hdf', 1, 'HDF (depth 1)', '#A6FBD0', ':', 'o'),
          ('hdf', 2, 'HDF (depth 2)', '#69F7AE', '--', 's'),
          ('hdf', 3, 'HDF (depth 3)', '#2FD07C', '-', 'D')]


def load(path: str) -> dict:
    """{(target, method, depth): {num_train: [mae of every seed]}}"""
    values = defaultdict(lambda: defaultdict(list))
    for row in csv.DictReader(open(path)):
        if row['mae'] not in ('', 'None'):
            values[(row['target'], row['method'], int(row['depth']))][int(row['num_train'])].append(float(row['mae']))
    return values


def main(path: str, out: str, png: bool):
    plt.style.use('default')
    plt.rcParams.update({'font.family': 'Roboto Condensed', 'font.size': FONT_SIZE})
    values = load(path)
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.6))
    for k, (ax, (target, title)) in enumerate(zip(axes.flat, TARGETS)):
        for method, depth, _label, color, style, marker in SERIES:
            series = values.get((target, method, depth), {})
            sizes = np.array(sorted(series))
            if not len(sizes):
                continue
            mean = np.array([np.mean(series[n]) for n in sizes])
            std = np.array([np.std(series[n], ddof=1) if len(series[n]) > 1 else 0.0 for n in sizes])
            # the band in log space would need positive lower bounds: clip at a tenth of the mean
            ax.fill_between(sizes, np.maximum(mean - std, mean / 10), mean + std, color=color, alpha=0.2,
                            linewidth=0, zorder=2)
            ax.plot(sizes, mean, color=color, linestyle=style, linewidth=1.5, marker=marker, markersize=4,
                    markeredgecolor='black', markeredgewidth=0.5, zorder=3)
        ax.set_xscale('log')
        ax.set_yscale('log')
        # 1-2-5 labels (plain numbers) on the MAE axis unless it spans more than about 2.5 decades, and four evenly
        # spaced labels if fewer than two 1-2-5 values fall into a short range (matplotlib would otherwise fall back
        # to dense linear ticks); the minor ticks show the scale without labels
        low, high = ax.get_ylim()
        if np.log10(high / low) <= 2.5:
            candidates = [k * 10.0 ** e for e in range(-5, 5) for k in (1, 2, 5)]
            if sum(low <= c <= high for c in candidates) >= 2:
                ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
            else:
                ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _pos: f'{v:g}'))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_title(title, fontsize=FONT_SIZE + 1)
        if k >= 2:
            ax.set_xlabel('Number of training molecules', fontsize=FONT_SIZE + 1)
        # every panel has its own scale, so every panel gets the y-label (Pascal, 2026-10-20)
        ax.set_ylabel('Test MAE $\\downarrow$', fontsize=FONT_SIZE + 1)
        ax.grid(True, which='major', linestyle='--', alpha=0.3)
        ax.set_axisbelow(True)
        ax.text(-0.15, 1.06, 'abcd'[k], transform=ax.transAxes, fontsize=FONT_SIZE + 4,
                fontweight='bold', va='bottom', ha='left')
    handles = [Line2D([], [], color=color, linestyle=style, linewidth=1.5, marker=marker, markersize=4,
                      markeredgecolor='black', markeredgewidth=0.5, label=label)
               for _m, _d, label, color, style, marker in SERIES]
    # the legend fills its columns first: interleave so that the rows are Morgan (radius 1-3) and HDF (depth 1-3)
    handles = [h for pair in zip(handles[:3], handles[3:]) for h in pair]
    fig.legend(handles=handles, loc='upper center', ncol=3, frameon=True, bbox_to_anchor=(0.5, 1.0),
               handlelength=2.6, columnspacing=1.4, fontsize=FONT_SIZE)
    fig.tight_layout(rect=(0, 0, 1, 0.89), h_pad=1.2, w_pad=1.2)
    fig.savefig(out, bbox_inches='tight', pad_inches=0.02)
    if png:
        fig.savefig(out.rsplit('.', 1)[0] + '.png', dpi=200, bbox_inches='tight', pad_inches=0.02)
    counts = sorted({len(v) for series in values.values() for v in series.values()})
    print(f'wrote {out}; seeds per point: {counts}')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], '--png' in sys.argv)
