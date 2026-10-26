"""
Figure 4 of the main text (ex_29): the test MAE of the MLP of Table 1 over the embedding dimension for HDF (depth 1
and 3) and Morgan fingerprints (radius 1 and 3) on (a) AqSolDB and (b) BACE: the mean over the 10 repetitions with a
band of one standard deviation. Paper style (Roboto Condensed, panel letters, dashed grid for line plots, markers with
black edges, log dimension axis with unlabeled minor ticks), sized for the full text width of the main text (fonts
print at about 8 pt, as Figure 2).

    python make_figure_size.py <fig4_runs.csv of analyze_ex_29.py> <output .pdf> [--png]
"""
import sys
import csv
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

FONT_SIZE = 11
TARGETS = [('aqsoldb_logs', 'AqSolDB (logS)'), ('bace_ic50', 'BACE (pIC50)')]
# (method, depth, label, colour, line style, marker): the lighter colour for depth / radius 1, as in the paper
SERIES = [('morgan', 1, 'Morgan (radius 1)', '#7C92FF', '--', 'o'),
          ('morgan', 3, 'Morgan (radius 3)', '#4C64EB', '-', 's'),
          ('hdf', 1, 'HDF (depth 1)', '#69F7AE', '--', 'o'),
          ('hdf', 3, 'HDF (depth 3)', '#4CEB99', '-', 's')]


def load(path: str) -> dict:
    """{(target, method, depth): {dimension: [mae of every seed]}}"""
    values = defaultdict(lambda: defaultdict(list))
    for row in csv.DictReader(open(path)):
        if row['mae'] not in ('', 'None'):
            values[(row['target'], row['method'], int(row['depth']))][int(row['dimension'])].append(float(row['mae']))
    return values


def main(path: str, out: str, png: bool):
    plt.style.use('default')
    plt.rcParams.update({'font.family': 'Roboto Condensed', 'font.size': FONT_SIZE})
    values = load(path)
    # 7.4 in wide: at the full text width of the main text (about 372 pt) the fonts print at about 8 pt
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9))
    for k, (ax, (target, title)) in enumerate(zip(axes, TARGETS)):
        for method, depth, _label, color, style, marker in SERIES:
            series = values.get((target, method, depth), {})
            dims = np.array(sorted(series))
            if not len(dims):
                continue
            mean = np.array([np.mean(series[d]) for d in dims])
            std = np.array([np.std(series[d], ddof=1) if len(series[d]) > 1 else 0.0 for d in dims])
            ax.fill_between(dims, mean - std, mean + std, color=color, alpha=0.2, linewidth=0, zorder=2)
            ax.plot(dims, mean, color=color, linestyle=style, linewidth=1.6, marker=marker, markersize=4.5,
                    markeredgecolor='black', markeredgewidth=0.6, zorder=3)
        ax.set_xscale('log')
        ax.set_title(title, fontsize=FONT_SIZE + 1)
        ax.set_xlabel('Embedding dimension', fontsize=FONT_SIZE + 1)
        # both panels have their own scale, so both get the y-label (Pascal, 2026-10-20)
        ax.set_ylabel('Test MAE $\\downarrow$', fontsize=FONT_SIZE + 1)
        ax.grid(True, which='major', linestyle='--', alpha=0.3)
        ax.set_axisbelow(True)
        ax.text(-0.14, 1.06, 'ab'[k], transform=ax.transAxes, fontsize=FONT_SIZE + 4,
                fontweight='bold', va='bottom', ha='left')
    handles = [Line2D([], [], color=color, linestyle=style, linewidth=1.6, marker=marker, markersize=4.5,
                      markeredgecolor='black', markeredgewidth=0.6, label=label)
               for _m, _d, label, color, style, marker in SERIES]
    fig.legend(handles=handles, loc='upper center', ncol=4, frameon=True, bbox_to_anchor=(0.5, 1.0),
               handlelength=2.4, columnspacing=1.4, fontsize=FONT_SIZE)
    fig.tight_layout(rect=(0, 0, 1, 0.87), w_pad=1.5)
    fig.savefig(out, bbox_inches='tight', pad_inches=0.02)
    if png:
        fig.savefig(out.rsplit('.', 1)[0] + '.png', dpi=200, bbox_inches='tight', pad_inches=0.02)
    counts = sorted({len(v) for series in values.values() for v in series.values()})
    print(f'wrote {out}; seeds per point: {counts}')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], '--png' in sys.argv)
