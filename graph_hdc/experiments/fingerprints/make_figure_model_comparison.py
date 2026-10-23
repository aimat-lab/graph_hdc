"""
Figure 2 of the main text (ex_28): the model comparison. Per target (one panel each) and model type (gradient boosting,
KNN, random forest, neural network), the test R2 of HDF, Morgan and RDKit fingerprints with the fixed standard
configuration and after tuning with the same budget, both in the colour of the representation (user, 2026-10-23): the
standard bar carries a black hatch and stands in front of the plain tuned bar, so that the plain part above it is the
gain from tuning (a tuned result below the standard one shows as a line inside the hatched bar). No error bars (user,
2026-10-23): the standard deviations over the 10 repetitions are at most 0.04 R2. Style of the
paper's figures (Roboto Condensed, panel letters, y-grid), cropped tightly so that it fills the text width.

    python make_figure_model_comparison.py <fig2_runs.csv of analyze_ex_28.py> <output .pdf> [--png]
"""
import sys
import csv
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

FONT_SIZE = 11   # user, 2026-10-23: slightly smaller than 12 (prints at about 8 pt)
TARGETS = [('clogp', "Crippen's logP"), ('aqsoldb_logs', 'AqSolDB logS'), ('qm9_energy', 'QM9 $U_0$'),
           ('qm9_gap', 'QM9 HOMO–LUMO gap')]
MODELS = ['GB', 'KNN', 'RF', 'NN']
METHODS = [('hdf', 'HDF', '#4CEB99'), ('morgan', 'Morgan', '#4C64EB'), ('rdkit', 'RDKit', '#A14BEB')]
WIDTH = 0.26
# the black hatch of the standard configuration (pattern and line width in points before scaling to the text width)
HATCH = '/////'
# user, 2026-10-23: '////' at 0.9 was too faint, '///' at 1.6 too thick; denser thin lines instead
HATCH_LINEWIDTH = 0.9


def darker(hex_color: str, factor: float = 0.55) -> tuple:
    rgb = matplotlib.colors.to_rgb(hex_color)
    return tuple(c * factor for c in rgb)


def load(path: str) -> dict:
    values = defaultdict(list)
    for row in csv.DictReader(open(path)):
        if row['r2'] not in ('', 'None'):
            values[(row['target'], row['method'], row['model'], row['setting'])].append(float(row['r2']))
    return {key: (np.mean(v), np.std(v, ddof=1) if len(v) > 1 else 0.0, len(v)) for key, v in values.items()}


def main(path: str, out: str, png: bool):
    plt.style.use('default')
    plt.rcParams.update({'font.family': 'Roboto Condensed', 'font.size': FONT_SIZE, 'hatch.linewidth': HATCH_LINEWIDTH})
    stats = load(path)
    # 7.4 in wide (about 7 in after cropping): at the full text width of the main text (about 372 pt) the fonts print
    # at about 8.5 pt, as in Figure 3
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.4), sharey=True)
    lows = [m for (m, _s, _n) in stats.values()] or [0.0]
    bottom = min(0.0, np.floor(min(lows) * 10) / 10)
    for k, (ax, (target, title)) in enumerate(zip(axes.flat, TARGETS)):
        x = np.arange(len(MODELS))
        for j, (method, _label, color) in enumerate(METHODS):
            pos = x + (j - 1) * (WIDTH + 0.02)
            for setting in ('fixed', 'tuned'):
                means = np.array([stats.get((target, method, model, setting), (np.nan, np.nan, 0))[0]
                                  for model in MODELS])
                if setting == 'fixed':
                    # the standard bar in front: the colour of the representation with a black hatch on top
                    ax.bar(pos, means, WIDTH, color=color, edgecolor=darker(color), linewidth=0.8, zorder=3)
                    ax.bar(pos, means, WIDTH, facecolor='none', edgecolor='black', hatch=HATCH, linewidth=0,
                           zorder=4)
                else:
                    # the plain tuned bar behind it: its part above the standard bar is the gain from tuning
                    ax.bar(pos, means, WIDTH, color=color, edgecolor=darker(color), linewidth=0.8, zorder=2)
                    # its outline on top, so that a tuned result below the standard one stays visible as a line
                    ax.bar(pos, means, WIDTH, facecolor='none', edgecolor=darker(color), linewidth=0.8, zorder=5)
        ax.set_xticks(x)
        ax.set_xlim(-0.5, len(MODELS) - 0.5)
        ax.set_xticklabels(MODELS)
        ax.set_title(title, fontsize=FONT_SIZE + 1)
        ax.grid(axis='y', alpha=0.3, zorder=0)
        ax.set_axisbelow(True)
        ax.set_ylim(bottom, 1.05)
        ax.set_yticks(np.arange(bottom, 1.001, 0.25))
        if k % 2 == 0:
            ax.set_ylabel('$R^2$ $\\uparrow$', fontsize=FONT_SIZE + 1)
        ax.text(-0.13 if k % 2 == 0 else -0.06, 1.06, 'abcd'[k], transform=ax.transAxes, fontsize=FONT_SIZE + 4,
                fontweight='bold', va='bottom', ha='left')
    handles = [Patch(facecolor=color, edgecolor=color, label=label) for _m, label, color in METHODS]
    handles += [Patch(facecolor='#B0B0B0', edgecolor='black', hatch=HATCH, linewidth=0.5,
                      label='standard configuration'),
                Patch(facecolor='#B0B0B0', edgecolor='#606060', linewidth=0.5, label='tuned')]
    fig.legend(handles=handles, loc='upper center', ncol=5, frameon=True, bbox_to_anchor=(0.5, 1.0),
               handlelength=1.6, handletextpad=0.5, columnspacing=1.1,
               fontsize=FONT_SIZE)
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=1.2, w_pad=0.8)
    fig.savefig(out, bbox_inches='tight', pad_inches=0.02)
    if png:
        fig.savefig(out.rsplit('.', 1)[0] + '.png', dpi=200, bbox_inches='tight', pad_inches=0.02)
    missing = [(t, m, model, s) for t, _ in TARGETS for m, *_ in METHODS for model in MODELS
               for s in ('fixed', 'tuned') if (t, m, model, s) not in stats]
    print(f'wrote {out}; {len(missing)} bars without data' + (f', e.g. {missing[:3]}' if missing else ''))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], '--png' in sys.argv)
