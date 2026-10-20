"""
Figure 2 of the main text (ex_28): the model comparison. Per target (one panel each) and model type (gradient boosting,
KNN, random forest, neural network), the test R2 of HDF, Morgan and RDKit fingerprints with the fixed standard
configuration (solid bar) and after tuning with the same budget (hatched outline drawn over it, so that the hatched
part above the solid bar is the gain from tuning; a tuned result below the fixed one ends inside the solid bar). Error
bars: standard deviation over the 10 repetitions. Style of the paper's figures (Roboto Condensed, panel letters,
y-grid).

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

FONT_SIZE = 12
TARGETS = [('clogp', "Crippen's logP"), ('aqsoldb_logs', 'AqSolDB logS'), ('qm9_energy', 'QM9 $U_0$'),
           ('qm9_gap', 'QM9 HOMO–LUMO gap')]
MODELS = ['GB', 'KNN', 'RF', 'NN']
METHODS = [('hdf', 'HDF', '#4CEB99'), ('morgan', 'Morgan', '#4C64EB'), ('rdkit', 'RDKit', '#A14BEB')]
WIDTH = 0.26


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
    plt.rcParams.update({'font.family': 'Roboto Condensed', 'font.size': FONT_SIZE, 'hatch.linewidth': 0.9})
    stats = load(path)
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.0), sharey=True)
    lows = [m - s for (m, s, _n) in stats.values()] or [0.0]
    bottom = min(0.0, np.floor(min(lows) * 10) / 10)
    for k, (ax, (target, title)) in enumerate(zip(axes.flat, TARGETS)):
        x = np.arange(len(MODELS))
        for j, (method, _label, color) in enumerate(METHODS):
            pos = x + (j - 1) * (WIDTH + 0.02)
            for setting in ('fixed', 'tuned'):
                means = np.array([stats.get((target, method, model, setting), (np.nan, np.nan, 0))[0]
                                  for model in MODELS])
                stds = np.array([stats.get((target, method, model, setting), (np.nan, np.nan, 0))[1]
                                 for model in MODELS])
                if setting == 'fixed':
                    # the solid fixed bar covers the overlap with the hatched tuned bar behind it
                    ax.bar(pos, means, WIDTH, color=color, edgecolor=color, linewidth=0.6, zorder=3)
                else:
                    # hatched tuned bar behind the fixed one: only the part above it (the gain) shows the hatching
                    ax.bar(pos, means, WIDTH, facecolor='white', edgecolor=darker(color), hatch='////',
                           linewidth=0, zorder=2)
                    # its outline on top, so that a tuned result below the fixed one stays visible as a line
                    ax.bar(pos, means, WIDTH, facecolor='none', edgecolor=darker(color), linewidth=1.0, zorder=4)
                ax.errorbar(pos + (-0.045 if setting == 'fixed' else 0.045), means, yerr=stds, fmt='none',
                            ecolor='black', elinewidth=0.9, capsize=2.5, capthick=0.9, zorder=5)
        ax.set_xticks(x)
        ax.set_xticklabels(MODELS)
        ax.set_title(title, fontsize=FONT_SIZE + 1)
        ax.grid(axis='y', alpha=0.3, zorder=0)
        ax.set_axisbelow(True)
        ax.set_ylim(bottom, 1.05)
        if k % 2 == 0:
            ax.set_ylabel('$R^2$ $\\uparrow$', fontsize=FONT_SIZE + 1)
        ax.text(-0.13 if k % 2 == 0 else -0.06, 1.06, 'abcd'[k], transform=ax.transAxes, fontsize=FONT_SIZE + 4,
                fontweight='bold', va='bottom', ha='left')
    handles = [Patch(facecolor=color, edgecolor=color, label=label) for _m, label, color in METHODS]
    handles += [Patch(facecolor='#B0B0B0', edgecolor='#B0B0B0', label='fixed configuration'),
                Patch(facecolor='none', edgecolor='#404040', hatch='////', label='tuned')]
    fig.legend(handles=handles, loc='upper center', ncol=5, frameon=True, bbox_to_anchor=(0.5, 1.0),
               fontsize=FONT_SIZE)
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=1.2, w_pad=0.8)
    fig.savefig(out, bbox_inches='tight')
    if png:
        fig.savefig(out.rsplit('.', 1)[0] + '.png', dpi=200, bbox_inches='tight')
    missing = [(t, m, model, s) for t, _ in TARGETS for m, *_ in METHODS for model in MODELS
               for s in ('fixed', 'tuned') if (t, m, model, s) not in stats]
    print(f'wrote {out}; {len(missing)} bars without data' + (f', e.g. {missing[:3]}' if missing else ''))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], '--png' in sys.argv)
