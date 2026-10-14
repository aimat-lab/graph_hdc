"""
Figures of the Bayesian optimization benchmark ex_24 (``bo_benchmark.py``) in the style of the original BO figure of
the paper (Fig. 4, made with analyze_ex_07.ipynb): Roboto Condensed, gray / light blue / light green, transparent
box plots with black medians, significance brackets, shaded progression bands, bold panel letters with
left-aligned titles.

- ``figure_bo``: main-text figure with Morgan fingerprints + Tanimoto kernel and HDF + Matern kernel only (user,
  2026-10-13). Top row: normalized regret per BO round at 16 and 1024 dimensions for ClogP (AqSolDB) and QED
  (ZINC250k), mean over the campaigns with the standard error as band. Bottom row: regret AUC against the embedding
  size for both objectives, box plots over the campaigns, random search as a dashed line, brackets for HDF against
  Morgan + Tanimoto (paired two-sided Wilcoxon signed-rank test, Holm-corrected within each objective as in
  analyze_ex_24.py).
- ``figure_bo_si``: box-plot panels of all nine objectives in a 3 x 3 grid (SI) with all four surrogates (Morgan +
  Tanimoto, Morgan PCA + Matern, HDF + Matern, HDF + linear kernel), at every second embedding size so that the
  panels stay readable (user, 2026-10-13).

Random search is explained by the legend only, without a label in every panel (user, 2026-10-13). Significance
brackets are black where HDF is better and red where Morgan + Tanimoto is better (user, 2026-10-14). Sizes without
data (e.g. 256 before its campaigns ran) are left out. Reads ``_ex24/campaigns.tsv``, ``_ex24/runs/`` and
``_ex24/analysis/tests.tsv`` (``bo_benchmark.py collect`` and ``analyze_ex_24.py``; EX24_ROOT points elsewhere).

    python make_figure_bo.py [output folder]
"""
import json
import os
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

PATH = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get('EX24_ROOT', os.path.join(PATH, '_ex24'))

SIZES = [8, 16, 32, 64, 128, 256, 512, 1024]
SI_SIZES = [8, 32, 128, 512, 1024]   # every second size plus the largest (user, 2026-10-13)
PROGRESSION_SIZES = [16, 1024]
# box order within a size: Morgan variants left, HDF right (as in the original figure)
MAIN_METHODS = ['morgan_tanimoto', 'hdf_matern']
SI_METHODS = ['morgan_tanimoto', 'morgan_pca', 'hdf_matern', 'hdf_linear']
COLORS = {
    'random': '#808080',
    'morgan_tanimoto': '#7C92FF',
    'morgan_pca': '#4C64EB',
    'hdf_matern': '#69F7AE',
    'hdf_linear': '#2DBE76',
}
LABELS = {
    'random': 'Random search',
    'morgan_tanimoto': 'Morgan FP (Tanimoto)',
    'morgan_pca': 'Morgan FP, PCA (Matérn)',
    'hdf_matern': 'HDF (Matérn)',
    'hdf_linear': 'HDF (linear)',
}
OBJECTIVES = ['clogp', 'qed', 'solubility', 'lipophilicity', 'freesolv', 'beet', 'qm9_gap', 'qm9_dipole',
              'melting_point']
TITLES = {
    'clogp': 'ClogP (AqSolDB)', 'qed': 'QED (ZINC250k)', 'solubility': 'Solubility (AqSolDB)',
    'lipophilicity': 'Lipophilicity', 'freesolv': 'Hydration free energy (FreeSolv)', 'beet': 'Bee toxicity (BEET)',
    'qm9_gap': 'HOMO-LUMO gap (QM9)', 'qm9_dipole': 'Dipole moment (QM9)', 'melting_point': 'Melting point',
}
SIGNIFICANCE = [(0.001, '***'), (0.01, '**'), (0.05, '*')]
# significance marks where Morgan + Tanimoto is significantly better than HDF + Matern
MORGAN_BETTER_COLOR = '#D62728'

# sizes in points at the final print size (the original figure: about 8 pt text, 10 pt panel letters)
FONT_SIZE = 7.5
TITLE_SIZE = 8.0
LETTER_SIZE = 10.0


def setup_style() -> None:
    plt.style.use('default')
    if any('Roboto Condensed' in f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams['font.family'] = 'Roboto Condensed'
    plt.rcParams.update({
        'font.size': FONT_SIZE, 'axes.labelsize': FONT_SIZE, 'xtick.labelsize': FONT_SIZE - 0.5,
        'ytick.labelsize': FONT_SIZE - 0.5, 'legend.fontsize': FONT_SIZE, 'axes.linewidth': 0.6,
        'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
        'svg.fonttype': 'none', 'pdf.fonttype': 42,
    })


def panel_title(ax, letter: str, text: str) -> None:
    """Bold panel letter followed by a left-aligned title, as in the original figure."""
    from matplotlib.transforms import offset_copy
    ax.text(0.0, 1.035, letter, transform=ax.transAxes, fontsize=LETTER_SIZE, fontweight='bold', ha='left',
            va='bottom')
    # the title starts a fixed distance (in points) after the letter, whatever the panel width
    shifted = offset_copy(ax.transAxes, fig=ax.figure, x=LETTER_SIZE * 0.95, y=0, units='points')
    ax.text(0.0, 1.04, text, transform=shifted, fontsize=TITLE_SIZE, ha='left', va='bottom')


def load_campaigns() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(ROOT, 'campaigns.tsv'), sep='\t')
    df['size'] = df['size'].astype(float)
    return df


def load_tests() -> pd.DataFrame:
    tests = pd.read_csv(os.path.join(ROOT, 'analysis', 'tests.tsv'), sep='\t')
    return tests[(tests.family == 'hdf_vs_morgan') & (tests.method_b == 'morgan_tanimoto')]


def available(df: pd.DataFrame, sizes: list) -> list:
    """The sizes of ``sizes`` for which the benchmark has campaigns."""
    present = set(df['size'].dropna().astype(int))
    return [s for s in sizes if s in present]


def regret_curves(objective: str, method: str, size) -> np.ndarray:
    """Normalized regret per BO round of all campaigns of one cell, shape (campaigns, rounds + 1)."""
    folder = os.path.join(ROOT, 'runs', objective, method, 'Dna' if size is None else f'D{int(size)}')
    curves = []
    # finished result files only (c<k>.npz), not the temporary files of an interrupted save
    for name in sorted(n for n in os.listdir(folder) if re.fullmatch(r'c\d+\.npz', n)):
        result = np.load(os.path.join(folder, name))
        meta = json.loads(str(result['meta']))
        best = result['best']
        curves.append((meta['objective_max'] - best) / (meta['objective_max'] - meta['objective_min']))
    return np.array(curves)


def progression_panel(ax, objective: str, size: int, methods: list) -> None:
    for method in ['random'] + methods:
        curves = regret_curves(objective, method, None if method == 'random' else size)
        rounds = np.arange(curves.shape[1])
        mean = curves.mean(axis=0)
        sem = curves.std(axis=0, ddof=1) / np.sqrt(len(curves))
        ax.plot(rounds, mean, color=COLORS[method], lw=1.6, zorder=3, label=LABELS[method])
        ax.fill_between(rounds, mean - sem, mean + sem, color=COLORS[method], alpha=0.15, lw=0, zorder=2)
    ax.set_xlim(0, rounds[-1])
    ax.set_ylim(bottom=0)
    ax.set_xlabel('BO round')
    ax.grid(True, alpha=0.3, linestyle='--', lw=0.5)


def bracket(ax, x0: float, x1: float, y: float, height: float, label: str, color: str = 'black') -> None:
    """Significance bracket between two box positions, opening upwards, with the label below (as before)."""
    ax.plot([x0, x0, x1, x1], [y + height, y, y, y + height], color=color, lw=0.7, clip_on=False)
    ax.text((x0 + x1) / 2, y - 0.4 * height, label, ha='center', va='top', fontsize=FONT_SIZE, color=color,
            clip_on=False)


def box_panel(ax, df: pd.DataFrame, tests: pd.DataFrame, objective: str, methods: list, sizes: list) -> None:
    group_width = 0.8
    box_width = group_width / len(methods)
    offsets = {m: (j - (len(methods) - 1) / 2) * box_width for j, m in enumerate(methods)}
    data = df[df.objective == objective]
    random = data[data.method == 'random']['auc'].to_numpy()
    random_median = float(np.median(random))
    low, high = np.inf, random_median
    for i, size in enumerate(sizes):
        for method in methods:
            values = data[(data.method == method) & (data['size'] == size)]['auc'].to_numpy()
            if len(values) == 0:
                continue  # Morgan-PCA cannot have more components than the library has molecules
            low, high = min(low, values.min()), max(high, values.max())
            ax.boxplot(
                [values], positions=[i + offsets[method]], widths=box_width * 0.85, patch_artist=True,
                showfliers=True,
                boxprops=dict(facecolor=COLORS[method], alpha=0.7, linewidth=0.6),
                medianprops=dict(color='black', linewidth=1.2),
                whiskerprops=dict(color=COLORS[method], linewidth=0.8),
                capprops=dict(color=COLORS[method], linewidth=0.8),
                flierprops=dict(marker='o', markerfacecolor=COLORS[method], markeredgecolor='none', markersize=2.5,
                                alpha=0.5),
            )
    ax.axhline(random_median, color=COLORS['random'], ls='--', lw=0.9, zorder=1)
    # the axis starts at zero unless all values lie far above it; then it starts just below the data, so that
    # the boxes fill the panel. Below the data, a strip holds the significance brackets.
    base = 0.0 if low < 0.15 * high else low - 0.06 * (high - low)
    top = high + 0.05 * (high - base)
    span = top - base
    for i, size in enumerate(sizes):
        row = tests[(tests.objective == objective) & (tests['size'] == size)]
        if row.empty or pd.isna(row.p_holm.iloc[0]) or 'morgan_tanimoto' not in offsets:
            continue
        p = float(row.p_holm.iloc[0])
        label = next((mark for threshold, mark in SIGNIFICANCE if p < threshold), None)
        if label:
            # black where HDF is better, red where Morgan + Tanimoto is better (user, 2026-10-14)
            morgan_better = float(row.auc_median_b.iloc[0]) < float(row.auc_median_a.iloc[0])
            bracket(ax, i + offsets['morgan_tanimoto'], i + offsets['hdf_matern'], base - 0.05 * span,
                    0.025 * span, label, MORGAN_BETTER_COLOR if morgan_better else 'black')
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([str(s) for s in sizes])
    ax.set_xlim(-0.55, len(sizes) - 0.45)
    # ticks only inside the data range (the strip below holds the brackets)
    ticks = MaxNLocator(nbins=5).tick_values(base, top)
    ax.set_yticks([t for t in ticks if base - 1e-12 <= t <= top])
    ax.set_ylim(base - 0.16 * span, top)
    ax.set_ylabel('Regret AUC $\\downarrow$')
    ax.grid(True, alpha=0.3, axis='y', lw=0.5)


def main_figure(df: pd.DataFrame, tests: pd.DataFrame, path: str) -> None:
    methods = MAIN_METHODS
    sizes = available(df, SIZES)
    fig = plt.figure(figsize=(7.8, 4.6))
    outer = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15], hspace=0.4, wspace=0.17,
                             left=0.065, right=0.985, top=0.875, bottom=0.085)
    letters = iter('abcdef')
    for column, objective in enumerate(('clogp', 'qed')):
        # the two progression panels of one objective share their y-axis and sit close together
        inner = outer[0, column].subgridspec(1, 2, wspace=0.13)
        left = None
        for k, size in enumerate(PROGRESSION_SIZES):
            ax = fig.add_subplot(inner[0, k], sharey=left)
            progression_panel(ax, objective, size, methods)
            panel_title(ax, next(letters), f'{TITLES[objective].split(" (")[0]}, {size} dimensions')
            if k == 0:
                left = ax
                if column == 0:
                    ax.set_ylabel('Normalized regret $\\downarrow$')
            else:
                ax.tick_params(labelleft=False)
    for column, objective in enumerate(('clogp', 'qed')):
        ax = fig.add_subplot(outer[1, column])
        box_panel(ax, df, tests, objective, methods, sizes)
        ax.set_xlabel('Embedding size')
        if column == 1:
            ax.set_ylabel('')
        panel_title(ax, next(letters), f'{TITLES[objective]}: regret AUC across embedding sizes')
    handles = [plt.Line2D([], [], color=COLORS[m], lw=1.6, label=LABELS[m]) for m in methods]
    handles.append(plt.Line2D([], [], color=COLORS['random'], lw=1.6, label=LABELS['random']))
    fig.legend(handles=handles, loc='upper center', ncol=3, frameon=True, framealpha=0.9,
               bbox_to_anchor=(0.5, 0.998), handlelength=1.8, columnspacing=1.6)
    fig.savefig(path + '.pdf')
    fig.savefig(path + '.png', dpi=300)
    plt.close(fig)


def si_figure(df: pd.DataFrame, tests: pd.DataFrame, path: str) -> None:
    methods = SI_METHODS
    sizes = available(df, SI_SIZES)
    fig, axes = plt.subplots(3, 3, figsize=(7.8, 7.0))
    fig.subplots_adjust(left=0.07, right=0.995, top=0.925, bottom=0.065, hspace=0.36, wspace=0.2)
    for ax, letter, objective in zip(axes.flat, 'abcdefghi', OBJECTIVES):
        box_panel(ax, df, tests, objective, methods, sizes)
        panel_title(ax, letter, TITLES[objective])
        if objective not in ('clogp', 'lipophilicity', 'qm9_gap'):
            ax.set_ylabel('')
        if objective in ('qm9_gap', 'qm9_dipole', 'melting_point'):
            ax.set_xlabel('Embedding size')
    handles = [Patch(facecolor=COLORS[m], alpha=0.7, label=LABELS[m]) for m in methods]
    handles.append(plt.Line2D([], [], color=COLORS['random'], ls='--', lw=0.9, label=LABELS['random']))
    fig.legend(handles=handles, loc='upper center', ncol=5, frameon=True, framealpha=0.9,
               bbox_to_anchor=(0.5, 0.998), handlelength=1.8, columnspacing=1.4)
    fig.savefig(path + '.pdf')
    fig.savefig(path + '.png', dpi=300)
    plt.close(fig)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'analysis')
    os.makedirs(out, exist_ok=True)
    setup_style()
    df = load_campaigns()
    tests = load_tests()
    main_figure(df, tests, os.path.join(out, 'figure_bo'))
    si_figure(df, tests, os.path.join(out, 'figure_bo_si'))
    print(f'wrote {out}/figure_bo.(pdf|png) and figure_bo_si.(pdf|png)')


if __name__ == '__main__':
    main()
