"""
Figures of the Bayesian optimization benchmark ex_24 (``bo_benchmark.py``) in the style of the original BO figure of
the paper (Fig. 4, made with analyze_ex_07.ipynb): Roboto Condensed, gray / light blue / light green, transparent
box plots with black medians, significance brackets, shaded progression bands, bold panel letters with
left-aligned titles.

- ``figure_bo``: main-text figure. Top row: normalized regret per BO round at 16 and 1024 dimensions for ClogP
  (AqSolDB) and QED (ZINC250k), mean over the 25 campaigns with the standard error as band. Bottom row: regret AUC
  against the embedding size for both objectives, box plots over the 25 campaigns, random search as a dashed line,
  brackets for HDF against Morgan + Tanimoto (paired two-sided Wilcoxon signed-rank test, Holm-corrected within
  each objective as in analyze_ex_24.py).
- ``figure_bo_si``: the box-plot panels of all nine objectives in a 3 x 3 grid (SI).

HDF appears with its standard kernel (Matern-5/2) only. Reads ``_ex24/campaigns.tsv``, ``_ex24/runs/`` and
``_ex24/analysis/tests.tsv`` (``bo_benchmark.py collect`` and ``analyze_ex_24.py``; EX24_ROOT points elsewhere).

    python make_figure_bo.py [output folder]
"""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

PATH = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get('EX24_ROOT', os.path.join(PATH, '_ex24'))

SIZES = [8, 16, 32, 64, 128, 512, 1024]
PROGRESSION_SIZES = [16, 1024]
# box order within a size: Morgan variants left, HDF right (as in the original figure)
METHODS = ['morgan_tanimoto', 'morgan_pca', 'hdf_matern']
COLORS = {
    'random': '#808080',
    'morgan_tanimoto': '#7C92FF',
    'morgan_pca': '#4C64EB',
    'hdf_matern': '#69F7AE',
}
LABELS = {
    'random': 'Random search',
    'morgan_tanimoto': 'Morgan FP (Tanimoto)',
    'morgan_pca': 'Morgan FP, PCA (Matérn)',
    'hdf_matern': 'HDF (Matérn)',
}
OBJECTIVES = ['clogp', 'qed', 'solubility', 'lipophilicity', 'freesolv', 'beet', 'qm9_gap', 'qm9_dipole',
              'melting_point']
TITLES = {
    'clogp': 'ClogP (AqSolDB)', 'qed': 'QED (ZINC250k)', 'solubility': 'Solubility (AqSolDB)',
    'lipophilicity': 'Lipophilicity', 'freesolv': 'Hydration free energy (FreeSolv)', 'beet': 'Bee toxicity (BEET)',
    'qm9_gap': 'HOMO-LUMO gap (QM9)', 'qm9_dipole': 'Dipole moment (QM9)', 'melting_point': 'Melting point',
}
SIGNIFICANCE = [(0.001, '***'), (0.01, '**'), (0.05, '*')]

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


def regret_curves(objective: str, method: str, size) -> np.ndarray:
    """Normalized regret per BO round of all campaigns of one cell, shape (campaigns, rounds + 1)."""
    folder = os.path.join(ROOT, 'runs', objective, method, 'Dna' if size is None else f'D{int(size)}')
    curves = []
    for name in sorted(os.listdir(folder)):
        result = np.load(os.path.join(folder, name))
        meta = json.loads(str(result['meta']))
        best = result['best']
        curves.append((meta['objective_max'] - best) / (meta['objective_max'] - meta['objective_min']))
    return np.array(curves)


def progression_panel(ax, objective: str, size: int) -> None:
    for method in ['random'] + METHODS:
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


def bracket(ax, x0: float, x1: float, y: float, height: float, label: str) -> None:
    """Significance bracket between two box positions, opening upwards, with the label below (as before)."""
    ax.plot([x0, x0, x1, x1], [y + height, y, y, y + height], color='black', lw=0.7, clip_on=False)
    ax.text((x0 + x1) / 2, y - 0.4 * height, label, ha='center', va='top', fontsize=FONT_SIZE, clip_on=False)


def box_panel(ax, df: pd.DataFrame, tests: pd.DataFrame, objective: str, legend: bool = False) -> None:
    box_width = 0.26
    offsets = {'morgan_tanimoto': -box_width, 'morgan_pca': 0.0, 'hdf_matern': box_width}
    data = df[df.objective == objective]
    top = 0.0
    for i, size in enumerate(SIZES):
        for method in METHODS:
            values = data[(data.method == method) & (data['size'] == size)]['auc'].to_numpy()
            if len(values) == 0:
                continue  # Morgan-PCA cannot have more components than the library has molecules
            top = max(top, values.max())
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
    random = data[data.method == 'random']['auc'].to_numpy()
    random_median = float(np.median(random))
    top = max(top, random_median)
    ax.axhline(random_median, color=COLORS['random'], ls='--', lw=0.9, zorder=1)
    ax.text(len(SIZES) - 0.5, random_median, 'random search', color=COLORS['random'], style='italic',
            fontsize=FONT_SIZE - 0.5, ha='right', va='bottom', zorder=5,
            bbox=dict(facecolor='white', alpha=0.75, edgecolor='none', pad=0.6))
    # room for the brackets below the boxes
    span = top * 1.08
    ax.set_ylim(-0.17 * span, span)
    for i, size in enumerate(SIZES):
        row = tests[(tests.objective == objective) & (tests['size'] == size)]
        if row.empty or pd.isna(row.p_holm.iloc[0]):
            continue
        p = float(row.p_holm.iloc[0])
        label = next((mark for threshold, mark in SIGNIFICANCE if p < threshold), None)
        if label:
            bracket(ax, i + offsets['morgan_tanimoto'], i + offsets['hdf_matern'], -0.05 * span, 0.025 * span, label)
    ax.set_xticks(range(len(SIZES)))
    ax.set_xticklabels([str(s) for s in SIZES])
    ax.set_xlim(-0.6, len(SIZES) - 0.4)
    # only non-negative ticks inside the data range (set_yticks would otherwise widen the axis)
    ax.set_yticks([t for t in ax.get_yticks() if 0 <= t <= span])
    ax.set_ylim(-0.17 * span, span)
    ax.set_xlabel('Embedding size')
    ax.set_ylabel('Regret AUC $\\downarrow$')
    ax.grid(True, alpha=0.3, axis='y', lw=0.5)
    if legend:
        handles = [Patch(facecolor=COLORS[m], alpha=0.7, label=LABELS[m]) for m in METHODS]
        ax.legend(handles=handles, loc='upper right', frameon=True, framealpha=0.9)


def main_figure(df: pd.DataFrame, tests: pd.DataFrame, path: str) -> None:
    fig = plt.figure(figsize=(7.8, 5.0))
    grid = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.15], hspace=0.62, wspace=0.42,
                            left=0.065, right=0.99, top=0.86, bottom=0.08)
    letters = iter('abcdef')
    first = None
    for column, (objective, size) in enumerate([(o, s) for o in ('clogp', 'qed') for s in PROGRESSION_SIZES]):
        ax = fig.add_subplot(grid[0, column])
        progression_panel(ax, objective, size)
        panel_title(ax, next(letters), f'{TITLES[objective].split(" (")[0]}, {size} dimensions')
        if column == 0:
            ax.set_ylabel('Normalized regret $\\downarrow$')
            first = ax
    for column, objective in enumerate(('clogp', 'qed')):
        ax = fig.add_subplot(grid[1, 2 * column:2 * column + 2])
        box_panel(ax, df, tests, objective)
        panel_title(ax, next(letters), f'{TITLES[objective]}: regret AUC across embedding sizes')
    handles = [plt.Line2D([], [], color=COLORS[m], lw=1.6, label=LABELS[m]) for m in METHODS + ['random']]
    fig.legend(handles=handles, loc='upper center', ncol=4, frameon=True, framealpha=0.9,
               bbox_to_anchor=(0.5, 0.995), handlelength=1.8, columnspacing=1.6)
    fig.savefig(path + '.pdf')
    fig.savefig(path + '.png', dpi=300)
    plt.close(fig)


def si_figure(df: pd.DataFrame, tests: pd.DataFrame, path: str) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(7.8, 7.4))
    fig.subplots_adjust(left=0.07, right=0.99, top=0.91, bottom=0.06, hspace=0.62, wspace=0.28)
    for ax, letter, objective in zip(axes.flat, 'abcdefghi', OBJECTIVES):
        box_panel(ax, df, tests, objective)
        panel_title(ax, letter, TITLES[objective])
        if objective not in ('clogp', 'lipophilicity', 'qm9_gap'):
            ax.set_ylabel('')
    handles = [Patch(facecolor=COLORS[m], alpha=0.7, label=LABELS[m]) for m in METHODS]
    handles.append(plt.Line2D([], [], color=COLORS['random'], ls='--', lw=0.9, label=LABELS['random']))
    fig.legend(handles=handles, loc='upper center', ncol=4, frameon=True, framealpha=0.9,
               bbox_to_anchor=(0.5, 0.995), handlelength=1.8, columnspacing=1.6)
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
