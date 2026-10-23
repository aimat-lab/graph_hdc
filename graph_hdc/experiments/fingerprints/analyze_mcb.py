"""
Multi-codebook HDF (hdf_mcb_rbf, branch revision/ex24-mcb) against the single-codebook arms of ex_24 on the solubility
objective (user, 2026-10-23: first test on one representative property).

Pairs the MCB campaigns with the ex_24 campaigns of the same index (shared initial designs), summarizes the regret AUC
and the hit rate per size, tests MCB against HDF + Matern (primary) and against Morgan + Tanimoto with paired two-sided
Wilcoxon signed-rank tests (Holm over the sizes within each comparison, as analyze_ex_24.py), and draws one box-plot
panel in the style of the SI figure: Morgan + Tanimoto, HDF + Matern, MCB, random search as a dashed line, brackets for
MCB against HDF + Matern (black where MCB is better, red where the single codebook is better).

    python analyze_mcb.py <ex_24 campaigns.tsv> <MCB campaigns.tsv> <output folder>
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import to_rgba  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

from analyze_ex_24 import holm, paired, wilcoxon_p  # noqa: E402
from make_figure_bo import (COLORS, LABELS, MORGAN_BETTER_COLOR, SIGNIFICANCE, SI_STYLE, bracket,  # noqa: E402
                            panel_title, setup_style, style_rc)

OBJECTIVE = 'solubility'
SIZES = [8, 16, 32, 64, 128, 256, 512, 1024]
MCB = 'hdf_mcb_rbf'
PANEL_METHODS = ['morgan_tanimoto', 'hdf_matern', MCB]
TABLE_METHODS = ['morgan_tanimoto', 'morgan_pca', 'hdf_matern', 'hdf_linear', MCB]
STYLE = {**SI_STYLE, 'colors': {**COLORS, MCB: '#3FC1C9'}}
NAMES = {**LABELS, MCB: 'HDF, 4 codebooks (block RBF)'}


def load(ex24_path: str, mcb_path: str) -> pd.DataFrame:
    old = pd.read_csv(ex24_path, sep='\t')
    new = pd.read_csv(mcb_path, sep='\t')
    df = pd.concat([old[old.objective == OBJECTIVE], new[new.objective == OBJECTIVE]], ignore_index=True)
    df['size'] = df['size'].astype(float)
    counts = df[df.method == MCB].groupby('size').size()
    if len(counts) != len(SIZES) or (counts != 100).any():
        print(f'!!! MCB campaigns per size: {counts.to_dict()}')
    return df


def summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in TABLE_METHODS:
        for size in SIZES:
            cell = df[(df.method == method) & (df['size'] == size)]
            if cell.empty:
                continue
            rows.append({'method': method, 'size': size, 'campaigns': len(cell), 'auc_median': cell.auc.median(),
                         'auc_q1': cell.auc.quantile(0.25), 'auc_q3': cell.auc.quantile(0.75),
                         'hit_rate': 100 * cell.hit.mean(), 'fit_failures': int(cell.fit_failures.sum()),
                         'seconds_median': cell.seconds_total.median()})
    random = df[df.method == 'random']
    rows.append({'method': 'random', 'size': np.nan, 'campaigns': len(random), 'auc_median': random.auc.median(),
                 'auc_q1': random.auc.quantile(0.25), 'auc_q3': random.auc.quantile(0.75),
                 'hit_rate': 100 * random.hit.mean(), 'fit_failures': 0, 'seconds_median': np.nan})
    return pd.DataFrame(rows)


def tests(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for other in ('hdf_matern', 'morgan_tanimoto', 'hdf_linear'):
        family = []
        for size in SIZES:
            pair = paired(df, OBJECTIVE, MCB, size, other, size)
            if pair is None:
                continue
            a, b = pair
            family.append({'method_a': MCB, 'method_b': other, 'size': size, 'campaigns': len(a),
                           'auc_median_a': np.median(a), 'auc_median_b': np.median(b),
                           'a_better': int((a < b).sum()), 'p': wilcoxon_p(a, b)})
        for row, p_holm in zip(family, holm([r['p'] for r in family])):
            rows.append({**row, 'p_holm': p_holm})
    return pd.DataFrame(rows)


@plt.rc_context(style_rc(SI_STYLE))
def figure(df: pd.DataFrame, test_rows: pd.DataFrame, path: str) -> None:
    style = STYLE
    fig, ax = plt.subplots(figsize=(7.8, 2.9))
    fig.subplots_adjust(left=0.075, right=0.995, top=0.8, bottom=0.17)
    group_width = 0.8
    box_width = group_width / len(PANEL_METHODS)
    offsets = {m: (j - (len(PANEL_METHODS) - 1) / 2) * box_width for j, m in enumerate(PANEL_METHODS)}
    random_median = float(df[df.method == 'random'].auc.median())
    low, high = np.inf, random_median
    for i, size in enumerate(SIZES):
        for method in PANEL_METHODS:
            values = df[(df.method == method) & (df['size'] == size)].auc.to_numpy()
            if len(values) == 0:
                continue
            low, high = min(low, values.min()), max(high, values.max())
            ax.boxplot([values], positions=[i + offsets[method]], widths=box_width * 0.85, patch_artist=True,
                       boxprops=dict(facecolor=to_rgba(style['colors'][method], 0.7), edgecolor='black',
                                     linewidth=style['box']),
                       medianprops=dict(color='black', linewidth=style['median']),
                       whiskerprops=dict(color='black', linewidth=style['box']),
                       capprops=dict(color='black', linewidth=style['box']),
                       flierprops=dict(marker='o', markerfacecolor=style['colors'][method], markeredgecolor='black',
                                       markeredgewidth=style['flier_edge'], markersize=style['flier']))
    ax.axhline(random_median, color=style['colors']['random'], ls='--', lw=style['random'], zorder=1)
    base = 0.0 if low < 0.15 * high else low - 0.06 * (high - low)
    top = high + 0.05 * (high - base)
    span = top - base
    for i, size in enumerate(SIZES):
        row = test_rows[(test_rows.method_b == 'hdf_matern') & (test_rows['size'] == size)]
        if row.empty or pd.isna(row.p_holm.iloc[0]):
            continue
        label = next((mark for threshold, mark in SIGNIFICANCE if float(row.p_holm.iloc[0]) < threshold), None)
        if label:
            single_better = float(row.auc_median_b.iloc[0]) < float(row.auc_median_a.iloc[0])
            bracket(ax, i + offsets['hdf_matern'], i + offsets[MCB], base - 0.05 * span, 0.025 * span, label,
                    style, MORGAN_BETTER_COLOR if single_better else 'black')
    ax.set_xticks(range(len(SIZES)))
    ax.set_xticklabels([str(s) for s in SIZES])
    ax.set_xlim(-0.55, len(SIZES) - 0.45)
    ticks = MaxNLocator(nbins=5).tick_values(base, top)
    ax.set_yticks([t for t in ticks if base - 1e-12 <= t <= top])
    ax.set_ylim(base - 0.16 * span, top)
    ax.set_ylabel('Regret AUC $\\downarrow$')
    ax.set_xlabel('Embedding size (MCB: 4 codebooks of size/4)')
    ax.grid(True, alpha=0.3, axis='y', lw=style['grid'])
    panel_title(ax, '', 'Solubility (AqSolDB)', style)
    handles = [Patch(facecolor=to_rgba(style['colors'][m], 0.7), edgecolor='black', linewidth=style['box'],
                     label=NAMES[m]) for m in PANEL_METHODS]
    handles.append(plt.Line2D([], [], color=style['colors']['random'], ls='--', lw=style['random'],
                              label=LABELS['random']))
    fig.legend(handles=handles, loc='upper center', ncol=4, frameon=True, framealpha=0.9, bbox_to_anchor=(0.5, 0.995),
               handlelength=1.8, columnspacing=1.4)
    fig.savefig(path + '.pdf')
    fig.savefig(path + '.png', dpi=300)
    plt.close(fig)


def main():
    ex24_path, mcb_path, out = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(out, exist_ok=True)
    setup_style()
    df = load(ex24_path, mcb_path)
    table = summary(df)
    test_rows = tests(df)
    table.to_csv(os.path.join(out, 'mcb_summary.tsv'), sep='\t', index=False)
    test_rows.to_csv(os.path.join(out, 'mcb_tests.tsv'), sep='\t', index=False)
    figure(df, test_rows, os.path.join(out, 'figure_bo_mcb_solubility'))
    pd.set_option('display.width', 200)
    medians = table.pivot(index='method', columns='size', values='auc_median').reindex(TABLE_METHODS)
    hits = table.pivot(index='method', columns='size', values='hit_rate').reindex(TABLE_METHODS)
    print('median regret AUC\n', medians.round(4).to_string())
    print('\nhit rate (%)\n', hits.round(0).to_string())
    print('\nrandom search: median AUC', round(float(table[table.method == 'random'].auc_median.iloc[0]), 4))
    print('\ntests (MCB = a)\n', test_rows[['method_b', 'size', 'auc_median_a', 'auc_median_b', 'a_better',
                                            'p_holm']].round(4).to_string(index=False))


if __name__ == '__main__':
    main()
