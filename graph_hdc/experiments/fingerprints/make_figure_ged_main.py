"""
Figure 3 of the main text, drawn as one figure in the style and at the print size of Figure 4 (7.4 in wide, 11 pt
text, printed at about 8 pt at the full text width), replacing the composite of make_figures_ged_edit.py and
make_figure_knn_ratio.py, whose text printed at about 5 pt (user, 2026-10-26).

a  |Pearson correlation| between the fingerprint distances and the number of edit steps (edit-based GED proxy, ex_08_c),
   one value per seed molecule, box plots per embedding size for Morgan (radius 2) and HDF (depth 2)
   (per_seed.csv of analyze_ex_08c.py).
b  KNN error ratio MAE_Morgan / MAE_HDF (k = 5, depth / radius 2, native distances) over the embedding size, the mean
   of the per-seed ratios over the 10 repetitions (ex_29, fig3b_runs.csv of analyze_ex_29.py); green where HDF is
   more accurate and blue where Morgan is, a dashed parity line.

    python make_figure_ged_main.py <per_seed.csv> <fig3b_runs.csv> <output .pdf> [--png]
"""
import sys
import csv
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import pandas as pd  # noqa: E402

FONT_SIZE = 11
COLOR_MORGAN, COLOR_HDF = '#7C92FF', '#69F7AE'
SIZES_A = [32, 128, 512, 2048]
DIMENSIONS_B = [32, 64, 128, 256, 512, 1024, 2048]
# (target, label, marker, line style, brightness) as in the original panel b
DATASETS = [('freesolv_hfe', 'FreeSolv', 'o', '-', -0.15),
            ('lipophilicity_logD', 'Lipophilicity', 's', '--', 0.05),
            ('aqsoldb_logs', 'AqSolDB', 'D', '-.', 0.25),
            ('qm9_gap', 'QM9 gap', 'p', (0, (5, 2)), 0.45)]
LINE = 1.6


def brightness(hex_color: str, amount: float) -> tuple:
    """amount > 0 blends toward white, < 0 toward black."""
    rgb = np.array(matplotlib.colors.to_rgb(hex_color))
    rgb = rgb + (1 - rgb) * amount if amount >= 0 else rgb * (1 + amount)
    return tuple(np.clip(rgb, 0, 1))


def ratios(path: str, variant: str = 'native') -> dict:
    """{target: {dimension: [MAE_Morgan / MAE_HDF of every seed]}}"""
    mae = {}
    for row in csv.DictReader(open(path)):
        if row['mae'] in ('', 'None') or row['distance'] != variant:
            continue
        mae[(row['target'], row['method'], int(row['dimension']), int(row['seed']))] = float(row['mae'])
    out = defaultdict(lambda: defaultdict(list))
    for (target, method, dim, seed), value in mae.items():
        if method == 'morgan' and (target, 'hdf', dim, seed) in mae:
            out[target][dim].append(value / mae[(target, 'hdf', dim, seed)])
    return out


def colored_line(ax, x, y, style, marker, color_above, color_below):
    """A line whose colour switches at the parity crossings (interpolated in log2 x), markers on top."""
    for i in range(len(x) - 1):
        (x0, x1), (y0, y1) = x[i:i + 2], y[i:i + 2]
        if (y0 - 1) * (y1 - 1) < 0:
            xc = 2 ** (np.log2(x0) + (1 - y0) / (y1 - y0) * (np.log2(x1) - np.log2(x0)))
            segments = [((x0, xc), (y0, 1.0)), ((xc, x1), (1.0, y1))]
        else:
            segments = [((x0, x1), (y0, y1))]
        for xs, ys in segments:
            above = (ys[0] + ys[1]) / 2 >= 1
            ax.plot(xs, ys, color=color_above if above else color_below, linestyle=style, linewidth=LINE, zorder=2)
    for xi, yi in zip(x, y):
        ax.plot(xi, yi, marker=marker, color=color_above if yi >= 1 else color_below, markersize=6,
                markeredgecolor='black', markeredgewidth=0.6, linestyle='none', zorder=3)


def panel_a(ax, per_seed: pd.DataFrame):
    for i, size in enumerate(SIZES_A):
        for representation, offset, color in (('morgan', -0.18, COLOR_MORGAN), ('hdf', 0.18, COLOR_HDF)):
            values = per_seed[(per_seed['representation'] == representation) & (per_seed['size'] == size)]
            ax.boxplot([values['abs_correlation'].dropna().to_numpy()], positions=[i + offset], widths=0.3,
                       patch_artist=True, showfliers=True, manage_ticks=False,
                       boxprops={'facecolor': color, 'edgecolor': 'black', 'linewidth': 1.2},
                       medianprops={'color': 'black', 'linewidth': 1.6},
                       whiskerprops={'color': 'black', 'linewidth': 1.2},
                       capprops={'color': 'black', 'linewidth': 1.2},
                       flierprops={'marker': 'o', 'markerfacecolor': color, 'markeredgecolor': 'black',
                                   'markeredgewidth': 0.5, 'markersize': 3.5})
    ax.set_xticks(range(len(SIZES_A)), [str(s) for s in SIZES_A])
    ax.set_xlim(-0.6, len(SIZES_A) - 0.4)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel('Embedding size', fontsize=FONT_SIZE + 1)
    ax.set_ylabel('|Pearson correlation| $\\uparrow$', fontsize=FONT_SIZE + 1)
    ax.set_title('Correlation with GED (edit steps)', fontsize=FONT_SIZE + 1)
    ax.legend(handles=[Patch(facecolor=COLOR_MORGAN, edgecolor='black', label='Morgan (radius 2)'),
                       Patch(facecolor=COLOR_HDF, edgecolor='black', label='HDF (depth 2)')],
              loc='lower right', fontsize=FONT_SIZE - 1, framealpha=0.9)


def panel_b(ax, data: dict):
    ax.axhline(1.0, color='gray', linestyle='--', linewidth=1.0, zorder=1)
    for target, _label, marker, style, amount in DATASETS:
        dims = [d for d in DIMENSIONS_B if data.get(target, {}).get(d)]
        means = [float(np.mean(data[target][d])) for d in dims]
        colored_line(ax, dims, means, style, marker, brightness(COLOR_HDF, amount), brightness(COLOR_MORGAN, amount))
    ax.set_xscale('log', base=2)
    ax.set_xticks(DIMENSIONS_B, [str(d) for d in DIMENSIONS_B])
    ax.minorticks_off()
    ax.set_xlim(DIMENSIONS_B[0] / 1.25, DIMENSIONS_B[-1] * 1.25)
    ax.set_xlabel('Embedding size', fontsize=FONT_SIZE + 1)
    ax.set_ylabel('Error$_{\\mathrm{FP}}$ / Error$_{\\mathrm{HDF}}$ $\\uparrow$', fontsize=FONT_SIZE + 1)  # notation of the caption
    ax.set_title('KNN error ratio', fontsize=FONT_SIZE + 1)
    handles = [Line2D([], [], color='black', linestyle=style, linewidth=1.0, marker=marker, markerfacecolor='white',
                      markeredgecolor='black', markersize=5.5, label=label) for _t, label, marker, style, _a in DATASETS]
    handles.append(Line2D([], [], color='gray', linestyle='--', linewidth=1.0, label='Parity'))
    ax.legend(handles=handles, loc='upper right', ncol=2, fontsize=FONT_SIZE - 1, framealpha=0.9, handlelength=2.2,
              columnspacing=1.0)


def main(per_seed_path: str, ratio_path: str, out: str, png: bool):
    plt.style.use('default')
    plt.rcParams.update({'font.family': 'Roboto Condensed', 'font.size': FONT_SIZE})
    per_seed = pd.read_csv(per_seed_path)
    data = ratios(ratio_path)
    # 7.4 in wide like Figure 4: at the full text width the fonts print at about 8 pt
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))
    panel_a(axes[0], per_seed)
    panel_b(axes[1], data)
    for k, ax in enumerate(axes):
        ax.grid(True, which='major', linestyle='--', alpha=0.3)
        ax.set_axisbelow(True)
        ax.text(-0.14, 1.06, 'ab'[k], transform=ax.transAxes, fontsize=FONT_SIZE + 4, fontweight='bold',
                va='bottom', ha='left')
    fig.tight_layout(w_pad=1.5)
    fig.savefig(out, bbox_inches='tight', pad_inches=0.02)
    if png:
        fig.savefig(out.rsplit('.', 1)[0] + '.png', dpi=200, bbox_inches='tight', pad_inches=0.02)
    counts = sorted({len(v) for per in data.values() for v in per.values()})
    print(f'wrote {out}; KNN seeds per point: {counts}; GED seeds per box: '
          f'{sorted(per_seed.groupby(["representation", "size"]).size().unique().tolist())}')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], '--png' in sys.argv)
