"""
Figure 3 of the main text with a new panel b (ex_29): the KNN error ratio MAE_Morgan / MAE_HDF over the embedding
dimension at depth / radius 2 (k = 5) on FreeSolv, Lipophilicity, AqSolDB and the QM9 HOMO-LUMO gap, the mean over the
10 repetitions of the per-seed ratios (HDF and Morgan share the split of a seed). Panel a (the GED correlation) is
clipped unchanged from the current composite.

The panel is drawn directly in the coordinates of the composite (546 x 169 pt; panel b from x = 280 pt) with the style
of the original panel b (analyze_ex_10.ipynb: line style and marker per dataset, green where HDF is more accurate and
blue where Morgan is, a dashed parity line) at the scale of the composite (tick labels 8.2 pt), and its frame and title
line up with panel a (see figure_main_composite in make_figures_ged_edit.py).

The distance variant selects the runs of both representations: 'native' (Jaccard for Morgan, cosine for HDF) or
'euclidean' (Euclidean for both).

    python make_figure_knn_ratio.py <fig3b_runs.csv> <native|euclidean> <current composite .pdf> <output .pdf> [--png]
"""
import os
import csv
import sys
import shutil
import tempfile
import subprocess
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

FONT_SIZE = 11
SCALE = 0.745                      # the composite's scale of the matplotlib sizes (tick labels 8.2 pt)
WIDTH, HEIGHT, SPLIT = 546.0, 169.0, 280.0   # pt
COLOR_HDF, COLOR_MORGAN = '#69F7AE', '#7C92FF'
# (target, label, marker, line style, brightness) as in the original panel b
DATASETS = [('freesolv_hfe', 'FreeSolv', 'o', '-', -0.15),
            ('lipophilicity_logD', 'Lipophilicity', 's', '--', 0.05),
            ('aqsoldb_logs', 'AqSolDB', 'D', '-.', 0.25),
            ('qm9_gap', 'QM9-Gap', 'p', (0, (5, 2)), 0.45)]
DIMENSIONS = [32, 64, 128, 256, 512, 1024, 2048]


def brightness(hex_color: str, amount: float) -> tuple:
    """amount > 0 blends toward white, < 0 toward black (as in the original panel)."""
    rgb = np.array(matplotlib.colors.to_rgb(hex_color))
    rgb = rgb + (1 - rgb) * amount if amount >= 0 else rgb * (1 + amount)
    return tuple(np.clip(rgb, 0, 1))


def ratios(path: str, variant: str) -> dict:
    """{target: {dimension: [MAE_Morgan / MAE_HDF of every seed]}}"""
    mae = {}
    for row in csv.DictReader(open(path)):
        if row['mae'] in ('', 'None'):
            continue
        if row['distance'] == variant:
            mae[(row['target'], row['method'], int(row['dimension']), int(row['seed']))] = float(row['mae'])
    out = defaultdict(lambda: defaultdict(list))
    for (target, method, dim, seed), value in mae.items():
        if method == 'morgan' and (target, 'hdf', dim, seed) in mae:
            out[target][dim].append(value / mae[(target, 'hdf', dim, seed)])
    return out


def colored_line(ax, x, y, style, marker, color_above, color_below, linewidth):
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
            ax.plot(xs, ys, color=color_above if above else color_below, linestyle=style, linewidth=linewidth,
                    zorder=2)
    for xi, yi in zip(x, y):
        ax.plot(xi, yi, marker=marker, color=color_above if yi >= 1 else color_below, markersize=9 * SCALE,
                markeredgecolor='black', markeredgewidth=1.0 * SCALE, linestyle='none', zorder=3)


def panel_b(data: dict, out: str):
    rc = {'font.family': 'Roboto Condensed', 'font.size': FONT_SIZE * SCALE, 'axes.linewidth': 0.8 * SCALE,
          'grid.linewidth': 0.8 * SCALE, 'xtick.major.size': 3.5 * SCALE, 'ytick.major.size': 3.5 * SCALE,
          'xtick.major.width': 0.8 * SCALE, 'ytick.major.width': 0.8 * SCALE, 'xtick.major.pad': 3.5 * SCALE,
          'ytick.major.pad': 3.5 * SCALE, 'axes.labelpad': 4.0 * SCALE, 'legend.fontsize': FONT_SIZE * SCALE,
          'pdf.fonttype': 42}
    plt.style.use('default')
    with plt.rc_context(rc):
        width = WIDTH - SPLIT
        fig = plt.figure(figsize=(width / 72, HEIGHT / 72))
        # frame in pt from the top left of the panel: top and bottom as panel a, left spine 28 pt right of the letter
        left, right, top, bottom = 31.4, 261.0, 25.0, 142.3
        ax = fig.add_axes((left / width, 1 - bottom / HEIGHT, (right - left) / width, (bottom - top) / HEIGHT))
        ax.axhline(1.0, color='gray', linestyle='--', linewidth=1.0 * SCALE, zorder=1)
        for target, _label, marker, style, amount in DATASETS:
            dims = [d for d in DIMENSIONS if data.get(target, {}).get(d)]
            if not dims:
                continue
            means = [float(np.mean(data[target][d])) for d in dims]
            colored_line(ax, dims, means, style, marker, brightness(COLOR_HDF, amount),
                         brightness(COLOR_MORGAN, amount), 2.0 * SCALE)
        ax.set_xscale('log', base=2)
        ax.set_xticks(DIMENSIONS, [str(d) for d in DIMENSIONS])
        ax.minorticks_off()
        ax.set_xlim(DIMENSIONS[0] / 1.25, DIMENSIONS[-1] * 1.25)
        ax.set_xlabel('Embedding Dimensionality', fontsize=(FONT_SIZE + 1.5) * SCALE)
        ax.set_ylabel('Error$_{\\mathrm{FP}}$ / Error$_{\\mathrm{HDF}}$ $\\uparrow$', fontsize=(FONT_SIZE + 1.5) * SCALE)
        ax.grid(True, alpha=0.3, linestyle='--')
        handles = [Line2D([], [], color='black', linestyle=style, linewidth=1.0 * SCALE, marker=marker,
                          markerfacecolor='white', markeredgecolor='black', markersize=7 * SCALE, label=label)
                   for _t, label, marker, style, _a in DATASETS]
        handles.append(Line2D([], [], color='gray', linestyle='--', linewidth=1.0 * SCALE, label='Parity'))
        ax.legend(handles=handles, loc='upper right', ncol=2, framealpha=0.9, handlelength=2.2, columnspacing=1.0)
        baseline = 1 - 7.54 / HEIGHT
        fig.text(3.4 / width, baseline, 'b', fontsize=10, fontweight='bold', va='baseline', ha='left')
        fig.text(11.85 / width, baseline, 'K-Nearest Neighbor Prediction Error Ratio', fontsize=8, va='baseline',
                 ha='left')
        fig.savefig(out)
        plt.close(fig)


def composite(panel: str, current: str, out: str):
    """Panel a clipped from the current composite (x < SPLIT) and the new panel b at x = SPLIT."""
    with tempfile.TemporaryDirectory() as folder:
        shutil.copy(panel, os.path.join(folder, 'panel_b.pdf'))
        shutil.copy(current, os.path.join(folder, 'current.pdf'))
        with open(os.path.join(folder, 'figure.tex'), 'w') as file:
            file.write('\n'.join([
                r'\documentclass[border=0pt]{standalone}',
                r'\usepackage{graphicx}',
                r'\begin{document}',
                r'\setlength{\unitlength}{1bp}%',
                rf'\begin{{picture}}({WIDTH:g},{HEIGHT:g})',
                rf'\put(0,0){{\includegraphics[trim=0bp 0bp {WIDTH - SPLIT:g}bp 0bp,clip]{{current.pdf}}}}',
                rf'\put({SPLIT:g},0){{\includegraphics{{panel_b.pdf}}}}',
                r'\end{picture}',
                r'\end{document}',
            ]))
        subprocess.run(['pdflatex', '-interaction=nonstopmode', '-halt-on-error', 'figure.tex'], cwd=folder,
                       check=True, stdout=subprocess.DEVNULL)
        shutil.copy(os.path.join(folder, 'figure.pdf'), out)


def main(path: str, variant: str, current: str, out: str, png: bool):
    assert variant in ('native', 'euclidean'), variant
    data = ratios(path, variant)
    panel = out.rsplit('.', 1)[0] + '_panel_b.pdf'
    panel_b(data, panel)
    composite(panel, current, out)
    if png:
        subprocess.run(['pdftoppm', '-png', '-r', '220', '-singlefile', out, out.rsplit('.', 1)[0]], check=True)
    seeds = sorted({len(v) for per in data.values() for v in per.values()})
    means = {t: {d: round(float(np.mean(v)), 3) for d, v in sorted(per.items())} for t, per in data.items()}
    print(f'wrote {out} ({variant}); seeds per point: {seeds}')
    for target, per in means.items():
        print(f'   {target:20s} {per}')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], '--png' in sys.argv)
