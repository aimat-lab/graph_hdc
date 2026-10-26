"""
Figures of the edit-based GED analysis with the HDF of 2026-10-09 (ex_08_c, reviewer comment R2.7), from the
outputs of ``analyze_ex_08c.py`` (``_ex08c/analysis/``). HDF: unit-modulus codebooks, bidirectional message
passing, total hydrogen counts (ex_08_c); Morgan: binary, radius 2 (ex_08_b, identical molecule pairs).

- ``figure_ged_main_panel_a``: the GED panel of main-text Figure 3, |Pearson r| between edit steps and distance
  over the 100 seed molecules, Morgan vs. HDF, as a standalone panel.
- ``figures/figure_ged_main_revised.pdf``: main-text Figure 3 with this panel as panel a, drawn in the layout and
  colours of the original panel a; panel b (KNN error ratio) is clipped unchanged from the original Inkscape
  composite ``figures/figure_ged_main.pdf``, which is only read (needs pdflatex).
- ``ged_edit_distances_per_step``: distances per edit step and embedding size (SI figure), Morgan, HDF and HDF
  without the graph size and diameter encodings.
- ``ged_edit_per_seed``: |Pearson r| per seed for Morgan, HDF and HDF without size/diameter (response figure).

Written to the paper repository (``revision/response_figures/``, ``figures/``).

    python make_figures_ged_edit.py [--smoke]
"""
import os
import shutil
import subprocess
import sys
import tempfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

PATH = os.path.dirname(os.path.abspath(__file__))
PAPER = '/media/ssd2/Programming/latex_hyperdimensional_fingerprints'
OUT = os.path.join(PAPER, 'revision', 'response_figures')
FIGURE_3_ORIGINAL = os.path.join(PAPER, 'figures', 'figure_ged_main.pdf')
FIGURE_3_REVISED = os.path.join(PAPER, 'figures', 'figure_ged_main_revised.pdf')
SIZES = [32, 128, 512, 2048]

FONT_SIZE = 11
COLOR_FP = '#4C64EB'
COLOR_HDC = '#4CEB99'
COLOR_HDC_LIGHT = '#B8F5D6'
STYLE = {
    'morgan': {'color': COLOR_FP, 'label': 'Morgan FP (Radius 2)', 'hatch': None},
    'hdf': {'color': COLOR_HDC, 'label': 'HDF (Depth 2)', 'hatch': None},
    'hdf_no_global': {'color': COLOR_HDC_LIGHT, 'label': 'HDF without global size/diameter', 'hatch': '///'},
}
plt.style.use('default')
if any('Roboto Condensed' in f.name for f in font_manager.fontManager.ttflist):
    plt.rcParams['font.family'] = 'Roboto Condensed'
plt.rcParams['font.size'] = FONT_SIZE
plt.rcParams['hatch.color'] = '#2E9E66'
plt.rcParams['hatch.linewidth'] = 0.8


def boxes(ax, data: list, positions: np.ndarray, representation: str, width: float, fliers: bool = True):
    style = STYLE[representation]
    parts = ax.boxplot(data, positions=positions, widths=width, patch_artist=True, showfliers=fliers,
                       manage_ticks=False,
                       flierprops={'marker': 'o', 'markersize': 3, 'markerfacecolor': '#555555',
                                   'markeredgecolor': 'none', 'alpha': 0.7},
                       medianprops={'color': 'black', 'linewidth': 1.2},
                       whiskerprops={'linewidth': 1.2}, capprops={'linewidth': 1.2},
                       boxprops={'linewidth': 1.2})
    for box in parts['boxes']:
        box.set_facecolor(style['color'])
        if style['hatch']:
            box.set_hatch(style['hatch'])
    return parts


def panel_letter(ax, letter: str, title: str, x: float = -0.13):
    ax.set_title(title, fontsize=FONT_SIZE + 1, pad=8)
    ax.text(x, 1.03, letter, transform=ax.transAxes, fontsize=FONT_SIZE + 4, fontweight='bold',
            va='bottom', ha='left')


def legend_handles(representations: list) -> list:
    return [Patch(facecolor=STYLE[r]['color'], edgecolor='black', hatch=STYLE[r]['hatch'], label=STYLE[r]['label'])
            for r in representations]


def figure_main_panel(per_seed: pd.DataFrame, suffix: str):
    """Same layout as panel a of figure_ged_main.pdf: Morgan vs. HDF box plots per embedding size."""
    fig, ax = plt.subplots(figsize=(3.9, 2.55))
    x = np.arange(len(SIZES))
    for representation, offset in (('morgan', -0.18), ('hdf', 0.18)):
        data = [per_seed[(per_seed['representation'] == representation) & (per_seed['size'] == s)]
                ['abs_correlation'].dropna().to_numpy() for s in SIZES]
        boxes(ax, data, x + offset, representation, width=0.32)
    ax.set_xticks(x, SIZES)
    ax.set_xlim(-0.6, len(SIZES) - 0.4)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel('Embedding Size', fontsize=FONT_SIZE)
    ax.set_ylabel('|Pearson Correlation w. GED| $\\uparrow$', fontsize=FONT_SIZE)
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(handles=legend_handles(['morgan', 'hdf']), loc='lower right', fontsize=FONT_SIZE - 2)
    save(fig, f'figure_ged_main_panel_a{suffix}')


def figure_main_composite(per_seed: pd.DataFrame):
    """Main-text Figure 3 with the new panel a. The original composite (546 x 169 pt) was assembled in Inkscape from
    matplotlib panels; panel a is redrawn here directly in its coordinates, with the box style, colours and labels of
    the original panel a (analyze_ex_08.ipynb), but in Roboto Condensed at the scale of panel b (the original panel a
    fell back to DejaVu Sans). Its frame shares the top and bottom of panel b's frame and its title the baseline of
    panel b's title. Panel b is clipped unchanged from the original composite."""
    width, height, split = 546.0, 169.0, 280.0  # pt; panel b starts at x = 283.4 pt
    scale = 0.745  # panel b: tick labels 8.2 pt = FONT_SIZE * 0.745
    colors = {'morgan': '#7C92FF', 'hdf': '#69F7AE'}
    labels = {'morgan': 'Morgan FP (Radius 2)', 'hdf': 'HDF (Depth 2)'}
    line = 1.5 * scale
    rc = {'font.size': FONT_SIZE * scale, 'axes.linewidth': 0.8 * scale, 'patch.linewidth': 1.0 * scale,
          'grid.linewidth': 0.8 * scale, 'xtick.major.size': 3.5 * scale, 'ytick.major.size': 3.5 * scale,
          'xtick.major.width': 0.8 * scale, 'ytick.major.width': 0.8 * scale, 'xtick.major.pad': 3.5 * scale,
          'ytick.major.pad': 3.5 * scale, 'axes.labelpad': 4.0 * scale, 'pdf.fonttype': 42}
    with plt.rc_context(rc):
        fig = plt.figure(figsize=(split / 72, height / 72))
        # frame in pt from the top left: top and bottom as in panel b, left spine 28 pt right of the panel letter
        # (as in panel b), right end as in the original panel a
        left, right, top, bottom = 28.0, 259.3, 25.0, 142.3
        ax = fig.add_axes((left / split, 1 - bottom / height, (right - left) / split, (bottom - top) / height))
        for i, size in enumerate(SIZES):
            for representation, offset in (('morgan', -0.175), ('hdf', 0.175)):
                values = per_seed[(per_seed['representation'] == representation) & (per_seed['size'] == size)]
                ax.boxplot([values['abs_correlation'].dropna().to_numpy()], positions=[i + offset],
                           widths=0.35 * 0.85, patch_artist=True, showfliers=True, manage_ticks=False,
                           boxprops={'facecolor': colors[representation], 'alpha': 0.95, 'edgecolor': 'black',
                                     'linewidth': line},
                           medianprops={'color': 'black', 'linewidth': line},
                           whiskerprops={'color': 'black', 'linewidth': line},
                           capprops={'color': 'black', 'linewidth': line},
                           flierprops={'marker': 'o', 'markerfacecolor': 'black', 'markersize': 4 * scale,
                                       'alpha': 0.5})
        ax.set_xticks(range(len(SIZES)), SIZES)
        ax.set_xlim(-0.675, len(SIZES) - 0.325)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel('Embedding Size', fontsize=(FONT_SIZE + 1.5) * scale)
        ax.set_ylabel('|Pearson Correlation w. GED| $\\uparrow$', fontsize=(FONT_SIZE + 1.5) * scale)
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(handles=[Patch(facecolor=colors[r], alpha=0.95, edgecolor='black', label=labels[r])
                           for r in ('morgan', 'hdf')], loc='lower right')
        baseline = 1 - 7.54 / height
        fig.text(0.0, baseline, 'a', fontsize=10, fontweight='bold', va='baseline', ha='left')
        fig.text(8.45 / split, baseline, 'Correlation of Molecular Representations with Graph Edit Distance (GED)',
                 fontsize=8, va='baseline', ha='left')
        with tempfile.TemporaryDirectory() as folder:
            fig.savefig(os.path.join(folder, 'panel_a.pdf'))
            plt.close(fig)
            shutil.copy(FIGURE_3_ORIGINAL, os.path.join(folder, 'original.pdf'))
            with open(os.path.join(folder, 'figure.tex'), 'w') as file:
                file.write('\n'.join([
                    r'\documentclass[border=0pt]{standalone}',
                    r'\usepackage{graphicx}',
                    r'\begin{document}',
                    r'\setlength{\unitlength}{1bp}%',
                    rf'\begin{{picture}}({width:g},{height:g})',
                    rf'\put({split:g},0){{\includegraphics[trim={split:g}bp 0bp 0bp 0bp,clip]{{original.pdf}}}}',
                    r'\put(0,0){\includegraphics{panel_a.pdf}}',
                    r'\end{picture}',
                    r'\end{document}',
                ]))
            subprocess.run(['pdflatex', '-interaction=nonstopmode', '-halt-on-error', 'figure.tex'], cwd=folder,
                           check=True, stdout=subprocess.DEVNULL)
            shutil.copy(os.path.join(folder, 'figure.pdf'), FIGURE_3_REVISED)
    print(f'saved {FIGURE_3_REVISED}')


def figure_per_seed(per_seed: pd.DataFrame, suffix: str):
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    x = np.arange(len(SIZES))
    layout = (('morgan', -0.25), ('hdf', 0.0), ('hdf_no_global', 0.25))
    for representation, offset in layout:
        data = [per_seed[(per_seed['representation'] == representation) & (per_seed['size'] == s)]
                ['abs_correlation'].dropna().to_numpy() for s in SIZES]
        boxes(ax, data, x + offset, representation, width=0.22)
    ax.set_xticks(x, SIZES)
    ax.set_xlim(-0.6, len(SIZES) - 0.4)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel('Embedding size', fontsize=FONT_SIZE + 1)
    ax.set_ylabel('|Pearson correlation w. edit steps| $\\uparrow$', fontsize=FONT_SIZE + 1)
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(handles=legend_handles([r for r, _ in layout]), loc='lower right', fontsize=FONT_SIZE - 1)
    ax.set_title('Correlation with the number of edit steps (100 seed molecules)', fontsize=FONT_SIZE + 1, pad=8)
    save(fig, f'ged_edit_per_seed{suffix}')


def figure_distances_per_step(pairs: pd.DataFrame, suffix: str):
    """One row per representation (own distance scale: Tanimoto for Morgan, cosine for HDF), one column per D."""
    sizes = [s for s in SIZES if s in set(pairs['size'])]
    rows = ('morgan', 'hdf', 'hdf_no_global')
    ylabels = {'morgan': 'Tanimoto distance', 'hdf': 'Cosine distance', 'hdf_no_global': 'Cosine distance'}
    fig, axes = plt.subplots(len(rows), len(sizes), figsize=(2.9 * len(sizes), 2.5 * len(rows)), sharey='row',
                             squeeze=False)
    for i, representation in enumerate(rows):
        for k, size in enumerate(sizes):
            ax = axes[i][k]
            part = pairs[pairs['size'] == size]
            data = [part[part['edit_steps'] == step][representation].to_numpy() for step in (1, 2, 3)]
            boxes(ax, data, np.arange(1, 4), representation, width=0.55, fliers=False)
            ax.set_xticks([1, 2, 3], ['1', '2', '3'] if i == len(rows) - 1 else ['', '', ''])
            ax.set_xlim(0.4, 3.6)
            ax.grid(True, axis='y', alpha=0.3)
            ax.set_axisbelow(True)
            if i == 0:
                ax.set_title(f'D = {size}', fontsize=FONT_SIZE + 1, pad=8)
            if i == len(rows) - 1:
                ax.set_xlabel('Edit steps', fontsize=FONT_SIZE + 1)
            if k == 0:
                ax.set_ylabel(ylabels[representation], fontsize=FONT_SIZE + 1)
                ax.text(-0.42, 1.05, 'abc'[i], transform=ax.transAxes, fontsize=FONT_SIZE + 4, fontweight='bold',
                        va='bottom', ha='left')
        # the row shares its y axis: scale to the data of all columns first, then start at zero
        top = max(line.get_ydata().max() for ax in axes[i] for line in ax.lines if len(line.get_ydata()))
        axes[i][0].set_ylim(0.0, 1.05 * top)
    fig.legend(handles=legend_handles(list(rows)), loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.04),
               fontsize=FONT_SIZE, frameon=True)
    fig.tight_layout(h_pad=1.2, w_pad=0.8, rect=(0, 0.03, 1, 1))
    save(fig, f'ged_edit_distances_per_step{suffix}')


def save(fig, name: str):
    os.makedirs(OUT, exist_ok=True)
    for extension in ('pdf', 'png'):
        fig.savefig(os.path.join(OUT, f'{name}.{extension}'), bbox_inches='tight',
                    dpi=200 if extension == 'png' else None)
    plt.close(fig)
    print(f'saved {name}.pdf/png')


def main(smoke: bool = False):
    folder = os.path.join(PATH, '_ex08c', 'analysis_smoke' if smoke else 'analysis')
    suffix = '_smoke' if smoke else ''
    per_seed = pd.read_csv(os.path.join(folder, 'per_seed.csv'))
    pairs = pd.read_csv(os.path.join(folder, 'pairs_deduplicated.csv'))
    figure_main_panel(per_seed, suffix)
    if not smoke:
        figure_main_composite(per_seed)
    figure_per_seed(per_seed, suffix)
    figure_distances_per_step(pairs, suffix)


if __name__ == '__main__':
    main(smoke='--smoke' in sys.argv)
