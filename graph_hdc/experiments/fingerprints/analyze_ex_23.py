"""
Figure and table of ex_23: training-free recovery of the atom composition from HDF vectors (decode_composition.py).

Collects results.csv of all decode_composition archives with the prefix PREFIX and writes to _ex23/:

- figure_si_decode.pdf/png: exact recovery of the atom-type set and the atom-type counts, and the share of atoms
  with the correct type, over the dimension D (depth L = 2) for the final vector, the vector without the size and
  diameter encodings, and the layer-0 readout; one row per molecule set; markers = mean over the codebook seeds,
  error bars = range over the seeds.
- decode_composition_table.tex: the values at D = 2048 for the depths L = 1, 2, 3 (SI table).
- summary.csv: mean, min and max over the seeds of every metric.

Usage:
    python analyze_ex_23.py [--prefix ex_23] [--out-figure PATH.pdf] [--out-table PATH.tex]
"""
import argparse
import glob
import os
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullFormatter

PATH = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(PATH, 'results', 'decode_composition')
OUT = os.path.join(PATH, '_ex23')

DATASETS = [('qm9_smiles', 'QM9'), ('zinc250k', 'ZINC250k')]
# the final vector in the HDF colour of the paper; the two reference vectors in Okabe-Ito colours that avoid the
# method colours (blue = Morgan, green = HDF)
VECTORS = [('final', 'Final embedding', '#4CEB99'),
           ('structure', 'Without size and diameter', '#E69F00'),
           ('layer0', 'Layer-0 readout', '#CC79A7')]
# titles say what has to be recovered: the set of atom types present (a, d), additionally the number of atoms of each
# type, i.e. the full composition (b, e), and the share of atoms whose type is recovered (c, f)
METRICS = [('exact_set', 'Which atom types are present', 'Molecules recovered exactly (%)'),
           ('exact_counts', 'How many atoms of each type', 'Molecules recovered exactly (%)'),
           ('atoms_correct', 'Type of each atom', 'Atoms with correct type (%)')]

FONT_SIZE = 11
DEPTH = 2           # depth of the dimension sweep
TABLE_DIM = 2048    # dimension of the depth sweep (table)


def collect(prefix: str) -> pd.DataFrame:
    frames = []
    for path in sorted(glob.glob(os.path.join(RESULTS, f'{prefix}__*', 'results.csv'))):
        frames.append(pd.read_csv(path))
    if not frames:
        raise SystemExit(f'no results.csv in {RESULTS}/{prefix}__*')
    results = pd.concat(frames, ignore_index=True)
    duplicates = results.duplicated(['dataset', 'dim', 'depth', 'seed', 'vector'])
    if duplicates.any():
        raise SystemExit(f'{duplicates.sum()} duplicate rows: several archives cover the same configuration')
    return results


def figure(summary: pd.DataFrame, counts: dict, path: str):
    plt.style.use('default')
    if any('Roboto Condensed' in f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams['font.family'] = 'Roboto Condensed'
    plt.rcParams['font.size'] = FONT_SIZE

    fig, axes = plt.subplots(len(DATASETS), len(METRICS), figsize=(10, 6.4), sharex=True)
    letters = iter('abcdefghij')
    for row, (dataset, dataset_label) in enumerate(DATASETS):
        for col, (metric, title, ylabel) in enumerate(METRICS):
            ax = axes[row, col]
            for vector, _, color in VECTORS[::-1]:   # the final vector last, on top
                data = summary[(summary['dataset'] == dataset) & (summary['depth'] == DEPTH) &
                               (summary['vector'] == vector)].sort_values('dim')
                mean = 100 * data[(metric, 'mean')].to_numpy()
                low = mean - 100 * data[(metric, 'min')].to_numpy()
                high = 100 * data[(metric, 'max')].to_numpy() - mean
                ax.errorbar(data['dim'], mean, yerr=[low, high], color=color, linewidth=2, capsize=3,
                            marker='o', markersize=6, markeredgecolor='black', markeredgewidth=0.8, zorder=3)
            ax.set_xscale('log', base=2)
            dims = sorted(summary['dim'].unique())
            ax.xaxis.set_major_locator(FixedLocator(dims))
            ax.set_xticklabels([str(d) for d in dims])
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.set_xlim(min(dims) / 1.35, max(dims) * 1.35)
            ax.set_ylim(-4, 104)
            ax.grid(True, alpha=0.3, linestyle='--')
            if row == 0:
                ax.set_title(title, fontsize=FONT_SIZE + 1)
            if row == len(DATASETS) - 1:
                ax.set_xlabel('Embedding size $D$', fontsize=FONT_SIZE + 1)
            ax.set_ylabel(ylabel, fontsize=FONT_SIZE + 1)
            ax.text(-0.24, 1.06, next(letters), transform=ax.transAxes, fontsize=FONT_SIZE + 4, fontweight='bold')
            if col == len(METRICS) - 1:
                n, mean_atoms = counts[dataset]
                ax.text(0.97, 0.05, f'{n:,} molecules\n{mean_atoms:.1f} heavy atoms on average',
                        transform=ax.transAxes, ha='right', va='bottom', fontsize=FONT_SIZE - 1,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='lightgray', alpha=0.9))
            if col == 0:   # row label outside the axes, left of the y-axis label
                ax.text(-0.42, 0.5, dataset_label, transform=ax.transAxes, ha='center', va='center', rotation=90,
                        fontsize=FONT_SIZE + 2, fontweight='bold')
    handles = [Line2D([], [], color=c, linewidth=2, marker='o', markersize=6, markeredgecolor='black',
                      markeredgewidth=0.8, label=label) for _, label, c in VECTORS]
    fig.tight_layout()
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=len(VECTORS),
               fontsize=FONT_SIZE, frameon=True)
    fig.savefig(path, bbox_inches='tight', dpi=300)
    fig.savefig(path.replace('.pdf', '.png'), bbox_inches='tight', dpi=150)
    plt.close(fig)


def table(summary: pd.DataFrame, path: str, table_dim: int = TABLE_DIM):
    """D = 2048: rows = molecule set x depth; columns = the three composition metrics for the final vector and the
    vector without size and diameter. The layer-0 readout does not depend on L and is given once per molecule set.
    The atom count (100% at D = 2048 for every depth) is reported in the text and the caption, not as a column."""
    pct = lambda v: f'{100 * v:.1f}'
    lines = ['% Generated by graph_hdc/experiments/fingerprints/analyze_ex_23.py from the ex_23 results (do not edit by hand).',
             r'\begin{tabular}{llcccccc}', r'\toprule',
             r' & & \multicolumn{2}{c}{Types present} & \multicolumn{2}{c}{Atoms per type} & '
             r'\multicolumn{2}{c}{Type of each atom} \\',
             r'\cmidrule(lr){3-4} \cmidrule(lr){5-6} \cmidrule(lr){7-8}',
             r'Dataset & & final & w/o global & final & w/o global & final & w/o global \\',
             r'\midrule']
    for k, (dataset, label) in enumerate(DATASETS):
        sub = summary[(summary['dataset'] == dataset) & (summary['dim'] == table_dim)]
        depths = sorted(sub['depth'].unique())
        for i, depth in enumerate(depths):
            final = sub[(sub['depth'] == depth) & (sub['vector'] == 'final')].iloc[0]
            struct = sub[(sub['depth'] == depth) & (sub['vector'] == 'structure')].iloc[0]
            cells = []
            for metric in ('exact_set', 'exact_counts', 'atoms_correct'):
                cells += [pct(final[(metric, 'mean')]), pct(struct[(metric, 'mean')])]
            first = rf'\multirow{{{len(depths) + 1}}}{{*}}{{{label}}}' if i == 0 else ''
            lines.append(f'{first} & $L = {depth}$ & ' + ' & '.join(cells) + r' \\')
        layer0 = sub[(sub['depth'] == DEPTH) & (sub['vector'] == 'layer0')].iloc[0]
        cells = []
        for metric in ('exact_set', 'exact_counts', 'atoms_correct'):
            cells += [r'\multicolumn{2}{c}{' + pct(layer0[(metric, 'mean')]) + '}']
        lines.append(' & Layer-0 readout & ' + ' & '.join(cells) + r' \\')
        if k < len(DATASETS) - 1:
            lines.append(r'\midrule')
    lines += [r'\bottomrule', r'\end{tabular}']
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', default='ex_23')
    parser.add_argument('--out-figure', default=None)
    parser.add_argument('--out-table', default=None)
    parser.add_argument('--table-dim', type=int, default=TABLE_DIM)
    args = parser.parse_args()

    results = collect(args.prefix)
    metrics = ['atom_count', 'exact_set', 'exact_counts', 'atoms_correct']
    summary = results.groupby(['dataset', 'dim', 'depth', 'vector'])[metrics].agg(['mean', 'min', 'max']).reset_index()
    seeds = results.groupby(['dataset', 'dim', 'depth', 'vector'])['seed'].nunique()
    print(f'{len(results)} rows; seeds per configuration: {sorted(seeds.unique())}')

    molecules = {}
    for path in sorted(glob.glob(os.path.join(RESULTS, f'{args.prefix}__*', 'molecules.csv'))):
        frame = pd.read_csv(path, usecols=['dataset', 'molecule', 'num_atoms']).drop_duplicates()
        for dataset, group in frame.groupby('dataset'):
            molecules[dataset] = (group['molecule'].nunique(), group.drop_duplicates('molecule')['num_atoms'].mean())

    os.makedirs(OUT, exist_ok=True)
    flat = summary.copy()
    flat.columns = ['_'.join(c).strip('_') if isinstance(c, tuple) else c for c in flat.columns]
    flat.to_csv(os.path.join(OUT, 'summary.csv'), index=False)
    with pd.option_context('display.width', 220, 'display.max_columns', 30, 'display.max_rows', 200):
        print(summary.round(3).to_string())

    fig_path = os.path.join(OUT, 'figure_si_decode.pdf')
    figure(summary, molecules, fig_path)
    tab_path = os.path.join(OUT, 'decode_composition_table.tex')
    table(summary, tab_path, args.table_dim)
    print(f'figure: {fig_path}\ntable: {tab_path}')
    if args.out_figure:
        shutil.copy(fig_path, args.out_figure)
        print(f'copied to {args.out_figure}')
    if args.out_table:
        shutil.copy(tab_path, args.out_table)
        print(f'copied to {args.out_table}')


if __name__ == '__main__':
    main()
