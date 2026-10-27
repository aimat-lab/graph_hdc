"""
Exact GED vs fingerprint distances on one fixed, GED-balanced subset of pairs (ex_17, reviewer comment R2.7).

For each pair set (random, near) and dataset (QM9, ZINC250k <= 12 heavy atoms), one subset is drawn once
(fixed seed): at most SUBSET_CAP pairs per exact GED value, so that the GED values are represented nearly
equally (values with fewer pairs contribute all of them). On this subset, the Pearson correlation between
distance and exact GED is computed once for Morgan fingerprints and once per HDF codebook seed, both for HDF
and for HDF without the global graph size and diameter encodings.

HDF is the encoder of 2026-10-09 (unit-modulus codebooks, bidirectional message passing, total hydrogen counts).
Its correlations come from ``analyze_ex_22_ged.py`` (ex_22; arms "unit" and "unit_noglobal", computed on exactly
these subsets), copied to ``_ex22/ged_arms.csv`` and ``_ex22/ged_morgan.csv``. The correlations of the original
encoder (gaussian codebooks, ex_17) stay cached in ``_ex17/analysis/balanced_subset_correlations.csv``.

Random pairs (user decision 2026-10-12): the sparse tails of the GED distribution are cut off, so that the GED
values are represented nearly uniformly: QM9 GED 4-15 and ZINC250k GED 7-16 of the same draw (TRIM of
analyze_ex_22_ged.py; correlations from its ``--trim`` run, ``_ex22/ged_arms_trimmed_*.csv``). Near pairs: the full
subset (GED 1-4). In all subsets, pairs with GED 0 (distinct molecules with identical heavy-atom graphs; one pair of
ZINC250k stereoisomers) are left out (user decision 2026-10-12). The figures no longer mark the cap of 100 pairs per GED value; the text describes how
the subsets were built.

Figures (to the paper repository's revision/response_figures/, styled like main-text Figure 3):
``ged_exact_random_balanced`` and ``ged_exact_near_balanced``; the same as SI figures (one legend, PDF) in the paper's
figures/ folder: ``figure_si_ged_exact_random`` and ``figure_si_ged_exact_near``. Left: number of pairs per GED value in the
subset. Right: Pearson r vs embedding size.

The original-encoder correlations are cached in ``_ex17/analysis/balanced_subset_correlations.csv`` (delete it
to recompute them; the figure only takes the numbers of available pairs from it).

    python make_figure_ged_balanced_subset.py
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

import analyze_ex_17
import analyze_ex_17_hdf
import ged_exact_correlation as gec
from analyze_ex_22_ged import TRIM

PATH = os.path.dirname(os.path.abspath(__file__))
# HDF_DATA_ROOT: folder with the results/ archives and the _ex17/_ex22 tables (default: this folder)
DATA_ROOT = os.environ.get('HDF_DATA_ROOT', PATH)
analyze_ex_17.RESULTS = os.path.join(DATA_ROOT, 'results', 'ged_exact_correlation')
OUT = '/media/ssd2/Programming/latex_hyperdimensional_fingerprints/revision/response_figures'
SI_OUT = os.environ.get('SI_OUT', '/media/ssd2/Programming/latex_hyperdimensional_fingerprints/figures')
CACHE = os.path.join(DATA_ROOT, '_ex17', 'analysis', 'balanced_subset_correlations.csv')
SUBSETS = os.path.join(DATA_ROOT, '_ex17', 'analysis', 'balanced_subset_{dataset}_{pair_set}.csv')
ARMS = os.path.join(DATA_ROOT, '_ex22', 'ged_arms.csv')
MORGAN = os.path.join(DATA_ROOT, '_ex22', 'ged_morgan.csv')
ARMS_TRIMMED = os.path.join(DATA_ROOT, '_ex22', 'ged_arms_trimmed_{dataset}.csv')
MORGAN_TRIMMED = os.path.join(DATA_ROOT, '_ex22', 'ged_morgan_trimmed_{dataset}.csv')
ARMS_NEAR = os.path.join(DATA_ROOT, '_ex22', 'ged_arms_trimmed_near.csv')
MORGAN_NEAR = os.path.join(DATA_ROOT, '_ex22', 'ged_morgan_trimmed_near.csv')
# figure representation -> ex_22 arm (HDF of 2026-10-09 with and without the size and diameter encodings)
ARM_OF = {'hdf': 'unit', 'hdf_no_global': 'unit_noglobal'}

SUBSET_CAP = 100
SUBSET_SEED = 0
SIZES = [32, 128, 512, 2048]
SEEDS = [1, 2, 3, 4, 5]
STATS = {'qm9_smiles': {'max_size': 9, 'max_diameter': 8}, 'zinc250k': {'max_size': 38, 'max_diameter': 23}}
DATASETS = [('qm9_smiles', 'QM9'), ('zinc250k', 'ZINC250k (≤ 12 heavy atoms)')]
PAIR_SET_LABEL = {'random': 'random pairs', 'near': 'near pairs'}

# paper style (analyze_ex_08.ipynb, make_figure_ged.py)
FONT_SIZE = 11
COLOR_FP = '#4C64EB'
COLOR_HDC = '#4CEB99'
plt.style.use('default')
if any('Roboto Condensed' in f.name for f in font_manager.fontManager.ttflist):
    plt.rcParams['font.family'] = 'Roboto Condensed'
plt.rcParams['font.size'] = FONT_SIZE
plt.rcParams['pdf.fonttype'] = 42  # TrueType: Type 3 glyph widths differ between viewers (user, 2026-10-27)


def balanced_subset(frame: pd.DataFrame) -> pd.DataFrame:
    """At most SUBSET_CAP pairs per GED value, drawn once with a fixed seed."""
    rng = np.random.default_rng(SUBSET_SEED)
    chosen = []
    for _, group in frame.groupby('ged'):
        chosen.extend(rng.choice(group.index.to_numpy(), size=min(SUBSET_CAP, len(group)), replace=False))
    return frame.loc[sorted(chosen)].reset_index(drop=True)


def correlations(pairs: pd.DataFrame) -> pd.DataFrame:
    if os.path.exists(CACHE):
        return pd.read_csv(CACHE)
    rows = []
    for pair_set in ('random', 'near'):
        for dataset, _ in DATASETS:
            frame = pairs[(pairs['dataset'] == dataset) & (pairs['pair_set'] == pair_set)].reset_index(drop=True)
            subset = balanced_subset(frame)
            subset.to_csv(SUBSETS.format(dataset=dataset, pair_set=pair_set), index=False)
            ged = subset['ged'].to_numpy(float)
            base = {'dataset': dataset, 'pair_set': pair_set, 'num_pairs': len(subset), 'num_available': len(frame)}
            smiles = sorted(set(subset['smiles_1']) | set(subset['smiles_2']))
            index = {s: i for i, s in enumerate(smiles)}
            i1, i2 = subset['smiles_1'].map(index).to_numpy(), subset['smiles_2'].map(index).to_numpy()
            print(f'{dataset} / {pair_set}: subset of {len(subset)} of {len(frame)} pairs, {len(smiles)} molecules')
            for size in SIZES:
                rows.append({**base, 'representation': 'morgan', 'size': size, 'seed': None,
                             'pearson': float(np.corrcoef(subset[f'morgan_{size}'], ged)[0, 1])})
                for seed in SEEDS:
                    rows.append({**base, 'representation': 'hdf', 'size': size, 'seed': seed,
                                 'pearson': float(np.corrcoef(subset[f'hdf_{size}_s{seed}'], ged)[0, 1])})
                    embedding = analyze_ex_17_hdf.encode(smiles, size, seed, 'no_attributes', STATS[dataset])
                    distance = gec.cosine_distance(embedding[i1], embedding[i2])
                    rows.append({**base, 'representation': 'hdf_no_global', 'size': size, 'seed': seed,
                                 'pearson': float(np.corrcoef(distance, ged)[0, 1])})
                print(f'  size {size} done', flush=True)
    table = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    table.to_csv(CACHE, index=False)
    return table


def load_subset(dataset: str, pair_set: str) -> pd.DataFrame:
    """The fixed subset of a pair set without GED-0 pairs; random pairs restricted to the central GED range TRIM."""
    subset = pd.read_csv(SUBSETS.format(dataset=dataset, pair_set=pair_set))
    subset = subset[subset['ged'] > 0].reset_index(drop=True)
    if pair_set == 'random':
        low, high = TRIM[dataset]
        subset = subset[subset['ged'].between(low, high)].reset_index(drop=True)
    return subset


def current_correlations(pairs: pd.DataFrame) -> pd.DataFrame:
    """Correlations of the current HDF (ex_22 arms) and Morgan on the fixed subsets (the --trim runs of
    analyze_ex_22_ged.py); ``pairs`` (all ex_17 pairs) gives the number of available pairs without GED 0."""
    arms, morgan = pd.read_csv(ARMS_NEAR), pd.read_csv(MORGAN_NEAR)
    for dataset, _ in DATASETS:
        arms = pd.concat([arms, pd.read_csv(ARMS_TRIMMED.format(dataset=dataset))], ignore_index=True)
        morgan = pd.concat([morgan, pd.read_csv(MORGAN_TRIMMED.format(dataset=dataset))], ignore_index=True)
    available = pairs[pairs['ged'] > 0].groupby(['dataset', 'pair_set']).size()
    rows = []
    for representation, arm in ARM_OF.items():
        for r in arms[arms['arm'] == arm].itertuples():
            rows.append({'dataset': r.dataset, 'pair_set': r.pair_set, 'num_pairs': r.num_pairs,
                         'representation': representation, 'size': r.size, 'seed': r.seed, 'pearson': r.pearson})
    for r in morgan.itertuples():
        rows.append({'dataset': r.dataset, 'pair_set': r.pair_set, 'num_pairs': r.num_pairs,
                     'representation': 'morgan', 'size': r.size, 'seed': None, 'pearson': r.pearson})
    table = pd.DataFrame(rows)
    table['num_available'] = [available[(d, p)] for d, p in zip(table['dataset'], table['pair_set'])]
    # the subsets of the --trim runs are those of load_subset: same pair counts, same Morgan correlations
    for pair_set in ('random', 'near'):
        for dataset, _ in DATASETS:
            subset = load_subset(dataset, pair_set)
            for size in SIZES:
                row = table[(table['representation'] == 'morgan') & (table['dataset'] == dataset)
                            & (table['pair_set'] == pair_set) & (table['size'] == size)]
                assert len(row) == 1 and int(row['num_pairs'].iloc[0]) == len(subset)
                assert np.isclose(row['pearson'].iloc[0], np.corrcoef(subset[f'morgan_{size}'], subset['ged'])[0, 1])
    return table


def panel_letter(ax, letter: str, title: str):
    ax.set_title(title, fontsize=FONT_SIZE + 1, pad=8)
    ax.text(-0.13, 1.03, letter, transform=ax.transAxes, fontsize=FONT_SIZE + 4, fontweight='bold',
            va='bottom', ha='left')


def draw_row(ax_left, ax_right, dataset: str, label: str, pair_set: str, table: pd.DataFrame, letters,
             legend: bool = True, short_title: bool = False):
    """Left: GED distribution of the fixed subset. Right: Pearson r with the exact GED vs. embedding size."""
    x = np.arange(len(SIZES))
    subset = load_subset(dataset, pair_set)
    values, counts = np.unique(subset['ged'].astype(int), return_counts=True)
    part = table[(table['dataset'] == dataset) & (table['pair_set'] == pair_set)]

    ax = ax_left
    ax.bar(values, counts, width=0.75, color='#C8C8C8', edgecolor='black', linewidth=1.0)
    ax.set_xticks(values)
    ax.set_xlabel('Exact GED', fontsize=FONT_SIZE + 1)
    ax.set_ylabel('Number of pairs', fontsize=FONT_SIZE + 1)
    ax.set_ylim(0, SUBSET_CAP * 1.1)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)
    # SI figure: only the dataset in the title (the pair counts are given in the text; user, 2026-10-27)
    panel_letter(ax, next(letters), label if short_title else
                 f'{label}: {len(subset):,} of {int(part["num_available"].iloc[0]):,} {PAIR_SET_LABEL[pair_set]}')

    ax = ax_right
    morgan = part[part['representation'] == 'morgan'].set_index('size').loc[SIZES, 'pearson']
    ax.plot(x, morgan, 'o-', color=COLOR_FP, linewidth=2, markersize=7, markeredgecolor='black',
            markeredgewidth=1.0, label='Morgan FP (Radius 2)', zorder=3)
    # the two HDF variants are shifted slightly left/right so that their error bars do not overlap
    for representation, style, alpha, shift, text in (
            ('hdf', '-', 1.0, -0.05, 'HDF (Depth 2), mean ± SD of 5 seeds'),
            ('hdf_no_global', '--', 0.55, 0.05, 'HDF without global size/diameter')):
        stats = part[part['representation'] == representation].groupby('size')['pearson'].agg(['mean', 'std']).loc[SIZES]
        ax.errorbar(x + shift, stats['mean'], yerr=stats['std'], fmt='o', linestyle=style, color=COLOR_HDC,
                    alpha=alpha, linewidth=2, markersize=7, markeredgecolor='black', markeredgewidth=1.0,
                    capsize=4, label=text, zorder=3)
    ax.set_xticks(x, SIZES)
    ax.set_xlabel('Embedding size', fontsize=FONT_SIZE + 1)
    ax.set_ylabel('Pearson correlation with GED $\\uparrow$', fontsize=FONT_SIZE + 1)
    low = min(part['pearson'].min(), 0.0) if pair_set == 'near' else 0.0
    ax.set_ylim(low, 1.0)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    if legend:
        # near pairs: the curves run low, so the legend goes to the empty upper left
        ax.legend(loc='lower right' if pair_set == 'random' else 'upper left', fontsize=FONT_SIZE - 1)
    panel_letter(ax, next(letters), f'Correlation with exact GED ({label.split(" (")[0]})')


def save(fig, folder: str, name: str):
    os.makedirs(folder, exist_ok=True)
    # the paper's figure folder only gets the PDF
    for extension in (('pdf',) if folder == SI_OUT else ('pdf', 'png')):
        fig.savefig(os.path.join(folder, f'{name}.{extension}'), bbox_inches='tight',
                    dpi=200 if extension == 'png' else None)
    plt.close(fig)
    print(f'saved {name}.pdf/png')


def figure(pair_set: str, table: pd.DataFrame, si: bool = False):
    """One pair set, QM9 and ZINC250k. Response figure, or with ``si`` the SI figure (one legend, PDF only)."""
    # the SI figure is drawn close to its printed size (the full SI text width), so that the text prints at about 8-9 pt
    fig, axes = plt.subplots(2, 2, figsize=(8.85, 6.6) if si else (10, 7.4), gridspec_kw={'width_ratios': [1.1, 1]})
    letters = iter('abcd')
    for row, (dataset, label) in enumerate(DATASETS):
        draw_row(axes[row][0], axes[row][1], dataset, label, pair_set, table, letters, legend=(row == 0 or not si),
                 short_title=si)
    fig.tight_layout(h_pad=2.2, w_pad=3.0)
    if si:
        save(fig, SI_OUT, f'figure_si_ged_exact_{pair_set}')
    else:
        save(fig, OUT, f'ged_exact_{pair_set}_balanced')


def main():
    archive = analyze_ex_17.latest_archive()
    pairs = pd.read_csv(os.path.join(archive, 'pairs_with_distances.csv'))
    table = current_correlations(pairs)
    # --si: only the SI figure(s), leaving the response figures as they are
    for pair_set in ('random', 'near'):
        if '--si' not in sys.argv:
            figure(pair_set, table)
        figure(pair_set, table, si=True)
    summary = (table.groupby(['pair_set', 'dataset', 'representation', 'size'])['pearson'].agg(['mean', 'std', 'count'])
               .round(3).reset_index())
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
