"""
Build the combined SI "virtual screening" figure (figure_ged.pdf, Supplementary Figure S2).

Panel a  -- Similarity-based virtual screening (bioactivity):
            grouped bars of ROC-AUC (upper sub-axis) and EF1% (lower sub-axis) for Morgan fingerprints (ECFP4),
            HDF and HDF without the graph size and diameter encodings on four benchmark collections (Riniker 1/2,
            MUV, DUD-E); mean and standard deviation over the targets.
            Data: predict_bioactivity__{fp,hdc} archives of the re-run ex_11_b (``_slurm_ex_11b.py``: HDF with the
            standard configuration of the revision; the two HDF arms differ in GRAPH_ATTRIBUTES), file
            aggregated_results.csv. The original runs (prefix ex_11_bioact) are no longer used.

Panel b  -- Correlation with the exact graph edit distance (GED) for the random pairs (the near-pairs panel c was
            removed on 2026-10-21 together with the near-pairs analysis of the SI)
            of ZINC250k molecules with at most 12 heavy atoms (the ZINC250k panels of Supplementary Figures S5 and
            S6, SI Section S8.2): Pearson r as a function of the embedding size for Morgan fingerprints, HDF (solid)
            and HDF without the size and diameter encodings (dashed, as in Figures S5 and S6); HDF as mean and
            standard deviation over five codebooks.
            Data: ``_ex22/ged_arms_trimmed_zinc250k.csv`` and ``_ex22/ged_arms_trimmed_near.csv`` (arms "unit" and
            "unit_noglobal") with their Morgan counterparts ``_ex22/ged_morgan_trimmed_*.csv``, all from
            ``analyze_ex_22_ged.py --trim``.

Usage:
    python make_figure_ged.py                      # writes figures/figure_ged.pdf (+ .svg, .png)
    python make_figure_ged.py --out /abs/path.pdf  # also save to an explicit path
    python make_figure_ged.py --draft              # allow missing virtual-screening runs (empty bars)
"""
import glob
import json
import os
import pathlib
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

PATH = str(pathlib.Path(__file__).parent.absolute())
RESULTS_PATH = os.path.join(PATH, 'results')

# ----------------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------------

# Bioactivity datasets and display labels (panel a)
BIO_DATASETS = ['riniker_1', 'riniker_2', 'muv', 'dud_e']
BIO_DATASET_LABELS = {
    'riniker_1': 'Riniker 1',
    'riniker_2': 'Riniker 2',
    'muv': 'MUV',
    'dud_e': 'DUD-E',
}
# virtual-screening runs: the re-run ex_11_b with the full protocol (50 repetitions per target)
BIO_PREFIX = 'ex_11_b'
BIO_REPETITIONS = 50

# exact-GED panel (b): ZINC250k molecules with at most 12 heavy atoms
GED_DATASET = 'zinc250k'
GED_SIZES = [32, 128, 512, 2048]
GED_FILES = {  # pair set: (HDF arms, Morgan)
    'random': ('_ex22/ged_arms_trimmed_zinc250k.csv', '_ex22/ged_morgan_trimmed_zinc250k.csv'),
}
GED_ARM_OF = {'hdc': 'unit', 'hdc_noglobal': 'unit_noglobal'}

# Paper palette and style (main text, Supplementary Figures S5 and S6)
FONT_SIZE = 11
COLOR_FP = '#4C64EB'      # Morgan fingerprints (blue)
COLOR_HDC = '#4CEB99'     # hyperdimensional fingerprints (green)
COLOR_HDC_LIGHT = '#B8F5D6'  # HDF without size/diameter (light green, hatched bars; dashed lines)
plt.style.use('default')
if any('Roboto Condensed' in f.name for f in font_manager.fontManager.ttflist):
    plt.rcParams['font.family'] = 'Roboto Condensed'
plt.rcParams['font.size'] = FONT_SIZE
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['hatch.color'] = '#2E9E66'
plt.rcParams['hatch.linewidth'] = 0.8

# (arm, bar colour, hatch, legend label) of panel a
BIO_ARMS = [
    ('fp', COLOR_FP, None, 'Morgan FP'),
    ('hdc', COLOR_HDC, None, 'HDF (ours)'),
    ('hdc_noglobal', COLOR_HDC_LIGHT, '///', 'HDF without global size/diameter'),
]


# ----------------------------------------------------------------------------
# DATA
# ----------------------------------------------------------------------------

def load_bioactivity(draft: bool = False) -> dict:
    """(dataset, arm) -> {'auc': (mean, std), 'ef': (mean, std), 'src': archive} from the finished ex_11_b runs."""
    data = {}
    for module in ('fp', 'hdc'):
        for path in sorted(glob.glob(os.path.join(RESULTS_PATH, f'predict_bioactivity__{module}', f'{BIO_PREFIX}__*'))):
            meta = json.load(open(os.path.join(path, 'experiment_meta.json')))
            params = {p: info['value'] for p, info in meta.get('parameters', {}).items()
                      if isinstance(info, dict) and 'value' in info}
            csv_path = os.path.join(path, 'aggregated_results.csv')
            if (params.get('__PREFIX__') != BIO_PREFIX or params.get('NUM_REPETITIONS') != BIO_REPETITIONS
                    or meta.get('status') != 'done' or not os.path.exists(csv_path)):
                continue
            arm = 'fp' if module == 'fp' else ('hdc' if params.get('GRAPH_ATTRIBUTES', True) else 'hdc_noglobal')
            key = (params['DATASET_NAME'], arm)
            if key in data:
                raise ValueError(f'two finished {BIO_PREFIX} runs for {key}: {data[key]["src"]} and {path}')
            df = pd.read_csv(csv_path).set_index('metric')
            data[key] = {'auc': (float(df.loc['auc', 'mean']), float(df.loc['auc', 'std'])),
                         'ef': (float(df.loc['ef1', 'mean']), float(df.loc['ef1', 'std'])),
                         'src': os.path.relpath(path, RESULTS_PATH)}
    missing = [(d, a) for d in BIO_DATASETS for a, *_ in BIO_ARMS if (d, a) not in data]
    if missing and not draft:
        raise SystemExit(f'missing virtual-screening runs: {missing} (use --draft for a preview)')
    return data


def load_ged_exact() -> dict:
    """pair set -> {'morgan': r per size, 'hdc'/'hdc_noglobal': (mean, std) per size over the codebooks, 'n': pairs}"""
    result = {}
    for pair_set, (arms_file, morgan_file) in GED_FILES.items():
        arms = pd.read_csv(os.path.join(PATH, arms_file))
        morgan = pd.read_csv(os.path.join(PATH, morgan_file))
        arms = arms[(arms['dataset'] == GED_DATASET) & (arms['pair_set'] == pair_set)]
        morgan = morgan[(morgan['dataset'] == GED_DATASET) & (morgan['pair_set'] == pair_set)]
        entry = {'morgan': morgan.set_index('size').loc[GED_SIZES, 'pearson'].to_numpy(),
                 'n': int(morgan['num_pairs'].iloc[0])}
        for arm, name in GED_ARM_OF.items():
            stats = arms[arms['arm'] == name].groupby('size')['pearson'].agg(['mean', 'std', 'count']).loc[GED_SIZES]
            assert (stats['count'] == 5).all(), (pair_set, arm, stats)
            entry[arm] = (stats['mean'].to_numpy(), stats['std'].to_numpy())
        assert (arms['num_pairs'] == entry['n']).all()
        result[pair_set] = entry
    return result


# ----------------------------------------------------------------------------
# PLOT
# ----------------------------------------------------------------------------

def panel_letter(ax, letter: str, title: str, x: float = -0.1):
    ax.set_title(title, fontsize=FONT_SIZE + 1, pad=8)
    ax.text(x, 1.04, letter, transform=ax.transAxes, fontsize=FONT_SIZE + 4, fontweight='bold',
            va='bottom', ha='left')


def draw_screening(ax_auc, ax_ef, bio: dict):
    x = np.arange(len(BIO_DATASETS))
    bw = 0.27
    for i, (arm, color, hatch, _) in enumerate(BIO_ARMS):
        off = (i - 1) * bw
        for ax, metric in ((ax_auc, 'auc'), (ax_ef, 'ef')):
            mean = [bio.get((d, arm), {}).get(metric, (np.nan, np.nan))[0] for d in BIO_DATASETS]
            std = [bio.get((d, arm), {}).get(metric, (np.nan, np.nan))[1] for d in BIO_DATASETS]
            ax.bar(x + off, mean, width=bw * 0.92, color=color, hatch=hatch, edgecolor='black', linewidth=1.1,
                   yerr=std, capsize=2.5, error_kw={'linewidth': 1.2, 'capthick': 1.2})

    values = [v for v in bio.values()]
    auc_hi = max(v['auc'][0] + v['auc'][1] for v in values)
    auc_lo = min(v['auc'][0] - v['auc'][1] for v in values)
    ax_auc.axhline(0.5, color='gray', linestyle='--', linewidth=1.1, alpha=0.8, zorder=0)
    ax_auc.set_ylim(min(0.5, auc_lo) - 0.04, auc_hi + 0.12)
    ax_auc.set_ylabel('ROC-AUC $\\uparrow$', fontsize=FONT_SIZE + 1)
    ax_auc.grid(True, axis='y', alpha=0.3)
    ax_auc.set_axisbelow(True)
    ax_auc.tick_params(labelbottom=False)
    # the dashed lines (random ranking: ROC-AUC 0.5, EF1% 1) are explained in the caption; with three bars per
    # benchmark there is no free space for a label next to the line

    ef_hi = max(v['ef'][0] + v['ef'][1] for v in values)
    ax_ef.axhline(1.0, color='gray', linestyle='--', linewidth=1.1, alpha=0.8, zorder=0)
    ax_ef.set_ylim(0, ef_hi * 1.08)
    ax_ef.set_ylabel('EF$_{1\\%}$ $\\uparrow$', fontsize=FONT_SIZE + 1)
    ax_ef.grid(True, axis='y', alpha=0.3)
    ax_ef.set_axisbelow(True)
    ax_ef.set_xticks(x)
    ax_ef.set_xticklabels([BIO_DATASET_LABELS[d] for d in BIO_DATASETS], fontsize=FONT_SIZE)

    handles = [Patch(facecolor=color, hatch=hatch, edgecolor='black', label=label)
               for _, color, hatch, label in BIO_ARMS]
    ax_auc.legend(handles=handles, loc='upper left', fontsize=FONT_SIZE - 1, ncol=3, framealpha=0.9,
                  columnspacing=1.0, handlelength=1.6)
    panel_letter(ax_auc, 'a', 'Similarity-based virtual screening', x=-0.13)


def draw_ged(ax, entry: dict, letter: str, title: str, legend: bool):
    x = np.arange(len(GED_SIZES))
    ax.plot(x, entry['morgan'], 'o-', color=COLOR_FP, linewidth=2, markersize=7, markeredgecolor='black',
            markeredgewidth=1.0, label='Morgan FP (Radius 2)', zorder=3)
    # the two HDF variants are shifted slightly left/right so that their error bars do not overlap (as in S5/S6)
    for arm, style, alpha, shift, label in (
            ('hdc', '-', 1.0, -0.05, 'HDF (Depth 2), mean ± SD of 5 seeds'),
            ('hdc_noglobal', '--', 0.55, 0.05, 'HDF without global size/diameter')):
        mean, std = entry[arm]
        ax.errorbar(x + shift, mean, yerr=std, fmt='o', linestyle=style, color=COLOR_HDC, alpha=alpha, linewidth=2,
                    markersize=7, markeredgecolor='black', markeredgewidth=1.0, capsize=4, label=label, zorder=3)
    ax.set_xticks(x, GED_SIZES)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel('Pearson correlation w. GED $\\uparrow$', fontsize=FONT_SIZE + 1)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    if legend:
        ax.legend(loc='lower right', fontsize=FONT_SIZE - 1, framealpha=0.9)
    panel_letter(ax, letter, title)


def main():
    draft = '--draft' in sys.argv
    bio = load_bioactivity(draft)
    ged = load_ged_exact()

    print('Virtual-screening runs used:')
    for k in sorted(bio):
        print(f'  {k}: AUC={bio[k]["auc"][0]:.3f}±{bio[k]["auc"][1]:.3f}  '
              f'EF1={bio[k]["ef"][0]:.2f}±{bio[k]["ef"][1]:.2f}  [{bio[k]["src"]}]')
    for pair_set, entry in ged.items():
        print(f'Exact GED, {pair_set} pairs ({entry["n"]}): Morgan {np.round(entry["morgan"], 2)}, '
              f'HDF {np.round(entry["hdc"][0], 2)}, without size/diameter {np.round(entry["hdc_noglobal"][0], 2)}')

    fig = plt.figure(figsize=(13, 6.6))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.2)
    left = outer[0].subgridspec(2, 1, hspace=0.1)
    ax_auc = fig.add_subplot(left[0])
    ax_ef = fig.add_subplot(left[1], sharex=ax_auc)
    ax_random = fig.add_subplot(outer[1])

    draw_screening(ax_auc, ax_ef, bio)
    draw_ged(ax_random, ged['random'], 'b', 'Exact GED: random pairs', legend=True)
    ax_random.set_xlabel('Embedding size', fontsize=FONT_SIZE + 1)

    # ---- Save ----
    fig_dir = os.path.join(PATH, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    out_pdf = os.path.join(fig_dir, 'figure_ged.pdf')
    fig.savefig(out_pdf, bbox_inches='tight', dpi=300)
    fig.savefig(os.path.join(fig_dir, 'figure_ged.svg'), bbox_inches='tight')
    fig.savefig(os.path.join(fig_dir, 'figure_ged.png'), bbox_inches='tight', dpi=150)
    print(f'\nSaved {out_pdf}')

    # Optional explicit copy target
    if '--out' in sys.argv:
        dst = sys.argv[sys.argv.index('--out') + 1]
        fig.savefig(dst, bbox_inches='tight', dpi=300)
        print(f'Also saved {dst}')


if __name__ == '__main__':
    main()
