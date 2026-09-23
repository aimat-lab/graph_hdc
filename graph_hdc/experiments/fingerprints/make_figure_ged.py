"""
Build the combined SI "virtual screening" figure (figure_ged.pdf).

Panel a  -- Similarity-based virtual screening (bioactivity):
            grouped bars of ROC-AUC (upper sub-axis) and EF1% (lower sub-axis)
            comparing Morgan fingerprints (ECFP4) vs hyperdimensional fingerprints (HDF)
            on four benchmark collections (Riniker 1/2, MUV, DUD-E).
            Data: predict_bioactivity__{fp,hdc} experiment archives (prefix ex_11_bioact),
            file aggregated_results.csv.

Panel b  -- Correlation with graph edit distance (GED):
            box plots of |Pearson correlation| between fingerprint distance and GED,
            as a function of embedding size, for Morgan vs HDF.
            Data: molecule_similarity__{fp,hdc} experiment archives (prefix ex_08_a),
            file ged_correlation_summary.csv  (REUSED, not re-run).

Usage:
    python make_figure_ged.py                      # writes figures/figure_ged.pdf
    python make_figure_ged.py --out /abs/path.pdf  # also copy to an explicit path
"""
import os
import sys
import json
import pathlib
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from pycomex.utils import is_experiment_archive
from pycomex.functional.experiment import Experiment

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
# Only consider bioactivity runs with this prefix (the fresh ex_11 benchmark). If a
# combination is missing under this prefix, fall back to the most recent run for it.
BIO_PREFIX = 'ex_11_bioact'
EF_METRIC = 'ef1'   # enrichment factor at 1%

# GED panel (panel b)
GED_PREFIX = 'ex_08_a'
GED_EMBEDDING_SIZES = [32, 128, 512, 2048]

# Paper palette (matches main text + GED box-plot legend)
COLOR_FP = '#4C64EB'    # Morgan fingerprints (blue)
COLOR_HDC = '#4CEB99'   # hyperdimensional fingerprints (green)

FONT_SIZE = 11
try:
    plt.rcParams['font.family'] = 'Roboto Condensed'
except Exception:
    pass
plt.rcParams['font.size'] = FONT_SIZE
plt.rcParams['svg.fonttype'] = 'none'


# ----------------------------------------------------------------------------
# ARCHIVE DISCOVERY
# ----------------------------------------------------------------------------

def iter_archives():
    """Yield (archive_path, metadata_dict, params_dict) for every experiment archive."""
    if not os.path.isdir(RESULTS_PATH):
        return
    for ns in os.listdir(RESULTS_PATH):
        ns_path = os.path.join(RESULTS_PATH, ns)
        if not os.path.isdir(ns_path):
            continue
        for dirpath, dirnames, _ in os.walk(ns_path):
            if is_experiment_archive(dirpath):
                dirnames.clear()
                meta_path = os.path.join(dirpath, Experiment.METADATA_FILE_NAME)
                if not os.path.exists(meta_path):
                    continue
                try:
                    meta = json.loads(open(meta_path).read())
                except Exception:
                    continue
                params = {
                    p: info['value']
                    for p, info in meta.get('parameters', {}).items()
                    if isinstance(info, dict) and 'value' in info
                }
                yield dirpath, meta, params


def encoding_of(name):
    if '__fp' in name or '_fp_' in name or 'fp_' in name:
        return 'fp'
    if '__hdc' in name or '_hdc_' in name or 'hdc_' in name:
        return 'hdc'
    return 'unknown'


# ----------------------------------------------------------------------------
# LOAD BIOACTIVITY DATA (panel a)
# ----------------------------------------------------------------------------

def load_bioactivity():
    # Collect candidate archives per (dataset, encoding): list of (is_preferred, end_time, path)
    candidates = defaultdict(list)
    for path, meta, params in iter_archives():
        name = meta.get('name', '')
        if 'bioactivity' not in name:
            continue
        dataset = params.get('DATASET_NAME', '')
        if dataset not in BIO_DATASETS:
            continue
        enc = encoding_of(name)
        if enc == 'unknown':
            continue
        prefix = str(params.get('__PREFIX__', ''))
        end_time = meta.get('end_time') or 0
        candidates[(dataset, enc)].append((prefix == BIO_PREFIX, end_time, path))

    data = {}
    for key, lst in candidates.items():
        # Only consider runs that actually produced an aggregated_results.csv, so an
        # incomplete or failed preferred run cannot blank out a combination.
        lst = [c for c in lst
               if os.path.exists(os.path.join(c[2], 'aggregated_results.csv'))]
        if not lst:
            continue
        preferred = [c for c in lst if c[0]]
        pool = preferred if preferred else lst
        pool.sort(key=lambda c: c[1], reverse=True)
        _, _, path = pool[0]
        csv_path = os.path.join(path, 'aggregated_results.csv')
        df = pd.read_csv(csv_path)
        def grab(metric):
            row = df[df['metric'] == metric]
            if len(row) == 0:
                return None, None
            return float(row['mean'].values[0]), float(row['std'].values[0])
        auc_m, auc_s = grab('auc')
        ef_m, ef_s = grab(EF_METRIC)
        data[key] = {
            'auc': (auc_m, auc_s),
            'ef': (ef_m, ef_s),
            'src': os.path.relpath(path, RESULTS_PATH),
        }
    return data


# ----------------------------------------------------------------------------
# LOAD GED DATA (panel b)
# ----------------------------------------------------------------------------

def load_ged():
    # (encoding, size) -> list of |correlation| values across queries
    data = defaultdict(list)
    for path, meta, params in iter_archives():
        if str(params.get('__PREFIX__', '')) != GED_PREFIX:
            continue
        name = meta.get('name', '')
        enc = encoding_of(name)
        if enc == 'fp':
            size = params.get('FINGERPRINT_SIZE', 0)
        elif enc == 'hdc':
            size = params.get('EMBEDDING_SIZE', 0)
        else:
            continue
        csv_path = os.path.join(path, 'ged_correlation_summary.csv')
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        df = df[df['query_id'] != 'AGGREGATE']
        if 'correlation' in df.columns:
            vals = df['correlation'].astype(float).tolist()
            data[(enc, int(size))].extend([abs(v) for v in vals])
    return data


# ----------------------------------------------------------------------------
# PLOT
# ----------------------------------------------------------------------------

def main():
    bio = load_bioactivity()
    ged = load_ged()

    print('Bioactivity sources used:')
    for k in sorted(bio):
        print(f'  {k}: AUC={bio[k]["auc"][0]:.3f}±{bio[k]["auc"][1]:.3f}  '
              f'EF1={bio[k]["ef"][0]:.2f}±{bio[k]["ef"][1]:.2f}  [{bio[k]["src"]}]')
    print('GED configs:', {k: len(v) for k, v in sorted(ged.items())})

    fig = plt.figure(figsize=(13, 5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.1],
                          height_ratios=[1, 1], hspace=0.16, wspace=0.2)
    ax_auc = fig.add_subplot(gs[0, 0])
    ax_ef = fig.add_subplot(gs[1, 0], sharex=ax_auc)
    ax_ged = fig.add_subplot(gs[:, 1])

    # ---- Panel a: bioactivity grouped bars ----
    encodings = [('fp', COLOR_FP, 'Morgan FP'), ('hdc', COLOR_HDC, 'HDF')]
    x = np.arange(len(BIO_DATASETS))
    bw = 0.38

    for enc_idx, (enc, color, _label) in enumerate(encodings):
        off = (enc_idx - 0.5) * bw
        auc_m = [bio.get((d, enc), {}).get('auc', (np.nan, np.nan))[0] for d in BIO_DATASETS]
        auc_s = [bio.get((d, enc), {}).get('auc', (np.nan, np.nan))[1] for d in BIO_DATASETS]
        ef_m = [bio.get((d, enc), {}).get('ef', (np.nan, np.nan))[0] for d in BIO_DATASETS]
        ef_s = [bio.get((d, enc), {}).get('ef', (np.nan, np.nan))[1] for d in BIO_DATASETS]
        ax_auc.bar(x + off, auc_m, width=bw * 0.92, color=color, edgecolor='black',
                   linewidth=1.1, yerr=auc_s, capsize=3,
                   error_kw={'linewidth': 1.3, 'capthick': 1.3})
        ax_ef.bar(x + off, ef_m, width=bw * 0.92, color=color, edgecolor='black',
                  linewidth=1.1, yerr=ef_s, capsize=3,
                  error_kw={'linewidth': 1.3, 'capthick': 1.3})

    # AUC sub-axis (upper)
    auc_all = [bio[k]['auc'][0] + bio[k]['auc'][1] for k in bio]
    auc_lo = [bio[k]['auc'][0] - bio[k]['auc'][1] for k in bio]
    ax_auc.axhline(0.5, color='gray', linestyle='--', linewidth=1.1, alpha=0.8, zorder=0)
    ax_auc.set_ylim(min(0.5, min(auc_lo)) - 0.04, max(auc_all) + 0.04)
    ax_auc.set_ylabel('ROC-AUC $\\uparrow$', fontsize=FONT_SIZE + 1)
    ax_auc.grid(True, axis='y', alpha=0.25)
    ax_auc.tick_params(labelbottom=False)
    ax_auc.annotate('random', xy=(len(BIO_DATASETS) - 0.5, 0.5), xytext=(0, 2),
                    textcoords='offset points', ha='right', va='bottom',
                    fontsize=FONT_SIZE - 3, color='gray', style='italic')

    # EF sub-axis (lower)
    ef_all = [bio[k]['ef'][0] + bio[k]['ef'][1] for k in bio]
    ax_ef.axhline(1.0, color='gray', linestyle='--', linewidth=1.1, alpha=0.8, zorder=0)
    ax_ef.set_ylim(0, max(ef_all) * 1.12)
    ax_ef.set_ylabel('EF$_{1\\%}$ $\\uparrow$', fontsize=FONT_SIZE + 1)
    ax_ef.grid(True, axis='y', alpha=0.25)
    ax_ef.set_xticks(x)
    ax_ef.set_xticklabels([BIO_DATASET_LABELS[d] for d in BIO_DATASETS], fontsize=FONT_SIZE)

    # Panel a legend + title
    a_handles = [Patch(facecolor=COLOR_FP, edgecolor='black', label='Morgan FP'),
                 Patch(facecolor=COLOR_HDC, edgecolor='black', label='HDF (ours)')]
    ax_auc.legend(handles=a_handles, loc='upper right', fontsize=FONT_SIZE - 1,
                  ncol=2, framealpha=0.9, columnspacing=1.0, handlelength=1.2)
    ax_auc.set_title('Similarity-based virtual screening',
                     fontsize=FONT_SIZE + 2, pad=8)
    ax_auc.text(-0.16, 1.04, 'a', transform=ax_auc.transAxes,
                fontsize=FONT_SIZE + 6, fontweight='bold', va='bottom', ha='left')

    # ---- Panel b: GED box plots ----
    box_w = 0.36
    for size_idx, size in enumerate(GED_EMBEDDING_SIZES):
        for enc_idx, (enc, color) in enumerate([('fp', COLOR_FP), ('hdc', COLOR_HDC)]):
            vals = ged.get((enc, size), [])
            if not vals:
                continue
            pos = size_idx + (enc_idx - 0.5) * box_w
            ax_ged.boxplot(
                [vals], positions=[pos], widths=box_w * 0.85, patch_artist=True,
                showfliers=True,
                boxprops=dict(facecolor=color, alpha=0.95, edgecolor='black', linewidth=1.3),
                medianprops=dict(color='black', linewidth=1.3),
                whiskerprops=dict(color='black', linewidth=1.3),
                capprops=dict(color='black', linewidth=1.3),
                flierprops=dict(marker='o', markerfacecolor='black', markersize=3, alpha=0.4),
            )
    ax_ged.set_xticks(range(len(GED_EMBEDDING_SIZES)))
    ax_ged.set_xticklabels(GED_EMBEDDING_SIZES, fontsize=FONT_SIZE)
    ax_ged.set_xlabel('Embedding size', fontsize=FONT_SIZE + 1)
    ax_ged.set_ylabel('$|$Pearson correlation w. GED$|$ $\\uparrow$', fontsize=FONT_SIZE + 1)
    ax_ged.set_ylim(0, 1.05)
    ax_ged.grid(True, axis='y', alpha=0.25)
    b_handles = [Patch(facecolor=COLOR_FP, alpha=0.95, edgecolor='black', label='Morgan FP (Radius 2)'),
                 Patch(facecolor=COLOR_HDC, alpha=0.95, edgecolor='black', label='HDF (Depth 2)')]
    ax_ged.legend(handles=b_handles, loc='lower right', fontsize=FONT_SIZE - 1)
    ax_ged.set_title('Correlation with graph edit distance',
                     fontsize=FONT_SIZE + 2, pad=8)
    ax_ged.text(-0.1, 1.04, 'b', transform=ax_ged.transAxes,
                fontsize=FONT_SIZE + 6, fontweight='bold', va='bottom', ha='left')

    # ---- Save ----
    fig_dir = os.path.join(PATH, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    out_pdf = os.path.join(fig_dir, 'figure_ged.pdf')
    fig.savefig(out_pdf, bbox_inches='tight', dpi=300)
    fig.savefig(os.path.join(fig_dir, 'figure_ged.svg'), bbox_inches='tight')
    print(f'\nSaved {out_pdf}')

    # Optional explicit copy target
    if '--out' in sys.argv:
        dst = sys.argv[sys.argv.index('--out') + 1]
        fig.savefig(dst, bbox_inches='tight', dpi=300)
        print(f'Also saved {dst}')


if __name__ == '__main__':
    main()
