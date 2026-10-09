"""
Analysis of Experiment 20 (ex_20): bit-depth ablation of HDF (reviewer comment R2.2).

Pairs the runs of ``_slurm_ex_20.py`` by (dataset, seed) and divides the test MAE of the MLP at each bit depth
(QUANTIZE_BITS = 16, 8, 4, 2, 1) by the test MAE of the float32 run (QUANTIZE_BITS = None) of the same seed. All
bit depths of a seed share the split, the codebooks and the MLP seed. The float32 run with another network seed
(NN_SEED set, label 'other seed') gives the ratio that the run-to-run variation of the MLP training alone produces.
Prefer the median ratio: the shared float32 denominator is noisy, which pushes the mean ratio above 1.

    python analyze_ex_20.py [prefix] [--out FIGURE.pdf] [--spectrum gaussian|unit]   # default: ex_20_bits_unit

The archives of ex_20_bits (2026-10-09) use the original HDF codebooks with random spectral magnitudes
(SPECTRUM='gaussian'; archives from before the SPECTRUM parameter existed count as 'gaussian'). Since graph_hdc
d740800, HDF defaults to unit-modulus codebooks (SPECTRUM='unit'). The analysis never mixes the two: if a prefix
contains both, --spectrum selects one.

Writes to ``_ex20/``: ``bits_<prefix>.md`` (table), ``bits_<prefix>.csv`` (all runs) and ``figure_bits_<prefix>.pdf``
(box plots: x = bits per dimension, one box per dataset, y = MAE relative to float32). ``--out`` also copies the
figure to the given path (e.g. the paper's figures folder).
"""
import os
import sys
import csv
import shutil
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch

from analyze_ex_15 import iter_archives

PATH = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(PATH, '_ex20')
MODEL = 'neural_net2'
BITS = [1, 2, 4, 8, 16, 32]  # 32 = float32 (QUANTIZE_BITS None)
RETRAINED = 'other seed'     # float32 with another network seed (NN_SEED)
ARMS = BITS + [RETRAINED]
DATASETS = [('aqsoldb_logs', 'AqSolDB'), ('freesolv_hfe', 'FreeSolv'), ('bace_ic50', 'BACE')]

# Figure style of the paper's box plots (analyze_ex_08.ipynb / analyze_ex_07.ipynb): Roboto Condensed 11 pt, boxes
# with black edges, light y-grid, arrow in the y-label for the better direction. The datasets get colour-blind-safe
# Okabe-Ito colours (yellow, orange, reddish purple) that avoid the paper's method colours (blue = Morgan,
# green = HDF, gray = random).
COLORS = ['#F0E442', '#E69F00', '#CC79A7']
FONT_SIZE = 11
FIGSIZE = (6.0, 3.4)


def collect(prefix: str, spectrum: str = None) -> dict:
    """{(dataset, seed): {arm: mae}}; a retried run keeps the newest archive."""
    archives = sorted(iter_archives(prefix), key=lambda t: t[1].get('start_time') or 0)
    spectra = defaultdict(int)
    for _, _, params, _ in archives:
        spectra[params.get('SPECTRUM') or 'gaussian'] += 1
    if spectrum is None:
        if len(spectra) > 1:
            raise SystemExit(f'prefix {prefix} mixes HDF codebooks {dict(spectra)}; select one with --spectrum')
        spectrum = next(iter(spectra), 'gaussian')
    print(f'HDF codebooks: SPECTRUM={spectrum!r} ({spectra.get(spectrum, 0)} archives)')
    runs = defaultdict(dict)
    for _, meta, params, data in archives:
        key = f'test_{MODEL}'
        if key not in data.get('metrics', {}) or 'QUANTIZE_BITS' not in params:
            continue
        if (params.get('SPECTRUM') or 'gaussian') != spectrum:
            continue
        bits = params['QUANTIZE_BITS']
        if params.get('NN_SEED') is not None:
            arm = RETRAINED
        else:
            arm = bits if bits is not None else 32
        # identical values share a level, so heavy ties could leave fewer than 2**bits levels per dimension
        levels = data.get('quantize', {}).get('mean_levels')
        if bits is not None and bits <= 8 and levels is not None and levels < 0.99 * 2 ** bits:
            print(f'WARNING {params["NOTE"]} seed {params["SEED"]} {bits} bits: only {levels:.1f} levels per dimension')
        runs[(params['NOTE'], params['SEED'])][arm] = data['metrics'][key]['mae']
    return runs


def main(prefix: str, out: str = None, spectrum: str = None):
    runs = collect(prefix, spectrum)
    relative = {}  # (dataset, bits) -> array of MAE ratios over the complete seeds
    rows, records = [], []
    for dataset, label in DATASETS:
        seeds = sorted(s for (d, s), v in runs.items() if d == dataset and all(a in v for a in ARMS))
        incomplete = sorted(s for (d, s), v in runs.items() if d == dataset and not all(a in v for a in ARMS))
        if incomplete:
            print(f'{label}: seeds {incomplete} are incomplete and left out')
        if not seeds:
            continue
        for bits in ARMS:
            ratio = np.array([runs[(dataset, s)][bits] / runs[(dataset, s)][32] for s in seeds])
            relative[(dataset, bits)] = ratio
            mae = np.array([runs[(dataset, s)][bits] for s in seeds])
            q1, med, q3 = np.percentile(ratio, [25, 50, 75])
            rows.append([label, bits, len(seeds), f'{mae.mean():.4g} ± {mae.std():.2g}',
                         f'{med:.3f} [{q1:.3f}, {q3:.3f}]', f'{ratio.mean():.3f}'])
        for s in seeds:
            records += [[dataset, s, bits, runs[(dataset, s)][bits]] for bits in ARMS]

    header = ['Dataset', 'Bits', 'n', 'MAE', 'MAE / float32 (median [IQR])', 'MAE / float32 (mean)']
    lines = ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
    lines += ['| ' + ' | '.join(str(c) for c in row) + ' |' for row in rows]
    print(f'prefix {prefix}: {len(records)} runs in complete seeds\n')
    print('\n'.join(lines))
    if not records:
        print('no complete seeds found, existing outputs are left unchanged')
        return

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f'bits_{prefix}.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    with open(os.path.join(OUT, f'bits_{prefix}.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['dataset', 'seed', 'bits', 'mae'])
        writer.writerows(records)

    # --- figure: groups along x = bits per dimension, one box per dataset ---
    plt.style.use('default')
    if any('Roboto Condensed' in f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams['font.family'] = 'Roboto Condensed'
    plt.rcParams['font.size'] = FONT_SIZE

    present = [(d, l, c) for (d, l), c in zip(DATASETS, COLORS) if (d, BITS[0]) in relative]
    width = 0.8 / len(present)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    # gray dashed reference line with a small italic label (as in Fig. 3a), placed in the gap between the 16- and
    # 32-bit groups
    ax.axhline(1.0, color='gray', linestyle='--', linewidth=1.1, alpha=0.8, zorder=0)
    ax.annotate('float32', xy=(BITS.index(16) + 0.5, 1.0), xytext=(0, 3), textcoords='offset points', ha='center',
                va='bottom', fontsize=FONT_SIZE - 3, color='gray', style='italic')
    for j, (dataset, label, color) in enumerate(present):
        positions = [i + (j - (len(present) - 1) / 2) * width for i in range(len(ARMS))]
        ax.boxplot([relative[(dataset, b)] for b in ARMS], positions=positions, widths=width * 0.85,
                   patch_artist=True, showfliers=True, manage_ticks=False,
                   boxprops=dict(facecolor=color, alpha=0.95, edgecolor='black', linewidth=1.5),
                   medianprops=dict(color='black', linewidth=1.5),
                   whiskerprops=dict(color='black', linewidth=1.5),
                   capprops=dict(color='black', linewidth=1.5),
                   flierprops=dict(marker='o', markerfacecolor='black', markeredgecolor='black', markersize=4,
                                   alpha=0.5))
    ax.axvline(len(BITS) - 0.5, color='gray', linestyle=':', linewidth=1.2)
    ax.set_xticks(range(len(ARMS)))
    ax.set_xticklabels([{32: '32\n(float32)', RETRAINED: 'float32,\nother seed'}.get(b, str(b)) for b in ARMS])
    ax.set_xlim(-0.6, len(ARMS) - 0.4)
    ax.set_xlabel('Bits per dimension', fontsize=FONT_SIZE + 1)
    ax.set_title('Bit-depth ablation of HDF', fontsize=FONT_SIZE + 1, pad=8)
    ax.set_ylabel('MAE / MAE$_{\\mathrm{float32}}$ $\\downarrow$', fontsize=FONT_SIZE + 1)
    ax.grid(True, alpha=0.3, axis='y')
    # framed legend; upper right because the lower right would cover the boxes of the other-seed group
    handles = [Patch(facecolor=c, alpha=0.95, edgecolor='black', label=l) for _, l, c in present]
    ax.legend(handles=handles, loc='upper right', fontsize=FONT_SIZE, framealpha=0.9)
    fig.tight_layout()
    fig_path = os.path.join(OUT, f'figure_bits_{prefix}.pdf')
    fig.savefig(fig_path, bbox_inches='tight', dpi=300)
    print(f'\nfigure: {fig_path}')
    if out:
        shutil.copy(fig_path, out)
        print(f'copied to {out}')


if __name__ == '__main__':
    out = sys.argv[sys.argv.index('--out') + 1] if '--out' in sys.argv else None
    spectrum = sys.argv[sys.argv.index('--spectrum') + 1] if '--spectrum' in sys.argv else None
    args = [a for a in sys.argv[1:] if not a.startswith('--') and a not in (out, spectrum)]
    main(args[0] if args else 'ex_20_bits_unit', out, spectrum)
