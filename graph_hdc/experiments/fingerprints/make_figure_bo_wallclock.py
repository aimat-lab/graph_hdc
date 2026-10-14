"""
Per-campaign wall-clock time of the original BO benchmark (ex_07_g, ``optimize_molecule_bo__*.py``) on ZINC250k,
for the runtime subsection of the SI (user, 2026-10-14: show the strong dependence on the embedding size with the
original figure, rescaled from whole jobs to single campaigns).

Each ex_07_g experiment is one SLURM job that runs NUM_TRIALS = 25 campaigns (20 initial molecules + 150 rounds of
one molecule, QED target 0.2) one after the other, with the expected improvement of every candidate computed on the
CPU and a Matern kernel for every representation. The bar height is the job's wall-clock time (``duration`` in the
experiment metadata, including the encoding of the library) divided by its number of campaigns. Jobs without a
recorded duration (aborted runs) are skipped; the figure of the original SI (bo_wallclock_.pdf, analyze_ex_07.ipynb)
showed the same jobs in days per job.

    python make_figure_bo_wallclock.py <results folder of graph_hdc/experiments/fingerprints> [output folder]
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

SIZES = [8, 16, 32, 64, 128, 512]
METHODS = ['random', 'fp', 'hdc']
FOLDERS = {'random': 'optimize_molecule_bo__random', 'fp': 'optimize_molecule_bo__fp',
           'hdc': 'optimize_molecule_bo__hdc'}
COLORS = {'random': '#808080', 'fp': '#7C92FF', 'hdc': '#69F7AE'}
LABELS = {'random': 'Random vectors', 'fp': 'Morgan FP', 'hdc': 'HDF'}
FONT_SIZE = 11


def size_of(parameters: dict) -> int:
    for key in ('EMBEDDING_SIZE', 'FINGERPRINT_SIZE', 'RANDOM_DIM'):
        if key in parameters and isinstance(parameters[key].get('value'), int):
            return parameters[key]['value']
    raise KeyError(f'no size parameter among {sorted(parameters)}')


def load(results: str) -> dict:
    """{(method, size): hours per campaign}, averaged over the completed ex_07_g jobs on ZINC250k."""
    hours = {}
    for method, folder in FOLDERS.items():
        for path in glob.glob(os.path.join(results, folder, 'ex_07_g__*', 'experiment_meta.json')):
            meta = json.load(open(path))
            params = meta['parameters']
            if params.get('DATASET_NAME', {}).get('value') != 'zinc250k' or not meta.get('duration'):
                continue
            campaigns = params['NUM_TRIALS']['value']
            hours.setdefault((method, size_of(params)), []).append(meta['duration'] / 3600 / campaigns)
    return {key: sum(values) / len(values) for key, values in hours.items()}


def main():
    results = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else '.'
    os.makedirs(out, exist_ok=True)
    hours = load(results)
    plt.style.use('default')
    if any('Roboto Condensed' in f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams['font.family'] = 'Roboto Condensed'
    plt.rcParams.update({'font.size': FONT_SIZE, 'axes.labelsize': FONT_SIZE + 1, 'pdf.fonttype': 42})
    # the size of the original figure (407 x 223 pt), which this one replaces at the same width in the SI
    fig, ax = plt.subplots(figsize=(5.65, 3.1))
    width = 0.26
    for i, size in enumerate(SIZES):
        for j, method in enumerate(METHODS):
            value = hours.get((method, size))
            if value is None:
                continue
            ax.bar(i + (j - 1) * width, value, width=width * 0.88, color=COLORS[method], edgecolor='black',
                   linewidth=1.5, alpha=0.95)
    ax.set_xticks(range(len(SIZES)))
    ax.set_xticklabels([str(s) for s in SIZES])
    ax.set_xlabel('Embedding size')
    ax.set_ylabel('Wall-clock time per campaign (h)')
    ax.grid(True, alpha=0.3, axis='y', lw=0.5)
    ax.set_axisbelow(True)
    handles = [Patch(facecolor=COLORS[m], edgecolor='black', linewidth=1.0, label=LABELS[m]) for m in METHODS]
    ax.legend(handles=handles, loc='upper left', frameon=True, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'figure_bo_wallclock.pdf'))
    fig.savefig(os.path.join(out, 'figure_bo_wallclock.png'), dpi=300)
    print('hours per campaign:')
    for size in SIZES:
        print(f'  D={size:5d}  ' + '  '.join(f'{m}={hours.get((m, size), float("nan")):6.3f}' for m in METHODS))


if __name__ == '__main__':
    main()
