"""
Analysis of the revision re-run of Experiment 08 (ex_08_b, see ``_slurm_ex_08b.py``; reviewer comment R2.7).

1. Reproduction check: the ex_08_b_repro runs (HDF with the old settings, Morgan without canonicalized seeds)
   are compared seed by seed with the original ex_08_a runs (``ged_correlation_summary.csv``).
2. New results: per-seed correlations of the ex_08_b runs (Morgan and HDF with the revision settings), as box
   plot statistics per embedding size (the data of Figure 3a of the main text), next to ex_08_a.
3. Distances per edit step: distribution of the distances between seeds and generated molecules at 1, 2 and 3
   edit steps (``ged_pairs.csv`` of the ex_08_b runs), for the SI figure. Molecules that occur twice for the same
   seed and edit step (same canonical SMILES) are counted once.

Outputs go to ``_ex08b/analysis/``: ``reproduction.csv``, ``summary.csv``, ``distances_per_step.csv`` and
``distances_per_step.pdf/png``.

    python analyze_ex_08b.py
"""
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.*')

PATH = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(PATH, 'results')
OUT = os.path.join(PATH, '_ex08b', 'analysis')
SIZES = [32, 128, 512, 2048]
NAMESPACES = {'hdc': 'molecule_similarity__hdc', 'fp': 'molecule_similarity__fp'}
SIZE_PARAMETER = {'hdc': 'EMBEDDING_SIZE', 'fp': 'FINGERPRINT_SIZE'}
COLORS = {'fp': '#4C72B0', 'hdc': '#55A868'}
LABELS = {'fp': 'Morgan', 'hdc': 'HDF'}


def load_runs(prefix: str) -> dict:
    """(encoding, size) -> archive path of the newest finished run with exactly this prefix."""
    runs = {}
    for encoding, namespace in NAMESPACES.items():
        folder = os.path.join(RESULTS, namespace)
        for name in sorted(os.listdir(folder)) if os.path.isdir(folder) else []:
            archive = os.path.join(folder, name)
            meta_path = os.path.join(archive, 'experiment_meta.json')
            if not os.path.exists(meta_path):
                continue
            with open(meta_path) as file:
                meta = json.load(file)
            params = {k: v.get('value') for k, v in meta.get('parameters', {}).items() if isinstance(v, dict)}
            if params.get('__PREFIX__') != prefix or meta.get('status') != 'done' or meta.get('has_error'):
                continue
            key = (encoding, int(params[SIZE_PARAMETER[encoding]]))
            if key not in runs or meta['start_time'] > runs[key][1]:
                runs[key] = (archive, meta['start_time'])
    return {key: archive for key, (archive, _) in runs.items()}


def per_seed(archive: str) -> pd.DataFrame:
    """Per-seed correlations of one run (without the aggregate row)."""
    frame = pd.read_csv(os.path.join(archive, 'ged_correlation_summary.csv'))
    frame = frame[frame['query_id'].astype(str) != 'AGGREGATE'].copy()
    frame['query_idx'] = frame['query_idx'].astype(int)
    frame['abs_correlation'] = frame['correlation'].astype(float).abs()
    return frame


def main():
    os.makedirs(OUT, exist_ok=True)
    original = load_runs('ex_08_a')
    repro = load_runs('ex_08_b_repro')
    new = load_runs('ex_08_b')
    print(f'runs found: ex_08_a {len(original)}, ex_08_b_repro {len(repro)}, ex_08_b {len(new)}')

    # --- 1. reproduction check ---
    rows = []
    for key in sorted(repro):
        if key not in original:
            continue
        a, b = per_seed(original[key]), per_seed(repro[key])
        merged = a.merge(b, on='query_idx', suffixes=('_a', '_b'))
        diff = (merged['correlation_a'] - merged['correlation_b']).abs()
        rows.append({'encoding': key[0], 'size': key[1], 'seeds_a': len(a), 'seeds_b': len(b),
                     'seeds_matched': len(merged), 'max_abs_diff': float(diff.max()),
                     'num_identical': int((diff < 1e-6).sum()),
                     'median_abs_r_a': float(a['abs_correlation'].median()),
                     'median_abs_r_b': float(b['abs_correlation'].median())})
    reproduction = pd.DataFrame(rows)
    reproduction.to_csv(os.path.join(OUT, 'reproduction.csv'), index=False)
    print('\nreproduction of ex_08_a (per-seed correlations):')
    print(reproduction.to_string(index=False) if len(reproduction) else '  (no runs yet)')

    # --- 2. new results next to ex_08_a ---
    rows = []
    for prefix, runs in (('ex_08_a', original), ('ex_08_b', new)):
        for key in sorted(runs):
            frame = per_seed(runs[key])
            q = frame['abs_correlation'].quantile([0.25, 0.5, 0.75])
            rows.append({'prefix': prefix, 'encoding': key[0], 'size': key[1], 'num_seeds': len(frame),
                         'q25': q[0.25], 'median': q[0.5], 'q75': q[0.75],
                         'mean_neighbors': float(frame['n_neighbors'].mean())})
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(OUT, 'summary.csv'), index=False)
    print('\n|Pearson r| per seed, quartiles over seeds:')
    print(summary.round(3).to_string(index=False) if len(summary) else '  (no runs yet)')

    # --- 3. distances per edit step (ex_08_b) ---
    pair_frames = []
    for (encoding, size), archive in sorted(new.items()):
        path = os.path.join(archive, 'ged_pairs.csv')
        if os.path.exists(path):
            frame = pd.read_csv(path)
            frame['encoding'], frame['size'] = encoding, size
            pair_frames.append(frame)
    if not pair_frames:
        print('\nno ged_pairs.csv in ex_08_b runs yet')
        return
    pairs = pd.concat(pair_frames, ignore_index=True)

    # The experiment recognizes already generated molecules by their (non-canonical) SMILES string, so the
    # same molecule can occur twice for a seed. Within one edit step such duplicates are dropped (user decision,
    # 2026-10-08); molecules that occur at two different edit steps of one seed are only counted.
    canonical = {s: Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in pairs['neighbor_smiles'].unique()}
    pairs['neighbor_canonical'] = pairs['neighbor_smiles'].map(canonical)
    run_keys = ['encoding', 'size', 'query_idx']
    before = len(pairs)
    pairs = pairs.drop_duplicates(subset=run_keys + ['edit_steps', 'neighbor_canonical']).reset_index(drop=True)
    cross_step = pairs.groupby(run_keys + ['neighbor_canonical'])['edit_steps'].nunique()
    print(f'\ndropped {before - len(pairs)} duplicate pairs within an edit step (of {before}, all runs); '
          f'{int((cross_step > 1).sum())} molecules occur at two or more edit steps of the same seed (kept)')

    stats = (pairs.groupby(['encoding', 'size', 'edit_steps'])['distance']
             .describe(percentiles=[0.25, 0.5, 0.75]).reset_index())
    stats.to_csv(os.path.join(OUT, 'distances_per_step.csv'), index=False)
    print('\ndistances per edit step (ex_08_b):')
    print(stats[['encoding', 'size', 'edit_steps', 'count', '25%', '50%', '75%']].round(3).to_string(index=False))

    # identical molecule pairs for HDF and Morgan? (expected with canonicalized seeds)
    for size in SIZES:
        sets = {enc: set(map(tuple, pairs[(pairs['encoding'] == enc) & (pairs['size'] == size)]
                             [['query_idx', 'neighbor_smiles']].values)) for enc in NAMESPACES}
        if all(sets.values()):
            print(f'size {size}: HDF and Morgan share {len(sets["hdc"] & sets["fp"])} of '
                  f'{len(sets["hdc"])} / {len(sets["fp"])} pairs')

    sizes = [s for s in SIZES if s in set(pairs['size'])]
    fig, axes = plt.subplots(1, len(sizes), figsize=(3.6 * len(sizes), 3.4), sharey=True, squeeze=False)
    for ax, size in zip(axes[0], sizes):
        for offset, encoding in ((-0.18, 'fp'), (0.18, 'hdc')):
            data = [pairs[(pairs['encoding'] == encoding) & (pairs['size'] == size) & (pairs['edit_steps'] == step)]
                    ['distance'].to_numpy() for step in (1, 2, 3)]
            if not any(len(d) for d in data):
                continue
            parts = ax.boxplot(data, positions=np.arange(1, 4) + offset, widths=0.3, patch_artist=True,
                               showfliers=False)
            for box in parts['boxes']:
                box.set_facecolor(COLORS[encoding])
                box.set_alpha(0.7)
            for median in parts['medians']:
                median.set_color('black')
        ax.set_xticks([1, 2, 3])
        ax.set_xlabel('Edit steps')
        ax.set_title(f'{size} dimensions')
        ax.grid(axis='y', alpha=0.3)
    axes[0][0].set_ylabel('Distance to seed molecule')
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[enc], alpha=0.7) for enc in ('fp', 'hdc')]
    axes[0][-1].legend(handles, [LABELS['fp'] + ' (Tanimoto)', LABELS['hdc'] + ' (cosine)'], fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'distances_per_step.pdf'))
    fig.savefig(os.path.join(OUT, 'distances_per_step.png'), dpi=150)
    print(f'\nsaved figures and tables to {OUT}')


if __name__ == '__main__':
    main()
