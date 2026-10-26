"""
Post-hoc analysis of ex_17 (exact GED vs fingerprint distances, reviewer comment R2.7).

GED-balanced correlations for the random pairs: random pairs are concentrated at intermediate GED values
(e.g. QM9: 2,021 pairs at GED 7, 2 pairs at GED 19), so the ordinary correlation is dominated by these values.
Here every GED value receives the same total weight, shared equally by its pairs (weighted Pearson and
Spearman correlation). GED values with fewer than MIN_GROUP_PAIRS pairs at the tails of the distribution are
merged with the nearest value that has enough pairs, so that a handful of pairs cannot dominate.

The uncertainty is estimated exactly as in ged_exact_correlation.py (pigeonhole, pair and corrected
intervals, HDF averaged over the codebook seeds with the seeds resampled in every replicate); the balancing
weights are recomputed from the resampled pairs in every replicate.

Reads ``pairs_with_distances.csv`` of the newest finished ged_exact_correlation run with prefix ex_17 and
writes ``_ex17/analysis/balanced_results.csv`` and ``balanced_differences.csv``.

    python analyze_ex_17.py
"""
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import norm

import ged_exact_correlation as gec

PATH = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(PATH, '_ex17', 'analysis')
RESULTS = os.path.join(PATH, 'results', 'ged_exact_correlation')
SIZES = [32, 128, 512, 2048]
SEEDS = [1, 2, 3, 4, 5]
MIN_GROUP_PAIRS = 20
NUM_BOOTSTRAP = 2000
CHUNK = 200
CONFIDENCE = 0.95


def latest_archive(prefix: str = 'ex_17') -> str:
    best = None
    for meta_path in glob.glob(os.path.join(RESULTS, '*', 'experiment_meta.json')):
        with open(meta_path) as file:
            meta = json.load(file)
        params = {k: v.get('value') for k, v in meta.get('parameters', {}).items() if isinstance(v, dict)}
        if params.get('__PREFIX__') == prefix and meta.get('status') == 'done' and not meta.get('has_error'):
            if best is None or meta['start_time'] > best[1]:
                best = (os.path.dirname(meta_path), meta['start_time'])
    return best[0]


def ged_groups(ged: np.ndarray, min_pairs: int) -> np.ndarray:
    """Group index per pair: one group per GED value, sparse tail values merged into the nearest full group."""
    values, counts = np.unique(ged, return_counts=True)
    full = np.flatnonzero(counts >= min(min_pairs, counts.max()))
    low, high = values[full[0]], values[full[-1]]
    clipped = np.clip(ged, low, high)
    _, groups = np.unique(clipped, return_inverse=True)
    return groups


def balanced(base: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Rescale the weights (B, n) so that every group has the same total weight in every row."""
    onehot = np.zeros((len(groups), groups.max() + 1))
    onehot[np.arange(len(groups)), groups] = 1.0
    totals = base @ onehot
    denominator = totals[:, groups]
    return np.divide(base, denominator, out=np.zeros_like(base), where=denominator > 0)


def correlations(vector: np.ndarray, ged: np.ndarray, weights: np.ndarray, ged_ranks: np.ndarray) -> tuple:
    pearson = gec.weighted_pearson(vector, ged, weights)
    spearman = gec.weighted_pearson(gec.weighted_midranks(vector, weights), ged_ranks, weights)
    return pearson, spearman


def analyze(frame: pd.DataFrame, rng: np.random.Generator) -> tuple:
    ged = frame['ged'].to_numpy(float)
    groups = ged_groups(ged, MIN_GROUP_PAIRS)
    local = {s: i for i, s in enumerate(sorted(set(frame['smiles_1']) | set(frame['smiles_2'])))}
    local_1 = frame['smiles_1'].map(local).to_numpy()
    local_2 = frame['smiles_2'].map(local).to_numpy()
    n = len(frame)
    vectors = {('morgan', size): frame[f'morgan_{size}'].to_numpy(float) for size in SIZES}
    vectors.update({('hdf', size, seed): frame[f'hdf_{size}_s{seed}'].to_numpy(float)
                    for size in SIZES for seed in SEEDS})

    point_weights = balanced(np.ones((1, n)), groups)
    point_ranks = gec.weighted_midranks(ged, point_weights)
    point = {key: [float(v[0]) for v in correlations(vec, ged, point_weights, point_ranks)]
             for key, vec in vectors.items()}

    boot = {m: {key: np.empty((2, NUM_BOOTSTRAP)) for key in vectors} for m in ('pigeonhole', 'pair')}
    for start in range(0, NUM_BOOTSTRAP, CHUNK):
        stop = min(start + CHUNK, NUM_BOOTSTRAP)
        bases = {'pigeonhole': gec.pigeonhole_weights(local_1, local_2, len(local), stop - start, rng),
                 'pair': rng.multinomial(n, np.full(n, 1.0 / n), size=stop - start).astype(float)}
        for method, base in bases.items():
            weights = balanced(base, groups)
            ranks = gec.weighted_midranks(ged, weights)
            for key, vec in vectors.items():
                boot[method][key][:, start:stop] = np.stack(correlations(vec, ged, weights, ranks))
    seed_resamples = rng.integers(0, len(SEEDS), size=(NUM_BOOTSTRAP, len(SEEDS)))
    alpha = (1.0 - CONFIDENCE) / 2.0

    rows, diffs = [], []
    for size in SIZES:
        estimate, replicate, conditional, per_seed = {}, {}, {}, {}
        for i, name in enumerate(('pearson', 'spearman')):
            per_seed[name] = np.array([point[('hdf', size, s)][i] for s in SEEDS])
            estimate[('hdf', name)] = float(per_seed[name].mean())
            estimate[('morgan', name)] = point[('morgan', size)][i]
            for method in ('pigeonhole', 'pair'):
                stacked = np.stack([boot[method][('hdf', size, s)][i] for s in SEEDS], axis=1)
                replicate[(method, 'hdf', name)] = np.take_along_axis(stacked, seed_resamples, axis=1).mean(axis=1)
                conditional[(method, 'hdf', name)] = stacked.mean(axis=1)
                replicate[(method, 'morgan', name)] = boot[method][('morgan', size)][i]
                conditional[(method, 'morgan', name)] = boot[method][('morgan', size)][i]
        for representation in ('hdf', 'morgan'):
            row = {'representation': representation, 'size': size, 'num_pairs': n,
                   'num_groups': int(groups.max() + 1)}
            for name in ('pearson', 'spearman'):
                est = estimate[(representation, name)]
                row[name] = est
                row[f'{name}_low'], row[f'{name}_high'] = gec.percentile_interval(
                    replicate[('pigeonhole', representation, name)], alpha)
                row[f'{name}_low_pair'], row[f'{name}_high_pair'] = gec.percentile_interval(
                    replicate[('pair', representation, name)], alpha)
                variance = gec.corrected_variance(conditional[('pigeonhole', representation, name)],
                                                  conditional[('pair', representation, name)])
                if representation == 'hdf':
                    variance += gec.seed_variance_of_mean(per_seed[name])
                    row[f'{name}_seed_min'] = float(per_seed[name].min())
                    row[f'{name}_seed_max'] = float(per_seed[name].max())
                    for s, value in zip(SEEDS, per_seed[name]):
                        row[f'{name}_seed{s}'] = float(value)
                row[f'{name}_low_corr'], row[f'{name}_high_corr'] = gec.fisher_interval(est, variance, alpha)
            rows.append(row)
        diff = {'size': size, 'num_pairs': n}
        for name in ('pearson', 'spearman'):
            est = estimate[('hdf', name)] - estimate[('morgan', name)]
            diff[f'{name}_diff'] = est
            for method, suffix in (('pigeonhole', ''), ('pair', '_pair')):
                diff[f'{name}_diff_low{suffix}'], diff[f'{name}_diff_high{suffix}'] = gec.percentile_interval(
                    replicate[(method, 'hdf', name)] - replicate[(method, 'morgan', name)], alpha)
            variance = gec.corrected_variance(
                conditional[('pigeonhole', 'hdf', name)] - conditional[('pigeonhole', 'morgan', name)],
                conditional[('pair', 'hdf', name)] - conditional[('pair', 'morgan', name)],
            ) + gec.seed_variance_of_mean(per_seed[name])
            half = float(norm.ppf(1.0 - alpha)) * np.sqrt(variance)
            diff[f'{name}_diff_low_corr'], diff[f'{name}_diff_high_corr'] = est - half, est + half
        diffs.append(diff)
    return rows, diffs


def main():
    archive = latest_archive()
    print(f'exact GED results: {archive}')
    pairs = pd.read_csv(os.path.join(archive, 'pairs_with_distances.csv'))
    rng = np.random.default_rng(0)
    all_rows, all_diffs = [], []
    for dataset in ('qm9_smiles', 'zinc250k'):
        frame = pairs[(pairs['dataset'] == dataset) & (pairs['pair_set'] == 'random')].reset_index(drop=True)
        groups = ged_groups(frame['ged'].to_numpy(float), MIN_GROUP_PAIRS)
        sizes = pd.Series(groups).value_counts().sort_index()
        ged_of_group = frame.groupby(groups)['ged'].agg(['min', 'max'])
        print(f'{dataset}: {len(frame)} random pairs in {len(sizes)} GED groups: '
              + ', '.join(f'{int(lo)}' + (f'-{int(hi)}' if hi > lo else '') + f' ({c})'
                          for (lo, hi), c in zip(ged_of_group.values, sizes.values)))
        rows, diffs = analyze(frame, rng)
        for row in rows + diffs:
            row['dataset'] = dataset
            row['pair_set'] = 'random'
            row['weighting'] = 'GED-balanced'
        all_rows += rows
        all_diffs += diffs
    os.makedirs(OUT, exist_ok=True)
    results, differences = pd.DataFrame(all_rows), pd.DataFrame(all_diffs)
    results.to_csv(os.path.join(OUT, 'balanced_results.csv'), index=False)
    differences.to_csv(os.path.join(OUT, 'balanced_differences.csv'), index=False)
    print(results[['dataset', 'representation', 'size', 'pearson', 'pearson_low_corr', 'pearson_high_corr',
                   'spearman']].round(3).to_string(index=False))
    print(differences[['dataset', 'size', 'pearson_diff', 'pearson_diff_low', 'pearson_diff_high',
                       'pearson_diff_low_corr', 'pearson_diff_high_corr']].round(3).to_string(index=False))


if __name__ == '__main__':
    main()
