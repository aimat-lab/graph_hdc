"""
Correlation between fingerprint distances and the exact graph edit distance (ex_17, reviewer comment R2.7).

This experiment collects the ``pairs.csv`` files of all ``ged_exact_pairs.py`` runs whose archive prefix is
PAIRS_PREFIX and evaluates, for every molecule set (DATASET_NAME, MAX_HEAVY_ATOMS) and pair set ('random',
'near'), how strongly the distances of molecular fingerprints correlate with the exact GED:

- HDF: hyperdimensional fingerprints in continuous mode with bidirectional message passing, the total
  hydrogen count of each atom (HYDROGEN_COUNT), depth
  NUM_LAYERS and the codebook seeds HDF_SEEDS, compared with the cosine distance. As in the other fingerprint
  experiments, the graph size and diameter encodings are scaled by the maximum graph size and diameter of
  the full dataset (all valid, connected molecules with at least one bond), not of the size-filtered subset.
- Morgan: binary Morgan fingerprints with radius MORGAN_RADIUS, compared with the Tanimoto distance.

Both are evaluated at every size in EMBEDDING_SIZES, HDF additionally with every codebook seed in HDF_SEEDS.
For each combination we report the Pearson (headline) and Spearman correlation between distance and GED over
all pairs of a pair set; for HDF, the correlation is averaged over the codebook seeds. Three kinds of
confidence intervals are reported, all paired across representations and sizes (same resamples):

- ``*_low/_high`` (headline): percentile intervals of a molecule-level ("pigeonhole") bootstrap. The molecules
  of a pair set are drawn with replacement and every pair is weighted with the product of the multiplicities
  of its molecules, so that pairs sharing a molecule are resampled together. Conservative (too wide) where
  molecules rarely repeat, because the product weights inflate the pair-level variance.
- ``*_low_pair/_high_pair``: percentile intervals of the ordinary pair bootstrap, which ignores shared
  molecules (too narrow where molecules repeat often, as for near pairs around a few molecules).
- ``*_low_corr/_high_corr``: normal intervals (Fisher z for correlations) from the corrected variance
  V_pigeonhole - 2 V_pair (see ``corrected_variance``), plus the codebook variance of the seed average for HDF.

For the bootstrap intervals, every resample also redraws the HDF codebook seeds with replacement and averages
over them, so the HDF intervals cover the codebook randomness of the reported seed average. The spread of
HDF over the codebook seeds is reported separately (``*_seed_sd``, ``*_seed_min/max``, ``seeds.csv``).

Pairs whose GED search did not finish within the time limit are excluded and counted; as a sensitivity
check, the random pairs are also evaluated with the unfinished ones at their upper bound (pair set
'random_incl_unfinished'). The near pair set consists of the screened pairs with a finished search and a GED
of at most NEAR_THRESHOLD. The experiment stops if a chunk of a pair set is missing or incomplete, unless
ALLOW_INCOMPLETE is set.

Run (after all ged_exact_pairs.py chunks are done):
    OMP_NUM_THREADS=2 python ged_exact_correlation.py --__DEBUG__="False" --__PREFIX__="'ex_17'" --PAIRS_PREFIX="'ex_17'"

Outputs in the archive folder: ``results.csv`` (one row per molecule set, pair set, representation and
size), ``differences.csv`` (HDF - Morgan per size), ``seeds.csv`` (HDF per codebook seed),
``pairs_with_distances.csv`` and ``ged_exact.pdf/png``
(panels a-c as planned for the SI figure).
"""
import csv
import glob
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from scipy.stats import norm

from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

from graph_hdc.models import HyperNet
from graph_hdc.special.molecules import (
    graph_dict_from_mol,
    make_molecule_node_encoder_map_cont,
    make_molecule_graph_encoder_map_cont,
)

# == INPUT PARAMETERS ==

# :param PAIRS_PREFIX:
#       The archive prefix (__PREFIX__) of the ged_exact_pairs.py runs whose pairs are evaluated.
PAIRS_PREFIX: str = 'ex_17'

# :param MOLECULE_SETS:
#       The molecule sets to evaluate as (DATASET_NAME, MAX_HEAVY_ATOMS) pairs, matching the parameters of
#       the ged_exact_pairs.py runs.
MOLECULE_SETS: list = [('qm9_smiles', 9), ('zinc250k', 12)]

# :param NEAR_THRESHOLD:
#       Pairs of the 'near' pair set with a GED of at most this value form the near pairs.
NEAR_THRESHOLD: float = 4.0

# :param ALLOW_INCOMPLETE:
#       By default the experiment stops with an error if a chunk of a pair set is missing, unfinished
#       (e.g. a job killed at its time limit) or has fewer rows than expected, so that no headline number is
#       computed from partial data by accident. Set to True to evaluate the available pairs anyway.
ALLOW_INCOMPLETE: bool = False

# == REPRESENTATION PARAMETERS ==

# :param EMBEDDING_SIZES:
#       The sizes of HDF and Morgan fingerprints at which the correlations are evaluated.
EMBEDDING_SIZES: list = [32, 128, 512, 2048]

# :param NUM_LAYERS:
#       The message-passing depth of HDF.
NUM_LAYERS: int = 2

# :param HDF_SEEDS:
#       The seeds of the HDF codebooks. HDF distances between dissimilar molecules depend noticeably on the
#       random codebooks, so HDF is evaluated with every seed: the reported HDF correlation is the average over
#       the seeds, and every bootstrap resample additionally draws one seed, so that the confidence intervals
#       cover both the sampling of the pairs and the randomness of the codebooks. Seed 1 is the default of the
#       other fingerprint experiments.
HDF_SEEDS: list = [1, 2, 3, 4, 5]

# :param BOOTSTRAP_CHUNK:
#       The number of bootstrap resamples that are evaluated at once (limits the memory use).
BOOTSTRAP_CHUNK: int = 200

# :param BIDIRECTIONAL:
#       Whether HDF passes messages along both directions of every bond (see molecule_similarity__hdc.py).
BIDIRECTIONAL: bool = True

# :param HYDROGEN_COUNT:
#       How the hydrogen count of each atom is determined (see molecule_similarity__hdc.py). "total" counts
#       all bonded hydrogens (RDKit GetTotalNumHs), as described in the paper. "implicit" (the behavior before
#       this parameter existed) only counts implicit hydrogens, which is 0 for every atom written in brackets.
HYDROGEN_COUNT: str = 'total'

# :param SPECTRUM:
#       Fourier magnitudes of the random HDF codebook vectors (see predict_molecules__hdc.py). "unit" (default
#       since 2026-10-09): unit magnitudes with random phases. "gaussian": the original encoder, which the ex_17
#       runs of 2026-10-08 used (before this parameter existed).
SPECTRUM: str = 'unit'

# :param MORGAN_RADIUS:
#       The radius of the binary Morgan fingerprints.
MORGAN_RADIUS: int = 2

# :param BATCH_SIZE:
#       The batch size of the HyperNet forward pass.
BATCH_SIZE: int = 600

# :param DEVICE:
#       The device of the HyperNet encoder.
DEVICE: str = 'cpu'

# == STATISTICS PARAMETERS ==

# :param NUM_BOOTSTRAP:
#       The number of bootstrap resamples for the confidence intervals.
NUM_BOOTSTRAP: int = 2000

# :param BOOTSTRAP_SEED:
#       The seed of the bootstrap resampling.
BOOTSTRAP_SEED: int = 0

# :param CONFIDENCE:
#       The level of the percentile bootstrap confidence intervals.
CONFIDENCE: float = 0.95

# == EXPERIMENT PARAMETERS ==

__DEBUG__: bool = True
__TESTING__: bool = False

# :param PAIRS_RESULTS_PATH:
#       The results folder of ged_exact_pairs.py, which contains the archives of the GED runs.
PAIRS_RESULTS_PATH: str = os.path.join(folder_path(__file__), 'results', 'ged_exact_pairs')

# :param COLOR_MORGAN:
#       Plot color of Morgan fingerprints (blue, as in the paper figures).
COLOR_MORGAN: str = '#4C72B0'

# :param COLOR_HDF:
#       Plot color of HDF (green, as in the paper figures).
COLOR_HDF: str = '#55A868'


def collect_pairs(e: Experiment) -> Dict[Tuple[str, int, str], pd.DataFrame]:
    """
    All pairs of the ged_exact_pairs runs with prefix PAIRS_PREFIX, grouped by (dataset, max heavy atoms,
    pair set). Every chunk has to be present and complete, i.e. its run finished without error (pycomex
    status 'done') and its pairs.csv holds all pairs of the chunk; otherwise the experiment stops unless
    ALLOW_INCOMPLETE is set. If a chunk was run more than once, the newest complete run is used.
    """
    chunks: Dict[tuple, dict] = defaultdict(lambda: defaultdict(list))
    for meta_path in sorted(glob.glob(os.path.join(PAIRS_RESULTS_PATH, '*', 'experiment_meta.json'))):
        archive = os.path.dirname(meta_path)
        with open(meta_path) as file:
            meta = json.load(file)
        params = {k: v.get('value') for k, v in meta.get('parameters', {}).items() if isinstance(v, dict)}
        if params.get('__PREFIX__') != e.PAIRS_PREFIX:
            continue
        key = (params['DATASET_NAME'], int(params['MAX_HEAVY_ATOMS']), params['PAIR_SET'])
        csv_path = os.path.join(archive, 'pairs.csv')
        frame = pd.read_csv(csv_path) if os.path.exists(csv_path) else pd.DataFrame()
        # the chunk boundaries follow from the parameters (same formula as ged_exact_pairs.py), so they are
        # also known for runs that were killed before pycomex wrote experiment_data.json
        num_pairs, num_chunks = int(params['NUM_PAIRS']), int(params['NUM_CHUNKS'])
        chunk_index = int(params['CHUNK_INDEX'])
        chunk_size = -(-num_pairs // num_chunks)
        begin = chunk_index * chunk_size
        expected = max(0, min(begin + chunk_size, num_pairs) - begin)
        done = meta.get('status') == 'done' and not meta.get('has_error', False)
        chunks[key][chunk_index].append({
            'archive': archive, 'frame': frame, 'expected': expected, 'start_time': meta.get('start_time', 0),
            'complete': done and len(frame) >= expected, 'num_chunks': num_chunks, 'num_pairs': num_pairs,
        })

    pair_sets: Dict[tuple, pd.DataFrame] = {}
    for key, by_index in sorted(chunks.items()):
        num_chunks = {entry['num_chunks'] for entries in by_index.values() for entry in entries}
        num_pairs = {entry['num_pairs'] for entries in by_index.values() for entry in entries}
        if len(num_chunks) != 1 or len(num_pairs) != 1:
            raise ValueError(f'{key}: chunks with different NUM_CHUNKS/NUM_PAIRS: {num_chunks}, {num_pairs}')
        chosen = {}
        for chunk_index, entries in by_index.items():
            complete = [entry for entry in entries if entry['complete']]
            # newest complete run; without a complete run (ALLOW_INCOMPLETE only) the run with the most rows
            chosen[chunk_index] = (max(complete, key=lambda x: x['start_time']) if complete
                                   else max(entries, key=lambda x: (len(x['frame']), x['start_time'])))
        missing = sorted(set(range(num_chunks.pop())) - set(chosen))
        incomplete = sorted(index for index, entry in chosen.items() if not entry['complete'])
        if missing or incomplete:
            message = f'{key}: missing chunks {missing}, incomplete chunks {incomplete}'
            if not e.ALLOW_INCOMPLETE:
                raise RuntimeError(message + ' (set ALLOW_INCOMPLETE=True to evaluate the available pairs)')
            e.log(f'WARNING: {message}')
        by_index = chosen
        frames = [entry['frame'] for _, entry in sorted(by_index.items()) if len(entry['frame']) > 0]
        if not frames:
            e.log(f'WARNING: {key}: no pairs available')
            continue
        frame = pd.concat(frames, ignore_index=True)
        frame = frame.drop_duplicates(subset='pair_index').sort_values('pair_index').reset_index(drop=True)
        e[f'pairs/{key[0]}/{key[2]}/num_chunks_found'] = len(by_index)
        e[f'pairs/{key[0]}/{key[2]}/missing_chunks'] = missing
        e[f'pairs/{key[0]}/{key[2]}/incomplete_chunks'] = sorted(incomplete)
        pair_sets[key] = frame
    return pair_sets


def dataset_statistics(dataset_name: str) -> dict:
    """Maximum graph size and diameter over the valid, connected molecules of the full dataset."""
    from chem_mat_data import load_smiles_dataset
    df = load_smiles_dataset(dataset_name)
    column = 'smiles' if 'smiles' in df.columns else df.columns[0]
    max_size = max_diameter = 0
    for smiles in df[column]:
        smiles = str(smiles)
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or mol.GetNumAtoms() < 2 or mol.GetNumBonds() < 1 or '.' in smiles:
            continue
        max_size = max(max_size, mol.GetNumAtoms())
        max_diameter = max(max_diameter, int(Chem.GetDistanceMatrix(mol).max()))
    return {'max_size': max_size, 'max_diameter': max_diameter}


def encode_hdf(smiles_list: List[str], dim: int, stats: dict, e: Experiment, seed: int) -> np.ndarray:
    """HDF embeddings (rows) of the molecules with codebook seed ``seed``, built like in molecule_similarity__hdc.py."""
    if e.SPECTRUM not in ('gaussian', 'unit'):
        raise ValueError(f'SPECTRUM must be "gaussian" or "unit", got {e.SPECTRUM!r}')
    hyper_net = HyperNet(
        hidden_dim=dim,
        depth=e.NUM_LAYERS,
        device=e.DEVICE,
        node_encoder_map=make_molecule_node_encoder_map_cont(dim=dim, seed=seed, unit_modulus=(e.SPECTRUM == 'unit')),
        graph_encoder_map=make_molecule_graph_encoder_map_cont(
            dim=dim,
            seed=seed,
            max_graph_size=stats['max_size'],
            max_graph_diameter=stats['max_diameter'],
            unit_modulus=(e.SPECTRUM == 'unit'),
        ),
        seed=seed,
        normalize_all=True,
        bidirectional=e.BIDIRECTIONAL,
    )
    graphs = []
    for smiles in smiles_list:
        graph = graph_dict_from_mol(Chem.MolFromSmiles(smiles), hydrogens=e.HYDROGEN_COUNT)
        graph.pop('graph_labels', None)
        graphs.append(graph)
    results = hyper_net.forward_graphs(graphs, batch_size=e.BATCH_SIZE)
    return np.stack([np.asarray(result['graph_embedding'], dtype=float) for result in results])


def encode_morgan(smiles_list: List[str], dim: int, radius: int) -> np.ndarray:
    """Binary Morgan fingerprints (rows) of the molecules."""
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=dim)
    return np.stack([np.array(generator.GetFingerprint(Chem.MolFromSmiles(s)), dtype=float)
                     for s in smiles_list])


def cosine_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return 1.0 - np.sum(a * b, axis=1) / norm


def tanimoto_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    dot = np.sum(a * b, axis=1)
    union = np.sum(a * a, axis=1) + np.sum(b * b, axis=1) - dot
    return 1.0 - dot / union


def weighted_pearson(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Weighted Pearson correlation of x and y (each of shape (n,) or (B, n)) for every row of the non-negative
    weights w (B, n). With integer weights this equals the Pearson correlation of the sample in which
    element k occurs w[b, k] times.
    """
    sw = w.sum(axis=-1, keepdims=True)
    dx = x - (w * x).sum(axis=-1, keepdims=True) / sw
    dy = y - (w * y).sum(axis=-1, keepdims=True) / sw
    return (w * dx * dy).sum(axis=-1) / np.sqrt((w * dx * dx).sum(axis=-1) * (w * dy * dy).sum(axis=-1))


def weighted_midranks(values: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Average ranks of ``values`` (n,) in the sample in which element k occurs w[b, k] times, for every row b
    of the integer weights w (B, n). For unit weights this equals ``scipy.stats.rankdata(values)``.
    """
    order = np.argsort(values, kind='stable')
    sorted_values = values[order]
    new_group = np.ones(len(values), dtype=bool)
    new_group[1:] = sorted_values[1:] != sorted_values[:-1]
    starts = np.flatnonzero(new_group)
    group = np.cumsum(new_group) - 1
    group_weight = np.add.reduceat(w[:, order], starts, axis=1).astype(float)
    midrank = np.cumsum(group_weight, axis=1) - group_weight + (group_weight + 1.0) / 2.0
    ranks = np.empty(w.shape, dtype=float)
    ranks[:, order] = midrank[:, group]
    return ranks


def pigeonhole_weights(index_1: np.ndarray, index_2: np.ndarray, num_molecules: int, num_resamples: int,
                       rng: np.random.Generator) -> np.ndarray:
    """
    Pair weights (num_resamples, n) of the molecule-level ("pigeonhole") bootstrap (Owen, Ann. Appl. Stat.
    2007): the molecules are drawn with replacement, and every pair is weighted with the product of the
    multiplicities of its two molecules, so that pairs sharing a molecule are resampled together.
    """
    counts = rng.multinomial(num_molecules, np.full(num_molecules, 1.0 / num_molecules), size=num_resamples)
    return (counts[:, index_1] * counts[:, index_2]).astype(float)


def percentile_interval(replicates: np.ndarray, alpha: float) -> Tuple[float, float]:
    """Percentile bootstrap interval."""
    return float(np.nanquantile(replicates, alpha)), float(np.nanquantile(replicates, 1.0 - alpha))


def corrected_variance(pigeonhole: np.ndarray, pair: np.ndarray) -> float:
    """
    Variance of an estimate from its pigeonhole and pair bootstrap replicates. The pair weights of the
    pigeonhole bootstrap (products of two multiplicities) have about three times the variance of the weights of
    the pair bootstrap, so the pigeonhole variance counts the pair-level variance about three times. Removing
    two pair-bootstrap variances corrects this; the result is floored at the pair-bootstrap variance.
    """
    v_pair = float(np.nanvar(pair))
    return max(float(np.nanvar(pigeonhole)) - 2.0 * v_pair, v_pair)


def seed_variance_of_mean(per_seed: np.ndarray) -> float:
    """Variance of the mean over the codebook seeds due to the codebook randomness."""
    return float(np.var(per_seed, ddof=1)) / len(per_seed) if len(per_seed) > 1 else 0.0


def fisher_interval(estimate: float, variance: float, alpha: float) -> Tuple[float, float]:
    """Normal interval of a correlation on the Fisher z scale, given the variance of the correlation."""
    estimate = float(np.clip(estimate, -0.999999, 0.999999))
    se_z = np.sqrt(variance) / (1.0 - estimate ** 2)
    z, q = np.arctanh(estimate), float(norm.ppf(1.0 - alpha))
    return float(np.tanh(z - q * se_z)), float(np.tanh(z + q * se_z))


@Experiment(
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)
def experiment(e: Experiment):

    @e.testing
    def testing(e: Experiment):
        e.NUM_BOOTSTRAP = 50
        e.EMBEDDING_SIZES = [32, 128]

    e.log_parameters()
    RDLogger.DisableLog('rdApp.*')

    e.log(f'collecting the pairs of the ged_exact_pairs runs with prefix "{e.PAIRS_PREFIX}"...')
    pair_sets = collect_pairs(e)
    if not pair_sets:
        raise RuntimeError(f'no ged_exact_pairs archives with prefix "{e.PAIRS_PREFIX}" found')

    rng = np.random.default_rng(e.BOOTSTRAP_SEED)
    result_rows: List[dict] = []
    difference_rows: List[dict] = []
    seed_rows: List[dict] = []
    distance_frames: List[pd.DataFrame] = []
    for dataset_name, max_heavy_atoms in e.MOLECULE_SETS:
        set_keys = [key for key in pair_sets if key[0] == dataset_name and key[1] == max_heavy_atoms]
        if not set_keys:
            e.log(f'WARNING: no pairs for {dataset_name} (<= {max_heavy_atoms} heavy atoms)')
            continue

        # --- select the pairs of every pair set ---
        selected: Dict[str, pd.DataFrame] = {}
        for key in set_keys:
            frame = pair_sets[key]
            pair_set = key[2]
            num_unfinished = int((~frame['finished'].astype(bool)).sum())
            if pair_set == 'near':
                keep = frame['finished'].astype(bool) & frame['ged'].notna() & (frame['ged'] <= e.NEAR_THRESHOLD)
            else:
                keep = frame['finished'].astype(bool) & frame['ged'].notna()
            chosen = frame[keep].reset_index(drop=True)
            selected[pair_set] = chosen
            e[f'pairs/{dataset_name}/{pair_set}/num_total'] = len(frame)
            e[f'pairs/{dataset_name}/{pair_set}/num_unfinished'] = num_unfinished
            e[f'pairs/{dataset_name}/{pair_set}/num_selected'] = len(chosen)
            e[f'pairs/{dataset_name}/{pair_set}/ged_counts'] = {
                str(int(k)): int(v) for k, v in chosen['ged'].value_counts().sort_index().items()
            }
            e.log(f'{dataset_name} / {pair_set}: {len(frame)} pairs, {num_unfinished} unfinished, '
                  f'{len(chosen)} selected, GED {chosen["ged"].min()}-{chosen["ged"].max()}')
            # Unfinished random pairs tend to be the largest and most dissimilar ones, so excluding them cuts
            # off the upper end of the GED range. As a sensitivity check, they are also evaluated at the best
            # upper bound that the search found within the time limit.
            if pair_set == 'random' and num_unfinished > 0:
                with_unfinished = frame[frame['ged'].notna()].reset_index(drop=True)
                selected['random_incl_unfinished'] = with_unfinished
                e[f'pairs/{dataset_name}/random_incl_unfinished/num_selected'] = len(with_unfinished)
                e.log(f'{dataset_name} / random_incl_unfinished: {len(with_unfinished)} pairs '
                      f'(unfinished ones at their upper bound)')

        # --- encode all molecules of this molecule set once per size (and codebook seed for HDF) ---
        smiles_all = sorted({s for frame in selected.values() for s in pd.concat([frame['smiles_1'], frame['smiles_2']])})
        smiles_index = {s: i for i, s in enumerate(smiles_all)}
        e.log(f'{dataset_name}: computing the full-dataset statistics for the HDF size/diameter encodings...')
        stats = dataset_statistics(dataset_name)
        e[f'stats/{dataset_name}'] = stats
        e.log(f'{dataset_name}: {stats}; encoding {len(smiles_all)} molecules...')
        embeddings = {}
        for dim in e.EMBEDDING_SIZES:
            embeddings[('morgan', dim)] = encode_morgan(smiles_all, dim, e.MORGAN_RADIUS)
            for seed in e.HDF_SEEDS:
                embeddings[('hdf', dim, seed)] = encode_hdf(smiles_all, dim, stats, e, seed)

        # --- distances, correlations and molecule-level bootstrap intervals per pair set ---
        alpha = (1.0 - e.CONFIDENCE) / 2.0
        for pair_set, frame in sorted(selected.items()):
            if len(frame) < 3:
                e.log(f'WARNING: {dataset_name} / {pair_set}: too few pairs ({len(frame)}) for correlations')
                continue
            num_pairs = len(frame)
            ged = frame['ged'].to_numpy(dtype=float)
            index_1 = frame['smiles_1'].map(smiles_index).to_numpy()
            index_2 = frame['smiles_2'].map(smiles_index).to_numpy()
            # molecule indices local to this pair set for the pigeonhole bootstrap
            local = {s: i for i, s in enumerate(sorted(set(frame['smiles_1']) | set(frame['smiles_2'])))}
            local_1 = frame['smiles_1'].map(local).to_numpy()
            local_2 = frame['smiles_2'].map(local).to_numpy()

            distances = frame[['pair_index', 'smiles_1', 'smiles_2', 'ged']].copy()
            distances.insert(0, 'pair_set', pair_set)
            distances.insert(0, 'dataset', dataset_name)
            vectors: Dict[tuple, np.ndarray] = {}
            for dim in e.EMBEDDING_SIZES:
                matrix = embeddings[('morgan', dim)]
                vectors[('morgan', dim)] = tanimoto_distance(matrix[index_1], matrix[index_2])
                distances[f'morgan_{dim}'] = vectors[('morgan', dim)]
                for seed in e.HDF_SEEDS:
                    matrix = embeddings[('hdf', dim, seed)]
                    vectors[('hdf', dim, seed)] = cosine_distance(matrix[index_1], matrix[index_2])
                    distances[f'hdf_{dim}_s{seed}'] = vectors[('hdf', dim, seed)]

            # point estimates on the full sample (unit weights)
            ones = np.ones((1, num_pairs))
            point = {key: {'pearson': float(weighted_pearson(vector, ged, ones)[0]),
                           'spearman': float(weighted_pearson(weighted_midranks(vector, ones),
                                                              weighted_midranks(ged, ones), ones)[0])}
                     for key, vector in vectors.items()}

            # Bootstrap with two kinds of weights, shared by all representations, sizes and seeds (paired):
            # 'pigeonhole' resamples molecules, so that pairs sharing a molecule move together (conservative
            # where molecules rarely repeat), and 'pair' resamples pairs (ignores shared molecules).
            num_seeds = len(e.HDF_SEEDS)
            boot = {method: {key: {name: np.empty(e.NUM_BOOTSTRAP) for name in ('pearson', 'spearman')}
                             for key in vectors}
                    for method in ('pigeonhole', 'pair')}
            for start in range(0, e.NUM_BOOTSTRAP, e.BOOTSTRAP_CHUNK):
                stop = min(start + e.BOOTSTRAP_CHUNK, e.NUM_BOOTSTRAP)
                weight_sets = {
                    'pigeonhole': pigeonhole_weights(local_1, local_2, len(local), stop - start, rng),
                    'pair': rng.multinomial(num_pairs, np.full(num_pairs, 1.0 / num_pairs),
                                            size=stop - start).astype(float),
                }
                for method, weights in weight_sets.items():
                    ged_ranks = weighted_midranks(ged, weights)
                    for key, vector in vectors.items():
                        boot[method][key]['pearson'][start:stop] = weighted_pearson(vector, ged, weights)
                        boot[method][key]['spearman'][start:stop] = weighted_pearson(
                            weighted_midranks(vector, weights), ged_ranks, weights)
            # Every replicate also resamples the codebook seeds with replacement and averages over them, so
            # that the HDF intervals cover the randomness of the codebooks for the reported seed average.
            seed_resamples = rng.integers(0, num_seeds, size=(e.NUM_BOOTSTRAP, num_seeds))

            for dim in e.EMBEDDING_SIZES:
                estimates, seed_values = {}, {}
                replicates = {}   # (method, representation, name) -> replicates, seeds resampled
                conditional = {}  # (method, representation, name) -> replicates, averaged over all seeds
                for name in ('pearson', 'spearman'):
                    per_seed = np.array([point[('hdf', dim, seed)][name] for seed in e.HDF_SEEDS])
                    seed_values[name] = per_seed
                    estimates[('hdf', name)] = float(per_seed.mean())
                    estimates[('morgan', name)] = point[('morgan', dim)][name]
                    for method in ('pigeonhole', 'pair'):
                        stacked = np.stack([boot[method][('hdf', dim, seed)][name] for seed in e.HDF_SEEDS], axis=1)
                        replicates[(method, 'hdf', name)] = np.take_along_axis(stacked, seed_resamples, axis=1).mean(axis=1)
                        conditional[(method, 'hdf', name)] = stacked.mean(axis=1)
                        replicates[(method, 'morgan', name)] = boot[method][('morgan', dim)][name]
                        conditional[(method, 'morgan', name)] = boot[method][('morgan', dim)][name]
                for seed_position, seed in enumerate(e.HDF_SEEDS):
                    seed_rows.append({'dataset': dataset_name, 'pair_set': pair_set, 'size': dim, 'seed': seed,
                                      'pearson': seed_values['pearson'][seed_position],
                                      'spearman': seed_values['spearman'][seed_position]})

                for representation in ('hdf', 'morgan'):
                    row = {'dataset': dataset_name, 'max_heavy_atoms': max_heavy_atoms, 'pair_set': pair_set,
                           'representation': representation, 'size': dim, 'num_pairs': num_pairs,
                           'num_molecules': len(local), 'num_seeds': num_seeds if representation == 'hdf' else 1}
                    for name in ('pearson', 'spearman'):
                        estimate = estimates[(representation, name)]
                        row[name] = estimate
                        # headline: percentile interval of the molecule-level (pigeonhole) bootstrap
                        row[f'{name}_low'], row[f'{name}_high'] = percentile_interval(
                            replicates[('pigeonhole', representation, name)], alpha)
                        row[f'{name}_low_pair'], row[f'{name}_high_pair'] = percentile_interval(
                            replicates[('pair', representation, name)], alpha)
                        variance = corrected_variance(conditional[('pigeonhole', representation, name)],
                                                      conditional[('pair', representation, name)])
                        if representation == 'hdf':
                            variance += seed_variance_of_mean(seed_values[name])
                            row[f'{name}_seed_sd'] = float(np.std(seed_values[name], ddof=1)) if num_seeds > 1 else 0.0
                            row[f'{name}_seed_min'] = float(seed_values[name].min())
                            row[f'{name}_seed_max'] = float(seed_values[name].max())
                        row[f'{name}_low_corr'], row[f'{name}_high_corr'] = fisher_interval(estimate, variance, alpha)
                    result_rows.append(row)
                    e[f'results/{dataset_name}/{pair_set}/{representation}/{dim}'] = row
                    e.log(f' * {dataset_name} / {pair_set} / {representation:6s} / {dim:4d}: '
                          f'Pearson {row["pearson"]:.3f} [{row["pearson_low"]:.3f}, {row["pearson_high"]:.3f}] '
                          f'(pair [{row["pearson_low_pair"]:.3f}, {row["pearson_high_pair"]:.3f}], '
                          f'corrected [{row["pearson_low_corr"]:.3f}, {row["pearson_high_corr"]:.3f}]), '
                          f'Spearman {row["spearman"]:.3f}'
                          + (f' | seeds {row["pearson_seed_min"]:.3f}-{row["pearson_seed_max"]:.3f}'
                             if representation == 'hdf' else ''))

                difference = {'dataset': dataset_name, 'pair_set': pair_set, 'size': dim, 'num_pairs': num_pairs}
                for name in ('pearson', 'spearman'):
                    estimate = estimates[('hdf', name)] - estimates[('morgan', name)]
                    difference[f'{name}_diff'] = estimate
                    for method, suffix in (('pigeonhole', ''), ('pair', '_pair')):
                        difference[f'{name}_diff_low{suffix}'], difference[f'{name}_diff_high{suffix}'] = \
                            percentile_interval(replicates[(method, 'hdf', name)]
                                                - replicates[(method, 'morgan', name)], alpha)
                    variance = corrected_variance(
                        conditional[('pigeonhole', 'hdf', name)] - conditional[('pigeonhole', 'morgan', name)],
                        conditional[('pair', 'hdf', name)] - conditional[('pair', 'morgan', name)],
                    ) + seed_variance_of_mean(seed_values[name])
                    half_width = float(norm.ppf(1.0 - alpha)) * np.sqrt(variance)
                    difference[f'{name}_diff_low_corr'] = estimate - half_width
                    difference[f'{name}_diff_high_corr'] = estimate + half_width
                difference_rows.append(difference)
                e[f'differences/{dataset_name}/{pair_set}/{dim}'] = difference
            distance_frames.append(distances)

    results = pd.DataFrame(result_rows)
    differences = pd.DataFrame(difference_rows)
    with e.open('results.csv', mode='w') as file:
        results.to_csv(file, index=False)
    with e.open('differences.csv', mode='w') as file:
        differences.to_csv(file, index=False)
    with e.open('pairs_with_distances.csv', mode='w') as file:
        pd.concat(distance_frames, ignore_index=True).to_csv(file, index=False)
    with e.open('seeds.csv', mode='w') as file:
        pd.DataFrame(seed_rows).to_csv(file, index=False)
    e.log('saved results.csv, differences.csv, seeds.csv and pairs_with_distances.csv')


@experiment.analysis
def analysis(e: Experiment):
    """Preliminary version of the SI figure: panels a (random pairs), b (near pairs), c (scatter, QM9)."""
    results = pd.read_csv(os.path.join(e.path, 'results.csv'))
    distances = pd.read_csv(os.path.join(e.path, 'pairs_with_distances.csv'))
    labels = {'qm9_smiles': 'QM9', 'zinc250k': 'ZINC250k'}
    styles = {'qm9_smiles': '-', 'zinc250k': '--'}
    fig, axes = plt.subplots(1, 4, figsize=(17, 4))
    for ax, pair_set, title in ((axes[0], 'random', 'a  Random pairs'), (axes[1], 'near', 'b  Near pairs')):
        for (dataset, representation), group in results[results['pair_set'] == pair_set].groupby(['dataset', 'representation']):
            group = group.sort_values('size')
            color = COLOR_HDF if representation == 'hdf' else COLOR_MORGAN
            name = 'HDF' if representation == 'hdf' else 'Morgan'
            x = np.arange(len(group))
            ax.errorbar(x, group['pearson'], yerr=[group['pearson'] - group['pearson_low'],
                                                   group['pearson_high'] - group['pearson']],
                        color=color, linestyle=styles.get(dataset, '-'), marker='o', capsize=3,
                        label=f'{name}, {labels.get(dataset, dataset)}')
            ax.set_xticks(x, group['size'])
        ax.set_xlabel('Embedding size')
        ax.set_ylabel('Pearson correlation with GED')
        ax.set_title(title, loc='left')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    qm9 = distances[(distances['dataset'] == 'qm9_smiles') & (distances['pair_set'] == 'random')]
    for ax, dim in ((axes[2], min(results['size'])), (axes[3], max(results['size']))):
        if len(qm9) == 0:
            continue
        jitter = np.random.default_rng(0).uniform(-0.15, 0.15, size=len(qm9))
        ax.scatter(qm9['ged'] - 0.2 + jitter, qm9[f'morgan_{dim}'], s=2, alpha=0.2, color=COLOR_MORGAN, label='Morgan')
        ax.scatter(qm9['ged'] + 0.2 + jitter, qm9[f'hdf_{dim}_s{e.HDF_SEEDS[0]}'], s=2, alpha=0.2, color=COLOR_HDF,
                   label=f'HDF (seed {e.HDF_SEEDS[0]})')
        ax.set_xlabel('GED')
        ax.set_ylabel('Distance')
        ax.set_title(f'c  QM9 random pairs, size {dim}', loc='left')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, markerscale=4)
    fig.tight_layout()
    fig.savefig(os.path.join(e.path, 'ged_exact.pdf'))
    fig.savefig(os.path.join(e.path, 'ged_exact.png'), dpi=150)
    plt.close(fig)
    e.log('saved ged_exact.pdf/png')


experiment.run_if_main()
