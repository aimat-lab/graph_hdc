"""
Analysis of ex_08_c (see ``_slurm_ex_08c.py``): the edit-based GED analysis (Figure 3, reviewer comment R2.7) with
the HDF default of 2026-10-09 (unit-modulus codebooks, bidirectional message passing, total hydrogen counts).

1. Pairs: the HDF runs of ex_08_c and the Morgan runs of ex_08_b must contain identical (seed, neighbor, edit step)
   pairs, since Morgan is not re-run.
2. HDF without the graph size and diameter encodings (the dashed variant of the GED figures): computed here from
   the pairs of ex_08_c with the same encoder construction as molecule_similarity__hdc.py (SEED, depth 2,
   bidirectional, total hydrogen counts, unit-modulus codebooks) but without graph encoders. As a check, the
   same construction WITH the size and diameter encodings must reproduce the distances of ex_08_c.
3. Per-seed |Pearson r| between edit steps and distance (the data of Figure 3b): HDF (ex_08_c), HDF without
   size/diameter, Morgan (ex_08_b), next to the HDF runs of ex_08_b (original codebooks).
4. Distances per edit step (SI figure), with duplicates of a molecule within one seed and edit step dropped as in
   analyze_ex_08b.py.

Writes ``_ex08c/analysis/`` (per_seed.csv, summary.csv, pairs.csv, distances_per_step.csv, rebuild_check.csv).

    python analyze_ex_08c.py            # ex_08_c (HDF) and ex_08_b (Morgan)
    python analyze_ex_08c.py --smoke    # the smoke runs ex_08_c_smoke / ex_08_b_smoke, to _ex08c/analysis_smoke
"""
import os
import sys

import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from scipy.stats import pearsonr

from analyze_ex_08b import SIZES, load_runs
from graph_hdc.models import HyperNet
from graph_hdc.special.molecules import (
    graph_dict_from_mol,
    make_molecule_graph_encoder_map_cont,
    make_molecule_node_encoder_map_cont,
)

RDLogger.DisableLog('rdApp.*')
jsonpickle_numpy.register_handlers()
PATH = os.path.dirname(os.path.abspath(__file__))
SEED = 1          # FIXED['SEED'] of _slurm_ex_08b.py / _slurm_ex_08c.py
NUM_LAYERS = 2


def graph_stats(archive: str) -> dict:
    """Size/diameter maxima of the run's dataset, read from its saved encoders (as used for the bandwidths)."""
    with open(os.path.join(archive, 'hyper_net.pth')) as file:
        data = jsonpickle.loads(file.read())
    graph_map = data['graph_encoder_map']
    if not all(getattr(encoder, 'unit_modulus', False) for encoder in graph_map.values()):
        raise RuntimeError(f'{archive}: the saved encoders are not unit-modulus')
    return {'max_size': graph_map['graph_size'].size, 'max_diameter': graph_map['graph_diameter'].size}


def encode(smiles: list, dim: int, stats: dict, graph_attributes: bool) -> np.ndarray:
    """HDF as in molecule_similarity__hdc.py (current default), optionally without the size/diameter encodings."""
    node_map = make_molecule_node_encoder_map_cont(dim=dim, seed=SEED)
    graph_map = (make_molecule_graph_encoder_map_cont(dim=dim, seed=SEED, max_graph_size=stats['max_size'],
                                                      max_graph_diameter=stats['max_diameter'])
                 if graph_attributes else {})
    hyper_net = HyperNet(hidden_dim=dim, depth=NUM_LAYERS, device='cpu', node_encoder_map=node_map,
                         graph_encoder_map=graph_map, seed=SEED, normalize_all=True, bidirectional=True)
    graphs = []
    for s in smiles:
        graph = graph_dict_from_mol(Chem.MolFromSmiles(s), hydrogens='total')
        graph.pop('graph_labels', None)
        graphs.append(graph)
    return np.stack([np.asarray(r['graph_embedding'], dtype=float)
                     for r in hyper_net.forward_graphs(graphs, batch_size=600)])


def cosine_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return 1.0 - np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))


def per_query_correlations(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    rows = []
    for query_idx, group in frame.groupby('query_idx'):
        r = pearsonr(group['edit_steps'], group[column])[0] if group[column].nunique() > 1 else np.nan
        rows.append({'query_idx': query_idx, 'correlation': r, 'abs_correlation': abs(r)})
    return pd.DataFrame(rows)


def main(smoke: bool = False):
    out = os.path.join(PATH, '_ex08c', 'analysis_smoke' if smoke else 'analysis')
    os.makedirs(out, exist_ok=True)
    unit, old = load_runs('ex_08_c_smoke' if smoke else 'ex_08_c'), load_runs('ex_08_b_smoke' if smoke else 'ex_08_b')
    sizes = [32] if smoke else SIZES
    print(f'runs: ex_08_c {sorted(unit)}, ex_08_b {sorted(old)}')
    key = ['query_idx', 'query_smiles', 'neighbor_smiles', 'edit_steps']
    pair_frames, per_seed_rows, check_rows = [], [], []
    for size in sizes:
        if ('hdc', size) not in unit:
            print(f'D={size}: no finished ex_08_c run yet')
            continue
        hdf = pd.read_csv(os.path.join(unit[('hdc', size)], 'ged_pairs.csv'))
        morgan = pd.read_csv(os.path.join(old[('fp', size)], 'ged_pairs.csv'))
        hdf_old = pd.read_csv(os.path.join(old[('hdc', size)], 'ged_pairs.csv'))

        # 1. identical pairs (as multisets of rows)
        for name, other in (('Morgan ex_08_b', morgan), ('HDF ex_08_b', hdf_old)):
            a = hdf[key].sort_values(key).reset_index(drop=True)
            b = other[key].sort_values(key).reset_index(drop=True)
            if not a.equals(b):
                raise RuntimeError(f'D={size}: the pairs of ex_08_c differ from {name}')
        frame = hdf[key + ['distance']].rename(columns={'distance': 'hdf'})
        # align Morgan and the original HDF by the pair key (a pair can occur twice, so merge on key + occurrence)
        for name, other in (('morgan', morgan), ('hdf_gaussian', hdf_old)):
            left = frame.assign(_n=frame.groupby(key).cumcount())
            right = other[key + ['distance']].assign(_n=other.groupby(key).cumcount())
            frame = left.merge(right.rename(columns={'distance': name}), on=key + ['_n'], how='left',
                               validate='one_to_one').drop(columns=['_n'])
            if frame[name].isna().any():
                raise RuntimeError(f'D={size}: {name} distances missing for some pairs')

        # 2. HDF without size/diameter (and the check with them)
        smiles = sorted(set(frame['query_smiles']) | set(frame['neighbor_smiles']))
        index = {s: i for i, s in enumerate(smiles)}
        i1, i2 = frame['query_smiles'].map(index).to_numpy(), frame['neighbor_smiles'].map(index).to_numpy()
        stats = graph_stats(unit[('hdc', size)])
        with_global = encode(smiles, size, stats, graph_attributes=True)
        rebuilt = cosine_distance(with_global[i1], with_global[i2])
        deviation = float(np.abs(rebuilt - frame['hdf'].to_numpy()).max())
        check_rows.append({'size': size, 'max_abs_diff_to_ex_08_c': deviation})
        print(f'D={size}: {len(frame)} pairs, {len(smiles)} molecules; rebuilt HDF vs ex_08_c max |diff| {deviation:.1e}')
        if deviation > 1e-5:
            raise RuntimeError(f'D={size}: the rebuilt HDF does not reproduce ex_08_c (max |diff| {deviation})')
        without = encode(smiles, size, stats, graph_attributes=False)
        frame['hdf_no_global'] = cosine_distance(without[i1], without[i2])
        frame['size'] = size
        pair_frames.append(frame)

        # 3. per-seed correlations
        for representation in ('hdf', 'hdf_no_global', 'morgan', 'hdf_gaussian'):
            corr = per_query_correlations(frame, representation)
            corr['representation'], corr['size'] = representation, size
            per_seed_rows.append(corr)

    if not pair_frames:
        return
    pairs = pd.concat(pair_frames, ignore_index=True)
    pairs.to_csv(os.path.join(out, 'pairs.csv'), index=False)
    per_seed = pd.concat(per_seed_rows, ignore_index=True)
    per_seed.to_csv(os.path.join(out, 'per_seed.csv'), index=False)
    pd.DataFrame(check_rows).to_csv(os.path.join(out, 'rebuild_check.csv'), index=False)
    summary = (per_seed.groupby(['representation', 'size'])['abs_correlation']
               .describe(percentiles=[0.25, 0.5, 0.75]).reset_index())
    summary.to_csv(os.path.join(out, 'summary.csv'), index=False)
    print('\nper-seed |Pearson r| between edit steps and distance (median [IQR]):')
    for (representation, size), row in summary.set_index(['representation', 'size']).iterrows():
        print(f'  {representation:14s} D={size:5d}: {row["50%"]:.3f} [{row["25%"]:.3f}, {row["75%"]:.3f}] (n={int(row["count"])})')

    # 4. distances per edit step, duplicates of a molecule within one seed and edit step dropped
    canonical = {s: Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in pairs['neighbor_smiles'].unique()}
    pairs['neighbor_canonical'] = pairs['neighbor_smiles'].map(canonical)
    deduplicated = pairs.drop_duplicates(subset=['size', 'query_idx', 'edit_steps', 'neighbor_canonical'])
    print(f'\ndropped {len(pairs) - len(deduplicated)} duplicate pairs within an edit step (of {len(pairs)})')
    long = deduplicated.melt(id_vars=['size', 'query_idx', 'edit_steps'],
                             value_vars=['hdf', 'hdf_no_global', 'morgan', 'hdf_gaussian'],
                             var_name='representation', value_name='distance')
    stats = (long.groupby(['representation', 'size', 'edit_steps'])['distance']
             .describe(percentiles=[0.25, 0.5, 0.75]).reset_index())
    stats.to_csv(os.path.join(out, 'distances_per_step.csv'), index=False)
    deduplicated.to_csv(os.path.join(out, 'pairs_deduplicated.csv'), index=False)


if __name__ == '__main__':
    main(smoke='--smoke' in sys.argv)
