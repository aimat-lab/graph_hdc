"""
Diagnostics for the HDF results of ex_17 (exact GED, reviewer comment R2.7): why do HDF distances scatter
strongly among pairs with the same GED, and why do the correlations depend on the codebook seed?

Observation: the continuous (FHRR) encoders of graph_hdc draw a complex Gaussian spectrum (random magnitudes,
not unit phasors) and raise it to the power value / bandwidth. For large values (e.g. diameter 8 with
bandwidth 2, power 4) a few spectral components dominate, so these encodings are effectively low-dimensional
and seed-dependent at any D. The graph size and diameter encodings form half of the final HDF vector.

This script re-encodes a random subsample of the QM9 random pairs of ex_17 with variants of HDF, for every
codebook seed, and reports the correlation with the exact GED (unweighted and GED-balanced), the rise of the
distance per GED step and the scatter of the distance among pairs with the same GED:

- ``full``: HDF as in ex_17 (bidirectional, total hydrogens, size and diameter encodings).
- ``no_attributes``: without the graph size and diameter encodings (structural part only).
- ``unit_fhrr``: all continuous encoders with unit-modulus spectra (proper FHRR phasors).
- ``unit_fhrr_no_attributes``: both changes.

Writes ``_ex17/analysis/hdf_variants.csv``.

    python analyze_ex_17_hdf.py [--pairs 3000] [--dim 2048]
"""
import argparse
import os

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

import analyze_ex_17
import ged_exact_correlation as gec
from graph_hdc.models import HyperNet
from graph_hdc.special.molecules import (
    graph_dict_from_mol,
    make_molecule_node_encoder_map_cont,
    make_molecule_graph_encoder_map_cont,
)
from graph_hdc.utils import ContinuousEncoder

RDLogger.DisableLog('rdApp.*')
PATH = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(PATH, '_ex17', 'analysis')
SEEDS = [1, 2, 3, 4, 5]
VARIANTS = ['full', 'no_attributes', 'unit_fhrr', 'unit_fhrr_no_attributes']


def unit_modulus(encoder_map: dict) -> dict:
    """Replace the spectra of all continuous encoders by unit-modulus phasors (same phases)."""
    for encoder in encoder_map.values():
        if isinstance(encoder, ContinuousEncoder):
            encoder.matrix = encoder.matrix / encoder.matrix.abs()
    return encoder_map


def encode(smiles: list, dim: int, seed: int, variant: str, stats: dict) -> np.ndarray:
    # the original (gaussian) codebooks of ex_17; unit-modulus codebooks are the default since 2026-10-09
    node_map = make_molecule_node_encoder_map_cont(dim=dim, seed=seed, unit_modulus=False)
    graph_map = make_molecule_graph_encoder_map_cont(dim=dim, seed=seed, max_graph_size=stats['max_size'],
                                                     max_graph_diameter=stats['max_diameter'], unit_modulus=False)
    if 'no_attributes' in variant:
        graph_map = {}
    if variant.startswith('unit_fhrr'):
        node_map, graph_map = unit_modulus(node_map), unit_modulus(graph_map)
    hyper_net = HyperNet(hidden_dim=dim, depth=2, device='cpu', node_encoder_map=node_map,
                         graph_encoder_map=graph_map, seed=seed, normalize_all=True, bidirectional=True)
    graphs = []
    for s in smiles:
        graph = graph_dict_from_mol(Chem.MolFromSmiles(s), hydrogens='total')
        graph.pop('graph_labels', None)
        graphs.append(graph)
    results = hyper_net.forward_graphs(graphs, batch_size=600)
    return np.stack([np.asarray(r['graph_embedding'], dtype=float) for r in results])


def describe(distance: np.ndarray, ged: np.ndarray, groups: np.ndarray) -> dict:
    ones = np.ones((1, len(ged)))
    weights = analyze_ex_17.balanced(ones, groups)
    frame = pd.DataFrame({'d': distance, 'g': ged})
    within = np.sqrt((frame.groupby('g')['d'].var(ddof=0) * frame.groupby('g').size()).sum() / len(frame))
    slope = np.polyfit(ged, distance, 1)[0]
    return {'pearson': float(gec.weighted_pearson(distance, ged, ones)[0]),
            'pearson_balanced': float(gec.weighted_pearson(distance, ged, weights)[0]),
            'slope_per_ged': float(slope), 'sd_within_ged': float(within), 'slope_over_sd': float(slope / within)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pairs', type=int, default=3000)
    parser.add_argument('--dim', type=int, default=2048)
    args = parser.parse_args()

    archive = analyze_ex_17.latest_archive()
    pairs = pd.read_csv(os.path.join(archive, 'pairs_with_distances.csv'))
    pairs = pairs[(pairs['dataset'] == 'qm9_smiles') & (pairs['pair_set'] == 'random')]
    pairs = pairs.sample(n=min(args.pairs, len(pairs)), random_state=0).reset_index(drop=True)
    ged = pairs['ged'].to_numpy(float)
    groups = analyze_ex_17.ged_groups(ged, analyze_ex_17.MIN_GROUP_PAIRS)
    smiles = sorted(set(pairs['smiles_1']) | set(pairs['smiles_2']))
    index = {s: i for i, s in enumerate(smiles)}
    i1, i2 = pairs['smiles_1'].map(index).to_numpy(), pairs['smiles_2'].map(index).to_numpy()
    stats = {'max_size': 9, 'max_diameter': 8}  # full-dataset QM9 statistics (as in ex_17)
    print(f'QM9 random pairs: {len(pairs)} pairs, {len(smiles)} molecules, D = {args.dim}')

    rows = [{'variant': 'morgan (reference)', 'seed': '-', **describe(pairs[f'morgan_{args.dim}'].to_numpy(float), ged, groups)}]
    for variant in VARIANTS:
        for seed in SEEDS:
            embedding = encode(smiles, args.dim, seed, variant, stats)
            distance = gec.cosine_distance(embedding[i1], embedding[i2])
            rows.append({'variant': variant, 'seed': seed, **describe(distance, ged, groups)})
            print(f'  {variant:24s} seed {seed}: ' + ', '.join(f'{k} {v:.3f}' for k, v in rows[-1].items()
                                                             if k not in ('variant', 'seed')), flush=True)
    table = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    table.to_csv(os.path.join(OUT, f'hdf_variants_d{args.dim}.csv'), index=False)
    summary = (table[table['seed'] != '-'].groupby('variant')[['pearson', 'pearson_balanced', 'slope_over_sd']]
               .agg(['mean', 'min', 'max']).round(3))
    print('\nper variant over the 5 seeds (mean, min, max):')
    print(summary.to_string())
    print('\nMorgan reference:', table.iloc[0][['pearson', 'pearson_balanced', 'slope_over_sd']].round(3).to_dict())


if __name__ == '__main__':
    main()
