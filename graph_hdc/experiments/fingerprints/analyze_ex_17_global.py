"""
Why does removing the global size/diameter encodings of HDF raise the correlation with the exact GED for the
random QM9 pairs, but not for the random ZINC pairs or the near pairs (ex_17, reviewer comment R2.7)?

The final HDF vector is normalize(normalize(S) + normalize(G)) (HyperNet.forward): S is the message passing
readout (structure), G the sum of the graph size and graph diameter encodings (ContinuousEncoder, bandwidth
max(3, max_size / 7) and max(2, max_diameter / 5) from the full-dataset statistics). This script rebuilds both
parts for the fixed GED-balanced subsets of make_figure_ged_balanced_subset.py, checks that their combination
reproduces the ex_17 HDF distances, and reports for every subset, D and codebook seed the Pearson r with the exact
GED of the full distance, of S alone and of G alone, and the spread of each part's distances; plus the same with
unit-modulus FHRR spectra (the combination that gave the large gain in analyze_ex_17_hdf.py).

Writes ``_ex17/analysis/global_parts{tag}.csv`` and ``global_parts_subsets{tag}.csv``.

    python analyze_ex_17_global.py [--sizes 32 2048] [--seeds 1 2 3 4 5] [--subsets qm9_smiles:random ...] [--tag _qm9]
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger

import analyze_ex_17_hdf as hdf
import ged_exact_correlation as gec
from graph_hdc.special.molecules import graph_dict_from_mol, make_molecule_graph_encoder_map_cont

RDLogger.DisableLog('rdApp.*')
PATH = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(PATH, '_ex17', 'analysis')
SUBSETS = os.path.join(OUT, 'balanced_subset_{dataset}_{pair_set}.csv')
STATS = {'qm9_smiles': {'max_size': 9, 'max_diameter': 8}, 'zinc250k': {'max_size': 38, 'max_diameter': 23}}


def normalize(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def global_part(sizes: np.ndarray, diameters: np.ndarray, dim: int, seed: int, stats: dict, unit: bool) -> np.ndarray:
    """G = size encoding + diameter encoding, exactly as HyperNet.encode_properties builds data.graph_hv."""
    graph_map = make_molecule_graph_encoder_map_cont(dim=dim, seed=seed, max_graph_size=stats['max_size'],
                                                     max_graph_diameter=stats['max_diameter'], unit_modulus=False)
    if unit:
        graph_map = hdf.unit_modulus(graph_map)
    hv = (graph_map['graph_size'].encode_batch(torch.tensor(sizes, dtype=torch.float64))
          + graph_map['graph_diameter'].encode_batch(torch.tensor(diameters, dtype=torch.float64)))
    return hv.numpy()


def pearson(distance: np.ndarray, ged: np.ndarray) -> float:
    return float(np.corrcoef(distance, ged)[0, 1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sizes', type=int, nargs='+', default=[32, 2048])
    parser.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    parser.add_argument('--subsets', nargs='+', default=['qm9_smiles:random', 'zinc250k:random',
                                                         'qm9_smiles:near', 'zinc250k:near'])
    parser.add_argument('--tag', default='', help='suffix of the output files (for parallel runs)')
    args = parser.parse_args()

    rows, subset_rows = [], []
    for key in args.subsets:
        dataset, pair_set = key.split(':')
        subset = pd.read_csv(SUBSETS.format(dataset=dataset, pair_set=pair_set))
        ged = subset['ged'].to_numpy(float)
        smiles = sorted(set(subset['smiles_1']) | set(subset['smiles_2']))
        index = {s: i for i, s in enumerate(smiles)}
        i1, i2 = subset['smiles_1'].map(index).to_numpy(), subset['smiles_2'].map(index).to_numpy()
        graphs = [graph_dict_from_mol(Chem.MolFromSmiles(s), hydrogens='total') for s in smiles]
        sizes = np.array([g['graph_size'] for g in graphs], dtype=float)
        diameters = np.array([g['graph_diameter'] for g in graphs], dtype=float)
        d_size, d_diameter = np.abs(sizes[i1] - sizes[i2]), np.abs(diameters[i1] - diameters[i2])
        subset_rows.append({
            'dataset': dataset, 'pair_set': pair_set, 'num_pairs': len(subset), 'num_molecules': len(smiles),
            'size_range': f'{int(sizes.min())}-{int(sizes.max())}',
            'diameter_range': f'{int(diameters.min())}-{int(diameters.max())}',
            'mean_abs_size_diff': d_size.mean(), 'mean_abs_diameter_diff': d_diameter.mean(),
            'r_abs_size_diff_ged': pearson(d_size, ged), 'r_abs_diameter_diff_ged': pearson(d_diameter, ged),
        })
        print(f'{dataset} / {pair_set}: {len(subset)} pairs, {len(smiles)} molecules', flush=True)
        for dim in args.sizes:
            for seed in args.seeds:
                row = {'dataset': dataset, 'pair_set': pair_set, 'size': dim, 'seed': seed}
                for unit in (False, True):
                    variant = 'unit_fhrr_no_attributes' if unit else 'no_attributes'
                    structure = normalize(hdf.encode(smiles, dim, seed, variant, STATS[dataset]))
                    glob = normalize(global_part(sizes, diameters, dim, seed, STATS[dataset], unit))
                    full = normalize(structure + glob)
                    parts = {'full': full, 'structure': structure, 'global': glob}
                    prefix = 'unit_' if unit else ''
                    for name, vectors in parts.items():
                        distance = gec.cosine_distance(vectors[i1], vectors[i2])
                        row[f'{prefix}r_{name}'] = pearson(distance, ged)
                        row[f'{prefix}sd_{name}'] = float(distance.std())
                        row[f'{prefix}mean_{name}'] = float(distance.mean())
                    if not unit:
                        stored = subset[f'hdf_{dim}_s{seed}'].to_numpy(float)
                        rebuilt = gec.cosine_distance(full[i1], full[i2])
                        row['max_abs_diff_to_ex17'] = float(np.abs(rebuilt - stored).max())
                rows.append(row)
                print(f'  D={dim} seed {seed}: r full {row["r_full"]:.3f}, structure {row["r_structure"]:.3f}, '
                      f'global {row["r_global"]:.3f} | unit: full {row["unit_r_full"]:.3f}, '
                      f'structure {row["unit_r_structure"]:.3f} | rebuild err {row["max_abs_diff_to_ex17"]:.1e}',
                      flush=True)
    os.makedirs(OUT, exist_ok=True)
    table, subsets = pd.DataFrame(rows), pd.DataFrame(subset_rows)
    table.to_csv(os.path.join(OUT, f'global_parts{args.tag}.csv'), index=False)
    subsets.to_csv(os.path.join(OUT, f'global_parts_subsets{args.tag}.csv'), index=False)
    print(subsets.round(3).to_string(index=False))
    columns = ['r_full', 'r_structure', 'r_global', 'sd_structure', 'sd_global', 'unit_r_full', 'unit_r_structure',
               'unit_r_global', 'max_abs_diff_to_ex17']
    print(table.groupby(['pair_set', 'dataset', 'size'])[columns].mean().round(3).to_string())


if __name__ == '__main__':
    main()
