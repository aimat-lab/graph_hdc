"""
ex_22, GED part: correlation of the four HDF arms of ex_22 with the exact graph edit distance.

Arms (as in _slurm_ex_22.py): the original encoder ("gaussian" spectra, with the graph size and diameter
encodings), unit-modulus spectra (``unit_modulus`` of the encoders), and both without the size and diameter
encodings.

Pairs: the fixed GED-balanced subsets of ex_17 (reviewer comment R2.7; at most 100 pairs per exact GED value, drawn
once with seed 0 by make_figure_ged_balanced_subset.py in the main checkout): QM9 and ZINC250k (<= 12 heavy atoms),
random pairs and near pairs (GED <= 4). They are read from the main checkout, where the ex_17 data lives.

The HDF vector is normalize(normalize(S) + normalize(G)) (HyperNet.forward), with S the message passing readout
and G the sum of the size and diameter encodings. S is computed with HyperNet without graph attributes, G directly
with the graph encoders, so every (D, seed) needs one HyperNet pass per spectrum. As a check, the original arm is
compared with the HDF distances stored in the subsets (ex_17 stage 2).

Writes ``_ex22/ged_arms.csv`` (Pearson r per subset, D, seed and arm) and ``_ex22/ged_arms.md`` (summary).

    python analyze_ex_22_ged.py [--subsets qm9_smiles:random ...] [--tag _qm9]
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger

from graph_hdc.models import HyperNet
from graph_hdc.special.molecules import (
    graph_dict_from_mol,
    make_molecule_graph_encoder_map_cont,
    make_molecule_node_encoder_map_cont,
)

RDLogger.DisableLog('rdApp.*')
PATH = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(PATH, '_ex22')
SUBSETS = ('/media/ssd2/Programming/graph_hdc/graph_hdc/experiments/fingerprints/_ex17/analysis/'
           'balanced_subset_{dataset}_{pair_set}.csv')
# full-dataset statistics for the size and diameter bandwidths, as in ex_17
STATS = {'qm9_smiles': {'max_size': 9, 'max_diameter': 8}, 'zinc250k': {'max_size': 38, 'max_diameter': 23}}
SIZES = [32, 128, 512, 2048]
SEEDS = [1, 2, 3, 4, 5]
ARMS = {  # arm: (spectrum, graph attributes)
    'gaussian': ('gaussian', True),
    'unit': ('unit', True),
    'gaussian_noglobal': ('gaussian', False),
    'unit_noglobal': ('unit', False),
}


def normalize(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return 1.0 - np.sum(normalize(a) * normalize(b), axis=1)


def structure_part(graphs: list, dim: int, seed: int, spectrum: str) -> np.ndarray:
    node_map = make_molecule_node_encoder_map_cont(dim=dim, seed=seed, unit_modulus=(spectrum == 'unit'))
    hyper_net = HyperNet(hidden_dim=dim, depth=2, device='cpu', node_encoder_map=node_map, graph_encoder_map={},
                         seed=seed, normalize_all=True, bidirectional=True)
    results = hyper_net.forward_graphs([dict(g) for g in graphs], batch_size=600)
    return np.stack([np.asarray(r['graph_embedding'], dtype=float) for r in results])


def global_part(sizes: np.ndarray, diameters: np.ndarray, dim: int, seed: int, stats: dict, spectrum: str) -> np.ndarray:
    graph_map = make_molecule_graph_encoder_map_cont(dim=dim, seed=seed, max_graph_size=stats['max_size'],
                                                     max_graph_diameter=stats['max_diameter'],
                                                     unit_modulus=(spectrum == 'unit'))
    hv = (graph_map['graph_size'].encode_batch(torch.tensor(sizes, dtype=torch.float64))
          + graph_map['graph_diameter'].encode_batch(torch.tensor(diameters, dtype=torch.float64)))
    return hv.numpy()


def summary(table: pd.DataFrame, morgan: pd.DataFrame) -> str:
    lines = []
    for (dataset, pair_set), part in table.groupby(['dataset', 'pair_set'], sort=False):
        n = int(part['num_pairs'].iloc[0])
        lines.append(f'\n**{dataset} / {pair_set}** ({n} pairs), Pearson r with the exact GED, mean ± SD over '
                     f'{part["seed"].nunique()} codebook seeds\n')
        header = ['Arm'] + [f'D={d}' for d in SIZES]
        lines += ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
        for arm in ARMS:
            cells = []
            for d in SIZES:
                values = part[(part['arm'] == arm) & (part['size'] == d)]['pearson']
                cells.append(f'{values.mean():.2f} ± {values.std():.2f}' if len(values) else '-')
            lines.append(f'| {arm} | ' + ' | '.join(cells) + ' |')
        m = morgan[(morgan['dataset'] == dataset) & (morgan['pair_set'] == pair_set)].set_index('size')['pearson']
        lines.append('| Morgan | ' + ' | '.join(f'{m.get(d, float("nan")):.2f}' for d in SIZES) + ' |')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subsets', nargs='+', default=['qm9_smiles:random', 'zinc250k:random',
                                                         'qm9_smiles:near', 'zinc250k:near'])
    parser.add_argument('--sizes', type=int, nargs='+', default=SIZES)
    parser.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    parser.add_argument('--tag', default='', help='suffix of the output files (for parallel runs)')
    args = parser.parse_args()

    rows, morgan_rows = [], []
    for key in args.subsets:
        dataset, pair_set = key.split(':')
        subset = pd.read_csv(SUBSETS.format(dataset=dataset, pair_set=pair_set))
        ged = subset['ged'].to_numpy(float)
        smiles = sorted(set(subset['smiles_1']) | set(subset['smiles_2']))
        index = {s: i for i, s in enumerate(smiles)}
        i1, i2 = subset['smiles_1'].map(index).to_numpy(), subset['smiles_2'].map(index).to_numpy()
        graphs = []
        for s in smiles:
            graph = graph_dict_from_mol(Chem.MolFromSmiles(s), hydrogens='total')
            graph.pop('graph_labels', None)
            graphs.append(graph)
        sizes = np.array([g['graph_size'] for g in graphs], dtype=float)
        diameters = np.array([g['graph_diameter'] for g in graphs], dtype=float)
        base = {'dataset': dataset, 'pair_set': pair_set, 'num_pairs': len(subset)}
        print(f'{dataset} / {pair_set}: {len(subset)} pairs, {len(smiles)} molecules', flush=True)
        for dim in args.sizes:
            morgan_rows.append({**base, 'size': dim,
                                'pearson': float(np.corrcoef(subset[f'morgan_{dim}'], ged)[0, 1])})
            for seed in args.seeds:
                parts = {}
                for spectrum in ('gaussian', 'unit'):
                    structure = normalize(structure_part(graphs, dim, seed, spectrum))
                    glob = normalize(global_part(sizes, diameters, dim, seed, STATS[dataset], spectrum))
                    parts[(spectrum, True)] = normalize(structure + glob)
                    parts[(spectrum, False)] = structure
                check = float(np.abs(cosine_distance(parts[('gaussian', True)][i1], parts[('gaussian', True)][i2])
                                     - subset[f'hdf_{dim}_s{seed}'].to_numpy(float)).max())
                for arm, (spectrum, attributes) in ARMS.items():
                    vectors = parts[(spectrum, attributes)]
                    distance = cosine_distance(vectors[i1], vectors[i2])
                    rows.append({**base, 'size': dim, 'seed': seed, 'arm': arm,
                                 'pearson': float(np.corrcoef(distance, ged)[0, 1]),
                                 'max_abs_diff_to_ex17': check})
                print(f'  D={dim} seed {seed}: ' + ', '.join(f'{r["arm"]} {r["pearson"]:.3f}' for r in rows[-4:])
                      + f' | original vs ex_17: {check:.1e}', flush=True)
    os.makedirs(OUT, exist_ok=True)
    table, morgan = pd.DataFrame(rows), pd.DataFrame(morgan_rows)
    table.to_csv(os.path.join(OUT, f'ged_arms{args.tag}.csv'), index=False)
    morgan.to_csv(os.path.join(OUT, f'ged_morgan{args.tag}.csv'), index=False)
    text = summary(table, morgan)
    with open(os.path.join(OUT, f'ged_arms{args.tag}.md'), 'w') as f:
        f.write(text + '\n')
    print(text)
    print(f'\nlargest deviation of the original arm from the ex_17 distances: {table["max_abs_diff_to_ex17"].max():.1e}')


if __name__ == '__main__':
    main()
