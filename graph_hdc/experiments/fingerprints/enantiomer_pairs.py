"""
Shared hooks of the enantiomer pair benchmarks of experiment ex_21 (predict_enantiomers__hdc.py with HDF and
predict_enantiomers__fp.py with Morgan fingerprints): loading a table of enantiomer pairs, its split (given, or random
by pair) and the evaluation by ranking accuracy within pairs. The experiment modules define the parameters (PAIRS_CSV,
CHIRO_CSV, TARGET_COLUMN, SPLIT_MODE, NUM_TEST, NUM_DATA, TIE_TOLERANCE, ACCURACY_THRESHOLD, SEED) and call
``register_pair_hooks(experiment)``.
"""
import os
import random
import shutil
from collections import defaultdict

import numpy as np
import pandas as pd
from rdkit import Chem


def read_table(e) -> pd.DataFrame:
    """The benchmark table with its split (given or random by pair), optionally subsampled by pairs; the row number
    is the index of a molecule."""
    table = pd.read_csv(e.PAIRS_CSV or e.CHIRO_CSV)
    if e.SPLIT_MODE == 'random':
        pairs = sorted(table['smiles_nostereo'].unique())
        num_test = e.NUM_TEST if isinstance(e.NUM_TEST, int) else int(round(e.NUM_TEST * len(pairs)))
        test = set(random.Random(e.SEED).sample(pairs, k=num_test))
        table['split'] = ['test' if pair in test else 'train' for pair in table['smiles_nostereo']]
    elif e.SPLIT_MODE != 'given':
        raise ValueError(f'unknown SPLIT_MODE "{e.SPLIT_MODE}", expected "given" or "random"')
    if e.NUM_DATA is not None and e.NUM_DATA != 1.0:
        rng = random.Random(e.SEED)
        keep = []
        for split in ('train', 'val', 'test'):
            pairs = sorted(table.loc[table['split'] == split, 'smiles_nostereo'].unique())
            num = e.NUM_DATA if isinstance(e.NUM_DATA, int) else int(e.NUM_DATA * len(pairs))
            keep += rng.sample(pairs, k=min(num, len(pairs)))
        table = table[table['smiles_nostereo'].isin(set(keep))]
    return table.reset_index(drop=True)


def load_dataset(e):
    """Graph dicts with the SMILES, the target and the pair; node and edge indices for the base experiment, which
    reports their number (the HDF module replaces them by the attributes of graph_dict_from_mol)."""
    # pycomex archives only the experiment modules; a copy of this helper keeps the archived code runnable
    shutil.copy(os.path.abspath(__file__), e.path)
    table = read_table(e)
    index_data_map = {}
    for index, (smiles, pair, target) in enumerate(zip(table['smiles'], table['smiles_nostereo'],
                                                       table[e.TARGET_COLUMN])):
        mol = Chem.MolFromSmiles(smiles)
        index_data_map[index] = {
            'graph_repr': smiles,
            'graph_labels': np.array([target], dtype=float),
            'smiles_nostereo': pair,
            'node_indices': np.arange(mol.GetNumAtoms()),
            'edge_indices': np.array([[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in mol.GetBonds()],
                                     dtype=int).reshape(-1, 2),
        }
    e.log(f'loaded {len(index_data_map)} enantiomers in {table["smiles_nostereo"].nunique()} pairs')
    return index_data_map, {}


def dataset_split(e, indices: list):
    split_of = dict(enumerate(read_table(e)['split']))
    train = [i for i in indices if split_of[i] == 'train']
    val = [i for i in indices if split_of[i] == 'val']
    test = [i for i in indices if split_of[i] == 'test']
    return train, val, test


def ranking_accuracy(pairs: dict, y_true: dict, y_pred: dict, tolerance: float) -> tuple:
    """Fraction of pairs whose predictions are in the order of the true scores (predictions closer than
    ``tolerance`` count as tied and half correct), number of pairs and number of tied pairs."""
    score, ties = 0.0, 0
    for a, b in pairs.values():
        delta_pred = y_pred[a] - y_pred[b]
        if abs(delta_pred) < tolerance:
            score += 0.5
            ties += 1
        elif np.sign(delta_pred) == np.sign(y_true[a] - y_true[b]):
            score += 1.0
    return score / len(pairs), len(pairs), ties


def evaluate_ranking(e, index_data_map: dict, indices: list, model, key: str, scaler=None, **kwargs):
    """Ranking accuracy over the complete enantiomer pairs among ``indices`` (runs after the default evaluation),
    and the accuracy of the thresholded predictions if ACCURACY_THRESHOLD is given."""
    y_pred = np.asarray(e.apply_hook('predict_model', index_data_map=index_data_map, model=model,
                                     indices=indices), dtype=float).reshape(-1)
    y_true = np.array([index_data_map[i]['graph_labels'][0] for i in indices], dtype=float)
    if scaler:
        y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        y_true = scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()

    members = defaultdict(list)
    for i in indices:
        members[index_data_map[i]['smiles_nostereo']].append(i)
    pairs = {k: v for k, v in members.items() if len(v) == 2}
    accuracy, num_pairs, num_ties = ranking_accuracy(pairs, dict(zip(indices, y_true)), dict(zip(indices, y_pred)),
                                                     e.TIE_TOLERANCE)
    # pairs whose two enantiomers have identical representations (up to floating-point noise): no model can rank
    # them, unlike ties of a model such as KNN that predicts the same value from the same neighbors
    num_identical = sum(np.allclose(np.asarray(index_data_map[a]['graph_features'], dtype=float),
                                    np.asarray(index_data_map[b]['graph_features'], dtype=float), rtol=0, atol=1e-12)
                        for a, b in pairs.values())
    e[f'metrics/{key}/ranking_accuracy'] = accuracy
    e[f'metrics/{key}/num_pairs'] = num_pairs
    e[f'metrics/{key}/num_tied_pairs'] = num_ties
    e[f'metrics/{key}/num_identical_pairs'] = int(num_identical)
    e.log(f' * ranking accuracy: {accuracy:.4f} over {num_pairs} pairs ({num_ties} tied, {num_identical} with '
          f'identical representations)')
    if e.ACCURACY_THRESHOLD is not None:
        threshold = e.ACCURACY_THRESHOLD
        e[f'metrics/{key}/accuracy'] = float(np.mean((y_pred > threshold) == (y_true > threshold)))
        e.log(f' * accuracy of the predictions thresholded at {threshold}: {e[f"metrics/{key}/accuracy"]:.4f}')
    if key.startswith('test'):
        e[f'predictions/{key}'] = {'indices': [int(i) for i in indices], 'y_pred': y_pred.tolist()}


def register_pair_hooks(experiment):
    """Registers the hooks of the pair benchmarks on ``experiment`` (which extends predict_molecules__hdc.py or
    predict_molecules__fp.py)."""
    experiment.hook('load_dataset', replace=True, default=False)(load_dataset)
    experiment.hook('dataset_split', replace=True, default=False)(dataset_split)
    experiment.hook('evaluate_model', replace=False, default=False)(evaluate_ranking)
