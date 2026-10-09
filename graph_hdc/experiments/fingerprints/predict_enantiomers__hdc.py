"""
Properties that differ between the two enantiomers of a pair, predicted with HDF (experiment ex_21).

Two benchmarks share this module:

* Ranking enantiomers by their docking scores (Adams et al., ChIRo, ICLR 2022): pairs of small enantiomers with one
  stereocenter, docked into the chiral pocket of PDB 4JBV; the best docking scores of the two enantiomers of a pair
  differ by at least 0.3 kcal/mol. Prepared by ``prepare_chiro_docking.py`` (target column top_score, the authors'
  training, validation and test splits).
* Elution order of enantiomers on the chiral HPLC column Chiralpak AD-H (CMRT dataset of Xu et al., Nat Commun 2023,
  in the subset of Baimacheva et al., J Cheminform 2025): the target is 1 for the enantiomer that elutes first and 0
  for the other. Prepared by ``prepare_cmrt_elution.py`` (target column first_eluted, the authors' training and test
  split).

A model is trained to predict the target (regression) on the training split. Its ranking accuracy is the fraction of
enantiomer pairs of the test split whose predictions are in the same order as their targets. A representation
without stereochemistry gives both enantiomers the same vector and hence the same prediction; such ties count as half
correct, so that it scores exactly 50 %. Predictions that differ by less than TIE_TOLERANCE also count as tied: the
HDF vectors of two enantiomers without stereo descriptors differ by floating-point noise (the summation order follows
the atom order), which the MLP turns into prediction differences of about 1e-7. For binary targets,
ACCURACY_THRESHOLD additionally gives the accuracy of the individual thresholded predictions, as reported by
Baimacheva et al.

Everything else (HDF encoding, models, regression metrics) is the same as in ``predict_molecules__hdc.py``; the given
splits of the benchmark (or random splits by pair) replace the random split by molecule.
"""
import os
import random
from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

# == DATASET PARAMETERS ==

# :param CHIRO_CSV:
#       The table written by prepare_chiro_docking.py (columns smiles, smiles_nostereo, top_score, split).
CHIRO_CSV: str = os.path.abspath(os.path.join(folder_path(__file__), '..', '..', '..', 'data', 'chiro_docking',
                                              'chiro_docking.csv'))
# :param PAIRS_CSV:
#       Another table of enantiomer pairs with the columns smiles, smiles_nostereo (identifies the pair), split and
#       TARGET_COLUMN (e.g. the one written by prepare_cmrt_elution.py). None uses CHIRO_CSV.
PAIRS_CSV: Optional[str] = None
# :param TARGET_COLUMN:
#       The column of the table that holds the target.
TARGET_COLUMN: str = 'top_score'
# :param SPLIT_MODE:
#       "given" uses the split column of the table. "random" instead puts a random fraction NUM_TEST of the pairs
#       (sampled with SEED) into the test split and all other pairs into the training split, without validation split.
SPLIT_MODE: str = 'given'
# :param DATASET_NAME:
#       Only used to name the caches of the HDF experiment.
DATASET_NAME: str = 'chiro_docking'
DATASET_NAME_ID: str = DATASET_NAME
DATASET_TYPE: str = 'regression'
TARGET_INDEX = None
# :param NUM_DATA:
#       If given, only this fraction (float) or number (int) of the enantiomer pairs of every split is used, sampled
#       with SEED (for smoke tests). None uses all pairs.
NUM_DATA = None
# :param TIE_TOLERANCE:
#       Predictions (in units of the target) of the two enantiomers of a pair that differ by less than this count as
#       tied. The docking scores are given in steps of 0.1 kcal/mol, and the scores of the two enantiomers of a pair
#       differ by at least 0.3 kcal/mol; the elution order is 0 or 1.
TIE_TOLERANCE: float = 1e-5
# :param ACCURACY_THRESHOLD:
#       If given, the accuracy of the individual predictions thresholded at this value is reported as well (for binary
#       targets such as the elution order, 0.5). None skips it.
ACCURACY_THRESHOLD: Optional[float] = None

# == EMBEDDING PARAMETERS ==

# :param STEREO:
#       Whether HDF encodes the CIP stereo descriptors of the atoms (see predict_molecules__hdc.py). Without them, the
#       two enantiomers of a pair receive identical fingerprints.
STEREO: bool = True

# == MODEL PARAMETERS ==

MODELS = ['neural_net2', 'k_neighbors']

# On a cache hit, predict_molecules__hdc.py only restores the HDF vectors, but the base experiment also needs the
# graph attributes, which the dicts of this module's load_dataset do not contain.
__CACHING__ = False

experiment = Experiment.extend(
    'predict_molecules__hdc.py',
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals()
)


def read_table(e: Experiment) -> pd.DataFrame:
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


@experiment.hook('load_dataset', replace=True, default=False)
def load_dataset(e: Experiment):
    table = read_table(e)
    index_data_map = {
        index: {
            'graph_repr': smiles,
            'graph_labels': np.array([target], dtype=float),
            'smiles_nostereo': pair,
        }
        for index, (smiles, pair, target) in enumerate(zip(table['smiles'], table['smiles_nostereo'],
                                                            table[e.TARGET_COLUMN]))
    }
    e.log(f'loaded {len(index_data_map)} enantiomers in {table["smiles_nostereo"].nunique()} pairs')
    return index_data_map, {}


@experiment.hook('dataset_split', replace=True, default=False)
def dataset_split(e: Experiment, indices: list):
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


@experiment.hook('evaluate_model', replace=False, default=False)
def evaluate_ranking(e: Experiment,
                     index_data_map: dict,
                     indices: list,
                     model,
                     key: str,
                     scaler=None,
                     **kwargs):
    """Ranking accuracy over the complete enantiomer pairs among ``indices`` (runs after the default evaluation)."""
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
    e[f'metrics/{key}/ranking_accuracy'] = accuracy
    e[f'metrics/{key}/num_pairs'] = num_pairs
    e[f'metrics/{key}/num_tied_pairs'] = num_ties
    e.log(f' * ranking accuracy: {accuracy:.4f} over {num_pairs} pairs ({num_ties} tied)')
    if e.ACCURACY_THRESHOLD is not None:
        threshold = e.ACCURACY_THRESHOLD
        e[f'metrics/{key}/accuracy'] = float(np.mean((y_pred > threshold) == (y_true > threshold)))
        e.log(f' * accuracy of the predictions thresholded at {threshold}: {e[f"metrics/{key}/accuracy"]:.4f}')
    if key.startswith('test'):
        e[f'predictions/{key}'] = {'indices': [int(i) for i in indices], 'y_pred': y_pred.tolist()}


experiment.run_if_main()
