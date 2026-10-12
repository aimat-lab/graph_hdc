"""
Properties that differ between the two enantiomers of a pair, predicted with HDF (experiment ex_21).

Two benchmarks share this module (and predict_enantiomers__fp.py, the same with Morgan fingerprints):

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
Baimacheva et al. The hooks are shared with the Morgan version in ``enantiomer_pairs.py``.

Everything else (HDF encoding, models, regression metrics) is the same as in ``predict_molecules__hdc.py``; the given
splits of the benchmark (or random splits by pair) replace the random split by molecule.
"""
import os
from typing import Optional

from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

from enantiomer_pairs import register_pair_hooks

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
#       "given" uses the split column of the table. "random" instead puts NUM_TEST pairs (an int) or a fraction
#       NUM_TEST of the pairs (a float), sampled with SEED, into the test split and all other pairs into the
#       training split, without validation split.
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
register_pair_hooks(experiment)

experiment.run_if_main()
