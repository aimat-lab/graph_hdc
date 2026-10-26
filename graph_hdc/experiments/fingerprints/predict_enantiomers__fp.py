"""
The enantiomer pair benchmarks of predict_enantiomers__hdc.py (ranking enantiomers by their docking scores, elution
order on Chiralpak AD-H) with Morgan fingerprints instead of HDF (experiment ex_21), as a reference with the same
splits, models and evaluation. FINGERPRINT_CHIRALITY computes the fingerprints with RDKit's includeChirality, which
takes the CIP descriptors of the stereocenters into account; without it, the two enantiomers of a pair receive
identical fingerprints. With it, RDKit still does not distinguish every pair (at radius 2, some enantiomers keep
identical fingerprints); the evaluation counts such pairs as num_identical_pairs. The hooks are shared with the HDF
version in ``enantiomer_pairs.py``.
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
#       tied (see predict_enantiomers__hdc.py).
TIE_TOLERANCE: float = 1e-5
# :param ACCURACY_THRESHOLD:
#       If given, the accuracy of the individual predictions thresholded at this value is reported as well (for binary
#       targets such as the elution order, 0.5). None skips it.
ACCURACY_THRESHOLD: Optional[float] = None

# == FINGERPRINT PARAMETERS ==

FINGERPRINT_TYPE: str = 'morgan'
FINGERPRINT_SIZE: int = 2048
FINGERPRINT_RADIUS: int = 2
# :param FINGERPRINT_CHIRALITY:
#       Whether the Morgan fingerprints are computed with RDKit's includeChirality (see predict_molecules__fp.py).
FINGERPRINT_CHIRALITY: bool = True

# == MODEL PARAMETERS ==

MODELS = ['neural_net2', 'k_neighbors']

__CACHING__ = False

experiment = Experiment.extend(
    'predict_molecules__fp.py',
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals()
)
register_pair_hooks(experiment)

experiment.run_if_main()
