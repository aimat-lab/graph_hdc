import numpy as np
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

# == DATASET PARAMETERS ==

# :param DATASET_NAME:
#       The name of the dataset to be used for the experiment. This name is used to download the dataset from the
#       ChemMatData file share.
DATASET_NAME: str = 'tadf'
# :param DATASET_NAME_ID:
#       The name of the dataset to be used later on for the identification of the dataset. This name will NOT be used 
#       for the downloading of the dataset but only later on for identification. In most cases these will be the same 
#       but in cases for example one dataset is used as the basis of some deterministic calculation of the target values 
#       and in this case the name should identify it as such.
DATASET_NAME_ID: str = DATASET_NAME
# :param DATASET_TYPE:
#       The type of the dataset, either 'classification' or 'regression'. This parameter is used to determine the
#       evaluation metrics and the type of the prediction target.
DATASET_TYPE: str = 'regression'
# :param NUM_TEST:
#       The number of test samples to be used for the evaluation of the models.
NUM_TEST: int = 100
# :param NUM_DATA:
#       The number of samples to be used for the experiment. This parameter can be either an integer or a float between 0 and 1.
#       In case of an integer we use it as the number of samples to be used, in case of a float we use it as the fraction
#       of the dataset to be used. This parameter is used to limit the size of the dataset for the experiment.
NUM_DATA: float = 1.0
# :param TARGET_INDEX:
#       The index of the target in the graph labels. This parameter is used to determine the target of the
#       prediction task.
# 0: tadf rate
# 1: singlet triplet energy
# 2: oscilator strength
TARGET_INDEX: int = 0
# :param NUM_TRAIN:
#       The number of training samples to be used for the training of the models. This parameter
#       can be either an integer or a float between 0 and 1. In case of an integer we use it as the number
#       of training samples to be used, in case of a float we use it as the fraction of the dataset to be used.
NUM_TRAIN = 1.0

# == FINGERPRINT PARAMETERS ==

# :param FINGERPRINT_SIZE:
#       The size of the fingerprint to be generated. This will be the number of elements in the 
#       fingerprint vector representation of each molecule.
FINGERPRINT_SIZE: int = 8096
# :param FINGERPRINT_RADIUS:
#       The radius of the fingerprint to be generated. This parameter determines the number of
#       bonds to be considered when generating the fingerprint.
FINGERPRINT_RADIUS: int = 2

MODELS = ['neural_net2']

# == EXPERIMENT PARAMETERS ==

experiment = Experiment.extend(
    'predict_molecules__fp.py',
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals()
)

@experiment.hook('get_graph_labels', replace=True, default=False)
def get_graph_labels(e: Experiment,
                     index: int,
                     graph: dict,
                     **kwargs,
                     ) -> np.ndarray:
    return graph['graph_labels'][e.TARGET_INDEX:e.TARGET_INDEX+1]

experiment.run_if_main()