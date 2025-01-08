import numpy as np
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

# == DATASET PARAMETERS ==

# :param DATASET_NAME:
#       The name of the dataset to be used for the experiment. This name is used to download the dataset from the
#       ChemMatData file share.
DATASET_NAME: str = 'qm9_smiles'
# :param DATASET_TYPE:
#       The type of the dataset, either 'classification' or 'regression'. This parameter is used to determine the
#       evaluation metrics and the type of the prediction target.
DATASET_TYPE: str = 'regression'
# :param NUM_TEST:
#       The number of test samples to be used for the evaluation of the models.
NUM_TEST: int = 1000
# :param TARGET_INDEX:
#       The index of the target in the graph labels. This parameter is used to determine the target of the
#       prediction task.
TARGET_INDEX: int = 7 # GAP

# == FINGERPRINT PARAMETERS ==

# :param FINGERPRINT_SIZE:
#       The size of the fingerprint to be generated. This will be the number of elements in the 
#       fingerprint vector representation of each molecule.
FINGERPRINT_SIZE: int = 2048
# :param FINGERPRINT_RADIUS:
#       The radius of the fingerprint to be generated. This parameter determines the number of
#       bonds to be considered when generating the fingerprint.
FINGERPRINT_RADIUS: int = 2

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
    # GAP
    return graph['graph_labels'][TARGET_INDEX:TARGET_INDEX+1]

experiment.run_if_main()