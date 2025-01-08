from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

# == DATASET PARAMETERS ==

# :param DATASET_NAME:
#       The name of the dataset to be used for the experiment. This name is used to download the dataset from the
#       ChemMatData file share.
DATASET_NAME: str = 'ames'
# :param DATASET_TYPE:
#       The type of the dataset, either 'classification' or 'regression'. This parameter is used to determine the
#       evaluation metrics and the type of the prediction target.
DATASET_TYPE: str = 'classification'
# :param NUM_TEST:
#       The number of test samples to be used for the evaluation of the models.
NUM_TEST: int = 100

# == EXPERIMENT PARAMETERS ==

experiment = Experiment.extend(
    'predict_molecules__gnn.py',
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals()
)

experiment.run_if_main()