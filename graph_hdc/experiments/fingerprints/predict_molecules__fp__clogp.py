import numpy as np
import rdkit.Chem as Chem
from rdkit.Chem.Crippen import MolLogP
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

# == DATASET PARAMETERS ==

# :param DATASET_NAME:
#       The name of the dataset to be used for the experiment. This name is used to download the dataset from the
#       ChemMatData file share.
DATASET_NAME: str = 'aqsoldb'
# :param DATASET_TYPE:
#       The type of the dataset, either 'classification' or 'regression'. This parameter is used to determine the
#       evaluation metrics and the type of the prediction target.
DATASET_TYPE: str = 'regression'
# :param NUM_TEST:
#       The number of test samples to be used for the evaluation of the models.
NUM_TEST: int = 1000

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

@experiment.hook('after_dataset', replace=False, default=False)
def after_dataset(e: Experiment,
                  index_data_map: dict[int, dict],
                  **kwargs,
                  ) -> None:
    """
    This hook is executed after the dataset is loaded. It is used to perform any additional processing
    on the dataset before the experiment is run.
    ---
    In this case, we use the RDKit library to calculate the CLogP values for the molecules in the dataset, 
    since we are using a dataset which does not contain the labels directly.
    """
    e.log('calculating CLogP values and replacing targets...')
    
    for _, graph in index_data_map.items():
        smiles = str(graph['graph_repr'])
        mol = Chem.MolFromSmiles(smiles)
        graph['graph_labels'] = np.array([MolLogP(mol)])

experiment.run_if_main()