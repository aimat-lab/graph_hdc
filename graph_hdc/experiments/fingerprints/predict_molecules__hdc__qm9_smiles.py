import numpy as np
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
NUM_TEST: int = 100
# :param TARGET_INDEX:
#       The index of the target in the graph labels. This parameter is used to determine the target of the
#       prediction task.
TARGET_INDEX: int = 7 # GAP

# == EMBEDDING PARAMETERS ==

# :param EMBEDDING_SIZE:
#       The size of the graph embedding vectors. This will be the number of elements in each of the 
#       hypervectors that represent the individual molecular graphs.
EMBEDDING_SIZE: int = 2048
# :param NUM_LAYERS:
#       The number of layers in the hypernetwork. This parameter determines the depth of the hypernetwork
#       which is used to generate the graph embeddings. This means it is the number of message passing 
#       steps applied in the encoder.
NUM_LAYERS: int = 2

# == EXPERIMENT PARAMETERS ==

experiment = Experiment.extend(
    'predict_molecules__hdc.py',
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