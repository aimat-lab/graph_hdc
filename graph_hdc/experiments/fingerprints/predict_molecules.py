import gc
import os
import time
import copy
import random
from typing import Any, List, Union, Tuple, Optional

import joblib
import torch
import torch.nn as nn
import pytorch_lightning as pl
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from torchmetrics import R2Score
from torchmetrics import Accuracy
from torchmetrics import MeanAbsoluteError
from rich.pretty import pprint
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdFingerprintGenerator
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, ElasticNet
from sklearn.svm import SVC, SVR
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.metrics import accuracy_score, f1_score, average_precision_score, log_loss
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace
from chem_mat_data._typing import GraphDict
from chem_mat_data.main import load_graph_dataset, load_dataset_metadata
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score


# :param IDENTIFIER:
#       String identifier that can be used to later on filter the experiment, for example.
IDENTIFIER: str = 'default'

# == DATASET PARAMETERS ==

# :param DATASET_NAME:
#       The name of the dataset to be used for the experiment. This name is used to download the dataset from the
#       ChemMatData file share.
DATASET_NAME: str = 'clintox'
# :param DATASET_NAME_ID:
#       The name of the dataset to be used later on for the identification of the dataset. This name will NOT be used 
#       for the downloading of the dataset but only later on for identification. In most cases these will be the same 
#       but in cases for example one dataset is used as the basis of some deterministic calculation of the target values 
#       and in this case the name should identify it as such.
DATASET_NAME_ID: str = DATASET_NAME
# :param TARGET_INDEX:
#       The index of the target in the graph labels. This parameter is used to determine the target of the
#       prediction task. If set to None, the full list of targets is used (for multi-target datasets).

TARGET_INDEX: Union[int, None] = None
# :param TARGET_INDICES:
#       Optional list of target indices that are all evaluated in this run (regression only), one after the other on
#       the same featurization and the same split, each with its own label scaler and models. The metrics of target t
#       are stored under the keys "test_<model>__t<t>" / "val_<model>__t<t>". Every target trains exactly as in a
#       single-target run with TARGET_INDEX=t (the models seed themselves before training), so this only saves the
#       loading and featurization (ex_27). None (default): one target given by TARGET_INDEX. Cannot be combined with
#       hooks that set the labels themselves, such as mixin_clogp.
TARGET_INDICES: Optional[List[int]] = None
# :param DATASET_TYPE:
#       The type of the dataset, either 'classification', 'binary', or 'regression'. This parameter is used to determine the
#       evaluation metrics and the type of the prediction target.
DATASET_TYPE: str = 'classification'
# :param NUM_DATA:
#       The number of samples to be used for the experiment. This parameter can be either an integer or a float between 0 and 1.
#       In case of an integer we use it as the number of samples to be used, in case of a float we use it as the fraction
#       of the dataset to be used. This parameter is used to limit the size of the dataset for the experiment.
NUM_DATA: Union[int, float] = None
# :param NUM_TEST:
#       The number of test samples to be used for the evaluation of the models. This parameter can be either an integer
#       or a float between 0 and 1. In case of an integer we use it as the number of test samples to be used, in case of
#       a float we use it as the fraction of the dataset to be used as test samples.
NUM_TEST: Union[int, float] = 0.1
# :param NUM_TRAIN:
#       The number of training samples to be used for the training of the models. This parameter can be either an integer
#       or a float between 0 and 1. In case of an integer we use it as the number of training samples to be used, in case
#       of a float we use it as the fraction of the dataset to be used as training samples.
NUM_TRAIN: Union[int, float] = 1.0
# :param DATASET_NOISE:
#       The additional amount of noise to be added to the dataset as a fraction. This noise will be added to both 
#       the training and testing samples.
DATASET_NOISE: float = 0.0
# :param NUM_VAL:
#       The number of validation samples to be used for the evaluation of the models during training.
NUM_VAL: int = 0.1
# :param SEED:
#       The random seed to be used for the experiment.
SEED: int = 1
# :param NN_SEED:
#       Optional separate seed for the neural network (neural_net2: weight initialization, internal validation
#       split and batch order). None uses SEED. Setting it retrains the network on the same split and the same
#       representation, which measures the run-to-run variation of the training itself (ex_20).
NN_SEED: Optional[int] = None
# :param USE_SMOTE:
#       Whether to use the SMOTE algorithm to oversample the minority class in the dataset. This is only used for
#       classification datasets. If set to True, the SMOTE algorithm will be applied to the training dataset after
#       the dataset has been split into training, validation, and test sets. The SMOTE algorithm will generate
#       synthetic samples for the minority class to balance the dataset.
USE_SMOTE: bool = True

# :param MODELS:
#       The list of models to be trained and evaluated. The models are trained and evaluated in the order they are
#       listed. The model names are dynamically evaluated as function names with the prefix 'train_model__{name}'.
#       if such a function exists in the experiment workspace, it is executed to train the model. The model is then
#       evaluated using the 'evaluate_model' function.
MODELS: List[str] = [
    'random_forest',
    'grad_boost',
    'k_neighbors',
    # 'gaussian_process',
    'neural_net',
    'linear',
    #'support_vector',
]

# :param SAVE_DATASET:
#       This flag determines whether the dataset should be saved to disk as a NPZ file
#       after being processed
SAVE_DATASET: bool = False

# == MODEL SPECIFIC PARAMETERS ==
# The following parameters are used to configure the individual models. They need to be 
# defined as top level parameters of the experiment like this to enable the hyperparameter 
# optimization in the outer loop.

RF_NUM_ESTIMATORS: int = 100
RF_MAX_DEPTH: int = 10
RF_MAX_FEATURES: str = 'sqrt'
# :param RF_NUM_JOBS:
#       The number of threads of the random forest (n_jobs of sklearn). -1 (default) uses all cores; when several runs
#       share a node (ex_28 on JUPITER), each run gets its share of the cores instead.
RF_NUM_JOBS: int = -1

GB_NUM_ESTIMATORS: int = 100
GB_LEARNING_RATE: float = 0.01
GB_MAX_DEPTH: int = 3

LN_ALPHA: float = 1e-2
LN_L1_RATIO: float = 0.5
LN_FIT_INTERCEPT: bool = True

KN_NUM_NEIGHBORS: int = 5
KN_WEIGHTS: str = 'uniform'
# Distance metric passed to sklearn's KNeighborsRegressor / KNeighborsClassifier.
# Common choices: 'minkowski' (sklearn default, p=2 = Euclidean), 'cosine',
# 'jaccard' (binary inputs only, equivalent to Tanimoto for Morgan fingerprints).
KN_METRIC: str = 'minkowski'
# :param KN_BACKEND:
#       'sklearn' (default): KNeighborsRegressor. 'torch': the same exact brute-force search with PyTorch, on the GPU if
#       one is available (regression with uniform or distance weights and the metrics 'minkowski' (Euclidean), 'cosine'
#       or 'jaccard'), which is much faster for the Jaccard (Tanimoto) distance on large datasets (ex_27, ex_28).
#       Neighbors at exactly equal distance may be chosen differently than by sklearn.
KN_BACKEND: str = 'sklearn'

NN_HIDDEN_LAYER_SIZES: Tuple[int] = (100, 100, 100)
NN_ALPHA: float = 0.0001
NN_LEARNING_RATE_INIT: float = 0.001
# :param NN_NUM_WORKERS:
#       Number of worker processes for the neural_net2 (PyTorch-Lightning) DataLoaders. Defaults to 0:
#       the feature vectors live in memory, so workers add little, and when hundreds of experiment tasks
#       are packed onto a single node (e.g. a JUPITER Booster node), spawning 4 workers per task thrashes
#       the machine. Increase only for a single-task GPU run on a very large dataset.
NN_NUM_WORKERS: int = 0
# :param NN_DEVICE_DATA:
#       If True, neural_net2 keeps the training and validation data as tensors on the training device (the GPU if one
#       is available) and indexes whole batches at once, instead of building one tensor per molecule in a DataLoader.
#       The network, the batches (same shuffling, drawn from the same seeded generator), the epochs and the optimizer
#       are those of the default path; only the data handling is faster (ex_27). Default False.
NN_DEVICE_DATA: bool = False

# == EXPERIMENT PARAMETERS ==

# :param NOTE:
#       A note that can be used to describe the experiment. This note will be stored as 
#       part of the experiment metadata and can later serve for identification and so on.
NOTE: str = ''

__DEBUG__: bool = True
__NOTIFY__: bool = False

experiment = Experiment(
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals()
)

# == UTILS ==


class BestModelRestorer(pl.Callback):
    """
    This class implements a PyTorch Lightning callback which will restore the model weights to 
    that state which achieved the best validation loss observed during the training process.
    
    This is done by monitoring a specific metric (e.g. 'val_loss') and saving the model state
    whenever the monitored metric improves. Using a hook at the very end of the training, the 
    model weights are reset to that best state.
    """
    
    def __init__(self, 
                 monitor: str = "val_loss", 
                 mode: str = "min"
                 ) -> None:
        super().__init__()
        self.monitor = monitor
        if mode not in ["min", "max"]:
            raise ValueError("mode must be 'min' or 'max'.")
        self.mode = mode

        # This will variable will store the best score observed during the training.
        self.best_score: float = None
        # This will store the best model state dict associated with the best score.
        self.best_state_dict = None
        # This will store the time when the best score was achieved.
        self.best_time = None

    def on_fit_start(self, trainer, pl_module):
        """
        Initialize the best score before starting the fit.
        """
        if self.mode == "min":
            self.best_score = float("inf")
        else:
            self.best_score = -float("inf")
        self.best_state_dict = None

    def on_validation_end(self, trainer, pl_module):
        """
        Called at the end of the validation loop. We check whether the monitored metric i
        mproved and if so, store the model state dict and log the improvement.
        """
        metrics = trainer.callback_metrics
        current_score = metrics.get(self.monitor)

        if current_score is None:
            # Metric not found, cannot update best score
            return

        if (
            (self.mode == "min" and current_score < self.best_score) or
            (self.mode == "max" and current_score > self.best_score)
        ):
            # Update best score and store model weights
            self.best_score = current_score
            self.best_state_dict = {
                k: copy.deepcopy(v.detach().cpu().clone())
                for k, v in pl_module.state_dict().items()
            }
            self.best_time = time.time()

            # Log the new best score (if the logger is available)
            if trainer.logger is not None:
                trainer.logger.log_metrics({f"best_{self.monitor}": current_score}, step=trainer.global_step)
                
            # You could also print a message if desired:
            trainer.print(
                f"New best {self.monitor}={current_score:.4f} at step={trainer.global_step}."
            )

    def on_fit_end(self, trainer, pl_module):
        """
        At the end of training, restore the model to the best recorded state.
        """
        if self.best_state_dict is not None:
            current_state_dict = pl_module.state_dict()
            pl_module.load_state_dict(self.best_state_dict)
            trainer.print(
                f"Restored the best model with {self.monitor}={self.best_score:.4f}."
            )


class NeuralNet(pl.LightningModule):
    """
    A simple multi-layer neural network for regression or classification tasks based 
    on tabular input of fixed size.
    """
    
    def __init__(self, 
                 input_dim: int, 
                 output_dim: int,
                 hidden_units: list[int] = [100, 100, 100],
                 learning_rate: float = 1e-5,
                 loss_function: str = 'mse', # or 'bce'
                 ) -> None:
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate
        self.loss_function = loss_function
        
        self.layers = nn.ModuleList()
        prev_units = self.input_dim
        for units in self.hidden_units:
            self.layers.append(nn.Sequential(
                nn.Linear(prev_units, units),
                nn.BatchNorm1d(units),
                nn.ReLU(),
            ))
            prev_units = units
            
        self.layers.append(nn.Linear(prev_units, self.output_dim))

        if self.loss_function == 'mse':
            self.criterion = nn.MSELoss()
            self.metric = MeanAbsoluteError(num_outputs=self.output_dim,)
            
        elif self.loss_function == 'bce':
            self.criterion = nn.BCEWithLogitsLoss()
            self.metric = Accuracy(num_classes=self.output_dim, average='macro')
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        for layer in self.layers:
            x = layer(x)
            
        return x

    def training_step(self, 
                      batch: Tuple[torch.Tensor, torch.Tensor], 
                      batch_idx: int
                      ) -> torch.Tensor:
        x, y = batch
        y_hat = self.forward(x)
        y_hat
        
        loss = self.criterion(y_hat, y)
        return loss
    
    def validation_step(self,
                        batch: Tuple[torch.Tensor, torch.Tensor],
                        batch_idx: int
                        ) -> torch.Tensor:
        x, y = batch
        
        y_hat = self.forward(x)
        value = self.metric(y_hat, y)
        self.log(
            'val_loss', value, 
            on_step=False, 
            on_epoch=True, 
            prog_bar=True, 
            logger=True
        )
        return value
        
    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.learning_rate,
            weight_decay=1e-5,
        )
        return optimizer
    
    # --- implement sklearn interface ---
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        
        x_tensor = torch.Tensor(X)
        loader = torch.utils.data.DataLoader(
            x_tensor, 
            batch_size=64, 
            shuffle=False
        )
        
        outputs: list[np.ndarray] = []
        with torch.no_grad():
            for data in loader:
                y_hat = self.forward(data)
                outputs.append(y_hat.cpu().numpy())
                
        return np.concatenate(outputs, axis=0)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        
        y = self.predict(X)
        if self.loss_function == 'bce':
            y = torch.sigmoid(torch.Tensor(y))
        elif self.loss_function == 'mse':
            raise ValueError('Cannot use predict_proba with MSE loss function.')
    
        return y.cpu().numpy()

class DeviceBatchDataset(torch.utils.data.Dataset):
    """
    The features and labels of the given indices as two tensors on a device; indexing with a list of positions
    returns the whole batch at once (NN_DEVICE_DATA). Used with a BatchSampler as the DataLoader's sampler.
    """

    def __init__(self, index_data_map: dict, indices: list, device: torch.device):
        self.features = torch.tensor(np.stack([index_data_map[i]['graph_features'] for i in indices]),
                                     dtype=torch.float32, device=device)
        self.labels = torch.tensor(np.stack([np.atleast_1d(index_data_map[i]['graph_labels']) for i in indices]),
                                   dtype=torch.float32, device=device)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, positions):
        return self.features[positions], self.labels[positions]


class TorchKNNRegressor:
    """
    Exact brute-force k-nearest-neighbor regression in PyTorch, on the GPU if one is available (KN_BACKEND='torch').
    The same search as sklearn's KNeighborsRegressor(algorithm='brute') for the metrics 'minkowski'/'euclidean'
    (Euclidean distance), 'cosine' and 'jaccard' (binary features: the Jaccard distance is one minus the Tanimoto
    similarity). Unlike sklearn, ties are broken deterministically: among training samples at
    exactly the same distance, the one that comes first in the training data is chosen. Binary fingerprints of small
    molecules often have several neighbors at the same Tanimoto distance (26 of 63 FreeSolv test molecules have a tie
    between their 5th and 6th neighbor), so the MAE can differ slightly from sklearn's, whose choice among ties is
    arbitrary. Similarities are computed in float64 (for bit vectors the Tanimoto similarity is a ratio of exact
    integer counts, so CPU and GPU agree exactly), and the neighbors are the first k entries of a stable descending
    sort of the similarities, which keeps tied entries in the training order. The queries are processed in batches
    of ``batch_size``; a batch needs about four batch_size x n_train float64 matrices of device memory besides the
    training matrix (QM9: 0.25 GB each for 256 queries).

    ``weights`` as in sklearn: 'uniform' averages the k neighbors, 'distance' weights them by the inverse of their
    distance (1 - Tanimoto similarity, 1 - cosine similarity, or the Euclidean distance), and a query with neighbors at
    distance zero takes the mean of those neighbors only (sklearn's rule for exact matches; ex_28).
    """

    def __init__(self, n_neighbors: int = 5, metric: str = 'minkowski', batch_size: int = 256,
                 weights: str = 'uniform'):
        if metric not in ('minkowski', 'euclidean', 'cosine', 'jaccard'):
            raise ValueError(f'TorchKNNRegressor does not support the metric "{metric}"')
        if weights not in ('uniform', 'distance'):
            raise ValueError(f'TorchKNNRegressor does not support the weights "{weights}"')
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.batch_size = batch_size
        self.weights = weights

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'TorchKNNRegressor':
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        X = np.asarray(X, dtype=np.float64)
        if self.metric == 'jaccard':
            X = (X != 0).astype(np.float64)
        self.X = torch.tensor(X, device=self.device)
        if self.metric == 'jaccard':
            self.counts = self.X.sum(dim=1)
        elif self.metric == 'cosine':
            # in place, otherwise as torch.nn.functional.normalize (x / max(||x||, 1e-12))
            self.X.div_(self.X.norm(dim=1, keepdim=True).clamp_min(1e-12))
        self.y = torch.tensor(np.asarray(y), dtype=torch.float64, device=self.device)
        if self.y.dim() == 1:
            self.y = self.y[:, None]
        return self

    def similarity(self, query: torch.Tensor) -> torch.Tensor:
        """Larger = closer, for a batch of queries against all training samples."""
        query = query.double()
        if self.metric == 'jaccard':
            query = (query != 0).double()
            intersection = query @ self.X.T
            union = self.counts[None, :] - intersection
            union += query.sum(dim=1, keepdim=True)
            # two empty fingerprints have Jaccard distance 0 (as in scipy)
            empty = union == 0
            similarity = intersection.div_(union.clamp_(min=1))
            similarity[empty] = 1.0
            return similarity
        if self.metric == 'cosine':
            return torch.nn.functional.normalize(query, dim=1) @ self.X.T
        return -torch.cdist(query, self.X)

    def distance(self, similarity: torch.Tensor) -> torch.Tensor:
        """The distances of the metric from the similarities returned by ``similarity``."""
        if self.metric in ('jaccard', 'cosine'):
            # sklearn clips the cosine distance to [0, 2]; the Jaccard distance is within [0, 1] anyway
            return (1.0 - similarity).clamp(min=0.0)
        return -similarity

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        out = []
        with torch.no_grad():
            for start in range(0, len(X), self.batch_size):
                query = torch.tensor(X[start:start + self.batch_size], dtype=torch.float64, device=self.device)
                similarity = self.similarity(query)
                order = torch.sort(similarity, dim=1, descending=True, stable=True).indices[:, :self.n_neighbors]
                neighbors = self.y[order]
                if self.weights == 'uniform':
                    out.append(neighbors.mean(dim=1))
                    continue
                distance = self.distance(torch.gather(similarity, 1, order))
                exact = distance == 0
                # 1 / distance, except for queries with exact matches, which use only those (as sklearn)
                weight = torch.where(exact.any(dim=1, keepdim=True), exact.double(),
                                     1.0 / distance.clamp(min=torch.finfo(torch.float64).tiny))
                out.append((neighbors * weight[:, :, None]).sum(dim=1) / weight.sum(dim=1, keepdim=True))
        return torch.cat(out).cpu().numpy()


# == EXPERIMENT IMPLEMENTATION ==

@experiment.hook('load_dataset', replace=False, default=True)
def load_dataset(e: Experiment) -> dict[int, GraphDict]:
    
    ## -- Dataset Loading --
    # This function will download the dataset from the ChemMatData file share and return the already pre-processed 
    # list of graph dict representations.
    graphs: list[GraphDict] = load_graph_dataset(
        e.DATASET_NAME,
        folder_path='/tmp'
    )
    
    # metadata = load_dataset_metadata(
    #     e.DATASET_NAME,
    # )
    metadata = {}
    
    index_data_map = dict(enumerate(graphs))
    
    ## -- Sub-sampling --
    # If the NUM_DATA parameter is set to either an integer or a float, we will sub-sample the dataset
    # randomly according to that number/fraction. 
    if e.NUM_DATA is not None:
        
        if isinstance(e.NUM_DATA, int):
            num_data = e.NUM_DATA
        elif isinstance(e.NUM_DATA, float):
            num_data = int(e.NUM_DATA * len(index_data_map))
            
        # subsample the dataset to the specified number of samples
        random.seed(e.SEED)
        index_data_map = dict(
            random.sample(
                list(index_data_map.items()), 
                k=num_data
            )
        )
    
    return index_data_map, metadata


@experiment.hook('get_graph_labels', default=True)
def get_graph_labels(e: Experiment,
                     index: int,
                     graph: dict
                     ) -> np.ndarray:
    """
    This hook gets called during the processing of the dataset and gets the index of an element
    in the dataset as well as the graph dict representing data element. This hook is supposed to 
    return the array of target values for the graph.
    """
    
    # If a specific index in the list of targets is declared that we use that one otherwise we return 
    # the full list of targets. This is the case for example in multi-target datasets where we want to
    # only predict one of the targets. For multi-classification we need to return the one hot target 
    # vector.
    if e.TARGET_INDEX is not None:
        return graph['graph_labels'][e.TARGET_INDEX:e.TARGET_INDEX+1]
    else:
        return graph['graph_labels']


# @experiment.hook('get_graph_labels', replace=True, default=False)
# def get_graph_labels(e: Experiment,
#                      index: int,
#                      graph: dict,
#                      **kwargs,
#                      ) -> np.ndarray:
#     return graph['graph_labels'][e.TARGET_INDEX:e.TARGET_INDEX+1]


@experiment.hook('filter_dataset', replace=False, default=True)
def filter_dataset(e: Experiment,
                   index_data_map: dict[int, dict],
                   ) -> tuple[list, list, list]:
    
    e.log(f'filtering dataset to remove invalid SMILES and unconnected graphs...')
    e.log(f'starting with {len(index_data_map)} samples...')
    indices = list(index_data_map.keys())
    for index in indices:
        graph = index_data_map[index]
        smiles = graph['graph_repr']
        
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            del index_data_map[index]
            continue
            
        if len(mol.GetAtoms()) < 2:
            del index_data_map[index]
            continue
            
        if len(mol.GetBonds()) < 1:
            del index_data_map[index]
            continue
            
        # disconnected graphs
        if '.' in smiles:
            del index_data_map[index]
            continue
            
    e.log(f'finished filtering dataset with {len(index_data_map)} samples remaining.')


@experiment.hook('dataset_split', replace=False, default=True)
def dataset_split(e: Experiment,
                  indices: list[int],
                  ) -> tuple[list, list, list]:
    
    random.seed(e.SEED)
    
    # We accept NUM_TEST here to be either an integer or a float between 0 and 1
    # in case of an integer we use it as the number of test samples to be used
    # in case of a float we use it as the fraction of the dataset to be used as test samples.
    if isinstance(e.NUM_TEST, int):
        num_test = e.NUM_TEST
    elif isinstance(e.NUM_TEST, float):
        num_test = int(e.NUM_TEST * len(indices))
    
    test_indices = random.sample(indices, k=num_test)
    indices = list(set(indices) - set(test_indices))
    
    if isinstance(e.NUM_VAL, int):
        num_val = e.NUM_VAL
    elif isinstance(e.NUM_VAL, float):
        num_val = int(e.NUM_VAL * len(indices))
        
    val_indices = random.sample(indices, k=num_val)
    indices = list(set(indices) - set(val_indices))
    
    # We accept NUM_TRAIN here to be either an integer or a float between 0 and 1
    # in case of an integer we use it as the number of training samples to be used
    # in case of a float we use it as the fraction of the dataset to be used as training samples.
    # We then sub-sample the training indices from the remaining indices.
    if isinstance(e.NUM_TRAIN, int):
        num_train = e.NUM_TRAIN
    elif isinstance(e.NUM_TRAIN, float):
        num_train = int(e.NUM_TRAIN * len(indices))
    
    # We need to make sure that even if num_train is a very low float that we keep at least 3 
    # samples in the train set.
    num_train = max(num_train, 3)
    train_indices = random.sample(indices, k=num_train)
    
    return train_indices, val_indices, test_indices
    
    
@experiment.hook('process_dataset', replace=False, default=True)
def process_dataset(e: Experiment,
                    index_data_map: dict
                    ) -> None:
    for index, graph in index_data_map.items():
        smiles: str = graph['graph_repr']
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
        fingerprint = gen.GetFingerprint(Chem.MolFromSmiles(smiles))
        graph['graph_features'] = np.array(fingerprint).astype(float)
    

@experiment.hook('after_dataset', replace=False, default=True)
def after_dataset(e: Experiment,
                  index_data_map: dict,
                  train_indices: list[int],
                  test_indices: list[int],
                  val_indices: list[int],
                  **kwargs,
                  ) -> None:
    
    # Plotting the histogram of graph sizes
    e.log('plotting histogram of graph sizes...')
    graph_sizes = [len(index_data_map[i]['node_indices']) for i in index_data_map.keys()]

    mean_size = np.mean(graph_sizes)
    p10_size = np.percentile(graph_sizes, 10)
    p90_size = np.percentile(graph_sizes, 90)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(graph_sizes, bins=30, ax=ax)
    ax.axvline(mean_size, color='black', linestyle='-', label=f'Mean: {mean_size:.2f}')
    ax.axvline(p10_size, color='black', linestyle='--', label=f'10th Percentile: {p10_size:.2f}')
    ax.axvline(p90_size, color='black', linestyle='--', label=f'90th Percentile: {p90_size:.2f}')
    ax.legend()
    ax.set_title(f'Histogram of Graph Sizes')
    ax.set_xlabel('Number of Nodes')
    ax.set_ylabel('Count')
    e.commit_fig('graph_size_histogram.png', fig)
    
    if e.DATASET_TYPE == 'classification':
        
        # ~ plotting the label distribution
        
        e.log('plotting label distribution...')
        labels = np.array([np.argmax(index_data_map[i]['graph_labels']) for i in train_indices])

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(x=labels.flatten(), ax=ax)
        ax.set_title('Label Distribution')
        ax.set_xlabel('Labels')
        ax.set_ylabel('Count')
        e.commit_fig('label_distribution.png', fig)

    elif e.DATASET_TYPE == 'regression':
        
        # ~ plotting the value distribution
        
        e.log('plotting value distribution...')
        values = np.array([index_data_map[i]['graph_labels'] for i in train_indices])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(values.flatten(), ax=ax)
        ax.set_title('Value Distribution')
        ax.set_xlabel('Values')
        ax.set_ylabel('Count')
        e.commit_fig('value_distribution.png', fig)


@experiment.hook('train_model__random_forest', replace=False, default=True)
def train_model__random_forest(e: Experiment,
                               index_data_map: dict,
                               train_indices: list[int],
                               val_indices: list[int],
                               ) -> Any:
    
    X_train = np.array([index_data_map[i]['graph_features'] for i in train_indices])
    y_train = np.array([index_data_map[i]['graph_labels'] for i in train_indices])
    
    kwargs = {
        'n_estimators': e.RF_NUM_ESTIMATORS,
        'max_depth': e.RF_MAX_DEPTH,
        'max_features': e.RF_MAX_FEATURES,
        'bootstrap': True,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'n_jobs': e.RF_NUM_JOBS,
        'random_state': e.SEED,
    }
    
    time_start = time.time()
    if e.DATASET_TYPE == 'classification':
        
        model = MultiOutputClassifier(RandomForestClassifier(**kwargs))
        model.fit(X_train, y_train)
    
    elif e.DATASET_TYPE == 'binary':
        
        y_train = np.argmax(y_train.astype(int), axis=-1)
        # Optionally using SMOTE to balance the labels of the dataset
        if not e.USE_SMOTE:
            model = RandomForestClassifier(**kwargs)
            model.fit(X_train, y_train)
        else:
            model = Pipeline([
                ('smote', SMOTE(random_state=e.SEED, k_neighbors=3)),
                ('rf', RandomForestClassifier(**kwargs))
            ])
            model.fit(X_train, y_train)
    
    elif e.DATASET_TYPE == 'regression':
        
        model = RandomForestRegressor(**kwargs)
        model.fit(X_train, y_train)
    
    time_end = time.time()
    e['train_time/random_forest'] = time_end - time_start    
    
    return model
    
    
@experiment.hook('train_model__grad_boost', replace=False, default=True)
def train_model__grad_boost(e: Experiment,
                            index_data_map: dict,
                            train_indices: list[int],
                            val_indices: list[int]
                            ) -> Any:
    
    X_train = np.array([index_data_map[i]['graph_features'] for i in train_indices])
    y_train = np.array([index_data_map[i]['graph_labels'] for i in train_indices])
    
    kwargs = {
        'n_estimators': e.GB_NUM_ESTIMATORS,
        'max_depth': e.GB_MAX_DEPTH,
        'learning_rate': e.GB_LEARNING_RATE,
        'subsample': 0.75,
        'max_features': 'sqrt',
        'random_state': e.SEED,
    }
    
    time_start = time.time()
    if e.DATASET_TYPE == 'classification':
        
        model = MultiOutputClassifier(GradientBoostingClassifier(**kwargs))
        model.fit(X_train, y_train)
        
    elif e.DATASET_TYPE == 'binary':
        
        y_train = np.argmax(y_train.astype(int), axis=-1)
        # Optionally using SMOTE to balance the labels of the dataset
        if not e.USE_SMOTE:
            model = GradientBoostingClassifier(
                **kwargs,
            )
            model.fit(X_train, y_train)
        else:
            model = Pipeline([
                ('smote', SMOTE(random_state=e.SEED, k_neighbors=3)),
                ('gb', GradientBoostingClassifier(**kwargs))
            ])
            model.fit(X_train, y_train)
    
    if e.DATASET_TYPE == 'regression':
        
        model = GradientBoostingRegressor(
            **kwargs,
        )
        model.fit(X_train, y_train)
        
    time_end = time.time()
    e['train_time/grad_boost'] = time_end - time_start

    return model

    
@experiment.hook('train_model__k_neighbors', replace=False, default=True)
def train_model__k_neighbors(e: Experiment,
                             index_data_map: dict,
                             train_indices: list[int],
                             val_indices: list[int]
                             ) -> Any:
    
    X_train = np.array([index_data_map[i]['graph_features'] for i in train_indices])
    y_train = np.array([index_data_map[i]['graph_labels'] for i in train_indices])
    
    kwargs = {
        'n_neighbors': e.KN_NUM_NEIGHBORS,
        'weights': e.KN_WEIGHTS,
        'metric': e.KN_METRIC,
        'n_jobs': -1,
    }
    
    time_start = time.time()
    if e.DATASET_TYPE == 'classification':
        
        model = MultiOutputClassifier(KNeighborsClassifier(**kwargs))
        model.fit(X_train, y_train)
        return model
    
    elif e.DATASET_TYPE == 'binary':
        y_train = np.argmax(y_train.astype(int), axis=-1)
        # Optionally using SMOTE to balance the labels of the dataset
        if not e.USE_SMOTE:
            model = KNeighborsClassifier(**kwargs)
            model.fit(X_train, y_train)
            return model
        else:
            model = Pipeline([
                ('smote', SMOTE(random_state=e.SEED, k_neighbors=3)),
                ('knn', KNeighborsClassifier(**kwargs))
            ])
            model.fit(X_train, y_train)
            return model
    
    elif e.DATASET_TYPE == 'regression':

        if e.KN_BACKEND == 'torch':
            model = TorchKNNRegressor(n_neighbors=e.KN_NUM_NEIGHBORS, metric=e.KN_METRIC, weights=e.KN_WEIGHTS)
            model.fit(X_train, y_train)
            e['train_time/k_neighbors'] = time.time() - time_start
            return model

        model = KNeighborsRegressor(**kwargs)
        model.fit(X_train, y_train)
        return model
    
    time_end = time.time()
    e['train_time/k_neighbors'] = time_end - time_start
    
    return model
    

@experiment.hook('train_model__gaussian_process', replace=False, default=True)
def train_model__gaussian_process(e: Experiment,
                                  index_data_map: dict,
                                  train_indices: list[int],
                                  val_indices: list[int]
                                  ) -> Any:
    
    X_train = np.array([index_data_map[i]['graph_features'] for i in train_indices])
    y_train = np.array([index_data_map[i]['graph_labels'] for i in train_indices])
    
    kwargs = {
        'n_restarts_optimizer': 3,
    }
    
    if e.DATASET_TYPE == 'classification':
        
        model = MultiOutputClassifier(GaussianProcessClassifier(**kwargs))
        model.fit(X_train, y_train)
        
        return model
    
    elif e.DATASET_TYPE == 'binary':
        
        y_train = np.argmax(y_train.astype(int), axis=-1)
        model = GaussianProcessClassifier(**kwargs)
        model.fit(X_train, y_train)
        
        return model
    
    elif e.DATASET_TYPE == 'regression':
        
        y_train = np.argmax(y_train.astype(int), axis=-1)
        model = GaussianProcessRegressor(**kwargs)
        model.fit(X_train, y_train)
        
        return model
    
    
@experiment.hook('train_model__neural_net', replace=False, default=True)    
def train_model__neural_net(e: Experiment,
                            index_data_map: dict,
                            train_indices: list[int],
                            val_indices: list
                            ) -> Any:
    
    X_train = np.array([index_data_map[i]['graph_features'] for i in train_indices])
    y_train = np.array([index_data_map[i]['graph_labels'] for i in train_indices])
    
    kwargs = {
        'hidden_layer_sizes': e.NN_HIDDEN_LAYER_SIZES,
        #'alpha': e.NN_ALPHA,
        #'learning_rate_init': e.NN_LEARNING_RATE_INIT,
        'max_iter': 2000,
        'early_stopping': True,
        'validation_fraction': 0.2,
        'n_iter_no_change': 100,
        'solver': 'adam',
        'activation': 'relu',
        'random_state': e.SEED,
    }
    
    time_start = time.time()
    if e.DATASET_TYPE == 'classification':
        
        model = MultiOutputClassifier(MLPClassifier(**kwargs))
        model.fit(X_train, y_train)
    
    elif e.DATASET_TYPE == 'binary':
        
        y_train = np.argmax(y_train.astype(int), axis=-1)
        
        # Optionally using SMOTE to balance the labels of the dataset
        if not e.USE_SMOTE:
            model = MLPClassifier(**kwargs)
            model.fit(X_train, y_train)
            
        else:
            model = Pipeline([
                ('smote', SMOTE(random_state=e.SEED, k_neighbors=3)),
                ('mlp', MLPClassifier(**kwargs))
            ])
            model.fit(X_train, y_train)
    
    elif e.DATASET_TYPE == 'regression':
        
        model = MLPRegressor(**kwargs)
        model.fit(X_train, y_train)

    time_end = time.time()
    e['train_time/neural_net'] = time_end - time_start
    
    return model


@experiment.hook('train_model__neural_net2', replace=False, default=True)
def train_model__neural_net2(e: Experiment,
                             index_data_map: dict,
                             train_indices: list[int],
                             val_indices: list[int]
                             ) -> Any:
    """
    This hook trains a PytorchLightning neural network model.
    """
    # Seed python, numpy and torch so that the internal validation split, the weight initialization and
    # the batch order are reproducible for a given SEED (and the internal validation split is identical
    # to the one of the end-to-end GNNs, which seed the same way).
    pl.seed_everything(e.SEED if e.NN_SEED is None else e.NN_SEED, workers=True)

    num_val = max(2, int(0.05 * len(train_indices)))
    val_indices_ = random.sample(train_indices, k=num_val)
    train_indices = list(set(train_indices) - set(val_indices_))
    e.log(f'internally using {len(train_indices)} training samples '
          f'and {len(val_indices_)} validation samples.')
    
    ## --- converting data to torch dataset ---
    # Memory-efficient dataset that creates tensors on-the-fly
    class LazyDataset(torch.utils.data.Dataset):
        def __init__(self, indices, data_map):
            self.indices = indices
            self.data_map = data_map

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            data_idx = self.indices[idx]
            features = torch.tensor(self.data_map[data_idx]['graph_features'], dtype=torch.float32)
            labels = torch.tensor(self.data_map[data_idx]['graph_labels'], dtype=torch.float32)
            return features, labels

    # DataLoader worker settings. With NN_NUM_WORKERS == 0 (the default, best for dense node packing)
    # loading happens in the main process; prefetch_factor/persistent_workers are only valid with workers.
    loader_kwargs = {'num_workers': e.NN_NUM_WORKERS, 'pin_memory': False}
    if e.NN_NUM_WORKERS and e.NN_NUM_WORKERS > 0:
        loader_kwargs['prefetch_factor'] = 2
        loader_kwargs['persistent_workers'] = True

    device_datasets = []
    if e.NN_DEVICE_DATA:
        # The data as tensors on the training device, indexed one whole batch at a time. The samplers are those that
        # the DataLoaders below build internally (RandomSampler / SequentialSampler in a BatchSampler of 64), so the
        # batches come in the same order, drawn from the same seeded global generator.
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device_datasets = [DeviceBatchDataset(index_data_map, train_indices, device),
                           DeviceBatchDataset(index_data_map, val_indices_, device)]
        train_loader = torch.utils.data.DataLoader(
            device_datasets[0],
            batch_size=None,
            sampler=torch.utils.data.BatchSampler(torch.utils.data.RandomSampler(range(len(train_indices))),
                                                  batch_size=64, drop_last=True),
        )
        val_loader = torch.utils.data.DataLoader(
            device_datasets[1],
            batch_size=None,
            sampler=torch.utils.data.BatchSampler(torch.utils.data.SequentialSampler(range(len(val_indices_))),
                                                  batch_size=64, drop_last=False),
        )
    else:
        # Create memory-efficient datasets
        train_dataset = LazyDataset(train_indices, index_data_map)

        # Create a DataLoader with optimized settings for performance
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=64,
            shuffle=True,
            drop_last=True,
            **loader_kwargs,
        )

        # Create memory-efficient validation dataset
        val_dataset = LazyDataset(val_indices_, index_data_map)

        # Create a DataLoader for the validation dataset with optimized settings
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=64,
            shuffle=False,
            **loader_kwargs,
        )

    ## --- creating the neural network model ---
    
    # input output dimensions from sample data
    sample_features = index_data_map[train_indices[0]]['graph_features']
    sample_labels = index_data_map[train_indices[0]]['graph_labels']
    input_dim = len(sample_features) if isinstance(sample_features, (list, tuple)) else sample_features.shape[0]
    output_dim = len(sample_labels) if isinstance(sample_labels, (list, tuple)) else (sample_labels.shape[0] if len(sample_labels.shape) > 0 else 1)
    
    # The model itself
    model = NeuralNet(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_units=e.NN_HIDDEN_LAYER_SIZES,
        learning_rate=e.NN_LEARNING_RATE_INIT,
        loss_function='mse' if e.DATASET_TYPE == 'regression' else 'bce',
    )
    
    # This callback will monitor the validation loss and restore the model weights 
    # to the state which achieved the best validation loss during any epoch of the 
    # training process.
    callback = BestModelRestorer(
        monitor='val_loss',
        mode='min',
    )
    trainer = pl.Trainer(
        max_epochs=200,
        accelerator='auto',
        devices=1,
        logger=False,
        # Disable Lightning's default ModelCheckpoint. It would write to a shared ./checkpoints/ dir with
        # epoch/step-based filenames that collide across concurrently-packed tasks (one task deletes another's
        # checkpoint -> FileNotFoundError). Best-weight restoration is handled in memory by BestModelRestorer.
        enable_checkpointing=False,
        callbacks=[callback],
        enable_progress_bar=e.__DEBUG__,
    )
    
    ## --- training the model ---
    # This method will perform the actual fitting of the model to the data.
    trainer.fit(
        model, 
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )
    # the device the network was trained on ('cpu' or 'cuda:0'), e.g. to tell GPU from CPU runs apart (ex_27)
    e['nn/device'] = str(trainer.strategy.root_device)
    # The model and the trainer reference each other and keep the data loaders alive until the garbage collector
    # runs; release the device copies of the data now (the next target of TARGET_INDICES or another run on the same
    # GPU needs the memory).
    for dataset in device_datasets:
        dataset.features = dataset.labels = None
    # important: After the training is done, we need to put the model into evaluation
    # mode to ensure it uses the running statistics for the batch norm.
    model.eval()

    return model
    

@experiment.hook('train_model__linear', replace=False, default=True)
def train_model__linear(e: Experiment,
                        index_data_map: dict,
                        train_indices: list[int],
                        val_indices: list
                        ) -> Any:
    
    X_train = np.array([index_data_map[i]['graph_features'] for i in train_indices])
    y_train = np.array([index_data_map[i]['graph_labels'] for i in train_indices])
    
    kwargs = {
        'l1_ratio': e.LN_L1_RATIO,
        'fit_intercept': e.LN_FIT_INTERCEPT,
        'random_state': e.SEED,
    }
    
    time_start = time.time()
    if e.DATASET_TYPE == 'classification':
        
        model = MultiOutputClassifier(LogisticRegression(
       
        ))
        model.fit(X_train, y_train)
        
    elif e.DATASET_TYPE == 'binary':
        
        y_train = np.argmax(y_train.astype(int), axis=-1)
        
        model = LogisticRegression(
            n_jobs=10,
        )
        model.fit(X_train, y_train)
    
    elif e.DATASET_TYPE == 'regression':
        
        model = LinearRegression()
        model = ElasticNet(
            alpha=e.LN_ALPHA,
            **kwargs,
        )
        model.fit(X_train, y_train)

    time_end = time.time()
    e['train_time/linear'] = time_end - time_start

    return model


@experiment.hook('train_model__support_vector', replace=False, default=True)
def train_model__support_vector(e: Experiment,
                                index_data_map: dict,
                                train_indices: list[int],
                                val_indices: list
                                ) -> Any:
    
    X_train = np.array([index_data_map[i]['graph_features'] for i in train_indices])
    y_train = np.array([index_data_map[i]['graph_labels'] for i in train_indices])
    
    kwargs = {
        'C': 1.0,
        'kernel': 'rbf',
        'max_iter': 250,
        #'probability': True,    
    }
    
    time_start = time.time()
    if e.DATASET_TYPE == 'classification':
        
        model = MultiOutputClassifier(SVC(**kwargs))
        model.fit(X_train, y_train)
    
    elif e.DATASET_TYPE == 'regression':
        
        model = SVR(**kwargs)
        model.fit(X_train, y_train)
    
    time_end = time.time()
    e['train_time/support_vector'] = time_end - time_start
    return model


@experiment.hook('predict_model', replace=False, default=True)
def predict_model(e: Experiment,
                  index_data_map: dict,
                  model: Any,
                  indices: list[int],
                  ) -> np.ndarray:
    X = np.array([index_data_map[i]['graph_features'] for i in indices])
    y_pred = model.predict(X)
    return y_pred


@experiment.hook('predict_model_proba', replace=False, default=True)
def predict_model_proba(e: Experiment,
                        index_data_map: dict,
                        model: Any,
                        indices: list[int],
                        y_pred: np.ndarray,
                        ) -> np.ndarray:
    
    X = np.array([index_data_map[i]['graph_features'] for i in indices])
    # Try to get probabilities if the model supports it
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X)
        
    # Fallback: use predicted labels as "probabilities"
    return y_pred.astype(float)

    
@experiment.hook('evaluate_model', replace=False, default=True)
def evaluate_model(e: Experiment,
                   index_data_map: dict,
                   indices: list[int],
                   model: Any,
                   key: str,
                   scaler: Any = None,
                   **kwargs,
                   ) -> None:

    y_eval = np.array([index_data_map[i]['graph_labels'] for i in indices])
    y_pred = e.apply_hook(
        'predict_model',
        index_data_map=index_data_map,
        model=model,
        indices=indices,
    )
    
    if e.DATASET_TYPE == 'classification' or e.DATASET_TYPE == 'binary':
        
        if e.DATASET_TYPE == 'binary':
            y_eval = np.argmax(y_eval.astype(int), axis=-1)
            
            labels_pred = y_pred
            labels_eval = y_eval
            
        else:
            labels_pred = np.array([np.argmax(y) for y in y_pred])
            labels_eval = np.array([np.argmax(y) for y in y_eval])
        
        proba_pred = e.apply_hook(
            'predict_model_proba',
            index_data_map=index_data_map,
            model=model,
            indices=indices,
            y_pred=y_pred,
        )
        
        # ~ simple metrics
        acc_value = accuracy_score(labels_eval, labels_pred)
        f1_value = f1_score(labels_eval, labels_pred, average='macro')
        ap_value = average_precision_score(labels_eval, labels_pred, average='macro')
        log_loss_value = log_loss(labels_eval, proba_pred)
        mse_value = mean_squared_error(y_eval, proba_pred)
        e[f'metrics/{key}/acc'] = acc_value
        e[f'metrics/{key}/f1'] = f1_value
        e[f'metrics/{key}/ap'] = ap_value
        e[f'metrics/{key}/log_loss'] = log_loss_value
        e[f'metrics/{key}/mse'] = mse_value

        # Conditionally calculate ROC AUC for binary classification
        auc_value = None
        if (y_eval.ndim == 2 and y_eval.shape[1] == 2):
            # Use the probability for the positive class
            if isinstance(proba_pred, list):
                proba_pred = np.array(proba_pred)
                
            # y_eval might be one-hot, so take the positive class column
            auc_value = roc_auc_score(y_eval[:, 1], proba_pred[:, 1])
            e[f'metrics/{key}/auc'] = auc_value
            
        if y_eval.ndim == 1:
            # If y_eval is 1D, we can directly use the predicted probabilities for the positive class
            auc_value = roc_auc_score(y_eval, proba_pred)

        log_msg = (
            f' * accuracy: {acc_value:.3f}'
            f' - f1 (macro): {f1_value:.3f}'
            f' - ap (macro): {ap_value:.3f}'
            f' - log_loss: {log_loss_value:.3f}'
            f' - mse: {mse_value:.3f}'
        )
        if auc_value is not None:
            log_msg += f' - auc: {auc_value:.3f}'
            
        e.log(log_msg)

        # ~ confusion matrix
        cm = confusion_matrix(labels_eval, labels_pred)
        fig, ax = plt.subplots(figsize=(8, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        title = (
            f'Confusion Matrix\n'
            f'Accuracy: {acc_value:.3f} - F1 (macro): {f1_value:.3f} - Average Precision (macro): {ap_value:.3f}'
        )
        if auc_value is not None:
            title += f' - AUC: {auc_value:.3f}'
        ax.set_title(title)
        e.commit_fig(f'{key}__confusion_matrix.png', fig)
        
    elif e.DATASET_TYPE == 'regression':
        
        if scaler:
            y_eval = scaler.inverse_transform(y_eval.reshape(-1, 1)).flatten()
            y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        
        # ~ simple metrics
        r2_value = r2_score(y_eval, y_pred)
        mse_value = mean_squared_error(y_eval, y_pred)
        mae_value = mean_absolute_error(y_eval, y_pred)
        e[f'metrics/{key}/r2'] = r2_value
        e[f'metrics/{key}/mse'] = mse_value
        e[f'metrics/{key}/mae'] = mae_value
        
        e.log(f' * r2: {r2_value:.3f}'
              f' - mse: {mse_value:.3f}'
              f' - mae: {mae_value:.3f}')
        
        # ~ plotting the regression plots
        fig, ax = plt.subplots(figsize=(10, 8))
        df = pd.DataFrame({
            'y_true': y_eval.flatten(),
            'y_pred': y_pred.flatten()
        })
        max_value = max(df['y_true'].max(), df['y_pred'].max())
        min_value = min(df['y_true'].min(), df['y_pred'].min())
        ax.plot([min_value, max_value], [min_value, max_value], color='black', linestyle='-', alpha=0.5)
        sns.histplot(
            df, 
            x='y_true', 
            y='y_pred', 
            ax=ax,
            bins=50,
            cbar=True,
            binrange=(min_value, max_value)
        )
        ax.set_title(f'Regression Plot\n'
                     f'R2: {r2_value:.3f} - MSE: {mse_value:.3f} - MAE: {mae_value:.3f}')
        ax.set_xlabel('True Value')
        ax.set_ylabel('Predicted Value')
        
        plt.tight_layout()
        e.commit_fig(f'{key}__regression_plots.png', fig)
        

@experiment
def experiment(e: Experiment):
    
    e.log('starting experiment to predict molecule dataset...')
    e.log_parameters()
    
    # --- data loading ---
    # First of all we need to load the dataset. Since this is a time consuming operation, we wrap this
    # as a cached operation so that it only has to be done once per dataset after which the result may just be
    # loaded from the disk.

    # The cached content is the *subsampled + filtered* dataset, whose molecule set depends on NUM_DATA
    # (the subsample fraction/count) and — only when actually subsampling — on SEED (which molecules are
    # drawn, see the load_dataset hook). The cache key must therefore reflect those parameters, otherwise a
    # subsampled run and a full-data run of the same dataset would collide on the key `load__{DATASET_NAME}`
    # and silently load each other's molecule set. For full-data runs (NUM_DATA is None or 1.0) the molecule
    # set is seed-independent, so we keep the original key to stay backward-compatible with existing caches.
    if e.NUM_DATA is None or e.NUM_DATA == 1.0:
        load_cache_name = f'load__{e.DATASET_NAME}'
    else:
        load_cache_name = f'load__{e.DATASET_NAME}__numdata_{e.NUM_DATA}__seed_{e.SEED}'

    @experiment.cache.cached(name=load_cache_name)
    def load_data():
        # This hook returns a dict whose keys are the unique integer indices of the dataset elements and the values 
        # are the corresponding graph dict representations.
        e.log(f'loading dataset "{e.DATASET_NAME}"...')
        index_data_map: dict[int, GraphDict]
        index_data_map, metadata = e.apply_hook(
            'load_dataset',
        )
        
        # :hook filter_dataset:
        #       An action hook that is called after the dataset has been loaded and before the dataset indices are 
        #       obtained, this optional hook presents the opportunity to filter the dataset based on certain criteria.
        e.apply_hook(
            'filter_dataset',
            index_data_map=index_data_map,
        )
        
        return index_data_map
        
    index_data_map: dict[int, GraphDict] = load_data()

    if e.TARGET_INDICES is not None and (e.TARGET_INDEX is not None or e.DATASET_TYPE != 'regression'):
        raise ValueError('TARGET_INDICES requires TARGET_INDEX=None and a regression dataset')

    e.log('determine the graph labels...')
    for index in list(index_data_map.keys()):

        graph = index_data_map[index]
        # :hook get_graph_labels:
        #       This hook is called on each graph in the dataset and is supposed to return the numpy array
        #       representing the graph labels to serve as the prediction target.
        graph_labels = e.apply_hook(
            'get_graph_labels',
            index=index,
            graph=graph
        )

        graph['graph_labels'] = graph_labels.astype(float)
        # multi-target runs: all targets are kept, and the per-target loop selects them one after the other
        if e.TARGET_INDICES is not None:
            graph['graph_labels_all'] = graph['graph_labels'].copy()
            
        if e.DATASET_NOISE > 0.0:
            
            if e.DATASET_TYPE == 'classification':
                if random.random() < e.DATASET_NOISE:
                    graph_labels = np.random.permutation(graph_labels)
                                        
            elif e.DATASET_TYPE == 'regression':
                noise = np.random.normal(0, e.DATASET_NOISE, graph_labels.shape)
                graph_labels += noise
    
    # --- dataset splitting ---
    # Now that we have loaded the dataset, we need to split it into training, validation and 
    # testing sets. We do this by first obtaining the list of all indices in the dataset
    # and then applying the dataset_split hook to obtain the actual splits which we then store 
    # into the experiment storage for later use.
    indices = list(index_data_map.keys())
    e.log(f'loaded dataset with {len(index_data_map)} elements...')
    
    e.log('creating train-val-test split...')
    train_indices, val_indices, test_indices = e.apply_hook(
        'dataset_split',
        indices=indices
    )
    e.log(f'train: {len(train_indices)}, val: {len(val_indices)}, test: {len(test_indices)}')
    e['indices/train'] = train_indices
    e['indices/val'] = val_indices
    e['indices/test'] = test_indices
    
    ## --- dataset processing ---
    # In this step we will process the dataset of molecular graph structures into some kind of 
    # fingerprint vector representation (e.g. Morgan fingerprint or HDC) so that they can then 
    # subsequently be used for the training of the models.
    
    e.log('processing dataset...')
    time_start = time.time()
    e.apply_hook(
        'process_dataset',
        index_data_map=index_data_map
    )
    time_end = time.time()
    duration = time_end - time_start
    e['process_time'] = duration
    e.log(f'processed dataset after {duration:.2f} seconds')
    
    ## --- exporting the converted dataset ---
    # Optionally export the featurized dataset as an NPZ artifact. This is guarded by SAVE_DATASET
    # (off by default) because materializing the full feature matrix (num_molecules x D) as a single
    # array is many GB for large datasets at high embedding dimension (e.g. ~4.4 GB for QM9 at D=8192).
    # Building it unconditionally roughly doubled per-task peak memory and OOM-killed tasks when many
    # were packed onto one node, so it now only runs when the artifact is actually requested.
    if e.SAVE_DATASET:
        indices = list(index_data_map.keys())
        smiles = [index_data_map[i]['graph_repr'] for i in indices]
        labels = np.array([index_data_map[i]['graph_labels'] for i in indices])
        features = np.array([index_data_map[i]['graph_features'] for i in indices])
        npz_path = os.path.join(e.path, 'dataset_converted.npz')
        np.savez_compressed(npz_path, indices=indices, smiles=smiles, labels=labels, features=features)
        e.log(f'💾 saved dataset as NPZ @ {npz_path}')
        
    # :hook after_dataset:
    #       An action hook that is called after the dataset has been loaded and processed. This hook
    #       presents the opportunity to perform additional processing on the dataset before training
    #       the models. It may also replace the labels (mixin_clogp sets the ClogP values here), which is why the
    #       labels are only standardized afterwards.
    e.apply_hook(
        'after_dataset',
        index_data_map=index_data_map,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
    )

    ## --- dataset scaling ---
    # For regression datasets we want to apply a standard scaler to the labels so it is easier for
    # the simple models to learn later on. Until 2026-10-13 the scaler was fitted before the after_dataset hook, so
    # for ClogP (mixin_clogp) it was fitted on the logS labels of AqSolDB while the models were trained on the raw
    # ClogP values, and the reported ClogP errors were the true ones times the standard deviation of logS in the
    # training split. "labels/scaled_after_hooks" marks the archives with the corrected order.

    def scale_train_evaluate(suffix: str = '') -> None:
        """
        Standardize the current labels (regression), then train and evaluate every model of MODELS. In a run with
        TARGET_INDICES this is called once per target, and the metric keys and the stored scaler carry the suffix
        "__t<target>".
        """
        scaler = None
        if e.DATASET_TYPE == 'regression':
            e.log('scaling the regression labels...')
            scaler = StandardScaler()
            y_train = np.array([index_data_map[i]['graph_labels'] for i in train_indices])
            e.log(f' * y_train shape: {y_train.shape}')
            scaler.fit(y_train)

            for index in index_data_map:
                index_data_map[index]['graph_labels'] = scaler.transform(index_data_map[index]['graph_labels'].reshape(1, -1)).flatten()

            scaler_path = os.path.join(e.path, f'standard_scaler{suffix}.joblib')
            joblib.dump(scaler, scaler_path)
            e.log(f'💾 saved standard scaler as joblib @ {scaler_path}')
            e[f'labels/scaler_mean{suffix}'] = scaler.mean_.tolist()
            e[f'labels/scaler_scale{suffix}'] = scaler.scale_.tolist()

        # ~ model training
        for model_name in e.MODELS:

            e.log(f'\ntraining model "{model_name}"...')
            time_start = time.time()
            model = e.apply_hook(
                f'train_model__{model_name}',
                index_data_map=index_data_map,
                train_indices=train_indices,
                val_indices=val_indices,
            )
            time_end = time.time()
            duration = time_end - time_start
            # Total wall time of the training hook, recorded uniformly for every model (some hooks record
            # their own "train_time/<model>", e.g. only up to the best epoch; that one is kept as is).
            e[f'fit_time/{model_name}{suffix}'] = duration
            e.log(f'training done after {duration:.2f} seconds')

            # ~ model evaluation
            e.log('evaluating model...')
            e.apply_hook(
                'evaluate_model',
                model=model,
                index_data_map=index_data_map,
                indices=test_indices,
                key=f'test_{model_name}{suffix}',
                scaler=scaler,
            )

            # ~ model evaluation on the held-out validation split
            # This validation metric is the selection signal for hyperparameter optimization (see
            # _slurm_ex_13 / analyze_ex_13): the best config per (representation, dataset) is chosen by the
            # validation metric, never the test metric, to avoid the optimistic bias of tuning on the test
            # split. The top-level validation split is otherwise unused by the neural-net models (which carve
            # their own internal validation set from the training indices), so it is a clean, held-out signal.
            if val_indices:
                e.apply_hook(
                    'evaluate_model',
                    model=model,
                    index_data_map=index_data_map,
                    indices=val_indices,
                    key=f'val_{model_name}{suffix}',
                    scaler=scaler,
                )

            # Release the device memory of the model (the torch KNN keeps its training matrix there) before the next
            # model or target: several runs share a GPU, and the caching allocator holds freed memory otherwise.
            del model
            if torch.cuda.is_available():
                gc.collect()
                torch.cuda.empty_cache()

    e['labels/scaled_after_hooks'] = True
    example_graph = index_data_map[train_indices[0]]
    e.log(f'example graph'
          f' - num_nodes: {len(example_graph["node_indices"])}'
          f' - num edges: {len(example_graph["edge_indices"])}'
          f' - embedding shape: {example_graph["graph_features"].shape}')

    if e.TARGET_INDICES is None:
        scale_train_evaluate()
    else:
        # a hook that set the labels itself (e.g. mixin_clogp) would be overwritten below
        if any(not np.array_equal(graph['graph_labels'], graph['graph_labels_all'], equal_nan=True)
               for graph in index_data_map.values()):
            raise ValueError('a hook changed the labels; TARGET_INDICES cannot be combined with it')
        for target in e.TARGET_INDICES:
            e.log(f'\n=== target {target} ===')
            for graph in index_data_map.values():
                graph['graph_labels'] = graph['graph_labels_all'][target:target + 1].copy()
            scale_train_evaluate(f'__t{target}')

    # peak device memory of the run, for packing several runs onto one GPU (ex_27)
    if torch.cuda.is_available():
        e['gpu/max_memory_allocated'] = torch.cuda.max_memory_allocated()
        e['gpu/max_memory_reserved'] = torch.cuda.max_memory_reserved()

    # ~ comparison of models
    # (none without models: MODELS=[] only featurizes the dataset, e.g. to write the HDF encoding cache in a warmup)
    if not e.MODELS:
        return

    e.log('creating model comparison plots...')
    keys = list(e['metrics'].keys())
    metrics = list(e['metrics'][keys[0]].keys())

    for metric in metrics:
        
        fig, ax = plt.subplots(figsize=(10, 8))
        values = [e['metrics'][key][metric] for key in keys]
        sns.barplot(x=keys, y=values, ax=ax)
        ax.set_title(f'Comparison of {metric}')
        ax.set_xlabel('Models')
        ax.set_ylabel(metric)
        ax.set_xticklabels(keys, rotation=45, ha='right')
        plt.tight_layout()
        e.commit_fig(f'comparison_{metric}.png', fig)
    

experiment.run_if_main()
