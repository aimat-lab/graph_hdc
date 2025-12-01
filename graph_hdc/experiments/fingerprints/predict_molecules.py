import os
import time
import copy
import random
from typing import Any, List, Union, Tuple

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

GB_NUM_ESTIMATORS: int = 100
GB_LEARNING_RATE: float = 0.01
GB_MAX_DEPTH: int = 3

LN_ALPHA: float = 1e-2
LN_L1_RATIO: float = 0.5
LN_FIT_INTERCEPT: bool = True

KN_NUM_NEIGHBORS: int = 5
KN_WEIGHTS: str = 'uniform'

NN_HIDDEN_LAYER_SIZES: Tuple[int] = (100, 100, 100)
NN_ALPHA: float = 0.0001
NN_LEARNING_RATE_INIT: float = 0.001

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
        'n_jobs': -1,
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
    
    # Create memory-efficient datasets
    train_dataset = LazyDataset(train_indices, index_data_map)
    
    # Create a DataLoader with optimized settings for performance
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=64, 
        shuffle=True,
        drop_last=True,
        num_workers=4,
        prefetch_factor=2,
        pin_memory=False,
    )
    
    # Create memory-efficient validation dataset
    val_dataset = LazyDataset(val_indices_, index_data_map)
    
    # Create a DataLoader for the validation dataset with optimized settings
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        prefetch_factor=2,
        pin_memory=False,
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
    
    @experiment.cache.cached(name=f'load__{e.DATASET_NAME}')
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
    e.log(f'processed dataset after {duration:.2f} seconds')
    
    ## --- exporting to CSV ---
    # Now we export the converted dataset into a CSV file and save that as an artifact
    
    # Collect data for DataFrame
    records = []
    for index, data in index_data_map.items():
        smiles: str = data['graph_repr']
        features: list = data['graph_features']
        labels: list = data['graph_labels']
        record = {
            'index': index,
            'smiles': smiles,
            'labels': labels,
            'features': features,
        }
        records.append(record)

    # Convert records to arrays for npz export
    indices = [record['index'] for record in records]
    smiles = [record['smiles'] for record in records]
    labels = np.array([record['labels'] for record in records])
    features = np.array([record['features'] for record in records])

    # Save dataset to NPZ file
    if e.SAVE_DATASET:
        npz_path = os.path.join(e.path, 'dataset_converted.npz')
        np.savez_compressed(npz_path, indices=indices, smiles=smiles, labels=labels, features=features)
        e.log(f'💾 saved dataset as NPZ @ {npz_path}')
        
    ## --- dataset scaling ---
    # For regression datasets we want to apply a standard scaler to the labels so it is easier for 
    # the simple models to learn later on.
    
    scaler = None
    if e.DATASET_TYPE == 'regression':
        e.log('scaling the regression labels...')
        scaler = StandardScaler()
        y_train = np.array([index_data_map[i]['graph_labels'] for i in train_indices])
        e.log(f' * y_train shape: {y_train.shape}')
        scaler.fit(y_train)

        for index in index_data_map:
            index_data_map[index]['graph_labels'] = scaler.transform(index_data_map[index]['graph_labels'].reshape(1, -1)).flatten()
    
        scaler_path = os.path.join(e.path, 'standard_scaler.joblib')
        joblib.dump(scaler, scaler_path)
        e.log(f'💾 saved standard scaler as joblib @ {scaler_path}')
    
    # :hook after_dataset:
    #       An action hook that is called after the dataset has been loaded and processed. This hook
    #       presents the opportunity to perform additional processing on the dataset before training
    #       the models.
    e.apply_hook(
        'after_dataset',
        index_data_map=index_data_map,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
    )
    
    example_graph = index_data_map[train_indices[0]]
    e.log(f'example graph'
          f' - num_nodes: {len(example_graph["node_indices"])}'
          f' - num edges: {len(example_graph["edge_indices"])}'
          f' - embedding shape: {example_graph["graph_features"].shape}')
    
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
        e.log(f'training done after {duration:.2f} seconds')
        
        # ~ model evaluation
        e.log('evaluating model...')
        e.apply_hook(
            'evaluate_model',
            model=model,
            index_data_map=index_data_map,
            indices=test_indices,
            key=f'test_{model_name}',
            scaler=scaler,
        )
        
    # ~ comparison of models
    
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
