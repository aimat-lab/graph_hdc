"""
Molecular Representation Optimization with Normalizing Flow Constraints

This is a simplified variant of optimize_molecule.py that uses a single neural
network (no ensemble) and adds normalizing flow constraints to keep optimization
in high-likelihood regions of the training distribution.

Key Workflow:
    1. Load and filter a molecular dataset from chem_mat_data
    2. Process molecules into representations (via hook - HDC, fingerprints, etc.)
    3. Extract target property values (via hook - default uses TARGET_INDEX)
    4. Train a single neural network on training split
    5. Train normalizing flow model on training representations
    6. On test split, perform gradient descent in original representation space
       to optimize towards target property values
    7. Evaluate by finding closest dataset element and comparing true property
    8. Visualize optimization trajectories

The optimization minimizes:
    loss = MSE(prediction, target) + λ_flow * (-log_prob) + λ_dist * distance

Gradient balancing is used to ensure no single loss term dominates: each gradient
is normalized by its L2 norm before being combined with the loss weights. This
prevents situations where one gradient has significantly larger magnitude than others.

Design Rationale:
    - Single model simplifies prediction (no ensemble uncertainty needed)
    - Normalizing flow provides distributional constraint via likelihood
    - Original space optimization (no PCA) maintains interpretability
    - Flow constraint prevents optimization from exploring unrealistic regions

Usage:
    Create configuration files extending this experiment and specifying the
    molecular representation method:

    .. code-block:: yaml

        extend: optimize_molecule_nf__hdc.py
        parameters:
            DATASET_NAME: "aqsoldb"
            TARGET_INDEX: 0
            TARGET_VALUE: 5.0
            FLOW_TYPE: "maf"
            FLOW_WEIGHT: 1.0
            OPTIMIZATION_EPOCHS: 200

Output Artifacts:
    - flow_training_history.png: Training history of normalizing flow (loss and log prob)
    - optimization_trajectory_{idx}.png: Individual optimization trajectories
    - optimization_summary.png: Summary statistics across all optimizations
    - optimization_results.csv: Detailed results for each optimization
    - flow_model.pth: Trained normalizing flow model
"""
import os
import time
import copy
import random
from typing import Any, List, Union, Tuple, Optional

import optuna
import joblib
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LinearLR
import pytorch_lightning as pl
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from rich.pretty import pprint
from rdkit import Chem
from rdkit.Chem import Draw, AllChem, DataStructs

from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace
from chem_mat_data._typing import GraphDict
from chem_mat_data.main import load_graph_dataset

# == DATASET PARAMETERS ==

# :param DATASET_NAME:
#       The name of the dataset to be used for the experiment. This name is used
#       to download the dataset from the ChemMatData file share.
DATASET_NAME: str = 'aqsoldb'

# :param DATASET_NAME_ID:
#       The name of the dataset to be used for identification purposes.
DATASET_NAME_ID: str = DATASET_NAME

# :param TARGET_INDEX:
#       The index of the target property in graph_labels. This parameter is used
#       to extract the property value that will be optimized. If None, the full
#       list of targets is used (for multi-target datasets).
TARGET_INDEX: Union[int, None] = 0

# :param NUM_DATA:
#       The number of samples to be used for the experiment. This parameter can be
#       either an integer or a float between 0 and 1. If None, the entire dataset
#       is used.
NUM_DATA: Union[int, float, None] = None

# :param NUM_TEST:
#       The number of test samples to be used for optimization experiments.
NUM_TEST: Union[int, float] = 0.4

# :param NUM_TRAIN:
#       The number of training samples to be used for model training.
NUM_TRAIN: Union[int, float] = 1.0

# :param NUM_VAL:
#       The number of validation samples used during model training.
NUM_VAL: Union[int, float] = 0.1

# :param SEED:
#       The random seed to be used for the experiment. If None, random processes
#       will not be seeded, resulting in different outcomes across repetitions.
SEED: Union[int, None] = 2

# == MODEL PARAMETERS ==

# :param HIDDEN_UNITS:
#       The architecture of the neural network, specified as a tuple of hidden layer sizes.
HIDDEN_UNITS: Tuple[int, ...] = (256, 256, 128)

# :param LEARNING_RATE:
#       The learning rate for training the neural network model.
LEARNING_RATE: float = 1e-3

# :param MAX_EPOCHS:
#       The maximum number of epochs to train the model.
MAX_EPOCHS: int = 100

# :param BATCH_SIZE:
#       The batch size for training the neural network model.
BATCH_SIZE: int = 256

# :param USE_BEST_MODEL_RESTORER:
#       Whether to use the BestModelRestorer callback during training. When enabled,
#       the model weights will be restored to the epoch with the best validation loss
#       after training completes. When disabled, the final epoch weights are used.
USE_BEST_MODEL_RESTORER: bool = True

# == OPTIMIZATION PARAMETERS ==

# :param TARGET_VALUE:
#       The target property value to optimize towards. If None, the median of the
#       test set will be used as the target.
TARGET_VALUE: Union[float, None] = None

# :param TARGET_RELATIVE:
#       Whether to interpret TARGET_VALUE as a relative offset.
#       - If False (default): TARGET_VALUE is an absolute target property value
#       - If True: TARGET_VALUE is added to each sample's initial property to
#         determine the target. For example, TARGET_VALUE=1.0 means "increase
#         property by 1.0 from initial value", TARGET_VALUE=-0.5 means "decrease
#         property by 0.5 from initial value".
#       When True, TARGET_VALUE cannot be None.
TARGET_RELATIVE: bool = False

# :param NUM_OPTIMIZATION_SAMPLES:
#       The number of test samples to optimize. If None, all test samples are used.
NUM_OPTIMIZATION_SAMPLES: Union[int, None] = 10

# :param OPTIMIZATION_EPOCHS:
#       The number of gradient descent epochs to perform for each optimization.
OPTIMIZATION_EPOCHS: int = 250

# :param OPTIMIZATION_LEARNING_RATE:
#       The learning rate for gradient descent in representation space.
OPTIMIZATION_LEARNING_RATE: float = 0.001

# :param OPTIMIZATION_LEARNING_RATE_REDUCTION:
#       The fraction of the initial learning rate to reach by the end of optimization.
#       The learning rate is linearly scheduled from the initial value to
#       (initial_lr * OPTIMIZATION_LEARNING_RATE_REDUCTION) over OPTIMIZATION_EPOCHS.
#       For example, 0.1 means the LR will decay to 10% of its initial value.
OPTIMIZATION_LEARNING_RATE_REDUCTION: float = 0.1

# :param ORIGINAL_DISTANCE_WEIGHT:
#       Weight for the original distance regularization term in optimization loss.
#       loss = MSE + λ_flow * (-log_prob) + λ_orig * distance(rep_optimized, rep_initial)
#       This term influences how the optimized representation relates to the initial one:
#       - Positive values (e.g., 0.1): Keep representation close to initial (exploration constraint)
#       - Zero (0.0): No constraint on distance from initial
#       - Negative values (e.g., -0.1): Actively push away from initial (encourage diversity)
ORIGINAL_DISTANCE_WEIGHT: float = 0.1

# :param DISTANCE_METRIC:
#       Distance metric to use for original distance regularization.
#       Options: "cosine", "manhattan", "euclidean"
#       - "cosine": 1 - cosine_similarity (direction-based similarity)
#       - "manhattan": L1 distance (sum of absolute differences)
#       - "euclidean": L2 distance (standard geometric distance)
DISTANCE_METRIC: str = "cosine"

# :param ENABLE_GRADIENT_CLIPPING:
#       Whether to enable gradient clipping during optimization. This can help
#       prevent the optimization from diverging.
ENABLE_GRADIENT_CLIPPING: bool = False

# :param GRADIENT_CLIP_VALUE:
#       The maximum gradient norm for gradient clipping.
GRADIENT_CLIP_VALUE: float = 0.5

# :param TRAJECTORY_SAMPLING_INTERVAL:
#       The interval (in epochs) at which to sample the optimization trajectory.
#       For each sampled point, we find the closest test molecule to track how
#       the optimization moves through the space of real molecules. A value of 10
#       means we sample every 10th epoch during optimization.
TRAJECTORY_SAMPLING_INTERVAL: int = 2

# == NORMALIZING FLOW PARAMETERS ==

# :param USE_FLOW_CONSTRAINT:
#       Whether to use normalizing flow likelihood as an optimization constraint.
#       When enabled, the optimization loss includes a negative log-likelihood term
#       that encourages the representation to stay in high-likelihood regions of
#       the flow distribution fitted on training data.
USE_FLOW_CONSTRAINT: bool = True

# :param FLOW_WEIGHT:
#       Weight (λ_flow) for the normalizing flow negative log-likelihood term in the
#       optimization loss. Higher values more strongly constrain optimization to stay
#       near the training distribution. The loss term is: λ_flow * (-log_prob)
FLOW_WEIGHT: float = 1.0

# :param FLOW_TYPE:
#       Type of normalizing flow architecture to use.
#       Options: 'maf' (Masked Autoregressive Flow), 'realnvp' (Real NVP),
#       'nsf' (Neural Spline Flow), 'residual' (Residual Flow)
FLOW_TYPE: str = 'maf'

# :param FLOW_NUM_LAYERS:
#       Number of flow transformation layers to stack. More layers increase
#       expressiveness but also training time and complexity.
FLOW_NUM_LAYERS: int = 8

# :param FLOW_HIDDEN_UNITS:
#       Number of hidden units in the flow transformation networks.
FLOW_HIDDEN_UNITS: int = 256

# :param FLOW_HIDDEN_LAYERS:
#       Number of hidden layers in each flow transformation network.
FLOW_HIDDEN_LAYERS: int = 2

# :param FLOW_LEARNING_RATE:
#       Learning rate for training the normalizing flow model.
FLOW_LEARNING_RATE: float = 1e-3

# :param FLOW_EPOCHS:
#       Number of epochs to train the normalizing flow model.
FLOW_EPOCHS: int = 100

# :param FLOW_BATCH_SIZE:
#       Batch size for training the normalizing flow model.
FLOW_BATCH_SIZE: int = 256

# :param FLOW_BASE_DIST:
#       Base distribution for the normalizing flow.
#       Options: 'gaussian' (diagonal Gaussian), 'uniform' (uniform Gaussian)
FLOW_BASE_DIST: str = 'gaussian'

# :param USE_NORM_PROJECTION:
#       Whether to explicitly project representations back to unit norm after each
#       optimization step. If False, relies purely on the flow model's implicit
#       regularization to keep representations near the training distribution.
#
#       - True: Hard constraint via projection (guarantees norm=1.0)
#       - False: Soft constraint via flow loss (encourages norm≈1.0)
#
#       For HDC embeddings which are inherently normalized, True is recommended unless
#       FLOW_WEIGHT is sufficiently large (≥1.0) to provide strong implicit regularization.
USE_NORM_PROJECTION: bool = True

# == VISUALIZATION PARAMETERS ==

# :param PLOT_INDIVIDUAL_TRAJECTORIES:
#       Whether to create individual plots for each optimization trajectory.
PLOT_INDIVIDUAL_TRAJECTORIES: bool = True

# :param TRAJECTORY_FIGURE_SIZE:
#       The figure size for trajectory plots.
#       Size adjusted for 4 subplots (distance, flow loss, distance loss, gradient norms).
TRAJECTORY_FIGURE_SIZE: Tuple[int, int] = (28, 6)

# :param PCA_COMPONENTS:
#       Number of PCA components to use for trajectory visualizations.
#       This parameter controls the dimensionality reduction for plotting
#       optimization trajectories in 2D space.
PCA_COMPONENTS: int = 2

# :param PCA_FIGURE_SIZE:
#       Figure size for PCA trajectory visualizations.
PCA_FIGURE_SIZE: Tuple[int, int] = (12, 10)

# == EXPERIMENT PARAMETERS ==

# :param NOTE:
#       A note that can be used to describe the experiment.
NOTE: str = ''

__DEBUG__: bool = True
__NOTIFY__: bool = False
__CACHING__: bool = True

experiment = Experiment(
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals()
)


# == UTILITY CLASSES ==


class NeuralNet(pl.LightningModule):
    """
    A simple feedforward neural network for regression tasks.

    This network is designed to predict a single continuous property from a
    fixed-size molecular representation. It includes batch normalization and
    dropout for regularization.

    This network serves as a single ensemble member when used within a
    ModelEnsemble.
    """

    def __init__(self,
                 input_dim: int,
                 output_dim: int = 1,
                 hidden_units: Tuple[int, ...] = (256, 256, 128),
                 learning_rate: float = 1e-3,
                 dropout_rate: float = 0.0,
                 loss_type: str = 'mse',
                 ) -> None:
        """
        Initialize the neural network.

        :param input_dim: The dimensionality of the input representation.
        :param output_dim: The dimensionality of the output (typically 1 for
            single property prediction).
        :param hidden_units: Tuple specifying the number of units in each hidden
            layer.
        :param learning_rate: The learning rate for optimization.
        :param dropout_rate: The dropout probability for regularization (default: 0.0).
        :param loss_type: The loss function to use for training. Options:
            - 'mse': Mean Squared Error (default)
            - 'mae': Mean Absolute Error (L1 loss)
            - 'huber': Huber loss (combines MSE and MAE properties)
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.loss_type = loss_type

        # Build the network architecture
        self.layers = nn.ModuleList()
        prev_units = self.input_dim

        for units in self.hidden_units:
            self.layers.append(nn.Sequential(
                nn.Linear(prev_units, units),
                nn.BatchNorm1d(units),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate),
            ))
            prev_units = units

        # Output layer
        self.layers.append(nn.Linear(prev_units, self.output_dim))

        # Loss function - select based on loss_type
        if self.loss_type == 'mse':
            self.criterion = nn.MSELoss()
        elif self.loss_type == 'mae':
            self.criterion = nn.L1Loss()
        elif self.loss_type == 'huber':
            self.criterion = nn.HuberLoss()
        else:
            raise ValueError(
                f"Unknown loss_type: '{self.loss_type}'. "
                f"Supported types: 'mse', 'mae', 'huber'"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        :param x: Input tensor of shape (batch_size, input_dim).

        :return: Output tensor of shape (batch_size, output_dim).
        """
        for layer in self.layers:
            x = layer(x)
        return x

    def training_step(self,
                      batch: Tuple[torch.Tensor, torch.Tensor],
                      batch_idx: int
                      ) -> torch.Tensor:
        """
        Training step for PyTorch Lightning.

        :param batch: Tuple of (features, labels).
        :param batch_idx: The index of the current batch.

        :return: The loss value for this batch.
        """
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self,
                        batch: Tuple[torch.Tensor, torch.Tensor],
                        batch_idx: int
                        ) -> torch.Tensor:
        """
        Validation step for PyTorch Lightning.

        :param batch: Tuple of (features, labels).
        :param batch_idx: The index of the current batch.

        :return: The validation loss value.
        """
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configure the optimizer for training.

        :return: The optimizer instance.
        """
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-5,
        )
        return optimizer

    def predict_with_grad(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make predictions while preserving gradients (for optimization).

        This method is used during representation optimization to allow gradients
        to flow back through the network to the input representation.

        :param x: Input tensor of shape (batch_size, input_dim).

        :return: Output tensor of shape (batch_size, output_dim).
        """
        return self.forward(x)


class BestModelRestorer(pl.Callback):
    """
    PyTorch Lightning callback to restore the best model weights after training.

    This callback monitors a validation metric and saves the model state whenever
    the metric improves. At the end of training, the model is restored to the
    state with the best observed metric.
    """

    def __init__(self,
                 monitor: str = "val_loss",
                 mode: str = "min"
                 ) -> None:
        """
        Initialize the callback.

        :param monitor: The name of the metric to monitor.
        :param mode: Either 'min' (lower is better) or 'max' (higher is better).
        """
        super().__init__()
        self.monitor = monitor
        if mode not in ["min", "max"]:
            raise ValueError("mode must be 'min' or 'max'.")
        self.mode = mode

        self.best_score: float = None
        self.best_state_dict = None

    def on_fit_start(self, trainer, pl_module):
        """Initialize the best score before training starts."""
        if self.mode == "min":
            self.best_score = float("inf")
        else:
            self.best_score = -float("inf")
        self.best_state_dict = None

    def on_validation_end(self, trainer, pl_module):
        """Check if the current model is the best seen so far."""
        metrics = trainer.callback_metrics
        current_score = metrics.get(self.monitor)

        if current_score is None:
            return

        if (
            (self.mode == "min" and current_score < self.best_score) or
            (self.mode == "max" and current_score > self.best_score)
        ):
            self.best_score = current_score
            self.best_state_dict = {
                k: copy.deepcopy(v.detach().cpu().clone())
                for k, v in pl_module.state_dict().items()
            }

    def on_fit_end(self, trainer, pl_module):
        """Restore the best model weights at the end of training."""
        if self.best_state_dict is not None:
            pl_module.load_state_dict(self.best_state_dict)


# == HYPEROPT SETUP ==
# The following hooks configure hyperparameter optimization using Optuna.

__OPTUNA_REPETITIONS__: int = 3

@experiment.hook('__optuna_parameters__')
def define_search_space(e: Experiment, trial) -> dict:
    """Define which parameters to optimize and their ranges."""
    return {
        # 'LEARNING_RATE': trial.suggest_float('LEARNING_RATE', 1e-4, 1e-1, log=True),
        'OPTIMIZATION_LEARNING_RATE': trial.suggest_float(
            'OPTIMIZATION_LEARNING_RATE', 
            1e-4, 1e-1, log=True
        ),
        # 'ORIGINAL_DISTANCE_WEIGHT': trial.suggest_float(
        #     'ORIGINAL_DISTANCE_WEIGHT',
        #     0.0, 1.0
        # ),
        # 'DISTANCE_METRIC': trial.suggest_categorical(
        #     'DISTANCE_METRIC',
        #     ['cosine', 'manhattan', 'euclidean']
        # ),
        'OPTIMIZATION_EPOCHS': trial.suggest_int(
            'OPTIMIZATION_EPOCHS',
            100, 500
        ),
    }

@experiment.hook('__optuna_objective__')
def extract_objective(e: Experiment) -> float:
    """Extract the metric to optimize (maximize by default)."""
    return e['summary/improvement_true_pct']

@experiment.hook('__optuna_direction__')
def optimization_direction(e: Experiment) -> str:
    return 'maximize'  # Default is 'maximize'

@experiment.hook('__optuna_sampler__', replace=True)
def configure_sampler(e: Experiment):
    return optuna.samplers.TPESampler(
        n_startup_trials=10,
        constant_liar=True,
        multivariate=True,
    )


# == HOOKS ==
# Defining hooks that can be reused throughout the experiment and overwritten by 
# subsequent sub-experiments.

@experiment.hook('load_dataset', replace=False, default=True)
def load_dataset(e: Experiment) -> dict[int, GraphDict]:
    """
    Load the molecular dataset from ChemMatData.

    This hook downloads and loads the dataset, creating a dictionary mapping
    integer indices to graph dictionaries representing molecules.

    :param e: The experiment instance.

    :return: Dictionary mapping indices to graph dictionaries.
    """
    e.log(f'loading dataset "{e.DATASET_NAME}"...')

    graphs: List[GraphDict] = load_graph_dataset(
        e.DATASET_NAME,
        folder_path='/tmp'
    )

    index_data_map = dict(enumerate(graphs))
    e.log(f'loaded {len(index_data_map)} molecules from dataset')

    # Optional subsampling
    if e.NUM_DATA is not None:
        if isinstance(e.NUM_DATA, int):
            num_data = e.NUM_DATA
        elif isinstance(e.NUM_DATA, float):
            num_data = int(e.NUM_DATA * len(index_data_map))

        random.seed(e.SEED)
        index_data_map = dict(
            random.sample(
                list(index_data_map.items()),
                k=num_data
            )
        )
        e.log(f'subsampled to {len(index_data_map)} molecules')

    return index_data_map


@experiment.hook('filter_dataset', replace=False, default=True)
def filter_dataset(e: Experiment,
                   index_data_map: dict[int, GraphDict],
                   ) -> None:
    """
    Filter the dataset to remove invalid SMILES and unconnected graphs.

    :param e: The experiment instance.
    :param index_data_map: Dictionary to be filtered in-place.

    :return: None. Modifies index_data_map in-place.
    """
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

        # Disconnected graphs
        if '.' in smiles:
            del index_data_map[index]
            continue

    e.log(f'finished filtering with {len(index_data_map)} samples remaining')


@experiment.hook('process_dataset', replace=False, default=True)
def process_dataset(e: Experiment,
                    index_data_map: dict[int, GraphDict]
                    ) -> None:
    """
    Process the dataset into molecular representations.

    **IMPORTANT:** This hook must be overridden in extending experiments to
    provide the actual molecular encoding method (HDC, fingerprints, etc.).

    The hook should add a 'graph_features' key to each graph dictionary containing
    a numpy array of the molecular representation.

    :param e: The experiment instance.
    :param index_data_map: Dictionary to be modified in-place with 'graph_features'.

    :return: None. Modifies index_data_map in-place.

    :raises NotImplementedError: This default implementation raises an error.
    """
    raise NotImplementedError(
        "The 'process_dataset' hook must be overridden to provide a molecular "
        "encoding method. Please extend this experiment with a concrete "
        "representation implementation (e.g., optimize_molecule__hdc.py)."
    )


@experiment.hook('extract_property', replace=False, default=True)
def extract_property(e: Experiment,
                     index: int,
                     graph: GraphDict
                     ) -> float:
    """
    Extract the target property value from a graph dictionary.

    The default implementation uses the TARGET_INDEX parameter to extract a
    specific value from the graph_labels array. This can be overridden to
    compute properties on-the-fly (e.g., calculated descriptors).

    As a safety mechanism, if the graph already contains a 'property_value' key
    (e.g., set by a mixin), that value will be used directly instead of extracting
    from graph_labels. This provides an alternative injection point for mixins.

    :param e: The experiment instance.
    :param index: The index of the molecule in the dataset.
    :param graph: The graph dictionary containing molecular information.

    :return: The property value as a float.
    """
    # Safety check: If property_value is already set (e.g., by a mixin),
    # use it directly without extracting from graph_labels
    if 'property_value' in graph:
        return float(graph['property_value'])

    if 'graph_labels' not in graph:
        raise KeyError(
            f"Graph at index {index} does not contain 'graph_labels'. "
            f"Ensure the dataset includes target properties."
        )

    labels = graph['graph_labels']

    if e.TARGET_INDEX is None:
        # Use first target by default
        if len(labels) == 0:
            raise ValueError(f"Graph at index {index} has empty graph_labels.")
        return float(labels[0])
    else:
        # Use specified target index
        if e.TARGET_INDEX >= len(labels):
            raise IndexError(
                f"TARGET_INDEX={e.TARGET_INDEX} is out of bounds for "
                f"graph_labels with length {len(labels)}."
            )
        return float(labels[e.TARGET_INDEX])


@experiment.hook('dataset_split', replace=False, default=True)
def dataset_split(e: Experiment,
                  indices: List[int],
                  ) -> Tuple[List[int], List[int], List[int]]:
    """
    Split the dataset into train, validation, and test sets.

    :param e: The experiment instance.
    :param indices: List of all dataset indices.

    :return: Tuple of (train_indices, val_indices, test_indices).
    """
    random.seed(e.SEED)

    # Determine test set size
    if isinstance(e.NUM_TEST, int):
        num_test = e.NUM_TEST
    elif isinstance(e.NUM_TEST, float):
        num_test = int(e.NUM_TEST * len(indices))

    test_indices = random.sample(indices, k=num_test)
    indices = list(set(indices) - set(test_indices))

    # Determine validation set size
    if isinstance(e.NUM_VAL, int):
        num_val = e.NUM_VAL
    elif isinstance(e.NUM_VAL, float):
        num_val = int(e.NUM_VAL * len(indices))

    val_indices = random.sample(indices, k=num_val)
    indices = list(set(indices) - set(val_indices))

    # Determine training set size
    if isinstance(e.NUM_TRAIN, int):
        num_train = e.NUM_TRAIN
    elif isinstance(e.NUM_TRAIN, float):
        num_train = int(e.NUM_TRAIN * len(indices))

    num_train = max(num_train, 10)  # Ensure minimum training samples
    train_indices = random.sample(indices, k=min(num_train, len(indices)))

    return train_indices, val_indices, test_indices


@experiment.hook('train_model', replace=False, default=True)
def train_model(e: Experiment,
                index_data_map: dict[int, GraphDict],
                train_indices: List[int],
                val_indices: List[int],
                ) -> NeuralNet:
    """
    Train a single neural network for property prediction.

    This is a simplified version that trains one model instead of an ensemble.
    No bootstrapping is used - the model is trained on the full training set.

    :param e: The experiment instance.
    :param index_data_map: Dictionary of graph data.
    :param train_indices: Indices of training samples.
    :param val_indices: Indices of validation samples.

    :return: Trained NeuralNet model.
    """
    e.log(f'\ntraining prediction model...')
    e.log(f' * train samples: {len(train_indices)}')
    e.log(f' * val samples: {len(val_indices)}')

    # Determine dimensions
    sample_graph = index_data_map[train_indices[0]]
    input_dim = len(sample_graph['graph_features'])
    output_dim = 1

    e.log(f' * input_dim: {input_dim}')
    e.log(f' * architecture: {e.HIDDEN_UNITS}')

    # Prepare training data
    train_features = np.array([index_data_map[i]['graph_features'] for i in train_indices])
    train_labels = np.array([[e.apply_hook('extract_property', index=i, graph=index_data_map[i])]
                            for i in train_indices])

    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(train_features, dtype=torch.float32),
        torch.tensor(train_labels, dtype=torch.float32)
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=e.BATCH_SIZE,
        shuffle=True,
    )

    # Prepare validation data
    val_features = np.array([index_data_map[i]['graph_features'] for i in val_indices])
    val_labels = np.array([[e.apply_hook('extract_property', index=i, graph=index_data_map[i])]
                          for i in val_indices])

    val_dataset = torch.utils.data.TensorDataset(
        torch.tensor(val_features, dtype=torch.float32),
        torch.tensor(val_labels, dtype=torch.float32)
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=e.BATCH_SIZE,
        shuffle=False
    )

    # Create model
    model = NeuralNet(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_units=e.HIDDEN_UNITS,
        learning_rate=e.LEARNING_RATE,
    )

    # Callbacks
    callbacks = []
    if e.USE_BEST_MODEL_RESTORER:
        callback = BestModelRestorer(monitor='val_loss', mode='min')
        callbacks.append(callback)

    # Train
    trainer = pl.Trainer(
        max_epochs=e.MAX_EPOCHS,
        accelerator='auto',
        devices=1,
        logger=False,
        callbacks=callbacks,
        enable_progress_bar=e.__DEBUG__,
        enable_checkpointing=False,
    )

    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    model.eval()

    if e.USE_BEST_MODEL_RESTORER:
        e.log(f' * training complete, best val_loss: {callbacks[0].best_score:.4f}')
    else:
        e.log(f' * training complete')

    return model


@experiment.hook('train_flow', replace=False, default=True)
def train_flow(e: Experiment,
               train_features: np.ndarray,
               ) -> Optional['NormalizingFlow']:
    """
    Train normalizing flow model on training distribution (original space).

    The flow learns the density of molecular representations in their original
    high-dimensional space. During optimization, the flow's log probability
    provides a constraint to keep optimized representations in high-likelihood
    regions of the training distribution.

    :param e: The experiment instance.
    :param train_features: Training features in original representation space.

    :return: Trained NormalizingFlow model or None if disabled.
    """
    if not e.USE_FLOW_CONSTRAINT:
        e.log('flow constraint disabled (USE_FLOW_CONSTRAINT=False)')
        return None

    e.log(f'\n=== TRAINING NORMALIZING FLOW ===')
    e.log(f' * flow type: {e.FLOW_TYPE}')
    e.log(f' * input dimension: {train_features.shape[1]} (original space)')
    e.log(f' * num layers: {e.FLOW_NUM_LAYERS}')
    e.log(f' * hidden units: {e.FLOW_HIDDEN_UNITS}')
    e.log(f' * hidden layers: {e.FLOW_HIDDEN_LAYERS}')
    e.log(f' * base distribution: {e.FLOW_BASE_DIST}')

    # Diagnostic: Check data properties
    e.log(f'\ndata statistics:')
    e.log(f' * shape: {train_features.shape}')
    e.log(f' * mean: {train_features.mean():.6f}')
    e.log(f' * std: {train_features.std():.6f}')
    e.log(f' * min: {train_features.min():.6f}')
    e.log(f' * max: {train_features.max():.6f}')
    e.log(f' * contains NaN: {np.isnan(train_features).any()}')
    e.log(f' * contains Inf: {np.isinf(train_features).any()}')

    # Check norms
    norms = np.linalg.norm(train_features, axis=1)
    e.log(f' * sample norms - mean: {norms.mean():.6f}, std: {norms.std():.6f}, min: {norms.min():.6f}, max: {norms.max():.6f}')

    from graph_hdc.models.flow import NormalizingFlow

    # Create flow model
    flow_model = NormalizingFlow(
        dim=train_features.shape[1],
        flow_type=e.FLOW_TYPE,
        num_layers=e.FLOW_NUM_LAYERS,
        hidden_units=e.FLOW_HIDDEN_UNITS,
        hidden_layers=e.FLOW_HIDDEN_LAYERS,
        learning_rate=e.FLOW_LEARNING_RATE,
        base_dist=e.FLOW_BASE_DIST,
    )

    # Prepare dataset
    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(train_features, dtype=torch.float32)
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=e.FLOW_BATCH_SIZE,
        shuffle=True,
    )

    # Create logger to capture training history
    from pytorch_lightning.loggers import CSVLogger
    logger = CSVLogger(
        save_dir=e.path,
        name='flow_training',
        version='',
    )

    # Train with gradient clipping for numerical stability
    trainer = pl.Trainer(
        max_epochs=e.FLOW_EPOCHS,
        accelerator='auto',
        devices=1,
        logger=logger,
        enable_progress_bar=e.__DEBUG__,
        enable_checkpointing=False,
        gradient_clip_val=1.0,  # Add gradient clipping
        gradient_clip_algorithm='norm',
    )

    e.log(f' * training for {e.FLOW_EPOCHS} epochs...')
    trainer.fit(flow_model, train_dataloaders=train_loader)
    flow_model.eval()

    # Evaluate flow on training data
    with torch.no_grad():
        train_tensor = torch.tensor(train_features, dtype=torch.float32)
        train_log_probs = flow_model.log_prob(train_tensor)
        mean_log_prob = train_log_probs.mean().item()
        std_log_prob = train_log_probs.std().item()

    e.log(f' * training complete')
    e.log(f' * train set log probability: {mean_log_prob:.3f} ± {std_log_prob:.3f}')

    e['flow/train_mean_log_prob'] = float(mean_log_prob)
    e['flow/train_std_log_prob'] = float(std_log_prob)

    # Create visualization of training history
    e.log(' * creating training history visualization...')
    metrics_file = os.path.join(e.path, 'flow_training', 'metrics.csv')

    if os.path.exists(metrics_file):
        # Read logged metrics
        metrics_df = pd.read_csv(metrics_file)

        # Create figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Normalizing Flow Training History', fontsize=16, fontweight='bold')

        # Plot 1: Training loss (step-level)
        if 'train_loss_step' in metrics_df.columns:
            ax = axes[0, 0]
            step_data = metrics_df[['step', 'train_loss_step']].dropna()
            ax.plot(step_data['step'], step_data['train_loss_step'],
                   alpha=0.6, linewidth=0.5, color='steelblue')
            ax.set_xlabel('Step')
            ax.set_ylabel('Training Loss')
            ax.set_title('Training Loss (per step)')
            ax.grid(True, alpha=0.3)

        # Plot 2: Training loss (epoch-level)
        if 'train_loss_epoch' in metrics_df.columns:
            ax = axes[0, 1]
            epoch_data = metrics_df[['epoch', 'train_loss_epoch']].dropna()
            ax.plot(epoch_data['epoch'], epoch_data['train_loss_epoch'],
                   marker='o', linewidth=2, markersize=4, color='darkblue')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Training Loss')
            ax.set_title('Training Loss (per epoch)')
            ax.grid(True, alpha=0.3)

        # Plot 3: Log probability (step-level)
        if 'train_log_prob_step' in metrics_df.columns:
            ax = axes[1, 0]
            step_data = metrics_df[['step', 'train_log_prob_step']].dropna()
            ax.plot(step_data['step'], step_data['train_log_prob_step'],
                   alpha=0.6, linewidth=0.5, color='coral')
            ax.set_xlabel('Step')
            ax.set_ylabel('Log Probability')
            ax.set_title('Log Probability (per step)')
            ax.grid(True, alpha=0.3)

        # Plot 4: Log probability (epoch-level)
        if 'train_log_prob_epoch' in metrics_df.columns:
            ax = axes[1, 1]
            epoch_data = metrics_df[['epoch', 'train_log_prob_epoch']].dropna()
            ax.plot(epoch_data['epoch'], epoch_data['train_log_prob_epoch'],
                   marker='o', linewidth=2, markersize=4, color='darkorange')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Log Probability')
            ax.set_title('Log Probability (per epoch)')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save the figure
        output_path = os.path.join(e.path, 'flow_training_history.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        e.log(f' * saved training history visualization to: {os.path.basename(output_path)}')

        # Log final epoch metrics to experiment
        if 'train_loss_epoch' in metrics_df.columns:
            final_epoch_data = metrics_df[['epoch', 'train_loss_epoch']].dropna()
            if len(final_epoch_data) > 0:
                e['flow/final_train_loss'] = float(final_epoch_data.iloc[-1]['train_loss_epoch'])

        if 'train_log_prob_epoch' in metrics_df.columns:
            final_epoch_data = metrics_df[['epoch', 'train_log_prob_epoch']].dropna()
            if len(final_epoch_data) > 0:
                e['flow/final_train_log_prob'] = float(final_epoch_data.iloc[-1]['train_log_prob_epoch'])
    else:
        e.log(f' * warning: metrics file not found at {metrics_file}')

    return flow_model


def compute_distance(a: torch.Tensor, b: torch.Tensor, metric: str) -> torch.Tensor:
    """
    Compute distance between two tensors using specified metric.

    This function calculates the distance between two representation vectors
    using one of several supported distance metrics. All computations are
    differentiable to support gradient-based optimization.

    :param a: First tensor (representation vector).
    :param b: Second tensor (representation vector).
    :param metric: Distance metric to use. Options:
        - "cosine": Cosine distance (1 - cosine_similarity)
        - "manhattan": Manhattan distance (L1 norm)
        - "euclidean": Euclidean distance (L2 norm)

    :return: Distance value as a differentiable tensor.

    :raises ValueError: If an unknown distance metric is specified.
    """
    if metric == "cosine":
        # Cosine distance: 1 - cosine_similarity
        # Add small epsilon to avoid division by zero
        a_norm = a / (a.norm() + 1e-8)
        b_norm = b / (b.norm() + 1e-8)
        cosine_sim = torch.dot(a_norm, b_norm)
        return 1.0 - cosine_sim
    elif metric == "manhattan":
        # Manhattan distance (L1 norm): sum of absolute differences
        return torch.sum(torch.abs(a - b))
    elif metric == "euclidean":
        # Euclidean distance (L2 norm): standard geometric distance
        return torch.norm(a - b)
    else:
        raise ValueError(f"Unknown distance metric: {metric}. "
                       f"Supported metrics: 'cosine', 'manhattan', 'euclidean'")


@experiment.hook('optimize_representation', replace=False, default=True)
def optimize_representation(e: Experiment,
                           model: NeuralNet,
                           initial_representation: np.ndarray,
                           target_value: float,
                           flow_model: Optional['NormalizingFlow'] = None,
                           ) -> Tuple[np.ndarray, List[dict], List[tuple]]:
    """
    Perform gradient descent in representation space to reach target property.

    This hook optimizes in the original representation space (no PCA) using
    a single model (no ensemble) with optional normalizing flow constraint:
        loss = MSE(prediction, target) + λ_flow * (-log_prob) + λ_dist * distance

    :param e: The experiment instance.
    :param model: The trained NeuralNet model.
    :param initial_representation: Starting representation vector.
    :param target_value: Target property value to optimize towards.
    :param flow_model: Optional NormalizingFlow for likelihood constraint.

    :return: Tuple of (optimized_representation, history, trajectory_samples)
    """
    e.log(f' * optimizing in original space: {len(initial_representation)}D')

    # Initialize representation
    representation = torch.tensor(
        initial_representation.copy(),
        dtype=torch.float32,
        requires_grad=True
    )

    # Store initial for distance calculation
    initial_representation_tensor = torch.tensor(
        initial_representation.copy(),
        dtype=torch.float32,
        requires_grad=False
    )

    # Freeze model parameters
    for param in model.parameters():
        param.requires_grad = False

    # Optimizer
    optimizer = torch.optim.Adam([representation], lr=e.OPTIMIZATION_LEARNING_RATE)

    # Learning rate scheduler
    scheduler = LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=e.OPTIMIZATION_LEARNING_RATE_REDUCTION,
        total_iters=e.OPTIMIZATION_EPOCHS
    )

    target_tensor = torch.tensor([[target_value]], dtype=torch.float32)

    # History tracking
    history = []
    trajectory_samples = []

    for epoch in range(e.OPTIMIZATION_EPOCHS):
        optimizer.zero_grad()

        # Get model prediction
        pred = model.predict_with_grad(representation.unsqueeze(0))

        # MSE loss
        mse_loss = nn.functional.mse_loss(pred, target_tensor)

        # Flow constraint (if enabled)
        flow_loss = torch.tensor(0.0)
        if flow_model is not None:
            flow_log_prob = flow_model.log_prob(representation.unsqueeze(0))
            flow_loss = -flow_log_prob.mean()  # Maximize likelihood

        # Original distance regularization
        original_distance_loss = compute_distance(
            representation,
            initial_representation_tensor,
            e.DISTANCE_METRIC
        )

        # Gradient balancing: compute and normalize gradients separately
        # This prevents any single loss term from dominating the optimization
        epsilon = 1e-8
        grad_list = []
        weight_list = []
        grad_norms = {}  # Track gradient norms for diagnostics

        # MSE gradient (always present, weight = 1.0)
        mse_loss.backward(retain_graph=True)
        mse_grad = representation.grad.clone()
        representation.grad.zero_()
        mse_grad_norm = torch.norm(mse_grad).item()
        grad_norms['mse_grad_norm'] = mse_grad_norm
        grad_list.append(mse_grad / (mse_grad_norm + epsilon))
        weight_list.append(1.0)

        # Flow gradient (if enabled)
        if flow_model is not None and e.FLOW_WEIGHT > 0:
            flow_loss.backward(retain_graph=True)
            flow_grad = representation.grad.clone()
            representation.grad.zero_()
            flow_grad_norm = torch.norm(flow_grad).item()
            grad_norms['flow_grad_norm'] = flow_grad_norm
            grad_list.append(flow_grad / (flow_grad_norm + epsilon))
            weight_list.append(e.FLOW_WEIGHT)
        else:
            grad_norms['flow_grad_norm'] = 0.0

        # Original distance gradient (if enabled)
        # Note: Negative weights flip gradient direction to push away from initial
        if e.ORIGINAL_DISTANCE_WEIGHT != 0:
            original_distance_loss.backward(retain_graph=True)
            distance_grad = representation.grad.clone()
            representation.grad.zero_()
            distance_grad_norm = torch.norm(distance_grad).item()
            grad_norms['distance_grad_norm'] = distance_grad_norm
            grad_list.append(distance_grad / (distance_grad_norm + epsilon))
            weight_list.append(e.ORIGINAL_DISTANCE_WEIGHT)
        else:
            grad_norms['distance_grad_norm'] = 0.0

        # Combine normalized gradients with weights
        combined_grad = sum(w * g for w, g in zip(weight_list, grad_list))
        representation.grad = combined_grad
        grad_norms['combined_grad_norm'] = torch.norm(combined_grad).item()

        # Compute total loss for tracking (no gradient needed)
        with torch.no_grad():
            total_loss = mse_loss + e.FLOW_WEIGHT * flow_loss + e.ORIGINAL_DISTANCE_WEIGHT * original_distance_loss

        # Optional gradient clipping on the combined gradient
        if e.ENABLE_GRADIENT_CLIPPING:
            torch.nn.utils.clip_grad_norm_([representation], e.GRADIENT_CLIP_VALUE)

        optimizer.step()

        # Optionally project representation back to unit norm
        # This provides a hard constraint (guarantees norm=1.0) for HDC embeddings.
        # If disabled, the flow model provides implicit soft regularization via its likelihood.
        if e.USE_NORM_PROJECTION:
            with torch.no_grad():
                representation.data = representation.data / torch.norm(representation.data)

        scheduler.step()

        # Record history
        with torch.no_grad():
            history.append({
                'epoch': epoch,
                'total_loss': total_loss.item(),
                'mse_loss': mse_loss.item(),
                'flow_loss': flow_loss.item() if flow_model else 0.0,
                'original_distance_loss': original_distance_loss.item(),
                'prediction': pred.item(),
                'distance_to_target': abs(pred.item() - target_value),
                'learning_rate': optimizer.param_groups[0]['lr'],
                'representation_norm': torch.norm(representation).item(),
                # Gradient norms (before normalization)
                'mse_grad_norm': grad_norms['mse_grad_norm'],
                'flow_grad_norm': grad_norms['flow_grad_norm'],
                'distance_grad_norm': grad_norms['distance_grad_norm'],
                'combined_grad_norm': grad_norms['combined_grad_norm'],
            })

            # Sample trajectory
            if epoch % e.TRAJECTORY_SAMPLING_INTERVAL == 0:
                sampled_rep = representation.detach().cpu().numpy().copy()
                trajectory_samples.append((epoch, sampled_rep))

    # Final representation
    optimized_representation = representation.detach().cpu().numpy()

    # Include final point if not already sampled
    if (e.OPTIMIZATION_EPOCHS - 1) % e.TRAJECTORY_SAMPLING_INTERVAL != 0:
        trajectory_samples.append((e.OPTIMIZATION_EPOCHS - 1, optimized_representation.copy()))

    return optimized_representation, history, trajectory_samples


@experiment.hook('find_closest_element', replace=False, default=True)
def find_closest_element(e: Experiment,
                         representation: np.ndarray,
                         index_data_map: dict[int, GraphDict],
                         candidate_indices: List[int],
                         ) -> Tuple[int, float]:
    """
    Find the closest element in the dataset to the given representation.

    This hook computes Manhattan distances (L1 norm) between the query representation
    and all candidate representations in original space, returning the index and
    distance of the closest match.

    :param e: The experiment instance.
    :param representation: Query representation vector.
    :param index_data_map: Dictionary of graph data.
    :param candidate_indices: Indices of candidate elements to search.

    :return: Tuple of (closest_index, manhattan_distance).
    """
    candidate_features = np.array([
        index_data_map[idx]['graph_features']
        for idx in candidate_indices
    ])

    # Compute Manhattan distances in original space
    distances = np.sum(np.abs(candidate_features - representation), axis=1)

    closest_idx_in_candidates = np.argmin(distances)
    closest_index = candidate_indices[closest_idx_in_candidates]
    closest_distance = distances[closest_idx_in_candidates]

    return closest_index, closest_distance


# == MAIN EXPERIMENT ==


@experiment
def main(e: Experiment):
    """
    Main experiment function for molecular representation optimization.

    Workflow:
    1. Load and filter molecular dataset
    2. Process molecules into representations
    3. Extract target properties
    4. Split dataset
    5. Train single neural network model
    6. Train normalizing flow model (optional)
    7. Perform optimization on test samples with gradient balancing
    8. Evaluate and visualize results

    :param e: The experiment instance.

    :return: None. Results are saved as artifacts.
    """
    e.log('starting molecular representation optimization experiment...')

    # Handle random seed: if None, pick a random seed for this run
    if e.SEED is None:
        e.SEED = random.randint(0, 2**31 - 1)
        e.log(f'SEED was None, randomly selected seed: {e.SEED}')

    e.log_parameters()

    # == DATASET LOADING ==

    e.log('\n=== LOADING DATASET ===')
    index_data_map: dict[int, GraphDict] = e.apply_hook('load_dataset')
    e.log(f'loaded dataset size: {len(index_data_map)}')

    # == DATASET FILTERING ==

    e.log('\n=== FILTERING DATASET ===')
    e.apply_hook('filter_dataset', index_data_map=index_data_map)
    e.log(f'filtered dataset size: {len(index_data_map)}')

    # == DATASET POST-PROCESSING ==
    # This hook point allows mixins to modify the dataset after filtering
    # (e.g., calculating CLogP values, adding computed properties)
    # Note: Runs after filtering so only valid molecules are processed
    e.apply_hook('after_dataset', index_data_map=index_data_map)

    # IMPORTANT: Guarantees for after_dataset hook and property value injection:
    #
    # If a mixin modifies graph_labels in the after_dataset hook, these changes
    # will be correctly preserved throughout the experiment:
    #
    # 1. process_dataset (next step) operates on molecular STRUCTURE only and
    #    does not read or modify graph_labels
    # 2. extract_property (called later) will read from the modified graph_labels
    # 3. All downstream code uses property_value set by extract_property
    #
    # This ensures mixins can safely inject custom target properties by either:
    # - Overwriting graph['graph_labels'] in after_dataset hook (standard approach)
    # - Setting graph['property_value'] directly (alternative, uses safety check)
    #
    # For caching: Embeddings/fingerprints are structure-based and don't depend
    # on target properties, so cache reuse across different targets is safe.

    # == DATASET PROCESSING ==

    e.log('\n=== PROCESSING DATASET ===')
    time_start = time.time()
    e.apply_hook('process_dataset', index_data_map=index_data_map)
    time_end = time.time()
    e.log(f'processed dataset after {time_end - time_start:.2f} seconds')

    # Verify all indices have graph_features
    missing_features = [idx for idx in index_data_map if 'graph_features' not in index_data_map[idx]]
    if missing_features:
        e.log(f'WARNING: {len(missing_features)} indices missing graph_features: {missing_features[:10]}...')
    else:
        e.log(f'all {len(index_data_map)} indices have graph_features')

    # Check feature dimensions
    first_idx = list(index_data_map.keys())[0]
    feature_dim = len(index_data_map[first_idx]['graph_features'])
    e.log(f'feature dimension: {feature_dim}')

    # == EXTRACT PROPERTIES ==

    e.log('\n=== EXTRACTING PROPERTIES ===')
    for index in index_data_map.keys():
        graph = index_data_map[index]
        property_value = e.apply_hook(
            'extract_property',
            index=index,
            graph=graph
        )
        graph['property_value'] = property_value

    properties = np.array([g['property_value'] for g in index_data_map.values()])
    e.log(f'extracted {len(properties)} property values')
    e.log(f'property range: [{properties.min():.3f}, {properties.max():.3f}]')
    e.log(f'property mean: {properties.mean():.3f}, std: {properties.std():.3f}')

    # == PROPERTY DISTRIBUTION PLOT ==

    e.log('\n=== PLOTTING PROPERTY DISTRIBUTION ===')

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot histogram of property values
    n_bins = min(50, len(properties) // 10)  # Adaptive bin count
    counts, bins, patches = ax.hist(
        properties,
        bins=n_bins,
        alpha=0.7,
        color='steelblue',
        edgecolor='black',
        linewidth=0.5
    )

    # Mark the target value (only in absolute mode)
    if not e.TARGET_RELATIVE and e.TARGET_VALUE is not None:
        ax.axvline(
            e.TARGET_VALUE,
            color='red',
            linestyle='--',
            linewidth=2.5,
            label=f'Target Value ({e.TARGET_VALUE:.3f})',
            zorder=10
        )
    elif e.TARGET_RELATIVE:
        # In relative mode, show the offset in annotation
        ax.text(
            0.02, 0.98,
            f'Relative mode: offset = {e.TARGET_VALUE:+.3f}',
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )

    # Mark the mean for reference
    mean_value = properties.mean()
    ax.axvline(
        mean_value,
        color='green',
        linestyle=':',
        linewidth=2,
        label=f'Mean ({mean_value:.3f})',
        alpha=0.7,
        zorder=9
    )

    # Add statistics text box
    stats_text = '\n'.join([
        f'N = {len(properties)}',
        f'Min = {properties.min():.3f}',
        f'Max = {properties.max():.3f}',
        f'Mean = {mean_value:.3f}',
        f'Median = {np.median(properties):.3f}',
        f'Std = {properties.std():.3f}',
    ])

    ax.text(
        0.02, 0.98,
        stats_text,
        transform=ax.transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
        fontsize=10,
        family='monospace'
    )

    # Calculate percentile of target value (only in absolute mode)
    if not e.TARGET_RELATIVE and e.TARGET_VALUE is not None:
        target_percentile = (properties < e.TARGET_VALUE).sum() / len(properties) * 100
        e.log(f'target value is at {target_percentile:.1f}th percentile of property distribution')

    ax.set_xlabel('Property Value', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    if e.TARGET_RELATIVE:
        title_text = (
            f'Distribution of Target Property\n'
            f'Dataset: {e.DATASET_NAME_ID} (Relative mode: offset={e.TARGET_VALUE:+.3f})'
        )
    elif e.TARGET_VALUE is not None:
        title_text = (
            f'Distribution of Target Property\n'
            f'Dataset: {e.DATASET_NAME_ID} (Target at {target_percentile:.1f}th percentile)'
        )
    else:
        title_text = (
            f'Distribution of Target Property\n'
            f'Dataset: {e.DATASET_NAME_ID}'
        )
    ax.set_title(title_text, fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()

    # Save the figure
    distribution_path = os.path.join(e.path, 'property_distribution.png')
    fig.savefig(distribution_path, dpi=300, bbox_inches='tight')
    e.log(f'saved property distribution plot to: {distribution_path}')
    plt.close(fig)

    # == DATASET SPLITTING ==

    e.log('\n=== SPLITTING DATASET ===')
    indices = list(index_data_map.keys())
    train_indices, val_indices, test_indices = e.apply_hook(
        'dataset_split',
        indices=indices
    )
    e.log(f'train: {len(train_indices)}, val: {len(val_indices)}, test: {len(test_indices)}')

    e['indices/train'] = train_indices
    e['indices/val'] = val_indices
    e['indices/test'] = test_indices

    # == TRAIN MODEL ==

    e.log('\n=== TRAINING MODEL ===')
    model = e.apply_hook(
        'train_model',
        index_data_map=index_data_map,
        train_indices=train_indices,
        val_indices=val_indices,
    )

    # Save model
    model_path = os.path.join(e.path, 'model.pth')
    torch.save(model.state_dict(), model_path)
    e.log(f'saved model to {model_path}')

    # == TRAIN NORMALIZING FLOW ==

    if e.USE_FLOW_CONSTRAINT:
        train_features = np.array([index_data_map[i]['graph_features'] for i in train_indices])

        flow_model = e.apply_hook(
            'train_flow',
            train_features=train_features,
        )

        # Save flow model
        if flow_model is not None:
            flow_path = os.path.join(e.path, 'flow_model.pth')
            torch.save(flow_model.state_dict(), flow_path)
            e.log(f'saved flow model to {flow_path}')
    else:
        flow_model = None
        e.log('\nflow constraint disabled')

    # Evaluate model on test set
    test_features = np.array([index_data_map[i]['graph_features'] for i in test_indices])
    test_properties = np.array([index_data_map[i]['property_value'] for i in test_indices])

    # == CALCULATE 5-NN BASELINE DISTANCE ==

    e.log('\n=== CALCULATING 5-NN BASELINE DISTANCE ===')

    # Subsample test set if too large (for computational efficiency)
    # For large datasets, computing all pairwise distances is O(n²) which becomes prohibitive
    MAX_5NN_SAMPLES = 2000
    if len(test_features) > MAX_5NN_SAMPLES:
        e.log(f'subsampling {MAX_5NN_SAMPLES} test points for 5-NN baseline (from {len(test_features)} total)')
        rng = np.random.RandomState(e.SEED)
        sample_indices = rng.choice(len(test_features), size=MAX_5NN_SAMPLES, replace=False)
        sampled_test_features = test_features[sample_indices]
    else:
        e.log(f'using all {len(test_features)} test points for 5-NN baseline')
        sampled_test_features = test_features

    # For each sampled test point, find average distance to its 5 nearest neighbors
    n_neighbors = 5
    test_5nn_distances = []

    for i, test_feat in enumerate(sampled_test_features):
        # Compute Manhattan distances to all sampled test points
        distances = np.sum(np.abs(sampled_test_features - test_feat), axis=1)

        # Use partial sort (np.partition) which is O(n) instead of full sort O(n log n)
        # We only need the 6 smallest values (self + 5 neighbors)
        if len(distances) > n_neighbors + 1:
            # partition puts the k smallest elements in the first k positions (unsorted)
            nearest_distances = np.partition(distances, n_neighbors)[:n_neighbors+1]
            # Remove self (distance = 0) and take mean of remaining 5
            nearest_5_distances = nearest_distances[nearest_distances > 0][:n_neighbors]
        else:
            # Edge case: fewer than 6 points total
            sorted_distances = np.sort(distances)
            nearest_5_distances = sorted_distances[1:min(n_neighbors+1, len(sorted_distances))]

        if len(nearest_5_distances) > 0:
            mean_5nn_distance = np.mean(nearest_5_distances)
            test_5nn_distances.append(mean_5nn_distance)

    # Average across all sampled test points
    avg_5nn_distance = np.mean(test_5nn_distances)

    e.log(f'average 5-nearest-neighbor distance in test set: {avg_5nn_distance:.4f}')
    e.log(f' * calculated over {len(sampled_test_features)} test points')
    e.log(f' * using Manhattan distance (L1 norm)')

    e['test_set/avg_5nn_distance'] = float(avg_5nn_distance)
    e['test_set/n_neighbors'] = n_neighbors
    e['test_set/5nn_samples_used'] = len(sampled_test_features)

    # == MODEL EVALUATION ==

    e.log('\n=== MODEL EVALUATION ===')

    # Evaluate model on test set
    model.eval()
    with torch.no_grad():
        test_features_tensor = torch.tensor(test_features, dtype=torch.float32)
        test_predictions = model(test_features_tensor).cpu().numpy().flatten()

    # Compute metrics
    test_mae = mean_absolute_error(test_properties, test_predictions)
    test_mse = mean_squared_error(test_properties, test_predictions)
    test_rmse = np.sqrt(test_mse)
    test_r2 = r2_score(test_properties, test_predictions)

    e.log(f'\nmodel test performance:')
    e.log(f' * MAE: {test_mae:.4f}')
    e.log(f' * MSE: {test_mse:.4f}')
    e.log(f' * RMSE: {test_rmse:.4f}')
    e.log(f' * R²: {test_r2:.4f}')

    e['model/test_mae'] = float(test_mae)
    e['model/test_mse'] = float(test_mse)
    e['model/test_rmse'] = float(test_rmse)
    e['model/test_r2'] = float(test_r2)

    # == MODEL EVALUATION VISUALIZATIONS ==

    e.log('\n=== CREATING MODEL EVALUATION PLOTS ===')

    # Prediction vs Actual scatter plot
    e.log('creating prediction vs actual scatter plot...')

    fig, ax = plt.subplots(figsize=(10, 8))

    # Scatter plot of predictions vs actual values
    ax.scatter(
        test_properties,
        test_predictions,
        alpha=0.6,
        s=50,
        color='steelblue',
        edgecolors='black',
        linewidth=0.5,
        label=f'Predictions (R²={test_r2:.3f})'
    )

    # Perfect prediction line
    min_val = min(test_properties.min(), test_predictions.min())
    max_val = max(test_properties.max(), test_predictions.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, alpha=0.7, label='Perfect prediction')

    ax.set_xlabel('Actual Property Value', fontsize=12)
    ax.set_ylabel('Predicted Property Value', fontsize=12)
    ax.set_title(f'Model: Prediction vs Actual\n'
                 f'MAE={test_mae:.3f}, RMSE={test_rmse:.3f}, R²={test_r2:.3f}\n'
                 f'{len(test_properties)} test samples',
                 fontsize=13)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    e.commit_fig('model_evaluation_predictions.png', fig)

    # == DETERMINE TARGET VALUE ==

    if e.TARGET_RELATIVE:
        # Relative mode: TARGET_VALUE is an offset applied to each sample's initial property
        if e.TARGET_VALUE is None:
            raise ValueError(
                "TARGET_VALUE cannot be None when TARGET_RELATIVE=True. "
                "Specify a relative offset (e.g., +1.0 to increase, -0.5 to decrease)."
            )
        target_offset = e.TARGET_VALUE
        e.log(f'\nusing relative target mode: offset = {target_offset:+.3f}')
        e.log('  (target will be: initial_property + offset for each sample)')
        e['optimization/target_mode'] = 'relative'
        e['optimization/target_offset'] = float(target_offset)
        target_value = None  # Will be calculated per-sample
    else:
        # Absolute mode: TARGET_VALUE is a fixed target for all samples
        if e.TARGET_VALUE is None:
            target_value = float(np.median(test_properties))
            e.log(f'\nusing absolute target mode: median test property = {target_value:.3f}')
        else:
            target_value = e.TARGET_VALUE
            e.log(f'\nusing absolute target mode: specified value = {target_value:.3f}')
        e['optimization/target_mode'] = 'absolute'
        e['optimization/target_value'] = float(target_value)

    # == FIND GLOBAL OPTIMAL TEST MOLECULE ==

    # Find the test molecule with property value closest to the target value.
    # This serves as the "gold star" reference - the best achievable result.
    # Note: In relative mode, this represents the average optimal distance
    if e.TARGET_RELATIVE:
        # In relative mode, the "optimal" is the offset itself (same for all samples)
        optimal_test_idx_position = 0  # Arbitrary, not meaningful in relative mode
        optimal_test_index = test_indices[optimal_test_idx_position]
        optimal_test_property = test_properties[optimal_test_idx_position] + target_offset
        optimal_test_representation = test_features[optimal_test_idx_position]

        e.log(f'\nrelative mode: target offset = {target_offset:+.3f} for all samples')
        e['optimal/target_offset'] = float(target_offset)
    else:
        test_property_distances = np.abs(test_properties - target_value)
        optimal_test_idx_position = np.argmin(test_property_distances)
        optimal_test_index = test_indices[optimal_test_idx_position]
        optimal_test_property = test_properties[optimal_test_idx_position]
        optimal_test_representation = test_features[optimal_test_idx_position]

        e.log(f'\nglobal optimal test molecule:')
        e.log(f' * index: {optimal_test_index}')
        e.log(f' * property value: {optimal_test_property:.4f}')
        e.log(f' * distance to target: {abs(optimal_test_property - target_value):.4f}')

        e['optimal/test_index'] = int(optimal_test_index)
        e['optimal/property_value'] = float(optimal_test_property)
        e['optimal/distance_to_target'] = float(abs(optimal_test_property - target_value))

    # == SELECT OPTIMIZATION SAMPLES ==

    if e.NUM_OPTIMIZATION_SAMPLES is None:
        optimization_indices = test_indices.copy()
    else:
        num_opt = min(e.NUM_OPTIMIZATION_SAMPLES, len(test_indices))
        random.seed(e.SEED)
        optimization_indices = random.sample(test_indices, k=num_opt)

    e.log(f'\noptimizing {len(optimization_indices)} test samples')

    # == PERFORM OPTIMIZATIONS ==

    e.log('\n=== PERFORMING OPTIMIZATIONS ===')

    # Prepare list of all indices for closest element search
    all_indices = train_indices + val_indices + test_indices
    e.log(f'\nusing all dataset indices for closest element search: {len(all_indices)} total '
          f'(train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)})')

    optimization_results = []
    all_histories = []

    for opt_idx, test_idx in enumerate(optimization_indices):
        e.log(f'\noptimization {opt_idx + 1}/{len(optimization_indices)} (test idx: {test_idx})')

        initial_representation = index_data_map[test_idx]['graph_features'].copy()
        initial_property = index_data_map[test_idx]['property_value']

        # Calculate target value (per-sample in relative mode, global in absolute mode)
        if e.TARGET_RELATIVE:
            sample_target_value = initial_property + target_offset
            e.log(f' * initial property: {initial_property:.3f}')
            e.log(f' * target offset: {target_offset:+.3f}')
            e.log(f' * target property: {sample_target_value:.3f}')
            e.log(f' * initial distance: {abs(target_offset):.3f}')
        else:
            sample_target_value = target_value
            e.log(f' * initial property: {initial_property:.3f}')
            e.log(f' * target property: {sample_target_value:.3f}')
            e.log(f' * initial distance: {abs(initial_property - sample_target_value):.3f}')

        # Optimize
        time_start = time.time()
        optimized_representation, history, trajectory_samples = e.apply_hook(
            'optimize_representation',
            model=model,
            initial_representation=initial_representation,
            target_value=sample_target_value,
            flow_model=flow_model,
        )
        time_end = time.time()

        e.log(f' * optimization time: {time_end - time_start:.2f}s')
        e.log(f' * trajectory samples: {len(trajectory_samples)}')

        # Process trajectory samples to find closest molecules from entire dataset
        trajectory_closest = []
        for epoch, sampled_rep in trajectory_samples:
            traj_closest_idx, traj_closest_distance = e.apply_hook(
                'find_closest_element',
                representation=sampled_rep,
                index_data_map=index_data_map,
                candidate_indices=all_indices,
            )
            traj_closest_property = index_data_map[traj_closest_idx]['property_value']
            trajectory_closest.append({
                'epoch': epoch,
                'closest_idx': traj_closest_idx,
                'closest_property': traj_closest_property,
                'closest_distance': traj_closest_distance,
            })

        # Find closest element from entire dataset to optimized representation
        closest_idx, closest_distance = e.apply_hook(
            'find_closest_element',
            representation=optimized_representation,
            index_data_map=index_data_map,
            candidate_indices=all_indices,
        )

        closest_property = index_data_map[closest_idx]['property_value']

        # Find top 3 closest elements for logging
        distances_to_all = []
        for idx in all_indices:
            dist = np.linalg.norm(optimized_representation - index_data_map[idx]['graph_features'])
            distances_to_all.append((idx, dist))

        # Sort by distance and get top 3
        distances_to_all.sort(key=lambda x: x[1])
        top3_closest = distances_to_all[:3]

        # Get final prediction
        model.eval()
        with torch.no_grad():
            final_pred_tensor = model(torch.tensor(optimized_representation.reshape(1, -1), dtype=torch.float32))
            final_pred = final_pred_tensor.cpu().numpy()[0, 0]

        e.log(f' * final predicted property: {final_pred:.3f}')
        e.log(f' * 3 closest element indices (from all data): {top3_closest[0][0]} ({top3_closest[0][1]:.3f}), '
              f'{top3_closest[1][0]} ({top3_closest[1][1]:.3f}), {top3_closest[2][0]} ({top3_closest[2][1]:.3f})')
        e.log(f' * closest element property: {closest_property:.3f}')
        e.log(f' * closest element distance (repr): {closest_distance:.3f}')
        e.log(f' * true distance to target: {abs(closest_property - sample_target_value):.3f}')

        # Distance comparison metrics
        closest_distance_ratio = closest_distance / avg_5nn_distance
        e.log(f' * average 5-NN baseline distance: {avg_5nn_distance:.3f}')
        e.log(f' * closest distance ratio: {closest_distance_ratio:.3f}x '
              f'({"closer" if closest_distance_ratio < 1.0 else "farther"} than avg 5-NN)')

        # Calculate Tanimoto similarity between initial and final molecules
        initial_smiles = index_data_map[test_idx]['graph_repr']
        final_smiles = index_data_map[closest_idx]['graph_repr']
        initial_mol = Chem.MolFromSmiles(initial_smiles)
        final_mol = Chem.MolFromSmiles(final_smiles)

        tanimoto_similarity = None
        if initial_mol is not None and final_mol is not None:
            initial_fp = AllChem.GetMorganFingerprintAsBitVect(initial_mol, radius=2, nBits=2048)
            final_fp = AllChem.GetMorganFingerprintAsBitVect(final_mol, radius=2, nBits=2048)
            tanimoto_similarity = DataStructs.TanimotoSimilarity(initial_fp, final_fp)
            e.log(f' * Tanimoto similarity (initial vs final): {tanimoto_similarity:.3f}')

        result = {
            'test_idx': test_idx,
            'initial_property': initial_property,
            'target_value': sample_target_value,
            'final_predicted_property': final_pred,
            'closest_idx': closest_idx,
            'closest_property': closest_property,
            'closest_distance': closest_distance,
            'avg_5nn_baseline': avg_5nn_distance,
            'closest_distance_ratio': closest_distance_ratio,
            'initial_distance_to_target': abs(initial_property - sample_target_value),
            'predicted_distance_to_target': abs(final_pred - sample_target_value),
            'true_distance_to_target': abs(closest_property - sample_target_value),
            'optimization_time': time_end - time_start,
            'tanimoto_similarity': tanimoto_similarity,
            'trajectory_closest': trajectory_closest,  # Store trajectory data for visualization
            'optimized_representation': optimized_representation.copy(),  # Raw optimized vector
            'initial_representation': initial_representation.copy(),      # Initial vector
        }

        optimization_results.append(result)

        # Store history with test_idx
        for h in history:
            h['test_idx'] = test_idx
        all_histories.extend(history)

        # Plot individual trajectory if enabled
        if e.PLOT_INDIVIDUAL_TRAJECTORIES:
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=e.TRAJECTORY_FIGURE_SIZE)

            epochs = [h['epoch'] for h in history]
            distances = [h['distance_to_target'] for h in history]
            flow_losses = [h['flow_loss'] for h in history]
            original_distance_losses = [h['original_distance_loss'] for h in history]
            mse_grad_norms = [h['mse_grad_norm'] for h in history]
            flow_grad_norms = [h['flow_grad_norm'] for h in history]
            distance_grad_norms = [h['distance_grad_norm'] for h in history]

            # Distance over epochs
            ax1.plot(epochs, distances, label='Predicted distance', linewidth=2)
            ax1.axhline(
                result['true_distance_to_target'],
                color='green',
                linestyle='--',
                label=f'True distance (closest): {result["true_distance_to_target"]:.3f}'
            )
            ax1.axhline(
                result['initial_distance_to_target'],
                color='red',
                linestyle='--',
                label=f'Initial distance: {result["initial_distance_to_target"]:.3f}'
            )
            # Add global optimal reference line
            # In relative mode, optimal is just the offset (same for all)
            if e.TARGET_RELATIVE:
                optimal_distance_to_target = abs(target_offset)
            else:
                optimal_distance_to_target = abs(optimal_test_property - target_value)
            ax1.axhline(
                optimal_distance_to_target,
                color='gold',
                linestyle=':',
                linewidth=2.5,
                label=f'Global optimal: {optimal_distance_to_target:.3f}'
            )
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Distance to Target')
            ax1.set_title(f'Optimization Trajectory (test_idx={test_idx})')
            ax1.legend(loc='best')
            ax1.grid(True, alpha=0.3)

            # Flow loss over epochs (if flow constraint is used)
            # Filter out invalid values (NaN, Inf)
            flow_losses_array = np.array(flow_losses)
            valid_mask = np.isfinite(flow_losses_array)

            if valid_mask.any():
                valid_epochs = [e for e, valid in zip(epochs, valid_mask) if valid]
                valid_flow_losses = flow_losses_array[valid_mask]

                ax2.plot(valid_epochs, valid_flow_losses, color='orange', linewidth=2)
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('Flow Loss (-log prob)')
                ax2.set_title('Normalizing Flow Constraint')

                # Use symlog scale to handle both positive and negative values
                # Check if we have positive values for log scale
                if (valid_flow_losses > 0).all():
                    ax2.set_yscale('log')
                elif (valid_flow_losses > 0).any():
                    # Mixed positive and negative - use symlog
                    ax2.set_yscale('symlog')
                # else: keep linear scale for all negative/zero values

                ax2.grid(True, alpha=0.3, which='both')
            else:
                # If no valid values, show empty plot with message
                ax2.text(0.5, 0.5, 'No valid flow loss values',
                        ha='center', va='center', transform=ax2.transAxes)
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('Flow Loss (-log prob)')
                ax2.set_title('Normalizing Flow Constraint')

            # Original distance loss over epochs
            ax3.plot(epochs, original_distance_losses, color='purple', linewidth=2)
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Original Distance Loss')
            ax3.set_title(f'Distance to Initial (metric={e.DISTANCE_METRIC})')
            ax3.grid(True, alpha=0.3)

            # Gradient norms over epochs (before balancing normalization)
            ax4.plot(epochs, mse_grad_norms, label='MSE grad', linewidth=2, color='blue', alpha=0.8)
            if flow_model is not None:
                ax4.plot(epochs, flow_grad_norms, label='Flow grad', linewidth=2, color='orange', alpha=0.8)
            if e.ORIGINAL_DISTANCE_WEIGHT != 0:
                ax4.plot(epochs, distance_grad_norms, label='Distance grad', linewidth=2, color='purple', alpha=0.8)
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Gradient Norm (before balancing)')
            ax4.set_title('Gradient Magnitudes')
            ax4.set_yscale('log')  # Log scale to see relative differences
            ax4.legend(loc='best')
            ax4.grid(True, alpha=0.3, which='both')

            plt.tight_layout()
            e.commit_fig(f'optimization_trajectory_{test_idx}.png', fig)

            # Plot molecular structures: initial vs final
            e.log(f' * creating molecular structure comparison...')

            # Get SMILES strings
            initial_smiles = index_data_map[test_idx]['graph_repr']
            final_smiles = index_data_map[closest_idx]['graph_repr']

            # Convert to RDKit molecules
            initial_mol = Chem.MolFromSmiles(initial_smiles)
            final_mol = Chem.MolFromSmiles(final_smiles)

            if initial_mol is not None and final_mol is not None:
                # Calculate Tanimoto similarity using Morgan fingerprints
                initial_fp = AllChem.GetMorganFingerprintAsBitVect(initial_mol, radius=2, nBits=2048)
                final_fp = AllChem.GetMorganFingerprintAsBitVect(final_mol, radius=2, nBits=2048)
                tanimoto_similarity = DataStructs.TanimotoSimilarity(initial_fp, final_fp)

                e.log(f' * Tanimoto similarity: {tanimoto_similarity:.3f}')

                # Create figure with molecules side by side
                fig_mol = plt.figure(figsize=(14, 6))

                # Draw initial molecule
                ax_initial = fig_mol.add_subplot(1, 2, 1)
                img_initial = Draw.MolToImage(initial_mol, size=(500, 500))
                ax_initial.imshow(img_initial)
                ax_initial.axis('off')
                ax_initial.set_title(
                    f'Initial Molecule (test_idx={test_idx})\n'
                    f'Property: {initial_property:.3f}',
                    fontsize=12,
                    fontweight='bold'
                )

                # Draw final molecule
                ax_final = fig_mol.add_subplot(1, 2, 2)
                img_final = Draw.MolToImage(final_mol, size=(500, 500))
                ax_final.imshow(img_final)
                ax_final.axis('off')
                ax_final.set_title(
                    f'Final Molecule (closest_idx={closest_idx})\n'
                    f'Property: {closest_property:.3f}',
                    fontsize=12,
                    fontweight='bold'
                )

                # Add caption with Tanimoto similarity
                fig_mol.suptitle(
                    f'Molecular Optimization: Initial → Final\n'
                    f'Tanimoto Similarity: {tanimoto_similarity:.3f} | '
                    f'Target: {sample_target_value:.3f} | '
                    f'Distance Improvement: {result["initial_distance_to_target"]:.3f} → {result["true_distance_to_target"]:.3f}',
                    fontsize=13,
                    fontweight='bold',
                    y=0.98
                )

                plt.tight_layout(rect=[0, 0, 1, 0.94])
                e.commit_fig(f'optimization_molecules_{test_idx}.png', fig_mol)
            else:
                e.log(f' * WARNING: Could not visualize molecules (invalid SMILES)')

    # == SAVE RESULTS ==

    e.log('\n=== SAVING RESULTS ===')

    results_df = pd.DataFrame(optimization_results)
    results_path = os.path.join(e.path, 'optimization_results.csv')
    results_df.to_csv(results_path, index=False)
    e.log(f'saved optimization results to {results_path}')

    history_df = pd.DataFrame(all_histories)
    history_path = os.path.join(e.path, 'optimization_history.csv')
    history_df.to_csv(history_path, index=False)
    e.log(f'saved optimization history to {history_path}')

    # Store optimization results as arrays in experiment metadata
    e['test/original'] = results_df['initial_property'].values.tolist()
    e['test/final'] = results_df['closest_property'].values.tolist()
    e['test/target'] = results_df['target_value'].values.tolist()
    e.log(f'stored test arrays: original={len(e["test/original"])}, final={len(e["test/final"])}, target={len(e["test/target"])}')

    # == SUMMARY STATISTICS ==

    e.log('\n=== SUMMARY STATISTICS ===')

    mean_initial_distance = results_df['initial_distance_to_target'].mean()
    mean_predicted_distance = results_df['predicted_distance_to_target'].mean()
    mean_true_distance = results_df['true_distance_to_target'].mean()

    e.log(f'mean initial distance to target: {mean_initial_distance:.4f}')
    e.log(f'mean predicted distance to target: {mean_predicted_distance:.4f}')
    e.log(f'mean true distance to target (closest): {mean_true_distance:.4f}')

    improvement_predicted = (mean_initial_distance - mean_predicted_distance) / mean_initial_distance * 100
    improvement_true = (mean_initial_distance - mean_true_distance) / mean_initial_distance * 100

    e.log(f'improvement (predicted): {improvement_predicted:.2f}%')
    e.log(f'improvement (true): {improvement_true:.2f}%')

    e['summary/mean_initial_distance'] = float(mean_initial_distance)
    e['summary/mean_predicted_distance'] = float(mean_predicted_distance)
    e['summary/mean_true_distance'] = float(mean_true_distance)
    e['summary/improvement_predicted_pct'] = float(improvement_predicted)
    e['summary/improvement_true_pct'] = float(improvement_true)

    # Tanimoto similarity statistics
    # Calculate average Tanimoto similarity between original and optimized molecules
    tanimoto_similarities = results_df['tanimoto_similarity'].dropna()
    if len(tanimoto_similarities) > 0:
        mean_tanimoto_similarity = tanimoto_similarities.mean()
        e.log(f'\nmean Tanimoto similarity (original vs optimized): {mean_tanimoto_similarity:.4f}')
        e['summary/mean_tanimoto_similarity'] = float(mean_tanimoto_similarity)
    else:
        e.log('\nwarning: no valid Tanimoto similarities calculated')
        e['summary/mean_tanimoto_similarity'] = None

    # Global optimal comparison statistics
    if e.TARGET_RELATIVE:
        optimal_distance_to_target = abs(target_offset)
        e.log(f'\nrelative mode optimal distance (offset): {optimal_distance_to_target:.4f}')
    else:
        optimal_distance_to_target = abs(optimal_test_property - target_value)
        e.log(f'\nglobal optimal distance to target: {optimal_distance_to_target:.4f}')

    # How many optimizations reached within X% of global optimal
    threshold_10pct = optimal_distance_to_target * 1.1  # Within 10% of optimal
    threshold_25pct = optimal_distance_to_target * 1.25  # Within 25% of optimal
    within_10pct = (results_df['true_distance_to_target'] <= threshold_10pct).sum()
    within_25pct = (results_df['true_distance_to_target'] <= threshold_25pct).sum()

    e.log(f'optimizations within 10% of optimal: {within_10pct}/{len(results_df)} '
          f'({within_10pct/len(results_df)*100:.1f}%)')
    e.log(f'optimizations within 25% of optimal: {within_25pct}/{len(results_df)} '
          f'({within_25pct/len(results_df)*100:.1f}%)')

    e['summary/optimal_distance'] = float(optimal_distance_to_target)
    e['summary/within_10pct_of_optimal'] = int(within_10pct)
    e['summary/within_25pct_of_optimal'] = int(within_25pct)
    e['summary/pct_within_10pct'] = float(within_10pct / len(results_df) * 100)
    e['summary/pct_within_25pct'] = float(within_25pct / len(results_df) * 100)

    # == SUMMARY PLOTS ==

    e.log('\n=== CREATING SUMMARY PLOTS ===')

    # Plot: Box plot comparison of distances
    fig, ax = plt.subplots(figsize=(12, 6))

    # Prepare data for box plots (distributions from individual optimizations)
    box_data = [
        results_df['initial_distance_to_target'],
        results_df['predicted_distance_to_target'],
        results_df['true_distance_to_target']
    ]

    labels = ['Initial', 'Predicted\n(Optimized)', 'True\n(Closest)']
    colors = ['red', 'blue', 'green']

    # Create box plots
    bp = ax.boxplot(
        box_data,
        labels=labels,
        patch_artist=True,
        widths=0.6,
        showmeans=True,
        meanprops=dict(marker='D', markerfacecolor='black', markeredgecolor='black', markersize=8)
    )

    # Color the boxes
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
        patch.set_edgecolor('black')
        patch.set_linewidth(1.5)

    # Style whiskers, caps, and medians
    for whisker in bp['whiskers']:
        whisker.set_color('black')
        whisker.set_linewidth(1.2)
    for cap in bp['caps']:
        cap.set_color('black')
        cap.set_linewidth(1.2)
    for median in bp['medians']:
        median.set_color('darkred')
        median.set_linewidth(2)

    # Add global optimal reference line
    ax.axhline(
        optimal_distance_to_target,
        color='gold',
        linestyle='--',
        linewidth=2.5,
        label=f'Global Optimal: {optimal_distance_to_target:.3f}',
        zorder=1
    )

    ax.set_ylabel('Distance to Target', fontsize=12)
    if e.TARGET_RELATIVE:
        title_suffix = f'Relative Target: {target_offset:+.3f}, {len(optimization_indices)} samples'
    else:
        title_suffix = f'Target: {target_value:.3f}, {len(optimization_indices)} samples'
    ax.set_title(f'Optimization Performance Distribution\n{title_suffix}', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='upper right', fontsize=10)

    # Add statistics annotation
    stats_text = (
        f'Initial: μ={mean_initial_distance:.3f}, σ={results_df["initial_distance_to_target"].std():.3f}\n'
        f'Predicted: μ={mean_predicted_distance:.3f}, σ={results_df["predicted_distance_to_target"].std():.3f}\n'
        f'True: μ={mean_true_distance:.3f}, σ={results_df["true_distance_to_target"].std():.3f}'
    )
    ax.text(
        0.02, 0.98,
        stats_text,
        transform=ax.transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
        fontsize=9,
        family='monospace'
    )

    plt.tight_layout()
    e.commit_fig('0_optimization_summary.png', fig)

    # == PCA VISUALIZATION ==

    e.log('\n=== CREATING PCA TRAJECTORY VISUALIZATION ===')

    # Collect all test representations
    all_test_features = np.array([index_data_map[i]['graph_features'] for i in test_indices])

    # Fit PCA on test set
    pca = PCA(n_components=e.PCA_COMPONENTS, random_state=e.SEED)
    test_pca = pca.fit_transform(all_test_features)

    e.log(f'PCA explained variance: {pca.explained_variance_ratio_.sum():.4f}')

    # Create PCA visualization
    fig, ax = plt.subplots(figsize=e.PCA_FIGURE_SIZE)

    # Plot all test samples
    test_props = np.array([index_data_map[i]['property_value'] for i in test_indices])
    scatter = ax.scatter(
        test_pca[:, 0],
        test_pca[:, 1],
        c=test_props,
        cmap='viridis',
        alpha=0.3,
        s=30,
        label='Test samples'
    )
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Property Value')

    # Plot global optimal test molecule (gold star)
    optimal_test_rep = optimal_test_representation
    optimal_pca = pca.transform(optimal_test_rep.reshape(1, -1))
    ax.scatter(
        optimal_pca[0, 0],
        optimal_pca[0, 1],
        marker='*',
        s=500,
        color='gold',
        edgecolors='black',
        linewidth=2,
        zorder=15,
        label=f'Global optimal (prop={optimal_test_property:.2f})'
    )

    # Plot optimization trajectories as sequence of closest molecules
    for opt_idx, test_idx in enumerate(optimization_indices):
        result = optimization_results[opt_idx]
        trajectory_closest = result['trajectory_closest']

        # Extract PCA coordinates for each point in trajectory
        trajectory_pca_points = []
        for traj_point in trajectory_closest:
            closest_idx = traj_point['closest_idx']
            closest_rep = index_data_map[closest_idx]['graph_features']
            closest_pca = pca.transform(closest_rep.reshape(1, -1))
            trajectory_pca_points.append(closest_pca[0])

        trajectory_pca_points = np.array(trajectory_pca_points)

        # Plot trajectory path (sequence of closest molecules)
        if len(trajectory_pca_points) > 0:
            # Draw line connecting trajectory points
            ax.plot(
                trajectory_pca_points[:, 0],
                trajectory_pca_points[:, 1],
                color='red',
                alpha=0.6,
                linewidth=1.5,
                marker='o',
                markersize=4,
                zorder=5
            )

            # Mark initial point
            ax.scatter(
                trajectory_pca_points[0, 0],
                trajectory_pca_points[0, 1],
                marker='o',
                s=100,
                color='blue',
                edgecolors='black',
                linewidth=1,
                zorder=8,
                label='Initial' if opt_idx == 0 else None
            )

            # Mark final point
            ax.scatter(
                trajectory_pca_points[-1, 0],
                trajectory_pca_points[-1, 1],
                marker='s',
                s=100,
                color='green',
                edgecolors='black',
                linewidth=1,
                zorder=8,
                label='Final' if opt_idx == 0 else None
            )

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    ax.set_title(f'Optimization Trajectories in PCA Space\n'
                 f'Dataset: {e.DATASET_NAME_ID}, {len(optimization_indices)} optimizations\n'
                 f'Red paths: trajectory through closest molecules, Gold star: global optimal')
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    e.commit_fig('optimization_pca_space.png', fig)

    # == PROPERTY LANDSCAPE HEATMAP ==

    e.log('\n=== CREATING PROPERTY LANDSCAPE HEATMAP ===')

    # Create a grid in PCA space
    grid_resolution = 50
    x_min, x_max = test_pca[:, 0].min() - 1, test_pca[:, 0].max() + 1
    y_min, y_max = test_pca[:, 1].min() - 1, test_pca[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_resolution),
        np.linspace(y_min, y_max, grid_resolution)
    )

    # Transform grid points back to original representation space
    grid_pca = np.c_[xx.ravel(), yy.ravel()]
    grid_original = pca.inverse_transform(grid_pca)

    e.log(f'predicting properties for {len(grid_original)} grid points...')

    # Predict property values at each grid point using single model
    model.eval()
    with torch.no_grad():
        grid_tensor = torch.tensor(grid_original, dtype=torch.float32)
        grid_predictions = model(grid_tensor).cpu().numpy().flatten()
        # No uncertainty estimates for single model (use zeros for compatibility)
        grid_stds = np.zeros_like(grid_predictions)

    # Reshape for plotting
    zz = grid_predictions.reshape(xx.shape)
    zz_std = grid_stds.reshape(xx.shape)

    e.log(f'property landscape range: [{zz.min():.3f}, {zz.max():.3f}]')

    # Create heatmap figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Plot 1: Property value heatmap
    im1 = ax1.contourf(xx, yy, zz, levels=20, cmap='RdYlGn', alpha=0.6)
    contour1 = ax1.contour(xx, yy, zz, levels=10, colors='black', alpha=0.3, linewidths=0.5)
    ax1.clabel(contour1, inline=True, fontsize=8, fmt='%.2f')

    # Overlay test samples
    scatter1 = ax1.scatter(
        test_pca[:, 0],
        test_pca[:, 1],
        c=test_props,
        cmap='viridis',
        alpha=0.7,
        s=50,
        edgecolors='black',
        linewidth=0.5,
        label='Test samples'
    )

    # Overlay global optimal test molecule
    ax1.scatter(
        optimal_pca[0, 0],
        optimal_pca[0, 1],
        marker='*',
        s=500,
        color='gold',
        edgecolors='black',
        linewidth=2,
        zorder=15,
        label=f'Global optimal'
    )

    # Overlay optimization trajectories (sequence of closest molecules)
    for opt_idx, test_idx in enumerate(optimization_indices):
        result = optimization_results[opt_idx]
        trajectory_closest = result['trajectory_closest']

        # Extract PCA coordinates for trajectory
        trajectory_pca_points = []
        for traj_point in trajectory_closest:
            closest_idx = traj_point['closest_idx']
            closest_rep = index_data_map[closest_idx]['graph_features']
            closest_pca = pca.transform(closest_rep.reshape(1, -1))
            trajectory_pca_points.append(closest_pca[0])

        trajectory_pca_points = np.array(trajectory_pca_points)

        if len(trajectory_pca_points) > 0:
            ax1.plot(
                trajectory_pca_points[:, 0],
                trajectory_pca_points[:, 1],
                'r-',
                alpha=0.5,
                linewidth=2,
                zorder=8
            )

    # Mark target value (only in absolute mode)
    if not e.TARGET_RELATIVE:
        target_contour = ax1.contour(
            xx, yy, zz,
            levels=[target_value],
            colors='red',
            linewidths=3,
            linestyles='--'
        )
        ax1.clabel(target_contour, inline=True, fontsize=12, fmt=f'Target: %.2f')

    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Predicted Property Value', fontsize=12)

    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)', fontsize=12)
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)', fontsize=12)
    if e.TARGET_RELATIVE:
        pca_title = (f'Property Landscape in PCA Space\n'
                     f'Relative mode: offset = {target_offset:+.3f}')
    else:
        pca_title = (f'Property Landscape in PCA Space\n'
                     f'Target: {target_value:.3f} (red dashed contour)')
    ax1.set_title(pca_title, fontsize=14)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Model uncertainty heatmap (zeros for single model)
    im2 = ax2.contourf(xx, yy, zz_std, levels=20, cmap='Reds', alpha=0.6)
    contour2 = ax2.contour(xx, yy, zz_std, levels=10, colors='black', alpha=0.3, linewidths=0.5)
    ax2.clabel(contour2, inline=True, fontsize=8, fmt='%.3f')

    # Overlay test samples
    ax2.scatter(
        test_pca[:, 0],
        test_pca[:, 1],
        c='blue',
        alpha=0.5,
        s=50,
        edgecolors='black',
        linewidth=0.5,
        label='Test samples'
    )

    # Overlay global optimal test molecule
    ax2.scatter(
        optimal_pca[0, 0],
        optimal_pca[0, 1],
        marker='*',
        s=500,
        color='gold',
        edgecolors='black',
        linewidth=2,
        zorder=15,
        label='Global optimal'
    )

    # Overlay optimization trajectories
    for opt_idx, test_idx in enumerate(optimization_indices):
        result = optimization_results[opt_idx]
        trajectory_closest = result['trajectory_closest']

        # Extract PCA coordinates for trajectory
        trajectory_pca_points = []
        for traj_point in trajectory_closest:
            closest_idx = traj_point['closest_idx']
            closest_rep = index_data_map[closest_idx]['graph_features']
            closest_pca = pca.transform(closest_rep.reshape(1, -1))
            trajectory_pca_points.append(closest_pca[0])

        trajectory_pca_points = np.array(trajectory_pca_points)

        if len(trajectory_pca_points) > 0:
            ax2.plot(
                trajectory_pca_points[:, 0],
                trajectory_pca_points[:, 1],
                'r-',
                alpha=0.5,
                linewidth=2,
                zorder=8
            )

    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('Model Uncertainty (N/A for single model)', fontsize=12)

    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)', fontsize=12)
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)', fontsize=12)
    ax2.set_title(f'Uncertainty Landscape in PCA Space\n'
                  f'(Not applicable for single model - all values are zero)', fontsize=14)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    e.commit_fig('property_landscape_heatmap.png', fig)
    e.log('saved property landscape heatmap')

    # == REPRESENTATION DISTRIBUTION ANALYSIS ==

    e.log('\n=== ANALYZING REPRESENTATION DISTRIBUTIONS ===')
    e.log('comparing optimized representations vs dataset representations...')

    # Collect all dataset representations (train + val + test combined)
    all_dataset_indices = train_indices + val_indices + test_indices
    all_dataset_features = np.array([
        index_data_map[idx]['graph_features']
        for idx in all_dataset_indices
    ])
    e.log(f'dataset representations: {all_dataset_features.shape}')

    # Collect optimized representations from optimization results
    optimized_representations = np.array([
        result['optimized_representation']
        for result in optimization_results
    ])
    e.log(f'optimized representations: {optimized_representations.shape}')

    # Collect initial representations for reference
    initial_representations = np.array([
        result['initial_representation']
        for result in optimization_results
    ])
    e.log(f'initial representations: {initial_representations.shape}')

    # Statistical Analysis
    e.log('\ncomputing statistical metrics...')

    # 1. Norm analysis (L1 and L2 norms)
    dataset_l2_norms = np.linalg.norm(all_dataset_features, axis=1)
    optimized_l2_norms = np.linalg.norm(optimized_representations, axis=1)
    initial_l2_norms = np.linalg.norm(initial_representations, axis=1)

    dataset_l1_norms = np.linalg.norm(all_dataset_features, ord=1, axis=1)
    optimized_l1_norms = np.linalg.norm(optimized_representations, ord=1, axis=1)
    initial_l1_norms = np.linalg.norm(initial_representations, ord=1, axis=1)

    e.log(f'L2 norms - dataset: μ={dataset_l2_norms.mean():.3f}, σ={dataset_l2_norms.std():.3f}')
    e.log(f'L2 norms - optimized: μ={optimized_l2_norms.mean():.3f}, σ={optimized_l2_norms.std():.3f}')
    e.log(f'L2 norms - initial: μ={initial_l2_norms.mean():.3f}, σ={initial_l2_norms.std():.3f}')

    # 2. Dimension-wise statistics
    dataset_mean_per_dim = all_dataset_features.mean(axis=0)
    dataset_std_per_dim = all_dataset_features.std(axis=0)
    optimized_mean_per_dim = optimized_representations.mean(axis=0)
    optimized_std_per_dim = optimized_representations.std(axis=0)

    # 3. Nearest-neighbor distances: optimized to dataset
    from scipy.spatial.distance import cdist
    e.log('\ncomputing nearest-neighbor distances...')
    distances_optimized_to_dataset = cdist(
        optimized_representations,
        all_dataset_features,
        metric='cityblock'  # Manhattan distance (L1)
    )
    nn_distances_optimized = distances_optimized_to_dataset.min(axis=1)
    e.log(f'NN distances (optimized → dataset): μ={nn_distances_optimized.mean():.3f}, σ={nn_distances_optimized.std():.3f}')
    e.log(f'NN distances (optimized → dataset): min={nn_distances_optimized.min():.3f}, max={nn_distances_optimized.max():.3f}')

    # 4. Statistical tests
    from scipy.stats import ks_2samp, wasserstein_distance
    e.log('\nperforming statistical tests...')

    # KS test on L2 norms
    ks_stat_l2, ks_pvalue_l2 = ks_2samp(dataset_l2_norms, optimized_l2_norms)
    e.log(f'KS test (L2 norms): statistic={ks_stat_l2:.4f}, p-value={ks_pvalue_l2:.4f}')

    # Wasserstein distance on L2 norms
    wasserstein_dist_l2 = wasserstein_distance(dataset_l2_norms, optimized_l2_norms)
    e.log(f'Wasserstein distance (L2 norms): {wasserstein_dist_l2:.4f}')

    # Per-dimension KS tests (sample first 10 dimensions for logging)
    dimension_ks_stats = []
    for dim in range(all_dataset_features.shape[1]):
        ks_stat, ks_pval = ks_2samp(
            all_dataset_features[:, dim],
            optimized_representations[:, dim]
        )
        dimension_ks_stats.append({
            'dimension': dim,
            'ks_statistic': ks_stat,
            'ks_pvalue': ks_pval
        })

    # Average KS statistic across all dimensions
    avg_ks_stat = np.mean([d['ks_statistic'] for d in dimension_ks_stats])
    avg_ks_pval = np.mean([d['ks_pvalue'] for d in dimension_ks_stats])
    e.log(f'Average per-dimension KS: statistic={avg_ks_stat:.4f}, p-value={avg_ks_pval:.4f}')

    # Jensen-Shannon divergence (compute on binned L2 norms)
    from scipy.spatial.distance import jensenshannon
    # Use density=False to get counts, then normalize manually
    hist_dataset, bin_edges = np.histogram(dataset_l2_norms, bins=50, density=False)
    hist_optimized, _ = np.histogram(optimized_l2_norms, bins=bin_edges, density=False)

    # Normalize to create probability distributions (avoiding division by zero)
    hist_dataset = hist_dataset.astype(float)
    hist_optimized = hist_optimized.astype(float)

    if hist_dataset.sum() > 0:
        hist_dataset = hist_dataset / hist_dataset.sum()
    if hist_optimized.sum() > 0:
        hist_optimized = hist_optimized / hist_optimized.sum()

    # Add small epsilon to avoid log(0) issues in jensenshannon
    eps = 1e-10
    hist_dataset = hist_dataset + eps
    hist_optimized = hist_optimized + eps
    # Renormalize after adding epsilon
    hist_dataset = hist_dataset / hist_dataset.sum()
    hist_optimized = hist_optimized / hist_optimized.sum()

    js_divergence = jensenshannon(hist_dataset, hist_optimized)
    e.log(f'Jensen-Shannon divergence (L2 norms): {js_divergence:.4f}')

    # == ARTIFACT 1: PCA DENSITY OVERLAY WITH KDE ==

    e.log('\n=== CREATING ARTIFACT 1: PCA DENSITY OVERLAY ===')

    # Fit PCA on combined dataset + optimized representations
    # Note: PCA is already imported at the top of the file
    combined_representations = np.vstack([all_dataset_features, optimized_representations])
    pca_combined = PCA(n_components=2, random_state=e.SEED)
    combined_pca = pca_combined.fit_transform(combined_representations)

    # Split back into dataset and optimized
    n_dataset = len(all_dataset_features)
    dataset_pca = combined_pca[:n_dataset]
    optimized_pca = combined_pca[n_dataset:]

    e.log(f'PCA explained variance: {pca_combined.explained_variance_ratio_.sum():.4f}')

    # Fit KDE on dataset representations
    from sklearn.neighbors import KernelDensity
    kde = KernelDensity(bandwidth=0.5, kernel='gaussian')
    kde.fit(dataset_pca)

    # Evaluate KDE on grid
    x_min, x_max = combined_pca[:, 0].min() - 1, combined_pca[:, 0].max() + 1
    y_min, y_max = combined_pca[:, 1].min() - 1, combined_pca[:, 1].max() + 1
    xx_kde, yy_kde = np.meshgrid(
        np.linspace(x_min, x_max, 100),
        np.linspace(y_min, y_max, 100)
    )
    grid_points = np.c_[xx_kde.ravel(), yy_kde.ravel()]
    log_density = kde.score_samples(grid_points)
    density = np.exp(log_density).reshape(xx_kde.shape)

    # Evaluate KDE density at optimized points
    optimized_log_density = kde.score_samples(optimized_pca)
    optimized_density = np.exp(optimized_log_density)

    e.log(f'optimized representations density: μ={optimized_density.mean():.6f}, '
          f'min={optimized_density.min():.6f}, max={optimized_density.max():.6f}')

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))

    # Plot density contours for dataset
    contour_levels = 10
    contourf = ax.contourf(xx_kde, yy_kde, density, levels=contour_levels, cmap='Blues', alpha=0.6)
    contour_lines = ax.contour(xx_kde, yy_kde, density, levels=contour_levels, colors='blue',
                                alpha=0.3, linewidths=0.8)
    ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%.2e')

    # Scatter plot of dataset representations (subsample for clarity)
    n_subsample = min(1000, len(dataset_pca))
    subsample_indices = np.random.RandomState(e.SEED).choice(len(dataset_pca), n_subsample, replace=False)
    ax.scatter(
        dataset_pca[subsample_indices, 0],
        dataset_pca[subsample_indices, 1],
        c='lightblue',
        alpha=0.3,
        s=20,
        label=f'Dataset (n={len(all_dataset_features)})',
        edgecolors='none'
    )

    # Scatter plot of optimized representations, colored by success
    # Color by improvement achieved
    improvements = [result['initial_distance_to_target'] - result['true_distance_to_target']
                   for result in optimization_results]
    scatter_opt = ax.scatter(
        optimized_pca[:, 0],
        optimized_pca[:, 1],
        c=improvements,
        cmap='RdYlGn',
        s=150,
        alpha=0.8,
        edgecolors='black',
        linewidth=1.5,
        label=f'Optimized (n={len(optimized_representations)})',
        zorder=10
    )

    cbar = plt.colorbar(scatter_opt, ax=ax)
    cbar.set_label('Improvement (initial dist - final dist)', fontsize=11)

    # Add density colorbar
    cbar_density = plt.colorbar(contourf, ax=ax, pad=0.1)
    cbar_density.set_label('Dataset Density (KDE)', fontsize=11)

    ax.set_xlabel(f'PC1 ({pca_combined.explained_variance_ratio_[0]:.2%} variance)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca_combined.explained_variance_ratio_[1]:.2%} variance)', fontsize=12)
    ax.set_title(
        f'Representation Distribution: Dataset vs Optimized\n'
        f'Blue contours: KDE density of dataset representations\n'
        f'Colored points: Optimized representations (color = improvement)',
        fontsize=14
    )
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    e.commit_fig('representation_density_overlay.png', fig)
    e.log('saved representation density overlay')

    # == ARTIFACT 2: MULTI-METRIC DISTRIBUTION COMPARISON DASHBOARD ==

    e.log('\n=== CREATING ARTIFACT 2: DISTRIBUTION COMPARISON DASHBOARD ===')

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    # Panel A: L2 norm histogram comparison
    ax_a = axes[0, 0]
    bins_l2 = np.linspace(
        min(dataset_l2_norms.min(), optimized_l2_norms.min()),
        max(dataset_l2_norms.max(), optimized_l2_norms.max()),
        50
    )
    ax_a.hist(dataset_l2_norms, bins=bins_l2, alpha=0.6, label='Dataset', color='blue', edgecolor='black')
    ax_a.hist(optimized_l2_norms, bins=bins_l2, alpha=0.6, label='Optimized', color='red', edgecolor='black')
    ax_a.hist(initial_l2_norms, bins=bins_l2, alpha=0.4, label='Initial', color='green',
              edgecolor='black', linestyle='--')
    ax_a.axvline(dataset_l2_norms.mean(), color='blue', linestyle='--', linewidth=2,
                 label=f'Dataset μ={dataset_l2_norms.mean():.2f}')
    ax_a.axvline(optimized_l2_norms.mean(), color='red', linestyle='--', linewidth=2,
                 label=f'Optimized μ={optimized_l2_norms.mean():.2f}')
    ax_a.set_xlabel('L2 Norm', fontsize=11)
    ax_a.set_ylabel('Frequency', fontsize=11)
    ax_a.set_title('Panel A: L2 Norm Distribution Comparison', fontsize=12, fontweight='bold')
    ax_a.legend(loc='best', fontsize=9)
    ax_a.grid(True, alpha=0.3)

    # Panel B: Per-dimension statistics comparison
    ax_b = axes[0, 1]
    # Sample dimensions for visualization (too many to show all)
    n_dims_to_show = min(20, all_dataset_features.shape[1])
    dim_indices = np.linspace(0, all_dataset_features.shape[1]-1, n_dims_to_show, dtype=int)

    x_dims = np.arange(len(dim_indices))
    width = 0.35

    dataset_means_sample = dataset_mean_per_dim[dim_indices]
    optimized_means_sample = optimized_mean_per_dim[dim_indices]

    ax_b.bar(x_dims - width/2, dataset_means_sample, width, label='Dataset',
             alpha=0.7, color='blue', edgecolor='black')
    ax_b.bar(x_dims + width/2, optimized_means_sample, width, label='Optimized',
             alpha=0.7, color='red', edgecolor='black')

    ax_b.set_xlabel(f'Dimension Index (sampled {n_dims_to_show}/{all_dataset_features.shape[1]})', fontsize=11)
    ax_b.set_ylabel('Mean Value', fontsize=11)
    ax_b.set_title('Panel B: Per-Dimension Mean Comparison', fontsize=12, fontweight='bold')
    ax_b.set_xticks(x_dims)
    ax_b.set_xticklabels(dim_indices, rotation=45)
    ax_b.legend(loc='best', fontsize=9)
    ax_b.grid(True, alpha=0.3, axis='y')

    # Panel C: Nearest-neighbor distance comparison
    ax_c = axes[1, 0]

    # Compute typical NN distances within dataset (subsample for efficiency)
    n_sample_nn = min(500, len(all_dataset_features))
    sample_indices_nn = np.random.RandomState(e.SEED).choice(len(all_dataset_features), n_sample_nn, replace=False)
    sampled_dataset = all_dataset_features[sample_indices_nn]

    distances_within_dataset = cdist(sampled_dataset, all_dataset_features, metric='cityblock')
    # For each sample, find distance to nearest neighbor (excluding self)
    # Set self-distances to infinity (can't use fill_diagonal with random indices)
    for i, idx in enumerate(sample_indices_nn):
        distances_within_dataset[i, idx] = np.inf
    nn_distances_within_dataset = distances_within_dataset.min(axis=1)

    # Box plot data
    box_data = [
        nn_distances_within_dataset,
        nn_distances_optimized,
        avg_5nn_distance * np.ones(len(nn_distances_optimized))  # Reference line
    ]

    labels_box = ['Dataset NN\n(intra-dataset)', 'Optimized NN\n(to dataset)', '5-NN Baseline\n(reference)']
    colors_box = ['blue', 'red', 'gold']

    bp = ax_c.boxplot(
        box_data,
        labels=labels_box,
        patch_artist=True,
        widths=0.6,
        showmeans=True,
        meanprops=dict(marker='D', markerfacecolor='black', markeredgecolor='black', markersize=8)
    )

    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
        patch.set_edgecolor('black')
        patch.set_linewidth(1.5)

    ax_c.set_ylabel('Manhattan Distance (L1)', fontsize=11)
    ax_c.set_title('Panel C: Nearest-Neighbor Distance Comparison', fontsize=12, fontweight='bold')
    ax_c.grid(True, alpha=0.3, axis='y')

    # Add statistics text
    stats_text_c = (
        f'Dataset NN: μ={nn_distances_within_dataset.mean():.2f}\n'
        f'Optimized NN: μ={nn_distances_optimized.mean():.2f}\n'
        f'5-NN Baseline: {avg_5nn_distance:.2f}'
    )
    ax_c.text(
        0.02, 0.98, stats_text_c,
        transform=ax_c.transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
        fontsize=9,
        family='monospace'
    )

    # Panel D: Statistical test results with visual metric cards
    ax_d = axes[1, 1]
    ax_d.axis('off')
    ax_d.set_xlim(0, 1)
    ax_d.set_ylim(0, 1)

    # Calculate metric assessments and colors
    # Safety check to avoid division by zero
    nn_dataset_mean = nn_distances_within_dataset.mean()
    if nn_dataset_mean > 0:
        nn_ratio = nn_distances_optimized.mean() / nn_dataset_mean
    else:
        e.log('WARNING: nn_distances_within_dataset mean is zero, setting nn_ratio to inf')
        nn_ratio = np.inf

    # Define color scheme: green (good/similar), yellow (moderate), red (different)
    def get_metric_color(metric_type, value):
        """Return color based on metric type and value."""
        if metric_type == 'ks_pvalue':
            return '#4CAF50' if value > 0.05 else '#F44336'  # Green if not significant, red if significant
        elif metric_type == 'js_divergence':
            if value < 0.1:
                return '#4CAF50'  # Green: very similar
            elif value < 0.3:
                return '#FFC107'  # Yellow: moderate
            else:
                return '#F44336'  # Red: different
        elif metric_type == 'nn_ratio':
            if value < 1.5:
                return '#4CAF50'  # Green: close
            elif value < 3.0:
                return '#FFC107'  # Yellow: moderate
            else:
                return '#F44336'  # Red: far
        else:
            return '#2196F3'  # Blue: neutral

    # Calculate overall verdict
    score = 0
    if ks_pvalue_l2 > 0.05:
        score += 1
    if js_divergence < 0.3:
        score += 1
    if nn_ratio < 3.0:
        score += 1

    if score >= 2:
        verdict = "✓ DISTRIBUTIONS SIMILAR"
        verdict_color = '#4CAF50'
    elif score >= 1:
        verdict = "⚠ MODERATE DIFFERENCE"
        verdict_color = '#FFC107'
    else:
        verdict = "✗ SIGNIFICANT DIFFERENCE"
        verdict_color = '#F44336'

    # Define metrics to display (2 rows x 3 columns)
    metrics = [
        {
            'name': 'KS Test p-value',
            'value': f'{ks_pvalue_l2:.4f}',
            'color': get_metric_color('ks_pvalue', ks_pvalue_l2),
            'bar_value': min(ks_pvalue_l2, 1.0),  # Clamp to [0, 1]
            'interpretation': 'Not Sig.' if ks_pvalue_l2 > 0.05 else 'Significant'
        },
        {
            'name': 'JS Divergence',
            'value': f'{js_divergence:.4f}' if not np.isnan(js_divergence) else 'NaN',
            'color': get_metric_color('js_divergence', js_divergence) if not np.isnan(js_divergence) else '#FF5722',
            'bar_value': js_divergence if not np.isnan(js_divergence) else 0.0,
            'interpretation': 'Similar' if js_divergence < 0.3 else 'Different' if not np.isnan(js_divergence) else 'Error'
        },
        {
            'name': 'Wasserstein Dist',
            'value': f'{wasserstein_dist_l2:.2f}',
            'color': '#2196F3',
            'bar_value': None,  # No bar for this one
            'interpretation': 'Earth Mover\'s'
        },
        {
            'name': 'NN Distance Ratio',
            'value': f'{nn_ratio:.2f}x' if not np.isinf(nn_ratio) else 'inf',
            'color': get_metric_color('nn_ratio', nn_ratio),
            'bar_value': min(nn_ratio / 5.0, 1.0) if not np.isinf(nn_ratio) else 1.0,  # Normalize to [0, 1], cap at 5x
            'interpretation': 'Close' if nn_ratio < 1.5 else ('Moderate' if nn_ratio < 3.0 else 'Very Far')
        },
        {
            'name': 'Avg Per-Dim KS',
            'value': f'{avg_ks_stat:.4f}',
            'color': '#2196F3',
            'bar_value': None,
            'interpretation': f'p={avg_ks_pval:.3f}'
        },
        {
            'name': 'Optimized Density',
            'value': f'{optimized_density.mean():.2e}',
            'color': '#2196F3',
            'bar_value': None,
            'interpretation': 'Mean KDE'
        },
    ]

    # Draw metric cards in 2x3 grid
    card_width = 0.30
    card_height = 0.18
    x_spacing = 0.33
    y_spacing = 0.22
    x_start = 0.05
    y_start = 0.75

    for i, metric in enumerate(metrics):
        row = i // 3
        col = i % 3

        x = x_start + col * x_spacing
        y = y_start - row * y_spacing

        # Draw card background
        card_rect = plt.Rectangle(
            (x, y - card_height),
            card_width,
            card_height,
            facecolor='white',
            edgecolor='gray',
            linewidth=1.5,
            alpha=0.9,
            transform=ax_d.transAxes,
            zorder=1
        )
        ax_d.add_patch(card_rect)

        # Metric name (header)
        ax_d.text(
            x + card_width / 2,
            y - 0.02,
            metric['name'],
            transform=ax_d.transAxes,
            fontsize=9,
            fontweight='bold',
            ha='center',
            va='top'
        )

        # Value (large)
        ax_d.text(
            x + card_width / 2,
            y - 0.07,
            metric['value'],
            transform=ax_d.transAxes,
            fontsize=14,
            fontweight='bold',
            ha='center',
            va='center',
            color=metric['color']
        )

        # Draw progress bar if applicable
        if metric['bar_value'] is not None:
            bar_y = y - 0.12
            bar_height = 0.02
            bar_width_full = card_width - 0.04

            # Background bar (gray)
            bg_bar = plt.Rectangle(
                (x + 0.02, bar_y),
                bar_width_full,
                bar_height,
                facecolor='#E0E0E0',
                edgecolor='none',
                transform=ax_d.transAxes,
                zorder=2
            )
            ax_d.add_patch(bg_bar)

            # Filled bar (colored)
            filled_bar = plt.Rectangle(
                (x + 0.02, bar_y),
                bar_width_full * metric['bar_value'],
                bar_height,
                facecolor=metric['color'],
                edgecolor='none',
                transform=ax_d.transAxes,
                zorder=3
            )
            ax_d.add_patch(filled_bar)

        # Interpretation (small text)
        ax_d.text(
            x + card_width / 2,
            y - card_height + 0.02,
            metric['interpretation'],
            transform=ax_d.transAxes,
            fontsize=8,
            ha='center',
            va='bottom',
            style='italic',
            color='#555555'
        )

    # Draw verdict box at bottom
    verdict_y = 0.28
    verdict_height = 0.08
    verdict_rect = plt.Rectangle(
        (0.05, verdict_y - verdict_height),
        0.90,
        verdict_height,
        facecolor=verdict_color,
        edgecolor='black',
        linewidth=2,
        alpha=0.3,
        transform=ax_d.transAxes,
        zorder=1
    )
    ax_d.add_patch(verdict_rect)

    ax_d.text(
        0.5,
        verdict_y - verdict_height / 2,
        verdict,
        transform=ax_d.transAxes,
        fontsize=14,
        fontweight='bold',
        ha='center',
        va='center',
        color=verdict_color
    )

    # Draw interpretation summary at bottom
    interpretation_lines = []
    if ks_pvalue_l2 > 0.05:
        interpretation_lines.append("• KS test: Distributions NOT significantly different (p > 0.05)")
    else:
        interpretation_lines.append("• KS test: Distributions ARE significantly different (p ≤ 0.05)")

    if js_divergence < 0.1:
        interpretation_lines.append("• JS divergence: Very similar (< 0.1)")
    elif js_divergence < 0.3:
        interpretation_lines.append("• JS divergence: Moderately similar (< 0.3)")
    else:
        interpretation_lines.append("• JS divergence: Quite different (≥ 0.3)")

    if nn_ratio < 1.5:
        interpretation_lines.append("• NN distance: Optimized reps close to dataset (< 1.5x)")
    elif nn_ratio < 3.0:
        interpretation_lines.append("• NN distance: Moderately far from dataset (< 3x)")
    else:
        interpretation_lines.append("• NN distance: Far from dataset (≥ 3x)")

    interpretation_text = '\n'.join(interpretation_lines)

    ax_d.text(
        0.5,
        0.10,
        interpretation_text,
        transform=ax_d.transAxes,
        fontsize=9,
        ha='center',
        va='center',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'),
        family='sans-serif'
    )

    ax_d.set_title('Panel D: Distribution Similarity Assessment',
                   fontsize=12, fontweight='bold', y=0.98)

    plt.tight_layout()
    e.commit_fig('distribution_comparison_dashboard.png', fig)
    e.log('saved distribution comparison dashboard')

    # == ARTIFACT 3: OPTIMIZATION TRAJECTORY DENSITY HEATMAP ==

    e.log('\n=== CREATING ARTIFACT 3: TRAJECTORY DENSITY HEATMAP ===')

    fig, ax = plt.subplots(figsize=(16, 12))

    # Plot KDE density heatmap as background
    heatmap = ax.contourf(xx_kde, yy_kde, density, levels=20, cmap='YlOrRd', alpha=0.6)
    contour_lines = ax.contour(xx_kde, yy_kde, density, levels=10, colors='black',
                                alpha=0.25, linewidths=0.6)

    # Overlay optimization trajectories
    e.log(f'plotting {len(optimization_results)} optimization trajectories...')

    for opt_idx, result in enumerate(optimization_results):
        # Get initial and final representations in PCA space
        initial_rep = result['initial_representation']
        optimized_rep = result['optimized_representation']

        # Transform to PCA space
        initial_pca_pt = pca_combined.transform(initial_rep.reshape(1, -1))[0]
        optimized_pca_pt = pca_combined.transform(optimized_rep.reshape(1, -1))[0]

        # Get intermediate trajectory points if available
        # Note: trajectory_closest contains indices of closest molecules, not raw reps
        # We'll use a simple linear path from initial to optimized for visualization
        trajectory_steps = 10
        trajectory_path = np.linspace(initial_pca_pt, optimized_pca_pt, trajectory_steps)

        # Color by improvement
        improvement = result['initial_distance_to_target'] - result['true_distance_to_target']

        # Normalize improvement for coloring
        improvements_array = np.array([r['initial_distance_to_target'] - r['true_distance_to_target']
                                       for r in optimization_results])
        norm_improvement = (improvement - improvements_array.min()) / (improvements_array.max() - improvements_array.min() + 1e-8)

        # Color map: red (poor) to green (good)
        cmap_trajectory = plt.cm.RdYlGn
        trajectory_color = cmap_trajectory(norm_improvement)

        # Plot trajectory line
        ax.plot(
            trajectory_path[:, 0],
            trajectory_path[:, 1],
            color=trajectory_color,
            alpha=0.5,
            linewidth=2,
            zorder=5
        )

        # Mark initial point (blue circle)
        ax.scatter(
            initial_pca_pt[0],
            initial_pca_pt[1],
            marker='o',
            s=80,
            color='blue',
            edgecolors='black',
            linewidth=1,
            zorder=8,
            alpha=0.7,
            label='Initial' if opt_idx == 0 else None
        )

        # Mark final point (green square)
        ax.scatter(
            optimized_pca_pt[0],
            optimized_pca_pt[1],
            marker='s',
            s=80,
            color=trajectory_color,
            edgecolors='black',
            linewidth=1,
            zorder=8,
            alpha=0.9,
            label='Optimized' if opt_idx == 0 else None
        )

    # Density colorbar
    cbar_heat = plt.colorbar(heatmap, ax=ax, pad=0.02)
    cbar_heat.set_label('Dataset Density (KDE)', fontsize=12)

    # Annotate density regions
    density_percentiles = np.percentile(density.ravel(), [25, 50, 75])
    ax.text(
        0.02, 0.98,
        f'Density Levels:\n'
        f'Low: < {density_percentiles[0]:.2e}\n'
        f'Med: {density_percentiles[0]:.2e} - {density_percentiles[2]:.2e}\n'
        f'High: > {density_percentiles[2]:.2e}',
        transform=ax.transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
        fontsize=10,
        family='monospace'
    )

    ax.set_xlabel(f'PC1 ({pca_combined.explained_variance_ratio_[0]:.2%} variance)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca_combined.explained_variance_ratio_[1]:.2%} variance)', fontsize=12)
    ax.set_title(
        f'Optimization Trajectories on Dataset Density Landscape\n'
        f'Background: KDE density heatmap | Paths: Initial (blue •) → Optimized (colored ■)\n'
        f'Path color: improvement achieved (red=poor, green=good)',
        fontsize=14
    )
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    e.commit_fig('optimization_trajectory_density.png', fig)
    e.log('saved optimization trajectory density heatmap')

    # == SAVE STATISTICAL RESULTS TO CSV ==

    e.log('\n=== SAVING STATISTICAL RESULTS ===')

    statistical_results = [
        {'metric_name': 'ks_statistic_l2_norms', 'value': ks_stat_l2,
         'interpretation': f'KS test statistic for L2 norm distributions'},
        {'metric_name': 'ks_pvalue_l2_norms', 'value': ks_pvalue_l2,
         'interpretation': f'p-value: {"not significant" if ks_pvalue_l2 > 0.05 else "significant"} (α=0.05)'},
        {'metric_name': 'wasserstein_distance_l2', 'value': wasserstein_dist_l2,
         'interpretation': 'Earth mover\'s distance between L2 norm distributions'},
        {'metric_name': 'jensen_shannon_divergence_l2', 'value': js_divergence,
         'interpretation': f'JS divergence (0=identical, 1=orthogonal): {"similar" if js_divergence < 0.3 else "different"}'},
        {'metric_name': 'avg_dimension_ks_statistic', 'value': avg_ks_stat,
         'interpretation': 'Average KS statistic across all dimensions'},
        {'metric_name': 'avg_dimension_ks_pvalue', 'value': avg_ks_pval,
         'interpretation': 'Average p-value across all dimensions'},
        {'metric_name': 'dataset_l2_mean', 'value': dataset_l2_norms.mean(),
         'interpretation': 'Mean L2 norm of dataset representations'},
        {'metric_name': 'dataset_l2_std', 'value': dataset_l2_norms.std(),
         'interpretation': 'Std dev of L2 norm of dataset representations'},
        {'metric_name': 'optimized_l2_mean', 'value': optimized_l2_norms.mean(),
         'interpretation': 'Mean L2 norm of optimized representations'},
        {'metric_name': 'optimized_l2_std', 'value': optimized_l2_norms.std(),
         'interpretation': 'Std dev of L2 norm of optimized representations'},
        {'metric_name': 'nn_distance_optimized_mean', 'value': nn_distances_optimized.mean(),
         'interpretation': 'Mean NN distance from optimized to dataset'},
        {'metric_name': 'nn_distance_optimized_std', 'value': nn_distances_optimized.std(),
         'interpretation': 'Std dev of NN distances from optimized to dataset'},
        {'metric_name': 'nn_distance_within_dataset_mean', 'value': nn_distances_within_dataset.mean(),
         'interpretation': 'Mean NN distance within dataset (intra-dataset)'},
        {'metric_name': 'nn_distance_ratio', 'value': nn_ratio,
         'interpretation': 'Ratio of optimized NN distance to dataset NN distance'},
        {'metric_name': 'optimized_kde_density_mean', 'value': optimized_density.mean(),
         'interpretation': 'Mean KDE density at optimized representation locations'},
        {'metric_name': 'optimized_kde_density_min', 'value': optimized_density.min(),
         'interpretation': 'Minimum KDE density at optimized representation locations'},
        {'metric_name': 'optimized_kde_density_max', 'value': optimized_density.max(),
         'interpretation': 'Maximum KDE density at optimized representation locations'},
    ]

    stats_df = pd.DataFrame(statistical_results)
    stats_path = os.path.join(e.path, 'representation_distribution_analysis.csv')
    stats_df.to_csv(stats_path, index=False)
    e.log(f'saved statistical results to {stats_path}')

    # Store key metrics in experiment metadata
    e['distribution_analysis/ks_pvalue_l2'] = float(ks_pvalue_l2)
    e['distribution_analysis/wasserstein_distance_l2'] = float(wasserstein_dist_l2)
    e['distribution_analysis/js_divergence_l2'] = float(js_divergence)
    e['distribution_analysis/nn_distance_ratio'] = float(nn_ratio)
    e['distribution_analysis/optimized_kde_density_mean'] = float(optimized_density.mean())

    e.log('\nrepresentation distribution analysis complete!')

    e.log('\nexperiment complete!')


experiment.run_if_main()
