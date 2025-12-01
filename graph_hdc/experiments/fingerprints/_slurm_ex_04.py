"""
SLURM Job Scheduler for Experiment 04: Data Efficiency Ablation Study

This script schedules a comprehensive ablation study that systematically evaluates the
data efficiency of different molecular encoding methods across varying training dataset
sizes. The primary goal is to compare how encoding methods (HDC, fingerprints, GNNs)
perform when trained on limited data, with the hypothesis that simpler encoding methods
may be more data-efficient than graph neural networks.

**Experiment Overview:**
    The ablation study explores the performance-data trade-off by training models on
    progressively larger subsets of the training data. This reveals which encoding methods
    can achieve good performance with fewer training examples, an important consideration
    for domains where labeled data is scarce or expensive to obtain.

**Parameter Space:**
    - Encoding methods: GNN (Graph Neural Networks), HDC (Hyperdimensional Computing), FP (Fingerprints)
    - Prediction models: GATv2, Neural Network (MLP), k-Nearest Neighbors
    - Training dataset sizes: Range from 10 to 100,000 molecules
    - Random seeds: Multiple seeds for statistical robustness
    - Fixed hyperparameters: Embedding sizes, depths, and model configurations

**Usage:**
    Before running this script, ensure that:
    1. The auto_slurm package is installed and configured
    2. SLURM cluster access is available
    3. The target experiment modules exist in the same directory

    To schedule the jobs:

    .. code-block:: bash

        python _slurm_ex_04.py

    The script will generate and submit all job combinations to the SLURM scheduler.

**Output:**
    Each experiment will generate its own archive with results stored using the PyComex framework.
    Archives are prefixed with the value specified in PREFIX for easy identification.

:note: The DRY_RUN flag in the submitter controls whether jobs are actually submitted or just validated.
:note: Adjust AUTOSLURM_CONFIG to match your specific cluster configuration.
"""
# ====================================================================
# IMPORTS
# ====================================================================

import os
import pathlib
from itertools import product
from rich.pretty import pprint
import subprocess
from tqdm import tqdm
from auto_slurm.aslurmx import ASlurmSubmitter


# ====================================================================
# EXPERIMENT CONFIGURATION
# ====================================================================
# This section defines the basic configuration parameters that apply
# to all experiments in this ablation study.

PATH = pathlib.Path(__file__).parent.absolute()

# The autoslurm config file to be used for the creation of the SLURM jobs.
# This should match a configuration profile set up in the auto_slurm package.
AUTOSLURM_CONFIG = 'euler_2'

# The random seeds to be used for the random number generator and more importantly
# for the dataset splitting. Multiple seeds enable statistical robustness by averaging
# results across different random initializations.
SEED: int = 0  # Legacy single seed (currently unused)

#SEEDS: list[int] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
SEEDS: list[int] = [0, 1, 2, 3, 4]

# The prefix for the experiment archives so that they can be easily identified later
# on. All generated experiment archives will start with this prefix.
PREFIX: str = 'ex_04_al'


# ====================================================================
# DATASET CONFIGURATION
# ====================================================================
# This section specifies the molecular dataset to be used for the
# ablation study and any dataset-specific parameters.

# The name of the dataset to be used for the data efficiency ablation study.
# This determines which molecular property prediction task is being evaluated.
DATASET_NAME: str = 'qm9_smiles'

# Combinations of encoding methods and prediction models to evaluate.
# Each tuple specifies (encoding_method, prediction_model):
#   - 'gnn': Graph neural networks (e.g., GATv2, GIN)
#   - 'hdc': Hyperdimensional computing-based graph encoding
#   - 'fp': Traditional molecular fingerprints (e.g., Morgan/ECFP)
#   - 'gatv2': Graph Attention Network v2 for graph-level predictions
#   - 'neural_net': Multi-layer perceptron (MLP) for regression
#   - 'k_neighbors': k-Nearest Neighbors algorithm
# The data efficiency ablation will be performed for each of these combinations.
ENCODING_METHOD_TUPLES: list[tuple[str, str]] = [
    ('gnn', 'gatv2'),
    #('gnn', 'gin'),
    ('hdc', 'neural_net'),
    #('hdc', 'random_forest'),
    ('fp', 'neural_net'),
    #('fp', 'random_forest'),
    ('hdc', 'k_neighbors'),
    ('fp', 'k_neighbors'),
]

# Fixed parameters that will be passed to all experiments regardless of encoding
# method or model type. This provides an opportunity to set dataset-specific
# configurations or override default experiment parameters.
PARAMETERS = {
    'NUM_DATA': 1.0,                # Use the entire dataset (full data pool for sampling)
    'NOTE': 'qm9_heat',              # Dataset identifier for experiment tracking
    'TARGET_INDEX': 14,             # Target property index (10 = HOMO-LUMO gap for QM9)
    'DATASET_NAME': 'qm9_smiles',   # Dataset name for loading
    'DATASET_TYPE': 'regression',   # Task type: regression vs classification
}


# ====================================================================
# ENCODING METHOD CONFIGURATIONS
# ====================================================================
# This section defines the fixed hyperparameters and method-specific
# configurations for each encoding approach. These parameters are held
# constant across the data efficiency ablation study.

# Fixed number of layers/radius for all encoding methods.
# For HDC, this is the number of message passing layers.
# For fingerprints, this is the radius (bond distance).
# For GNNs, this affects the receptive field of the network.
NUM_LAYERS = 3

# Method-specific parameter configurations that remain constant across the ablation.
# Each encoding method has its own set of hyperparameters for the prediction models
# (Random Forest, Neural Network, k-NN) and method-specific settings (embedding sizes,
# fingerprint types, etc.). These are fixed to isolate the effect of training data size.
ENCODING_PARAMETERS_MAP: dict[str, dict] = {
    'gnn': {
        # Graph Neural Network hyperparameters:
        'EPOCHS': 150,                              # Training epochs for GNN baseline
    },
    'hdc': {
        # Random Forest hyperparameters:
        'RF_NUM_ESTIMATORS': 100,                   # Number of trees in the forest
        'RF_MAX_DEPTH': 10,                         # Maximum depth of each tree
        'RF_MAX_FEATURES': 'sqrt',                  # Number of features for split
        # Neural Network hyperparameters:
        'NN_HIDDEN_LAYER_SIZES': (100, 100),        # Two hidden layers with 100 neurons each
        'NN_ALPHA': 0.001,                          # L2 regularization strength
        'NN_LEARNING_RATE_INIT': 0.001,             # Initial learning rate
        # HDC-specific settings:
        'EMBEDDING_SIZE': 2048,                     # Hypervector dimension (fixed)
        'NUM_LAYERS': NUM_LAYERS,                   # Number of message passing layers
    },
    'fp': {
        # Random Forest hyperparameters:
        'RF_NUM_ESTIMATORS': 100,                   # Number of trees in the forest
        'RF_MAX_DEPTH': 10,                         # Maximum depth of each tree
        'RF_MAX_FEATURES': 'sqrt',                  # Number of features for split
        # Neural Network hyperparameters:
        'NN_HIDDEN_LAYER_SIZES': (100, 100),        # Two hidden layers with 100 neurons each
        'NN_ALPHA': 0.001,                          # L2 regularization strength
        'NN_LEARNING_RATE_INIT': 0.001,             # Initial learning rate
        # Fingerprint-specific settings:
        'FINGERPRINT_SIZE': 2048,                   # Bit vector length (fixed)
        'FINGERPRINT_RADIUS': NUM_LAYERS,           # Radius for Morgan fingerprint
        'FINGERPRINT_TYPE': 'morgan',               # Type of molecular fingerprint (Morgan/ECFP)
    },
}


# ====================================================================
# ABLATION STUDY PARAMETERS
# ====================================================================
# This section defines the parameter sweep ranges for the data efficiency
# ablation study. The primary variable being explored is training dataset
# size, with all other hyperparameters held constant.

# Training dataset size sweep: the number of training examples to use.
# This is the core ablation variable for studying data efficiency. The sweep
# ranges from very small training sets (10 molecules) to large training sets
# (100,000+ molecules), allowing us to observe:
#   - Which encoding methods can learn from limited data
#   - The rate of improvement as training data increases
#   - Whether GNNs require more data than simpler encoding methods
#   - Potential saturation points where additional data provides diminishing returns
# Values can be integers (absolute counts) or floats (fractions of the dataset).
NUM_TRAIN_SWEEP: list[int | float] = [
    10,         # Extremely limited data regime
    50,
    100,
    250,
    500,
    1000,       # Low data regime
    2500,
    5000,
    7500,
    10_000,     # Medium data regime
    25_000,
    50_000,
    100_000,    # High data regime
#    150_000,    # Uncomment for very large dataset experiments
#    200_000,
]


# ====================================================================
# EXPERIMENT STATISTICS
# ====================================================================
# Calculate the total number of experiments that will be generated
# by taking the Cartesian product of all parameter combinations.

num_experiments = len(ENCODING_METHOD_TUPLES) * len(NUM_TRAIN_SWEEP) * len(SEEDS)
print(f'Preparing to schedule {num_experiments} experiments for the ablation study.')


# ====================================================================
# SLURM JOB SUBMISSION
# ====================================================================
# This section handles the generation and submission of SLURM jobs
# for all parameter combinations in the data efficiency ablation study.

# Initialize the SLURM submitter with the specified configuration.
# batch_size=1: Submit jobs one at a time (can be increased for efficiency)
# randomize=True: Randomize job submission order to avoid systematic biases
submitter = ASlurmSubmitter(
    config_name=AUTOSLURM_CONFIG,
    batch_size=1,
    randomize=True,
)

# Generate and submit all experiment combinations.
# This nested loop iterates through all possible combinations of:
# encoding methods, models, training dataset sizes, and random seeds.
with tqdm(total=num_experiments, desc="Scheduling experiments") as pbar:

    # Iterate over each (encoding method, prediction model) combination
    for (encoding, model) in ENCODING_METHOD_TUPLES:

        # Iterate over each training dataset size in the sweep
        for num_train in NUM_TRAIN_SWEEP:

            # Iterate over each random seed for statistical robustness
            for seed in SEEDS:

                # --------------------------------------------------------
                # Assemble Python Command for SLURM Execution
                # --------------------------------------------------------
                # Build the Python command that will be executed within each
                # SLURM job. The command calls the appropriate experiment module
                # with encoding-specific parameters and the current training size.

                # Construct the experiment module filename based on encoding method and dataset.
                # First, try the dataset-specific YAML config (e.g., predict_molecules__hdc__qm9_smiles.yml).
                # If it doesn't exist, try the Python file (e.g., predict_molecules__hdc__qm9_smiles.py).
                # Finally, fall back to the generic version (e.g., predict_molecules__hdc.py).

                # Try dataset-specific YAML config first
                experiment_module = f'predict_molecules__{encoding}__{DATASET_NAME}.yml'
                experiment_path = os.path.join(PATH, experiment_module)

                # If YAML doesn't exist, try dataset-specific Python file
                if not os.path.exists(experiment_path):
                    experiment_module = f'predict_molecules__{encoding}__{DATASET_NAME}.py'
                    experiment_path = os.path.join(PATH, experiment_module)

                # If dataset-specific Python file doesn't exist, fall back to base encoding module
                if not os.path.exists(experiment_path):
                    experiment_module = f'predict_molecules__{encoding}.py'
                    experiment_path = os.path.join(PATH, experiment_module)

                # Build the command line arguments list.
                # Using repr() ensures proper Python string formatting in the command.
                python_command_list = [
                    'python',
                    experiment_path,
                    f'--__DEBUG__=False ',                                          # Disable debug mode (full dataset)
                    f'--__PREFIX__="{repr(PREFIX)}" ',                              # Set experiment archive prefix
                    f'--SEED="{seed}" ',                                            # Random seed for reproducibility
                    f'--MODELS="{repr([model])}" ',                                 # Prediction model to use
                    f'--NUM_TEST="{repr(0.1)}" ',                                   # 10% test set size (fixed)
                    f'--NUM_TRAIN="{repr(num_train)}" ',                            # Number of training examples (ABLATION VARIABLE)
                ]

                # Merge the global fixed parameters with encoding-specific parameters.
                # This creates a complete parameter dictionary for this specific encoding method.
                param_dict = PARAMETERS.copy()                                      # Start with global parameters
                param_dict.update(ENCODING_PARAMETERS_MAP[encoding])                # Add encoding-specific parameters

                # Append all parameters to the command as command-line arguments
                for key, value in param_dict.items():
                    python_command_list.append(f'--{key}="{repr(value)}"')

                # Join all command parts into a single string
                python_command_string = ' '.join(python_command_list)

                # Add the command to the SLURM submitter queue
                submitter.add_command(python_command_string)

                # Update the progress bar
                pbar.update(1)

    # Submit all queued jobs to the SLURM scheduler
    print(f'Submitting {submitter.count_jobs()} jobs to SLURM...')
    submitter.submit() 
