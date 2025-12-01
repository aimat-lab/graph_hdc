"""
SLURM Job Scheduler for Experiment 03: Embedding Size and Depth Ablation Study

This script schedules a comprehensive ablation study that systematically evaluates the impact
of embedding dimensions and network depths on molecular property prediction performance across
two encoding paradigms: traditional molecular fingerprints (FP) and hyperdimensional computing (HDC).

**Experiment Overview:**
    The ablation study explores the performance trade-offs between embedding size (dimensionality
    of the molecular representation) and embedding depth (number of layers/radius for message
    passing or fingerprint generation). This helps identify optimal hyperparameters for both
    encoding methods.

**Parameter Space:**
    - Encoding methods: Fingerprints (FP) and Hyperdimensional Computing (HDC)
    - Prediction models: Random Forest and Neural Network (MLP)
    - Embedding sizes: Range from 8 to 16,384 dimensions
    - Embedding depths: 1 and 3 layers (for HDC) or radius (for FP)
    - Random seeds: Multiple seeds for statistical robustness

**Usage:**
    Before running this script, ensure that:
    1. The auto_slurm package is installed and configured
    2. SLURM cluster access is available
    3. The target experiment modules exist in the same directory

    To schedule the jobs:

    .. code-block:: bash

        python _slurm_ex_03.py

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
SEEDS: list[int] = [0, 1, 2, 3, 4]

# The prefix for the experiment archives so that they can be easily identified later
# on. All generated experiment archives will start with this prefix.
PREFIX: str = 'ex_03_ae'


# ====================================================================
# DATASET CONFIGURATION
# ====================================================================
# This section specifies the molecular dataset to be used for the
# ablation study and any dataset-specific parameters.

# The name of the dataset to be used for the ablation study.
# This determines which molecular property prediction task is being evaluated.
DATASET_NAME: str = 'bace_reg'  # 'aqsoldb', 'clogp', 'freesolv', 'lipophilicity', 'bace_reg', 'hopv15_exp', 'compas'

# Fixed parameters that will be passed to all experiments regardless of encoding
# method or model type. This provides an opportunity to set dataset-specific
# configurations or override default experiment parameters.
FIXED_PARAMETERS: dict = {
    'NOTE': 'bace_ic50',  # Dataset identifier for experiment tracking
    'DATASET_NAME': 'bace_reg',  # The actual dataset to load
    'DATASET_TYPE': 'regression',
    'NUM_DATA': 1.0,    # Use the entire dataset (no subsampling)
    'TARGET_INDEX': 0,  # Optional: Specify target property index (e.g., U0 for QM9)
    '__CACHING__': False,  # Disable caching of processed datasets
}


# ====================================================================
# ABLATION STUDY PARAMETERS
# ====================================================================
# This section defines the parameter sweep ranges for the ablation study.
# The study systematically explores combinations of encoding methods,
# prediction models, embedding sizes, and embedding depths.

# Combinations of encoding methods and prediction models to evaluate.
# Each tuple specifies (encoding_method, prediction_model):
#   - 'hdc': Hyperdimensional computing-based graph encoding
#   - 'fp': Traditional molecular fingerprints (e.g., Morgan/ECFP)
#   - 'neural_net2': Multi-layer perceptron (MLP) for regression
#   - 'random_forest': Random Forest ensemble method
# The full ablation study will be performed for each of these combinations.
ENCODING_PARAMETER_TUPLES: list[tuple[str, dict]] = [
    ('hdc', {
        'MODELS': ['neural_net2'], 
    }),
    ('hdc', {
        'MODELS': ['k_neighbors'],
    }),
    ('fp', {
        'MODELS': ['neural_net2'],
        'FINGERPRINT_TYPE': 'morgan',  # Use Morgan/ECFP fingerprints
    }),
    ('fp', {
        'MODELS': ['k_neighbors'],
        'FINGERPRINT_TYPE': 'morgan',  # Use Morgan/ECFP fingerprints
    }),
]

# Embedding size sweep: the dimensionality of the molecular representation.
# For HDC, this is the hypervector dimension. For fingerprints, this is the
# bit vector length. Larger sizes can capture more information but increase
# computational cost and may lead to overfitting.
EMBEDDING_SIZE_SWEEP: list[int | float] = [
    8,
    16,
    32,
    64,
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    16384,
    # 32768,  # Uncomment for very high-dimensional experiments
]

# Embedding depth sweep: controls the receptive field of the encoding.
# For HDC, this is the number of message passing layers (depth of graph aggregation).
# For fingerprints, this is the radius (how many bonds away from each atom to consider).
# Deeper encodings capture more global structure but may include irrelevant information.
EMBEDDING_DEPTH_SWEEP: list[int] = [1, 2, 3]


# ====================================================================
# ENCODING METHOD CONFIGURATIONS
# ====================================================================
# This section defines the parameter mappings and method-specific
# configurations for each encoding approach.

# Parameter name mapping for embedding size: translates the abstract concept of
# "embedding size" into the specific parameter name used by each encoding method.
# This allows the job generation loop to use a unified interface for different methods.
ENCDODING_SIZE_PARAMETER_MAP: dict[str, str] = {
    'fp': 'FINGERPRINT_SIZE',      # For fingerprints: bit vector length
    'hdc': 'EMBEDDING_SIZE',        # For HDC: hypervector dimension
}

# Parameter name mapping for embedding depth: translates the abstract concept of
# "embedding depth" into the specific parameter name used by each encoding method.
ENCODING_DEPTH_PARAMETER_MAP: dict[str, str] = {
    'fp': 'FINGERPRINT_RADIUS',    # For fingerprints: radius (bond distance)
    'hdc': 'NUM_LAYERS',            # For HDC: number of message passing layers
}

# Method-specific parameter configurations that remain constant across the ablation.
# Each encoding method has its own set of hyperparameters for the prediction models
# (Random Forest and Neural Network) and method-specific settings.
ENCODING_PARAMETERS_MAP: dict[str, dict] = {
    'gnn': {
        'NUM_DATA': 1.0,        # Use the entire dataset (no subsampling)
        'EPOCHS': 150           # Training epochs for GNN baseline
    },
    'hdc': {
        'NUM_DATA': 1.0,                        # Use the entire dataset
        # Random Forest hyperparameters:
        'RF_NUM_ESTIMATORS': 100,               # Number of trees in the forest
        'RF_MAX_DEPTH': 10,                     # Maximum depth of each tree
        'RF_MAX_FEATURES': 'sqrt',              # Number of features for split
        # Neural Network hyperparameters:
        'NN_HIDDEN_LAYER_SIZES': (100, 100),    # Two hidden layers with 100 neurons each
        'NN_ALPHA': 0.001,                      # L2 regularization strength
        'NN_LEARNING_RATE_INIT': 0.001,         # Initial learning rate
    },
    'fp': {
        'NUM_DATA': 1.0,                        # Use the entire dataset
        # Random Forest hyperparameters:
        'RF_NUM_ESTIMATORS': 100,               # Number of trees in the forest
        'RF_MAX_DEPTH': 10,                     # Maximum depth of each tree
        'RF_MAX_FEATURES': 'sqrt',              # Number of features for split
        # Neural Network hyperparameters:
        'NN_HIDDEN_LAYER_SIZES': (100, 100),    # Two hidden layers with 100 neurons each
        'NN_ALPHA': 0.001,                      # L2 regularization strength
        'NN_LEARNING_RATE_INIT': 0.001,         # Initial learning rate
        # Fingerprint-specific settings:
        'FINGERPRINT_TYPE': 'morgan',           # Type of molecular fingerprint (Morgan/ECFP)
    },
}


# ====================================================================
# EXPERIMENT STATISTICS
# ====================================================================
# Calculate the total number of experiments that will be generated
# by taking the Cartesian product of all parameter combinations.

num_experiments = (
    len(ENCODING_PARAMETER_TUPLES) *      # Number of (encoding, model) combinations
    len(EMBEDDING_SIZE_SWEEP) *         # Number of embedding sizes to test
    len(EMBEDDING_DEPTH_SWEEP) *        # Number of embedding depths to test
    len(SEEDS)                          # Number of random seeds for robustness
)
print(f'Preparing to schedule {num_experiments} experiments for the ablation study.')


# ====================================================================
# SLURM JOB SUBMISSION
# ====================================================================
# This section handles the generation and submission of SLURM jobs
# for all parameter combinations in the ablation study.

# Initialize the SLURM submitter with the specified configuration.
# batch_size=1: Submit jobs one at a time (can be increased for efficiency)
# randomize=True: Randomize job submission order to avoid systematic biases
# dry_run=True: Set to False to actually submit jobs (True for testing)
submitter = ASlurmSubmitter(
    config_name=AUTOSLURM_CONFIG,
    batch_size=1,
    randomize=True,
    dry_run=False,  # Change to False to actually submit jobs
)

# Generate and submit all experiment combinations.
# This nested loop iterates through all possible combinations of:
# encoding methods, models, embedding sizes, depths, and random seeds.
with tqdm(total=num_experiments, desc="Scheduling experiments") as pbar:

    # Iterate over each (encoding method, prediction model) combination
    for (encoding, encoding_params) in ENCODING_PARAMETER_TUPLES:

        # Iterate over each embedding size in the sweep
        for embedding_size in EMBEDDING_SIZE_SWEEP:

            # Iterate over each embedding depth in the sweep
            for embedding_depth in EMBEDDING_DEPTH_SWEEP:

                # Iterate over each random seed for statistical robustness
                for seed in SEEDS:

                    # --------------------------------------------------------
                    # Assemble Python Command for SLURM Execution
                    # --------------------------------------------------------
                    # Build the Python command that will be executed within each
                    # SLURM job. The command calls the appropriate experiment module
                    # with encoding-specific parameters.

                    # Construct the experiment module filename based on encoding method and dataset
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

                    # Look up the encoding-specific parameter names for size and depth
                    embedding_size_parameter = ENCDODING_SIZE_PARAMETER_MAP[encoding]
                    embedding_depth_parameter = ENCODING_DEPTH_PARAMETER_MAP[encoding]

                    # Build the command line arguments list
                    # Using repr() ensures proper Python string formatting in the command
                    python_command_list = [
                        'pycomex run',
                        experiment_path,
                        f'--__DEBUG__=False ',                                          # Disable debug mode (full dataset)
                        f'--__PREFIX__="{repr(PREFIX)}" ',                              # Set experiment archive prefix
                        f'--SEED="{seed}" ',                                            # Random seed for reproducibility                               # Prediction model to use
                        f'--NUM_TEST="{repr(0.1)}" ',                                   # 10% test set size
                        f'--NUM_TRAIN="{repr(1.0)}" ',                                  # Use 100% of training data
                        f'--{embedding_size_parameter}="{repr(embedding_size)}" ',      # Set embedding size
                        f'--{embedding_depth_parameter}="{repr(embedding_depth)}" '     # Set embedding depth
                    ]

                    # Add encoding-specific parameters (model hyperparameters, etc.)
                    param_dict = ENCODING_PARAMETERS_MAP[encoding]

                    # Merge in any fixed parameters that apply to all experiments
                    param_dict.update(FIXED_PARAMETERS)
                    
                    # Merge with the specific parameters for this encoding-method combination
                    param_dict.update(encoding_params)

                    # Append all method-specific parameters to the command
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
