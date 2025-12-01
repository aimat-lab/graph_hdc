"""
SLURM Job Scheduler for Experiment 01: Comprehensive Multi-Dataset Benchmark

This script schedules a comprehensive benchmark study that evaluates multiple encoding methods
(HDC, fingerprints, GNNs) across multiple molecular property prediction datasets using optimized
hyperparameters. This experiment represents a systematic comparison of all encoding approaches
on diverse prediction tasks to establish baseline performance metrics.

**Experiment Overview:**
    The benchmark study tests the performance of different molecular encoding methods across
    multiple datasets and prediction models. Unlike ablation studies that vary specific parameters,
    this experiment uses pre-optimized hyperparameters (loaded from a JSON file) to evaluate
    each method at its best configuration for each dataset-model combination.

**Parameter Space:**
    - Encoding methods: HDC (Hyperdimensional Computing), FP (Molecular Fingerprints with Morgan/RDKit), GNN (Graph Neural Networks)
    - Prediction models: Random Forest, Gradient Boosting, k-Nearest Neighbors, Neural Network, GATv2
    - Datasets: Multiple molecular property prediction tasks (FreeSolv, Lipophilicity, BACE)
    - Hyperparameters: Loaded from pre-computed optimization results
    - Random seeds: Multiple seeds for statistical robustness

**Hyperparameter Loading:**
    This experiment loads optimized hyperparameters from a JSON file that contains the best
    hyperparameters for each (encoding, dataset, model) combination, as determined by prior
    hyperparameter optimization experiments. For certain encoding methods (GNN, HDC), fixed
    hyperparameters are used instead of the optimized ones.

**Usage:**
    Before running this script, ensure that:
    1. The auto_slurm package is installed and configured
    2. SLURM cluster access is available
    3. The hyperparameter optimization results file exists at HYPERPARAMETER_PATH
    4. The target experiment modules exist in the same directory

    To schedule the jobs:

    .. code-block:: bash

        python _slurm_ex_01.py

    The script will load hyperparameters, generate all job combinations, and submit them to SLURM.

**Output:**
    Each experiment will generate its own archive with results stored using the PyComex framework.
    Archives are prefixed with the value specified in PREFIX for easy identification.

:note: Adjust AUTOSLURM_CONFIG to match your specific cluster configuration.
:note: The HYPERPARAMETER_PATH must point to a valid JSON file with optimization results.
"""
# ====================================================================
# IMPORTS
# ====================================================================

import os
import sys
import json
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
# to all experiments in this benchmark study.

PATH = pathlib.Path(__file__).parent.absolute()

# The autoslurm config file to be used for the creation of the SLURM jobs.
# This should match a configuration profile set up in the auto_slurm package.
AUTOSLURM_CONFIG = 'euler_2'

# The random seeds to be used for the random number generator and more importantly
# for the dataset splitting. Multiple seeds enable statistical robustness by averaging
# results across different random initializations. Using 10 seeds provides strong
# statistical confidence in the results.
#SEEDS: list[int] = [0, 1, 2, 3, 4]  # First batch option
#SEEDS: list[int] = [5, 6, 7, 8, 9]  # Second batch option
SEEDS: list[int] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # All seeds for complete study

# The prefix for the experiment archives so that they can be easily identified later
# on. All generated experiment archives will start with this prefix.
PREFIX: str = 'ex_01_aa'

# Path to the JSON file containing pre-optimized hyperparameters for each
# (encoding_method, dataset, model) combination. These hyperparameters were
# determined through prior hyperparameter optimization experiments and represent
# the best-performing configurations for each combination.
HYPERPARAMETER_PATH: str = '/media/ssd2/Programming/graph_hdc/graph_hdc/experiments/fingerprints/experiment_best_parameters_map.json'


# ====================================================================
# ENCODING METHOD CONFIGURATIONS
# ====================================================================
# This section defines the encoding methods to evaluate and their
# configurations. The benchmark tests multiple encoding paradigms,
# each with different model options.

# Encoding methods and their variants to test in the benchmark.
# Each tuple contains an encoding method identifier and method-specific parameters:
#   - 'fp': Traditional molecular fingerprints (Morgan or RDKit fingerprints)
#   - 'gnn': Graph neural networks (end-to-end learnable graph representations)
#   - 'hdc': Hyperdimensional computing (high-dimensional vector encoding)
# For fingerprints, both Morgan (ECFP) and RDKit fingerprints are tested as separate variants.
ENCODING_PARAMETER_TUPLES: list[tuple[str, str]] = [
    ('fp', {
        'FINGERPRINT_TYPE': 'morgan',  # Morgan/ECFP fingerprints (circular)
    }),
    ('fp', {
        'FINGERPRINT_TYPE': 'rdkit',   # RDKit fingerprints (topological paths)
    }),
    ('gnn', {}),  # Graph neural networks (no additional parameters needed)
    ('hdc', {}),  # Hyperdimensional computing (no additional parameters needed)
]

# Mapping from encoding methods to compatible prediction models.
# This data structure specifies which prediction models can be used with each
# encoding method. GNNs are end-to-end learnable, so they only use built-in models.
# Fingerprints and HDC encodings produce fixed-size vectors that can be fed to
# various traditional ML models or neural networks.
ENCODING_MODEL_MAP = {
    'gnn': ['gatv2',],  # GATv2: Graph Attention Network (end-to-end differentiable)
    'fp': [
        'random_forest',   # Random Forest ensemble
        'grad_boost',      # Gradient Boosting ensemble
        'k_neighbors',     # k-Nearest Neighbors
        'neural_net',      # Multi-layer perceptron (sklearn)
        'neural_net2'      # Alternative neural network architecture
    ],
    'hdc': [
        'random_forest',   # Random Forest ensemble
        'grad_boost',      # Gradient Boosting ensemble
        'k_neighbors',     # k-Nearest Neighbors
        'neural_net',      # Multi-layer perceptron (sklearn)
        'neural_net2'      # Alternative neural network architecture
    ],
}

# Override hyperparameters for specific encoding methods.
# If an encoding method appears in this map, these fixed hyperparameters will be used
# instead of the optimized ones loaded from the JSON file. This is primarily used for
# GNN and HDC methods where we want to use specific, manually-chosen hyperparameters
# rather than hyperparameter optimization results (e.g., for consistency across experiments).
ENCODING_PARAMETER_MAP = {
    'gnn': {
        'CONV_UNITS': (128, 128, 128),  # Three graph convolution layers with 128 units each
        'LEARNING_RATE': 0.001,          # Adam optimizer learning rate
        'EPOCHS': 250,                   # Number of training epochs
    },
    'hdc': {
        'EMBEDDING_SIZE': 8192,  # Hypervector dimension (high-dimensional space)
        'NUM_LAYERS': 2,         # Number of message passing layers
    },
}


# ====================================================================
# DATASET CONFIGURATION
# ====================================================================
# This section specifies the molecular property prediction datasets
# to benchmark. Each dataset represents a different prediction task
# with its own target property and characteristics.

# List of datasets and their specific configurations.
# Each tuple contains (dataset_name, dataset_parameters) where:
#   - dataset_name: The identifier used to load the dataset
#   - dataset_parameters: Dictionary of dataset-specific settings including:
#       * NOTE: A label/identifier for tracking in experiment archives
#       * DATASET_NAME: The name for loading the dataset
#       * DATASET_TYPE: 'regression' or 'classification'
#       * TARGET_INDEX: Index of the target property to predict
# Commented datasets can be uncommented to include them in the benchmark.
DATASET_PARAMETER_TUPLES: list[tuple[str, dict]] = [
    # (
    #     'qm9_smiles', 
    #     {
    #         'NUM_DATA': 0.1,
    #         'NOTE': 'qm9_gap',
    #         'TARGET_INDEX': 7,  # GAP
    #     },
    # ),
    # (
    #     'qm9_smiles', 
    #     {
    #         'NUM_DATA': 0.1,
    #         'NOTE': 'qm9_energy',
    #         'TARGET_INDEX': 10,  # U0
    #     },
    # ),
    # (
    #     'clogp', 
    #     {
    #         'NOTE': 'clogp',
    #     },
    # ),
    # (
    #     'aqsoldb', 
    #     {
    #         'NOTE': 'aqsoldb',
    #     },
    # ),
    # FreeSolv: Predicting hydration free energy (solvation in water)
    # Small dataset (~640 molecules) with experimental measurements
    (
        'freesolv',
        {
            'NOTE': 'freesolv',                  # Identifier for tracking
            'DATASET_NAME': 'freesolv',          # Dataset name for loading
            'DATASET_TYPE': 'regression',        # Regression task (continuous values)
            'TARGET_INDEX': 0,                   # Experimental hydration free energy (kcal/mol)
        }
    ),
    # Lipophilicity: Predicting octanol/water partition coefficient (logD at pH 7.4)
    # Medium dataset (~4,200 molecules) measuring lipophilicity
    (
        'lipophilicity',
        {
            'NOTE': 'lipophilicity',             # Identifier for tracking
            'DATASET_NAME': 'lipophilicity',     # Dataset name for loading
            'DATASET_TYPE': 'regression',        # Regression task (continuous values)
            'TARGET_INDEX': 0,                   # Experimental octanol/water distribution coefficient
        }
    ),
    # BACE: Predicting binding affinity to beta-secretase enzyme (regression version)
    # Medium dataset (~1,500 molecules) for Alzheimer's drug discovery
    (
        'bace_reg',
        {
            'NOTE': 'bace_reg',                  # Identifier for tracking
            'DATASET_NAME': 'bace_reg',          # Dataset name for loading
            'DATASET_TYPE': 'regression',        # Regression task (continuous values)
            'TARGET_INDEX': 0,                   # Experimental binding affinity (pIC50)
        }
    )
]


# ====================================================================
# MAIN EXECUTION
# ====================================================================

if __name__ == "__main__":

    # ================================================================
    # HYPERPARAMETER LOADING
    # ================================================================
    # Load pre-optimized hyperparameters from JSON file. This file contains
    # the best hyperparameters for each (encoding, dataset, model) combination,
    # as determined by prior hyperparameter optimization experiments.

    if os.path.exists(HYPERPARAMETER_PATH):
        with open(HYPERPARAMETER_PATH, 'r') as file:
            content: list = json.load(file)
            # Convert the list of [[key_tuple, value_dict], ...] pairs into a dictionary
            # where keys are tuples of (encoding_method, dataset_name, model_name)
            # and values are dictionaries of optimized hyperparameters
            HYPERPARAMETER_MAP: dict[tuple, dict] = {
                tuple(element[0]): element[1]
                for element in content
            }

            # Display loaded hyperparameters for verification
            pprint(HYPERPARAMETER_MAP)

    else:
        raise FileNotFoundError(f"Hyperparameter file not found: {HYPERPARAMETER_PATH}")

    # ================================================================
    # EXPERIMENT STATISTICS
    # ================================================================
    # Calculate the total number of experiments by taking the product of all
    # parameter combinations. Uses max model count since different encodings
    # support different numbers of models.

    num_experiments = (
        len(ENCODING_PARAMETER_TUPLES) *                        # Encoding method variants (4)
        max(len(l) for l in ENCODING_MODEL_MAP.values()) *      # Maximum models per encoding (5)
        len(DATASET_PARAMETER_TUPLES) *                         # Number of datasets (3)
        len(SEEDS)                                              # Number of random seeds (10)
    )
    print(f'Preparing to schedule {num_experiments} experiments for the ablation study.')

    # ================================================================
    # SLURM JOB SUBMISSION SETUP
    # ================================================================
    # Initialize the SLURM submitter with the specified configuration.

    # batch_size=1: Submit jobs one at a time (can be increased for efficiency)
    # dry_run=False: Actually submit jobs (set to True for testing without submission)
    # randomize=True: Randomize job submission order to avoid systematic biases
    submitter = ASlurmSubmitter(
        config_name=AUTOSLURM_CONFIG,
        batch_size=1,
        dry_run=False,
        randomize=True,
    )

    # ================================================================
    # JOB GENERATION AND SUBMISSION
    # ================================================================
    # Generate and submit all experiment combinations by iterating through
    # all combinations of datasets, encoding methods, models, and random seeds.
    # Each combination produces one SLURM job with its own unique command.

    with tqdm(total=num_experiments, desc="Scheduling experiments") as pbar:

        # Iterate over each dataset to benchmark
        for dataset, dataset_params in DATASET_PARAMETER_TUPLES:

            # Iterate over each encoding method and its parameters
            for encoding, encoding_params in ENCODING_PARAMETER_TUPLES:

                # Get the list of compatible models for this encoding method
                models = ENCODING_MODEL_MAP[encoding]

                print(f'scheduling ({encoding}, {dataset}) - {len(models) * len(SEEDS)} experiments')

                # Iterate over each prediction model compatible with this encoding
                for model in models:

                    # Iterate over each random seed for statistical robustness
                    for seed in SEEDS:

                        # --------------------------------------------------------
                        # Determine Experiment Module Path
                        # --------------------------------------------------------
                        # Construct the experiment module filename. First, try the
                        # dataset-specific YAML config, then Python file, then fall back to the generic version.

                        # Try dataset-specific YAML config first
                        experiment_module = f'predict_molecules__{encoding}__{dataset}.yml'
                        experiment_path = os.path.join(PATH, experiment_module)

                        # If YAML doesn't exist, try dataset-specific Python file
                        if not os.path.exists(experiment_path):
                            experiment_module = f'predict_molecules__{encoding}__{dataset}.py'
                            experiment_path = os.path.join(PATH, experiment_module)

                        # If dataset-specific Python file doesn't exist, fall back to base encoding module
                        if not os.path.exists(experiment_path):
                            experiment_module = f'predict_molecules__{encoding}.py'
                            experiment_path = os.path.join(PATH, experiment_module)

                        # --------------------------------------------------------
                        # Select Hyperparameters
                        # --------------------------------------------------------
                        # Determine which hyperparameters to use for this experiment.
                        # If the encoding is in ENCODING_PARAMETER_MAP (GNN, HDC), use
                        # those fixed hyperparameters. Otherwise, look up the optimized
                        # hyperparameters from the loaded HYPERPARAMETER_MAP.

                        if encoding in ENCODING_PARAMETER_MAP:
                            # Use fixed hyperparameters for this encoding method (GNN, HDC)
                            optimal_params = ENCODING_PARAMETER_MAP[encoding]

                        else:
                            # Use optimized hyperparameters from hyperparameter optimization
                            # For fingerprints, use the fingerprint type as the encoding key
                            _encoding = encoding
                            if 'FINGERPRINT_TYPE' in encoding_params:
                                _encoding = encoding_params['FINGERPRINT_TYPE']  # e.g., 'morgan' or 'rdkit'

                            # Use the NOTE field as the dataset key if available
                            _dataset = dataset
                            if 'NOTE' in dataset_params:
                                _dataset = dataset_params['NOTE']

                            # Look up optimized hyperparameters using the (encoding, dataset, model) key
                            optimal_params: dict[str, any] = HYPERPARAMETER_MAP[(_encoding, _dataset, model)]

                        # --------------------------------------------------------
                        # Assemble Python Command for SLURM Execution
                        # --------------------------------------------------------
                        # Build the Python command that will be executed within each
                        # SLURM job. The command calls the experiment module with all
                        # necessary parameters.

                        # Start with the base command and common parameters
                        # Using repr() ensures proper Python string formatting in the command
                        python_command_list = [
                            'python',
                            experiment_path,
                            f'--__DEBUG__=False ',                                          # Disable debug mode (full dataset)
                            f'--__PREFIX__="{repr(PREFIX)}" ',                              # Set experiment archive prefix
                            f'--SEED="{seed}" ',                                            # Random seed for reproducibility
                            f'--MODELS="{repr([model])}" ',                                 # Prediction model to use
                            f'--NUM_TEST="{repr(0.1)}" ',                                   # 10% test set size (fixed)
                        ]

                        # Append encoding-specific parameters (e.g., FINGERPRINT_TYPE for fingerprints)
                        for key, value in encoding_params.items():
                            python_command_list.append(f'--{key}="{repr(value)}"')

                        # Append the optimized or fixed hyperparameters for this encoding/model combination
                        for key, value in optimal_params.items():
                            python_command_list.append(f'--{key}="{repr(value)}"')

                        # Append dataset-specific parameters (DATASET_NAME, DATASET_TYPE, TARGET_INDEX, etc.)
                        for key, value in dataset_params.items():
                            python_command_list.append(f'--{key}="{repr(value)}"')

                        # Join all command parts into a single string
                        python_command_string = ' '.join(python_command_list)

                        # Add the command to the SLURM submitter queue
                        submitter.add_command(python_command_string)

                        # Legacy code: Direct subprocess-based submission (now replaced by ASlurmSubmitter)
                        # aslurm_command = [
                        #     'aslurmx',
                        #     '-cn', AUTOSLURM_CONFIG,
                        #     #'--dry-run',
                        #     'cmd',
                        #     python_command_string
                        # ]
                        # #print(aslurm_command)
                        # result = subprocess.run(aslurm_command, cwd=PATH, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                        # #print(result.stdout.decode('utf-8'), result.stderr.decode('utf-8'))
                        # if result.returncode != 0:
                        #     print(f"Warning: aslurmx command failed for {encoding}/{model}/{dataset} with params {optimal_params}")

                        # Update the progress bar
                        pbar.update(1)

    # Submit all queued jobs to the SLURM scheduler
    print(f'Submitting {submitter.count_jobs()} jobs to SLURM...')
    submitter.submit()
