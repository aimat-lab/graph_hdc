"""
SLURM Job Scheduler for Experiment 06: Molecular Optimization PCA Components Sweep

This script schedules a parameter sweep study that evaluates the impact of PCA optimization
components on molecular property optimization performance. The study compares traditional
molecular fingerprints (FP) with hyperdimensional computing (HDC) representations across
multiple random seeds to establish statistical robustness.

**Experiment Overview:**
    The sweep study explores how different PCA component counts affect the optimization
    landscape and convergence properties when optimizing molecular representations to
    target specific property values. PCA compression reduces the high-dimensional
    representation space (e.g., 4096D) to a lower-dimensional manifold that may be
    easier to optimize via gradient descent.

**Parameter Space:**
    - Encoding methods: Fingerprints (FP with Morgan) and Hyperdimensional Computing (HDC)
    - PCA components: Number of principal components for optimization compression [3, 5, 10, 20]
    - Random seeds: Multiple seeds for statistical robustness [0, 1, 2, 3, 4]
    - Dataset: Configurable molecular property dataset (default: clogp)

**Key Research Questions:**
    1. How does the number of PCA components affect optimization performance?
    2. Do HDC representations benefit differently from PCA compression than fingerprints?
    3. Is there an optimal dimensionality reduction ratio for molecular optimization?
    4. How consistent are results across different random initializations?

**Usage:**
    Before running this script, ensure that:
    1. The auto_slurm package is installed and configured
    2. SLURM cluster access is available
    3. The optimize_molecule__fp.py and optimize_molecule__hdc.py modules exist
    4. Configuration YAML files exist for the target dataset

    To schedule the jobs:

    .. code-block:: bash

        python _slurm_ex_06.py

    The script will generate all parameter combinations and submit them to SLURM.

**Output:**
    Each experiment will generate its own archive with results stored using the PyComex framework.
    Archives are prefixed with the value specified in PREFIX for easy identification.

:note: Adjust AUTOSLURM_CONFIG to match your specific cluster configuration.
:note: Modify DATASET_NAME to test different molecular property prediction tasks.
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
# to all experiments in this sweep study.

PATH = pathlib.Path(__file__).parent.absolute()

# The autoslurm config file to be used for the creation of the SLURM jobs.
# This should match a configuration profile set up in the auto_slurm package.
AUTOSLURM_CONFIG = 'euler_2'

# The random seeds to be used for the random number generator and more importantly
# for the optimization initialization. Multiple seeds enable statistical robustness
# by averaging results across different random initializations.
SEEDS: list[int] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# The prefix for the experiment archives so that they can be easily identified later
# on. All generated experiment archives will start with this prefix.
PREFIX: str = 'ex_06_b'

# The dataset to be used for molecular optimization.
# This determines which molecular property prediction task is being optimized.
# Options include: 'clogp', 'aqsoldb', 'zinc250', etc.
DATASET_NAME: str = 'clogp'


# ====================================================================
# PARAMETER SWEEP CONFIGURATION
# ====================================================================
# This section defines the parameter sweep ranges for the study.
# The study systematically explores combinations of encoding methods,
# PCA component counts, and random seeds.

# PCA optimization components sweep: the number of principal components to use
# when compressing the high-dimensional representation space for optimization.
# Smaller values provide more aggressive dimensionality reduction but may lose
# important structural information. Larger values preserve more information but
# may make the optimization landscape more complex.
PCA_OPTIMIZATION_COMPONENTS_SWEEP: list[int] = [3, 5, 10, 20]

# Combinations of encoding methods to evaluate.
# Each tuple specifies (encoding_method, method_specific_parameters):
#   - 'hdc': Hyperdimensional computing-based molecular encoding
#   - 'fp': Traditional molecular fingerprints (Morgan/ECFP)
ENCODING_PARAMETER_TUPLES: list[tuple[str, dict]] = [
    ('hdc', {
        'EMBEDDING_SIZE': 2048,        # High-dimensional hypervectors
        'NUM_LAYERS': 3,               # Message passing layers
        'ENCODING_MODE': 'continuous', # Use FHRR for continuous features
    }),
    ('fp', {
        'FINGERPRINT_TYPE': 'morgan',  # Use Morgan/ECFP fingerprints
        'FINGERPRINT_SIZE': 2048,      # Fingerprint bit vector length
        'FINGERPRINT_RADIUS': 3,       # Circular radius (ECFP4)
    }),
]


# ====================================================================
# OPTIMIZATION CONFIGURATION
# ====================================================================
# This section defines fixed parameters for the molecular optimization
# process that remain constant across all experiments.

# Fixed parameters for ensemble training and optimization
FIXED_PARAMETERS: dict = {
    # Ensemble parameters
    'ENSEMBLE_SIZE': 10,                    # Number of models in ensemble
    'BOOTSTRAP_FRACTION': 0.8,              # Fraction of data for each bootstrap
    'HIDDEN_UNITS': (256, 256, 128),        # Neural network architecture
    'MAX_EPOCHS': 100,                      # Training epochs per model

    # Optimization parameters
    'NUM_OPTIMIZATION_SAMPLES': 100,        # Number of optimization samples to generate
    'OPTIMIZATION_EPOCHS': 200,             # Optimization iterations
    'OPTIMIZATION_LEARNING_RATE': 0.1,      # Learning rate for gradient descent
    'USE_PCA_OPTIMIZATION': True,           # Enable PCA compression for optimization
    'UNCERTAINTY_WEIGHT': 1.0,              # Weight for uncertainty in acquisition function
    'DISTANCE_METRIC': 'cosine',            # Distance metric for nearest neighbor search

    # Dataset parameters
    'NUM_DATA': None,                       # Use all available data (no subsampling)
    'NUM_TEST': 0.5,                        # Fraction of data for test set
    'NUM_TRAIN': 1.0,                       # Use 100% of training data
    'TARGET_INDEX': 0,                      # Target property index
}


# ====================================================================
# EXPERIMENT STATISTICS
# ====================================================================
# Calculate the total number of experiments that will be generated
# by taking the product of all parameter combinations.

num_experiments = (
    len(ENCODING_PARAMETER_TUPLES) *              # Number of encoding methods (2)
    len(PCA_OPTIMIZATION_COMPONENTS_SWEEP) *      # Number of PCA component values (4)
    len(SEEDS)                                     # Number of random seeds (5)
)
print(f'Preparing to schedule {num_experiments} experiments for the PCA components sweep.')
print(f'  Encodings: {len(ENCODING_PARAMETER_TUPLES)} (fp, hdc)')
print(f'  PCA components: {len(PCA_OPTIMIZATION_COMPONENTS_SWEEP)} {PCA_OPTIMIZATION_COMPONENTS_SWEEP}')
print(f'  Seeds: {len(SEEDS)} {SEEDS}')
print(f'  Dataset: {DATASET_NAME}')


# ====================================================================
# SLURM JOB SUBMISSION
# ====================================================================
# This section handles the generation and submission of SLURM jobs
# for all parameter combinations in the sweep study.

if __name__ == "__main__":

    # Initialize the SLURM submitter with the specified configuration.
    # batch_size=1: Submit jobs one at a time (can be increased for efficiency)
    # randomize=True: Randomize job submission order to avoid systematic biases
    # dry_run=False: Actually submit jobs (set to True for testing without submission)
    submitter = ASlurmSubmitter(
        config_name=AUTOSLURM_CONFIG,
        batch_size=1,
        randomize=True,
        dry_run=False,  # Change to True for testing without actual submission
    )

    # Generate and submit all experiment combinations.
    # This nested loop iterates through all possible combinations of:
    # encoding methods, PCA component counts, and random seeds.
    with tqdm(total=num_experiments, desc="Scheduling experiments") as pbar:

        # Iterate over each encoding method (HDC, FP)
        for encoding, encoding_params in ENCODING_PARAMETER_TUPLES:

            print(f'Scheduling {encoding} experiments...')

            # Iterate over each PCA component count in the sweep
            for pca_components in PCA_OPTIMIZATION_COMPONENTS_SWEEP:

                # Iterate over each random seed for statistical robustness
                for seed in SEEDS:

                    # --------------------------------------------------------
                    # Determine Experiment Module Path
                    # --------------------------------------------------------
                    # Construct the experiment module filename based on encoding
                    # method and dataset. Try dataset-specific YAML config first,
                    # then fall back to base encoding module.

                    # Try dataset-specific YAML config first
                    experiment_module = f'optimize_molecule__{encoding}__{DATASET_NAME}.yml'
                    experiment_path = os.path.join(PATH, experiment_module)
                    
                    print(experiment_path, os.path.exists(experiment_path))

                    # If YAML doesn't exist, use base encoding Python module
                    if not os.path.exists(experiment_path):
                        experiment_module = f'optimize_molecule__{encoding}.py'
                        experiment_path = os.path.join(PATH, experiment_module)

                    # --------------------------------------------------------
                    # Assemble Python Command for SLURM Execution
                    # --------------------------------------------------------
                    # Build the Python command that will be executed within each
                    # SLURM job. The command calls the experiment module with all
                    # necessary parameters.

                    # Start with the base command and common parameters
                    # Using repr() ensures proper Python string formatting in the command
                    python_command_list = [
                        'pycomex run',
                        experiment_path,
                        f'--__DEBUG__=False ',                                          # Disable debug mode
                        f'--__PREFIX__="{repr(PREFIX)}" ',                              # Set experiment archive prefix
                        f'--SEED="{seed}" ',                                            # Random seed for reproducibility
                        f'--PCA_OPTIMIZATION_COMPONENTS="{pca_components}" ',           # PCA components (sweep variable)
                    ]

                    # Append encoding-specific parameters (EMBEDDING_SIZE, NUM_LAYERS, etc.)
                    for key, value in encoding_params.items():
                        python_command_list.append(f'--{key}="{repr(value)}"')

                    # Append fixed optimization parameters
                    for key, value in FIXED_PARAMETERS.items():
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
