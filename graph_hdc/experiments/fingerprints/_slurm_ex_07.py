"""
SLURM Job Scheduler for Experiment 07: Bayesian Optimization Representation Size Comparison

This script schedules a systematic comparison of molecular representation methods for Bayesian
Optimization-based molecular search, evaluating the impact of embedding dimensionality on
convergence speed and final solution quality.

**Experiment Overview:**
    This experiment compares three representation methods (random baseline, Morgan fingerprints,
    and hyperdimensional computing) across varying embedding sizes for molecular optimization
    using Bayesian Optimization. The goal is to understand:
    1. How representation quality affects BO convergence speed
    2. Whether larger embeddings improve molecular search performance
    3. The relative efficiency of structured representations vs random baseline
    4. Optimal embedding sizes for each representation method

**Parameter Space:**
    - Representation methods: Random baseline, Morgan fingerprints (ECFP4), HDC
    - Embedding sizes: 8, 16, 32, 64, 128 dimensions
    - BO configuration: 20 initial samples, 25 BO rounds, 25 independent trials per experiment
    - Acquisition function: Expected Improvement (EI)
    - Target property: CLogP (calculated LogP) with target value optimization

**Key Features:**
    - Each experiment internally runs NUM_TRIALS=25 independent BO trials (no seed loop needed)
    - Comparison metrics computed: AUC, simple regret, rounds to threshold
    - Total jobs: 15 (5 sizes × 3 methods)
    - Total BO optimization runs: 375 (15 jobs × 25 trials each)

**Usage:**
    Before running this script, ensure that:
    1. The auto_slurm package is installed and configured
    2. SLURM cluster access is available
    3. The optimize_molecule_bo__*.py modules exist in the same directory
    4. Any required mixin files (e.g., mixin_clogp.py) exist in the same directory

    To schedule the jobs:

    .. code-block:: bash

        python _slurm_ex_07.py

    The script will generate and submit all job combinations to the SLURM scheduler.

**Output:**
    Each experiment will generate:
    - Convergence plots showing BO trajectory (mean ± std across trials)
    - comparison_metrics.csv with AUC, simple regret, and convergence speed metrics
    - bo_results.csv with detailed per-trial, per-round results
    - Best molecules visualization
    - All metrics saved to PyComex data store under 'metrics/' prefix

**Expected Insights:**
    - Random baseline should show worst performance, validating that structured representations
      capture meaningful molecular similarity
    - AUC metric enables direct comparison of convergence speed across methods
    - Optimal embedding size identification for each representation method
    - Quantification of diminishing returns from increasing dimensionality

:note: Each SLURM job may take hours to complete (25 BO trials × 25 rounds each)
:note: Adjust AUTOSLURM_CONFIG to match your specific cluster configuration
"""
# ====================================================================
# IMPORTS
# ====================================================================

import os
import pathlib
from rich.pretty import pprint
import subprocess
from tqdm import tqdm
from auto_slurm.aslurmx import ASlurmSubmitter


# ====================================================================
# EXPERIMENT CONFIGURATION
# ====================================================================
# This section defines the basic configuration parameters that apply
# to all experiments in this Bayesian Optimization comparison study.

PATH = pathlib.Path(__file__).parent.absolute()

# The autoslurm config file to be used for the creation of the SLURM jobs.
# This should match a configuration profile set up in the auto_slurm package.
AUTOSLURM_CONFIG = 'euler'

# The prefix for the experiment archives so that they can be easily identified later on.
# All generated experiment archives will start with this prefix.
PREFIX: str = 'ex_07_c'

# Base random seed for reproducibility. Each experiment uses this seed, with trial
# variation handled internally by NUM_TRIALS parameter (not via seed loop).
SEED: int = 1


# ====================================================================
# DATASET CONFIGURATION
# ====================================================================
# This section specifies the molecular dataset and target property
# configuration for the Bayesian Optimization experiments.

# The name of the dataset to be used for the BO experiments.
DATASET_NAME: str = 'zinc250k'

# List of mixin files to include for dataset preprocessing/property calculation.
# Can be empty list if no mixins needed, or contain multiple mixin files.
# Example: ['mixin_clogp.py'] for CLogP calculation
# Example: ['mixin_clogp.py', 'mixin_conjugated.py'] for multiple properties
DATASET_MIXINS: list[str] = ['/media/ssd2/Programming/graph_hdc/graph_hdc/experiments/fingerprints/mixin_clogp.py']
DATASET_MIXINS = []

# Target property configuration for molecular optimization
TARGET_INDEX: int = 1           # Property index after mixin replaces labels
TARGET_VALUE: float = 0.2      # Target CLogP value to search for
TARGET_MODE: str = 'minimize_distance'  # Optimization objective

# Fixed dataset parameters that apply to all experiments
FIXED_PARAMETERS: dict = {
    'DATASET_NAME': DATASET_NAME,
    'TARGET_INDEX': TARGET_INDEX,
    'TARGET_VALUE': TARGET_VALUE,
    'TARGET_MODE': TARGET_MODE,
    'NUM_HOLDOUT': 0.1,         # Hold out 10% of data
    '__CACHING__': False,        # Disable caching
    '__DEBUG__': False,          # Run on full dataset
}


# ====================================================================
# BAYESIAN OPTIMIZATION CONFIGURATION
# ====================================================================
# This section defines the BO-specific parameters that control the
# optimization process for all experiments.

# Number of random molecules to observe before starting BO
NUM_INITIAL_SAMPLES: int = 10

# Number of Bayesian Optimization iterations to perform
NUM_BO_ROUNDS: int = 25

# Number of molecules to select per BO round
NUM_SAMPLES_PER_ROUND: int = 3

# Number of independent BO trials to run per experiment (for statistical averaging)
# This replaces the seed loop used in other _slurm_ex_*.py files
NUM_TRIALS: int = 25

# Acquisition function to use for candidate selection
ACQUISITION_FUNCTION: str = 'EI'  # Expected Improvement

# Whether to normalize representations before GP training
NORMALIZE_REPRESENTATIONS: bool = True

# Whether to use PCA compression on representations
USE_PCA_COMPRESSION: bool = False

# Threshold for comparison metrics (rounds to threshold calculation)
METRICS_THRESHOLD: float = 0.5

# Add BO configuration to fixed parameters
FIXED_PARAMETERS.update({
    'NUM_INITIAL_SAMPLES': NUM_INITIAL_SAMPLES,
    'NUM_BO_ROUNDS': NUM_BO_ROUNDS,
    'NUM_SAMPLES_PER_ROUND': NUM_SAMPLES_PER_ROUND,
    'NUM_TRIALS': NUM_TRIALS,
    'ACQUISITION_FUNCTION': ACQUISITION_FUNCTION,
    'NORMALIZE_REPRESENTATIONS': NORMALIZE_REPRESENTATIONS,
    'USE_PCA_COMPRESSION': USE_PCA_COMPRESSION,
    'METRICS_THRESHOLD': METRICS_THRESHOLD,
    'SEED': SEED,
})


# ====================================================================
# REPRESENTATION SIZE SWEEP
# ====================================================================
# This section defines the parameter sweep for embedding dimensionality.

# Embedding size sweep: the dimensionality of the molecular representation.
# Testing smaller sizes to understand minimum effective dimensionality and
# scaling behavior of different representation methods.
EMBEDDING_SIZE_SWEEP: list[int] = [
    8,      # Very low dimensional - tests minimum viable size
    16,     # Low dimensional
    32,     # Small dimensional
    64,     # Medium dimensional
    128,    # Standard dimensional
    2048,   # 
]


# ====================================================================
# REPRESENTATION METHOD CONFIGURATIONS
# ====================================================================
# This section defines the three representation methods to compare:
# random baseline, Morgan fingerprints, and HDC.

# Combinations of representation methods and their specific parameters.
# Each tuple specifies (method_name, method_specific_params).
ENCODING_PARAMETER_TUPLES: list[tuple[str, dict]] = [
    # Random baseline: truly random Gaussian vectors with no molecular structure
    ('random', {
        'RANDOM_DISTRIBUTION': 'normal',  # Gaussian N(0,1)
    }),

    # Morgan fingerprints: traditional molecular fingerprints (ECFP4)
    ('fp', {
        'FINGERPRINT_TYPE': 'morgan',     # Morgan/ECFP fingerprints
        'FINGERPRINT_RADIUS': 2,          # Radius 2 = ECFP4
    }),

    # Hyperdimensional computing: graph-based hypervector encoding
    ('hdc', {
        'NUM_LAYERS': 2,                  # 2-layer message passing
        'ENCODING_MODE': 'continuous',    # Continuous FHRR encodings
        'DEVICE': 'cpu',                  # CPU computation
    }),
]

# Parameter name mapping for embedding size: translates the abstract concept of
# "embedding size" into the specific parameter name used by each representation method.
ENCODING_SIZE_PARAMETER_MAP: dict[str, str] = {
    'random': 'RANDOM_DIM',           # For random: vector dimensionality
    'fp': 'FINGERPRINT_SIZE',         # For fingerprints: bit vector length
    'hdc': 'EMBEDDING_SIZE',          # For HDC: hypervector dimension
}


# ====================================================================
# EXPERIMENT STATISTICS
# ====================================================================
# Calculate and display the total number of experiments to be scheduled.

num_experiments = (
    len(ENCODING_PARAMETER_TUPLES) *  # 3 representation methods
    len(EMBEDDING_SIZE_SWEEP)         # 5 embedding sizes
)
# Note: No seed multiplication because NUM_TRIALS handles trial variation internally

print('=' * 80)
print('SLURM Job Submission for Experiment 07: BO Representation Size Comparison')
print('=' * 80)
print(f'\nExperiment Configuration:')
mixins_str = ', '.join(DATASET_MIXINS) if DATASET_MIXINS else 'none'
print(f'  Dataset: {DATASET_NAME} (mixins: {mixins_str})')
print(f'  Target Property: CLogP = {TARGET_VALUE}')
print(f'  BO Configuration: {NUM_INITIAL_SAMPLES} init samples, {NUM_BO_ROUNDS} rounds')
print(f'  Trials per Experiment: {NUM_TRIALS} (internal averaging)')
print(f'\nSweep Parameters:')
print(f'  Representation Methods: {len(ENCODING_PARAMETER_TUPLES)} (random, fp, hdc)')
print(f'  Embedding Sizes: {EMBEDDING_SIZE_SWEEP}')
print(f'\nJob Statistics:')
print(f'  Total SLURM Jobs: {num_experiments}')
print(f'  Total BO Optimization Runs: {num_experiments * NUM_TRIALS}')
print(f'  Prefix: {PREFIX}')
print('=' * 80)
print()


# ====================================================================
# SLURM JOB SUBMISSION
# ====================================================================
# This section handles the generation and submission of SLURM jobs
# for all parameter combinations in the comparison study.

# Initialize the SLURM submitter with the specified configuration.
# batch_size=1: Submit jobs one at a time
# randomize=True: Randomize job submission order to avoid systematic biases
# dry_run=False: Actually submit jobs (set to True for testing)
submitter = ASlurmSubmitter(
    config_name=AUTOSLURM_CONFIG,
    batch_size=1,
    randomize=True,
    dry_run=False,
)

# Generate and submit all experiment combinations.
# Note: Unlike other _slurm_ex_*.py files, we do NOT loop over seeds here
# because NUM_TRIALS parameter handles multiple independent runs internally.
with tqdm(total=num_experiments, desc="Scheduling BO experiments") as pbar:

    # Iterate over each representation method
    for (encoding, encoding_params) in ENCODING_PARAMETER_TUPLES:

        # Iterate over each embedding size in the sweep
        for embedding_size in EMBEDDING_SIZE_SWEEP:

            # --------------------------------------------------------
            # Assemble Python Command for SLURM Execution
            # --------------------------------------------------------
            # Build the Python command that will be executed within each
            # SLURM job. The command calls the appropriate BO experiment module
            # with encoding-specific parameters.

            # Construct the experiment module filename based on representation method
            experiment_module = f'optimize_molecule_bo__{encoding}.py'
            experiment_path = os.path.join(PATH, experiment_module)

            # Verify that the experiment module exists
            if not os.path.exists(experiment_path):
                raise FileNotFoundError(
                    f"Experiment module not found: {experiment_path}\n"
                    f"Expected module: optimize_molecule_bo__{encoding}.py"
                )

            # Look up the encoding-specific parameter name for embedding size
            embedding_size_parameter = ENCODING_SIZE_PARAMETER_MAP[encoding]

            # Build parameter dict for this specific experiment
            param_dict = FIXED_PARAMETERS.copy()
            param_dict.update(encoding_params)

            # Set the embedding size parameter (encoding-specific name)
            param_dict[embedding_size_parameter] = embedding_size

            # Set experiment name ID for identification
            param_dict['DATASET_NAME_ID'] = f"{DATASET_NAME}_clogp_bo_{encoding}_s{embedding_size}"

            # Build the command line arguments list
            # Using repr() ensures proper Python string formatting in the command
            python_command_list = [
                'python',
                experiment_path,
                f'--__PREFIX__="{repr(PREFIX)}"',                    # Set experiment archive prefix
            ]

            # Add mixin includes if any are specified
            # PyComex accepts --include as either a single value or a list
            if DATASET_MIXINS:
                # Multiple mixins: pass as list
                python_command_list.append(f'--__INCLUDE__="{repr(DATASET_MIXINS)}"')

            # Append all experiment parameters to the command
            for key, value in param_dict.items():
                python_command_list.append(f'--{key}="{repr(value)}"')

            # Join all command parts into a single string
            python_command_string = ' '.join(python_command_list)

            # Add the command to the SLURM submitter queue
            submitter.add_command(python_command_string)

            # Update progress bar
            pbar.update(1)

# Submit all queued jobs to the SLURM scheduler
print('\nSubmitting all jobs to SLURM...')
submitter.submit()
print(f'\nSuccessfully submitted {num_experiments} SLURM jobs!')
print(f'Each job will run {NUM_TRIALS} independent BO trials.')
print(f'Total optimization runs: {num_experiments * NUM_TRIALS}')
print('\nMonitor job progress with: squeue -u $USER')
print('View results in PyComex archives after completion.')
