"""
SLURM Job Scheduler for Experiment 09: Bioactivity Prediction vs Embedding Size

This script schedules a systematic comparison of molecular representation methods for
similarity-based bioactivity prediction, evaluating the impact of embedding dimensionality
on virtual screening performance metrics (AUC, BEDROC, Enrichment Factors).

**Experiment Overview:**
    This experiment compares two representation methods (Morgan fingerprints and
    hyperdimensional computing) across varying embedding sizes for bioactivity prediction.
    The goal is to understand:
    1. How embedding size affects virtual screening performance
    2. Whether larger embeddings improve early recognition (BEDROC, EF)
    3. The relative efficiency of HDC vs fingerprints at different dimensionalities
    4. Optimal embedding sizes for bioactivity prediction tasks

**Parameter Space:**
    - Representation methods: Morgan fingerprints (ECFP4), HDC
    - Embedding sizes: 32, 128, 512, 2048 dimensions
    - Evaluation: 50 repetitions, 3 query actives per repetition

**Key Features:**
    - Each experiment evaluates all targets in the dataset
    - Uses standard virtual screening protocol (Riniker & Landrum 2013)
    - Computes AUC, BEDROC(α=20), EF1%, EF5% metrics
    - Total jobs: 8 (4 sizes × 2 methods)

**Usage:**
    Before running this script, ensure that:
    1. The auto_slurm package is installed and configured
    2. SLURM cluster access is available
    3. The predict_bioactivity__hdc.py and predict_bioactivity__fp.py modules exist

    To schedule the jobs:

    .. code-block:: bash

        python _slurm_ex_09.py

    The script will generate and submit all job combinations to the SLURM scheduler.

**Output:**
    Each experiment will generate:
    - per_target_results.csv with detailed metrics for each target
    - aggregated_results.csv with summary statistics (mean ± std)
    - per_target_auc.png, per_target_bedroc.png bar charts
    - metric_distributions.png violin plots
    - performance_scatter.png correlation plots

**Expected Insights:**
    - Larger embeddings may improve AUC but with diminishing returns
    - BEDROC/EF metrics may show different scaling behavior than AUC
    - HDC may require larger dimensions than fingerprints for optimal performance
    - Identification of minimum effective embedding size for virtual screening

:note: Each SLURM job may take significant time depending on dataset size
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
# to all experiments in this bioactivity prediction comparison study.

PATH = pathlib.Path(__file__).parent.absolute()

# The autoslurm config file to be used for the creation of the SLURM jobs.
# This should match a configuration profile set up in the auto_slurm package.
AUTOSLURM_CONFIG = 'euler'

# The prefix for the experiment archives so that they can be easily identified later on.
# All generated experiment archives will start with this prefix.
PREFIX: str = 'ex_09_a'

# Base random seed for reproducibility.
SEED: int = 1


# ====================================================================
# DATASET CONFIGURATION
# ====================================================================
# This section specifies the bioactivity dataset configuration.
# Change this to 'muv' or 'riniker_1' as needed.

# The name of the dataset to be used for the bioactivity experiments.
# Options: 'riniker_1', 'muv', 'riniker_2', 'bl_chembl_cls'
DATASET_NAME: str = 'riniker_1'

# Fixed dataset parameters that apply to all experiments
FIXED_PARAMETERS: dict = {
    'DATASET_NAME': DATASET_NAME,
    '__CACHING__': False,        # Disable caching
    '__DEBUG__': False,          # Run on full dataset
    'SEED': SEED,
}


# ====================================================================
# EVALUATION CONFIGURATION
# ====================================================================
# This section defines the virtual screening evaluation parameters.

# Number of active compounds to use as queries per repetition
NUM_QUERY_ACTIVES: int = 3

# Number of repetitions per target for statistical robustness
NUM_REPETITIONS: int = 50

# Minimum actives required to include a target in evaluation
MIN_ACTIVES_PER_TARGET: int = 10

# BEDROC alpha parameter (standard is 20.0 for virtual screening)
BEDROC_ALPHA: float = 20.0

# Enrichment factor percentages
EF_PERCENTAGES: list = [0.01, 0.05]

# Add evaluation configuration to fixed parameters
FIXED_PARAMETERS.update({
    'NUM_QUERY_ACTIVES': NUM_QUERY_ACTIVES,
    'NUM_REPETITIONS': NUM_REPETITIONS,
    'MIN_ACTIVES_PER_TARGET': MIN_ACTIVES_PER_TARGET,
    'BEDROC_ALPHA': BEDROC_ALPHA,
    'EF_PERCENTAGES': EF_PERCENTAGES,
})


# ====================================================================
# EMBEDDING SIZE SWEEP
# ====================================================================
# This section defines the parameter sweep for embedding dimensionality.

# Embedding size sweep: the dimensionality of the molecular representation.
# Testing sizes from small to large to understand scaling behavior.
EMBEDDING_SIZE_SWEEP: list[int] = [
    32,     # Small dimensional - tests minimum viable size
    128,    # Medium dimensional
    512,    # Standard dimensional
    2048,   # Large dimensional - typical HDC size
]


# ====================================================================
# REPRESENTATION METHOD CONFIGURATIONS
# ====================================================================
# This section defines the two representation methods to compare:
# Morgan fingerprints and HDC.

# Combinations of representation methods and their specific parameters.
# Each tuple specifies (method_name, method_specific_params).
ENCODING_PARAMETER_TUPLES: list[tuple[str, dict]] = [
    # Morgan fingerprints: traditional molecular fingerprints (ECFP4)
    ('fp', {
        'FP_TYPE': 'ecfp4',           # ECFP4 fingerprints
        'FP_RADIUS': 2,               # Radius 2 = ECFP4
        'USE_COUNTS': False,          # Binary fingerprints
    }),

    # Hyperdimensional computing: graph-based hypervector encoding
    ('hdc', {
        'NUM_LAYERS': 2,              # 2-layer message passing
        'ENCODING_MODE': 'continuous', # Continuous FHRR encodings
        'DEVICE': 'cpu',              # CPU computation
        'BATCH_SIZE': 600,            # Batch size for encoding
    }),
]

# Parameter name mapping for embedding size: translates the abstract concept of
# "embedding size" into the specific parameter name used by each representation method.
ENCODING_SIZE_PARAMETER_MAP: dict[str, str] = {
    'fp': 'FP_SIZE',              # For fingerprints: bit vector length
    'hdc': 'EMBEDDING_SIZE',      # For HDC: hypervector dimension
}


# ====================================================================
# EXPERIMENT STATISTICS
# ====================================================================
# Calculate and display the total number of experiments to be scheduled.

num_experiments = (
    len(ENCODING_PARAMETER_TUPLES) *  # 2 representation methods
    len(EMBEDDING_SIZE_SWEEP)         # 4 embedding sizes
)

print('=' * 80)
print('SLURM Job Submission for Experiment 09: Bioactivity Prediction vs Embedding Size')
print('=' * 80)
print(f'\nExperiment Configuration:')
print(f'  Dataset: {DATASET_NAME}')
print(f'  Query Actives: {NUM_QUERY_ACTIVES} per repetition')
print(f'  Repetitions: {NUM_REPETITIONS} per target')
print(f'\nSweep Parameters:')
print(f'  Representation Methods: {len(ENCODING_PARAMETER_TUPLES)} (fp, hdc)')
print(f'  Embedding Sizes: {EMBEDDING_SIZE_SWEEP}')
print(f'\nJob Statistics:')
print(f'  Total SLURM Jobs: {num_experiments}')
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
with tqdm(total=num_experiments, desc="Scheduling bioactivity experiments") as pbar:

    # Iterate over each representation method
    for (encoding, encoding_params) in ENCODING_PARAMETER_TUPLES:

        # Iterate over each embedding size in the sweep
        for embedding_size in EMBEDDING_SIZE_SWEEP:

            # --------------------------------------------------------
            # Assemble Python Command for SLURM Execution
            # --------------------------------------------------------
            # Build the Python command that will be executed within each
            # SLURM job. The command calls the appropriate bioactivity experiment
            # module with encoding-specific parameters.

            # Construct the experiment module filename based on representation method
            experiment_module = f'predict_bioactivity__{encoding}.py'
            experiment_path = os.path.join(PATH, experiment_module)

            # Verify that the experiment module exists
            if not os.path.exists(experiment_path):
                raise FileNotFoundError(
                    f"Experiment module not found: {experiment_path}\n"
                    f"Expected module: predict_bioactivity__{encoding}.py"
                )

            # Look up the encoding-specific parameter name for embedding size
            embedding_size_parameter = ENCODING_SIZE_PARAMETER_MAP[encoding]

            # Build parameter dict for this specific experiment
            param_dict = FIXED_PARAMETERS.copy()
            param_dict.update(encoding_params)

            # Set the embedding size parameter (encoding-specific name)
            param_dict[embedding_size_parameter] = embedding_size

            # Set experiment name ID for identification
            param_dict['DATASET_NAME_ID'] = f"{DATASET_NAME}_bioact_{encoding}_s{embedding_size}"

            # Build the command line arguments list
            # Using repr() ensures proper Python string formatting in the command
            python_command_list = [
                'python',
                experiment_path,
                f'--__PREFIX__="{repr(PREFIX)}"',                    # Set experiment archive prefix
            ]

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
print(f'Dataset: {DATASET_NAME}')
print(f'Each job will evaluate all targets with {NUM_REPETITIONS} repetitions.')
print('\nMonitor job progress with: squeue -u $USER')
print('View results in PyComex archives after completion.')
