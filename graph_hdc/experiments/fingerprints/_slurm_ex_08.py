"""
SLURM Job Scheduler for Experiment 08: Similarity GED Correlation vs Embedding Size

This script schedules a systematic comparison of molecular representation methods for
similarity-based GED correlation analysis, evaluating the impact of embedding dimensionality
on the relationship between graph edit distance and embedding similarity.

**Experiment Overview:**
    This experiment compares two representation methods (Morgan fingerprints and
    hyperdimensional computing) across varying embedding sizes for molecular similarity
    analysis. The goal is to understand:
    1. How embedding size affects GED-similarity correlation
    2. Whether larger embeddings improve structural discrimination
    3. The relative efficiency of HDC vs fingerprints at different dimensionalities
    4. Optimal embedding sizes for capturing molecular structure

**Parameter Space:**
    - Representation methods: Morgan fingerprints (ECFP4), MAP4, HDC
    - Embedding sizes: 8, 32, 128, 512, 2048 dimensions
    - GED analysis: 100 samples per experiment, 3 hops, 20 neighbors per hop

**Key Features:**
    - Each experiment computes GED correlation statistics over 100 query molecules
    - Generates concentration diagnostic plots for each configuration
    - Computes R² and Pearson correlation between GED and embedding similarity
    - Total jobs: 15 (5 sizes × 3 methods)

**Usage:**
    Before running this script, ensure that:
    1. The auto_slurm package is installed and configured
    2. SLURM cluster access is available
    3. The molecule_similarity__hdc.py and molecule_similarity__fp.py modules exist
    4. The vgd_counterfactuals package is installed (for GED neighborhood generation)

    To schedule the jobs:

    .. code-block:: bash

        python _slurm_ex_08.py

    The script will generate and submit all job combinations to the SLURM scheduler.

**Output:**
    Each experiment will generate:
    - ged_correlation_summary.csv with per-molecule and aggregate correlation statistics
    - ged_regression_query_*.png regression plots for each query molecule
    - ged_concentration_diagnostic.png showing similarity distribution analysis
    - similarity_summary.csv with nearest neighbor results

**Expected Insights:**
    - Larger embeddings should show better GED-similarity correlation (higher R²)
    - HDC may show different scaling behavior compared to fingerprints
    - Identification of minimum effective embedding size for structural discrimination
    - Understanding of concentration of measure effects at different dimensionalities

:note: Each SLURM job may take significant time (100 GED samples with neighborhood generation)
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
# to all experiments in this GED correlation comparison study.

PATH = pathlib.Path(__file__).parent.absolute()

# The autoslurm config file to be used for the creation of the SLURM jobs.
# This should match a configuration profile set up in the auto_slurm package.
AUTOSLURM_CONFIG = 'euler'

# The prefix for the experiment archives so that they can be easily identified later on.
# All generated experiment archives will start with this prefix.
PREFIX: str = 'ex_08_a'

# Base random seed for reproducibility.
SEED: int = 1


# ====================================================================
# DATASET CONFIGURATION
# ====================================================================
# This section specifies the molecular dataset configuration for the
# similarity experiments.

# The name of the dataset to be used for the similarity experiments.
DATASET_NAME: str = 'zinc250k'

# Fixed dataset parameters that apply to all experiments
FIXED_PARAMETERS: dict = {
    'DATASET_NAME': DATASET_NAME,
    '__CACHING__': True,         # Enable caching to reuse GED neighborhoods across configurations
    '__DEBUG__': False,          # Run on full dataset
    'SEED': SEED,
}


# ====================================================================
# SIMILARITY SEARCH CONFIGURATION
# ====================================================================
# This section defines the similarity search parameters.

# Number of query molecules for basic similarity search (separate from GED)
NUM_SAMPLES: int = 10

# Number of nearest neighbors to find per query
NUM_NEIGHBORS: int = 5

# Whether to also find dissimilar molecules
FIND_DISSIMILAR: bool = True

# Add similarity search configuration to fixed parameters
FIXED_PARAMETERS.update({
    'NUM_SAMPLES': NUM_SAMPLES,
    'NUM_NEIGHBORS': NUM_NEIGHBORS,
    'FIND_DISSIMILAR': FIND_DISSIMILAR,
})


# ====================================================================
# GED CORRELATION ANALYSIS CONFIGURATION
# ====================================================================
# This section defines the GED correlation analysis parameters.

# Enable GED analysis for all experiments
ENABLE_GED_ANALYSIS: bool = True

# Number of query molecules for GED correlation analysis
GED_NUM_SAMPLES: int = 100

# Maximum hop distance for neighborhood generation
NUM_HOPS: int = 3

# Number of neighbors to sample per branch during neighborhood generation
NUM_NEIGHBOR_BRANCHES: int = 5

# Total number of neighbors to generate per hop level
NUM_NEIGHBOR_TOTAL: int = 20

# Add GED configuration to fixed parameters
FIXED_PARAMETERS.update({
    'ENABLE_GED_ANALYSIS': ENABLE_GED_ANALYSIS,
    'GED_NUM_SAMPLES': GED_NUM_SAMPLES,
    'NUM_HOPS': NUM_HOPS,
    'NUM_NEIGHBOR_BRANCHES': NUM_NEIGHBOR_BRANCHES,
    'NUM_NEIGHBOR_TOTAL': NUM_NEIGHBOR_TOTAL,
})


# ====================================================================
# EMBEDDING SIZE SWEEP
# ====================================================================
# This section defines the parameter sweep for embedding dimensionality.

# Embedding size sweep: the dimensionality of the molecular representation.
# Testing sizes from small to large to understand scaling behavior.
EMBEDDING_SIZE_SWEEP: list[int] = [
    8,      # Very small dimensional - tests extreme compression
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
        'FINGERPRINT_TYPE': 'morgan',     # Morgan/ECFP fingerprints
        'FINGERPRINT_RADIUS': 2,          # Radius 2 = ECFP4
        'USE_COUNTS': False,              # Binary fingerprints
    }),

    # MAP4 fingerprints: MinHashed Atom-Pair fingerprints
    ('fp_map4', {
        'FINGERPRINT_TYPE': 'map4',       # MAP4 fingerprints
        'FINGERPRINT_RADIUS': 2,          # Radius 2
        'USE_COUNTS': False,              # Binary (folded) fingerprints
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
    'fp': 'FINGERPRINT_SIZE',         # For fingerprints: bit vector length
    'fp_map4': 'FINGERPRINT_SIZE',    # For MAP4: folded fingerprint length
    'hdc': 'EMBEDDING_SIZE',          # For HDC: hypervector dimension
}

# Mapping from encoding name to experiment module name. Encodings not listed here
# default to molecule_similarity__{encoding}.py.
ENCODING_MODULE_MAP: dict[str, str] = {
    'fp_map4': 'molecule_similarity__fp.py',
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
print('SLURM Job Submission for Experiment 08: Similarity GED Correlation Analysis')
print('=' * 80)
print(f'\nExperiment Configuration:')
print(f'  Dataset: {DATASET_NAME}')
print(f'  GED Samples: {GED_NUM_SAMPLES} query molecules')
print(f'  GED Hops: {NUM_HOPS} hops, {NUM_NEIGHBOR_TOTAL} neighbors per hop')
print(f'\nSweep Parameters:')
print(f'  Representation Methods: {len(ENCODING_PARAMETER_TUPLES)} (fp, fp_map4, hdc)')
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
with tqdm(total=num_experiments, desc="Scheduling similarity experiments") as pbar:

    # Iterate over each representation method
    for (encoding, encoding_params) in ENCODING_PARAMETER_TUPLES:

        # Iterate over each embedding size in the sweep
        for embedding_size in EMBEDDING_SIZE_SWEEP:

            # --------------------------------------------------------
            # Assemble Python Command for SLURM Execution
            # --------------------------------------------------------
            # Build the Python command that will be executed within each
            # SLURM job. The command calls the appropriate similarity experiment
            # module with encoding-specific parameters.

            # Construct the experiment module filename based on representation method
            experiment_module = ENCODING_MODULE_MAP.get(encoding, f'molecule_similarity__{encoding}.py')
            experiment_path = os.path.join(PATH, experiment_module)

            # Verify that the experiment module exists
            if not os.path.exists(experiment_path):
                raise FileNotFoundError(
                    f"Experiment module not found: {experiment_path}\n"
                    f"Expected module: molecule_similarity__{encoding}.py"
                )

            # Look up the encoding-specific parameter name for embedding size
            embedding_size_parameter = ENCODING_SIZE_PARAMETER_MAP[encoding]

            # Build parameter dict for this specific experiment
            param_dict = FIXED_PARAMETERS.copy()
            param_dict.update(encoding_params)

            # Set the embedding size parameter (encoding-specific name)
            param_dict[embedding_size_parameter] = embedding_size

            # Set experiment name ID for identification
            param_dict['DATASET_NAME_ID'] = f"{DATASET_NAME}_sim_ged_{encoding}_s{embedding_size}"

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
print(f'Each job will analyze {GED_NUM_SAMPLES} molecules for GED correlation.')
print('\nMonitor job progress with: squeue -u $USER')
print('View results in PyComex archives after completion.')
