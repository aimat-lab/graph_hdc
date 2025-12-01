"""
Bayesian Optimization Molecular Search Experiment

This experiment uses Bayesian Optimization (BO) with BotTorch to efficiently search
a molecular dataset for compounds with target property values. Instead of gradient
descent in representation space, this approach treats the dataset as a discrete
black-box optimization problem where we iteratively:

1. Start with a small random sample of molecules (initial data)
2. Train a Gaussian Process (GP) to model property predictions and uncertainty
3. Use an acquisition function to select the most promising molecules to evaluate next
4. Observe their true property values and update the GP
5. Repeat until finding molecules close to the target or exhausting the budget

Key Workflow:
    1. Load and filter molecular dataset from chem_mat_data
    2. Process molecules into representations (via hook - HDC, fingerprints, etc.)
    3. Extract target property values (via hook)
    4. Split dataset into candidate pool and holdout test set
    5. For each trial:
        a. Sample random initial molecules
        b. For each BO round:
            - Train GP on observed data (representations → property values)
            - Compute acquisition function over remaining candidates
            - Select top-k candidates with best acquisition values
            - "Observe" their true property values (no function calls needed)
            - Track distance to target over iterations
    6. Average results across multiple trials
    7. Visualize BO search trajectories and performance

Design Rationale:
    - Gaussian Processes provide calibrated uncertainty estimates
    - Acquisition functions balance exploration and exploitation
    - Multiple trials provide robust performance estimates
    - Visualizations show how BO navigates the molecular space

Usage:
    Create configuration files extending this experiment and specifying the
    molecular representation method:

    .. code-block:: yaml

        extend: optimize_molecule_bo__hdc.py
        parameters:
            DATASET_NAME: "aqsoldb"
            TARGET_INDEX: 0
            TARGET_VALUE: 5.0
            NUM_INITIAL_SAMPLES: 10
            NUM_BO_ROUNDS: 20
            NUM_SAMPLES_PER_ROUND: 2
            NUM_TRIALS: 5
            ACQUISITION_FUNCTION: "EI"  # Expected Improvement

Output Artifacts:
    - bo_trial_{trial}_trajectory.png: BO search trajectory for each trial
    - bo_summary.png: Summary statistics across all trials
    - bo_convergence.png: Convergence plot showing distance to target over rounds
    - bo_results.csv: Detailed results for each trial and round
    - bo_best_molecules.png: Visualization of best molecules found
"""
import gc
import os
import time
import copy
import random
from typing import Any, List, Union, Tuple, Optional

import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.integrate import trapezoid
from scipy.stats import spearmanr, pearsonr
from rich.pretty import pprint
from rdkit import Chem
from rdkit.Chem import Draw, AllChem, DataStructs

# BotTorch imports
import gpytorch
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import ExpectedImprovement, UpperConfidenceBound, ProbabilityOfImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

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

# :param NUM_HOLDOUT:
#       The number of samples to hold out for final evaluation (not used in BO).
NUM_HOLDOUT: Union[int, float] = 0.1

# :param DATASET_USE_CACHE:
#       Whether to use the local file system cache for the dataset or force a
#       re-download. Set to False if you suspect the cached dataset is corrupted.
DATASET_USE_CACHE: bool = False

# :param SEED:
#       The random seed to be used for the experiment. If None, random processes
#       will not be seeded, resulting in different outcomes across repetitions.
SEED: Union[int, None] = 1

# == BAYESIAN OPTIMIZATION PARAMETERS ==

# :param TARGET_VALUE:
#       The target property value to search for. If None, the median of the
#       candidate pool will be used as the target.
TARGET_VALUE: Union[float, None] = None

# :param TARGET_MODE:
#       How to define the optimization objective:
#       - "minimize_distance": Minimize absolute distance to TARGET_VALUE
#       - "maximize": Maximize the property value (ignore TARGET_VALUE)
#       - "minimize": Minimize the property value (ignore TARGET_VALUE)
TARGET_MODE: str = "minimize_distance"

# :param NUM_INITIAL_SAMPLES:
#       The number of random molecules to sample as initial data for the GP.
NUM_INITIAL_SAMPLES: int = 10

# :param NUM_BO_ROUNDS:
#       The number of Bayesian Optimization rounds to perform.
NUM_BO_ROUNDS: int = 20

# :param NUM_SAMPLES_PER_ROUND:
#       The number of molecules to select and evaluate in each BO round.
NUM_SAMPLES_PER_ROUND: int = 2

# :param NUM_TRIALS:
#       The number of independent BO trials to run (for averaging results).
NUM_TRIALS: int = 5

# :param ACQUISITION_FUNCTION:
#       The acquisition function to use for selecting next samples.
#       Options: "EI" (Expected Improvement), "UCB" (Upper Confidence Bound),
#                "PI" (Probability of Improvement)
ACQUISITION_FUNCTION: str = "EI"

# :param UCB_BETA:
#       The exploration-exploitation trade-off parameter for UCB acquisition.
#       Higher values encourage more exploration. Only used if ACQUISITION_FUNCTION="UCB".
UCB_BETA: float = 2.0

# :param NORMALIZE_REPRESENTATIONS:
#       Whether to standardize molecular representations before training GP.
#       Recommended for high-dimensional representations.
NORMALIZE_REPRESENTATIONS: bool = True

# :param USE_PCA_COMPRESSION:
#       Whether to apply PCA compression to representations before GP training.
#       This can improve GP scalability for very high-dimensional representations.
USE_PCA_COMPRESSION: bool = False

# :param PCA_COMPONENTS:
#       Number of PCA components to use if USE_PCA_COMPRESSION is enabled.
PCA_COMPONENTS: int = 50

# :param GP_NOISE_CONSTRAINT:
#       Lower bound for GP observation noise. Higher values encourage more exploration.
GP_NOISE_CONSTRAINT: float = 1e-4

# :param NORMALIZE_OUTPUTS:
#       Whether to standardize target property values (y) before GP fitting.
#       This helps the GP work with normalized values and can improve optimization
#       performance, especially for properties with arbitrary scales. The GP will
#       internally standardize outputs to zero mean and unit variance, which often
#       leads to better kernel hyperparameter fitting and more calibrated uncertainty.
NORMALIZE_OUTPUTS: bool = True

# == VISUALIZATION PARAMETERS ==

# :param PLOT_INDIVIDUAL_TRIALS:
#       Whether to create detailed plots for each individual BO trial.
PLOT_INDIVIDUAL_TRIALS: bool = True

# :param PLOT_GP_POSTERIOR:
#       Whether to plot GP posterior predictions on the candidate pool.
#       Warning: Can be slow for large datasets.
PLOT_GP_POSTERIOR: bool = False

# :param NUM_BEST_MOLECULES_TO_SHOW:
#       Number of best molecules to visualize at the end.
NUM_BEST_MOLECULES_TO_SHOW: int = 5

# == REPRESENTATION ANALYSIS PARAMETERS ==

# :param NUM_CORRELATION_PAIRS:
#       Number of random molecule pairs to sample for the representation-property
#       correlation analysis. This analysis measures how well the representation
#       similarity predicts property similarity. Higher values give more accurate
#       estimates but take longer to compute.
NUM_CORRELATION_PAIRS: int = 5000

# :param PLOT_CORRELATION_ANALYSIS:
#       Whether to create the representation-property correlation plot.
#       This plot shows the relationship between representation similarity
#       and property difference, which helps diagnose whether the representation
#       is suitable for the target property.
PLOT_CORRELATION_ANALYSIS: bool = True

# == COMPARISON METRICS PARAMETERS ==

# :param METRICS_THRESHOLD:
#       Threshold for "rounds to threshold" metric. This measures how many rounds
#       are needed to get within this distance of the target value. Lower thresholds
#       are more strict. For CLogP prediction, 0.5 is a reasonable threshold.
#       For other properties, adjust based on typical property value ranges.
METRICS_THRESHOLD: float = 0.5

# == EXPERIMENT PARAMETERS ==

# :param NOTE:
#       A note that can be used to describe the experiment.
NOTE: str = ''

__DEBUG__: bool = True
__NOTIFY__: bool = False
__CACHING__: bool = False

experiment = Experiment(
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals()
)


# == UTILITY FUNCTIONS ==

def compute_acquisition_function(
    gp_model: SingleTaskGP,
    X_candidates: torch.Tensor,
    y_best: float,
    acq_type: str = "EI",
    beta: float = 2.0,
) -> torch.Tensor:
    """
    Compute acquisition function values for candidate points.

    :param gp_model: Trained BotTorch SingleTaskGP model.
    :param X_candidates: Candidate points tensor (n_candidates, n_features).
    :param y_best: Best observed value so far (for EI/PI).
    :param acq_type: Type of acquisition function ("EI", "UCB", "PI").
    :param beta: Exploration parameter for UCB.

    :return: Acquisition values for each candidate (n_candidates,).
    """
    gp_model.eval()

    if acq_type == "EI":
        acq_fn = ExpectedImprovement(model=gp_model, best_f=y_best)
    elif acq_type == "UCB":
        acq_fn = UpperConfidenceBound(model=gp_model, beta=beta)
    elif acq_type == "PI":
        acq_fn = ProbabilityOfImprovement(model=gp_model, best_f=y_best)
    else:
        raise ValueError(f"Unknown acquisition function: {acq_type}")

    # Compute acquisition values
    with torch.no_grad():
        acq_values = acq_fn(X_candidates.unsqueeze(-2))  # Add batch dimension

    return acq_values.squeeze()


def select_top_k_candidates(
    acq_values: torch.Tensor,
    k: int,
    already_selected: Union[set, List[int]],
) -> List[int]:
    """
    Select top-k candidates by acquisition value, excluding already selected ones.

    :param acq_values: Acquisition values for all candidates (n_candidates,).
    :param k: Number of candidates to select.
    :param already_selected: Set or list of indices that have already been selected.

    :return: List of k candidate indices.
    """
    # Create mask for unselected candidates
    mask = torch.ones_like(acq_values, dtype=torch.bool)
    for idx in already_selected:
        mask[idx] = False

    # Mask out selected candidates
    masked_acq_values = acq_values.clone()
    masked_acq_values[~mask] = -float('inf')

    # Select top-k
    _, top_k_indices = torch.topk(masked_acq_values, k=min(k, mask.sum().item()))

    return top_k_indices.tolist()


def compute_bo_comparison_metrics(
    results_df: pd.DataFrame,
    target_mode: str,
    optimal_value: float,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """
    Compute standard Bayesian Optimization comparison metrics for method evaluation.

    These metrics enable fair comparison between different molecular representation
    methods (e.g., HDC vs fingerprints) on the same optimization task. All metrics
    are computed per trial and then averaged.

    Key Metrics:
        1. **Area Under Convergence Curve (AUC)**: Most important metric.
           Computes the area under the "distance/regret vs rounds" curve.
           Lower values indicate faster convergence. This is the standard metric
           used in BO literature (SMAC, Spearmint, etc.) because it captures
           the entire optimization trajectory, not just final performance.

        2. **Simple Regret**: Gap between best found value and global optimum
           after all rounds. Measures final solution quality.

        3. **Rounds to Threshold**: Number of rounds needed to get within a
           threshold distance of the target. Measures convergence speed to
           "good enough" solutions.

        4. **Initial vs Final Improvement**: Percentage improvement from random
           initialization to final result. Measures optimization effectiveness.

    :param results_df: DataFrame with columns ['trial', 'round', 'best_distance' or 'best_property'].
    :param target_mode: One of "minimize_distance", "maximize", or "minimize".
    :param optimal_value: The global optimal distance or property value.
    :param threshold: Threshold for "rounds to threshold" metric (default: 0.5).

    :return: Dictionary containing all computed metrics with mean and std values.
    """
    metrics = {
        'auc_per_trial': [],
        'simple_regret_per_trial': [],
        'rounds_to_threshold_per_trial': [],
        'initial_value_per_trial': [],
        'final_value_per_trial': [],
        'improvement_pct_per_trial': [],
    }

    value_column = 'best_distance' if target_mode == "minimize_distance" else 'best_property'

    for trial_idx in results_df['trial'].unique():
        trial_data = results_df[results_df['trial'] == trial_idx].sort_values('round')

        rounds = trial_data['round'].values
        values = trial_data[value_column].values

        # 1. Area Under Curve (AUC) - PRIMARY METRIC
        # Use trapezoidal integration to compute area under convergence curve
        auc = trapezoid(y=values, x=rounds)
        metrics['auc_per_trial'].append(auc)

        # 2. Simple Regret (final performance)
        final_value = values[-1]
        if target_mode == "minimize_distance":
            simple_regret = final_value - optimal_value
        elif target_mode == "minimize":
            simple_regret = final_value - optimal_value
        else:  # maximize
            simple_regret = optimal_value - final_value
        metrics['simple_regret_per_trial'].append(simple_regret)

        # 3. Rounds to Threshold
        # Find first round where value crosses threshold
        if target_mode == "minimize_distance":
            crossing_indices = np.where(values < threshold)[0]
        elif target_mode == "minimize":
            crossing_indices = np.where(values < (optimal_value + threshold))[0]
        else:  # maximize
            crossing_indices = np.where(values > (optimal_value - threshold))[0]

        if len(crossing_indices) > 0:
            rounds_to_threshold = rounds[crossing_indices[0]]
        else:
            rounds_to_threshold = len(rounds)  # Never reached threshold
        metrics['rounds_to_threshold_per_trial'].append(rounds_to_threshold)

        # 4. Initial vs Final Improvement
        initial_value = values[0]
        metrics['initial_value_per_trial'].append(initial_value)
        metrics['final_value_per_trial'].append(final_value)

        if target_mode == "minimize_distance" or target_mode == "minimize":
            improvement_pct = (initial_value - final_value) / initial_value * 100
        else:  # maximize
            improvement_pct = (final_value - initial_value) / abs(initial_value + 1e-10) * 100
        metrics['improvement_pct_per_trial'].append(improvement_pct)

    # Aggregate statistics across trials
    summary = {}
    for key in ['auc', 'simple_regret', 'rounds_to_threshold',
                'initial_value', 'final_value', 'improvement_pct']:
        values = metrics[f'{key}_per_trial']
        summary[f'{key}_mean'] = float(np.mean(values))
        summary[f'{key}_std'] = float(np.std(values))
        summary[f'{key}_min'] = float(np.min(values))
        summary[f'{key}_max'] = float(np.max(values))
        # Store individual trial values too
        summary[f'{key}_per_trial'] = [float(v) for v in values]

    return summary


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
    if not e.DATASET_USE_CACHE:
        e.log('cache disabled - forcing re-download of dataset')

    graphs: List[GraphDict] = load_graph_dataset(
        e.DATASET_NAME,
        folder_path='/tmp',
        use_cache=e.DATASET_USE_CACHE,
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
        "representation implementation (e.g., optimize_molecule_bo__hdc.py)."
    )


@experiment.hook('extract_property', replace=False, default=True)
def extract_property(e: Experiment,
                     index: int,
                     graph: GraphDict
                     ) -> float:
    """
    Extract the target property value from a graph dictionary.

    The default implementation uses the TARGET_INDEX parameter to extract a
    specific value from the graph_labels array.

    :param e: The experiment instance.
    :param index: The index of the molecule in the dataset.
    :param graph: The graph dictionary containing molecular information.

    :return: The property value as a float.
    """
    if 'property_value' in graph:
        return float(graph['property_value'])

    if 'graph_labels' not in graph:
        raise KeyError(
            f"Graph at index {index} does not contain 'graph_labels'. "
            f"Ensure the dataset includes target properties."
        )

    labels = graph['graph_labels']

    if e.TARGET_INDEX is None:
        if len(labels) == 0:
            raise ValueError(f"Graph at index {index} has empty graph_labels.")
        return float(labels[0])
    else:
        if e.TARGET_INDEX >= len(labels):
            raise IndexError(
                f"TARGET_INDEX={e.TARGET_INDEX} is out of bounds for "
                f"graph_labels with length {len(labels)}."
            )
        return float(labels[e.TARGET_INDEX])


@experiment.hook('dataset_split', replace=False, default=True)
def dataset_split(e: Experiment,
                  indices: List[int],
                  ) -> Tuple[List[int], List[int]]:
    """
    Split the dataset into candidate pool and holdout set.

    :param e: The experiment instance.
    :param indices: List of all dataset indices.

    :return: Tuple of (candidate_indices, holdout_indices).
    """
    random.seed(e.SEED)

    # Determine holdout set size
    if isinstance(e.NUM_HOLDOUT, int):
        num_holdout = e.NUM_HOLDOUT
    elif isinstance(e.NUM_HOLDOUT, float):
        num_holdout = int(e.NUM_HOLDOUT * len(indices))

    holdout_indices = random.sample(indices, k=num_holdout)
    candidate_indices = list(set(indices) - set(holdout_indices))

    return candidate_indices, holdout_indices


# == MAIN EXPERIMENT ==

@experiment
def main(e: Experiment):
    """
    Main experiment function for Bayesian Optimization molecular search.

    Workflow:
    1. Load and filter molecular dataset
    2. Process molecules into representations
    3. Extract target properties
    4. Split dataset into candidates and holdout
    5. Run multiple BO trials
    6. Aggregate and visualize results

    :param e: The experiment instance.

    :return: None. Results are saved as artifacts.
    """
    e.log('starting Bayesian Optimization molecular search experiment...')

    # Handle random seed
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

    # == DATASET PROCESSING ==

    e.log('\n=== PROCESSING DATASET ===')
    time_start = time.time()
    e.apply_hook('process_dataset', index_data_map=index_data_map)
    time_end = time.time()
    e.log(f'processed dataset after {time_end - time_start:.2f} seconds')

    # == DATASET POST-PROCESSING ==
    # This hook point allows mixins to modify the dataset after filtering
    # (e.g., calculating CLogP values, adding computed properties)
    # Note: Runs after filtering so only valid molecules are processed
    e.log('\n=== DATASET POST-PROCESSING ===')
    e.apply_hook('after_dataset', index_data_map=index_data_map)

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

    # Mark the target value if specified
    if e.TARGET_VALUE is not None:
        ax.axvline(
            float(e.TARGET_VALUE),
            color='red',
            linestyle='--',
            linewidth=2.5,
            label=f'Target Value ({float(e.TARGET_VALUE):.3f})',
            zorder=10
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

    # Calculate percentile of target value if specified
    if e.TARGET_VALUE is not None:
        target_value_float = float(e.TARGET_VALUE)
        target_percentile = (properties < target_value_float).sum() / len(properties) * 100
        e.log(f'target value is at {target_percentile:.1f}th percentile of property distribution')

        title_text = (
            f'Distribution of Target Property\n'
            f'Dataset: {e.DATASET_NAME_ID} (Target at {target_percentile:.1f}th percentile)'
        )
    else:
        title_text = (
            f'Distribution of Target Property\n'
            f'Dataset: {e.DATASET_NAME_ID}'
        )

    ax.set_xlabel('Property Value', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
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
    candidate_indices, holdout_indices = e.apply_hook(
        'dataset_split',
        indices=indices
    )
    e.log(f'candidates: {len(candidate_indices)}, holdout: {len(holdout_indices)}')

    e['indices/candidates'] = candidate_indices
    e['indices/holdout'] = holdout_indices

    # Extract candidate data
    X_candidates = np.array([index_data_map[i]['graph_features'] for i in candidate_indices])
    y_candidates = np.array([index_data_map[i]['property_value'] for i in candidate_indices])

    e.log(f'candidate features shape: {X_candidates.shape}')
    e.log(f'candidate properties shape: {y_candidates.shape}')

    # == REPRESENTATION-PROPERTY CORRELATION ANALYSIS ==
    # This analysis measures how well the representation similarity predicts
    # property similarity. A strong negative correlation (high similarity → small
    # property difference) indicates the representation is well-suited for the
    # target property. Weak or positive correlation suggests the representation
    # may mislead the GP during Bayesian Optimization.

    if e.PLOT_CORRELATION_ANALYSIS:
        e.log('\n=== REPRESENTATION-PROPERTY CORRELATION ANALYSIS ===')
        e.log(f'sampling {e.NUM_CORRELATION_PAIRS} random molecule pairs...')

        np.random.seed(e.SEED)

        n_candidates = len(candidate_indices)
        n_pairs = min(e.NUM_CORRELATION_PAIRS, n_candidates * (n_candidates - 1) // 2)

        # Sample random pairs of indices
        pair_indices_1 = np.random.randint(0, n_candidates, size=n_pairs)
        pair_indices_2 = np.random.randint(0, n_candidates, size=n_pairs)

        # Ensure we don't compare a molecule with itself
        same_idx_mask = pair_indices_1 == pair_indices_2
        pair_indices_2[same_idx_mask] = (pair_indices_2[same_idx_mask] + 1) % n_candidates

        # Compute cosine similarities between representation vectors
        e.log('computing representation similarities (cosine)...')

        # Normalize vectors for cosine similarity
        X_norms = np.linalg.norm(X_candidates, axis=1, keepdims=True)
        X_norms[X_norms == 0] = 1  # Avoid division by zero
        X_normalized = X_candidates / X_norms

        # Compute cosine similarities for sampled pairs
        rep_similarities = np.sum(
            X_normalized[pair_indices_1] * X_normalized[pair_indices_2],
            axis=1
        )

        # Compute property differences (absolute)
        e.log('computing property differences...')
        property_differences = np.abs(
            y_candidates[pair_indices_1] - y_candidates[pair_indices_2]
        )

        # Compute correlation metrics
        e.log('computing correlation metrics...')

        # Spearman correlation (rank-based, robust to outliers)
        spearman_corr, spearman_pval = spearmanr(rep_similarities, property_differences)

        # Pearson correlation (linear relationship)
        pearson_corr, pearson_pval = pearsonr(rep_similarities, property_differences)

        e.log(f'\nCorrelation Results:')
        e.log(f' * Spearman correlation: {spearman_corr:.4f} (p={spearman_pval:.2e})')
        e.log(f' * Pearson correlation:  {pearson_corr:.4f} (p={pearson_pval:.2e})')
        e.log(f'\nInterpretation:')
        if spearman_corr < -0.3:
            e.log(f' * GOOD: Strong negative correlation - similar representations')
            e.log(f'         tend to have similar properties. The representation')
            e.log(f'         should help the GP generalize well.')
        elif spearman_corr < -0.1:
            e.log(f' * MODERATE: Weak negative correlation - some relationship')
            e.log(f'             between representation and property similarity.')
        elif spearman_corr < 0.1:
            e.log(f' * WEAK: Near-zero correlation - representation similarity')
            e.log(f'         does not predict property similarity. The GP may')
            e.log(f'         struggle to learn meaningful patterns.')
        else:
            e.log(f' * WARNING: Positive correlation - similar representations')
            e.log(f'            tend to have DIFFERENT properties! This could')
            e.log(f'            actively mislead the GP during optimization.')

        # Store metrics in experiment data
        e['correlation/spearman_corr'] = float(spearman_corr)
        e['correlation/spearman_pval'] = float(spearman_pval)
        e['correlation/pearson_corr'] = float(pearson_corr)
        e['correlation/pearson_pval'] = float(pearson_pval)
        e['correlation/num_pairs'] = int(n_pairs)

        # Create correlation plot
        e.log('\ncreating correlation plot...')

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Left plot: Scatter plot with hexbin for density
        ax1 = axes[0]
        hb = ax1.hexbin(
            rep_similarities,
            property_differences,
            gridsize=50,
            cmap='viridis',
            mincnt=1,
            alpha=0.8
        )
        plt.colorbar(hb, ax=ax1, label='Count')

        # Add trend line
        z = np.polyfit(rep_similarities, property_differences, 1)
        p = np.poly1d(z)
        x_line = np.linspace(rep_similarities.min(), rep_similarities.max(), 100)
        ax1.plot(x_line, p(x_line), 'r--', linewidth=2, label=f'Linear fit')

        ax1.set_xlabel('Representation Similarity (Cosine)', fontsize=12)
        ax1.set_ylabel('Property Difference (|y_i - y_j|)', fontsize=12)
        ax1.set_title(
            f'Representation-Property Correlation\n'
            f'Spearman: {spearman_corr:.3f} (p={spearman_pval:.1e}), '
            f'Pearson: {pearson_corr:.3f} (p={pearson_pval:.1e})',
            fontsize=11
        )
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)

        # Right plot: Distribution of similarities and property differences
        ax2 = axes[1]

        # Create a 2D histogram / joint distribution
        ax2_hist = ax2.hist2d(
            rep_similarities,
            property_differences,
            bins=50,
            cmap='viridis',
            density=True
        )
        plt.colorbar(ax2_hist[3], ax=ax2, label='Density')

        ax2.set_xlabel('Representation Similarity (Cosine)', fontsize=12)
        ax2.set_ylabel('Property Difference (|y_i - y_j|)', fontsize=12)
        ax2.set_title(
            f'Joint Distribution\n'
            f'n={n_pairs} pairs, Dataset: {e.DATASET_NAME_ID}',
            fontsize=11
        )

        plt.suptitle(
            f'Representation Quality Analysis for Bayesian Optimization\n'
            f'Negative correlation = good for BO; Positive/zero = representation may mislead GP',
            fontsize=12,
            fontweight='bold',
            y=1.02
        )

        plt.tight_layout()

        # Save the figure
        correlation_path = os.path.join(e.path, '0_representation_property_correlation.png')
        fig.savefig(correlation_path, dpi=300, bbox_inches='tight')
        e.log(f'saved correlation plot to: {correlation_path}')
        plt.close(fig)

    # == DETERMINE TARGET VALUE ==

    if e.TARGET_VALUE is None:
        target_value = float(np.median(y_candidates))
        e.log(f'\nusing median candidate property as target: {target_value:.3f}')
    else:
        target_value = float(e.TARGET_VALUE)
        e.log(f'\nusing specified target value: {target_value:.3f}')

    e['optimization/target_value'] = float(target_value)
    e['optimization/target_mode'] = e.TARGET_MODE

    # Find global optimal in candidate pool
    if e.TARGET_MODE == "minimize_distance":
        distances_to_target = np.abs(y_candidates - target_value)
        optimal_idx_position = np.argmin(distances_to_target)
        optimal_distance = distances_to_target[optimal_idx_position]
    elif e.TARGET_MODE == "maximize":
        optimal_idx_position = np.argmax(y_candidates)
        optimal_distance = 0.0  # Not meaningful for maximize
    elif e.TARGET_MODE == "minimize":
        optimal_idx_position = np.argmin(y_candidates)
        optimal_distance = 0.0  # Not meaningful for minimize

    optimal_idx = candidate_indices[optimal_idx_position]
    optimal_property = y_candidates[optimal_idx_position]

    e.log(f'\nglobal optimal in candidate pool:')
    e.log(f' * index: {optimal_idx}')
    e.log(f' * property value: {optimal_property:.3f}')
    if e.TARGET_MODE == "minimize_distance":
        e.log(f' * distance to target: {optimal_distance:.3f}')

    e['optimization/optimal_idx'] = int(optimal_idx)
    e['optimization/optimal_property'] = float(optimal_property)
    if e.TARGET_MODE == "minimize_distance":
        e['optimization/optimal_distance'] = float(optimal_distance)

    # == REPRESENTATION PREPROCESSING ==

    e.log('\n=== PREPROCESSING REPRESENTATIONS ===')

    # Optional PCA compression
    if e.USE_PCA_COMPRESSION:
        e.log(f'applying PCA compression to {e.PCA_COMPONENTS} components...')
        pca = PCA(n_components=e.PCA_COMPONENTS, random_state=e.SEED)
        X_candidates_processed = pca.fit_transform(X_candidates)
        explained_variance = pca.explained_variance_ratio_.sum()
        e.log(f' * explained variance: {explained_variance:.4f} ({explained_variance*100:.2f}%)')
    else:
        X_candidates_processed = X_candidates.copy()

    # Optional normalization
    if e.NORMALIZE_REPRESENTATIONS:
        e.log('normalizing representations (StandardScaler)...')
        scaler = StandardScaler()
        X_candidates_processed = scaler.fit_transform(X_candidates_processed)
    else:
        scaler = None

    # Log output normalization setting
    if e.NORMALIZE_OUTPUTS:
        e.log('output normalization enabled (BotTorch Standardize transform)')
    else:
        e.log('output normalization disabled')

    e.log(f'processed features shape: {X_candidates_processed.shape}')

    # Convert to torch tensors
    X_candidates_tensor = torch.tensor(X_candidates_processed, dtype=torch.float32)
    y_candidates_tensor = torch.tensor(y_candidates, dtype=torch.float32).unsqueeze(-1)

    # == RUN BAYESIAN OPTIMIZATION TRIALS ==

    e.log(f'\n=== RUNNING {e.NUM_TRIALS} BO TRIALS ===')

    all_trial_results = []

    for trial_idx in range(e.NUM_TRIALS):
        e.log(f'\n{"="*60}')
        e.log(f'TRIAL {trial_idx + 1}/{e.NUM_TRIALS}')
        e.log(f'{"="*60}')

        # Set trial-specific seed for reproducibility
        trial_seed = e.SEED + trial_idx if e.SEED is not None else trial_idx
        random.seed(trial_seed)
        np.random.seed(trial_seed)
        torch.manual_seed(trial_seed)

        # == INITIAL RANDOM SAMPLING ==

        e.log(f'\nsampling {e.NUM_INITIAL_SAMPLES} initial molecules...')

        # Sample initial indices from candidate pool
        initial_positions = random.sample(range(len(candidate_indices)), k=e.NUM_INITIAL_SAMPLES)
        # Use list to maintain insertion order (corresponds to y_observed tensor order)
        observed_positions = list(initial_positions)

        # Extract initial data
        X_observed = X_candidates_tensor[initial_positions]
        y_observed_raw = y_candidates_tensor[initial_positions]  # Raw property values

        e.log(f'initial sample property range: [{y_observed_raw.min().item():.3f}, {y_observed_raw.max().item():.3f}]')

        # For minimize_distance mode, transform y values to negative distances
        # so that maximizing the GP prediction = minimizing distance to target.
        # The GP needs to be trained in the same space as the acquisition function.
        if e.TARGET_MODE == "minimize_distance":
            # Use view(-1) instead of squeeze() to handle single-sample case correctly
            y_observed = -torch.abs(y_observed_raw.view(-1) - target_value).unsqueeze(-1)
            e.log(f'transformed to negative distances for GP: [{y_observed.min().item():.4f}, {y_observed.max().item():.4f}]')
        else:
            y_observed = y_observed_raw

        # Track best found per round
        round_results = []

        # Initial round (before any BO)
        # Note: Use y_observed_raw for tracking actual distances/properties
        if e.TARGET_MODE == "minimize_distance":
            initial_distances = torch.abs(y_observed_raw.view(-1) - target_value)
            best_initial_distance = initial_distances.min().item()
            best_initial_idx = initial_positions[initial_distances.argmin().item()]
            best_initial_property = y_observed_raw[initial_distances.argmin()].item()

            round_results.append({
                'trial': trial_idx,
                'round': 0,
                'n_observed': len(observed_positions),
                'best_distance': best_initial_distance,
                'best_property': best_initial_property,
                'best_idx': candidate_indices[best_initial_idx],
                'acquisition_type': 'random',
            })

            e.log(f'initial best distance to target: {best_initial_distance:.4f}')

        elif e.TARGET_MODE == "maximize":
            best_initial_property = y_observed_raw.max().item()
            best_initial_idx = initial_positions[y_observed_raw.argmax().item()]

            round_results.append({
                'trial': trial_idx,
                'round': 0,
                'n_observed': len(observed_positions),
                'best_property': best_initial_property,
                'best_idx': candidate_indices[best_initial_idx],
                'acquisition_type': 'random',
            })

            e.log(f'initial best property value: {best_initial_property:.4f}')

        elif e.TARGET_MODE == "minimize":
            best_initial_property = y_observed_raw.min().item()
            best_initial_idx = initial_positions[y_observed_raw.argmin().item()]

            round_results.append({
                'trial': trial_idx,
                'round': 0,
                'n_observed': len(observed_positions),
                'best_property': best_initial_property,
                'best_idx': candidate_indices[best_initial_idx],
                'acquisition_type': 'random',
            })

            e.log(f'initial best property value: {best_initial_property:.4f}')

        # == BO ROUNDS ==

        for bo_round in range(1, e.NUM_BO_ROUNDS + 1):
            e.log(f'\n--- BO Round {bo_round}/{e.NUM_BO_ROUNDS} ---')

            # Check if we've evaluated all candidates
            if len(observed_positions) >= len(candidate_indices):
                e.log('all candidates have been evaluated, stopping early')
                break

            # == TRAIN GAUSSIAN PROCESS ==

            e.log('training Gaussian Process...')
            time_start = time.time()

            # Create GP model with optional output normalization
            outcome_transform = Standardize(m=1) if e.NORMALIZE_OUTPUTS else None
            gp = SingleTaskGP(
                X_observed,
                y_observed,
                outcome_transform=outcome_transform,
                covar_module=gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.MaternKernel(nu=2.5)
                )
            )

            # Set noise constraint
            gp.likelihood.noise_covar.register_constraint(
                "raw_noise",
                gpytorch.constraints.GreaterThan(float(e.GP_NOISE_CONSTRAINT))
            )

            # Fit GP
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)

            time_end = time.time()
            e.log(f'GP training completed in {time_end - time_start:.2f}s')

            # == COMPUTE ACQUISITION FUNCTION ==

            e.log(f'computing {e.ACQUISITION_FUNCTION} acquisition function...')

            # Determine best observed value for acquisition
            # Note: y_observed is already transformed (negative distances) for minimize_distance mode
            if e.TARGET_MODE == "minimize_distance":
                # y_observed contains negative distances, so max = best (smallest distance)
                best_f = y_observed.max().item()
            elif e.TARGET_MODE == "maximize":
                best_f = y_observed.max().item()
            elif e.TARGET_MODE == "minimize":
                # For minimization, negate values to use maximization-based acquisition
                best_f = -y_observed.min().item()

            # Compute acquisition values for all candidates
            acq_values = compute_acquisition_function(
                gp_model=gp,
                X_candidates=X_candidates_tensor,
                y_best=best_f,
                acq_type=e.ACQUISITION_FUNCTION,
                beta=float(e.UCB_BETA),
            )

            e.log(f'acquisition values: min={acq_values.min().item():.4f}, '
                  f'max={acq_values.max().item():.4f}, '
                  f'mean={acq_values.mean().item():.4f}')

            # == SELECT NEXT SAMPLES ==

            num_to_select = min(e.NUM_SAMPLES_PER_ROUND, len(candidate_indices) - len(observed_positions))
            selected_positions = select_top_k_candidates(
                acq_values=acq_values,
                k=num_to_select,
                already_selected=observed_positions,
            )

            e.log(f'selected {len(selected_positions)} new samples')

            # == OBSERVE NEW SAMPLES ==

            # Add to observed list (maintains insertion order)
            observed_positions.extend(selected_positions)

            # Update observed data
            X_new = X_candidates_tensor[selected_positions]
            y_new_raw = y_candidates_tensor[selected_positions]

            # Transform y_new for GP training (same transform as initial data)
            if e.TARGET_MODE == "minimize_distance":
                # Use view(-1) instead of squeeze() to handle single-sample case correctly
                y_new = -torch.abs(y_new_raw.view(-1) - target_value).unsqueeze(-1)
            else:
                y_new = y_new_raw

            X_observed = torch.cat([X_observed, X_new], dim=0)
            y_observed = torch.cat([y_observed, y_new], dim=0)
            y_observed_raw = torch.cat([y_observed_raw, y_new_raw], dim=0)

            e.log(f'total observed samples: {len(observed_positions)}')

            # == TRACK BEST FOUND ==
            # Note: Use y_observed_raw for tracking actual distances/properties

            if e.TARGET_MODE == "minimize_distance":
                observed_distances = torch.abs(y_observed_raw.view(-1) - target_value)
                best_distance = observed_distances.min().item()
                best_idx_pos = observed_distances.argmin().item()
                best_property = y_observed_raw[best_idx_pos].item()

                # Get actual candidate index (observed_positions is ordered list matching y_observed)
                best_idx = candidate_indices[observed_positions[best_idx_pos]]

                round_results.append({
                    'trial': trial_idx,
                    'round': bo_round,
                    'n_observed': len(observed_positions),
                    'best_distance': best_distance,
                    'best_property': best_property,
                    'best_idx': best_idx,
                    'acquisition_type': e.ACQUISITION_FUNCTION,
                })

                e.log(f'current best distance to target: {best_distance:.4f} (property: {best_property:.3f})')

            elif e.TARGET_MODE == "maximize":
                best_property = y_observed_raw.max().item()
                best_idx_pos = y_observed_raw.argmax().item()
                best_idx = candidate_indices[observed_positions[best_idx_pos]]

                round_results.append({
                    'trial': trial_idx,
                    'round': bo_round,
                    'n_observed': len(observed_positions),
                    'best_property': best_property,
                    'best_idx': best_idx,
                    'acquisition_type': e.ACQUISITION_FUNCTION,
                })

                e.log(f'current best property: {best_property:.4f}')

            elif e.TARGET_MODE == "minimize":
                best_property = y_observed_raw.min().item()
                best_idx_pos = y_observed_raw.argmin().item()
                best_idx = candidate_indices[observed_positions[best_idx_pos]]

                round_results.append({
                    'trial': trial_idx,
                    'round': bo_round,
                    'n_observed': len(observed_positions),
                    'best_property': best_property,
                    'best_idx': best_idx,
                    'acquisition_type': e.ACQUISITION_FUNCTION,
                })

                e.log(f'current best property: {best_property:.4f}')

        # Store trial results
        all_trial_results.extend(round_results)

        # == TRIAL SUMMARY ==

        final_result = round_results[-1]
        e.log(f'\nTrial {trial_idx + 1} completed:')
        e.log(f' * total evaluations: {final_result["n_observed"]}')
        if e.TARGET_MODE == "minimize_distance":
            e.log(f' * final best distance: {final_result["best_distance"]:.4f}')
            e.log(f' * final best property: {final_result["best_property"]:.3f}')
            improvement = (round_results[0]["best_distance"] - final_result["best_distance"]) / round_results[0]["best_distance"] * 100
            e.log(f' * improvement over random: {improvement:.2f}%')
        else:
            e.log(f' * final best property: {final_result["best_property"]:.3f}')

        # == PLOT TRIAL TRAJECTORY ==

        if e.PLOT_INDIVIDUAL_TRIALS:
            fig, ax = plt.subplots(figsize=(12, 6))

            rounds = [r['round'] for r in round_results]

            if e.TARGET_MODE == "minimize_distance":
                best_distances = [r['best_distance'] for r in round_results]

                ax.plot(rounds, best_distances, 'o-', linewidth=2, markersize=8, label='Best Distance')
                ax.axhline(optimal_distance, color='green', linestyle='--', linewidth=2,
                          label=f'Global Optimal: {optimal_distance:.3f}')
                ax.set_ylabel('Distance to Target', fontsize=12)
            else:
                best_properties = [r['best_property'] for r in round_results]

                ax.plot(rounds, best_properties, 'o-', linewidth=2, markersize=8, label='Best Property')
                ax.axhline(optimal_property, color='green', linestyle='--', linewidth=2,
                          label=f'Global Optimal: {optimal_property:.3f}')
                ax.set_ylabel('Property Value', fontsize=12)

            ax.set_xlabel('BO Round', fontsize=12)
            ax.set_title(f'Trial {trial_idx + 1}: Bayesian Optimization Trajectory\n'
                        f'Acquisition: {e.ACQUISITION_FUNCTION}, Dataset: {e.DATASET_NAME_ID}',
                        fontsize=13)
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            e.commit_fig(f'bo_trial_{trial_idx + 1}_trajectory.png', fig)
            plt.close(fig)

        # == INTER-TRIAL MEMORY CLEANUP ==
        # Explicitly free memory to prevent accumulation across trials
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # == SAVE ALL RESULTS ==

    e.log('\n=== SAVING RESULTS ===')

    results_df = pd.DataFrame(all_trial_results)
    results_path = os.path.join(e.path, 'bo_results.csv')
    results_df.to_csv(results_path, index=False)
    e.log(f'saved BO results to {results_path}')

    # == AGGREGATE STATISTICS ==

    e.log('\n=== AGGREGATE STATISTICS ===')

    # Group by round and compute statistics
    round_stats = results_df.groupby('round').agg({
        'n_observed': 'mean',
        'best_distance' if e.TARGET_MODE == "minimize_distance" else 'best_property': ['mean', 'std', 'min', 'max'],
    })

    e.log(f'\nRound statistics (averaged over {e.NUM_TRIALS} trials):')
    print(round_stats)

    # Final performance
    final_round_results = results_df[results_df['round'] == results_df['round'].max()]

    if e.TARGET_MODE == "minimize_distance":
        mean_final_distance = final_round_results['best_distance'].mean()
        std_final_distance = final_round_results['best_distance'].std()

        e.log(f'\nfinal performance:')
        e.log(f' * mean best distance: {mean_final_distance:.4f} ± {std_final_distance:.4f}')
        e.log(f' * global optimal distance: {optimal_distance:.4f}')
        e.log(f' * optimality gap: {(mean_final_distance - optimal_distance) / optimal_distance * 100:.2f}%')

        e['summary/mean_final_distance'] = float(mean_final_distance)
        e['summary/std_final_distance'] = float(std_final_distance)
        e['summary/optimality_gap_pct'] = float((mean_final_distance - optimal_distance) / optimal_distance * 100)
    else:
        mean_final_property = final_round_results['best_property'].mean()
        std_final_property = final_round_results['best_property'].std()

        e.log(f'\nfinal performance:')
        e.log(f' * mean best property: {mean_final_property:.4f} ± {std_final_property:.4f}')
        e.log(f' * global optimal property: {optimal_property:.4f}')

        e['summary/mean_final_property'] = float(mean_final_property)
        e['summary/std_final_property'] = float(std_final_property)

    # == COMPUTE COMPARISON METRICS ==

    e.log('\n=== COMPUTING COMPARISON METRICS ===')
    e.log('these metrics enable fair comparison between different representation methods')
    e.log(f'using threshold: {e.METRICS_THRESHOLD}')

    # Compute metrics
    if e.TARGET_MODE == "minimize_distance":
        optimal_value = optimal_distance
    else:
        optimal_value = optimal_property

    comparison_metrics = compute_bo_comparison_metrics(
        results_df=results_df,
        target_mode=e.TARGET_MODE,
        optimal_value=optimal_value,
        threshold=float(e.METRICS_THRESHOLD),
    )

    # Log metrics
    e.log('\nComparison Metrics (averaged over trials):')
    e.log(f' * AUC (area under curve): {comparison_metrics["auc_mean"]:.3f} ± {comparison_metrics["auc_std"]:.3f}')
    e.log(f'   -> Lower is better (faster convergence)')
    e.log(f' * Simple Regret: {comparison_metrics["simple_regret_mean"]:.4f} ± {comparison_metrics["simple_regret_std"]:.4f}')
    e.log(f'   -> Gap from global optimum')
    e.log(f' * Rounds to Threshold (<{e.METRICS_THRESHOLD}): {comparison_metrics["rounds_to_threshold_mean"]:.1f} ± {comparison_metrics["rounds_to_threshold_std"]:.1f}')
    e.log(f'   -> Fewer rounds is better')
    e.log(f' * Improvement: {comparison_metrics["improvement_pct_mean"]:.2f}% ± {comparison_metrics["improvement_pct_std"]:.2f}%')
    e.log(f'   -> Improvement over random initialization')

    # Save to experiment data store
    e.log('\nsaving metrics to experiment data store...')
    for key, value in comparison_metrics.items():
        e[f'metrics/{key}'] = value

    # Also save a summary metrics CSV for easy comparison
    metrics_summary_data = {
        'metric': ['auc', 'simple_regret', 'rounds_to_threshold', 'improvement_pct'],
        'mean': [
            comparison_metrics['auc_mean'],
            comparison_metrics['simple_regret_mean'],
            comparison_metrics['rounds_to_threshold_mean'],
            comparison_metrics['improvement_pct_mean'],
        ],
        'std': [
            comparison_metrics['auc_std'],
            comparison_metrics['simple_regret_std'],
            comparison_metrics['rounds_to_threshold_std'],
            comparison_metrics['improvement_pct_std'],
        ],
        'min': [
            comparison_metrics['auc_min'],
            comparison_metrics['simple_regret_min'],
            comparison_metrics['rounds_to_threshold_min'],
            comparison_metrics['improvement_pct_min'],
        ],
        'max': [
            comparison_metrics['auc_max'],
            comparison_metrics['simple_regret_max'],
            comparison_metrics['rounds_to_threshold_max'],
            comparison_metrics['improvement_pct_max'],
        ],
    }
    metrics_summary_df = pd.DataFrame(metrics_summary_data)
    metrics_path = os.path.join(e.path, 'comparison_metrics.csv')
    metrics_summary_df.to_csv(metrics_path, index=False)
    e.log(f'saved comparison metrics to {metrics_path}')

    # == CONVERGENCE PLOT ==

    e.log('\n=== CREATING CONVERGENCE PLOT ===')

    fig, ax = plt.subplots(figsize=(14, 7))

    # Plot individual trials (thin lines)
    for trial_idx in range(e.NUM_TRIALS):
        trial_data = results_df[results_df['trial'] == trial_idx]
        rounds = trial_data['round'].values

        if e.TARGET_MODE == "minimize_distance":
            values = trial_data['best_distance'].values
        else:
            values = trial_data['best_property'].values

        ax.plot(rounds, values, alpha=0.3, linewidth=1, color='steelblue')

    # Plot mean across trials (thick line)
    mean_by_round = results_df.groupby('round')[
        'best_distance' if e.TARGET_MODE == "minimize_distance" else 'best_property'
    ].mean()
    std_by_round = results_df.groupby('round')[
        'best_distance' if e.TARGET_MODE == "minimize_distance" else 'best_property'
    ].std()

    rounds = mean_by_round.index
    mean_values = mean_by_round.values
    std_values = std_by_round.values

    ax.plot(rounds, mean_values, linewidth=3, color='darkblue', label=f'Mean (n={e.NUM_TRIALS} trials)')
    ax.fill_between(rounds, mean_values - std_values, mean_values + std_values,
                     alpha=0.2, color='darkblue', label='±1 std dev')

    # Optimal reference
    if e.TARGET_MODE == "minimize_distance":
        ax.axhline(optimal_distance, color='green', linestyle='--', linewidth=2,
                  label=f'Global Optimal: {optimal_distance:.3f}')
        ylabel = 'Best Distance to Target Found'
    else:
        ax.axhline(optimal_property, color='green', linestyle='--', linewidth=2,
                  label=f'Global Optimal: {optimal_property:.3f}')
        ylabel = 'Best Property Value Found'

    ax.set_xlabel('BO Round', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f'Bayesian Optimization Convergence\n'
                f'Acquisition: {e.ACQUISITION_FUNCTION}, Dataset: {e.DATASET_NAME_ID}, '
                f'Initial samples: {e.NUM_INITIAL_SAMPLES}',
                fontsize=13)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    e.commit_fig('0_bo_convergence.png', fig)
    plt.close(fig)

    # == BEST MOLECULES VISUALIZATION ==

    e.log('\n=== VISUALIZING BEST MOLECULES ===')

    # Get best molecules found across all trials
    best_indices = final_round_results['best_idx'].unique()[:e.NUM_BEST_MOLECULES_TO_SHOW]

    e.log(f'visualizing top {len(best_indices)} molecules found...')

    n_cols = min(3, len(best_indices))
    n_rows = (len(best_indices) + n_cols - 1) // n_cols

    fig = plt.figure(figsize=(6 * n_cols, 6 * n_rows))

    for plot_idx, mol_idx in enumerate(best_indices):
        graph = index_data_map[mol_idx]
        smiles = graph['graph_repr']
        property_val = graph['property_value']

        mol = Chem.MolFromSmiles(smiles)

        if mol is not None:
            ax = fig.add_subplot(n_rows, n_cols, plot_idx + 1)
            img = Draw.MolToImage(mol, size=(400, 400))
            ax.imshow(img)
            ax.axis('off')

            if e.TARGET_MODE == "minimize_distance":
                distance = abs(property_val - target_value)
                ax.set_title(
                    f'Molecule {mol_idx}\n'
                    f'Property: {property_val:.3f}\n'
                    f'Distance to target: {distance:.4f}',
                    fontsize=11,
                    fontweight='bold'
                )
            else:
                ax.set_title(
                    f'Molecule {mol_idx}\n'
                    f'Property: {property_val:.3f}',
                    fontsize=11,
                    fontweight='bold'
                )

    plt.suptitle(
        f'Best Molecules Found via Bayesian Optimization\n'
        f'Target: {target_value:.3f}, Mode: {e.TARGET_MODE}',
        fontsize=14,
        fontweight='bold',
        y=0.98
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    e.commit_fig('bo_best_molecules.png', fig)
    plt.close(fig)

    # == CLEANUP BEFORE EXIT ==
    # Explicitly clean up objects with C++ backing (PyTorch, matplotlib, GPyTorch)
    # to prevent segfaults during Python's exit garbage collection.
    # The order of cleanup matters: tensors before CUDA context, figures before backend.
    e.log('\n=== CLEANING UP ===')

    # 1. Delete PyTorch tensors explicitly
    del X_candidates_tensor, y_candidates_tensor

    # 2. Clear any remaining matplotlib state
    plt.close('all')

    # 3. Force garbage collection to clean up C++ objects in correct order
    gc.collect()

    # 4. Clear CUDA cache if available (prevents CUDA context issues at exit)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    # 5. Final garbage collection pass
    gc.collect()

    e.log('\n=== EXPERIMENT COMPLETED ===')


experiment.run_if_main()
