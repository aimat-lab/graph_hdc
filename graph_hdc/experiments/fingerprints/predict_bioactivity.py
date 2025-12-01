"""
Similarity-Based Bioactivity Prediction Experiment

This experiment implements the standard evaluation protocol for similarity-based
virtual screening and bioactivity prediction. It follows established benchmarking
practices from the cheminformatics literature to evaluate molecular representations
for their ability to identify bioactive compounds.

Key Workflow:
    1. Load bioactivity dataset (e.g., BL ChEMBL with 35 targets)
    2. Encode molecules into vector representations (via hook - HDC, fingerprints, etc.)
    3. Group molecules by target protein (per-target evaluation)
    4. For each target with sufficient actives:
       a. Run 50 repetitions with different random query selections
       b. For each repetition, randomly select query actives
       c. For each query, rank all other molecules by similarity
       d. Calculate metrics: AUC, BEDROC, Enrichment Factors
       e. Average metrics across queries and repetitions
    5. Aggregate results across all targets (mean ± std)
    6. Visualize per-target and aggregated performance

The experiment supports standard bioactivity prediction metrics:
    - AUC: Area under ROC curve (overall ranking quality)
    - BEDROC(α=20): Boltzmann-Enhanced Discrimination of ROC (early recognition)
    - EF1%/EF5%: Enrichment Factor at 1% and 5% (practical early enrichment)

Design Rationale:
    - Per-target evaluation follows established benchmarking standards (Riniker &
      Landrum 2013, Truchon & Bayly 2007)
    - Multiple repetitions provide robust statistics and confidence estimates
    - Early recognition metrics (BEDROC, EF) crucial for virtual screening
    - Hook-based architecture enables comparison of different representations
    - Results comparable to published baselines (ECFP4 typically AUC ~0.75-0.85)

Usage:
    Create configuration files extending this experiment and specifying the
    molecular representation method:

    .. code-block:: yaml

        extend: predict_bioactivity__hdc.py
        parameters:
            DATASET_NAME: "bl_chembl_reg"
            NUM_QUERY_ACTIVES: 5
            NUM_REPETITIONS: 50
            BEDROC_ALPHA: 20.0
            SEED: 1

Output Artifacts:
    - per_target_results.csv: Detailed metrics for each target
    - aggregated_results.csv: Summary statistics (mean ± std) across targets
    - per_target_auc.png: Bar chart of AUC per target
    - per_target_bedroc.png: Bar chart of BEDROC per target
    - metric_distributions.png: Violin plots of metric distributions
    - example_roc_curves.png: ROC curves from example targets
"""
import os
import time
import random
import hashlib
from typing import Any, List, Union, Tuple, Dict
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
from rich.pretty import pprint
from rdkit import Chem
from sklearn.metrics import roc_curve, auc

from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace
from chem_mat_data._typing import GraphDict
from chem_mat_data.main import load_graph_dataset

# == DATASET PARAMETERS ==

# :param DATASET_NAME:
#       The name of the bioactivity dataset to be used for the experiment.
#       Default is 'bl_chembl_cls' (Briem and Lessel ChEMBL multi-label classification with 35 targets).
DATASET_NAME: str = 'bl_chembl_cls'

# :param DATASET_NAME_ID:
#       The name of the dataset to be used for identification purposes.
DATASET_NAME_ID: str = DATASET_NAME

# :param NUM_DATA:
#       The number of samples to be used for the experiment. This parameter can be
#       either an integer or a float between 0 and 1. If None, the entire dataset
#       is used. Useful for quick testing with smaller subsets.
NUM_DATA: Union[int, float, None] = None

# :param SEED:
#       The random seed to be used for the experiment. If None, random processes
#       will not be seeded, resulting in different outcomes across repetitions.
SEED: Union[int, None] = 1

# == EVALUATION PARAMETERS ==

# :param NUM_QUERY_ACTIVES:
#       The number of active compounds to randomly select as queries in each
#       repetition. These actives will be used as similarity search queries, and
#       the remaining compounds will be ranked by their similarity to each query.
#       Standard practice is 5 queries per repetition.
NUM_QUERY_ACTIVES: int = 5

# :param NUM_REPETITIONS:
#       The number of repetitions to perform for each target. Each repetition uses
#       a different random selection of query actives. Multiple repetitions provide
#       robust statistics and confidence estimates. Standard practice is 50 repetitions.
NUM_REPETITIONS: int = 50

# :param MIN_ACTIVES_PER_TARGET:
#       Minimum number of active compounds required for a target to be included in
#       the evaluation. Targets with fewer actives are skipped because they don't
#       provide sufficient statistical power for meaningful evaluation.
MIN_ACTIVES_PER_TARGET: int = 10

# :param USE_UNLABELED_AS_DECOYS:
#       Whether to include "unlabeled" compounds (7.0 <= pKi < 9.0) as decoys in
#       addition to confirmed inactives (pKi < 7.0). Setting to True increases the
#       size of the screening library, making the task more realistic but potentially
#       adding noise since some unlabeled compounds may have moderate activity.
USE_UNLABELED_AS_DECOYS: bool = True

# == METRIC PARAMETERS ==

# :param BEDROC_ALPHA:
#       The alpha parameter for BEDROC (Boltzmann-Enhanced Discrimination of ROC).
#       This controls the exponential weighting function that emphasizes early
#       recognition. Standard values:
#       - α = 20.0: 80% of max contribution from top 8% of ranked list (standard)
#       - α = 80.5: More aggressive early recognition emphasis
BEDROC_ALPHA: float = 20.0

# :param EF_PERCENTAGES:
#       List of fractions at which to calculate Enrichment Factors. Common values:
#       - 0.01 (1%): Very early recognition, most stringent
#       - 0.05 (5%): Early recognition, practical screening scenario
#       - 0.10 (10%): Moderate early recognition
EF_PERCENTAGES: List[float] = [0.01, 0.05]

# == VISUALIZATION PARAMETERS ==

# :param NUM_EXAMPLE_TARGETS:
#       Number of example targets to show in detailed visualizations (e.g., ROC curves).
#       Typically show best, median, and worst performing targets.
NUM_EXAMPLE_TARGETS: int = 3

# :param FIGURE_DPI:
#       DPI (dots per inch) for saved figures. Higher values produce better quality
#       but larger file sizes.
FIGURE_DPI: int = 300

# == EXPERIMENT PARAMETERS ==

# :param NOTE:
#       A note that can be used to describe the experiment.
NOTE: str = ''

__DEBUG__: bool = True
__NOTIFY__: bool = False

experiment = Experiment(
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals()
)


# == UTILITY FUNCTIONS ==


def calculate_bedroc(active_ranks: np.ndarray,
                    n_actives: int,
                    n_total: int,
                    alpha: float
                    ) -> float:
    """
    Calculate BEDROC (Boltzmann-Enhanced Discrimination of ROC).

    BEDROC emphasizes early recognition using an exponential weighting function.
    It is bounded in [0, 1] where 1 is perfect early recognition and 0 is worse
    than random. The alpha parameter controls the degree of early recognition
    emphasis.

    Reference:
        Truchon & Bayly (2007). "Evaluating Virtual Screening Methods: Good and
        Bad Metrics for the 'Early Recognition' Problem." J. Chem. Inf. Model.

    :param active_ranks: Array of 1-indexed ranks where actives were found.
        For example, if actives are at positions 1, 5, 10, pass [1, 5, 10].
    :param n_actives: Total number of active compounds in the dataset.
    :param n_total: Total number of compounds in the dataset.
    :param alpha: Exponential weight parameter. Standard value is 20.0 for
        virtual screening (80% contribution from top 8%).

    :return: BEDROC score in [0, 1]. Higher is better.

    Example:

    .. code-block:: python

        # If 3 actives are at ranks 1, 2, 50 out of 100 compounds
        bedroc = calculate_bedroc(
            active_ranks=np.array([1, 2, 50]),
            n_actives=3,
            n_total=100,
            alpha=20.0
        )
    """
    if len(active_ranks) == 0 or n_actives == 0:
        return 0.0

    # Sort ranks in ascending order
    ranks_sorted = np.sort(active_ranks)

    N = n_total
    n = n_actives

    # Calculate sum of exponential weights at active positions
    # Ranks are 1-indexed, so we use them directly
    sum_exp = np.sum(np.exp(-alpha * ranks_sorted / N))

    # Calculate expected values for random and perfect ranking
    # Random: actives uniformly distributed
    random_sum = (n / N) * (1 - np.exp(-alpha)) / (1 - np.exp(-alpha / N))

    # Perfect: all actives at top positions
    perfect_sum = (1 / n) * ((1 - np.exp(-alpha * n / N)) / (1 - np.exp(-alpha / N)))
    perfect_sum *= n

    # Normalize to [0, 1]
    if perfect_sum - random_sum == 0:
        return 0.0

    bedroc = (sum_exp - random_sum) / (perfect_sum - random_sum)

    # Clip to [0, 1] to handle numerical issues
    return float(np.clip(bedroc, 0.0, 1.0))


def calculate_enrichment_factor(active_ranks: np.ndarray,
                                n_actives: int,
                                n_total: int,
                                percentage: float
                                ) -> float:
    """
    Calculate Enrichment Factor at specified percentage of ranked list.

    Enrichment Factor measures how many more actives are found in the top X% of
    the ranked list compared to random selection. An EF of 1.0 indicates random
    performance, while higher values indicate enrichment of actives at early ranks.

    :param active_ranks: Array of 1-indexed ranks where actives were found.
    :param n_actives: Total number of active compounds.
    :param n_total: Total number of compounds.
    :param percentage: Fraction of ranked list to consider (e.g., 0.01 for 1%).

    :return: Enrichment factor. Values > 1.0 indicate enrichment, 1.0 is random.

    Example:

    .. code-block:: python

        # If 10 of 100 actives are found in top 1% (10 compounds) of 1000 total
        ef = calculate_enrichment_factor(
            active_ranks=np.array([1, 2, 3, ..., 10]),  # top 10 ranks
            n_actives=100,
            n_total=1000,
            percentage=0.01
        )
        # Result: 10 found vs 1 expected → EF = 10.0
    """
    if n_actives == 0:
        return 0.0

    # Calculate cutoff rank (number of molecules in top X%)
    cutoff = int(np.ceil(n_total * percentage))
    cutoff = max(1, cutoff)  # At least consider rank 1

    # Count how many actives are at or before cutoff rank
    actives_found = np.sum(active_ranks <= cutoff)

    # Calculate expected number if random
    expected_random = n_actives * percentage

    if expected_random == 0:
        return 0.0

    # Enrichment factor
    ef = actives_found / expected_random

    return float(ef)


# == HOOKS ==
# Defining hooks that can be reused throughout the experiment and overwritten by
# subsequent sub-experiments.


@experiment.hook('load_dataset', replace=False, default=True)
def load_dataset(e: Experiment) -> dict[int, GraphDict]:
    """
    Load the bioactivity dataset from ChemMatData.

    This hook downloads and loads the dataset, creating a dictionary mapping
    integer indices to graph dictionaries representing molecules with bioactivity
    labels.

    :param e: The experiment instance.

    :return: Dictionary mapping indices to graph dictionaries.
    """
    e.log(f'loading dataset "{e.DATASET_NAME}"...')

    # Use experiment-specific download folder to avoid stale caches
    download_folder = os.path.join('/tmp', f'chem_mat_{e.DATASET_NAME}')

    # Clear old cache to ensure fresh download
    if os.path.exists(download_folder):
        e.log(f'clearing old cache at {download_folder}...')
        import shutil
        shutil.rmtree(download_folder)

    os.makedirs(download_folder, exist_ok=True)

    graphs: List[GraphDict] = load_graph_dataset(
        e.DATASET_NAME,
        folder_path=download_folder
    )

    index_data_map = dict(enumerate(graphs))
    e.log(f'loaded {len(index_data_map)} molecules from dataset')

    # Log dataset version info
    if len(graphs) > 0:
        first_graph = graphs[0]
        if 'graph_labels' in first_graph:
            num_labels = len(first_graph['graph_labels'])
            e.log(f'dataset has {num_labels}-element label vectors')

    # Optional subsampling for testing
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
        "representation implementation (e.g., predict_bioactivity__hdc.py)."
    )


@experiment.hook('compute_distance', replace=False, default=True)
def compute_distance(e: Experiment,
                     features1: np.ndarray,
                     features2: np.ndarray
                     ) -> float:
    """
    Compute the distance between two molecular representations.

    **IMPORTANT:** This hook must be overridden in extending experiments to
    provide the appropriate distance metric for the representation method.
    Common choices:
        - HDC: cosine distance (1 - cosine_similarity)
        - Fingerprints: Tanimoto/Jaccard distance (1 - Tanimoto_similarity)

    :param e: The experiment instance.
    :param features1: First molecular representation vector.
    :param features2: Second molecular representation vector.

    :return: Distance value (lower = more similar).

    :raises NotImplementedError: This default implementation raises an error.
    """
    raise NotImplementedError(
        "The 'compute_distance' hook must be overridden to provide a distance "
        "metric appropriate for the representation method. For HDC, use cosine "
        "distance; for fingerprints, use Tanimoto distance."
    )


@experiment.hook('group_by_target', replace=False, default=True)
def group_by_target(e: Experiment,
                   index_data_map: dict[int, GraphDict]
                   ) -> Dict[int, List[Dict[str, Any]]]:
    """
    Group molecules by their target protein for per-target evaluation.

    For bl_chembl_cls dataset, this extracts actives and decoys per target from
    multi-label classification format where:
    - labels[target_idx] = 0: not active/not tested
    - labels[target_idx] = 1: active for this target
    - labels[target_idx] = 2: decoy for this target

    :param e: The experiment instance.
    :param index_data_map: Dictionary of all molecules in the dataset.

    :return: Dictionary mapping target indices (0-34) to lists of molecules
        (both actives and decoys for that target). Each molecule dict contains:
        index, smiles, features, is_active, target_idx, target_name
    """
    e.log('grouping molecules by target from multi-label classification...')

    # Target metadata from dataset documentation
    target_names = {
        0: 'CHEMBL1862 - Tyrosine-protein kinase ABL',
        1: 'CHEMBL204 - Thrombin',
        2: 'CHEMBL205 - Carbonic anhydrase II',
        3: 'CHEMBL4794 - Vanilloid receptor',
        4: 'CHEMBL264 - Histamine H3 receptor',
        5: 'CHEMBL214 - Serotonin 1a (5-HT1a) receptor',
        6: 'CHEMBL217 - Dopamine D2 receptor',
        7: 'CHEMBL4552 - Peripheral-type benzodiazepine receptor',
        8: 'CHEMBL2147 - Serine/threonine-protein kinase PIM1',
        9: 'CHEMBL224 - Serotonin 2a (5-HT2a) receptor',
        10: 'CHEMBL229 - Alpha-1a adrenergic receptor',
        11: 'CHEMBL233 - Mu opioid receptor',
        12: 'CHEMBL234 - Dopamine D3 receptor',
        13: 'CHEMBL236 - Delta opioid receptor',
        14: 'CHEMBL237 - Kappa opioid receptor',
        15: 'CHEMBL2366517 - Protease (HIV-1)',
        16: 'CHEMBL4409 - Phosphodiesterase 10A',
        17: 'CHEMBL2835 - Tyrosine-protein kinase JAK1',
        18: 'CHEMBL2971 - Tyrosine-protein kinase JAK2',
        19: 'CHEMBL3952 - Kappa opioid receptor (Guinea pig)',
        20: 'CHEMBL243 - HIV-1 protease',
        21: 'CHEMBL244 - Coagulation factor X',
        22: 'CHEMBL339 - Dopamine D2 receptor (Rat)',
        23: 'CHEMBL245 - Muscarinic acetylcholine receptor M3',
        24: 'CHEMBL251 - Adenosine A2a receptor',
        25: 'CHEMBL253 - Cannabinoid CB2 receptor',
        26: 'CHEMBL256 - Adenosine A3 receptor',
        27: 'CHEMBL269 - Delta opioid receptor (Rat)',
        28: 'CHEMBL270 - Mu opioid receptor (Rat)',
        29: 'CHEMBL1946 - Melatonin receptor 1B',
        30: 'CHEMBL273 - Serotonin 1a (5-HT1a) receptor (Rat)',
        31: 'CHEMBL1907596 - Neuronal acetylcholine receptor alpha4/beta2',
        32: 'CHEMBL3371 - Serotonin 6 (5-HT6) receptor',
        33: 'CHEMBL313 - Serotonin transporter (Rat)',
        34: 'CHEMBL4860 - Apoptosis regulator Bcl-2',
    }

    target_groups = defaultdict(list)

    # Determine number of targets and dataset version from first molecule
    first_graph = next(iter(index_data_map.values()))
    num_label_elements = len(first_graph['graph_labels'])
    e.log(f'detected {num_label_elements} elements in label vector')

    # Detect dataset format by checking for value=2 in the dataset
    has_value_2 = any(2 in graph['graph_labels'] for graph in list(index_data_map.values())[:100])

    # Check if we're using old format (36 elements) or newer formats
    if num_label_elements == 36:
        # Old format: position 35 is decoy indicator, positions 0-34 are targets
        e.log('using OLD dataset format: position 35 = decoy indicator (bl_chembl_cls old)')
        num_targets = 35
        use_decoy_field = True  # Use labels[35] or graph_is_decoy field
        use_value_2_decoys = False
    elif has_value_2:
        # New format with value=2: all positions are targets, value=2 indicates decoy
        e.log('using NEW dataset format: value=2 indicates per-target decoys (riniker/bl_chembl_cls)')
        num_targets = num_label_elements
        use_decoy_field = False
        use_value_2_decoys = True
    else:
        # MUV-style format: value=0 means inactive/decoy, value=1 means active
        e.log('using MUV-style format: value=0 treated as decoy, value=1 as active')
        num_targets = num_label_elements
        use_decoy_field = False
        use_value_2_decoys = False

    for index, graph in index_data_map.items():
        smiles = graph['graph_repr']
        features = graph['graph_features']
        labels = graph['graph_labels']

        # Check if this molecule is a shared decoy (old format only)
        if use_decoy_field:
            # Old format: check graph_is_decoy field
            is_shared_decoy = graph.get('graph_is_decoy', 0) == 1
        else:
            is_shared_decoy = False

        if is_shared_decoy:
            # Old format: Shared decoy molecule, add to all targets as decoy
            for target_idx in range(num_targets):
                molecule_info = {
                    'index': index,
                    'smiles': smiles,
                    'features': features,
                    'is_active': False,
                    'target_idx': target_idx,
                    'target_name': target_names.get(target_idx, f'Target {target_idx}'),
                }
                target_groups[target_idx].append(molecule_info)
        else:
            # Per-target processing: add molecule to each target based on its label
            for target_idx in range(num_targets):
                label_value = labels[target_idx]

                if label_value == 1:
                    # Active for this target
                    molecule_info = {
                        'index': index,
                        'smiles': smiles,
                        'features': features,
                        'is_active': True,
                        'target_idx': target_idx,
                        'target_name': target_names.get(target_idx, f'Target {target_idx}'),
                    }
                    target_groups[target_idx].append(molecule_info)

                elif label_value == 2 and use_value_2_decoys:
                    # Decoy for this target (riniker/bl_chembl_cls format)
                    molecule_info = {
                        'index': index,
                        'smiles': smiles,
                        'features': features,
                        'is_active': False,
                        'target_idx': target_idx,
                        'target_name': target_names.get(target_idx, f'Target {target_idx}'),
                    }
                    target_groups[target_idx].append(molecule_info)

                elif label_value == 0 and not use_value_2_decoys and not use_decoy_field:
                    # Inactive/decoy for this target (MUV-style format)
                    molecule_info = {
                        'index': index,
                        'smiles': smiles,
                        'features': features,
                        'is_active': False,
                        'target_idx': target_idx,
                        'target_name': target_names.get(target_idx, f'Target {target_idx}'),
                    }
                    target_groups[target_idx].append(molecule_info)

    e.log(f'grouped molecules into {len(target_groups)} targets')

    # Log target statistics
    for target_idx in sorted(target_groups.keys()):
        molecules = target_groups[target_idx]
        actives = [m for m in molecules if m['is_active']]
        decoys = [m for m in molecules if not m['is_active']]
        target_name = target_names.get(target_idx, f'Target {target_idx}')
        e.log(f' * target {target_idx} ({target_name}): {len(actives)} actives, {len(decoys)} decoys')

    return dict(target_groups)


@experiment.hook('rank_by_similarity', replace=False, default=True)
def rank_by_similarity(e: Experiment,
                      query_molecule: Dict[str, Any],
                      candidate_molecules: List[Dict[str, Any]]
                      ) -> List[Tuple[Dict[str, Any], float]]:
    """
    Rank candidate molecules by their similarity to the query molecule.

    This hook computes distances from the query to all candidates and returns
    them sorted by distance (ascending = most similar first).

    :param e: The experiment instance.
    :param query_molecule: Dictionary containing query molecule information
        including 'features' key.
    :param candidate_molecules: List of candidate molecule dictionaries.

    :return: List of (molecule_dict, distance) tuples sorted by distance.
    """
    query_features = query_molecule['features']

    distances = []
    for candidate in candidate_molecules:
        candidate_features = candidate['features']

        # Compute distance using the hook
        distance = e.apply_hook(
            'compute_distance',
            features1=query_features,
            features2=candidate_features
        )

        distances.append((candidate, distance))

    # Sort by distance (ascending = most similar first)
    distances.sort(key=lambda x: x[1])

    return distances


@experiment.hook('evaluate_single_query', replace=False, default=True)
def evaluate_single_query(e: Experiment,
                         query_molecule: Dict[str, Any],
                         ranked_molecules: List[Tuple[Dict[str, Any], float]],
                         ) -> Dict[str, float]:
    """
    Evaluate metrics for a single query active.

    This hook calculates AUC, BEDROC, and Enrichment Factors for one query
    active based on the ranked list of all other molecules.

    :param e: The experiment instance.
    :param query_molecule: Dictionary of the query active molecule.
    :param ranked_molecules: List of (molecule_dict, distance) tuples sorted by
        distance (most similar first).

    :return: Dictionary containing metric values: auc, bedroc, ef1, ef5, etc.
    """
    # Extract labels and distances
    labels = []
    distances_array = []

    for molecule, distance in ranked_molecules:
        # Binary label: 1 if active, 0 if decoy/inactive
        if molecule.get('is_active', False):
            labels.append(1)  # Active = positive
        else:
            labels.append(0)  # Decoy/inactive = negative

        distances_array.append(distance)

    labels = np.array(labels)
    distances_array = np.array(distances_array)

    if len(labels) == 0 or np.sum(labels) == 0:
        # No molecules or no actives in ranked list
        return {
            'auc': 0.0,
            'bedroc': 0.0,
            **{f'ef{int(pct*100)}': 0.0 for pct in e.EF_PERCENTAGES}
        }

    # Calculate AUC using sklearn
    # For AUC, we treat distance as a "score" where lower distance = higher similarity
    # So we negate distances or use (1 - distance_normalized) as scores
    # Simpler: convert distances to similarities (lower distance = higher score)
    # We'll use negative distances as scores (more negative = closer)
    scores = -distances_array

    try:
        fpr, tpr, thresholds = roc_curve(labels, scores)
        auc_value = auc(fpr, tpr)
    except ValueError:
        # Handle edge case where all labels are same
        auc_value = 0.5

    # Calculate BEDROC
    # Find ranks (1-indexed) of active molecules
    active_ranks = np.where(labels == 1)[0] + 1  # Convert 0-indexed to 1-indexed
    n_actives = int(np.sum(labels))
    n_total = len(labels)

    bedroc_value = calculate_bedroc(
        active_ranks=active_ranks,
        n_actives=n_actives,
        n_total=n_total,
        alpha=e.BEDROC_ALPHA
    )

    # Calculate Enrichment Factors
    ef_values = {}
    for percentage in e.EF_PERCENTAGES:
        ef = calculate_enrichment_factor(
            active_ranks=active_ranks,
            n_actives=n_actives,
            n_total=n_total,
            percentage=percentage
        )
        # Store with key like 'ef1' for 1%, 'ef5' for 5%
        key = f'ef{int(percentage * 100)}'
        ef_values[key] = ef

    # Combine all metrics
    metrics = {
        'auc': float(auc_value),
        'bedroc': float(bedroc_value),
        **ef_values
    }

    return metrics


@experiment.hook('evaluate_target', replace=False, default=True)
def evaluate_target(e: Experiment,
                   target_id: int,
                   target_name: str,
                   molecules: List[Dict[str, Any]]
                   ) -> Union[Dict[str, Any], None]:
    """
    Evaluate one target with multiple repetitions.

    This hook implements the standard virtual screening evaluation protocol:
    - Run NUM_REPETITIONS repetitions
    - In each repetition, randomly select NUM_QUERY_ACTIVES actives as queries
    - For each query, rank all other molecules (remaining actives + decoys)
    - Average metrics across queries within each repetition
    - Average across all repetitions

    :param e: The experiment instance.
    :param target_id: Target index (0-34).
    :param target_name: Human-readable name of the target.
    :param molecules: List of molecule dictionaries for this target (actives + decoys).

    :return: Dictionary with aggregated metrics, or None if target has insufficient
        actives.
    """
    # Separate actives and decoys
    actives = [m for m in molecules if m['is_active']]
    decoys = [m for m in molecules if not m['is_active']]

    e.log(f' * target {target_id} ({target_name}):')
    e.log(f'   - {len(actives)} actives, {len(decoys)} decoys')

    # Check if sufficient actives
    if len(actives) < e.MIN_ACTIVES_PER_TARGET:
        e.log(f'   - SKIPPING: insufficient actives (< {e.MIN_ACTIVES_PER_TARGET})')
        return None

    # Check if enough actives for query selection
    if len(actives) < e.NUM_QUERY_ACTIVES:
        e.log(f'   - WARNING: fewer actives ({len(actives)}) than NUM_QUERY_ACTIVES ({e.NUM_QUERY_ACTIVES})')
        e.log(f'   - will use all {len(actives)} actives as queries')
        num_queries = len(actives)
    else:
        num_queries = e.NUM_QUERY_ACTIVES

    # Storage for repetition results
    repetition_results = []

    for rep_idx in range(e.NUM_REPETITIONS):
        # Randomly select query actives for this repetition
        query_actives = random.sample(actives, k=num_queries)

        # Storage for query results within this repetition
        query_metrics = []

        for query in query_actives:
            # Get all non-query molecules (other actives + decoys)
            candidate_molecules = [m for m in molecules if m['index'] != query['index']]

            # Rank candidates by similarity to query
            ranked_molecules = e.apply_hook(
                'rank_by_similarity',
                query_molecule=query,
                candidate_molecules=candidate_molecules
            )

            # Evaluate this single query
            metrics = e.apply_hook(
                'evaluate_single_query',
                query_molecule=query,
                ranked_molecules=ranked_molecules
            )

            query_metrics.append(metrics)

        # Average metrics across all queries in this repetition
        rep_avg = {}
        if len(query_metrics) > 0:
            metric_keys = query_metrics[0].keys()
            for key in metric_keys:
                values = [m[key] for m in query_metrics]
                rep_avg[key] = np.mean(values)
        else:
            # Shouldn't happen, but handle gracefully
            rep_avg = {key: 0.0 for key in ['auc', 'bedroc'] + [f'ef{int(p*100)}' for p in e.EF_PERCENTAGES]}

        repetition_results.append(rep_avg)

    # Average across all repetitions
    metric_keys = repetition_results[0].keys()
    target_result = {
        'target_id': target_id,
        'target_name': target_name,
        'n_actives': len(actives),
        'n_decoys': len(decoys),
        'n_total': len(molecules),
    }

    for key in metric_keys:
        values = [r[key] for r in repetition_results]
        target_result[f'{key}_mean'] = float(np.mean(values))
        target_result[f'{key}_std'] = float(np.std(values))
        target_result[f'{key}_median'] = float(np.median(values))

    e.log(f'   - results: AUC={target_result["auc_mean"]:.3f}±{target_result["auc_std"]:.3f}, '
          f'BEDROC={target_result["bedroc_mean"]:.3f}±{target_result["bedroc_std"]:.3f}')

    return target_result


# == VISUALIZATION HOOKS ==
# Hooks for creating publication-quality plots and visualizations


def compute_enrichment_curve(ranked_molecules: List[Tuple[Dict[str, Any], float]],
                            n_points: int = 100
                            ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute enrichment curve showing fraction of actives found vs fraction screened.

    This function takes a ranked list of molecules and computes how many actives
    are recovered at different fractions of the database. This is the fundamental
    data for enrichment plots.

    :param ranked_molecules: List of (molecule_dict, distance) tuples sorted by
        distance (most similar first).
    :param n_points: Number of points to sample along the curve.

    :return: Tuple of (fractions_screened, fractions_actives_found) arrays.

    Example:

    .. code-block:: python

        fractions, actives_found = compute_enrichment_curve(ranked, n_points=100)
        plt.plot(fractions, actives_found, label='Method')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')  # Diagonal reference
    """
    # Extract labels (1 if active, 0 otherwise)
    labels = np.array([
        1 if m[0]['activity_label'] == 'active' else 0
        for m in ranked_molecules
    ])

    n_total = len(labels)
    n_actives = np.sum(labels)

    if n_actives == 0:
        # No actives in list, return flat line at 0
        fractions = np.linspace(0, 1, n_points)
        return fractions, np.zeros(n_points)

    # Compute cumulative actives found at each position
    cumulative_actives = np.cumsum(labels)

    # Sample at n_points positions
    fractions = np.linspace(0, 1, n_points)
    actives_found = []

    for frac in fractions:
        cutoff = int(frac * n_total)
        if cutoff == 0:
            actives_found.append(0.0)
        else:
            found = cumulative_actives[min(cutoff - 1, len(cumulative_actives) - 1)]
            actives_found.append(found / n_actives)

    return fractions, np.array(actives_found)


@experiment.hook('plot_per_target_bars', replace=False, default=True)
def plot_per_target_bars(e: Experiment,
                        target_results: List[Dict[str, Any]],
                        metric: str
                        ) -> Figure:
    """
    Create horizontal bar chart for one metric across all targets.

    This visualization shows performance for each target sorted by the metric,
    with error bars and color coding to quickly identify best and worst performers.

    :param e: The experiment instance.
    :param target_results: List of per-target result dictionaries.
    :param metric: Metric name ('auc', 'bedroc', 'ef1', 'ef5', etc.).

    :return: Matplotlib figure object.
    """
    # Extract data
    targets = [r['target_name'] for r in target_results]
    means = [r[f'{metric}_mean'] for r in target_results]
    stds = [r[f'{metric}_std'] for r in target_results]

    # Sort by performance (ascending)
    sorted_idx = np.argsort(means)
    targets = [targets[i] for i in sorted_idx]
    means_sorted = [means[i] for i in sorted_idx]
    stds_sorted = [stds[i] for i in sorted_idx]

    # Color coding based on performance (normalize to [0, 1])
    max_mean = max(means)
    min_mean = min(means)
    if max_mean > min_mean:
        normalized = [(m - min_mean) / (max_mean - min_mean) for m in means_sorted]
    else:
        normalized = [0.5] * len(means_sorted)

    colors = plt.cm.RdYlGn(normalized)

    # Create figure with appropriate height
    fig_height = max(8, len(targets) * 0.35)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    # Horizontal bars with error bars
    y_pos = np.arange(len(targets))
    ax.barh(y_pos, means_sorted, xerr=stds_sorted, color=colors,
            alpha=0.8, capsize=3, edgecolor='black', linewidth=0.5)

    # Formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels(targets, fontsize=9)
    ax.set_xlabel(f'{metric.upper()} Score', fontsize=12, fontweight='bold')
    ax.set_title(f'Per-Target {metric.upper()} Performance\n(n={len(targets)} targets)',
                fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    # Reference lines
    if metric == 'auc':
        ax.axvline(0.5, color='red', linestyle='--', alpha=0.7,
                  linewidth=2, label='Random (AUC=0.5)')
        ax.legend(loc='lower right', fontsize=10)
    elif metric == 'bedroc':
        ax.axvline(0.0, color='red', linestyle='--', alpha=0.7,
                  linewidth=2, label='Random (BEDROC=0.0)')
        ax.legend(loc='lower right', fontsize=10)

    # Add value labels on bars
    for i, (mean, std) in enumerate(zip(means_sorted, stds_sorted)):
        ax.text(mean + std + 0.02, i, f'{mean:.3f}',
               va='center', fontsize=8, fontweight='bold')

    plt.tight_layout()
    return fig


@experiment.hook('plot_metric_distributions', replace=False, default=True)
def plot_metric_distributions(e: Experiment,
                             target_results: List[Dict[str, Any]]
                             ) -> Figure:
    """
    Create violin plots showing distribution of each metric across targets.

    This visualization helps understand the variance and distribution shape
    of performance metrics, revealing outliers and overall spread.

    :param e: The experiment instance.
    :param target_results: List of per-target result dictionaries.

    :return: Matplotlib figure object.
    """
    # Determine which metrics to plot
    metrics_to_plot = ['auc', 'bedroc'] + [f'ef{int(p*100)}' for p in e.EF_PERCENTAGES]

    # Extract data for each metric
    data = {}
    for metric in metrics_to_plot:
        values = [r[f'{metric}_mean'] for r in target_results]
        data[metric] = values

    # Create subplots
    n_metrics = len(metrics_to_plot)
    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 6))

    if n_metrics == 1:
        axes = [axes]

    # Plot each metric
    for idx, (metric, ax) in enumerate(zip(metrics_to_plot, axes)):
        values = data[metric]

        # Violin plot
        parts = ax.violinplot([values], positions=[0], widths=0.7,
                              showmeans=True, showmedians=True, showextrema=True)

        # Color the violin
        for pc in parts['bodies']:
            pc.set_facecolor('steelblue')
            pc.set_alpha(0.7)

        # Add boxplot overlay for clarity
        bp = ax.boxplot([values], positions=[0], widths=0.3,
                        patch_artist=True, showfliers=False)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_alpha(0.5)

        # Formatting
        ax.set_ylabel(f'{metric.upper()} Score', fontsize=12, fontweight='bold')
        ax.set_title(f'{metric.upper()} Distribution\n(n={len(values)} targets)',
                    fontsize=11, fontweight='bold')
        ax.set_xticks([])
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        # Add statistics text
        mean_val = np.mean(values)
        median_val = np.median(values)
        std_val = np.std(values)

        stats_text = f'Mean: {mean_val:.3f}\nMedian: {median_val:.3f}\nStd: {std_val:.3f}'
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
               verticalalignment='top', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Reference line for random performance
        if metric == 'auc':
            ax.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Random')
            ax.legend(loc='lower right', fontsize=8)
        elif metric == 'bedroc':
            ax.axhline(0.0, color='red', linestyle='--', alpha=0.5, label='Random')
            ax.legend(loc='lower right', fontsize=8)

    plt.tight_layout()
    return fig


@experiment.hook('plot_performance_scatter', replace=False, default=True)
def plot_performance_scatter(e: Experiment,
                            target_results: List[Dict[str, Any]]
                            ) -> Figure:
    """
    Create scatter plots comparing different metrics.

    This visualization shows relationships and correlations between different
    performance metrics, helping to understand if methods excel at specific
    aspects of virtual screening.

    :param e: The experiment instance.
    :param target_results: List of per-target result dictionaries.

    :return: Matplotlib figure object.
    """
    # Extract metric data
    auc_values = [r['auc_mean'] for r in target_results]
    bedroc_values = [r['bedroc_mean'] for r in target_results]

    # Get EF metrics dynamically
    ef_keys = [f'ef{int(p*100)}' for p in e.EF_PERCENTAGES]
    ef_data = {key: [r[f'{key}_mean'] for r in target_results] for key in ef_keys}

    target_names = [r['target_name'] for r in target_results]

    # Create 2x2 subplot grid
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    # Plot 1: AUC vs BEDROC
    ax = axes[0]
    ax.scatter(auc_values, bedroc_values, alpha=0.6, s=100, c='steelblue', edgecolors='black')

    # Add trend line
    z = np.polyfit(auc_values, bedroc_values, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(auc_values), max(auc_values), 100)
    ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)

    # Calculate correlation
    corr = np.corrcoef(auc_values, bedroc_values)[0, 1]

    ax.set_xlabel('AUC', fontsize=12, fontweight='bold')
    ax.set_ylabel('BEDROC', fontsize=12, fontweight='bold')
    ax.set_title(f'AUC vs BEDROC\n(R = {corr:.3f})', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3, linestyle='--')

    # Plot 2: AUC vs EF1% (or first EF)
    ax = axes[1]
    ef1_key = ef_keys[0]
    ef1_values = ef_data[ef1_key]

    ax.scatter(auc_values, ef1_values, alpha=0.6, s=100, c='coral', edgecolors='black')

    z = np.polyfit(auc_values, ef1_values, 1)
    p = np.poly1d(z)
    ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)

    corr = np.corrcoef(auc_values, ef1_values)[0, 1]

    ax.set_xlabel('AUC', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{ef1_key.upper()}', fontsize=12, fontweight='bold')
    ax.set_title(f'AUC vs {ef1_key.upper()}\n(R = {corr:.3f})', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3, linestyle='--')

    # Plot 3: BEDROC vs EF1%
    ax = axes[2]
    ax.scatter(bedroc_values, ef1_values, alpha=0.6, s=100, c='mediumseagreen', edgecolors='black')

    z = np.polyfit(bedroc_values, ef1_values, 1)
    p = np.poly1d(z)
    x_line_bedroc = np.linspace(min(bedroc_values), max(bedroc_values), 100)
    ax.plot(x_line_bedroc, p(x_line_bedroc), "r--", alpha=0.8, linewidth=2)

    corr = np.corrcoef(bedroc_values, ef1_values)[0, 1]

    ax.set_xlabel('BEDROC', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{ef1_key.upper()}', fontsize=12, fontweight='bold')
    ax.set_title(f'BEDROC vs {ef1_key.upper()}\n(R = {corr:.3f})', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3, linestyle='--')

    # Plot 4: Target difficulty (AUC vs n_actives)
    ax = axes[3]
    n_actives = [r['n_actives'] for r in target_results]

    ax.scatter(n_actives, auc_values, alpha=0.6, s=100, c='mediumpurple', edgecolors='black')

    z = np.polyfit(n_actives, auc_values, 1)
    p = np.poly1d(z)
    x_line_actives = np.linspace(min(n_actives), max(n_actives), 100)
    ax.plot(x_line_actives, p(x_line_actives), "r--", alpha=0.8, linewidth=2)

    corr = np.corrcoef(n_actives, auc_values)[0, 1]

    ax.set_xlabel('Number of Actives', fontsize=12, fontweight='bold')
    ax.set_ylabel('AUC', fontsize=12, fontweight='bold')
    ax.set_title(f'Target Size vs Performance\n(R = {corr:.3f})', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3, linestyle='--')
    ax.axhline(0.5, color='red', linestyle='--', alpha=0.5, linewidth=1.5, label='Random')
    ax.legend(loc='best', fontsize=9)

    plt.tight_layout()
    return fig


@experiment.hook('plot_summary_table', replace=False, default=True)
def plot_summary_table(e: Experiment,
                      aggregated: Dict[str, float]
                      ) -> Figure:
    """
    Create publication-ready summary statistics table as image.

    This creates a formatted table showing mean, std, median, min, and max
    for all metrics, suitable for inclusion in papers or reports.

    :param e: The experiment instance.
    :param aggregated: Dictionary of aggregated statistics.

    :return: Matplotlib figure object.
    """
    # Extract metrics
    metrics = ['auc', 'bedroc'] + [f'ef{int(p*100)}' for p in e.EF_PERCENTAGES]

    # Build table data
    table_data = []
    headers = ['Metric', 'Mean', 'Std', 'Median', 'Min', 'Max']

    for metric in metrics:
        if f'{metric}/mean' in aggregated:
            row = [
                metric.upper(),
                f"{aggregated[f'{metric}/mean']:.3f}",
                f"{aggregated[f'{metric}/std']:.3f}",
                f"{aggregated[f'{metric}/median']:.3f}",
                f"{aggregated[f'{metric}/min']:.3f}",
                f"{aggregated[f'{metric}/max']:.3f}",
            ]
            table_data.append(row)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, len(table_data) * 0.6 + 1))
    ax.axis('tight')
    ax.axis('off')

    # Create table
    table = ax.table(cellText=table_data, colLabels=headers,
                    cellLoc='center', loc='center',
                    colWidths=[0.15, 0.15, 0.15, 0.15, 0.15, 0.15])

    # Style table
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)

    # Header styling
    for i in range(len(headers)):
        cell = table[(0, i)]
        cell.set_facecolor('#4CAF50')
        cell.set_text_props(weight='bold', color='white')

    # Row styling with alternating colors
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#f0f0f0')
            else:
                cell.set_facecolor('#ffffff')

    # Title
    title_text = f'Summary Statistics Across {len(table_data)} Metrics\nDataset: {e.DATASET_NAME_ID}'
    ax.set_title(title_text, fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    return fig


@experiment.hook('create_visualizations', replace=False, default=True)
def create_visualizations(e: Experiment,
                         target_results: List[Dict[str, Any]],
                         aggregated: Dict[str, float]
                         ) -> None:
    """
    Orchestrate creation of all visualizations.

    This hook coordinates the creation of all plots and saves them as artifacts.
    It provides a central place to control which visualizations are generated.

    :param e: The experiment instance.
    :param target_results: List of per-target result dictionaries.
    :param aggregated: Dictionary of aggregated statistics across targets.

    :return: None. Saves plots as artifacts.
    """
    e.log('creating visualization artifacts...')

    if len(target_results) == 0:
        e.log('WARNING: No target results available for visualization')
        return

    # 1. Per-target bar charts for each metric
    e.log(' * creating per-target bar charts...')
    metrics_to_plot = ['auc', 'bedroc'] + [f'ef{int(p*100)}' for p in e.EF_PERCENTAGES]

    for metric in metrics_to_plot:
        try:
            fig = e.apply_hook(
                'plot_per_target_bars',
                target_results=target_results,
                metric=metric
            )
            filename = f'per_target_{metric}.png'
            e.commit_fig(filename, fig)
            plt.close(fig)
            e.log(f'   - saved {filename}')
        except Exception as ex:
            e.log(f'   - ERROR creating bar chart for {metric}: {ex}')

    # 2. Metric distributions
    e.log(' * creating metric distribution plots...')
    try:
        fig = e.apply_hook(
            'plot_metric_distributions',
            target_results=target_results
        )
        e.commit_fig('metric_distributions.png', fig)
        plt.close(fig)
        e.log('   - saved metric_distributions.png')
    except Exception as ex:
        e.log(f'   - ERROR creating distribution plots: {ex}')

    # 3. Performance scatter plots
    e.log(' * creating performance scatter plots...')
    try:
        fig = e.apply_hook(
            'plot_performance_scatter',
            target_results=target_results
        )
        e.commit_fig('performance_scatter.png', fig)
        plt.close(fig)
        e.log('   - saved performance_scatter.png')
    except Exception as ex:
        e.log(f'   - ERROR creating scatter plots: {ex}')

    # 4. Summary table
    e.log(' * creating summary statistics table...')
    try:
        fig = e.apply_hook(
            'plot_summary_table',
            aggregated=aggregated
        )
        e.commit_fig('summary_table.png', fig)
        plt.close(fig)
        e.log('   - saved summary_table.png')
    except Exception as ex:
        e.log(f'   - ERROR creating summary table: {ex}')

    e.log(f'visualization creation complete! Created plots for {len(target_results)} targets')


# == MAIN EXPERIMENT ==


@experiment
def main(e: Experiment):
    """
    Main experiment function for similarity-based bioactivity prediction.

    Workflow:
    1. Load and filter bioactivity dataset
    2. Process molecules into representations
    3. Group molecules by target protein
    4. Evaluate each target with multiple repetitions
    5. Aggregate results across all targets
    6. Create visualizations and save results

    :param e: The experiment instance.

    :return: None. Results are saved as artifacts.
    """
    e.log('starting similarity-based bioactivity prediction experiment...')

    # Handle random seed
    if e.SEED is None:
        e.SEED = random.randint(0, 2**31 - 1)
        e.log(f'SEED was None, randomly selected seed: {e.SEED}')

    # Set random seed for reproducibility
    random.seed(e.SEED)
    np.random.seed(e.SEED)

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

    # Verify all indices have graph_features
    missing_features = [idx for idx in index_data_map
                       if 'graph_features' not in index_data_map[idx]]
    if missing_features:
        e.log(f'ERROR: {len(missing_features)} indices missing graph_features')
        raise RuntimeError('Dataset processing did not add graph_features to all molecules')
    else:
        e.log(f'all {len(index_data_map)} indices have graph_features')

    # Check feature dimensions
    first_idx = list(index_data_map.keys())[0]
    feature_dim = len(index_data_map[first_idx]['graph_features'])
    e.log(f'feature dimension: {feature_dim}')

    # == GROUP BY TARGET ==

    e.log('\n=== GROUPING BY TARGET ===')
    target_groups = e.apply_hook('group_by_target', index_data_map=index_data_map)
    e.log(f'grouped into {len(target_groups)} targets')

    # == EVALUATE PER TARGET ==

    e.log('\n=== EVALUATING TARGETS ===')
    e.log(f'running {e.NUM_REPETITIONS} repetitions per target with {e.NUM_QUERY_ACTIVES} queries each')

    target_results = []

    for target_id, molecules in target_groups.items():
        target_name = molecules[0]['target_name'] if len(molecules) > 0 else f'Target {target_id}'

        result = e.apply_hook(
            'evaluate_target',
            target_id=target_id,
            target_name=target_name,
            molecules=molecules
        )

        if result is not None:
            target_results.append(result)

    e.log(f'\nevaluated {len(target_results)} targets (skipped {len(target_groups) - len(target_results)})')

    if len(target_results) == 0:
        e.log('ERROR: No targets could be evaluated!')
        return

    # == AGGREGATE RESULTS ==

    e.log('\n=== AGGREGATING RESULTS ===')

    # Extract metric keys (e.g., 'auc_mean', 'bedroc_mean', etc.)
    metric_keys = [key for key in target_results[0].keys()
                  if key.endswith('_mean')]

    aggregated = {}
    for key in metric_keys:
        values = [r[key] for r in target_results]
        base_metric = key.replace('_mean', '')

        aggregated[f'{base_metric}/mean'] = float(np.mean(values))
        aggregated[f'{base_metric}/std'] = float(np.std(values))
        aggregated[f'{base_metric}/median'] = float(np.median(values))
        aggregated[f'{base_metric}/min'] = float(np.min(values))
        aggregated[f'{base_metric}/max'] = float(np.max(values))

    # Log aggregated results
    e.log('\nAggregated Performance (mean ± std):')
    for metric in ['auc', 'bedroc'] + [f'ef{int(p*100)}' for p in e.EF_PERCENTAGES]:
        if f'{metric}/mean' in aggregated:
            mean = aggregated[f'{metric}/mean']
            std = aggregated[f'{metric}/std']
            e.log(f' * {metric.upper()}: {mean:.3f} ± {std:.3f}')

    # Store in experiment
    for key, value in aggregated.items():
        e[f'metrics/{key}'] = value

    # == SAVE RESULTS ==

    e.log('\n=== SAVING RESULTS ===')

    # Per-target results CSV
    per_target_df = pd.DataFrame(target_results)
    per_target_path = os.path.join(e.path, 'per_target_results.csv')
    per_target_df.to_csv(per_target_path, index=False)
    e.log(f'saved per-target results to {per_target_path}')

    # Aggregated results CSV
    agg_rows = []
    for metric in ['auc', 'bedroc'] + [f'ef{int(p*100)}' for p in e.EF_PERCENTAGES]:
        if f'{metric}/mean' in aggregated:
            agg_rows.append({
                'metric': metric,
                'mean': aggregated[f'{metric}/mean'],
                'std': aggregated[f'{metric}/std'],
                'median': aggregated[f'{metric}/median'],
                'min': aggregated[f'{metric}/min'],
                'max': aggregated[f'{metric}/max'],
            })

    agg_df = pd.DataFrame(agg_rows)
    agg_path = os.path.join(e.path, 'aggregated_results.csv')
    agg_df.to_csv(agg_path, index=False)
    e.log(f'saved aggregated results to {agg_path}')

    # == CREATE VISUALIZATIONS ==

    e.log('\n=== CREATING VISUALIZATIONS ===')

    try:
        e.apply_hook(
            'create_visualizations',
            target_results=target_results,
            aggregated=aggregated
        )
    except Exception as ex:
        e.log(f'ERROR during visualization creation: {ex}')
        import traceback
        e.log(traceback.format_exc())

    e.log('\nexperiment complete!')


experiment.run_if_main()
