"""
Molecular Similarity Experiment

This experiment analyzes molecular similarity by encoding molecules into vector
representations and finding nearest neighbors based on representation distance.
The encoding method and distance metric are configurable via hooks, allowing
comparison of different molecular representation schemes (HDC, fingerprints, etc.).

Key Workflow:
    1. Load and filter molecular dataset from chem_mat_data
    2. Encode molecules into vector representations (via hook - HDC, fingerprints, etc.)
    3. Select random query molecules for similarity analysis
    4. For each query, find K nearest (most similar) neighbors using distance metric
    5. Optionally, find K most dissimilar molecules (furthest neighbors)
    6. Visualize query molecules alongside similar/dissimilar neighbors with scores

GED Correlation Analysis (Optional):
    When ENABLE_GED_ANALYSIS is set to True, the experiment additionally performs:
    1. Generate N-hop molecular neighborhoods using graph edit operations
    2. Track graph edit distance (GED) for each neighbor based on hop level
    3. Compute embedding similarity between query and all neighbors
    4. Calculate Pearson correlation and R² between GED and embedding similarity
    5. Create regression plots showing GED vs similarity relationship
    6. Aggregate statistics (average R², correlation) across all queries

The experiment is designed to be extended by sub-experiments that implement
specific encoding methods and distance metrics:
    - molecule_similarity__hdc.py: HDC encoding with cosine distance
    - molecule_similarity__fp.py: Fingerprint encoding with Tanimoto distance

Design Rationale:
    - Hook-based architecture enables comparison of different representation methods
    - Individual visualizations per query enable detailed inspection
    - Both similar and dissimilar molecules provide full spectrum of diversity
    - GED correlation analysis reveals relationship between structural and embedding distance
    - Configurable parameters allow adaptation to different use cases
    - Results stored in experiment storage for programmatic access

Usage:
    Create configuration files extending this experiment and specifying the
    molecular representation method:

    .. code-block:: yaml

        extend: molecule_similarity__hdc.py
        parameters:
            DATASET_NAME: "qm9_smiles"
            NUM_SAMPLES: 10
            NUM_NEIGHBORS: 5
            FIND_DISSIMILAR: true
            # Enable GED correlation analysis
            ENABLE_GED_ANALYSIS: true
            NUM_HOPS: 3
            NUM_NEIGHBOR_BRANCHES: 5
            NUM_NEIGHBOR_TOTAL: 20
            GED_NUM_SAMPLES: 10
            SEED: 1

Output Artifacts:
    Similarity Search:
        - similarity_query_{idx}_{hash}.png: Individual similarity visualizations
        - dissimilarity_query_{idx}_{hash}.png: Individual dissimilarity visualizations
        - similarity_summary.csv: Summary statistics with similar/dissimilar molecules
    GED Correlation Analysis (if enabled):
        - ged_regression_query_{idx}_{hash}.png: GED vs similarity regression plots
        - ged_correlation_summary.csv: Per-molecule and aggregate correlation statistics
"""
import os
import time
import random
import hashlib
from typing import Any, List, Union, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from rich.pretty import pprint
from rdkit import Chem
from rdkit.Chem import Draw

from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace
from chem_mat_data._typing import GraphDict
from chem_mat_data.main import load_graph_dataset

# GED correlation analysis imports
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
import seaborn as sns

# vgd_counterfactuals for neighborhood generation
import sys
sys.path.insert(0, '/tmp/vgd_counterfactuals')
from vgd_counterfactuals.generate.molecules import get_neighborhood

# == DATASET PARAMETERS ==

# :param DATASET_NAME:
#       The name of the dataset to be used for the experiment. This name is used
#       to download the dataset from the ChemMatData file share.
DATASET_NAME: str = 'aqsoldb'

# :param DATASET_NAME_ID:
#       The name of the dataset to be used for identification purposes.
DATASET_NAME_ID: str = DATASET_NAME

# :param NUM_DATA:
#       The number of samples to be used for the experiment. This parameter can be
#       either an integer or a float between 0 and 1. If None, the entire dataset
#       is used.
NUM_DATA: Union[int, float, None] = None

# :param SEED:
#       The random seed to be used for the experiment. If None, random processes
#       will not be seeded, resulting in different outcomes across repetitions.
SEED: Union[int, None] = 1

# == SIMILARITY SEARCH PARAMETERS ==

# :param NUM_SAMPLES:
#       The number of query molecules to randomly select for similarity analysis.
#       For each query molecule, the K nearest neighbors will be found and
#       visualized. A value of 10 provides a good balance between coverage and
#       computational cost.
NUM_SAMPLES: int = 10

# :param NUM_NEIGHBORS:
#       The number of nearest neighbors (K) to find for each query molecule.
#       These are the most similar molecules according to the distance metric.
#       The default value of 5 provides a manageable visualization while showing
#       the diversity of similar molecules.
NUM_NEIGHBORS: int = 5

# :param FIND_DISSIMILAR:
#       Whether to find and visualize the K most dissimilar molecules in addition
#       to the most similar ones. When enabled, the experiment will identify the
#       furthest neighbors for each query molecule and create separate visualization
#       artifacts. This helps understand the full spectrum of molecular diversity
#       and provides insights into what makes molecules different from each other.
FIND_DISSIMILAR: bool = True

# :param USE_FULL_DATASET_FOR_SEARCH:
#       Whether to search the entire dataset for neighbors (True) or exclude
#       the query molecule itself (False). Setting to False is recommended to
#       avoid trivial self-matches.
USE_FULL_DATASET_FOR_SEARCH: bool = False

# == VISUALIZATION PARAMETERS ==

# :param MOLECULE_IMAGE_SIZE:
#       The size (width, height) in pixels for rendering individual molecule images.
#       Larger sizes provide more detail but increase file size.
MOLECULE_IMAGE_SIZE: Tuple[int, int] = (300, 300)

# :param GRID_COLS:
#       The number of columns in the neighbor grid visualization. The query molecule
#       is shown in the first row (centered), and neighbors are arranged in subsequent
#       rows with this many columns per row.
GRID_COLS: int = 3

# :param SMILES_TRUNCATE_LENGTH:
#       Maximum length for SMILES strings displayed in plot titles. Long SMILES
#       strings are truncated to this length with '...' appended.
SMILES_TRUNCATE_LENGTH: int = 30

# == GED CORRELATION ANALYSIS PARAMETERS ==

# :param ENABLE_GED_ANALYSIS:
#       Whether to enable graph edit distance (GED) correlation analysis. When enabled,
#       the experiment will generate N-hop molecular neighborhoods, track graph edit
#       distances, and analyze the correlation between GED and embedding similarity.
ENABLE_GED_ANALYSIS: bool = False

# :param NUM_HOPS:
#       Maximum hop distance to explore when generating molecular neighborhoods.
#       A value of 3 means exploring molecules up to 3 graph edits away from the
#       original query molecule. Higher values provide more comprehensive analysis
#       but increase computational cost exponentially.
NUM_HOPS: int = 3

# :param NUM_NEIGHBOR_BRANCHES:
#       Number of neighbors to sample per branch when building the neighborhood tree.
#       For each molecule at the current level, this many neighbors are randomly
#       sampled from its 1-hop neighborhood to continue the branching process.
NUM_NEIGHBOR_BRANCHES: int = 5

# :param NUM_NEIGHBOR_TOTAL:
#       Total number of neighbors to generate per hop level. The branching process
#       continues until this many molecules are collected at each level. This controls
#       the size of the neighborhood at each hop distance.
NUM_NEIGHBOR_TOTAL: int = 20

# :param GED_NUM_SAMPLES:
#       Number of query molecules to analyze for GED correlation. This is separate
#       from NUM_SAMPLES to allow different sampling strategies for similarity search
#       and GED analysis.
GED_NUM_SAMPLES: int = 10

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

    This hook removes molecules with invalid SMILES strings, molecules with
    fewer than 2 atoms, molecules with no bonds, and disconnected graphs
    (indicated by '.' in SMILES).

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
        "representation implementation (e.g., molecule_similarity__hdc.py)."
    )


@experiment.hook('encode_molecule', replace=False, default=True)
def encode_molecule(e: Experiment,
                   smiles: str
                   ) -> Union[np.ndarray, None]:
    """
    Encode a single molecule using the already-initialized encoder.

    **IMPORTANT:** This hook must be overridden in extending experiments to
    provide molecular encoding using the EXISTING encoder that was created
    during process_dataset. This hook should NOT create a new encoder.

    The encoder should be stored as a private instance attribute (e.g., e._encoder)
    during the initial process_dataset call, and this hook should retrieve and
    use that same encoder. Use private attributes to avoid JSON serialization errors.

    :param e: The experiment instance.
    :param smiles: SMILES string of the molecule to encode.

    :return: Feature vector as numpy array, or None if encoding fails.

    :raises NotImplementedError: This default implementation raises an error.
    """
    raise NotImplementedError(
        "The 'encode_molecule' hook must be overridden to provide single-molecule "
        "encoding using the existing encoder. Store the encoder as a private instance "
        "attribute (e.g., e._encoder = encoder) to avoid JSON serialization issues."
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


@experiment.hook('select_query_samples', replace=False, default=True)
def select_query_samples(e: Experiment,
                        index_data_map: dict[int, GraphDict]
                        ) -> List[int]:
    """
    Select random query molecules for similarity analysis.

    This hook randomly selects NUM_SAMPLES molecules from the dataset to use
    as queries for nearest neighbor search. The selection is deterministic
    when SEED is set.

    :param e: The experiment instance.
    :param index_data_map: Dictionary of all molecules in the dataset.

    :return: List of selected query indices.
    """
    e.log(f'selecting {e.NUM_SAMPLES} random query molecules...')

    available_indices = list(index_data_map.keys())

    if e.NUM_SAMPLES > len(available_indices):
        e.log(f'WARNING: NUM_SAMPLES ({e.NUM_SAMPLES}) exceeds dataset size '
              f'({len(available_indices)}). Using all samples.')
        query_indices = available_indices
    else:
        random.seed(e.SEED)
        query_indices = random.sample(available_indices, k=e.NUM_SAMPLES)

    e.log(f'selected query indices: {query_indices}')
    return query_indices


@experiment.hook('find_neighbors', replace=False, default=True)
def find_neighbors(e: Experiment,
                   query_idx: int,
                   query_features: np.ndarray,
                   index_data_map: dict[int, GraphDict],
                   candidate_indices: List[int]
                   ) -> List[Tuple[int, float]]:
    """
    Find K nearest neighbors for a query molecule.

    This hook computes distances from the query molecule to all candidate
    molecules and returns the K nearest neighbors sorted by distance.

    :param e: The experiment instance.
    :param query_idx: Index of the query molecule.
    :param query_features: Feature vector of the query molecule.
    :param index_data_map: Dictionary of all molecules in the dataset.
    :param candidate_indices: Indices of candidate molecules to search.

    :return: List of (index, distance) tuples for the K nearest neighbors,
        sorted by distance (ascending).
    """
    distances = []

    for candidate_idx in candidate_indices:
        if candidate_idx == query_idx:
            continue  # Skip self

        candidate_features = index_data_map[candidate_idx]['graph_features']

        # Compute distance using the hook
        distance = e.apply_hook(
            'compute_distance',
            features1=query_features,
            features2=candidate_features
        )

        distances.append((candidate_idx, distance))

    # Sort by distance (ascending) and take top K
    distances.sort(key=lambda x: x[1])
    return distances[:e.NUM_NEIGHBORS]


@experiment.hook('find_dissimilar_neighbors', replace=False, default=True)
def find_dissimilar_neighbors(e: Experiment,
                              query_idx: int,
                              query_features: np.ndarray,
                              index_data_map: dict[int, GraphDict],
                              candidate_indices: List[int]
                              ) -> List[Tuple[int, float]]:
    """
    Find K most dissimilar molecules (furthest neighbors) for a query molecule.

    This hook computes distances from the query molecule to all candidate
    molecules and returns the K most dissimilar molecules sorted by distance
    in descending order. This helps identify molecules that are maximally
    different from the query molecule according to the distance metric.

    :param e: The experiment instance.
    :param query_idx: Index of the query molecule.
    :param query_features: Feature vector of the query molecule.
    :param index_data_map: Dictionary of all molecules in the dataset.
    :param candidate_indices: Indices of candidate molecules to search.

    :return: List of (index, distance) tuples for the K most dissimilar molecules,
        sorted by distance (descending).
    """
    distances = []

    for candidate_idx in candidate_indices:
        if candidate_idx == query_idx:
            continue  # Skip self

        candidate_features = index_data_map[candidate_idx]['graph_features']

        # Compute distance using the hook
        distance = e.apply_hook(
            'compute_distance',
            features1=query_features,
            features2=candidate_features
        )

        distances.append((candidate_idx, distance))

    # Sort by distance (descending) and take top K most dissimilar
    distances.sort(key=lambda x: x[1], reverse=True)
    return distances[:e.NUM_NEIGHBORS]


@experiment.hook('visualize_neighbors', replace=False, default=True)
def visualize_neighbors(e: Experiment,
                       query_idx: int,
                       neighbor_results: List[Tuple[int, float]],
                       index_data_map: dict[int, GraphDict]
                       ) -> Figure:
    """
    Create a visualization of a query molecule and its nearest neighbors.

    This hook generates a grid visualization showing the query molecule in
    the first row (centered) and its K nearest neighbors in subsequent rows,
    with similarity scores displayed in the titles.

    :param e: The experiment instance.
    :param query_idx: Index of the query molecule.
    :param neighbor_results: List of (neighbor_idx, distance) tuples.
    :param index_data_map: Dictionary of all molecules in the dataset.

    :return: Matplotlib figure object.
    """
    n_neighbors = len(neighbor_results)
    n_rows = ((n_neighbors + e.GRID_COLS - 1) // e.GRID_COLS) + 1  # +1 for query row

    fig, axes = plt.subplots(
        n_rows, e.GRID_COLS,
        figsize=(e.GRID_COLS * 4, n_rows * 4)
    )

    # Ensure axes is 2D array even for single row
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    # Plot query in first row, centered
    query_smiles = index_data_map[query_idx]['graph_repr']
    query_mol = Chem.MolFromSmiles(query_smiles)

    if query_mol is not None:
        query_img = Draw.MolToImage(query_mol, size=e.MOLECULE_IMAGE_SIZE)
        center_col = e.GRID_COLS // 2

        axes[0, center_col].imshow(query_img)
        truncated_smiles = (query_smiles[:e.SMILES_TRUNCATE_LENGTH] + '...'
                           if len(query_smiles) > e.SMILES_TRUNCATE_LENGTH
                           else query_smiles)
        axes[0, center_col].set_title(
            f'Query Molecule\n{truncated_smiles}',
            fontsize=12,
            fontweight='bold'
        )
        axes[0, center_col].axis('off')

    # Hide unused query row axes
    for col in range(e.GRID_COLS):
        if col != e.GRID_COLS // 2:
            axes[0, col].axis('off')

    # Plot neighbors in subsequent rows
    for i, (neighbor_idx, distance) in enumerate(neighbor_results):
        row = (i // e.GRID_COLS) + 1
        col = i % e.GRID_COLS

        neighbor_smiles = index_data_map[neighbor_idx]['graph_repr']
        neighbor_mol = Chem.MolFromSmiles(neighbor_smiles)

        if neighbor_mol is not None:
            neighbor_img = Draw.MolToImage(neighbor_mol, size=e.MOLECULE_IMAGE_SIZE)

            axes[row, col].imshow(neighbor_img)
            similarity = 1 - distance  # Convert distance to similarity
            truncated_smiles = (neighbor_smiles[:e.SMILES_TRUNCATE_LENGTH] + '...'
                               if len(neighbor_smiles) > e.SMILES_TRUNCATE_LENGTH
                               else neighbor_smiles)
            axes[row, col].set_title(
                f'Neighbor {i + 1}\nSimilarity: {similarity:.3f}\n{truncated_smiles}',
                fontsize=10
            )
            axes[row, col].axis('off')

    # Hide unused axes in the last row
    total_plots = 1 + n_neighbors  # query + neighbors
    for i in range(total_plots, n_rows * e.GRID_COLS):
        row = i // e.GRID_COLS
        col = i % e.GRID_COLS
        if row < n_rows:
            axes[row, col].axis('off')

    plt.tight_layout()
    return fig


@experiment.hook('visualize_dissimilar_neighbors', replace=False, default=True)
def visualize_dissimilar_neighbors(e: Experiment,
                                   query_idx: int,
                                   dissimilar_results: List[Tuple[int, float]],
                                   index_data_map: dict[int, GraphDict]
                                   ) -> Figure:
    """
    Create a visualization of a query molecule and its most dissimilar molecules.

    This hook generates a grid visualization showing the query molecule in
    the first row (centered) and its K most dissimilar molecules in subsequent rows,
    with dissimilarity scores displayed in the titles. Dissimilarity is shown as
    the distance value itself (higher = more dissimilar).

    :param e: The experiment instance.
    :param query_idx: Index of the query molecule.
    :param dissimilar_results: List of (molecule_idx, distance) tuples.
    :param index_data_map: Dictionary of all molecules in the dataset.

    :return: Matplotlib figure object.
    """
    n_dissimilar = len(dissimilar_results)
    n_rows = ((n_dissimilar + e.GRID_COLS - 1) // e.GRID_COLS) + 1  # +1 for query row

    fig, axes = plt.subplots(
        n_rows, e.GRID_COLS,
        figsize=(e.GRID_COLS * 4, n_rows * 4)
    )

    # Ensure axes is 2D array even for single row
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    # Plot query in first row, centered
    query_smiles = index_data_map[query_idx]['graph_repr']
    query_mol = Chem.MolFromSmiles(query_smiles)

    if query_mol is not None:
        query_img = Draw.MolToImage(query_mol, size=e.MOLECULE_IMAGE_SIZE)
        center_col = e.GRID_COLS // 2

        axes[0, center_col].imshow(query_img)
        truncated_smiles = (query_smiles[:e.SMILES_TRUNCATE_LENGTH] + '...'
                           if len(query_smiles) > e.SMILES_TRUNCATE_LENGTH
                           else query_smiles)
        axes[0, center_col].set_title(
            f'Query Molecule\n{truncated_smiles}',
            fontsize=12,
            fontweight='bold'
        )
        axes[0, center_col].axis('off')

    # Hide unused query row axes
    for col in range(e.GRID_COLS):
        if col != e.GRID_COLS // 2:
            axes[0, col].axis('off')

    # Plot dissimilar molecules in subsequent rows
    for i, (dissimilar_idx, distance) in enumerate(dissimilar_results):
        row = (i // e.GRID_COLS) + 1
        col = i % e.GRID_COLS

        dissimilar_smiles = index_data_map[dissimilar_idx]['graph_repr']
        dissimilar_mol = Chem.MolFromSmiles(dissimilar_smiles)

        if dissimilar_mol is not None:
            dissimilar_img = Draw.MolToImage(dissimilar_mol, size=e.MOLECULE_IMAGE_SIZE)

            axes[row, col].imshow(dissimilar_img)
            dissimilarity = distance  # Distance itself is dissimilarity
            truncated_smiles = (dissimilar_smiles[:e.SMILES_TRUNCATE_LENGTH] + '...'
                               if len(dissimilar_smiles) > e.SMILES_TRUNCATE_LENGTH
                               else dissimilar_smiles)
            axes[row, col].set_title(
                f'Dissimilar {i + 1}\nDissimilarity: {dissimilarity:.3f}\n{truncated_smiles}',
                fontsize=10
            )
            axes[row, col].axis('off')

    # Hide unused axes in the last row
    total_plots = 1 + n_dissimilar  # query + dissimilar molecules
    for i in range(total_plots, n_rows * e.GRID_COLS):
        row = i // e.GRID_COLS
        col = i % e.GRID_COLS
        if row < n_rows:
            axes[row, col].axis('off')

    plt.tight_layout()
    return fig


# == GED CORRELATION ANALYSIS HOOKS ==


@experiment.hook('calculate_n_hop_neighborhood', replace=False, default=True)
def calculate_n_hop_neighborhood(e: Experiment,
                                 query_smiles: str
                                 ) -> List[Tuple[str, int]]:
    """
    Calculate N-hop neighborhood of a molecule using branching strategy.

    This hook generates a multi-hop molecular neighborhood by:
    1. Starting with the query molecule
    2. For each hop level (1 to NUM_HOPS):
       - Sample molecules from the previous level
       - Generate 1-hop neighbors using get_neighborhood()
       - Sample NUM_NEIGHBOR_BRANCHES from each parent
       - Continue until NUM_NEIGHBOR_TOTAL molecules are collected at this level
    3. Track graph edit distance (GED) for each neighbor based on hop level

    :param e: The experiment instance.
    :param query_smiles: SMILES string of the query molecule.

    :return: List of tuples (smiles, ged) where ged is the hop distance (1, 2, 3, ...).
    """
    e.log(f'calculating {e.NUM_HOPS}-hop neighborhood for query molecule...')

    # Track all neighbors with their GED
    all_neighbors = []

    # Track unique SMILES to avoid duplicates
    seen_smiles = {query_smiles}

    # Current level starts with just the query molecule
    current_level = [query_smiles]

    for hop in range(1, e.NUM_HOPS + 1):
        e.log(f'  generating hop {hop} neighbors...')
        next_level = []
        next_level_set = set()

        # Keep sampling until we reach NUM_NEIGHBOR_TOTAL for this level
        attempts = 0
        max_attempts = e.NUM_NEIGHBOR_TOTAL * 10  # Prevent infinite loops

        while len(next_level) < e.NUM_NEIGHBOR_TOTAL and attempts < max_attempts:
            attempts += 1

            # Sample a random molecule from current level
            if not current_level:
                e.log(f'    WARNING: No molecules left in current level at hop {hop}')
                break

            parent_smiles = random.choice(current_level)

            try:
                # Generate 1-hop neighbors using vgd_counterfactuals
                neighbors_data = get_neighborhood(
                    parent_smiles,
                    use_atom_additions=True,
                    use_bond_additions=False,
                    use_bond_removals=True,
                )

                # Extract SMILES strings from the returned data
                neighbor_smiles_list = [n['value'] for n in neighbors_data]

                # Sample up to NUM_NEIGHBOR_BRANCHES neighbors
                num_to_sample = min(e.NUM_NEIGHBOR_BRANCHES, len(neighbor_smiles_list))
                if num_to_sample > 0:
                    sampled_neighbors = random.sample(neighbor_smiles_list, num_to_sample)

                    # Add new unique neighbors
                    for neighbor_smiles in sampled_neighbors:
                        if neighbor_smiles not in seen_smiles and neighbor_smiles not in next_level_set:
                            next_level.append(neighbor_smiles)
                            next_level_set.add(neighbor_smiles)
                            seen_smiles.add(neighbor_smiles)
                            all_neighbors.append((neighbor_smiles, hop))

                            # Check if we've reached the target
                            if len(next_level) >= e.NUM_NEIGHBOR_TOTAL:
                                break

            except Exception as ex:
                e.log(f'    WARNING: Failed to generate neighbors for {parent_smiles[:30]}: {ex}')
                continue

        e.log(f'    collected {len(next_level)} unique neighbors at hop {hop}')
        current_level = next_level

        # If no new neighbors were generated, stop early
        if len(current_level) == 0:
            e.log(f'    No new neighbors generated, stopping at hop {hop}')
            break

    e.log(f'  total neighborhood size: {len(all_neighbors)} molecules')
    return all_neighbors


@experiment.hook('compute_ged_similarity_correlation', replace=False, default=True)
def compute_ged_similarity_correlation(e: Experiment,
                                      query_idx: int,
                                      query_features: np.ndarray,
                                      neighbors_with_ged: List[Tuple[str, int]],
                                      index_data_map: dict[int, GraphDict]
                                      ) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    """
    Compute correlation between graph edit distance and embedding similarity.

    This hook:
    1. Encodes neighbor molecules using the EXISTING encoder (no recreation!)
    2. Computes embedding similarity (1 - distance) for each neighbor
    3. Calculates Pearson correlation between GED and similarity
    4. Calculates R² score using linear regression

    :param e: The experiment instance.
    :param query_idx: Index of the query molecule.
    :param query_features: Feature vector of the query molecule.
    :param neighbors_with_ged: List of (smiles, ged) tuples for all neighbors.
    :param index_data_map: Dictionary of all molecules in the dataset.

    :return: Tuple of (r2_score, correlation, p_value, ged_array, similarity_array).
    """
    e.log(f'  computing GED-similarity correlation for {len(neighbors_with_ged)} neighbors...')
    e.log(f'    encoding neighbors using EXISTING encoder (no recreation)...')

    ged_values = []
    similarity_values = []
    skipped_invalid = 0
    skipped_disconnected = 0
    encoding_failed = 0

    for neighbor_smiles, ged in neighbors_with_ged:
        # Pre-validate molecule
        mol = Chem.MolFromSmiles(neighbor_smiles)
        if mol is None:
            skipped_invalid += 1
            continue

        # Check for disconnected graphs or single atoms
        if '.' in neighbor_smiles or len(mol.GetAtoms()) < 2:
            skipped_disconnected += 1
            continue

        try:
            # Use the EXISTING encoder via encode_molecule hook
            # This reuses the encoder created during initial process_dataset
            neighbor_features = e.apply_hook('encode_molecule', smiles=neighbor_smiles)

            if neighbor_features is None:
                encoding_failed += 1
                continue

            # Compute distance using the hook
            distance = e.apply_hook(
                'compute_distance',
                features1=query_features,
                features2=neighbor_features
            )

            # Convert distance to similarity (higher = more similar)
            similarity = 1 - distance

            ged_values.append(ged)
            similarity_values.append(similarity)

        except Exception as ex:
            e.log(f'    WARNING: Error encoding neighbor {neighbor_smiles[:30]}: {ex}')
            encoding_failed += 1
            continue

    # Log statistics
    total_neighbors = len(neighbors_with_ged)
    successful = len(ged_values)
    e.log(f'    successfully encoded {successful}/{total_neighbors} neighbors')
    if skipped_invalid > 0:
        e.log(f'    skipped {skipped_invalid} invalid SMILES')
    if skipped_disconnected > 0:
        e.log(f'    skipped {skipped_disconnected} disconnected/single-atom molecules')
    if encoding_failed > 0:
        e.log(f'    {encoding_failed} encoding failures')

    if len(ged_values) < 2:
        e.log(f'    ERROR: Insufficient valid neighbors ({len(ged_values)}) for correlation')
        return 0.0, 0.0, 1.0, np.array([]), np.array([])

    ged_array = np.array(ged_values)
    similarity_array = np.array(similarity_values)

    # Calculate Pearson correlation
    correlation, p_value = pearsonr(ged_array, similarity_array)

    # Calculate R² using linear regression
    ged_reshaped = ged_array.reshape(-1, 1)
    reg = LinearRegression()
    reg.fit(ged_reshaped, similarity_array)
    predictions = reg.predict(ged_reshaped)
    r2 = r2_score(similarity_array, predictions)

    e.log(f'    Pearson correlation: {correlation:.4f} (p={p_value:.4e})')
    e.log(f'    R² score: {r2:.4f}')

    return r2, correlation, p_value, ged_array, similarity_array


@experiment.hook('visualize_ged_regression', replace=False, default=True)
def visualize_ged_regression(e: Experiment,
                             query_smiles: str,
                             ged_array: np.ndarray,
                             similarity_array: np.ndarray,
                             r2: float,
                             correlation: float,
                             p_value: float
                             ) -> Figure:
    """
    Create a regression plot showing GED vs embedding similarity.

    This hook generates a scatter plot with:
    - GED on x-axis
    - Embedding similarity on y-axis
    - Linear regression line
    - Annotated with R² and correlation coefficient

    :param e: The experiment instance.
    :param query_smiles: SMILES string of the query molecule.
    :param ged_array: Array of graph edit distances.
    :param similarity_array: Array of embedding similarities.
    :param r2: R² score.
    :param correlation: Pearson correlation coefficient.
    :param p_value: P-value for correlation.

    :return: Matplotlib figure object.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create regression plot using seaborn
    sns.regplot(
        x=ged_array,
        y=similarity_array,
        ax=ax,
        scatter_kws={'alpha': 0.6, 's': 50},
        line_kws={'color': 'red', 'linewidth': 2}
    )

    # Customize plot
    ax.set_xlabel('Graph Edit Distance (GED)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Embedding Similarity', fontsize=14, fontweight='bold')

    truncated_smiles = (query_smiles[:e.SMILES_TRUNCATE_LENGTH] + '...'
                       if len(query_smiles) > e.SMILES_TRUNCATE_LENGTH
                       else query_smiles)
    ax.set_title(
        f'GED vs Embedding Similarity\nQuery: {truncated_smiles}',
        fontsize=16,
        fontweight='bold'
    )

    # Add statistics annotation
    stats_text = (
        f'R² = {r2:.4f}\n'
        f'Pearson r = {correlation:.4f}\n'
        f'p-value = {p_value:.4e}\n'
        f'n = {len(ged_array)} neighbors'
    )

    ax.text(
        0.05, 0.95,
        stats_text,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    )

    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    return fig


# == MAIN EXPERIMENT ==


@experiment
def main(e: Experiment):
    """
    Main experiment function for molecular similarity analysis.

    Workflow:
    1. Load and filter molecular dataset
    2. Process molecules into representations
    3. Select query molecules
    4. For each query:
       - Find nearest (most similar) neighbors
       - Optionally find furthest (most dissimilar) molecules
       - Visualize both similar and dissimilar molecules
    5. Save summary statistics and visualizations

    :param e: The experiment instance.

    :return: None. Results are saved as artifacts.
    """
    e.log('starting molecular similarity experiment...')

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
        e.log(f'Missing indices: {missing_features[:10]}...')
        raise RuntimeError('Dataset processing did not add graph_features to all molecules')
    else:
        e.log(f'all {len(index_data_map)} indices have graph_features')

    # Check feature dimensions
    first_idx = list(index_data_map.keys())[0]
    feature_dim = len(index_data_map[first_idx]['graph_features'])
    e.log(f'feature dimension: {feature_dim}')

    # == SELECT QUERY SAMPLES ==

    e.log('\n=== SELECTING QUERY SAMPLES ===')
    query_indices = e.apply_hook('select_query_samples', index_data_map=index_data_map)
    e.log(f'selected {len(query_indices)} query molecules')

    # == SIMILARITY SEARCH ==

    e.log('\n=== PERFORMING SIMILARITY SEARCH ===')

    # Determine candidate indices for neighbor search
    if e.USE_FULL_DATASET_FOR_SEARCH:
        candidate_indices = list(index_data_map.keys())
        e.log(f'searching across full dataset ({len(candidate_indices)} molecules)')
    else:
        candidate_indices = list(index_data_map.keys())
        e.log(f'searching across dataset excluding query '
              f'({len(candidate_indices)} molecules)')

    # Store results
    all_results = []

    for i, query_idx in enumerate(query_indices):
        e.log(f'\nprocessing query {i + 1}/{len(query_indices)} (index={query_idx})...')

        query_smiles = index_data_map[query_idx]['graph_repr']
        query_features = index_data_map[query_idx]['graph_features']

        e.log(f' * query SMILES: {query_smiles}')

        # Find neighbors
        time_start = time.time()
        neighbor_results = e.apply_hook(
            'find_neighbors',
            query_idx=query_idx,
            query_features=query_features,
            index_data_map=index_data_map,
            candidate_indices=candidate_indices
        )
        time_end = time.time()

        e.log(f' * found {len(neighbor_results)} neighbors in {time_end - time_start:.3f}s')

        # Log neighbor details
        for j, (neighbor_idx, distance) in enumerate(neighbor_results):
            neighbor_smiles = index_data_map[neighbor_idx]['graph_repr']
            similarity = 1 - distance
            e.log(f'   {j + 1}. idx={neighbor_idx}, similarity={similarity:.4f}, '
                  f'SMILES={neighbor_smiles[:40]}...')

        # Store results in experiment storage
        query_result = {
            'query_idx': query_idx,
            'query_smiles': query_smiles,
            'neighbors': [
                {
                    'index': idx,
                    'distance': dist,
                    'similarity': 1 - dist,
                    'smiles': index_data_map[idx]['graph_repr']
                }
                for idx, dist in neighbor_results
            ]
        }
        all_results.append(query_result)

        e[f'similarity/query_{i}/index'] = query_idx
        e[f'similarity/query_{i}/smiles'] = query_smiles
        e[f'similarity/query_{i}/neighbors'] = query_result['neighbors']

        # Visualize
        e.log(f' * creating visualization...')
        fig = e.apply_hook(
            'visualize_neighbors',
            query_idx=query_idx,
            neighbor_results=neighbor_results,
            index_data_map=index_data_map
        )

        # Generate safe filename
        smiles_hash = hashlib.md5(query_smiles.encode()).hexdigest()[:8]
        filename = f'similarity_query_{i}_{smiles_hash}.png'
        e.commit_fig(filename, fig)
        e.log(f' * saved visualization to {filename}')

        plt.close(fig)

        # == DISSIMILARITY SEARCH (if enabled) ==
        if e.FIND_DISSIMILAR:
            e.log(f' * finding most dissimilar molecules...')

            # Find dissimilar neighbors
            time_start_dissim = time.time()
            dissimilar_results = e.apply_hook(
                'find_dissimilar_neighbors',
                query_idx=query_idx,
                query_features=query_features,
                index_data_map=index_data_map,
                candidate_indices=candidate_indices
            )
            time_end_dissim = time.time()

            e.log(f' * found {len(dissimilar_results)} dissimilar molecules in '
                  f'{time_end_dissim - time_start_dissim:.3f}s')

            # Log dissimilar molecule details
            for j, (dissimilar_idx, distance) in enumerate(dissimilar_results):
                dissimilar_smiles = index_data_map[dissimilar_idx]['graph_repr']
                e.log(f'   {j + 1}. idx={dissimilar_idx}, dissimilarity={distance:.4f}, '
                      f'SMILES={dissimilar_smiles[:40]}...')

            # Store results in experiment storage
            e[f'dissimilarity/query_{i}/index'] = query_idx
            e[f'dissimilarity/query_{i}/smiles'] = query_smiles
            e[f'dissimilarity/query_{i}/dissimilar_molecules'] = [
                {
                    'index': idx,
                    'distance': dist,
                    'dissimilarity': dist,
                    'smiles': index_data_map[idx]['graph_repr']
                }
                for idx, dist in dissimilar_results
            ]

            # Add to query_result for summary (using -1 index since we already appended)
            all_results[-1]['dissimilar_molecules'] = [
                {
                    'index': idx,
                    'distance': dist,
                    'dissimilarity': dist,
                    'smiles': index_data_map[idx]['graph_repr']
                }
                for idx, dist in dissimilar_results
            ]

            # Visualize dissimilar molecules
            e.log(f' * creating dissimilarity visualization...')
            fig_dissim = e.apply_hook(
                'visualize_dissimilar_neighbors',
                query_idx=query_idx,
                dissimilar_results=dissimilar_results,
                index_data_map=index_data_map
            )

            # Generate safe filename
            filename_dissim = f'dissimilarity_query_{i}_{smiles_hash}.png'
            e.commit_fig(filename_dissim, fig_dissim)
            e.log(f' * saved dissimilarity visualization to {filename_dissim}')

            plt.close(fig_dissim)

    # == SAVE SUMMARY ==

    e.log('\n=== SAVING SUMMARY ===')

    # Store global parameters
    e['similarity/num_queries'] = len(query_indices)
    e['similarity/num_neighbors'] = e.NUM_NEIGHBORS
    e['similarity/dataset_size'] = len(index_data_map)

    # Create summary DataFrame with both similar and dissimilar molecules
    summary_rows = []
    for i, result in enumerate(all_results):
        query_idx = result['query_idx']
        query_smiles = result['query_smiles']

        # Add similar neighbors
        for neighbor_info in result['neighbors']:
            summary_rows.append({
                'query_id': i,
                'query_idx': query_idx,
                'query_smiles': query_smiles,
                'type': 'similar',
                'molecule_idx': neighbor_info['index'],
                'molecule_smiles': neighbor_info['smiles'],
                'distance': neighbor_info['distance'],
                'similarity': neighbor_info['similarity'],
            })

        # Add dissimilar molecules (if enabled and available)
        if e.FIND_DISSIMILAR and 'dissimilar_molecules' in result:
            for dissimilar_info in result['dissimilar_molecules']:
                summary_rows.append({
                    'query_id': i,
                    'query_idx': query_idx,
                    'query_smiles': query_smiles,
                    'type': 'dissimilar',
                    'molecule_idx': dissimilar_info['index'],
                    'molecule_smiles': dissimilar_info['smiles'],
                    'distance': dissimilar_info['distance'],
                    'similarity': 1 - dissimilar_info['distance'],  # For consistency
                })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(e.path, 'similarity_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    e.log(f'saved summary to {summary_path}')

    # Compute and log statistics
    e.log('\n=== SUMMARY STATISTICS ===')

    # Similar molecules stats
    similar_df = summary_df[summary_df['type'] == 'similar']
    avg_similarity = similar_df['similarity'].mean()
    avg_distance_sim = similar_df['distance'].mean()
    e.log(f'SIMILAR MOLECULES:')
    e.log(f' * average similarity: {avg_similarity:.4f}')
    e.log(f' * average distance: {avg_distance_sim:.4f}')

    e['similarity/avg_similarity'] = float(avg_similarity)
    e['similarity/avg_distance'] = float(avg_distance_sim)

    # Dissimilar molecules stats (if enabled)
    if e.FIND_DISSIMILAR:
        dissimilar_df = summary_df[summary_df['type'] == 'dissimilar']
        if len(dissimilar_df) > 0:
            avg_dissimilarity = dissimilar_df['distance'].mean()
            avg_similarity_dissim = dissimilar_df['similarity'].mean()
            e.log(f'\nDISSIMILAR MOLECULES:')
            e.log(f' * average dissimilarity (distance): {avg_dissimilarity:.4f}')
            e.log(f' * average similarity: {avg_similarity_dissim:.4f}')

            e['dissimilarity/avg_dissimilarity'] = float(avg_dissimilarity)
            e['dissimilarity/avg_similarity'] = float(avg_similarity_dissim)

            # Diversity metric: range between most similar and most dissimilar
            diversity_range = avg_dissimilarity - avg_distance_sim
            e.log(f'\nDIVERSITY METRICS:')
            e.log(f' * diversity range (dissimilarity - similarity): {diversity_range:.4f}')
            e['diversity/range'] = float(diversity_range)

    # == GED CORRELATION ANALYSIS ==

    if e.ENABLE_GED_ANALYSIS:
        e.log('\n=== GED CORRELATION ANALYSIS ===')
        e.log(f'analyzing correlation between graph edit distance and embedding similarity')

        # Select query samples for GED analysis (can be same or different from similarity search)
        e.log(f'\nselecting {e.GED_NUM_SAMPLES} query molecules for GED analysis...')

        available_indices = list(index_data_map.keys())
        if e.GED_NUM_SAMPLES > len(available_indices):
            e.log(f'WARNING: GED_NUM_SAMPLES ({e.GED_NUM_SAMPLES}) exceeds dataset size '
                  f'({len(available_indices)}). Using all samples.')
            ged_query_indices = available_indices
        else:
            random.seed(e.SEED)
            ged_query_indices = random.sample(available_indices, k=e.GED_NUM_SAMPLES)

        e.log(f'selected GED query indices: {ged_query_indices}')

        # Store results for all queries
        ged_results = []

        for i, query_idx in enumerate(ged_query_indices):
            e.log(f'\nprocessing GED query {i + 1}/{len(ged_query_indices)} (index={query_idx})...')

            query_smiles = index_data_map[query_idx]['graph_repr']
            query_features = index_data_map[query_idx]['graph_features']

            e.log(f' * query SMILES: {query_smiles}')

            # Generate N-hop neighborhood with GED tracking
            time_start = time.time()
            neighbors_with_ged = e.apply_hook(
                'calculate_n_hop_neighborhood',
                query_smiles=query_smiles
            )
            time_end = time.time()

            e.log(f' * generated {len(neighbors_with_ged)} neighbors in {time_end - time_start:.2f}s')

            if len(neighbors_with_ged) == 0:
                e.log(f' * WARNING: No neighbors generated, skipping this query')
                continue

            # Compute correlation between GED and embedding similarity
            time_start = time.time()
            r2, correlation, p_value, ged_array, similarity_array = e.apply_hook(
                'compute_ged_similarity_correlation',
                query_idx=query_idx,
                query_features=query_features,
                neighbors_with_ged=neighbors_with_ged,
                index_data_map=index_data_map
            )
            time_end = time.time()

            e.log(f' * computed correlation in {time_end - time_start:.2f}s')

            if len(ged_array) < 2:
                e.log(f' * WARNING: Insufficient valid neighbors for correlation, skipping')
                continue

            # Store results
            ged_result = {
                'query_idx': query_idx,
                'query_smiles': query_smiles,
                'r2': r2,
                'correlation': correlation,
                'p_value': p_value,
                'n_neighbors': len(ged_array),
                'ged_range': (int(ged_array.min()), int(ged_array.max())),
                'similarity_range': (float(similarity_array.min()), float(similarity_array.max()))
            }
            ged_results.append(ged_result)

            # Store in experiment storage
            e[f'ged_analysis/query_{i}/index'] = query_idx
            e[f'ged_analysis/query_{i}/smiles'] = query_smiles
            e[f'ged_analysis/query_{i}/r2'] = float(r2)
            e[f'ged_analysis/query_{i}/correlation'] = float(correlation)
            e[f'ged_analysis/query_{i}/p_value'] = float(p_value)
            e[f'ged_analysis/query_{i}/n_neighbors'] = len(ged_array)

            # Create regression visualization
            e.log(f' * creating regression plot...')
            fig = e.apply_hook(
                'visualize_ged_regression',
                query_smiles=query_smiles,
                ged_array=ged_array,
                similarity_array=similarity_array,
                r2=r2,
                correlation=correlation,
                p_value=p_value
            )

            # Generate safe filename
            smiles_hash = hashlib.md5(query_smiles.encode()).hexdigest()[:8]
            filename = f'ged_regression_query_{i}_{smiles_hash}.png'
            e.commit_fig(filename, fig)
            e.log(f' * saved regression plot to {filename}')

            plt.close(fig)

        # == AGGREGATE GED STATISTICS ==

        if len(ged_results) > 0:
            e.log('\n=== AGGREGATE GED CORRELATION STATISTICS ===')

            # Calculate average metrics
            avg_r2 = np.mean([r['r2'] for r in ged_results])
            avg_correlation = np.mean([r['correlation'] for r in ged_results])
            std_r2 = np.std([r['r2'] for r in ged_results])
            std_correlation = np.std([r['correlation'] for r in ged_results])

            e.log(f'average R² across all queries: {avg_r2:.4f} (±{std_r2:.4f})')
            e.log(f'average correlation across all queries: {avg_correlation:.4f} (±{std_correlation:.4f})')

            # Store aggregate metrics
            e['ged_analysis/avg_r2'] = float(avg_r2)
            e['ged_analysis/std_r2'] = float(std_r2)
            e['ged_analysis/avg_correlation'] = float(avg_correlation)
            e['ged_analysis/std_correlation'] = float(std_correlation)
            e['ged_analysis/num_queries'] = len(ged_results)

            # Save summary CSV
            summary_rows = []
            for i, result in enumerate(ged_results):
                summary_rows.append({
                    'query_id': i,
                    'query_idx': result['query_idx'],
                    'query_smiles': result['query_smiles'],
                    'r2': result['r2'],
                    'correlation': result['correlation'],
                    'p_value': result['p_value'],
                    'n_neighbors': result['n_neighbors'],
                    'ged_min': result['ged_range'][0],
                    'ged_max': result['ged_range'][1],
                    'similarity_min': result['similarity_range'][0],
                    'similarity_max': result['similarity_range'][1],
                })

            # Add aggregate row
            summary_rows.append({
                'query_id': 'AGGREGATE',
                'query_idx': '',
                'query_smiles': '',
                'r2': avg_r2,
                'correlation': avg_correlation,
                'p_value': '',
                'n_neighbors': '',
                'ged_min': '',
                'ged_max': '',
                'similarity_min': '',
                'similarity_max': '',
            })

            ged_summary_df = pd.DataFrame(summary_rows)
            ged_summary_path = os.path.join(e.path, 'ged_correlation_summary.csv')
            ged_summary_df.to_csv(ged_summary_path, index=False)
            e.log(f'saved GED correlation summary to {ged_summary_path}')

            # Log detailed results
            e.log('\nPer-query GED correlation results:')
            for i, result in enumerate(ged_results):
                e.log(f'  Query {i}: R²={result["r2"]:.4f}, r={result["correlation"]:.4f}, '
                      f'n={result["n_neighbors"]}, SMILES={result["query_smiles"][:40]}...')

        else:
            e.log('\nWARNING: No valid GED results were generated')

    e.log('\nexperiment complete!')


experiment.run_if_main()
