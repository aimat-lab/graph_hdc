"""
Molecule Representation Investigation Experiment

This experiment module investigates the properties of molecular vector representations
through dimensionality reduction and visualization techniques. It provides a flexible
framework for analyzing different molecular encoding methods (HDC, fingerprints, GNN
embeddings, etc.) via hooks.

Key Features:
- 2D PCA and UMAP visualizations with multiple property-based color codings
- Support for multiple color properties (clogp, target, custom) via hooks
- PCA variance analysis across multiple dimensions
- Extensible through PyComex hooks for different representations and properties

Usage:
    This module serves as a base for investigating molecular representations.
    The actual encoding method should be specified by overriding the
    'process_dataset' hook in an extending experiment or via a configuration file.

    Multiple color properties can be visualized by specifying COLOR_PROPERTIES
    parameter. Each color property generates separate PCA and UMAP plots.

Example:
    Create a config file molecule_representation__hdc__aqsoldb.yml:

    .. code-block:: yaml

        extend: molecule_representation__hdc.py
        parameters:
            DATASET_NAME: "aqsoldb"
            PCA_DIMENSIONS: [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
            COLOR_PROPERTIES: ['clogp', 'target']
            TARGET_INDEX: 0

Output Artifacts:
    For each color property 'prop_name':
    - pca_2d_{prop_name}.png: 2D PCA scatter plot with color coding
    - umap_2d_{prop_name}.png: 2D UMAP scatter plot with color coding

    Variance analysis:
    - pca_variance_sweep.png: Explained variance vs dimensions
    - pca_variance_cumulative.png: Cumulative variance plots
    - variance_statistics.csv: Numerical variance data
"""
import os
import time
import random
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem.Crippen import MolLogP
from sklearn.decomposition import PCA
import umap
from rich.pretty import pprint

from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace
from chem_mat_data._typing import GraphDict
from chem_mat_data.main import load_graph_dataset

# == DATASET PARAMETERS ==

# :param DATASET_NAME:
#       The name of the dataset to be used for the experiment. This name is used to
#       download the dataset from the ChemMatData file share.
DATASET_NAME: str = 'aqsoldb'

# :param DATASET_NAME_ID:
#       The name of the dataset to be used for identification purposes. This name will
#       NOT be used for downloading the dataset but only for identification in outputs.
#       In most cases these will be the same, but in cases where one dataset is used
#       as the basis for some deterministic calculation of the target values, this name
#       should identify it as such.
DATASET_NAME_ID: str = DATASET_NAME

# :param NUM_DATA:
#       The number of samples to be used for the experiment. This parameter can be either
#       an integer or a float between 0 and 1. In case of an integer we use it as the
#       number of samples to be used, in case of a float we use it as the fraction of
#       the dataset to be used. This parameter is used to limit the size of the dataset
#       for the experiment. If None, the entire dataset is used.
NUM_DATA: Union[int, float, None] = None

# :param SEED:
#       The random seed to be used for the experiment. This ensures reproducibility of
#       the subsampling, dimensionality reduction, and any other stochastic operations.
SEED: int = 1

# :param TARGET_INDEX:
#       The index of the target property in graph_labels to use for the 'target' color
#       property. If None, the first target (index 0) will be used. This parameter is
#       only relevant when 'target' is included in COLOR_PROPERTIES.
TARGET_INDEX: Union[int, None] = 0

# :param COLOR_PROPERTIES:
#       A list of color property names to use for visualization. For each property in
#       this list, separate PCA and UMAP plots will be generated with color-coding based
#       on that property. The experiment will look for hooks named 'get_color_property__{name}'
#       for each property. Built-in properties include:
#       - 'clogp': Crippen's logP (lipophilicity)
#       - 'target': Target property from graph_labels[TARGET_INDEX]
COLOR_PROPERTIES: List[str] = ['clogp', 'target']

# == DIMENSIONALITY REDUCTION PARAMETERS ==

# :param PCA_DIMENSIONS:
#       A list of dimensions to sweep over for the PCA variance analysis. For each
#       dimension in this list, we will perform PCA reduction and calculate the
#       explained variance ratio. This helps understand how much information is
#       retained at different dimensionalities.
PCA_DIMENSIONS: List[int] = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

# :param UMAP_N_NEIGHBORS:
#       The size of local neighborhood (in terms of number of neighboring sample points)
#       used for manifold approximation. Larger values result in more global views of
#       the manifold, while smaller values result in more local data being preserved.
UMAP_N_NEIGHBORS: int = 150

# :param UMAP_MIN_DIST:
#       The minimum distance between points in the low-dimensional representation.
#       Smaller values will result in a more clustered/clumped embedding where nearby
#       points are packed more tightly.
UMAP_MIN_DIST: float = 0.0

# :param UMAP_METRIC:
#       The metric to use for computing distances in high dimensional space. Common
#       choices are 'euclidean', 'manhattan', 'cosine', 'correlation'.
UMAP_METRIC: str = 'cosine'

# == VISUALIZATION PARAMETERS ==

# :param FIGURE_SIZE:
#       The default figure size (width, height) in inches for all plots generated by
#       this experiment.
FIGURE_SIZE: Tuple[int, int] = (10, 8)

# :param COLOR_PERCENTILES:
#       The percentiles to use for clipping color values in the scatter plots. This
#       helps deal with outliers that might skew the color scale. Values are clipped
#       to (lower_percentile, upper_percentile). Default (2, 98) clips the extreme
#       2% on each end.
COLOR_PERCENTILES: Tuple[float, float] = (2, 98)

# :param SCATTER_POINT_SIZE:
#       The size of scatter plot points in the 2D visualizations.
SCATTER_POINT_SIZE: int = 10

# :param SCATTER_ALPHA:
#       The transparency of scatter plot points (0 = fully transparent, 1 = fully opaque).
SCATTER_ALPHA: float = 0.5

# :param COLORMAP:
#       The matplotlib colormap to use for color-coding the scatter plots.
COLORMAP: str = 'viridis'

# == OUTPUT PARAMETERS ==

# :param SAVE_EMBEDDINGS:
#       Whether to save the computed embeddings to disk as a compressed NPZ file.
#       This allows for later analysis without recomputing the embeddings.
SAVE_EMBEDDINGS: bool = False

# == EXPERIMENT PARAMETERS ==

# :param NOTE:
#       A note that can be used to describe the experiment. This note will be stored
#       as part of the experiment metadata and can later serve for identification.
NOTE: str = ''

__DEBUG__: bool = True
__NOTIFY__: bool = False

experiment = Experiment(
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals()
)


# == HOOKS ==

@experiment.hook('load_dataset', replace=False, default=True)
def load_dataset(e: Experiment) -> dict[int, GraphDict]:
    """
    Load the molecular dataset from ChemMatData.

    This hook downloads the dataset from the ChemMatData file share and returns
    a dictionary mapping integer indices to graph dict representations of molecules.

    :param e: The experiment instance providing access to parameters and logging.

    :return: A dictionary mapping indices to graph dictionaries containing molecular
        structure and property information.
    """
    e.log(f'loading dataset "{e.DATASET_NAME}"...')

    # Download and load the dataset from ChemMatData
    graphs: List[GraphDict] = load_graph_dataset(
        e.DATASET_NAME,
        folder_path='/tmp'
    )

    # Create index mapping
    index_data_map = dict(enumerate(graphs))
    e.log(f'loaded {len(index_data_map)} molecules from dataset')

    # Optional subsampling
    if e.NUM_DATA is not None:
        if isinstance(e.NUM_DATA, int):
            num_data = e.NUM_DATA
        elif isinstance(e.NUM_DATA, float):
            num_data = int(e.NUM_DATA * len(index_data_map))

        # Subsample the dataset to the specified number of samples
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
    Filter the dataset to remove invalid SMILES and unconnected molecular graphs.

    This hook removes molecules that:
    - Have invalid SMILES strings (cannot be parsed by RDKit)
    - Have fewer than 2 atoms
    - Have no bonds
    - Are disconnected (contain multiple fragments)

    :param e: The experiment instance providing access to parameters and logging.
    :param index_data_map: Dictionary mapping indices to graph dictionaries. This
        dictionary is modified in-place to remove invalid entries.

    :return: None. Modifies index_data_map in-place by removing invalid entries.
    """
    e.log(f'filtering dataset to remove invalid SMILES and unconnected graphs...')
    e.log(f'starting with {len(index_data_map)} samples...')

    indices = list(index_data_map.keys())
    for index in indices:
        graph = index_data_map[index]
        smiles = graph['graph_repr']

        # Try to parse the SMILES string
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            del index_data_map[index]
            continue

        # Check for minimum number of atoms
        if len(mol.GetAtoms()) < 2:
            del index_data_map[index]
            continue

        # Check for at least one bond
        if len(mol.GetBonds()) < 1:
            del index_data_map[index]
            continue

        # Check for disconnected graphs (multiple fragments)
        if '.' in smiles:
            del index_data_map[index]
            continue

    e.log(f'finished filtering dataset with {len(index_data_map)} samples remaining')


@experiment.hook('process_dataset', replace=False, default=True)
def process_dataset(e: Experiment,
                    index_data_map: dict[int, GraphDict]
                    ) -> None:
    """
    Process the dataset into molecular vector representations.

    **IMPORTANT:** This hook must be overridden in extending experiments to provide
    the actual molecular encoding method (HDC, fingerprints, GNN embeddings, etc.).

    The hook should add a 'graph_features' key to each graph dictionary containing
    a numpy array of the molecular representation.

    Example Implementation (Morgan Fingerprints):

        .. code-block:: python

            from rdkit.Chem import rdFingerprintGenerator

            gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
            for index, graph in index_data_map.items():
                mol = Chem.MolFromSmiles(graph['graph_repr'])
                fp = gen.GetFingerprint(mol)
                graph['graph_features'] = np.array(fp).astype(float)

    :param e: The experiment instance providing access to parameters and logging.
    :param index_data_map: Dictionary mapping indices to graph dictionaries. This
        dictionary should be modified in-place to add 'graph_features' to each entry.

    :return: None. Modifies index_data_map in-place by adding 'graph_features' key.

    :raises NotImplementedError: This default implementation raises an error to ensure
        users override it with their specific encoding method.
    """
    raise NotImplementedError(
        "The 'process_dataset' hook must be overridden to provide a molecular "
        "encoding method. Please extend this experiment and implement this hook "
        "with your chosen representation method (HDC, fingerprints, GNN, etc.).\n\n"
        "Example: Create a file 'molecule_representation__fp.py' that extends this "
        "experiment and implements the 'process_dataset' hook using Morgan fingerprints."
    )


@experiment.hook('get_color_property__clogp', replace=False, default=True)
def get_color_property_clogp(e: Experiment,
                             index: int,
                             graph: GraphDict,
                             ) -> float:
    """
    Compute Crippen's logP for color-coding in visualizations.

    This hook calculates the Crippen logP (calculated partition coefficient) for
    each molecule, which is a measure of lipophilicity. This is a commonly used
    descriptor in drug discovery and chemistry.

    :param e: The experiment instance providing access to parameters and logging.
    :param index: The integer index of the molecule in the dataset.
    :param graph: The graph dictionary containing molecular structure and properties.

    :return: A float value representing the Crippen logP.
    """
    smiles = graph['graph_repr']
    mol = Chem.MolFromSmiles(smiles)
    return MolLogP(mol)


@experiment.hook('get_color_property__target', replace=False, default=True)
def get_color_property_target(e: Experiment,
                              index: int,
                              graph: GraphDict,
                              ) -> float:
    """
    Extract target property from graph_labels for color-coding in visualizations.

    This hook extracts a target property value from the graph_labels array. The
    specific index to extract is controlled by the TARGET_INDEX parameter. If
    TARGET_INDEX is None, the first target (index 0) is used.

    This is useful for visualizing how the molecular representations cluster with
    respect to the prediction target, which can provide insights into whether the
    representation captures the relevant chemical features.

    :param e: The experiment instance providing access to parameters and logging.
    :param index: The integer index of the molecule in the dataset.
    :param graph: The graph dictionary containing molecular structure and properties.

    :return: A float value representing the target property.

    :raises KeyError: If the graph does not contain 'graph_labels'.
    :raises IndexError: If TARGET_INDEX is out of bounds for graph_labels.
    """
    if 'graph_labels' not in graph:
        raise KeyError(
            f"Graph at index {index} does not contain 'graph_labels'. "
            f"Make sure the dataset includes target labels."
        )

    labels = graph['graph_labels']

    if e.TARGET_INDEX is None:
        # Use first target by default
        if len(labels) == 0:
            raise ValueError(f"Graph at index {index} has empty graph_labels array.")
        return float(labels[0])
    else:
        # Use specified target index
        if e.TARGET_INDEX >= len(labels):
            raise IndexError(
                f"TARGET_INDEX={e.TARGET_INDEX} is out of bounds for graph_labels "
                f"with length {len(labels)} at graph index {index}."
            )
        return float(labels[e.TARGET_INDEX])


@experiment.hook('after_visualization', replace=False, default=True)
def after_visualization(e: Experiment,
                       index_data_map: dict[int, GraphDict],
                       embeddings: np.ndarray,
                       reduced_pca: np.ndarray,
                       reduced_umap: np.ndarray,
                       **kwargs
                       ) -> None:
    """
    Optional hook for additional analysis after the main visualizations.

    This hook is called after all the main visualizations have been generated for
    all color properties. It provides an opportunity to perform additional custom
    analysis or create supplementary plots.

    :param e: The experiment instance providing access to parameters and logging.
    :param index_data_map: Dictionary mapping indices to graph dictionaries.
    :param embeddings: The numpy array of molecular embeddings (shape: [n_molecules, dim]).
    :param reduced_pca: The 2D PCA reduction of embeddings (shape: [n_molecules, 2]).
    :param reduced_umap: The 2D UMAP reduction of embeddings (shape: [n_molecules, 2]).
    :param kwargs: Additional keyword arguments for future extensibility.

    :return: None.
    """
    # Default implementation does nothing - can be overridden for custom analysis
    pass


# == MAIN EXPERIMENT ==

@experiment
def main(e: Experiment):
    """
    Main experiment function for investigating molecular representation properties.

    This function orchestrates the entire experiment workflow:
    1. Load and filter the molecular dataset
    2. Process molecules into vector representations (via hook)
    3. Compute 2D PCA and UMAP reductions (once)
    4. For each color property in COLOR_PROPERTIES:
       a. Compute color property values for all molecules (via hook)
       b. Generate 2D PCA visualization with color coding
       c. Generate 2D UMAP visualization with color coding
    5. Perform PCA variance analysis across multiple dimensions
    6. Generate summary plots and statistics
    7. Run optional after_visualization hook for custom analysis

    :param e: The experiment instance providing access to parameters, hooks, and logging.

    :return: None. Results are saved as artifacts in the experiment directory.
    """
    e.log('starting molecule representation investigation experiment...')
    e.log_parameters()

    # == DATASET LOADING ==

    e.log('\n=== LOADING DATASET ===')
    index_data_map: dict[int, GraphDict] = e.apply_hook('load_dataset')

    # == DATASET FILTERING ==

    e.log('\n=== FILTERING DATASET ===')
    e.apply_hook('filter_dataset', index_data_map=index_data_map)

    # == DATASET PROCESSING ==

    e.log('\n=== PROCESSING DATASET ===')
    e.log('converting molecules to vector representations...')
    time_start = time.time()
    e.apply_hook('process_dataset', index_data_map=index_data_map)
    time_end = time.time()
    duration = time_end - time_start
    e.log(f'processed dataset after {duration:.2f} seconds')

    # Collect embeddings and verify they exist
    embeddings_list = []
    valid_indices = []
    for index in sorted(index_data_map.keys()):
        graph = index_data_map[index]
        if 'graph_features' not in graph:
            e.log(f'WARNING: graph {index} missing graph_features, skipping')
            continue
        embeddings_list.append(graph['graph_features'])
        valid_indices.append(index)

    embeddings = np.array(embeddings_list)
    e.log(f'collected {len(embeddings)} embeddings with shape {embeddings.shape}')

    # Store basic statistics
    e['dataset/num_molecules'] = len(embeddings)
    e['dataset/embedding_dim'] = embeddings.shape[1]

    # Optionally save embeddings
    if e.SAVE_EMBEDDINGS:
        embeddings_path = os.path.join(e.path, 'embeddings.npz')
        smiles_list = [index_data_map[idx]['graph_repr'] for idx in valid_indices]
        np.savez_compressed(
            embeddings_path,
            embeddings=embeddings,
            indices=valid_indices,
            smiles=smiles_list
        )
        e.log(f'saved embeddings to {embeddings_path}')

    # == DIMENSIONALITY REDUCTION ==
    # Compute the 2D reductions once, then use them for all color properties

    e.log('\n=== COMPUTING 2D DIMENSIONALITY REDUCTIONS ===')

    # PCA reduction
    e.log('performing 2D PCA...')
    pca_2d = PCA(n_components=2, random_state=e.SEED)
    reduced_pca = pca_2d.fit_transform(embeddings)

    explained_var_2d = pca_2d.explained_variance_ratio_
    e.log(f'PCA explained variance: {explained_var_2d[0]:.4f}, {explained_var_2d[1]:.4f}')
    e.log(f'PCA total explained variance: {explained_var_2d.sum():.4f}')

    e['pca_2d/explained_variance_1'] = float(explained_var_2d[0])
    e['pca_2d/explained_variance_2'] = float(explained_var_2d[1])
    e['pca_2d/total_explained_variance'] = float(explained_var_2d.sum())

    # UMAP reduction
    e.log(f'performing 2D UMAP with n_neighbors={e.UMAP_N_NEIGHBORS}, '
          f'min_dist={e.UMAP_MIN_DIST}, metric={e.UMAP_METRIC}...')

    reducer = umap.UMAP(
        n_components=2,
        random_state=e.SEED,
        n_neighbors=e.UMAP_N_NEIGHBORS,
        min_dist=e.UMAP_MIN_DIST,
        metric=e.UMAP_METRIC,
    )
    reduced_umap = reducer.fit_transform(embeddings)
    e.log('UMAP reduction complete')

    # == VISUALIZATIONS WITH MULTIPLE COLOR PROPERTIES ==
    # For each color property, compute the property values and create visualizations

    for color_prop_name in e.COLOR_PROPERTIES:
        e.log(f'\n=== COLOR PROPERTY: {color_prop_name} ===')

        # Compute color property values for all molecules
        e.log(f'computing color property "{color_prop_name}"...')
        color_properties = []
        for index in valid_indices:
            graph = index_data_map[index]
            try:
                prop_value = e.apply_hook(
                    f'get_color_property__{color_prop_name}',
                    index=index,
                    graph=graph
                )
                color_properties.append(prop_value)
            except Exception as ex:
                e.log(f'ERROR computing color property for molecule {index}: {ex}')
                raise

        color_properties = np.array(color_properties)
        e.log(f'computed {len(color_properties)} color property values')
        e.log(f'color property range: [{color_properties.min():.3f}, {color_properties.max():.3f}]')

        # Clip color values to percentiles for better visualization
        vmin, vmax = np.percentile(color_properties, e.COLOR_PERCENTILES)
        color_properties_clipped = np.clip(color_properties, vmin, vmax)
        e.log(f'clipped to percentiles {e.COLOR_PERCENTILES}: [{vmin:.3f}, {vmax:.3f}]')

        # Store statistics for this color property
        e[f'color_properties/{color_prop_name}/min'] = float(color_properties.min())
        e[f'color_properties/{color_prop_name}/max'] = float(color_properties.max())
        e[f'color_properties/{color_prop_name}/mean'] = float(color_properties.mean())
        e[f'color_properties/{color_prop_name}/std'] = float(color_properties.std())

        # == 2D PCA VISUALIZATION ==

        e.log(f'creating 2D PCA plot with {color_prop_name} coloring...')

        fig, ax = plt.subplots(figsize=e.FIGURE_SIZE)
        scatter = ax.scatter(
            reduced_pca[:, 0],
            reduced_pca[:, 1],
            c=color_properties_clipped,
            cmap=e.COLORMAP,
            alpha=e.SCATTER_ALPHA,
            s=e.SCATTER_POINT_SIZE,
            edgecolors='none'
        )

        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(f'{color_prop_name}', fontsize=12)

        ax.set_xlabel(f'PC1 ({explained_var_2d[0]:.2%} variance)', fontsize=12)
        ax.set_ylabel(f'PC2 ({explained_var_2d[1]:.2%} variance)', fontsize=12)
        ax.set_title(
            f'2D PCA of Molecular Representations\n'
            f'Dataset: {e.DATASET_NAME_ID} ({len(embeddings)} molecules, '
            f'{embeddings.shape[1]}D embeddings)\n'
            f'Color: {color_prop_name}',
            fontsize=14
        )
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig_name = f'pca_2d_{color_prop_name}.png'
        e.commit_fig(fig_name, fig)
        e.log(f'saved 2D PCA plot as {fig_name}')

        # == 2D UMAP VISUALIZATION ==

        e.log(f'creating 2D UMAP plot with {color_prop_name} coloring...')

        fig, ax = plt.subplots(figsize=e.FIGURE_SIZE)
        scatter = ax.scatter(
            reduced_umap[:, 0],
            reduced_umap[:, 1],
            c=color_properties_clipped,
            cmap=e.COLORMAP,
            alpha=e.SCATTER_ALPHA,
            s=e.SCATTER_POINT_SIZE,
            edgecolors='none'
        )

        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(f'{color_prop_name}', fontsize=12)

        ax.set_xlabel('UMAP Dimension 1', fontsize=12)
        ax.set_ylabel('UMAP Dimension 2', fontsize=12)
        ax.set_title(
            f'2D UMAP of Molecular Representations\n'
            f'Dataset: {e.DATASET_NAME_ID} ({len(embeddings)} molecules, '
            f'{embeddings.shape[1]}D embeddings)\n'
            f'Color: {color_prop_name}',
            fontsize=14
        )
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig_name = f'umap_2d_{color_prop_name}.png'
        e.commit_fig(fig_name, fig)
        e.log(f'saved 2D UMAP plot as {fig_name}')

    # == PCA VARIANCE SWEEP ==

    e.log('\n=== PCA VARIANCE ANALYSIS ===')
    e.log(f'sweeping over dimensions: {e.PCA_DIMENSIONS}')

    # Filter dimensions that are valid for this embedding size
    max_dim = min(embeddings.shape[0], embeddings.shape[1])
    valid_dimensions = [d for d in e.PCA_DIMENSIONS if d <= max_dim]
    e.log(f'valid dimensions (≤ {max_dim}): {valid_dimensions}')

    variance_results = []

    for n_components in valid_dimensions:
        e.log(f'computing PCA for {n_components} components...')
        pca = PCA(n_components=n_components, random_state=e.SEED)
        pca.fit(embeddings)

        explained_var = pca.explained_variance_ratio_
        total_var = explained_var.sum()

        variance_results.append({
            'n_components': n_components,
            'total_explained_variance': total_var,
            'mean_explained_variance_per_component': explained_var.mean(),
        })

        e.log(f'  {n_components} components: {total_var:.4f} total variance')
        e[f'pca_sweep/{n_components}/total_variance'] = float(total_var)
        e[f'pca_sweep/{n_components}/mean_variance'] = float(explained_var.mean())

    # Create DataFrame for easy plotting
    df_variance = pd.DataFrame(variance_results)

    # Save variance statistics
    csv_path = os.path.join(e.path, 'variance_statistics.csv')
    df_variance.to_csv(csv_path, index=False)
    e.log(f'saved variance statistics to {csv_path}')

    # == VARIANCE PLOTS ==

    # Plot 1: Total explained variance vs dimensions
    fig, ax = plt.subplots(figsize=e.FIGURE_SIZE)
    ax.plot(
        df_variance['n_components'],
        df_variance['total_explained_variance'],
        marker='o',
        linewidth=2,
        markersize=8
    )
    ax.set_xlabel('Number of PCA Components', fontsize=12)
    ax.set_ylabel('Total Explained Variance Ratio', fontsize=12)
    ax.set_title(
        f'PCA Explained Variance vs Dimensionality\n'
        f'Dataset: {e.DATASET_NAME_ID} ({embeddings.shape[1]}D embeddings)',
        fontsize=14
    )
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])

    # Add horizontal lines for reference
    ax.axhline(y=0.9, color='r', linestyle='--', alpha=0.5, label='90% variance')
    ax.axhline(y=0.95, color='orange', linestyle='--', alpha=0.5, label='95% variance')
    ax.axhline(y=0.99, color='g', linestyle='--', alpha=0.5, label='99% variance')
    ax.legend()

    plt.tight_layout()
    e.commit_fig('pca_variance_sweep.png', fig)
    e.log('saved PCA variance sweep plot')

    # Plot 2: Cumulative variance with both linear and log scale
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Linear scale
    ax1.plot(
        df_variance['n_components'],
        df_variance['total_explained_variance'],
        marker='o',
        linewidth=2,
        markersize=8,
        color='steelblue'
    )
    ax1.set_xlabel('Number of PCA Components', fontsize=12)
    ax1.set_ylabel('Cumulative Explained Variance Ratio', fontsize=12)
    ax1.set_title('Linear Scale', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.05])
    ax1.axhline(y=0.9, color='r', linestyle='--', alpha=0.5)
    ax1.axhline(y=0.95, color='orange', linestyle='--', alpha=0.5)
    ax1.axhline(y=0.99, color='g', linestyle='--', alpha=0.5)

    # Log scale
    ax2.plot(
        df_variance['n_components'],
        df_variance['total_explained_variance'],
        marker='o',
        linewidth=2,
        markersize=8,
        color='steelblue'
    )
    ax2.set_xlabel('Number of PCA Components', fontsize=12)
    ax2.set_ylabel('Cumulative Explained Variance Ratio', fontsize=12)
    ax2.set_title('Log Scale', fontsize=12)
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.05])
    ax2.axhline(y=0.9, color='r', linestyle='--', alpha=0.5, label='90%')
    ax2.axhline(y=0.95, color='orange', linestyle='--', alpha=0.5, label='95%')
    ax2.axhline(y=0.99, color='g', linestyle='--', alpha=0.5, label='99%')
    ax2.legend()

    fig.suptitle(
        f'PCA Cumulative Explained Variance\n'
        f'Dataset: {e.DATASET_NAME_ID} ({embeddings.shape[1]}D embeddings)',
        fontsize=14
    )

    plt.tight_layout()
    e.commit_fig('pca_variance_cumulative.png', fig)
    e.log('saved cumulative variance plots')

    # == SUMMARY STATISTICS ==

    e.log('\n=== SUMMARY ===')
    e.log(f'Total molecules analyzed: {len(embeddings)}')
    e.log(f'Embedding dimensionality: {embeddings.shape[1]}')
    e.log(f'2D PCA explained variance: {explained_var_2d.sum():.4f}')

    # Find dimensions needed for different variance thresholds
    for threshold in [0.90, 0.95, 0.99]:
        matching = df_variance[df_variance['total_explained_variance'] >= threshold]
        if len(matching) > 0:
            min_components = matching.iloc[0]['n_components']
            actual_var = matching.iloc[0]['total_explained_variance']
            e.log(f'Dimensions for {threshold:.0%} variance: {min_components} (actual: {actual_var:.4f})')
            e[f'summary/dims_for_{int(threshold*100)}pct_variance'] = int(min_components)

    # == OPTIONAL POST-PROCESSING ==

    e.log('\n=== RUNNING OPTIONAL HOOKS ===')
    e.apply_hook(
        'after_visualization',
        index_data_map=index_data_map,
        embeddings=embeddings,
        reduced_pca=reduced_pca,
        reduced_umap=reduced_umap,
    )

    e.log('\nexperiment complete!')


experiment.run_if_main()
