"""
Random Baseline for Bayesian Optimization Molecular Search

This experiment extends the base optimize_molecule_bo.py experiment to use
random molecular representations as a baseline for comparison. Each molecule
is assigned a random vector of configurable dimensionality.

This baseline is crucial for understanding whether structured representations
(HDC, fingerprints, etc.) provide any advantage over random features for
Bayesian Optimization. If a method performs similarly to random features,
it suggests the GP is not effectively learning from the molecular structure.

Key Insights from Random Baseline:
    - If random performs well: The dataset may be easy or GP is overfitting
    - If random performs poorly: Structured representations are capturing
      meaningful molecular similarities that help the GP generalize
    - Difference in AUC: Quantifies value of structured representations

Design Rationale:
    Random representations are independently sampled for each molecule,
    so there's no correlation between similar molecules. This breaks the
    GP's ability to generalize based on molecular structure, providing
    a true "uninformed search" baseline.

Usage:
    Run directly or create configuration YAML files:

    .. code-block:: yaml

        extend: optimize_molecule_bo__random.py
        parameters:
            DATASET_NAME: "aqsoldb"
            TARGET_INDEX: 0
            TARGET_VALUE: 5.0
            RANDOM_DIM: 2048  # Match other methods
            NUM_INITIAL_SAMPLES: 10
            NUM_BO_ROUNDS: 20
            ACQUISITION_FUNCTION: "EI"

Example:
    .. code-block:: bash

        # Run with debug mode
        python optimize_molecule_bo__random.py

        # Run with configuration
        python -m pycomex run optimize_molecule_bo__random__clogp.yml
"""
import numpy as np

from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace
from chem_mat_data._typing import GraphDict

# == RANDOM REPRESENTATION PARAMETERS ==

# :param RANDOM_DIM:
#       The dimensionality of the random feature vectors. Should match the
#       dimensionality of the methods you're comparing against (e.g., 2048 for
#       fingerprints, HDC embeddings). Higher dimensions may give GP more
#       capacity but won't add meaningful structure.
RANDOM_DIM: int = 2048

# :param RANDOM_SEED:
#       Seed for random representation generation. Set to None to use the main
#       experiment seed. Using a fixed seed ensures reproducibility.
RANDOM_SEED: int = None

# :param RANDOM_DISTRIBUTION:
#       Distribution to sample random features from:
#       - 'normal': Gaussian N(0,1) - standard choice
#       - 'uniform': Uniform [-1, 1]
#       - 'binary': Binary {0, 1} - like fingerprints
RANDOM_DISTRIBUTION: str = 'normal'

# == EXPERIMENT ==

experiment = Experiment.extend(
    'optimize_molecule_bo.py',
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals()
)


@experiment.hook('process_dataset', replace=True, default=False)
def process_dataset(e: Experiment,
                    index_data_map: dict[int, GraphDict]
                    ) -> None:
    """
    Process molecules into random feature representations.

    This hook assigns each molecule a random vector of dimensionality RANDOM_DIM.
    The vectors are independently sampled, so there's no correlation between
    similar molecules - this is the key property that makes this a true baseline.

    The random features are sampled from the specified distribution (normal,
    uniform, or binary) and stored in the 'graph_features' field.

    Random Baseline Interpretation:
        - If GP performs well with random features, the search space may be
          easy to optimize or the GP is relying too much on the acquisition
          function rather than learned structure
        - If GP performs poorly (similar to random search), it validates that
          the Bayesian Optimization setup is working correctly
        - Comparing structured methods to random baseline quantifies the value
          of molecular representation learning

    :param e: The experiment instance providing access to parameters and logging.
    :param index_data_map: Dictionary mapping indices to graph dictionaries. This
        dictionary is modified in-place to add 'graph_features' to each entry.

    :return: None. Modifies index_data_map in-place by adding 'graph_features' key.
    """
    e.log(f'Random baseline process_dataset called with {len(index_data_map)} molecules')
    e.log(f' * RANDOM_DIM: {e.RANDOM_DIM}')
    e.log(f' * RANDOM_DISTRIBUTION: {e.RANDOM_DISTRIBUTION}')

    # Set random seed
    random_seed = e.RANDOM_SEED if e.RANDOM_SEED is not None else e.SEED
    e.log(f' * RANDOM_SEED: {random_seed}')
    np.random.seed(random_seed)

    # Generate random features for each molecule
    e.log('generating random feature vectors...')

    for c, (index, graph) in enumerate(index_data_map.items()):
        # Sample random vector based on distribution
        if e.RANDOM_DISTRIBUTION == 'normal':
            # Gaussian N(0, 1)
            random_vec = np.random.randn(e.RANDOM_DIM)
        elif e.RANDOM_DISTRIBUTION == 'uniform':
            # Uniform [-1, 1]
            random_vec = np.random.uniform(-1, 1, size=e.RANDOM_DIM)
        elif e.RANDOM_DISTRIBUTION == 'binary':
            # Binary {0, 1} - similar to binary fingerprints
            random_vec = np.random.randint(0, 2, size=e.RANDOM_DIM).astype(float)
        else:
            raise ValueError(
                f"Unknown random distribution: {e.RANDOM_DISTRIBUTION}. "
                f"Supported: 'normal', 'uniform', 'binary'"
            )

        graph['graph_features'] = random_vec

        if c % 1000 == 0 and c > 0:
            e.log(f' * generated {c} random vectors')

    e.log(f'completed random feature generation for {len(index_data_map)} molecules')
    e.log('NOTE: These are truly random features with no molecular structure information')


experiment.run_if_main()
