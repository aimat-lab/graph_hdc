"""
Atom-level explanation experiment for the Ames mutagenicity dataset with
the HDC encoder.

Extends the base explanation experiment with Ames-specific dataset parameters.
"""
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace
from pycomex import INHERIT

# == DATASET PARAMETERS ==

# :param DATASET_NAME:
#       The name of the dataset to be used for the experiment.
DATASET_NAME: str = 'ames'
# :param DATASET_TYPE:
#       The type of the dataset, either 'classification' or 'regression'.
DATASET_TYPE: str = 'classification'
# :param NUM_TEST:
#       The number of test samples to be used for the evaluation of the models.
NUM_TEST: int = INHERIT

# == EMBEDDING PARAMETERS ==

# :param EMBEDDING_SIZE:
#       The size of the graph embedding vectors (hypervector dimensionality).
EMBEDDING_SIZE: int = INHERIT
# :param NUM_LAYERS:
#       The number of message-passing layers in the hypernetwork.
NUM_LAYERS: int = INHERIT

# == EXPLANATION PARAMETERS ==

# :param EXPLAIN_MODEL_NAME:
#       The name of the trained model to explain. Must match one of the
#       entries in the MODELS list.
EXPLAIN_MODEL_NAME: str = INHERIT
# :param NUM_EXPLAIN_MOLECULES:
#       The number of test molecules to compute attributions for.
NUM_EXPLAIN_MOLECULES: int = INHERIT
# :param EXPLAIN_METHOD:
#       Which explanation method to use. 'leave_one_out' (fast, deterministic),
#       'layerwise' (masks at each MP layer with neighborhood distribution),
#       'shap' (KernelSHAP, slower but theoretically grounded), or
#       'myerson' (graph-restricted Shapley values, topology-aware).
EXPLAIN_METHOD: str = INHERIT

# == LAYERWISE-SPECIFIC PARAMETERS (only used when EXPLAIN_METHOD='layerwise') ==

# :param LAYERWISE_DECAY:
#       Exponential decay factor for distributing attribution to neighbors.
LAYERWISE_DECAY: float = INHERIT

# == SHAP-SPECIFIC PARAMETERS (only used when EXPLAIN_METHOD='shap') ==

# :param SHAP_NSAMPLES:
#       Number of coalition samples for KernelSHAP. None uses 'auto'.
SHAP_NSAMPLES: int = INHERIT
# :param SHAP_MASKING_MODE:
#       How to evaluate coalitions. 'additive' or 'causal'.
SHAP_MASKING_MODE: str = INHERIT
# :param SHAP_BACKGROUND:
#       Baseline for SHAP attributions. 'zero' or 'average'.
SHAP_BACKGROUND: str = INHERIT

# == MYERSON-SPECIFIC PARAMETERS (only used when EXPLAIN_METHOD='myerson') ==

# :param MYERSON_MODE:
#       Computation mode. 'exact' enumerates all 2^N coalitions (feasible for
#       ~12 atoms), 'sampled' uses Monte Carlo, 'auto' picks automatically.
MYERSON_MODE: str = INHERIT
# :param MYERSON_MAX_EXACT_NODES:
#       Threshold for auto mode: use exact below this node count.
MYERSON_MAX_EXACT_NODES: int = INHERIT
# :param MYERSON_NUM_SAMPLES:
#       Number of Monte Carlo samples for 'sampled' mode.
MYERSON_NUM_SAMPLES: int = INHERIT
# :param MYERSON_SEED:
#       Random seed for 'sampled' mode reproducibility.
MYERSON_SEED: int = INHERIT
# :param MYERSON_NORM_MODE:
#       Masking strategy for coalition evaluation. 'additive' uses pre-computed
#       node components (fast, no normalization artifacts). 'frozen' reuses
#       norms from the full pass (causal but deflates magnitude). 'recomputed'
#       renormalizes per coalition (causal but may rotate direction).
MYERSON_NORM_MODE: str = INHERIT

# == VISUALIZATION PARAMETERS ==

# :param EXPLAIN_VIZ_CAP:
#       Maximum absolute attribution value for the 2D molecule coloring.
#       Attributions are clipped to [-cap, cap] before rendering so that
#       the color scale is consistent across molecules.
EXPLAIN_VIZ_CAP: float = INHERIT

# Remove the INHERIT sentinel from globals before extend() so pycomex
# does not mistake the uppercase import name for an experiment parameter.
del INHERIT

# == EXPERIMENT PARAMETERS ==

experiment = Experiment.extend(
    'predict_molecules__hdc__explain.py',
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals()
)

experiment.run_if_main()
