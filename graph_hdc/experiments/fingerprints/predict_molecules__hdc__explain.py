"""
Base atom-level explanation experiment for the HDC encoder.

Extends the HDC experiment by adding per-atom attribution of the prediction
using a configurable explanation method. After model training and evaluation,
this experiment re-encodes a subset of test molecules, computes per-atom
attributions, and visualizes them.

Supported explanation methods (``EXPLAIN_METHOD``):

- ``'leave_one_out'``: Zeros each atom's initial feature vector in turn,
  re-runs the encoder, and measures prediction change. Fast and deterministic.
- ``'layerwise'``: Masks at each MP layer with neighborhood distribution.
- ``'shap'``: KernelSHAP over node-level coalition masks.
- ``'myerson'``: Graph-restricted Shapley values, topology-aware.

Sub-experiments should extend this file and override dataset parameters
(``DATASET_NAME``, ``DATASET_TYPE``) and any explanation/visualization
parameters as needed. Per-molecule visualization can be customized by
overriding the ``explain_visualize_molecule`` hook; summary visualization
can be added via the ``explain_summary`` hook.
"""
import os

import numpy as np
import matplotlib.pyplot as plt
import rdkit.Chem as Chem
from rdkit.Chem.Draw import SimilarityMaps, rdMolDraw2D
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

from graph_hdc.models import HyperNet
from graph_hdc.special.molecules import graph_dict_from_mol, make_molecule_node_encoder_map
from graph_hdc.explain import LeaveOneOutExplainer, LayerwiseExplainer, ShapExplainer, MyersonExplainer

# == DATASET PARAMETERS ==

# :param DATASET_NAME:
#       The name of the dataset to be used for the experiment.
DATASET_NAME: str = 'aqsoldb'
# :param DATASET_TYPE:
#       The type of the dataset, either 'classification' or 'regression'.
DATASET_TYPE: str = 'regression'
# :param NUM_TEST:
#       The number of test samples to be used for the evaluation of the models.
NUM_TEST: int = 100

# == EMBEDDING PARAMETERS ==

# :param EMBEDDING_SIZE:
#       The size of the graph embedding vectors (hypervector dimensionality).
EMBEDDING_SIZE: int = 2048
# :param NUM_LAYERS:
#       The number of message-passing layers in the hypernetwork.
NUM_LAYERS: int = 2
# :param HYPER_NET_LAYER_NORMALIZE:
#       Whether to apply L2 normalization to node embeddings after each
#       message-passing layer. Set to False to disable, which preserves the
#       additive decomposition ``g = Σ_i c_i`` and can improve explanation
#       quality. Independent of ``normalize_all`` (post-summation normalization).
HYPER_NET_LAYER_NORMALIZE: bool = True
# :param NN2_USE_BATCHNORM:
#       Whether to use BatchNorm1d in the hidden layers of the neural_net2
#       (PyTorch Lightning) model. Set to False to disable batch normalization,
#       which can improve explanation quality by removing distribution shift
#       between full-graph embeddings and partial coalition embeddings.
NN2_USE_BATCHNORM: bool = False

# == EXPLANATION PARAMETERS ==

# :param EXPLAIN_MODEL_NAME:
#       The name of the trained model to explain. Must match one of the
#       entries in the MODELS list.
EXPLAIN_MODEL_NAME: str = 'neural_net2'
# :param NUM_EXPLAIN_MOLECULES:
#       The number of test molecules to compute attributions for.
NUM_EXPLAIN_MOLECULES: int = 10
# :param EXPLAIN_METHOD:
#       Which explanation method to use. 'leave_one_out' (fast, deterministic),
#       'layerwise' (masks at each MP layer with neighborhood distribution),
#       'shap' (KernelSHAP, slower but theoretically grounded), or
#       'myerson' (graph-restricted Shapley values, topology-aware).
EXPLAIN_METHOD: str = 'myerson'

# == LAYERWISE-SPECIFIC PARAMETERS (only used when EXPLAIN_METHOD='layerwise') ==

# :param LAYERWISE_DECAY:
#       Exponential decay factor for distributing attribution to neighbors.
LAYERWISE_DECAY: float = 0.1

# == SHAP-SPECIFIC PARAMETERS (only used when EXPLAIN_METHOD='shap') ==

# :param SHAP_NSAMPLES:
#       Number of coalition samples for KernelSHAP. Set to None to use the
#       SHAP library's internal 'auto' heuristic.
SHAP_NSAMPLES: int = None
# :param SHAP_MASKING_MODE:
#       How to evaluate coalitions. 'additive' uses pre-computed component sums
#       (fast but information can leak). 'causal' re-runs the forward pass with
#       masked nodes zeroed out at layer 0 (slower but properly removes
#       downstream influence).
SHAP_MASKING_MODE: str = 'causal'
# :param SHAP_BACKGROUND:
#       Baseline for SHAP attributions. 'zero' uses the zero vector (SHAP
#       values explain f(graph) - f(zero)). 'average' uses the mean graph
#       embedding over the dataset (SHAP values explain f(graph) - f(avg)).
#       Only affects 'additive' mode; 'causal' always uses zero.
SHAP_BACKGROUND: str = 'zero'

# == MYERSON-SPECIFIC PARAMETERS (only used when EXPLAIN_METHOD='myerson') ==

# :param MYERSON_MODE:
#       Computation mode. 'exact' enumerates all 2^N coalitions (feasible for
#       ~12 atoms), 'sampled' uses Monte Carlo, 'auto' picks automatically.
MYERSON_MODE: str = 'auto'
# :param MYERSON_MAX_EXACT_NODES:
#       Threshold for auto mode: use exact below this node count.
MYERSON_MAX_EXACT_NODES: int = 12
# :param MYERSON_NUM_SAMPLES:
#       Number of Monte Carlo samples for 'sampled' mode.
MYERSON_NUM_SAMPLES: int = 1000
# :param MYERSON_SEED:
#       Random seed for 'sampled' mode reproducibility.
MYERSON_SEED: int = None
# :param MYERSON_NORM_MODE:
#       Masking strategy for coalition evaluation. 'additive' uses pre-computed
#       node components (fast, no normalization artifacts). 'frozen' reuses
#       norms from the full pass (causal but deflates magnitude). 'recomputed'
#       renormalizes per coalition (causal but may rotate direction).
MYERSON_NORM_MODE: str = 'frozen'

# == VISUALIZATION PARAMETERS ==

# :param EXPLAIN_VIZ_CAP:
#       Maximum absolute attribution value for the 2D molecule coloring.
#       Attributions are clipped to [-cap, cap] before rendering so that
#       the color scale is consistent across molecules.
EXPLAIN_VIZ_CAP: float = 0.25

# == EXPERIMENT PARAMETERS ==

experiment = Experiment.extend(
    'predict_molecules__hdc.py',
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals()
)


METHOD_LABELS = {
    'leave_one_out': 'LOO',
    'layerwise': 'Layerwise',
    'shap': 'SHAP',
    'myerson': 'Myerson',
}


def _build_explainer(e, hyper_net, predict_fn, index_data_map):
    """Build the configured explainer instance."""
    method = e.EXPLAIN_METHOD

    if method == 'leave_one_out':
        return LeaveOneOutExplainer(hyper_net, predict_fn)
    elif method == 'layerwise':
        return LayerwiseExplainer(
            hyper_net, predict_fn, decay=e.LAYERWISE_DECAY,
        )
    elif method == 'shap':
        nsamples = e.SHAP_NSAMPLES if e.SHAP_NSAMPLES is not None else 'auto'
        background_embedding = None
        if e.SHAP_BACKGROUND == 'average':
            all_embeddings = [index_data_map[i]['graph_features']
                              for i in index_data_map]
            background_embedding = np.mean(all_embeddings, axis=0)
            e.log(f'  using average background embedding '
                  f'(computed from {len(all_embeddings)} graphs)')
        return ShapExplainer(
            hyper_net, predict_fn,
            mode=e.SHAP_MASKING_MODE,
            nsamples=nsamples,
            background_embedding=background_embedding,
        )
    elif method == 'myerson':
        return MyersonExplainer(
            hyper_net, predict_fn,
            mode=e.MYERSON_MODE,
            max_exact_nodes=e.MYERSON_MAX_EXACT_NODES,
            num_samples=e.MYERSON_NUM_SAMPLES,
            seed=e.MYERSON_SEED,
            norm_mode=e.MYERSON_NORM_MODE,
        )
    else:
        raise ValueError(
            f"Unknown EXPLAIN_METHOD: {method!r}. "
            f"Use 'leave_one_out', 'layerwise', 'shap', or 'myerson'."
        )


@experiment.hook('evaluate_model', replace=False, default=False)
def evaluate_model(e: Experiment,
                   index_data_map: dict,
                   indices: list,
                   model,
                   key: str,
                   scaler=None,
                   **kwargs,
                   ) -> None:
    """
    After the base evaluation completes, compute per-atom attributions for a
    subset of test molecules when the model matches EXPLAIN_MODEL_NAME.
    """
    model_name = key.replace('test_', '')
    if model_name != e.EXPLAIN_MODEL_NAME:
        return

    e.log('\n== Explanation Parameters ==')
    e.log_parameters()

    method = e.EXPLAIN_METHOD
    e.log(f'\ncomputing {method} attributions '
          f'({e.NUM_EXPLAIN_MOLECULES} molecules, model={model_name})...')

    # Reload the saved HyperNet encoder
    model_path = os.path.join(e.path, 'hyper_net.pth')
    node_encoder_map = make_molecule_node_encoder_map(dim=e.EMBEDDING_SIZE)
    hyper_net = HyperNet(
        hidden_dim=e.EMBEDDING_SIZE,
        depth=e.NUM_LAYERS,
        device='cpu',
        node_encoder_map=node_encoder_map,
    )
    hyper_net.load_from_path(model_path)
    e.log(f'  layer_normalize={hyper_net.layer_normalize}')

    atom_encoder = hyper_net.node_encoder_map.get('node_atoms')

    # Build predict function: use predict_proba for classification
    if e.DATASET_TYPE == 'classification' and hasattr(model, 'predict_proba'):
        predict_fn = lambda X: model.predict_proba(X)[:, 1]
    else:
        predict_fn = lambda X: model.predict(X).astype(float)

    # Build the explainer
    explainer = _build_explainer(e, hyper_net, predict_fn, index_data_map)

    method_label = METHOD_LABELS.get(method, method)
    explain_indices = indices[:e.NUM_EXPLAIN_MOLECULES]

    for mol_idx, idx in enumerate(explain_indices):
        data = index_data_map[idx]
        smiles = str(data['graph_repr'])
        mol = Chem.MolFromSmiles(smiles)

        graph = graph_dict_from_mol(mol)
        del graph['graph_labels']

        # Compute attributions
        node_attr = explainer.explain(graph)
        num_nodes = len(graph['node_indices'])

        # -- Atom labels --
        atom_labels = []
        for i, atom_num in enumerate(graph['node_atoms']):
            if atom_encoder is not None:
                symbol = atom_encoder.get_atomic_symbol(int(atom_num))
            else:
                symbol = str(int(atom_num))
            atom_labels.append(f"{symbol}_{i}")

        # -- Per-molecule visualization via hook --
        e.apply_hook(
            'explain_visualize_molecule',
            mol_idx=mol_idx,
            smiles=smiles,
            mol=mol,
            graph=graph,
            node_attr=node_attr,
            atom_labels=atom_labels,
            method_label=method_label,
            num_nodes=num_nodes,
        )

        e.log(f'  [{mol_idx}] {smiles} (nodes={num_nodes})')

    # -- Summary visualization via hook --
    e.apply_hook(
        'explain_summary',
        method_label=method_label,
        explain_indices=explain_indices,
    )

    e.log(f'finished {method} attributions for {len(explain_indices)} molecules')
    e['explain/method'] = method


@experiment.hook('explain_visualize_molecule')
def default_visualize_molecule(e: Experiment,
                               mol_idx: int,
                               smiles: str,
                               mol,
                               node_attr,
                               atom_labels: list,
                               method_label: str,
                               num_nodes: int,
                               **kwargs,
                               ) -> None:
    """Default per-molecule visualization: bar chart + 2D molecule coloring."""
    # 1. Per-atom bar chart
    fig, ax = plt.subplots(figsize=(max(6, num_nodes * 0.5), 5))
    x = np.arange(num_nodes)
    colors = ['coral' if v > 0 else 'steelblue' for v in node_attr]
    ax.bar(x, node_attr, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(atom_labels, rotation=45, ha='right', fontsize=8)
    ax.set_xlabel('Atom')
    ax.set_ylabel(f'{method_label} Attribution')
    ax.set_title(f'Per-Atom Attribution: {smiles}')
    ax.axhline(y=0, color='black', linewidth=0.5)
    plt.tight_layout()
    e.commit_fig(f'attr_{mol_idx}.png', fig)
    plt.close(fig)

    # 2. 2D molecule with continuous atom coloring via SimilarityMaps
    try:
        cap = e.EXPLAIN_VIZ_CAP
        clipped_attr = np.clip(node_attr, -cap, cap)
        d2d = rdMolDraw2D.MolDraw2DCairo(400, 400)
        SimilarityMaps.GetSimilarityMapFromWeights(
            mol,
            list(clipped_attr),
            draw2d=d2d,
            colorMap='RdBu_r',
        )
        d2d.FinishDrawing()
        png_data = d2d.GetDrawingText()
        mol_path = os.path.join(e.path, f'attr_mol_{mol_idx}.png')
        with open(mol_path, 'wb') as f:
            f.write(png_data)
    except Exception as ex:
        e.log(f'  warning: could not render 2D molecule: {ex}')


experiment.run_if_main()
