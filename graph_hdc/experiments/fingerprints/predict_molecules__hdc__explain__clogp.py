"""
Atom-level explanation experiment for CLogP prediction with the HDC encoder.

Extends the base explanation experiment with CLogP-specific behavior:
replaces dataset targets with computed Crippen LogP values and adds
per-atom comparison against known Crippen atom contributions as a
ground-truth validation.
"""
import os

import numpy as np
import matplotlib.pyplot as plt
import rdkit.Chem as Chem
from rdkit.Chem.Draw import SimilarityMaps, rdMolDraw2D
from rdkit.Chem.Crippen import MolLogP, _GetAtomContribs
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace
from pycomex import INHERIT

# == DATASET PARAMETERS ==

# :param DATASET_NAME:
#       The name of the dataset to be used for the experiment.
DATASET_NAME: str = 'aqsoldb'
# :param DATASET_TYPE:
#       The type of the dataset, either 'classification' or 'regression'.
DATASET_TYPE: str = 'regression'
# :param NUM_TEST:
#       The number of test samples to be used for the evaluation of the models.
NUM_TEST: int = INHERIT

# == EMBEDDING PARAMETERS ==

# :param EMBEDDING_SIZE:
#       The size of the graph embedding vectors (hypervector dimensionality).
EMBEDDING_SIZE: int = INHERIT
# :param NUM_LAYERS:
#       The number of message-passing layers in the hypernetwork.
NUM_LAYERS: int = 3

# == EXPLANATION PARAMETERS ==

# :param EXPLAIN_MODEL_NAME:
#       The name of the trained model to explain. Must match one of the
#       entries in the MODELS list.
EXPLAIN_MODEL_NAME: str = INHERIT
# :param NUM_EXPLAIN_MOLECULES:
#       The number of test molecules to compute attributions for.
NUM_EXPLAIN_MOLECULES: int = 15
# :param EXPLAIN_METHOD:
#       Which explanation method to use. 'leave_one_out' (fast, deterministic),
#       'layerwise' (masks at each MP layer with neighborhood distribution),
#       'shap' (KernelSHAP, slower but theoretically grounded), or
#       'myerson' (graph-restricted Shapley values, topology-aware).
EXPLAIN_METHOD: str = INHERIT

# == LAYERWISE-SPECIFIC PARAMETERS (only used when EXPLAIN_METHOD='layerwise') ==

# :param LAYERWISE_DECAY:
#       Exponential decay factor for distributing attribution to neighbors.
LAYERWISE_DECAY: float = 0.5

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
MYERSON_SEED: int = 2
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
EXPLAIN_VIZ_CAP: float = INHERIT

SEED = 2

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


@experiment.hook('after_dataset', replace=False, default=False)
def after_dataset(e: Experiment,
                  index_data_map: dict[int, dict],
                  **kwargs,
                  ) -> None:
    """Replace dataset targets with computed CLogP values."""
    e.log('calculating CLogP values and replacing targets...')
    for _, graph in index_data_map.items():
        smiles = str(graph['graph_repr'])
        mol = Chem.MolFromSmiles(smiles)
        graph['graph_labels'] = np.array([MolLogP(mol)])


@experiment.hook('explain_visualize_molecule', default=False)
def visualize_molecule_crippen(e: Experiment,
                               mol_idx: int,
                               smiles: str,
                               mol,
                               graph: dict,
                               node_attr,
                               atom_labels: list,
                               method_label: str,
                               num_nodes: int,
                               **kwargs,
                               ) -> None:
    """Per-molecule visualization comparing attribution vs Crippen contributions."""
    # Crippen atom contributions for comparison
    crippen = _GetAtomContribs(mol)
    crippen_logp = np.array([c[0] for c in crippen])

    # Accumulate for summary correlation plot
    if '_crippen_contribs' not in e.data:
        e.data['_crippen_contribs'] = []
        e.data['_explain_contribs'] = []
    e.data['_crippen_contribs'].extend(crippen_logp.tolist())
    e.data['_explain_contribs'].extend(node_attr.tolist())

    # 1. Per-atom bar chart comparing attribution vs Crippen
    fig, ax = plt.subplots(figsize=(max(6, num_nodes * 0.5), 5))
    x = np.arange(num_nodes)
    width = 0.35
    ax.bar(x - width / 2, node_attr, width, label=method_label,
           color='steelblue')
    ax.bar(x + width / 2, crippen_logp, width, label='Crippen',
           color='coral')
    ax.set_xticks(x)
    ax.set_xticklabels(atom_labels, rotation=45, ha='right', fontsize=8)
    ax.set_xlabel('Atom')
    ax.set_ylabel('Attribution')
    ax.set_title(f'Per-Atom Attribution: {smiles}')
    ax.legend()
    ax.axhline(y=0, color='black', linewidth=0.5)
    plt.tight_layout()
    e.commit_fig(f'attr_vs_crippen_{mol_idx}.png', fig)
    plt.close(fig)

    # 2. 2D molecule with continuous atom coloring via SimilarityMaps
    try:
        d2d = rdMolDraw2D.MolDraw2DCairo(400, 400)
        SimilarityMaps.GetSimilarityMapFromWeights(
            mol,
            list(node_attr),
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


@experiment.hook('explain_summary', default=False)
def summary_crippen_correlation(e: Experiment,
                                method_label: str,
                                **kwargs,
                                ) -> None:
    """Summary: correlation scatter plot of attribution vs Crippen."""
    all_crippen = np.array(e.data.get('_crippen_contribs', []))
    all_explain = np.array(e.data.get('_explain_contribs', []))

    if len(all_crippen) == 0:
        return

    corr = np.corrcoef(all_crippen, all_explain)[0, 1]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(all_crippen, all_explain,
               alpha=0.5, s=20, color='steelblue')
    ax.set_xlabel('Crippen Atom Contribution')
    ax.set_ylabel(f'{method_label} Atom Attribution')
    ax.set_title(f'{method_label} vs Crippen (r={corr:.3f})')
    lims = [
        min(ax.get_xlim()[0], ax.get_ylim()[0]),
        max(ax.get_xlim()[1], ax.get_ylim()[1]),
    ]
    ax.plot(lims, lims, 'k--', alpha=0.3)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    plt.tight_layout()
    e.commit_fig('attr_vs_crippen_correlation.png', fig)
    plt.close(fig)

    e.log(f'{method_label} vs Crippen correlation: r={corr:.3f}')
    e['explain/crippen_correlation'] = float(corr)


experiment.run_if_main()
