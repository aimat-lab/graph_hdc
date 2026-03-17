"""
SHAP attribution experiment for CLogP prediction with the HDC encoder.

Extends the HDC CLogP experiment by adding Shapley-value-based attribution of
the prediction to individual atoms (nodes). After model training and
evaluation, this experiment re-encodes a subset of test molecules, treats each
atom as a SHAP player, and runs KernelSHAP.

The per-atom SHAP attributions are compared against the known Crippen atom
contributions (Wildman-Crippen method) as a ground-truth validation.
"""
import os

import numpy as np
import matplotlib.pyplot as plt
import rdkit.Chem as Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import SimilarityMaps, rdMolDraw2D
from rdkit.Chem.Crippen import MolLogP, _GetAtomContribs
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

from graph_hdc.models import HyperNet
from graph_hdc.special.molecules import graph_dict_from_mol, make_molecule_node_encoder_map
from graph_hdc.shap import (
    get_node_components,
    compute_shap_values,
)

# == DATASET PARAMETERS ==

DATASET_NAME: str = 'aqsoldb'
DATASET_TYPE: str = 'regression'
NUM_TEST: int = 100

# == EMBEDDING PARAMETERS ==

EMBEDDING_SIZE: int = 2048
NUM_LAYERS: int = 2

# == SHAP PARAMETERS ==

# :param SHAP_MODEL_NAME:
#       The name of the trained model to explain with SHAP. Must match one of
#       the entries in the MODELS list.
SHAP_MODEL_NAME: str = 'neural_net'
# :param NUM_SHAP_MOLECULES:
#       The number of test molecules to compute SHAP attributions for.
NUM_SHAP_MOLECULES: int = 10
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
#       Only affects 'additive' mode; 'causal' always uses zero (absent
#       nodes are physically removed from the graph).
SHAP_BACKGROUND: str = 'zero'

# == EXPERIMENT PARAMETERS ==

experiment = Experiment.extend(
    'predict_molecules__hdc__clogp.py',
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals()
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
    After the base evaluation completes, compute SHAP attributions for a subset
    of test molecules when the model matches SHAP_MODEL_NAME.
    """
    model_name = key.replace('test_', '')
    if model_name != e.SHAP_MODEL_NAME:
        return

    e.log(f'\ncomputing SHAP attributions ({e.NUM_SHAP_MOLECULES} molecules, model={model_name})...')

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

    atom_encoder = hyper_net.node_encoder_map.get('node_atoms')

    shap_indices = indices[:e.NUM_SHAP_MOLECULES]

    nsamples = e.SHAP_NSAMPLES if e.SHAP_NSAMPLES is not None else 'auto'

    # Compute background embedding if requested
    background_embedding = None
    if e.SHAP_BACKGROUND == 'average':
        all_embeddings = [index_data_map[i]['graph_features'] for i in index_data_map]
        background_embedding = np.mean(all_embeddings, axis=0)
        e.log(f'  using average background embedding (computed from {len(all_embeddings)} graphs)')

    all_crippen_contribs = []
    all_shap_contribs = []

    for mol_idx, idx in enumerate(shap_indices):
        data = index_data_map[idx]
        smiles = str(data['graph_repr'])
        mol = Chem.MolFromSmiles(smiles)

        # Re-encode this single molecule with node_hv_stack
        graph = graph_dict_from_mol(mol)
        del graph['graph_labels']
        results = hyper_net.forward_graphs([graph], return_node_hv_stack=True)
        result = results[0]

        node_hv_stack = result['node_hv_stack']
        graph_embedding = result['graph_embedding']
        components = get_node_components(node_hv_stack)
        num_nodes = node_hv_stack.shape[0]

        # Verify additive decomposition (pooling='sum' guard)
        assert np.allclose(components.sum(axis=0), graph_embedding, atol=1e-4), \
            "Components do not sum to graph embedding — pooling may not be 'sum'"

        # Compute SHAP values (one per node)
        node_sv = compute_shap_values(
            components, model,
            nsamples=nsamples,
            mode=e.SHAP_MASKING_MODE,
            hyper_net=hyper_net,
            graph_dict=graph,
            background_embedding=background_embedding,
        )

        # Crippen atom contributions for comparison
        crippen = _GetAtomContribs(mol)
        crippen_logp = np.array([c[0] for c in crippen])

        all_crippen_contribs.extend(crippen_logp.tolist())
        all_shap_contribs.extend(node_sv.tolist())

        # -- Atom labels --
        atom_labels = []
        for i, atom_num in enumerate(graph['node_atoms']):
            if atom_encoder is not None:
                symbol = atom_encoder.get_atomic_symbol(int(atom_num))
            else:
                symbol = str(int(atom_num))
            atom_labels.append(f"{symbol}_{i}")

        # -- Per-molecule visualizations --

        # 1. Per-atom bar chart comparing SHAP vs Crippen
        fig, ax = plt.subplots(figsize=(max(6, num_nodes * 0.5), 5))
        x = np.arange(num_nodes)
        width = 0.35
        ax.bar(x - width / 2, node_sv, width, label='SHAP', color='steelblue')
        ax.bar(x + width / 2, crippen_logp, width, label='Crippen', color='coral')
        ax.set_xticks(x)
        ax.set_xticklabels(atom_labels, rotation=45, ha='right', fontsize=8)
        ax.set_xlabel('Atom')
        ax.set_ylabel('Attribution')
        ax.set_title(f'Per-Atom Attribution: {smiles}')
        ax.legend()
        ax.axhline(y=0, color='black', linewidth=0.5)
        plt.tight_layout()
        e.commit_fig(f'shap_vs_crippen_{mol_idx}.png', fig)
        plt.close(fig)

        # 2. 2D molecule with continuous atom coloring via SimilarityMaps
        try:
            d2d = rdMolDraw2D.MolDraw2DCairo(400, 400)
            SimilarityMaps.GetSimilarityMapFromWeights(
                mol,
                list(node_sv),
                draw2d=d2d,
                colorMap='RdBu_r',
            )
            d2d.FinishDrawing()
            png_data = d2d.GetDrawingText()
            mol_path = os.path.join(e.path, f'shap_mol_{mol_idx}.png')
            with open(mol_path, 'wb') as f:
                f.write(png_data)
        except Exception as ex:
            e.log(f'  warning: could not render 2D molecule: {ex}')

        e.log(f'  [{mol_idx}] {smiles} (nodes={num_nodes})')

    # -- Summary visualization --

    # Correlation scatter: SHAP vs Crippen
    all_crippen_contribs = np.array(all_crippen_contribs)
    all_shap_contribs = np.array(all_shap_contribs)
    corr = np.corrcoef(all_crippen_contribs, all_shap_contribs)[0, 1]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(all_crippen_contribs, all_shap_contribs, alpha=0.5, s=20, color='steelblue')
    ax.set_xlabel('Crippen Atom Contribution')
    ax.set_ylabel('SHAP Atom Attribution')
    ax.set_title(f'SHAP vs Crippen (r={corr:.3f})')
    lims = [
        min(ax.get_xlim()[0], ax.get_ylim()[0]),
        max(ax.get_xlim()[1], ax.get_ylim()[1]),
    ]
    ax.plot(lims, lims, 'k--', alpha=0.3)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    plt.tight_layout()
    e.commit_fig('shap_vs_crippen_correlation.png', fig)
    plt.close(fig)

    e.log(f'SHAP vs Crippen correlation: r={corr:.3f}')
    e['shap/crippen_correlation'] = float(corr)


experiment.run_if_main()
