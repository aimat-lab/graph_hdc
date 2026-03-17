"""
SHAP-based feature attribution using HDC node components as players.

The HDC encoder produces a graph embedding as a plain sum of per-node, per-layer
hypervectors: g = Σ_k Σ_i h_i^(k). This module treats each **node** as a player
in a Shapley value game, with its contribution being the sum across all layers:
c_i = Σ_k h_i^(k).

Two masking modes are supported:

- **additive**: Coalition embeddings are computed as ``mask @ components`` —
  simply summing the included players' pre-computed node vectors. Fast (matrix
  multiply) but information can leak since later-layer vectors were computed
  with the full graph.

- **causal**: For each coalition, the HyperNet forward pass is re-run with
  masked nodes zeroed out at layer 0. Since bind(0, x) = 0 for circular
  convolution, a node masked at layer 0 stays zero at all subsequent layers
  and does not contribute messages to neighbors. More expensive but more correct.

Requires ``pooling='sum'`` in the HyperNet encoder (the default).
"""
from typing import List, Optional

import numpy as np
import torch
import shap
from torch_geometric.loader import DataLoader

from graph_hdc.graph import data_list_from_graph_dicts


def get_node_components(node_hv_stack: np.ndarray) -> np.ndarray:
    """
    Sum per-node embeddings across layers to get one component per node.

    Each row of the output is one player c_i = Σ_k h_i^(k). The sum of all
    rows equals the graph embedding (when ``pooling='sum'``).

    :param node_hv_stack: Array of shape ``(num_nodes, num_layers, hidden_dim)``
        as returned by ``HyperNet.forward_graphs(..., return_node_hv_stack=True)``.

    :returns: Array of shape ``(num_nodes, hidden_dim)``.
    """
    return node_hv_stack.sum(axis=1)


def get_player_components(node_hv_stack: np.ndarray) -> np.ndarray:
    """
    Reshape per-node, per-layer embeddings into a flat array of player components.

    Each row of the output is one player h_i^(k). The sum of all rows equals the
    graph embedding (when ``pooling='sum'``).

    :param node_hv_stack: Array of shape ``(num_nodes, num_layers, hidden_dim)``
        as returned by ``HyperNet.forward_graphs(..., return_node_hv_stack=True)``.

    :returns: Array of shape ``(num_nodes * num_layers, hidden_dim)``.
    """
    num_nodes, num_layers, hidden_dim = node_hv_stack.shape
    return node_hv_stack.reshape(num_nodes * num_layers, hidden_dim)


def make_coalition_predict_fn(components: np.ndarray,
                              model,
                              background_embedding: Optional[np.ndarray] = None,
                              ):
    """
    Create a prediction function that maps binary coalition masks to model
    predictions using the **additive** approach.

    When ``background_embedding`` is ``None`` (zero baseline), a coalition
    embedding is simply ``mask @ components``. When a background embedding is
    provided, absent nodes contribute their equal share of the background
    so that f(empty) = f(background_embedding).

    :param components: Array of shape ``(n_players, hidden_dim)``.
    :param model: A trained sklearn-compatible model with a ``predict`` method.
    :param background_embedding: Optional array of shape ``(hidden_dim,)``.
        When provided, absent players contribute ``background_embedding / n``
        instead of zero.

    :returns: A callable ``f(masks) -> predictions`` where ``masks`` has shape
        ``(n_samples, n_players)`` with binary values.
    """
    if background_embedding is not None:
        n_players = components.shape[0]
        share = background_embedding / n_players

        def predict(masks):
            # included → component, absent → equal share of background
            embeddings = masks @ (components - share) + background_embedding
            return model.predict(embeddings)
    else:
        def predict(masks):
            embeddings = masks @ components
            return model.predict(embeddings)

    return predict


def make_causal_coalition_predict_fn(hyper_net, graph_dict, model):
    """
    Create a prediction function that maps binary node-level coalition masks
    to model predictions using the **causal** approach.

    For each coalition, the HyperNet forward pass is re-run with masked nodes
    zeroed out at layer 0. Since bind(0, x) = 0 for circular convolution,
    masked nodes stay zero at all subsequent layers and do not contribute
    messages to their neighbors.

    :param hyper_net: A ``HyperNet`` instance (already loaded with encoders).
    :param graph_dict: The graph dict for the molecule to explain.
    :param model: A trained sklearn-compatible model with a ``predict`` method.

    :returns: A callable ``f(masks) -> predictions`` where ``masks`` has shape
        ``(n_samples, num_nodes)`` with binary values.
    """
    data_list = data_list_from_graph_dicts([graph_dict])
    base_data = next(iter(DataLoader(data_list, batch_size=1, shuffle=False)))

    def predict(masks):
        results = []
        for mask_row in masks:
            node_mask = torch.tensor(
                mask_row,
                dtype=torch.float32,
                device=hyper_net.device,
            )
            with torch.no_grad():
                result = hyper_net.forward_masked(base_data.clone(), node_mask)
            embedding = result['graph_embedding'].cpu().numpy()
            pred = model.predict(embedding)
            results.append(pred.flatten()[0])
        return np.array(results)

    return predict


def compute_shap_values(components: np.ndarray,
                        model,
                        nsamples='auto',
                        mode: str = 'additive',
                        hyper_net=None,
                        graph_dict: dict = None,
                        background_embedding: Optional[np.ndarray] = None,
                        ) -> np.ndarray:
    """
    Compute SHAP values for a single graph using KernelSHAP.

    Players are nodes: one SHAP value per node.

    :param components: Array of shape ``(num_nodes, hidden_dim)``. Required for
        ``mode='additive'``, used only for determining ``n_players`` in
        ``mode='causal'``.
    :param model: A trained sklearn-compatible model with a ``predict`` method.
    :param nsamples: Number of coalition samples for KernelSHAP. ``'auto'``
        uses the SHAP library's internal heuristic.
    :param mode: ``'additive'`` (fast, pre-computed sum) or ``'causal'``
        (re-runs forward pass with node masking at layer 0).
    :param hyper_net: Required for ``mode='causal'``. The HyperNet encoder.
    :param graph_dict: Required for ``mode='causal'``. The graph dict for the
        molecule being explained.
    :param background_embedding: Optional array of shape ``(hidden_dim,)``.
        Only used for ``mode='additive'``. When provided, absent players
        contribute their share of this embedding instead of zero, so SHAP
        values explain ``f(graph) - f(background)`` instead of
        ``f(graph) - f(zero)``.

    :returns: Array of shape ``(num_nodes,)`` with the SHAP value for each node.
    """
    n_players = components.shape[0]

    if mode == 'additive':
        predict_fn = make_coalition_predict_fn(
            components, model, background_embedding=background_embedding,
        )
    elif mode == 'causal':
        if hyper_net is None or graph_dict is None:
            raise ValueError("mode='causal' requires hyper_net and graph_dict")
        predict_fn = make_causal_coalition_predict_fn(hyper_net, graph_dict, model)
    else:
        raise ValueError(f"Unknown mode: {mode!r}. Use 'additive' or 'causal'.")

    background = np.zeros((1, n_players))
    explainer = shap.KernelExplainer(predict_fn, background)
    instance = np.ones((1, n_players))
    sv = explainer.shap_values(instance, nsamples=nsamples)
    return np.array(sv).flatten()


def compute_exact_shap_linear(components: np.ndarray, model) -> np.ndarray:
    """
    Compute exact Shapley values for a linear regression model (no sampling).

    For ``f(x) = w^T x + b``, the marginal contribution of player ``i`` to
    any coalition is constant: ``w^T c_i``. The intercept ``b`` is
    distributed equally across all players to satisfy the efficiency axiom.

    This serves as a verification oracle for the KernelSHAP pipeline: if
    KernelSHAP produces matching results on a linear model, the pipeline is
    correctly wired.

    :param components: Array of shape ``(n_players, hidden_dim)``.
    :param model: A trained ``sklearn.linear_model.LinearRegression`` instance.

    :returns: Array of shape ``(n_players,)`` with exact SHAP values.
    """
    w = model.coef_.flatten()
    b = float(model.intercept_) if np.isscalar(model.intercept_) else float(model.intercept_[0])
    n_players = components.shape[0]
    shap_values = components @ w + b / n_players
    return shap_values


def aggregate_shap_by_node(shap_values: np.ndarray,
                           num_nodes: int,
                           num_layers: int,
                           ) -> np.ndarray:
    """
    Aggregate SHAP values per node by summing across layers.

    Preserves the efficiency property: the sum of node-level values equals
    ``f(full) - f(empty)``.

    :param shap_values: Array of shape ``(num_nodes * num_layers,)``.
    :param num_nodes: Number of nodes in the graph.
    :param num_layers: Number of layers (``depth + 1``).

    :returns: Array of shape ``(num_nodes,)``.
    """
    return shap_values.reshape(num_nodes, num_layers).sum(axis=1)


def aggregate_shap_by_layer(shap_values: np.ndarray,
                            num_nodes: int,
                            num_layers: int,
                            absolute: bool = True,
                            ) -> np.ndarray:
    """
    Aggregate SHAP values per layer by summing across nodes.

    :param shap_values: Array of shape ``(num_nodes * num_layers,)``.
    :param num_nodes: Number of nodes in the graph.
    :param num_layers: Number of layers (``depth + 1``).
    :param absolute: If ``True``, sum absolute values (importance magnitude;
        breaks efficiency). If ``False``, sum signed values (preserves
        efficiency but cancellation may hide contributions).

    :returns: Array of shape ``(num_layers,)``.
    """
    reshaped = shap_values.reshape(num_nodes, num_layers)
    if absolute:
        return np.abs(reshaped).sum(axis=0)
    else:
        return reshaped.sum(axis=0)


def get_player_labels(node_atoms: np.ndarray,
                      num_layers: int,
                      atom_encoder=None,
                      ) -> List[str]:
    """
    Generate human-readable labels for each player.

    :param node_atoms: Array of atomic numbers, shape ``(num_nodes,)``.
    :param num_layers: Number of layers (``depth + 1``).
    :param atom_encoder: Optional ``AtomEncoder`` instance for resolving atomic
        numbers to element symbols. If ``None``, raw atomic numbers are used.

    :returns: List of strings like ``'C_0 (k=0)'``, ``'O_3 (k=1)'``, etc.
    """
    labels = []
    for i, atom_num in enumerate(node_atoms):
        if atom_encoder is not None:
            symbol = atom_encoder.get_atomic_symbol(int(atom_num))
        else:
            symbol = str(int(atom_num))
        for k in range(num_layers):
            labels.append(f"{symbol}_{i} (k={k})")
    return labels
