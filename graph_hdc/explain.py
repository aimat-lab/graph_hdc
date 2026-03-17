"""
Explanation methods for HDC graph encoders.

Provides classes that compute per-node attribution scores explaining how much
each node contributes to a downstream prediction.

Available methods:

- :class:`LeaveOneOutExplainer` — for each node, zeros its initial embedding,
  re-runs the encoder, and measures the change in prediction. Fast and
  deterministic (one forward pass per node).

- :class:`LayerwiseExplainer` — for each node and each message-passing layer
  (skipping layer 0), zeros the node at that layer, re-runs remaining layers,
  and distributes the prediction change to the node and its k-hop neighbors
  with exponential decay.

- :class:`ShapExplainer` — uses KernelSHAP over node-level coalition masks
  with either additive (pre-computed sum) or causal (re-run forward pass)
  masking. More principled but slower.

- :class:`MyersonExplainer` — uses Myerson values (graph-restricted Shapley
  values) to produce topology-aware attributions. Supports additive
  (pre-computed component sums, no normalization artifacts) and causal
  (masked forward pass) coalition evaluation.
"""
from typing import Optional

import numpy as np
import torch
import shap
from torch_geometric.loader import DataLoader

from graph_hdc.graph import data_list_from_graph_dicts


def nx_graph_from_graph_dict(graph_dict: dict):
    """Convert a graph_dict to a NetworkX undirected graph.

    :param graph_dict: A graph dict with ``'node_indices'`` and
        ``'edge_indices'`` (shape ``(num_edges, 2)``).

    :returns: A ``networkx.Graph`` with integer node IDs.
    """
    import networkx as nx

    G = nx.Graph()
    for i in graph_dict['node_indices']:
        G.add_node(int(i))
    for src, dst in graph_dict['edge_indices']:
        G.add_edge(int(src), int(dst))
    return G


class LeaveOneOutExplainer:
    """
    Leave-one-out node attribution.

    For each node *i*, the initial feature vector is set to zero and the HDC
    forward pass is re-run. The attribution is the difference between the
    full prediction and the leave-one-out prediction:
    ``attribution[i] = pred_full - pred_without_i``.

    A positive value means the node pushes the prediction *up*.

    :param hyper_net: A loaded ``HyperNet`` instance.
    :param predict_fn: A callable ``f(X) -> y`` that maps embeddings of shape
        ``(n, hidden_dim)`` to scalar predictions of shape ``(n,)`` or ``(n, 1)``.
        For regression, use ``model.predict``. For classification, use a
        function that returns continuous probabilities (e.g.
        ``lambda X: model.predict_proba(X)[:, 1]``).
    """

    def __init__(self, hyper_net, predict_fn):
        self.hyper_net = hyper_net
        self.predict_fn = predict_fn

    def explain(self, graph_dict: dict) -> np.ndarray:
        """
        Compute per-node attributions for a single graph.

        :param graph_dict: Graph dict for the molecule (as returned by
            ``graph_dict_from_mol``). Must **not** contain ``'graph_labels'``.

        :returns: Array of shape ``(num_nodes,)`` with the attribution for
            each node.
        """
        data_list = data_list_from_graph_dicts([graph_dict])
        base_data = next(iter(DataLoader(data_list, batch_size=1, shuffle=False)))

        num_nodes = len(graph_dict['node_indices'])

        # Full prediction (all nodes present)
        full_mask = torch.ones(num_nodes, device=self.hyper_net.device)
        with torch.no_grad():
            full_result = self.hyper_net.forward_masked(base_data.clone(), full_mask)
        full_embedding = full_result['graph_embedding'].cpu().numpy()
        pred_full = float(self.predict_fn(full_embedding).flatten()[0])

        # Leave-one-out: mask each node in turn
        attributions = np.empty(num_nodes)
        for i in range(num_nodes):
            mask = torch.ones(num_nodes, device=self.hyper_net.device)
            mask[i] = 0.0
            with torch.no_grad():
                result = self.hyper_net.forward_masked(base_data.clone(), mask)
            embedding = result['graph_embedding'].cpu().numpy()
            pred_without = float(self.predict_fn(embedding).flatten()[0])
            attributions[i] = pred_full - pred_without

        return attributions


class LayerwiseExplainer:
    """
    Layerwise node attribution with neighborhood distribution and frozen
    normalization.

    For each node *i* and each message-passing layer *k* (from 0 to depth),
    zeros node *i*'s embedding at layer *k*, re-runs the remaining forward
    pass, and measures the prediction change. The change is then distributed
    to node *i* and its k-hop neighbors with exponential decay:

    - node *i* receives ``delta``
    - 1-hop neighbors receive ``delta * decay``
    - 2-hop neighbors receive ``delta * decay^2``
    - ... up to k-hop neighbors

    At layer 0 (order-zero / atom identity), the distribution radius is 0
    so only the masked node itself receives the delta (equivalent to
    leave-one-out). At higher layers the neighborhood distribution spreads
    credit to all atoms that participate in the compositional pattern.

    The masked forward passes use **frozen normalization**: the L2 norms
    from the full (unmasked) forward pass are reused so that removing a
    node does not cause remaining nodes' embeddings to artificially rotate
    on the unit sphere.

    :param hyper_net: A loaded ``HyperNet`` instance.
    :param predict_fn: A callable ``f(X) -> y``. See
        :class:`LeaveOneOutExplainer` for details.
    :param decay: Exponential decay factor for distributing attribution to
        neighbors. Default 0.5.
    """

    def __init__(self, hyper_net, predict_fn, decay: float = 0.5):
        self.hyper_net = hyper_net
        self.predict_fn = predict_fn
        self.decay = decay

    def explain(self, graph_dict: dict) -> np.ndarray:
        """
        Compute per-node attributions for a single graph.

        :param graph_dict: Graph dict for the molecule. Must **not** contain
            ``'graph_labels'``.

        :returns: Array of shape ``(num_nodes,)`` with the accumulated
            attribution for each node.
        """
        data_list = data_list_from_graph_dicts([graph_dict])
        base_data = next(iter(DataLoader(data_list, batch_size=1, shuffle=False)))

        num_nodes = len(graph_dict['node_indices'])
        depth = self.hyper_net.depth

        # Full forward pass — also returns frozen norms for masked passes
        with torch.no_grad():
            full_result = self.hyper_net.forward_full_with_norms(
                base_data.clone(),
            )
        full_embedding = full_result['graph_embedding'].cpu().numpy()
        frozen_norms = full_result['norms']
        pred_full = float(self.predict_fn(full_embedding).flatten()[0])

        # Build adjacency list from edge_index
        adj = self._build_adjacency(base_data.edge_index, num_nodes)

        attributions = np.zeros(num_nodes)

        for layer_k in range(0, depth + 1):
            for node_i in range(num_nodes):
                mask = torch.ones(num_nodes, device=self.hyper_net.device)
                mask[node_i] = 0.0

                with torch.no_grad():
                    result = self.hyper_net.forward_masked_frozen_norms(
                        base_data.clone(), mask,
                        mask_layer=layer_k,
                        frozen_norms=frozen_norms,
                    )
                embedding = result['graph_embedding'].cpu().numpy()
                pred_masked = float(self.predict_fn(embedding).flatten()[0])

                delta = pred_full - pred_masked

                # Distribute delta to node_i and its k-hop neighbors
                attributions[node_i] += delta
                visited = {node_i}
                current_frontier = {node_i}
                for hop in range(1, layer_k + 1):
                    next_frontier = set()
                    for n in current_frontier:
                        next_frontier.update(adj[n])
                    next_frontier -= visited
                    for n in next_frontier:
                        attributions[n] += delta * (self.decay ** hop)
                    visited.update(next_frontier)
                    current_frontier = next_frontier

        return attributions

    @staticmethod
    def _build_adjacency(edge_index, num_nodes):
        """Build a symmetric adjacency list from a PyG edge_index."""
        adj = [set() for _ in range(num_nodes)]
        srcs = edge_index[0].cpu().numpy()
        dsts = edge_index[1].cpu().numpy()
        for s, d in zip(srcs, dsts):
            adj[s].add(int(d))
            adj[d].add(int(s))
        return adj


class ShapExplainer:
    """
    KernelSHAP node attribution.

    Treats each node as a player in a coalition game and uses KernelSHAP to
    approximate Shapley values.

    Two masking modes are supported:

    - ``'additive'``: coalition embeddings are computed as ``mask @ components``
      (fast, but information leaks from pre-computed higher-layer vectors).
    - ``'causal'``: the forward pass is re-run with masked nodes zeroed at
      layer 0 using frozen normalization factors from the full pass (slower
      but properly removes downstream influence without normalization
      artifacts).

    :param hyper_net: A loaded ``HyperNet`` instance.
    :param predict_fn: A callable ``f(X) -> y`` that maps embeddings to scalar
        predictions. See :class:`LeaveOneOutExplainer` for details.
    :param mode: ``'additive'`` or ``'causal'``.
    :param nsamples: Number of coalition samples for KernelSHAP.
        ``'auto'`` uses the library's internal heuristic.
    :param background_embedding: Optional array of shape ``(hidden_dim,)``.
        Only used for ``mode='additive'``. Absent players contribute their
        share of this embedding instead of zero.
    """

    def __init__(self,
                 hyper_net,
                 predict_fn,
                 mode: str = 'causal',
                 nsamples='auto',
                 background_embedding: Optional[np.ndarray] = None,
                 ):
        self.hyper_net = hyper_net
        self.predict_fn = predict_fn
        self.mode = mode
        self.nsamples = nsamples
        self.background_embedding = background_embedding

    def explain(self, graph_dict: dict) -> np.ndarray:
        """
        Compute per-node SHAP attributions for a single graph.

        :param graph_dict: Graph dict for the molecule. Must **not** contain
            ``'graph_labels'``.

        :returns: Array of shape ``(num_nodes,)`` with the SHAP value for
            each node.
        """
        # Encode to get node components (needed for additive mode & n_players)
        results = self.hyper_net.forward_graphs(
            [graph_dict], return_node_hv_stack=True,
        )
        node_hv_stack = results[0]['node_hv_stack']
        components = node_hv_stack.sum(axis=1)  # (num_nodes, hidden_dim)
        n_players = components.shape[0]

        if self.mode == 'additive':
            predict_fn = self._make_additive_predict_fn(components)
        elif self.mode == 'causal':
            predict_fn = self._make_causal_predict_fn(graph_dict)
        else:
            raise ValueError(
                f"Unknown mode: {self.mode!r}. Use 'additive' or 'causal'."
            )

        background = np.zeros((1, n_players))
        explainer = shap.KernelExplainer(predict_fn, background)
        instance = np.ones((1, n_players))
        sv = explainer.shap_values(instance, nsamples=self.nsamples)
        return np.array(sv).flatten()

    def _make_additive_predict_fn(self, components: np.ndarray):
        predict_fn = self.predict_fn
        bg = self.background_embedding

        if bg is not None:
            n = components.shape[0]
            share = bg / n

            def predict(masks):
                embeddings = masks @ (components - share) + bg
                return predict_fn(embeddings)
        else:
            def predict(masks):
                embeddings = masks @ components
                return predict_fn(embeddings)

        return predict

    def _make_causal_predict_fn(self, graph_dict: dict):
        data_list = data_list_from_graph_dicts([graph_dict])
        base_data = next(iter(DataLoader(data_list, batch_size=1, shuffle=False)))
        hyper_net = self.hyper_net
        predict_fn = self.predict_fn

        # Pre-compute frozen norms from the full forward pass
        with torch.no_grad():
            full_result = hyper_net.forward_full_with_norms(base_data.clone())
        frozen_norms = full_result['norms']

        def predict(masks):
            results = []
            for mask_row in masks:
                node_mask = torch.tensor(
                    mask_row,
                    dtype=torch.float32,
                    device=hyper_net.device,
                )
                with torch.no_grad():
                    result = hyper_net.forward_masked_frozen_norms(
                        base_data.clone(), node_mask,
                        mask_layer=0,
                        frozen_norms=frozen_norms,
                    )
                embedding = result['graph_embedding'].cpu().numpy()
                pred = predict_fn(embedding)
                results.append(pred.flatten()[0])
            return np.array(results)

        return predict


class MyersonExplainer:
    """
    Myerson value node attribution.

    Uses Myerson values — graph-restricted Shapley values — to attribute
    predictions to individual nodes. Unlike standard Shapley values, Myerson
    values only allow coalitions that form connected subgraphs, making the
    attribution respect the molecular topology.

    Two computation modes are supported:

    - ``'exact'``: Enumerates all 2^N coalitions. Feasible for small graphs
      (up to ~12 nodes).
    - ``'sampled'``: Monte Carlo approximation using random permutations.
      Required for larger graphs.

    The mode can be set explicitly or chosen automatically based on
    ``max_exact_nodes``.

    Uses causal masking at layer 0 so that masked nodes are zeroed from the
    first layer onward, preventing information leakage through message
    passing.

    Three masking strategies are available via ``norm_mode``:

    - ``'additive'`` (recommended): No re-running of the encoder. A single
      full forward pass pre-computes each node's per-layer contribution
      ``c_i = Σ_k h_i^(k)``. Coalition embeddings are then
      ``g(S) = Σ_{i ∈ S} c_i``. This completely avoids normalization
      artifacts and is orders of magnitude faster. The trade-off is a small
      amount of information leakage: higher-layer components were computed
      with all neighbors present. For Myerson values this leakage is
      minimal because coalitions are connected subgraphs, so most relevant
      neighbors are included.
    - ``'frozen'``: Causal masking with frozen L2 norms from the full pass.
      Preserves embedding directions when neighbors are removed but
      systematically deflates the magnitude of small-coalition embeddings
      (fewer messages → shorter raw vectors divided by the full-graph norm).
    - ``'recomputed'``: Causal masking with recomputed L2 norms per
      coalition. Embeddings always have unit norm but their directions
      may rotate when neighbors are removed, with cascading effects across
      layers.

    :param hyper_net: A loaded ``HyperNet`` instance.
    :param predict_fn: A callable ``f(X) -> y`` that maps embeddings of shape
        ``(n, hidden_dim)`` to scalar predictions of shape ``(n,)`` or
        ``(n, 1)``. See :class:`LeaveOneOutExplainer` for details.
    :param mode: ``'exact'``, ``'sampled'``, or ``'auto'`` (default). When
        ``'auto'``, uses exact computation for graphs with at most
        ``max_exact_nodes`` nodes, otherwise samples.
    :param max_exact_nodes: Threshold for auto mode. Default 12.
    :param num_samples: Number of Monte Carlo samples for ``'sampled'`` mode.
        Default 1000.
    :param seed: Random seed for ``'sampled'`` mode reproducibility.
        Default ``None``.
    :param norm_mode: Masking strategy for coalition evaluation.
        ``'additive'`` (default), ``'frozen'``, or ``'recomputed'``.
    """

    def __init__(self,
                 hyper_net,
                 predict_fn,
                 mode: str = 'auto',
                 max_exact_nodes: int = 12,
                 num_samples: int = 1000,
                 seed=None,
                 norm_mode: str = 'additive',
                 ):
        self.hyper_net = hyper_net
        self.predict_fn = predict_fn
        self.mode = mode
        self.max_exact_nodes = max_exact_nodes
        self.num_samples = num_samples
        self.seed = seed
        self.norm_mode = norm_mode

    def explain(self, graph_dict: dict) -> np.ndarray:
        """
        Compute per-node Myerson value attributions for a single graph.

        :param graph_dict: Graph dict for the molecule. Must **not** contain
            ``'graph_labels'``.

        :returns: Array of shape ``(num_nodes,)`` with the Myerson value for
            each node.
        """
        try:
            from myerson import MyersonCalculator, MyersonSampler
        except ImportError:
            raise ImportError(
                "The 'myerson' package is required for MyersonExplainer. "
                "Install it with: pip install myerson"
            )

        num_nodes = len(graph_dict['node_indices'])

        # Full forward pass — always needed for frozen norms, node components,
        # and/or baseline computation.
        data_list = data_list_from_graph_dicts([graph_dict])
        base_data = next(iter(DataLoader(data_list, batch_size=1, shuffle=False)))

        with torch.no_grad():
            full_result = self.hyper_net.forward_full_with_norms(
                base_data.clone(),
            )
        frozen_norms = full_result['norms']

        # For additive mode, pre-compute per-node components from the full
        # forward pass.  c_i = Σ_k h_i^(k) is node i's total contribution
        # to the graph embedding.  Coalition embeddings are then just
        # Σ_{i ∈ S} c_i — no re-running, no normalization artifacts.
        if self.norm_mode == 'additive':
            # node_hv_stack shape: (depth+1, num_nodes, hidden_dim)
            node_hv_stack = full_result['node_hv_stack']
            # Sum across layers → (num_nodes, hidden_dim)
            components = node_hv_stack.sum(dim=0).cpu().numpy()
            coalition_fn = self._make_additive_coalition_function(
                components, num_nodes,
            )
        else:
            # Causal modes (frozen / recomputed): build mask-based closure
            # Compute baseline prediction (empty coalition = all zeros).
            baseline_v = self._eval_coalition(
                base_data, (), num_nodes, frozen_norms,
            )
            coalition_fn = self._make_causal_coalition_function(
                base_data, frozen_norms, num_nodes, baseline_v,
            )

        # Convert to NetworkX for Myerson's coalition enumeration
        nx_graph = nx_graph_from_graph_dict(graph_dict)

        # Determine computation mode
        mode = self.mode
        if mode == 'auto':
            mode = 'exact' if num_nodes <= self.max_exact_nodes else 'sampled'

        # Compute Myerson values
        if mode == 'exact':
            calculator = MyersonCalculator(
                graph=nx_graph,
                coalition_function=coalition_fn,
                disable_tqdm=True,
            )
            values_dict = calculator.calculate_all_myerson_values()
        elif mode == 'sampled':
            sampler = MyersonSampler(
                graph=nx_graph,
                coalition_function=coalition_fn,
                seed=self.seed,
                number_of_samples=self.num_samples,
                disable_tqdm=True,
            )
            values_dict = sampler.sample_all_myerson_values()
        else:
            raise ValueError(
                f"Unknown mode: {self.mode!r}. "
                "Use 'exact', 'sampled', or 'auto'."
            )

        # Convert dict {node_idx: value} to ordered array
        attributions = np.array([values_dict[i] for i in range(num_nodes)])
        return attributions

    # -- Additive mode helpers -------------------------------------------

    def _make_additive_coalition_function(self, components, num_nodes):
        """Create an additive coalition function from pre-computed components.

        For a coalition S the embedding is ``Σ_{i ∈ S} c_i`` where ``c_i``
        is node *i*'s total contribution (sum across all layers) from the
        full forward pass.  The empty coalition produces a zero embedding.

        The value function is automatically centered: ``v(∅) = 0`` because
        ``predict_fn(zero) - predict_fn(zero) = 0``.
        """
        predict_fn = self.predict_fn
        hidden_dim = components.shape[1]

        # Baseline: empty coalition → zero embedding
        baseline_v = float(
            predict_fn(np.zeros((1, hidden_dim))).flatten()[0]
        )

        def coalition_function(coalition, nx_graph):
            if len(coalition) == 0:
                embedding = np.zeros((1, hidden_dim))
            else:
                embedding = components[list(coalition)].sum(
                    axis=0, keepdims=True,
                )
            return float(predict_fn(embedding).flatten()[0]) - baseline_v

        return coalition_function

    # -- Causal mode helpers --------------------------------------------

    def _eval_coalition(self, base_data, coalition, num_nodes, frozen_norms):
        """Evaluate the raw (uncentered) prediction for a single coalition.

        Dispatches to the appropriate forward method based on
        ``self.norm_mode``.
        """
        mask = torch.zeros(
            num_nodes, dtype=torch.float32, device=self.hyper_net.device,
        )
        for node_idx in coalition:
            mask[node_idx] = 1.0

        with torch.no_grad():
            if self.norm_mode == 'frozen':
                result = self.hyper_net.forward_masked_frozen_norms(
                    base_data.clone(), mask,
                    mask_layer=0, frozen_norms=frozen_norms,
                )
            elif self.norm_mode == 'recomputed':
                result = self.hyper_net.forward_masked_at_layer(
                    base_data.clone(), mask, mask_layer=0,
                )
            else:
                raise ValueError(
                    f"Unknown norm_mode: {self.norm_mode!r}. "
                    "Use 'additive', 'frozen', or 'recomputed'."
                )

        embedding = result['graph_embedding'].cpu().numpy()
        return float(self.predict_fn(embedding).flatten()[0])

    def _make_causal_coalition_function(self, base_data, frozen_norms,
                                        num_nodes, baseline_v):
        """Create a causal (mask-based) coalition function.

        Each evaluation re-runs the encoder with masked nodes zeroed at
        layer 0.  The value function is centered so that ``v(∅) = 0``.
        """
        explainer = self

        def coalition_function(coalition, nx_graph):
            raw = explainer._eval_coalition(
                base_data, coalition, num_nodes, frozen_norms,
            )
            return raw - baseline_v

        return coalition_function
