import numpy as np
import pytest
import torch
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from graph_hdc.models import HyperNet, CategoricalIntegerEncoder
from graph_hdc.testing import generate_random_graphs
from graph_hdc.shap import (
    get_node_components,
    get_player_components,
    make_coalition_predict_fn,
    make_causal_coalition_predict_fn,
    compute_shap_values,
    compute_exact_shap_linear,
    aggregate_shap_by_node,
    aggregate_shap_by_layer,
    get_player_labels,
)


# -- Helper --

def _make_simple_hyper_net(dim=200, depth=2):
    """Create a small HyperNet for testing."""
    return HyperNet(
        hidden_dim=dim,
        depth=depth,
        node_encoder_map={
            'node_type': CategoricalIntegerEncoder(dim=dim, num_categories=5),
        },
    )


def _make_test_graphs(num_graphs=3, num_node_range=(5, 10)):
    """Generate random graphs with a 'node_type' property."""
    graphs = generate_random_graphs(
        num_graphs,
        num_node_range=num_node_range,
    )
    for g in graphs:
        g['node_type'] = np.random.randint(0, 5, size=len(g['node_indices']))
    return graphs


def _train_linear_model(dim=20, n_samples=50):
    """Train a simple linear regression on random data."""
    model = LinearRegression()
    X = np.random.randn(n_samples, dim)
    y = np.random.randn(n_samples)
    model.fit(X, y)
    return model


# -- Tests --

class TestGetNodeComponents:

    def test_shape(self):
        """(5 nodes, 3 layers, 100 dim) -> (5, 100)"""
        stack = np.random.randn(5, 3, 100)
        components = get_node_components(stack)
        assert components.shape == (5, 100)

    def test_single_node(self):
        """Edge case: single-node graph."""
        stack = np.random.randn(1, 3, 100)
        components = get_node_components(stack)
        assert components.shape == (1, 100)

    def test_sum_equals_embedding(self):
        """Sum of all node components must equal graph_embedding from HyperNet."""
        dim = 200
        depth = 2
        hyper_net = _make_simple_hyper_net(dim=dim, depth=depth)
        graphs = _make_test_graphs(num_graphs=3, num_node_range=(5, 10))

        results = hyper_net.forward_graphs(graphs, return_node_hv_stack=True)

        for result in results:
            components = get_node_components(result['node_hv_stack'])
            embedding = result['graph_embedding']
            np.testing.assert_allclose(
                components.sum(axis=0),
                embedding,
                atol=1e-4,
                err_msg="Node components do not sum to graph embedding",
            )


class TestGetPlayerComponents:

    def test_shape(self):
        """(5 nodes, 3 layers, 100 dim) -> (15, 100)"""
        stack = np.random.randn(5, 3, 100)
        components = get_player_components(stack)
        assert components.shape == (15, 100)

    def test_sum_equals_embedding(self):
        """Sum of all player components must equal graph_embedding from HyperNet."""
        dim = 200
        depth = 2
        hyper_net = _make_simple_hyper_net(dim=dim, depth=depth)
        graphs = _make_test_graphs(num_graphs=3, num_node_range=(5, 10))

        results = hyper_net.forward_graphs(graphs, return_node_hv_stack=True)

        for result in results:
            components = get_player_components(result['node_hv_stack'])
            embedding = result['graph_embedding']
            np.testing.assert_allclose(
                components.sum(axis=0),
                embedding,
                atol=1e-4,
                err_msg="Components do not sum to graph embedding",
            )


class TestCoalitionPredict:

    def test_full_coalition(self):
        """Full mask gives same prediction as full embedding."""
        components = np.random.randn(10, 50)
        full_embedding = components.sum(axis=0)

        model = _train_linear_model(dim=50)
        predict_fn = make_coalition_predict_fn(components, model)

        pred_coalition = predict_fn(np.ones((1, 10)))
        pred_direct = model.predict(full_embedding.reshape(1, -1))

        np.testing.assert_allclose(pred_coalition, pred_direct)

    def test_empty_coalition(self):
        """Empty mask predicts from zero vector."""
        components = np.random.randn(10, 50)

        model = _train_linear_model(dim=50)
        predict_fn = make_coalition_predict_fn(components, model)

        pred = predict_fn(np.zeros((1, 10)))
        zero_pred = model.predict(np.zeros((1, 50)))

        np.testing.assert_allclose(pred, zero_pred)

    def test_partial_mask(self):
        """Mask [1,0,1,0] produces sum of selected components only."""
        components = np.random.randn(4, 50)

        model = _train_linear_model(dim=50)
        predict_fn = make_coalition_predict_fn(components, model)

        mask = np.array([[1, 0, 1, 0]])
        pred = predict_fn(mask)
        expected_embedding = (components[0] + components[2]).reshape(1, -1)
        expected_pred = model.predict(expected_embedding)

        np.testing.assert_allclose(pred, expected_pred)


class TestLinearModelOracle:

    def test_exact_shap_matches_kernel_shap(self):
        """For a linear model, compute_exact_shap_linear and compute_shap_values
        should produce matching results. This verifies the full SHAP pipeline.
        """
        np.random.seed(42)
        components = np.random.randn(6, 20).astype(np.float64)

        model = _train_linear_model(dim=20, n_samples=50)

        exact_sv = compute_exact_shap_linear(components, model)
        kernel_sv = compute_shap_values(components, model, nsamples=5000)

        np.testing.assert_allclose(
            exact_sv, kernel_sv, atol=0.15,
            err_msg="KernelSHAP does not match exact linear SHAP values",
        )


class TestShapValuesEfficiency:

    def test_shap_values_sum_property(self):
        """SHAP values must sum to f(full) - f(empty)."""
        np.random.seed(42)
        components = np.random.randn(6, 20).astype(np.float64)

        model = _train_linear_model(dim=20, n_samples=50)

        sv = compute_shap_values(components, model, nsamples=5000)

        full_pred = model.predict(components.sum(axis=0).reshape(1, -1))[0]
        empty_pred = model.predict(np.zeros((1, 20)))[0]

        np.testing.assert_allclose(
            sv.sum(), full_pred - empty_pred, atol=0.15,
            err_msg="SHAP values do not sum to f(full) - f(empty)",
        )


class TestAggregation:

    def test_aggregate_by_node(self):
        """Correct summation across layers per node."""
        sv = np.arange(12, dtype=float)
        result = aggregate_shap_by_node(sv, 4, 3)
        assert result.shape == (4,)
        assert result[0] == 0 + 1 + 2
        assert result[1] == 3 + 4 + 5
        assert result[2] == 6 + 7 + 8
        assert result[3] == 9 + 10 + 11

    def test_aggregate_by_layer_absolute(self):
        """Absolute summation across nodes per layer."""
        sv = np.array([1, -2, 3, -4, 5, -6], dtype=float)
        result = aggregate_shap_by_layer(sv, 2, 3, absolute=True)
        assert result.shape == (3,)
        assert result[0] == abs(1) + abs(-4)
        assert result[1] == abs(-2) + abs(5)
        assert result[2] == abs(3) + abs(-6)

    def test_aggregate_by_layer_signed(self):
        """Signed summation across nodes per layer, preserves efficiency."""
        sv = np.array([1, -2, 3, -4, 5, -6], dtype=float)
        result = aggregate_shap_by_layer(sv, 2, 3, absolute=False)
        assert result.shape == (3,)
        assert result[0] == 1 + (-4)
        assert result[1] == -2 + 5
        assert result[2] == 3 + (-6)

    def test_signed_preserves_efficiency(self):
        """Both node and signed-layer aggregation preserve total sum."""
        sv = np.random.randn(12)
        node_agg = aggregate_shap_by_node(sv, 4, 3)
        layer_agg = aggregate_shap_by_layer(sv, 4, 3, absolute=False)
        np.testing.assert_allclose(node_agg.sum(), sv.sum())
        np.testing.assert_allclose(layer_agg.sum(), sv.sum())


class TestForwardMasked:

    def test_full_mask_matches_forward(self):
        """forward_masked with all-ones mask should match regular forward."""
        dim = 200
        depth = 2
        hyper_net = _make_simple_hyper_net(dim=dim, depth=depth)
        graphs = _make_test_graphs(num_graphs=2, num_node_range=(4, 7))

        results_normal = hyper_net.forward_graphs(graphs)

        from graph_hdc.graph import data_list_from_graph_dicts
        from torch_geometric.loader import DataLoader

        data_list = data_list_from_graph_dicts(graphs)
        loader = DataLoader(data_list, batch_size=1, shuffle=False)

        for i, batch_data in enumerate(loader):
            num_nodes = len(graphs[i]['node_indices'])
            node_mask = torch.ones(num_nodes)
            with torch.no_grad():
                result_masked = hyper_net.forward_masked(batch_data, node_mask)
            np.testing.assert_allclose(
                result_masked['graph_embedding'].cpu().numpy(),
                results_normal[i]['graph_embedding'].reshape(1, -1),
                atol=1e-4,
                err_msg="forward_masked with full mask differs from forward",
            )

    def test_zero_mask_gives_zero_embedding(self):
        """forward_masked with all-zeros mask should give zero embedding."""
        dim = 200
        depth = 2
        hyper_net = _make_simple_hyper_net(dim=dim, depth=depth)
        graphs = _make_test_graphs(num_graphs=1, num_node_range=(4, 7))

        from graph_hdc.graph import data_list_from_graph_dicts
        from torch_geometric.loader import DataLoader

        data_list = data_list_from_graph_dicts(graphs)
        batch_data = next(iter(DataLoader(data_list, batch_size=1, shuffle=False)))

        num_nodes = len(graphs[0]['node_indices'])
        node_mask = torch.zeros(num_nodes)
        with torch.no_grad():
            result = hyper_net.forward_masked(batch_data, node_mask)

        np.testing.assert_allclose(
            result['graph_embedding'].cpu().numpy(),
            np.zeros((1, dim)),
            atol=1e-6,
            err_msg="forward_masked with zero mask should give zero embedding",
        )

    def test_partial_mask_differs(self):
        """Masking out a node should change the embedding."""
        dim = 200
        depth = 2
        hyper_net = _make_simple_hyper_net(dim=dim, depth=depth)
        graphs = _make_test_graphs(num_graphs=1, num_node_range=(5, 8))

        from graph_hdc.graph import data_list_from_graph_dicts
        from torch_geometric.loader import DataLoader

        data_list = data_list_from_graph_dicts(graphs)

        num_nodes = len(graphs[0]['node_indices'])
        full_mask = torch.ones(num_nodes)
        partial_mask = torch.ones(num_nodes)
        partial_mask[0] = 0  # mask out node 0

        with torch.no_grad():
            batch1 = next(iter(DataLoader(data_list, batch_size=1, shuffle=False)))
            r_full = hyper_net.forward_masked(batch1, full_mask)
            batch2 = next(iter(DataLoader(data_list, batch_size=1, shuffle=False)))
            r_partial = hyper_net.forward_masked(batch2, partial_mask)

        assert not np.allclose(
            r_full['graph_embedding'].cpu().numpy(),
            r_partial['graph_embedding'].cpu().numpy(),
            atol=1e-4,
        ), "Masking a node should change the embedding"


    def test_masked_node_zero_at_all_layers(self):
        """A masked node must have a zero embedding at every layer."""
        dim = 200
        depth = 3
        hyper_net = _make_simple_hyper_net(dim=dim, depth=depth)
        graphs = _make_test_graphs(num_graphs=1, num_node_range=(5, 8))
        graph = graphs[0]

        from graph_hdc.graph import data_list_from_graph_dicts
        from torch_geometric.loader import DataLoader

        data_list = data_list_from_graph_dicts([graph])
        batch_data = next(iter(DataLoader(data_list, batch_size=1, shuffle=False)))

        num_nodes = len(graph['node_indices'])
        masked_idx = 1
        node_mask = torch.ones(num_nodes)
        node_mask[masked_idx] = 0.0

        with torch.no_grad():
            result = hyper_net.forward_masked(
                batch_data, node_mask, return_node_hv_stack=True,
            )

        stack = result['node_hv_stack'].cpu().numpy()
        for layer in range(depth + 1):
            np.testing.assert_allclose(
                stack[layer, masked_idx],
                np.zeros(dim),
                atol=1e-7,
                err_msg=f"Masked node {masked_idx} is not zero at layer {layer}",
            )

    def test_unmasked_nodes_nonzero(self):
        """Unmasked nodes should still have non-zero embeddings."""
        dim = 200
        depth = 2
        hyper_net = _make_simple_hyper_net(dim=dim, depth=depth)
        graphs = _make_test_graphs(num_graphs=1, num_node_range=(5, 8))
        graph = graphs[0]

        from graph_hdc.graph import data_list_from_graph_dicts
        from torch_geometric.loader import DataLoader

        data_list = data_list_from_graph_dicts([graph])
        batch_data = next(iter(DataLoader(data_list, batch_size=1, shuffle=False)))

        num_nodes = len(graph['node_indices'])
        node_mask = torch.ones(num_nodes)
        node_mask[0] = 0.0

        with torch.no_grad():
            result = hyper_net.forward_masked(
                batch_data, node_mask, return_node_hv_stack=True,
            )

        stack = result['node_hv_stack'].cpu().numpy()
        # At least one unmasked node should be non-zero at every layer
        for layer in range(depth + 1):
            unmasked = stack[layer, 1:]  # all nodes except node 0
            assert not np.allclose(unmasked, 0, atol=1e-6), \
                f"All unmasked nodes are zero at layer {layer}"


class TestFrozenNorms:

    def test_full_mask_matches_forward(self):
        """Frozen norms with all-ones mask should match regular forward."""
        dim = 200
        depth = 2
        hyper_net = _make_simple_hyper_net(dim=dim, depth=depth)
        graphs = _make_test_graphs(num_graphs=2, num_node_range=(4, 7))

        results_normal = hyper_net.forward_graphs(graphs)

        from graph_hdc.graph import data_list_from_graph_dicts
        from torch_geometric.loader import DataLoader

        data_list = data_list_from_graph_dicts(graphs)
        loader = DataLoader(data_list, batch_size=1, shuffle=False)

        for i, batch_data in enumerate(loader):
            num_nodes = len(graphs[i]['node_indices'])
            node_mask = torch.ones(num_nodes)
            with torch.no_grad():
                full_result = hyper_net.forward_full_with_norms(batch_data.clone())
                frozen_result = hyper_net.forward_masked_frozen_norms(
                    batch_data.clone(), node_mask,
                    mask_layer=0, frozen_norms=full_result['norms'],
                )
            np.testing.assert_allclose(
                frozen_result['graph_embedding'].cpu().numpy(),
                results_normal[i]['graph_embedding'].reshape(1, -1),
                atol=1e-4,
                err_msg="frozen norms with full mask differs from forward",
            )

    def test_frozen_norms_differ_from_recomputed(self):
        """With a masked node, frozen norms should differ from recomputed norms."""
        dim = 200
        depth = 2
        hyper_net = _make_simple_hyper_net(dim=dim, depth=depth)
        graphs = _make_test_graphs(num_graphs=1, num_node_range=(5, 8))

        from graph_hdc.graph import data_list_from_graph_dicts
        from torch_geometric.loader import DataLoader

        data_list = data_list_from_graph_dicts(graphs)
        batch_data = next(iter(DataLoader(data_list, batch_size=1, shuffle=False)))

        num_nodes = len(graphs[0]['node_indices'])
        node_mask = torch.ones(num_nodes)
        node_mask[0] = 0.0

        with torch.no_grad():
            full_result = hyper_net.forward_full_with_norms(batch_data.clone())

            frozen = hyper_net.forward_masked_frozen_norms(
                batch_data.clone(), node_mask,
                mask_layer=1, frozen_norms=full_result['norms'],
            )
            recomputed = hyper_net.forward_masked_at_layer(
                batch_data.clone(), node_mask, mask_layer=1,
            )

        # Both should produce valid embeddings but they should differ
        # (frozen norms prevent the normalization-induced rotation)
        emb_frozen = frozen['graph_embedding'].cpu().numpy()
        emb_recomputed = recomputed['graph_embedding'].cpu().numpy()

        assert not np.allclose(emb_frozen, emb_recomputed, atol=1e-4), \
            "Frozen and recomputed norms should produce different embeddings"


class TestCausalCoalitionPredict:

    def test_full_coalition_causal(self):
        """Full mask in causal mode should match normal forward prediction."""
        dim = 200
        depth = 2
        hyper_net = _make_simple_hyper_net(dim=dim, depth=depth)
        graphs = _make_test_graphs(num_graphs=1, num_node_range=(4, 7))
        graph = graphs[0]

        results = hyper_net.forward_graphs(graphs)
        full_embedding = results[0]['graph_embedding']

        model = _train_linear_model(dim=dim)
        predict_fn = make_causal_coalition_predict_fn(hyper_net, graph, model)

        num_nodes = len(graph['node_indices'])
        full_mask = np.ones((1, num_nodes))

        pred_causal = predict_fn(full_mask)
        pred_direct = model.predict(full_embedding.reshape(1, -1))

        np.testing.assert_allclose(pred_causal, pred_direct, atol=1e-3)

    def test_empty_coalition_causal(self):
        """Empty mask in causal mode should predict from zero embedding."""
        dim = 200
        depth = 2
        hyper_net = _make_simple_hyper_net(dim=dim, depth=depth)
        graphs = _make_test_graphs(num_graphs=1, num_node_range=(4, 7))
        graph = graphs[0]

        model = _train_linear_model(dim=dim)
        predict_fn = make_causal_coalition_predict_fn(hyper_net, graph, model)

        num_nodes = len(graph['node_indices'])
        empty_mask = np.zeros((1, num_nodes))

        pred_causal = predict_fn(empty_mask)
        pred_zero = model.predict(np.zeros((1, dim)))

        np.testing.assert_allclose(pred_causal, pred_zero, atol=1e-3)


class TestPlayerLabels:

    def test_label_count(self):
        """Correct number of labels for given nodes and layers."""
        node_atoms = np.array([6, 8, 7])
        labels = get_player_labels(node_atoms, 3)
        assert len(labels) == 9  # 3 atoms * 3 layers

    def test_label_format_without_encoder(self):
        """Labels use raw atomic numbers when no encoder is provided."""
        node_atoms = np.array([6, 8])
        labels = get_player_labels(node_atoms, 2)
        assert labels[0] == '6_0 (k=0)'
        assert labels[1] == '6_0 (k=1)'
        assert labels[2] == '8_1 (k=0)'
        assert labels[3] == '8_1 (k=1)'
