import numpy as np
import pytest
import torch
from sklearn.linear_model import LinearRegression

from graph_hdc.models import HyperNet, CategoricalIntegerEncoder
from graph_hdc.testing import generate_random_graphs
from graph_hdc.explain import LeaveOneOutExplainer, LayerwiseExplainer, ShapExplainer, MyersonExplainer


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


def _train_linear_model(dim=200, n_samples=50):
    """Train a simple linear regression on random data."""
    model = LinearRegression()
    X = np.random.randn(n_samples, dim)
    y = np.random.randn(n_samples)
    model.fit(X, y)
    return model


# -- Tests --

class TestLeaveOneOutExplainer:

    def test_output_shape(self):
        """Returns one attribution per node."""
        dim = 200
        hyper_net = _make_simple_hyper_net(dim=dim)
        graphs = _make_test_graphs(num_graphs=1, num_node_range=(5, 8))
        graph = graphs[0]
        model = _train_linear_model(dim=dim)

        explainer = LeaveOneOutExplainer(hyper_net, model.predict)
        attr = explainer.explain(graph)

        num_nodes = len(graph['node_indices'])
        assert attr.shape == (num_nodes,)

    def test_attributions_sum_nonzero(self):
        """Attributions should generally be nonzero (model is non-trivial)."""
        dim = 200
        hyper_net = _make_simple_hyper_net(dim=dim)
        graphs = _make_test_graphs(num_graphs=1, num_node_range=(5, 8))
        graph = graphs[0]
        model = _train_linear_model(dim=dim)

        explainer = LeaveOneOutExplainer(hyper_net, model.predict)
        attr = explainer.explain(graph)

        assert not np.allclose(attr, 0, atol=1e-6), \
            "All attributions are zero — model or graph is degenerate"

    def test_single_node_graph(self):
        """Edge case: graph with one node."""
        dim = 200
        hyper_net = _make_simple_hyper_net(dim=dim, depth=1)
        graphs = _make_test_graphs(num_graphs=1, num_node_range=(1, 2))
        graph = graphs[0]
        model = _train_linear_model(dim=dim)

        explainer = LeaveOneOutExplainer(hyper_net, model.predict)
        attr = explainer.explain(graph)

        assert attr.shape == (1,)

    def test_deterministic(self):
        """Running explain twice gives the same result."""
        dim = 200
        hyper_net = _make_simple_hyper_net(dim=dim)
        graphs = _make_test_graphs(num_graphs=1, num_node_range=(5, 8))
        graph = graphs[0]
        model = _train_linear_model(dim=dim)

        explainer = LeaveOneOutExplainer(hyper_net, model.predict)
        attr1 = explainer.explain(graph)
        attr2 = explainer.explain(graph)

        np.testing.assert_array_equal(attr1, attr2)

    def test_multiple_graphs(self):
        """Can explain different graphs with the same explainer."""
        dim = 200
        hyper_net = _make_simple_hyper_net(dim=dim)
        graphs = _make_test_graphs(num_graphs=3, num_node_range=(4, 8))
        model = _train_linear_model(dim=dim)

        explainer = LeaveOneOutExplainer(hyper_net, model.predict)
        for graph in graphs:
            attr = explainer.explain(graph)
            assert attr.shape == (len(graph['node_indices']),)


class TestLayerwiseExplainer:

    def test_output_shape(self):
        """Returns one attribution per node."""
        dim = 200
        hyper_net = _make_simple_hyper_net(dim=dim)
        graphs = _make_test_graphs(num_graphs=1, num_node_range=(5, 8))
        graph = graphs[0]
        model = _train_linear_model(dim=dim)

        explainer = LayerwiseExplainer(hyper_net, model.predict)
        attr = explainer.explain(graph)

        num_nodes = len(graph['node_indices'])
        assert attr.shape == (num_nodes,)

    def test_attributions_nonzero(self):
        """Attributions should generally be nonzero."""
        dim = 200
        hyper_net = _make_simple_hyper_net(dim=dim)
        graphs = _make_test_graphs(num_graphs=1, num_node_range=(5, 8))
        graph = graphs[0]
        model = _train_linear_model(dim=dim)

        explainer = LayerwiseExplainer(hyper_net, model.predict)
        attr = explainer.explain(graph)

        assert not np.allclose(attr, 0, atol=1e-6), \
            "All attributions are zero"

    def test_deterministic(self):
        """Running explain twice gives the same result."""
        dim = 200
        hyper_net = _make_simple_hyper_net(dim=dim)
        graphs = _make_test_graphs(num_graphs=1, num_node_range=(5, 8))
        graph = graphs[0]
        model = _train_linear_model(dim=dim)

        explainer = LayerwiseExplainer(hyper_net, model.predict)
        attr1 = explainer.explain(graph)
        attr2 = explainer.explain(graph)

        np.testing.assert_array_equal(attr1, attr2)

    def test_custom_decay(self):
        """Different decay values produce different attributions."""
        dim = 200
        hyper_net = _make_simple_hyper_net(dim=dim)
        graphs = _make_test_graphs(num_graphs=1, num_node_range=(5, 8))
        graph = graphs[0]
        model = _train_linear_model(dim=dim)

        attr_05 = LayerwiseExplainer(hyper_net, model.predict, decay=0.5).explain(graph)
        attr_00 = LayerwiseExplainer(hyper_net, model.predict, decay=0.0).explain(graph)

        assert not np.allclose(attr_05, attr_00, atol=1e-6), \
            "Different decay values should produce different attributions"


class TestShapExplainer:

    def test_output_shape(self):
        """Returns one attribution per node."""
        dim = 200
        hyper_net = _make_simple_hyper_net(dim=dim)
        graphs = _make_test_graphs(num_graphs=1, num_node_range=(4, 6))
        graph = graphs[0]
        model = _train_linear_model(dim=dim)

        explainer = ShapExplainer(hyper_net, model.predict, mode='additive')
        attr = explainer.explain(graph)

        num_nodes = len(graph['node_indices'])
        assert attr.shape == (num_nodes,)

    def test_causal_mode(self):
        """Causal mode produces attributions."""
        dim = 200
        hyper_net = _make_simple_hyper_net(dim=dim)
        graphs = _make_test_graphs(num_graphs=1, num_node_range=(4, 6))
        graph = graphs[0]
        model = _train_linear_model(dim=dim)

        explainer = ShapExplainer(hyper_net, model.predict, mode='causal')
        attr = explainer.explain(graph)

        num_nodes = len(graph['node_indices'])
        assert attr.shape == (num_nodes,)
        assert not np.allclose(attr, 0, atol=1e-6)


@pytest.mark.skipif(
    not pytest.importorskip("myerson", reason="myerson not installed"),
    reason="myerson not installed",
)
class TestMyersonExplainer:

    def test_output_shape(self):
        """Returns one attribution per node."""
        dim = 200
        hyper_net = _make_simple_hyper_net(dim=dim)
        graphs = _make_test_graphs(num_graphs=1, num_node_range=(4, 6))
        graph = graphs[0]
        model = _train_linear_model(dim=dim)

        explainer = MyersonExplainer(hyper_net, model.predict, mode='exact')
        attr = explainer.explain(graph)

        num_nodes = len(graph['node_indices'])
        assert attr.shape == (num_nodes,)

    def test_attributions_nonzero(self):
        """Attributions should generally be nonzero."""
        dim = 200
        hyper_net = _make_simple_hyper_net(dim=dim)
        graphs = _make_test_graphs(num_graphs=1, num_node_range=(4, 6))
        graph = graphs[0]
        model = _train_linear_model(dim=dim)

        explainer = MyersonExplainer(hyper_net, model.predict, mode='exact')
        attr = explainer.explain(graph)

        assert not np.allclose(attr, 0, atol=1e-6), \
            "All attributions are zero"

    def test_deterministic(self):
        """Running explain twice in exact mode gives the same result."""
        dim = 200
        hyper_net = _make_simple_hyper_net(dim=dim)
        graphs = _make_test_graphs(num_graphs=1, num_node_range=(4, 6))
        graph = graphs[0]
        model = _train_linear_model(dim=dim)

        explainer = MyersonExplainer(hyper_net, model.predict, mode='exact')
        attr1 = explainer.explain(graph)
        attr2 = explainer.explain(graph)

        np.testing.assert_array_equal(attr1, attr2)

    def test_sampled_mode(self):
        """Sampled mode produces attributions of correct shape."""
        dim = 200
        hyper_net = _make_simple_hyper_net(dim=dim)
        graphs = _make_test_graphs(num_graphs=1, num_node_range=(4, 6))
        graph = graphs[0]
        model = _train_linear_model(dim=dim)

        explainer = MyersonExplainer(
            hyper_net, model.predict,
            mode='sampled', num_samples=500, seed=42,
        )
        attr = explainer.explain(graph)

        num_nodes = len(graph['node_indices'])
        assert attr.shape == (num_nodes,)
        assert not np.allclose(attr, 0, atol=1e-6)

    def test_auto_mode_small_graph(self):
        """Auto mode uses exact computation for small graphs."""
        dim = 200
        hyper_net = _make_simple_hyper_net(dim=dim)
        graphs = _make_test_graphs(num_graphs=1, num_node_range=(4, 6))
        graph = graphs[0]
        model = _train_linear_model(dim=dim)

        explainer = MyersonExplainer(
            hyper_net, model.predict,
            mode='auto', max_exact_nodes=20,
        )
        attr = explainer.explain(graph)

        num_nodes = len(graph['node_indices'])
        assert attr.shape == (num_nodes,)

    def test_additive_mode(self):
        """Additive mode produces valid attributions."""
        dim = 200
        hyper_net = _make_simple_hyper_net(dim=dim)
        graphs = _make_test_graphs(num_graphs=1, num_node_range=(4, 6))
        graph = graphs[0]
        model = _train_linear_model(dim=dim)

        explainer = MyersonExplainer(
            hyper_net, model.predict, mode='exact', norm_mode='additive',
        )
        attr = explainer.explain(graph)

        num_nodes = len(graph['node_indices'])
        assert attr.shape == (num_nodes,)
        assert not np.allclose(attr, 0, atol=1e-6)

    def test_additive_efficiency_axiom(self):
        """Additive mode: attributions sum to v(N) - v(empty)."""
        dim = 200
        hyper_net = _make_simple_hyper_net(dim=dim)
        graphs = _make_test_graphs(num_graphs=1, num_node_range=(4, 6))
        graph = graphs[0]
        model = _train_linear_model(dim=dim)

        explainer = MyersonExplainer(
            hyper_net, model.predict, mode='exact', norm_mode='additive',
        )
        attr = explainer.explain(graph)

        # v(N): sum of all components → full embedding → prediction
        # v(∅): zero embedding → prediction
        # These are computed independently to cross-check
        results = hyper_net.forward_graphs([graph], return_node_hv_stack=True)
        node_hv_stack = results[0]['node_hv_stack']
        components = node_hv_stack.sum(axis=1)  # (num_nodes, hidden_dim)
        full_embedding = components.sum(axis=0, keepdims=True)
        zero_embedding = np.zeros_like(full_embedding)
        full_pred = float(model.predict(full_embedding).flatten()[0])
        empty_pred = float(model.predict(zero_embedding).flatten()[0])

        expected_sum = full_pred - empty_pred
        np.testing.assert_allclose(
            attr.sum(), expected_sum, atol=1e-4,
            err_msg="Efficiency axiom violated: sum(attr) != v(N) - v(empty)",
        )

    def test_causal_frozen_efficiency_axiom(self):
        """Causal frozen mode: attributions sum to v(N) - v(empty)."""
        dim = 200
        hyper_net = _make_simple_hyper_net(dim=dim)
        graphs = _make_test_graphs(num_graphs=1, num_node_range=(4, 6))
        graph = graphs[0]
        model = _train_linear_model(dim=dim)

        explainer = MyersonExplainer(
            hyper_net, model.predict, mode='exact', norm_mode='frozen',
        )
        attr = explainer.explain(graph)

        # Compute v(N) - v(empty) independently
        from graph_hdc.graph import data_list_from_graph_dicts
        from torch_geometric.loader import DataLoader
        data_list = data_list_from_graph_dicts([graph])
        base_data = next(iter(DataLoader(data_list, batch_size=1, shuffle=False)))
        num_nodes = len(graph['node_indices'])

        with torch.no_grad():
            full_result = hyper_net.forward_full_with_norms(base_data.clone())
        frozen_norms = full_result['norms']
        full_pred = float(model.predict(
            full_result['graph_embedding'].cpu().numpy()
        ).flatten()[0])

        empty_mask = torch.zeros(num_nodes, dtype=torch.float32)
        with torch.no_grad():
            empty_result = hyper_net.forward_masked_frozen_norms(
                base_data.clone(), empty_mask,
                mask_layer=0, frozen_norms=frozen_norms,
            )
        empty_pred = float(model.predict(
            empty_result['graph_embedding'].cpu().numpy()
        ).flatten()[0])

        expected_sum = full_pred - empty_pred
        np.testing.assert_allclose(
            attr.sum(), expected_sum, atol=1e-4,
            err_msg="Efficiency axiom violated: sum(attr) != v(N) - v(empty)",
        )

    def test_recomputed_norm_mode(self):
        """Recomputed norm mode produces valid attributions."""
        dim = 200
        hyper_net = _make_simple_hyper_net(dim=dim)
        graphs = _make_test_graphs(num_graphs=1, num_node_range=(4, 6))
        graph = graphs[0]
        model = _train_linear_model(dim=dim)

        explainer = MyersonExplainer(
            hyper_net, model.predict, mode='exact', norm_mode='recomputed',
        )
        attr = explainer.explain(graph)

        num_nodes = len(graph['node_indices'])
        assert attr.shape == (num_nodes,)
        assert not np.allclose(attr, 0, atol=1e-6)

    def test_norm_modes_differ(self):
        """Frozen and recomputed norm modes produce different attributions."""
        dim = 200
        hyper_net = _make_simple_hyper_net(dim=dim)
        graphs = _make_test_graphs(num_graphs=1, num_node_range=(4, 6))
        graph = graphs[0]
        model = _train_linear_model(dim=dim)

        attr_frozen = MyersonExplainer(
            hyper_net, model.predict, mode='exact', norm_mode='frozen',
        ).explain(graph)
        attr_recomputed = MyersonExplainer(
            hyper_net, model.predict, mode='exact', norm_mode='recomputed',
        ).explain(graph)

        assert not np.allclose(attr_frozen, attr_recomputed, atol=1e-6), \
            "Frozen and recomputed norm modes should produce different attributions"
