import numpy as np

from graph_hdc.testing import generate_random_graphs


def test_generate_random_graphs():
    """
    The generate_random_graphs function should return a list of mock graph dicts that can be used 
    for testing purposes.
    """
    graphs = generate_random_graphs(
        num_graphs=10,
        num_node_features=5,
        num_edge_features=3
    )
    assert isinstance(graphs, list)
    assert len(graphs) == 10
    for graph in graphs:
        assert isinstance(graph, dict)
        assert isinstance(graph['node_indices'], np.ndarray)
        assert isinstance(graph['node_attributes'], np.ndarray)
        assert graph['node_attributes'].shape[1] == 5
        assert isinstance(graph['edge_indices'], np.ndarray)
        assert isinstance(graph['edge_attributes'], np.ndarray)
        assert graph['edge_attributes'].shape[1] == 3