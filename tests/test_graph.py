from typing import List

import torch
import numpy as np
from rich.pretty import pprint
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from graph_hdc.testing import generate_random_graphs
from graph_hdc.utils import CategoricalIntegerEncoder
from graph_hdc.graph import data_from_graph_dict
from graph_hdc.graph import data_list_from_graph_dicts
from graph_hdc.graph import constraints_order_zero_from_graph_dict
from graph_hdc.graph import constraints_order_one_from_graph_dict


def test_data_from_graph_dict():
    """
    data_from_graph should create a Data object from a graph dict.
    """
    graph: dict = generate_random_graphs(1)[0]
    data: Data = data_from_graph_dict(graph)
    assert isinstance(data, Data)
    assert isinstance(data.x, torch.Tensor)
    assert isinstance(data.edge_index, torch.Tensor)
    assert isinstance(data.edge_attr, torch.Tensor)
    

def test_data_list_from_graphs():
    """
    data_list_from_graphs should create a list of Data objects from a list of graph dicts
    """
    graphs: List[dict] = generate_random_graphs(10)
    data_list: List[Data] = data_list_from_graph_dicts(graphs)
    
    assert isinstance(data_list, list)
    
    # It should also work that to accumuilate the individual data objects into a data loader
    loader = DataLoader(data_list, batch_size=5, shuffle=False)
    for batch in loader:
        assert isinstance(batch, Data)
        assert isinstance(batch.batch, torch.Tensor)
        assert torch.max(batch.batch) == 4
        
    
def test_constraints_order_zero_from_graph_dict_basically_works():
    """
    The function "constraints_order_zero_from_graph_dict" should take a graph dict as an input and 
    construct the true order zero constraint list with the same format as it would be returned by the 
    decode function of a hypernet.
    """
    graph = {
        'node_indices': np.array([0, 1, 2, 3], dtype=int),
        'node_attributes': np.array([[0], [0], [1], [2]], dtype=float),
        'edge_indices': np.array([[0, 1], [1, 2], [2, 3]], dtype=int),
        'edge_attributes': np.array([[1], [1], [1]], dtype=float),
    }
    node_encoder_map = {
        'node_attributes': CategoricalIntegerEncoder(dim=1000, num_categories=4),
    }
    
    # The list of "order zero" constraints essentially contains information about what kinds of nodes 
    # exist in the graph.
    constraints_order_zero = constraints_order_zero_from_graph_dict(
        graph=graph, 
        node_encoder_map=node_encoder_map
    )
    target = [
        {'src': {'node_attributes': 0}},
        {'src': {'node_attributes': 0}},
        {'src': {'node_attributes': 1}},
        {'src': {'node_attributes': 2}},
    ]
    pprint(constraints_order_zero)
    assert constraints_order_zero == target
    
    
def test_constraints_order_one_from_graph_dict_basically_works():
    """
    The function "constraints_order_one_from_graph_dict" should take a graph dict as an input and
    construct the true order one constraint list with the same format as it would be returned by the
    decode function of a hypernet.
    """
    graph = {
        'node_indices': np.array([0, 1, 2, 3], dtype=int),
        'node_attributes': np.array([[0], [0], [1], [2]], dtype=float),
        'edge_indices': np.array([[0, 1], [1, 2], [2, 3]], dtype=int),
        'edge_attributes': np.array([[1], [1], [1]], dtype=float),
    }
    node_encoder_map = {
        'node_attributes': CategoricalIntegerEncoder(dim=1000, num_categories=4),
    }
    
    # The list of "order one" constraints essentially contains information about what kinds of edges
    # exist in the graph (with source and destination nodes).
    constraints_order_one = constraints_order_one_from_graph_dict(
        graph=graph, 
        node_encoder_map=node_encoder_map
    )
    target = [
        {'src': {'node_attributes': 0}, 'dst': {'node_attributes': 0}},
        {'src': {'node_attributes': 0}, 'dst': {'node_attributes': 1}},
        {'src': {'node_attributes': 1}, 'dst': {'node_attributes': 2}},
    ]
    pprint(constraints_order_one)
    assert constraints_order_one == target