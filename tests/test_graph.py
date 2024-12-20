from typing import List

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from graph_hdc.testing import generate_random_graphs
from graph_hdc.graph import data_from_graph_dict
from graph_hdc.graph import data_list_from_graph_dicts


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