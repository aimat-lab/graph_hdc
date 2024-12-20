import torch
from torch_geometric.data import Data


def data_from_graph_dict(graph: dict) -> Data:
    """
    Given a ``graph`` dict representation of a graph, returns a torch_geometric Data object 
    to represent the graph.
    
    :param graph: A graph dict.
    
    :returns: A Data object.
    """
    data = Data(
        x=torch.tensor(graph['node_attributes'], dtype=torch.float),
        edge_index=torch.tensor(graph['edge_indices'].T, dtype=torch.long),
        edge_attr=torch.tensor(graph['edge_attributes'], dtype=torch.float),
    )
    
    if 'graph_labels' in graph:
        data.y = torch.tensor(graph['graph_labels'], dtype=torch.float)
    
    if 'edge_weights' in graph:
        data.edge_weight = torch.tensor(graph['edge_weights'], dtype=torch.float)
    
    for key, value in graph.items():
        if key not in ['node_attributes', 'edge_indices', 'edge_attributes', 'graph_labels', 'edge_weights']:
            setattr(data, key, torch.tensor(value, dtype=torch.float))
    
    return data


def data_list_from_graph_dicts(graphs: list[dict]) -> list[Data]:
    """
    Given a list ``graphs`` of graph dicts, returns a list of torch_geometric Data objects
    to represent the graphs.
    
    :param graphs: A list of graph dicts.
    
    :returns: A list of Data objects.
    """
    data_list = [data_from_graph_dict(graph) for graph in graphs]
    return data_list