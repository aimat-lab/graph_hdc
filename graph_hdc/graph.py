from typing import Dict, List, Tuple

import torch
from torch_geometric.data import Data

from graph_hdc.utils import AbstractEncoder


def evaluate_constraint(constraints_true: List[Dict[str, dict]],
                        constraints_pred: List[Dict[str, dict]],
                        ) -> Tuple[int, int]:
    """
    
    """
    constraints_true_copy = constraints_true.copy()
    constraints_pred_copy = constraints_pred.copy()
    for constraint in constraints_pred:
        if constraint in constraints_true_copy:
            constraints_true_copy.remove(constraint)
            constraints_pred_copy.remove(constraint)
            
    # false predictions: The number of predicted constraints that are not in the true constraints
    # missed trues: The number of true constraints that were not covered by the predictions 
    false_preds = len(constraints_pred_copy)
    missd_trues = len(constraints_true_copy)

    return false_preds, missd_trues


def constraints_order_zero_from_graph_dict(graph: dict,
                                           node_encoder_map: Dict[str, AbstractEncoder],
                                           ) -> List[Dict[str, dict]]:
    """
    Given a ``graph`` dict representation and a ``node_encoder_map`` dictionary, this function 
    constructs a list of *true* zero order constraints (nodes) which can then be used to compare 
    with a predicted list of zero order constraints, for example.
    
    Example
    
    .. code-block:: python

        graph = {
            'node_indices': [0, 1, 2],
            'node_attributes': [[0], [1], [0]],
            'edge_indices': [[0, 1], [1, 2]],
            'edge_attributes': [[0], [1]],
        }
        node_encoder_map = {
            'label': CategoricalIntegerEncoder(dim=10, num_categories=2),
        }
        
        constraints_order_zero = constraints_order_zero_from_graph_dict(
            graph=graph,
            node_encoder_map=node_encoder_map,
        )
        
        # [
        #     {'src': {'label': 0}},
        #     {'src': {'label': 1}},
        #     {'src': {'label': 0}},
        # ]
    
    :param graph: A graph dict.
    :param node_encoder_map: A dictionary mapping node attribute names to their respective implementations 
        of the AbstractEncoder interface. The returned zero order constraints will define the node attributes 
        with the same names as the keys of this dict.
        
    :returns a list of zero order constraints which is a list of dictionaries with string keys and dict values
        where the dicts contain a single key 'src' and the value is a dictionary of node attribute names and
        their values.
    """
    constraints_order_zero: List[Dict[str, dict]] = []
    for i in graph['node_indices']:
        
        constraint = {'src': {}}
        for name, encoder in node_encoder_map.items():
            value_enc = encoder.encode(graph[name][i])
            value_dec = encoder.decode(value_enc)
            constraint['src'][name] = value_dec
            
        constraints_order_zero.append(constraint)
        
    return constraints_order_zero


def constraints_order_one_from_graph_dict(graph: dict,
                                          node_encoder_map: Dict[str, AbstractEncoder],
                                          ) -> List[Dict[str, dict]]:
    """
    Given a ``graph`` dict representation and a ``node_encoder_map`` dictionary, this function
    constructs a list of *true* first order constraints (edges) which can then be used to compare
    with a predicted list of first order constraints, for example.
    
    Example
    
    .. code-block:: python
    
        graph = {
            'node_indices': [0, 1, 2],
            'node_attributes': [[0], [1], [0]],
            'edge_indices': [[0, 1], [1, 2]],
            'edge_attributes': [[0], [1]],
        }
        
        node_encoder_map = {
            'label': CategoricalIntegerEncoder(dim=10, num_categories=2),
        }
        
        constraints_order_one = constraints_order_one_from_graph_dict(
            graph=graph,
            node_encoder_map=node_encoder_map,
        )
        
        # [
        #     {'src': {'label': 0}, 'dst': {'label': 1}},
        #     {'src': {'label': 1}, 'dst': {'label': 0}},
        # ]
        
    :param graph: A graph dict.
    :param node_encoder_map: A dictionary mapping node attribute names to their respective implementations
        of the AbstractEncoder interface. The returned first order constraints will define the node attributes
        with the same names as the keys of this dict.
        
    :returns a list of first order constraints which is a list of dictionaries with string keys and dict values
        where the dicts contain two keys 'src' and 'dst' and the values are dictionaries of node attribute names
        and their values. Each element represents an edge in the graph.
    """
    
    constraints_order_zero = constraints_order_zero_from_graph_dict(
        graph=graph,
        node_encoder_map=node_encoder_map,
    )
    
    constraints_order_one: List[Dict[str, dict]] = []
    for i, j in graph['edge_indices']:
        constraint = {
            'src': constraints_order_zero[i]['src'], 
            'dst': constraints_order_zero[j]['src'],
        }
        constraints_order_one.append(constraint)
        
    return constraints_order_one
        

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
            try:
                setattr(data, key, torch.tensor(value, dtype=torch.float))
            # It can happen that we attach a numpy array full of strings to the graph dict as well in which case 
            # we want to ignore that property because that is not supported by torch.
            except TypeError:
                pass
    
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