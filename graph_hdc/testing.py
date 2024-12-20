import numpy as np
from typing import Tuple, List


def generate_random_graphs(num_graphs: int,
                           num_node_range: Tuple[int, int] = (10, 20),
                           num_node_features: int = 10,
                           num_edge_features: int = 5,
                           num_graph_labels: int = 2,
                           ) -> List[dict]:
    """
    Randomly generate a list of graphs for testing.
    
    :returns: A list of graph dicts.
    """
    graphs: List[dict] = []
    
    for _ in range(num_graphs):
        node_indices = np.arange(np.random.randint(*num_node_range))
        edge_indices = np.array(
            [(i, (i + 1) % len(node_indices)) for i in range(len(node_indices))] +
            [(i, (i + 2) % len(node_indices)) for i in range(len(node_indices))]    
        , dtype=int)
        graph = {
            'node_indices': node_indices,
            'node_attributes': np.random.rand(len(node_indices), num_node_features),
            'edge_indices': edge_indices,
            'edge_attributes': np.random.rand(len(edge_indices), num_edge_features),
            'edge_weights': np.random.uniform(0, 1, size=(len(edge_indices), 1)),
            'graph_labels': np.random.rand(1, num_graph_labels),
        }
        graphs.append(graph)
        
    return graphs