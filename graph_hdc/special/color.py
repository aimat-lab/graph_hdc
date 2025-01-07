from typing import List, Union, Optional, Any

import torch
import numpy as np
import matplotlib.colors as mcolors
import networkx as nx

from graph_hdc.models import AbstractEncoder
from graph_hdc.models import CategoricalOneHotEncoder
from graph_hdc.models import CategoricalIntegerEncoder


def generate_random_color_nx(num_nodes: int,
                             num_edges: int,
                             colors: list[str] = ['red', 'green', 'blue'],
                             seed: Optional[int] = None,
                             ) -> nx.Graph:
    nx.Graph()
    
    random = np.random.default_rng(seed)
    graph = nx.gnm_random_graph(num_nodes, num_edges, seed=seed)
    
    for node in graph.nodes:
        graph.nodes[node]['color'] = random.choice(colors)
        graph.nodes[node]['degree'] = graph.degree[node]
    
    return graph



class ColorEncoder(AbstractEncoder):
    
    def __init__(self,
                 dim: int,
                 colors: List[Union[str, tuple]],
                 seed: Optional[int] = None,
                 ) -> None:
        AbstractEncoder.__init__(self, dim, seed)
        self.colors: List[Any] = colors
        
        # The given "colors" list could be any definition of a color such as a string or a tuple of 
        # rgb values. We convert all colors to rgb tuples to have a consistent format.
        self.colors_rgb: List[List[int]] = []
        for color in colors:
            if isinstance(color, str):
                color_rgb = mcolors.to_rgb(color)
            elif isinstance(color, tuple) and len(color) == 3:
                color_rgb = color
            else:
                raise ValueError(f"Unsupported color format: {color}")
            
            self.colors_rgb.append(color_rgb)
        
        self.num_categories = len(self.colors)
        
        random = np.random.default_rng(seed)
        self.embeddings: torch.Tensor = torch.tensor(random.normal(
            # This scaling is important to have normalized base vectors
            loc=0.0,
            scale=(1.0 / np.sqrt(dim)), 
            size=(self.num_categories, dim)
        ).astype(np.float32))
        
    def encode(self, value: Any) -> torch.Tensor:
        value_rgb = np.array(value)
        distances = [np.linalg.norm(np.array(color_rgb) - value_rgb) for color_rgb in self.colors_rgb]
        closest_color_index = int(np.argmin(distances))
        return self.embeddings[closest_color_index]
    
    def decode(self, hv: torch.Tensor) -> Any:
        distances = [torch.norm(hv - embedding) for embedding in self.embeddings]
        closest_embedding_index = int(torch.argmin(torch.tensor(distances)))
        return self.colors[closest_embedding_index]
    
    def get_encoder_hv_dict(self):
        return dict(zip(self.colors, self.embeddings))


def graph_dict_from_color_nx(g: nx.Graph,
                             color_attribute: str = 'color',
                             default_color: str = 'gray',
                             ) -> dict:
    node_indices = np.array(sorted(g.nodes), dtype=int)
    node_attributes = np.array([
        mcolors.to_rgb(g.nodes[node_index].get(color_attribute, default_color))
        for node_index in node_indices
    ], dtype=float)
        
    edge_indices = list()
    edge_weight = list()
    for u, v, data in g.edges(data=True):
        edge_indices.append((u, v))
        #edge_indices.append((v, u))
        edge_weight.append([data.get('edge_weight', 10.0)])
        #edge_weight.append([data.get('edge_weight', 1.0)])
    
    edge_indices = np.array(edge_indices, dtype=int)
    edge_weight = np.array(edge_weight, dtype=float)
    edge_attributes = edge_weight
    
    graph = {
        'node_indices': node_indices,
        'node_attributes': node_attributes,
        'edge_indices': edge_indices,
        'edge_attributes': edge_attributes,
        'edge_weights': edge_weight,
    }
    
    # ~ additional color graph specific attributes
    graph['node_color'] = node_attributes
    graph['node_degree'] = np.array([g.nodes[node_index].get('degree', 1) for node_index in node_indices], dtype=int)
    
    return graph



def make_color_node_encoder_map(dim: int, 
                                colors: List[str] = ['red', 'green', 'blue']
                                ) -> dict:
    return {
        'node_color': ColorEncoder(dim, colors),
        'node_degree': CategoricalIntegerEncoder(dim, 20),
    }


