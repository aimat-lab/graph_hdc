import os
import zipfile
import tempfile
from typing import Dict, Optional, Callable, Any

import numpy as np
import torch
from torch.nn.functional import normalize
from torch_geometric.utils import scatter


# === BASIC PROPERTY ENCODERS ===

class AbstractEncoder:
    
    def __init__(self,
                 dim: int,
                 seed: Optional[int] = None,
                 ):
        self.dim = dim
        self.seed = seed
    
    # Turns whatever property into a random tensor
    def encode(self, value: Any) -> torch.Tensor:
        raise NotImplementedError()
    
    # Turns the random tensor back into whatever the original property was
    def decode(self, hv: torch.Tensor) -> Any:
        raise NotImplementedError()
    
    
    
class CategoricalOneHotEncoder(AbstractEncoder):
    
    def __init__(self,
                 dim: int,
                 num_categories: int,
                 seed: Optional[int] = None,
                 ):
        AbstractEncoder.__init__(self, dim, seed)
        self.num_categories = num_categories
        
        random = np.random.default_rng(seed)
        self.embeddings: torch.Tensor = torch.tensor(random.normal(
            # This scaling is important to have normalized base vectors
            loc=(1/dim), 
            size=(num_categories, dim)
        ))
    
    def encode(self, value: Any
               ) -> torch.Tensor:
        
        index = np.argmax(value)
        return torch.tensor(self.embeddings[index])
    
    def decode(self, 
               hv: torch.Tensor, 
               distance: str ='euclidean'
               ) -> Any:
        
        if distance == 'euclidean':
            distances = np.linalg.norm(self.embeddings - hv.numpy(), axis=1)
        elif distance == 'cosine':
            similarities = np.dot(self.embeddings, hv.numpy()) / (np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(hv.numpy()))
            distances = 1 - similarities
        else:
            raise ValueError(f"Unsupported distance metric: {distance}")
        
        return np.argmin(distances)


# === HYPERDIMENSIONAL MESSAGE PASSING NETWORKS ===

class AbstractHyperNet:
    
    def __init__(self):
        self.hparams = {}
    
    def save(self, path: str):
        with tempfile.TemporaryDirectory() as path:
            pass

    @classmethod
    def load(cls, path: str):
        raise NotImplementedError()
        
    # == To be implemented ==
    
    def forward(self, data: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def forward_graphs(self, graphs: list[dict]) -> torch.Tensor:
        raise NotImplementedError()



class HyperNet(AbstractHyperNet):
    
    def __init__(self,
                 hidden_dim: int,
                 depth: int,
                 node_encoder_map: Dict[str, AbstractEncoder],
                 bind_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 unbind_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 pooling: str = 'sum',
                 seed: Optional[int] = None,
                 device: str = 'cpu',
                 ):
        AbstractHyperNet.__init__(self)
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.node_encoder_map = node_encoder_map
        self.bind_fn = bind_fn
        self.unbind_fn = unbind_fn
        self.seed = seed
        self.device = device
        
        self.hparams = {
            'hidden_dim': hidden_dim,
            'depth': depth,
            'seed': seed,
            'device': device,
        }
    
    def forward(self, data: dict) -> dict:
    
        if hasattr(data, edge_weight):
            edge_weight = data.edge_weight
        else:
            edge_weight = torch.ones(data.edge_index.shape[1], device=self.device)
            
        srcs, dsts = data.edge_index
    
        node_hv: torch.Tensor
        node_hv_stack: torch.Tensor = torch.zeros((self.depth + 1, self.hidden_dim), device=self.device)
        for layer_index in range(self.depth):
            place_holder = scatter(node_hv_stack[layer_index][dsts] * edge_weight, srcs, reduce='sum')
            bindings = torch.stack((node_hv_stack[0], place_holder))
            node_hv_stack[layer_index + 1] = normalize(self.convolution_fn(bindings))
        
        node_hv = node_hv_stack.sum(dim=0)
        readout = scatter(node_hv, data.batch_idx, reduce=self.pooling)
    
    def forward_graphs(self, graphs: list[dict]) -> list[dict]:
        
        # first of all we need to convert the graphs into a format that can be used by the hypernet. 
        # This primarly means that we need to embedd all the node properties into a tensor.
        pass
    
    def save(self, path: str):
        pass
    
    def load(self, path: str):
        pass