import os
import zipfile
import tempfile
from itertools import product
from typing import Dict, Optional, Callable, Any, List

import jsonpickle
import numpy as np
import torch
import pytorch_lightning as pl
from torch.nn.functional import normalize, sigmoid
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import scatter

import graph_hdc.utils
import graph_hdc.binding
from graph_hdc.utils import torch_pairwise_reduce
from graph_hdc.utils import HypervectorCombinations
from graph_hdc.graph import data_list_from_graph_dicts
from graph_hdc.functions import resolve_function, desolve_function


# === BASIC PROPERTY ENCODERS ===

class AbstractEncoder:
    """
    Abstract base class for the property encoders. An encoder class is used to encode individual properties 
    of graph elements (nodes, edges, etc.) into a high-dimensional hypervector representation.
    
    Specific subclasses should implement the ``encode`` and ``decode`` methods to encode and decode the 
    property to and from a high-dimensional hypervector representation.
    """
    
    def __init__(self,
                 dim: int,
                 seed: Optional[int] = None,
                 ):
        self.dim = dim
        self.seed = seed
    
    # Turns whatever property into a random tensor
    def encode(self, value: Any) -> torch.Tensor:
        """
        This method takes the property ``value`` and encodes it into a high-dimensional hypervector 
        as a torch.Tensor.
        This method should be implemented by the specific subclasses.
        """
        raise NotImplementedError()
    
    # Turns the random tensor back into whatever the original property was
    def decode(self, hv: torch.Tensor) -> Any:
        """
        This method takes the hypervector ``hv`` and decodes it back into the original property.
        This method should be implemented by the specific subclasses.
        """
        raise NotImplementedError()
    
    # Returns a dictionary representation of the encoder mapping
    def get_encoder_hv_dict(self) -> Dict[Any, torch.Tensor]:
        """
        This method should return a dictionary representation of the encoder mapping where the keys 
        are the properties that are being encoded and the values are hypervector representations 
        that are used to represent the corresponding property values.
        """
        raise NotImplementedError()
    
    
    
class CategoricalOneHotEncoder(AbstractEncoder):
    """
    This specific encoder is used to encode categorical properties given as a one-hot encoding vector
    into a high-dimensional hypervector. This is done by creating random continuous base vectors for 
    each of the categories and then selecting the corresponding base vector by the index of the one-hot
    encoding. The decoding is done by calculating the base vector with the smallest distance to the
    the given hypervector and returning the corresponding category index.
    
    :param dim: The dimensionality of the hypervectors.
    :param num_categories: The number of categories that can be encoded.
    :param seed: The random seed to use for the generation of the base vectors. Default is None, but 
        can be set for reproducibility.
    """
    
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
            loc=0.0,
            scale=(1.0 / np.sqrt(dim)), 
            size=(num_categories, dim)
        ).astype(np.float32))
    
    def encode(self, value: Any
               ) -> torch.Tensor:
        
        index = torch.argmax(value)
        return self.embeddings[index]
    
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


class CategoricalIntegerEncoder(AbstractEncoder):
    
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
            loc=0.0,
            scale=(1.0 / np.sqrt(dim)), 
            size=(num_categories, dim),
        ).astype(np.float32))
    
    def encode(self, value: Any) -> torch.Tensor:
        return self.embeddings[int(value)]
    
    def decode(self, 
               hv: torch.Tensor, 
               distance: str ='euclidean',
               ) -> Any:
        
        if distance == 'euclidean':
            distances = np.linalg.norm(self.embeddings - hv.numpy(), axis=1)
            
        elif distance == 'cosine':
            similarities = np.dot(self.embeddings, hv.numpy()) / (np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(hv.numpy()))
            distances = 1 - similarities
        
        else:
            raise ValueError(f"Unsupported distance metric: {distance}")
        
        return np.argmin(distances)
    
    def get_encoder_hv_dict(self) -> Dict[Any, torch.Tensor]:
        return {
            index: hv
            for index, hv in enumerate(self.embeddings)
        }


# === HYPERDIMENSIONAL MESSAGE PASSING NETWORKS ===

class AbstractHyperNet(pl.LightningModule):
        
    def __init__(self, **kwargs):
        pl.LightningModule.__init__(self)
        
    def forward_graphs(self, 
                       graphs: List[dict],
                       batch_size: int = 128,
                       ) -> List[Dict[str, np.ndarray]]:
        """
        Given a list of ``graphs`` this method will run the hypernet "forward" pass on all of the graphs 
        and return a list of dictionaries where each dict represents the result of the forward pass for 
        each of the given graphs.
        
        :param graphs: A list of graph dict representations where each dict contains the information
            about the nodes, edges, and properties of the graph.
        :param batch_size: The batch size to use for the forward pass internally.
        
        :returns: A list of result dictionaries where each dict contains the same string keys as the 
            result of the "forward" method.
        """
        
        # first of all we need to convert the graphs into a format that can be used by the hypernet. 
        # For this task there is the utility function "data_list_from_graph_dicts" which will convert 
        # the list of graph dicts into a list of torch_geometric Data objects.
        data_list: List[Data] = data_list_from_graph_dicts(graphs)
        data_loader = DataLoader(data_list, batch_size=batch_size, shuffle=False)
        
        result_list: List[Dict[str, np.ndarray]] = []
        for data in data_loader:
            
            # The problem here is that the "data" object yielded by the data loader contains multiple 
            # batched graphs but to return the results we would like to disentangle this information 
            # back to the individual graphs.
            result: Dict[str, torch.Tensor] = self.forward(data)
    
            # The "extract_graph_results" method will take the batched results and disentangle them 
            # into a list of dictionaries with the same string keys as the batched results but where 
            # the values are the numpy array representations of the tensors only for the specific graphs.
            results: List[Dict[str, np.ndarray]] = self.extract_graph_results(data, result)
            result_list.extend(results)
            
        return result_list
    
    def extract_graph_results(self,
                              data: Data,
                              graph_results: Dict[str, torch.Tensor],
                              ) -> List[Dict[str, np.ndarray]]:
        """
        Given an input ``data`` object and the ``graph_results`` dict that is returned by the "forward" method 
        of the hyper net, this method will disentangle these *batched* results into a list of individual 
        dictionaries where each dict contains the results of the individual graphs in the batch in the form 
        of numpy arrays.
        
        This disentanglement is done dynamically based on the string key names that can be found in the results 
        dict returned by the "forward" method. The following prefix naming conventions should be used when returning 
        properties as part of the results:
        - "graph_": for properties that are related to the overall graph with a shape of (batch_size, ?)
        - "node_": for properties that are related to the individual nodes with a shape of (batch_size * num_nodes, ?)
        - "edge_": for properties that are related to the individual edges with a shape of (batch_size * num_edges, ?)
        
        :param data: The PyG Data object that represents the batch of graphs.
        :param graph_results: The dictionary that contains the results of the forward pass for the batch of
            graphs.
        
        :returns: A list of dictionaries where each dict contains the results of the individual graphs in
            the batch.
        """
        # The batch size as calculated from the data object 
        batch_size = torch.max(data.batch.detach().numpy()) + 1
        
        # In this list we will store the disentangled results for each of the individual graphs in the batch
        # in the form of a dictionary with the same keys as the batched dict results "graph_results" but 
        # where the values are the numpy array representations of the tensors only for the specific graphs.
        results: List[Dict[str, np.ndarray]] = []
        for index in range(batch_size):
            
            node_mask: torch.Tensor = (data.batch == index)
            edge_mask: torch.Tensor = node_mask[data.edge_index[0]] & node_mask[data.edge_index[1]]
            
            result: Dict[str, np.ndarray] = {}
            for key, tens in graph_results.items():
                
                if key.startswith('graph'):
                    result[key] = tens[index].cpu().detach().numpy()
                    
                elif key.startswith('node'):
                    result[key] = tens[node_mask].cpu().detach().numpy()
                    
                elif key.startswith('edge'):
                    result[key] = tens[edge_mask].cpu().detach().numpy()
                
            results.append(result)
            
        return results
        
    # == To be implemented ==
    
    def forward(self, 
                data: Data,
                ) -> Dict[str, torch.Tensor]:
        """
        This method accepts a PyG Data object which represents a *batch* of graphs and is supposed 
        to implement the forward pass encoding of these graphs into the hyperdimensional vector.
        The method should return a dictionary which contains at least the key "graph_embedding" 
        which should be the torch Tensor representation of the encoded graph embeddings for the 
        various graphs in the batch.
        """
        raise NotImplementedError()
    
    # Replacing the instance attributes with loaded state from a given path
    def load_from_path(self, path: str):
        """
        Given an existing absolute file ``path`` this method should implement the loading of the 
        properties from that file to replace the current properties of the HyperNet object instance
        """
        raise NotImplementedError()

    # Saving the instance attributes to a given path
    def save_to_path(self, path: str):
        """
        Given an absolute file ``path`` this method should implement the saving of the current properties
        of the HyperNet object instance to that file.
        """
        raise NotImplementedError()


class HyperNet(AbstractHyperNet):
    
    def __init__(self,
                 hidden_dim: int = 100,
                 depth: int = 3,
                 node_encoder_map: Dict[str, AbstractEncoder] = {},
                 graph_encoder_map: Dict[str, AbstractEncoder] = {},
                 bind_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = 'circular_convolution_fft',
                 unbind_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = 'circular_correlation_fft',
                 pooling: str = 'sum',
                 seed: Optional[int] = None,
                 device: str = 'cpu',
                 ):
        AbstractHyperNet.__init__(self, device=device)
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.node_encoder_map = node_encoder_map
        self.graph_encoder_map = graph_encoder_map
        self.bind_fn = resolve_function(bind_fn)
        self.unbind_fn = resolve_function(unbind_fn)
        self.pooling = pooling
        self.seed = seed
        
        # ~ computed attributes
        
        # This is a dictionary that will itself store the dictionary representations of the individual node 
        # encoder mappings. So each encoder will be some mapping of (node_property -> hyper_vector) and this 
        # is where we store these mappings.
        self.node_encoder_hv_dicts: Dict[str, dict] = {
            name: encoder.get_encoder_hv_dict()
            for name, encoder in self.node_encoder_map.items()
        }
        
        # HypervectorCombinations is a special custom data structure which is used to construct all possible 
        # binding combinations between individual hypervector representations. In this case, this data structure 
        # can be used to access the bound hypervector representation of any combination of individual node 
        # properties as determined by the node_encoder_map.
        # For example, if there are two properties "node_number" and "node_degree" which are each categorically
        # encoded using fixed hypervectors then this data structure can be used to access the bound hypervector
        # for any combination of possible "node_number" and "node_degree" values.
        self.node_hv_combinations = HypervectorCombinations(
            self.node_encoder_hv_dicts,
            bind_fn=self.bind_fn,
        )
    
    def encode_properties(self, data: Data) -> Data:
        """
        Given the ``data`` instance that represents the input graphs, this function will use the individual 
        property encoders specified in the self.node_encoder_map to encode the various properties of the 
        graph into hypervectors.
        
        :param data: The collated torch_geometric data batch object instance
        
        :returns: The updated Data instance with the additional properties that represent the node and graph
            hypervectors.
        """
        
        # ~ node properties
        # generally, we want to generate a single high-dimensional hypervector representation for each of the 
        # nodes in the graph. However, we might want to individually encode different properties of the nodes
        # using different encoders that are specified as entries in the "node_encoder_map". In this case we 
        # generate all the individual encodings and then use the binding function to bind them into a single 
        # vector to represent the overall node.
        
        node_property_hvs: List[torch.Tensor] = []
        for node_property, encoder in self.node_encoder_map.items():
            # property_value: (batch_size * num_nodes, num_node_features)
            property_value = getattr(data, node_property)
            # property_hv: (batch_size * num_nodes, hidden_dim)
            #property_hv = torch.vmap(encoder.encode, in_dims=0)(property_value)
            property_hv = torch.stack([encoder.encode(tens) for tens in property_value])
            node_property_hvs.append(property_hv)
        
        if node_property_hvs:
            # property_hvs: (num_properties, batch_size * num_nodes, hidden_dim)
            node_property_hvs = torch.stack(node_property_hvs, dim=0)
            
            # The "torch_pairwise_reduce" function will iteratively reduce the given dimension "dim" using the 
            # function "func" which only takes two tensor arguments. This is done by applying each previous 
            # function result as the first argument and the next element in the given tensor dimension as the 
            # second argument until all the elements along that dimension are processed.
            # In this case, this means that we iteratively bind all of the individual property hypervectors 
            # into a single hypervector.
            # property_hv = (batch_size * num_nodes, hidden_dim)
            node_property_hv = torch_pairwise_reduce(node_property_hvs, func=self.bind_fn, dim=0)
            node_property_hv = node_property_hv.squeeze()
            
        else:
            node_property_hv = torch.zeros(data.x.size(0), self.hidden_dim, device=self.device)
            
        # Finally we update the data object with the "node_hv" property so that we can later access this 
        # in the forward pass of the model
        setattr(data, 'node_hv', node_property_hv)
        
        # ~ graph properties
        # There is also the option to encode a high-dimensional hypervector representation containing the 
        # properties of the overall graph (e.g. encoding the size of the graph). Here we also want to support 
        # the possibility to encode multiple properties of the graph using different encoders that are specified
        # in the "graph_encoder_map". In this case we generate all the individual encodings and then use the
        # binding function to bind them into a single vector to represent the overall graph.
        
        graph_property_hvs: List[torch.Tensor] = []
        for graph_property, encoder in self.graph_encoder_map.items():
            # property_value: (batch_size, num_graph_features)
            property_value = getattr(data, graph_property)
            # property_hv: (batch_size, hidden_dim)
            #property_hv = torch.vmap(encoder.encode, in_dims=0)(property_value)
            property_hv = torch.stack([encoder.encode(tens) for tens in property_value])
            graph_property_hvs.append(property_hv)
            
        if graph_property_hvs:
            # graph_property_hvs: (num_properties, batch_size, hidden_dim)
            graph_property_hvs = torch.stack(graph_property_hvs, dim=0)
            
            # graph_property_hv: (batch_size, hidden_dim)
            graph_property_hv = torch_pairwise_reduce(graph_property_hvs, func=self.bind_fn, dim=0)
            graph_property_hv = graph_property_hv.squeeze()
            
        else:
            graph_property_hv = torch.zeros(torch.max(data.batch) + 1, self.hidden_dim, device=self.device)
            
        # Finally we update the data object with the "graph_hv" property so that we can later access this
        # in the forward pass of the model.
        setattr(data, 'graph_hv', graph_property_hv)
            
        return data
    
    def forward(self, data: Data) -> dict:
        
        # The "encoder_properties" method will actually manage the encoding of the node and graph properties of 
        # the graph (as represented by the Data object) into representative 
        # Afterwards, the data object contains the additional properties "data.node_hv" and "data.graph_hv" 
        # which represent the encoded hypervectors for the individual nodes or for the overall graphs respectively.
        data = self.encode_properties(data)
        
        if hasattr(data, 'edge_weight') and data.edge_weight is not None:
            edge_weight = data.edge_weight
        else:
            edge_weight = 10 * torch.ones(data.edge_index.shape[1], 1, device=self.device)
            
        # node_dim: (batch_size * num_nodes)
        node_dim = data.x.size(0)
            
        edge_index = torch.cat([data.edge_index, data.edge_index[[1, 0]]], dim=1)
        edge_weight = torch.cat([edge_weight, edge_weight], dim=0)
        # data.edge_index: (2, batch_size * num_edges)
        # srcs: (batch_size * num_edges)
        # dsts: (batch_size * num_edges)
        srcs, dsts = edge_index
    
        # node_hv_stack: (num_layers + 1, batch_size * num_nodes, hidden_dim)
        node_hv_stack: torch.Tensor = torch.zeros(
            size=(self.depth + 1, node_dim, self.hidden_dim), 
            device=self.device
        )
        node_hv_stack[0] = data.node_hv
        
        for layer_index in range(self.depth):
            messages = node_hv_stack[layer_index][dsts] * sigmoid(edge_weight)
            place_holder = scatter(messages, srcs, reduce='sum')
            #node_hv_stack[layer_index + 1] = self.bind_fn(node_hv_stack[0], place_holder)
            node_hv_stack[layer_index + 1] = normalize(self.bind_fn(node_hv_stack[0], place_holder))
        
        node_hv = node_hv_stack.sum(dim=0)
        readout = scatter(node_hv, data.batch, reduce=self.pooling)
        embedding = readout
        
        return {
            # This the main result of the forward pass which is the individual graph embedding vectors of the 
            # input graphs.
            # graph_embedding: (batch_size, hidden_dim)
            'graph_embedding': embedding,
            # As additional information that might be useful we also pass the stack of the node embeddings across
            # the various convolutional depths.
            # node_hv_stack: (batch_size * num_nodes, num_layers + 1, hidden_dim)
            'node_hv_stack': node_hv_stack.transpose(0, 1),
        }
        
    # -- decoding
        
    def decode_order_zero(self, 
                          embedding: torch.Tensor
                          ) -> List[dict]:
        
        constraints_order_zero: List[Dict[str, dict]] = []
        for comb_dict, hv in self.node_hv_combinations:
            value = torch.dot(hv, embedding.squeeze()).detach().item()
            if np.round(value) > 0:
                result_dict = {'src': comb_dict.copy(), 'num': round(value)}
                constraints_order_zero.append(result_dict)
                
        return constraints_order_zero
    
    def decode_order_one(self,
                         embedding: torch.Tensor,
                         constraints_order_zero: List[dict],
                         correction_factor_map: Dict[int, float] = {
                             0: 1.0,
                             1: 1.0,
                             2: 1.0,
                             3: 1.0,
                         }
                         ) -> List[dict]:
        
        constraints_order_one: List[Dict[str, dict]] = []
        for const_i, const_j in product(constraints_order_zero, repeat=2):
            
            # Here we calculate how many core properties are shared between the two nodes that 
            # constitute the edge. So in the simple example of a node being identified by a color 
            # and the node degree, this number would be 1 if the nodes either share the same degree 
            # or the same color and would be 2 if the nodes share both the same color and the same.
            # etc.
            num_shared: int = len(set(const_i['src'].keys()) & set(const_j['src'].keys()))
            
            hv_i = self.node_hv_combinations.get(const_i['src'])
            hv_j = self.node_hv_combinations.get(const_j['src'])

            hv = self.bind_fn(hv_i, hv_j)
            value = (torch.dot(hv, embedding.squeeze())).detach().item()
            value *= correction_factor_map[num_shared]
            
            if np.round(value) > 0:
                result_dict = {
                    'src': const_i['src'].copy(), 
                    'dst': const_j['src'].copy(),
                    'num': np.round(value)
                }
                constraints_order_one.append(result_dict)
                
        return constraints_order_one
    
    # -- saving and loading
    # methods that 
    
    def save_to_path(self, path: str):
        """
        Saves the current state of the current instance to the given ``path`` using jsonpickle.
        
        :param path: The absolute path to the file where the instance should be saved. Will overwrite
            if the file already exists.
        
        :returns:
        """
        data = {
            'attributes': {
                'hidden_dim': self.hidden_dim,
                'depth': self.depth,
                'seed': self.seed,
                'pooling': self.pooling,
            },
            'node_encoder_map': self.node_encoder_map,
            'graph_encoder_map': self.graph_encoder_map,
            'bind_fn': desolve_function(self.bind_fn),
            'unbind_fn': desolve_function(self.unbind_fn),
        }
        with open(path, mode='w') as file:
            content = jsonpickle.dumps(data)
            file.write(content)
    
    def load_from_path(self, path: str):
        """
        Given the absolute string ``path`` to an existing file, this will load the saved state that 
        has been saved using the "save_to_path" method. This will overwrite the values of the 
        current object instance.
        
        :param path: The absolute path to the file where a HyperNet instance has previously been 
            saved to.
            
        :returns: None
        """
        
        with open(path, mode='r') as file:
            data = jsonpickle.loads(file.read())
            
        for key, value in data['attributes'].items():
            setattr(self, key, value)
            
        self.node_encoder_map = data['node_encoder_map']
        self.graph_encoder_map = data['graph_encoder_map']
        
        self.bind_fn = resolve_function(data['bind_fn'])
        self.unbind_fn = resolve_function(data['unbind_fn'])