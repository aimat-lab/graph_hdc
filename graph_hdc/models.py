import os
import zipfile
import tempfile
from itertools import product
from typing import Dict, Optional, Callable, Any, List, Tuple, Set

import jsonpickle
import numpy as np
import torch
import pytorch_lightning as pl
import torch.optim as optim
from torch.nn.functional import normalize, sigmoid
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import scatter

import graph_hdc.utils
import graph_hdc.binding
from graph_hdc.utils import torch_pairwise_reduce
from graph_hdc.utils import shallow_dict_equal
from graph_hdc.utils import HypervectorCombinations
from graph_hdc.utils import AbstractEncoder
from graph_hdc.utils import CategoricalOneHotEncoder
from graph_hdc.utils import CategoricalIntegerEncoder
from graph_hdc.graph import data_list_from_graph_dicts
from graph_hdc.functions import resolve_function, desolve_function


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
        batch_size = torch.max(data.batch).detach().numpy() + 1
        
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
        
    # -- encoding
    # These methods handle the encoding of the graph structures into the graph embedding vector
    
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
            #property_hv = torch.stack([encoder.encode(tens) for tens in property_value])
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
    
    def forward(self, 
                data: Data,
                bidirectional: bool = True,
                ) -> dict:
        """
        Performs a forward pass on the given PyG ``data`` object which represents a batch of graphs. Primarily 
        this method will encode the graphs into high-dimensional graph embedding vectors.
        
        :param data: The PyG Data object that represents the batch of graphs.
        
        :returns: A dict with string keys and torch Tensor values. The "graph_embedding" key should contain the
            high-dimensional graph embedding vectors for the input graphs with shape (batch_size, hidden_dim)
        """
        # node_dim: (batch_size * num_nodes)
        node_dim = data.x.size(0)
        
        # ~ mapping node & graph properties as hypervectors
        # The "encoder_properties" method will actually manage the encoding of the node and graph properties of 
        # the graph (as represented by the Data object) into representative 
        # Afterwards, the data object contains the additional properties "data.node_hv" and "data.graph_hv" 
        # which represent the encoded hypervectors for the individual nodes or for the overall graphs respectively.
        data = self.encode_properties(data)
        
        # ~ handling continuous edge weights
        # Optionally it is possible for the input graph structures to also define a "edge_weight" property which 
        # should be a continuous value that represents the weight of the edge. This weight will later be used 
        # to weight/gate the message passing over the corresponding edge during the message-passing steps.
        # Specifically, the values in the "edge_weight" property should be the edge weight LOGITS, which will 
        # later be transformed into a [0, 1] range using the sigmoid function!
        
        if hasattr(data, 'edge_weight') and data.edge_weight is not None:
            edge_weight = data.edge_weight
        else:
            # If the given graphs do not define any edge weights we set the default values to 10 for all edges 
            # because sigmoid(10) ~= 1.0 which will effectively be the same as discrete edges.
            edge_weight = 100 * torch.ones(data.edge_index.shape[1], 1, device=self.device)
            
        # ~ handling edge bi-directionality
        # If the bidirectional flag is given we will duplicate each edge in the input graphs and reverse the 
        # order of node indices such that each node of each edge is always considered as a source and a target 
        # for the message passing operation.
        # Similarly we also duplicate the edge weights such that the same edge weight is used for both edge 
        # "directions".
        
        if bidirectional:
            edge_index = torch.cat([data.edge_index, data.edge_index[[1, 0]]], dim=1)
            edge_weight = torch.cat([edge_weight, edge_weight], dim=0)
        else:
            edge_index = data.edge_index
            edge_weight = edge_weight
            
        # data.edge_index: (2, batch_size * num_edges)
        srcs, dsts = edge_index
    
        # In this data structure we will stack all the intermediate node embeddings for the various message-passing 
        # depths.
        # node_hv_stack: (num_layers + 1, batch_size * num_nodes, hidden_dim)
        node_hv_stack: torch.Tensor = torch.zeros(
            size=(self.depth + 1, node_dim, self.hidden_dim), 
            device=self.device
        )
        node_hv_stack[0] = data.node_hv
        
        # ~ message passing
        for layer_index in range(self.depth):
            # messages are gated with the corresponding edge weights!
            messages = node_hv_stack[layer_index][dsts] * sigmoid(edge_weight)
            aggregated = scatter(messages, srcs, reduce='sum')
            #node_hv_stack[layer_index + 1] = normalize(self.bind_fn(node_hv_stack[0], aggregated))
            node_hv_stack[layer_index + 1] = normalize(self.bind_fn(node_hv_stack[layer_index], aggregated))
        
        # We calculate the final graph-level embedding as the sum of all the node embeddings over all the various 
        # message passing depths and as the sum over all the nodes.
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
    # These methods handle the inverse operation -> The decoding of the graph embedding vectors back into 
    # the original graph structure.
        
    def decode_order_zero(self, 
                          embedding: torch.Tensor
                          ) -> List[dict]:
        """
        Returns information about the kind and number of nodes (order zero information) that were contained in 
        the original graph represented by the given ``embedding`` vector.
        
        **Node Decoding**
        
        The aim of this method is to reconstruct the information about what kinds of nodes existed in the original 
        graph based on the given graph embedding vector ``embedding``. The way in which this works is that for 
        every possible combination of node properties we know the corresponding base hypervector encoding which 
        is stored in the self.node_hv_combinations data structure. Multiplying each of these node hypervectors 
        with the final graph embedding is essentially a projection along that node type's dimension. The magnitude
        of this projection should be proportional to the number of times that node type was present in the original
        graph.
        
        Therefore, we iterate over all the possible node property combinations and calculate the projection of the
        graph embedding along the direction of the node hypervector. If the magnitude of this projection is non-zero
        we can assume that this node type was present in the original graph and we derive the number of times it was
        present from the magnitude of the projection.
        
        :returns: A list of constraints where each constraint is represented by a dictionary with the keys:
            - src: A dictionary that represents the properties of the node as they were originally encoded 
              by the node encoders. The keys in this dict are the same as the names of the node encoders 
              given to the constructor.
            - num: The integer number of how many of these nodes are present in the graph.
        """
        
        # In this list we'll store the final decoded constraints about which kinds of nodes are present in the 
        # graph. Each constraints is represented as a dictionary which contains information about which kind of 
        # node is present (as a combination of node properties) and how many of these nodes are present.
        constraints_order_zero: List[Dict[str, dict]] = []
        for comb_dict, hv in self.node_hv_combinations:
            
            # By multiplying the embedding with the specific node hypervector we essentially calculate the
            # projection of the graph along the direction of the node. This projection should be proportional 
            # to the number of times that a node of that specific type was included in the original graph.
            value = torch.dot(hv, embedding.squeeze()).detach().item()
            if np.round(value) > 0:
                result_dict = {
                    'src': comb_dict.copy(), 
                    'num': round(value)
                }
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
        """
        Returns information about the kind and number of edges (order one information) that were contained in the
        original graph represented by the given ``embedding`` vector.
        
        **Edge Decoding**
        
        The aim of this method is to reconstruct the first order information about what kinds of edges existed in 
        the original graph based on the given graph embedding vector ``embedding``. The way in which this works is
        that we already get the zero oder constraints (==informations about which nodes are present) passed as an 
        argument. Based on that we construct all possible combinations of node pairs (==edges) and calculate the 
        corresponding binding of the hypervector representations. Then we can multiply each of these edge hypervectors
        with the final graph embedding to get a projection along that edge type's dimension. The magnitude of this
        projection should be proportional to the number of times that edge type was present in the original graph 
        (except for a correction factor).
        
        Therefore, we iterate over all the possible node pairs and calculate the projection of the graph embedding
        along the direction of the edge hypervector. If the magnitude of this projection is non-zero we can assume
        that this edge type was present in the original graph and we derive the number of times it was present from
        the magnitude of the projection.
        
        :param embedding: The high-dimensional graph embedding vector that represents the graph.
        :param constraints_order_zero: The list of constraints that represent the zero order information about the
            nodes that were present in the original graph.
        :param correction_factor_map: A dictionary that contains correction factors for the number of shared core
            properties between the nodes that constitute the edge. The keys are the number of shared core properties
            and the values are the correction factors that should be applied to the calculated edge count.
            
        :returns: A list of constraints where each constraint is represented by a dictionary with the keys:
            - src: A dictionary that represents the properties of the source node as they were originally encoded
                by the node encoders. The keys in this dict are the same as the names of the node encoders given to
                the constructor.
            - dst: A dictionary that represents the properties of the destination node as they were originally
                encoded by the node encoders. The keys in this dict are the same as the names of the node encoders
                given to the constructor.
            - num: The integer number of how many of these edges are present in the graph.
        """
        constraints_order_one: List[Dict[str, dict]] = []
        # The "product" here will give us all the possible combinations between the zero order constraints
        # (==nodes) thus giving us all the possible edges that could have existed in the original graph.
        for const_i, const_j in product(constraints_order_zero, repeat=2):
            
            # Here we calculate how many core properties are shared between the two nodes that 
            # constitute the edge. So in the simple example of a node being identified by a color 
            # and the node degree, this number would be 1 if the nodes either share the same degree 
            # or the same color and would be 2 if the nodes share both the same color and the same.
            # etc.
            num_shared: int = len(set(const_i['src'].keys()) & set(const_j['src'].keys()))
            
            # We can query the corresponding hypervector representations for the two nodes that
            # constitute the edge from the node_hv_combinations data structure.
            hv_i = self.node_hv_combinations.get(const_i['src'])
            hv_j = self.node_hv_combinations.get(const_j['src'])

            hv = self.bind_fn(hv_i, hv_j)
            value = (torch.dot(hv, embedding.squeeze())).detach().item()
            value *= correction_factor_map[num_shared]
            
            if np.round(value) > 0:
                result_dict = {
                    'src': const_i['src'].copy(), 
                    'dst': const_j['src'].copy(),
                    'num': round(value)
                }
                constraints_order_one.append(result_dict)
                
        return constraints_order_one
    
    # -- saving and loading
    # methods that handle the storage of the HyperNet instance to and from a file.
    
    def save_to_path(self, path: str) -> None:
        """
        Saves the current state of the current instance to the given ``path`` using jsonpickle.
        
        :param path: The absolute path to the file where the instance should be saved. Will overwrite
            if the file already exists.
        
        :returns: None
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
    
    def load_from_path(self, path: str) -> None:
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
        
    @classmethod
    def load(cls, path: str):
        """
        Given the absolute string ``path`` to an existing file, this will load the saved state that 
        has been saved using the "save_to_path" method. This will overwrite the values of the 
        current object instance.
        
        :param path: The absolute path to the file where a HyperNet instance has previously been 
            saved to.
            
        :returns: A new instance of the HyperNet class with the loaded state.
        """
        instance = cls(
            hidden_dim=100,
            node_encoder_map={
                'node': CategoricalOneHotEncoder(dim=100, num_categories=2)
            }
        )
        instance.load_from_path(path)
        return instance
    
    def possible_graph_from_constraints(self,
                                        zero_order_constraints: List[dict],
                                        first_order_constraints: List[dict],
                                        ) -> Tuple[dict, list]:
        
        # ~ Build node information from constraints list 
        # This data structure will contain a unique integer node index as the key and the value will 
        # be the dictionary which contains the node properties that were originally decoded.
        index_node_map: Dict[int, dict] = {}
        index: int = 0
        for nc in zero_order_constraints:
            num = nc['num']
            for _ in range(num):
                index_node_map[index] = nc['src']
                index += 1
        
        # ~ Build edge information from constraints list
        edge_indices: Set[Tuple[int, int]] = set()
        for ec in first_order_constraints:
            src = ec['src']
            dst = ec['dst']
            
            # Now we need to find all the node indices which match the description of the edge source
            # and destination. This is done by iterating over the index_node_map and checking if the
            # node properties match the source and destination properties of the edge.
            # For each matching pair, we insert an edge into the edge_indices list.
            for i, node_i in index_node_map.items():
                if shallow_dict_equal(node_i, src):
                    for j, node_j in index_node_map.items():
                        if shallow_dict_equal(node_j, dst) and i != j:
                            hi = max(i, j)
                            lo = min(i, j)
                            edge_indices.add((hi, lo))
                            
        return index_node_map, list(edge_indices)

    def reconstruct(self, 
                    graph_hv: torch.Tensor, 
                    num_iterations: int = 100, 
                    learning_rate: float = 0.1,
                    batch_size: int = 20,
                    low: float = 0.2,
                    high: float = 0.8) -> dict:
        """
        Reconstructs a graph dict representation from the given graph hypervector by first decoding
        the order constraints for nodes and edges to build an initial guess and then refining the 
        structure using gradient descent optimization.
        
        Now, instead of optimizing a single candidate, a whole batch of candidates are optimized.
        The edge weights are randomly initialized between low and high and, after optimization,
        are discretized. The candidate with the best similarity to graph_hv is selected.
        """
        # ~ Decode node and edge constraints
        node_constraints = self.decode_order_zero(graph_hv)
        edge_constraints = self.decode_order_one(graph_hv, node_constraints)
        
        node_keys = list(node_constraints[0]['src'].keys())
        
        # Given the node and edge constraints, this method will assemble a first guess of the graph 
        # structure by inserting all of the nodes that were defined by the node constraints and inserting 
        # all possible edges that match any of the given edge constraints.
        index_node_map, edge_indices = self.possible_graph_from_constraints(
            node_constraints, 
            edge_constraints
        )
        
        data = Data()
        for key in node_keys:
            tens = torch.tensor([self.node_encoder_map[key].normalize(node[key])
                                 for node in index_node_map.values()])
            setattr(data, key, tens)
        
        data.edge_index = torch.tensor(list(edge_indices), dtype=torch.long).t()
        data.batch = torch.tensor([0] * len(index_node_map), dtype=torch.long)
        data.x = torch.zeros(len(index_node_map), self.hidden_dim)
        data = self.encode_properties(data)
        
        data_list: List[Data] = []
        for _ in range(batch_size):
            data = data.clone()
            data.edge_weight = torch.tensor(np.random.uniform(low=low, high=high, size=(data.edge_index.size(1), 1)))
            data_list.append(data)
            
        batch = Batch.from_data_list(data_list)
        batch.edge_weight.requires_grad = True
        
        num_nodes = batch.edge_index.max().item() + 1
        
        optimizer = torch.optim.Adam([batch.edge_weight], lr=learning_rate)
        #optimizer = torch.optim.LBFGS([batch.edge_weight], lr=learning_rate)
        
        # Optimization loop over candidate batch
        for _ in range(num_iterations):
            
            optimizer.zero_grad()
            result = self.forward(batch)
            embedding = result['graph_embedding']  # shape (candidate_batch_size, hidden_dim)
            # Compute mean squared error loss for each candidate (compare each to graph_hv)
            losses = torch.square((embedding - graph_hv.expand_as(embedding))).mean(dim=1)
            loss = losses.mean()
            
            if 'node_degree' in node_keys or 'node_degrees' in node_keys:
                
                true_degree = batch.node_degree if hasattr(batch, 'node_degree') else batch.node_degrees
                
                _edge_weight = torch.sigmoid(2 * batch.edge_weight)
                print(_edge_weight)
                _edges_src = scatter(torch.ones_like(_edge_weight), batch.edge_index[0], dim_size=num_nodes, reduce='sum')
                _edges_dst = scatter(torch.ones_like(_edge_weight), batch.edge_index[1], dim_size=num_nodes, reduce='sum')
                _num_edges = _edges_src + _edges_dst
                
                #_edge_weight = torch.where(_edge_weight > 0.5, _edge_weight, _edge_weight * 0.001)
                #_edge_weight = torch.where(_edge_weight > 0.2, torch.ones_like(_edge_weight), torch.zeros_like(_edge_weight))
                scatter_src = scatter(_edge_weight, batch.edge_index[0], dim_size=num_nodes, reduce='sum')
                scatter_dst = scatter(_edge_weight, batch.edge_index[1], dim_size=num_nodes, reduce='sum')
                # Calculate the actual node degree by summing over the edge weights of all the in and out going edges of a node
                node_degree = scatter_src + scatter_dst
                
                # Calculate the loss between the actual node degree and the expected node degree
                degree_loss = torch.abs(node_degree - true_degree).mean()
                
                # print('node_degree', node_degree)
                # print('true_degree', true_degree)
                # print('_edge_weight', _edge_weight)
                
                # Add the degree loss to the total loss
                loss += 1e-2 * degree_loss
                
                # Entropy loss to promote edge weights to be either 0 or 1
                sparsity_loss = torch.abs(_edge_weight).mean()
                #loss += 1e-2 * sparsity_loss
                        
            loss.backward()
            optimizer.step()
            
            print(loss.item())
            
        # discretizing the still continuous edge weights and constructing a new "edge_index"
        # connectivity structure based only on the edges that have a weight > 0.5
        print(batch.edge_weight)
        batch.edge_weight = (batch.edge_weight >= 0).float()
        result = self.forward(batch)
        embedding = result['graph_embedding']  # shape (candidate_batch_size, hidden_dim)
        losses = torch.abs((embedding - graph_hv.expand_as(embedding))).mean(dim=1)
            
        # We get the index of the best candidate according to the loss of the final epoch
        losses = losses.detach().cpu().numpy()
        index_best = np.argmin(losses)
        data_best = batch.to_data_list()[index_best]
        print('final edge weight', data_best.edge_weight)
        num_nodes = data_best.edge_index.max().item() + 1
        scatter_src = scatter(data_best.edge_weight, data_best.edge_index[0], dim_size=num_nodes, reduce='sum')
        scatter_dst = scatter(data_best.edge_weight, data_best.edge_index[1], dim_size=num_nodes, reduce='sum')
        print('final degrees', scatter_src + scatter_dst)
        
        # select the edges that have a weight > 0.5
        edge_weight = data_best.edge_weight
        edge_index = data_best.edge_index[:, edge_weight.flatten() > 0.5]
        print('edge index', edge_index.detach().cpu().numpy().T)
        
        # Prepare final graph dict representation using best candidate's discrete edge weights
        graph_dict = {
            'node_indices': np.array(list(index_node_map.keys()), dtype=int),
            'node_attributes': data_best.x.detach().cpu().numpy(),  # placeholder attributes
            'edge_indices': edge_index.detach().cpu().numpy().T,
            'edge_attributes': edge_weight,
        }
        for key in node_keys:
            graph_dict[key] = [node[key] for node in index_node_map.values()]
        
        return graph_dict