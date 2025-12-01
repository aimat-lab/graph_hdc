"""
Main HyperNet implementation for hyperdimensional computing on graphs.

This module contains the primary HyperNet class which implements message passing
networks using hyperdimensional vector operations for encoding and decoding graph structures.
"""

from itertools import product
from typing import Dict, Optional, Callable, Any, List, Tuple, Set
from collections import Counter

import jsonpickle
import numpy as np
import torch
import torch.optim as optim
from torch.nn.functional import normalize, sigmoid
from torch_geometric.data import Data, Batch
from torch_geometric.utils import scatter

import graph_hdc.utils
import graph_hdc.binding
from graph_hdc.utils import torch_pairwise_reduce
from graph_hdc.utils import shallow_dict_equal
from graph_hdc.utils import HypervectorCombinations
from graph_hdc.utils import AbstractEncoder
from graph_hdc.utils import CategoricalOneHotEncoder
from graph_hdc.utils import CategoricalIntegerEncoder
from graph_hdc.utils import ContinuousEncoder
from graph_hdc.functions import resolve_function, desolve_function
from graph_hdc.binding import _circular_convolution_fft

from .base import AbstractHyperNet


class HyperNet(AbstractHyperNet):

    def __init__(self,
                 hidden_dim: int = 100,
                 depth: int = 3,
                 node_encoder_map: Dict[str, AbstractEncoder] = {},
                 graph_encoder_map: Dict[str, AbstractEncoder] = {},
                 bind_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = 'circular_convolution_fft',
                 unbind_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = 'circular_correlation_fft',
                 pooling: str = 'sum',
                 normalize_all: bool = False,
                 bidirectional: bool = False,
                 seed: Optional[int] = None,
                 device: str = 'cpu',
                 distance_func: Optional[Callable[[torch.Tensor, torch.Tensor], float]] = None,
                 ):
        AbstractHyperNet.__init__(self, device=device)
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.node_encoder_map = node_encoder_map
        self.graph_encoder_map = graph_encoder_map
        self.bind_fn = resolve_function(bind_fn)
        self.unbind_fn = resolve_function(unbind_fn)
        self.pooling = pooling
        self.normalize_all = normalize_all
        self.bidirectional = bidirectional
        self.seed = seed

        # Store distance function for get_distance method
        # If not provided, the default implementation from AbstractHyperNet will be used
        if distance_func is not None:
            if isinstance(distance_func, str):
                self.distance_func = resolve_function(distance_func)
            else:
                self.distance_func = distance_func
        else:
            # Import here to avoid circular imports
            from graph_hdc.reconstruct import cosine_distance
            self.distance_func = cosine_distance
        
        # ~ computed attributes
        
        # This is a dictionary that will itself store the dictionary representations of the individual node 
        # encoder mappings. So each encoder will be some mapping of (node_property -> hyper_vector) and this 
        # is where we store these mappings.
        self.node_encoder_hv_dicts: Dict[str, dict] = {
            name: encoder.get_encoder_hv_dict()
            for name, encoder in self.node_encoder_map.items()
            if not isinstance(encoder, ContinuousEncoder)
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
        
        self.node_hv_combination_keys: list[dict] = []
        self.node_hv_combination_stack: torch.Tensor = []
        for key, hv in self.node_hv_combinations:
            self.node_hv_combination_keys.append(key)
            self.node_hv_combination_stack.append(hv)
            
        self.node_hv_combination_stack = torch.stack(self.node_hv_combination_stack, dim=0)
        
    # --- encoding ---
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
        
        # --- node properties ---
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
            if hasattr(encoder, 'encode_batch'):
                property_hv = encoder.encode_batch(property_value)
            else: 
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
            #node_property_hv = torch_pairwise_reduce(node_property_hvs, func=self.bind_fn, dim=0)
            #node_property_hv = node_property_hv.squeeze()
            node_property_hv = _circular_convolution_fft(node_property_hvs)
            
        else:
            node_property_hv = torch.zeros(data.x.size(0), self.hidden_dim, dtype=torch.float64, device=self.device)
            
        # Finally we update the data object with the "node_hv" property so that we can later access this 
        # in the forward pass of the model
        setattr(data, 'node_hv', node_property_hv)
        
        # --- graph properties ---
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
            if hasattr(encoder, 'encode_batch'):
                property_hv = encoder.encode_batch(property_value)
            else: 
                property_hv = torch.stack([encoder.encode(tens) for tens in property_value])
            
            graph_property_hvs.append(property_hv)
            
        if graph_property_hvs:
            # graph_property_hvs: (num_properties, batch_size, hidden_dim)
            graph_property_hvs = torch.stack(graph_property_hvs, dim=0)
            
            # graph_property_hv: (batch_size, hidden_dim)
            #graph_property_hv = torch_pairwise_reduce(graph_property_hvs, func=self.bind_fn, dim=0)
            graph_property_hv = torch.sum(graph_property_hvs, dim=0)
            graph_property_hv = graph_property_hv#.squeeze()
            
        else:
            graph_property_hv = torch.zeros(torch.max(data.batch) + 1, self.hidden_dim, dtype=torch.float64, device=self.device)
            
        # Finally we update the data object with the "graph_hv" property so that we can later access this
        # in the forward pass of the model.
        setattr(data, 'graph_hv', graph_property_hv)
            
        return data
    
    @property
    def uses_graph_properties(self) -> bool:
        """
        A simple utility function that returns True if the self.graph_encoder_map contains any entries 
        and therefore the encoding of graph properties is used.
        """
        return bool(len(self.graph_encoder_map) > 0)
    
    def forward(self,
                data: Data,
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
        
        # --- mapping node & graph properties as hypervectors ---
        # The "encoder_properties" method will actually manage the encoding of the node and graph properties of 
        # the graph (as represented by the Data object) into representative 
        # Afterwards, the data object contains the additional properties "data.node_hv" and "data.graph_hv" 
        # which represent the encoded hypervectors for the individual nodes or for the overall graphs respectively.
        data = self.encode_properties(data)
        
        # --- handling continuous edge weights ---
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
            edge_weight = 1000. * torch.ones(data.edge_index.shape[1], 1, device=self.device)
            
        # --- handling edge bi-directionality ---
        # If the bidirectional flag is given we will duplicate each edge in the input graphs and reverse the
        # order of node indices such that each node of each edge is always considered as a source and a target
        # for the message passing operation.
        # Similarly we also duplicate the edge weights such that the same edge weight is used for both edge
        # "directions".

        if self.bidirectional:
            edge_index = torch.cat([data.edge_index, data.edge_index[[1, 0]]], dim=1)
            edge_weight = torch.cat([edge_weight, edge_weight], dim=0)
        else:
            edge_index = data.edge_index
            edge_weight = edge_weight
                        
        # --- pushing to device ---
        # Its possible to use gpu acceleration and therefore we need to push all the relevant tensors to the
        # correct device that was selected when the HyperNet instance was created.
        data = data.to(self.device)
        edge_weight = edge_weight.to(self.device)
        edge_index = edge_index.to(self.device)
            
        # data.edge_index: (2, batch_size * num_edges)
        srcs, dsts = edge_index
    
        # In this data structure we will stack all the intermediate node embeddings for the various
        # message-passing depths.
        # node_hv_stack: (num_layers + 1, batch_size * num_nodes, hidden_dim)
        node_hv_layers = [data.node_hv]

        # --- message passing ---
        # Finally we perform the message passing itself over the given number of layers (depth).
        # Te message passing will create messages from the source nodes to the target nodes (weighted by
        # their edge weights) and then aggregate all the incoming messages for each target node by summing
        # up the messages. The node representation of the next layer is then calcualted by binding the
        # current node representation with the aggregated message representation and normalizing the result.
        for layer_index in range(self.depth):
            # messages are gated with the corresponding edge weights!
            messages = node_hv_layers[layer_index][dsts] * sigmoid(edge_weight)
            aggregated = scatter(messages, srcs, reduce='sum', dim_size=data.node_hv.size(0))
            #print(node_hv_layers[layer_index].shape, aggregated.shape)
            next_layer = normalize(self.bind_fn(node_hv_layers[layer_index], aggregated))
            #next_layer = normalize(_circular_convolution_fft(torch.stack([node_hv_layers[0], aggregated])))
            node_hv_layers.append(next_layer)

        # Stack all layers into a tensor
        node_hv_stack = torch.stack(node_hv_layers, dim=0)
        
        # We calculate the final graph-level embedding as the sum of all the node embeddings over all the various 
        # message passing depths and as the sum over all the nodes.
        # If the self.normalize_all flag is set to True we also normalize the node embeddings after summing
        # over the message passing depths as well as the final graph embedding.
        node_hv = node_hv_stack.sum(dim=0)
        if self.normalize_all:
            node_hv = normalize(node_hv)
        
        readout = scatter(node_hv, data.batch, reduce=self.pooling)
        
        # This is the main result of the message passing part.
        embedding: torch.Tensor = readout
        
        # --- graph properties ---
        # The hypernet may not only use node properties for a graph but also graph-level global properties.
        # If that is the case we simply add those to the final results.
        if self.uses_graph_properties:
            embedding = normalize(normalize(embedding) + normalize(data.graph_hv))
        
        # Graph hv stack is supposed to contain the graph hypervectors for each of the graphs in the
        # batch at the different message passing depths. Whcih means that the different layers
        # will have to be aggregated with scatter individually.
        # graph_hv_stack: (batch_size, num_layers +1, hidden_dim)
        graph_hv_stack = torch.zeros(
            size=(torch.max(data.batch) + 1, self.depth + 1, self.hidden_dim),
            dtype=torch.float64,
            device=self.device
        )
        
        for layer_index in range(self.depth + 1):
            # We aggregate the node hypervectors for each graph in the batch at the current message passing 
            # depth and store them in the graph_hv_stack.
            graph_hv_stack[:, layer_index] = scatter(
                node_hv_stack[layer_index], 
                data.batch, 
                reduce=self.pooling
            )
        
        
        return {
            
            # This the main result of the forward pass which is the individual graph embedding vectors of the 
            # input graphs.
            # graph_embedding: (batch_size, hidden_dim)
            'graph_embedding': embedding,
            
            # As additional information that might be useful we also pass the stack of the node embeddings across
            # the various convolutional depths.
            # node_hv_stack: (batch_size * num_nodes, num_layers + 1, hidden_dim)
            #'node_hv_stack': node_hv_stack.transpose(0, 1),
            
            # graph hv stack is supposed to contain the graph hypervectors for each of the graphs in the 
            # batch at the different message passing depths. Whcih means that the different layers 
            # will have to be aggregated with scatter individually.
            # graph_hv_stack: (batch_size, num_layers +1, hidden_dim)
            'graph_hv_stack': graph_hv_stack,
        }
        
    # --- decoding ---
    # These methods handle the inverse operation -> The decoding of the graph embedding vectors back into 
    # the original graph structure.
        
    def extract_distance_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Extracts the component of the embedding to use for distance calculations.

        This method allows different HyperNet variants to specify which part of their
        embedding should be used when computing distances during reconstruction or
        similarity comparisons. The default implementation returns the full embedding.

        **Design Rationale**

        During graph reconstruction, we compute distances between candidate graphs and
        target embeddings to guide the search. Different embedding schemes may want to
        use different components for this comparison:

        - Standard HyperNet: Uses full embedding (default)
        - CompositeHyperNet: Uses only g component (global structure)

        **Example**

        .. code-block:: python

            # Standard usage in reconstruction
            encoder = HyperNet(...)
            result = encoder.forward(data)
            embedding = result['graph_embedding']

            # Extract component for distance calculation
            distance_embedding = encoder.extract_distance_embedding(embedding)

            # Compute distance
            distance = distance_func(target_embedding, distance_embedding)

        :param embedding: Full embedding tensor from forward pass
        :type embedding: torch.Tensor

        :return: Component of embedding to use for distance calculation
        :rtype: torch.Tensor
        """
        # Default implementation: use full embedding
        return embedding

    def get_distance(self, hv1: torch.Tensor, hv2: torch.Tensor) -> float:
        """
        Calculate the distance between two hypervectors using the configured distance function.

        This method overrides the default implementation from AbstractHyperNet to use the
        distance function specified during initialization. This allows each HyperNet instance
        to use a custom distance metric (cosine, Manhattan, Euclidean, etc.) that is
        consistently applied during graph reconstruction and similarity calculations.

        **Example Usage**

        .. code-block:: python

            from graph_hdc.reconstruct import manhattan_distance

            # Create encoder with custom distance function
            encoder = HyperNet(
                hidden_dim=1000,
                distance_func=manhattan_distance
            )

            # Encode two graphs
            result1 = encoder.forward(data1)
            result2 = encoder.forward(data2)

            # Calculate distance using Manhattan metric
            distance = encoder.get_distance(
                result1['graph_embedding'],
                result2['graph_embedding']
            )

        :param hv1: First hypervector with shape (hidden_dim,) or (batch_size, hidden_dim)
        :type hv1: torch.Tensor
        :param hv2: Second hypervector with shape (hidden_dim,) or (batch_size, hidden_dim)
        :type hv2: torch.Tensor

        :return: Distance value as a float
        :rtype: float
        """
        return self.distance_func(hv1, hv2)

    def decode_order_zero(self,
                          embedding: torch.Tensor,
                          iterations: int = 1,
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
        
        # If the embedding is not just a vector but instead a 2D tensor we assume that it actually is a stack 
        # of the graph embeddings at the different message passing depths. In this case it most beneficial to 
        # decode the nodes from the zero-layer embedding which is the first in the stack.
        if embedding.dim() == 2:
            embedding = embedding[0, :]
        
        # In this list we'll store the final decoded constraints about which kinds of nodes are present in the 
        # graph. Each constraints is represented as a dictionary which contains information about which kind of 
        # node is present (as a combination of node properties) and how many of these nodes are present.
        constraints_order_zero: List[Dict[str, dict]] = []
    
        # self.node_hv_combination_stack is a tensor of shape (num_combinations, hidden_dim)
        # embedding is a tensor of shape (hidden_dim,)
        # dot_products should be a tensor of shape (num_combinations,)
        dot_products = torch.matmul(self.node_hv_combination_stack, embedding.squeeze())
        for comb_dict, value in zip(self.node_hv_combination_keys, dot_products):
            
            # By multiplying the embedding with the specific node hypervector we essentially calculate the
            # projection of the graph along the direction of the node. This projection should be proportional 
            # to the number of times that a node of that specific type was included in the original graph.
            #value = torch.dot(hv, embedding.squeeze()).detach().item()
            if np.round(value) > 0:
                num = int(np.round(value))
                result_dict = {
                    'src': comb_dict.copy(),
                    'num': num,
                }
                constraints_order_zero.append(result_dict)
                
        return constraints_order_zero

    def decode_nodes(self,
                     embedding: torch.Tensor,
                     iterations: int = 1,
                     ) -> dict:
        """
        Returns decoded node information in graph dict format from the given ``embedding`` vector.

        This method builds upon the decode_order_zero method to provide nodes in the standard graph dict
        format that can be used by other parts of the codebase. It creates a full connectivity edge structure
        and extracts all node properties as properly ordered numpy arrays.

        :param embedding: The high-dimensional graph embedding vector that represents the graph.
        :param iterations: Number of iterations for the decoding process (passed to decode_order_zero).

        :returns: A graph dict containing:
            - "node_indices": A numpy array of node indices (0, 1, 2, ...)
            - "edge_index_full": A numpy array of shape (num_edges, 2) representing full connectivity
            - "edge_weight_full": A numpy array of shape (num_edges,) with all zeros
            - Additional arrays for each node property found in the constraints
        """

        # Get the constraints from decode_order_zero
        constraints = self.decode_order_zero(embedding, iterations)

        if not constraints:
            # Return empty graph dict if no constraints found
            return {
                'node_indices': np.array([], dtype=int),
                'edge_index_full': np.array([], dtype=int).reshape(0, 2),
                'edge_weight_full': np.array([], dtype=float),
            }

        # Build ordered list of nodes from constraints
        node_list = []
        node_index = 0

        for constraint in constraints:
            num_nodes = constraint['num']
            node_properties = constraint['src']

            for _ in range(num_nodes):
                node_dict = {'index': node_index}
                node_dict.update(node_properties)
                node_list.append(node_dict)
                node_index += 1

        total_nodes = len(node_list)

        # Create node_indices tensor
        node_indices = np.arange(total_nodes, dtype=int)

        # Create full connectivity edge structure
        edge_index_full = []
        for i in range(total_nodes):
            for j in range(i):
                edge_index_full.append([i, j])

        edge_index_full = np.array(edge_index_full, dtype=int)
        edge_weight_full = np.zeros((len(edge_index_full), 1), dtype=float)

        # Build the graph dict
        graph_dict = {
            'node_indices': node_indices,
            'edge_index_full': edge_index_full,
            'edge_weight_full': edge_weight_full,
        }

        # Extract node properties into properly ordered arrays
        if total_nodes > 0:
            # Get all unique property names from the node encoders
            property_names = set()
            for node in node_list:
                property_names.update(node.keys())
            property_names.discard('index')  # Remove the index key

            # Create arrays for each property
            for prop_name in property_names:
                prop_values = [node.get(prop_name) for node in node_list]
                graph_dict[prop_name] = np.array(prop_values)

        return graph_dict

    def _decode_order_one_iterative(
        self,
        embedding: torch.Tensor,
        constraints_order_zero: List[dict],
        max_iters: int = 100,
        threshold: float = 0.1,
        use_break: bool = True,
    ) -> List[dict]:
        """
        Iterative "explain away" edge decoding implementation.

        This method progressively identifies edges by:
        1. Scoring all possible edge candidates
        2. Selecting the top-scoring edges (using canonical form to avoid duplicates)
        3. Subtracting the contribution of found edges from the working embedding
        4. Repeating until convergence

        **Algorithm Details**

        - **Node Indexing**: Maps each constraint dict to a unique integer index
        - **Edge Candidates**: All directed pairs (i, j) where i != j
        - **Scoring**: Dot product of bind(node_i, node_j) with working embedding
        - **Selection**: Canonical edge tracking per iteration to avoid (i,j)/(j,i) duplicates
        - **Subtraction**: Removes bind(i, j) and bind(j, i) from working embedding
        - **Convergence**: Stops when max score ≤ 0, threshold not met, or max_iters reached

        :param embedding: Graph embedding vector (may be 2D stack, will use first layer)
        :param constraints_order_zero: List of node constraints from decode_order_zero
        :param max_iters: Maximum number of iterations
        :param threshold: Minimum score threshold for accepting edges
        :param use_break: Whether to break after first match per iteration

        :returns: List of edge constraints with 'src', 'dst', 'num' keys
        """
        # Extract first layer if embedding is a stack
        if embedding.dim() == 2:
            work_embedding = embedding[0, :].clone()
        else:
            work_embedding = embedding.clone()

        # Build node index mapping: constraint_dict -> unique_index
        # This creates a list where each element is a node property dict
        node_list: List[dict] = []
        for constraint in constraints_order_zero:
            for _ in range(constraint['num']):
                node_list.append(constraint['src'].copy())

        # Create bidirectional mapping
        node_to_idx: Dict[int, int] = {}  # position in node_list -> unique index
        idx_to_constraint: Dict[int, dict] = {}  # unique index -> constraint dict

        idx = 0
        for pos, node_dict in enumerate(node_list):
            node_to_idx[pos] = idx
            idx_to_constraint[idx] = node_dict
            idx += 1

        # Get all node indices
        node_indices = list(idx_to_constraint.keys())

        # Build all directed edge candidates (i, j) where i != j
        edge_candidates: List[Tuple[int, int]] = [
            (i, j) for i in node_indices for j in node_indices if i != j
        ]

        # Track found edges as a Counter
        found_edges: Counter = Counter()

        iteration = 1
        should_break = False

        while iteration <= max_iters and edge_candidates and not should_break:
            # Track edges processed in THIS iteration (resets each iteration)
            processed_this_iteration = set()

            # Score all remaining candidates
            scores = []
            for u, v in edge_candidates:
                # Get node hypervectors
                hv_u = self.node_hv_combinations.get(idx_to_constraint[u])
                hv_v = self.node_hv_combinations.get(idx_to_constraint[v])

                # Compute edge hypervector
                edge_hv = self.bind_fn(hv_u, hv_v)

                # Score against working embedding
                score = torch.dot(edge_hv, work_embedding).item()
                scores.append(score)

            # Check convergence: stop if top score ≤ 0
            if not scores or max(scores) <= 0.0:
                break

            # Sort by score descending
            sorted_indices = sorted(
                range(len(scores)),
                key=lambda i: scores[i],
                reverse=True
            )

            # Process edges in score order, skipping directional duplicates within this iteration
            for idx in sorted_indices:
                u, v = edge_candidates[idx]
                score = scores[idx]

                # Use canonical form to detect if we've already processed this edge pair
                canonical_edge = (min(u, v), max(u, v))
                if canonical_edge in processed_this_iteration:
                    continue  # Skip the reverse direction

                # Mark this edge pair as processed in this iteration
                processed_this_iteration.add(canonical_edge)

                if score > threshold:
                    # Add only canonical direction to found edges
                    found_edges[canonical_edge] += 1

                    # Subtract edge contribution from working embedding (both directions)
                    hv_u = self.node_hv_combinations.get(idx_to_constraint[u])
                    hv_v = self.node_hv_combinations.get(idx_to_constraint[v])

                    work_embedding = work_embedding - self.bind_fn(hv_u, hv_v)
                    work_embedding = work_embedding - self.bind_fn(hv_v, hv_u)

                    # Remove both directions from candidates
                    edge_candidates = [
                        e for e in edge_candidates
                        if e not in {(u, v), (v, u)}
                    ]

                    if use_break:
                        break

                elif score <= 0:
                    should_break = True
                    break

            iteration += 1

        # Convert Counter to List[dict] format matching decode_order_one output
        # Group edges by (src_constraint, dst_constraint) and count
        edge_type_counts: Dict[Tuple[tuple, tuple], int] = {}

        for (u, v), count in found_edges.items():
            src_dict = idx_to_constraint[u]
            dst_dict = idx_to_constraint[v]

            # Create hashable keys for grouping
            src_key = tuple(sorted(src_dict.items()))
            dst_key = tuple(sorted(dst_dict.items()))

            edge_type_key = (src_key, dst_key)

            if edge_type_key not in edge_type_counts:
                edge_type_counts[edge_type_key] = 0
            edge_type_counts[edge_type_key] += count

        # Convert to output format
        constraints_order_one: List[dict] = []
        for (src_key, dst_key), count in edge_type_counts.items():
            src_dict = dict(src_key)
            dst_dict = dict(dst_key)

            constraints_order_one.append({
                'src': src_dict,
                'dst': dst_dict,
                'num': count
            })

        return constraints_order_one

    def decode_order_one(self,
                         embedding: torch.Tensor,
                         constraints_order_zero: List[dict],
                         correction_factor_map: Dict[int, float] = {
                             0: 1.0,
                             1: 1.0,
                             2: 1.0,
                             3: 1.0,
                         },
                         use_iterative: bool = False,
                         max_iters: int = 50,
                         threshold: float = 0.5,
                         use_break: bool = True,
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

        **Iterative Explain-Away Mode**

        When ``use_iterative=True``, this method uses an iterative refinement approach that:

        1. Builds all possible directed edge candidates from decoded nodes
        2. Iteratively scores all candidates against the current embedding
        3. Selects top-scoring edges (using every-other slot to avoid duplicates)
        4. Subtracts found edges from the working embedding
        5. Continues until convergence (max score ≤ 0, threshold not met, or max iterations)

        This "explain away" approach progressively removes the contribution of found edges, reducing
        interference and improving decoding accuracy for complex graphs.

        :param embedding: The high-dimensional graph embedding vector that represents the graph.
        :param constraints_order_zero: The list of constraints that represent the zero order information about the
            nodes that were present in the original graph.
        :param correction_factor_map: A dictionary that contains correction factors for the number of shared core
            properties between the nodes that constitute the edge. The keys are the number of shared core properties
            and the values are the correction factors that should be applied to the calculated edge count.
            Only used in non-iterative mode.
        :param use_iterative: If True, use iterative explain-away decoding scheme. Default: False.
        :param max_iters: Maximum number of iterations for iterative mode. Default: 50.
        :param threshold: Minimum score threshold for iterative mode. Default: 0.5.
        :param use_break: In iterative mode, whether to break after first match per iteration. Default: True.

        :returns: A list of constraints where each constraint is represented by a dictionary with the keys:
            - src: A dictionary that represents the properties of the source node as they were originally encoded
                by the node encoders. The keys in this dict are the same as the names of the node encoders given to
                the constructor.
            - dst: A dictionary that represents the properties of the destination node as they were originally
                encoded by the node encoders. The keys in this dict are the same as the names of the node encoders
                given to the constructor.
            - num: The integer number of how many of these edges are present in the graph.
        """
        if use_iterative:
            return self._decode_order_one_iterative(
                embedding=embedding,
                constraints_order_zero=constraints_order_zero,
                max_iters=max_iters,
                threshold=threshold,
                use_break=use_break,
            )

        # Original single-pass implementation
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
                'bidirectional': self.bidirectional,
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
                    num_iterations: int = 25, 
                    learning_rate: float = 1.0,
                    batch_size: int = 10,
                    low: float = 0.0,
                    high: float = 1.0
                    ) -> dict:
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
        print('node constraints', len(node_constraints))
        edge_constraints = self.decode_order_one(graph_hv, node_constraints)
        print('edge constraints', len(edge_constraints))
        
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
            tens = torch.tensor([
                self.node_encoder_map[key].normalize(node[key])
                for node in index_node_map.values()
            ])
            setattr(data, key, tens)
        
        data.edge_index = torch.tensor(list(edge_indices), dtype=torch.long).t()
        data.batch = torch.tensor([0] * len(index_node_map), dtype=torch.long)
        data.x = torch.zeros(len(index_node_map), self.hidden_dim)
        data = self.encode_properties(data)
        
        data_list: List[Data] = []
        for _ in range(batch_size):
            data = data.clone()
            data.edge_weight = torch.tensor(np.random.uniform(
                low=low, 
                high=high, 
                size=(data.edge_index.size(1), 1)
            ))
            data_list.append(data)
            
        batch = Batch.from_data_list(data_list)
        batch.edge_weight.requires_grad = True
        
        num_nodes = batch.edge_index.max().item() + 1
        
        optimizer = torch.optim.Adam([batch.edge_weight], lr=learning_rate)
        
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
                _edges_src = scatter(torch.ones_like(_edge_weight), batch.edge_index[0], dim_size=num_nodes, reduce='sum')
                _edges_dst = scatter(torch.ones_like(_edge_weight), batch.edge_index[1], dim_size=num_nodes, reduce='sum')
                _num_edges = _edges_src + _edges_dst
                
                #_edge_weight = torch.where(_edge_weight > 0.5, _edge_weight, _edge_weight * 0.001)
                #_edge_weight = torch.where(_edge_weight > 0.2, torch.ones_like(_edge_weight), torch.zeros_like(_edge_weight))
                scatter_src = scatter(_edge_weight, batch.edge_index[0], dim_size=num_nodes, reduce='sum')
                scatter_dst = scatter(_edge_weight, batch.edge_index[1], dim_size=num_nodes, reduce='sum')
                # Calculate the actual node degree by summing over the edge weights of all the in and out going edges of a node
                node_degree = scatter_src + scatter_dst
                        
            loss.backward()
            optimizer.step()
            
            print(loss.item())
            
        # discretizing the still continuous edge weights and constructing a new "edge_index"
        # connectivity structure based only on the edges that have a weight > 0.5
        print(batch.edge_weight)
        # Create a new batch to avoid in-place operations on tensors with gradients
        batch_discrete = batch.clone()
        batch_discrete.edge_weight = (batch.edge_weight >= 0).float().detach()
        result = self.forward(batch_discrete)
        embedding = result['graph_embedding']  # shape (candidate_batch_size, hidden_dim)
        losses = torch.square((embedding - graph_hv.expand_as(embedding))).mean(dim=1)
            
        # We get the index of the best candidate according to the loss of the final epoch
        losses = losses.detach().cpu().numpy()
        index_best = np.argmin(losses)
        print(losses, losses[index_best])
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
    
