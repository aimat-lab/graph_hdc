"""
Composite HyperNet implementation with explicit structural separation.

This module contains variants of HyperNet that explicitly separate different
orders of structural information (nodes, edges, global context) in the embedding.
"""

from itertools import product
from typing import Dict, Optional, List

import numpy as np
import torch
from torch.nn.functional import normalize, sigmoid
from torch_geometric.data import Data
from torch_geometric.utils import scatter

from .base import AbstractHyperNet
from .main import HyperNet


class CompositeHyperNet(HyperNet):
    """
    A variant of HyperNet with composite embeddings that explicitly separate
    order-0 (nodes), order-1 (edges), and global (message-passed) information.

    This class inherits from HyperNet and modifies the forward pass to create
    a composite embedding that concatenates three components:

    - **h_0** (Order-0): Sum of initial node hypervectors before message passing
    - **h_1** (Order-1): Sum of initial edge hypervectors before message passing
    - **g** (Global): Standard HyperNet embedding after message passing

    The final embedding is the concatenation: h_0 | h_1 | g with shape
    (batch_size, 3 * hidden_dim). This explicit separation enables more accurate
    and efficient decoding of graph structure.

    **Design Rationale**

    The standard HyperNet mixes structural information at different levels (nodes,
    edges, paths) into a single embedding through message passing. While this captures
    rich relational information, it makes decoding challenging as order-0 and order-1
    information becomes entangled with higher-order patterns.

    CompositeHyperNet addresses this by explicitly preserving:

    1. Node-level information (h_0) - enables direct node decoding
    2. Edge-level information (h_1) - enables direct edge decoding
    3. Global context (g) - preserves message-passed relational patterns

    **Usage Example**

    .. code-block:: python

        from graph_hdc.models import CompositeHyperNet
        from graph_hdc.utils import CategoricalIntegerEncoder

        # Create encoder with same parameters as HyperNet
        encoder = CompositeHyperNet(
            hidden_dim=1000,
            depth=3,
            node_encoder_map={
                'node_atoms': CategoricalIntegerEncoder(dim=1000, num_categories=10),
                'node_degrees': CategoricalIntegerEncoder(dim=1000, num_categories=5),
            }
        )

        # Forward pass creates composite embedding
        result = encoder.forward(data)
        composite_embedding = result['graph_embedding']  # Shape: (batch_size, 3000)

        # Components are also returned separately
        h_0 = result['h_0']  # Node information
        h_1 = result['h_1']  # Edge information
        g = result['g']      # Global information

        # Decoding is more accurate using explicit components
        node_constraints = encoder.decode_order_zero(composite_embedding)
        edge_constraints = encoder.decode_order_one(composite_embedding)

    :param hidden_dim: Dimensionality of each component (total embedding is 3 * hidden_dim)
    :type hidden_dim: int
    :param depth: Number of message passing iterations
    :type depth: int
    :param node_encoder_map: Dictionary mapping node property names to encoders
    :type node_encoder_map: Dict[str, AbstractEncoder]
    :param graph_encoder_map: Dictionary mapping graph property names to encoders
    :type graph_encoder_map: Dict[str, AbstractEncoder]
    :param bind_fn: Function for binding two hypervectors (default: circular convolution)
    :type bind_fn: Callable or str
    :param unbind_fn: Function for unbinding two hypervectors (default: circular correlation)
    :type unbind_fn: Callable or str
    :param pooling: Aggregation method for graph pooling ('sum' or 'mean')
    :type pooling: str
    :param normalize_all: Whether to normalize embeddings after each operation
    :type normalize_all: bool
    :param bidirectional: Whether to treat edges as bidirectional
    :type bidirectional: bool
    :param seed: Random seed for reproducibility
    :type seed: Optional[int]
    :param device: Device for tensor operations ('cpu' or 'cuda')
    :type device: str
    """

    def __init__(self, **kwargs):
        """
        Initialize CompositeHyperNet by inheriting all parameters from HyperNet.

        All initialization is delegated to the parent HyperNet class, ensuring
        compatibility with existing encoder configurations and binding functions.

        :param kwargs: All keyword arguments accepted by HyperNet.__init__()
        """
        # Initialize parent HyperNet with all provided arguments
        super().__init__(**kwargs)

    def forward(self, data: Data) -> dict:
        """
        Encodes graphs into composite hypervectors separating structural information.

        This method extends the standard HyperNet encoding by creating a composite
        embedding that explicitly separates order-0 (nodes), order-1 (edges), and
        global (message-passed) information. This separation enables more accurate
        and efficient decoding of graph structure.

        **Algorithm**

        1. Encode node and graph properties using parent's encode_properties()
        2. Compute h_0: Sum of initial node hypervectors
        3. Compute h_1: Sum of bound edge hypervectors
        4. Perform message passing to compute g (global embedding)
        5. Concatenate h_0 | h_1 | g to form composite embedding

        **The Composite Embedding Structure**

        - **h_0** (dimensions 0 to hidden_dim-1):
          Sum of all initial node representations. This captures the "bag of nodes"
          information and makes decode_order_zero() straightforward.

        - **h_1** (dimensions hidden_dim to 2*hidden_dim-1):
          Sum of all initial edge representations, where each edge is represented
          as bind(node_i, node_j). This captures the "bag of edges" information
          and enables direct edge decoding.

        - **g** (dimensions 2*hidden_dim to 3*hidden_dim-1):
          Standard HyperNet embedding after message passing. This preserves global
          context and higher-order structural patterns.

        **Example**

        .. code-block:: python

            encoder = CompositeHyperNet(hidden_dim=1000, depth=3, node_encoder_map=...)

            # Single graph
            result = encoder.forward(data)
            embedding = result['graph_embedding']  # Shape: (1, 3000)

            # Access components
            h_0 = result['h_0']  # (1, 1000) - nodes
            h_1 = result['h_1']  # (1, 1000) - edges
            g = result['g']      # (1, 1000) - global

            # Verify concatenation
            assert torch.allclose(embedding, torch.cat([h_0, h_1, g], dim=-1))

        :param data: PyG Data object representing batched graphs
        :type data: torch_geometric.data.Data

        :return: Dictionary containing:
            - 'graph_embedding': Composite embedding (batch_size, 3*hidden_dim)
            - 'graph_hv_stack': Composite embeddings per layer (batch, depth+1, 3*hidden_dim)
            - 'h_0': Order-0 component (batch_size, hidden_dim)
            - 'h_1': Order-1 component (batch_size, hidden_dim)
            - 'g': Global component (batch_size, hidden_dim)
        :rtype: Dict[str, torch.Tensor]
        """
        # Step 1: Encode node and graph properties using parent's method
        data = self.encode_properties(data)

        # Determine batch size for initializing tensors
        batch_size = int(torch.max(data.batch).item()) + 1

        # Step 2: Compute h_0 (sum of initial node representations)
        # This is the "bag of nodes" representation before any message passing
        h_0 = scatter(data.node_hv, data.batch, reduce=self.pooling, dim_size=batch_size)

        # Step 3: Compute h_1 (sum of initial edge representations)
        # Handle edge directionality and weights
        if hasattr(data, 'edge_weight') and data.edge_weight is not None:
            edge_weight = data.edge_weight
        else:
            # Default to discrete edges (sigmoid(1000) ≈ 1.0)
            edge_weight = 1000. * torch.ones(data.edge_index.shape[1], 1, device=self.device)

        # Handle bidirectional edges
        if self.bidirectional:
            edge_index = torch.cat([data.edge_index, data.edge_index[[1, 0]]], dim=1)
            edge_weight = torch.cat([edge_weight, edge_weight], dim=0)
        else:
            edge_index = data.edge_index

        # Push to device
        data = data.to(self.device)
        edge_weight = edge_weight.to(self.device)
        edge_index = edge_index.to(self.device)

        # Compute edge representations as bind(src_node, dst_node)
        if edge_index.size(1) > 0:  # Only if edges exist
            srcs, dsts = edge_index[0], edge_index[1]
            edge_hvs = self.bind_fn(data.node_hv[srcs], data.node_hv[dsts])

            # Map edge indices to batch indices
            batch_per_edge = data.batch[srcs]
            h_1 = scatter(edge_hvs, batch_per_edge, reduce=self.pooling, dim_size=batch_size)
        else:
            # No edges: h_1 is zero vector
            h_1 = torch.zeros(batch_size, self.hidden_dim, dtype=torch.float64, device=self.device)

        # Step 4: Compute g (global embedding via message passing)
        # Reuse parent's message passing logic
        node_hv_layers = [data.node_hv]

        for layer_index in range(self.depth):
            if edge_index.size(1) > 0:
                messages = node_hv_layers[layer_index][dsts] * sigmoid(edge_weight)
                aggregated = scatter(messages, srcs, reduce='sum', dim_size=data.node_hv.size(0))
                next_layer = normalize(self.bind_fn(node_hv_layers[layer_index], aggregated))
            else:
                # No edges: just normalize current layer
                next_layer = normalize(node_hv_layers[layer_index])
            node_hv_layers.append(next_layer)

        # Stack all layers
        node_hv_stack = torch.stack(node_hv_layers, dim=0)

        # Sum over layers and pool to graph level
        node_hv = node_hv_stack.sum(dim=0)
        if self.normalize_all:
            node_hv = normalize(node_hv)

        g = scatter(node_hv, data.batch, reduce=self.pooling, dim_size=batch_size)

        # Add graph properties if specified
        if self.uses_graph_properties:
            g = normalize(normalize(g) + normalize(data.graph_hv))

        # Step 5: Concatenate to form composite embedding
        composite_embedding = torch.cat([h_0, h_1, g], dim=-1)

        # Create graph_hv_stack with composite structure for each layer
        # Each layer has the composite structure: h_0 | h_1 | g_layer
        graph_hv_stack = torch.zeros(
            size=(batch_size, self.depth + 1, 3 * self.hidden_dim),
            dtype=torch.float64,
            device=self.device
        )

        for layer_index in range(self.depth + 1):
            # Pool nodes at this layer to graph level
            g_layer = scatter(
                node_hv_stack[layer_index],
                data.batch,
                reduce=self.pooling,
                dim_size=batch_size
            )
            # h_0 and h_1 remain constant across layers (no message passing)
            graph_hv_stack[:, layer_index, :] = torch.cat([h_0, h_1, g_layer], dim=-1)

        return {
            # Main composite embedding
            'graph_embedding': composite_embedding,  # (batch_size, 3*hidden_dim)

            # Stack of composite embeddings per message passing layer
            'graph_hv_stack': graph_hv_stack,  # (batch_size, depth+1, 3*hidden_dim)

            # Individual components
            'h_0': h_0,  # (batch_size, hidden_dim) - order-0 information
            'h_1': h_1,  # (batch_size, hidden_dim) - order-1 information
            'g': g,      # (batch_size, hidden_dim) - global information
        }

    def extract_distance_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Extracts the g component (global structure) for distance calculations.

        For CompositeHyperNet, only the g component (dimensions 2*hidden_dim to 3*hidden_dim)
        should be used for distance calculations during reconstruction. This is because:

        - h_0 captures only node counts (bag of nodes)
        - h_1 captures only edge types (bag of edges)
        - g captures global structural patterns after message passing

        The g component is most similar to standard HyperNet embeddings and best represents
        the overall graph structure for similarity comparisons.

        **Example**

        .. code-block:: python

            encoder = CompositeHyperNet(hidden_dim=1000, ...)
            result = encoder.forward(data)
            composite_embedding = result['graph_embedding']  # Shape: (1, 3000)

            # Extract g component for distance
            g_component = encoder.extract_distance_embedding(composite_embedding)
            # g_component shape: (1, 1000)

            # Use in reconstruction
            distance = distance_func(target_g, g_component)

        **Handling Different Input Shapes**

        - 1D tensor (3*hidden_dim,): Single composite embedding
        - 2D tensor (batch, 3*hidden_dim): Batch of composite embeddings
        - 2D tensor (layers, 3*hidden_dim): Stack of embeddings across layers

        :param embedding: Composite embedding tensor
        :type embedding: torch.Tensor

        :return: The g component for distance calculation
        :rtype: torch.Tensor
        """
        if embedding.dim() == 1:
            # Single composite embedding: (3*hidden_dim,)
            # Extract g: dimensions 2*hidden_dim to 3*hidden_dim
            return embedding[2*self.hidden_dim:3*self.hidden_dim]
        elif embedding.dim() == 2:
            # Batch or stack: (n, 3*hidden_dim)
            # Extract g from all rows
            return embedding[:, 2*self.hidden_dim:3*self.hidden_dim]
        else:
            raise ValueError(f"Expected 1D or 2D embedding, got {embedding.dim()}D")

    def decode_order_zero(
        self,
        embedding: torch.Tensor,
        iterations: int = 1
    ) -> List[dict]:
        """
        Decodes node information from the h_0 component of the composite embedding.

        This method extracts order-0 (node-level) information directly from the h_0
        component of the composite embedding. Since h_0 is simply the sum of initial
        node representations, decoding is straightforward and more accurate than the
        parent HyperNet's method, which must disentangle node information from the
        mixed embedding.

        **Algorithm**

        1. Extract h_0 component (first hidden_dim dimensions) from composite embedding
        2. Project h_0 onto all possible node type combinations
        3. The magnitude of each projection gives the count of that node type

        **Advantages over HyperNet.decode_order_zero()**

        - **More accurate**: h_0 is pure node information, not mixed with edges/paths
        - **Simpler**: No need to handle message passing layers
        - **Faster**: Direct projection without iterative processing

        **Example**

        .. code-block:: python

            encoder = CompositeHyperNet(...)
            result = encoder.forward(data)
            embedding = result['graph_embedding']  # Shape: (1, 3000)

            # Decode nodes from h_0 component
            node_constraints = encoder.decode_order_zero(embedding)

            # Example output:
            # [
            #     {'src': {'node_atoms': 6, 'node_degrees': 2}, 'num': 3},
            #     {'src': {'node_atoms': 7, 'node_degrees': 3}, 'num': 1},
            #     {'src': {'node_atoms': 8, 'node_degrees': 1}, 'num': 2},
            # ]

        **Handling Different Input Shapes**

        - 1D tensor: Single composite embedding (hidden_dim * 3,)
        - 2D tensor: Stack of embeddings (num_layers, hidden_dim * 3) or batch
        - In case of 2D, uses first row (layer 0 or first batch item)

        :param embedding: Composite embedding or embedding stack
        :type embedding: torch.Tensor
        :param iterations: Unused (kept for API compatibility)
        :type iterations: int

        :return: List of node constraints, each containing:
            - 'src': Dict of node properties (keys match node_encoder_map keys)
            - 'num': Integer count of how many such nodes exist
        :rtype: List[dict]
        """
        # Extract h_0 component from composite embedding
        if embedding.dim() == 1:
            # Single composite embedding: (3*hidden_dim,)
            h_0 = embedding[:self.hidden_dim]
        elif embedding.dim() == 2:
            # Stack or batch: (n, 3*hidden_dim)
            # Use first row (layer 0 or first batch item)
            h_0 = embedding[0, :self.hidden_dim]
        else:
            raise ValueError(f"Expected 1D or 2D embedding, got {embedding.dim()}D")

        # Project h_0 onto all possible node type combinations
        # self.node_hv_combination_stack: (num_combinations, hidden_dim)
        # h_0: (hidden_dim,)
        # dot_products: (num_combinations,)
        dot_products = torch.matmul(self.node_hv_combination_stack, h_0.squeeze())

        # Extract non-zero projections as node constraints
        constraints_order_zero: List[dict] = []
        for comb_dict, value in zip(self.node_hv_combination_keys, dot_products):
            count = np.round(value.item())
            if count > 0:
                constraints_order_zero.append({
                    'src': comb_dict.copy(),
                    'num': int(count),
                })

        return constraints_order_zero

    def decode_order_one(
        self,
        embedding: torch.Tensor,
        constraints_order_zero: Optional[List[dict]] = None,
        correction_factor_map: Optional[Dict[int, float]] = None,
        use_iterative: bool = True,
        max_iters: int = 50,
        threshold: float = 0.25,
        use_break: bool = True,
    ) -> List[dict]:
        """
        Decodes edge information from the h_1 component of the composite embedding.

        This method extracts order-1 (edge-level) information directly from the h_1
        component of the composite embedding. Since h_1 contains the sum of bound
        edge representations, decoding is direct and more accurate than the parent
        HyperNet's method.

        **Algorithm**

        1. Extract h_1 component (dimensions hidden_dim to 2*hidden_dim) from embedding
        2. Get zero-order constraints (if not provided, decode them from h_0)
        3. For each possible edge (node_i, node_j):
           - Compute edge hypervector as bind(hv_i, hv_j)
           - Project h_1 onto this edge hypervector
           - The magnitude gives the count of this edge type

        **Advantages over HyperNet.decode_order_one()**

        - **More accurate**: h_1 is pure edge information
        - **Simpler**: No correction factors needed (h_1 is clean)
        - **Independent**: Can decode edges without zero-order constraints (optional)

        **Iterative Explain-Away Mode**

        When ``use_iterative=True``, this method uses an iterative refinement approach on
        the h_1 component, following the same "explain away" scheme as the parent HyperNet
        but operating on the cleaner h_1 edge embedding.

        **Example**

        .. code-block:: python

            encoder = CompositeHyperNet(...)
            result = encoder.forward(data)
            embedding = result['graph_embedding']

            # Option 1: Decode with provided node constraints (single-pass)
            node_constraints = encoder.decode_order_zero(embedding)
            edge_constraints = encoder.decode_order_one(embedding, node_constraints)

            # Option 2: Decode with iterative mode for better accuracy
            edge_constraints = encoder.decode_order_one(
                embedding,
                node_constraints,
                use_iterative=True
            )

            # Example output:
            # [
            #     {'src': {'node_atoms': 6}, 'dst': {'node_atoms': 6}, 'num': 2},
            #     {'src': {'node_atoms': 6}, 'dst': {'node_atoms': 7}, 'num': 1},
            # ]

        **Design Note: Correction Factors**

        The parent HyperNet.decode_order_one() uses correction_factor_map to account
        for interference when nodes share properties. In CompositeHyperNet, h_1 is
        computed directly from edges before message passing, so interference is
        minimal and correction factors are typically unnecessary. The parameter is
        kept for API compatibility but defaults to no correction (all 1.0).

        :param embedding: Composite embedding or embedding stack
        :type embedding: torch.Tensor
        :param constraints_order_zero: Pre-computed node constraints (optional)
        :type constraints_order_zero: Optional[List[dict]]
        :param correction_factor_map: Unused (kept for API compatibility)
        :type correction_factor_map: Optional[Dict[int, float]]
        :param use_iterative: If True, use iterative explain-away decoding scheme. Default: False.
        :param max_iters: Maximum number of iterations for iterative mode. Default: 50.
        :param threshold: Minimum score threshold for iterative mode. Default: 0.5.
        :param use_break: In iterative mode, whether to break after first match per iteration. Default: True.

        :return: List of edge constraints, each containing:
            - 'src': Dict of source node properties
            - 'dst': Dict of destination node properties
            - 'num': Integer count of how many such edges exist
        :rtype: List[dict]
        """
        # Get zero-order constraints if not provided
        if constraints_order_zero is None:
            constraints_order_zero = self.decode_order_zero(embedding)

        # Extract h_1 component from composite embedding
        if embedding.dim() == 1:
            # Single composite embedding: (3*hidden_dim,)
            h_1 = embedding[self.hidden_dim:2*self.hidden_dim]
        elif embedding.dim() == 2:
            # Stack or batch: (n, 3*hidden_dim)
            # Use first row (layer 0 or first batch item)
            h_1 = embedding[0, self.hidden_dim:2*self.hidden_dim]
        else:
            raise ValueError(f"Expected 1D or 2D embedding, got {embedding.dim()}D")

        # Use iterative decoding if requested
        if use_iterative:
            return self._decode_order_one_iterative(
                embedding=h_1,  # Pass h_1 component instead of full embedding
                constraints_order_zero=constraints_order_zero,
                max_iters=max_iters,
                threshold=threshold,
                use_break=use_break,
            )

        # Original single-pass implementation
        # Decode edges by projecting h_1 onto all possible edge combinations
        constraints_order_one: List[dict] = []

        for const_i, const_j in product(constraints_order_zero, repeat=2):
            # Get node hypervectors
            hv_i = self.node_hv_combinations.get(const_i['src'])
            hv_j = self.node_hv_combinations.get(const_j['src'])

            # Compute edge hypervector as bind(node_i, node_j)
            hv_edge = self.bind_fn(hv_i, hv_j)

            # Project h_1 onto this edge type
            value = torch.dot(hv_edge, h_1.squeeze()).detach().item()

            # Note: In CompositeHyperNet, h_1 is clean edge information,
            # so correction factors are typically not needed. We apply them
            # if provided for compatibility, but default to 1.0 (no correction).
            if correction_factor_map is not None:
                num_shared = len(set(const_i['src'].keys()) & set(const_j['src'].keys()))
                value *= correction_factor_map.get(num_shared, 1.0)

            count = np.round(value)
            if count > 0:
                constraints_order_one.append({
                    'src': const_i['src'].copy(),
                    'dst': const_j['src'].copy(),
                    'num': int(count)
                })

        return constraints_order_one
