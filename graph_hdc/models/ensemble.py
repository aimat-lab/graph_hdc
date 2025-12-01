"""
Ensemble HyperNet implementation for improved robustness through majority voting.

This module provides the HyperNetEnsemble class which combines multiple HyperNet
models to improve encoding and decoding accuracy through ensemble methods.
"""

from typing import Dict, Optional, List, Callable
import numpy as np
import torch
from torch_geometric.data import Data

from .base import AbstractHyperNet
from .main import HyperNet


class HyperNetEnsemble(AbstractHyperNet):
    """
    An ensemble of HyperNet models that aggregates predictions through majority voting.

    This class combines multiple HyperNet models with the same dimensions to improve
    encoding and decoding accuracy through ensemble methods. The ensemble performs:

    - **Forward pass**: Stacks embeddings from all models
    - **Distance calculation**: Returns mean distance across all models
    - **Decoding**: Uses majority voting to select constraints that appear in >50% of models

    **Design Rationale**

    Ensemble methods can improve robustness and accuracy by combining predictions from
    multiple models. Different models may have different random initializations or
    training histories, leading to complementary strengths. By requiring majority
    agreement, the ensemble filters out spurious predictions that only appear in
    individual models.

    **Usage Example**

    .. code-block:: python

        from graph_hdc.models import HyperNet, HyperNetEnsemble
        from graph_hdc.utils import CategoricalIntegerEncoder

        # Create multiple HyperNet models with same configuration
        model1 = HyperNet(
            hidden_dim=1000,
            depth=3,
            node_encoder_map={'node_atoms': CategoricalIntegerEncoder(dim=1000, num_categories=10)},
            seed=42
        )

        model2 = HyperNet(
            hidden_dim=1000,
            depth=3,
            node_encoder_map={'node_atoms': CategoricalIntegerEncoder(dim=1000, num_categories=10)},
            seed=123
        )

        # Create ensemble
        ensemble = HyperNetEnsemble([model1, model2])

        # Forward pass returns stacked embeddings
        result = ensemble.forward(data)
        stacked_embedding = result['graph_embedding']  # Shape: (2, batch_size, 1000)

        # Distance calculation uses mean of individual distances
        dist = ensemble.get_distance(hv1, hv2)

        # Decoding uses majority voting
        node_constraints = ensemble.decode_order_zero(stacked_embedding)
        edge_constraints = ensemble.decode_order_one(stacked_embedding)

    :param hyper_nets: List of HyperNet models to ensemble (must have same hidden_dim)
    :type hyper_nets: List[HyperNet]
    :param distance_func: Optional distance function to use (defaults to cosine_distance)
    :type distance_func: Optional[Callable]

    **Note on Model Depths**

    Models can have different depths. When depths differ, the `graph_hv_stack` will not be
    included in the forward pass results (since tensors of different shapes cannot be stacked).
    Only the final `graph_embedding` will be returned in this case.
    """

    def __init__(
        self,
        hyper_nets: List[HyperNet],
        distance_func: Optional[Callable] = None,
        **kwargs
    ):
        """
        Initialize the HyperNetEnsemble with a list of HyperNet models.

        This method validates that all models have the same hidden_dim.
        Models can have different depths - when depths differ, graph_hv_stack
        will not be included in forward pass results.

        :param hyper_nets: List of HyperNet models to ensemble
        :type hyper_nets: List[HyperNet]
        :param distance_func: Optional distance function (defaults to cosine_distance)
        :type distance_func: Optional[Callable]

        :raises ValueError: If models have different hidden_dim
        :raises ValueError: If hyper_nets list is empty
        """
        if not hyper_nets:
            raise ValueError("hyper_nets list cannot be empty")

        # Validate that all models have the same hidden_dim
        # Note: depths can differ, but hidden_dim must match for stacking embeddings
        first_model = hyper_nets[0]
        self.hidden_dim = first_model.hidden_dim
        self.num_models = len(hyper_nets)

        for i, model in enumerate(hyper_nets[1:], start=1):
            if model.hidden_dim != self.hidden_dim:
                raise ValueError(
                    f"Model {i} has hidden_dim={model.hidden_dim}, "
                    f"expected {self.hidden_dim}"
                )

        # Check if all models have the same depth (for graph_hv_stack stacking)
        depths = [model.depth for model in hyper_nets]
        self.uniform_depth = all(d == depths[0] for d in depths)
        if self.uniform_depth:
            self.depth = depths[0]
        else:
            # Store depths individually if they differ
            self.depths = depths
            self.depth = None  # No single depth value

        # Store the models
        self.hyper_nets = hyper_nets

        # Set distance function
        if distance_func is not None:
            from graph_hdc.functions import resolve_function
            if isinstance(distance_func, str):
                self.distance_func = resolve_function(distance_func)
            else:
                self.distance_func = distance_func
        else:
            # Use default cosine distance
            from graph_hdc.reconstruct import cosine_distance
            self.distance_func = cosine_distance

        # Initialize parent class
        super().__init__(**kwargs)

    def forward(self, data: Data) -> dict:
        """
        Performs a forward pass on all models and stacks the results.

        This method calls forward() on each model in the ensemble with the same
        input data and stacks all the graph embeddings into a single tensor of
        shape (num_models, batch_size, hidden_dim).

        **Example**

        .. code-block:: python

            ensemble = HyperNetEnsemble([model1, model2, model3])
            result = ensemble.forward(data)

            # Stacked embeddings from 3 models
            embeddings = result['graph_embedding']  # Shape: (3, batch_size, hidden_dim)

            # Extract individual model embeddings
            model1_embedding = embeddings[0]  # Shape: (batch_size, hidden_dim)
            model2_embedding = embeddings[1]  # Shape: (batch_size, hidden_dim)

        :param data: PyG Data object representing batched graphs
        :type data: torch_geometric.data.Data

        :return: Dictionary containing:
            - 'graph_embedding': Stacked embeddings (num_models, batch_size, hidden_dim)
            - 'graph_hv_stack': Stacked embedding stacks (num_models, batch_size, depth+1, hidden_dim)
        :rtype: Dict[str, torch.Tensor]
        """
        # Collect embeddings from all models
        graph_embeddings = []
        graph_hv_stacks = []

        for model in self.hyper_nets:
            result = model.forward(data)
            graph_embeddings.append(result['graph_embedding'])

            if 'graph_hv_stack' in result:
                graph_hv_stacks.append(result['graph_hv_stack'])

        # Stack all embeddings
        # graph_embeddings shape: (num_models, batch_size, hidden_dim)
        stacked_embeddings = torch.stack(graph_embeddings, dim=0)

        result_dict = {
            'graph_embedding': stacked_embeddings,
        }

        # Only stack graph_hv_stacks if all models have the same depth
        # (otherwise they have incompatible shapes and cannot be stacked)
        if graph_hv_stacks and self.uniform_depth:
            # graph_hv_stacks shape: (num_models, batch_size, depth+1, hidden_dim)
            result_dict['graph_hv_stack'] = torch.stack(graph_hv_stacks, dim=0)

        return result_dict

    def extract_graph_results(
        self,
        data: Data,
        graph_results: Dict[str, torch.Tensor],
    ) -> List[Dict[str, np.ndarray]]:
        """
        Extract individual graph results from batched ensemble results.

        This method overrides the parent implementation to handle stacked embeddings
        from the ensemble. The ensemble's forward method returns tensors with shape
        (num_models, batch_size, ...) rather than (batch_size, ...), so we need to
        handle the extraction differently.

        **Algorithm**

        1. Determine batch size from data object
        2. For each graph in the batch:
           - Extract graph-level properties by indexing the model dimension first
           - Convert stacked tensors (num_models, ...) to numpy arrays
        3. Return list of result dicts, one per graph

        :param data: The PyG Data object that represents the batch of graphs
        :type data: Data
        :param graph_results: The dictionary that contains the ensemble results
        :type graph_results: Dict[str, torch.Tensor]

        :return: List of dictionaries, one per graph in the batch
        :rtype: List[Dict[str, np.ndarray]]
        """
        # The batch size as calculated from the data object
        batch_size = int(torch.max(data.batch).detach().item()) + 1

        # In this list we will store the disentangled results for each graph
        results: List[Dict[str, np.ndarray]] = []

        for index in range(batch_size):
            result: Dict[str, np.ndarray] = {}

            for key, tens in graph_results.items():
                if key.startswith('graph'):
                    # For ensemble, graph tensors have shape (num_models, batch_size, ...)
                    # We need to extract [:, index, :] to get (num_models, ...)
                    if tens.dim() >= 2:
                        result[key] = tens[:, index].cpu().detach().numpy()
                    else:
                        # Fallback for unexpected shapes
                        result[key] = tens.cpu().detach().numpy()

                # Note: We don't handle node_ or edge_ prefixes since ensemble
                # only returns graph-level embeddings

            results.append(result)

        return results

    def extract_distance_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Extract distance embeddings from each model in the ensemble.

        This method applies extract_distance_embedding to each model's embedding
        and returns the stacked results. For models like CompositeHyperNet, this
        extracts only the relevant component (e.g., the g component) for distance
        calculations during reconstruction.

        **Example**

        .. code-block:: python

            ensemble = HyperNetEnsemble([model1, model2, model3])
            result = ensemble.forward(data)
            stacked_embedding = result['graph_embedding']  # (3, batch_size, hidden_dim)

            # Extract distance embeddings for each model
            dist_embeddings = ensemble.extract_distance_embedding(stacked_embedding)
            # Shape: (3, batch_size, distance_dim)

        :param embedding: Stacked embeddings from forward pass (num_models, ..., hidden_dim)
        :type embedding: torch.Tensor

        :return: Stacked distance embeddings (num_models, ..., distance_dim)
        :rtype: torch.Tensor
        """
        # Convert numpy array to torch tensor if needed
        if isinstance(embedding, np.ndarray):
            embedding = torch.from_numpy(embedding)

        # Extract distance embeddings from each model
        distance_embeddings = []

        for i, model in enumerate(self.hyper_nets):
            # Extract i-th embedding for i-th model
            model_embedding = embedding[i]

            # Apply model's extract_distance_embedding method
            dist_emb = model.extract_distance_embedding(model_embedding)
            distance_embeddings.append(dist_emb)

        # Stack all distance embeddings
        # Shape: (num_models, ..., distance_dim)
        return torch.stack(distance_embeddings, dim=0)

    def get_distance(self, hv1: torch.Tensor, hv2: torch.Tensor) -> float:
        """
        Calculate the mean distance between two stacked hypervectors.

        This method computes the distance between corresponding hypervectors
        from each model and returns the mean of all individual distances.

        **Example**

        .. code-block:: python

            # hv1 and hv2 are stacked embeddings from forward pass
            hv1 = result1['graph_embedding']  # Shape: (num_models, batch_size, hidden_dim)
            hv2 = result2['graph_embedding']  # Shape: (num_models, batch_size, hidden_dim)

            # Calculate mean distance across all models
            mean_dist = ensemble.get_distance(hv1, hv2)

        :param hv1: First stacked hypervector (num_models, ..., hidden_dim)
        :type hv1: torch.Tensor
        :param hv2: Second stacked hypervector (num_models, ..., hidden_dim)
        :type hv2: torch.Tensor

        :return: Mean distance across all models
        :rtype: float
        """
        distances = []

        for i, model in enumerate(self.hyper_nets):
            # Extract i-th embedding for i-th model and squeeze batch dimension
            model_hv1 = hv1[i].squeeze()
            model_hv2 = hv2[i].squeeze()

            dist = model.get_distance(model_hv1, model_hv2)
            distances.append(dist)

        # Return mean of all distances
        return float(np.mean(distances))

    @staticmethod
    def _constraint_to_key(constraint: dict, include_num: bool = False) -> tuple:
        """
        Convert a constraint dict to a hashable key for grouping.

        This helper method creates a canonical representation of a constraint
        that can be used as a dictionary key for grouping and counting.

        :param constraint: Constraint dict with 'src' and optionally 'dst' and 'num'
        :type constraint: dict
        :param include_num: Whether to include the 'num' field in the key
        :type include_num: bool

        :return: Hashable tuple representation of the constraint
        :rtype: tuple
        """
        parts = []

        # Add src component
        if 'src' in constraint:
            src_items = tuple(sorted(constraint['src'].items()))
            parts.append(('src', src_items))

        # Add dst component for order-one constraints
        if 'dst' in constraint:
            dst_items = tuple(sorted(constraint['dst'].items()))
            parts.append(('dst', dst_items))

        # Optionally include num
        if include_num and 'num' in constraint:
            parts.append(('num', constraint['num']))

        return tuple(parts)

    @staticmethod
    def _key_to_constraint(key: tuple) -> dict:
        """
        Convert a hashable key back to a constraint dict.

        This is the inverse of _constraint_to_key().

        :param key: Hashable tuple from _constraint_to_key()
        :type key: tuple

        :return: Reconstructed constraint dict
        :rtype: dict
        """
        constraint = {}

        for field_name, field_value in key:
            if field_name == 'src':
                constraint['src'] = dict(field_value)
            elif field_name == 'dst':
                constraint['dst'] = dict(field_value)
            elif field_name == 'num':
                constraint['num'] = field_value

        return constraint

    def _majority_vote_constraints(
        self,
        all_constraints: List[List[dict]],
        threshold: float = 0.5
    ) -> List[dict]:
        """
        Perform majority voting on constraints from multiple models.

        This method groups constraints by their type (same src/dst properties)
        and only includes those that appear in more than threshold * num_models.
        For constraints that pass the threshold, the 'num' field is set to the
        median value from all models that included this constraint.

        **Algorithm**

        1. Group all constraints by their src/dst properties (ignoring num)
        2. Count how many models include each constraint type
        3. Filter to only constraints appearing in > threshold * num_models
        4. For each passing constraint, set num to median of all num values

        :param all_constraints: List of constraint lists from each model
        :type all_constraints: List[List[dict]]
        :param threshold: Minimum fraction of models that must include constraint (default: 0.5)
        :type threshold: float

        :return: Filtered list of constraints with majority agreement
        :rtype: List[dict]
        """
        # Group constraints by their type (ignoring num)
        # constraint_key -> List[constraint_dicts]
        grouped_constraints = {}

        for model_constraints in all_constraints:
            for constraint in model_constraints:
                key = self._constraint_to_key(constraint, include_num=False)

                if key not in grouped_constraints:
                    grouped_constraints[key] = []

                grouped_constraints[key].append(constraint)

        # Filter to only constraints with majority agreement
        majority_constraints = []
        min_count = int(np.ceil(threshold * self.num_models))

        for key, constraint_list in grouped_constraints.items():
            # Check if this constraint appears in enough models
            if len(constraint_list) >= min_count:
                # Reconstruct the constraint dict
                constraint = self._key_to_constraint(key)

                # Set num to median of all num values
                num_values = [c['num'] for c in constraint_list]
                constraint['num'] = int(np.median(num_values))

                majority_constraints.append(constraint)

        return majority_constraints

    def decode_order_zero(
        self,
        embedding,  # Can be torch.Tensor or np.ndarray
        iterations: int = 1
    ) -> List[dict]:
        """
        Decode node information using majority voting across all models.

        This method extracts node constraints from each model in the ensemble
        and returns only those constraints that appear in more than 50% of models.

        **Example**

        .. code-block:: python

            ensemble = HyperNetEnsemble([model1, model2, model3])
            result = ensemble.forward(data)
            stacked_embedding = result['graph_embedding']  # (3, batch_size, hidden_dim)

            # Decode with majority voting
            node_constraints = ensemble.decode_order_zero(stacked_embedding)

            # Only constraints appearing in >= 2 out of 3 models are included

        :param embedding: Stacked embeddings from forward pass (num_models, batch_size, hidden_dim)
            Can be torch.Tensor or np.ndarray
        :param iterations: Number of decoding iterations (passed to individual models)
        :type iterations: int

        :return: List of node constraints with majority agreement
        :rtype: List[dict]
        """
        # Convert numpy array to torch tensor if needed
        if isinstance(embedding, np.ndarray):
            embedding = torch.from_numpy(embedding)

        # Collect constraints from all models
        all_constraints = []

        for i, model in enumerate(self.hyper_nets):
            # Extract embedding for this model
            model_embedding = embedding[i]

            # Decode with this model
            constraints = model.decode_order_zero(model_embedding, iterations=iterations)
            all_constraints.append(constraints)

        # Perform majority voting
        return self._majority_vote_constraints(all_constraints)

    def decode_order_one(
        self,
        embedding,  # Can be torch.Tensor or np.ndarray
        constraints_order_zero: Optional[List[dict]] = None,
        **kwargs
    ) -> List[dict]:
        """
        Decode edge information using majority voting across all models.

        This method extracts edge constraints from each model in the ensemble
        and returns only those constraints that appear in more than 50% of models.

        **Example**

        .. code-block:: python

            ensemble = HyperNetEnsemble([model1, model2, model3])
            result = ensemble.forward(data)
            stacked_embedding = result['graph_embedding']

            # Decode nodes first (optional)
            node_constraints = ensemble.decode_order_zero(stacked_embedding)

            # Decode edges with majority voting
            edge_constraints = ensemble.decode_order_one(
                stacked_embedding,
                node_constraints
            )

        :param embedding: Stacked embeddings from forward pass (num_models, batch_size, hidden_dim)
            Can be torch.Tensor or np.ndarray
        :param constraints_order_zero: Pre-computed node constraints (optional)
        :type constraints_order_zero: Optional[List[dict]]
        :param kwargs: Additional arguments passed to individual models

        :return: List of edge constraints with majority agreement
        :rtype: List[dict]
        """
        # Convert numpy array to torch tensor if needed
        if isinstance(embedding, np.ndarray):
            embedding = torch.from_numpy(embedding)

        # Collect constraints from all models
        all_constraints = []

        for i, model in enumerate(self.hyper_nets):
            # Extract embedding for this model
            model_embedding = embedding[i]

            # Decode with this model
            constraints = model.decode_order_one(
                model_embedding,
                constraints_order_zero=constraints_order_zero,
                **kwargs
            )
            all_constraints.append(constraints)

        # Perform majority voting
        return self._majority_vote_constraints(all_constraints)

    def save_to_path(self, path: str) -> None:
        """
        Save the ensemble configuration and models to disk.

        This method saves the ensemble structure by saving each individual model
        to a separate file and creating a metadata file with the ensemble configuration.

        :param path: Base path for saving (will create multiple files)
        :type path: str

        :returns: None
        """
        import os
        import json

        # Create directory if it doesn't exist
        base_dir = os.path.dirname(path)
        if base_dir and not os.path.exists(base_dir):
            os.makedirs(base_dir)

        # Save each model
        model_paths = []
        for i, model in enumerate(self.hyper_nets):
            model_path = f"{path}_model_{i}.json"
            model.save_to_path(model_path)
            model_paths.append(model_path)

        # Save ensemble metadata
        metadata = {
            'num_models': self.num_models,
            'hidden_dim': self.hidden_dim,
            'uniform_depth': self.uniform_depth,
            'model_paths': model_paths,
        }

        # Include depth info based on whether depths are uniform
        if self.uniform_depth:
            metadata['depth'] = self.depth
        else:
            metadata['depths'] = self.depths

        with open(path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def load_from_path(self, path: str) -> None:
        """
        Load the ensemble configuration and models from disk.

        This method loads the ensemble by reading the metadata file and loading
        each individual model from its saved file.

        :param path: Path to the ensemble metadata file
        :type path: str

        :returns: None
        """
        import json

        # Load metadata
        with open(path, 'r') as f:
            metadata = json.load(f)

        # Load each model
        hyper_nets = []
        for model_path in metadata['model_paths']:
            model = HyperNet.load(model_path)
            hyper_nets.append(model)

        # Update instance attributes
        self.hyper_nets = hyper_nets
        self.num_models = metadata['num_models']
        self.hidden_dim = metadata['hidden_dim']
        self.uniform_depth = metadata['uniform_depth']

        # Restore depth info based on whether depths are uniform
        if self.uniform_depth:
            self.depth = metadata['depth']
        else:
            self.depths = metadata['depths']
            self.depth = None
