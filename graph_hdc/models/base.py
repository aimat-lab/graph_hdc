"""
Base classes for hyperdimensional computing models.

This module provides the abstract base class for all HyperNet implementations.
"""

from typing import Dict, Optional, Callable, Any, List
import torch
import pytorch_lightning as pl
from torch_geometric.data import Data
import numpy as np


# === HYPERDIMENSIONAL MESSAGE PASSING NETWORKS ===

class AbstractHyperNet(pl.LightningModule):

    def __init__(self, **kwargs):
        pl.LightningModule.__init__(self)

    def forward_graphs(self,
                       graphs: List[dict],
                       batch_size: int = 600,
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
        from graph_hdc.graph import data_list_from_graph_dicts
        from torch_geometric.loader import DataLoader

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

    def forward_graph(self, graph: dict) -> Dict[str, np.ndarray]:
        """
        Given a single ``graph`` dict representation, this method will run the hypernet forward pass
        on that graph and return a dictionary containing the result.

        This is a convenience method that simply calls forward_graphs with a single-element list
        and returns the first (and only) result.

        :param graph: A graph dict representation containing the information about the nodes,
            edges, and properties of the graph.

        :returns: A result dictionary containing the same string keys as the result of the
            "forward" method but with numpy array values for the single graph.
        """
        results = self.forward_graphs([graph])
        return results[0]

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

    def get_distance(self, hv1: torch.Tensor, hv2: torch.Tensor) -> float:
        """
        Calculate the distance between two hypervectors.

        This method provides a unified interface for computing distances between hyperdimensional
        vectors. Different HyperNet implementations can use different distance metrics by overriding
        this method or by configuring a distance function during initialization.

        The default implementation uses cosine distance, which is defined as 1 - cosine_similarity.
        This is a common choice for hyperdimensional computing as it measures angular distance
        between vectors in high-dimensional space.

        **Design Rationale**

        Encapsulating distance calculation within the HyperNet class allows each encoder to define
        its own notion of similarity. This is particularly useful during graph reconstruction where
        the same distance metric should be consistently applied when comparing embeddings.

        **Example Usage**

        .. code-block:: python

            encoder = HyperNet(hidden_dim=1000, ...)

            # Encode two graphs
            result1 = encoder.forward(data1)
            result2 = encoder.forward(data2)

            emb1 = result1['graph_embedding']
            emb2 = result2['graph_embedding']

            # Calculate distance using encoder's distance metric
            distance = encoder.get_distance(emb1, emb2)

        **Distance Metrics**

        Common distance functions include:
        - Cosine distance: 1 - cosine_similarity (default)
        - Euclidean distance: L2 norm of difference
        - Manhattan distance: L1 norm of difference
        - Dot product distance: negative dot product

        :param hv1: First hypervector with shape (hidden_dim,) or (batch_size, hidden_dim)
        :type hv1: torch.Tensor
        :param hv2: Second hypervector with shape (hidden_dim,) or (batch_size, hidden_dim)
        :type hv2: torch.Tensor

        :return: Distance value as a float. Lower values indicate greater similarity.
        :rtype: float
        """
        # Default implementation uses cosine distance
        # Import here to avoid circular imports
        from graph_hdc.reconstruct import cosine_distance
        return cosine_distance(hv1, hv2)
