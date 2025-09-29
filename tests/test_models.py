import os
import tempfile

import torch
import numpy as np
import jsonpickle
from rich.pretty import pprint
from torch.nn.functional import normalize
from torch_geometric.loader import DataLoader

import graph_hdc.utils
from graph_hdc.models import AbstractEncoder
from graph_hdc.models import CategoricalOneHotEncoder
from graph_hdc.models import CategoricalIntegerEncoder
from graph_hdc.models import HyperNet
from graph_hdc.binding import circular_convolution_fft, circular_correlation_fft
from graph_hdc.testing import generate_random_graphs
from graph_hdc.graph import data_list_from_graph_dicts


class TestCategoricalOneHotEncoder:
    
    def test_construction_basically_works(self):
        """
        If an encoder instance can be constructed without error
        """
        encoder = CategoricalOneHotEncoder(
            dim=1000,
            num_categories=3,
        )
        assert isinstance(encoder, AbstractEncoder)
        assert encoder.dim == 1000
        assert isinstance(encoder.embeddings, torch.Tensor)
        assert encoder.embeddings.shape == (3, 1000)
        
        
    def test_seeding_basically_works(self):
        """
        When setting an explicit seed, the encoder should produce the same result whenever the same 
        seed is chosen.
        """
        encoder1 = CategoricalOneHotEncoder(
            dim=1000,
            num_categories=3,
            seed=1,
        )
        assert encoder1.embeddings.shape == (3, 1000)
        
        encoder2 = CategoricalOneHotEncoder(
            dim=1000,
            num_categories=3,
            seed=1,
        )
        assert encoder1 != encoder2
        assert torch.allclose(encoder1.embeddings, encoder2.embeddings)
        
    def test_encode_decode_basically_works(self):
        """
        The "encode" method should take a one-hot encoded vector and return the corresponding random hv embedding.
        The "decode" method takes the hv vector and returns the one-hot index that best (!) matches that given 
        embedding.
        """
        value1 = torch.tensor([1, 0, 0])
        value2 = torch.tensor([0, 0, 1])
        
        encoder = CategoricalOneHotEncoder(
            dim=2000,
            num_categories=3,
        )
        encoded1 = encoder.encode(value1)
        assert isinstance(encoded1, torch.Tensor)
        assert torch.allclose(encoded1, encoder.embeddings[0])
        decoded1 = encoder.decode(encoded1)
        assert decoded1 == (1, 0, 0)
        
        encoded2 = encoder.encode(value2)
        assert isinstance(encoded2, torch.Tensor)
        assert torch.allclose(encoded2, encoder.embeddings[2])
        decoded2 = encoder.decode(encoded2)
        assert decoded2 == (0, 0, 1)
        
    def test_save_load_basically_works(self):
        """
        The encoder should be able to be exported and imported to a file using the jsonpickle 
        library
        """
        encoder = CategoricalOneHotEncoder(
            dim=3000,
            num_categories=3,
            seed=1,
        )
        
        content = jsonpickle.dumps(encoder)
        print(content)
        assert isinstance(content, str)
        
        encoder_loaded = jsonpickle.loads(content)
        assert isinstance(encoder_loaded, AbstractEncoder)
        assert torch.allclose(encoder.embeddings, encoder_loaded.embeddings) 
        
        
class TestHyperNet: 
    """
    Unittests for the HyperNet class.
    """
    
    def test_construction_basically_works(self):
        """
        If a new HyperNet object can be constructed without error.
        """
        
        dim = 1000
        hyper_net = HyperNet(
            hidden_dim=dim,
            depth=3,
            node_encoder_map={
                'type': CategoricalOneHotEncoder(dim, 3),
            }
        )
        assert isinstance(hyper_net, HyperNet)
        assert hyper_net.hidden_dim == dim
        assert hyper_net.depth == 3
        assert isinstance(hyper_net.node_encoder_map, dict)
        
    def test_saving_loading_works(self):
        """
        It should be possible to use the "save_to_path" and "load_from_path" methods to save and load the 
        an instance of a HyperNet object to and from a file.
        """
        
        dim = 1000
        type_encoder = CategoricalOneHotEncoder(dim, 3)
        size_encoder = CategoricalOneHotEncoder(dim, 2)
        hyper_net = HyperNet(
            hidden_dim=dim,
            depth=3,
            node_encoder_map={
                'type': type_encoder,
            },
            graph_encoder_map={
                'size': size_encoder,
            }
        )
        
        with tempfile.TemporaryDirectory() as path:
            model_path = os.path.join(path, 'model.pt')
            hyper_net.save_to_path(model_path)
            
            assert os.path.exists(model_path)
            
            hyper_net_loaded = HyperNet(100, 2, {'1': CategoricalOneHotEncoder(100, 2)})
            hyper_net_loaded.load_from_path(model_path)
            
            assert isinstance(hyper_net_loaded, HyperNet)
            assert hyper_net_loaded.hidden_dim == hyper_net.hidden_dim
            assert hyper_net_loaded.depth == hyper_net.depth
            assert isinstance(hyper_net_loaded.node_encoder_map, dict)
            
            # The encoder should be loaded as well and should be the exact same as before!
            assert isinstance(hyper_net_loaded.node_encoder_map['type'], AbstractEncoder)
            assert torch.allclose(hyper_net_loaded.node_encoder_map['type'].embeddings, type_encoder.embeddings)
            
            # The same for the graph encoder map
            assert isinstance(hyper_net_loaded.graph_encoder_map['size'], AbstractEncoder)
            assert torch.allclose(hyper_net_loaded.graph_encoder_map['size'].embeddings, size_encoder.embeddings)
            
    def test_encode_properties_basically_works(self):
        """
        The HyperNet.encode_properties method should apply the initial encoding of the generic graph data into 
        the node and graph hypervectors on top of which the message passing is then performed.
        """
        num_graphs = 10
        graphs = generate_random_graphs(
            num_graphs=num_graphs, 
            num_node_features=3,
            num_graph_labels=2,    
        )
            
        dim = 1000
        hyper_net = HyperNet(
            hidden_dim=dim,
            depth=3,
            node_encoder_map={
                'x': CategoricalOneHotEncoder(dim, 3),
            },
            graph_encoder_map={
                'y': CategoricalOneHotEncoder(dim, 2),
            }
        )
        
        data_list = data_list_from_graph_dicts(graphs)
        data_loader = DataLoader(data_list, batch_size=num_graphs, shuffle=False)
        data = next(iter(data_loader))
            
        data = hyper_net.encode_properties(data)
        
        assert hasattr(data, 'node_hv')
        assert isinstance(data.node_hv, torch.Tensor)
        assert data.node_hv.shape == (data.x.size(0), dim)
        
        assert hasattr(data, 'graph_hv')
        assert isinstance(data.graph_hv, torch.Tensor)
        assert data.graph_hv.shape == (num_graphs, dim)
            
    def test_forward_basically_works(self):
        """
        A forward pass of the HyperNet model should take the PyG Data object and return a dictionary which 
        most prominently includes the high-dimensional vector representation of the graph.
        """
        # generating mock data
        num_graphs = 10
        graphs = generate_random_graphs(10, num_node_features=3)
        
        # setting up model
        dim = 1000
        hyper_net = HyperNet(
            hidden_dim=dim,
            depth=3,
            node_encoder_map={
                'x': CategoricalOneHotEncoder(dim, 3),
            }
        )
        
        # converting graphs to pyg data object
        data_list = data_list_from_graph_dicts(graphs)
        data_loader = DataLoader(data_list, batch_size=num_graphs, shuffle=False)
        data = next(iter(data_loader))
        
        # forward pass
        result: dict = hyper_net.forward(data)
        embedding = result['graph_embedding']
        
        assert isinstance(result, dict)
        assert isinstance(embedding, torch.Tensor)
        assert embedding.shape == (num_graphs, dim)
        
    def test_gradient_for_edge_weights_basically_works(self):
        """
        It should be possible to pass the additional "edge_weight" property which does then get a gradient 
        assigned to it. 
        """
        # generating mock data
        num_graphs = 10
        graphs = generate_random_graphs(num_graphs, num_node_features=3)
        
        # setting up model
        dim = 10_000
        hyper_net = HyperNet(
            hidden_dim=dim,
            depth=3,
            node_encoder_map={
                'x': CategoricalOneHotEncoder(dim, 3),
            }
        )
        
        # converting graphs to pyg data object
        data_list = data_list_from_graph_dicts(graphs)
        data_loader = DataLoader(data_list, batch_size=num_graphs, shuffle=False)
        data = next(iter(data_loader))
        data.edge_weight.requires_grad = True
        
        # forward pass
        result: dict = hyper_net.forward(data)
        embedding = result['graph_embedding']
        
        # defining a loss based on the embedding proximity to a random vector
        graphs_ = generate_random_graphs(num_graphs, num_node_features=3)
        data_list_ = data_list_from_graph_dicts(graphs_)
        data_loader_ = DataLoader(data_list_, batch_size=num_graphs, shuffle=False)
        data_ = next(iter(data_loader_))
        
        result: dict = hyper_net.forward(data_)
        target = result['graph_embedding']
        loss = ((target - embedding).pow(2).mean(dim=1)).mean()
        loss.backward()
        
        # checking if the gradients on the edge weights exist
        print(data.edge_weight.grad)
        print(loss)
        assert data.edge_weight.grad is not None
        
    def test_recovering_nodes_from_embedding(self):
        """
        Does not test any particular method but generally tests the method by which it should be possible 
        to recover/decode individual nodes from the overall graph embedding. This should work by multiplying 
        the graph embedding with the node encodings and if the result is close to zero, then the node was 
        not present, otherwise the result should be proportional to the number of times the node was present.
        """
        # setting up model
        dim = 10_000
        encoder = CategoricalIntegerEncoder(dim, 5)
        hyper_net = HyperNet(
            hidden_dim=dim,
            depth=2,
            node_encoder_map={
                'x': encoder,
            },
            bind_fn=circular_convolution_fft,
            unbind_fn=circular_correlation_fft,
        )
        print('encoder_hv_dict', encoder.get_encoder_hv_dict())
        
        # setting up simple test graph
        graph = {
            'node_indices': np.array([0, 1, 2, 3], dtype=int),
            'node_attributes': np.array([[0], [0], [1], [2]], dtype=float),
            'edge_indices': np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=int),
            'edge_attributes': np.array([[1], [1], [1]], dtype=float),
        }
        data_list = data_list_from_graph_dicts([graph])
        data = next(iter(DataLoader(data_list, batch_size=1)))

        # forward pass - encoding into embedding
        result: dict = hyper_net.forward(data)
        embedding = result['graph_embedding']
        
        assert isinstance(embedding, torch.Tensor)
        assert embedding.shape == (1, dim)
        
        # Compute and print the matrix multiplication between node encodings and graph embedding
        node_encodings = encoder.embeddings
        graph_embedding = embedding.squeeze(0)  # Remove batch dimension
        
        matmul_result = torch.matmul(node_encodings, graph_embedding)
        print('nodes', matmul_result)
        assert np.allclose(matmul_result, [2, 1, 1, 0, 0], atol=0.2)
        
        edge_encodings = torch.stack([
            circular_convolution_fft(node_encodings[0], node_encodings[0]),
            circular_convolution_fft(node_encodings[0], node_encodings[1]),
            circular_convolution_fft(node_encodings[1], node_encodings[3]),
        ])
        matmul_result = torch.matmul(edge_encodings, graph_embedding)
        print('edges', matmul_result)
        
    def test_decode_order_zero(self):
        """
        The decode_order_zero method should return a number of constraints which define the types of nodes that were
        part of the original graph and also the node properties and the number of times a node of that type was
        present.
        """
        # setting up model
        dim = 10_000
        encoder = CategoricalIntegerEncoder(dim, 5)
        hyper_net = HyperNet(
            hidden_dim=dim,
            depth=3,
            node_encoder_map={
                'x': encoder,
            },
            bind_fn=circular_convolution_fft,
            unbind_fn=circular_correlation_fft,
        )
        
        # setting up simple test graph
        graph = {
            'node_indices': np.array([0, 1, 2, 3], dtype=int),
            'node_attributes': np.array([[0], [0], [1], [2]], dtype=float),
            'edge_indices': np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=int),
            'edge_attributes': np.array([[1], [1], [1]], dtype=float),
        }
        data_list = data_list_from_graph_dicts([graph])
        data = next(iter(DataLoader(data_list, batch_size=1)))

        # forward pass - encoding into embedding
        result: dict = hyper_net.forward(data)
        embedding = result['graph_embedding']
        
        # zero order decoding
        # This method should return a number of constraints which define the types of nodes that were 
        # part of the original graph and the number of nodes of each type.
        
        constraints = hyper_net.decode_order_zero(embedding)
        pprint(constraints)
        assert isinstance(constraints, list)
        target_constraints = [
            {'src': {'x': 0}, 'num': 2},
            {'src': {'x': 1}, 'num': 1},
            {'src': {'x': 2}, 'num': 1},
        ]
        assert constraints == target_constraints
        
    def test_decode_order_one(self):
        """
        The decode_order_one method should return a number of constraints which define the kinds of edges that were
        part of the original graph and the number of how many of that type of edge exists.
        """
        # setting up model
        dim = 10_000
        encoder = CategoricalIntegerEncoder(dim, 5)
        hyper_net = HyperNet(
            hidden_dim=dim,
            depth=3,
            node_encoder_map={
                'x': encoder,
            },
            bind_fn=circular_convolution_fft,
            unbind_fn=circular_correlation_fft,
        )
        
        # setting up simple test graph
        graph = {
            'node_indices': np.array([0, 1, 2, 3], dtype=int),
            'node_attributes': np.array([[0], [0], [1], [2]], dtype=float),
            'edge_indices': np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=int),
            'edge_attributes': np.array([[1], [1], [1]], dtype=float),
        }
        data_list = data_list_from_graph_dicts([graph])
        data = next(iter(DataLoader(data_list, batch_size=1)))

        # forward pass - encoding into embedding
        result: dict = hyper_net.forward(data)
        embedding = result['graph_embedding']
        
        # zero order decoding
        # This method should return a number of constraints which define the types of nodes that were 
        # part of the original graph and the number of nodes of each type.
        constraints_order_zero = hyper_net.decode_order_zero(embedding)
        
        # first order decoding
        # This method should return a number of constraints which define the kinds of edges that were 
        # part of the original graph and the number of how many of that type of edge exists.
        constraints_order_one = hyper_net.decode_order_one(
            embedding=embedding,
            constraints_order_zero=constraints_order_zero,
        )
        pprint(constraints_order_one)
        assert isinstance(constraints_order_one, list)
        assert len(constraints_order_one) > 0
        for constraint in constraints_order_one:
            assert 'src' in constraint
            assert 'dst' in constraint
            assert 'num' in constraint

    def test_decode_nodes(self):
        """
        The decode_nodes method should return a graph dict with properly formatted node information
        including node_indices, full connectivity edges, and node properties as arrays.
        """
        # setting up model
        dim = 10_000
        encoder = CategoricalIntegerEncoder(dim, 5)
        hyper_net = HyperNet(
            hidden_dim=dim,
            depth=3,
            node_encoder_map={
                'x': encoder,
            },
            bind_fn=circular_convolution_fft,
            unbind_fn=circular_correlation_fft,
        )

        # setting up simple test graph with 4 nodes: 2 of type 0, 1 of type 1, 1 of type 2
        graph = {
            'node_indices': np.array([0, 1, 2, 3, 4], dtype=int),
            'node_attributes': np.array([[0], [0], [1], [2], [2]], dtype=float),
            'edge_indices': np.array([[0, 1], [1, 2], [2, 3], [3, 0], [4, 1]], dtype=int),
            'edge_attributes': np.array([[1], [1], [1], [1], [1]], dtype=float),
        }
        data_list = data_list_from_graph_dicts([graph])
        data = next(iter(DataLoader(data_list, batch_size=1)))

        # forward pass - encoding into embedding
        result: dict = hyper_net.forward(data)
        embedding = result['graph_embedding']

        # decode nodes into graph dict format
        decoded_graph = hyper_net.decode_nodes(embedding)
        pprint(decoded_graph)

        assert isinstance(decoded_graph, dict)

        # Check required keys exist
        assert 'node_indices' in decoded_graph
        assert 'edge_index_full' in decoded_graph
        assert 'edge_weight_full' in decoded_graph
        assert 'x' in decoded_graph  # Should have the node property

        # Check node_indices
        node_indices = decoded_graph['node_indices']
        assert isinstance(node_indices, np.ndarray)
        assert len(node_indices) == 5  # Should have 4 total nodes (2+1+1)
        assert np.array_equal(node_indices, np.arange(5))

        # Check node properties
        x_values = decoded_graph['x']
        assert isinstance(x_values, np.ndarray)
        assert len(x_values) == 5
        # Should have 2 nodes of type 0, 1 of type 1, 1 of type 2
        unique_values, counts = np.unique(x_values, return_counts=True)
        expected_counts = {0: 2, 1: 1, 2: 2}
        for val, count in zip(unique_values, counts):
            assert expected_counts[val] == count

        # Check full connectivity edges
        edge_index_full = decoded_graph['edge_index_full']
        assert isinstance(edge_index_full, np.ndarray)
        assert edge_index_full.shape == (25, 2)  # 4*4 = 16 edges for full connectivity

        # Check edge weights (should all be zero)
        edge_weight_full = decoded_graph['edge_weight_full']
        assert isinstance(edge_weight_full, np.ndarray)
        assert edge_weight_full.shape == (25, 1)
        assert np.all(edge_weight_full == 0.0)

        # Verify full connectivity structure
        expected_edges = set()
        for i in range(5):
            for j in range(5):
                expected_edges.add((i, j))

        actual_edges = set()
        for i, j in edge_index_full:
            actual_edges.add((i, j))

        assert expected_edges == actual_edges

    def test_decode_nodes_empty_graph(self):
        """
        Test that decode_nodes handles empty graphs properly.
        """
        # setting up model
        dim = 1000
        encoder = CategoricalIntegerEncoder(dim, 5)
        hyper_net = HyperNet(
            hidden_dim=dim,
            depth=3,
            node_encoder_map={
                'x': encoder,
            },
        )

        # Create a zero embedding (should decode to empty graph)
        empty_embedding = torch.zeros(dim)

        # decode nodes
        decoded_graph = hyper_net.decode_nodes(empty_embedding)

        assert isinstance(decoded_graph, dict)
        assert 'node_indices' in decoded_graph
        assert 'edge_index_full' in decoded_graph
        assert 'edge_weight_full' in decoded_graph

        # Should all be empty
        assert len(decoded_graph['node_indices']) == 0
        assert decoded_graph['edge_index_full'].shape == (0, 2)
        assert len(decoded_graph['edge_weight_full']) == 0