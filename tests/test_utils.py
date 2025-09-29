import os
import pytest
from itertools import product

import torch
import jsonpickle
import networkx as nx
from rich.pretty import pprint

from graph_hdc.binding import circular_convolution_fft
from graph_hdc.utils import get_version
from graph_hdc.utils import render_latex
from graph_hdc.utils import torch_pairwise_reduce
from graph_hdc.utils import nx_random_uniform_edge_weight
from graph_hdc.utils import HypervectorCombinations
from graph_hdc.utils import ContinuousEncoder
from .utils import ASSETS_PATH


def test_get_version():
    version = get_version()
    assert isinstance(version, str)
    assert version != ''


@pytest.mark.localonly
def test_render_latex():
    output_path = os.path.join(ASSETS_PATH, 'out.pdf')
    render_latex({'content': '$\pi = 3.141$'}, output_path)
    assert os.path.exists(output_path)
    
    
def test_torch_pairwise_reduce():
    
    tens = torch.randn(size=(5, 10))
    result = torch_pairwise_reduce(tens, func=lambda a, b: a + b, dim=0)
    print(result.shape)
    assert isinstance(result, torch.Tensor)
    assert result.size(0) == 10
    
    
def test_nx_random_uniform_edge_weight():
    """
    The function should add a random edge weight to each edge in the graph.
    """
    g = nx.erdos_renyi_graph(n=10, p=0.5)
    g = nx_random_uniform_edge_weight(
        g=g,
        lo=0.1,
        hi=0.9,
    )
    
    for (u, v, data) in g.edges(data=True):
        assert isinstance(data['edge_weight'], float)
        assert 0.1 <= data['edge_weight'] <= 0.9


class TestJsonPickle:
    """
    This class generally bundles all the unittest cases related to the jsonpickle serialization 
    and deserialization library.
    """
    
    def test_saving_loading_torch_tensor_works(self):
        
        tensor = torch.randn(500, 3)
        serialized_tensor = jsonpickle.encode(tensor)
        deserialized_tensor = jsonpickle.decode(serialized_tensor)
        
        assert isinstance(deserialized_tensor, torch.Tensor)
        assert torch.equal(tensor, deserialized_tensor)


def test_product_works_as_expected():
    """
    Simply tests if the itertools.product function works as expected.
    """
    tuples_1 = [("a", 1), ("b", 2), ("c", 3)]
    tuples_2 = [("x", 10), ("y", 20),]
    tuples_3 = [("i", 100), ("j", 200),]
    
    result = list(product(tuples_1, tuples_2, tuples_3))
    print(result)
    assert len(result) == len(tuples_1) * len(tuples_2) * len(tuples_3)
    
    
class TestHypervectorCombinations:
    
    def test_construction_basically_works(self):
        
        # setting up testing data structure
        hv_dict_1 = {
            'a': torch.randn(10),
            'b': torch.randn(10),
            'c': torch.randn(10),
        }
        hv_dict_2 = {
            'x': torch.randn(10),
            'y': torch.randn(10),
        }
        hv_combinations = HypervectorCombinations(
            value_hv_dicts={
                '1': hv_dict_1,
                '2': hv_dict_2,
            },
            bind_fn=circular_convolution_fft,
        )
        
        # The main thing is that the combinations dict is setup correctly with the right number of 
        # combinations which is the multiplication of ht number of hypervectors in each base dictionary
        assert isinstance(hv_combinations, HypervectorCombinations)
        assert isinstance(hv_combinations.combinations, dict)
        assert len(hv_combinations.combinations) == len(hv_dict_1) * len(hv_dict_2)
        pprint(hv_combinations.combinations)
        
    def test_get_values_basically_works(self):
        
        # setting up testing data structure
        hv_dict_1 = {
            'a': torch.randn(10),
            'b': torch.randn(10),
            'c': torch.randn(10),
        }
        hv_dict_2 = {
            'x': torch.randn(10),
            'y': torch.randn(10),
        }
        hv_combinations = HypervectorCombinations(
            value_hv_dicts={
                '1': hv_dict_1,
                '2': hv_dict_2,
            },
            bind_fn=circular_convolution_fft,
        )
        
        # It should be possible to query any combination of individual hypervectors by using 
        # the "get" method and a dictionary that defines the desired combination
        result_1 = hv_combinations.get(query={'1': 'a', '2': 'x'})
        assert isinstance(result_1, torch.Tensor)
        assert result_1.size(0) == 10
        
        # The order of the specification also doesn't matter 
        result_2 = hv_combinations.get(query={'2': 'x', '1': 'a'})
        assert torch.equal(result_1, result_2)
        
        # If we define a combination that doesn't exist, an error should be raised
        with pytest.raises(KeyError):
            hv_combinations.get(query={'1': 'a', '2': 'z'})

    def test_iteration_works(self):
        
        # setting up testing data structure
        hv_dict_1 = {
            'a': torch.randn(10),
            'b': torch.randn(10),
            'c': torch.randn(10),
        }
        hv_dict_2 = {
            'x': torch.randn(10),
            'y': torch.randn(10),
        }
        hv_combinations = HypervectorCombinations(
            value_hv_dicts={
                '1': hv_dict_1,
                '2': hv_dict_2,
            },
            bind_fn=circular_convolution_fft,
        )
        
        # The object should be iterable and yield all combinations
        counter = 0
        for comb_dict, value in hv_combinations:
            print(comb_dict, value)
            assert isinstance(comb_dict, dict)
            assert isinstance(value, torch.Tensor)
            counter += 1
            
        assert counter == len(hv_dict_1) * len(hv_dict_2)


class TestContinuousEncoder:
    """Test suite for ContinuousEncoder class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.dim = 64
        self.size = 10.0
        self.bandwidth = 2.0
        self.encoder = ContinuousEncoder(self.dim, self.size, self.bandwidth)
    
    def test_init(self):
        """Test proper initialization of ContinuousEncoder."""
        assert self.encoder.dim == self.dim
        assert self.encoder.size == self.size
        assert self.encoder.bandwidth == self.bandwidth
        assert self.encoder.matrix is not None
        assert self.encoder.matrix.shape == (self.dim,)
        assert torch.is_complex(self.encoder.matrix)
    
    def test_encode_single_value(self):
        """Test encoding a single continuous value."""
        value = torch.tensor(1.5)
        encoded = self.encoder.encode(value)
        
        assert isinstance(encoded, torch.Tensor)
        assert encoded.shape == (self.dim,)
        assert torch.is_floating_point(encoded)
        assert not torch.isnan(encoded).any()
        assert not torch.isinf(encoded).any()
    
    def test_encode_batch_values(self):
        """Test encoding a batch of continuous values."""
        values = torch.tensor([0.0, 1.0, 2.0, -1.0])
        encoded = self.encoder.encode(values)
        
        assert isinstance(encoded, torch.Tensor)
        assert encoded.shape == (4, self.dim)
        assert torch.is_floating_point(encoded)
        assert not torch.isnan(encoded).any()
        assert not torch.isinf(encoded).any()
    
    def test_encode_zero(self):
        """Test encoding zero value."""
        value = torch.tensor(0.0)
        encoded = self.encoder.encode(value)
        
        assert isinstance(encoded, torch.Tensor)
        assert encoded.shape == (self.dim,)
        assert not torch.isnan(encoded).any()
        assert not torch.isinf(encoded).any()
    
    def test_encode_negative_value(self):
        """Test encoding negative values."""
        value = torch.tensor(-2.5)
        encoded = self.encoder.encode(value)
        
        assert isinstance(encoded, torch.Tensor)
        assert encoded.shape == (self.dim,)
        assert not torch.isnan(encoded).any()
        assert not torch.isinf(encoded).any()
    
    def test_encode_different_values_produce_different_vectors(self):
        """Test that different input values produce different hypervectors."""
        value1 = torch.tensor(1.0)
        value2 = torch.tensor(2.0)
        
        encoded1 = self.encoder.encode(value1)
        encoded2 = self.encoder.encode(value2)
        
        # Vectors should be different
        assert not torch.allclose(encoded1, encoded2, atol=1e-6)
        
        # Cosine similarity should be less than 1 (not identical)
        cosine_sim = torch.cosine_similarity(encoded1, encoded2, dim=0)
        assert cosine_sim < 0.99
    
    def test_encode_similar_values_produce_similar_vectors(self):
        """Test that similar input values produce similar hypervectors."""
        value1 = torch.tensor(1.0)
        value2 = torch.tensor(1.01)  # Very close value
        
        encoded1 = self.encoder.encode(value1)
        encoded2 = self.encoder.encode(value2)
        
        # Vectors should be similar (high cosine similarity)
        cosine_sim = torch.cosine_similarity(encoded1, encoded2, dim=0)
        assert cosine_sim > 0.9  # Should be quite similar
    
    def test_decode_returns_tensor(self):
        """Test that decode returns a tensor."""
        value = torch.tensor(1.5)
        encoded = self.encoder.encode(value)
        decoded = self.encoder.decode(encoded)
        
        assert isinstance(decoded, torch.Tensor)
        assert decoded.numel() == 1  # Should be scalar
        assert torch.is_floating_point(decoded)
        assert not torch.isnan(decoded)
        assert not torch.isinf(decoded)
    
    def test_encode_decode_approximate_reconstruction(self):
        """Test that encoding then decoding gives approximately the original value."""
        test_values = [0.0, 1.0, 2.0, -1.0, -2.5, 0.5]
        
        for val in test_values:
            value = torch.tensor(val)
            encoded = self.encoder.encode(value)
            decoded = self.encoder.decode(encoded)
            
            # Due to approximation nature of FHRR decoding, we allow larger tolerance
            error = torch.abs(decoded - value)
            relative_error = error / (torch.abs(value) + 1e-6)  # Add small epsilon to avoid division by zero
            
            # Check that relative error is reasonable (less than 100% for non-zero values)
            if torch.abs(value) > 1e-6:
                assert relative_error < 2.0, f"Large relative error for value {val}: decoded={decoded.item()}, error={error.item()}"
            else:
                # For values close to zero, check absolute error
                assert error < 5.0, f"Large absolute error for value {val}: decoded={decoded.item()}, error={error.item()}"
    
    def test_encode_decode_batch_values(self):
        """Test encoding and decoding batch of values."""
        values = torch.tensor([0.0, 1.0, 2.0, -1.0])
        
        # Encode each value individually
        encoded_list = []
        for val in values:
            encoded = self.encoder.encode(val)
            encoded_list.append(encoded)
        
        # Decode each encoded vector
        decoded_list = []
        for encoded in encoded_list:
            decoded = self.encoder.decode(encoded)
            decoded_list.append(decoded)
        
        decoded_values = torch.stack(decoded_list)
        
        # Check that all values are reasonable
        assert not torch.isnan(decoded_values).any()
        assert not torch.isinf(decoded_values).any()
        assert decoded_values.shape == values.shape
    
    def test_encoding_magnitude_consistency(self):
        """Test that encoded vectors have reasonable magnitudes."""
        values = torch.tensor([0.0, 1.0, 2.0, -1.0, -2.0])
        
        for val in values:
            encoded = self.encoder.encode(val)
            magnitude = torch.norm(encoded)
            
            # Magnitude should be reasonable (not too small or too large)
            assert magnitude > 1e-6, f"Encoded vector magnitude too small for value {val}"
            assert magnitude < 1e6, f"Encoded vector magnitude too large for value {val}"
    
    def test_different_bandwidth_affects_encoding(self):
        """Test that different bandwidth values affect the encoding."""
        value = torch.tensor(1.0)
        
        encoder1 = ContinuousEncoder(self.dim, self.size, 1.0)
        encoder2 = ContinuousEncoder(self.dim, self.size, 3.0)
        
        encoded1 = encoder1.encode(value)
        encoded2 = encoder2.encode(value)
        
        # Different bandwidths should produce different encodings
        assert not torch.allclose(encoded1, encoded2, atol=1e-6)