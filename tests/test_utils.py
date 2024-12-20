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
from .utils import ASSETS_PATH


def test_get_version():
    version = get_version()
    assert isinstance(version, str)
    assert version != ''


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