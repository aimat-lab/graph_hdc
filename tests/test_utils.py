import os

import torch
import jsonpickle

from graph_hdc.utils import get_version
from graph_hdc.utils import render_latex
from .utils import ASSETS_PATH


def test_get_version():
    version = get_version()
    assert isinstance(version, str)
    assert version != ''


def test_render_latex():
    output_path = os.path.join(ASSETS_PATH, 'out.pdf')
    render_latex({'content': '$\pi = 3.141$'}, output_path)
    assert os.path.exists(output_path)


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
    