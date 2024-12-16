import os
import tempfile

import torch
import numpy as np
import jsonpickle

import graph_hdc.utils
from graph_hdc.models import AbstractEncoder
from graph_hdc.models import CategoricalOneHotEncoder


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
        value1 = [1, 0, 0]
        value2 = [0, 0, 1]
        
        encoder = CategoricalOneHotEncoder(
            dim=2000,
            num_categories=3,
        )
        encoded1 = encoder.encode(value1)
        assert isinstance(encoded1, torch.Tensor)
        assert torch.allclose(encoded1, encoder.embeddings[0])
        decoded1 = encoder.decode(encoded1)
        assert decoded1 == 0
        
        encoded2 = encoder.encode(value2)
        assert isinstance(encoded2, torch.Tensor)
        assert torch.allclose(encoded2, encoder.embeddings[2])
        decoded2 = encoder.decode(encoded2)
        assert decoded2 == 2
        
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