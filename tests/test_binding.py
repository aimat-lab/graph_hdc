import torch
import numpy as np
from torch.nn.functional import normalize

from graph_hdc.binding import circular_convolution
from graph_hdc.binding import circular_convolution_fft
from graph_hdc.binding import circular_correlation
from graph_hdc.binding import circular_correlation_fft


class TestCircular:
    
    def test_circular_convolution_basically_works(self):
        
        a = torch.randn(10)
        b = torch.randn(10)
        result = circular_convolution(a, b)
        assert result is not None
        assert isinstance(result, torch.Tensor)
        assert result.shape == a.shape
        

    def test_circular_correlation_basically_works(self):
            
        a = torch.randn(10)
        b = torch.randn(10)
        result = circular_correlation(a, b)
        assert result is not None
        assert isinstance(result, torch.Tensor)
        assert result.shape == a.shape
        

    def test_circular_convolution_circular_correlation_inverting(self):
        """
        The circular correlation function should be a valid unbinding operation to the circular_convolution 
        function in the sense that it should create a vector which is very highly correlated with the original
        vector but not exactly the same.
        """
        dim = 10_000
        a = torch.tensor(np.random.normal(scale=1/np.sqrt(dim), size=(dim, )))
        b = torch.tensor(np.random.normal(scale=1/np.sqrt(dim), size=(dim, )))
        ab_dot = torch.dot(a, b).item()
        print(f"Dot product between a and b: {ab_dot}")
        assert np.isclose(ab_dot, 0.0, atol=0.05)
        
        # when binding and unbinding we should get a vector which is very highly correlated with the original vector
        # but not exactly the same.
        ab = circular_convolution(a, b)
        b_ = circular_correlation(a, ab)
        
        bb_dot = torch.dot(b, b_).item()
        assert np.isclose(bb_dot, 1.0, atol=0.05)
        
    def test_circular_convolution_fft_basically_works(self):
        """
        The fft version of the circular_convolution version should yield basically the same result 
        as the non-fft version.
        """
        a = torch.randn(10)
        b = torch.randn(10)
        result = circular_convolution(a, b)
        result_fft = circular_convolution_fft(a, b)
        assert torch.allclose(result, result_fft)
        
    def test_circular_correlation_fft_basically_works(self):
        """
        The fft version of the circular_correlation version should yield basically the same result
        as the non-fft version.
        """
        a = torch.randn(10)
        b = torch.randn(10)
        result = circular_correlation(a, b)
        result_fft = circular_correlation_fft(a, b)
        assert torch.allclose(result, result_fft)
        
    def test_circular_convolution_fft_differentiable(self):
        """
        The circular_convolution_fft function should be differentiable in the sense that it should be 
        possible to propagate a torch.grad through the function.
        """
        a = torch.randn(10, requires_grad=True)
        b = torch.randn(10, requires_grad=True)
        result = circular_convolution_fft(a, b)
        result.sum().backward()
        assert a.grad is not None
        assert b.grad is not None
        assert not torch.allclose(a.grad, torch.zeros_like(a))
        assert not torch.allclose(b.grad, torch.zeros_like(b))
        
    def test_circular_correlation_fft_differentiable(self):
        """
        The circular_correlation_fft function should be differentiable in the sense that it should be 
        possible to propagate a torch.grad through the function.
        """
        a = torch.randn(10, requires_grad=True)
        b = torch.randn(10, requires_grad=True)
        result = circular_correlation_fft(a, b)
        result.sum().backward()
        assert a.grad is not None
        assert b.grad is not None
        assert not torch.allclose(a.grad, torch.zeros_like(a))
        assert not torch.allclose(b.grad, torch.zeros_like(b))