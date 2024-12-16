from typing import Union, Callable

import torch
import numpy as np


# == BINDING FUNCTIONS ==
# This section contains all the "binding" functions which are used to bind two tensors together.

def circulant_matrix(x: torch.Tensor) -> torch.Tensor:
    x_flipped = torch.flip(x, dims=[0])
    return torch.stack([torch.roll(x_flipped, shifts=j+1) for j in range(len(x))])

def circular_convolution(tens1: torch.Tensor,
                         tens2: torch.Tensor
                         ) -> torch.Tensor:
    return circulant_matrix(tens1) @ tens2


def circular_convolution_fft(tens1: torch.Tensor, 
                             tens2: torch.Tensor
                             ) -> torch.Tensor:
    """
    Performs a circular convolution between the given tensors ``tens1`` and ``tens2``. 
    
    Note that this function will peform the convolution in the frequency domain using the fast 
    fourier transform (FFT) algorithm for to computational efficiency.
    
    :param tens1: The first tensor to be convolved
    :param tens2: The second tensor to be convolved
    
    :return: The circular convolution of the two tensors
    """
    ffts = torch.vmap(torch.fft.fft)(torch.stack([tens1, tens2]))
    ffts = ffts.prod(0)
    return torch.fft.ifft(ffts).real


# == UNBIND FUNCTIONS ==
# This section contains all the "unbind" function which are used to unbind two tensors aka to 
# invert the "bind" operation.

def circular_correlation(tens1: torch.Tensor,
                         tens2: torch.Tensor
                         ) -> torch.Tensor:
    return circulant_matrix(tens1).T @ tens2

def circular_correlation_fft(tens1: torch.Tensor,
                             tens2: torch.Tensor
                             ) -> torch.Tensor:
    """
    Performs the circular convolution between the given tensors ``tens1`` and ``tens2``. 
    
    Note that this function will perform the convolution in the frequency domain using the fast
    fourier transform (FFT) algorithm for computational efficiency.
    
    :param tens1: The first tensor to be convolved
    :param tens2: The second tensor to be convolved
    
    :return: The circular correlation of the two tensors
    """
    ffts = torch.vmap(torch.fft.fft)(torch.stack([tens1, tens2]))
    fft_a_conj = ffts[0].conj() 
    fft_product = fft_a_conj * ffts[1]
    return torch.fft.ifft(fft_product).real



# == REGISTRY ==

FUNCTION_REGISTRY = {
    # bind functions
    'circular_convolution': circular_convolution,
    'circular_convolution_fft': circular_convolution_fft,
    # unbind functions
    'circular_correlation': circular_correlation,
    'circular_correlation_fft': circular_correlation_fft
}

def resolve_function(value: Union[str, Callable]
                     ) -> Callable:
    if isinstance(value, str):
        return FUNCTION_REGISTRY[value]
    else:
        return value