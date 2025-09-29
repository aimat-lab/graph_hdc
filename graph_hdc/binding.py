from typing import Dict, Union, Callable, Literal

import torch
import numpy as np

from graph_hdc.functions import register_function


# == BINDING FUNCTIONS ==
# This section contains all the "binding" functions which are used to bind two tensors together.

def circulant_matrix(x: torch.Tensor) -> torch.Tensor:
    """
    Given a the tensor ``x``, this function will return the circulant matrix of ``x``.
    If the input tensor is of shape (n,) the output matrix will be of shape (n, n).
    
    :param x: The input tensor
    
    :return: The circulant matrix of the input tensor
    """
    x_flipped = torch.flip(x, dims=[0])
    return torch.stack([torch.roll(x_flipped, shifts=j+1) for j in range(len(x))])


@register_function('circular_convolution', 'bind')
def circular_convolution(tens1: torch.Tensor,
                         tens2: torch.Tensor
                         ) -> torch.Tensor:
    """
    Performs a circular convolution between the given tensors ``tens1`` and ``tens2``.
    
    :param tens1: The first tensor to be convolved
    :param tens2: The second tensor to be convolved
    
    :return: The circular convolution of the two tensors
    """
    return circulant_matrix(tens1) @ tens2


def _circular_convolution_fft(tens: torch.Tensor) -> torch.Tensor:
    # Original implementation for 1D tensors
    ffts = torch.vmap(torch.fft.fft)(tens)
    ffts = ffts.prod(0)
    return torch.fft.ifft(ffts).real


@register_function('circular_convolution_fft', 'bind')
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
    # Original implementation for 1D tensors
    ffts = torch.vmap(torch.fft.fft)(torch.stack([tens1, tens2]))
    ffts = ffts.prod(0)
    return torch.fft.ifft(ffts).real


# == UNBIND FUNCTIONS ==
# This section contains all the "unbind" function which are used to unbind two tensors aka to 
# invert the "bind" operation.

@register_function('circular_correlation', 'unbind')
def circular_correlation(tens1: torch.Tensor,
                         tens2: torch.Tensor
                         ) -> torch.Tensor:
    """
    Performs the circular correlation between the given tensors ``tens1`` and ``tens2``.
    
    :param tens1: The first tensor to be convolved
    :param tens2: The second tensor to be convolved
    
    :return: The circular correlation of the two tensors
    """
    return circulant_matrix(tens1).T @ tens2


@register_function('circular_correlation_fft', 'unbind')
def circular_correlation_fft(tens1: torch.Tensor,
                             tens2: torch.Tensor
                             ) -> torch.Tensor:
    """
    Performs the circular correlation between the given tensors ``tens1`` and ``tens2``. 
    
    Note that this function will perform the correlation in the frequency domain using the fast
    fourier transform (FFT) algorithm for computational efficiency.
    
    :param tens1: The first tensor to be convolved
    :param tens2: The second tensor to be convolved
    
    :return: The circular correlation of the two tensors
    """
    ffts = torch.vmap(torch.fft.fft)(torch.stack([tens1, tens2]))
    fft_a_conj = ffts[0].conj() 
    fft_product = fft_a_conj * ffts[1]
    return torch.fft.ifft(fft_product).real

    