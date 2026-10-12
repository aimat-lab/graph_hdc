"""
Diagnostics for the Fourier spectra of HDF hypervectors.

HDF binds by circular convolution, which multiplies Fourier spectra elementwise. If the random codebook vectors
have random Fourier magnitudes (the original encoder: Gaussian element vectors and complex Gaussian base vectors
of the fractional power encoders), repeated binding concentrates the embeddings on a few Fourier components
(about 10-100 effective components at any D for the message passing part). Since 2026-10-09 the encoders draw
unit-modulus codebooks by default (``unit_modulus`` of AtomEncoder and ContinuousEncoder; ablation ex_22).
"""
import torch


def effective_components(embeddings) -> float:
    """Median participation ratio of the power spectra of the given vectors (n, D): how many Fourier components
    effectively carry the energy (D for a flat spectrum, 1 if a single component dominates)."""
    vectors = torch.as_tensor(embeddings, dtype=torch.float64)
    power = torch.fft.fft(vectors, dim=-1).abs() ** 2
    ratio = power.sum(-1) ** 2 / (power ** 2).sum(-1)
    return float(ratio.median())
