"""
Unit-modulus spectra for the random codebook vectors of HDF (ablation ex_22 in experiments/fingerprints).

HDF binds by circular convolution, which multiplies Fourier spectra elementwise. The codebook vectors of graph_hdc
have random Fourier magnitudes: the element vectors (``AtomEncoder``) have i.i.d. Gaussian entries, and the base
vectors of the fractional power encoders (``ContinuousEncoder``) have a complex Gaussian spectrum that is raised to
the power value / bandwidth. Repeated binding multiplies these magnitudes, so a few Fourier components dominate
the embeddings (about 10-100 effective components at any D for the structure part of HDF).

``unit_spectrum`` sets every Fourier magnitude of these vectors to one and keeps the phases (unitary HRR vectors,
i.e. FHRR seen through the FFT). The element vectors stay independent random vectors, one per element; only the
way they are drawn changes. The fractional power encoders still encode a value by rotating the phases, but every
value now has the same norm and the similarity of two values depends only on their difference.
"""
import torch

from graph_hdc.special.molecules import AtomEncoder
from graph_hdc.utils import ContinuousEncoder


def unit_spectrum(encoder_map: dict) -> dict:
    """Give every codebook vector of the encoders in ``encoder_map`` unit Fourier magnitudes (in place)."""
    for name, encoder in encoder_map.items():
        if isinstance(encoder, ContinuousEncoder):
            # the spectrum is already scaled so that the vector of value 1 has norm 1; with unit magnitudes
            # every encoded value has norm 1
            encoder.matrix = encoder.matrix / encoder.matrix.abs()
        elif isinstance(encoder, AtomEncoder):
            # real vectors, so the spectrum is Hermitian and stays so; ifft of a unit-modulus spectrum has norm 1
            spectrum = torch.fft.fft(encoder.embeddings, dim=-1)
            encoder.embeddings = torch.fft.ifft(spectrum / spectrum.abs(), dim=-1).real.to(encoder.embeddings.dtype)
        else:
            raise TypeError(f'unit_spectrum: no unit-modulus version for encoder {name!r} ({type(encoder).__name__})')
    return encoder_map


def effective_components(embeddings) -> float:
    """Median participation ratio of the power spectra of the given vectors (n, D): how many Fourier components
    effectively carry the energy (D for a flat spectrum, 1 if a single component dominates)."""
    vectors = torch.as_tensor(embeddings, dtype=torch.float64)
    power = torch.fft.fft(vectors, dim=-1).abs() ** 2
    ratio = power.sum(-1) ** 2 / (power ** 2).sum(-1)
    return float(ratio.median())
