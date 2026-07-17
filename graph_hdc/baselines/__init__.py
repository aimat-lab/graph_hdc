"""
External baseline representations for molecular property prediction.

This subpackage collects *baseline* molecular representations that are used to
contextualize the hyperdimensional fingerprint (HDF) method against prior work.
Unlike the HDF encoders in :mod:`graph_hdc.models`, the representations here are
re-implementations of methods proposed elsewhere in the literature, kept in one
place so they can be shared across the various ``experiments`` scripts.

Currently available:

- :class:`graph_hdc.baselines.sherlock.SherlockFingerprint` -- the entropy-ranked,
  collision-free Morgan variant ("Sherlock Fingerprint") from Xu et al., *SPECTRE*
  (J. Chem. Inf. Model. 2026, 66, 2501-2512).
"""

from graph_hdc.baselines.sherlock import SherlockFingerprint

__all__ = ['SherlockFingerprint']
