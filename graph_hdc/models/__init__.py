"""
Hyperdimensional computing models for graph structures.

This package provides various HyperNet implementations for encoding
and decoding graphs using high-dimensional vectors.

The module is organized into:
- base.py: Abstract base class (AbstractHyperNet)
- main.py: Main implementation (HyperNet)
- composite.py: Composite variants (CompositeHyperNet)
- ensemble.py: Ensemble implementation (HyperNetEnsemble)
"""

# Import main classes from submodules
from .base import AbstractHyperNet
from .main import HyperNet
from .composite import CompositeHyperNet
from .ensemble import HyperNetEnsemble

# Re-export encoder classes from utils for backwards compatibility
# These were previously imported in the old models.py file and some
# external code imports them from graph_hdc.models
from graph_hdc.utils import (
    AbstractEncoder,
    CategoricalOneHotEncoder,
    CategoricalIntegerEncoder,
    ContinuousEncoder,
)

__all__ = [
    # Core model classes
    'AbstractHyperNet',
    'HyperNet',
    'CompositeHyperNet',
    'HyperNetEnsemble',
    # Re-exported from utils for backwards compatibility
    'AbstractEncoder',
    'CategoricalOneHotEncoder',
    'CategoricalIntegerEncoder',
    'ContinuousEncoder',
]
