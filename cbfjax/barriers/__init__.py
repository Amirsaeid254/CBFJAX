"""
Barriers module for CBF-JAX

Provides control barrier function implementations using JAX and Equinox.
"""

from .barrier import Barrier
from .composite_barrier import SoftCompositionBarrier, NonSmoothCompositionBarrier

__all__ = [
    "Barrier",
    "SoftCompositionBarrier",
    "NonSmoothCompositionBarrier",
]