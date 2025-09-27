"""
Barriers module for CBF-JAX

Provides control barrier function implementations using JAX and Equinox.
"""

from .barrier import Barrier
from .multi_barrier import MultiBarriers
from .composite_barrier import SoftCompositionBarrier, NonSmoothCompositionBarrier

__all__ = [
    "Barrier",
    "MultiBarriers",
    "SoftCompositionBarrier",
    "NonSmoothCompositionBarrier",
]