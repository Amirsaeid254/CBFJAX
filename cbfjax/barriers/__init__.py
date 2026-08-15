"""
Barriers module for CBF-JAX

Provides control barrier function implementations using JAX and Equinox.
"""

from .barrier import Barrier
from .multi_barrier import MultiBarriers
from .composite_barrier import SoftCompositionBarrier, HardCompositionBarrier
from .backup_barrier import BackupBarrier
from .parametric_flow_barrier import FlowBarrier
from .parametric_flow_barrier2 import FlowBarrier2

__all__ = [
    "Barrier",
    "MultiBarriers",
    "SoftCompositionBarrier",
    "HardCompositionBarrier",
    "BackupBarrier",
    "FlowBarrier",
    "FlowBarrier2",
]