"""
CBF-JAX: Control Barrier Functions in JAX

A JAX-based implementation of Control Barrier Functions for safe control,
migrated from CBFTorch with improved performance through JIT compilation
and functional programming paradigms.
"""

__version__ = "0.1.0"

# Core modules
from . import dynamics
from . import barriers
from . import safe_controls
from . import utils

__all__ = [
    "dynamics",
    "barriers",
    "safe_controls",
    "utils"
]