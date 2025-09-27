"""
CBF-JAX: Control Barrier Functions in JAX

A JAX-based implementation of Control Barrier Functions for safe control
with high-performance JIT compilation and functional programming paradigms.
"""

__version__ = "0.1.0"
__author__ = "Amirsaeid Safari"
__email__ = "safari.amirsaeid@gmail.com"
__license__ = "MIT"

# Import configuration first to set up JAX
from . import config

# Core modules
from . import dynamics
from . import barriers
from . import safe_controls
from . import utils

# Main classes for convenience
from .dynamics import UnicycleDynamics, AffineInControlDynamics
from .barriers import Barrier, MultiBarriers, SoftCompositionBarrier, NonSmoothCompositionBarrier
from .safe_controls import (
    MinIntervCFSafeControl,
    MinIntervQPSafeControl,
    InputConstQPSafeControl,
    MinIntervInputConstQPSafeControl
)

# Configuration functions
from .config import configure_jax, get_jax_config, set_default_dtype, get_default_dtype

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__email__",
    "__license__",

    # Modules
    "config",
    "dynamics",
    "barriers",
    "safe_controls",
    "utils",

    # Main classes
    "UnicycleDynamics",
    "AffineInControlDynamics",
    "Barrier",
    "MultiBarriers",
    "SoftCompositionBarrier",
    "NonSmoothCompositionBarrier",
    "MinIntervCFSafeControl",
    "MinIntervQPSafeControl",
    "InputConstQPSafeControl",
    "MinIntervInputConstQPSafeControl",

    # Configuration
    "configure_jax",
    "get_jax_config",
    "set_default_dtype",
    "get_default_dtype",
]