"""
CBFJAX: Control Barrier Functions in JAX.

A JAX-based implementation of Control Barrier Functions for safe control,
featuring JIT-compiled barrier functions, multiple safe-control backends
(closed-form, QP, backup-CBF, NMPC, iLQR), and a rich set of system dynamics.
"""

__version__ = "0.1.0"
__author__ = "Amirsaeid Safari"
__email__ = "safari.amirsaeid@gmail.com"
__license__ = "MIT"

# Configure JAX before anything else imports from it.
from . import config

# Core public submodules
from . import dynamics
from . import barriers
from . import controls
from . import safe_controls
from . import utils

# Commonly used classes promoted to the top-level namespace
from .dynamics import (
    AffineInControlDynamics,
    UnicycleDynamics,
    DoubleIntegratorDynamics,
    SingleIntegratorDynamics,
    BicycleDynamics,
    InvertedPendulumDynamics,
    UnicycleReducedOrderDynamics,
)
from .barriers import (
    Barrier,
    MultiBarriers,
    SoftCompositionBarrier,
    NonSmoothCompositionBarrier,
    BackupBarrier,
    StackedBarrier,
)
from .controls import BaseControl, GoalControl
from .safe_controls import (
    CFSafeControl,
    MinIntervCFSafeControl,
    InputConstCFSafeControl,
    MinIntervInputConstCFSafeControl,
    QPSafeControl,
    MinIntervQPSafeControl,
    InputConstQPSafeControl,
    MinIntervInputConstQPSafeControl,
    BackupSafeControl,
    MinIntervBackupSafeControl,
)

# Config-driven construction
from .factory import from_config, build_barrier

# Ensemble utilities
from .utils.utils import stack_ensemble, unstack_ensemble

# Configuration helpers
from .config import (
    configure_jax,
    get_jax_config,
    set_default_dtype,
    get_default_dtype,
)

__all__ = [
    # Metadata
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    # Submodules
    "config",
    "dynamics",
    "barriers",
    "controls",
    "safe_controls",
    "utils",
    # Dynamics
    "AffineInControlDynamics",
    "UnicycleDynamics",
    "DoubleIntegratorDynamics",
    "SingleIntegratorDynamics",
    "BicycleDynamics",
    "InvertedPendulumDynamics",
    "UnicycleReducedOrderDynamics",
    # Barriers
    "Barrier",
    "MultiBarriers",
    "SoftCompositionBarrier",
    "NonSmoothCompositionBarrier",
    "BackupBarrier",
    "StackedBarrier",
    # Controls
    "BaseControl",
    "GoalControl",
    # Safe controls (always available)
    "CFSafeControl",
    "MinIntervCFSafeControl",
    "InputConstCFSafeControl",
    "MinIntervInputConstCFSafeControl",
    "QPSafeControl",
    "MinIntervQPSafeControl",
    "InputConstQPSafeControl",
    "MinIntervInputConstQPSafeControl",
    "BackupSafeControl",
    "MinIntervBackupSafeControl",
    # Config-driven construction
    "from_config",
    "build_barrier",
    # Ensemble utilities
    "stack_ensemble",
    "unstack_ensemble",
    # Configuration
    "configure_jax",
    "get_jax_config",
    "set_default_dtype",
    "get_default_dtype",
]
