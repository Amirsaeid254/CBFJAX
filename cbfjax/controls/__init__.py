"""
Controls module for CBFJAX.

Provides base control classes that can be extended with safety constraints.

All controllers follow the stateful interface:
- optimal_control(x, state) -> (u, new_state)
- get_init_state() -> initial controller state
"""

from .base_control import BaseControl
from .goal_control import GoalControl
from .mppi_control import MPPIControl
from .control_types import (
    ILQRState,
    ConstrainedILQRState,
    ILQRInfo,
    ConstrainedILQRInfo,
    CFInfo,
    QPInfo,
    BackupInfo,
    NMPCInfo,
    MPPIState,
    MPPIInfo,
)
from ..dynamics.base_dynamic import DummyDynamics

# Names that require optional dependencies (acados/casadi for NMPC, trajax for iLQR).
# These are loaded lazily so that ``import cbfjax`` works even when the optional
# deps are not installed. Accessing the attribute triggers the heavy import and
# raises a clear ImportError with installation instructions.
_LAZY_IMPORTS = {
    "NMPCControl":    (".nmpc_control",  "nmpc"),
    "QuadraticNMPCControl": (".nmpc_control", "nmpc"),
    "iLQRControl": (".ilqr_control", "ilqr"),
    "QuadraticiLQRControl": (".ilqr_control", "ilqr"),
    "ConstrainediLQRControl": (".ilqr_control", "ilqr"),
    "QuadraticConstrainediLQRControl": (".ilqr_control", "ilqr"),
}


def __getattr__(name):
    if name in _LAZY_IMPORTS:
        module_name, extra = _LAZY_IMPORTS[name]
        try:
            from importlib import import_module
            module = import_module(module_name, package=__name__)
        except ImportError as e:
            raise ImportError(
                f"{name} requires the optional '{extra}' dependencies. "
                f"Install with: pip install cbfjax[{extra}]\n"
                f"Original error: {e}"
            ) from e
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BaseControl",
    "GoalControl",
    "MPPIControl",
    "NMPCControl",
    "QuadraticNMPCControl",
    "iLQRControl",
    "QuadraticiLQRControl",
    "ConstrainediLQRControl",
    "QuadraticConstrainediLQRControl",
    "DummyDynamics",
    # State and info types
    "ILQRState",
    "ConstrainedILQRState",
    "ILQRInfo",
    "ConstrainedILQRInfo",
    "CFInfo",
    "QPInfo",
    "BackupInfo",
    "NMPCInfo",
    "MPPIState",
    "MPPIInfo",
]
