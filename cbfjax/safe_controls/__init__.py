"""
Safe controls module for CBF-JAX

Provides safe control implementations using Control Barrier Functions.
"""

from .closed_form_safe_control import (
    CFSafeControl,
    MinIntervCFSafeControl,
    InputConstCFSafeControl,
    MinIntervInputConstCFSafeControl,
    MinIntervInputConstCFSafeControlRaw
)
from .qp_safe_control import (
    QPSafeControl,
    MinIntervQPSafeControl,
    InputConstQPSafeControl,
    MinIntervInputConstQPSafeControl
)
from .backup_safe_control import (
    BackupSafeControl,
    MinIntervBackupSafeControl
)
from .parametric_flow_safe_control import ParametricFlowSafeControl

# NMPC and iLQR safe controls depend on optional packages (acados/casadi, trajax).
# They are lazily imported so ``import cbfjax`` succeeds without those deps.
_LAZY_IMPORTS = {
    "NMPCSafeControl": (".nmpc_safe_control", "nmpc"),
    "QuadraticNMPCSafeControl": (".nmpc_safe_control", "nmpc"),
    "iLQRSafeControl": (".ilqr_safe_control", "ilqr"),
    "QuadraticiLQRSafeControl": (".ilqr_safe_control", "ilqr"),
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
    # Closed-form controllers
    "CFSafeControl",
    "MinIntervCFSafeControl",
    "InputConstCFSafeControl",
    "MinIntervInputConstCFSafeControl",
    "MinIntervInputConstCFSafeControlRaw",
    # QP-based controllers
    "QPSafeControl",
    "MinIntervQPSafeControl",
    "InputConstQPSafeControl",
    "MinIntervInputConstQPSafeControl",
    # Backup controllers
    "BackupSafeControl",
    "MinIntervBackupSafeControl",
    # Parametric flow controller
    "ParametricFlowSafeControl",
    # NMPC controllers (optional - requires nmpc extra)
    "NMPCSafeControl",
    "QuadraticNMPCSafeControl",
    # iLQR controllers (optional - requires ilqr extra)
    "iLQRSafeControl",
    "QuadraticiLQRSafeControl",
]
