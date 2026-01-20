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
from .nmpc_safe_control import (
    NMPCSafeControl,
    QuadraticNMPCSafeControl,
)

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
    # NMPC controllers
    "NMPCSafeControl",
    "QuadraticNMPCSafeControl",
]