"""
Controls module for CBFJAX.

Provides base control classes that can be extended with safety constraints.

All controllers follow the stateful interface:
- _optimal_control_single(x, state) -> (u, new_state)
- get_init_state() -> initial controller state
"""

from .base_control import BaseControl
from .nmpc_control import NMPCControl, QuadraticNMPCControl
from .ilqr_control import (
    iLQRControl,
    QuadraticiLQRControl,
    ConstrainediLQRControl,
    QuadraticConstrainediLQRControl,
)
from .control_types import (
    ILQRState,
    ConstrainedILQRState,
    ILQRInfo,
    ConstrainedILQRInfo,
    CFInfo,
    QPInfo,
    BackupInfo,
    NMPCInfo,
)
from ..dynamics.base_dynamic import DummyDynamics

__all__ = [
    "BaseControl",
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
]