"""
Controls module for CBFJAX.

Provides base control classes that can be extended with safety constraints.
"""

from .base_control import BaseControl, DummyDynamics

__all__ = [
    "BaseControl",
    "DummyDynamics",
]