"""
Safe controls module for CBF-JAX

Provides safe control implementations using Control Barrier Functions.
"""

from .closed_form_safe_control import MinIntervCFSafeControl

__all__ = [
    "MinIntervCFSafeControl"
]