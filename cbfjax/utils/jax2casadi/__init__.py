"""
JAX2CasADi - Convert JAX functions to CasADi symbolic expressions

This library enables seamless integration of JAX codebases with CasADi-based
tools like acados for optimal control and numerical optimization.

Quick Start:
    >>> import jax.numpy as jnp
    >>> from jax2casadi import convert
    >>> 
    >>> def dynamics(x, u):
    ...     return jnp.array([x[1], -jnp.sin(x[0]) + u[0]])
    >>> 
    >>> casadi_fn = convert(dynamics, [('x', (2,)), ('u', (1,))])
    >>> # Now use casadi_fn with acados, CasADi optimization, etc.

Features:
    - Automatic conversion of JAX functions to CasADi
    - Support for 50+ mathematical operations
    - Numerical validation
    - Clean error messages

Author: JAX2CasADi Contributors
License: MIT
"""

__version__ = "0.1.0"
__author__ = "JAX2CasADi Contributors"
__license__ = "MIT"

from ._converter import convert
from ._validator import validate

try:
    from ._ops import (
        list_supported_operations,
        print_supported_operations,
        is_supported,
        get_operation_category
    )
except ImportError:
    # Fallback if these don't exist in _ops
    from ._ops import is_supported, get_operation_category
    list_supported_operations = None
    print_supported_operations = None

__all__ = [
    'convert',
    'validate',
    'is_supported',
    'get_operation_category',
]

# Add optional exports if they exist
if list_supported_operations is not None:
    __all__.extend(['list_supported_operations', 'print_supported_operations'])
