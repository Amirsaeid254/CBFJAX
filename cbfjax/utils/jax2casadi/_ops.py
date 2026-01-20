"""
Operation Mapping: JAX Primitives → CasADi Operations

This module contains the COMPLETE REVERSE of JAXADi's OP_JAX_VALUE_DICT.
Each JAX primitive operation is mapped to its CasADi equivalent.

Source: JAXADi _ops.py analyzed on 2026-01-16
Total Operations Mapped: 40+ core operations from JAXADi

Reverse Engineering from JAXADi's _ops.py:
- JAXADi: CasADi OP_* → JAX jnp.* 
- JAX2CasADi: JAX primitive → CasADi ca.* (THIS FILE)
"""

import casadi as ca
from typing import Callable, Dict, Union

CasADiOp = Union[Callable, type(ca.sin)]

# ============================================================================
# SMART BROADCASTING HELPERS
# ============================================================================

def _smart_broadcast_op(x, y, op):
    """
    Smart broadcasting for binary operations
    
    Handles JAX vs CasADi shape mismatches:
    - JAX: 1D arrays are (n,), 2D row vectors are (1, n)
    - CasADi: vectors are always (n, 1) column vectors
    
    This causes issues like: (1, 2) - (2, 1) → dimension mismatch
    We fix by detecting compatible shapes and transposing if needed.
    """
    # Get shapes
    x_shape = x.shape if hasattr(x, 'shape') else ()
    y_shape = y.shape if hasattr(y, 'shape') else ()
    
    # If either is scalar, just do the operation
    if len(x_shape) == 0 or len(y_shape) == 0:
        return op(x, y)
    
    # If shapes are already compatible, just do the operation
    if x_shape == y_shape:
        return op(x, y)
    
    # Check for (1, n) vs (n, 1) mismatch - common JAX/CasADi issue
    if (len(x_shape) == 2 and len(y_shape) == 2 and
        x_shape[0] == 1 and y_shape[1] == 1 and
        x_shape[1] == y_shape[0]):
        # x is (1, n) and y is (n, 1) - transpose y to (1, n)
        y_transposed = ca.transpose(y)
        return op(x, y_transposed)
    
    # Check reverse: (n, 1) vs (1, n)
    if (len(x_shape) == 2 and len(y_shape) == 2 and
        x_shape[1] == 1 and y_shape[0] == 1 and
        x_shape[0] == y_shape[1]):
        # x is (n, 1) and y is (1, n) - transpose x to (1, n)
        x_transposed = ca.transpose(x)
        return op(x_transposed, y)
    
    # Check for (n,) vs (1, n) - CasADi might represent (n,) as (n, 1)
    if (len(x_shape) == 2 and len(y_shape) == 2 and
        ((x_shape[0] == 1 and y_shape[1] == 1 and x_shape[1] == y_shape[0]) or
         (x_shape[1] == 1 and y_shape[0] == 1 and x_shape[0] == y_shape[1]))):
        # Try transposing the column vector
        if x_shape[1] == 1 and y_shape[0] == 1:
            x_transposed = ca.transpose(x)
            return op(x_transposed, y)
        elif x_shape[0] == 1 and y_shape[1] == 1:
            y_transposed = ca.transpose(y)
            return op(x, y_transposed)
    
    # Otherwise, let CasADi handle it (might error or broadcast)
    return op(x, y)

def _smart_add(x, y):
    """Smart addition with broadcasting"""
    return _smart_broadcast_op(x, y, lambda a, b: a + b)

def _smart_sub(x, y):
    """Smart subtraction with broadcasting"""
    return _smart_broadcast_op(x, y, lambda a, b: a - b)

def _smart_mul(x, y):
    """Smart multiplication with broadcasting"""
    return _smart_broadcast_op(x, y, lambda a, b: a * b)

def _smart_div(x, y):
    """Smart division with broadcasting"""
    return _smart_broadcast_op(x, y, lambda a, b: a / b)

# ============================================================================
# CORE OPERATION MAPPING - REVERSE OF JAXADi
# ============================================================================

JAX_PRIMITIVE_TO_CASADI: Dict[str, CasADiOp] = {
    # ========== Arithmetic Operations ==========
    # JAXADi: OP_ADD -> "work[{0}] + work[{1}]" -> jnp.add
    # JAX2CasADi: 'add' -> ca.add (x + y)
    # Using smart broadcasting to handle JAX (1,n) vs CasADi (n,1) mismatches
    'add': _smart_add,
    'sub': _smart_sub,
    'mul': _smart_mul,
    'div': _smart_div,
    'neg': lambda x: -x,
    
    # Power operations
    # JAX uses integer_pow for x^n where n is integer constant
    'integer_pow': lambda x, *, y: ca.power(x, y),  
    'pow': ca.power,
    
    # Special arithmetic
    'sqrt': ca.sqrt,
    'sq': lambda x: x * x,  # JAXADi: OP_SQ -> "work[{0}] * work[{0}]"
    'inv': lambda x: 1.0 / x,  # JAXADi: OP_INV -> "1.0 / work[{0}]"
    
    # ========== Trigonometric Operations ==========
    # Note: JAX uses 'asin', 'acos', 'atan' (not arcsin, arccos, arctan in primitives)
    'sin': ca.sin,
    'cos': ca.cos,
    'tan': ca.tan,
    'asin': ca.asin,  # JAX primitive name (JAXADi maps from OP_ASIN)
    'acos': ca.acos,
    'atan': ca.atan,
    'atan2': ca.atan2,
    
    # ========== Hyperbolic Operations ==========
    'sinh': ca.sinh,
    'cosh': ca.cosh,
    'tanh': ca.tanh,
    'asinh': ca.asinh,  # JAX primitive name
    'acosh': ca.acosh,
    'atanh': ca.atanh,
    
    # ========== Exponential & Logarithmic ==========
    'exp': ca.exp,
    'log': ca.log,
    
    # ========== Comparison Operations ==========
    # JAXADi: OP_LT -> "work[{0}] < work[{1}]"
    'lt': lambda x, y: x < y,
    'le': lambda x, y: x <= y,
    'eq': lambda x, y: x == y,
    'ne': lambda x, y: x != y,
    'gt': lambda x, y: x > y,
    'ge': lambda x, y: x >= y,
    
    # ========== Logical Operations ==========
    # JAX primitives: 'and', 'or', 'not' → JAXADi: jnp.logical_and/or/not
    'and': lambda x, y: ca.logic_and(x, y),
    'or': lambda x, y: ca.logic_or(x, y),
    'not': lambda x: ca.logic_not(x),
    
    # ========== Rounding & Sign Operations ==========
    'floor': ca.floor,
    'ceil': ca.ceil,
    'abs': ca.fabs,  # JAXADi: OP_FABS -> jnp.abs
    'sign': ca.sign,
    
    # ========== Min/Max Operations ==========
    # JAXADi: OP_FMIN -> jnp.minimum, OP_FMAX -> jnp.maximum
    'min': ca.fmin,
    'max': ca.fmax,
    
    # ========== Special Functions ==========
    'erf': ca.erf,  # JAXADi: OP_ERF -> jax.scipy.special.erf
    
    # ========== Other Operations ==========
    'fmod': ca.fmod,  # JAXADi: OP_FMOD -> jnp.fmod
    'copysign': ca.copysign,
    
    # ========== Type Conversion ==========
    # CasADi is dynamically typed, these are no-ops
    'convert_element_type': lambda x, **kwargs: x,
    'bitcast_convert_type': lambda x, **kwargs: x,
    
    # ========== Finite/NaN/Inf Checks ==========
    # In symbolic context, we assume all expressions are finite
    # These checks are meaningful at runtime, but for symbolic optimization
    # we return True (1.0) as we assume finite values
    'is_finite': lambda x: ca.DM.ones(x.shape),  # Assume finite in symbolic context
    
    # ========== Gradient Operations ==========
    # stop_gradient: Stops gradient propagation in reverse mode
    # In forward-only CasADi context (optimization), this is a no-op
    'stop_gradient': lambda x: x,  # No-op in forward mode
}


# ============================================================================
# OPERATION CATEGORIES
# For better error messages and debugging
# Based on JAXADi's operation structure
# ============================================================================

ARITHMETIC_OPS = {'add', 'sub', 'mul', 'div', 'neg', 'pow', 'integer_pow', 'sqrt', 'sq', 'inv'}
TRIGONOMETRIC_OPS = {'sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'atan2'}
HYPERBOLIC_OPS = {'sinh', 'cosh', 'tanh', 'asinh', 'acosh', 'atanh'}
EXPONENTIAL_OPS = {'exp', 'log'}
COMPARISON_OPS = {'lt', 'le', 'eq', 'ne', 'gt', 'ge'}
LOGICAL_OPS = {'and', 'or', 'not'}
ROUNDING_OPS = {'floor', 'ceil', 'abs', 'sign'}
MINMAX_OPS = {'min', 'max'}
SPECIAL_OPS = {'erf', 'fmod', 'copysign', 'convert_element_type', 'bitcast_convert_type', 'is_finite', 'stop_gradient'}

SUPPORTED_PRIMITIVES = set(JAX_PRIMITIVE_TO_CASADI.keys())

# ============================================================================
# ARRAY OPERATIONS
# These need special handling in the expression generator
# ============================================================================

ARRAY_OPERATIONS = {
    'slice',
    'dynamic_slice', 
    'dynamic_update_slice',
    'concatenate',
    'reshape',
    'transpose',
    'broadcast_in_dim',
    'squeeze',
    'expand_dims',
    'dot_general',  # Matrix multiplication
    'reduce_sum',
    'reduce_max',
    'reduce_min',
    'reduce_prod',
    'gather',
    'scatter',
    'pad',  # Array padding operation
    'jit',  # JIT wrapper (contains nested jaxpr)
    'pjit',  # Parallel JIT wrapper (contains nested jaxpr)
    'select_n',  # Multi-way conditional selection
}

# ============================================================================
# CONTROL FLOW
# Limited support - document restrictions
# ============================================================================

CONTROL_FLOW_OPS = {
    'cond',  # if-else
    'while',  # while loop
    'scan',  # sequential scan
}

# ============================================================================
# UNSUPPORTED OPERATIONS
# Document what doesn't work
# ============================================================================

KNOWN_UNSUPPORTED = {
    'fft',
    'conv_general_dilated',
    'custom_jvp',
    'custom_vjp',
    'pjit',
    'xla_call',
}


def get_operation_category(primitive_name: str) -> str:
    """
    Get the category of a JAX primitive for error messages
    
    Args:
        primitive_name: Name of JAX primitive
        
    Returns:
        Category string
        
    Example:
        >>> get_operation_category('sin')
        'trigonometric'
        >>> get_operation_category('add')
        'arithmetic'
    """
    if primitive_name in ARITHMETIC_OPS:
        return "arithmetic"
    elif primitive_name in TRIGONOMETRIC_OPS:
        return "trigonometric"
    elif primitive_name in HYPERBOLIC_OPS:
        return "hyperbolic"
    elif primitive_name in EXPONENTIAL_OPS:
        return "exponential"
    elif primitive_name in COMPARISON_OPS:
        return "comparison"
    elif primitive_name in LOGICAL_OPS:
        return "logical"
    elif primitive_name in ROUNDING_OPS:
        return "rounding"
    elif primitive_name in MINMAX_OPS:
        return "min/max"
    elif primitive_name in SPECIAL_OPS:
        return "special"
    elif primitive_name in ARRAY_OPERATIONS:
        return "array"
    elif primitive_name in CONTROL_FLOW_OPS:
        return "control flow"
    elif primitive_name in KNOWN_UNSUPPORTED:
        return "known unsupported"
    else:
        return "unknown"


def is_supported(primitive_name: str) -> bool:
    """
    Check if a JAX primitive is supported
    
    Args:
        primitive_name: Name of JAX primitive
        
    Returns:
        True if supported, False otherwise
        
    Example:
        >>> is_supported('sin')
        True
        >>> is_supported('fft')
        False
    """
    return (primitive_name in JAX_PRIMITIVE_TO_CASADI or 
            primitive_name in ARRAY_OPERATIONS)


def list_supported_operations() -> Dict[str, list]:
    """
    List all supported operations by category
    
    Returns:
        Dictionary mapping category names to lists of operations
    """
    categories = {
        'Arithmetic': sorted(ARITHMETIC_OPS),
        'Trigonometric': sorted(TRIGONOMETRIC_OPS),
        'Hyperbolic': sorted(HYPERBOLIC_OPS),
        'Exponential': sorted(EXPONENTIAL_OPS),
        'Comparison': sorted(COMPARISON_OPS),
        'Logical': sorted(LOGICAL_OPS),
        'Rounding': sorted(ROUNDING_OPS),
        'Min/Max': sorted(MINMAX_OPS),
        'Special': sorted(SPECIAL_OPS),
        'Array': sorted(ARRAY_OPERATIONS),
    }
    return categories


def print_supported_operations():
    """Print all supported operations in a formatted table"""
    categories = list_supported_operations()
    
    print("=" * 60)
    print("JAX2CasADi Supported Operations")
    print("=" * 60)
    
    for category, ops in categories.items():
        print(f"\n{category} ({len(ops)} operations):")
        for op in ops:
            print(f"  - {op}")
    
    print(f"\nTotal: {sum(len(ops) for ops in categories.values())} operations")
    print("=" * 60)

