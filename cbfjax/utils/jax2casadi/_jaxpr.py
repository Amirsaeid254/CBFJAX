"""
Jaxpr Parser - Extract computation graph from JAX's intermediate representation

This module parses JAX's Jaxpr (JAX expression) IR to extract:
- Input variables
- Output variables  
- Computational equations
- Operation primitives
- Parameters

Jaxpr Structure:
```
{ lambda ; a:f32[] b:f32[]. let
    c:f32[] = sin a
    d:f32[] = add c b
  in (d,) }
```
"""

import jax
import jax.numpy as jnp
from jax import core
from jax._src import core as src_core
from typing import List, Tuple, Dict, Any
import numpy as np


class JaxprInfo:
    """Container for parsed Jaxpr information"""
    
    def __init__(self, jaxpr):
        self.jaxpr = jaxpr
        self.invars: List = list(jaxpr.invars)
        self.outvars: List = list(jaxpr.outvars)
        self.eqns: List = list(jaxpr.eqns)
        self.constvars: List = list(jaxpr.constvars)
        
    def __repr__(self) -> str:
        return f"JaxprInfo(inputs={len(self.invars)}, outputs={len(self.outvars)}, equations={len(self.eqns)})"


def trace_jax_function(func: callable, input_shapes: List[Tuple[int, ...]]):
    """
    Trace a JAX function to obtain its Jaxpr
    
    Args:
        func: JAX function to trace
        input_shapes: List of input shapes
        
    Returns:
        ClosedJaxpr: Traced JAX expression
        
    Example:
        >>> def f(x, u):
        ...     return jnp.sin(x) + u
        >>> jaxpr = trace_jax_function(f, [(2,), (2,)])
    """
    # Create dummy inputs with the specified shapes
    dummy_inputs = [jnp.ones(shape) for shape in input_shapes]
    
    # Trace the function
    closed_jaxpr = jax.make_jaxpr(func)(*dummy_inputs)
    
    return closed_jaxpr


def parse_jaxpr(closed_jaxpr) -> JaxprInfo:
    """
    Parse a ClosedJaxpr into structured information
    
    Args:
        closed_jaxpr: Traced JAX expression
        
    Returns:
        JaxprInfo: Parsed information
    """
    return JaxprInfo(closed_jaxpr.jaxpr)


def get_variable_shape(var) :
    """
    Extract shape from a Jaxpr variable
    
    Args:
        var: Jaxpr variable
        
    Returns:
        Shape tuple
    """
    if hasattr(var, 'aval'):
        return var.aval.shape
    else:
        # Fallback
        return ()


def get_variable_dtype(var):
    """
    Extract dtype from a Jaxpr variable
    
    Args:
        var: Jaxpr variable
        
    Returns:
        numpy dtype
    """
    if hasattr(var, 'aval'):
        return var.aval.dtype
    else:
        return jnp.float64


def get_equation_inputs(eqn) :
    """Get input variables of an equation"""
    return list(eqn.invars)


def get_equation_outputs(eqn) :
    """Get output variables of an equation"""
    return list(eqn.outvars)


def get_primitive_name(eqn) -> str:
    """Get the name of the primitive operation"""
    return eqn.primitive.name


def get_equation_params(eqn) -> Dict[str, Any]:
    """Get parameters of an equation"""
    return dict(eqn.params)


def is_literal(var) -> bool:
    """Check if a variable is a literal constant"""
    return isinstance(var, src_core.Literal)


def get_literal_value(var):
    """Get the value of a literal constant"""
    return var.val


def build_variable_dependency_graph(jaxpr_info: JaxprInfo) :
    """
    Build a dependency graph showing which variables depend on which
    
    Args:
        jaxpr_info: Parsed Jaxpr information
        
    Returns:
        Dictionary mapping variables to their dependencies
    """
    dependencies = {}
    
    for eqn in jaxpr_info.eqns:
        for outvar in eqn.outvars:
            dependencies[outvar] = list(eqn.invars)
    
    return dependencies


def topological_sort_equations(jaxpr_info: JaxprInfo) :
    """
    Sort equations in topological order (already done by JAX, but useful for verification)
    
    Args:
        jaxpr_info: Parsed Jaxpr information
        
    Returns:
        Topologically sorted equations
    """
    # JAX already provides equations in topological order
    # But we can verify this if needed
    return jaxpr_info.eqns


class JaxprDebugPrinter:
    """Utility class to pretty-print Jaxpr for debugging"""
    
    @staticmethod
    def print_jaxpr(jaxpr_info: JaxprInfo):
        """Print Jaxpr in human-readable format"""
        print("=" * 60)
        print("JAXPR DEBUG INFO")
        print("=" * 60)
        
        print(f"\nInputs ({len(jaxpr_info.invars)}):")
        for i, var in enumerate(jaxpr_info.invars):
            shape = get_variable_shape(var)
            dtype = get_variable_dtype(var)
            print(f"  [{i}] {var} : {dtype}{shape}")
        
        print(f"\nConstants ({len(jaxpr_info.constvars)}):")
        for i, var in enumerate(jaxpr_info.constvars):
            print(f"  [{i}] {var}")
        
        print(f"\nEquations ({len(jaxpr_info.eqns)}):")
        for i, eqn in enumerate(jaxpr_info.eqns):
            prim_name = get_primitive_name(eqn)
            invars = get_equation_inputs(eqn)
            outvars = get_equation_outputs(eqn)
            params = get_equation_params(eqn)
            
            print(f"  [{i}] {outvars} = {prim_name}({invars})", end="")
            if params:
                print(f" {params}", end="")
            print()
        
        print(f"\nOutputs ({len(jaxpr_info.outvars)}):")
        for i, var in enumerate(jaxpr_info.outvars):
            shape = get_variable_shape(var)
            dtype = get_variable_dtype(var)
            print(f"  [{i}] {var} : {dtype}{shape}")
        
        print("=" * 60)
    
    @staticmethod
    def print_equation(eqn, index: int = None):
        """Print a single equation"""
        prefix = f"[{index}]" if index is not None else ""
        prim_name = get_primitive_name(eqn)
        invars = get_equation_inputs(eqn)
        outvars = get_equation_outputs(eqn)
        params = get_equation_params(eqn)
        
        print(f"{prefix} {outvars} = {prim_name}({invars})", end="")
        if params:
            print(f" {params}", end="")
        print()


# Convenience functions
def trace_and_parse(func: callable, input_shapes: List[Tuple[int, ...]]) -> JaxprInfo:
    """One-step trace and parse"""
    closed_jaxpr = trace_jax_function(func, input_shapes)
    return parse_jaxpr(closed_jaxpr)


def debug_jax_function(func: callable, input_shapes: List[Tuple[int, ...]]):
    """Trace, parse, and print Jaxpr for debugging"""
    jaxpr_info = trace_and_parse(func, input_shapes)
    JaxprDebugPrinter.print_jaxpr(jaxpr_info)
    return jaxpr_info
