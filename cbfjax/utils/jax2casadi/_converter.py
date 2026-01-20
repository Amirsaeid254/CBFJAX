"""
Main Converter: JAX Function → CasADi Function

This module provides the main convert() function that orchestrates the entire
JAX to CasADi conversion pipeline.

Pipeline:
    JAX Function → Jaxpr → Parse → Build CasADi Expressions → ca.Function
"""

import jax
import jax.numpy as jnp
import casadi as ca
from typing import Callable, List, Tuple, Optional, Union
from jax import core

from ._jaxpr import trace_jax_function, parse_jaxpr, JaxprInfo, get_primitive_name, get_equation_params
from ._symbols import SymbolManager, create_input_symbols, extract_output_expressions
from ._generator import ExpressionGenerator
from ._ops import is_supported, get_operation_category


class ConversionError(Exception):
    """Custom exception for conversion errors"""
    pass


def convert(
    jax_fn: Callable,
    input_specs: List[Tuple[str, Tuple[int, ...]]],
    name: str = 'converted',
    use_sx: bool = False,
    validate: bool = False,
    tolerance: float = 1e-6,
    debug: bool = False
) -> ca.Function:
    """
    Convert a JAX function to a CasADi Function
    
    This is the main entry point for JAX2CasADi conversion. It traces the JAX
    function to obtain its Jaxpr representation, parses it, and builds equivalent
    CasADi symbolic expressions.
    
    Args:
        jax_fn: JAX function to convert (can be @jax.jit decorated)
        input_specs: List of (name, shape) tuples specifying inputs
                    Example: [('x', (2,)), ('u', (1,))]
        name: Name for the resulting CasADi function (default: 'converted')
        use_sx: Use SX instead of MX (SX is faster but less flexible)
        validate: Numerically validate the conversion (default: False)
        tolerance: Validation tolerance (default: 1e-6)
        debug: Print debug information during conversion (default: False)
        
    Returns:
        ca.Function: CasADi symbolic function
        
    Raises:
        ConversionError: If conversion fails
        
    Example:
        >>> import jax.numpy as jnp
        >>> from jax2casadi import convert
        >>> 
        >>> def dynamics(x, u):
        ...     return jnp.array([x[1], -jnp.sin(x[0]) + u[0]])
        >>> 
        >>> casadi_fn = convert(dynamics, [('x', (2,)), ('u', (1,))])
        >>> 
        >>> # Now use with acados, CasADi optimization, etc.
        >>> x_test = ca.DM([1.0, 0.5])
        >>> u_test = ca.DM([0.3])
        >>> result = casadi_fn(x_test, u_test)
    """
    
    if debug:
        print("=" * 60)
        print("JAX2CasADi Conversion")
        print("=" * 60)
        print(f"Function: {jax_fn.__name__ if hasattr(jax_fn, '__name__') else 'anonymous'}")
        print(f"Inputs: {input_specs}")
        print(f"Output name: {name}")
        print()
    
    try:
        # Step 1: Trace JAX function to get Jaxpr
        if debug:
            print("Step 1: Tracing JAX function to Jaxpr...")
        
        input_shapes = [shape for _, shape in input_specs]
        closed_jaxpr = trace_jax_function(jax_fn, input_shapes)
        jaxpr_info = parse_jaxpr(closed_jaxpr)
        
        if debug:
            print(f"  → Found {len(jaxpr_info.eqns)} equations")
            print(f"  → {len(jaxpr_info.invars)} inputs, {len(jaxpr_info.outvars)} outputs")
            print()
        
        # Step 2: Create CasADi symbol manager
        if debug:
            print("Step 2: Creating CasADi symbols...")
        
        symbol_manager = SymbolManager(use_sx=use_sx)
        casadi_inputs = create_input_symbols(jaxpr_info.invars, input_specs, symbol_manager)
        
        # Register constant variables (constvars)
        # These are constants captured in closures (e.g., arrays defined in function)
        if len(jaxpr_info.constvars) > 0:
            if debug:
                print(f"  → Registering {len(jaxpr_info.constvars)} constant variables")
            
            # closed_jaxpr.consts contains the actual constant values
            for const_var, const_val in zip(jaxpr_info.constvars, closed_jaxpr.consts):
                # Convert numpy/jax array to CasADi constant
                import numpy as np
                const_array = np.array(const_val)
                casadi_const = ca.DM(const_array)
                symbol_manager.register_variable(const_var, casadi_const)
                
                if debug:
                    print(f"    Registered constvar: {const_var} = {const_array.flatten()[:3]}...")
        
        if debug:
            print(f"  → Created {len(casadi_inputs)} input symbols")
            print()
        
        # Step 3: Build CasADi expressions
        if debug:
            print("Step 3: Building CasADi expressions...")
        
        generator = ExpressionGenerator(symbol_manager)
        
        for i, eqn in enumerate(jaxpr_info.eqns):
            prim_name = get_primitive_name(eqn)
            
            if debug and i < 10:  # Print first 10 for debugging
                print(f"  [{i}] {prim_name}")
            
            # Check if operation is supported
            if not is_supported(prim_name):
                category = get_operation_category(prim_name)
                raise ConversionError(
                    f"Unsupported JAX primitive: '{prim_name}' (category: {category})\n"
                    f"This operation cannot be converted to CasADi.\n"
                    f"Please open an issue at: https://github.com/yourusername/jax2casadi/issues"
                )
            
            # Process the equation to build CasADi expression
            result = generator.process_equation(eqn)
            
            # Register the result(s) in the symbol manager
            if isinstance(result, (list, tuple)):
                # Multiple outputs
                for out_var, res in zip(eqn.outvars, result):
                    symbol_manager.register_variable(out_var, res)
            else:
                # Single output
                symbol_manager.register_variable(eqn.outvars[0], result)
        
        if debug:
            print(f"  → Processed {len(jaxpr_info.eqns)} equations")
            print()
        
        # Step 4: Extract outputs
        if debug:
            print("Step 4: Extracting outputs...")
        
        casadi_outputs = extract_output_expressions(jaxpr_info.outvars, symbol_manager)
        
        if debug:
            print(f"  → Extracted {len(casadi_outputs)} outputs")
            print()
        
        # Step 5: Create CasADi Function
        if debug:
            print("Step 5: Creating CasADi Function...")
        
        input_names = [name for name, _ in input_specs]
        output_names = [f'out{i}' for i in range(len(casadi_outputs))]
        
        casadi_function = ca.Function(
            name,
            casadi_inputs,
            casadi_outputs,
            input_names,
            output_names
        )
        
        if debug:
            print(f"  → Created function: {casadi_function}")
            print()
        
        # Step 6: Validation (optional)
        if validate:
            if debug:
                print("Step 6: Validating conversion...")
            
            _validate_conversion(jax_fn, casadi_function, input_shapes, tol=tolerance, debug=debug)
            
            if debug:
                print("  → Validation passed!")
                print()
        
        if debug:
            print("=" * 60)
            print("Conversion successful!")
            print("=" * 60)
        
        return casadi_function
        
    except ConversionError:
        raise
    except Exception as e:
        raise ConversionError(
            f"Conversion failed with error: {type(e).__name__}: {str(e)}"
        ) from e


def _validate_conversion(
    jax_fn: Callable,
    casadi_fn: ca.Function,
    input_shapes: List[Tuple[int, ...]],
    tol: float = 1e-10,
    debug: bool = False
) -> bool:
    """
    Validate that the CasADi function produces the same output as JAX
    
    Args:
        jax_fn: Original JAX function
        casadi_fn: Converted CasADi function
        input_shapes: List of input shapes
        tol: Numerical tolerance (default: 1e-10)
        debug: Print debug information
        
    Returns:
        True if validation passes
        
    Raises:
        ConversionError: If outputs don't match
    """
    import numpy as np
    
    # Generate random test inputs
    np.random.seed(42)
    jax_inputs = [np.random.randn(*shape) for shape in input_shapes]
    casadi_inputs = [ca.DM(inp) for inp in jax_inputs]
    
    # Run JAX function
    jax_output = jax_fn(*jax_inputs)
    if not isinstance(jax_output, (list, tuple)):
        jax_output = [jax_output]
    
    # Run CasADi function
    casadi_output = casadi_fn(*casadi_inputs)
    if not isinstance(casadi_output, (list, tuple)):
        casadi_output = [casadi_output]
    
    # Compare outputs
    for i, (jax_out, casadi_out) in enumerate(zip(jax_output, casadi_output)):
        # Flatten both arrays for consistent comparison
        jax_out = np.array(jax_out).flatten()
        casadi_out = np.array(casadi_out).flatten()
        
        error = np.max(np.abs(jax_out - casadi_out))
        
        if debug:
            print(f"  Output {i}: max error = {error:.2e}")
        
        if error > tol:
            raise ConversionError(
                f"Validation failed! Output {i} error = {error:.2e} > tolerance {tol}\n"
                f"JAX output:\n{jax_out}\n"
                f"CasADi output:\n{casadi_out}"
            )
    
    return True
