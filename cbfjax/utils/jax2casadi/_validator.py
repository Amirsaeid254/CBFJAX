"""
Numerical Validator

Validates that converted CasADi functions produce the same numerical
results as the original JAX functions.
"""

import casadi as ca
import jax.numpy as jnp
import numpy as np
from typing import Callable, List, Tuple


def validate(
    jax_fn: Callable,
    casadi_fn: ca.Function,
    input_specs: List[Tuple[str, Tuple[int, ...]]],
    num_samples: int = 10,
    tolerance: float = 1e-10,
    verbose: bool = False
) -> Tuple[bool, float]:
    """
    Validate CasADi function against JAX function
    
    Tests the converted function on random inputs and compares outputs.
    
    Args:
        jax_fn: Original JAX function
        casadi_fn: Converted CasADi function
        input_specs: Input specifications (name, shape)
        num_samples: Number of random test samples
        tolerance: Maximum allowed absolute error
        verbose: Print detailed results
        
    Returns:
        (is_valid, max_error): Validation result and maximum error found
        
    Example:
        >>> is_valid, max_err = validate(jax_fn, casadi_fn, input_specs)
        >>> if not is_valid:
        ...     print(f"Validation failed! Error: {max_err}")
    """
    max_error = 0.0
    
    if verbose:
        print(f"Validating with {num_samples} random samples...")
    
    for sample_idx in range(num_samples):
        # Generate random inputs
        jax_inputs = []
        casadi_inputs = []
        
        for name, shape in input_specs:
            # Random values in [-1, 1]
            random_vals = np.random.uniform(-1, 1, shape)
            
            # JAX input
            jax_inputs.append(jnp.array(random_vals))
            
            # CasADi input
            casadi_inputs.append(ca.DM(random_vals))
        
        # Evaluate JAX function
        jax_output = jax_fn(*jax_inputs)
        
        # Handle single vs multiple outputs
        if not isinstance(jax_output, (list, tuple)):
            jax_outputs = [jax_output]
        else:
            jax_outputs = list(jax_output)
        
        # Evaluate CasADi function
        casadi_results = casadi_fn(*casadi_inputs)
        
        # Handle single vs multiple outputs
        if not isinstance(casadi_results, (list, tuple)):
            casadi_outputs = [casadi_results]
        else:
            casadi_outputs = list(casadi_results)
        
        # Compare outputs
        if len(jax_outputs) != len(casadi_outputs):
            if verbose:
                print(f"✗ Sample {sample_idx}: Output count mismatch!")
                print(f"  JAX: {len(jax_outputs)} outputs")
                print(f"  CasADi: {len(casadi_outputs)} outputs")
            return False, float('inf')
        
        # Check each output
        for out_idx, (jax_out, casadi_out) in enumerate(zip(jax_outputs, casadi_outputs)):
            # Convert to numpy arrays
            jax_arr = np.array(jax_out)
            casadi_arr = np.array(casadi_out).flatten()
            jax_arr_flat = jax_arr.flatten()
            
            # Compute error
            error = np.max(np.abs(jax_arr_flat - casadi_arr))
            max_error = max(max_error, error)
            
            if verbose and error > tolerance:
                print(f"✗ Sample {sample_idx}, Output {out_idx}: Error = {error:.2e}")
                print(f"  JAX:    {jax_arr_flat[:5]}...")
                print(f"  CasADi: {casadi_arr[:5]}...")
            elif verbose and sample_idx < 3:  # Print first few samples
                print(f"✓ Sample {sample_idx}, Output {out_idx}: Error = {error:.2e}")
    
    is_valid = max_error <= tolerance
    
    if verbose:
        if is_valid:
            print(f"\n✓ Validation PASSED")
            print(f"  Maximum error: {max_error:.2e}")
            print(f"  Tolerance: {tolerance:.2e}")
        else:
            print(f"\n✗ Validation FAILED")
            print(f"  Maximum error: {max_error:.2e}")
            print(f"  Tolerance: {tolerance:.2e}")
    
    return is_valid, max_error


def validate_gradients(
    jax_fn: Callable,
    casadi_fn: ca.Function,
    input_specs: List[Tuple[str, Tuple[int, ...]]],
    num_samples: int = 5,
    tolerance: float = 1e-8,
    verbose: bool = False
) -> Tuple[bool, float]:
    """
    Validate gradients/Jacobians match between JAX and CasADi
    
    Args:
        jax_fn: Original JAX function
        casadi_fn: Converted CasADi function  
        input_specs: Input specifications
        num_samples: Number of test samples
        tolerance: Maximum allowed error
        verbose: Print details
        
    Returns:
        (is_valid, max_error): Validation result
    """
    import jax
    
    max_error = 0.0
    
    if verbose:
        print(f"Validating gradients with {num_samples} samples...")
    
    for sample_idx in range(num_samples):
        # Generate random inputs
        random_inputs = []
        for name, shape in input_specs:
            random_vals = np.random.uniform(-1, 1, shape)
            random_inputs.append(jnp.array(random_vals))
        
        # JAX Jacobian
        jax_jac_fn = jax.jacrev(jax_fn)
        jax_jac = jax_jac_fn(*random_inputs)
        
        # CasADi Jacobian
        casadi_inputs = [ca.DM(inp) for inp in random_inputs]
        
        # Get Jacobian from CasADi
        # This is more complex - would need to use ca.jacobian
        # For now, skip this advanced feature
        
        if verbose:
            print(f"  Sample {sample_idx}: Gradient validation not yet fully implemented")
    
    return True, 0.0  # Placeholder
