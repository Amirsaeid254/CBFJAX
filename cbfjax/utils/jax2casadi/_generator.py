"""
CasADi Expression Generator

Applies JAX operations to build CasADi symbolic expressions.
Handles array operations, slicing, reshaping, etc.
"""

import casadi as ca
from jax import core
from typing import List, Any, Dict, Tuple
import numpy as np

from ._ops import JAX_PRIMITIVE_TO_CASADI, ARRAY_OPERATIONS, is_supported, get_operation_category
from ._symbols import SymbolManager, CasADiExpression


class ExpressionGenerator:
    """
    Generates CasADi expressions from Jaxpr equations
    """
    
    def __init__(self, symbol_manager: SymbolManager):
        """
        Initialize expression generator
        
        Args:
            symbol_manager: Symbol manager for variable lookup
        """
        self.symbol_manager = symbol_manager
    
    def process_equation(self, eqn) -> CasADiExpression:
        """
        Process a single Jaxpr equation to generate CasADi expression
        
        Args:
            eqn: Jaxpr equation
            
        Returns:
            CasADi expression
            
        Raises:
            NotImplementedError: If operation not supported
        """
        primitive_name = eqn.primitive.name
        
        # Get input expressions
        input_exprs = [self.symbol_manager.get_expression(var) for var in eqn.invars]
        
        # Check if operation is supported
        if not is_supported(primitive_name):
            category = get_operation_category(primitive_name)
            raise NotImplementedError(
                f"JAX primitive '{primitive_name}' (category: {category}) is not yet supported.\n"
                f"Please open an issue at: https://github.com/yourusername/jax2casadi/issues"
            )
        
        # Apply operation
        if primitive_name in JAX_PRIMITIVE_TO_CASADI:
            # Direct mapping
            result = self._apply_primitive(primitive_name, input_exprs, eqn.params)
        elif primitive_name in ARRAY_OPERATIONS:
            # Array operation - needs special handling
            result = self._apply_array_operation(primitive_name, input_exprs, eqn.params)
        else:
            raise NotImplementedError(f"Primitive '{primitive_name}' not implemented")
        
        return result
    
    def _apply_primitive(
        self,
        primitive_name: str,
        inputs: List[CasADiExpression],
        params: Dict[str, Any]
    ) -> CasADiExpression:
        """
        Apply a primitive operation
        
        Args:
            primitive_name: Name of the primitive
            inputs: Input CasADi expressions
            params: Operation parameters
            
        Returns:
            Result expression
        """
        op = JAX_PRIMITIVE_TO_CASADI[primitive_name]
        
        # Handle parameters if present
        if params:
            # Some operations have keyword parameters
            try:
                return op(*inputs, **params)
            except TypeError:
                # Operation doesn't accept these params
                return op(*inputs)
        else:
            return op(*inputs)
    
    def _apply_array_operation(
        self,
        op_name: str,
        inputs: List[CasADiExpression],
        params: Dict[str, Any]
    ) -> CasADiExpression:
        """
        Apply array operation with special handling
        
        Args:
            op_name: Operation name
            inputs: Input expressions
            params: Operation parameters
            
        Returns:
            Result expression
        """
        if op_name == 'slice':
            return self._handle_slice(inputs[0], params)
        elif op_name == 'dynamic_slice':
            return self._handle_dynamic_slice(inputs, params)
        elif op_name == 'concatenate':
            return self._handle_concatenate(inputs, params)
        elif op_name == 'reshape':
            return self._handle_reshape(inputs[0], params)
        elif op_name == 'transpose':
            return self._handle_transpose(inputs[0], params)
        elif op_name == 'broadcast_in_dim':
            return self._handle_broadcast(inputs[0], params)
        elif op_name == 'dot_general':
            return self._handle_dot_general(inputs, params)
        elif op_name == 'reduce_sum':
            return self._handle_reduce_sum(inputs[0], params)
        elif op_name == 'reduce_max':
            return self._handle_reduce_max(inputs[0], params)
        elif op_name == 'reduce_min':
            return self._handle_reduce_min(inputs[0], params)
        elif op_name == 'squeeze':
            return self._handle_squeeze(inputs[0], params)
        elif op_name == 'expand_dims':
            return self._handle_expand_dims(inputs[0], params)
        elif op_name == 'gather':
            return self._handle_gather(inputs, params)
        elif op_name == 'scatter':
            return self._handle_scatter(inputs, params)
        elif op_name == 'pad':
            return self._handle_pad(inputs, params)
        elif op_name == 'select_n':
            return self._handle_select_n(inputs, params)
        elif op_name == 'jit':
            return self._handle_jit(inputs, params)
        elif op_name == 'pjit':
            return self._handle_pjit(inputs, params)
        else:
            raise NotImplementedError(f"Array operation '{op_name}' not yet implemented")
    
    # ========== Array Operation Handlers ==========
    
    def _handle_slice(self, x: CasADiExpression, params: Dict) -> CasADiExpression:
        """
        Handle array slicing
        
        JAX: x[start:limit:stride]
        CasADi: x[start:limit] (stride must be 1)
        
        Key challenge: JAX and CasADi may have different shape orientations
        - JAX: (1, n) row vectors from atleast_2d
        - CasADi: (n, 1) column vectors by default
        
        This handler intelligently detects and handles these mismatches.
        """
        start_indices = params.get('start_indices', (0,))
        limit_indices = params.get('limit_indices')
        strides = params.get('strides', None)
        
        if strides is not None and any(s != 1 for s in strides):
            raise NotImplementedError("Strided slicing not supported in CasADi")
        
        # Get actual CasADi shape
        casadi_shape = x.shape if hasattr(x, 'shape') else ()
        casadi_rows = int(casadi_shape[0]) if len(casadi_shape) > 0 else 1
        casadi_cols = int(casadi_shape[1]) if len(casadi_shape) > 1 else 1
        casadi_total = casadi_rows * casadi_cols
        
        if len(start_indices) == 1:
            # 1D slice: x[start:end]
            start = int(start_indices[0])
            end = int(limit_indices[0])
            slice_len = end - start
            
            # Determine how to slice based on CasADi shape
            if casadi_cols == 1:
                # Column vector (n, 1) - slice along rows
                if end <= casadi_rows:
                    if slice_len == 1:
                        return x[start, 0]
                    else:
                        return x[start:end, 0]
                else:
                    raise ValueError(f"1D slice [{start}:{end}] out of bounds for shape {casadi_shape}")
            elif casadi_rows == 1:
                # Row vector (1, n) - slice along cols
                if end <= casadi_cols:
                    if slice_len == 1:
                        return x[0, start]
                    else:
                        return x[0, start:end]
                else:
                    raise ValueError(f"1D slice [{start}:{end}] out of bounds for shape {casadi_shape}")
            else:
                # True 2D matrix - treat as flattened
                if slice_len == 1:
                    return x[start]
                else:
                    return x[start:end]
                    
        elif len(start_indices) == 2:
            # 2D slice: x[row_start:row_end, col_start:col_end]
            row_start, col_start = int(start_indices[0]), int(start_indices[1])
            row_end, col_end = int(limit_indices[0]), int(limit_indices[1])
            
            jax_rows = row_end - row_start
            jax_cols = col_end - col_start
            
            # Check if JAX and CasADi shapes are transposed relative to each other
            # JAX might want (1, n) slice but CasADi has (m, 1) column vector
            
            # Case 1: CasADi shape is compatible as-is
            if row_end <= casadi_rows and col_end <= casadi_cols:
                return x[row_start:row_end, col_start:col_end]
            
            # Case 2: JAX wants (1, n) from row 0, but CasADi has (m, 1) column vector
            # Example: JAX slice [0:1, 0:2] on what JAX thinks is (1, 4)
            #          but CasADi has (4, 1)
            if (row_end == 1 and col_end > 1 and 
                casadi_cols == 1 and casadi_rows >= col_end):
                # JAX thinks it's slicing columns from a row vector
                # But CasADi has a column vector - slice rows instead
                result = x[col_start:col_end, 0]
                # Return as (1, n) row vector to match JAX expectation
                return ca.transpose(result)
            
            # Case 3: JAX wants (n, 1) from col 0, but CasADi has (1, m) row vector
            if (col_end == 1 and row_end > 1 and
                casadi_rows == 1 and casadi_cols >= row_end):
                # JAX thinks it's slicing rows from a column vector
                # But CasADi has a row vector - slice cols instead
                result = x[0, row_start:row_end]
                # Return as (n, 1) column vector to match JAX expectation
                return ca.transpose(result)
            
            # Case 4: CasADi has transposed shape - transpose and slice
            if col_end <= casadi_rows and row_end <= casadi_cols:
                x_t = ca.transpose(x)
                return x_t[row_start:row_end, col_start:col_end]
            
            # Case 5: Empty slice (this can happen with atleast_2d edge cases)
            if jax_rows == 0 or jax_cols == 0:
                # Return empty matrix of correct shape
                return ca.MX(jax_rows, jax_cols)
            
            # Case 6: Last resort - detailed error
            raise ValueError(
                f"Slice mismatch: JAX requests [{row_start}:{row_end}, {col_start}:{col_end}] "
                f"(shape {jax_rows}x{jax_cols}) but CasADi array has shape ({casadi_rows}, {casadi_cols}). "
                f"This may indicate a shape tracking issue in the conversion."
            )
        else:
            raise NotImplementedError(f"Slicing with {len(start_indices)} dimensions not supported")
    
    def _handle_dynamic_slice(self, inputs: List, params: Dict) -> CasADiExpression:
        """Handle dynamic slicing (runtime-determined indices)"""
        # This is challenging in symbolic math
        raise NotImplementedError(
            "Dynamic slicing requires runtime indices, which is incompatible with "
            "symbolic CasADi expressions. Consider using static slicing instead."
        )
    
    def _handle_concatenate(self, inputs: List, params: Dict) -> CasADiExpression:
        """
        Handle array concatenation
        
        JAX: jnp.concatenate([a, b], axis=0)
        CasADi: ca.vertcat(a, b) or ca.horzcat(a, b)
        """
        dimension = params.get('dimension', 0)
        
        if dimension == 0:
            # Vertical concatenation
            return ca.vertcat(*inputs)
        elif dimension == 1:
            # Horizontal concatenation
            return ca.horzcat(*inputs)
        else:
            raise NotImplementedError(f"Concatenation along axis {dimension} not supported")
    
    def _handle_reshape(self, x: CasADiExpression, params: Dict) -> CasADiExpression:
        """
        Handle array reshaping
        
        JAX: x.reshape(new_shape)
        CasADi: ca.reshape(x, new_shape)
        """
        new_sizes = params.get('new_sizes')
        dimensions = params.get('dimensions')
        
        if len(new_sizes) > 2:
            raise NotImplementedError("Reshaping to >2D not supported")
        
        if len(new_sizes) == 1:
            # Reshape to vector
            return ca.reshape(x, new_sizes[0], 1)
        elif len(new_sizes) == 2:
            # Reshape to matrix
            return ca.reshape(x, new_sizes[0], new_sizes[1])
        else:
            # Scalar
            return x
    
    def _handle_transpose(self, x: CasADiExpression, params: Dict) -> CasADiExpression:
        """
        Handle matrix transpose
        
        JAX: x.T or jnp.transpose(x, axes)
        CasADi: x.T
        """
        permutation = params.get('permutation', (1, 0))
        
        if permutation == (1, 0):
            # Standard 2D transpose
            return x.T
        else:
            # General transpose (not supported in CasADi)
            raise NotImplementedError(f"Transpose with permutation {permutation} not supported")
    
    def _handle_broadcast(self, x: CasADiExpression, params: Dict) -> CasADiExpression:
        """
        Handle broadcasting (broadcast_in_dim)
        
        JAX: broadcast_in_dim broadcasts a value to a larger shape
        CasADi: needs explicit tiling/repmat for most cases
        
        Key patterns:
        - Scalar to vector: x -> [x, x, ..., x]
        - Scalar to matrix: x -> [[x, ...], [...]]
        - Vector to matrix: add dimension
        """
        shape = params.get('shape')
        broadcast_dimensions = params.get('broadcast_dimensions', ())
        
        if shape is None:
            return x
        
        # Get current shape
        x_shape = x.shape if hasattr(x, 'shape') else ()
        x_rows = int(x_shape[0]) if len(x_shape) > 0 else 1
        x_cols = int(x_shape[1]) if len(x_shape) > 1 else 1
        x_is_scalar = (x_rows == 1 and x_cols == 1)
        
        target_shape = tuple(int(s) for s in shape)
        
        # Case 1: Scalar to any shape
        if x_is_scalar:
            if len(target_shape) == 1:
                # Scalar to 1D vector
                n = target_shape[0]
                return ca.repmat(x, n, 1)
            elif len(target_shape) == 2:
                # Scalar to 2D matrix
                rows, cols = target_shape
                return ca.repmat(x, rows, cols)
            else:
                raise NotImplementedError(f"Broadcast scalar to shape {target_shape} not supported")
        
        # Case 2: Vector to matrix (add dimension)
        if len(target_shape) == 2:
            target_rows, target_cols = target_shape
            
            # Check if we need to tile
            if x_cols == 1 and target_cols > 1 and x_rows == target_rows:
                # Column vector to matrix - tile columns
                return ca.repmat(x, 1, target_cols)
            elif x_rows == 1 and target_rows > 1 and x_cols == target_cols:
                # Row vector to matrix - tile rows
                return ca.repmat(x, target_rows, 1)
            elif x_rows == target_rows and x_cols == target_cols:
                # Already correct shape
                return x
            elif x_cols == 1 and x_rows == 1:
                # Scalar - tile to target shape
                return ca.repmat(x, target_rows, target_cols)
            # Check if shapes are transposed
            elif x_rows == target_cols and x_cols == target_rows:
                # Might need transpose
                return ca.transpose(x)
        
        # Case 3: 1D to 1D (might need reshape)
        if len(target_shape) == 1:
            target_n = target_shape[0]
            total_elements = x_rows * x_cols
            
            if total_elements == target_n:
                # Reshape to vector
                return ca.reshape(x, target_n, 1)
            elif total_elements == 1:
                # Scalar to vector
                return ca.repmat(x, target_n, 1)
        
        # Default: return as-is and let CasADi handle it
        # This works for many automatic broadcasting cases
        return x
    
    def _handle_dot_general(self, inputs: List, params: Dict) -> CasADiExpression:
        """
        Handle generalized dot product (matrix multiplication)
        
        JAX: jax.lax.dot_general(a, b, dimension_numbers)
        CasADi: ca.mtimes(a, b) with proper transposes
        
        Key issue: JAX vectors are 1D, CasADi vectors are (n,1) column vectors.
        For vector operations, we need to transpose appropriately.
        """
        if len(inputs) != 2:
            raise ValueError("dot_general requires exactly 2 inputs")
        
        a, b = inputs
        dimension_numbers = params.get('dimension_numbers')
        
        # Get shapes
        a_shape = a.shape
        b_shape = b.shape
        
        # dimension_numbers = (((contracting_dims_a,), (contracting_dims_b,)), 
        #                      ((batch_dims_a,), (batch_dims_b,)))
        contracting_dims = dimension_numbers[0] if dimension_numbers else ((), ())
        contract_a, contract_b = contracting_dims
        
        # Determine if inputs are vectors (column vectors in CasADi)
        # CasADi represents 1D as (n, 1), so check if second dimension is 1
        a_is_vector = len(a_shape) == 2 and a_shape[1] == 1
        b_is_vector = len(b_shape) == 2 and b_shape[1] == 1
        
        # Case 1: Vector @ Vector (dot product)
        # JAX: (n,) @ (n,) = scalar
        # CasADi: (n,1) @ (n,1) → need (1,n) @ (n,1) = (1,1) = scalar
        if a_is_vector and b_is_vector:
            # Transpose first vector to row vector
            a_T = ca.transpose(a)  # (n,1) → (1,n)
            result = ca.mtimes(a_T, b)  # (1,n) @ (n,1) = (1,1)
            # Return as scalar if result is (1,1)
            if result.shape == (1, 1):
                return result[0, 0]
            return result
        
        # Case 2: Vector @ Matrix
        # JAX: (n,) @ (n,m) = (m,)
        # CasADi: (n,1) @ (n,m) → need (1,n) @ (n,m) = (1,m)
        elif a_is_vector and not b_is_vector:
            # Check if we're contracting on axis 0 of a
            if contract_a == (0,):
                a_T = ca.transpose(a)  # (n,1) → (1,n)
                result = ca.mtimes(a_T, b)  # (1,n) @ (n,m) = (1,m)
                # Squeeze result back to column vector for consistency
                if result.shape[0] == 1:
                    return ca.transpose(result)  # (1,m) → (m,1)
                return result
            else:
                # Standard multiplication
                return ca.mtimes(a, b)
        
        # Case 3: Matrix @ Vector  
        # JAX: (n,m) @ (m,) = (n,)
        # CasADi: (n,m) @ (m,1) = (n,1) ✓ This works as-is
        elif not a_is_vector and b_is_vector:
            return ca.mtimes(a, b)
        
        # Case 4: Matrix @ Matrix
        # Standard matrix multiplication
        else:
            return ca.mtimes(a, b)
    
    def _handle_reduce_sum(self, x: CasADiExpression, params: Dict) -> CasADiExpression:
        """
        Handle sum reduction
        
        JAX: jnp.sum(x, axis)
        CasADi: ca.sum1(x) or ca.sum2(x)
        """
        axes = params.get('axes', None)
        
        if axes is None or axes == ():
            # Sum all elements
            return ca.sum1(ca.sum2(x))
        elif len(axes) == 1:
            axis = axes[0]
            if axis == 0:
                # Sum along axis 0 (sum columns)
                return ca.sum1(x)
            elif axis == 1:
                # Sum along axis 1 (sum rows)
                return ca.sum2(x)
        
        # Multiple axes
        result = x
        for axis in sorted(axes, reverse=True):
            if axis == 0:
                result = ca.sum1(result)
            elif axis == 1:
                result = ca.sum2(result)
        return result
    
    def _handle_reduce_max(self, x: CasADiExpression, params: Dict) -> CasADiExpression:
        """
        Handle max reduction
        
        JAX: jnp.max(x, axis)
        CasADi: mmax(x) for global max, manual iteration for axis-wise
        
        Supports:
        - Global max (no axis or all axes)
        - Axis-wise max for 1D and 2D arrays
        """
        axes = params.get('axes', None)
        shape = x.shape
        ndim = len(shape)
        
        # Check if this is a global reduction (all axes)
        is_global = (axes is None or 
                    axes == () or 
                    (axes is not None and len(axes) == ndim))
        
        if is_global:
            # Max of all elements
            return ca.mmax(x)
        
        # Axis-wise reduction - implement manually
        if len(axes) == 1:
            axis = int(axes[0])
            
            if ndim == 1:
                # 1D array - any axis is global
                return ca.mmax(x)
            
            elif ndim == 2:
                rows, cols = int(shape[0]), int(shape[1])
                
                if axis == 0:
                    # Max along axis 0 (max of each column) -> shape (cols,) or (1, cols)
                    result_list = []
                    for j in range(cols):
                        col = x[:, j]
                        col_max = ca.mmax(col)
                        result_list.append(col_max)
                    
                    # Return as row vector
                    if len(result_list) == 1:
                        return result_list[0]
                    result = ca.horzcat(*result_list)
                    # Transpose to column vector for consistency with JAX
                    return ca.transpose(result)
                    
                elif axis == 1:
                    # Max along axis 1 (max of each row) -> shape (rows,) or (rows, 1)
                    result_list = []
                    for i in range(rows):
                        row = x[i, :]
                        row_max = ca.mmax(row)
                        result_list.append(row_max)
                    
                    # Return as column vector
                    if len(result_list) == 1:
                        return result_list[0]
                    result = ca.vertcat(*result_list)
                    return result
                else:
                    raise ValueError(f"Invalid axis {axis} for 2D array")
            
            else:
                raise NotImplementedError(f"reduce_max for {ndim}D arrays not yet implemented (only 1D and 2D supported)")
        
        else:
            # Multiple axes - reduce sequentially
            result = x
            for ax in sorted(axes, reverse=True):
                result = self._handle_reduce_max(result, {'axes': (ax,)})
            return result
    
    def _handle_reduce_min(self, x: CasADiExpression, params: Dict) -> CasADiExpression:
        """
        Handle min reduction
        
        JAX: jnp.min(x, axis)
        CasADi: mmin(x) for global min, manual iteration for axis-wise
        
        Supports:
        - Global min (no axis or all axes)
        - Axis-wise min for 1D and 2D arrays
        """
        axes = params.get('axes', None)
        shape = x.shape
        ndim = len(shape)
        
        # Check if this is a global reduction (all axes)
        is_global = (axes is None or 
                    axes == () or 
                    (axes is not None and len(axes) == ndim))
        
        if is_global:
            # Min of all elements
            return ca.mmin(x)
        
        # Axis-wise reduction - implement manually
        if len(axes) == 1:
            axis = int(axes[0])
            
            if ndim == 1:
                # 1D array - any axis is global
                return ca.mmin(x)
            
            elif ndim == 2:
                rows, cols = int(shape[0]), int(shape[1])
                
                if axis == 0:
                    # Min along axis 0 (min of each column) -> shape (cols,) or (1, cols)
                    result_list = []
                    for j in range(cols):
                        col = x[:, j]
                        col_min = ca.mmin(col)
                        result_list.append(col_min)
                    
                    # Return as row vector
                    if len(result_list) == 1:
                        return result_list[0]
                    result = ca.horzcat(*result_list)
                    # Transpose to column vector for consistency with JAX
                    return ca.transpose(result)
                    
                elif axis == 1:
                    # Min along axis 1 (min of each row) -> shape (rows,) or (rows, 1)
                    result_list = []
                    for i in range(rows):
                        row = x[i, :]
                        row_min = ca.mmin(row)
                        result_list.append(row_min)
                    
                    # Return as column vector
                    if len(result_list) == 1:
                        return result_list[0]
                    result = ca.vertcat(*result_list)
                    return result
                else:
                    raise ValueError(f"Invalid axis {axis} for 2D array")
            
            else:
                raise NotImplementedError(f"reduce_min for {ndim}D arrays not yet implemented (only 1D and 2D supported)")
        
        else:
            # Multiple axes - reduce sequentially
            result = x
            for ax in sorted(axes, reverse=True):
                result = self._handle_reduce_min(result, {'axes': (ax,)})
            return result
    
    def _handle_squeeze(self, x: CasADiExpression, params: Dict) -> CasADiExpression:
        """Handle squeeze (remove dimensions of size 1)"""
        # CasADi handles this implicitly in most cases
        return x
    
    def _handle_expand_dims(self, x: CasADiExpression, params: Dict) -> CasADiExpression:
        """Handle expand_dims (add dimension of size 1)"""
        dimensions = params.get('dimensions', (0,))
        # For most CasADi cases, this is implicit
        return x
    
    def _handle_gather(self, inputs: List, params: Dict) -> CasADiExpression:
        """Handle gather operation (advanced indexing)"""
        raise NotImplementedError("gather operation requires advanced indexing not supported in CasADi")
    
    def _handle_scatter(self, inputs: List, params: Dict) -> CasADiExpression:
        """Handle scatter operation (advanced indexing)"""
        raise NotImplementedError("scatter operation requires advanced indexing not supported in CasADi")
    
    def _handle_select_n(self, inputs: List, params: Dict) -> CasADiExpression:
        """
        Handle multi-way conditional selection
        
        JAX: jax.lax.select_n(index, *cases)
        Returns: cases[index]
        
        Example:
            select_n(0, a, b, c) -> a
            select_n(1, a, b, c) -> b
            select_n(2, a, b, c) -> c
        
        CasADi: Nested if_else statements
        """
        if len(inputs) < 2:
            raise ValueError("select_n requires at least index and one case")
        
        index = inputs[0]
        cases = inputs[1:]  # All remaining inputs are cases
        
        if len(cases) == 1:
            # Only one case - just return it
            return cases[0]
        
        if len(cases) == 2:
            # Two cases - simple if_else (index == 0 ? case0 : case1)
            return ca.if_else(index == 0, cases[0], cases[1])
        
        # Multiple cases - build nested if_else from right to left
        # Start with the last case as default
        result = cases[-1]
        
        # Work backwards through cases, wrapping in if_else
        for i in range(len(cases) - 2, -1, -1):
            # if index == i, return cases[i], else continue to previous result
            result = ca.if_else(index == i, cases[i], result)
        
        return result

    
    def _handle_pad(self, inputs: List, params: Dict) -> CasADiExpression:
        """
        Handle array padding operation
        
        JAX: jnp.pad(array, pad_width, constant_values=0)
        CasADi: Use vertcat/horzcat with constant arrays
        
        Args:
            inputs: [array_to_pad, padding_value]
            params: {'padding_config': ((low_pad, high_pad, stride), ...)}
        
        Returns:
            Padded array
        """
        if len(inputs) < 2:
            raise ValueError("pad operation requires at least 2 inputs: array and padding value")
        
        array = inputs[0]
        pad_value = inputs[1]  # Scalar padding value
        
        padding_config = params.get('padding_config', ())
        
        if len(padding_config) == 0:
            # No padding
            return array
        
        # Get array shape
        array_shape = array.shape
        ndim = len(array_shape)
        
        # CasADi represents 1D arrays as 2D column vectors (n, 1)
        # So if we have shape (n, 1) and padding_config for 1D, treat as 1D
        if ndim == 2 and array_shape[1] == 1 and len(padding_config) == 1:
            # This is actually a 1D array in CasADi's representation
            ndim = 1
            array_shape = (array_shape[0],)
        
        if len(padding_config) != ndim:
            raise ValueError(f"padding_config length ({len(padding_config)}) doesn't match array dimensions ({ndim})")
        
        result = array
        
        # Handle 1D case
        if ndim == 1:
            low_pad, high_pad, stride = padding_config[0]
            
            if stride != 0:
                raise NotImplementedError("Strided padding not supported")
            
            if low_pad > 0 or high_pad > 0:
                # Create padding arrays
                if low_pad > 0:
                    low_padding = ca.DM.ones(int(low_pad), 1) * pad_value
                    result = ca.vertcat(low_padding, result)
                
                if high_pad > 0:
                    high_padding = ca.DM.ones(int(high_pad), 1) * pad_value
                    result = ca.vertcat(result, high_padding)
        
        # Handle 2D case
        elif ndim == 2:
            rows, cols = array_shape
            row_pad_low, row_pad_high, row_stride = padding_config[0]
            col_pad_low, col_pad_high, col_stride = padding_config[1]
            
            if row_stride != 0 or col_stride != 0:
                raise NotImplementedError("Strided padding not supported")
            
            # Pad rows first (top and bottom)
            if row_pad_low > 0:
                top_padding = ca.DM.ones(int(row_pad_low), cols) * pad_value
                result = ca.vertcat(top_padding, result)
            
            if row_pad_high > 0:
                bottom_padding = ca.DM.ones(int(row_pad_high), cols) * pad_value
                result = ca.vertcat(result, bottom_padding)
            
            # Update rows after padding
            new_rows = int(rows + row_pad_low + row_pad_high)
            
            # Pad columns (left and right)
            if col_pad_low > 0:
                left_padding = ca.DM.ones(new_rows, int(col_pad_low)) * pad_value
                result = ca.horzcat(left_padding, result)
            
            if col_pad_high > 0:
                right_padding = ca.DM.ones(new_rows, int(col_pad_high)) * pad_value
                result = ca.horzcat(result, right_padding)
        
        else:
            raise NotImplementedError(f"Padding for {ndim}D arrays not yet supported (only 1D and 2D)")
        
        return result
    
    def _handle_jit(self, inputs: List, params: Dict) -> CasADiExpression:
        """
        Handle JIT wrapper primitive
        
        JAX sometimes wraps operations (like pad) in a jit primitive.
        We need to process the nested jaxpr inside.
        
        Args:
            inputs: Input expressions to the jit operation
            params: Contains 'jaxpr' - the nested jaxpr to process
        
        Returns:
            Result of processing the nested jaxpr
        """
        nested_jaxpr = params.get('jaxpr')
        
        if nested_jaxpr is None:
            raise ValueError("jit primitive missing 'jaxpr' parameter")
        
        # Handle both ClosedJaxpr and Jaxpr
        # ClosedJaxpr has .jaxpr attribute, Jaxpr has .invars directly
        if hasattr(nested_jaxpr, 'jaxpr'):
            # It's a ClosedJaxpr - extract the inner Jaxpr
            actual_jaxpr = nested_jaxpr.jaxpr
        else:
            # It's already a Jaxpr
            actual_jaxpr = nested_jaxpr
        
        # Create a temporary environment for the nested jaxpr
        # Map the nested jaxpr's inputs to our current inputs
        for jaxpr_var, casadi_expr in zip(actual_jaxpr.invars, inputs):
            self.symbol_manager.register_variable(jaxpr_var, casadi_expr)
        
        # Process each equation in the nested jaxpr
        for eqn in actual_jaxpr.eqns:
            result = self.process_equation(eqn)
            
            # Register outputs
            if len(eqn.outvars) == 1:
                self.symbol_manager.register_variable(eqn.outvars[0], result)
            else:
                if isinstance(result, (list, tuple)):
                    for out_var, res in zip(eqn.outvars, result):
                        self.symbol_manager.register_variable(out_var, res)
                else:
                    self.symbol_manager.register_variable(eqn.outvars[0], result)
        
        # Extract the output(s) from the nested jaxpr
        nested_outputs = [self.symbol_manager.get_expression(var) for var in actual_jaxpr.outvars]
        
        # Return single output or tuple
        if len(nested_outputs) == 1:
            return nested_outputs[0]
        else:
            return tuple(nested_outputs)
    
    def _handle_pjit(self, inputs: List, params: Dict) -> CasADiExpression:
        """
        Handle pjit (parallel JIT) wrapper primitive
        
        pjit is JAX's parallel/distributed JIT compilation primitive.
        For CasADi conversion (symbolic), it works exactly like jit -
        we just need to process the nested jaxpr.
        
        Args:
            inputs: Input expressions to the pjit operation
            params: Contains 'jaxpr' - the nested jaxpr to process
        
        Returns:
            Result of processing the nested jaxpr
        """
        # pjit works exactly like jit for symbolic conversion
        return self._handle_jit(inputs, params)
