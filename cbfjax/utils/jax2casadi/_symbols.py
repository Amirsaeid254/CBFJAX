"""
CasADi Symbol Manager

Handles creation and management of CasADi symbolic variables.
Maintains mapping between Jaxpr variables and CasADi expressions.
"""

import casadi as ca
from jax import core
from jax._src import core as src_core  # For Literal, Var types
from typing import Dict, Tuple, Any, Union
import numpy as np


CasADiExpression = Union[ca.MX, ca.SX, ca.DM]


class SymbolManager:
    """
    Manages CasADi symbolic variables and their mapping to Jaxpr variables
    """
    
    def __init__(self, use_sx: bool = False):
        """
        Initialize symbol manager
        
        Args:
            use_sx: Use SX instead of MX (SX is faster but less flexible)
        """
        self.use_sx = use_sx
        self.env: Dict[ CasADiExpression] = {}
        self.symbol_type = ca.SX if use_sx else ca.MX
        
    def create_symbol(self, name: str, shape: Tuple[int, ...]) -> CasADiExpression:
        """
        Create a CasADi symbolic variable
        
        Args:
            name: Variable name
            shape: Variable shape
            
        Returns:
            CasADi symbolic variable
            
        Example:
            >>> sm = SymbolManager()
            >>> x = sm.create_symbol('x', (3,))  # 3D vector
            >>> A = sm.create_symbol('A', (3, 3))  # 3x3 matrix
        """
        if len(shape) == 0:
            # Scalar
            return self.symbol_type.sym(name)
        elif len(shape) == 1:
            # Vector
            return self.symbol_type.sym(name, shape[0])
        elif len(shape) == 2:
            # Matrix
            return self.symbol_type.sym(name, shape[0], shape[1])
        else:
            # Higher dimensional - flatten
            total_size = int(np.prod(shape))
            sym = self.symbol_type.sym(name, total_size)
            # Store original shape as metadata
            sym.shape_hint = shape
            return sym
    
    def register_variable(self, var, casadi_expr: CasADiExpression):
        """
        Register a Jaxpr variable with its CasADi expression
        
        Args:
            var: Jaxpr variable
            casadi_expr: Corresponding CasADi expression
        """
        self.env[var] = casadi_expr
    
    def get_expression(self, var) -> CasADiExpression:
        """
        Get CasADi expression for a Jaxpr variable
        
        Args:
            var: Jaxpr variable
            
        Returns:
            CasADi expression
            
        Raises:
            KeyError: If variable not registered
        """
        if isinstance(var, src_core.Literal):
            # Literal constant - convert directly
            return self._literal_to_casadi(var)
        
        if var not in self.env:
            raise KeyError(f"Variable {var} not found in environment")
        
        return self.env[var]
    
    def _literal_to_casadi(self, literal) -> CasADiExpression:
        """Convert JAX literal to CasADi constant"""
        val = literal.val
        
        if np.isscalar(val):
            return ca.DM(float(val))
        else:
            # Array literal
            return ca.DM(np.array(val))
    
    def has_variable(self, var) -> bool:
        """Check if variable is registered"""
        return var in self.env or isinstance(var, src_core.Literal)
    
    def clear(self):
        """Clear all registered variables"""
        self.env.clear()
    
    def __len__(self) -> int:
        """Number of registered variables"""
        return len(self.env)
    
    def __repr__(self) -> str:
        return f"SymbolManager(variables={len(self.env)}, type={'SX' if self.use_sx else 'MX'})"


def infer_shape_from_jaxpr_var(var) -> Tuple[int, ...]:
    """
    Infer shape from Jaxpr variable
    
    Args:
        var: Jaxpr variable
        
    Returns:
        Shape tuple
    """
    if isinstance(var, src_core.Literal):
        val = var.val
        if np.isscalar(val):
            return ()
        else:
            return np.array(val).shape
    
    if hasattr(var, 'aval') and hasattr(var.aval, 'shape'):
        return var.aval.shape
    
    # Default to scalar
    return ()


def create_input_symbols(
    jaxpr_vars: list,
    input_specs: list,
    symbol_manager: SymbolManager
) -> list:
    """
    Create CasADi symbols for JAX function inputs
    
    Args:
        jaxpr_vars: List of Jaxpr input variables
        input_specs: List of (name, shape) tuples
        symbol_manager: Symbol manager to register with
        
    Returns:
        List of CasADi symbolic inputs
        
    Example:
        >>> specs = [('x', (2,)), ('u', (1,))]
        >>> sm = SymbolManager()
        >>> inputs = create_input_symbols(jaxpr_info.invars, specs, sm)
    """
    if len(jaxpr_vars) != len(input_specs):
        raise ValueError(
            f"Number of Jaxpr inputs ({len(jaxpr_vars)}) doesn't match "
            f"number of input specs ({len(input_specs)})"
        )
    
    casadi_inputs = []
    
    for jaxpr_var, (name, shape) in zip(jaxpr_vars, input_specs):
        # Create CasADi symbol
        casadi_sym = symbol_manager.create_symbol(name, shape)
        
        # Register in environment
        symbol_manager.register_variable(jaxpr_var, casadi_sym)
        
        # Add to outputs
        casadi_inputs.append(casadi_sym)
    
    return casadi_inputs


def extract_output_expressions(
    jaxpr_outvars: list,
    symbol_manager: SymbolManager
) -> list:
    """
    Extract CasADi expressions for output variables
    
    Args:
        jaxpr_outvars: List of Jaxpr output variables
        symbol_manager: Symbol manager with registered variables
        
    Returns:
        List of CasADi expressions
    """
    casadi_outputs = []
    
    for var in jaxpr_outvars:
        expr = symbol_manager.get_expression(var)
        casadi_outputs.append(expr)
    
    return casadi_outputs


class ShapeTracker:
    """Track shapes through computation for validation"""
    
    def __init__(self):
        self.shapes: Dict[ Tuple[int, ...]] = {}
    
    def register_shape(self, var, shape: Tuple[int, ...]):
        """Register shape for a variable"""
        self.shapes[var] = shape
    
    def get_shape(self, var) -> Tuple[int, ...]:
        """Get shape of a variable"""
        if var in self.shapes:
            return self.shapes[var]
        return infer_shape_from_jaxpr_var(var)
    
    def verify_shapes(self, var1, var2) -> bool:
        """Verify two variables have compatible shapes"""
        shape1 = self.get_shape(var1)
        shape2 = self.get_shape(var2)
        
        # Broadcasting rules (simplified)
        if shape1 == shape2:
            return True
        
        # One is scalar
        if shape1 == () or shape2 == ():
            return True
        
        # TODO: Full broadcasting rules
        return False
