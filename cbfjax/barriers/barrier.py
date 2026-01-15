"""
Barrier class for Control Barrier Functions.

This module implements barrier functions and higher-order control barrier functions
using automatic differentiation. The design ensures immutability and functional
purity for computational efficiency.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from typing import List, Callable, Optional, Any, Dict
from immutabledict import immutabledict

from cbfjax.utils.utils import apply_and_batchize, apply_and_batchize_tuple
from cbfjax.dynamics.base_dynamic import DummyDynamics


class Barrier(eqx.Module):
    """
    Barrier function implementation for control barrier functions.

    This class implements barrier functions and higher-order control barrier functions
    (HOCBFs) using automatic differentiation. All data structures are designed to be
    immutable and hashable.

    Attributes:
        _barrier_func: The barrier function R^n -> R
        _dynamics: System dynamics object with f and g methods
        _rel_deg: Relative degree of the barrier function
        _alphas: Tuple of class-K functions for higher-order barriers
        _barriers: Tuple of all barrier functions in the series
        _hocbf_func: The highest-order barrier function
        cfg: Configuration dictionary as immutable mapping
    """

    # Core barrier configuration
    _barrier_func: Callable = eqx.field(static=True)
    _dynamics: Any = eqx.field(static=True)
    _rel_deg: int = eqx.field(static=True)
    _alphas: tuple = eqx.field(static=True)

    # Computed barrier functions
    _barriers: tuple = eqx.field(static=True)
    _hocbf_func: Callable = eqx.field(static=True)

    # Configuration
    cfg: immutabledict = eqx.field(static=True)

    def __init__(self, barrier_func=None, dynamics=None, rel_deg=1, alphas=None,
                 barriers=None, hocbf_func=None, cfg=None):
        """
        Initialize Barrier instance.

        Args:
            barrier_func: Barrier function R^n -> R (None uses dummy function)
            dynamics: System dynamics object (None uses dummy dynamics)
            rel_deg: Relative degree for higher-order barriers
            alphas: List/tuple of class-K functions
            barriers: List/tuple of barrier function series
            hocbf_func: Highest-order barrier function (None uses dummy)
            cfg: Configuration dictionary
        """
        # Initialize with defaults to avoid None values
        self._barrier_func = barrier_func or self._create_dummy_barrier()
        self._dynamics = dynamics or DummyDynamics()
        self._rel_deg = rel_deg
        self._alphas = tuple(alphas or [])
        self._barriers = tuple(barriers or [])
        self._hocbf_func = hocbf_func or self._create_dummy_barrier()
        self.cfg = immutabledict(cfg or {})


    @staticmethod
    def _create_dummy_barrier():
        """Create dummy barrier function that returns empty array."""
        def dummy_barrier(x):
            return jnp.array([])
        return dummy_barrier

    @classmethod
    def create_empty(cls, cfg=None):
        """
        Create an empty barrier instance.

        Args:
            cfg: Optional configuration dictionary

        Returns:
            Empty Barrier instance ready for assignment
        """
        return cls(cfg=cfg)

    def _create_updated_instance(self, **kwargs):
        """
        Create new instance with updated fields.

        This helper method enables clean assignment methods by collecting
        all current field values and updating only the changed ones.

        Args:
            **kwargs: Fields to update

        Returns:
            New instance of the same class with updated fields
        """
        defaults = {
            'barrier_func': self._barrier_func,
            'dynamics': self._dynamics,
            'rel_deg': self._rel_deg,
            'alphas': self._alphas,
            'barriers': self._barriers,
            'hocbf_func': self._hocbf_func,
            'cfg': self.cfg
        }
        defaults.update(kwargs)
        return self.__class__(**defaults)

    def assign(self, barrier_func: Callable, rel_deg: int = 1,
               alphas: Optional[List[Callable]] = None) -> 'Barrier':
        """
        Create new Barrier with assigned barrier function.

        Args:
            barrier_func: Barrier function R^n -> R
            rel_deg: Relative degree for higher-order barriers
            alphas: List of class-K functions for HOCBF construction

        Returns:
            New Barrier instance with assigned function

        Raises:
            AssertionError: If barrier_func is not callable
        """
        assert callable(barrier_func), "barrier_func must be a callable function"
        processed_alphas = self._handle_alphas(alphas, rel_deg)

        return self._create_updated_instance(
            barrier_func=barrier_func,
            rel_deg=rel_deg,
            alphas=processed_alphas
        )

    def assign_dynamics(self, dynamics) -> 'Barrier':
        """
        Create new Barrier with assigned dynamics and computed HOCBF.

        Args:
            dynamics: System dynamics object with f and g methods

        Returns:
            New Barrier instance with computed barrier series

        Raises:
            ValueError: If barrier function not assigned first
        """
        if self._is_dummy_barrier(self._barrier_func):
            raise ValueError("Barrier function must be assigned first. Use assign() method.")

        # Generate higher-order barrier function series
        barriers = self._make_hocbf_series(
            barrier=self._barrier_func,
            dynamics=dynamics,
            rel_deg=self._rel_deg,
            alphas=self._alphas
        )
        hocbf_func = barriers[-1]

        return self._create_updated_instance(
            dynamics=dynamics,
            barriers=tuple(barriers),
            hocbf_func=hocbf_func
        )

    def _is_dummy_barrier(self, func):
        """Check if function is a dummy barrier."""
        return func.__name__ == 'dummy_barrier'

    def _barrier_single(self, x: jnp.ndarray) -> float:
        """Compute barrier function value for single state vector."""
        return self._barrier_func(x)

    def barrier(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Compute barrier function value(s) with consistent batching.

        Args:
            x: State vector (n,) or batch (batch, n)

        Returns:
            Barrier value(s) with shape (batch, 1)
        """
        return apply_and_batchize(self._barrier_func, x)

    def _hocbf_single(self, x: jnp.ndarray) -> float:
        """Compute HOCBF value for single state vector."""
        if len(self._barriers) == 0:
            raise ValueError("HOCBF not computed. Use assign_dynamics() first.")
        return self._hocbf_func(x)

    def hocbf(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Compute highest-order barrier function value(s).

        Args:
            x: State vector (n,) or batch (batch, n)

        Returns:
            HOCBF value(s) with shape (batch, 1)

        Raises:
            ValueError: If HOCBF not computed (dynamics not assigned)
        """
        if len(self._barriers) == 0:
            raise ValueError("HOCBF not computed. Use assign_dynamics() first.")
        return apply_and_batchize(self._hocbf_func, x)

    def get_hocbf_and_lie_derivs(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Compute HOCBF and its Lie derivatives efficiently.

        Args:
            x: State vector (batch, n)

        Returns:
            Tuple of (hocbf_value, Lf_hocbf, Lg_hocbf) with shapes:
            - hocbf_value: (batch, 1)
            - Lf_hocbf: (batch, 1)
            - Lg_hocbf: (batch, action_dim)
        """
        return apply_and_batchize_tuple(self._get_hocbf_and_lie_derivs_single, x)


    def _get_hocbf_and_lie_derivs_single (self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Compute HOCBF and its Lie derivatives.

        Args:
            x: State vector (n,)

        Returns:
            Tuple of (hocbf_value, Lf_hocbf, Lg_hocbf) with shapes:
            - hocbf_value: (1,)
            - Lf_hocbf: (1,)
            - Lg_hocbf: (action_dim,)
        """

        hocbf_val, grad_hocbf = jax.value_and_grad(self._hocbf_func)(x)


        f_val = self._dynamics.f(x)
        g_val = self._dynamics.g(x)

        Lf_hocbf = jnp.dot(grad_hocbf, f_val)
        Lg_hocbf = grad_hocbf @ g_val

        return hocbf_val, Lf_hocbf, Lg_hocbf




    def Lf_hocbf(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Compute Lie derivative of HOCBF with respect to drift dynamics.

        Args:
            x: State vector (n,) or batch (batch, n)

        Returns:
            Lie derivative Lf(hocbf) with shape (batch, 1)
        """
        def compute_single(x_single):
            grad_hocbf = jax.grad(self._hocbf_func)(x_single)
            f_val = self._dynamics.f(x_single)
            return jnp.dot(grad_hocbf, f_val)

        return apply_and_batchize(compute_single, x)

    def Lg_hocbf(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Compute Lie derivative of HOCBF with respect to control dynamics.

        Args:
            x: State vector (n,) or batch (batch, n)

        Returns:
            Lie derivative Lg(hocbf) with shape (batch, action_dim)
        """
        def compute_single(x_single):
            grad_hocbf = jax.grad(self._hocbf_func)(x_single)
            g_val = self._dynamics.g(x_single)
            return grad_hocbf @ g_val

        return apply_and_batchize(compute_single, x)

    def compute_barriers_at(self, x: jnp.ndarray) -> List[jnp.ndarray]:
        """
        Compute all barrier values in the series at given state(s).

        Args:
            x: State vector (n,) or batch (batch, n)

        Returns:
            List of barrier values, each with appropriate batch dimensions
        """
        if len(self._barriers) == 0:
            raise ValueError("Barriers not computed. Use assign_dynamics() first.")
        return [apply_and_batchize(barrier, x) for barrier in self.barriers_flatten]

    def get_min_barrier_at(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Get minimum barrier value among all barriers in the series.

        Args:
            x: State vector (n,) or batch (batch, n)

        Returns:
            Minimum barrier value(s) with shape (batch, 1)
        """
        barrier_vals = self.compute_barriers_at(x)
        stacked_vals = jnp.concatenate(barrier_vals, axis=1)
        return jnp.min(stacked_vals, axis=1, keepdims=True)

    def min_barrier(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Calculate the minimum value among all barrier values computed at point x.

        This method matches CBFTorch behavior by computing the minimum across
        the base barrier function values (not the full barrier series).

        Args:
            x: State vector (n,) or batch (batch, n)

        Returns:
            Minimum barrier value with shape (batch, 1)
        """
        barrier_vals = self.barrier(x)  # Get base barrier values
        return jnp.min(barrier_vals, axis=-1, keepdims=True)

    def _make_hocbf_series(self, barrier: Callable, dynamics, rel_deg: int,
                          alphas: tuple) -> List[Callable]:
        """
        Generate higher-order barrier function series.

        Creates a series of barrier functions: [h, Lf(h) + α₁(h), ...]
        where each function represents one level of the HOCBF construction.

        Args:
            barrier: Initial barrier function
            dynamics: System dynamics object
            rel_deg: Relative degree (number of derivatives)
            alphas: Tuple of class-K functions

        Returns:
            List of barrier functions in the series
        """
        barriers = [barrier]

        for i in range(rel_deg - 1):
            # Get current alpha function, avoiding closure capture of full tuple
            current_alpha = alphas[i] if i < len(alphas) else (lambda x: x)
            current_prev_barrier = barriers[i]

            def create_next_hocbf(prev_barrier, alpha, f_dynamics):
                def next_hocbf(x):
                    grad_prev = jax.grad(prev_barrier)(x)
                    f_val = f_dynamics(x)
                    lie_deriv_val = jnp.dot(grad_prev, f_val)
                    alpha_val = alpha(prev_barrier(x))
                    return lie_deriv_val + alpha_val
                return next_hocbf

            next_barrier = create_next_hocbf(current_prev_barrier, current_alpha, dynamics.f)
            barriers.append(next_barrier)

        return barriers

    def _handle_alphas(self, alphas: Optional[List[Callable]], rel_deg: int) -> tuple:
        """
        Process and validate alpha functions for HOCBF construction.

        Args:
            alphas: List of class-K functions or None
            rel_deg: Relative degree

        Returns:
            Tuple of validated alpha functions

        Raises:
            AssertionError: If alphas format is invalid
        """
        if rel_deg > 1:
            if alphas is None:
                # Create identity functions for each level
                def create_identity_alpha():
                    def identity_alpha(x):
                        return x
                    return identity_alpha
                alphas = [create_identity_alpha() for _ in range(rel_deg - 1)]

            assert (isinstance(alphas, (list, tuple)) and
                   len(alphas) == rel_deg - 1 and
                   callable(alphas[0])), \
                   "alphas must be a list/tuple with length (rel_deg - 1) of callable functions"

            return tuple(alphas)
        return tuple()

    # Properties for external access
    @property
    def rel_deg(self) -> int:
        """Relative degree of the barrier function."""
        return self._rel_deg

    @property
    def barriers(self) -> tuple:
        """Tuple of all barrier functions in the series."""
        return self._barriers

    @property
    def barriers_flatten(self) -> tuple:
        """Flattened tuple of barrier functions (same as barriers for single barrier)."""
        return self.barriers

    @property
    def dynamics(self):
        """Associated system dynamics object."""
        return self._dynamics

    @property
    def num_barriers(self) -> int:
        """Number of barrier functions in the series."""
        return len(self._barriers)


