"""
Barrier class for Control Barrier Functions.

This module implements barrier functions and higher-order control barrier functions
using automatic differentiation. Barriers are complete at construction: the HOCBF
series is derived on demand from the stored fields rather than precomputed closures,
so instances are valid pytrees and numeric parameters stay traced.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from typing import List, Callable, Optional, Any
from immutabledict import immutabledict

from cbfjax.dynamics.base_dynamic import DummyDynamics


class Barrier(eqx.Module):
    """
    Barrier function implementation for control barrier functions.

    Implements barrier functions and higher-order control barrier functions
    (HOCBFs) using automatic differentiation. A barrier is fully usable once
    constructed with a barrier function and dynamics:

        barrier = Barrier(barrier_func=h, dynamics=dyn, rel_deg=2, alphas=(10.0,))

    alphas may be numbers (linear class-K gains, stored as traced leaves) or
    callables (stored statically).
    """

    _barrier_func: Callable = eqx.field(static=True)
    _dynamics: Any
    _rel_deg: int = eqx.field(static=True)
    _alphas_static: tuple = eqx.field(static=True)
    _alpha_coefs: Optional[jnp.ndarray]
    _explicit_barriers: tuple = eqx.field(static=True)
    _explicit_hocbf_func: Optional[Callable] = eqx.field(static=True)
    cfg: immutabledict = eqx.field(static=True)

    def __init__(self, barrier_func=None, dynamics=None, rel_deg=1, alphas=None,
                 barriers=None, hocbf_func=None, cfg=None):
        """
        Initialize Barrier instance.

        Args:
            barrier_func: Barrier function R^n -> R (None uses dummy function)
            dynamics: System dynamics object (None uses dummy dynamics)
            rel_deg: Relative degree for higher-order barriers
            alphas: List/tuple of class-K gains (numbers) or functions
            barriers: Optional explicit barrier function series (overrides derivation)
            hocbf_func: Optional explicit highest-order barrier function
            cfg: Configuration dictionary
        """
        self._barrier_func = barrier_func or self._create_dummy_barrier()
        self._dynamics = dynamics if dynamics is not None else DummyDynamics()
        self._rel_deg = rel_deg
        self._alphas_static, self._alpha_coefs = self._handle_alphas(alphas, rel_deg)
        self._explicit_barriers = tuple(barriers or [])
        if hocbf_func is not None and self._is_dummy_barrier(hocbf_func):
            hocbf_func = None
        self._explicit_hocbf_func = hocbf_func
        self.cfg = immutabledict(cfg or {})

    @staticmethod
    def _create_dummy_barrier():
        def dummy_barrier(x):
            return jnp.array([])
        return dummy_barrier

    def _ctor_defaults(self) -> dict:
        """Constructor kwargs capturing current field values (per-class)."""
        return {
            'barrier_func': self._barrier_func,
            'dynamics': self._dynamics,
            'rel_deg': self._rel_deg,
            'alphas': self._alphas,
            'barriers': self._explicit_barriers,
            'hocbf_func': self._explicit_hocbf_func,
            'cfg': self.cfg
        }

    def _replace(self, **kwargs):
        """Rebuild instance through its constructor with updated fields."""
        defaults = self._ctor_defaults()
        defaults.update(kwargs)
        return self.__class__(**defaults)

    @staticmethod
    def _is_dummy_barrier(func):
        return getattr(func, '__name__', '') == 'dummy_barrier'

    def _has_real_dynamics(self):
        return not isinstance(self._dynamics, DummyDynamics)

    def _is_ready(self):
        if self._explicit_hocbf_func is not None:
            return True
        return self._has_real_dynamics() and not self._is_dummy_barrier(self._barrier_func)

    # ------------------------------------------------------------- series

    def _alpha_apply(self, i, val):
        if self._alpha_coefs is not None:
            return self._alpha_coefs[i] * val
        if i < len(self._alphas_static):
            return self._alphas_static[i](val)
        return val

    def _series_value(self, i, x):
        """Value of the i-th barrier in the HOCBF series at a single state."""
        if i == 0:
            return self._barrier_func(x)
        prev = lambda y: self._series_value(i - 1, y)
        val, grad = jax.value_and_grad(prev)(x)
        return jnp.dot(grad, self._dynamics.f(x)) + self._alpha_apply(i - 1, val)

    def _make_series_fn(self, i):
        def series_fn(x):
            return self._series_value(i, x)
        return series_fn

    # ------------------------------------------------------------- evaluation

    def barrier(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Compute barrier function value at a single state.

        Args:
            x: State vector (n,)

        Returns:
            Scalar barrier value. Batch with jax.vmap(self.barrier).
        """
        return self._barrier_func(x)

    def hocbf(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Compute highest-order barrier function value at a single state.

        Args:
            x: State vector (n,)

        Returns:
            Scalar HOCBF value. Batch with jax.vmap(self.hocbf).
        """
        if self._explicit_hocbf_func is not None:
            return self._explicit_hocbf_func(x)
        if not self._is_ready():
            raise ValueError("HOCBF not computed. Construct with barrier_func and dynamics.")
        return self._series_value(self._rel_deg - 1, x)

    def get_hocbf_and_lie_derivs(self, x: jnp.ndarray):
        """
        Compute HOCBF and its Lie derivatives at a single state.

        Returns:
            Tuple of (hocbf_value, Lf_hocbf, Lg_hocbf) with shapes
            ((,), (,), (action_dim,))
        """
        hocbf_val, grad_hocbf = jax.value_and_grad(self.hocbf)(x)
        Lf_hocbf = jnp.dot(grad_hocbf, self._dynamics.f(x))
        Lg_hocbf = grad_hocbf @ self._dynamics.g(x)
        return hocbf_val, Lf_hocbf, Lg_hocbf

    def Lf_hocbf(self, x: jnp.ndarray) -> jnp.ndarray:
        """Lie derivative of HOCBF along drift dynamics at a single state, scalar."""
        grad_hocbf = jax.grad(self.hocbf)(x)
        return jnp.dot(grad_hocbf, self._dynamics.f(x))

    def Lg_hocbf(self, x: jnp.ndarray) -> jnp.ndarray:
        """Lie derivative of HOCBF along control dynamics at a single state, shape (action_dim,)."""
        grad_hocbf = jax.grad(self.hocbf)(x)
        return grad_hocbf @ self._dynamics.g(x)

    def compute_barriers_at(self, x: jnp.ndarray) -> List[jnp.ndarray]:
        """Compute all barrier values in the series at a single state."""
        if len(self.barriers) == 0:
            raise ValueError("Barrier series unavailable. Construct with barrier_func and dynamics.")
        return [barrier(x) for barrier in self.barriers_flatten]

    def get_min_barrier_at(self, x: jnp.ndarray) -> jnp.ndarray:
        """Minimum barrier value among all barriers in the series at a single state, scalar."""
        barrier_vals = self.compute_barriers_at(x)
        return jnp.min(jnp.stack([jnp.asarray(v).reshape(()) for v in barrier_vals]))

    def min_barrier(self, x: jnp.ndarray) -> jnp.ndarray:
        """Minimum among base barrier values at a single state, scalar."""
        return jnp.min(jnp.atleast_1d(self.barrier(x)))

    def _make_hocbf_series(self, barrier: Callable, dynamics, rel_deg: int,
                           alphas: tuple) -> List[Callable]:
        """
        Generate an explicit higher-order barrier function series
        [h, Lf(h) + alpha_1(h), ...] as closures. Retained for subclasses
        that compose explicit series.
        """
        barriers = [barrier]

        for i in range(rel_deg - 1):
            current_alpha = alphas[i] if i < len(alphas) else (lambda x: x)
            current_prev_barrier = barriers[i]

            def create_next_hocbf(prev_barrier, alpha, f_dynamics):
                def next_hocbf(x):
                    val, grad_prev = jax.value_and_grad(prev_barrier)(x)
                    return jnp.dot(grad_prev, f_dynamics(x)) + alpha(val)
                return next_hocbf

            barriers.append(create_next_hocbf(current_prev_barrier, current_alpha, dynamics.f))

        return barriers

    def _handle_alphas(self, alphas, rel_deg: int):
        """
        Process alphas into (static_callables, traced_coefs).

        Numeric alphas become a traced coefficient array; callables stay static.
        """
        if rel_deg <= 1:
            return tuple(), None
        if alphas is None:
            return tuple(), None
        assert isinstance(alphas, (list, tuple)) and len(alphas) == rel_deg - 1, \
            "alphas must be a list/tuple with length (rel_deg - 1)"
        if all(isinstance(a, (int, float)) or (hasattr(a, 'ndim') and a.ndim == 0) for a in alphas):
            return tuple(), jnp.array([float(a) for a in alphas])
        assert all(callable(a) for a in alphas), \
            "alphas must be numbers or callable functions"
        return tuple(alphas), None

    # ------------------------------------------------------------- properties

    @property
    def rel_deg(self) -> int:
        return self._rel_deg

    @property
    def barriers(self) -> tuple:
        """Tuple of all barrier functions in the series."""
        return self._barriers

    @property
    def barriers_flatten(self) -> tuple:
        return self.barriers

    @property
    def dynamics(self):
        return self._dynamics

    @property
    def num_barriers(self) -> int:
        return len(self.barriers)

    @property
    def num_constraints(self) -> int:
        return len(self._hocbf_funcs)

    @property
    def _alphas(self) -> tuple:
        if self._alpha_coefs is not None:
            return tuple(float(c) for c in self._alpha_coefs)
        return self._alphas_static

    @property
    def _hocbf_func(self) -> Callable:
        if self._explicit_hocbf_func is not None:
            return self._explicit_hocbf_func
        return self.hocbf

    @property
    def _barrier_funcs(self) -> tuple:
        return (self._barrier_func,)

    @property
    def _hocbf_funcs(self) -> tuple:
        return (self._hocbf_func,)

    @property
    def _barriers(self) -> tuple:
        if self._explicit_barriers:
            return self._explicit_barriers
        if not self._is_ready():
            return tuple()
        return tuple(self._make_series_fn(i) for i in range(self._rel_deg))
