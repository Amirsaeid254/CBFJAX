"""
iLQR Safe Control with barrier constraint support.

Uses cooperative multiple inheritance pattern where all classes:
- Accept **kwargs and pass them up via super().__init__(**kwargs)
- Extract only the parameters they need

Classes:
    iLQRSafeControl: Constrained iLQR with barrier constraints (general cost)
    QuadraticiLQRSafeControl: Constrained iLQR with barrier constraints (quadratic cost)
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Callable, Optional, Any

from ..controls.ilqr_control import ConstrainediLQRControl
from ..controls.base_control import QuadraticCostMixin
from .base_safe_control import BaseSafeControl, DummyBarrier


class iLQRSafeControl(ConstrainediLQRControl, BaseSafeControl):
    """
    iLQR Safe Control with barrier inequality constraints.

    Inherits from (cooperative multiple inheritance):
    - ConstrainediLQRControl: iLQR solving with box constraints
    - BaseSafeControl: barrier interface

    Additional params:
        'log_barrier_gain': 0.0,  # Gain for optional log barrier cost penalty
        'safety_margin': 0.0,     # Safety margin for barrier constraint
    """

    def __init__(self, **kwargs):
        """
        Initialize iLQRSafeControl.

        Args:
            **kwargs: All args passed via cooperative inheritance
                - barrier: Handled by BaseSafeControl
                - control_low, control_high, etc.: Handled by ConstrainediLQRControl
                - cost_func: Handled by iLQRControl
                - action_dim, params, dynamics: Handled by BaseControl
        """
        # Add safe control specific params
        params = kwargs.get('params', None)
        safe_params = {
            'log_barrier_gain': 0.0,
            'safety_margin': 0.0,
        }
        if params is not None:
            safe_params.update(params)
        kwargs['params'] = safe_params

        super().__init__(**kwargs)

    @classmethod
    def create_empty(cls, action_dim: int, params: Optional[dict] = None) -> 'iLQRSafeControl':
        return cls(action_dim=action_dim, params=params)

    def _create_updated_instance(self, **kwargs):
        defaults = {
            'action_dim': self._action_dim,
            'params': dict(self._params) if self._params else None,
            'dynamics': self._dynamics,
            'cost_func': self._cost_func,
            'control_low': list(self._control_low) if self._has_control_bounds else None,
            'control_high': list(self._control_high) if self._has_control_bounds else None,
            'state_bounds_idx': list(self._state_bounds_idx) if self._has_state_bounds else None,
            'state_low': list(self._state_low) if self._has_state_bounds else None,
            'state_high': list(self._state_high) if self._has_state_bounds else None,
            'barrier': self._barrier,
        }
        defaults.update(kwargs)
        return self.__class__(**defaults)

    # Barrier interface inherited from BaseSafeControl via cooperative inheritance

    @property
    def log_barrier_gain(self) -> float:
        return self._params.get('log_barrier_gain', 0.0)

    @property
    def safety_margin(self) -> float:
        return self._params.get('safety_margin', 0.0)

    def _get_cost(self) -> Callable:
        """
        Get cost function with optional log barrier penalty.

        If log_barrier_gain > 0 and barrier is assigned, adds:
        -log_barrier_gain * sum(log(h(x))) to the cost for smooth repulsion.
        """
        base_cost_func = super()._get_cost()

        if self.log_barrier_gain <= 0.0 or not self.has_barrier:
            return base_cost_func

        barrier = self._barrier
        gain = self.log_barrier_gain

        def cost_with_log_barrier(x, u, t):
            base_cost = base_cost_func(x, u, t)
            h_values = barrier._hocbf_single(x)
            h_safe = jnp.maximum(h_values, 1e-8)
            log_barrier_penalty = -gain * jnp.sum(jnp.log(h_safe))
            return base_cost + log_barrier_penalty

        return cost_with_log_barrier

    def _get_inequality_constraint(self) -> Callable:
        """
        Build inequality constraint including barrier constraints.

        Barrier constraint h(x) >= 0 is converted to -h(x) <= 0.
        """
        base_constraint_func = super()._get_inequality_constraint()

        if not self.has_barrier:
            return base_constraint_func

        barrier = self._barrier
        margin = self.safety_margin

        def inequality_constraint(x, u, t):
            parts = []
            if base_constraint_func is not None:
                parts.append(base_constraint_func(x, u, t))
            h_values = barrier._hocbf_single(x)
            parts.append(jnp.atleast_1d(-(h_values - margin)).flatten())
            return jnp.concatenate(parts)

        return inequality_constraint

    def get_barrier_along_trajectory(self, x: jnp.ndarray) -> jnp.ndarray:
        """Return min barrier values along predicted trajectory."""
        assert self.has_barrier, "No barrier assigned"
        X, _ = self.get_predicted_trajectory(x)
        return jnp.min(self._barrier.hocbf(X), axis=0)

    def get_barrier_values_full(self, x: jnp.ndarray) -> jnp.ndarray:
        """Return all barrier values along predicted trajectory."""
        assert self.has_barrier, "No barrier assigned"
        X, _ = self.get_predicted_trajectory(x)
        return self._barrier.hocbf(X)


class QuadraticiLQRSafeControl(QuadraticCostMixin, iLQRSafeControl):
    """
    Quadratic iLQR Safe Control with barrier inequality constraints.

    Uses cooperative multiple inheritance:
    - QuadraticCostMixin: quadratic cost (Q, R matrices)
    - iLQRSafeControl: iLQR + constraints + barrier
    """

    # Cost matrices as Callable for JIT compatibility
    _Q: Optional[Callable] = eqx.field(static=True)
    _R: Optional[Callable] = eqx.field(static=True)
    _Q_e: Optional[Callable] = eqx.field(static=True)
    _x_ref: Optional[Callable] = eqx.field(static=True)

    def __init__(self, **kwargs):
        """
        Initialize QuadraticiLQRSafeControl.

        Args:
            **kwargs: All args passed via cooperative inheritance
                - Q, R, Q_e, x_ref: Handled by QuadraticCostMixin
                - barrier: Handled by BaseSafeControl
                - control_low, control_high, etc.: Handled by ConstrainediLQRControl
                - action_dim, params, dynamics: Handled by BaseControl
        """
        super().__init__(cost_func=None, **kwargs)

    @classmethod
    def create_empty(cls, action_dim: int, params: Optional[dict] = None) -> 'QuadraticiLQRSafeControl':
        return cls(action_dim=action_dim, params=params)

    def _create_updated_instance(self, **kwargs):
        defaults = {
            'action_dim': self._action_dim,
            'params': dict(self._params) if self._params else None,
            'dynamics': self._dynamics,
            'control_low': list(self._control_low) if self._has_control_bounds else None,
            'control_high': list(self._control_high) if self._has_control_bounds else None,
            'state_bounds_idx': list(self._state_bounds_idx) if self._has_state_bounds else None,
            'state_low': list(self._state_low) if self._has_state_bounds else None,
            'state_high': list(self._state_high) if self._has_state_bounds else None,
            'Q': self._Q,
            'R': self._R,
            'Q_e': self._Q_e,
            'x_ref': self._x_ref,
            'barrier': self._barrier,
        }
        defaults.update(kwargs)
        return self.__class__(**defaults)

    def _get_cost(self) -> Callable:
        """
        Get quadratic cost function with optional log barrier penalty.
        """
        base_cost_func = self._get_quadratic_cost_func()

        if self.log_barrier_gain <= 0.0 or not self.has_barrier:
            return base_cost_func

        barrier = self._barrier
        gain = self.log_barrier_gain

        def cost_with_log_barrier(x, u, t):
            base_cost = base_cost_func(x, u, t)
            h_values = barrier._hocbf_single(x)
            h_safe = jnp.maximum(h_values, 1e-8)
            log_barrier_penalty = -gain * jnp.sum(jnp.log(h_safe))
            return base_cost + log_barrier_penalty

        return cost_with_log_barrier
