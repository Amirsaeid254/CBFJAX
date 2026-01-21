"""
iLQR Safe Control with barrier cost support.

This module provides iLQR safe controllers which extend ConstrainediLQRControl
with barrier functions added to the cost for safety.

Unlike NMPC which uses hard/soft constraints via acados, iLQR safe control
adds barrier violations as penalty terms in the cost function.

Classes:
    iLQRSafeControl: Constrained iLQR with barrier penalty (general cost)
    QuadraticiLQRSafeControl: Constrained iLQR with barrier penalty (quadratic cost)
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Callable, Optional, Any

from ..controls.ilqr_control import ConstrainediLQRControl, QuadraticConstrainediLQRControl
from .base_safe_control import DummyBarrier


class iLQRSafeControl(ConstrainediLQRControl):
    """
    iLQR Safe Control with barrier cost penalty.

    Extends ConstrainediLQRControl to include barrier penalty in cost.
    Barrier violations are penalized as: barrier_gain * max(0, -h(x))^2

    Additional params:
        params = {
            ...  # inherited from ConstrainediLQRControl
            'barrier_gain': 1000.0,  # Penalty weight for barrier violations
        }
    """

    _barrier: Any = eqx.field(static=True)

    def __init__(
        self,
        action_dim: int,
        params: Optional[dict] = None,
        dynamics=None,
        cost_func: Optional[Callable] = None,
        control_low: Optional[list] = None,
        control_high: Optional[list] = None,
        state_bounds_idx: Optional[list] = None,
        state_low: Optional[list] = None,
        state_high: Optional[list] = None,
        barrier=None,
    ):
        safe_params = {'barrier_gain': 1000.0}
        if params is not None:
            safe_params.update(params)

        super().__init__(
            action_dim=action_dim,
            params=safe_params,
            dynamics=dynamics,
            cost_func=cost_func,
            control_low=control_low,
            control_high=control_high,
            state_bounds_idx=state_bounds_idx,
            state_low=state_low,
            state_high=state_high,
        )

        self._barrier = barrier if barrier is not None else DummyBarrier()

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

    # ==========================================
    # Barrier Assignment
    # ==========================================

    def assign_state_barrier(self, barrier) -> 'iLQRSafeControl':
        return self._create_updated_instance(barrier=barrier)

    def _is_dummy_barrier(self, barrier) -> bool:
        return isinstance(barrier, DummyBarrier)

    @property
    def barrier(self):
        return self._barrier

    @property
    def has_barrier(self) -> bool:
        return not self._is_dummy_barrier(self._barrier)

    # ==========================================
    # Override cost to include barrier penalty
    # ==========================================

    def _get_cost(self) -> Callable:
        """Get cost function with barrier penalty."""
        original_cost = self._cost_func
        barrier_gain = self._params.get('barrier_gain', 1000.0)

        if not self.has_barrier:
            return original_cost

        barrier = self._barrier

        def cost_with_barrier(x, u, t):
            base_cost = original_cost(x, u, t)
            h_values = barrier.hocbf(x)
            violations = jnp.maximum(0.0, -h_values)
            barrier_penalty = barrier_gain * jnp.sum(violations ** 2)
            return base_cost + barrier_penalty

        return cost_with_barrier

    # ==========================================
    # Barrier Evaluation
    # ==========================================

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


class QuadraticiLQRSafeControl(iLQRSafeControl, QuadraticConstrainediLQRControl):
    """
    Quadratic iLQR Safe Control with barrier cost penalty.

    Uses multiple inheritance:
    - iLQRSafeControl: barrier methods
    - QuadraticConstrainediLQRControl: quadratic cost + constraints

    Only overrides _get_quadratic_cost_func() to add barrier penalty.
    """

    def __init__(
        self,
        action_dim: int,
        params: Optional[dict] = None,
        dynamics=None,
        control_low: Optional[list] = None,
        control_high: Optional[list] = None,
        state_bounds_idx: Optional[list] = None,
        state_low: Optional[list] = None,
        state_high: Optional[list] = None,
        Q: Optional[jnp.ndarray] = None,
        R: Optional[jnp.ndarray] = None,
        Q_e: Optional[jnp.ndarray] = None,
        x_ref: Optional[jnp.ndarray] = None,
        barrier=None,
    ):
        safe_params = {'barrier_gain': 1000.0}
        if params is not None:
            safe_params.update(params)

        # Initialize QuadraticConstrainediLQRControl
        QuadraticConstrainediLQRControl.__init__(
            self,
            action_dim=action_dim,
            params=safe_params,
            dynamics=dynamics,
            control_low=control_low,
            control_high=control_high,
            state_bounds_idx=state_bounds_idx,
            state_low=state_low,
            state_high=state_high,
            Q=Q,
            R=R,
            Q_e=Q_e,
            x_ref=x_ref,
        )

        # Set barrier from iLQRSafeControl
        self._barrier = barrier if barrier is not None else DummyBarrier()

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

    # ==========================================
    # Override cost to include barrier penalty
    # ==========================================

    def _get_cost(self) -> Callable:
        """Get quadratic cost function with barrier penalty."""
        assert self._Q is not None and self._R is not None, "Cost matrices must be assigned"
        Q = self._Q
        R = self._R
        Q_e = self._Q_e if self._Q_e is not None else Q
        T = self.N_horizon
        x_ref = self._x_ref if self._x_ref is not None else jnp.zeros(Q.shape[0])
        barrier_gain = self._params.get('barrier_gain', 1000.0)

        if not self.has_barrier:
            def cost(x, u, t):
                x_err = x - x_ref
                return jax.lax.cond(
                    t == T,
                    lambda: 0.5 * x_err @ Q_e @ x_err,
                    lambda: 0.5 * x_err @ Q @ x_err + 0.5 * u @ R @ u
                )
            return cost

        barrier = self._barrier

        def cost_with_barrier(x, u, t):
            x_err = x - x_ref
            base_cost = jax.lax.cond(
                t == T,
                lambda: 0.5 * x_err @ Q_e @ x_err,
                lambda: 0.5 * x_err @ Q @ x_err + 0.5 * u @ R @ u
            )
            h_values = barrier.hocbf(x)
            violations = jnp.maximum(0.0, -h_values)
            barrier_penalty = barrier_gain * jnp.sum(violations ** 2)
            return base_cost + barrier_penalty

        return cost_with_barrier
