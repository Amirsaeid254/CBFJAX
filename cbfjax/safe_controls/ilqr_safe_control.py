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
from typing import Callable, Optional
from immutabledict import immutabledict

from ..controls.ilqr_control import ConstrainediLQRControl
from ..controls.base_control import QuadraticCostMixin
from .base_safe_control import BaseSafeControl, DummyBarrier


class iLQRSafeControl(ConstrainediLQRControl, BaseSafeControl):
    """
    iLQR Safe Control with barrier inequality constraints.

    Inherits from (cooperative multiple inheritance):
    - ConstrainediLQRControl: iLQR solving with box constraints
    - BaseSafeControl: barrier interface

    Barrier is enforced purely via augmented Lagrangian inequality constraint.
    Supports optional terminal barrier applied only at t == T.
    """

    @classmethod
    def create_empty(cls, action_dim: int, params: Optional[dict] = None) -> 'iLQRSafeControl':
        return cls(action_dim=action_dim, params=params)

    def _create_updated_instance(self, **kwargs):
        defaults = {
            'action_dim': self._action_dim,
            'params': immutabledict(self._params) if self._params else None,
            'dynamics': self._dynamics,
            'cost_func': self._cost_func,
            'control_low': list(self._control_low) if self._has_control_bounds else None,
            'control_high': list(self._control_high) if self._has_control_bounds else None,
            'state_bounds_idx': list(self._state_bounds_idx) if self._has_state_bounds else None,
            'state_low': list(self._state_low) if self._has_state_bounds else None,
            'state_high': list(self._state_high) if self._has_state_bounds else None,
            'barrier': self._barrier,
            'terminal_barrier': self._terminal_barrier,
        }
        defaults.update(kwargs)
        return self.__class__(**defaults)

    def _get_inequality_constraint(self) -> Callable:
        """
        Build inequality constraint including barrier and terminal barrier constraints.

        Barrier constraint h(x) >= 0 is converted to -h(x) <= 0.
        Terminal barrier is only evaluated at t == T; zeros otherwise.
        """
        base_constraint_func = super()._get_inequality_constraint()

        if not self.has_barrier and not self.has_terminal_barrier:
            return base_constraint_func

        barrier = self._barrier if self.has_barrier else None
        terminal_barrier = self._terminal_barrier if self.has_terminal_barrier else None
        T = self.N_horizon

        def inequality_constraint(x, u, t):
            parts = []

            if base_constraint_func is not None:
                parts.append(base_constraint_func(x, u, t))

            # Path barrier: -h(x) <= 0 at all timesteps
            if barrier is not None:
                h_values = barrier._hocbf_single(x)
                parts.append(jnp.atleast_1d(-h_values).flatten())

            # Terminal barrier: only evaluated at t == T
            if terminal_barrier is not None:
                terminal_constraint = jax.lax.cond(
                    t == T,
                    lambda: jnp.atleast_1d(-terminal_barrier._hocbf_single(x)).flatten(),
                    lambda: jnp.zeros(1),
                )
                parts.append(terminal_constraint)

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
            'params': immutabledict(self._params) if self._params else None,
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
            'terminal_barrier': self._terminal_barrier,
        }
        defaults.update(kwargs)
        return self.__class__(**defaults)

    def _get_cost(self) -> Callable:
        """Get quadratic cost function."""
        return self._get_quadratic_cost_func()
