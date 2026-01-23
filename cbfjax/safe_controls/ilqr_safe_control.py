"""
iLQR Safe Control with barrier constraint support.

This module provides iLQR safe controllers which extend ConstrainediLQRControl
with barrier functions as inequality constraints plus optional log barrier penalty.

Two mechanisms for barrier handling:
1. AL Inequality Constraint: h(x) >= 0 converted to -h(x) <= 0, handled by
   Augmented Lagrangian method in trajax's constrained_ilqr (hard constraint)
2. Log Barrier Penalty (optional): -log_barrier_gain * sum(log(h(x))) added to cost
   for smooth gradient away from constraint boundary (soft repulsion)

Classes:
    iLQRSafeControl: Constrained iLQR with barrier constraints (general cost)
    QuadraticiLQRSafeControl: Constrained iLQR with barrier constraints (quadratic cost)
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Callable, Optional, Any

from ..controls.ilqr_control import ConstrainediLQRControl, QuadraticConstrainediLQRControl
from .base_safe_control import DummyBarrier


class iLQRSafeControl(ConstrainediLQRControl):
    """
    iLQR Safe Control with barrier inequality constraints.

    Extends ConstrainediLQRControl to include barrier as inequality constraints.
    Two mechanisms for barrier handling:

    1. AL Inequality Constraint: h(x) >= 0 converted to -h(x) <= 0, handled by
       Augmented Lagrangian method in trajax's constrained_ilqr (hard constraint)
    2. Log Barrier Penalty (optional): -log_barrier_gain * sum(log(h(x))) added to cost
       for smooth gradient away from constraint boundary (soft repulsion)

    The AL method properly handles these constraints with:
    - Dual variables (Lagrange multipliers)
    - Penalty updates
    - Proper convergence guarantees

    The optional log barrier provides additional smooth gradient information
    to help the optimizer stay away from constraint boundaries.
    """

    _barrier: Any = eqx.field(static=True)
    _log_barrier_gain: float = eqx.field(static=True)
    _safety_margin: float = eqx.field(static=True)

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
        log_barrier_gain: float = 0.0,
        safety_margin: float = 0.0,
    ):
        super().__init__(
            action_dim=action_dim,
            params=params,
            dynamics=dynamics,
            cost_func=cost_func,
            control_low=control_low,
            control_high=control_high,
            state_bounds_idx=state_bounds_idx,
            state_low=state_low,
            state_high=state_high,
        )

        self._barrier = barrier if barrier is not None else DummyBarrier()
        self._log_barrier_gain = log_barrier_gain
        self._safety_margin = safety_margin

    @classmethod
    def create_empty(cls, action_dim: int, params: Optional[dict] = None) -> 'iLQRSafeControl':
        log_barrier_gain = params.get('log_barrier_gain', 0.0) if params else 0.0
        safety_margin = params.get('safety_margin', 0.0) if params else 0.0
        return cls(action_dim=action_dim, params=params, log_barrier_gain=log_barrier_gain, safety_margin=safety_margin)

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
            'log_barrier_gain': self._log_barrier_gain,
            'safety_margin': self._safety_margin,
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

    @property
    def log_barrier_gain(self) -> float:
        return self._log_barrier_gain

    @property
    def safety_margin(self) -> float:
        return self._safety_margin

    # ==========================================
    # Override cost to include optional log barrier penalty
    # ==========================================

    def _get_cost(self) -> Callable:
        """
        Get cost function with optional log barrier penalty.

        If log_barrier_gain > 0 and barrier is assigned, adds:
        -log_barrier_gain * sum(log(h(x))) to the cost for smooth repulsion.
        """
        base_cost_func = super()._get_cost()

        if self._log_barrier_gain <= 0.0 or not self.has_barrier:
            return base_cost_func

        barrier = self._barrier
        gain = self._log_barrier_gain

        def cost_with_log_barrier(x, u, t):
            base_cost = base_cost_func(x, u, t)

            # Log barrier penalty: -gain * sum(log(h(x)))
            h_values = barrier._hocbf_single(x)
            h_safe = jnp.maximum(h_values, 1e-8)  # Numerical stability
            log_barrier_penalty = -gain * jnp.sum(jnp.log(h_safe))

            return base_cost + log_barrier_penalty

        return cost_with_log_barrier

    # ==========================================
    # Override inequality constraint to include barrier
    # ==========================================

    def _get_inequality_constraint(self) -> Callable:
        """
        Build inequality constraint including barrier constraints.

        Barrier constraint h(x) >= 0 is converted to -h(x) <= 0.
        Combined with control/state box constraints from parent class.

        Returns:
            Function g(x, u, t) where g <= 0 represents all constraints
        """
        # Get base constraints from parent (control bounds, state bounds)
        base_constraint_func = super()._get_inequality_constraint()

        if not self.has_barrier:
            return base_constraint_func

        barrier = self._barrier
        margin = self._safety_margin

        def inequality_constraint(x, u, t):
            parts = []

            # Add base constraints if they exist
            if base_constraint_func is not None:
                parts.append(base_constraint_func(x, u, t))

            # Add barrier constraint: h(x) >= margin  =>  -(h(x) - margin) <= 0
            h_values = barrier._hocbf_single(x)
            parts.append(jnp.atleast_1d(-(h_values - margin)).flatten())

            return jnp.concatenate(parts)

        return inequality_constraint

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
    Quadratic iLQR Safe Control with barrier inequality constraints.

    Uses multiple inheritance:
    - iLQRSafeControl: barrier constraint methods (_get_inequality_constraint)
    - QuadraticConstrainediLQRControl: quadratic cost + box constraints

    Two mechanisms for barrier handling:
    1. AL Inequality Constraint: h(x) >= 0 converted to -h(x) <= 0, handled by
       Augmented Lagrangian method in trajax's constrained_ilqr (hard constraint)
    2. Log Barrier Penalty (optional): -log_barrier_gain * sum(log(h(x))) added to cost
       for smooth gradient away from constraint boundary (soft repulsion)
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
        log_barrier_gain: float = 0.0,
        safety_margin: float = 0.0,
    ):
        # Initialize QuadraticConstrainediLQRControl
        QuadraticConstrainediLQRControl.__init__(
            self,
            action_dim=action_dim,
            params=params,
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
        self._log_barrier_gain = log_barrier_gain
        self._safety_margin = safety_margin

    @classmethod
    def create_empty(cls, action_dim: int, params: Optional[dict] = None) -> 'QuadraticiLQRSafeControl':
        log_barrier_gain = params.get('log_barrier_gain', 0.0) if params else 0.0
        safety_margin = params.get('safety_margin', 0.0) if params else 0.0
        return cls(action_dim=action_dim, params=params, log_barrier_gain=log_barrier_gain, safety_margin=safety_margin)

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
            'log_barrier_gain': self._log_barrier_gain,
            'safety_margin': self._safety_margin,
        }
        defaults.update(kwargs)
        return self.__class__(**defaults)

    # ==========================================
    # Cost function (quadratic + optional log barrier)
    # ==========================================

    def _get_cost(self) -> Callable:
        """
        Get quadratic cost function with optional log barrier penalty.

        Barrier constraints are handled via AL inequality constraints.
        If log_barrier_gain > 0, adds -gain * sum(log(h(x))) for smooth repulsion.
        """
        base_cost_func = self._get_quadratic_cost_func()

        if self._log_barrier_gain <= 0.0 or not self.has_barrier:
            return base_cost_func

        barrier = self._barrier
        gain = self._log_barrier_gain

        def cost_with_log_barrier(x, u, t):
            base_cost = base_cost_func(x, u, t)

            # Log barrier penalty: -gain * sum(log(h(x)))
            h_values = barrier._hocbf_single(x)
            h_safe = jnp.maximum(h_values, 1e-8)  # Numerical stability
            log_barrier_penalty = -gain * jnp.sum(jnp.log(h_safe))

            return base_cost + log_barrier_penalty

        return cost_with_log_barrier
