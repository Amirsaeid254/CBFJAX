"""
iLQR (Iterative Linear Quadratic Regulator) Control using trajax.

This module provides iLQR controllers where dynamics and cost functions are
defined in JAX and solved using trajax's iLQR implementation.

Classes:
    iLQRControl: Base iLQR controller with general cost (unconstrained)
    QuadraticiLQRControl: iLQR with quadratic cost (Q, R matrices)
    ConstrainediLQRControl: iLQR with control/state box constraints
    QuadraticConstrainediLQRControl: Constrained iLQR with quadratic cost
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Callable, Optional, Tuple

from trajax.optimizers import ilqr, constrained_ilqr

from .base_control import BaseControl


class iLQRControl(BaseControl):
    """
    Iterative Linear Quadratic Regulator using trajax (unconstrained).

    Takes only cost and dynamics. No control or state bounds.
    Fully JIT-compatible.

    Params:
        params = {
            'horizon': 2.0,              # Prediction horizon [s]
            'time_steps': 0.04,          # Timestep [s]
            'maxiter': 100,              # Max iLQR iterations
            'grad_norm_threshold': 1e-4,
            'make_psd': False,
            'psd_delta': 0.0,
            'alpha_0': 1.0,
            'alpha_min': 0.00005,
        }
    """

    _cost_func: Optional[Callable] = eqx.field(static=True)

    def __init__(
        self,
        action_dim: int,
        params: Optional[dict] = None,
        dynamics=None,
        cost_func: Optional[Callable] = None,
    ):
        default_params = {
            'horizon': 2.0,
            'time_steps': 0.04,
            'maxiter': 100,
            'grad_norm_threshold': 1e-4,
            'relative_grad_norm_threshold': 0.0,
            'obj_step_threshold': 0.0,
            'inputs_step_threshold': 0.0,
            'make_psd': False,
            'psd_delta': 0.0,
            'alpha_0': 1.0,
            'alpha_min': 0.00005,
        }
        if params is not None:
            default_params.update(params)

        super().__init__(action_dim, default_params, dynamics)
        self._cost_func = cost_func

    @classmethod
    def create_empty(cls, action_dim: int, params: Optional[dict] = None) -> 'iLQRControl':
        return cls(action_dim=action_dim, params=params)

    def _create_updated_instance(self, **kwargs):
        defaults = {
            'action_dim': self._action_dim,
            'params': dict(self._params) if self._params else None,
            'dynamics': self._dynamics,
            'cost_func': self._cost_func,
        }
        defaults.update(kwargs)
        return self.__class__(**defaults)

    def assign_dynamics(self, dynamics) -> 'iLQRControl':
        return self._create_updated_instance(dynamics=dynamics)

    def assign_cost_func(self, cost_func: Callable) -> 'iLQRControl':
        """Assign cost function f(x, u, t) -> scalar."""
        return self._create_updated_instance(cost_func=cost_func)

    @property
    def horizon(self) -> float:
        return self._params['horizon']

    @property
    def time_steps(self) -> float:
        return self._params['time_steps']

    @property
    def N_horizon(self) -> int:
        return int(self.horizon / self.time_steps)

    def _get_discrete_dynamics(self):
        """Get discrete dynamics wrapped for trajax (x, u, t) -> x_next."""
        discrete_rhs = self._dynamics.discrete_rhs
        def dynamics(x, u, t):
            return discrete_rhs(x, u)
        return dynamics

    def _get_cost(self) -> Callable:
        """Get cost function. Override in subclass to modify cost."""
        return self._cost_func

    def _solve_ilqr_single(self, x0: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, dict]:
        """Solve iLQR for single initial state."""
        T = self.N_horizon
        U_init = jnp.zeros((T, self._action_dim))

        dynamics = self._get_discrete_dynamics()
        cost = self._get_cost()

        X, U_opt, obj, gradient, adjoints, lqr_val, alpha = ilqr(
            cost=cost,
            dynamics=dynamics,
            x0=x0,
            U=U_init,
            maxiter=self._params['maxiter'],
            grad_norm_threshold=self._params['grad_norm_threshold'],
            relative_grad_norm_threshold=self._params['relative_grad_norm_threshold'],
            obj_step_threshold=self._params['obj_step_threshold'],
            inputs_step_threshold=self._params['inputs_step_threshold'],
            make_psd=self._params['make_psd'],
            psd_delta=self._params['psd_delta'],
            alpha_0=self._params['alpha_0'],
            alpha_min=self._params['alpha_min'],
        )

        info = {
            'objective': obj,
            'gradient': gradient,
            'x_traj': X,
            'u_traj': U_opt,
        }

        return X, U_opt, info

    @jax.jit
    def _optimal_control_single(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, float]:
        """Compute optimal control for single state. Returns (u, objective)."""
        X, U, info = self._solve_ilqr_single(x)
        return U[0], info

    def optimal_control(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, dict]:
        """
        Compute optimal control with vmap for batching.

        Args:
            x: State(s) (state_dim,) or (batch, state_dim)

        Returns:
            Tuple (u, info)
        """
        if x.ndim == 1:
            u, info = self._optimal_control_single(x)
            return u, info
        else:
            u_batch, info_batch = jax.vmap(self._optimal_control_single)(x)
            return u_batch, info_batch

    def get_predicted_trajectory(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Get predicted trajectory (x_traj, u_traj)."""
        X, U, _ = self._solve_ilqr_single(x)
        return X, U

class QuadraticiLQRControl(iLQRControl):
    """
    iLQR with quadratic cost: (x - x_ref)^T Q (x - x_ref) + u^T R u
    """

    _Q: Optional[jnp.ndarray]
    _R: Optional[jnp.ndarray]
    _Q_e: Optional[jnp.ndarray]
    _x_ref: Optional[jnp.ndarray]

    def __init__(
        self,
        action_dim: int,
        params: Optional[dict] = None,
        dynamics=None,
        Q: Optional[jnp.ndarray] = None,
        R: Optional[jnp.ndarray] = None,
        Q_e: Optional[jnp.ndarray] = None,
        x_ref: Optional[jnp.ndarray] = None,
    ):
        super().__init__(action_dim=action_dim, params=params, dynamics=dynamics, cost_func=None)

        self._Q = Q
        self._R = R
        self._Q_e = Q_e
        self._x_ref = x_ref

    def _create_updated_instance(self, **kwargs):
        defaults = {
            'action_dim': self._action_dim,
            'params': dict(self._params) if self._params else None,
            'dynamics': self._dynamics,
            'Q': self._Q,
            'R': self._R,
            'Q_e': self._Q_e,
            'x_ref': self._x_ref,
        }
        defaults.update(kwargs)
        return self.__class__(**defaults)

    def assign_cost_matrices(self, Q: jnp.ndarray, R: jnp.ndarray,
                            Q_e: Optional[jnp.ndarray] = None,
                            x_ref: Optional[jnp.ndarray] = None) -> 'QuadraticiLQRControl':
        if Q_e is None:
            Q_e = Q
        return self._create_updated_instance(Q=Q, R=R, Q_e=Q_e, x_ref=x_ref)

    def assign_reference(self, x_ref: jnp.ndarray) -> 'QuadraticiLQRControl':
        return self._create_updated_instance(x_ref=x_ref)

    def _get_quadratic_cost_func(self) -> Callable:
        """Build quadratic cost function from Q, R matrices."""
        assert self._Q is not None and self._R is not None, "Cost matrices must be assigned"
        Q = self._Q
        R = self._R
        Q_e = self._Q_e if self._Q_e is not None else Q
        T = self.N_horizon
        x_ref = self._x_ref if self._x_ref is not None else jnp.zeros(Q.shape[0])

        def cost(x, u, t):
            x_err = x - x_ref
            return jax.lax.cond(
                t == T,
                lambda: 0.5 * x_err @ Q_e @ x_err,
                lambda: 0.5 * x_err @ Q @ x_err + 0.5 * u @ R @ u
            )

        return cost

    def _get_cost(self) -> Callable:
        """Override to return quadratic cost."""
        return self._get_quadratic_cost_func()


class ConstrainediLQRControl(iLQRControl):
    """
    Constrained iLQR with control bounds and state bounds (box constraints).

    Inherits from iLQRControl and adds constraint handling via trajax's constrained_ilqr.

    Additional params:
        'maxiter_al': 20,
        'constraints_threshold': 1e-4,
        'penalty_init': 1.0,
        'penalty_update_rate': 10.0,
    """

    # Control bounds
    _control_low: tuple = eqx.field(static=True)
    _control_high: tuple = eqx.field(static=True)
    _has_control_bounds: bool = eqx.field(static=True)

    # State bounds
    _state_bounds_idx: tuple = eqx.field(static=True)
    _state_low: tuple = eqx.field(static=True)
    _state_high: tuple = eqx.field(static=True)
    _has_state_bounds: bool = eqx.field(static=True)

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
    ):
        # Add constrained iLQR specific params
        constrained_params = {
            'maxiter_al': 20,
            'constraints_threshold': 1e-4,
            'penalty_init': 1.0,
            'penalty_update_rate': 10.0,
        }
        if params is not None:
            constrained_params.update(params)

        super().__init__(action_dim=action_dim, params=constrained_params, dynamics=dynamics, cost_func=cost_func)

        # Control bounds
        if control_low is not None and control_high is not None:
            self._control_low = tuple(control_low)
            self._control_high = tuple(control_high)
            self._has_control_bounds = True
        else:
            self._control_low = tuple()
            self._control_high = tuple()
            self._has_control_bounds = False

        # State bounds
        if state_bounds_idx is not None and state_low is not None and state_high is not None:
            self._state_bounds_idx = tuple(state_bounds_idx)
            self._state_low = tuple(state_low)
            self._state_high = tuple(state_high)
            self._has_state_bounds = True
        else:
            self._state_bounds_idx = tuple()
            self._state_low = tuple()
            self._state_high = tuple()
            self._has_state_bounds = False

    @classmethod
    def create_empty(cls, action_dim: int, params: Optional[dict] = None) -> 'ConstrainediLQRControl':
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
        }
        defaults.update(kwargs)
        return self.__class__(**defaults)

    def assign_control_bounds(self, low: list, high: list) -> 'ConstrainediLQRControl':
        assert len(low) == len(high) == self._action_dim
        return self._create_updated_instance(control_low=low, control_high=high)

    def assign_state_bounds(self, idx: list, low: list, high: list) -> 'ConstrainediLQRControl':
        assert len(idx) == len(low) == len(high)
        return self._create_updated_instance(state_bounds_idx=idx, state_low=low, state_high=high)

    def _get_inequality_constraint(self) -> Optional[Callable]:
        """Build inequality constraint from control and state bounds."""
        constraints = []

        if self._has_control_bounds:
            u_low = jnp.array(self._control_low)
            u_high = jnp.array(self._control_high)
            constraints.append(('control', u_low, u_high))

        if self._has_state_bounds:
            idx = jnp.array(self._state_bounds_idx)
            x_low = jnp.array(self._state_low)
            x_high = jnp.array(self._state_high)
            constraints.append(('state', idx, x_low, x_high))

        if not constraints:
            return None

        def inequality_constraint(x, u, t):
            parts = []

            for c in constraints:
                if c[0] == 'control':
                    _, u_low, u_high = c
                    parts.append(u - u_high)  # u <= u_high
                    parts.append(u_low - u)   # u >= u_low
                elif c[0] == 'state':
                    _, idx, x_low, x_high = c
                    x_bounded = x[idx]
                    parts.append(x_bounded - x_high)  # x <= x_high
                    parts.append(x_low - x_bounded)   # x >= x_low

            return jnp.concatenate(parts)

        return inequality_constraint

    def _get_cost(self) -> Callable:
        """Get cost function. Override in subclass to modify cost."""
        return self._cost_func
    @jax.jit
    def _solve_ilqr_single(self, x0: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, dict]:
        T = self.N_horizon
        U_init = jnp.zeros((T, self._action_dim))

        dynamics = self._get_discrete_dynamics()
        cost = self._get_cost()
        inequality_constraint = self._get_inequality_constraint()

        (X, U_opt, dual_eq, dual_ineq, penalty, eq_constraints, ineq_constraints,
         max_constraint_violation, obj, gradient, iter_ilqr, iter_al) = constrained_ilqr(
            cost=cost,
            dynamics=dynamics,
            x0=x0,
            U=U_init,
            inequality_constraint=inequality_constraint,
            maxiter_al=self._params['maxiter_al'],
            maxiter_ilqr=self._params['maxiter'],
            grad_norm_threshold=self._params['grad_norm_threshold'],
            relative_grad_norm_threshold=self._params['relative_grad_norm_threshold'],
            obj_step_threshold=self._params['obj_step_threshold'],
            inputs_step_threshold=self._params['inputs_step_threshold'],
            constraints_threshold=self._params['constraints_threshold'],
            penalty_init=self._params['penalty_init'],
            penalty_update_rate=self._params['penalty_update_rate'],
            make_psd=self._params['make_psd'],
            psd_delta=self._params['psd_delta'],
            alpha_0=self._params['alpha_0'],
            alpha_min=self._params['alpha_min'],
        )

        return X, U_opt, {
            'objective': obj,
            'gradient': gradient,
            'max_constraint_violation': max_constraint_violation,
            'x_traj': X,
            'u_traj': U_opt,
        }


class QuadraticConstrainediLQRControl(ConstrainediLQRControl):
    """
    Constrained iLQR with quadratic cost and box constraints.
    """

    _Q: Optional[jnp.ndarray]
    _R: Optional[jnp.ndarray]
    _Q_e: Optional[jnp.ndarray]
    _x_ref: Optional[jnp.ndarray]

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
    ):
        super().__init__(
            action_dim=action_dim,
            params=params,
            dynamics=dynamics,
            cost_func=None,
            control_low=control_low,
            control_high=control_high,
            state_bounds_idx=state_bounds_idx,
            state_low=state_low,
            state_high=state_high,
        )

        self._Q = Q
        self._R = R
        self._Q_e = Q_e
        self._x_ref = x_ref

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
        }
        defaults.update(kwargs)
        return self.__class__(**defaults)

    def assign_cost_matrices(self, Q: jnp.ndarray, R: jnp.ndarray,
                            Q_e: Optional[jnp.ndarray] = None,
                            x_ref: Optional[jnp.ndarray] = None) -> 'QuadraticConstrainediLQRControl':
        if Q_e is None:
            Q_e = Q
        return self._create_updated_instance(Q=Q, R=R, Q_e=Q_e, x_ref=x_ref)

    def assign_reference(self, x_ref: jnp.ndarray) -> 'QuadraticConstrainediLQRControl':
        return self._create_updated_instance(x_ref=x_ref)

    def _get_quadratic_cost_func(self) -> Callable:
        """Build quadratic cost function from Q, R matrices."""
        assert self._Q is not None and self._R is not None, "Cost matrices must be assigned"
        Q = self._Q
        R = self._R
        Q_e = self._Q_e if self._Q_e is not None else Q
        T = self.N_horizon
        x_ref = self._x_ref if self._x_ref is not None else jnp.zeros(Q.shape[0])

        def cost(x, u, t):
            x_err = x - x_ref
            return jax.lax.cond(
                t == T,
                lambda: 0.5 * x_err @ Q_e @ x_err,
                lambda: 0.5 * x_err @ Q @ x_err + 0.5 * u @ R @ u
            )

        return cost

    def _get_cost(self) -> Callable:
        """Override to return quadratic cost."""
        return self._get_quadratic_cost_func()