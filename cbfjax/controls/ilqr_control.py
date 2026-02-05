"""
iLQR (Iterative Linear Quadratic Regulator) Control using trajax.

This module provides iLQR controllers where dynamics and cost functions are
defined in JAX and solved using trajax's iLQR implementation.

Uses cooperative multiple inheritance pattern where all classes:
- Accept **kwargs and pass them up via super().__init__(**kwargs)
- Extract only the parameters they need

All controllers follow the stateful interface:
- _optimal_control_single(x, state) -> (u, new_state)
- get_init_state() -> ILQRState or ConstrainedILQRState
- State contains warm-start U trajectory

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
from immutabledict import immutabledict

from trajax.optimizers import ilqr, constrained_ilqr

from .base_control import BaseControl, QuadraticCostMixin
from .control_types import ILQRState, ConstrainedILQRState, ILQRInfo, ConstrainedILQRInfo


class iLQRControl(BaseControl):
    """
    Iterative Linear Quadratic Regulator using trajax (unconstrained).

    Takes only cost and dynamics. No control or state bounds.
    Fully JIT-compatible. Uses cooperative multiple inheritance.

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

    def __init__(self, cost_func: Optional[Callable] = None, **kwargs):
        """
        Initialize iLQRControl.

        Args:
            cost_func: Cost function f(x, u, t) -> scalar
            **kwargs: Passed to next class in MRO (includes action_dim, params, dynamics)
        """
        # Set default iLQR params
        params = kwargs.get('params', None)
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
        kwargs['params'] = immutabledict(default_params)

        super().__init__(**kwargs)
        self._cost_func = cost_func

    @classmethod
    def create_empty(cls, action_dim: int, params: Optional[dict] = None) -> 'iLQRControl':
        return cls(action_dim=action_dim, params=params)

    def _create_updated_instance(self, **kwargs):
        defaults = {
            'action_dim': self._action_dim,
            'params': immutabledict(self._params) if self._params else None,
            'dynamics': self._dynamics,
            'cost_func': self._cost_func,
        }
        defaults.update(kwargs)
        return self.__class__(**defaults)

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

    def get_init_state(self):
        """Get initial controller state with zero U trajectory."""
        return ILQRState(U=jnp.zeros((self.N_horizon, self._action_dim)))

    def _get_default_guess(self) -> jnp.ndarray:
        """Get default U guess (zeros)."""
        return jnp.zeros((self.N_horizon, self._action_dim))

    def set_init_guess(self, U: jnp.ndarray = None, state=None) -> ILQRState:
        """
        Set initial guess U on a controller state.

        If state is None, creates one from get_init_state() first.
        If U is None, uses default zeros.

        Args:
            U: Control trajectory guess (N_horizon, action_dim). If None, uses default.
            state: Existing controller state to update. If None, uses default.

        Returns:
            ILQRState with the provided (or default) U trajectory
        """
        if state is None:
            state = self.get_init_state()
        if U is None:
            U = self._get_default_guess()
        return state._replace(U=jnp.asarray(U))

    def _get_discrete_dynamics(self):
        """Get discrete dynamics wrapped for trajax (x, u, t) -> x_next."""
        discrete_rhs = self._dynamics.discrete_rhs
        def dynamics(x, u, t):
            return discrete_rhs(x, u)
        return dynamics

    def _get_cost(self) -> Callable:
        """Get cost function. Override in subclass to modify cost."""
        return self._cost_func

    def _optimal_control_single(self, x: jnp.ndarray, state=None) -> Tuple[jnp.ndarray, ILQRState]:
        """Compute optimal control for single state. Returns (u, new_state)."""
        if state is None:
            state = self.get_init_state()

        dynamics = self._get_discrete_dynamics()
        cost = self._get_cost()

        X, U, obj, gradient, adjoints, lqr_val, alpha = ilqr(
            cost=cost,
            dynamics=dynamics,
            x0=x,
            U=state.U,
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

        return U[0], ILQRState(U=U)

    def _optimal_control_single_with_info(self, x: jnp.ndarray, state=None) -> Tuple[jnp.ndarray, ILQRState, ILQRInfo]:
        """Compute optimal control with diagnostic info. Returns (u, new_state, info)."""
        if state is None:
            state = self.get_init_state()

        dynamics = self._get_discrete_dynamics()
        cost = self._get_cost()

        X, U, obj, gradient, adjoints, lqr_val, alpha = ilqr(
            cost=cost,
            dynamics=dynamics,
            x0=x,
            U=state.U,
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

        info = ILQRInfo(objective=obj, gradient=gradient, x_traj=X, u_traj=U)
        return U[0], ILQRState(U=U), info

    @jax.jit
    def optimal_control(self, x: jnp.ndarray, state=None) -> Tuple[jnp.ndarray, ILQRState]:
        """
        Compute optimal control with vmap for batching.

        Args:
            x: State(s) (state_dim,) or (batch, state_dim)
            state: Controller state (optional, uses get_init_state() if None)

        Returns:
            Tuple (u, new_state)
        """
        if state is None:
            state = self.get_init_state()
        return jax.vmap(self._optimal_control_single, in_axes=(0, None))(x, state)

    @jax.jit
    def optimal_control_with_info(self, x: jnp.ndarray, state=None) -> tuple:
        """Compute optimal control with diagnostic info."""
        if state is None:
            state = self.get_init_state()
        return jax.vmap(self._optimal_control_single_with_info, in_axes=(0, None))(x, state)

    @jax.jit
    def get_predicted_trajectory(self, x: jnp.ndarray, state=None) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Get predicted trajectory (x_traj, u_traj)."""
        if state is None:
            state = self.get_init_state()
        _, _, info = self._optimal_control_single_with_info(x, state)
        return info.x_traj, info.u_traj

class QuadraticiLQRControl(QuadraticCostMixin, iLQRControl):
    """
    iLQR with quadratic cost: (x - x_ref)^T Q (x - x_ref) + u^T R u

    Uses cooperative multiple inheritance. QuadraticCostMixin provides Q, R fields.
    """

    # Cost matrices as Callable for JIT compatibility (static fields)
    _Q: Optional[Callable] = eqx.field(static=True)
    _R: Optional[Callable] = eqx.field(static=True)
    _Q_e: Optional[Callable] = eqx.field(static=True)
    _x_ref: Optional[Callable] = eqx.field(static=True)

    def __init__(self, **kwargs):
        """
        Initialize QuadraticiLQRControl.

        Args:
            **kwargs: All args passed via cooperative inheritance
                - Q, R, Q_e, x_ref: Handled by QuadraticCostMixin
                - action_dim, params, dynamics: Handled by iLQRControl -> BaseControl
        """
        # QuadraticCostMixin.__init__ extracts Q, R, Q_e, x_ref and passes rest to iLQRControl
        super().__init__(cost_func=None, **kwargs)

    def _create_updated_instance(self, **kwargs):
        defaults = {
            'action_dim': self._action_dim,
            'params': immutabledict(self._params) if self._params else None,
            'dynamics': self._dynamics,
            'Q': self._Q,
            'R': self._R,
            'Q_e': self._Q_e,
            'x_ref': self._x_ref,
        }
        defaults.update(kwargs)
        return self.__class__(**defaults)

    def _get_cost(self) -> Callable:
        """Override to return quadratic cost."""
        return self._get_quadratic_cost_func()


class ConstrainediLQRControl(iLQRControl):
    """
    Constrained iLQR with control bounds and state bounds (box constraints).

    Uses cooperative multiple inheritance and trajax's constrained_ilqr.

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
        control_low: Optional[list] = None,
        control_high: Optional[list] = None,
        state_bounds_idx: Optional[list] = None,
        state_low: Optional[list] = None,
        state_high: Optional[list] = None,
        **kwargs
    ):
        """
        Initialize ConstrainediLQRControl.

        Args:
            control_low: Lower bounds for control inputs
            control_high: Upper bounds for control inputs
            state_bounds_idx: Indices of bounded states
            state_low: Lower bounds for bounded states
            state_high: Upper bounds for bounded states
            **kwargs: Passed to next class in MRO (includes action_dim, params, dynamics, cost_func)
        """
        # Add constrained iLQR specific params
        params = kwargs.get('params', None)
        constrained_params = {
            'maxiter_al': 20,
            'constraints_threshold': 1e-4,
            'penalty_init': 1.0,
            'penalty_update_rate': 10.0,
        }
        if params is not None:
            constrained_params.update(params)
        kwargs['params'] = immutabledict(constrained_params)

        super().__init__(**kwargs)

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
            'params': immutabledict(self._params) if self._params else None,
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

    def get_init_state(self):
        """Get initial controller state with zero U trajectory."""
        return ConstrainedILQRState(U=jnp.zeros((self.N_horizon, self._action_dim)))

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
    def _optimal_control_single(self, x: jnp.ndarray, state=None) -> Tuple[jnp.ndarray, ConstrainedILQRState]:
        """Compute optimal control for single state. Returns (u, new_state)."""
        if state is None:
            state = self.get_init_state()

        dynamics = self._get_discrete_dynamics()
        cost = self._get_cost()
        inequality_constraint = self._get_inequality_constraint()

        (X, U, dual_eq, dual_ineq, penalty, eq_constraints, ineq_constraints,
         max_constraint_violation, obj, gradient, iter_ilqr, iter_al) = constrained_ilqr(
            cost=cost,
            dynamics=dynamics,
            x0=x,
            U=state.U,
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

        return U[0], ConstrainedILQRState(U=U)

    @jax.jit
    def _optimal_control_single_with_info(self, x: jnp.ndarray, state=None) -> Tuple[jnp.ndarray, ConstrainedILQRState, ConstrainedILQRInfo]:
        """Compute optimal control with diagnostic info. Returns (u, new_state, info)."""
        if state is None:
            state = self.get_init_state()

        dynamics = self._get_discrete_dynamics()
        cost = self._get_cost()
        inequality_constraint = self._get_inequality_constraint()

        (X, U, dual_eq, dual_ineq, penalty, eq_constraints, ineq_constraints,
         max_constraint_violation, obj, gradient, iter_ilqr, iter_al) = constrained_ilqr(
            cost=cost,
            dynamics=dynamics,
            x0=x,
            U=state.U,
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

        info = ConstrainedILQRInfo(
            objective=obj,
            gradient=gradient,
            max_constraint_violation=max_constraint_violation,
            x_traj=X,
            u_traj=U,
        )
        return U[0], ConstrainedILQRState(U=U), info

    @jax.jit
    def optimal_control(self, x: jnp.ndarray, state=None) -> tuple:
        """Compute optimal control with vmap for batching."""
        if state is None:
            state = self.get_init_state()

        return jax.vmap(self._optimal_control_single, in_axes=(0, None))(x, state)

    @jax.jit
    def optimal_control_with_info(self, x: jnp.ndarray, state=None) -> tuple:
        """Compute optimal control with diagnostic info."""
        if state is None:
            state = self.get_init_state()

        return jax.vmap(self._optimal_control_single_with_info, in_axes=(0, None))(x, state)


class QuadraticConstrainediLQRControl(QuadraticCostMixin, ConstrainediLQRControl):
    """
    Constrained iLQR with quadratic cost and box constraints.

    Uses cooperative multiple inheritance. QuadraticCostMixin provides Q, R fields.
    """

    # Cost matrices as Callable for JIT compatibility (static fields)
    _Q: Optional[Callable] = eqx.field(static=True)
    _R: Optional[Callable] = eqx.field(static=True)
    _Q_e: Optional[Callable] = eqx.field(static=True)
    _x_ref: Optional[Callable] = eqx.field(static=True)

    def __init__(self, **kwargs):
        """
        Initialize QuadraticConstrainediLQRControl.

        Args:
            **kwargs: All args passed via cooperative inheritance
                - Q, R, Q_e, x_ref: Handled by QuadraticCostMixin
                - control_low, control_high, etc.: Handled by ConstrainediLQRControl
                - action_dim, params, dynamics: Handled by BaseControl
        """
        super().__init__(cost_func=None, **kwargs)

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
        }
        defaults.update(kwargs)
        return self.__class__(**defaults)

    def _get_cost(self) -> Callable:
        """Override to return quadratic cost."""
        return self._get_quadratic_cost_func()
