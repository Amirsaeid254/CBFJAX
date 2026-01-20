"""
QP-based Safe Control classes with JAX JIT compatibility.

This module implements QP-based safe control algorithms using qpax for solving
quadratic programs.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Callable, Optional, Any, Dict
from immutabledict import immutabledict
from functools import partial

from qpax import solve_qp_primal

from .base_safe_control import BaseCBFSafeControl, BaseMinIntervSafeControl
from ..utils.utils import ensure_batch_dim


class QPSafeControl(BaseCBFSafeControl):
    """
    QP-based Safe Control with full JAX JIT compatibility.

    Uses quadratic programming to solve for safe control inputs that
    minimize a cost function while satisfying barrier constraints.

    Attributes:
        _slacked: Whether to use slack variables
        _slack_gain: Gain for slack variables in objective
    """

    # Static parameters for JIT compatibility
    _slacked: bool = eqx.field(static=True)
    _slack_gain: float = eqx.field(static=True)

    def __init__(self, action_dim: int, alpha: Optional[Callable] = None,
                 params: Optional[dict] = None, dynamics=None, barrier=None,
                 Q=None, c=None, slacked: bool = False, slack_gain: float = 100.0):
        """
        Initialize QPSafeControl.

        Args:
            action_dim: Control input dimension
            alpha: Class-K function for barrier constraint
            params: Legacy parameter dictionary
            dynamics: System dynamics object
            barrier: Barrier function object
            Q: Cost matrix function
            c: Cost vector function
            slacked: Whether to use slack variables
            slack_gain: Gain for slack variables
        """
        # Handle legacy params dict
        if params is not None:
            slacked = params.get('slacked', slacked)
            slack_gain = params.get('slack_gain', slack_gain)

        # Initialize base class
        super().__init__(action_dim, alpha, immutabledict({'slacked': slacked, 'slack_gain': slack_gain}),
                        dynamics=dynamics, barrier=barrier, Q=Q, c=c)

        # Set static parameters
        self._slacked = slacked
        self._slack_gain = slack_gain

    @classmethod
    def create_empty(cls, action_dim: int, alpha: Optional[Callable] = None,
                     params: Optional[dict] = None) -> 'QPSafeControl':
        """
        Create empty QPSafeControl instance for assignment chain.

        Args:
            action_dim: Control input dimension
            alpha: Class-K function for barrier constraint
            params: Optional parameter dictionary

        Returns:
            Empty QPSafeControl instance ready for assignment
        """
        return cls(action_dim=action_dim, alpha=alpha, params=params)

    def _create_updated_instance(self, **kwargs):
        """
        Create new instance with updated fields.

        Args:
            **kwargs: Fields to update

        Returns:
            New QPSafeControl instance with updated fields
        """
        defaults = {
            'action_dim': self._action_dim,
            'alpha': self._alpha,
            'params': dict(self._params),
            'dynamics': self._dynamics,
            'barrier': self._barrier,
            'Q': self._Q,
            'c': self._c,
            'slacked': self._slacked,
            'slack_gain': self._slack_gain
        }
        defaults.update(kwargs)
        return self.__class__(**defaults)

    def assign_state_barrier(self, barrier) -> 'QPSafeControl':
        """
        Assign state barrier to controller.

        Args:
            barrier: Barrier function object

        Returns:
            New QPSafeControl instance with assigned barrier
        """
        return self._create_updated_instance(barrier=barrier)

    def assign_dynamics(self, dynamics) -> 'QPSafeControl':
        """
        Assign dynamics to controller.

        Args:
            dynamics: System dynamics object

        Returns:
            New QPSafeControl instance with assigned dynamics
        """
        return self._create_updated_instance(dynamics=dynamics)

    def assign_cost(self, Q: Callable, c: Callable) -> 'QPSafeControl':
        """
        Assign quadratic cost function.

        Args:
            Q: Function that computes cost matrix from state
            c: Function that computes cost vector from state

        Returns:
            New QPSafeControl instance with assigned cost
        """
        return self._create_updated_instance(Q=Q, c=c)

    @jax.jit
    def _optimal_control_single(self, x: jnp.ndarray) -> tuple:
        """
        Compute safe optimal control for a single state using QP.

        Args:
            x: Single state vector (state_dim,)

        Returns:
            Tuple (u, info) where info is a dict with slack_vars and constraint_at_u
        """
        if self._slacked:
            return self._optimal_control_single_slacked(x)

        # Make objective for single state
        Q_matrix, c_vector = self._make_objective_single(x)

        # Make inequality constraints for single state
        G, h = self._make_ineq_const_single(x)

        # Make equality constraints (empty by default)
        A, b = self._make_eq_const_single(x, Q_matrix)

        # Solve QP
        # qpax expects: min 0.5 x^T Q x + c^T x s.t. Gx <= h, Ax = b
        u = solve_qp_primal(Q_matrix, c_vector, A, b, G, h)

        # Compute constraint at u for info
        constraint_at_u = jnp.dot(G, u) - h
        slack_vars = jnp.zeros(1)

        info = {'slack_vars': slack_vars, 'constraint_at_u': constraint_at_u}
        return u, info


    def _optimal_control_single_slacked(self, x: jnp.ndarray) -> tuple:
        """
        Compute safe optimal control with slack variables for single state.

        Args:
            x: Single state vector (state_dim,)

        Returns:
            Tuple (u, info) where info is a dict with slack_vars and constraint_at_u
        """
        # Make inequality constraints for slacked version
        G, h = self._make_ineq_const_slacked_single(x)
        num_constraints = h.shape[0]

        # Make objective for slacked version
        Q_matrix, c_vector = self._make_objective_slacked_single(x, num_constraints)

        # Make equality constraints
        A, b = self._make_eq_const_single(x, Q_matrix)

        # Solve QP for augmented decision variable [u, slack]
        res = solve_qp_primal(Q_matrix, c_vector, A, b, G, h)

        # Extract control and slack
        u = res[:self._action_dim]
        slack_vars = res[self._action_dim:]

        # Compute constraint at solution
        constraint_at_u = jnp.dot(G, res) - h

        info = {'slack_vars': slack_vars, 'constraint_at_u': constraint_at_u}
        return u, info

    def _make_objective_single(self, x: jnp.ndarray) -> tuple:
        """
        Create objective matrices for single state.

        Args:
            x: Single state vector (state_dim,)

        Returns:
            Tuple (Q, c) for quadratic objective
        """
        Q_matrix = self._Q(x)  # (action_dim, action_dim)
        c_vector = self._c(x)  # (action_dim,)

        return Q_matrix, c_vector

    def _make_objective_slacked_single(self, x: jnp.ndarray, num_constraints: int) -> tuple:
        """
        Create objective matrices with slack variables for single state.

        Args:
            x: Single state vector (state_dim,)
            num_constraints: Number of constraints (slack variables)

        Returns:
            Tuple (Q, c) for augmented quadratic objective
        """
        Q_base, c_base = self._make_objective_single(x)

        # Create block diagonal Q matrix
        Q_slack = self._slack_gain * 0.5 * jnp.eye(num_constraints)
        Q_matrix = jnp.block([
            [Q_base, jnp.zeros((self._action_dim, num_constraints))],
            [jnp.zeros((num_constraints, self._action_dim)), Q_slack]
        ])

        # Extend c vector with zeros for slack
        c_slack = jnp.zeros(num_constraints)
        c_vector = jnp.concatenate([c_base, c_slack])

        return Q_matrix, c_vector

    def _make_ineq_const_single(self, x: jnp.ndarray) -> tuple:
        """
        Create inequality constraints for single state.

        CBF constraint: Lf h + Lg h * u + alpha(h) >= 0
        Rewritten as: -Lg h * u <= Lf h + alpha(h)

        Args:
            x: Single state vector (state_dim,)

        Returns:
            Tuple (G, h) for inequality Gu <= h
        """
        # Get barrier values and Lie derivatives for single state
        hocbf, lf_hocbf, lg_hocbf = self._barrier._get_hocbf_and_lie_derivs_single(x)

        # Convert to QP form: Gu <= h
        # CBF constraint: -Lg_hocbf * u <= Lf_hocbf + alpha(hocbf)
        G = -lg_hocbf  # Shape: (num_barriers, action_dim)
        h = (lf_hocbf + jax.vmap(self._alpha)(hocbf))  # Shape: (num_barriers,)

        return G, h

    def _make_ineq_const_slacked_single(self, x: jnp.ndarray) -> tuple:
        """
        Create inequality constraints with slack for single state.

        CBF constraint with slack: Lf h + Lg h * u + alpha(h) + slack * h >= 0

        Args:
            x: Single state vector (state_dim,)

        Returns:
            Tuple (G, h) for inequality G[u; slack] <= h
        """
        # Get barrier values and Lie derivatives for single state
        hocbf, lf_hocbf, lg_hocbf = self._barrier._get_hocbf_and_lie_derivs_single(x)

        # Create constraint matrix for [u, slack]
        # -Lg_hocbf * u - hocbf * slack <= Lf_hocbf + alpha(hocbf)
        G_u = -lg_hocbf  # Shape: (num_barriers, action_dim)
        G_slack = -jnp.diag(hocbf)  # Shape: (num_barriers, num_barriers) - diagonal matrix
        G = jnp.concatenate([G_u, G_slack], axis=1)  # Shape: (num_barriers, action_dim + num_barriers)

        h = (lf_hocbf + jax.vmap(self._alpha)(hocbf))  # Shape: (num_barriers,)

        return G, h

    def _make_eq_const_single(self, x: jnp.ndarray, Q_matrix: jnp.ndarray) -> tuple:
        """
        Create equality constraints for single state.

        Default implementation returns empty constraints using zeros as required by qpax.

        Args:
            x: Single state vector (state_dim,)

        Returns:
            Tuple (A, b) for equality Au = b (empty by default)
        """
        A = jnp.zeros((0, Q_matrix.shape[0]))
        b = jnp.zeros(0)
        return A, b


class MinIntervQPSafeControl(QPSafeControl, BaseMinIntervSafeControl):
    """
    Minimum Intervention QP-based Safe Control.

    Automatically sets up quadratic cost to minimize deviation from
    desired control: min ||u - u_d||^2
    """

    def __init__(self, action_dim: int, alpha: Optional[Callable] = None,
                 params: Optional[dict] = None, dynamics=None, barrier=None,
                 desired_control=None, Q=None, c=None, slacked: bool = False, slack_gain: float = 100.0):
        """Initialize MinIntervQPSafeControl."""
        # Initialize BaseMinIntervSafeControl first
        BaseMinIntervSafeControl.__init__(self, action_dim, alpha, params, desired_control,
                                         dynamics, barrier, Q, c)
        # Set QP-specific fields
        self._slacked = slacked if params is None else params.get('slacked', slacked)
        self._slack_gain = slack_gain if params is None else params.get('slack_gain', slack_gain)

    def _create_updated_instance(self, **kwargs):
        """
        Create new instance with updated fields.

        Args:
            **kwargs: Fields to update

        Returns:
            New MinIntervQPSafeControl instance with updated fields
        """
        defaults = {
            'action_dim': self._action_dim,
            'alpha': self._alpha,
            'params': dict(self._params),
            'dynamics': self._dynamics,
            'barrier': self._barrier,
            'desired_control': self._desired_control,
            'Q': self._Q,
            'c': self._c,
            'slacked': self._slacked,
            'slack_gain': self._slack_gain
        }
        defaults.update(kwargs)
        return self.__class__(**defaults)

    def assign_desired_control(self, desired_control: Callable) -> 'MinIntervQPSafeControl':
        """
        Assign desired control and automatically set up cost.

        Args:
            desired_control: Function that computes desired control

        Returns:
            New MinIntervQPSafeControl instance with cost set up
        """
        # Create cost functions for minimum intervention
        # min ||u - u_d||^2 = min u^T*I*u - 2*u_d^T*u + u_d^T*u_d
        # We only need the parts with u: u^T*I*u - 2*u_d^T*u
        Q_func = lambda x: 2.0 * jnp.eye(self._action_dim)
        c_func = lambda x: -2.0 * desired_control(x)

        return self._create_updated_instance(
            desired_control=desired_control,
            Q=Q_func,
            c=c_func
        )

    def assign_cost(self, Q: Callable, c: Callable) -> 'MinIntervQPSafeControl':
        """
        Internal method to assign cost functions.

        Args:
            Q: Cost matrix function
            c: Cost vector function

        Returns:
            New instance with assigned cost
        """
        return self._create_updated_instance(Q=Q, c=c)

    def assign_state_barrier(self, barrier) -> 'MinIntervQPSafeControl':
        """Assign state barrier."""
        return self._create_updated_instance(barrier=barrier)

    def assign_dynamics(self, dynamics) -> 'MinIntervQPSafeControl':
        """Assign dynamics."""
        return self._create_updated_instance(dynamics=dynamics)


class InputConstQPSafeControl(QPSafeControl):
    """
    Input-Constrained QP-based Safe Control.

    Handles control input bounds as additional linear constraints
    in the QP formulation.
    """

    # Input bounds
    _control_low: tuple = eqx.field(static=True)
    _control_high: tuple = eqx.field(static=True)
    _has_control_bounds: bool = eqx.field(static=True)

    def __init__(self, action_dim: int, alpha: Optional[Callable] = None,
                 params: Optional[dict] = None, dynamics=None, barrier=None,
                 Q=None, c=None, control_low=None, control_high=None,
                 slacked: bool = False, slack_gain: float = 100.0):
        """Initialize InputConstQPSafeControl."""
        super().__init__(action_dim, alpha, params, dynamics, barrier, Q, c, slacked, slack_gain)

        # Set control bounds as tuples
        if control_low is not None and control_high is not None:
            # Convert to tuples for static fields
            self._control_low = tuple(control_low) if not isinstance(control_low, tuple) else control_low
            self._control_high = tuple(control_high) if not isinstance(control_high, tuple) else control_high
            self._has_control_bounds = True
        else:
            self._control_low = tuple([0.0] * action_dim)
            self._control_high = tuple([0.0] * action_dim)
            self._has_control_bounds = False

    def _create_updated_instance(self, **kwargs):
        """
        Create new instance with updated fields.

        Args:
            **kwargs: Fields to update

        Returns:
            New InputConstQPSafeControl instance with updated fields
        """
        defaults = {
            'action_dim': self._action_dim,
            'alpha': self._alpha,
            'params': dict(self._params),
            'dynamics': self._dynamics,
            'barrier': self._barrier,
            'Q': self._Q,
            'c': self._c,
            'control_low': self._control_low if self._has_control_bounds else None,
            'control_high': self._control_high if self._has_control_bounds else None,
            'slacked': self._slacked,
            'slack_gain': self._slack_gain
        }
        defaults.update(kwargs)
        return self.__class__(**defaults)

    def assign_control_bounds(self, low: list, high: list) :
        """
        Assign control input bounds.

        Args:
            low: Lower bounds for control inputs
            high: Upper bounds for control inputs

        Returns:
            New InputConstQPSafeControl with bounds assigned
        """
        assert len(low) == len(high), 'low and high should have the same length'
        assert len(low) == self._action_dim, 'bounds length should match action dimension'

        return self._create_updated_instance(control_low=low, control_high=high)

    @jax.jit
    def _optimal_control_single(self, x: jnp.ndarray) -> tuple:
        """
        Compute safe optimal control for single state with input constraints.

        Args:
            x: Single state vector (state_dim,)

        Returns:
            Tuple (u, info) where info is a dict with slack_vars and constraint_at_u
        """
        if self._slacked:
            return self._optimal_control_single_slacked(x)

        # Make objective for single state
        Q_matrix, c_vector = self._make_objective_single(x)

        # Get CBF constraints
        hocbf, lf_hocbf, lg_hocbf = self._barrier._get_hocbf_and_lie_derivs_single(x)
        G_cbf = -lg_hocbf
        h_cbf = (lf_hocbf + jax.vmap(self._alpha)(hocbf))

        if self._has_control_bounds:
            # Add control bound constraints
            G_low = -jnp.eye(self._action_dim)
            h_low = -jnp.array(self._control_low)
            G_high = jnp.eye(self._action_dim)
            h_high = jnp.array(self._control_high)

            # Combine constraints
            G = jnp.vstack([G_cbf, G_low, G_high])
            h = jnp.concatenate([h_cbf, h_low, h_high])
        else:
            G, h = G_cbf, h_cbf

        # Make equality constraints
        A, b = self._make_eq_const_single(x, Q_matrix)

        # Solve QP
        u = solve_qp_primal(Q_matrix, c_vector, A, b, G, h)

        # Compute constraint at u for info
        constraint_at_u = jnp.dot(G, u) - h
        slack_vars = jnp.zeros(1)

        info = {'slack_vars': slack_vars, 'constraint_at_u': constraint_at_u}
        return u, info

    def _optimal_control_single_slacked(self, x: jnp.ndarray) -> tuple:
        """
        Compute safe optimal control with slack variables for single state.

        Args:
            x: Single state vector (state_dim,)

        Returns:
            Tuple (u, info) where info is a dict with slack_vars and constraint_at_u
        """
        # Get CBF constraints with slack (base method)
        G_cbf, h_cbf = super()._make_ineq_const_slacked_single(x)
        num_cbf_constraints = h_cbf.shape[0]

        # Make objective with slack for CBF constraints only
        Q_matrix, c_vector = self._make_objective_slacked_single(x, num_cbf_constraints)

        if self._has_control_bounds:
            # Add control bound constraints (no slack variables for these)
            num_slack = G_cbf.shape[1] - self._action_dim

            # Control bound constraints with zero columns for slack
            G_low = jnp.hstack([-jnp.eye(self._action_dim), jnp.zeros((self._action_dim, num_slack))])
            h_low = -jnp.array(self._control_low)

            G_high = jnp.hstack([jnp.eye(self._action_dim), jnp.zeros((self._action_dim, num_slack))])
            h_high = jnp.array(self._control_high)

            # Combine CBF constraints (with slack) and control bound constraints
            G = jnp.vstack([G_cbf, G_low, G_high])
            h = jnp.concatenate([h_cbf, h_low, h_high])
        else:
            G, h = G_cbf, h_cbf

        # Make equality constraints
        A, b = self._make_eq_const_single(x, Q_matrix)

        # Solve QP for augmented decision variable [u, slack]
        res = solve_qp_primal(Q_matrix, c_vector, A, b, G, h)

        # Extract control and slack
        u = res[:self._action_dim]
        slack_vars = res[self._action_dim:]

        # Compute constraint at solution
        constraint_at_u = jnp.dot(G, res) - h

        info = {'slack_vars': slack_vars, 'constraint_at_u': constraint_at_u}
        return u, info

    def assign_state_barrier(self, barrier) -> 'InputConstQPSafeControl':
        """Assign state barrier."""
        return self._create_updated_instance(barrier=barrier)

    def assign_dynamics(self, dynamics) -> 'InputConstQPSafeControl':
        """Assign dynamics."""
        return self._create_updated_instance(dynamics=dynamics)


class MinIntervInputConstQPSafeControl(InputConstQPSafeControl, MinIntervQPSafeControl):
    """
    Minimum Intervention Input-Constrained QP-based Safe Control.

    Combines minimum intervention with input constraints.
    """

    # Input bounds
    _control_low: tuple = eqx.field(static=True)
    _control_high: tuple = eqx.field(static=True)
    _has_control_bounds: bool = eqx.field(static=True)

    def __init__(self, action_dim: int, alpha: Optional[Callable] = None,
                 params: Optional[dict] = None, dynamics=None, barrier=None,
                 desired_control=None, Q=None, c=None, control_low=None, control_high=None,
                 slacked: bool = False, slack_gain: float = 100.0):
        """Initialize MinIntervInputConstQPSafeControl."""

        MinIntervQPSafeControl.__init__(self, action_dim, alpha, params, dynamics,
                                       barrier, desired_control, Q, c, slacked, slack_gain)

        # Set control bounds as tuples
        if control_low is not None and control_high is not None:
            # Convert to tuples for static fields
            self._control_low = tuple(control_low) if not isinstance(control_low, tuple) else control_low
            self._control_high = tuple(control_high) if not isinstance(control_high, tuple) else control_high
            self._has_control_bounds = True
        else:
            self._control_low = tuple([0.0] * action_dim)
            self._control_high = tuple([0.0] * action_dim)
            self._has_control_bounds = False

    def _create_updated_instance(self, **kwargs):
        """
        Create new instance with updated fields.

        Args:
            **kwargs: Fields to update

        Returns:
            New MinIntervInputConstQPSafeControl instance with updated fields
        """
        defaults = {
            'action_dim': self._action_dim,
            'alpha': self._alpha,
            'params': dict(self._params),
            'dynamics': self._dynamics,
            'barrier': self._barrier,
            'desired_control': self._desired_control,
            'Q': self._Q,
            'c': self._c,
            'control_low': self._control_low if self._has_control_bounds else None,
            'control_high': self._control_high if self._has_control_bounds else None,
            'slacked': self._slacked,
            'slack_gain': self._slack_gain
        }
        defaults.update(kwargs)
        return self.__class__(**defaults)

    def assign_desired_control(self, desired_control: Callable) -> 'MinIntervInputConstQPSafeControl':
        """Assign desired control and set up cost."""
        # Set up minimum intervention cost
        Q_func = lambda x: 2.0 * jnp.eye(self._action_dim)
        c_func = lambda x: -2.0 * desired_control(x)

        return self._create_updated_instance(
            desired_control=desired_control,
            Q=Q_func,
            c=c_func
        )