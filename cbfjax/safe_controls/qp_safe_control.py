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

from ..utils.qp import get_qp_solver, get_qp_init_state

from .base_safe_control import BaseCBFSafeControl, BaseMinIntervSafeControl
from ..controls.control_types import QPInfo


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
    _slack_gain: float
    _qp_solver: Callable = eqx.field(static=True)
    _qp_init_state_fn: Callable = eqx.field(static=True)

    def __init__(self, **kwargs):
        """
        Initialize QPSafeControl with cooperative inheritance.

        Args:
            **kwargs: Passed via cooperative inheritance (alpha, Q, c, barrier, dynamics, action_dim, params)
                      Options slacked, slack_gain, qp_solver are read from params.
        """
        params = dict(kwargs.pop('params', None) or {})
        slacked = params.get('slacked', False)
        slack_gain = params.get('slack_gain', 100.0)
        params.setdefault('slacked', slacked)
        params.setdefault('slack_gain', slack_gain)
        params.setdefault('qp_solver', 'qpax')
        kwargs['params'] = params

        super().__init__(**kwargs)

        self._slacked = slacked
        self._slack_gain = slack_gain
        self._qp_solver = get_qp_solver(params['qp_solver'])
        self._qp_init_state_fn = get_qp_init_state(params['qp_solver'])

    def _ctor_defaults(self) -> dict:
        return {
            'action_dim': self._action_dim,
            'alpha': self._alpha,
            'params': dict(self._params),
            'dynamics': self._dynamics,
            'barrier': self._barrier,
            'Q': self._Q,
            'c': self._c,
        }

    @property
    def _qp_warm_start(self) -> bool:
        """Whether the selected backend threads a real QP warm-start state."""
        return self._qp_init_state_fn(1, 1, 0) is not None

    def _split_state(self, state):
        """
        Split the controller state into (ctrl_state, qp_state).

        When warm-start is active the controller state is a (ctrl_state, qp_state)
        tuple; otherwise it is the bare ctrl_state and the QP state is None.
        """
        if self._qp_warm_start and isinstance(state, tuple) and len(state) == 2:
            return state[0], state[1]
        return state, None

    def _merge_state(self, ctrl_state, qp_state):
        """Recombine ctrl_state and qp_state into the controller state."""
        if self._qp_warm_start:
            return (ctrl_state, qp_state)
        return ctrl_state

    def _qp_init_state_for(self, n: int, m: int, n_eq: int = 0):
        """Build the QP warm-start init-state for problem dims (None if stateless)."""
        return self._qp_init_state_fn(n, m, n_eq)

    def get_init_state(self):
        """
        Initial controller state.

        Without warm-start (e.g. qpax) this returns the bare desired/objective
        controller state (None for a stateless desired control). With a
        warm-start backend (jaxopt_osqp) it returns a (ctrl_state, qp_state)
        tuple whose qp_state is a structured zero state sized for the QP this
        filter assembles, so the state pytree carried across lax.scan keeps an
        identical structure every step.
        """
        ctrl_state = super().get_init_state()
        if not self._qp_warm_start:
            return ctrl_state
        qp_state = self._infer_qp_init_state(ctrl_state)
        return (ctrl_state, qp_state)

    def _infer_qp_init_state(self, ctrl_state):
        """
        Derive the structured QP warm-start init-state from assembled dims.

        Assembles the QP once for a dummy state to read (n, m, n_eq) from the
        Q/G/A shapes, then asks the backend builder for a matching zero state.
        Only the static shapes are used, so the assembled values are discarded.
        """
        x0 = jnp.zeros(self._dynamics.state_dim)
        Q_m, _c, G, _h, A, _b, _st = self._assemble_qp(x0, ctrl_state)
        n = Q_m.shape[0]
        m = G.shape[0]
        n_eq = A.shape[0]
        return self._qp_init_state_for(n, m, n_eq)

    def _assemble_qp(self, x: jnp.ndarray, state) -> tuple:
        """
        Assemble the QP for a single state: (Q, c, G, h, A, b, new_state).

        Threads the desired/objective controller state through the objective.
        Subclasses override to add bounds (input-constrained) or slack columns.
        """
        if self._slacked:
            G, h = self._make_ineq_const_slacked(x)
            num_constraints = h.shape[0]
            Q_matrix, c_vector, state = self._make_objective_slacked(
                x, num_constraints, state)
        else:
            Q_matrix, c_vector, state = self._make_objective(x, state)
            G, h = self._make_ineq_const(x)
        A, b = self._make_eq_const(x, Q_matrix)
        return Q_matrix, c_vector, G, h, A, b, state

    @jax.jit
    def optimal_control(self, x: jnp.ndarray, state=None) -> tuple:
        """
        Compute safe optimal control for a single state using QP.

        Args:
            x: Single state vector (state_dim,)
            state: Controller state (carries objective state + QP warm start)

        Returns:
            Tuple (u, new_state)
        """
        if self._slacked:
            return self._optimal_control_slacked(x, state)

        ctrl_state, qp_state = self._split_state(state)

        Q_matrix, c_vector, ctrl_state = self._make_objective(x, ctrl_state)

        # Make inequality constraints for single state
        G, h = self._make_ineq_const(x)

        # Make equality constraints (empty by default)
        A, b = self._make_eq_const(x, Q_matrix)

        # Solve QP: min 0.5 u^T Q u + c^T u s.t. Gu <= h, Au = b
        u, qp_state = self._qp_solver(Q_matrix, c_vector, G, h, A, b, qp_state)

        return u, self._merge_state(ctrl_state, qp_state)

    def optimal_control_with_info(self, x: jnp.ndarray, state=None) -> tuple:
        """
        Compute safe optimal control with diagnostic info.

        Args:
            x: Single state vector (state_dim,)
            state: Controller state (threaded through stateful Q/c)

        Returns:
            Tuple (u, new_state, info)
        """
        if self._slacked:
            return self._optimal_control_slacked_with_info(x, state)

        ctrl_state, qp_state = self._split_state(state)
        Q_matrix, c_vector, ctrl_state = self._make_objective(x, ctrl_state)
        G, h = self._make_ineq_const(x)
        A, b = self._make_eq_const(x, Q_matrix)
        u, qp_state = self._qp_solver(Q_matrix, c_vector, G, h, A, b, qp_state)

        u_desired = -jnp.linalg.solve(Q_matrix, c_vector)
        constraint_at_u = jnp.dot(G, u) - h
        slack_vars = jnp.zeros(1)
        info = QPInfo(slack_vars=slack_vars, constraint_at_u=constraint_at_u, u_desired=u_desired)
        return u, self._merge_state(ctrl_state, qp_state), info

    def _optimal_control_slacked(self, x: jnp.ndarray, state=None) -> tuple:
        """
        Compute safe optimal control with slack variables for single state.

        Args:
            x: Single state vector (state_dim,)
            state: Controller state (threaded through stateful Q/c)

        Returns:
            Tuple (u, new_state)
        """
        ctrl_state, qp_state = self._split_state(state)

        # Make inequality constraints for slacked version
        G, h = self._make_ineq_const_slacked(x)
        num_constraints = h.shape[0]

        # Make objective for slacked version (stateful)
        Q_matrix, c_vector, ctrl_state = self._make_objective_slacked(x, num_constraints, ctrl_state)

        # Make equality constraints
        A, b = self._make_eq_const(x, Q_matrix)

        # Solve QP for augmented decision variable [u, slack]
        res, qp_state = self._qp_solver(Q_matrix, c_vector, G, h, A, b, qp_state)

        # Extract control
        u = res[:self._action_dim]

        return u, self._merge_state(ctrl_state, qp_state)

    def _optimal_control_slacked_with_info(self, x: jnp.ndarray, state=None) -> tuple:
        """
        Compute safe optimal control with slack variables and diagnostic info.

        Args:
            x: Single state vector (state_dim,)
            state: Controller state (threaded through stateful Q/c)

        Returns:
            Tuple (u, new_state, info)
        """
        ctrl_state, qp_state = self._split_state(state)
        G, h = self._make_ineq_const_slacked(x)
        num_constraints = h.shape[0]
        Q_matrix, c_vector, ctrl_state = self._make_objective_slacked(x, num_constraints, ctrl_state)
        A, b = self._make_eq_const(x, Q_matrix)
        res, qp_state = self._qp_solver(Q_matrix, c_vector, G, h, A, b, qp_state)

        u = res[:self._action_dim]
        slack_vars = res[self._action_dim:]
        constraint_at_u = jnp.dot(G, res) - h

        # Extract u_desired from non-augmented Q/c
        Q_orig = Q_matrix[:self._action_dim, :self._action_dim]
        c_orig = c_vector[:self._action_dim]
        u_desired = -jnp.linalg.solve(Q_orig, c_orig)
        info = QPInfo(slack_vars=slack_vars, constraint_at_u=constraint_at_u, u_desired=u_desired)
        return u, self._merge_state(ctrl_state, qp_state), info

    def _make_objective(self, x: jnp.ndarray, state=None) -> tuple:
        """
        Create objective matrices for single state.

        Args:
            x: Single state vector (state_dim,)
            state: Controller state (threaded through stateful Q/c)

        Returns:
            Tuple (Q, c, new_state) for quadratic objective
        """
        Q_matrix, state = self._Q(x, state)  # (action_dim, action_dim)
        c_vector, state = self._c(x, state)  # (action_dim,)

        return Q_matrix, c_vector, state

    def _make_objective_slacked(self, x: jnp.ndarray, num_constraints: int, state=None) -> tuple:
        """
        Create objective matrices with slack variables for single state.

        Args:
            x: Single state vector (state_dim,)
            num_constraints: Number of constraints (slack variables)
            state: Controller state (threaded through stateful Q/c)

        Returns:
            Tuple (Q, c, new_state) for augmented quadratic objective
        """
        Q_base, c_base, state = self._make_objective(x, state)

        # Create block diagonal Q matrix
        Q_slack = self._slack_gain * 0.5 * jnp.eye(num_constraints)
        Q_matrix = jnp.block([
            [Q_base, jnp.zeros((self._action_dim, num_constraints))],
            [jnp.zeros((num_constraints, self._action_dim)), Q_slack]
        ])

        # Extend c vector with zeros for slack
        c_slack = jnp.zeros(num_constraints)
        c_vector = jnp.concatenate([c_base, c_slack])

        return Q_matrix, c_vector, state

    def _make_ineq_const(self, x: jnp.ndarray) -> tuple:
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
        hocbf, lf_hocbf, lg_hocbf = self._barrier.get_hocbf_and_lie_derivs(x)

        # Ensure proper shapes (handles both scalar and array barriers)
        hocbf = jnp.atleast_1d(hocbf)
        lf_hocbf = jnp.atleast_1d(lf_hocbf)
        lg_hocbf = jnp.atleast_2d(lg_hocbf)

        # Convert to QP form: Gu <= h
        # CBF constraint: -Lg_hocbf * u <= Lf_hocbf + alpha(hocbf)
        G = -lg_hocbf  # Shape: (num_barriers, action_dim)
        h = (lf_hocbf + jax.vmap(self._alpha)(hocbf))  # Shape: (num_barriers,)

        return G, h

    def _make_ineq_const_slacked(self, x: jnp.ndarray) -> tuple:
        """
        Create inequality constraints with slack for single state.

        CBF constraint with slack: Lf h + Lg h * u + alpha(h) + slack * h >= 0

        Args:
            x: Single state vector (state_dim,)

        Returns:
            Tuple (G, h) for inequality G[u; slack] <= h
        """
        # Get barrier values and Lie derivatives for single state
        hocbf, lf_hocbf, lg_hocbf = self._barrier.get_hocbf_and_lie_derivs(x)

        # Ensure proper shapes (handles both scalar and array barriers)
        hocbf = jnp.atleast_1d(hocbf)
        lf_hocbf = jnp.atleast_1d(lf_hocbf)
        lg_hocbf = jnp.atleast_2d(lg_hocbf)

        # Create constraint matrix for [u, slack]
        # -Lg_hocbf * u - hocbf * slack <= Lf_hocbf + alpha(hocbf)
        G_u = -lg_hocbf  # Shape: (num_barriers, action_dim)
        G_slack = -jnp.diag(hocbf)  # Shape: (num_barriers, num_barriers) - diagonal matrix
        G = jnp.concatenate([G_u, G_slack], axis=1)  # Shape: (num_barriers, action_dim + num_barriers)

        h = (lf_hocbf + jax.vmap(self._alpha)(hocbf))  # Shape: (num_barriers,)

        return G, h

    def _make_eq_const(self, x: jnp.ndarray, Q_matrix: jnp.ndarray) -> tuple:
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

    Uses cooperative multiple inheritance.
    """

    def __init__(self, **kwargs):
        """
        Initialize MinIntervQPSafeControl with cooperative inheritance.

        When desired_control is given and no explicit cost is provided, the
        minimum intervention cost min ||u - u_d||^2 is derived ON DEMAND in
        ``_make_objective`` from the live ``_desired_control`` property,
        so a goal swapped through the leaf path (eqx.tree_at) propagates.

        Args:
            **kwargs: All args passed via cooperative inheritance
                - desired_control: Handled by BaseMinIntervSafeControl
                - alpha, Q, c, barrier: Handled by BaseCBFSafeControl
                - dynamics, action_dim, params: Handled by BaseControl
                  (slacked, slack_gain read from params)
        """
        super().__init__(**kwargs)

    def _make_objective(self, x: jnp.ndarray, state=None) -> tuple:
        """
        Min-intervention QP objective derived on demand from the desired control.

        When an explicit Q/c was provided, defer to the base implementation.
        Otherwise derive min ||u - u_d||^2 (Q = 2I, c = -2 u_d) from the live
        ``_desired_control`` property so leaf (e.g. goal) changes propagate.
        """
        if self._Q is not None or self._c is not None or self._desired_control is None:
            return super()._make_objective(x, state)
        action_dim = self._action_dim
        u_d, state = self._desired_control(x, state)
        Q_matrix = 2.0 * jnp.eye(action_dim)
        c_vector = -2.0 * u_d
        return Q_matrix, c_vector, state

    def _ctor_defaults(self) -> dict:
        return {
            'action_dim': self._action_dim,
            'alpha': self._alpha,
            'params': dict(self._params),
            'dynamics': self._dynamics,
            'barrier': self._barrier,
            'desired_control': self._emit_desired_control(),
            'desired_control_init_state': self._desired_control_init_state,
            'Q': self._Q,
            'c': self._c,
        }


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

    def __init__(
        self,
        control_low=None,
        control_high=None,
        **kwargs
    ):
        """
        Initialize InputConstQPSafeControl with cooperative inheritance.

        Args:
            control_low: Lower bounds for control inputs
            control_high: Upper bounds for control inputs
            **kwargs: Passed via cooperative inheritance (alpha, Q, c, barrier, dynamics, action_dim, params; slacked/slack_gain read from params)
        """
        # Get action_dim for default bounds
        action_dim = kwargs.get('action_dim', 1)

        # Initialize via cooperative inheritance
        super().__init__(**kwargs)

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

    def _ctor_defaults(self) -> dict:
        return {
            'action_dim': self._action_dim,
            'alpha': self._alpha,
            'params': dict(self._params),
            'dynamics': self._dynamics,
            'barrier': self._barrier,
            'Q': self._Q,
            'c': self._c,
            'control_low': self._control_low if self._has_control_bounds else None,
            'control_high': self._control_high if self._has_control_bounds else None,
        }

    def _assemble_qp(self, x: jnp.ndarray, state) -> tuple:
        """Assemble the input-constrained QP: (Q, c, G, h, A, b, new_state)."""
        if self._slacked:
            G_cbf, h_cbf = super()._make_ineq_const_slacked(x)
            num_cbf_constraints = h_cbf.shape[0]
            Q_matrix, c_vector, state = self._make_objective_slacked(
                x, num_cbf_constraints, state)
            if self._has_control_bounds:
                num_slack = G_cbf.shape[1] - self._action_dim
                G_low = jnp.hstack([-jnp.eye(self._action_dim), jnp.zeros((self._action_dim, num_slack))])
                h_low = -jnp.array(self._control_low)
                G_high = jnp.hstack([jnp.eye(self._action_dim), jnp.zeros((self._action_dim, num_slack))])
                h_high = jnp.array(self._control_high)
                G = jnp.vstack([G_cbf, G_low, G_high])
                h = jnp.concatenate([h_cbf, h_low, h_high])
            else:
                G, h = G_cbf, h_cbf
        else:
            Q_matrix, c_vector, state = self._make_objective(x, state)
            hocbf, lf_hocbf, lg_hocbf = self._barrier.get_hocbf_and_lie_derivs(x)
            hocbf = jnp.atleast_1d(hocbf)
            lf_hocbf = jnp.atleast_1d(lf_hocbf)
            lg_hocbf = jnp.atleast_2d(lg_hocbf)
            G_cbf = -lg_hocbf
            h_cbf = (lf_hocbf + jax.vmap(self._alpha)(hocbf))
            if self._has_control_bounds:
                G_low = -jnp.eye(self._action_dim)
                h_low = -jnp.array(self._control_low)
                G_high = jnp.eye(self._action_dim)
                h_high = jnp.array(self._control_high)
                G = jnp.vstack([G_cbf, G_low, G_high])
                h = jnp.concatenate([h_cbf, h_low, h_high])
            else:
                G, h = G_cbf, h_cbf
        A, b = self._make_eq_const(x, Q_matrix)
        return Q_matrix, c_vector, G, h, A, b, state

    @jax.jit
    def optimal_control(self, x: jnp.ndarray, state=None) -> tuple:
        """
        Compute safe optimal control for single state with input constraints.

        Args:
            x: Single state vector (state_dim,)
            state: Controller state (carries objective state + QP warm start)

        Returns:
            Tuple (u, new_state)
        """
        if self._slacked:
            return self._optimal_control_slacked(x, state)

        ctrl_state, qp_state = self._split_state(state)

        # Make objective for single state (stateful)
        Q_matrix, c_vector, ctrl_state = self._make_objective(x, ctrl_state)

        # Get CBF constraints
        hocbf, lf_hocbf, lg_hocbf = self._barrier.get_hocbf_and_lie_derivs(x)

        # Ensure proper shapes (handles both scalar and array barriers)
        hocbf = jnp.atleast_1d(hocbf)
        lf_hocbf = jnp.atleast_1d(lf_hocbf)
        lg_hocbf = jnp.atleast_2d(lg_hocbf)

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
        A, b = self._make_eq_const(x, Q_matrix)

        # Solve QP
        u, qp_state = self._qp_solver(Q_matrix, c_vector, G, h, A, b, qp_state)

        return u, self._merge_state(ctrl_state, qp_state)

    def optimal_control_with_info(self, x: jnp.ndarray, state=None) -> tuple:
        """
        Compute safe optimal control with diagnostic info for input-constrained QP.

        Args:
            x: Single state vector (state_dim,)
            state: Controller state (threaded through stateful Q/c)

        Returns:
            Tuple (u, new_state, info)
        """
        if self._slacked:
            return self._optimal_control_slacked_with_info(x, state)

        ctrl_state, qp_state = self._split_state(state)
        Q_matrix, c_vector, ctrl_state = self._make_objective(x, ctrl_state)
        hocbf, lf_hocbf, lg_hocbf = self._barrier.get_hocbf_and_lie_derivs(x)
        hocbf = jnp.atleast_1d(hocbf)
        lf_hocbf = jnp.atleast_1d(lf_hocbf)
        lg_hocbf = jnp.atleast_2d(lg_hocbf)

        G_cbf = -lg_hocbf
        h_cbf = (lf_hocbf + jax.vmap(self._alpha)(hocbf))

        if self._has_control_bounds:
            G_low = -jnp.eye(self._action_dim)
            h_low = -jnp.array(self._control_low)
            G_high = jnp.eye(self._action_dim)
            h_high = jnp.array(self._control_high)
            G = jnp.vstack([G_cbf, G_low, G_high])
            h = jnp.concatenate([h_cbf, h_low, h_high])
        else:
            G, h = G_cbf, h_cbf

        A, b = self._make_eq_const(x, Q_matrix)
        u, qp_state = self._qp_solver(Q_matrix, c_vector, G, h, A, b, qp_state)

        u_desired = -jnp.linalg.solve(Q_matrix, c_vector)
        constraint_at_u = jnp.dot(G, u) - h
        slack_vars = jnp.zeros(1)
        info = QPInfo(slack_vars=slack_vars, constraint_at_u=constraint_at_u, u_desired=u_desired)
        return u, self._merge_state(ctrl_state, qp_state), info

    def _optimal_control_slacked(self, x: jnp.ndarray, state=None) -> tuple:
        """
        Compute safe optimal control with slack variables for single state.

        Args:
            x: Single state vector (state_dim,)
            state: Controller state (threaded through stateful Q/c)

        Returns:
            Tuple (u, new_state)
        """
        ctrl_state, qp_state = self._split_state(state)

        # Get CBF constraints with slack (base method)
        G_cbf, h_cbf = super()._make_ineq_const_slacked(x)
        num_cbf_constraints = h_cbf.shape[0]

        # Make objective with slack for CBF constraints only (stateful)
        Q_matrix, c_vector, ctrl_state = self._make_objective_slacked(x, num_cbf_constraints, ctrl_state)

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
        A, b = self._make_eq_const(x, Q_matrix)

        # Solve QP for augmented decision variable [u, slack]
        res, qp_state = self._qp_solver(Q_matrix, c_vector, G, h, A, b, qp_state)

        # Extract control
        u = res[:self._action_dim]

        return u, self._merge_state(ctrl_state, qp_state)

    def _optimal_control_slacked_with_info(self, x: jnp.ndarray, state=None) -> tuple:
        """
        Compute safe optimal control with slack variables and diagnostic info.

        Args:
            x: Single state vector (state_dim,)
            state: Controller state (threaded through stateful Q/c)

        Returns:
            Tuple (u, new_state, info)
        """
        ctrl_state, qp_state = self._split_state(state)
        G_cbf, h_cbf = super()._make_ineq_const_slacked(x)
        num_cbf_constraints = h_cbf.shape[0]
        Q_matrix, c_vector, ctrl_state = self._make_objective_slacked(x, num_cbf_constraints, ctrl_state)

        if self._has_control_bounds:
            num_slack = G_cbf.shape[1] - self._action_dim
            G_low = jnp.hstack([-jnp.eye(self._action_dim), jnp.zeros((self._action_dim, num_slack))])
            h_low = -jnp.array(self._control_low)
            G_high = jnp.hstack([jnp.eye(self._action_dim), jnp.zeros((self._action_dim, num_slack))])
            h_high = jnp.array(self._control_high)
            G = jnp.vstack([G_cbf, G_low, G_high])
            h = jnp.concatenate([h_cbf, h_low, h_high])
        else:
            G, h = G_cbf, h_cbf

        A, b = self._make_eq_const(x, Q_matrix)
        res, qp_state = self._qp_solver(Q_matrix, c_vector, G, h, A, b, qp_state)

        u = res[:self._action_dim]
        slack_vars = res[self._action_dim:]
        constraint_at_u = jnp.dot(G, res) - h

        # Extract u_desired from non-augmented Q/c
        Q_orig = Q_matrix[:self._action_dim, :self._action_dim]
        c_orig = c_vector[:self._action_dim]
        u_desired = -jnp.linalg.solve(Q_orig, c_orig)
        info = QPInfo(slack_vars=slack_vars, constraint_at_u=constraint_at_u, u_desired=u_desired)
        return u, self._merge_state(ctrl_state, qp_state), info


class MinIntervInputConstQPSafeControl(InputConstQPSafeControl, MinIntervQPSafeControl):
    """
    Minimum Intervention Input-Constrained QP-based Safe Control.

    Combines minimum intervention with input constraints using cooperative
    inheritance. Desired control normalization and automatic cost derivation
    are inherited from MinIntervQPSafeControl.
    """

    def _ctor_defaults(self) -> dict:
        return {
            'action_dim': self._action_dim,
            'alpha': self._alpha,
            'params': dict(self._params),
            'dynamics': self._dynamics,
            'barrier': self._barrier,
            'desired_control': self._emit_desired_control(),
            'desired_control_init_state': self._desired_control_init_state,
            'Q': self._Q,
            'c': self._c,
            'control_low': self._control_low if self._has_control_bounds else None,
            'control_high': self._control_high if self._has_control_bounds else None,
        }