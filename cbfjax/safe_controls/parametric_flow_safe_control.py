"""
Parametric Flow Safe Control for JAX.

This module implements parametric flow-based safe control using QP formulation
with cost functionals over augmented state s = [x, θ, γ].
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import diffrax

from typing import Callable, Optional, Any, Dict, Tuple, NamedTuple
from functools import partial
from immutabledict import immutabledict

from cbfjax.safe_controls.qp_safe_control import InputConstQPSafeControl
from cbfjax.utils.integration import get_trajs_from_state_action_func, get_trajs_from_state_action_func_zoh, get_solver
from cbfjax.controls.control_types import QPInfo
from jaxopt import OSQP


class FlowControlState(NamedTuple):
    """State for FlowBarrier controller (warm-start OSQP)."""
    osqp_params: Any  # KKTSolution pytree from jaxopt OSQP


def _matvec_Q_diag(Q_diag, x):
    return Q_diag * x


def _matvec_dense(M, x):
    return M @ x


# OSQP instances: one for the diagonal-Hessian primal, one for the small dense dual.
_osqp = OSQP(matvec_Q=_matvec_Q_diag, tol=1e-5, maxiter=200, jit=True)
_osqp_dual = OSQP(matvec_Q=_matvec_dense, tol=1e-5, maxiter=200, jit=True)


@jax.jit
def _solve_qp_primal(Q_diag, c, G, h, state=None):
    """Solve the n-variable primal QP directly with OSQP.

        min 0.5 v' diag(Q) v + c' v   s.t.  G v <= h
    """
    osqp_params = state.osqp_params if state is not None else None
    sol = _osqp.run(init_params=osqp_params, params_obj=(Q_diag, c), params_ineq=(G, h))
    return sol.params.primal, FlowControlState(osqp_params=sol.params)


@jax.jit
def _solve_qp_dual(Q_diag, c, G, h, state=None):
    """Solve the m-variable dual QP and recover the primal.

        primal:  min 0.5 v' diag(Q) v + c' v   s.t.  G v <= h     (n vars, m constraints)
        dual:    min 0.5 lam' M lam + q' lam    s.t.  lam >= 0     (m vars)
                 M = G Q^-1 G',   q = G Q^-1 c + h
        recover: v* = -Q^-1 (c + G' lam*)
    """
    Qinv = 1.0 / Q_diag
    GQi  = G * Qinv[None, :]                       # (m, n) = G diag(Q^-1)
    M    = GQi @ G.T                               # (m, m) dense dual Hessian (PSD)
    q    = GQi @ c + h                             # (m,)
    m    = h.shape[0]
    G_dual = -jnp.eye(m, dtype=Q_diag.dtype)       # lam >= 0  ->  -I lam <= 0
    h_dual = jnp.zeros(m, dtype=Q_diag.dtype)
    osqp_params = state.osqp_params if state is not None else None
    sol = _osqp_dual.run(init_params=osqp_params, params_obj=(M, q), params_ineq=(G_dual, h_dual))
    lam = sol.params.primal
    v   = -Qinv * (c + G.T @ lam)
    return v, FlowControlState(osqp_params=sol.params)


# Registry for the `qp_mode` flag (read from controller params; default 'primal').
_QP_SOLVERS = {'primal': _solve_qp_primal, 'dual': _solve_qp_dual}


class ParametricFlowSafeControl(InputConstQPSafeControl):
    """
    Parametric Flow Safe Control using QP formulation with cost functional.

    This controller optimizes over augmented control inputs v_aug = [u, ω, z]
    for the augmented state s = [x, θ, γ] where:
    - u: physical control for original system
    - ω: parameter update rate for θ (dθ/dt = ω)
    - z: time-shift update rate for γ (dγ/dt = z)

    The objective combines:
    - Control cost: (u - u_p)^T R (u - u_p) where u_p is parametric control
    - Parameter regularization: ω^T Λ ω
    - Time-shift cost: z^T Μ z + λ_linear * z
    - Cost functional gradient: ∇J(s)^T G(s) v_aug

    All fields are immutable following Equinox patterns.
    """

    # Flow-specific components (static)
    _cost_functional: Optional[Callable] = eqx.field(static=True)
    _flow_barrier: Any = eqx.field(static=True)

    # Alpha functions for different barrier types
    alpha_trajectory: Optional[Callable] = eqx.field(static=True)
    alpha_backup: Optional[Callable] = eqx.field(static=True)
    alpha_action: Optional[Callable] = eqx.field(static=True)
    alpha_time_shift: Optional[Callable] = eqx.field(static=True)

    # Cached dimensions
    _aug_action_dim: int = eqx.field(static=True)
    _theta_flat_dim: int = eqx.field(static=True)

    def __init__(
            self,
            action_dim: int,
            alpha: Optional[Callable] = None,
            params: Optional[dict] = None,
            dynamics=None,
            barrier=None,
            Q=None,
            c=None,
            control_low=None,
            control_high=None,
            slacked: bool = False,
            slack_gain: float = 100.0,
            # Flow-specific parameters
            cost_functional=None,
            flow_barrier=None,
            alpha_trajectory=None,
            alpha_backup=None,
            alpha_action=None,
            alpha_time_shift=None,
            aug_action_dim=0,
            theta_flat_dim=0,
            # Cost matrices (alternative to callable Q/c)
            R=None,
            Lambda=None,
            Mu=None,
            lambda_linear=None
    ):
        """
        Initialize ParametricFlowSafeControl.

        Args:
            action_dim: Control input dimension
            alpha: Class-K function for barrier constraint
            params: Legacy parameter dictionary
            dynamics: System dynamics object
            barrier: Barrier function object
            Q: Cost matrix function
            c: Cost vector function
            control_low: Lower control bounds
            control_high: Upper control bounds
            slacked: Whether to use slack variables
            slack_gain: Gain for slack variables
            cost_functional: Cost functional J(s)
            flow_barrier: FlowBarrier instance
            alpha_trajectory: Alpha function for trajectory constraints
            alpha_backup: Alpha function for backup constraints
            alpha_action: Alpha function for action constraints
            alpha_time_shift: Alpha function for time shift constraint
            aug_action_dim: Augmented action dimension
            theta_flat_dim: Flattened theta dimension
        """
        from ..barriers.parametric_flow_barrier import FlowBarrier

        # A FlowBarrier passed as barrier= (e.g. factory wiring) sets the flow
        # fields and augmented dynamics automatically.
        if isinstance(barrier, FlowBarrier) and flow_barrier is None:
            flow_barrier = barrier
            theta_flat_dim = (barrier.original_dynamics.action_dim *
                              barrier.control_param_num)
            aug_action_dim = (barrier.original_dynamics.action_dim
                              + theta_flat_dim + 1)
            dynamics = barrier._augmented_dynamics

        # Cost matrices -> callable Q/c (same construction as assign_cost_matrices)
        if R is not None:
            if flow_barrier is None:
                raise ValueError("cost matrices require a FlowBarrier "
                                 "('barrier' or 'flow_barrier')")
            _fb = flow_barrier
            _ad, _td, _gd = int(action_dim), int(theta_flat_dim), int(aug_action_dim)
            Q_matrix = jnp.zeros((_gd, _gd))
            Q_matrix = Q_matrix.at[:_ad, :_ad].set(R)
            Q_matrix = Q_matrix.at[_ad:_ad + _td, _ad:_ad + _td].set(Lambda)
            Q_matrix = Q_matrix.at[-1, -1].set(Mu)

            def _Q_func(x, theta, gamma):
                return Q_matrix

            def _c_func(x, theta, gamma):
                c_vec = jnp.zeros(_gd)
                u_p = _fb._parametric_control(gamma, theta)
                c_vec = c_vec.at[:_ad].set(-R @ u_p)
                c_vec = c_vec.at[-1].set(lambda_linear)
                return c_vec

            Q, c = _Q_func, _c_func

        # Initialize parent class
        super().__init__(
            action_dim=action_dim,
            alpha=alpha,
            params=params,
            dynamics=dynamics,
            barrier=barrier,
            Q=Q,
            c=c,
            control_low=control_low,
            control_high=control_high,
            slacked=slacked,
            slack_gain=slack_gain
        )

        # Flow-specific fields
        self._cost_functional = cost_functional
        self._flow_barrier = flow_barrier

        # Alpha functions
        self.alpha_trajectory = alpha_trajectory
        self.alpha_backup = alpha_backup
        self.alpha_action = alpha_action
        self.alpha_time_shift = alpha_time_shift

        # Cached dimensions
        self._aug_action_dim = int(aug_action_dim)
        self._theta_flat_dim = int(theta_flat_dim)

    @classmethod
    def create_empty(cls, action_dim: int, alpha: Optional[Callable] = None,
                     params: Optional[dict] = None) -> 'ParametricFlowSafeControl':
        """
        Create empty ParametricFlowSafeControl instance.

        Args:
            action_dim: Control input dimension
            alpha: Class-K function
            params: Optional parameter dictionary

        Returns:
            Empty ParametricFlowSafeControl instance
        """
        return cls(action_dim=action_dim, alpha=alpha, params=params)

    def assign_cost_functional(
            self,
            cost_functional: Callable[[jnp.ndarray], jnp.ndarray]
    ) -> 'ParametricFlowSafeControl':
        """
        Assign cost functional J(s) that takes augmented state s = [x, θ, γ].

        Args:
            cost_functional: Function s -> scalar cost

        Returns:
            New instance with assigned cost functional
        """
        return self._create_updated_instance(cost_functional=cost_functional)

    def assign_state_barrier(self, barrier) -> 'ParametricFlowSafeControl':
        """
        Assign FlowBarrier and automatically set dynamics.

        Args:
            barrier: FlowBarrier instance

        Returns:
            New instance with assigned barrier
        """
        from ..barriers.parametric_flow_barrier import FlowBarrier

        if not isinstance(barrier, FlowBarrier):
            raise TypeError(f"Expected FlowBarrier, got {type(barrier)}")

        # Cache augmented dimensions
        theta_flat_dim = (barrier.original_dynamics.action_dim *
                          barrier.control_param_num)
        aug_action_dim = barrier.original_dynamics.action_dim + theta_flat_dim + 1

        return self._create_updated_instance(
            flow_barrier=barrier,
            barrier=barrier,
            dynamics=barrier._augmented_dynamics,
            theta_flat_dim=theta_flat_dim,
            aug_action_dim=aug_action_dim
        )

    def assign_alpha_functions(
            self,
            alpha_trajectory: Optional[Callable] = None,
            alpha_backup: Optional[Callable] = None,
            alpha_action: Optional[Callable] = None,
            alpha_time_shift: Optional[Callable] = None
    ) -> 'ParametricFlowSafeControl':
        """
        Assign different alpha functions for different barrier types.

        Args:
            alpha_trajectory: Class-K function for trajectory constraints
            alpha_backup: Class-K function for backup barrier
            alpha_action: Class-K function for action constraints
            alpha_time_shift: Class-K function for time shift barrier

        Returns:
            New instance with assigned alpha functions
        """
        return self._create_updated_instance(
            alpha_trajectory=alpha_trajectory,
            alpha_backup=alpha_backup,
            alpha_action=alpha_action,
            alpha_time_shift=alpha_time_shift
        )

    def assign_cost_matrices(
            self,
            R: jnp.ndarray,
            Lambda: jnp.ndarray,
            Mu: float,
            lambda_linear: float
    ) -> 'ParametricFlowSafeControl':
        """
        Assign cost matrices and create Q and c functions.

        Similar to MinIntervQPSafeControl, this creates callable Q and c functions
        that are stored as static fields (inherited from parent class).

        Args:
            R: Control cost matrix (action_dim, action_dim)
            Lambda: Parameter regularization matrix (theta_flat_dim, theta_flat_dim)
            Mu: Time-shift cost scalar
            lambda_linear: Linear time-shift cost

        Returns:
            New instance with Q and c functions assigned
        """
        action_dim = self._action_dim
        theta_flat_dim = self._theta_flat_dim
        aug_action_dim = self._aug_action_dim
        flow_barrier = self._flow_barrier

        # Precompute Q matrix
        Q_matrix = jnp.zeros((aug_action_dim, aug_action_dim))
        Q_matrix = Q_matrix.at[:action_dim, :action_dim].set(R)
        Q_matrix = Q_matrix.at[action_dim:action_dim+theta_flat_dim, action_dim:action_dim+theta_flat_dim].set(Lambda)
        Q_matrix = Q_matrix.at[-1, -1].set(Mu)

        def Q_func(x, theta, gamma):
            return Q_matrix

        def c_func(x, theta, gamma):
            """Static part of c vector (no trajectory computation).
            The cost functional gradient term is added in optimal_control."""
            c = jnp.zeros(aug_action_dim)
            u_p = flow_barrier._parametric_control(gamma, theta)
            c = c.at[:action_dim].set(-R @ u_p)
            c = c.at[-1].set(lambda_linear)
            return c

        return self._create_updated_instance(Q=Q_func, c=c_func)

    def assign_control_bounds(self, low, high) -> 'ParametricFlowSafeControl':
        """
        Assign control input bounds.

        Args:
            low: Lower bounds for control inputs (list or tuple)
            high: Upper bounds for control inputs (list or tuple)

        Returns:
            New ParametricFlowSafeControl with bounds assigned
        """
        assert len(low) == len(high), 'low and high should have the same length'
        assert len(low) == self._action_dim, 'bounds length should match action dimension'

        # Convert to tuples if needed
        low_tuple = tuple(low) if not isinstance(low, tuple) else low
        high_tuple = tuple(high) if not isinstance(high, tuple) else high

        return self._create_updated_instance(
            control_low=low_tuple,
            control_high=high_tuple
        )

    def _create_updated_instance(self, **kwargs):
        """Create new instance with updated fields."""
        defaults = {
            'action_dim': self._action_dim,
            'alpha': self._alpha,
            'params': dict(self._params) if self._params else None,
            'dynamics': self._dynamics,
            'barrier': self._barrier,
            'Q': self._Q,
            'c': self._c,
            'control_low': self._control_low if self._has_control_bounds else None,
            'control_high': self._control_high if self._has_control_bounds else None,
            'slacked': self._slacked,
            'slack_gain': self._slack_gain,
            'cost_functional': self._cost_functional,
            'flow_barrier': self._flow_barrier,
            'alpha_trajectory': self.alpha_trajectory,
            'alpha_backup': self.alpha_backup,
            'alpha_action': self.alpha_action,
            'alpha_time_shift': self.alpha_time_shift,
            'aug_action_dim': self._aug_action_dim,
            'theta_flat_dim': self._theta_flat_dim
        }
        defaults.update(kwargs)
        return self.__class__(**defaults)

    def optimal_control(self, x: jnp.ndarray, theta: jnp.ndarray = None, gamma: jnp.ndarray = None, state = None) -> tuple:
        """
        Compute optimal augmented control v_aug = [u, ω, z] for single state.

        Args:
            x: Physical state vector (state_dim,)
            theta: Control parameters (action_dim, num_params), optional
            gamma: Time shift scalar, optional
            state: Controller state for warm-starting QP solver

        Returns:
            Tuple (v_aug, new_state)
        """
        if theta is None or gamma is None:
            theta_default, gamma_default = self._flow_barrier._get_default_parameters()
            theta = theta_default if theta is None else theta
            gamma = gamma_default if gamma is None else gamma

        Q, c, G, h = self._compute_qp_data(x, theta, gamma)
        solver = _QP_SOLVERS[self._params.get('qp_mode', 'primal')]
        v_aug, new_state = solver(jnp.diag(Q), c, G, h, state)

        return v_aug, new_state

    def optimal_control_with_info(self, x: jnp.ndarray, theta: jnp.ndarray = None, gamma: jnp.ndarray = None, state = None) -> tuple:
        if theta is None or gamma is None:
            theta_default, gamma_default = self._flow_barrier._get_default_parameters()
            theta = theta_default if theta is None else theta
            gamma = gamma_default if gamma is None else gamma

        Q, c, G, h = self._compute_qp_data(x, theta, gamma)
        solver = _QP_SOLVERS[self._params.get('qp_mode', 'primal')]
        v_aug, new_state = solver(jnp.diag(Q), c, G, h, state)

        constraint_at_u = jnp.dot(G, v_aug) - h
        slack_vars = jnp.zeros(1)
        u_desired = self._flow_barrier._parametric_control(gamma, theta)
        info = QPInfo(slack_vars=slack_vars, constraint_at_u=constraint_at_u, u_desired=u_desired)
        return v_aug, new_state, info

    @jax.jit
    def _compute_qp_data(self, x, theta, gamma):
        """
        JIT-compiled fused computation of all QP matrices.
        Single ODE solve for both cost gradient and barrier Jacobians.
        """
        s = self._flow_barrier._create_augmented_state(x, theta, gamma)

        flow_barrier = self._flow_barrier
        cost_functional = self._cost_functional

        def combined(s_inner):
            x_i, theta_i, gamma_i = flow_barrier._extract_parameters_from_state(s_inner)
            trajectory, dense_func = flow_barrier.compute_trajectory(x_i, theta_i, gamma_i)

            # Cost on shared trajectory
            J = cost_functional(trajectory)

            # Trajectory + backup barriers on shared trajectory
            h_traj_backup = flow_barrier._evaluate_traj_backup_on_trajectory(
                trajectory, dense_func, theta_i, gamma_i)

            # Other barriers (action, time_shift) — no trajectory needed
            h_other_list = [jnp.atleast_1d(func(s_inner))
                            for func in flow_barrier._hocbf_funcs[1:]]
            h_other = jnp.concatenate(h_other_list)

            return jnp.concatenate([jnp.atleast_1d(J), h_traj_backup, h_other])

        def combined_with_aux(s_inner):
            vals = combined(s_inner)
            return vals, vals

        jac, vals = jax.jacrev(combined_with_aux, has_aux=True)(s)

        grad_J = jac[0]
        jac_h = jac[1:]
        h_vals = vals[1:]

        f_s = self._dynamics.f(s)
        g_s = self._dynamics.g(s)

        # Q from precomputed matrix, c from static part + cost gradient
        Q = self._Q(x, theta, gamma)
        c = self._c(x, theta, gamma) + grad_J @ g_s

        # Constraints: -Lg_h v <= Lf_h + alpha(h)
        Lf_h = jac_h @ f_s
        Lg_h = jac_h @ g_s

        h_safety = self._apply_alpha_functions(h_vals, Lf_h)
        G_safety = -Lg_h

        if self._has_control_bounds:
            G_bounds, h_bounds = self._extend_control_bounds(x)
            G = jnp.vstack([G_safety, G_bounds])
            h = jnp.concatenate([h_safety, h_bounds])
        else:
            G, h = G_safety, h_safety

        return Q, c, G, h

    def _apply_alpha_functions(self, h_vals, Lf_h):
        """Apply appropriate alpha function to each barrier based on type."""
        num_barriers = h_vals.shape[0]
        h_safety = jnp.zeros(num_barriers)
        barrier_idx = 0

        # Trajectory barriers
        if self.alpha_trajectory is not None and barrier_idx < num_barriers:
            if not getattr(self._flow_barrier, 'compose_state_barriers', True):
                target_points = int(self._flow_barrier.horizon / self._flow_barrier.time_steps)
                num_trajectory_only = target_points - 1
                for _ in range(num_trajectory_only):
                    if barrier_idx < num_barriers:
                        h_safety = h_safety.at[barrier_idx].set(
                            Lf_h[barrier_idx] + self.alpha_trajectory(h_vals[barrier_idx]))
                        barrier_idx += 1
            else:
                h_safety = h_safety.at[barrier_idx].set(
                    Lf_h[barrier_idx] + self.alpha_trajectory(h_vals[barrier_idx]))
                barrier_idx += 1

        # Backup barrier
        if self.alpha_backup is not None and barrier_idx < num_barriers:
            h_safety = h_safety.at[barrier_idx].set(
                Lf_h[barrier_idx] + self.alpha_backup(h_vals[barrier_idx]))
            barrier_idx += 1

        # Action barriers
        if self.alpha_action is not None:
            if not getattr(self._flow_barrier, 'compose_action_barriers', True):
                num_action_barriers = self._theta_flat_dim * 2
            else:
                num_action_barriers = 1
            for _ in range(num_action_barriers):
                if barrier_idx < num_barriers:
                    h_safety = h_safety.at[barrier_idx].set(
                        Lf_h[barrier_idx] + self.alpha_action(h_vals[barrier_idx]))
                    barrier_idx += 1

        # Time shift barrier
        if self.alpha_time_shift is not None and barrier_idx < num_barriers:
            h_safety = h_safety.at[barrier_idx].set(
                Lf_h[barrier_idx] + self.alpha_time_shift(h_vals[barrier_idx]))

        return h_safety


    @jax.jit
    def _make_eq_const(self, x: jnp.ndarray, Q_matrix: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        JIT-compiled override of parent class method for equality constraints.

        Args:
            x: State (state_dim,)
            Q_matrix: Quadratic cost matrix

        Returns:
            Tuple (A, b) for equality constraints Av = b
        """
        return super()._make_eq_const(x, Q_matrix)

    def _extend_control_bounds(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Extend control bounds to augmented action space for single state.
        Bounds apply to physical control u and time shift rate z.

        Args:
            x: State (state_dim,)

        Returns:
            Tuple (G, h) for control bound constraints
        """
        # Create bounds for physical control u and time shift rate z
        # Total constraints: 2*action_dim (for u) + 1 (for z upper bound)
        num_constraints = 2 * self._action_dim + 1
        G_extended = jnp.zeros((num_constraints, self._aug_action_dim))

        # Lower bounds on u: -u <= -u_low
        G_extended = G_extended.at[:self._action_dim, :self._action_dim].set(-jnp.eye(self._action_dim))
        h_low = -jnp.array(self._control_low)

        # Upper bounds on u: u <= u_high
        G_extended = G_extended.at[self._action_dim:2*self._action_dim, :self._action_dim].set(jnp.eye(self._action_dim))
        h_high = jnp.array(self._control_high)

        # Upper bound on z: z <= 1
        G_extended = G_extended.at[2*self._action_dim, -1].set(1.0)
        h_z = jnp.array([1.0])

        h_extended = jnp.concatenate([h_low, h_high, h_z])

        return G_extended, h_extended

    def get_parametric_control_value(
            self,
            theta: jnp.ndarray,
            gamma: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Get current parametric control value u_p(γ, θ). Batch with jax.vmap.

        Args:
            theta: Control parameters (action_dim, num_params)
            gamma: Time shift scalar

        Returns:
            Parametric control (action_dim,)
        """
        return self._flow_barrier._parametric_control(gamma, theta)

    def get_init_state(self):
        """Get initial controller state for warm-starting OSQP.

        Runs one cold-start solve with default parameters to get the
        KKTSolution pytree with correct shapes.
        """
        theta, gamma = self._flow_barrier._get_default_parameters()
        x_dummy = jnp.zeros(self._flow_barrier.original_dynamics.state_dim)
        _, init_state = self.optimal_control(x_dummy, theta, gamma, state=None)
        return init_state

    def _validate_setup(self):
        """Validate that all required components are assigned."""
        if self._flow_barrier is None:
            raise ValueError("FlowBarrier must be assigned using assign_state_barrier()")
        if self._aug_action_dim == 0:
            raise ValueError("Augmented dimensions not computed. Ensure FlowBarrier is properly initialized.")

    # === Trajectory Generation Methods ===

    def get_flow_safe_trajs(
            self,
            x0: jnp.ndarray,
            theta0: Optional[jnp.ndarray] = None,
            gamma0: Optional[jnp.ndarray] = None,
            timestep: float = 0.01,
            sim_time: float = 2.0,
            method: str = 'tsit5'
    ) -> jnp.ndarray:
        """Generate safe trajectories using parametric flow control."""
        self._validate_setup()

        # Single-trajectory (batch with jax.vmap at the caller)
        if theta0 is None or gamma0 is None:
            theta_default, gamma_default = self._flow_barrier._get_default_parameters()
            theta0 = theta_default if theta0 is None else theta0
            gamma0 = gamma_default if gamma0 is None else gamma0

        s0 = self._flow_barrier._create_augmented_state(x0, theta0, gamma0)

        def augmented_control_func(s_current: jnp.ndarray) -> jnp.ndarray:
            """Control function for SINGLE augmented state."""
            current_x, current_theta, current_gamma = \
                self._flow_barrier._extract_parameters_from_state(s_current)

            v_aug, _ = self.optimal_control(current_x, current_theta, current_gamma)
            return v_aug

        # Use non-vmap version since CVXOPT is not JAX-compatible
        return get_trajs_from_state_action_func(
            x0=s0,
            dynamics=self._dynamics,
            action_func=augmented_control_func,  # Single-state function
            timestep=timestep,
            sim_time=sim_time,
            method=method
        )

    def get_flow_safe_trajs_zoh(
            self,
            x0: jnp.ndarray,
            theta0: Optional[jnp.ndarray] = None,
            gamma0: Optional[jnp.ndarray] = None,
            timestep: float = 0.01,
            sim_time: float = 2.0,
            intermediate_steps: int = 2,
            method: str = 'tsit5',
            use_disturbed: bool = False
    ) -> jnp.ndarray:
        """
        Generate safe trajectories using Zero-Order Hold (ZOH) parametric flow control.

        Uses ZOH where control is computed less frequently and held constant between updates.
        This is more realistic for digital control systems with discrete sampling.

        Args:
            x0: Initial state (state_dim,) or (batch, state_dim)
            theta0: Initial parameters (action_dim, num_params) or (batch, action_dim, num_params)
            gamma0: Initial time shift scalar or (batch,)
            timestep: Integration timestep
            sim_time: Total simulation time
            intermediate_steps: Number of integration steps between control updates
            method: Integration method
            use_disturbed: If True, use disturbed_rhs for closed-loop simulation

        Returns:
            Augmented trajectory (time_steps, batch, aug_state_dim)
        """
        self._validate_setup()

        # Single-trajectory (batch with jax.vmap at the caller)
        if theta0 is None or gamma0 is None:
            theta_default, gamma_default = self._flow_barrier._get_default_parameters()
            theta0 = theta_default if theta0 is None else theta0
            gamma0 = gamma_default if gamma0 is None else gamma0

        s0 = self._flow_barrier._create_augmented_state(x0, theta0, gamma0)

        def augmented_control_func_zoh(s_current: jnp.ndarray) -> jnp.ndarray:
            """
            ZOH control function for SINGLE augmented state.

            This gets called at discrete intervals (every intermediate_steps * timestep)
            and the returned control is held constant until the next update.

            Args:
                s_current: Current augmented state (aug_state_dim,)

            Returns:
                Augmented control (aug_action_dim,) to be held constant
            """
            current_x, current_theta, current_gamma = \
                self._flow_barrier._extract_parameters_from_state(s_current)

            v_aug, _ = self.optimal_control(current_x, current_theta, current_gamma)
            return v_aug

        # Use non-vmap ZOH version since CVXOPT is not JAX-compatible
        return get_trajs_from_state_action_func_zoh(
            x0=s0,
            dynamics=self._dynamics,  # Use AUGMENTED dynamics
            action_func=augmented_control_func_zoh,
            timestep=timestep,
            sim_time=sim_time,
            intermediate_steps=intermediate_steps,
            method=method,
            use_disturbed=use_disturbed
        )

    def get_flow_safe_trajs_action_zoh(
            self,
            x0: jnp.ndarray,
            theta0: Optional[jnp.ndarray] = None,
            gamma0: Optional[jnp.ndarray] = None,
            timestep: float = 0.01,
            sim_time: float = 2.0,
            intermediate_steps: int = 2,
            method: str = 'tsit5',
            use_disturbed: bool = False
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Generate safe trajectories using ZOH parametric flow control, returning trajectories AND actions.

        This method avoids recomputing actions by storing them during trajectory generation.

        Args:
            x0: Initial state (state_dim,) or (batch, state_dim)
            theta0: Initial parameters (action_dim, num_params) or (batch, action_dim, num_params)
            gamma0: Initial time shift scalar or (batch,)
            timestep: Integration timestep
            sim_time: Total simulation time
            intermediate_steps: Number of integration steps between control updates
            method: Integration method

        Returns:
            Tuple of:
            - Augmented trajectory (time_steps, batch, aug_state_dim)
            - Actions array (time_steps-1, batch, aug_action_dim) - ZOH controls at each timestep
        """
        self._validate_setup()

        # Single-trajectory (batch with jax.vmap at the caller)
        if theta0 is None or gamma0 is None:
            theta_default, gamma_default = self._flow_barrier._get_default_parameters()
            theta0 = theta_default if theta0 is None else theta0
            gamma0 = gamma_default if gamma0 is None else gamma0

        s0 = self._flow_barrier._create_augmented_state(x0, theta0, gamma0)

        num_steps = int(sim_time / timestep) + 1
        solver = get_solver(method)
        adjoint = diffrax.RecursiveCheckpointAdjoint()
        rhs_func = self._dynamics.disturbed_rhs if use_disturbed else self._dynamics.rhs

        def step_forward(carry, _):
            current_state, ctrl_state = carry

            current_x, current_theta, current_gamma = \
                self._flow_barrier._extract_parameters_from_state(current_state)
            v_aug, new_ctrl_state = self.optimal_control(
                current_x, current_theta, current_gamma, state=ctrl_state)

            def ode_func(t, y, args):
                return rhs_func(y, args)

            term = diffrax.ODETerm(ode_func)
            solution = diffrax.diffeqsolve(
                terms=term,
                solver=solver,
                t0=0.0,
                t1=timestep,
                dt0=timestep / intermediate_steps,
                y0=current_state,
                args=v_aug,
                adjoint=adjoint,
                saveat=diffrax.SaveAt(t1=True),
                max_steps=intermediate_steps * 5,
            )
            next_state = solution.ys[0]

            return (next_state, new_ctrl_state), (next_state, v_aug)

        init_ctrl_state = self.get_init_state()
        _, (states_seq, actions_seq) = jax.lax.scan(
            step_forward, (s0, init_ctrl_state), jnp.arange(num_steps - 1))
        trajs = jnp.concatenate([jnp.expand_dims(s0, 0), states_seq], axis=0)

        return trajs, actions_seq

