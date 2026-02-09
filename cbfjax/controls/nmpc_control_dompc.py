"""
NMPC Control using do-mpc (CasADi + IPOPT) backend.

This module provides NMPC controllers that use do-mpc with IPOPT as the NLP solver,
as an alternative to the acados backend. The API is identical to the acados classes.

Classes:
    DoMPCNMPCControl: Base NMPC controller with EXTERNAL cost (JAX callables)
    QuadraticDoMPCNMPCControl: NMPC with quadratic cost (Q, R matrices)
"""

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import casadi as ca
from typing import Callable, Optional, Any, Tuple

from ..utils.jax2casadi import convert
from .base_control import BaseControl, QuadraticCostMixin
from .control_types import NMPCInfo
from ..dynamics.base_dynamic import DummyDynamics
from .nmpc_control import NMPCControl


class DoMPCNMPCControl(NMPCControl):
    """
    Nonlinear Model Predictive Control using do-mpc with IPOPT.

    Drop-in replacement for NMPCControl that uses do-mpc/IPOPT instead of acados.
    Dynamics and cost are defined in JAX, converted to CasADi via jax2casadi.

    Additional/overridden params compared to NMPCControl:
        params = {
            'horizon': 2.0,
            'time_steps': 0.04,
            'nlp_solver_max_iter': 200,
            'tol': 1e-4,
            'collocation_deg': 3,         # do-mpc collocation degree
            'nlpsol_opts': {},             # extra options passed to IPOPT
            # acados-specific params (qp_solver, hessian_approx, etc.) are ignored
        }
    """

    # do-mpc specific state
    _mpc: Any
    _dompc_model: Any

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._mpc = None
        self._dompc_model = None

    def _create_dompc_model(self, dynamics_ca: ca.Function):
        """Create a do-mpc model from converted CasADi dynamics."""
        import do_mpc

        nx = self._dynamics.state_dim
        nu = self._action_dim

        model_type = 'continuous'
        model = do_mpc.model.Model(model_type)

        # State variables
        x = model.set_variable(var_type='_x', var_name='x', shape=(nx, 1))

        # Control inputs
        u = model.set_variable(var_type='_u', var_name='u', shape=(nu, 1))

        # Set dynamics: dx/dt = f(x, u)
        # dynamics_ca expects flat vectors, do-mpc uses column vectors
        x_flat = ca.reshape(x, nx, 1)
        u_flat = ca.reshape(u, nu, 1)
        dx = dynamics_ca(ca.vertcat(*[x_flat[i] for i in range(nx)]),
                         ca.vertcat(*[u_flat[i] for i in range(nu)]))
        dx_col = ca.reshape(dx, nx, 1)
        model.set_rhs('x', dx_col)

        model.setup()
        return model

    def _setup_cost(self, mpc, model):
        """Setup EXTERNAL cost in do-mpc using JAX-converted functions."""
        nx = self._dynamics.state_dim
        nu = self._action_dim

        assert self._cost_running is not None, \
            "Running cost must be assigned. Use assign_cost_running()."

        cost_running_ca = self._convert_cost_running_to_casadi()

        if self._cost_terminal is not None:
            cost_terminal_ca = self._convert_cost_terminal_to_casadi()
        else:
            cost_terminal_ca = lambda x: cost_running_ca(x, ca.DM.zeros(nu))

        # Build CasADi expressions from model variables
        x_sym = ca.vertcat(*[model.x['x', i] for i in range(nx)])
        u_sym = ca.vertcat(*[model.u['u', i] for i in range(nu)])

        lterm = cost_running_ca(x_sym, u_sym)
        mterm = cost_terminal_ca(x_sym)

        mpc.set_objective(mterm=mterm, lterm=lterm)

    def _setup_constraints(self, mpc, model, x0: np.ndarray):
        """Setup box constraints in do-mpc."""
        nu = self._action_dim

        # Control constraints
        if self._has_control_bounds:
            for i in range(nu):
                mpc.bounds['lower', '_u', 'u', i] = self._control_low[i]
                mpc.bounds['upper', '_u', 'u', i] = self._control_high[i]

        # State constraints
        if self._has_state_bounds:
            for j, idx in enumerate(self._state_bounds_idx):
                mpc.bounds['lower', '_x', 'x', idx] = self._state_low[j]
                mpc.bounds['upper', '_x', 'x', idx] = self._state_high[j]

    def _setup_solver_options(self, mpc):
        """Setup do-mpc/IPOPT solver options from params."""
        N = self.N_horizon
        dt = self.time_steps
        collocation_deg = self._params.get('collocation_deg', 3)

        ipopt_opts = {
            'ipopt.max_iter': self._params.get('nlp_solver_max_iter', 200),
            'ipopt.tol': self._params.get('tol', 1e-4),
            'ipopt.print_level': 0,
            'print_time': 0,
        }
        # Merge user-provided IPOPT options
        user_opts = self._params.get('nlpsol_opts', {})
        ipopt_opts.update(user_opts)

        mpc.set_param(
            n_horizon=N,
            t_step=dt,
            n_robust=0,
            store_full_solution=True,
            nlpsol_opts={'ipopt.max_iter': ipopt_opts['ipopt.max_iter'],
                         'ipopt.tol': ipopt_opts['ipopt.tol'],
                         'ipopt.print_level': ipopt_opts['ipopt.print_level'],
                         'print_time': ipopt_opts['print_time']},
            collocation_deg=collocation_deg,
        )

    def make(self, x0: Optional[np.ndarray] = None) -> 'DoMPCNMPCControl':
        """
        Build the NMPC controller using do-mpc.

        Args:
            x0: Initial state for the OCP

        Returns:
            Self with solver built
        """
        import do_mpc

        assert self.has_dynamics, "Dynamics must be assigned before make()"
        assert self._has_control_bounds, "Control bounds must be assigned before make()"
        assert self._cost_running is not None, \
            "Running cost must be assigned. Use assign_cost_running()."

        if x0 is None:
            x0 = np.zeros(self._dynamics.state_dim)

        print("Converting JAX dynamics to CasADi...")
        dynamics_casadi = self._convert_dynamics_to_casadi()
        object.__setattr__(self, '_dynamics_casadi', dynamics_casadi)

        print("Building do-mpc model...")
        dompc_model = self._create_dompc_model(dynamics_casadi)
        object.__setattr__(self, '_dompc_model', dompc_model)

        print("Setting up do-mpc MPC...")
        mpc = do_mpc.controller.MPC(dompc_model)

        self._setup_solver_options(mpc)
        self._setup_cost(mpc, dompc_model)
        self._setup_constraints(mpc, dompc_model, x0)

        mpc.setup()

        # Set initial state
        mpc.x0 = x0.reshape(-1, 1)
        mpc.set_initial_guess()

        object.__setattr__(self, '_mpc', mpc)
        object.__setattr__(self, '_solver', mpc)  # alias for compatibility
        object.__setattr__(self, '_is_built', True)
        print("NMPC (do-mpc/IPOPT) ready!")

        return self

    # ==========================================
    # Control Methods
    # ==========================================

    def _optimal_control_single(self, x: jnp.ndarray, state=None) -> Tuple[jnp.ndarray, Any]:
        """Compute optimal control using do-mpc."""
        assert self._is_built, "Must call make() before computing control"

        x_np = np.array(x).reshape(-1, 1)
        u_opt = self._mpc.make_step(x_np)
        u_jax = jnp.array(u_opt.flatten())

        return u_jax, state

    def _optimal_control_single_with_info(self, x: jnp.ndarray, state=None) -> Tuple[jnp.ndarray, Any, dict]:
        """Compute optimal control with diagnostic info using do-mpc."""
        assert self._is_built, "Must call make() before computing control"

        x_np = np.array(x).reshape(-1, 1)
        u_opt = self._mpc.make_step(x_np)
        u_jax = jnp.array(u_opt.flatten())

        # Extract trajectory and cost from do-mpc data
        x_traj, u_traj = self._extract_trajectory()
        cost = self._mpc.data['_aux', -1, 'nlp_cost'] if '_aux' in self._mpc.data.dtype.names else 0.0

        info = NMPCInfo(
            status=jnp.array(0),
            cost=jnp.array(float(cost) if np.isscalar(cost) else 0.0),
            x_traj=x_traj,
            u_traj=u_traj,
        )

        return u_jax, state, info

    def _extract_trajectory(self) -> Tuple[np.ndarray, np.ndarray]:
        """Extract predicted trajectory from do-mpc after a solve."""
        N = self.N_horizon
        nx = self._dynamics.state_dim
        nu = self._action_dim

        x_traj = np.zeros((N + 1, nx))
        u_traj = np.zeros((N, nu))

        for k in range(N + 1):
            x_pred = self._mpc.data.prediction(('_x', 'x'), t_ind=-1)
            if x_pred is not None and x_pred.shape[0] >= nx:
                x_traj[k] = x_pred[:nx, k, 0] if k < x_pred.shape[1] else x_pred[:nx, -1, 0]

        for k in range(N):
            u_pred = self._mpc.data.prediction(('_u', 'u'), t_ind=-1)
            if u_pred is not None and u_pred.shape[0] >= nu:
                u_traj[k] = u_pred[:nu, k, 0] if k < u_pred.shape[1] else u_pred[:nu, -1, 0]

        return x_traj, u_traj

    def get_predicted_trajectory(self, x: jnp.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Solve OCP for given state and return predicted trajectory."""
        assert self._is_built, "Must call make() before getting trajectory"

        x_np = np.array(x).reshape(-1, 1)
        self._mpc.make_step(x_np)
        return self._extract_trajectory()

    # ==========================================
    # Warm-start (do-mpc handles internally)
    # ==========================================

    def set_init_guess(self, x_traj=None, u_traj=None) -> None:
        """Set initial guess. do-mpc handles warm-starting internally."""
        assert self._is_built, "Must call make() before setting initial guess"
        # do-mpc manages warm-start via make_step shift
        # We can reset if needed
        if x_traj is not None:
            self._mpc.x0 = np.asarray(x_traj[0]).reshape(-1, 1)
        self._mpc.set_initial_guess()

    def set_init_guess_linear(self, x0: np.ndarray, x_target: np.ndarray) -> None:
        """Set a linear interpolation initial guess."""
        self._mpc.x0 = np.asarray(x0).reshape(-1, 1)
        self._mpc.set_initial_guess()

    def _post_solve(self):
        """No-op: do-mpc handles shift internally in make_step."""
        pass

    def _shift_warm_start(self):
        """No-op: do-mpc handles shift internally in make_step."""
        pass


class QuadraticDoMPCNMPCControl(QuadraticCostMixin, DoMPCNMPCControl):
    """
    NMPC Control with quadratic cost using do-mpc/IPOPT.

    Uses Q, R matrices for cost: (x - x_ref)^T Q (x - x_ref) + u^T R u

    Drop-in replacement for QuadraticNMPCControl using do-mpc backend.
    """

    _Q: Optional[Callable] = eqx.field(static=True)
    _R: Optional[Callable] = eqx.field(static=True)
    _Q_e: Optional[Callable] = eqx.field(static=True)
    _x_ref: Optional[Callable] = eqx.field(static=True)

    # Mutable reference for online updates
    _current_x_ref: Any

    def __init__(self, **kwargs):
        super().__init__(cost_running=None, cost_terminal=None, **kwargs)
        self._current_x_ref = None

    @classmethod
    def create_empty(cls, action_dim: int, params: Optional[dict] = None) -> 'QuadraticDoMPCNMPCControl':
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
        }
        defaults.update(kwargs)
        return self.__class__(**defaults)

    def _create_dompc_model(self, dynamics_ca: ca.Function):
        """Create do-mpc model with TVP for reference tracking."""
        import do_mpc

        nx = self._dynamics.state_dim
        nu = self._action_dim

        model = do_mpc.model.Model('continuous')

        x = model.set_variable(var_type='_x', var_name='x', shape=(nx, 1))
        u = model.set_variable(var_type='_u', var_name='u', shape=(nu, 1))

        # TVP for reference state (updated online)
        x_ref_tvp = model.set_variable(var_type='_tvp', var_name='x_ref', shape=(nx, 1))

        # Set dynamics
        x_flat = ca.vertcat(*[x[i] for i in range(nx)])
        u_flat = ca.vertcat(*[u[i] for i in range(nu)])
        dx = dynamics_ca(x_flat, u_flat)
        dx_col = ca.reshape(dx, nx, 1)
        model.set_rhs('x', dx_col)

        model.setup()
        return model

    def _setup_cost(self, mpc, model):
        """Setup quadratic cost using TVP reference."""
        nx = self._dynamics.state_dim
        nu = self._action_dim

        assert self._Q is not None and self._R is not None, \
            "Cost matrices must be assigned. Use assign_cost_matrices()."

        Q = np.array(self._Q())
        R = np.array(self._R())
        Q_e = np.array(self._Q_e()) if self._Q_e is not None else Q

        x_sym = model.x['x']
        u_sym = model.u['u']
        x_ref_sym = model.tvp['x_ref']

        x_err = x_sym - x_ref_sym

        # Running cost: x_err^T Q x_err + u^T R u
        lterm = ca.mtimes([x_err.T, Q, x_err]) + ca.mtimes([u_sym.T, R, u_sym])

        # Terminal cost: x_err^T Q_e x_err
        mterm = ca.mtimes([x_err.T, Q_e, x_err])

        mpc.set_objective(mterm=mterm, lterm=lterm)

    def make(self, x0: Optional[np.ndarray] = None) -> 'QuadraticDoMPCNMPCControl':
        """Build the quadratic NMPC controller using do-mpc."""
        import do_mpc

        assert self._Q is not None and self._R is not None, \
            "Cost matrices must be assigned. Use assign_cost_matrices()."
        assert self.has_dynamics, "Dynamics must be assigned before make()"
        assert self._has_control_bounds, "Control bounds must be assigned before make()"

        nx = self._dynamics.state_dim

        if x0 is None:
            x0 = np.zeros(nx)

        # Initialize mutable reference
        if self._x_ref is not None:
            x_ref_val = np.array(self._x_ref()).reshape(-1, 1)
        else:
            x_ref_val = np.zeros((nx, 1))
        object.__setattr__(self, '_current_x_ref', x_ref_val)

        print("Converting JAX dynamics to CasADi...")
        dynamics_casadi = self._convert_dynamics_to_casadi()
        object.__setattr__(self, '_dynamics_casadi', dynamics_casadi)

        print("Building do-mpc model...")
        dompc_model = self._create_dompc_model(dynamics_casadi)
        object.__setattr__(self, '_dompc_model', dompc_model)

        print("Setting up do-mpc MPC...")
        mpc = do_mpc.controller.MPC(dompc_model)

        self._setup_solver_options(mpc)
        self._setup_cost(mpc, dompc_model)
        self._setup_constraints(mpc, dompc_model, x0)

        # Setup TVP function for reference
        N = self.N_horizon
        tvp_template = mpc.get_tvp_template()

        # We need a closure over self to access _current_x_ref
        controller_self = self

        def tvp_fun(t_now):
            for k in range(N + 1):
                tvp_template['_tvp', k, 'x_ref'] = controller_self._current_x_ref
            return tvp_template

        mpc.set_tvp_fun(tvp_fun)

        mpc.setup()

        mpc.x0 = x0.reshape(-1, 1)
        mpc.set_initial_guess()

        object.__setattr__(self, '_mpc', mpc)
        object.__setattr__(self, '_solver', mpc)
        object.__setattr__(self, '_is_built', True)
        print("NMPC (do-mpc/IPOPT, quadratic cost) ready!")

        return self

    def update_reference(self, x_ref: np.ndarray, u_ref: Optional[np.ndarray] = None):
        """
        Update the reference trajectory online.

        Args:
            x_ref: Reference state (nx,) or trajectory (N+1, nx).
                   For do-mpc, only a single reference point is supported via TVP.
            u_ref: Ignored for do-mpc (only state reference is tracked).
        """
        assert self._is_built, "Must call make() before updating reference"

        if x_ref.ndim > 1:
            # Use first reference point for constant reference
            x_ref = x_ref[0]

        object.__setattr__(self, '_current_x_ref', np.asarray(x_ref).reshape(-1, 1))
