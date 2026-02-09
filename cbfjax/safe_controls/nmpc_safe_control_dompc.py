"""
NMPC Safe Control with barrier constraints using do-mpc (CasADi + IPOPT) backend.

This module provides safe NMPC controllers that use do-mpc with IPOPT,
as an alternative to the acados backend. The API is identical to the acados classes.

Classes:
    DoMPCNMPCSafeControl: NMPC + barriers with EXTERNAL cost
    QuadraticDoMPCNMPCSafeControl: NMPC + barriers with quadratic cost (Q, R)
"""

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
from typing import Callable, Optional, Any

import casadi as ca
from ..utils.jax2casadi import convert

from ..controls.nmpc_control_dompc import DoMPCNMPCControl, QuadraticDoMPCNMPCControl
from ..controls.base_control import QuadraticCostMixin
from .base_safe_control import BaseSafeControl, DummyBarrier


class DoMPCNMPCSafeControl(DoMPCNMPCControl, BaseSafeControl):
    """
    NMPC Safe Control with barrier constraints using do-mpc/IPOPT.

    Drop-in replacement for NMPCSafeControl using do-mpc backend.

    Barrier constraints are added as nonlinear constraints in do-mpc:
    - Path barriers: soft constraints via set_nl_cons with L2 penalty
    - Terminal barriers: big-M + TVP approach (is_terminal parameter)

    Additional params:
        params = {
            ...  # inherited from DoMPCNMPCControl
            'slacked': False,
            'slack_gain_l2': 1000.0,
            'slacked_e': False,
            'slack_gain_l2_e': 1000.0,
        }
    """

    def __init__(self, **kwargs):
        # Add default slack params
        params = kwargs.get('params', None)
        safe_params = {
            'slacked': False,
            'slack_gain_l2': 1000.0,
            'slacked_e': False,
            'slack_gain_l2_e': 1000.0,
        }
        if params is not None:
            safe_params.update(params)
        kwargs['params'] = safe_params

        super().__init__(**kwargs)

    @classmethod
    def create_empty(cls, action_dim: int, params: Optional[dict] = None) -> 'DoMPCNMPCSafeControl':
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
            'cost_running': self._cost_running,
            'cost_terminal': self._cost_terminal,
            'barrier': self._barrier,
            'terminal_barrier': self._terminal_barrier,
        }
        defaults.update(kwargs)
        return self.__class__(**defaults)

    def _create_dompc_model(self, dynamics_ca: ca.Function):
        """Create do-mpc model with TVP for terminal barrier big-M approach."""
        import do_mpc

        nx = self._dynamics.state_dim
        nu = self._action_dim

        model = do_mpc.model.Model('continuous')

        x = model.set_variable(var_type='_x', var_name='x', shape=(nx, 1))
        u = model.set_variable(var_type='_u', var_name='u', shape=(nu, 1))

        # Add TVP for terminal barrier if needed
        if self.has_terminal_barrier:
            model.set_variable(var_type='_tvp', var_name='is_terminal', shape=(1, 1))

        # Set dynamics
        x_flat = ca.vertcat(*[x[i] for i in range(nx)])
        u_flat = ca.vertcat(*[u[i] for i in range(nu)])
        dx = dynamics_ca(x_flat, u_flat)
        dx_col = ca.reshape(dx, nx, 1)
        model.set_rhs('x', dx_col)

        model.setup()
        return model

    def _setup_constraints(self, mpc, model, x0: np.ndarray):
        """Setup box + barrier constraints in do-mpc."""
        # Box constraints from parent
        super()._setup_constraints(mpc, model, x0)

        nx = self._dynamics.state_dim
        x_sym = ca.vertcat(*[model.x['x', i] for i in range(nx)])

        # Add path barrier constraints
        if self.has_barrier:
            nh = self._barrier.num_constraints
            slacked = self._params.get('slacked', False)
            slack_l2 = self._params.get('slack_gain_l2', 1000.0)

            print(f"Converting {nh} JAX path barrier(s) to CasADi...")
            for i, hocbf_func in enumerate(self._barrier._hocbf_funcs):
                barrier_ca = convert(
                    hocbf_func,
                    [('x', (nx,))],
                    name=f'barrier_{i}',
                    validate=True,
                    tolerance=1e-6
                )
                # h(x) >= 0  =>  -h(x) <= 0
                h_expr = -barrier_ca(x_sym)

                if slacked:
                    mpc.set_nl_cons(
                        f'barrier_{i}', h_expr,
                        ub=0.0,
                        soft_constraint=True,
                        penalty_term_cons=slack_l2,
                    )
                    print(f"  Path barrier {i}: soft constraint (L2={slack_l2})")
                else:
                    mpc.set_nl_cons(f'barrier_{i}', h_expr, ub=0.0)
                    print(f"  Path barrier {i}: hard constraint")

        # Add terminal barrier constraints via big-M + TVP
        if self.has_terminal_barrier:
            nh_e = self._terminal_barrier.num_constraints
            slacked_e = self._params.get('slacked_e', False)
            slack_l2_e = self._params.get('slack_gain_l2_e', 1000.0)
            big_M = 1e6

            is_terminal = model.tvp['is_terminal']

            print(f"Converting {nh_e} JAX terminal barrier(s) to CasADi...")
            for i, hocbf_func in enumerate(self._terminal_barrier._hocbf_funcs):
                barrier_ca = convert(
                    hocbf_func,
                    [('x', (nx,))],
                    name=f'barrier_e_{i}',
                    validate=True,
                    tolerance=1e-6
                )
                # Terminal: -h(x) - M*(1 - is_terminal) <= 0
                # When is_terminal=0: -h(x) - M <= 0  (always satisfied)
                # When is_terminal=1: -h(x) <= 0  (enforces h(x) >= 0)
                h_expr = -barrier_ca(x_sym) - big_M * (1 - is_terminal)

                if slacked_e:
                    mpc.set_nl_cons(
                        f'barrier_e_{i}', h_expr,
                        ub=0.0,
                        soft_constraint=True,
                        penalty_term_cons=slack_l2_e,
                    )
                    print(f"  Terminal barrier {i}: soft constraint (L2={slack_l2_e})")
                else:
                    mpc.set_nl_cons(f'barrier_e_{i}', h_expr, ub=0.0)
                    print(f"  Terminal barrier {i}: hard constraint")

    def make(self, x0: Optional[np.ndarray] = None) -> 'DoMPCNMPCSafeControl':
        """Build the safe NMPC controller using do-mpc."""
        import do_mpc

        assert self.has_barrier, "Barrier must be assigned before make() for DoMPCNMPCSafeControl"
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

        print("Setting up do-mpc MPC with barriers...")
        mpc = do_mpc.controller.MPC(dompc_model)

        self._setup_solver_options(mpc)
        self._setup_cost(mpc, dompc_model)
        self._setup_constraints(mpc, dompc_model, x0)

        # Setup TVP function for terminal barrier
        if self.has_terminal_barrier:
            N = self.N_horizon
            tvp_template = mpc.get_tvp_template()

            def tvp_fun(t_now):
                for k in range(N + 1):
                    tvp_template['_tvp', k, 'is_terminal'] = 1.0 if k == N else 0.0
                return tvp_template

            mpc.set_tvp_fun(tvp_fun)

        mpc.setup()

        mpc.x0 = x0.reshape(-1, 1)
        mpc.set_initial_guess()

        object.__setattr__(self, '_mpc', mpc)
        object.__setattr__(self, '_solver', mpc)
        object.__setattr__(self, '_is_built', True)
        print("NMPC Safe Control (do-mpc/IPOPT) ready!")

        return self

    # ==========================================
    # Barrier Evaluation Methods (inherited from concept, reimplemented)
    # ==========================================

    def get_barrier_along_trajectory(self, x: jnp.ndarray) -> jnp.ndarray:
        """Evaluate barriers along predicted trajectory, return min over time."""
        assert self._is_built, "Must call make() before evaluating barriers"
        assert self.has_barrier, "No barrier assigned"

        x_traj, _ = self.get_predicted_trajectory(x)
        barrier_values = self._barrier.hocbf(jnp.array(x_traj))
        return jnp.min(barrier_values, axis=0)

    def get_barrier_values_full(self, x: jnp.ndarray) -> jnp.ndarray:
        """Evaluate all barriers at all time steps along predicted trajectory."""
        assert self._is_built, "Must call make() before evaluating barriers"
        assert self.has_barrier, "No barrier assigned"

        x_traj, _ = self.get_predicted_trajectory(x)
        return self._barrier.hocbf(jnp.array(x_traj))


class QuadraticDoMPCNMPCSafeControl(QuadraticCostMixin, DoMPCNMPCSafeControl):
    """
    NMPC Safe Control with quadratic cost and barrier constraints using do-mpc/IPOPT.

    Drop-in replacement for QuadraticNMPCSafeControl using do-mpc backend.

    Combines:
    - QuadraticCostMixin: Q, R cost matrices with TVP reference
    - DoMPCNMPCSafeControl: barrier constraints + do-mpc backend
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
    def create_empty(cls, action_dim: int, params: Optional[dict] = None) -> 'QuadraticDoMPCNMPCSafeControl':
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
            'terminal_barrier': self._terminal_barrier,
        }
        defaults.update(kwargs)
        return self.__class__(**defaults)

    def _create_dompc_model(self, dynamics_ca: ca.Function):
        """Create do-mpc model with TVP for both reference and terminal barrier."""
        import do_mpc

        nx = self._dynamics.state_dim
        nu = self._action_dim

        model = do_mpc.model.Model('continuous')

        x = model.set_variable(var_type='_x', var_name='x', shape=(nx, 1))
        u = model.set_variable(var_type='_u', var_name='u', shape=(nu, 1))

        # TVP for reference tracking
        model.set_variable(var_type='_tvp', var_name='x_ref', shape=(nx, 1))

        # TVP for terminal barrier
        if self.has_terminal_barrier:
            model.set_variable(var_type='_tvp', var_name='is_terminal', shape=(1, 1))

        # Set dynamics
        x_flat = ca.vertcat(*[x[i] for i in range(nx)])
        u_flat = ca.vertcat(*[u[i] for i in range(nu)])
        dx = dynamics_ca(x_flat, u_flat)
        dx_col = ca.reshape(dx, nx, 1)
        model.set_rhs('x', dx_col)

        model.setup()
        return model

    def _setup_cost(self, mpc, model):
        """Setup quadratic cost with TVP reference."""
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

        lterm = ca.mtimes([x_err.T, Q, x_err]) + ca.mtimes([u_sym.T, R, u_sym])
        mterm = ca.mtimes([x_err.T, Q_e, x_err])

        mpc.set_objective(mterm=mterm, lterm=lterm)

    def make(self, x0: Optional[np.ndarray] = None) -> 'QuadraticDoMPCNMPCSafeControl':
        """Build the quadratic safe NMPC controller using do-mpc."""
        import do_mpc

        assert self._Q is not None and self._R is not None, \
            "Cost matrices must be assigned. Use assign_cost_matrices()."
        assert self.has_barrier, "Barrier must be assigned before make()"
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

        print("Setting up do-mpc MPC with barriers...")
        mpc = do_mpc.controller.MPC(dompc_model)

        self._setup_solver_options(mpc)
        self._setup_cost(mpc, dompc_model)
        # Call DoMPCNMPCSafeControl's _setup_constraints (includes barriers)
        DoMPCNMPCSafeControl._setup_constraints(self, mpc, dompc_model, x0)

        # Setup TVP function combining x_ref and is_terminal
        N = self.N_horizon
        tvp_template = mpc.get_tvp_template()
        has_terminal = self.has_terminal_barrier
        controller_self = self

        def tvp_fun(t_now):
            for k in range(N + 1):
                tvp_template['_tvp', k, 'x_ref'] = controller_self._current_x_ref
                if has_terminal:
                    tvp_template['_tvp', k, 'is_terminal'] = 1.0 if k == N else 0.0
            return tvp_template

        mpc.set_tvp_fun(tvp_fun)

        mpc.setup()

        mpc.x0 = x0.reshape(-1, 1)
        mpc.set_initial_guess()

        object.__setattr__(self, '_mpc', mpc)
        object.__setattr__(self, '_solver', mpc)
        object.__setattr__(self, '_is_built', True)
        print("NMPC Safe Control (do-mpc/IPOPT, quadratic cost) ready!")

        return self

    def update_reference(self, x_ref: np.ndarray, u_ref: Optional[np.ndarray] = None):
        """
        Update the reference trajectory online.

        Args:
            x_ref: Reference state (nx,) or trajectory (N+1, nx).
            u_ref: Ignored for do-mpc.
        """
        assert self._is_built, "Must call make() before updating reference"

        if x_ref.ndim > 1:
            x_ref = x_ref[0]

        object.__setattr__(self, '_current_x_ref', np.asarray(x_ref).reshape(-1, 1))
