"""
NMPC Safe Control with barrier constraint support.

This module provides NMPCSafeControl which extends NMPCControl with
nonlinear barrier constraints for safety guarantees.

Classes:
    NMPCSafeControl: NMPC controller with barrier constraints (EXTERNAL cost)
    QuadraticNMPCSafeControl: NMPC with barrier constraints (LINEAR_LS cost)
"""

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
from typing import Callable, Optional

import casadi as ca
from ..utils.jax2casadi import convert
from acados_template import AcadosModel, AcadosOcp

from ..controls.nmpc_control import NMPCControl
from ..controls.base_control import QuadraticCostMixin
from .base_safe_control import BaseSafeControl, DummyBarrier


class NMPCSafeControl(NMPCControl, BaseSafeControl):
    """
    NMPC Safe Control with barrier constraint support (EXTERNAL cost).

    Inherits from (cooperative multiple inheritance):
    - NMPCControl: NMPC solving with acados
    - BaseSafeControl: barrier interface (assign_state_barrier, has_barrier, etc.)

    Additional params for safe control:
        params = {
            ...  # inherited from NMPCControl
            'slacked': False,        # True = soft path constraint, False = hard
            'slack_gain_l1': 0.0,    # L1 penalty weight (linear) for path
            'slack_gain_l2': 1000.0, # L2 penalty weight (quadratic) for path
            'slacked_e': False,        # True = soft terminal constraint
            'slack_gain_l1_e': 0.0,    # L1 penalty for terminal
            'slack_gain_l2_e': 1000.0, # L2 penalty for terminal
        }
    """

    def __init__(self, **kwargs):
        """
        Initialize NMPCSafeControl.

        Args:
            **kwargs: All args passed via cooperative inheritance
                - barrier: Handled by BaseSafeControl
                - terminal_barrier: Handled by BaseSafeControl
                - control_low, control_high, cost_running, etc.: Handled by NMPCControl
                - action_dim, params, dynamics: Handled by BaseControl
        """
        # Add default slack params
        params = kwargs.get('params', None)
        safe_params = {
            'slacked': False,
            'slack_gain_l1': 0.0,
            'slack_gain_l2': 1000.0,
            'slacked_e': False,
            'slack_gain_l1_e': 0.0,
            'slack_gain_l2_e': 1000.0,
        }
        if params is not None:
            safe_params.update(params)
        kwargs['params'] = safe_params

        super().__init__(**kwargs)

    @classmethod
    def create_empty(cls, action_dim: int, params: Optional[dict] = None) -> 'NMPCSafeControl':
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

    # ==========================================
    # Barrier interface inherited from BaseSafeControl:
    # - assign_state_barrier()
    # - barrier property
    # - has_barrier property
    # - _is_dummy_barrier()
    # ==========================================

    # ==========================================
    # Build Methods (Override to add barrier)
    # ==========================================

    def _setup_constraints(self, ocp: AcadosOcp, model: AcadosModel, x0: np.ndarray):
        """Setup constraints in OCP including path and terminal barrier constraints with optional slack."""
        # Call parent to setup control and state bounds
        super()._setup_constraints(ocp, model, x0)

        nx = self._dynamics.state_dim

        # Add path barrier constraint if assigned
        if self.has_barrier:
            nh = self._barrier.num_constraints

            # Convert each HOCBF function to CasADi
            print(f"Converting {nh} JAX path barrier(s) to CasADi...")
            h_exprs = []
            for i, hocbf_func in enumerate(self._barrier._hocbf_funcs):
                barrier_ca = convert(
                    hocbf_func,
                    [('x', (nx,))],
                    name=f'barrier_{i}',
                    validate=True,
                    tolerance=1e-6
                )
                h_exprs.append(barrier_ca(model.x))

            # Stack all barrier constraints: h(x) >= 0
            model.con_h_expr = ca.vertcat(*h_exprs)

            # Constraint bounds: h >= 0
            ocp.constraints.lh = np.zeros(nh)
            ocp.constraints.uh = 1e9 * np.ones(nh)

            # Setup slack if enabled
            if self._params.get('slacked', False):
                ocp.constraints.idxsh = np.arange(nh)

                slack_l1 = self._params.get('slack_gain_l1', 0.0)
                slack_l2 = self._params.get('slack_gain_l2', 1000.0)

                ocp.cost.zl = slack_l1 * np.ones(nh)
                ocp.cost.zu = np.zeros(nh)
                ocp.cost.Zl = slack_l2 * np.ones(nh)
                ocp.cost.Zu = np.zeros(nh)

                print(f"Path barrier slacked with L1={slack_l1}, L2={slack_l2}")

        # Add terminal barrier constraint if assigned
        if self.has_terminal_barrier:
            nh_e = self._terminal_barrier.num_constraints

            print(f"Converting {nh_e} JAX terminal barrier(s) to CasADi...")
            h_e_exprs = []
            for i, hocbf_func in enumerate(self._terminal_barrier._hocbf_funcs):
                barrier_ca = convert(
                    hocbf_func,
                    [('x', (nx,))],
                    name=f'barrier_e_{i}',
                    validate=True,
                    tolerance=1e-6
                )
                h_e_exprs.append(barrier_ca(model.x))

            # Stack terminal barrier constraints: h_e(x_N) >= 0
            model.con_h_expr_e = ca.vertcat(*h_e_exprs)

            # Terminal constraint bounds: h_e >= 0
            ocp.constraints.lh_e = np.zeros(nh_e)
            ocp.constraints.uh_e = 1e9 * np.ones(nh_e)

            # Setup terminal slack if enabled
            if self._params.get('slacked_e', False):
                ocp.constraints.idxsh_e = np.arange(nh_e)

                slack_l1_e = self._params.get('slack_gain_l1_e', 0.0)
                slack_l2_e = self._params.get('slack_gain_l2_e', 1000.0)

                ocp.cost.zl_e = slack_l1_e * np.ones(nh_e)
                ocp.cost.zu_e = np.zeros(nh_e)
                ocp.cost.Zl_e = slack_l2_e * np.ones(nh_e)
                ocp.cost.Zu_e = np.zeros(nh_e)

                print(f"Terminal barrier slacked with L1={slack_l1_e}, L2={slack_l2_e}")

    def make(self, x0: Optional[np.ndarray] = None) -> 'NMPCSafeControl':
        """
        Build the NMPC safe controller.

        Converts JAX functions (including barrier) to CasADi,
        creates acados OCP, and initializes solver.

        Args:
            x0: Initial state for the OCP (required for acados)

        Returns:
            Self with solver built

        Raises:
            AssertionError: If required components are not assigned
        """
        # Additional assertion for safe control
        assert self.has_barrier, "Barrier must be assigned before make() for NMPCSafeControl"

        # Call parent make
        return super().make(x0)

    # ==========================================
    # Barrier Evaluation Methods
    # ==========================================

    def get_barrier_along_trajectory(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Evaluate barriers along the predicted trajectory and return min over time.

        Args:
            x: Initial state vector (state_dim,)

        Returns:
            Minimum barrier values along trajectory with shape (num_barriers,)
        """
        assert self._is_built, "Must call make() before evaluating barriers"
        assert self.has_barrier, "No barrier assigned"

        # Get predicted trajectory for this state
        x_traj, _ = self.get_predicted_trajectory(x)

        # Evaluate barriers along trajectory: (N+1, num_barriers)
        barrier_values = self._barrier.hocbf(jnp.array(x_traj))

        # Min along time axis (axis=0) -> shape (num_barriers,)
        return jnp.min(barrier_values, axis=0)

    def get_barrier_values_full(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Evaluate all barriers at all time steps along the predicted trajectory.

        Args:
            x: Initial state vector (state_dim,)

        Returns:
            Barrier values with shape (N+1, num_barriers)
        """
        assert self._is_built, "Must call make() before evaluating barriers"
        assert self.has_barrier, "No barrier assigned"

        x_traj, _ = self.get_predicted_trajectory(x)
        return self._barrier.hocbf(jnp.array(x_traj))


class QuadraticNMPCSafeControl(QuadraticCostMixin, NMPCSafeControl):
    """
    NMPC Safe Control with barrier constraint support (LINEAR_LS cost).

    Uses cooperative multiple inheritance:
    - QuadraticCostMixin: quadratic cost (Q, R matrices)
    - NMPCSafeControl: NMPC + barrier
    """

    # Cost matrices as Callable (from QuadraticCostMixin)
    _Q: Optional[Callable] = eqx.field(static=True)
    _R: Optional[Callable] = eqx.field(static=True)
    _Q_e: Optional[Callable] = eqx.field(static=True)
    _x_ref: Optional[Callable] = eqx.field(static=True)

    def __init__(self, **kwargs):
        """
        Initialize QuadraticNMPCSafeControl.

        Args:
            **kwargs: All args passed via cooperative inheritance
                - Q, R, Q_e, x_ref: Handled by QuadraticCostMixin
                - barrier: Handled by BaseSafeControl
                - control_low, control_high, etc.: Handled by NMPCControl
                - action_dim, params, dynamics: Handled by BaseControl
        """
        # Initialize via cooperative inheritance (no cost_running/cost_terminal)
        super().__init__(cost_running=None, cost_terminal=None, **kwargs)

    @classmethod
    def create_empty(cls, action_dim: int, params: Optional[dict] = None) -> 'QuadraticNMPCSafeControl':
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

    # assign_cost_matrices, assign_reference from QuadraticCostMixin

    def _setup_cost(self, ocp: AcadosOcp, model: AcadosModel):
        """Setup LINEAR_LS cost function in OCP."""
        nx = self._dynamics.state_dim
        nu = self._action_dim

        assert self._Q is not None and self._R is not None, \
            "Cost matrices must be assigned. Use assign_cost_matrices()."

        # Use LINEAR_LS cost type
        ocp.cost.cost_type = "LINEAR_LS"
        ocp.cost.cost_type_e = "LINEAR_LS"

        # Get matrices from callables
        Q = np.array(self._Q())
        R = np.array(self._R())
        Q_e = np.array(self._Q_e()) if self._Q_e is not None else Q

        # Reference
        if self._x_ref is not None:
            x_ref = np.array(self._x_ref())
        else:
            x_ref = np.zeros(nx)

        # Output dimension: y = [x; u]
        ny = nx + nu

        # Weight matrices
        ocp.cost.W = np.block([
            [Q, np.zeros((nx, nu))],
            [np.zeros((nu, nx)), R]
        ])
        ocp.cost.W_e = Q_e

        # Output selection matrices: y = Vx @ x + Vu @ u
        ocp.cost.Vx = np.vstack([np.eye(nx), np.zeros((nu, nx))])
        ocp.cost.Vu = np.vstack([np.zeros((nx, nu)), np.eye(nu)])
        ocp.cost.Vx_e = np.eye(nx)

        # Reference (y = [x; u] for stage, y_e = x for terminal)
        ocp.cost.yref = np.concatenate([x_ref, np.zeros(nu)])
        ocp.cost.yref_e = x_ref

    def make(self, x0: Optional[np.ndarray] = None) -> 'QuadraticNMPCSafeControl':
        """Build the NMPC safe controller."""
        # Check quadratic cost is assigned
        assert self._Q is not None and self._R is not None, \
            "Cost matrices must be assigned. Use assign_cost_matrices()."
        assert self.has_barrier, "Barrier must be assigned before make() for QuadraticNMPCSafeControl"
        assert self.has_dynamics, "Dynamics must be assigned before make()"
        assert self._has_control_bounds, "Control bounds must be assigned before make()"

        if x0 is None:
            x0 = np.zeros(self._dynamics.state_dim)

        print("Converting JAX dynamics to CasADi...")
        dynamics_casadi = self._convert_dynamics_to_casadi()
        object.__setattr__(self, '_dynamics_casadi', dynamics_casadi)

        print("Building acados OCP...")
        ocp = AcadosOcp()

        model = self._create_acados_model(dynamics_casadi)
        ocp.model = model

        self._setup_cost(ocp, model)
        self._setup_constraints(ocp, model, x0)
        self._setup_solver_options(ocp)

        object.__setattr__(self, '_ocp', ocp)

        print("Creating acados solver...")
        from acados_template import AcadosOcpSolver
        solver = AcadosOcpSolver(ocp)
        object.__setattr__(self, '_solver', solver)

        object.__setattr__(self, '_is_built', True)
        print("NMPC ready!")

        return self

    def update_reference(self, x_ref: np.ndarray, u_ref: Optional[np.ndarray] = None):
        """
        Update the reference trajectory online.

        Args:
            x_ref: Reference state (nx,) or trajectory (N+1, nx)
            u_ref: Reference control (nu,) or trajectory (N, nu), optional
        """
        assert self._is_built, "Must call make() before updating reference"

        nx = self._dynamics.state_dim
        nu = self._action_dim

        # Handle single reference vs trajectory
        if x_ref.ndim == 1:
            x_ref_traj = np.tile(x_ref, (self.N_horizon + 1, 1))
        else:
            x_ref_traj = x_ref

        if u_ref is None:
            u_ref_traj = np.zeros((self.N_horizon, nu))
        elif u_ref.ndim == 1:
            u_ref_traj = np.tile(u_ref, (self.N_horizon, 1))
        else:
            u_ref_traj = u_ref

        # Update references in solver
        for k in range(self.N_horizon):
            yref = np.concatenate([x_ref_traj[k], u_ref_traj[k]])
            self._solver.set(k, "yref", yref)

        # Terminal reference
        self._solver.set(self.N_horizon, "yref", x_ref_traj[self.N_horizon])
