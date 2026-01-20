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
from typing import Callable, Optional, Any

import casadi as ca
from ..utils.jax2casadi import convert
from acados_template import AcadosModel, AcadosOcp

from ..controls.nmpc_control import NMPCControl, QuadraticNMPCControl
from .base_safe_control import DummyBarrier


class NMPCSafeControl(NMPCControl):
    """
    NMPC Safe Control with barrier constraint support (EXTERNAL cost).

    Extends NMPCControl to include nonlinear barrier constraints h(x) >= 0
    converted from JAX via jax2casadi.

    Additional params for safe control:
        params = {
            ...  # inherited from NMPCControl
            'slacked': False,        # True = soft constraint, False = hard
            'slack_gain_l1': 0.0,    # L1 penalty weight (linear)
            'slack_gain_l2': 1000.0, # L2 penalty weight (quadratic)
        }

    Attributes:
        _barrier: Barrier function object
    """

    # Barrier
    _barrier: Any = eqx.field(static=True)

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
        cost_running: Optional[Callable] = None,
        cost_terminal: Optional[Callable] = None,
        barrier=None,
    ):
        """
        Initialize NMPCSafeControl.

        Args:
            action_dim: Dimension of control input
            params: Configuration parameters dictionary. In addition to NMPCControl params:
                - slacked: Whether barrier constraint is soft (default: False)
                - slack_gain_l1: L1 penalty weight for slack (default: 0.0)
                - slack_gain_l2: L2 penalty weight for slack (default: 1000.0)
            dynamics: System dynamics object (AffineInControlDynamics)
            control_low: Lower bounds for control inputs
            control_high: Upper bounds for control inputs
            state_bounds_idx: Indices of bounded states
            state_low: Lower bounds for bounded states
            state_high: Upper bounds for bounded states
            cost_running: Running cost function f(x, u) -> scalar (JAX)
            cost_terminal: Terminal cost function f(x) -> scalar (JAX)
            barrier: Barrier object (e.g., from CBFJAX Map)
        """
        # Add default slack params
        safe_params = {
            'slacked': False,
            'slack_gain_l1': 0.0,
            'slack_gain_l2': 1000.0,
        }
        if params is not None:
            safe_params.update(params)

        super().__init__(
            action_dim=action_dim,
            params=safe_params,
            dynamics=dynamics,
            control_low=control_low,
            control_high=control_high,
            state_bounds_idx=state_bounds_idx,
            state_low=state_low,
            state_high=state_high,
            cost_running=cost_running,
            cost_terminal=cost_terminal,
        )

        self._barrier = barrier if barrier is not None else DummyBarrier()

    @classmethod
    def create_empty(cls, action_dim: int, params: Optional[dict] = None) -> 'NMPCSafeControl':
        """
        Create an empty NMPC safe controller for assignment chain.

        Args:
            action_dim: Dimension of control input
            params: Optional configuration parameters

        Returns:
            Empty NMPCSafeControl instance ready for assignment
        """
        return cls(action_dim=action_dim, params=params)

    def _create_updated_instance(self, **kwargs):
        """
        Create new instance with updated fields.

        Args:
            **kwargs: Fields to update

        Returns:
            New NMPCSafeControl instance with updated fields
        """
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
        }
        defaults.update(kwargs)
        return self.__class__(**defaults)

    # ==========================================
    # Barrier Assignment Methods
    # ==========================================

    def assign_state_barrier(self, barrier) -> 'NMPCSafeControl':
        """
        Assign state barrier object to controller.

        The barrier's min_barrier method will be used as constraint.

        Args:
            barrier: Barrier object (e.g., from CBFJAX Map)

        Returns:
            New NMPCSafeControl instance with assigned barrier
        """
        return self._create_updated_instance(barrier=barrier)

    def _is_dummy_barrier(self, barrier) -> bool:
        """Check if barrier is a dummy object."""
        return isinstance(barrier, DummyBarrier)

    @property
    def barrier(self):
        """Get assigned barrier object."""
        return self._barrier

    @property
    def has_barrier(self) -> bool:
        """Check if real barrier assigned."""
        return not self._is_dummy_barrier(self._barrier)

    # ==========================================
    # Build Methods (Override to add barrier)
    # ==========================================

    def _setup_constraints(self, ocp: AcadosOcp, model: AcadosModel, x0: np.ndarray):
        """Setup constraints in OCP including barrier constraint with optional slack."""
        # Call parent to setup control and state bounds
        super()._setup_constraints(ocp, model, x0)

        # Add barrier constraint if assigned
        if self.has_barrier:
            nx = self._dynamics.state_dim
            nh = self._barrier.num_constraints

            # Convert each HOCBF function to CasADi
            print(f"Converting {nh} JAX barrier(s) to CasADi...")
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

                print(f"Barrier constraint slacked with L1={slack_l1}, L2={slack_l2}")

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


class QuadraticNMPCSafeControl(QuadraticNMPCControl):
    """
    NMPC Safe Control with barrier constraint support (LINEAR_LS cost).

    Extends QuadraticNMPCControl to include nonlinear barrier constraints.

    Attributes:
        _barrier: Barrier function object
    """

    # Barrier
    _barrier: Any = eqx.field(static=True)

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
        Q: Optional[np.ndarray] = None,
        R: Optional[np.ndarray] = None,
        Q_e: Optional[np.ndarray] = None,
        x_ref: Optional[np.ndarray] = None,
        barrier=None,
    ):
        """
        Initialize QuadraticNMPCSafeControl.

        Args:
            action_dim: Dimension of control input
            params: Configuration parameters dictionary. In addition to QuadraticNMPCControl params:
                - slacked: Whether barrier constraint is soft (default: False)
                - slack_gain_l1: L1 penalty weight for slack (default: 0.0)
                - slack_gain_l2: L2 penalty weight for slack (default: 1000.0)
            dynamics: System dynamics object (AffineInControlDynamics)
            control_low: Lower bounds for control inputs
            control_high: Upper bounds for control inputs
            state_bounds_idx: Indices of bounded states
            state_low: Lower bounds for bounded states
            state_high: Upper bounds for bounded states
            Q: State cost matrix (nx, nx)
            R: Control cost matrix (nu, nu)
            Q_e: Terminal state cost matrix (nx, nx)
            x_ref: Reference state (nx,)
            barrier: Barrier object (e.g., from CBFJAX Map)
        """
        # Add default slack params
        safe_params = {
            'slacked': False,
            'slack_gain_l1': 0.0,
            'slack_gain_l2': 1000.0,
        }
        if params is not None:
            safe_params.update(params)

        super().__init__(
            action_dim=action_dim,
            params=safe_params,
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

        self._barrier = barrier if barrier is not None else DummyBarrier()

    @classmethod
    def create_empty(cls, action_dim: int, params: Optional[dict] = None) -> 'QuadraticNMPCSafeControl':
        """Create an empty controller for assignment chain."""
        return cls(action_dim=action_dim, params=params)

    def _create_updated_instance(self, **kwargs):
        """Create new instance with updated fields."""
        Q = kwargs.get('Q', self._Q)
        R = kwargs.get('R', self._R)
        Q_e = kwargs.get('Q_e', self._Q_e)
        x_ref = kwargs.get('x_ref', self._x_ref)

        if Q is not None and isinstance(Q, tuple):
            Q = np.array(Q)
        if R is not None and isinstance(R, tuple):
            R = np.array(R)
        if Q_e is not None and isinstance(Q_e, tuple):
            Q_e = np.array(Q_e)
        if x_ref is not None and isinstance(x_ref, tuple):
            x_ref = np.array(x_ref)

        defaults = {
            'action_dim': self._action_dim,
            'params': dict(self._params) if self._params else None,
            'dynamics': self._dynamics,
            'control_low': list(self._control_low) if self._has_control_bounds else None,
            'control_high': list(self._control_high) if self._has_control_bounds else None,
            'state_bounds_idx': list(self._state_bounds_idx) if self._has_state_bounds else None,
            'state_low': list(self._state_low) if self._has_state_bounds else None,
            'state_high': list(self._state_high) if self._has_state_bounds else None,
            'Q': Q,
            'R': R,
            'Q_e': Q_e,
            'x_ref': x_ref,
            'barrier': self._barrier,
        }
        defaults.update(kwargs)
        return self.__class__(**defaults)

    # ==========================================
    # Barrier Assignment Methods
    # ==========================================

    def assign_state_barrier(self, barrier) -> 'QuadraticNMPCSafeControl':
        """Assign state barrier object to controller."""
        return self._create_updated_instance(barrier=barrier)

    def _is_dummy_barrier(self, barrier) -> bool:
        """Check if barrier is a dummy object."""
        return isinstance(barrier, DummyBarrier)

    @property
    def barrier(self):
        """Get assigned barrier object."""
        return self._barrier

    @property
    def has_barrier(self) -> bool:
        """Check if real barrier assigned."""
        return not self._is_dummy_barrier(self._barrier)

    # ==========================================
    # Build Methods (Override to add barrier)
    # ==========================================

    def _setup_constraints(self, ocp: AcadosOcp, model: AcadosModel, x0: np.ndarray):
        """Setup constraints in OCP including barrier constraint with optional slack."""
        # Call parent to setup control and state bounds
        super()._setup_constraints(ocp, model, x0)

        # Add barrier constraint if assigned
        if self.has_barrier:
            nx = self._dynamics.state_dim
            nh = self._barrier.num_constraints

            # Convert each HOCBF function to CasADi
            print(f"Converting {nh} JAX barrier(s) to CasADi...")
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

                print(f"Barrier constraint slacked with L1={slack_l1}, L2={slack_l2}")

    def make(self, x0: Optional[np.ndarray] = None) -> 'QuadraticNMPCSafeControl':
        """Build the NMPC safe controller."""
        assert self.has_barrier, "Barrier must be assigned before make() for QuadraticNMPCSafeControl"
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
