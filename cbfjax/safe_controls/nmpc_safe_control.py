"""
NMPC Safe Control with barrier constraint support.

Supports both acados and do-mpc/IPOPT backends via params['nlp_solver'].

Classes:
    NMPCSafeControl: NMPC controller with barrier constraints (EXTERNAL cost)
    QuadraticNMPCSafeControl: NMPC with barrier constraints (quadratic cost)
"""

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
from typing import Callable, Optional

import casadi as ca
from ..utils.jax2casadi import convert

from ..controls.nmpc_control import NMPCControl
from ..controls.base_control import QuadraticCostMixin
from .base_safe_control import BaseSafeControl, DummyBarrier


class NMPCSafeControl(NMPCControl, BaseSafeControl):
    """
    NMPC Safe Control with barrier constraint support (EXTERNAL cost).

    Inherits from (cooperative multiple inheritance):
    - NMPCControl: NMPC solving with backend dispatch
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
    # Hook Methods
    # ==========================================

    def _validate_for_make(self):
        """Validate: barrier must be assigned, then parent checks."""
        assert self.has_barrier, "Barrier must be assigned before make() for NMPCSafeControl"
        super()._validate_for_make()

    def _prepare_barrier_config(self) -> Optional[dict]:
        """Convert JAX barrier funcs to CasADi and return barrier config dict."""
        nx = self._dynamics.state_dim
        config = {}

        # Path barriers
        if self.has_barrier:
            nh = self._barrier.num_constraints
            print(f"Converting {nh} JAX path barrier(s) to CasADi...")
            path_ca_funcs = []
            for i, hocbf_func in enumerate(self._barrier._hocbf_funcs):
                barrier_ca = convert(
                    hocbf_func,
                    [('x', (nx,))],
                    name=f'barrier_{i}',
                    validate=True,
                    tolerance=1e-6
                )
                path_ca_funcs.append(barrier_ca)
            config['path_ca_funcs'] = path_ca_funcs
        else:
            config['path_ca_funcs'] = []

        # Terminal barriers
        if self.has_terminal_barrier:
            nh_e = self._terminal_barrier.num_constraints
            print(f"Converting {nh_e} JAX terminal barrier(s) to CasADi...")
            terminal_ca_funcs = []
            for i, hocbf_func in enumerate(self._terminal_barrier._hocbf_funcs):
                barrier_ca = convert(
                    hocbf_func,
                    [('x', (nx,))],
                    name=f'barrier_e_{i}',
                    validate=True,
                    tolerance=1e-6
                )
                terminal_ca_funcs.append(barrier_ca)
            config['terminal_ca_funcs'] = terminal_ca_funcs
        else:
            config['terminal_ca_funcs'] = []

        # Slack settings
        config['slacked'] = self._params.get('slacked', False)
        config['slack_gain_l1'] = self._params.get('slack_gain_l1', 0.0)
        config['slack_gain_l2'] = self._params.get('slack_gain_l2', 1000.0)
        config['slacked_e'] = self._params.get('slacked_e', False)
        config['slack_gain_l1_e'] = self._params.get('slack_gain_l1_e', 0.0)
        config['slack_gain_l2_e'] = self._params.get('slack_gain_l2_e', 1000.0)

        return config

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

        x_traj, _ = self.get_predicted_trajectory(x)
        barrier_values = self._barrier.hocbf(jnp.array(x_traj))
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
    NMPC Safe Control with barrier constraint support (quadratic cost).

    Uses cooperative multiple inheritance:
    - QuadraticCostMixin: quadratic cost (Q, R matrices)
    - NMPCSafeControl: NMPC + barrier + backend dispatch
    """

    _Q: Optional[Callable] = eqx.field(static=True)
    _R: Optional[Callable] = eqx.field(static=True)
    _Q_e: Optional[Callable] = eqx.field(static=True)
    _x_ref: Optional[Callable] = eqx.field(static=True)

    def __init__(self, **kwargs):
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

    def _validate_for_make(self):
        """Check quadratic cost is assigned, then parent (safe + base) checks."""
        assert self._Q is not None and self._R is not None, \
            "Cost matrices must be assigned. Use assign_cost_matrices()."
        # Skip cost_running check — use quadratic cost instead
        assert self.has_barrier, "Barrier must be assigned before make() for NMPCSafeControl"
        assert self.has_dynamics, "Dynamics must be assigned before make()"
        assert self._has_control_bounds, "Control bounds must be assigned before make()"
        if not self._params.get('continuous', True):
            assert self._dynamics._dt is not None, \
                "Discrete dynamics require discretization_dt in dynamics params."
            assert abs(self._dynamics._dt - self.time_steps) < 1e-10, \
                f"Discrete dynamics dt ({self._dynamics._dt}) must match " \
                f"controller time_steps ({self.time_steps})."

    def _prepare_cost_config(self) -> dict:
        """Return quadratic cost config for backend."""
        nx = self._dynamics.state_dim

        Q = np.array(self._Q())
        R = np.array(self._R())
        Q_e = np.array(self._Q_e()) if self._Q_e is not None else Q

        if self._x_ref is not None:
            x_ref = np.array(self._x_ref())
        else:
            x_ref = np.zeros(nx)

        return {
            'type': 'quadratic',
            'Q': Q,
            'R': R,
            'Q_e': Q_e,
            'x_ref': x_ref,
        }

    def update_reference(self, x_ref: np.ndarray, u_ref: Optional[np.ndarray] = None):
        """Update the reference trajectory online."""
        assert self._is_built, "Must call make() before updating reference"
        self._backend.update_reference(x_ref, u_ref)
