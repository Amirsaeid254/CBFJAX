"""
BackupSafeControl implementation for JAX.

This module implements Backup Safe Control extending InputConstQPSafeControl,
blending between backup controls and desired controls based on barrier values.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Callable, Optional, Any
from functools import partial
from mpax import create_lp, r2HPDHG, raPDHG
from qpax import solve_qp_primal
from cbfjax.utils import profile_jax


from cbfjax.safe_controls.qp_safe_control import InputConstQPSafeControl, MinIntervQPSafeControl
from cbfjax.safe_controls.base_safe_control import BaseMinIntervSafeControl
from cbfjax.controls.control_types import BackupInfo


class BackupSafeControl(InputConstQPSafeControl):
    """
    Backup Safe Control implementation.

    Extends InputConstQPSafeControl to blend between backup controls and
    QP-based safe controls using a smooth blending factor based on barrier values.
    """

    # BackupBarrier configuration (stored from barrier.cfg)
    barrier_cfg: Any = eqx.field(static=True)

    def __init__(self, barrier_cfg=None, **kwargs):
        """
        Initialize BackupSafeControl with cooperative inheritance.

        Args:
            barrier_cfg: Barrier configuration dictionary
            **kwargs: Passed via cooperative inheritance (control_low, control_high, slacked, slack_gain, alpha, Q, c, barrier, dynamics, action_dim, params)
        """
        super().__init__(**kwargs)
        self.barrier_cfg = barrier_cfg

    def _create_updated_instance(self, **kwargs):
        """
        Create new instance with updated fields.

        Args:
            **kwargs: Fields to update

        Returns:
            New BackupSafeControl instance with updated fields
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
            'slack_gain': self._slack_gain,
            'barrier_cfg': self.barrier_cfg
        }
        defaults.update(kwargs)
        return self.__class__(**defaults)

    def assign_state_barrier(self, barrier):
        """Assign state barrier and store its configuration."""
        return self._create_updated_instance(barrier=barrier, barrier_cfg=barrier.cfg)

    def assign_dynamics(self, dynamics):
        """Assign dynamics."""
        return self._create_updated_instance(dynamics=dynamics)

    def optimal_control(self, x: jnp.ndarray, state=None):
        """
        Compute backup safe optimal control with automatic batch support.

        Args:
            x: State(s) - shape (state_dim,) or (batch, state_dim)
            state: Controller state (optional, uses get_init_state() if None)

        Returns:
            Tuple (u, new_state)
        """
        if state is None:
            state = self.get_init_state()
        x = jnp.atleast_2d(x)

        # Vmap over batch dimension
        u, new_state = jax.vmap(self._optimal_control_single, in_axes=(0, None))(x, state)

        return u, new_state

    def optimal_control_with_info(self, x: jnp.ndarray, state=None):
        """
        Compute backup safe optimal control with diagnostic info.

        Args:
            x: State(s) - shape (state_dim,) or (batch, state_dim)
            state: Controller state (optional, uses get_init_state() if None)

        Returns:
            Tuple (u, new_state, info)
        """
        if state is None:
            state = self.get_init_state()
        x = jnp.atleast_2d(x)

        u, new_state, info = jax.vmap(self._optimal_control_single_with_info, in_axes=(0, None))(x, state)

        return u, new_state, info

    @profile_jax("_safe_optimal_control_single")
    def _optimal_control_single(self, x: jnp.ndarray, state=None):
        """
        Compute backup safe optimal control for SINGLE state.

        Blends between backup control and QP-based safe control using a smooth
        blending factor based on barrier values and feasibility.

        Args:
            x: State vector (state_dim,)
            state: Controller state (unused for backup, passed through)

        Returns:
            Tuple (u, new_state)
        """
        # Get HOCBF and Lie derivatives for single state
        hocbf, Lf_hocbf, Lg_hocbf = self._barrier._get_hocbf_and_lie_derivs_single(x)

        # Compute feasibility factor using LP
        feas_fact = self._get_feasibility_factor(x, Lf_hocbf, Lg_hocbf, hocbf)

        # TODO: Optimize to avoid redundant trajectory computation
        h_star_vals = self._barrier.get_h_stars(x)  # (action_num,)
        action_num = h_star_vals.shape[0]

        # Compute backup control selection
        if action_num > 1:
            # Get backup controls for all policies (action_num, action_dim)
            ub_vals = jnp.stack([policy(x) for policy in self._barrier.backup_policies])

            # Blend backup controls
            ub_blend = self._get_backup_blend(h_star_vals, ub_vals)

            # Select backup control (blend or argmax)
            max_ind = jnp.argmax(h_star_vals)
            max_val = jnp.max(h_star_vals)

            # Use blend if max_val > epsilon, else use selected backup
            ub_selected = ub_vals[max_ind]
            ub_select = jnp.where(max_val > self.barrier_cfg['epsilon'], ub_blend, ub_selected)
        else:
            # Single backup policy
            ub_select = self._barrier.backup_policies[0](x)

        # Compute gamma blending factor
        gamma = jnp.minimum(
            (hocbf - self.barrier_cfg['epsilon']) / self.barrier_cfg['h_scale'],
            feas_fact / self.barrier_cfg['feas_scale']
        )

        # Compute QP-based safe control (always compute, select based on gamma)
        Q_matrix, c_vector = self._make_objective_single(x)

        # CBF constraints - ensure correct shapes for QP
        G_cbf = -jnp.atleast_2d(Lg_hocbf).reshape(1, -1)  # (1, action_dim)
        h_cbf_val = Lf_hocbf + self._alpha(hocbf - self.barrier_cfg['epsilon'])
        h_cbf = jnp.atleast_1d(h_cbf_val.squeeze())  # (1,)

        # Add control bound constraints
        G_low = -jnp.eye(self._action_dim)
        h_low = -jnp.array(self._control_low)
        G_high = jnp.eye(self._action_dim)
        h_high = jnp.array(self._control_high)

        # Combine constraints
        G = jnp.vstack([G_cbf, G_low, G_high])
        h = jnp.concatenate([h_cbf, h_low, h_high])

        # Make equality constraints
        A, b = self._make_eq_const_single(x, Q_matrix)

        # Solve QP
        u_qp = solve_qp_primal(Q_matrix, c_vector, A, b, G, h)

        u_star = jnp.where(gamma >= 0, u_qp, ub_select)

        # Compute beta blending factor
        beta = jnp.where(gamma > 0,
                        jnp.where(gamma >= 1, 1.0, gamma),
                        0.0)

        # Blend backup and safe controls
        u = (1 - beta) * ub_select + beta * u_star

        return u, state

    def _optimal_control_single_with_info(self, x: jnp.ndarray, state=None):
        """
        Compute backup safe optimal control with diagnostic info for SINGLE state.

        Args:
            x: State vector (state_dim,)
            state: Controller state (unused for backup, passed through)

        Returns:
            Tuple (u, new_state, info)
        """
        # Get HOCBF and Lie derivatives for single state
        hocbf, Lf_hocbf, Lg_hocbf = self._barrier._get_hocbf_and_lie_derivs_single(x)

        # Compute feasibility factor using LP
        feas_fact = self._get_feasibility_factor(x, Lf_hocbf, Lg_hocbf, hocbf)

        h_star_vals = self._barrier.get_h_stars(x)
        action_num = h_star_vals.shape[0]

        # Compute backup control selection
        if action_num > 1:
            ub_vals = jnp.stack([policy(x) for policy in self._barrier.backup_policies])
            ub_blend = self._get_backup_blend(h_star_vals, ub_vals)
            max_ind = jnp.argmax(h_star_vals)
            max_val = jnp.max(h_star_vals)
            ub_selected = ub_vals[max_ind]
            ub_select = jnp.where(max_val > self.barrier_cfg['epsilon'], ub_blend, ub_selected)
        else:
            ub_select = self._barrier.backup_policies[0](x)

        # Compute gamma blending factor
        gamma = jnp.minimum(
            (hocbf - self.barrier_cfg['epsilon']) / self.barrier_cfg['h_scale'],
            feas_fact / self.barrier_cfg['feas_scale']
        )

        # Compute QP-based safe control
        Q_matrix, c_vector = self._make_objective_single(x)

        G_cbf = -jnp.atleast_2d(Lg_hocbf).reshape(1, -1)
        h_cbf_val = Lf_hocbf + self._alpha(hocbf - self.barrier_cfg['epsilon'])
        h_cbf = jnp.atleast_1d(h_cbf_val.squeeze())

        G_low = -jnp.eye(self._action_dim)
        h_low = -jnp.array(self._control_low)
        G_high = jnp.eye(self._action_dim)
        h_high = jnp.array(self._control_high)

        G = jnp.vstack([G_cbf, G_low, G_high])
        h = jnp.concatenate([h_cbf, h_low, h_high])

        A, b = self._make_eq_const_single(x, Q_matrix)
        u_qp = solve_qp_primal(Q_matrix, c_vector, A, b, G, h)

        u_star = jnp.where(gamma >= 0, u_qp, ub_select)

        beta = jnp.where(gamma > 0,
                        jnp.where(gamma >= 1, 1.0, gamma),
                        0.0)

        u = (1 - beta) * ub_select + beta * u_star

        constraint_at_u = jnp.dot(G, u) - h

        info = BackupInfo(
            constraint_at_u=constraint_at_u,
            u_star=u_star,
            ub_select=ub_select,
            feas_fact=feas_fact,
            beta=beta,
        )
        return u, state, info

    def _get_backup_blend(self, h_star_vals: jnp.ndarray, ub_vals: jnp.ndarray) -> jnp.ndarray:
        """
        Compute weighted blend of backup controls based on h_star values.

        Args:
            h_star_vals: Barrier values for each backup policy (action_num,)
            ub_vals: Backup controls (action_num, action_dim)

        Returns:
            Blended control (action_dim,)
        """
        # Only use policies with h_star >= epsilon
        valid_mask = h_star_vals >= self.barrier_cfg['epsilon']  # (action_num,)

        # Weights based on h_star - epsilon
        weights = (h_star_vals - self.barrier_cfg['epsilon']) * valid_mask  # (action_num,)

        # Compute weighted sum
        num = jnp.sum(weights[:, None] * ub_vals, axis=0)  # (action_dim,)
        den = jnp.sum(weights) + 1e-8  # Add small epsilon to avoid division by zero

        return num / den

    def _get_feasibility_factor(self, x: jnp.ndarray, Lf_hocbf: jnp.ndarray,
                                       Lg_hocbf: jnp.ndarray, hocbf: jnp.ndarray) -> float:
        """
        Compute feasibility factor using linear programming for single state.

        Args:
            x: State vector (state_dim,)
            Lf_hocbf: Lie derivative wrt f
            Lg_hocbf: Lie derivative wrt g
            hocbf: HOCBF value

        Returns:
            Feasibility factor (scalar)
        """
        # LP objective: minimize -Lg_hocbf @ u
        c = -Lg_hocbf.squeeze()

        # LP format: min c @ u s.t. A @ u == b, G @ u >= h, l <= u <= u_high
        # Lower: I @ u >= control_low  (u >= l)
        # Upper: -I @ u >= -control_high  (-u >= -u_high => u <= u_high)
        G_low = jnp.eye(self._action_dim)
        h_low = jnp.array(self._control_low)
        G_high = -jnp.eye(self._action_dim)
        h_high = -jnp.array(self._control_high)
        A_ub = jnp.vstack([G_low, G_high])
        b_ub = jnp.concatenate([h_low, h_high])

        # No equality constraints
        A_eq = jnp.zeros((0, self._action_dim))
        b_eq = jnp.zeros(0)

        # Unbounded (since constraints already encode bounds)
        l_bound = -jnp.inf * jnp.ones(self._action_dim)
        u_bound = jnp.inf * jnp.ones(self._action_dim)
        lp = create_lp(c, A_eq, b_eq, A_ub, b_ub, l_bound, u_bound, use_sparse_matrix=False)
        solver = raPDHG(eps_abs=1e-4, eps_rel=1e-4, verbose=False)
        result = solver.optimize(lp)
        u_lp = result.primal_solution

        # Feasibility factor = Lf_hocbf + alpha(hocbf - epsilon) + Lg_hocbf @ u_lp
        feas = Lf_hocbf + self._alpha(hocbf - self.barrier_cfg['epsilon']) + jnp.dot(Lg_hocbf.squeeze(), u_lp)
        return feas


class MinIntervBackupSafeControl(BackupSafeControl, BaseMinIntervSafeControl):
    """
    Minimum Intervention Backup Safe Control.

    Combines backup control with minimum intervention QP-based safe control
    using cooperative multiple inheritance.
    """

    def __init__(self, **kwargs):
        """
        Initialize MinIntervBackupSafeControl with cooperative inheritance.

        Args:
            **kwargs: All args passed via cooperative inheritance
                - barrier_cfg: Handled by BackupSafeControl
                - desired_control: Handled by BaseMinIntervSafeControl
                - control_low, control_high: Handled by InputConstQPSafeControl
                - slacked, slack_gain: Handled by QPSafeControl
                - alpha, Q, c, barrier: Handled by BaseCBFSafeControl
                - dynamics, action_dim, params: Handled by BaseControl
        """
        super().__init__(**kwargs)

    def _create_updated_instance(self, **kwargs):
        """
        Create new instance with updated fields.

        Args:
            **kwargs: Fields to update

        Returns:
            New MinIntervBackupSafeControl instance with updated fields
        """
        defaults = {
            'action_dim': self._action_dim,
            'alpha': self._alpha,
            'params': dict(self._params),
            'dynamics': self._dynamics,
            'barrier': self._barrier,
            'desired_control': self._desired_control,
            'desired_control_init_state': self._desired_control_init_state,
            'Q': self._Q,
            'c': self._c,
            'control_low': self._control_low if self._has_control_bounds else None,
            'control_high': self._control_high if self._has_control_bounds else None,
            'slacked': self._slacked,
            'slack_gain': self._slack_gain,
            'barrier_cfg': self.barrier_cfg
        }
        defaults.update(kwargs)
        return self.__class__(**defaults)

    def assign_desired_control(self, desired_control) -> 'MinIntervBackupSafeControl':
        """
        Assign desired control function.

        Accepts either:
        - A controller object with _optimal_control_single and get_init_state methods
        - A plain function f(x) -> u (wrapped to stateful form)
        """
        if hasattr(desired_control, '_optimal_control_single') and hasattr(desired_control, 'get_init_state'):
            ctrl_obj = desired_control
            def stateful_desired(x, state):
                return ctrl_obj._optimal_control_single(x, state)
            init_state_fn = ctrl_obj.get_init_state
            return self._create_updated_instance(
                desired_control=stateful_desired,
                desired_control_init_state=init_state_fn,
            )
        else:
            func = desired_control
            def stateful_desired(x, state):
                return func(x), state
            return self._create_updated_instance(
                desired_control=stateful_desired,
                desired_control_init_state=lambda: None,
            )

    @property
    def desired_control(self):
        return self._desired_control

    def _make_objective_single(self, x: jnp.ndarray) -> tuple:
        """
        Create QP objective for minimum intervention.

        Args:
            x: State vector (state_dim,)

        Returns:
            Tuple (Q, c) for QP objective
        """
        # Get desired control (stateless call for QP objective)
        u_des, _ = self._desired_control(x, self.get_init_state())

        # Minimum intervention objective: min ||u - u_des||^2
        Q = jnp.eye(self._action_dim)
        c = -u_des

        return Q, c
