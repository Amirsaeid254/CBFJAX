"""
FlowBarrier implementation for JAX.

This module implements Flow Barrier extending MultiBarriers, combining state barriers,
backup barriers, action constraints, and time-shift barriers using augmented state
s = [x, θ_flat, γ].
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Callable, List, Optional, Tuple, Any
from functools import partial

from cbfjax.barriers.multi_barrier import MultiBarriers
from cbfjax.barriers.barrier import Barrier
from cbfjax.barriers.composite_barrier import SoftCompositionBarrier
from cbfjax.dynamics.base_dynamic import AffineInControlDynamics, CustomDynamics, create_augmented_dynamics
from cbfjax.dynamics.single_integrator import SingleIntegratorDynamics
from cbfjax.utils.utils import softmin
from cbfjax.utils.integration import get_trajs_from_time_action_func


class FlowBarrier(MultiBarriers):
    """
    Flow Barrier implementation extending MultiBarriers.

    Combines state barriers, backup barriers, action constraints, and time-shift barriers
    using augmented state s = [x, θ_flat, γ] where:
    - x: original state
    - θ: control parameters
    - γ: time shift parameter

    All fields are immutable following Equinox patterns.
    """

    # Configuration
    _rel_deg: int = eqx.field(static=True)
    horizon: float = eqx.field(static=True)
    time_steps: float = eqx.field(static=True)
    control_param_method: str = eqx.field(static=True)
    control_param_num: int = eqx.field(static=True)

    # User-assigned components (static as they contain functions)
    _state_barrier: Any = eqx.field(static=True)
    _backup_barriers: tuple = eqx.field(static=True)
    _original_dynamics: Any = eqx.field(static=True)

    # Flow-specific components
    _parametric_control: Any = eqx.field(static=True)
    _augmented_dynamics: Any = eqx.field(static=True)

    # Parameter dimensions
    _theta_flat_dim: int = eqx.field(static=True)
    _aug_state_dim: int = eqx.field(static=True)
    _aug_action_dim: int = eqx.field(static=True)

    # Flags for barrier creation strategy
    compose_state_barriers: bool = eqx.field(static=True)
    compose_action_barriers: bool = eqx.field(static=True)
    danskin_state_barriers: bool = eqx.field(static=True)

    # Softmin parameters
    traj_softmin_rho: float = eqx.field(static=True)
    action_softmin_rho: float = eqx.field(static=True)

    # Integration method
    integration_method: str = eqx.field(static=True)

    # Control bounds (optional) - stored as tuples for static field compatibility
    control_low: Any = eqx.field(static=True)
    control_high: Any = eqx.field(static=True)

    # Default plan: per-action-channel constants for the initial theta
    # (None -> zeros)
    theta_init: Any = eqx.field(static=True)

    def __init__(
            self,
            barrier_func=None,
            dynamics=None,
            rel_deg=1,
            alphas=None,
            barriers=None,
            hocbf_func=None,
            cfg=None,
            barrier_funcs=None,
            hocbf_funcs=None,
            multidim_indices=None,
            # FlowBarrier specific
            state_barrier_rel_deg=1,
            horizon=1.0,
            time_steps=0.1,
            control_param_method="zoh",
            control_param_num=10,
            state_barrier=None,
            backup_barriers=None,
            original_dynamics=None,
            parametric_control=None,
            augmented_dynamics=None,
            theta_flat_dim=0,
            aug_state_dim=0,
            aug_action_dim=0,
            compose_state_barriers=True,
            compose_action_barriers=True,
            danskin_state_barriers=False,
            traj_softmin_rho=1.0,
            action_softmin_rho=1.0,
            integration_method='tsit5',
            control_low=None,
            control_high=None,
            theta_init=None
    ):
        """Initialize FlowBarrier with all parameters."""
        # Initialize parent MultiBarriers with augmented dynamics
        super().__init__(
            barrier_func=barrier_func,
            dynamics=augmented_dynamics,  # Use augmented dynamics, not original
            rel_deg=rel_deg,
            alphas=alphas,
            barriers=barriers,
            hocbf_func=hocbf_func,
            cfg=cfg,
            barrier_funcs=barrier_funcs,
            hocbf_funcs=hocbf_funcs,
            multidim_indices=multidim_indices
        )

        # FlowBarrier specific fields
        self._rel_deg = state_barrier_rel_deg
        self.horizon = float(horizon)
        self.time_steps = float(time_steps)
        self.control_param_method = control_param_method
        self.control_param_num = int(control_param_num)

        self._state_barrier = state_barrier
        self._backup_barriers = tuple(backup_barriers) if backup_barriers else tuple()
        self._original_dynamics = original_dynamics

        self._parametric_control = parametric_control
        self._augmented_dynamics = augmented_dynamics

        self._theta_flat_dim = int(theta_flat_dim)
        self._aug_state_dim = int(aug_state_dim)
        self._aug_action_dim = int(aug_action_dim)

        self.compose_state_barriers = compose_state_barriers
        self.compose_action_barriers = compose_action_barriers
        self.danskin_state_barriers = danskin_state_barriers

        self.traj_softmin_rho = float(traj_softmin_rho)
        self.action_softmin_rho = float(action_softmin_rho)

        self.integration_method = integration_method
        # Convert to tuples for static field compatibility
        self.control_low = tuple(control_low) if control_low is not None else None
        self.control_high = tuple(control_high) if control_high is not None else None
        self.theta_init = tuple(theta_init) if theta_init is not None else None

    @classmethod
    def create_empty(cls, cfg=None):
        """Create an empty FlowBarrier instance."""
        cfg = cfg or {}
        return cls(
            cfg=cfg,
            state_barrier_rel_deg=cfg.get('state_barrier_rel_deg', 1),
            horizon=cfg.get('horizon', 1.0),
            time_steps=cfg.get('time_steps', 0.1),
            control_param_method=cfg.get('control_param_method', 'zoh'),
            control_param_num=cfg.get('control_param_num', 10),
            compose_state_barriers=cfg.get('compose_state_barriers', True),
            compose_action_barriers=cfg.get('compose_action_barriers', True),
            danskin_state_barriers=cfg.get('danskin_state_barriers', False),
            traj_softmin_rho=cfg.get('traj_softmin_rho', 1.0),
            action_softmin_rho=cfg.get('action_softmin_rho', 1.0),
            integration_method=cfg.get('integration_method', 'tsit5'),
            control_low=cfg.get('control_low', None),
            control_high=cfg.get('control_high', None),
            theta_init=cfg.get('theta_init', None)
        )

    # === Helper for Immutable Updates ===

    def _create_updated_instance(self, **kwargs):
        """
        Create new FlowBarrier instance with updated fields.

        Args:
            **kwargs: Fields to update

        Returns:
            New FlowBarrier instance with updated fields
        """
        defaults = super()._ctor_defaults()
        defaults.update({
            # FlowBarrier specific fields
            'state_barrier_rel_deg': self._rel_deg,
            'horizon': self.horizon,
            'time_steps': self.time_steps,
            'control_param_method': self.control_param_method,
            'control_param_num': self.control_param_num,
            'state_barrier': self._state_barrier,
            'backup_barriers': self._backup_barriers,
            'original_dynamics': self._original_dynamics,
            'parametric_control': self._parametric_control,
            'augmented_dynamics': self._augmented_dynamics,
            'theta_flat_dim': self._theta_flat_dim,
            'aug_state_dim': self._aug_state_dim,
            'aug_action_dim': self._aug_action_dim,
            'compose_state_barriers': self.compose_state_barriers,
            'compose_action_barriers': self.compose_action_barriers,
            'danskin_state_barriers': self.danskin_state_barriers,
            'traj_softmin_rho': self.traj_softmin_rho,
            'action_softmin_rho': self.action_softmin_rho,
            'integration_method': self.integration_method,
            'control_low': self.control_low,
            'control_high': self.control_high,
            'theta_init': self.theta_init
        })
        defaults.update(kwargs)
        return self.__class__(**defaults)

    # === Public Assignment Interface ===

    def assign_state_barrier(self, state_barrier):
        """
        Assign state barrier - can be a single Barrier or list of Barriers.

        Args:
            state_barrier: Single Barrier or list of Barriers

        Returns:
            New FlowBarrier instance with assigned state barrier
        """
        if isinstance(state_barrier, list):
            assigned_barrier = SoftCompositionBarrier(barriers=state_barrier,
                                                      rule='i', cfg=self.cfg)
        elif isinstance(state_barrier, Barrier):
            assigned_barrier = state_barrier
        else:
            raise TypeError(f"state_barrier must be Barrier or list of Barriers, got {type(state_barrier)}")

        return self._create_updated_instance(state_barrier=assigned_barrier)

    def assign_backup_barrier(self, backup_barriers):
        """
        Assign backup barrier(s) - can be a single Barrier or list of Barriers.

        Args:
            backup_barriers: Single Barrier or list of Barriers

        Returns:
            New FlowBarrier instance with assigned backup barriers
        """
        if isinstance(backup_barriers, list):
            assert len(backup_barriers) > 0, 'backup_barriers list must have at least one item'
            assert all(isinstance(f, Barrier) for f in backup_barriers), \
                "all backup barriers must be Barrier instances"
            barriers_tuple = tuple(backup_barriers)
        elif isinstance(backup_barriers, Barrier):
            barriers_tuple = (backup_barriers,)
        else:
            raise TypeError(f"backup_barriers must be Barrier or list of Barriers, got {type(backup_barriers)}")

        return self._create_updated_instance(backup_barriers=barriers_tuple)

    def assign_dynamics(self, dynamics):
        """
        Assign original dynamics and compute dimensions.

        Args:
            dynamics: System dynamics object

        Returns:
            New FlowBarrier instance with assigned dynamics and computed dimensions
        """
        # Compute augmented state dimensions
        theta_flat_dim = dynamics.action_dim * self.control_param_num
        aug_state_dim = dynamics.state_dim + theta_flat_dim + 1  # x + θ + γ
        aug_action_dim = dynamics.action_dim + theta_flat_dim + 1  # u + ω + z

        return self._create_updated_instance(
            original_dynamics=dynamics,
            theta_flat_dim=theta_flat_dim,
            aug_state_dim=aug_state_dim,
            aug_action_dim=aug_action_dim
        )

    # === Main Build Method ===

    def make(self):
        """Build the flow barrier system"""
        self._validate_configuration()

        # Initialize parametric control
        flow_barrier_with_control = self._initialize_parametric_control()

        # Create augmented dynamics
        flow_barrier_with_aug_dynamics = flow_barrier_with_control._create_augmented_dynamics()

        # Create and add the three main barriers
        final_flow_barrier = flow_barrier_with_aug_dynamics._create_and_add_barriers()

        return final_flow_barrier

    # === Public Interface Methods (single-sample; batch with jax.vmap) ===

    def hocbf(self, x, theta=None, gamma=None):
        """
        Compute highest order CBF using augmented state.

        Args:
            x: State vector (n,)
            theta: Control parameters or None
            gamma: Time shift parameter or None

        Returns:
            HOCBF values (num_barriers,)
        """
        s = self._create_augmented_state(x, theta, gamma)
        return super().hocbf(s)

    def barrier(self, x, theta=None, gamma=None):
        """
        Compute barrier values using augmented state.

        Args:
            x: State vector (n,)
            theta: Control parameters or None
            gamma: Time shift parameter or None

        Returns:
            Barrier values (num_barriers,)
        """
        s = self._create_augmented_state(x, theta, gamma)
        return super().barrier(s)

    def get_hocbf_and_lie_derivs(self, x, theta=None, gamma=None):
        """
        Get HOCBF and Lie derivatives with respect to augmented state.

        Args:
            x: State vector (n,)
            theta: Control parameters or None
            gamma: Time shift parameter or None

        Returns:
            Tuple of (hocbf_values, Lf_hocbf, Lg_hocbf), per-member shapes
        """
        s = self._create_augmented_state(x, theta, gamma)
        return super().get_hocbf_and_lie_derivs(s)

    def get_flow_info(self, x, theta, gamma):
        """
        Get detailed flow barrier information for batched states.

        Args:
            x: State vectors (batch, state_dim)
            theta: Control parameters (batch, action_dim, num_params)
            gamma: Time shift parameters (batch,)

        Returns:
            Dictionary with flow information
        """
        if theta is None or gamma is None:
            raise ValueError("theta and gamma must be provided")

        trajectory = self.compute_trajectory(x, theta, gamma)
        flow_safety = self.hocbf(x, theta, gamma)
        terminal_state = trajectory[-1]
        h_backup = self._backup_barriers[0].hocbf(terminal_state)

        return {
            'flow_safety': flow_safety,
            'trajectory': trajectory,
            'h_backup': h_backup,
            'theta': theta,
            'gamma': gamma
        }

    # === Properties ===

    @property
    def state_barrier(self):
        return self._state_barrier

    @property
    def backup_barriers(self):
        return self._backup_barriers

    @property
    def original_dynamics(self):
        return self._original_dynamics

    # === Private Implementation ===

    def _validate_configuration(self):
        """Validate that all required components are assigned"""
        assert self._state_barrier is not None, \
            "State barrier must be assigned using assign_state_barrier()"
        assert len(self._backup_barriers) > 0, \
            "Backup barriers must be assigned using assign_backup_barrier()"
        assert self._original_dynamics is not None, \
            "Dynamics must be assigned using assign_dynamics()"

    def _initialize_parametric_control(self):
        """Initialize parametric control structure"""
        # Import here to avoid circular dependency
        from ..controls.parametric_control import create_parametric_control

        parametric_control = create_parametric_control(
            method=self.control_param_method,
            horizon=self.horizon,
            control_dim=self._original_dynamics.action_dim,
            num_params=self.control_param_num,
            dt=self.time_steps
        )

        # Set control bounds if available
        if self.control_low is not None and self.control_high is not None:
            parametric_control = parametric_control.set_control_bounds(self.control_low, self.control_high)

        return self._create_updated_instance(parametric_control=parametric_control)

    def _create_augmented_dynamics(self):
        """Create augmented dynamics for s = [x, θ_flat, γ] using block-diagonal composition."""

        augmented_dynamics = create_augmented_dynamics([
            self._original_dynamics,
            SingleIntegratorDynamics(dim=self._theta_flat_dim),
            SingleIntegratorDynamics(dim=1),
        ])

        return self._create_updated_instance(augmented_dynamics=augmented_dynamics)

    def _create_and_add_barriers(self):
        """Create and add the three main barriers to MultiBarriers"""
        traj_backup_barriers = []
        other_barriers = []

        # 1. Trajectory + Backup barrier
        if not self.compose_state_barriers:
            if self.danskin_state_barriers:
                danskin_state_barriers = self._create_danskin_trajectory_backup_barriers()
                traj_backup_barriers.append(danskin_state_barriers)
            else:
                # Use separate barriers for each state constraint along trajectory
                individual_state_barriers = self._create_individual_trajectory_backup_barriers()
                traj_backup_barriers.append(individual_state_barriers)
        else:
            # Use composed barrier with softmin (returns 2D: trajectory + backup)
            traj_backup_barrier = self._create_trajectory_backup_barrier()
            traj_backup_barriers.append(traj_backup_barrier)

        # 2. Action constraint barriers
        if not self.compose_action_barriers:
            # Use separate barriers for each constraint
            individual_action_barriers = self._create_individual_action_constraint_barriers()
            other_barriers.extend(individual_action_barriers)
        else:
            # Use combined barrier with softmin
            action_barrier = self._create_action_constraint_barrier()
            if action_barrier is not None:
                other_barriers.append(action_barrier)

        # 3. Time shift barrier
        time_shift_barrier = self._create_time_shift_barrier()
        other_barriers.append(time_shift_barrier)

        # Add trajectory/backup barriers with multidim=True
        flow_barrier_with_traj = self.add_barriers(traj_backup_barriers, infer_dynamics=True, multidim=True)

        # Add other barriers with multidim=False
        return flow_barrier_with_traj.add_barriers(other_barriers, infer_dynamics=False, multidim=False)

    def _create_trajectory_backup_barrier(self):
        """Create combined trajectory + backup barrier using softmin, returning 2 separate constraints"""

        state_barrier = self._state_barrier
        backup_barrier = self._backup_barriers[0]
        compute_traj= self.compute_trajectory
        extract_params = self._extract_parameters_from_state
        traj_rho = self.traj_softmin_rho

        def trajectory_backup_func(s):
            # Extract parameters from augmented state
            x, theta, gamma = extract_params(s)

            # Compute trajectory
            trajectory = compute_traj(x, theta, gamma)  # Shape: (time_steps, state_dim)

            # Evaluate state barrier along trajectory
            h_traj_values = jax.vmap(state_barrier.hocbf)(trajectory[:-1])  # Shape: (time_steps-2,)
            h_traj_combined = softmin(h_traj_values, rho=traj_rho, dim=0)

            # Evaluate backup barrier at terminal state
            terminal_state = trajectory[-1]
            h_backup = backup_barrier.hocbf(terminal_state)

            # Return both trajectory and backup as separate constraints
            return jnp.array([h_traj_combined, h_backup])

        return Barrier(barrier_func=trajectory_backup_func,
                       rel_deg=self._rel_deg,
                       dynamics=self._augmented_dynamics)

    def _create_individual_trajectory_backup_barriers(self):
        """Create separate barriers for each individual state constraint along trajectory"""

        # Capture necessary attributes
        state_barrier = self._state_barrier
        backup_barrier = self._backup_barriers[0]
        compute_traj = self.compute_trajectory
        extract_params = self._extract_parameters_from_state

        def individual_trajectory_backup_func(s):
            """
            Multi-dimensional barrier function that returns state constraints at each trajectory point.
            Returns shape: (num_trajectory_points,)
            """
            # Extract parameters from augmented state
            x, theta, gamma = extract_params(s)

            # Compute trajectory
            trajectory, _ = compute_traj(x, theta, gamma)  # Shape: (time_steps, state_dim)

            # Evaluate state barrier along trajectory (exclude first and last)
            h_traj = jax.vmap(state_barrier.hocbf)(trajectory[1:-1])  # Shape: (time_steps-2,)

            # Evaluate backup barrier at terminal state
            terminal_state = trajectory[-1]
            h_backup = backup_barrier.hocbf(terminal_state)

            # Concatenate trajectory constraints with backup constraint
            # h_backup could be scalar or 1D array depending on barrier type
            h_backup_arr = jnp.atleast_1d(h_backup)
            h_combined = jnp.concatenate([h_traj, h_backup_arr])

            return h_combined.squeeze(-1)

        # Create single barrier that returns multi-dimensional output
        return Barrier(barrier_func=individual_trajectory_backup_func,
                       rel_deg=self._rel_deg,
                       dynamics=self._augmented_dynamics)

    def _create_danskin_trajectory_backup_barriers(self):
        """
        Create barriers using Danskin approach: only constrain global minimum points.

        Uses _compute_minimizer to find critical points and returns fixed-size padded output.
        """
        # Capture necessary attributes
        backup_barrier = self._backup_barriers[0]
        compute_traj = self.compute_trajectory
        extract_params = self._extract_parameters_from_state
        compute_minimizer = self._compute_minimizer

        def danskin_trajectory_backup_func(s):
            """
            Multi-dimensional barrier function that returns constraints only at
            global minimum points along trajectory, plus backup barrier.

            Returns shape: (horizon_steps-1,) where:
            - First r entries: actual barrier values at global minimum points
            - Remaining (horizon_steps-2 - r) entries: large constants (100.0) - always satisfied
            - Last entry: backup barrier at terminal state
            """
            # Extract parameters from augmented state
            x, theta, gamma = extract_params(s)

            # Compute trajectory
            trajectory, dense_func = compute_traj(x, theta, gamma)  # Shape: (horizon_steps, state_dim)

            # Get padded barrier values at global minimum points
            # Returns: (horizon_steps-2,) with actual h values at global mins, rest are 100.0
            h_traj = compute_minimizer(trajectory, dense_func, theta, gamma)

            # Evaluate backup barrier at terminal state
            terminal_state = trajectory[-1]
            h_backup = backup_barrier.hocbf(terminal_state)

            # Concatenate trajectory global mins with backup constraint
            h_backup_arr = jnp.atleast_1d(h_backup)
            h_combined = jnp.concatenate([h_traj, h_backup_arr])

            return h_combined  # Shape: (horizon_steps-2+1,) = (horizon_steps-1,)

        # Create single barrier that returns multi-dimensional output
        return Barrier(barrier_func=danskin_trajectory_backup_func,
                       rel_deg=self._rel_deg,
                       dynamics=self._augmented_dynamics)

    def _create_action_constraint_barrier(self):
        """Create action constraint barrier if control bounds exist"""
        if self.control_low is None or self.control_high is None:
            return None

        theta_barrier_funcs = self._parametric_control.get_action_barrier_functions()
        if not theta_barrier_funcs:
            return None

        # Capture attributes
        original_state_dim = self._original_dynamics.state_dim
        theta_flat_dim = self._theta_flat_dim
        action_dim = self._original_dynamics.action_dim
        control_param_num = self.control_param_num
        action_rho = self.action_softmin_rho

        def action_constraint_func(s):
            # Extract theta_flat from augmented state
            theta_start = original_state_dim
            theta_end = theta_start + theta_flat_dim
            theta_flat = s[theta_start:theta_end]

            # Reshape to (action_dim, num_params)
            theta = theta_flat.reshape(action_dim, control_param_num)

            # Apply all constraint functions and collect results
            constraint_values = []
            for theta_func in theta_barrier_funcs:
                segment_vals = theta_func(theta)  # Shape: (num_segments,)
                constraint_values.append(segment_vals)

            if constraint_values:
                h_action = jnp.concatenate(constraint_values)
                return softmin(h_action, rho=action_rho, dim=0)
            else:
                return 1.0

        return Barrier(barrier_func=action_constraint_func,
                       rel_deg=1,
                       dynamics=self._augmented_dynamics)

    def _create_individual_action_constraint_barriers(self):
        """Create separate barriers for each individual action constraint (no softmin)"""
        if self.control_low is None or self.control_high is None:
            return []

        theta_barrier_funcs = self._parametric_control.get_action_barrier_functions()
        if not theta_barrier_funcs:
            return []

        # Capture attributes
        original_state_dim = self._original_dynamics.state_dim
        theta_flat_dim = self._theta_flat_dim
        action_dim = self._original_dynamics.action_dim
        control_param_num = self.control_param_num

        individual_barriers = []

        for func_idx, theta_func in enumerate(theta_barrier_funcs):
            # For each constraint function (e.g., lower/upper bounds for each control dim)

            for seg_idx in range(control_param_num):
                # Create a barrier for this specific segment constraint

                def create_individual_constraint_func(constraint_func, segment_index):
                    def individual_constraint_func(s):
                        # Extract theta_flat from augmented state
                        theta_start = original_state_dim
                        theta_end = theta_start + theta_flat_dim
                        theta_flat = s[theta_start:theta_end]

                        # Reshape to (action_dim, num_params)
                        theta = theta_flat.reshape(action_dim, control_param_num)

                        # Apply constraint function and extract specific segment
                        segment_vals = constraint_func(theta)  # Shape: (num_segments,)
                        return segment_vals[segment_index]  # Return scalar, not slice

                    return individual_constraint_func

                # Create barrier with closure-captured function and index
                barrier_func = create_individual_constraint_func(theta_func, seg_idx)

                barrier = Barrier(barrier_func=barrier_func,
                       rel_deg=1,
                       dynamics=self._augmented_dynamics)

                individual_barriers.append(barrier)

        return individual_barriers

    def _create_time_shift_barrier(self):
        """Create time shift barrier: γ ≥ 0"""

        def time_shift_func(s):
            return s[-1]  # Extract γ (last element) as scalar

        return Barrier(barrier_func=time_shift_func,
                       rel_deg=1,
                       dynamics=self._augmented_dynamics)

    # === Helper Methods for State and Parameter Management ===

    def _create_augmented_state(self, x, theta=None, gamma=None):
        """
        Create augmented state s = [x, θ_flat, γ] for single state.

        Args:
            x: State vector (n,)
            theta: Control parameters (action_dim, num_params) or None
            gamma: Time shift parameter (scalar) or None

        Returns:
            Augmented state s (aug_state_dim,)
        """
        if theta is None or gamma is None:
            theta, gamma = self._get_default_parameters()

        # Flatten theta and concatenate
        theta_flat = theta.flatten()
        gamma_scalar = jnp.atleast_1d(gamma)

        return jnp.concatenate([x, theta_flat, gamma_scalar])

    def _extract_parameters_from_state(self, s):
        """
        Extract x, theta, gamma from augmented state s for single state.

        Args:
            s: Augmented state (aug_state_dim,)

        Returns:
            Tuple of (x, theta, gamma)
        """
        x = s[:self._original_dynamics.state_dim]

        theta_start = self._original_dynamics.state_dim
        theta_end = theta_start + self._theta_flat_dim
        theta_flat = s[theta_start:theta_end]
        theta = theta_flat.reshape(self._original_dynamics.action_dim, self.control_param_num)

        gamma = s[-1]

        return x, theta, gamma

    def compute_trajectory(self, x, theta, gamma):
        """
        Compute flow trajectory φ(τ; x, θ, γ) for a single state.

        Uses adaptive time step to maintain fixed number of trajectory points.

        Args:
            x: Initial state (n,)
            theta: Control parameters (action_dim, num_params)
            gamma: Time shift parameter (scalar)

        Returns:
            Trajectory (time_steps, state_dim)
        """

        target_points = int(self.horizon / self.time_steps) + 1

        parametric_control_fn = self._parametric_control

        def action_func(tau):
            return parametric_control_fn(tau, theta)

        trajectory= get_trajs_from_time_action_func(
            x0=x,
            dynamics=self._original_dynamics,
            action_func=action_func,
            start_time=gamma,
            sim_time=self.horizon,
            num_steps=target_points,
            method=self.integration_method
        )

        return trajectory


    def _evaluate_traj_backup_on_trajectory(self, trajectory, theta, gamma):
        """
        Evaluate trajectory+backup barriers on a pre-computed trajectory.

        Replicates the barrier logic from _create_*_trajectory_backup_barrier
        but avoids recomputing the ODE.

        Returns:
            Barrier values array matching _hocbf_funcs[0] output shape
        """
        backup_barrier = self._backup_barriers[0]
        terminal_state = trajectory[-1]
        h_backup = backup_barrier.hocbf(terminal_state)

        if self.compose_state_barriers:
            h_traj_values = jax.vmap(self._state_barrier.hocbf)(trajectory[:-1])
            h_traj_combined = softmin(h_traj_values, rho=self.traj_softmin_rho, dim=0)
            return jnp.array([h_traj_combined, h_backup])
        elif self.danskin_state_barriers:
            h_traj = self._compute_minimizer(trajectory, dense_func, theta, gamma)
            return jnp.concatenate([h_traj, jnp.atleast_1d(h_backup)])
        else:
            h_traj = jax.vmap(self._state_barrier.hocbf)(trajectory[1:-1])
            return jnp.concatenate([h_traj, jnp.atleast_1d(h_backup)]).squeeze(-1)

    def _compute_minimizer(self, trajectory, dense_func, theta, gamma, tolerance=1e-3, epsilon_threshold=1e-3):
        """
        Find global minimum barrier points along a controlled trajectory.

        Returns a padded array where entries at global minima contain barrier values
        and all other entries are 100.0 (indicating satisfied constraints).

        Algorithm:
            1. Compute dh/dt = ∇h·f + ∇h·g·u_p along trajectory
            2. Identify candidate times: start, stationary points, zero crossings
            3. Evaluate barrier at all candidates using dense ODE output
            4. Return global minima (within tolerance) as padded array

        Args:
            trajectory: Discretized trajectory states, shape (horizon_steps, state_dim)
            dense_func: Dense output function from ODE solver
            theta: Parametric control parameters, shape (action_dim, num_params)
            gamma: Time shift parameter (initial time)
            tolerance: Tolerance for identifying global minima
            epsilon_threshold: Threshold for |dh/dt| ≈ 0

        Returns:
            Barrier values at global minima, padded with 100.0, shape (horizon_steps-2,)
        """
        horizon_steps = trajectory.shape[0]

        # ===== SIMPLE VERSION: Use only discretized trajectory points (no interpolation) =====
        USE_SIMPLE_VERSION = True  # Set to False to revert to full version

        if USE_SIMPLE_VERSION:
            # Evaluate barrier at all trajectory points (excluding first and last)
            h_traj = jax.vmap(self._state_barrier.hocbf)(trajectory[1:-1])  # Shape: (horizon_steps-2,)

            # Find global minimum
            min_h_value = jnp.min(h_traj)

            # Identify all values within tolerance of global minimum
            # Stop gradient through comparison to avoid second derivatives
            is_global_min = jax.lax.stop_gradient(
                jnp.abs(h_traj - min_h_value) < tolerance
            )

            # Create output: global min values, else 100.0 (satisfied constraint)
            # Gradients flow through h_traj but not through the selection mask
            h_output = jnp.where(is_global_min, h_traj, 100.0)

            return h_output  # Shape: (horizon_steps-2,)
        # ===== END SIMPLE VERSION =====

        # Step 1: Compute barrier time derivatives
        # Stop gradient here to avoid second derivatives (gradient of gradient)
        barrier_derivs = jax.lax.stop_gradient(
            self._compute_trajectory_barrier_derivatives(
                trajectory, theta, gamma, horizon_steps
            )
        )

        # Step 2: Build candidate tau values
        # Get candidates - keep gradients through tau values themselves
        tau_candidates, valid_mask = self._build_candidate_times(
            barrier_derivs, gamma, horizon_steps, epsilon_threshold
        )

        # Stop gradient on the validity mask only (which candidates to use)
        # But allow gradients through the tau values themselves
        valid_mask = jax.lax.stop_gradient(valid_mask)

        # Step 3: Evaluate barriers at all valid candidates
        # Gradients flow through both tau and barrier evaluation (Danskin's theorem)
        h_values = self._evaluate_barriers_at_candidates(
            tau_candidates, valid_mask, dense_func
        )

        # Step 4: Extract global minima and pad output
        h_output = self._extract_global_minima(
            h_values, tolerance, horizon_steps
        )

        # ==================== TEMPORARY: Debug Visualization ====================
        def debug_plot(h_traj, barrier_derivs, tau_candidates, valid_mask, h_values, gamma_val):
            """Debug callback executed outside traced context."""
            import matplotlib.pyplot as plt
            import numpy as np

            # Convert to numpy (now safe because we're outside traced context)
            h_traj_np = np.array(h_traj).flatten()
            barrier_derivs_np = np.array(barrier_derivs).flatten()
            horizon_times = np.linspace(gamma_val, self.horizon, horizon_steps)
            tau_cand_np = np.array(tau_candidates)
            valid_np = np.array(valid_mask)
            h_vals_np = np.array(h_values)

            # Find global minima
            min_h = np.min(h_vals_np)
            is_global = np.abs(h_vals_np - min_h) < tolerance
            global_mask = is_global & valid_np

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9))

            # Top: Barrier values
            ax1.plot(horizon_times, h_traj_np, 'b-', linewidth=2, label=r'$h(x(\tau))$')

            # Candidates (X marks)
            cand_mask = valid_np & ~global_mask
            if np.any(cand_mask):
                ax1.scatter(tau_cand_np[cand_mask], h_vals_np[cand_mask],
                           color='orange', s=80, marker='x', linewidths=2.5, zorder=4,
                           label=f'Candidates (n={np.sum(cand_mask)})')

            # Global minima (RED dots)
            if np.any(global_mask):
                ax1.scatter(tau_cand_np[global_mask], h_vals_np[global_mask],
                           color='red', s=200, marker='o', edgecolors='black', linewidths=2,
                           zorder=5, label=f'Global Min (n={np.sum(global_mask)})')

            ax1.axhline(0, color='black', linestyle='--', alpha=0.5)
            ax1.set_ylabel(r'$h_s(x)$', fontsize=13)
            ax1.legend(fontsize=10)
            ax1.grid(True, alpha=0.3)

            # Bottom: Derivatives (skip first point to match barrier_derivs shape)
            ax2.plot(horizon_times[1:], barrier_derivs_np, 'purple', linewidth=2, label=r'$\dot{h}$')

            # Zero crossings
            zero_crossings = valid_np[horizon_steps:] & (np.abs(h_vals_np[horizon_steps:]) < 1e6)
            if np.any(zero_crossings):
                ax2.scatter(tau_cand_np[horizon_steps:][zero_crossings],
                           np.zeros(np.sum(zero_crossings)),
                           color='orange', s=80, marker='x', linewidths=2.5, zorder=4,
                           label='Zero Crossings')

            # Global minima on derivative plot
            if np.any(global_mask):
                for tau_abs, h_val in zip(tau_cand_np[global_mask], h_vals_np[global_mask]):
                    idx = int(np.argmin(np.abs(horizon_times[1:] - tau_abs)))
                    if idx < len(barrier_derivs_np):
                        ax2.scatter([tau_abs], [barrier_derivs_np[idx]],
                                   color='red', s=200, marker='o', edgecolors='black',
                                   linewidths=2, zorder=5)

            ax2.axhline(0, color='black', linestyle='--', alpha=0.5)
            ax2.set_xlabel(r'Time $\tau$ (s)', fontsize=13)
            ax2.set_ylabel(r'$\dot{h}_s$', fontsize=13)
            ax2.set_title('Barrier Derivative', fontsize=13)
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3)

            plt.savefig(f'figs/DEBUG_minimizer_g{float(gamma_val):.2f}.png', dpi=120)
            plt.show()
            plt.close()

        # Execute visualization outside traced context
        h_traj = jax.vmap(self._state_barrier.hocbf)(trajectory)
        gamma_scalar = jnp.squeeze(gamma)
        jax.debug.callback(debug_plot, h_traj, barrier_derivs, tau_candidates,
                          valid_mask, h_values, gamma_scalar)
        # ==================== END TEMPORARY ====================

        return h_output

    def _trajectory_duration(self, gamma):
        """Length of the prediction window, here [gamma, horizon]."""
        return self.horizon - gamma

    def _compute_trajectory_barrier_derivatives(self, trajectory, theta, gamma, horizon_steps):
        """
        Compute dh/dt = ∇h·f + ∇h·g·u_p at each trajectory point (excluding first point).

        Args:
            trajectory: States at discretized times, shape (horizon_steps, state_dim)
            theta: Control parameters, shape (action_dim, num_params)
            gamma: Initial time
            horizon_steps: Number of trajectory points

        Returns:
            Barrier time derivatives, shape (horizon_steps-1,) - excludes first point
        """
        dt_actual = self._trajectory_duration(gamma) / (horizon_steps - 1)

        grad_barrier = jax.grad(lambda x: self._state_barrier.hocbf(x))

        def compute_dhdt_at_index(i):
            """Compute dh/dt at trajectory index i."""
            x = trajectory[i]
            tau = gamma + i * dt_actual

            # Get barrier gradient and dynamics
            grad_h = grad_barrier(x)
            f_x = self._original_dynamics.f(x)
            g_x = self._original_dynamics.g(x)

            # Get parametric control at this time
            u_p = self._parametric_control(tau, theta)

            # Lie derivatives: L_f h + L_g h · u_p
            Lf_h = jnp.dot(grad_h, f_x)
            Lg_h_u = jnp.dot(grad_h, g_x @ u_p)

            return Lf_h + Lg_h_u

        return jax.vmap(compute_dhdt_at_index)(jnp.arange(1, horizon_steps))

    def _build_candidate_times(self, barrier_derivs, gamma, horizon_steps, epsilon_threshold):
        """
        Build candidate time points where barrier minima may occur.

        Candidates include:
            - First point in barrier_derivs (trajectory[1]) - added unconditionally
            - Discretization points where |dh/dt| < epsilon_threshold
            - Interpolated zero crossings of dh/dt

        The discretization uses linspace over [gamma, self.horizon] to match
        the actual integration interval.

        Args:
            barrier_derivs: Time derivatives of barrier, shape (horizon_steps-1,) - excludes trajectory[0]
            gamma: Initial time
            horizon_steps: Number of trajectory points
            epsilon_threshold: Threshold for stationary points

        Returns:
            tau_candidates: Candidate times, shape (max_candidates,)
            valid_mask: Boolean mask for valid candidates, shape (max_candidates,)
        """
        dt_actual = self._trajectory_duration(gamma) / (horizon_steps - 1)

        # Discretization times for trajectory[1:] (excluding trajectory[0])
        # barrier_derivs[i] corresponds to trajectory[i+1]
        tau_discrete = gamma + jnp.arange(1, horizon_steps) * dt_actual  # Shape (horizon_steps-1,)

        # Mask for discretization candidates
        # First point (barrier_derivs[0] = trajectory[1]) is always a candidate
        is_start = jnp.arange(horizon_steps - 1) == 0  # Shape (horizon_steps-1,)
        is_stationary = jnp.abs(barrier_derivs) < epsilon_threshold  # Shape (horizon_steps-1,)
        include_discrete = is_start | is_stationary  # Shape (horizon_steps-1,)

        # Zero crossing detection and interpolation
        epsilon_i = barrier_derivs[:-1]  # Shape (horizon_steps-2,)
        epsilon_ip1 = barrier_derivs[1:]  # Shape (horizon_steps-2,)
        has_zero_crossing = (epsilon_i * epsilon_ip1) < 0  # Shape (horizon_steps-2,)

        # Interpolate: tau where dh/dt = 0 between points i and i+1
        alpha = -epsilon_i / (epsilon_ip1 - epsilon_i + 1e-12)
        i_indices = jnp.arange(1, horizon_steps - 1)  # Shape (horizon_steps-2,), starts at 1
        tau_crossings = gamma + (i_indices + alpha) * dt_actual

        # Build combined arrays with sentinel values
        tau_discrete_masked = jnp.where(include_discrete, tau_discrete, jnp.inf)
        tau_crossing_masked = jnp.where(has_zero_crossing, tau_crossings, jnp.inf)

        # Combine all candidates (fixed-size arrays for JIT)
        tau_candidates = jnp.concatenate([tau_discrete_masked, tau_crossing_masked])
        valid_mask = jnp.concatenate([include_discrete, has_zero_crossing])

        return tau_candidates, valid_mask

    def _evaluate_barriers_at_candidates(self, tau_candidates, valid_mask, dense_func):
        """
        Evaluate barrier function at all candidate time points.

        Args:
            tau_candidates: Candidate times, shape (max_candidates,)
            valid_mask: Validity mask, shape (max_candidates,)
            dense_func: Dense ODE solution interpolator

        Returns:
            Barrier values (inf for invalid candidates), shape (max_candidates,)
        """
        def evaluate_single_candidate(tau, is_valid):
            """Evaluate barrier at tau if valid, else return inf."""
            state = dense_func(tau)
            h_value = self._state_barrier.hocbf(state)
            return jnp.where(is_valid, h_value, jnp.inf)

        return jax.vmap(evaluate_single_candidate)(tau_candidates, valid_mask)

    def _extract_global_minima(self, h_values, tolerance, horizon_steps):
        """
        Extract global minimum barrier values and pad to fixed size.

        Uses stop_gradient on the selection logic to prevent second derivatives
        while preserving gradients through the barrier values (Danskin's theorem).

        Args:
            h_values: Barrier values at candidates, shape (max_candidates,)
            tolerance: Tolerance for identifying global minima
            horizon_steps: Output size (number of trajectory points)

        Returns:
            Padded barrier values (global mins or 100.0), shape (horizon_steps-2,) - excludes first and last points
        """
        # Find global minimum among valid candidates
        min_h_value = jnp.min(h_values)

        # Identify all values within tolerance of global minimum
        # Stop gradient through comparison to avoid second derivatives
        is_global_min = jax.lax.stop_gradient(
            jnp.abs(h_values - min_h_value) < tolerance
        )

        # Create output: global min values, else 100.0 (satisfied constraint)
        # Gradients flow through h_values but not through the selection mask
        h_output_full = jnp.where(is_global_min, h_values, 100.0)

        # Truncate to fixed output size (horizon_steps-2 to match Individual barriers' trajectory[1:-1])
        return h_output_full[:horizon_steps-2]

    def _get_default_parameters(self):
        """
        Get default parameter values for single state.

        Returns:
            Tuple of (theta, gamma); theta filled from the per-channel
            'theta_init' constants when configured, zeros otherwise
        """
        action_dim = self._original_dynamics.action_dim
        if self.theta_init is not None:
            theta = jnp.tile(jnp.array(self.theta_init)[:, None],
                             (1, self.control_param_num))
        else:
            theta = jnp.zeros((action_dim, self.control_param_num))
        gamma = jnp.array([0.0])
        return theta, gamma

    # === Override parent methods that don't apply ===

    def assign(self, barrier_func=None, rel_deg=1, alphas=None):
        """Override to provide clear error message for proper usage"""
        raise NotImplementedError(
            "FlowBarrier uses specialized assignment methods. "
            "Use assign_state_barrier(), assign_backup_barrier(), and assign_dynamics() instead."
        )

    def raise_rel_deg(self, x, raise_rel_deg_by=1, alphas=None):
        """Relative degree raising not implemented for FlowBarrier"""
        raise NotImplementedError("Relative degree raising not supported for FlowBarrier")

    def add_barriers(self, barriers, infer_dynamics=False, multidim=False):
        """
        Override MultiBarriers.add_barriers to preserve FlowBarrier fields.

        Args:
            barriers: List of Barrier objects to add
            infer_dynamics: If True, infer dynamics from first barrier
            multidim: If True, mark these barriers as multi-dimensional

        Returns:
            New FlowBarrier instance with added barriers and preserved FlowBarrier fields
        """
        new_funcs = tuple(b.barrier for b in barriers)
        new_hocbfs = tuple(b.hocbf for b in barriers)
        new_series = tuple(b.barriers for b in barriers)
        start = len(self._mb_barrier_funcs)
        new_idx = tuple(range(start, start + len(barriers))) if multidim else ()
        return self._create_updated_instance(
            barrier_funcs=self._mb_barrier_funcs + new_funcs,
            hocbf_funcs=self._mb_hocbf_funcs + new_hocbfs,
            barriers=self._mb_barriers + new_series,
            multidim_indices=tuple(self._multidim_indices) + new_idx,
        )