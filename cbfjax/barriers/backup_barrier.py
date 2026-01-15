"""
BackupBarrier implementation for JAX.

This module implements Backup Barrier extending Barrier, computing barrier values
based on backup trajectories generated from backup policies.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Callable, List, Optional, Tuple, Any
from functools import partial

from cbfjax.barriers.barrier import Barrier
from cbfjax.barriers.composite_barrier import SoftCompositionBarrier
from cbfjax.dynamics.base import AffineInControlDynamics
from cbfjax.utils.utils import softmin, softmax, apply_and_batchize
from cbfjax.utils.integration import get_trajs_from_state_action_func


class BackupBarrier(Barrier):
    """
    Backup Barrier implementation extending Barrier.

    Computes barrier values by:
    1. Simulating backup trajectories using predefined backup policies
    2. Evaluating state barriers along each backup trajectory
    3. Combining trajectory and terminal backup barrier values using softmin/softmax

    All fields are immutable following Equinox patterns.
    """

    # Configuration
    _rel_deg: int = eqx.field(static=True)

    # User-assigned components
    _state_barrier: Any = eqx.field(static=True)
    _backup_barriers: tuple = eqx.field(static=True)
    _backup_policies: tuple = eqx.field(static=True)

    def __init__(
            self,
            barrier_func=None,
            dynamics=None,
            rel_deg=1,
            alphas=None,
            barriers=None,
            hocbf_func=None,
            cfg=None,
            # BackupBarrier specific
            state_barrier=None,
            backup_barriers=None,
            backup_policies=None
    ):
        """Initialize BackupBarrier with all parameters."""
        # Initialize parent Barrier
        super().__init__(
            barrier_func=barrier_func,
            dynamics=dynamics,
            rel_deg=rel_deg,
            alphas=alphas,
            barriers=barriers,
            hocbf_func=hocbf_func,
            cfg=cfg
        )

        # BackupBarrier specific fields
        self._rel_deg = int(rel_deg)

        self._state_barrier = state_barrier or tuple()
        self._backup_barriers = tuple(backup_barriers) if backup_barriers else tuple()
        self._backup_policies = tuple(backup_policies) if backup_policies else tuple()

    @classmethod
    def create_empty(cls, cfg=None):
        """Create an empty BackupBarrier instance."""
        return cls(cfg=cfg)

    def _create_updated_instance(self, **kwargs):
        """
        Create new instance with updated fields.

        Args:
            **kwargs: Fields to update

        Returns:
            New BackupBarrier instance with updated fields
        """
        defaults = {
            'barrier_func': self._barrier_func,
            'dynamics': self._dynamics,
            'rel_deg': self._rel_deg,
            'alphas': self._alphas,
            'barriers': self._barriers,
            'hocbf_func': self._hocbf_func,
            'cfg': self.cfg,
            'state_barrier': self._state_barrier,
            'backup_barriers': self._backup_barriers,
            'backup_policies': self._backup_policies
        }
        defaults.update(kwargs)
        return self.__class__(**defaults)

    # === Public Assignment Interface ===

    def assign_state_barrier(self, state_barrier):
        """
        Assign state barrier - can be a single Barrier or list of Barriers.

        Args:
            state_barrier: Single Barrier or list of Barriers

        Returns:
            New BackupBarrier instance with assigned state barrier
        """
        if isinstance(state_barrier, list):
            assigned_barrier = SoftCompositionBarrier.create_empty(self.cfg).assign_barriers_and_rule(
                barriers=state_barrier,
                rule='i',
                infer_dynamics=True
            )
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
            New BackupBarrier instance with assigned backup barriers
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

    def assign_backup_policies(self, backup_policies):
        """
        Assign backup policies - must be a list of callable functions.

        Args:
            backup_policies: List of policy functions (state -> action)

        Returns:
            New BackupBarrier instance with assigned backup policies
        """
        if isinstance(backup_policies, list):
            assert len(backup_policies) > 0, 'backup_policies list must have at least one item'
            assert all(callable(f) for f in backup_policies), "all backup policies must be callable"
            policies_tuple = tuple(backup_policies)
        else:
            raise TypeError(f"backup_policies must be a list of callables, got {type(backup_policies)}")

        return self._create_updated_instance(backup_policies=policies_tuple)

    def assign_dynamics(self, dynamics):
        """
        Assign dynamics.

        Args:
            dynamics: System dynamics object

        Returns:
            New BackupBarrier instance with assigned dynamics
        """
        return self._create_updated_instance(dynamics=dynamics)

    # === Main Build Method ===

    def make(self):
        """Build the backup barrier system."""
        self._validate_configuration()

        # Create HOCBF series using parent method
        hocbf_series = self._make_hocbf_series(
            barrier=self._backup_barrier_func,
            dynamics=self._dynamics,
            rel_deg=self._rel_deg,
            alphas=[]
        )

        # Create updated instance
        return self._create_updated_instance(
            barrier_func=self._backup_barrier_func,
            alphas=[],
            barriers=hocbf_series,
            hocbf_func=hocbf_series[-1]
        )

    # === Core Backup Barrier Computation ===

    def _backup_barrier_func(self, x, ret_info=False):
        """
        Compute backup barrier value for single state.

        Args:
            x: State vector (n,)
            ret_info: If True, return dictionary with additional info

        Returns:
            Backup barrier value or info dict
        """
        # Get backup trajectories: (action_num, time_steps, state_dim)
        trajs = self._get_backup_traj_single(x)
        action_num = len(self._backup_policies)

        # Evaluate barriers for each backup policy
        h_list = []
        for i in range(action_num):
            traj = trajs[i]  # (time_steps, state_dim)
            backup_barrier = self._backup_barriers[i]

            # Evaluate state barrier along trajectory (excluding terminal state)
            h_traj = self._state_barrier.hocbf(traj[:-1, :]).squeeze(-1)  # (time_steps-1,)

            # Evaluate backup barrier at terminal state
            h_terminal = backup_barrier._hocbf_single(traj[-1])  # scalar

            # Concatenate and compute softmin
            h_combined = jnp.concatenate([h_traj, h_terminal])
            h_min = softmin(h_combined, rho=self.cfg['softmin_rho'], dim=0)
            h_list.append(h_min)

        # Stack and compute softmax over backup policies
        h_values = jnp.stack(h_list)  # (action_num,)
        final_h_val = softmax(h_values, rho=self.cfg['softmax_rho'], dim=0)

        if not ret_info:
            return final_h_val
        else:
            h_stars = jnp.stack([
                jnp.min(jnp.concatenate([
                    self._state_barrier.hocbf(trajs[i][:-1]).flatten(),
                    jnp.atleast_1d(self._backup_barriers[i]._hocbf_single(trajs[i][-1]))
                ]))
                for i in range(action_num)
            ])
            h_star = jnp.max(h_stars)
            return {'h_star': h_star, 'h_stars': h_stars, 'h': final_h_val, 'h_list': h_list}

    def _get_backup_traj_single(self, x):
        """
        Compute backup trajectories for all backup policies from single state.

        Args:
            x: State vector (n,)

        Returns:
            Trajectories array (action_num, time_steps, state_dim)
        """
        # Python for loop is necessary here since we're iterating over different policy functions,
        # not batched data. Cannot use vmap/lax.map over different functions.
        trajs_list = []
        for policy in self._backup_policies:
            traj = get_trajs_from_state_action_func(
                x0=x,
                dynamics=self._dynamics,
                action_func=policy,
                timestep=self.cfg['time_steps'],
                sim_time=self.cfg['horizon'],
                method=self.cfg['integration_method']
            ).squeeze(1)  # Remove batch dim: (time_steps, state_dim)
            trajs_list.append(traj)

        # Stack trajectories: (action_num, time_steps, state_dim)
        return jnp.stack(trajs_list, axis=0)

    def get_backup_traj(self, x):
        """
        Compute backup trajectories for batched states.

        Args:
            x: State vectors (batch, state_dim)

        Returns:
            Trajectories array (batch, action_num, time_steps, state_dim)
        """
        x = jnp.atleast_2d(x)
        return jax.vmap(self._get_backup_traj_single)(x)

    def get_h_stars(self, x):
        """
        Get minimum barrier values for each backup policy.

        Args:
            x: State vector (state_dim,)

        Returns:
            h_stars values (action_num,)
        """
        # TODO: Implement proper caching that works with JIT compilation
        info = self._backup_barrier_func(x, ret_info=True)
        return info['h_stars']

    def get_h_star(self, x):
        """
        Get maximum minimum barrier value over all backup policies.

        Args:
            x: State vector (state_dim,)

        Returns:
            h_star value (scalar)
        """
        # TODO: Implement proper caching that works with JIT compilation
        info = self._backup_barrier_func(x, ret_info=True)
        return info['h_star']

    # === Properties ===

    @property
    def state_barrier(self):
        return self._state_barrier

    @property
    def backup_barriers(self):
        return self._backup_barriers

    @property
    def backup_policies(self):
        return self._backup_policies

    # === Private Implementation ===

    def _validate_configuration(self):
        """Validate that all required components are assigned."""
        assert self._state_barrier is not None, \
            "State barrier must be assigned using assign_state_barrier()"
        assert len(self._backup_barriers) > 0, \
            "Backup barriers must be assigned using assign_backup_barrier()"
        assert len(self._backup_policies) > 0, \
            "Backup policies must be assigned using assign_backup_policies()"
        assert self._dynamics is not None, \
            "Dynamics must be assigned using assign_dynamics()"
        assert len(self._backup_policies) == len(self._backup_barriers), \
            "Number of backup policies must match number of backup barriers"

    def assign(self, barrier_func=None, rel_deg=1, alphas=None):
        """Override to provide clear error message for proper usage."""
        raise NotImplementedError(
            "BackupBarrier uses specialized assignment methods. "
            "Use assign_state_barrier(), assign_backup_barrier(), "
            "assign_backup_policies(), and assign_dynamics() instead."
        )

    def raise_rel_deg(self, x, raise_rel_deg_by=1, alphas=None):
        """Relative degree raising not implemented for BackupBarrier."""
        raise NotImplementedError("Relative degree raising not supported for BackupBarrier")
