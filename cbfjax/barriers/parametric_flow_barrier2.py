"""
FlowBarrier2 implementation for JAX.

Flow barrier over augmented state s = [x, θ_flat, γ] whose plan blends the
parametric control into a backup policy:

    π(τ; x, θ) = (1 − λ(τ)) u_p(τ; θ) + λ(τ) u_b(x)

with λ a smoothstep from T_b = blend_fraction · horizon to T = horizon. The
physical control is π itself, not a decision variable, so the augmented action
is v = [ω, z] and the augmented dynamics are

    f̄(s) = [f(x, π(γ; x, θ)); 0_{d+1}],   ḡ = [[0_{n×(d+1)}], [I_{d+1}]].

The original dynamics only need rhs(x, u) (not control-affine).
"""

import jax.numpy as jnp
import equinox as eqx
from typing import Any

from cbfjax.barriers.parametric_flow_barrier import FlowBarrier
from cbfjax.dynamics.base_dynamic import CustomDynamics
from cbfjax.utils.integration import get_trajs_from_time_state_action_func


def smoothstep(tau, t_b, t_end):
    """λ(τ): 0 for τ ≤ t_b, 1 for τ ≥ t_end, 3s² − 2s³ with s = (τ−t_b)/(t_end−t_b)."""
    s = jnp.clip((tau - t_b) / (t_end - t_b), 0.0, 1.0)
    return s * s * (3.0 - 2.0 * s)


class FlowBarrier2(FlowBarrier):
    """
    Flow Barrier with backup-policy blended plan.

    All fields are immutable following Equinox patterns.
    """

    _backup_policy: Any = eqx.field(static=True)
    blend_fraction: float = eqx.field(static=True)

    def __init__(self, backup_policy=None, blend_fraction=0.75, **kwargs):
        super().__init__(**kwargs)
        self._backup_policy = backup_policy
        self.blend_fraction = float(blend_fraction)

    @classmethod
    def create_empty(cls, cfg=None):
        """Create an empty FlowBarrier2 instance."""
        cfg = cfg or {}
        base = super().create_empty(cfg)
        return base._create_updated_instance(
            blend_fraction=cfg.get('blend_fraction', 0.75))

    def _create_updated_instance(self, **kwargs):
        kwargs.setdefault('backup_policy', self._backup_policy)
        kwargs.setdefault('blend_fraction', self.blend_fraction)
        return super()._create_updated_instance(**kwargs)

    def _trajectory_duration(self, gamma):
        """Window [gamma, gamma + horizon] has fixed length for every gamma."""
        return self.horizon

    # === Public Assignment Interface ===

    def assign_backup_policy(self, policy):
        """
        Assign backup policy u_b(x). Must be bounded by design.

        Args:
            policy: Callable x -> u_b (action_dim,)

        Returns:
            New FlowBarrier2 instance with assigned backup policy
        """
        assert callable(policy), "backup policy must be callable"
        return self._create_updated_instance(backup_policy=policy)

    def assign_dynamics(self, dynamics):
        """
        Assign original dynamics and compute dimensions.

        Args:
            dynamics: System dynamics object

        Returns:
            New FlowBarrier2 instance with assigned dynamics and computed dimensions
        """
        theta_flat_dim = dynamics.action_dim * self.control_param_num
        aug_state_dim = dynamics.state_dim + theta_flat_dim + 1  # x + θ + γ
        aug_action_dim = theta_flat_dim + 1  # ω + z

        return self._create_updated_instance(
            original_dynamics=dynamics,
            theta_flat_dim=theta_flat_dim,
            aug_state_dim=aug_state_dim,
            aug_action_dim=aug_action_dim
        )

    # === Properties ===

    @property
    def backup_policy(self):
        return self._backup_policy

    # === Blended Plan ===

    def blended_control(self, tau, x, theta):
        """
        Blended plan π(τ; x, θ) = (1 − λ(τ)) u_p(τ; θ) + λ(τ) u_b(x).

        λ takes ABSOLUTE plan time: T_b = blend_fraction · horizon, T = horizon.
        Single-sample; batch with jax.vmap.

        Args:
            tau: Plan time (scalar)
            x: State vector (n,)
            theta: Control parameters (action_dim, num_params)

        Returns:
            Control input (action_dim,)
        """
        lam = smoothstep(tau, self.blend_fraction * self.horizon, self.horizon)
        u_p = self._parametric_control(tau, theta)
        u_b = self._backup_policy(x)
        return (1.0 - lam) * u_p + lam * u_b

    # === Private Implementation ===

    def _validate_configuration(self):
        """Validate that all required components are assigned"""
        super()._validate_configuration()
        assert self._backup_policy is not None, \
            "Backup policy must be assigned using assign_backup_policy()"

    def _create_augmented_dynamics(self):
        """Create augmented dynamics: f̄(s) = [f(x, π(γ; x, θ)); 0], ḡ = [0; I]."""
        original_dynamics = self._original_dynamics
        n = original_dynamics.state_dim
        aug_action_dim = self._aug_action_dim
        extract_params = self._extract_parameters_from_state
        blended_control = self.blended_control

        g_const = jnp.vstack([jnp.zeros((n, aug_action_dim)),
                              jnp.eye(aug_action_dim)])

        def aug_f(s):
            x, theta, gamma = extract_params(s)
            u = blended_control(gamma, x, theta)
            return jnp.concatenate([original_dynamics.rhs(x, u),
                                    jnp.zeros(aug_action_dim)])

        def aug_g(s):
            return g_const

        augmented_dynamics = CustomDynamics(
            state_dim=self._aug_state_dim,
            action_dim=aug_action_dim,
            f_func=aug_f,
            g_func=aug_g
        )

        return self._create_updated_instance(augmented_dynamics=augmented_dynamics)

    def compute_trajectory(self, x, theta, gamma):
        """
        Compute flow trajectory φ(τ; x, θ, γ) under the blended plan for a single state.

        The integrand needs the current flow state for u_b(φ(τ)).

        Args:
            x: Initial state (n,)
            theta: Control parameters (action_dim, num_params)
            gamma: Time shift parameter (scalar)

        Returns:
            Tuple (trajectory (time_steps, state_dim) evaluate function)
        """
        target_points = int(self.horizon / self.time_steps) + 1

        blended_control = self.blended_control

        def action_func(tau, y):
            return blended_control(tau, y, theta)

        return get_trajs_from_time_state_action_func(
            x0=x,
            dynamics=self._original_dynamics,
            action_func=action_func,
            start_time=gamma,
            sim_time=gamma + self.horizon,
            timestep=self.time_steps,
            num_steps=target_points,
            method=self.integration_method
        )
