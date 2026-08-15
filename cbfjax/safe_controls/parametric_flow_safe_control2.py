"""
Parametric Flow Safe Control II for JAX.

QP formulation over v = [ω, z] for the augmented state s = [x, θ, γ]: the
physical control u = π(γ; x, θ) is the FlowBarrier2 blended plan, not a QP
decision variable. The objective is

    J(v̂) = (∂J/∂θ)ᵀ ω̂ + ω̂ᵀ Λ ω̂ + Mu ẑ² + λ_linear ẑ

with Q = blockdiag(Λ, Mu) and c = [∂J/∂θ; λ_linear].
"""

import jax
import jax.numpy as jnp

from typing import Tuple

from cbfjax.safe_controls.parametric_flow_safe_control import ParametricFlowSafeControl


class ParametricFlowSafeControl2(ParametricFlowSafeControl):
    """
    Parametric Flow Safe Control over v = [ω, z] with a FlowBarrier2.

    All fields are immutable following Equinox patterns.
    """

    def __init__(
            self,
            action_dim,
            alpha=None,
            params=None,
            dynamics=None,
            barrier=None,
            Q=None,
            c=None,
            control_low=None,
            control_high=None,
            slacked=False,
            slack_gain=100.0,
            cost_functional=None,
            flow_barrier=None,
            alpha_trajectory=None,
            alpha_backup=None,
            alpha_action=None,
            alpha_time_shift=None,
            aug_action_dim=0,
            theta_flat_dim=0,
            Lambda=None,
            Mu=None,
            lambda_linear=None
    ):
        from ..barriers.parametric_flow_barrier2 import FlowBarrier2

        # A FlowBarrier2 passed as barrier= sets the flow fields and
        # augmented dynamics automatically.
        if isinstance(barrier, FlowBarrier2) and flow_barrier is None:
            flow_barrier = barrier
            theta_flat_dim = (barrier.original_dynamics.action_dim *
                              barrier.control_param_num)
            aug_action_dim = theta_flat_dim + 1
            dynamics = barrier._augmented_dynamics

        # Cost matrices -> callable Q/c
        if Lambda is not None:
            if flow_barrier is None:
                raise ValueError("cost matrices require a FlowBarrier2 "
                                 "('barrier' or 'flow_barrier')")
            _td, _gd = int(theta_flat_dim), int(aug_action_dim)
            Q_matrix = jnp.zeros((_gd, _gd))
            Q_matrix = Q_matrix.at[:_td, :_td].set(Lambda)
            Q_matrix = Q_matrix.at[-1, -1].set(Mu)
            c_vector = jnp.zeros(_gd).at[-1].set(lambda_linear)

            def _Q_func(x, theta, gamma):
                return Q_matrix

            def _c_func(x, theta, gamma):
                return c_vector

            Q, c = _Q_func, _c_func

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
            slack_gain=slack_gain,
            cost_functional=cost_functional,
            flow_barrier=flow_barrier,
            alpha_trajectory=alpha_trajectory,
            alpha_backup=alpha_backup,
            alpha_action=alpha_action,
            alpha_time_shift=alpha_time_shift,
            aug_action_dim=aug_action_dim,
            theta_flat_dim=theta_flat_dim
        )

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
            trajectory = flow_barrier.compute_trajectory(x_i, theta_i, gamma_i)

            # Cost on shared trajectory
            J = cost_functional(trajectory)

            # Trajectory + backup barriers on shared trajectory
            h_traj_backup = flow_barrier._evaluate_traj_backup_on_trajectory(
                trajectory, theta_i, gamma_i)

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

        # z is pure regularization: only the θ block of the cost gradient
        # enters c, so the ∂J/∂γ coupling on z is dropped
        n = flow_barrier.original_dynamics.state_dim
        grad_J_theta = grad_J[n:n + self._theta_flat_dim]

        Q = self._Q(x, theta, gamma)
        c = self._c(x, theta, gamma).at[:self._theta_flat_dim].add(grad_J_theta)

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

    # def _infeasible_fallback(self, v: jnp.ndarray) -> jnp.ndarray:
    #     """
    #     v = [omega, z] = [0, 1]: hold the plan and advance the planning time at
    #     real-time rate, so the predicted flow slides along itself. This
    #     direction is always admissible for the fixed-length window.
    #     """
    #     return jnp.zeros_like(v).at[-1].set(1.0)

    def _extend_control_bounds(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Extend control bounds to augmented action space for single state.
        u is not a decision variable, so the only bound row is z <= 1.

        Args:
            x: State (state_dim,)

        Returns:
            Tuple (G, h) for control bound constraints
        """
        G_extended = jnp.zeros((1, self._aug_action_dim)).at[0, -1].set(1.0)
        h_extended = jnp.ones(1)
        return G_extended, h_extended

    def get_applied_control(
            self,
            x: jnp.ndarray,
            theta: jnp.ndarray,
            gamma: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Get the applied physical control u = π(γ; x, θ). Batch with jax.vmap.

        Args:
            x: State vector (state_dim,)
            theta: Control parameters (action_dim, num_params)
            gamma: Time shift scalar

        Returns:
            Blended control (action_dim,)
        """
        return self._flow_barrier.blended_control(gamma, x, theta)
