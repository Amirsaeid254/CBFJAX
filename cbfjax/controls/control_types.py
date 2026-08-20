"""
State and info types for stateful controller interface.

All controller states are NamedTuples, which are JAX-native pytree types.
This follows the Optax pattern where controllers return (output, new_state)
and state is threaded through jax.lax.scan during integration.
"""
from typing import NamedTuple
import jax.numpy as jnp


# =============================================
# Controller States (threaded through scan)
# =============================================

class ILQRState(NamedTuple):
    """State for iLQR controller (warm-start trajectory)."""
    U: jnp.ndarray  # (N_horizon, action_dim)


class ConstrainedILQRState(NamedTuple):
    """State for constrained iLQR controller (warm-start trajectory)."""
    U: jnp.ndarray  # (N_horizon, action_dim)


# =============================================
# Controller Info (diagnostic, not threaded)
# =============================================

class ILQRInfo(NamedTuple):
    """Diagnostic info from iLQR solve."""
    objective: jnp.ndarray
    gradient: jnp.ndarray
    x_traj: jnp.ndarray
    u_traj: jnp.ndarray


class ConstrainedILQRInfo(NamedTuple):
    """Diagnostic info from constrained iLQR solve."""
    objective: jnp.ndarray
    gradient: jnp.ndarray
    max_constraint_violation: jnp.ndarray
    x_traj: jnp.ndarray
    u_traj: jnp.ndarray


class CFInfo(NamedTuple):
    """Diagnostic info from closed-form safe control."""
    slack_vars: jnp.ndarray
    constraint_at_u: jnp.ndarray
    u_desired: jnp.ndarray


class QPInfo(NamedTuple):
    """Diagnostic info from QP-based safe control."""
    slack_vars: jnp.ndarray
    constraint_at_u: jnp.ndarray
    u_desired: jnp.ndarray


class BackupInfo(NamedTuple):
    """Diagnostic info from backup safe control."""
    constraint_at_u: jnp.ndarray
    u_desired: jnp.ndarray
    u_star: jnp.ndarray
    ub_select: jnp.ndarray
    feas_fact: jnp.ndarray
    beta: jnp.ndarray


class NMPCInfo(NamedTuple):
    """Diagnostic info from NMPC solve."""
    status: jnp.ndarray
    cost: jnp.ndarray
    x_traj: jnp.ndarray
    u_traj: jnp.ndarray


class MPPIState(NamedTuple):
    """State for MPPI controller (warm-start trajectory + RNG key)."""
    U:   jnp.ndarray  # (N_horizon, action_dim) nominal control trajectory
    key: jnp.ndarray  # PRNGKey — threaded so the controller is a pure function


class CADPState(NamedTuple):
    """
    State for C-ADP controller: the cost-to-go parameters from the previous
    backward pass, which define the optimal functions used by the next forward
    pass, plus the update counter that paces re-planning.
    """
    P_next: jnp.ndarray  # (N, n, n) P_1 .. P_N
    T_next: jnp.ndarray  # (N, n)    T_1 .. T_N
    step:   jnp.ndarray  # ()        update counter


class MPPIInfo(NamedTuple):
    """Diagnostic info from MPPI solve."""
    S:       jnp.ndarray  # (K,)        per-sample total costs
    weights: jnp.ndarray  # (K,)        normalized importance weights
    U_new:   jnp.ndarray  # (N, m)      updated nominal trajectory (before warm-start shift)


class CADPInfo(NamedTuple):
    """Diagnostic info from a C-ADP solve."""
    slack_vars:      jnp.ndarray  # ()      the slack delta* of the applied step
    constraint_at_u: jnp.ndarray  # ()      a(x) + b(x)' u*, nonnegative by Theorem 1
    u_desired:       jnp.ndarray  # (l_v,)  performance control (lam = 0), the paper's v_d
    lam:             jnp.ndarray  # ()      constraint multiplier of the applied step
    nominal_traj:    jnp.ndarray  # (N+1, n) forward-pass nominal trajectory
