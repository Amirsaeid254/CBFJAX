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


class QPInfo(NamedTuple):
    """Diagnostic info from QP-based safe control."""
    slack_vars: jnp.ndarray
    constraint_at_u: jnp.ndarray


class BackupInfo(NamedTuple):
    """Diagnostic info from backup safe control."""
    constraint_at_u: jnp.ndarray
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
