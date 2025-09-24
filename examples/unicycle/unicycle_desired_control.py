"""
JAX implementation of unicycle desired control.
Converted from PyTorch to JAX with JIT compilation for performance.
"""

import jax
import jax.numpy as jnp
from math import pi


@jax.jit
def desired_control(x, goal_pos, k1=0.2, k2=1.0, k3=2.0):
    """
    JIT-compiled desired control for unicycle dynamics.

    Args:
        x: State tensor (batch, 4) - [q_x, q_y, v, theta]
        goal_pos: Goal position tensor (batch, 2) - [goal_x, goal_y]
        k1, k2, k3: Control gains

    Returns:
        Control input tensor (batch, 2) - [u1, u2]
    """
    dist_to_goal = jnp.linalg.norm(x[:, :2] - goal_pos[:, :2], axis=-1)
    q_x, q_y, v, theta = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
    psi = jnp.arctan2(q_y - goal_pos[:, 1], q_x - goal_pos[:, 0]) - theta + pi

    ud1 = (-(k1 + k3) * v + (1 + k1 * k3) * dist_to_goal * jnp.cos(psi) +
           k1 * (k2 * dist_to_goal + v) * jnp.sin(psi) ** 2)

    ud2 = jnp.where(dist_to_goal > 0.1, (k2 + v / dist_to_goal) * jnp.sin(psi), 0.0)

    return jnp.column_stack([ud1, ud2])