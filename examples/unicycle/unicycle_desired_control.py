"""
Desired control for unicycle dynamics.
"""

import jax
import jax.numpy as jnp
from math import pi


def desired_control(x, goal_pos, k1=0.2, k2=1.0, k3=2.0):
    """
    Desired control for unicycle dynamics (single state).

    Args:
        x: State array (4,) - [q_x, q_y, v, theta]
        goal_pos: Goal position array (N, 2) - [goal_x, goal_y]
        k1, k2, k3: Control gains

    Returns:
        Control input array (2,) - [u1, u2]
    """
    dist_to_goal = jnp.linalg.norm(x[:2] - goal_pos[:2], axis=-1)
    q_x, q_y, v, theta = x[0], x[1], x[2], x[3]
    psi = jnp.arctan2(q_y - goal_pos[:, 1], q_x - goal_pos[:, 0]) - theta + pi

    ud1 = (-(k1 + k3) * v + (1 + k1 * k3) * dist_to_goal * jnp.cos(psi) +
           k1 * (k2 * dist_to_goal + v) * jnp.sin(psi) ** 2)

    ud2 = jnp.where(dist_to_goal > 0.1, (k2 + v / dist_to_goal) * jnp.sin(psi), 0.0)

    return jnp.column_stack([ud1, ud2]).squeeze(0)

import equinox as eqx


class UnicycleGoalControl(eqx.Module):
    """
    Unicycle goal-reaching desired control with parametric (leaf) goal and gains.

    Attributes:
        goal: Goal position (2,) - [goal_x, goal_y] (traced leaf)
        gains: Control gains (3,) - [k1, k2, k3] (traced leaf)
    """

    goal: jax.Array       # (2,)
    gains: jax.Array      # (k1, k2, k3)

    def __init__(self, goal, gains=(0.2, 1.0, 2.0)):
        self.goal = jnp.asarray(goal)
        self.gains = jnp.asarray(gains)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Desired control for a single state x = [q_x, q_y, v, theta].

        Returns control (2,) - [u1, u2].
        """
        k1, k2, k3 = self.gains[0], self.gains[1], self.gains[2]
        dist_to_goal = jnp.linalg.norm(x[:2] - self.goal[:2])
        q_x, q_y, v, theta = x[0], x[1], x[2], x[3]
        psi = jnp.arctan2(q_y - self.goal[1], q_x - self.goal[0]) - theta + pi

        ud1 = (-(k1 + k3) * v + (1 + k1 * k3) * dist_to_goal * jnp.cos(psi) +
               k1 * (k2 * dist_to_goal + v) * jnp.sin(psi) ** 2)

        ud2 = jnp.where(dist_to_goal > 0.1, (k2 + v / dist_to_goal) * jnp.sin(psi), 0.0)

        return jnp.stack([ud1, ud2])
