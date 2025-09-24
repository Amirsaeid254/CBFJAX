"""
Desired control functions for CBF-JAX.

This module implements various desired control laws that can be used
as the nominal control in safe control frameworks.
"""

import jax.numpy as jnp
from typing import Tuple
from math import pi


def unicycle_desired_control(x: jnp.ndarray, goal_pos: jnp.ndarray,
                           k1: float = 0.2, k2: float = 1.0, k3: float = 2.0) -> jnp.ndarray:
    """
    Desired control for unicycle to reach a goal position.

    This implements a nonlinear control law that drives the unicycle
    to a desired goal position with exponential convergence.

    Args:
        x: State [x, y, v, theta] of shape (4,)
        goal_pos: Goal position [x_g, y_g] of shape (2,)
        k1, k2, k3: Control gains

    Returns:
        Desired control [a, omega] of shape (2,)
    """
    # Extract state variables
    q_x, q_y, v, theta = x[0], x[1], x[2], x[3]

    # Extract goal position
    goal_x, goal_y = goal_pos[0], goal_pos[1]

    # Distance to goal
    dist_to_goal = jnp.sqrt((q_x - goal_x) ** 2 + (q_y - goal_y) ** 2)

    # Angle from current position to goal, adjusted by heading
    psi = jnp.arctan2(q_y - goal_y, q_x - goal_x) - theta + pi

    # Desired acceleration (first control input)
    ud1 = (-(k1 + k3) * v + (1 + k1 * k3) * dist_to_goal * jnp.cos(psi) +
           k1 * (k2 * dist_to_goal + v) * jnp.sin(psi) ** 2)

    # Desired angular velocity (second control input)
    # Use conditional to avoid division by zero when very close to goal
    ud2 = jnp.where(
        dist_to_goal > 0.1,
        (k2 + v / dist_to_goal) * jnp.sin(psi),
        0.0
    )

    return jnp.array([ud1, ud2])

