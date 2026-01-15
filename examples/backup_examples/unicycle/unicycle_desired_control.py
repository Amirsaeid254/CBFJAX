import jax.numpy as jnp


def desired_control(x, goal_pos, dyn_params, k1=0.8, k2=0.8):
    """
    Desired control for unicycle dynamics in backup examples.

    Note: This function works on SINGLE states (state_dim,) as it's called inside vmap.

    Args:
        x: State (4,) - [x, y, v, theta]
        goal_pos: Goal position (2,) - [x_goal, y_goal]
        dyn_params: Dynamics parameters dict with 'control_bounds' and 'd'
        k1: Control gain 1
        k2: Control gain 2

    Returns:
        Control (2,) - [u1, u2]
    """
    # Extract max control limits from tuple structure
    # control_bounds = ((min_u1, min_u2), (max_u1, max_u2))
    max_ac_lim = dyn_params['control_bounds'][1]  # (max_u1, max_u2)

    # State variables
    q_x, q_y, v, theta = x[0], x[1], x[2], x[3]

    # Rotation matrix from world to body frame
    s, c = jnp.sin(theta), jnp.cos(theta)
    rot_mat = jnp.array([[c, s], [-s, c]])

    # Distance to goal in body frame
    dist_to_goal = jnp.array([q_x, q_y]) - goal_pos
    e = rot_mat @ dist_to_goal

    # Control law
    vd = -(k1 + k2) * v - (1 + k1 * k2) * e[0] + jnp.pow(e[1] * k1, 2) / dyn_params['d']
    wd = -k1 / dyn_params['d'] * e[1]

    ud1 = max_ac_lim[0] * jnp.tanh(vd)
    ud2 = jnp.where(jnp.linalg.norm(dist_to_goal) > 0.1, max_ac_lim[1] * jnp.tanh(wd), 0.0)

    return jnp.array([ud1, ud2])
