import jax.numpy as jnp
from .base_dynamic import AffineInControlDynamics
from immutabledict import immutabledict


class BicycleDynamics(AffineInControlDynamics):
    """
    Kinematic bicycle dynamics, NON-affine in control.

    State: x = [q_x, q_y, v, theta]
    Control: u = [u1 (acceleration), u2 (front steering angle)]

    rhs(x, u) = [v*cos(theta), v*sin(theta), u1, (v/l)*tan(u2)]

    Only rhs(x, u) is exposed; f/g are undefined since the model is
    non-affine in u.
    """

    def __init__(self, params=None, **kwargs):
        super().__init__(params, **kwargs)
        self._state_dim = 4
        self._action_dim = 2

        default_params = immutabledict({'l': 1.0})
        if params is None:
            self._params = default_params
        else:
            self._params = immutabledict({**default_params, **params})

    def _f(self, x):
        raise NotImplementedError(
            "BicycleDynamics is non-affine in control and only exposes "
            "rhs(x, u); f(x) is undefined."
        )

    def _g(self, x):
        raise NotImplementedError(
            "BicycleDynamics is non-affine in control and only exposes "
            "rhs(x, u); g(x) is undefined."
        )

    def rhs(self, x, action):
        """
        x: (4,) = [q_x, q_y, v, theta]
        action: (2,) = [acceleration, front steering angle]
        output: (4,) = [v*cos(theta), v*sin(theta), u1, (v/l)*tan(u2)]
        """
        if action.shape != (self.action_dim,):
            raise ValueError(f"Expected action shape {(self.action_dim,)}, got {action.shape}")
        return jnp.array([x[2] * jnp.cos(x[3]),
                          x[2] * jnp.sin(x[3]),
                          action[0],
                          (x[2] / self._params['l']) * jnp.tan(action[1])])

    def get_pos(self, x):
        """Get position from state"""
        return x[0:2]

    def get_rot(self, x):
        """Get rotation from state"""
        return x[3]