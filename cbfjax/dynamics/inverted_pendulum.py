import jax.numpy as jnp
from .base_dynamic import AffineInControlDynamics


class InvertedPendulumDynamics(AffineInControlDynamics):
    """Inverted Pendulum Dynamics"""

    def __init__(self, params=None, **kwargs):
        super().__init__(params, **kwargs)
        self._state_dim = 2
        self._action_dim = 1

    def _f(self, x):
        """
        x: (2,) = [theta, theta_dot]
        output: (2,) = [theta_dot, sin(theta)]
        """
        return jnp.array([x[1],
                          jnp.sin(x[0])])

    def _g(self, x):
        """
        x: (2,) - state vector
        output: (2, 1) - control matrix
        """
        return jnp.array([[0.0],
                          [1.0]])

    def get_angle(self, x):
        """Get angle from state"""
        return x[0]

    def get_angular_velocity(self, x):
        """Get angular velocity from state"""
        return x[1]