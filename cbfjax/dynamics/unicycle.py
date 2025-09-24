import jax.numpy as jnp
from .base import AffineInControlDynamics


class UnicycleDynamics(AffineInControlDynamics):
    """Unicycle Dynamics"""
    
    def __init__(self, params=None, **kwargs):
        super().__init__(params, **kwargs)
        self._state_dim = 4
        self._action_dim = 2

    def _f(self, x):
        """
        x: (4,) = [x, y, v, theta]
        output: (4,) = [v*cos(theta), v*sin(theta), 0, 0]
        """
        return jnp.array([x[2] * jnp.cos(x[3]),
                          x[2] * jnp.sin(x[3]),
                          0.0,
                          0.0])

    def _g(self, x):
        """
        x: (4,) - state vector
        output: (4, 2) - control matrix  
        """
        return jnp.array([[0.0, 0.0],
                          [0.0, 0.0],
                          [1.0, 0.0], 
                          [0.0, 1.0]])

    def get_pos(self, x):
        """Get position from state"""
        return x[..., 0:2]

    def get_rot(self, x):
        """Get rotation from state"""
        return x[..., -1]
