import jax.numpy as jnp
from .base_dynamic import AffineInControlDynamics

class DIDynamics(AffineInControlDynamics):
    """Double Integrator Dynamics"""

    def __init__(self, params=None, **kwargs):
        super().__init__(params, **kwargs)
        self._state_dim = 4
        self._action_dim = 2

    def _f(self, x):
        """
        x: (4,) = [x, y, vx, vy]
        output: (4,) = [vx, vy, 0, 0]
        """
        return jnp.array([x[2], x[3], 0.0, 0.0])

    def _g(self, x):
        """
        x: (4,) - state vector
        output: (4, 2) - control matrix
        """
        return jnp.array([[0.0, 0.0],
                          [0.0, 0.0],
                          [1.0, 0.0],
                          [0.0, 1.0]])