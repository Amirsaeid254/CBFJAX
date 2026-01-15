import jax.numpy as jnp
from .base_dynamic import AffineInControlDynamics


class SingleIntegratorDynamics(AffineInControlDynamics):
    """Single Integrator Dynamics"""

    def __init__(self, params=None, **kwargs):
        super().__init__(params, **kwargs)
        self._state_dim = 2
        self._action_dim = 2

    def _f(self, x):
        """
        x: (2,) = [x, y]
        output: (2,) = [0, 0]
        """
        return jnp.zeros(2)

    def _g(self, x):
        """
        x: (2,) - state vector
        output: (2, 2) - control matrix (identity)
        """
        return jnp.eye(2)

    def get_pos(self, x):
        """Get position from state"""
        return x[0:2]