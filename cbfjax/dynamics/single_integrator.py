import jax.numpy as jnp
from .base_dynamic import AffineInControlDynamics


class SingleIntegratorDynamics(AffineInControlDynamics):
    """
    Single Integrator Dynamics: dx/dt = u

    State and action dimensions are equal. Defaults to 2D if dim not specified.
    """

    def __init__(self, dim=2, params=None, **kwargs):
        super().__init__(params, **kwargs)
        self._state_dim = dim
        self._action_dim = dim

    def _f(self, x):
        return jnp.zeros(self._state_dim)

    def _g(self, x):
        return jnp.eye(self._state_dim)

    def get_pos(self, x):
        """Get position from state"""
        return x[0:2]