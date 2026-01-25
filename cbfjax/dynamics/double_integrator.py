import jax.numpy as jnp
from .base_dynamic import AffineInControlDynamics


class DoubleIntegratorDynamics(AffineInControlDynamics):
    """
    Double Integrator Dynamics.

    State: [x, y, vx, vy] - position and velocity in 2D
    Control: [ax, ay] - acceleration in x and y
    """

    def __init__(self, params=None, **kwargs):
        super().__init__(params, **kwargs)
        self._state_dim = 4
        self._action_dim = 2

    def _f(self, x):
        """
        Drift dynamics.

        Args:
            x: (4,) = [x, y, vx, vy]

        Returns:
            (4,) = [vx, vy, 0, 0]
        """
        return jnp.array([x[2], x[3], 0.0, 0.0])

    def _g(self, x):
        """
        Control matrix.

        Args:
            x: (4,) - state vector

        Returns:
            (4, 2) - control matrix
        """
        return jnp.array([[0.0, 0.0],
                          [0.0, 0.0],
                          [1.0, 0.0],
                          [0.0, 1.0]])

    def get_pos(self, x):
        """Get position from state."""
        return x[0:2]

    def get_vel(self, x):
        """Get velocity from state."""
        return x[2:4]
