"""
Backup policies for unicycle dynamics in JAX.
"""

import jax
import jax.numpy as jnp


class UnicycleBackupControl:
    """
    Backup control policy for unicycle dynamics.

    Simple policy: brake as hard as possible to bring velocity to zero.

    Note: Policy function works on SINGLE states (state_dim,) as it's called inside vmap.
    """

    def __init__(self, gain, control_bounds):
        """
        Initialize backup control.

        Args:
            gain: Braking gain
            ac_lim: Action limits (2, 2) - [[u1_min, u1_max], [u2_min, u2_max]]
        """

        self.braking_gain = gain[0][0]
        self.ac_max = control_bounds[1][1]  # Max acceleration

    def __call__(self):
        """
        Return backup policy function.

        Returns:
            List containing single backup policy function taking state (state_dim,)
        """
        def brake_policy(x):
            """Backup policy: brake to bring velocity to zero."""
            # u1: brake based on velocity x[2], u2: no angular control
            u1 = self.ac_max * jnp.tanh(self.braking_gain * x[2])
            return jnp.array([u1, 0.0])

        # Return as list for compatibility with BackupBarrier expecting list of policies
        return [brake_policy]