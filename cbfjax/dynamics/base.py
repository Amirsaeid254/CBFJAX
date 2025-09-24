import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Optional, Callable, Dict, Any
from abc import abstractmethod
# Import configuration
from cbfjax.config import DEFAULT_DTYPE

from immutabledict import immutabledict


class AffineInControlDynamics(eqx.Module):
    """
    Base class for affine-in-control dynamics: dx/dt = f(x) + g(x) * u

    In JAX version, functions work on single examples.
    Use vmap for batching: jax.vmap(dynamics.f)(x_batch)
    """
    _state_dim: int
    _action_dim: int
    _params: Optional[Dict[str, Any]] = eqx.field(static=True)

    def __init__(self, params=None, **kwargs):
        self._params = immutabledict(params or {})  # ✅ Handle None case
        if "state_dim" in kwargs:
            self._state_dim = kwargs["state_dim"]
        if "action_dim" in kwargs:
            self._action_dim = kwargs["action_dim"]

    @property
    def state_dim(self):
        return self._state_dim

    @property
    def action_dim(self):
        return self._action_dim

    @property
    def params(self):
        return self._params

    def f(self, x):
        """
        Drift term of dynamics
        x: (state_dim,) - single state vector
        output: (state_dim,) - drift vector
        """
        assert x.shape == (self.state_dim,), f"Expected shape {(self.state_dim,)}, got {x.shape}"
        out = self._f(x)
        assert out.shape == x.shape, f"Output shape {out.shape} doesn't match input {x.shape}"
        return out

    def g(self, x):
        """
        Control matrix of dynamics
        x: (state_dim,) - single state vector
        output: (state_dim, action_dim) - control matrix
        """
        assert x.shape == (self.state_dim,), f"Expected shape {(self.state_dim,)}, got {x.shape}"
        out = self._g(x)
        expected_shape = (self.state_dim, self.action_dim)
        assert out.shape == expected_shape, f"Expected shape {expected_shape}, got {out.shape}"
        return out

    @abstractmethod
    def _f(self, x):
        """
        x: (state_dim,) - single state vector
        output: (state_dim,) - drift vector
        """
        raise NotImplementedError

    @abstractmethod
    def _g(self, x):
        """
        x: (state_dim,) - single state vector
        output: (state_dim, action_dim) - control matrix
        """
        raise NotImplementedError

    def rhs(self, x, action):
        """
        Right-hand-side of dynamics: f(x) + g(x) @ u
        x: (state_dim,) - single state vector
        action: (action_dim,) - single action vector
        output: (state_dim,) - derivative
        """
        assert action.shape == (self.action_dim,), f"Expected action shape {(self.action_dim,)}, got {action.shape}"
        return self.f(x) + self.g(x) @ action

    def set_f(self, f_func: Callable) -> 'AffineInControlDynamics':
        """
        Create new dynamics with different drift function.

        JAX-compatible version of CBFTorch's set_f using eqx.tree_at.
        Preserves original type and is more efficient than creating new objects.

        Args:
            f_func: New drift function (state,) -> (state_dim,)

        Returns:
            New dynamics instance with updated drift function (preserves original type)

        Example:
            def new_drift(x):
                return jnp.array([x[1], -x[0] + 0.1*jnp.sin(x[0])])

            new_dynamics = dynamics.set_f(new_drift)
        """
        if not callable(f_func):
            raise TypeError("f_func must be a callable function")

        return eqx.tree_at(lambda d: d._f, self, f_func)

    def set_g(self, g_func: Callable) -> 'AffineInControlDynamics':
        """
        Create new dynamics with different control matrix function.

        JAX-compatible version of CBFTorch's set_g using eqx.tree_at.
        Preserves original type and is more efficient than creating new objects.

        Args:
            g_func: New control matrix function (state,) -> (state_dim, action_dim)

        Returns:
            New dynamics instance with updated control matrix (preserves original type)

        Example:
            def new_control(x):
                return 2.0 * original_dynamics._g(x)  # Double the control gains

            new_dynamics = dynamics.set_g(new_control)
        """
        if not callable(g_func):
            raise TypeError("g_func must be a callable function")

        return eqx.tree_at(lambda d: d._g, self, g_func)

    def set_f_and_g(self, f_func: Callable, g_func: Callable) -> 'AffineInControlDynamics':
        """
        Create new dynamics with both drift and control functions changed.

        Convenient method to update both f and g at once using eqx.tree_at.

        Args:
            f_func: New drift function
            g_func: New control matrix function

        Returns:
            New dynamics instance with both functions updated (preserves original type)

        Example:
            new_dynamics = dynamics.set_f_and_g(my_f, my_g)
        """
        if not callable(f_func):
            raise TypeError("f_func must be a callable function")
        if not callable(g_func):
            raise TypeError("g_func must be a callable function")

        return eqx.tree_at(
            lambda d: (d._f, d._g),
            self,
            (f_func, g_func)
        )


class LowPassFilterDynamics(AffineInControlDynamics):
    """Low-pass filter dynamics"""
    _gains: jnp.ndarray
    _gains_mat: jnp.ndarray

    def __init__(self, params, state_dim, action_dim):
        assert state_dim == action_dim, "state_dim and action_dim should be the same"
        super().__init__(params=params, state_dim=state_dim, action_dim=action_dim)
        assert params is not None, "params should include low pass filter gains"
        assert (
                len(params["gains"]) == state_dim
        ), "gains should be a list of gains of length state_dim"

        self._gains = jnp.array(params["gains"], dtype=DEFAULT_DTYPE)
        self._gains_mat = jnp.diag(self._gains)

    def _f(self, x):
        """
        x: (state_dim,) - single state vector
        output: (state_dim,) - drift vector
        """
        return -self._gains * x

    def _g(self, x):
        """
        x: (state_dim,) - single state vector
        output: (state_dim, action_dim) - control matrix
        """
        return self._gains_mat