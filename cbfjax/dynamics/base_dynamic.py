import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Optional, Callable, Dict, Any
# Import configuration
from cbfjax.config import DEFAULT_DTYPE

from immutabledict import immutabledict


class AffineInControlDynamics(eqx.Module):
    """
    Base class for affine-in-control dynamics: dx/dt = f(x) + g(x) * u

    In JAX version, functions work on single examples.
    Use vmap for batching: jax.vmap(dynamics.f)(x_batch)

    Optional discretization via params:
        - 'discretization_dt': Timestep for discrete dynamics
        - 'discretization_method': 'euler' or 'rk4'
    """
    _state_dim: int
    _action_dim: int
    _params: Optional[Dict[str, Any]] = eqx.field(static=True)
    _dt: Optional[float] = eqx.field(static=True)
    _discretization_method: Optional[str] = eqx.field(static=True)
    _disturbance_func: Optional[Callable] = eqx.field(static=True)

    def __init__(self, params=None, **kwargs):
        self._params = immutabledict(params or {})
        if "state_dim" in kwargs:
            self._state_dim = kwargs["state_dim"]
        if "action_dim" in kwargs:
            self._action_dim = kwargs["action_dim"]

        # Optional discretization config from params
        self._dt = self._params.get('discretization_dt', None)
        self._discretization_method = self._params.get('discretization_method', None)
        self._disturbance_func = self._params.get('disturbance_func', None)

        if self._discretization_method is not None and self._discretization_method not in ('euler', 'rk4'):
            raise ValueError(f"Unknown discretization method: {self._discretization_method}. Use 'euler' or 'rk4'.")

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


    def _f(self, x):
        """
        x: (state_dim,) - single state vector
        output: (state_dim,) - drift vector
        """
        raise NotImplementedError


    def _g(self, x):
        """
        x: (state_dim,) - single state vector
        output: (state_dim, action_dim) - control matrix
        """
        raise NotImplementedError

    def rhs(self, x, action):
        """
        Nominal right-hand-side of dynamics: f(x) + g(x) @ u
        Used by barriers, controllers, and forward propagation.
        x: (state_dim,) - single state vector
        action: (action_dim,) - single action vector
        output: (state_dim,) - derivative
        """
        assert action.shape == (self.action_dim,), f"Expected action shape {(self.action_dim,)}, got {action.shape}"
        return self.f(x) + self.g(x) @ action

    def disturbed_rhs(self, x, action):
        """
        Disturbed right-hand-side: f(x) + g(x) @ u + d(x)
        For closed-loop simulation only. Falls back to nominal rhs if no disturbance set.
        x: (state_dim,) - single state vector
        action: (action_dim,) - single action vector
        output: (state_dim,) - derivative
        """
        nominal = self.rhs(x, action)
        if self._disturbance_func is None:
            return nominal
        return nominal + self._disturbance_func(x)

    def _euler_step(self, x, action, rhs_func=None):
        rhs_func = rhs_func or self.rhs
        return x + self._dt * rhs_func(x, action)

    def _rk4_step(self, x, action, rhs_func=None):
        rhs_func = rhs_func or self.rhs
        k1 = rhs_func(x, action)
        k2 = rhs_func(x + 0.5 * self._dt * k1, action)
        k3 = rhs_func(x + 0.5 * self._dt * k2, action)
        k4 = rhs_func(x + self._dt * k3, action)
        return x + (self._dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def discrete_rhs(self, x, action):
        """
        Nominal discrete dynamics: x_{k+1} = integrate(f(x) + g(x)@u)
        Used by controllers and forward propagation.
        """
        if self._discretization_method == 'euler':
            return self._euler_step(x, action)
        else:
            return self._rk4_step(x, action)

    def disturbed_discrete_rhs(self, x, action):
        """
        Disturbed discrete dynamics: x_{k+1} = integrate(f(x) + g(x)@u + d(x))
        For closed-loop simulation only. Falls back to nominal if no disturbance set.
        """
        if self._disturbance_func is None:
            return self.discrete_rhs(x, action)
        if self._discretization_method == 'euler':
            return self._euler_step(x, action, rhs_func=self.disturbed_rhs)
        else:
            return self._rk4_step(x, action, rhs_func=self.disturbed_rhs)


class CustomDynamics(AffineInControlDynamics):
    """
    Custom dynamics class where users provide f and g functions.

    This allows users to create custom dynamics by passing callable functions
    for the drift (f) and control matrix (g) without needing to subclass.
    """
    _f_func: Callable = eqx.field(static=True)
    _g_func: Callable = eqx.field(static=True)

    def __init__(self, state_dim: int, action_dim: int, f_func: Callable, g_func: Callable, params=None):
        """
        Initialize custom dynamics with user-provided functions.

        Args:
            state_dim: State dimension
            action_dim: Action/control dimension
            f_func: Drift function with signature f(x) -> (state_dim,)
            g_func: Control matrix function with signature g(x) -> (state_dim, action_dim)
            params: Optional parameters dictionary
        """
        super().__init__(params=params, state_dim=state_dim, action_dim=action_dim)
        self._f_func = f_func
        self._g_func = g_func

    def _f(self, x):
        """
        x: (state_dim,) - single state vector
        output: (state_dim,) - drift vector
        """
        return self._f_func(x)

    def _g(self, x):
        """
        x: (state_dim,) - single state vector
        output: (state_dim, action_dim) - control matrix
        """
        return self._g_func(x)


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


class DummyDynamics:
    """
    Dummy dynamics class for default initialization.

    Provides zero dynamics to avoid None values during object construction.
    Should only be used during the construction phase.
    """

    @property
    def state_dim(self) -> int:
        return 1

    @property
    def action_dim(self) -> int:
        return 1

    def f(self, x: jnp.ndarray) -> jnp.ndarray:
        """Zero drift dynamics."""
        return jnp.zeros_like(x)

    def g(self, x: jnp.ndarray) -> jnp.ndarray:
        """Zero control matrix."""
        return jnp.zeros((x.shape[0], 1))

    def rhs(self, x, action):
        """Zero right-hand side."""
        return jnp.zeros_like(x)
