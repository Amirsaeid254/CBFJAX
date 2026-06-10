import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Optional, Callable, Dict, Any, List
from cbfjax.config import get_default_dtype

from immutabledict import immutabledict


class AffineInControlDynamics(eqx.Module):
    """
    Base class for affine-in-control dynamics: dx/dt = f(x) + g(x) * u


    Use vmap for batching: jax.vmap(dynamics.f)(x_batch)

    Optional discretization via params:
        - 'discretization_dt': Timestep for discrete dynamics
        - 'discretization_method': 'euler' or 'rk4'
    """
    _state_dim: int = eqx.field(static=True)
    _action_dim: int = eqx.field(static=True)
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
        if x.shape != (self._state_dim,):
            raise ValueError(f"Expected state shape {(self._state_dim,)}, got {x.shape}")
        return self._f(x)

    def g(self, x):
        """
        Control matrix of dynamics
        x: (state_dim,) - single state vector
        output: (state_dim, action_dim) - control matrix
        """
        if x.shape != (self._state_dim,):
            raise ValueError(f"Expected state shape {(self._state_dim,)}, got {x.shape}")
        return self._g(x)


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
        if action.shape != (self.action_dim,):
            raise ValueError(f"Expected action shape {(self.action_dim,)}, got {action.shape}")
        return self.f(x) + self.g(x) @ action

    def disturbed_rhs(self, x, action):
        """
        Disturbed right-hand-side: f(x) + g(x) @ u + d(x, u)
        For closed-loop simulation only. Falls back to nominal rhs if no disturbance set.
        x: (state_dim,) - single state vector
        action: (action_dim,) - single action vector
        output: (state_dim,) - derivative
        """
        nominal = self.rhs(x, action)
        if self._disturbance_func is None:
            return nominal
        return nominal + self._disturbance_func(x, action)

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

        self._gains = jnp.array(params["gains"], dtype=get_default_dtype())
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


def create_augmented_dynamics(dynamics_list: List['AffineInControlDynamics']) -> 'CustomDynamics':
    """
    Create block-diagonal augmented dynamics from a list of dynamics.

    Given dynamics [dyn_1, dyn_2, ..., dyn_n], produces:
        state  = [x1, x2, ..., xn]
        action = [u1, u2, ..., un]
        f_aug(s) = [f1(x1), f2(x2), ..., fn(xn)]
        g_aug(s) = block_diag(g1(x1), g2(x2), ..., gn(xn))
        d_aug(s, u) = [d1(x1,u1), d2(x2,u2), ..., dn(xn,un)]

    Disturbance is automatically propagated: if any sub-dynamics has a
    disturbance_func, the augmented dynamics will have one too.
    The Python for loop is unrolled at trace time — JIT-compatible since
    the dynamics list is static (fixed at construction).

    Args:
        dynamics_list: List of AffineInControlDynamics instances

    Returns:
        CustomDynamics with block-diagonal structure
    """
    assert len(dynamics_list) > 0, "dynamics_list must be non-empty"

    # Precompute dimensions and split indices (all static)
    n = len(dynamics_list)
    state_dims = tuple(d.state_dim for d in dynamics_list)
    action_dims = tuple(d.action_dim for d in dynamics_list)
    total_state_dim = sum(state_dims)
    total_action_dim = sum(action_dims)

    # Cumulative split points (static tuples)
    state_splits = []
    s_acc = 0
    for sd in state_dims:
        state_splits.append((s_acc, s_acc + sd))
        s_acc += sd
    state_splits = tuple(state_splits)

    action_splits = []
    a_acc = 0
    for ad in action_dims:
        action_splits.append((a_acc, a_acc + ad))
        a_acc += ad
    action_splits = tuple(action_splits)

    # Capture functions to avoid closure issues with mutable objects
    f_funcs = tuple(d.f for d in dynamics_list)
    g_funcs = tuple(d.g for d in dynamics_list)

    def aug_f(s):
        parts = [f_funcs[i](s[state_splits[i][0]:state_splits[i][1]]) for i in range(n)]
        return jnp.concatenate(parts)

    def aug_g(s):
        g_aug = jnp.zeros((total_state_dim, total_action_dim))
        for i in range(n):
            si, se = state_splits[i]
            ai, ae = action_splits[i]
            g_aug = g_aug.at[si:se, ai:ae].set(g_funcs[i](s[si:se]))
        return g_aug

    # Build augmented disturbance if any sub-dynamics has one
    disturbance_funcs = tuple(d._disturbance_func for d in dynamics_list)
    has_any_disturbance = any(df is not None for df in disturbance_funcs)

    params = {}
    if has_any_disturbance:
        def aug_disturbance(s, action):
            parts = []
            for i in range(n):
                si, se = state_splits[i]
                ai, ae = action_splits[i]
                if disturbance_funcs[i] is not None:
                    parts.append(disturbance_funcs[i](s[si:se], action[ai:ae]))
                else:
                    parts.append(jnp.zeros(state_dims[i]))
            return jnp.concatenate(parts)
        params['disturbance_func'] = aug_disturbance

    return CustomDynamics(
        state_dim=total_state_dim,
        action_dim=total_action_dim,
        f_func=aug_f,
        g_func=aug_g,
        params=params if params else None
    )


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
