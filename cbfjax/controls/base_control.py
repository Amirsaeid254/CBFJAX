"""
Base classes for control systems.

This module provides base classes for implementing control algorithms
with JAX JIT-compatible immutable patterns.

Uses cooperative multiple inheritance pattern where all classes:
- Accept **kwargs and pass them up via super().__init__(**kwargs)
- Extract only the parameters they need

All controllers follow the stateful interface pattern (like Optax):
- optimal_control(x, state) -> (u, new_state)
- get_init_state() -> initial state pytree
- State is threaded through jax.lax.scan during ZOH integration
"""
import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Callable, Optional, Any
from abc import abstractmethod
from immutabledict import immutabledict

from ..utils.integration import (
    get_trajs_from_state_action_func,
    get_trajs_from_state_action_func_zoh,
)
from ..dynamics.base_dynamic import DummyDynamics



class BaseControl(eqx.Module):
    """
    Base class for control systems.

    This class provides the fundamental structure for control algorithms
    that optimize a given cost function.

    All subclasses implement the stateful interface:
    - optimal_control(x, state) -> (u, new_state)
    - get_init_state() -> initial controller state

    Attributes:
        _dynamics: System dynamics object
        _action_dim: Dimension of control input
        _params: Configuration parameters
    """

    # Assigned components
    _dynamics: Any

    # Core configuration
    _action_dim: int = eqx.field(static=True)
    _params: immutabledict = eqx.field(static=True)

    def __init__(
        self,
        action_dim: int,
        params: Optional[dict] = None,
        dynamics=None,
        **kwargs  # Accept and ignore remaining kwargs (end of chain)
    ):
        """
        Initialize BaseControl.

        Args:
            action_dim: Dimension of control input
            params: Configuration parameters dictionary
            dynamics: System dynamics object (default: dummy)
            **kwargs: Ignored (cooperative inheritance terminator)
        """


        self._action_dim = action_dim

        # Set default parameters
        default_params = {}
        if params is not None:
            default_params.update(params)
        self._params = immutabledict(default_params)

        # Initialize components with dummy objects instead of None
        self._dynamics = dynamics if dynamics is not None else DummyDynamics()

    def _ctor_defaults(self) -> dict:
        """Constructor kwargs capturing current field values (per-class)."""
        return {
            'action_dim': self._action_dim,
            'params': dict(self._params) if self._params else None,
            'dynamics': self._dynamics,
        }

    def _replace(self, **kwargs):
        """Rebuild instance through its constructor with updated fields."""
        defaults = self._ctor_defaults()
        defaults.update(kwargs)
        return self.__class__(**defaults)

    def get_init_state(self):
        """
        Get initial controller state.

        Returns:
            Initial state pytree (None for stateless controllers)
        """
        return None

    @abstractmethod
    def optimal_control(self, x: jnp.ndarray, state=None) -> tuple:
        """
        Compute optimal control for a single state.

        This is the core SINGLE-SAMPLE method that concrete controller classes
        must implement. To evaluate a batch, vmap at the call site:
        ``jax.vmap(filt.optimal_control, in_axes=(0, None))(x_batch, state)``.

        Args:
            x: Single state vector (state_dim,)
            state: Controller state (from get_init_state or previous call)

        Returns:
            Tuple (u, new_state) where:
            - u: Control vector (action_dim,)
            - new_state: Updated controller state
        """
        raise NotImplementedError

    def optimal_control_with_info(self, x: jnp.ndarray, state=None) -> tuple:
        """
        Compute optimal control with diagnostic info for a single state.

        Default implementation calls optimal_control and returns empty info.
        Subclasses can override to provide diagnostic information.

        Args:
            x: Single state vector (state_dim,)
            state: Controller state

        Returns:
            Tuple (u, new_state, info)
        """
        u, new_state = self.optimal_control(x, state)
        return u, new_state, {}

    def _optimal_control_for_ode(self) -> Callable:
        """
        Create a stateless control function for ODE integration.

        Caches init_state once to avoid per-step overhead.

        Returns:
            Function x -> u for ODE integration
        """
        init_state = self.get_init_state()
        def control_for_ode(x):
            u, _ = self.optimal_control(x, init_state)
            return u
        return control_for_ode

    def get_optimal_trajs(self, x0: jnp.ndarray, timestep: float = 0.001,
                          sim_time: float = 4.0, method: str = 'tsit5') -> jnp.ndarray:
        """
        Generate optimal trajectories using continuous integration.

        Args:
            x0: Initial states (batch, state_dim) or (state_dim,)
            timestep: Integration timestep
            sim_time: Total simulation time
            method: Integration method ('tsit5', 'euler', 'rk4', 'dopri5')

        Returns:
            Trajectories (time_steps, batch, state_dim)
        """
        action_func = self._optimal_control_for_ode()

        def single(x):
            return get_trajs_from_state_action_func(
                x0=x,
                dynamics=self._dynamics,
                action_func=action_func,
                timestep=timestep,
                sim_time=sim_time,
                method=method,
            )

        x0 = jnp.atleast_2d(x0)
        # (batch, time_steps, state_dim) -> (time_steps, batch, state_dim)
        trajs = jax.vmap(single)(x0)
        return jnp.swapaxes(trajs, 0, 1)

    def get_optimal_trajs_zoh(self, x0: jnp.ndarray, timestep: float = 0.001,
                              sim_time: float = 4.0, intermediate_steps: int = 2,
                              method: str = 'tsit5') -> jnp.ndarray:
        """
        Generate optimal trajectories using zero-order hold with state threading.

        Args:
            x0: Initial states (batch, state_dim) or (state_dim,)
            timestep: Control update timestep
            sim_time: Total simulation time
            intermediate_steps: Integration steps per control update
            method: Integration method

        Returns:
            Trajectories (time_steps, batch, state_dim)
        """
        init_ctrl_state = self.get_init_state()

        if init_ctrl_state is not None:
            # Stateful controller: thread state through scan (per-trajectory)
            def stateful_action_func(x, ctrl_state):
                return self.optimal_control(x, ctrl_state)

            def single(x):
                return get_trajs_from_state_action_func_zoh(
                    x0=x,
                    dynamics=self._dynamics,
                    action_func=stateful_action_func,
                    timestep=timestep,
                    sim_time=sim_time,
                    intermediate_steps=intermediate_steps,
                    method=method,
                    init_ctrl_state=init_ctrl_state,
                )
        else:
            # Stateless controller: use simple x -> u wrapper
            action_func = self._optimal_control_for_ode()

            def single(x):
                return get_trajs_from_state_action_func_zoh(
                    x0=x,
                    dynamics=self._dynamics,
                    action_func=action_func,
                    timestep=timestep,
                    sim_time=sim_time,
                    intermediate_steps=intermediate_steps,
                    method=method,
                    init_ctrl_state=None,
                )

        x0 = jnp.atleast_2d(x0)
        # (batch, time_steps, state_dim) -> (time_steps, batch, state_dim)
        trajs = jax.vmap(single)(x0)
        return jnp.swapaxes(trajs, 0, 1)

    def get_optimal_trajs_zoh_no_vmap(self, x0: jnp.ndarray, timestep: float = 0.001,
                                       sim_time: float = 4.0, intermediate_steps: int = 2,
                                       method: str = 'tsit5') -> jnp.ndarray:
        """
        Generate optimal trajectories using ZOH with Python loop (non-vmappable).

        Args:
            x0: Initial states (batch, state_dim) or (state_dim,)
            timestep: Control update timestep
            sim_time: Total simulation time
            intermediate_steps: Integration steps per control update
            method: Integration method

        Returns:
            Trajectories (time_steps, batch, state_dim)
        """
        init_ctrl_state = self.get_init_state()

        if init_ctrl_state is not None:
            # Stateful controller: thread state through the single-trajectory integrator
            def stateful_action_func(x, ctrl_state):
                return self.optimal_control(x, ctrl_state)

            def single(x):
                return get_trajs_from_state_action_func_zoh(
                    x0=x,
                    dynamics=self._dynamics,
                    action_func=stateful_action_func,
                    timestep=timestep,
                    sim_time=sim_time,
                    intermediate_steps=intermediate_steps,
                    method=method,
                    init_ctrl_state=init_ctrl_state,
                )
        else:
            # Stateless controller: use simple x -> u wrapper
            action_func = self._optimal_control_for_ode()

            def single(x):
                return get_trajs_from_state_action_func_zoh(
                    x0=x,
                    dynamics=self._dynamics,
                    action_func=action_func,
                    timestep=timestep,
                    sim_time=sim_time,
                    intermediate_steps=intermediate_steps,
                    method=method,
                    init_ctrl_state=None,
                )

        x0 = jnp.atleast_2d(x0)
        # Host Python loop over the batch (action_func may be non-vmappable, e.g. CVXOPT)
        trajectories = [single(x0[i]) for i in range(x0.shape[0])]
        # (time_steps, batch, state_dim)
        return jnp.stack(trajectories, axis=1)

    def _is_dummy_dynamics(self, dynamics) -> bool:
        """Check if dynamics is a dummy object."""
        return isinstance(dynamics, DummyDynamics)

    @property
    def dynamics(self):
        """Get assigned dynamics object."""
        return self._dynamics

    @property
    def action_dim(self) -> int:
        """Get control input dimension."""
        return self._action_dim

    @property
    def has_dynamics(self) -> bool:
        """Check if real dynamics assigned."""
        return not self._is_dummy_dynamics(self._dynamics)


class QuadraticCostMixin:
    """
    Mixin providing quadratic cost functionality.

    Uses cooperative multiple inheritance - extracts Q, R, Q_e, x_ref
    and passes remaining kwargs up the chain.

    Cost: (x - x_ref)^T Q (x - x_ref) + u^T R u

    Cost quantities use dual storage:
      - array inputs are stored in LEAF fields ``_<name>_value`` so they are
        traced and a controller stack can be built with ``eqx.filter_vmap``
        without per-member recompiles;
      - callable inputs are kept in static ``_<name>_func`` fields (escape hatch).
    A property per matrix (``_Q``/``_R``/``_Q_e``/``_x_ref``) exposes a zero-arg
    callable interface for downstream code.
    """

    # Type hints for paired fields (actual fields declared in using class)
    _Q_value: Optional[jax.Array]
    _R_value: Optional[jax.Array]
    _Q_e_value: Optional[jax.Array]
    _x_ref_value: Optional[jax.Array]
    _Q_func: Optional[Callable]
    _R_func: Optional[Callable]
    _Q_e_func: Optional[Callable]
    _x_ref_func: Optional[Callable]

    def __init__(
        self,
        Q: Optional[Callable] = None,
        R: Optional[Callable] = None,
        Q_e: Optional[Callable] = None,
        x_ref: Optional[Callable] = None,
        **kwargs
    ):
        """
        Initialize QuadraticCostMixin.

        Args:
            Q: Callable returning state cost matrix (nx, nx), or the matrix itself
            R: Callable returning control cost matrix (nu, nu), or the matrix itself
            Q_e: Callable returning terminal cost matrix (nx, nx), or the matrix itself
            x_ref: Callable returning reference state (nx,), or the vector itself
            **kwargs: Passed to next class in MRO
        """
        super().__init__(**kwargs)
        self._Q_value, self._Q_func = self._split_cost(Q)
        self._R_value, self._R_func = self._split_cost(R)
        self._Q_e_value, self._Q_e_func = self._split_cost(Q_e)
        self._x_ref_value, self._x_ref_func = self._split_cost(x_ref)

    @staticmethod
    def _split_cost(value):
        """Split into (leaf_array, static_callable). Arrays become traced leaves."""
        if value is None:
            return None, None
        if callable(value):
            return None, value
        return jnp.asarray(value), None

    @staticmethod
    def _cost_emit(value, func):
        """Round-trip kwarg: the raw array or the callable, whichever is set."""
        if value is not None:
            return value
        return func

    @staticmethod
    def _make_const_func(arr):
        def const_func():
            return arr
        return const_func

    def _cost_callable(self, value, func):
        """Zero-arg callable interface over the dual storage (None if unset)."""
        if value is not None:
            return self._make_const_func(value)
        return func

    @property
    def _Q(self):
        return self._cost_callable(self._Q_value, self._Q_func)

    @property
    def _R(self):
        return self._cost_callable(self._R_value, self._R_func)

    @property
    def _Q_e(self):
        return self._cost_callable(self._Q_e_value, self._Q_e_func)

    @property
    def _x_ref(self):
        return self._cost_callable(self._x_ref_value, self._x_ref_func)

    def _get_quadratic_cost_func(self) -> Callable:
        """
        Build quadratic cost function from Q, R matrices.

        Returns:
            Cost function f(x, u, t) -> scalar
        """
        assert self._Q is not None and self._R is not None, "Cost matrices must be assigned"

        Q = self._Q()
        R = self._R()
        Q_e = self._Q_e() if self._Q_e is not None else Q
        T = self.N_horizon
        x_ref = self._x_ref() if self._x_ref is not None else jnp.zeros(Q.shape[0])

        def cost(x, u, t):
            x_err = x - x_ref
            return jax.lax.cond(
                t == T,
                lambda: 0.5 * x_err @ Q_e @ x_err,
                lambda: 0.5 * x_err @ Q @ x_err + 0.5 * u @ R @ u
            )

        return cost
