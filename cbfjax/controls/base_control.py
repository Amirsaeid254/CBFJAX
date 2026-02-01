"""
Base classes for control systems.

This module provides base classes for implementing control algorithms
with JAX JIT-compatible immutable patterns.

Uses cooperative multiple inheritance pattern where all classes:
- Accept **kwargs and pass them up via super().__init__(**kwargs)
- Extract only the parameters they need

All controllers follow the stateful interface pattern (like Optax):
- _optimal_control_single(x, state) -> (u, new_state)
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
    get_trajs_from_state_action_func_zoh_no_vmap,
)
from ..utils.utils import ensure_batch_dim
from ..dynamics.base_dynamic import DummyDynamics



class BaseControl(eqx.Module):
    """
    Base class for control systems.

    This class provides the fundamental structure for control algorithms
    that optimize a given cost function.

    All subclasses implement the stateful interface:
    - _optimal_control_single(x, state) -> (u, new_state)
    - get_init_state() -> initial controller state

    Attributes:
        _dynamics: System dynamics object
        _action_dim: Dimension of control input
        _params: Configuration parameters
    """

    # Assigned components
    _dynamics: Any = eqx.field(static=True)

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

    @classmethod
    def create_empty(cls, action_dim: int, params: Optional[dict] = None):
        """
        Create an empty controller instance.

        Args:
            action_dim: Dimension of control input
            params: Optional configuration parameters

        Returns:
            Empty controller instance ready for component assignment
        """
        return cls(action_dim=action_dim, params=params)

    def _create_updated_instance(self, **kwargs):
        """
        Create new instance with updated fields.

        Args:
            **kwargs: Fields to update

        Returns:
            New instance of the same class with updated fields
        """
        defaults = {
            'action_dim': self._action_dim,
            'params': dict(self._params) if self._params else None,
            'dynamics': self._dynamics,
        }
        defaults.update(kwargs)
        return self.__class__(**defaults)

    def assign_dynamics(self, dynamics):
        """
        Assign dynamics to controller.

        Args:
            dynamics: System dynamics object

        Returns:
            New controller instance with assigned dynamics
        """
        return self._create_updated_instance(dynamics=dynamics)

    def get_init_state(self):
        """
        Get initial controller state.

        Returns:
            Initial state pytree (None for stateless controllers)
        """
        return None

    @abstractmethod
    def _optimal_control_single(self, x: jnp.ndarray, state=None) -> tuple:
        """
        Compute optimal control for a single state.

        This is the core method that concrete controller classes must implement.

        Args:
            x: Single state vector (state_dim,)
            state: Controller state (from get_init_state or previous call)

        Returns:
            Tuple (u, new_state) where:
            - u: Control vector (action_dim,)
            - new_state: Updated controller state
        """
        raise NotImplementedError

    def _optimal_control_single_with_info(self, x: jnp.ndarray, state=None) -> tuple:
        """
        Compute optimal control with diagnostic info for a single state.

        Default implementation calls _optimal_control_single and returns empty info.
        Subclasses can override to provide diagnostic information.

        Args:
            x: Single state vector (state_dim,)
            state: Controller state

        Returns:
            Tuple (u, new_state, info)
        """
        u, new_state = self._optimal_control_single(x, state)
        return u, new_state, {}

    def optimal_control(self, x: jnp.ndarray, state=None) -> tuple:
        """
        Compute optimal control with automatic batch support.

        Args:
            x: State(s) (state_dim,) or (batch, state_dim)
            state: Controller state (optional, uses get_init_state() if None)

        Returns:
            Tuple (u, new_state) with control(s)
        """
        if state is None:
            state = self.get_init_state()
        x_batched = ensure_batch_dim(x)
        return jax.vmap(self._optimal_control_single, in_axes=(0, None))(x_batched, state)

    def optimal_control_with_info(self, x: jnp.ndarray, state=None) -> tuple:
        """
        Compute optimal control with diagnostic info.

        Args:
            x: State(s) (state_dim,) or (batch, state_dim)
            state: Controller state (optional, uses get_init_state() if None)

        Returns:
            Tuple (u, new_state, info)
        """
        if state is None:
            state = self.get_init_state()
        x_batched = ensure_batch_dim(x)
        return jax.vmap(self._optimal_control_single_with_info, in_axes=(0, None))(x_batched, state)

    def _optimal_control_for_ode(self) -> Callable:
        """
        Create a stateless control function for ODE integration.

        Caches init_state once to avoid per-step overhead.

        Returns:
            Function x -> u for ODE integration
        """
        init_state = self.get_init_state()
        def control_for_ode(x):
            u, _ = self._optimal_control_single(x, init_state)
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
        return get_trajs_from_state_action_func(
            x0=x0,
            dynamics=self._dynamics,
            action_func=action_func,
            timestep=timestep,
            sim_time=sim_time,
            method=method
        )

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
            # Stateful controller: thread state through scan
            def stateful_action_func(x, ctrl_state):
                return self._optimal_control_single(x, ctrl_state)

            return get_trajs_from_state_action_func_zoh(
                x0=x0,
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

            return get_trajs_from_state_action_func_zoh(
                x0=x0,
                dynamics=self._dynamics,
                action_func=action_func,
                timestep=timestep,
                sim_time=sim_time,
                intermediate_steps=intermediate_steps,
                method=method,
                init_ctrl_state=None,
            )

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
            # Stateful controller: thread state through loop
            def stateful_action_func(x, ctrl_state):
                return self._optimal_control_single(x, ctrl_state)

            return get_trajs_from_state_action_func_zoh_no_vmap(
                x0=x0,
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

            return get_trajs_from_state_action_func_zoh_no_vmap(
                x0=x0,
                dynamics=self._dynamics,
                action_func=action_func,
                timestep=timestep,
                sim_time=sim_time,
                intermediate_steps=intermediate_steps,
                method=method,
                init_ctrl_state=None,
            )

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
    """

    # Type hints for fields (actual fields declared in using class)
    _Q: Optional[Callable]
    _R: Optional[Callable]
    _Q_e: Optional[Callable]
    _x_ref: Optional[Callable]

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
            Q: Callable returning state cost matrix (nx, nx)
            R: Callable returning control cost matrix (nu, nu)
            Q_e: Callable returning terminal cost matrix (nx, nx)
            x_ref: Callable returning reference state (nx,)
            **kwargs: Passed to next class in MRO
        """
        super().__init__(**kwargs)
        self._Q = Q
        self._R = R
        self._Q_e = Q_e
        self._x_ref = x_ref

    def assign_cost_matrices(
        self,
        Q: Callable,
        R: Callable,
        Q_e: Optional[Callable] = None,
        x_ref: Optional[Callable] = None
    ):
        """
        Assign quadratic cost matrices as Callable functions.

        Cost: (x - x_ref)^T Q (x - x_ref) + u^T R u

        Args:
            Q: Callable returning state cost matrix (nx, nx)
            R: Callable returning control cost matrix (nu, nu)
            Q_e: Callable returning terminal cost matrix (nx, nx), defaults to Q
            x_ref: Callable returning reference state (nx,)

        Returns:
            New instance with cost matrices assigned
        """
        if Q_e is None:
            Q_e = Q
        return self._create_updated_instance(Q=Q, R=R, Q_e=Q_e, x_ref=x_ref)

    def assign_reference(self, x_ref: Callable):
        """
        Assign reference state as Callable.

        Args:
            x_ref: Callable returning reference state (nx,)

        Returns:
            New instance with reference assigned
        """
        return self._create_updated_instance(x_ref=x_ref)

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
