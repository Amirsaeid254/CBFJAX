"""
Base classes for control systems.

This module provides base classes for implementing control algorithms
with JAX JIT-compatible immutable patterns.
"""
import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Callable, Optional, Any
from abc import abstractmethod
from immutabledict import immutabledict

from ..utils.integration import get_trajs_from_state_action_func, get_trajs_from_state_action_func_zoh
from ..utils.utils import ensure_batch_dim
from ..dynamics.base_dynamic import DummyDynamics



class BaseControl(eqx.Module):
    """
    Base class for control systems.

    This class provides the fundamental structure for control algorithms
    that optimize a given cost function.

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

    def __init__(self, action_dim: int, params: Optional[dict] = None,
                 dynamics=None):
        """
        Initialize BaseControl.

        Args:
            action_dim: Dimension of control input
            params: Configuration parameters dictionary
            dynamics: System dynamics object (default: dummy)
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


    @abstractmethod
    def _optimal_control_single(self, x: jnp.ndarray) -> tuple:
        """
        Compute optimal control for a single state.

        This is the core method that concrete controller classes must implement.

        Args:
            x: Single state vector (state_dim,)

        Returns:
            Tuple (u, info) where:
            - u: Control vector (action_dim,)
            - info: Dictionary containing additional information
        """
        raise NotImplementedError

    def optimal_control(self, x: jnp.ndarray) -> tuple:
        """
        Compute optimal control with automatic batch support.

        Args:
            x: State(s) (state_dim,) or (batch, state_dim)

        Returns:
            Tuple (u, info) with control(s) shape (batch, action_dim)
        """
        x_batched = ensure_batch_dim(x)
        return jax.vmap(self._optimal_control_single)(x_batched)

    def _optimal_control_for_ode(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Internal method for ODE integration.

        Args:
            x: State vector (state_dim,) - single state, not batched

        Returns:
            Control vector (action_dim,)
        """
        u, _ = self._optimal_control_single(x)
        return u

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
        return get_trajs_from_state_action_func(
            x0=x0,
            dynamics=self._dynamics,
            action_func=self._optimal_control_for_ode,
            timestep=timestep,
            sim_time=sim_time,
            method=method
        )

    def get_optimal_trajs_zoh(self, x0: jnp.ndarray, timestep: float = 0.001,
                              sim_time: float = 4.0, intermediate_steps: int = 2,
                              method: str = 'tsit5') -> jnp.ndarray:
        """
        Generate optimal trajectories using zero-order hold.

        Args:
            x0: Initial states (batch, state_dim) or (state_dim,)
            timestep: Control update timestep
            sim_time: Total simulation time
            intermediate_steps: Integration steps per control update
            method: Integration method

        Returns:
            Trajectories (time_steps, batch, state_dim)
        """
        return get_trajs_from_state_action_func_zoh(
            x0=x0,
            dynamics=self._dynamics,
            action_func=self._optimal_control_for_ode,
            timestep=timestep,
            sim_time=sim_time,
            intermediate_steps=intermediate_steps,
            method=method
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

