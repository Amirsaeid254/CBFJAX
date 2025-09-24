"""
Base classes for safe control using Control Barrier Functions.

This module provides base classes for implementing safe control algorithms
that guarantee system safety through barrier function constraints.
"""

import jax.numpy as jnp
import equinox as eqx
from typing import Callable, Optional, Any, Union
from abc import abstractmethod
from immutabledict import immutabledict

from ..utils.integration import _safe_optimal_control_impl, get_trajs_from_action_func, get_trajs_from_action_func_zoh


class BaseSafeControl(eqx.Module):
    """
    Base class for safe control using Control Barrier Functions.

    This class provides the fundamental structure for safe control algorithms
    that guarantee system safety through barrier function constraints while
    optimizing a given cost function.

    Attributes:
        _dynamics: System dynamics object
        _barrier: Barrier function object for safety constraints
        _Q: Cost matrix function
        _c: Cost vector function
        _action_dim: Dimension of control input
        _alpha: Class-K function for barrier constraint
        _params: Configuration parameters
    """

    # Assigned components
    _dynamics: Optional[Any]
    _barrier: Optional[Any]
    _Q: Optional[Callable]
    _c: Optional[Callable]

    # Core configuration
    _action_dim: int = eqx.field(static=True)
    _alpha: Callable = eqx.field(static=True)
    _params: immutabledict = eqx.field(static=True)

    def __init__(self, action_dim: int, alpha: Optional[Callable] = None,
                 params: Optional[dict] = None):
        """
        Initialize BaseSafeControl.

        Args:
            action_dim: Dimension of control input
            alpha: Class-K function for barrier constraint (default: identity)
            params: Configuration parameters dictionary
        """
        self._action_dim = action_dim
        self._alpha = alpha if alpha is not None else (lambda x: x)

        # Set default parameters and ensure buffer exists
        default_params = {'buffer': 0.0}
        if params is not None:
            default_params.update(params)
        self._params = immutabledict(default_params)

        # Initialize assignable components as None
        self._dynamics = None
        self._barrier = None
        self._Q = None
        self._c = None

    @abstractmethod
    def _safe_optimal_control_single(self, x: jnp.ndarray, ret_info: bool = False) -> Union[jnp.ndarray, tuple]:
        """
        Compute safe optimal control for a single state.

        This is the core method that concrete controller classes must implement
        to define their specific safe control algorithm.

        Args:
            x: Single state vector (state_dim,)
            ret_info: Whether to return additional information

        Returns:
            Control vector (action_dim,) or tuple with additional info
        """
        raise NotImplementedError

    def safe_optimal_control(self, x: jnp.ndarray, ret_info: bool = False) -> Union[jnp.ndarray, tuple]:
        """
        Compute safe optimal control with automatic batch support.

        This is the main user-facing method that handles both single states
        and batches of states efficiently.

        Args:
            x: State(s) (state_dim,) or (batch, state_dim)
            ret_info: Whether to return additional information

        Returns:
            Control(s) with shape (batch, action_dim) or tuple with info
        """
        return _safe_optimal_control_impl(self, x, ret_info)

    def _safe_optimal_control_for_ode(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Internal method for ODE integration.

        Args:
            x: State vector (state_dim,) - single state, not batched

        Returns:
            Control vector (action_dim,)
        """
        return self._safe_optimal_control_single(x, ret_info=False)

    def assign_state_barrier(self, barrier) -> 'BaseSafeControl':
        """
        Assign state barrier to controller.

        Args:
            barrier: Barrier function object for safety constraints

        Returns:
            New controller instance with assigned barrier
        """
        raise NotImplementedError

    def assign_dynamics(self, dynamics) -> 'BaseSafeControl':
        """
        Assign dynamics to controller.

        Args:
            dynamics: System dynamics object

        Returns:
            New controller instance with assigned dynamics
        """
        raise NotImplementedError

    def assign_cost(self, Q: Callable, c: Callable) -> 'BaseSafeControl':
        """
        Assign quadratic cost function.

        Args:
            Q: Function that computes cost matrix from state
            c: Function that computes cost vector from state

        Returns:
            New controller instance with assigned cost
        """
        return self.__class__(
            action_dim=self._action_dim,
            alpha=self._alpha,
            params=dict(self._params),
            dynamics=self._dynamics,
            barrier=self._barrier,
            Q=Q,
            c=c
        )

    def get_safe_optimal_trajs(self, x0: jnp.ndarray, timestep: float = 0.001,
                              sim_time: float = 4.0, method: str = 'tsit5') -> jnp.ndarray:
        """
        Generate safe optimal trajectories using continuous integration.

        Args:
            x0: Initial states (batch, state_dim) or (state_dim,)
            timestep: Integration timestep
            sim_time: Total simulation time
            method: Integration method ('tsit5', 'euler', 'rk4', 'dopri5')

        Returns:
            Trajectories (time_steps, batch, state_dim)
        """
        return get_trajs_from_action_func(
            x0=x0,
            dynamics=self._dynamics,
            action_func=self._safe_optimal_control_for_ode,
            timestep=timestep,
            sim_time=sim_time,
            method=method
        )

    def get_safe_optimal_trajs_zoh(self, x0: jnp.ndarray, timestep: float = 0.001,
                                  sim_time: float = 4.0, intermediate_steps: int = 2,
                                  method: str = 'tsit5') -> jnp.ndarray:
        """
        Generate safe optimal trajectories using zero-order hold.

        Args:
            x0: Initial states (batch, state_dim) or (state_dim,)
            timestep: Control update timestep
            sim_time: Total simulation time
            intermediate_steps: Integration steps per control update
            method: Integration method

        Returns:
            Trajectories (time_steps, batch, state_dim)
        """
        return get_trajs_from_action_func_zoh(
            x0=x0,
            dynamics=self._dynamics,
            action_func=self._safe_optimal_control_for_ode,
            timestep=timestep,
            sim_time=sim_time,
            intermediate_steps=intermediate_steps,
            method=method
        )

    @property
    def barrier(self):
        """Get assigned barrier function object."""
        return self._barrier

    @property
    def dynamics(self):
        """Get assigned dynamics object."""
        return self._dynamics

    @property
    def action_dim(self) -> int:
        """Get control input dimension."""
        return self._action_dim


class BaseMinIntervSafeControl(BaseSafeControl):
    """
    Base class for minimum intervention safe control.

    This class extends BaseSafeControl for minimum intervention control,
    where the cost function is automatically defined as the deviation
    from a desired control policy.
    """

    # Additional field for desired control
    _desired_control: Optional[Callable]

    def __init__(self, action_dim: int, alpha: Optional[Callable] = None,
                 params: Optional[dict] = None, desired_control: Optional[Callable] = None):
        """
        Initialize BaseMinIntervSafeControl.

        Args:
            action_dim: Dimension of control input
            alpha: Class-K function for barrier constraint
            params: Configuration parameters dictionary
            desired_control: Desired control function
        """
        super().__init__(action_dim, alpha, params)
        self._desired_control = desired_control

    def assign_desired_control(self, desired_control: Callable) -> 'BaseMinIntervSafeControl':
        """
        Assign desired control function.

        Args:
            desired_control: Function that computes desired control

        Returns:
            New controller instance with assigned desired control
        """
        return self.__class__(
            action_dim=self._action_dim,
            alpha=self._alpha,
            params=dict(self._params),
            desired_control=desired_control
        )

    def assign_cost(self, Q: jnp.ndarray, c: jnp.ndarray) -> 'BaseMinIntervSafeControl':
        """
        Override to prevent manual cost assignment.

        Minimum intervention cost is automatically computed from the
        deviation from the desired control policy.

        Raises:
            ValueError: Always raised to direct users to assign_desired_control
        """
        raise ValueError(
            'Use assign_desired_control to assign desired control. '
            'The minimum intervention cost is automatically computed.'
        )

    def _desired_control_for_ode(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Internal method for ODE integration using desired control.

        Args:
            x: State vector (state_dim,) - single state, not batched

        Returns:
            Control vector (action_dim,)
        """
        if self._desired_control is None:
            raise ValueError("Desired control not assigned. Use assign_desired_control first.")
        return self._desired_control(x)

    def get_desired_control_trajs(self, x0: jnp.ndarray, timestep: float = 0.001,
                                 sim_time: float = 4.0, method: str = 'tsit5') -> jnp.ndarray:
        """
        Generate trajectories using desired control without safety constraints.

        Args:
            x0: Initial states (batch, state_dim) or (state_dim,)
            timestep: Integration timestep
            sim_time: Total simulation time
            method: Integration method

        Returns:
            Trajectories (time_steps, batch, state_dim)
        """
        return get_trajs_from_action_func(
            x0=x0,
            dynamics=self._dynamics,
            action_func=self._desired_control_for_ode,
            timestep=timestep,
            sim_time=sim_time,
            method=method
        )

    def get_desired_control_trajs_zoh(self, x0: jnp.ndarray, timestep: float = 0.001,
                                     sim_time: float = 4.0, intermediate_steps: int = 2,
                                     method: str = 'tsit5') -> jnp.ndarray:
        """
        Generate trajectories using desired control with zero-order hold.

        Args:
            x0: Initial states (batch, state_dim) or (state_dim,)
            timestep: Control update timestep
            sim_time: Total simulation time
            intermediate_steps: Integration steps per control update
            method: Integration method

        Returns:
            Trajectories (time_steps, batch, state_dim)
        """
        return get_trajs_from_action_func_zoh(
            x0=x0,
            dynamics=self._dynamics,
            action_func=self._desired_control_for_ode,
            timestep=timestep,
            sim_time=sim_time,
            intermediate_steps=intermediate_steps,
            method=method
        )

    @property
    def desired_control(self):
        """Get assigned desired control function."""
        return self._desired_control

    # Abstract method that concrete classes must implement
    @abstractmethod
    def _safe_optimal_control_single(self, x: jnp.ndarray, ret_info: bool = False) -> Union[jnp.ndarray, tuple]:
        """
        Compute safe optimal control for a single state.

        Concrete minimum intervention controllers should implement this method
        to compute the control that minimizes deviation from desired control
        while satisfying safety constraints.
        """
        raise NotImplementedError