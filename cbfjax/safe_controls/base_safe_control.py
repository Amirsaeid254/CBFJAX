"""
Base classes for safe control using Control Barrier Functions.

This module provides base classes for implementing safe control algorithms
that guarantee system safety through barrier function constraints.
Fixed to follow JAX JIT-compatible immutable patterns.
"""
import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Callable, Optional, Any, Union
from abc import abstractmethod
from immutabledict import immutabledict

from ..utils.integration import get_trajs_from_state_action_func, get_trajs_from_state_action_func_zoh
from ..utils.utils import ensure_batch_dim


class DummyBarrier:
    """
    Dummy barrier class for default initialization.

    Provides zero barrier value to avoid None values during object construction.
    Should only be used during the construction phase.
    """

    def hocbf(self, x: jnp.ndarray) -> jnp.ndarray:
        """Zero barrier function."""
        batch_size = x.shape[0] if x.ndim > 1 else 1
        return jnp.zeros((batch_size, 1))

    def get_hocbf_and_lie_derivs(self, x: jnp.ndarray):
        """Zero barrier and derivatives."""
        batch_size = x.shape[0] if x.ndim > 1 else 1
        action_dim = 1  # Default action dimension
        return (jnp.zeros((batch_size, 1)),
                jnp.zeros((batch_size, 1)),
                jnp.zeros((batch_size, action_dim)))


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

    # Assigned components - using dummy objects instead of None
    _dynamics: Any = eqx.field(static=True)
    _barrier: Any = eqx.field(static=True)
    _Q: Optional[Callable] = eqx.field(static=True)
    _c: Optional[Callable] = eqx.field(static=True)

    # Core configuration
    _action_dim: int = eqx.field(static=True)
    _alpha: Callable = eqx.field(static=True)
    _params: immutabledict = eqx.field(static=True)

    def __init__(self, action_dim: int, alpha: Optional[Callable] = None,
                 params: Optional[dict] = None, dynamics=None, barrier=None, Q=None, c=None):
        """
        Initialize BaseSafeControl.

        Args:
            action_dim: Dimension of control input
            alpha: Class-K function for barrier constraint (default: identity)
            params: Configuration parameters dictionary
            dynamics: System dynamics object (default: dummy)
            barrier: Barrier function object (default: dummy)
            Q: Cost matrix function (default: None)
            c: Cost vector function (default: None)
        """
        self._action_dim = action_dim
        self._alpha = alpha if alpha is not None else (lambda x: x)

        # Set default parameters and ensure buffer exists
        default_params = {'buffer': 0.0}
        if params is not None:
            default_params.update(params)
        self._params = immutabledict(default_params)

        # Initialize components with dummy objects instead of None
        self._dynamics = dynamics if dynamics is not None else DummyDynamics()
        self._barrier = barrier if barrier is not None else DummyBarrier()
        self._Q = Q
        self._c = c

    @staticmethod
    def _create_dummy_cost_Q():
        """Create dummy cost matrix function."""
        def dummy_Q(x):
            batch_size = x.shape[0] if x.ndim > 1 else 1
            action_dim = 1  # Default
            return jnp.eye(action_dim).reshape(1, action_dim, action_dim).repeat(batch_size, axis=0)
        return dummy_Q

    @staticmethod
    def _create_dummy_cost_c():
        """Create dummy cost vector function."""
        def dummy_c(x):
            batch_size = x.shape[0] if x.ndim > 1 else 1
            action_dim = 1  # Default
            return jnp.zeros((batch_size, action_dim))
        return dummy_c

    @classmethod
    def create_empty(cls, action_dim: int, alpha: Optional[Callable] = None,
                     params: Optional[dict] = None):
        """
        Create an empty safe controller instance.

        Args:
            action_dim: Dimension of control input
            alpha: Class-K function for barrier constraint
            params: Optional configuration parameters

        Returns:
            Empty controller instance ready for component assignment
        """
        return cls(action_dim=action_dim, alpha=alpha, params=params)

    def _create_updated_instance(self, **kwargs):
        """
        Create new instance with updated fields.

        This helper method enables clean assignment methods by collecting
        all current field values and updating only the changed ones.

        Args:
            **kwargs: Fields to update

        Returns:
            New instance of the same class with updated fields
        """
        defaults = {
            'action_dim': self._action_dim,
            'alpha': self._alpha,
            'params': dict(self._params) if self._params else None,
            'dynamics': self._dynamics,
            'barrier': self._barrier,
            'Q': self._Q,
            'c': self._c
        }
        defaults.update(kwargs)
        return self.__class__(**defaults)

    def assign_state_barrier(self, barrier):
        """
        Assign state barrier to controller.

        Args:
            barrier: Barrier function object for safety constraints

        Returns:
            New controller instance with assigned barrier
        """
        return self._create_updated_instance(barrier=barrier)

    def assign_dynamics(self, dynamics):
        """
        Assign dynamics to controller.

        Args:
            dynamics: System dynamics object

        Returns:
            New controller instance with assigned dynamics
        """
        return self._create_updated_instance(dynamics=dynamics)

    def assign_cost(self, Q: Callable, c: Callable):
        """
        Assign quadratic cost function.

        Args:
            Q: Function that computes cost matrix from state
            c: Function that computes cost vector from state

        Returns:
            New controller instance with assigned cost
        """
        return self._create_updated_instance(Q=Q, c=c)

    @abstractmethod
    def _safe_optimal_control_single(self, x: jnp.ndarray) -> tuple:
        """
        Compute safe optimal control for a single state.

        This is the core method that concrete controller classes must implement
        to define their specific safe control algorithm.

        Args:
            x: Single state vector (state_dim,)

        Returns:
            Tuple (u, info) where:
            - u: Control vector (action_dim,)
            - info: Dictionary containing additional information (e.g., slack_vars, constraint_at_u)
        """
        raise NotImplementedError

    def safe_optimal_control(self, x: jnp.ndarray) -> tuple:
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

        x_batched = ensure_batch_dim(x)


        return jax.vmap(self._safe_optimal_control_single)(x_batched)

    def _safe_optimal_control_for_ode(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Internal method for ODE integration - optimized for repeated calls.

        Args:
            x: State vector (state_dim,) - single state, not batched

        Returns:
            Control vector (action_dim,)
        """
        u, _ = self._safe_optimal_control_single(x)
        return u

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
        return get_trajs_from_state_action_func(
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
        return get_trajs_from_state_action_func_zoh(
            x0=x0,
            dynamics=self._dynamics,
            action_func=self._safe_optimal_control_for_ode,
            timestep=timestep,
            sim_time=sim_time,
            intermediate_steps=intermediate_steps,
            method=method
        )

    def _is_dummy_dynamics(self, dynamics) -> bool:
        """Check if dynamics is a dummy object."""
        return isinstance(dynamics, DummyDynamics)

    def _is_dummy_barrier(self, barrier) -> bool:
        """Check if barrier is a dummy object."""
        return isinstance(barrier, DummyBarrier)

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

    @property
    def has_dynamics(self) -> bool:
        """Check if real dynamics assigned."""
        return not self._is_dummy_dynamics(self._dynamics)

    @property
    def has_barrier(self) -> bool:
        """Check if real barrier assigned."""
        return not self._is_dummy_barrier(self._barrier)

    @property
    def has_cost(self) -> bool:
        """Check if cost functions assigned."""
        return self._Q is not None and self._c is not None


class BaseMinIntervSafeControl(BaseSafeControl):
    """
    Base class for minimum intervention safe control.

    This class extends BaseSafeControl for minimum intervention control,
    where the cost function is automatically defined as the deviation
    from a desired control policy.
    """

    # Additional field for desired control
    _desired_control: Optional[Callable] = eqx.field(static=True)

    def __init__(self, action_dim: int, alpha: Optional[Callable] = None,
                 params: Optional[dict] = None, desired_control: Optional[Callable] = None,
                 dynamics=None, barrier=None, Q=None, c=None):
        """
        Initialize BaseMinIntervSafeControl.

        Args:
            action_dim: Dimension of control input
            alpha: Class-K function for barrier constraint
            params: Configuration parameters dictionary
            desired_control: Desired control function
            dynamics: System dynamics object
            barrier: Barrier function object
            Q: Cost matrix function (should be None for min intervention)
            c: Cost vector function (should be None for min intervention)
        """
        super().__init__(action_dim, alpha, params, dynamics, barrier, Q, c)
        self._desired_control = desired_control

    @staticmethod
    def _create_dummy_desired_control():
        """Create dummy desired control function that returns zeros."""
        def dummy_desired_control(x):
            batch_size = x.shape[0] if x.ndim > 1 else 1
            action_dim = 1  # Default
            return jnp.zeros((batch_size, action_dim))
        return dummy_desired_control

    @classmethod
    def create_empty(cls, action_dim: int, alpha: Optional[Callable] = None,
                     params: Optional[dict] = None):
        """
        Create an empty minimum intervention controller instance.

        Args:
            action_dim: Dimension of control input
            alpha: Class-K function for barrier constraint
            params: Optional configuration parameters

        Returns:
            Empty controller instance ready for component assignment
        """
        return cls(action_dim=action_dim, alpha=alpha, params=params)

    def _create_updated_instance(self, **kwargs):
        """
        Create new instance with updated fields (extends parent).

        This helper method extends BaseSafeControl by adding desired_control field.

        Args:
            **kwargs: Fields to update

        Returns:
            New instance of the same class with updated fields
        """
        defaults = {
            'action_dim': self._action_dim,
            'alpha': self._alpha,
            'params': dict(self._params) if self._params else None,
            'dynamics': self._dynamics,
            'barrier': self._barrier,
            'Q': self._Q,
            'c': self._c,
            'desired_control': self._desired_control
        }
        defaults.update(kwargs)
        return self.__class__(**defaults)

    def assign_state_barrier(self, barrier):
        """
        Assign state barrier to controller.

        Args:
            barrier: Barrier function object

        Returns:
            New controller instance with assigned barrier
        """
        return self._create_updated_instance(barrier=barrier)

    def assign_dynamics(self, dynamics):
        """
        Assign dynamics to controller.

        Args:
            dynamics: System dynamics object

        Returns:
            New controller instance with assigned dynamics
        """
        return self._create_updated_instance(dynamics=dynamics)

    def assign_desired_control(self, desired_control: Callable):
        """
        Assign desired control function.

        Args:
            desired_control: Function that computes desired control

        Returns:
            New controller instance with assigned desired control
        """
        return self._create_updated_instance(desired_control=desired_control)

    def assign_cost(self, Q: jnp.ndarray, c: jnp.ndarray):
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
        # Add batch dimension for desired control computation
        x_batch = jnp.expand_dims(x, 0)
        u_batch = self._desired_control(x_batch)
        return jnp.squeeze(u_batch, axis=0)

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
        return get_trajs_from_state_action_func(
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
        return get_trajs_from_state_action_func_zoh(
            x0=x0,
            dynamics=self._dynamics,
            action_func=self._desired_control_for_ode,
            timestep=timestep,
            sim_time=sim_time,
            intermediate_steps=intermediate_steps,
            method=method
        )

    def _is_dummy_desired_control(self, desired_control) -> bool:
        """Check if desired control is dummy."""
        if desired_control is None:
            return True
        # Check if it has dummy function characteristics
        return (hasattr(desired_control, '__name__') and
                'dummy' in desired_control.__name__)

    @property
    def desired_control(self):
        """Get assigned desired control function."""
        return self._desired_control

    @property
    def has_desired_control(self) -> bool:
        """Check if real desired control assigned."""
        return self._desired_control is not None and not self._is_dummy_desired_control(self._desired_control)

    # Abstract method that concrete classes must implement
    @abstractmethod
    def _safe_optimal_control_single(self, x: jnp.ndarray) -> tuple:
        """
        Compute safe optimal control for a single state.

        Concrete minimum intervention controllers should implement this method
        to compute the control that minimizes deviation from desired control
        while satisfying safety constraints.

        Args:
            x: Single state vector (state_dim,)

        Returns:
            Tuple (u, info) where:
            - u: Control vector (action_dim,)
            - info: Dictionary containing additional information (e.g., slack_vars, constraint_at_u)
        """
        raise NotImplementedError