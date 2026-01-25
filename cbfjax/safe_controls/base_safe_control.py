"""
Base classes for safe control using Control Barrier Functions.

This module provides base classes for implementing safe control algorithms
that guarantee system safety through barrier function constraints.
"""
import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Callable, Optional, Any

from ..controls.base_control import BaseControl
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


class BaseSafeControl(BaseControl):
    """
    Base class for safe control with state constraints.

    Extends BaseControl with barrier function for safety guarantees.
    This is the base class for all safe control methods.

    Uses cooperative multiple inheritance pattern.

    Attributes:
        _barrier: Barrier function object for safety constraints
    """

    # Safety-specific field
    _barrier: Any = eqx.field(static=True)

    def __init__(self, barrier=None, **kwargs):
        """
        Initialize BaseSafeControl.

        Args:
            barrier: Barrier function object (default: dummy)
            **kwargs: Passed to next class in MRO (includes action_dim, params, dynamics)
        """
        # Add default buffer param
        params = kwargs.get('params', None)
        default_params = {'buffer': 0.0}
        if params is not None:
            default_params.update(params)
        kwargs['params'] = default_params

        super().__init__(**kwargs)
        self._barrier = barrier if barrier is not None else DummyBarrier()

    @classmethod
    def create_empty(cls, action_dim: int, params: Optional[dict] = None):
        """
        Create an empty safe controller instance.

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
            'barrier': self._barrier,
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

    def _is_dummy_barrier(self, barrier) -> bool:
        """Check if barrier is a dummy object."""
        return isinstance(barrier, DummyBarrier)

    @property
    def barrier(self):
        """Get assigned barrier function object."""
        return self._barrier

    @property
    def has_barrier(self) -> bool:
        """Check if real barrier assigned."""
        return not self._is_dummy_barrier(self._barrier)


class BaseCBFSafeControl(BaseSafeControl):
    """
    Base class for CBF-based safe control.

    Extends BaseSafeControl with class-K alpha function for
    Control Barrier Function constraints and quadratic cost.

    Uses cooperative multiple inheritance pattern.

    Attributes:
        _alpha: Class-K function for barrier constraint
        _Q: Function that computes cost matrix from state
        _c: Function that computes cost vector from state
    """

    # CBF-specific fields
    _alpha: Callable = eqx.field(static=True)
    _Q: Optional[Callable] = eqx.field(static=True)
    _c: Optional[Callable] = eqx.field(static=True)

    def __init__(
        self,
        alpha: Optional[Callable] = None,
        Q: Optional[Callable] = None,
        c: Optional[Callable] = None,
        **kwargs
    ):
        """
        Initialize BaseCBFSafeControl.

        Args:
            alpha: Class-K function for barrier constraint (default: identity)
            Q: Function that computes cost matrix from state
            c: Function that computes cost vector from state
            **kwargs: Passed to next class in MRO (includes action_dim, params, dynamics, barrier)
        """
        super().__init__(**kwargs)
        self._alpha = alpha if alpha is not None else (lambda x: x)
        self._Q = Q
        self._c = c

    @classmethod
    def create_empty(cls, action_dim: int, alpha: Optional[Callable] = None,
                     params: Optional[dict] = None):
        """
        Create an empty CBF safe controller instance.

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
        }
        defaults.update(kwargs)
        return self.__class__(**defaults)

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


class BaseMinIntervSafeControl(BaseCBFSafeControl):
    """
    Base class for minimum intervention safe control.

    Extends BaseCBFSafeControl with a desired control function.
    The cost is automatically computed as deviation from the desired control.

    Uses cooperative multiple inheritance pattern.

    Attributes:
        _desired_control: Function that computes desired control from state
    """

    # Minimum intervention specific field
    _desired_control: Optional[Callable] = eqx.field(static=True)

    def __init__(self, desired_control: Optional[Callable] = None, **kwargs):
        """
        Initialize BaseMinIntervSafeControl.

        Args:
            desired_control: Desired control function
            **kwargs: Passed to next class in MRO (includes action_dim, alpha, params, dynamics, barrier, Q, c)
        """
        super().__init__(**kwargs)
        self._desired_control = desired_control

    @classmethod
    def create_empty(cls, action_dim: int, alpha: Optional[Callable] = None,
                     params: Optional[dict] = None):
        """
        Create an empty minimum intervention safe controller instance.

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

    def assign_desired_control(self, desired_control: Callable):
        """
        Assign desired control function.

        Args:
            desired_control: Function that computes desired control

        Returns:
            New controller instance with assigned desired control
        """
        return self._create_updated_instance(desired_control=desired_control)