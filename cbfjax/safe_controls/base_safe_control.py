"""
Base classes for safe control using Control Barrier Functions.

This module provides base classes for implementing safe control algorithms
that guarantee system safety through barrier function constraints.

All safe controllers follow the stateful interface:
- _optimal_control_single(x, state) -> (u, new_state)
- get_init_state() -> initial controller state
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
        return cls(action_dim=action_dim, params=params)

    def _create_updated_instance(self, **kwargs):
        defaults = {
            'action_dim': self._action_dim,
            'params': dict(self._params) if self._params else None,
            'dynamics': self._dynamics,
            'barrier': self._barrier,
        }
        defaults.update(kwargs)
        return self.__class__(**defaults)

    def assign_state_barrier(self, barrier):
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

    The _Q and _c callables follow the stateful pattern:
    - _Q: (x, state) -> (Q_matrix, new_state)
    - _c: (x, state) -> (c_vector, new_state)

    Uses cooperative multiple inheritance pattern.

    Attributes:
        _alpha: Class-K function for barrier constraint
        _Q: Stateful function (x, state) -> (Q_matrix, new_state)
        _c: Stateful function (x, state) -> (c_vector, new_state)
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
            Q: Stateful function (x, state) -> (Q_matrix, new_state),
               or simple function x -> Q_matrix (auto-wrapped)
            c: Stateful function (x, state) -> (c_vector, new_state),
               or simple function x -> c_vector (auto-wrapped)
            **kwargs: Passed to next class in MRO
        """
        super().__init__(**kwargs)
        self._alpha = alpha if alpha is not None else (lambda x: x)
        self._Q = Q
        self._c = c

    @classmethod
    def create_empty(cls, action_dim: int, alpha: Optional[Callable] = None,
                     params: Optional[dict] = None):
        return cls(action_dim=action_dim, alpha=alpha, params=params)

    def _create_updated_instance(self, **kwargs):
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

        Wraps plain x -> value functions to stateful (x, state) -> (value, state).

        Args:
            Q: Function x -> Q_matrix
            c: Function x -> c_vector

        Returns:
            New controller instance with assigned cost
        """
        def stateful_Q(x, state):
            return Q(x), state
        def stateful_c(x, state):
            return c(x), state
        return self._create_updated_instance(Q=stateful_Q, c=stateful_c)


class BaseMinIntervSafeControl(BaseCBFSafeControl):
    """
    Base class for minimum intervention safe control.

    Extends BaseCBFSafeControl with a desired control function.
    The desired control follows the stateful pattern:
    - _desired_control: (x, state) -> (u, new_state)

    Uses cooperative multiple inheritance pattern.

    Attributes:
        _desired_control: Stateful desired control function
        _desired_control_init_state: Callable returning init state for desired control
    """

    # Minimum intervention specific fields
    _desired_control: Optional[Callable] = eqx.field(static=True)
    _desired_control_init_state: Optional[Callable] = eqx.field(static=True)

    def __init__(self, desired_control: Optional[Callable] = None,
                 desired_control_init_state: Optional[Callable] = None, **kwargs):
        """
        Initialize BaseMinIntervSafeControl.

        Args:
            desired_control: Stateful desired control (x, state) -> (u, new_state),
                           or simple function x -> u (auto-wrapped by assign_desired_control)
            desired_control_init_state: Callable returning init state for desired control
            **kwargs: Passed to next class in MRO
        """
        super().__init__(**kwargs)
        self._desired_control = desired_control
        self._desired_control_init_state = desired_control_init_state

    @classmethod
    def create_empty(cls, action_dim: int, alpha: Optional[Callable] = None,
                     params: Optional[dict] = None):
        return cls(action_dim=action_dim, alpha=alpha, params=params)

    def _create_updated_instance(self, **kwargs):
        defaults = {
            'action_dim': self._action_dim,
            'alpha': self._alpha,
            'params': dict(self._params) if self._params else None,
            'dynamics': self._dynamics,
            'barrier': self._barrier,
            'Q': self._Q,
            'c': self._c,
            'desired_control': self._desired_control,
            'desired_control_init_state': self._desired_control_init_state,
        }
        defaults.update(kwargs)
        return self.__class__(**defaults)

    def get_init_state(self):
        """Get initial controller state (from desired controller if present)."""
        if self._desired_control_init_state is not None:
            return self._desired_control_init_state()
        return None

    def assign_desired_control(self, desired_control):
        """
        Assign desired control function.

        Accepts either:
        - A controller object with _optimal_control_single and get_init_state methods
        - A plain function f(x) -> u (wrapped to stateful form)
        - A stateful function f(x, state) -> (u, new_state)

        Args:
            desired_control: Controller object or callable

        Returns:
            New controller instance with assigned desired control
        """
        if hasattr(desired_control, '_optimal_control_single') and hasattr(desired_control, 'get_init_state'):
            # Controller object -> wrap to stateful function
            ctrl_obj = desired_control
            def stateful_desired(x, state):
                return ctrl_obj._optimal_control_single(x, state)
            init_state_fn = ctrl_obj.get_init_state
            return self._create_updated_instance(
                desired_control=stateful_desired,
                desired_control_init_state=init_state_fn,
            )
        else:
            func = desired_control
            def stateful_desired(x, state):
                return func(x), state
            return self._create_updated_instance(
                desired_control=stateful_desired,
                desired_control_init_state=lambda: None,
            )
