"""
Base classes for safe control using Control Barrier Functions.

This module provides base classes for implementing safe control algorithms
that guarantee system safety through barrier function constraints.

All safe controllers follow the stateful interface:
- optimal_control(x, state) -> (u, new_state)
- get_init_state() -> initial controller state
"""
import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Callable, Optional, Any

from ..controls.base_control import BaseControl


class DummyBarrier:
    """
    Dummy barrier class for default initialization.

    Provides zero barrier value to avoid None values during object construction.
    Should only be used during the construction phase.
    """

    def hocbf(self, x: jnp.ndarray) -> jnp.ndarray:
        """Zero barrier function (single state)."""
        return jnp.zeros(())

    def get_hocbf_and_lie_derivs(self, x: jnp.ndarray):
        """Zero barrier and derivatives (single state)."""
        action_dim = 1  # Default action dimension
        return jnp.zeros(()), jnp.zeros(()), jnp.zeros(action_dim)


class BaseSafeControl(BaseControl):
    """
    Base class for safe control with state constraints.

    Extends BaseControl with barrier function for safety guarantees.
    This is the base class for all safe control methods.

    Uses cooperative multiple inheritance pattern.

    Attributes:
        _barrier: Barrier function object for safety constraints
        _terminal_barrier: Optional terminal barrier for end-of-horizon constraint
    """

    # Safety-specific fields
    _barrier: Any
    _terminal_barrier: Any

    def __init__(self, barrier=None, terminal_barrier=None, **kwargs):
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
        self._terminal_barrier = terminal_barrier

    def _ctor_defaults(self) -> dict:
        return {
            'action_dim': self._action_dim,
            'params': dict(self._params) if self._params else None,
            'dynamics': self._dynamics,
            'barrier': self._barrier,
            'terminal_barrier': self._terminal_barrier,
        }

    def _is_dummy_barrier(self, barrier) -> bool:
        """Check if barrier is a dummy object."""
        return isinstance(barrier, DummyBarrier)

    @property
    def barrier(self):
        """Get assigned barrier function object."""
        return self._barrier

    @property
    def terminal_barrier(self):
        """Get assigned terminal barrier."""
        return self._terminal_barrier

    @property
    def has_barrier(self) -> bool:
        """Check if real barrier assigned."""
        return not self._is_dummy_barrier(self._barrier)

    @property
    def has_terminal_barrier(self) -> bool:
        """Check if terminal barrier is assigned."""
        return self._terminal_barrier is not None


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
        cost: Optional[tuple] = None,
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
            cost: Optional (Q, c) tuple of plain x -> value functions,
                  wrapped to stateful form
            **kwargs: Passed to next class in MRO
        """
        super().__init__(**kwargs)
        self._alpha = alpha if alpha is not None else (lambda x: x)
        if cost is not None and Q is None and c is None:
            Q, c = self._wrap_plain_cost(*cost)
        self._Q = Q
        self._c = c

    @staticmethod
    def _wrap_plain_cost(Q: Callable, c: Callable) -> tuple:
        """Wrap plain x -> value cost functions to stateful form."""
        def stateful_Q(x, state):
            return Q(x), state
        def stateful_c(x, state):
            return c(x), state
        return stateful_Q, stateful_c

    def _ctor_defaults(self) -> dict:
        return {
            'action_dim': self._action_dim,
            'alpha': self._alpha,
            'params': dict(self._params) if self._params else None,
            'dynamics': self._dynamics,
            'barrier': self._barrier,
            'terminal_barrier': self._terminal_barrier,
            'Q': self._Q,
            'c': self._c,
        }


class BaseMinIntervSafeControl(BaseCBFSafeControl):
    """
    Base class for minimum intervention safe control.

    Extends BaseCBFSafeControl with a desired control function.
    The desired control follows the stateful pattern:
    - _desired_control: (x, state) -> (u, new_state)

    Uses cooperative multiple inheritance pattern.

    Attributes:
        _desired_control: Stateful desired control function (property over dual storage)
        _desired_control_module: eqx.Module desired control kept as a traced LEAF
        _desired_control_static: stateful desired control for plain callables (static)
        _desired_control_init_state: Callable returning init state for desired control

    Dual storage (mirrors Barrier._alpha_coefs): a plain callable / controller is
    normalized to a stateful function and kept in the static field
    ``_desired_control_static`` (it cannot be a leaf - subclasses jit on ``self``,
    so a raw lambda leaf would crash jit). An ``eqx.Module`` desired control is
    kept as a traced LEAF in ``_desired_control_module`` so its parameters (e.g.
    a goal) stay traced and the controller can be vmapped/tree_at-ed without
    recompiles. The ``_desired_control`` property reconstructs the stateful
    ``(x, state) -> (u, new_state)`` interface from whichever is set.
    """

    # Minimum intervention specific fields (dual storage)
    _desired_control_module: Optional[Any]
    _desired_control_module_stateful: bool = eqx.field(static=True)
    _desired_control_static: Optional[Callable] = eqx.field(static=True)
    _desired_control_init_state: Optional[Callable] = eqx.field(static=True)

    def __init__(self, desired_control: Optional[Callable] = None,
                 desired_control_init_state: Optional[Callable] = None, **kwargs):
        """
        Initialize BaseMinIntervSafeControl.

        Args:
            desired_control: Controller object with optimal_control and
                           get_init_state, an eqx.Module mapping x -> u (kept as a
                           traced leaf), or plain function x -> u (normalized to
                           stateful form), or already-stateful function when
                           desired_control_init_state is also given
            desired_control_init_state: Callable returning init state for desired control
            **kwargs: Passed to next class in MRO
        """
        super().__init__(**kwargs)
        module, module_stateful, static = None, False, None
        if desired_control is None:
            pass
        elif isinstance(desired_control, eqx.Module):
            # ANY eqx.Module (plain x -> u callable OR stateful controller
            # object) is kept as a traced LEAF so its parameter arrays stay
            # traced (no array-as-static warning) and the controller can be
            # vmapped/tree_at-ed. A static bool records the calling convention.
            module = desired_control
            module_stateful = (hasattr(desired_control, 'optimal_control')
                               and hasattr(desired_control, 'get_init_state'))
            if module_stateful:
                # init state is produced from the leaf module in get_init_state,
                # so the module is NOT captured by a static closure here.
                pass
            elif desired_control_init_state is None:
                desired_control_init_state = (lambda: None)
        elif desired_control_init_state is not None:
            # Already-stateful plain function supplied with its init state.
            static = desired_control
        else:
            # Plain function / lambda x -> u: must stay static (jit constraint).
            static, desired_control_init_state = \
                self._normalize_desired_control(desired_control)
        self._desired_control_module = module
        self._desired_control_module_stateful = module_stateful
        self._desired_control_static = static
        self._desired_control_init_state = desired_control_init_state

    @property
    def _desired_control(self) -> Optional[Callable]:
        """Stateful (x, state) -> (u, new_state) over the dual storage."""
        if self._desired_control_module is not None:
            module = self._desired_control_module
            if self._desired_control_module_stateful:
                def stateful_desired(x, state):
                    return module.optimal_control(x, state)
                return stateful_desired
            def stateful_desired(x, state):
                return module(x), state
            return stateful_desired
        return self._desired_control_static

    @staticmethod
    def _normalize_desired_control(desired_control) -> tuple:
        """
        Normalize desired control to (stateful_fn, init_state_fn).

        Accepts a controller object (with optimal_control and
        get_init_state) or a plain function f(x) -> u.
        """
        if hasattr(desired_control, 'optimal_control') and hasattr(desired_control, 'get_init_state'):
            ctrl_obj = desired_control
            def stateful_desired(x, state):
                return ctrl_obj.optimal_control(x, state)
            return stateful_desired, ctrl_obj.get_init_state
        func = desired_control
        def stateful_desired(x, state):
            return func(x), state
        return stateful_desired, (lambda: None)

    def _ctor_defaults(self) -> dict:
        return {
            'action_dim': self._action_dim,
            'alpha': self._alpha,
            'params': dict(self._params) if self._params else None,
            'dynamics': self._dynamics,
            'barrier': self._barrier,
            'terminal_barrier': self._terminal_barrier,
            'Q': self._Q,
            'c': self._c,
            'desired_control': self._emit_desired_control(),
            'desired_control_init_state': self._desired_control_init_state,
        }

    def _emit_desired_control(self):
        """Round-trip: the raw module (leaf) or the static stateful function."""
        if self._desired_control_module is not None:
            return self._desired_control_module
        return self._desired_control_static

    def get_init_state(self):
        """Get initial controller state (from desired controller if present)."""
        if (self._desired_control_module is not None
                and self._desired_control_module_stateful):
            return self._desired_control_module.get_init_state()
        if self._desired_control_init_state is not None:
            return self._desired_control_init_state()
        return None
