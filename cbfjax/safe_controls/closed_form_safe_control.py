"""
Closed-Form Safe Control classes with JAX JIT compatibility.

This module implements closed-form safe control algorithms using immutable
data structures that are JIT-compatible for high performance.

All controllers follow the stateful interface:
- _optimal_control_single(x, state) -> (u, new_state)
- get_init_state() -> initial controller state
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Callable, Optional, Any, Dict, Union
from immutabledict import immutabledict

from .base_safe_control import BaseCBFSafeControl, BaseMinIntervSafeControl
from ..barriers.composite_barrier import SoftCompositionBarrier
from ..dynamics.base_dynamic import AffineInControlDynamics, CustomDynamics
from cbfjax.utils.utils import make_higher_order_lie_deriv_series, lie_deriv, update_dict_no_overwrite


class CFSafeControl(BaseCBFSafeControl):
    """
    Closed-Form Safe Control with full JAX JIT compatibility.

    Uses complete immutability pattern with static fields and cooperative inheritance.
    All data structures are hashable and JAX JIT-compatible.

    Attributes:
        _slack_gain: Slack variable gain parameter
        _use_softplus: Whether to use softplus activation
        _softplus_gain: Softplus gain parameter
        _buffer: Safety buffer parameter
    """

    # Static parameters for JIT compatibility
    _slack_gain: float = eqx.field(static=True)
    _use_softplus: bool = eqx.field(static=True)
    _softplus_gain: float = eqx.field(static=True)
    _buffer: float = eqx.field(static=True)

    def __init__(
        self,
        slack_gain: float = 1e24,
        use_softplus: bool = False,
        softplus_gain: float = 2.0,
        buffer: float = 0.0,
        **kwargs
    ):
        # Handle legacy params dict extraction
        params = kwargs.get('params', None)
        if params is not None:
            slack_gain = params.get('slack_gain', slack_gain)
            use_softplus = params.get('use_softplus', use_softplus)
            softplus_gain = params.get('softplus_gain', softplus_gain)
            buffer = params.get('buffer', buffer)

        # Ensure buffer is in params for parent
        if params is None:
            kwargs['params'] = {'buffer': buffer}
        else:
            params['buffer'] = buffer
            kwargs['params'] = params

        # Initialize via cooperative inheritance
        super().__init__(**kwargs)

        # Set static parameters
        self._slack_gain = slack_gain
        self._use_softplus = use_softplus
        self._softplus_gain = softplus_gain
        self._buffer = buffer

    @classmethod
    def create_empty(cls, action_dim: int, alpha: Optional[Callable] = None,
                     params: Optional[dict] = None) -> 'CFSafeControl':
        return cls(action_dim=action_dim, alpha=alpha, params=params)

    def _create_updated_instance(self, **kwargs):
        defaults = {
            'action_dim': self._action_dim,
            'alpha': self._alpha,
            'dynamics': self._dynamics,
            'barrier': self._barrier,
            'Q': self._Q,
            'c': self._c,
            'slack_gain': self._slack_gain,
            'use_softplus': self._use_softplus,
            'softplus_gain': self._softplus_gain,
            'buffer': self._buffer
        }
        defaults.update(kwargs)
        return self.__class__(**defaults)

    def assign_state_barrier(self, barrier) -> 'CFSafeControl':
        return self._create_updated_instance(barrier=barrier)

    def assign_dynamics(self, dynamics) -> 'CFSafeControl':
        return self._create_updated_instance(dynamics=dynamics)

    def assign_cost(self, Q: Callable, c: Callable) -> 'CFSafeControl':
        return self._create_updated_instance(Q=Q, c=c)

    def _optimal_control_single(self, x: jnp.ndarray, state=None) -> tuple:
        """
        Compute safe optimal control for a single state using closed-form solution.

        Args:
            x: Single state vector (state_dim,)
            state: Controller state (unused for CF, passed through)

        Returns:
            Tuple (u, new_state)
        """
        # Q and c are single-state functions
        Q_matrix = self._Q(x)  # (action_dim, action_dim)
        c_vector = self._c(x)  # (action_dim,)
        Q_inv = jnp.linalg.inv(Q_matrix)

        # Get barrier values and Lie derivatives (single state version for efficiency)
        hocbf, lf_hocbf, lg_hocbf = self._barrier._get_hocbf_and_lie_derivs_single(x)

        # Apply buffer
        hocbf = hocbf - self._buffer

        # Compute closed-form solution
        omega = lf_hocbf - jnp.dot(lg_hocbf, Q_inv @ c_vector) + self._alpha(hocbf)
        den = jnp.dot(lg_hocbf, Q_inv @ lg_hocbf) + (1 / self._slack_gain) * hocbf ** 2

        # JIT-friendly conditional using static fields
        num = jax.lax.cond(
            self._use_softplus,
            lambda val: jax.nn.softplus(val * self._softplus_gain) / self._softplus_gain,
            lambda val: jax.nn.relu(val),
            -omega
        )

        lam = num / den

        # Compute control
        u = -Q_inv @ (c_vector - lg_hocbf * lam)

        return u, state

    def _optimal_control_single_with_info(self, x: jnp.ndarray, state=None) -> tuple:
        """Compute safe optimal control with diagnostic info."""
        Q_matrix = self._Q(x)
        c_vector = self._c(x)
        Q_inv = jnp.linalg.inv(Q_matrix)

        hocbf, lf_hocbf, lg_hocbf = self._barrier._get_hocbf_and_lie_derivs_single(x)
        hocbf = hocbf - self._buffer

        omega = lf_hocbf - jnp.dot(lg_hocbf, Q_inv @ c_vector) + self._alpha(hocbf)
        den = jnp.dot(lg_hocbf, Q_inv @ lg_hocbf) + (1 / self._slack_gain) * hocbf ** 2

        num = jax.lax.cond(
            self._use_softplus,
            lambda val: jax.nn.softplus(val * self._softplus_gain) / self._softplus_gain,
            lambda val: jax.nn.relu(val),
            -omega
        )

        lam = num / den
        u = -Q_inv @ (c_vector - lg_hocbf * lam)

        slack_vars = hocbf * lam / self._slack_gain
        constraint_at_u = (lf_hocbf + jnp.dot(lg_hocbf, u) +
                           self._alpha(hocbf) + slack_vars * hocbf)

        info = {'slack_vars': slack_vars, 'constraint_at_u': constraint_at_u}
        return u, state, info

    def eval_barrier(self, x: jnp.ndarray) -> jnp.ndarray:
        """Evaluate barrier function at state x."""
        return self._barrier.hocbf(x)


class MinIntervCFSafeControl(BaseMinIntervSafeControl):
    """
    Minimum-Intervention Closed-Form Safe Control with full JAX JIT compatibility.

    Implements minimum intervention control using cooperative inheritance.
    All methods return new instances following functional programming principles.

    Attributes:
        _slack_gain: Slack variable gain parameter
        _use_softplus: Whether to use softplus activation
        _softplus_gain: Softplus gain parameter
        _buffer: Safety buffer parameter
    """

    # Static parameters for JIT compatibility
    _slack_gain: float = eqx.field(static=True)
    _use_softplus: bool = eqx.field(static=True)
    _softplus_gain: float = eqx.field(static=True)
    _buffer: float = eqx.field(static=True)

    def __init__(
        self,
        slack_gain: float = 1e24,
        use_softplus: bool = False,
        softplus_gain: float = 2.0,
        buffer: float = 0.0,
        **kwargs
    ):
        # Handle legacy params dict extraction
        params = kwargs.get('params', None)
        if params is not None:
            slack_gain = params.get('slack_gain', slack_gain)
            use_softplus = params.get('use_softplus', use_softplus)
            softplus_gain = params.get('softplus_gain', softplus_gain)
            buffer = params.get('buffer', buffer)

        # Ensure buffer is in params for parent
        if params is None:
            kwargs['params'] = {'buffer': buffer}
        else:
            params['buffer'] = buffer
            kwargs['params'] = params

        # Initialize via cooperative inheritance
        super().__init__(**kwargs)

        # Set static parameters
        self._slack_gain = slack_gain
        self._use_softplus = use_softplus
        self._softplus_gain = softplus_gain
        self._buffer = buffer

    @classmethod
    def create_empty(cls, action_dim: int, alpha: Optional[Callable] = None,
                     params: Optional[dict] = None) -> 'MinIntervCFSafeControl':
        return cls(action_dim=action_dim, alpha=alpha, params=params)

    def _create_updated_instance(self, **kwargs):
        defaults = {
            'action_dim': self._action_dim,
            'alpha': self._alpha,
            'dynamics': self._dynamics,
            'barrier': self._barrier,
            'desired_control': self._desired_control,
            'desired_control_init_state': self._desired_control_init_state,
            'slack_gain': self._slack_gain,
            'use_softplus': self._use_softplus,
            'softplus_gain': self._softplus_gain,
            'buffer': self._buffer
        }
        defaults.update(kwargs)
        return self.__class__(**defaults)

    def assign_state_barrier(self, barrier) -> 'MinIntervCFSafeControl':
        return self._create_updated_instance(barrier=barrier)

    def assign_dynamics(self, dynamics) -> 'MinIntervCFSafeControl':
        return self._create_updated_instance(dynamics=dynamics)

    def assign_desired_control(self, desired_control) -> 'MinIntervCFSafeControl':
        """
        Assign desired control function.

        Accepts controller objects, plain functions, or stateful functions.
        """
        if hasattr(desired_control, '_optimal_control_single') and hasattr(desired_control, 'get_init_state'):
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

    @jax.jit
    def _optimal_control_single(self, x: jnp.ndarray, state=None) -> tuple:
        """
        Compute minimum intervention safe control for a single state.

        Args:
            x: Single state vector (state_dim,)
            state: Controller state (from desired controller)

        Returns:
            Tuple (u, new_state)
        """
        # Get barrier values and Lie derivatives (single state version for efficiency)
        hocbf, lf_hocbf, lg_hocbf = self._barrier._get_hocbf_and_lie_derivs_single(x)

        # Apply buffer
        hocbf = hocbf - self._buffer

        # Get desired control (stateful)
        u_d, new_state = self._desired_control(x, state)

        # Compute closed-form solution
        omega = lf_hocbf + jnp.dot(lg_hocbf, u_d) + self._alpha(hocbf)
        den = jnp.dot(lg_hocbf, lg_hocbf) + (1 / self._slack_gain) * hocbf ** 2

        # JIT-friendly conditional
        num = jax.lax.cond(
            self._use_softplus,
            lambda val: jax.nn.softplus(val * self._softplus_gain) / self._softplus_gain,
            lambda val: jax.nn.relu(val),
            -omega
        )

        lam = num / den

        # Compute control
        u = u_d + lg_hocbf * lam

        return u, new_state

    def _optimal_control_single_with_info(self, x: jnp.ndarray, state=None) -> tuple:
        """Compute minimum intervention safe control with diagnostic info."""
        hocbf, lf_hocbf, lg_hocbf = self._barrier._get_hocbf_and_lie_derivs_single(x)
        hocbf = hocbf - self._buffer

        u_d, new_state = self._desired_control(x, state)

        omega = lf_hocbf + jnp.dot(lg_hocbf, u_d) + self._alpha(hocbf)
        den = jnp.dot(lg_hocbf, lg_hocbf) + (1 / self._slack_gain) * hocbf ** 2

        num = jax.lax.cond(
            self._use_softplus,
            lambda val: jax.nn.softplus(val * self._softplus_gain) / self._softplus_gain,
            lambda val: jax.nn.relu(val),
            -omega
        )

        lam = num / den
        u = u_d + lg_hocbf * lam

        slack_vars = hocbf * lam / self._slack_gain
        constraint_at_u = (lf_hocbf + jnp.dot(lg_hocbf, u) +
                           self._alpha(hocbf) + slack_vars * hocbf)

        info = {'slack_vars': slack_vars, 'constraint_at_u': constraint_at_u}
        return u, new_state, info


class InputConstCFSafeControl(CFSafeControl):
    """
    Input-constrained closed-form safe control with full JAX JIT compatibility.

    This class handles systems with input constraints by using augmented dynamics
    that combine state dynamics with action dynamics.
    """

    # Static fields for JIT compatibility
    _softmin_rho: float = eqx.field(static=True)
    _softmax_rho: float = eqx.field(static=True)
    _sigma: tuple = eqx.field(static=True)

    # Input constraint fields (marked static for JIT)
    _state_dyn: Optional[Any] = eqx.field(static=True)
    _ac_dyn: Optional[Any] = eqx.field(static=True)
    _ac_out_func: Optional[Callable] = eqx.field(static=True)
    _state_barrier: tuple = eqx.field(static=True)
    _ac_barrier: tuple = eqx.field(static=True)
    _ac_rel_deg: int = eqx.field(static=True)
    aux_desired_action: Optional[Callable] = eqx.field(static=True)
    _desired_control: Optional[Callable] = eqx.field(static=True)

    @staticmethod
    def _create_identity_func():
        """Create identity function for action output."""
        def identity(x):
            return x
        return identity

    def __init__(
        self,
        state_dyn=None,
        ac_dyn=None,
        ac_out_func=None,
        state_barrier=None,
        ac_barrier=None,
        ac_rel_deg=None,
        aux_desired_action=None,
        softmin_rho: float = 1.0,
        softmax_rho: float = 1.0,
        sigma: tuple = (1.0,),
        desired_control=None,
        **kwargs
    ):
        # Extract and merge params
        params = kwargs.get('params', None)
        default_params = {
            'softmin_rho': softmin_rho,
            'softmax_rho': softmax_rho,
            'sigma': sigma,
        }
        if params is not None:
            default_params.update(params)
            softmin_rho = default_params.get('softmin_rho', softmin_rho)
            softmax_rho = default_params.get('softmax_rho', softmax_rho)
            sigma = default_params.get('sigma', sigma)
        kwargs['params'] = default_params

        # Convert sigma to tuple if needed
        if isinstance(sigma, (list, jnp.ndarray)):
            sigma = tuple(float(x) for x in sigma)
        elif not isinstance(sigma, tuple):
            sigma = (float(sigma),)

        # Initialize via cooperative inheritance
        super().__init__(**kwargs)

        # Set additional static fields
        self._softmin_rho = float(softmin_rho)
        self._softmax_rho = float(softmax_rho)
        self._sigma = sigma

        # Initialize input constraint specific fields with defaults
        self._state_dyn = state_dyn
        self._ac_dyn = ac_dyn
        self._ac_out_func = ac_out_func or self._create_identity_func()
        self._state_barrier = tuple(state_barrier) if state_barrier is not None else ()
        self._ac_barrier = tuple(ac_barrier) if ac_barrier is not None else ()
        self._ac_rel_deg = ac_rel_deg if ac_rel_deg is not None else 1
        self.aux_desired_action = aux_desired_action
        self._desired_control = desired_control

    def _create_updated_instance(self, **kwargs):
        defaults = {
            'action_dim': self._action_dim,
            'alpha': self._alpha,
            'params': dict(self._params),
            'dynamics': self._dynamics,
            'barrier': self._barrier,
            'Q': self._Q,
            'c': self._c,
            'state_dyn': self._state_dyn,
            'ac_dyn': self._ac_dyn,
            'ac_out_func': self._ac_out_func,
            'state_barrier': self._state_barrier,
            'ac_barrier': self._ac_barrier,
            'ac_rel_deg': self._ac_rel_deg,
            'aux_desired_action': self.aux_desired_action,
            'softmin_rho': self._softmin_rho,
            'softmax_rho': self._softmax_rho,
            'sigma': self._sigma,
            'buffer': self._buffer,
            'slack_gain': self._slack_gain,
            'use_softplus': self._use_softplus,
            'softplus_gain': self._softplus_gain,
            'desired_control': self._desired_control
        }
        defaults.update(kwargs)
        return self.__class__(**defaults)

    @classmethod
    def create_empty(cls, action_dim: int, alpha: Optional[Callable] = None,
                    params: Optional[dict] = None):
        return cls(action_dim=action_dim, alpha=alpha, params=params)

    def assign_dynamics(self, dynamics):
        """Override to prevent direct dynamics assignment."""
        raise ValueError("Use 'assign_state_action_dynamics' method to assign state and action dynamics")

    def assign_state_action_dynamics(self, state_dynamics, action_dynamics,
                                     action_output_function: Optional[Callable] = None) -> 'InputConstCFSafeControl':
        if action_output_function is None:
            action_output_function = self._create_identity_func()

        return self._create_updated_instance(
            state_dyn=state_dynamics,
            ac_dyn=action_dynamics,
            ac_out_func=action_output_function
        )

    def assign_state_barrier(self, barrier) -> 'InputConstCFSafeControl':
        return self._create_updated_instance(state_barrier=barrier)

    def assign_action_barrier(self, action_barrier, rel_deg: int) -> 'InputConstCFSafeControl':
        return self._create_updated_instance(
            ac_barrier=action_barrier,
            ac_rel_deg=rel_deg
        )

    def make(self) -> 'InputConstCFSafeControl':
        """Build the complete input-constrained controller."""
        updated_ctrl = self._make_augmented_dynamics()
        updated_ctrl = updated_ctrl._make_composed_barrier()
        updated_ctrl = updated_ctrl._make_aux_desired_action()
        return updated_ctrl

    @jax.jit
    def _optimal_control_single(self, x: jnp.ndarray, state=None) -> tuple:
        """
        Compute safe optimal control for input-constrained system.

        Args:
            x: Single augmented state vector (state_dim + action_dim,)
            state: Controller state (unused, passed through)

        Returns:
            Tuple (u, new_state)
        """
        hocbf, lf_hocbf, lg_hocbf = self._barrier._get_hocbf_and_lie_derivs_single(x)
        hocbf = hocbf - self._buffer

        u_d = self.aux_desired_action(x)

        omega = lf_hocbf + jnp.dot(lg_hocbf, u_d) + self._alpha(hocbf)
        den = jnp.dot(lg_hocbf, lg_hocbf) + (1 / self._slack_gain) * hocbf ** 2

        num = jax.lax.cond(
            self._use_softplus,
            lambda val: jax.nn.softplus(val * self._softplus_gain) / self._softplus_gain,
            lambda val: jax.nn.relu(val),
            -omega
        )

        lam = num / den
        u = u_d + lg_hocbf * lam

        return u, state

    def _optimal_control_single_with_info(self, x: jnp.ndarray, state=None) -> tuple:
        """Compute safe optimal control with diagnostic info."""
        hocbf, lf_hocbf, lg_hocbf = self._barrier._get_hocbf_and_lie_derivs_single(x)
        hocbf = hocbf - self._buffer

        u_d = self.aux_desired_action(x)

        omega = lf_hocbf + jnp.dot(lg_hocbf, u_d) + self._alpha(hocbf)
        den = jnp.dot(lg_hocbf, lg_hocbf) + (1 / self._slack_gain) * hocbf ** 2

        num = jax.lax.cond(
            self._use_softplus,
            lambda val: jax.nn.softplus(val * self._softplus_gain) / self._softplus_gain,
            lambda val: jax.nn.relu(val),
            -omega
        )

        lam = num / den
        u = u_d + lg_hocbf * lam

        slack_vars = hocbf * lam / self._slack_gain
        constraint_at_u = (lf_hocbf + jnp.dot(lg_hocbf, u) +
                           self._alpha(hocbf) + slack_vars * hocbf)

        info = {'slack_vars': slack_vars, 'constraint_at_u': constraint_at_u}
        return u, state, info

    def _make_composed_barrier(self) -> 'InputConstCFSafeControl':
        """Create composed barrier from state and action barriers."""
        state_barriers = [barrier.assign_dynamics(self._dynamics) for barrier in self._state_barrier]
        action_barriers = [barrier.assign_dynamics(self._dynamics) for barrier in self._ac_barrier]

        barrier = SoftCompositionBarrier(
            cfg={'softmin_rho': self._softmin_rho,
                 'softmax_rho': self._softmax_rho}
        ).assign_dynamics(self._dynamics).assign_barriers_and_rule(
            barriers=[*state_barriers, *action_barriers],
            rule='i'
        )

        return self._create_updated_instance(barrier=barrier)

    def _make_aux_desired_action(self) -> 'InputConstCFSafeControl':
        """Create auxiliary desired action function."""
        assert len(self._sigma) == self._ac_rel_deg + 1, \
            "sigma must be of length 1 + action relative degree"

        updated_ctrl = self._make_desired_control()

        def aux_desired_action_func(x):
            ac_out_func = lambda state: updated_ctrl._ac_out_func(state[updated_ctrl._state_dyn.state_dim:])

            desired_control_lie_derivs = make_higher_order_lie_deriv_series(
                func=updated_ctrl._desired_control,
                field=updated_ctrl._dynamics.f,
                deg=updated_ctrl._ac_rel_deg
            )

            ac_out_func_lie_derivs = make_higher_order_lie_deriv_series(
                func=ac_out_func,
                field=updated_ctrl._dynamics.f,
                deg=updated_ctrl._ac_rel_deg
            )

            ac_out_Lg = jnp.linalg.inv(
                lie_deriv(ac_out_func_lie_derivs[-2], updated_ctrl._dynamics.g, x)
            )

            weighted_differences = jnp.stack([
                sigma * (dc(x) - of(x))
                for dc, of, sigma in zip(desired_control_lie_derivs,
                                         ac_out_func_lie_derivs,
                                         updated_ctrl._sigma)
            ])

            return ac_out_Lg @ jnp.sum(weighted_differences, axis=0)

        return updated_ctrl._create_updated_instance(aux_desired_action=aux_desired_action_func)

    def _make_desired_control(self) -> 'InputConstCFSafeControl':
        """Create desired control function for state part."""

        def desired_control_func(x):
            state_part = x[:self._state_dyn.state_dim]
            Q = self._Q(state_part)
            c = self._c(state_part)
            return -jnp.linalg.inv(Q) @ c

        return self

    def _make_augmented_dynamics(self) -> 'InputConstCFSafeControl':
        """Create augmented dynamics combining state and action dynamics."""
        assert self._state_dyn.action_dim == self._ac_dyn.action_dim, \
            'Dimension mismatch between state and action dynamics'

        aug_state_dim = self._state_dyn.state_dim + self._ac_dyn.state_dim
        aug_action_dim = self._state_dyn.action_dim

        state_dyn = self._state_dyn
        ac_dyn = self._ac_dyn
        ac_out_func = self._ac_out_func

        def aug_f(x):
            state_part = x[:state_dyn.state_dim]
            action_part = x[state_dyn.state_dim:]
            action_output = ac_out_func(action_part)
            state_rhs = state_dyn.rhs(state_part, action_output)
            action_rhs = ac_dyn.f(action_part)
            return jnp.concatenate([state_rhs, action_rhs])

        def aug_g(x):
            action_part = x[state_dyn.state_dim:]
            state_g = jnp.zeros((state_dyn.state_dim, state_dyn.action_dim))
            action_g = ac_dyn.g(action_part)
            return jnp.concatenate([state_g, action_g], axis=0)

        dynamics = CustomDynamics(
            state_dim=aug_state_dim,
            action_dim=aug_action_dim,
            f_func=aug_f,
            g_func=aug_g,
            params=None
        )

        return self._create_updated_instance(dynamics=dynamics)


class MinIntervInputConstCFSafeControl(InputConstCFSafeControl, BaseMinIntervSafeControl):
    """Minimum intervention input-constrained safe control."""

    def assign_desired_control(self, desired_control) -> 'MinIntervInputConstCFSafeControl':
        """Assign desired control and build controller."""
        if hasattr(desired_control, '_optimal_control_single') and hasattr(desired_control, 'get_init_state'):
            ctrl_obj = desired_control
            def stateful_desired(x, state):
                return ctrl_obj._optimal_control_single(x, state)
            init_state_fn = ctrl_obj.get_init_state
            updated_ctrl = self._create_updated_instance(
                desired_control=stateful_desired,
                desired_control_init_state=init_state_fn,
            )
        else:
            func = desired_control
            def stateful_desired(x, state):
                return func(x), state
            updated_ctrl = self._create_updated_instance(
                desired_control=stateful_desired,
                desired_control_init_state=lambda: None,
            )
        return updated_ctrl.make()

    def _make_desired_control(self) -> 'MinIntervInputConstCFSafeControl':
        """Override to skip making desired control since it's directly assigned."""
        return self


class MinIntervInputConstCFSafeControlRaw(InputConstCFSafeControl):
    """
    Raw minimum intervention input-constrained safe control.

    Uses desired control directly as auxiliary desired action.
    """

    def assign_desired_control(self, desired_control) -> 'MinIntervInputConstCFSafeControlRaw':
        """Assign desired control as auxiliary desired action."""
        if hasattr(desired_control, '_optimal_control_single') and hasattr(desired_control, 'get_init_state'):
            ctrl_obj = desired_control
            def stateful_desired(x, state):
                return ctrl_obj._optimal_control_single(x, state)
            init_state_fn = ctrl_obj.get_init_state
            updated_ctrl = self._create_updated_instance(
                aux_desired_action=desired_control,
                desired_control=stateful_desired,
                desired_control_init_state=init_state_fn,
            )
        else:
            updated_ctrl = self._create_updated_instance(
                aux_desired_action=desired_control,
                desired_control=desired_control
            )
        return updated_ctrl.make()

    def _make_desired_control(self) -> 'MinIntervInputConstCFSafeControlRaw':
        """Override to skip making desired control."""
        return self

    def _make_aux_desired_action(self) -> 'MinIntervInputConstCFSafeControlRaw':
        """Override to skip making auxiliary desired action."""
        return self
