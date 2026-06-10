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
from ..controls.control_types import CFInfo
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
    _slack_gain: float
    _use_softplus: bool = eqx.field(static=True)
    _softplus_gain: float
    _buffer: float

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

    def _ctor_defaults(self) -> dict:
        return {
            'action_dim': self._action_dim,
            'alpha': self._alpha,
            'dynamics': self._dynamics,
            'barrier': self._barrier,
            'Q': self._Q,
            'c': self._c,
            'slack_gain': self._slack_gain,
            'use_softplus': self._use_softplus,
            'softplus_gain': self._softplus_gain,
            'buffer': self._buffer,
            'params': {k: v for k, v in self._params.items()
                       if k not in ('slack_gain', 'use_softplus', 'softplus_gain', 'buffer')}
        }

    def _optimal_control_single(self, x: jnp.ndarray, state=None) -> tuple:
        """
        Compute safe optimal control for a single state using closed-form solution.

        Args:
            x: Single state vector (state_dim,)
            state: Controller state (unused for CF, passed through)

        Returns:
            Tuple (u, new_state)
        """
        # Q and c are stateful functions
        Q_matrix, state = self._Q(x, state)  # (action_dim, action_dim)
        c_vector, state = self._c(x, state)  # (action_dim,)
        Q_inv = jnp.linalg.inv(Q_matrix)

        # Get barrier values and Lie derivatives (single state version for efficiency)
        hocbf, lf_hocbf, lg_hocbf = self._barrier.get_hocbf_and_lie_derivs(x)

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
        Q_matrix, state = self._Q(x, state)
        c_vector, state = self._c(x, state)
        Q_inv = jnp.linalg.inv(Q_matrix)

        hocbf, lf_hocbf, lg_hocbf = self._barrier.get_hocbf_and_lie_derivs(x)
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

        u_desired = -Q_inv @ c_vector
        slack_vars = hocbf * lam / self._slack_gain
        constraint_at_u = (lf_hocbf + jnp.dot(lg_hocbf, u) +
                           self._alpha(hocbf) + slack_vars * hocbf)

        info = CFInfo(slack_vars=slack_vars, constraint_at_u=constraint_at_u, u_desired=u_desired)
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
    _slack_gain: float
    _use_softplus: bool = eqx.field(static=True)
    _softplus_gain: float
    _buffer: float

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

    def _ctor_defaults(self) -> dict:
        return {
            'action_dim': self._action_dim,
            'alpha': self._alpha,
            'dynamics': self._dynamics,
            'barrier': self._barrier,
            'desired_control': self._desired_control,
            'desired_control_init_state': self._desired_control_init_state,
            'slack_gain': self._slack_gain,
            'use_softplus': self._use_softplus,
            'softplus_gain': self._softplus_gain,
            'buffer': self._buffer,
            'params': {k: v for k, v in self._params.items()
                       if k not in ('slack_gain', 'use_softplus', 'softplus_gain', 'buffer')}
        }

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
        hocbf, lf_hocbf, lg_hocbf = self._barrier.get_hocbf_and_lie_derivs(x)

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
        hocbf, lf_hocbf, lg_hocbf = self._barrier.get_hocbf_and_lie_derivs(x)
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

        info = CFInfo(slack_vars=slack_vars, constraint_at_u=constraint_at_u, u_desired=u_d)
        return u, new_state, info


class InputConstCFSafeControl(CFSafeControl):
    """
    Input-constrained closed-form safe control with full JAX JIT compatibility.

    This class handles systems with input constraints by using augmented dynamics
    that combine state dynamics with action dynamics.

    The augmented dynamics and composed barrier are built in the constructor as
    soon as the required components are available; the auxiliary desired action
    is derived on demand.
    """

    # Static fields for JIT compatibility
    _softmin_rho: float = eqx.field(static=True)
    _softmax_rho: float = eqx.field(static=True)
    _sigma: tuple = eqx.field(static=True)

    # Input constraint fields (marked static for JIT)
    _state_dyn: Optional[Any] = eqx.field(static=True)
    _ac_dyn: Optional[Any] = eqx.field(static=True)
    _ac_out_func: Optional[Callable] = eqx.field(static=True)
    _state_barrier: tuple
    _ac_barrier: tuple
    _ac_rel_deg: int = eqx.field(static=True)
    _aux_desired_action: Optional[Callable] = eqx.field(static=True)
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
        self._aux_desired_action = aux_desired_action
        self._desired_control = desired_control

        # Complete construction: build augmented dynamics and composed barrier
        # as soon as the components are available.
        if self._state_dyn is not None and self._ac_dyn is not None:
            self._dynamics = self._build_augmented_dynamics()
            if self._state_barrier or self._ac_barrier:
                self._barrier = self._build_composed_barrier()

    def _ctor_defaults(self) -> dict:
        return {
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
            'aux_desired_action': self._aux_desired_action,
            'softmin_rho': self._softmin_rho,
            'softmax_rho': self._softmax_rho,
            'sigma': self._sigma,
            'buffer': self._buffer,
            'slack_gain': self._slack_gain,
            'use_softplus': self._use_softplus,
            'softplus_gain': self._softplus_gain,
            'desired_control': self._desired_control
        }

    @property
    def aux_desired_action(self) -> Callable:
        """Auxiliary desired action: explicit if assigned, derived otherwise."""
        if self._aux_desired_action is not None:
            return self._aux_desired_action
        return self._derive_aux_desired_action()

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
        hocbf, lf_hocbf, lg_hocbf = self._barrier.get_hocbf_and_lie_derivs(x)
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
        hocbf, lf_hocbf, lg_hocbf = self._barrier.get_hocbf_and_lie_derivs(x)
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

        info = CFInfo(slack_vars=slack_vars, constraint_at_u=constraint_at_u, u_desired=u_d)
        return u, state, info

    def _build_composed_barrier(self):
        """Compose state and action barriers over the augmented dynamics."""
        rebind = lambda b: b._replace(dynamics=self._dynamics, barriers=None, hocbf_func=None)
        state_barriers = [rebind(barrier) for barrier in self._state_barrier]
        action_barriers = [rebind(barrier) for barrier in self._ac_barrier]

        return SoftCompositionBarrier(
            barriers=[*state_barriers, *action_barriers],
            rule='i',
            dynamics=self._dynamics,
            cfg={'softmin_rho': self._softmin_rho,
                 'softmax_rho': self._softmax_rho},
        )

    def _derive_aux_desired_action(self) -> Callable:
        """Derive the auxiliary desired action function on demand."""
        assert len(self._sigma) == self._ac_rel_deg + 1, \
            "sigma must be of length 1 + action relative degree"

        def aux_desired_action_func(x):
            ac_out_func = lambda state: self._ac_out_func(state[self._state_dyn.state_dim:])

            desired_control_lie_derivs = make_higher_order_lie_deriv_series(
                func=self._desired_control_for_aux(),
                field=self._dynamics.f,
                deg=self._ac_rel_deg
            )

            ac_out_func_lie_derivs = make_higher_order_lie_deriv_series(
                func=ac_out_func,
                field=self._dynamics.f,
                deg=self._ac_rel_deg
            )

            ac_out_Lg = jnp.linalg.inv(
                lie_deriv(ac_out_func_lie_derivs[-2], self._dynamics.g, x)
            )

            weighted_differences = jnp.stack([
                sigma * (dc(x) - of(x))
                for dc, of, sigma in zip(desired_control_lie_derivs,
                                         ac_out_func_lie_derivs,
                                         self._sigma)
            ])

            return ac_out_Lg @ jnp.sum(weighted_differences, axis=0)

        return aux_desired_action_func

    def _desired_control_for_aux(self) -> Callable:
        """Plain x -> u desired control used in the aux derivation."""
        if self._desired_control is not None:
            return self._desired_control

        def desired_control_func(x):
            state_part = x[:self._state_dyn.state_dim]
            Q, _ = self._Q(state_part, None)
            c, _ = self._c(state_part, None)
            return -jnp.linalg.inv(Q) @ c

        return desired_control_func

    def _build_augmented_dynamics(self):
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

        return CustomDynamics(
            state_dim=aug_state_dim,
            action_dim=aug_action_dim,
            f_func=aug_f,
            g_func=aug_g,
            params=None
        )


class MinIntervInputConstCFSafeControl(InputConstCFSafeControl, BaseMinIntervSafeControl):
    """Minimum intervention input-constrained safe control."""

    def __init__(self, **kwargs):
        desired_control = kwargs.pop('desired_control', None)
        init_state_fn = kwargs.get('desired_control_init_state', None)
        if desired_control is not None and init_state_fn is None:
            desired_control, init_state_fn = \
                BaseMinIntervSafeControl._normalize_desired_control(desired_control)
            kwargs['desired_control_init_state'] = init_state_fn
        super().__init__(desired_control=desired_control, **kwargs)

    def _ctor_defaults(self) -> dict:
        return {
            **super()._ctor_defaults(),
            'desired_control_init_state': self._desired_control_init_state,
        }

    def _desired_control_for_aux(self) -> Callable:
        """Wrap the stateful desired control into a plain x -> u function."""
        desired = self._desired_control
        init_state = self.get_init_state()

        def desired_control_func(x):
            u, _ = desired(x, init_state)
            return u

        return desired_control_func


class MinIntervInputConstCFSafeControlRaw(InputConstCFSafeControl):
    """
    Raw minimum intervention input-constrained safe control.

    The desired control doubles as the auxiliary desired action
    (pass both aux_desired_action and desired_control at construction).
    """
