"""
Closed-Form Safe Control classes with full JAX JIT compatibility.

This module implements closed-form safe control algorithms using the complete
immutability pattern. All data structures are JAX JIT-compatible with proper
hashability and functional design.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Callable, Optional, Any, Dict, Union
from immutabledict import immutabledict

from .base_safe_control import BaseSafeControl, BaseMinIntervSafeControl
from ..barriers.composite_barrier import SoftCompositionBarrier
from ..dynamics.base import AffineInControlDynamics
from cbfjax.utils.utils import make_higher_order_lie_deriv_series, lie_deriv, update_dict_no_overwrite


class CFSafeControl(BaseSafeControl):
    """
    Closed-Form Safe Control with full JAX JIT compatibility.

    Uses complete immutability pattern with static fields and explicit constructors.
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

    def __init__(self, action_dim: int, alpha: Optional[Callable] = None,
                 params: Optional[dict] = None, dynamics=None, barrier=None,
                 Q=None, c=None, desired_control=None,
                 slack_gain: float = 1e24, use_softplus: bool = False,
                 softplus_gain: float = 2.0, buffer: float = 0.0):
        """
        Initialize CFSafeControl with explicit parameters.

        Args:
            action_dim: Control input dimension
            alpha: Class-K function for barrier constraint
            params: Legacy parameter dictionary (deprecated)
            dynamics: System dynamics object
            barrier: Barrier function object
            Q: Cost matrix function
            c: Cost vector function
            desired_control: Desired control function
            slack_gain: Slack variable gain
            use_softplus: Whether to use softplus activation
            softplus_gain: Softplus gain parameter
            buffer: Safety buffer
        """
        # Handle legacy params dict
        if params is not None:
            slack_gain = params.get('slack_gain', slack_gain)
            use_softplus = params.get('use_softplus', use_softplus)
            softplus_gain = params.get('softplus_gain', softplus_gain)
            buffer = params.get('buffer', buffer)

        # Initialize base class with explicit constructor
        super().__init__(action_dim, alpha, immutabledict({'buffer': buffer}), dynamics, barrier, Q, c)

        # Set static parameters
        self._slack_gain = slack_gain
        self._use_softplus = use_softplus
        self._softplus_gain = softplus_gain
        self._buffer = buffer

    @classmethod
    def create_empty(cls, action_dim: int, alpha: Optional[Callable] = None,
                     params: Optional[dict] = None) -> 'CFSafeControl':
        """
        Create empty CFSafeControl instance for assignment chain.

        Args:
            action_dim: Control input dimension
            alpha: Class-K function for barrier constraint
            params: Optional parameter dictionary

        Returns:
            Empty CFSafeControl instance ready for assignment
        """
        return cls(action_dim=action_dim, alpha=alpha, params=params)

    def assign_state_barrier(self, barrier) -> 'CFSafeControl':
        """
        Assign state barrier to controller.

        Args:
            barrier: Barrier function object

        Returns:
            New CFSafeControl instance with assigned barrier
        """
        return CFSafeControl(
            action_dim=self._action_dim,
            alpha=self._alpha,
            dynamics=self._dynamics,
            barrier=barrier,
            Q=self._Q,
            c=self._c,
            slack_gain=self._slack_gain,
            use_softplus=self._use_softplus,
            softplus_gain=self._softplus_gain,
            buffer=self._buffer
        )

    def assign_dynamics(self, dynamics) -> 'CFSafeControl':
        """
        Assign dynamics to controller.

        Args:
            dynamics: System dynamics object

        Returns:
            New CFSafeControl instance with assigned dynamics
        """
        return CFSafeControl(
            action_dim=self._action_dim,
            alpha=self._alpha,
            dynamics=dynamics,
            barrier=self._barrier,
            Q=self._Q,
            c=self._c,
            slack_gain=self._slack_gain,
            use_softplus=self._use_softplus,
            softplus_gain=self._softplus_gain,
            buffer=self._buffer
        )

    def assign_cost(self, Q: Callable, c: Callable) -> 'CFSafeControl':
        """
        Assign quadratic cost function.

        Args:
            Q: Function that computes cost matrix from state
            c: Function that computes cost vector from state

        Returns:
            New CFSafeControl instance with assigned cost
        """
        return CFSafeControl(
            action_dim=self._action_dim,
            alpha=self._alpha,
            dynamics=self._dynamics,
            barrier=self._barrier,
            Q=Q,
            c=c,
            slack_gain=self._slack_gain,
            use_softplus=self._use_softplus,
            softplus_gain=self._softplus_gain,
            buffer=self._buffer
        )

    def _safe_optimal_control_single(self, x: jnp.ndarray, ret_info: bool = False) -> Union[jnp.ndarray, tuple]:
        """
        Compute safe optimal control for a single state using closed-form solution.

        Args:
            x: Single state vector (state_dim,)
            ret_info: Whether to return additional information

        Returns:
            If ret_info=False: Control vector (action_dim,)
            If ret_info=True: Tuple (u, slack_vars, constraint_at_u)
        """
        # Q and c are single-state functions
        Q_matrix = self._Q(x)  # (action_dim, action_dim)
        c_vector = self._c(x)  # (action_dim,)
        Q_inv = jnp.linalg.inv(Q_matrix)

        # Get barrier values and Lie derivatives (barrier handles single states)
        hocbf, lf_hocbf, lg_hocbf = self._barrier.get_hocbf_and_lie_derivs(x)

        # Apply buffer - use static field instead of dict access
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

        # JIT-friendly return with consistent tuple structure
        return jax.lax.cond(
            ret_info,
            lambda _: self._add_optimal_control_info(u, hocbf, lf_hocbf, lg_hocbf, lam),
            lambda _: u,
            None
        )

    def _add_optimal_control_info(self, u: jnp.ndarray, hocbf: jnp.ndarray,
                                  lf_hocbf: jnp.ndarray, lg_hocbf: jnp.ndarray,
                                  lam: jnp.ndarray) -> tuple:
        """
        Add optimal control information using simple tuple return.

        Args:
            u: Control vector
            hocbf: Barrier function value
            lf_hocbf: Lie derivative of barrier w.r.t. drift
            lg_hocbf: Lie derivative of barrier w.r.t. control
            lam: Lagrange multiplier

        Returns:
            Tuple (u, slack_vars, constraint_at_u)
        """
        slack_vars = hocbf * lam / self._slack_gain
        constraint_at_u = (lf_hocbf + jnp.dot(lg_hocbf, u) +
                           self._alpha(hocbf) + slack_vars * hocbf)

        return u, slack_vars, constraint_at_u

    def eval_barrier(self, x: jnp.ndarray) -> jnp.ndarray:
        """Evaluate barrier function at state x."""
        return self._barrier.hocbf(x)


class MinIntervCFSafeControl(BaseMinIntervSafeControl):
    """
    Minimum-Intervention Closed-Form Safe Control with full JAX JIT compatibility.

    Implements minimum intervention control using the complete immutability pattern.
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

    def __init__(self, action_dim: int, alpha: Optional[Callable] = None,
                 params: Optional[dict] = None, dynamics=None, barrier=None,
                 desired_control=None,
                 slack_gain: float = 1e24, use_softplus: bool = False,
                 softplus_gain: float = 2.0, buffer: float = 0.0):
        """
        Initialize MinIntervCFSafeControl with explicit parameters.

        Args:
            action_dim: Control input dimension
            alpha: Class-K function for barrier constraint
            params: Legacy parameter dictionary (deprecated)
            dynamics: System dynamics object
            barrier: Barrier function object
            desired_control: Desired control function
            slack_gain: Slack variable gain
            use_softplus: Whether to use softplus activation
            softplus_gain: Softplus gain parameter
            buffer: Safety buffer
        """
        # Handle legacy params dict
        if params is not None:
            slack_gain = params.get('slack_gain', slack_gain)
            use_softplus = params.get('use_softplus', use_softplus)
            softplus_gain = params.get('softplus_gain', softplus_gain)
            buffer = params.get('buffer', buffer)

        # Initialize base class with explicit constructor
        super().__init__(action_dim, alpha, immutabledict({'buffer': buffer}), desired_control, dynamics, barrier)

        # Set static parameters
        self._slack_gain = slack_gain
        self._use_softplus = use_softplus
        self._softplus_gain = softplus_gain
        self._buffer = buffer

    @classmethod
    def create_empty(cls, action_dim: int, alpha: Optional[Callable] = None,
                     params: Optional[dict] = None) -> 'MinIntervCFSafeControl':
        """
        Create empty MinIntervCFSafeControl instance for assignment chain.

        Args:
            action_dim: Control input dimension
            alpha: Class-K function for barrier constraint
            params: Optional parameter dictionary

        Returns:
            Empty MinIntervCFSafeControl instance ready for assignment
        """
        return cls(action_dim=action_dim, alpha=alpha, params=params)

    def assign_state_barrier(self, barrier) -> 'MinIntervCFSafeControl':
        """
        Assign state barrier to controller.

        Args:
            barrier: Barrier function object

        Returns:
            New MinIntervCFSafeControl instance with assigned barrier
        """
        return MinIntervCFSafeControl(
            action_dim=self._action_dim,
            alpha=self._alpha,
            dynamics=self._dynamics,
            barrier=barrier,
            desired_control=self._desired_control,
            slack_gain=self._slack_gain,
            use_softplus=self._use_softplus,
            softplus_gain=self._softplus_gain,
            buffer=self._buffer
        )

    def assign_dynamics(self, dynamics) -> 'MinIntervCFSafeControl':
        """
        Assign dynamics to controller.

        Args:
            dynamics: System dynamics object

        Returns:
            New MinIntervCFSafeControl instance with assigned dynamics
        """
        return MinIntervCFSafeControl(
            action_dim=self._action_dim,
            alpha=self._alpha,
            dynamics=dynamics,
            barrier=self._barrier,
            desired_control=self._desired_control,
            slack_gain=self._slack_gain,
            use_softplus=self._use_softplus,
            softplus_gain=self._softplus_gain,
            buffer=self._buffer
        )

    def assign_desired_control(self, desired_control: Callable) -> 'MinIntervCFSafeControl':
        """
        Assign desired control function.

        Args:
            desired_control: Function that computes desired control

        Returns:
            New MinIntervCFSafeControl instance with assigned desired control
        """
        return MinIntervCFSafeControl(
            action_dim=self._action_dim,
            alpha=self._alpha,
            dynamics=self._dynamics,
            barrier=self._barrier,
            desired_control=desired_control,
            slack_gain=self._slack_gain,
            use_softplus=self._use_softplus,
            softplus_gain=self._softplus_gain,
            buffer=self._buffer
        )

    def _safe_optimal_control_single(self, x: jnp.ndarray, ret_info: bool = False) -> Union[jnp.ndarray, tuple]:
        """
        Compute minimum intervention safe control for a single state.

        Args:
            x: Single state vector (state_dim,)
            ret_info: Whether to return additional information

        Returns:
            If ret_info=False: Control vector (action_dim,)
            If ret_info=True: Tuple (u, slack_vars, constraint_at_u)
        """
        # Get barrier values and Lie derivatives (barrier handles single states)
        hocbf, lf_hocbf, lg_hocbf = self._barrier.get_hocbf_and_lie_derivs(x)

        # Apply buffer - use static field
        hocbf = hocbf - self._buffer

        # Get desired control (single state input)
        u_d = self._desired_control(x)

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

        # JIT-friendly return
        return jax.lax.cond(
            ret_info,
            lambda _: self._add_optimal_control_info(u, hocbf, lf_hocbf, lg_hocbf, lam),
            lambda _: u,
            None
        )

    def _add_optimal_control_info(self, u: jnp.ndarray, hocbf: jnp.ndarray,
                                  lf_hocbf: jnp.ndarray, lg_hocbf: jnp.ndarray,
                                  lam: jnp.ndarray) -> tuple:
        """
        Add optimal control information using simple tuple return.

        Args:
            u: Control vector
            hocbf: Barrier function value
            lf_hocbf: Lie derivative of barrier w.r.t. drift
            lg_hocbf: Lie derivative of barrier w.r.t. control
            lam: Lagrange multiplier

        Returns:
            Tuple (u, slack_vars, constraint_at_u)
        """
        slack_vars = hocbf * lam / self._slack_gain
        constraint_at_u = (lf_hocbf + jnp.dot(lg_hocbf, u) +
                           self._alpha(hocbf) + slack_vars * hocbf)

        return u, slack_vars, constraint_at_u


class InputConstCFSafeControl(CFSafeControl):
    """
    Input-constrained closed-form safe control with full JAX JIT compatibility.

    This class handles systems with input constraints by using augmented dynamics
    that combine state dynamics with action dynamics. The action dynamics model
    the actuator behavior, while state dynamics represent the main system.

    The controller uses barrier functions for both state constraints and action
    constraints, composing them to ensure overall system safety.
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

    def __init__(self, action_dim: int, alpha: Optional[Callable] = None,
                 params: Optional[dict] = None, dynamics=None, barrier=None,
                 Q=None, c=None, state_dyn=None, ac_dyn=None, ac_out_func=None,
                 state_barrier=None, ac_barrier=None, ac_rel_deg=None,
                 aux_desired_action=None, softmin_rho=None, softmax_rho=None,
                 sigma=None, buffer=None, slack_gain=None, use_softplus=None,
                 softplus_gain=None, desired_control=None):
        # Add default parameters for input constrained control
        default_params = {
            'slack_gain': 1e24,
            'use_softplus': False,
            'softplus_gain': 2.0,
            'buffer': 0.0,
            'softmin_rho': 1.0,
            'softmax_rho': 1.0,
            'sigma': (1.0,)  # Default sigma as tuple for JIT
        }

        if params is None:
            params = default_params
        else:
            params = update_dict_no_overwrite(params, default_params)

        # Extract scalar parameters for static fields
        buffer_val = buffer if buffer is not None else params['buffer']
        slack_gain_val = slack_gain if slack_gain is not None else params['slack_gain']
        use_softplus_val = use_softplus if use_softplus is not None else params['use_softplus']
        softplus_gain_val = softplus_gain if softplus_gain is not None else params['softplus_gain']
        softmin_rho_val = softmin_rho if softmin_rho is not None else params['softmin_rho']
        softmax_rho_val = softmax_rho if softmax_rho is not None else params['softmax_rho']
        sigma_val = sigma if sigma is not None else params['sigma']

        # Convert sigma to tuple if needed
        if isinstance(sigma_val, (list, jnp.ndarray)):
            sigma_val = tuple(float(x) for x in sigma_val)
        elif not isinstance(sigma_val, tuple):
            sigma_val = (float(sigma_val),)

        # Initialize parent with explicit constructor pattern
        super().__init__(action_dim, alpha, immutabledict({'buffer': buffer_val}),
                        dynamics, barrier, Q, c, buffer_val, slack_gain_val,
                        use_softplus_val, softplus_gain_val)

        # Set additional static fields
        self._softmin_rho = float(softmin_rho_val)
        self._softmax_rho = float(softmax_rho_val)
        self._sigma = sigma_val

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
        """Create new instance with updated fields using explicit constructor."""
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
        """Create empty InputConstCFSafeControl for builder pattern."""
        return cls(action_dim=action_dim, alpha=alpha, params=params)

    def assign_dynamics(self, dynamics):
        """Override to prevent direct dynamics assignment."""
        raise ValueError("Use 'assign_state_action_dynamics' method to assign state and action dynamics")

    def assign_state_action_dynamics(self, state_dynamics, action_dynamics,
                                     action_output_function: Optional[Callable] = None) -> 'InputConstCFSafeControl':
        """Assign state and action dynamics separately."""
        if action_output_function is None:
            action_output_function = self._create_identity_func()

        return self._create_updated_instance(
            state_dyn=state_dynamics,
            ac_dyn=action_dynamics,
            ac_out_func=action_output_function
        )

    def assign_state_barrier(self, barrier) -> 'InputConstCFSafeControl':
        """Assign state barriers."""
        return self._create_updated_instance(state_barrier=barrier)

    def assign_action_barrier(self, action_barrier, rel_deg: int) -> 'InputConstCFSafeControl':
        """Assign action barriers with relative degree."""
        return self._create_updated_instance(
            ac_barrier=action_barrier,
            ac_rel_deg=rel_deg
        )

    def make(self) -> 'InputConstCFSafeControl':
        """Build the complete input-constrained controller."""
        # Make augmented dynamics
        updated_ctrl = self._make_augmented_dynamics()

        # Make composed barrier function
        updated_ctrl = updated_ctrl._make_composed_barrier()

        # Make auxiliary desired action
        updated_ctrl = updated_ctrl._make_aux_desired_action()

        return updated_ctrl

    def _safe_optimal_control_single(self, x: jnp.ndarray, ret_info: bool = False) -> Union[jnp.ndarray, tuple]:
        """
        Compute safe optimal control for input-constrained system.

        Args:
            x: Single augmented state vector (state_dim + action_dim,)
            ret_info: Whether to return additional information

        Returns:
            If ret_info=False: Control vector (action_dim,)
            If ret_info=True: Tuple (u, slack_vars, constraint_at_u)
        """
        # Get barrier values and Lie derivatives
        hocbf, Lf_hocbf, Lg_hocbf = self._barrier.get_hocbf_and_lie_derivs(x)

        # Apply buffer
        hocbf = hocbf - self._buffer

        # Get auxiliary desired action (single state input)
        u_d = self.aux_desired_action(x)

        # Compute closed-form solution
        omega = Lf_hocbf + jnp.dot(Lg_hocbf, u_d) + self._alpha(hocbf)
        den = jnp.dot(Lg_hocbf, Lg_hocbf) + (1 / self._slack_gain) * hocbf ** 2

        # JIT-friendly conditional
        num = jax.lax.cond(
            self._use_softplus,
            lambda val: jax.nn.softplus(val * self._softplus_gain) / self._softplus_gain,
            lambda val: jax.nn.relu(val),
            -omega
        )

        lam = num / den

        # Compute control
        u = u_d + Lg_hocbf * lam

        # JIT-friendly return
        return jax.lax.cond(
            ret_info,
            lambda _: self._add_optimal_control_info(u, hocbf, Lf_hocbf, Lg_hocbf, lam),
            lambda _: u,
        )

    def _make_composed_barrier(self) -> 'InputConstCFSafeControl':
        """Create composed barrier from state and action barriers."""
        # Remake state barriers with augmented dynamics
        state_barriers = [barrier.assign_dynamics(self._dynamics) for barrier in self._state_barrier]

        # Remake action barriers with augmented dynamics
        action_barriers = [barrier.assign_dynamics(self._dynamics) for barrier in self._ac_barrier]

        # Create composed barrier using static fields
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

        # Make desired control first
        updated_ctrl = self._make_desired_control()

        # Create auxiliary desired action function
        def aux_desired_action_func(x):
            # Action output function applied to action part of state
            ac_out_func = lambda state: updated_ctrl._ac_out_func(state[updated_ctrl._state_dyn.state_dim:])

            # Compute Lie derivative series
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

            # Compute control matrix inverse for auxiliary action
            ac_out_Lg = jnp.linalg.inv(
                lie_deriv(ac_out_func_lie_derivs[-2], updated_ctrl._dynamics.g, x)
            )

            # Compute weighted sum of differences using JAX-compatible operations
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
            # Extract state part and compute desired control
            state_part = x[:self._state_dyn.state_dim]
            Q = self._Q(state_part)  # Q is single-state function
            c = self._c(state_part)  # c is single-state function
            return -jnp.linalg.inv(Q) @ c

        return self

    def _make_augmented_dynamics(self) -> 'InputConstCFSafeControl':
        """Create augmented dynamics combining state and action dynamics."""
        assert self._state_dyn.action_dim == self._ac_dyn.action_dim, \
            'Dimension mismatch between state and action dynamics'

        aug_state_dim = self._state_dyn.state_dim + self._ac_dyn.state_dim
        aug_action_dim = self._state_dyn.action_dim

        # Create augmented dynamics
        dynamics = AffineInControlDynamics(params=None, state_dim=aug_state_dim, action_dim=aug_action_dim)

        def aug_f(x):
            """Augmented drift dynamics for single state."""
            state_part = x[:self._state_dyn.state_dim]
            action_part = x[self._state_dyn.state_dim:]

            # State dynamics with action output
            action_output = self._ac_out_func(action_part)
            state_rhs = self._state_dyn.rhs(state_part, action_output)

            # Action dynamics
            action_rhs = self._ac_dyn.f(action_part)

            return jnp.concatenate([state_rhs, action_rhs])

        def aug_g(x):
            """Augmented control dynamics for single state."""
            action_part = x[self._state_dyn.state_dim:]

            # Zero control influence on state part
            state_g = jnp.zeros((self._state_dyn.state_dim, self._state_dyn.action_dim))

            # Action dynamics control matrix
            action_g = self._ac_dyn.g(action_part)

            return jnp.concatenate([state_g, action_g], axis=0)

        dynamics = dynamics.set_f(aug_f).set_g(aug_g)

        return self._create_updated_instance(dynamics=dynamics)


class MinIntervInputConstCFSafeControl(InputConstCFSafeControl, BaseMinIntervSafeControl):
    """Minimum intervention input-constrained safe control."""

    def assign_desired_control(self, desired_control: Callable) -> 'MinIntervInputConstCFSafeControl':
        """Assign desired control and build controller."""
        updated_ctrl = self._create_updated_instance(
            desired_control=desired_control
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

    def assign_desired_control(self, desired_control: Callable) -> 'MinIntervInputConstCFSafeControlRaw':
        """Assign desired control as auxiliary desired action."""
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