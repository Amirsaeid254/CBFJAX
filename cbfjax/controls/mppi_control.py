"""
MPPI (Model Predictive Path Integral) Control using JAX.

Fully JIT-compiled: vmap over K trajectory samples, lax.scan over N time steps.
No Python loops in the hot path.

Stateful interface (Optax-style):
- optimal_control(x, state) -> (u, new_state)
- get_init_state()                  -> MPPIState(U, key)

State carries the warm-start nominal trajectory U and the PRNGKey so the
controller is a pure function — suitable for jax.lax.scan in ZOH integration.

Dynamics must expose discrete_rhs(x, u), i.e. be configured with
  params = {'discretization_dt': dt, 'discretization_method': 'euler'|'rk4'}
using the same dt as params['time_steps'] here.

Usage::

    ctrl = MPPIControl(
        action_dim=2,
        params={'num_samples': 1000, 'horizon': 2.0,
                'time_steps': 0.1, 'temperature': 1.0},
        dynamics=dynamics,
        cost_func=lambda x, u, t: ...,          # running cost
        terminal_cost_func=lambda x: ...,       # optional
        noise_sigma=[0.5, 0.5],
        control_low=[-1., -1.],                 # optional
        control_high=[1., 1.],
    )

    state = ctrl.get_init_state()
    u, state = ctrl.optimal_control(x, state)
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Callable, Optional, Tuple
from immutabledict import immutabledict

from .base_control import BaseControl
from .control_types import MPPIState, MPPIInfo


class MPPIControl(BaseControl):
    """
    Model Predictive Path Integral (MPPI) controller.

    Parallelism structure:
      - jax.vmap over K samples    (outer, embarrassingly parallel)
      - jax.lax.scan over N steps  (inner, sequential within each rollout)

    All K*N forward-model evaluations compile into a single XLA kernel.

    Params (passed via params dict or directly):
        num_samples  (int,   default 1000)  K — number of trajectory samples
        horizon      (float, default 2.0)   prediction horizon in seconds
        time_steps   (float, default 0.1)   dt — must match dynamics.discrete_rhs
        temperature  (float, default 1.0)   lambda — information-theoretic temperature
        init_seed    (int,   default 0)     seed used only in get_init_state()
    """

    _cost_func:          Optional[Callable] = eqx.field(static=True)
    _terminal_cost_func: Optional[Callable] = eqx.field(static=True)
    _noise_sigma:        tuple               = eqx.field(static=True)
    _control_low:        tuple               = eqx.field(static=True)
    _control_high:       tuple               = eqx.field(static=True)
    _has_control_bounds: bool                = eqx.field(static=True)

    def __init__(
        self,
        cost_func:          Optional[Callable] = None,
        terminal_cost_func: Optional[Callable] = None,
        noise_sigma:        Optional[tuple]    = None,
        control_low:        Optional[tuple]    = None,
        control_high:       Optional[tuple]    = None,
        **kwargs
    ):
        params = kwargs.get('params', None)
        default_params = {
            'num_samples': 1000,
            'horizon':     2.0,
            'time_steps':  0.1,
            'temperature': 1.0,
            'init_seed':   0,
        }
        if params is not None:
            default_params.update(params)
        kwargs['params'] = immutabledict(default_params)

        super().__init__(**kwargs)

        self._cost_func          = cost_func
        self._terminal_cost_func = terminal_cost_func or self._make_zero_terminal()

        action_dim = self._action_dim
        self._noise_sigma = tuple(float(s) for s in (noise_sigma or [1.0] * action_dim))

        if control_low is not None and control_high is not None:
            self._control_low        = tuple(float(v) for v in control_low)
            self._control_high       = tuple(float(v) for v in control_high)
            self._has_control_bounds = True
        else:
            self._control_low        = tuple(0.0 for _ in range(action_dim))
            self._control_high       = tuple(0.0 for _ in range(action_dim))
            self._has_control_bounds = False

    @staticmethod
    def _make_zero_terminal() -> Callable:
        def zero_terminal(x):
            return jnp.zeros(())
        return zero_terminal

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _ctor_defaults(self) -> dict:
        return {
            'action_dim':         self._action_dim,
            'params':             immutabledict(self._params) if self._params else None,
            'dynamics':           self._dynamics,
            'cost_func':          self._cost_func,
            'terminal_cost_func': self._terminal_cost_func,
            'noise_sigma':        self._noise_sigma,
            'control_low':        self._control_low  if self._has_control_bounds else None,
            'control_high':       self._control_high if self._has_control_bounds else None,
        }

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def N_horizon(self) -> int:
        return int(self._params['horizon'] / self._params['time_steps'])

    @property
    def num_samples(self) -> int:
        return int(self._params['num_samples'])

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def get_init_state(self) -> MPPIState:
        """Return initial state: zero nominal trajectory + seeded PRNGKey."""
        key = jax.random.PRNGKey(int(self._params['init_seed']))
        return MPPIState(
            U=jnp.zeros((self.N_horizon, self._action_dim)),
            key=key,
        )

    def set_init_guess(self, U: jnp.ndarray = None,
                       state: MPPIState = None) -> MPPIState:
        """Overwrite the nominal trajectory in a controller state."""
        if state is None:
            state = self.get_init_state()
        if U is None:
            U = jnp.zeros((self.N_horizon, self._action_dim))
        return state._replace(U=jnp.asarray(U))

    # ------------------------------------------------------------------
    # Core MPPI computation (factored to avoid duplication)
    # ------------------------------------------------------------------

    def _mppi_update(
        self,
        x:   jnp.ndarray,   # (state_dim,)
        U:   jnp.ndarray,   # (N, action_dim)  current nominal
        key: jnp.ndarray,   # PRNGKey
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        One MPPI update step.

        Returns:
            u_star  (action_dim,)   first action of updated nominal
            U_new   (N, action_dim) updated nominal trajectory (before shift)
            new_key PRNGKey         to store in next MPPIState
            S       (K,)            per-sample total costs
            w       (K,)            normalized importance weights
        """
        K    = self.num_samples
        N    = self.N_horizon
        lam  = float(self._params['temperature'])
        sigma = jnp.array(self._noise_sigma)             # (action_dim,)

        # Consume one key split; carry new_key forward into next state
        new_key, use_key = jax.random.split(key)

        # ε ~ N(0, diag(σ²))  shape: (K, N, action_dim)
        eps = jax.random.normal(use_key, shape=(K, N, self._action_dim)) * sigma[None, None, :]

        # Perturbed controls — clipped to bounds before simulation so the
        # cost and dynamics see feasible actions.  V shape: (K, N, action_dim)
        V = U[None] + eps                               # broadcast nominal over K
        if self._has_control_bounds:
            V = jnp.clip(V, jnp.array(self._control_low), jnp.array(self._control_high))

        # Effective perturbation = what was actually applied to the dynamics.
        # Using eps_eff (not raw eps) in the weighted update ensures consistency:
        # the update direction matches the simulated trajectories exactly.
        eps_eff = V - U[None]                           # (K, N, action_dim)

        # -- rollout: vmap over K, scan over N --------------------------
        def rollout_one(v_k: jnp.ndarray) -> jnp.ndarray:
            """Roll out one perturbed trajectory; return total cost."""

            def step(x_t, args):
                t, u_t = args
                cost_t = self._cost_func(x_t, u_t, t)
                x_next = self._dynamics.discrete_rhs(x_t, u_t)
                return x_next, cost_t

            x_final, step_costs = jax.lax.scan(step, x, (jnp.arange(N), v_k))
            return jnp.sum(step_costs) + self._terminal_cost_func(x_final)

        S = jax.vmap(rollout_one)(V)                 # (K,)

        # -- importance weights (numerically stable) --------------------
        # Guard: if any S is non-finite (fp32 overflow), replace with large finite value
        S    = jnp.where(jnp.isfinite(S), S, jnp.finfo(S.dtype).max / 2)
        beta = jnp.min(S)
        w    = jnp.exp(-(S - beta) / lam)
        w    = w / (jnp.sum(w) + jnp.finfo(w.dtype).tiny)  # guard all-zero weights

        # -- weighted perturbation update  ------------------------------
        delta_U = jnp.einsum('k,knt->nt', w, eps_eff)  # (N, action_dim)

        U_new = U + delta_U
        if self._has_control_bounds:
            U_new = jnp.clip(
                U_new,
                jnp.array(self._control_low),
                jnp.array(self._control_high),
            )

        u_star = U_new[0]
        return u_star, U_new, new_key, S, w

    # ------------------------------------------------------------------
    # Stateful interface
    # ------------------------------------------------------------------

    @jax.jit
    def optimal_control(
        self,
        x:     jnp.ndarray,
        state: MPPIState = None,
    ) -> Tuple[jnp.ndarray, MPPIState]:
        if state is None:
            state = self.get_init_state()

        u_star, U_new, new_key, _, _ = self._mppi_update(x, state.U, state.key)

        # Shift nominal left; repeat last action for the new tail slot
        U_shifted = jnp.concatenate([U_new[1:], U_new[-1:]], axis=0)
        return u_star, MPPIState(U=U_shifted, key=new_key)

    def optimal_control_with_info(
        self,
        x:     jnp.ndarray,
        state: MPPIState = None,
    ) -> Tuple[jnp.ndarray, MPPIState, MPPIInfo]:
        if state is None:
            state = self.get_init_state()

        u_star, U_new, new_key, S, w = self._mppi_update(x, state.U, state.key)

        U_shifted = jnp.concatenate([U_new[1:], U_new[-1:]], axis=0)
        info = MPPIInfo(S=S, weights=w, U_new=U_new)
        return u_star, MPPIState(U=U_shifted, key=new_key), info

    # ------------------------------------------------------------------
    # Trajectory prediction — all K sampled rollouts
    # ------------------------------------------------------------------

    @jax.jit
    def get_predicted_trajectories(
        self,
        x:     jnp.ndarray,
        state: MPPIState = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Roll out all K sampled trajectories from the current state.

        Uses state.key for noise sampling without advancing it, so this is
        a pure read-only visualisation call that does not affect the control loop.

        Returns:
            x_trajs  (K, N+1, state_dim)  all sampled trajectories
            u_trajs  (K, N,   action_dim) corresponding perturbed controls
            weights  (K,)                  importance weights (for colour-coding)
        """
        if state is None:
            state = self.get_init_state()

        K    = self.num_samples
        N    = self.N_horizon
        lam  = float(self._params['temperature'])
        sigma = jnp.array(self._noise_sigma)
        U    = state.U

        # Sample noise — read state.key without splitting so the control loop
        # is unaffected.
        eps = jax.random.normal(state.key, shape=(K, N, self._action_dim)) * sigma[None, None, :]

        # Perturbed control sequences: (K, N, action_dim)
        V = U[None] + eps
        if self._has_control_bounds:
            V = jnp.clip(V, jnp.array(self._control_low), jnp.array(self._control_high))

        def rollout_one(v_k: jnp.ndarray):
            """Roll out one perturbed sequence; collect every state visited."""

            def step(x_t, args):
                t, u_t = args
                cost_t = self._cost_func(x_t, u_t, t)
                x_next = self._dynamics.discrete_rhs(x_t, u_t)
                return x_next, (x_t, cost_t)        # carry, (visited_state, cost)

            x_final, (x_visited, step_costs) = jax.lax.scan(
                step, x, (jnp.arange(N), v_k)
            )
            # x_visited: (N, state_dim) — states x_0 … x_{N-1}
            x_traj  = jnp.concatenate([x_visited, x_final[None]], axis=0)  # (N+1,)
            total_cost = jnp.sum(step_costs) + self._terminal_cost_func(x_final)
            return x_traj, total_cost

        x_trajs, S = jax.vmap(rollout_one)(V)   # (K, N+1, state_dim), (K,)

        # Importance weights — same formula as _mppi_update
        S       = jnp.where(jnp.isfinite(S), S, jnp.finfo(S.dtype).max / 2)
        beta    = jnp.min(S)
        weights = jnp.exp(-(S - beta) / lam)
        weights = weights / (jnp.sum(weights) + jnp.finfo(weights.dtype).tiny)

        return x_trajs, V, weights