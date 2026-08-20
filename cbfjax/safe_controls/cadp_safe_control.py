"""
Constrained Approximate Dynamic Programming (C-ADP) safe control.

Receding-horizon nonlinear optimal control with state constraints, from

    R. Gutierrez and J. B. Hoagg, "Receding-Horizon Nonlinear Optimal Control
    With Safety Constraints Using Constrained Approximate Dynamic Programming."

The continuous system is xdot = f(x) + g(x) v. The horizon is discretized with
planning step Tp into x_{i+1} = F(x_i) + G(x_i) u_i, where the decision variable
u = [v; delta] carries the CBF slack and

    F(x) = x + Tp f(x),      G(x) = [Tp g(x), 0_{n x 1}].

The slack column of G is zero, so delta does not move the state; it is priced by
r_delta in R_i and relaxes the barrier constraint.

At each step the constraint is the CBF condition on the CONTINUOUS dynamics,

    a(x) = Lf psi(x) + alpha(psi(x)),     b(x) = [Lg psi(x); psi(x)],

with psi the assigned (composed) barrier's HOCBF -- exactly the triple returned
by ``barrier.get_hocbf_and_lie_derivs``.

Each backward step is the closed-form minimizer of a QP with one affine
constraint (eqs. 34-37); this reproduces the algebra of the closed-form filters
in ``closed_form_safe_control`` with H = R_i + G' P_{i+1} G and the slack folded
into the augmented b, but is kept local here so the two evolve independently.
One difference is deliberate: the smoothed multiplier (eq. 41) applies softplus
to the RATIO, whereas the closed-form filters apply their activation to the
numerator. The two agree for the hard multiplier, since the denominator is
positive.

The backward recursion carries (P, T) and evaluates the Jacobians K_i and
Atilde_i of the SMOOTHED step maps at the nominal point (eq. 40). Those
Jacobians are taken with respect to x only, holding (P_{i+1}, T_{i+1}) fixed as
scan carries -- that is what keeps this a Riccati-like recursion rather than
differentiation through the whole horizon.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Callable, Optional

from .base_safe_control import BaseCBFSafeControl
from ..controls.control_types import CADPState, CADPInfo


def _lam_hard(num, den):
    """Exact multiplier of eq. 35: max{0, num/den}."""
    return jax.nn.relu(num / den)


class CADPSafeControl(BaseCBFSafeControl):
    """
    Receding-horizon C-ADP safe control.

    ``optimal_control`` returns the physical control v = [I 0] u*_0(x) of
    eq. 58; the slack delta* of eq. 59 is reported by
    ``optimal_control_with_info``.

    Options are read from ``params``:
        horizon_steps:  N, number of planning steps (static; sets array shapes)
        planning_dt:    Tp, planning step used to discretize the horizon
        softplus_gain:  eta of eq. 12, the smoothing of the multiplier
        num_iter:       forward/backward passes per update. Each pass re-rolls
                        the nominal trajectory under the freshly computed
                        optimal functions and re-runs the recursion, so the
                        nominal converges toward the closed loop it induces.
        refresh_every:  replan every this many calls, so the update period Ts
                        can be a multiple of the zero-order-hold period
        den_eps:        guard added to b'Wb (Assumption 1 makes a > 0 wherever
                        b vanishes, so the multiplier is driven to zero there)
        buffer:         subtracted from the barrier value before use

    Cost weights are constructor arguments and stay traced, so they can be tuned
    or swapped without recompiling. Each accepts a constant, a full length-N
    sequence, or a callable i -> value; all are materialized to sequences.
    """

    # Shapes and pacing must be static.
    _N: int = eqx.field(static=True)
    _num_iter: int = eqx.field(static=True)
    _refresh_every: int = eqx.field(static=True)

    # Traced scalars
    _Tp: jnp.ndarray
    _eta: jnp.ndarray
    _den_eps: jnp.ndarray
    _cadp_buffer: jnp.ndarray

    # Traced cost sequences over the horizon
    _Q_seq: jnp.ndarray      # (N, n, n)
    _Gamma_seq: jnp.ndarray  # (N, n)
    _R_seq: jnp.ndarray      # (N, m, m)  blockdiag(R_v, r_delta)
    _Omega_seq: jnp.ndarray  # (N, m)     [Omega_v; 0]
    _P_N: jnp.ndarray        # (n, n)     terminal Q_N
    _T_N: jnp.ndarray        # (n,)       terminal Gamma_N

    def __init__(
        self,
        Q_state=None,
        Gamma=None,
        R_v=None,
        Omega_v=None,
        r_delta=1.0,
        Q_terminal=None,
        Gamma_terminal=None,
        x_ref=None,
        **kwargs
    ):
        """
        Initialize CADPSafeControl.

        Args:
            Q_state: State cost Q_i, (n, n) positive semidefinite
            Gamma: Linear state cost Gamma_i, (n,); defaults to zeros, or to
                -Q_state @ x_ref when x_ref is given
            R_v: Control cost R_v, (l_v, l_v) positive definite
            Omega_v: Linear control cost, (l_v,); defaults to zeros
            r_delta: Slack weight, positive scalar
            Q_terminal: Terminal Q_N, (n, n); defaults to Q_state
            Gamma_terminal: Terminal Gamma_N, (n,); defaults to Gamma
            x_ref: Convenience goal state; sets Gamma = -Q_state @ x_ref, which
                makes the stage cost 1/2 (x - x_ref)' Q (x - x_ref) up to a
                constant. Pass Gamma directly to control the sign yourself.
            **kwargs: Passed via cooperative inheritance (alpha, barrier,
                dynamics, action_dim, params)
        """
        params = dict(kwargs.pop('params', None) or {})
        params.setdefault('horizon_steps', 10)
        params.setdefault('planning_dt', 0.05)
        params.setdefault('softplus_gain', 1.0)
        params.setdefault('num_iter', 1)
        params.setdefault('refresh_every', 1)
        params.setdefault('den_eps', 1e-12)
        kwargs['params'] = params

        super().__init__(**kwargs)

        if not self.has_dynamics:
            raise ValueError("CADPSafeControl requires dynamics: F and G are "
                             "built by discretizing f and g.")
        if Q_state is None or R_v is None:
            raise ValueError("CADPSafeControl requires 'Q_state' and 'R_v'.")

        params = self._params
        N = int(params['horizon_steps'])
        if N < 1:
            raise ValueError(f"'horizon_steps' must be >= 1, got {N}")
        self._N = N
        self._num_iter = int(params['num_iter'])
        if self._num_iter < 1:
            raise ValueError(f"'num_iter' must be >= 1, got {self._num_iter}")
        self._refresh_every = int(params['refresh_every'])

        self._Tp = jnp.asarray(params['planning_dt'])
        self._eta = jnp.asarray(params['softplus_gain'])
        self._den_eps = jnp.asarray(params['den_eps'])
        self._cadp_buffer = jnp.asarray(params['buffer'])

        n = self._dynamics.state_dim
        lv = self._action_dim
        m = lv + 1

        Q_seq = self._as_seq(Q_state, N, 2, 'Q_state')
        if Gamma is None:
            Gamma = (-jnp.asarray(Q_state) @ jnp.asarray(x_ref)
                     if x_ref is not None else jnp.zeros(n))
        Gamma_seq = self._as_seq(Gamma, N, 1, 'Gamma')

        R_v_seq = self._as_seq(R_v, N, 2, 'R_v')
        Omega_v_seq = self._as_seq(
            jnp.zeros(lv) if Omega_v is None else Omega_v, N, 1, 'Omega_v')
        r_delta_seq = self._as_seq(r_delta, N, 0, 'r_delta')

        self._Q_seq = Q_seq
        self._Gamma_seq = Gamma_seq
        self._R_seq = (jnp.zeros((N, m, m))
                       .at[:, :lv, :lv].set(R_v_seq)
                       .at[:, lv, lv].set(r_delta_seq))
        self._Omega_seq = jnp.zeros((N, m)).at[:, :lv].set(Omega_v_seq)

        self._P_N = jnp.asarray(Q_seq[-1] if Q_terminal is None else Q_terminal)
        self._T_N = jnp.asarray(
            Gamma_seq[-1] if Gamma_terminal is None else Gamma_terminal)

    @staticmethod
    def _as_seq(value, N, base_ndim, name):
        """Materialize a constant, a length-N sequence, or a callable i -> value."""
        if callable(value):
            arr = jnp.stack([jnp.asarray(value(i)) for i in range(N)])
        else:
            arr = jnp.asarray(value)
        if arr.ndim == base_ndim:
            return jnp.broadcast_to(arr, (N,) + arr.shape)
        if arr.ndim == base_ndim + 1 and arr.shape[0] == N:
            return arr
        raise ValueError(
            f"'{name}' must have {base_ndim} dims (constant) or shape "
            f"({N}, ...) (per-step), got shape {arr.shape}"
        )

    def _ctor_defaults(self) -> dict:
        lv = self._action_dim
        return {
            'action_dim': lv,
            'alpha': self._alpha,
            'params': dict(self._params),
            'dynamics': self._dynamics,
            'barrier': self._barrier,
            'terminal_barrier': self._terminal_barrier,
            'Q_state': self._Q_seq,
            'Gamma': self._Gamma_seq,
            'R_v': self._R_seq[:, :lv, :lv],
            'Omega_v': self._Omega_seq[:, :lv],
            'r_delta': self._R_seq[:, lv, lv],
            'Q_terminal': self._P_N,
            'Gamma_terminal': self._T_N,
        }

    # ------------------------------------------------------------ properties

    @property
    def horizon_steps(self) -> int:
        return self._N

    @property
    def aug_action_dim(self) -> int:
        """Dimension of u = [v; delta]."""
        return self._action_dim + 1

    # ------------------------------------------------------------ ingredients

    def _discrete_FG(self, x):
        """Forward-Euler F(x) and G(x), with the zero slack column (eq. 57)."""
        n = self._dynamics.state_dim
        F = x + self._Tp * self._dynamics.f(x)
        G = jnp.concatenate(
            [self._Tp * self._dynamics.g(x), jnp.zeros((n, 1))], axis=1)
        return F, G

    def _constraint(self, x):
        """CBF constraint (eq. 54) on the continuous dynamics: a(x), b(x)."""
        hocbf, lf_hocbf, lg_hocbf = self._barrier.get_hocbf_and_lie_derivs(x)
        hocbf = hocbf - self._cadp_buffer
        a = lf_hocbf + self._alpha(hocbf)
        b = jnp.concatenate([jnp.atleast_1d(lg_hocbf), jnp.atleast_1d(hocbf)])
        return jnp.squeeze(a), b

    def _lam_soft(self):
        """Smoothed multiplier of eq. 41: softplus applied to the ratio."""
        eta = self._eta

        def lam_from(num, den):
            return jax.nn.softplus(eta * (num / den)) / eta
        return lam_from

    def _step(self, x, P_next, T_next, R_i, Omega_i, lam_from):
        """
        One C-ADP step (eqs. 34-37): the closed-form minimizer of
        1/2 u'Hu + c'u subject to a + b'u >= 0, with W = H^-1 and k = -Wc.

        Returns:
            Tuple (u, F, G, k, lam)
        """
        F, G = self._discrete_FG(x)
        a, b = self._constraint(x)

        H = R_i + G.T @ (P_next @ G)
        c = G.T @ (P_next @ F + T_next) + Omega_i

        # One factorization for both solves: columns are W b and W c.
        sol = jnp.linalg.solve(H, jnp.stack([b, c], axis=1))
        Wb, Wc = sol[:, 0], sol[:, 1]
        k = -Wc

        den = jnp.dot(b, Wb) + self._den_eps
        lam = lam_from(-(a + jnp.dot(b, k)), den)
        return k + lam * Wb, F, G, k, lam

    # ------------------------------------------------------------ the passes

    def _backward_pass(self, x_bars):
        """
        Backward recursion (eqs. 38-40) over the nominal trajectory.

        Args:
            x_bars: Nominal trajectory (N+1, n)

        Returns:
            Tuple (P_next, T_next) holding P_1..P_N and T_1..T_N
        """
        m = self.aug_action_dim
        lam_soft = self._lam_soft()

        def body(carry, xs):
            P_next, T_next = carry
            x_bar, Q_i, Gamma_i, R_i, Omega_i = xs

            def smooth_pair(xx):
                u_t, F_t, G_t, _, _ = self._step(
                    xx, P_next, T_next, R_i, Omega_i, lam_soft)
                return jnp.concatenate([u_t, F_t + G_t @ u_t])

            # Jacobians w.r.t. x only; (P_next, T_next) are held fixed.
            jac = jax.jacrev(smooth_pair)(x_bar)
            K_i, A_i = jac[:m], jac[m:]

            u_bar, F_bar, G_bar, _, _ = self._step(
                x_bar, P_next, T_next, R_i, Omega_i, _lam_hard)
            F_star = F_bar + G_bar @ u_bar

            P_i = Q_i + K_i.T @ (R_i @ K_i) + A_i.T @ (P_next @ A_i)
            P_i = 0.5 * (P_i + P_i.T)
            T_i = (A_i.T @ (P_next @ (F_star - A_i @ x_bar) + T_next)
                   + K_i.T @ (R_i @ u_bar - R_i @ (K_i @ x_bar) + Omega_i)
                   + Gamma_i)
            return (P_i, T_i), (P_i, T_i)

        _, (P_seq, T_seq) = jax.lax.scan(
            body,
            (self._P_N, self._T_N),
            (x_bars[:-1], self._Q_seq, self._Gamma_seq,
             self._R_seq, self._Omega_seq),
            reverse=True,
        )

        # scan emits P_0..P_{N-1}; the optimal functions consume P_1..P_N.
        P_full = jnp.concatenate([P_seq, self._P_N[None]])
        T_full = jnp.concatenate([T_seq, self._T_N[None]])
        return P_full[1:], T_full[1:]

    def _forward_pass(self, x0, P_next, T_next):
        """
        Forward pass (eq. 60): roll the previous update's optimal functions.

        Args:
            x0: Current state (n,)
            P_next: P_1..P_N from the previous backward pass (N, n, n)
            T_next: T_1..T_N from the previous backward pass (N, n)

        Returns:
            Nominal trajectory (N+1, n)
        """
        def body(x_bar, xs):
            P_i, T_i, R_i, Omega_i = xs
            u, F_t, G_t, _, _ = self._step(
                x_bar, P_i, T_i, R_i, Omega_i, _lam_hard)
            return F_t + G_t @ u, x_bar

        x_N, x_seq = jax.lax.scan(
            body, x0, (P_next, T_next, self._R_seq, self._Omega_seq))
        return jnp.concatenate([x_seq, x_N[None]])

    def _replan(self, x, state):
        """
        One update: ``num_iter`` forward/backward passes from the stored
        optimal functions. Each iteration re-rolls the nominal trajectory under
        the functions the previous iteration produced, then re-runs the
        recursion along it.
        """
        n = self._dynamics.state_dim

        def one_iter(carry, _):
            P_next, T_next, _ = carry
            x_bars = self._forward_pass(x, P_next, T_next)
            P_new, T_new = self._backward_pass(x_bars)
            return (P_new, T_new, x_bars), None

        init = (state.P_next, state.T_next, jnp.zeros((self._N + 1, n)))
        (P_next, T_next, x_bars), _ = jax.lax.scan(
            one_iter, init, None, length=self._num_iter)
        return P_next, T_next, x_bars

    def _maybe_replan(self, x, state):
        """Replan on the update period; otherwise reuse the stored functions."""
        if self._refresh_every == 1:
            return self._replan(x, state)

        n = self._dynamics.state_dim

        def keep(_):
            return state.P_next, state.T_next, jnp.zeros((self._N + 1, n))

        return jax.lax.cond(
            state.step % self._refresh_every == 0,
            lambda _: self._replan(x, state),
            keep,
            operand=None,
        )

    def _solve(self, x, state):
        """Run the update and evaluate u*_0 at the live state."""
        if state is None:
            state = self.get_init_state()
        P_next, T_next, x_bars = self._maybe_replan(x, state)
        u, _, _, k, lam = self._step(
            x, P_next[0], T_next[0], self._R_seq[0], self._Omega_seq[0],
            _lam_hard)
        new_state = CADPState(P_next=P_next, T_next=T_next, step=state.step + 1)
        return u, k, lam, x_bars, new_state

    # ------------------------------------------------------------ public API

    def get_init_state(self):
        """
        Cold-start controller state.

        Sets P_i = Q_N and T_i = Gamma_N for every i, i.e. the first forward
        pass rolls a terminal-cost-greedy policy. The state carries no x0, so
        no initial nominal trajectory is required from the caller.
        """
        P = jnp.broadcast_to(self._P_N, (self._N,) + self._P_N.shape)
        T = jnp.broadcast_to(self._T_N, (self._N,) + self._T_N.shape)
        return CADPState(P_next=P, T_next=T, step=jnp.asarray(0))

    @jax.jit
    def optimal_control(self, x: jnp.ndarray, state=None) -> tuple:
        """
        Compute the receding-horizon C-ADP control for a single state.

        Args:
            x: Single state vector (state_dim,)
            state: CADPState from get_init_state or the previous call

        Returns:
            Tuple (v, new_state) with v the physical control (eq. 58)
        """
        u, _, _, _, new_state = self._solve(x, state)
        return u[:self._action_dim], new_state

    def optimal_control_with_info(self, x: jnp.ndarray, state=None) -> tuple:
        """
        Compute the C-ADP control with diagnostic info for a single state.

        Args:
            x: Single state vector (state_dim,)
            state: CADPState from get_init_state or the previous call

        ``info.nominal_traj`` is the forward-pass trajectory of this update;
        on a step held by ``refresh_every`` no forward pass runs and it is zero.

        Returns:
            Tuple (v, new_state, info)
        """
        u, k, lam, x_bars, new_state = self._solve(x, state)
        a, b = self._constraint(x)
        info = CADPInfo(
            slack_vars=u[-1],
            constraint_at_u=a + jnp.dot(b, u),
            u_desired=k[:self._action_dim],
            lam=lam,
            nominal_traj=x_bars,
        )
        return u[:self._action_dim], new_state, info

    def get_nominal_traj(self, x: jnp.ndarray, state=None) -> jnp.ndarray:
        """Nominal trajectory (N+1, n) the forward pass builds from x."""
        if state is None:
            state = self.get_init_state()
        return self._forward_pass(x, state.P_next, state.T_next)
