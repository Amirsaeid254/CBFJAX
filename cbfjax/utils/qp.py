"""
Central QP solver module with a configurable backend registry.

All backends solve:
    min  0.5 u'Qu + c'u    s.t.  Gu <= h  (+ optional Au = b)

Backends are thin adapters with a unified signature, selected by name once
(statically) via get_qp_solver(); the returned function is jit/vmap-compatible.
"""

import dataclasses

import jax.numpy as jnp
from typing import Callable, Dict, Optional, Tuple

from qpax import solve_qp_primal
from jaxopt import OSQP
from jaxopt._src.base import KKTSolution
from mpax import create_qp, raPDHG

from .qp_gpu import solve_admm_woodbury
from .reluqp_jax import solve_reluqp_ineq


def _has_eq_const(A, b) -> bool:
    return (A is not None and A.shape[0] > 0) or (b is not None and b.shape[0] > 0)


def _no_init_state(n: int, m: int, n_eq: int = 0):
    """Init warm-start state for stateless backends: always None."""
    return None


def _solve_qpax(Q, c, G, h, A=None, b=None, state=None, **opts):
    """qpax primal interior-point backend (default)."""
    if A is None:
        A = jnp.zeros((0, Q.shape[0]))
        b = jnp.zeros(0)
    u = solve_qp_primal(Q, c, A, b, G, h, **opts)
    return u, state


def _solve_admm_woodbury(Q, c, G, h, A=None, b=None, state=None, **opts):
    """
    EXPERIMENTAL: ADMM + Woodbury backend (diagonal Q, inequality-only).

    Extracts diag(Q) when a 2-D Q is given; off-diagonal terms are ignored.
    Fixed iteration count (see cbfjax.utils.qp_gpu.solve_admm_woodbury).
    """
    if _has_eq_const(A, b):
        raise ValueError("'admm_woodbury' backend supports inequality constraints only")
    Q_diag = jnp.diag(Q) if Q.ndim == 2 else Q
    l = jnp.full(h.shape[0], -jnp.inf, dtype=h.dtype)
    u = solve_admm_woodbury(Q_diag, c, G, l, h, **opts)
    return u, state


def _solve_reluqp(Q, c, G, h, A=None, b=None, state=None, **opts):
    """
    EXPERIMENTAL: ReLU-QP backend (inequality-only).

    Fixed iteration count (see cbfjax.utils.reluqp_jax.solve_reluqp_ineq).
    """
    if _has_eq_const(A, b):
        raise ValueError("'reluqp' backend supports inequality constraints only")
    u = solve_reluqp_ineq(Q, c, G, h, **opts)
    return u, state


_OSQP_SETTINGS = {'tol': 1e-8, 'maxiter': 10000}
_OSQP = OSQP(**_OSQP_SETTINGS)


def _solve_jaxopt_osqp(Q, c, G, h, A=None, b=None, state=None, **opts):
    """
    jaxopt boxed-OSQP backend.

    Warm-starts from state (a jaxopt KKTSolution) when given; returns the
    final KKTSolution as the new state. opts (e.g. tol, maxiter) override
    the module-level solver settings.
    """
    solver = OSQP(**{**_OSQP_SETTINGS, **opts}) if opts else _OSQP
    params_eq = (A, b) if _has_eq_const(A, b) else None
    sol = solver.run(
        init_params=state,
        params_obj=(Q, c),
        params_eq=params_eq,
        params_ineq=(G, h),
    )
    return sol.params.primal, sol.params


def _init_state_jaxopt_osqp(n: int, m: int, n_eq: int = 0) -> KKTSolution:
    """
    Structured zero warm-start state (a jaxopt KKTSolution) for OSQP.

    n: decision variables, m: inequality rows, n_eq: equality rows.
    The pytree structure must match what ``solver.run`` returns so the state
    can be carried unchanged across a ``lax.scan`` (cold start = this zero
    state, which OSQP treats as an ordinary warm start from the origin).
    ``dual_eq`` is None when there are no equality constraints, matching the
    structure OSQP produces with ``params_eq=None``.
    """
    return KKTSolution(
        primal=jnp.zeros(n),
        dual_eq=jnp.zeros(n_eq) if n_eq > 0 else None,
        dual_ineq=jnp.zeros(m),
    )


def _solve_mpax(Q, c, G, h, A=None, b=None, state=None, **opts):
    """
    EXPERIMENTAL: mpax raPDHG first-order backend.

    Gu <= h is passed to mpax as -Gu >= -h; variables are unbounded.
    opts (e.g. eps_abs, eps_rel, iteration_limit) are forwarded to raPDHG.
    """
    n = Q.shape[0]
    if A is None:
        A = jnp.zeros((0, n))
        b = jnp.zeros(0)
    l = jnp.full(n, -jnp.inf, dtype=h.dtype)
    ub = jnp.full(n, jnp.inf, dtype=h.dtype)
    qp = create_qp(Q, c, A, b, -G, -h, l, ub, use_sparse_matrix=False)
    # dense create_qp sets is_lp = jnp.all(Q == 0), a traced bool under jit;
    # force the concrete QP path
    qp = dataclasses.replace(qp, is_lp=False)
    solver = raPDHG(**{'eps_abs': 1e-8, 'eps_rel': 1e-8, **opts})
    res = solver.optimize(qp)
    return res.primal_solution, state


QP_SOLVERS: Dict[str, Callable] = {
    'qpax': _solve_qpax,
    'admm_woodbury': _solve_admm_woodbury,
    'reluqp': _solve_reluqp,
    'jaxopt_osqp': _solve_jaxopt_osqp,
    'mpax': _solve_mpax,
}

# Parallel registry of warm-start init-state builders, keyed identically to
# QP_SOLVERS. init_state_fn(n, m, n_eq) -> warm-start state for the next solve.
# Stateless backends use _no_init_state (always None); their solve adapter
# threads that None straight through, so a None state lane is a valid no-op.
QP_INIT_STATES: Dict[str, Callable] = {
    'qpax': _no_init_state,
    'admm_woodbury': _no_init_state,
    'reluqp': _no_init_state,
    'jaxopt_osqp': _init_state_jaxopt_osqp,
    'mpax': _no_init_state,
}


def get_qp_solver(name: str) -> Callable:
    """Return the backend adapter registered under name."""
    try:
        return QP_SOLVERS[name]
    except KeyError:
        raise ValueError(
            f"Unknown QP solver '{name}'. Available: {sorted(QP_SOLVERS)}"
        ) from None


def get_qp_init_state(name: str) -> Callable:
    """
    Return the warm-start init-state builder registered under name.

    The returned callable has signature ``(n, m, n_eq=0) -> state`` where n is
    the number of decision variables, m the number of inequality rows, and n_eq
    the number of equality rows. It returns None for stateless backends and a
    structured zero state (matching the solver's output pytree) for warm-start
    backends such as jaxopt_osqp.
    """
    try:
        return QP_INIT_STATES[name]
    except KeyError:
        raise ValueError(
            f"Unknown QP solver '{name}'. Available: {sorted(QP_INIT_STATES)}"
        ) from None


def solve_qp(Q, c, G, h, A=None, b=None, state=None, *, solver='qpax', **opts):
    """
    Solve min 0.5 u'Qu + c'u s.t. Gu <= h (+ optional Au = b).

    state is solver-specific warm-start state (None for stateless backends);
    it is threaded through unchanged. Returns (u, state).
    """
    return get_qp_solver(solver)(Q, c, G, h, A, b, state, **opts)
