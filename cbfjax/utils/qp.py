"""
Central QP solver registry.

All backends solve:
    min  0.5 u'Qu + c'u    s.t.  Gu <= h  (+ optional Au = b)

Adapters share the signature (Q, c, G, h, A=None, b=None, state=None, **opts)
-> (u, state) and are jit/vmap-compatible. Each adapter extracts the options
it understands from opts (`tol`, `maxiter`) and passes its real solver's
arguments; anything else is ignored. Each adapter carries its warm-start
builder as its `init_state` attribute, signature (n, m, n_eq=0) -> state.
"""

import dataclasses
from typing import Callable

import jax
import jax.numpy as jnp
from qpax import solve_qp_primal
from jaxopt import OSQP
from jaxopt._src.base import KKTSolution
from mpax import create_qp, raPDHG

# Solver name -> adapter function name; get_qp_solver resolves lazily.
QP_SOLVERS: dict[str, str] = {
    'qpax': '_solve_qpax',
    'jaxopt_osqp': '_solve_jaxopt_osqp',
    'mpax': '_solve_mpax',
    'cvxopt': '_solve_cvxopt',
}


def solve_qp(Q, c, G, h, A=None, b=None, state=None, *, solver='qpax', **opts):
    """
    Solve min 0.5 u'Qu + c'u s.t. Gu <= h (+ optional Au = b).

    state is solver-specific warm-start state (None for stateless backends);
    it is threaded through unchanged. Returns (u, state).
    """
    return get_qp_solver(solver)(Q, c, G, h, A, b, state, **opts)


def get_qp_solver(name: str) -> Callable:
    """Return the backend adapter registered under name."""
    try:
        return globals()[QP_SOLVERS[name]]
    except KeyError:
        raise ValueError(
            f"Unknown QP solver '{name}'. Available: {sorted(QP_SOLVERS)}"
        ) from None


def get_qp_init_state(name: str) -> Callable:
    """Return the backend's warm-start init-state builder."""
    return get_qp_solver(name).init_state


def _no_init_state(n: int, m: int, n_eq: int = 0):
    """Warm-start state for stateless backends: always None."""
    return None


def _solve_qpax(Q, c, G, h, A=None, b=None, state=None, **opts):
    """qpax primal interior-point backend (default). No iteration cap."""
    if A is None:
        A = jnp.zeros((0, Q.shape[0]))
        b = jnp.zeros(0)
    u = solve_qp_primal(Q, c, A, b, G, h, solver_tol=opts.get('tol', 1e-5))
    return u, state


_solve_qpax.init_state = _no_init_state


def _solve_cvxopt(Q, c, G, h, A=None, b=None, state=None, **opts):
    """
    CVXOPT backend. CVXOPT is host-side and not traceable, so the solve runs
    through jax.pure_callback: the surrounding computation stays jitted, JAX
    hands the concrete QP data to the host, and the solution comes back as a
    device array. Costs one host sync per solve and is not differentiable.

    opts: maxiter (default 100), tol (feastol/abstol/reltol, default 1e-8).
    """
    n = Q.shape[0]
    if A is None:
        A = jnp.zeros((0, n), dtype=Q.dtype)
        b = jnp.zeros(0, dtype=Q.dtype)

    maxiter = int(opts.get('maxiter', 100))
    tol = float(opts.get('tol', 1e-8))

    def _host_solve(Q_h, c_h, G_h, h_h, A_h, b_h):
        import numpy as np
        from cvxopt import matrix, solvers

        def _m(arr):
            return matrix(np.asarray(arr, dtype=np.float64))

        args = [_m(Q_h), _m(c_h), _m(G_h), _m(h_h)]
        if A_h.shape[0] > 0:
            args += [_m(A_h), _m(b_h)]
        sol = solvers.qp(*args, options={'show_progress': False,
                                         'maxiters': maxiter,
                                         'abstol': tol, 'reltol': tol,
                                         'feastol': tol})
        x = sol['x']
        if x is None:
            return np.zeros(n, dtype=np.float64)
        return np.asarray(x, dtype=np.float64).reshape(-1)

    u = jax.pure_callback(
        _host_solve,
        jax.ShapeDtypeStruct((n,), Q.dtype),
        Q, c, G, h, A, b,
        vmap_method='sequential',
    )
    return u, state


_solve_cvxopt.init_state = _no_init_state


def _solve_jaxopt_osqp(Q, c, G, h, A=None, b=None, state=None, **opts):
    """jaxopt OSQP backend; warm-starts from state (a KKTSolution) and
    returns the final KKTSolution as the new state."""
    solver = OSQP(tol=opts.get('tol', 1e-8), maxiter=opts.get('maxiter', 10000))
    has_eq = (A is not None and A.shape[0] > 0) or (b is not None and b.shape[0] > 0)
    sol = solver.run(
        init_params=state,
        params_obj=(Q, c),
        params_eq=(A, b) if has_eq else None,
        params_ineq=(G, h),
    )
    return sol.params.primal, sol.params


def _init_state_jaxopt_osqp(n: int, m: int, n_eq: int = 0) -> KKTSolution:
    """Zero warm-start state for OSQP (n vars, m inequality rows, n_eq
    equality rows).

    The pytree structure must match what ``solver.run`` returns -- including
    ``dual_eq=None`` when there are no equality constraints -- so the state
    can be carried unchanged across a ``lax.scan``.
    """
    return KKTSolution(
        primal=jnp.zeros(n),
        dual_eq=jnp.zeros(n_eq) if n_eq > 0 else None,
        dual_ineq=jnp.zeros(m),
    )


_solve_jaxopt_osqp.init_state = _init_state_jaxopt_osqp


def _solve_mpax(Q, c, G, h, A=None, b=None, state=None, **opts):
    """mpax raPDHG first-order backend. Gu <= h is passed to mpax as
    -Gu >= -h; variables are unbounded."""
    n = Q.shape[0]
    if A is None:
        A = jnp.zeros((0, n))
        b = jnp.zeros(0)
    lb = jnp.full(n, -jnp.inf, dtype=h.dtype)
    ub = jnp.full(n, jnp.inf, dtype=h.dtype)
    qp = create_qp(Q, c, A, b, -G, -h, lb, ub, use_sparse_matrix=False)
    # dense create_qp sets is_lp = jnp.all(Q == 0), a traced bool under jit;
    # force the concrete QP path
    qp = dataclasses.replace(qp, is_lp=False)
    solver = raPDHG(eps_abs=opts.get('tol', 1e-8),
                    eps_rel=opts.get('tol', 1e-8),
                    iteration_limit=opts.get('maxiter', 2147483647))
    res = solver.optimize(qp)
    return res.primal_solution, state


_solve_mpax.init_state = _no_init_state