"""
QP solver utilities for FlowBarrier.

Provides helper functions for converting JAX arrays to CVXOPT format and solving QPs.
"""

import jax.numpy as jnp
import numpy as np
import cvxopt

# Suppress CVXOPT output
cvxopt.solvers.options['show_progress'] = False


def solve_qp_cvxopt(Q, c, A=None, b = None, G=None ,h=None):
    """
    Solve QP using CVXOPT with JAX array conversion.

    Solves:
        minimize    (1/2) x^T Q x + c^T x
        subject to  G x <= h
                    A x = b

    Args:
        Q: Quadratic cost matrix (n, n) - JAX array
        c: Linear cost vector (n,) - JAX array
        A: Equality constraint matrix (m, n) - JAX array
        b: Equality constraint vector (m,) - JAX array
        G: Inequality constraint matrix (p, n) - JAX array
        h: Inequality constraint vector (p,) - JAX array

    Returns:
        Solution vector x (n,) as JAX array
    """
    # Convert JAX arrays to numpy then to CVXOPT matrices
    P = cvxopt.matrix(np.array(Q, dtype=np.float64))
    q = cvxopt.matrix(np.array(c, dtype=np.float64))

    G_cvx = None
    h_cvx = None
    if G is not None and h is not None:
        G_cvx = cvxopt.matrix(np.array(G, dtype=np.float64))
        h_cvx = cvxopt.matrix(np.array(h, dtype=np.float64))

    A_cvx = None
    b_cvx = None
    if A is not None and b is not None:
        A_cvx = cvxopt.matrix(np.array(A, dtype=np.float64))
        b_cvx = cvxopt.matrix(np.array(b, dtype=np.float64))

    # Solve QP
    sol = cvxopt.solvers.qp(P, q, G_cvx, h_cvx, A_cvx, b_cvx)

    # Convert solution back to JAX array
    return jnp.array(np.array(sol['x']).flatten(), dtype=jnp.float64)