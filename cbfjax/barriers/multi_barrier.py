"""
MultiBarriers class

This module implements MultiBarriers.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from typing import List, Optional, Tuple

from .barrier import Barrier
from cbfjax.dynamics.base_dynamic import DummyDynamics

class MultiBarriers(Barrier):
    """
    MultiBarriers implementation.

    Manages multiple barrier functions, storing their barrier functions,
    HOCBF functions, and barrier series separately.
    """

    # Additional fields for multi-barriers
    _mb_barrier_funcs: tuple = eqx.field(static=True)
    _mb_hocbf_funcs: tuple = eqx.field(static=True)
    _mb_barriers: tuple = eqx.field(static=True)
    _multidim_indices: tuple = eqx.field(static=True)  # Indices of multi-dimensional barriers

    def __init__(self, barrier_func=None, dynamics=None, rel_deg=1, alphas=None,
                 barriers=None, hocbf_func=None, cfg=None,
                 barrier_funcs=None, hocbf_funcs=None, multidim_indices=None,
                 multidim=False):
        """
        Initialize MultiBarriers.

        Construction from Barrier objects:

            mb = MultiBarriers(barriers=[b1, b2], dynamics=dyn)

        If dynamics is None it is inferred from the first barrier.

        Args:
            barrier_func: Not used in MultiBarriers
            dynamics: System dynamics object
            rel_deg: Not used in MultiBarriers
            alphas: Not used in MultiBarriers
            barriers: List of Barrier objects, or tuple of barrier series
                (internal use by _replace)
            hocbf_func: Not used in MultiBarriers
            cfg: Configuration dictionary
            barrier_funcs: Tuple of barrier functions from added barriers
            hocbf_funcs: Tuple of HOCBF functions from added barriers
            multidim_indices: Tuple of indices for multi-dimensional barriers
            multidim: If True, mark constructor-provided barriers as multi-dimensional
        """
        # Barriers given as Barrier objects: derive functions and series from them
        if barriers and all(isinstance(b, Barrier) for b in barriers):
            barrier_objs = list(barriers)
            if dynamics is None or isinstance(dynamics, DummyDynamics):
                dynamics = barrier_objs[0]._dynamics
            barrier_funcs = tuple(b.barrier for b in barrier_objs)
            hocbf_funcs = tuple(b.hocbf for b in barrier_objs)
            barriers = tuple(b.barriers for b in barrier_objs)
            multidim_indices = tuple(range(len(barrier_objs))) if multidim else ()

        # Initialize parent with minimal fields
        super().__init__(
            barrier_func=barrier_func,
            dynamics=dynamics,
            rel_deg=rel_deg,
            alphas=alphas,
            barriers=tuple(),  # series are stored in _mb_barriers
            hocbf_func=hocbf_func,
            cfg=cfg
        )

        # Initialize multi-barrier specific fields
        self._mb_barrier_funcs = tuple(barrier_funcs or ())
        self._mb_hocbf_funcs = tuple(hocbf_funcs or ())
        self._mb_barriers = tuple(barriers or ())
        self._multidim_indices = tuple(multidim_indices or ())

    @property
    def _barrier_funcs(self) -> tuple:
        return self._mb_barrier_funcs

    @property
    def _hocbf_funcs(self) -> tuple:
        return self._mb_hocbf_funcs

    @property
    def _barriers(self) -> tuple:
        return self._mb_barriers

    def _ctor_defaults(self) -> dict:
        """Constructor kwargs capturing current field values (per-class)."""
        return {
            'dynamics': self._dynamics,
            'cfg': self.cfg,
            'barrier_funcs': self._barrier_funcs,
            'hocbf_funcs': self._hocbf_funcs,
            'barriers': self._barriers,
            'multidim_indices': self._multidim_indices
        }

    def barrier(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Compute main barrier values at a single state.

        Main barrier value is the barrier which defines all the higher order cbfs
        involved in the composite barrier function expression.

        Args:
            x: Single state vector (n,)

        Returns:
            Array of barrier values (num_barriers,). Batch with jax.vmap(self.barrier).
        """
        if not self._barrier_funcs:
            raise ValueError("No barriers added. Construct with a barriers list.")

        return jnp.array([barrier_func(x) for barrier_func in self._barrier_funcs])

    def hocbf(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Compute HOCBF values at a single state.

        Args:
            x: Single state vector (n,)

        Returns:
            Array of HOCBF values (num_barriers,). Batch with jax.vmap(self.hocbf).
        """
        if not self._hocbf_funcs:
            raise ValueError("No barriers added. Construct with a barriers list.")

        return jnp.concatenate([jnp.atleast_1d(hocbf_func(x)) for hocbf_func in self._hocbf_funcs])

    def get_hocbf_and_lie_derivs(self, x: jnp.ndarray) -> tuple:
        """
        Compute HOCBF and Lie derivatives at a single state.

        Args:
            x: Single state vector (n,)

        Returns:
            Tuple of (hocbf_values, Lf_hocbf, Lg_hocbf) with shapes
            ((M,), (M,), (M, action_dim)) for all barriers
        """
        if not self._hocbf_funcs:
            raise ValueError("No barriers added. Construct with a barriers list.")
        if self._dynamics is None:
            raise ValueError("Dynamics not assigned. Construct with dynamics.")

        f_val = self._dynamics.f(x)
        g_val = self._dynamics.g(x)

        hocbf_values = []
        lf_values = []
        lg_values = []

        # Process each barrier function
        for i, hocbf_func in enumerate(self._hocbf_funcs):
            if i in self._multidim_indices:
                # Multi-dimensional barrier: use jacrev with has_aux
                def _hocbf_with_aux(x, _func=hocbf_func):
                    val = _func(x)
                    return val, val
                jac_hocbf, barrier_val = jax.jacrev(_hocbf_with_aux, has_aux=True)(x)

                lf_vals = jnp.einsum('ij,j->i', jac_hocbf, f_val)
                lg_vals = jnp.einsum('ij,jk->ik', jac_hocbf, g_val)

                hocbf_values.extend(barrier_val)
                lf_values.extend(lf_vals)
                lg_values.extend(lg_vals)
            else:
                # Scalar barrier: use value_and_grad
                barrier_val, grad_hocbf = jax.value_and_grad(hocbf_func)(x)

                lf_val = jnp.dot(grad_hocbf, f_val)
                lg_val = grad_hocbf @ g_val

                hocbf_values.append(barrier_val)
                lf_values.append(lf_val)
                lg_values.append(lg_val)

        # Stack results
        hocbf_vals = jnp.array(hocbf_values)
        lf_hocbf = jnp.array(lf_values)
        lg_hocbf = jnp.array(lg_values)

        return hocbf_vals, lf_hocbf, lg_hocbf

    def Lf_hocbf(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Compute Lie derivative of highest-order barrier function w.r.t. f.

        Args:
            x: Single state vector (n,)

        Returns:
            Lie derivatives with shape (total_barriers,)
        """
        _, lf_hocbf, _ = self.get_hocbf_and_lie_derivs(x)
        return lf_hocbf

    def Lg_hocbf(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Compute Lie derivative of highest-order barrier function w.r.t. g.

        Args:
            x: Single state vector (n,)

        Returns:
            Lie derivatives with shape (total_barriers, action_dim)
        """
        _, _, lg_hocbf = self.get_hocbf_and_lie_derivs(x)
        return lg_hocbf

    def min_barrier(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Calculate the minimum value among all the barrier values at a single state.

        Args:
            x: Single state vector (n,)

        Returns:
            Minimum barrier value, scalar
        """
        return jnp.min(self.barrier(x))


    @property
    def barriers_flatten(self) -> tuple:
        """
        Get flattened list of all barrier functions.

        Returns:
            Tuple of all barrier functions from all barrier series
        """
        flat = []
        for barrier_series in self._barriers:
            for b in barrier_series:
                flat.append(b)
        return tuple(flat)