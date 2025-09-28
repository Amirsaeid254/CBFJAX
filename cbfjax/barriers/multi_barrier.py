"""
MultiBarriers class

This module implements MultiBarriers.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from typing import List, Optional, Tuple

from .barrier import Barrier
from cbfjax.utils.utils import apply_and_batchize, apply_and_batchize_tuple, lie_deriv, ensure_batch_dim
from cbfjax.dynamics.base import DummyDynamics

class MultiBarriers(Barrier):
    """
    MultiBarriers implementation.

    Manages multiple barrier functions, storing their barrier functions,
    HOCBF functions, and barrier series separately.
    """

    # Additional fields for multi-barriers
    _barrier_funcs: tuple = eqx.field(static=True)
    _hocbf_funcs: tuple = eqx.field(static=True)
    _barriers: tuple = eqx.field(static=True)  # Override parent's _barriers to store list of barrier series
    _multidim_indices: tuple = eqx.field(static=True)  # Indices of multi-dimensional barriers

    def __init__(self, barrier_func=None, dynamics=None, rel_deg=1, alphas=None,
                 barriers=None, hocbf_func=None, cfg=None,
                 barrier_funcs=None, hocbf_funcs=None, multidim_indices=None):
        """
        Initialize MultiBarriers.

        Args:
            barrier_func: Not used in MultiBarriers
            dynamics: System dynamics object
            rel_deg: Not used in MultiBarriers
            alphas: Not used in MultiBarriers
            barriers: Tuple of barrier series from added barriers
            hocbf_func: Not used in MultiBarriers
            cfg: Configuration dictionary
            barrier_funcs: Tuple of barrier functions from added barriers
            hocbf_funcs: Tuple of HOCBF functions from added barriers
            multidim_indices: Tuple of indices for multi-dimensional barriers
        """
        # Initialize parent with minimal fields
        super().__init__(
            barrier_func=barrier_func,
            dynamics=dynamics,
            rel_deg=rel_deg,
            alphas=alphas,
            barriers=tuple(),  # Start with empty
            hocbf_func=hocbf_func,
            cfg=cfg
        )

        # Initialize multi-barrier specific fields
        self._barrier_funcs = tuple(barrier_funcs or ())
        self._hocbf_funcs = tuple(hocbf_funcs or ())
        self._barriers = tuple(barriers or ())
        self._multidim_indices = tuple(multidim_indices or ())

    @classmethod
    def create_empty(cls, cfg=None):
        """
        Create empty MultiBarriers instance.

        Args:
            cfg: Optional configuration dictionary

        Returns:
            Empty MultiBarriers ready for barrier addition
        """
        return cls(cfg=cfg)

    def assign(self, barrier_func, rel_deg=1, alphas=None):
        """

        Raises:
            Exception: Always raised to direct to add_barriers method
        """
        raise Exception("Use add_barriers method to add barriers for MultiBarriers class")

    def add_barriers(self, barriers: List[Barrier], infer_dynamics: bool = False, multidim: bool = False) -> 'MultiBarriers':
        """
        Add barriers to MultiBarriers collection.

        Args:
            barriers: List of Barrier objects to add
            infer_dynamics: If True, infer dynamics from first barrier
            multidim: If True, mark these barriers as multi-dimensional

        Returns:
            New MultiBarriers instance with added barriers
        """
        # Infer dynamics of the first barrier if infer_dynamics = True and dynamics is not already assigned
        dynamics = self._dynamics
        if infer_dynamics:
            # Check if dynamics is None or DummyDynamics
            if self._dynamics is None or isinstance(self._dynamics, DummyDynamics):
                dynamics = barriers[0]._dynamics

        # Extend the lists (convert to tuples for JAX)
        new_barrier_funcs = list(self._barrier_funcs)
        new_hocbf_funcs = list(self._hocbf_funcs)
        new_barriers = list(self._barriers)
        new_multidim_indices = list(self._multidim_indices)

        # Add new barriers - store the _single methods to match CBFJAX structure
        base_idx = len(self._hocbf_funcs)
        new_barrier_funcs.extend([barrier._barrier_single for barrier in barriers])
        new_hocbf_funcs.extend([barrier._hocbf_single for barrier in barriers])
        new_barriers.extend([barrier.barriers for barrier in barriers])

        # If multidim=True, mark these new barriers as multi-dimensional
        if multidim:
            for i in range(len(barriers)):
                new_multidim_indices.append(base_idx + i)

        return MultiBarriers(
            dynamics=dynamics,
            cfg=self.cfg,
            barrier_funcs=tuple(new_barrier_funcs),
            hocbf_funcs=tuple(new_hocbf_funcs),
            barriers=tuple(new_barriers),
            multidim_indices=tuple(new_multidim_indices)
        )

    def assign_dynamics(self, dynamics) -> 'MultiBarriers':
        """
        Assign dynamics to MultiBarriers.

        Args:
            dynamics: System dynamics object

        Returns:
            New MultiBarriers with assigned dynamics
        """
        if self._dynamics is not None and hasattr(self._dynamics, 'f'):
            import warnings
            warnings.warn('The assigned dynamics is overridden by the dynamics of the'
                         ' first barrier on the barriers list')

        return MultiBarriers(
            dynamics=dynamics,
            cfg=self.cfg,
            barrier_funcs=self._barrier_funcs,
            hocbf_funcs=self._hocbf_funcs,
            barriers=self._barriers,
            multidim_indices=self._multidim_indices
        )

    def _barrier_single(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Compute barrier values for single state vector.

        Args:
            x: Single state vector (n,)

        Returns:
            Array of barrier values (num_barriers,)
        """
        if not self._barrier_funcs:
            raise ValueError("No barriers added. Use add_barriers() first.")

        # Compute all barrier values for single state
        return jnp.array([barrier_func(x) for barrier_func in self._barrier_funcs])

    def barrier(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Compute main barrier value at x.

        Main barrier value is the barrier which defines all the higher order cbfs
        involved in the composite barrier function expression.
        This method returns a horizontally stacked array of the value of barriers at x.

        Args:
            x: State vector (n,) or batch (batch, n)

        Returns:
            Barrier values with shape (batch, len(self._barrier_funcs), 1)
        """
        if not self._barrier_funcs:
            raise ValueError("No barriers added. Use add_barriers() first.")

        # Stack and transpose
        stacked = jnp.stack([apply_and_batchize(barrier_func, x) for barrier_func in self._barrier_funcs])
        return jnp.transpose(stacked, (1, 0, 2))

    def _hocbf_single(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Compute HOCBF values for single state vector.

        Args:
            x: Single state vector (n,)

        Returns:
            Array of HOCBF values (num_barriers,)
        """
        if not self._hocbf_funcs:
            raise ValueError("No barriers added. Use add_barriers() first.")

        # Compute all HOCBF values for single state
        return jnp.array([hocbf_func(x) for hocbf_func in self._hocbf_funcs])

    def hocbf(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the highest-order barrier function hocbf(x) of self._hocbf_funcs.

        This method returns a horizontally stacked array of the value of barriers at x.

        Args:
            x: State vector (n,) or batch (batch, n)

        Returns:
            HOCBF values with shape (batch, len(self._hocbf_funcs), 1)
        """
        if not self._hocbf_funcs:
            raise ValueError("No barriers added. Use add_barriers() first.")

        # Stack and transpose
        stacked = jnp.stack([apply_and_batchize(hocbf_func, x) for hocbf_func in self._hocbf_funcs])
        return jnp.transpose(stacked, (1, 0, 2))

    def _get_hocbf_and_lie_derivs_single(self, x: jnp.ndarray) -> tuple:
        """
        Compute HOCBF and Lie derivatives.

        Args:
            x: Single state vector (n,)

        Returns:
            Tuple of (hocbf_values, Lf_hocbf, Lg_hocbf) for all barriers
        """
        if not self._hocbf_funcs:
            raise ValueError("No barriers added. Use add_barriers() first.")
        if self._dynamics is None:
            raise ValueError("Dynamics not assigned. Use assign_dynamics() first.")

        f_val = self._dynamics.f(x)
        g_val = self._dynamics.g(x)

        hocbf_values = []
        lf_values = []
        lg_values = []

        # Process each barrier function
        for i, hocbf_func in enumerate(self._hocbf_funcs):
            if i in self._multidim_indices:
                # Multi-dimensional barrier: use jacrev with has_aux
                jac_hocbf, barrier_val = jax.jacrev(lambda x: (hocbf_func(x), hocbf_func(x)), has_aux=True)(x)

                lf_vals = jnp.einsum('ij,j->i', jac_hocbf, f_val)
                lg_vals = jnp.einsum('ij,jk->ik', jac_hocbf, g_val)

                hocbf_values.extend(barrier_val)
                lf_values.extend(lf_vals)
                lg_values.extend(lg_vals)
            else:
                # Scalar barrier: use efficient value_and_grad
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

    def get_hocbf_and_lie_derivs(self, x: jnp.ndarray) -> tuple:
        """
        Compute HOCBF and Lie derivatives with batching support.

        Args:
            x: State vector (n,) or batch (batch, n)

        Returns:
            Tuple of (hocbf_values, Lf_hocbf, Lg_hocbf) with proper batch dimensions
        """
        # Use apply_and_batchize_tuple for consistent batching
        return apply_and_batchize_tuple(self._get_hocbf_and_lie_derivs_single, x)

    def Lf_hocbf(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Compute Lie derivative of highest-order barrier function w.r.t. f.

        Args:
            x: State vector (n,) or batch (batch, n)

        Returns:
            Lie derivatives with shape (batch, len(self._hocbf_funcs), f dimension)
        """
        if not self._hocbf_funcs:
            raise ValueError("No barriers added. Use add_barriers() first.")
        if self._dynamics is None:
            raise ValueError("Dynamics not assigned. Use assign_dynamics() first.")

        # Compute Lie derivatives and stack
        lie_derivs = jnp.stack([lie_deriv(x, hocbf_func, self._dynamics.f) for hocbf_func in self._hocbf_funcs])
        return jnp.transpose(lie_derivs, (1, 0, 2))

    def Lg_hocbf(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Compute Lie derivative of highest-order barrier function w.r.t. g.

        Args:
            x: State vector (n,) or batch (batch, n)

        Returns:
            Lie derivatives with shape (batch, len(self._hocbf_funcs), g.shape)
        """
        if not self._hocbf_funcs:
            raise ValueError("No barriers added. Use add_barriers() first.")
        if self._dynamics is None:
            raise ValueError("Dynamics not assigned. Use assign_dynamics() first.")

        # Compute Lie derivatives and stack
        lie_derivs = jnp.stack([lie_deriv(x, hocbf_func, self._dynamics.g) for hocbf_func in self._hocbf_funcs])
        return jnp.transpose(lie_derivs, (1, 0, 2))

    def min_barrier(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Calculate the minimum value among all the barrier values computed at point x.

        Args:
            x: State vector (n,) or batch (batch, n)

        Returns:
            Minimum barrier value with shape (batch, 1)
        """
        barrier_vals = self.barrier(x)  # (batch, num_barriers, 1)
        return jnp.min(barrier_vals, axis=-2)  # Min across barriers dimension


    @property
    def barriers_flatten(self) -> tuple:
        """
        Get flattened list of all barrier functions.

        Returns:
            Tuple of all barrier functions from all barrier series
        """
        # Flatten: [b for barrier in self._barriers for b in barrier]
        flat = []
        for barrier_series in self._barriers:
            for b in barrier_series:
                flat.append(b)
        return tuple(flat)