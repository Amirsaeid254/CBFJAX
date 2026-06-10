"""
CompositionBarrier classes for barrier function composition.

This module implements barrier function composition using union and intersection
rules. Supports both smooth (soft) and non-smooth (hard) composition methods
for combining multiple barrier constraints.
"""

import jax.numpy as jnp
import equinox as eqx
from typing import List, Callable, Optional, Dict, Any, Tuple
from abc import abstractmethod

from .barrier import Barrier
from cbfjax.utils.utils import softmin, softmax
from cbfjax.dynamics.base_dynamic import DummyDynamics


class CompositionBarrier(Barrier):
    """
    Abstract base class for composing multiple barrier functions.

    This class enables the combination of multiple barrier constraints using
    composition rules such as union (maximum) or intersection (minimum) operations.
    The composition creates a single barrier function that represents the combined
    constraint from all individual barriers.

    Attributes:
        _barrier_list: Tuple of individual Barrier objects being composed
        _composition_rule: String identifier for the composition rule ('union', 'intersection')
        _barriers_raw: Tuple of raw barrier objects for reconstruction
        _composed_barrier_func: Composed barrier function for evaluation
    """

    # Additional fields for composition
    _barrier_list: tuple
    _composition_rule: str = eqx.field(static=True)
    _barriers_raw: tuple
    _composed_barrier_func: Callable = eqx.field(static=True)

    def __init__(self, barrier_func=None, dynamics=None, rel_deg=1, alphas=None,
                 barriers=None, hocbf_func=None, cfg=None,
                 barrier_list=None, composition_rule="", barriers_raw=None,
                 composed_barrier_func=None, rule=None):
        """
        Initialize CompositionBarrier with all parameters.

        Complete construction from Barrier objects:

            barrier = SoftCompositionBarrier(barriers=[b1, b2], rule='intersection', cfg=cfg)

        If dynamics is None (or dummy) it is inferred from the first barrier.

        Args:
            barrier_func: Composed barrier function
            dynamics: System dynamics object
            rel_deg: Relative degree for higher-order barriers
            alphas: Tuple of class-K functions
            barriers: List of Barrier objects (when rule is given), or tuple of
                barrier function series (internal use)
            hocbf_func: Highest-order composed barrier function
            cfg: Configuration dictionary
            barrier_list: Tuple of individual Barrier objects
            composition_rule: Composition rule identifier
            barriers_raw: Tuple of raw barrier objects
            composed_barrier_func: Function for computing individual barrier values
            rule: Composition rule ('intersection', 'union', 'i', 'u'); triggers
                complete construction from the barriers list
        """
        if rule is not None:
            valid_rules = ['intersection', 'union', 'i', 'u']
            if rule not in valid_rules:
                raise ValueError(f"Rule must be one of {valid_rules}, got '{rule}'")
            barrier_objs = list(barriers or [])
            assert barrier_objs and all(isinstance(b, Barrier) for b in barrier_objs), \
                "barriers must be a non-empty list of Barrier objects when rule is given"
            if dynamics is None or isinstance(dynamics, DummyDynamics):
                dynamics = barrier_objs[0].dynamics
            composed_barrier_func = self._create_barrier_composition_func(barrier_objs)
            hocbf_func = self._create_hocbf_composition_func(barrier_objs, rule)
            barriers = self._build_composed_barrier_series(barrier_objs, hocbf_func)
            barrier_func = composed_barrier_func
            rel_deg = 1
            alphas = ()
            barrier_list = tuple(barrier_objs)
            composition_rule = rule
            barriers_raw = tuple(barrier_objs)

        super().__init__(barrier_func, dynamics, rel_deg, alphas, barriers, hocbf_func, cfg)
        self._barrier_list = tuple(barrier_list or [])
        self._composition_rule = composition_rule
        self._barriers_raw = tuple(barriers_raw or [])
        self._composed_barrier_func = composed_barrier_func or self._create_dummy_barrier()

    @staticmethod
    def _create_dummy_barrier():
        """Create dummy barrier function that returns empty array."""
        def dummy_barrier(x):
            return jnp.array([])
        return dummy_barrier

    def _ctor_defaults(self) -> dict:
        """Constructor kwargs capturing current field values (per-class)."""
        return {
            'barrier_func': self._barrier_func,
            'dynamics': self._dynamics,
            'rel_deg': self._rel_deg,
            'alphas': self._alphas,
            'barriers': self._barriers,
            'hocbf_func': self._hocbf_func,
            'cfg': self.cfg,
            'barrier_list': self._barrier_list,
            'composition_rule': self._composition_rule,
            'barriers_raw': self._barriers_raw,
            'composed_barrier_func': self._composed_barrier_func
        }

    def _create_barrier_composition_func(self, barriers: List[Barrier]) -> Callable:
        """
        Create function that computes all individual barrier values.

        Args:
            barriers: List of barrier objects to compose

        Returns:
            Function that returns array of all barrier values
        """
        def barrier_composition_func(x):
            barrier_values = [barrier.barrier(x) for barrier in barriers]
            return jnp.array(barrier_values)
        return barrier_composition_func

    def _create_hocbf_composition_func(self, barriers: List[Barrier], rule: str) -> Callable:
        """
        Create function that computes composed HOCBF value.

        Args:
            barriers: List of barrier objects to compose
            rule: Composition rule identifier

        Returns:
            Function that returns composed HOCBF value
        """
        def hocbf_composition_func(x):
            hocbf_values = jnp.array([barrier.hocbf(x) for barrier in barriers])

            # Apply composition rule
            if rule in ['union', 'u']:
                rule_func = self._get_union_rule()
            elif rule in ['intersection', 'i']:
                rule_func = self._get_intersection_rule()
            else:
                raise ValueError(f"Invalid composition rule: {rule}")

            return rule_func(hocbf_values)
        return hocbf_composition_func

    def _build_composed_barrier_series(self, barriers: List[Barrier],
                                     hocbf_func: Callable) -> tuple:
        """
        Build the composed barrier function series.

        Args:
            barriers: List of individual barriers
            hocbf_func: Composed HOCBF function

        Returns:
            Tuple containing all barrier series plus composed HOCBF
        """
        barriers_series = [barrier.barriers for barrier in barriers]
        barriers_series.append((hocbf_func,))

        # Convert to tuple of tuples for hashability
        return tuple(
            tuple(series) if isinstance(series, list) else series
            for series in barriers_series
        )

    def barrier(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Compute composed barrier values at a single state.

        Args:
            x: Single state vector (n,)

        Returns:
            Array of all individual barrier values with shape (num_barriers,).
            Batch with jax.vmap(self.barrier).

        Raises:
            ValueError: If barriers not assigned
        """
        if not self._barriers_raw:
            raise ValueError("Barriers not assigned. Construct with barriers and rule.")
        return self._composed_barrier_func(x)

    def compose(self, rule_key: str) -> Callable:
        """
        Get composition rule function by key.

        Args:
            rule_key: Composition rule key ('union', 'u', 'intersection', 'i')

        Returns:
            Composition function implementing the specified rule

        Raises:
            ValueError: If rule key is invalid
        """
        if rule_key in ['union', 'u']:
            return self._get_union_rule()
        elif rule_key in ['intersection', 'i']:
            return self._get_intersection_rule()
        else:
            raise ValueError(f"Invalid composition rule key: '{rule_key}'")

    @abstractmethod
    def _get_union_rule(self) -> Callable:
        """
        Get the union composition rule function.

        Returns:
            Function that implements union (maximum-like) composition
        """
        raise NotImplementedError

    @abstractmethod
    def _get_intersection_rule(self) -> Callable:
        """
        Get the intersection composition rule function.

        Returns:
            Function that implements intersection (minimum-like) composition
        """
        raise NotImplementedError

    @property
    def barriers_flatten(self) -> tuple:
        """
        Get flattened tuple of all barrier functions.

        Returns:
            Tuple containing all barrier functions from all series
        """
        if not self._barriers:
            return ()
        return tuple(
            barrier_func
            for barrier_group in self._barriers
            for barrier_func in barrier_group
        )

    @property
    def num_individual_barriers(self) -> int:
        """Number of individual barriers in the composition."""
        return len(self._barrier_list)


class SoftCompositionBarrier(CompositionBarrier):
    """
    Soft composition barrier using smooth approximations.

    Implements barrier composition using smooth approximations of max/min operations
    through softmax and softmin functions. This approach provides differentiable
    composition suitable for gradient-based optimization.
    """

    def _get_union_rule(self) -> Callable:
        """
        Get soft union rule using smooth maximum approximation.

        Returns:
            Function implementing soft union via softmax
        """
        rho = self.cfg.get('softmax_rho', 1.0)

        def soft_union(barrier_values):
            return softmax(barrier_values, rho=rho, conservative=True, dim=0)
        return soft_union

    def _get_intersection_rule(self) -> Callable:
        """
        Get soft intersection rule using smooth minimum approximation.

        Returns:
            Function implementing soft intersection via softmin
        """
        rho = self.cfg.get('softmin_rho', 1.0)

        def soft_intersection(barrier_values):
            return softmin(barrier_values, rho=rho, conservative=False, dim=0)
        return soft_intersection


class NonSmoothCompositionBarrier(CompositionBarrier):
    """
    Non-smooth composition barrier using exact operations.

    Implements barrier composition using exact maximum and minimum operations.
    This approach provides precise composition but may not be differentiable
    at points where multiple barriers have equal values.
    """

    def _get_union_rule(self) -> Callable:
        """
        Get hard union rule using exact maximum.

        Returns:
            Function implementing exact union via maximum operation
        """
        def hard_union(barrier_values):
            return jnp.max(barrier_values)
        return hard_union

    def _get_intersection_rule(self) -> Callable:
        """
        Get hard intersection rule using exact minimum.

        Returns:
            Function implementing exact intersection via minimum operation
        """
        def hard_intersection(barrier_values):
            return jnp.min(barrier_values)
        return hard_intersection