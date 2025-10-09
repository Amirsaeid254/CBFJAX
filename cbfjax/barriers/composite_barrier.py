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
from cbfjax.utils.utils import apply_and_batchize, softmin, softmax


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
        _barrier_funcs: Composed barrier function for evaluation
    """

    # Additional fields for composition
    _barrier_list: tuple = eqx.field(static=True)
    _composition_rule: str = eqx.field(static=True)
    _barriers_raw: tuple = eqx.field(static=True)
    _barrier_funcs: Callable = eqx.field(static=True)

    def __init__(self, barrier_func=None, dynamics=None, rel_deg=1, alphas=None,
                 barriers=None, hocbf_func=None, cfg=None,
                 barrier_list=None, composition_rule="", barriers_raw=None, barrier_funcs=None):
        """
        Initialize CompositionBarrier with all parameters.

        Args:
            barrier_func: Composed barrier function
            dynamics: System dynamics object
            rel_deg: Relative degree for higher-order barriers
            alphas: Tuple of class-K functions
            barriers: Tuple of barrier function series
            hocbf_func: Highest-order composed barrier function
            cfg: Configuration dictionary
            barrier_list: Tuple of individual Barrier objects
            composition_rule: Composition rule identifier
            barriers_raw: Tuple of raw barrier objects
            barrier_funcs: Function for computing individual barrier values
        """
        super().__init__(barrier_func, dynamics, rel_deg, alphas, barriers, hocbf_func, cfg)
        self._barrier_list = tuple(barrier_list or [])
        self._composition_rule = composition_rule
        self._barriers_raw = tuple(barriers_raw or [])
        self._barrier_funcs = barrier_funcs or self._create_empty_barrier_func()

    @staticmethod
    def _create_empty_barrier_func():
        """Create empty barrier function that returns empty array."""
        def empty_barrier_func(x):
            return jnp.array([])
        return empty_barrier_func

    @classmethod
    def create_empty(cls, cfg=None):
        """
        Create an empty composition barrier instance.

        Args:
            cfg: Optional configuration dictionary

        Returns:
            Empty CompositionBarrier instance ready for barrier assignment
        """
        return cls(cfg=cfg)

    def _create_updated_instance(self, **kwargs):
        """
        Create new instance with updated fields.

        This helper method extends Barrier by adding composition-specific fields.

        Args:
            **kwargs: Fields to update

        Returns:
            New instance of the same class with updated fields
        """
        defaults = {
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
            'barrier_funcs': self._barrier_funcs
        }
        defaults.update(kwargs)
        return self.__class__(**defaults)

    def assign(self, barrier_func: Callable, rel_deg: int = 1,
               alphas: Optional[List[Callable]] = None) -> 'CompositionBarrier':
        """
        Override parent assign method to prevent direct barrier assignment.

        CompositionBarrier requires barriers to be assigned through the
        assign_barriers_and_rule method for proper composition setup.

        Raises:
            ValueError: Always raised to direct users to proper assignment method
        """
        raise ValueError(
            'CompositionBarrier assignment must be done through assign_barriers_and_rule method'
        )

    def assign_dynamics(self, dynamics) -> 'CompositionBarrier':
        """
        Assign dynamics and update composition if barriers already configured.

        Args:
            dynamics: System dynamics object

        Returns:
            New CompositionBarrier instance with updated dynamics
        """
        # If composition already exists, recreate with new dynamics
        if self._composition_rule and self._barriers_raw:
            return self.assign_barriers_and_rule(
                barriers=list(self._barriers_raw),
                rule=self._composition_rule,
                infer_dynamics=False,
                dynamics_override=dynamics
            )

        # Otherwise update dynamics only
        return self._create_updated_instance(dynamics=dynamics)

    def assign_barriers_and_rule(self, barriers: List[Barrier], rule: str,
                                infer_dynamics: bool = False,
                                dynamics_override=None) -> 'CompositionBarrier':
        """
        Assign multiple barriers and composition rule to create composed barrier.

        Args:
            barriers: List of Barrier objects to compose
            rule: Composition rule ('intersection', 'union', 'i', 'u')
            infer_dynamics: Whether to infer dynamics from first barrier
            dynamics_override: Optional dynamics object to override inference

        Returns:
            New CompositionBarrier instance with barriers composed

        Raises:
            ValueError: If rule is invalid or dynamics cannot be determined
        """
        # Validate composition rule
        valid_rules = ['intersection', 'union', 'i', 'u']
        if rule not in valid_rules:
            raise ValueError(f"Rule must be one of {valid_rules}, got '{rule}'")

        # Determine dynamics source
        dynamics = self._resolve_dynamics(barriers, infer_dynamics, dynamics_override)

        # Create composition functions
        barrier_funcs = self._create_barrier_composition_func(barriers)
        hocbf_func = self._create_hocbf_composition_func(barriers, rule)

        # Build composed barrier series
        barriers_series = self._build_composed_barrier_series(barriers, hocbf_func)

        # Create new composed instance
        return self._create_updated_instance(
            barrier_func=barrier_funcs,
            dynamics=dynamics,
            rel_deg=1,
            alphas=(),
            barriers=barriers_series,
            hocbf_func=hocbf_func,
            barrier_list=tuple(barriers),
            composition_rule=rule,
            barriers_raw=tuple(barriers),
            barrier_funcs=barrier_funcs
        )

    def _resolve_dynamics(self, barriers: List[Barrier], infer_dynamics: bool,
                         dynamics_override) -> Any:
        """
        Resolve which dynamics object to use for the composition.

        Args:
            barriers: List of barrier objects
            infer_dynamics: Whether to infer from first barrier
            dynamics_override: Optional explicit dynamics override

        Returns:
            Resolved dynamics object

        Raises:
            ValueError: If dynamics cannot be determined
        """
        if dynamics_override is not None:
            return dynamics_override
        elif infer_dynamics:
            return barriers[0].dynamics
        elif hasattr(self._dynamics, 'f'):
            return self._dynamics
        else:
            raise ValueError(
                'Dynamics must be assigned. Use infer_dynamics=True or provide dynamics_override'
            )

    def _create_barrier_composition_func(self, barriers: List[Barrier]) -> Callable:
        """
        Create function that computes all individual barrier values.

        Args:
            barriers: List of barrier objects to compose

        Returns:
            Function that returns array of all barrier values
        """
        def barrier_composition_func(x):
            barrier_values = [barrier._barrier_single(x) for barrier in barriers]
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
            hocbf_values = jnp.array([barrier._hocbf_single(x) for barrier in barriers])

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
        Compute composed barrier values at given state(s).

        Args:
            x: State vector (n,) or batch (batch, n)

        Returns:
            Array of all individual barrier values with shape (batch, num_barriers)

        Raises:
            ValueError: If barriers not assigned
        """
        if not self._barriers_raw:
            raise ValueError("Barriers not assigned. Use assign_barriers_and_rule first.")
        return apply_and_batchize(self._barrier_funcs, x)

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