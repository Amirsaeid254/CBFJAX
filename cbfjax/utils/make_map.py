"""
Map class for creating barriers from geometric primitives.

This module provides functionality to create barrier functions from geometric
specifications such as circles, rectangles, and ellipses. It supports both
obstacle avoidance and boundary constraints.
"""

import equinox as eqx
from typing import Dict, List, Tuple, Any, Optional
from abc import ABC, abstractmethod
from immutabledict import immutabledict

from ..barriers.barrier import Barrier
from ..barriers.composite_barrier import SoftCompositionBarrier, NonSmoothCompositionBarrier
from .utils import (
    make_circle_barrier_functional,
    make_norm_rectangular_barrier_functional,
    make_affine_rectangular_barrier_functional,
    make_norm_rectangular_boundary_functional,
    make_affine_rectangular_boundary_functional,
    make_box_barrier_functionals,
    make_linear_alpha_function_form_list_of_coef,
    make_ellipse_barrier_functional
)


class GeometryProvider(ABC):
    """Abstract base class for providing geometry data to the map."""

    @abstractmethod
    def get_geometries(self):
        """
        Return list of (geom_type, geom_info) tuples.

        geom_type can be: 'cylinder', 'box', 'norm_box', 'boundary', 'norm_boundary'
        geom_info is a dict with parameters specific to each geometry type
        """
        pass

    def get_velocity_constraints(self):
        """
        Return velocity constraints as (idx, bounds) tuple or None.

        idx: indices of velocity components to constrain
        bounds: list of (min, max) tuples for each component
        """
        return None


class StaticGeometryProvider(GeometryProvider):
    """Provides pre-defined static geometry from configuration data."""

    def __init__(self, geoms_config):
        """
        Initialize with geometry configuration.

        Args:
            geoms_config: List of (geom_type, geom_info) tuples or dict with 'geoms' key
        """
        if isinstance(geoms_config, dict):
            # Convert to tuples for hashability
            self.geoms = tuple(
                tuple(item) if isinstance(item, list) else item
                for item in geoms_config.get('geoms', [])
            )
            self.velocity = geoms_config.get('velocity', None)
        else:
            # Convert to tuples for hashability
            self.geoms = tuple(
                tuple(item) if isinstance(item, list) else item
                for item in geoms_config
            )
            self.velocity = None

    def get_geometries(self):
        """
        Convert stored tuples back to geometry specifications.

        Returns:
            List of (geom_type, geom_info) tuples
        """
        converted_geoms = []
        for geom_type, geom_info in self.geoms:
            converted_geoms.append((geom_type, geom_info))
        return converted_geoms

    def get_velocity_constraints(self):
        """Return velocity constraints if any."""
        return self.velocity


class ImageGeometryProvider(GeometryProvider):
    """
    Provides geometry by processing image files.

    Note: This implementation is a placeholder. Full implementation requires
    image processing and support vector machine functionality.
    """

    def __init__(self, image_path, synthesis_cfg):
        """
        Initialize image-based geometry provider.

        Args:
            image_path: Path to image file
            synthesis_cfg: Configuration for image synthesis
        """
        self.image_path = image_path
        self.synthesis_cfg = synthesis_cfg if synthesis_cfg else {}
        self._barrier_func = None
        print("Warning: ImageGeometryProvider not fully implemented")
        print("Consider using geometric primitives instead")

    def get_geometries(self):
        """Return empty list as image processing is not implemented."""
        return []

    def get_barrier_function(self, dynamics):
        """Get barrier function created from image processing."""
        if self._barrier_func is None:
            raise RuntimeError("Image processing not implemented")
        return lambda x: self._barrier_func(dynamics.get_pos(x))


class Map(eqx.Module):
    """
    Map class for creating barrier functions from geometric specifications.

    This class takes geometric specifications (circles, rectangles, boundaries)
    and creates corresponding barrier functions. It supports both position-based
    obstacles and velocity constraints.

    Attributes:
        dynamics: System dynamics object
        pos_barriers: Tuple of position-based barrier objects
        vel_barriers: Tuple of velocity constraint barrier objects
        barrier: Composed barrier using soft composition rules
        map_barrier: Composed barrier using hard composition rules
        cfg: Configuration dictionary
        geometry_provider: Provider for geometry specifications
    """

    # Core configuration
    dynamics: Any

    # Computed barriers
    pos_barriers: tuple = eqx.field(static=True)
    vel_barriers: tuple = eqx.field(static=True)
    barrier: Optional[SoftCompositionBarrier]
    map_barrier: Optional[NonSmoothCompositionBarrier]

    # Static configuration
    cfg: immutabledict = eqx.field(static=True)
    geometry_provider: GeometryProvider = eqx.field(static=True)

    def __init__(self, dynamics, cfg, geometry_provider=None, barriers_info=None,
                 image_path=None, synthesis_cfg=None,
                 pos_barriers=None, vel_barriers=None, barrier=None, map_barrier=None):
        """
        Initialize Map with geometry provider and optional pre-computed barriers.

        Args:
            dynamics: System dynamics object
            cfg: Configuration dictionary (can be Box object)
            geometry_provider: Custom GeometryProvider instance
            barriers_info: Dictionary with geometry information
            image_path: Path to image file for image-based barriers
            synthesis_cfg: Configuration for image synthesis
            pos_barriers: Pre-computed position barriers
            vel_barriers: Pre-computed velocity barriers
            barrier: Pre-computed soft composition barrier
            map_barrier: Pre-computed hard composition barrier
        """
        self.dynamics = dynamics
        self.cfg = immutabledict(cfg)

        # Initialize geometry provider
        self.geometry_provider = self._create_geometry_provider(
            geometry_provider, barriers_info, image_path, synthesis_cfg
        )

        # Initialize barriers as tuples for hashability
        self.pos_barriers = tuple(pos_barriers or [])
        self.vel_barriers = tuple(vel_barriers or [])
        self.barrier = barrier
        self.map_barrier = map_barrier

    def _create_geometry_provider(self, geometry_provider, barriers_info,
                                 image_path, synthesis_cfg) -> GeometryProvider:
        """
        Create appropriate geometry provider based on inputs.

        Args:
            geometry_provider: Custom provider instance
            barriers_info: Barrier configuration dictionary
            image_path: Path to image file
            synthesis_cfg: Image synthesis configuration

        Returns:
            Configured GeometryProvider instance

        Raises:
            ValueError: If no valid geometry source provided
        """
        if geometry_provider is not None:
            return geometry_provider
        elif barriers_info is not None:
            if isinstance(barriers_info, dict) and 'image' in barriers_info:
                return ImageGeometryProvider(
                    barriers_info['image'],
                    synthesis_cfg or self.cfg.get('synthesis_cfg')
                )
            else:
                return StaticGeometryProvider(barriers_info)
        elif image_path is not None:
            return ImageGeometryProvider(image_path, synthesis_cfg)
        else:
            raise ValueError(
                "Must provide one of: geometry_provider, barriers_info, or image_path"
            )

    def create_barriers(self) -> 'Map':
        """
        Create position and velocity barriers from geometry provider.

        Returns:
            New Map instance with computed barriers
        """
        # Create position barriers from geometry
        if isinstance(self.geometry_provider, ImageGeometryProvider):
            pos_barriers = self._create_image_barriers()
        else:
            pos_barriers = self._create_geometric_barriers()

        # Create velocity barriers if constraints exist
        velocity_constraints = self.geometry_provider.get_velocity_constraints()
        vel_barriers = (
            self._create_velocity_barriers(velocity_constraints)
            if velocity_constraints else []
        )

        # Create composed barriers
        all_barriers = pos_barriers + vel_barriers
        barrier = self._create_soft_composition_barrier(all_barriers)
        map_barrier = self._create_hard_composition_barrier(pos_barriers)

        # Return new instance with all barriers
        return Map(
            dynamics=self.dynamics,
            cfg=self.cfg,
            geometry_provider=self.geometry_provider,
            pos_barriers=pos_barriers,
            vel_barriers=vel_barriers,
            barrier=barrier,
            map_barrier=map_barrier
        )

    def _create_geometric_barriers(self) -> List[Barrier]:
        """
        Create barriers from geometric primitives.

        Returns:
            List of Barrier objects created from geometry specifications
        """
        barriers = []
        geoms = self.geometry_provider.get_geometries()

        for geom_type, geom_info in geoms:
            barrier_func_factory, alpha_key = self._get_barrier_config(geom_type)
            alphas = make_linear_alpha_function_form_list_of_coef(
                self.cfg.get(alpha_key, (1.0,))
            )

            barrier = (
                Barrier.create_empty(cfg=self.cfg)
                .assign(
                    barrier_func=barrier_func_factory(**geom_info),
                    rel_deg=self.cfg.get('pos_barrier_rel_deg', 1),
                    alphas=alphas
                )
                .assign_dynamics(self.dynamics)
            )

            barriers.append(barrier)

        return barriers

    def _create_image_barriers(self) -> List[Barrier]:
        """
        Create barriers from image-based geometry provider.

        Raises:
            NotImplementedError: Image processing not implemented
        """
        raise NotImplementedError(
            "Image-based barriers not fully implemented. "
            "Consider using geometric primitives instead."
        )

    def _create_velocity_barriers(self, velocity_constraints) -> List[Barrier]:
        """
        Create velocity constraint barriers.

        Args:
            velocity_constraints: Tuple of (idx, bounds) for velocity limits

        Returns:
            List of Barrier objects for velocity constraints
        """
        idx, bounds = velocity_constraints
        alphas = make_linear_alpha_function_form_list_of_coef(
            self.cfg.get('velocity_alpha', (1.0,))
        )

        vel_barrier_funcs = make_box_barrier_functionals(bounds=bounds, idx=idx)

        barriers = [
            (
                Barrier.create_empty(cfg=self.cfg)
                .assign(
                    barrier_func=vel_barrier,
                    rel_deg=self.cfg.get('vel_barrier_rel_deg', 1),
                    alphas=alphas
                )
                .assign_dynamics(self.dynamics)
            )
            for vel_barrier in vel_barrier_funcs
        ]

        return barriers

    def _create_soft_composition_barrier(self, barriers: List[Barrier]) -> SoftCompositionBarrier:
        """Create soft composition barrier from individual barriers."""
        return (
            SoftCompositionBarrier.create_empty(cfg=self.cfg)
            .assign_barriers_and_rule(
                barriers=barriers,
                rule='intersection',
                infer_dynamics=True
            )
        )

    def _create_hard_composition_barrier(self, barriers: List[Barrier]) -> NonSmoothCompositionBarrier:
        """Create hard composition barrier from position barriers only."""
        return (
            NonSmoothCompositionBarrier.create_empty(cfg=self.cfg)
            .assign_barriers_and_rule(
                barriers=barriers,
                rule='intersection',
                infer_dynamics=True
            )
        )

    def _get_barrier_config(self, geom_type: str) -> Tuple[callable, str]:
        """
        Get barrier function factory and alpha configuration key for geometry type.

        Args:
            geom_type: Type of geometry ('cylinder', 'box', etc.)

        Returns:
            Tuple of (barrier_function_factory, alpha_config_key)

        Raises:
            NotImplementedError: If geometry type not supported
        """
        geometry_mapping = {
            'cylinder': (make_circle_barrier_functional, 'obstacle_alpha'),
            'box': (make_affine_rectangular_barrier_functional, 'obstacle_alpha'),
            'norm_box': (make_norm_rectangular_barrier_functional, 'obstacle_alpha'),
            'boundary': (make_affine_rectangular_boundary_functional, 'boundary_alpha'),
            'norm_boundary': (make_norm_rectangular_boundary_functional, 'boundary_alpha'),
            'ellipse': (make_ellipse_barrier_functional, 'obstacle_alpha'),
        }

        if geom_type not in geometry_mapping:
            raise NotImplementedError(f"Geometry type '{geom_type}' not supported")

        return geometry_mapping[geom_type]

    def get_barriers(self) -> Tuple[tuple, tuple]:
        """
        Get position and velocity barriers separately.

        Returns:
            Tuple of (position_barriers, velocity_barriers)
        """
        return self.pos_barriers, self.vel_barriers

    @property
    def num_position_barriers(self) -> int:
        """Number of position-based barriers."""
        return len(self.pos_barriers)

    @property
    def num_velocity_barriers(self) -> int:
        """Number of velocity constraint barriers."""
        return len(self.vel_barriers)

    @property
    def total_barriers(self) -> int:
        """Total number of individual barriers."""
        return self.num_position_barriers + self.num_velocity_barriers


# Convenience functions for common use cases
def make_map_from_geoms(geoms: List[Tuple[str, Dict]], dynamics, cfg,
                        velocity_constraints: Optional[Tuple[int, Tuple[float, float]]] = None) -> Map:
    """
    Create a map from a list of geometry specifications.

    Args:
        geoms: List of (geom_type, geom_info) tuples
        dynamics: System dynamics object
        cfg: Barrier configuration dictionary
        velocity_constraints: Optional (idx, bounds) tuple for velocity limits

    Returns:
        Map instance with barriers created from specifications
    """
    barriers_info: Dict[str, Any] = {'geoms': geoms}
    if velocity_constraints:
        barriers_info['velocity'] = velocity_constraints

    return Map(dynamics=dynamics, cfg=cfg, barriers_info=barriers_info).create_barriers()


def make_map_from_image(image_path: str, dynamics, cfg, synthesis_cfg=None) -> Map:
    """
    Create a map from an image file.

    Note: Not fully implemented in current version.

    Args:
        image_path: Path to image file
        dynamics: System dynamics object
        cfg: Barrier configuration dictionary
        synthesis_cfg: Optional synthesis configuration for image processing

    Returns:
        Map instance configured for image-based barriers
    """
    return Map(
        dynamics=dynamics,
        cfg=cfg,
        image_path=image_path,
        synthesis_cfg=synthesis_cfg or cfg.get('synthesis_cfg')
    ).create_barriers()


def create_simple_obstacle_map(dynamics, obstacles: List[Dict], cfg=None) -> Map:
    """
    Helper to create a map with simple circular obstacles.

    Args:
        dynamics: System dynamics object
        obstacles: List of obstacle dictionaries with 'center' and 'radius' keys
        cfg: Optional configuration dictionary

    Returns:
        Map instance with circular obstacle barriers
    """
    if cfg is None:
        cfg = immutabledict({
            'pos_barrier_rel_deg': 1,
            'obstacle_alpha': (1.0,),
        })

    geoms = [('cylinder', obs) for obs in obstacles]
    return make_map_from_geoms(geoms, dynamics, cfg)


def create_rectangular_boundary_map(dynamics, bounds: Dict, cfg=None) -> Map:
    """
    Helper to create a map with rectangular boundaries.

    Args:
        dynamics: System dynamics object
        bounds: Boundary specification with 'center', 'size', and optional 'rotation'
        cfg: Optional configuration dictionary

    Returns:
        Map instance with rectangular boundary barriers
    """
    if cfg is None:
        cfg = immutabledict({
            'pos_barrier_rel_deg': 1,
            'boundary_alpha': (1.0,),
        })

    geoms = [('boundary', bounds)]
    return make_map_from_geoms(geoms, dynamics, cfg)


def create_map_from_config(barriers_info, dynamics, cfg) -> Map:
    """
    Create map from configuration dictionary.

    This function provides compatibility with existing configuration formats.

    Args:
        barriers_info: Barrier configuration dictionary or geometry list
        dynamics: System dynamics object
        cfg: Barrier configuration dictionary

    Returns:
        Map instance with barriers created from configuration
    """
    return Map(dynamics=dynamics, cfg=cfg, barriers_info=barriers_info).create_barriers()