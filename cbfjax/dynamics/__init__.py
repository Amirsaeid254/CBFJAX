"""
Dynamics module for CBF-JAX

Provides system dynamics implementations using JAX and Equinox.
"""

from .base_dynamic import AffineInControlDynamics, CustomDynamics, create_augmented_dynamics
from .unicycle import UnicycleDynamics
from .double_integrator import DoubleIntegratorDynamics
from .single_integrator import SingleIntegratorDynamics
from .bicycle import BicycleDynamics
from .inverted_pendulum import InvertedPendulumDynamics
from .unicycle_reduced_order import UnicycleReducedOrderDynamics
from .unicycle_5th_order import Unicycle5thOrderDynamics

__all__ = [
    "AffineInControlDynamics",
    "CustomDynamics",
    "create_augmented_dynamics",
    "UnicycleDynamics",
    "DoubleIntegratorDynamics",
    "SingleIntegratorDynamics",
    "BicycleDynamics",
    "InvertedPendulumDynamics",
    "UnicycleReducedOrderDynamics",
    "Unicycle5thOrderDynamics"
]