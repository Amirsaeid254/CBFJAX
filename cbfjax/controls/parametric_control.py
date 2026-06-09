"""
Parametric control utilities for FlowBarrier implementation.

This module provides parametric control classes using equinox for JIT compatibility,
following the CBFJAX architecture patterns.
"""

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
from typing import Callable, List
from immutabledict import immutabledict


class ParametricControl(eqx.Module):
    """
    Base class for parametric control functions u(τ; θ).

    All fields are static
    """

    # Configuration parameters
    horizon: float = eqx.field(static=True)
    control_dim: int = eqx.field(static=True)
    param_dim: int = eqx.field(static=True)
    dt: float = eqx.field(static=True)
    control_duration: float = eqx.field(static=True)

    # Control bounds as tuples
    _control_low: tuple = eqx.field(static=True)
    _control_high: tuple = eqx.field(static=True)
    _has_control_bounds: bool = eqx.field(static=True)

    # Barrier functions for constraints
    _action_barrier_funcs: tuple = eqx.field(static=True)

    def __init__(self, horizon: float, control_dim: int, param_dim: int, dt: float,
                 control_low: tuple = None, control_high: tuple = None,
                 action_barrier_funcs: tuple = None):
        """
        Initialize parametric control.

        Args:
            horizon: Time horizon T > 0
            control_dim: Dimension of control input (action_dim) > 0
            param_dim: Dimension of parameter vector θ > 0
            dt: Time step for control application
            control_low: Lower bounds tuple or None
            control_high: Upper bounds tuple or None
            action_barrier_funcs: Tuple of barrier functions or None
        """
        self.horizon = horizon
        self.control_dim = control_dim
        self.param_dim = param_dim
        self.dt = dt
        self.control_duration = horizon - dt

        # Set control bounds as tuples
        if control_low is not None and control_high is not None:
            self._control_low = tuple(control_low) if not isinstance(control_low, tuple) else control_low
            self._control_high = tuple(control_high) if not isinstance(control_high, tuple) else control_high
            self._has_control_bounds = True
        else:
            self._control_low = tuple([0.0] * control_dim)
            self._control_high = tuple([0.0] * control_dim)
            self._has_control_bounds = False

        self._action_barrier_funcs = tuple(action_barrier_funcs if action_barrier_funcs is not None else [])

    def __call__(self, tau: float, theta: jnp.ndarray) -> jnp.ndarray:
        """
        Evaluate parametric control at time tau using parameters theta.

        Args:
            tau: Time point (scalar) in [0, T]
            theta: Parameters [control_dim, param_dim]

        Returns:
            Control input [control_dim]
        """
        raise NotImplementedError

    def set_control_bounds(self, low: list, high: list) -> 'ParametricControl':
        """
        Return new instance with control bounds set.

        Args:
            low: Lower bounds for control inputs
            high: Upper bounds for control inputs

        Returns:
            New ParametricControl instance with bounds
        """
        raise NotImplementedError

    def get_action_barrier_functions(self) -> tuple:
        """
        Get barrier functions for control constraints.

        Returns:
            Tuple of barrier functions
        """
        return self._action_barrier_funcs


class ZOHParametricControl(ParametricControl):
    """Zero-Order Hold (piecewise constant) parametric control."""

    num_segments: int = eqx.field(static=True)
    dt_segment: float = eqx.field(static=True)

    def __init__(self, horizon: float, control_dim: int, num_segments: int, dt: float,
                 control_low: tuple = None, control_high: tuple = None):
        """
        Initialize ZOH parametric control.

        Args:
            horizon: Time horizon T
            control_dim: Dimension of control input
            num_segments: Number of constant control segments
            dt: Time step for control application
            control_low: Lower bounds tuple or None
            control_high: Upper bounds tuple or None
        """
        # Create action barrier functions if bounds provided
        action_barrier_funcs = None
        if control_low is not None and control_high is not None:
            action_barrier_funcs = self._create_action_barrier_functions(
                control_low, control_high, control_dim
            )

        super().__init__(
            horizon=horizon,
            control_dim=control_dim,
            param_dim=num_segments,
            dt=dt,
            control_low=control_low,
            control_high=control_high,
            action_barrier_funcs=action_barrier_funcs
        )

        self.num_segments = num_segments
        self.dt_segment = self.control_duration / num_segments

    def __call__(self, tau: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
        """
        Evaluate ZOH control at time tau.

        Args:
            tau: Time point (scalar)
            theta: Parameters [control_dim, num_segments]

        Returns:
            Control values [control_dim]
        """
        # Ensure tau is scalar
        tau_scalar = jnp.squeeze(tau) if jnp.ndim(tau) > 0 else tau

        # Clamp tau to valid range
        tau_scalar = jnp.clip(tau_scalar, 0, self.horizon)

        # Compute segment index
        segment_idx = jnp.floor(tau_scalar / self.dt_segment).astype(jnp.int32)
        segment_idx = jnp.clip(segment_idx, 0, self.num_segments - 1)

        # Return control value for this segment
        return theta[:, segment_idx]

    def _create_action_barrier_functions(self, control_low: tuple, control_high: tuple,
                                         control_dim: int) -> tuple:
        """
        Create action barrier functions for bounds.

        Args:
            control_low: Lower bounds tuple
            control_high: Upper bounds tuple
            control_dim: Control dimension

        Returns:
            Tuple of barrier functions
        """
        barrier_funcs = []

        for ctrl_idx in range(control_dim):
            lb = control_low[ctrl_idx]
            ub = control_high[ctrl_idx]

            # Create closures capturing the values
            def create_lower_barrier(idx: int, bound: float):
                def barrier_func(theta: jnp.ndarray) -> jnp.ndarray:
                    return theta[idx, :] - bound
                return barrier_func

            def create_upper_barrier(idx: int, bound: float):
                def barrier_func(theta: jnp.ndarray) -> jnp.ndarray:
                    return bound - theta[idx, :]
                return barrier_func

            barrier_funcs.append(create_lower_barrier(ctrl_idx, lb))
            barrier_funcs.append(create_upper_barrier(ctrl_idx, ub))

        return tuple(barrier_funcs)

    def set_control_bounds(self, low: tuple, high: tuple) -> 'ZOHParametricControl':
        """
        Return new instance with control bounds.

        Args:
            low: Lower bounds for control inputs (tuple)
            high: Upper bounds for control inputs (tuple)

        Returns:
            New ZOHParametricControl with bounds
        """
        assert len(low) == len(high), 'low and high should have the same length'
        assert len(low) == self.control_dim, 'bounds length should match control dimension'

        return ZOHParametricControl(
            horizon=self.horizon,
            control_dim=self.control_dim,
            num_segments=self.num_segments,
            dt=self.dt,
            control_low=low,
            control_high=high
        )


class FOHParametricControl(ParametricControl):
    """First-Order Hold (FOH) parametric control with linear interpolation between waypoints."""

    num_waypoints: int = eqx.field(static=True)
    waypoint_times: tuple = eqx.field(static=True)  # Stored as tuple for hashability

    def __init__(self, horizon: float, control_dim: int, num_waypoints: int, dt: float,
                 control_low: tuple = None, control_high: tuple = None):
        """
        Initialize FOH control.

        Args:
            horizon: Time horizon T
            control_dim: Dimension of control input
            num_waypoints: Number of waypoints (>= 2)
            dt: Time step
            control_low: Lower bounds tuple or None
            control_high: Upper bounds tuple or None
        """
        if num_waypoints < 2:
            raise ValueError(f"num_waypoints must be >= 2, got {num_waypoints}")

        # Create action barrier functions if bounds provided
        action_barrier_funcs = None
        if control_low is not None and control_high is not None:
            action_barrier_funcs = self._create_action_barrier_functions(
                control_low, control_high, control_dim
            )

        super().__init__(
            horizon=horizon,
            control_dim=control_dim,
            param_dim=num_waypoints,
            dt=dt,
            control_low=control_low,
            control_high=control_high,
            action_barrier_funcs=action_barrier_funcs
        )

        self.num_waypoints = num_waypoints
        control_duration = horizon - dt
        self.waypoint_times = tuple(float(t) for t in np.linspace(0, control_duration, num_waypoints))

    def __call__(self, tau: float, theta: jnp.ndarray) -> jnp.ndarray:
        """
        Evaluate FOH control with linear interpolation.

        Args:
            tau: Time point (scalar)
            theta: Parameters [control_dim, num_waypoints]

        Returns:
            Control values [control_dim]
        """
        # Ensure tau is scalar
        tau_scalar = jnp.squeeze(tau) if jnp.ndim(tau) > 0 else tau

        # Clamp tau to valid range
        tau_scalar = jnp.clip(tau_scalar, 0, self.horizon)

        # Convert tuple to array for searchsorted
        waypoint_times_arr = jnp.array(self.waypoint_times)

        # Find interpolation index
        idx = jnp.searchsorted(waypoint_times_arr[1:], tau_scalar, side='left')
        idx = jnp.clip(idx, 0, self.num_waypoints - 2)

        # Compute interpolation weight
        t0 = waypoint_times_arr[idx]
        t1 = waypoint_times_arr[idx + 1]
        dt = t1 - t0
        dt = jnp.where(dt > 0, dt, 1.0)  # Avoid division by zero
        weight = (tau_scalar - t0) / dt
        weight = jnp.clip(weight, 0, 1)

        # Linear interpolation
        val0 = theta[:, idx]
        val1 = theta[:, idx + 1]
        return val0 + weight * (val1 - val0)

    def _create_action_barrier_functions(self, control_low: tuple, control_high: tuple,
                                         control_dim: int) -> tuple:
        """
        Create action barrier functions for bounds.

        Args:
            control_low: Lower bounds tuple
            control_high: Upper bounds tuple
            control_dim: Control dimension

        Returns:
            Tuple of barrier functions
        """
        barrier_funcs = []

        for ctrl_idx in range(control_dim):
            lb = control_low[ctrl_idx]
            ub = control_high[ctrl_idx]

            def create_lower_barrier(idx: int, bound: float):
                def barrier(theta: jnp.ndarray) -> jnp.ndarray:
                    return theta[idx, :] - bound
                return barrier

            def create_upper_barrier(idx: int, bound: float):
                def barrier(theta: jnp.ndarray) -> jnp.ndarray:
                    return bound - theta[idx, :]
                return barrier

            barrier_funcs.append(create_lower_barrier(ctrl_idx, lb))
            barrier_funcs.append(create_upper_barrier(ctrl_idx, ub))

        return tuple(barrier_funcs)

    def set_control_bounds(self, low: tuple, high: tuple) -> 'FOHParametricControl':
        """
        Return new instance with control bounds.

        Args:
            low: Lower bounds for control inputs (tuple)
            high: Upper bounds for control inputs (tuple)

        Returns:
            New FOHParametricControl with bounds
        """
        assert len(low) == len(high), 'low and high should have the same length'
        assert len(low) == self.control_dim, 'bounds length should match control dimension'

        return FOHParametricControl(
            horizon=self.horizon,
            control_dim=self.control_dim,
            num_waypoints=self.num_waypoints,
            dt=self.dt,
            control_low=low,
            control_high=high
        )


def create_parametric_control(method: str, horizon: float, control_dim: int,
                             num_params: int, dt: float,
                             control_low: tuple = None, control_high: tuple = None) -> ParametricControl:
    """
    Factory function to create parametric control instances.

    Args:
        method: 'zoh' or 'foh' (also accepts 'linear_interp' for backward compatibility)
        horizon: Time horizon T
        control_dim: Control dimension
        num_params: Number of parameters
        dt: Time step
        control_low: Lower bounds tuple or None
        control_high: Upper bounds tuple or None

    Returns:
        ParametricControl instance
    """
    method = method.lower().strip()

    if method == 'zoh':
        return ZOHParametricControl(horizon, control_dim, num_params, dt, control_low, control_high)
    elif method  == 'foh':
        return FOHParametricControl(horizon, control_dim, num_params, dt, control_low, control_high)
    else:
        raise ValueError(f"Unknown method: {method}")