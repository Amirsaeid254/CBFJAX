import jax
import jax.numpy as jnp
import diffrax
from typing import Callable, Union
from functools import partial

from .utils import ensure_batch_dim

def get_solver(method: str):
    """
    Get diffrax solver based on method name.

    Args:
        method: Solver method name

    Returns:
        Diffrax solver instance
    """
    solver_map = {
        'euler': diffrax.Euler(),
        'tsit5': diffrax.Tsit5(),
        'dopri5': diffrax.Dopri5(),
        'dopri8': diffrax.Dopri8(),
        'bosh3': diffrax.Bosh3(),
        'heun': diffrax.Heun(),
        'midpoint': diffrax.Midpoint(),
    }

    if method not in solver_map:
        raise ValueError(f"Unknown solver method: {method}. Available: {list(solver_map.keys())}")

    return solver_map[method]


def get_trajs_from_state_action_func(x0: jnp.ndarray, dynamics, action_func: Callable,
                                     timestep: float, sim_time: float, method: str = 'tsit5') -> jnp.ndarray:
    """
    Generate trajectories from action function using diffrax.

    Args:
        x0: Initial states (batch, state_dim) or (state_dim,)
        dynamics: Dynamics object with rhs method
        action_func: Function that computes control given state
        timestep: Integration timestep (must be static)
        sim_time: Total simulation time (must be static)
        method: Integration method

    Returns:
        Trajectories (time_steps, batch, state_dim)
    """
    # Ensure batch dimension using existing utility
    x0 = ensure_batch_dim(x0)

    steps = int(sim_time / timestep) + 1

    # Time points - now safe because timestep and sim_time are static
    t_eval = jnp.linspace(0.0, sim_time, steps)

    # Define ODE function
    def ode_func(t, y, args):
        # y shape: (batch, state_dim)
        control = jax.vmap(action_func)(y)  # Vectorize action function over batch
        return jax.vmap(dynamics.rhs, in_axes=(0, 0))(y, control)  # Vectorize dynamics

    # Set up diffrax problem
    solver = get_solver(method)

    # Use RecursiveCheckpointAdjoint for gradient computation
    adjoint = diffrax.RecursiveCheckpointAdjoint()

    term = diffrax.ODETerm(ode_func)
    solution = diffrax.diffeqsolve(
        terms=term,
        solver=solver,
        t0=0.0,
        t1=sim_time,
        dt0=timestep,
        y0=x0,
        saveat=diffrax.SaveAt(ts=t_eval),
        adjoint=adjoint,
        max_steps=steps * 5,  # Conservative buffer for adaptive methods
    )

    # Extract trajectories: (time_steps, batch, state_dim)
    return solution.ys


def get_trajs_from_state_action_func_zoh(x0: jnp.ndarray, dynamics, action_func: Callable,
                                         timestep: float, sim_time: float, intermediate_steps: int = 2,
                                         method: str = 'tsit5') -> jnp.ndarray:
    """
    Generate trajectories with zero-order hold control using diffrax.

    Args:
        x0: Initial states (batch, state_dim) or (state_dim,)
        dynamics: Dynamics object with rhs method
        action_func: Function that computes control given state
        timestep: Control update timestep (must be static)
        sim_time: Total simulation time (must be static)
        intermediate_steps: Integration substeps per control update
        method: Integration method

    Returns:
        Trajectories (time_steps, batch, state_dim)
    """
    # Ensure batch dimension using existing utility
    x0 = ensure_batch_dim(x0)

    batch_size, state_dim = x0.shape
    num_steps = int(sim_time / timestep) + 1


    solver = get_solver(method)
    adjoint = diffrax.RecursiveCheckpointAdjoint()

    def step_forward(carry, i):
        current_state = carry

        # Compute control for current batch of states (zero-order hold)
        current_controls = jax.vmap(action_func)(current_state)

        def ode_func(t, y, args):
            return jax.vmap(dynamics.rhs, in_axes=(0, 0))(y, current_controls)

        # Set up integration time points
        t_eval = jnp.linspace(0.0, timestep, intermediate_steps)
        saveat = diffrax.SaveAt(ts=t_eval)

        # Integrate over one timestep with fixed control
        term = diffrax.ODETerm(ode_func)
        solution = diffrax.diffeqsolve(
            terms=term,
            solver=solver,
            t0=0.0,
            t1=timestep,
            dt0=timestep / (intermediate_steps - 1) if intermediate_steps > 1 else timestep,
            y0=current_state,
            adjoint=adjoint,
            saveat=saveat,
            max_steps=intermediate_steps * 5,  # Conservative buffer for adaptive methods
        )

        # Extract final state
        next_state = solution.ys[-1]

        return next_state, next_state

    # Use scan to iterate through timesteps
    _, states_sequence = jax.lax.scan(step_forward, x0, jnp.arange(num_steps - 1))

    # Concatenate initial state with computed states
    trajs = jnp.concatenate([jnp.expand_dims(x0, 0), states_sequence], axis=0)

    return trajs




def get_trajs_from_time_action_func(x0: jnp.ndarray, dynamics, action_func: Callable,
                                     timestep: float, sim_time: float, method: str = 'tsit5') -> jnp.ndarray:
    """
    Generate trajectories from action function using diffrax.

    Args:
        x0: Initial states (batch, state_dim) or (state_dim,)
        dynamics: Dynamics object with rhs method
        action_func: Function that computes control given state
        timestep: Integration timestep (must be static)
        sim_time: Total simulation time (must be static)
        method: Integration method

    Returns:
        Trajectories (time_steps, batch, state_dim)
    """
    # Ensure batch dimension
    x0 = ensure_batch_dim(x0)

    steps = int(sim_time / timestep) + 1

    # Time points - now safe because timestep and sim_time are static
    t_eval = jnp.linspace(0.0, sim_time, steps)

    # Define ODE function
    def ode_func(t, y, args):
        # y shape: (batch, state_dim)
        control = jax.vmap(action_func)(t)  # Vectorize action function over batch
        return jax.vmap(dynamics.rhs, in_axes=(0, 0))(y, control)  # Vectorize dynamics

    # Set up diffrax problem
    solver = get_solver(method)

    # Use RecursiveCheckpointAdjoint for gradient computation
    adjoint = diffrax.RecursiveCheckpointAdjoint()

    term = diffrax.ODETerm(ode_func)
    solution = diffrax.diffeqsolve(
        terms=term,
        solver=solver,
        t0=0.0,
        t1=sim_time,
        dt0=timestep,
        y0=x0,
        saveat=diffrax.SaveAt(ts=t_eval),
        adjoint=adjoint,
        max_steps=steps * 5,  # Conservative buffer for adaptive methods
    )

    # Extract trajectories: (time_steps, batch, state_dim)
    return solution.ys


def get_trajs_from_time_action_func_zoh(x0: jnp.ndarray, dynamics, action_func: Callable,
                                         timestep: float, sim_time: float, intermediate_steps: int = 2,
                                         method: str = 'tsit5') -> jnp.ndarray:
    """
    Generate trajectories with zero-order hold control using diffrax.

    Args:
        x0: Initial states (batch, state_dim) or (state_dim,)
        dynamics: Dynamics object with rhs method
        action_func: Function that computes control given time
        timestep: Control update timestep (must be static)
        sim_time: Total simulation time (must be static)
        intermediate_steps: Integration substeps per control update
        method: Integration method

    Returns:
        Trajectories (time_steps, batch, state_dim)
    """
    # Ensure batch dimension
    x0 = ensure_batch_dim(x0)

    batch_size, state_dim = x0.shape
    num_steps = int(sim_time / timestep) + 1


    solver = get_solver(method)
    adjoint = diffrax.RecursiveCheckpointAdjoint()

    def step_forward(carry, i):
        current_state = carry
        current_time = i * timestep

        # Compute control at current time
        current_controls = jax.vmap(action_func)(jnp.full(batch_size, current_time))

        def ode_func(t, y, args):
            return jax.vmap(dynamics.rhs, in_axes=(0, 0))(y, current_controls)

        # Set up integration time points
        t_eval = jnp.linspace(0.0, timestep, intermediate_steps)
        saveat = diffrax.SaveAt(ts=t_eval)

        # Integrate over one timestep with fixed control
        term = diffrax.ODETerm(ode_func)
        solution = diffrax.diffeqsolve(
            terms=term,
            solver=solver,
            t0=0.0,
            t1=timestep,
            dt0=timestep / (intermediate_steps - 1) if intermediate_steps > 1 else timestep,
            y0=current_state,
            adjoint=adjoint,
            saveat=saveat,
            max_steps=intermediate_steps * 5,  # Conservative buffer for adaptive methods
        )

        # Extract final state
        next_state = solution.ys[-1]

        return next_state, next_state

    # Use scan to iterate through timesteps
    _, states_sequence = jax.lax.scan(step_forward, x0, jnp.arange(num_steps - 1))

    # Concatenate initial state with computed states
    trajs = jnp.concatenate([jnp.expand_dims(x0, 0), states_sequence], axis=0)

    return trajs