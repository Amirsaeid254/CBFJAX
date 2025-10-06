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

    # Time points
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
        action_func: Function that computes control given SINGLE state
        timestep: Control update timestep
        sim_time: Total simulation time
        intermediate_steps: Integration substeps per control update
        method: Integration method

    Returns:
        Trajectories (time_steps, batch, state_dim)
    """
    x0 = ensure_batch_dim(x0)
    num_steps = int(sim_time / timestep) + 1

    solver = get_solver(method)
    adjoint = diffrax.RecursiveCheckpointAdjoint()

    def step_forward(carry, i):
        current_state = carry
        current_controls = jax.vmap(action_func)(current_state)

        def ode_func(t, y, args):
            controls = args
            return jax.vmap(dynamics.rhs, in_axes=(0, 0))(y, controls)

        term = diffrax.ODETerm(ode_func)
        solution = diffrax.diffeqsolve(
            terms=term,
            solver=solver,
            t0=0.0,
            t1=timestep,
            dt0=timestep / intermediate_steps,
            y0=current_state,
            args=current_controls,
            adjoint=adjoint,
            saveat=diffrax.SaveAt(t1=True),
            max_steps=intermediate_steps * 5,
        )

        next_state = solution.ys[0]
        return next_state, next_state

    _, states_sequence = jax.lax.scan(step_forward, x0, jnp.arange(num_steps - 1))
    trajs = jnp.concatenate([jnp.expand_dims(x0, 0), states_sequence], axis=0)

    return trajs


def get_trajs_from_time_action_func_single(x0: jnp.ndarray, dynamics, action_func: Callable,
                                     timestep: Union[float, jnp.ndarray] = None,
                                     start_time: Union[float, jnp.ndarray] = 0.0,
                                     sim_time: Union[float, jnp.ndarray] = None,
                                     num_steps: int = None, method: str = 'tsit5') -> jnp.ndarray:
    """
    Generate trajectories from action function using diffrax.

    Args:
        x0: Initial states (state_dim,)
        dynamics: Dynamics object with rhs method
        action_func: Function that computes control given time
        timestep: Integration timestep (optional if num_steps provided, can be jnp.ndarray)
        start_time: Start time for integration (can be jnp.ndarray for gradient)
        sim_time: Total simulation time (can be jnp.ndarray for gradient)
        num_steps: Number of time steps (static, required when sim_time is traced)
        method: Integration method

    Returns:
        Trajectories (time_steps, state_dim)
    """
    # Convert to jnp arrays if not already
    start_time = jnp.asarray(start_time)
    if sim_time is not None:
        sim_time = jnp.asarray(sim_time)

    # Handle static num_steps with traced sim_time
    if num_steps is not None:
        steps = num_steps
        # timestep is computed adaptively from sim_time / (steps - 1)
        if timestep is None:
            timestep = sim_time / (steps - 1) if steps > 1 else sim_time
    else:
        if timestep is None or sim_time is None:
            raise ValueError("Must provide either (timestep, sim_time) or (num_steps, sim_time)")
        steps = int(sim_time / timestep) + 1

    t_eval = jnp.linspace(start_time, sim_time, steps)

    def vector_field(t, y, args):
        return dynamics.rhs(y, action_func(t))

    # Set up diffrax problem
    solver = get_solver(method)

    # Use RecursiveCheckpointAdjoint for gradient computation
    adjoint = diffrax.RecursiveCheckpointAdjoint()


    term = diffrax.ODETerm(vector_field)
    solution = diffrax.diffeqsolve(
        terms=term,
        solver=solver,
        t0=start_time,
        t1=sim_time,
        dt0=timestep,
        y0=x0,
        saveat=diffrax.SaveAt(ts=t_eval),
        adjoint=adjoint,
        max_steps=steps * 5,  # Conservative buffer for adaptive methods
    )

    # Extract trajectories: (time_steps, state_dim)
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
        timestep: Control update timestep
        sim_time: Total simulation time
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


def get_trajs_from_state_action_func_no_vmap(x0: jnp.ndarray, dynamics, action_func: Callable,
                                              timestep: float, sim_time: float, method: str = 'tsit5') -> jnp.ndarray:
    """
    Generate trajectories from action function using diffrax with Python loop for batching.

    Use this when action_func is not JAX-compatible (e.g., uses external libraries like CVXOPT).

    Args:
        x0: Initial states (batch, state_dim) or (state_dim,)
        dynamics: Dynamics object with rhs method
        action_func: Function that computes control given SINGLE state (not vmappable)
        timestep: Integration timestep (must be static)
        sim_time: Total simulation time (must be static)
        method: Integration method

    Returns:
        Trajectories (time_steps, batch, state_dim)
    """
    # Ensure batch dimension using existing utility
    x0 = ensure_batch_dim(x0)
    batch_size = x0.shape[0]

    steps = int(sim_time / timestep) + 1
    t_eval = jnp.linspace(0.0, sim_time, steps)

    # Process each initial state independently
    trajectories = []
    for i in range(batch_size):
        # Define ODE function for single trajectory
        def ode_func(t, y, args):
            control = action_func(y)
            return dynamics.rhs(y, control)

        # Set up diffrax problem
        solver = get_solver(method)
        adjoint = diffrax.RecursiveCheckpointAdjoint()

        term = diffrax.ODETerm(ode_func)
        solution = diffrax.diffeqsolve(
            terms=term,
            solver=solver,
            t0=0.0,
            t1=sim_time,
            dt0=timestep,
            y0=x0[i],
            saveat=diffrax.SaveAt(ts=t_eval),
            adjoint=adjoint,
            max_steps=steps * 5,
        )

        trajectories.append(solution.ys)

    # Stack trajectories: (time_steps, batch, state_dim)
    return jnp.stack(trajectories, axis=1)


def get_trajs_from_state_action_func_zoh_no_vmap(x0: jnp.ndarray, dynamics, action_func: Callable,
                                                   timestep: float, sim_time: float, intermediate_steps: int = 2,
                                                   method: str = 'tsit5') -> jnp.ndarray:
    """
    Generate trajectories with zero-order hold control using diffrax with Python loop for batching.

    Use this when action_func is not JAX-compatible (e.g., uses external libraries like CVXOPT).
    The ODE integration between control updates is JIT-compiled.

    Args:
        x0: Initial states (batch, state_dim) or (state_dim,)
        dynamics: Dynamics object with rhs method
        action_func: Function that computes control given SINGLE state (not vmappable)
        timestep: Control update timestep
        sim_time: Total simulation time
        intermediate_steps: Integration substeps per control update
        method: Integration method

    Returns:
        Trajectories (time_steps, batch, state_dim)
    """
    x0 = ensure_batch_dim(x0)
    batch_size = x0.shape[0]
    num_steps = int(sim_time / timestep) + 1

    solver = get_solver(method)
    adjoint = diffrax.RecursiveCheckpointAdjoint()

    # JIT-compiled ODE step with fixed control (ZOH)
    @jax.jit
    def integrate_with_fixed_control(current_state, control):
        def ode_func(t, y, args):
            return dynamics.rhs(y, args)

        term = diffrax.ODETerm(ode_func)
        solution = diffrax.diffeqsolve(
            terms=term,
            solver=solver,
            t0=0.0,
            t1=timestep,
            dt0=timestep / intermediate_steps,
            y0=current_state,
            args=control,
            adjoint=adjoint,
            saveat=diffrax.SaveAt(t1=True),
            max_steps=intermediate_steps * 5,
        )
        return solution.ys[0]

    # Process each initial state independently
    trajectories = []
    for i in range(batch_size):
        traj = [x0[i]]
        current_state = x0[i]

        # Python loop over timesteps (cannot use lax.scan due to CVXOPT in action_func)
        for _ in range(num_steps - 1):
            current_control = action_func(current_state)
            next_state = integrate_with_fixed_control(current_state, current_control)
            traj.append(next_state)
            current_state = next_state

        trajectories.append(jnp.stack(traj, axis=0))

    # Stack trajectories: (time_steps, batch, state_dim)
    return jnp.stack(trajectories, axis=1)