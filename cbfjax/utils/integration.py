import jax
import jax.numpy as jnp
import diffrax
from typing import Callable, Union


from .utils import ensure_batch_dim


@jax.jit
def _safe_optimal_control_impl(controller, x: jnp.ndarray, ret_info: bool = False) -> Union[jnp.ndarray, tuple]:
    """
    JIT-compiled implementation for safe optimal control.

    Args:
        controller: Controller instance
        x: State(s) (state_dim,) or (batch, state_dim)
        ret_info: Whether to return additional information

    Returns:
        Control(s) with shape (batch, action_dim) or tuple with info
    """
    # Ensure batch dimension
    x_batched = ensure_batch_dim(x)

    # Get results from _safe_optimal_control_single (either control array or tuple)
    results = jax.vmap(
        lambda state: controller._safe_optimal_control_single(state, ret_info=ret_info)
    )(x_batched)

    return results



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
        'rk4': diffrax.Runge_Kutta(tableau=diffrax.ButcherTableau.rk4()),
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


@jax.jit
def get_trajs_from_action_func(x0: jnp.ndarray, dynamics, action_func: Callable,
                               timestep: float, sim_time: float, method: str = 'tsit5') -> jnp.ndarray:
    """
    Generate trajectories from action function using diffrax.

    JAX version of PyTorch get_trajs_from_action_func with RecursiveCheckpointAdjoint.

    Args:
        x0: Initial states (batch, state_dim) or (state_dim,)
        dynamics: Dynamics object with rhs method
        action_func: Function that computes control given state
        timestep: Integration timestep
        sim_time: Total simulation time
        method: Integration method

    Returns:
        Trajectories (time_steps, batch, state_dim)
    """
    # Ensure batch dimension using existing utility
    x0 = ensure_batch_dim(x0)

    # Time points
    t_eval = jnp.linspace(0.0, sim_time, int(sim_time / timestep) + 1)

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
        max_steps=int(sim_time / timestep) * 10  # Safety factor
    )

    # Extract trajectories: (time_steps, batch, state_dim)
    return solution.ys


@jax.jit
def get_trajs_from_action_func_zoh(x0: jnp.ndarray, dynamics, action_func: Callable,
                                   timestep: float, sim_time: float, intermediate_steps: int = 2,
                                   method: str = 'tsit5') -> jnp.ndarray:
    """
    Generate trajectories with zero-order hold control using diffrax.

    JAX version of PyTorch get_trajs_from_action_func_zoh.

    Args:
        x0: Initial states (batch, state_dim) or (state_dim,)
        dynamics: Dynamics object with rhs method
        action_func: Function that computes control given state
        timestep: Control update timestep
        sim_time: Total simulation time
        intermediate_steps: Integration substeps per control update
        method: Integration method

    Returns:
        Trajectories (time_steps, batch, state_dim)
    """
    # Ensure batch dimension using existing utility
    x0 = ensure_batch_dim(x0)

    batch_size, state_dim = x0.shape
    num_steps = int(sim_time / timestep) + 1

    # Integration substep size
    dt_sub = timestep / intermediate_steps

    solver = get_solver(method)
    adjoint = diffrax.RecursiveCheckpointAdjoint()

    def step_forward(carry, i):
        current_state = carry

        # Compute control for current state (held constant for this timestep)
        current_controls = jax.vmap(action_func)(current_state)

        # Define ODE with fixed control
        def ode_func_fixed(t, y, args):
            return jax.vmap(dynamics.rhs, in_axes=(0, 0))(y, current_controls)

        # Integrate one timestep with fixed control
        term = diffrax.ODETerm(ode_func_fixed)
        solution = diffrax.diffeqsolve(
            terms=term,
            solver=solver,
            t0=0.0,
            t1=timestep,
            dt0=dt_sub,
            y0=current_state,
            adjoint=adjoint,
            max_steps=intermediate_steps * 2  # Safety factor
        )

        next_state = solution.ys  # Final state
        return next_state, next_state

    # Use scan to iterate through timesteps
    _, states_sequence = jax.lax.scan(step_forward, x0, jnp.arange(num_steps - 1))

    # Concatenate initial state with computed states
    return jnp.concatenate([jnp.expand_dims(x0, 0), states_sequence], axis=0)
