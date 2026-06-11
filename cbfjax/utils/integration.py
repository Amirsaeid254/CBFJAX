"""
Single-trajectory ODE integration utilities + one ensemble rollout template.

Every single-trajectory function here integrates ONE trajectory: inputs are
(n,), outputs are (T, n). No batch axes, no internal vmap. All batching
(jax.vmap or host Python loops) is the caller's responsibility (see
base_control, backup_barrier).

The one exception is ``get_ensemble_trajs_zoh`` (bottom of this file): the
authoritative home for an N-robot ensemble rollout. It wraps the single-traj
ZOH integrator in ``eqx.filter_vmap`` over the members of a stacked controller
pytree, giving each member its own dynamics, action function, and controller
state lane.

Eager diffrax recompiles (~250ms/call) on every invocation, so these functions
MUST be called from within a caller's jit/vmap context.
"""
import jax
import jax.numpy as jnp
import diffrax
from typing import Callable, Union


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
                                     timestep: float, sim_time: float, method: str = 'tsit5',
                                     use_disturbed: bool = False) -> jnp.ndarray:
    """
    Generate a single trajectory from a state-feedback action function using diffrax.

    Must be called under jit/vmap (eager diffrax recompiles ~250ms/call).

    Args:
        x0: Initial state (state_dim,)
        dynamics: Dynamics object with rhs method
        action_func: Function that computes control given state (stateless: x -> u)
        timestep: Integration timestep (must be static)
        sim_time: Total simulation time (must be static)
        method: Integration method
        use_disturbed: If True, use disturbed_rhs for closed-loop simulation

    Returns:
        Trajectory (time_steps, state_dim)
    """
    if x0.ndim != 1:
        raise ValueError(f"x0 must be a single state of shape (n,), got shape {x0.shape}")

    rhs_func = dynamics.disturbed_rhs if use_disturbed else dynamics.rhs

    steps = int(sim_time / timestep) + 1
    t_eval = jnp.linspace(0.0, sim_time, steps)

    def ode_func(t, y, args):
        control = action_func(y)
        return rhs_func(y, control)

    solver = get_solver(method)
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

    # Extract trajectory: (time_steps, state_dim)
    return solution.ys


def get_trajs_from_state_action_func_zoh(x0: jnp.ndarray, dynamics, action_func: Callable,
                                         timestep: float, sim_time: float, intermediate_steps: int = 2,
                                         method: str = 'tsit5', init_ctrl_state=None,
                                         use_disturbed: bool = False) -> jnp.ndarray:
    """
    Generate a single trajectory with zero-order hold control using diffrax.

    Supports stateful action functions that return (u, new_state).
    Must be called under jit/vmap (eager diffrax recompiles ~250ms/call).

    Args:
        x0: Initial state (state_dim,)
        dynamics: Dynamics object with rhs method
        action_func: Stateful function (x, ctrl_state) -> (u, new_ctrl_state),
                     or stateless function x -> u (if init_ctrl_state is None)
        timestep: Control update timestep
        sim_time: Total simulation time
        intermediate_steps: Integration substeps per control update
        method: Integration method
        init_ctrl_state: Initial controller state (None for stateless)
        use_disturbed: If True, use disturbed_rhs for closed-loop simulation

    Returns:
        Trajectory (time_steps, state_dim)
    """
    if x0.ndim != 1:
        raise ValueError(f"x0 must be a single state of shape (n,), got shape {x0.shape}")

    num_steps = int(sim_time / timestep) + 1
    rhs_func = dynamics.disturbed_rhs if use_disturbed else dynamics.rhs

    solver = get_solver(method)
    adjoint = diffrax.RecursiveCheckpointAdjoint()

    if init_ctrl_state is not None:
        # Stateful action function: (x, ctrl_state) -> (u, new_ctrl_state)
        def step_forward(carry, i):
            current_state, ctrl_state = carry
            current_control, new_ctrl_state = action_func(current_state, ctrl_state)

            def ode_func(t, y, args):
                return rhs_func(y, args)

            term = diffrax.ODETerm(ode_func)
            solution = diffrax.diffeqsolve(
                terms=term,
                solver=solver,
                t0=0.0,
                t1=timestep,
                dt0=timestep / intermediate_steps,
                y0=current_state,
                args=current_control,
                adjoint=adjoint,
                saveat=diffrax.SaveAt(t1=True),
                max_steps=intermediate_steps * 5,
            )

            next_state = solution.ys[0]
            return (next_state, new_ctrl_state), next_state

        _, states_sequence = jax.lax.scan(step_forward, (x0, init_ctrl_state), jnp.arange(num_steps - 1))
    else:
        # Stateless action function: x -> u
        def step_forward(carry, i):
            current_state = carry
            current_control = action_func(current_state)

            def ode_func(t, y, args):
                return rhs_func(y, args)

            term = diffrax.ODETerm(ode_func)
            solution = diffrax.diffeqsolve(
                terms=term,
                solver=solver,
                t0=0.0,
                t1=timestep,
                dt0=timestep / intermediate_steps,
                y0=current_state,
                args=current_control,
                adjoint=adjoint,
                saveat=diffrax.SaveAt(t1=True),
                max_steps=intermediate_steps * 5,
            )

            next_state = solution.ys[0]
            return next_state, next_state

        _, states_sequence = jax.lax.scan(step_forward, x0, jnp.arange(num_steps - 1))

    trajs = jnp.concatenate([jnp.expand_dims(x0, 0), states_sequence], axis=0)

    return trajs


def get_trajs_from_time_action_func(x0: jnp.ndarray, dynamics, action_func: Callable,
                                    timestep: Union[float, jnp.ndarray] = None,
                                    start_time: Union[float, jnp.ndarray] = 0.0,
                                    sim_time: Union[float, jnp.ndarray] = None,
                                    num_steps: int = None, method: str = 'tsit5',
                                    use_disturbed: bool = False) -> jnp.ndarray:
    """
    Generate a single trajectory from a time-indexed action function using diffrax.

    Must be called under jit/vmap (eager diffrax recompiles ~250ms/call).

    Args:
        x0: Initial state (state_dim,)
        dynamics: Dynamics object with rhs method
        action_func: Function that computes control given time
        timestep: Integration timestep (optional if num_steps provided, can be jnp.ndarray)
        start_time: Start time for integration (can be jnp.ndarray for gradient)
        sim_time: Total simulation time (can be jnp.ndarray for gradient)
        num_steps: Number of time steps (static, required when sim_time is traced)
        method: Integration method
        use_disturbed: If True, use disturbed_rhs for closed-loop simulation

    Returns:
        Trajectory (time_steps, state_dim)
    """
    if x0.ndim != 1:
        raise ValueError(f"x0 must be a single state of shape (n,), got shape {x0.shape}")

    rhs_func = dynamics.disturbed_rhs if use_disturbed else dynamics.rhs

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
        return rhs_func(y, action_func(t))

    solver = get_solver(method)
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

    # Extract trajectory: (time_steps, state_dim)
    return solution.ys


def get_trajs_from_time_action_func_with_dense(x0: jnp.ndarray, dynamics, action_func: Callable,
                                               timestep: Union[float, jnp.ndarray] = None,
                                               start_time: Union[float, jnp.ndarray] = 0.0,
                                               sim_time: Union[float, jnp.ndarray] = None,
                                               num_steps: int = None, method: str = 'tsit5',
                                               use_disturbed: bool = False) -> jnp.ndarray:
    """
    Generate a single trajectory from a time-indexed action function with dense output.

    Must be called under jit/vmap (eager diffrax recompiles ~250ms/call).

    Args:
        x0: Initial state (state_dim,)
        dynamics: Dynamics object with rhs method
        action_func: Function that computes control given time
        timestep: Integration timestep (optional if num_steps provided, can be jnp.ndarray)
        start_time: Start time for integration (can be jnp.ndarray for gradient)
        sim_time: Total simulation time (can be jnp.ndarray for gradient)
        num_steps: Number of time steps (static, required when sim_time is traced)
        method: Integration method
        use_disturbed: If True, use disturbed_rhs for closed-loop simulation

    Returns:
        Tuple (trajectory (time_steps, state_dim), dense evaluate function)
    """
    if x0.ndim != 1:
        raise ValueError(f"x0 must be a single state of shape (n,), got shape {x0.shape}")

    rhs_func = dynamics.disturbed_rhs if use_disturbed else dynamics.rhs

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
        return rhs_func(y, action_func(t))

    solver = get_solver(method)
    adjoint = diffrax.RecursiveCheckpointAdjoint()

    term = diffrax.ODETerm(vector_field)
    solution = diffrax.diffeqsolve(
        terms=term,
        solver=solver,
        t0=start_time,
        t1=sim_time,
        dt0=timestep,
        y0=x0,
        saveat=diffrax.SaveAt(ts=t_eval, dense=True),
        adjoint=adjoint,
        max_steps=steps * 5,  # Conservative buffer for adaptive methods
    )

    # Extract trajectory: (time_steps, state_dim)
    return solution.ys, solution.evaluate


# --------------------------------------------------------------- ensemble
def get_ensemble_trajs_zoh(ensemble, x0s: jnp.ndarray, timestep: float,
                           sim_time: float, intermediate_steps: int = 2,
                           method: str = 'tsit5', init_ctrl_states=None) -> jnp.ndarray:
    """
    Roll out an N-robot ensemble with zero-order-hold control, one compiled call.

    This is the authoritative ensemble counterpart of the single-trajectory
    ``get_trajs_from_state_action_func_zoh``. It ``eqx.filter_vmap``s over the
    members of a stacked controller pytree; each member rolls out through the
    SINGLE-trajectory ZOH integrator with its OWN dynamics, its OWN stateful
    action function, and its OWN controller state lane. Must be called from a
    jit context (eager diffrax recompiles ~250ms/call).

    Args:
        ensemble: A stacked controller pytree (from ``stack_ensemble`` /
                  ``eqx.filter_vmap``): every leaf array carries a leading axis
                  of size N; static parts are shared. Each member must expose
                  ``_dynamics``, ``optimal_control(x, state) -> (u, new_state)``
                  and ``get_init_state()`` (the BaseControl interface).
        x0s: Initial states, one per robot (N, state_dim).
        timestep: Control update timestep (static).
        sim_time: Total simulation time (static).
        intermediate_steps: Integration substeps per control update.
        method: Integration method (see ``get_solver``).
        init_ctrl_states: Per-robot initial controller states. See STATEFUL below.

    Returns:
        Trajectories (N, time_steps, state_dim). The single-trajectory ZOH
        return shape is unchanged; this just adds the leading robot axis. Final
        controller states are NOT returned — per-robot lanes evolve internally
        inside each member's ``lax.scan`` and are not surfaced (keep the API
        symmetric with the single-trajectory integrator).

    STATEFUL handling (per-robot controller state lanes)
    ----------------------------------------------------
    Controller state is threaded functionally inside the single-trajectory ZOH
    scan (``u, state = member.optimal_control(x, state)``), so under the
    ``filter_vmap`` each member carries its OWN independent state lane (per-robot
    QP warm starts, iLQR nominal trajectories, MPPI nominal + PRNG key, ...).
    State STRUCTURE must be identical across members — guaranteed when the
    ensemble is built from one ``stack_ensemble`` template.

    Two modes for the initial lane:

    - ``init_ctrl_states is None`` (default): each member's initial state comes
      from its OWN ``get_init_state()``, evaluated PER MEMBER under the vmap.
      This is genuinely per-member when the state depends on a traced leaf
      (verified: an ensemble whose ``get_init_state`` reads a stacked leaf
      yields distinct per-member states under ``filter_vmap``). When the state
      depends only on STATIC params it is identical for every member — notably
      MPPI seeds its PRNG key from the static ``init_seed`` param, so every
      robot samples the SAME noise (correlated rollouts). To break that
      correlation, use the explicit mode below with per-robot keys.

    - ``init_ctrl_states`` given: a stacked pytree (leading axis N, same
      structure as one member's ``get_init_state()``) injected per robot. This
      is the documented path for per-robot MPPI PRNG keys
      (``jax.vmap(jax.random.PRNGKey)(seeds)`` or ``jax.random.split``) and for
      per-robot warm starts / nominal-trajectory guesses. For stateless
      controllers (``get_init_state() is None``) pass ``None``.

    (This block is the authoritative location for the ensemble stateful
    semantics; ``stack_ensemble``'s docstring points here.)
    """
    import equinox as eqx

    if init_ctrl_states is None:
        @eqx.filter_vmap
        def _rollout(member, x0):
            init_ctrl_state = member.get_init_state()
            if init_ctrl_state is None:
                action_func = member._optimal_control_for_ode()
                return get_trajs_from_state_action_func_zoh(
                    x0=x0, dynamics=member._dynamics, action_func=action_func,
                    timestep=timestep, sim_time=sim_time,
                    intermediate_steps=intermediate_steps, method=method,
                    init_ctrl_state=None,
                )

            def stateful_action_func(x, ctrl_state):
                return member.optimal_control(x, ctrl_state)

            return get_trajs_from_state_action_func_zoh(
                x0=x0, dynamics=member._dynamics, action_func=stateful_action_func,
                timestep=timestep, sim_time=sim_time,
                intermediate_steps=intermediate_steps, method=method,
                init_ctrl_state=init_ctrl_state,
            )

        return _rollout(ensemble, x0s)

    @eqx.filter_vmap
    def _rollout_injected(member, x0, init_ctrl_state):
        def stateful_action_func(x, ctrl_state):
            return member.optimal_control(x, ctrl_state)

        return get_trajs_from_state_action_func_zoh(
            x0=x0, dynamics=member._dynamics, action_func=stateful_action_func,
            timestep=timestep, sim_time=sim_time,
            intermediate_steps=intermediate_steps, method=method,
            init_ctrl_state=init_ctrl_state,
        )

    return _rollout_injected(ensemble, x0s, init_ctrl_states)
