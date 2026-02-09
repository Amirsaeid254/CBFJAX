"""
NMPC (Nonlinear Model Predictive Control) using JAX + jax2casadi + acados.

This module provides NMPC controllers where dynamics and cost functions are
defined in JAX, converted to CasADi via jax2casadi, and solved using acados.

Classes:
    NMPCControl: Base NMPC controller with EXTERNAL cost (JAX callables)
    QuadraticNMPCControl: NMPC with LINEAR_LS cost (Q, R matrices)
"""

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import casadi as ca
from typing import Callable, Optional, Any, Tuple

from ..utils.jax2casadi import convert

from .base_control import BaseControl, QuadraticCostMixin


def _import_acados():
    from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver, AcadosSim, AcadosSimSolver
    return AcadosModel, AcadosOcp, AcadosOcpSolver, AcadosSim, AcadosSimSolver
from .control_types import NMPCInfo
from ..dynamics.base_dynamic import DummyDynamics


class NMPCControl(BaseControl):
    """
    Nonlinear Model Predictive Control using acados with EXTERNAL cost.

    Dynamics and cost are defined in JAX, converted to CasADi via jax2casadi,
    and the resulting OCP is solved using acados.

    All configuration is stored in params dict following codebase patterns:
        params = {
            # Horizon settings
            'horizon': 2.0,          # Total prediction horizon time [s]
            'time_steps': 0.04,      # Timestep [s] (horizon/N_horizon)

            # Solver options
            'qp_solver': 'PARTIAL_CONDENSING_HPIPM',
            'hessian_approx': 'GAUSS_NEWTON',
            'integrator_type': 'ERK',
            'sim_method_num_stages': 4,
            'nlp_solver_type': 'SQP_RTI',
            'nlp_solver_max_iter': 200,
            'qp_solver_iter_max': 100,
            'tol': 1e-4,
        }

    Attributes:
        _control_low: Lower bounds for control inputs (tuple)
        _control_high: Upper bounds for control inputs (tuple)
        _state_bounds_idx: Indices of bounded states (tuple)
        _state_low: Lower bounds for bounded states (tuple)
        _state_high: Upper bounds for bounded states (tuple)
        _cost_running: Running cost function (JAX callable)
        _cost_terminal: Terminal cost function (JAX callable)
    """

    # Control bounds
    _control_low: tuple = eqx.field(static=True)
    _control_high: tuple = eqx.field(static=True)
    _has_control_bounds: bool = eqx.field(static=True)

    # State bounds (optional)
    _state_bounds_idx: tuple = eqx.field(static=True)
    _state_low: tuple = eqx.field(static=True)
    _state_high: tuple = eqx.field(static=True)
    _has_state_bounds: bool = eqx.field(static=True)

    # Cost functions (JAX callables)
    _cost_running: Optional[Callable] = eqx.field(static=True)
    _cost_terminal: Optional[Callable] = eqx.field(static=True)

    # Build state (non-static since solver is mutable)
    _is_built: bool
    _solver: Any
    _dynamics_casadi: Any
    _ocp: Any
    _sim_solver: Any  # AcadosSimSolver for shift propagation

    def __init__(
        self,
        control_low: Optional[list] = None,
        control_high: Optional[list] = None,
        state_bounds_idx: Optional[list] = None,
        state_low: Optional[list] = None,
        state_high: Optional[list] = None,
        cost_running: Optional[Callable] = None,
        cost_terminal: Optional[Callable] = None,
        **kwargs
    ):
        """
        Initialize NMPCControl.

        Args:
            control_low: Lower bounds for control inputs
            control_high: Upper bounds for control inputs
            state_bounds_idx: Indices of bounded states
            state_low: Lower bounds for bounded states
            state_high: Upper bounds for bounded states
            cost_running: Running cost function f(x, u) -> scalar (JAX)
            cost_terminal: Terminal cost function f(x) -> scalar (JAX)
            **kwargs: Passed via cooperative inheritance (action_dim, params, dynamics)
        """
        # Default params for NMPC
        params = kwargs.get('params', None)
        default_params = {
            # Horizon settings
            'horizon': 2.0,
            'time_steps': 0.04,
            # Solver options
            'qp_solver': 'PARTIAL_CONDENSING_HPIPM',
            'hessian_approx': 'GAUSS_NEWTON',
            'integrator_type': 'ERK',
            'sim_method_num_stages': 4,
            'nlp_solver_type': 'SQP_RTI',
            'nlp_solver_max_iter': 200,
            'qp_solver_iter_max': 100,
            'tol': 1e-4,
        }
        if params is not None:
            default_params.update(params)
        kwargs['params'] = default_params

        super().__init__(**kwargs)

        # Control bounds
        if control_low is not None and control_high is not None:
            self._control_low = tuple(control_low)
            self._control_high = tuple(control_high)
            self._has_control_bounds = True
        else:
            self._control_low = tuple()
            self._control_high = tuple()
            self._has_control_bounds = False

        # State bounds
        if state_bounds_idx is not None and state_low is not None and state_high is not None:
            self._state_bounds_idx = tuple(state_bounds_idx)
            self._state_low = tuple(state_low)
            self._state_high = tuple(state_high)
            self._has_state_bounds = True
        else:
            self._state_bounds_idx = ()
            self._state_low = ()
            self._state_high = ()
            self._has_state_bounds = False

        # Cost functions
        self._cost_running = cost_running
        self._cost_terminal = cost_terminal

        # Build state
        self._is_built = False
        self._solver = None
        self._dynamics_casadi = None
        self._ocp = None
        self._sim_solver = None

    @classmethod
    def create_empty(cls, action_dim: int, params: Optional[dict] = None) -> 'NMPCControl':
        """
        Create an empty NMPC controller for assignment chain.

        Args:
            action_dim: Dimension of control input
            params: Optional configuration parameters

        Returns:
            Empty NMPCControl instance ready for assignment
        """
        return cls(action_dim=action_dim, params=params)

    def _create_updated_instance(self, **kwargs):
        """
        Create new instance with updated fields.

        Args:
            **kwargs: Fields to update

        Returns:
            New NMPCControl instance with updated fields
        """
        defaults = {
            'action_dim': self._action_dim,
            'params': dict(self._params) if self._params else None,
            'dynamics': self._dynamics,
            'control_low': list(self._control_low) if self._has_control_bounds else None,
            'control_high': list(self._control_high) if self._has_control_bounds else None,
            'state_bounds_idx': list(self._state_bounds_idx) if self._has_state_bounds else None,
            'state_low': list(self._state_low) if self._has_state_bounds else None,
            'state_high': list(self._state_high) if self._has_state_bounds else None,
            'cost_running': self._cost_running,
            'cost_terminal': self._cost_terminal,
        }
        defaults.update(kwargs)
        return self.__class__(**defaults)

    # ==========================================
    # Assignment Methods
    # ==========================================

    def assign_dynamics(self, dynamics) -> 'NMPCControl':
        """
        Assign dynamics to controller.

        Args:
            dynamics: System dynamics object (AffineInControlDynamics)

        Returns:
            New NMPCControl instance with assigned dynamics
        """
        return self._create_updated_instance(dynamics=dynamics)

    def assign_control_bounds(self, low: list, high: list) -> 'NMPCControl':
        """
        Assign control input bounds.

        Args:
            low: Lower bounds for control inputs
            high: Upper bounds for control inputs

        Returns:
            New NMPCControl instance with bounds assigned
        """
        assert len(low) == len(high), 'low and high should have the same length'
        assert len(low) == self._action_dim, 'bounds length should match action dimension'
        return self._create_updated_instance(control_low=low, control_high=high)

    def assign_state_bounds(self, idx: list, low: list, high: list) -> 'NMPCControl':
        """
        Assign state bounds for specific state indices.

        Args:
            idx: Indices of bounded states
            low: Lower bounds for bounded states
            high: Upper bounds for bounded states

        Returns:
            New NMPCControl instance with state bounds assigned
        """
        assert len(idx) == len(low) == len(high), 'idx, low, high should have same length'
        return self._create_updated_instance(
            state_bounds_idx=idx, state_low=low, state_high=high
        )

    def assign_cost_running(self, cost_func: Callable) -> 'NMPCControl':
        """
        Assign running (stage) cost function.

        Args:
            cost_func: JAX function f(x, u) -> scalar

        Returns:
            New NMPCControl instance with running cost assigned
        """
        return self._create_updated_instance(cost_running=cost_func)

    def assign_cost_terminal(self, cost_func: Callable) -> 'NMPCControl':
        """
        Assign terminal cost function.

        Args:
            cost_func: JAX function f(x) -> scalar

        Returns:
            New NMPCControl instance with terminal cost assigned
        """
        return self._create_updated_instance(cost_terminal=cost_func)

    # ==========================================
    # Initial Guess / Warm-Starting
    # ==========================================

    def _get_default_guess(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get default guess: x0 replicated across horizon, zeros for u."""
        N = self.N_horizon
        nx = self._dynamics.state_dim
        nu = self._action_dim
        x0 = self._solver.get(0, "x")
        x_traj = np.tile(x0, (N + 1, 1))
        return x_traj, np.zeros((N, nu))

    def set_init_guess(
        self,
        x_traj: Optional[np.ndarray] = None,
        u_traj: Optional[np.ndarray] = None,
    ) -> None:
        """
        Set initial guess for the solver (warm-starting).

        If x_traj or u_traj is None, uses default zeros for that component.

        Args:
            x_traj: State trajectory guess (N+1, nx). If None, uses default.
            u_traj: Control trajectory guess (N, nu). If None, uses default.
        """
        assert self._is_built, "Must call make() before setting initial guess"

        N = self.N_horizon
        x_default, u_default = self._get_default_guess()

        if x_traj is None:
            x_traj = x_default
        x_traj = np.asarray(x_traj)
        for k in range(N + 1):
            self._solver.set(k, "x", x_traj[k])

        if u_traj is None:
            u_traj = u_default
        u_traj = np.asarray(u_traj)
        for k in range(N):
            self._solver.set(k, "u", u_traj[k])

    def set_init_guess_linear(
        self,
        x0: np.ndarray,
        x_target: np.ndarray,
    ) -> None:
        """
        Set a linear interpolation initial guess from x0 to x_target.

        Args:
            x0: Initial state (nx,)
            x_target: Target state (nx,)
        """
        x0 = np.asarray(x0)
        x_target = np.asarray(x_target)
        N = self.N_horizon

        alphas = np.linspace(0.0, 1.0, N + 1)
        x_traj = x0[None, :] + alphas[:, None] * (x_target - x0)[None, :]

        self.set_init_guess(x_traj=x_traj)

    # ==========================================
    # Post-Solve Warm-Start
    # ==========================================

    def _create_sim_solver(self, model):
        """Create an AcadosSimSolver for shift propagation using Euler integration."""
        _, _, _, AcadosSim, AcadosSimSolver = _import_acados()
        sim = AcadosSim()
        sim.model = model
        sim.solver_options.T = self.time_steps
        sim.solver_options.integrator_type = 'ERK'
        sim.solver_options.num_stages = 1
        sim.solver_options.num_steps = 1
        return AcadosSimSolver(sim)

    def _post_solve(self):
        """Post-solve hook: shift warm-start if sim_solver is available, no-op otherwise."""
        if self._sim_solver is None:
            return
        self._shift_warm_start()

    def _shift_warm_start(self):
        """Shift solution forward by one step and propagate dynamics."""
        N = self.N_horizon

        # Extract current solution
        x_traj = np.array([self._solver.get(k, "x") for k in range(N + 1)])
        u_traj = np.array([self._solver.get(k, "u") for k in range(N)])

        # Shift controls: drop first, repeat last
        u_shifted = np.empty_like(u_traj)
        u_shifted[:-1] = u_traj[1:]
        u_shifted[-1] = u_traj[-1]

        # Propagate states from x_traj[1] using shifted controls
        x_shifted = np.empty_like(x_traj)
        x_shifted[0] = x_traj[1]
        for k in range(N):
            x_shifted[k + 1] = self._sim_solver.simulate(x=x_shifted[k], u=u_shifted[k])

        # Set back into solver
        for k in range(N + 1):
            self._solver.set(k, "x", x_shifted[k])
        for k in range(N):
            self._solver.set(k, "u", u_shifted[k])

    # ==========================================
    # Properties for horizon/timestep access
    # ==========================================

    @property
    def horizon(self) -> float:
        """Get prediction horizon time [s]."""
        return self._params['horizon']

    @property
    def time_steps(self) -> float:
        """Get timestep [s]."""
        return self._params['time_steps']

    @property
    def N_horizon(self) -> int:
        """Get number of prediction horizon steps."""
        return int(self.horizon / self.time_steps)

    # ==========================================
    # Build Methods
    # ==========================================

    def _convert_dynamics_to_casadi(self) -> ca.Function:
        """Convert JAX dynamics to CasADi function."""
        nx = self._dynamics.state_dim
        nu = self._action_dim

        dynamics_ca = convert(
            self._dynamics.rhs,
            [('x', (nx,)), ('u', (nu,))],
            name='dynamics',
            validate=True,
            tolerance=1e-6
        )
        return dynamics_ca

    def _convert_cost_running_to_casadi(self) -> ca.Function:
        """Convert JAX running cost to CasADi function."""
        nx = self._dynamics.state_dim
        nu = self._action_dim

        cost_ca = convert(
            self._cost_running,
            [('x', (nx,)), ('u', (nu,))],
            name='cost_running',
            validate=True,
            tolerance=1e-4
        )
        return cost_ca

    def _convert_cost_terminal_to_casadi(self) -> ca.Function:
        """Convert JAX terminal cost to CasADi function."""
        nx = self._dynamics.state_dim
        nu = self._action_dim

        cost_ca = convert(
                self._cost_terminal,
                [('x', (nx,))],
                name='cost_terminal',
                validate=True,
                tolerance=1e-3
            )
        return cost_ca

    def _create_acados_model(self, dynamics_casadi: ca.Function):
        """Create acados model using converted dynamics."""
        AcadosModel, _, _, _, _ = _import_acados()
        model_name = "nmpc_model"

        nx = self._dynamics.state_dim
        nu = self._action_dim

        # CasADi symbolic variables
        x = ca.SX.sym('x', nx)
        u = ca.SX.sym('u', nu)
        xdot = ca.SX.sym('xdot', nx)

        # Use JAX-converted dynamics
        f_expl = dynamics_casadi(x, u)
        f_impl = xdot - f_expl

        # Create model
        model = AcadosModel()
        model.f_impl_expr = f_impl
        model.f_expl_expr = f_expl
        model.x = x
        model.xdot = xdot
        model.u = u
        model.name = model_name

        return model

    def _setup_cost(self, ocp, model):
        """Setup EXTERNAL cost function in OCP using JAX-converted functions."""
        nx = self._dynamics.state_dim
        nu = self._action_dim

        assert self._cost_running is not None, \
            "Running cost must be assigned. Use assign_cost_running()."

        # Use EXTERNAL cost with JAX-converted functions
        ocp.cost.cost_type = "EXTERNAL"
        ocp.cost.cost_type_e = "EXTERNAL"

        # Convert cost functions to CasADi
        cost_running_ca = self._convert_cost_running_to_casadi()

        if self._cost_terminal is not None:
            cost_terminal_ca = self._convert_cost_terminal_to_casadi()
        else:
            # Default terminal cost: same as running cost with u=0
            cost_terminal_ca = lambda x: cost_running_ca(x, ca.DM.zeros(nu))

        model.cost_expr_ext_cost = cost_running_ca(model.x, model.u)
        model.cost_expr_ext_cost_e = cost_terminal_ca(model.x)

    def _setup_constraints(self, ocp, model, x0: np.ndarray):
        """Setup constraints in OCP."""
        nu = self._action_dim

        # Control constraints
        if self._has_control_bounds:
            ocp.constraints.lbu = np.array(self._control_low)
            ocp.constraints.ubu = np.array(self._control_high)
            ocp.constraints.idxbu = np.arange(nu)

        # State constraints
        if self._has_state_bounds:
            ocp.constraints.lbx = np.array(self._state_low)
            ocp.constraints.ubx = np.array(self._state_high)
            ocp.constraints.idxbx = np.array(self._state_bounds_idx)

        # Initial state constraint
        ocp.constraints.x0 = x0

    def _setup_solver_options(self, ocp):
        """Setup solver options in OCP from params."""
        ocp.solver_options.N_horizon = self.N_horizon
        ocp.solver_options.tf = self.horizon

        ocp.solver_options.qp_solver = self._params['qp_solver']
        ocp.solver_options.hessian_approx = self._params['hessian_approx']
        ocp.solver_options.integrator_type = self._params['integrator_type']
        ocp.solver_options.sim_method_num_stages = self._params['sim_method_num_stages']
        ocp.solver_options.nlp_solver_type = self._params['nlp_solver_type']
        ocp.solver_options.nlp_solver_max_iter = self._params['nlp_solver_max_iter']
        ocp.solver_options.qp_solver_iter_max = self._params['qp_solver_iter_max']
        ocp.solver_options.tol = self._params['tol']

        # Set code export directory
        ocp.code_export_directory = 'c_generated_code'

    def make(self, x0: Optional[np.ndarray] = None) -> 'NMPCControl':
        """
        Build the NMPC controller.

        Converts JAX functions to CasADi, creates acados OCP, and initializes solver.

        Args:
            x0: Initial state for the OCP (required for acados)

        Returns:
            Self with solver built

        Raises:
            AssertionError: If required components are not assigned
        """
        # Assertions
        assert self.has_dynamics, "Dynamics must be assigned before make()"
        assert self._has_control_bounds, "Control bounds must be assigned before make()"
        assert self._cost_running is not None, \
            "Running cost must be assigned. Use assign_cost_running()."

        if x0 is None:
            # Default initial state
            x0 = np.zeros(self._dynamics.state_dim)

        print("Converting JAX dynamics to CasADi...")
        dynamics_casadi = self._convert_dynamics_to_casadi()
        object.__setattr__(self, '_dynamics_casadi', dynamics_casadi)

        _, AcadosOcp, AcadosOcpSolver, _, _ = _import_acados()

        print("Building acados OCP...")
        ocp = AcadosOcp()

        # Create model
        model = self._create_acados_model(dynamics_casadi)
        ocp.model = model

        # Setup cost
        self._setup_cost(ocp, model)

        # Setup constraints
        self._setup_constraints(ocp, model, x0)

        # Setup solver options
        self._setup_solver_options(ocp)

        # Store OCP
        object.__setattr__(self, '_ocp', ocp)

        # Create solver (force regeneration to avoid stale code)
        print("Creating acados solver...")
        solver = AcadosOcpSolver(ocp, build=True, generate=True)
        object.__setattr__(self, '_solver', solver)

        # Create sim solver for shift warm-start if enabled
        if self._params.get('shift_warm_start', False):
            print("Creating acados integrator for shift warm-start...")
            sim_solver = self._create_sim_solver(model)
            object.__setattr__(self, '_sim_solver', sim_solver)

        object.__setattr__(self, '_is_built', True)
        print("NMPC ready!")

        return self

    # ==========================================
    # Control Methods
    # ==========================================

    def _optimal_control_single(self, x: jnp.ndarray, state=None) -> Tuple[jnp.ndarray, Any]:
        """
        Compute optimal control for a single state.

        Args:
            x: Single state vector (state_dim,)
            state: Controller state (unused for NMPC, passed through)

        Returns:
            Tuple (u, new_state)
        """
        assert self._is_built, "Must call make() before computing control"

        # Convert JAX array to numpy for acados
        x_np = np.array(x)

        # Set initial state constraint
        self._solver.set(0, "lbx", x_np)
        self._solver.set(0, "ubx", x_np)

        # Solve
        status = self._solver.solve()
        u_opt = self._solver.get(0, "u")
        self._post_solve()

        # Convert back to JAX
        u_jax = jnp.array(u_opt)

        return u_jax, state

    def _optimal_control_single_with_info(self, x: jnp.ndarray, state=None) -> Tuple[jnp.ndarray, Any, dict]:
        """
        Compute optimal control with diagnostic info for a single state.

        Args:
            x: Single state vector (state_dim,)
            state: Controller state (unused for NMPC, passed through)

        Returns:
            Tuple (u, new_state, info)
        """
        assert self._is_built, "Must call make() before computing control"

        x_np = np.array(x)
        self._solver.set(0, "lbx", x_np)
        self._solver.set(0, "ubx", x_np)

        status = self._solver.solve()
        u_opt = self._solver.get(0, "u")
        cost = self._solver.get_cost()
        x_traj, u_traj = self._extract_trajectory()
        self._post_solve()

        u_jax = jnp.array(u_opt)

        info = NMPCInfo(
            status=jnp.array(status),
            cost=jnp.array(cost),
            x_traj=x_traj,
            u_traj=u_traj,
        )

        return u_jax, state, info

    def _extract_trajectory(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract predicted trajectory from solver after a solve.

        Returns:
            Tuple (x_traj, u_traj) where:
            - x_traj: Predicted states (N+1, nx)
            - u_traj: Predicted controls (N, nu)
        """
        N = self.N_horizon
        nx = self._dynamics.state_dim
        nu = self._action_dim

        x_traj = np.zeros((N + 1, nx))
        u_traj = np.zeros((N, nu))

        for k in range(N + 1):
            x_traj[k] = self._solver.get(k, "x")
        for k in range(N):
            u_traj[k] = self._solver.get(k, "u")

        return x_traj, u_traj

    def get_predicted_trajectory(self, x: jnp.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve OCP for given state and return predicted trajectory.

        This method solves the OCP starting from state x and returns
        the full predicted state and control trajectory.

        Args:
            x: Initial state vector (state_dim,)

        Returns:
            Tuple (x_traj, u_traj) where:
            - x_traj: Predicted states (N+1, nx)
            - u_traj: Predicted controls (N, nu)
        """
        assert self._is_built, "Must call make() before getting trajectory"

        # Convert JAX array to numpy for acados
        x_np = np.array(x)

        # Set initial state constraint
        self._solver.set(0, "lbx", x_np)
        self._solver.set(0, "ubx", x_np)

        # Solve
        self._solver.solve()

        # Extract and return trajectory
        return self._extract_trajectory()

    def optimal_control(self, x: jnp.ndarray, state=None) -> tuple:
        """
        Compute optimal control with batch support.

        Note: NMPC is not JIT-compatible since it uses acados.
        For batched inputs, controls are computed sequentially.

        Args:
            x: State(s) (state_dim,) or (batch, state_dim)
            state: Controller state (optional, uses get_init_state() if None)

        Returns:
            Tuple (u, new_state)
        """
        if state is None:
            state = self.get_init_state()
        if x.ndim == 1:
            return self._optimal_control_single(x, state)
        else:
            # Batch processing (sequential, not parallelized)
            u_list = []
            for i in range(x.shape[0]):
                u_i, state = self._optimal_control_single(x[i], state)
                u_list.append(u_i)

            u_batch = jnp.stack(u_list)
            return u_batch, state

    def optimal_control_with_info(self, x: jnp.ndarray, state=None) -> tuple:
        """
        Compute optimal control with diagnostic info.

        Args:
            x: State(s) (state_dim,) or (batch, state_dim)
            state: Controller state (optional, uses get_init_state() if None)

        Returns:
            Tuple (u, new_state, info)
        """
        if state is None:
            state = self.get_init_state()
        if x.ndim == 1:
            return self._optimal_control_single_with_info(x, state)
        else:
            u_list = []
            info_list = []
            for i in range(x.shape[0]):
                u_i, state, info_i = self._optimal_control_single_with_info(x[i], state)
                u_list.append(u_i)
                info_list.append(info_i)

            u_batch = jnp.stack(u_list)
            info_batch = NMPCInfo(
                status=jnp.array([info_i.status for info_i in info_list]),
                cost=jnp.array([info_i.cost for info_i in info_list]),
                x_traj=jnp.stack([info_i.x_traj for info_i in info_list]),
                u_traj=jnp.stack([info_i.u_traj for info_i in info_list]),
            )
            return u_batch, state, info_batch

    def _optimal_control_for_ode(self) -> Callable:
        """
        Create a stateless control function for ODE integration.

        Returns:
            Function x -> u for ODE integration
        """
        init_state = self.get_init_state()
        def control_for_ode(x):
            u, _ = self._optimal_control_single(x, init_state)
            return u
        return control_for_ode

    # ==========================================
    # Trajectory Integration (ZOH)
    # ==========================================

    def get_optimal_trajs_zoh(
        self,
        s0: jnp.ndarray,
        sim_time: float,
        timestep: float,
        method: str = 'tsit5',
        intermediate_steps: int = 10,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute optimal trajectories using Zero-Order Hold (ZOH) control.

        Since NMPC uses acados (not JIT-compatible), this uses a Python loop
        for time stepping with JIT-compiled integration between steps.

        Args:
            s0: Initial states (batch, state_dim) or (state_dim,)
            sim_time: Total simulation time [s]
            timestep: Control timestep [s]
            method: ODE solver method ('tsit5', 'dopri5', 'euler', etc.)
            intermediate_steps: Number of integration substeps per control step

        Returns:
            Tuple (trajectories, actions) where:
            - trajectories: (num_steps, batch, state_dim)
            - actions: (num_steps-1, batch, action_dim)
        """
        import diffrax
        from ..utils.integration import get_solver

        assert self._is_built, "Must call make() before computing trajectories"

        # Handle single state
        if s0.ndim == 1:
            s0 = s0[None, :]
        batch_size = s0.shape[0]

        num_steps = int(sim_time / timestep) + 1
        solver = get_solver(method)
        adjoint = diffrax.RecursiveCheckpointAdjoint()

        # JIT-compiled ODE step with fixed control (ZOH)
        @jax.jit
        def integrate_with_fixed_control(current_state, control):
            def ode_func(t, y, args):
                return self._dynamics.rhs(y, args)

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
        actions_list = []
        init_state = self.get_init_state()

        for i in range(batch_size):
            traj = [s0[i]]
            actions = []
            current_state = s0[i]

            # Python loop over timesteps (cannot use lax.scan due to acados)
            for _ in range(num_steps - 1):
                # Get optimal control from NMPC
                u_opt, _ = self._optimal_control_single(current_state, init_state)

                # Store action
                actions.append(u_opt)

                # Integrate with this control
                next_state = integrate_with_fixed_control(current_state, u_opt)
                traj.append(next_state)
                current_state = next_state

            trajectories.append(jnp.stack(traj, axis=0))
            actions_list.append(jnp.stack(actions, axis=0))

        # Stack: trajectories (num_steps, batch, state_dim), actions (num_steps-1, batch, action_dim)
        trajs = jnp.stack(trajectories, axis=1)
        actions = jnp.stack(actions_list, axis=1)

        return trajs, actions

    # ==========================================
    # Properties
    # ==========================================

    @property
    def is_built(self) -> bool:
        """Check if controller has been built."""
        return self._is_built

    @property
    def solver(self):
        """Get acados solver (None if not built)."""
        return self._solver


class QuadraticNMPCControl(QuadraticCostMixin, NMPCControl):
    """
    NMPC Control with quadratic (LINEAR_LS) cost.

    Uses Q, R matrices for cost: (x - x_ref)^T Q (x - x_ref) + u^T R u

    Cost matrices are stored as Callable functions for consistency with other
    controllers in the codebase. Use assign_cost_matrices() to set the matrices.

    Uses cooperative multiple inheritance:
    - QuadraticCostMixin: Q, R, Q_e, x_ref cost matrices
    - NMPCControl: NMPC solving with acados
    """

    # Cost matrices as Callable for consistency with iLQR and QP controllers (static fields)
    _Q: Optional[Callable] = eqx.field(static=True)
    _R: Optional[Callable] = eqx.field(static=True)
    _Q_e: Optional[Callable] = eqx.field(static=True)
    _x_ref: Optional[Callable] = eqx.field(static=True)

    def __init__(self, **kwargs):
        """
        Initialize QuadraticNMPCControl.

        Args:
            **kwargs: All args passed via cooperative inheritance
                - Q, R, Q_e, x_ref: Handled by QuadraticCostMixin
                - control_low, control_high, etc.: Handled by NMPCControl
                - action_dim, params, dynamics: Handled by BaseControl
        """
        # Initialize via cooperative inheritance (no cost_running/cost_terminal)
        super().__init__(cost_running=None, cost_terminal=None, **kwargs)

    @classmethod
    def create_empty(cls, action_dim: int, params: Optional[dict] = None) -> 'QuadraticNMPCControl':
        return cls(action_dim=action_dim, params=params)

    def _create_updated_instance(self, **kwargs):
        """Create new instance with updated fields."""
        defaults = {
            'action_dim': self._action_dim,
            'params': dict(self._params) if self._params else None,
            'dynamics': self._dynamics,
            'control_low': list(self._control_low) if self._has_control_bounds else None,
            'control_high': list(self._control_high) if self._has_control_bounds else None,
            'state_bounds_idx': list(self._state_bounds_idx) if self._has_state_bounds else None,
            'state_low': list(self._state_low) if self._has_state_bounds else None,
            'state_high': list(self._state_high) if self._has_state_bounds else None,
            'Q': self._Q,
            'R': self._R,
            'Q_e': self._Q_e,
            'x_ref': self._x_ref,
        }
        defaults.update(kwargs)
        return self.__class__(**defaults)

    # assign_cost_matrices, assign_reference from QuadraticCostMixin
    # Note: _get_quadratic_cost_func not used by NMPC (uses _setup_cost with acados LINEAR_LS)

    def _setup_cost(self, ocp, model):
        """Setup LINEAR_LS cost function in OCP."""
        nx = self._dynamics.state_dim
        nu = self._action_dim

        assert self._Q is not None and self._R is not None, \
            "Cost matrices must be assigned. Use assign_cost_matrices()."

        # Use LINEAR_LS cost type
        ocp.cost.cost_type = "LINEAR_LS"
        ocp.cost.cost_type_e = "LINEAR_LS"

        # Get matrices from callables
        Q = np.array(self._Q())
        R = np.array(self._R())
        Q_e = np.array(self._Q_e()) if self._Q_e is not None else Q

        # Reference
        if self._x_ref is not None:
            x_ref = np.array(self._x_ref())
        else:
            x_ref = np.zeros(nx)

        # Output dimension: y = [x; u]
        ny = nx + nu

        # Weight matrices
        ocp.cost.W = np.block([
            [Q, np.zeros((nx, nu))],
            [np.zeros((nu, nx)), R]
        ])
        ocp.cost.W_e = Q_e

        # Output selection matrices: y = Vx @ x + Vu @ u
        # Vx: (ny, nx), Vu: (ny, nu)
        ocp.cost.Vx = np.vstack([np.eye(nx), np.zeros((nu, nx))])
        ocp.cost.Vu = np.vstack([np.zeros((nx, nu)), np.eye(nu)])
        ocp.cost.Vx_e = np.eye(nx)

        # Reference (y = [x; u] for stage, y_e = x for terminal)
        ocp.cost.yref = np.concatenate([x_ref, np.zeros(nu)])
        ocp.cost.yref_e = x_ref

    def make(self, x0: Optional[np.ndarray] = None) -> 'QuadraticNMPCControl':
        """
        Build the NMPC controller.

        Args:
            x0: Initial state for the OCP (required for acados)

        Returns:
            Self with solver built
        """
        # Check quadratic cost is assigned
        assert self._Q is not None and self._R is not None, \
            "Cost matrices must be assigned. Use assign_cost_matrices()."

        # Skip parent's cost check by calling grandparent's assertions
        assert self.has_dynamics, "Dynamics must be assigned before make()"
        assert self._has_control_bounds, "Control bounds must be assigned before make()"

        if x0 is None:
            x0 = np.zeros(self._dynamics.state_dim)

        print("Converting JAX dynamics to CasADi...")
        dynamics_casadi = self._convert_dynamics_to_casadi()
        object.__setattr__(self, '_dynamics_casadi', dynamics_casadi)

        _, AcadosOcp, AcadosOcpSolver, _, _ = _import_acados()

        print("Building acados OCP...")
        ocp = AcadosOcp()

        model = self._create_acados_model(dynamics_casadi)
        ocp.model = model

        self._setup_cost(ocp, model)
        self._setup_constraints(ocp, model, x0)
        self._setup_solver_options(ocp)

        object.__setattr__(self, '_ocp', ocp)

        # Create solver (force regeneration to avoid stale code)
        print("Creating acados solver...")
        solver = AcadosOcpSolver(ocp, build=True, generate=True)
        object.__setattr__(self, '_solver', solver)

        # Create sim solver for shift warm-start if enabled
        if self._params.get('shift_warm_start', False):
            print("Creating acados integrator for shift warm-start...")
            sim_solver = self._create_sim_solver(model)
            object.__setattr__(self, '_sim_solver', sim_solver)

        object.__setattr__(self, '_is_built', True)
        print("NMPC ready!")

        return self

    def update_reference(self, x_ref: np.ndarray, u_ref: Optional[np.ndarray] = None):
        """
        Update the reference trajectory online.

        Args:
            x_ref: Reference state (nx,) or trajectory (N+1, nx)
            u_ref: Reference control (nu,) or trajectory (N, nu), optional
        """
        assert self._is_built, "Must call make() before updating reference"

        nx = self._dynamics.state_dim
        nu = self._action_dim

        # Handle single reference vs trajectory
        if x_ref.ndim == 1:
            x_ref_traj = np.tile(x_ref, (self.N_horizon + 1, 1))
        else:
            x_ref_traj = x_ref

        if u_ref is None:
            u_ref_traj = np.zeros((self.N_horizon, nu))
        elif u_ref.ndim == 1:
            u_ref_traj = np.tile(u_ref, (self.N_horizon, 1))
        else:
            u_ref_traj = u_ref

        # Update references in solver
        for k in range(self.N_horizon):
            yref = np.concatenate([x_ref_traj[k], u_ref_traj[k]])
            self._solver.set(k, "yref", yref)

        # Terminal reference
        self._solver.set(self.N_horizon, "yref", x_ref_traj[self.N_horizon])
