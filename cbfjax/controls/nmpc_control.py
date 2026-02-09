"""
NMPC (Nonlinear Model Predictive Control) using JAX + jax2casadi.

Supports two NLP solver backends selected via params['nlp_solver']:
- 'SQP' or 'SQP_RTI' (default) -> acados backend
- 'IPOPT' -> do-mpc/IPOPT backend

Classes:
    AcadosNMPCBackend: Internal acados backend adapter
    DoMPCNMPCBackend: Internal do-mpc/IPOPT backend adapter
    NMPCControl: Base NMPC controller with EXTERNAL cost (JAX callables)
    QuadraticNMPCControl: NMPC with LINEAR_LS / quadratic cost (Q, R matrices)
"""

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import casadi as ca
from typing import Callable, Optional, Any, Tuple

from ..utils.jax2casadi import convert

from .base_control import BaseControl, QuadraticCostMixin
from .control_types import NMPCInfo
from ..dynamics.base_dynamic import DummyDynamics


# ==========================================
# Backend: acados (SQP / SQP_RTI)
# ==========================================

def _import_acados():
    from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver, AcadosSim, AcadosSimSolver
    return AcadosModel, AcadosOcp, AcadosOcpSolver, AcadosSim, AcadosSimSolver


class AcadosNMPCBackend:
    """Backend adapter for acados NLP solver."""

    def __init__(self):
        self.solver = None
        self._ocp = None
        self._sim_solver = None
        self.N = None
        self.nx = None
        self.nu = None

    def build(self, dynamics_casadi, nx, nu, continuous, params,
              cost_config, control_bounds, state_bounds, barrier_config):
        self.nx = nx
        self.nu = nu
        self.N = int(params['horizon'] / params['time_steps'])

        AcadosModel, AcadosOcp, AcadosOcpSolver, _, _ = _import_acados()

        print("Building acados OCP...")
        ocp = AcadosOcp()

        model = self._create_model(dynamics_casadi, nx, nu, continuous)
        ocp.model = model

        self._setup_cost(ocp, model, cost_config, nx, nu)
        self._setup_constraints(ocp, model, np.zeros(nx), nu, control_bounds, state_bounds, barrier_config, params)
        self._setup_solver_options(ocp, params, continuous)

        self._ocp = ocp

        print("Creating acados solver...")
        self.solver = AcadosOcpSolver(ocp, build=True, generate=True)

        if params.get('shift_warm_start', False):
            print("Creating acados integrator for shift warm-start...")
            self._sim_solver = self._create_sim_solver(model, params['time_steps'])

    def _create_model(self, dynamics_casadi, nx, nu, continuous):
        AcadosModel, _, _, _, _ = _import_acados()

        x = ca.SX.sym('x', nx)
        u = ca.SX.sym('u', nu)

        model = AcadosModel()
        model.x = x
        model.u = u
        model.name = "nmpc_model"

        if continuous:
            xdot = ca.SX.sym('xdot', nx)
            f_expl = dynamics_casadi(x, u)
            model.f_impl_expr = xdot - f_expl
            model.f_expl_expr = f_expl
            model.xdot = xdot
        else:
            model.disc_dyn_expr = dynamics_casadi(x, u)

        return model

    def _setup_cost(self, ocp, model, cost_config, nx, nu):
        if cost_config['type'] == 'external':
            ocp.cost.cost_type = "EXTERNAL"
            ocp.cost.cost_type_e = "EXTERNAL"

            running_ca = cost_config['running_ca']
            terminal_ca = cost_config.get('terminal_ca', None)

            if terminal_ca is None:
                terminal_ca = lambda x: running_ca(x, ca.DM.zeros(nu))

            model.cost_expr_ext_cost = running_ca(model.x, model.u)
            model.cost_expr_ext_cost_e = terminal_ca(model.x)

        elif cost_config['type'] == 'quadratic':
            ocp.cost.cost_type = "LINEAR_LS"
            ocp.cost.cost_type_e = "LINEAR_LS"

            Q = cost_config['Q']
            R = cost_config['R']
            Q_e = cost_config['Q_e']
            x_ref = cost_config['x_ref']

            ocp.cost.W = np.block([
                [Q, np.zeros((nx, nu))],
                [np.zeros((nu, nx)), R]
            ])
            ocp.cost.W_e = Q_e

            ocp.cost.Vx = np.vstack([np.eye(nx), np.zeros((nu, nx))])
            ocp.cost.Vu = np.vstack([np.zeros((nx, nu)), np.eye(nu)])
            ocp.cost.Vx_e = np.eye(nx)

            ocp.cost.yref = np.concatenate([x_ref, np.zeros(nu)])
            ocp.cost.yref_e = x_ref

    def _setup_constraints(self, ocp, model, x0, nu, control_bounds, state_bounds, barrier_config, params):
        if control_bounds is not None:
            ocp.constraints.lbu = control_bounds[0]
            ocp.constraints.ubu = control_bounds[1]
            ocp.constraints.idxbu = np.arange(nu)

        if state_bounds is not None:
            idx, low, high = state_bounds
            ocp.constraints.lbx = low
            ocp.constraints.ubx = high
            ocp.constraints.idxbx = idx

        ocp.constraints.x0 = x0

        if barrier_config is not None:
            self._setup_barrier_constraints(ocp, model, barrier_config, params)

    def _setup_barrier_constraints(self, ocp, model, barrier_config, params):
        path_ca_funcs = barrier_config.get('path_ca_funcs', [])
        if path_ca_funcs:
            nh = len(path_ca_funcs)
            h_exprs = []
            for barrier_ca in path_ca_funcs:
                h_exprs.append(barrier_ca(model.x))

            model.con_h_expr = ca.vertcat(*h_exprs)
            ocp.constraints.lh = np.zeros(nh)
            ocp.constraints.uh = 1e9 * np.ones(nh)

            if barrier_config.get('slacked', False):
                ocp.constraints.idxsh = np.arange(nh)
                slack_l1 = barrier_config.get('slack_gain_l1', 0.0)
                slack_l2 = barrier_config.get('slack_gain_l2', 1000.0)
                ocp.cost.zl = slack_l1 * np.ones(nh)
                ocp.cost.zu = np.zeros(nh)
                ocp.cost.Zl = slack_l2 * np.ones(nh)
                ocp.cost.Zu = np.zeros(nh)
                print(f"Path barrier slacked with L1={slack_l1}, L2={slack_l2}")

        terminal_ca_funcs = barrier_config.get('terminal_ca_funcs', [])
        if terminal_ca_funcs:
            nh_e = len(terminal_ca_funcs)
            h_e_exprs = []
            for barrier_ca in terminal_ca_funcs:
                h_e_exprs.append(barrier_ca(model.x))

            model.con_h_expr_e = ca.vertcat(*h_e_exprs)
            ocp.constraints.lh_e = np.zeros(nh_e)
            ocp.constraints.uh_e = 1e9 * np.ones(nh_e)

            if barrier_config.get('slacked_e', False):
                ocp.constraints.idxsh_e = np.arange(nh_e)
                slack_l1_e = barrier_config.get('slack_gain_l1_e', 0.0)
                slack_l2_e = barrier_config.get('slack_gain_l2_e', 1000.0)
                ocp.cost.zl_e = slack_l1_e * np.ones(nh_e)
                ocp.cost.zu_e = np.zeros(nh_e)
                ocp.cost.Zl_e = slack_l2_e * np.ones(nh_e)
                ocp.cost.Zu_e = np.zeros(nh_e)
                print(f"Terminal barrier slacked with L1={slack_l1_e}, L2={slack_l2_e}")

    def _setup_solver_options(self, ocp, params, continuous):
        ocp.solver_options.N_horizon = self.N
        ocp.solver_options.tf = params['horizon']

        ocp.solver_options.qp_solver = params['qp_solver']
        ocp.solver_options.hessian_approx = params['hessian_approx']
        ocp.solver_options.sim_method_num_stages = params['sim_method_num_stages']
        ocp.solver_options.nlp_solver_type = params['nlp_solver']
        ocp.solver_options.nlp_solver_max_iter = params['nlp_solver_max_iter']
        ocp.solver_options.qp_solver_iter_max = params['qp_solver_iter_max']
        ocp.solver_options.tol = params['tol']

        if continuous:
            ocp.solver_options.integrator_type = params['integrator_type']
        else:
            ocp.solver_options.integrator_type = 'DISCRETE'

        ocp.code_export_directory = 'c_generated_code'

    def _create_sim_solver(self, model, dt):
        _, _, _, AcadosSim, AcadosSimSolver = _import_acados()
        sim = AcadosSim()
        sim.model = model
        sim.solver_options.T = dt
        sim.solver_options.integrator_type = 'ERK'
        sim.solver_options.num_stages = 1
        sim.solver_options.num_steps = 1
        return AcadosSimSolver(sim)

    def solve(self, x_np):
        self.solver.set(0, "lbx", x_np)
        self.solver.set(0, "ubx", x_np)
        self.solver.solve()
        return self.solver.get(0, "u")

    def solve_with_info(self, x_np):
        self.solver.set(0, "lbx", x_np)
        self.solver.set(0, "ubx", x_np)
        status = self.solver.solve()
        u_opt = self.solver.get(0, "u")
        cost = self.solver.get_cost()
        x_traj, u_traj = self.extract_trajectory()
        return u_opt, status, cost, x_traj, u_traj

    def extract_trajectory(self):
        N, nx, nu = self.N, self.nx, self.nu
        x_traj = np.zeros((N + 1, nx))
        u_traj = np.zeros((N, nu))
        for k in range(N + 1):
            x_traj[k] = self.solver.get(k, "x")
        for k in range(N):
            u_traj[k] = self.solver.get(k, "u")
        return x_traj, u_traj

    def get_predicted_trajectory(self, x_np):
        self.solver.set(0, "lbx", x_np)
        self.solver.set(0, "ubx", x_np)
        self.solver.solve()
        return self.extract_trajectory()

    def set_init_guess(self, x_traj, u_traj):
        N = self.N
        if x_traj is None:
            x0 = self.solver.get(0, "x")
            x_traj = np.tile(x0, (N + 1, 1))
        x_traj = np.asarray(x_traj)
        for k in range(N + 1):
            self.solver.set(k, "x", x_traj[k])

        if u_traj is None:
            u_traj = np.zeros((N, self.nu))
        u_traj = np.asarray(u_traj)
        for k in range(N):
            self.solver.set(k, "u", u_traj[k])

    def set_init_guess_linear(self, x0, x_target):
        x0 = np.asarray(x0)
        x_target = np.asarray(x_target)
        alphas = np.linspace(0.0, 1.0, self.N + 1)
        x_traj = x0[None, :] + alphas[:, None] * (x_target - x0)[None, :]
        self.set_init_guess(x_traj=x_traj, u_traj=None)

    def post_solve(self):
        if self._sim_solver is None:
            return
        self._shift_warm_start()

    def _shift_warm_start(self):
        N = self.N
        x_traj = np.array([self.solver.get(k, "x") for k in range(N + 1)])
        u_traj = np.array([self.solver.get(k, "u") for k in range(N)])

        u_shifted = np.empty_like(u_traj)
        u_shifted[:-1] = u_traj[1:]
        u_shifted[-1] = u_traj[-1]

        x_shifted = np.empty_like(x_traj)
        x_shifted[0] = x_traj[1]
        for k in range(N):
            x_shifted[k + 1] = self._sim_solver.simulate(x=x_shifted[k], u=u_shifted[k])

        for k in range(N + 1):
            self.solver.set(k, "x", x_shifted[k])
        for k in range(N):
            self.solver.set(k, "u", u_shifted[k])

    def update_reference(self, x_ref, u_ref):
        nx, nu, N = self.nx, self.nu, self.N

        if x_ref.ndim == 1:
            x_ref_traj = np.tile(x_ref, (N + 1, 1))
        else:
            x_ref_traj = x_ref

        if u_ref is None:
            u_ref_traj = np.zeros((N, nu))
        elif u_ref.ndim == 1:
            u_ref_traj = np.tile(u_ref, (N, 1))
        else:
            u_ref_traj = u_ref

        for k in range(N):
            yref = np.concatenate([x_ref_traj[k], u_ref_traj[k]])
            self.solver.set(k, "yref", yref)
        self.solver.set(N, "yref", x_ref_traj[N])


# ==========================================
# Backend: do-mpc / IPOPT
# ==========================================

class DoMPCNMPCBackend:
    """Backend adapter for do-mpc / IPOPT NLP solver."""

    def __init__(self):
        self.solver = None
        self.mpc = None
        self._dompc_model = None
        self.N = None
        self.nx = None
        self.nu = None
        self._current_x_ref = None

    def build(self, dynamics_casadi, nx, nu, continuous, params,
              cost_config, control_bounds, state_bounds, barrier_config):
        import do_mpc

        self.nx = nx
        self.nu = nu
        self.N = int(params['horizon'] / params['time_steps'])

        has_tvp = (cost_config['type'] == 'quadratic')
        has_terminal_barrier = (barrier_config is not None and
                                len(barrier_config.get('terminal_ca_funcs', [])) > 0)

        print("Building do-mpc model...")
        model = self._create_model(dynamics_casadi, nx, nu, continuous, has_tvp)
        self._dompc_model = model

        print("Setting up do-mpc MPC...")
        mpc = do_mpc.controller.MPC(model)

        self._setup_solver_options(mpc, params, continuous)
        self._setup_cost(mpc, model, cost_config, nx, nu)
        self._setup_constraints(mpc, model, nu, control_bounds, state_bounds, barrier_config)

        if has_tvp:
            self._setup_tvp(mpc, cost_config, nx)

        if has_terminal_barrier:
            mpc.prepare_nlp()
            self._setup_terminal_constraints(mpc, barrier_config, nx, params)
            mpc.create_nlp()
        else:
            mpc.setup()

        mpc.x0 = np.zeros((nx, 1))
        mpc.set_initial_guess()

        self.mpc = mpc
        self.solver = mpc

    def _create_model(self, dynamics_ca, nx, nu, continuous, has_tvp):
        import do_mpc

        model_type = 'continuous' if continuous else 'discrete'
        model = do_mpc.model.Model(model_type)

        x = model.set_variable(var_type='_x', var_name='x', shape=(nx, 1))
        u = model.set_variable(var_type='_u', var_name='u', shape=(nu, 1))

        if has_tvp:
            model.set_variable(var_type='_tvp', var_name='x_ref', shape=(nx, 1))

        x_flat = ca.vertcat(*[x[i] for i in range(nx)])
        u_flat = ca.vertcat(*[u[i] for i in range(nu)])
        result = dynamics_ca(x_flat, u_flat)
        result_col = ca.reshape(result, nx, 1)
        model.set_rhs('x', result_col)

        model.setup()
        return model

    def _setup_cost(self, mpc, model, cost_config, nx, nu):
        if cost_config['type'] == 'external':
            running_ca = cost_config['running_ca']
            terminal_ca = cost_config.get('terminal_ca', None)

            if terminal_ca is None:
                terminal_ca = lambda x: running_ca(x, ca.DM.zeros(nu))

            x_sym = ca.vertcat(*[model.x['x', i] for i in range(nx)])
            u_sym = ca.vertcat(*[model.u['u', i] for i in range(nu)])

            lterm = running_ca(x_sym, u_sym)
            mterm = terminal_ca(x_sym)
            mpc.set_objective(mterm=mterm, lterm=lterm)

        elif cost_config['type'] == 'quadratic':
            Q = cost_config['Q']
            R = cost_config['R']
            Q_e = cost_config['Q_e']

            x_sym = model.x['x']
            u_sym = model.u['u']
            x_ref_sym = model.tvp['x_ref']

            x_err = x_sym - x_ref_sym

            lterm = ca.mtimes([x_err.T, Q, x_err]) + ca.mtimes([u_sym.T, R, u_sym])
            mterm = ca.mtimes([x_err.T, Q_e, x_err])
            mpc.set_objective(mterm=mterm, lterm=lterm)

    def _setup_constraints(self, mpc, model, nu, control_bounds, state_bounds, barrier_config):
        if control_bounds is not None:
            low, high = control_bounds
            for i in range(nu):
                mpc.bounds['lower', '_u', 'u', i] = low[i]
                mpc.bounds['upper', '_u', 'u', i] = high[i]

        if state_bounds is not None:
            idx, low, high = state_bounds
            for j, state_idx in enumerate(idx):
                mpc.bounds['lower', '_x', 'x', int(state_idx)] = low[j]
                mpc.bounds['upper', '_x', 'x', int(state_idx)] = high[j]

        if barrier_config is not None:
            self._setup_path_barriers(mpc, model, barrier_config)

    def _setup_path_barriers(self, mpc, model, barrier_config):
        nx = self.nx
        path_ca_funcs = barrier_config.get('path_ca_funcs', [])
        if not path_ca_funcs:
            return

        slacked = barrier_config.get('slacked', False)
        slack_l2 = barrier_config.get('slack_gain_l2', 1000.0)

        x_sym = ca.vertcat(*[model.x['x', i] for i in range(nx)])

        print(f"Converting {len(path_ca_funcs)} path barrier(s) to do-mpc constraints...")
        for i, barrier_ca in enumerate(path_ca_funcs):
            h_expr = -barrier_ca(x_sym)

            if slacked:
                mpc.set_nl_cons(
                    f'barrier_{i}', h_expr,
                    ub=0.0,
                    soft_constraint=True,
                    penalty_term_cons=slack_l2,
                )
                print(f"  Path barrier {i}: soft constraint (L2={slack_l2})")
            else:
                mpc.set_nl_cons(f'barrier_{i}', h_expr, ub=0.0)
                print(f"  Path barrier {i}: hard constraint")

    def _setup_terminal_constraints(self, mpc, barrier_config, nx, params):
        terminal_ca_funcs = barrier_config.get('terminal_ca_funcs', [])
        if not terminal_ca_funcs:
            return

        n_horizon = self.N
        slacked_e = barrier_config.get('slacked_e', False)
        slack_l2_e = barrier_config.get('slack_gain_l2_e', 1000.0)

        x_terminal = mpc.opt_x['_x', n_horizon, 0, -1, 'x']
        x_terminal_flat = ca.vertcat(*[x_terminal[i] for i in range(nx)])

        for i, barrier_ca in enumerate(terminal_ca_funcs):
            h_val = barrier_ca(x_terminal_flat)
            cons_expr = -h_val

            mpc.nlp_cons.append(cons_expr)
            mpc.nlp_cons_lb.append(np.full(cons_expr.shape, -np.inf))
            mpc.nlp_cons_ub.append(np.zeros(cons_expr.shape))

            if slacked_e:
                print(f"  Terminal barrier {i}: soft constraint (L2={slack_l2_e}) via NLP")
            else:
                print(f"  Terminal barrier {i}: hard constraint via NLP")

    def _setup_solver_options(self, mpc, params, continuous):
        N = self.N
        dt = params['time_steps']

        ipopt_opts = {
            'ipopt.max_iter': params.get('nlp_solver_max_iter', 200),
            'ipopt.tol': params.get('tol', 1e-4),
            'ipopt.print_level': 0,
            'print_time': 0,
        }
        user_opts = params.get('nlpsol_opts', {})
        ipopt_opts.update(user_opts)

        mpc_params = dict(
            n_horizon=N,
            t_step=dt,
            n_robust=0,
            store_full_solution=True,
            nlpsol_opts={'ipopt.max_iter': ipopt_opts['ipopt.max_iter'],
                         'ipopt.tol': ipopt_opts['ipopt.tol'],
                         'ipopt.print_level': ipopt_opts['ipopt.print_level'],
                         'print_time': ipopt_opts['print_time']},
        )

        if continuous:
            mpc_params['collocation_deg'] = params.get('collocation_deg', 3)

        mpc.set_param(**mpc_params)

    def _setup_tvp(self, mpc, cost_config, nx):
        N = self.N

        x_ref = cost_config.get('x_ref', np.zeros(nx))
        self._current_x_ref = np.asarray(x_ref).reshape(-1, 1)

        tvp_template = mpc.get_tvp_template()
        backend_self = self

        def tvp_fun(t_now):
            for k in range(N + 1):
                tvp_template['_tvp', k, 'x_ref'] = backend_self._current_x_ref
            return tvp_template

        mpc.set_tvp_fun(tvp_fun)

    def solve(self, x_np):
        x_col = x_np.reshape(-1, 1)
        u_opt = self.mpc.make_step(x_col)
        return u_opt.flatten()

    def solve_with_info(self, x_np):
        x_col = x_np.reshape(-1, 1)
        u_opt = self.mpc.make_step(x_col)
        u_flat = u_opt.flatten()

        x_traj, u_traj = self.extract_trajectory()

        try:
            cost = float(self.mpc.data['_aux', -1, 'nlp_cost'])
        except Exception:
            cost = 0.0

        return u_flat, 0, cost, x_traj, u_traj

    def extract_trajectory(self):
        N, nx, nu = self.N, self.nx, self.nu
        x_traj = np.zeros((N + 1, nx))
        u_traj = np.zeros((N, nu))

        for k in range(N + 1):
            x_pred = self.mpc.data.prediction(('_x', 'x'), t_ind=-1)
            if x_pred is not None and x_pred.shape[0] >= nx:
                x_traj[k] = x_pred[:nx, k, 0] if k < x_pred.shape[1] else x_pred[:nx, -1, 0]

        for k in range(N):
            u_pred = self.mpc.data.prediction(('_u', 'u'), t_ind=-1)
            if u_pred is not None and u_pred.shape[0] >= nu:
                u_traj[k] = u_pred[:nu, k, 0] if k < u_pred.shape[1] else u_pred[:nu, -1, 0]

        return x_traj, u_traj

    def get_predicted_trajectory(self, x_np):
        x_col = x_np.reshape(-1, 1)
        self.mpc.make_step(x_col)
        return self.extract_trajectory()

    def set_init_guess(self, x_traj, u_traj):
        if x_traj is not None:
            self.mpc.x0 = np.asarray(x_traj[0]).reshape(-1, 1)
        self.mpc.set_initial_guess()

    def set_init_guess_linear(self, x0, x_target):
        self.mpc.x0 = np.asarray(x0).reshape(-1, 1)
        self.mpc.set_initial_guess()

    def post_solve(self):
        pass

    def update_reference(self, x_ref, u_ref):
        if x_ref.ndim > 1:
            x_ref = x_ref[0]
        self._current_x_ref = np.asarray(x_ref).reshape(-1, 1)


# ==========================================
# User-facing controller: NMPCControl
# ==========================================

class NMPCControl(BaseControl):
    """
    Nonlinear Model Predictive Control with backend dispatch.

    Dynamics and cost are defined in JAX, converted to CasADi via jax2casadi,
    and the resulting OCP is solved using the selected backend.

    Backend selection via params['nlp_solver']:
        'SQP' or 'SQP_RTI' -> acados
        'IPOPT'             -> do-mpc/IPOPT

    All configuration is stored in params dict:
        params = {
            # Horizon settings
            'horizon': 2.0,          # Total prediction horizon time [s]
            'time_steps': 0.04,      # Timestep [s]

            # Backend selection
            'nlp_solver': 'SQP',     # 'SQP'/'SQP_RTI' -> acados, 'IPOPT' -> do-mpc

            # Solver options (shared)
            'nlp_solver_max_iter': 200,
            'tol': 1e-4,

            # Acados-specific (ignored by do-mpc):
            'qp_solver': 'PARTIAL_CONDENSING_HPIPM',
            'hessian_approx': 'GAUSS_NEWTON',
            'integrator_type': 'ERK',
            'sim_method_num_stages': 4,
            'qp_solver_iter_max': 100,

            # Do-mpc-specific (ignored by acados):
            'collocation_deg': 3,
            'nlpsol_opts': {},
        }
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

    # Build state (non-static since backend is mutable)
    _is_built: bool
    _backend: Any
    _dynamics_casadi: Any

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
        # Default params for NMPC (unified)
        params = kwargs.get('params', None)
        default_params = {
            # Horizon settings
            'horizon': 2.0,
            'time_steps': 0.04,
            # Backend selection
            'nlp_solver': 'SQP',
            # Shared solver options
            'nlp_solver_max_iter': 200,
            'tol': 1e-4,
            # Acados-specific
            'qp_solver': 'PARTIAL_CONDENSING_HPIPM',
            'hessian_approx': 'GAUSS_NEWTON',
            'integrator_type': 'ERK',
            'sim_method_num_stages': 4,
            'qp_solver_iter_max': 100,
            # Do-mpc-specific
            'collocation_deg': 3,
            'nlpsol_opts': {},
        }
        if params is not None:
            default_params.update(params)
        # Backward compat: map old 'nlp_solver_type' to 'nlp_solver'
        if 'nlp_solver_type' in default_params and 'nlp_solver' not in (params or {}):
            default_params['nlp_solver'] = default_params.pop('nlp_solver_type')
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
        self._backend = None
        self._dynamics_casadi = None

    @classmethod
    def create_empty(cls, action_dim: int, params: Optional[dict] = None) -> 'NMPCControl':
        return cls(action_dim=action_dim, params=params)

    def _create_updated_instance(self, **kwargs):
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

    def assign_dynamics(self, dynamics, continuous=True) -> 'NMPCControl':
        params = dict(self._params)
        params['continuous'] = continuous
        return self._create_updated_instance(dynamics=dynamics, params=params)

    def assign_control_bounds(self, low: list, high: list) -> 'NMPCControl':
        assert len(low) == len(high), 'low and high should have the same length'
        assert len(low) == self._action_dim, 'bounds length should match action dimension'
        return self._create_updated_instance(control_low=low, control_high=high)

    def assign_state_bounds(self, idx: list, low: list, high: list) -> 'NMPCControl':
        assert len(idx) == len(low) == len(high), 'idx, low, high should have same length'
        return self._create_updated_instance(
            state_bounds_idx=idx, state_low=low, state_high=high
        )

    def assign_cost_running(self, cost_func: Callable) -> 'NMPCControl':
        return self._create_updated_instance(cost_running=cost_func)

    def assign_cost_terminal(self, cost_func: Callable) -> 'NMPCControl':
        return self._create_updated_instance(cost_terminal=cost_func)

    # ==========================================
    # Properties
    # ==========================================

    @property
    def horizon(self) -> float:
        return self._params['horizon']

    @property
    def time_steps(self) -> float:
        return self._params['time_steps']

    @property
    def N_horizon(self) -> int:
        return int(self.horizon / self.time_steps)

    @property
    def is_built(self) -> bool:
        return self._is_built

    @property
    def solver(self):
        """Get underlying solver (acados solver or do-mpc MPC object)."""
        if self._backend is not None:
            return self._backend.solver
        return None

    # ==========================================
    # Backend Selection
    # ==========================================

    def _is_ipopt(self) -> bool:
        """Check if IPOPT (do-mpc) backend is selected."""
        return self._params.get('nlp_solver', 'SQP').upper() == 'IPOPT'

    def _create_backend(self):
        """Create the appropriate backend based on nlp_solver param."""
        if self._is_ipopt():
            return DoMPCNMPCBackend()
        else:
            return AcadosNMPCBackend()

    # ==========================================
    # JAX -> CasADi Conversion
    # ==========================================

    def _convert_dynamics_to_casadi(self) -> ca.Function:
        """Convert JAX dynamics to CasADi function."""
        nx = self._dynamics.state_dim
        nu = self._action_dim

        if self._params.get('continuous', True):
            jax_func = self._dynamics.rhs
            name = 'dynamics'
        else:
            jax_func = self._dynamics.discrete_rhs
            name = 'dynamics_discrete'

        dynamics_ca = convert(
            jax_func,
            [('x', (nx,)), ('u', (nu,))],
            name=name,
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

        cost_ca = convert(
                self._cost_terminal,
                [('x', (nx,))],
                name='cost_terminal',
                validate=True,
                tolerance=1e-3
            )
        return cost_ca

    # ==========================================
    # Hook Methods (overridden by subclasses)
    # ==========================================

    def _validate_for_make(self):
        """Validation checks before make(). Override in subclasses."""
        assert self.has_dynamics, "Dynamics must be assigned before make()"
        assert self._has_control_bounds, "Control bounds must be assigned before make()"
        assert self._cost_running is not None, \
            "Running cost must be assigned. Use assign_cost_running()."

        if not self._params.get('continuous', True):
            assert self._dynamics._dt is not None, \
                "Discrete dynamics require discretization_dt in dynamics params."
            assert abs(self._dynamics._dt - self.time_steps) < 1e-10, \
                f"Discrete dynamics dt ({self._dynamics._dt}) must match " \
                f"controller time_steps ({self.time_steps})."

    def _prepare_cost_config(self) -> dict:
        """Prepare cost config dict for backend. Override in subclasses."""
        running_ca = self._convert_cost_running_to_casadi()
        terminal_ca = self._convert_cost_terminal_to_casadi() if self._cost_terminal is not None else None
        return {
            'type': 'external',
            'running_ca': running_ca,
            'terminal_ca': terminal_ca,
        }

    def _prepare_barrier_config(self) -> Optional[dict]:
        """Prepare barrier config dict for backend. Override in safe subclasses."""
        return None

    # ==========================================
    # Build (single unified make)
    # ==========================================

    def make(self) -> 'NMPCControl':
        """
        Build the NMPC controller.

        Converts JAX functions to CasADi, creates the OCP, and initializes the solver.

        Returns:
            Self with solver built
        """
        self._validate_for_make()

        nx = self._dynamics.state_dim
        nu = self._action_dim
        continuous = self._params.get('continuous', True)

        print("Converting JAX dynamics to CasADi...")
        dynamics_casadi = self._convert_dynamics_to_casadi()
        object.__setattr__(self, '_dynamics_casadi', dynamics_casadi)

        cost_config = self._prepare_cost_config()
        barrier_config = self._prepare_barrier_config()

        control_bounds = None
        if self._has_control_bounds:
            control_bounds = (np.array(self._control_low), np.array(self._control_high))

        state_bounds = None
        if self._has_state_bounds:
            state_bounds = (np.array(self._state_bounds_idx),
                            np.array(self._state_low),
                            np.array(self._state_high))

        backend = self._create_backend()
        backend.build(
            dynamics_casadi=dynamics_casadi,
            nx=nx,
            nu=nu,
            continuous=continuous,
            params=dict(self._params),
            cost_config=cost_config,
            control_bounds=control_bounds,
            state_bounds=state_bounds,
            barrier_config=barrier_config,
        )

        object.__setattr__(self, '_backend', backend)
        object.__setattr__(self, '_is_built', True)
        print("NMPC ready!")

        return self

    # ==========================================
    # Initial Guess / Warm-Starting
    # ==========================================

    def set_init_guess(
        self,
        x_traj: Optional[np.ndarray] = None,
        u_traj: Optional[np.ndarray] = None,
    ) -> None:
        """Set initial guess for the solver (warm-starting)."""
        assert self._is_built, "Must call make() before setting initial guess"
        self._backend.set_init_guess(x_traj, u_traj)

    def set_init_guess_linear(
        self,
        x0: np.ndarray,
        x_target: np.ndarray,
    ) -> None:
        """Set a linear interpolation initial guess from x0 to x_target."""
        assert self._is_built, "Must call make() before setting initial guess"
        self._backend.set_init_guess_linear(x0, x_target)

    # ==========================================
    # Control Methods
    # ==========================================

    def _optimal_control_single(self, x: jnp.ndarray, state=None) -> Tuple[jnp.ndarray, Any]:
        assert self._is_built, "Must call make() before computing control"

        x_np = np.array(x)
        u_opt = self._backend.solve(x_np)
        self._backend.post_solve()

        return jnp.array(u_opt), state

    def _optimal_control_single_with_info(self, x: jnp.ndarray, state=None) -> Tuple[jnp.ndarray, Any, dict]:
        assert self._is_built, "Must call make() before computing control"

        x_np = np.array(x)
        u_opt, status, cost, x_traj, u_traj = self._backend.solve_with_info(x_np)
        self._backend.post_solve()

        info = NMPCInfo(
            status=jnp.array(status),
            cost=jnp.array(float(cost)),
            x_traj=x_traj,
            u_traj=u_traj,
        )

        return jnp.array(u_opt), state, info

    def _extract_trajectory(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._backend.extract_trajectory()

    def get_predicted_trajectory(self, x: jnp.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        assert self._is_built, "Must call make() before getting trajectory"
        x_np = np.array(x)
        return self._backend.get_predicted_trajectory(x_np)

    def optimal_control(self, x: jnp.ndarray, state=None) -> tuple:
        if state is None:
            state = self.get_init_state()
        if x.ndim == 1:
            return self._optimal_control_single(x, state)
        else:
            u_list = []
            for i in range(x.shape[0]):
                u_i, state = self._optimal_control_single(x[i], state)
                u_list.append(u_i)
            u_batch = jnp.stack(u_list)
            return u_batch, state

    def optimal_control_with_info(self, x: jnp.ndarray, state=None) -> tuple:
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

        Since NMPC uses external solvers (not JIT-compatible), this uses a Python loop
        for time stepping with JIT-compiled integration between steps.
        """
        import diffrax
        from ..utils.integration import get_solver

        assert self._is_built, "Must call make() before computing trajectories"

        if s0.ndim == 1:
            s0 = s0[None, :]
        batch_size = s0.shape[0]

        num_steps = int(sim_time / timestep) + 1
        solver = get_solver(method)
        adjoint = diffrax.RecursiveCheckpointAdjoint()

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

        trajectories = []
        actions_list = []
        init_state = self.get_init_state()

        for i in range(batch_size):
            traj = [s0[i]]
            actions = []
            current_state = s0[i]

            for _ in range(num_steps - 1):
                u_opt, _ = self._optimal_control_single(current_state, init_state)
                actions.append(u_opt)
                next_state = integrate_with_fixed_control(current_state, u_opt)
                traj.append(next_state)
                current_state = next_state

            trajectories.append(jnp.stack(traj, axis=0))
            actions_list.append(jnp.stack(actions, axis=0))

        trajs = jnp.stack(trajectories, axis=1)
        actions = jnp.stack(actions_list, axis=1)

        return trajs, actions


# ==========================================
# User-facing controller: QuadraticNMPCControl
# ==========================================

class QuadraticNMPCControl(QuadraticCostMixin, NMPCControl):
    """
    NMPC Control with quadratic (LINEAR_LS) cost.

    Uses Q, R matrices for cost: (x - x_ref)^T Q (x - x_ref) + u^T R u

    Supports both acados (LINEAR_LS) and do-mpc (TVP quadratic) backends.
    """

    _Q: Optional[Callable] = eqx.field(static=True)
    _R: Optional[Callable] = eqx.field(static=True)
    _Q_e: Optional[Callable] = eqx.field(static=True)
    _x_ref: Optional[Callable] = eqx.field(static=True)

    def __init__(self, **kwargs):
        super().__init__(cost_running=None, cost_terminal=None, **kwargs)

    @classmethod
    def create_empty(cls, action_dim: int, params: Optional[dict] = None) -> 'QuadraticNMPCControl':
        return cls(action_dim=action_dim, params=params)

    def _create_updated_instance(self, **kwargs):
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

    def _validate_for_make(self):
        assert self._Q is not None and self._R is not None, \
            "Cost matrices must be assigned. Use assign_cost_matrices()."
        assert self.has_dynamics, "Dynamics must be assigned before make()"
        assert self._has_control_bounds, "Control bounds must be assigned before make()"
        if not self._params.get('continuous', True):
            assert self._dynamics._dt is not None, \
                "Discrete dynamics require discretization_dt in dynamics params."
            assert abs(self._dynamics._dt - self.time_steps) < 1e-10, \
                f"Discrete dynamics dt ({self._dynamics._dt}) must match " \
                f"controller time_steps ({self.time_steps})."

    def _prepare_cost_config(self) -> dict:
        nx = self._dynamics.state_dim

        Q = np.array(self._Q())
        R = np.array(self._R())
        Q_e = np.array(self._Q_e()) if self._Q_e is not None else Q

        if self._x_ref is not None:
            x_ref = np.array(self._x_ref())
        else:
            x_ref = np.zeros(nx)

        return {
            'type': 'quadratic',
            'Q': Q,
            'R': R,
            'Q_e': Q_e,
            'x_ref': x_ref,
        }

    def update_reference(self, x_ref: np.ndarray, u_ref: Optional[np.ndarray] = None):
        """Update the reference trajectory online."""
        assert self._is_built, "Must call make() before updating reference"
        self._backend.update_reference(x_ref, u_ref)
