"""
Parametric Flow Barrier Simulation for JAX - Unicycle Example

Demonstrates:
- FlowBarrier for parametric control barrier functions
- ParametricFlowSafeControl for safe control synthesis
- Time-shift parameter evolution
- Danskin approach for state barriers
"""

import os
import datetime
import time
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation as animation
from immutabledict import immutabledict

# CBFJAX configuration
import cbfjax
cbfjax.configure_jax(platform="cpu", enable_x64=True)

from cbfjax.dynamics import UnicycleDynamics
from map_config import map_config

# Get script directory for saving figures
script_dir = os.path.dirname(os.path.abspath(__file__))

# Configure matplotlib
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['axes.labelsize'] = 18
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16
mpl.rcParams['legend.fontsize'] = 16

# ============================================
# Configuration
# ============================================

# Control bounds
control_low = (-2.0, -1.0)   # (min linear velocity, min angular velocity)
control_high = (2.0, 1.0)    # (max linear velocity, max angular velocity)

# FlowBarrier configuration
cfg = immutabledict({
    'softmin_rho': 20,
    'traj_softmin_rho': 30,
    'action_softmin_rho': 50,
    'state_barrier_rel_deg': 1,
    'horizon': 4.0,
    'time_steps': 0.05,
    'integration_method': 'euler',
    'control_param_method': 'ZOH',
    'control_param_num': 80,
    'control_low': control_low,
    'control_high': control_high,
    'compose_action_barriers': True,   # Use composition (softmin) for action barriers
    'compose_state_barriers': True,    # Use composition (softmin) for state barriers
    'danskin_state_barriers': True,    # Use Danskin approach (global minima only)
})

# Map configuration
map_cfg = immutabledict({
    'softmin_rho': 20,
    'velocity_alpha': (),
    'pos_barrier_rel_deg': 1,
    'vel_barrier_rel_deg': 1
})

# Cost matrices for safety filter
cost_matrices = {
    'R': 1e5,            # Control effort weight
    'Lambda': 30,        # Parameter update weight
    'Mu': 1e-1,          # Time shift weight
    'lambda_linear': 1000.0  # Linear penalty
}

# Alpha functions for CBF constraints
alpha_gains = {
    'trajectory': 12.0,
    'backup': 1.0,
    'action': 10.0,
    'time_shift': 0.1
}

# Goal position
goal_pos = jnp.array([[8.0, 8.0]])

# Initial condition
x0 = jnp.array([-8.0, -8.0, 0.0, 0.0])

# Simulation parameters
timestep = 0.001
sim_time = 20.0
make_animation = False  # MP4 render over all frames; enable when needed

# ============================================
# Setup Dynamics
# ============================================

print("Setting up dynamics...")


# Nominal dynamics (used for barrier/controller - no disturbance)
dynamics = UnicycleDynamics()

nx = dynamics.state_dim   # 4: [q_x, q_y, v, theta]
nu = dynamics.action_dim  # 2: [acceleration, angular_velocity]

print(f"  State dim: {nx}, Action dim: {nu}")

# ============================================
# Setup State Barriers
# ============================================

print("Setting up state barriers...")

state_parts = cbfjax.from_config({
    'dynamics': dynamics,
    'barriers': {
        'map':   {'type': 'map', **map_config, 'cfg': map_cfg},
        'state': {'type': 'soft_composition', 'barriers': ['map'], 'cfg': map_cfg},
    },
})
map_ = state_parts.barriers['map']
state_barrier = state_parts.barriers['state']

print(f"  State barrier: {len(map_.pos_barriers) + len(map_.vel_barriers)} "
      "obstacle/boundary barriers")

# ============================================
# Setup Backup Barrier
# ============================================

print("Setting up backup barrier...")

def backup_barrier_functional(x):
    """Backup barrier: state_barrier(x) - 0.5 * v^2 / u_max"""
    state_h = state_barrier.hocbf(x)
    velocity_term = 0.5 * jnp.pow(x[2], 2) / control_high[0]
    return state_h - velocity_term

print("  Backup barrier configured (as a 'func' entry in the main config)")

# ============================================
# Setup Cost Functional
# ============================================

print("Setting up cost functional...")

def cost_functional(trajectory):
    """
    Cost = sum_t (x_t - x_goal)^T Q_t (x_t - x_goal)
    with Gaussian proximity weighting near the goal.
    """
    time_steps_traj, state_dim = trajectory.shape

    # Cost weights - running and terminal
    Q_running = jnp.array([1.0, 1.0, 0.0, 0.0])
    Q_terminal = jnp.array([1.0, 1.0, 0.0, 0.0])

    Q_weights = jnp.tile(Q_running, (time_steps_traj - 1, 1))
    Q_weights = jnp.concatenate([Q_weights, Q_terminal.reshape(1, -1)], axis=0)

    # Goal trajectory
    goal_state = jnp.zeros(state_dim)
    goal_state = goal_state.at[:2].set(goal_pos[0])
    goal_traj = jnp.tile(goal_state, (time_steps_traj, 1))

    # Compute errors
    errors = trajectory - goal_traj

    # Gaussian proximity weighting
    pos_errors = errors[:, :2]
    squared_distances = jnp.sum(pos_errors ** 2, axis=1)
    sigma_squared = 2.0
    max_scaling = 40.0
    gaussian_weights = jnp.exp(-squared_distances / sigma_squared)

    # Apply Gaussian scaling
    scaling_factor = 1.0 + max_scaling * gaussian_weights
    Q_weights_scaled = Q_weights * jax.lax.stop_gradient(scaling_factor[:, None])

    # Compute cost
    cost_per_timestep = jnp.sum(Q_weights_scaled * errors ** 2, axis=1)
    total_cost = jnp.sum(cost_per_timestep)

    return total_cost

print("  Cost functional configured")

# ============================================
# Setup ParametricFlowSafeControl
# ============================================

print("Setting up FlowBarrier + ParametricFlowSafeControl via cbfjax.from_config...")

system = cbfjax.from_config({
    'dynamics': dynamics,
    'barriers': {
        'state':  state_barrier,
        'backup': {'type': 'func', 'h': backup_barrier_functional, 'rel_deg': 1},
        'flow':   {'type': 'flow', 'state_barrier': 'state',
                   'backup_barriers': ['backup'], 'cfg': cfg},
    },
    'filter': {
        'type': 'parametric_flow',
        'barrier': 'flow',
        'action_dim': dynamics.action_dim,
        'params': {'qp_mode': 'dual', 'qp_solver': 'jaxopt_osqp'},
        'alpha_trajectory': lambda x: alpha_gains['trajectory'] * x,
        'alpha_backup': lambda x: alpha_gains['backup'] * x,
        'alpha_action': lambda x: alpha_gains['action'] * x,
        'alpha_time_shift': lambda x: alpha_gains['time_shift'] * x,
        'cost_functional': cost_functional,
        'control_low': control_low,
        'control_high': control_high,
        'R': jnp.eye(2) * cost_matrices['R'],
        'Lambda': jnp.eye(dynamics.action_dim * cfg['control_param_num']) * cost_matrices['Lambda'],
        'Mu': jnp.array(cost_matrices['Mu']),
        'lambda_linear': jnp.array(cost_matrices['lambda_linear']),
    },
})
safety_filter = system.filter
flow_barrier = system.barriers['flow']

# Test FlowBarrier
h_test = flow_barrier.hocbf(x0)
print(f"  FlowBarrier test value: {np.array(h_test)}")

print("  ParametricFlowSafeControl configured successfully")

# ============================================
# Test Controller
# ============================================

print("\nTesting controller...")
print(f"  Device: {jax.devices()[0]}")

v_aug, _, info = safety_filter.optimal_control_with_info(x0)
print(f"  Physical control: {np.array(v_aug[:dynamics.action_dim])}")
print(f"  Parameter update shape: {v_aug[dynamics.action_dim:-1].shape}")
print(f"  Time shift rate: {np.array(v_aug[-1:])}")

# ============================================
# Closed-Loop Simulation
# ============================================

print("\nStarting closed-loop simulation...")

start_time = time.time()

aug_trajs, actions = safety_filter.get_flow_safe_trajs_action_zoh(
    x0=x0,
    timestep=timestep,
    sim_time=sim_time,
    method='euler',
    use_disturbed=False
)

# single-trajectory rollout: (T, dim); keep a batch axis for the plotting code
aug_trajs = aug_trajs[:, None, :]
actions = actions[:, None, :]

simulation_time = time.time() - start_time
num_steps = int(sim_time / timestep)
print(f"Simulation completed in {simulation_time:.2f}s  |  {num_steps} steps  |  avg {simulation_time/num_steps*1000:.2f} ms/step")

# ============================================
# Extract Trajectories
# ============================================

print("\nExtracting trajectories...")

time_steps_total, batch_size, aug_state_dim = aug_trajs.shape
state_dim = dynamics.state_dim
theta_flat_dim = dynamics.action_dim * cfg['control_param_num']

# Extract state trajectory
state_trajs = aug_trajs[:, :, :state_dim]

# Extract theta trajectory
theta_flat_trajs = aug_trajs[:, :, state_dim:-1]
theta_trajs = theta_flat_trajs.reshape(time_steps_total, batch_size, dynamics.action_dim, cfg['control_param_num'])

# Extract gamma trajectory
gamma_trajs = aug_trajs[:, :, -1:]

# Convert to list format for plotting
state_trajs_transposed = jnp.transpose(state_trajs, (1, 0, 2))
trajs = [state_trajs_transposed[i] for i in range(batch_size)]

n_steps = time_steps_total - 1
print(f"  Trajectory shape: {aug_trajs.shape}")

# ============================================
# Compute Control Actions and Barrier Values
# ============================================

print("\nComputing barrier values and parametric controls...")

barrier_values = []
u_parametric = []
predicted_trajectories = []
h_backup_values = []
parameter_updates = []
time_shift_rates = []

for i in range(batch_size):
    x_i = state_trajs[:, i, :]
    theta_i = theta_trajs[:, i, :, :]
    gamma_i = gamma_trajs[:, i, 0]

    flow_info = jax.vmap(flow_barrier.get_flow_info)(x_i, theta_i, gamma_i)

    barrier_values.append(flow_info['flow_safety'])
    predicted_trajectories.append(flow_info['trajectory'])
    h_backup_values.append(flow_info['h_backup'].reshape(-1, 1))

    u_p_i = jax.vmap(safety_filter.get_parametric_control_value)(theta_i, gamma_i)
    u_parametric.append(u_p_i)

    v_aug = actions[:, i, :]
    u_i = v_aug[:, :dynamics.action_dim]
    omega_i = v_aug[:, dynamics.action_dim:-1]
    z_i = v_aug[:, -1:]

    parameter_updates.append(omega_i)
    time_shift_rates.append(z_i)

# Stack for easier processing
barrier_values_stacked = jnp.stack(barrier_values, axis=1)
u_parametric_stacked = jnp.stack(u_parametric, axis=1)
actions_stacked = actions
parameter_updates_stacked = jnp.stack(parameter_updates, axis=1)
time_shift_rates_stacked = jnp.stack(time_shift_rates, axis=1)
predicted_trajectories_stacked = jnp.stack(predicted_trajectories, axis=1)
h_backup_values_stacked = jnp.stack(h_backup_values, axis=1)

# Convert to list format for plotting
actions_transposed = jnp.transpose(actions_stacked, (1, 0, 2))
actions_list_plot = [actions_transposed[i] for i in range(batch_size)]

u_parametric_transposed = jnp.transpose(u_parametric_stacked, (1, 0, 2))
u_parametric = [u_parametric_transposed[i] for i in range(batch_size)]

barrier_values_transposed = jnp.transpose(barrier_values_stacked, (1, 0, 2))
barrier_values = [barrier_values_transposed[i] for i in range(batch_size)]

predicted_trajectories_transposed = jnp.transpose(predicted_trajectories_stacked, (1, 0, 2, 3))
predicted_trajectories = [predicted_trajectories_transposed[i] for i in range(batch_size)]

h_backup_transposed = jnp.transpose(h_backup_values_stacked, (1, 0, 2))
h_backup_values = [h_backup_transposed[i] for i in range(batch_size)]

time_shift_rates = time_shift_rates_stacked

# Extract final predicted states and backup barrier values
final_pred_states = [pred_traj[:, -1, :] for pred_traj in predicted_trajectories]
backup_barrier_at_final_states = [h_backup[:, -1] for h_backup in h_backup_values]

print("  Barrier and parametric control computation complete")

# ============================================
# Statistics
# ============================================

# Convert to numpy for statistics
traj_np = np.array(trajs[0])
actions_np = np.array(actions_list_plot[0])
barrier_vals_np = np.array(barrier_values[0])
gamma_np = np.array(gamma_trajs[:, 0, 0])
goal_pos_np = np.array(goal_pos[0])

num_state_points = trajs[0].shape[0]
num_control_points = num_state_points - 1
time_array_states = np.linspace(0.0, (num_state_points - 1) * timestep, num_state_points)
time_array_controls = np.linspace(0.0, (num_control_points - 1) * timestep, num_control_points)

print(f"\n{'='*60}")
print(f"Simulation statistics ({n_steps} steps):")
print(f"  Total time: {simulation_time:.2f} s")
print(f"  Avg time per step: {simulation_time/n_steps*1000:.3f} ms")
print(f"{'='*60}")
print(f"Trajectory statistics:")
print(f"  Initial position: ({traj_np[0, 0]:.2f}, {traj_np[0, 1]:.2f})")
print(f"  Final position: ({traj_np[-1, 0]:.2f}, {traj_np[-1, 1]:.2f})")
print(f"  Goal position: ({goal_pos_np[0]:.2f}, {goal_pos_np[1]:.2f})")
print(f"  Final distance to goal: {np.linalg.norm(traj_np[-1, :2] - goal_pos_np):.3f} m")
print(f"{'='*60}")
print(f"Barrier statistics:")
print(f"  Min barrier value: {np.min(barrier_vals_np):.6f}")
print(f"{'='*60}")
print(f"Control statistics:")
print(f"  u1: min={actions_np[:, 0].min():.3f}, max={actions_np[:, 0].max():.3f}")
print(f"  u2: min={actions_np[:, 1].min():.3f}, max={actions_np[:, 1].max():.3f}")
print(f"  Control bounds: u1 in [{control_low[0]}, {control_high[0]}], u2 in [{control_low[1]}, {control_high[1]}]")
print(f"{'='*60}")
print(f"FlowBarrier statistics:")
print(f"  Final gamma value: {gamma_np[-1]:.4f}")
print(f"  Control updates: {num_control_points} (ZOH) vs {num_state_points} state points")
print(f"{'='*60}")

# Prepare plotting variables
u_p_np = np.array(u_parametric[0])
theta_u1 = np.array(theta_trajs[:, 0, 0, :])
theta_u2 = np.array(theta_trajs[:, 0, 1, :])
omega_u1 = np.array(parameter_updates_stacked[:, 0, :cfg['control_param_num']])
omega_u2 = np.array(parameter_updates_stacked[:, 0, cfg['control_param_num']:])
time_shift_np = np.array(time_shift_rates[:, 0, 0])
final_pred_states_0 = np.array(final_pred_states[0])
backup_barrier_0 = np.array(backup_barrier_at_final_states[0])
predicted_traj_0 = np.array(predicted_trajectories[0])

# ============================================
# Plots
# ============================================

print("\nGenerating plots...")

current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Create figs directory
os.makedirs(os.path.join(script_dir, 'figs'), exist_ok=True)

# Create mesh for map visualization
x_grid = np.linspace(-10.5, 10.5, 100)
y_grid = np.linspace(-10.5, 10.5, 100)
X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
points = np.column_stack((X_grid.flatten(), Y_grid.flatten()))
points_jax = jnp.array(points)
points_jax = jnp.concatenate([points_jax, jnp.zeros((points_jax.shape[0], 2))], axis=-1)
Z = jax.vmap(map_.barrier.min_barrier)(points_jax)
Z = np.array(Z).reshape(X_grid.shape)

# --- Plot 1: Trajectory ---
fig, ax = plt.subplots(figsize=(8, 8))

ax.contour(X_grid, Y_grid, Z, levels=[0], colors='red', linewidths=2)
ax.plot([], [], 'r-', linewidth=2, label=r'$\mathcal{S}_{\rm s}$')

ax.plot(traj_np[0, 0], traj_np[0, 1], 'o', color='blue', markersize=10, label=r'$x_0$')
ax.plot(traj_np[-1, 0], traj_np[-1, 1], 's', color='blue', markersize=10, label=r'$x_f$')
ax.plot(traj_np[:, 0], traj_np[:, 1], 'b-', linewidth=2, label='Trajectory')
ax.plot(goal_pos_np[0], goal_pos_np[1], '*', markersize=15, color='limegreen', label='Goal')

ax.set_xlabel(r'$q_{\rm x}$ (m)')
ax.set_ylabel(r'$q_{\rm y}$ (m)')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3, frameon=False, columnspacing=5.0)
ax.grid(False)
ax.set_aspect('equal', adjustable='box')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xticks(np.arange(-10, 11, 5))
ax.set_yticks(np.arange(-10, 11, 5))
ax.set_xlim(-10.5, 10.5)
ax.set_ylim(-10.5, 10.5)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, f'figs/FlowBarrier_Trajectory_{current_time}.png'), dpi=600)
plt.savefig(os.path.join(script_dir, f'figs/FlowBarrier_Trajectory_{current_time}.svg'))
plt.savefig(os.path.join(script_dir, f'figs/FlowBarrier_Trajectory_{current_time}.pdf'))
plt.show()

# --- Plot 2: Combined States, Controls, and Barriers ---
fig, axs = plt.subplots(6, 1, figsize=(10, 10))

# States
axs[0].plot(time_array_states, traj_np[:, 0], 'b-', linewidth=2, label=r'$q_{\rm x}$')
axs[0].plot(time_array_states, traj_np[:, 1], 'r-', linewidth=2, label=r'$q_{\rm y}$')
axs[0].axhline(y=goal_pos_np[0], color='b', linestyle=(0, (2, 2)), alpha=0.7)
axs[0].axhline(y=goal_pos_np[1], color='r', linestyle=(2.0, (2, 2)), alpha=0.7)
axs[0].set_ylabel(r'$q_{\rm x}, q_{\rm y}$ (m)')
from matplotlib.lines import Line2D
legend_handles = [
    Line2D([0], [0], color='b', linewidth=2, label=r'$q_{\rm x}$'),
    Line2D([0], [0], color='b', linestyle='--', alpha=0.7, label=r'$q_{{\rm d},x}$'),
    Line2D([0], [0], color='r', linewidth=2, label=r'$q_{\rm y}$'),
    Line2D([0], [0], color='r', linestyle='--', alpha=0.7, label=r'$q_{{\rm d},y}$'),
]
axs[0].legend(handles=legend_handles, loc='upper right', frameon=False, ncol=2)

axs[1].plot(time_array_states, traj_np[:, 2], 'b-', linewidth=2)
axs[1].set_ylabel(r'$v$ (m/s)')
axs[1].axhline(y=2, color='k', linestyle='--', alpha=0.5)

axs[2].plot(time_array_states, traj_np[:, 3], 'b-', linewidth=2)
axs[2].set_ylabel(r'$\vartheta$ (rad)')

# Controls
axs[3].plot(time_array_controls, actions_np[:, 0], 'b-', linewidth=2, label=r'$u_1$')
axs[3].plot(time_array_states, u_p_np[:, 0], 'r--', linewidth=2, label=r'$u_{p,1}(\gamma, \theta)$')
axs[3].axhline(y=control_low[0], color='k', linestyle='--', alpha=0.5)
axs[3].axhline(y=control_high[0], color='k', linestyle='--', alpha=0.5)
axs[3].set_ylabel(r'$u_1$ (m/s$^2$)')
axs[3].legend(loc='upper right', frameon=False, ncol=3)

axs[4].plot(time_array_controls, actions_np[:, 1], 'b-', linewidth=2, label=r'$u_2$')
axs[4].plot(time_array_states, u_p_np[:, 1], 'r--', linewidth=2, label=r'$u_{p,2}(\gamma, \theta)$')
axs[4].axhline(y=control_low[1], color='k', linestyle='--', alpha=0.5)
axs[4].axhline(y=control_high[1], color='k', linestyle='--', alpha=0.5)
axs[4].set_ylabel(r'$u_2$ (rad/s)')
axs[4].legend(loc='upper right', frameon=False, ncol=3)

# All Barriers in single subplot
num_barriers = barrier_vals_np.shape[1]
colors = ['blue', 'orange', 'red']
compose_action_barriers = cfg.get('compose_action_barriers', True)
compose_state_barriers = cfg.get('compose_state_barriers', True)
barrier_idx = 0

# Trajectory barrier (psi_c)
if not compose_state_barriers:
    target_points = int(cfg['horizon'] / cfg['time_steps'])
    num_trajectory_only = target_points - 1
    if barrier_idx + num_trajectory_only <= num_barriers:
        traj_barriers = barrier_vals_np[:, barrier_idx:barrier_idx + num_trajectory_only]
        min_traj_barriers = np.min(traj_barriers, axis=1)
        axs[5].plot(time_array_states, min_traj_barriers, color=colors[0], linewidth=2, label=r'$\bar\psi_{\rm m}$')
        barrier_idx += num_trajectory_only
else:
    if barrier_idx < num_barriers:
        axs[5].plot(time_array_states, barrier_vals_np[:, barrier_idx], color=colors[0], linewidth=2, label=r'$\bar\psi_{\rm m}$')
        barrier_idx += 1

# Backup barrier (psi_b)
if barrier_idx < num_barriers:
    axs[5].plot(time_array_states, barrier_vals_np[:, barrier_idx], color=colors[1], linewidth=2, label=r'$\bar\psi_{\rm t}$')
    barrier_idx += 1

# Action barrier (k)
if 'control_low' in cfg and cfg['control_low'] is not None:
    if not compose_action_barriers:
        num_action_barriers = dynamics.action_dim * 2 * cfg['control_param_num']
        if barrier_idx + num_action_barriers <= num_barriers:
            action_barriers = barrier_vals_np[:, barrier_idx:barrier_idx + num_action_barriers]
            min_action_barriers = np.min(action_barriers, axis=1)
            axs[5].plot(time_array_states, min_action_barriers, color=colors[2], linewidth=2, label=r'$\kappa$')
            barrier_idx += num_action_barriers
    else:
        if barrier_idx < num_barriers:
            axs[5].plot(time_array_states, barrier_vals_np[:, barrier_idx], color=colors[2], linewidth=2, label=r'$\kappa$')
            barrier_idx += 1

axs[5].set_yscale('linear')
axs[5].axhline(y=0, color='k', linestyle='--', alpha=0.5)
axs[5].set_ylabel(r'$\bar\psi_{\rm m}, \bar\psi_{\rm t}, \kappa$')
axs[5].set_xlabel(r'$t$ (s)')
axs[5].legend(loc='center right', ncol=3, frameon=False)

# Hide x labels except for last subplot
for i in range(5):
    axs[i].tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)

for ax in axs:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(time_array_states[0], time_array_states[-1])
    ax.xaxis.label.set_fontsize(22)
    ax.yaxis.label.set_fontsize(22)
    ax.tick_params(axis='both', labelsize=20)

plt.tight_layout()
plt.savefig(os.path.join(script_dir, f'figs/FlowBarrier_Combined_{current_time}.png'), dpi=600)
plt.savefig(os.path.join(script_dir, f'figs/FlowBarrier_Combined_{current_time}.svg'))
plt.savefig(os.path.join(script_dir, f'figs/FlowBarrier_Combined_{current_time}.pdf'))
plt.show()

# --- Plot 4: Parameters Evolution (Theta, Omega, Gamma, z) ---
from matplotlib.gridspec import GridSpec
fig = plt.figure(figsize=(12, 10), constrained_layout=True)
gs = GridSpec(4, 2, width_ratios=[1, 0.015], wspace=0.02, figure=fig)

axs = [fig.add_subplot(gs[i, 0]) for i in range(4)]
cax0 = fig.add_subplot(gs[0, 1])
cax1 = fig.add_subplot(gs[1, 1])

# Theta parameters - all lines with colormap
colormap = plt.cm.ocean
total_params = 2 * cfg['control_param_num']
norm = plt.Normalize(0, total_params - 1)

for i in range(cfg['control_param_num']):
    color_u1 = colormap(norm(i))
    color_u2 = colormap(norm(i + cfg['control_param_num']))
    axs[0].plot(time_array_states, theta_u1[:, i], color=color_u1, alpha=0.8, linewidth=0.8)
    axs[0].plot(time_array_states, theta_u2[:, i], color=color_u2, alpha=0.8, linewidth=0.8)
axs[0].set_ylabel(r'$\theta$')
sm0 = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
sm0.set_array([])
cbar0 = plt.colorbar(sm0, cax=cax0)
cbar0.set_label(r'Index', )

# Omega (parameter updates) - all lines with colormap
for i in range(cfg['control_param_num']):
    color_u1 = colormap(norm(i))
    color_u2 = colormap(norm(i + cfg['control_param_num']))
    axs[1].plot(time_array_controls, omega_u1[:, i], color=color_u1, alpha=0.8, linewidth=0.8)
    axs[1].plot(time_array_controls, omega_u2[:, i], color=color_u2, alpha=0.8, linewidth=0.8)
axs[1].set_ylabel(r'$\omega$')
sm1 = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
sm1.set_array([])
cbar1 = plt.colorbar(sm1, cax=cax1)
cbar1.set_label(r'Index', )

# Gamma
axs[2].plot(time_array_states, gamma_np, 'b-', linewidth=2)
axs[2].set_ylabel(r'$\gamma$')

# z (time shift rate)
axs[3].plot(time_array_controls, time_shift_np, 'b-', linewidth=2)
axs[3].set_ylabel(r'$z$')
axs[3].set_xlabel(r'$t$ (s)')

# Hide x labels except for last subplot
for i in range(3):
    axs[i].tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)

from matplotlib.ticker import ScalarFormatter

class _FloatScalarFormatter(ScalarFormatter):
    """ScalarFormatter that always uses float format (e.g. 0.0 not 0)."""
    def _set_format(self):
        self.format = '%.1f'

for ax in axs:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(time_array_states[0], time_array_states[-1])
    fmt = _FloatScalarFormatter(useMathText=True)
    fmt.set_powerlimits((-1, 1))
    ax.yaxis.set_major_formatter(fmt)
    ax.xaxis.label.set_fontsize(28)
    ax.yaxis.label.set_fontsize(28)
    ax.tick_params(axis='both', labelsize=24)
    ax.yaxis.get_offset_text().set_fontsize(20)

for cbar in (cbar0, cbar1):
    cbar.set_ticks([0, 50, 100, 150])
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label(r'Index', fontsize=22)

plt.savefig(os.path.join(script_dir, f'figs/FlowBarrier_Parameters_{current_time}.png'), dpi=600)
plt.savefig(os.path.join(script_dir, f'figs/FlowBarrier_Parameters_{current_time}.svg'))
plt.savefig(os.path.join(script_dir, f'figs/FlowBarrier_Parameters_{current_time}.pdf'))
plt.show()

# --- Plot 5: Final Predicted State Analysis ---
fig, axs = plt.subplots(5, 1, figsize=(12, 12))

valid_indices = min(len(final_pred_states_0), len(backup_barrier_0))
time_array_valid = time_array_states[:valid_indices]

axs[0].plot(time_array_valid, final_pred_states_0[:valid_indices, 0], 'b-', linewidth=2)
axs[0].set_ylabel(r'$q_{\rm x}$ (m)')

axs[1].plot(time_array_valid, final_pred_states_0[:valid_indices, 1], 'g-', linewidth=2)
axs[1].set_ylabel(r'$q_{\rm y}$ (m)')

axs[2].plot(time_array_valid, final_pred_states_0[:valid_indices, 2], 'r-', linewidth=2)
axs[2].set_ylabel(r'$v$ (m/s)')

axs[3].plot(time_array_valid, final_pred_states_0[:valid_indices, 3], 'm-', linewidth=2)
axs[3].set_ylabel(r'$\theta$ (rad)')

axs[4].plot(time_array_valid, backup_barrier_0[:valid_indices], 'orange', linewidth=2)
axs[4].axhline(y=0, color='black', linestyle='--', alpha=0.5)
axs[4].set_ylabel(r'$h_{backup}$')
axs[4].set_xlabel(r'$t$ (s)')

for i in range(4):
    axs[i].tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)

for ax in axs:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.suptitle('Final Predicted State and Backup Barrier Analysis', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, f'figs/FlowBarrier_BackupAnalysis_{current_time}.png'), dpi=600)
plt.savefig(os.path.join(script_dir, f'figs/FlowBarrier_BackupAnalysis_{current_time}.svg'))
plt.savefig(os.path.join(script_dir, f'figs/FlowBarrier_BackupAnalysis_{current_time}.pdf'))
plt.show()

# --- Animation ---
if make_animation:
    print("\nCreating animation...")

def create_animation():
    xy_points = np.column_stack((X_grid.flatten(), Y_grid.flatten()))

    @jax.jit
    def compute_backup_contour_batch(final_states):
        def single_backup(final_state):
            v, theta = final_state
            state_grid = jnp.column_stack((
                jnp.array(xy_points),
                jnp.full(xy_points.shape[0], v),
                jnp.full(xy_points.shape[0], theta)
            ))
            return jax.vmap(system.barriers['backup'].hocbf)(state_grid)
        return jax.vmap(single_backup)(final_states)

    # Extract final predicted states
    time_steps_total = len(traj_np)
    final_states_all = []
    for frame in range(time_steps_total):
        if frame < len(predicted_traj_0):
            pred_traj = predicted_traj_0[frame]
            final_pred_state = pred_traj[-1]
            final_states_all.append([final_pred_state[2], final_pred_state[3]])

    # Batch processing
    batch_size_anim = 500
    num_batches = (len(final_states_all) + batch_size_anim - 1) // batch_size_anim

    print(f"  Computing backup contours in {num_batches} batches...")

    backup_contours_cache = {}

    def get_backup_contour(frame):
        if frame not in backup_contours_cache:
            batch_idx = frame // batch_size_anim
            batch_start = batch_idx * batch_size_anim
            batch_end = min(batch_start + batch_size_anim, len(final_states_all))

            print(f"    Computing batch {batch_idx + 1}/{num_batches}...")

            backup_contours_cache.clear()

            batch_states = jnp.array(final_states_all[batch_start:batch_end])
            batch_contours = compute_backup_contour_batch(batch_states)

            for i, frame_idx in enumerate(range(batch_start, batch_end)):
                backup_contours_cache[frame_idx] = np.array(batch_contours[i]).reshape(X_grid.shape)

        return backup_contours_cache[frame]

    fig_anim, ax_anim = plt.subplots(figsize=(10, 10))

    def animate(frame):
        ax_anim.clear()

        current_x = traj_np[frame, 0]
        current_y = traj_np[frame, 1]
        current_v = traj_np[frame, 2]

        past_x = traj_np[:frame + 1, 0]
        past_y = traj_np[:frame + 1, 1]

        ax_anim.contour(X_grid, Y_grid, Z, levels=[0], colors='red', linewidths=2, alpha=0.7)
        ax_anim.plot(goal_pos_np[0], goal_pos_np[1], '*', markersize=20,
                color='limegreen', label='Goal', zorder=5)
        ax_anim.plot(past_x, past_y, 'b-', linewidth=2, label='Trajectory', zorder=3)
        ax_anim.scatter([current_x], [current_y], s=100, c='blue', marker='o',
                   edgecolors='black', linewidths=2, label='Current', zorder=4)

        if frame < len(predicted_traj_0):
            pred_traj = predicted_traj_0[frame]
            pred_x = pred_traj[:, 0]
            pred_y = pred_traj[:, 1]
            ax_anim.plot(pred_x, pred_y, 'c--', linewidth=1.5, alpha=0.7,
                    label='Predicted', zorder=2)
            sample_indices = np.arange(0, len(pred_x), max(1, len(pred_x) // 20))
            ax_anim.scatter(pred_x[sample_indices], pred_y[sample_indices],
                       s=20, c='cyan', alpha=0.5, zorder=2)

            if frame < len(final_states_all):
                backup_contour = get_backup_contour(frame)
                ax_anim.contour(X_grid, Y_grid, backup_contour, levels=[0],
                           colors='orange', linewidths=1.5, alpha=0.8, linestyles='--')

        ax_anim.set_xlabel(r'$q_{\rm x}$ (m)')
        ax_anim.set_ylabel(r'$q_{\rm y}$ (m)')
        ax_anim.set_xlim(-10, 10)
        ax_anim.set_ylim(-10, 10)
        ax_anim.set_aspect('equal', adjustable='box')
        ax_anim.legend(loc='upper left', frameon=False)
        ax_anim.spines['top'].set_visible(False)
        ax_anim.spines['right'].set_visible(False)

        current_time_val = frame * timestep
        ax_anim.text(0.98, 0.98,
                f'Time: {current_time_val:.2f}s\nVel: {current_v:.2f} m/s',
                transform=ax_anim.transAxes, fontsize=11, verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        return []

    anim = animation.FuncAnimation(fig_anim, animate, frames=time_steps_total,
                                   interval=50, blit=True)

    animation_file = os.path.join(script_dir, f'figs/FlowBarrier_Animation_{current_time}.mp4')
    writer = animation.FFMpegWriter(fps=20, metadata=dict(artist='FlowBarrierJAX'), bitrate=1800)
    anim.save(animation_file, writer=writer)
    print(f"Animation saved as: {animation_file}")
    plt.show()

if make_animation:
    create_animation()

print(f"\nPlots saved with timestamp: {current_time}")
print("Simulation complete!")