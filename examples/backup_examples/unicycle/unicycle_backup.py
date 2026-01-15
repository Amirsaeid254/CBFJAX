"""
Unicycle Backup Safe Control Example

This example demonstrates backup safe control for unicycle dynamics using CBFJAX.
It combines:
- Backup trajectories from predefined policies
- State barriers for obstacle avoidance
- Backup barriers for terminal constraints
- QP-based safe control with smooth blending
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from functools import partial
import datetime
from immutabledict import immutabledict

# CBFJAX imports
import cbfjax
from cbfjax.dynamics import UnicycleReducedOrderDynamics
from cbfjax.barriers import Barrier, BackupBarrier
from cbfjax.safe_controls import MinIntervBackupSafeControl
from cbfjax.utils.make_map import Map

# Local imports
from map_config import map_config
from backup_policies import UnicycleBackupControl
from unicycle_desired_control import desired_control

# Configure matplotlib
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'

# ============================================================================
# Configuration
# ============================================================================

# Backup barrier configuration
backup_cfg = immutabledict({
    'softmax_rho': 50.0,
    'softmin_rho': 50.0,
    'rel_deg': 1,
    'horizon': 4.0,
    'time_steps': 0.05,
    'integration_method': 'euler',
    'epsilon': 0.0,
    'h_scale': 0.0012,
    'feas_scale': 0.05,
})

# Map configuration
map_cfg = immutabledict({
    'softmin_rho': 20.0,
    'boundary_alpha': (),
    'obstacle_alpha': (),
    'velocity_alpha': (),
    'pos_barrier_rel_deg': 1,
})

# Dynamics parameters
control_bounds = ((-2.0, -1.0), (2.0, 1.0))
dynamics_params = immutabledict({'d': 1.0, 'control_bounds': control_bounds})

# Backup policy parameters
ub_gain = ((-15.0, 0.0),)

# Control gains for desired control
control_gains = immutabledict(k1=1.0, k2=0.8)

# Goal positions
# goal_pos = jnp.array([
#     [2.0, 4.5],
#     [-1.0, 0.0],
#     [-4.5, 8.0],
# ])

goal_pos = jnp.array([
    [2.0, 4.5]
])


# Initial conditions
x0 = jnp.array([[-3.0, -8.5, 0.0, 0.0]]).repeat(goal_pos.shape[0], axis=0)

# Simulation parameters
timestep = 0.01
sim_time = 20.0

# ============================================================================
# Setup
# ============================================================================

print("Setting up backup safe control system...")

# 1. Create dynamics
dynamics = UnicycleReducedOrderDynamics(params=dynamics_params)
print(f"  - Dynamics: UnicycleReducedOrderDynamics")

# 2. Create map with state barriers
map_ = Map(dynamics=dynamics, cfg=map_cfg, barriers_info=map_config).create_barriers()
state_barrier = map_.barrier
print(f"  - State barrier: {len(map_.barrier._barriers)} obstacle/boundary barriers")

# 3. Create backup policies
backup_controls = UnicycleBackupControl(ub_gain, control_bounds)()
print(f"  - Backup policies: {len(backup_controls)} policies")

# 4. Create backup barriers
# Backup barrier = state barrier - velocity penalty
def backup_barrier_func(x):
    """Backup barrier: combine state safety with velocity constraint."""
    h_state = state_barrier._hocbf_single(x)
    velocity_penalty = (1.0 * jnp.pow(x[2:3], 2)) / control_bounds[1][0]
    return h_state - velocity_penalty

backup_barriers = [
    Barrier().assign(
        barrier_func=backup_barrier_func,
        rel_deg=1,
        alphas=[]).assign_dynamics(dynamics)]

print(f"  - Backup barriers: {len(backup_barriers)} barriers")

# 5. Create backup barrier system
fwd_barrier = (BackupBarrier.create_empty(cfg=backup_cfg)
               .assign_state_barrier([state_barrier])
               .assign_backup_policies(backup_controls)
               .assign_backup_barrier(backup_barriers)
               .assign_dynamics(dynamics)
               .make())
print("  - Backup barrier system built")

# Test backup barrier
h_test = fwd_barrier.hocbf(x0)
print(f"  - Backup barrier test value: {h_test}")

# 6. Create safety filter
safety_filter = MinIntervBackupSafeControl(
    action_dim=dynamics.action_dim,
    alpha=lambda x: 1.0 * x,
    slacked=False,
    control_low=list(control_bounds[0]),
    control_high=list(control_bounds[1])
).assign_dynamics(dynamics).assign_state_barrier(fwd_barrier)

print("  - Safety filter configured")

# ============================================================================
# Simulation
# ============================================================================

print("\nRunning simulation...")

# Assign desired control based on goal positions
def desired_control_func(x):
    """Desired control toward goal."""
    return desired_control(x, goal_pos[0], dynamics_params, **control_gains)

safety_filter = safety_filter.assign_desired_control(desired_control_func)

# Test single control computation
safety_filter._safe_optimal_control_single(x0.squeeze(0))

# Simulate trajectories using ZOH (zero-order hold)
print(f"  - Simulating {sim_time}s with timestep {timestep}s...")
import time
start_time = time.time()

trajs = safety_filter.get_safe_optimal_trajs_zoh(
    x0=x0,
    sim_time=sim_time,
    timestep=timestep,
    method='dopri5'
)
elapsed = time.time() - start_time
print(f"  - Simulation completed in {elapsed:.2f}s")
print(f"  - Trajectory shape: {trajs.shape}")

# Print profiling summary
from cbfjax.utils import print_profile_summary
print_profile_summary()


# ============================================================================
# Analysis
# ============================================================================

print("\nAnalyzing trajectory...")

# Get control and barrier values along trajectory
traj = trajs[:, 0, :]  # (time_steps, state_dim)
num_points = traj.shape[0]
time_array = jnp.linspace(0.0, (num_points - 1) * timestep, num_points)

print(f"  - Computing controls and barriers...")
u_vals, info = safety_filter.safe_optimal_control(traj, ret_info=True)

# Compute desired controls for each state
des_ctrls = jax.vmap(desired_control_func)(traj)

# Extract barrier values
h_vals = fwd_barrier.hocbf(traj)
h_s = state_barrier.hocbf(traj)

print(f"  - Final state: {traj[-1]}")
print(f"  - Min barrier value: {jnp.min(h_vals):.4f}")
print(f"  - Max control magnitude: {jnp.max(jnp.linalg.norm(u_vals, axis=1)):.4f}")



# ============================================================================
# Plotting
# ============================================================================

print("\nGenerating plots...")

current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# 1. Create map visualization
x = np.linspace(-10.5, 10.5, 500)
y = np.linspace(-10.5, 10.5, 500)
X, Y = np.meshgrid(x, y)
points = np.column_stack((X.flatten(), Y.flatten()))
points_jax = jnp.array(points)
points_with_zeros = jnp.concatenate([points_jax, jnp.zeros((points_jax.shape[0], 1))], axis=-1)
Z = map_.barrier.min_barrier(points_with_zeros)
Z = Z.reshape(X.shape)

# Plot 1: Trajectory in workspace
fig, ax = plt.subplots(figsize=(8, 8))

# Draw obstacles
contour = ax.contour(X, Y, Z, levels=[0], colors='red', linewidths=2)

# Draw trajectory
ax.plot(traj[0, 0], traj[0, 1], 'x', color='blue', markersize=10, label='Start')
ax.plot(traj[-1, 0], traj[-1, 1], '+', color='blue', markersize=10, label='End')
ax.plot(traj[:, 0], traj[:, 1], 'black', linewidth=2, label='Trajectory')
ax.plot(goal_pos[0, 0], goal_pos[0, 1], '*', markersize=15, color='limegreen', label='Goal')

ax.set_xlabel(r'$x$', fontsize=14)
ax.set_ylabel(r'$y$', fontsize=14)
ax.set_aspect('equal', adjustable='box')
ax.legend(loc='upper right', fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(f'figs/Trajectory_Backup_Safe_Control_{current_time}.png', dpi=300)
print(f"  - Saved: Trajectory_Backup_Safe_Control_{current_time}.png")

# Plot 2: States and Controls (matching CBFTorch)
fig, axs = plt.subplots(6, 1, figsize=(8, 8))

# States
axs[0].plot(time_array, traj[:, 0], label=r'$r_x$', color='black')
axs[0].set_ylabel(r'$r_x$', fontsize=14)

axs[1].plot(time_array, traj[:, 1], label=r'$r_y$', color='black')
axs[1].set_ylabel(r'$r_y$', fontsize=14)

axs[2].plot(time_array, traj[:, 2], label=r'$v$', color='black')
axs[2].set_ylabel(r'$v$', fontsize=14)

axs[3].plot(time_array, traj[:, 3], label=r'$\theta$', color='black')
axs[3].set_ylabel(r'$\theta$', fontsize=14)

# Controls with comparisons
axs[4].plot(time_array, u_vals[:, 0], label='u', color='black')
axs[4].plot(time_array, info['ub_select'][:, 0], label=r'$u_b$', color='green')
axs[4].plot(time_array, info['u_star'][:, 0], label=r'$u_*$', color='blue')
axs[4].plot(time_array, des_ctrls[:, 0], label=r'$u_d$', color='red', linestyle='--')
axs[4].legend(fontsize=14, loc='upper right', frameon=False, ncol=4)
axs[4].set_ylabel(r'$u_1$', fontsize=14)

axs[5].plot(time_array, u_vals[:, 1], label='u', color='black')
axs[5].plot(time_array, info['ub_select'][:, 1], label=r'$u_b$', color='green')
axs[5].plot(time_array, info['u_star'][:, 1], label=r'$u_*$', color='blue')
axs[5].plot(time_array, des_ctrls[:, 1], label=r'$u_d$', color='red', linestyle='--')
axs[5].legend(fontsize=14, loc='upper right', frameon=False, ncol=4)
axs[5].set_ylabel(r'$u_2$', fontsize=14)
axs[5].set_xlabel(r'$t~(\rm {s})$', fontsize=14)

plt.tight_layout()
plt.savefig(f'figs/Controls_QP_Backup_Safe_Control_{current_time}.png', dpi=600)
print(f"  - Saved: Controls_QP_Backup_Safe_Control_{current_time}.png")

# Plot 3: Barriers (matching CBFTorch, without h_star)
fig, axs = plt.subplots(3, 1, figsize=(8, 8))

# Barrier values
axs[0].plot(time_array, h_vals[:, 0], label=r'$h$', color='blue')
axs[0].plot(time_array, h_s[:, 0], label=r'$h_s$', color='red', linestyle='--')
axs[0].plot(time_array, jnp.zeros(time_array.shape[0]), color='green', linestyle='dotted')
axs[0].set_ylabel(r'$h$', fontsize=14)
axs[0].legend(fontsize=14, loc='best', frameon=False)
axs[0].tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)
axs[0].set_xlim(0, sim_time)
axs[0].set_yscale('log')
# Set y-axis limits to ensure all data is visible on log scale
h_min = jnp.min(jnp.concatenate([h_vals[:, 0], h_s[:, 0]]))
h_max = jnp.max(jnp.concatenate([h_vals[:, 0], h_s[:, 0]]))
axs[0].set_ylim(bottom=h_min * 0.5, top=h_max * 2.0)

# Normalized factors
h_normalized = (h_vals[:, 0] - backup_cfg['epsilon']) / backup_cfg['h_scale']
feas_normalized = info['feas_fact'][:] / backup_cfg['feas_scale']
axs[1].plot(time_array, h_normalized, label=r'$\frac{h - \epsilon}{\kappa_h}$', color='blue')
axs[1].plot(time_array, feas_normalized, label=r'$\frac{\beta}{\kappa_\beta}$', color='red')
axs[1].plot(time_array, jnp.zeros(time_array.shape[0]), color='green')
axs[1].set_ylabel(r'$\frac{h - \epsilon}{\kappa_h}, \frac{\beta}{\kappa_\beta}$', fontsize=14)
axs[1].legend(fontsize=14, loc='best', frameon=False)
axs[1].set_xlim(0, sim_time)
axs[1].set_yscale('log')
# Set y-axis limits for normalized factors
norm_min = jnp.min(jnp.concatenate([h_normalized, feas_normalized]))
norm_max = jnp.max(jnp.concatenate([h_normalized, feas_normalized]))
axs[1].set_ylim(bottom=norm_min * 0.5, top=norm_max * 2.0)

# Blending factor (sigma)
axs[2].plot(time_array, info['beta'][:], label=r'$\sigma$', color='blue')
axs[2].set_ylabel(r'$\sigma$', fontsize=14)
axs[2].set_xlabel(r'$t~(\rm {s})$', fontsize=14)
axs[2].set_xlim(0, sim_time)

plt.tight_layout()
plt.savefig(f'figs/Barriers_QP_Backup_Safe_Control_{current_time}.png', dpi=600)
print(f"  - Saved: Barriers_QP_Backup_Safe_Control_{current_time}.png")

plt.show()

print("\nExample completed successfully!")