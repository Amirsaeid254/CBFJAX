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
import os
import time
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

# Get script directory for saving figures
script_dir = os.path.dirname(os.path.abspath(__file__))

# Configure matplotlib
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'

# ============================================
# Configuration
# ============================================

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

# Goal position
goal_pos = jnp.array([[2.0, 4.5]])

# Initial condition
x0 = jnp.array([[-3.0, -8.5, 0.0, 0.0]]).repeat(goal_pos.shape[0], axis=0)

# Simulation parameters
timestep = 0.01
sim_time = 20.0

# ============================================
# Setup Dynamics
# ============================================

print("Setting up dynamics...")

dynamics = UnicycleReducedOrderDynamics(params=dynamics_params)
nx = dynamics.state_dim  # 4: [q_x, q_y, v, theta]
nu = dynamics.action_dim  # 2: [acceleration, angular_velocity]

print(f"  State dim: {nx}, Action dim: {nu}")

# ============================================
# Setup State Barriers
# ============================================

print("Setting up state barriers...")

map_ = Map(dynamics=dynamics, cfg=map_cfg, barriers_info=map_config).create_barriers()
state_barrier = map_.barrier

print(f"  State barrier: {len(map_.barrier._barriers)} obstacle/boundary barriers")

# ============================================
# Setup Backup Policies
# ============================================

print("Setting up backup policies...")

backup_controls = UnicycleBackupControl(ub_gain, control_bounds)()

print(f"  Backup policies: {len(backup_controls)} policies")

# ============================================
# Setup Backup Barriers
# ============================================

print("Setting up backup barriers...")

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

print(f"  Backup barriers: {len(backup_barriers)} barriers")

# ============================================
# Setup Backup Barrier System
# ============================================

print("Setting up backup barrier system...")

fwd_barrier = (BackupBarrier.create_empty(cfg=backup_cfg)
               .assign_state_barrier([state_barrier])
               .assign_backup_policies(backup_controls)
               .assign_backup_barrier(backup_barriers)
               .assign_dynamics(dynamics)
               .make())

print("  Backup barrier system built")

# Test backup barrier
h_test = fwd_barrier.hocbf(x0)
print(f"  Backup barrier test value: {np.array(h_test)}")

# ============================================
# Setup Safety Filter
# ============================================

print("Setting up safety filter...")

safety_filter = MinIntervBackupSafeControl(
    action_dim=dynamics.action_dim,
    alpha=lambda x: 1.0 * x,
    slacked=False,
    control_low=list(control_bounds[0]),
    control_high=list(control_bounds[1])
).assign_dynamics(dynamics).assign_state_barrier(fwd_barrier)

print("  Safety filter configured")

# ============================================
# Test Controller
# ============================================

print("\nTesting controller...")

# Assign desired control based on goal positions
def desired_control_func(x):
    """Desired control toward goal."""
    return desired_control(x, goal_pos[0], dynamics_params, **control_gains)

safety_filter = safety_filter.assign_desired_control(desired_control_func)

u_test, _ = safety_filter._optimal_control_single(x0.squeeze(0))
print(f"  Test control: u = {np.array(u_test)}")

# ============================================
# Closed-Loop Simulation
# ============================================

print("\nStarting closed-loop simulation...")
print(f"  Device: {jax.devices()[0]}")

start_time = time.time()

trajs = safety_filter.get_optimal_trajs_zoh(
    x0=x0,
    sim_time=sim_time,
    timestep=timestep,
    method='dopri5'
)

simulation_time = time.time() - start_time
print(f"Simulation completed in {simulation_time:.2f} seconds")

# Print profiling summary
from cbfjax.utils import print_profile_summary
print_profile_summary()

# ============================================
# Compute Control Actions and Barrier Values
# ============================================

print("\nComputing control actions and barrier values...")

# Get control and barrier values along trajectory
traj = trajs[:, 0, :]  # (time_steps, state_dim)
n_steps = traj.shape[0] - 1
time_array = np.linspace(0.0, sim_time, n_steps + 1)

u_vals, info = safety_filter.optimal_control(traj, ret_info=True)

# Compute desired controls for each state
des_ctrls = jax.vmap(desired_control_func)(traj)

# Extract barrier values
h_vals = fwd_barrier.hocbf(traj)
h_s = state_barrier.hocbf(traj)

# Convert to numpy
traj_np = np.array(traj)
u_vals_np = np.array(u_vals)
h_vals_np = np.array(h_vals)
h_s_np = np.array(h_s)
goal_pos_np = np.array(goal_pos[0])

# ============================================
# Statistics
# ============================================

print(f"\n{'='*60}")
print(f"Simulation statistics ({n_steps} steps):")
print(f"  Total time: {simulation_time:.2f} s")
print(f"  Avg time per step: {simulation_time/n_steps*1000:.3f} ms")
print(f"{'='*60}")
print(f"Barrier statistics:")
print(f"  Min backup barrier h(x): {np.min(h_vals_np):.6f}")
print(f"  Min state barrier h_s(x): {np.min(h_s_np):.6f}")
print(f"{'='*60}")
print(f"Control statistics:")
print(f"  u1: min={u_vals_np[:, 0].min():.3f}, max={u_vals_np[:, 0].max():.3f}")
print(f"  u2: min={u_vals_np[:, 1].min():.3f}, max={u_vals_np[:, 1].max():.3f}")
print(f"  Control bounds: u1 in [{control_bounds[0][0]}, {control_bounds[1][0]}], u2 in [{control_bounds[0][1]}, {control_bounds[1][1]}]")
print(f"{'='*60}")
print(f"State statistics:")
print(f"  Velocity: min={traj_np[:, 2].min():.3f}, max={traj_np[:, 2].max():.3f}")
print(f"  Final position: ({traj_np[-1, 0]:.3f}, {traj_np[-1, 1]:.3f})")
print(f"  Distance to goal: {np.linalg.norm(traj_np[-1, :2] - goal_pos_np):.3f}")
print(f"{'='*60}")

# ============================================
# Plots
# ============================================

print("\nGenerating plots...")

current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Create meshgrid for contour plot
x_grid = np.linspace(-10.5, 10.5, 500)
y_grid = np.linspace(-10.5, 10.5, 500)
X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
points = np.column_stack((X_grid.flatten(), Y_grid.flatten()))
points_jax = jnp.array(points)
points_with_zeros = jnp.concatenate([points_jax, jnp.zeros((points_jax.shape[0], 1))], axis=-1)
Z = map_.barrier.min_barrier(points_with_zeros)
Z = np.array(Z).reshape(X_grid.shape)

# Create figs directory
os.makedirs(os.path.join(script_dir, 'figs'), exist_ok=True)

# --- Trajectory Plot ---
fig, ax = plt.subplots(figsize=(8, 8))

contour = ax.contour(X_grid, Y_grid, Z, levels=[0], colors='red', linewidths=2)

ax.plot(traj_np[0, 0], traj_np[0, 1], 'x', color='blue', markersize=10, label=r'$x_0$')
ax.plot(traj_np[-1, 0], traj_np[-1, 1], '+', color='blue', markersize=10, label=r'$x_f$')
ax.plot(traj_np[:, 0], traj_np[:, 1], 'black', linewidth=2, label='Trajectory')
ax.plot(goal_pos_np[0], goal_pos_np[1], '*', markersize=15, color='limegreen', label='Goal')

ax.set_xlabel(r'$q_{\rm x}$', fontsize=14)
ax.set_ylabel(r'$q_{\rm y}$', fontsize=14)
ax.set_aspect('equal', adjustable='box')
ax.legend(loc='upper right', fontsize=12)
ax.grid(True, alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()

plt.savefig(os.path.join(script_dir, f'figs/01_Backup_Trajectory_{current_time}.png'), dpi=300)
plt.show()

# --- States and Controls Plot ---
fig, axs = plt.subplots(6, 1, figsize=(8, 8))

# States
axs[0].plot(time_array, traj_np[:, 0], label=r'$r_x$', color='black')
axs[0].set_ylabel(r'$r_x$', fontsize=14)

axs[1].plot(time_array, traj_np[:, 1], label=r'$r_y$', color='black')
axs[1].set_ylabel(r'$r_y$', fontsize=14)

axs[2].plot(time_array, traj_np[:, 2], label=r'$v$', color='black')
axs[2].set_ylabel(r'$v$', fontsize=14)

axs[3].plot(time_array, traj_np[:, 3], label=r'$\theta$', color='black')
axs[3].set_ylabel(r'$\theta$', fontsize=14)

# Controls with comparisons
axs[4].plot(time_array, u_vals_np[:, 0], label='u', color='black')
axs[4].plot(time_array, info['ub_select'][:, 0], label=r'$u_b$', color='green')
axs[4].plot(time_array, info['u_star'][:, 0], label=r'$u_*$', color='blue')
axs[4].plot(time_array, des_ctrls[:, 0], label=r'$u_d$', color='red', linestyle='--')
axs[4].legend(fontsize=14, loc='upper right', frameon=False, ncol=4)
axs[4].set_ylabel(r'$u_1$', fontsize=14)

axs[5].plot(time_array, u_vals_np[:, 1], label='u', color='black')
axs[5].plot(time_array, info['ub_select'][:, 1], label=r'$u_b$', color='green')
axs[5].plot(time_array, info['u_star'][:, 1], label=r'$u_*$', color='blue')
axs[5].plot(time_array, des_ctrls[:, 1], label=r'$u_d$', color='red', linestyle='--')
axs[5].legend(fontsize=14, loc='upper right', frameon=False, ncol=4)
axs[5].set_ylabel(r'$u_2$', fontsize=14)
axs[5].set_xlabel(r'$t~(\rm {s})$', fontsize=14)

for i in range(5):
    axs[i].tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)

for ax in axs:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=14)
    ax.set_xlim(time_array[0], time_array[-1])

plt.tight_layout()
plt.savefig(os.path.join(script_dir, f'figs/01_Backup_States_{current_time}.png'), dpi=600)
plt.show()

# --- Barrier Values Plot ---
fig, axs = plt.subplots(3, 1, figsize=(8, 8))

# Barrier values
axs[0].plot(time_array, h_vals_np[:, 0], label=r'$h$', color='blue')
axs[0].plot(time_array, h_s_np[:, 0], label=r'$h_s$', color='red', linestyle='--')
axs[0].axhline(y=0, color='green', linestyle='dotted')
axs[0].set_ylabel(r'$h$', fontsize=14)
axs[0].legend(fontsize=14, loc='best', frameon=False)
axs[0].tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)
axs[0].set_xlim(0, sim_time)
axs[0].set_yscale('log')
h_min = np.min(np.concatenate([h_vals_np[:, 0], h_s_np[:, 0]]))
h_max = np.max(np.concatenate([h_vals_np[:, 0], h_s_np[:, 0]]))
axs[0].set_ylim(bottom=h_min * 0.5, top=h_max * 2.0)

# Normalized factors
h_normalized = (h_vals_np[:, 0] - backup_cfg['epsilon']) / backup_cfg['h_scale']
feas_normalized = np.array(info['feas_fact'][:]) / backup_cfg['feas_scale']
axs[1].plot(time_array, h_normalized, label=r'$\frac{h - \epsilon}{\kappa_h}$', color='blue')
axs[1].plot(time_array, feas_normalized, label=r'$\frac{\beta}{\kappa_\beta}$', color='red')
axs[1].axhline(y=0, color='green')
axs[1].set_ylabel(r'$\frac{h - \epsilon}{\kappa_h}, \frac{\beta}{\kappa_\beta}$', fontsize=14)
axs[1].legend(fontsize=14, loc='best', frameon=False)
axs[1].set_xlim(0, sim_time)
axs[1].set_yscale('log')
norm_min = np.min(np.concatenate([h_normalized, feas_normalized]))
norm_max = np.max(np.concatenate([h_normalized, feas_normalized]))
axs[1].set_ylim(bottom=norm_min * 0.5, top=norm_max * 2.0)

# Blending factor (sigma)
axs[2].plot(time_array, info['beta'][:], label=r'$\sigma$', color='blue')
axs[2].set_ylabel(r'$\sigma$', fontsize=14)
axs[2].set_xlabel(r'$t~(\rm {s})$', fontsize=14)
axs[2].set_xlim(0, sim_time)

for ax in axs:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=14)

plt.tight_layout()
plt.savefig(os.path.join(script_dir, f'figs/01_Backup_Barrier_{current_time}.png'), dpi=600)
plt.show()

# --- Animation ---
print("\nCreating animation...")
import matplotlib.animation as animation

def create_animation():
    fig_anim, ax_anim = plt.subplots(figsize=(8, 8))

    frame_indices = np.arange(0, n_steps, 10)

    def animate(frame_idx):
        frame = frame_indices[frame_idx]
        ax_anim.clear()

        current_x = traj_np[frame, 0]
        current_y = traj_np[frame, 1]
        current_v = traj_np[frame, 2]

        past_x = traj_np[:frame + 1, 0]
        past_y = traj_np[:frame + 1, 1]

        ax_anim.contour(X_grid, Y_grid, Z, levels=[0], colors='red', linewidths=2)
        ax_anim.plot(goal_pos_np[0], goal_pos_np[1], '*', markersize=15,
                     color='limegreen', label='Goal', zorder=5)
        ax_anim.plot(past_x, past_y, 'b-', linewidth=2, label='Trajectory', zorder=3)
        ax_anim.scatter([current_x], [current_y], s=100, c='blue', marker='o',
                        edgecolors='black', linewidths=2, label='Current', zorder=4)

        ax_anim.set_xlabel(r'$q_{\rm x}$', fontsize=14)
        ax_anim.set_ylabel(r'$q_{\rm y}$', fontsize=14)
        ax_anim.set_xlim(-10.5, 10.5)
        ax_anim.set_ylim(-10.5, 10.5)
        ax_anim.set_aspect('equal', adjustable='box')
        ax_anim.legend(loc='upper left', fontsize=10)
        ax_anim.spines['right'].set_visible(False)
        ax_anim.spines['top'].set_visible(False)

        current_time_val = frame * timestep
        ax_anim.text(0.98, 0.98,
                     f'Time: {current_time_val:.2f}s\nVel: {current_v:.2f} m/s',
                     transform=ax_anim.transAxes, fontsize=11, verticalalignment='top',
                     horizontalalignment='right',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        return []

    anim = animation.FuncAnimation(fig_anim, animate, frames=len(frame_indices),
                                   interval=50, blit=True)

    animation_file = os.path.join(script_dir, f'figs/01_Backup_Animation_{current_time}.mp4')
    writer = animation.FFMpegWriter(fps=20, metadata=dict(artist='CBFJAX'), bitrate=1800)
    anim.save(animation_file, writer=writer)
    print(f"Animation saved as: {animation_file}")
    plt.show()

create_animation()

print(f"\nPlots saved with timestamp: {current_time}")
print("Simulation complete!")
