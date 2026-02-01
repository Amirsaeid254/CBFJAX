"""
Minimum intervention closed-form safe control for unicycle dynamics.

Demonstrates:
- Composite barrier function creation via Map
- MinIntervCFSafeControl for safety filtering
- Closed-form CBF solution (no QP needed)
"""

import jax
import jax.numpy as jnp
import matplotlib as mpl
from math import pi
import numpy as np
from time import time
import datetime
import os
from immutabledict import immutabledict

# CBFJAX imports
import cbfjax.config
from cbfjax.dynamics.unicycle import UnicycleDynamics
from cbfjax.utils.make_map import Map
from cbfjax.safe_controls.closed_form_safe_control import MinIntervCFSafeControl
from map_config import map_config
from unicycle_desired_control import desired_control

# Get script directory for saving figures
script_dir = os.path.dirname(os.path.abspath(__file__))
import matplotlib.pyplot as plt

mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'

# ============================================
# Configuration
# ============================================

# Barrier configuration
cfg = immutabledict({
    'softmax_rho': 20,
    'softmin_rho': 20,
    'pos_barrier_rel_deg': 2,
    'vel_barrier_rel_deg': 1,
    'obstacle_alpha': (10.0,),
    'boundary_alpha': (10.0,),
    'velocity_alpha': (),
})

# CF safety filter parameters
cf_params = {
    'slack_gain': 1e24,
    'use_softplus': False,
    'softplus_gain': 2.0,
}

# Control gains for desired control
control_gains = immutabledict(k1=0.2, k2=1.0, k3=2.0)

# Goal position
goal_pos = jnp.array([[3.0, 4.5]])

# Initial condition
x0 = jnp.array([-1.0, -8.5, 0.0, pi / 2])

# Simulation parameters
sim_time = 10.0
dt_sim = 0.01

# ============================================
# Setup Dynamics
# ============================================

print("Setting up dynamics...")

dynamics = UnicycleDynamics()
nx = dynamics.state_dim  # 4: [q_x, q_y, v, theta]
nu = dynamics.action_dim  # 2: [acceleration, angular_velocity]

print(f"  State dim: {nx}, Action dim: {nu}")

# ============================================
# Setup Barriers
# ============================================

print("Setting up barriers...")

map_ = Map(barriers_info=map_config, dynamics=dynamics, cfg=cfg).create_barriers()
barrier = map_.barrier

print(f"  Barrier setup complete")

# ============================================
# Setup CF Safety Filter
# ============================================

print("Setting up CF safety filter...")

safety_filter = (
    MinIntervCFSafeControl(
        action_dim=nu,
        alpha=lambda x: 0.5 * x,
        params=cf_params,
    )
    .assign_dynamics(dynamics)
    .assign_state_barrier(barrier)
    .assign_desired_control(lambda x: desired_control(x, goal_pos))
)

print(f"  Alpha: 0.5 * h")

# ============================================
# Test Controller
# ============================================

print("\nTesting controller...")

u_test, _ = safety_filter.optimal_control(x0[None], safety_filter.get_init_state())
print(f"  Test control: u = {np.array(u_test[0])}")

# ============================================
# Closed-Loop Simulation
# ============================================

print("\nStarting closed-loop simulation...")
print(f"  Device: {jax.devices()[0]}")

x0_batch = x0.reshape(1, -1)

start_time = time()
trajs = safety_filter.get_optimal_trajs_zoh(
    x0=x0_batch,
    sim_time=sim_time,
    timestep=dt_sim,
    method='euler'
)
simulation_time = time() - start_time

print(f"Simulation completed in {simulation_time:.2f} seconds")

# ============================================
# Compute Control Actions and Barrier Values
# ============================================

print("\nComputing control actions and barrier values...")

# Extract trajectory
x_hist = trajs[:, 0, :]  # (time_steps, state_dim)
n_steps = x_hist.shape[0] - 1
time_array = np.linspace(0, sim_time, n_steps + 1)

# Compute controls and info
u_hist, _, info_hist = safety_filter.optimal_control_with_info(x_hist, safety_filter.get_init_state())

# Compute barrier values
h_vals = barrier.hocbf(x_hist)
min_barrier_vals = barrier.min_barrier(x_hist)

# Convert to numpy
x_hist_np = np.array(x_hist)
u_hist_np = np.array(u_hist)
h_vals_np = np.array(h_vals)
min_barrier_np = np.array(min_barrier_vals)
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
print(f"  Min h(x): {np.min(h_vals_np):.6f}")
print(f"  Min composite barrier: {np.min(min_barrier_np):.6f}")
print(f"{'='*60}")
print(f"Control statistics:")
print(f"  u1: min={u_hist_np[:, 0].min():.3f}, max={u_hist_np[:, 0].max():.3f}")
print(f"  u2: min={u_hist_np[:, 1].min():.3f}, max={u_hist_np[:, 1].max():.3f}")
print(f"{'='*60}")
print(f"State statistics:")
print(f"  Velocity: min={x_hist_np[:, 2].min():.3f}, max={x_hist_np[:, 2].max():.3f}")
print(f"  Final position: ({x_hist_np[-1, 0]:.3f}, {x_hist_np[-1, 1]:.3f})")
print(f"  Distance to goal: {np.linalg.norm(x_hist_np[-1, :2] - goal_pos_np):.3f}")
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
points_with_vel = np.column_stack((points, np.zeros((points.shape[0], 2))))
points_jax = jnp.array(points_with_vel, dtype=jnp.float32)

Z = barrier.min_barrier(points_jax)
Z = np.array(Z).reshape(X_grid.shape)

# --- Trajectory Plot ---
fig, ax = plt.subplots(figsize=(6, 6))

contour_plot = ax.contour(X_grid, Y_grid, Z, levels=[0], colors='red')

ax.set_xlabel(r'$q_{\rm x}$', fontsize=16)
ax.set_ylabel(r'$q_{\rm y}$', fontsize=16)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_aspect('equal', adjustable='box')
ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=16)
ax.set_xticks([-10, -5, 0, 5, 10])
ax.set_yticks([-10, -5, 0, 5, 10])

# Plot trajectory
ax.plot(x_hist_np[0, 0], x_hist_np[0, 1], 'x', color='blue', markersize=8, label=r'$x_0$')
ax.plot(goal_pos_np[0], goal_pos_np[1], '*', markersize=10, color='limegreen', label='Goal')
ax.plot(x_hist_np[-1, 0], x_hist_np[-1, 1], '+', color='blue', markersize=8, label=r'$x_f$')
ax.plot(x_hist_np[:, 0], x_hist_np[:, 1], color='black', label='Trajectory')

# Legend
from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color='red', lw=1.5)]
handles, labels = ax.get_legend_handles_labels()
handles.insert(0, custom_lines[0])
labels.insert(0, r'$\mathcal{S}_{\rm s}$')
ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.12),
          ncol=3, frameon=False, fontsize=12)

plt.tight_layout()
os.makedirs(os.path.join(script_dir, 'figs'), exist_ok=True)
plt.savefig(os.path.join(script_dir, f'figs/02_CF_Trajectory_{current_time}.png'), dpi=200)
plt.show()

# --- States and Control Plot ---
fig, axs = plt.subplots(5, 1, figsize=(8, 8))

# Position
axs[0].plot(time_array, x_hist_np[:, 0], label=r'$q_{\rm x}$', color='red')
axs[0].plot(time_array, x_hist_np[:, 1], label=r'$q_{\rm y}$', color='blue')
axs[0].axhline(y=goal_pos_np[0], color='red', linestyle=':', alpha=0.7)
axs[0].axhline(y=goal_pos_np[1], color='blue', linestyle=':', alpha=0.7)
axs[0].set_ylabel(r'$q_{\rm x}, q_{\rm y}$', fontsize=16)
axs[0].legend(loc='lower center', ncol=4, frameon=False, fontsize=14)

# Velocity
axs[1].plot(time_array, x_hist_np[:, 2], color='black')
axs[1].set_ylabel(r'$v$', fontsize=16)

# Heading
axs[2].plot(time_array, x_hist_np[:, 3], color='black')
axs[2].set_ylabel(r'$\theta$', fontsize=16)

# Control 1
axs[3].plot(time_array, u_hist_np[:, 0], color='black')
axs[3].set_ylabel(r'$u_1$ (a)', fontsize=16)

# Control 2
axs[4].plot(time_array, u_hist_np[:, 1], color='black')
axs[4].set_ylabel(r'$u_2$ ($\omega$)', fontsize=16)
axs[4].set_xlabel(r'$t~(\rm {s})$', fontsize=16)

for i in range(4):
    axs[i].tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)

for ax in axs:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=16)
    ax.set_xlim(time_array[0], time_array[-1])

plt.subplots_adjust(wspace=0, hspace=0.2)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, f'figs/02_CF_States_{current_time}.png'), dpi=200)
plt.show()

# --- Barrier Values Plot ---
fig, axs = plt.subplots(2, 1, figsize=(8, 4))

axs[0].plot(time_array, h_vals_np, color='black')
axs[0].axhline(y=0, color='red', linestyle='--', linewidth=1.5)
axs[0].set_ylabel(r'$h(x)$', fontsize=16)

axs[1].plot(time_array, min_barrier_np, color='black')
axs[1].axhline(y=0, color='red', linestyle='--', linewidth=1.5, label='Constraint')
axs[1].set_ylabel(r'$\min_i b_i(x)$', fontsize=16)
axs[1].set_xlabel(r'$t~(\rm {s})$', fontsize=16)
axs[1].legend(loc='lower right', frameon=False, fontsize=12)

axs[0].tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)

for ax in axs:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=16)
    ax.set_xlim(time_array[0], time_array[-1])

plt.tight_layout()
plt.savefig(os.path.join(script_dir, f'figs/02_CF_Barrier_{current_time}.png'), dpi=200)
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

        current_x = x_hist_np[frame, 0]
        current_y = x_hist_np[frame, 1]
        current_v = x_hist_np[frame, 2]

        past_x = x_hist_np[:frame + 1, 0]
        past_y = x_hist_np[:frame + 1, 1]

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

        current_time_val = frame * dt_sim
        ax_anim.text(0.98, 0.98,
                     f'Time: {current_time_val:.2f}s\nVel: {current_v:.2f} m/s',
                     transform=ax_anim.transAxes, fontsize=11, verticalalignment='top',
                     horizontalalignment='right',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        return []

    anim = animation.FuncAnimation(fig_anim, animate, frames=len(frame_indices),
                                   interval=50, blit=True)

    animation_file = os.path.join(script_dir, f'figs/02_CF_Animation_{current_time}.mp4')
    writer = animation.FFMpegWriter(fps=20, metadata=dict(artist='CBFJAX'), bitrate=1800)
    anim.save(animation_file, writer=writer)
    print(f"Animation saved as: {animation_file}")
    plt.show()

create_animation()

print(f"\nPlots saved with timestamp: {current_time}")
print("Simulation complete!")
