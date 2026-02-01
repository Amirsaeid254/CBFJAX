"""
iLQR Safe Control for unicycle dynamics with barrier constraints.

Demonstrates QuadraticiLQRSafeControl with:
- Barrier as AL inequality constraint (h(x) >= 0 -> -h(x) <= 0)
- Optional log barrier penalty for smooth gradient repulsion
- Uses map_.barrier directly (all barriers combined)
- Quadratic tracking cost (Q, R matrices)
- Control bounds via constrained iLQR

Uses trajax for iLQR solving.
"""

import jax
import jax.numpy as jnp
import matplotlib as mpl
from math import pi
import numpy as np
from time import time
import datetime
import os

# CBFJAX imports
import cbfjax.config
from cbfjax.dynamics.unicycle import UnicycleDynamics
from cbfjax.utils.make_map import Map
from cbfjax.safe_controls.ilqr_safe_control import QuadraticiLQRSafeControl
from immutabledict import immutabledict

# Local imports
from map_config import map_config

# Get script directory for saving figures
script_dir = os.path.dirname(os.path.abspath(__file__))
import matplotlib.pyplot as plt

mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'

# ============================================
# Configuration
# ============================================

# Barrier configuration (rel_deg=1 since we use barrier as penalty, not CBF)
cfg = immutabledict({
    'softmax_rho': 20,
    'softmin_rho': 20,
    'pos_barrier_rel_deg': 1,
    'vel_barrier_rel_deg': 1,
    'obstacle_alpha': (),
    'boundary_alpha': (),
    'velocity_alpha': (),
})

# iLQR parameters
ilqr_params = {
    'horizon': 4.0,
    'time_steps': 0.05,
    'maxiter': 20,
    'grad_norm_threshold': 1e-4,
    'maxiter_al': 20,
    'constraints_threshold': 1e-4,
    'penalty_init': 1e5,
    'penalty_update_rate': 500.0,
}

# ============================================
# Setup Dynamics and Barriers
# ============================================



print("Setting up dynamics and barriers...")

dynamics_params = {
    'discretization_dt': ilqr_params['time_steps'],
    'discretization_method': 'rk4',
}
dynamics = UnicycleDynamics(params=dynamics_params)

# Create barrier map
map_ = Map(barriers_info=map_config, dynamics=dynamics, cfg=cfg).create_barriers()
barrier = map_.barrier

print(f"  Barrier setup complete")

# ============================================
# Setup iLQR Controller
# ============================================

print("Setting up iLQR controller...")

# State dimensions: [q_x, q_y, v, theta]
nx = dynamics.state_dim  # 4
nu = dynamics.action_dim  # 2

# Cost matrices for tracking (as Callable for JIT compatibility)
Q = jnp.diag(jnp.array([0.01, 0.01, 0.001, 0.001]))  # State cost
R = jnp.diag(jnp.array([0.1, 0.1]))               # Control cost
Q_e = 5.0 * Q                                    # Terminal cost

# Control bounds: [acceleration, angular velocity]
control_low = [-2.0, -1.0]
control_high = [2.0, 1.0]

# Goal position
goal_pos = jnp.array([3.0, 4.5])
x_ref = jnp.array([goal_pos[0], goal_pos[1], 0.0, 0.0])

# Initial state
x0 = jnp.array([-1.0, -8.5, 0.0, pi / 2])

# Create iLQR controller (cost matrices wrapped as Callable)
controller = (
    QuadraticiLQRSafeControl.create_empty(action_dim=nu, params=ilqr_params)
    .assign_dynamics(dynamics)
    .assign_control_bounds(control_low, control_high)
    .assign_cost_matrices(lambda: Q, lambda: R, lambda: Q_e, lambda: x_ref)
    .assign_state_barrier(barrier)
)

print(f"  Horizon: {controller.horizon}s, N={controller.N_horizon}")

# ============================================
# Closed-Loop Simulation
# ============================================

print("\nStarting closed-loop simulation...")

# Simulation parameters
sim_time = 20.0
dt_sim = 0.01

start_time = time()

# Use get_optimal_trajs for simulation
trajs = controller.get_optimal_trajs_zoh(
    x0=x0,
    sim_time=sim_time,
    timestep=dt_sim,
    method='tsit5'
)

simulation_time = time() - start_time
print(f"\nSimulation completed in {simulation_time:.2f} seconds")

# trajs shape: (time_steps, batch, state_dim) -> squeeze batch dim
x_hist = trajs[:, 0, :]  # (time_steps+1, state_dim)
n_steps = x_hist.shape[0] - 1
time_array = np.linspace(0, sim_time, n_steps + 1)

# Compute controls, state, and info along trajectory using lax.scan
print("\nComputing control actions and predicted trajectories...")

def scan_step(state, x):
    u, new_state, info = controller._optimal_control_single_with_info(x, state)
    return new_state, (u, info)

init_state = controller.get_init_state()
_, (u_hist, info_hist) = jax.lax.scan(scan_step, init_state, x_hist)


# Compute min barrier along each predicted trajectory
print("Computing barrier values along predicted trajectories...")
pred_trajs = info_hist.x_traj  # (n_steps+1, N+1, state_dim)
h_pred = jax.vmap(barrier.hocbf)(pred_trajs)  # (n_steps+1, N+1, num_barriers)
h_pred_min = jnp.min(h_pred, axis=1)  # (n_steps+1, num_barriers)

# ============================================
# Statistics
# ============================================

# Convert to numpy for plotting and statistics
x_hist_np = np.array(x_hist)
u_hist_np = np.array(u_hist)
h_pred_min_np = np.array(h_pred_min)
pred_trajs_np = np.array(pred_trajs)
goal_pos_np = np.array(goal_pos)

print(f"\n{'='*50}")
print(f"Simulation statistics ({n_steps} steps):")
print(f"  Total time: {simulation_time:.2f} s")
print(f"  Avg time per step: {simulation_time/n_steps*1000:.3f} ms")
print(f"{'='*50}")
print(f"Barrier statistics (min over predicted horizon):")
print(f"  Min h(x): {np.min(h_pred_min_np):.6f}")
print(f"  Violations (h < 0): {np.sum(np.min(h_pred_min_np, axis=1) < 0)}")
print(f"{'='*50}")
print(f"Control statistics:")
print(f"  u1 (accel): min={u_hist_np[:, 0].min():.3f}, max={u_hist_np[:, 0].max():.3f}")
print(f"  u2 (omega): min={u_hist_np[:, 1].min():.3f}, max={u_hist_np[:, 1].max():.3f}")
print(f"{'='*50}")
print(f"State statistics:")
print(f"  Velocity: min={x_hist_np[:, 2].min():.3f}, max={x_hist_np[:, 2].max():.3f}")
print(f"  Final position: ({x_hist_np[-1, 0]:.3f}, {x_hist_np[-1, 1]:.3f})")
print(f"{'='*50}")

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

# Use map barrier for contour plot
Z = map_.barrier.min_barrier(points_jax)
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
plt.savefig(os.path.join(script_dir, f'figs/08_iLQR_Trajectory_{current_time}.png'), dpi=200)
plt.show()

# --- States Plot ---
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

# Control 1 (acceleration)
axs[3].plot(time_array, u_hist_np[:, 0], color='black')
axs[3].axhline(y=control_low[0], color='gray', linestyle=':', alpha=0.7, label='Bounds')
axs[3].axhline(y=control_high[0], color='gray', linestyle=':', alpha=0.7)
axs[3].set_ylabel(r'$u_1$ (a)', fontsize=16)
axs[3].legend(loc='lower right', frameon=False, fontsize=12)

# Control 2 (angular velocity)
axs[4].plot(time_array, u_hist_np[:, 1], color='black')
axs[4].axhline(y=control_low[1], color='gray', linestyle=':', alpha=0.7, label='Bounds')
axs[4].axhline(y=control_high[1], color='gray', linestyle=':', alpha=0.7)
axs[4].set_ylabel(r'$u_2$ ($\omega$)', fontsize=16)
axs[4].legend(loc='lower right', frameon=False, fontsize=12)

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
plt.savefig(os.path.join(script_dir, f'figs/08_iLQR_States_{current_time}.png'), dpi=200)
plt.show()

# --- Barrier Values Plot (min over predicted horizon) ---
fig, ax = plt.subplots(figsize=(8, 3))

ax.plot(time_array, np.min(h_pred_min_np, axis=1), color='black',
        label=r'$\min_{t \in [0,T]} \min_i h_i(x(t))$')
ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5, label='Constraint boundary')
ax.set_xlabel(r'$t~(\rm {s})$', fontsize=16)
ax.set_ylabel(r'$h(x)$', fontsize=16)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(labelsize=16)
ax.set_xlim(time_array[0], time_array[-1])
ax.legend(loc='lower right', frameon=False, fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(script_dir, f'figs/08_iLQR_Barrier_{current_time}.png'), dpi=200)
plt.show()

# --- Animation with predicted trajectories ---
print("\nCreating animation with predicted trajectories...")
import matplotlib.animation as animation

def create_ilqr_animation():
    fig_anim, ax_anim = plt.subplots(figsize=(8, 8))

    def animate(frame):
        ax_anim.clear()

        # Current state
        current_x = x_hist_np[frame, 0]
        current_y = x_hist_np[frame, 1]
        current_v = x_hist_np[frame, 2]

        # Past trajectory
        past_x = x_hist_np[:frame + 1, 0]
        past_y = x_hist_np[:frame + 1, 1]

        # Draw static elements (map contour)
        ax_anim.contour(X_grid, Y_grid, Z, levels=[0], colors='red', linewidths=2)

        # Draw goal
        ax_anim.plot(goal_pos_np[0], goal_pos_np[1], '*', markersize=15,
                     color='limegreen', label='Goal', zorder=5)

        # Draw past trajectory
        ax_anim.plot(past_x, past_y, 'b-', linewidth=2, label='Past Trajectory', zorder=3)

        # Draw current position
        ax_anim.scatter([current_x], [current_y], s=100, c='blue', marker='o',
                        edgecolors='black', linewidths=2, label='Current Position', zorder=4)

        # Draw predicted trajectory from iLQR
        if frame < pred_trajs_np.shape[0]:
            pred_traj = pred_trajs_np[frame]  # (N+1, state_dim)
            pred_x = pred_traj[:, 0]
            pred_y = pred_traj[:, 1]
            ax_anim.plot(pred_x, pred_y, 'c--', linewidth=2, alpha=0.8,
                         label='iLQR Prediction', zorder=2)

            # Mark prediction horizon points
            sample_indices = np.arange(0, len(pred_x), max(1, len(pred_x) // 10))
            ax_anim.scatter(pred_x[sample_indices], pred_y[sample_indices],
                            s=30, c='cyan', alpha=0.6, zorder=2)

        # Set plot properties
        ax_anim.set_xlabel(r'$q_{\rm x}$', fontsize=14)
        ax_anim.set_ylabel(r'$q_{\rm y}$', fontsize=14)
        ax_anim.set_xlim(-10.5, 10.5)
        ax_anim.set_ylim(-10.5, 10.5)
        ax_anim.set_aspect('equal', adjustable='box')
        ax_anim.legend(loc='upper left', fontsize=10)
        ax_anim.spines['right'].set_visible(False)
        ax_anim.spines['top'].set_visible(False)

        # Info text
        current_time_val = frame * dt_sim
        ax_anim.text(0.98, 0.98,
                     f'Time: {current_time_val:.2f}s\nVel: {current_v:.2f} m/s',
                     transform=ax_anim.transAxes, fontsize=12, verticalalignment='top',
                     horizontalalignment='right',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        return []

    # Create animation
    anim = animation.FuncAnimation(fig_anim, animate, frames=n_steps,
                                   interval=50, blit=True)

    # Save animation
    animation_file = os.path.join(script_dir, f'figs/08_iLQR_Animation_{current_time}.mp4')
    writer = animation.FFMpegWriter(fps=20, metadata=dict(artist='CBFJAX'), bitrate=1800)
    anim.save(animation_file, writer=writer)
    print(f"Animation saved as: {animation_file}")
    plt.show()

create_ilqr_animation()

print(f"\nPlots saved with timestamp: {current_time}")
print("Simulation complete!")