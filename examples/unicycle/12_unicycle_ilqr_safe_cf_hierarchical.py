"""
Hierarchical iLQR-Safe + Closed-Form Safe Control for unicycle dynamics.

Demonstrates a two-layer hierarchical control architecture where BOTH layers
are safety-aware:
1. High-level: QuadraticiLQRSafeControl - iLQR with barrier as AL constraint
   (plans ahead while considering obstacles as soft constraints)
2. Low-level: MinIntervCFSafeControl - CF filter for hard CBF constraint
   enforcement (closed-form solution)

This double-layer safety approach:
- iLQR plans trajectories that proactively avoid barriers (soft penalty)
- CF provides hard safety guarantees via closed-form solution
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
from cbfjax.safe_controls.closed_form_safe_control import MinIntervCFSafeControl
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

# Barrier configuration for iLQR (rel_deg=1 for penalty)
cfg_ilqr = immutabledict({
    'softmax_rho': 20,
    'softmin_rho': 20,
    'pos_barrier_rel_deg': 1,
    'vel_barrier_rel_deg': 1,
    'obstacle_alpha': (),
    'boundary_alpha': (),
    'velocity_alpha': (),
})

# Barrier configuration for CF (needs HOCBF)
cfg_cf = immutabledict({
    'softmax_rho': 20,
    'softmin_rho': 20,
    'pos_barrier_rel_deg': 2,
    'vel_barrier_rel_deg': 1,
    'obstacle_alpha': (10.0,),
    'boundary_alpha': (10.0,),
    'velocity_alpha': (),
})

# iLQR parameters
ilqr_params = {
    'horizon': 4.0,
    'time_steps': 0.05,
    'maxiter': 10,
    'grad_norm_threshold': 1e-4,
    'maxiter_al': 10,
    'constraints_threshold': 1e-4,
    'penalty_init': 1e5,
    'penalty_update_rate': 500.0,
}

# CF safety filter parameters
cf_params = {
    'slack_gain': 1e24,
    'use_softplus': False,
    'softplus_gain': 2.0,
}

# Control bounds for iLQR (CF doesn't enforce these, but iLQR can)
control_low = [-2.0, -1.0]
control_high = [2.0, 1.0]

# ============================================
# Setup Dynamics
# ============================================

print("Setting up dynamics...")

# Dynamics for iLQR (needs discretization)
dynamics_params = {
    'discretization_dt': ilqr_params['time_steps'],
    'discretization_method': 'rk4',
}
dynamics = UnicycleDynamics(params=dynamics_params)

# State/action dimensions
nx = dynamics.state_dim  # 4: [q_x, q_y, v, theta]
nu = dynamics.action_dim  # 2: [acceleration, angular_velocity]

# ============================================
# Setup Barriers
# ============================================

print("Setting up barriers...")

# Create barrier for iLQR (uses barrier as AL inequality constraint)
map_ilqr = Map(barriers_info=map_config, dynamics=dynamics, cfg=cfg_ilqr).create_barriers()
barrier_ilqr = map_ilqr.barrier

# Create barrier for CF (uses composite barrier with HOCBF)
map_cf = Map(barriers_info=map_config, dynamics=dynamics, cfg=cfg_cf).create_barriers()
barrier_cf = map_cf.barrier

print(f"  iLQR barrier: composite barrier as AL inequality constraint")
print(f"  CF barrier: composite barrier for closed-form safety")

# ============================================
# Setup iLQR Safe Controller (High-Level)
# ============================================

print("Setting up iLQR safe controller...")

# Cost matrices for trajectory tracking
Q = jnp.diag(jnp.array([0.01, 0.01, 0.001, 0.001]))   # State cost
R = jnp.diag(jnp.array([0.1, 0.1]))              # Control cost
Q_e = 5.0 * Q                                    # Terminal cost

# Goal position
goal_pos = jnp.array([3.0, 4.5])
x_ref = jnp.array([goal_pos[0], goal_pos[1], 0.0, 0.0])

# Create iLQR controller WITH barrier as AL inequality constraint
ilqr_controller = (
    QuadraticiLQRSafeControl.create_empty(action_dim=nu, params=ilqr_params)
    .assign_dynamics(dynamics)
    .assign_control_bounds(control_low, control_high)
    .assign_cost_matrices(lambda: Q, lambda: R, lambda: Q_e, lambda: x_ref)
    .assign_state_barrier(barrier_ilqr)  # Barrier as AL constraint
)

print(f"  Horizon: {ilqr_controller.horizon}s, N={ilqr_controller.N_horizon}")
print(f"  Barrier: handled as AL inequality constraint")

# ============================================
# Setup CF Safety Filter (Low-Level)
# ============================================

print("Setting up CF safety filter...")

# Create CF safety filter with iLQR-safe as desired control
safety_filter = (
    MinIntervCFSafeControl(
        action_dim=nu,
        alpha=lambda h: 0.5 * h,
        params=cf_params,
    )
    .assign_dynamics(dynamics)
    .assign_state_barrier(barrier_cf)
    .assign_desired_control(ilqr_controller)
)

print(f"  Alpha: 0.5 * h")

# ============================================
# Test Controllers
# ============================================

print("\nTesting controllers...")

# Initial state
x0 = jnp.array([-1.0, -8.5, 0.0, pi / 2])

# Test iLQR-safe alone
print("  Testing iLQR-safe controller...")
u_ilqr_test, _ = ilqr_controller.optimal_control(x0[None], ilqr_controller.get_init_state())
print(f"    iLQR-safe control: u = {np.array(u_ilqr_test[0])}")

# Test CF safety filter
print("  Testing CF safety filter...")
u_safe_test, _ = safety_filter.optimal_control(x0[None], safety_filter.get_init_state())
print(f"    CF-safe control: u = {np.array(u_safe_test[0])}")
print(f"    Intervention: du = {np.array(u_safe_test[0] - u_ilqr_test[0])}")

# ============================================
# Closed-Loop Simulation
# ============================================

print("\nStarting closed-loop simulation...")

# Simulation parameters
sim_time = 20.0
dt_sim = 0.01

x0_batch = x0.reshape(1, -1)  # (1, state_dim)

start_time = time()

# Use safety filter's simulation method
trajs = safety_filter.get_optimal_trajs_zoh(
    x0=x0_batch,
    sim_time=sim_time,
    timestep=dt_sim,
    method='euler'
)

simulation_time = time() - start_time
print(f"Simulation completed in {simulation_time:.2f} seconds")

# Extract trajectory
x_hist = trajs[:, 0, :]  # (time_steps, state_dim)
n_steps = x_hist.shape[0] - 1
time_array = np.linspace(0, sim_time, n_steps + 1)

# ============================================
# Compute Control Actions and Barrier Values
# ============================================

print("\nComputing control actions and analysis data...")

# Thread safety filter state through trajectory via scan (threads iLQR warm-start)
def safe_scan_step(state, x):
    u, new_state, info = safety_filter._optimal_control_single_with_info(x, state)
    return new_state, (u, info)

safe_init_state = safety_filter.get_init_state()
_, (u_safe_hist, safe_info_hist) = jax.lax.scan(safe_scan_step, safe_init_state, x_hist)

# Extract desired control from safety filter info (warm-started iLQR)
u_ilqr_hist = safe_info_hist.u_desired

# Thread iLQR state separately for predicted trajectories (visualization only)
def ilqr_scan_step(state, x):
    u, new_state, info = ilqr_controller._optimal_control_single_with_info(x, state)
    return new_state, (u, info)

ilqr_init_state = ilqr_controller.get_init_state()
_, (_, ilqr_info_hist) = jax.lax.scan(ilqr_scan_step, ilqr_init_state, x_hist)

# Predicted trajectories from iLQR info
pred_trajs_np = np.array(ilqr_info_hist.x_traj)

# Compute barrier values along trajectory
h_vals = barrier_cf.hocbf(x_hist)
min_barrier_vals = barrier_cf.min_barrier(x_hist)

# Compute intervention magnitude
intervention = u_safe_hist - u_ilqr_hist
intervention_norm = jnp.linalg.norm(intervention, axis=1)

# Convert to numpy
x_hist_np = np.array(x_hist)
u_safe_np = np.array(u_safe_hist)
u_ilqr_np = np.array(u_ilqr_hist)
h_vals_np = np.array(h_vals)
min_barrier_np = np.array(min_barrier_vals)
intervention_np = np.array(intervention)
intervention_norm_np = np.array(intervention_norm)
goal_pos_np = np.array(goal_pos)

# ============================================
# Statistics
# ============================================

print(f"\n{'='*60}")
print(f"Simulation statistics ({n_steps} steps):")
print(f"  Total time: {simulation_time:.2f} s")
print(f"  Avg time per step: {simulation_time/n_steps*1000:.3f} ms")
print(f"{'='*60}")
print(f"Barrier statistics (HOCBF values):")
print(f"  Min h(x): {np.min(h_vals_np):.6f}")
print(f"  Min composite barrier: {np.min(min_barrier_np):.6f}")
print(f"{'='*60}")
print(f"Control statistics (CF-safe, actually applied):")
print(f"  u1 (accel): min={u_safe_np[:, 0].min():.3f}, max={u_safe_np[:, 0].max():.3f}")
print(f"  u2 (omega): min={u_safe_np[:, 1].min():.3f}, max={u_safe_np[:, 1].max():.3f}")
print(f"{'='*60}")
print(f"Control statistics (iLQR-safe desired):")
print(f"  u1 (accel): min={u_ilqr_np[:, 0].min():.3f}, max={u_ilqr_np[:, 0].max():.3f}")
print(f"  u2 (omega): min={u_ilqr_np[:, 1].min():.3f}, max={u_ilqr_np[:, 1].max():.3f}")
print(f"{'='*60}")
print(f"Intervention statistics (CF modification of iLQR-safe):")
print(f"  Avg intervention norm: {intervention_norm_np.mean():.4f}")
print(f"  Max intervention norm: {intervention_norm_np.max():.4f}")
print(f"  Steps with intervention > 0.01: {np.sum(intervention_norm_np > 0.01)}/{n_steps}")
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

# Use map barrier for contour plot
Z = barrier_cf.min_barrier(points_jax)
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
plt.savefig(os.path.join(script_dir, f'figs/12_iLQR_Safe_CF_Trajectory_{current_time}.png'), dpi=200)
plt.show()

# --- States and Control Plot ---
fig, axs = plt.subplots(6, 1, figsize=(8, 10))

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

# Control 1 - comparing iLQR-safe vs CF-safe
axs[3].plot(time_array, u_ilqr_np[:, 0], 'r--', alpha=0.7, label=r'$u_{\rm iLQR-safe}$')
axs[3].plot(time_array, u_safe_np[:, 0], 'k-', label=r'$u_{\rm CF-safe}$')
axs[3].set_ylabel(r'$u_1$ (a)', fontsize=16)
axs[3].legend(loc='lower center', ncol=3, frameon=False, fontsize=12)

# Control 2
axs[4].plot(time_array, u_ilqr_np[:, 1], 'r--', alpha=0.7, label=r'$u_{\rm iLQR-safe}$')
axs[4].plot(time_array, u_safe_np[:, 1], 'k-', label=r'$u_{\rm CF-safe}$')
axs[4].set_ylabel(r'$u_2$ ($\omega$)', fontsize=16)
axs[4].legend(loc='lower center', ncol=3, frameon=False, fontsize=12)

# Intervention magnitude
axs[5].plot(time_array, intervention_norm_np, color='purple')
axs[5].set_ylabel(r'$\|u_{\rm CF} - u_{\rm iLQR}\|$', fontsize=16)
axs[5].set_xlabel(r'$t~(\rm {s})$', fontsize=16)

for i in range(5):
    axs[i].tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)

for ax in axs:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=14)
    ax.set_xlim(time_array[0], time_array[-1])

plt.subplots_adjust(wspace=0, hspace=0.2)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, f'figs/12_iLQR_Safe_CF_States_{current_time}.png'), dpi=200)
plt.show()

# --- Barrier Values Plot ---
fig, axs = plt.subplots(2, 1, figsize=(8, 4))

# HOCBF value
axs[0].plot(time_array, h_vals_np, color='black')
axs[0].axhline(y=0, color='red', linestyle='--', linewidth=1.5)
axs[0].set_ylabel(r'$h(x)$', fontsize=16)

# Min barrier (composite)
axs[1].plot(time_array, min_barrier_np, color='black')
axs[1].axhline(y=0, color='red', linestyle='--', linewidth=1.5, label='Constraint')
axs[1].set_ylabel(r'$\min_i b_i(x)$', fontsize=16)
axs[1].set_xlabel(r'$t~(\rm {s})$', fontsize=16)
axs[1].legend(loc='lower right', frameon=False, fontsize=12)

axs[0].tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)

for ax in axs:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=14)
    ax.set_xlim(time_array[0], time_array[-1])

plt.tight_layout()
plt.savefig(os.path.join(script_dir, f'figs/12_iLQR_Safe_CF_Barrier_{current_time}.png'), dpi=200)
plt.show()

# --- Animation ---
print("\nCreating animation...")
import matplotlib.animation as animation

def create_animation():
    fig_anim, ax_anim = plt.subplots(figsize=(8, 8))

    # Sample frames for animation (every 10 steps)
    frame_indices = np.arange(0, n_steps, 10)

    def animate(frame_idx):
        frame = frame_indices[frame_idx]
        ax_anim.clear()

        # Current state
        current_x = x_hist_np[frame, 0]
        current_y = x_hist_np[frame, 1]
        current_v = x_hist_np[frame, 2]

        # Past trajectory
        past_x = x_hist_np[:frame + 1, 0]
        past_y = x_hist_np[:frame + 1, 1]

        # Draw map contour
        ax_anim.contour(X_grid, Y_grid, Z, levels=[0], colors='red', linewidths=2)

        # Draw goal
        ax_anim.plot(goal_pos_np[0], goal_pos_np[1], '*', markersize=15,
                     color='limegreen', label='Goal', zorder=5)

        # Draw past trajectory
        ax_anim.plot(past_x, past_y, 'b-', linewidth=2, label='Trajectory', zorder=3)

        # Draw current position
        ax_anim.scatter([current_x], [current_y], s=100, c='blue', marker='o',
                        edgecolors='black', linewidths=2, label='Current', zorder=4)

        # Draw iLQR predicted trajectory
        pred_traj = pred_trajs_np[frame]
        ax_anim.plot(pred_traj[:, 0], pred_traj[:, 1], 'c--', linewidth=2,
                     alpha=0.6, label='iLQR Prediction', zorder=2)

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
        intervention_val = intervention_norm_np[frame] if frame < len(intervention_norm_np) else 0
        ax_anim.text(0.98, 0.98,
                     f'Time: {current_time_val:.2f}s\n'
                     f'Vel: {current_v:.2f} m/s\n'
                     f'Intervention: {intervention_val:.3f}',
                     transform=ax_anim.transAxes, fontsize=11, verticalalignment='top',
                     horizontalalignment='right',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        return []

    # Create animation
    anim = animation.FuncAnimation(fig_anim, animate, frames=len(frame_indices),
                                   interval=50, blit=True)

    # Save animation
    animation_file = os.path.join(script_dir, f'figs/12_iLQR_Safe_CF_Animation_{current_time}.mp4')
    writer = animation.FFMpegWriter(fps=20, metadata=dict(artist='CBFJAX'), bitrate=1800)
    anim.save(animation_file, writer=writer)
    print(f"Animation saved as: {animation_file}")
    plt.show()

create_animation()

print(f"\nPlots saved with timestamp: {current_time}")
print("Simulation complete!")
