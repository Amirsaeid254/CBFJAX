"""
NMPC Safe Control for unicycle dynamics with barrier constraints.

Demonstrates QuadraticNMPCSafeControl with:
- Position barriers (obstacles + boundary) using MultiBarriers
- Velocity constraints via state bounds (not barrier)
- Quadratic tracking cost (Q, R matrices)

Uses acados for NMPC solving with JAX-defined dynamics and barriers
converted to CasADi via jax2casadi.
"""

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
from cbfjax.barriers.multi_barrier import MultiBarriers
from cbfjax.safe_controls.nmpc_safe_control import QuadraticNMPCSafeControl
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

# Barrier configuration
cfg = immutabledict({
    'softmax_rho': 20,
    'softmin_rho': 10,
    'pos_barrier_rel_deg': 1,  # rel_deg=1 for NMPC (no CBF derivative needed)
    'vel_barrier_rel_deg': 1,
    'obstacle_alpha': (),
    'boundary_alpha': (),
    'velocity_alpha': (),
})


# NMPC parameters
nmpc_params = {
    'horizon': 4.0,
    'time_steps': 0.05,
    'qp_solver': 'PARTIAL_CONDENSING_HPIPM',
    'hessian_approx': 'GAUSS_NEWTON',
    'integrator_type': 'ERK',
    'sim_method_num_stages': 4,
    'nlp_solver_type': 'SQP',
    'nlp_solver_max_iter': 200,
    'qp_solver_iter_max': 100,
    'tol': 1e-4,
    'slacked': True,
    'slack_gain_l1': 0.0,
    'slack_gain_l2': 1e6,
    'shift_warm_start': False,
}

# ============================================
# Setup Dynamics and Barriers
# ============================================

print("Setting up dynamics and barriers...")

# Instantiate dynamics
dynamics = UnicycleDynamics()

# Create barrier map and get position barriers only
map_ = Map(barriers_info=map_config, dynamics=dynamics, cfg=cfg).create_barriers()
pos_barriers, _ = map_.get_barriers()

# Create MultiBarriers for position constraints
barrier = MultiBarriers.create_empty(cfg=cfg)
barrier = barrier.add_barriers(pos_barriers, infer_dynamics=True)

print(f"  Number of position barriers: {barrier.num_constraints}")

# ============================================
# Setup NMPC Controller
# ============================================

print("Setting up NMPC controller...")

# State dimensions: [q_x, q_y, v, theta]
nx = dynamics.state_dim  # 4
nu = dynamics.action_dim  # 2

# Cost matrices for tracking (as Callable for consistency)
Q = np.diag([10.0, 10.0, 1.0, 1.0])  # State cost
R = np.diag([0.1, 0.1])               # Control cost
Q_e = 100.0 * Q                        # Terminal cost

# Control bounds: [acceleration, angular velocity]
control_low = [-2.0, -1.0]
control_high = [2.0, 1.0]

# State bounds: velocity constraint (state index 2)
vel_idx, vel_bounds = map_config['velocity']
state_bounds_idx = [vel_idx]
state_low = [vel_bounds[0]]
state_high = [vel_bounds[1]]

# Goal position
goal_pos = np.array([3.0, 4.5])
x_ref = np.array([goal_pos[0], goal_pos[1], 0.0, 0.0])

# Initial state
x0 = np.array([-1.0, -8.5, 0.0, pi / 2])

# Create NMPC controller (cost matrices wrapped as Callable)
controller = (
    QuadraticNMPCSafeControl.create_empty(action_dim=nu, params=nmpc_params)
    .assign_dynamics(dynamics)
    .assign_control_bounds(control_low, control_high)
    .assign_state_bounds(state_bounds_idx, state_low, state_high)
    .assign_cost_matrices(lambda: Q, lambda: R, lambda: Q_e, lambda: x_ref)
    .assign_state_barrier(barrier)
)

# Build the controller
print("Building NMPC solver (this may take a moment)...")
controller = controller.make(x0)
controller.set_init_guess()

# ============================================
# Closed-Loop Simulation
# ============================================

print("\nStarting closed-loop simulation...")

# Simulation parameters
sim_time = 20.0
dt_sim = 0.01  # Control timestep

start_time = time()

# Use get_optimal_trajs_zoh for simulation
trajs, actions = controller.get_optimal_trajs_zoh(
    s0=jnp.array(x0),
    sim_time=sim_time,
    timestep=dt_sim,
    method='euler',
)

simulation_time = time() - start_time
print(f"\nSimulation completed in {simulation_time:.2f} seconds")

# Convert to arrays: trajs is (num_steps, batch, state_dim), actions is (num_steps-1, batch, action_dim)
x_hist = np.array(trajs[:, 0, :])  # (num_steps, state_dim)
u_hist = np.array(actions[:, 0, :])  # (num_steps-1, action_dim)
n_steps = x_hist.shape[0] - 1
time_array = np.linspace(0, sim_time, n_steps + 1)

# Compute predicted trajectories and min barrier along each prediction horizon
print("\nComputing predicted trajectories and barrier values...")
pred_trajs = []  # Predicted trajectory at each step (for animation)
h_pred_min = []  # Min barrier over prediction horizon at each step
for i in range(n_steps):
    # Get predicted trajectory at this state
    x_traj_pred, u_traj_pred = controller.get_predicted_trajectory(jnp.array(x_hist[i]))
    pred_trajs.append(x_traj_pred)

    # Compute barrier along predicted trajectory and take min over time
    h_along_pred = barrier.hocbf(jnp.array(x_traj_pred))  # (N+1, num_barriers)
    h_min = jnp.min(h_along_pred, axis=0)  # min over time -> (num_barriers,)
    h_pred_min.append(np.array(h_min))

    if (i + 1) % 100 == 0:
        print(f"  Step {i + 1}/{n_steps}")

pred_trajs = np.array(pred_trajs)  # (n_steps, N+1, state_dim)
h_pred_min = np.array(h_pred_min)  # (n_steps, num_barriers)

# ============================================
# Statistics
# ============================================

print(f"\n{'='*50}")
print(f"Simulation statistics ({n_steps} steps):")
print(f"  Total time: {simulation_time:.2f} s")
print(f"  Avg time per step: {simulation_time/n_steps*1000:.3f} ms")
print(f"{'='*50}")
print(f"Barrier statistics (min over predicted horizon):")
print(f"  Min h(x): {np.min(h_pred_min):.6f}")
print(f"  Violations (h < 0): {np.sum(np.min(h_pred_min, axis=1) < 0)}")
print(f"{'='*50}")
print(f"Control statistics:")
print(f"  u1 (accel): min={u_hist[:, 0].min():.3f}, max={u_hist[:, 0].max():.3f}")
print(f"  u2 (omega): min={u_hist[:, 1].min():.3f}, max={u_hist[:, 1].max():.3f}")
print(f"{'='*50}")
print(f"State statistics:")
print(f"  Velocity: min={x_hist[:, 2].min():.3f}, max={x_hist[:, 2].max():.3f}")
print(f"  Final position: ({x_hist[-1, 0]:.3f}, {x_hist[-1, 1]:.3f})")
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
ax.plot(x_hist[0, 0], x_hist[0, 1], 'x', color='blue', markersize=8, label=r'$x_0$')
ax.plot(goal_pos[0], goal_pos[1], '*', markersize=10, color='limegreen', label='Goal')
ax.plot(x_hist[-1, 0], x_hist[-1, 1], '+', color='blue', markersize=8, label=r'$x_f$')
ax.plot(x_hist[:, 0], x_hist[:, 1], color='black', label='Trajectory')

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
plt.savefig(os.path.join(script_dir, f'figs/07_NMPC_Trajectory_{current_time}.png'), dpi=200)
plt.show()

# --- States Plot ---
fig, axs = plt.subplots(5, 1, figsize=(8, 8))

# Position
axs[0].plot(time_array, x_hist[:, 0], label=r'$q_{\rm x}$', color='red')
axs[0].plot(time_array, x_hist[:, 1], label=r'$q_{\rm y}$', color='blue')
axs[0].axhline(y=goal_pos[0], color='red', linestyle=':', alpha=0.7)
axs[0].axhline(y=goal_pos[1], color='blue', linestyle=':', alpha=0.7)
axs[0].set_ylabel(r'$q_{\rm x}, q_{\rm y}$', fontsize=16)
axs[0].legend(loc='lower center', ncol=4, frameon=False, fontsize=14)

# Velocity
axs[1].plot(time_array, x_hist[:, 2], color='black')
axs[1].axhline(y=state_low[0], color='gray', linestyle=':', alpha=0.7, label='Bounds')
axs[1].axhline(y=state_high[0], color='gray', linestyle=':', alpha=0.7)
axs[1].set_ylabel(r'$v$', fontsize=16)
axs[1].legend(loc='lower right', frameon=False, fontsize=12)

# Heading
axs[2].plot(time_array, x_hist[:, 3], color='black')
axs[2].set_ylabel(r'$\theta$', fontsize=16)

# Control 1 (acceleration)
axs[3].plot(time_array[:-1], u_hist[:, 0], color='black')
axs[3].axhline(y=control_low[0], color='gray', linestyle=':', alpha=0.7, label='Bounds')
axs[3].axhline(y=control_high[0], color='gray', linestyle=':', alpha=0.7)
axs[3].set_ylabel(r'$u_1$ (a)', fontsize=16)
axs[3].legend(loc='lower right', frameon=False, fontsize=12)

# Control 2 (angular velocity)
axs[4].plot(time_array[:-1], u_hist[:, 1], color='black')
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
plt.savefig(os.path.join(script_dir, f'figs/07_NMPC_States_{current_time}.png'), dpi=200)
plt.show()

# --- Barrier Values Plot (min over predicted horizon) ---
fig, ax = plt.subplots(figsize=(8, 3))

ax.plot(time_array[:-1], np.min(h_pred_min, axis=1), color='black',
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
plt.savefig(os.path.join(script_dir, f'figs/07_NMPC_Barrier_{current_time}.png'), dpi=200)
plt.show()

# --- Animation with predicted trajectories ---
print("\nCreating animation with predicted trajectories...")
import matplotlib.animation as animation

def create_nmpc_animation():
    fig_anim, ax_anim = plt.subplots(figsize=(8, 8))

    def animate(frame):
        ax_anim.clear()

        # Current state
        current_x = x_hist[frame, 0]
        current_y = x_hist[frame, 1]
        current_v = x_hist[frame, 2]

        # Past trajectory
        past_x = x_hist[:frame + 1, 0]
        past_y = x_hist[:frame + 1, 1]

        # Draw static elements (map contour)
        ax_anim.contour(X_grid, Y_grid, Z, levels=[0], colors='red', linewidths=2)

        # Draw goal
        ax_anim.plot(goal_pos[0], goal_pos[1], '*', markersize=15,
                     color='limegreen', label='Goal', zorder=5)

        # Draw past trajectory
        ax_anim.plot(past_x, past_y, 'b-', linewidth=2, label='Past Trajectory', zorder=3)

        # Draw current position
        ax_anim.scatter([current_x], [current_y], s=100, c='blue', marker='o',
                        edgecolors='black', linewidths=2, label='Current Position', zorder=4)

        # Draw predicted trajectory from NMPC
        if frame < len(pred_trajs):
            pred_traj = pred_trajs[frame]  # (N+1, state_dim)
            pred_x = pred_traj[:, 0]
            pred_y = pred_traj[:, 1]
            ax_anim.plot(pred_x, pred_y, 'c--', linewidth=2, alpha=0.8,
                         label='NMPC Prediction', zorder=2)

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
    animation_file = os.path.join(script_dir, f'figs/07_NMPC_Animation_{current_time}.mp4')
    writer = animation.FFMpegWriter(fps=20, metadata=dict(artist='CBFJAX'), bitrate=1800)
    anim.save(animation_file, writer=writer)
    print(f"Animation saved as: {animation_file}")
    plt.show()

create_nmpc_animation()

print(f"\nPlots saved with timestamp: {current_time}")
print("Simulation complete!")