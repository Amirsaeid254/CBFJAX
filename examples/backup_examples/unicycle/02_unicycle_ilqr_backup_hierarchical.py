"""
Hierarchical iLQR + Backup Safe Control for unicycle dynamics.

Demonstrates a two-layer hierarchical control architecture:
1. High-level: QuadraticiLQRControl computes optimal trajectory-tracking control
2. Low-level: MinIntervBackupSafeControl filters the iLQR control
              using backup barriers and smooth blending

This approach combines:
- iLQR's ability to plan ahead and track references
- Backup safe control's guarantees via backup trajectories and blending
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from time import time
import datetime
from immutabledict import immutabledict

# CBFJAX imports
import cbfjax.config
from cbfjax.dynamics import UnicycleReducedOrderDynamics
from cbfjax.barriers import Barrier, BackupBarrier
from cbfjax.safe_controls import MinIntervBackupSafeControl
from cbfjax.controls.ilqr_control import QuadraticiLQRControl
from cbfjax.utils.make_map import Map

# Local imports
from map_config import map_config
from backup_policies import UnicycleBackupControl

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

# Map configuration for state barriers
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

# iLQR parameters (no barrier - backup handles safety)
ilqr_params = {
    'horizon': 4.0,
    'time_steps': 0.05,
    'maxiter': 10,
    'grad_norm_threshold': 1e-4,
}

# Backup policy parameters
ub_gain = ((-15.0, 0.0),)

# Goal position
goal_pos = jnp.array([2.0, 4.5])
x_ref = jnp.array([goal_pos[0], goal_pos[1], 0.0, 0.0])

# Initial condition
x0 = jnp.array([-3.0, -8.5, 0.0, 0.0])

# Simulation parameters
timestep = 0.01
sim_time = 20.0

# ============================================================================
# Setup Dynamics
# ============================================================================

print("Setting up dynamics...")

# Dynamics for iLQR (needs discretization)
ilqr_dynamics_params = {
    'd': dynamics_params['d'],
    'control_bounds': control_bounds,
    'discretization_dt': ilqr_params['time_steps'],
    'discretization_method': 'rk4',
}
dynamics = UnicycleReducedOrderDynamics(params=ilqr_dynamics_params)

nx = dynamics.state_dim  # 4: [x, y, v, theta]
nu = dynamics.action_dim  # 2: [accel, omega]

print(f"  - Dynamics: UnicycleReducedOrderDynamics")

# ============================================================================
# Setup Barriers
# ============================================================================

print("Setting up barriers...")

# 1. Create map with state barriers
map_ = Map(dynamics=dynamics, cfg=map_cfg, barriers_info=map_config).create_barriers()
state_barrier = map_.barrier
print(f"  - State barrier: {len(map_.barrier._barriers)} obstacle/boundary barriers")

# 2. Create backup policies
backup_controls = UnicycleBackupControl(ub_gain, control_bounds)()
print(f"  - Backup policies: {len(backup_controls)} policies")

# 3. Create backup barriers
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

# 4. Create backup barrier system
fwd_barrier = (BackupBarrier.create_empty(cfg=backup_cfg)
               .assign_state_barrier([state_barrier])
               .assign_backup_policies(backup_controls)
               .assign_backup_barrier(backup_barriers)
               .assign_dynamics(dynamics)
               .make())
print("  - Backup barrier system built")

# ============================================================================
# Setup iLQR Controller (High-Level)
# ============================================================================

print("\nSetting up iLQR controller...")

# Cost matrices for trajectory tracking
Q = jnp.diag(jnp.array([10.0, 10.0, 1.0, 1.0]))   # State cost
R = jnp.diag(jnp.array([10.0, 10.0]))              # Control cost
Q_e = 100.0 * Q                                    # Terminal cost

# Create unconstrained iLQR controller (no barrier awareness)
ilqr_controller = (
    QuadraticiLQRControl.create_empty(action_dim=nu, params=ilqr_params)
    .assign_dynamics(dynamics)
    .assign_cost_matrices(Q, R, Q_e, x_ref)
)

print(f"  - Horizon: {ilqr_controller.horizon}s, N={ilqr_controller.N_horizon}")

# ============================================================================
# Setup Backup Safety Filter (Low-Level)
# ============================================================================

print("Setting up backup safety filter...")

# Create function that wraps iLQR controller as desired control
def ilqr_desired_control(x: jnp.ndarray) -> jnp.ndarray:
    """Compute iLQR optimal control for state x."""
    u, _ = ilqr_controller._optimal_control_single(x)
    return u

# Create backup safety filter with iLQR as desired control
safety_filter = MinIntervBackupSafeControl(
    action_dim=nu,
    alpha=lambda x: 1.0 * x,
    slacked=False,
    control_low=list(control_bounds[0]),
    control_high=list(control_bounds[1])
).assign_dynamics(dynamics).assign_state_barrier(fwd_barrier)

safety_filter = safety_filter.assign_desired_control(ilqr_desired_control)

print(f"  - Control bounds: low={control_bounds[0]}, high={control_bounds[1]}")

# ============================================================================
# Test Controllers
# ============================================================================

print("\nTesting controllers...")

# Test iLQR alone
print("  Testing iLQR controller...")
u_ilqr_test, info_ilqr = ilqr_controller.optimal_control(x0)
print(f"    iLQR control: u = {np.array(u_ilqr_test)}")

# Test backup safety filter
print("  Testing backup safety filter...")
u_safe_test, info_backup = safety_filter._optimal_control_single(x0)
print(f"    Backup-safe control: u = {np.array(u_safe_test)}")
print(f"    Intervention: du = {np.array(u_safe_test - u_ilqr_test)}")

# ============================================================================
# Simulation
# ============================================================================

print("\nRunning simulation...")

x0_batch = x0.reshape(1, -1)

start_time = time()
trajs = safety_filter.get_optimal_trajs_zoh(
    x0=x0_batch,
    sim_time=sim_time,
    timestep=timestep,
    method='dopri5'
)
elapsed = time() - start_time
print(f"  - Simulation completed in {elapsed:.2f}s")

# ============================================================================
# Analysis
# ============================================================================

print("\nAnalyzing trajectory...")

traj = trajs[:, 0, :]  # (time_steps, state_dim)
n_steps = traj.shape[0] - 1
time_array = np.linspace(0.0, sim_time, n_steps + 1)

# Get safe controls and info
u_safe, info = safety_filter.optimal_control(traj, ret_info=True)

# Get iLQR controls (what iLQR would have applied)
u_ilqr, _ = ilqr_controller.optimal_control(traj)

# Compute intervention
intervention = u_safe - u_ilqr
intervention_norm = np.array(jnp.linalg.norm(intervention, axis=1))

# Barrier values
h_vals = fwd_barrier.hocbf(traj)
h_s = state_barrier.hocbf(traj)

# Convert to numpy
traj_np = np.array(traj)
u_safe_np = np.array(u_safe)
u_ilqr_np = np.array(u_ilqr)
h_vals_np = np.array(h_vals)
h_s_np = np.array(h_s)
goal_pos_np = np.array(goal_pos)

# ============================================================================
# Statistics
# ============================================================================

print(f"\n{'='*60}")
print(f"Simulation statistics ({n_steps} steps):")
print(f"  Total time: {elapsed:.2f} s")
print(f"  Avg time per step: {elapsed/n_steps*1000:.3f} ms")
print(f"{'='*60}")
print(f"Barrier statistics:")
print(f"  Min backup barrier h(x): {np.min(h_vals_np):.6f}")
print(f"  Min state barrier h_s(x): {np.min(h_s_np):.6f}")
print(f"{'='*60}")
print(f"Control statistics (Backup-safe, applied):")
print(f"  u1: min={u_safe_np[:, 0].min():.3f}, max={u_safe_np[:, 0].max():.3f}")
print(f"  u2: min={u_safe_np[:, 1].min():.3f}, max={u_safe_np[:, 1].max():.3f}")
print(f"{'='*60}")
print(f"Control statistics (iLQR desired):")
print(f"  u1: min={u_ilqr_np[:, 0].min():.3f}, max={u_ilqr_np[:, 0].max():.3f}")
print(f"  u2: min={u_ilqr_np[:, 1].min():.3f}, max={u_ilqr_np[:, 1].max():.3f}")
print(f"{'='*60}")
print(f"Intervention statistics:")
print(f"  Avg norm: {intervention_norm.mean():.4f}")
print(f"  Max norm: {intervention_norm.max():.4f}")
print(f"  Steps with intervention > 0.01: {np.sum(intervention_norm > 0.01)}/{n_steps}")
print(f"{'='*60}")
print(f"State statistics:")
print(f"  Final position: ({traj_np[-1, 0]:.3f}, {traj_np[-1, 1]:.3f})")
print(f"  Distance to goal: {np.linalg.norm(traj_np[-1, :2] - goal_pos_np):.3f}")
print(f"{'='*60}")

# ============================================================================
# Plotting
# ============================================================================

print("\nGenerating plots...")

current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Create map visualization
x_grid = np.linspace(-10.5, 10.5, 500)
y_grid = np.linspace(-10.5, 10.5, 500)
X, Y = np.meshgrid(x_grid, y_grid)
points = np.column_stack((X.flatten(), Y.flatten()))
points_jax = jnp.array(points)
points_with_zeros = jnp.concatenate([points_jax, jnp.zeros((points_jax.shape[0], 2))], axis=-1)
Z = map_.barrier.min_barrier(points_with_zeros)
Z = np.array(Z).reshape(X.shape)

# Plot 1: Trajectory
fig, ax = plt.subplots(figsize=(8, 8))
contour = ax.contour(X, Y, Z, levels=[0], colors='red', linewidths=2)
ax.plot(traj_np[0, 0], traj_np[0, 1], 'x', color='blue', markersize=10, label=r'$x_0$')
ax.plot(traj_np[-1, 0], traj_np[-1, 1], '+', color='blue', markersize=10, label=r'$x_f$')
ax.plot(traj_np[:, 0], traj_np[:, 1], 'black', linewidth=2, label='Trajectory')
ax.plot(goal_pos_np[0], goal_pos_np[1], '*', markersize=15, color='limegreen', label='Goal')

ax.set_xlabel(r'$x$', fontsize=14)
ax.set_ylabel(r'$y$', fontsize=14)
ax.set_aspect('equal', adjustable='box')
ax.legend(loc='upper right', fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'figs/02_iLQR_Backup_Trajectory_{current_time}.png', dpi=300)

# Plot 2: Controls comparison
fig, axs = plt.subplots(6, 1, figsize=(8, 10))

# States
axs[0].plot(time_array, traj_np[:, 0], color='black')
axs[0].set_ylabel(r'$x$', fontsize=14)

axs[1].plot(time_array, traj_np[:, 1], color='black')
axs[1].set_ylabel(r'$y$', fontsize=14)

axs[2].plot(time_array, traj_np[:, 2], color='black')
axs[2].set_ylabel(r'$v$', fontsize=14)

axs[3].plot(time_array, traj_np[:, 3], color='black')
axs[3].set_ylabel(r'$\theta$', fontsize=14)

# Control 1
axs[4].plot(time_array, u_ilqr_np[:, 0], 'r--', alpha=0.7, label=r'$u_{\rm iLQR}$')
axs[4].plot(time_array, u_safe_np[:, 0], 'k-', label=r'$u_{\rm backup}$')
axs[4].plot(time_array, info['ub_select'][:, 0], 'g:', alpha=0.7, label=r'$u_b$')
axs[4].legend(fontsize=10, loc='upper right', ncol=3, frameon=False)
axs[4].set_ylabel(r'$u_1$', fontsize=14)

# Control 2
axs[5].plot(time_array, u_ilqr_np[:, 1], 'r--', alpha=0.7, label=r'$u_{\rm iLQR}$')
axs[5].plot(time_array, u_safe_np[:, 1], 'k-', label=r'$u_{\rm backup}$')
axs[5].plot(time_array, info['ub_select'][:, 1], 'g:', alpha=0.7, label=r'$u_b$')
axs[5].legend(fontsize=10, loc='upper right', ncol=3, frameon=False)
axs[5].set_ylabel(r'$u_2$', fontsize=14)
axs[5].set_xlabel(r'$t~(\rm {s})$', fontsize=14)

for i in range(5):
    axs[i].tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)

for ax in axs:
    ax.set_xlim(0, sim_time)

plt.tight_layout()
plt.savefig(f'figs/02_iLQR_Backup_States_{current_time}.png', dpi=300)

# Plot 3: Barriers and blending
fig, axs = plt.subplots(3, 1, figsize=(8, 6))

axs[0].plot(time_array, h_vals_np[:, 0], label=r'$h$', color='blue')
axs[0].plot(time_array, h_s_np[:, 0], label=r'$h_s$', color='red', linestyle='--')
axs[0].axhline(y=0, color='green', linestyle=':')
axs[0].set_ylabel(r'$h$', fontsize=14)
axs[0].legend(fontsize=12, frameon=False)
axs[0].set_xlim(0, sim_time)

axs[1].plot(time_array, info['beta'][:], color='blue')
axs[1].set_ylabel(r'$\beta$ (blend)', fontsize=14)
axs[1].set_xlim(0, sim_time)

axs[2].plot(time_array, intervention_norm, color='purple')
axs[2].set_ylabel(r'$\|u - u_{\rm iLQR}\|$', fontsize=14)
axs[2].set_xlabel(r'$t~(\rm {s})$', fontsize=14)
axs[2].set_xlim(0, sim_time)

for i in range(2):
    axs[i].tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)

plt.tight_layout()
plt.savefig(f'figs/02_iLQR_Backup_Barriers_{current_time}.png', dpi=300)

plt.show()

# --- Animation ---
print("\nCreating animation...")
import matplotlib.animation as animation

# Store predicted trajectories for animation
pred_trajs = []
sample_indices_pred = np.arange(0, n_steps, 100)
for i in sample_indices_pred:
    x_traj_pred, _ = ilqr_controller.get_predicted_trajectory(traj[i])
    pred_trajs.append(np.array(x_traj_pred))

def create_animation():
    fig_anim, ax_anim = plt.subplots(figsize=(8, 8))

    # Sample frames for animation (every 10 steps)
    frame_indices = np.arange(0, n_steps, 10)

    def animate(frame_idx):
        frame = frame_indices[frame_idx]
        ax_anim.clear()

        # Current state
        current_x = traj_np[frame, 0]
        current_y = traj_np[frame, 1]
        current_v = traj_np[frame, 2]

        # Past trajectory
        past_x = traj_np[:frame + 1, 0]
        past_y = traj_np[:frame + 1, 1]

        # Draw map contour
        ax_anim.contour(X, Y, Z, levels=[0], colors='red', linewidths=2)

        # Draw goal
        ax_anim.plot(goal_pos_np[0], goal_pos_np[1], '*', markersize=15,
                     color='limegreen', label='Goal', zorder=5)

        # Draw past trajectory
        ax_anim.plot(past_x, past_y, 'b-', linewidth=2, label='Trajectory', zorder=3)

        # Draw current position
        ax_anim.scatter([current_x], [current_y], s=100, c='blue', marker='o',
                        edgecolors='black', linewidths=2, label='Current', zorder=4)

        # Draw iLQR predicted trajectory (if available)
        pred_idx = frame // 100
        if pred_idx < len(pred_trajs):
            pred_traj = pred_trajs[pred_idx]
            ax_anim.plot(pred_traj[:, 0], pred_traj[:, 1], 'c--', linewidth=2,
                         alpha=0.6, label='iLQR Prediction', zorder=2)

        # Set plot properties
        ax_anim.set_xlabel(r'$x$', fontsize=14)
        ax_anim.set_ylabel(r'$y$', fontsize=14)
        ax_anim.set_xlim(-10.5, 10.5)
        ax_anim.set_ylim(-10.5, 10.5)
        ax_anim.set_aspect('equal', adjustable='box')
        ax_anim.legend(loc='upper left', fontsize=10)
        ax_anim.grid(True, alpha=0.3)

        # Info text
        current_time_val = frame * timestep
        intervention_val = intervention_norm[frame] if frame < len(intervention_norm) else 0
        beta_val = float(info['beta'][frame]) if frame < len(info['beta']) else 0
        ax_anim.text(0.98, 0.98,
                     f'Time: {current_time_val:.2f}s\n'
                     f'Vel: {current_v:.2f} m/s\n'
                     f'Beta: {beta_val:.3f}\n'
                     f'Intervention: {intervention_val:.3f}',
                     transform=ax_anim.transAxes, fontsize=11, verticalalignment='top',
                     horizontalalignment='right',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        return []

    # Create animation
    anim = animation.FuncAnimation(fig_anim, animate, frames=len(frame_indices),
                                   interval=50, blit=True)

    # Save animation
    animation_file = f'figs/02_iLQR_Backup_Animation_{current_time}.mp4'
    writer = animation.FFMpegWriter(fps=20, metadata=dict(artist='CBFJAX'), bitrate=1800)
    anim.save(animation_file, writer=writer)
    print(f"Animation saved as: {animation_file}")
    plt.show()

create_animation()

print(f"\nPlots saved with timestamp: {current_time}")
print("Simulation complete!")