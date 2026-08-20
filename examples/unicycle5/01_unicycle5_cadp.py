"""
Receding-horizon C-ADP safe control for the 5th-order nonholonomic ground robot.

Demonstrates:
- One-call construction of the whole pipeline via cbfjax.from_config
- Relative-degree lifting h_i = Lf phi_i + zeta phi_i, then a log-sum-exponential
  composition of the 43 barriers into the single CBF psi_0
- CADPSafeControl: N-step backward Riccati-like recursion with closed-form
  constrained optimizers, replanned on the update period T_s
"""

import jax
import jax.numpy as jnp
import matplotlib as mpl
import numpy as np
from time import time
import datetime
import os

# CBFJAX imports
import cbfjax
cbfjax.configure_jax(platform="cpu", enable_x64=True)
from cbfjax.dynamics import Unicycle5thOrderDynamics
from map_config import map_config

script_dir = os.path.dirname(os.path.abspath(__file__))
import matplotlib.pyplot as plt

mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'

# ============================================
# Configuration
# ============================================

ZETA = 0.5          # relative-degree lifting gain for the obstacles
RHO = 750.0         # log-sum-exponential composition sharpness
S_BAR = 1.5         # speed limit, m/s
OMEGA_BAR = 0.5     # angular velocity limit, rad/s
ALPHA_GAIN = 1.0    # class-K gain of the CBF constraint (eq. 54)

# Barrier configuration: obstacles enter at relative degree 2 and are lifted
# to relative degree 1 by the alpha gain zeta.
cfg = {
    'softmin_rho': RHO,
    'pos_barrier_rel_deg': 2,
    'obstacle_alpha': (ZETA,),
    'boundary_alpha': (ZETA,),
}

# C-ADP parameters (Method 1)
cadp_params = {
    'horizon_steps': 400,    # N = T / Tp
    'planning_dt': 0.05,     # Tp
    'softplus_gain': 1.0,    # eta
    'num_iter': 1,           # forward/backward passes per update
    'refresh_every': 5,      # Ts = 0.05 s at a 100 Hz zero-order hold
}

# Cost weights
Q_state = jnp.diag(jnp.array([1.0, 1.0, 0.0, 16.0, 160.0]))
R_v = jnp.diag(jnp.array([80.0, 80.0]))
Omega_v = jnp.zeros(2)
r_delta = 0.2e10

# Goal and initial condition: x = [q_x, q_y, gamma, s, omega]
goal_pos = jnp.array([[8.0, 8.0]])
x_d = jnp.concatenate([goal_pos[0], jnp.zeros(3)])
x0 = jnp.array([-8.0, -8.0, 0.0, 0.0, 0.0])

# Simulation parameters
sim_time = 60.0
dt_sim = 0.01           # 100 Hz zero-order hold
d_tol = 0.25            # arrival tolerance, m

# ============================================
# Build the full pipeline
# ============================================

print("Building C-ADP safety filter via cbfjax.from_config...")

dynamics = Unicycle5thOrderDynamics()


def speed_barrier(x):
    """h_42(x) = sbar^2 - s^2, relative degree 1."""
    return S_BAR ** 2 - x[3] ** 2


def yaw_rate_barrier(x):
    """h_43(x) = omegabar^2 - omega^2, relative degree 1."""
    return OMEGA_BAR ** 2 - x[4] ** 2


parts = cbfjax.from_config({
    'dynamics': dynamics,
    'barriers': {
        'map':      {'type': 'map', **map_config, 'cfg': cfg},
        'speed':    {'type': 'func', 'h': speed_barrier, 'rel_deg': 1},
        'yaw_rate': {'type': 'func', 'h': yaw_rate_barrier, 'rel_deg': 1},
        'psi0':     {'type': 'soft_composition',
                     'barriers': ['map', 'speed', 'yaw_rate'],
                     'cfg': cfg},
    },
    'filter': {
        'type': 'cadp',
        'barrier': 'psi0',
        'action_dim': 2,
        'alpha': lambda z: ALPHA_GAIN * z,
        'params': cadp_params,
        'Q_state': Q_state,
        # x_ref sets Gamma = -Q x_d, which makes the stage cost
        # 1/2 (x - x_d)' Q (x - x_d) up to a constant.
        'x_ref': x_d,
        'R_v': R_v,
        'Omega_v': Omega_v,
        'r_delta': r_delta,
    },
})

safety_filter = parts.filter
barrier = parts.barriers['psi0']
map_ = parts.barriers['map']

nx = dynamics.state_dim   # 5: [q_x, q_y, gamma, s, omega]
nu = dynamics.action_dim  # 2: [u_r, u_l]

n_barriers = len(map_.pos_barriers) + 2
print(f"  State dim: {nx}, Action dim: {nu}")
print(f"  psi_0 composes {n_barriers} barriers with rho = {RHO:.0f}")
print(f"  Horizon: N = {cadp_params['horizon_steps']}, "
      f"Tp = {cadp_params['planning_dt']} s, "
      f"T = {cadp_params['horizon_steps'] * cadp_params['planning_dt']:.0f} s")

# ============================================
# Test Controller
# ============================================

print("\nTesting controller...")

start_time = time()
u_test, _, info_test = safety_filter.optimal_control_with_info(
    x0, safety_filter.get_init_state())
jax.block_until_ready(u_test)
print(f"  Compile + first solve: {time() - start_time:.2f} s")
print(f"  Test control: u = {np.array(u_test)}")
print(f"  psi_0(x0) = {float(barrier.hocbf(x0)):.4f}, "
      f"constraint at u = {float(info_test.constraint_at_u):.4e}")

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
jax.block_until_ready(trajs)
simulation_time = time() - start_time

print(f"Simulation completed in {simulation_time:.2f} seconds")

# ============================================
# Compute Control Actions and Barrier Values
# ============================================

print("\nComputing control actions and barrier values...")

x_hist = trajs[:, 0, :]  # (time_steps, state_dim)
n_steps = x_hist.shape[0] - 1
time_array = np.linspace(0, sim_time, n_steps + 1)


def replay(ctrl_state, x):
    """Re-thread the controller state along the recorded trajectory."""
    u, next_state, info = safety_filter.optimal_control_with_info(x, ctrl_state)
    return next_state, (u, info.lam, info.slack_vars, info.constraint_at_u)


_, (u_hist, lam_hist, slack_hist, constr_hist) = jax.lax.scan(
    replay, safety_filter.get_init_state(), x_hist)

psi_vals = jax.vmap(barrier.hocbf)(x_hist)
min_obstacle_vals = jax.vmap(map_.barrier.min_barrier)(x_hist)

x_hist_np = np.array(x_hist)
u_hist_np = np.array(u_hist)
psi_vals_np = np.array(psi_vals)
min_obstacle_np = np.array(min_obstacle_vals)
lam_np = np.array(lam_hist)
constr_np = np.array(constr_hist)
goal_pos_np = np.array(goal_pos[0])

dist_to_goal = np.linalg.norm(x_hist_np[:, :2] - goal_pos_np, axis=1)
reached = np.where(dist_to_goal <= d_tol)[0]
arrival_time = time_array[reached[0]] if reached.size else np.inf

# ============================================
# Statistics
# ============================================

print(f"\n{'='*60}")
print(f"Simulation statistics ({n_steps} steps):")
print(f"  Total time: {simulation_time:.2f} s")
print(f"  Avg time per step: {simulation_time/n_steps*1000:.3f} ms")
print(f"{'='*60}")
print(f"Barrier statistics:")
print(f"  Min psi_0(x): {np.min(psi_vals_np):.6f}")
print(f"  Min obstacle barrier: {np.min(min_obstacle_np):.6f}")
print(f"  Min constraint a + b'u: {np.min(constr_np):.4e}")
print(f"  Max multiplier lambda: {np.max(lam_np):.4e}")
print(f"{'='*60}")
print(f"Control statistics:")
print(f"  u_r: min={u_hist_np[:, 0].min():.3f}, max={u_hist_np[:, 0].max():.3f}")
print(f"  u_l: min={u_hist_np[:, 1].min():.3f}, max={u_hist_np[:, 1].max():.3f}")
print(f"{'='*60}")
print(f"State statistics:")
print(f"  Speed s: min={x_hist_np[:, 3].min():.3f}, "
      f"max={x_hist_np[:, 3].max():.3f} (limit {S_BAR})")
print(f"  Yaw rate omega: min={x_hist_np[:, 4].min():.3f}, "
      f"max={x_hist_np[:, 4].max():.3f} (limit {OMEGA_BAR})")
print(f"  Final position: ({x_hist_np[-1, 0]:.3f}, {x_hist_np[-1, 1]:.3f})")
print(f"  Distance to goal: {dist_to_goal[-1]:.3f}")
print(f"  Arrival time (tol {d_tol} m): "
      f"{arrival_time if np.isfinite(arrival_time) else 'not reached'}")
print(f"{'='*60}")

# ============================================
# Plots
# ============================================

print("\nGenerating plots...")

current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
os.makedirs(os.path.join(script_dir, 'figs'), exist_ok=True)

x_grid = np.linspace(-10.5, 10.5, 400)
y_grid = np.linspace(-10.5, 10.5, 400)
X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
points = np.column_stack((X_grid.flatten(), Y_grid.flatten()))
points_full = np.column_stack((points, np.zeros((points.shape[0], 3))))
points_jax = jnp.array(points_full)

Z = jax.vmap(map_.barrier.min_barrier)(points_jax)
Z = np.array(Z).reshape(X_grid.shape)

# --- Trajectory Plot ---
fig, ax = plt.subplots(figsize=(6, 6))

ax.contour(X_grid, Y_grid, Z, levels=[0], colors='red')

ax.set_xlabel(r'$q_{\rm x}$', fontsize=16)
ax.set_ylabel(r'$q_{\rm y}$', fontsize=16)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_aspect('equal', adjustable='box')
ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=16)
ax.set_xticks([-10, -5, 0, 5, 10])
ax.set_yticks([-10, -5, 0, 5, 10])

ax.plot(x_hist_np[0, 0], x_hist_np[0, 1], 'x', color='blue', markersize=8,
        label=r'$x_0$')
ax.plot(goal_pos_np[0], goal_pos_np[1], '*', markersize=10, color='limegreen',
        label='Goal')
ax.plot(x_hist_np[-1, 0], x_hist_np[-1, 1], '+', color='blue', markersize=8,
        label=r'$x_f$')
ax.plot(x_hist_np[:, 0], x_hist_np[:, 1], color='black', label='Trajectory')

from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color='red', lw=1.5)]
handles, labels = ax.get_legend_handles_labels()
handles.insert(0, custom_lines[0])
labels.insert(0, r'$\mathcal{S}_{\rm s}$')
ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.12),
          ncol=3, frameon=False, fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(script_dir,
                         f'figs/01_CADP_Trajectory_{current_time}.png'), dpi=200)
plt.show()

# --- States and Control Plot ---
fig, axs = plt.subplots(5, 1, figsize=(8, 8))

axs[0].plot(time_array, x_hist_np[:, 0], label=r'$q_{\rm x}$', color='red')
axs[0].plot(time_array, x_hist_np[:, 1], label=r'$q_{\rm y}$', color='blue')
axs[0].axhline(y=goal_pos_np[0], color='red', linestyle=':', alpha=0.7)
axs[0].axhline(y=goal_pos_np[1], color='blue', linestyle=':', alpha=0.7)
axs[0].set_ylabel(r'$q_{\rm x}, q_{\rm y}$', fontsize=16)
axs[0].legend(loc='lower center', ncol=2, frameon=False, fontsize=14)

axs[1].plot(time_array, x_hist_np[:, 3], color='black')
axs[1].axhline(y=S_BAR, color='red', linestyle='--', linewidth=1.2)
axs[1].axhline(y=-S_BAR, color='red', linestyle='--', linewidth=1.2)
axs[1].set_ylabel(r'$s$', fontsize=16)

axs[2].plot(time_array, x_hist_np[:, 4], color='black')
axs[2].axhline(y=OMEGA_BAR, color='red', linestyle='--', linewidth=1.2)
axs[2].axhline(y=-OMEGA_BAR, color='red', linestyle='--', linewidth=1.2)
axs[2].set_ylabel(r'$\omega$', fontsize=16)

axs[3].plot(time_array, u_hist_np[:, 0], color='black')
axs[3].set_ylabel(r'$u_{\rm r}$', fontsize=16)

axs[4].plot(time_array, u_hist_np[:, 1], color='black')
axs[4].set_ylabel(r'$u_{\rm l}$', fontsize=16)
axs[4].set_xlabel(r'$t~(\rm {s})$', fontsize=16)

for i in range(4):
    axs[i].tick_params(axis='x', which='both', bottom=True, top=False,
                       labelbottom=False)

for ax in axs:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=16)
    ax.set_xlim(time_array[0], time_array[-1])

plt.subplots_adjust(wspace=0, hspace=0.2)
plt.tight_layout()
plt.savefig(os.path.join(script_dir,
                         f'figs/01_CADP_States_{current_time}.png'), dpi=200)
plt.show()

# --- Barrier Values Plot ---
fig, axs = plt.subplots(3, 1, figsize=(8, 4.5))

axs[0].plot(time_array, psi_vals_np, color='black')
axs[0].axhline(y=0, color='red', linestyle='--', linewidth=1.5)
axs[0].set_ylabel(r'$\psi_0(x)$', fontsize=16)

axs[1].plot(time_array, min_obstacle_np, color='black')
axs[1].axhline(y=0, color='red', linestyle='--', linewidth=1.5)
axs[1].set_ylabel(r'$\min_i \phi_i(x)$', fontsize=16)

axs[2].plot(time_array, lam_np, color='black')
axs[2].set_ylabel(r'$\lambda_0$', fontsize=16)
axs[2].set_xlabel(r'$t~(\rm {s})$', fontsize=16)

for i in range(2):
    axs[i].tick_params(axis='x', which='both', bottom=True, top=False,
                       labelbottom=False)

for ax in axs:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=16)
    ax.set_xlim(time_array[0], time_array[-1])

plt.tight_layout()
plt.savefig(os.path.join(script_dir,
                         f'figs/01_CADP_Barriers_{current_time}.png'), dpi=200)
plt.show()

print("Done.")
