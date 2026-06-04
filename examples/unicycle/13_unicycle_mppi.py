"""
MPPI control for unicycle goal-reaching with soft obstacle avoidance.

Demonstrates:
- MPPIControl setup and assignment chain
- Running + terminal cost with soft barrier penalty
- Stateful ZOH simulation loop (state carries warm-start U and PRNGKey)
- Warm-up JIT compilation and per-step timing

State:    [q_x, q_y, v, theta]
Controls: [a (acceleration), omega (angular velocity)]

Note: MPPI provides no hard safety guarantees. For guaranteed safety,
use MPPIControl as the desired control inside MinIntervQPSafeControl.
"""

import os
import datetime
from math import pi
from time import perf_counter

# Configure JAX platform BEFORE any other cbfjax or jax imports so the
# platform override takes effect before the backend is initialised.
from cbfjax.config import configure_jax, get_jax_config
configure_jax(platform='cuda', enable_x64=False)

import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import numpy as np

from cbfjax.dynamics.unicycle import UnicycleDynamics
from cbfjax.controls.mppi_control import MPPIControl
from cbfjax.utils.make_map import Map
from cbfjax.barriers.composite_barrier import SoftCompositionBarrier
from immutabledict import immutabledict

from map_config import map_config

script_dir = os.path.dirname(os.path.abspath(__file__))

mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'

# ============================================================
# Configuration
# ============================================================

# MPPI parameters
mppi_params = {
    'num_samples': 1000,
    'horizon':     4.0,     # seconds  → N = 20 steps
    'time_steps':  0.1,     # dt (must match dynamics discretization_dt)
    'temperature': 0.5,     # lambda: lower = more peaked weights
    'init_seed':   0,
}

# Noise standard deviations  [acceleration, angular velocity]
noise_sigma = [1.5, 2.0]

# Control bounds  [a_min, omega_min], [a_max, omega_max]
ctrl_low  = [-2.0, -1.0]
ctrl_high = [  2.0,  1.0]

# Cost weights
w_pos      = 2.0     # running position error
w_vel      = 0.05    # penalise high speed
w_ctrl     = 0.01    # control effort
w_barrier  = 200.0   # soft obstacle / boundary penalty
w_terminal = 40.0    # terminal position error

# Barrier configuration (rel_deg=2 for position, 1 for velocity)
cfg = immutabledict({
    'softmax_rho': 20,
    'softmin_rho': 20,
    'pos_barrier_rel_deg': 2,
    'vel_barrier_rel_deg': 1,
    'obstacle_alpha': (10.0,),
    'boundary_alpha': (10.0,),
    'velocity_alpha': (),
})

# Animation
N_vis = 100    # sampled trajectories to display per animation frame

# Scenario
goal_pos = jnp.array([3.0, 4.5])
x0       = jnp.array([-1.0, -8.5, 0.0, pi / 2])

# Simulation
sim_time     = 20.0     # seconds
dt_ctrl      = mppi_params['time_steps']          # control update period
dt_sim       = 0.01                               # integration sub-step
substeps     = int(round(dt_ctrl / dt_sim))       # 10
n_ctrl_steps = int(round(sim_time / dt_ctrl))     # 200

# ============================================================
# Dynamics
# ============================================================

print("Setting up dynamics...")
print(f"  Device : {jax.devices()[0]}")
print(f"  Config : {get_jax_config()}")

dynamics = UnicycleDynamics(params={
    'discretization_dt':     dt_ctrl,
    'discretization_method': 'euler',
})
nx = dynamics.state_dim    # 4: [q_x, q_y, v, theta]
nu = dynamics.action_dim   # 2: [a, omega]
print(f"  state_dim={nx}, action_dim={nu}")

# ============================================================
# Barriers  (used only for soft cost penalty inside MPPI)
# ============================================================

print("Setting up barriers...")

map_ = Map(barriers_info=map_config, dynamics=dynamics, cfg=cfg).create_barriers()
pos_barriers, vel_barriers = map_.get_barriers()

barrier = (
    SoftCompositionBarrier(cfg={'softmin_rho': cfg['softmin_rho'],
                                'softmax_rho': cfg['softmax_rho']})
    .assign_dynamics(dynamics)
    .assign_barriers_and_rule(barriers=[*pos_barriers, *vel_barriers], rule='i')
)
print(f"  SoftCompositionBarrier: {len(pos_barriers)} position + {len(vel_barriers)} velocity barriers")

# ============================================================
# MPPI Cost Functions
# ============================================================

_goal = goal_pos   # captured in closures below

def running_cost(x, u, t):
    pos_err = x[:2] - _goal
    h_val   = barrier._hocbf_single(x)
    # softmin with rho=20 overflows fp32 when any h_i < -4.4 → inf → NaN weights
    h_val   = jnp.where(jnp.isfinite(h_val), h_val, -100.0)
    obs_pen = jnp.maximum(0.0, -h_val) ** 2
    return (w_pos    * jnp.sum(pos_err ** 2)
            + w_vel  * x[2] ** 2
            + w_ctrl * jnp.sum(u ** 2)
            + w_barrier * obs_pen)

def terminal_cost(x):
    pos_err = x[:2] - _goal
    h_val   = barrier._hocbf_single(x)
    h_val   = jnp.where(jnp.isfinite(h_val), h_val, -100.0)
    obs_pen = jnp.maximum(0.0, -h_val) ** 2
    return w_terminal * jnp.sum(pos_err ** 2) + w_barrier * obs_pen

# ============================================================
# MPPI Controller
# ============================================================

print("Setting up MPPI controller...")

ctrl = (
    MPPIControl.create_empty(action_dim=nu, params=mppi_params)
    .assign_dynamics(dynamics)
    .assign_cost_func(running_cost)
    .assign_terminal_cost_func(terminal_cost)
    .assign_noise_sigma(noise_sigma)
    .assign_control_bounds(ctrl_low, ctrl_high)
)

print(f"  K={ctrl.num_samples} samples, N={ctrl.N_horizon} steps, dt={dt_ctrl}s")

# Fixed random subset of K trajectories to draw each frame (consistent across time)
_vis_rng = np.random.default_rng(seed=0)
vis_idx  = _vis_rng.choice(ctrl.num_samples, N_vis, replace=False)

# ============================================================
# Warm-Up  (trigger JIT compilation before timing)
# ============================================================

print("\nWarm-up (JIT compilation)...")
_state_warmup = ctrl.get_init_state()
_t0 = perf_counter()
_, _ = ctrl._optimal_control_single(x0, _state_warmup)
jax.block_until_ready(_)
ctrl.get_predicted_trajectories(x0, _state_warmup)
print(f"  Compilation time: {perf_counter() - _t0:.2f} s")

# ============================================================
# Closed-Loop Simulation  (ZOH via framework)
# ============================================================

print(f"\nRunning simulation  ({n_ctrl_steps} steps × {substeps} sub-steps)...")

_t0_sim = perf_counter()
trajs = ctrl.get_optimal_trajs_zoh_no_vmap(
    x0=x0[None],
    timestep=dt_ctrl,
    sim_time=sim_time,
    intermediate_steps=substeps,
    method='euler',
)
jax.block_until_ready(trajs)
sim_elapsed = perf_counter() - _t0_sim

x_hist     = np.array(trajs[:, 0, :])                        # (n_steps+1, nx)
n_steps    = x_hist.shape[0] - 1
time_array = np.linspace(0.0, sim_time, n_steps + 1)

# ============================================================
# Post-sim pass: recover u_hist + collect animation predictions
#
# Same initial key + same states -> identical controls to what was applied.
# Predictions are collected only at animation frame indices to avoid overhead.
# ============================================================

print("Recovering controls and collecting MPPI predictions...")

_frame_stride = max(1, n_steps // 200)
_anim_steps   = set(range(0, n_steps, _frame_stride))

pred_xy_hist = {}   # step -> (N_vis, N+1, 2)
pred_w_hist  = {}   # step -> (N_vis,)
u_hist_list  = []

_post_state = ctrl.get_init_state()
for step in range(n_steps):
    x_step = jnp.array(x_hist[step])
    if step in _anim_steps:
        x_tr, _, w = ctrl.get_predicted_trajectories(x_step, _post_state)
        pred_xy_hist[step] = np.array(x_tr)[vis_idx, :, :2]
        pred_w_hist[step]  = np.array(w)[vis_idx]
    u, _post_state = ctrl._optimal_control_single(x_step, _post_state)
    u_hist_list.append(np.array(u))

u_hist = np.array(u_hist_list)   # (n_steps, nu)

# ============================================================
# Statistics
# ============================================================

avg_step_ms     = sim_elapsed / n_steps * 1000.0
goal_dist_final = float(jnp.linalg.norm(x_hist[-1, :2] - np.array(goal_pos)))

h_vals_hist = np.array(barrier.hocbf(jnp.array(x_hist)))   # (T, 1) — composed value
min_h_hist  = h_vals_hist.squeeze(-1)                        # (T,)

print(f"\n{'='*60}")
print(f"Simulation statistics ({n_steps} steps):")
print(f"  Total sim time  : {sim_elapsed:.2f} s")
print(f"  Avg step time   : {avg_step_ms:.2f} ms")
print(f"{'='*60}")
print(f"Trajectory:")
print(f"  Final position  : ({x_hist[-1,0]:.3f}, {x_hist[-1,1]:.3f})")
print(f"  Distance to goal: {goal_dist_final:.3f} m")
print(f"  Min barrier h(x): {min_h_hist.min():.4f}  (< 0 = constraint violated)")
print(f"{'='*60}")

# ============================================================
# Barrier contour for plots
# ============================================================

x_grid  = np.linspace(-10.5, 10.5, 400)
y_grid  = np.linspace(-10.5, 10.5, 400)
Xg, Yg  = np.meshgrid(x_grid, y_grid)
pts     = np.column_stack([Xg.ravel(), Yg.ravel(), np.zeros((Xg.size, 2))])
Z       = np.array(barrier.hocbf(jnp.array(pts, dtype=jnp.float32)))
Z       = Z.reshape(Xg.shape)

goal_np = np.array(goal_pos)
current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
os.makedirs(os.path.join(script_dir, 'figs'), exist_ok=True)

# ============================================================
# Plot 1: Trajectory
# ============================================================

fig, ax = plt.subplots(figsize=(6, 6))
ax.contour(Xg, Yg, Z, levels=[0], colors='red', linewidths=1.5)
ax.plot(x_hist[0, 0],  x_hist[0, 1],  'x',  color='blue',      markersize=8,  label=r'$x_0$')
ax.plot(goal_np[0],    goal_np[1],     '*',  color='limegreen', markersize=12, label='Goal')
ax.plot(x_hist[-1, 0], x_hist[-1, 1], '+',  color='blue',      markersize=8,  label=r'$x_f$')
ax.plot(x_hist[:, 0],  x_hist[:, 1],  '-',  color='black',     linewidth=1.2, label='Trajectory')

ax.set_xlabel(r'$q_{\rm x}$', fontsize=16)
ax.set_ylabel(r'$q_{\rm y}$', fontsize=16)
ax.set_aspect('equal', adjustable='box')
ax.set_xticks([-10, -5, 0, 5, 10])
ax.set_yticks([-10, -5, 0, 5, 10])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(labelsize=14)

from matplotlib.lines import Line2D
handles, labels = ax.get_legend_handles_labels()
handles.insert(0, Line2D([0], [0], color='red', lw=1.5))
labels.insert(0, r'$\partial\mathcal{S}$')
ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.13),
          ncol=3, frameon=False, fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(script_dir, f'figs/13_MPPI_Trajectory_{current_time}.png'), dpi=200)
plt.show()

# ============================================================
# Plot 2: States and Controls
# ============================================================

fig, axs = plt.subplots(6, 1, figsize=(9, 10))

axs[0].plot(time_array, x_hist[:, 0], color='red',  label=r'$q_{\rm x}$')
axs[0].plot(time_array, x_hist[:, 1], color='blue', label=r'$q_{\rm y}$')
axs[0].axhline(goal_np[0], color='red',  linestyle=':', alpha=0.6)
axs[0].axhline(goal_np[1], color='blue', linestyle=':', alpha=0.6)
axs[0].set_ylabel(r'$q_{\rm x}, q_{\rm y}$', fontsize=14)
axs[0].legend(ncol=2, frameon=False, fontsize=12, loc='upper right')

axs[1].plot(time_array, x_hist[:, 2], color='black')
axs[1].set_ylabel(r'$v$', fontsize=14)

axs[2].plot(time_array, x_hist[:, 3], color='black')
axs[2].set_ylabel(r'$\theta$', fontsize=14)

axs[3].plot(time_array[:-1], u_hist[:, 0], color='black')
axs[3].set_ylabel(r'$a$', fontsize=14)

axs[4].plot(time_array[:-1], u_hist[:, 1], color='black')
axs[4].set_ylabel(r'$\omega$', fontsize=14)

axs[5].plot(time_array, min_h_hist, color='black')
axs[5].axhline(0, color='red', linestyle='--', linewidth=1.5, label=r'$h=0$')
axs[5].set_ylabel(r'$\min_i h_i(x)$', fontsize=14)
axs[5].set_xlabel(r'$t~(\rm{s})$', fontsize=14)
axs[5].legend(frameon=False, fontsize=12)

for i in range(5):
    axs[i].tick_params(axis='x', labelbottom=False)
for ax in axs:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=12)
    ax.set_xlim(time_array[0], time_array[-1])

plt.subplots_adjust(hspace=0.25)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, f'figs/13_MPPI_States_{current_time}.png'), dpi=200)
plt.show()

# ============================================================
# Animation
# ============================================================

print("\nCreating animation...")

fig_anim, ax_anim = plt.subplots(figsize=(6, 6))

frame_indices = np.array(sorted(_anim_steps))

def animate(fi):
    frame = frame_indices[fi]
    ax_anim.clear()

    # --- sampled trajectories (LineCollection for speed) ---
    trajs  = pred_xy_hist[frame]   # (N_vis, N+1, 2)
    w      = pred_w_hist[frame]    # (N_vis,)
    w_norm = (w - w.min()) / (w.max() - w.min() + 1e-10)

    colors       = cm.cool(w_norm)       # cyan (low weight) → magenta (high weight)
    colors[:, 3] = 0.45                  # fixed alpha — all trajectories equally visible

    lc = LineCollection(list(trajs), colors=colors, linewidths=0.7, zorder=2)
    ax_anim.add_collection(lc)

    # --- weighted-mean trajectory = nominal plan ---
    w_sum    = w.sum()
    w_renorm = w / w_sum if w_sum > 1e-10 else np.ones_like(w) / len(w)
    nominal_xy = np.einsum('k,knt->nt', w_renorm, trajs)    # (N+1, 2)
    ax_anim.plot(nominal_xy[:, 0], nominal_xy[:, 1],
                 '--', color='darkorange', linewidth=2.0, zorder=3, label='Plan')

    # --- obstacle boundary ---
    ax_anim.contour(Xg, Yg, Z, levels=[0], colors='red', linewidths=1.5, zorder=4)

    # --- past executed trajectory ---
    ax_anim.plot(x_hist[:frame + 1, 0], x_hist[:frame + 1, 1],
                 '-', color='black', linewidth=1.5, zorder=5, label='Trajectory')

    # --- current pose: circle + heading arrow ---
    cx, cy, cv, ctheta = x_hist[frame]
    ax_anim.scatter([cx], [cy], s=80, c='blue',
                    edgecolors='black', linewidths=1.2, zorder=6)
    arr_len = 0.8
    ax_anim.annotate(
        '', xy=(cx + arr_len * np.cos(ctheta), cy + arr_len * np.sin(ctheta)),
        xytext=(cx, cy),
        arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
        zorder=6,
    )

    # --- goal ---
    ax_anim.plot(goal_np[0], goal_np[1], '*',
                 color='limegreen', markersize=14, zorder=7, label='Goal')

    # --- info text ---
    t_now = frame * dt_ctrl
    dist  = np.linalg.norm(x_hist[frame, :2] - goal_np)
    info  = (f'$t = {t_now:.1f}$ s\n'
             f'$v = {cv:.2f}$ m/s\n'
             f'$d_{{\\rm goal}} = {dist:.2f}$ m')
    ax_anim.text(0.02, 0.97, info, transform=ax_anim.transAxes,
                 fontsize=11, va='top',
                 bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))

    # --- formatting (match Plot 1) ---
    ax_anim.set_xlim(-10.5, 10.5)
    ax_anim.set_ylim(-10.5, 10.5)
    ax_anim.set_aspect('equal', adjustable='box')
    ax_anim.set_xticks([-10, -5, 0, 5, 10])
    ax_anim.set_yticks([-10, -5, 0, 5, 10])
    ax_anim.set_xlabel(r'$q_{\rm x}$', fontsize=16)
    ax_anim.set_ylabel(r'$q_{\rm y}$', fontsize=16)
    ax_anim.tick_params(labelsize=14)
    ax_anim.spines['top'].set_visible(False)
    ax_anim.spines['right'].set_visible(False)

    handles, labels = ax_anim.get_legend_handles_labels()
    handles.insert(0, Line2D([0], [0], color='red', lw=1.5))
    labels.insert(0, r'$\partial\mathcal{S}$')
    ax_anim.legend(handles, labels, loc='upper center',
                   bbox_to_anchor=(0.5, 1.12), ncol=4,
                   frameon=False, fontsize=11)
    return []

anim = animation.FuncAnimation(fig_anim, animate,
                                frames=len(frame_indices), interval=50, blit=True)
anim_path = os.path.join(script_dir, f'figs/13_MPPI_Animation_{current_time}.mp4')
try:
    writer = animation.FFMpegWriter(fps=20, metadata=dict(artist='CBFJAX'), bitrate=2400)
    anim.save(anim_path, writer=writer)
    print(f"  Animation saved: {anim_path}")
except Exception as e:
    print(f"  Animation save failed ({e}). Displaying interactively.")
plt.show()

print(f"\nDone. Figures saved with timestamp {current_time}.")
