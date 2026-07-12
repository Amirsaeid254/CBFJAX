"""
Multi-robot ensemble: 8 unicycles with a single compiled rollout.

Demonstrates:
- stack_ensemble / unstack_ensemble from cbfjax.utils
- Shared softmin map barrier via from_config with composition='soft'
- UnicycleGoalControl as a traced leaf — goal swap = no retrace
- eqx.filter_jit + eqx.filter_vmap over lax.scan for N robots
"""

import os
import sys
import datetime
from math import pi
from time import time

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import matplotlib as mpl
import matplotlib.pyplot as plt
from immutabledict import immutabledict

import cbfjax
cbfjax.configure_jax(platform="cpu", enable_x64=True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from map_config import map_config

from cbfjax.dynamics.unicycle import UnicycleDynamics
from unicycle_desired_control import UnicycleGoalControl

script_dir = os.path.dirname(os.path.abspath(__file__))
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'

# ============================================================
# Configuration
# ============================================================

N = 8
dt = 0.01
T = 20.0
n_steps = int(T / dt)

barrier_cfg = immutabledict({
    'softmax_rho': 20,
    'softmin_rho': 20,
    'pos_barrier_rel_deg': 2,
    'vel_barrier_rel_deg': 1,
    'obstacle_alpha': (10.0,),
    'boundary_alpha': (10.0,),
    'velocity_alpha': (),
})

cf_params = {
    'slack_gain': 1e24,
    'use_softplus': False,
    'softplus_gain': 2.0,
}

# 8 starts: spread around the map periphery, away from obstacles
starts = jnp.array([
    [-1.0, -8.5, 0.0, pi / 2],     # robot 0 — bottom-centre
    [-8.0,  0.0, 0.0, 0.0],        # robot 1 — far left
    [ 0.0, -8.0, 0.0, pi / 2],     # robot 2 — bottom, right of centre
    [ 7.0, -1.0, 0.0, pi],         # robot 3 — right side
    [-8.0, -7.0, 0.0, pi / 4],     # robot 4 — bottom-left
    [ 7.0,  2.0, 0.0, pi],         # robot 5 — right, mid
    [-1.0,  8.5, 0.0, -pi / 2],    # robot 6 — top-centre
    [ 9.0, -8.5, 0.0, pi / 2],     # robot 7 — bottom-right corridor
], dtype=jnp.float64)

# 8 goals: distinct, reachable, no obstacle directly between start and goal
goals = jnp.array([
    [ 3.0,  4.5],    # robot 0
    [ 4.0, -3.0],    # robot 1
    [-4.0,  0.0],    # robot 2
    [-3.0,  4.0],    # robot 3
    [ 3.0, -2.0],    # robot 4
    [-6.0, -2.0],    # robot 5
    [ 0.0, -5.0],    # robot 6
    [ 8.5,  5.0],    # robot 7
], dtype=jnp.float64)

control_gains = jnp.array([0.2, 1.0, 2.0])

# ============================================================
# Build template controller (single robot, goal = goals[0])
# ============================================================

print("Building template safety filter via cbfjax.from_config ...")

parts = cbfjax.from_config({
    'dynamics': 'unicycle',
    'barriers': {
        'map':  {'type': 'map', **map_config, 'cfg': barrier_cfg},
        'state': {'type': 'soft_composition', 'barriers': ['map'], 'cfg': barrier_cfg},
    },
    'filter': {
        'type': 'min_interv_cf',
        'action_dim': 2,
        'alpha': lambda h: 0.5 * h,
        'params': cf_params,
        'desired_control': UnicycleGoalControl(goal=goals[0], gains=control_gains),
    },
})

template_filter = parts.filter
dynamics = parts.dynamics
barrier = parts.barriers['state']

print(f"  dynamics: {type(dynamics).__name__}, barrier: {type(barrier).__name__}")
print(f"  filter: {type(template_filter).__name__}")

# ============================================================
# Build ensemble — N robots, each with a distinct goal
# ============================================================

print(f"\nBuilding ensemble of {N} robots with stack_ensemble ...")

ensemble = cbfjax.stack_ensemble(
    template_filter,
    where=lambda f: f._desired_control_module.goal,
    values=goals,
)

print(f"  ensemble type: {type(ensemble).__name__}")

# ============================================================
# Vectorised rollout: library diffrax ZOH integrator, vmapped per robot
# ============================================================

@eqx.filter_jit
def rollout_ensemble(ens, x0s):
    # (N, n_steps, state_dim) — ensemble ZOH rollout, per-robot state lanes
    return cbfjax.utils.get_ensemble_trajs_zoh(
        ens, x0s, timestep=dt, sim_time=T, method='dopri5')


print("\nCompiling + running rollout ...")
t0 = time()
trajs = rollout_ensemble(ensemble, starts)
xf = trajs[:, -1, :]
compile_run_time = time() - t0
print(f"  First call (compile + run): {compile_run_time:.2f} s")

t1 = time()
trajs2 = rollout_ensemble(ensemble, starts)
run_time = time() - t1
print(f"  Second call (run only):     {run_time:.3f} s")

# trajs shape: (N, n_steps, state_dim)
trajs_np = np.array(trajs)    # (N, n_steps, 4)
xf_np = np.array(xf)          # (N, 4)
goals_np = np.array(goals)    # (N, 2)
starts_np = np.array(starts)  # (N, 4)

# ============================================================
# Statistics
# ============================================================

final_dists = np.linalg.norm(xf_np[:, :2] - goals_np, axis=1)

# Min barrier along each trajectory — use the stacked barrier's hocbf via vmap
# trajs shape per robot: (n_steps, 4); vmap over robots then over time steps
@eqx.filter_jit
def compute_min_h(traj_batch):
    def min_h_one_robot(traj):
        h_vals = jax.vmap(barrier.hocbf)(traj)
        return jnp.min(h_vals)
    return jax.vmap(min_h_one_robot)(traj_batch)

min_h_vals = np.array(compute_min_h(trajs))

print(f"\n{'='*60}")
print(f"Per-robot final distance to goal and min barrier value:")
print(f"{'='*60}")
all_reached = True
all_safe = True
for i in range(N):
    reached = final_dists[i] < 0.3
    safe = min_h_vals[i] > 0.0
    if not reached:
        all_reached = False
    if not safe:
        all_safe = False
    print(f"  Robot {i}: dist_to_goal={final_dists[i]:.4f}  "
          f"min_h={min_h_vals[i]:.4f}  "
          f"reached={'YES' if reached else 'NO'}  "
          f"safe={'YES' if safe else 'NO'}")

print(f"{'='*60}")
print(f"All reached (<0.3): {'YES' if all_reached else 'NO'}")
print(f"All safe (h>0):     {'YES' if all_safe else 'NO'}")
print(f"{'='*60}")

# ============================================================
# Plot: trajectories + obstacle contours + goals
# ============================================================

print("\nGenerating trajectory plot ...")

# Build meshgrid barrier map (min_barrier for obstacle contours)
x_grid = np.linspace(-10.5, 10.5, 400)
y_grid = np.linspace(-10.5, 10.5, 400)
X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
pts = np.column_stack((X_grid.flatten(), Y_grid.flatten(),
                       np.zeros((X_grid.size, 2))))
pts_jax = jnp.array(pts, dtype=jnp.float64)
Z = np.array(jax.vmap(barrier.min_barrier)(pts_jax)).reshape(X_grid.shape)

colors = plt.cm.tab10(np.linspace(0, 1, N))

fig, ax = plt.subplots(figsize=(7, 7))

ax.contour(X_grid, Y_grid, Z, levels=[0], colors='red', linewidths=1.5)

for i in range(N):
    traj = trajs_np[i]  # (n_steps, 4)
    ax.plot(traj[:, 0], traj[:, 1], color=colors[i], linewidth=0.8, alpha=0.7)
    ax.plot(starts_np[i, 0], starts_np[i, 1], 'x', color=colors[i],
            markersize=7, markeredgewidth=1.5)
    ax.plot(goals_np[i, 0], goals_np[i, 1], '*', color=colors[i],
            markersize=10, markeredgewidth=1.0)
    ax.plot(xf_np[i, 0], xf_np[i, 1], '+', color=colors[i],
            markersize=7, markeredgewidth=1.5)

ax.set_xlabel(r'$q_{\rm x}$', fontsize=14)
ax.set_ylabel(r'$q_{\rm y}$', fontsize=14)
ax.set_xlim(-10.5, 10.5)
ax.set_ylim(-10.5, 10.5)
ax.set_aspect('equal', adjustable='box')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(labelsize=12)
ax.set_xticks([-10, -5, 0, 5, 10])
ax.set_yticks([-10, -5, 0, 5, 10])

from matplotlib.lines import Line2D
legend_els = [
    Line2D([0], [0], color='red', lw=1.5, label=r'$\mathcal{S}_{\rm s}$'),
    Line2D([0], [0], marker='x', color='k', lw=0, markersize=7, label=r'$x_0$'),
    Line2D([0], [0], marker='*', color='k', lw=0, markersize=9, label='Goal'),
    Line2D([0], [0], marker='+', color='k', lw=0, markersize=7, label=r'$x_f$'),
]
ax.legend(handles=legend_els, loc='upper center', bbox_to_anchor=(0.5, 1.10),
          ncol=4, frameon=False, fontsize=11)

plt.tight_layout()

current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
os.makedirs(os.path.join(script_dir, 'figs'), exist_ok=True)
fig_path = os.path.join(script_dir, f'figs/14_multi_robot_{current_time}.png')
plt.savefig(fig_path, dpi=200)
print(f"  Saved: {fig_path}")
plt.show()

print("\nSimulation complete!")
