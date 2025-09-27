"""
Minimum intervention QP-based safe control for unicycle dynamics using MultiBarriers.
Demonstrates MultiBarriers creation and QP safe control synthesis.
"""

import jax
import jax.numpy as jnp
import matplotlib as mpl
from math import pi
import numpy as np
from functools import partial
from time import time
import datetime
import os
from immutabledict import immutabledict

# Configure JAX for CPU computation
jax.config.update("jax_platforms", "cpu")
jax.config.update("jax_enable_x64", True)

# CBFJAX imports
from cbfjax.dynamics.unicycle import UnicycleDynamics
from cbfjax.utils.make_map import Map
from cbfjax.barriers.multi_barrier import MultiBarriers
from cbfjax.safe_controls.qp_safe_control import MinIntervQPSafeControl
from map_config import map_config
from unicycle_desired_control import desired_control

# Get the directory of this script for saving figures
script_dir = os.path.dirname(os.path.abspath(__file__))
import matplotlib.pyplot as plt

mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'

# Control gains
control_gains = immutabledict(k1=0.2, k2=1.0, k3=2.0)

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

# Instantiate dynamics
dynamics = UnicycleDynamics()

# Create barrier map and get individual barriers
map_ = Map(barriers_info=map_config, dynamics=dynamics, cfg=cfg).create_barriers()
pos_barriers, vel_barriers = map_.get_barriers()

# Create MultiBarriers and add all barriers
barrier = MultiBarriers.create_empty(cfg=cfg)
barrier = barrier.add_barriers([*pos_barriers, *vel_barriers], infer_dynamics=True)

# Goal positions
goal_pos = jnp.array([
    [3.0, 4.5],
    [-7.0, 0.0],
    [7.0, 1.5],
    [-1.0, 7.0]
])

goal_pos = jnp.array([
    [3.0, 4.5]])


# Initial conditions
x0 = jnp.tile(jnp.array([-1.0, -8.5, 0.0, pi / 2]), (goal_pos.shape[0], 1))
timestep = 0.01
sim_time = 20.0

# Test barrier function compilation
print("Testing MultiBarriers compilation...")
print(jax.jit(barrier.hocbf)(x0))

# Make safety filter and assign dynamics and barrier
safety_filter = MinIntervQPSafeControl(
    action_dim=dynamics.action_dim,
    alpha=lambda x: 0.5 * x,
    params={
        'slack_gain': 1e24,
        'slacked': False,
        'use_softplus': False,
        'softplus_gain': 2.0
    }
).assign_dynamics(dynamics=dynamics).assign_state_barrier(barrier=barrier)

# Assign desired control based on the goal positions
safety_filter = safety_filter.assign_desired_control(
    desired_control=lambda x: desired_control(x, goal_pos)
)

safety_filter._safe_optimal_control_single(x0.squeeze(0))

print("Starting trajectory simulation...")
print(f"  - Device: {jax.devices()[0]}")
# Simulate trajectories
start_time = time()
trajs = safety_filter.get_safe_optimal_trajs(x0=x0, sim_time=sim_time, timestep=timestep, method='euler')
simulation_time = time() - start_time
print(f"Simulation completed in {simulation_time:.4f} seconds")

# Rearrange trajs to match CBFTorch format
# trajs shape: (time_steps, batch, state_dim) -> list of (time_steps, state_dim) per trajectory
trajs_list = [trajs[:, i, :] for i in range(goal_pos.shape[0])]

# Get actions values along the trajs
print("Computing control actions and barrier values...")
actions = []
des_ctrls = []
h_vals = []
min_barriers = []
min_constraint = []

for i, traj in enumerate(trajs_list):
    # Get actions for this trajectory
    action_vals, _, _ = safety_filter.safe_optimal_control(traj)
    actions.append(action_vals)

    # Get desired control for this trajectory
    goal_single = jnp.tile(goal_pos[i], (traj.shape[0], 1))
    # Compute barrier values using MultiBarriers
    h_vals.append(barrier.hocbf(traj))
    min_barriers.append(barrier.get_min_barrier_at(traj))
    min_constraint.append(barrier.min_barrier(traj))

print("Generating plots...")

############
#  Plots   #
############

current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Create meshgrid for contour plot
x = np.linspace(-10.5, 10.5, 500)
y = np.linspace(-10.5, 10.5, 500)
X, Y = np.meshgrid(x, y)
points = np.column_stack((X.flatten(), Y.flatten()))
points_with_vel = np.column_stack((points, np.zeros((points.shape[0], 2))))  # Add zero velocity and angle
points_jax = jnp.array(points_with_vel, dtype=jnp.float32)

# Use composite barrier for contour plot
map_composite = map_.create_barriers()
Z = map_composite.barrier.min_barrier(points_jax)
Z = np.array(Z).reshape(X.shape)

# Create trajectory plot
fig, ax = plt.subplots(figsize=(6, 6))

contour_plot = ax.contour(X, Y, Z, levels=[0], colors='red')
# Adding a custom legend handle for the contour
from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color='red', lw=1.5)]

ax.set_xlabel(r'$q_{\\rm x}$', fontsize=16)
ax.set_ylabel(r'$q_{\\rm y}$', fontsize=16)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_aspect('equal', adjustable='box')
ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=16)
ax.set_xticks([-10, -5, 0, 5, 10])
ax.set_yticks([-10, -5, 0, 5, 10])

# Plot initial condition
ax.plot(trajs_list[0][0, 0], trajs_list[0][0, 1], 'x', color='blue', markersize=8, label=r'$x_0$')

# Plot trajectories and goals
for i in range(goal_pos.shape[0]):
    ax.plot(goal_pos[i, 0], goal_pos[i, 1], '*', markersize=10, color='limegreen',
            label='Goal' if i == 0 else None)
    ax.plot(trajs_list[i][-1, 0], trajs_list[i][-1, 1], '+', color='blue', markersize=8,
            label=r'$x_f$' if i == 0 else None)
    ax.plot(trajs_list[i][:, 0], trajs_list[i][:, 1],
            label='Trajectories' if i == 0 else None, color='black')

# Creating the legend
handles, labels = ax.get_legend_handles_labels()
handles.insert(0, custom_lines[0])
labels.insert(0, r'$\\mathcal{S}_{\\rm s}$')

ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=3, frameon=False, fontsize=12)

custom_order = [r'$\\mathcal{S}_{\\rm s}$', 'Goal', 'Trajectories', r'$x_0$', r'$x_f$']
handle_dict = dict(zip(labels, handles))
ordered_handles = [handle_dict[label] for label in custom_order]
ordered_labels = custom_order

plt.tight_layout()

# Create directory if it doesn't exist
os.makedirs(os.path.join(script_dir, 'figs'), exist_ok=True)

# Save the contour plot
plt.savefig(os.path.join(script_dir, f'figs/03_Trajectories_QP_Safe_Control_{current_time}.png'), dpi=200)
plt.show()

# Calculate time array
num_points = trajs_list[0].shape[0]
time_array = np.linspace(0.0, (num_points - 1) * timestep, num_points)

# Create subplot for states and action variables
fig, axs = plt.subplots(5, 1, figsize=(8, 8))

# Plot state variables
axs[0].plot(time_array, trajs_list[0][:, 0], label=r'$q_{\\rm x}$', color='red')
axs[0].plot(time_array, trajs_list[0][:, 1], label=r'$q_{\\rm y}$', color='blue')
axs[0].plot(time_array, np.ones(time_array.shape) * goal_pos[0, 0], label=r'$q_{\\rm d, x}$', color='red', linestyle=':')
axs[0].plot(time_array, np.ones(time_array.shape) * goal_pos[0, 1], label=r'$q_{\\rm d, y}$', color='blue', linestyle=':')
axs[0].legend(loc='lower center', ncol=4, frameon=False, fontsize=14)

axs[0].set_ylabel(r'$q_{\\rm x}, q_{\\rm y}$', fontsize=16)
axs[0].legend(fontsize=14, loc='lower center', ncol=4, frameon=False)
axs[0].tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)

axs[1].plot(time_array, trajs_list[0][:, 2], label='v', color='black')
axs[1].set_ylabel(r'$v$', fontsize=16)

axs[2].plot(time_array, trajs_list[0][:, 3], label='theta', color='black')
axs[2].set_ylabel(r'$\\theta$', fontsize=16)

# Plot actions
axs[3].plot(time_array, actions[0][:, 0], label=r'$u_1$', color='black')
# axs[3].plot(time_array, des_ctrls[0][:, 0], color='red', linestyle='--', label=r'$u_{{\\rm d}_1}$')
axs[3].legend(loc='lower center', ncol=2, frameon=False, fontsize=14)
axs[3].set_ylabel(r'$u_1$', fontsize=16)

axs[4].plot(time_array, actions[0][:, 1], label=r'$u_2$', color='black')
# axs[4].plot(time_array, des_ctrls[0][:, 1], color='red', linestyle='--', label=r'$u_{{\\rm d}_2}$')
axs[4].legend(loc='lower center', ncol=2, frameon=False, fontsize=14)
axs[4].set_ylabel(r'$u_2$', fontsize=16)

axs[4].set_xlabel(r'$t~(\\rm {s})$', fontsize=16)

for i in range(4):
    axs[i].tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)

# Format axes
for ax in axs:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=16)
    ax.set_xlim(time_array[0], time_array[-1])

# Adjust layout and save the combined plot
plt.subplots_adjust(wspace=0, hspace=0.2)
plt.tight_layout()

plt.savefig(os.path.join(script_dir, f'figs/03_States_QP_Safe_Control_{current_time}.png'), dpi=200)
plt.show()

# Barrier values plot
fig, axs = plt.subplots(3, 1, figsize=(8, 4.5))

# Plot barrier values (MultiBarriers HOCBF)
axs[0].plot(time_array, h_vals[0][:, 0, 0], color='black')  # First barrier from MultiBarriers
axs[0].set_ylabel(r'$h$', fontsize=16)

axs[1].plot(time_array, min_barriers[0], color='black')
axs[1].set_ylabel(r'$\\min b_{j, i}$', fontsize=16)

axs[2].plot(time_array, min_constraint[0], color='black')
axs[2].set_ylabel(r'$\\min h_j$', fontsize=16)

axs[2].set_xlabel(r'$t~(\\rm {s})$', fontsize=16)

for i in range(2):
    axs[i].tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)

for i, ax in enumerate(axs):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=16)
    ax.set_xlim(time_array[0], time_array[-1])
    if i != 0:
        ax.set_yticks([0, 0.5])

plt.subplots_adjust(wspace=0, hspace=0.2)
plt.tight_layout()

plt.savefig(os.path.join(script_dir, f'figs/03_Barriers_QP_Safe_Control_{current_time}.png'), dpi=200)
plt.show()

print(f"Simulation complete! Plots saved with timestamp: {current_time}")
print(f"Total simulation time: {simulation_time:.4f} seconds")