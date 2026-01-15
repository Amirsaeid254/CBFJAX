# CBFJAX: Control Barrier Functions in JAX

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-0.4.20+-orange.svg)](https://jax.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

CBFJAX is a high-performance JAX implementation of Control Barrier Functions (CBFs) for safe control, adapted from the pioneering [CBFTorch](https://github.com/pedramrabiee/cbftorch) framework. This implementation leverages JAX's JIT compilation and functional programming paradigms to deliver superior computational performance for safety-critical control applications.


## Features

- **Pure JAX Implementation**: Fully JIT-compiled for maximum performance
- **Control Barrier Functions**: Complete CBF framework with Higher-Order CBFs (HOCBFs)
- **Multiple Safe Control Methods**:
  - Closed-form safe control
  - QP-based safe control with quadratic programming
  - Input-constrained QP safe control
  - Minimum intervention control
  - Backup Control Barrier Function 
- **Advanced Barrier Types**:
  - Single barriers with automatic differentiation
  - MultiBarriers for handling multiple constraints
  - Composite barriers with soft/hard composition
- **High Performance**: 64-bit precision, JIT compilation, and optimized algorithms
- **Modern Dependencies**: Diffrax for ODE solving, qpax for QP optimization

## Installation

Install from PyPI:
```bash
pip install cbfjax
```

For development installation:
```bash
git clone https://github.com/amirsaeid254/cbfjax.git
cd cbfjax
pip install -e .
```

## Quick Start

### Basic Usage

```python
import jax.numpy as jnp
import cbfjax

# Create unicycle dynamics
dynamics = cbfjax.dynamics.UnicycleDynamics()

# Create a simple barrier (e.g., avoid obstacles)
def barrier_func(x):
    return jnp.linalg.norm(x[:2]) - 1.0  # Stay outside unit circle

barrier = cbfjax.barriers.Barrier(
    barrier_func=barrier_func,
    rel_deg=1,
    alpha=lambda h: 0.5 * h
)

# Create safe control filter
safety_filter = cbfjax.safe_controls.MinIntervCFSafeControl(
    action_dim=dynamics.action_dim,
    alpha=lambda x: 0.5 * x
).assign_dynamics(dynamics).assign_state_barrier(barrier)

# Define desired control
def desired_control(x):
    return jnp.array([1.0, 0.1])  # Move forward, slight turn

safety_filter = safety_filter.assign_desired_control(desired_control)

# Compute safe control
state = jnp.array([0.5, 0.5, 0.0, 0.0])  # [x, y, v, theta]
safe_control, _, _ = safety_filter.safe_optimal_control(state)
print(f"Safe control: {safe_control}")
```

### QP-based Safe Control

```python
import cbfjax

# Create QP-based safety filter with multiple barriers
safety_filter = cbfjax.safe_controls.MinIntervQPSafeControl(
    action_dim=2,
    alpha=lambda x: 1.0 * x,
    params={
        'slack_gain': 200,
        'slacked': True,
        'use_softplus': False,
        'softplus_gain': 2.0
    }
)

# With input constraints
constrained_filter = cbfjax.safe_controls.MinIntervInputConstQPSafeControl(
    action_dim=2,
    alpha=lambda x: 1.0 * x,
    control_low=[-2.0, -1.0],   # Lower bounds
    control_high=[2.0, 1.0],    # Upper bounds
    params={'slack_gain': 200, 'slacked': True}
)
```

### MultiBarriers for Multiple Constraints

```python
import cbfjax

# Create multiple barriers
barriers = []
# Obstacle avoidance barriers
for center in [[2.0, 2.0], [-1.0, 3.0]]:
    barrier_func = lambda x, c=center: jnp.linalg.norm(x[:2] - jnp.array(c)) - 0.5
    barriers.append(cbfjax.barriers.Barrier(barrier_func, rel_deg=1))

# Boundary constraints
def boundary_barrier(x):
    return jnp.min(jnp.array([10.0 - jnp.abs(x[0]), 10.0 - jnp.abs(x[1])]))

barriers.append(cbfjax.barriers.Barrier(boundary_barrier, rel_deg=1))

# Combine into MultiBarriers
multi_barrier = cbfjax.barriers.MultiBarriers.create_empty()
multi_barrier = multi_barrier.add_barriers(barriers, infer_dynamics=True)
```

### Complete Example with Trajectory Simulation

```python
import jax.numpy as jnp
import cbfjax

# Setup
dynamics = cbfjax.dynamics.UnicycleDynamics()
barrier = cbfjax.barriers.Barrier(
    barrier_func=lambda x: jnp.linalg.norm(x[:2]) - 1.0,
    rel_deg=1
)

safety_filter = cbfjax.safe_controls.MinIntervCFSafeControl(
    action_dim=2,
    alpha=lambda x: 0.5 * x
).assign_dynamics(dynamics).assign_state_barrier(barrier)

# Desired control toward goal
goal = jnp.array([5.0, 5.0])
def desired_control(x):
    pos_error = goal - x[:2]
    return 0.5 * pos_error

safety_filter = safety_filter.assign_desired_control(desired_control)

# Simulate trajectory
x0 = jnp.array([[0.0, 0.0, 0.0, 0.0]])  # Initial state
trajectory = safety_filter.get_safe_optimal_trajs(
    x0=x0,
    sim_time=10.0,
    timestep=0.01,
    method='euler'
)

print(f"Trajectory shape: {trajectory.shape}")
```

## Examples

The `examples/` directory contains complete examples:

- `02_unicycle_cf.py`: Closed-form safe control for unicycle
- `03_unicycle_qp.py`: QP-based safe control
- `06_unicycle_input_constrained_qp.py`: Input-constrained QP safe control

Run examples:
```bash
cd examples/unicycle
python 02_unicycle_cf.py
```

## Architecture

```
cbfjax/
├── barriers/           # Barrier function implementations
│   ├── barrier.py         # Single barrier functions
│   ├── multi_barrier.py   # Multiple barrier handling
│   ├── composite_barrier.py # Barrier composition
│   └── backup_barrier.py    # Backup Barrier Function
├── dynamics/           # System dynamics
│   ├── base.py                    # Base dynamics classes
│   ├── unicycle.py               # Unicycle dynamics
│   ├── double_integrator.py      # Double integrator dynamics
│   ├── single_integrator.py      # Single integrator dynamics
│   ├── bicycle.py                # Bicycle dynamics
│   ├── inverted_pendulum.py      # Inverted pendulum dynamics
│   └── unicycle_reduced_order.py # Reduced-order unicycle dynamics
├── controls/           # General control implementations
│   └── base_control.py       # Base control classes
├── safe_controls/      # Safe control implementations (extends controls)
│   ├── base_safe_control.py      # Base safe control classes
│   ├── closed_form_safe_control.py # Analytical solutions
│   ├── qp_safe_control.py         # QP-based control
│   └── backup_safe_control.py     # Backup-CBF Control 
├── utils/              # Utilities and helpers
│   ├── integration.py     # ODE integration
│   ├── utils.py           # Mathematical utilities
│   ├── make_map.py        # Environment/map creation
│   └── profile_utils.py   # Profiling utilities for JAX
└── config.py           # Configuration and JAX setup
```

## Key Concepts

### Control Barrier Functions (CBFs)

CBFs provide safety guarantees by ensuring the system stays in a safe set. For a dynamical system ẋ = f(x) + g(x)u with safe set C = {x | h(x) ≥ 0}, the CBF condition is:

```
L_f h(x) + L_g h(x) u ≥ -α(h(x))
```

Where α is a class-K function ensuring forward invariance.

### Higher-Order CBFs (HOCBFs)

For barriers with relative degree > 1, HOCBFs extend the framework:

```
L_f^n h(x) + L_g L_f^{n-1} h(x) u ≥ -α(ψ(x))
```

## Citation

If you use CBFJAX in your research, please cite it as:

```bibtex
@software{CBFJAX,
  author       = {Safari, Amirsaeid},
  title        = {{CBFJAX}: Control Barrier Functions in {JAX}},
  howpublished = {[Online]. Available: \url{https://github.com/amirsaeid254/cbfjax}},
  year         = {2025}
}
```

## Related Work

- [CBFTorch](https://github.com/pedramrabiee/cbftorch): PyTorch implementation of CBFs
