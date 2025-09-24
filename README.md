# JAX FlowBarrier

High-performance JAX implementation of FlowBarrier for safe control with Control Barrier Functions (CBFs).

## Features

- **Pure JAX Implementation**: JIT-compiled for maximum performance
- **Analytical Derivatives**: Hand-derived gradients for critical paths
- **Production Ready**: Single-state processing, float64 precision
- **Modern Stack**: Diffrax for ODE solving, Qpax for QP optimization
- **No Batching**: Optimized for single-trajectory processing

## Installation

```bash
pip install -e .
```

For GPU support:
```bash
pip install -e ".[gpu]"
```

For development:
```bash
pip install -e ".[dev]"
```

## Quick Start

```python
import jax.numpy as jnp
from jax_flowbarrier import FlowBarrier, UnicycleDynamics

# Create dynamics
dynamics = UnicycleDynamics()

# Create FlowBarrier
flow_barrier = FlowBarrier(
    dynamics=dynamics,
    horizon=4.0,
    time_steps=0.05,
    control_param_num=80
)

# Compute safe control
state = jnp.array([0.0, 0.0, 0.0, 0.0])  # [x, y, v, theta]
control, info = flow_barrier.safe_control(state)
```

## Architecture

- `core/`: Core dynamics, barriers, and control modules
- `integration/`: ODE solving and trajectory computation
- `optimization/`: QP solving and cost functions
- `utils/`: Utilities, types, and configuration
- `examples/`: Usage examples and demos

## Performance

Designed for >10x speedup over PyTorch implementations through:
- JIT compilation of entire pipeline
- Analytical derivatives
- Memory-efficient algorithms
- Float64 precision throughout