# CBFJAX — Control Barrier Functions in JAX

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-0.6+-orange.svg)](https://jax.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://img.shields.io/pypi/v/cbfjax.svg)](https://pypi.org/project/cbfjax/)

**CBFJAX** is a high-performance [JAX](https://jax.readthedocs.io/) implementation of
**Control Barrier Functions (CBFs)** for safety-critical control. It provides a clean,
functional, JIT-compatible API for building safe controllers — from simple closed-form
filters up to Backup-CBFs and NMPC with barrier constraints — and runs efficiently on
CPU and GPU.

This project is the JAX successor to the
[CBFTorch](https://github.com/pedramrabiee/cbftorch) framework.

---

## Features

- **Pure JAX, end-to-end JIT** — barriers, dynamics, and safety filters are all
  `equinox` modules with functional semantics; trajectory rollouts use
  [diffrax](https://github.com/patrick-kidger/diffrax).
- **Higher-Order CBFs (HOCBFs)** with automatic differentiation for arbitrary relative
  degree.
- **A toolbox of controllers and safe-control backends**:
  - Closed-form min-intervention safe control (`MinIntervCFSafeControl`)
  - QP-based safe control with slack variables (`MinIntervQPSafeControl`)
  - Input-constrained QP (`MinIntervInputConstQPSafeControl`)
  - Backup-CBF with forward invariance (`MinIntervBackupSafeControl`)
  - MPPI with barrier-aware cost (`MPPIControl`)
  - NMPC with barrier constraints (acados / do-mpc — optional)
  - Constrained iLQR with barrier-aware cost (trajax — optional)
- **Composable barrier algebra**: `MultiBarriers`, `SoftCompositionBarrier`,
  `NonSmoothCompositionBarrier`, `BackupBarrier`.
- **Built-in dynamics**: unicycle, single/double integrator, bicycle, inverted pendulum,
  reduced-order unicycle — plus a generic `AffineInControlDynamics` base.
- **Map editor** (`cbfjax-map-editor`) for visually authoring obstacle/boundary maps.
- **64-bit precision by default** for the numerical stability that CBF methods require.

---

## Installation

### From PyPI

```bash
pip install cbfjax
```

The core install is lightweight — it pulls in JAX, Equinox, Diffrax, qpax, NumPy, and
SciPy, and is sufficient for the closed-form, QP, and Backup-CBF safety filters.

### Optional extras

| Extra | Adds | Install |
|-------|------|---------|
| `examples` | matplotlib, animation deps | `pip install cbfjax[examples]` |
| `gpu` | JAX CUDA 12 wheels | `pip install cbfjax[gpu]` |
| `nmpc` | CasADi + do-mpc (IPOPT backend) | `pip install cbfjax[nmpc]` |
| `dev` | pytest, build, twine, ruff, black, mypy, … | `pip install cbfjax[dev]` |

The `nmpc` extra provides an IPOPT-based NMPC backend out of the box. For the
acados SQP backend, install [acados](https://docs.acados.org/) separately from source.

The `iLQR` controllers depend on Google's `trajax`, which is not on PyPI; install it
directly from GitHub:

```bash
pip install "trajax @ git+https://github.com/google/trajax.git"
```

NMPC and iLQR controllers are lazily imported, so the core package keeps working even
when these optional dependencies are not installed — the `ImportError` is only raised
when you actually instantiate the controller.

### From source

```bash
git clone https://github.com/amirsaeid254/cbfjax.git
cd cbfjax
pip install -e .[dev,examples]
```

---

## Quick Start

The pattern is the same for every safe controller in CBFJAX:

```text
dynamics  ──┐
            ├──► safety_filter ──► safe action u(x)
barrier   ──┤
            ├──► assign_desired_control(u_des)
desired u ──┘
```

### Minimal example — QP safety filter on a unicycle

```python
import jax.numpy as jnp
import cbfjax
from cbfjax.barriers import Barrier

# 1. Dynamics
dynamics = cbfjax.UnicycleDynamics()  # state: [x, y, v, theta], control: [a, omega]

# 2. Barrier: stay outside the unit disk centered at the origin
#    h(x) = ||p|| - 1.0 >= 0  (relative degree 2 for unicycle position)
barrier = (
    Barrier.create_empty()
    .assign(barrier_func=lambda x: jnp.linalg.norm(x[:2]) - 1.0, rel_deg=2)
    .assign_dynamics(dynamics)
)

# 3. Desired (nominal) controller — drive to a goal
goal = jnp.array([5.0, 5.0])
def desired_control(x):
    return 0.5 * jnp.array([goal[0] - x[0], goal[1] - x[1]])

# 4. QP-based min-intervention safety filter
safety_filter = (
    cbfjax.MinIntervQPSafeControl(
        action_dim=dynamics.action_dim,
        alpha=lambda h: 1.0 * h,
        params={"slack_gain": 200.0, "slacked": True},
    )
    .assign_dynamics(dynamics)
    .assign_state_barrier(barrier)
    .assign_desired_control(desired_control)
)

# 5. Query the safe action
x0 = jnp.array([[-2.0, -2.0, 0.0, 0.0]])     # batched (1, 4)
u_safe, _ = safety_filter.optimal_control(x0, safety_filter.get_init_state())
print(u_safe)
```

### Multiple barriers via `MultiBarriers`

```python
import jax.numpy as jnp
from cbfjax.barriers import Barrier, MultiBarriers
import cbfjax

dynamics = cbfjax.UnicycleDynamics()

# Two obstacle barriers + workspace boundary, each with dynamics already assigned.
def obstacle(center, radius):
    return lambda x: jnp.linalg.norm(x[:2] - jnp.array(center)) - radius

barriers = [
    Barrier.create_empty().assign(obstacle([2.0, 2.0], 0.5), rel_deg=2).assign_dynamics(dynamics),
    Barrier.create_empty().assign(obstacle([-1.0, 3.0], 0.5), rel_deg=2).assign_dynamics(dynamics),
    Barrier.create_empty().assign(
        lambda x: 10.0 - jnp.maximum(jnp.abs(x[0]), jnp.abs(x[1])), rel_deg=2
    ).assign_dynamics(dynamics),
]

# Pass infer_dynamics=True to pick up the dynamics from the first barrier.
multi = MultiBarriers.create_empty().add_barriers(barriers, infer_dynamics=True)
```

### Closed-loop simulation

```python
trajs = safety_filter.get_optimal_trajs(
    x0=x0,
    sim_time=10.0,
    timestep=0.01,
    method="euler",
)
print(trajs.shape)  # (T, batch, state_dim)
```

More end-to-end scripts live under `examples/unicycle/` (closed-form, QP, input-constrained
QP, NMPC, iLQR, hierarchical) and `examples/backup_examples/` (Backup-CBF).

```bash
cd examples/unicycle
python 03_unicycle_qp.py
```

---

## Map editor

CBFJAX ships with a browser-based visual editor for authoring obstacle/boundary maps:

```bash
cbfjax-map-editor
```

It opens an HTML canvas in your default browser where you can drop in cylinders,
ellipses, norm-boxes, and boundaries and export a CBFJAX `map_config.py`.

---

## Architecture

```
cbfjax/
├── barriers/                       # CBF & HOCBF
│   ├── barrier.py                  #  Single barrier
│   ├── multi_barrier.py            #  Multiple barriers
│   ├── composite_barrier.py        #  Soft / non-smooth composition
│   └── backup_barrier.py           #  Backup-CBF
├── dynamics/                       # Affine-in-control system dynamics
│   ├── base_dynamic.py
│   ├── unicycle.py
│   ├── unicycle_reduced_order.py
│   ├── double_integrator.py
│   ├── single_integrator.py
│   ├── bicycle.py
│   └── inverted_pendulum.py
├── controls/                       # Nominal/optimal controllers
│   ├── base_control.py
│   ├── mppi_control.py             #  MPPI (GPU-parallel, vmap+scan)
│   ├── ilqr_control.py             #  (optional: trajax)
│   ├── nmpc_control.py             #  (optional: casadi + acados/do-mpc)
│   └── control_types.py
├── safe_controls/                  # Safety filters
│   ├── base_safe_control.py
│   ├── closed_form_safe_control.py
│   ├── qp_safe_control.py
│   ├── backup_safe_control.py
│   ├── nmpc_safe_control.py        #  (optional)
│   └── ilqr_safe_control.py        #  (optional)
├── utils/
│   ├── integration.py              #  Diffrax-based ODE rollouts
│   ├── make_map.py                 #  Map / barrier factory
│   ├── jax2casadi/                 #  JAX → CasADi conversion (used by NMPC)
│   ├── map_editor/                 #  HTML map editor
│   ├── profile_utils.py
│   ├── run_map_editor.py
│   └── utils.py
└── config.py                       # JAX configuration helpers
```

---

## Key concepts

### Control Barrier Functions

For a control-affine system `ẋ = f(x) + g(x) u` and a safe set
`C = {x | h(x) ≥ 0}`, a barrier function `h` ensures forward invariance of `C`
whenever there exists `u` such that

```
L_f h(x) + L_g h(x) · u ≥ -α(h(x))
```

where `α` is a class-K function.

### Higher-Order CBFs

For barriers of relative degree `r > 1`, CBFJAX automatically constructs the HOCBF
series `ψ_0, ψ_1, …, ψ_r` from a user-provided list of class-K functions
(via `Barrier(...).assign(barrier_func, rel_deg=r, alphas=[α_1, …, α_r])`).

---

## Citation

If you use CBFJAX in your research, please cite it as:

```bibtex
@software{CBFJAX,
  author       = {Safari, Amirsaeid},
  title        = {{CBFJAX}: Control Barrier Functions in {JAX}},
  howpublished = {\url{https://github.com/amirsaeid254/cbfjax}},
  year         = {2025}
}
```

## Related work

- [CBFTorch](https://github.com/pedramrabiee/cbftorch) — PyTorch implementation of CBFs.
