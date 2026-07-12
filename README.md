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
  `HardCompositionBarrier`, `BackupBarrier`.
- **Config-driven construction** (`cbfjax.from_config`): a named barrier
  namespace — define barriers by name, reference them by name, wire one into
  the filter; unused entries stay available for plotting/analysis.
- **Built-in dynamics**: unicycle, single/double integrator, bicycle, inverted pendulum,
  reduced-order unicycle — plus a generic `AffineInControlDynamics` base.
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

Everything is built from one config dict — named barriers, referenced by
name, and one controller consuming the barrier it names:

```text
dynamics ──► barriers {name: spec, ...} ──► filter ──► safe action u(x)
                                              ▲
                       desired_control (goal controller / iLQR / MPPI planner)
```

### Minimal example — QP safety filter on a unicycle

```python
import jax.numpy as jnp
import cbfjax

goal = jnp.array([5.0, 5.0])

system = cbfjax.from_config({
    # 1. Dynamics — state [x, y, v, theta], control [a, omega]
    'dynamics': 'unicycle',

    # 2. Barriers — every barrier gets a name
    'barriers': {
        # stay outside the unit disk at the origin (relative degree 2)
        'disk': {'type': 'func',
                 'h': lambda x: jnp.linalg.norm(x[:2]) - 1.0,
                 'rel_deg': 2, 'alphas': (10.0,)},
    },

    # 3. QP-based min-intervention safety filter consuming 'disk'
    'filter': {
        'type': 'min_interv_qp',
        'barrier': 'disk',
        'action_dim': 2,
        'alpha': lambda h: 1.0 * h,
        'params': {'slack_gain': 200.0, 'slacked': True, 'qp_solver': 'qpax'},
        'desired_control': lambda x: 0.5 * jnp.array([goal[0] - x[0],
                                                      goal[1] - x[1]]),
    },
})

# 4. Query the safe action (single state in, single action out)
x0 = jnp.array([-2.0, -2.0, 0.0, 0.0])
u_safe, _ = system.filter.optimal_control(x0, system.filter.get_init_state())
print(u_safe)
```

### Maps and compositions

A `map` entry turns a geometric world description into member barriers, and
compositions reduce them: `soft_composition` (scalar softmin — what
closed-form filters need), `hard_composition` (scalar exact min), or
`multi_barrier` (one QP constraint per member). Entries nobody consumes are
still built — handy for plotting and analysis.

```python
cfg = {
    'dynamics': 'unicycle',
    'barriers': {
        'map': {'type': 'map',
                'geoms': (
                    ('cylinder', {'center': (2.0, 2.0), 'radius': 0.5}),
                    ('norm_box', {'center': (-1.0, 3.0), 'size': (1.0, 1.0)}),
                    ('norm_boundary', {'center': (0.0, 0.0), 'size': (10.0, 10.0)}),
                ),
                'velocity': (2, (-2.0, 2.0)),
                'cfg': {'softmin_rho': 20, 'pos_barrier_rel_deg': 2,
                        'vel_barrier_rel_deg': 1, 'obstacle_alpha': (10.0,),
                        'boundary_alpha': (10.0,), 'velocity_alpha': ()}},
        'rows': {'type': 'multi_barrier', 'barriers': ['map']},
    },
    'filter': {'type': 'min_interv_input_const_qp', 'barrier': 'rows', ...},
}
system = cbfjax.from_config(cfg)
system.barriers['map']     # the Map instance — plotting, member access
system.barriers['rows']    # the MultiBarriers the filter consumes
```

### Closed-loop simulation

```python
trajs = system.filter.get_optimal_trajs(
    x0=x0[None],           # (batch, state_dim)
    sim_time=10.0,
    timestep=0.01,
    method='euler',
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

## Architecture

```
cbfjax/
├── barriers/                       # CBF & HOCBF
│   ├── barrier.py                  #  Single barrier
│   ├── multi_barrier.py            #  Multiple barriers
│   ├── composite_barrier.py        #  Soft / hard composition
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
├── factory.py                      # from_config: named-barrier construction
├── utils/
│   ├── integration.py              #  Diffrax-based ODE rollouts
│   ├── make_map.py                 #  Map / barrier factory
│   ├── jax2casadi/                 #  JAX → CasADi conversion (used by NMPC)
│   ├── profile_utils.py
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
(via `Barrier(barrier_func=h, rel_deg=r, alphas=[α_1, …, α_r], dynamics=dyn)`
or a `{'type': 'func', 'h': h, 'rel_deg': r, 'alphas': [...]}` config entry).

---

## Citation

If you use CBFJAX in your research, please cite it as:

```bibtex
@article{safari2026predicted,
  title={Predicted-Flow Control Barrier Functions for Real-Time Safe Optimal Control},
  author={Safari, Amirsaeid and Hoagg, Jesse B},
  journal={arXiv preprint arXiv:2606.00297},
  year={2026}
}
```

## Related work

- [CBFTorch](https://github.com/pedramrabiee/cbftorch) — PyTorch implementation of CBFs.
