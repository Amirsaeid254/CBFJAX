"""
Config-driven construction for CBFJAX: a named barrier namespace.

Every barrier gets a name; entries reference earlier entries by name; the
filter names the barrier it consumes. Entries nobody references are still
built — they are yours (plotting, analysis, debugging)::

    cfg = {
        'dynamics': 'unicycle',          # name, {'type': ..., **params}, instance
        'barriers': {
            'map':   {'type': 'map', 'geoms': [...], 'velocity': (idx, bounds),
                      'cfg': {...}},                    # -> the Map instance
            'state': {'type': 'soft_composition', 'barriers': ['map'],
                      'cfg': {...}},
            'rows':  {'type': 'multi_barrier', 'barriers': ['map'],
                      'cfg': {...}},
        },
        'filter': {'type': 'min_interv_qp', 'barrier': 'state',
                   'action_dim': 2, ...},               # XOR 'control'
    }
    system = cbfjax.from_config(cfg)
    system.filter                   # the ready controller
    system.barriers['rows']         # any named entry, used or not
    system.barriers['map']          # the Map (plotting)
    system.dynamics

Rules
-----
- Entries build in declaration order; a name reference must point to an
  already-defined entry.
- A 'map' entry's value is the Map instance. Referencing it in a 'barriers'
  list contributes its position + velocity member barriers; wiring it
  directly into a filter is an error (compose it first).
- 'barriers' reference lists accept names and built instances. Built
  Barrier/Map instances may also be used directly as entry values.
- Spec keys are the constructor kwargs; string values in barrier positions
  are name references.

Barrier entry types
-------------------
    {'type': 'map', 'geoms': [...], 'velocity': (idx, bounds), 'cfg': {...}}
    {'type': 'func', 'h': callable, 'rel_deg': 1, 'alphas': ..., 'cfg': {...}}
    {'type': 'soft_composition', 'barriers': [...], 'rule': 'i'|'u', 'cfg': {...}}
    {'type': 'hard_composition', 'barriers': [...], 'rule': 'i'|'u', 'cfg': {...}}
        # scalar softmin / exact-min reduction (closed-form filters need a
        # scalar barrier)
    {'type': 'multi_barrier', 'barriers': [...], 'cfg': {...}}
        # no reduction: one CBF constraint per member (QP filters)
    {'type': 'backup', 'state_barrier': <name>, 'backup_barriers': [<name>, ...],
     'backup_policies': [...], 'rel_deg': 1, 'cfg': {...}}
    {'type': 'flow', 'state_barrier': <name>, 'backup_barriers': [<name>, ...],
     'cfg': {...}}   # FlowBarrier over augmented state [x, theta, gamma];
                     # consumed by the 'parametric_flow' filter
    {'type': 'flow2', 'state_barrier': <name>, 'backup_barriers': [<name>, ...],
     'backup_policy': callable, 'cfg': {...}}
                     # FlowBarrier2 with backup-policy blended plan (cfg also
                     # reads 'blend_fraction'); consumed by 'parametric_flow2'
        # cfg: horizon, time_steps, integration_method, softmin_rho,
        # softmax_rho; backup filters also read epsilon, h_scale, feas_scale.
        # A terminal like h_terminal = state.hocbf + margin is just another
        # barrier: build the state barrier first (own config or by hand) and
        # define {'type': 'func', 'h': lambda x: state.hocbf(x) + margin(x)}.

Controllers
-----------
Exactly one of 'filter' (FILTER_TYPES) or 'control' (CONTROL_TYPES —
performance controllers such as iLQR/MPPI/NMPC; never barrier-wired) per
config; both may be omitted for barrier-only configs. Layered designs use one
config per layer, passing system.control on as 'desired_control'. NMPC
controllers require their post-construction .make() step.

Filter wiring: 'barrier' (and 'terminal_barrier') given as a string resolve
against the named entries. If 'barrier' is omitted and exactly one entry
holds a single Barrier, it is wired; with several, name one explicitly.
"""

from dataclasses import dataclass
from typing import Any, Dict

from .dynamics import (
    UnicycleDynamics,
    DoubleIntegratorDynamics,
    SingleIntegratorDynamics,
    BicycleDynamics,
    InvertedPendulumDynamics,
    UnicycleReducedOrderDynamics,
    Unicycle5thOrderDynamics,
)
from .safe_controls import (
    CFSafeControl,
    MinIntervCFSafeControl,
    InputConstCFSafeControl,
    MinIntervInputConstCFSafeControl,
    QPSafeControl,
    MinIntervQPSafeControl,
    InputConstQPSafeControl,
    MinIntervInputConstQPSafeControl,
    BackupSafeControl,
    MinIntervBackupSafeControl,
    ParametricFlowSafeControl,
    ParametricFlowSafeControl2,
    CADPSafeControl,
)
from .barriers import (
    Barrier,
    MultiBarriers,
    SoftCompositionBarrier,
    HardCompositionBarrier,
    BackupBarrier,
    FlowBarrier,
    FlowBarrier2,
)
from .utils.make_map import Map

__all__ = ['from_config', 'System']


# --------------------------------------------------------------- helpers

def _unknown(kind, value, choices):
    return ValueError(f"Unknown {kind} {value!r}. Available: {sorted(choices)}")


def _requires(spec_name, key):
    return ValueError(f"{spec_name!r} spec requires a {key!r} key")


def _check_keys(spec, valid, spec_name):
    unknown = set(spec) - set(valid)
    if unknown:
        raise ValueError(
            f"Unknown keys {sorted(unknown)} in {spec_name!r} spec. "
            f"Valid keys: {sorted(valid)}"
        )


def _resolve(ref, built, want):
    """A reference is a name (str) or a built instance."""
    if isinstance(ref, str):
        if ref not in built:
            raise ValueError(
                f"Unknown barrier name {ref!r} in {want}. "
                f"Defined (in order): {list(built)}"
            )
        return built[ref]
    return ref


def _resolve_members(refs, built, want):
    """Flatten name/instance references into a member Barrier list."""
    if not isinstance(refs, (list, tuple)):
        refs = [refs]
    members = []
    for ref in refs:
        value = _resolve(ref, built, want)
        if isinstance(value, Map):
            members.extend([*value.pos_barriers, *value.vel_barriers])
        elif isinstance(value, Barrier):
            members.append(value)
        else:
            raise TypeError(
                f"{want}: expected barrier name or Barrier/Map instance, "
                f"got {type(value).__name__}"
            )
    return members


def _resolve_single(ref, built, want):
    value = _resolve(ref, built, want)
    if isinstance(value, Map):
        raise ValueError(
            f"{want} needs a single barrier, but {ref!r} is a Map; compose "
            "its members first ({'type': 'soft_composition' | "
            "'hard_composition' | 'multi_barrier', 'barriers': [...]})"
        )
    if not isinstance(value, Barrier):
        raise TypeError(
            f"{want}: expected a Barrier, got {type(value).__name__}"
        )
    return value


# --------------------------------------------------------------- builders

def _build_map(name, spec, built, dynamics):
    _check_keys(spec, ('geoms', 'velocity', 'cfg'), 'map')
    if 'geoms' not in spec:
        raise _requires('map', 'geoms')
    if dynamics is None:
        raise ValueError("barrier type 'map' requires a 'dynamics' entry")
    return Map(dynamics=dynamics, cfg=spec.get('cfg', {}),
               barriers_info={k: v for k, v in spec.items()
                              if k in ('geoms', 'velocity')})


def _build_func(name, spec, built, dynamics):
    _check_keys(spec, ('h', 'rel_deg', 'alphas', 'cfg'), 'func')
    if 'h' not in spec:
        raise _requires('func', 'h')
    return Barrier(barrier_func=spec['h'],
                   rel_deg=spec.get('rel_deg', 1),
                   alphas=spec.get('alphas'),
                   dynamics=dynamics,
                   cfg=spec.get('cfg'))


def _build_soft_composition(name, spec, built, dynamics):
    _check_keys(spec, ('barriers', 'rule', 'cfg'), 'soft_composition')
    if not spec.get('barriers'):
        raise _requires('soft_composition', 'barriers')
    members = _resolve_members(spec['barriers'], built, f"'{name}'")
    return SoftCompositionBarrier(barriers=members,
                                  rule=spec.get('rule', 'intersection'),
                                  cfg=spec.get('cfg', {}))


def _build_hard_composition(name, spec, built, dynamics):
    _check_keys(spec, ('barriers', 'rule', 'cfg'), 'hard_composition')
    if not spec.get('barriers'):
        raise _requires('hard_composition', 'barriers')
    members = _resolve_members(spec['barriers'], built, f"'{name}'")
    return HardCompositionBarrier(barriers=members,
                                       rule=spec.get('rule', 'intersection'),
                                       cfg=spec.get('cfg', {}))


def _build_multi_barrier(name, spec, built, dynamics):
    _check_keys(spec, ('barriers', 'cfg'), 'multi_barrier')
    if not spec.get('barriers'):
        raise _requires('multi_barrier', 'barriers')
    members = _resolve_members(spec['barriers'], built, f"'{name}'")
    return MultiBarriers(barriers=members, cfg=spec.get('cfg', {}))


def _build_backup(name, spec, built, dynamics):
    _check_keys(spec, ('state_barrier', 'backup_barriers', 'backup_policies',
                       'rel_deg', 'cfg'), 'backup')
    for key in ('state_barrier', 'backup_barriers', 'backup_policies'):
        if key not in spec:
            raise _requires('backup', key)
    where = f"'{name}'"
    return BackupBarrier(
        state_barrier=_resolve_single(spec['state_barrier'], built, where),
        backup_barriers=[_resolve_single(t, built, where)
                         for t in spec['backup_barriers']],
        backup_policies=list(spec['backup_policies']),
        rel_deg=spec.get('rel_deg', 1),
        dynamics=dynamics,
        cfg=spec.get('cfg', {}),
    )


# --------------------------------------------------------------- registries

DYNAMICS_TYPES = {
    'unicycle': UnicycleDynamics,
    'double_integrator': DoubleIntegratorDynamics,
    'single_integrator': SingleIntegratorDynamics,
    'bicycle': BicycleDynamics,
    'inverted_pendulum': InvertedPendulumDynamics,
    'unicycle_reduced_order': UnicycleReducedOrderDynamics,
    'unicycle_5th_order': Unicycle5thOrderDynamics,
}

def _build_flow(name, spec, built, dynamics):
    _check_keys(spec, ('state_barrier', 'backup_barriers', 'cfg'), 'flow')
    for key in ('state_barrier', 'backup_barriers'):
        if key not in spec:
            raise _requires('flow', key)
    where = f"'{name}'"
    return (FlowBarrier.create_empty(cfg=dict(spec.get('cfg', {})))
            .assign_state_barrier(_resolve_single(spec['state_barrier'], built, where))
            .assign_backup_barrier([_resolve_single(b, built, where)
                                    for b in spec['backup_barriers']])
            .assign_dynamics(dynamics)
            .make())


def _build_flow2(name, spec, built, dynamics):
    _check_keys(spec, ('state_barrier', 'backup_barriers', 'backup_policy',
                       'cfg'), 'flow2')
    for key in ('state_barrier', 'backup_barriers', 'backup_policy'):
        if key not in spec:
            raise _requires('flow2', key)
    where = f"'{name}'"
    return (FlowBarrier2.create_empty(cfg=dict(spec.get('cfg', {})))
            .assign_state_barrier(_resolve_single(spec['state_barrier'], built, where))
            .assign_backup_barrier([_resolve_single(b, built, where)
                                    for b in spec['backup_barriers']])
            .assign_backup_policy(spec['backup_policy'])
            .assign_dynamics(dynamics)
            .make())


BARRIER_TYPES = {
    'map': _build_map,
    'func': _build_func,
    'soft_composition': _build_soft_composition,
    'hard_composition': _build_hard_composition,
    'multi_barrier': _build_multi_barrier,
    'backup': _build_backup,
    'flow': _build_flow,
    'flow2': _build_flow2,
}

# String values are class names with optional dependencies (acados/casadi,
# trajax), resolved on first use so `import cbfjax` succeeds without them.
FILTER_TYPES = {
    'cf': CFSafeControl,
    'min_interv_cf': MinIntervCFSafeControl,
    'input_const_cf': InputConstCFSafeControl,
    'min_interv_input_const_cf': MinIntervInputConstCFSafeControl,
    'qp': QPSafeControl,
    'min_interv_qp': MinIntervQPSafeControl,
    'input_const_qp': InputConstQPSafeControl,
    'min_interv_input_const_qp': MinIntervInputConstQPSafeControl,
    'backup': BackupSafeControl,
    'min_interv_backup': MinIntervBackupSafeControl,
    'parametric_flow': ParametricFlowSafeControl,
    'parametric_flow2': ParametricFlowSafeControl2,
    'cadp': CADPSafeControl,
    'nmpc': 'NMPCSafeControl',
    'quadratic_nmpc': 'QuadraticNMPCSafeControl',
    'ilqr': 'iLQRSafeControl',
    'quadratic_ilqr': 'QuadraticiLQRSafeControl',
}

# Performance (desired-control) controllers for the 'control' section.
# (module path, class name) — imported on first use.
CONTROL_TYPES = {
    'ilqr': ('cbfjax.controls.ilqr_control', 'iLQRControl'),
    'quadratic_ilqr': ('cbfjax.controls.ilqr_control', 'QuadraticiLQRControl'),
    'cilqr': ('cbfjax.controls.ilqr_control', 'ConstrainediLQRControl'),
    'quadratic_cilqr': ('cbfjax.controls.ilqr_control', 'QuadraticConstrainediLQRControl'),
    'mppi': ('cbfjax.controls.mppi_control', 'MPPIControl'),
    'nmpc': ('cbfjax.controls.nmpc_control', 'NMPCControl'),
    'quadratic_nmpc': ('cbfjax.controls.nmpc_control', 'QuadraticNMPCControl'),
}

_TOP_LEVEL_KEYS = ('dynamics', 'barriers', 'filter', 'control')


# --------------------------------------------------------------- assembly

@dataclass(frozen=True)
class System:
    """Everything one from_config call built. Absent sections are None."""

    dynamics: Any
    barriers: Dict[str, Any]
    control: Any
    filter: Any

    def __repr__(self):
        body = ', '.join(f'{name}={type(getattr(self, name)).__name__}'
                         for name in ('dynamics', 'control', 'filter')
                         if getattr(self, name) is not None)
        if self.barriers:
            body += f", barriers=[{', '.join(self.barriers)}]"
        return f'System({body})'


def _build_dynamics(spec):
    """Name or {'type': ..., **params} hits the registry; instances pass through."""
    if isinstance(spec, str):
        spec = {'type': spec}
    if not isinstance(spec, dict):
        return spec
    spec = dict(spec)
    dyn_type = spec.pop('type', None)
    if dyn_type not in DYNAMICS_TYPES:
        raise _unknown('dynamics type', dyn_type, DYNAMICS_TYPES)
    return DYNAMICS_TYPES[dyn_type](**spec)


def _build_barriers(section, dynamics):
    """The 'barriers' dict -> {name: built}, in declaration order."""
    built = {}
    if section is None:
        return built
    if not isinstance(section, dict):
        raise TypeError(
            f"'barriers' must be a dict of named specs, got {type(section).__name__}"
        )
    for name, spec in section.items():
        if isinstance(spec, (Barrier, Map)):
            built[name] = spec
            continue
        if not isinstance(spec, dict):
            raise TypeError(
                f"barrier entry {name!r}: expected a spec dict or Barrier/Map "
                f"instance, got {type(spec).__name__}"
            )
        spec = dict(spec)
        barrier_type = spec.pop('type', None)
        builder = BARRIER_TYPES.get(barrier_type)
        if builder is None:
            raise _unknown('barrier type', barrier_type, BARRIER_TYPES)
        built[name] = builder(name, spec, built, dynamics)
    return built


def _default_filter_barrier(built):
    """The single Barrier-valued entry, if there is exactly one."""
    singles = {n: v for n, v in built.items() if isinstance(v, Barrier)}
    if len(singles) == 1:
        return next(iter(singles.values()))
    if not singles:
        return None
    raise ValueError(
        f"several barriers defined ({list(singles)}); name one in the "
        "filter section: {'barrier': '<name>'}"
    )


def _resolve_filter(filt_type):
    if filt_type is None:
        raise _requires('filter', 'type')
    cls = FILTER_TYPES.get(filt_type)
    if cls is None:
        raise _unknown('filter type', filt_type, FILTER_TYPES)
    if isinstance(cls, str):
        from . import safe_controls
        cls = getattr(safe_controls, cls)
    return cls


def _resolve_control(ctrl_type):
    if ctrl_type is None:
        raise _requires('control', 'type')
    entry = CONTROL_TYPES.get(ctrl_type)
    if entry is None:
        raise _unknown('control type', ctrl_type, CONTROL_TYPES)
    from importlib import import_module
    module_name, cls_name = entry
    return getattr(import_module(module_name), cls_name)


def _build_controller(spec, resolver, dynamics, built):
    """Resolve a controller spec to an instance; instances pass through."""
    if not isinstance(spec, dict):
        return spec
    spec = dict(spec)
    cls = resolver(spec.pop('type', None))
    if dynamics is not None:
        spec.setdefault('dynamics', dynamics)
    for key in ('barrier', 'terminal_barrier'):
        if isinstance(spec.get(key), str):
            spec[key] = _resolve_single(spec[key], built, f"filter {key!r}")
    if 'barrier' not in spec and resolver is _resolve_filter:
        default = _default_filter_barrier(built)
        if default is not None:
            spec['barrier'] = default
    return cls(**spec)


def from_config(cfg: dict) -> System:
    """
    Build named barriers and one controller from a config dict (see module
    docstring).

    Returns:
        System(dynamics, barriers, control, filter) — `barriers` is the
        {name: built} dict.
    """
    unknown = set(cfg) - set(_TOP_LEVEL_KEYS)
    if unknown:
        raise ValueError(
            f"Unknown top-level cfg keys {sorted(unknown)}. "
            f"Valid keys: {sorted(_TOP_LEVEL_KEYS)}"
        )
    if 'filter' in cfg and 'control' in cfg:
        raise ValueError(
            "one controller per config: build the planner with a 'control' "
            "config first, then pass system.control as 'desired_control' in "
            "the filter config"
        )

    dynamics = _build_dynamics(cfg.get('dynamics'))
    built = _build_barriers(cfg.get('barriers'), dynamics)

    control = None
    if cfg.get('control') is not None:
        control = _build_controller(cfg['control'], _resolve_control,
                                    dynamics, built)

    filter_ = None
    if cfg.get('filter') is not None:
        filter_ = _build_controller(cfg['filter'], _resolve_filter,
                                    dynamics, built)

    return System(dynamics=dynamics, barriers=built,
                  control=control, filter=filter_)
