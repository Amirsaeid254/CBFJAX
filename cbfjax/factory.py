"""
Config-driven construction for CBFJAX.

cbfjax.from_config(cfg) builds one controller per config dict, each section a
thin layer over the constructors::

    cfg = {
        'dynamics': 'unicycle',          # name, {'type': ..., **params}, or instance
        'barrier': {...barrier spec...}, # or a Barrier instance
        'safety_filter': {               # XOR 'control', see below
            'type': 'min_interv_qp',
            'action_dim': 2,
            'alpha': ...,                # remaining keys forward to the constructor
            'desired_control': ...,
            'params': {'slack_gain': 200, 'qp_solver': 'qpax'},
        },
    }
    parts = cbfjax.from_config(cfg)
    parts.safety_filter             # the ready filter
    parts.dynamics, parts.barrier   # built instances
    parts.map                       # the Map, when exactly one was built
    parts.maps                      # every Map built during the config

A config holds exactly ONE controller section: 'safety_filter' (FILTER_TYPES)
or 'control' (CONTROL_TYPES — performance controllers such as iLQR/MPPI/NMPC).
Layered designs use one config per layer; pass the planner instance on::

    planner = cbfjax.from_config({
        'dynamics': dyn,
        'control': {'type': 'quadratic_ilqr', 'action_dim': 2, ...},
    }).control
    filt = cbfjax.from_config({
        'dynamics': dyn,
        'barrier': {...},
        'safety_filter': {..., 'desired_control': planner},
    }).safety_filter

A 'barrier' section in a 'control' config is built and returned but never
auto-wired into the controller; pass it explicitly (e.g. CiLQR's barrier=).
NMPC controllers still require their post-construction .make() step.

Barrier specs are recursive: wherever a barrier is expected, either a built
Barrier instance or a {'type': ...} spec dict is accepted.

    {'type': 'map', 'geoms': [...], 'velocity': (idx, bounds),
     'composition': 'soft' | 'hard' | 'multi', 'cfg': {...}}

    {'type': 'barrier', 'func': h, 'rel_deg': 2, 'alphas': (10.0,), 'cfg': {...}}

    {'type': 'soft' | 'hard' | 'multi',     # combine ANY barriers
     'barriers': [<spec | instance>, ...],  # a 'map' spec/instance in this list
     'rule': 'intersection',                #   expands to its member barriers
     'cfg': {...}}                          # ('rule' not valid for 'multi')

    {'type': 'backup',
     'state_barrier': <barrier spec | instance | list of either>,
     'backup_policies': [pi_1, ...],
     'backup_barriers': [<barrier spec | instance | state_margin spec>, ...],
     'cfg': {...}}   # horizon, time_steps, integration_method, softmin_rho,
                     # softmax_rho; backup filters also read epsilon, h_scale,
                     # feas_scale from this cfg

    {'type': 'state_margin', 'margin': lambda x: ...,  # only inside 'backup':
     'rel_deg': 1, 'alphas': ..., 'cfg': {...}}        # terminal barrier =
                                                       # state_barrier.hocbf + margin
    {'type': 'lidar', ...}   # reserved: built at runtime from sensor data

Wiring rule (the only one): the built barrier is passed to the filter
constructor as barrier=; an explicit 'barrier' key inside 'safety_filter' wins.
"""

from dataclasses import dataclass
from typing import Any, Optional, Tuple

from .dynamics import (
    UnicycleDynamics,
    DoubleIntegratorDynamics,
    SingleIntegratorDynamics,
    BicycleDynamics,
    InvertedPendulumDynamics,
    UnicycleReducedOrderDynamics,
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
)
from .barriers import (
    Barrier,
    MultiBarriers,
    SoftCompositionBarrier,
    NonSmoothCompositionBarrier,
    BackupBarrier,
)
from .utils.make_map import Map

__all__ = ['from_config', 'build_barrier', 'Parts']


DYNAMICS_TYPES = {
    'unicycle': UnicycleDynamics,
    'double_integrator': DoubleIntegratorDynamics,
    'single_integrator': SingleIntegratorDynamics,
    'bicycle': BicycleDynamics,
    'inverted_pendulum': InvertedPendulumDynamics,
    'unicycle_reduced_order': UnicycleReducedOrderDynamics,
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

# Barrier spec types: map, barrier, soft, hard, multi, backup, state_margin,
# lidar. Builders below register themselves via @_barrier_type.
BARRIER_TYPES = {}

_TOP_LEVEL_KEYS = ('dynamics', 'barrier', 'control', 'safety_filter')

COMPOSITIONS = ('soft', 'hard', 'multi')


def _barrier_type(name):
    def register(builder):
        if name in BARRIER_TYPES:
            raise ValueError(f"barrier type {name!r} already registered")
        BARRIER_TYPES[name] = builder
        return builder
    return register


@dataclass(frozen=True)
class Parts:
    """Built instances of one from_config call. Absent sections are None."""

    dynamics: Any
    map: Optional[Map]
    maps: Tuple[Map, ...]
    barrier: Optional[Barrier]
    control: Any
    safety_filter: Any

    def __repr__(self):
        fields = ('dynamics', 'map', 'barrier', 'control', 'safety_filter')
        body = ', '.join(f'{name}={type(getattr(self, name)).__name__}'
                         for name in fields if getattr(self, name) is not None)
        return f'Parts({body})'


def _unknown(kind, value, choices):
    return ValueError(f"Unknown {kind} {value!r}. Available: {sorted(choices)}")


def _requires(spec_name, key):
    return ValueError(f"{spec_name!r} spec requires a {key!r} key")


class _BuildContext:
    """Carries intermediates across the recursive build."""

    def __init__(self):
        self.maps = []
        self.state_barrier = None


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


def _check_keys(spec, valid, spec_name):
    unknown = set(spec) - set(valid)
    if unknown:
        raise ValueError(
            f"Unknown keys {sorted(unknown)} in {spec_name!r} spec. "
            f"Valid keys: {sorted(valid)}"
        )


def _make_map(spec, dynamics, ctx):
    """Build a Map from the geometry keys of a validated map spec."""
    if 'geoms' not in spec:
        raise _requires('map', 'geoms')
    if dynamics is None:
        raise ValueError("barrier type 'map' requires a 'dynamics' entry")
    map_ = Map(dynamics=dynamics, cfg=spec.get('cfg', {}),
               barriers_info={k: v for k, v in spec.items()
                              if k in ('geoms', 'velocity')})
    ctx.maps.append(map_)
    return map_


@_barrier_type('map')
def _build_map_barrier(spec, dynamics, ctx):
    """Sugar over the combinators for the single-map case.

    'soft'/'hard' return the compositions the Map already built; 'multi'
    delegates to the 'multi' combinator over the map's member barriers.
    """
    _check_keys(spec, ('geoms', 'velocity', 'composition', 'cfg'), 'map')
    composition = spec.get('composition', 'soft')
    if composition not in COMPOSITIONS:
        raise _unknown('composition', composition, COMPOSITIONS)

    map_ = _make_map(spec, dynamics, ctx)
    if composition == 'soft':
        return map_.barrier
    if composition == 'hard':
        return map_.map_barrier  # position barriers only
    return _build_multi(
        {'barriers': [*map_.pos_barriers, *map_.vel_barriers],
         'cfg': spec.get('cfg', {})},
        dynamics, ctx,
    )


def _build_members(specs, dynamics, ctx):
    """Expand a 'barriers' list into member Barrier instances.

    Map specs (without 'composition') and Map instances contribute their
    member barriers; everything else contributes one barrier.
    """
    members = []
    for s in specs:
        if isinstance(s, Map):
            ctx.maps.append(s)
            members.extend([*s.pos_barriers, *s.vel_barriers])
        elif isinstance(s, dict) and s.get('type') == 'map':
            s = {k: v for k, v in s.items() if k != 'type'}
            _check_keys(s, ('geoms', 'velocity', 'cfg'), 'map member')
            map_ = _make_map(s, dynamics, ctx)
            members.extend([*map_.pos_barriers, *map_.vel_barriers])
        else:
            members.append(_build_barrier(s, dynamics, ctx))
    return members


def _combinator_members(spec, valid_keys, spec_name, dynamics, ctx):
    _check_keys(spec, valid_keys, spec_name)
    if not spec.get('barriers'):
        raise _requires(spec_name, 'barriers')
    return _build_members(spec['barriers'], dynamics, ctx)


@_barrier_type('soft')
def _build_soft(spec, dynamics, ctx):
    members = _combinator_members(spec, ('barriers', 'rule', 'cfg'), 'soft', dynamics, ctx)
    return SoftCompositionBarrier(barriers=members,
                                  rule=spec.get('rule', 'intersection'),
                                  cfg=spec.get('cfg', {}))


@_barrier_type('hard')
def _build_hard(spec, dynamics, ctx):
    members = _combinator_members(spec, ('barriers', 'rule', 'cfg'), 'hard', dynamics, ctx)
    return NonSmoothCompositionBarrier(barriers=members,
                                       rule=spec.get('rule', 'intersection'),
                                       cfg=spec.get('cfg', {}))


@_barrier_type('multi')
def _build_multi(spec, dynamics, ctx):
    members = _combinator_members(spec, ('barriers', 'cfg'), 'multi', dynamics, ctx)
    return MultiBarriers(barriers=members, cfg=spec.get('cfg', {}))


@_barrier_type('barrier')
def _build_leaf_barrier(spec, dynamics, ctx):
    _check_keys(spec, ('func', 'rel_deg', 'alphas', 'cfg'), 'barrier')
    if 'func' not in spec:
        raise _requires('barrier', 'func')
    return Barrier(
        barrier_func=spec['func'],
        rel_deg=spec.get('rel_deg', 1),
        alphas=spec.get('alphas'),
        dynamics=dynamics,
        cfg=spec.get('cfg'),
    )


@_barrier_type('state_margin')
def _build_state_margin(spec, dynamics, ctx):
    if ctx.state_barrier is None:
        raise ValueError(
            "'state_margin' barriers are only valid inside a 'backup' spec's "
            "'backup_barriers' list"
        )
    _check_keys(spec, ('margin', 'rel_deg', 'alphas', 'cfg'), 'state_margin')
    if 'margin' not in spec:
        raise _requires('state_margin', 'margin')
    margin = spec['margin']
    state_barrier = ctx.state_barrier

    def terminal_barrier(x):
        return state_barrier.hocbf(x) + margin(x)

    return Barrier(
        barrier_func=terminal_barrier,
        rel_deg=spec.get('rel_deg', 1),
        alphas=spec.get('alphas'),
        dynamics=dynamics,
        cfg=spec.get('cfg'),
    )


@_barrier_type('backup')
def _build_backup_barrier(spec, dynamics, ctx):
    _check_keys(spec, ('state_barrier', 'backup_policies', 'backup_barriers',
                       'rel_deg', 'cfg'), 'backup')
    for key in ('state_barrier', 'backup_policies', 'backup_barriers'):
        if key not in spec:
            raise _requires('backup', key)
    backup_cfg = spec.get('cfg', {})

    sb_spec = spec['state_barrier']
    if isinstance(sb_spec, (list, tuple)) and len(sb_spec) > 1:
        state_barrier = _build_soft(
            {'barriers': list(sb_spec), 'cfg': backup_cfg}, dynamics, ctx)
    else:
        if isinstance(sb_spec, (list, tuple)):
            sb_spec = sb_spec[0]
        state_barrier = _build_barrier(sb_spec, dynamics, ctx)

    prev_state_barrier = ctx.state_barrier
    ctx.state_barrier = state_barrier
    try:
        backup_barriers = [_build_barrier(s, dynamics, ctx)
                           for s in spec['backup_barriers']]
    finally:
        ctx.state_barrier = prev_state_barrier

    return BackupBarrier(
        state_barrier=state_barrier,
        backup_barriers=backup_barriers,
        backup_policies=list(spec['backup_policies']),
        rel_deg=spec.get('rel_deg', 1),
        dynamics=dynamics,
        cfg=backup_cfg,
    )


@_barrier_type('lidar')
def _build_lidar(spec, dynamics, ctx):
    raise NotImplementedError(
        "barrier type 'lidar' is reserved: lidar barriers are built at runtime "
        "from sensor data, not from a static config. Construct the barrier "
        "instance yourself and pass it in place of the spec."
    )


def _build_barrier(spec, dynamics, ctx):
    if isinstance(spec, Barrier):
        return spec
    if isinstance(spec, Map):
        ctx.maps.append(spec)
        return spec.barrier
    if not isinstance(spec, dict):
        raise TypeError(
            f"Expected a barrier spec dict or Barrier instance, got {type(spec).__name__}"
        )
    spec = dict(spec)
    barrier_type = spec.pop('type', None)
    if barrier_type not in BARRIER_TYPES:
        raise _unknown('barrier type', barrier_type, BARRIER_TYPES)
    return BARRIER_TYPES[barrier_type](spec, dynamics, ctx)


def build_barrier(spec, dynamics=None) -> Barrier:
    """
    Build a barrier from a spec dict (see module docstring for spec types).

    Instances pass through unchanged. Use this to build a shared barrier once
    and reference the instance from multiple places in a config.
    """
    return _build_barrier(spec, dynamics, _BuildContext())


def _resolve_filter(filt_type):
    if filt_type is None:
        raise _requires('safety_filter', 'type')
    cls = FILTER_TYPES.get(filt_type)
    if cls is None:
        raise _unknown('safety_filter type', filt_type, FILTER_TYPES)
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


def _build_controller(spec, resolver, dynamics, extra=None):
    """Resolve a controller spec to an instance; instances pass through."""
    if not isinstance(spec, dict):
        return spec
    spec = dict(spec)
    cls = resolver(spec.pop('type', None))
    if dynamics is not None:
        spec.setdefault('dynamics', dynamics)
    for key, value in (extra or {}).items():
        spec.setdefault(key, value)
    return cls(**spec)


def from_config(cfg: dict) -> Parts:
    """
    Build the full pipeline from a config dict.

    Args:
        cfg: dict with 'dynamics', 'barrier', and exactly one of
             'safety_filter' or 'control' (callables allowed as values;
             instances pass through unchanged). 'control' builds a
             performance controller from CONTROL_TYPES. Layered designs use
             one config per layer (see module docstring).

    Returns:
        Parts with the built instances:
        (dynamics, map, maps, barrier, control, safety_filter).
    """
    if 'map' in cfg:
        raise ValueError(
            "the 'map' entry was replaced by the 'barrier' section: "
            "{'barrier': {'type': 'map', 'geoms': ..., 'composition': ..., 'cfg': ...}}"
        )
    unknown = set(cfg) - set(_TOP_LEVEL_KEYS)
    if unknown:
        raise ValueError(
            f"Unknown top-level cfg keys {sorted(unknown)}. "
            f"Valid keys: {sorted(_TOP_LEVEL_KEYS)}"
        )
    if 'safety_filter' not in cfg and 'control' not in cfg:
        raise ValueError("cfg must contain a 'safety_filter' or 'control' entry")
    if 'safety_filter' in cfg and 'control' in cfg:
        raise ValueError(
            "one controller per config: build the planner with a 'control' "
            "config first, then pass parts.control as 'desired_control' in "
            "the safety_filter config"
        )

    dynamics = _build_dynamics(cfg.get('dynamics'))

    ctx = _BuildContext()
    barrier = None
    if cfg.get('barrier') is not None:
        barrier = _build_barrier(cfg['barrier'], dynamics, ctx)

    control = None
    if cfg.get('control') is not None:
        control = _build_controller(cfg['control'], _resolve_control, dynamics)

    safety_filter = None
    if cfg.get('safety_filter') is not None:
        extra = {'barrier': barrier} if barrier is not None else None
        safety_filter = _build_controller(
            cfg['safety_filter'], _resolve_filter, dynamics, extra)

    return Parts(
        dynamics=dynamics,
        map=ctx.maps[0] if len(ctx.maps) == 1 else None,
        maps=tuple(ctx.maps),
        barrier=barrier,
        control=control,
        safety_filter=safety_filter,
    )
