import jax.numpy as jnp
import equinox as eqx
from immutabledict import immutabledict

from .base_dynamic import AffineInControlDynamics


class Unicycle5thOrderDynamics(AffineInControlDynamics):
    """
    Nonholonomic differential-drive ground robot, affine in control.

    State:   x = [q_x, q_y, gamma, s, omega]
    Control: u = [u_r, u_l]  (right / left motor voltages)

        f(x) = [ s cos(gamma) - l_d omega sin(gamma),
                 s sin(gamma) + l_d omega cos(gamma),
                 omega,
                 -c_1 s     - c_2 s^2     tanh(s / eps_1),
                 -c_3 omega - c_4 omega^2 tanh(omega / eps_2) ]

        g(x) = [0_{3x2}; M],   M = (k_m / (r R_a)) [[1/m, 1/m], [l/I, -l/I]]

    (q_x, q_y) is a point of interest offset l_d from the mass center, gamma is
    the velocity direction, s the speed and omega the angular velocity. The
    model carries back-EMF and ground-friction linear damping plus quadratic
    drag damping.

    The damping coefficients follow the source exactly:

        c_1 = 2 k_b k_m / (m r R_a)   + 2 eps_3 / (m r)
        c_3 =   k_b k_m l^2 / (I r^2 R_a) + l eps_3 / (I r^2)

    Note the differing powers of r between the two. Pass 'c_1' or 'c_3' in
    params to override either with your own value.

    Params (all optional, defaults as published):
        k_m, r, l, l_d, R_a, m, I, k_b, eps_3, c_2, c_4, eps_1, eps_2,
        and the overrides c_1, c_3.
    """

    _l_d: jnp.ndarray
    _c_1: jnp.ndarray
    _c_2: jnp.ndarray
    _c_3: jnp.ndarray
    _c_4: jnp.ndarray
    _eps_1: jnp.ndarray
    _eps_2: jnp.ndarray
    _M: jnp.ndarray

    def __init__(self, params=None, **kwargs):
        super().__init__(params, **kwargs)
        self._state_dim = 5
        self._action_dim = 2

        default_params = immutabledict({
            'k_m': 0.1,      # torque constant, N-m/Amp
            'r': 0.1,        # wheel radius, m
            'l': 0.5,        # distance between wheels, m
            'l_d': 0.25,     # mass center to point of interest, m
            'R_a': 0.27,     # armature resistance, ohms
            'm': 10.0,       # mass, kg
            'I': 0.83,       # moment of inertia, kg-m^2
            'k_b': 0.0487,   # back-EMF constant, V-s/rad
            'eps_3': 0.01,   # friction coefficient, N-m-s
            'c_2': 0.4581,   # drag damping on speed, 1/m
            'c_4': 0.3477,   # drag damping on angular velocity
            'eps_1': 0.4,
            'eps_2': 0.4,
        })
        p = immutabledict({**default_params, **(params or {})})
        self._params = p

        k_m, r, l = p['k_m'], p['r'], p['l']
        R_a, m, I = p['R_a'], p['m'], p['I']
        k_b, eps_3 = p['k_b'], p['eps_3']

        self._l_d = jnp.asarray(p['l_d'])
        self._c_1 = jnp.asarray(p.get(
            'c_1', 2.0 * k_b * k_m / (m * r * R_a) + 2.0 * eps_3 / (m * r)))
        self._c_2 = jnp.asarray(p['c_2'])
        self._c_3 = jnp.asarray(p.get(
            'c_3', k_b * k_m * l ** 2 / (I * r ** 2 * R_a)
            + l * eps_3 / (I * r ** 2)))
        self._c_4 = jnp.asarray(p['c_4'])
        self._eps_1 = jnp.asarray(p['eps_1'])
        self._eps_2 = jnp.asarray(p['eps_2'])

        self._M = (k_m / (r * R_a)) * jnp.array([[1.0 / m, 1.0 / m],
                                                 [l / I, -l / I]])

    def _f(self, x):
        """
        x: (5,) = [q_x, q_y, gamma, s, omega]
        output: (5,) - drift vector
        """
        gamma, s, omega = x[2], x[3], x[4]
        return jnp.array([
            s * jnp.cos(gamma) - self._l_d * omega * jnp.sin(gamma),
            s * jnp.sin(gamma) + self._l_d * omega * jnp.cos(gamma),
            omega,
            -self._c_1 * s - self._c_2 * s ** 2 * jnp.tanh(s / self._eps_1),
            -self._c_3 * omega
            - self._c_4 * omega ** 2 * jnp.tanh(omega / self._eps_2),
        ])

    def _g(self, x):
        """
        x: (5,) - single state vector
        output: (5, 2) - control matrix
        """
        return jnp.vstack([jnp.zeros((3, 2)), self._M])

    def get_pos(self, x):
        """Get position from state"""
        return x[0:2]

    def get_rot(self, x):
        """Get rotation from state"""
        return x[2]
