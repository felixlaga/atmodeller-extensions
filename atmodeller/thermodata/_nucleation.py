# SPDX-License-Identifier: Apache-2.0

"""Square-gradient (Cahn-Hilliard) interfaces and critical nuclei of a subregular solution.

Everything here is dimensionless and on the free-energy fit basis: free energies
are per mixing-basis mole over RT, and lengths are in units of sqrt(kappa / RT),
where kappa (J m²/mol) multiplies the squared composition gradient. With the
molar volume v of a mixing-basis mole, an energy I here becomes
(RT / v) (kappa / RT)^(3/2) I in joules. The barriers therefore scale exactly as
kappa^(3/2), so one table serves every gradient coefficient.

The critical nucleus is the radial saddle of the square-gradient functional
(Cahn & Hilliard 1959, J. Chem. Phys. 31, 688). It is found by shooting with
bisection on the central composition (overshoot/undershoot), as for a bounce
solution. The centre may lie exponentially close to the nucleus composition,
so the first part of the profile uses the linearised solution sinh(m r)/(m r).
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import xlogy

from atmodeller.jax_utils import FloatArray

_GAUSS_NODES, _GAUSS_WEIGHTS = (jnp.asarray(v) for v in np.polynomial.legendre.leggauss(160))
_LOG_DEPTH = 700.0
"""Largest log distance of the nucleus centre from the nucleus composition"""
_MAX_STEPS = 200000
"""Step cap per trajectory; reaching it marks the nucleus unresolved"""
_STEP = 0.01
"""RK4 step in units of the local curvature length"""
_BISECTIONS = 60


def _bisect(
    function: Callable[[FloatArray], FloatArray],
    lower: FloatArray,
    upper: FloatArray,
    iterations: int = 64,
) -> FloatArray:
    """Bisect an increasing, bracketed scalar function, with broadcasting."""

    def step(_, bounds):
        lo, hi = bounds
        mid = lo + (hi - lo) / 2
        negative = function(mid) < 0
        return jnp.where(negative, mid, lo), jnp.where(negative, hi, mid)

    lo, hi = jax.lax.fori_loop(0, iterations, step, (lower, upper))
    return lo + (hi - lo) / 2


def free_energy(a: FloatArray, b: FloatArray, z: FloatArray) -> FloatArray:
    """g_mix / RT per mixing-basis mole."""
    return xlogy(z, z) + xlogy(1 - z, 1 - z) + z * (1 - z) * (a * (1 - z) + b * z)


def slope(a: FloatArray, b: FloatArray, z: FloatArray) -> FloatArray:
    """d(g_mix / RT)/dz."""
    return jnp.log(z) - jnp.log1p(-z) + a + 2 * (b - 2 * a) * z + 3 * (a - b) * z**2


def curvature(a: FloatArray, b: FloatArray, z: FloatArray) -> FloatArray:
    """d²(g_mix / RT)/dz²."""
    return 1 / (z * (1 - z)) + 2 * b - 4 * a + 6 * (a - b) * z


def _logit_slope(a, b, q):
    z = jax.nn.sigmoid(q)
    return q + a + 2 * (b - 2 * a) * z + 3 * (a - b) * z**2


def _gauss(function, lower, upper):
    """Gauss-Legendre quadrature of a smooth function on [lower, upper]."""
    half = (upper - lower) / 2
    nodes = lower[..., None] + half[..., None] * (_GAUSS_NODES + 1)
    return half * jnp.sum(_GAUSS_WEIGHTS * function(nodes), axis=-1)


def _planar_interface(a, b, q_lower, q_upper):
    """Tension and 10-90 % width integrals of the flat coexistence interface.

    Args:
        a, b: Dimensionless interaction parameters, scalars.
        q_lower, q_upper: Logits of the coexisting compositions.

    Returns:
        (s, w): s = int sqrt(dg) dz, so the tension is 2 s in these units, and
        w = int_{z10}^{z90} dz / sqrt(dg), the 10-90 % width.
    """
    z_lower, z_upper = jax.nn.sigmoid(q_lower), jax.nn.sigmoid(q_upper)
    tangent = slope(a, b, z_lower)
    base = free_energy(a, b, z_lower)

    def excess(z):
        return jnp.maximum(free_energy(a, b, z) - base - tangent * (z - z_lower), 0.0)

    def integrand(q):
        z = jax.nn.sigmoid(q)
        return jnp.sqrt(excess(z)) * z * (1 - z)

    s = _gauss(integrand, q_lower, q_upper)
    span = z_upper - z_lower
    z10, z90 = z_lower + 0.1 * span, z_lower + 0.9 * span
    w = _gauss(lambda z: 1 / jnp.sqrt(excess(z)), z10, z90)
    return s, w


planar_interface = jnp.vectorize(_planar_interface)
"""Elementwise, broadcasting version of the planar interface integrals"""


def _metastable_geometry(a, b, z0, q_spinodal):
    """Nucleus composition, barrier top and escape point for parent z0.

    The nucleus composition has the parent's chemical potential difference on
    the other stable branch. The escape point lies between the barrier top
    and the nucleus, where the relative free energy returns to zero.
    """
    q_left, q_right = q_spinodal
    q0 = jnp.log(z0) - jnp.log1p(-z0)
    lower_branch = q0 < q_left
    target = _logit_slope(a, b, q0)

    def increasing(q):
        return _logit_slope(a, b, q) - target

    qn_upper = _bisect(increasing, q_right, jnp.full_like(q0, 700.0))
    qn_lower = _bisect(increasing, jnp.full_like(q0, -700.0), q_left)
    qn = jnp.where(lower_branch, qn_upper, qn_lower)
    # Inside the spinodal the chemical potential slope decreases.
    qt = _bisect(lambda q: -increasing(q), q_left, q_right)
    zn, zt = jax.nn.sigmoid(qn), jax.nn.sigmoid(qt)
    base, tangent = free_energy(a, b, z0), slope(a, b, z0)

    def relative(z):
        return free_energy(a, b, z) - base - tangent * (z - z0)

    sign = jnp.where(lower_branch, 1.0, -1.0)
    # Along the direction from the barrier top to the nucleus, relative drops.
    escape = _bisect(lambda s: -relative(zt + sign * s), 0.0 * z0, jnp.abs(zn - zt))
    ze = zt + sign * escape
    return zn, ze, relative(zn), lower_branch


def critical_nucleus(a: FloatArray, b: FloatArray, z0: FloatArray, q_spinodal) -> tuple:
    """Square-gradient critical nucleus of a metastable parent composition z0.

    Args:
        a, b: Dimensionless interaction parameters, scalars.
        z0: Parent composition on the mixing basis, between binodal and spinodal.
        q_spinodal: Logits of the two spinodal compositions.

    Returns:
        (barrier, centre, driving_force, valid): the saddle energy int(dg + z'^2) dV,
        from the virial form (2/3) int z'^2 dV; the composition at r = 0; the
        bulk driving force -dg at the nucleus composition, for classical theory.
        ``valid`` is false if the bracket or the virial check fails.
    """
    zn, ze, dg_nucleus, lower_branch = _metastable_geometry(a, b, z0, q_spinodal)
    sign = jnp.where(lower_branch, 1.0, -1.0)
    # Direction from the nucleus composition back toward the parent.
    toward = -sign
    span = jnp.abs(zn - ze)
    base, tangent = free_energy(a, b, z0), slope(a, b, z0)

    def relative(z):
        return free_energy(a, b, z) - base - tangent * (z - z0)

    def force(z):
        return (slope(a, b, z) - tangent) / 2

    m = jnp.sqrt(curvature(a, b, zn) / 2)
    m_parent = jnp.sqrt(jnp.maximum(curvature(a, b, z0), 0.0) / 2)
    linear_limit = 1e-6 * span

    def trajectory(depth):
        delta = span * jnp.exp(-depth)
        thin = delta < linear_limit
        # Linear region: u(r) = delta sinh(m r)/(m r), up to |u| = linear_limit.
        log_ratio = jnp.log(linear_limit) - jnp.log(delta)

        def excess_log(y):
            log_sinh = y + jnp.log1p(-jnp.exp(-2 * y)) - jnp.log(2.0)
            return log_sinh - jnp.log(y) - log_ratio

        y = _bisect(excess_log, jnp.full_like(delta, 1e-8), jnp.full_like(delta, _LOG_DEPTH + 50))
        # u'(r) = delta m [cosh y / y - sinh y / y²], evaluated in logs.
        log_cosh = y + jnp.log1p(jnp.exp(-2 * y)) - jnp.log(2.0)
        du_lin = jnp.exp(jnp.log(delta) + log_cosh - jnp.log(y)) * m - linear_limit * m / y
        r_lin, u_lin = y / m, linear_limit
        # Thick nuclei start from the series z = centre + c r²/6 at a tiny radius.
        centre = zn + toward * delta
        r_series = 1e-4 / m
        c = force(centre)
        r0 = jnp.where(thin, r_lin, r_series)
        z_start = jnp.where(thin, zn + toward * u_lin, centre + c * r_series**2 / 6)
        v_start = jnp.where(thin, toward * du_lin, c * r_series / 3)
        # Bulk free energy of the uniform interior, int_0^r0 r² dg dr.
        interior = jnp.where(thin, dg_nucleus * r_lin**3 / 3, 0.0)

        def rhs(r, z, v):
            return v, force(z) - 2 * v / r

        # Step lengths follow the local curvature, floored by the parent's, so a
        # nearly pure nucleus phase (huge m) cannot exhaust the integration.
        floor = (0.1 * m_parent) ** 2 + (1e-6 * m) ** 2
        r_stop = r0 + 60 / jnp.maximum(m_parent, 1e-3 * m)

        def running(carry):
            n, r, _, _, _, _, over, under = carry
            return ~(over | under) & (r < r_stop) & (n < _MAX_STEPS)

        def step(carry):
            n, r, z, v, grad, pot, over, under = carry
            scale = jnp.sqrt(jnp.abs(curvature(a, b, z)) / 2 + floor)
            h = jnp.minimum(_STEP / scale, 0.25 * r + _STEP / m)
            k1 = rhs(r, z, v)
            k2 = rhs(r + h / 2, z + h / 2 * k1[0], v + h / 2 * k1[1])
            k3 = rhs(r + h / 2, z + h / 2 * k2[0], v + h / 2 * k2[1])
            k4 = rhs(r + h, z + h * k3[0], v + h * k3[1])
            z_new = z + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
            v_new = v + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
            r_mid = r + h / 2
            z_mid = (z + z_new) / 2
            v_mid = (v + v_new) / 2
            crossed = toward * (z_new - z0) > 0
            turned = toward * v_new < 0
            return (
                n + 1,
                r + h,
                z_new,
                v_new,
                grad + h * r_mid**2 * v_mid**2,
                pot + h * r_mid**2 * relative(z_mid),
                crossed,
                ~crossed & turned,
            )

        zero = 0.0 * r0
        start = (0, r0, z_start, v_start, zero, interior, r0 < 0, r0 < 0)
        n, _, _, _, grad, pot, over, under = jax.lax.while_loop(running, step, start)
        # Stopping at r_stop without an event means the profile has settled on the
        # parent within exp(-60); hitting the step cap means it is unresolved.
        resolved = n < _MAX_STEPS
        return over, under, grad, pot, centre, resolved

    def overshoots(depth):
        over = trajectory(depth)[0]
        return jnp.where(over, 1.0, -1.0)

    lowest, highest = jnp.zeros_like(z0), jnp.full_like(z0, _LOG_DEPTH)
    bracketed = (overshoots(lowest) < 0) & (overshoots(highest) > 0)
    depth = _bisect(overshoots, lowest, highest, _BISECTIONS)
    _, _, grad, pot, centre, resolved = trajectory(depth)
    barrier = 4 * jnp.pi * (2 / 3) * grad
    direct = 4 * jnp.pi * (grad + pot)
    consistent = jnp.abs(direct / barrier - 1) < 0.02
    valid = bracketed & consistent & resolved & (barrier > 0)
    return barrier, centre, -dg_nucleus, valid
