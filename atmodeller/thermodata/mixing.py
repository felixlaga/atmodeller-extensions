# SPDX-License-Identifier: Apache-2.0

"""Binary mixing thermodynamics, spinodals, and equilibrium common tangents.

All energies are per mole of *components*, temperatures are K, pressures are bar,
and ``x`` is the mole fraction of component 1 in the ordered pair (0, 1).
Energies of mixing and separation come from the same free energy. Nucleation
barriers are thermodynamic: they include no rate prefactor, transport or growth.
"""

from collections.abc import Callable
import math
from typing import Self

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.scipy.special import xlogy
from jaxtyping import Array, ArrayLike, Bool

from atmodeller.jax_utils import FloatArray, as_j64
from atmodeller.sci_utils import GAS_CONSTANT
from atmodeller.thermodata import _nucleation
from atmodeller.thermodata._nucleation import _bisect

AVOGADRO = 6.02214076e23
"""Avogadro constant (1/mol)"""


def _derivative(function: Callable[[FloatArray], FloatArray], value: ArrayLike) -> FloatArray:
    """Elementwise forward derivative that keeps NaN where the function is NaN.

    The derivative of a NaN branch of ``jnp.where`` is zero, so an invalid
    input would otherwise give a finite derivative.
    """
    value = as_j64(value)
    primal, tangent = jax.jvp(function, (value,), (jnp.ones_like(value),))
    return jnp.where(jnp.isnan(primal), jnp.nan, tangent)


def _curvature(a: FloatArray, b: FloatArray, x: FloatArray) -> FloatArray:
    """Second composition derivative of g_mix / RT."""
    return 1 / (x * (1 - x)) + 2 * b - 4 * a + 6 * (a - b) * x


def _third_derivative(a: FloatArray, b: FloatArray, x: FloatArray) -> FloatArray:
    return -1 / x**2 + 1 / (1 - x) ** 2 + 6 * (a - b)


def _spinodal_limits_impl(a: FloatArray, b: FloatArray) -> FloatArray:
    # The fourth derivative is positive, so the curvature has exactly one minimum.
    # Bounded composition roots here are checked below; the binodal uses log odds
    # to retain accuracy for exponentially dilute coexisting components.
    lower = jnp.full_like(a, 1e-14)
    upper = jnp.full_like(a, 1 - 1e-14)
    minimum = _bisect(lambda x: _third_derivative(a, b, x), lower, upper)
    has_gap = _curvature(a, b, minimum) < 0
    left = _bisect(lambda x: -_curvature(a, b, x), lower, minimum)
    right = _bisect(lambda x: _curvature(a, b, x), minimum, upper)
    return jnp.where(has_gap[..., None], jnp.stack((left, right), axis=-1), 0.5)


_spinodal_limits = jax.custom_jvp(_spinodal_limits_impl)


@_spinodal_limits.defjvp
def _spinodal_limits_jvp(primals, tangents):
    a, b = primals
    da, db = tangents
    x = _spinodal_limits(a, b)
    gap = x[..., 0] < x[..., 1]
    denominator = jnp.where(gap[..., None], _third_derivative(a[..., None], b[..., None], x), 1.0)
    dx = -((-4 + 6 * x) * da[..., None] + (2 - 6 * x) * db[..., None]) / denominator
    return x, jnp.where(gap[..., None], dx, 0.0)


def _spinodal(a: FloatArray, b: FloatArray) -> tuple[FloatArray, FloatArray, Bool[Array, "..."]]:
    limits = _spinodal_limits(a, b)
    left, right = limits[..., 0], limits[..., 1]
    return left, right, left < right


def _excess_potentials(a: FloatArray, b: FloatArray, x: FloatArray) -> FloatArray:
    """Partial molar excess Gibbs energies divided by RT, in component order."""
    mu0 = x**2 * (2 * a - b + 2 * (b - a) * x)
    mu1 = (1 - x) ** 2 * (a + 2 * (b - a) * x)
    return jnp.stack((mu0, mu1), axis=-1)


def _potentials_from_logit(a: FloatArray, b: FloatArray, q: FloatArray) -> FloatArray:
    x, y = jax.nn.sigmoid(q), jax.nn.sigmoid(-q)
    # Evaluate both complementary fractions and their logs independently. Near a
    # pure endpoint, 1 - sigmoid(q) can round to zero even when exp(-q) is resolved.
    mu0 = -jax.nn.softplus(q) + x**2 * (2 * a - b + 2 * (b - a) * x)
    mu1 = -jax.nn.softplus(-q) + y**2 * (a + 2 * (b - a) * x)
    return jnp.stack((mu0, mu1), axis=-1)


def _binodal_logits_impl(a: FloatArray, b: FloatArray) -> FloatArray:
    """Common tangent on the two stable branches; no equal-composition root."""
    sl, sr, has_gap = _spinodal(a, b)
    # Use a well-conditioned dummy problem where the true mixture has no gap.
    a_work, b_work = jnp.where(has_gap, a, 3.0), jnp.where(has_gap, b, 3.0)
    sl = jnp.where(has_gap, sl, (1 - jnp.sqrt(1 / 3)) / 2)
    sr = jnp.where(has_gap, sr, (1 + jnp.sqrt(1 / 3)) / 2)
    ql, qr = jnp.log(sl) - jnp.log1p(-sl), jnp.log(sr) - jnp.log1p(-sr)
    qmin, qmax = jnp.full_like(a, -700.0), jnp.full_like(a, 700.0)

    def slope(q):
        x = jax.nn.sigmoid(q)
        return q + a_work + 2 * (b_work - 2 * a_work) * x + 3 * (a_work - b_work) * x**2

    def endpoints(m):
        left = _bisect(lambda q: slope(q) - m, qmin, ql)
        right = _bisect(lambda q: slope(q) - m, qr, qmax)
        return left, right

    def tangent_difference(m):
        left, right = endpoints(m)
        # mu0 = g/RT - x*m is the tangent intercept. Its right-minus-left
        # difference decreases strictly with m, hence this sign for bisection.
        return (
            _potentials_from_logit(a_work, b_work, left)[..., 0]
            - _potentials_from_logit(a_work, b_work, right)[..., 0]
        )

    m = _bisect(tangent_difference, slope(qr), slope(ql))
    left, right = endpoints(m)
    return jnp.where(has_gap[..., None], jnp.stack((left, right), axis=-1), 0.0)


# Construct the JAX callable after definition so the test suite's import-time
# type instrumentation wraps the function, rather than the custom_jvp object.
_binodal_logits = jax.custom_jvp(_binodal_logits_impl)


@_binodal_logits.defjvp
def _binodal_logits_jvp(primals, tangents):
    """Implicit derivative of equality of both component chemical potentials.

    Differentiating the comparisons in bisection would give incorrect derivatives.
    The common-tangent Jacobian supplies derivatives with respect to both model
    parameters, and thus also temperature and pressure. At a critical point the
    two-phase derivative is singular; on the no-gap branch the sentinel is constant.
    """
    a, b = primals
    da, db = tangents
    q = _binodal_logits(a, b)
    x, y = jax.nn.sigmoid(q), jax.nn.sigmoid(-q)
    gap = q[..., 0] < q[..., 1]
    c = 1 + x * y * (2 * b[..., None] - 4 * a[..., None] + 6 * (a - b)[..., None] * x)
    j00, j01 = -x[..., 0] * c[..., 0], x[..., 1] * c[..., 1]
    j10, j11 = y[..., 0] * c[..., 0], -y[..., 1] * c[..., 1]
    dmu0 = x**2 * (2 * y * da[..., None] + (2 * x - 1) * db[..., None])
    dmu1 = y**2 * ((1 - 2 * x) * da[..., None] + 2 * x * db[..., None])
    d0, d1 = dmu0[..., 0] - dmu0[..., 1], dmu1[..., 0] - dmu1[..., 1]
    determinant = jnp.where(gap, j00 * j11 - j01 * j10, 1.0)
    dl = (-d0 * j11 + j01 * d1) / determinant
    dr = (-j00 * d1 + j10 * d0) / determinant
    dq = jnp.where(gap[..., None], jnp.stack((dl, dr), axis=-1), 0.0)
    return q, dq


class BinaryCoexistence(eqx.Module):
    """Equilibrium coexistence at specified temperature and pressure.

    ``logits`` stores the two compositions as log(x/(1-x)), preserving dilute
    limits even when a mole fraction rounds to unity. Where there is no gap the
    logits are zero, both compositions are 0.5, and ``has_gap`` is false. These
    are sentinels, not coexistence endpoints. Always inspect ``has_gap`` and
    ``valid``. Invalid numerical results must not be used as an equilibrium.
    """

    logits: FloatArray
    has_gap: Bool[Array, "..."]
    valid: Bool[Array, "..."]
    log_activity: FloatArray
    """Common component activities, natural log, shape (..., 2); NaN without a gap"""
    residual: FloatArray
    """Maximum chemical potential mismatch divided by RT (zero without a gap)"""

    @property
    def lower(self) -> FloatArray:
        """Component-1 mole fraction of the component-0-rich phase."""
        return jax.nn.sigmoid(self.logits[..., 0])

    @property
    def upper(self) -> FloatArray:
        """Component-1 mole fraction of the component-1-rich phase."""
        return jax.nn.sigmoid(self.logits[..., 1])

    def contains(self, x: ArrayLike) -> Bool[Array, "..."]:
        """Whether the bulk composition lies strictly inside the coexistence gap."""
        x = as_j64(x)
        return self.has_gap & self.valid & (x > self.lower) & (x < self.upper)

    def upper_fraction(self, x: ArrayLike) -> FloatArray:
        """Mole fraction of the upper-composition phase from the lever rule.

        Returns zero outside the gap, where a second phase is not present, and
        NaN for an invalid calculation. This is a *phase amount*, not a component
        abundance or a mass fraction.
        """
        x = as_j64(x)
        width = jnp.where(self.has_gap, self.upper - self.lower, 1.0)
        fraction = jnp.where(self.contains(x), (x - self.lower) / width, 0.0)
        return jnp.where(self.valid & (x >= 0) & (x <= 1), fraction, jnp.nan)


class NucleationBarrier(eqx.Module):
    """Critical-nucleus energy of a homogeneous parent composition.

    ``energy`` is the square-gradient (Cahn-Hilliard) saddle; ``classical_energy``
    is classical nucleation theory with the same planar tension. Both are
    infinite for a stable parent outside the gap and zero inside the spinodal.
    NaN marks an invalid input (including a nonpositive gradient coefficient
    or volume) or, for ``energy`` only, a failed nucleus solution. ``valid``
    refers to ``energy``; the classical value is usable wherever it is finite.
    """

    energy: FloatArray
    """Square-gradient nucleation barrier (J)"""
    classical_energy: FloatArray
    """Classical barrier 16 pi sigma³ / (3 dg_v²) with the planar tension (J)"""
    nucleus_composition: FloatArray
    """Native component-1 mole fraction at the nucleus centre (NaN if no nucleus)"""
    energy_over_kT: FloatArray
    """``energy`` over Boltzmann's constant times temperature"""
    valid: Bool[Array, "..."]


class SubregularSolution(eqx.Module):
    r"""A thermodynamically consistent binary subregular solution.

    .. math::
        g_{\rm mix} = RT[(1-x)\ln(1-x)+x\ln x]
          + x(1-x)[A(1-x)+Bx](1-T/C+P/D).

    ``x`` is component 1's mole fraction. Pure-component standard Gibbs energies
    are supplied separately by Atmodeller's species and must use compatible
    component and standard-state conventions. Linear standard-state terms do
    not change a common tangent's compositions or the spinodal.

    Args:
        interaction_0: A (J/mol of components).
        interaction_1: B (J/mol of components).
        temperature_scale: C (K); infinity disables temperature scaling.
        pressure_scale: D (bar); infinity disables pressure scaling. May be negative.
        component_0_multiplicity: Number of fit-basis units in one native
            component-0 formula unit; component 1 contributes one. With k != 1,
            the equation above is evaluated at z=x/[k+(1-k)x] and multiplied by
            k+(1-k)x, preserving the extensive free energy. Defaults to one.

    Inputs broadcast elementwise. Invalid T <= 0, P < 0, nonfinite T/P or x
    outside [0,1] produce NaN. No extrapolation validity is implied by a finite
    answer. A binary parameter set is not transferable to a different mixture.
    """

    interaction_0: FloatArray
    interaction_1: FloatArray
    temperature_scale: FloatArray
    pressure_scale: FloatArray
    component_0_multiplicity: float = eqx.field(static=True)
    """Mixing-basis units per native component-0 formula unit (component 1: one)."""

    def __init__(
        self,
        interaction_0: ArrayLike,
        interaction_1: ArrayLike,
        *,
        temperature_scale: ArrayLike = jnp.inf,
        pressure_scale: ArrayLike = jnp.inf,
        component_0_multiplicity: float = 1.0,
    ):
        if not math.isfinite(component_0_multiplicity) or component_0_multiplicity <= 0:
            raise ValueError("Component multiplicity must be finite and positive.")
        self.interaction_0 = as_j64(interaction_0)
        self.interaction_1 = as_j64(interaction_1)
        self.temperature_scale = as_j64(temperature_scale)
        self.pressure_scale = as_j64(pressure_scale)
        self.component_0_multiplicity = float(component_0_multiplicity)

    @classmethod
    def regular_solution(cls, interaction: ArrayLike) -> Self:
        """Symmetric regular solution with g_excess = interaction*x*(1-x), J/mol."""
        return cls(interaction, interaction)

    @classmethod
    def hydrogen_silicate_rogers25(cls) -> Self:
        """H2-MgSiO3 parameterization of Rogers et al. (2025), equation 7.

        Component 0 is MgSiO3 and component 1 is H2 (formula-unit mole fractions).
        This reproduces the published phenomenological mixing model, not a new
        calibration to the final 2026 DFT paper. The published D=-35 GPa is
        converted to -350000 bar. See the spinodal guide for provenance and scope.
        """
        return cls(-6.26e3, 7.86e5, temperature_scale=4.67e3, pressure_scale=-3.5e5)

    @classmethod
    def hydrogen_silicate_gilmore26(cls) -> Self:
        """Final Gilmore & Stixrude (2026) fit, on native (MgSiO3, H2) components.

        DOI: 10.1038/s41586-025-09970-4, equations 1 and 8. The paper uses
        z = n(H2)/(n(H2) + n(MgO) + n(SiO2)), so one MgSiO3 formula unit
        contributes TWO mixing-basis units. This conversion applies to the
        extensive Gibbs energy and both chemical potentials, not only to z.
        All public compositions, phase amounts and energies remain on the
        native formula-unit basis. This does not calibrate pure standard states
        for exchange with a separate reacting gas reservoir.
        """
        return cls(
            75e3,
            424e3,
            temperature_scale=4400.0,
            pressure_scale=-3.5e5,
            component_0_multiplicity=2.0,
        )

    def mixing_composition(self, x: ArrayLike) -> FloatArray:
        """Convert native component-1 mole fraction to the free-energy fit basis."""
        x = as_j64(x)
        k = self.component_0_multiplicity
        return jnp.where((x >= 0) & (x <= 1), x / (k + (1 - k) * x), jnp.nan)

    def native_composition(self, x: ArrayLike) -> FloatArray:
        """Convert component-1 mole fraction in the fit basis to the native basis."""
        x = as_j64(x)
        k = self.component_0_multiplicity
        return jnp.where((x >= 0) & (x <= 1), k * x / (1 + (k - 1) * x), jnp.nan)

    def _parameters(
        self, temperature: ArrayLike, pressure: ArrayLike
    ) -> tuple[FloatArray, FloatArray]:
        t, p = as_j64(temperature), as_j64(pressure)
        factor = (1 - t / self.temperature_scale + p / self.pressure_scale) / (GAS_CONSTANT * t)
        valid = (t > 0) & (p >= 0) & jnp.isfinite(t) & jnp.isfinite(p)
        a, b = jnp.broadcast_arrays(self.interaction_0 * factor, self.interaction_1 * factor)
        return jnp.where(valid, a, jnp.nan), jnp.where(valid, b, jnp.nan)

    def gibbs_energy(
        self, temperature: ArrayLike, pressure: ArrayLike, x: ArrayLike
    ) -> FloatArray:
        """Homogeneous molar Gibbs energy of mixing (J/mol); finite at x=0 and 1."""
        a, b = self._parameters(temperature, pressure)
        x = as_j64(x)
        z = self.mixing_composition(x)
        scale = self.component_0_multiplicity + (1 - self.component_0_multiplicity) * x
        f = xlogy(z, z) + xlogy(1 - z, 1 - z) + z * (1 - z) * (a * (1 - z) + b * z)
        return jnp.where((x >= 0) & (x <= 1), scale * GAS_CONSTANT * temperature * f, jnp.nan)

    def curvature(self, temperature: ArrayLike, pressure: ArrayLike, x: ArrayLike) -> FloatArray:
        """Homogeneous d²g_mix/dx² (J/mol); negative means spinodal instability."""
        a, b = self._parameters(temperature, pressure)
        x = as_j64(x)
        k = self.component_0_multiplicity
        scale = k + (1 - k) * x
        value = (
            GAS_CONSTANT
            * temperature
            * k**2
            / scale**3
            * _curvature(a, b, self.mixing_composition(x))
        )
        return jnp.where((x >= 0) & (x <= 1), value, jnp.nan)

    def spinodal(self, temperature: ArrayLike, pressure: ArrayLike) -> FloatArray:
        """Lower and upper spinodal compositions, shape (..., 2), or NaN if absent.

        Implicit derivatives are defined away from a critical point. These are
        local-stability limits, not the coexisting compositions (binodal).
        """
        a, b = self._parameters(temperature, pressure)
        lower, upper, gap = _spinodal(a, b)
        limits = self.native_composition(jnp.stack((lower, upper), axis=-1))
        return jnp.where(gap[..., None], limits, jnp.nan)

    def spinodal_temperature(self, pressure: ArrayLike, x: ArrayLike) -> FloatArray:
        """Positive temperature at which the homogeneous curvature vanishes.

        Analytical inversion of this model's T-linear Gibbs energy. NaN means
        no finite positive root, invalid P/x, or a composition endpoint. This
        locates a local-stability boundary; it predicts neither a nucleation
        temperature nor a decomposition rate. A positive temperature derivative
        of curvature is additionally needed to interpret it as a cooling onset.
        """
        p, z = as_j64(pressure), self.mixing_composition(x)
        a, b = self.interaction_0, self.interaction_1
        second = 2 * b - 4 * a + 6 * (a - b) * z
        denominator = GAS_CONSTANT / (z * (1 - z)) - second / self.temperature_scale
        temperature = -second * (1 + p / self.pressure_scale) / denominator
        valid = (
            jnp.isfinite(p)
            & (p >= 0)
            & (z > 0)
            & (z < 1)
            & jnp.isfinite(temperature)
            & (temperature > 0)
            & (denominator != 0)
        )
        return jnp.where(valid, temperature, jnp.nan)

    def binodal_temperature(
        self,
        pressure: ArrayLike,
        x: ArrayLike,
        *,
        lowest: float = 100.0,
        highest: float = 1e5,
    ) -> FloatArray:
        """Highest temperature at which composition x lies inside the coexistence gap.

        The equilibrium separation onset of a cooling parcel. A scan of 96
        log-spaced temperatures between ``lowest`` and ``highest`` (K) finds the
        highest one inside the gap whose neighbour above is outside it, and
        bisection refines that bracket. A gap that closes and reopens within one
        grid step (7.5 %) is missed. NaN means no onset in range, a gap that
        persists at ``highest``, an invalid tangent, or invalid P/x.
        """
        p, x = jnp.broadcast_arrays(as_j64(pressure), as_j64(x))
        grid = jnp.geomspace(lowest, highest, 96)
        c = self.coexistence(grid, p[..., None])
        inside = c.contains(x[..., None])
        crossing = inside[..., :-1] & c.valid[..., 1:] & ~inside[..., 1:]
        # Index of the highest crossing, counted from the low end.
        index = crossing.shape[-1] - 1 - jnp.argmax(crossing[..., ::-1], axis=-1)
        grid = jnp.broadcast_to(grid, inside.shape)
        lower = jnp.take_along_axis(grid, index[..., None], axis=-1)[..., 0]
        upper = jnp.take_along_axis(grid, index[..., None] + 1, axis=-1)[..., 0]

        def contains(t):
            return self.coexistence(t, p).contains(x)

        temperature = _bisect(lambda t: jnp.where(contains(t), -1.0, 1.0), lower, upper)
        valid = jnp.any(crossing, axis=-1) & ~inside[..., -1]
        valid &= self.coexistence(temperature, p).valid & jnp.isfinite(p) & (p >= 0)
        return jnp.where(valid, temperature, jnp.nan)

    def log_activity_coefficients(
        self, temperature: ArrayLike, pressure: ArrayLike, x: ArrayLike
    ) -> FloatArray:
        """Homogeneous ln(gamma_0, gamma_1), derived from the same Gibbs energy."""
        a, b = self._parameters(temperature, pressure)
        x = as_j64(x)
        z = self.mixing_composition(x)
        coefficients = _excess_potentials(a, b, z)
        k = self.component_0_multiplicity
        if k != 1:
            # Derivatives of G = (k*n0+n1)*g(z), relative to native ideal
            # activities. Transforming only the binodal composition is wrong.
            log_scale = jnp.log(k + (1 - k) * x)
            mu0 = k * coefficients[..., 0] + (k - 1) * jnp.log1p(-x)
            mu0 += k * (jnp.log(k) - log_scale)
            mu1 = coefficients[..., 1] - log_scale
            coefficients = jnp.stack((mu0, mu1), axis=-1)
        return jnp.where(((x >= 0) & (x <= 1))[..., None], coefficients, jnp.nan)

    def coexistence(self, temperature: ArrayLike, pressure: ArrayLike) -> BinaryCoexistence:
        """Compute the binodal by equality of both component chemical potentials.

        Uses bracketed solves on the two stable branches, avoiding the trivial
        equal-composition root. The calculation is JIT/vmap compatible and has
        implicit derivatives with respect to its parameters, T and P. A mismatch
        exceeding 1e-8 in mu/RT, a lost bracket, or the log-odds bound +/-700
        marks the result invalid. It must then not be used as a phase equilibrium.
        """
        a, b = self._parameters(temperature, pressure)
        q = _binodal_logits(a, b)
        sl, sr, gap = _spinodal(a, b)
        left = _potentials_from_logit(a, b, q[..., 0])
        right = _potentials_from_logit(a, b, q[..., 1])
        multiplicity = jnp.array([self.component_0_multiplicity, 1.0])
        left, right = left * multiplicity, right * multiplicity
        residual = jnp.where(gap, jnp.max(jnp.abs(left - right), axis=-1), 0.0)
        brackets = (jax.nn.sigmoid(q[..., 0]) < sl) & (jax.nn.sigmoid(q[..., 1]) > sr)
        resolved = (residual < 1e-8) & brackets & jnp.all(jnp.abs(q) < 699.0, axis=-1)
        edges_resolved = (
            (_curvature(a, b, as_j64(1e-14)) > 0)
            & (_curvature(a, b, as_j64(1 - 1e-14)) > 0)
            & (_third_derivative(a, b, as_j64(1e-14)) < 0)
            & (_third_derivative(a, b, as_j64(1 - 1e-14)) > 0)
        )
        valid = jnp.isfinite(a) & jnp.isfinite(b) & edges_resolved & (~gap | resolved)
        activity = jnp.where((gap & valid)[..., None], (left + right) / 2, jnp.nan)
        native_q = jnp.where(gap[..., None], q + jnp.log(self.component_0_multiplicity), 0.0)
        return BinaryCoexistence(native_q, gap, valid, activity, residual)

    def homogeneous_log_activities(
        self, temperature: ArrayLike, pressure: ArrayLike, log_mole_fraction: ArrayLike
    ) -> FloatArray:
        """Homogeneous activities from native log fractions, preserving trace limits.

        Used by the reaction engine: rounding exp(log_x1) to one must not erase
        a finite log_x0. Both ideal and nonideal terms use the same component basis.
        """
        log_x = as_j64(log_mole_fraction)
        k = self.component_0_multiplicity
        log_scale = jnp.logaddexp(jnp.log(k) + log_x[..., 0], log_x[..., 1])
        log_z = log_x + jnp.array([jnp.log(k), 0.0]) - log_scale[..., None]
        a, b = self._parameters(temperature, pressure)
        return (log_z + _excess_potentials(a, b, jnp.exp(log_z[..., 1]))) * jnp.array([k, 1.0])

    def log_activities(
        self,
        temperature: ArrayLike,
        pressure: ArrayLike,
        x: ArrayLike,
        *,
        equilibrate: bool = True,
    ) -> FloatArray:
        """Log component activities, with equilibrium demixing enabled by default.

        Set ``equilibrate=False`` only to examine a constrained homogeneous state,
        which can be metastable or unstable. Common-tangent activities are constant
        across the coexistence gap. No extra solubility law is applied.
        """
        x = as_j64(x)
        ideal = jnp.stack((jnp.log1p(-x), jnp.log(x)), axis=-1)
        homogeneous = self.homogeneous_log_activities(temperature, pressure, ideal)
        homogeneous = jnp.where(((x >= 0) & (x <= 1))[..., None], homogeneous, jnp.nan)
        if not equilibrate:
            return homogeneous
        coexistence = self.coexistence(temperature, pressure)
        values = jnp.where(
            coexistence.contains(x)[..., None], coexistence.log_activity, homogeneous
        )
        return jnp.where(coexistence.valid[..., None], values, jnp.nan)

    def equilibrium_gibbs_energy(
        self, temperature: ArrayLike, pressure: ArrayLike, x: ArrayLike
    ) -> FloatArray:
        """Convex-envelope mixing energy (J/native mol), including pure endpoints."""
        x = as_j64(x)
        c = self.coexistence(temperature, pressure)
        activity = jnp.where((c.has_gap & c.valid)[..., None], c.log_activity, 0.0)
        tangent = GAS_CONSTANT * temperature * ((1 - x) * activity[..., 0] + x * activity[..., 1])
        result = jnp.where(c.contains(x), tangent, self.gibbs_energy(temperature, pressure, x))
        return jnp.where(c.valid, result, jnp.nan)

    def entropy(self, temperature: ArrayLike, pressure: ArrayLike, x: ArrayLike) -> FloatArray:
        """Homogeneous molar entropy of mixing, -dg_mix/dT (J/K/native mol)."""
        return -_derivative(lambda v: self.gibbs_energy(v, pressure, x), temperature)

    def enthalpy(self, temperature: ArrayLike, pressure: ArrayLike, x: ArrayLike) -> FloatArray:
        """Homogeneous molar enthalpy of mixing, g + Ts (J/native mol).

        Derived from :meth:`gibbs_energy` by automatic differentiation, so it
        stays consistent with the binodal and the activities. Positive values
        mean mixing absorbs heat, so separation releases it.
        """
        t = as_j64(temperature)
        return self.gibbs_energy(t, pressure, x) + t * self.entropy(t, pressure, x)

    def volume(self, temperature: ArrayLike, pressure: ArrayLike, x: ArrayLike) -> FloatArray:
        """Homogeneous molar volume of mixing, dg_mix/dP (m³/native mol)."""
        # J/bar to m³: one bar is 1e5 Pa.
        return _derivative(lambda v: self.gibbs_energy(temperature, v, x), pressure) * 1e-5

    def heat_capacity(
        self, temperature: ArrayLike, pressure: ArrayLike, x: ArrayLike
    ) -> FloatArray:
        """Homogeneous heat capacity of mixing, dh_mix/dT (J/K/native mol).

        Zero for this model, whose excess Gibbs energy is linear in temperature.
        It excludes the pure-component heat capacities.
        """
        return _derivative(lambda v: self.enthalpy(v, pressure, x), temperature)

    def partial_enthalpies(
        self, temperature: ArrayLike, pressure: ArrayLike, x: ArrayLike
    ) -> FloatArray:
        """Homogeneous partial molar enthalpies of mixing, shape (..., 2), J/native mol.

        These are -RT² d(ln gamma_i)/dT, the heat each native component carries
        when it leaves this composition. Their mole-fraction average is
        :meth:`enthalpy`.
        """
        t = as_j64(temperature)
        dlng_dt = _derivative(lambda v: self.log_activity_coefficients(v, pressure, x), t)
        return -GAS_CONSTANT * t[..., None] ** 2 * dlng_dt

    def _lever_rule(
        self,
        prop: Callable[[ArrayLike, ArrayLike, ArrayLike], FloatArray],
        temperature: ArrayLike,
        pressure: ArrayLike,
        x: ArrayLike,
        coexistence: BinaryCoexistence | None = None,
    ) -> FloatArray:
        """Equilibrium value of an extensive property: lever-rule average in the gap."""
        x = as_j64(x)
        c = self.coexistence(temperature, pressure) if coexistence is None else coexistence
        fraction = c.upper_fraction(x)
        split = (1 - fraction) * prop(temperature, pressure, c.lower)
        split += fraction * prop(temperature, pressure, c.upper)
        result = jnp.where(c.contains(x), split, prop(temperature, pressure, x))
        return jnp.where(c.valid, result, jnp.nan)

    def equilibrium_enthalpy(
        self, temperature: ArrayLike, pressure: ArrayLike, x: ArrayLike
    ) -> FloatArray:
        """Mixing enthalpy after equilibrium demixing (J/native mol); NaN if invalid."""
        return self._lever_rule(self.enthalpy, temperature, pressure, x)

    def equilibrium_entropy(
        self, temperature: ArrayLike, pressure: ArrayLike, x: ArrayLike
    ) -> FloatArray:
        """Mixing entropy after equilibrium demixing (J/K/native mol); NaN if invalid."""
        return self._lever_rule(self.entropy, temperature, pressure, x)

    def equilibrium_volume(
        self, temperature: ArrayLike, pressure: ArrayLike, x: ArrayLike
    ) -> FloatArray:
        """Mixing volume after equilibrium demixing (m³/native mol); NaN if invalid."""
        return self._lever_rule(self.volume, temperature, pressure, x)

    def equilibrium_heat_capacity(
        self, temperature: ArrayLike, pressure: ArrayLike, x: ArrayLike
    ) -> FloatArray:
        """Apparent mixing heat capacity of a closed mixture kept in equilibrium.

        dh/dT at fixed bulk composition (J/K/native mol). Inside the gap it
        includes the heat released as the coexisting compositions move, so it
        acts as a latent heat spread over the cooling range. Outside the gap it
        equals :meth:`heat_capacity`. It is singular at a critical point.
        """
        return _derivative(lambda v: self.equilibrium_enthalpy(v, pressure, x), temperature)

    def demixing_heat(
        self, temperature: ArrayLike, pressure: ArrayLike, x: ArrayLike
    ) -> FloatArray:
        """Heat released when a homogeneous mixture separates at fixed T and P.

        Enthalpy of the homogeneous state minus that of the equilibrium state
        (J/native mol). Zero outside the gap, continuous at the binodal, and
        NaN for an invalid common tangent. It does not say whether or when the
        separation happens; that needs a kinetic or stability criterion.
        """
        c = self.coexistence(temperature, pressure)
        equilibrium = self._lever_rule(self.enthalpy, temperature, pressure, x, c)
        return self.enthalpy(temperature, pressure, x) - equilibrium

    def exsolution_enthalpy(self, temperature: ArrayLike, pressure: ArrayLike) -> FloatArray:
        """Heat released per native mole of upper phase exsolved at coexistence.

        A small amount of the component-1-rich phase forms from the lower phase
        and is carried away, as when buoyant hydrogen-rich liquid leaves a
        magma. The result, (1-y)[h0(x)-h0(y)] + y[h1(x)-h1(y)] with lower and
        upper compositions x and y, is in J per native mole of removed phase.
        NaN where there is no gap or the tangent is invalid.
        """
        c = self.coexistence(temperature, pressure)
        lower = self.partial_enthalpies(temperature, pressure, c.lower)
        upper = self.partial_enthalpies(temperature, pressure, c.upper)
        # 1 - y from its own logit, so an almost pure upper phase keeps precision.
        y0, y1 = jax.nn.sigmoid(-c.logits[..., 1]), c.upper
        heat = y0 * (lower[..., 0] - upper[..., 0]) + y1 * (lower[..., 1] - upper[..., 1])
        return jnp.where(c.has_gap & c.valid, heat, jnp.nan)

    def gradient_coefficient(
        self, temperature: ArrayLike, pressure: ArrayLike, interface_width: ArrayLike
    ) -> FloatArray:
        """Square-gradient coefficient that gives a planar coexistence interface width.

        The coefficient kappa (J m²/mol) multiplies the squared gradient of the
        mixing-basis composition. The width (m) is the 10-90 % distance of the
        planar profile between the coexisting phases at this T and P. A width is
        a property of one state; kappa carries it to other temperatures. NaN
        without a gap or for an invalid tangent.
        """
        a, b = self._parameters(temperature, pressure)
        c = self.coexistence(temperature, pressure)
        k = jnp.log(self.component_0_multiplicity)
        _, width = _nucleation.planar_interface(a, b, c.logits[..., 0] - k, c.logits[..., 1] - k)
        kappa = GAS_CONSTANT * as_j64(temperature) * (as_j64(interface_width) / width) ** 2
        return jnp.where(c.has_gap & c.valid, kappa, jnp.nan)

    def interfacial_tension(
        self,
        temperature: ArrayLike,
        pressure: ArrayLike,
        gradient_coefficient: ArrayLike,
        molar_volume: ArrayLike,
    ) -> FloatArray:
        """Tension (J/m²) of the planar interface between the coexisting phases.

        Args:
            temperature: Temperature (K).
            pressure: Pressure (bar).
            gradient_coefficient: kappa (J m²/mol), see :meth:`gradient_coefficient`.
            molar_volume: Volume of one mixing-basis mole (m³/mol), taken as
                constant across the interface.
        """
        a, b = self._parameters(temperature, pressure)
        c = self.coexistence(temperature, pressure)
        k = jnp.log(self.component_0_multiplicity)
        s, _ = _nucleation.planar_interface(a, b, c.logits[..., 0] - k, c.logits[..., 1] - k)
        rt = GAS_CONSTANT * as_j64(temperature)
        length = jnp.sqrt(as_j64(gradient_coefficient) / rt)
        tension = rt / as_j64(molar_volume) * length * 2 * s
        return jnp.where(c.has_gap & c.valid, tension, jnp.nan)

    def nucleation_barrier(
        self,
        temperature: ArrayLike,
        pressure: ArrayLike,
        x: ArrayLike,
        gradient_coefficient: ArrayLike,
        molar_volume: ArrayLike,
    ) -> NucleationBarrier:
        """Energy of the critical nucleus that a homogeneous parent must form to separate.

        Square-gradient theory (Cahn & Hilliard 1959) with a constant gradient
        coefficient and molar volume. The barrier scales exactly as kappa^(3/2)
        and as 1/volume. Near the binodal it approaches classical theory; at the
        spinodal it vanishes as (distance)^(3/2). This is a thermodynamic
        barrier: a rate also needs a kinetic prefactor and a volume and time,
        usually through a criterion such as W/kT = ln(J0 V t).

        Args:
            temperature: Temperature (K).
            pressure: Pressure (bar).
            x: Native component-1 mole fraction of the homogeneous parent.
            gradient_coefficient: kappa (J m²/mol) of the mixing-basis composition.
            molar_volume: Volume of one mixing-basis mole (m³/mol).
        """
        t, p, x = jnp.broadcast_arrays(as_j64(temperature), as_j64(pressure), as_j64(x))
        shape = t.shape
        t, p, x = t.reshape(-1), p.reshape(-1), x.reshape(-1)
        a, b = self._parameters(t, p)
        a, b = jnp.broadcast_to(a, t.shape), jnp.broadcast_to(b, t.shape)
        sl, sr, gap = _spinodal(a, b)
        c = self.coexistence(t, p)
        z0 = self.mixing_composition(x)
        inside = c.contains(x)
        unstable = inside & (z0 > sl) & (z0 < sr)
        metastable = inside & ~unstable
        # Solve a harmless dummy problem where no nucleus exists.
        qs = (jnp.log(sl) - jnp.log1p(-sl), jnp.log(sr) - jnp.log1p(-sr))
        dummy = jnp.where(z0 < 0.5, sl / 2, (1 + sr) / 2)
        z_work = jnp.where(metastable, z0, jnp.where(gap, dummy, 0.5))
        a_work, b_work = jnp.where(gap, a, 3.0), jnp.where(gap, b, 3.0)
        # Spinodal logits of the a = b = 3 dummy, z = (1 -/+ sqrt(1/3)) / 2.
        q_dummy = math.log((1 + math.sqrt(1 / 3)) / (1 - math.sqrt(1 / 3)))
        qs = tuple(jnp.where(gap, q, v) for q, v in zip(qs, (-q_dummy, q_dummy)))
        energy, centre, drive, solved = jax.vmap(_nucleation.critical_nucleus)(
            a_work, b_work, z_work, qs
        )
        k = jnp.log(self.component_0_multiplicity)
        s, _ = _nucleation.planar_interface(a, b, c.logits[..., 0] - k, c.logits[..., 1] - k)
        classical = 4 * jnp.pi * (32 / 3) * s**3 / drive**2
        rt = GAS_CONSTANT * t
        kappa, volume = as_j64(gradient_coefficient), as_j64(molar_volume)
        scale = rt / volume * (kappa / rt) ** 1.5
        inputs = c.valid & (x >= 0) & (x <= 1) & (kappa > 0) & (volume > 0)
        inputs &= jnp.isfinite(kappa) & jnp.isfinite(volume)
        valid = inputs & (~metastable | solved)
        # Classical theory needs only the driving force and the planar tension.
        classical_valid = inputs & (~metastable | (jnp.isfinite(drive) & (drive > 0)))

        def assemble(values, metastable_value, ok):
            values = jnp.where(metastable, metastable_value, values)
            return jnp.where(ok, values, jnp.nan).reshape(shape)

        stable_or_unstable = jnp.where(unstable, 0.0, jnp.inf)
        energy = assemble(stable_or_unstable, scale * energy, valid)
        return NucleationBarrier(
            energy=energy,
            classical_energy=assemble(stable_or_unstable, scale * classical, classical_valid),
            nucleus_composition=assemble(jnp.nan * t, self.native_composition(centre), valid),
            energy_over_kT=energy * AVOGADRO / (GAS_CONSTANT * t.reshape(shape)),
            valid=valid.reshape(shape),
        )

    def diagnostics(
        self, temperature: ArrayLike, pressure: ArrayLike, x: ArrayLike
    ) -> dict[str, Array]:
        """Binary phase diagnostics; all entries are scalar or elementwise batches.

        Stability refers to the *homogeneous* free energy, even when equilibrium
        activities use its convex envelope. The lever-rule fraction describes
        the equilibrium end state, not a decomposition rate or elapsed progress.
        Spinodal endpoints are NaN when absent; no-gap binodal sentinels are hidden.
        """
        x = as_j64(x)
        a, b = self._parameters(temperature, pressure)
        sl, sr, gap = _spinodal(a, b)
        sl, sr = self.native_composition(sl), self.native_composition(sr)
        coexistence = self.coexistence(temperature, pressure)
        curvature = self.curvature(temperature, pressure, x)
        separated = coexistence.contains(x)
        enthalpy = self.enthalpy(temperature, pressure, x)
        equilibrium_enthalpy = self._lever_rule(
            self.enthalpy, temperature, pressure, x, coexistence
        )
        return {
            "bulk_component_1_mole_fraction": x,
            "mixing_basis_component_1_mole_fraction": self.mixing_composition(x),
            "homogeneous_curvature_J_per_mol": curvature,
            "spinodally_unstable": curvature < 0,
            "metastable": separated & (curvature >= 0),
            "equilibrium_two_phase": separated,
            "valid": coexistence.valid & (x >= 0) & (x <= 1),
            "spinodal_lower": jnp.where(gap, sl, jnp.nan),
            "spinodal_upper": jnp.where(gap, sr, jnp.nan),
            "binodal_lower": jnp.where(gap, coexistence.lower, jnp.nan),
            "binodal_upper": jnp.where(gap, coexistence.upper, jnp.nan),
            "equilibrium_upper_phase_mole_fraction": coexistence.upper_fraction(x),
            "common_tangent_residual": coexistence.residual,
            "homogeneous_mixing_enthalpy_J_per_mol": enthalpy,
            "equilibrium_mixing_enthalpy_J_per_mol": equilibrium_enthalpy,
            "homogeneous_mixing_entropy_J_per_mol_K": self.entropy(temperature, pressure, x),
            "equilibrium_mixing_entropy_J_per_mol_K": self._lever_rule(
                self.entropy, temperature, pressure, x, coexistence
            ),
            "homogeneous_mixing_volume_m3_per_mol": self.volume(temperature, pressure, x),
            "equilibrium_mixing_volume_m3_per_mol": self._lever_rule(
                self.volume, temperature, pressure, x, coexistence
            ),
            "demixing_heat_J_per_mol": enthalpy - equilibrium_enthalpy,
        }
