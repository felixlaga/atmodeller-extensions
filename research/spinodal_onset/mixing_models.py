# SPDX-License-Identifier: Apache-2.0
"""Independent scalar thermodynamics for T-linear binary excess models.

Used by the recalibration and model-class analysis. Compositions here are the
published fit-basis H2 fraction z = n(H2)/[n(H2)+n(MgO)+n(SiO2)]; native
formula-unit fractions are y = 2z/(1+z). Energies are J per fit-basis mole.

The excess Gibbs energy is a Redlich-Kister series with constant excess
enthalpy, entropy and volume coefficients (zero excess heat capacity):

    G_ex = sum_k (h_k - T s_k + P v_k) z(1-z)(1-2z)^k,   P in GPa.

The published Gilmore & Stixrude (2026) model is the k <= 1 case with
s_k = h_k/tau and v_k = h_k/pi. This module is pure Python/NumPy and shares no
code with Atmodeller, so agreement with ``SubregularSolution`` is a real check.
"""

from dataclasses import dataclass
import math

import numpy as np
from scipy.optimize import brentq, minimize_scalar

R = 8.314462618  # J/mol/K
M_H2 = 2.015882  # molmass Formula("H2").mass, as in run_study.py
M_MGSIO3 = 100.388815  # molmass Formula("MgSiO3").mass
EDGE = 36.0  # logit bound: z within 2e-16 of 0 or 1


def logit(z):
    return math.log(z / (1 - z))


def expit(t):
    return 1 / (1 + math.exp(-t)) if t >= 0 else math.exp(t) / (1 + math.exp(t))


def weight_to_z(w):
    """Local binary H2 mass fraction -> fit-basis H2 fraction z."""
    y = (w / M_H2) / (w / M_H2 + (1 - w) / M_MGSIO3)
    return y / (2 - y)


def z_to_y(z):
    return 2 * z / (1 + z)


def basis(z, k, order):
    """d^order/dz^order of z(1-z)(1-2z)^k, written with u = 1-2z."""
    u = 1 - 2 * z

    def p(n):
        return u**n if n >= 0 else 0.0

    if order == 0:
        return (p(k) - p(k + 2)) / 4
    if order == 1:
        return -(k * p(k - 1) - (k + 2) * p(k + 1)) / 2
    if order == 2:
        return k * (k - 1) * p(k - 2) - (k + 1) * (k + 2) * p(k)
    if order == 3:
        return -2 * (k * (k - 1) * (k - 2) * p(k - 3) - (k + 1) * (k + 2) * k * p(k - 1))
    raise ValueError(order)


_CURVATURE = {}


def _curvature_polynomial(k):
    """Coefficients (low to high) of z(1-z) d2/dz2[z(1-z)(1-2z)^k]."""
    if k not in _CURVATURE:
        u = np.polynomial.Polynomial([1.0, -2.0])
        second = -(k + 1) * (k + 2) * u**k
        if k >= 2:
            second = second + k * (k - 1) * u ** (k - 2)
        _CURVATURE[k] = (np.polynomial.Polynomial([0.0, 1.0, -1.0]) * second).coef
    return _CURVATURE[k]


@dataclass(frozen=True)
class ExcessModel:
    """Excess coefficients h (J/mol), s (J/mol/K) and v (J/mol/GPa) per RK order."""

    h: tuple
    s: tuple
    v: tuple

    @classmethod
    def published_form(cls, a, b, tau, pi, extra=()):
        """(A(1-z)+Bz + higher RK terms)(1 - T/tau + P/pi); tau in K, pi in GPa.

        ``extra`` adds RK coefficients L_2, L_3, ... sharing the same T/P factor.
        Infinite tau or pi disables that dependence.
        """
        h = ((a + b) / 2, (a - b) / 2, *extra)
        inv_tau = 0.0 if math.isinf(tau) else 1 / tau
        inv_pi = 0.0 if math.isinf(pi) else 1 / pi
        return cls(h, tuple(c * inv_tau for c in h), tuple(c * inv_pi for c in h))

    @classmethod
    def subregular_hsv(cls, ah, bh, as_, bs, av, bv):
        """Independent subregular H (J/mol), S (J/mol/K) and V (J/mol/GPa) shapes."""
        return cls(
            ((ah + bh) / 2, (ah - bh) / 2),
            ((as_ + bs) / 2, (as_ - bs) / 2),
            ((av + bv) / 2, (av - bv) / 2),
        )

    def coefficients(self, t, p):
        return [h - t * s + p * v for h, s, v in zip(self.h, self.s, self.v)]

    def excess(self, z, t, p, order=0):
        return sum(c * basis(z, k, order) for k, c in enumerate(self.coefficients(t, p)))

    # Homogeneous properties, per fit mole: H includes the P*V_ex contribution.
    def enthalpy(self, z, p):
        return sum((h + p * v) * basis(z, k, 0) for k, (h, v) in enumerate(zip(self.h, self.v)))

    def entropy(self, z):
        return sum(s * basis(z, k, 0) for k, s in enumerate(self.s))

    def volume_cm3(self, z):
        return 1e-3 * sum(v * basis(z, k, 0) for k, v in enumerate(self.v))

    # Reduced homogeneous free energy f = g_mix/(RT) and derivatives.
    def f(self, z, t, p):
        ideal = z * math.log(z) + (1 - z) * math.log1p(-z) if 0 < z < 1 else 0.0
        return ideal + self.excess(z, t, p) / (R * t)

    def f_array(self, zs, t, p):
        """Vectorized f for 0 < zs < 1 (NumPy array)."""
        return zs * np.log(zs) + (1 - zs) * np.log1p(-zs) + self.excess(zs, t, p) / (R * t)

    def df(self, z, t, p):
        return math.log(z) - math.log1p(-z) + self.excess(z, t, p, 1) / (R * t)

    def d2f(self, z, t, p):
        return 1 / (z * (1 - z)) + self.excess(z, t, p, 2) / (R * t)

    def spinodal_temperature(self, z, p):
        """Temperature where d2g/dz2 = 0 at z, or NaN if no cooling instability."""
        num = -sum((h + p * v) * basis(z, k, 2) for k, (h, v) in enumerate(zip(self.h, self.v)))
        den = R / (z * (1 - z)) - sum(s * basis(z, k, 2) for k, s in enumerate(self.s))
        if den <= 0 or num <= 0:
            return math.nan
        return num / den

    def spinodal_roots(self, t, p):
        """Sorted compositions in (0, 1) where d2f changes sign.

        z(1-z)RT*d2f = RT + z(1-z)G_ex'' is a polynomial, so its real roots in
        (0, 1) are the spinodal compositions; each is polished by bisection.
        """
        coef = np.zeros(len(self.h) + 2)
        coef[0] = R * t
        for k, c in enumerate(self.coefficients(t, p)):
            q = _curvature_polynomial(k)
            coef[: len(q)] += c * q
        roots = []
        for r in np.polynomial.polynomial.polyroots(np.trim_zeros(coef, "b")):
            if abs(r.imag) < 1e-9 and 1e-12 < r.real < 1 - 1e-12:
                z = float(r.real)
                lo, hi = max(z - 1e-7, 1e-15), min(z + 1e-7, 1 - 1e-15)
                if self.d2f(lo, t, p) * self.d2f(hi, t, p) < 0:
                    z = brentq(self.d2f, lo, hi, args=(t, p), xtol=1e-16, rtol=1e-15)
                    roots.append(z)
        return sorted(roots)

    def unstable_intervals(self, t, p):
        """Composition intervals with d2f < 0.

        d2f > 0 at both edges because the ideal term diverges there, so the
        spinodal roots pair up into unstable intervals.
        """
        roots = self.spinodal_roots(t, p)
        if len(roots) % 2:
            raise ValueError("Odd number of spinodal roots; model invalid here")
        return list(zip(roots[::2], roots[1::2]))

    def _branch(self, m, t, p, lo, hi):
        """Solve df(z)=m for z in (expit(lo), expit(hi)), where df increases."""
        g = lambda x: self.df(expit(x), t, p) - m  # noqa: E731
        return expit(brentq(g, lo, hi, xtol=1e-14, rtol=1e-15, maxiter=200))

    def binodal(self, t, p):
        """Coexisting fit-basis compositions (z1, z2), or None for one phase.

        Single unstable interval only (true for every accepted model here);
        more complex topologies raise instead of returning a partial answer.
        """
        intervals = self.unstable_intervals(t, p)
        if not intervals:
            return None
        if len(intervals) > 1:
            raise ValueError("Several unstable intervals: three-phase topology not handled")
        s1, s2 = intervals[0]
        m_hi, m_lo = self.df(s1, t, p), self.df(s2, t, p)
        x1, x2 = logit(s1), logit(s2)

        def delta(m):
            z1 = self._branch(m, t, p, -EDGE, x1)
            z2 = self._branch(m, t, p, x2, EDGE)
            return (self.f(z2, t, p) - m * z2) - (self.f(z1, t, p) - m * z1)

        pad = 1e-12 * max(1.0, abs(m_hi - m_lo))
        m = brentq(delta, m_lo + pad, m_hi - pad, xtol=1e-14, rtol=1e-15, maxiter=200)
        return self._branch(m, t, p, -EDGE, x1), self._branch(m, t, p, x2, EDGE)

    def tangent_gap(self, z0, t, p):
        """min over z of f(z) - tangent at z0, excluding z0's own convex branch.

        Negative: z0 lies inside the equilibrium gap. Used for root finding in T.
        A locally unstable z0 is inside the gap by definition (returns -1).
        The own branch runs between the spinodal compositions next to z0. f is
        convex there, so it holds no negative value and is skipped, however
        close the other phase is (near T_C it lies within 0.02 of z0). Elsewhere
        the minimum is a spinodal composition, a branch end or the point of
        another convex branch where df = df(z0); all are evaluated exactly.
        """
        if self.d2f(z0, t, p) <= 0:
            return -1.0
        f0, d0 = self.f(z0, t, p), self.df(z0, t, p)

        def gap(z):
            return self.f(z, t, p) - f0 - d0 * (z - z0)

        roots = self.spinodal_roots(t, p)
        if not roots:
            return 1.0  # f convex everywhere: z0 is on the hull (any positive value)
        values = [gap(s) for s in roots]
        edges = [-EDGE, *(logit(s) for s in roots), EDGE]
        for xl, xh in zip(edges[:-1], edges[1:]):
            zl, zh = expit(xl), expit(xh)
            if zl <= z0 <= zh or self.d2f(expit((xl + xh) / 2), t, p) <= 0:
                continue  # own branch, or concave: minimum at its ends (spinodal roots)
            if self.df(zl, t, p) < d0 < self.df(zh, t, p):
                values.append(gap(self._branch(d0, t, p, xl, xh)))
            else:
                values += [gap(zl), gap(zh)]
        return min(values)

    def critical_point(self, p):
        """(T_C, z_C): maximum of the spinodal temperature over composition."""
        zs = np.linspace(0.005, 0.995, 397)
        ts = np.array([self.spinodal_temperature(z, p) for z in zs])
        if not np.isfinite(ts).any():
            return math.nan, math.nan
        i = int(np.nanargmax(ts))
        lo, hi = zs[max(i - 1, 0)], zs[min(i + 1, len(zs) - 1)]
        r = minimize_scalar(
            lambda z: -self.spinodal_temperature(z, p),
            bounds=(lo, hi),
            method="bounded",
            options={"xatol": 1e-12},
        )
        return -r.fun, r.x

    def binodal_temperature(self, z0, p):
        """Highest temperature at which bulk z0 lies inside the equilibrium gap.

        Steps down from T_C in 2% increments to bracket the crossing, so very
        cold, edge-hugging gaps (beyond the logit grid) are never needed.
        """
        tc, _ = self.critical_point(p)
        if not math.isfinite(tc):
            return math.nan
        ts = self.spinodal_temperature(z0, p)
        floor = 0.9999 * ts if math.isfinite(ts) else 0.05 * tc  # strictly unstable at floor
        hi = tc + 1e-6
        for fraction in np.arange(0.98, 0.0, -0.02):
            lo = max(fraction * tc, floor)
            if self.tangent_gap(z0, lo, p) < 0:
                return brentq(lambda t: self.tangent_gap(z0, t, p), lo, hi, xtol=1e-6)
            if lo == floor:
                break
            hi = lo
        return math.nan  # z0 is never inside a gap above the floor

    def rich_hydrogen_fraction(self, z0, t, p):
        """Equilibrium fraction of all H in the H-rich phase (native lever rule)."""
        gap = self.binodal(t, p)
        if gap is None or not gap[0] < z0 < gap[1]:
            return 0.0
        y0, y1, y2 = z_to_y(z0), z_to_y(gap[0]), z_to_y(gap[1])
        phase = (y0 - y1) / (y2 - y1)  # native moles of the upper phase per mole
        return phase * y2 / y0


def published_gilmore26():
    return ExcessModel.published_form(75e3, 424e3, 4400.0, -35.0)
