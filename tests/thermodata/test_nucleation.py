# SPDX-License-Identifier: Apache-2.0

"""Square-gradient nucleation barriers against independent solutions and known limits."""

import jax.numpy as jnp
import numpy as np
import pytest

from atmodeller.sci_utils import GAS_CONSTANT as R
from atmodeller.thermodata.mixing import SubregularSolution

P = 20000.0
MODEL = SubregularSolution.hydrogen_silicate_gilmore26()


def native_x(mass_fraction_h2):
    n_h2 = mass_fraction_h2 / 2.01588
    return n_h2 / (n_h2 + (1 - mass_fraction_h2) / 100.3887)


X = native_x(0.03)
T_BINODAL = float(MODEL.binodal_temperature(P, X))


def dimensionless(barrier, t):
    # With kappa = RT (J m²/mol) and v = 1 m³/mol, the joules are RT times the
    # dimensionless integral; divide by 4 pi to match int(...) r² dr.
    return np.asarray(barrier) / (R * t) / (4 * np.pi)


@pytest.mark.parametrize(
    "below,expected,classical",
    # Independent finite-volume Newton saddles and classical integrals (SciPy),
    # published fit, 3 wt% H2 at 2 GPa; only Newton-converged references.
    [
        (20, 4.808987, 5.55173),
        (50, 1.071858, 1.62331),
        (100, 0.3729982, 0.898877),
        (200, 0.1130128, 0.693210),
        (1, 1405.726, 1413.53),
    ],
)
def test_barrier_matches_an_independent_saddle_solution(below, expected, classical):
    t = T_BINODAL - below
    b = MODEL.nucleation_barrier(t, P, X, R * t, 1.0)
    assert bool(b.valid)
    assert float(dimensionless(b.energy, t)) == pytest.approx(expected, rel=1e-3)
    assert float(dimensionless(b.classical_energy, t)) == pytest.approx(classical, rel=1e-5)


def test_barrier_approaches_classical_theory_near_the_binodal():
    ratios = []
    for below in (0.5, 2.0, 8.0):
        b = MODEL.nucleation_barrier(T_BINODAL - below, P, X, 1e-15, 1.8e-5)
        ratios.append(float(b.energy / b.classical_energy))
    assert ratios[0] > 0.99
    assert ratios[0] > ratios[1] > ratios[2]


def test_barrier_vanishes_at_the_spinodal_as_the_three_halves_power():
    t_sp = float(MODEL.spinodal_temperature(P, X))
    above = np.array([0.5, 2.0])
    energy = [float(MODEL.nucleation_barrier(t_sp + d, P, X, 1e-15, 1.8e-5).energy) for d in above]
    exponent = np.log(energy[1] / energy[0]) / np.log(above[1] / above[0])
    assert exponent == pytest.approx(1.5, abs=0.03)
    inside = MODEL.nucleation_barrier(t_sp - 1.0, P, X, 1e-15, 1.8e-5)
    assert bool(inside.valid) and float(inside.energy) == 0


def test_barrier_scales_with_gradient_coefficient_and_volume():
    t = T_BINODAL - 10
    base = MODEL.nucleation_barrier(t, P, X, 1e-15, 1.8e-5).energy
    scaled = MODEL.nucleation_barrier(t, P, X, 4e-15, 0.9e-5).energy
    assert float(scaled / base) == pytest.approx(16.0, rel=1e-12)


def test_stable_parent_needs_no_nucleus_and_bad_input_is_nan():
    hot = MODEL.nucleation_barrier(T_BINODAL + 1, P, X, 1e-15, 1.8e-5)
    assert bool(hot.valid) and np.isinf(hot.energy) and np.isnan(hot.nucleus_composition)
    bad = MODEL.nucleation_barrier(3000.0, -1.0, X, 1e-15, 1.8e-5)
    assert not bool(bad.valid) and np.isnan(bad.energy)


def test_batches_match_single_evaluations():
    t = jnp.array([T_BINODAL - 5, T_BINODAL - 50, T_BINODAL + 5])
    batch = MODEL.nucleation_barrier(t, P, X, 1e-15, 1.8e-5)
    single = MODEL.nucleation_barrier(t[1], P, X, 1e-15, 1.8e-5)
    assert batch.energy.shape == (3,)
    np.testing.assert_allclose(batch.energy[1], single.energy, rtol=1e-12)
    assert np.isinf(batch.energy[2])


def test_gradient_coefficient_reproduces_the_interface_width():
    # Independent planar-profile integral: 0.3 nm at 2 GPa and 3000 K.
    kappa = MODEL.gradient_coefficient(3000.0, P, 0.3e-9)
    assert float(kappa) == pytest.approx(6.250e-16, rel=1e-3)
    tension = MODEL.interfacial_tension(3000.0, P, kappa, 1.819e-5)
    assert 0 < float(tension) < 1.0
    assert np.isnan(MODEL.gradient_coefficient(6000.0, P, 0.3e-9))


def test_nearly_pure_nucleus_phase_does_not_exhaust_the_integration():
    # Independent SciPy DOP853 shooting: 0.219233 (W = 12RT) and 0.275874 (W = 11RT).
    for w, expected in ((12, 0.219233), (11, 0.275874)):
        model = SubregularSolution.regular_solution(w * R * 3000)
        x = 0.5 * float(model.spinodal(3000.0, 0.0)[0])
        b = model.nucleation_barrier(3000.0, 0.0, x, R * 3000, 1.0)
        assert bool(b.valid)
        assert float(b.energy) / (R * 3000) == pytest.approx(expected, rel=2e-5)


@pytest.mark.parametrize("kappa,volume", [(np.nan, 1.8e-5), (-1e-16, 1.8e-5), (1e-15, 0.0)])
def test_nonphysical_gradient_coefficient_or_volume_is_invalid(kappa, volume):
    for t in (T_BINODAL - 10, T_BINODAL - 500, T_BINODAL + 10):
        b = MODEL.nucleation_barrier(t, P, X, kappa, volume)
        assert not bool(b.valid) and np.isnan(b.energy) and np.isnan(b.classical_energy)


def test_classical_barrier_survives_a_failed_square_gradient_solve():
    # Within a microkelvin of the binodal the nucleus is too thin to shoot.
    b = MODEL.nucleation_barrier(T_BINODAL - 1e-6, P, X, 1e-15, 1.8e-5)
    assert not bool(b.valid) and np.isnan(b.energy)
    assert np.isfinite(b.classical_energy) and float(b.classical_energy) > 0
