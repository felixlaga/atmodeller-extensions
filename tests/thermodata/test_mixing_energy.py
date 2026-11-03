# SPDX-License-Identifier: Apache-2.0

"""Mixing enthalpy, entropy, volume and separation heat, checked independently."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import root
from scipy.special import expit, logit

from atmodeller.sci_utils import GAS_CONSTANT as R
from atmodeller.thermodata.mixing import SubregularSolution

MODELS = {
    "gilmore26": SubregularSolution.hydrogen_silicate_gilmore26,
    "rogers25": SubregularSolution.hydrogen_silicate_rogers25,
    "scaled": lambda: SubregularSolution(
        -2000.0, 110000.0, temperature_scale=5000, pressure_scale=-4e5
    ),
}
MGSIO3, H2 = 100.3887, 2.01588
"""Molar masses (g/mol) of the two native Gilmore & Stixrude components"""


def native_x(mass_fraction_h2):
    """Native H2 mole fraction of an H2-MgSiO3 mixture with this H2 mass fraction."""
    n_h2 = mass_fraction_h2 / H2
    return n_h2 / (n_h2 + (1 - mass_fraction_h2) / MGSIO3)


def analytic(model, t, p, x):
    """Enthalpy, entropy and volume per native mole, written out from equation 1."""
    k = model.component_0_multiplicity
    a, b = float(model.interaction_0), float(model.interaction_1)
    tau, pi = float(model.temperature_scale), float(model.pressure_scale)
    x = np.asarray(x, dtype=float)
    scale = k + (1 - k) * x
    z = x / scale
    w = z * (1 - z) * (a * (1 - z) + b * z)
    with np.errstate(divide="ignore", invalid="ignore"):
        ideal = np.where((z > 0) & (z < 1), z * np.log(z) + (1 - z) * np.log1p(-z), 0.0)
    return (
        scale * w * (1 + p / pi),
        scale * (w / tau - R * ideal),
        scale * w / pi * 1e-5,
    )


def published_binodal(t, p):
    """Gilmore & Stixrude coexistence in the oxide basis, solved with SciPy."""
    factor = (1 - t / 4400 + p / -350000) / (R * t)
    a, b = 75000 * factor, 424000 * factor

    def potentials(z):
        f = (1 - z) * np.log1p(-z) + z * np.log(z) + z * (1 - z) * (a * (1 - z) + b * z)
        df = np.log(z / (1 - z)) + a + 2 * (b - 2 * a) * z + 3 * (a - b) * z * z
        return np.array([f - z * df, f + (1 - z) * df])

    guess = SubregularSolution(75000, 424000, temperature_scale=4400, pressure_scale=-350000)
    c = guess.coexistence(t, p)
    answer = root(
        lambda pair: potentials(expit(pair[0])) - potentials(expit(pair[1])),
        logit(np.array([float(c.lower), float(c.upper)])) + np.array([0.01, -0.01]),
        tol=1e-12,
    )
    assert answer.success
    return expit(answer.x)


@pytest.mark.parametrize("name", MODELS)
def test_homogeneous_energies_match_the_written_out_model(name):
    model = MODELS[name]()
    t, p = 3000.0, 20000.0
    x = jnp.array([0.0, 0.02, 0.3, 0.7, 1.0])
    h, s, v = analytic(model, t, p, x)
    np.testing.assert_allclose(model.enthalpy(t, p, x), h, rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(model.entropy(t, p, x), s, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(model.volume(t, p, x), v, rtol=1e-12, atol=1e-20)
    # Temperature enters the excess energy linearly, so Cp of mixing is zero.
    np.testing.assert_allclose(model.heat_capacity(t, p, x), 0, atol=1e-9)
    np.testing.assert_allclose(
        model.enthalpy(t, p, x) - t * model.entropy(t, p, x),
        model.gibbs_energy(t, p, x),
        rtol=1e-12,
        atol=1e-8,
    )


@pytest.mark.parametrize("name", MODELS)
def test_partial_enthalpies_are_derivatives_of_the_extensive_enthalpy(name):
    model = MODELS[name]()
    t, p = 3000.0, 20000.0

    def total_enthalpy(n):
        return jnp.sum(n) * model.enthalpy(t, p, n[1] / jnp.sum(n))

    for x in (0.01, 0.3, 0.7, 0.99):
        n = jnp.array([1 - x, x])
        partial = model.partial_enthalpies(t, p, x)
        np.testing.assert_allclose(partial, jax.grad(total_enthalpy)(n), rtol=1e-10, atol=1e-6)
        np.testing.assert_allclose(jnp.dot(n, partial), model.enthalpy(t, p, x), rtol=1e-12)


def test_hydrogen_partial_enthalpy_has_the_closed_form():
    model = SubregularSolution.hydrogen_silicate_gilmore26()
    p = 20000.0
    x = native_x(0.01)
    z = x / (2 - x)
    expected = (1 - z) ** 2 * (75000 + 2 * (424000 - 75000) * z) * (1 - p / 350000)
    for t in (2500.0, 4000.0):
        assert float(model.partial_enthalpies(t, p, x)[1]) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("t,p", [(3000.0, 20000.0), (3300.0, 20000.0), (3000.0, 50000.0)])
def test_equilibrium_energies_obey_the_envelope_theorem(t, p):
    model = SubregularSolution.hydrogen_silicate_gilmore26()
    x = native_x(0.03)
    c = model.coexistence(t, p)
    assert c.contains(x)
    g = model.equilibrium_gibbs_energy(t, p, x)
    dg_dt = jax.jvp(lambda v: model.equilibrium_gibbs_energy(v, p, x), (t,), (1.0,))[1]
    dg_dp = jax.jvp(lambda v: model.equilibrium_gibbs_energy(t, v, x), (p,), (1.0,))[1]
    np.testing.assert_allclose(model.equilibrium_entropy(t, p, x), -dg_dt, rtol=1e-9)
    np.testing.assert_allclose(model.equilibrium_volume(t, p, x), dg_dp * 1e-5, rtol=1e-9)
    np.testing.assert_allclose(
        model.equilibrium_enthalpy(t, p, x) - t * model.equilibrium_entropy(t, p, x),
        g,
        rtol=1e-11,
    )


def test_demixing_heat_is_zero_outside_the_gap_and_continuous_at_the_binodal():
    model = SubregularSolution.hydrogen_silicate_gilmore26()
    t, p = 3000.0, 20000.0
    c = model.coexistence(t, p)
    lower, upper = float(c.lower), float(c.upper)
    outside = jnp.array([lower * (1 - 1e-7), 0.5 * lower, 1.0, 0.0])
    np.testing.assert_array_equal(model.demixing_heat(t, p, outside), 0.0)
    inside = jnp.linspace(lower, upper, 101)[1:-1]
    heat = model.demixing_heat(t, p, inside)
    assert bool(jnp.all(heat > 0))
    edge = model.demixing_heat(t, p, lower * (1 + 1e-7))
    assert 0 < float(edge) < 1e-6 * float(jnp.max(heat))
    hot = model.demixing_heat(6000.0, p, inside)
    np.testing.assert_array_equal(hot, 0.0)


def test_demixing_heat_matches_an_independent_lever_rule():
    model = SubregularSolution.hydrogen_silicate_gilmore26()
    t, p, w = 2942.2, 20000.0, 0.03
    x = native_x(w)
    zl, zu = published_binodal(t, p)
    z = x / (2 - x)
    fraction = (z - zl) / (zu - zl)

    def excess_enthalpy(v):
        return v * (1 - v) * (75000 * (1 - v) + 424000 * v) * (1 - p / 350000)

    # Lever rule on oxide-basis units: the enthalpy per oxide mole, times 2-x.
    released = excess_enthalpy(z) - (1 - fraction) * excess_enthalpy(zl)
    released -= fraction * excess_enthalpy(zu)
    expected = released * (2 - x)
    assert float(model.demixing_heat(t, p, x)) == pytest.approx(expected, rel=1e-9)
    per_kg = expected / ((1 - x) * MGSIO3 + x * H2) * 1e3
    # About 1.2 MJ/kg released once the 3 wt% parcel has reached its spinodal.
    assert per_kg == pytest.approx(1.2e6, rel=0.05)


def test_equilibrium_heat_capacity_matches_a_finite_difference():
    model = SubregularSolution.hydrogen_silicate_gilmore26()
    p, x = 20000.0, native_x(0.03)
    for t in (2800.0, 3200.0):
        step = 1e-2
        difference = (
            model.equilibrium_enthalpy(t + step, p, x) - model.equilibrium_enthalpy(t - step, p, x)
        ) / (2 * step)
        capacity = model.equilibrium_heat_capacity(t, p, x)
        np.testing.assert_allclose(capacity, difference, rtol=1e-6)
        # Cooling inside the gap releases enthalpy, so the apparent Cp is positive.
        assert float(capacity) > 0


@pytest.mark.parametrize("t,p", [(2500.0, 20000.0), (3300.0, 20000.0), (3000.0, 50000.0)])
def test_exsolution_enthalpy_matches_a_finite_removal_step(t, p):
    model = SubregularSolution.hydrogen_silicate_gilmore26()
    c = model.coexistence(t, p)
    lower, upper = float(c.lower), float(c.upper)
    removed = 1e-7
    remaining = 1 - removed
    x = (lower - removed * upper) / remaining
    before = float(model.enthalpy(t, p, lower))
    after = remaining * float(model.enthalpy(t, p, x)) + removed * float(
        model.enthalpy(t, p, upper)
    )
    expected = (before - after) / removed
    assert float(model.exsolution_enthalpy(t, p)) == pytest.approx(expected, rel=1e-5)
    assert float(model.exsolution_enthalpy(t, p)) > 0


def test_exsolution_enthalpy_is_nan_without_a_gap():
    model = SubregularSolution.hydrogen_silicate_gilmore26()
    assert np.isnan(model.exsolution_enthalpy(6000.0, 20000.0))


@pytest.mark.parametrize("t,p,x", [(-1.0, 1.0, 0.3), (3000.0, -1.0, 0.3), (3000.0, 1.0, 1.1)])
def test_invalid_inputs_give_nan(t, p, x):
    model = SubregularSolution.hydrogen_silicate_gilmore26()
    for method in (
        model.enthalpy,
        model.entropy,
        model.volume,
        model.equilibrium_enthalpy,
        model.equilibrium_entropy,
        model.demixing_heat,
    ):
        assert np.isnan(method(t, p, x))
    assert np.isnan(model.partial_enthalpies(t, p, x)).all()


def test_energy_functions_jit_and_broadcast():
    model = SubregularSolution.hydrogen_silicate_gilmore26()
    temperatures = jnp.array([2800.0, 3200.0, 6000.0])
    x = native_x(0.03)
    batch = eqx.filter_jit(model.demixing_heat)(temperatures, 20000.0, x)
    mapped = jax.jit(jax.vmap(lambda t: model.demixing_heat(t, 20000.0, x)))(temperatures)
    np.testing.assert_allclose(batch, mapped, rtol=1e-12)
    assert batch.shape == (3,)
    assert float(batch[-1]) == 0


def test_diagnostics_report_both_states_and_the_separation_heat():
    model = SubregularSolution.hydrogen_silicate_gilmore26()
    t, p, x = 3000.0, 20000.0, native_x(0.03)
    d = model.diagnostics(t, p, x)
    np.testing.assert_allclose(d["homogeneous_mixing_enthalpy_J_per_mol"], model.enthalpy(t, p, x))
    np.testing.assert_allclose(
        d["equilibrium_mixing_enthalpy_J_per_mol"], model.equilibrium_enthalpy(t, p, x)
    )
    np.testing.assert_allclose(
        d["homogeneous_mixing_entropy_J_per_mol_K"], model.entropy(t, p, x)
    )
    np.testing.assert_allclose(
        d["equilibrium_mixing_entropy_J_per_mol_K"], model.equilibrium_entropy(t, p, x)
    )
    np.testing.assert_allclose(d["homogeneous_mixing_volume_m3_per_mol"], model.volume(t, p, x))
    np.testing.assert_allclose(
        d["equilibrium_mixing_volume_m3_per_mol"], model.equilibrium_volume(t, p, x)
    )
    np.testing.assert_allclose(d["demixing_heat_J_per_mol"], model.demixing_heat(t, p, x))
