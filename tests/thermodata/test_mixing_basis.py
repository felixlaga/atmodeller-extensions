# SPDX-License-Identifier: Apache-2.0

"""Independent extensive-energy and stability tests for the 2026 component basis."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import brentq

from atmodeller import BinarySolutionPhase, ChemicalSpecies, SubregularSolution
from atmodeller.sci_utils import GAS_CONSTANT as R


def published_total_gibbs(n, t, p):
    # Gilmore & Stixrude eqs. 1 and 8, from native (MgSiO3, H2) amounts.
    total = 2 * n[0] + n[1]
    z = n[1] / total
    ideal = R * t * ((1 - z) * jnp.log1p(-z) + z * jnp.log(z))
    excess = z * (1 - z) * (75000 * (1 - z) + 424000 * z)
    return total * (ideal + excess * (1 - t / 4400 - p / 350000))


@pytest.mark.parametrize("x", [0.02, 0.3, 0.7, 0.98])
def test_basis_preserves_extensive_energy_chemical_potentials_and_curvature(x):
    m = SubregularSolution.hydrogen_silicate_gilmore26()
    n = jnp.array([1 - x, x])
    t, p = 3000.0, 20000.0
    np.testing.assert_allclose(m.gibbs_energy(t, p, x), published_total_gibbs(n, t, p), rtol=1e-13)
    expected = jax.grad(lambda v: published_total_gibbs(v, t, p))(n) / (R * t)
    actual = m.log_activities(t, p, x, equilibrate=False)
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    ideal = jnp.log(n)
    np.testing.assert_allclose(m.log_activity_coefficients(t, p, x), actual - ideal, atol=1e-12)
    curvature = jax.grad(jax.grad(lambda z: published_total_gibbs(jnp.array([1 - z, z]), t, p)))(x)
    np.testing.assert_allclose(m.curvature(t, p, x), curvature, rtol=1e-12)
    dmu = jax.jacfwd(lambda z: m.log_activities(t, p, z, equilibrate=False))(x)
    np.testing.assert_allclose(jnp.dot(n, dmu), 0, atol=1e-12)


def test_binodal_compositions_activities_and_native_lever_rule_transform_together():
    m = SubregularSolution.hydrogen_silicate_gilmore26()
    raw = SubregularSolution(75000, 424000, temperature_scale=4400, pressure_scale=-350000)
    c, r = m.coexistence(3000.0, 20000.0), raw.coexistence(3000.0, 20000.0)
    assert c.valid and c.has_gap
    np.testing.assert_allclose(
        [c.lower, c.upper], 2 * np.array([r.lower, r.upper]) / (1 + np.array([r.lower, r.upper]))
    )
    np.testing.assert_allclose(c.log_activity, r.log_activity * jnp.array([2.0, 1.0]), atol=1e-12)
    x = (2 * c.lower + c.upper) / 3
    fraction = c.upper_fraction(x)
    assert float(fraction) == pytest.approx(1 / 3)
    np.testing.assert_allclose((1 - fraction) * c.lower + fraction * c.upper, x)
    grid = jnp.linspace(0, 1, 1001)
    energy = eqx.filter_jit(m.equilibrium_gibbs_energy)(3000.0, 20000.0, grid)
    homogeneous = m.gibbs_energy(3000.0, 20000.0, grid)
    assert float(jnp.min(homogeneous - energy)) >= -1e-8
    np.testing.assert_array_equal(energy[jnp.array([0, -1])], 0)


@pytest.mark.parametrize(
    "factory",
    [
        SubregularSolution.hydrogen_silicate_rogers25,
        SubregularSolution.hydrogen_silicate_gilmore26,
    ],
)
def test_spinodal_temperature_against_independent_root(factory):
    m = factory()
    p, x = 20000.0, 0.7
    root = brentq(lambda t: float(m.curvature(t, p, x)), 1000, 4500, xtol=1e-10)
    assert isinstance(root, float)
    np.testing.assert_allclose(m.spinodal_temperature(p, x), root, rtol=1e-12)
    assert float(m.curvature(root - 1, p, x)) < 0
    assert float(m.curvature(root + 1, p, x)) > 0
    np.testing.assert_allclose(m.mixing_composition(m.native_composition(0.4)), 0.4)
    assert np.isnan(m.spinodal_temperature(p, jnp.array([0.0, 1.0, -0.1, np.nan]))).all()
    assert np.isnan(m.spinodal_temperature(-1.0, x))


def test_one_percent_mass_has_no_positive_spinodal_in_final_fit():
    # Exact nominal masses suffice for the qualitative test; use formula-unit x.
    x = (0.01 / 2.016) / (0.01 / 2.016 + 0.99 / 100.39)
    m = SubregularSolution.hydrogen_silicate_gilmore26()
    assert np.isnan(m.spinodal_temperature(20000.0, x))
    assert (np.asarray(m.curvature(jnp.array([2000.0, 2500.0, 3000.0]), 20000.0, x)) > 0).all()


def test_transformed_engine_activities_preserve_extremely_dilute_component():
    m = SubregularSolution.hydrogen_silicate_gilmore26()
    species = [ChemicalSpecies.create_solution(s) for s in ("Fe", "Si")]
    # Synthetic components isolate engine behavior; this is not a calibration for Fe-Si.
    phase = BinarySolutionPhase(species, m, "metal", equilibrate=False)
    logs = jnp.array([-100.0, 0.0])
    value = phase.get_log_activity(logs, jnp.array(4000.0), jnp.array(20000.0))
    assert np.isfinite(value).all()
    expected_mu0 = 2 * (
        -100 + np.log(2) + 424000 * (1 - 4000 / 4400 - 20000 / 350000) / (R * 4000)
    )
    assert float(value[0]) == pytest.approx(expected_mu0, abs=1e-12)


def test_transformed_derivatives_batching_and_no_gap_energy():
    m = SubregularSolution.hydrogen_silicate_gilmore26()

    def fn(t):
        return m.coexistence(t, 20000.0).logits

    derivative = jax.jacfwd(fn)(3000.0)
    difference = (fn(3000.01) - fn(2999.99)) / 0.02
    np.testing.assert_allclose(derivative, difference, rtol=1e-7)
    temperatures = jnp.array([2500.0, 3000.0, 4000.0])
    batch = eqx.filter_jit(m.diagnostics)(temperatures, 20000.0, 0.7)
    assert np.asarray(batch["valid"]).all()
    for t in (3000.0, 4000.0):
        assert np.isfinite(jax.grad(lambda x: m.equilibrium_gibbs_energy(t, 20000.0, x))(0.7))


@pytest.mark.parametrize("k", [0.0, -1.0, np.nan, np.inf])
def test_invalid_basis_multiplicity_is_rejected(k):
    with pytest.raises(ValueError, match="multiplicity"):
        SubregularSolution(1.0, 2.0, component_0_multiplicity=k)
