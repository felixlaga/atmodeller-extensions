# SPDX-License-Identifier: Apache-2.0

"""Cooling paths of a closed H2-MgSiO3 parcel: energy, separation onset and recalescence."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from atmodeller import ChemicalSpecies, EquilibriumModel, SolverParameters, SubregularSolution
from atmodeller.cooling import ParcelCooling, nucleation_onset_temperature, solve_path_with_engine
from atmodeller.thermodata.defaults import default_properties

P = 20000.0
"""2 GPa in bar"""


def mixture():
    return (
        ChemicalSpecies.create_solution("MgSiO3"),
        ChemicalSpecies.create_solution("H2", thermo=default_properties["H2_g"]),
    )


@pytest.fixture(scope="module")
def parcel():
    return ParcelCooling.from_species(
        mixture(), SubregularSolution.hydrogen_silicate_gilmore26(), P, mass_fraction=0.03
    )


def test_composition_and_heat_capacity_follow_the_species_data(parcel):
    n_h2 = 0.03 / 2.01588e-3
    n_si = 0.97 / 100.3887e-3
    assert float(parcel.composition) == pytest.approx(n_h2 / (n_h2 + n_si), rel=1e-5)
    step = 1.0
    cp = (parcel.sensible_enthalpy(3001.0) - parcel.sensible_enthalpy(2999.0)) / (2 * step)
    # MgSiO3 liquid 1.46 and H2 about 18 kJ/kg/K, mixed by mass.
    assert float(cp) == pytest.approx(0.97 * 1458 + 0.03 * 18400, rel=0.02)
    doubled = ParcelCooling.from_species(
        mixture(), parcel.mixing_model, P, mass_fraction=0.03, heat_capacity_scale=2.0
    )
    np.testing.assert_allclose(
        doubled.sensible_enthalpy(3100.0) - doubled.sensible_enthalpy(3000.0),
        2 * (parcel.sensible_enthalpy(3100.0) - parcel.sensible_enthalpy(3000.0)),
        rtol=1e-12,
    )


def test_equilibrium_path_starts_separating_at_the_binodal(parcel):
    path = parcel.path(3600.0, 2600.0, steps=60)
    assert bool(path.valid)
    assert float(path.binodal_temperature) == pytest.approx(3355.9, abs=0.05)
    t, q = np.asarray(path.temperature), np.asarray(path.heat_lost)
    separated = np.asarray(path.separated)
    assert not separated[t > float(path.binodal_temperature) + 1e-6].any()
    assert separated[t < float(path.binodal_temperature) - 1e-6].all()
    assert np.all(np.diff(t) <= 0) and np.all(np.diff(q) >= -1e-9)
    # Equilibrium separation is gradual: no reheating, no heat jump.
    assert float(path.separation_temperature) == pytest.approx(float(path.onset_temperature))
    assert float(path.released_heat) == pytest.approx(0, abs=1e-6)


def test_delayed_separation_reheats_and_conserves_enthalpy(parcel):
    onset = parcel.mixing_model.spinodal_temperature(P, parcel.composition)
    assert float(onset) == pytest.approx(2942.2, abs=0.05)
    path = parcel.path(3600.0, 2600.0, onset, steps=60)
    assert bool(path.valid)
    jump = float(path.separation_temperature - path.onset_temperature)
    # About 1.2 MJ/kg released at about 2 kJ/kg/K: several hundred kelvin of reheating.
    assert 250 < jump < 320
    assert float(path.separation_temperature) < float(path.binodal_temperature)
    q = np.asarray(path.heat_lost)
    # The last homogeneous row and the first separated row hold the same enthalpy.
    np.testing.assert_allclose(q[60], q[59], rtol=1e-9)
    assert float(path.released_heat) == pytest.approx(1.2e6, rel=0.05)


def test_final_state_and_heat_lost_do_not_depend_on_the_path(parcel):
    onset = parcel.mixing_model.spinodal_temperature(P, parcel.composition)
    paths = [
        parcel.path(3600.0, 2600.0, steps=40),
        parcel.path(3600.0, 2600.0, onset, steps=40),
        parcel.path(3600.0, 2600.0, onset, separation="isothermal", steps=40),
    ]
    finals = [float(p.heat_lost[-1]) for p in paths]
    np.testing.assert_allclose(finals, finals[0], rtol=1e-9)
    fractions = [float(p.upper_fraction[-1]) for p in paths]
    np.testing.assert_allclose(fractions, fractions[0], rtol=1e-12)


def test_isothermal_separation_removes_the_demixing_heat_at_the_onset(parcel):
    onset = 3100.0
    path = parcel.path(3600.0, 2600.0, onset, separation="isothermal", steps=40)
    q, t = np.asarray(path.heat_lost), np.asarray(path.temperature)
    assert t[39] == pytest.approx(onset) and t[40] == pytest.approx(onset)
    expected = parcel.mixing_model.demixing_heat(onset, P, parcel.composition)
    expected = float(expected) / float(parcel.molar_mass)
    assert q[40] - q[39] == pytest.approx(expected, rel=1e-9)
    assert float(path.released_heat) == pytest.approx(expected, rel=1e-9)


def test_parcels_that_never_separate_have_no_separated_rows(parcel):
    for onset in (jnp.nan, 2500.0):
        path = parcel.path(3600.0, 2600.0, onset, steps=20)
        assert bool(path.valid)
        assert not bool(jnp.any(path.separated))
        assert np.isnan(path.separation_temperature)
        np.testing.assert_allclose(path.temperature[19], 2600.0)
        assert bool(jnp.all(jnp.isnan(path.temperature[20:])))


def test_paths_batch_over_parameter_samples(parcel):
    def window(b):
        model = SubregularSolution(
            75e3, b, temperature_scale=4400.0, pressure_scale=-3.5e5, component_0_multiplicity=2.0
        )
        sample = ParcelCooling.from_species(mixture(), model, P, mass_fraction=0.03)
        onset = model.spinodal_temperature(P, sample.composition)
        path = sample.path(3800.0, 2400.0, onset, steps=20)
        return path.separation_temperature - path.onset_temperature

    jumps = jax.jit(jax.vmap(window))(jnp.array([400e3, 424e3, 450e3]))
    assert jumps.shape == (3,)
    assert bool(jnp.all(jumps > 0))
    np.testing.assert_allclose(jumps[1], window(424e3), rtol=1e-10)


def test_engine_path_agrees_with_the_fast_functions(parcel):
    steps = 12
    model = EquilibriumModel.from_species(
        [],
        temperature=jnp.full(steps, 3000.0),
        pressure=P,
        mass=0.0,
        silicate_melt_species=mixture(),
        mixing_models={"silicate_melt": parcel.mixing_model},
        phase_equilibrium="spinodal",
        mass_constraints={"H2": 0.03, "MgSiO3": 0.97},
        solver_parameters=SolverParameters(atol=1e-12, rtol=1e-12),
    )
    onset = parcel.mixing_model.spinodal_temperature(P, parcel.composition)
    path = parcel.path(3600.0, 2600.0, onset, steps=steps)
    engine = solve_path_with_engine(
        path, model.with_phase_equilibrium("homogeneous"), model, "silicate_melt"
    )
    assert bool(np.all(engine["success"]))
    np.testing.assert_allclose(engine["mixing_enthalpy"], path.mixing_enthalpy, rtol=1e-8)
    np.testing.assert_allclose(
        engine["upper_fraction"], path.upper_fraction, rtol=1e-8, atol=1e-12
    )


@pytest.mark.parametrize(
    "criterion,theory,expected",
    # Independent SciPy solutions: 0.3 nm interface at 2 GPa and 3000 K, 18.19 cm³/mol.
    [
        (69.0776, "square_gradient", 5.57),
        (123.186, "square_gradient", 4.11),
        (69.0776, "classical", 5.67),
        (123.186, "classical", 4.17),
    ],
)
def test_nucleation_ends_the_delay_a_few_kelvin_below_the_binodal(
    parcel, criterion, theory, expected
):
    model, x = parcel.mixing_model, parcel.composition
    kappa = model.gradient_coefficient(3000.0, P, 0.3e-9)
    onset = nucleation_onset_temperature(model, P, x, kappa, 1.819e-5, criterion, theory=theory)
    delay = float(model.binodal_temperature(P, x) - onset)
    assert delay == pytest.approx(expected, abs=0.02)
    path = parcel.path(3600.0, 2600.0, onset, steps=20)
    assert bool(path.valid)
    # Nucleation-limited separation is close to equilibrium: little reheating.
    assert float(path.separation_temperature - path.onset_temperature) < delay


def test_interior_tracks_the_binodal_and_releases_the_hydrogen_partial_enthalpy():
    from atmodeller.cooling import InteriorCooling

    model = SubregularSolution.hydrogen_silicate_gilmore26()
    interior = InteriorCooling.from_species(mixture(), model, P, mass_fraction=0.03)
    path = interior.path(3600.0, 2800.0, steps=400)
    assert bool(path.valid)
    t, x = np.asarray(path.temperature), np.asarray(path.composition)
    onset = float(model.binodal_temperature(P, interior.initial_composition))
    np.testing.assert_allclose(x[t > onset], interior.initial_composition, rtol=1e-12)
    below = t < onset - 1
    lower = model.coexistence(t[below], P).lower
    np.testing.assert_allclose(x[below], lower, rtol=1e-9)
    # Heat per kg of exsolved hydrogen: the H2 partial enthalpy, 40-75 MJ/kg.
    per_kg = float(path.exsolution_heat[-1] / path.exsolved_hydrogen[-1])
    assert 40e6 < per_kg < 75e6
    # Energy bookkeeping: heat lost is sensible plus exsolution heat.
    np.testing.assert_allclose(
        path.heat_lost, path.sensible_heat + path.exsolution_heat, rtol=1e-12, atol=1e-6
    )
    ratio = np.asarray(path.heat_capacity_ratio)
    assert np.all(np.abs(ratio[t > onset]) < 1e-12) and np.all(ratio[below] > 0)


def test_interior_exsolution_heat_matches_the_integral_of_the_partial_enthalpy():
    from atmodeller.cooling import InteriorCooling

    model = SubregularSolution.hydrogen_silicate_gilmore26()
    interior = InteriorCooling.from_species(mixture(), model, P, mass_fraction=0.03)
    path = interior.path(3400.0, 2900.0, steps=2000)
    t = np.asarray(path.temperature)
    x = np.asarray(path.composition)
    h = np.asarray(model.partial_enthalpies(t, P, x)[..., 1])
    m_si, m_h = (float(v) for v in interior.parcel.molar_masses)
    # Moles of H2 per kg of system: silicate fixed, dissolved H2 follows x.
    silicate_moles = (1 - 0.03) / m_si
    dissolved = silicate_moles * x / (1 - x)
    expected = np.concatenate(([0.0], np.cumsum(-np.diff(dissolved) * (h[1:] + h[:-1]) / 2)))
    np.testing.assert_allclose(path.exsolution_heat, expected, rtol=1e-4, atol=1.0)
    exsolved = (dissolved[0] - dissolved) * m_h
    np.testing.assert_allclose(path.exsolved_hydrogen, exsolved, rtol=1e-12, atol=1e-15)


def test_adiabat_temperature_factor():
    from atmodeller.cooling import adiabat_temperature_factor

    assert float(adiabat_temperature_factor(1e4, 1e4, 3e-6)) == pytest.approx(1.0)
    # Mass uniform in pressure: mean of exp(g dP) over the column.
    gradient, top, bottom = 3e-6, 2e4, 1.2e5
    expected = (np.exp(gradient * (bottom - top)) - 1) / (gradient * (bottom - top))
    assert float(adiabat_temperature_factor(top, bottom, gradient)) == pytest.approx(expected)


def test_onset_is_nan_for_a_missing_gradient_coefficient(parcel):
    model, x = parcel.mixing_model, parcel.composition
    for kappa in (model.gradient_coefficient(6000.0, P, 0.3e-9), -1e-16):
        assert np.isnan(nucleation_onset_temperature(model, P, x, kappa, 1.819e-5, 123.19))


@pytest.mark.parametrize("mass_fraction,width", [(0.03, 0.06e-9), (0.006, 0.3e-9)])
def test_onset_meets_its_criterion_or_is_nan(mass_fraction, width):
    model = SubregularSolution.hydrogen_silicate_gilmore26()
    sample = ParcelCooling.from_species(mixture(), model, P, mass_fraction=mass_fraction)
    x = sample.composition
    kappa = model.gradient_coefficient(3000.0, P, width)
    onsets = []
    for criterion in (80.0, 160.0):
        onset = nucleation_onset_temperature(model, P, x, kappa, 1.819e-5, criterion)
        onsets.append(float(onset))
        if np.isfinite(onset):
            b = model.nucleation_barrier(onset, P, x, kappa, 1.819e-5)
            assert float(b.energy_over_kT) == pytest.approx(criterion, rel=1e-6)
    # A larger criterion allows a larger barrier, so nucleation starts earlier.
    if all(np.isfinite(onsets)):
        assert onsets[1] > onsets[0]
    # The 0.6 wt% parcel has no spinodal; the scan still brackets its onset.
    if mass_fraction == 0.006:
        assert all(np.isfinite(onsets))


def test_path_is_invalid_when_the_parcel_meets_a_gap_without_a_binodal():
    model = SubregularSolution(-50000.0, -50000.0, temperature_scale=1000.0)
    thermo = (mixture()[0].thermo, mixture()[1].thermo)
    sample = ParcelCooling(model, thermo, jnp.array([0.1003887, 0.00201588]), 1.0, 0.5)
    assert np.isnan(model.binodal_temperature(1.0, 0.5))
    assert not bool(sample.path(3000.0, 2000.0, steps=10).valid)


def test_paths_reject_batched_parcels():
    model = SubregularSolution.hydrogen_silicate_gilmore26()
    batch = ParcelCooling.from_species(mixture(), model, P, mass_fraction=jnp.array([0.01, 0.03]))
    with pytest.raises(ValueError, match="vmap"):
        batch.path(3600.0, 2600.0, steps=10)


def test_interior_richer_than_the_critical_composition_is_invalid():
    from atmodeller.cooling import InteriorCooling

    model = SubregularSolution.hydrogen_silicate_gilmore26()
    interior = InteriorCooling.from_species(mixture(), model, P, mass_fraction=0.20)
    assert not bool(interior.path(3600.0, 2800.0, steps=50).valid)


def test_interior_started_inside_the_gap_counts_the_immediate_exsolution():
    from atmodeller.cooling import InteriorCooling

    model = SubregularSolution.hydrogen_silicate_gilmore26()
    interior = InteriorCooling.from_species(mixture(), model, P, mass_fraction=0.03)
    onset = float(model.binodal_temperature(P, interior.initial_composition))
    inside = interior.path(onset - 50, 2800.0, steps=50)
    outside = interior.path(onset + 50, 2800.0, steps=50)
    assert float(inside.exsolution_heat[0]) > 0 and float(inside.exsolved_hydrogen[0]) > 0
    # The mixing enthalpy does not depend on temperature, so both reach the same heat.
    np.testing.assert_allclose(inside.exsolution_heat[-1], outside.exsolution_heat[-1], rtol=1e-9)
    np.testing.assert_allclose(
        inside.exsolved_hydrogen[-1], outside.exsolved_hydrogen[-1], rtol=1e-12
    )
