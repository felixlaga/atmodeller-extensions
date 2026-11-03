# SPDX-License-Identifier: Apache-2.0

"""Cooling paths of a closed H2-MgSiO3 parcel: energy, separation onset and recalescence."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from atmodeller import ChemicalSpecies, EquilibriumModel, SolverParameters, SubregularSolution
from atmodeller.cooling import ParcelCooling, solve_path_with_engine
from atmodeller.thermodata import default_properties

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
