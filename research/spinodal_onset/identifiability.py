# SPDX-License-Identifier: Apache-2.0
"""Construct a binodal-preserving counterexample to inferred spinodal uniqueness.

This is an identifiability experiment, not a recommended material model.
The added nonnegative C2 bump is zero with zero first/second derivatives
at both binodal endpoints, and zero everywhere outside the gap. Its amplitude
is tied to the baseline barrier at the gap midpoint, vanishing at criticality.
It preserves all equilibrium free energies/compositions and all homogeneous
states outside the gap, while changing the interior homogeneous curvature.

The bump vanishes at T_b(x) and grows below it, so it adds a negative heat
capacity -T d2(bump)/dT2 at fixed x and P; no nonnegative bump of this kind
avoids that. The amplitude is therefore capped so that every homogeneous state
inside the gap, at every composition and between 2,000 K and T_C, keeps a
positive total heat capacity.
"""

import hashlib
import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy.constants import gas_constant
from scipy.optimize import brentq, minimize

from atmodeller import SubregularSolution
from research.spinodal_onset.run_study import OUT, critical, onset_zero_pressure, to_x

MODEL = SubregularSolution(75000, 424000, temperature_scale=4400, pressure_scale=-350000)
# Baseline heat capacity of the homogeneous mixture. The model has zero excess
# heat capacity, so this is the endmember value. 3.2 kB per atom is the smallest
# of the single-phase ED Table 1 estimates (dH/dT with the V dP term removed:
# 3.2-3.9 kB per atom), so the cap holds for any larger true value.
CP_BASE_KB_PER_ATOM = 3.2
FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)  # of the largest admissible amplitude


@eqx.filter_jit
def shape(temperature, pressure, z):
    """Return fit-basis homogeneous energy, curvature and bump coefficients."""
    c = MODEL.coexistence(temperature, pressure)
    a, b = c.lower, c.upper
    width = jnp.where(c.has_gap, b - a, 1.0)
    mid = (a + b) / 2
    barrier = MODEL.gibbs_energy(temperature, pressure, mid) - MODEL.equilibrium_gibbs_energy(
        temperature, pressure, mid
    )
    u = jnp.clip((z - a) / width, 0.0, 1.0)
    active = c.contains(z)
    bump = jnp.where(active, barrier * 64 * u**3 * (1 - u) ** 3, 0.0)
    second = 384 * u * (1 - u) * (1 - 5 * u + 5 * u**2) / width**2
    curvature_bump = jnp.where(active, barrier * second, 0.0)
    return (
        MODEL.gibbs_energy(temperature, pressure, z),
        MODEL.curvature(temperature, pressure, z),
        bump,
        curvature_bump,
        barrier,
    )


def bump_heat_capacity(temperature, pressure, z):
    """Heat capacity -T d2(bump)/dT2 per unit amplitude (J per fit mol per K)."""
    second = jax.grad(jax.grad(lambda t: shape(t, pressure, z)[2]))
    return -temperature * second(temperature)


def baseline_heat_capacity(z):
    """CP_BASE_KB_PER_ATOM per fit mole: H2 has 2 atoms, MgO/SiO2 on average 2.5."""
    return CP_BASE_KB_PER_ATOM * gas_constant * (2 * z + 2.5 * (1 - z))


def admissible_amplitude(pressure, t_low, t_high):
    """Largest amplitude keeping Cp_base + amplitude*Cp_bump >= 0 at every gap state.

    One amplitude for the whole free-energy surface. An amplitude fitted to each
    composition would leave negative heat capacities in metastable states of
    other compositions near T_C, where the bump's heat capacity is most negative.
    """
    zs = np.linspace(0.005, 0.995, 199)
    ts = np.linspace(t_low, t_high, 1200)
    cp = jax.jit(bump_heat_capacity)
    grid = jax.jit(jax.vmap(jax.vmap(bump_heat_capacity, (0, None, None)), (None, None, 0)))
    ratio = np.asarray(grid(jnp.asarray(ts), pressure, jnp.asarray(zs)))
    ratio = ratio / baseline_heat_capacity(zs)[:, None]
    i, j = np.unravel_index(np.argmin(ratio), ratio.shape)
    point = minimize(
        lambda v: float(cp(v[1], pressure, v[0])) / baseline_heat_capacity(v[0]),
        [zs[i], ts[j]],
        method="Nelder-Mead",
        options=dict(xatol=1e-7, fatol=1e-12),
    )
    worst = min(point.fun, ratio[i, j])
    z, t = point.x if point.fun <= ratio[i, j] else (zs[i], ts[j])
    return -1 / worst, dict(fit_z=float(z), temperature_K=float(t), cp_ratio=float(worst))


def main():
    native = SubregularSolution.hydrogen_silicate_gilmore26()
    tc = critical(MODEL)[1] * (1 + 20000.0 / float(MODEL.pressure_scale))
    lam_max, worst = admissible_amplitude(20000.0, 2000.0, tc)
    amplitudes = [f * lam_max for f in FRACTIONS]
    # Autodiff heat capacity of the bump against a central difference in T.
    zw, tw = worst["fit_z"], worst["temperature_K"]

    def energy_bump(t):
        return float(shape(t, 20000.0, zw)[2])

    fd = -tw * (energy_bump(tw + 0.5) - 2 * energy_bump(tw) + energy_bump(tw - 0.5)) / 0.25
    heat_capacity_error = abs(fd / float(bump_heat_capacity(tw, 20000.0, zw)) - 1)
    assert heat_capacity_error < 1e-3
    rows = []
    for weight in (0.01, 0.02, 0.03, 0.05):
        x = float(to_x(weight))
        z = float(native.mixing_composition(x))
        tb = float(onset_zero_pressure(native, jnp.array([x]))[0]) * (1 - 2 / 35)
        temperatures = np.linspace(tb - 1e-5, 2000, 1001)
        _, curv, _, extra, _ = shape(temperatures, 20000.0, z)
        for amplitude in amplitudes:
            values = np.asarray(curv + amplitude * extra)
            crossings = np.flatnonzero((values[:-1] >= 0) & (values[1:] < 0))
            ts, h = np.nan, np.nan
            if len(crossings):
                i = crossings[0]

                def objective(t):
                    _, base, _, delta, _ = shape(t, 20000.0, z)
                    return float(base + amplitude * delta)

                ts = brentq(objective, temperatures[i + 1], temperatures[i], xtol=1e-7)
                c = native.coexistence(ts, 20000.0)
                h = float(c.upper_fraction(x) * c.upper / x)
            rows.append(
                dict(
                    hydrogen_wt_percent=weight * 100,
                    bump_amplitude=amplitude,
                    pressure_GPa=2.0,
                    binodal_temperature_K=tb,
                    spinodal_temperature_K=ts,
                    metastable_interval_K=tb - ts,
                    H_in_rich_phase_at_spinodal=h,
                    positive_to_negative_crossings_in_2000K_window=len(crossings),
                )
            )
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "identifiability.csv", index=False)

    # A deliberately off-binodal grid checks global supporting-tangent margins.
    # Positivity of the added term proves preservation everywhere, not only here.
    margins, endpoint_changes = [], []
    plot_rows = []
    for t in (2500.0, 3000.0, 3400.0, 4000.0):
        z = np.linspace(0, 1, 1001)
        g, curvature, bump, _, _ = shape(t, 20000.0, z)
        equilibrium = np.asarray(MODEL.equilibrium_gibbs_energy(t, 20000.0, z))
        assert np.min(bump) >= 0
        for amplitude in amplitudes:
            margins.append(float(np.min(g + amplitude * bump - equilibrium)))
            for i, zi in enumerate(z):
                plot_rows.append(
                    dict(
                        temperature_K=t,
                        fit_z=zi,
                        bump_amplitude=amplitude,
                        energy_above_equilibrium_J_per_fit_mol=float(
                            g[i] + amplitude * bump[i] - equilibrium[i]
                        ),
                    )
                )
        c = MODEL.coexistence(t, 20000.0)
        if c.has_gap:
            _, _, endpoint_bump, endpoint_second, _ = shape(
                t, 20000.0, jnp.array([c.lower, c.upper])
            )
            endpoint_changes.extend(np.asarray(endpoint_bump).tolist())
            endpoint_changes.extend(np.asarray(endpoint_second).tolist())
    assert min(margins) > -1e-7
    assert max(np.abs(endpoint_changes)) < 1e-8
    assert np.asarray(shape(4000.0, 50000.0, np.linspace(0, 1, 51))[2] == 0).all()
    derivative_errors = []
    for t, z in ((2500.0, 0.4), (3000.0, 0.6), (3400.0, 0.7), (4000.0, 0.5)):

        def perturbed_energy(composition):
            energy, _, bump, _, _ = shape(t, 20000.0, composition)
            return energy + 4 * bump

        autodiff = float(jax.grad(jax.grad(perturbed_energy))(z))
        _, base_second, _, bump_second, _ = shape(t, 20000.0, z)
        analytic = float(base_second + 4 * bump_second)
        derivative_errors.append(abs(autodiff - analytic) / max(abs(analytic), 1.0))
    assert max(derivative_errors) < 1e-10
    baseline = df.loc[(df.hydrogen_wt_percent == 3) & (df.bump_amplitude == 0)].iloc[0]
    np.testing.assert_allclose(
        baseline.spinodal_temperature_K,
        native.spinodal_temperature(20000.0, to_x(0.03)),
        atol=1e-6,
    )
    pd.DataFrame(plot_rows).to_csv(OUT / "identifiability_energy.csv", index=False)
    Path(OUT / "identifiability_checks.json").write_text(
        json.dumps(
            dict(
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                minimum_sampled_support_margin_J_per_fit_mol=min(margins),
                maximum_endpoint_energy_or_curvature_change=max(np.abs(endpoint_changes)),
                homogeneous_EDTable2_states_unchanged=True,
                maximum_relative_curvature_error_against_autodiff=max(derivative_errors),
                heat_capacity_baseline_kB_per_atom=CP_BASE_KB_PER_ATOM,
                admissible_amplitude=lam_max,
                most_negative_bump_heat_capacity_kB_per_atom_per_unit_amplitude=worst["cp_ratio"]
                * CP_BASE_KB_PER_ATOM,
                most_negative_bump_heat_capacity_at=dict(
                    fit_z=worst["fit_z"], temperature_K=worst["temperature_K"], critical_K=tc
                ),
                bump_heat_capacity_relative_error_against_finite_difference=heat_capacity_error,
                scope="Constructed thermodynamic counterexample; amplitudes are not calibrated material uncertainties. They are capped so that every homogeneous state inside the gap keeps a positive heat capacity.",
            ),
            indent=2,
        )
        + "\n"
    )
    print(df.loc[df.hydrogen_wt_percent == 3].to_string(index=False))


if __name__ == "__main__":
    main()
