#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Heat and kinetics of hydrogen exsolution, run with Atmodeller's energy and cooling layers.

Stages (each one process, single-threaded, a few minutes):

    energy     H2 partial molar enthalpy (MJ per kg H) over the refit posterior
    edtable2   model excess enthalpies against Extended Data Table 2
    interior   convecting-interior paths: exsolution heat against sensible heat
    barriers   dimensionless square-gradient barrier tables (use --start/--stop)
    delays     nucleation delays from the tables, for DFT-anchored interfaces
    paths      main-case parcel and interior paths, and the solver replay check
    checks     sensitivity checks quoted in the text: interior end state, sensible
               heat basis, interval length, rising top pressure, and the interface
               width that would give half the binodal-spinodal window
    planet     exsolution heat along the digitized fiducial evolution of Rogers et al.
               (2025) against its energy budget, and the 2026 critical temperature

Usage: uv run python research/spinodal_onset/heat_study.py STAGE [--start N --stop M]
"""

import argparse
import csv
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from atmodeller import ChemicalSpecies, EquilibriumModel, SolverParameters, SubregularSolution
from atmodeller.cooling import (
    InteriorCooling,
    ParcelCooling,
    nucleation_onset_temperature,
    solve_path_with_engine,
)
from atmodeller.sci_utils import GAS_CONSTANT as R
from atmodeller.thermodata.defaults import default_properties

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs" / "heat"
SAMPLES = HERE / "outputs" / "calibration_samples.csv"
M_H2, M_MGSIO3 = 2.01588e-3, 100.3887e-3
"""kg/mol"""
AVOGADRO = 6.02214076e23
VERSIONS = ("table", "figure", "inflated", "no_volume", "phase_hom", "phase_comp")
PER_VERSION = 400
"""Thinned samples per M1 version for path stages"""
KINETIC_SAMPLES = 400
CRITERIA = (69.08, 123.19)
"""ln(J0 V t): J0 1e30 m^-3 s^-1 over 1 m³ s, to 1e40 over 1 m³ Myr"""
MOLAR_VOLUMES = (18.19e-6 * 0.7, 18.19e-6, 18.19e-6 * 1.3)
"""m³ per mixing-basis mole, scanned +/-30%: the cell of the ED Table 1 two-phase run at
2.05 GPa and 3000 K, 258 atoms of 13.52 Å³ over 115.5 mixing units"""
INTERFACES = (
    # (label, width m, T K, P bar): states where a DFT profile was measured.
    ("fig1c_0.28nm", 0.28e-9, 2500.0, 1e5),
    ("fig1c_0.33nm", 0.33e-9, 2500.0, 1e5),
    ("young_0.5nm", 0.50e-9, 3000.0, 2.5e4),
)


def mixture():
    return (
        ChemicalSpecies.create_solution("MgSiO3"),
        ChemicalSpecies.create_solution("H2", thermo=default_properties["H2_g"]),
    )


def native_x(w):
    n = w / M_H2
    return n / (n + (1 - w) / M_MGSIO3)


def model_from(a_kj, b_kj, inv_tau, inv_pi_gpa):
    """Refit row to the core model; pressures in bar (1 GPa = 1e4 bar)."""
    return SubregularSolution(
        a_kj * 1e3,
        b_kj * 1e3,
        temperature_scale=1 / inv_tau,
        pressure_scale=1e4 / inv_pi_gpa,
        component_0_multiplicity=2.0,
    )


PUBLISHED = (75.0, 424.0, 1 / 4400, -1 / 35)


def load_samples():
    rows = list(csv.DictReader(open(SAMPLES)))
    groups = {}
    for r in rows:
        if r["model"] != "M1":
            continue
        groups.setdefault(r["version"], []).append(
            [float(r[k]) for k in ("A_kJ", "B_kJ", "inv_tau_per_K", "inv_pi_per_GPa", "log_post")]
        )
    return {k: np.array(v) for k, v in groups.items()}


def thin(values, n):
    index = np.linspace(0, len(values) - 1, n).round().astype(int)
    return values[index]


def quantiles(values):
    finite = np.asarray(values)[np.isfinite(values)]
    if finite.size == 0:
        return [np.nan] * 3 + [0]
    return [*np.percentile(finite, [2.5, 50, 97.5]), finite.size]


def write(name, header, rows):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / name, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print("wrote", OUT / name)


def stage_energy():
    groups = load_samples()
    rows = []
    sets = {"published": np.array([PUBLISHED])} | {f"M1_{v}": groups[v] for v in VERSIONS}
    for label, params in sets.items():
        model = model_from(*(jnp.asarray(params[:, i]) for i in range(4)))
        for w in (0.01, 0.02, 0.03):
            for p_gpa in (2.0, 5.0):
                h = model.partial_enthalpies(3000.0, p_gpa * 1e4, native_x(w))[..., 1]
                rows.append([label, w, p_gpa, *quantiles(np.asarray(h) / M_H2 / 1e6)])
    write(
        "hydrogen_partial_enthalpy.csv",
        ["set", "w_H2", "P_GPa", "MJ_per_kgH_q025", "q50", "q975", "n"],
        rows,
    )


def stage_edtable2():
    # The excess enthalpies the refits were fitted to: sources/EDTable2.xlsx, with the
    # endmembers moved from 5.00 GPa to the mixture pressure (Table B.2).
    from research.spinodal_onset.calibration import homogeneous_data

    states, values, covariance = homogeneous_data()
    sigma = np.sqrt(np.diag(covariance))
    order = np.argsort([s["z"] for s in states])
    z = np.array([states[i]["z"] for i in order])
    p = np.array([states[i]["p"] for i in order])
    t = np.array([states[i]["t"] for i in order])
    excess, error = values[order] / 1e3, sigma[order] / 1e3
    groups = load_samples()
    rows = []
    sets = {"published": np.array([PUBLISHED])} | {f"M1_{v}": groups[v] for v in VERSIONS}
    for label, params in sets.items():
        model = model_from(*(jnp.asarray(params[:, i, None]) for i in range(4)))
        x = 2 * z / (1 + z)
        # Native enthalpy per native mole over (2 - x) native-to-oxide units.
        hm = model.enthalpy(t, p * 1e4, x) / (2 - x) / 1e3
        for i in range(len(z)):
            rows.append([label, z[i], p[i], excess[i], error[i], *quantiles(np.asarray(hm)[:, i])])
    write(
        "edtable2_excess_enthalpy.csv",
        [
            "set",
            "z_ox",
            "P_GPa",
            "data_kJ_per_mol",
            "data_sigma_kJ_per_mol",
            "model_q025",
            "model_q50",
            "model_q975",
            "n",
        ],
        rows,
    )


def interior_summary(params, w, p_bar, below=300.0, steps=301):
    def one(row):
        model = model_from(*row)
        interior = InteriorCooling.from_species(mixture(), model, p_bar, mass_fraction=w)
        onset = model.binodal_temperature(p_bar, interior.initial_composition)
        path = interior.path(onset + 1e-6, onset - below, steps=steps)
        exsolved = path.exsolved_hydrogen[-1]
        return jnp.stack(
            (
                onset,
                exsolved / w,
                path.exsolution_heat[-1] / jnp.where(exsolved > 0, exsolved, jnp.nan) / 1e6,
                path.exsolution_heat[-1] / path.sensible_heat[-1],
                path.exsolution_heat[-1] / 1e6,
                jnp.max(path.heat_capacity_ratio),
                path.valid.astype(float),
            )
        )

    return np.asarray(jax.jit(jax.vmap(one))(jnp.asarray(params[:, :4])))


def stage_interior():
    groups = load_samples()
    sets = {"published": np.array([PUBLISHED + (0.0,)])} | {
        f"M1_{v}": thin(groups[v], PER_VERSION) for v in VERSIONS
    }
    rows = []
    names = [
        "onset_K",
        "exsolved_fraction",
        "MJ_per_kgH",
        "exsolution_over_sensible",
        "exsolution_MJ_per_kg",
        "max_heat_capacity_ratio",
    ]
    for label, params in sets.items():
        for w in (0.01, 0.03):
            for p_gpa in (2.0, 5.0):
                start = time.time()
                values = interior_summary(params, w, p_gpa * 1e4)
                ok = values[:, -1] > 0
                for j, name in enumerate(names):
                    rows.append([label, w, p_gpa, name, *quantiles(values[ok, j])])
                print(label, w, p_gpa, f"{time.time() - start:.1f}s", int(ok.sum()))
    write(
        "interior_300K.csv",
        ["set", "w_H2", "P_GPa", "quantity", "q025", "q50", "q975", "n"],
        rows,
    )


def barrier_grid(row, x, p_bar, below):
    """Dimensionless barriers, int(dg + z'^2) dV, at T_binodal - below."""
    model = model_from(*row)
    onset = model.binodal_temperature(p_bar, x)
    t = onset - below
    rt = R * t
    b = model.nucleation_barrier(t, p_bar, x, rt, 1.0)
    kappas = jnp.stack([model.gradient_coefficient(tr, pr, wd) for _, wd, tr, pr in INTERFACES])
    return onset, model.spinodal_temperature(p_bar, x), b.energy / rt, kappas


BELOW = np.geomspace(0.2, 80.0, 18)


def stage_barriers(start, stop):
    groups = load_samples()
    params = thin(groups["table"], KINETIC_SAMPLES)
    best = groups["table"][np.argmax(groups["table"][:, 4])]
    params = np.vstack([np.array(PUBLISHED + (0.0,)), best, params])[start:stop]
    x, p_bar = native_x(0.03), 2e4
    run = jax.jit(jax.vmap(lambda row: barrier_grid(row, x, p_bar, jnp.asarray(BELOW))))
    t0 = time.time()
    onset, spinodal, table, kappas = (np.asarray(v) for v in run(jnp.asarray(params[:, :4])))
    print(f"{stop - start} samples in {time.time() - t0:.1f}s")
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT / f"barriers_{start:04d}_{stop:04d}.npz",
        params=params,
        onset=onset,
        spinodal=spinodal,
        table=table,
        kappas=kappas,
        below=BELOW,
    )


def stage_delays():
    parts = sorted(OUT.glob("barriers_*.npz"))
    data = [np.load(p) for p in parts]
    params = np.vstack([d["params"] for d in data])
    onset = np.concatenate([d["onset"] for d in data])
    spinodal = np.concatenate([d["spinodal"] for d in data])
    table = np.vstack([d["table"] for d in data])
    kappas = np.vstack([d["kappas"] for d in data])
    x, p_bar = native_x(0.03), 2e4
    rows = []
    results = {}
    for i, (label, *_rest) in enumerate(INTERFACES):
        for v in MOLAR_VOLUMES:
            for criterion in CRITERIA:
                delays = np.full(len(params), np.nan)
                for s in range(len(params)):
                    t = onset[s] - BELOW
                    good = np.isfinite(table[s]) & (table[s] > 0)
                    if not np.isfinite(kappas[s, i]) or good.sum() < 3:
                        continue
                    over_kt = AVOGADRO * (kappas[s, i] / (R * t)) ** 1.5 * table[s] / v
                    lo = np.log(over_kt[good])
                    # The barrier falls monotonically with undercooling.
                    if not (lo[0] > np.log(criterion) > lo[-1]):
                        continue
                    delays[s] = np.exp(np.interp(-np.log(criterion), -lo, np.log(BELOW[good])))
                window = onset - spinodal
                key = (label, v, criterion)
                results[key] = delays
                rows.append([label, v * 1e6, criterion, "published", delays[0], window[0]])
                rows.append([label, v * 1e6, criterion, "M1_MAP", delays[1], window[1]])
                rows.append(
                    [
                        label,
                        v * 1e6,
                        criterion,
                        "M1_table_q025_q50_q975",
                        *np.round(quantiles(delays[2:])[:3], 3),
                    ]
                )
    write(
        "nucleation_delays.csv",
        [
            "interface",
            "Vm_cm3_per_mol",
            "criterion",
            "set",
            "delay_K_or_q025",
            "window_K_or_q50",
            "q975",
        ],
        rows,
    )
    # Hydrogen withheld at the nucleation onset, for the published fit and M1 MAP.
    withheld = []
    for s, label in ((0, "published"), (1, "M1_MAP")):
        model = model_from(*params[s, :4])
        for key, delays in results.items():
            t_on = onset[s] - delays[s]
            if not np.isfinite(t_on):
                continue
            c = model.coexistence(t_on, p_bar)
            f = float(c.upper_fraction(x))
            withheld.append(
                [label, *key[:1], key[1] * 1e6, key[2], delays[s], f * float(c.upper) / x]
            )
    write(
        "nucleation_withheld.csv",
        ["set", "interface", "Vm_cm3_per_mol", "criterion", "delay_K", "H_fraction_withheld"],
        withheld,
    )


def stage_paths():
    groups = load_samples()
    best = groups["table"][np.argmax(groups["table"][:, 4])]
    summary = {}
    rows = []
    for label, row in (("published", PUBLISHED), ("M1_MAP", tuple(best[:4]))):
        model = model_from(*row)
        parcel = ParcelCooling.from_species(mixture(), model, 2e4, mass_fraction=0.03)
        x = parcel.composition
        kappa = model.gradient_coefficient(2500.0, 1e5, 0.30e-9)
        onsets = {
            "equilibrium": None,
            "spinodal": model.spinodal_temperature(2e4, x),
            "nucleation_L69": nucleation_onset_temperature(model, 2e4, x, kappa, 18.19e-6, 69.08),
            "nucleation_L123": nucleation_onset_temperature(
                model, 2e4, x, kappa, 18.19e-6, 123.19
            ),
        }
        for name, onset in onsets.items():
            for separation in ("isenthalpic", "isothermal"):
                if name == "equilibrium" and separation == "isothermal":
                    continue
                path = parcel.path(3700.0, 2700.0, onset, separation=separation, steps=150)
                summary[f"{label}/{name}/{separation}"] = {
                    "binodal_K": float(path.binodal_temperature),
                    "onset_K": float(path.onset_temperature),
                    "separation_K": float(path.separation_temperature),
                    "released_MJ_per_kg": float(path.released_heat) / 1e6,
                    "final_heat_lost_MJ_per_kg": float(path.heat_lost[-1]) / 1e6,
                    "valid": bool(path.valid),
                }
                for i in range(path.temperature.shape[-1]):
                    if np.isfinite(path.temperature[i]):
                        rows.append(
                            [
                                label,
                                name,
                                separation,
                                float(path.heat_lost[i]) / 1e6,
                                float(path.temperature[i]),
                                bool(path.separated[i]),
                                float(path.upper_fraction[i]),
                            ]
                        )
        for p_gpa in (2.0, 5.0):
            interior = InteriorCooling.from_species(
                mixture(), model, p_gpa * 1e4, mass_fraction=0.03
            )
            path = interior.path(3700.0, 2700.0, steps=400)
            for i in range(400):
                rows.append(
                    [
                        label,
                        f"interior_{p_gpa:.0f}GPa",
                        "open",
                        float(path.heat_lost[i]) / 1e6,
                        float(path.temperature[i]),
                        float(path.heat_capacity_ratio[i]),
                        float(path.exsolved_hydrogen[i]) / 0.03,
                    ]
                )
    write(
        "paths.csv",
        [
            "set",
            "path",
            "separation",
            "heat_lost_MJ_per_kg",
            "T_K",
            "separated_or_ratio",
            "upper_fraction_or_exsolved",
        ],
        rows,
    )
    # Replay the published spinodal path with the full equilibrium solver.
    steps = 40
    model = model_from(*PUBLISHED)
    parcel = ParcelCooling.from_species(mixture(), model, 2e4, mass_fraction=0.03)
    engine_model = EquilibriumModel.from_species(
        [],
        temperature=jnp.full(steps, 3000.0),
        pressure=2e4,
        mass=0.0,
        silicate_melt_species=mixture(),
        mixing_models={"silicate_melt": model},
        phase_equilibrium="spinodal",
        mass_constraints={"H2": 0.03, "MgSiO3": 0.97},
        solver_parameters=SolverParameters(atol=1e-12, rtol=1e-12),
    )
    path = parcel.path(
        3700.0, 2700.0, model.spinodal_temperature(2e4, parcel.composition), steps=steps
    )
    engine = solve_path_with_engine(
        path, engine_model.with_phase_equilibrium("homogeneous"), engine_model, "silicate_melt"
    )
    difference = np.abs(engine["mixing_enthalpy"] / np.asarray(path.mixing_enthalpy) - 1)
    summary["engine_replay"] = {
        "rows": 2 * steps,
        "all_success": bool(np.all(engine["success"])),
        "max_relative_difference": float(np.nanmax(difference)),
    }
    (OUT / "paths_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


TOP_PRESSURE_RISE = 4.0e4 / 925.0
"""bar/K: the binodal pressure of Rogers et al. (2025) rises from 0.9 to 4.9 GPa while its
temperature falls by 925 K (their fiducial evolution, digitized)"""


def interior_checks(row, w=0.03, p_bar=2e4, below=300.0):
    """Sensitivity of the interior result to its assumptions, for one parameter row.

    Each entry is (value, valid). End states of the exsolved hydrogen: pure H2 (the
    model), the coexisting hydrogen-rich phase leaving the melt (``phase``), and H2
    joining an envelope already at the coexisting composition (``net``).
    """
    model = model_from(*row)
    interior = InteriorCooling.from_species(mixture(), model, p_bar, mass_fraction=w)
    parcel = interior.parcel
    x0 = interior.initial_composition
    m0, m1 = parcel.molar_masses[0], parcel.molar_masses[1]
    n_s = (1 - x0) / ((1 - x0) * m0 + x0 * m1)  # native moles of silicate per kg
    onset = model.binodal_temperature(p_bar, x0)
    path = interior.path(onset + 1e-6, onset - below, steps=int(below) + 1)
    t, y, gamma = path.temperature, path.composition, path.heat_capacity_ratio
    sensible = path.sensible_heat[-1]
    ok = path.valid

    def mean(a):
        return 0.5 * (a[1:] + a[:-1])

    # Sensible heat of the exsolved hydrogen after it has left, cooled with the top.
    h1 = parcel.thermo[1].enthalpy(t) * parcel.heat_capacity_scale / m1
    exsolved_sensible = jnp.sum(mean(path.exsolved_hydrogen) * (h1[:-1] - h1[1:]))

    c = model.coexistence(t, p_bar)
    upper = c.upper
    h_lower = model.partial_enthalpies(t, p_bar, y)[..., 1]
    h_upper = model.partial_enthalpies(t, p_bar, upper)[..., 1]
    phase = model.exsolution_enthalpy(t, p_bar)  # J per native mole of upper phase removed
    # Pure H2 leaves: the silicate stays, so dn(H2) = n_s d[y/(1-y)].
    dn = n_s * (y[:-1] / (1 - y[:-1]) - y[1:] / (1 - y[1:]))
    pure = jnp.sum(dn * mean(h_lower))
    net = jnp.sum(dn * mean(h_lower - h_upper))
    # The upper phase leaves: the melt loses silicate as well, by mass balance.
    upper_step = mean(upper)
    melt = (
        n_s
        / (1 - x0)
        * jnp.concatenate((jnp.ones(1), jnp.cumprod((upper_step - y[:-1]) / (upper_step - y[1:]))))
    )
    removed = jnp.sum((melt[:-1] - melt[1:]) * mean(phase))
    # Release per kelvin relative to pure H2, at the same melt composition change.
    per_kelvin = melt * (1 - y) / n_s * (1 - y) / (upper - y) * phase / h_lower
    per_hydrogen = phase / upper / h_lower
    net_fraction = 1 - h_upper / h_lower
    upper_wt = upper * m1 / (upper * m1 + (1 - upper) * m0)

    def ratio_over(span):
        p = interior.path(onset + 1e-6, onset - span, steps=int(span) + 1)
        return p.exsolution_heat[-1] / p.sensible_heat[-1], p.valid

    # Top pressure rising at the rate of the published evolution, from the 2 GPa binodal.
    p_rise = p_bar + TOP_PRESSURE_RISE * (t[0] - t)
    c_rise = model.coexistence(t, p_rise)
    y_rise = jnp.where(c_rise.contains(x0), c_rise.lower, x0)
    dn_rise = n_s * (y_rise[:-1] / (1 - y_rise[:-1]) - y_rise[1:] / (1 - y_rise[1:]))
    h_rise = model.partial_enthalpies(t, p_rise, y_rise)[..., 1]
    rise = jnp.sum(dn_rise * mean(h_rise))

    return {
        "exsolution_over_sensible": (path.exsolution_heat[-1] / sensible, ok),
        "max_heat_capacity_ratio": (jnp.max(gamma), ok),
        "pure_end_state_check": (pure / sensible, ok),
        "exsolution_over_interior_sensible": (
            path.exsolution_heat[-1] / (sensible - exsolved_sensible),
            ok,
        ),
        "upper_phase_H_wt_1K": (upper_wt[1], ok),
        "phase_per_hydrogen_1K": (per_hydrogen[1], ok),
        "phase_per_kelvin_1K": (per_kelvin[1], ok),
        "phase_per_kelvin_100K": (per_kelvin[100], ok),
        "phase_max_heat_capacity_ratio": (jnp.nanmax(gamma * per_kelvin), ok),
        "phase_over_sensible": (removed / sensible, ok),
        "net_fraction_1K": (net_fraction[1], ok),
        "net_max_heat_capacity_ratio": (jnp.nanmax(gamma * net_fraction), ok),
        "net_over_sensible": (net / sensible, ok),
        "exsolution_over_sensible_100K": ratio_over(100.0),
        "exsolution_over_sensible_600K": ratio_over(600.0),
        "rising_pressure_change": (rise / pure - 1, ok & jnp.all(c_rise.valid)),
    }


def half_window_widths(row, x, p_bar=2e4, reference=0.28e-9):
    """Interface widths at 10 GPa and 2500 K that put nucleation at half the window.

    Same convention as the delays and fig5_3_nucleation_delay: kappa follows from the
    width at the Fig. 1c state and is held fixed, so W*/kT scales as width³/volume.
    """
    model = model_from(*row)
    onset = float(model.binodal_temperature(p_bar, x))
    window = onset - float(model.spinodal_temperature(p_bar, x))
    t = onset - window / 2
    table = float(model.nucleation_barrier(t, p_bar, x, R * t, 1.0).energy) / (R * t)
    kappa_ref = float(model.gradient_coefficient(2500.0, 1e5, reference))
    widths = [
        reference * np.sqrt(R * t * (c * v / (AVOGADRO * table)) ** (2 / 3) / kappa_ref)
        for v in MOLAR_VOLUMES
        for c in CRITERIA
    ]
    return window, widths


def stage_checks():
    groups = load_samples()
    best = groups["table"][np.argmax(groups["table"][:, 4])]
    sets = {
        "published": np.array([PUBLISHED]),
        "M1_MAP": best[None, :4],
        "M1_table": thin(groups["table"], PER_VERSION)[:, :4],
    }
    run = jax.jit(jax.vmap(interior_checks))
    summary = {"interior_3wt_2GPa_300K": {}, "half_window": {}}
    for label, params in sets.items():
        start = time.time()
        values = run(jnp.asarray(params))
        summary["interior_3wt_2GPa_300K"][label] = {
            name: [float(q) for q in quantiles(np.asarray(v)[np.asarray(ok)])]
            for name, (v, ok) in values.items()
        }
        print(label, f"{time.time() - start:.1f}s")
    simulated = [width for label, width, *_ in INTERFACES if label.startswith("fig1c")]
    for label, row in (("published", PUBLISHED), ("M1_MAP", tuple(best[:4]))):
        window, widths = half_window_widths(row, native_x(0.03))
        summary["half_window"][label] = {
            "window_K": window,
            "width_nm": [1e9 * min(widths), 1e9 * max(widths)],
            "times_simulated": [min(widths) / max(simulated), max(widths) / min(simulated)],
        }
    # Sensible-heat basis: Atmodeller end members at 3000 K, mixed by mass.
    parcel = ParcelCooling.from_species(mixture(), model_from(*PUBLISHED), 2e4, mass_fraction=0.03)
    cp = [
        float(th.enthalpy(3000.5) - th.enthalpy(2999.5)) / m
        for th, m in zip(parcel.thermo, np.asarray(parcel.molar_masses))
    ]
    summary["sensible_basis_3000K"] = {
        "cp_silicate_kJ_per_kg_K": cp[0] / 1e3,
        "cp_H2_kJ_per_kg_K": cp[1] / 1e3,
        "H2_share_at_3wt": 0.03 * cp[1] / (0.03 * cp[1] + 0.97 * cp[0]),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "checks.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


SOURCES = HERE / "sources"
EARTH_MASS, YEAR = 5.9722e24, 3.15576e7
PLANET_MASS, PLANET_HYDROGEN = 6 * EARTH_MASS, 0.03
"""Fiducial planet of Rogers et al. (2025): 6 Earth masses, 3% hydrogen by mass"""
SPAN = (6.6989, 9.9543)
"""log10 age (yr): 5 Myr to 9 Gyr, the digitized span"""
WITH_ENTHALPY = ("table", "figure", "inflated", "no_volume", "phase_hom")


def rogers_path(n=4001):
    """Digitized fiducial evolution and energy budget of Rogers et al. (2025, Figs. 5-6)."""
    fig5 = json.loads((SOURCES / "RYS25_Fig5_miscible_digitized.json").read_text())
    fig6 = json.loads((SOURCES / "RYS25_Fig6_miscible_digitized.json").read_text())
    g = np.linspace(*SPAN, n)

    def curve(points, grid=g):
        a = np.asarray(points)
        return np.interp(grid, a[:, 0], a[:, 1])

    def energy(key):
        # Integral over the span of one Fig. 6 term (W), within its own digitized range.
        a = np.asarray(fig6[key])
        grid = np.linspace(max(SPAN[0], a[0, 0]), min(SPAN[1], a[-1, 0]), 20001)
        return np.trapezoid(10 ** curve(a, grid) * 1e-7, 10**grid * YEAR)

    budget = {
        "radiated_J": energy("L_rcb"),
        "gravitational_J": energy("minus_dU_sil") + energy("minus_dU_H2"),
        "thermal_J": energy("minus_dE_sil")
        + energy("minus_dE_H2")
        - energy("minus_dE_sil_negative_branch"),
    }
    return dict(
        log_age=g,
        seconds=10**g * YEAR,
        temperature=curve(fig5["T_binodal_K"]),
        pressure_bar=curve(fig5["P_binodal_Pa"]) / 1e5,
        interior_fraction=curve(fig5["F_interior_H"]),
        luminosity=10 ** curve(fig6["L_rcb"]) * 1e-7,
        budget=budget,
    )


def planet_heat(params, path):
    """Exsolution heat along the fiducial path for parameter rows, against its budget.

    The interior holds all silicate and the fraction F(t) of the hydrogen; its melt
    composition and the released heat follow from F(t), as in Rogers et al. The
    partial molar internal energy (enthalpy minus P times partial volume) matches the
    energy basis of their budget; the enthalpy is kept for comparison.
    """
    t, p = jnp.asarray(path["temperature"]), jnp.asarray(path["pressure_bar"])
    hydrogen = PLANET_MASS * PLANET_HYDROGEN
    silicate = PLANET_MASS - hydrogen
    n_h = path["interior_fraction"] * hydrogen / M_H2
    y = jnp.asarray(n_h / (n_h + silicate / M_MGSIO3))
    model = model_from(*(jnp.asarray(params[:, i, None]) for i in range(4)))
    enthalpy = model.partial_enthalpies(t, p, y)[..., 1]
    _, dlng_dp = jax.jvp(
        lambda q: model.log_activity_coefficients(t, q, y), (p,), (jnp.ones_like(p),)
    )
    energy = enthalpy - p * R * t * dlng_dp[..., 1]
    dn = -np.diff(n_h)

    def mean(a):
        return 0.5 * (np.asarray(a)[..., 1:] + np.asarray(a)[..., :-1])

    released = mean(energy) * dn
    q_u, q_h = released.sum(-1), (mean(enthalpy) * dn).sum(-1)
    # Sensible heat of the interior (silicate and dissolved hydrogen) along T_b(t).
    parcel = ParcelCooling.from_species(mixture(), model_from(*PUBLISHED), 2e4, mass_fraction=0.03)
    h_s, h_h = (np.asarray(th.enthalpy(t)) for th in parcel.thermo)
    sensible = np.sum(silicate / M_MGSIO3 * -np.diff(h_s) + mean(n_h) * -np.diff(h_h))
    # Extra time at unchanged luminosity, against the span.
    extra = np.sum(released / mean(path["luminosity"]), axis=-1)
    span = path["seconds"][-1] - path["seconds"][0]
    # Critical temperature along the path: the maximum spinodal temperature over
    # composition at zero pressure, scaled by 1 + P/pi (exact for this model form).
    xs = jnp.linspace(1e-4, 1 - 1e-4, 4001)
    t_c0 = np.nanmax(np.asarray(model.spinodal_temperature(0.0, xs)), axis=-1)
    t_c = t_c0[:, None] * (1 + np.asarray(path["pressure_bar"]) * params[:, 3:4] / 1e4)
    below = np.asarray(path["temperature"]) < t_c
    crosses = below.any(-1)
    first = np.argmax(below, axis=-1)
    f = path["interior_fraction"]
    budget = path["budget"]
    return {
        "Q_J": q_u,
        "MJ_per_kgH": q_u / ((f[0] - f[-1]) * hydrogen) / 1e6,
        "over_radiated": q_u / budget["radiated_J"],
        "over_radiated_enthalpy_basis": q_h / budget["radiated_J"],
        "over_gravitational": q_u / budget["gravitational_J"],
        "over_thermal": q_u / budget["thermal_J"],
        "over_interior_sensible": q_u / sensible,
        "time_extension": extra / span,
        "crosses_critical": crosses.astype(float),
        "cross_Gyr": np.where(crosses, 10 ** path["log_age"][first] / 1e9, np.nan),
        "exsolved_before_cross": np.where(crosses, (f[0] - f[first]) / (f[0] - f[-1]), np.nan),
        "T_b_minus_T_C_at_end_K": path["temperature"][-1] - t_c[:, -1],
    }


def stage_planet():
    path = rogers_path()
    groups = load_samples()
    best = groups["table"][np.argmax(groups["table"][:, 4])]
    f = path["interior_fraction"]
    summary = {
        "source": "Rogers et al. (2025) fiducial miscible planet, digitized (sources/RYS25_*)",
        "path": {
            "span_log10_yr": list(SPAN),
            "exsolved_fraction_of_H": float(f[0] - f[-1]),
            "T_b_K": [float(path["temperature"][0]), float(path["temperature"][-1])],
            "P_b_GPa": [
                float(path["pressure_bar"][0] / 1e4),
                float(path["pressure_bar"][-1] / 1e4),
            ],
        },
        "budget_J": path["budget"],
        "sets": {},
    }
    # Check the internal-energy basis: for this model form u_H2 is h_H2 at zero pressure.
    zero = dict(path, pressure_bar=np.zeros_like(path["pressure_bar"]))
    check = planet_heat(np.array([PUBLISHED]), zero)["Q_J"][0]
    published = planet_heat(np.array([PUBLISHED]), path)
    summary["internal_energy_check_relative"] = float(abs(published["Q_J"][0] / check - 1))
    sets = {
        "published": np.array([PUBLISHED]),
        "M1_MAP": best[None, :4],
        "M1_table": thin(groups["table"], PER_VERSION)[:, :4],
        "M1_with_enthalpy": np.vstack(
            [thin(groups[v], PER_VERSION)[:, :4] for v in WITH_ENTHALPY]
        ),
    }
    for label, params in sets.items():
        start = time.time()
        chunks = [planet_heat(params[i : i + 400], path) for i in range(0, len(params), 400)]
        values = {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]}
        summary["sets"][label] = {
            name: [float(q) for q in quantiles(v)] for name, v in values.items()
        }
        summary["sets"][label]["fraction_crossing"] = float(values["crosses_critical"].mean())
        print(label, f"{time.time() - start:.1f}s")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "planet.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "sets"}, indent=1))
    for label in ("published", "M1_MAP", "M1_with_enthalpy"):
        print(label, json.dumps(summary["sets"][label]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=KINETIC_SAMPLES + 2)
    args = parser.parse_args()
    if args.stage == "barriers":
        stage_barriers(args.start, args.stop)
    else:
        {
            "energy": stage_energy,
            "edtable2": stage_edtable2,
            "interior": stage_interior,
            "delays": stage_delays,
            "paths": stage_paths,
            "checks": stage_checks,
            "planet": stage_planet,
        }[args.stage]()


if __name__ == "__main__":
    main()
