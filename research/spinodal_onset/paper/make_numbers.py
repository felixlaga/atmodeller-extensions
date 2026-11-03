# SPDX-License-Identifier: Apache-2.0
"""Write every computed number quoted in the draft to paper/numbers.tex.

Run from the repository root after the analysis scripts. The text only uses
these macros, so a rerun of the analysis updates the draft consistently.
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from research.spinodal_onset.calibration import coexistence_runs, homogeneous_data
from research.spinodal_onset.mixing_models import published_gilmore26, weight_to_z

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "outputs"
SUFFIX = {
    ("M1", "table"): "MOneTable",
    ("M1", "figure"): "MOneFigure",
    ("M1", "inflated"): "MOneInflated",
    ("M1", "no_volume"): "MOneNoVolume",
    ("M1", "phase_comp"): "MOnePhaseComp",
    ("M1", "phase_hom"): "MOnePhaseHom",
    ("M2", "table"): "MTwo",
    ("M3", "table"): "MThree",
    ("M4", "table"): "MFour",
}
macros = {}


def put(name, value, fmt="{:.0f}"):
    macros[name] = value if isinstance(value, str) else fmt.format(value)


def latex_power(value):
    """1.8e-10 -> $2\\times10^{-10}$ (one significant digit)."""
    mantissa, exponent = f"{value:.0e}".split("e")
    return f"${mantissa}\\times10^{{{int(exponent)}}}$"


def signed(value, digits=3):
    return f"{value:+.{digits}f}".replace("-", "$-$").replace("+", "$+$")


def main():
    put("draftdate", "29 October 2026")
    onsets = pd.read_csv(OUT / "onsets.csv")
    for model, prefix in (
        ("gilmore26", "pub"),
        ("rogers25", "old"),
        ("wrong_basis_control", "wrong"),
    ):
        for p, tag in ((2.0, "Two"), (5.0, "Five"), (10.0, "Ten")):
            r = onsets[
                (onsets.model == model)
                & np.isclose(onsets.pressure_GPa, p)
                & np.isclose(onsets.hydrogen_wt_percent, 3.0)
            ].iloc[0]
            put(f"{prefix}Tb{tag}", r.binodal_temperature_K)
            put(f"{prefix}Tsp{tag}", r.spinodal_temperature_K)
            put(f"{prefix}Int{tag}", r.metastable_interval_K)
            put(f"{prefix}Fh{tag}", 100 * r.H_in_rich_phase_at_spinodal, "{:.1f}")
    crit = pd.read_csv(OUT / "critical_points.csv").set_index("model")
    put(
        "pubThreshold",
        crit.loc["gilmore26"].minimum_spinodal_hydrogen_wt_percent_F_positive,
        "{:.2f}",
    )
    put(
        "oldThreshold",
        crit.loc["rogers25"].minimum_spinodal_hydrogen_wt_percent_F_positive,
        "{:.2f}",
    )
    put("pubXc", crit.loc["gilmore26"].critical_z_fit, "{:.3f}")
    put("pubWc", crit.loc["gilmore26"].critical_hydrogen_wt_percent, "{:.2f}")
    put("pubTcZero", crit.loc["gilmore26"].critical_temperature_zero_pressure_K)
    m = published_gilmore26()
    for p, tag in (
        (2.0, "Two"),
        (5.0, "Five"),
        (10.0, "Ten"),
        (1.65, "OneSixFive"),
        (5.14, "FiveOneFour"),
    ):
        put(f"pubTc{tag}", m.critical_point(p)[0])
    put("tcAboveRunNearCritLow", m.critical_point(1.65)[0] - 3500)
    two, _ = coexistence_runs()
    run = next(r for r in two if r["t"] == 3500)
    from research.spinodal_onset.calibration import Problem

    put("pubMissOneSixFive", Problem.misclassification_K(m, run, True))
    put("pubTbBulkOneSixFive", m.binodal_temperature(run["bulk"], run["p"]))
    put("tcAboveRunNearCritHigh", m.critical_point(5.14)[0] - 3000)
    key = json.loads((OUT / "key_results.json").read_text())
    put("stressIntLo", key["parameter_stress_interval_range_K"][0])
    put("stressIntHi", key["parameter_stress_interval_range_K"][1])
    put("stressFhLo", 100 * key["parameter_stress_H_fraction_range"][0], "{:.1f}")
    put("stressFhHi", 100 * key["parameter_stress_H_fraction_range"][1], "{:.1f}")
    k3 = key["at_3000K_2GPa_3wt"]
    put("eqFhThreeThousand", 100 * k3["H_equilibrium_fraction"], "{:.1f}")
    put("gramsWithheld", 30 * k3["H_equilibrium_fraction"], "{:.1f}")
    put("energyLowering", k3["equilibrium_energy_lowering_J_per_native_mol"] / 1e3, "{:.2f}")
    put(
        "fhPressureSpread",
        latex_power(key["maximum_pressure_dependence_excluding_0_1_wt_percent_from_critical"]),
    )
    put(
        "fhPressureSpreadAll",
        latex_power(key["maximum_pressure_dependence_of_H_fraction_at_spinodal"]),
    )
    summary = json.loads((OUT / "summary.json").read_text())
    put("nOnsetRows", f"{summary['onset_rows']:,}")
    put("nCoolingRows", f"{summary['cooling_rows']:,}")
    put("nStressRows", f"{summary['stress_rows']}")
    put("nNativeSolves", f"{summary['native_solver']['number_solves']}")
    validation = json.loads((HERE.parent / "validation/summary.json").read_text())
    put("nTests", f"{validation['full_suite']['passed']}")

    # Independent solver vs Atmodeller over the full published-model grid; the
    # near-critical binodals (within 0.02 of x_C) are reported apart as a check.
    g = onsets[onsets.model == "gilmore26"]
    x_c = crit.loc["gilmore26"].critical_z_fit
    diffs, near = [], []
    for _, r in g.iterrows():
        z = weight_to_z(r.hydrogen_wt_percent / 100)
        if math.isfinite(r.spinodal_temperature_K):
            diffs.append(abs(m.spinodal_temperature(z, r.pressure_GPa) - r.spinodal_temperature_K))
        if math.isfinite(r.binodal_temperature_K):
            d = abs(m.binodal_temperature(z, r.pressure_GPa) - r.binodal_temperature_K)
            (near if abs(z - x_c) < 0.02 else diffs).append(d)
    exponent = math.ceil(math.log10(max(diffs)))
    put("indepMaxDiff", f"$10^{{{exponent}}}$")
    put("indepMaxDiffCrit", latex_power(max(near)))
    put("indepRows", f"{len(g):,}")

    dig = json.loads((OUT / "source_figure_digitized.json").read_text())
    put("digRms", max(c["rms_residual_K"] for c in dig["curves"].values()), "{:.1f}")
    put("digMax", max(c["max_abs_residual_K"] for c in dig["curves"].values()))
    put("digMarker", dig["wide_gap_marker_max_abs_difference_from_table"], "{:.4f}")
    for series, t, tag in (("red", 3500, "OneSixFive"), ("grey", 3000, "FiveOneFour")):
        pair = sorted(
            (
                mk
                for mk in dig["markers"]
                if mk["series"] == series and round(mk["temperature_K"], -2) == t
            ),
            key=lambda mk: mk["x"],
        )
        for side, mk in zip(("Lo", "Hi"), pair):
            put(f"fig{side}{tag}", mk["x"], "{:.3f}")
            put(f"diff{side}{tag}", signed(mk["figure_minus_table"]))

    ident = pd.read_csv(OUT / "identifiability.csv")
    for w, tag in ((2.0, "Two"), (3.0, "Three"), (5.0, "Five")):
        f = ident[np.isclose(ident.hydrogen_wt_percent, w)].sort_values("bump_amplitude")
        shift = f.spinodal_temperature_K.iloc[-1] - f.spinodal_temperature_K.iloc[0]
        put(f"bump{tag}", shift, "{:.1f}")
        put(f"bump{tag}Abs", abs(shift), "{:.1f}")
    checks = json.loads((OUT / "identifiability_checks.json").read_text())
    put("bumpLambdaMax", checks["admissible_amplitude"], "{:.2f}")
    put("bumpCpBase", checks["heat_capacity_baseline_kB_per_atom"], "{:.1f}")
    put(
        "bumpCpWorst",
        -checks["most_negative_bump_heat_capacity_kB_per_atom_per_unit_amplitude"],
        "{:.1f}",
    )

    # Excess volume at the middle composition, with the full error budget of Table B.2.
    states, values, cov = homogeneous_data()
    i = int(np.argsort([st["z"] for st in states])[1])
    z_mid = states[i]["z"]
    put("vObsMid", values[6 + i], "{:.2f}")
    put("vSigMid", math.sqrt(cov[6 + i, 6 + i]), "{:.2f}")
    put("vModMid", m.volume_cm3(z_mid), "{:.2f}")
    put("xMid", z_mid, "{:.3f}")

    calib = OUT / "calibration_summary.json"
    if calib.exists():
        c = json.loads(calib.read_text())
        pub = c["published_fit"]["chi2"]
        put("pubChiPhase", pub["phase_state"], "{:.1f}")
        put("pubChiComp", pub["coexistence"], "{:.1f}")
        put("pubChiHom", pub["homogeneous"], "{:.1f}")
        put("pubChiTotal", c["published_fit"]["chi2_total"], "{:.0f}")
        medians = []
        for run in c["runs"]:
            sfx = SUFFIX[(run["model"], run["version"])]
            q = run["quantiles"]
            put(f"Int{sfx}", q["interval_K"]["p50"])
            put(f"Int{sfx}Lo", q["interval_K"]["p16"])
            put(f"Int{sfx}Hi", q["interval_K"]["p84"])
            put(f"Tc{sfx}", q["T_C_K"]["p50"])
            put(f"Tc{sfx}Lo", q["T_C_K"]["p16"])
            put(f"Tc{sfx}Hi", q["T_C_K"]["p84"])
            put(f"Fh{sfx}", 100 * q["H_rich_fraction_at_spinodal"]["p50"], "{:.0f}")
            put(f"Xc{sfx}", q["z_C"]["p50"], "{:.2f}")
            put(f"Chi{sfx}", run["map_chi2_total"], "{:.0f}")
            put(f"ChiPhase{sfx}", run["map_chi2"]["phase_state"], "{:.1f}")
            put(f"ChiComp{sfx}", run["map_chi2"]["coexistence"], "{:.1f}")
            put(f"ChiHom{sfx}", run["map_chi2"]["homogeneous"], "{:.1f}")
            put(
                f"Rhat{sfx}",
                max(v["rhat"] for v in run["diagnostics"].values() if math.isfinite(v["rhat"])),
                "{:.2f}",
            )
            put(f"Undef{sfx}", 100 * run["undefined_spinodal_fraction"], "{:.0f}")
            miss = [t for t in run["map_two_phase"] if t["misclassified_K"] > 0]
            put(f"Miss{sfx}", max([t["misclassified_K"] for t in miss], default=0.0))
            medians.append(q["interval_K"]["p50"])
        m4 = next(r for r in c["runs"] if r["model"] == "M4")["quantiles"]["interval_K"]
        put("IntMFourLoNinetyFive", m4["p2.5"])
        put("IntMFourHiNinetyFive", m4["p97.5"])
        samples = pd.read_csv(OUT / "calibration_samples.csv")
        m4s = samples[(samples.model == "M4") & (samples.version == "table")].interval_K
        tail = np.mean(m4s >= c["published_fit"]["derived"]["interval_K"])
        put("IntMFourPubTail", 100 * tail, "{:.1f}")
        from research.spinodal_onset.calibration import CLASSES

        for run in c["runs"]:
            if run["model"] != "M1":
                continue
            cls = CLASSES["M1"]
            model = cls.build([run["map"][k] for k in cls.labels])
            pi = 1 / run["map"]["inv_pi_per_GPa"]
            sfx = SUFFIX[("M1", run["version"])]
            put(f"Slope{sfx}", model.critical_point(0.0)[0] / abs(pi))
            put(f"BA{sfx}", run["map"]["B_kJ"] / run["map"]["A_kJ"], "{:.1f}")
        put("slopePublished", m.critical_point(0.0)[0] / 35.0)
        put("slopeMinimum", (3500 - 3000) / (5.14 - 1.65))
        miss = [
            t["misclassified_K"]
            for r in c["runs"]
            if r["version"] != "phase_hom"
            for t in r["map_two_phase"]
            if t["t"] == 3500
        ]
        put("missLo", min(miss))
        put("missHi", max(miss))
        put("IntRangeLo", min(medians))
        put("IntRangeHi", max(medians))
        rhat = {
            SUFFIX[(r["model"], r["version"])]: max(
                v["rhat"] for v in r["diagnostics"].values() if math.isfinite(v["rhat"])
            )
            for r in c["runs"]
        }
        put("maxRhat", max(rhat.values()), "{:.3f}")
        put("nRhatAbove", f"{sum(v > 1.01 for v in rhat.values())}")
    design = OUT / "design_summary.json"
    if design.exists():
        d = json.loads(design.read_text())
        best = {b["group"]: b for b in d["best_single_homogeneous_composition"]}
        pooled = best["pooled (equal class weights)"]
        put("designBestX", pooled["composition_x"], "{:.2f}")
        put("designSdNow", pooled["current_sd_K"])
        put("designSdAfter", pooled["expected_sd_K"])
        top = d["curvature_measurement"]  # lowest T at which every pooled sample is one phase
        put("curvSdPooled", top["expected_sd_K"])
        put("curvT", top["temperature_K"])
        put("curvFracStable", 100 * top["fraction_locally_stable"], "{:.0f}")
        put("curvFracMeta", 100 * top["fraction_metastable"], "{:.0f}")
        put("curvFracOnePhase", 100 * top["fraction_one_phase"], "{:.0f}")

    tables(m)
    lines = ["% Generated by paper/make_numbers.py from research/spinodal_onset/outputs."]
    for name, value in sorted(macros.items()):
        lines.append(f"\\newcommand{{\\{name}}}{{{value}}}")
    (HERE / "numbers.tex").write_text("\n".join(lines) + "\n")
    print(f"{len(macros)} macros written")


def tables(m):
    """Table bodies for the appendix, written from the source files and outputs."""
    (HERE / "tables").mkdir(exist_ok=True)
    two, one = coexistence_runs()
    dig = json.loads((OUT / "source_figure_digitized.json").read_text())
    fig = {}
    for series, t, run_t in (("red", 3500, 3500), ("grey", 3000, -3000)):
        fig[run_t] = sorted(
            mk["x"]
            for mk in dig["markers"]
            if mk["series"] == series and round(mk["temperature_K"], -2) == t
        )
    rows = []
    for r in sorted(two + one, key=lambda r: (r["p"] > 3, r["p"] > 7, r["t"])):
        tc = m.critical_point(r["p"])[0]
        gap = m.binodal(r["t"], r["p"])
        model = f"{gap[0]:.3f}, {gap[1]:.3f}" if gap else "one phase"
        if "lo" in r:
            key = 3500 if r["t"] == 3500 else (-3000 if r["t"] == 3000 and r["p"] > 4 else None)
            figure = f"{fig[key][0]:.3f}, {fig[key][1]:.3f}" if key else "same"
            obs = f"{r['lo']:.3f}, {r['hi']:.3f}"
            state = "two"
        else:
            figure, obs, state = "--", f"({r['x']:.3f})", "one"
        rows.append(
            f"{r['p']:.2f}({round(r['sp'] * 100):d}) & {r['t']:.0f} & {state} & {obs} & "
            f"{figure} & {model} & {tc - r['t']:+.0f} \\\\"
        )
    (HERE / "tables/ed_table1.tex").write_text(
        "\n".join(rows).replace("-", "$-$").replace("$-$$-$", "--") + "\n"
    )
    states, values, cov = homogeneous_data()
    sd = np.sqrt(np.diag(cov))
    rows = []
    for i, st in enumerate(states):
        z, p = st["z"], st["p"]
        rows.append(
            f"{z:.3f} & {values[i] / 1e3:.1f} $\\pm$ {sd[i] / 1e3:.1f} & {m.enthalpy(z, p) / 1e3:.1f} & "
            f"{values[3 + i]:.1f} $\\pm$ {sd[3 + i]:.1f} & {m.entropy(z):.1f} & "
            f"{values[6 + i]:.2f} $\\pm$ {sd[6 + i]:.2f} & {m.volume_cm3(z):.2f} \\\\"
        )
    (HERE / "tables/homogeneous.tex").write_text("\n".join(rows).replace("-", "$-$") + "\n")
    calib = OUT / "calibration_summary.json"
    if not calib.exists():
        return
    c = json.loads(calib.read_text())
    labels = {
        ("M1", "table"): "M1, all data",
        ("M1", "figure"): "M1, Fig.~2a compositions",
        ("M1", "inflated"): "M1, $\\sigma_x=0.15$ near $T_C$",
        ("M1", "no_volume"): "M1, without volumes",
        ("M1", "phase_comp"): "M1, phases + compositions",
        ("M1", "phase_hom"): "M1, phases + homogeneous",
        ("M2", "table"): "M2, all data",
        ("M3", "table"): "M3, all data",
        ("M4", "table"): "M4, all data",
    }
    pub = c["published_fit"]
    rows = [
        f"Published fit & {pub['chi2']['phase_state']:.1f} & {pub['chi2']['coexistence']:.1f} & "
        f"{pub['chi2']['homogeneous']:.1f} & {pub['derived']['T_C_K']:.0f} & "
        f"{pub['derived']['interval_K']:.0f} & "
        f"{100 * pub['derived']['H_rich_fraction_at_spinodal']:.0f} \\\\ \\midrule"
    ]
    reduced = {"no_volume", "phase_comp", "phase_hom"}  # fewer data: chi2 not comparable
    for run in c["runs"]:
        q, x = run["quantiles"], run["map_chi2"]
        comp = "--" if run["version"] == "phase_hom" else f"{x['coexistence']:.1f}"
        hom = "--" if run["version"] == "phase_comp" else f"{x['homogeneous']:.1f}"
        mark = "$^\\dagger$" if run["version"] in reduced else ""

        def band(k, scale=1.0, fmt="{:.0f}"):
            v = q[k]
            return (
                fmt.format(scale * v["p50"])
                + " {\\scriptsize["
                + fmt.format(scale * v["p16"])
                + "--"
                + fmt.format(scale * v["p84"])
                + "]}"
            )

        rows.append(
            f"{labels[(run['model'], run['version'])]}{mark} & {x['phase_state']:.1f} & "
            f"{comp} & {hom} & {band('T_C_K')} & "
            f"{band('interval_K')} & {band('H_rich_fraction_at_spinodal', 100)} \\\\"
        )
    (HERE / "tables/recalibration.tex").write_text("\n".join(rows) + "\n")
    rows = ["Published fit & 75 & 424 & 5.65 & 4400 & $-$35 \\\\ \\midrule"]
    for run in c["runs"]:
        if run["model"] != "M1":
            continue
        p = run["map"]
        a, b = p["A_kJ"], p["B_kJ"]
        tau = 1 / p["inv_tau_per_K"] if p["inv_tau_per_K"] else math.inf
        pi = 1 / p["inv_pi_per_GPa"] if p["inv_pi_per_GPa"] else math.inf
        rows.append(
            f"{labels[('M1', run['version'])]} & {a:.0f} & {b:.0f} & {b / a:.2f} & "
            f"{tau:.0f} & {pi:.0f} \\\\".replace("-", "$-$")
        )
    (HERE / "tables/map_parameters.tex").write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
