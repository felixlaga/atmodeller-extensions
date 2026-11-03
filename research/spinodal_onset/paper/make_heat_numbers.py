# SPDX-License-Identifier: Apache-2.0
"""Write the heat and kinetics numbers quoted in the draft to paper/heat_numbers.tex.

Run from the repository root after heat_study.py. Kept separate from
make_numbers.py so the two analyses can be rerun independently.
"""

import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
HEAT = HERE.parent / "outputs" / "heat"
WITH_ENTHALPY = ["M1_table", "M1_figure", "M1_inflated", "M1_no_volume", "M1_phase_hom"]
macros = {}


def put(name, value, fmt="{:.0f}"):
    macros[name] = fmt.format(value)


h = pd.read_csv(HEAT / "hydrogen_partial_enthalpy.csv")
pub = h[h.set == "published"]
put("hbarPubLo", pub.q50.min())
put("hbarPubHi", pub.q50.max())
m1 = h[h.set.isin(WITH_ENTHALPY)]
put("hbarMOneLo", m1.MJ_per_kgH_q025.min())
put("hbarMOneHi", m1.q975.max())
table = h[h.set == "M1_table"].set_index(["w_H2", "P_GPa"])
put("hbarMOneTableOne", table.loc[(0.01, 2.0), "q50"])
put("hbarMOneTableThree", table.loc[(0.03, 2.0), "q50"])
comp = h[h.set == "M1_phase_comp"]
put("hbarPhaseCompLo", comp.MJ_per_kgH_q025.min())
put("hbarPhaseCompHi", comp.q975.max())

# Excess enthalpies of ED Table 2 after the endmember pressure correction (Table B.2).
e = pd.read_csv(HEAT / "edtable2_excess_enthalpy.csv")
for prefix, z in (("edHex", 0.2381), ("edHexRich", 0.8333)):
    rows = e[e.z_ox.round(4) == z].set_index("set")
    put(prefix + "Data", rows.loc["published", "data_kJ_per_mol"], "{:.1f}")
    put(prefix + "Sig", rows.loc["published", "data_sigma_kJ_per_mol"], "{:.1f}")
    put(prefix + "Pub", rows.loc["published", "model_q50"], "{:.1f}")
    put(prefix + "MOne", rows.loc["M1_table", "model_q50"], "{:.1f}")
low = e[e.z_ox.round(4) == 0.2381].set_index("set")
put("edHexPhaseComp", low.loc["M1_phase_comp", "model_q50"], "{:.0f}")
put(
    "edHexPull",
    (low.loc["published", "data_kJ_per_mol"] - low.loc["published", "model_q50"])
    / low.loc["published", "data_sigma_kJ_per_mol"],
    "{:.1f}",
)

# Comparisons with the latent heat of melting of silicate, 0.73 MJ/kg (Tejada Arevalo et al.).
LATENT = 0.73
put("latentRatioLo", m1.MJ_per_kgH_q025.min() / LATENT)
put("latentRatioHi", m1.q975.max() / LATENT)

# An excess heat capacity c_ex = 50 x(1-x) J/mol/K, zero excess at 4000 K, shifts the
# H2 partial enthalpy at 3000 K by 50 (1-x)^2 kJ/mol (oxide basis), relative to the
# published and M1 medians over the energy grid.
grid = h[h.set.isin(["published", "M1_table"])]
y = (grid.w_H2 / 2.01588e-3) / (grid.w_H2 / 2.01588e-3 + (1 - grid.w_H2) / 100.3887e-3)
x_ox = y / (2 - y)
shift = 50e3 * (1 - x_ox) ** 2 / 2.01588e-3 / 1e6
put("cexShiftLo", 100 * (shift / grid.q50).min())
put("cexShiftHi", 100 * (shift / grid.q50).max())

i = pd.read_csv(HEAT / "interior_300K.csv").set_index(["set", "w_H2", "P_GPa", "quantity"])


def interior(prefix, label, w, quantity, fmt="{:.1f}", p=2.0):
    row = i.loc[(label, w, p, quantity)]
    put(prefix, row.q50, fmt)
    put(prefix + "Lo", row.q025, fmt)
    put(prefix + "Hi", row.q975, fmt)


interior("intRatioThree", "M1_table", 0.03, "exsolution_over_sensible")
interior("intRatioOne", "M1_table", 0.01, "exsolution_over_sensible")
interior("intGammaThree", "M1_table", 0.03, "max_heat_capacity_ratio")
interior("intGammaOne", "M1_table", 0.01, "max_heat_capacity_ratio")
interior("intPerKgThree", "M1_table", 0.03, "MJ_per_kgH", "{:.0f}")
exsolved = i.loc[("M1_table", 0.03, 2.0, "exsolved_fraction")]
for suffix, column in (("", "q50"), ("Lo", "q025"), ("Hi", "q975")):
    put("intExsolvedThree" + suffix, 100 * exsolved[column])
put("intPubRatioThree", i.loc[("published", 0.03, 2.0, "exsolution_over_sensible")].q50, "{:.1f}")
put("intPubGammaThree", i.loc[("published", 0.03, 2.0, "max_heat_capacity_ratio")].q50, "{:.1f}")
interior("intHeatThree", "M1_table", 0.03, "exsolution_MJ_per_kg")
interior("intHeatOne", "M1_table", 0.01, "exsolution_MJ_per_kg")
for w_label, w_value in (("Three", 0.03), ("One", 0.01)):
    heat = i.loc[("M1_table", w_value, 2.0, "exsolution_MJ_per_kg")].q50
    put("latentInterior" + w_label, heat / LATENT, "{:.1f}")
# The effective heat capacity is the sensible one times (1 + ratio).
gamma = i.loc[("M1_table", 0.03, 2.0, "max_heat_capacity_ratio")]
for suffix, column in (("", "q50"), ("Lo", "q025"), ("Hi", "q975")):
    put("intCeffThree" + suffix, 1 + gamma[column], "{:.1f}")
ratio = i.loc[("M1_table", 0.03, 2.0, "exsolution_over_sensible")].q50
put("intSlowThree", 1 + ratio, "{:.1f}")
put("intRatioThreeAdiabat", ratio / 1.3, "{:.1f}")

# Sensitivity checks of the interior (heat_study.py checks).
checks = json.loads((HEAT / "checks.json").read_text())
check_post = checks["interior_3wt_2GPa_300K"]["M1_table"]
check_pub = checks["interior_3wt_2GPa_300K"]["published"]


def check(prefix, quantity, fmt="{:.1f}", scale=1.0, bounds=True):
    q025, q50, q975 = check_post[quantity][:3]
    put(prefix, scale * q50, fmt)
    if bounds:
        put(prefix + "Lo", scale * q025, fmt)
        put(prefix + "Hi", scale * q975, fmt)


check("intInteriorOnlyThree", "exsolution_over_interior_sensible")
check("intRatioThreeHundredK", "exsolution_over_sensible_100K", bounds=False)
check("intRatioThreeSixHundredK", "exsolution_over_sensible_600K", bounds=False)
put("intPubRatioThreeHundredK", check_pub["exsolution_over_sensible_100K"][1], "{:.1f}")
put("intPubRatioThreeSixHundredK", check_pub["exsolution_over_sensible_600K"][1], "{:.1f}")
check("intUpperWtThree", "upper_phase_H_wt_1K", "{:.0f}", 100, bounds=False)
check("intPhasePerKgThree", "phase_per_hydrogen_1K", "{:.0f}", 100, bounds=False)
check("intPhasePerKThree", "phase_per_kelvin_1K", "{:.0f}", 100, bounds=False)
check("intPhasePerKHundredThree", "phase_per_kelvin_100K", "{:.0f}", 100, bounds=False)
check("intPhaseGammaThree", "phase_max_heat_capacity_ratio")
check("intPhaseRatioThree", "phase_over_sensible")
check("intNetFracThree", "net_fraction_1K", "{:.0f}", 100, bounds=False)
check("intNetGammaThree", "net_max_heat_capacity_ratio")
check("risePressMOne", "rising_pressure_change", "{:.0f}", -100, bounds=False)
put("risePressPub", -100 * check_pub["rising_pressure_change"][1])
half = checks["half_window"]
put("halfWinWidthLo", half["published"]["width_nm"][0], "{:.1f}")
put("halfWinWidthHi", half["published"]["width_nm"][1], "{:.1f}")
put("halfWinRatioLo", half["published"]["times_simulated"][0])
put("halfWinRatioHi", half["published"]["times_simulated"][1])
put("halfWinMOneWidthLo", half["M1_MAP"]["width_nm"][0], "{:.1f}")
put("halfWinMOneWidthHi", half["M1_MAP"]["width_nm"][1], "{:.1f}")
basis = checks["sensible_basis_3000K"]
put("cpSilicate", basis["cp_silicate_kJ_per_kg_K"], "{:.2f}")
put("cpHydrogen", basis["cp_H2_kJ_per_kg_K"], "{:.1f}")
put("cpHydrogenShare", 100 * basis["H2_share_at_3wt"])

d = pd.read_csv(HEAT / "nucleation_delays.csv")
fig1c = d[d.interface.str.startswith("fig1c")]
young = d[d.interface.str.startswith("young")]
for prefix, frame in (("dlyFigOneC", fig1c), ("dlyYoung", young)):
    pub_rows = frame[frame.set == "published"].delay_K_or_q025.astype(float)
    put(prefix + "PubLo", pub_rows.min(), "{:.1f}")
    put(prefix + "PubHi", pub_rows.max(), "{:.1f}")
    post = frame[frame.set == "M1_table_q025_q50_q975"]
    put(prefix + "MOneLo", post.delay_K_or_q025.astype(float).min(), "{:.1f}")
    put(prefix + "MOneHi", post.q975.astype(float).max(), "{:.1f}")
put("winMOneMap", d[d.set == "M1_MAP"].window_K_or_q50.astype(float).iloc[0])
w = pd.read_csv(HEAT / "nucleation_withheld.csv")
for prefix, key in (("whFigOneC", "fig1c"), ("whYoung", "young")):
    rows = w[(w.set == "published") & w.interface.str.startswith(key)]
    put(prefix + "PubHi", 100 * rows.H_fraction_withheld.max(), "{:.1f}")

p = json.loads((HEAT / "paths_summary.json").read_text())
spin = p["published/spinodal/isenthalpic"]
put("recalPubJump", spin["separation_K"] - spin["onset_K"])
put("recalPubHeat", spin["released_MJ_per_kg"], "{:.1f}")
spin = p["M1_MAP/spinodal/isenthalpic"]
put("recalMOneJump", spin["separation_K"] - spin["onset_K"])
put("recalMOneHeat", spin["released_MJ_per_kg"], "{:.2f}")
nuc = p["published/nucleation_L69/isenthalpic"]
put("nucPubOnset", nuc["binodal_K"] - nuc["onset_K"], "{:.1f}")
put("nucPubJump", nuc["separation_K"] - nuc["onset_K"], "{:.1f}")
put("nucPubHeat", 1e3 * nuc["released_MJ_per_kg"], "{:.0f}")
put("pathFinalPub", p["published/equilibrium/isenthalpic"]["final_heat_lost_MJ_per_kg"], "{:.2f}")
replay = p["engine_replay"]
put("replayRows", replay["rows"])
macros["replayDiff"] = (
    f"{replay['max_relative_difference']:.0e}".replace("e-", r"\times10^{-") + "}"
)

# Exsolution heat along the digitized fiducial evolution of Rogers et al. (2025).
planet = json.loads((HEAT / "planet.json").read_text())
put("planetExsolvedPct", 100 * planet["path"]["exsolved_fraction_of_H"])
pooled = planet["sets"]["M1_with_enthalpy"]
for prefix, key, scale, fmt in (
    ("planetRadiated", "over_radiated", 100, "{:.0f}"),
    ("planetGravity", "over_gravitational", 100, "{:.0f}"),
    ("planetThermal", "over_thermal", 1, "{:.1f}"),
    ("planetSensible", "over_interior_sensible", 1, "{:.1f}"),
    ("planetDelay", "time_extension", 100, "{:.0f}"),
):
    q025, q50, q975 = pooled[key][:3]
    put(prefix, scale * q50, fmt)
    put(prefix + "Lo", scale * q025, fmt)
    put(prefix + "Hi", scale * q975, fmt)
    put(prefix + "Pub", scale * planet["sets"]["published"][key][1], fmt)
    put(prefix + "Map", scale * planet["sets"]["M1_MAP"][key][1], fmt)
put("planetCrossPubGyr", planet["sets"]["published"]["cross_Gyr"][1])
put("planetCrossPubExsolved", 100 * planet["sets"]["published"]["exsolved_before_cross"][1])
put("planetCrossPooledPct", 100 * pooled["fraction_crossing"])
put("planetCrossTablePct", 100 * planet["sets"]["M1_table"]["fraction_crossing"])

lines = ["% Generated by make_heat_numbers.py; do not edit."]
lines += [rf"\newcommand{{\{k}}}{{{v}}}" for k, v in sorted(macros.items())]
(HERE / "heat_numbers.tex").write_text("\n".join(lines) + "\n")
print(f"wrote {len(macros)} macros to {HERE / 'heat_numbers.tex'}")
