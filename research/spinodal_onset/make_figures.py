# SPDX-License-Identifier: Apache-2.0
"""Render standalone scientific figures and verify reported summary quantities."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
FIG = HERE / "figures"
COLORS = {"gilmore26": "#166b79", "rogers25": "#a45a32", "wrong_basis_control": "#8c7aa6"}
LABELS = {
    "gilmore26": "Final 2026, correct basis",
    "rogers25": "Historical 2025",
    "wrong_basis_control": "2026, incorrect basis control",
}


def read(name):
    return pd.read_csv(OUT / f"{name}.csv")


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=220, facecolor="white", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.svg", facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "legend.frameon": False,
            "axes.labelcolor": "#273a45",
            "text.color": "#273a45",
            "svg.fonttype": "none",
        }
    )
    onsets = read("onsets")
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4), constrained_layout=True)
    for name in COLORS:
        d = onsets.loc[(onsets.model == name) & (onsets.pressure_GPa == 2)]
        axes[0].plot(
            d.hydrogen_wt_percent, d.binodal_temperature_K, color=COLORS[name], label=LABELS[name]
        )
        axes[0].plot(d.hydrogen_wt_percent, d.spinodal_temperature_K, "--", color=COLORS[name])
        resolved = d.spinodal_temperature_K >= 2000
        axes[1].plot(
            d.loc[resolved, "hydrogen_wt_percent"],
            d.loc[resolved, "metastable_interval_K"],
            color=COLORS[name],
        )
    axes[0].axhspan(2000, 2500, color="#edc77e", alpha=0.23)
    axes[0].set(
        xlim=(0.1, 20),
        ylim=(2000, 4100),
        xscale="log",
        ylabel="Temperature (K)",
        title="A   Binodal (solid) and spinodal (dashed)",
    )
    axes[0].legend(fontsize=8.5, loc="upper left")
    axes[0].text(0.13, 2080, "Below lowest two-phase DFT temperature", fontsize=8)
    axes[1].set(
        xlim=(0.7, 10),
        ylim=(0, 2100),
        xscale="log",
        ylabel="Binodal - spinodal temperature (K)",
        title="B   Available metastable cooling interval",
    )
    axes[1].scatter([3], [413.738162], color=COLORS["gilmore26"], zorder=3)
    axes[1].annotate(
        "3 wt%: 414 K",
        (3, 413.738162),
        xytext=(3.65, 700),
        arrowprops={"arrowstyle": "-", "color": "#777"},
        fontsize=9,
    )
    for ax in axes:
        ax.set_xlabel("Local binary hydrogen mass fraction (wt%)")
        ax.grid(alpha=0.15)
    save(fig, "01_onset_comparison")

    paths = read("cooling_paths")
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.5), constrained_layout=True, sharey=True)
    for ax, p in zip(axes, (2, 5, 10), strict=True):
        d = paths.loc[
            (paths.model == "gilmore26")
            & (paths.pressure_GPa == p)
            & (paths.hydrogen_wt_percent == 3)
        ]
        b = onsets.loc[
            (onsets.model == "gilmore26")
            & (onsets.pressure_GPa == p)
            & np.isclose(onsets.hydrogen_wt_percent, 3)
        ].iloc[0]
        ax.axvspan(2000, 2500, color="#edc77e", alpha=0.23)
        ax.axvspan(b.spinodal_temperature_K, b.binodal_temperature_K, color="#166b79", alpha=0.09)
        ax.plot(
            d.temperature_K,
            d.H_equilibrium_fraction * 100,
            color="#166b79",
            label="Instant equilibrium",
        )
        ax.plot(
            d.temperature_K,
            d.H_delayed_limit_fraction * 100,
            "--",
            color="#a45a32",
            label="No nucleation until spinodal",
        )
        ax.axvline(b.spinodal_temperature_K, color="#888", linewidth=0.8)
        ax.set(
            xlim=(4000, 2000),
            ylim=(-2, 102),
            xlabel="Cooling temperature (K)",
            title=f"{p} GPa | {b.metastable_interval_K:.0f} K interval",
        )
        ax.grid(alpha=0.15)
    axes[0].set_ylabel("Parcel hydrogen in the separate rich phase (%)")
    axes[0].legend(fontsize=7.4, loc="upper left")
    save(fig, "02_cooling_bounds")

    audit = read("source_data_audit")
    d = audit.loc[(audit.model == "gilmore26") & audit.observed_two_phase].reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.7), constrained_layout=True)
    for i, row in d.iterrows():
        for phase in ("low", "high"):
            observed, predicted = row[f"observed_z_{phase}"], row[f"predicted_z_{phase}"]
            axes[0].errorbar(
                observed,
                i + 0.1,
                xerr=0.05,
                fmt="o",
                color="#273a45",
                capsize=3,
                label="Tabulated simulations" if i == 0 and phase == "low" else None,
            )
            axes[0].plot(
                predicted,
                i - 0.1,
                "x",
                color="#bc6237",
                markersize=7,
                label="Published fit reproduced" if i == 0 and phase == "low" else None,
            )
    axes[0].set_yticks(
        range(4), [f"{r.pressure_GPa:.2f} GPa\n{r.temperature_K:.0f} K" for _, r in d.iterrows()]
    )
    axes[0].set(
        xlim=(0, 1.03),
        ylim=(-0.5, 4.05),
        xlabel="Published oxide-basis H$_2$ mole fraction $z$",
        title="A   Coexistence compositions",
    )
    axes[0].legend(fontsize=8, loc="upper left")
    axes[0].grid(axis="x", alpha=0.15)
    props = read("homogeneous_data_audit").sort_values("fit_z")
    prop_labels = [
        ("enthalpy_J_per_fit_mol", "Enthalpy"),
        ("excess_entropy_J_per_fit_mol_K", "Excess entropy"),
        ("volume_cm3_per_fit_mol", "Excess volume"),
    ]
    for i, (key, label) in enumerate(prop_labels):
        res = (props[f"model_{key}"] - props[f"observed_{key}"]) / props[f"sigma_{key}"]
        axes[1].plot(props.fit_z, res, marker=("o", "s", "^")[i], label=label)
    axes[1].axhspan(-1, 1, color="#166b79", alpha=0.1)
    axes[1].axhline(0, color="#777", linewidth=0.7)
    axes[1].set(
        xlim=(0.15, 0.92),
        ylim=(-10, 3),
        xlabel="Published oxide-basis H$_2$ mole fraction $z$",
        ylabel="(Model - reconstructed value) / quoted error",
        title="B   Homogeneous-property reproduction",
    )
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.15)
    save(fig, "03_source_audit")

    stress = read("sensitivity")
    d = stress.loc[(stress.pressure_GPa == 2) & (stress.hydrogen_wt_percent == 3)]
    iden = read("identifiability")
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.7), constrained_layout=True)
    for key, marker, label, color in (
        ("corner", "o", "Combined corners", "#8c7aa6"),
        ("other", "s", "One parameter at a time", "#a45a32"),
    ):
        selected = (
            d.perturbation.str.startswith("corner")
            if key == "corner"
            else ~d.perturbation.str.startswith("corner") & (d.perturbation != "nominal")
        )
        axes[0].scatter(
            d.loc[selected, "metastable_interval_K"],
            d.loc[selected, "H_in_rich_phase_at_spinodal"] * 100,
            marker=marker,
            label=label,
            color=color,
            alpha=0.8,
        )
    nominal = d.loc[d.perturbation == "nominal"].iloc[0]
    axes[0].scatter(
        nominal.metastable_interval_K,
        nominal.H_in_rich_phase_at_spinodal * 100,
        marker="*",
        s=160,
        label="Nominal",
        color="#166b79",
        zorder=5,
    )
    axes[0].set(
        xlabel="Metastable interval (K)",
        ylabel="Hydrogen in rich phase at spinodal (%)",
        title="A   +/-10% parameter stress tests | 3 wt%",
    )
    axes[0].legend(fontsize=7.5)
    for w, color in ((2, "#a45a32"), (3, "#166b79"), (5, "#8c7aa6")):
        subset = iden.loc[iden.hydrogen_wt_percent == w]
        shift = subset.spinodal_temperature_K - subset.iloc[0].spinodal_temperature_K
        axes[1].plot(subset.bump_amplitude, shift, "o-", color=color, label=f"{w} wt%")
    axes[1].set(
        xlabel="Constructed bump amplitude $\lambda$",
        ylabel="Spinodal shift from baseline (K)",
        title="B   Identical binodal, different spinodal",
    )
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.15)
    save(fig, "04_sensitivity_and_identifiability")

    stops = read("stopping_temperatures")
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.3), constrained_layout=True, sharey=True)
    for ax, p in zip(axes, (2, 5, 10), strict=True):
        for t, color in ((3000, "#166b79"), (2500, "#8c7aa6"), (2000, "#a45a32")):
            subset = stops.loc[(stops.pressure_GPa == p) & (stops.stopping_temperature_K == t)]
            ax.plot(
                subset.hydrogen_wt_percent,
                subset.withheld_H_fraction * 100,
                color=color,
                label=f"Stop at {t} K",
            )
        ax.set(xlim=(0.1, 9), ylim=(-2, 102), xlabel="Local hydrogen (wt%)", title=f"{p} GPa")
        ax.grid(alpha=0.15)
    axes[0].set_ylabel("Equilibrium / delayed-limit H discrepancy (%)")
    axes[0].legend(fontsize=8)
    save(fig, "05_cooling_endpoint")

    # The pressure invariance is algebraic; test it on every resolved grid point.
    resolved = onsets.loc[
        (onsets.model == "gilmore26") & onsets.H_in_rich_phase_at_spinodal.notna()
    ]
    spread = (
        resolved.groupby("hydrogen_wt_percent")
        .H_in_rich_phase_at_spinodal.agg(lambda s: s.max() - s.min())
        .max()
    )
    # The lever rule is ill-conditioned as binodal endpoints merge. Keep the
    # measured near-critical discrepancy visible, rather than silently dropping it.
    critical_w = (
        read("critical_points").set_index("model").loc["gilmore26", "critical_hydrogen_wt_percent"]
    )
    away = resolved.loc[abs(resolved.hydrogen_wt_percent - critical_w) >= 0.1]
    away_spread = (
        away.groupby("hydrogen_wt_percent")
        .H_in_rich_phase_at_spinodal.agg(lambda s: s.max() - s.min())
        .max()
    )
    assert away_spread < 1e-9
    assert spread < 1e-4  # includes a sampled state only 0.0213 wt% from critical
    picked = d.loc[d.perturbation.isin(["nominal", "C_0.9", "C_1.1", "abs_D_0.9", "abs_D_1.1"])]
    assert np.ptp(picked.H_in_rich_phase_at_spinodal.to_numpy()) < 1e-10
    fixed = paths.loc[
        (paths.model == "gilmore26")
        & (paths.pressure_GPa == 2)
        & (paths.hydrogen_wt_percent == 3)
        & (paths.temperature_K == 3000)
    ].iloc[0]
    primary = onsets.loc[
        (onsets.model == "gilmore26")
        & (onsets.pressure_GPa == 2)
        & np.isclose(onsets.hydrogen_wt_percent, 3)
    ].iloc[0]
    key = {
        "at_2GPa_3wt": {
            "binodal_K": float(primary.binodal_temperature_K),
            "spinodal_K": float(primary.spinodal_temperature_K),
            "interval_K": float(nominal.metastable_interval_K),
            "H_rich_fraction_at_spinodal": float(nominal.H_in_rich_phase_at_spinodal),
        },
        "parameter_stress_interval_range_K": [
            float(d.metastable_interval_K.min()),
            float(d.metastable_interval_K.max()),
        ],
        "parameter_stress_H_fraction_range": [
            float(d.H_in_rich_phase_at_spinodal.min()),
            float(d.H_in_rich_phase_at_spinodal.max()),
        ],
        "at_3000K_2GPa_3wt": fixed.to_dict(),
        "maximum_pressure_dependence_of_H_fraction_at_spinodal": float(spread),
        "maximum_pressure_dependence_excluding_0_1_wt_percent_from_critical": float(away_spread),
        "constructed_3wt_spinodal_shift_K": float(
            iden.loc[
                (iden.hydrogen_wt_percent == 3)
                & (iden.bump_amplitude == iden.bump_amplitude.max()),
                "spinodal_temperature_K",
            ].iloc[0]
            - nominal.spinodal_temperature_K
        ),
    }
    (OUT / "key_results.json").write_text(json.dumps(key, indent=2) + "\n")
    print(json.dumps(key, indent=2))


if __name__ == "__main__":
    main()
