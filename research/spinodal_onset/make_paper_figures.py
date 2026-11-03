# SPDX-License-Identifier: Apache-2.0
"""Vector figures for the thesis-layout research draft (paper/main.tex).

Reads only checked-in outputs; computes nothing that the analysis scripts do
not already record, except thin curves from the independent solver in
mixing_models.py. Run from the repository root after the analysis scripts.
"""

import json
import math
from pathlib import Path

import matplotlib as mpl
from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from research.spinodal_onset.mixing_models import (
    published_gilmore26,
    weight_to_z,
)

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
FIG = HERE / "paper/figures"
COLUMN, FULL = 3.48, 7.30  # inches: thesis column and text widths
P_COLOURS = {2.0: "#c0392b", 5.0: "#4d4d4d", 10.0: "#1f4e9c"}
PALATINO = Path("/System/Library/Fonts/Palatino.ttc")  # matches the Palatino body text


def style():
    family = "serif"
    if PALATINO.exists():
        font_manager.fontManager.addfont(str(PALATINO))
        family = "Palatino"
    mpl.rcParams.update(
        {
            "font.family": family,
            "mathtext.fontset": "custom",
            "mathtext.rm": family,
            "mathtext.it": f"{family}:italic",
            "mathtext.bf": f"{family}:bold",
            "font.size": 8.5,
            "axes.titlesize": 9,
            "axes.labelsize": 8.5,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.linewidth": 0.6,
            "lines.linewidth": 1.3,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "legend.frameon": False,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "pdf.fonttype": 42,
        }
    )


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf")
    plt.close(fig)


def label(ax, text):
    ax.text(
        0.02,
        0.97,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontweight="bold",
        fontsize=9,
    )


def onsets():
    return pd.read_csv(OUT / "onsets.csv")


# ----------------------------------------------------------------------------- Fig 1.1


def concept():
    """Free-energy geometry at 2 GPa and the cooling path of a 3 wt% parcel."""
    m = published_gilmore26()
    p, t = 2.0, 3150.0
    z0 = weight_to_z(0.03)
    z1, z2 = m.binodal(t, p)
    s1, s2 = m.unstable_intervals(t, p)[0]
    zs = np.linspace(0.002, 0.998, 2000)
    g = np.array([m.f(z, t, p) for z in zs]) * 8.314462618 * t / 1e3
    g1, g2 = (m.f(z, t, p) * 8.314462618 * t / 1e3 for z in (z1, z2))
    tangent = g1 + (g2 - g1) * (zs - z1) / (z2 - z1)
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.55), gridspec_kw=dict(wspace=0.28))
    ax = axes[0]
    ax.plot(zs, g - tangent, color="k")
    ax.axhline(0, color="0.5", lw=0.7, ls=":")
    meta = ((zs > z1) & (zs < s1)) | ((zs > s2) & (zs < z2))
    ax.fill_between(zs, 0, g - tangent, where=meta, color="#f3c6c0", lw=0, label="metastable")
    ax.fill_between(
        zs,
        0,
        g - tangent,
        where=(zs >= s1) & (zs <= s2),
        color="#c0392b",
        alpha=0.55,
        lw=0,
        label="unstable (spinodal)",
    )
    for z in (z1, z2):
        ax.plot(z, 0, "o", ms=4.5, mfc="w", mec="k", mew=0.9)
    for z in (s1, s2):
        ax.plot(z, np.interp(z, zs, g - tangent), "x", ms=5, color="k", mew=1.0)
    ax.axvline(z0, color="#1f4e9c", lw=0.9, ls="--")
    ax.text(z0 - 0.012, 3.2, "3 wt% H", color="#1f4e9c", fontsize=7.5, ha="right")
    ax.set(
        xlabel=r"H$_2$ mole fraction $x$ (oxide basis)",
        ylabel=r"$g_{\rm mix}$ above common tangent (kJ mol$^{-1}$)",
        xlim=(0, 1),
        ylim=(-0.25, 3.5),
    )
    ax.legend(loc="upper right")
    label(ax, "(a)")
    ax.text(0.12, 0.88, f"{t:.0f} K, {p:.0f} GPa", transform=ax.transAxes, fontsize=7.5)

    ax = axes[1]
    d = onsets()
    d = d[(d.model == "gilmore26") & (d.pressure_GPa == 2.0)]
    ax.plot(d.hydrogen_wt_percent, d.binodal_temperature_K, color="k", label="binodal")
    ax.plot(d.hydrogen_wt_percent, d.spinodal_temperature_K, color="k", ls="--", label="spinodal")
    ax.fill_between(
        d.hydrogen_wt_percent,
        d.spinodal_temperature_K.fillna(0),
        d.binodal_temperature_K,
        color="#f3c6c0",
        lw=0,
        label="metastable",
    )
    key = json.loads((OUT / "key_results.json").read_text())["at_2GPa_3wt"]
    ax.annotate(
        "",
        xy=(3, 2500),
        xytext=(3, 3950),
        arrowprops=dict(arrowstyle="-|>", color="#1f4e9c", lw=1.0),
    )
    ax.plot(
        [3, 3],
        [key["spinodal_K"], key["binodal_K"]],
        color="#1f4e9c",
        lw=3.2,
        solid_capstyle="butt",
    )
    ax.text(
        2.75,
        (key["spinodal_K"] + key["binodal_K"]) / 2,
        f"{key['interval_K']:.0f} K",
        color="#1f4e9c",
        va="center",
        ha="right",
        fontsize=7.5,
    )
    ax.set(
        xscale="log",
        xlim=(0.3, 20),
        ylim=(1500, 4000),
        xlabel="local hydrogen mass fraction (wt%)",
        ylabel="temperature (K)",
    )
    ax.set_xticks([0.3, 1, 3, 10, 20], ["0.3", "1", "3", "10", "20"])
    ax.legend(loc="lower right")
    label(ax, "(b)")
    save(fig, "fig1_1_concept")


# ----------------------------------------------------------------------------- Fig 2.1


def verification():
    dig = json.loads((OUT / "source_figure_digitized.json").read_text())
    m = published_gilmore26()
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.55), gridspec_kw=dict(wspace=0.3))
    ax = axes[0]
    zs = np.linspace(0.03, 0.995, 150)
    tb0 = np.array([m.binodal_temperature(z, 0.0) for z in zs])
    for name, p in (("red", 2.0), ("black", 5.0), ("blue", 10.0)):
        c = P_COLOURS[p]
        ax.plot(zs, tb0 * (1 - p / 35), color=c, lw=1.1, label=f"this work, {p:.0f} GPa")
        s = dig["curves"][name]["samples"]
        ax.plot(
            [q["x"] for q in s],
            [q["T_digitized"] for q in s],
            "o",
            ms=2.2,
            mfc="none",
            mec=c,
            mew=0.6,
        )
    ax.plot([], [], "o", ms=2.5, mfc="none", mec="k", mew=0.6, label="digitized published curves")
    ax.set(
        xlim=(0, 1),
        ylim=(1500, 4000),
        xlabel=r"H$_2$ mole fraction $x$",
        ylabel="binodal temperature (K)",
    )
    ax.legend(loc="lower center", ncol=1)
    rms = max(c["rms_residual_K"] for c in dig["curves"].values())
    ax.text(
        0.98,
        0.95,
        f"RMS residual $\\leq$ {rms:.1f} K",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.5,
    )
    label(ax, "(a)")

    ax = axes[1]
    d = onsets()
    d = d[d.model == "gilmore26"]
    for p, c in P_COLOURS.items():
        e = d[(d.pressure_GPa == p) & np.isfinite(d.spinodal_temperature_K)].iloc[::6]
        tb = [m.binodal_temperature(weight_to_z(w / 100), p) for w in e.hydrogen_wt_percent]
        ts = [m.spinodal_temperature(weight_to_z(w / 100), p) for w in e.hydrogen_wt_percent]
        ax.plot(
            e.hydrogen_wt_percent,
            np.abs(np.array(tb) - e.binodal_temperature_K) + 1e-12,
            "o",
            ms=2.6,
            color=c,
            label=f"binodal, {p:.0f} GPa",
        )
        ax.plot(
            e.hydrogen_wt_percent,
            np.abs(np.array(ts) - e.spinodal_temperature_K) + 1e-12,
            "x",
            ms=3.0,
            color=c,
            label=f"spinodal, {p:.0f} GPa",
        )
    ax.set(
        xscale="log",
        yscale="log",
        ylim=(1e-12, 1e-3),
        xlabel="local hydrogen mass fraction (wt%)",
        ylabel=r"|independent $-$ Atmodeller| (K)",
    )
    ax.set_xticks([1.5, 3, 5, 10, 20], ["1.5", "3", "5", "10", "20"])
    ax.minorticks_off()
    ax.legend(ncol=2, loc="upper left", fontsize=6.5)
    label(ax, "(b)")
    save(fig, "fig2_1_verification")


# ----------------------------------------------------------------------------- Fig 3.1, 3.2


def phase_maps():
    d = onsets()
    fig, ax = plt.subplots(figsize=(COLUMN, 2.7))
    for p, c in P_COLOURS.items():
        e = d[(d.model == "gilmore26") & (d.pressure_GPa == p)]
        ax.plot(e.hydrogen_wt_percent, e.binodal_temperature_K, color=c, label=f"{p:.0f} GPa")
        ax.plot(e.hydrogen_wt_percent, e.spinodal_temperature_K, color=c, ls="--")
        ax.fill_between(
            e.hydrogen_wt_percent,
            e.spinodal_temperature_K.fillna(0),
            e.binodal_temperature_K,
            color=c,
            alpha=0.08,
            lw=0,
        )
    e = d[(d.model == "rogers25") & (d.pressure_GPa == 2.0)]
    ax.plot(
        e.hydrogen_wt_percent,
        e.binodal_temperature_K,
        color="0.6",
        lw=0.9,
        label="2025 model, 2 GPa",
    )
    ax.plot(e.hydrogen_wt_percent, e.spinodal_temperature_K, color="0.6", lw=0.9, ls="--")
    ax.axvline(3, color="0.3", lw=0.6, ls=":")
    ax.plot([], [], color="k", label="binodal")
    ax.plot([], [], color="k", ls="--", label="spinodal")
    ax.set(
        xscale="log",
        xlim=(0.3, 20),
        ylim=(1500, 4300),
        xlabel="local hydrogen mass fraction (wt%)",
        ylabel="temperature (K)",
    )
    ax.set_xticks([0.3, 1, 3, 10, 20], ["0.3", "1", "3", "10", "20"])
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
        fontsize=6.6,
        columnspacing=1.0,
        handlelength=1.8,
    )
    save(fig, "fig3_1_phase_maps")


def inventory():
    c = pd.read_csv(OUT / "cooling_paths.csv")
    d = onsets()
    fig, axes = plt.subplots(2, 1, figsize=(COLUMN, 4.6), gridspec_kw=dict(hspace=0.38))
    ax = axes[0]
    for p, col in P_COLOURS.items():
        e = c[
            (c.model == "gilmore26")
            & (c.pressure_GPa == p)
            & np.isclose(c.hydrogen_wt_percent, 3.0)
        ].sort_values("temperature_K")
        ax.plot(
            e.temperature_K,
            100 * e.H_equilibrium_fraction,
            color=col,
            label=f"equilibrium, {p:.0f} GPa",
        )
        ax.plot(e.temperature_K, 100 * e.H_delayed_limit_fraction, color=col, ls="--")
    ax.plot([], [], color="k", ls="--", label="delayed limit")
    ax.set(
        xlim=(2000, 3700), ylim=(-2, 100), xlabel="temperature (K)", ylabel="H in H-rich phase (%)"
    )
    ax.invert_xaxis()
    ax.legend(loc="upper left", fontsize=6.8)
    label(ax, "(a)")
    ax.text(0.03, 0.10, "3 wt% H; cooling to the right", transform=ax.transAxes, fontsize=7.5)
    ax = axes[1]
    crit = pd.read_csv(OUT / "critical_points.csv").set_index("model").loc["gilmore26"]
    w_min, w_c = (
        crit.minimum_spinodal_hydrogen_wt_percent_F_positive,
        crit.critical_hydrogen_wt_percent,
    )
    for p, col in P_COLOURS.items():
        e = d[(d.model == "gilmore26") & (d.pressure_GPa == p) & (d.hydrogen_wt_percent < w_c)]
        ax.plot(
            e.hydrogen_wt_percent,
            100 * e.H_in_rich_phase_at_spinodal,
            color=col,
            lw=2.6 - 0.8 * list(P_COLOURS).index(p),
            label=f"{p:.0f} GPa",
        )
    w_window = d[
        (d.model == "gilmore26") & np.isfinite(d.H_in_rich_phase_at_spinodal)
    ].hydrogen_wt_percent.min()
    ax.axvspan(0.5, w_min, color="0.88", lw=0)
    ax.axvspan(w_min, w_window, color="0.95", lw=0)
    ax.text(
        0.85, 50, "no spinodal", rotation=90, va="center", ha="center", fontsize=7, color="0.35"
    )
    ax.text(
        math.sqrt(w_min * w_window),
        50,
        "spinodal < 2000 K",
        rotation=90,
        va="center",
        ha="center",
        fontsize=6.5,
        color="0.45",
    )
    ax.axvline(3, color="0.3", lw=0.6, ls=":")
    ax.set(
        xscale="log",
        xlim=(0.5, w_c),
        ylim=(0, 100),
        xlabel="local hydrogen mass fraction (wt%)",
        ylabel="H withheld at spinodal (%)",
    )
    ax.set_xticks([0.5, 1, 2, 3, 5, 9], ["0.5", "1", "2", "3", "5", "9"])
    ax.minorticks_off()
    ax.legend(loc="upper right", fontsize=6.8, title="published model", title_fontsize=7)
    label(ax, "(b)")
    save(fig, "fig3_2_inventory")


# ----------------------------------------------------------------------------- Fig 3.x identifiability


def identifiability():
    e = pd.read_csv(OUT / "identifiability_energy.csv")
    s = pd.read_csv(OUT / "identifiability.csv")
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.4), gridspec_kw=dict(wspace=0.3))
    ax = axes[0]
    t = sorted(e.temperature_K.unique())[len(e.temperature_K.unique()) // 2]
    for amp, col in zip(
        sorted(e.bump_amplitude.unique()),
        plt.cm.viridis(np.linspace(0, 0.9, e.bump_amplitude.nunique())),
    ):
        f = e[(e.temperature_K == t) & (e.bump_amplitude == amp)]
        ax.plot(
            f.fit_z,
            f.energy_above_equilibrium_J_per_fit_mol / 1e3,
            color=col,
            label=f"$\\lambda$ = {amp:.2f}" + (" (published)" if amp == 0 else ""),
        )
    ax.set(
        xlabel=r"H$_2$ mole fraction $x$", ylabel=r"$g - g_{\rm eq}$ (kJ mol$^{-1}$)", xlim=(0, 1)
    )
    ax.legend(ncol=1, fontsize=6.5, loc="upper left", bbox_to_anchor=(0.0, 0.9))
    ax.text(
        0.98, 0.95, f"{t:.0f} K, 2 GPa", transform=ax.transAxes, ha="right", va="top", fontsize=7.5
    )
    label(ax, "(a)")
    ax = axes[1]
    for w, col in zip((2.0, 3.0, 5.0), ("#c0392b", "#1f4e9c", "#4d4d4d")):
        f = s[np.isclose(s.hydrogen_wt_percent, w)].sort_values("bump_amplitude")
        base = f.spinodal_temperature_K.iloc[0]
        ax.plot(
            f.bump_amplitude,
            f.spinodal_temperature_K - base,
            "o-",
            ms=3,
            color=col,
            label=f"{w:g} wt% H",
        )
        ax.plot(
            f.bump_amplitude,
            f.binodal_temperature_K - f.binodal_temperature_K.iloc[0],
            color=col,
            lw=0.6,
            ls=":",
        )
    ax.axhline(0, color="0.5", lw=0.6)
    ax.set(xlabel=r"perturbation amplitude $\lambda$", ylabel="change in onset temperature (K)")
    ax.legend(loc="center right", fontsize=7, bbox_to_anchor=(1.0, 0.62))
    ax.text(
        0.98, 0.33, "dotted: binodal (unchanged)", transform=ax.transAxes, ha="right", fontsize=7
    )
    label(ax, "(b)")
    save(fig, "fig3_6_identifiability")


def main():
    style()
    concept()
    verification()
    phase_maps()
    inventory()
    identifiability()
    calib = OUT / "calibration_summary.json"
    if calib.exists():
        from research.spinodal_onset.make_paper_calibration_figures import main as cal

        cal()


if __name__ == "__main__":
    main()
