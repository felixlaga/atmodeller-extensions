#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Figures for the heat and kinetics of hydrogen exsolution (run heat_study.py first).

Usage: uv run python research/spinodal_onset/heat_figures.py
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter
import numpy as np

from atmodeller.cooling import InteriorCooling
from atmodeller.sci_utils import GAS_CONSTANT as R

from heat_study import (
    AVOGADRO,
    CRITERIA,
    MOLAR_VOLUMES,
    OUT,
    PUBLISHED,
    load_samples,
    mixture,
    model_from,
    native_x,
)

FIGURES = Path(__file__).resolve().parent / "paper" / "figures"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
"""Default categorical slots 1-3, validated for CVD separation on white"""
INK, MUTED, GRID = "#1f1f1e", "#6b6b68", "#d9d8d4"
plt.rcParams.update(
    {
        "font.size": 8,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.6,
        "legend.frameon": False,
    }
)


def m1_map():
    table = load_samples()["table"]
    return tuple(table[np.argmax(table[:, 4])][:4])


def parcel_figure():
    rows = list(csv.DictReader(open(OUT / "paths.csv")))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=False)
    styles = {
        ("equilibrium", "isenthalpic"): (BLUE, "-", "equilibrium"),
        ("spinodal", "isenthalpic"): (ORANGE, "--", "delayed to spinodal"),
        ("nucleation_L69", "isenthalpic"): (AQUA, ":", "nucleation-limited"),
    }
    for ax, label, title in zip(
        axes, ("published", "M1_MAP"), ("Published fit", "Refit M1 (MAP)")
    ):
        for (path, separation), (color, style, name) in styles.items():
            sel = [
                r
                for r in rows
                if (r["set"], r["path"], r["separation"]) == (label, path, separation)
            ]
            q = np.array([float(r["heat_lost_MJ_per_kg"]) for r in sel])
            t = np.array([float(r["T_K"]) for r in sel])
            ax.plot(q, t, style, color=color, label=name, zorder=3 if path != "equilibrium" else 2)
        ax.set_title(title, color=INK, loc="left", fontsize=8.5)
        ax.set_xlabel("heat lost (MJ/kg)")
        ax.grid(color=GRID, lw=0.5)
        ax.set_xlim(0, 3.6)
    axes[0].set_ylabel("temperature (K)")
    axes[0].legend(loc="lower left", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig5_1_parcel_paths.pdf")
    plt.close(fig)


def interior_figure():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    for params, color, name in ((PUBLISHED, BLUE, "published"), (m1_map(), ORANGE, "M1 MAP")):
        model = model_from(*params)
        for p_gpa, style in ((2.0, "-"), (5.0, "--")):
            interior = InteriorCooling.from_species(
                mixture(), model, p_gpa * 1e4, mass_fraction=0.03
            )
            path = interior.path(3500.0, 2600.0, steps=600)
            q = np.asarray(path.heat_lost) / 1e6
            t = np.asarray(path.temperature)
            label = f"{name}, {p_gpa:.0f} GPa"
            axes[0].plot(q, t, style, color=color, label=label)
            axes[1].plot(t, np.asarray(path.heat_capacity_ratio), style, color=color, label=label)
        # Without exsolution heat, the same heat loss cools the interior further.
        sensible = np.asarray(path.sensible_heat) / 1e6
        if name == "M1 MAP":
            axes[0].plot(sensible, t, ":", color=MUTED, label="no exsolution heat")
    axes[0].set_xlabel("heat lost (MJ/kg)")
    axes[0].set_ylabel("temperature at top of interior (K)")
    axes[0].set_xlim(0, 3.0)
    axes[1].set_xlabel("temperature at top of interior (K)")
    axes[1].set_ylabel("exsolution heat / sensible heat per K")
    axes[1].invert_xaxis()
    for ax in axes:
        ax.grid(color=GRID, lw=0.5)
    axes[0].legend(fontsize=6.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig5_2_interior.pdf")
    plt.close(fig)


def nucleation_figure():
    x, p_bar = native_x(0.03), 2e4
    # Up to 6 nm, so the widths that give half of each window (3.8-5.8 nm) are visible.
    widths = np.geomspace(0.15e-9, 6e-9, 46)
    fig, ax = plt.subplots(figsize=(3.6, 2.9))
    for params, color, name in ((PUBLISHED, BLUE, "published"), (m1_map(), ORANGE, "M1 MAP")):
        model = model_from(*params)
        onset = float(model.binodal_temperature(p_bar, x))
        window = onset - float(model.spinodal_temperature(p_bar, x))
        below = np.geomspace(0.01, 0.995 * window, 56)
        t = onset - below
        table = np.asarray(model.nucleation_barrier(t, p_bar, x, R * t, 1.0).energy) / (R * t)
        # Widths are measured on the Fig. 1c profile at 10 GPa and 2500 K.
        kappa_ref = float(model.gradient_coefficient(2500.0, 1e5, 0.28e-9))
        bands = []
        for v in MOLAR_VOLUMES:
            for criterion in CRITERIA:
                delays = []
                for w in widths:
                    kappa = kappa_ref * (w / 0.28e-9) ** 2
                    over_kt = AVOGADRO * (kappa / (R * t)) ** 1.5 * table / v
                    good = np.isfinite(over_kt) & (over_kt > 0)
                    lo = np.log(over_kt[good])
                    if lo[0] > np.log(criterion) > lo[-1]:
                        delays.append(
                            np.exp(np.interp(-np.log(criterion), -lo, np.log(below[good])))
                        )
                    else:
                        delays.append(window if lo[-1] >= np.log(criterion) else np.nan)
                bands.append(delays)
        bands = np.array(bands)
        ax.fill_between(
            widths * 1e9, np.nanmin(bands, 0), np.nanmax(bands, 0), color=color, alpha=0.25, lw=0
        )
        ax.plot(widths * 1e9, np.nanmedian(bands, 0), color=color, label=name)
        ax.axhline(window, color=color, lw=0.8, ls="--")
        ax.text(0.155, window * 1.12, f"spinodal window {window:.0f} K", color=MUTED, fontsize=6.5)
    ax.axvspan(0.28, 0.33, color=MUTED, alpha=0.2, lw=0)
    ax.text(0.34, 0.25, "DFT width\n(Fig. 1c)", color=MUTED, fontsize=6.5)
    ax.set_xscale("log")
    # Plain major labels only: the default log minor labels overlap at this size.
    ax.set_xticks([0.2, 0.3, 0.5, 1, 2, 4, 6])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_yscale("log")
    ax.set_xlabel("interface width at 10 GPa, 2500 K (nm)")
    ax.set_ylabel("delay below binodal (K)")
    ax.grid(color=GRID, lw=0.5, which="major")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig5_3_nucleation_delay.pdf")
    plt.close(fig)


if __name__ == "__main__":
    parcel_figure()
    interior_figure()
    nucleation_figure()
    print("wrote", *sorted(FIGURES.glob("fig5_*.pdf")))
