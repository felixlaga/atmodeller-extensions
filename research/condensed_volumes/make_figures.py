"""Figures of the condensed-volume study, from outputs/ (run the study scripts first)

    uv run python research/condensed_volumes/make_figures.py

Colours are the validated categorical slots 1-4 of the dataviz reference palette; every line is
also labelled directly or by marker, so identity does not rest on colour alone.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from common import HERE, OUT  # noqa: E402

FIGURES = HERE / "figures"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
MUTED = "#8a8985"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
MARKERS = ("o", "s", "^")

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": TEXT,
        "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_SECONDARY,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.6,
    }
)


def boundary_map() -> None:
    data = json.loads((OUT / "boundary_map.json").read_text())
    pressure = np.asarray(data["P_GPa"])
    cases = data["cases"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
    for ax, metallicity in zip(axes, (1, 100)):
        ax.axhline(1.0, color=MUTED, lw=0.8)
        for k, hydrogen in enumerate((0.1, 0.3, 1.0)):
            i = cases.index([hydrogen, metallicity])
            for variant, style in (("HP11 x0", "--"), ("HP11 x1", "-")):
                v = data["variants"][variant]
                ratio = np.asarray(v["P_mech_over_P"][i], dtype=float)
                ratio[~np.asarray(v["success"][i])] = np.nan
                ax.plot(pressure, ratio, style, color=SERIES[k], marker=MARKERS[k], ms=3)
            for crossing in data["variants"]["HP11 x0"]["crossings"][i]:
                if crossing["stable"]:
                    ax.plot(crossing["P_GPa"], 1.0, "o", ms=6, mfc="white", mec=SERIES[k], mew=1.5)
            last = np.asarray(data["variants"]["HP11 x1"]["P_mech_over_P"][i], dtype=float)
            j = int(np.flatnonzero(np.isfinite(last))[-1])
            ax.annotate(
                f"{hydrogen:g} wt% H",
                (pressure[j], last[j]),
                xytext=(3, 0),
                textcoords="offset points",
                color=TEXT_SECONDARY,
                fontsize=8,
                va="center",
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Boundary pressure held fixed (GPa)")
        ax.set_title(f"{metallicity}x solar", color=TEXT, fontsize=9)
    axes[0].set_ylabel("Pressure the gas exerts / pressure held")
    axes[0].plot([], [], "--", color=MUTED, label="1-bar SiO2 liquid (published)")
    axes[0].plot([], [], "-", color=MUTED, label="SiO2 liquid with its volume")
    axes[0].plot([], [], "o", ms=6, mfc="white", mec=MUTED, label="Published boundary")
    axes[0].legend(frameon=False, fontsize=7.5, loc="lower left")
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(FIGURES / f"boundary_map.{suffix}", dpi=200)
    plt.close(fig)


def correction_factors() -> None:
    table = pd.read_csv(OUT / "correction_table.csv")
    t = table[table.T_K == 3000.0]
    lines = [
        ("SiO2(l) = SiO + 1/2 O2", "SiO2 liquid to SiO, SiH4 or H2O", SERIES[0], "-"),
        ("MgSiO3(l) = MgO(l) + SiO + 1/2 O2", "MgSiO3 liquid to SiO", SERIES[1], "-"),
        ("MgO(l) = Mg + 1/2 O2", "MgO liquid to Mg (Werlen 2026)", SERIES[2], "-"),
        ("FeO(l) + H2 = Fe(l) + H2O", "FeO + H2 to Fe + H2O (bracket)", SERIES[3], "--"),
        ("MgSiO3(l) = MgO(l) + SiO2(l)", "Within the melt (Werlen 2026)", MUTED, ":"),
    ]
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    ax.axhline(1.0, color=MUTED, lw=0.8)
    for reaction, label, colour, style in lines:
        r = t[t.reaction == reaction].sort_values("P_GPa")
        ax.plot(r.P_GPa, r.factor, style, color=colour, marker="o", ms=3)
        if not np.allclose(r.factor_min, r.factor_max):
            ax.fill_between(r.P_GPa, r.factor_min, r.factor_max, color=colour, alpha=0.15, lw=0)
        ax.annotate(
            label,
            (r.P_GPa.iloc[-1], r.factor.iloc[-1]),
            xytext=(4, 0),
            textcoords="offset points",
            color=TEXT_SECONDARY,
            fontsize=7.5,
            va="center",
        )
    ax.set_yscale("log")
    ax.set_xlim(0, 13.5)
    ax.set_xlabel("Pressure (GPa)")
    ax.set_ylabel("Factor on the equilibrium constant, 3000 K")
    fig.tight_layout()
    fig.subplots_adjust(right=0.62)
    for suffix in ("pdf", "png"):
        fig.savefig(FIGURES / f"correction_factors.{suffix}", dpi=200)
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    correction_factors()
    if (OUT / "boundary_map.json").exists():
        boundary_map()


if __name__ == "__main__":
    main()
