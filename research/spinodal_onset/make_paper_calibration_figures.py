# SPDX-License-Identifier: Apache-2.0
"""Recalibration and design figures for the research draft.

Called by make_paper_figures.py once calibration.py and design.py have run.
"""

import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from research.spinodal_onset.calibration import (
    CLASSES,
    coexistence_runs,
    figure_values,
    homogeneous_data,
)
from research.spinodal_onset.make_paper_figures import FULL, OUT, label, save
from research.spinodal_onset.mixing_models import published_gilmore26, weight_to_z

RUN_LABELS = {
    ("M1", "table"): "M1, all data",
    ("M1", "figure"): "M1, Fig. 2a compositions",
    ("M1", "inflated"): r"M1, near-critical $\sigma_x$ = 0.15",
    ("M1", "no_volume"): "M1, without volumes",
    ("M1", "phase_comp"): "M1, phases + compositions",
    ("M1", "phase_hom"): "M1, phases + homogeneous",
    ("M2", "table"): "M2, independent H, S, V",
    ("M3", "table"): "M3, Redlich–Kister order 2",
    ("M4", "table"): "M4, Redlich–Kister order 3",
}
RUN_COLOURS = {"M1": "#c0392b", "M2": "#7d3c98", "M3": "#1f4e9c", "M4": "#16a085"}
PUBLISHED = "k"


def summary():
    return json.loads((OUT / "calibration_summary.json").read_text())


def samples():
    return pd.read_csv(OUT / "calibration_samples.csv")


def model_of(row):
    cls = CLASSES[row["model"]]
    return cls.build([row[k] for k in cls.labels])


def map_model(run):
    cls = CLASSES[run["model"]]
    return cls.build([run["map"][k] for k in cls.labels])


def audit():
    s = summary()
    runs = {(r["model"], r["version"]): r for r in s["runs"]}
    two, one = coexistence_runs()
    fig_vals = figure_values()
    pub = published_gilmore26()
    m1 = map_model(runs[("M1", "table")])
    fig = plt.figure(figsize=(FULL, 4.9))
    gs = fig.add_gridspec(2, 6, hspace=0.42, wspace=1.1, height_ratios=(1.15, 1))

    # (a) phase states and critical curves
    ax = fig.add_subplot(gs[0, :3])
    ps = np.linspace(0, 12, 61)
    ax.plot(ps, [pub.critical_point(p)[0] for p in ps], color=PUBLISHED, label="published fit")
    draws = samples()
    draws = draws[(draws.model == "M1") & (draws.version == "table")].sample(120, random_state=2)
    band = np.array([[model_of(r).critical_point(p)[0] for p in ps] for _, r in draws.iterrows()])
    lo, mid, hi = np.percentile(band, [16, 50, 84], axis=0)
    ax.fill_between(ps, lo, hi, color=RUN_COLOURS["M1"], alpha=0.25, lw=0)
    ax.plot(ps, mid, color=RUN_COLOURS["M1"], ls="--", label="M1 refit, all data (68%)")
    for r in two:
        ax.errorbar(r["p"], r["t"], xerr=r["sp"], fmt="o", ms=5, color="k", mfc="k", lw=0.8)
    for r in one:
        ax.errorbar(r["p"], r["t"], xerr=r["sp"], fmt="o", ms=5, color="k", mfc="w", lw=0.8)
    ax.plot([], [], "o", color="k", label="two phases (simulation)")
    ax.plot([], [], "o", color="k", mfc="w", label="one phase (simulation)")
    ax.set(
        yscale="log",
        ylim=(2000, 9000),
        xlim=(0, 12),
        xlabel="pressure (GPa)",
        ylabel="temperature (K)",
    )
    ax.set_yticks([2000, 3000, 4000, 6000, 8000], ["2000", "3000", "4000", "6000", "8000"])
    ax.minorticks_off()
    ax.legend(
        loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, fontsize=6.4, columnspacing=1.0
    )
    label(ax, "(a)")

    # (b) composition residuals
    ax = fig.add_subplot(gs[0, 3:])
    names = []
    for i, r in enumerate(two):
        names.append(f"{r['p']:.2f} GPa\n{r['t']:.0f} K")
        key = 3500 if r["t"] == 3500 else (-3000 if r["t"] == 3000 and r["p"] > 4 else None)
        for end, obs, side in (("lo", r["lo"], -0.2), ("hi", r["hi"], 0.2)):
            for model, colour, dx in ((pub, PUBLISHED, -0.07), (m1, RUN_COLOURS["M1"], 0.07)):
                gap = model.binodal(r["t"], r["p"])
                if gap is None:
                    zs = np.linspace(0.01, 0.99, 197)
                    pred = float(zs[np.argmin([model.d2f(v, r["t"], r["p"]) for v in zs])])
                else:
                    pred = gap[0] if end == "lo" else gap[1]
                ax.plot(i + side + dx, pred - obs, "o", ms=4.2, color=colour)
                if key is not None:
                    alt = fig_vals[key][0 if end == "lo" else 1]
                    ax.plot(i + side + dx, pred - alt, "^", ms=3.6, color=colour, mfc="w", mew=0.8)
        ax.axvline(i + 0.5, color="0.85", lw=0.5)
    ax.axhspan(-0.05, 0.05, color="0.88", lw=0)
    ax.axhline(0, color="0.5", lw=0.6)
    ax.set_xticks(range(len(names)), names, fontsize=6.6)
    ax.set(ylabel="model $-$ simulation ($x$)", xlim=(-0.5, len(names) - 0.5), ylim=(-0.3, 0.32))
    ax.text(
        0.02,
        0.06,
        "left: H-poor phase, right: H-rich phase",
        transform=ax.transAxes,
        fontsize=6.2,
        color="0.35",
    )
    ax.plot([], [], "o", color=PUBLISHED, label="published fit")
    ax.plot([], [], "o", color=RUN_COLOURS["M1"], label="M1 refit, all data")
    ax.plot([], [], "^", color="0.3", mfc="w", label="vs Fig. 2a value")
    ax.legend(
        loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3, fontsize=6.4, columnspacing=1.0
    )
    label(ax, "(b)")

    # (c-e) homogeneous excess properties with model curves
    states, values, cov = homogeneous_data()
    sd = np.sqrt(np.diag(cov))
    zs = np.linspace(0.001, 0.999, 300)
    p = np.mean([st["p"] for st in states])
    m2 = map_model(runs[("M2", "table")])
    panels = (
        ("H", r"excess enthalpy (kJ mol$^{-1}$)", 1e-3),
        ("S", r"excess entropy (J mol$^{-1}$ K$^{-1}$)", 1.0),
        ("V", r"excess volume (cm$^{3}$ mol$^{-1}$)", 1.0),
    )
    for j, (kind, ylabel, scale) in enumerate(panels):
        ax = fig.add_subplot(gs[1, 2 * j : 2 * j + 2])
        for model, colour, ls, name in (
            (pub, PUBLISHED, "-", "published fit"),
            (m1, RUN_COLOURS["M1"], "--", "M1 refit"),
            (m2, RUN_COLOURS["M2"], ":", "M2 refit"),
        ):
            f = {"H": lambda z: model.enthalpy(z, p), "S": model.entropy, "V": model.volume_cm3}[
                kind
            ]
            ax.plot(zs, [scale * f(z) for z in zs], color=colour, ls=ls, label=name)
        ax.errorbar(
            [st["z"] for st in states],
            scale * values[3 * j : 3 * j + 3],
            yerr=scale * sd[3 * j : 3 * j + 3],
            fmt="o",
            ms=3.5,
            color="k",
            capsize=1.5,
            lw=0.8,
        )
        ax.set(xlabel=r"H$_2$ mole fraction $x$", ylabel=ylabel, xlim=(0, 1))
        label(ax, "(" + "cde"[j] + ")")
        if j == 0:
            ax.legend(loc="lower center", fontsize=6.2)
    save(fig, "fig3_3_source_audit")


def posteriors():
    s = summary()
    d = samples()
    order = [k for k in RUN_LABELS if any((r["model"], r["version"]) == k for r in s["runs"])]
    chi = {(r["model"], r["version"]): r for r in s["runs"]}
    pub = s["published_fit"]["derived"]
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(FULL, 2.9),
        sharey=True,
        gridspec_kw=dict(wspace=0.08, width_ratios=(1.35, 1, 1)),
    )
    quantities = (
        ("interval_K", "metastable interval at 3 wt% (K)", pub["interval_K"]),
        ("T_C_K", "critical temperature at 2 GPa (K)", pub["T_C_K"]),
        (
            "H_rich_fraction_at_spinodal",
            "H withheld at spinodal (%)",
            100 * pub["H_rich_fraction_at_spinodal"],
        ),
    )
    for ax, (key, xlabel, ref) in zip(axes, quantities):
        for i, k in enumerate(order):
            v = d[(d.model == k[0]) & (d.version == k[1])][key].to_numpy(float)
            v = v[np.isfinite(v)] * (100 if key.startswith("H_") else 1)
            if not len(v):
                continue
            q = np.percentile(v, [2.5, 16, 50, 84, 97.5])
            c = RUN_COLOURS[k[0]]
            ax.plot([q[0], q[4]], [i, i], color=c, lw=0.8)
            ax.plot([q[1], q[3]], [i, i], color=c, lw=3.4, solid_capstyle="butt")
            ax.plot(q[2], i, "o", ms=3.8, color="w", mec=c, mew=1.0)
        ax.axvline(ref, color=PUBLISHED, lw=0.9, ls="--")
        ax.set(xlabel=xlabel)
        if key == "T_C_K":
            ax.axvspan(3500, 4000, color="0.9", lw=0, zorder=0)
    reduced = {"no_volume", "phase_comp", "phase_hom"}  # fewer data: chi2 not comparable
    axes[0].set_yticks(
        range(len(order)),
        [
            f"{RUN_LABELS[k]}  ($\\chi^2$ = {chi[k]['map_chi2_total']:.0f}"
            + ("$^\\dagger$)" if k[1] in reduced else ")")
            for k in order
        ],
        fontsize=6.6,
    )
    axes[0].invert_yaxis()
    axes[0].text(
        pub["interval_K"] - 6,
        len(order) - 0.75,
        "published",
        fontsize=6.6,
        ha="right",
        va="center",
    )
    axes[1].text(
        3750,
        len(order) - 1.2,
        "bracket from\n1.65–1.89 GPa\nruns",
        fontsize=6,
        ha="center",
        va="center",
        color="0.35",
    )
    for i, ax in enumerate(axes):
        label(ax, f"({'abc'[i]})")
    save(fig, "fig3_4_posteriors")


def bands():
    d = samples()
    ws = np.geomspace(0.6, 9.0, 36)
    pub = published_gilmore26()
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.6), sharey=True, gridspec_kw=dict(wspace=0.06))
    sets = (
        (("M1", "table"), ("M1", "phase_comp"), ("M1", "phase_hom")),
        (("M2", "table"), ("M3", "table"), ("M4", "table")),
    )
    colours = (("#c0392b", "#e67e22", "#2c3e50"), ("#7d3c98", "#1f4e9c", "#16a085"))
    for ax, keys, cols in zip(axes, sets, colours):
        ax.plot(
            ws,
            [pub.binodal_temperature(weight_to_z(w / 100), 2.0) for w in ws],
            color=PUBLISHED,
            lw=1.2,
            label="published fit",
        )
        ax.plot(
            ws,
            [pub.spinodal_temperature(weight_to_z(w / 100), 2.0) for w in ws],
            color=PUBLISHED,
            lw=1.2,
            ls="--",
        )
        for key, c in zip(keys, cols):
            e = d[(d.model == key[0]) & (d.version == key[1])]
            if not len(e):
                continue
            e = e.sample(min(150, len(e)), random_state=3)
            tb, ts = [], []
            for _, r in e.iterrows():
                m = model_of(r)
                tb.append([m.binodal_temperature(weight_to_z(w / 100), 2.0) for w in ws])
                ts.append([m.spinodal_temperature(weight_to_z(w / 100), 2.0) for w in ws])
            for arr, ls in ((np.array(tb), "-"), (np.array(ts), "--")):
                enough = np.mean(np.isfinite(arr), axis=0) >= 0.8
                arr = np.where(enough, arr, np.nan)
                lo, mid, hi = np.nanpercentile(arr, [16, 50, 84], axis=0)
                ax.fill_between(ws, lo, hi, color=c, alpha=0.18, lw=0)
                ax.plot(
                    ws, mid, color=c, ls=ls, lw=1.0, label=RUN_LABELS[key] if ls == "-" else None
                )
        ax.axvline(3, color="0.3", lw=0.6, ls=":")
        ax.set(
            xscale="log",
            xlim=(0.6, 9),
            ylim=(1500, 3800),
            xlabel="local hydrogen mass fraction (wt%)",
        )
        ax.set_xticks([1, 2, 3, 5, 9], ["1", "2", "3", "5", "9"])
        ax.minorticks_off()
        ax.legend(loc="lower right", fontsize=6.4)
    axes[0].set_ylabel("temperature at 2 GPa (K)")
    label(axes[0], "(a)")
    label(axes[1], "(b)")
    save(fig, "fig3_5_posterior_phase_diagrams")


def design():
    h = pd.read_csv(OUT / "design_homogeneous.csv")
    c = pd.read_csv(OUT / "design_curvature.csv")
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.5), gridspec_kw=dict(wspace=0.28))
    ax = axes[0]
    colours = dict(RUN_COLOURS, **{"pooled (equal class weights)": "k"})
    for g, e in h[h.precision == "ED Table 2 precision"].groupby("group"):
        col = colours.get(g, "k")
        pooled = g.startswith("pooled")
        ax.plot(
            e.composition_x,
            e.expected_sd_K,
            color=col,
            lw=1.8 if pooled else 1.0,
            label="all classes pooled" if pooled else g,
        )
        ax.axhline(e.current_sd_K.iloc[0], color=col, lw=0.6, ls=":")
    e = h[(h.precision == "twice as precise") & (h.group.str.startswith("pooled"))]
    ax.plot(
        e.composition_x,
        e.expected_sd_K,
        color="k",
        ls="--",
        lw=1.0,
        label="pooled, twice as precise",
    )
    ax.set(
        xlabel=r"composition of one new homogeneous simulation ($x$)",
        ylabel=r"expected SD of $T_{\rm sp}$ at 3 wt% (K)",
        xlim=(0, 1),
        ylim=(0, None),
    )
    ax.legend(
        fontsize=6.4, loc="upper center", bbox_to_anchor=(0.5, 0.9), ncol=3, columnspacing=0.9
    )
    label(ax, "(a)")
    ax = axes[1]
    for g, e in c.groupby("group"):
        col = colours.get(g, "k")
        ax.plot(
            e.temperature_K,
            e.expected_sd_K,
            "o-",
            ms=3,
            color=col,
            lw=1.6 if "pooled" in g else 1.0,
        )
    ax.set(
        xlabel="temperature of the curvature measurement (K)",
        ylabel=r"expected SD of $T_{\rm sp}$ (K)",
        ylim=(0, None),
    )
    label(ax, "(b)")
    ax.text(
        0.97,
        0.95,
        r"10% measurement of $\partial^2 g/\partial x^2$"
        "\n"
        r"at 3 wt% H and 2 GPa",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7,
    )
    save(fig, "fig4_1_design")


def main():
    warnings.filterwarnings("ignore", "All-NaN slice", RuntimeWarning)
    audit()
    posteriors()
    bands()
    if (OUT / "design_homogeneous.csv").exists():
        design()
