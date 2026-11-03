# SPDX-License-Identifier: Apache-2.0
"""Which new simulation would most reduce the spinodal uncertainty?

Uses the recalibration posteriors (calibration_samples.csv). For a candidate
measurement y with Gaussian noise N, the linear-Gaussian expected posterior
variance of the target is Var(T) - c^T (C_yy + N)^-1 c, with c = Cov(T, y) and
C_yy = Cov(y) estimated from the posterior samples. This is a design screen,
not a replacement for refitting after a real measurement.

Candidates:
* homogeneous excess H, S and V at one new composition (any T: these model
  classes have T-independent excess enthalpy and entropy), with the precision
  of the existing Extended Data Table 2 entries or twice as good;
* the homogeneous curvature d2g/dx2 at the target composition, e.g. from the
  long-wavelength concentration structure factor S_cc(0) = RT/g'' (per mole
  of mixing units), with 10% relative precision, at fixed T and 2 GPa.

The quoted curvature measurement is the lowest screened temperature at which
every pooled sample is still one phase (3,600 K for the October 2026 samples;
at 3,400 K one M4 sample in 2,400 is already two-phase). There the simulated
parcel is a stable homogeneous state for every model, so the measurement does
not rely on a metastable liquid that may decompose during the run. At 3,200 K
about 40% of the samples are already inside the binodal. The state is also above
every sampled critical temperature, so it lies outside every miscibility gap and
constrains the spinodal only through the assumed model forms.
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from research.spinodal_onset.calibration import CLASSES
from research.spinodal_onset.mixing_models import R, weight_to_z

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
P_TARGET = 2.0
Z0 = weight_to_z(0.03)
NOISE = {  # (sigma_H J/mol, sigma_S J/mol/K, sigma_V cm3/mol)
    "ED Table 2 precision": (4000.0, 4.0, 0.15),
    "twice as precise": (2000.0, 2.0, 0.075),
}
COMPOSITIONS = np.round(np.arange(0.05, 0.951, 0.05), 3)
CURVATURE_T = (3200.0, 3400.0, 3600.0)


def expected_sd(target, ys, noise):
    """Expected posterior SD of target after observing ys (n x k) with noise (k,)."""
    t = target - target.mean()
    y = ys - ys.mean(axis=0)
    c = y.T @ t / (len(t) - 1)
    cyy = y.T @ y / (len(t) - 1) + np.diag(np.square(noise))
    var = t @ t / (len(t) - 1) - c @ np.linalg.solve(cyy, c)
    return math.sqrt(max(var, 0.0))


def main():
    samples = pd.read_csv(OUT / "calibration_samples.csv")
    samples = samples[(samples.version == "table") & np.isfinite(samples.spinodal_K)]
    rows, curvature_rows = [], []
    groups = {m: g for m, g in samples.groupby("model")}
    groups["pooled (equal class weights)"] = pd.concat(
        g.sample(n=min(len(x) for x in groups.values()), random_state=1) for g in groups.values()
    )
    for name, g in groups.items():
        models = []
        for _, r in g.iterrows():
            cls = CLASSES[r.model]
            models.append(cls.build([r[k] for k in cls.labels]))
        target = g.spinodal_K.to_numpy()
        base = float(np.std(target, ddof=1))
        for z in COMPOSITIONS:
            ys = np.array(
                [[m.enthalpy(z, P_TARGET), m.entropy(z), m.volume_cm3(z)] for m in models]
            )
            for label, noise in NOISE.items():
                rows.append(
                    dict(
                        group=name,
                        composition_x=float(z),
                        precision=label,
                        current_sd_K=base,
                        expected_sd_K=expected_sd(target, ys, np.array(noise)),
                    )
                )
        for t_m in CURVATURE_T:
            g2 = np.array([m.d2f(Z0, t_m, P_TARGET) * R * t_m for m in models])
            noise = 0.1 * np.median(np.abs(g2))
            # g'' > 0 means locally stable: either still one phase or metastable.
            one_phase = g.binodal_K.to_numpy() < t_m
            curvature_rows.append(
                dict(
                    group=name,
                    temperature_K=t_m,
                    fraction_locally_stable=float(np.mean(g2 > 0)),
                    fraction_one_phase=float(np.mean(one_phase)),
                    fraction_metastable=float(np.mean((g2 > 0) & ~one_phase)),
                    median_curvature_J_per_mol=float(np.median(g2)),
                    noise_J_per_mol=float(noise),
                    current_sd_K=base,
                    expected_sd_K=expected_sd(target, g2[:, None], np.array([noise])),
                )
            )
    design = pd.DataFrame(rows)
    design.to_csv(OUT / "design_homogeneous.csv", index=False)
    curv = pd.DataFrame(curvature_rows)
    curv.to_csv(OUT / "design_curvature.csv", index=False)
    pooled = curv[curv.group.str.startswith("pooled")].sort_values("temperature_K")
    one_phase = pooled[pooled.fraction_one_phase >= 1.0]
    quoted = (one_phase if len(one_phase) else pooled.tail(1)).iloc[0]
    best = (
        design[design.precision == "ED Table 2 precision"]
        .sort_values("expected_sd_K")
        .groupby("group")
        .head(1)
    )
    summary = dict(
        target="spinodal temperature at 3 wt% local H and 2 GPa",
        method="linear-Gaussian expected posterior SD from recalibration samples",
        best_single_homogeneous_composition=best.to_dict(orient="records"),
        curvature=curv.to_dict(orient="records"),
        curvature_measurement=dict(
            rule="pooled samples; lowest screened temperature at which every sample is one phase",
            **quoted.to_dict(),
        ),
    )
    (OUT / "design_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(best.to_string(index=False))
    print(curv.to_string(index=False))


if __name__ == "__main__":
    main()
