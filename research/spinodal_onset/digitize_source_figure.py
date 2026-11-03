# SPDX-License-Identifier: Apache-2.0
"""Digitize Gilmore & Stixrude (2026) Fig. 2a and compare it with our model.

Two questions: (1) does our implementation reproduce the published binodal
curves, and (2) do the plotted simulation compositions equal Extended Data
Table 1? Axes are calibrated from detected tick marks, markers from colour
masks after a morphological opening (which removes lines and error bars).
Run from the repository root.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from research.spinodal_onset.mixing_models import published_gilmore26

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "sources/GS26_Fig2.png"
SHA256 = "cd63521616c8fb9c783c10a822ef31397a03143c8d81cb8b61250f4833c2a72c"
PANEL_A = 640  # rows of panel a
SERIES = {  # colour, tolerance, curve pressure (GPa)
    "red": ((255, 0, 0), 40, 2.0),
    "grey": ((136, 136, 136), 12, None),
    "black": ((0, 0, 0), 60, 5.0),
    "blue": ((0, 62, 255), 40, 10.0),
}
TABLE = {  # ED Table 1 (x of H-poor, H-rich) for the runs plotted in Fig. 2a
    ("red", 3000): (0.209, 0.931),
    ("red", 3500): (0.458, 0.576),
    ("grey", 3000): (0.587, 0.706),
    ("blue", 2500): (0.316, 0.976),
}


def groups(indices):
    out, cur = [], [indices[0]]
    for i in indices[1:]:
        if i - cur[-1] <= 2:
            cur.append(i)
        else:
            out.append(cur)
            cur = [i]
    return out + [cur]


def calibrate(gray):
    """Linear maps pixel -> x and pixel -> T from the labelled ticks."""
    ink = gray < 230
    axis_row = max(range(540, 600), key=lambda r: ink[r, 200:880].sum())
    axis_col = max(range(120, 150), key=lambda c: ink[60:500, c].sum())
    ticks_x = []
    for g in groups(np.where(ink[axis_row - 12 : axis_row - 1].sum(0) >= 8)[0]):
        if len(g) <= 3:  # inner ticks only; edge groups merge with curves
            w = (255 - gray[axis_row - 12 : axis_row - 1, g]).sum(0)
            ticks_x.append(float((np.array(g) * w).sum() / w.sum()))
    ticks_t = []
    for g in groups(np.where(ink[:PANEL_A, axis_col + 1 : axis_col + 12].sum(1) >= 8)[0]):
        if len(g) <= 3:
            w = (255 - gray[g, axis_col + 1 : axis_col + 12]).sum(1)
            ticks_t.append(float((np.array(g) * w).sum() / w.sum()))
    assert len(ticks_x) == 4 and len(ticks_t) == 5, (ticks_x, ticks_t)
    px = np.polyfit(ticks_x, [0.2, 0.4, 0.6, 0.8], 1)
    pt = np.polyfit(ticks_t, [4000, 3500, 3000, 2500, 2000], 1)
    rx = np.polyval(px, ticks_x) - [0.2, 0.4, 0.6, 0.8]
    rt = np.polyval(pt, ticks_t) - [4000, 3500, 3000, 2500, 2000]
    return px, pt, float(np.abs(rx).max()), float(np.abs(rt).max())


def main():
    data = SOURCE.read_bytes()
    assert hashlib.sha256(data).hexdigest() == SHA256, "Source figure changed"
    rgb = np.asarray(Image.open(SOURCE).convert("RGB")).astype(int)[:PANEL_A]
    px, pt, rx, rt = calibrate(rgb.mean(-1))
    yy, xx = np.mgrid[-10:11, -10:11]
    disk = xx**2 + yy**2 <= 100
    markers, keepout = [], np.zeros(rgb.shape[:2], bool)
    for name, (colour, tol, _) in SERIES.items():
        if name == "black":
            continue
        mask = np.abs(rgb - colour).max(-1) <= tol
        lab, n = ndi.label(ndi.binary_opening(mask, structure=disk))
        for i in range(1, n + 1):
            rows, cols = np.where(lab == i)
            if len(rows) < 600:
                continue
            cx, cy = cols.mean(), rows.mean()
            markers.append(
                dict(
                    series=name,
                    x=float(np.polyval(px, cx)),
                    temperature_K=float(np.polyval(pt, cy)),
                )
            )
            keepout[int(cy) - 26 : int(cy) + 27, int(cx) - 26 : int(cx) + 27] = True
            keepout[int(cy) - 4 : int(cy) + 5, int(cx) - 80 : int(cx) + 81] = True  # error bars
    markers.sort(key=lambda m: (m["series"], m["temperature_K"], m["x"]))
    comparison = []
    for m in markers:
        key = (m["series"], int(round(m["temperature_K"] / 100) * 100))
        pair = TABLE[key]
        m["table_x"] = pair[0] if m["x"] < sum(pair) / 2 else pair[1]
        m["figure_minus_table"] = m["x"] - m["table_x"]
        comparison.append(abs(m["figure_minus_table"]))

    # Curves: column-wise mean row of thin curve pixels, away from markers/text.
    model = published_gilmore26()
    zgrid = np.linspace(0.03, 0.995, 194)
    tb0 = np.array([model.binodal_temperature(z, 0.0) for z in zgrid])
    zgrid, tb0 = zgrid[np.isfinite(tb0)], tb0[np.isfinite(tb0)]  # flat crest: T_b ~ T_C
    curves = {}
    for name, (colour, tol, pressure) in SERIES.items():
        if pressure is None:
            continue
        mask = (np.abs(rgb - colour).max(-1) <= tol) & ~keepout
        mask[:, :150] = False  # y axis and tick labels
        mask[555:, :] = False  # x axis
        if name == "black":  # drop text labels: keep only long components
            lab, n = ndi.label(mask, structure=np.ones((3, 3)))
            sizes = ndi.sum(mask, lab, range(1, n + 1))
            mask = np.isin(lab, 1 + np.where(sizes > 2000)[0])
        samples = []
        for c in range(150, 1030):
            rows = np.where(mask[:, c])[0]
            if 3 <= len(rows) <= 12 and np.ptp(rows) <= 12:  # single, non-vertical stroke
                x, t = float(np.polyval(px, c)), float(np.polyval(pt, rows.mean()))
                if 0.08 <= x <= 0.95:
                    t_model = float(np.interp(x, zgrid, tb0)) * (1 + pressure / -35.0)
                    samples.append((x, t, t - t_model))
        res = np.array([s[2] for s in samples])
        curves[name] = dict(
            pressure_GPa=pressure,
            n_columns=len(samples),
            median_residual_K=float(np.median(res)),
            max_abs_residual_K=float(np.abs(res).max()),
            rms_residual_K=float(np.sqrt(np.mean(res**2))),
            samples=[dict(x=x, T_digitized=t, T_minus_model=r) for x, t, r in samples[::10]],
        )

    result = dict(
        source=SOURCE.name,
        sha256=SHA256,
        tick_fit_max_error=dict(x=rx, temperature_K=rt),
        kelvin_per_pixel=float(-pt[0]),
        x_per_pixel=float(px[0]),
        markers=markers,
        wide_gap_marker_max_abs_difference_from_table=max(
            abs(m["figure_minus_table"])
            for m in markers
            if (m["series"], round(m["temperature_K"], -2)) in (("red", 3000), ("blue", 2500))
        ),
        curves=curves,
        note=(
            "Curves are compared with our implementation of the published model at "
            "exactly 2, 5 and 10 GPa. The line is about 6 px (~30 K) thick."
        ),
    )
    (HERE / "outputs/source_figure_digitized.json").write_text(json.dumps(result, indent=2) + "\n")
    for m in markers:
        print(
            f"{m['series']:5s} {m['temperature_K']:6.0f} K  fig x={m['x']:.4f}  "
            f"table x={m['table_x']:.3f}  diff={m['figure_minus_table']:+.4f}"
        )
    for name, c in curves.items():
        print(
            name,
            c["pressure_GPa"],
            "GPa: n=%d median %.1f K, rms %.1f K, max %.1f K"
            % (
                c["n_columns"],
                c["median_residual_K"],
                c["rms_residual_K"],
                c["max_abs_residual_K"],
            ),
        )


if __name__ == "__main__":
    main()
