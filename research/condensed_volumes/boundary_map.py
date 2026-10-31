"""Does a magma-envelope boundary still exist once SiO2 liquid has a volume?

The MEB pressure of the Hakim et al. (2026) model is set by the weight of the gas,
P = m_gas g / A. Here the pressure is instead fixed on a grid, which removes that feedback, and
the pressure the equilibrium gas would exert, P_mech, is computed at each grid pressure. A
self-consistent boundary is a crossing of P_mech / P = 1. It is stable if P_mech / P falls
through 1 as P rises: a small rise in pressure then gives a gas too light to hold it.

Variants: the SiO2 liquid volume of Holland & Powell (2011) scaled by 0 to 1; the lower bracket
of Lange & Carmichael (1987) with K' = 4; 2500 K and 3500 K; and ideal gases. Failed rows
are restarted at most twice and marked as failed; crossings use converged neighbours only.

    uv run python research/condensed_volumes/boundary_map.py
"""

from __future__ import annotations

import json

import numpy as np
from common import (
    OUT,
    ScaledVolume,
    budgets,
    mechanical_pressure_bar,
    meb_model,
    summarise,
)
from correction_table import MODELS

from atmodeller.thermodata.holland_powell import LIQUID_VOLUMES

CASES: list[tuple[float, int]] = [(0.1, 1), (0.3, 1), (1.0, 1), (0.1, 100), (0.3, 100), (1.0, 100)]
PRESSURE_GPA: np.ndarray = np.logspace(np.log10(0.3), np.log10(20.0), 17)
HP11 = LIQUID_VOLUMES["O2Si"]

# name: (volume model or None, scale, temperature, real gas)
VARIANTS: dict[str, tuple] = {
    **{f"HP11 x{s:g}": (HP11, s, 3000.0, True) for s in (0.0, 0.1, 0.2, 0.3, 0.5, 1.0)},
    "LC87 K'=4": (MODELS["SiO2"]["LC87 K'=4"], 1.0, 3000.0, True),
    "2500 K none": (None, 1.0, 2500.0, True),
    "2500 K HP11": (HP11, 1.0, 2500.0, True),
    "3500 K none": (None, 1.0, 3500.0, True),
    "3500 K HP11": (HP11, 1.0, 3500.0, True),
    "ideal none": (None, 1.0, 3000.0, False),
    "ideal HP11": (HP11, 1.0, 3000.0, False),
}


def crossings(pressure_gpa: np.ndarray, ratio: np.ndarray, ok: np.ndarray) -> list[dict]:
    """Crossings of P_mech / P = 1 between neighbouring converged grid points"""
    found = []
    log_ratio = np.log(ratio)
    for i in range(len(pressure_gpa) - 1):
        if not (ok[i] and ok[i + 1]) or np.sign(log_ratio[i]) == np.sign(log_ratio[i + 1]):
            continue
        # Interpolate in log P
        w = log_ratio[i] / (log_ratio[i] - log_ratio[i + 1])
        log_p = np.log(pressure_gpa[i]) + w * (
            np.log(pressure_gpa[i + 1]) - np.log(pressure_gpa[i])
        )
        found.append({"P_GPa": float(np.exp(log_p)), "stable": bool(log_ratio[i] > 0)})
    return found


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    h = np.repeat([c[0] for c in CASES], PRESSURE_GPA.size)
    z = np.repeat([c[1] for c in CASES], PRESSURE_GPA.size)
    p_bar = np.tile(PRESSURE_GPA * 1e4, len(CASES))
    n = h.size
    mass = budgets(h, z)
    shape = (len(CASES), PRESSURE_GPA.size)
    result: dict = {"cases": CASES, "P_GPa": PRESSURE_GPA.tolist(), "variants": {}}
    for name, (volume, scale, temperature, real_gas) in VARIANTS.items():
        model_volume = None if volume is None or scale == 0 else ScaledVolume(volume, scale)
        model = meb_model(
            mass,
            volume=model_volume,
            real_gas=real_gas,
            temperature=temperature,
            pressure=p_bar,
            max_starts=2,
        )
        s = summarise(model.solve_with_default(), n)
        ratio = (mechanical_pressure_bar(s["gas_element_mass"]) / p_bar).reshape(shape)
        ok = s["success"].reshape(shape)
        x_si = (s["x"]["OSi_g"] + s["x"]["H4Si_g"]).reshape(shape)
        h2o_h2 = (s["x"]["H2O_g"] / s["x"]["H2_g"]).reshape(shape)
        result["variants"][name] = {
            "success": ok.tolist(),
            "P_mech_over_P": ratio.tolist(),
            "x_SiO_plus_SiH4": x_si.tolist(),
            "x_H2O_over_x_H2": h2o_h2.tolist(),
            "x_SiO": s["x"]["OSi_g"].reshape(shape).tolist(),
            "x_SiH4": s["x"]["H4Si_g"].reshape(shape).tolist(),
            "x_H2": s["x"]["H2_g"].reshape(shape).tolist(),
            "silica_activity": s["silica_activity"].reshape(shape).tolist(),
            "crossings": [crossings(PRESSURE_GPA, ratio[i], ok[i]) for i in range(len(CASES))],
        }
        summary = "; ".join(
            f"H{c[0]:g}Z{c[1]}: "
            + (
                ", ".join(
                    f"{x['P_GPa']:.2f}{'' if x['stable'] else '(u)'}"
                    for x in result["variants"][name]["crossings"][i]
                )
                or "none"
            )
            for i, c in enumerate(CASES)
        )
        print(f"{name:14s} {int(ok.sum())}/{n} converged | boundary GPa: {summary}", flush=True)
        (OUT / "boundary_map.json").write_text(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
