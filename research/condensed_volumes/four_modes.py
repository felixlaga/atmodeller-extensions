"""The Hakim et al. (2026) MEB grid in four modes, plus the published pressures held fixed

Modes: ideal or real gases, each without and with the Holland & Powell (2011) volume of SiO2
liquid. All solve the MEB pressure from the weight of the gas, from a cold start. A fifth mode
keeps the published pressure of each case (the real-gas solution without volumes) fixed and adds
the volume, which shows the shift of the published composition before any pressure feedback.
The gas of the published cases, and of the fifth mode, is then put in a closed 920 K, 10 mbar
parcel with condensates, to compare the SiH4/CH4 that a spectrum would see. This parcel follows
the earlier hydrogen-storage study on this fork. Hakim et al. (2026) instead took the 1 wt% H gas
only and solved the pressure over 1 mbar to 100 bar for a scaled atmospheric mass.

    uv run python research/condensed_volumes/four_modes.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from common import HYDROGEN_GRID, OUT, budgets, meb_model, parcel_model, summarise

from atmodeller.thermodata.holland_powell import LIQUID_VOLUMES

HP11 = LIQUID_VOLUMES["O2Si"]
METALLICITIES: tuple[int, ...] = (1, 100)


def table(name: str, s: dict, h: np.ndarray, z: np.ndarray, mass: dict) -> pd.DataFrame:
    x = s["x"]
    return pd.DataFrame(
        {
            "mode": name,
            "H_wt_percent": h,
            "Z": z,
            "success": s["success"],
            "P_GPa": s["pressure_bar"] / 1e4,
            "gas_Si_fraction_of_budget": s["gas_element_mass"]["Si"] / mass["Si"],
            "silica_activity": s["silica_activity"],
            "x_H2": x["H2_g"],
            "x_H2O": x["H2O_g"],
            "x_SiO": x["OSi_g"],
            "x_SiH4": x["H4Si_g"],
            "x_CH4": x["CH4_g"],
            "H2O_over_H2": x["H2O_g"] / x["H2_g"],
            "log10_fO2": np.log10(s["f"]["O2_g"]),
        }
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    # Every fifth H value of the authors' grid, and its last, 1 wt%
    grid = HYDROGEN_GRID[np.r_[0:50:5, 49]]
    h = np.tile(grid, len(METALLICITIES))
    z = np.repeat(METALLICITIES, grid.size)
    n = h.size
    mass = budgets(h, z)
    frames = []
    published = None
    inventory: dict[str, dict] = {}
    for real_gas in (False, True):
        for volume in (None, HP11):
            name = f"{'real' if real_gas else 'ideal'} gas, {'with' if volume else 'no'} volume"
            s = summarise(
                meb_model(mass, volume=volume, real_gas=real_gas).solve_with_default(), n
            )
            frames.append(table(name, s, h, z, mass))
            if real_gas and volume is None:
                published = s["pressure_bar"]
                inventory["published"] = s["gas_element_mass"]
            print(f"{name}: {int(s['success'].sum())}/{n} converged", flush=True)

    name = "real gas, with volume, published P fixed"
    s = summarise(meb_model(mass, volume=HP11, pressure=published).solve_with_default(), n)
    frames.append(table(name, s, h, z, mass))
    inventory["volume"] = s["gas_element_mass"]
    print(f"{name}: {int(s['success'].sum())}/{n} converged", flush=True)

    parcel = {}
    stable = {}
    for key, elements in inventory.items():
        total = sum(elements.values())
        output = parcel_model({e: v / total for e, v in elements.items()}).solve_with_default()
        out = summarise(output, n)
        data = output.to_dict(to_numpy=True)
        # Condensates at unit activity are stable; record them per case
        solids = [k for k in data if k.endswith("_s")]
        activity = {
            k: np.ravel(np.asarray(data[k]["species"]["activity"][k], dtype=float)) for k in solids
        }
        stable[key] = [
            "+".join(k[:-2] for k in solids if abs(activity[k][i] - 1) < 1e-6) or "none"
            for i in range(n)
        ]
        parcel[key] = out
        print(f"parcel {key}: {int(out['success'].sum())}/{n} converged", flush=True)

    result = pd.concat(frames, ignore_index=True)
    result.to_csv(OUT / "four_modes.csv", index=False)
    summary = (
        result.groupby("mode", sort=False)
        .agg(
            converged=("success", "sum"),
            P_GPa_min=("P_GPa", "min"),
            P_GPa_max=("P_GPa", "max"),
            gas_Si_min=("gas_Si_fraction_of_budget", "min"),
            gas_Si_max=("gas_Si_fraction_of_budget", "max"),
            silica_activity_min=("silica_activity", "min"),
        )
        .reset_index()
    )
    print(summary.to_string(index=False))

    base = result[result["mode"] == "real gas, no volume"].reset_index(drop=True)
    fixed = result[result["mode"] == name].reset_index(drop=True)
    shift = pd.DataFrame(
        {
            "H_wt_percent": base.H_wt_percent,
            "Z": base.Z,
            "P_GPa": base.P_GPa,
            "SiO_factor": fixed.x_SiO / base.x_SiO,
            "SiH4_factor": fixed.x_SiH4 / base.x_SiH4,
            "H2O_over_H2_factor": fixed.H2O_over_H2 / base.H2O_over_H2,
            "delta_log10_fO2": fixed.log10_fO2 - base.log10_fO2,
            "parcel_SiH4_CH4_published": parcel["published"]["x"]["H4Si_g"]
            / parcel["published"]["x"]["CH4_g"],
            "parcel_SiH4_CH4_volume": parcel["volume"]["x"]["H4Si_g"]
            / parcel["volume"]["x"]["CH4_g"],
            "parcel_success": parcel["published"]["success"] & parcel["volume"]["success"],
            "parcel_stable_published": stable["published"],
            "parcel_stable_volume": stable["volume"],
        }
    )
    shift["parcel_delta_dex"] = np.log10(
        shift.parcel_SiH4_CH4_volume / shift.parcel_SiH4_CH4_published
    )
    shift.to_csv(OUT / "published_pressure_shift.csv", index=False)
    print(shift.round(3).to_string(index=False))
    (OUT / "four_modes_summary.json").write_text(
        json.dumps(summary.to_dict(orient="records"), indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
