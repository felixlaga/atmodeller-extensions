"""Checks that the MEB model reproduces the authors' published workbooks

All 100 cases of the fully molten grid (50 H mass fractions at 1x and 100x solar) are solved from
a cold start with the current Atmodeller, without condensed volumes, and compared with the OSF
workbooks of Hakim et al. (2026). Also checks that mechanical_pressure_bar gives the solved
pressure, which boundary_map.py relies on.

    uv run python research/condensed_volumes/fetch_sources.py
    uv run python research/condensed_volumes/port_check.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from common import (
    HYDROGEN_GRID,
    OUT,
    SOURCES,
    budgets,
    mechanical_pressure_bar,
    meb_model,
    summarise,
)

METALLICITIES: tuple[int, ...] = (1, 100)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    books = [
        pd.read_excel(SOURCES / f"HHeCNOSi_magma_sol_real_{z}xsolar_melt100.xlsx", sheet_name=None)
        for z in METALLICITIES
    ]
    h = np.tile(HYDROGEN_GRID, len(METALLICITIES))
    z = np.repeat(METALLICITIES, len(HYDROGEN_GRID))
    n = h.size
    result = summarise(meb_model(budgets(h, z)).solve_with_default(), n)

    worst = {}
    for species, values in result["x"].items():
        reference = np.concatenate(
            [b[species]["volume_mixing_ratio"].to_numpy(float) for b in books]
        )
        worst[species] = float(np.max(np.abs(values / reference - 1)))
    reference_pressure = np.concatenate([b["atmosphere"].pressure.to_numpy(float) for b in books])
    check = {
        "n_cases": int(n),
        "n_success": int(result["success"].sum()),
        "max_relative_difference_mole_fraction": worst,
        "max_relative_difference_pressure": float(
            np.max(np.abs(result["pressure_bar"] / reference_pressure - 1))
        ),
        "pressure_range_GPa": [
            float(result["pressure_bar"].min() / 1e4),
            float(result["pressure_bar"].max() / 1e4),
        ],
        "max_relative_difference_mechanical_pressure": float(
            np.max(
                np.abs(
                    mechanical_pressure_bar(result["gas_element_mass"]) / result["pressure_bar"]
                    - 1
                )
            )
        ),
    }
    (OUT / "port_check.json").write_text(json.dumps(check, indent=2) + "\n")
    print(json.dumps(check, indent=2))


if __name__ == "__main__":
    main()
