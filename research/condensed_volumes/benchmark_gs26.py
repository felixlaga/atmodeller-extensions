"""How much silicate does the MEB gas hold, against the simulations of Gilmore & Stixrude (2026)?

Gilmore & Stixrude (2026, doi:10.1038/s41586-025-09970-4) simulate H2-MgSiO3 by first-principles
molecular dynamics. Their Extended Data Table 1 gives the atoms of the H-rich phase in each
two-phase run. Its silicate-to-hydrogen mass ratio, (Mg + Si + O) / H, is compared here with
(Si + O) / H in the MEB gas of the Hakim et al. (2026) model at the same T and P, held fixed.

The systems differ: the simulations have MgSiO3, the MEB model SiO2 liquid and no Mg gas. The
H-rich phase of the 2.05 GPa run holds no Mg, so there the two compare directly. The atom counts
are small (2 Si at 2.05 GPa) and uncertain by a few percent (table footnote), so the reference
ratios carry tens of percent of uncertainty.

The published fit to the simulations (A = 75 kJ/mol, B = 424 kJ/mol, 4400 K, -35 GPa) gives a
much leaner H-rich branch at 2.05 GPa than the simulation itself, as the audit of the fit on the
spinodal branch of this fork found. Its values are also listed (FIT), from
SubregularSolution.hydrogen_silicate_gilmore26().coexistence on that branch (commit a82879e2).

    uv run python research/condensed_volumes/benchmark_gs26.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from common import OUT, budgets, meb_model, summarise

from atmodeller.eos import get_eos_models
from atmodeller.thermodata.holland_powell import LIQUID_VOLUMES

MASS = {"H": 1.008, "Mg": 24.305, "O": 15.999, "Si": 28.085}
# Two-phase runs of Extended Data Table 1: P (GPa), T (K), atoms H/Mg/O/Si of the H-rich phase
SIMULATIONS: list[tuple[float, float, tuple[int, int, int, int]]] = [
    (2.05, 3000.0, (96, 0, 5, 2)),
    (5.14, 3000.0, (71, 6, 23, 8)),
    (9.58, 2500.0, (166, 1, 3, 1)),
]
# H-rich branch of the published fit at the same (P, T): native mole fraction of H2
FIT_H_RICH_X_H2: dict[tuple[float, float], float] = {
    (2.05, 3000.0): 0.99482,
    (5.14, 3000.0): 0.969695,
    (9.58, 2500.0): 0.980021,
}


def simulation_ratio(atoms: tuple[int, int, int, int]) -> float:
    h, mg, o, si = atoms
    return (mg * MASS["Mg"] + o * MASS["O"] + si * MASS["Si"]) / (h * MASS["H"])


def fit_ratio(x_h2: float) -> float:
    return (1 - x_h2) * 100.389 / (x_h2 * 2.016)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    eos = get_eos_models()
    hp11 = LIQUID_VOLUMES["O2Si"]
    proxy = {"OSi": eos["CO_cs_shi92"]}
    variants = {
        "published": (None, None),
        "SiO2 liquid volume": (hp11, None),
        "SiO as CO": (None, proxy),
        "both": (hp11, proxy),
    }
    pressure = np.array([p for p, _, _ in SIMULATIONS])
    temperature = np.array([t for _, t, _ in SIMULATIONS])
    n = pressure.size
    # The composition of the gas at fixed P barely depends on the H budget; 0.3 wt% H, 1x solar
    mass = budgets(np.full(n, 0.3), np.full(n, 1.0))
    rows = []
    for name, (volume, gas_activity) in variants.items():
        model = meb_model(
            mass,
            volume=volume,
            temperature=temperature,
            pressure=pressure * 1e4,
            gas_activity=gas_activity,
        )
        s = summarise(model.solve_with_default(), n)
        m = s["gas_element_mass"]
        ratio = (m["Si"] + m["O"]) / m["H"]
        for i, (p, t, atoms) in enumerate(SIMULATIONS):
            reference = simulation_ratio(atoms)
            rows.append(
                {
                    "variant": name,
                    "P_GPa": p,
                    "T_K": t,
                    "success": bool(s["success"][i]),
                    "silicate_over_H": float(ratio[i]),
                    "simulation_silicate_over_H": reference,
                    "fit_silicate_over_H": fit_ratio(FIT_H_RICH_X_H2[(p, t)]),
                    "log10_model_over_simulation": float(np.log10(ratio[i] / reference)),
                }
            )
        print(f"{name}: {int(s['success'].sum())}/{n} converged", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "benchmark_gs26.csv", index=False)
    wide = table.pivot_table(
        index=["P_GPa", "T_K"], columns="variant", values="silicate_over_H", sort=False
    )[list(variants)]
    reference = table.drop_duplicates(["P_GPa", "T_K"]).set_index(["P_GPa", "T_K"])
    wide.insert(0, "fit", reference.fit_silicate_over_H)
    wide.insert(0, "simulation", reference.simulation_silicate_over_H)
    print("Silicate-to-hydrogen mass ratio of the H-rich phase or gas:")
    print(wide.round(2).to_string())
    misfit = table.groupby("variant", sort=False).log10_model_over_simulation.apply(
        lambda v: float(np.mean(np.abs(v)))
    )
    print("mean |log10(model / simulation)|:")
    print(misfit.round(2).to_string())
    (OUT / "benchmark_gs26.json").write_text(
        json.dumps({"mean_abs_log10_misfit": misfit.to_dict()}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
