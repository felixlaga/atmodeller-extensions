"""How much extra fluid volume would SiO and SiH4 need to bring the boundary back?

With the volume of SiO2 liquid included, the TOI-421 b model has no MEB below 20 GPa for 0.3 wt% H
or more (boundary_map.py). The gas is an ideal mixture of pure real gases, so the partial molar
volumes of SiO and SiH4 in dense H2 are those of the pure gases. Here each gets a constant extra
partial molar volume dV (ExcessVolumeGas), which raises its chemical potential by dV (P - 1 bar),
and the boundary map is redone. The smallest dV that restores the boundaries, set beside the
molar volumes of the species, says how well the fluid volumes must be known to decide whether a
boundary exists. Gilmore & Stixrude (2026) find two phases at 3000 K below about 6.6 GPa.

    uv run python research/condensed_volumes/gas_volume_scan.py
"""

from __future__ import annotations

import json

import numpy as np
from boundary_map import CASES, PRESSURE_GPA, crossings
from common import (
    EOS,
    OUT,
    ExcessVolumeGas,
    budgets,
    mechanical_pressure_bar,
    meb_model,
    summarise,
)

from atmodeller.eos import get_eos_models
from atmodeller.thermodata.holland_powell import LIQUID_VOLUMES

HP11 = LIQUID_VOLUMES["O2Si"]
TEMPERATURE: float = 3000.0
# Extra partial molar volume (cm3/mol) on SiO and SiH4, and optionally H2O
VARIANTS: dict[str, tuple[float, tuple[str, ...]]] = {
    **{f"dV {v:g} SiO+SiH4": (v, ("OSi", "H4Si")) for v in (0.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0)},
    "dV 8 SiO+SiH4+H2O": (8.0, ("OSi", "H4Si", "H2O")),
}


def molar_volumes() -> dict[str, list[float]]:
    """Molar volumes (cm3/mol) at 3000 K of the pure gases and of SiO2 liquid"""
    eos = get_eos_models()
    pressures = (1e4, 2e4, 5e4, 1e5)
    table: dict[str, list[float]] = {"P_GPa": [p / 1e4 for p in pressures]}
    for species in ("H2", "H2O", "OSi", "H4Si"):
        model = eos[EOS[species]]
        try:
            table[species] = [float(model.volume(TEMPERATURE, p)) * 1e6 for p in pressures]
        except (AttributeError, NotImplementedError, TypeError):
            # Equations of state without an explicit volume: V = RT dln f/dP by a central step
            values = []
            for p in pressures:
                step = 1e-3 * p
                d = model.log_fugacity(TEMPERATURE, p + step) - model.log_fugacity(
                    TEMPERATURE, p - step
                )
                values.append(float(8.314462618 * TEMPERATURE * d / (2 * step * 1e5)) * 1e6)
            table[species] = values
    table["SiO2_l"] = [float(HP11.volume(TEMPERATURE, p)) * 10 for p in pressures]
    table["dV SiO2(l) + H2 = SiO + H2O"] = [
        table["OSi"][i] + table["H2O"][i] - table["H2"][i] - table["SiO2_l"][i]
        for i in range(len(pressures))
    ]
    return table


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    volumes = molar_volumes()
    print("Molar volumes at 3000 K (cm3/mol):")
    for key, values in volumes.items():
        print(f"  {key:28s} " + " ".join(f"{v:8.2f}" for v in values), flush=True)

    eos = get_eos_models()
    h = np.repeat([c[0] for c in CASES], PRESSURE_GPA.size)
    z = np.repeat([c[1] for c in CASES], PRESSURE_GPA.size)
    p_bar = np.tile(PRESSURE_GPA * 1e4, len(CASES))
    n = h.size
    mass = budgets(h, z)
    shape = (len(CASES), PRESSURE_GPA.size)
    result: dict = {"cases": CASES, "P_GPa": PRESSURE_GPA.tolist(), "volumes": volumes}
    result["variants"] = {}
    for name, (excess, species) in VARIANTS.items():
        gas_activity = {s: ExcessVolumeGas(eos[EOS[s]], excess) for s in species}
        model = meb_model(
            mass,
            volume=HP11,
            temperature=TEMPERATURE,
            pressure=p_bar,
            max_starts=2,
            gas_activity=gas_activity,
        )
        s = summarise(model.solve_with_default(), n)
        ratio = (mechanical_pressure_bar(s["gas_element_mass"]) / p_bar).reshape(shape)
        ok = s["success"].reshape(shape)
        result["variants"][name] = {
            "success": ok.tolist(),
            "P_mech_over_P": ratio.tolist(),
            "x_SiO_plus_SiH4": (s["x"]["OSi_g"] + s["x"]["H4Si_g"]).reshape(shape).tolist(),
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
        print(f"{name:20s} {int(ok.sum())}/{n} converged | boundary GPa: {summary}", flush=True)
        (OUT / "gas_volume_scan.json").write_text(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
