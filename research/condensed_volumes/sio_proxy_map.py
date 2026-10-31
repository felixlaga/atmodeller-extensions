"""Do the boundaries return when SiO is repulsive at GPa pressures, as other gases are?

The SiO equation of state of the Hakim et al. (2026) setup is the Redlich-Kwong fit of Connolly
(2016), calibrated to 5 GPa. At 3000 K it gives ln phi(SiO) of -0.8 to -1.4 between 1 and 10 GPa,
an attractive fluid, whereas CO, H2O, SiH4 and H2 have ln phi of +2 to +3.4 at 5 GPa. Here SiO
instead follows the corresponding-states equation of state of CO (Shi & Saxena 1992), a lower
bound on its repulsion, since SiO is the larger molecule. The boundary map is redone with and
without the volume of SiO2 liquid.

    uv run python research/condensed_volumes/sio_proxy_map.py
"""

from __future__ import annotations

import json

import numpy as np
from boundary_map import CASES, PRESSURE_GPA, crossings
from common import OUT, budgets, mechanical_pressure_bar, meb_model, summarise

from atmodeller.eos import get_eos_models
from atmodeller.thermodata.holland_powell import LIQUID_VOLUMES

HP11 = LIQUID_VOLUMES["O2Si"]
TEMPERATURE: float = 3000.0


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    eos = get_eos_models()
    proxy = {"OSi": eos["CO_cs_shi92"]}
    variants = {
        "SiO as CO, no volume": (None, proxy),
        "SiO as CO, SiO2 liquid volume": (HP11, proxy),
    }
    h = np.repeat([c[0] for c in CASES], PRESSURE_GPA.size)
    z = np.repeat([c[1] for c in CASES], PRESSURE_GPA.size)
    p_bar = np.tile(PRESSURE_GPA * 1e4, len(CASES))
    n = h.size
    mass = budgets(h, z)
    shape = (len(CASES), PRESSURE_GPA.size)
    result: dict = {"cases": CASES, "P_GPa": PRESSURE_GPA.tolist(), "variants": {}}
    for name, (volume, gas_activity) in variants.items():
        model = meb_model(
            mass,
            volume=volume,
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
            "x_SiO": s["x"]["OSi_g"].reshape(shape).tolist(),
            "x_SiH4": s["x"]["H4Si_g"].reshape(shape).tolist(),
            "x_H2O_over_x_H2": (s["x"]["H2O_g"] / s["x"]["H2_g"]).reshape(shape).tolist(),
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
        print(f"{name:32s} {int(ok.sum())}/{n} converged | boundary GPa: {summary}", flush=True)
        (OUT / "sio_proxy_map.json").write_text(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
