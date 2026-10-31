"""Bower et al. (2025) K2-18 b: the iron-wustite buffer at 1 bar or at the MEB pressure

The K2-18 b runs of Bower et al. (2025, ApJ 995, 59; notebooks/examples/subneptune.ipynb) have no
condensed species in their reactions: the oxygen fugacity is imposed at IW-3. Condensed volumes
reach them only through the buffer. Atmodeller evaluates the buffer at 1 bar by default; with
``evaluation_pressure=None`` it is evaluated at the total pressure, which includes the volume
change of Fe + 1/2 O2 = FeO (Hirschmann 2021). This compares the two over the paper's hydrogen
sweep (0.1 to 3% of the planet mass, C/H at 100x solar, 3000 K, real gases).

    uv run python research/condensed_volumes/bower_iw.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from common import OUT

from atmodeller import ChemicalSpecies, EquilibriumModel, Planet, ReservoirSpecies
from atmodeller.eos import get_eos_models
from atmodeller.solubility import get_solubility_models
from atmodeller.thermodata import IronWustiteBuffer

PLANET_MASS: float = 5.154e25
SURFACE_RADIUS: float = 1.1225e7
EOS = {
    "H2O": "H2O_cork_holland98",
    "H2": "H2_chabrier21",
    "CO": "CO_cs_shi92",
    "CO2": "CO2_cs_shi92",
    "CH4": "CH4_cs_shi92",
}
SOLUBILITY = {
    "H2O": "H2O_basalt_dixon95",
    "H2": "H2_basalt_hirschmann12",
    "CO": "CO_basalt_yoshioka19",
    "CO2": "CO2_basalt_dixon95",
    "CH4": "CH4_basalt_ardia13",
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    eos = get_eos_models()
    solubility = get_solubility_models()
    gases = tuple(
        ChemicalSpecies.create_gas(s, **({"activity": eos[EOS[s]]} if s in EOS else {}))
        for s in ("H2O", "H2", "O2", "CO", "CO2", "CH4")
    )
    melt = tuple(
        ReservoirSpecies.create_dissolved(
            s, solubility=solubility[law], include_in_phase_mass=False
        )
        for s, law in SOLUBILITY.items()
    )
    state = Planet.from_species(
        gases,
        silicate_melt_species=melt,
        temperature=3000.0,
        planet_mass=PLANET_MASS,
        surface_radius=SURFACE_RADIUS,
        mantle_melt_fraction=1.0,
    )
    log10_h = np.linspace(-1.0, np.log10(3.0), 25)
    h_kg = 10**log10_h / 100 * PLANET_MASS
    mass = {"H": h_kg, "C": h_kg * 10**-0.5}
    frames = []
    guess = None
    for label, pressure in (("IW at 1 bar", 1.0), ("IW at MEB pressure", None)):
        buffer = IronWustiteBuffer(np.full(log10_h.size, -3.0), evaluation_pressure=pressure)
        model = EquilibriumModel.from_state(
            state, mass_constraints=mass, activity_constraints={"O2_g": buffer}
        )
        # The buffer at the MEB pressure starts from the 1-bar solution, which is close to it
        output = model.solve_with_default() if guess is None else model.solve(guess)
        guess = np.asarray(output.solution)
        success = np.ravel(np.asarray(output.multi_attempt_solution.success)).astype(bool)
        data = output.to_dict(to_numpy=True)
        x = {s: np.ravel(v) for s, v in data["gas"]["species"]["mole_fraction"].items()}
        p = sum(np.ravel(v) for v in data["gas"]["species"]["partial_pressure"].values())
        frames.append(
            pd.DataFrame(
                {
                    "buffer": label,
                    "H_percent": 10**log10_h,
                    "success": success,
                    "P_GPa": p / 1e4,
                    "x_H2": x["H2_g"],
                    "x_CH4": x["CH4_g"],
                    "x_H2O": x["H2O_g"],
                    "x_CO": x["CO_g"],
                    "CH4_over_H2": x["CH4_g"] / x["H2_g"],
                }
            )
        )
    result = pd.concat(frames, ignore_index=True)
    result.to_csv(OUT / "bower_iw.csv", index=False)
    converged = result.groupby("buffer", sort=False).success.sum().to_dict()
    result = result[result.groupby("H_percent").success.transform("all")]

    lines = []
    for label, group in result.groupby("buffer", sort=False):
        ratio = np.log(group.CH4_over_H2.to_numpy())
        h = group.H_percent.to_numpy()
        cross = np.flatnonzero(np.diff(np.sign(ratio)))
        if cross.size:
            i = cross[0]
            w = ratio[i] / (ratio[i] - ratio[i + 1])
            h_cross = float(np.exp(np.log(h[i]) + w * (np.log(h[i + 1]) - np.log(h[i]))))
            lines.append(f"{label}: CH4 = H2 at {h_cross:.3f}% H")
        else:
            lines.append(f"{label}: no CH4 = H2 crossing in 0.1-3% H")
    both = result.pivot(index="H_percent", columns="buffer")
    lines.append(
        "max |delta log10 CH4/H2|: "
        f"{np.max(np.abs(np.log10(both.CH4_over_H2['IW at MEB pressure'] / both.CH4_over_H2['IW at 1 bar']))):.3f}"
    )
    lines.append(
        "max relative P change: "
        f"{np.max(np.abs(both.P_GPa['IW at MEB pressure'] / both.P_GPa['IW at 1 bar'] - 1)):.3f}"
    )
    lines.append(f"converged per buffer: {converged}; compared where both converged")
    text = "\n".join(lines) + "\n"
    (OUT / "bower_iw.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
