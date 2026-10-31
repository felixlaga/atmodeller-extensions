"""Correction factors of equilibrium constants for the volumes of the condensed phases

For a reaction with condensed species i (stoichiometric coefficient nu_i, positive for products),
the condensed volumes change the equilibrium constant by

    Delta ln K = -sum_i nu_i * integral_1bar^P V_i dP / (R T)

Gas species are excluded, since their pressure dependence is in their fugacities, which the
models already include. Factor = exp(Delta ln K).

Volumes: the Holland & Powell (2011) liquids in Atmodeller (LIQUID_VOLUMES), and two brackets
that are not in Atmodeller and are used here only:
- SiO2 melt from Lange & Carmichael (1987): 26.86 cm3/mol at 1773 K, dV/dT = 0,
  dV/dP = -1.89e-4 cm3/mol/bar, so K = 142 kbar, with K' = 4 or 9.46 (the HP11 value).
- FeO melt from Guo, Lange & Ai (2014, doi:10.1016/j.epsl.2013.12.005, abstract values):
  12.86 cm3/mol at 1723 K, dV/dT = 3.69e-3 cm3/mol/K, compressibility 4.72e-2 /GPa (K = 212
  kbar), K' = 4 to 8.
- Fe liquid from Dorogokupets et al. (2017, doi:10.1038/srep41863, Table 1): 7.958 cm3/mol,
  K = 83.7 GPa and K' = 5.97 at 1811 K, with thermal expansion 0 or 1.2e-4 /K (assumed bracket).
The FeO and Fe values are an illustrative bracket for the sign of the iron reaction, as in the
Horn et al. (2025) experiments, not data for the package.

    uv run python research/condensed_volumes/correction_table.py
"""

from __future__ import annotations

import itertools

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pandas as pd
from common import OUT

from atmodeller.thermodata.holland_powell import LIQUID_VOLUMES

GAS_CONSTANT: float = 8.314462618
TEMPERATURES: tuple[float, ...] = (2500.0, 3000.0, 3500.0)
PRESSURES_GPA: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 13.0)


class ReferenceMurnaghan(eqx.Module):
    """Murnaghan volume with its parameters at a reference temperature (J/bar, bar, K)"""

    V_ref: float
    T_ref: float
    alpha: float
    K_ref: float
    dKdP: float

    def volume_integral(self, temperature, pressure):
        volume = self.V_ref * jnp.exp(self.alpha * (temperature - self.T_ref))
        bulk = self.K_ref
        return (
            volume
            * bulk
            / (self.dKdP - 1)
            * ((1 + self.dKdP * (pressure - 1) / bulk) ** (1 - 1 / self.dKdP) - 1)
        )


# Every volume model of each condensed species: the first is the one in Atmodeller, if any
MODELS: dict[str, dict[str, eqx.Module]] = {
    "SiO2": {
        "HP11": LIQUID_VOLUMES["O2Si"],
        "LC87 K'=4": ReferenceMurnaghan(2.686, 1773.0, 0.0, 2.686 / 1.89e-5, 4.0),
        "LC87 K'=9.46": ReferenceMurnaghan(2.686, 1773.0, 0.0, 2.686 / 1.89e-5, 9.46),
    },
    "MgO": {"HP11": LIQUID_VOLUMES["MgO"]},
    "MgSiO3": {"HP11": LIQUID_VOLUMES["MgO3Si"]},
    "Mg2SiO4": {"HP11": LIQUID_VOLUMES["Mg2O4Si"]},
    "FeO": {
        f"Guo14 K'={k}": ReferenceMurnaghan(1.286, 1723.0, 3.69e-4 / 1.286, 1 / 4.72e-6, k)
        for k in (4.0, 8.0)
    },
    "Fe": {
        f"Dorogokupets17 alpha={a}": ReferenceMurnaghan(0.7958, 1811.0, a, 8.37e5, 5.97)
        for a in (0.0, 1.2e-4)
    },
}

# Reactions: condensed species and their coefficients (products positive), and a label of the
# whole reaction. "Werlen26" marks the two reactions that Werlen et al. (2026) tested.
REACTIONS: list[tuple[str, dict[str, int], str]] = [
    ("SiO2(l) = SiO + 1/2 O2", {"SiO2": -1}, "gas-coupled"),
    ("SiO2(l) + H2 = SiO + H2O", {"SiO2": -1}, "gas-coupled"),
    ("SiO2(l) + 4 H2 = SiH4 + 2 H2O", {"SiO2": -1}, "gas-coupled"),
    ("MgSiO3(l) = MgO(l) + SiO + 1/2 O2", {"MgSiO3": -1, "MgO": 1}, "gas-coupled"),
    ("MgO(l) = Mg + 1/2 O2", {"MgO": -1}, "gas-coupled, Werlen26"),
    ("FeO(l) + H2 = Fe(l) + H2O", {"FeO": -1, "Fe": 1}, "gas-coupled, illustrative bracket"),
    ("MgSiO3(l) = MgO(l) + SiO2(l)", {"MgSiO3": -1, "MgO": 1, "SiO2": 1}, "melt only, Werlen26"),
    ("Mg2SiO4(l) = 2 MgO(l) + SiO2(l)", {"Mg2SiO4": -1, "MgO": 2, "SiO2": 1}, "melt only"),
]


def delta_ln_k(stoichiometry: dict[str, int], choice: dict[str, str], T: float, P_bar: float):
    total = sum(
        nu * float(MODELS[s][choice[s]].volume_integral(T, P_bar))
        for s, nu in stoichiometry.items()
    )
    return -total / (GAS_CONSTANT * T)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for (reaction, stoichiometry, kind), T, P in itertools.product(
        REACTIONS, TEMPERATURES, PRESSURES_GPA
    ):
        species = list(stoichiometry)
        choices = [dict(zip(species, c)) for c in itertools.product(*(MODELS[s] for s in species))]
        values = np.array([delta_ln_k(stoichiometry, c, T, P * 1e4) for c in choices])
        primary = delta_ln_k(stoichiometry, choices[0], T, P * 1e4)
        rows.append(
            {
                "reaction": reaction,
                "kind": kind,
                "T_K": T,
                "P_GPa": P,
                "delta_lnK": primary,
                "delta_lnK_min": values.min(),
                "delta_lnK_max": values.max(),
                "factor": np.exp(primary),
                "factor_min": np.exp(values.min()),
                "factor_max": np.exp(values.max()),
                "models": "; ".join(f"{s}: {', '.join(MODELS[s])}" for s in species),
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "correction_table.csv", index=False)

    # A compact view at 3000 K: factor, with the range over the volume models
    lines = ["| Reaction | " + " | ".join(f"{p:g} GPa" for p in PRESSURES_GPA) + " |"]
    lines.append("|---|" + "---:|" * len(PRESSURES_GPA))
    for reaction, _, kind in REACTIONS:
        cells = []
        for P in PRESSURES_GPA:
            r = table[(table.reaction == reaction) & (table.T_K == 3000.0) & (table.P_GPa == P)]
            r = r.iloc[0]
            if np.isclose(r.factor_min, r.factor_max, rtol=1e-6):
                cells.append(f"{r.factor:.3g}")
            else:
                cells.append(f"{r.factor:.3g} ({r.factor_min:.3g}-{r.factor_max:.3g})")
        lines.append(f"| {reaction} [{kind}] | " + " | ".join(cells) + " |")
    markdown = "\n".join(lines) + "\n"
    (OUT / "correction_table_3000K.md").write_text(markdown)
    print(markdown)

    werlen = table[(table.reaction == "MgO(l) = Mg + 1/2 O2") & (table.P_GPa == 4.0)]
    print("MgO(l) = Mg + 1/2 O2 at 4 GPa, Delta ln K (Werlen et al. 2026 Fig. 5 read as 2-4.5):")
    print(werlen[["T_K", "delta_lnK"]].to_string(index=False))


if __name__ == "__main__":
    main()
