"""Shared setup for the condensed-volume study

The magma-envelope boundary (MEB) model is the TOI-421 b setup of Hakim et al. (2026, MNRAS 546,
stag133; arXiv:2508.19235), taken from the authors' notebook generate_data.ipynb on OSF
(https://osf.io/ac6bw): 6.7 Earth masses, MEB radius 1.65 Earth radii, 3000 K, fully molten
mantle, SiO2 liquid at unit activity holding the mantle Si and O, real gases and the authors'
solubility laws. The only change studied here is the volume of the condensed phases.

Compute limits: run one script at a time, single-threaded (see README.md).
"""

from __future__ import annotations

import os

# Atmodeller turns Equinox runtime errors off when it is imported first. Set it here too, so the
# scripts behave the same whatever is imported first.
os.environ.setdefault("EQX_ON_ERROR", "off")

from pathlib import Path  # noqa: E402

import equinox as eqx  # noqa: E402
import numpy as np  # noqa: E402

from atmodeller import (  # noqa: E402
    ChemicalSpecies,
    EquilibriumModel,
    Planet,
    PurePhase,
    ReservoirSpecies,
    SolverParameters,
)
from atmodeller.database import DataBase  # noqa: E402
from atmodeller.eos import get_eos_models  # noqa: E402
from atmodeller.interfaces import VolumeProtocol  # noqa: E402
from atmodeller.solubility import get_solubility_models  # noqa: E402

HERE: Path = Path(__file__).resolve().parent
SOURCES: Path = HERE / "sources"
OUT: Path = HERE / "outputs"

EARTH_MASS: float = 5.972e24
EARTH_RADIUS: float = 6371000.0
GRAVITATIONAL_CONSTANT: float = 6.6743e-11

PLANET_MASS: float = 6.7 * EARTH_MASS
MEB_RADIUS: float = 1.65 * EARTH_RADIUS
MEB_TEMPERATURE: float = 3000.0
CORE_MASS_FRACTION: float = 0.295334691460966
# Mantle Si and O from 45.4 wt% SiO2 (Palme & O'Neill 2014) with a core mass fraction of 0.327,
# as in generate_data.ipynb
SILICON_FRACTION: float = round(45.4 / 98.41 * (1 - 0.327) * 28.0855 / (28.0855 + 2 * 15.999), 3)
OXYGEN_FRACTION: float = SILICON_FRACTION * 2 * 15.999 / 28.0855

GASES: tuple[str, ...] = ("H2O", "H2", "O2", "OSi", "H4Si", "CO2", "CO", "CH4", "N2", "H3N", "He")
EOS: dict[str, str] = {
    "H2O": "H2O_cork_holland98",
    "H2": "H2_chabrier21",
    "O2": "O2_cs_shi92",
    "OSi": "OSi_rk49_connolly16",
    "H4Si": "H4Si_wang18",
    "CO2": "CO2_cs_shi92",
    "CO": "CO_cs_shi92",
    "CH4": "CH4_cs_shi92",
    "N2": "N2_cs_saxena87",
    "H3N": "H3N_rk49_reid87",
    "He": "He_chabrier21",
}
SOLUBILITY: dict[str, str] = {
    "H2O": "H2O_peridotite_sossi23",
    "H2": "H2_basalt_hirschmann12",
    "CO2": "CO2_basalt_dixon95",
    "CO": "CO_basalt_yoshioka19",
    "CH4": "CH4_basalt_ardia13",
    "N2": "N2_basalt_libourel03",
    "He": "He_basalt_jambon86",
}
HYDROGEN_GRID: np.ndarray = np.logspace(-1, 0, 50)
"""Hydrogen mass fractions (wt% of the planet) of the authors' grid"""


class ScaledVolume(eqx.Module):
    """A volume model whose integral is multiplied by a factor

    For sensitivity tests only: a factor of 0 removes the volume and 1 keeps it.
    """

    inner: VolumeProtocol
    factor: float = eqx.field(converter=float)

    def volume_integral(self, temperature, pressure):
        return self.factor * self.inner.volume_integral(temperature, pressure)


class ExcessVolumeGas(eqx.Module):
    """A real gas with a constant extra partial molar volume in the fluid

    Adds dV (P - 1 bar) / RT to the log fugacity of the inner equation of state, which raises the
    chemical potential of the species as a larger partial molar volume would. For sensitivity
    tests of the gas side only.
    """

    inner: eqx.Module
    excess_volume_cm3: float = eqx.field(converter=float)

    def log_activity(self, temperature, pressure, mole_fractions=None):
        excess = self.excess_volume_cm3 * 1e-6 * (pressure - 1.0) * 1e5
        return self.inner.log_activity(temperature, pressure, mole_fractions) + excess / (
            8.314462618 * temperature
        )


def budgets(hydrogen_wt_percent, metallicity) -> dict[str, np.ndarray]:
    """Element masses (kg) of the authors' grid, from their H mass and metallicity"""
    h = np.asarray(hydrogen_wt_percent, dtype=float) / 100 * PLANET_MASS
    z = np.asarray(metallicity, dtype=float)
    return {
        "H": h,
        "He": h * 24.69 / 73.9,
        "C": h * 0.22 / 73.9 * z,
        "N": h * 0.07 / 73.9 * z,
        "Si": h * 0.07 / 73.9 * z + SILICON_FRACTION * PLANET_MASS,
        "O": h * 0.63 / 73.9 * z + OXYGEN_FRACTION * PLANET_MASS,
    }


def silica(volume: VolumeProtocol | None) -> PurePhase:
    """SiO2 liquid at unit activity, with the given volume model or none"""
    database: DataBase = DataBase.default()
    if volume is not None:
        liquid = database.get_end_member("O2Si", "liquid")
        database.set_end_member("O2Si", "liquid", liquid.with_volume(volume))

    return PurePhase.from_end_member("liquid", formula="SiO2", database=database)


def meb_model(
    mass: dict[str, np.ndarray],
    *,
    volume: VolumeProtocol | None = None,
    real_gas: bool = True,
    temperature: float | np.ndarray = MEB_TEMPERATURE,
    pressure: float | np.ndarray = np.nan,
    max_starts: int = 10,
    gas_activity: dict | None = None,
) -> EquilibriumModel:
    """The MEB model. A finite pressure fixes it; NaN solves it from the weight of the gas.

    ``max_starts`` caps the restarts of rows that fail (Atmodeller's default is 10).
    ``gas_activity`` replaces the activity model of the gases it names, e.g. ``{"OSi": ...}``.
    """
    eos = get_eos_models()
    solubility = get_solubility_models()
    activity = {s: eos[EOS[s]] for s in GASES} if real_gas else {}
    activity.update(gas_activity or {})
    gases = tuple(
        ChemicalSpecies.create_gas(s, **({"activity": activity[s]} if s in activity else {}))
        for s in GASES
    )
    dissolved = tuple(
        ReservoirSpecies.create_dissolved(
            s, solubility=solubility[law], include_in_phase_mass=False
        )
        for s, law in SOLUBILITY.items()
    )
    state = Planet.from_species(
        gases,
        silicate_melt_species=dissolved,
        condensates=(silica(volume),),
        planet_mass=PLANET_MASS,
        core_mass_fraction=CORE_MASS_FRACTION,
        surface_radius=MEB_RADIUS,
        temperature=temperature,
        pressure=pressure,
    )

    return EquilibriumModel.from_state(
        state,
        mass_constraints=mass,
        solver_parameters=SolverParameters(atol=1e-8, rtol=1e-8, max_starts=max_starts),
    )


def parcel_model(
    mass: dict[str, np.ndarray], *, temperature: float = 920.0, pressure: float = 0.01
) -> EquilibriumModel:
    """A closed upper-atmosphere parcel with the element masses of a MEB gas

    Ideal gases at the equilibrium temperature of TOI-421 b, 920 K, and a fixed 10 mbar, with
    the condensates of Hakim et al. (2026): SiO2, graphite, SiC, Si and Si3N4. No rainout,
    quenching or photochemistry. Hakim et al. (2026) solved the pressure instead, over 1 mbar to
    100 bar, so this parcel is a comparison between cases, not a rerun of their atmosphere.
    """
    n = np.asarray(next(iter(mass.values()))).size
    gases = tuple(ChemicalSpecies.create_gas(s) for s in GASES)
    solids = tuple(
        PurePhase.from_formula(s, liquid=False) for s in ("O2Si", "C", "CSi", "Si", "N4Si3")
    )
    state = Planet.from_species(
        gases,
        condensates=solids,
        planet_mass=PLANET_MASS,
        surface_radius=2.64 * EARTH_RADIUS,
        mantle_melt_fraction=0.0,
        temperature=np.full(n, temperature),
        pressure=np.full(n, pressure),
    )
    return EquilibriumModel.from_state(
        state, mass_constraints=mass, solver_parameters=SolverParameters(atol=1e-8, rtol=1e-8)
    )


def summarise(output, n: int) -> dict:
    """Gas mole fractions, fugacities, element masses, pressure and any SiO2 liquid left"""
    data = output.to_dict(to_numpy=True)
    gas = data["gas"]

    def flat(x) -> np.ndarray:
        return np.broadcast_to(np.asarray(x, dtype=float).reshape(-1), (n,)).copy()

    partial = {s: flat(v) for s, v in gas["species"]["partial_pressure"].items()}
    return {
        "success": flat(output.multi_attempt_solution.success).astype(bool),
        "pressure_bar": sum(partial.values()),
        "x": {s: flat(v) for s, v in gas["species"]["mole_fraction"].items()},
        "f": {s: flat(v) for s, v in gas["species"]["activity"].items()},
        "gas_element_mass": {e: flat(v) for e, v in gas["elements"]["mass"].items()},
        "silica_mass": (
            flat(data["O2Si_l"]["species"]["mass"]["O2Si_l"]) if "O2Si_l" in data else None
        ),
        "silica_activity": (
            flat(data["O2Si_l"]["species"]["activity"]["O2Si_l"]) if "O2Si_l" in data else None
        ),
    }


def mechanical_pressure_bar(gas_element_mass: dict[str, np.ndarray]) -> np.ndarray:
    """Pressure (bar) that the gas would exert at the MEB, m_gas g / A, as the model computes it

    Atmodeller takes the gravity from the background mantle and the core, which here add up to
    the planet mass. The SiO2 liquid and the gas do not count, so the gravity stays the same
    however much SiO2 goes into the gas. port_check.py checks this against the solved pressure.
    """
    gas_mass = sum(gas_element_mass.values())
    gravity = GRAVITATIONAL_CONSTANT * PLANET_MASS / MEB_RADIUS**2
    return gas_mass * gravity / (4 * np.pi * MEB_RADIUS**2) / 1e5
