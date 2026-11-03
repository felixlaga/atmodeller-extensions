# SPDX-License-Identifier: Apache-2.0

"""Cooling paths of a closed binary-solution parcel at fixed pressure.

A parcel of fixed bulk composition loses heat at constant pressure. It stays
homogeneous down to a chosen onset temperature, then takes its equilibrium
state and keeps it. Separation is either isenthalpic, so the released heat
warms the parcel (recalescence), or isothermal, so that heat leaves at once.
An onset at the binodal gives the equilibrium path. Heat lost per kilogram is
the path coordinate, so no cooling rate is needed.

The fast functions batch over parameter samples with :func:`jax.vmap`.
:func:`solve_path_with_engine` replays a path with two prebuilt
:class:`~atmodeller.classes.EquilibriumModel` instances, one homogeneous and one
demixing, so the solver never recompiles. This module adds no gravitational
settling, no exchange with a gas and no pressure dependence of the sensible
heat capacity.
"""

from collections.abc import Iterable
from typing import Any, Literal, Self

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, Bool

from atmodeller.containers import ChemicalSpecies
from atmodeller.jax_utils import FloatArray, as_j64
from atmodeller.thermodata.core import ThermodynamicProperties
from atmodeller.sci_utils import GAS_CONSTANT
from atmodeller.thermodata.mixing import AVOGADRO, SubregularSolution, _bisect

REFERENCE_TEMPERATURE = 298.15
"""Sensible enthalpies are measured from this temperature (K)"""


def _require_single(parcel: "ParcelCooling") -> None:
    """Paths handle one parcel; parameter batches go through jax.vmap."""
    model = parcel.mixing_model
    leaves = (parcel.pressure, parcel.composition, parcel.heat_capacity_scale)
    leaves += (model.interaction_0, model.interaction_1)
    leaves += (model.temperature_scale, model.pressure_scale)
    if any(jnp.ndim(leaf) for leaf in leaves):
        raise ValueError("A cooling path describes one parcel; batch it with jax.vmap.")


class CoolingPath(eqx.Module):
    """Rows of a cooling path: the homogeneous segment, then the separated segment.

    Row arrays have shape (..., 2 * steps). The second segment starts right
    after separation. If the parcel never separates within the range, the
    first segment reaches the final temperature and the second is NaN.
    """

    heat_lost: FloatArray
    """Heat removed since the initial homogeneous state (J/kg)"""
    temperature: FloatArray
    """Temperature (K)"""
    separated: Bool[Array, "..."]
    """Whether the parcel is two-phase in its equilibrium state"""
    upper_fraction: FloatArray
    """Native mole fraction of the parcel in the upper (component-1-rich) phase"""
    mixing_enthalpy: FloatArray
    """Mixing enthalpy of the actual state (J/kg)"""
    binodal_temperature: FloatArray
    """Equilibrium separation onset (K)"""
    onset_temperature: FloatArray
    """Temperature at which separation starts (K); NaN if it never starts in range"""
    separation_temperature: FloatArray
    """Temperature right after separation (K); above the onset if isenthalpic"""
    released_heat: FloatArray
    """Demixing heat of the homogeneous parcel at the onset temperature (J/kg)"""
    valid: Bool[Array, "..."]
    """False for an invalid input, common tangent or separation temperature"""


class ParcelCooling(eqx.Module):
    """A closed parcel of a binary solution at fixed pressure and composition.

    The sensible heat comes from the pure-component standard-state enthalpies,
    mixed by amount, without a pressure correction. ``heat_capacity_scale``
    multiplies that sensible part, for uncertainty scans. The mixing model
    supplies the heat of mixing and of separation.

    Args:
        mixing_model: Binary mixing free energy on the native component basis.
        thermo: Pure-component standard-state properties, in component order.
        molar_masses: Native component molar masses (kg/mol).
        pressure: Pressure (bar).
        composition: Native component-1 mole fraction.
        heat_capacity_scale: Factor on the sensible heat capacity. Defaults to one.
    """

    mixing_model: SubregularSolution
    thermo: tuple[ThermodynamicProperties, ThermodynamicProperties]
    molar_masses: FloatArray
    pressure: FloatArray
    composition: FloatArray
    heat_capacity_scale: FloatArray

    def __init__(
        self,
        mixing_model: SubregularSolution,
        thermo: tuple[ThermodynamicProperties, ThermodynamicProperties],
        molar_masses: ArrayLike,
        pressure: ArrayLike,
        composition: ArrayLike,
        heat_capacity_scale: ArrayLike = 1.0,
    ):
        self.mixing_model = mixing_model
        self.thermo = thermo
        self.molar_masses = as_j64(molar_masses)
        self.pressure = as_j64(pressure)
        self.composition = as_j64(composition)
        self.heat_capacity_scale = as_j64(heat_capacity_scale)

    @classmethod
    def from_species(
        cls,
        species: Iterable[ChemicalSpecies],
        mixing_model: SubregularSolution,
        pressure: ArrayLike,
        *,
        mass_fraction: ArrayLike,
        heat_capacity_scale: ArrayLike = 1.0,
    ) -> Self:
        """Builds a parcel from the two ordered solution components.

        Args:
            species: The two components, in the mixing model's order.
            mixing_model: Binary mixing free energy on the native component basis.
            pressure: Pressure (bar).
            mass_fraction: Mass fraction of component 1.
            heat_capacity_scale: Factor on the sensible heat capacity.
        """
        components = tuple(species)
        if len(components) != 2:
            raise ValueError("A binary parcel requires exactly two ordered components.")
        masses = jnp.array([c.data.molar_mass for c in components], dtype=float)
        w = as_j64(mass_fraction)
        moles_0, moles_1 = (1 - w) / masses[0], w / masses[1]
        composition = moles_1 / (moles_0 + moles_1)
        thermo = (components[0].thermo, components[1].thermo)
        return cls(mixing_model, thermo, masses, pressure, composition, heat_capacity_scale)

    @property
    def molar_mass(self) -> FloatArray:
        """Mass per native mole of the parcel (kg/mol)."""
        x = self.composition
        return (1 - x) * self.molar_masses[0] + x * self.molar_masses[1]

    def sensible_enthalpy(self, temperature: ArrayLike) -> FloatArray:
        """Pure-component enthalpy above the reference temperature (J/kg)."""
        t, x = as_j64(temperature), self.composition
        reference = as_j64(REFERENCE_TEMPERATURE)
        rise = [th.enthalpy(t) - th.enthalpy(reference) for th in self.thermo]
        per_mole = (1 - x) * rise[0] + x * rise[1]
        return self.heat_capacity_scale * per_mole / self.molar_mass

    def enthalpy(self, temperature: ArrayLike, separated: bool = False) -> FloatArray:
        """Specific enthalpy of the homogeneous or the equilibrium state (J/kg)."""
        model, p, x = self.mixing_model, self.pressure, self.composition
        if separated:
            mixing = model.equilibrium_enthalpy(temperature, p, x)
        else:
            mixing = model.enthalpy(temperature, p, x)
        return self.sensible_enthalpy(temperature) + mixing / self.molar_mass

    def path(
        self,
        initial_temperature: ArrayLike,
        final_temperature: ArrayLike,
        onset_temperature: ArrayLike | None = None,
        *,
        separation: Literal["isenthalpic", "isothermal"] = "isenthalpic",
        steps: int = 100,
    ) -> CoolingPath:
        """Cools the parcel from a homogeneous initial state to a final temperature.

        Args:
            initial_temperature: Starting temperature (K); the parcel starts homogeneous.
            final_temperature: Last temperature (K), below the initial one.
            onset_temperature: Temperature at which the homogeneous parcel separates,
                for example a spinodal or nucleation temperature. Defaults to the
                binodal, which gives the equilibrium path. Values above the binodal
                are clipped to it. NaN means it never separates.
            separation: ``isenthalpic`` keeps the enthalpy through separation, so
                the parcel reheats. ``isothermal`` removes the released heat at the
                onset temperature.
            steps: Rows per segment.

        Returns:
            The path, with 2 * steps rows.
        """
        if separation not in ("isenthalpic", "isothermal"):
            raise ValueError("Separation must be 'isenthalpic' or 'isothermal'.")
        _require_single(self)
        model, p, x = self.mixing_model, self.pressure, self.composition
        t0, tf = as_j64(initial_temperature), as_j64(final_temperature)
        binodal = model.binodal_temperature(p, x)
        onset = binodal if onset_temperature is None else as_j64(onset_temperature)
        onset = jnp.minimum(jnp.minimum(onset, binodal), t0)
        separates = jnp.isfinite(onset) & (onset > tf)
        t_on = jnp.where(separates, onset, tf)
        h_on = self.enthalpy(t_on)
        if separation == "isenthalpic":
            # The equilibrium enthalpy rises with temperature and equals the
            # homogeneous one at the binodal, so the root lies in [onset, binodal].
            upper = jnp.where(separates, jnp.maximum(binodal, t_on), t_on)
            t_sep = _bisect(lambda t: self.enthalpy(t, separated=True) - h_on, t_on, upper)
            closed = jnp.abs(self.enthalpy(t_sep, separated=True) - h_on) <= 1e-6
        else:
            t_sep, closed = t_on, jnp.asarray(True)
        t_sep = jnp.where(separates, t_sep, jnp.nan)

        temperature = jnp.concatenate(
            (jnp.linspace(t0, t_on, steps, axis=-1), jnp.linspace(t_sep, tf, steps, axis=-1)),
            axis=-1,
        )
        second = jnp.arange(2 * steps) >= steps
        absent = second & ~separates[..., None]
        pe, xe, me = p[..., None], x[..., None], self.molar_mass[..., None]
        c = model.coexistence(temperature, pe)
        equilibrium = model._lever_rule(model.enthalpy, temperature, pe, xe, c)
        mixing = jnp.where(second, equilibrium, model.enthalpy(temperature, pe, xe)) / me
        heat_lost = self.enthalpy(t0)[..., None] - self.sensible_enthalpy(temperature) - mixing
        fraction = jnp.where(second, c.upper_fraction(xe), 0.0)

        def hide(values):
            return jnp.where(absent, jnp.nan, values)

        released = model.demixing_heat(t_on, p, x) / self.molar_mass
        valid = jnp.isfinite(t0) & jnp.isfinite(tf) & (t0 > tf) & (steps > 1)
        valid &= jnp.all(c.valid | absent, axis=-1) & (closed | ~separates)
        valid &= model.coexistence(t0, p).valid
        # A missing binodal is only an answer if the parcel never meets the gap.
        meets_gap = jnp.any(c.contains(xe) & ~second, axis=-1)
        valid &= jnp.isfinite(binodal) | ~meets_gap
        return CoolingPath(
            heat_lost=hide(heat_lost),
            temperature=hide(temperature),
            separated=second & ~absent & c.contains(xe),
            upper_fraction=hide(fraction),
            mixing_enthalpy=hide(mixing),
            binodal_temperature=binodal,
            onset_temperature=jnp.where(separates, t_on, jnp.nan),
            separation_temperature=t_sep,
            released_heat=jnp.where(separates, released, jnp.nan),
            valid=valid,
        )


def adiabat_temperature_factor(
    top_pressure: ArrayLike, bottom_pressure: ArrayLike, gradient: ArrayLike
) -> FloatArray:
    """Mass-weighted mean of T/T_top over a column on the adiabat T_top exp(g (P - P_top)).

    Mass is spread uniformly in pressure, as in a thin shell under constant
    gravity. ``gradient`` is dlnT/dP (1/bar), alpha / (rho c_p) for a melt.
    The factor converts the top temperature change into the column's mean one.
    """
    span = as_j64(gradient) * (as_j64(bottom_pressure) - as_j64(top_pressure))
    safe = jnp.where(jnp.abs(span) > 1e-12, span, 1.0)
    return jnp.where(jnp.abs(span) > 1e-12, jnp.expm1(safe) / safe, 1.0 + span / 2)


class InteriorPath(eqx.Module):
    """Rows of an interior cooling path, one per top temperature, shape (..., steps).

    Energies are per kilogram of the whole system: silicate plus all of the
    hydrogen, dissolved or exsolved.
    """

    temperature: FloatArray
    """Temperature at the top of the interior (K)"""
    heat_lost: FloatArray
    """Sensible plus exsolution heat removed since the start (J/kg)"""
    sensible_heat: FloatArray
    """Sensible heat removed (J/kg)"""
    exsolution_heat: FloatArray
    """Heat released by exsolution, cumulative (J/kg)"""
    composition: FloatArray
    """Native component-1 mole fraction of the interior melt"""
    exsolved_hydrogen: FloatArray
    """Component-1 mass that has left the interior, per kg of system"""
    heat_capacity_ratio: FloatArray
    """Exsolution heat per kelvin of top cooling over the sensible heat capacity"""
    onset_temperature: FloatArray
    """Binodal temperature of the initial interior (K)"""
    valid: Bool[Array, "..."]


class InteriorCooling(eqx.Module):
    """A convecting interior that exsolves component 1 at its top into an envelope.

    Convection keeps the interior uniform and brings every parcel to the top,
    where the interior meets the envelope at the top pressure. Deeper layers
    are hotter and, for the hydrogen-silicate models, at higher pressure, so
    more miscible. The interior composition therefore follows the lower
    binodal branch at the top once it enters the gap, and the exsolved
    component leaves as a pure phase. That pure end state gives the full heat
    of unmixing, the integral of the component's partial molar enthalpy. Its
    sensible heat follows the top temperature times ``temperature_factor``
    (see :func:`adiabat_temperature_factor`). No settling energy, no change of
    the top pressure and no envelope physics are included.

    Args:
        parcel: The interior's initial state; its pressure is the top pressure.
        temperature_factor: Mean column temperature change per top temperature change.
    """

    parcel: ParcelCooling
    temperature_factor: FloatArray

    def __init__(self, parcel: ParcelCooling, temperature_factor: ArrayLike = 1.0):
        self.parcel = parcel
        self.temperature_factor = as_j64(temperature_factor)

    @classmethod
    def from_species(
        cls,
        species: Iterable[ChemicalSpecies],
        mixing_model: SubregularSolution,
        top_pressure: ArrayLike,
        *,
        mass_fraction: ArrayLike,
        heat_capacity_scale: ArrayLike = 1.0,
        temperature_factor: ArrayLike = 1.0,
    ) -> Self:
        """Builds an interior from its two ordered components and initial mass fraction."""
        parcel = ParcelCooling.from_species(
            species,
            mixing_model,
            top_pressure,
            mass_fraction=mass_fraction,
            heat_capacity_scale=heat_capacity_scale,
        )
        return cls(parcel, temperature_factor)

    @property
    def initial_composition(self) -> FloatArray:
        """Native component-1 mole fraction before exsolution."""
        return self.parcel.composition

    def composition(self, temperature: ArrayLike) -> FloatArray:
        """Interior melt composition at a top temperature, NaN for an invalid tangent."""
        x0 = self.initial_composition
        c = self.parcel.mixing_model.coexistence(temperature, self.parcel.pressure)
        return jnp.where(c.valid, jnp.where(c.contains(x0), c.lower, x0), jnp.nan)

    def _component_0_moles(self) -> FloatArray:
        x0, masses = self.initial_composition, self.parcel.molar_masses
        return (1 - x0) / ((1 - x0) * masses[0] + x0 * masses[1])

    def mixing_heat(
        self, temperature: ArrayLike, composition: ArrayLike | None = None
    ) -> FloatArray:
        """Mixing enthalpy of the interior melt per kg of system (J/kg).

        The melt composition defaults to :meth:`composition` at this temperature.
        """
        x = self.composition(temperature) if composition is None else as_j64(composition)
        total_moles = self._component_0_moles() / (1 - x)
        enthalpy = self.parcel.mixing_model.enthalpy(temperature, self.parcel.pressure, x)
        return total_moles * enthalpy

    def path(
        self, initial_temperature: ArrayLike, final_temperature: ArrayLike, *, steps: int = 200
    ) -> InteriorPath:
        """Cools the top of the interior from an initial to a final temperature.

        The reference state is the homogeneous interior at the initial temperature,
        so a start inside the gap counts the hydrogen that exsolves at once. Only
        interiors that enter the gap on the component-0-rich side are valid: an
        interior richer than the critical composition would exsolve the other
        phase, which this model does not describe.
        """
        _require_single(self.parcel)
        t0, tf = as_j64(initial_temperature), as_j64(final_temperature)
        x0 = self.initial_composition
        temperature = jnp.linspace(t0, tf, steps, axis=-1)
        factor = self.temperature_factor
        sensible = factor * (
            self.parcel.sensible_enthalpy(t0)[..., None]
            - self.parcel.sensible_enthalpy(temperature)
        )
        exsolution = self.mixing_heat(t0, x0)[..., None] - self.mixing_heat(temperature)
        x = self.composition(temperature)
        moles_0 = self._component_0_moles()
        dissolved = moles_0 * x / (1 - x)
        exsolved = (moles_0 * x0 / (1 - x0) - dissolved) * self.parcel.molar_masses[1]
        ones = jnp.ones_like(temperature)
        _, release = jax.jvp(self.mixing_heat, (temperature,), (ones,))
        _, capacity = jax.jvp(self.parcel.sensible_enthalpy, (temperature,), (ones,))
        valid = jnp.isfinite(t0) & jnp.isfinite(tf) & (t0 > tf) & (steps > 1)
        valid &= jnp.all(jnp.isfinite(x), axis=-1)
        model, p = self.parcel.mixing_model, self.parcel.pressure
        onset = model.binodal_temperature(p, x0)
        # Just below the onset, x0 sits next to the branch it entered through.
        entry = model.coexistence(onset * (1 - 1e-7), p)
        lower_side = (x0 - entry.lower) <= (entry.upper - x0)
        valid &= jnp.isnan(onset) | lower_side
        return InteriorPath(
            temperature=temperature,
            heat_lost=sensible + exsolution,
            sensible_heat=sensible,
            exsolution_heat=exsolution,
            composition=x,
            exsolved_hydrogen=exsolved,
            heat_capacity_ratio=release / (factor * capacity),
            onset_temperature=onset,
            valid=valid,
        )


def nucleation_onset_temperature(
    mixing_model: SubregularSolution,
    pressure: ArrayLike,
    x: ArrayLike,
    gradient_coefficient: ArrayLike,
    molar_volume: ArrayLike,
    criterion: ArrayLike,
    *,
    theory: Literal["square_gradient", "classical"] = "square_gradient",
    iterations: int = 48,
) -> FloatArray:
    """Temperature at which a cooling homogeneous parcel first nucleates the other phase.

    The highest temperature below the binodal where the barrier over kT falls to
    ``criterion``, typically ln(J0 V t) of about 70-180. A scan of undercoolings
    down to the spinodal, where the barrier vanishes, or to half the binodal
    temperature if there is no spinodal, brackets the crossing with finite
    barriers on both sides; bisection refines it. NaN if no such bracket exists,
    for example when the barrier solve fails at the crossing.

    Args:
        mixing_model: Binary mixing free energy on the native component basis.
        pressure: Pressure (bar).
        x: Native component-1 mole fraction of the parcel.
        gradient_coefficient: kappa (J m²/mol) of the mixing-basis composition.
        molar_volume: Volume of one mixing-basis mole (m³/mol).
        criterion: Barrier over kT at which nucleation becomes effective.
        theory: Square-gradient saddle, or classical theory with the planar tension.
        iterations: Bisection steps.
    """
    p, x = as_j64(pressure), as_j64(x)
    kappa, volume, criterion = (as_j64(v) for v in (gradient_coefficient, molar_volume, criterion))
    binodal = mixing_model.binodal_temperature(p, x)
    spinodal = mixing_model.spinodal_temperature(p, x)
    has_spinodal = jnp.isfinite(spinodal) & (spinodal < binodal)
    span = jnp.where(has_spinodal, binodal - spinodal, binodal / 2)

    def excess(t, expand=False):
        args = (p, x, kappa, volume, criterion)
        if expand:
            args = tuple(v[..., None] for v in args)
        b = mixing_model.nucleation_barrier(t, args[0], args[1], args[2], args[3])
        energy = b.classical_energy if theory == "classical" else b.energy
        return energy * AVOGADRO / (GAS_CONSTANT * t) - args[4]

    # Scan undercoolings from tiny to the full window for a finite sign change,
    # so a failed barrier (NaN) never counts as either side.
    grid = binodal[..., None] - span[..., None] * jnp.geomspace(1e-7, 1 - 1e-9, 40)
    values = excess(grid, expand=True)
    below = jnp.isfinite(values) & (values < 0)
    first = jnp.argmax(below, axis=-1)
    previous = jnp.maximum(first - 1, 0)

    def pick(array, index):
        return jnp.take_along_axis(array, index[..., None], axis=-1)[..., 0]

    bracketed = jnp.any(below, axis=-1) & (first > 0)
    bracketed &= jnp.isfinite(pick(values, previous)) & (pick(values, previous) > 0)
    lower, upper = pick(grid, first), pick(grid, previous)

    def step(_, bounds):
        lo, hi = bounds
        mid = lo + (hi - lo) / 2
        negative = excess(mid) < 0
        return jnp.where(negative, mid, lo), jnp.where(negative, hi, mid)

    lower, upper = jax.lax.fori_loop(0, iterations, step, (lower, upper))
    # Both final bracket ends must be finite and straddle the criterion, up to
    # rounding: they differ from it by about 1e-12 after the bisection.
    e_lo, e_hi = excess(lower), excess(upper)
    valid = bracketed & jnp.isfinite(e_lo) & jnp.isfinite(e_hi)
    valid &= (e_lo < 1e-6) & (e_hi > -1e-6)
    return jnp.where(valid, lower + (upper - lower) / 2, jnp.nan)


def solve_path_with_engine(
    path: CoolingPath,
    homogeneous_model: Any,
    equilibrium_model: Any,
    phase_name: str,
) -> dict[str, np.ndarray]:
    """Replays an unbatched path with the full equilibrium solver.

    The homogeneous model solves the first segment and the demixing model the
    second. Build both once, with a temperature batch of ``steps`` entries and
    the parcel's composition, so only their temperatures change here. For a
    closed parcel the solver must reproduce the fast functions; its value is
    that other phases and reactions can later join the same path.

    Args:
        path: A path from :meth:`ParcelCooling.path` for one parcel.
        homogeneous_model: :class:`~atmodeller.classes.EquilibriumModel` in
            ``homogeneous`` mode.
        equilibrium_model: The same model in a demixing mode.
        phase_name: Name of the binary solution phase.

    Returns:
        Per-row ``mixing_enthalpy`` (J/kg), ``upper_fraction`` and ``success``.
        Rows absent from the path are NaN, and count as successful.
    """
    temperature = np.asarray(path.temperature)
    steps = temperature.shape[-1] // 2
    columns: dict[str, list[np.ndarray]] = {
        "mixing_enthalpy": [],
        "upper_fraction": [],
        "success": [],
    }
    for model, rows in (
        (homogeneous_model, temperature[:steps]),
        (equilibrium_model, temperature[steps:]),
    ):
        present = np.isfinite(rows)
        if not present.any():
            columns["mixing_enthalpy"].append(np.full(steps, np.nan))
            columns["upper_fraction"].append(np.full(steps, np.nan))
            columns["success"].append(np.ones(steps, dtype=bool))
            continue
        filled = np.where(present, rows, rows[present][0])
        output = model.update_state(temperature=jnp.asarray(filled)).solve_with_default()
        phase = output.to_dict(to_numpy=True)[phase_name]["phase"]
        upper = phase["subphases"]["upper"]
        moles = phase["subphases"]["lower"]["number_moles"] + upper["number_moles"]
        molar_mass = phase["mass"] / moles
        enthalpy = phase["mixing"]["actual_mixing_enthalpy_J_per_mol"] / molar_mass
        columns["mixing_enthalpy"].append(np.where(present, enthalpy[:, 0], np.nan))
        columns["upper_fraction"].append(
            np.where(present, upper["mole_fraction_of_parent"][:, 0], np.nan)
        )
        success = np.asarray(output.multi_attempt_solution.success).reshape(-1)
        columns["success"].append(success | ~present)
    return {key: np.concatenate(value) for key, value in columns.items()}
