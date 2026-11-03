Binary solutions and spinodal stability
=======================================

This fork adds a binary mixing free energy, its spinodal and binodal, and
equilibrium demixing through the existing phase/activity interface. It is the
thermodynamic foundation for a spinodal-decomposition model. It does not yet
evolve composition in space or time.

The earlier hydrogen-storage investigation is preserved under
``research/hydrogen_storage/``. Those results used hypothetical solubility
multipliers and the unmodified upstream engine. They are not results of this
extension. See ``research/README.rst`` for provenance and reproduction.

Thermodynamic model
-------------------

:class:`~atmodeller.thermodata.mixing.SubregularSolution` uses an ordered pair
of components, with :math:`x=n_1/(n_0+n_1)`. All public energies are per mole of
those native components, temperature is K, and pressure is bar. With the
default ``component_0_multiplicity=1``:

.. math::

   g_{\rm mix}=RT[(1-x)\ln(1-x)+x\ln x]
      +x(1-x)[A(1-x)+Bx](1-T/C+P/D).

Infinite C or D disables the respective dependence. Setting A=B and disabling
both dependences recovers the symmetric regular solution. The homogeneous
partial molar mixing energies are calculated from this same function:

.. math::

   \mu_0^{\rm mix}=g_{\rm mix}-xg'_{\rm mix},\qquad
   \mu_1^{\rm mix}=g_{\rm mix}+(1-x)g'_{\rm mix},\qquad
   \ln a_i=\mu_i^{\rm mix}/RT.

The pure-component standard Gibbs energies remain in ``ChemicalSpecies.thermo``.
They must use the same component basis and compatible standard states. Do not
add a solubility correction to the mixing activities without deriving the
combined free energy and its reference states.

* **Spinodal:** homogeneous :math:`g''_{\rm mix}<0`; infinitesimal composition
  fluctuations lower free energy. The limits solve :math:`g''_{\rm mix}=0`.
* **Binodal:** the compositions of two coexisting phases, found by equal chemical
  potentials for both components (a common tangent).
* **Metastable:** inside the binodal but outside the spinodal. A finite
  fluctuation can lower free energy even though the local curvature is positive.

The equilibrium activity uses the common tangent throughout the binodal gap,
including the metastable part. Separation is therefore not triggered only when
the spinodal is crossed. For bulk composition x between endpoints
:math:`x_\alpha,x_\beta`, the fraction of component moles in the upper-composition
phase is :math:`f_\beta=(x-x_\alpha)/(x_\beta-x_\alpha)`. This is not a mass
fraction. The two component inventories are recovered by the lever rule.

Hydrogen-silicate parameterization
----------------------------------

``SubregularSolution.hydrogen_silicate_gilmore26()`` implements the final
`Gilmore & Stixrude (2026) model
<https://doi.org/10.1038/s41586-025-09970-4>`_, equations 1 and 8. It uses
A=75000 and B=424000 J/mol of fit-basis units, C=4400 K and D=-350000 bar.
The paper defines :math:`z=n_{\rm H2}/(n_{\rm H2}+n_{\rm MgO}+n_{\rm SiO2})`.
One native MgSiO3 formula unit contributes two fit-basis units, so this
constructor sets ``component_0_multiplicity=2``. Consequently,

.. math::

   z=\frac{x}{2-x},\quad x=\frac{2z}{1+z},\quad
   G_{\rm mix}=(2n_0+n_1)g_{\rm fit}(z),\quad
   g_{\rm native}(x)=(2-x)g_{\rm fit}(z).

Both activities follow derivatives of this extensive energy:
:math:`\mu_{0,\rm native}=2\mu_{0,\rm fit}` and
:math:`\mu_{1,\rm native}=\mu_{1,\rm fit}`. Transforming compositions alone
does not preserve chemical potentials. ``mixing_composition`` and
``native_composition`` expose this conversion; all public phase fractions,
inventories and spinodal/binodal endpoints remain on the native basis.

For a closed parcel containing 3 wt% H2, use the native molar masses to
convert the mass fraction before calling the mixing model::

    from molmass import Formula
    from atmodeller import SubregularSolution

    w = 0.03
    nh = w / Formula("H2").mass
    ns = (1 - w) / Formula("MgSiO3").mass
    x = nh / (nh + ns)
    mixing = SubregularSolution.hydrogen_silicate_gilmore26()
    diagnostics = mixing.diagnostics(3000.0, 20000.0, x)
    spinodal_temperature = mixing.spinodal_temperature(20000.0, x)

This model predicts a metastable homogeneous parcel at these conditions,
with a spinodal temperature of about 2942 K. ``spinodal_temperature(P, x)``
analytically inverts the curvature; NaN denotes no finite positive root or
invalid input. It supplies no nucleation or growth rate. The helper
``equilibrium_gibbs_energy(T, P, x)`` returns the common-tangent convex envelope.

The focused study in ``research/spinodal_onset/`` audits this fit against the
authors' tabulated simulations. It does not reproduce every coexistence pair
within the quoted composition uncertainties. Treat its delayed-exsolution
results as model-dependent bounds, not validated planetary predictions.

Historical parameterization
~~~~~~~~~~~~~~~~~~~~~~~~~~~

``SubregularSolution.hydrogen_silicate_rogers25()`` implements equation 7 of
`Rogers, Young & Schlichting (2025)
<https://doi.org/10.1093/mnras/staf1940>`_ (the explicit equation and parameters are
also in `arXiv:2509.13320v1 <https://arxiv.org/html/2509.13320v1#S2.SS1.SSS1>`_).
Component 0 is MgSiO3 and component 1 is H2. A=-6260 and B=786000 J/mol,
C=4670 K, and D=-35 GPa, converted here to -350000 bar.

This is the published phenomenological fit, not a new fit to the final 2026
DFT results, nor a calibration for water-silicate, hydrogen-helium, or a
multicomponent magma. A finite numerical result does not establish that a
parameterization is valid outside its source's physical regime.

For example::

    from atmodeller import SubregularSolution

    mixing = SubregularSolution.hydrogen_silicate_rogers25()
    result = mixing.diagnostics(3975.0, 10000.0, 0.6)
    coexistence = mixing.coexistence(3975.0, 10000.0)

At these inputs the implementation gives binodal H2 mole fractions
0.4349126578 and 0.9469194554, and spinodal limits 0.5721591077 and 0.8787976303.
The bulk composition 0.6 is locally unstable and the equilibrium upper-phase
mole fraction is 0.3224319344. These numbers describe that mixing model and do
not by themselves give atmospheric molecular abundances.

Using the native equilibrium engine
-----------------------------------

Select the treatment beside the temperature, pressure and composition inputs::

    from atmodeller import ChemicalSpecies, EquilibriumModel, SubregularSolution

    model = EquilibriumModel.from_species(
        [ChemicalSpecies.create_gas(s) for s in ("He", "Fe", "Si")],
        temperature=2000.0,
        pressure=1.0,                       # bar
        mass=0.0,                           # no inert solvent
        metal_species=[ChemicalSpecies.create_solution(s) for s in ("Fe", "Si")],
        mixing_models={"metal": SubregularSolution.regular_solution(50000.0)},
        phase_equilibrium="spinodal",
        mass_constraints={"He": 1.0, "Fe": 0.75, "Si": 0.25},
        mass_units="moles",
    )
    demixed = model.solve_with_default()
    homogeneous = model.with_phase_equilibrium("homogeneous").solve_with_default()
    warmer = model.update_state(temperature=2500.0, phase_equilibrium="spinodal")

``spinodal`` is the API name for common-tangent **equilibrium** demixing. It
includes metastable mixtures inside the binodal; it is not a rule that waits for
negative curvature. ``homogeneous`` retains the same nonideal free energy but
constrains the solution to one phase. This isolates separation from the effect
of switching the interaction model itself.

The selector is also accepted by ``ThermodynamicState.from_species``,
``Planet.from_species``, ``EquilibriumModel.from_state`` and the main
``EquilibriumModel(parameters, phase_equilibrium=...)`` constructor.
``None`` preserves existing phase settings. Selecting spinodal equilibrium
without a configured binary mixing model raises an error.

For a binary planetary metal phase use ``core_mass_fraction=None``. A fully
tracked binary silicate phase likewise needs zero background mass. Temperature,
pressure, inventories and equilibrium mode can be updated normally; changing
planetary background mass allocation with binary phases requires reconstructing
the state so a hidden solvent cannot be introduced.

:class:`~atmodeller.phases.BinarySolutionPhase` subclasses ``CondensedPhase``.
It overrides the same ``get_log_activity`` method used by the reaction engine,
with no new residual block, duplicate component inventory or separate solver.
Its ``BinarySolutionPhaseOutput`` follows the existing phase/output split and
adds diagnostics at ``output[phase_name]["phase"]["mixing"]`` in all dictionary
formats and DataFrame exports.

``spinodally_unstable`` and ``metastable`` describe the hypothetical homogeneous
mixture at the bulk composition. A stable, demixed equilibrium can therefore
report ``spinodally_unstable=True``: its two actual coexisting compositions lie
on the stable branches. ``demixing_enabled`` records which activities were used.

Create its two components with ``ChemicalSpecies.create_solution``. Their
amounts are solved, their masses count in the phase, and they have no separate
pure-component stability variables. A binary solution cannot contain an inert
background, reservoir/solubility species, or additional independent activity
models. The constructor rejects inconsistent component configurations.

The phase assumes the solution remains present. Collective appearance or
disappearance of the entire solution phase is not implemented. Check that the
solution has a finite inventory in the intended physical regime; the pure-phase
stability variables are not a substitute for this missing collective condition.

For low-level assembly, use ``PhaseSystem`` and ``ReactionSystem``. Preserve
the engine's phase order: gas, silicate melt, silicate solid, metal, then any
additional phases, using empty phases as placeholders where necessary. A
planetary state can likewise be built with a reaction system containing this
phase; no change to the elemental conservation equations is needed.

The public factories handle that ordering automatically. Run the complete example::

    uv sync --locked --all-extras --dev
    uv run python examples/spinodal_equilibrium.py

.. literalinclude:: ../examples/spinodal_equilibrium.py
   :language: python
   :pyobject: gas_solution_example

The example uses database Fe/Si species and an **illustrative, uncalibrated**
regular-solution interaction. It verifies gas coupling; it does not predict real
Fe-Si immiscibility. At 2000 K and 1 bar its ideal-solution Si partial pressure
is 1.08830e-5 bar, compared with 4.10549e-5 bar after equilibrium demixing with
Omega=3RT. Each elemental inventory is conserved. Setting ``equilibrate=False``
computes the constrained homogeneous case, which can be unstable; numerical
convergence is not a claim of stable physical equilibrium.

Numerics and differentiability
------------------------------

The subregular curvature has one minimum because its fourth derivative is
positive. Bracketed searches find the spinodal limits, then the common tangent
between the two stable branches. This excludes the spurious equal-composition
solution of unconstrained chemical-potential matching. Log-odds coordinates
resolve exponentially dilute coexisting components.

Equinox modules retain the model parameters in their pytrees. Array inputs
broadcast elementwise; the methods support JIT and vmap. Implicit derivatives
of the spinodal and common-tangent equations are used instead of differentiating
through bisection decisions. The two-phase derivatives are singular at a critical
point and phase-boundary derivatives are branch-dependent. Do not assume global
smoothness across changes of phase regime.

``BinaryCoexistence.valid`` checks finite inputs, the separated branch brackets,
log odds within +/-699 (the search interval is +/-700), and chemical-potential
mismatch below 1e-8 in mu/RT. An invalid tangent gives NaN activities, rather
than reverting to a homogeneous state. When there is no gap, ``has_gap=False``;
the raw coexistence object uses equal-composition sentinels and diagnostics
report NaN boundaries. Always inspect these flags. Spinodal searches use
composition bounds 1e-14 and 1-1e-14; extremely large interaction parameters
outside this numerical range are not supported.

Tests check analytic ideal/regular limits, Gibbs-Duhem consistency, a global
supporting tangent, independent SciPy coexistence roots, component conservation,
component reversal, pressure conversion, JIT/vmap, derivatives, dilute and
near-critical cases, invalid calculations, native gas equilibrium and exports.

What is needed for an atmospheric prediction
--------------------------------------------

The next physical step is to reconcile the H2-MgSiO3 free energy and both
endmember standard states with the reacting atmospheric network. An H2-rich
branch of this binary model is not automatically an independently speciated
ideal-gas atmosphere; counting both as the same hydrogen reservoir would be
inconsistent. Partitioning of additional elements needs a multicomponent
free energy or independently justified transfer chemical potentials.

A time-dependent spinodal model additionally needs a gradient-energy coefficient
and a mobility/diffusion model, with a consistent density or molar-volume
convention. Transport, settling and cooling then determine whether the
equilibrium end state is reached. None of those material parameters are inferred
from the binodal or supplied here. The present code supplies thermodynamic
driving forces and a validated equilibrium limit for that later development.

Phase-resolved output
---------------------

``output.to_dict()[phase_name]["phase"]["subphases"]`` contains ``lower`` and
``upper`` inventories for the **selected** mode. Each provides ``present``,
``mass`` (kg), ``number_moles`` (mol), ``component_1_mole_fraction`` and a
``species_number_moles`` dictionary. In homogeneous mode all material is in
``lower`` at the bulk composition and ``upper`` is absent. An absent phase has
zero inventory and a NaN composition. These output subdivisions sum to the
parent inventory and are not counted again in the solver.

The ``mixing`` output also includes ``actual_mixing_gibbs_J_per_mol`` and
``homogeneous_mixing_gibbs_J_per_mol``. Their difference quantifies the mixing
free-energy reduction, not heat release or a decomposition timescale.

Heat of mixing and of separation
--------------------------------

The same free energy gives the energies of mixing by automatic differentiation:
``enthalpy``, ``entropy``, ``volume`` and ``heat_capacity`` of the homogeneous
mixture, and ``partial_enthalpies`` of its two components. ``equilibrium_enthalpy``,
``equilibrium_entropy`` and ``equilibrium_volume`` average the coexisting phases
by the lever rule; the envelope theorem makes them equal the derivatives of
``equilibrium_gibbs_energy``. ``equilibrium_heat_capacity`` is the apparent heat
capacity of a closed mixture kept in equilibrium, which includes the heat
released as the phase compositions move.

``demixing_heat`` is the heat a homogeneous mixture releases when it separates at
fixed temperature and pressure (J per native mole). It is zero outside the gap
and continuous at the binodal. ``exsolution_enthalpy`` is the heat released per
mole of upper phase that forms at coexistence and is carried away. For the
hydrogen-silicate fits, the hydrogen partial enthalpy is 40-90 MJ per kg of
hydrogen, so separation is strongly exothermic. ``binodal_temperature`` gives the
equilibrium separation onset of a composition.

The ``mixing`` output reports the homogeneous and equilibrium enthalpy, entropy
and volume, ``demixing_heat_J_per_mol``, and ``actual_mixing_enthalpy_J_per_mol``
for the state the solver used.

Nucleation barriers
-------------------

``nucleation_barrier`` returns the square-gradient (Cahn-Hilliard) critical
nucleus of a metastable homogeneous parent and the classical estimate with the
same planar tension. It needs a gradient coefficient kappa (J m²/mol, for the
mixing-basis composition) and a constant molar volume per mixing-basis mole.
``gradient_coefficient`` converts a measured 10-90 % interface width at one state
into kappa. The barrier scales exactly as kappa^(3/2) and as 1/volume, tends to
classical theory near the binodal and vanishes as the 3/2 power at the spinodal.
These are thermodynamic barriers; a rate needs a prefactor, volume and time.

Cooling paths
-------------

``atmodeller.cooling.ParcelCooling`` cools a closed parcel at fixed pressure,
with heat lost per kilogram as the path coordinate. The parcel stays homogeneous
to an onset temperature: the binodal (equilibrium path), the spinodal, or
``nucleation_onset_temperature`` for a barrier criterion W/kT = ln(J0 V t). It
then separates isenthalpically, so it reheats (recalescence), or isothermally.
The final state and heat lost do not depend on the onset. Sensible heat comes
from the pure-component standard-state enthalpies, without a pressure term.

``InteriorCooling`` describes a convecting interior that exsolves at its top into
an envelope. The interior composition follows the lower binodal branch at the
top pressure, and the exsolved component leaves as a pure phase. The path reports
exsolution and sensible heat, and their ratio per kelvin of cooling.
``solve_path_with_engine`` replays a parcel path with two prebuilt equilibrium
models, homogeneous and demixing, so the solver never recompiles. For a closed
parcel the solver reproduces the fast functions; gas exchange can join later.

None of these add gravitational settling energy, transport, or envelope physics.

Worked studies
--------------

``research/spinodal_onset/README.md`` examines binodal-to-spinodal delay during
isobaric cooling using the final 2026 parameterization, an explicit basis
conversion, published-data audits and parameter stress tests. The native
solver is checked in 27 closed-binary cases. An imposed no-nucleation limit
is compared with instantaneous equilibrium; neither calculates kinetics.

``research/spinodal_study/README.md`` compares homogeneous and demixed H2-MgSiO3
inventories through the public API, then runs conditional hot atmospheric parcel
chemistry with selective transfer, entrainment, condensation and complete-remixing
controls. Scripts, all numerical outputs, figures and validation are included.
This earlier study uses the historical 2025 fit and is retained unchanged.
It is a one-way inventory-transfer experiment, not a calibrated reacting
high-pressure fluid or self-consistent planetary atmosphere calculation.
