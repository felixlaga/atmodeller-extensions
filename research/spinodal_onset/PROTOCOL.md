# Delayed hydrogen separation: focused study protocol

Date: 2026-10-29. This protocol was written after literature review and four exploratory phase-diagram evaluations, before running the study grid. It is a local analysis plan, not a registered preregistration.

## Question and primary estimands

For a closed, initially homogeneous H2-MgSiO3 parcel cooling at fixed pressure, how far apart are the first equilibrium separation temperature (binodal) and loss of local stability (spinodal), and how much hydrogen would already occupy a distinct H-rich phase at the latter temperature under instantaneous equilibrium?

The second number is a **maximum inventory discrepancy between two limiting assumptions**, not a calculated gas escape rate. The delayed limit suppresses nucleation until the spinodal and assumes rapid equilibration afterwards. The equilibrium limit demixes throughout the binodal. Neither determines a nucleation rate, the post-spinodal growth rate, settling, or connection to the observable atmosphere. Existing interfaces can remove the nucleation barrier altogether.

## Literature basis and novelty boundary

- Gilmore & Stixrude, Nature 650, 60-64 (2026), [final paper](https://doi.org/10.1038/s41586-025-09970-4), equations 1 and 8; [original tabulated simulations](https://doi.org/10.5281/zenodo.17716049). Primary thermodynamic model and external validation data. The tabulated results helped calibrate this published model, so the comparison is a reproduction audit, not independent out-of-sample validation.
- Rogers, Young & Schlichting (2025), [MNRAS model](https://doi.org/10.1093/mnras/staf1940), equation 7. Historical parameterization retained unchanged as a control.
- Rogers & Schlichting, [2026 v2, revised October 14](https://arxiv.org/html/2606.30644v2). Current motivation: hydrogen exsolution supplies evolving envelopes; the models assume thermochemical equilibrium. We do not reproduce their radius, escape or survey predictions.
- Tejada Arevalo et al., [2026 v2, revised March 18](https://arxiv.org/html/2601.00059v2). Coupled thermal/compositional evolution and silicate rain already exist; this study does not claim their invention.
- Young & Werlen, [2026 v2, revised June 28](https://arxiv.org/html/2604.28135v2), and Young et al., PSJ 6, 251 (2025). Multicomponent miscibility and binary binodal/spinodal diagrams are already published. Merely plotting a spinodal is not novel.
- Hakim et al., [MNRAS 2026](https://doi.org/10.1093/mnras/stag133). Atmospheric speciation and condensation are downstream processes; our prior hot-parcel enhancement factors are not observational predictions.

The candidate contribution is a quantitatively audited, composition-consistent bound on delayed exsolution, including its sensitivity to calibration and the limited constraint provided by existing coexistence simulations. Literature searches on October 29 included arXiv current version pages, journal sources, and Consensus. These establish nearby prior work, not an exhaustive proof of novelty.

## Required mechanism improvement

The 2026 paper defines z = n(H2)/[n(H2)+n(MgO)+n(SiO2)]. Atmodeller represents native components (MgSiO3,H2), with y=n(H2)/[n(H2)+n(MgSiO3)]. Therefore z=y/(2-y), and the native molar mixing energy is (2-y) times the published molar mixing energy. Native chemical potentials are (2 mu_oxide, mu_H2). Conversion of mole fractions alone fails the extensive-energy derivative test. The original 2025 constructor remains unchanged for reproducibility.

## Models and controls

1. Published 2026 fit, A=75 kJ/mol, B=424 kJ/mol, C=4400 K, D=-35 GPa, with the oxide-to-formula-unit conversion.
2. Historical 2025 fit on its stated formula-unit basis.
3. Deliberately incorrect 2026 direct parameter import without the basis conversion: a diagnostic control, never a recommended model.
4. Deterministic parameter perturbations at +/-10% in A, B, C and |D|, individually and in combination. These are stress tests, not posterior credible intervals or empirically calibrated parameter uncertainties. Exclude or flag states outside each model's positive excess-factor domain.

## Design

- Pressures 2, 5 and 10 GPa, with a finer 2-10 GPa threshold map.
- Local binary hydrogen mass fractions 0.1-20 wt%, explicitly distinct from whole-planet fractions.
- A primary fluid analysis window of 2000-4000 K; report inaccessible spinodals below that window separately. This is not a melting-curve model. Also test stopping at 2500 and 3000 K to expose dependence on the assumed cooling endpoint.
- Compute binodal crossings with bracketed roots; compare analytical spinodal temperatures to independent numerical roots of curvature. Verify continuity just above/below each crossing.
- Compare tabulated two-phase endpoints and single-phase classifications at the reported P,T and compositions. Preserve all mismatches; do not tune the published fit or discard inconvenient data.
- Use native EquilibriumModel solves for selected closed-binary cases to verify that the public API, activities, inventories and stability diagnostics agree with the standalone thermodynamics.
- A matched reduced parcel chemistry check may test whether the corrected inventory changes H2/H2O/Si gases at cool atmospheric conditions. Any such calculation is conditional on an imposed transport assumption and must be labelled secondary.

## Validation and interpretation

Numerical checks: extensive free-energy derivatives, Gibbs-Duhem, common-tangent support, conservation of each native component, independent SciPy roots, JAX forward/reverse derivatives, trace-component stability, baseline regression tests, and source/artifact hashes. Report absolute quantities and numerical tolerances.

No physical nucleation time, planetary mass loss, observable spectrum, fully calibrated multicomponent free energy or new ab initio calculation is claimed. Publication readiness requires a convincing novelty assessment by a domain specialist and resolution of any material mismatch with the source data. A robust null result or a calibration inconsistency is retained as a result.

## Audit extensions after the first grid run

The nominal coexistence mismatch motivated two additional checks: scanning the quoted pressure uncertainty while requiring a common pressure for both endpoints, and reconstructing excess enthalpy, entropy and volume from Extended Data Table 2. These are post-result diagnostic analyses, not pre-specified independent validation. Table 2's pure endpoints are at 5.00 GPa and mixtures at 5.06-5.08 GPa; no EOS correction is available here. Propagated errors assume independent table entries, with shared-endmember covariance between compositions retained as a limitation. No parameter is refitted.

The optional atmospheric chemistry calculation is not repeated: the previous study already establishes its dependence on imposed transfer and condensation. The present contribution isolates the upstream thermodynamic inventory uncertainty without adding a second arbitrary atmospheric transport assumption.

A further post-result structural test constructs a nonnegative, C2 free-energy bump inside the binodal, with zero value and first/second derivatives at each endpoint. It preserves all equilibrium boundaries and homogeneous states outside the gap. We evaluate amplitudes 0, 0.5, 1, 2, 4 and 8 at 2 GPa for 1, 2, 3 and 5 wt% local hydrogen. This demonstrates non-uniqueness of an inferred spinodal; the amplitudes are not a calibrated material uncertainty or a replacement physical model. Baseline supporting-tangent inequalities and unchanged Table 2 states are explicitly checked.
