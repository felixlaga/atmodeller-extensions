# Protocol amendment: resolving the source-data audit

This file extends [PROTOCOL.md](PROTOCOL.md), which stays unchanged because the original analysis records its hash.

Date: 29 October 2026 (evening).

Written after the audit in PROTOCOL.md and after one short smoke test of the recalibration. That test showed that a composition-only treatment of phase states lets a fit drop an observed two-phase run, which motivated the phase-state term below. The full runs came afterwards. These are post-result analyses and are labelled as such everywhere.

1. **Read the source paper's own equations and figures.** The full text (Europe PMC XML) gives equation 8 and the parameters, but not which component carries `A`. Assigning `B` to the H2-rich side must reproduce the published Figure 2 curves; this is checked by digitizing Figure 2a (`digitize_source_figure.py`) against an independent scalar implementation (`mixing_models.py`) that shares no code with Atmodeller.
2. **Compare ED Table 1 with Figure 2a.** Both are kept unmodified. Where they differ, both are analysed as separate data versions; neither replaces the other.
3. **Recalibrate (`calibration.py`).** Joint likelihood of the 14 phase states (in temperature, sigma_T = 50 K plus 100 K/GPa times the quoted pressure error), the 8 coexisting compositions (quoted +/-0.05 as 1-sigma) and the 9 homogeneous excess properties (full covariance, including shared endmembers and the 5.00 vs 5.06-5.08 GPa offset). Four model classes with constant excess H, S, V: M1 (published form), M2 (independent subregular H, S, V), M3/M4 (Redlich-Kister order 2/3 with the published T, P scaling). Data variants for M1: Figure 2a compositions, sigma_x = 0.15 for the two near-critical runs, no volumes, phases + compositions only, phases + homogeneous only. Uniform priors (bounds in the code), inverse tau and pi sampled, adaptive Metropolis with 8 chains x 12,000 steps (3,000 adaptive burn-in discarded), split R-hat < 1.1 required.
4. **Design screen (`design.py`).** Linear-Gaussian expected posterior SD of the 3 wt%, 2 GPa spinodal after one new homogeneous simulation, or after a 10% measurement of the composition curvature. A screen, not a substitute for refitting real data.

The published fit stays the baseline model of the study and of Atmodeller. The recalibrated classes are uncertainty estimates, not recommended replacements: none of them fits all source data within the quoted errors, and all share the assumption of temperature-independent excess enthalpy and entropy.
