# Delayed hydrogen exsolution: what the source data allow

**Answer:** hydrogen that leaves the magma releases heat, and none of the evolution models we checked includes it. Refits that use the simulated excess enthalpies give 35–93 MJ per kg of hydrogen. The published 2026 fit gives 51–64 MJ/kg. Both ranges cover 1–3 wt% hydrogen at 2–5 GPa.

**Exsolution stays within a few kelvin of equilibrium.** Take a 3 wt% parcel at 2 GPa with the simulated interface width. Homogeneous nucleation then ends metastability 1.2–2.9 K below the binodal in the published fit, and 0.2–1.6 K in refit M1. At most 1.3% of the hydrogen is withheld. So the published 414 K window between binodal and spinodal is a mean-field upper bound. A cooling parcel does not reach it. Refits of the same source data give median windows of 41 to 242 K. A 10% measurement of the composition curvature at 3600 K would lower the spinodal uncertainty only from 80 to 64 K.

## The question

A homogeneous H2–MgSiO3 parcel that cools enters the miscibility gap at the **binodal**. It becomes unstable only at the **spinodal**. In between it is metastable: separation needs nucleation. If nucleation is suppressed, hydrogen release is delayed until the spinodal.

The study asks how much heat exsolution releases, how far below the binodal nucleation lets a parcel go, and how well the existing simulations constrain both.

## Results

The table gives the mean-field bound: the window between binodal and spinodal, and the hydrogen a parcel delayed to the spinodal would withhold.

| Model, 3 wt% H at 2 GPa | Binodal (K) | Spinodal (K) | Window (K), 68% range | H withheld at spinodal |
|---|---:|---:|---:|---:|
| Published 2026 fit | 3356 | 2942 | 414 | 65.5% |
| Refit M1, all data | 3182 | 3096 | 92 (69–120) | 49% |
| Refit M2, independent H, S, V | 3190 | 3152 | 41 (22–65) | 40% |
| Refit M4, most flexible | 3183 | 3072 | 161 (98–284) | 48% |
| Historical 2025 fit | 3962 | 3908 | 53 | 39.0% |

Binodal and spinodal of the refits are best-fit values. Window and withheld hydrogen are posterior medians. The 95% range of M4 is 62–479 K, and 4.9% of its samples reach the published 414 K, so the most flexible class does not exclude it. `outputs/calibration_summary.json` lists all nine refits.

### How the audit mismatch was resolved

The earlier audit found that the published fit matches only one of four two-phase compositions. That mismatch is in the source, not in Atmodeller:

- **Our implementation reproduces the published Figure 2a curves** to 2.3 K RMS (largest deviation 15 K). An independent solver agrees with Atmodeller to 10⁻⁶ K over the grid, and to 10⁻⁷ K within 0.02 of the critical composition.
- **The source table and figure disagree** for the two runs closest to the critical curve, by up to 0.07 in composition. For the two wide-gap runs they agree to 0.0006.
- **The published fit also misses its own excess volumes**: −1.94 cm³/mol predicted against −0.94 ± 0.16 simulated at x = 0.588. The fitted pressure scale follows the critical curve instead.
- **It also underestimates the excess enthalpy of the most hydrogen-poor mixture**: 24.5 kJ/mol predicted against 36.3 ± 5.9 simulated at x = 0.238, two standard deviations low.

The source data disagree with each other. Without the homogeneous data, the M1 refit returns to the published asymmetry (B/A ≈ 6) but with a larger excess entropy, and gives 134 K. With them, M1 is nearly symmetric (B/A ≈ 1.4–1.5), and the window is 67 to 95 K. Wider errors (σx = 0.15) on the near-critical compositions give B/A = 2.5 and 242 K. The window depends on the asymmetry, the excess entropy and the interaction strength together.

### Why equilibrium data cannot settle it

A perturbation that leaves every equilibrium observable unchanged still moves the spinodal. We cap its amplitude at λ = 0.22, the largest value that keeps every heat capacity positive. At that cap the spinodal rises by 124 K at 2 wt% hydrogen and by 16 K at 3 wt%, and falls by 5 K at 5 wt%. So any spinodal inferred from coexistence data depends on the chosen model form.

### What would narrow it

- One more homogeneous simulation of excess H, S and V, at x ≈ 0.35, helps little. It lowers the spinodal uncertainty (posterior standard deviation) from 80 to 72 K.
- **A 10% measurement of the composition curvature** ∂²g/∂x² at 3 wt%, 2 GPa and 3600 K lowers it to 64 K. The concentration structure factor S_cc(0) = RT/(∂²g/∂x²) gives this curvature without a free-energy model.
- Neither measurement settles the spinodal. At 3600 K every screened posterior sample is above its critical temperature (at most 3547 K), so the parcel is one phase at every composition. The measurement reaches the spinodal only through the temperature dependence of the model classes. The counterexample above changes the free energy only inside the gap, so this measurement cannot detect it.

## Scope

These are thermodynamic estimates and square-gradient nucleation-barrier onsets for a closed binary parcel and a convecting interior. The study does not compute planetary radii or spectra. The published fit stays the baseline model of Atmodeller. The refits are uncertainty estimates, not replacements: none fits all source data within the quoted errors.

All post-audit analyses are listed in the [protocol amendment](PROTOCOL_AMENDMENT.md). The source data of Gilmore & Stixrude (2026) are used under CC-BY-4.0 and stored unmodified in `sources/`.

## Reproduce

From the repository root:

```sh
uv sync --locked --all-extras --dev
PYTHONPATH=. uv run python research/spinodal_onset/run_study.py
PYTHONPATH=. uv run python research/spinodal_onset/identifiability.py
PYTHONPATH=. uv run python research/spinodal_onset/digitize_source_figure.py
SPINODAL_WORKERS=2 PYTHONPATH=. uv run python research/spinodal_onset/calibration.py   # workers default 2; about 40 min with 8
PYTHONPATH=. uv run python research/spinodal_onset/design.py
PYTHONPATH=. uv run python research/spinodal_onset/make_figures.py
for s in energy edtable2 interior; do
  PYTHONPATH=. uv run python research/spinodal_onset/heat_study.py $s; done
PYTHONPATH=. uv run python research/spinodal_onset/heat_study.py barriers --start 0 --stop 201
PYTHONPATH=. uv run python research/spinodal_onset/heat_study.py barriers --start 201 --stop 402
for s in delays paths checks planet; do
  PYTHONPATH=. uv run python research/spinodal_onset/heat_study.py $s; done
cd research/spinodal_onset && PYTHONPATH=../.. uv run python heat_figures.py && cd ../..
```

Check the delivered artifacts with:

```sh
PYTHONPATH=. uv run python research/spinodal_onset/verify_artifacts.py
```

| Files | Purpose |
|---|---|
| `mixing_models.py` | Independent scalar solver for T-linear Redlich–Kister models |
| `digitize_source_figure.py`, `outputs/source_figure_digitized.json` | Published Figure 2a against our implementation |
| `calibration.py`, `outputs/calibration_*` | Nine recalibrations, posterior samples and diagnostics |
| `design.py`, `outputs/design_*` | Which new simulation constrains the spinodal |
| `run_study.py`, `identifiability.py` | Onset grid, cooling paths, audit and the identifiability test |
| `heat_study.py`, `heat_figures.py`, `outputs/heat/` | Heat of exsolution, nucleation onsets, cooling paths, the interior case, its sensitivity checks and the planet-scale scaling |
| `sources/` | ED Tables 1–2 and Figure 2 of the source paper, and digitized Figures 5–6 of Rogers et al. (2025), with hashes |
| `explorer.html` | Interactive cooling paths of the published model |

## Primary references (checked 29 October 2026)

- [Gilmore & Stixrude 2026, Nature 650, 60–64](https://doi.org/10.1038/s41586-025-09970-4), equations 1 and 8–19; [source data](https://doi.org/10.5281/zenodo.17716049).
- [Rogers, Young & Schlichting 2025, MNRAS 544, 3496](https://doi.org/10.1093/mnras/staf1940); [Rogers & Schlichting 2026](https://arxiv.org/abs/2606.30644v2).
- [Young et al. 2025, PSJ 6, 251](https://doi.org/10.3847/PSJ/ae1012); [Young & Werlen 2026](https://arxiv.org/abs/2604.28135v2); [Tejada Arevalo et al. 2026](https://arxiv.org/abs/2601.00059v2).
- [Hakim et al. 2026, MNRAS 546, stag133](https://doi.org/10.1093/mnras/stag133); [Bower et al. 2025, ApJ 995, 59](https://doi.org/10.3847/1538-4357/ae1479).
