# Condensed-phase volumes at sub-Neptune magma-envelope boundaries

**Answer: the volume of liquid SiO2 is not a small correction.** At the GPa pressures of a sub-Neptune magma-envelope boundary (MEB), it raises the equilibrium constants of SiO2 vaporisation and reduction by 55 times at 4 GPa and about 2,000 times at 8 GPa (3000 K). In the TOI-421 b model of Hakim et al. (2026), adding it leaves no boundary below 20 GPa for 0.3 wt% H or more. At the published pressures, it erases the metallicity signal in SiH4/CH4 for most 100x solar cases.

Published sub-Neptune equilibrium models keep melt and metal at their 1-bar Gibbs energies. Some of them correct the gases for pressure. Schlichting & Young (2022) and Werlen et al. (2026) tested the condensed term and called it small, mostly on reactions within the melt, where it nearly cancels. This study measures it for the reactions that exchange matter with the gas.

## What changed in Atmodeller

- `LiquidTaitEOS` in `atmodeller/thermodata/holland_powell.py`: the liquid equation of state of Holland & Powell (2011). `LIQUID_VOLUMES` holds the SiO2, MgO, MgSiO3 and Mg2SiO4 liquids of their data set ds62. Volume integrals agree with BurnMan 2.1.0 to about 1e-15.
- `DataBase.default(liquid_volumes=True)` gives these four liquids their volumes. The option is off by default, so published results do not change. With it, the SiO2 and MgO liquids are selected on their own, not stitched to their solids.
- Tests: `test_holland_powell_liquid_volume` (ds62 values, BurnMan, quadrature), `test_default_liquid_volumes` and `test_silica_liquid_volume`. The last solves SiO2 liquid with SiO, O2, H2 and H2O at 3000 K and checks that the vapour shifts by exactly exp(integral of V dP / RT).

## Results

All runs use the setup of Hakim et al. (2026) from the authors' OSF notebook. `port_check.py` reproduces all 100 cases of their fully molten grid from a cold start: mole fractions agree with their workbooks to within 2.1% and pressures to within 0.15%.

### 1. The correction is large for reactions with the gas, and small within the melt

Factor on the equilibrium constant at 3000 K from the condensed volumes alone (`correction_table.py`, `figures/correction_factors.png`). The range in brackets covers the volume models of the condensed species.

| Reaction | 1 GPa | 2 GPa | 4 GPa | 8 GPa | 13 GPa |
|---|---:|---:|---:|---:|---:|
| SiO2(l) to SiO, SiH4 or H2O | 2.9 | 7.9 | 55 (48-55) | 2,100 (1,400-2,100) | 160,000 (65,000-160,000) |
| MgSiO3(l) = MgO(l) + SiO + 1/2 O2 | 2.8 | 7.4 | 46 | 1,400 | 67,000 |
| MgO(l) = Mg + 1/2 O2 | 1.8 | 3.3 | 10 | 94 | 1,300 |
| FeO(l) + H2 = Fe(l) + H2O (bracket) | 1.5 | 2.2 | 4.5 (3.7-4.7) | 16 (11-18) | 64 (35-90) |
| MgSiO3(l) = MgO(l) + SiO2(l) | 0.97 | 0.93 | 0.84 (0.84-0.95) | 0.65 (0.65-0.96) | 0.43 (0.43-1.03) |
| Mg2SiO4(l) = 2 MgO(l) + SiO2(l) | 0.91 | 0.84 | 0.73 (0.73-0.83) | 0.60 (0.60-0.89) | 0.49 (0.49-1.18) |

- For MgO(l) = Mg + 1/2 O2 at 4 GPa, Delta ln K is 2.2 to 2.5 between 2500 and 3500 K. Werlen et al. (2026, Fig. 5) computed the same reaction and called the change small.
- Within the melt, the correction stays below a factor of about 2, and its sign depends on the SiO2 volume model.
- The FeO and Fe volumes are an illustrative literature bracket and are not in Atmodeller. The sign agrees with Horn et al. (2025): pressure favours water and metal. The size, about 4.5 to 16 times at 4 to 8 GPa, is far below the 2,000-3,000 times excess of water that they report.

### 2. The published boundaries disappear

`boundary_map.py` fixes the MEB pressure on a grid from 0.3 to 20 GPa and computes the pressure that the equilibrium gas would exert. A boundary exists where the two are equal (`figures/boundary_map.png`). Stable boundaries in GPa at 1x solar and 3000 K, with the SiO2 liquid volume scaled from 0 to 1:

| Volume included | 0.1 wt% H | 0.3 wt% H | 1 wt% H |
|---|---:|---:|---:|
| none (published) | 1.26 | 2.96 | 7.67 |
| 10% | 1.32 | 3.14 | 8.39 |
| 20% | 1.38 | 3.39 | 10.1 |
| 30% | 1.46 | 3.80 | none |
| 50% | 1.72 | none | none |
| 100% | 14.7 | none | none |

- At 100x solar the pattern is the same: 1.06, 2.60 and 6.95 GPa without the volume, and with it only 0.1 wt% H keeps a boundary, at 13.7 GPa.
- The 14.7 GPa boundary is a different state, with a gas of 99.9% SiO and SiH4.
- For 0.3 wt% H or more, no boundary remains with a second SiO2 melt volume (Lange & Carmichael 1987, K' = 4), at 3500 K, or with ideal gases.
- For 0.1 wt% H the equation has several roots. With ideal gases, a stable boundary survives near 1.5 GPa (1.35 at 100x solar), beside an unstable root near 8.6 GPa and a second stable one near 14 GPa. At 2500 K the 0.1 wt% H boundaries survive near 0.6-0.7 GPa, and 0.3 wt% H at 100x solar keeps one at 2.4 GPa.
- From a cold start, the solver puts all 22 cases of `four_modes.py` at 161-211 GPa, with all the Si in the gas and no SiO2 liquid left. These states are artefacts: with real gases the gas has a mole fraction of H2 below 1e-28 and log10 fO2 of 30 to 37, and every equation of state and solubility law is far outside its range. They mark the absence of a boundary below 20 GPa, and say nothing about the real planet.

### 3. At the published pressures, the gas becomes more oxidised

Holding the published pressure of each case fixed (1.06-7.69 GPa) and adding the volume (`four_modes.py`, `outputs/published_pressure_shift.csv`):

- SiO rises 1.4 to 20 times, and SiH4 falls to 0.006 to 0.7 of its value, as H2 turns into H2O.
- H2O/H2 rises 1.9 to 111 times, and fO2 rises by 0.6 to 4.1 log units.
- In a closed 920 K, 10 mbar parcel of each boundary gas, SiH4/CH4 stays at 0.0064 at 1x solar, where Si, SiC, SiO2 and Si3N4 condense. At 100x solar it jumps from 2.3e-10 to 0.0064 for 0.16 wt% H or more, because graphite gives way to Si among the stable condensates.
- So the 7.4 dex contrast in SiH4/CH4 between 1x and 100x solar, which Hakim et al. (2026) propose as a diagnostic of metallicity, is gone in these cases. This is at the published pressures, which section 2 shows are not self-consistent once the volume is included. The parcel differs from the authors' atmosphere, which solved the pressure over 1 mbar to 100 bar for the 1 wt% H gas.

### 4. Bower et al. (2025) K2-18 b: only through the iron-wustite buffer

The K2-18 b runs have no condensed species in their reactions, so the volumes enter only through the IW buffer (`bower_iw.py`). Evaluating it at the MEB pressure instead of 1 bar moves the CH4 = H2 crossing from 0.227% to 0.245% H, changes CH4/H2 by up to 2.7 times (at 2.3% H) and lowers the MEB pressure by up to 14%.

## What this does and does not show

- **The gas side is still not consistent.** Gases mix ideally, each with its pure-species fugacity coefficient (SiO from Connolly 2016, SiH4 from Wang 2018). A dense H2-silicate fluid has its own mixing volumes, which would resist taking up silicate. So the loss of the boundary shows that the model has no consistent two-phase state, not that the real planet has none. Gilmore & Stixrude (2026) still find two phases at 3000 K below about 6.6 GPa.
- **What it does show:** the condensed term is as large as the gas terms that models already include. A GPa boundary model that applies one without the other is biased in a known direction, toward a melt that is too stable.
- The setup is the authors', including two choices that this study keeps: SiO2 liquid at unit activity, and the mantle Si and O counted both in the SiO2 liquid and in the background mantle mass, which sets the gravity. A real melt has a lower SiO2 activity, which lowers the vapour but not the correction factor.
- The solubility laws carry no volume for dissolved H2 or H2O.
- The parcel is a closed equilibrium: no rainout, quenching or photochemistry.

## Reproduce

From the repository root, one script at a time, single-threaded:

```sh
export OMP_NUM_THREADS=1 XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
cd research/condensed_volumes
uv run python fetch_sources.py      # authors' OSF files, checked by SHA-256
uv run python port_check.py         # about 10 min
uv run python correction_table.py   # seconds
uv run python boundary_map.py       # about 40 min
uv run python four_modes.py         # about 15 min
uv run python bower_iw.py           # about 5 min
uv run python make_figures.py
```

Outputs go to `outputs/` and figures to `figures/`. The authors' files are downloaded to `sources/` and are not stored in the repository.
