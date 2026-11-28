# Examples

Each folder holds the starting geometries (`*.xyz`) and a commented `input.ini`. The geometries are rough
guesses: the package optimizes them with ORCA first and then searches on the DMC surface.

| folder | what it shows | starting geometries |
|---|---|---|
| `water/` | one molecule, no path: the search directions are the softest Hessian modes | `geometry.xyz` |
| `ammonia_inversion/` | a reaction path: the reactant minimum and the inversion saddle | `reactant.xyz`, `product.xyz`, `saddle_guess.xyz` |
| `hcn_isomerization/` | a reaction path with three targets: HCN, the saddle and HNC | `reactant.xyz`, `product.xyz`, `saddle_guess.xyz` |
| `water_multidet/` | the water search plus the optional, experimental last stage: DMC energies with a trial wave function of several determinants from Quantum Package 2. The input asks for `cutoff = auto`: the package chooses the cutoffs and tests them ([docs/multireference.md](../docs/multireference.md), section 10) | `geometry.xyz` |
| `model_study/` | the comparison of the search methods on analytic model surfaces ([docs/protocols.md](../docs/protocols.md), section 5) | none |

## Without a cluster

The demonstration runs the whole workflow on analytic model molecules. It needs only the Python packages:

```bash
python -m montepass demo water demo_water          # 12 s on a cluster node (2026-11-26)
python -m montepass demo ammonia demo_ammonia      # 41 s on the same node
```

## On a cluster

The examples need ORCA, PySCF and QMCPACK (and Quantum Package 2 for `water_multidet`) and a machine
profile (`machines/example_slurm.ini`, [docs/user_guide.md](../docs/user_guide.md), section 2). Copy an example
folder, set `profile` in `[machine]` to the name of your profile, then:

```bash
python -m montepass check  water      # checks the input and parts of the profile (user guide, section 2)
python -m montepass submit water      # one small driver job runs the whole workflow
python -m montepass status water      # stages, targets and the last lines of the log
```

## Our runs of these inputs

We ran the four inputs of version 0.1.0 twice with that version of the package, unchanged except for the
profile name, on a SLURM cluster with 32 cores per QMC job, with ORCA 6.1.0, QMCPACK 4.1.9 and Quantum Package 2.
The wall time is the time from the command `submit` to the report; it includes the time that the jobs waited
in the queue. The core-hours of the search are the column `core-hours: total` of `results/summary.md`, summed
over the targets. They count every job of the QMC points: the PySCF orbitals, `convert4qmc`, the VMC check,
the Jastrow optimization and DMC. The core-hours of the last stage of `water_multidet` are the column
`core-hours` of `results/trial_wavefunctions.md`, summed over its rows; they also hold the Quantum Package jobs.
The pilot runs, the ORCA jobs, the PySCF jobs of the surrogate and the driver jobs are not included. DMC is
stochastic, so a run of the same input on another machine gives other numbers within the error bars.

The input of `water_multidet` has changed since these runs. In version 0.1.0 its section `[trial]` was
`n_det_max = 10000`, `cutoff = 1e-3, hf` and `series = 2`, and every result of `water_multidet` in the first two
runs below is a result of that input. The input in this folder now asks for `cutoff = auto` (with the defaults
`n_det_max = 100000` and `auto_cutoffs = 2.5e-4, 1e-4, 5e-5`). The third run below ran that input with the
default ladder of that day, `2.5e-4, 1e-4`; for water the stage did not need a third cutoff. Why the
input changed: with the cutoff 1e-3 the DMC energies were above those of the single RHF determinant (see the
first two runs below and [docs/multireference.md](../docs/multireference.md), section 9).

### First run, 2026-11-25

Python environment of `requirements-cluster.txt` (Python 3.9.21, PySCF 2.1.1).

| example | status of every target | wall time | core-hours |
|---|---|---|---|
| `water` | CONVERGED after 2 scan iterations | 25 min | 43.2 |
| `ammonia_inversion` | reactant and saddle CONVERGED, 2 scan iterations each | 17 min | 22.7 |
| `hcn_isomerization` | HCN, saddle and HNC CONVERGED after 1, 3 and 1 scan iterations | 1 h 27 min | 281.1 |
| `water_multidet` | CONVERGED after 2 scan iterations; last stage: 8 of 8 DMC runs done | 2 h 0 min | 48.4 for the search, 88.3 for the last stage |

Results (1 standard error; kcal/mol unless stated):

- `water`: DMC energy at the final geometry minus that at the starting geometry -0.06 +/- 0.22; the geometry
  moved by 0.0089 angstrom (RMSD after rigid alignment).
- `ammonia_inversion`: DMC barrier 5.27 +/- 0.22 at the B3LYP geometries and 4.98 +/- 0.22 at the optimized
  geometries (change -0.29 +/- 0.31).
- `hcn_isomerization`: DMC barrier from HCN 50.36 +/- 0.19 at the B3LYP geometries and 50.96 +/- 0.20 at the
  optimized geometries; DMC barrier from HNC 34.66 +/- 0.20 and 35.15 +/- 0.22.
- `water_multidet` (input of version 0.1.0: `n_det_max = 10000`, `cutoff = 1e-3, hf`): Quantum Package made
  CIPSI wave functions of 10,590 and 12,737 determinants at the start and the final geometry; the cutoff 1e-3
  kept 50 and 42 of them. All 4 checks of the trial wave functions passed (VMC energy without the Jastrow
  factor equal to the energy that Quantum Package printed). The DMC energies with these wave functions were
  1.53 +/- 0.31 and 1.93 +/- 0.28 millihartree above those with the single RHF determinant of Quantum Package
  at the same geometries ([docs/multireference.md](../docs/multireference.md), section 9).

The DMC barriers at the B3LYP geometries of the ammonia and HCN paths equal those of
[docs/validation.md](../docs/validation.md) (sections 3 and 4) to the last printed digit. The input is the
same, and the random seeds come from the point names (docs/validation.md, section 1), so this run is not an
independent repeat of those runs at the starting geometries.

### Second run, 2026-11-26

A fresh clone of version 0.1.0 in a new virtual environment of Python 3.13.5. After `pip install -e .` all 551
unit tests passed (11 minutes; PySCF and h5py were not installed yet), and the demonstrations of water and
ammonia took 12 s and 41 s on a cluster node. Then `pip install pyscf h5py`. After this second install the
environment held numpy 2.5.5, scipy 1.18.1, matplotlib 3.11.2, pillow 13.0.0, PySCF 2.15.1 and h5py 3.16.0,
and the 4 examples below ran with it.

| example | status of every target | wall time | core-hours |
|---|---|---|---|
| `water` | CONVERGED after 2 scan iterations | 43 min | 52.1 |
| `ammonia_inversion` | reactant and saddle CONVERGED, 2 scan iterations each | 32 min | 26.3 |
| `hcn_isomerization` | HCN, saddle and HNC CONVERGED after 1, 2 and 1 scan iterations | 50 min | 121.3 |
| `water_multidet` | CONVERGED after 2 scan iterations; last stage: 8 of 8 DMC runs done | 48 min | 38.1 for the search, 35.1 for the last stage |

Results (1 standard error; kcal/mol unless stated):

- `water`: DMC energy at the final geometry minus that at the starting geometry -0.09 +/- 0.24; the geometry
  moved by 0.0034 angstrom (RMSD after rigid alignment).
- `ammonia_inversion`: DMC barrier 5.26 +/- 0.19 at the B3LYP geometries and 5.49 +/- 0.21 at the optimized
  geometries (change +0.22 +/- 0.28). The pilot runs chose the time step 0.02 Ha^-1, as on 2026-11-25.
- `hcn_isomerization`: DMC barrier from HCN 50.64 +/- 0.22 at the B3LYP geometries and 50.80 +/- 0.21 at the
  optimized geometries; DMC barrier from HNC 34.71 +/- 0.21 and 35.28 +/- 0.19. Here the pilot runs chose the
  time step 0.01 Ha^-1, and on 2026-11-25 0.005, so the two runs also differ by their time-step error.
- `water_multidet` (input of version 0.1.0: `n_det_max = 10000`, `cutoff = 1e-3, hf`): CIPSI wave functions of
  19,442 and 12,106 determinants; the cutoff 1e-3 kept 55 and 42 of them. All 4 checks of the trial wave
  functions passed. The DMC energies with these wave functions were 2.09 +/- 0.38 and 2.21 +/- 0.42
  millihartree above those with the single RHF determinant at the same geometries: the rise of the first run
  was seen again.

On both days the random seeds come from the point names. Which DMC blocks the two runs share was not checked.

### Third run, 2026-11-27: the automatic choice of the cutoff

The code of version 0.2.0 in a new virtual environment of Python 3.13.5 (numpy 2.5.5, scipy 1.18.1, PySCF 2.15.1,
h5py 3.16.0); all unit tests passed before the runs. Two inputs ran: `water_multidet` as it is in this folder
(`cutoff = auto`) except for the profile name, and `ammonia_inversion` with the profile name changed and three
lines added at its end (`[trial]`, `kind = multidet`, `cutoff = auto`). The default ladder of that day was
`2.5e-4, 1e-4`. The last stage chose its cutoffs and its number of DMC series itself
([docs/multireference.md](../docs/multireference.md), section 10).

| example | status of every target | wall time | core-hours |
|---|---|---|---|
| `water_multidet` | CONVERGED after 2 scan iterations; last stage: `CONVERGED` after 12 DMC runs | 1 h 18 min | 52.5 for the search, 136.0 for the last stage |
| `ammonia_inversion` with `cutoff = auto` | reactant and saddle CONVERGED, 2 scan iterations each; last stage: `NOT_DECIDED` after 48 DMC runs | 1 h 19 min | 28.8 for the search, 192.7 for the last stage |
| the same project, continued with `auto_cutoffs = 2.5e-4, 1e-4, 5e-5` | last stage: `CONVERGED` after 16 more DMC runs | 25 min | 115.1 for the 16 runs |

Results (1 standard error; kcal/mol unless stated):

- `water_multidet`, search: DMC energy at the final geometry minus that at the starting geometry 0.15 +/- 0.22;
  the geometry moved by 0.0038 angstrom (RMSD after rigid alignment).
- `water_multidet`, last stage: the cutoffs 2.5e-4 and 1e-4 kept 1,006 and 2,356 determinants at the start
  geometry and 1,048 and 2,434 at the final geometry. All 6 checks of the trial wave functions passed. No cut
  was above the single RHF determinant (-0.60 +/- 0.30 to -3.74 +/- 0.29 millihartree). The energy change of
  the search was -0.26 +/- 0.17 at the cutoff 2.5e-4 and 0.02 +/- 0.17 at the cutoff 1e-4: a change of 0.28 +/-
  0.24 between the two cutoffs, so the verdict was `CONVERGED` within 1.0 kcal/mol after the first 2 DMC series.
- `ammonia_inversion`, search: DMC barrier 5.26 +/- 0.19 at the B3LYP geometries and 4.97 +/- 0.19 at the
  optimized geometries (change -0.30 +/- 0.27). The pilot runs chose the time step 0.02 Ha^-1. The DMC barrier at
  the B3LYP geometries equals that of the second run to the last printed digit: the random seeds come from the
  point names, so the runs at the starting geometries are not independent repeats.
- `ammonia_inversion`, last stage with the ladder 2.5e-4, 1e-4: no cut was above the single RHF determinant. Two
  of the 4 tested differences were `NOT_DECIDED`. The first was the barrier at the start geometries: 4.98 +/-
  0.13 at the cutoff 2.5e-4 and 5.85 +/- 0.12 at the cutoff 1e-4, a change of 0.87 +/- 0.18, bound 1.23. The
  second was the energy change of the reactant: -0.28 +/- 0.12 and 0.45 +/- 0.12, a change of 0.73 +/- 0.17,
  bound 1.07. For both, the bound is above the tolerance of 1.0 kcal/mol, and the change minus 2 standard errors
  is below it. The stage added a third and a fourth DMC series and then ended with `NOT_DECIDED`, because no
  further series was allowed and the ladder held no further cutoff.
- `ammonia_inversion`, continued with a third cutoff: the stage kept its 48 runs and added the cutoff 5e-5 with
  4 DMC series. Between the cutoffs 1e-4 and 5e-5 all 4 tested differences changed by 0.11 or less, with errors
  of 0.16 to 0.17, so the verdict was `CONVERGED`. Barrier at the cutoff 5e-5: 5.92 +/- 0.11 at the start
  geometries and 5.56 +/- 0.12 at the final geometries; with the single RHF determinant of Quantum Package:
  4.89 +/- 0.14 and 4.62 +/- 0.12. Because of this run the default ladder is now `2.5e-4, 1e-4, 5e-5`: with
  it, the stage adds the cutoff 5e-5 by itself in this case (the rule of the final code, applied to the
  recorded numbers).

[docs/multireference.md](../docs/multireference.md), section 9, gives the energies against `hf`, the tested
differences and the cost of the last stage of these runs. The runs used the code of version 0.2.0 before its last
corrections of 2026-11-27; with the recorded numbers the rule of the final code makes the same decisions.

The tests of earlier versions of the package on these and other molecules are in
[docs/validation.md](../docs/validation.md).
