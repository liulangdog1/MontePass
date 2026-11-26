# Examples

Each folder holds the starting geometries (`*.xyz`) and a commented `input.ini`. The geometries are rough
guesses: the package optimizes them with ORCA first and then searches on the DMC surface.

| folder | what it shows | starting geometries |
|---|---|---|
| `water/` | one molecule, no path: the search directions are the softest Hessian modes | `geometry.xyz` |
| `ammonia_inversion/` | a reaction path: the reactant minimum and the inversion saddle | `reactant.xyz`, `product.xyz`, `saddle_guess.xyz` |
| `hcn_isomerization/` | a reaction path with three targets: HCN, the saddle and HNC | `reactant.xyz`, `product.xyz`, `saddle_guess.xyz` |
| `water_multidet/` | the water search plus the optional, experimental last stage: DMC energies with a trial wave function of several determinants from Quantum Package 2 | `geometry.xyz` |
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
- `water_multidet`: Quantum Package made CIPSI wave functions of 10,590 and 12,737 determinants at the start and
  the final geometry; the cutoff 1e-3 kept 50 and 42 of them. All 4 checks of the trial wave functions passed (VMC
  energy without the Jastrow factor equal to the energy that Quantum Package printed). The DMC energies with these
  wave functions were 1.53 +/- 0.31 and 1.93 +/- 0.28 millihartree above those with the single RHF determinant of
  Quantum Package at the same geometries ([docs/multireference.md](../docs/multireference.md), section 9).

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
- `water_multidet`: CIPSI wave functions of 19,442 and 12,106 determinants; the cutoff 1e-3 kept 55 and 42 of them.
  All 4 checks of the trial wave functions passed. The DMC energies with these wave functions were 2.09 +/- 0.38
  and 2.21 +/- 0.42 millihartree above those with the single RHF determinant at the same geometries: the rise of
  the first run was seen again.

On both days the random seeds come from the point names. Which DMC blocks the two runs share was not checked.

The tests of earlier versions of the package on these and other molecules are in
[docs/validation.md](../docs/validation.md).
