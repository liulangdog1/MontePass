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
python -m montepass demo water demo_water          # about 3 seconds on a laptop
python -m montepass demo ammonia demo_ammonia      # about 9 seconds
```

## On a cluster

The examples need ORCA, PySCF and QMCPACK (and Quantum Package 2 for `water_multidet`) and a machine
profile (`machines/example_slurm.ini`, [docs/user_guide.md](../docs/user_guide.md), section 2). Copy an example
folder, set `profile` in `[machine]` to the name of your profile, then:

```bash
python -m montepass check  water      # reads the input and the profile; lists every problem
python -m montepass submit water      # one small driver job runs the whole workflow
python -m montepass status water      # stages, targets and the last lines of the log
```

## Our runs of these inputs

We ran the four inputs unchanged (only the profile name differs) with version 0.1.0 of the package on
2026-11-25, on a SLURM cluster with 32 cores per QMC job, with ORCA 6.1.0, PySCF, QMCPACK 4.1.9 and Quantum
Package 2.
The wall time includes the time that the jobs waited in the queue. The core-hours count the DMC jobs only (the
column `core-hours: total` of `results/summary.md`); the ORCA, PySCF and driver jobs are not included. DMC is
stochastic, so a run of the same input on another machine gives other numbers within the error bars.

| example | status of every target | wall time | core-hours of the DMC jobs |
|---|---|---|---|
| `water` | CONVERGED after 2 scan iterations | 25 min | 43.2 |
| `ammonia_inversion` | reactant and saddle CONVERGED, 2 scan iterations each | 17 min | 22.7 |
| `hcn_isomerization` | HCN, saddle and HNC CONVERGED after 1, 3 and 1 scan iterations | 1 h 27 min | 281.1 |
| `water_multidet` | CONVERGED after 2 scan iterations; last stage: 8 of 8 DMC runs done | 2 h 0 min | 48.4 for the search, 88.3 for the last stage |

Results (1 standard error; kcal/mol unless stated):

- `water`: DMC energy at the final geometry minus that at the starting geometry -0.06 +/- 0.22; the geometry
  moved by 0.0089 angstrom (RMSD after rigid alignment).
- `ammonia_inversion`: barrier 5.27 +/- 0.22 at the B3LYP geometries and 4.98 +/- 0.22 at the optimized
  geometries (change -0.29 +/- 0.31).
- `hcn_isomerization`: barrier from HCN 50.36 +/- 0.19 at the B3LYP geometries and 50.96 +/- 0.20 at the
  optimized geometries; barrier from HNC 34.66 +/- 0.20 and 35.15 +/- 0.22.
- `water_multidet`: Quantum Package made CIPSI wave functions of 10,590 and 12,737 determinants at the start and
  the final geometry; the cutoff 1e-3 kept 50 and 42 of them. All 4 checks of the trial wave functions passed (VMC
  energy without the Jastrow factor equal to the energy that Quantum Package printed). The DMC energies with these
  wave functions were 1.53 +/- 0.31 and 1.93 +/- 0.28 millihartree above those with the single RHF determinant of
  Quantum Package at the same geometries ([docs/multireference.md](../docs/multireference.md), section 9).

The tests of earlier versions of the package on these and other molecules are in
[docs/validation.md](../docs/validation.md).
