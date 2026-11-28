# Validation: tests on small molecules and on model surfaces

This file lists the tests of the package beyond the unit tests: runs of the whole workflow with real
programs on small molecules, and the comparison of the search methods on model surfaces. The file
gives the results of the runs and the defects that the runs found.

The unit tests (`tests/`) need no cluster program: `python -m unittest discover -s tests`.

All runs on real molecules in this file, except the example runs of version 0.1.0 and the runs of
version 0.2.0, were made on 2026-11-22 and 2026-11-23 on a SLURM cluster with ORCA 6.1.0, PySCF
2.1.1, QMCPACK 4.1.9 (legacy drivers), Python 3.9.21, numpy 1.26.4 and matplotlib 3.7.1. A QMC job
used 32 cores. The example runs of version 0.1.0 were made on 2026-11-25 and 2026-11-26. Limit 6 of
section 6 and section 10 cite them, and [examples/README.md](../examples/README.md) gives their
program versions. The runs of the automatic choice of the cutoff were made on 2026-11-27 with the
code of version 0.2.0 before its last corrections of that day, with the ladder 2.5e-4, 1e-4 (the
default of that day), Python 3.13.5, numpy 2.5.5 and PySCF 2.15.1. Section 10 names them, and
[multireference.md](multireference.md), section 9, gives their numbers.

**Which version of the code was tested.** The package changed during these tests, because the
tests found defects (sections 8 and 9). "Final package" below means the version of the code that
was fixed on 2026-11-22 at 16:37 and was not changed while the runs of the acceptance test
(section 6) were made. The table of section 1 says which runs used the final package. The code in
this repository is newer: it adds path images as search targets and the stage with a
multi-determinant trial wave function ([multireference.md](multireference.md)). The modules of the
search methods (`src/montepass/search/`) differ from the final package only in comments.

Every run of the workflow started from an input file and geometry files only. One run stopped with
the status `NEEDS_ATTENTION` (defect 10 below) and was continued with the command `retry`. No other
calculation of these runs was started or repaired by hand. Two results of section 6 do not come
from a run of the workflow: the plain DMC slopes of limit 2 were computed by hand from the block
energies of one run, and the follow-up test of limit 6 was made with a separate script that ran DMC
series with the QMC engine of the package.

## 1. Overview

The energy column is the DMC energy of the final geometry minus that of the starting geometry,
each from two full-length runs. A negative number means that the search lowered the energy.

**Runs with the final package** (the `classic` row: see the note below the table)

| system | start | method and DMC settings | result | DMC energy, final minus start (kcal/mol) | search blocks | core-hours | section |
|---|---|---|---|---|---|---|---|
| water, ammonia, HCN, formaldehyde, ethylene: 17 runs | every atom moved at random; root mean square displacement per atom 0, 0.05, 0.10 or 0.20 angstrom | `dense-v2`; settings chosen by the pilot runs | 17 of 17 `CONVERGED`, 1 to 7 scan iterations | -395.90 to +0.41 | 49,698 | 1,725 | 6 |
| water (3 starts) and ammonia (2 starts) | every atom moved at random; root mean square displacement per atom 0.10 or 0.20 angstrom | `dense-v2`; time step 0.005 Ha^-1 fixed | 5 of 5 `CONVERGED`, 3 to 7 scan iterations | -395.97 to -11.05 | 17,933 | 788 | 5 |
| the same 5 starts | the same | `classic`; the same DMC settings | all 5 ended after the 4 fixed iterations; 2 of the 5 did not reach the minimum | -165.45 to -10.92 | 42,032 | 1,452 | 5 |
| ammonia inversion: reactant, saddle | B3LYP geometries of the path | `dense-v2`; pilot runs (time step 0.02 Ha^-1) | `CONVERGED` (2 iterations), `CONVERGED` (2) | +0.32 +/- 0.22, -0.11 +/- 0.22 | 922, 747 | 11.8, 10.8 | 3 |
| HCN to HNC: HCN, saddle, HNC | B3LYP geometries of the path | `dense-v2`; pilot runs (time step 0.005 Ha^-1) | `CONVERGED` (1 iteration), `CONVERGED` (3), `CONVERGED` (1) | +0.45 +/- 0.22, +0.48 +/- 0.21, +0.11 +/- 0.20 | 362, 5,036, 707 | 45.9, 235.2, 58.8 | 4 |

The 5 `classic` runs started on 2026-11-22 between 15:35 and 15:42, before the final package was made
at 16:37. Each run used the copy of the package that existed when it started. In that copy the module
of the classic search (`search/classic.py`) was the same as in the final package, and the other
differences of that copy did not change these runs.

**Runs with earlier versions of the package** (2026-11-22, before the final package). The rules
changed between these runs; sections 8 and 9 list the changes. The time is the start of the run.

| system | version | result | section |
|---|---|---|---|
| water, B3LYP geometry | first rules of `dense-v2` (10:02) | `CONVERGED`, 1 iteration, -0.19 +/- 0.27 kcal/mol | 2 |
| water, B3LYP geometry | second rules (10:45) | `CONVERGED`, 1 iteration, -0.07 +/- 0.24 | 2 |
| water, B3LYP geometry | two later versions | both `CONVERGED`, 1 iteration, -0.43 +/- 0.25 and +0.42 +/- 0.25 | 2 |
| ammonia inversion | first rules (10:10) | reactant and saddle `CONVERGED`, 1 iteration each; barrier change -0.47 +/- 0.32 | 3 |
| ammonia inversion | second rules (11:00), rules of 13:32 | both runs: reactant `CONVERGED` (2 iterations), saddle `CONVERGED` (1); barrier changes +0.10 +/- 0.26 and +0.28 +/- 0.31 | 3 |
| HCN to HNC | second rules (11:13) | HCN and HNC `CONVERGED`, saddle `NOT_CONVERGED` | 4 |
| HCN to HNC | rules of 13:39 | all 3 `CONVERGED` (the saddle after 2 iterations); barrier changes +0.24 +/- 0.29 and +0.06 +/- 0.29 | 4 |
| water, start moved by 0.10 angstrom along the symmetric stretch | `classic` (10:45), `dense-v2` and `auto` (10:59) | all 3 recover the 10 kcal/mol of the start | 5 |
| water, start moved by 0.10 angstrom along the symmetric stretch | `dense-v1` (10:45) | `NOT_CONVERGED` after 10 iterations (remaining-energy bound of protocol v1: 0.051 kcal/mol); final minus start -9.92 +/- 0.25 kcal/mol, from final DMC runs that had not all ended when the report was written | - |
| water, start moved by 0.25 angstrom along the symmetric stretch | `classic` (13:47) | ended after its fixed number of 4 iterations; audit `UNRESOLVED` in all 3 directions; final minus start -43.89 +/- 0.16 | - |
| water, start moved by 0.25 angstrom along the symmetric stretch | `dense-v2` (13:46) | `NOT_CONVERGED` after 5 iterations (joint bound 0.023 against the target 0.02); final minus start -43.68 +/- 0.21 | - |
| water, start moved by 0.25 angstrom along the symmetric stretch | `dense-v1` (13:47) | `NOT_CONVERGED` after 10 iterations (remaining-energy bound of protocol v1: 6.537 kcal/mol); the centre of iteration 3 was 40.56 +/- 0.21 kcal/mol below the start, and in each of the 7 later iterations the centre energy rose, so the search moved away from the minimum; final minus start -18.59 +/- 0.21, from final DMC runs that had not all ended when the report was written | - |

The last 4 rows ended on 2026-11-23. Their numbers come from the file `results/summary.md` of each run.
Runs that are not in the tables include runs that were stopped before they finished and were started
again (after the repair of a defect, after the code review, or in another batch queue; for the
systems of this file their second runs are in the tables), the HCN tests of limit 6 in section 6,
the example runs of version 0.1.0 ([examples/README.md](../examples/README.md)), and other runs
that this file does not report (section 10).

**Random numbers of repeated runs.** The random seeds of a QMC run are computed from the name of
its point (for example `geometry/it1/centre`). The input file of the tested version has no option
that changes them. Two runs of the same input therefore use the same random numbers at every
point that has the same name and the same geometry: the DMC blocks of such a point are equal in
both runs, digit for digit. The runs differ only where another version of the rules read another
number of blocks or took another step. The shared blocks were checked in the block files: the 4 water
runs with the time step 0.005 Ha^-1 (section 2), 3 of the 4 ammonia runs (section 3) and the last 2
HCN runs (section 4) share the blocks of all points of their first scan iteration. Such runs show
that the rules are stable. They are not independent repeats, and their results must not be
averaged as if they were.

## 2. Water: is the B3LYP geometry the DMC minimum?

Input: `examples/water` (one geometry file; ORCA B3LYP/cc-pVTZ optimization; the 3 internal
coordinates are searched; `eps_total = 0.02` kcal/mol). The run was made twice: once with the first
set of rules of protocol v2 and once with the second set of rules (both are earlier versions of the
package; see the overview table). The two runs are not independent: they used the same random
numbers at the starting geometry and at the 5 scan points (see "Random numbers of repeated runs"
in section 1). They differ in the number of blocks that each version read (58 and 78 per scan
point), in the step, and in the full-length runs at the final geometry.

| | first rules | second rules |
|---|---|---|
| time step chosen by the pilot runs | 0.005 Ha^-1 | 0.005 Ha^-1 |
| status, iterations | `CONVERGED`, 1 | `CONVERGED`, 1 |
| blocks per scan point at the decision | 58 | 78 |
| O-H bond, B3LYP start (angstrom) | 0.96145 | 0.96145 |
| O-H bond, final (angstrom) | 0.95490 +/- 0.00177 | 0.95467 +/- 0.00216 |
| H-O-H angle, start and final (degrees) | 104.52, 104.46 +/- 0.47 | 104.52, 104.54 +/- 0.41 |
| DMC energy, final minus start (kcal/mol) | -0.19 +/- 0.27 | -0.07 +/- 0.24 |
| DMC blocks: search, all | 313, 737 | 406, 822 |
| core-hours: pilot, search, all | 2.1, 9.9, 21.2 | 2.0, 12.1, 22.5 |
| wall time from the input files to the report | 18 minutes | not recorded (the jobs waited in the queue) |

Reading. Both runs shorten the O-H bond by 0.007 angstrom, which is 3.7 and 3.1 standard errors.
The two scans read the same DMC blocks, so the second run does not confirm the first one
independently. The energy that this step gains is below the error bar of the two full-length runs
(0.24 to 0.27 kcal/mol), so the report says "no significant change" for the energy. The angle does
not move. For comparison, a fit to rotation-vibration spectra gives an equilibrium O-H distance of
0.9578 angstrom and an angle of 104.54 degrees (P. Jensen and co-workers, "A refined potential
energy surface for the electronic ground state of the water molecule", 1994; the numbers were taken
from the abstract). The B3LYP bond is 0.00365 angstrom longer than that value, and the two DMC bonds
are 0.00290 and 0.00313 angstrom shorter, which is 1.6 and 1.4 of their standard errors. The
antisymmetric stretch has a slope of zero by symmetry and got 1 scan point in place of 2, so a scan
of water has 5 points and the centre.

The pilot runs chose the time step 0.005 Ha^-1: at 0.01 the predicted time-step error of an energy
difference is 0.275 kcal/mol, above the limit of 0.20 (rule R4 of
[automatic_settings.md](automatic_settings.md)).

**Later runs of the same input.** Two runs with later versions of the package gave -0.43 +/- 0.25
and +0.42 +/- 0.25 kcal/mol (in the second one the pilot runs chose the time step 0.0039 Ha^-1). The
reference run of the acceptance test, with the final package and a tolerance of 0.2 kcal/mol, gave
-0.25 +/- 0.23 (section 6). All ended as `CONVERGED` after 1 scan iteration. Only the run with the
time step 0.0039 has random numbers of its own. The other 4 runs of this section share the DMC
blocks of the starting geometry, so their 4 energy changes are 4 final energies minus one start
energy. The 4 final energies agree with each other (chi-square 0.9 on 3 degrees of freedom), and
their weighted mean is 0.20 +/- 0.19 kcal/mol below the start energy. This section therefore holds 2
independent comparisons: -0.20 +/- 0.19 and +0.42 +/- 0.25 kcal/mol. They differ by 2.0 of their
combined standard error, and their weighted mean is +0.02 +/- 0.15: for water the search does not
change the DMC energy within about 0.15 kcal/mol.

## 3. Ammonia inversion: a path, a minimum and a saddle

Input: `examples/ammonia_inversion` (two end-point geometries). ORCA optimized both ends, made a
path of 9 images with the nudged elastic band method and optimized the saddle. The principal
component analysis of the path kept 2 components (explained variance 0.9703 and 0.0297). Targets:
the reactant and the saddle. The tables show the run with the first set of rules; the run with the
final package is at the end of this section.

| | reactant (minimum) | saddle |
|---|---|---|
| surrogate force constants along the 2 search directions (kcal/mol/A^2) | 244.4, 1338.3 | -181.3, 1121.3 |
| status, iterations | `CONVERGED`, 1 | `CONVERGED`, 1 |
| blocks per scan point at the decision | 41 | 55 |
| N-H bond, start and final (angstrom) | 1.01386, 1.01230 +/- 0.00143 | 0.99677, 0.99150 +/- 0.00131 |
| H-N-H angle, start and final (degrees) | 106.51, 106.77 +/- 0.18 | 120.00, 120.00 |
| DMC energy, final minus start (kcal/mol) | +0.18 +/- 0.24 | -0.29 +/- 0.21 |
| DMC blocks: search, all | 380, 832 | 182, 634 |
| core-hours: search, all | 7.3, 10.7 | 2.1, 5.2 |

| Inversion barrier (kcal/mol) | |
|---|---|
| B3LYP/cc-pVTZ, along the path | 5.28 |
| DMC at the B3LYP geometries | 5.27 +/- 0.22 |
| DMC at the optimized geometries | 4.80 +/- 0.23 |
| change | -0.47 +/- 0.32 |

Reading. At the saddle the N-H bond shortens by 0.0053 angstrom (4.0 standard errors). The barrier
change of -0.47 +/- 0.32 kcal/mol is 1.5 standard errors, so it is not significant. The slope along
the reaction coordinate at the saddle is zero by symmetry (the planar geometry has a mirror plane
that reverses this direction). The search used 1 scan point along it, for the curvature, in place
of 2: the saddle needed 3 scan points and the reactant 4.

The pilot runs chose the time step 0.02 Ha^-1 (acceptance 0.990; the energy does not depend on the
time step within the errors: slope -7 +/- 29 kcal/mol per Ha^-1). A DMC run of ammonia is then 4
times cheaper than with the conventional 0.005. Wall time from the input files to the report: 19
minutes.

**Run with the final package** (same input; the pilot runs chose 0.02 Ha^-1 again). Reactant and
saddle ended as `CONVERGED` after 2 scan iterations each, with 922 and 747 search blocks. DMC
energy, final minus start: +0.32 +/- 0.22 and -0.11 +/- 0.22 kcal/mol. Barrier: 5.27 +/- 0.22
kcal/mol at the B3LYP geometries and 4.83 +/- 0.22 at the optimized geometries, a change of -0.43
+/- 0.31.

This input was run 4 times in all (first rules, two later versions, final package). The barrier
changes are -0.47 +/- 0.32, +0.10 +/- 0.26, +0.28 +/- 0.31 and -0.43 +/- 0.31 kcal/mol. Three of
the 4 runs (all but the second one) are not independent: they have the same DMC blocks at both
starting geometries and at all scan points of the first iteration (see "Random numbers of repeated
runs" in section 1). They share the barrier at the B3LYP geometries, 5.27 +/- 0.22 kcal/mol, and
differ in their final geometries. Their 3 final barriers are 4.80 +/- 0.23, 5.54 +/- 0.22 and
4.83 +/- 0.22 kcal/mol. The weighted mean is 5.07 +/- 0.13, and the chi-square about it is 7.2 on 2
degrees of freedom, so the 3 final barriers scatter more than their error bars say. Counted
correctly there are 2 independent barrier changes: -0.20 +/- 0.25 (the mean of the 3 final barriers
minus the shared start barrier) and +0.10 +/- 0.26 (the second run). Their weighted mean is
-0.06 +/- 0.18 kcal/mol: for this reaction the geometry search does not change the barrier within
about 0.2 kcal/mol.

## 4. HCN to HNC: linear end points

Input: `examples/hcn_isomerization` (reactant, product and a saddle guess; `eps_total = 0.05`
kcal/mol). ORCA optimized the two linear end points and made a path of 11 images. The principal
component analysis kept 3 components (explained variance 0.5278, 0.3582, 0.0812). Targets: HCN,
the saddle and HNC.

| | HCN | saddle | HNC |
|---|---|---|---|
| status, iterations | `CONVERGED`, 1 | `NOT_CONVERGED`, 1 | `CONVERGED`, 1 |
| DMC energy, final minus start (kcal/mol) | -0.14 +/- 0.21 | -0.19 +/- 0.20 | +0.07 +/- 0.20 |
| C-N bond, start and final (angstrom) | 1.1465, 1.1501 +/- 0.0021 | 1.1809, 1.1824 +/- 0.0020 | 1.1669, 1.1685 +/- 0.0023 |
| bond to H, start and final (angstrom) | C-H 1.0655, 1.0664 +/- 0.0025 | C-H 1.1894, 1.1780 +/- 0.0104 | N-H 0.9983, 0.9948 +/- 0.0026 |
| DMC blocks: search, all | 365, 1,049 | 1,026, 1,710 | 681, 1,365 |
| core-hours: search, all | 21.7, 56.0 | 55.7, 89.9 | 38.1, 73.7 |

| Barrier (kcal/mol) | from HCN | from HNC |
|---|---|---|
| B3LYP/cc-pVTZ, along the path | 47.53 | 33.30 |
| DMC at the B3LYP geometries | 50.71 +/- 0.20 | 34.96 +/- 0.19 |
| DMC at the optimized geometries | 50.66 +/- 0.21 | 34.70 +/- 0.21 |
| change | -0.04 +/- 0.29 | -0.26 +/- 0.28 |

Reading. No geometry changes by more than 2 standard errors, and the barriers do not change. DMC
puts both barriers above the B3LYP values, by 3.2 and 1.7 kcal/mol; that difference comes from the
energy method and not from the geometry.

The saddle ended as `NOT_CONVERGED`: after 171 blocks per point (the limit that the pilot runs had
set for a final run) the joint bound was 0.077 kcal/mol, above the tolerance of 0.05. Along the
reaction coordinate a saddle is scanned over a short span, so its slope needs more blocks than a
minimum. This run led to the rule that repeats a scan with more blocks (defect 13 below).

**Run with the final package** (same input; time step 0.005 Ha^-1 chosen by the pilot runs). All 3
targets ended as `CONVERGED`. The saddle needed 3 scan iterations, because the rule of defect 13
repeated its scan with more blocks.

| | HCN | saddle | HNC |
|---|---|---|---|
| status, iterations | `CONVERGED`, 1 | `CONVERGED`, 3 | `CONVERGED`, 1 |
| DMC energy, final minus start (kcal/mol) | +0.45 +/- 0.22 | +0.48 +/- 0.21 | +0.11 +/- 0.20 |
| DMC blocks: search, all | 362, 1,042 | 5,036, 5,716 | 707, 1,387 |
| core-hours: search, all | 17.8, 45.9 | 207.5, 235.2 | 30.6, 58.8 |

| Barrier (kcal/mol) | from HCN | from HNC |
|---|---|---|
| DMC at the B3LYP geometries | 50.36 +/- 0.19 | 34.66 +/- 0.20 |
| DMC at the optimized geometries | 50.39 +/- 0.23 | 35.03 +/- 0.21 |
| change | +0.03 +/- 0.30 | +0.37 +/- 0.29 |

Reading. The barriers do not change within their errors. For HCN the report of this run prints a
warning: the optimized geometry is 0.45 +/- 0.22 kcal/mol higher than the starting geometry, so the
search did not lower the energy, although the bound of the search was 0.038 kcal/mol. Limit 6 of
section 6 puts this number next to the other runs that started at a B3LYP geometry.

## 5. A wrong starting geometry on purpose

Two experiments start the search from a geometry that is wrong on purpose. In both the surrogate is
moved with the start (`move_surrogate = yes`): the surrogate calls the wrong start its own minimum,
so only the DMC energies know where the minimum is.

**Water moved by 0.10 angstrom along one search direction** (earlier versions of the package; these
runs found defects 4 and 10 of section 8). Input: `examples/water` with `[experiment] shift = 0.10,
direction = 1` and `eps_total = 0.02` kcal/mol. Direction 1 is the symmetric stretch: both O-H bonds
start 0.093 angstrom too long, about 10 kcal/mol above the minimum.

| method | status, scan iterations | time step chosen by the pilot runs (Ha^-1) | DMC energy, final minus start (kcal/mol) | search blocks | core-hours: search, all |
|---|---|---|---|---|---|
| `classic` | ended after the 4 fixed iterations | 0.005 | -10.02 +/- 0.15 | 6,080 | 162.8, 200.4 |
| `dense-v2` | `CONVERGED`, 3 | 0.0019 | -10.13 +/- 0.21 | 1,160 | 84.2, 145.8 |
| `auto` (one classic iteration, then `dense-v2`) | `CONVERGED`, 3 | 0.0016 | -9.89 +/- 0.18 | 1,547 | 156.9, 235.6 |

Reading. All three methods recover the 10 kcal/mol. The dense scan needs 5.2 times fewer search
blocks than the classic search. Its core-hours for the search are 1.9 times lower (1.4 times for
the whole run), because its pilot runs chose a shorter time step, which makes a block more
expensive. The classic run is the one that stopped with `NEEDS_ATTENTION` (defect 10) and was
continued with `retry`. The first runs of `dense-v2` and `auto` were stopped when defect 4 was
found; the rows show the runs after the repair.

**Every atom moved at random: the classic line search against the dense scan** (the dense scan with
the final package; the classic search with an earlier copy of the package whose module
`search/classic.py` is the same as in the final package: see the note below the table of section 1).
Input of a run: the geometry file, `random_rms`, `random_seed`, the method, and fixed DMC settings
for both methods (`[qmc] mode = fixed`, time step 0.005 Ha^-1, 2,048 walkers, `cap_blocks = 400`,
`eps_total = 0.1` kcal/mol). The two methods start from the same geometry. The columns "above the reference geometry" are the B3LYP energy of a geometry minus that
of the final geometry of the reference run of the molecule (section 6); this energy has no
statistical noise.

| molecule, displacement (seed) | method | status | scan iterations | start above the reference geometry (kcal/mol) | final above the reference geometry (kcal/mol) | DMC energy, final minus start (kcal/mol) | search blocks | core-hours, all |
|---|---|---|---|---|---|---|---|---|
| water, 0.10 A (seed 1) | `classic` | ended | 4 | 9.87 | -0.002 | -10.92 +/- 0.17 | 6,080 | 250 |
| water, 0.10 A (seed 1) | `dense-v2` | `CONVERGED` | 3 | 9.87 | +0.018 | -11.05 +/- 0.13 | 1,213 | 76 |
| water, 0.20 A (seed 1) | `classic` | ended | 4 | 33.27 | -0.052 | -35.46 +/- 0.15 | 6,080 | 205 |
| water, 0.20 A (seed 1) | `dense-v2` | `CONVERGED` | 4 | 33.27 | -0.047 | -35.59 +/- 0.14 | 1,754 | 98 |
| water, 0.20 A (seed 2) | `classic` | ended | 4 | 147.75 | **+4.388** | -144.37 +/- 0.13 | 6,080 | 209 |
| water, 0.20 A (seed 2) | `dense-v2` | `CONVERGED` | 5 | 147.75 | +0.018 | -148.10 +/- 0.13 | 2,586 | 116 |
| ammonia, 0.10 A (seed 1) | `classic` | ended | 4 | 55.20 | -0.024 | -55.37 +/- 0.12 | 11,840 | 396 |
| ammonia, 0.10 A (seed 1) | `dense-v2` | `CONVERGED` | 5 | 55.20 | -0.018 | -55.22 +/- 0.11 | 5,157 | 213 |
| ammonia, 0.20 A (seed 1) | `classic` | ended | 4 | 394.59 | **+249.117** | -165.45 +/- 0.15 | 11,952 | 392 |
| ammonia, 0.20 A (seed 1) | `dense-v2` | `CONVERGED` | 7 | 394.59 | +0.006 | -395.97 +/- 0.12 | 7,223 | 284 |

DMC energy of the final geometry, classic minus dense scan (kcal/mol), in the order of the table:
+0.17 +/- 0.15, +0.09 +/- 0.14, **+3.71 +/- 0.13**, -0.14 +/- 0.12, **+230.53 +/- 0.14**.

Reading.

1. From the 3 starts that are 9.9 to 55 kcal/mol above the reference geometry, both methods end
   within 0.06 kcal/mol (B3LYP energy) of the reference geometry, and their final DMC energies agree
   within 1.3 standard errors (ratios 1.11, 0.68 and 1.24, from the unrounded energies). The dense
   scan needs 2.3 to 5.0 times fewer search blocks and 1.9 to 3.3 times fewer core-hours.
2. From the 2 starts at 148 and 395 kcal/mol the classic search does not reach the minimum: it ends
   4.4 and 249 kcal/mol above the reference geometry, and its final DMC energy is 3.71 +/- 0.13 and
   230.53 +/- 0.14 kcal/mol above that of the dense scan. The dense scan ends within 0.02 kcal/mol
   of the reference geometry in both cases.
3. What happened in the classic run of ammonia at 0.20 angstrom: 13 of its 24 fitted steps had no
   minimum inside the scanned window. Along one direction the 7 DMC energies of the first iteration
   go from +2,174 to -301 kcal/mol over a window of +/- 0.40 angstrom; the cubic fit has no minimum
   there and the step goes to the end of the window. After the second iteration the geometry was
   5,145 kcal/mol above the reference geometry. The classic search accepts every step and has no
   convergence test. Its audit marked directions `UNRESOLVED` in all 5 classic runs, also in the 3
   runs that reached the minimum, so the audit does not tell these cases apart.
4. What the dense scan did from the same start: the first scan measured a curvature with the wrong
   sign along 3 of the 6 directions. The step went downhill to the edge of the scanned span along
   every direction (0.065 to 0.202 angstrom), and the scan was repeated at the new centre. After 7
   scan iterations the bound was below the tolerance.
5. Limits. One random seed per start, so 5 pairs. The surrogate has the right shape and the wrong
   place. Along a single direction, with a surrogate of the right shape, the classic search is as
   good as the dense scan or cheaper ([protocols.md](protocols.md), section 5, statement 4).

## 6. Acceptance test: does the search repair a wrong starting geometry by itself?

The rules of this section were written on 2026-11-22 at 15:50, before the runs were started.

**One run.** One molecule and one input file with four entries: the geometry file, the size of the
random displacement (`random_rms`), its seed (`random_seed`), and the machine profile. Every other
value is a default of the package: the search directions, the scan distances, the time step, the
walkers, the number of blocks, the method (`dense-v2`), the tolerance (0.2 kcal/mol) and the stopping
rule. ORCA optimizes the geometry (B3LYP/cc-pVTZ). Then every atom is moved at random, and the
surrogate is moved with the start (`move_surrogate = yes`): the surrogate calls the wrong geometry
its own minimum, so only the DMC energies know where the minimum is.

**Systems.** Water, ammonia, HCN (linear), formaldehyde and ethylene (12 internal coordinates).
Displacements of 0.05 and 0.10 angstrom per atom (root mean square) for each; 0.20 angstrom also for
water and ammonia. One reference run per molecule with no displacement.

**A run passes if all of these hold.**

| Rule | Condition |
|---|---|
| P1, autonomous | The project ends as `DONE` and the target as `CONVERGED`. The only command was `submit`; no file was edited and no job was started by hand. |
| P2, geometry | `d_run <= d_ref + 0.6 kcal/mol`. `d` is the B3LYP energy of the final geometry above the B3LYP minimum (the energy of the trial-orbital calculation of the final DMC point; it has no statistical noise). `d_ref` is the same number for the reference run of the molecule. 0.6 is 3 times the tolerance: two geometries that are each within the tolerance of the DMC minimum can differ by that much on this scale. |
| P3, DMC energy | The DMC energy of the final geometry equals that of the reference run within 2 combined standard errors plus the tolerance (0.2 kcal/mol). This rule is used only if the pilot runs of the two projects chose the same time step; the table says where they did not. |

Also reported, not a pass rule: the share of the starting error that was removed, `1 - d_run /
d_start`, and the DMC blocks.

**The claim.** The algorithm counts as successful on this test only if every run passes. If a run
fails, the cause is found, the code is changed, and all runs of the test are made again; the first
set is then reported as failed.

**Result (2026-11-22).** The 17 runs used the final package (see the start of this file). The last
run ended at 22:48.

All 17 runs ended as `DONE` with the target `CONVERGED`. The only command was `submit`, and no driver
log holds a `retry` line. All 12 displaced runs pass the three rules.

| molecule | displacement (A per atom) | scan iterations | time step of the pilot runs | start above the reference geometry (B3LYP, kcal/mol) | final above the reference geometry (P2: at most 0.6) | DMC energy, final minus reference final (P3, kcal/mol) | distance to the reference geometry, start and final (A per atom) | search blocks | core-hours |
|---|---|---|---|---|---|---|---|---|---|
| water | none | 1 | 0.005 | -0.06 | 0 (reference) | reference | reference | 306 | 21 |
| water | 0.05 | 2 | 0.01 | 2.61 | +0.04 | not used (time step 0.01, reference 0.005) | 0.055 -> 0.003 | 978 | 27 |
| water | 0.10 | 2 | 0.005 | 9.87 | -0.01 | -0.05 +/- 0.22 | 0.105 -> 0.001 | 944 | 46 |
| water | 0.20 | 3 | 0.01 | 33.27 | -0.02 | not used (time step 0.01, reference 0.005) | 0.205 -> 0.003 | 1481 | 36 |
| ammonia | none | 1 | 0.02 | -0.03 | 0 (reference) | reference | reference | 517 | 9 |
| ammonia | 0.05 | 3 | 0.01 | 10.71 | -0.01 | not used (time step 0.01, reference 0.02) | 0.049 -> 0.004 | 2901 | 61 |
| ammonia | 0.10 | 4 | 0.01 | 55.20 | -0.02 | not used (time step 0.01, reference 0.02) | 0.099 -> 0.003 | 4069 | 82 |
| ammonia | 0.20 | 7 | 0.01 | 394.59 | -0.02 | not used (time step 0.01, reference 0.02) | 0.199 -> 0.003 | 7853 | 154 |
| HCN | none | 1 | 0.01 | -0.08 | 0 (reference) | reference | reference | 624 | 30 |
| HCN | 0.05 | 2 | 0.0036 | 0.80 | +0.05 | not used (time step 0.0036, reference 0.01) | 0.049 -> 0.017 | 828 | 94 |
| HCN | 0.10 | 7 | 0.01 | 3.53 | -0.04 | -0.33 +/- 0.19 | 0.099 -> 0.001 | 4402 | 131 |
| formaldehyde | none | 1 | 0.005 | -0.03 | 0 (reference) | reference | reference | 542 | 81 |
| formaldehyde | 0.05 | 2 | 0.01 | 3.66 | -0.02 | not used (time step 0.01, reference 0.005) | 0.050 -> 0.004 | 2219 | 87 |
| formaldehyde | 0.10 | 3 | 0.005 | 15.22 | -0.01 | -0.25 +/- 0.19 | 0.099 -> 0.004 | 2618 | 188 |
| ethylene | none | 1 | 0.01 | -0.04 | 0 (reference) | reference | reference | 1080 | 55 |
| ethylene | 0.05 | 3 | 0.01 | 6.01 | -0.02 | -0.26 +/- 0.22 | 0.052 -> 0.006 | 7831 | 269 |
| ethylene | 0.10 | 4 | 0.01 | 23.44 | -0.02 | -0.38 +/- 0.22 | 0.102 -> 0.004 | 10505 | 353 |

The distance column is extra information and not a pass rule: the root mean square distance per
atom between a geometry and the final geometry of the reference run, after the best rigid fit (a
mirror image is allowed). The 17 runs used 49,698 search blocks and 1,725 core-hours in total.

**What limits this result.**

1. **Rule P3 covers 5 of the 12 displaced runs.** In the other 7 the pilot runs chose another time
   step than in the reference run, so the two DMC energies cannot be compared. The rule was written
   that way before the runs.
2. **The run of HCN with a displacement of 0.10 angstrom needed 7 scan iterations because of a flaw
   of the estimator.** The search reached the minimum in iteration 3. There the plain DMC slopes along the
   two bending directions were +2.6 +/- 0.7 and -0.9 +/- 0.9 kcal/mol/angstrom (computed by hand
   from the block energies of the four scan points), but the estimator gave +9.0 and -10.8, a
   predicted gain of 0.45 kcal/mol, and the search stepped away. The cause: in this test the
   surrogate is moved with the start, so the surrogate has its linear geometry at the bent starting
   geometry. At the true (linear) minimum the moved surrogate is 0.17 angstrom away from its own
   linear geometry, and its curve along a bend is strongly asymmetric over the scan (cubic part 6.7
   and -9.2 kcal/mol/angstrom), while the DMC curve is symmetric there. Subtracting the surrogate
   added that asymmetry to the slope. The search then stepped back and forth (0.071, 0.058 and
   0.045 angstrom) while the rule that shortens the spans reduced the error, and it ended in
   iteration 7. At that centre the surrogate was 5.2 kcal/mol above its own stationary point, just
   above the limit of 5.0 above which the surrogate is not subtracted, so the end came partly by
   chance. A surrogate that is not moved has its symmetric geometries at the same places as the DMC
   surface, so this failure is expected only in this kind of test; that expectation was not tested.
   The estimator should not trust the cubic part of the surrogate where the surrogate is far from
   its own stationary point; this is not repaired in the tested version.
3. **At a geometry that is already good, the search does not help and can cost up to the
   tolerance.** In the 5 reference runs the DMC energy of the final geometry minus that of the
   starting geometry (the B3LYP minimum) is -0.25 +/- 0.23, +0.21 +/- 0.23, +0.41 +/- 0.19, +0.25
   +/- 0.20, +0.18 +/- 0.23 kcal/mol (water, ammonia, HCN, formaldehyde, ethylene); the weighted
   mean is +0.18 +/- 0.10. The last step of a search is not checked by a scan, and its expected
   energy error is the noise energy `N`. The noise energy is inside the tolerance of 0.2 kcal/mol.
   The same cost in the reference runs explains why all 5 values of the P3 column are negative
   (weighted mean -0.26 +/- 0.09; they share 4 reference energies).
4. **The test moves the surrogate with the start.** The test therefore checks a surrogate minimum at
   the wrong place with the right shape. A surrogate with the wrong shape was tested on model
   surfaces (section 7 and [protocols.md](protocols.md), section 5); this file reports no test
   with real DMC in which the surrogate was given a wrong shape on purpose.
5. **Queue handling.** During the test a helper script moved waiting jobs between two batch queues
   of the cluster. The script edited no file of a run and submitted no calculation.
6. **With a tighter tolerance the cost at a good geometry is larger than the tolerance, and HCN
   shows it most.** The two path runs with the final package (sections 3 and 4) start at B3LYP
   geometries with tolerances of 0.02 and 0.05 kcal/mol. The DMC energy of the final geometry minus
   that of the start is +0.32 +/- 0.22 (ammonia), +0.45 +/- 0.22 (HCN) and +0.11 +/- 0.20 (HNC)
   kcal/mol for the 3 minima; their mean is +0.29 +/- 0.12, which is 2.0 standard errors above the
   larger tolerance. For HCN the two runs with the final package agree: +0.41 +/- 0.19 (reference
   run of this section) and +0.45 +/- 0.22 (section 4), together +0.43 +/- 0.14 kcal/mol, 3.0
   standard errors above zero, while the bounds of the two searches were 0.037 and 0.038 kcal/mol.
   Runs of HCN with earlier versions gave -0.14 +/- 0.21 and +0.12 +/- 0.19. Two later tests of HCN
   (2026-11-23) did not settle whether the search makes this good geometry slightly worse:
   - 8 repeat runs from the B3LYP minimum of HCN, with the DMC settings that the pilot runs of
     section 4 chose (time step 0.005 Ha^-1, held fixed) and 8 different random seeds. All 8 ended as
     `CONVERGED` after 1 scan iteration, and in all 8 the final geometry was higher in DMC energy
     than the start: weighted mean +0.26 +/- 0.07 kcal/mol. These runs shifted the random seeds with
     an input option that no released version (0.1.0 to 0.2.1) has.
   - A follow-up test ran 96 new DMC series with new random seeds. A separate script, not the
     workflow, ran them with the QMC engine of the package. 64 of the series were at the start
     geometry and at the 8 final geometries of these repeat runs. The other 32 were at two
     geometries near the start: one moved by 0.006 angstrom along a bend direction, and one moved by
     0.0025 angstrom along a stretch direction. Final geometries minus start: -0.02 +/- 0.05
     kcal/mol. The rise was not seen again.

   The cause of the rise in the earlier runs was not found. The run of `examples/hcn_isomerization`
   with version 0.1.0 on 2026-11-25 ([examples/README.md](../examples/README.md)) gave +0.02 +/- 0.20
   kcal/mol for HCN (bound 0.035). That run is not an independent repeat at the starting geometries:
   it has the input of section 4, and the random seeds come from the point names, so its DMC
   energies at the 3 starting geometries equal those of the run of section 4 to the last printed
   digit. A second run of the example on 2026-11-26, in which the pilot runs chose the time step
   0.01 Ha^-1, gave +0.08 +/- 0.21 kcal/mol for HCN (bound 0.020).

## 7. Tests on model surfaces

On a model surface the true minimum is known, so the error of a final geometry can be measured.
[protocols.md](protocols.md), section 5, holds the tables; this section gives the main numbers.
`examples/model_study/compare_protocols.py` runs all search methods on analytic models of water (3
search directions) and ammonia (6 search directions). A "true" surface stands for DMC, with
Gaussian noise of 5 kcal/mol per block, and a second surface stands for the surrogate. Every case
has 16 runs with different noise. These are model surfaces and not QMC runs.

- **A start that is wrong in a random direction** (tables D and E of protocols.md). The classic
  search ends within 0.2 kcal/mol in every run up to a displacement of 0.10 angstrom per atom for
  water and 0.05 angstrom for ammonia; beyond that it fails in a growing part of the runs. Protocol
  v2 ends within 0.2 kcal/mol in all 16 runs up to 0.20 angstrom for water and up to 0.10 angstrom
  for ammonia, and in 14 or 15 of 16 runs for the larger displacements.
- **A surrogate with the wrong shape** (table C). With a turned surrogate surface the classic search
  ended outside the tolerance in all 16 runs, for both molecules. Protocol v2 ended within the
  tolerance in 16, 16, 14 and 15 of 16 runs of the four turned cases.
- **Cost at the stationary point of the surrogate** (table A). For ammonia protocol v2 needs 493
  search blocks, the classic search with lean block limits 6,474 and the default classic search
  14,082.
- **Where the classic search is as good or better** (table B). For water with a start that is wrong
  along one direction, the classic search with lean block limits needs fewer blocks than protocol v2
  for an equal error in most rows.
- **Is the `CONVERGED` statement true?** `examples/model_study/calibrate_v2.py` runs 96 random seeds
  for each of 22 cases. At the tolerance 0.2 kcal/mol 2,102 of 2,112 runs ended as `CONVERGED`, and
  2,082 of these were within the tolerance (99.0 %). At the tolerance 0.05 kcal/mol 2,091 of 2,112
  runs ended as `CONVERGED`, and 2,000 of these were within the tolerance (95.6 %). The statement
  holds at the level of 95 % in every case in which the surrogate has the right couplings (18 cases,
  lowest value 97 %). The statement does not hold in every case with a turned surrogate: 92 % and 94 % at the
  tolerance 0.2, and 68 % to 82 % at the tolerance 0.05 (three of the four turned cases).

## 8. Defects that these runs found

Each of these passed the unit tests that existed at the time. All are fixed, and each fix has a unit
test.

| # | Found by | Defect | Fix |
|---|---|---|---|
| 1 | first check run on water | The check "VMC without Jastrow factor equals the determinant energy of PySCF" failed by 0.039 hartree. The input held a Jastrow element with all coefficients zero, and QMCPACK still applies the electron-electron cusp of such an element. | The check input holds no Jastrow element. The check then passes within 1.3 standard errors. |
| 2 | a check job on a node shared with other jobs | With a plain `--bind-to core` Open MPI refused to start, because its default placement put 2 processes on 1 core. With `overload-allowed` the same run was 7 times slower. | The launch line `--map-by core --bind-to core` in the machine profile. |
| 3 | water, single geometry | The Hessian of the ORCA optimization was not used again, because ORCA writes it in a frame shifted to the centre of mass. One more frequency job ran. | The Hessian is put into the frame of the geometry (shift and rotation). |
| 4 | water, shifted start | While a DMC point was still running, the search received the surrogate energy of the true geometry instead of the moved surrogate. The slope signal cancelled, and the first step was 0.0045 angstrom where 0.07 or more was right. | The wrapper for the moved surrogate now replaces that energy in every state of a point. |
| 5 | HCN | An optimized linear molecule is off its axis by about 1e-7 angstrom. The code counted a sixth rigid motion, which is really a bend, and the Hessian reader refused the file. | A molecule counts as linear within 1e-3 of its size. |
| 6 | model study made for the shifted start | Protocol v2 could report `CONVERGED` up to 0.5 kcal/mol above the minimum for a start far away (2 of 8 model runs). Three causes: slope information carried between centres, a rank-one correction of the model Hessian, and the use of the surrogate far from its own stationary point. | See [protocols.md](protocols.md), section 3. With the final rules 752 of 765 converged model runs of that study are within the tolerance. |
| 7 | pilot runs with a slow start-up phase (unit test left unfinished) | Start-up blocks that are discarded anyway were counted as outlier blocks, and the pilot stopped. | The outlier test uses the blocks that are kept. |
| 8 | unit tests run on the cluster | Two tests depended on the environment (a batch job number, parser diagnostics of an older matplotlib). | The tests accept both environments; the diagnostics are not recorded as figure warnings. |
| 9 | 5 pilot runs of water, ammonia and HCN | The chosen time step was not the same from run to run: 0.0016 to 0.005 Ha^-1 for water. Check M6 ("the acceptance falls during the run") used a fixed limit of 0.001. In a pilot run with 512 walkers at time step 0.02 the acceptance of one block scatters by 0.0005, so 4 to 33 of 1,620 blocks were below the limit. The run at 0.02 was then excluded, the fit of the energy against the time step lost its largest time step, and the error of the slope grew from about 30 to about 140 kcal/mol per Ha^-1. | M6 measures the scatter of the block acceptance of the run itself, and it also looks for a fall that lasts 10 blocks. On the 15 stored traces of these pilots the new check reports nothing, and the slope from 3 time steps gives 0.005 for water (3 pilots) and for HCN, and 0.01 for ammonia. |
| 10 | water, classic search, the full-length run at the start | One QMC job ran 10 times slower than the others (29.5 against 2.9 to 3.0 seconds per block for the same molecule), on a node that it shared with jobs of other users. Of the first 338 QMC jobs of these tests that ended, it was the only one. The job reached its time limit after 242 of 400 blocks, and the project stopped with `NEEDS_ATTENTION`. | The engine compares the time per block of every running job with that of the other points. A run that is more than 3 times slower is cancelled and submitted again once, away from its nodes. A run that reaches its time limit because it was slow also gets one more attempt. |
| 11 | the same failure | A project with the status `NEEDS_ATTENTION` could not be continued without editing its state files. | The command `retry` gives every failed calculation a new set of attempts and clears the status. The command was used on this project: the failed run was submitted again and the search went on. |
| 12 | a run with 28 QMC points on a queue that ran 7 at a time | The decision of protocol v2 needs data from every direction. The 7 points that ran first went on to the block limit of 400 while 21 points waited. | A running point is stopped when it holds 3 times the usable blocks that the precision target is expected to need (measured from the noise of the points). In this run the first points were stopped after 215 and 221 blocks. |
| 13 | HCN, the saddle | The search ended as `NOT_CONVERGED` because the block limit of a scan point was the length of a final run (171 blocks), and the saddle needed more. | A scan that does not reach the precision target is repeated at the new centre with more blocks per point. The number follows from the noise energy that was measured and is at most 4 times the limit. With fixed QMC settings the limit of the user is kept. |

## 9. Defects that a code review found

Before the acceptance test of section 6, the search code of protocol v2 was reviewed with four
questions: the estimator, the driver, a start far from the minimum, and the fairness of the
comparison with the classic search. Each finding was then checked again by an attempt to refute it.
All 18 findings were confirmed. None of them had been found by a run or by a unit test. The 13 findings in the search code are fixed and have unit tests; the 5 findings about
the comparison changed `examples/model_study/compare_protocols.py` and the numbers of
[protocols.md](protocols.md), section 5.

| # | Part | Defect | Fix |
|---|---|---|---|
| R1 | stopping rule | The factor of the joint 95 % bound was that of all `n` directions. At a symmetric centre, or with a reaction coordinate, fewer directions carry the noise energy, and the right factor is larger (3.00 in place of 2.61 for water). | The factor follows the effective number of directions that carry the noise energy. The block plan of a saddle search uses the same number. |
| R2 | estimator | Where the surrogate is not subtracted, the size of the shape error was taken from the surrogate at a centre where its shape is known to be wrong (up to 2.8 times too large or 3 times too small). The slope itself was not corrected. | The cubic part that the surrogate had at its own stationary point is added to the slope, with an uncertainty of 50 %. |
| R3 | estimator | At such a centre a measured force constant that is mostly noise was used as it was. | The force constant of the surrogate at its own stationary point, with an error of 100 %, is the value that is assumed before the scan. |
| R4 | estimator | An error bar of exactly zero (a model without noise) stopped the search with a division by zero. | A run with an error of zero counts as exact. |
| R5 | driver | A failed surrogate calculation at a centre made the search wait with no end. | The search ends as `NEEDS_ATTENTION` with the message of the job, and `retry` submits the job again. |
| R6 | driver | A decision was saved only at the end of a pass, after its jobs were submitted. If the process ended in between, the next process decided again with more blocks, got another centre, and read the jobs of the first decision as its own. | A decision is saved before a job is submitted. A point that exists with another geometry is an error. The final run is asked for on every pass. |
| R7 | driver | A failed run at a centre was not noticed while the search ran. | The search ends as `NEEDS_ATTENTION`. A failed point of a moved surrogate (section 5) no longer shows the surrogate energy of the geometry that was not moved. |
| R8 | driver | A rejected step at the iteration limit ended the search without full-length runs at the final geometry. | The search ends at the centre that the step started from, with full-length runs there. |
| R9 | driver | With a shape part between 82 % and 100 % of the tolerance, neither "shorter spans" nor "more blocks" was chosen. | Shorter spans are chosen when the shape part is at least half of the tolerance and more blocks cannot help. |
| R10 | driver | With a block limit of 50 to 60, a finished point could have no usable data, and the search could not continue. | A run that has ended gets a smaller discard. |
| R11 | far start | The span of a scan had no upper limit relative to the first scan. A force constant that was measured far too low gave a scan 3.5 times longer than normal and a step that crossed the minimum. | A span is at most 1.5 times the span of the first scan. |
| R12 | far start | The factor that shortens a span was computed for the span of the present scan and applied to the span of the next one, which comes from a new force constant. | The span that the rule asks for is stored and applied. |
| R13 | far start | A weak curvature measurement replaced the remembered force constant whenever it fluctuated upward. Such a replacement makes the bound too small and can end the search too early. | A measurement replaces the remembered value only if it is at least as precise, or smaller in size. |
| R14 | comparison | Every point of the classic search cost 80 blocks because of two lower limits (50 discarded, 30 usable), not because of its own noise rule. The block counts of the classic search in the first table were therefore not a property of the method. | A second variant `classic-lean` (10 discarded, at least 8 usable) is run next to the default. |
| R15 | comparison | Reliability was compared only by a count of runs within 0.2 kcal/mol. In all water rows the classic search ended closer to the minimum than protocol v2. | The tables give the mean error of every method, and protocol v2 is also run at a tolerance of 0.05 kcal/mol. |
| R16 | comparison | The shifted starts moved one direction that has no cubic term, and the "random" start was one fixed vector. | Rows for the symmetric stretch with both signs, and an experiment in which every run has its own random displacement of all atoms. |
| R17 | comparison | For a saddle the error was a signed energy, so a fall along the reaction coordinate could cancel a rise in the other directions. | For a saddle the error is the harmonic energy of the displacement, with absolute force constants. |
| R18 | comparison | The search blocks of protocol v1 were counted with another rule than those of the other methods (too low by 18 % or more). | One rule for all methods. |

The review also showed that `surrogate/pyscf_jobs.py`, which runs the gradient and Hessian jobs of the
surrogate, had no unit test. The module now has 9: the job script is run with a model in place of PySCF
(exact Hessian of a quadratic function, continuation after a stop), and the handling of the batch
jobs is tested with a recorded batch system.

## 10. What was not tested, and what this file does not report

- Open-shell molecules, charged molecules, elements beyond the second row, and other pseudopotentials than ccECP.
- A batch system other than SLURM. The class `LocalScheduler` runs scripts on one computer and was used only in unit tests.
- The batched QMCPACK drivers. The inputs use the legacy drivers.
- ORCA 5: the input writers and readers have unit tests with ORCA 5.0.3 files, but every complete run used ORCA 6.1.0.
- A complete search with protocol v2 on a molecule larger than 6 atoms: not reported in this file.
- `[qmc] mode = iterative` on a real molecule.
- A path whose search needs more than 3 coordinates: not reported in this file.
- The automatic repeat of a slow run: it has unit tests, and no slow run appeared after it was added.
- The command `retry` for a failed ORCA or PySCF job: it was used once, for a failed QMC point.
- The two additions to the code after the final package, path images as search targets and the
  multi-determinant trial wave function ([multireference.md](multireference.md)), were not part of
  these runs. Their unit tests are in `tests/test_workflow_model.py`, `tests/test_multidet.py` and
  `tests/test_trial_stage.py`. The multi-determinant stage also ran with real programs (Quantum
  Package 2 and QMCPACK) in the example `examples/water_multidet` with version 0.1.0, on 2026-11-25 and
  on 2026-11-26 ([examples/README.md](../examples/README.md)). The input of that example was then
  `n_det_max = 10000` and `cutoff = 1e-3, hf`.
- The automatic choice of the cutoff (`[trial] cutoff = auto`, added in version 0.2.0;
  [multireference.md](multireference.md), section 10) for a molecule larger than ammonia. It ran
  with real programs on 2026-11-27, with the ladder 2.5e-4, 1e-4 (the default of that day), for
  water (`CONVERGED` after 12 DMC runs) and for the ammonia inversion (`NOT_DECIDED`; `CONVERGED`
  after a third cutoff was added to the input). The default ladder is now 2.5e-4, 1e-4, 5e-5.
  [multireference.md](multireference.md), section 9, gives the numbers. Its unit
  tests are in `tests/test_trial_auto.py`: they use a stand-in engine whose energies the tests set,
  and one test gives the plan to the real engine with a fake batch system.
