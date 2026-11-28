# DMC energies with several determinants (`[trial] kind = multidet`, experimental)

This page describes the option that computes DMC energies with a multi-determinant trial wave function
from Quantum Package 2 (QP2). The user guide (`docs/user_guide.md`, sections 2, 3 and 6) gives the
input keys and the result files in short form. This page gives every step, every check and every
message, and the limits of this version.

## 1. What the option does, and what it does not do

The geometry search always uses one Slater determinant with Kohn-Sham orbitals from PySCF. The option
does not change the search. It adds one stage, `trial`, between the stages `search` and `report`. The
stage starts when the search of every target has ended. For every target it takes two geometries:
the start geometry and the final geometry of the search. For every geometry and every cutoff it runs
independent DMC runs with a multi-determinant trial wave function. The stage is experimental:
section 9 gives the measured energies and the limits.

The cutoffs come from one of two modes:

- `cutoff = auto` (the automatic mode, section 10): the package chooses the cutoffs from a ladder. It
  tests each cut against the single RHF determinant, tests whether every tested energy difference
  (section 10, step 3) is converged with the size of the expansion, adds DMC series or the next cutoff
  when the test cannot decide or finds a change above the tolerance, and reports a verdict.
- `cutoff = <list>` (the manual mode): you choose the cutoffs and the number of DMC series. The
  report makes the same two tests and gives a verdict, but the package adds no run by itself.

Why the automatic mode exists: a short expansion can give a DMC energy ABOVE that of one determinant.
For water and ammonia this was measured for the cutoffs 1e-3 and 5e-4 (section 9).

The trial wave function of one cutoff is made in this order:

1. QP2 computes RHF orbitals and one CIPSI (selected configuration interaction) wave function with
   about `n_det_max` determinants. All cutoffs of a geometry use this one CIPSI calculation.
2. QP2 removes from a copy of the CIPSI wave function every determinant whose alpha part or beta part
   has a summed weight below the cutoff (QP2 program `truncate_wf_spin`, setting `ci_threshold`).
3. QMCPACK optimizes a Jastrow factor (one-body and two-body terms) for this determinant expansion.
   The coefficients of the determinants stay as QP2 computed them.

The cutoff `hf` is the single RHF determinant of QP2. It needs no CIPSI calculation.

What the report prints: the DMC energy of every geometry and cutoff, the energy change of every
target (final geometry minus start geometry), and every barrier (saddle minus minimum) at the start
geometries, at the final geometries, and its change. Every such difference uses rows of one cutoff
only. The report never subtracts the single-determinant energy of the search from a multi-determinant
energy: the two trial wave functions have different nodes, so their fixed-node errors differ.

The report also prints two tests of the size of the expansion, in both modes (section 10 has the
rule):

- "DMC energy against the single RHF determinant": the DMC energy of every cut minus that of the row
  `hf` of this stage at the same geometry. Both rows use the RHF orbitals of QP2 and the same chain of
  programs. So the lower energy has the better nodes, if the time-step errors of the two rows are
  equal; section 9 says that this was not tested. A cut whose energy is higher by more than 2
  standard errors is "above", and its cutoff is NOT USABLE.
- "Convergence with the size of the expansion": every tested energy difference (section 10, step 3)
  at the two smallest usable cutoffs, its change between them, and a verdict.

In the automatic mode a third section, "Automatic choice of the cutoff", lists every decision of the
stage, the final verdict and the result.

## 2. How to switch it on

### Input section `[trial]`

| Key | Default | Meaning |
|---|---|---|
| `kind` | `single` | `single`: nothing more is computed, and the other keys are not read. `multidet`: the stage `trial` is added. |
| `cutoff` | empty | needed for `multidet`. Either the word `auto` (the automatic mode; the word must stand alone), or a list: one value, or several values separated by commas, for example `2.5e-4, 1e-4, hf`. A number must be between 0 and 1. A smaller number keeps more determinants. The word `hf` is the single RHF determinant of QP2. Two cutoffs that agree in their first 6 significant digits count as the same cutoff, and the input file is refused. |
| `n_det_max` | empty | size of the CIPSI wave function of each geometry: a whole number, at least 2. With a list of cutoffs it is needed, unless every cutoff is `hf`. With `cutoff = auto` an empty value means 100000. `n_det_max = auto` is not built. |
| `auto_cutoffs` | `2.5e-4, 1e-4, 5e-5` | only with `cutoff = auto`: the ladder, which holds the cutoffs that the package may use, in the order of their use. At least 2 numbers between 0 and 1, each smaller than the one before it; no `hf` (the automatic mode always computes `hf`). |
| `tolerance` | 1.0 | kcal/mol, larger than 0. An energy difference counts as converged between two cutoffs when the size of its change plus 2 standard errors is at most this value. Used by the convergence test of the report in both modes, and by the automatic mode. |
| `series` | 2 | independent DMC runs for each geometry and each cutoff (at least 1). With `cutoff = auto`: the number at the start. |
| `series_max` | 4 | only with `cutoff = auto`: the largest number of DMC runs for each geometry and each cutoff. At least `series`. |
| `qp2_basis`, `qp2_pseudo` | `auto` | names of the basis file and of the pseudopotential file of QP2. `auto` takes them from `[qmc] basis` and `ecp`: `ccecpccpvdz`, `ccecpccpvtz`, `ccecpccpvqz`, `ccecpccpv5z` give `cc-pvdz_ecp_ccecp` to `cc-pv5z_ecp_ccecp`; `ccecp` gives `ccecp`. |
| `not_converged` | `stop` | a target whose search ended `NOT_CONVERGED`: `stop` = the stage does not start and the project needs attention; `run` = the stage runs, and the tables mark every value that uses the final geometry of that target. |

The automatic mode with its defaults (the input of `examples/water_multidet`):

```
[trial]
kind = multidet
cutoff = auto
```

The same with every key written out:

```
[trial]
kind = multidet
cutoff = auto
n_det_max = 100000
auto_cutoffs = 2.5e-4, 1e-4, 5e-5
tolerance = 1.0
series = 2
series_max = 4
```

The manual mode:

```
[trial]
kind = multidet
n_det_max = 100000
cutoff = 2.5e-4, 1e-4, hf
series = 2
```

In the manual mode the keys `auto_cutoffs` and `series_max` are not read. A manual list should hold
`hf` and at least two numeric cutoffs: then the report can make both tests. Section 9 says why the
cutoffs 1e-3 and 5e-4 are a bad choice for water and ammonia.

### Machine profile

| Section | Key | Meaning |
|---|---|---|
| `[setup.qp2]` | `lines` | shell lines at the start of every QP2 job. The first line must be `set +u` (the rc file of QP2 reads variables that are not set). Then the modules that QP2 was built with. The last line: `source /path/to/qp2/quantum_package.rc`. |
| `[job.qp2]` | `nodes`, `tasks`, `cpus_per_task`, `mem`, `time` | batch options of a QP2 job. `tasks` must be 1. `cpus_per_task` is the number of threads of QP2. `cpus_per_task`, `mem` and `time` must be set. `time` must be more than 15 minutes, because every step ends 10 minutes before the time limit of its job. |
| `[executables]` | `determinants_tools` | path of `utils/determinants_tools.py` of the QMCPACK source tree. |
| `[executables]` | `qp2_python_bin` | folder of a Python with `h5py`. The export program of QP2 gets it first on its PATH. May be empty if the Python of `[setup.qp2]` has `h5py`. |
| `[job.qmc]` | `time_multidet` | time limit of the QMC job of a multi-determinant point. Default: `time`. |
| `[job.qmc]` | `mem_multidet` | memory of the QMC job of a multi-determinant point. Default: `mem`. |

`montepass check <folder>` lists every key that the profile lacks for QP2 as a `PROBLEM:` line, and
then prints the plan of the stage: the number of geometries, QP2 calculations, cutoffs and DMC runs.
In the automatic mode the plan names the ladder, the tolerance and the two numbers of series, and it
gives the number of DMC runs at the start and at most. The profile must allow `series_max` DMC runs of
one geometry (in the manual mode: `series`): at most 1024 / (2 x `[job.qmc] tasks`), which is 16 with
32 MPI ranks.
`montepass submit` refuses to submit the driver job while such a problem exists. The commands `run`,
`report` and `retry` print the problem and end with exit code 1.

## 3. What runs for one geometry

The QP2 calculations of one geometry live in one folder, `qp2_n<n_det_max>` (`qp2` when every cutoff
is `hf`). The engine keeps their record in `qp2.json` there. Three kinds of jobs run, in this order.

### The parent job (folder `qp2_n<n_det_max>`)

| Step | What runs | Check after the step |
|---|---|---|
| `qpcreate` | `qp_create_ezfio -q -b <qp2_basis> -p <qp2_pseudo> -o rhf.ezfio geom.xyz` | the EZFIO has the convention 20210101 and `basis/ao_normalized` is `T` |
| `qpscf` | `qp set scf_utils thresh_scf 1e-10`; `qp_run scf rhf.ezfio` | the output holds the line `SCF energy`; the energy file exists. The files of `rhf.ezfio` are recorded with their checksums (`rhf.manifest`). |
| `cipsi` | a copy of `rhf.ezfio` becomes `cipsi.ezfio`; `qp set determinants n_det_max <n_det_max>`; `qp_run fci cipsi.ezfio` | the stored wave function is complete (every file of `determinants/` unzips, and its two size fields equal `n_det`). The step is done only if the wave function holds at least `n_det_max` determinants, or if QP2 stopped by its own rule because the second-order energy (PT2) fell below `perturbation/pt2_max`. The files are recorded with their checksums (`cipsi.manifest`). |

A later job continues the CIPSI calculation from the stored wave function (`read_wf` true). If the
stored wave function is already larger than `n_det_max`, only `qp_run diagonalize_h` runs, and the
number of determinants must not change.

A request for the cutoff `hf` needs only the steps `qpcreate` and `qpscf`. The stage asks for the
numeric cutoffs of a geometry first, so that one parent job holds all three steps.

### The cut job (folder `qp2_n<n_det_max>/cut_<label>`)

The label of a cutoff is `c` and the cutoff with 6 significant digits, for example `c1e-03`; the
label of the RHF determinant is `hf`.

| Step | What runs | Check after the step |
|---|---|---|
| `trunc` | the stored parent must still have the files of `cipsi.manifest`; a copy gets `qp set qmcpack ci_threshold <cutoff>`; `qp_run truncate_wf_spin` | the output holds `Number of determinants:`; the cut wave function is complete |
| `qpenergy` | `qp_run print_energy` on a copy of the cut wave function | the output holds `E(CI)` |
| `qpexport` | `qp_run save_for_qmcpack` on a copy (the program changes the EZFIO it runs on), with 1 thread | an h5 file exists; the basis coefficients of the exported copy hold at least one value and none of the factors of the known export error (the square roots of 3, 5 and 15) |

For the cutoff `hf` only the step `qpexport` runs, on a copy of `rhf.ezfio`.

When the cut job has ended, the engine reads its results and checks them (section 4).

### The QMC job (folder of the point)

| Step | What runs |
|---|---|
| `scf` | the PySCF step of the package. For a multi-determinant point it also records the RHF energy of PySCF (`e_rhf` in `system.json`). |
| `mdc4q` | the h5 file of QP2 is copied to `trial.h5`; `determinants_tools.py --transform` converts the determinants; `convert4qmc -orbitals trial.h5 -multidet trial.h5` writes the QMCPACK files; then the check `check-mdc4q` |
| `mdvmc` | VMC with the determinants alone (no Jastrow factor), in one or more runs, until the check `check-mdvmc` passes or fails |
| `opt` | Jastrow optimization; the coefficients of the determinants stay fixed |
| `dmc` | DMC |

A further series (`series` 2, 3, ...) runs only DMC. It reads `trial.h5` and the determinant block of
its series-1 point, and gets random numbers of its own.

## 4. The checks

| Check | Rule | Setting and default |
|---|---|---|
| EZFIO convention | the export program handles only the convention 20210101; any other is refused before the export. Only the test feature `import_ezfio` accepts the newer convention 20250211, when the basis is stored in normalized form (`ao_normalized` and `primitives_normalized` are `T`, and no `prim_normalization_factor` file exists): the copy for the export then gets the number 20210101 | fixed |
| Export fingerprint | the first coefficient of every basis function in the exported copy (file `ao_basis/ao_coef.gz`) lies outside 1.73204 to 1.73207, 2.23605 to 2.23608 and 3.87296 to 3.87300, and the file holds at least one value | fixed |
| Same molecule, basis and pseudopotentials | the RHF energy of QP2 minus the RHF energy of PySCF at the same geometry is at most the limit in size | `rhf_tolerance` = 1e-5 hartree |
| Atoms and electrons | the export has the atoms of `system.json` in the same order, each within 1e-5 bohr of its place, the same valence charges, and the same numbers of up and down electrons | fixed |
| Determinant count | the wave function file holds as many determinants as QP2 printed; for N > 1, every VMC and DMC output says `Found N terms in the MSD expansion` with that N | fixed |
| Energy of the cut | `print_energy` finds the energy that `truncate_wf_spin` printed | `energy_tolerance` = 1e-6 hartree |
| Size of the cut | at least 2 determinants; the CIPSI wave function holds at least 2 times as many | `min_parent_ratio` = 2 |
| Cut and parent | the energy of the cut lies between the energy of the CIPSI wave function and the RHF energy; `truncate_wf_spin` read a wave function of the recorded size | fixed |
| VMC without Jastrow factor | see below | `check_*` settings |

The VMC check compares the VMC energy of the determinants alone with `e_ref`, the energy that QP2
printed for the same wave function. All runs of the check are combined. With the combined difference
D and its standard error s:

- PASS: |D| <= 3 s + 0.1 mHa, and s <= 0.6 mHa, and every run read the right number of determinants;
- FAIL: |D| > 5 s + 0.1 mHa, or a run read another number of determinants, or the last allowed run
  gave no PASS;
- MORE: neither; one more run with new random numbers follows.

The first run is short (60 blocks of 40 steps, 32 walkers per MPI rank): it measures the noise. The
next run is as long as the error bar needs, at most 6000 blocks. When the error bar is small enough but
the difference lies between the two limits, the next run is as long as all runs so far. At most 4 runs
are made, and on many MPI ranks fewer (at most 1024 / (2 x ranks)), so that no two runs share random
numbers. A second attempt of the job keeps the runs that the first attempt finished.

A check that fails stops the point before any DMC block. The verdict and the numbers stay in the
record of the point, and the report shows them.

## 5. Time limits, restarts and lost jobs

- Every step runs under its own time limit, never longer than the time that its job has left minus
  600 s (`margin_seconds`). Step limits in seconds: `qpcreate` 600, `qpexport` 1800, the other steps only
  the job limit (`time_limits`).
- A CIPSI step that reaches the time limit continues in a new job from the stored wave function. At
  most 4 CIPSI jobs of one geometry may reach their time limit (`max_segments`). A job that fails for
  another reason does not count, and `retry` starts the count again. If the wave function did not grow
  in a job that ran for at least a quarter of its time limit, the parent fails at once.
- When the last CIPSI job ended at its time limit after QP2 saved the wave function but before QP2
  printed the summary of that size, the variational energy and the PT2 of the CIPSI wave function are
  not known. The report then shows `not computed` for them. The cut and the DMC energies do not change.
- While QP2 runs, a copy of the last complete wave function (`cipsi.bak.ezfio`) is kept and renewed
  after every larger save. A wave function that the end of a job left incomplete is replaced by that
  copy.
- Any other failure of a QP2 step that a new job can repair gets 2 attempts in total (`max_attempts`
  of the engine). A failure whose reason begins with `QUALITY:` fails at once. A QP2 job that is
  cancelled outside the package (for example with `scancel`) also fails at once; use `retry`. A job that
  the package cancels itself (because no point needs it any more) is no failure: the next point that
  needs the folder gets a new job, which continues from the files.
- Two QP2 jobs never work in the same job folder at the same time (the folder `qp2_n<n_det_max>` itself,
  or one folder `cut_<label>`). The cut jobs of different cutoffs run at the same time. Every QP2 job
  writes the time and its job number into its file `job.heartbeat` once a minute. A new job that finds
  a time in that file less than 3 minutes old, written by another job, waits up to 4 minutes; if the
  other job is still writing then, the new job ends with exit code 98.
- When the batch system gives no answer about a job, the engine counts the job as running while the
  time in its `job.heartbeat` is less than 3 minutes old. Otherwise it waits `unknown_seconds` (600 s),
  cancels the job, and only then submits a new one. A job that the engine finds writing `job.heartbeat`
  in a folder, and that its record does not hold (a submission whose answer was lost), is taken into the
  record.
- The decision about the next job is saved before the job is submitted. A refused submission is made
  again on the next pass and counted once.
- A failure of one point does not stop the other points of the stage. The stage needs attention only
  when no point can make progress any more. A request that the batch system refuses is made again on
  every pass; after 1 hour of refusals it counts as a failure of the point.
- When a value of `[trial]` changes, the points of the new plan are asked for first. Then the points of
  the old plan that are not in the new plan are stopped. So a QP2 job that a point of the new plan
  needs (the CIPSI job of a geometry, for example) is kept. A stopped point is not continued later:
  delete its folder to compute it again.
- The automatic mode writes each decision and the plan that follows from it to its record in one step.
  A driver job that is started again continues from that record: the same plan, and no decision is
  made twice. Section 10 says which changes of the input keep the plan that was reached.

## 6. Threads and cores

QP2 is one process with threads. Only the `qp_run` lines get `cpus_per_task` threads; every other line
of a QP2 job runs with 1 thread, and the export runs with 1 thread. QP2 plans its memory with 90 % of
the memory of the job (`QP_MAXMEM`; set only under SLURM). The QMC job of a point is an MPI job like every other QMC job of
the package: QMCPACK and `convert4qmc` run with `OMP_NUM_THREADS=1`. The PySCF step follows the rule of
every point of the package (the threads of `[job.scf]` when it runs as a job of its own).

## 7. Messages and what to do

The message of a point that failed in its QP2 part begins with `the Quantum Package calculation failed: `,
followed by `step <name>: ` and the reason of the step (for a failed QP2 step), or by
`the cut <label> cannot be used: ` (for a check of the results of the cut job). The reason of a QP2 step
begins with `RUN:` (a new job can repair it), `TIMEOUT:` (the step reached its limit) or `QUALITY:` (the
result is wrong; a new job would give the same result). A check of the QMC job gives `step mdc4q: ` or
`step mdvmc: ` and the reason. The log of a QP2 job is `job_parent.out` or `job_cut.out` in its folder.

To compute the QP2 calculations of a geometry again: wait until no driver job of the project runs,
delete the folder `qp2_n<n_det_max>` (or `qp2`) of that geometry, then run `montepass retry` and
`montepass submit`. The package makes the folder again from the geometry of the point, and computes RHF
and CIPSI again.

| Message contains | Meaning | What to do |
|---|---|---|
| `QUALITY: the EZFIO has the convention` | the QP2 installation writes another EZFIO convention than the export program reads | use a QP2 version whose `qp_create_ezfio` writes 20210101 |
| `QUALITY: basis/ao_normalized of the EZFIO is not T` | the basis was not stored in normalized form | check the basis file of QP2 |
| `QUALITY: the exported basis holds the factors of the known export error` | the h5 file has wrong orbitals | check the QP2 version and the convention |
| `QUALITY: files of the stored CIPSI wave function are missing or changed` | the parent was changed or partly deleted after its job ended (for example by the purge of a scratch file system) | compute the QP2 calculations of the geometry again (see above) |
| `QUALITY: files of rhf.ezfio are missing or changed since the SCF` | the RHF files were changed or partly deleted after the RHF step | compute the QP2 calculations of the geometry again (see above) |
| `step qpcreate did not finish and left no reason` | the job ended before its first step: a line of `[setup.qp2]` failed, or `qp_run` is not on the PATH | read the line `SETUP_FAILED` in `job_parent.out` and repair `[setup.qp2]` |
| `RUN: qp_run fci ended with N determinants (wanted: at least M)` | CIPSI stopped below the wanted size without the PT2 reason (for example a stop request) | the first time, the next job continues. The second time, the point fails: use `retry` |
| `RUN: the CIPSI wave function is not complete after the run` | the stored wave function is damaged | the next job takes the copy `cipsi.bak.ezfio`; if this repeats, compute the QP2 calculations again |
| `RUN: the diagonalization changed the number of determinants` | `diagonalize_h` changed a wave function that was already large enough | look at `cipsi.<n>.log`; compute the QP2 calculations again |
| `RUN: the task server of QP2 did not start` | the communication server of QP2 did not start within 300 s | the engine submits a new job; if this repeats, check the network setup of QP2 on the compute nodes |
| `TIMEOUT: no end within` (QP2 steps `qpscf`, `cipsi`, `trunc`, `qpenergy`) | a step reached the time limit of its job | raise `[job.qp2] time` |
| `TIMEOUT: no end within 600 s` or `1800 s` (steps `qpcreate`, `qpexport`) | these steps have fixed limits of the engine (`time_limits`), which the input file cannot change | look at `create.log` or `export/save.log`; report the case |
| `TIMEOUT: no end within` (steps of the QMC job) or `the job reached its time limit` | a QMC step of a multi-determinant point needs more time | raise `[job.qmc] time_multidet`, then `retry` |
| `the CIPSI wave function did not grow in a whole job` | one CIPSI iteration needs more time than one job has | raise `[job.qp2] time` or lower `n_det_max` |
| `the CIPSI run reached the time limit in N jobs` | the wanted size needs more than `max_segments` jobs | raise `[job.qp2] time`, or lower `n_det_max` |
| `the cutoff ... leaves 1 determinant` | the cutoff removes every determinant but one | use a smaller cutoff, or `hf` |
| `the cutoff ... keeps N of the M determinants` | the CIPSI wave function is less than 2 times larger than the cut | raise `n_det_max` |
| `print_energy finds` | QP2 does not find the energy of the cut wave function again | look at `truncate.log` and `print_energy.log` of the cut |
| `truncate_wf_spin printed N determinants, the stored wave function holds M` | the cut wave function on disk is not the one that QP2 printed | compute the QP2 calculations of the geometry again |
| `the export wrote no file` | `save_for_qmcpack` ended without an h5 file | look at `export/save.log` of the cut; check `qp2_python_bin` (a Python with `h5py`) |
| `the result of the QP2 parent job cannot be used` | the RHF or CIPSI output does not hold what the record needs | look at `scf.log` and `cipsi.<n>.log`; compute the QP2 calculations again |
| `the energy of the cut wave function ... is not between` | the cut does not come from the recorded parent, or the orbitals changed | compute the QP2 calculations of the geometry again |
| `truncate_wf_spin read a wave function of N determinants` | the cut was made from another parent than the recorded one | compute the QP2 calculations of the geometry again |
| `step mdc4q: the RHF energy of QP2 ... differs from the RHF energy of PySCF` | the two programs do not describe the same molecule, basis or pseudopotentials | check `[trial] qp2_basis`, `qp2_pseudo` and `[qmc] basis`, `ecp` |
| `step mdc4q: an atom of the QP2 export is` or `step mdc4q: the atoms of the QP2 export` | the atom order or the geometry of the export differs from the point | this points to an error of the package: keep the folders, and compare `geom.xyz` of the QP2 folder with `system.json` of the point |
| `step mdc4q: the wave function file holds N determinant(s), QP2 printed M` | `convert4qmc` read another number of determinants | check the QMCPACK version and `determinants_tools.py` |
| `step mdvmc: ... the difference is above 5 standard errors` | the VMC energy of the determinants is not the energy of QP2: wrong orbitals, a lost determinant, a wrong atom order or a wrong pseudopotential | look at `check_mdvmc.json` of the point |
| `step mdvmc: ... no pass` after the last run | the difference stayed between the two limits in all allowed runs: it is small but not zero | look at `check_mdvmc.json` of the point (combined difference and error bar). `retry` gives the same verdict, because the finished runs are kept |
| `determinant check: the QMCPACK output of the DMC run does not say` | a DMC run did not read all determinants | check the QMCPACK input of the point |
| `jastrow check: ... the Jastrow factor does not lower the energy` | the optimized VMC energy is not below the energy of the determinants without Jastrow factor. For a multi-determinant point this energy is `e_ref`, the energy that QP2 printed | look at the optimization outputs `opt.s*.scalar.dat` of the point |
| `point ... exists with another trial wave function` | a run of the stage on disk was made with another trial wave function than the one that the stage asks for now. The message gives the stored values and the asked values (cutoff, `n_det_max`, basis file, pseudopotential file). The stage does not use the energy of such a run, also when the run is `DONE`, and it needs attention | set the key of `[trial]` that differs back to the stored value. To keep the new value, compute the stage again from its first job (next row) |
| `point ... exists with another geometry` or `the QP2 folder ... holds another geometry` | a run of the stage on disk, or the QP2 calculations of a geometry, were made at another geometry than the one that the stage asks for now. The start geometry or the final geometry of a target changed after they were made, for example because the search of that target ran again. The stage does not use the energy of such a run, also when the run is `DONE`, and it needs attention. A report that you write with `montepass report` in this state still prints the energy of such a run; its first lines then say that the stage needed attention | compute the stage again from its first job: wait until no driver job of the project runs and no job of the stage runs or waits in the queue, delete the record of the stage (`work/trial/`) and the points of the stage (the folder `trial` below `work/qmc/`), then run `montepass submit`. All QP2 calculations and all DMC runs of the stage are computed again |
| `the search of target(s) ... ended NOT_CONVERGED, so the stage does not start` | the rule `[trial] not_converged = stop` | set `not_converged = run` to compute the stage at that geometry |
| `... asks for [trial] kind = multidet, but the machine profile ... cannot run that stage` | the profile lacks a QP2 section or key, `[job.qp2] time` is 15 minutes or less, or `[trial] series` (with `cutoff = auto`: `[trial] series_max`) is larger than 1024 / (2 x `[job.qmc] tasks`) | repair what the `PROBLEM:` lines name |
| `automatic cutoff: NOT_CONVERGED`, `NOT_DECIDED`, `NO_DIFFERENCE` or `NO_VERDICT` at the end of the message of the stage | the verdict of the automatic mode. The stage is `DONE`: a verdict is a result, not a failure | read section 10 and the section "Automatic choice of the cutoff" of `results/trial_wavefunctions.md`. What to do depends on the verdict. `NOT_CONVERGED`: to continue, add a smaller cutoff at the end of `auto_cutoffs`. `NOT_DECIDED`: the end of the message says what can help. "to continue, raise [trial] series_max above N": the errors are too large for the verdict `CONVERGED` also with a smaller cutoff, so raise `series_max` above N, the number of series of the plan; a smaller cutoff does not help then. "to continue, add a smaller cutoff at the end of [trial] auto_cutoffs": the errors are small enough, so add a smaller cutoff. `NO_DIFFERENCE`: nothing to do; the project prints no energy difference, so the ladder and `series_max` change nothing. `NO_VERDICT`: read the reason in the message; it names every row that is not complete, or every difference that cannot be compared |
| `refused for N minutes` | the batch system refused the request of a point for more than 1 hour (for example a time limit above the limit of the partition) | read the refusal in the message; repair the profile, then `retry` |

## 8. Files

```
work/trial/state.json                   record of the stage: rule, plan, state of every point at the last pass;
                                        with cutoff = auto also the entry "auto": the plan now (cutoffs, series),
                                        every decision with its numbers, and the final verdict
work/qmc/trial/<target>/<start|final>/
    n<n_det_max>_c<cutoff>/             DMC run of one cutoff (series 1); ..._s2/ its second series
    hf/                                 DMC run with the RHF determinant of QP2
    qp2_n<n_det_max>/                   QP2 calculations of this geometry (qp2/ when every cutoff is hf)
        geom.xyz, qp2.json, job_parent.sh, job_parent.out, job.heartbeat, job.info
        rhf.ezfio, rhf.manifest, scf.log, create.log
        cipsi.ezfio, cipsi.bak.ezfio, cipsi.manifest, cipsi.segments, cipsi.<n>.log
        cut_<label>/  job_cut.sh, job_cut.out, job.heartbeat, pre.ezfio, truncate.log, print_energy.log,
                      export/QP2QMCPACK.h5, export/save.log, export/fingerprint, facts.json
work/qmc/qp2_jobs.json                  job table of the QP2 jobs
```

In the folder of a series-1 point: `trial_facts.json` (the facts of the cut), `trial.h5`,
`md_determinantset.xml`, `check_mdc4q.json`, `check_mdvmc.json`, and the VMC runs `mdvmc<r>.*`.

Result files: `results/trial_wavefunctions.md` and `results/trial_wavefunctions.csv` (user guide,
section 6).

## 9. Limits of this version, and what was measured

The stage is experimental. This section gives the limits, the measured energies behind the rule of
section 10, the results of the automatic mode for two molecules, and what is still not known.

### Limits

- The molecule must have charge 0 and multiplicity 1 (closed shell). One electronic state.
- `n_det_max = auto` is not built. `cutoff = auto` chooses the cutoffs from the ladder of the input
  (section 10); it does not choose `n_det_max`.
- The coefficients of the determinants are not optimized with the Jastrow factor. They stay as QP2
  computed them, without a Jastrow factor. The next two parts show what this costs.
- The orbitals are the RHF orbitals of QP2; natural orbitals are not used.
- The search never uses these wave functions. The energies are computed after the geometry is fixed.
- A wave function that exists already can be used only through the engine setting `import_ezfio` (a
  test feature), not through the input file.
- The automatic choice of `qp2_basis` knows only the ccECP cc-pVnZ basis sets.
- The cost of the QP2 jobs and of the DMC runs depends on the molecule and on `n_det_max`. Section 10
  gives the measured cost of two cutoff ladders of small molecules; there the QP2 jobs are a small
  part of the cost. For a larger molecule measure the cost of one geometry before a large run.

### Measured: short expansions give DMC energies above one determinant

Runs of 2026-11-27 with the engine of version 0.1.0, started by scripts and not by the stage. Water
and ammonia, one geometry each. One CIPSI wave function per molecule with `n_det_max = 100000`
(106,988 determinants for water, 199,135 for ammonia), cut at four cutoffs. DMC with 2048 walkers, 200
blocks of 400 steps and the time step 0.005 Ha^-1; one DMC series for each cut and two for `hf`.

DMC energy of the cut minus the DMC energy of the single RHF determinant (`hf`, same orbitals), in
millihartree, with the number of determinants in brackets:

| cutoff | water | ammonia |
|---|---|---|
| 1e-3 | +2.14 +/- 0.38 (73) | +3.15 +/- 0.40 (196) |
| 5e-4 | +2.45 +/- 0.37 (439) | +2.49 +/- 0.39 (508) |
| 2.5e-4 | -0.95 +/- 0.39 (1,006) | -0.78 +/- 0.36 (1,610) |
| 1e-4 | -4.05 +/- 0.34 (2,362) | -2.02 +/- 0.36 (4,105) |

What the table shows:

- For both molecules the cuts at 1e-3 and 5e-4 give DMC energies above the single determinant, by 5.6
  to 7.9 standard errors. These short expansions are worse than one determinant.
- The cuts at 2.5e-4 and 1e-4 give energies below the single determinant.
- The energy does not fall steadily with the number of determinants.
- Total energies are not converged at these cutoffs: between the cutoffs 2.5e-4 and 1e-4 the DMC
  energy fell by 3.10 +/- 0.41 millihartree for water and by 1.24 +/- 0.38 millihartree for ammonia
  (differences of the DMC energies of these runs).

### Measured for water and ammonia: the cause is the coefficients

The coefficients of the determinants come from QP2, which computes them without a Jastrow factor. The
package optimizes only the Jastrow factor. A test by hand outside the package (2026-11-27) took the
same cut wave functions and optimized their coefficients together with the Jastrow factor in QMCPACK
(linear method, starting from the Jastrow factor of the run with fixed coefficients). Then DMC ran
with the settings of the table above and another seed. DMC energy minus that of `hf`, in
millihartree:

| molecule | cutoff | determinants | coefficients of QP2 (what the package does) | optimized: 8 cycles of 100,000 samples | optimized: 6 more cycles of 400,000 samples |
|---|---|---|---|---|---|
| water | 1e-3 | 73 | +2.14 +/- 0.38 | +0.31 +/- 0.33 | -0.20 +/- 0.34 |
| ammonia | 1e-3 | 196 | +3.15 +/- 0.40 | +1.52 +/- 0.38 | -0.02 +/- 0.31 |
| water | 5e-4 | 439 | +2.45 +/- 0.37 | -1.39 +/- 0.35 | not run |
| ammonia | 5e-4 | 508 | +2.49 +/- 0.39 | -0.77 +/- 0.35 | not run |

What the table shows:

- In all 4 cases the first optimization of the coefficients lowered the DMC energy: by 1.63 +/- 0.44
  millihartree (ammonia, 1e-3) to 3.84 +/- 0.41 millihartree (water, 5e-4), which is 3.7 to 9.3
  standard errors.
- With the longer optimization both cuts at 1e-3 give the energy of the single determinant within
  their errors (-0.20 +/- 0.34 and -0.02 +/- 0.31 millihartree). So, within these errors, the
  coefficients of QP2 explain the whole excess of the cuts at 1e-3 for water and ammonia.
- A coefficient optimization needs enough samples. 8 cycles of 100,000 samples were not enough for
  the 196 determinants of ammonia; 6 more cycles of 400,000 samples were.
- Limits of this test: one optimization for each case, and one DMC series for each case.

The package does not optimize the coefficients, and this version does not change that. Instead the
automatic mode starts at the cutoffs 2.5e-4 and 1e-4, tests every cut against the single determinant,
and tests the energy differences for convergence (section 10).

One paper reports the same effect: E. Giner, R. Assaraf, J. Toulouse, Mol. Phys. 114, 910 (2016),
DOI 10.1080/00268976.2016.1149630.

### Measured: a cut without spin error is also above one determinant

QP2 makes the cut by spin strings: it removes every determinant whose alpha part or beta part has a
summed weight below the cutoff (program `truncate_wf_spin`, section 1). Such a cut is not an exact
singlet: for the 8 cuts of the first table the expectation value of the squared total spin, S^2, is
between 0.0150 and 0.0303 (an exact singlet has 0). A second test by hand (2026-11-27) checked whether
this spin error causes the higher energies. It used cuts that are exact singlets by construction: the
QP2 program `truncate_wf` with `s2_eig = true` keeps or removes all determinants of one configuration
(one occupation pattern of the orbitals) together. These cuts were made from the same two CIPSI wave
functions and were read through the engine setting `import_ezfio`. DMC ran with the settings of the
first table, one series for each cut. DMC energy minus that of `hf` (the two `hf` runs of the first
table), in millihartree:

| molecule | threshold of `truncate_wf` | determinants of the exact-singlet cut | DMC minus `hf` |
|---|---|---|---|
| water | 3e-4 | 134 | +4.65 +/- 0.33 |
| water | 1e-4 | 620 | +4.77 +/- 0.32 |
| water | 2e-5 | 1,907 | +1.21 +/- 0.32 |
| ammonia | 3e-4 | 155 | +4.46 +/- 0.36 |
| ammonia | 1e-4 | 576 | +4.08 +/- 0.36 |

What the table shows:

- Every exact-singlet cut that was computed is above the single determinant, by 3.7 to 15.1 standard
  errors, also the cut of water with 1,907 determinants. The cuts by spin strings with 1,006 or more
  determinants (water) and with 1,610 or more determinants (ammonia) are below the single determinant
  (first table).
- A new diagonalization of the Hamiltonian in the space of the smallest cut did not lower the DMC
  energy: +4.82 +/- 0.36 millihartree for water (134 determinants) and +5.56 +/- 0.38 millihartree
  for ammonia (155).
- So a cut without spin error does not remove the higher energies. The spin error of the cut by spin
  strings is not needed to explain them, and the stage keeps this cut. The test does not measure
  which part of the excess of a cut by spin strings comes from its spin error, because the two kinds
  of cut hold different determinants.
- Limits of this test: one DMC series for each cut. The exact-singlet cut of ammonia with the
  threshold 2e-5 (2,381 determinants) gave no DMC energy: the check of the number of determinants
  (section 4) stopped the point, because QMCPACK did not read all 2,381 determinants (2 of them have
  the coefficient 0).

### Measured: the automatic mode on water and on the ammonia inversion

Runs of 2026-11-27 with the code of version 0.2.0, from the xyz files to the report, with real
programs on a cluster. Inputs: `examples/water_multidet` as it is, and `examples/ammonia_inversion`
with the lines `[trial]`, `kind = multidet` and `cutoff = auto` added. All other values of `[trial]`
were the defaults of that day: `n_det_max = 100000`, the ladder 2.5e-4 and 1e-4, the tolerance 1.0
kcal/mol, 2 DMC series at the start and at most 4. (Because of these runs the default ladder is now
2.5e-4, 1e-4, 5e-5; see the end of this part.) The DMC settings came from the pilot runs of each project:
the time step 0.005 Ha^-1 with 155 blocks of 500 steps for water, the time step 0.02 Ha^-1 with 157
blocks of 125 steps for ammonia, 2048 walkers for both.

What the stage did:

- Water: 1 decision after the first 12 DMC runs, `CONVERGED` within 1.0 kcal/mol between the cutoffs
  2.5e-4 and 1e-4.
- Ammonia, ladder 2.5e-4, 1e-4: 3 decisions, `NOT_DECIDED` with 2, with 3 and with 4 series. The stage
  ended with `NOT_DECIDED` after 48 DMC runs, because `series_max = 4` allowed no further series and
  the ladder held no further cutoff.
- Ammonia, continued: we wrote `auto_cutoffs = 2.5e-4, 1e-4, 5e-5` into the input and submitted the
  project again. The stage kept its 48 runs, decided again, added the cutoff 5e-5 with 4 series (16
  DMC runs), and ended `CONVERGED` within 1.0 kcal/mol between the cutoffs 1e-4 and 5e-5.

DMC energy of each cut minus that of `hf` at the same geometry, in millihartree, with the number of
determinants in brackets (water: 2 series; ammonia: 4 series):

| geometry | determinants of the CIPSI wave function | 2.5e-4 | 1e-4 | 5e-5 |
|---|---|---|---|---|
| water, start | 103,253 | -0.60 +/- 0.30 (1,006) | -3.74 +/- 0.29 (2,356) | not run |
| water, final | 191,086 | -0.65 +/- 0.29 (1,048) | -3.35 +/- 0.30 (2,434) | not run |
| ammonia reactant, start | 100,908 | -0.62 +/- 0.19 (2,588) | -3.06 +/- 0.19 (7,094) | -3.38 +/- 0.18 (12,506) |
| ammonia reactant, final | 196,204 | -1.02 +/- 0.20 (1,590) | -2.29 +/- 0.20 (3,992) | -2.74 +/- 0.20 (9,406) |
| ammonia saddle, start | 194,116 | -0.48 +/- 0.24 (1,381) | -1.53 +/- 0.23 (3,525) | -1.75 +/- 0.22 (6,942) |
| ammonia saddle, final | 184,399 | +0.25 +/- 0.20 (1,137) | -0.97 +/- 0.19 (3,016) | -1.24 +/- 0.19 (6,198) |

The energy differences that the rule tests, in kcal/mol ("start" and "final" are the geometries
before and after the search):

| difference | `hf` | 2.5e-4 | 1e-4 | 5e-5 | change from 2.5e-4 to 1e-4 | change from 1e-4 to 5e-5 |
|---|---|---|---|---|---|---|
| water, final minus start | -0.22 +/- 0.19 | -0.26 +/- 0.17 | +0.02 +/- 0.17 | not run | +0.28 +/- 0.24 | not run |
| ammonia reactant, final minus start | -0.03 +/- 0.12 | -0.28 +/- 0.12 | +0.45 +/- 0.12 | +0.37 +/- 0.11 | +0.73 +/- 0.17 | -0.08 +/- 0.16 |
| ammonia saddle, final minus start | -0.31 +/- 0.14 | +0.15 +/- 0.14 | +0.04 +/- 0.13 | +0.01 +/- 0.12 | -0.11 +/- 0.19 | -0.03 +/- 0.17 |
| ammonia barrier at the start geometries | 4.89 +/- 0.14 | 4.98 +/- 0.13 | 5.85 +/- 0.12 | 5.92 +/- 0.11 | +0.87 +/- 0.18 | +0.06 +/- 0.17 |
| ammonia barrier at the final geometries | 4.62 +/- 0.12 | 5.41 +/- 0.13 | 5.45 +/- 0.12 | 5.56 +/- 0.12 | +0.03 +/- 0.18 | +0.11 +/- 0.17 |

What the runs show:

- No cut was above the single determinant: 16 comparisons at 6 geometries, the largest value +0.25
  +/- 0.20 millihartree. The cuts of ammonia were computed at the time step 0.02 Ha^-1, the cuts of
  the first table of this section at 0.005 Ha^-1.
- For ammonia two differences depend on the cutoff between 2.5e-4 and 1e-4, by 0.73 +/- 0.17 and
  0.87 +/- 0.18 kcal/mol. Between 1e-4 and 5e-5 all four differences change by 0.11 kcal/mol or
  less, with errors of 0.16 to 0.17 kcal/mol. So a ladder of two cutoffs was not enough for ammonia,
  and one more cutoff was. For this reason the default ladder is now 2.5e-4, 1e-4, 5e-5. With the
  recorded numbers and this ladder, the rule of this version adds the cutoff 5e-5 by itself after
  the fourth series of ammonia (decision 6 of section 10), with no change of the input.
- The same cutoff can keep different numbers of determinants at two geometries 0.0049 angstrom apart
  (root mean square displacement after rigid alignment; largest displacement of one atom 0.0057
  angstrom): 2,588 and 1,590 determinants for the start and the final geometry of the ammonia
  reactant at the cutoff 2.5e-4. The size of the CIPSI wave function is not the reason: for water,
  wave functions of 103,253 and 191,086 determinants give cuts of 1,006 and 1,048 determinants. The
  reason was not investigated.
- Cost of the stage, in the core-hours that the engine counted (QP2 jobs included): 136.0 for water
  (12 DMC runs); 192.7 for the rows of the first two cutoffs and of `hf` of ammonia (48 DMC runs)
  and 115.1 for its third cutoff (16 DMC runs).
- Limits of these runs: two small molecules, and one run of each project. The runs used the code of
  version 0.2.0 before its last corrections of 2026-11-27. With the recorded numbers, the rule of the
  final code of version 0.2.0 makes the same 6 decisions (1 for water, 5 for ammonia).

### What is still not known

- No test of the time step was made at a fixed geometry. The cutoff ladders of the first table used
  the time step 0.005 Ha^-1, and the automatic run of ammonia used 0.02 Ha^-1 at other geometries.
  So a time-step error that differs between the trial wave functions is not excluded.
- The rule of section 10 and the automatic mode were tested with real programs for two small
  molecules only: water and the ammonia inversion (the part above). They were not tested on a larger
  molecule.
- Why the same cutoff keeps different numbers of determinants at two geometries 0.0049 angstrom
  apart (the part above) was not investigated. A choice of the cut that gives wave functions of
  the same quality at every geometry is not built.

### Earlier runs (version 0.1.0 and development code)

- Example `examples/water_multidet`, runs of version 0.1.0 with the input of that version
  (`n_det_max = 10000` and `cutoff = 1e-3, hf`): at the start and at the final geometry of water,
  the trial wave function cut at 1e-3 gave DMC energies above those of the single RHF determinant of
  QP2 (same orbitals, same Jastrow form): 1.53 +/- 0.31 and 1.93 +/- 0.28 millihartree in the run of
  2026-11-25 (50 and 42 determinants), and 2.09 +/- 0.38 and 2.21 +/- 0.42 millihartree in the run of
  2026-11-26 (55 and 42 determinants).
- Engine tests (development code; started on 2026-11-24; 120 blocks of 100 steps, time step 0.005
  Ha^-1), against the single RHF determinant: with the cutoff 1e-3, water +2.6 +/- 0.8 (55
  determinants), ammonia +1.4 +/- 0.9 (178) and HCN -4.9 +/- 1.1 (186) millihartree; with the cutoff
  1e-4, water -2.9 +/- 0.8 (2,092), ammonia -3.1 +/- 1.0 (3,602) and HCN -12.7 +/- 1.2 (4,888)
  millihartree.

In these runs the expansions of the cutoff 1e-3 of water and ammonia gave higher energies than the
single determinant (ammonia in the engine test by 1.5 standard errors). HCN with the cutoff 1e-3, and
all three molecules with the cutoff 1e-4, gave lower energies. For water and ammonia the newer
measurements of this section show the same.

## 10. The automatic choice of the cutoff (`cutoff = auto`)

### The rule

The rule has five steps. The report applies steps 1 to 4 in both modes. The constants are fixed in
the code (`src/montepass/trial_rule.py`); `tolerance` comes from the input.

1. **Each cut against the single determinant.** For every numeric cutoff and every geometry:
   `delta = E(cut) - E(hf)`, with the error `sqrt(error(cut)^2 + error(hf)^2)`. Both energies are DMC
   energies of this stage at the same geometry. The cut is "above" at that geometry if `delta` is
   larger than 2 standard errors. A cutoff is USABLE if it is above at no geometry. A row that is not
   complete (a DMC series that is not `DONE`, no energy, or a VMC check without Jastrow factor that
   did not pass or was not made) gives "not available" with the reason.
2. **The pair.** Of the usable cutoffs, the two with the smallest values are compared. "coarse" is
   the larger value (fewer determinants), "fine" the smaller one. A manual list without `hf` tests no
   cut; then every numeric cutoff counts as "not tested", and the two smallest cutoffs are compared.
3. **Every tested difference.** The tested differences are: (a) final minus start of every target
   whose search moved the geometry, (b) the barrier at the start geometries and (c) the barrier at
   the final geometries of every pair of a saddle target and a minimum target. For each difference
   `D`: `S = D(fine) - D(coarse)`, `e = sqrt(error(D fine)^2 + error(D coarse)^2)`, and
   `bound = |S| + 2 e`. The difference is `CONVERGED` if `bound <= tolerance`, `NOT_CONVERGED` if
   `|S| - 2 e > tolerance`, and `NOT_DECIDED` otherwise. The change of a barrier (final minus start)
   is not tested: it is the difference of (b) and (c).
4. **The verdict of the project.** `NOT_CONVERGED` if one tested difference is `NOT_CONVERGED`, or if
   fewer than two cutoffs are usable; else `NOT_DECIDED` if one is `NOT_DECIDED`; else `CONVERGED`.
5. **The result.** The energy differences of the project are those at the fine cutoff of the pair.
   The report prints them with the verdict and with the bound.

What the verdicts mean:

| Verdict | Meaning |
|---|---|
| `CONVERGED` | every tested difference changed by less than the tolerance between the two largest usable expansions that were computed, with 2 standard errors of margin. It is a statement about these two cutoffs. It does not prove that a much larger expansion gives the same value. |
| `NOT_CONVERGED` | at least one difference changed by more than the tolerance, with 2 standard errors of margin; or fewer than two cutoffs are usable. The reason names every such difference, or every cut that is above the single determinant with its geometry and `delta +/- error`. |
| `NOT_DECIDED` | the error bars are too large to say either. |
| `NO_DIFFERENCE` | the project prints no energy difference (for example one target that the search did not move), so nothing can be tested. |
| `NO_VERDICT` | a row is not complete (a DMC run is not `DONE`, the row has no DMC energy, or the VMC check without Jastrow factor did not pass or was not made), or a difference cannot be compared. No test is possible. The report can give this verdict before the stage has ended, because a DMC run that is not `DONE` makes its row not complete. |

Total energies are not converged with the size of the expansion at the first two cutoffs of the default ladder
(section 9: 3.10 +/- 0.41 and 1.24 +/- 0.38 millihartree between 2.5e-4 and 1e-4). The verdict is
about energy differences only. Every report of a stage with a numeric cutoff says so.

### What the stage does

The stage starts with the first two cutoffs of `auto_cutoffs` and with `hf`, with `series` DMC runs
for each geometry and each cutoff. When every DMC run of its plan is `DONE`, it applies the rule and
makes one decision. The decisions, in this order:

1. `NO_DIFFERENCE` (the project prints no energy difference): the stage ends with this verdict.
2. `NO_VERDICT` (a row is not complete although its runs are `DONE`): the stage ends with this
   verdict.
3. `CONVERGED`: the stage ends with this verdict. It ends also when a smaller cutoff of the plan is
   above by at most 3 standard errors: the two usable cutoffs with the smallest values are converged.
4. A cutoff is "above" at one geometry or more (step 1: by more than 2 standard errors) and at none
   of them by more than 3 standard errors, and the number of series is below `series_max`: every row
   gets one more DMC series. The result is then tested again with more data. The stage makes this
   decision only when more data can change the two cutoffs that are compared. So three conditions
   must hold. First, the cutoff is above by more than 3 standard errors at no geometry: the stage
   takes such a result as final and does not test it again, so the cutoff stays not usable, whatever
   the results at the other geometries are. Second, fewer than two usable cutoffs have a smaller
   value: the pair is the two usable cutoffs with the smallest values, so a cutoff that has two
   usable cutoffs with smaller values would not be in the pair even if it were usable. Third, the
   usable cutoffs and the cutoffs that meet the first two conditions are at least two together: with
   fewer, no pair exists even if no such result is above after more data. If these conditions do
   not hold, decision 5 or 6 follows.
5. `NOT_CONVERGED` (a difference, or fewer than two usable cutoffs): the next cutoff of the ladder is
   added to the plan, with the number of series that the other rows have. If the ladder holds no
   further cutoff, the stage ends with `NOT_CONVERGED`.
6. `NOT_DECIDED`: every row gets one more DMC series. When `series_max` allows no further series,
   the stage first tests the errors. A smaller cutoff would be compared with the fine cutoff, and
   its rows would get the same number of series, so the stage estimates the error of every tested
   difference for that pair as `sqrt(2)` times its error at the fine cutoff (an estimate: it assumes
   that the new rows have the error of the fine rows).
   - If `2 x sqrt(2) x` the error at the fine cutoff is larger than the tolerance for a difference,
     the errors are too large for the verdict `CONVERGED` also with a smaller cutoff. The stage ends
     with `NOT_DECIDED`, and its message names every such difference with its value of `2 x
     sqrt(2) x` the error and says to raise `series_max` above the number of series of the plan.
   - Else, if the ladder holds a further cutoff, it is added to the plan, with the number of series
     that the other rows have.
   - Else the stage ends with `NOT_DECIDED`, and its message says to add a smaller cutoff at the end
     of `auto_cutoffs`. (The default ladder has three cutoffs, so with the defaults the second item
     can add the cutoff 5e-5 once.)

The plan keeps one number of series: every cutoff of the plan, `hf` included, has the same number of
series at every geometry. Every decision ends the stage, adds one series (at most `series_max`) or adds one
cutoff (at most the ladder), so the stage always ends. A run that failed, was stopped, or whose
request the engine refused (section 7, `point ... exists with another geometry`) is handled as in the
manual mode: the stage needs attention, and no decision is made.

After the last decision the status of the stage is `DONE` for every verdict: a verdict is a result,
not a failure. The message of the stage ends with the verdict, and the command `status` and
`results/summary.md` show it, for example:

```
all 18 DMC run(s) are DONE; automatic cutoff: CONVERGED within 1.0 kcal/mol between the cutoffs 2.5e-4 and 1e-4 (3 series)
```

Every decision is one line in `work/driver.log` and one entry in `work/trial/state.json` (entry
`auto`, list `decisions`): the time, the plan, what was measured against `hf`, and for every tested
difference its name, `S`, `e`, the bound and the verdict. The section "Automatic choice of the
cutoff" of `results/trial_wavefunctions.md` prints the same decisions.

### Cost

The number of DMC runs is (number of geometries) x (number of cutoffs with `hf`) x (number of series).
A target has 2 geometries, or 1 if the search did not move it. With the defaults:

| | cutoffs with `hf` | series | DMC runs for 2 geometries |
|---|---|---|---|
| at the start | 3 | 2 | 12 |
| at most | 4 | 4 | 32 |

`montepass check` prints these two numbers for your input. A longer ladder raises the largest number
of cutoffs. The third cutoff of the default ladder is computed only when the first two cutoffs do
not give `CONVERGED`.

Measured cost of the two cutoff ladders of section 9 (one geometry; `hf` with 2 series and 4 cuts
with 1 series each; 2048 walkers, 200 blocks of 400 steps, 32 MPI ranks): 68.1 core-hours for water
and 79.8 core-hours for ammonia. These are the core-hours that the engine counted for the 6 points of
a ladder: every step of each point (the PySCF step, the conversion of the wave function, the VMC
check, the Jastrow optimization and DMC) and the QP2 jobs. One DMC block took, for water: 2.2 s with
`hf`, 7.4 s with the cutoff 2.5e-4 and 9.1 s with the cutoff 1e-4 (5.3 s and 6.2 s with 1e-3 and
5e-4). For ammonia: 2.4 s, 9.2 s and 11.5 s (6.8 s and 7.5 s). So the DMC blocks are the largest
part: about 58 of the 68.1 core-hours for water and about 71 of the 79.8 core-hours for ammonia. The
QP2 jobs are a small part: the CIPSI calculation of the geometry (`n_det_max = 100000`) took 133 s
for water and 395 s for ammonia, in one job with 8 threads, that is 0.30 and 0.88 core-hours.

Measured cost of the stage in the two automatic runs of section 9 (the DMC settings of the pilot runs
of each project; 32 MPI ranks): 136.0 core-hours for water (2 geometries, 12 DMC runs); for ammonia (4
geometries) 192.7 core-hours for the first two cutoffs and `hf` with 4 series (48 DMC runs) and 115.1
core-hours for the third cutoff with 4 series (16 DMC runs).

### What you can change later

- `tolerance`, `series`, `series_max`, `not_converged`: the plan that was reached (its cutoffs and its
  number of series) is kept, no run is stopped, and the stage decides again with the new values. If
  `series` is now larger than the number that the plan reached, every row gets that number.
- `auto_cutoffs`, when the cutoffs that the plan reached are still the first cutoffs of the new ladder
  (for example one more cutoff at the end of the ladder): the same as the first item. The plan is
  kept, no run is stopped, and the stage decides again. A stage that ended with `NOT_CONVERGED`
  because its ladder had no further cutoff then adds the new cutoff. A stage that ended with
  `NOT_DECIDED` and the message "add a smaller cutoff" adds it too (decision 6).
- `auto_cutoffs`, when a cutoff that the plan reached was changed or removed (the cutoffs that the
  plan reached are then no longer the first cutoffs of the new ladder), or `n_det_max`: the stage
  starts from the input again, with the first two cutoffs of `auto_cutoffs`, with `hf`, and with
  `series` DMC runs for each row. A run that is `DONE` is never computed again: the stage finds it
  by its name. A new `n_det_max` changes the name of every run except the runs of `hf`, so all runs
  of the numeric cutoffs are computed again. A run of the old plan that this first plan does not
  hold, and that has not ended, is stopped.
- From a list to `cutoff = auto`, and back: the stage starts from the input of the new mode. A run
  that both plans hold is kept. A run that only the old plan holds, and that has not ended, is
  stopped.
- Make the changes of the two items above (a changed or removed cutoff of the plan, `n_det_max`, a
  switch between a list and `cutoff = auto`) only while no DMC run of the stage is running or waiting.
  The message of the stage then begins with `all N DMC run(s) are DONE`. The reason: a stopped run is
  not continued later (section 5), and a later decision of the automatic mode can add the same run to
  its plan again, for example the third series of a row, or the runs of the third cutoff. The stage
  then needs attention, and its message names every such run: `stopped after N of M blocks; the
  stage needs every run complete`. To continue, wait until no driver job of the project runs, delete the
  folder of each of these runs (the name in the message, below `work/qmc/`), then run
  `montepass retry` and `montepass submit`. These runs start again from their first block.
- Do not change `qp2_basis` or `qp2_pseudo` after the stage has started (as in the manual mode).

A project that version 0.1.0 computed (its record names no `tolerance`) is read by this version. The
stage and the report become `PENDING` once, no run is computed again, and the report gets the tests
of the size of the expansion that its cutoffs allow: the test against `hf` needs `hf` and a numeric
cutoff, and the convergence test needs two numeric cutoffs.
