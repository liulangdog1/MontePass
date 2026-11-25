# DMC energies with several determinants (`[trial] kind = multidet`)

This page describes the option that computes DMC energies with a multi-determinant trial wave function
from Quantum Package 2 (QP2). The user guide (`docs/user_guide.md`, sections 2, 3 and 6) gives the
input keys and the result files in short form. This page gives every step, every check and every
message, and the limits of this version.

## 1. What the option does, and what it does not do

The geometry search always uses one Slater determinant with Kohn-Sham orbitals from PySCF. The option
does not change the search. It adds one stage, `trial`, between the stages `search` and `report`. The
stage starts when the search of every target has ended. For every target it takes two geometries:
the start geometry and the final geometry of the search. For every geometry and every cutoff of the
input it runs `series` independent DMC runs with a multi-determinant trial wave function.

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
geometries, at the final geometries, and its change. Every difference uses rows of one cutoff only.
The report never subtracts a single-determinant energy from a multi-determinant energy: the two trial
wave functions have different nodes, so their fixed-node errors differ.

## 2. How to switch it on

### Input section `[trial]`

| Key | Default | Meaning |
|---|---|---|
| `kind` | `single` | `single`: nothing more is computed, and the other keys are not read. `multidet`: the stage `trial` is added. |
| `n_det_max` | empty | size of the CIPSI wave function of each geometry: a whole number, at least 2. Needed unless every cutoff is `hf`. |
| `cutoff` | empty | one value, or several values separated by commas, for example `1e-3, 1e-4, hf`. A number must be between 0 and 1. A smaller number keeps more determinants. Two cutoffs that agree in their first 6 significant digits count as the same cutoff, and the input file is refused. |
| `series` | 2 | independent DMC runs for each geometry and each cutoff (at least 1). |
| `qp2_basis`, `qp2_pseudo` | `auto` | names of the basis file and of the pseudopotential file of QP2. `auto` takes them from `[qmc] basis` and `ecp`: `ccecpccpvdz`, `ccecpccpvtz`, `ccecpccpvqz`, `ccecpccpv5z` give `cc-pvdz_ecp_ccecp` to `cc-pv5z_ecp_ccecp`; `ccecp` gives `ccecp`. |
| `not_converged` | `stop` | a target whose search ended `NOT_CONVERGED`: `stop` = the stage does not start and the project needs attention; `run` = the stage runs, and the tables mark every value that uses the final geometry of that target. |

Example:

```
[trial]
kind = multidet
n_det_max = 100000
cutoff = 1e-3, 5e-4, hf
series = 2
```

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
  needs (the CIPSI job of a geometry, for example) is kept.

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
| `point ... exists with another trial wave function` | a key was asked for again with another wave function | use a new key; the stage gives every cutoff and `n_det_max` its own key |
| `the search of target(s) ... ended NOT_CONVERGED, so the stage does not start` | the rule `[trial] not_converged = stop` | set `not_converged = run` to compute the stage at that geometry |
| `... asks for [trial] kind = multidet, but the machine profile ... cannot run that stage` | the profile lacks a QP2 section or key, `[job.qp2] time` is 15 minutes or less, or `[trial] series` is larger than 1024 / (2 x `[job.qmc] tasks`) | repair what the `PROBLEM:` lines name |
| `refused for N minutes` | the batch system refused the request of a point for more than 1 hour (for example a time limit above the limit of the partition) | read the refusal in the message; repair the profile, then `retry` |

## 8. Files

```
work/trial/state.json                   record of the stage: rule, plan, state of every point at the last pass
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

## 9. Limits of this version

- The molecule must have charge 0 and multiplicity 1 (closed shell). One electronic state.
- The user chooses `n_det_max` and the cutoffs. `n_det_max = auto` and `cutoff = auto` are not built.
- The coefficients of the determinants are not optimized with the Jastrow factor.
- Measured (examples/water_multidet, 2026-11-25): at the start and at the final geometry of water, the trial wave
  function cut at 1e-3 (50 and 42 determinants) gave DMC energies 1.53 +/- 0.31 and 1.93 +/- 0.28 millihartree
  ABOVE those of the single RHF determinant of QP2 (same orbitals, same Jastrow form). The engine tests saw the same
  for water and ammonia. The cause was not investigated, so treat the energies of this stage as experimental.
- The orbitals are the RHF orbitals of QP2; natural orbitals are not used.
- The search never uses these wave functions. The energies are computed after the geometry is fixed.
- A wave function that exists already can be used only through the engine setting `import_ezfio` (a
  test feature), not through the input file.
- The automatic choice of `qp2_basis` knows only the ccECP cc-pVnZ basis sets.
- The cost of the QP2 jobs and of the DMC runs depends on the molecule and on `n_det_max`. This
  version has no table of measured costs; measure the cost of one geometry before a large run.
