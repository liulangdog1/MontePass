# Design notes for developers

This file describes how the package is built: the parts, the interfaces between them, and the
rules that every module follows. Read it before changing or adding a module.

## 1. What the package does

Input: molecular geometries (a reactant and a product, optionally a saddle guess; or a few
images; or one geometry) and a small input file.

Steps:

1. **Surrogate path (ORCA).** Optimize the end points, find the path between them (nudged
   elastic band) and the saddle. A path given by the user replaces this step.
2. **Subspace.** A principal component analysis (PCA) of the path images gives a few orthonormal
   Cartesian displacement vectors. With one geometry and no path, the softest Hessian modes are used.
3. **Search directions.** The surrogate Hessian, projected into the subspace and diagonalized,
   gives the search directions and their curvatures at each target geometry.
4. **QMC settings.** Short pilot runs measure variance, correlation time, acceptance and timing,
   and the package chooses the time step, the block length and the block counts. The walker count is
   `[qmc] walkers` (default 2048), rounded up to a multiple of the MPI ranks.
5. **Search.** Each target geometry (minimum or saddle) is optimized on the DMC energy surface:
   the classic line search, the stability audit, and the dense-scan search (protocol v1 or v2).
6. **Final energies.** Full-length DMC runs at the starting geometry and at the optimized
   geometry; barriers with error bars.
7. **Report.** Tables, plots and animations in a results folder.

## 2. Layout of the source

```
src/montepass/
  units.py            conversion constants and unit conventions
  linalg.py           a matrix product that checks its result
  geometry.py         xyz files, rigid alignment, rigid-body motions, bonds
  subspace.py         PCA subspace, mode subspace, search directions
  stats.py            QMCPACK block traces: reading, error bars, equilibration
  model.py            analytic model molecules, small numerical tools (tests and demonstrations)
  machine.py          machine profile (programs, modules, launch lines, batch options)
  scheduler/          base.py (interface, job table), slurm.py, local.py
  engine/             base.py (interface), mock.py (model surface + noise),
                      qmcpack_inputs.py (text of the QMC chain inputs), qmcpack.py (real engine),
                      multidet.py (trial wave functions with several determinants from Quantum Package 2:
                      its jobs, its checks, the QMC steps mdc4q and mdvmc; see docs/multireference.md)
  surrogate/          base.py (interface), analytic.py (model surfaces),
                      orca.py (ORCA inputs, outputs, jobs), pyscf_jobs.py (PySCF gradient and Hessian jobs)
  search/             protocol_v1.py, protocol_v2.py (rules), dense.py, dense_v2.py (drivers),
                      classic.py (classic line search), audit.py (stability audit),
                      auto.py (classic first, dense scans if the audit fails), symmetry.py
  autotune.py         automatic choice of the QMC settings
  monitor.py          checks on a running or finished QMC trace
  config.py           the input file
  project.py          folder layout and state of one project
  workflow.py         the stages and their order; the backend for model molecules
  trial.py            the optional stage "trial": after the search, DMC energies with several determinants at the
                      start and the final geometry of every target (input section [trial]; tables in report/trial.py;
                      docs/multireference.md). With [trial] cutoff = auto the stage chooses its cutoffs and its
                      number of DMC series itself, and keeps every decision in its record
  trial_rule.py       the rule for the size of the determinant expansion: each cut against the single RHF
                      determinant, and the convergence of every tested energy difference between two cutoffs.
                      Pure functions; the stage uses them to decide, the report to print the same tests
  cluster.py          the backend that runs ORCA, PySCF and QMCPACK through a batch system
  shifted.py          a rigidly moved surrogate, for tests with a wrong starting geometry
  report/             tables, plots, animations
  cli.py              command line
```

## 3. Conventions (every module)

- Python 3.9 or newer. Only numpy, scipy, matplotlib and pillow are needed. No other third-party package.
- Geometries: Cartesian, angstrom, shape (n_atoms, 3). The atom order of the user's files is kept in
  every file that the user sees. A module that needs another order (QMCPACK needs atoms grouped by
  element) keeps the permutation and applies it only inside its own files.
- Energies of programs are stored in hartree. The search works in kcal/mol and angstrom.
- Curvature: the force constant `k` (kcal/mol/A^2, E = 0.5 k t^2) is the stored quantity. Protocol v1
  uses `Lambda = k / 2`; the conversion is done in one place (`Directions.lam()`).
- A direction is a unit-norm Cartesian vector of length 3 n_atoms. A displacement `t` along it is in angstrom.
- xyz files are read only with `geometry.read_xyz` / `read_xyz_frames`. The comment line may be empty
  and is never skipped.
- QMCPACK traces are read by column name (`stats.read_scalar`), never by column number.
- Every number that is reported has an error bar (1 standard error) or is marked as exact.
- State is kept in JSON files that are rewritten atomically (write to `*.tmp`, then rename). No pickle files.
  A process that is started again must continue from the files alone.
- A step that fails must say so: raise an exception with the file and the reason, or set a state
  `FAILED` / `NEEDS_ATTENTION` with a message. A missing or empty file is never read as "nothing to do".
- No path, module name, user name or cluster name in the source. Such values come from the machine profile.
- Tests: `python -m unittest discover -s tests`, standard library only; no test file needs a cluster
  program. In a run of the development version of the tests on the cluster on 2026-11-24 (22 files
  and 538 tests; one of these files is not in this repository), `test_dense_search`, `test_report`,
  `test_trial_stage` and `test_workflow_model` took 61 to 125 s each, and each of the other 17 files
  of this repository took 17 s or less. Version 0.1.0 has 21 test files and 551 tests; version 0.2.0 has 22 test files and 632 tests. Code that talks to a program (ORCA, PySCF, QMCPACK, SLURM) is tested
  against saved output files and a fake scheduler.
- Docstrings and messages are written in plain English: short sentences, exact numbers, units given.

## 4. Interfaces

### 4.1 Engine (`engine/base.py`)

The search asks an engine for DMC block energies of geometries.

```python
engine.submit(PointSpec(key, symbols, coords, blocks, role, series, base_key, settings))
data = engine.read(key)      # PointData: state, energy[], weight[], e_surrogate, message, meta
engine.stop(key)             # stop early; the blocks written so far stay usable
engine.advance()             # mock engine only: let time pass
engine.retry(key)            # a FAILED point gets new attempts (False if the engine cannot do this)
```

- `key` is unique in a project and path-like (`min/it2/d0_p091`). It is also the folder of the point
  below the engine's root.
- `submit` for an existing key does nothing. In `QmcpackEngine` the request must then name the same
  geometry and the same trial wave function (one determinant, or the same cut of the same
  multi-determinant wave function); otherwise `submit` raises ValueError, because the stored energies
  would not be those of the request. A further series that names no trial wave function gets the one of
  its base point. The mock engine makes no such check. `read` of an unknown key gives state `ABSENT`.
- States: `ABSENT, QUEUED, RUNNING, DONE, STOPPED, FAILED`.
- `energy` and `weight` hold the DMC blocks written so far (hartree), also while the job runs.
- `e_surrogate` is the surrogate (DFT) energy of the same geometry in hartree, as soon as it is known.
- A point with `series > 1` is a further independent DMC run of the geometry of `base_key`: same orbitals,
  same Jastrow factor, random seeds that do not overlap with any other series of that geometry.
- `role` is `scan`, `centre`, `final` or `pilot`. `settings` holds the QMC settings of the point.

### 4.2 Surrogate (`surrogate/base.py`)

```python
surrogate.request(tag, coords, hessian=True, gradient=True)
res = surrogate.result(tag)        # None while running; SurrogateResult(energy, gradient, hessian or hessian_directions)
surrogate.request_scan(tag, geometries);  energies = surrogate.scan(tag)
```

Units: energy hartree, gradient kcal/mol/A (n_atoms, 3), Hessian kcal/mol/A^2.

### 4.3 Scheduler (`scheduler/base.py`)

```python
job_id = scheduler.submit(script_path, workdir, name, after=[job ids])
state = scheduler.state(job_id)     # PENDING, RUNNING, COMPLETED, FAILED, CANCELLED, TIMEOUT, UNKNOWN
scheduler.cancel(job_id)
```

Every submitted job is recorded in a `JobTable` (JSON). Jobs are never found by their names.

### 4.4 Machine profile (`machine.py`)

`Machine(path)` reads an INI file. `machine.header(kind, name, workdir, time=...)` returns the first
part of a job script (batch options, environment, thread variables) for the kinds `scf`, `qmc`, `orca`,
`driver`. `machine.launch("qmc", ranks)` returns the MPI launch prefix. `machine.executable(name)`
returns the command of a program. `machine.pseudo_dir` is the pseudopotential folder.

### 4.5 Job scripts and marker files

Every calculation runs as a shell script in its own folder. A script is a list of steps. Each step

- is skipped if `<step>.done` exists,
- writes `<step>.started`, runs under a time limit, checks the exit code and the expected output,
- writes `<step>.done` or `<step>.failed` (with one line that says why).

So a job that is stopped and started again continues with the first unfinished step. The reader of a
folder decides from the marker files and the output files, not from the state of the batch system alone.

What the engine repairs by itself, and what it leaves to a person (`engine/qmcpack.py`):

| Event | Reaction |
|---|---|
| a job ends without `dmc.done` (the program did not start, the job vanished) | one more attempt; steps that are done are skipped |
| a DMC run needs more than `slow_factor` (3) times the time per block of the other points | the job is cancelled and submitted again once, with `--exclude` for its nodes; the slow blocks are set aside |
| a job reaches its time limit after it wrote blocks | one more attempt if the run was slow in the sense above; otherwise the point fails, because a second attempt would reach the same limit |
| a quality check fails (electron count, VMC of the determinant, Jastrow factor) | the point fails at once: a repeat would give the same result |
| a point failed for good | the search reports `NEEDS_ATTENTION`; `engine.retry(key)` (command `retry`) gives the point a new set of attempts |
| a point is asked for again with another geometry | `submit` raises `ValueError`: the state of the search does not fit the points on disk |
| a PySCF job of the surrogate fails (not at its time limit, or with no progress) | the search reports `NEEDS_ATTENTION` with the message of the job; `PyscfSurrogate.retry_failed()` (command `retry`) submits it again |

### 4.6 Search drivers (`search/dense.py`, `search/dense_v2.py`, `search/classic.py`)

A driver optimizes one target geometry. It is built from an engine, a surrogate, the symbols, the start
geometry, the direction vectors and their surrogate curvatures, a protocol object and a state folder.
`driver.tick()` is one pass: submit what is missing, read what exists, decide, and write the state file.
It returns the status: `RUNNING`, `FINALIZING`, `CONVERGED`, `NOT_CONVERGED` or `NEEDS_ATTENTION`
(the classic search ends as `DONE`, and as `UNRESOLVED` when it runs inside the automatic search and
its first iteration fails the audit). `driver.final_geometry()`, `driver.trajectory()` and the state
file hold the results.

Protocol v2 has two rules about the number of blocks of a scan point, both in `dense_v2.py`:
`_trim_points` stops a running point that holds 3 times the usable blocks that the precision target
is expected to need (so that points do not keep cores while others wait in the queue), and `_decide`
repeats a scan with more blocks per point when the blocks of the scan did not reach the precision
target. Every search has `clear_attention()`, which the command `retry` calls.

A driver must be safe against a stop at any line. The rule in `dense_v2.py`: a decision (the next
centre, or the final geometry) is saved with `_save()` before a job of the next stage is submitted,
and every pass asks again for the jobs that the saved state needs (`submit` of an existing key does
nothing). A process that starts again then finds the decision and completes the submissions. Without
this order, the new process would decide again with more blocks, get another centre, and read the
jobs of the first decision as if they belonged to it.

A search reads the blocks of a point while the point still runs. So an engine, and every wrapper
around an engine, must give the same kind of data in every state of a point. A defect of this
kind was found in the first real test of `shifted.py`: the wrapper replaced the surrogate energy
only for finished points.

### 4.7 Backends (`workflow.py`, `cluster.py`)

The workflow asks a backend for five things: the geometries of the targets and the path, an engine,
a surrogate for a set of directions, a Cartesian Hessian, and the QMC settings. `ModelBackend` answers
from analytic model molecules. `ClusterBackend` answers with ORCA jobs (path), PySCF jobs (surrogate of
the search, at the level of the trial orbitals) and the QMCPACK engine. A new program is added as a
new backend or as a new engine or surrogate class; the search drivers do not change.

## 5. The QMC chain of one geometry

```
python scf.py                              -> scf.h5, energy.dat, system.json [, gradient.dat]
convert4qmc -prefix c4q -orbitals scf.h5   -> check of the orbital file
qmcpack vmc0.in.xml                        -> check: VMC of the determinant alone (no Jastrow element) = its PySCF energy
qmcpack opt.in.xml                         -> Jastrow optimization
qmcpack dmc.in.xml                         -> VMC (series 000) and DMC (series 001)
```

`engine/qmcpack_inputs.py` writes these files. Trial wave function: one Kohn-Sham determinant
(B3LYP, ccECP pseudopotentials, cc-pVTZ basis for ccECP), one-body and two-body B-spline Jastrow factor.
The atoms are grouped by element before the SCF script is written, because QMCPACK lists ions by
element while the orbital file keeps the SCF order.

## 6. Tests on real systems

The unit tests need no cluster. The system tests run the whole workflow on a cluster for small
molecules (water, ammonia inversion, HCN to HNC), also from wrong starting geometries, and compare
with reference values. Their results are in [validation.md](validation.md).

## 7. Rules for changes

- Protocol v1 is a copy of registered rules. Do not change `search/protocol_v1.py`.
- A change of protocol v2 is checked with `examples/model_study/calibrate_v2.py` on the model
  surfaces (96 runs per case; docs/protocols.md, section 5, has the numbers of the last run). The
  fraction of `CONVERGED` runs whose final geometry is within the tolerance must stay at 95 % or
  more in every case in which the surrogate has the right couplings (18 cases; lowest value of the
  last run 97 %). In the four cases with a turned surrogate it must not fall below the values of the
  last run (92 % to 99 % at the tolerance 0.2, 68 % to 95 % at the tolerance 0.05).
  `compare_protocols.py` must not show a higher cost.
- A rule that did not help stays in the code as an option that is off, with the measured numbers in
  docs/protocols.md (`pool`, `secant`, `shrink_final`, `coupling_fit`, `shape_distrust`). So nobody
  tries it again without knowing the result.
- Every number that a module prints for the user carries its unit and, if it is a measurement,
  its error bar (1 standard error).
