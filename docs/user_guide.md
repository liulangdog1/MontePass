# User guide

## 1. What you need

| Program | Used for | Where it runs |
|---|---|---|
| Python 3.9 or newer with numpy, scipy, matplotlib, pillow | the package itself | your computer and the cluster |
| ORCA | geometry optimization, path between the end points, saddle | cluster |
| PySCF (with h5py) | trial orbitals for QMC, surrogate gradients and Hessians | cluster |
| QMCPACK with `convert4qmc`, and `PyscfToQmcpack.py` from its source tree | Jastrow optimization, VMC, DMC | cluster |
| ccECP pseudopotential files `<Element>.ccECP.xml` | QMCPACK | cluster |
| Quantum Package 2 | only for `[trial] kind = multidet` (an experimental stage): trial wave functions with several determinants | cluster |

The package does not need a workflow manager. It writes the input of every program itself and starts
every calculation as a batch job.

Tested versions:

- Unit tests of version 0.1.0: with the versions of `requirements.txt` (development computer, Python
  3.9.6 and 3.12.2) and of `requirements-cluster.txt` (the cluster of the system tests: Python
  3.9.21, numpy 1.26.4, scipy 1.10.1, matplotlib 3.7.1, pillow 9.5.0, PySCF 2.1.1, h5py 3.8.0). For
  version 0.2.0: the whole suite on the cluster (the item after the next one), and 5 test files on
  the development computer with Python 3.9.6 (`tests/test_trial_auto.py`, `tests/test_trial_stage.py`,
  `tests/test_report.py`, `tests/test_multidet.py`, `tests/test_workflow_model.py`).
- A fresh install on the cluster on 2026-11-26: a new clone of version 0.1.0 in a new virtual
  environment of Python 3.13.5. After `pip install -e .` all 551 unit tests passed; PySCF and h5py
  were not installed yet (the unit tests do not need them). Then `pip install pyscf h5py`. After this
  second install the environment held numpy 2.5.5, scipy 1.18.1, matplotlib 3.11.2, pillow 13.0.0,
  PySCF 2.15.1 and h5py 3.16.0, and all 4 examples ran ([examples/README.md](../examples/README.md)).
  These versions were read after the second install, not at the time of the unit tests.
- An install of the code of version 0.2.0 on the cluster on 2026-11-27, in a new virtual
  environment of Python 3.13.5 with numpy 2.5.5, scipy 1.18.1, PySCF 2.15.1 and h5py 3.16.0: all 632
  unit tests of the final code passed. The examples `water_multidet` and `ammonia_inversion` ran
  with `[trial] cutoff = auto` and the ladder 2.5e-4, 1e-4 (the default of that day), with the
  code of that day before its last corrections ([examples/README.md](../examples/README.md)).
- The code of version 0.2.1 in the same environment of the cluster on 2026-11-27: all 634 unit tests
  passed (an archive of the working tree, with its `src` folder first on `PYTHONPATH`). The
  development computer (Python 3.9.6, without h5py, so 1 test is skipped) ran 3 test files:
  `tests/test_multidet.py`, `tests/test_qmcpack_engine.py` and `tests/test_trial_stage.py`.
- Cluster programs of the test runs: ORCA 6.1.0, QMCPACK 4.1.9 (a development build; the inputs that
  the package writes use the legacy drivers) and, for the stage with several determinants, Quantum
  Package 2.
- `PyscfToQmcpack.py` is not part of PySCF. The script is in the folder `src/QMCTools` of a QMCPACK
  source tree, and `[executables] pyscf_qmctools` (section 2) names that folder. The test runs took
  the script from an older QMCPACK source tree than that of the 4.1.9 programs. A copy from the source
  tree of the QMCPACK programs that you use is the safer choice, because `convert4qmc` reads the file
  that the script writes.

## 2. The machine profile

A machine profile is one INI file that holds everything that depends on your cluster: the lines that
prepare the environment of a job, the commands of the programs, the MPI launch line, the folder of the
pseudopotential files, and the batch options of each kind of job.

Make the folder and copy the example, then edit the copy:

```bash
mkdir -p machines/local
cp machines/example_slurm.ini machines/local/<name>.ini
```

The folder `machines/local/` is not tracked by git, so private paths stay private; for the same
reason a fresh clone does not have it. A profile is found by its name in these places, in this
order: the project folder, `$MONTEPASS_MACHINES`, `~/.config/montepass/machines`, `machines/local/`,
`machines/`. The folders `machines/local/` and `machines/` are those of the clone. The package finds
them from the location of its own source files, not from the current folder, so it finds them only
when it is installed from the clone with `pip install -e .`. The input file of a project names its profile with
`[machine] profile = <name>` (section 3).

| Section | Keys | Meaning |
|---|---|---|
| `[machine]` | `name`, `scheduler` | `scheduler = slurm` or `local` (run scripts on this computer, one after the other) |
| `[setup.scf]`, `[setup.qmc]`, `[setup.orca]`, `[setup.driver]` | `lines` | shell lines run at the start of every job of that kind (module loads, exports, activation of a Python environment). A failing line stops the job. |
| `[executables]` | `python`, `qmcpack`, `convert4qmc`, `orca`, `pyscf_qmctools` | commands of the programs. ORCA needs its full path for parallel runs. `pyscf_qmctools` is the folder that holds `PyscfToQmcpack.py`. |
| `[launch]` | `qmc` | MPI launch prefix. `{ranks}`, `{nodes}` and `{ranks_per_node}` are filled in. |
| `[pseudopotentials]` | `dir` | folder with the `<Element>.ccECP.xml` files |
| `[slurm]` | `account`, `qos`, `partition`, `constraint`, `extra` | batch options common to all jobs |
| `[job.scf]`, `[job.qmc]`, `[job.orca]`, `[job.driver]` | `nodes`, `tasks`, `tasks_per_node`, `cpus_per_task`, `mem`, `time`, `constraint` | batch options of one kind of job |

Use the launch line `mpirun -np {ranks} --map-by core --bind-to core`. It gives every MPI process its
own core, also on a node that the job shares with other jobs. The default placement of Open MPI (by
socket) can put two processes on one core when the cores of the job are not spread evenly over the
sockets. Measured on a node where the 32 cores of a job were split 20 + 12 over two sockets: with
`--map-by core --bind-to core` 32 processes ran on 32 cores in 3.0 s; with a plain `--bind-to core`
Open MPI refused to start; with `--bind-to core:overload-allowed` 32 processes ran on 28 cores and
the same run took 21.1 s. Never use `overload-allowed`.

`python -m montepass check <folder>` runs no job. The command checks every key of the input file and
that the geometry files exist. Then the command finds the profile and checks only these points: the
section `[executables]` exists; with SLURM the sections `[slurm]`, `[job.scf]` and `[job.qmc]` exist
(with `scheduler = local` the section `[machine]`); and the pseudopotential folder exists. With
`[trial] kind = multidet` the command also checks the keys for Quantum Package 2 (next part of this
section). The command does not test the programs of `[executables]` (not the ORCA path, not the
folder `pyscf_qmctools`) and does not run the lines of the
`[setup.*]` sections. A wrong path there shows only when the first job of that kind runs.

### Profile sections for Quantum Package 2

These sections and keys are needed only for the input option `[trial] kind = multidet` (section 3), an
experimental stage.
A project without that option does not read them.

| Section | Keys | Meaning |
|---|---|---|
| `[setup.qp2]` | `lines` | shell lines run at the start of every job of Quantum Package 2 (QP2). The first line must be `set +u`. The next lines load the modules that QP2 was built with. The last line reads the rc file of QP2: `source /path/to/qp2/quantum_package.rc`. After these lines the commands `qp_run` and `qp_create_ezfio` must be on the PATH. |
| `[job.qp2]` | `nodes`, `tasks`, `cpus_per_task`, `mem`, `time` | batch options of a QP2 job. QP2 is one process with threads, so `tasks` must be 1. `cpus_per_task` is the number of threads. `cpus_per_task`, `mem` and `time` must be set. |
| `[executables]` | `qp2_python_bin` | folder of a Python installation that has the package `h5py`. The export program of QP2 (`save_for_qmcpack`) gets this folder first on its PATH. The key may be empty if the Python of `[setup.qp2]` has `h5py`. |
| `[job.qmc]` | `time_multidet` | time limit of the QMC job of a point with several determinants. Such a job needs more time for every step. Without this key the value of `time` is used. |
| `[job.qmc]` | `mem_multidet` | memory of the QMC job of a point with several determinants. Without this key the value of `mem` is used. |

Why the first line is `set +u`. Every job script of the package stops when it reads a variable that
is not set. The rc file of QP2 reads such variables. The line `set +u` switches this rule off for
the QP2 job.

The CIPSI calculation of one geometry can need more time than one job has. Then the package submits
a new job that continues the calculation. At most 4 jobs of one geometry may reach their time limit. If
the wave function did not grow in a job that ran for at least a quarter of its time limit, the
calculation fails with a message that names `[job.qp2] time`. `[job.qp2] time` must be more than 15
minutes. Two QP2 jobs never work in the same job folder at the same time: every QP2 job writes the time
into its file `job.heartbeat` once a minute, and a new job that finds a time there less than 3 minutes
old waits up to 4 minutes, then ends with exit code 98. The file `docs/multireference.md` describes
every step, check and message of these jobs.

With `[trial] kind = multidet`, the command `check` also lists what the profile lacks for QP2, and it
prints the plan of the stage: the number of geometries, QP2 calculations, cutoffs and DMC runs. With
`cutoff = auto` the plan names the ladder, the tolerance, the number of DMC series at the start and
the largest number, and it gives the number of DMC runs at the start and at most.

## 3. The input file

`python -m montepass init <folder>` writes a commented `input.ini`. Every key has a default, so a
minimal input names only the geometry files.

### `[system]`

| Key | Default | Meaning |
|---|---|---|
| `name` | `my_molecule` | label used in reports |
| `charge` | 0 | total charge |
| `multiplicity` | 1 | spin multiplicity 2S+1 |

### `[geometry]`

| Key | Meaning |
|---|---|
| `mode` | `path`: a reactant and a product, the path between them is searched. `images`: a path or a few images that you already have, as one multi-frame xyz file. `single`: one geometry, no path. |
| `reactant`, `product`, `saddle_guess` | xyz files for mode `path`. The atom order must be the same in all files. The saddle guess is optional. |
| `images` | multi-frame xyz file for mode `images` |
| `geometry` | xyz file for mode `single` |
| `targets` | the geometries that are optimized on the DMC surface. Mode `path`: any of `reactant`, `product`, `saddle`. Mode `images`: image numbers counted from 0. Mode `single`: `geometry`. The modes `path` and `images` also take one word in place of a list. `barrier`: the first minimum and the saddle of the path, 2 searches, which give the barrier height. `mep`: every image of the path, one search per image; the two ends are searched as minima, the saddle as a saddle, and every other image across the path (kind `image`, next row). |
| `kinds` | only for modes `images` and `single`: `minimum` or `saddle` for each target, in the order of `targets`. Mode `images` has a third kind, `image`: an image between the two ends of the path, which is no stationary point. The package computes the path direction at this image (the difference of the image after it and the image before it), removes this direction from the search subspace, and searches the image as a minimum in the directions that are left. The image moves across the path only. The first and the last image of the path cannot have the kind `image`. Mode `images` also takes `kinds = auto`: the package sets the kinds from the place on the path (first and last image: `minimum`; the saddle image: `saddle`; every other image: `image`). The words `barrier` and `mep` always use these kinds. |
| `saddle_image` | only for mode `images` with `barrier`, `mep` or `kinds = auto`: the number of the saddle image. Default `auto`: the image with the highest energy in the comment lines of the path file (ORCA writes them as `... E -56.4730990`). If the file holds no energies, give the number. |

### `[surrogate]`

| Key | Default | Meaning |
|---|---|---|
| `program` | `orca` | `orca`, or `none` to use the given geometries as they are |
| `method`, `basis` | `B3LYP`, `cc-pVTZ` | level of the ORCA calculations |
| `optimize` | `yes` | optimize the given end points (or the single geometry) first. Mode `images` uses the given images as they are. |
| `frequencies` | `yes` | mode `path`: compute frequencies of the optimized end points and warn if one is not a minimum |
| `neb_images` | 8 | images of the path search |
| `neb_kind` | `NEB-TS` | `NEB-TS` (path, then saddle optimization) or `NEB-CI` (climbing image only) |

### `[subspace]`

| Key | Default | Meaning |
|---|---|---|
| `kind` | `auto` | `pca`: principal components of the path images. `modes`: softest modes of the surrogate Hessian. `auto`: `pca` when a path exists, else `modes`. |
| `dim` | `auto` | number of search coordinates |
| `variance` | 0.999 | with `dim = auto` and `pca`: the smallest number of components that explains this fraction of the path variance |
| `max_dim` | `auto` | upper limit of `dim = auto`. `auto`: 6 principal components for a path; for a single geometry all internal coordinates of the molecule up to 12. If a single geometry has more than 12, the 12 softest modes are searched and the log says how many coordinates are not covered. |

### `[qmc]`

| Key | Default | Meaning |
|---|---|---|
| `mode` | `auto` | `auto`: pilot runs choose the settings. `fixed`: the values below are used as they are, except that the walker count is rounded up to a multiple of the MPI ranks. `iterative`: as `auto`, and a pilot is repeated when a check fails. |
| `max_repeats` | 2 | limit of repeats in mode `iterative` |
| `xc`, `basis`, `ecp` | `b3lyp`, `ccecpccpvtz`, `ccecp` | Kohn-Sham calculation of the trial orbitals (PySCF names) |
| `sigma_target` | 0.2 | wanted error bar (kcal/mol) of the final DMC energy of one geometry. In the modes `auto` and `iterative` the number of blocks of a final run follows from it. |
| `timestep`, `walkers`, `steps`, `warmup_steps` | 0.005, 2048, 500, 2000 | DMC settings. In the modes `auto` and `iterative` values chosen from the pilot runs replace `timestep`, `steps` and `warmup_steps`. `walkers` is used in every mode (the pilot runs use their own walker count), rounded up to a multiple of the MPI ranks. |
| `jastrow_reuse` | `centre` | `centre`: scan points start their Jastrow optimization from the Jastrow factor of their centre. `none`: every point starts from zero. |
| `slow_factor` | 3.0 | a DMC run whose blocks take this many times longer than those of the other points is cancelled and submitted again once, away from its nodes. 0 switches the check off. |

### `[trial]`

This section asks for DMC energies with several determinants. The package computes these energies
after the geometry search. The geometry search itself always uses one determinant.

This stage is experimental. A short expansion can give a DMC energy above that of one determinant.
For water and ammonia the cuts at 1e-3 and 5e-4 gave DMC energies 2.14 to 3.15 millihartree above the
single RHF determinant (standard errors 0.37 to 0.40 millihartree). The cuts at 2.5e-4 and 1e-4 gave
energies 0.78 to 4.05 millihartree below it (standard errors 0.34 to 0.39 millihartree). The measured
cause: the coefficients of the determinants come from Quantum Package without a Jastrow factor, and
the package does not optimize them again ([multireference.md](multireference.md), section 9, gives
all measured values).

With `cutoff = auto` the package chooses the cutoffs itself. It tests each cut against the single
determinant, tests whether every tested energy difference (the energy change of every target that
the search moved, and every barrier at the start geometries and at the final geometries) is
converged with the size of the expansion, adds DMC series or the next cutoff when the test cannot
decide or finds a change above the tolerance, and reports a verdict
([multireference.md](multireference.md), section 10). With a list of cutoffs the report makes the
same two tests, but the package adds no run by itself.

| Key | Default | Meaning |
|---|---|---|
| `kind` | `single` | `single`: nothing more is computed, and the other keys of this section are not read. `multidet`: the workflow gets one more stage, `trial`, between the stages `search` and `report`. |
| `cutoff` | empty | needed for `multidet`. The word `auto`: the automatic mode (the word must stand alone). Or a list that you choose (the manual mode): one value, or several values separated by commas, for example `2.5e-4, 1e-4, hf`. A number between 0 and 1 is the Quantum Package setting `ci_threshold`: Quantum Package removes every determinant whose alpha part or beta part has a summed weight below this number. A smaller number keeps more determinants. The word `hf` means the single RHF determinant of Quantum Package. |
| `n_det_max` | empty | size of the CIPSI wave function of each geometry: a whole number, at least 2. With a list of cutoffs it is needed, unless every cutoff is `hf`. With `cutoff = auto` an empty value means 100000. `n_det_max = auto` is not built. |
| `auto_cutoffs` | `2.5e-4, 1e-4, 5e-5` | only with `cutoff = auto`: the ladder, which holds the cutoffs that the package may use, in the order of their use. At least 2 numbers between 0 and 1, each smaller than the one before it; no `hf`. |
| `tolerance` | 1.0 | kcal/mol, larger than 0. An energy difference counts as converged between two cutoffs when the size of its change plus 2 standard errors is at most this value. The report uses it in both modes. |
| `series` | 2 | number of independent DMC runs for each geometry and each cutoff (at least 1). With `cutoff = auto`: the number at the start. The random numbers of QMCPACK set an upper limit: 16 series with 32 MPI ranks, 8 with 64, 4 with 128. The command `check` reports a number that is too large. |
| `series_max` | 4 | only with `cutoff = auto`: the largest number of DMC runs for each geometry and each cutoff. At least `series`. The upper limit of `series` holds for this number. |
| `qp2_basis`, `qp2_pseudo` | `auto` | names of the basis file and of the pseudopotential file of Quantum Package. `auto` chooses them from `[qmc] basis` and `ecp` (table below). For any other basis, write the name of the Quantum Package file here. |
| `not_converged` | `stop` | what happens when the search of a target ended `NOT_CONVERGED` (it did not reach its precision target). `stop`: the stage does not start, and the project needs attention. `run`: the stage runs, and the tables mark every value that uses the final geometry of such a target. |

What `qp2_basis = auto` chooses:

| `[qmc] basis` | `qp2_basis` |
|---|---|
| `ccecpccpvdz` | `cc-pvdz_ecp_ccecp` |
| `ccecpccpvtz` | `cc-pvtz_ecp_ccecp` |
| `ccecpccpvqz` | `cc-pvqz_ecp_ccecp` |
| `ccecpccpv5z` | `cc-pv5z_ecp_ccecp` |

`[qmc] ecp = ccecp` gives `qp2_pseudo = ccecp`.

The smallest input of the automatic mode:

```
[trial]
kind = multidet
cutoff = auto
```

An input of the manual mode:

```
[trial]
kind = multidet
n_det_max = 100000
cutoff = 2.5e-4, 1e-4, hf
```

What the automatic mode does. The stage starts with the first two cutoffs of `auto_cutoffs` and with
`hf`, with `series` DMC runs for each geometry and each cutoff. When all these runs are done, it
applies the rule of [multireference.md](multireference.md), section 10:

- A cut whose DMC energy is above that of `hf` by more than 2 standard errors at a geometry is not
  usable.
- Every tested energy difference is compared at the two smallest usable cutoffs. `CONVERGED`: for
  every tested difference, the size of its change plus 2 standard errors is at most `tolerance`. The
  stage then ends, also when a smaller cutoff of the plan is above `hf` by at most 3 standard errors.
- If the verdict is `NOT_CONVERGED` or `NOT_DECIDED`, a cutoff is above by at most 3 standard errors, and the number
  of series is below `series_max`, every row gets one more DMC series, so that this result is tested
  again with more data. The stage does this only when more data can change the two cutoffs that are
  compared: the cutoff is above by more than 3 standard errors at no geometry, fewer than two usable
  cutoffs have a smaller value, and two cutoffs can then be compared
  ([multireference.md](multireference.md), section 10, decision 4).
- `NOT_CONVERGED` (a difference changes by more than `tolerance`, with 2 standard errors of margin,
  or fewer than two cutoffs are usable): the next cutoff of `auto_cutoffs` is added. If there is
  none, the stage ends with this verdict.
- `NOT_DECIDED` (the test can say neither): every row gets one more DMC series, up to
  `series_max`. Then the stage estimates the errors that a smaller cutoff would give. If they are
  too large for the verdict `CONVERGED`, the stage ends, and its message says to raise `series_max`. Else the next
  cutoff of `auto_cutoffs` is added; if there is none, the stage ends, and its message says to add
  one ([multireference.md](multireference.md), section 10, decision 6).

The stage is `DONE` after its last decision, whatever the verdict is: a verdict is a result, not a
failure. The message of the stage ends with the verdict, and the command `status` shows it.
`CONVERGED` is a statement about the two cutoffs that were compared. It does not prove that a much
larger expansion gives the same value. Total energies are not converged at the first two default cutoffs; the
test is about energy differences only.

What the stage `trial` does for one cutoff. The stage starts when the search of every target has
ended. The stage takes two geometries of every target: the start geometry and the final geometry of
the search. If the search did not move a geometry (the two are equal within 0.000001 angstrom), only
the start geometry is computed. For every geometry:

1. Quantum Package computes RHF orbitals and one CIPSI wave function with about `n_det_max`
   determinants. All cutoffs of the geometry use this one calculation.
2. For every cutoff, Quantum Package removes determinants from a copy of the CIPSI wave function, as
   the cutoff says. The result is the trial wave function of that cutoff. Its coefficients are not
   optimized again.
3. For every cutoff, one QMC job runs. First it runs VMC with the determinants alone (no Jastrow
   factor). This VMC energy must equal the energy that Quantum Package printed for the same wave
   function. If it does not, the point fails and no DMC runs. Then the job optimizes the Jastrow
   factor and runs DMC. The coefficients of the determinants stay fixed.
4. Every further series is one more DMC run with the same wave function and new random numbers.

The DMC settings are those of the full-length runs of the search: the same time step, walkers,
steps per block and number of blocks. One setting differs: every point of the stage optimizes its
own Jastrow factor from zero. The check of `[qmc] slow_factor` works as for the search points. The
engine compares a run only with runs of the same kind of wave function (one determinant, or the same
cutoff), so the longer steps of a run with several determinants do not count as slow.

For every geometry the numeric cutoffs are asked for before `hf`. Then one Quantum Package job holds
both the RHF and the CIPSI calculation of the geometry.

When the stage needs attention. A failed point does not stop the other points of the stage. The
stage gets the status `NEEDS_ATTENTION` only when no point can make progress any more and at least
one point failed, was stopped, or was refused by the engine. The message of the stage names every
such point with its reason. The engine refuses a run on disk that was made at another geometry or
with another trial wave function than the stage asks for now. The stage does not use the energy of
such a run, also when the run is `DONE`. A refusal by the batch system (for example a time-out of
`sbatch`) is no failure: the request is made again on the next pass.

A target whose search ended `NOT_CONVERGED`. The energies with several determinants are meant for a
geometry that the search has fixed. So with the default `not_converged = stop` the stage does not
start when such a target exists, and its message names the target. Continue the search of that target
first, or set `not_converged = run`.

Limits of this version:

- The molecule must have charge 0 and multiplicity 1.
- `n_det_max = auto` is not built yet. `cutoff = auto` chooses the cutoffs from `auto_cutoffs`; it
  does not choose `n_det_max`.
- The coefficients of the determinants are not optimized with the Jastrow factor.
- The automatic mode ran with real programs for two small molecules only: water and the ammonia
  inversion, with the ladder 2.5e-4, 1e-4 (the default of that day), and for ammonia then with
  2.5e-4, 1e-4, 5e-5 ([multireference.md](multireference.md), section 9).
- A project with a section `[model]` cannot use `kind = multidet`.
- The CIPSI wave function must hold at least 2 times as many determinants as the wave function
  after the cutoff. If it does not, the point fails, and the message says: raise `n_det_max`.
- A cutoff that leaves only 1 determinant fails. Use a smaller cutoff, or `hf`.

What you can change later:

- You can switch the option on after a project is done. The stage `trial` is then added, and the
  report is written again. No point of the search is computed again.
- If you switch the option off again before the stage has started, the stage `report` and the
  project get back the status that they had before, unless something else changed that status in
  between (for example a search point that failed).
- If you switch the option off while the stage runs, every point of the stage that has not ended is
  stopped, and the log names each one. Their blocks stay on disk. The report then has no files of
  the stage.
- You can add a cutoff or raise `series`. Only the new DMC runs are computed.
- You can change `n_det_max`. The names of the points hold `n_det_max`, so every geometry gets a
  new CIPSI calculation and new DMC runs. When the plan of the stage changes (a cutoff removed,
  `series` lowered, `n_det_max` changed), every old point that has not ended is stopped, and the log
  names each one. A stopped point is not continued later. To compute it again, delete its folder.
- With `cutoff = auto`: you can change `tolerance`, `series` and `series_max` at any time. The plan
  that the stage reached (its cutoffs and its number of series) is kept, no run is stopped, and the
  stage decides again with the new values. The same holds for `auto_cutoffs` when every cutoff that
  the plan reached is still in the same place at the start of the new list (for example one more
  cutoff at its end). If you change or remove a cutoff that the plan reached, or change `n_det_max`,
  or switch between `auto` and a list, the stage starts from the input again: a run that is `DONE` is
  not computed again, and a run that the first plan of the input does not hold and that has not ended
  is stopped. Make such a change only while no DMC run of the stage is running or waiting (the message
  of the stage then begins with `all N DMC run(s) are DONE`). A stopped run is not continued later.
  When a later decision of the automatic mode adds a stopped run to its plan again, the stage needs
  attention and names the run ([multireference.md](multireference.md), section 10, says what to do).
- A project that version 0.1.0 computed can be opened with this version. The stage and the report
  are written once more with the new tests that its cutoffs allow; no run is computed again.
- Do not change `qp2_basis` or `qp2_pseudo` after the stage has started. The stage then stops with
  a message, because points with two different basis files cannot be compared.

The results are in `results/trial_wavefunctions.md` and `results/trial_wavefunctions.csv` (section 6).

### `[search]`

| Key | Default | Meaning |
|---|---|---|
| `method` | `dense-v2` | `classic`, `dense-v1`, `dense-v2`, or `auto` (one classic iteration; if its stability audit or its bracket check fails, `dense-v2` starts from the same geometry). See [protocols.md](protocols.md). |
| `eps_total` | 0.2 | tolerance on the energy of the final geometry in kcal/mol |
| `e_cap` | 5.0 | energy rise at the outer scan points in kcal/mol |
| `max_iter` | 20 | largest number of scan iterations per target. In the tests a start near the minimum needed 1 to 3. Real DMC runs from a start moved by 0.2 angstrom (root mean square per atom) needed 3 to 7, and runs on the model surfaces needed 5 to 7 on average ([validation.md](validation.md), sections 5 and 6; `examples/model_study/results.jsonl`). |
| `cap_blocks` | 400 | largest number of DMC blocks per point, and the length of a final run. Used with `[qmc] mode = fixed`; in the other modes the pilot runs set this number from `sigma_target`. |

Small molecules have small DMC error bars, so a tighter tolerance is affordable for them
(`eps_total = 0.02` for water).

### `[experiment]`

Tests of a wrong starting geometry. Leave these keys at their defaults for a normal run.

- `shift = 0.10`, `direction = 1`: the search starts 0.10 angstrom away from the surrogate geometry
  along one search direction.
- `random_rms = 0.10`, `random_seed = 1`: every atom is moved at random (no rotation or translation
  of the whole molecule), with a root mean square displacement of 0.10 angstrom per atom. The seed
  chooses the displacement. `random_space = subspace` keeps only the part inside the search
  directions, for a molecule whose search does not cover all coordinates.
- `move_surrogate = yes` (default) moves the surrogate with the start: the surrogate then calls the
  wrong start its own minimum (right shape, wrong place), and only the DMC energies know where the
  minimum is.

### `[machine]` and `[report]`

| Key | Default | Meaning |
|---|---|---|
| `[machine] profile` | `example_slurm` | name of the machine profile (section 2), or a path to one. The default is the example profile, whose paths are placeholders: write the name of your own profile here. |
| `[report] gif` | `yes` | `yes`: the report writes the animations of `results/gifs/`; `no`: no animations |

## 4. Commands

```
python -m montepass init   [folder]      write a commented input file
python -m montepass check  [folder]      check input and machine profile
python -m montepass submit [folder]      start the driver as a one-core batch job
python -m montepass run    [folder]      run the driver in this terminal (--once for one pass)
python -m montepass status [folder]      stages, targets, last log lines
python -m montepass retry  [folder]      submit failed calculations again and clear NEEDS_ATTENTION
python -m montepass report [folder]      write the results folder again
python -m montepass demo   water|ammonia [folder]     complete run on a model molecule, no cluster
```

The driver is one small process. Each pass it looks at the files of the project, submits what is
missing, reads what has finished, decides, writes the state and sleeps. It can be stopped at any time
and started again; it continues from the files.

What the driver repairs by itself:

- A calculation that failed for a reason of the cluster (the job vanished, a program did not start)
  is submitted again once.
- A DMC run whose blocks take more than 3 times longer than those of the other points of the
  molecule is cancelled and submitted again once, away from the nodes it ran on. Such a run
  usually shares its cores with a job of another user. The setting is `[qmc] slow_factor`
  (0 switches the check off).
- A scan point that holds more blocks than the decision can need is stopped, so that its cores go
  to the points that wait (protocol v2).

If something needs a person, the stage gets the status `NEEDS_ATTENTION` with a message that names
the file and the reason. After the cause is repaired (a module that was missing, a full disk, a time
limit that was too short), give every failed calculation a new set of attempts and start the driver
again:

```
python -m montepass retry  [folder]
python -m montepass submit [folder]
```

`retry` keeps every step that is done. The files of the failed step and a partial DMC trace are set
aside with the ending `.retry<n>`. A check that failed for a reason of the calculation itself (a
Jastrow factor that does not lower the energy, a wrong electron count) fails again in the same way.
`retry` submits again: failed QMC points, failed ORCA jobs, and failed PySCF jobs of the surrogate
(the gradient and Hessian at a scan centre). Jobs that wait, run or are done are not touched.
For a failed point of the stage `trial`, `retry` also gives the failed Quantum Package jobs of its
geometry a new set of attempts. A failed point of an older plan of the stage (for example of a cutoff
that you removed from the input) is not submitted again.

## 5. The project folder

```
<project>/
  input.ini
  work/project.json          state of the workflow
  work/driver.log            what the workflow did, one line per event
  work/surrogate/            ORCA calculations, one folder each; jobs.json lists the submitted jobs
  work/path.xyz              the path images
  work/subspace.npz          the search subspace
  work/targets/<target>/     start.xyz, directions.npz, search/<target>_state.json, search/<target>.log
  work/qmc/                  one folder per QMC point: scf.py, opt.in.xml, dmc.in.xml, outputs, marker files;
                             jobs.json lists the submitted jobs
  work/pyscf/                surrogate gradient, Hessian and energy jobs; jobs.json lists them
  work/autotune/             qmc_settings.json (the chosen settings and their basis), pilot/ (state of the pilot runs)
  results/                   see below
```

A project with `[trial] kind = multidet` has two more places:

```
  work/trial/state.json      record of the stage trial: its rule, the names of its points, and the state and
                             message of every point at the last pass; with cutoff = auto also the plan now
                             (cutoffs and DMC series), every decision with its numbers, and the final verdict
  work/qmc/trial/<target>/<start or final>/
      n<n_det_max>_c<cutoff>/        the DMC run of one cutoff (series 1), for example n10000_c1e-03
      n<n_det_max>_c<cutoff>_s2/     its second series
      hf/                            the DMC run with the single RHF determinant of Quantum Package
      qp2_n<n_det_max>/              the Quantum Package calculations of this geometry: RHF, CIPSI,
                                     and one folder cut_<cutoff> for every cutoff
                                     (the folder is named qp2 when every cutoff is hf)
```

## 6. The results folder

| File | Content |
|---|---|
| `summary.md`, `summary.json` | for every target: DMC energy of the starting (DFT) geometry and of the optimized geometry with error bars, their difference, the status of the search, the geometry change, and one sentence that says whether the starting geometry is the DMC minimum within the error bar. For a path: the barrier at the starting geometries, at the optimized geometries, and of the surrogate. |
| `energies.csv` | one row per DMC point: key, role, blocks, energy, error, surrogate energy |
| `trial_wavefunctions.md`, `trial_wavefunctions.csv` | only with `[trial] kind = multidet`: the DMC energies with several determinants (see below) |
| `geometries/` | start, final and trajectory xyz files of every target; `bond_table.csv` with every bond and angle before and after the search; `path_internal_coordinates.csv` with every bond length and angle of every path image |
| `plots/` | explained variance of the PCA (`pca_variance`), energy profile of the path (`path_profile`), bond lengths and angles along the path (`path_geometry`, `path_geometry_map`), and for every target the convergence of the search, the scans of the last iteration and the geometry change |
| `gifs/` | `path.gif`: the molecule moving along the path, next to the energy profile and to the bond lengths and angles that change most, with their values at the shown image. `<target>_optimization.gif`: the molecule moving from the starting geometry to the optimized geometry |

How to read `trial_wavefunctions.md` (only with `[trial] kind = multidet`). The file has one row
for every target, geometry and cutoff. A row shows:

- the size of the CIPSI wave function, the number of determinants after the cutoff, and the energies
  that Quantum Package printed;
- the checks: the RHF energy of Quantum Package minus the RHF energy of PySCF at the same geometry,
  and the VMC check without Jastrow factor (the VMC energy, its difference to the energy of Quantum
  Package, the limit, and the verdict). A check that failed inside a job shows the verdict `FAIL`;
- the DMC energy of all series together, with its error bar, the core-hours, and the message of
  every run that is not `DONE`;
- in a column of its own, the single-determinant DMC energy of the search at the same geometry.

A sentence above the first table names the basis file, the pseudopotential file and the version of
Quantum Package. The file also gives the search status of every target.

The rows use RHF orbitals of Quantum Package. The single-determinant energy of the search uses
Kohn-Sham orbitals of PySCF (B3LYP, unless `[qmc] xc` names another functional). Both use the same
basis set and pseudopotentials: the RHF energies of the two programs agree (the column of the checks).
The two trial wave functions have different nodes, so their fixed-node errors differ, and the report
never subtracts one of them from the other.

Below the rows the file gives, for every cutoff, the energy change of every target (final geometry
minus start geometry) and the barrier (saddle minus minimum) at the start geometries, at the final
geometries, and its change (final minus start). A difference is printed only if all its rows have
the same cutoff and the same `n_det_max`, every DMC series of every row is `DONE`, and every row
passed the VMC check. Else the cell reads `NOT COMPARABLE:` and the reason. The error of a difference
is the square root of the sum of the squared errors of its rows. A value that uses the final
geometry of a target whose search ended `NOT_CONVERGED` is marked `(search NOT_CONVERGED)`.

The file tests the size of the expansion for you, in up to three sections that follow the DMC
energies ([multireference.md](multireference.md), section 10, has the rule):

- "DMC energy against the single RHF determinant" (when the stage holds `hf` and a numeric cutoff).
  One line for every target, geometry and numeric cutoff: the number of determinants, the DMC energy
  of the cut minus that of the row `hf` at the same geometry in millihartree, and the result:
  `above` (higher by more than 2 standard errors), `not above`, or `not available:` and the reason.
  Then one line for every cutoff: `USABLE` (above at no geometry), `NOT USABLE` (above at one
  geometry or more) or `no verdict yet` (a row of that cutoff, or a row `hf`, is not complete). A cut that is
  above has worse nodes than one determinant at that geometry. This is the one table that subtracts
  rows of two cutoffs. It uses the row `hf` of this stage (RHF orbitals of Quantum Package), not the
  single-determinant energy of the search. A note marks every cut whose CIPSI wave function holds
  fewer than 10 times its determinants: such a cut depends on `n_det_max`.
- "Convergence with the size of the expansion" (with at least two numeric cutoffs). The two smallest
  usable cutoffs are compared. One line for every tested difference (final minus start of every
  target that the search moved; the barrier at the start geometries and at the final geometries):
  its value at the coarse cutoff and at the fine cutoff, the change between them, the bound (the size
  of the change plus 2 standard errors) and the verdict. Then the verdict of the project with its
  reason: `CONVERGED`, `NOT_CONVERGED`, `NOT_DECIDED`, `NO_DIFFERENCE` or `NO_VERDICT`. With a list
  of cutoffs this is a verdict only: the package adds no run by itself.
- "Automatic choice of the cutoff" (only with `cutoff = auto`). The rule, the ladder, every decision
  of the stage in order (what was measured, what was added and why), the final verdict, and the
  result: the energy differences at the fine cutoff, each with its verdict and its bound.

`CONVERGED` means that every tested difference changed by less than `tolerance` between the two
largest usable expansions that were computed, with 2 standard errors of margin. It is a statement
about these two cutoffs. It does not prove that a much larger expansion gives the same value. Total
energies are not converged at the first two default cutoffs: the test is about energy differences only. One
numeric cutoff cannot show convergence.

The file `trial_wavefunctions.csv` holds the same numbers. Its first column, `row`, says what a line
is: `energy` (one geometry and one cutoff), `final minus start`, `barrier` (the column `geometry` says
`start` or `final`), `barrier change`, `against hf` (one cut at one geometry against the row `hf`)
or `convergence` (one tested difference at the two cutoffs of the pair). The column
`uses_not_converged_geometry` is `yes` for a difference that uses the final geometry of a target
whose search ended `NOT_CONVERGED`. The columns of the lines `against hf` and `convergence` stand at
the end of every line, after the columns of version 0.1.0: `dmc_minus_hf_millihartree` and its error,
`against_hf`, `cutoff_usable`, `coarse_cutoff`, `fine_cutoff`, the difference at the coarse and at
the fine cutoff with their errors, `change_fine_minus_coarse_kcal_per_mol` and its error,
`bound_kcal_per_mol`, `tolerance_kcal_per_mol`, `verdict`, `project_verdict` with its reason, and
`automatic_final_verdict`.

How the geometry along a path is shown. A small molecule has a few bonds and angles, and each one
can be a line. A large molecule has too many. The report therefore shows the same information at
three levels:

- `plots/path_geometry` and the animation: lines for the 4 (animation: 3) bond lengths and angles
  that change most. Curves that a figure could not show apart are one line ("and 2 similar").
- `plots/path_geometry_map`: one row for every bond and every angle that changes, one column for
  every path image. The colour is the change from image 0: orange = smaller, purple = larger. The
  rows are sorted from the largest increase to the largest decrease. Next to the map, the molecule
  is drawn with every bond in the colour of its change, so you see where the molecule changes. The
  colour scale ends at 0.3 angstrom and 30 degrees; a bond that breaks goes beyond that and gets the
  end colour. This figure is made when the molecule has more than 4 different bond curves or angle
  curves.
- `summary.md` (section "Geometry along the path") lists the 8 largest changes of each kind, and
  `geometries/path_internal_coordinates.csv` holds every value.

In the animation the bonds of the moving molecule carry the same colours.

How a molecule is drawn. The drawing plane is the best-fit plane of the atoms, so a planar molecule is
shown without distortion. If the atoms move out of that plane along the path (the inversion of ammonia
is an example), two drawings are shown: one from above and one from the side. No drawing uses an oblique
viewing angle, because equal bonds then look unequal. The atom labels (`N1`, `H2`, ...) carry the
position of the atom in your geometry file, counted from 1, and the tables use the same labels.

## 7. How to read the status of a search

| Status | Meaning |
|---|---|
| `RUNNING` | scans are going |
| `FINALIZING` | the search has ended and the full-length runs at the final geometry are going |
| `CONVERGED` | the precision target was reached: the energy that the slope errors can still cost is below `eps_total` (joint 95 % statement) and the predicted gain of a further step is small |
| `NOT_CONVERGED` | the iteration limit or the block limit was reached first. The results are reported with their error bars. |
| `DONE` | the classic line search ended after its fixed number of iterations (it has no convergence test) |
| `NEEDS_ATTENTION` | a calculation failed twice or a check failed. The message says what to look at. The command `retry` continues after the repair. |

## 8. Known problems

- **A point can be submitted twice.** When `sbatch` returns an error (for example a time-out of the SLURM
  controller) but SLURM has still created the job, the package does not see that job and submits the point
  again. Two jobs then write into the same folder. This happened 3 times during our tests. Sign: `squeue` shows
  two jobs with the name of one point. Remedy: cancel the job whose id the file `point.json` of that point does
  not record. A later version will check the queue before it submits a point again.
- **Repeated runs share their random numbers.** The random seeds of a QMC run are computed from the name of its
  point, so two projects with the same input use the same random numbers and are not independent repeats
  ([validation.md](validation.md), section 1, "Random numbers of repeated runs").
