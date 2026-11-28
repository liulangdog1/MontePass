# MontePass

[![tests](https://github.com/liulangdog1/MontePass/actions/workflows/tests.yml/badge.svg)](https://github.com/liulangdog1/MontePass/actions/workflows/tests.yml)

Geometry and reaction-path optimization on the diffusion Monte Carlo (DMC) energy surface,
guided by a density-functional (DFT) surrogate.

The name joins Monte Carlo and mountain pass: a transition state is the mountain pass between
two valleys of the potential energy surface. The Python package and its command are `montepass`.

You give the package unoptimized geometries: a reactant and a product (and a saddle guess if
you have one), or a few images of a path, or a single molecule. The package then runs the whole
chain by itself:

1. **Surrogate path.** ORCA optimizes the end points and finds the path between them
   (nudged elastic band) and the saddle.
2. **Search coordinates.** A principal component analysis (PCA) of the path images gives a few
   collective coordinates. The DFT Hessian, projected on them, gives the search directions.
   With a single geometry and no path, the softest Hessian modes are used.
3. **QMC settings.** Short pilot runs measure variance, correlation time, acceptance and timing.
   The package chooses the time step, the block length and the block counts from them. The walker
   count does not come from the pilot runs: it is `[qmc] walkers` (default 2048), rounded up to a
   multiple of the MPI ranks.
4. **Search on the DMC surface.** Each target geometry (a minimum, a saddle, or an image of the
   path) is optimized with DMC energies. Four methods are built in: the classic line search, the dense-scan search with
   the rules of the first application (protocol v1), an improved dense-scan search (protocol v2,
   the default), and `auto` (one classic iteration, then protocol v2 if its stability audit fails).
5. **Final energies.** Full-length DMC runs at the DFT geometry and at the optimized geometry,
   and the barrier with its error bar. The report says whether the DFT geometry is the DMC
   minimum within the error bar or not.
6. **Report.** A results folder with tables, plots and animations of the path and of the
   optimization.

The QMC chain of one geometry is PySCF (orbitals) -> `convert4qmc` -> QMCPACK (Jastrow
optimization, VMC, DMC). Every calculation is a batch job with marker files, so a run that is
stopped continues where it stopped. No workflow manager is needed.

An optional, experimental stage after the search computes DMC energies with a trial wave function of
several determinants from Quantum Package 2 ([docs/multireference.md](docs/multireference.md)). With
`[trial] cutoff = auto` the package chooses the size of the expansion itself: it tests each cut
against the single determinant, tests whether every tested energy difference (the energy change of
every target that the search moved, and every barrier at the start geometries and at the final
geometries) is converged with the size of the expansion, and reports a verdict.

## Install

```bash
git clone https://github.com/liulangdog1/MontePass.git
cd MontePass
pip install -e .          # needs numpy, scipy, matplotlib, pillow (Python 3.9 or newer)
python -m unittest discover -s tests      # about 10 minutes on one core, needs no cluster program
```

Use the editable install (`pip install -e .`), as above. Among other places, the package looks for
machine profiles in the folder `machines/` of the clone. It finds that folder from the location of
its own source files, not from the current folder, so it finds the folder only when it is installed
from the clone with `pip install -e .` (docs/user_guide.md, section 2).

On the cluster you also need ORCA, PySCF (with h5py) and QMCPACK (with `convert4qmc`), and ccECP
pseudopotential files (`<Element>.ccECP.xml`). Tell the package where they are in a machine
profile, and give the name of the profile in the input file (`[machine] profile = <name>`):

```bash
mkdir -p machines/local        # a fresh clone has no such folder: git does not track it
cp machines/example_slurm.ini machines/local/<name>.ini        # then edit the copy
```

See [docs/user_guide.md](docs/user_guide.md), sections 1 and 2.

Tested versions. The unit tests of version 0.1.0 passed with the package versions of
`requirements.txt` (development computer) and of `requirements-cluster.txt` (the cluster of the
system tests: Python 3.9.21, PySCF 2.1.1, h5py 3.8.0). On 2026-11-26 a fresh clone of version 0.1.0 was installed on the cluster in a
new virtual environment of Python 3.13.5. After `pip install -e .` all 551 unit tests passed (PySCF
and h5py were not installed yet; the unit tests do not need them). After `pip install pyscf h5py`
the environment held numpy 2.5.5, scipy 1.18.1, matplotlib 3.11.2, pillow 13.0.0, PySCF 2.15.1 and
h5py 3.16.0, and all 4 examples ran ([examples/README.md](examples/README.md)).
On 2026-11-27 the code of version 0.2.0 was installed on the cluster from an archive of the
development working tree (not a git clone), in a new virtual environment of Python 3.13.5:
`pip install -e .`, then `pip install pyscf h5py` (numpy 2.5.5, scipy 1.18.1, PySCF 2.15.1, h5py
3.16.0). In this environment all 632 unit tests of the final code passed. For version 0.2.0 the
development computer (Python 3.9.6) ran 5 test files, not the whole suite:
`tests/test_trial_auto.py`, `tests/test_trial_stage.py`, `tests/test_report.py`,
`tests/test_multidet.py` and `tests/test_workflow_model.py`.
On 2026-11-27 the code of version 0.2.1 passed all 634 unit tests in the same environment of the
cluster, run from an archive of the working tree with its `src` folder first on `PYTHONPATH`. The
development computer (Python 3.9.6, without h5py, so 1 test is skipped) ran 3 test files:
`tests/test_multidet.py`, `tests/test_qmcpack_engine.py` and `tests/test_trial_stage.py`.
The cluster programs were ORCA 6.1.0, QMCPACK 4.1.9 (a development build; the inputs use the legacy
drivers) and Quantum Package 2.

## Try it without a cluster

```bash
python -m montepass demo ammonia demo_ammonia     # model molecule, noise like DMC; 41 s on a cluster node
python -m montepass status demo_ammonia
ls demo_ammonia/results
```

The demonstration runs the complete workflow (path, PCA, search of the minimum and of the
inversion saddle, final energies, report) on an analytic model of ammonia. The "true" surface of
the model plays the role of DMC and a second, slightly different surface plays the role of DFT.

## Run a real calculation

```bash
mkdir my_reaction && cd my_reaction
python -m montepass init .          # writes a commented input.ini
# put reactant.xyz and product.xyz here; in input.ini set [machine] profile = <name>.
# With a saddle guess: put saddle_guess.xyz here and set [geometry] saddle_guess = saddle_guess.xyz
python -m montepass check .         # checks the input and parts of the machine profile
python -m montepass submit .        # starts a one-core driver job that runs everything
python -m montepass status .        # what is done, what runs, what needs attention
```

When the driver job has finished, the folder `results/` holds `summary.md`, the tables, the plots
and the animations. Ready-made inputs for water, the ammonia inversion and the HCN to HNC
isomerization are in `examples/`. For water the whole run of version 0.1.0, from the raw geometry to
the report, took 25 minutes of wall time on 2026-11-25 and 43 minutes on 2026-11-26, on a cluster
with 32 cores per QMC job. The QMC jobs other than the pilot runs took 43 and 52 core-hours, and the
pilot runs 2.1 and 3.5 core-hours. [examples/README.md](examples/README.md) gives our results for
every example.

## What has been tested

The package was run on real molecules on a SLURM cluster. [docs/validation.md](docs/validation.md)
lists these runs and the tests on model surfaces, with their results and the defects that the runs
found. In short:

- The complete workflow ran for water, the ammonia inversion and the HCN to HNC isomerization
  (sections 2, 3 and 4). Version 0.1.0 ran all four example inputs from the xyz files to the report,
  the multi-determinant stage included, on 2026-11-25 and again on 2026-11-26 from a fresh clone with
  newer Python packages; every search ended as `CONVERGED` ([examples/README.md](examples/README.md)).
  The input of `examples/water_multidet` was `cutoff = 1e-3, hf` in those runs. It now asks for
  `cutoff = auto`. On 2026-11-27 this input ran with real programs, with the ladder 2.5e-4, 1e-4
  (the default of that day): its last stage ended `CONVERGED` after 12 DMC runs. The ammonia
  inversion with `cutoff = auto` ended `NOT_DECIDED` with the ladder
  2.5e-4, 1e-4 (the default of that day), and `CONVERGED` after a third cutoff was added to its input
  ([examples/README.md](examples/README.md); [docs/multireference.md](docs/multireference.md),
  section 9). The default ladder is now 2.5e-4, 1e-4, 5e-5, so the stage adds the third cutoff by
  itself in such a case.
- Acceptance test (section 6) of the code of 2026-11-22, 16:37, which was not changed during the test:
  17 runs on water, ammonia, HCN, formaldehyde and ethylene, started from geometries in which every
  atom was moved at random, with a root mean square displacement per atom of 0, 0.05, 0.10 or 0.20
  angstrom. All 17 ended as `CONVERGED` after one command, and all 12 runs with a moved start passed
  the rules that were written before the runs. The search modules of version 0.1.0 are the same as in
  that code, except for comments; version 0.1.0 adds path images as search targets and the
  multi-determinant stage. Version 0.2.0 adds the automatic choice of the cutoff
  (`[trial] cutoff = auto`); it does not change the search modules or the QMC engine. Version 0.2.1
  repairs one step of the multi-determinant stage: the package converts the determinant strings of
  Quantum Package 2 itself, because the script of QMCPACK that did it stops with numpy 2 for some wave
  functions ([docs/multireference.md](docs/multireference.md), section 3). As in earlier versions, `retry`
  writes the job files of a failed point again with the installed code before it submits the job.
- The classic line search and the dense scan were compared with real DMC from the same 5 moved
  starts (section 5). The energies of this item are B3LYP energies above the reference geometry (the
  final geometry of the reference run of the acceptance test); a B3LYP energy has no statistical
  noise. From the 3 starts that are 9.9 to 55 kcal/mol above the reference geometry, both methods end
  within 0.06 kcal/mol of it. From the 2 starts at 148 and 395 kcal/mol, the line search ends 4.4 and
  249 kcal/mol above it, and the dense scan ends within 0.02 kcal/mol of it. In DMC energy, these 2
  final geometries of the line search are 3.71 +/- 0.13 and 230.53 +/- 0.14 kcal/mol above those of
  the dense scan.

The package was not tested with open shells, with transition metals, or with a batch system other
than SLURM (section 10 of the same file).

## What is in the repository

```
src/montepass/     the package (see docs/DESIGN.md for the architecture)
tests/              unit tests (standard library, no cluster program needed)
examples/           inputs of the test systems, and a protocol comparison on model surfaces
machines/           machine profiles (an example for a SLURM cluster)
docs/               user guide, method description, design notes, test results
```

## Documentation

- [docs/user_guide.md](docs/user_guide.md): the input file, the machine profile, the commands, the results folder.
- [docs/methods.md](docs/methods.md): what each stage computes, with formulas.
- [docs/protocols.md](docs/protocols.md): the dense-scan protocols v1 and v2 and how they differ, with measured costs.
- [docs/automatic_settings.md](docs/automatic_settings.md): how the QMC settings are chosen.
- [docs/multireference.md](docs/multireference.md): DMC energies with a trial wave function of several determinants (`[trial] kind = multidet`), an experimental stage; its section 9 gives the measured energies, and its section 10 the automatic choice of the cutoff (`[trial] cutoff = auto`).
- [docs/DESIGN.md](docs/DESIGN.md): notes for developers.
- [docs/validation.md](docs/validation.md): the tests on real molecules and on model surfaces, and their results.

## References

- The method `classic` is the surrogate Hessian line search: J. Tiihonen, P. R. C. Kent and J. T. Krogel,
  "Surrogate Hessian accelerated structural optimization for stochastic electronic structure theories",
  J. Chem. Phys. 156, 054104 (2022), doi:10.1063/5.0079046.
- Its use for minimum-energy paths and transition states: G. R. Iyer, N. Whelpley, J. Tiihonen, P. R. C. Kent,
  J. T. Krogel and B. M. Rubenstein, "Force-free identification of minimum-energy pathways and transition
  states for stochastic electronic structure theories", J. Chem. Theory Comput. 20, 7416-7429 (2024),
  doi:10.1021/acs.jctc.4c00214.

## License

MIT, see [LICENSE](LICENSE).
