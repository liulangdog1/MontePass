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
   The package chooses time step, walkers, block length and block counts from them.
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
several determinants from Quantum Package 2 ([docs/multireference.md](docs/multireference.md)).

## Install

```bash
git clone https://github.com/liulangdog1/MontePass.git
cd MontePass
pip install -e .          # needs numpy, scipy, matplotlib, pillow (Python 3.9 or newer)
python -m unittest discover -s tests      # about 10 minutes on one core, needs no cluster program
```

On the cluster you also need ORCA, PySCF and QMCPACK (with `convert4qmc`), and ccECP
pseudopotential files (`<Element>.ccECP.xml`). Tell the package where they are in a machine
profile: copy `machines/example_slurm.ini` to `machines/local/<name>.ini` and edit it.
See [docs/user_guide.md](docs/user_guide.md).

## Try it without a cluster

```bash
python -m montepass demo ammonia demo_ammonia     # model molecule, noise like DMC, about 15 seconds
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
# put reactant.xyz and product.xyz (and saddle_guess.xyz) here, edit input.ini
python -m montepass check .         # checks the input and the machine profile
python -m montepass submit .        # starts a one-core driver job that runs everything
python -m montepass status .        # what is done, what runs, what needs attention
```

When the driver job has finished, the folder `results/` holds `summary.md`, the tables, the plots
and the animations. Ready-made inputs for water, the ammonia inversion and the HCN to HNC
isomerization are in `examples/`. For water the whole run of version 0.1.0, from the raw geometry to
the report, took 25 minutes of wall time and 43 core-hours of DMC jobs on a cluster with 32 cores per
QMC job ([examples/README.md](examples/README.md), which also gives our results for every example).

## What has been tested

The package was run on real molecules on a SLURM cluster. [docs/validation.md](docs/validation.md)
lists these runs and the tests on model surfaces, with their results and the defects that the runs
found. In short:

- The complete workflow ran for water, the ammonia inversion and the HCN to HNC isomerization
  (sections 2, 3 and 4). Version 0.1.0 ran all four example inputs from the xyz files to the report,
  the multi-determinant stage included; every search ended as `CONVERGED`
  ([examples/README.md](examples/README.md)).
- Acceptance test of the final code (section 6): 17 runs on water, ammonia, HCN, formaldehyde and
  ethylene, started from geometries in which every atom was moved at random by 0, 0.05, 0.10 or
  0.20 angstrom. All 17 ended as `CONVERGED` after one command, and all 12 runs with a moved start
  passed the rules that were written before the runs.
- The classic line search and the dense scan were compared with real DMC from the same 5 moved
  starts (section 5). Both recover the 3 starts that are up to 55 kcal/mol above the minimum. From
  the 2 starts at 148 and 395 kcal/mol the line search ends 4.4 and 249 kcal/mol above the minimum,
  and the dense scan ends within 0.02 kcal/mol.

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
- [docs/multireference.md](docs/multireference.md): DMC energies with a trial wave function of several determinants (`[trial] kind = multidet`).
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
