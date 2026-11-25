# Methods

This file says what each stage of the workflow computes. The stage names are the names shown by
`python -m montepass status`.

Symbols: `n` atoms, `x` a geometry (3 n Cartesian coordinates in angstrom), `d` search coordinates,
`t` a displacement along a unit direction (angstrom), `g` an energy slope (kcal/mol/angstrom),
`k` a force constant (kcal/mol/angstrom^2, `E = k t^2 / 2`). Energies of programs are in hartree.
All error bars are 1 standard error.

## 1. Stage `path`: the surrogate path

The surrogate is a density-functional calculation. ORCA makes the path, because it has a path search.

| Input mode | What ORCA does |
|---|---|
| `path` | Optimizes the reactant and the product (`optimize = yes`), with frequencies as a check that both are minima. Then a nudged elastic band search between them (`NEB-TS`: path search followed by a saddle optimization; `NEB-CI`: climbing image only). |
| `images` | Nothing. The given images are the path. |
| `single` | Optimizes the one geometry, with frequencies. |

Two results of the path search are read separately: whether the path converged, and whether the
saddle optimization converged. If the path converged and the saddle optimization did not, the
climbing image is used as the starting saddle geometry and the log says so.

The path images are aligned on image 0 (rigid superposition, no reflection). Every target geometry is
then put on its nearest image by the same kind of superposition. The search directions are defined
in the frame of the aligned path, so a target that is rotated against that frame would give
directions that contain rigid rotation.

## 2. Stage `subspace`: a few search coordinates

A molecule has 3 n - 6 internal coordinates. A search on the DMC surface in all of them is too
expensive, because every coordinate needs its own DMC points. The stage chooses `d` coordinates.

**Principal components of the path (`kind = pca`).** With the aligned images `x_i`, their mean `x_m`
and the matrix `A` whose rows are `x_i - x_m`, the singular value decomposition `A = U S V^T` gives
the components (rows of `V^T`) and the share of the path variance that each one explains,
`S_j^2 / sum S^2`. With `dim = auto` the package takes the smallest `d` whose components together
explain the fraction `variance` (default 0.999), at most `max_dim`. Translation and rotation are
projected out of every component and the components are made orthonormal again.

**Soft modes (`kind = modes`).** Without a path, the `d` softest eigenvectors of the surrogate
Hessian in internal coordinates are used.

The state file records how much of the displacement between two targets the subspace can
represent (1 = all of it).

## 3. Stage `directions`: search directions and their curvatures

For every target the surrogate Hessian is computed in the `d` subspace coordinates and
diagonalized. The eigenvectors are the search directions and the eigenvalues are their surrogate
force constants `k_j`. A minimum must have no negative `k_j` and a saddle exactly one. If that is
not the case the stage stops with a message, because the starting geometry is then not the kind of
stationary point that was asked for.

This Hessian does not come from ORCA. It comes from PySCF with the settings of the QMC trial
orbitals (functional, basis set, pseudopotentials), as central differences of the analytic gradient
(displacement 0.005 angstrom, 1 + 2 d gradient calculations). The reason is the control variate of
section 6: surrogate energies, slopes and curvatures must belong to the same function.

## 4. Stage `autotune`: the QMC settings

See [automatic_settings.md](automatic_settings.md).

## 5. The QMC chain of one geometry

Every DMC point is one batch job in its own folder. The job runs these steps and writes a marker
file after each one, so a job that is started again continues with the first unfinished step.

| Step | Program | What it does |
|---|---|---|
| `scf` | PySCF | Kohn-Sham determinant (default B3LYP, basis `ccecpccpvtz`, ccECP pseudopotentials). Writes the orbitals, the Kohn-Sham energy and the energy of the determinant. The atoms are grouped by element first, because QMCPACK needs that. |
| `c4q` | `convert4qmc` | Converts the orbitals to the QMCPACK format. |
| `vmc0` | QMCPACK | A short VMC run of the determinant alone. Only for the first point of a molecule. |
| `opt` | QMCPACK | Optimizes a one-body and a two-body Jastrow factor (linear method, 8 cycles of 100,000 samples). |
| `dmc` | QMCPACK | A VMC run that makes the walkers, then DMC with T-moves. |

Checks inside the job. A failed check stops the point with a message. It is not repeated, because a
second attempt would give the same wrong result.

1. **System.** The atoms of the orbital file are the atoms of the geometry, the electron count fits
   charge and spin, and every pseudopotential file has the valence charge that the SCF used.
2. **Determinant.** The VMC energy of the determinant alone equals the energy that PySCF computes
   for the same determinant, within 4 standard errors plus 0.005 hartree. A wrong atom order, orbital
   file or pseudopotential fails this test by hartrees. The QMCPACK input of this run holds no
   Jastrow element at all. (A Jastrow element whose coefficients are all zero is not the same thing:
   QMCPACK still applies the electron-electron cusp, which lowers the energy by 0.04 hartree for
   water.)
3. **Jastrow factor.** All numbers of the optimization are finite, and the VMC energy with the chosen
   Jastrow factor is below the energy of the determinant by more than 2 standard errors.

Which optimization cycle is used: among the cycles whose energy variance is within 1.5 times the
smallest one, the cycle with the lowest energy.

Scan points can start their optimization from the Jastrow factor of the centre of their iteration
(`jastrow_reuse = centre`).

Random seeds. Every run gets its own explicit seed. Two independent runs of one geometry use
windows of the prime-number table of QMCPACK that do not overlap, so their energies can be averaged.

## 6. Stage `search`: the geometry search on the DMC surface

Each target is searched separately in its own `d` directions. Four methods exist.

| `method` | What it is |
|---|---|
| `classic` | The surrogate Hessian line search (J. Tiihonen, P. R. C. Kent and J. T. Krogel, J. Chem. Phys. 156, 054104 (2022)) as its first applications used it: per direction a cubic fit through 7 DMC energies in a window chosen from a surrogate scan; a fixed number of iterations (4). It has no convergence test. The package adds a bracket check and a stability audit of every iteration; neither changes a step. |
| `dense-v1` | Dense symmetric scans with the rules of an earlier application of the method to a larger molecule. Kept unchanged as a reference. |
| `dense-v2` | The same idea with a cheaper estimator, one stopping rule for all directions together, and safeguards for a start that is far from the minimum. The default. |
| `auto` | One iteration of `classic`. If its audit resolves every direction and every fitted step lies inside its grid, `classic` continues. Otherwise `dense-v2` starts from the same starting geometry and uses the full-length runs at that geometry again. |

[protocols.md](protocols.md) describes the dense-scan protocols and compares the methods.

A saddle is searched with the same machinery. Along the direction with negative surrogate
curvature the energy is maximized, along all others it is minimized.

## 7. Final energies and what the report states

At the starting geometry and at the final geometry of every target the package runs full-length
DMC runs (2 independent runs each by default). The report gives:

- the DMC energy of both geometries and their difference with its error bar;
- one sentence per target. If the difference is not larger than 2 standard errors the sentence says
  that there is no significant change. If the optimized geometry is lower by more than that, the
  sentence says that the starting geometry is not the DMC minimum, and by how much;
- for a path, the barrier between the starting geometries (this is the usual fixed-geometry DMC
  barrier), the barrier between the optimized geometries, their difference, and the barrier of the
  surrogate;
- the status of every search. `CONVERGED` is a statement of protocol v2 or v1 about the remaining
  energy error. `DONE` is the end of the classic search and is not a convergence statement.

The starting geometry of a search is a geometry of the surrogate, so "start" in the report means
"density-functional geometry".

## 8. What the package does not do

- It does not compute DMC forces. Slopes come from energy differences.
- It does not relax the path between the targets on the DMC surface. It optimizes the chosen
  targets (minima and the saddle).
- It does not correct the fixed-node error. An optimized geometry is the stationary point of the
  fixed-node DMC surface of the chosen trial wave function.
- The trial wave function is one determinant with a one-body and a two-body Jastrow factor.
