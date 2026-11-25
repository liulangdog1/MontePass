"""Unit conversion constants used across the package.

Internal conventions
--------------------
* Geometries are Cartesian, in angstrom, as arrays of shape (n_atoms, 3) or flat (3 n_atoms,).
* Raw electronic energies (QMCPACK, PySCF, ORCA) are stored in hartree.
* The search works in kcal/mol and angstrom: slopes in kcal/mol/A, curvatures in kcal/mol/A^2.
"""

HARTREE_TO_KCAL = 627.509474
BOHR_TO_ANGSTROM = 0.529177210903
ANGSTROM_TO_BOHR = 1.0 / BOHR_TO_ANGSTROM
