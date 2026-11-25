"""Molecular geometries: xyz files, rigid alignment, rigid-body motions, bond lists.

All coordinates are in angstrom. A geometry is a list of element symbols plus an
array of shape (n_atoms, 3). The atom order of the user's input files is kept
everywhere in the package.
"""
from __future__ import annotations

import re
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .linalg import product

_COMMENT_ENERGY = re.compile(r"(?:^|\s)E\s+([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)(?=\s|$)")


def comment_energy(comment: str) -> Optional[float]:
    """Energy in the comment line of an xyz frame: the number after a single letter ``E``, or None.

    ORCA writes the images of a path this way (``Coordinates from ORCA-job neb_MEP E -56.473099``), and
    this package does it in ``work/path.xyz`` (``image 3 E -56.5411304047``). The unit is not known here:
    the value is only good for comparing the images of one file with each other.
    """
    found = _COMMENT_ENERGY.search(comment or "")
    return float(found.group(1)) if found else None

# Covalent radii in angstrom (Cordero et al., Dalton Trans. 2008, 2832), elements H to Kr.
COVALENT_RADIUS = {
    "H": 0.31, "He": 0.28, "Li": 1.28, "Be": 0.96, "B": 0.84, "C": 0.76, "N": 0.71, "O": 0.66,
    "F": 0.57, "Ne": 0.58, "Na": 1.66, "Mg": 1.41, "Al": 1.21, "Si": 1.11, "P": 1.07, "S": 1.05,
    "Cl": 1.02, "Ar": 1.06, "K": 2.03, "Ca": 1.76, "Sc": 1.70, "Ti": 1.60, "V": 1.53, "Cr": 1.39,
    "Mn": 1.39, "Fe": 1.32, "Co": 1.26, "Ni": 1.24, "Cu": 1.32, "Zn": 1.22, "Ga": 1.22, "Ge": 1.20,
    "As": 1.19, "Se": 1.20, "Br": 1.20, "Kr": 1.16,
}

ATOMIC_NUMBER = {s: i + 1 for i, s in enumerate(
    ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
     "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr"])}

# Number of valence electrons left by the ccECP pseudopotentials (used only for cost estimates;
# the electron count of a calculation is always taken from PySCF itself).
CCECP_VALENCE = {
    "H": 1, "He": 2, "Li": 1, "Be": 2, "B": 3, "C": 4, "N": 5, "O": 6, "F": 7, "Ne": 8,
    "Na": 1, "Mg": 2, "Al": 3, "Si": 4, "P": 5, "S": 6, "Cl": 7, "Ar": 8,
    "K": 9, "Ca": 10, "Sc": 11, "Ti": 12, "V": 13, "Cr": 14, "Mn": 15, "Fe": 16, "Co": 17, "Ni": 18,
    "Cu": 19, "Zn": 20, "Ga": 3, "Ge": 4, "As": 5, "Se": 6, "Br": 7, "Kr": 8,
}


def normalize_symbol(symbol: str) -> str:
    """'c' -> 'C', 'CL' -> 'Cl'. Raises ValueError for an unknown element."""
    s = symbol.strip()
    s = s[:1].upper() + s[1:].lower()
    if s not in ATOMIC_NUMBER:
        raise ValueError("unknown element symbol: %r" % symbol)
    return s


# ----------------------------------------------------------------------------- xyz files
def _parse_frames(lines: List[str], path: str) -> List[Tuple[List[str], np.ndarray, str]]:
    """Parse one or more xyz frames.

    The second line of a frame is a comment and may be EMPTY. It is never
    skipped as a blank line: doing so shifts every atom by one line and gives a
    wrong geometry with no error message.
    """
    frames = []
    i = 0
    n_lines = len(lines)
    while i < n_lines:
        if not lines[i].strip():
            i += 1          # blank lines are allowed only between frames
            continue
        try:
            n = int(lines[i].split()[0])
        except (ValueError, IndexError):
            raise ValueError("%s: line %d should hold the atom count, found %r" % (path, i + 1, lines[i]))
        if n <= 0:
            raise ValueError("%s: atom count %d on line %d" % (path, n, i + 1))
        if i + 2 + n > n_lines:
            raise ValueError("%s: frame starting on line %d announces %d atoms but the file ends early"
                             % (path, i + 1, n))
        comment = lines[i + 1].rstrip("\n")
        symbols, coords = [], []
        for k in range(n):
            parts = lines[i + 2 + k].split()
            if len(parts) < 4:
                raise ValueError("%s: line %d is not an atom line: %r" % (path, i + 3 + k, lines[i + 2 + k]))
            symbols.append(normalize_symbol(parts[0]))
            coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
        frames.append((symbols, np.array(coords, dtype=float), comment))
        i += 2 + n
    if not frames:
        raise ValueError("%s: no geometry found" % path)
    return frames


def read_xyz_frames(path: str) -> List[Tuple[List[str], np.ndarray, str]]:
    """Read every frame of an xyz file: a list of (symbols, coordinates, comment)."""
    with open(path) as fh:
        lines = fh.read().split("\n")
    return _parse_frames(lines, path)


def read_xyz(path: str) -> Tuple[List[str], np.ndarray]:
    """Read a single-geometry xyz file. Raises ValueError if the file holds several frames."""
    frames = read_xyz_frames(path)
    if len(frames) != 1:
        raise ValueError("%s holds %d frames; use read_xyz_frames" % (path, len(frames)))
    return frames[0][0], frames[0][1]


def format_xyz(symbols: Sequence[str], coords: np.ndarray, comment: str = "") -> str:
    coords = np.asarray(coords, dtype=float).reshape(-1, 3)
    if len(symbols) != len(coords):
        raise ValueError("%d symbols but %d coordinate rows" % (len(symbols), len(coords)))
    if "\n" in comment:
        raise ValueError("the comment line of an xyz file cannot contain a line break")
    out = ["%d" % len(symbols), comment]
    for s, r in zip(symbols, coords):
        out.append("%-2s %18.10f %18.10f %18.10f" % (s, r[0], r[1], r[2]))
    return "\n".join(out) + "\n"


def write_xyz(path: str, symbols: Sequence[str], coords: np.ndarray, comment: str = "") -> None:
    """Write one geometry and read it back to check the atom order and the coordinates."""
    with open(path, "w") as fh:
        fh.write(format_xyz(symbols, coords, comment))
    back_symbols, back_coords = read_xyz(path)
    if list(back_symbols) != [normalize_symbol(s) for s in symbols]:
        raise IOError("%s: atom order changed on write" % path)
    if np.abs(back_coords - np.asarray(coords, dtype=float).reshape(-1, 3)).max() > 1e-9:
        raise IOError("%s: coordinates changed on write" % path)


def write_xyz_frames(path: str, symbols: Sequence[str], frames: Sequence[np.ndarray],
                     comments: Optional[Sequence[str]] = None) -> None:
    with open(path, "w") as fh:
        for i, coords in enumerate(frames):
            fh.write(format_xyz(symbols, coords, comments[i] if comments else ""))


# ----------------------------------------------------------------------------- alignment
def kabsch(mobile: np.ndarray, target: np.ndarray, allow_reflection: bool = False
           ) -> Tuple[np.ndarray, float, bool]:
    """Best rigid superposition of ``mobile`` on ``target`` (same atom order, unit weights).

    Returns ``(aligned, rmsd, reflected)``. ``aligned`` is the moved copy of
    ``mobile``. With ``allow_reflection=True`` the mirror image is also tried and
    the better of the two is returned; ``reflected`` says which one was used.
    """
    P = np.asarray(mobile, dtype=float).reshape(-1, 3)
    Q = np.asarray(target, dtype=float).reshape(-1, 3)
    if P.shape != Q.shape:
        raise ValueError("kabsch: shapes differ, %s and %s" % (P.shape, Q.shape))
    pc, qc = P.mean(axis=0), Q.mean(axis=0)
    P0, Q0 = P - pc, Q - qc
    V, _S, Wt = np.linalg.svd(P0.T @ Q0)
    det = np.sign(np.linalg.det(V @ Wt))

    def apply(sign):
        D = np.diag([1.0, 1.0, sign])
        rot = V @ D @ Wt
        moved = P0 @ rot + qc
        return moved, float(np.sqrt(((moved - Q) ** 2).sum() / len(P)))

    proper, rmsd_proper = apply(det if det != 0 else 1.0)
    if not allow_reflection:
        return proper, rmsd_proper, False
    mirror, rmsd_mirror = apply(-det if det != 0 else -1.0)
    if rmsd_mirror < rmsd_proper - 1e-12:
        return mirror, rmsd_mirror, True
    return proper, rmsd_proper, False


def superposition(mobile: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Best proper rotation of ``mobile`` onto ``target`` (same atom order, unit weights).

    Returns ``(rot, centre_mobile, centre_target, rmsd)`` with
    ``aligned = (mobile - centre_mobile) @ rot + centre_target``. Use it when not only the
    coordinates but also vectors or matrices of the mobile frame must be moved (see
    :func:`rotate_hessian`).
    """
    P = np.asarray(mobile, dtype=float).reshape(-1, 3)
    Q = np.asarray(target, dtype=float).reshape(-1, 3)
    if P.shape != Q.shape:
        raise ValueError("superposition: shapes differ, %s and %s" % (P.shape, Q.shape))
    pc, qc = P.mean(axis=0), Q.mean(axis=0)
    V, _S, Wt = np.linalg.svd((P - pc).T @ (Q - qc))
    det = np.sign(np.linalg.det(V @ Wt))
    rot = V @ np.diag([1.0, 1.0, det if det != 0 else 1.0]) @ Wt
    moved = (P - pc) @ rot + qc
    return rot, pc, qc, float(np.sqrt(((moved - Q) ** 2).sum() / len(P)))


def rotate_hessian(hessian: np.ndarray, rot: np.ndarray) -> np.ndarray:
    """Cartesian Hessian (3 n x 3 n) in the frame that is reached by ``coords @ rot``.

    A displacement ``u`` (n x 3) of the old frame is ``u @ rot`` in the new frame, so the
    Hessian becomes ``B H B^T`` with ``B = I_n (x) rot^T``. A translation does not change a Hessian.
    """
    H = np.asarray(hessian, dtype=float)
    n3 = H.shape[0]
    if H.shape != (n3, n3) or n3 % 3:
        raise ValueError("rotate_hessian: the Hessian has the shape %s" % (H.shape,))
    B = np.kron(np.eye(n3 // 3), np.asarray(rot, dtype=float).T)
    return product(B, H, B.T)


def rigid_body_basis(coords: np.ndarray, tol: float = 1e-3) -> np.ndarray:
    """Orthonormal basis of the rigid-body displacements of a geometry (unit masses).

    Rows are flat Cartesian vectors of length 3 n. There are 6 rows for a
    nonlinear molecule and 5 for a linear one (3 for a single atom).

    A molecule counts as linear when its atoms lie within about ``tol`` times its size of a line. The
    tolerance must be far above the accuracy of a geometry optimization: an optimized linear molecule is
    off its axis by about 1e-7 angstrom, and with a tolerance below that the rotation about the axis would
    be counted as a sixth rigid motion. That vector is not a rigid motion: it is one of the two bends.
    """
    X = np.asarray(coords, dtype=float).reshape(-1, 3)
    n = len(X)
    X0 = X - X.mean(axis=0)
    rows = []
    for axis in range(3):
        v = np.zeros((n, 3)); v[:, axis] = 1.0
        rows.append(v.reshape(-1))
    for axis in range(3):
        e = np.zeros(3); e[axis] = 1.0
        rows.append(np.cross(e, X0).reshape(-1))
    M = np.array(rows)
    _u, s, vt = np.linalg.svd(M, full_matrices=False)
    keep = s > tol * max(s[0], 1.0)
    return vt[keep]


def remove_rigid_motion(vectors: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """Project the rigid-body part out of each row of ``vectors`` (flat Cartesian rows)."""
    V = np.atleast_2d(np.asarray(vectors, dtype=float))
    B = rigid_body_basis(coords)
    return V - (V @ B.T) @ B


def random_displacement(coords: np.ndarray, rms: float, seed: int = 0) -> np.ndarray:
    """A random displacement of all atoms without rigid motion, as an (n, 3) array in angstrom.

    The Cartesian components are drawn from a normal distribution, the translation and rotation of the
    whole molecule are projected out, and the result is scaled so that the root mean square displacement
    of an atom is ``rms``. The same ``seed`` gives the same displacement."""
    X = np.asarray(coords, dtype=float).reshape(-1, 3)
    if not float(rms) > 0.0:
        return np.zeros_like(X)
    rng = np.random.default_rng(int(seed))
    v = remove_rigid_motion(rng.normal(size=X.size)[None, :], X)[0].reshape(-1, 3)
    size = float(np.sqrt(np.mean(np.sum(v * v, axis=1))))
    if not size > 0.0:
        raise ValueError("random_displacement: the molecule has no internal coordinate")
    return v * (float(rms) / size)


def n_internal(coords: np.ndarray) -> int:
    """Number of internal degrees of freedom: 3 n - 6, or 3 n - 5 for a linear molecule."""
    X = np.asarray(coords, dtype=float).reshape(-1, 3)
    return 3 * len(X) - len(rigid_body_basis(X))


# ----------------------------------------------------------------------------- bonds and internal coordinates
def bond_list(symbols: Sequence[str], coords: np.ndarray, scale: float = 1.25) -> List[Tuple[int, int]]:
    """Pairs (i, j), i < j, closer than ``scale`` times the sum of their covalent radii."""
    X = np.asarray(coords, dtype=float).reshape(-1, 3)
    out = []
    for i in range(len(X)):
        for j in range(i + 1, len(X)):
            limit = scale * (COVALENT_RADIUS[symbols[i]] + COVALENT_RADIUS[symbols[j]])
            if np.linalg.norm(X[i] - X[j]) < limit:
                out.append((i, j))
    return out


def distance(coords: np.ndarray, i: int, j: int) -> float:
    X = np.asarray(coords, dtype=float).reshape(-1, 3)
    return float(np.linalg.norm(X[i] - X[j]))


def angle(coords: np.ndarray, i: int, j: int, k: int) -> float:
    """Angle i-j-k in degrees (j is the middle atom)."""
    X = np.asarray(coords, dtype=float).reshape(-1, 3)
    a, b = X[i] - X[j], X[k] - X[j]
    c = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def angle_list(bonds: Sequence[Tuple[int, int]]) -> List[Tuple[int, int, int]]:
    """All angles i-j-k formed by two bonds that share atom j."""
    neighbours = {}
    for i, j in bonds:
        neighbours.setdefault(i, []).append(j)
        neighbours.setdefault(j, []).append(i)
    out = []
    for j, nb in sorted(neighbours.items()):
        nb = sorted(nb)
        for a in range(len(nb)):
            for b in range(a + 1, len(nb)):
                out.append((nb[a], j, nb[b]))
    return out


def valence_electrons(symbols: Sequence[str], charge: int = 0) -> int:
    """Valence electrons with ccECP pseudopotentials (for cost estimates only)."""
    return sum(CCECP_VALENCE[s] for s in symbols) - charge
