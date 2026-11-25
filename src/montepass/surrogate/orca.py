"""ORCA: input files, output readers and batch jobs for the surrogate path.

ORCA builds the surrogate path of a project. It optimizes the end points, finds the
path between them with the nudged elastic band method (NEB), optimizes the saddle
and computes Hessians. This module has three parts.

1. Input writers. Functions that return the text of an ORCA input file:
   :func:`optimization_input`, :func:`frequency_input`, :func:`neb_input`,
   :func:`single_point_input`.
2. Readers. Functions that read ORCA results: the output text (:func:`final_energy`,
   :func:`optimization_converged`, :func:`neb_path_converged`,
   :func:`neb_saddle_converged`), the path file (:func:`read_path`), the saddle file
   (:func:`read_saddle`) and the Hessian file (:func:`read_hess`).
3. Jobs. :class:`OrcaJob` runs one ORCA calculation in its own folder as a batch
   job. The job script writes marker files, so the state of a job is read from
   its folder (:class:`StepJob`).

Units. ORCA prints energies in hartree, xyz files in angstrom, and the Hessian
file in hartree/bohr^2 with coordinates in bohr. The readers return energies in
hartree, coordinates in angstrom and Hessians in kcal/mol/angstrom^2.

Tested with ORCA 5.0.3 and ORCA 6.1.0. Both versions read the input keys that this
module writes, print the same phrases, and use the same file names. Three
differences between the versions matter to a reader:

* ORCA 6.1.0 writes one more section into the Hessian file, and it writes the
  geometry of that file with the centre of mass at the origin (:func:`read_hess`).
* ORCA 6.1.0 adds the energy to the comment line of the saddle files of a path job.
* The Hessian matrix of ORCA 5.0.3 is not exactly symmetric in the file.

A geometry optimization that stops at its step limit still ends with the line of
normal termination, so a job counts as finished only if the convergence phrase of
its kind is in the output (:class:`OrcaJob`).
"""
from __future__ import annotations

import json
import os
import re
import shlex
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..geometry import format_xyz, n_internal, normalize_symbol, read_xyz, read_xyz_frames
from ..scheduler.base import FAILED, FINISHED_STATES, PENDING, RUNNING, TIMEOUT, JobTable, Scheduler
from ..subspace import hessian_to_kcal_per_A2
from ..units import BOHR_TO_ANGSTROM

# ----------------------------------------------------------------------------- input writers
DEFAULT_METHOD = "B3LYP"
DEFAULT_BASIS = "cc-pVTZ"
DEFAULT_MAXCORE = 3000          # memory per process in MB (ORCA key %maxcore)

NEB_KINDS = ("NEB-TS", "NEB-CI")

# Spelling of the keys of the %neb block. "end" names the xyz file of the second end point, "ts" the
# xyz file of the saddle guess, "images" the number of images between the end points, "preopt" the
# switch that optimizes both end points before the path search.
#   "orca5": the keys of the inputs written for ORCA 5.0.3.
#   "orca6": the keys printed in the ORCA 6.1 manual.
# ORCA 5.0.3 and ORCA 6.1.0 both read both sets (tested with the two programs). The spelling stays
# a setting, so that a later version that drops one of the sets needs no change of this module.
NEB_KEYS: Dict[str, Dict[str, str]] = {
    "orca5": {"end": "NEB_END_XYZFILE", "ts": "NEB_TS_XYZFILE", "images": "NImages", "preopt": "Preopt"},
    "orca6": {"end": "Product", "ts": "TS", "images": "NImages", "preopt": "PreOpt"},
}
DEFAULT_NEB_KEYS = "orca5"


def neb_keys(keys=None) -> Dict[str, str]:
    """Key spellings of the %neb block: a name of :data:`NEB_KEYS`, or a dictionary with the four
    entries ``end``, ``ts``, ``images``, ``preopt``. ``None`` gives the default set."""
    if keys is None:
        keys = DEFAULT_NEB_KEYS
    if isinstance(keys, str):
        if keys not in NEB_KEYS:
            raise ValueError("unknown set of NEB keys %r; known sets: %s" % (keys, ", ".join(sorted(NEB_KEYS))))
        return dict(NEB_KEYS[keys])
    out = dict(keys)
    missing = [k for k in ("end", "ts", "images", "preopt") if not out.get(k)]
    if missing:
        raise ValueError("NEB keys: missing entries %s" % ", ".join(missing))
    return out


def _file_name(name: str) -> str:
    """Check a file name that is written into an ORCA input."""
    text = str(name)
    if not text or any(ch in text for ch in ' \t\n"\'*'):
        raise ValueError("file name %r cannot be written into an ORCA input "
                         "(empty, or it holds a blank, a quote or a star)" % name)
    return text


def _first_lines(method: str, basis: str, keywords: Sequence[str], nprocs: int, maxcore: Optional[int]) -> List[str]:
    if int(nprocs) < 1:
        raise ValueError("nprocs must be at least 1, got %s" % nprocs)
    words = [w for w in [method, basis] + list(keywords) if w]
    out = ["! " + " ".join(words)]
    if maxcore:
        if int(maxcore) < 1:
            raise ValueError("maxcore must be a positive number of MB, got %s" % maxcore)
        out.append("%%maxcore %d" % int(maxcore))
    out.append("%%pal nprocs %d end" % int(nprocs))
    return out


def _geometry_lines(charge: int, multiplicity: int, symbols, coords, xyz_file) -> List[str]:
    if int(multiplicity) < 1:
        raise ValueError("multiplicity is 2S+1 and must be at least 1, got %s" % multiplicity)
    if xyz_file is not None:
        if symbols is not None or coords is not None:
            raise ValueError("give either an xyz file name or symbols with coordinates, not both")
        return ["* xyzfile %d %d %s" % (charge, multiplicity, _file_name(xyz_file))]
    if symbols is None or coords is None:
        raise ValueError("a geometry is needed: an xyz file name, or symbols with coordinates")
    syms = [normalize_symbol(s) for s in symbols]
    X = np.asarray(coords, dtype=float).reshape(-1, 3)
    if len(X) != len(syms):
        raise ValueError("%d symbols but %d coordinate rows" % (len(syms), len(X)))
    if not np.all(np.isfinite(X)):
        raise ValueError("the coordinates hold a value that is not a finite number")
    lines = ["* xyz %d %d" % (charge, multiplicity)]
    lines += ["  %-2s %18.10f %18.10f %18.10f" % (s, r[0], r[1], r[2]) for s, r in zip(syms, X)]
    lines.append("*")
    return lines


def _extra(extra_blocks: str) -> List[str]:
    return [ln.rstrip() for ln in extra_blocks.strip().splitlines()] if extra_blocks and extra_blocks.strip() else []


def single_point_input(symbols=None, coords=None, xyz_file=None, charge: int = 0, multiplicity: int = 1,
                       method: str = DEFAULT_METHOD, basis: str = DEFAULT_BASIS, nprocs: int = 1,
                       maxcore: Optional[int] = DEFAULT_MAXCORE, extra_keywords: Sequence[str] = (),
                       extra_blocks: str = "") -> str:
    """Input of one energy calculation at a fixed geometry.

    The geometry is either ``xyz_file`` (a file name, read by ORCA) or ``symbols`` with ``coords``
    (angstrom, written into the input). ``multiplicity`` is 2S+1. ``maxcore`` is the memory per
    process in MB. ``extra_keywords`` are added to the first line; ``extra_blocks`` is text placed
    before the geometry (for example a ``%scf ... end`` block).
    """
    lines = _first_lines(method, basis, ["TightSCF"] + list(extra_keywords), nprocs, maxcore)
    lines += _extra(extra_blocks)
    lines += _geometry_lines(charge, multiplicity, symbols, coords, xyz_file)
    return "\n".join(lines) + "\n"


def optimization_input(symbols=None, coords=None, xyz_file=None, charge: int = 0, multiplicity: int = 1,
                       method: str = DEFAULT_METHOD, basis: str = DEFAULT_BASIS, nprocs: int = 1,
                       maxcore: Optional[int] = DEFAULT_MAXCORE, saddle: bool = False, frequencies: bool = False,
                       tight: bool = True, calc_hess: Optional[bool] = None, recalc_hess: int = 0,
                       extra_keywords: Sequence[str] = (), extra_blocks: str = "") -> str:
    """Input of a geometry optimization.

    ``saddle=False`` searches a minimum (``TightOpt`` with ``tight=True``, else ``Opt``).
    ``saddle=True`` searches a first-order saddle point (``OptTS``). A saddle search starts from an
    exact Hessian (``Calc_Hess true``) unless ``calc_hess=False``; ``recalc_hess = n`` computes the
    exact Hessian again every n steps. ``frequencies=True`` adds a frequency calculation at the
    final geometry, which writes the Hessian file ``<name>.hess``.
    """
    words = []
    if saddle:
        words += ["TightSCF", "OptTS"] + (["TightOpt"] if tight else [])
    else:
        words += ["TightOpt" if tight else "Opt", "TightSCF"]
    if frequencies:
        words.append("Freq")
    lines = _first_lines(method, basis, words + list(extra_keywords), nprocs, maxcore)
    if calc_hess is None:
        calc_hess = bool(saddle)
    if int(recalc_hess) < 0:
        raise ValueError("recalc_hess must be 0 or a positive number of steps, got %s" % recalc_hess)
    if calc_hess or recalc_hess:
        lines.append("%geom")
        if calc_hess:
            lines.append("  Calc_Hess true")
        if recalc_hess:
            lines.append("  Recalc_Hess %d" % int(recalc_hess))
        lines.append("end")
    lines += _extra(extra_blocks)
    lines += _geometry_lines(charge, multiplicity, symbols, coords, xyz_file)
    return "\n".join(lines) + "\n"


def frequency_input(symbols=None, coords=None, xyz_file=None, charge: int = 0, multiplicity: int = 1,
                    method: str = DEFAULT_METHOD, basis: str = DEFAULT_BASIS, nprocs: int = 1,
                    maxcore: Optional[int] = DEFAULT_MAXCORE, numerical: bool = False,
                    extra_keywords: Sequence[str] = (), extra_blocks: str = "") -> str:
    """Input of a frequency calculation at a fixed geometry. It writes the Hessian file ``<name>.hess``.
    ``numerical=True`` asks for a Hessian by finite differences of gradients (``NumFreq``), for
    methods that have no analytic Hessian."""
    words = ["TightSCF", "NumFreq" if numerical else "Freq"]
    lines = _first_lines(method, basis, words + list(extra_keywords), nprocs, maxcore)
    lines += _extra(extra_blocks)
    lines += _geometry_lines(charge, multiplicity, symbols, coords, xyz_file)
    return "\n".join(lines) + "\n"


def neb_input(end_xyz_file: str, symbols=None, coords=None, xyz_file=None, ts_xyz_file: Optional[str] = None,
              images: int = 8, kind: str = "NEB-TS", preopt: bool = False, keys=None,
              charge: int = 0, multiplicity: int = 1, method: str = DEFAULT_METHOD, basis: str = DEFAULT_BASIS,
              nprocs: int = 1, maxcore: Optional[int] = DEFAULT_MAXCORE, extra_keywords: Sequence[str] = (),
              extra_blocks: str = "") -> str:
    """Input of a nudged elastic band calculation between two end points.

    The first end point is the geometry of the input (``xyz_file``, or ``symbols`` with ``coords``).
    The second end point is the file ``end_xyz_file``. ``ts_xyz_file`` is an optional guess of the
    saddle. Both end points and the guess must list the atoms in the same order.

    ``images`` is the number of images between the end points, so the path has ``images + 2``
    geometries. ``kind`` is ``NEB-TS`` (path with a climbing image, then a saddle optimization
    that starts from the climbing image) or ``NEB-CI`` (path with a climbing image only).
    ``preopt=True`` lets ORCA optimize both end points before the path search.
    ``keys`` selects the spelling of the block keys, see :func:`neb_keys`.
    """
    kind = kind.upper()
    if kind not in NEB_KINDS:
        raise ValueError("NEB kind %r; allowed: %s" % (kind, ", ".join(NEB_KINDS)))
    if int(images) < 1:
        raise ValueError("a path needs at least 1 image between the end points, got %s" % images)
    k = neb_keys(keys)
    lines = _first_lines(method, basis, ["TightSCF", kind] + list(extra_keywords), nprocs, maxcore)
    lines.append("%neb")
    lines.append('  %s "%s"' % (k["end"], _file_name(end_xyz_file)))
    if ts_xyz_file is not None:
        lines.append('  %s "%s"' % (k["ts"], _file_name(ts_xyz_file)))
    lines.append("  %s %d" % (k["images"], int(images)))
    lines.append("  %s %s" % (k["preopt"], "true" if preopt else "false"))
    lines.append("end")
    lines += _extra(extra_blocks)
    lines += _geometry_lines(charge, multiplicity, symbols, coords, xyz_file)
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------- readers: output text
NORMAL_END = "ORCA TERMINATED NORMALLY"
OPT_CONVERGED = "THE OPTIMIZATION HAS CONVERGED"
NEB_CONVERGED = "THE NEB OPTIMIZATION HAS CONVERGED"
TS_CONVERGED = "THE TS OPTIMIZATION HAS CONVERGED"

_ENERGY_RE = re.compile(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+(?:[eE][-+]?\d+)?)")
_VERSION_RE = re.compile(r"Program Version\s+(\d+(?:\.\d+)+)")
_PATH_ENERGY_RE = re.compile(r"\bE\s+(-?\d+\.\d+(?:[eE][-+]?\d+)?)\s*$")
_ERROR_RE = re.compile(r"error|aborting|did not converge|not converged|cannot|unknown identifier|invalid", re.IGNORECASE)


def read_output(path: str) -> str:
    """Text of an output file. A missing or empty file is an error."""
    if not os.path.exists(path):
        raise FileNotFoundError("ORCA output not found: %s" % path)
    with open(path, errors="replace") as fh:
        text = fh.read()
    if not text.strip():
        raise ValueError("ORCA output is empty: %s" % path)
    return text


def program_version(text: str) -> Optional[str]:
    """Version printed in the banner of an output, for example ``6.1.0``, or None."""
    m = _VERSION_RE.search(text)
    return m.group(1) if m else None


def terminated_normally(text: str) -> bool:
    """True if ORCA printed its line of normal termination."""
    return NORMAL_END in text


def final_energy(text: str) -> float:
    """The last ``FINAL SINGLE POINT ENERGY`` of an output, in hartree."""
    found = _ENERGY_RE.findall(text)
    if not found:
        raise ValueError("no line 'FINAL SINGLE POINT ENERGY' in the ORCA output")
    return float(found[-1])


def optimization_converged(text: str) -> bool:
    """True if a geometry optimization (minimum or saddle) reached its convergence criteria."""
    return OPT_CONVERGED in text


def neb_path_converged(text: str) -> bool:
    """True if the path search of an NEB job converged. This says nothing about the saddle search."""
    return NEB_CONVERGED in text


def neb_saddle_converged(text: str) -> bool:
    """True if the saddle optimization that follows the path search of an NEB-TS job converged."""
    return TS_CONVERGED in text


def error_lines(text: str, limit: int = 4) -> List[str]:
    """The last lines of an output that look like an error message (at most ``limit`` lines).

    Two kinds of lines are skipped: the copy of the input (``| 12> ...``) and the lines of the
    settings tables (``DIIS Error  TolErr  ....  5.000e-07``).
    """
    found = []
    for line in text.splitlines():
        s = line.strip()
        if s and _ERROR_RE.search(s) and not s.startswith("|") and "...." not in s:
            found.append(s[:200])
    return found[-limit:]


# ----------------------------------------------------------------------------- readers: geometry files
def read_geometry(path: str) -> Tuple[List[str], np.ndarray]:
    """Symbols and coordinates (angstrom) of a single-geometry xyz file written by ORCA, for example
    the optimized geometry ``<name>.xyz``."""
    if not os.path.exists(path):
        raise FileNotFoundError("ORCA geometry file not found: %s" % path)
    return read_xyz(path)


def read_path(path: str) -> Tuple[List[str], List[np.ndarray], np.ndarray]:
    """Read a path file ``<name>_MEP_trj.xyz``.

    Returns ``(symbols, frames, energies)``: the images in the order of the path, each of shape
    (n_atoms, 3) in angstrom, and their energies in hartree. ORCA writes the energy of an image at
    the end of its comment line (``... E -56.473099016534``). A frame without an energy is an error.

    The path has ``NImages + 2`` frames: the two end points and the images between them. ORCA moves
    and rotates the images (centre at the origin, smallest distance between neighbours), so the first
    and the last frame are the end points of the input in another position. Superpose the frames on
    a reference geometry before they are compared with geometries of other files.
    """
    if not os.path.exists(path):
        raise FileNotFoundError("ORCA path file not found: %s" % path)
    frames = read_xyz_frames(path)
    symbols = frames[0][0]
    coords, energies = [], []
    for i, (syms, X, comment) in enumerate(frames):
        if syms != symbols:
            raise ValueError("%s: frame %d lists other atoms than frame 0" % (path, i))
        m = _PATH_ENERGY_RE.search(comment)
        if not m:
            raise ValueError("%s: no energy in the comment line of frame %d: %r" % (path, i, comment))
        coords.append(X)
        energies.append(float(m.group(1)))
    return symbols, coords, np.array(energies)


def saddle_file(folder: str, name: str) -> Tuple[Optional[str], str]:
    """The file with the best saddle geometry of an NEB job, and what it holds.

    Returns ``(path, kind)``. ``kind`` is ``TS`` for the optimized saddle of an NEB-TS job
    (``<name>_NEB-TS_converged.xyz``), ``CI`` for the converged climbing image
    (``<name>_NEB-CI_converged.xyz``), or ``NONE`` with ``path = None`` if neither file exists.
    A climbing image is an estimate of the saddle, not an optimized saddle.
    """
    for kind, tail in (("TS", "_NEB-TS_converged.xyz"), ("CI", "_NEB-CI_converged.xyz")):
        p = os.path.join(folder, name + tail)
        if os.path.exists(p) and os.path.getsize(p) > 0:
            return p, kind
    return None, "NONE"


def read_saddle(folder: str, name: str) -> Tuple[List[str], np.ndarray, str]:
    """Saddle geometry of an NEB job: ``(symbols, coords, kind)`` with ``kind`` as in :func:`saddle_file`."""
    path, kind = saddle_file(folder, name)
    if path is None:
        raise FileNotFoundError("no saddle file of the NEB job %r in %s (looked for %s_NEB-TS_converged.xyz "
                                "and %s_NEB-CI_converged.xyz)" % (name, folder, name, name))
    symbols, coords = read_xyz(path)
    return symbols, coords, kind


# ----------------------------------------------------------------------------- readers: the Hessian file
@dataclass
class OrcaHessian:
    """Content of an ORCA Hessian file ``<name>.hess``."""

    symbols: List[str]
    coords: np.ndarray              # (n_atoms, 3), angstrom
    masses: np.ndarray              # (n_atoms,), atomic mass units, as used by ORCA
    hessian: np.ndarray             # (3 n, 3 n), kcal/mol/angstrom^2, symmetric, Cartesian, not mass-weighted
    frequencies: np.ndarray         # vibrational frequencies in 1/cm; an imaginary frequency is negative
    asymmetry: float                # largest |H_ij - H_ji| in the file, hartree/bohr^2
    path: str = ""

    def n_imaginary(self, tolerance: float = 0.0) -> int:
        return count_imaginary(self.frequencies, tolerance)

    def problem(self, kind: str, tolerance: float = 0.0) -> str:
        return stationary_point_problem(self.frequencies, kind, tolerance)

    def in_frame_of(self, coords: np.ndarray, tolerance: float = 1e-3) -> np.ndarray:
        """The Hessian (kcal/mol/angstrom^2) in the Cartesian frame of ``coords``.

        ``coords`` is the same geometry in the same atom order, possibly moved and rotated against the
        geometry of the Hessian file (ORCA 6.1.0 moves the centre of mass to the origin). The best
        rigid superposition of the two geometries gives the rotation, and the Hessian is rotated with
        it. A shift changes nothing. If the two geometries differ by more than ``tolerance``
        (root-mean-square distance in angstrom after the superposition) a ValueError is raised.
        """
        X = np.asarray(coords, dtype=float).reshape(-1, 3)
        if X.shape != self.coords.shape:
            raise ValueError("%d atoms given, the Hessian file %s has %d" % (len(X), self.path, len(self.coords)))
        P0 = self.coords - self.coords.mean(axis=0)
        Q0 = X - X.mean(axis=0)
        V, _s, Wt = np.linalg.svd(_matmul(P0.T, Q0))
        sign = float(np.sign(np.linalg.det(_matmul(V, Wt)))) or 1.0
        rot = _matmul(_matmul(V, np.diag([1.0, 1.0, sign])), Wt)     # row vectors: moved = P0 @ rot
        rmsd = float(np.sqrt(((_matmul(P0, rot) - Q0) ** 2).sum() / len(X)))
        if rmsd > tolerance:
            raise ValueError("the given geometry is not the geometry of the Hessian file %s: they differ by %.4f "
                             "angstrom (root mean square) after the best superposition" % (self.path, rmsd))
        T = np.kron(np.eye(len(X)), rot)                         # old displacement = T @ new displacement
        return _matmul(_matmul(T.T, self.hessian), T)


def _hess_sections(path: str) -> Dict[str, List[str]]:
    if not os.path.exists(path):
        raise FileNotFoundError("ORCA Hessian file not found: %s" % path)
    sections: Dict[str, List[str]] = {}
    current = None
    with open(path, errors="replace") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            s = line.strip()
            if s.startswith("$"):
                current = s.split()[0]
                sections[current] = []
            elif current is not None and s and not s.startswith("#"):
                sections[current].append(s)
    if not sections:
        raise ValueError("%s: no section found; this is not an ORCA Hessian file" % path)
    return sections


def _hess_matrix(lines: List[str], path: str, section: str) -> np.ndarray:
    """Read a square matrix written in blocks of at most 5 columns."""
    try:
        n = int(lines[0].split()[0])
    except (IndexError, ValueError):
        raise ValueError("%s: section %s does not start with its dimension" % (path, section))
    M = np.zeros((n, n))
    filled = np.zeros((n, n), dtype=bool)
    k = 1
    while not filled.all():
        if k >= len(lines):
            raise ValueError("%s: section %s ends before all %d x %d values are read" % (path, section, n, n))
        try:
            cols = [int(c) for c in lines[k].split()]
        except ValueError:
            raise ValueError("%s: section %s: expected a line of column numbers, found %r" % (path, section, lines[k]))
        if not cols or min(cols) < 0 or max(cols) >= n:
            raise ValueError("%s: section %s: bad column numbers %r" % (path, section, lines[k]))
        k += 1
        for _ in range(n):
            if k >= len(lines):
                raise ValueError("%s: section %s ends inside a block" % (path, section))
            parts = lines[k].split()
            k += 1
            try:
                row = int(parts[0])
                values = [float(v) for v in parts[1:]]
            except (IndexError, ValueError):
                raise ValueError("%s: section %s: cannot read the row %r" % (path, section, lines[k - 1]))
            if row < 0 or row >= n or len(values) != len(cols):
                raise ValueError("%s: section %s: row %r does not fit its block of %d columns"
                                 % (path, section, lines[k - 1], len(cols)))
            M[row, cols] = values
            filled[row, cols] = True
    return M


def read_hess(path: str) -> OrcaHessian:
    """Read an ORCA Hessian file.

    The file is made of sections that start with a line ``$name``. The sections are found by
    name, so the reader does not depend on their order: ORCA 6.1.0 writes one more section
    (``$multiplicity``) before ``$hessian`` than ORCA 5.0.3.

    * ``$hessian``: Cartesian second derivatives in hartree/bohr^2, in blocks of 5 columns. The
      matrix of ORCA 5.0.3 is symmetric only to about 1e-4 hartree/bohr^2, so it is symmetrized here.
    * ``$atoms``: element, mass (atomic mass units) and position in bohr of every atom. ORCA 5.0.3
      writes the positions of the input. ORCA 6.1.0 writes them with the centre of mass at the origin,
      so they are shifted against ``<name>.xyz`` (0.058 angstrom for ammonia; no rotation was seen).
      A shift does not change the Hessian. :meth:`OrcaHessian.in_frame_of` gives the Hessian in the
      frame of another copy of the geometry and checks that the two geometries agree.
    * ``$vibrational_frequencies``: 3 n values in 1/cm. The first 6 (5 for a linear molecule) are the
      translations and rotations and are zero; they are removed here.

    Index ``3 i + c`` of the matrix belongs to atom ``i`` (order of ``$atoms``) and Cartesian component ``c``.
    """
    sec = _hess_sections(path)
    for needed in ("$hessian", "$atoms", "$vibrational_frequencies"):
        if needed not in sec or not sec[needed]:
            raise ValueError("%s: section %s is missing or empty" % (path, needed))

    atoms = sec["$atoms"]
    try:
        n_atoms = int(atoms[0].split()[0])
        rows = [ln.split() for ln in atoms[1:1 + n_atoms]]
        symbols = [normalize_symbol(r[0]) for r in rows]
        masses = np.array([float(r[1]) for r in rows])
        coords = np.array([[float(v) for v in r[2:5]] for r in rows]) * BOHR_TO_ANGSTROM
    except (IndexError, ValueError) as exc:
        raise ValueError("%s: cannot read section $atoms: %s" % (path, exc))
    if len(rows) != n_atoms or coords.shape != (n_atoms, 3):
        raise ValueError("%s: section $atoms announces %d atoms but holds %d complete lines" % (path, n_atoms, len(rows)))

    raw = _hess_matrix(sec["$hessian"], path, "$hessian")
    if raw.shape[0] != 3 * n_atoms:
        raise ValueError("%s: the Hessian has dimension %d but the file lists %d atoms" % (path, raw.shape[0], n_atoms))
    asymmetry = float(np.abs(raw - raw.T).max())
    hessian = hessian_to_kcal_per_A2(0.5 * (raw + raw.T))

    freq_lines = sec["$vibrational_frequencies"]
    try:
        n_freq = int(freq_lines[0].split()[0])
        all_freq = np.array([float(ln.split()[1]) for ln in freq_lines[1:1 + n_freq]])
    except (IndexError, ValueError) as exc:
        raise ValueError("%s: cannot read section $vibrational_frequencies: %s" % (path, exc))
    if len(all_freq) != n_freq or n_freq != 3 * n_atoms:
        raise ValueError("%s: expected %d frequencies, found %d" % (path, 3 * n_atoms, len(all_freq)))
    n_rigid = 3 * n_atoms - n_internal(coords)
    if np.abs(all_freq[:n_rigid]).max() > 1e-3:
        raise ValueError("%s: the first %d entries of $vibrational_frequencies should be zero (translations and "
                         "rotations), found %s" % (path, n_rigid, all_freq[:n_rigid]))
    return OrcaHessian(symbols=symbols, coords=coords, masses=masses, hessian=hessian,
                       frequencies=all_freq[n_rigid:].copy(), asymmetry=asymmetry, path=path)


def count_imaginary(frequencies: Sequence[float], tolerance: float = 0.0) -> int:
    """Number of imaginary frequencies. ORCA writes an imaginary frequency as a negative number.
    A value is counted if it is below ``-tolerance`` (1/cm)."""
    f = np.asarray(frequencies, dtype=float)
    return int(np.sum(f < -abs(tolerance)))


def stationary_point_problem(frequencies: Sequence[float], kind: str, tolerance: float = 0.0) -> str:
    """Check the number of imaginary frequencies of a stationary point.

    ``kind`` is ``minimum`` (0 imaginary frequencies) or ``saddle`` (exactly 1). Returns an empty
    string if the count is right, else a sentence that says what was found.
    """
    expected = {"minimum": 0, "saddle": 1}
    if kind not in expected:
        raise ValueError("kind must be 'minimum' or 'saddle', got %r" % kind)
    f = np.asarray(frequencies, dtype=float)
    found = np.sort(f[f < -abs(tolerance)])
    if len(found) == expected[kind]:
        return ""
    values = ", ".join("%.1f" % v for v in found) if len(found) else "none"
    return ("a %s must have %d imaginary frequencies, found %d (1/cm: %s)"
            % (kind, expected[kind], len(found), values))


# 1 kcal/mol/angstrom^2/amu in s^-2, then the wavenumber: sqrt(value) / (2 pi c), c in cm/s
_KCAL_A2_AMU_TO_S2 = 4184.0 / 6.02214076e23 / 1e-20 / 1.66053906660e-27
_LIGHT_CM_S = 2.99792458e10


def _matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Matrix product of finite numbers. Some linear-algebra libraries raise floating-point warnings in a
    product whose result is correct; the warnings are switched off here and the result is checked instead."""
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        out = np.matmul(a, b)
    if not np.all(np.isfinite(out)):
        raise FloatingPointError("a matrix product gave a value that is not a finite number")
    return out


def harmonic_frequencies(hessian: np.ndarray, masses: Sequence[float], coords: np.ndarray) -> np.ndarray:
    """Harmonic vibrational frequencies (1/cm, ascending, imaginary ones negative) of a Cartesian Hessian.

    ``hessian`` in kcal/mol/angstrom^2, ``masses`` in atomic mass units (one per atom), ``coords`` in
    angstrom. Translations and rotations are projected out of the mass-weighted Hessian, so the result
    has 3 n - 6 values (3 n - 5 for a linear molecule). Used to check the units and the index order of
    a Hessian against the frequencies that the program printed.
    """
    X = np.asarray(coords, dtype=float).reshape(-1, 3)
    m = np.asarray(masses, dtype=float).reshape(-1)
    H = np.asarray(hessian, dtype=float)
    n = len(X)
    if len(m) != n or H.shape != (3 * n, 3 * n):
        raise ValueError("shapes do not fit: %d atoms, %d masses, Hessian %s" % (n, len(m), H.shape))
    if np.any(m <= 0):
        raise ValueError("every mass must be positive")
    w = np.repeat(np.sqrt(m), 3)
    Hmw = 0.5 * (H + H.T) / np.outer(w, w)
    # rigid-body vectors in mass-weighted coordinates
    centre = (m[:, None] * X).sum(axis=0) / m.sum()
    X0 = X - centre
    rows = []
    for axis in range(3):
        v = np.zeros((n, 3)); v[:, axis] = 1.0
        rows.append((v * np.sqrt(m)[:, None]).reshape(-1))
    for axis in range(3):
        e = np.zeros(3); e[axis] = 1.0
        rows.append((np.cross(e, X0) * np.sqrt(m)[:, None]).reshape(-1))
    _u, s, vt = np.linalg.svd(np.array(rows), full_matrices=False)
    B = vt[s > 1e-8 * max(s[0], 1.0)]
    P = np.eye(3 * n) - _matmul(B.T, B)
    lam = np.linalg.eigvalsh(_matmul(_matmul(P, Hmw), P))
    keep = np.sort(np.argsort(np.abs(lam))[len(B):])            # drop the projected rigid-body modes (zero)
    lam = lam[keep]
    return np.sign(lam) * np.sqrt(np.abs(lam) * _KCAL_A2_AMU_TO_S2) / (2.0 * np.pi * _LIGHT_CM_S)


# ----------------------------------------------------------------------------- job scripts with marker files
ABSENT = "ABSENT"            # never submitted
DONE = "DONE"                # the step finished and its results passed the checks
# PENDING, RUNNING and FAILED are the words of the scheduler module.


def time_seconds(text: str) -> int:
    """Seconds of a batch time limit: ``MM``, ``MM:SS``, ``HH:MM:SS``, ``D-HH``, ``D-HH:MM`` or ``D-HH:MM:SS``."""
    t = str(text).strip()
    try:
        days = 0
        if "-" in t:
            d, t = t.split("-", 1)
            days = int(d)
            parts = [int(p) for p in t.split(":")]
            parts += [0] * (3 - len(parts))
            h, m, s = parts[:3]
        else:
            parts = [int(p) for p in t.split(":")]
            if len(parts) == 1:
                h, m, s = 0, parts[0], 0
            elif len(parts) == 2:
                h, m, s = 0, parts[0], parts[1]
            elif len(parts) == 3:
                h, m, s = parts
            else:
                raise ValueError
    except ValueError:
        raise ValueError("cannot read the time limit %r" % text)
    return ((days * 24 + h) * 60 + m) * 60 + s


def memory_mb(text: str) -> int:
    """MB of a batch memory request such as ``32G``, ``32000M`` or ``32000`` (a bare number is MB)."""
    t = str(text).strip().upper().rstrip("B")
    factor = {"K": 1.0 / 1024.0, "M": 1.0, "G": 1024.0, "T": 1024.0 * 1024.0}
    try:
        if t and t[-1] in factor:
            return int(float(t[:-1]) * factor[t[-1]])
        return int(float(t))
    except ValueError:
        raise ValueError("cannot read the memory request %r" % text)


def step_limit(job_time: Optional[str]) -> Optional[int]:
    """Time limit of a step in seconds, for a job with the batch time limit ``job_time``.

    The step stops 3 % (at least 120 s) before the batch system would stop the job, so the script
    itself can still write the marker file that says why. ``None`` means no limit.
    """
    if not job_time:
        return None
    total = time_seconds(job_time)
    return max(60, total - max(120, int(0.03 * total)))


SCRIPT_PREAMBLE = """\
# ---- marker files: <step>.started, then <step>.done or <step>.failed (one line that says why)
current_step=""
stopped() {
    if [ -n "$current_step" ] && [ ! -e "$current_step.done" ] && [ ! -e "$current_step.failed" ]; then
        echo "the job was stopped by a signal before the step ended" > "$current_step.failed"
    fi
    exit 143
}
trap stopped TERM INT
LIMIT=""
"""


def step_block(step: str, command: str, limit_seconds: Optional[int] = None,
               before: Sequence[Tuple[str, str]] = (), after: Sequence[Tuple[str, str]] = (),
               clean: Sequence[str] = ()) -> str:
    """Shell text of one step of a job script.

    The step is skipped if ``<step>.done`` exists. Otherwise it writes ``<step>.started``, runs
    ``command`` (a shell command line) under the time limit, and writes ``<step>.done`` or
    ``<step>.failed``. ``before`` and ``after`` are lists of ``(test, reason)``: ``test`` is a shell
    command that must succeed before the command starts, or after it ended with exit code 0;
    ``reason`` is the line written into ``<step>.failed`` if the test fails. ``clean`` lists result
    files that are deleted before the command starts, so a file of an earlier attempt cannot pass a test.
    The script exits with code 1 after a failed step. Use after :data:`SCRIPT_PREAMBLE`.
    """
    if not re.match(r"^[A-Za-z0-9_]+$", step):
        raise ValueError("a step name may hold letters, digits and '_' only, got %r" % step)
    q = shlex.quote
    out = ["# ---- step %s" % step, "current_step=%s" % step, "if [ -e %s.done ]; then" % step,
           '    echo "step %s: %s.done exists, skipped"' % (step, step), "else",
           "    rm -f " + " ".join([step + ".failed"] + [q(c) for c in clean]),
           "    echo \"$(date '+%%Y-%%m-%%d %%H:%%M:%%S') job ${SLURM_JOB_ID:-none} on $(hostname)\" > %s.started" % step,
           '    reason=""']
    for test, reason in before:
        out.append('    if [ -z "$reason" ] && ! %s; then reason=%s; fi' % (test, q(reason)))
    out.append('    if [ -z "$reason" ]; then')
    if limit_seconds:
        out.append('        if command -v timeout > /dev/null 2>&1; then LIMIT="timeout -k 30 %d"; fi' % int(limit_seconds))
    out += ["        $LIMIT %s" % command, "        rc=$?", '        LIMIT=""',
            '        if [ "$rc" -eq 124 ]; then',
            '            reason="time limit of %d s reached"' % int(limit_seconds or 0),
            '        elif [ "$rc" -ne 0 ]; then', '            reason="exit code $rc"', "        fi", "    fi"]
    for test, reason in after:
        out.append('    if [ -z "$reason" ] && ! %s; then reason=%s; fi' % (test, q(reason)))
    out += ['    if [ -z "$reason" ]; then', "        date '+%%Y-%%m-%%d %%H:%%M:%%S' > %s.done" % step,
            '        echo "step %s: done"' % step, "    else", '        echo "$reason" > %s.failed' % step,
            '        echo "step %s: FAILED: $reason"' % step, "        exit 1", "    fi", "fi", 'current_step=""', ""]
    return "\n".join(out)


def write_text(path: str, text: str) -> None:
    """Write a text file atomically: write ``<path>.tmp``, then rename."""
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        fh.write(text)
    os.replace(tmp, path)


def write_json(path: str, data) -> None:
    write_text(path, json.dumps(data, indent=1) + "\n")


def _tail(path: str, lines: int = 3, width: int = 200) -> str:
    """The last non-empty lines of a text file joined by ' | ', or '' if the file is missing or empty."""
    if not os.path.exists(path):
        return ""
    with open(path, errors="replace") as fh:
        rows = [ln.strip()[:width] for ln in fh.read().splitlines() if ln.strip()]
    return " | ".join(rows[-lines:])


@dataclass
class JobStatus:
    state: str                      # ABSENT, PENDING, RUNNING, DONE or FAILED
    message: str = ""
    timed_out: bool = False         # FAILED because the step or the batch job reached its time limit


class StepJob:
    """One batch job that runs one step in its own folder.

    The job script is ``<folder>/job.sh`` and holds one step written by :func:`step_block`. The state
    is decided from the marker files of the step and from the result files (:meth:`verify`); the
    batch system is asked only while no final marker exists. Every submission is recorded in the job
    table under ``key``, and a job is submitted once: :meth:`submit` does nothing if the table
    already holds a job for the key.
    """

    SCRIPT = "job.sh"
    LOGS = ("job.out", "local_job.out")         # output of the job script (batch system, local scheduler)

    def __init__(self, folder: str, step: str, scheduler: Scheduler, job_table: JobTable, key: Optional[str] = None,
                 kind: str = "", job_name: Optional[str] = None):
        self.folder = os.path.abspath(folder)
        self.step = step
        self.scheduler = scheduler
        self.job_table = job_table
        base = os.path.dirname(os.path.abspath(job_table.path))
        self.key = key or os.path.relpath(self.folder, base).replace(os.sep, "/")
        self.kind = kind
        self.job_name = job_name or (kind + "_" + os.path.basename(self.folder) if kind else os.path.basename(self.folder))
        self.message = ""

    # ---- files
    def file(self, name: str) -> str:
        return os.path.join(self.folder, name)

    def _marker(self, ext: str) -> bool:
        return os.path.exists(self.file(self.step + ext))

    def verify(self) -> str:
        """Check the result files of a finished step. Returns '' or the reason why they are not usable."""
        return ""

    def explain(self) -> str:
        """More facts about a failure, read from the output files ('' if there are none)."""
        return ""

    # ---- submission
    def submitted(self) -> bool:
        return self.job_table.current(self.key) is not None

    def job_id(self) -> Optional[str]:
        entry = self.job_table.current(self.key)
        return entry["id"] if entry else None

    def submit(self, after: Optional[List[str]] = None) -> str:
        """Submit the job script once. A second call returns the id of the job that exists."""
        entry = self.job_table.current(self.key)
        if entry is not None:
            return entry["id"]
        script = self.file(self.SCRIPT)
        if not os.path.exists(script):
            raise FileNotFoundError("no job script %s: write the input files first" % script)
        job_id = self.scheduler.submit(script, self.folder, name=self.job_name, after=after)
        self.job_table.add(self.key, job_id, self.folder, name=self.job_name, kind=self.kind)
        return str(job_id)

    def resubmit(self, after: Optional[List[str]] = None) -> str:
        """Submit a failed job again. The step starts again; a finished step is never repeated.
        A job that waits or runs is not touched: that is an error here."""
        st = self.status()
        if st.state == DONE:
            return self.job_id() or ""
        if st.state in (PENDING, RUNNING):
            raise RuntimeError("job %s of %s is %s: it cannot be submitted again" % (self.job_id(), self.folder, st.state))
        for ext in (".failed", ".started"):
            p = self.file(self.step + ext)
            if os.path.exists(p):
                os.remove(p)
        script = self.file(self.SCRIPT)
        if not os.path.exists(script):
            raise FileNotFoundError("no job script %s: write the input files first" % script)
        job_id = self.scheduler.submit(script, self.folder, name=self.job_name, after=after)
        self.job_table.add(self.key, job_id, self.folder, name=self.job_name, kind=self.kind)
        return str(job_id)

    def cancel(self) -> None:
        job_id = self.job_id()
        if job_id is not None:
            self.scheduler.cancel(job_id)

    # ---- state
    def _from_markers(self) -> Optional[JobStatus]:
        if self._marker(".done"):
            problem = self.verify()
            if problem:
                return JobStatus(FAILED, "%s.done exists in %s, but %s" % (self.step, self.folder, problem))
            return JobStatus(DONE, "")
        if self._marker(".failed"):
            reason = _tail(self.file(self.step + ".failed"), lines=1) or "no reason was written"
            more = self.explain()
            return JobStatus(FAILED, "step %s failed in %s: %s%s" % (self.step, self.folder, reason,
                                                                      ("; " + more) if more else ""),
                             timed_out=reason.startswith("time limit"))
        return None

    def status(self) -> JobStatus:
        """State of the job with a message. The message of the last call is also kept in ``self.message``."""
        st = self._status()
        self.message = st.message
        return st

    def state(self) -> str:
        """ABSENT, PENDING, RUNNING, DONE or FAILED. ``self.message`` says why after FAILED."""
        return self.status().state

    def _status(self) -> JobStatus:
        st = self._from_markers()
        if st is not None:
            return st
        entry = self.job_table.current(self.key)
        if entry is None:
            if self._marker(".started"):
                return JobStatus(RUNNING, "step %s was started in %s by a job that is not in the job table"
                                 % (self.step, self.folder))
            return JobStatus(ABSENT, "not submitted")
        batch = self.scheduler.state(entry["id"])
        if batch in FINISHED_STATES:
            st = self._from_markers()            # the job may have ended between the two looks at the folder
            if st is not None:
                return st
            log = ""
            for name in self.LOGS:
                log = _tail(self.file(name))
                if log:
                    break
            where = "after the step started" if self._marker(".started") else "before the step started"
            return JobStatus(FAILED, "batch job %s ended in state %s %s and left no result in %s%s"
                             % (entry["id"], batch, where, self.folder, ("; last lines of its log: " + log) if log else ""),
                             timed_out=(batch == TIMEOUT))
        if batch == RUNNING:
            return JobStatus(RUNNING, "")
        if batch == PENDING:
            return JobStatus(PENDING, "")
        # the batch system has no record of the job (it may answer again later)
        return JobStatus(RUNNING if self._marker(".started") else PENDING,
                         "the batch system has no record of job %s" % entry["id"])


# ----------------------------------------------------------------------------- ORCA jobs
class OrcaJob(StepJob):
    """One ORCA calculation in its own folder, run as a batch job.

        job = OrcaJob(folder, machine, scheduler, job_table, name="opt")
        job.write_optimization(symbols, coords, frequencies=True)     # input, xyz file, job script
        job.submit()                                                    # once
        ...
        if job.state() == DONE:
            symbols, coords = job.geometry()
        elif job.state() == FAILED:
            print(job.message)

    ``name`` is the base name of the ORCA files (``<name>.inp``, ``<name>.out``, ``<name>.xyz`` ...).
    The number of processes is the value ``tasks`` of the kind ``orca`` in the machine profile unless
    ``nprocs`` is given; it is written into the input and into the batch request, which must agree.
    ``maxcore`` (MB per process) defaults to 75 % of the memory request of the profile divided by the
    number of processes. ORCA is started with the full path of its program, as a parallel run needs.

    All methods can be called again at any time, also by a new process: files are written once, the
    job is submitted once, and the state is read from the folder.
    """

    STEP = "orca"
    DESCRIPTION = "orca_job.json"

    def __init__(self, folder: str, machine, scheduler: Scheduler, job_table: JobTable, name: str = "orca",
                 key: Optional[str] = None, nprocs: Optional[int] = None, maxcore: Optional[int] = None,
                 time: Optional[str] = None, job_name: Optional[str] = None):
        super().__init__(folder, self.STEP, scheduler, job_table, key=key, kind="orca", job_name=job_name)
        self.machine = machine
        self.name = _file_name(name)
        profile = machine.job("orca")
        self.nprocs = int(nprocs or profile.get("tasks", "1"))
        self.time = time or profile.get("time")
        if maxcore is None:
            maxcore = max(100, int(0.75 * memory_mb(profile.get("mem", "8G")) / self.nprocs))
        self.maxcore = int(maxcore)
        self.require: List[str] = []           # phrases that the output must hold
        self.expect: List[str] = []            # result files that must exist and hold data
        self.calculation = ""                  # "optimization", "frequencies", "neb" or "custom"
        self._load()

    # ---- description of the calculation (kept in the folder, so a new process knows what the job is)
    def _load(self) -> None:
        p = self.file(self.DESCRIPTION)
        if not os.path.exists(p):
            return
        with open(p) as fh:
            d = json.load(fh)
        if d.get("name") != self.name:
            raise ValueError("%s holds the ORCA job %r, not %r" % (self.folder, d.get("name"), self.name))
        self.require = list(d.get("require", []))
        self.expect = list(d.get("expect", []))
        self.calculation = d.get("calculation", "")

    @property
    def input_file(self) -> str:
        return self.name + ".inp"

    @property
    def output_file(self) -> str:
        return self.name + ".out"

    def script_text(self) -> str:
        """Text of the job script."""
        q = shlex.quote
        out, err, inp = self.output_file, self.name + ".err", self.input_file
        head = self.machine.header("orca", self.job_name, self.folder, time=self.time, tasks=self.nprocs)
        program = self.machine.executable("orca")
        body = [SCRIPT_PREAMBLE.rstrip("\n"),
                "# ORCA must be started with its full path when it runs in parallel.",
                "ORCA=%s" % q(program),
                'case "$ORCA" in /*) ;; *) ORCA=$(command -v "$ORCA" || echo "$ORCA") ;; esac', ""]
        before = [('[ -x "$ORCA" ]', "the ORCA program was not found or cannot be executed: %s" % program),
                  ("[ -s %s ]" % q(inp), "the input file %s is missing or empty" % inp)]
        after = [("grep -q -F -- %s %s" % (q(NORMAL_END), q(out)), "no line '%s' in %s" % (NORMAL_END, out))]
        after += [("grep -q -F -- %s %s" % (q(p), q(out)), "no line '%s' in %s" % (p, out)) for p in self.require]
        after += [("[ -s %s ]" % q(f), "the result file %s is missing or empty" % f) for f in self.expect]
        command = '"$ORCA" %s > %s 2> %s' % (q(inp), q(out), q(err))
        body.append(step_block(self.STEP, command, step_limit(self.time), before=before, after=after,
                               clean=[out] + list(self.expect)))
        return head + "\n".join(body)

    def write(self, input_text: str, files: Optional[Dict[str, str]] = None, require: Sequence[str] = (),
              expect: Sequence[str] = (), calculation: str = "custom") -> None:
        """Write the input file, further files (``{file name: text}``, for example xyz files) and the job
        script. ``require`` lists phrases that the output must hold and ``expect`` result files that
        must exist before the job counts as DONE.

        A second call with the same input changes nothing. A call with another input is an error
        once the job was submitted.
        """
        os.makedirs(self.folder, exist_ok=True)
        inp = self.file(self.input_file)
        if os.path.exists(inp):
            with open(inp) as fh:
                old = fh.read()
            if old != input_text and self.submitted():
                raise ValueError("%s was submitted with another input; use a new folder for a new calculation" % inp)
        self.require, self.expect, self.calculation = list(require), list(expect), calculation
        for fname, text in (files or {}).items():
            write_text(self.file(_file_name(fname)), text)
        write_text(inp, input_text)
        write_json(self.file(self.DESCRIPTION), {"name": self.name, "calculation": calculation, "require": self.require,
                                                 "expect": self.expect, "nprocs": self.nprocs, "maxcore": self.maxcore,
                                                 "time": self.time})
        write_text(self.file(self.SCRIPT), self.script_text())

    # ---- the three calculations of the surrogate path
    def write_optimization(self, symbols: Sequence[str], coords: np.ndarray, charge: int = 0, multiplicity: int = 1,
                           method: str = DEFAULT_METHOD, basis: str = DEFAULT_BASIS, saddle: bool = False,
                           frequencies: bool = True, **options) -> None:
        """Geometry optimization of a minimum or of a saddle (``saddle=True``), by default followed by
        frequencies. Further ``options`` go to :func:`optimization_input`. Results: ``<name>.xyz``
        and, with frequencies, ``<name>.hess``."""
        start = self.name + "_start.xyz"
        text = optimization_input(xyz_file=start, charge=charge, multiplicity=multiplicity, method=method, basis=basis,
                                  nprocs=self.nprocs, maxcore=self.maxcore, saddle=saddle, frequencies=frequencies,
                                  **options)
        expect = [self.name + ".xyz"] + ([self.name + ".hess"] if frequencies else [])
        self.write(text, {start: format_xyz(symbols, coords, "start geometry")}, require=[OPT_CONVERGED],
                   expect=expect, calculation="optimization")

    def write_frequencies(self, symbols: Sequence[str], coords: np.ndarray, charge: int = 0, multiplicity: int = 1,
                          method: str = DEFAULT_METHOD, basis: str = DEFAULT_BASIS, **options) -> None:
        """Frequencies and Hessian at a fixed geometry. Result: ``<name>.hess``."""
        geom = self.name + "_geom.xyz"
        text = frequency_input(xyz_file=geom, charge=charge, multiplicity=multiplicity, method=method, basis=basis,
                               nprocs=self.nprocs, maxcore=self.maxcore, **options)
        self.write(text, {geom: format_xyz(symbols, coords, "geometry")}, expect=[self.name + ".hess"],
                   calculation="frequencies")

    def write_neb(self, symbols: Sequence[str], reactant: np.ndarray, product: np.ndarray,
                  saddle_guess: Optional[np.ndarray] = None, images: int = 8, kind: str = "NEB-TS",
                  preopt: bool = False, charge: int = 0, multiplicity: int = 1, method: str = DEFAULT_METHOD,
                  basis: str = DEFAULT_BASIS, keys=None, require_saddle: bool = True, **options) -> None:
        """Path between two end points (same atom order), with an optional saddle guess.

        With ``kind = "NEB-TS"`` and ``require_saddle=True`` the job is DONE only if the path search
        and the saddle optimization both converged. With ``require_saddle=False`` a converged path
        is enough; :meth:`neb_result` then says whether the saddle optimization converged too.
        ``keys`` defaults to the value ``neb_keys`` of the section ``[orca]`` of the machine profile,
        or to the default set of :func:`neb_keys`.
        """
        if keys is None:
            keys = self.machine.cp.get("orca", "neb_keys", fallback=DEFAULT_NEB_KEYS).strip() or DEFAULT_NEB_KEYS
        kind = kind.upper()
        files = {"reactant.xyz": format_xyz(symbols, reactant, "reactant"),
                 "product.xyz": format_xyz(symbols, product, "product")}
        ts_file = None
        if saddle_guess is not None:
            ts_file = "saddle_guess.xyz"
            files[ts_file] = format_xyz(symbols, saddle_guess, "saddle guess")
        text = neb_input("product.xyz", xyz_file="reactant.xyz", ts_xyz_file=ts_file, images=images, kind=kind,
                         preopt=preopt, keys=keys, charge=charge, multiplicity=multiplicity, method=method,
                         basis=basis, nprocs=self.nprocs, maxcore=self.maxcore, **options)
        require = [NEB_CONVERGED] + ([TS_CONVERGED] if (kind == "NEB-TS" and require_saddle) else [])
        self.write(text, files, require=require, expect=[self.name + "_MEP_trj.xyz"], calculation="neb")

    # ---- checks used by the state
    def verify(self) -> str:
        out = self.file(self.output_file)
        try:
            text = read_output(out)
        except (FileNotFoundError, ValueError) as exc:
            return str(exc)
        if not terminated_normally(text):
            return "%s has no line '%s'" % (out, NORMAL_END)
        for phrase in self.require:
            if phrase not in text:
                return "%s has no line '%s'" % (out, phrase)
        for name in self.expect:
            p = self.file(name)
            if not os.path.exists(p) or os.path.getsize(p) == 0:
                return "the result file %s is missing or empty" % p
        return ""

    def explain(self) -> str:
        out = self.file(self.output_file)
        try:
            text = read_output(out)
        except (FileNotFoundError, ValueError):
            return _tail(self.file(self.name + ".err"), lines=2)
        facts = []
        if self.calculation == "neb":
            facts.append("path search converged: %s" % ("yes" if neb_path_converged(text) else "no"))
            if "NEB-TS" in text:
                facts.append("saddle optimization converged: %s" % ("yes" if neb_saddle_converged(text) else "no"))
        errors = error_lines(text)
        if errors:
            facts.append("last error lines of %s: %s" % (self.output_file, " | ".join(errors)))
        return "; ".join(facts)

    # ---- results
    def output(self) -> str:
        return read_output(self.file(self.output_file))

    def energy(self) -> float:
        """Final energy in hartree."""
        return final_energy(self.output())

    def version(self) -> Optional[str]:
        return program_version(self.output())

    def geometry(self) -> Tuple[List[str], np.ndarray]:
        """Final geometry ``<name>.xyz`` of an optimization (angstrom)."""
        return read_geometry(self.file(self.name + ".xyz"))

    def hessian(self) -> OrcaHessian:
        return read_hess(self.file(self.name + ".hess"))

    def path(self) -> Tuple[List[str], List[np.ndarray], np.ndarray]:
        """Converged path of an NEB job: symbols, images (angstrom) and their energies (hartree)."""
        return read_path(self.file(self.name + "_MEP_trj.xyz"))

    def saddle(self) -> Tuple[List[str], np.ndarray, str]:
        return read_saddle(self.folder, self.name)

    def neb_result(self) -> Dict[str, object]:
        """The two separate results of an NEB job: ``path_converged`` and ``saddle_converged`` (both
        read from the output), and ``saddle_kind`` (``TS``, ``CI`` or ``NONE``, the best saddle file)."""
        text = self.output()
        return {"path_converged": neb_path_converged(text), "saddle_converged": neb_saddle_converged(text),
                "saddle_kind": saddle_file(self.folder, self.name)[1], "terminated_normally": terminated_normally(text)}
