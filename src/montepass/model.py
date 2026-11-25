"""Analytic model surfaces for tests and demonstrations (no quantum chemistry needed).

A :class:`ForceField` is a sum of Morse bonds, harmonic bends, and stretch-stretch
and stretch-bend coupling terms. It is a function of the Cartesian geometry, so it is
anharmonic and its directions couple, like a real molecule. Two force fields with
slightly different parameters play the two roles of the method:

* the *true* surface stands for the DMC energy (the mock engine adds statistical noise to it);
* the *surrogate* surface stands for the DFT energy. Its minimum is displaced from the true one.

Energies are in kcal/mol here (the mock engine converts to hartree), lengths in angstrom.

The module also holds small numerical tools that work on any energy function of the
geometry: gradient, Hessian, minimization, saddle refinement, and a nudged elastic band.
They are meant for molecules of a few atoms.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from .linalg import product

from .geometry import kabsch, remove_rigid_motion, rigid_body_basis

Energy = Callable[[np.ndarray], float]


# ----------------------------------------------------------------------------- force-field terms
@dataclass(frozen=True)
class Morse:
    """E = de * (1 - exp(-a (r - re)))^2 ; harmonic force constant k = 2 de a^2."""
    i: int
    j: int
    de: float       # kcal/mol
    a: float        # 1/angstrom
    re: float       # angstrom


@dataclass(frozen=True)
class Bend:
    """E = 0.5 k (theta - theta0)^2 + g3 (theta - theta0)^3 for the angle i-j-k (j in the middle), theta in radians."""
    i: int
    j: int
    k: int
    kth: float      # kcal/mol/rad^2
    th0: float      # degrees
    g3: float = 0.0  # kcal/mol/rad^3


@dataclass(frozen=True)
class StretchStretch:
    """E = kss (r_ab - re_ab) (r_cd - re_cd) for two bonds."""
    a: int
    b: int
    c: int
    d: int
    kss: float      # kcal/mol/angstrom^2
    re_ab: float
    re_cd: float


@dataclass(frozen=True)
class StretchBend:
    """E = ksb (r_ab - re) (theta_ijk - theta0), theta in radians."""
    a: int
    b: int
    re: float
    i: int
    j: int
    k: int
    th0: float      # degrees
    ksb: float      # kcal/mol/angstrom/rad


def _angle_rad(X: np.ndarray, i: int, j: int, k: int) -> float:
    u, v = X[i] - X[j], X[k] - X[j]
    c = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    return float(np.arccos(np.clip(c, -1.0, 1.0)))


@dataclass(frozen=True)
class ForceField:
    """A molecule-specific analytic energy surface (kcal/mol)."""

    symbols: Tuple[str, ...]
    terms: Tuple[object, ...]
    e0: float = 0.0                 # constant offset, kcal/mol

    def energy(self, coords: np.ndarray) -> float:
        X = np.asarray(coords, dtype=float).reshape(-1, 3)
        e = self.e0
        for t in self.terms:
            if isinstance(t, Morse):
                r = np.linalg.norm(X[t.i] - X[t.j])
                e += t.de * (1.0 - np.exp(-t.a * (r - t.re))) ** 2
            elif isinstance(t, Bend):
                d = _angle_rad(X, t.i, t.j, t.k) - np.radians(t.th0)
                e += 0.5 * t.kth * d * d + t.g3 * d ** 3
            elif isinstance(t, StretchStretch):
                e += t.kss * (np.linalg.norm(X[t.a] - X[t.b]) - t.re_ab) * (np.linalg.norm(X[t.c] - X[t.d]) - t.re_cd)
            elif isinstance(t, StretchBend):
                e += t.ksb * (np.linalg.norm(X[t.a] - X[t.b]) - t.re) * (_angle_rad(X, t.i, t.j, t.k) - np.radians(t.th0))
            else:
                raise TypeError("unknown force-field term %r" % (t,))
        return float(e)

    __call__ = energy

    def shifted(self, displacement: np.ndarray) -> "ShiftedSurface":
        """The same surface moved rigidly by ``displacement``: E'(x) = E(x - displacement)."""
        return ShiftedSurface(self, np.asarray(displacement, dtype=float).reshape(-1, 3))


@dataclass(frozen=True)
class ShiftedSurface:
    """E'(x) = E(x - shift). Models a surrogate with the right shape and the wrong minimum."""

    base: ForceField
    shift: np.ndarray

    @property
    def symbols(self):
        return self.base.symbols

    def energy(self, coords: np.ndarray) -> float:
        return self.base.energy(np.asarray(coords, dtype=float).reshape(-1, 3) - self.shift)

    __call__ = energy


@dataclass(frozen=True)
class RotatedSurface:
    """E'(x) = E(origin + A (x - centre)) with a matrix ``A`` (3N x 3N).

    With an orthogonal ``A`` that turns two directions into each other, this is a surrogate with the right
    curvatures along the wrong directions: its stationary point is at ``centre``, and its Hessian there is
    ``A^T H A`` with ``H`` the Hessian of the base surface at ``origin``."""

    base: ForceField
    origin: np.ndarray
    centre: np.ndarray
    matrix: np.ndarray

    @property
    def symbols(self):
        return self.base.symbols

    def energy(self, coords: np.ndarray) -> float:
        d = (np.asarray(coords, dtype=float).reshape(-1, 3) - self.centre).reshape(-1)
        return self.base.energy(self.origin + np.dot(self.matrix, d).reshape(-1, 3))

    __call__ = energy


@dataclass
class TurnedHessianSurface:
    """E'(x) = E(origin + d) + 0.5 d^T (A^T H A - H) d, with d = x - centre and H the Hessian of the base surface
    at ``origin``.

    The base surface moved rigidly to ``centre``, with only its harmonic part changed: the Hessian at the
    stationary point is ``A^T H A``. The anharmonic part is that of the base surface. With an orthogonal ``A``
    this is a surrogate with the right anharmonic shape and the right curvatures along the wrong directions.
    (:class:`RotatedSurface` turns the whole surface, so its anharmonic part is wrong too.)"""

    base: ForceField
    origin: np.ndarray
    centre: np.ndarray
    matrix: np.ndarray

    def __post_init__(self):
        H = hessian(self.base, np.asarray(self.origin, dtype=float))
        A = np.asarray(self.matrix, dtype=float)
        self._dH = product(A.T, H, A) - H

    @property
    def symbols(self):
        return self.base.symbols

    def energy(self, coords: np.ndarray) -> float:
        d = (np.asarray(coords, dtype=float).reshape(-1, 3) - self.centre).reshape(-1)
        return self.base.energy(self.origin + d.reshape(-1, 3)) + 0.5 * float(np.dot(d, np.dot(self._dH, d)))   # vectors: no matmul

    __call__ = energy


def plane_rotation(u: np.ndarray, v: np.ndarray, degrees: float) -> np.ndarray:
    """Orthogonal matrix that turns the unit vectors ``u`` and ``v`` (orthogonal to each other) into each
    other by an angle, and leaves every vector orthogonal to both unchanged."""
    u = np.asarray(u, dtype=float).reshape(-1); v = np.asarray(v, dtype=float).reshape(-1)
    c, s = np.cos(np.radians(degrees)), np.sin(np.radians(degrees))
    return np.eye(len(u)) + (c - 1.0) * (np.outer(u, u) + np.outer(v, v)) + s * (np.outer(v, u) - np.outer(u, v))


# ----------------------------------------------------------------------------- ready-made molecules
def water(true: bool = True) -> ForceField:
    """Model water. ``true=False`` gives the surrogate: O-H longer by 0.012 A, angle smaller by 1.5 degrees,
    force constants changed by -6 % (stretch) and +8 % (bend)."""
    if true:
        de, a, re, kth, th0, kss = 118.0, 2.250, 0.9584, 109.0, 104.48, -14.0
    else:
        de, a, re, kth, th0, kss = 118.0, 2.181, 0.9704, 117.7, 102.98, -10.0
    terms = (Morse(0, 1, de, a, re), Morse(0, 2, de, a, re), Bend(1, 0, 2, kth, th0, g3=-20.0),
             StretchStretch(0, 1, 0, 2, kss, re, re))
    return ForceField(symbols=("O", "H", "H"), terms=terms)


def water_geometry(r: float = 0.9584, theta: float = 104.48) -> np.ndarray:
    """Water in the yz plane, oxygen first, C2 axis along z."""
    h = np.radians(theta) / 2.0
    return np.array([[0.0, 0.0, 0.0], [0.0, r * np.sin(h), -r * np.cos(h)], [0.0, -r * np.sin(h), -r * np.cos(h)]])


def ammonia(true: bool = True) -> ForceField:
    """Model ammonia with an inversion barrier of about 5 to 6 kcal/mol (planar saddle).
    ``true=False`` gives the surrogate: N-H longer by 0.010 A, equilibrium H-N-H angle larger by 1.2 degrees
    (so the surrogate barrier is lower), softer bend."""
    if true:
        de, a, re, kth, th0, ksb = 110.0, 2.150, 1.0124, 72.0, 106.67, 37.0
    else:
        de, a, re, kth, th0, ksb = 110.0, 2.100, 1.0224, 66.0, 107.87, 30.0
    terms: List[object] = [Morse(0, k, de, a, re) for k in (1, 2, 3)]
    for i, k in ((1, 2), (1, 3), (2, 3)):
        terms.append(Bend(i, 0, k, kth, th0))
        terms.append(StretchBend(0, i, re, i, 0, k, th0, ksb))
        terms.append(StretchBend(0, k, re, i, 0, k, th0, ksb))
    return ForceField(symbols=("N", "H", "H", "H"), terms=tuple(terms))


def ammonia_geometry(r: float = 1.0124, theta: float = 106.67, up: bool = True) -> np.ndarray:
    """Pyramidal ammonia, nitrogen first, C3 axis along z. ``theta`` is the H-N-H angle in degrees.
    ``theta = 120`` gives the planar structure. ``up=False`` gives the inverted pyramid."""
    # for C3v: sin(beta) = (2 / sqrt(3)) sin(theta / 2), beta = angle between the C3 axis and an N-H bond
    s = 2.0 / np.sqrt(3.0) * np.sin(np.radians(theta) / 2.0)
    s = min(1.0, s)
    cosb = np.sqrt(max(0.0, 1.0 - s * s))
    z = -r * cosb if up else r * cosb
    X = [[0.0, 0.0, 0.0]]
    for k in range(3):
        phi = 2.0 * np.pi * k / 3.0
        X.append([r * s * np.cos(phi), r * s * np.sin(phi), z])
    return np.array(X)


# ----------------------------------------------------------------------------- numerical tools
def gradient(energy: Energy, coords: np.ndarray, h: float = 1e-4) -> np.ndarray:
    """Central-difference gradient, shape (n_atoms, 3), in energy units per angstrom."""
    X = np.asarray(coords, dtype=float).reshape(-1, 3)
    g = np.zeros_like(X)
    for idx in np.ndindex(X.shape):
        xp = X.copy(); xp[idx] += h
        xm = X.copy(); xm[idx] -= h
        g[idx] = (energy(xp) - energy(xm)) / (2.0 * h)
    return g


def hessian(energy: Energy, coords: np.ndarray, h: float = 2e-3) -> np.ndarray:
    """Central-difference Cartesian Hessian (3 n, 3 n) from energies, symmetrized."""
    X = np.asarray(coords, dtype=float).reshape(-1)
    n = X.size
    e0 = energy(X.reshape(-1, 3))
    H = np.zeros((n, n))
    for i in range(n):
        xp = X.copy(); xp[i] += h
        xm = X.copy(); xm[i] -= h
        ep, em = energy(xp.reshape(-1, 3)), energy(xm.reshape(-1, 3))
        H[i, i] = (ep - 2.0 * e0 + em) / h ** 2
        for j in range(i):
            xpp = X.copy(); xpp[i] += h; xpp[j] += h
            xpm = X.copy(); xpm[i] += h; xpm[j] -= h
            xmp = X.copy(); xmp[i] -= h; xmp[j] += h
            xmm = X.copy(); xmm[i] -= h; xmm[j] -= h
            v = (energy(xpp.reshape(-1, 3)) - energy(xpm.reshape(-1, 3))
                 - energy(xmp.reshape(-1, 3)) + energy(xmm.reshape(-1, 3))) / (4.0 * h * h)
            H[i, j] = H[j, i] = v
    return H


def minimize(energy: Energy, coords: np.ndarray, gtol: float = 1e-6, max_iter: int = 500) -> np.ndarray:
    """Local minimum of an energy function of the geometry (BFGS, numerical gradient)."""
    from scipy.optimize import minimize as _minimize

    X0 = np.asarray(coords, dtype=float).reshape(-1, 3)
    shape = X0.shape

    def f(v):
        return energy(v.reshape(shape))

    def g(v):
        return gradient(energy, v.reshape(shape)).reshape(-1)

    res = _minimize(f, X0.reshape(-1), jac=g, method="BFGS", options={"gtol": gtol, "maxiter": max_iter})
    return res.x.reshape(shape)


def internal_basis(coords: np.ndarray) -> np.ndarray:
    """Orthonormal basis (rows) of the displacements with no rigid-body part: 3 n - 6 rows (3 n - 5 if linear)."""
    B = rigid_body_basis(coords)
    _u, _s, vt = np.linalg.svd(B, full_matrices=True)
    return vt[len(B):]


def refine_stationary(energy: Energy, coords: np.ndarray, steps: int = 30, tol: float = 1e-7) -> np.ndarray:
    """Newton iterations in the internal coordinates: converges to the nearest stationary point,
    which is a saddle when the start is close to one."""
    X = np.asarray(coords, dtype=float).reshape(-1, 3).copy()
    for _ in range(steps):
        T = internal_basis(X)
        g = T @ gradient(energy, X).reshape(-1)
        if np.abs(g).max() < tol:
            break
        Hi = T @ hessian(energy, X) @ T.T
        w, V = np.linalg.eigh(0.5 * (Hi + Hi.T))
        keep = np.abs(w) > 1e-8 * np.abs(w).max()
        z = -(V[:, keep] * (1.0 / w[keep])) @ (V[:, keep].T @ g)
        step = z @ T
        norm = np.linalg.norm(step)
        if norm > 0.2:
            step *= 0.2 / norm
        X += step.reshape(-1, 3)
    return X


def interpolate_path(start: np.ndarray, end: np.ndarray, n_images: int, align: bool = True) -> List[np.ndarray]:
    """Straight Cartesian interpolation between two geometries (both ends included).
    With ``align=True`` the end geometry is first superposed on the start geometry, so that the
    path holds no overall translation or rotation."""
    a = np.asarray(start, dtype=float).reshape(-1, 3); b = np.asarray(end, dtype=float).reshape(-1, 3)
    if align:
        b = kabsch(b, a)[0]
    return [a + (b - a) * (k / (n_images - 1.0)) for k in range(n_images)]


def _band_forces(energy: Energy, path: List[np.ndarray], spring: float, climb: bool
                 ) -> Tuple[List[np.ndarray], np.ndarray, float]:
    """Nudged-elastic-band forces on the inner images (improved tangent, optional climbing image)."""
    n = len(path)
    E = np.array([energy(x) for x in path])
    top = int(np.argmax(E[1:-1])) + 1
    forces = [np.zeros_like(path[0]) for _ in range(n)]
    worst = 0.0
    for i in range(1, n - 1):
        g = gradient(energy, path[i])
        # neighbours are superposed on image i first, so that distances and tangents hold no rigid motion
        tp = kabsch(path[i + 1], path[i])[0] - path[i]
        tm = path[i] - kabsch(path[i - 1], path[i])[0]
        if E[i + 1] > E[i] > E[i - 1]:
            tau = tp
        elif E[i + 1] < E[i] < E[i - 1]:
            tau = tm
        else:
            dmax = max(abs(E[i + 1] - E[i]), abs(E[i - 1] - E[i]))
            dmin = min(abs(E[i + 1] - E[i]), abs(E[i - 1] - E[i]))
            tau = tp * dmax + tm * dmin if E[i + 1] > E[i - 1] else tp * dmin + tm * dmax
        tau = tau / np.linalg.norm(tau)
        g_par = float(np.sum(g * tau))
        if climb and i == top:
            f = -g + 2.0 * g_par * tau
        else:
            f = -(g - g_par * tau) + spring * (np.linalg.norm(tp) - np.linalg.norm(tm)) * tau
        f = remove_rigid_motion(f.reshape(1, -1), path[i]).reshape(-1, 3)
        forces[i] = f
        worst = max(worst, float(np.abs(f).max()))
    return forces, E, worst


def nudged_elastic_band(energy: Energy, images: Sequence[np.ndarray], spring: float = 100.0,
                        max_iter: int = 3000, ftol: float = 0.05, climb: bool = True,
                        dt_max: float = 0.01) -> Tuple[List[np.ndarray], np.ndarray]:
    """A small nudged elastic band (improved tangent, climbing image, FIRE optimizer).

    ``images`` holds the first guess; its two end points stay fixed. ``spring`` is in energy units
    per angstrom^2. ``ftol`` is the largest allowed force component in energy units per angstrom.
    The climbing image is switched on once the band has relaxed to ``10 * ftol``.
    Returns the relaxed images and their energies. Meant for molecules of a few atoms.
    """
    path = [np.asarray(x, dtype=float).reshape(-1, 3).copy() for x in images]
    n = len(path)
    if n < 3:
        raise ValueError("a band needs at least 3 images")
    vel = [np.zeros_like(path[0]) for _ in range(n)]
    dt, alpha, n_pos = dt_max / 5.0, 0.1, 0
    climbing = False
    for _it in range(max_iter):
        forces, _E, worst = _band_forces(energy, path, spring, climbing)
        if worst < ftol:
            if climb and not climbing:
                climbing = True
                continue
            break
        if climb and not climbing and worst < 10.0 * ftol:
            climbing = True
            vel = [np.zeros_like(path[0]) for _ in range(n)]
            dt, alpha, n_pos = dt_max / 5.0, 0.1, 0
            continue
        F = np.concatenate([f.reshape(-1) for f in forces]); V = np.concatenate([v.reshape(-1) for v in vel])
        power = float(F @ V)
        if power > 0.0:
            fn, vn = np.linalg.norm(F), np.linalg.norm(V)
            V = (1.0 - alpha) * V + alpha * vn * F / fn if fn > 0 else V
            n_pos += 1
            if n_pos > 5:
                dt = min(dt * 1.1, dt_max); alpha *= 0.99
        else:
            V = np.zeros_like(V); dt *= 0.5; alpha = 0.1; n_pos = 0
        V = V + dt * F
        step = dt * V
        biggest = np.abs(step).max()
        if biggest > 0.03:
            step *= 0.03 / biggest; V *= 0.03 / biggest
        m = path[0].size
        for i in range(1, n - 1):
            path[i] = path[i] + step[i * m:(i + 1) * m].reshape(-1, 3)
            vel[i] = V[i * m:(i + 1) * m].reshape(-1, 3)
    for i in range(1, n):            # remove rigid offsets between consecutive images
        path[i] = kabsch(path[i], path[i - 1])[0]
    return path, np.array([energy(x) for x in path])
