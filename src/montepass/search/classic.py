"""The classic line search: one polynomial fit per direction, a fixed number of iterations.

This is the surrogate-Hessian line search (J. Tiihonen, P. R. C. Kent and J. T. Krogel,
J. Chem. Phys. 156, 054104 (2022)) as the first applications used it. It is kept as a
reference method, so its rules are reproduced and not improved:

1. **Surrogate scan.** For every search direction j the surrogate energy is computed at 17
   points between ``-R_scan`` and ``+R_scan``. The half-width comes from the window rule
   ``W_max = 0.5 * sqrt(|Lambda_j|)``, in which ``Lambda_j`` is the surrogate force constant
   in hartree/A^2 and the number ``W_max`` is read as an energy in hartree. With
   ``W = 0.5 * |k| * R**2`` this gives ``R_scan = (|k_j| / 627.509474)**-0.25`` angstrom for a
   force constant ``k_j`` in kcal/mol/A^2 (1.33 A for 200 kcal/mol/A^2, 0.71 A for 2500).
2. **Window and noise ("the sizer").** A direction is searched with a line of 7 points between
   ``-R_j`` and ``+R_j``, ``R_j = sqrt(2 W_j / |k_j|)``. The sizer chooses the window ``W_j``
   and the noise ``sigma_j`` of one point: on a mesh of (W, sigma) pairs it adds Gaussian
   noise to the surrogate curve, fits, and measures the error of the fitted stationary point.
   It takes the largest ``sigma_j`` for which that error is below the tolerance ``epsilon_j``.
   See :func:`size_direction`.
3. **Iterations.** Every iteration computes the DMC energy of the shared centre and of 6 more
   points per direction, with enough blocks to reach ``sigma_j``. An unweighted cubic is
   fitted to the 7 energies of a direction. The step of the direction is the stationary point
   of the cubic (a minimum, or a maximum along a direction of negative surrogate curvature).
   All directions are searched from the same centre and all steps are applied together.
   Windows, noises and directions stay the same in all iterations.
4. **End.** After a fixed number of iterations (4 by default) the search stops. Full-length
   DMC runs at the final geometry and at the starting geometry give the energy change.

Three things are added to the old method. None of them changes a step.

* A bracket check. For every fitted step the driver records whether the cubic had a
  stationary point of the wanted kind and whether that point lies inside the grid. The old
  method accepts a point outside the grid without a check, and it takes an end of the grid
  when the cubic has no such point.
* The stability audit of every iteration (module ``audit``).
* Error bars of one standard error. The old method reports the 95 % half-width, which is
  about twice as large. Both numbers are stored (``sm`` and ``sm95``).

Conventions: displacements ``t`` in angstrom along unit-norm Cartesian directions, energies of
programs in hartree, fit energies in kcal/mol, force constants ``k`` in kcal/mol/A^2 with
``E = 0.5 * k * t**2``. Windows ``W`` and noises ``sigma`` are in kcal/mol.

Names of the points: ``<name>/it<k>/centre`` and ``<name>/it<k>/d<j>_p0352`` (direction j,
``p`` = positive side, ``m`` = negative side, the four digits are the displacement in units
of 0.0001 A, here +0.0352 A).
"""
from __future__ import annotations

import json
import os
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.interpolate import CubicSpline

from ..engine.base import DONE, FAILED, STOPPED, Engine, PointSpec
from ..stats import reblock
from ..surrogate.base import Surrogate
from ..units import HARTREE_TO_KCAL
from .dense import NEEDS_ATTENTION, RUNNING_STATUS
from .dense_v2 import FINALIZING

DONE_STATUS = "DONE"             # the fixed number of iterations is finished (this is not a convergence statement)
UNRESOLVED_STATUS = "UNRESOLVED"  # option stop_if_unresolved: the first iteration failed its audit or its bracket check

# Bracket check of one fitted step
INSIDE = "INSIDE"                # the cubic has a stationary point of the wanted kind, inside the grid
OUTSIDE = "OUTSIDE"              # it has one, but outside the grid: the step is an extrapolation
NO_ROOT = "NO_ROOT"              # it has none: the step is the end of the grid with the lower (higher) fitted energy

# Smallest window and smallest half-width that the sizer uses: 0.0001 hartree and 0.0001 angstrom
WINDOW_FLOOR = 1.0e-4 * HARTREE_TO_KCAL
HALF_WIDTH_FLOOR = 1.0e-4
# The surrogate curve is read at 0.99999 times the wanted displacement, so that rounding can
# never put the last point of a line outside the scanned range.
_EDGE_SHRINK = 0.99999
# Half-width (angstrom) of the line that defines the surrogate stationary point of a direction
_REFERENCE_HALF_WIDTH = 0.02 ** 0.5
# A point needs this many blocks after the discarded ones before its error bar can be computed
_MIN_USABLE_BLOCKS = 8
# A fitted stationary point farther from the centre than this many grid half-widths is a rounding artefact
_FAR_AWAY = 1.0e8

DEFAULTS: Dict[str, object] = {
    "iterations": 4,                 # number of line-search iterations (fixed)
    "warmup": 2,                     # the reported mean leaves out the start and the first (warmup - 1) later centres
    "points": 7,                     # points of one DMC line; the middle one is the shared centre
    "scan_points": 17,               # points of the surrogate scan of one direction
    "window_frac": 0.5,              # scan rule: W_max = window_frac * sqrt(|Lambda|), Lambda in hartree/A^2, W_max in hartree
    "scan_half_width_max": None,     # optional upper limit of the scan half-width in angstrom (the old method has none)
    "epsilon": None,                 # tolerance in angstrom: None = rule below, one number, or one number per direction
    "epsilon_energy": 0.1,           # kcal/mol; rule epsilon_j = sqrt(epsilon_energy / |k_j|)
    "windows": None,                 # kcal/mol, one per direction; with "noises" they replace the scan and the sizer
    "noises": None,                  # kcal/mol, one per direction
    "window_scales": [1, 4, 16, 64],  # the sizer runs on meshes up to W_max / scale and keeps the largest noise
    "mesh": 25,                      # mesh points for the window and for the noise
    "noise_frac": 0.05,              # largest noise of a mesh as a fraction of its largest window
    "resamples": 500,                # noise draws of the sizer
    "fit_resamples": 1000,           # noise draws for the error bar of a fitted step
    "seed": 0,                       # seed of all random numbers of the search (not of the QMC runs)
    "drop": 50,                      # start-up blocks discarded from every DMC trace
    "min_blocks": 30,                # lower limit of the usable blocks of a point
    "max_blocks": 400,               # upper limit of the blocks of a line-search point (discarded blocks included)
    "final_blocks": 400,             # blocks of a full-length run at the starting and at the final geometry
    "final_series": 2,               # independent full-length runs at each of these two geometries
    "noise_blocks": 100,             # usable blocks used to measure the block noise when settings hold no sigma_block
    "audit_floor": 0.01,             # angstrom; the audit threshold is max(2 epsilon_j, audit_floor)
    "min_distance": 0.6,             # angstrom; a requested geometry with two atoms closer than this is reported
    "stop_if_unresolved": False,     # True: stop after the first iteration if its audit or its bracket check fails
}


class SizerError(ValueError):
    """No window and noise on the mesh meet the tolerance of a direction."""


# ============================================================================= small rules
def scan_half_width(k: float, window_frac: float = 0.5) -> float:
    """Half-width (angstrom) of the surrogate scan of a direction with force constant ``k``
    (kcal/mol/A^2; the sign is ignored).

    The rule is ``W_max = window_frac * sqrt(|Lambda|)``. It mixes units: ``Lambda`` is the force
    constant in hartree/A^2 and ``W_max`` is read as hartree. So the force constant is converted to
    hartree/A^2 first. The result is ``sqrt(2 * window_frac) * |Lambda|**-0.25``.
    """
    lam = abs(float(k)) / HARTREE_TO_KCAL
    if not (lam > 0.0 and np.isfinite(lam)):
        raise ValueError("the force constant of a direction must be finite and not zero (found %r)" % (k,))
    return float(np.sqrt(2.0 * window_frac) * lam ** -0.25)


def window_half_width(window: float, k: float) -> float:
    """Half-width ``R = sqrt(2 W / |k|)`` (angstrom) of a line with window ``W`` (kcal/mol): the displacement
    at which a harmonic curve of force constant ``|k|`` (kcal/mol/A^2) has risen by ``W``."""
    return float(max(np.sqrt(2.0 * max(float(window), WINDOW_FLOOR) / abs(float(k))), HALF_WIDTH_FLOOR))


def default_tolerance(k: float, energy: float = 0.1) -> float:
    """Tolerance ``epsilon = sqrt(energy / |k|)`` (angstrom) of a direction with force constant ``k``
    (kcal/mol/A^2), ``energy`` in kcal/mol.

    This is the rule ``|Lambda| * epsilon**2 = 0.1 kcal/mol`` of an earlier application. ``k`` is the full second
    derivative, so a displacement of ``epsilon`` raises the harmonic energy by ``energy / 2``: 0.05 kcal/mol
    for the default, not 0.1."""
    return float(np.sqrt(float(energy) / abs(float(k))))


def blocks_for_noise(sigma_block: float, sigma: float, drop: int, min_blocks: int, max_blocks: int) -> Tuple[int, bool]:
    """Blocks to request for a point that should reach the noise ``sigma`` (kcal/mol), when one block has the
    noise ``sigma_block``: ``(sigma_block / sigma)**2`` usable blocks, at least ``min_blocks``, plus the ``drop``
    discarded blocks, at most ``max_blocks`` in total. The second value is True when the upper limit was
    applied, that is when the point will not reach ``sigma``."""
    if sigma > 0.0:
        need = (float(sigma_block) / float(sigma)) ** 2
        usable = int(max(int(min_blocks), int(np.ceil(min(need, 1.0e9) - 1.0e-9))))      # 16.0000000001 counts as 16
    else:
        usable = int(max_blocks)
    total = usable + int(drop)
    if total > int(max_blocks):
        return int(max_blocks), True
    return total, False


def min_distance(coords: np.ndarray) -> float:
    """Smallest distance between two atoms of a geometry (angstrom). Infinite for one atom."""
    X = np.asarray(coords, dtype=float).reshape(-1, 3)
    if len(X) < 2:
        return float("inf")
    d = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
    return float(d[np.triu_indices(len(X), 1)].min())


# ============================================================================= polynomial fit and the root rule
def fit_polynomial(t: Sequence[float], e: Sequence[float], order: int,
                   sigma: Optional[Sequence[float]] = None) -> np.ndarray:
    """Least-squares polynomial of the given order through the points ``(t, e)``.

    Returns the coefficients ``c[0] + c[1] t + ... + c[order] t**order``, lowest power first.
    With ``sigma`` (one error bar per point) the points have the weights ``1 / sigma**2``; without it
    all points count the same. The fit is done in the variable ``t / max|t|`` so that it stays well
    conditioned for small displacements.
    """
    t = np.asarray(t, dtype=float); e = np.asarray(e, dtype=float)
    if t.ndim != 1 or t.shape != e.shape:
        raise ValueError("t and e must be one-dimensional and have the same length")
    if len(t) < order + 1:
        raise ValueError("a polynomial of order %d needs at least %d points, got %d" % (order, order + 1, len(t)))
    scale = float(np.abs(t).max())
    if not scale > 0.0:
        raise ValueError("all displacements are zero: no polynomial can be fitted")
    V = np.vander(t / scale, order + 1, increasing=True)
    offset = float(e.mean())                     # total energies are large numbers; fit the differences
    rhs = e - offset
    if sigma is not None:
        s = np.asarray(sigma, dtype=float)
        if s.shape != t.shape or not np.all(np.isfinite(s)) or np.any(s <= 0.0):
            raise ValueError("every error bar of a weighted fit must be finite and larger than zero")
        V = V / s[:, None]; rhs = rhs / s
    a, _res, rank, _sv = np.linalg.lstsq(V, rhs, rcond=None)
    if rank < order + 1:
        raise ValueError("the points do not determine a polynomial of order %d (rank %d)" % (order, rank))
    c = a / scale ** np.arange(order + 1)
    c[0] += offset
    return c


def _polyval(c: np.ndarray, x) -> np.ndarray:
    """Value of polynomials with coefficients ``c[..., p]`` (lowest power first) at ``x``."""
    x = np.asarray(x, dtype=float)
    out = np.zeros(np.broadcast(c[..., 0], x).shape)
    for p in range(c.shape[-1] - 1, -1, -1):
        out = out * x + c[..., p]
    return out


def stationary_point(c, sign: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Stationary point of the wanted kind of a quadratic or cubic polynomial.

    ``c[..., p]`` are the coefficients, lowest power first (3 or 4 of them); any leading axes are kept.
    ``sign = +1`` asks for a minimum (second derivative positive), ``sign = -1`` for a maximum. A cubic
    has at most one of each, so no choice between roots is needed. Returns ``(x, found)``. Where
    ``found`` is False no such point exists and ``x`` holds no meaning.
    """
    c = np.asarray(c, dtype=float)
    n = c.shape[-1]
    if n not in (3, 4):
        raise ValueError("only quadratic and cubic polynomials are supported (got %d coefficients)" % n)
    c1, c2 = c[..., 1], c[..., 2]
    c3 = c[..., 3] if n == 4 else np.zeros_like(c1)
    # derivative c1 + 2 c2 x + 3 c3 x^2; its roots are (-c2 +- s) / (3 c3) with s^2 = c2^2 - 3 c1 c3,
    # and the second derivative at them is +- 2 s. The wanted root is written in two equal forms and
    # each form is used where it loses no digits.
    disc = c2 * c2 - 3.0 * c1 * c3
    s = np.sqrt(np.where(disc > 0.0, disc, 0.0))
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        x_a = -c1 / (c2 + sign * s)
        x_b = (-c2 + sign * s) / (3.0 * c3)
    x = np.where(sign * c2 > 0.0, x_a, x_b)
    found = (disc > 0.0) & np.isfinite(x)
    return np.where(found, x, 0.0), found


def _library_step(c: np.ndarray, sign: int, lo, hi) -> Tuple[np.ndarray, np.ndarray]:
    """The step rule of the old method for cubics ``c[..., 4]``: the stationary point of the wanted kind,
    wherever it lies; if there is none, the end of the grid (``lo`` or ``hi``) with the lower fitted energy
    (the higher one for ``sign = -1``; ``lo`` if both are equal).

    One case is treated differently from the old code: a stationary point more than 1e8 grid lengths
    away. It appears when the energies are an exact parabola of the wrong curvature and the cubic
    coefficient is a rounding error. Such a point counts as absent."""
    x, found = stationary_point(c, sign)
    found = found & (np.abs(x) <= _FAR_AWAY * max(abs(lo), abs(hi)))
    e_lo, e_hi = _polyval(c, lo), _polyval(c, hi)
    end = np.where(sign * e_lo <= sign * e_hi, lo, hi)
    return np.where(found, x, end), found


def _fraction_error(samples: np.ndarray, fraction: float = 0.025) -> np.ndarray:
    """Error bar of the old method from a set of resampled values (last axis): the larger of the two
    distances from the median to the ``fraction`` and ``1 - fraction`` quantiles. With ``fraction = 0.025``
    this is a 95 % half-width, about 1.96 standard errors for a normal distribution."""
    x = np.sort(np.asarray(samples, dtype=float), axis=-1)
    n = x.shape[-1]
    med = np.median(x, axis=-1)
    lo = np.abs(x[..., int((n - 1) * fraction)] - med)
    hi = np.abs(x[..., int((n - 1) * (1 - fraction))] - med)
    return np.maximum(lo, hi)


def _standard_error(samples: np.ndarray) -> float:
    """One standard error from resampled values: half the width of their central 68.27 % interval.
    For a normal distribution this is the standard deviation. A few resampled fits without a stationary
    point (they jump to an end of the grid) do not inflate it."""
    lo, hi = np.percentile(np.asarray(samples, dtype=float), [15.8655, 84.1345])
    return float(0.5 * (hi - lo))


def fit_step(t: Sequence[float], e: Sequence[float], sign: int = 1, errors: Optional[Sequence[float]] = None,
             resamples: int = 1000, rng: Optional[np.random.Generator] = None) -> Dict[str, object]:
    """Fit of one line of the classic search, with the bracket check and the error bars.

    ``t`` displacements (angstrom), ``e`` energies (kcal/mol, any zero), ``errors`` their standard errors.
    The fit is an unweighted cubic. The step is its stationary point of the wanted kind (``sign = +1``
    minimum, ``-1`` maximum), also when that point is outside the grid. If the cubic has no such point,
    the step is the end of the grid with the lower fitted energy (higher for ``sign = -1``).

    Returns a dict:
      ``m``            the step (angstrom)
      ``status``       ``INSIDE``, ``OUTSIDE`` or ``NO_ROOT`` (the bracket check)
      ``root``         True if the cubic has a stationary point of the wanted kind
      ``inside``       True if that point lies inside the grid
      ``R``            largest displacement of the grid
      ``coefficients`` of the cubic, lowest power first (kcal/mol and angstrom)
      ``slope``        fitted slope at t = 0 (kcal/mol/A)
      ``k``            fitted second derivative at the step (kcal/mol/A^2)
      ``gain``         fitted energy at the step minus fitted energy at t = 0 (kcal/mol)
      ``sm``           1 standard error of the step from resampling (0 without ``errors``)
      ``sm95``         95 % half-width of the step, the "+/-" of the old method
      ``inside_fraction``  fraction of the resampled fits whose stationary point is inside the grid
    """
    t = np.asarray(t, dtype=float); e = np.asarray(e, dtype=float)
    if len(t) < 4:
        raise ValueError("a cubic fit needs at least 4 points, got %d" % len(t))
    if not (np.all(np.isfinite(t)) and np.all(np.isfinite(e))):
        raise ValueError("displacements and energies of a fit must be finite numbers")
    scale = float(np.abs(t).max())
    if not scale > 0.0:
        raise ValueError("all displacements are zero: no line can be fitted")
    u = t / scale
    lo, hi = float(u.min()), float(u.max())
    P = np.linalg.pinv(np.vander(u, 4, increasing=True))          # coefficients = P . energies (unweighted least squares)
    offset = float(e.mean())                                      # total energies are large numbers; fit the differences
    a = np.dot(P, e - offset)
    x, found = _library_step(a, sign, lo, hi)
    x = float(x); found = bool(found)
    inside = bool(found and lo <= x <= hi)
    c = a / scale ** np.arange(4)
    m = x * scale
    out: Dict[str, object] = {
        "m": float(m), "status": INSIDE if inside else (OUTSIDE if found else NO_ROOT), "root": found, "inside": inside,
        "R": scale, "coefficients": [float(c[0] + offset)] + [float(v) for v in c[1:]], "slope": float(c[1]),
        "k": float(2.0 * c[2] + 6.0 * c[3] * m), "gain": float(_polyval(c, m) - c[0]),
        "sm": 0.0, "sm95": 0.0, "inside_fraction": 1.0 if inside else 0.0,
    }
    if errors is not None and resamples > 1:
        s = np.asarray(errors, dtype=float)
        if s.shape != t.shape or not np.all(np.isfinite(s)) or np.any(s < 0.0):
            raise ValueError("error bars of a fit must be finite and not negative, one per point")
        if np.any(s > 0.0):
            rng = np.random.default_rng(0) if rng is None else rng
            # np.dot and not the @ operator: for arrays of this size @ can report floating-point
            # warnings that are not real on some machines, and tests treat warnings as errors
            A = np.dot(e - offset + s * rng.standard_normal((int(resamples), len(t))), P.T)
            xs, fs = _library_step(A, sign, lo, hi)
            out["sm"] = _standard_error(xs) * scale
            out["sm95"] = float(_fraction_error(xs)) * scale
            out["inside_fraction"] = float(np.mean(fs & (xs >= lo) & (xs <= hi)))
    return out


# ============================================================================= the sizer
class _ErrorMesh:
    """Error of the fitted stationary point on a mesh of windows (columns) and noises (rows).

    error(W, sigma) = bias(W) + errorbar(W, sigma). ``bias`` is the distance between the stationary
    point fitted to the noise-free surrogate curve on the line of window W and the reference stationary
    point of the surrogate. ``errorbar`` is the 95 % half-width of the fitted point when Gaussian noise
    sigma is added to the energies. The same noise draws are used for every pair of the mesh.
    """

    def __init__(self, curve: Callable[[np.ndarray], np.ndarray], k_abs: float, sign: int, x_ref: float,
                 draws: np.ndarray, windows: np.ndarray, noises: np.ndarray):
        self.curve, self.k_abs, self.sign, self.x_ref, self.draws = curve, float(k_abs), int(sign), float(x_ref), draws
        self.u = np.linspace(-1.0, 1.0, draws.shape[1])
        self.P = np.linalg.pinv(np.vander(self.u, 4, increasing=True))
        self.w = np.array(windows, dtype=float)
        self.s = np.array(noises, dtype=float)
        self.e = self._errors(self.w, self.s)

    def _errors(self, windows: np.ndarray, noises: np.ndarray) -> np.ndarray:
        out = np.empty((len(noises), len(windows)))
        for i, w in enumerate(windows):
            R = window_half_width(w, self.k_abs)
            values = self.curve(R * self.u)
            bias = abs(R * float(_library_step(np.dot(self.P, values), self.sign, -1.0, 1.0)[0]) - self.x_ref)
            noisy = values[None, None, :] + noises[:, None, None] * self.draws[None, :, :]
            x = R * _library_step(np.dot(noisy, self.P.T), self.sign, -1.0, 1.0)[0]
            out[:, i] = bias + _fraction_error(x)
        return out

    def add_window(self, w: float) -> None:
        col = self._errors(np.array([w]), self.s)
        order = np.argsort(np.append(self.w, w), kind="stable")
        self.w = np.append(self.w, w)[order]
        self.e = np.hstack([self.e, col])[:, order]

    def add_noise(self, s: float) -> None:
        row = self._errors(self.w, np.array([s]))
        order = np.argsort(np.append(self.s, s), kind="stable")
        self.s = np.append(self.s, s)[order]
        self.e = np.vstack([self.e, row])[order]

    def select(self, epsilon: float) -> Optional[Tuple[int, int]]:
        """Indices (window, noise) of the choice: the largest noise that has at least one window with
        error < epsilon and window >= noise; among the windows of that row, the one closest to the
        middle of the allowed range. Two windows at the same distance from the middle (this happens
        whenever an even number of equally spaced windows is allowed): the smaller one.
        None if no pair is allowed."""
        for si in range(len(self.s) - 1, -1, -1):
            ok = np.where((self.e[si] < epsilon) & (self.w >= self.s[si]))[0]
            if len(ok):
                distance = np.abs(self.w[ok] - 0.5 * (self.w[ok[0]] + self.w[ok[-1]]))
                nearest = np.where(distance <= distance.min() * (1.0 + 1.0e-9))[0][0]
                return int(ok[nearest]), si
        return None

    def refine(self, epsilon: float, w_max: float) -> Optional[Tuple[int, int]]:
        """The choice after the mesh was refined around it. Four steps, each repeated while it applies:

        1. the chosen window is the smallest of the mesh: add half of the second smallest window
           (not below ``w_max / 1000``);
        2. the chosen noise is the smallest (zero): add half of the second smallest noise;
        3. the chosen noise is the largest: add twice that noise (not above ``w_max``);
        4. the error of the choice is below 0.9 epsilon: add the mid points between the choice and its
           neighbours (noise above, window above, window below), unless they are closer than
           0.005 of the largest noise or 0.02 ``w_max``.
        """
        pick = self.select(epsilon)
        if pick is None:
            return None
        while pick[0] == 0:
            w_new = self.w[1] / 2.0
            if w_new < 1.0e-3 * w_max:
                break
            self.add_window(w_new); pick = self.select(epsilon)
        while pick[1] == 0:
            s_new = self.s[1] / 2.0
            if s_new < 1.0e-8 * w_max:
                break
            self.add_noise(s_new); pick = self.select(epsilon)
        while pick[1] == len(self.s) - 1:
            s_new = 2.0 * self.s[pick[1]]
            if s_new > w_max:
                break
            self.add_noise(s_new); pick = self.select(epsilon)
        while self.e[pick[1], pick[0]] / epsilon < 0.9:
            wi, si = pick
            w_this, s_this = self.w[wi], self.s[si]
            new_noise, new_windows = None, []
            if si < len(self.s) - 1:
                mid = 0.5 * (self.s[si] + self.s[si + 1])
                if abs(mid - s_this) > 5.0e-3 * self.s.max():
                    new_noise = mid
            if wi < len(self.w) - 1:
                mid = 0.5 * (self.w[wi] + self.w[wi + 1])
                if abs(mid - w_this) > 2.0e-2 * w_max:
                    new_windows.append(mid)
            if wi > 0:
                mid = 0.5 * (self.w[wi - 1] + self.w[wi])
                if abs(mid - w_this) > 2.0e-2 * w_max:
                    new_windows.append(mid)
            if new_noise is None and not new_windows:
                break
            if new_noise is not None:
                self.add_noise(new_noise)
            for w in new_windows:
                self.add_window(w)
            pick = self.select(epsilon)
        return pick

    def flags(self, pick: Tuple[int, int], epsilon: float) -> List[str]:
        wi, si = pick
        out = []
        if wi == 0:
            out.append("smallest window of the mesh")
        elif wi == len(self.w) - 1:
            out.append("largest window of the mesh")
        if si == 0:
            out.append("zero noise")
        elif si == len(self.s) - 1:
            out.append("largest noise of the mesh")
        if self.e[si, wi] / epsilon < 0.9:
            out.append("error below 0.9 epsilon")
        return out


def size_direction(t_scan: Sequence[float], e_scan: Sequence[float], k: float, epsilon: float, sign: int = 1,
                   points: int = 7, scales: Sequence[float] = (1, 4, 16, 64), mesh: int = 25,
                   noise_frac: float = 0.05, resamples: int = 500, rng: Optional[np.random.Generator] = None,
                   refine: bool = True) -> Dict[str, object]:
    """Window and noise of one direction: the largest noise for which the fitted stationary point of the
    surrogate curve stays within ``epsilon``.

    ``t_scan`` (angstrom) and ``e_scan`` (kcal/mol) are the surrogate scan of the direction, ``k`` its
    force constant (kcal/mol/A^2), ``sign`` +1 for a minimum and -1 for a maximum along the direction.

    The rules, in the order in which they are applied:

    * The surrogate curve is the cubic spline through the scan points.
    * The reference stationary point ``x_ref`` is fitted with a cubic to as many points as the scan has,
      on the line of half-width sqrt(0.02) = 0.141 A about the scan centre.
    * A mesh holds ``mesh`` windows from 0 to ``W_top`` and ``mesh`` noises from 0 to
      ``noise_frac * W_top``. A window below 0.0001 hartree is replaced by that value.
    * For a pair (W, sigma): a line of ``points`` points with half-width ``sqrt(2 W / |k|)``; bias =
      distance of the fitted point (no noise) from ``x_ref``; error bar = 95 % half-width of the fitted
      point over ``resamples`` noise draws (the same draws for every pair); error = bias + error bar.
    * The choice is the largest noise that has a window with error < ``epsilon`` and window >= noise,
      and in that row the window closest to the middle of the allowed windows. With ``refine`` the
      mesh is then refined around the choice (see ``_ErrorMesh.refine``).
    * This is done for ``W_top = W_max / scale`` for every entry of ``scales``, ``W_max`` being the window
      of the whole scan. The scale with the largest noise is kept and computed once more with new noise
      draws; that last result is returned. (If the new draws find no pair, the first result is kept.)
      With a single scale there is no second computation.

    ``scales = (1,)`` and ``mesh = 10`` are the settings of the library that the first applications
    used. ``scales = (1, 4, 16, 64)`` and ``mesh = 25`` are the defaults; an earlier application used them.

    Differences from the old code:

    * The search for direction tolerances from tolerances of the subspace coordinates is not part of
      this function. Tolerances are given per direction.
    * Two allowed windows at the same distance from the middle of the allowed range: the old code
      takes the one that rounding errors favour, so its choice between the two is arbitrary. Here
      the smaller window is taken. With the same noise draws this is the only case in which the two
      codes choose differently (checked on six test curves).
    * In refinement step 4 the old code takes, for the smallest window, the largest window as its
      "lower neighbour" and can then repeat the same insertion without end. Here the smallest window
      has no lower neighbour.

    Returns a dict with ``window`` and ``noise`` (kcal/mol), ``half_width`` (angstrom), ``error`` and
    ``bias`` (angstrom) of the chosen pair, ``x_ref``, ``scale``, ``w_max``, ``r_max``, ``flags`` (list of
    notes about the position of the choice on its mesh) and ``trials`` (the result of every scale).
    Raises :class:`SizerError` if no scale has an allowed pair.
    """
    t_scan = np.asarray(t_scan, dtype=float); e_scan = np.asarray(e_scan, dtype=float)
    if t_scan.ndim != 1 or t_scan.shape != e_scan.shape or len(t_scan) < 4:
        raise ValueError("the surrogate scan needs at least 4 points, with one energy per displacement")
    if not (np.all(np.isfinite(t_scan)) and np.all(np.isfinite(e_scan))):
        raise ValueError("the surrogate scan holds a value that is not a finite number")
    if not epsilon > 0.0:
        raise ValueError("the tolerance epsilon must be larger than zero (found %r)" % (epsilon,))
    order = np.argsort(t_scan)
    t_scan, e_scan = t_scan[order], e_scan[order]
    k_abs = abs(float(k))
    r_max = float(min(-t_scan[0], t_scan[-1]))
    if not r_max > 0.0:
        raise ValueError("the surrogate scan must reach to both sides of its centre")
    w_max = 0.5 * k_abs * r_max ** 2
    spline = CubicSpline(t_scan, e_scan)
    t_lo, t_hi = float(t_scan[0]), float(t_scan[-1])

    def curve(t: np.ndarray) -> np.ndarray:
        return spline(_EDGE_SHRINK * np.clip(t, t_lo, t_hi))

    # reference stationary point of the surrogate along this direction
    r_ref = window_half_width(0.01 * k_abs, k_abs)                 # sqrt(0.02) A unless the direction is very soft
    t_ref = np.clip(np.linspace(-r_ref, r_ref, len(t_scan)), t_lo, t_hi)
    x_ref = float(fit_step(t_ref, curve(t_ref), sign, errors=None)["m"])

    rng = np.random.default_rng(0) if rng is None else rng

    def one(scale: float) -> Optional[Dict[str, object]]:
        w_top = w_max / float(scale)
        draws = rng.standard_normal((int(resamples), int(points)))
        m = _ErrorMesh(curve, k_abs, sign, x_ref, draws, np.linspace(0.0, w_top, int(mesh)),
                       np.linspace(0.0, noise_frac * w_top, int(mesh)))
        pick = m.refine(epsilon, w_max) if refine else m.select(epsilon)
        if pick is None:
            return None
        wi, si = pick
        w = float(max(m.w[wi], WINDOW_FLOOR))
        R = window_half_width(w, k_abs)
        bias = abs(R * float(_library_step(np.dot(m.P, curve(R * m.u)), sign, -1.0, 1.0)[0]) - x_ref)
        return {"window": w, "noise": float(m.s[si]), "half_width": R, "error": float(m.e[si, wi]), "bias": float(bias),
                "scale": float(scale), "flags": m.flags(pick, epsilon)}

    scales = [float(v) for v in scales]
    trials = []
    best: Optional[Dict[str, object]] = None
    for scale in scales:
        res = one(scale)
        trials.append({"scale": scale, "found": res is not None,
                       "window": None if res is None else res["window"], "noise": None if res is None else res["noise"]})
        if res is not None and (best is None or res["noise"] > best["noise"]):
            best = res
    if best is None:
        raise SizerError("no window and noise meet the tolerance epsilon = %.5f A (window scales %s, largest window "
                         "%.3f kcal/mol, reference stationary point at %+.5f A)" % (epsilon, scales, w_max, x_ref))
    if len(scales) > 1:
        again = one(best["scale"])
        if again is not None:
            best = again
        else:
            best = dict(best, flags=list(best["flags"]) + ["the second computation with new noise draws found no pair"])
    best.update(x_ref=x_ref, w_max=float(w_max), r_max=r_max, epsilon=float(epsilon), trials=trials)
    return best


# ============================================================================= the driver
def _tkey(t: float) -> str:
    return "%+.4f" % t


def _point_name(j: int, t: float) -> str:
    return "d%d_%s%04d" % (j, "p" if t > 0 else "m", int(round(abs(t) * 1.0e4)))


def _jsonable(obj):
    """Copy of ``obj`` that the json module can write: numpy types become Python types, and a float
    that is not finite becomes None."""
    if isinstance(obj, dict):
        return {str(key): _jsonable(val) for key, val in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj) if np.isfinite(obj) else None
    return obj


def _options(settings: Dict[str, object], d: int) -> Dict[str, object]:
    """The options of the classic search: ``DEFAULTS`` changed by ``settings["classic"]``, checked."""
    user = settings.get("classic") or {}
    if not isinstance(user, dict):
        raise ValueError('settings["classic"] must be a dict of options, found %r' % type(user).__name__)
    unknown = sorted(set(user) - set(DEFAULTS))
    if unknown:
        raise ValueError("unknown option(s) of the classic search: %s. Known options: %s"
                         % (", ".join(unknown), ", ".join(sorted(DEFAULTS))))
    opt = dict(DEFAULTS); opt.update(user)
    for key in ("iterations", "points", "scan_points", "mesh", "resamples", "fit_resamples", "seed", "drop", "min_blocks",
                "max_blocks", "final_blocks", "final_series", "noise_blocks", "warmup"):
        if int(opt[key]) != opt[key]:
            raise ValueError("classic option %s must be a whole number (found %r)" % (key, opt[key]))
        opt[key] = int(opt[key])
    if opt["iterations"] < 1:
        raise ValueError("classic option iterations must be at least 1")
    if opt["points"] < 7 or opt["points"] % 2 == 0:
        raise ValueError("classic option points must be an odd number of at least 7 (found %d): the middle point is "
                         "the shared centre, and the audit needs 5 points in a cubic fit that leaves one point out"
                         % opt["points"])
    if opt["scan_points"] < 5 or opt["scan_points"] % 2 == 0:
        raise ValueError("classic option scan_points must be an odd number of at least 5 (found %d)" % opt["scan_points"])
    if opt["mesh"] < 3 or opt["resamples"] < 20:
        raise ValueError("classic options mesh (at least 3) and resamples (at least 20) are too small")
    if opt["drop"] < 0 or opt["min_blocks"] < _MIN_USABLE_BLOCKS:
        raise ValueError("classic options: drop must not be negative and min_blocks must be at least %d" % _MIN_USABLE_BLOCKS)
    if opt["max_blocks"] < opt["drop"] + opt["min_blocks"]:
        raise ValueError("classic option max_blocks = %d is below drop + min_blocks = %d"
                         % (opt["max_blocks"], opt["drop"] + opt["min_blocks"]))
    if opt["final_blocks"] < opt["drop"] + opt["min_blocks"] or opt["final_series"] < 1:
        raise ValueError("classic options: final_blocks must be at least drop + min_blocks = %d and final_series at least 1"
                         % (opt["drop"] + opt["min_blocks"]))
    opt["window_scales"] = [float(s) for s in opt["window_scales"]]
    if not opt["window_scales"] or min(opt["window_scales"]) < 1.0:
        raise ValueError("classic option window_scales must hold numbers of at least 1")
    for key in ("windows", "noises"):
        if opt[key] is not None:
            vals = [float(v) for v in opt[key]]
            if len(vals) != d or not all(np.isfinite(v) and v > 0.0 for v in vals):
                raise ValueError("classic option %s must hold one positive number per direction (%d)" % (key, d))
            opt[key] = vals
    if (opt["windows"] is None) != (opt["noises"] is None):
        raise ValueError("classic options windows and noises must be given together")
    return opt


class ClassicLineSearch:
    """Classic line search of one geometry (a minimum or a saddle).

    Parameters
    ----------
    name        short label of the target, used in keys and file names (for example ``min``).
    engine      source of DMC block energies.
    surrogate   source of the surrogate energies of the scan (``request_scan`` / ``scan``). May be None
                if the options ``windows`` and ``noises`` are given.
    symbols, start
                element symbols and the starting geometry (n_atoms, 3), angstrom.
    vectors     search directions, shape (d, 3 n), unit norm, Cartesian. They stay fixed.
    curvature   signed surrogate force constants k_j along the directions at the start
                (kcal/mol/A^2, E = 0.5 k t^2). Along a direction with negative k the search looks for
                a maximum (the reaction coordinate of a saddle search); its window uses |k|.
    settings    the QMC settings; they are passed to the engine with every point. Two entries are read here:
                ``settings["sigma_block"]``, the noise of one DMC block in kcal/mol (from it the number of
                blocks of a point is ``(sigma_block / sigma_j)**2`` plus the discarded blocks), and
                ``settings["classic"]``, a dict that changes entries of ``DEFAULTS`` in this module. Without
                ``sigma_block`` the noise of a block is measured on the first run at the starting geometry.
    state_dir   folder for the state file ``<name>_state.json`` and the log ``<name>.log``.

    Status values: ``RUNNING``, ``FINALIZING`` (the full-length runs are going), ``DONE`` and
    ``NEEDS_ATTENTION``. The classic search never reports ``CONVERGED``: it runs a fixed number of
    iterations and has no test that the geometry stopped moving or that the remaining energy error is
    below a tolerance. ``DONE`` means only that the iterations and the final runs are finished. The
    bracket check and the audit in the state file say how far each step can be trusted.

    Results: :meth:`final_geometry` (the last centre; the full-length final runs are made there),
    :meth:`mean_geometry` (the mean of the last centres, the number that the old method reported),
    :meth:`trajectory`, :meth:`final_energy`, :meth:`start_energy`, ``state["final"]["energy_change"]``
    (kcal/mol, with 1 standard error), :meth:`audit`, :meth:`brackets`.
    """

    def __init__(self, name: str, engine: Engine, surrogate: Optional[Surrogate], symbols: Sequence[str],
                 start: np.ndarray, vectors: np.ndarray, curvature: Sequence[float],
                 settings: Optional[Dict[str, object]] = None, state_dir: str = ".",
                 log: Optional[Callable[[str], None]] = None):
        self.name = name
        self.engine = engine
        self.surrogate = surrogate
        self.settings = dict(settings or {})
        self.state_dir = state_dir
        self._log_sink = log
        os.makedirs(state_dir, exist_ok=True)
        self.state_file = os.path.join(state_dir, "%s_state.json" % name)
        self.log_file = os.path.join(state_dir, "%s.log" % name)
        if os.path.exists(self.state_file):
            with open(self.state_file) as fh:
                self.state = json.load(fh)
            if self.state.get("protocol") != "classic":
                raise ValueError("%s is the state file of search protocol %r, not of the classic line search; "
                                 "use another state folder or name" % (self.state_file, self.state.get("protocol")))
            self.vectors = np.array(self.state["vectors"], dtype=float)
            self.symbols = list(self.state["symbols"])
            self.opt = dict(self.state["options"])
            return
        vectors = np.atleast_2d(np.asarray(vectors, dtype=float))
        norms = np.linalg.norm(vectors, axis=1)
        if np.abs(norms - 1.0).max() > 1e-6:
            raise ValueError("search directions must have unit norm (found %s)" % np.round(norms, 6))
        start = np.asarray(start, dtype=float).reshape(-1, 3)
        if vectors.shape[1] != start.size:
            raise ValueError("direction length %d does not match %d atoms" % (vectors.shape[1], len(start)))
        if len(symbols) != len(start):
            raise ValueError("%d symbols for %d atoms" % (len(symbols), len(start)))
        curvature = np.asarray(curvature, dtype=float).reshape(-1)
        if len(curvature) != len(vectors):
            raise ValueError("%d curvatures for %d directions" % (len(curvature), len(vectors)))
        if not np.all(np.isfinite(curvature)) or np.any(curvature == 0.0):
            raise ValueError("every surrogate curvature must be finite and not zero (found %s)" % curvature.tolist())
        d = len(vectors)
        self.vectors = vectors
        self.symbols = list(symbols)
        self.opt = _options(self.settings, d)
        opt = self.opt
        if opt["windows"] is None and surrogate is None:
            raise ValueError("the classic search needs a surrogate for its energy scan, or the options windows and noises")
        eps = opt["epsilon"]
        if eps is None:
            epsilon = [default_tolerance(k, float(opt["epsilon_energy"])) for k in curvature]
        elif np.ndim(eps) == 0:
            epsilon = [float(eps)] * d
        else:
            epsilon = [float(v) for v in eps]
        if len(epsilon) != d or not all(np.isfinite(v) and v > 0.0 for v in epsilon):
            raise ValueError("classic option epsilon must give one positive tolerance per direction (%d)" % d)
        sigma_block = self.settings.get("sigma_block")
        if sigma_block is not None:
            sigma_block = float(sigma_block)
            if not (np.isfinite(sigma_block) and sigma_block > 0.0):
                raise ValueError('settings["sigma_block"] must be a positive noise in kcal/mol (found %r)'
                                 % (self.settings.get("sigma_block"),))
        self.state = {
            "name": name, "protocol": "classic", "status": RUNNING_STATUS, "message": "",
            "symbols": self.symbols, "vectors": vectors.tolist(), "curvature": curvature.tolist(),
            # role of every direction: +1 = the energy is minimized along it, -1 = maximized
            "roles": [1 if float(c) > 0 else -1 for c in curvature],
            "options": _jsonable(opt), "epsilon": epsilon,
            "sigma_block": sigma_block, "sigma_block_source": None if sigma_block is None else "settings",
            "scan": None, "sizer": None, "plan": None, "warnings": [], "warning_keys": [],
            "iteration": 1, "iters": {},
        }
        if opt["windows"] is not None:
            self.state["sizer"] = [
                {"window": float(w), "noise": float(s), "half_width": window_half_width(w, k), "epsilon": epsilon[j],
                 "source": "settings"} for j, (w, s, k) in enumerate(zip(opt["windows"], opt["noises"], curvature))]
        self._new_iteration(1, start, np.zeros(d))
        self._save()

    # ------------------------------------------------------------------ small helpers
    @property
    def dim(self) -> int:
        return int(self.vectors.shape[0])

    @property
    def status(self) -> str:
        return self.state["status"]

    def say(self, text: str) -> None:
        line = "%s  [%s] %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), self.name, text)
        with open(self.log_file, "a") as fh:
            fh.write(line + "\n")
        if self._log_sink:
            self._log_sink(line)

    def _save(self) -> None:
        tmp = self.state_file + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(self.state, fh, indent=1)
        os.replace(tmp, self.state_file)

    def _it(self, k: Optional[int] = None) -> dict:
        return self.state["iters"][str(self.state["iteration"] if k is None else k)]

    def centre(self, k: Optional[int] = None) -> np.ndarray:
        return np.array(self._it(k)["centre"], dtype=float)

    def _key(self, k: int, leaf: str) -> str:
        return "%s/it%d/%s" % (self.name, k, leaf)

    def _attention(self, message: str) -> None:
        self.state["status"] = NEEDS_ATTENTION
        self.state["message"] = message
        self.say("NEEDS ATTENTION: " + message)

    def _warn(self, message: str, key: Optional[str] = None) -> None:
        """Record a warning in the state and in the log. Warnings with the same ``key`` (or, without a key,
        the same text) are recorded only the first time."""
        known = self.state.setdefault("warning_keys", [])
        tag = key or message
        if tag not in known:
            known.append(tag)
            self.state["warnings"].append(message)
            self.say("WARNING: " + message)

    def clear_attention(self) -> str:
        """Set a ``NEEDS_ATTENTION`` status back to running, after a person repaired the cause (for example
        a failed point that was run again). The next :meth:`tick` checks everything again."""
        if self.state["status"] == NEEDS_ATTENTION:
            self.state["status"] = FINALIZING if self.state.get("final") else RUNNING_STATUS
            self.state["message"] = ""
            self.say("the NEEDS_ATTENTION status was cleared by request; checking again")
            self._save()
        return self.state["status"]

    def _check_distance(self, geometries: Sequence[np.ndarray], what: str, key: str) -> None:
        """Warn (once per ``key``) when one of the geometries has two atoms closer than ``min_distance``. The old
        method has no such check; the search goes on, because the classic rules must not be changed here."""
        limit = float(self.opt["min_distance"])
        dmin = min(min_distance(x) for x in geometries)
        if dmin < limit:
            self._warn("%s: two atoms are only %.3f A apart (limit of this check: %.2f A). The energy of such a "
                       "geometry may have no meaning" % (what, dmin, limit), key=key)

    # ------------------------------------------------------------------ building
    def _new_iteration(self, k: int, centre: np.ndarray, coordinates: np.ndarray) -> None:
        n_series = int(self.opt["final_series"]) if k == 1 else 1
        self.state["iters"][str(k)] = {
            "centre": np.asarray(centre, dtype=float).reshape(-1, 3).tolist(),
            # position of the centre along the directions, relative to the start (angstrom)
            "coordinates": [float(v) for v in coordinates],
            "grid": None, "points": {}, "blocks": None, "centre_blocks": None,
            # iteration 1: the full-length runs at the starting geometry; the first of them is also the centre
            # point of the line search. Later iterations: one centre point.
            "centre_keys": [self._key(k, "centre" if s == 1 else "centre_s%d" % s) for s in range(1, n_series + 1)],
            "decided": {}, "composed": None, "centre_energy": None, "audit": None, "waiting": None,
        }
        self.state["iteration"] = k

    def _build_points(self, k: int, it: dict) -> None:
        """Grids and point keys of an iteration, once windows and block numbers are known."""
        plan, sizer = self.state["plan"], self.state["sizer"]
        n = int(self.opt["points"])
        grid, pts = [], {}
        centre = np.array(it["centre"], dtype=float)
        for j in range(self.dim):
            ts = np.linspace(-sizer[j]["half_width"], sizer[j]["half_width"], n)
            ts[n // 2] = 0.0
            grid.append([float(t) for t in ts])
            pts[str(j)] = {}
            for i, t in enumerate(ts):
                if i != n // 2:
                    pts[str(j)][_tkey(t)] = self._key(k, _point_name(j, t))
            if len(pts[str(j)]) != n - 1:
                raise ValueError("direction %d: the grid of half-width %.6f A is too narrow; its points do not have "
                                 "distinct names (names hold the displacement in units of 0.0001 A)"
                                 % (j, sizer[j]["half_width"]))
            self._check_distance([self._geometry(centre, j, ts[0]), self._geometry(centre, j, ts[-1])],
                                 "iteration %d, line of direction %d (half-width %.4f A)" % (k, j, ts[-1]), "line d%d" % j)
        it["grid"], it["points"] = grid, pts
        it["blocks"] = list(plan["blocks"])
        it["centre_blocks"] = int(plan["centre_blocks"])

    def _geometry(self, centre: np.ndarray, j: int, t: float) -> np.ndarray:
        return np.asarray(centre, dtype=float).reshape(-1, 3) + t * self.vectors[j].reshape(-1, 3)

    def _submit(self, key: str, coords: np.ndarray, blocks: int, role: str, series: int = 1,
                base_key: Optional[str] = None) -> None:
        self.engine.submit(PointSpec(key=key, symbols=self.symbols, coords=np.asarray(coords, dtype=float),
                                     blocks=int(blocks), role=role, series=series, base_key=base_key,
                                     settings=self.settings))

    def _ensure_start_runs(self) -> None:
        """The full-length runs at the starting geometry. Further series start once the first one writes blocks."""
        it1 = self._it(1)
        keys = it1["centre_keys"]
        self._submit(keys[0], np.array(it1["centre"]), self.opt["final_blocks"], "centre")
        if len(keys) > 1 and self.engine.read(keys[0]).nblocks > 0:
            for s, key in enumerate(keys[1:], start=2):
                self._submit(key, np.array(it1["centre"]), self.opt["final_blocks"], "centre", series=s, base_key=keys[0])

    def _ensure_points(self, k: int, it: dict) -> None:
        centre = np.array(it["centre"], dtype=float)
        if k > 1:
            self._submit(it["centre_keys"][0], centre, it["centre_blocks"], "centre")
        for j in range(self.dim):
            for t in it["grid"][j]:              # the labels of "points" are rounded; the grid holds the exact values
                if t != 0.0:
                    self._submit(it["points"][str(j)][_tkey(t)], self._geometry(centre, j, t), it["blocks"][j], "scan")

    # ------------------------------------------------------------------ stage: surrogate scan and sizer
    def _scan_geometries(self) -> Tuple[List[np.ndarray], List[List[float]], List[float]]:
        opt = self.opt
        n = int(opt["scan_points"])
        start = self.centre(1)
        geoms: List[np.ndarray] = [start]
        grids, widths = [], []
        for j, k in enumerate(self.state["curvature"]):
            R = scan_half_width(k, float(opt["window_frac"]))
            if opt["scan_half_width_max"] is not None:
                R = min(R, float(opt["scan_half_width_max"]))
            ts = np.linspace(-R, R, n)
            ts[n // 2] = 0.0
            widths.append(float(R)); grids.append([float(t) for t in ts])
            for i, t in enumerate(ts):
                if i != n // 2:
                    geoms.append(self._geometry(start, j, t))
            self._check_distance([self._geometry(start, j, ts[0]), self._geometry(start, j, ts[-1])],
                                 "surrogate scan of direction %d (half-width %.4f A)" % (j, R), "scan d%d" % j)
        return geoms, grids, widths

    def _tick_sizer(self) -> bool:
        """Request the surrogate scan, wait for it, and choose windows and noises. True when they exist."""
        st, opt = self.state, self.opt
        if st["scan"] is None or st["scan"]["energy"] is None:
            geoms, grids, widths = self._scan_geometries()
            if st["scan"] is None:
                st["scan"] = {"tag": "%s/scan" % self.name, "half_width": widths, "grid": grids, "energy": None,
                              "centre_energy": None, "points": len(geoms)}
                self.say("surrogate scan requested: %d geometries, half-widths %s A"
                         % (len(geoms), [round(w, 4) for w in widths]))
            scan = st["scan"]
            try:
                self.surrogate.request_scan(scan["tag"], geoms)
                energies = self.surrogate.scan(scan["tag"])
            except NotImplementedError:
                self._attention("the surrogate offers no energy scan (request_scan / scan); the classic search needs it, "
                                "or the options windows and noises")
                return False
            if energies is None:
                return False
            energies = np.asarray(energies, dtype=float).reshape(-1)
            if len(energies) != len(geoms):
                self._attention("the surrogate scan %s returned %d energies for %d geometries"
                                % (scan["tag"], len(energies), len(geoms)))
                return False
            n = int(opt["scan_points"]); per = n - 1
            table = []
            for j in range(self.dim):
                row = list(energies[1 + j * per: 1 + (j + 1) * per])
                row.insert(n // 2, energies[0])
                bad = [scan["grid"][j][i] for i, v in enumerate(row) if not np.isfinite(v)]
                if bad:
                    self._attention("the surrogate scan %s has no finite energy at displacement(s) %s A of direction %d"
                                    % (scan["tag"], ", ".join("%+.4f" % t for t in bad), j))
                    return False
                table.append([float(v) for v in row])
            scan["energy"] = table; scan["centre_energy"] = float(energies[0])
        scan = st["scan"]
        sizer = []
        for j in range(self.dim):
            k = st["curvature"][j]
            t = np.array(scan["grid"][j]); e = (np.array(scan["energy"][j]) - scan["centre_energy"]) * HARTREE_TO_KCAL
            rng = np.random.default_rng([int(opt["seed"]), 1, j])
            try:
                res = size_direction(t, e, k, st["epsilon"][j], sign=st["roles"][j], points=int(opt["points"]),
                                     scales=opt["window_scales"], mesh=int(opt["mesh"]), noise_frac=float(opt["noise_frac"]),
                                     resamples=int(opt["resamples"]), rng=rng)
            except SizerError as exc:
                self._attention("direction %d (surrogate force constant %.1f kcal/mol/A^2): %s" % (j, k, exc))
                return False
            res["source"] = "sizer"
            sizer.append(_jsonable(res))
            self.say("sizer d%d: k %.1f kcal/mol/A^2, epsilon %.4f A -> window %.3f kcal/mol (half-width %.4f A of %.4f A "
                     "scanned), noise %.3f kcal/mol, error %.4f A (bias %.4f A)%s"
                     % (j, k, st["epsilon"][j], res["window"], res["half_width"], res["r_max"], res["noise"], res["error"],
                        res["bias"], "" if not res["flags"] else "; " + "; ".join(res["flags"])))
            if abs(res["x_ref"]) > st["epsilon"][j]:
                self._warn("direction %d: the surrogate stationary point is %+.4f A from the start along this direction, "
                           "more than the tolerance %.4f A. The sizer assumes that the search starts at the surrogate "
                           "stationary point" % (j, res["x_ref"], st["epsilon"][j]))
        st["sizer"] = sizer
        return True

    # ------------------------------------------------------------------ stage: noise of one block
    def _tick_noise(self) -> bool:
        """Measure the noise of one block on the first run at the starting geometry. True when it is known."""
        st, opt = self.state, self.opt
        key = self._it(1)["centre_keys"][0]
        d = self.engine.read(key)
        if d.state == FAILED:
            self._attention("the run %s at the starting geometry failed: %s" % (key, d.message or "no message"))
            return False
        want = min(int(opt["drop"]) + int(opt["noise_blocks"]), int(opt["final_blocks"]))
        n = min(d.nblocks, want)
        if n < want and d.state not in (DONE, STOPPED):
            return False
        if n - int(opt["drop"]) < _MIN_USABLE_BLOCKS:
            self._attention("the run %s at the starting geometry ended with %d blocks; the noise of a block cannot be "
                            "measured (the first %d blocks are discarded)" % (key, d.nblocks, opt["drop"]))
            return False
        used = n - int(opt["drop"])
        _mean, err = reblock(d.energy[opt["drop"]:n], d.weight[opt["drop"]:n])
        sigma_block = float(err * np.sqrt(used) * HARTREE_TO_KCAL)
        if not (np.isfinite(sigma_block) and sigma_block > 0.0):
            self._attention("the block energies of %s do not vary; the noise of a block cannot be measured" % key)
            return False
        st["sigma_block"] = sigma_block
        st["sigma_block_source"] = "measured on blocks %d to %d of %s" % (opt["drop"] + 1, n, key)
        self.say("noise of one block: %.3f kcal/mol (%s)" % (sigma_block, st["sigma_block_source"]))
        return True

    def _make_plan(self) -> None:
        st, opt = self.state, self.opt
        sb = float(st["sigma_block"])
        blocks, capped, expected = [], [], []
        for j, sz in enumerate(st["sizer"]):
            n, cap = blocks_for_noise(sb, sz["noise"], opt["drop"], opt["min_blocks"], opt["max_blocks"])
            blocks.append(n); capped.append(cap); expected.append(sb / np.sqrt(n - opt["drop"]))
            if cap:
                self._warn("direction %d: the noise %.3f kcal/mol of the sizer needs more than the upper limit of %d "
                           "blocks; the points will have about %.3f kcal/mol" % (j, sz["noise"], opt["max_blocks"], expected[-1]))
        n_c = max(blocks)                    # the centre is shared: it gets the smallest noise of all directions
        st["plan"] = {"sigma_block": sb, "blocks": blocks, "capped": capped, "expected_noise": [float(v) for v in expected],
                      "centre_blocks": int(n_c), "centre_noise": float(sb / np.sqrt(n_c - opt["drop"]))}
        self.say("blocks per point (the first %d are discarded): %s by direction, %d at the centre; expected noise %s kcal/mol"
                 % (opt["drop"], blocks, n_c, [round(float(v), 3) for v in expected]))

    # ------------------------------------------------------------------ reading data
    def _energy(self, key: str, n: Optional[int] = None) -> Optional[Tuple[float, float, int]]:
        """Mean and reblocked error (hartree) of a point after the discarded blocks, and the blocks used.
        ``n`` limits the trace to its first n blocks. None if too few blocks exist."""
        d = self.engine.read(key)
        m = d.nblocks if n is None else min(int(n), d.nblocks)
        drop = int(self.opt["drop"])
        if m - drop < _MIN_USABLE_BLOCKS:
            return None
        mean, err = reblock(d.energy[drop:m], d.weight[drop:m])
        return float(mean), float(err), int(m)

    def _combined(self, keys: Sequence[str]) -> Optional[Tuple[float, float]]:
        """Mean and error (hartree) over independent series of one geometry, weighted by 1 / error^2."""
        vals = [v for v in (self._energy(key) for key in keys) if v is not None]
        if not vals:
            return None
        if any(v[1] <= 0.0 for v in vals):
            return float(np.mean([v[0] for v in vals])), 0.0
        w = np.array([1.0 / v[1] ** 2 for v in vals])
        return float(np.sum(w * np.array([v[0] for v in vals])) / w.sum()), float(1.0 / np.sqrt(w.sum()))

    def _gather(self, k: int, it: dict) -> Optional[dict]:
        """Energies of all points of an iteration, or None while a point is missing (or after a failure)."""
        opt = self.opt
        drop = int(opt["drop"])
        ckey = it["centre_keys"][0]
        n_centre = min(int(it["centre_blocks"]), int(opt["final_blocks"])) if k == 1 else None
        todo = [(ckey, n_centre)] + [(key, None) for j in range(self.dim) for key in it["points"][str(j)].values()]
        ready, vals = 0, {}
        for key, n_use in todo:
            d = self.engine.read(key)
            if d.state == FAILED:
                self._attention("iteration %d: point %s failed: %s" % (k, key, d.message or "no message"))
                return None
            complete = d.state in (DONE, STOPPED) or (n_use is not None and d.nblocks >= n_use)
            if not complete:
                continue
            e = self._energy(key, n_use)
            if e is None:
                self._attention("iteration %d: point %s ended with %d blocks; at least %d are needed (the first %d are "
                                "discarded)" % (k, key, d.nblocks, drop + _MIN_USABLE_BLOCKS, drop))
                return None
            vals[key] = e; ready += 1
        if ready < len(todo):
            if it.get("waiting") != ready:
                it["waiting"] = ready
                self.say("iteration %d: %d of %d points are complete" % (k, ready, len(todo)))
            return None
        return vals

    # ------------------------------------------------------------------ one pass
    def tick(self) -> str:
        """One pass: submit what is missing, read what exists, fit when an iteration is complete."""
        st = self.state
        if st["status"] == FINALIZING:
            self._tick_final()
            self._save()
            return st["status"]
        if st["status"] != RUNNING_STATUS:
            return st["status"]
        self._ensure_start_runs()
        if st["sizer"] is None and not self._tick_sizer():
            self._save(); return st["status"]
        if st["sigma_block"] is None and not self._tick_noise():
            self._save(); return st["status"]
        if st["plan"] is None:
            self._make_plan()
        k = st["iteration"]; it = self._it(k)
        if it["grid"] is None:
            self._build_points(k, it)
            self.say("iteration %d built: %d points about the centre" % (k, 1 + sum(len(v) for v in it["points"].values())))
        self._ensure_points(k, it)
        vals = self._gather(k, it)
        if vals is not None:
            self._fit_iteration(k, it, vals)
        self._save()
        return st["status"]

    def _fit_iteration(self, k: int, it: dict, vals: Dict[str, Tuple[float, float, int]]) -> None:
        st, opt = self.state, self.opt
        e0, s0, n0 = vals[it["centre_keys"][0]]
        it["centre_energy"] = [e0, s0]
        steps = np.zeros(self.dim)
        lines = []
        for j in range(self.dim):
            ts = np.array(it["grid"][j], dtype=float)
            e, s, nb = [], [], []
            for t in ts:
                v = (e0, s0, n0) if t == 0.0 else vals[it["points"][str(j)][_tkey(t)]]
                e.append(v[0]); s.append(v[1]); nb.append(v[2])
            ek = (np.array(e) - e0) * HARTREE_TO_KCAL; sk = np.array(s) * HARTREE_TO_KCAL
            rng = np.random.default_rng([int(opt["seed"]), 2, k, j])
            res = fit_step(ts, ek, sign=st["roles"][j], errors=sk, resamples=int(opt["fit_resamples"]), rng=rng)
            res.update(t=[float(v) for v in ts], e=[float(v) for v in ek], s=[float(v) for v in sk], blocks_used=nb,
                       energy_hartree=[float(v) for v in e], error_hartree=[float(v) for v in s], role=st["roles"][j])
            note = {INSIDE: "stationary point inside the grid",
                    OUTSIDE: "stationary point OUTSIDE the grid of half-width %.4f A: the step is an extrapolation" % res["R"],
                    NO_ROOT: "the cubic has NO %s: the step is an end of the grid" % ("minimum" if st["roles"][j] > 0 else "maximum"),
                    }[res["status"]]
            it["decided"][str(j)] = _jsonable({"status": res["status"], "m": res["m"], "blocks": int(it["blocks"][j]),
                                               "note": note, "res": res})
            steps[j] = res["m"]
            lines.append("d%d: step %+.4f +/- %.4f A (1 standard error; 95 %% half-width %.4f), %s"
                         % (j, res["m"], res["sm"], res["sm95"], note))
        if not np.all(np.isfinite(steps)):
            self._attention("iteration %d: a fitted step is not a finite number (%s)" % (k, steps.tolist()))
            return
        it["composed"] = {"rule": "sum of the one-dimensional steps", "components": [float(v) for v in steps],
                          "step_norm": float(np.linalg.norm(steps)),
                          "not_bracketed": [j for j in range(self.dim) if it["decided"][str(j)]["status"] != INSIDE]}
        self.say("iteration %d fitted: |step| = %.5f A, centre energy %.6f +/- %.6f hartree"
                 % (k, float(np.linalg.norm(steps)), e0, s0))
        for line in lines:
            self.say("  " + line)
        self._record_audit(k, it)
        if k == 1 and opt.get("stop_if_unresolved"):
            reason = self.untrusted(1)
            if reason:
                # the automatic search (module ``auto``) continues from here with the dense-scan search
                st["status"] = UNRESOLVED_STATUS
                st["message"] = "the steps of the first iteration cannot be trusted: %s" % reason
                self.say("the search stops here: %s" % st["message"])
                return
        new_centre = np.array(it["centre"], dtype=float) + np.dot(steps, self.vectors).reshape(-1, 3)
        new_coordinates = np.array(it["coordinates"], dtype=float) + steps
        self._check_distance([new_centre], "the centre after iteration %d" % k, "centre")
        if k >= int(opt["iterations"]):
            self._start_final(k, new_centre, new_coordinates)
            return
        self._new_iteration(k + 1, new_centre, new_coordinates)
        nxt = self._it(k + 1)
        self._build_points(k + 1, nxt)
        self._ensure_points(k + 1, nxt)
        self.say("iteration %d built: %d points about the new centre" % (k + 1, 1 + sum(len(v) for v in nxt["points"].values())))

    # ------------------------------------------------------------------ the audit
    def _audit_data(self, k: int) -> Tuple[list, list, list]:
        it = self._it(k)
        if len(it["decided"]) < self.dim:
            raise ValueError("iteration %d of %s is not fitted yet: no data for the audit" % (k, self.name))
        res = [it["decided"][str(j)]["res"] for j in range(self.dim)]
        return [r["t"] for r in res], [r["e"] for r in res], [r["s"] for r in res]

    def audit(self, k: Optional[int] = None) -> List[Dict[str, object]]:
        """Stability audit of iteration ``k`` (default: the last fitted iteration). One dict per direction with
        ``verdict``, ``spread``, ``threshold`` and ``candidates``; see :func:`montepass.search.audit.audit_iteration`."""
        from .audit import audit_iteration

        if k is None:
            done = [int(s) for s, it in self.state["iters"].items() if len(it["decided"]) == self.dim]
            if not done:
                raise ValueError("no iteration of %s is fitted yet: no data for the audit" % self.name)
            k = max(done)
        t, e, s = self._audit_data(int(k))
        return audit_iteration(t, e, s, self.state["epsilon"], signs=self.state["roles"],
                               floor=float(self.opt["audit_floor"]))

    def untrusted(self, k: int) -> str:
        """Why the steps of the fitted iteration ``k`` cannot be trusted, or '' if they can: directions that the
        stability audit did not resolve, and fitted steps that their grid does not bracket."""
        it = self._it(k)
        parts = []
        audit = it.get("audit")
        if isinstance(audit, list):
            bad = [j for j, r in enumerate(audit) if r["verdict"] != "RESOLVED"]
            if bad:
                parts.append("the stability audit did not resolve direction(s) %s" % ", ".join("%d" % j for j in bad))
        else:
            parts.append("the stability audit could not be made")
        outside = [j for j in range(self.dim) if it["decided"][str(j)]["status"] != INSIDE]
        if outside:
            parts.append("the fitted step of direction(s) %s is not bracketed by its grid" % ", ".join("%d" % j for j in outside))
        return "; ".join(parts)

    def _record_audit(self, k: int, it: dict) -> None:
        """Run the audit of a fitted iteration and keep it in the state. The audit never changes a step."""
        try:
            result = self.audit(k)
        except ValueError as exc:
            it["audit"] = {"error": str(exc)}
            self.say("  audit of iteration %d not possible: %s" % (k, exc))
            return
        it["audit"] = _jsonable(result)
        self.say("  audit: " + ", ".join(
            "d%d %s%s" % (j, r["verdict"], "" if r["spread"] is None else " (spread %.4f, threshold %.4f A)"
                          % (r["spread"], r["threshold"])) for j, r in enumerate(result)))

    # ------------------------------------------------------------------ the end of the search
    def _start_final(self, k: int, centre: np.ndarray, coordinates: np.ndarray) -> None:
        n = int(self.opt["final_series"])
        keys = [self._key(k, "final" if i == 1 else "final_s%d" % i) for i in range(1, n + 1)]
        self.state["final"] = {"iteration": k, "centre": np.asarray(centre).reshape(-1, 3).tolist(),
                               "coordinates": [float(v) for v in coordinates], "keys": keys,
                               # the classic search makes no statement about the remaining error
                               "certified": False, "energy_change": None}
        self.state["status"] = FINALIZING
        self._submit(keys[0], centre, self.opt["final_blocks"], "final")
        self.say("the %d iterations are finished. Full-length runs requested at the final geometry." % k)

    def _tick_final(self) -> None:
        st, opt = self.state, self.opt
        fin = st["final"]
        centre = np.array(fin["centre"], dtype=float)
        self._submit(fin["keys"][0], centre, opt["final_blocks"], "final")
        if self.engine.read(fin["keys"][0]).nblocks > 0:
            for i, key in enumerate(fin["keys"][1:], start=2):
                self._submit(key, centre, opt["final_blocks"], "final", series=i, base_key=fin["keys"][0])
        self._ensure_start_runs()
        keys = list(fin["keys"]) + list(self._it(1)["centre_keys"])
        states = {key: self.engine.read(key) for key in keys}
        failed = [key for key, d in states.items() if d.state == FAILED]
        if failed:
            self._attention("full-length run(s) failed: %s" % ", ".join(failed))
            return
        if not all(d.state in (DONE, STOPPED) for d in states.values()):
            return
        short = [key for key, d in states.items() if d.nblocks - int(opt["drop"]) < _MIN_USABLE_BLOCKS]
        if short:
            self._attention("full-length run(s) ended with too few blocks: %s" % ", ".join(short))
            return
        ef, e0 = self.final_energy(), self.start_energy()
        de = (ef[0] - e0[0]) * HARTREE_TO_KCAL; sd = float(np.hypot(ef[1], e0[1]) * HARTREE_TO_KCAL)
        fin["energy_change"] = [float(de), sd]
        fin["energy"] = [ef[0], ef[1]]; fin["start_energy"] = [e0[0], e0[1]]
        self.say("energy of the final geometry minus energy of the starting geometry: %+.3f +/- %.3f kcal/mol" % (de, sd))
        st["mean"] = self._mean_of_last_centres()
        n_it = int(opt["iterations"])
        bad = [(int(ks), j) for ks, it in st["iters"].items() for j in range(self.dim)
               if str(j) in it["decided"] and it["decided"][str(j)]["status"] != INSIDE]
        last_audit = self._it(n_it).get("audit")
        unresolved = [j for j, r in enumerate(last_audit) if r["verdict"] != "RESOLVED"] if isinstance(last_audit, list) else None
        st["status"] = DONE_STATUS
        st["message"] = ("ended after %d iterations (a fixed number; the classic search has no convergence test). "
                         "%d of %d fitted steps were not bracketed by their grid. Audit of the last iteration: %s."
                         % (n_it, len(bad), n_it * self.dim,
                            "not available" if unresolved is None else
                            ("all %d directions RESOLVED" % self.dim if not unresolved else
                             "direction(s) %s UNRESOLVED" % unresolved)))
        if st["warnings"]:
            st["message"] += " %d warning(s), see the state file." % len(st["warnings"])
        self.say(st["message"])

    def _mean_of_last_centres(self) -> Optional[dict]:
        """The numbers that the old method reported at its end: the mean of the centres after at least
        ``warmup`` steps, and the mean of their energies.

        Centre 0 is the start, centre i the geometry after i steps. With 4 iterations and warmup 2 the mean
        runs over centres 2, 3 and 4. The error of a coordinate is sqrt(mean of the squared errors of the
        last steps), with 1 standard error per step (the old method used 95 % half-widths). The error of
        the energy is built the same way. The energy of the last centre comes from the full-length final runs.
        """
        st, opt = self.state, self.opt
        n_it, warm = int(opt["iterations"]), int(opt["warmup"])
        idx = [i for i in range(0, n_it + 1) if i >= warm]          # centre i = result of iteration i (0 = the start)
        if not idx:
            return None
        coords, errs, energies, e_errs = [], [], [], []
        for i in idx:
            if i == 0:
                coords.append(np.zeros(self.dim)); errs.append([0.0] * self.dim)
            else:
                it = self._it(i)
                coords.append(np.array(it["coordinates"]) + np.array(it["composed"]["components"]))
                errs.append([it["decided"][str(j)]["res"]["sm"] for j in range(self.dim)])
            e = self._it(i + 1)["centre_energy"] if i < n_it else st["final"]["energy"]
            energies.append(e[0]); e_errs.append(e[1])
        z_mean = np.mean(coords, axis=0)
        geometry = self.centre(1) + np.dot(z_mean, self.vectors).reshape(-1, 3)
        return {"centres": idx, "coordinates": [float(v) for v in z_mean],
                "coordinate_errors": [float(v) for v in np.sqrt(np.mean(np.square(errs), axis=0))],
                "geometry": geometry.tolist(), "energy": float(np.mean(energies)),
                "energy_error": float(np.sqrt(np.mean(np.square(e_errs))))}

    # ------------------------------------------------------------------ results
    def final_geometry(self) -> np.ndarray:
        """The last centre: after the last finished iteration, or the current centre while the search runs."""
        if self.state.get("final"):
            return np.array(self.state["final"]["centre"], dtype=float)
        return self.centre()

    def mean_geometry(self) -> Optional[np.ndarray]:
        """Mean of the last centres as the old method reported it (see ``state["mean"]``), or None before the end."""
        mean = self.state.get("mean")
        return None if not mean else np.array(mean["geometry"], dtype=float)

    def trajectory(self) -> List[np.ndarray]:
        """Centres of all iterations in order, and the final geometry once it exists."""
        out = [self.centre(k) for k in sorted(int(s) for s in self.state["iters"])]
        if self.state.get("final"):
            out.append(np.array(self.state["final"]["centre"], dtype=float))
        return out

    def coordinates(self) -> np.ndarray:
        """Positions of the trajectory along the directions, relative to the start (angstrom), one row per centre."""
        rows = [self._it(k)["coordinates"] for k in sorted(int(s) for s in self.state["iters"])]
        if self.state.get("final"):
            rows.append(self.state["final"]["coordinates"])
        return np.array(rows, dtype=float)

    def brackets(self) -> List[Dict[str, object]]:
        """The bracket check of every fitted step: iteration, direction, status, step and grid half-width."""
        out = []
        for k in sorted(int(s) for s in self.state["iters"]):
            for js, dec in sorted(self._it(k)["decided"].items(), key=lambda kv: int(kv[0])):
                out.append({"iteration": k, "direction": int(js), "status": dec["status"], "m": dec["m"],
                            "R": dec["res"]["R"], "sm": dec["res"]["sm"]})
        return out

    def centre_energy(self, k: Optional[int] = None) -> Optional[Tuple[float, float]]:
        """DMC energy (hartree) and error of the centre of iteration k as the line search used it, or None."""
        e = self._it(k)["centre_energy"]
        return None if e is None else (float(e[0]), float(e[1]))

    def final_energy(self) -> Optional[Tuple[float, float]]:
        """Combined DMC energy (hartree) and error of the full-length runs at the final geometry."""
        if not self.state.get("final"):
            return None
        return self._combined(self.state["final"]["keys"])

    def start_energy(self) -> Optional[Tuple[float, float]]:
        """Combined DMC energy (hartree) and error of the full-length runs at the starting geometry."""
        return self._combined(self._it(1)["centre_keys"])

    def total_blocks(self) -> int:
        """All DMC blocks acquired by this search so far (line points, centres, full-length runs)."""
        n = 0
        for it in self.state["iters"].values():
            n += sum(self.engine.read(key).nblocks for key in it["centre_keys"])
            n += sum(self.engine.read(key).nblocks for pts in it["points"].values() for key in pts.values())
        if self.state.get("final"):
            n += sum(self.engine.read(key).nblocks for key in self.state["final"]["keys"])
        return n

    def search_blocks(self) -> int:
        """Blocks spent on the search alone: the line points and the centre points. Of the full-length runs
        only the blocks that iteration 1 used as its centre point are counted."""
        n = 0
        for ks, it in self.state["iters"].items():
            n += sum(self.engine.read(key).nblocks for pts in it["points"].values() for key in pts.values())
            first = self.engine.read(it["centre_keys"][0]).nblocks
            if int(ks) == 1:
                n += min(first, int(it["centre_blocks"] or 0))
            else:
                n += first
        return n

    def run(self, max_ticks: int = 10000) -> str:
        """Tick until the search ends (for the mock engine and for tests)."""
        for _ in range(max_ticks):
            self.engine.advance()
            if self.tick() not in (RUNNING_STATUS, FINALIZING):
                break
        return self.state["status"]
