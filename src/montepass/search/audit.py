"""The stability audit of one classic line-search iteration.

The classic search fits one cubic to the 7 energies of a direction and takes its stationary
point as the step. That step can depend on the choice of the model more than its error bar
says: when the window is much wider than the range in which the energy is a cubic, a
quadratic fit, a cubic fit and a cubic fit without one of the points give different answers.
The error bar of the classic fit does not see this, because it assumes that the cubic is right.

The audit measures that dependence. It uses only the energies that the iteration computed
itself, so it needs no reference and no extra calculation. For one direction:

1. weighted least-squares fits (weights 1 / error^2): a quadratic through all points, a cubic
   through all points, and one cubic for every point left out;
2. of every fit, the stationary point of the wanted kind (a minimum; a maximum along a
   direction of negative curvature) that lies inside the range of the points used by that
   fit. A fit without such a point gives no candidate;
3. fewer than 3 candidates: the direction is ``UNRESOLVED``. Otherwise the spread is the largest
   candidate minus the smallest one, and the direction is ``RESOLVED`` if
   ``spread <= max(2 * epsilon, 0.01 A)``, with ``epsilon`` the tolerance of the direction.

``UNRESOLVED`` means: the fitted step is a choice of the model, not a measurement. The step of
that direction should not be trusted before the direction is scanned again in a narrower window.

Notes on the rule as it was used in an earlier application:

* The range in step 2 is the range of the points of that fit. For a cubic that leaves out an end
  point of the grid the range is therefore the remaining six points, not the whole grid.
* That audit looked for minima only. Here a direction with ``sign = -1`` is searched for a
  maximum, with the same rule.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from .classic import fit_polynomial, stationary_point

RESOLVED = "RESOLVED"
UNRESOLVED = "UNRESOLVED"
SKIPPED = "SKIPPED"              # too few points (the iteration is not finished): no verdict


def _weights(errors: np.ndarray) -> Optional[np.ndarray]:
    """Error bars for the weighted fits, or None for equal weights when every error bar is zero
    (energies without noise). A mix of zero and non-zero error bars is refused."""
    if not np.all(np.isfinite(errors)) or np.any(errors < 0.0):
        raise ValueError("the audit needs a finite error bar that is not negative for every point")
    if np.all(errors == 0.0):
        return None
    if np.any(errors == 0.0):
        raise ValueError("the audit found error bars of zero next to error bars that are not zero; "
                         "a weighted fit is not defined for them")
    return errors


def fitted_extremum(t: Sequence[float], e: Sequence[float], s: Optional[Sequence[float]], order: int,
                    sign: int = 1, leave_out: Optional[int] = None) -> Optional[float]:
    """Stationary point of a weighted polynomial fit, or None if the fit gives no candidate.

    ``t`` displacements, ``e`` energies, ``s`` their error bars (None = equal weights), ``order`` 2 or 3.
    ``leave_out`` is the index of a point that is not used. The result is the minimum of the fit
    (the maximum for ``sign = -1``) if it lies between the smallest and the largest displacement of
    the points that were used. A fit with fewer than ``order + 2`` points gives no candidate.
    """
    t = np.asarray(t, dtype=float); e = np.asarray(e, dtype=float)
    keep = np.ones(len(t), dtype=bool)
    if leave_out is not None:
        keep[int(leave_out)] = False
    tt, ee = t[keep], e[keep]
    ss = None if s is None else np.asarray(s, dtype=float)[keep]
    if len(tt) < order + 2 or not np.abs(tt).max() > 0.0:
        return None
    try:
        c = fit_polynomial(tt, ee, order, ss)
    except ValueError:
        return None                       # the points do not determine the polynomial
    x, found = stationary_point(c, sign)
    if not bool(found):
        return None
    x = float(x)
    if not (float(tt.min()) <= x <= float(tt.max())):
        return None
    return x


def audit_direction(t: Sequence[float], e: Sequence[float], s: Sequence[float], epsilon: float, sign: int = 1,
                    floor: float = 0.01, min_points: int = 5) -> Dict[str, object]:
    """Audit of one direction.

    ``t`` displacements of the points (angstrom, the centre included), ``e`` their energies and ``s``
    the standard errors of the energies (same unit as ``e``, any zero of energy), ``epsilon`` the
    tolerance of the direction (angstrom), ``sign`` +1 for a minimum and -1 for a maximum.

    Returns a dict:
      ``verdict``        ``RESOLVED``, ``UNRESOLVED``, or ``SKIPPED`` with fewer than ``min_points`` points
      ``spread``         largest minus smallest candidate (angstrom); None with fewer than 3 candidates
      ``threshold``      ``max(2 * epsilon, floor)`` (angstrom)
      ``candidates``     the stationary points that exist, in the order quadratic, cubic, leave-one-out
      ``quadratic``, ``cubic``     the stationary points of the two fits to all points (None = no candidate)
      ``leave_one_out``  one entry per point, in the order of ``t`` (None = no candidate)
      ``n_points``, ``sign``, ``note``
    """
    t = np.asarray(t, dtype=float); e = np.asarray(e, dtype=float); s = np.asarray(s, dtype=float)
    if not (t.ndim == 1 and t.shape == e.shape == s.shape):
        raise ValueError("the audit needs one energy and one error bar per displacement")
    if not (np.all(np.isfinite(t)) and np.all(np.isfinite(e))):
        raise ValueError("displacements and energies of the audit must be finite numbers")
    if not epsilon > 0.0:
        raise ValueError("the tolerance epsilon must be larger than zero (found %r)" % (epsilon,))
    threshold = float(max(2.0 * float(epsilon), float(floor)))
    out: Dict[str, object] = {"verdict": SKIPPED, "spread": None, "threshold": threshold, "candidates": [],
                              "quadratic": None, "cubic": None, "leave_one_out": [], "n_points": int(len(t)),
                              "sign": int(sign), "note": ""}
    if len(t) < int(min_points):
        out["note"] = "%d points; the audit needs at least %d" % (len(t), min_points)
        return out
    w = _weights(s)
    quad = fitted_extremum(t, e, w, 2, sign)
    cubic = fitted_extremum(t, e, w, 3, sign)
    loo = [fitted_extremum(t, e, w, 3, sign, leave_out=i) for i in range(len(t))]
    cand = [x for x in [quad, cubic] + loo if x is not None]
    out.update(quadratic=quad, cubic=cubic, leave_one_out=loo, candidates=cand)
    kind = "minimum" if sign > 0 else "maximum"
    if len(cand) < 3:
        out["verdict"] = UNRESOLVED
        out["note"] = "only %d of %d fits have a %s inside their points" % (len(cand), 2 + len(t), kind)
        return out
    spread = float(max(cand) - min(cand))
    out["spread"] = spread
    out["verdict"] = RESOLVED if spread <= threshold else UNRESOLVED
    out["note"] = "%d of %d fits have a %s inside their points; they span %.4f A (threshold %.4f A)" % (
        len(cand), 2 + len(t), kind, spread, threshold)
    return out


def audit_iteration(grids: Sequence[Sequence[float]], energies: Sequence[Sequence[float]],
                    errors: Sequence[Sequence[float]], epsilons: Sequence[float],
                    signs: Optional[Sequence[int]] = None, floor: float = 0.01, min_points: int = 5
                    ) -> List[Dict[str, object]]:
    """Audit of all directions of one classic iteration.

    ``grids[j]``, ``energies[j]`` and ``errors[j]`` hold the displacements, energies and error bars of the
    points of direction j (the shared centre is part of every direction). ``epsilons[j]`` is the
    tolerance of direction j in angstrom. ``signs[j]`` is +1 where the energy is minimized along the
    direction and -1 where it is maximized (default: +1 everywhere).

    Returns one dict per direction, see :func:`audit_direction`.
    """
    d = len(grids)
    if not (len(energies) == len(errors) == len(epsilons) == d):
        raise ValueError("the audit needs grids, energies, errors and tolerances for the same number of directions")
    if signs is None:
        signs = [1] * d
    if len(signs) != d:
        raise ValueError("%d signs for %d directions" % (len(signs), d))
    return [audit_direction(grids[j], energies[j], errors[j], float(epsilons[j]), sign=int(signs[j]), floor=floor,
                            min_points=min_points) for j in range(d)]


def unresolved(result: Sequence[Dict[str, object]]) -> List[int]:
    """Numbers of the directions whose verdict is not ``RESOLVED``."""
    return [j for j, r in enumerate(result) if r["verdict"] != RESOLVED]
