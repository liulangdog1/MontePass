"""Dense-scan protocol v1: the estimator and the stopping rules of an earlier application.

This module is a line-by-line port of the registered rules. It must not be
changed. Improvements belong in a separate protocol module.

Conventions of protocol v1
--------------------------
* ``t`` is a signed displacement in angstrom along a unit-norm Cartesian direction.
* The even part of a scan is modelled as ``E(t) = E0 + Lambda * t**2``. So
  ``Lambda`` (kcal/mol/A^2) is HALF of the usual force constant ``k`` in
  ``E = E0 + 0.5 * k * t**2``.
* The odd part is modelled as ``D(t) = E(+t) - E(-t) = b * t`` (kcal/mol), so the
  energy slope at the centre is ``b / 2``.
* The one-dimensional minimum is at ``m = -b / (4 * Lambda)``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from ..units import HARTREE_TO_KCAL as H


@dataclass(frozen=True)
class ProtocolV1:
    """Fixed constants of protocol v1 (identical for every system)."""

    e_cap: float = 5.0                          # kcal/mol: energy rise at the outer shell, R_j = sqrt(e_cap / Lambda_j)
    shells: Tuple[float, ...] = (0.5, 1.0)      # shell positions as fractions of R_j
    drop: int = 50                              # equilibration blocks discarded from every trace
    min_blocks_after_drop: int = 30             # a point is usable once it holds more than drop + this many blocks
    chunk: int = 100                            # blocks acquired per request
    cap: int = 400                              # largest number of blocks per point
    z: float = 4.0                              # rule A: significance of the slope
    chi2_max: float = 3.0                       # acceptance of the linear odd model (chi2 per degree of freedom)
    eps_e: float = 0.05                         # kcal/mol: rule B fallback threshold for one direction
    eps_total: float = 0.2                      # kcal/mol: convergence tolerance (all directions together)
    max_iter: int = 5

    def half_span(self, lam: float) -> float:
        """Outer-shell displacement R (angstrom) for a direction of curvature ``lam``."""
        return (self.e_cap / lam) ** 0.5


V1 = ProtocolV1()


def analyse(pairs: Sequence[Tuple[float, float, float, float, float]]) -> Dict[str, float]:
    """Odd/even estimator of one direction.

    ``pairs`` is a list of ``(t, E_plus, s_plus, E_minus, s_minus)`` with the
    energies and their standard errors in hartree. The result is in kcal/mol
    and angstrom: slope ``b`` of the odd part with error ``sb``, significance
    ``z``, chi-squared per degree of freedom of the linear odd model ``chi2``,
    curvature ``lam``, outer displacement ``R``, minimum ``m`` with error ``sm``,
    and, with three or more shells, the cubic odd model (``b1_cubic``,
    ``b3_cubic``, ``chi2_cubic``, ``m_cubic``).
    """
    T = np.array([p[0] for p in pairs])
    D = np.array([(p[1] - p[3]) * H for p in pairs]); S = np.array([np.hypot(p[2], p[4]) * H for p in pairs])
    Se = np.array([(p[1] + p[3]) / 2 * H for p in pairs])
    w = 1 / S ** 2
    b = float(np.sum(w * T * D) / np.sum(w * T ** 2)); sb = float(1 / np.sqrt(np.sum(w * T ** 2)))
    chi2 = float(np.sum(((D - b * T) / S) ** 2) / (len(T) - 1)) if len(T) > 1 else 0.0
    a = float(np.linalg.lstsq(np.vstack([T ** 2, np.ones_like(T)]).T, Se, rcond=None)[0][0]) if len(T) >= 2 else float('nan')
    out = dict(b=b, sb=sb, z=abs(b) / sb, chi2=chi2, lam=a, R=float(T.max()), npairs=len(T))
    if a > 0:
        out.update(m=-b / (4 * a), sm=sb / (4 * a))
    else:
        out.update(m=float('nan'), sm=float('nan'))
    if len(T) >= 3:      # two-parameter odd model D = b1 t + b3 t^3, minimiser consistent with the fitted curvature
        X = np.vstack([T, T ** 3]).T; C = np.linalg.inv(X.T @ np.diag(w) @ X); p = C @ (X.T @ np.diag(w) @ D)
        chi2c = float(np.sum(w * (D - X @ p) ** 2) / max(len(T) - 2, 1))
        b1, b3 = float(p[0]), float(p[1])
        disc = 4 * a * a - 3 * b1 * b3
        if a > 0 and b3 != 0 and disc > 0:
            r1 = (-2 * a + disc ** 0.5) / (3 * b3); r2 = (-2 * a - disc ** 0.5) / (3 * b3)
            out['m_cubic'] = r1 if abs(r1) < abs(r2) else r2
        out.update(chi2_cubic=chi2c, b1_cubic=b1, b3_cubic=b3)
    return out


def remaining_bound(res: Dict[str, float]) -> float:
    """Upper bound (kcal/mol) on the energy still to gain along one direction:
    ``Lambda * (|m| + 2 sigma_m)**2``. NaN when the curvature is not positive."""
    return res['lam'] * (abs(res['m']) + 2 * res['sm']) ** 2


def decide(res: Dict[str, float], blocks: int, cap: Optional[int] = None, eps_e: Optional[float] = None,
           protocol: ProtocolV1 = V1) -> Tuple[str, str]:
    """State of one direction after a chunk of blocks. Returns ``(status, note)``.

    status is one of
      ``CONTINUE``         request more blocks
      ``DISPLACED``        the minimum is displaced: step ``m``
      ``DISPLACED_CUBIC``  as above, with the cubic odd model: step ``m_cubic``
      ``ZERO``             nothing worth finding (rule B), or not resolved at the cap
      ``RESCUE``           add a shell at 1.5 R

    ``eps_e`` is the threshold of rule B for this direction. The caller passes
    ``eps_total / n_directions``, so that n directions stopped by rule B sum to
    less than ``eps_total``.
    """
    if cap is None:
        cap = protocol.cap
    if eps_e is None:
        eps_e = protocol.eps_e
    if np.isnan(res['m']):
        return ('CONTINUE' if blocks < cap else 'ZERO'), 'curvature not positive'
    if res['z'] >= protocol.z and res['chi2'] <= protocol.chi2_max:
        if abs(res['m']) >= res['R']:
            return 'RESCUE', 'rule A but bracket fails (|m| %.4f >= R %.3f)' % (abs(res['m']), res['R'])
        return 'DISPLACED', 'rule A at %d blocks (%.1f sigma, chi2/dof %.2f)' % (blocks, res['z'], res['chi2'])
    if res['lam'] * (abs(res['m']) + 2 * res['sm']) ** 2 < eps_e:
        return 'ZERO', 'rule B at %d blocks (bound %.4f < %.4f kcal/mol)' % (blocks, res['lam'] * (abs(res['m']) + 2 * res['sm']) ** 2, eps_e)
    if blocks < cap:
        return 'CONTINUE', ''
    # at the cap: registered verdict
    if res['chi2'] > protocol.chi2_max:
        if 'm_cubic' in res and res.get('chi2_cubic', 9) <= protocol.chi2_max and abs(res['m_cubic']) < res['R']:
            return 'DISPLACED_CUBIC', 'linear rejected (chi2/dof %.2f), IV.9a cubic accepted' % res['chi2']
        return 'RESCUE', 'linear model rejected at the cap (chi2/dof %.2f)' % res['chi2']
    if res['z'] >= 2.0:
        if abs(res['m']) >= res['R']:
            return 'RESCUE', '2 sigma but bracket fails'
        return 'DISPLACED', '2 sigma verdict at the cap (%.1f sigma)' % res['z']
    return 'ZERO', 'not resolved at the cap (%.1f sigma)' % res['z']
