"""Dense-scan protocol v2: a cheaper estimator, a joint stopping rule and safe steps.

Protocol v1 (the rules of an earlier application, module ``protocol_v1``) measures every
direction on two shells, decides every direction on its own, and starts from
nothing at every new centre. Protocol v2 keeps the idea of symmetric scans and
changes the following points. Each point can be switched off, so that the two
protocols can be compared term by term.

1. One shell and the centre. The slope needs only the outer pair of points. The
   curvature comes from the outer pair and the centre energy. This halves the
   number of scan points.
2. The surrogate as a control variate. Every DMC point also has a surrogate (DFT)
   energy at the same geometry, at no extra cost. The estimator works on the
   difference "DMC minus surrogate", which is smoother than the DMC energy, and
   adds the exact surrogate slope and curvature at the centre:

       g_j = g_sur_j + [ (E+ - E-) - (S+ - S-) ] / (2 t)
       k_j = k_sur_j + [ (E+ + E- - 2 E0) - (S+ + S- - 2 S0) ] / t**2

   The anharmonic part of the scan cancels to the extent that the surrogate has
   the same shape as the DMC surface. That holds near the stationary point of the
   surrogate. So the surrogate is used at a centre only while it is within the
   energy window of the scans from that point, sum_j g_sur_j^2 / (2 |k_sur_j|) <=
   cv_window (default: e_cap). At other centres the plain differences of the DMC
   energies are used, and neither the surrogate curvatures nor its couplings.
3. The measured curvature is combined with the surrogate curvature by their
   uncertainties, but only where the two agree. Where they disagree clearly, the
   measurement is used and the surrogate couplings are not trusted.
4. A joint stopping rule. With slope vector g, its covariance S and model Hessian H:

       G = 1/2 g^T |H|^-1 g            energy that the Newton step is predicted to gain
       N = 1/2 tr(|H|^-1 S)            expected energy error of that step from noise
       V = g^T |H|^-1 S |H|^-1 g + 1/2 tr((|H|^-1 S)^2)     variance of G

   * G - N - k_sig sqrt(V) > g_move : the gain is clear. Stop acquiring and move.
   * c_n N < eps_total              : the precision is reached (a joint 95 % statement for
                                      n directions, c_n = chi2(0.95, n) / n). If the predicted
                                      gain is small the search ends here, else it moves.
   One decision is made for all directions together. No direction is "gated".
5. Optional (``pool``, off by default): information is carried from one centre to
   the next. The difference "DMC slope minus surrogate slope" at the previous centre
   is used as a prior for the new centre, with an error that grows with the step.
   The slope error that enters S has a statistical part and a shape part. The shape part
   covers the case that the surrogate has another cubic term than the DMC surface along
   a scan; it is ``shape_mismatch`` times the cubic term of the surrogate, which is
   known exactly. More blocks do not reduce it. If it alone keeps the joint bound above
   the tolerance, the scan is repeated at the next centre with shorter spans (the term
   falls with the fourth power of the span in energy).
6. Safeguards for a start that is far from the minimum: a step is never longer than
   the trusted span; a step that raises the centre energy is rejected and halved;
   the trust factor follows the ratio of measured to predicted energy change; the
   surrogate couplings between directions are used only if they predict the slope
   change measured along the last step at least as well as the model without
   couplings; where the measured curvature has the wrong sign, the step goes downhill
   by the trusted span.
   Points do not wait for each other with their cores. When a batch system starts only
   some of the points, the others have no blocks yet and no decision can be made. A point
   that runs is then stopped as soon as it holds ``plan_factor`` times the number of usable
   blocks (blocks after the discarded ones) that the precision target is expected to need,
   computed from the noise of the points themselves. Its cores go to the points that wait.
   If the blocks of a scan do not reach the precision target (the block limit ``cap`` was too
   low for this centre), the search does not end: it takes the step and repeats the scan with
   more blocks per point. The number comes from the measured noise energy of the scan that
   was too short, and is at most ``cap_scan_factor`` times ``cap``.
   A force constant that was measured and found to disagree with the surrogate is remembered:
   at the next centres it takes the place of the surrogate value. Without this memory a scan
   with a short span, whose own curvature measurement is weak, fell back to the surrogate
   value; where that value is too small by a factor of 2 or more, the steps overshoot and the
   search oscillates (found on a model surrogate whose Hessian is rotated by 34 degrees).
7. Fewer equilibration blocks are discarded: the number is found for every trace
   (rule MSER-5) and limited to the range ``drop_min`` to ``drop_max``.
8. Symmetry. At a symmetric centre the slope along a direction with no totally
   symmetric part is exactly zero. Such a direction gets no pair of points. If an
   operation reverses the direction, one point measures its curvature, so that an
   unstable symmetric geometry is still found (the search then leaves the symmetry).

Conventions: ``g`` is the energy slope at the centre along a unit direction
(kcal/mol/A), ``k`` the force constant (kcal/mol/A^2, E = 0.5 k t^2), ``m = -g/k``
the one-dimensional stationary point. For the reaction coordinate of a saddle
search ``k`` is negative and the stationary point is a maximum along it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from ..units import HARTREE_TO_KCAL as H
from ..linalg import product

# chi-square 95 % points for 1..12 degrees of freedom (used for c_n = chi2(0.95, n) / n)
_CHI2_95 = [3.841, 5.991, 7.815, 9.488, 11.070, 12.592, 14.067, 15.507, 16.919, 18.307, 19.675, 21.026]


def joint_factor(n: int) -> float:
    """c_n = chi2(0.95, n) / n: factor that turns the expected noise energy N of n directions
    into a joint 95 % bound."""
    if n < 1:
        raise ValueError("n must be at least 1")
    if n <= len(_CHI2_95):
        return _CHI2_95[n - 1] / n
    # Wilson-Hilferty approximation for larger n
    return float((1.0 - 2.0 / (9.0 * n) + 1.6449 * np.sqrt(2.0 / (9.0 * n))) ** 3)


def bound_factor(nu: float) -> float:
    """chi2(0.95, nu) / nu for a number of degrees of freedom ``nu`` that need not be whole (at least 1).

    The noise energy of a step is a sum of terms N_j z_j^2 with standard normal z_j. Its 95 % point is c_n times
    its mean only if all N_j are equal. If the noise sits in few directions, the 95 % point is higher: it is that
    of a chi-square variable with the effective number nu = (sum N_j)^2 / sum N_j^2 (Satterthwaite). Between
    whole numbers the 95 % point is interpolated; it is close to a straight line in nu."""
    nu = float(nu)
    if not np.isfinite(nu) or nu <= 1.0:
        return _CHI2_95[0]
    lo = int(np.floor(nu))
    if lo >= len(_CHI2_95):
        return joint_factor(int(round(nu)))
    q = _CHI2_95[lo - 1] + (nu - lo) * (_CHI2_95[lo] - _CHI2_95[lo - 1])
    return float(q / nu)


@dataclass(frozen=True)
class ProtocolV2:
    e_cap: float = 5.0                          # kcal/mol: energy rise at the outer shell (surrogate curvature)
    e_cap_negative: float = 0.5                 # kcal/mol: energy drop at the outer shell along a reaction coordinate
    r_min: float = 0.02                         # angstrom: limits of every span
    r_max: float = 0.25
    shells: Tuple[float, ...] = (1.0,)          # shell positions as fractions of R_j
    drop: object = "auto"                       # equilibration blocks discarded: "auto" (MSER-5) or a fixed number
    drop_min: int = 10
    drop_max: int = 50
    min_blocks_after_drop: int = 30             # no decision before a point holds this many usable blocks
    cap: int = 400                              # largest number of blocks per point
    eps_total: float = 0.2                      # kcal/mol: tolerance on the energy of the final geometry (joint 95 %)
    k_sig: float = 2.0                          # sigma multiple in the "move early" test
    g_move: float = 0.5                         # kcal/mol: above this predicted gain the search scans again at the new centre
    max_iter: int = 10
    control_variate: bool = True
    cv_window: Optional[float] = None           # kcal/mol; None = e_cap. The surrogate is used at a centre only while it is
                                                # this close to its own stationary point: sum_j g_sur_j^2 / (2 |k_sur_j|)
                                                # <= cv_window (see surrogate_in_window)
    curvature_prior: float = 0.20               # relative uncertainty of the surrogate curvature
    curvature_consistency: float = 3.0          # measured and surrogate curvature are combined only within this many sigma
    coupling_tolerance: float = 0.5             # surrogate couplings are used only if |k_measured / k_surrogate - 1| is below this
    trust: float = 1.0                          # largest step in units of the scanned span
    trust_min: float = 0.25                     # limits of the adaptive trust factor
    trust_max: float = 4.0
    pool: bool = False                          # True: carry slope information to the next centre. Off by default: on
                                                # model surfaces it saved no blocks and made the stopping rule less
                                                # reliable for a start far from the minimum (see docs/protocols.md)
    pool_model_error: float = 0.3               # relative curvature error assumed when a slope is carried forward
    coupling_test: bool = True                  # use the surrogate couplings only if they predict the slope change of the
                                                # last step at least as well as the model without couplings
    coupling_fit: bool = False                  # True: fit the couplings between the directions to the slope changes that
                                                # DMC measured along the steps between centres, when the model Hessian
                                                # does not predict these changes (see fit_couplings). Off by default: on
                                                # the model with a turned surrogate surface it made the result worse
                                                # (11 of 16 runs within the tolerance, against 16 of 16 without it; see
                                                # docs/protocols.md)
    coupling_fit_z: float = 3.09                # the fit is made only if the model is rejected at this level (3.09: 99.9 %)
    coupling_fit_prior: float = 0.5             # width of the prior of a coupling K_ij, in units of sqrt(|K_ii K_jj|)
    coupling_fit_memory: int = 6                # number of steps between centres that the fit uses
    coupling_fit_floor: float = 0.1             # smallest allowed |eigenvalue| of the fitted Hessian, in units of its
                                                # diagonal (limits the growth of a step and of the bound to a factor 10)
    secant: bool = False                        # True: also correct the model Hessian with a rank-one (SR1) update from the
                                                # measured slope changes. Off by default: with noisy slopes the update can
                                                # give curvatures that are wrong by a factor of 10 (see docs/protocols.md)
    secant_memory: int = 3
    accept_test: bool = True                    # reject a step that raises the centre energy (minimum searches only)
    accept_sigma: float = 3.0
    shrink_final: bool = False                  # True: multiply the last step by max(0, 1 - N / G). Off by default:
                                                # the joint bound c_n N is a statement about the full step
    shape_mismatch: float = 0.3                 # relative difference assumed between the cubic part of the DMC surface
                                                # and of the surrogate along a scan (0 switches the shape term off)
    shape_mismatch_transfer: float = 0.5        # the same where the surrogate is not subtracted, but the cubic part that
                                                # it had along this direction at its own stationary point is added to the
                                                # pair slope (the anharmonic shape of a bond is nearly the same on both
                                                # surfaces near their own minima)
    shape_distrust: bool = False                # True: along a direction where DMC contradicted the force constant of
                                                # the surrogate, use shape_mismatch_plain in place of shape_mismatch.
                                                # Off by default: on the models with a turned surrogate Hessian it cost
                                                # up to 70 % more blocks and did not raise the number of runs within the
                                                # tolerance (see docs/protocols.md)
    shape_mismatch_plain: float = 1.0           # the same where the surrogate is NOT subtracted (a centre far from the
                                                # stationary point of the surrogate): the slope of a plain pair of DMC
                                                # points carries the whole cubic part of the DMC surface. Its size is
                                                # taken from the cubic part of the surrogate along the scan (0: no term)
    span_factor_min: float = 0.25               # a span is shortened to no less than this fraction of its normal length
    span_growth_max: float = 1.5                # after the first scan, a span is at most this multiple of the span that
                                                # the direction had in the first scan (0: no limit). A force constant
                                                # that was measured far too low would otherwise give a scan that
                                                # reaches far up the walls, and a step of the same length
    plan_factor: float = 3.0                    # a point is stopped when it holds plan_factor times the usable blocks that
                                                # the precision target is expected to need (0: points run until the decision).
                                                # The margin covers the error of the noise estimate, which comes from short traces
    final_gain_factor: float = 1.0              # the search ends at a centre only if the gain that the last step is
                                                # predicted to bring, G - N, is at most this many times ``eps_total``
                                                # (0: the limit is ``g_move``). The last step is not checked by a scan,
                                                # and its gain is only as good as the model Hessian
    endgame_sig: float = 1.0                    # with the precision reached and a predicted gain above the limit of the
                                                # last step: the search moves only if the gain is above that limit by
                                                # this many sigma, or no more blocks will come. Else it waits for more
                                                # blocks. (0: it moves at once, on a gain that may be noise)
    cap_scan_factor: float = 4.0                # a repeated scan may ask for up to this many times ``cap`` blocks per point
                                                # (1: never more than ``cap``; then a scan that is too short ends the search)
    rescan_margin: float = 1.5                  # margin on the blocks that a repeated scan asks for
    z_resolved: float = 2.0                     # a slope counts as resolved at this many sigma (used for downhill steps)
    final_series: int = 2                       # independent full-length runs at the start and at the final geometry
    symmetry: bool = True                       # use the point-group symmetry of each centre (module ``symmetry``)
    symmetry_tol: float = 1e-3                  # angstrom: two atoms closer than this count as the same place

    def half_span(self, lam: float) -> float:
        """Span R of a direction with curvature ``lam`` = k / 2 (kcal/mol/A^2)."""
        cap = self.e_cap if lam > 0 else self.e_cap_negative
        return float(min(self.r_max, max(self.r_min, (cap / abs(lam)) ** 0.5)))

    def plan_noise(self, n_directions: int, n_negative: int = 0) -> Tuple[float, float]:
        """Noise energy of a planned scan per squared point error (1 / kcal/mol), and the factor that turns it
        into a joint 95 % bound. A direction scanned at +-t with a point error s adds s^2 / (4 t^2 |k|) =
        s^2 / (8 e), where e is the energy at the scan point: ``e_cap`` for a positive curvature and
        ``e_cap_negative`` for a reaction coordinate. With the default values a reaction coordinate adds 10
        times the noise energy of another direction, so the factor is that of fewer effective directions
        (:func:`bound_factor`)."""
        n, m = int(n_directions), int(n_negative)
        if n < 1 or m < 0 or m > n:
            raise ValueError("plan_noise: need at least 1 direction and 0 <= n_negative <= n_directions")
        terms = [1.0 / (8.0 * self.e_cap)] * (n - m) + [1.0 / (8.0 * self.e_cap_negative)] * m
        total = float(sum(terms)); squares = float(sum(v * v for v in terms))
        return total, bound_factor(total * total / squares)

    def planned_blocks(self, n_directions: int, sigma_block: float, n_negative: int = 0) -> int:
        """Blocks per point at which the joint precision target is expected to be reached, for a noise
        of ``sigma_block`` kcal/mol per block: c s^2 sum_j 1 / (8 e_j) / eps_total, plus the discarded blocks
        (:meth:`plan_noise`). Without a reaction coordinate this is c_n n s^2 / (8 e_cap eps_total)."""
        total, factor = self.plan_noise(n_directions, n_negative)
        b = factor * total * sigma_block ** 2 / self.eps_total
        return int(min(self.cap, np.ceil(b) + self.drop_min + self.min_blocks_after_drop))


V2 = ProtocolV2()


def mser_drop(energy: np.ndarray, lo: int = 10, hi: int = 50, batch: int = 5) -> int:
    """Number of start-up blocks to discard, by the MSER-5 rule: the prefix whose removal makes the
    estimated variance of the mean smallest, limited to ``lo`` .. ``hi``."""
    x = np.asarray(energy, dtype=float)
    n = len(x)
    if n < 2 * batch + lo:
        return int(lo)
    best, best_d = None, lo
    last = min(hi, n // 2)
    for d in range(0, last + 1, batch):
        tail = x[d:]
        val = tail.var(ddof=1) / len(tail)
        if best is None or val < best:
            best, best_d = val, d
    return int(min(hi, max(lo, best_d)))


@dataclass
class Shell:
    """One symmetric pair of scan points. Energies and errors in hartree."""
    t: float
    e_plus: float
    s_plus: float
    e_minus: float
    s_minus: float
    sur_plus: Optional[float] = None
    sur_minus: Optional[float] = None


def combine_curvature(k_meas: float, sk_meas: float, k_sur: Optional[float], protocol: ProtocolV2,
                      s_prior: Optional[float] = None) -> Tuple[float, float, str]:
    """Force constant used for the step, its error, and where it comes from.

    * measured and surrogate values agree within ``curvature_consistency`` sigma: the weighted mean of the two;
    * they disagree and the measurement is clear (at least 2 sigma from zero): the measurement. The surrogate
      does not describe the surface at this centre. (Against a value from earlier centres, ``s_prior``, the
      measurement must also be at least as precise as that value, or smaller in size than it);
    * no usable measurement: the surrogate value.

    ``k_sur`` is the value that is known before the measurement. Its error is ``curvature_prior`` times its
    size, or ``s_prior`` when given (a force constant that was measured at an earlier centre).
    """
    p = protocol
    have_meas = np.isfinite(k_meas) and np.isfinite(sk_meas) and sk_meas > 0
    have_sur = k_sur is not None and np.isfinite(k_sur) and k_sur != 0.0
    if have_sur and s_prior is not None:
        sp = float(s_prior)
        if have_meas:
            if abs(k_meas - k_sur) <= p.curvature_consistency * np.hypot(sk_meas, sp):
                w1, w2 = 1.0 / sk_meas ** 2, 1.0 / sp ** 2
                return float((w1 * k_meas + w2 * k_sur) / (w1 + w2)), float(1.0 / np.sqrt(w1 + w2)), "combined with earlier centres"
            # They disagree. A clear measurement replaces the earlier value if it is at least as precise. A less
            # precise one replaces it only if it is the smaller of the two in size: a force constant that is too
            # large makes the predicted gain and the bound too small, so the search could end too early; one
            # that is too small only makes the search more careful. (Without this, an upward fluctuation of a
            # weak measurement would always win: it is then more than 2 sigma from zero.)
            if abs(k_meas) >= 2.0 * sk_meas and (sk_meas <= sp or (abs(k_meas) < abs(k_sur) and k_meas * k_sur > 0)):
                return float(k_meas), float(sk_meas), "measured"
        return float(k_sur), sp, "earlier centres"
    if have_sur and have_meas:
        sp = p.curvature_prior * abs(k_sur)
        if abs(k_meas - k_sur) <= p.curvature_consistency * np.hypot(sk_meas, sp):
            w1, w2 = 1.0 / sk_meas ** 2, 1.0 / sp ** 2
            return float((w1 * k_meas + w2 * k_sur) / (w1 + w2)), float(1.0 / np.sqrt(w1 + w2)), "combined"
        if abs(k_meas) >= 2.0 * sk_meas:
            return float(k_meas), float(sk_meas), "measured"
        return float(k_sur), float(sp), "surrogate"
    if have_sur:
        return float(k_sur), float(p.curvature_prior * abs(k_sur)), "surrogate"
    if np.isfinite(k_meas):
        return float(k_meas), float(sk_meas), "measured"
    return float("nan"), float("inf"), "none"


def chi2_point(dof: int, z: float) -> float:
    """Point of the chi-square distribution with ``dof`` degrees of freedom that a standard normal variable
    exceeds with the same probability as ``z`` (Wilson-Hilferty; z = 1.645: 95 %, z = 3.09: 99.9 %)."""
    dof = max(int(dof), 1)
    return float(dof * (1.0 - 2.0 / (9.0 * dof) + z * np.sqrt(2.0 / (9.0 * dof))) ** 3)


def fit_couplings(kdiag: np.ndarray, base: np.ndarray, pairs: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
                  roles: np.ndarray, protocol: ProtocolV2) -> Optional[Dict[str, object]]:
    """Couplings between the search directions, fitted to measured slope changes.

    A scan measures the slope and the force constant along every direction, but not the couplings K_ij between
    two directions. The step model takes them from the surrogate. If the surrogate couplings are wrong (a DMC
    surface that is turned against the surrogate surface), every step is wrong in the same way: the search
    then converges slowly, and the predicted gain and the noise energy are too small.

    The slope changes between centres hold the missing information. For a step ``s`` from one centre to the
    next, the slope changes by ``y = K s``. With the diagonal known from the scans,

        r_i = y_i - k_i s_i = sum_{j != i} K_ij s_j        (one equation per direction and step).

    ``pairs`` holds, for each step, (s, y, error of y, mean of the diagonal force constants at the two centres).
    ``base`` is the model Hessian without the fit; its off-diagonal part is the prior mean of the couplings,
    with a width of ``coupling_fit_prior * sqrt(|k_i k_j|)``. The fit is a weighted least-squares fit with
    this prior (it stays at the prior where the data say nothing).

    Returns None if the base model predicts the slope changes (chi-square below the ``coupling_fit_z`` point), if
    the fit does not at least halve chi-square, or if no safe Hessian results. Otherwise a dictionary with the
    off-diagonal matrix ``C`` and the two chi-square values. The fitted Hessian must have as many negative
    eigenvalues as ``roles`` has negative entries, and no eigenvalue smaller in size than
    ``coupling_fit_floor`` in units of the diagonal; the correction is halved up to 4 times to reach that."""
    p = protocol
    kdiag = np.asarray(kdiag, dtype=float); n = len(kdiag)
    if n < 2 or not pairs or not np.all(np.isfinite(kdiag)) or np.any(kdiag == 0.0):
        return None
    C0 = np.asarray(base, dtype=float).copy(); C0[np.diag_indices(n)] = 0.0
    index = [(i, j) for i in range(n) for j in range(i + 1, n)]
    rows, rhs, weights = [], [], []
    for s, y, sy, kd in pairs:
        s = np.asarray(s, dtype=float); r = np.asarray(y, dtype=float) - np.asarray(kd, dtype=float) * s - C0 @ s
        for i in range(n):
            if not (np.isfinite(sy[i]) and sy[i] > 0.0 and np.isfinite(r[i])):
                continue                        # a slope fixed by symmetry carries no information
            row = np.zeros(len(index))
            for c, (a, b) in enumerate(index):
                if a == i:
                    row[c] = s[b]
                elif b == i:
                    row[c] = s[a]
            rows.append(row); rhs.append(r[i]); weights.append(1.0 / sy[i] ** 2)
    dof = len(rows)
    if dof == 0:
        return None
    A = np.array(rows); r = np.array(rhs); w = np.array(weights)
    chi0 = float(np.sum(w * r * r))
    if chi0 <= chi2_point(dof, p.coupling_fit_z):
        return None
    tau = np.array([p.coupling_fit_prior * np.sqrt(abs(kdiag[a] * kdiag[b])) for a, b in index])
    lhs = product(A.T * w, A) + np.diag(1.0 / tau ** 2)
    delta = np.linalg.solve(lhs, A.T @ (w * r))
    chi1 = float(np.sum(w * (r - A @ delta) ** 2))
    if not chi1 < 0.5 * chi0:
        return None                             # couplings do not explain the slope changes
    want_negative = int(np.sum(np.asarray(roles) < 0))
    scale = np.sqrt(np.abs(kdiag))
    for shrink in (1.0, 0.5, 0.25, 0.125, 0.0625):
        C = C0.copy()
        for c, (a, b) in enumerate(index):
            C[a, b] += shrink * delta[c]; C[b, a] += shrink * delta[c]
        mu = np.linalg.eigvalsh((np.diag(kdiag) + C) / np.outer(scale, scale))
        if int(np.sum(mu < 0)) == want_negative and float(np.abs(mu).min()) >= p.coupling_fit_floor:
            return {"C": C, "chi2_base": chi0, "chi2_fit": chi1, "dof": dof, "shrink": shrink, "steps": len(pairs)}
    return None


def memory_error(k_memory: Tuple[float, float], protocol: ProtocolV2) -> float:
    """Error of a force constant that was measured at an earlier centre, for its use at a new centre: its
    statistical error, and at least ``curvature_prior`` times its size, because the force constant changes
    from centre to centre."""
    return float(max(abs(float(k_memory[1])), protocol.curvature_prior * abs(float(k_memory[0]))))


def surrogate_in_window(g_sur: Sequence[Optional[float]], k_sur: Sequence[Optional[float]],
                        protocol: "ProtocolV2") -> Tuple[bool, float]:
    """Whether the surrogate may be used at a centre, and its height above its own stationary point.

    The surrogate helps only where it has the shape of the DMC surface over the scans. That is the case near
    its own stationary point. Far from it the cubic and quartic parts of the surrogate are other ones (for a
    bond that is 0.2 A too long or too short the cubic part is 2 to 3 times larger), and subtracting the
    surrogate would add more error than it removes. The height is estimated from the surrogate slopes and
    force constants along the search directions, sum_j g_j^2 / (2 |k_j|) in kcal/mol, and compared with
    ``cv_window`` (default: ``e_cap``, the energy range that the scans cover). Without the numbers for this
    estimate the answer is True.
    """
    height = 0.0
    for g, k in zip(g_sur, k_sur):
        if g is None or k is None or not np.isfinite(g) or not np.isfinite(k) or k == 0.0:
            continue
        height += float(g) ** 2 / (2.0 * abs(float(k)))
    window = protocol.e_cap if protocol.cv_window is None else float(protocol.cv_window)
    return bool(height <= window), float(height)


def estimate(shells: Sequence[Shell], centre: Optional[Tuple[float, float]] = None,
             sur_centre: Optional[float] = None, g_sur: Optional[float] = None, k_sur: Optional[float] = None,
             prior_g: Optional[Tuple[float, float]] = None, role: int = 1, protocol: ProtocolV2 = V2,
             surrogate_ok: bool = True, k_memory: Optional[Tuple[float, float]] = None,
             shape_q: Optional[float] = None, shape_trusted: bool = True) -> Dict[str, float]:
    """Slope, curvature and stationary point of one direction.

    ``centre`` is (energy, error) of the scan centre in hartree, ``sur_centre`` its surrogate energy.
    ``g_sur`` and ``k_sur`` are the surrogate slope and force constant at the centre along the direction.
    ``prior_g`` is an optional (value, error) for the slope, carried over from the previous centre.
    ``role`` is +1 for a direction along which the energy is minimized and -1 for the reaction coordinate
    of a saddle search (the energy is maximized along it). ``surrogate_ok = False`` says that the surrogate
    is too far from its own stationary point at this centre (:func:`surrogate_in_window`): then it is not
    subtracted and its force constant is used only if no measured one exists. ``k_memory`` is (value, error)
    of the force constant that was measured at earlier centres, for a direction along which the measurement
    disagreed with the surrogate: it then takes the place of the surrogate value as the value that is known
    before the measurement. ``shape_q`` is the cubic part of the surrogate slope along this direction divided
    by the squared span, measured at the stationary point of the surrogate (kcal/mol/A^3). Where the surrogate
    is not subtracted, ``shape_q * t^2`` is added to the slope of the pair at +-t.

    Returns a dict. Main entries: slope ``g`` with error ``sg`` (kcal/mol/A), force constant ``k`` with
    error ``sk`` and its origin ``k_source``, one-dimensional stationary point ``m`` with error ``sm`` (A),
    span ``R``. ``curv_ok`` is False when the sign of the curvature does not fit the role: then no
    stationary point of the wanted kind is near along this direction.

    ``sg`` holds two parts: the statistical error ``sg_meas`` of the pair difference, and the shape term
    ``sg_shape`` = ``shape_mismatch`` * |``shape``|, where ``shape`` is the cubic part of the surrogate along
    the scan (see the comment in the code). ``sg_meas_total`` is their combination before the carried-over
    slope is used.
    """
    p = protocol
    T = np.array([s.t for s in shells], dtype=float)
    D = np.array([(s.e_plus - s.e_minus) * H for s in shells])
    # an error of exactly zero (a model without noise) gets a tiny value, so that the weights stay finite
    SD = np.maximum(np.array([np.hypot(s.s_plus, s.s_minus) * H for s in shells]), 1e-12)
    have_sur = all(s.sur_plus is not None and s.sur_minus is not None for s in shells)
    use_cv = bool(p.control_variate and surrogate_ok and have_sur and g_sur is not None)
    w = 1.0 / SD ** 2
    shape, sg_shape = float("nan"), 0.0
    if use_cv:
        Ds = np.array([(s.sur_plus - s.sur_minus) * H for s in shells])
        Dfit = D - Ds + 2.0 * g_sur * T          # surrogate odd part replaced by its exact linear term
        # The shape term. A slope from a symmetric pair at +-t is exact for a quadratic function. For another
        # function f its error is f3 t^2 / 6, with f3 the third derivative (plus higher terms). Here f is "DMC minus
        # surrogate". The cubic
        # part of the surrogate alone is known exactly: it is the difference between the surrogate slope at
        # the centre and the slope of the surrogate pair,
        #     c = g_sur - (S+ - S-) / (2 t).
        # The cubic part of the DMC surface is assumed to equal that of the surrogate within the relative
        # mismatch ``shape_mismatch``. That gives the slope a systematic uncertainty of shape_mismatch * |c|.
        c_each = g_sur - Ds / (2.0 * T)
        shape = float(np.sum(w * T ** 2 * c_each) / np.sum(w * T ** 2))
        # ``shape_trusted`` False: DMC contradicted the force constant of the surrogate along this direction at an
        # earlier centre. Then its cubic part is not trusted to 30 % either: the whole of it is the uncertainty.
        mismatch = p.shape_mismatch if (shape_trusted or not p.shape_distrust) else max(p.shape_mismatch, p.shape_mismatch_plain)
        sg_shape = float(mismatch * abs(shape))
    else:
        Dfit = D
        if shape_q is not None and np.isfinite(shape_q) and p.shape_mismatch_transfer:
            # The pair slope of a plain DMC pair carries the cubic part of the DMC surface over the span. Near its
            # own minimum the DMC surface has nearly the cubic part that the surrogate has near ITS minimum; that
            # one was measured there (shape_q). It is added, with a relative uncertainty.
            Dfit = D + 2.0 * float(shape_q) * T ** 3
            shape = float(np.sum(w * T ** 2 * (float(shape_q) * T ** 2)) / np.sum(w * T ** 2))
            sg_shape = float(p.shape_mismatch_transfer * abs(shape))
        elif have_sur and g_sur is not None and p.shape_mismatch_plain:
            # The surrogate is not subtracted here, so the pair slope has the error of the whole cubic part of
            # the DMC surface over the span. The surrogate tells how large such a part is along this scan.
            Ds = np.array([(s.sur_plus - s.sur_minus) * H for s in shells])
            c_each = g_sur - Ds / (2.0 * T)
            shape = float(np.sum(w * T ** 2 * c_each) / np.sum(w * T ** 2))
            sg_shape = float(p.shape_mismatch_plain * abs(shape))
    b = float(np.sum(w * T * Dfit) / np.sum(w * T ** 2))
    sb = float(1.0 / np.sqrt(np.sum(w * T ** 2)))
    chi2 = float(np.sum(((Dfit - b * T) / SD) ** 2) / (len(T) - 1)) if len(T) > 1 else 0.0
    g_meas, sg_meas = b / 2.0, sb / 2.0
    sg_total = float(np.hypot(sg_meas, sg_shape))        # statistical error and shape term of this measurement
    g, sg = g_meas, sg_total
    pooled = False
    if prior_g is not None and prior_g[1] > 0 and np.isfinite(prior_g[0]):
        # the carried-over slope is used only if it agrees with the new measurement within 3 sigma
        if abs(prior_g[0] - g_meas) <= 3.0 * np.hypot(prior_g[1], sg_total):
            w1, w2 = 1.0 / sg_total ** 2, 1.0 / prior_g[1] ** 2
            g = (w1 * g_meas + w2 * prior_g[0]) / (w1 + w2); sg = float(1.0 / np.sqrt(w1 + w2)); pooled = True

    # curvature: even part against the centre, corrected by the surrogate even part
    k_meas, sk_meas = float("nan"), float("inf")
    if centre is not None:
        e0, s0 = centre
        vals, errs = [], []
        for s in shells:
            ev = (s.e_plus + s.e_minus - 2.0 * e0) * H
            se = np.sqrt(s.s_plus ** 2 + s.s_minus ** 2 + 4.0 * s0 ** 2) * H
            if use_cv and sur_centre is not None and k_sur is not None:
                ev = ev - (s.sur_plus + s.sur_minus - 2.0 * sur_centre) * H + k_sur * s.t ** 2
            vals.append(ev / s.t ** 2); errs.append(se / s.t ** 2)
        ww = 1.0 / np.maximum(np.array(errs), 1e-12) ** 2
        k_meas = float(np.sum(ww * np.array(vals)) / ww.sum()); sk_meas = float(1.0 / np.sqrt(ww.sum()))
    elif len(T) >= 2:
        Se = np.array([(s.e_plus + s.e_minus) / 2.0 * H for s in shells])
        a = float(np.linalg.lstsq(np.vstack([T ** 2, np.ones_like(T)]).T, Se, rcond=None)[0][0])
        k_meas, sk_meas = 2.0 * a, float("inf")          # no error estimate: used only when no surrogate value exists
    if k_memory is not None and np.isfinite(k_memory[0]) and k_memory[0] != 0.0:
        k, sk, k_source = combine_curvature(k_meas, sk_meas, float(k_memory[0]), p, s_prior=memory_error(k_memory, p))
    else:
        k_prior = k_sur if (surrogate_ok or not (np.isfinite(k_meas) and np.isfinite(sk_meas))) else None
        k, sk, k_source = combine_curvature(k_meas, sk_meas, k_prior, p)
    R = float(T.max())
    curv_ok = bool(np.isfinite(k) and k != 0.0 and np.sign(k) == np.sign(role))
    out = dict(g=float(g), sg=float(sg), g_meas=float(g_meas), sg_meas=float(sg_meas), pooled=pooled,
               shape=shape, sg_shape=sg_shape, sg_meas_total=sg_total,
               k=float(k), sk=float(sk), k_meas=float(k_meas), sk_meas=float(sk_meas), k_source=k_source,
               k_sur=float(k_sur) if k_sur is not None else float("nan"),
               g_sur=float(g_sur) if g_sur is not None else float("nan"),
               chi2=chi2, R=R, npairs=int(len(T)), control_variate=use_cv, role=int(role), curv_ok=curv_ok,
               z=float(abs(g) / sg))
    if curv_ok:
        out.update(m=float(-g / k), sm=float(sg / abs(k)))
    else:
        out.update(m=float("nan"), sm=float("nan"))
    return out


def joint_statistics(g: np.ndarray, sg: np.ndarray, hessian: np.ndarray) -> Dict[str, float]:
    """The three energies of the joint rule (kcal/mol) for slope vector ``g`` with errors ``sg`` and
    model Hessian ``hessian`` (the absolute values of its eigenvalues are used)."""
    g = np.asarray(g, dtype=float); sg = np.asarray(sg, dtype=float)
    Hm = np.asarray(hessian, dtype=float)
    if not (np.all(np.isfinite(g)) and np.all(np.isfinite(sg)) and np.all(np.isfinite(Hm))):
        raise FloatingPointError("the slopes, their errors or the model Hessian hold a value that is not a finite number")
    w, Q = np.linalg.eigh(0.5 * (Hm + Hm.T))
    aw = np.maximum(np.abs(w), 1e-8 * max(1.0, float(np.abs(w).max())))
    A = product(Q / aw, Q.T)                            # |H|^-1
    S = np.diag(sg ** 2)
    AS = product(A, S)
    Ag = product(A, g)
    G = 0.5 * float(np.dot(g, Ag))
    N = 0.5 * float(np.trace(AS))
    AS2 = float(np.trace(product(AS, AS)))
    V = float(np.dot(Ag, sg ** 2 * Ag)) + 0.5 * AS2
    # effective number of directions that carry the noise energy (between 1 and the number of directions)
    nu = float(np.trace(AS)) ** 2 / AS2 if AS2 > 0.0 else float(len(g))
    return {"G": G, "N": N, "V": max(V, 0.0), "nu": float(min(max(nu, 1.0), len(g)))}


def joint_decision(stats: Dict[str, float], n: int, final: bool, protocol: ProtocolV2 = V2) -> Tuple[str, str]:
    """Decision for one centre from the joint statistics. Returns ``(verdict, note)``.

    verdict is ``CONTINUE`` (acquire more blocks), ``MOVE`` (take the step and scan again),
    or ``FINAL`` (take the last, shrunken step and end the search). ``final`` says that no more
    blocks will come for this centre.
    """
    p = protocol
    G, N, V = stats["G"], stats["N"], stats["V"]
    cn = bound_factor(stats["nu"]) if "nu" in stats else joint_factor(n)
    if G - N - p.k_sig * np.sqrt(V) > p.g_move:
        return "MOVE", "clear gain: G - N - %.0f sqrt(V) = %.3f > %.2f kcal/mol" % (
            p.k_sig, G - N - p.k_sig * np.sqrt(V), p.g_move)
    if cn * N < p.eps_total:
        g_final = p.g_move if not p.final_gain_factor else min(p.g_move, p.final_gain_factor * p.eps_total)
        if G - N > g_final:
            # A step from here gains more than the last step may. But G is a noisy number: near the stationary point
            # a search that moved on every such G would wander at the level of the noise. So the gain must be
            # resolved, or the blocks used up; until then more blocks are taken at this centre.
            if final or not p.endgame_sig or G - N - p.endgame_sig * np.sqrt(V) > g_final:
                return "MOVE", "precision reached (c_n N = %.3f) and gain G - N = %.3f > %.3g kcal/mol" % (cn * N, G - N, g_final)
            return "CONTINUE", ""
        return "FINAL", "precision reached: joint 95 %% bound c_n N = %.3f < %.2f kcal/mol; predicted gain %.3f" % (
            cn * N, p.eps_total, max(G - N, 0.0))
    if final:
        if G - N > p.g_move:
            return "MOVE", "no more blocks; gain G - N = %.3f > %.2f kcal/mol" % (G - N, p.g_move)
        return "FINAL", "no more blocks: joint 95 %% bound c_n N = %.3f is above %.2f kcal/mol" % (cn * N, p.eps_total)
    return "CONTINUE", ""
