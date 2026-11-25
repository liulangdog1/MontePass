"""Driver of dense-scan protocol v2 (see ``protocol_v2`` for the rules).

One decision is made per centre, for all directions together. The points of an
iteration keep running until that decision; then they are all stopped and the
search either moves the centre or ends.

At the end of the search the driver requests full-length runs at the final
geometry. Together with the full-length runs at the starting geometry they give
the energy change of the optimization with a small error bar.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from ..engine.base import DONE, FAILED, QUEUED, RUNNING, STOPPED, PointSpec
from ..stats import reblock
from ..subspace import max_coupling
from ..units import HARTREE_TO_KCAL
from . import protocol_v2 as P2
from . import symmetry as SYM
from .dense import (CONVERGED, NEEDS_ATTENTION, NOT_CONVERGED, RUNNING_STATUS, DenseScanSearch, _point_name, _tkey,
                    combine_series)

FINALIZING = "FINALIZING"        # the search has ended; the full-length runs at the final geometry are running


def _clean(d: dict) -> dict:
    return {k: (None if (isinstance(v, float) and not np.isfinite(v)) else v) for k, v in d.items()}


class DenseScanSearchV2(DenseScanSearch):
    """Protocol v2. Same constructor as :class:`DenseScanSearch`."""

    # ------------------------------------------------------------------ hooks of the base class
    def _drop(self, energy: np.ndarray) -> int:
        p = self.protocol
        if p.drop == "auto":
            return P2.mser_drop(energy, p.drop_min, p.drop_max)
        return int(p.drop)

    def _stat(self, key: str, n: Optional[int] = None) -> Optional[Tuple[float, float, int]]:
        """As in the base class. One addition: a run that has ENDED with so few blocks that the discard chosen by
        the MSER rule would leave no usable data gets a smaller discard (not below ``drop_min``). Such a run
        will never have more blocks, so the search could not continue otherwise."""
        p = self.protocol
        d = self.engine.read(key)
        m = d.nblocks if n is None else min(n, d.nblocks)
        drop = self._drop(d.energy[:m])
        keep = int(p.min_blocks_after_drop)
        if m <= drop + keep and p.drop == "auto" and m == d.nblocks and d.state in (DONE, STOPPED):
            drop = max(int(p.drop_min), m - keep - 1)
        if m <= drop + keep:
            return None
        mean, err = reblock(d.energy[drop:m], d.weight[drop:m])
        return mean, err, m

    def _scaled_span(self, j: int, R: float) -> float:
        # shorter spans where the shape term of the slope was too large at the last centre (see _decide)
        p = self.protocol
        factors = self.state.get("span_factor")
        wanted = (self.state.get("span_wanted") or {}).get(str(j))
        if factors and wanted is not None and R > 0.0:
            # The shape term was measured with the span of the last scan, and the rule gave the span that is
            # wanted now. The normal span R of this centre comes from a new force constant, so the factor is
            # set here, relative to R.
            factors[j] = float(min(1.0, max(p.span_factor_min, float(wanted) / R)))
            del self.state["span_wanted"][str(j)]
        f = 1.0 if not factors else float(factors[j])
        return float(max(p.r_min, f * R)) if f < 1.0 else float(R)

    def _direction_span(self, j: int, lam: Optional[float], previous: Optional[float]) -> float:
        span = super()._direction_span(j, lam, previous)
        first = (self.state.get("iters") or {}).get("1")
        limit = float(self.protocol.span_growth_max or 0.0)
        if first is not None and limit > 0.0 and len(first.get("half_span") or []) > j:
            span = min(span, limit * float(first["half_span"][j]))
        return float(span)

    def _shape_energies(self, ests: List[dict]) -> np.ndarray:
        """Energy (kcal/mol) that the shape term of each slope can cost: sg_shape^2 / (2 |k|)."""
        out = np.zeros(self.dim)
        for j, e in enumerate(ests):
            kk = abs(float(e["k"])) if np.isfinite(e["k"]) else 0.0
            if kk > 0.0:
                out[j] = float(e.get("sg_shape", 0.0)) ** 2 / (2.0 * kk)
        return out

    def _shape_target(self) -> float:
        """Share of the tolerance that the shape term of one direction may take: a quarter of the joint
        tolerance, divided evenly over the directions."""
        n = self.dim
        return float(self.protocol.eps_total / (4.0 * n * P2.joint_factor(n)))

    def _update_span_factors(self, it: dict, ests: List[dict]) -> List[int]:
        """Shorten the span of a direction whose shape term took more than its share at this centre, and let a
        shortened span grow back when the term is far below its share. The shape energy falls with the fourth
        power of the span. Returns the directions that were shortened."""
        p = self.protocol
        factors = list(self.state.get("span_factor") or [1.0] * self.dim)
        target = self._shape_target()
        energies = self._shape_energies(ests)
        shortened = []
        wanted = {}
        for j in range(self.dim):
            if energies[j] > target:
                new = max(p.span_factor_min, factors[j] * (target / energies[j]) ** 0.25)
                if new < factors[j] - 1e-9 and float(it["half_span"][j]) > p.r_min + 1e-9:
                    # the span that the fourth-power rule asks for, from the span that this scan had
                    wanted[str(j)] = float(it["half_span"][j]) * float((target / energies[j]) ** 0.25)
                    factors[j] = float(new); shortened.append(j)
            elif factors[j] < 1.0 and energies[j] < target / 16.0:
                grow = 2.0 if energies[j] <= 0.0 else min(2.0, (target / energies[j]) ** 0.25)
                factors[j] = float(min(1.0, factors[j] * grow))
        self.state["span_factor"] = factors
        self.state["span_wanted"] = wanted
        return shortened

    def _n_centre_series(self, k: int) -> int:
        # full-length reference runs at the starting geometry; one run at every later centre
        return int(self.protocol.final_series) if k == 1 else 1

    def _new_iteration(self, k, centre, lam, half_span_prev) -> None:
        super()._new_iteration(k, centre, lam, half_span_prev)
        it = self._it(k)
        it.update({"base": None, "accepted": None, "step_scale": 1.0, "prior": None, "closed": False,
                   "decision": None})
        # blocks per scan point: the block limit, or more after a scan that was too short (see _decide)
        it["point_blocks"] = int(self.state.get("scan_blocks") or self.protocol.cap)
        it["usable_wanted"] = self.state.get("usable_wanted")
        self.state.setdefault("roles", None)
        self.state.setdefault("trust_scale", 1.0)
        self.state.setdefault("rejected", 0)
        if self.state["roles"] is None:
            # role of every direction, from the sign of the surrogate curvature at the start:
            # +1 = the energy is minimized along it, -1 = maximized (reaction coordinate of a saddle search)
            self.state["roles"] = [1 if (v is not None and v > 0) else -1 for v in lam]
        # symmetry of this centre: directions whose slope is zero by symmetry need no pair of points
        classes = ["symmetric"] * self.dim
        order = 1
        if getattr(self.protocol, "symmetry", False):
            classes, order = SYM.classify_directions(self.vectors, self.symbols, np.array(it["centre"]),
                                                     tol=self.protocol.symmetry_tol)
        it["classes"], it["group_order"] = classes, int(order)
        for j, cls in enumerate(classes):
            pts = it["points"][str(j)]
            if cls == "odd":
                for tk in [tk for tk in pts if float(tk) < 0]:
                    del pts[tk]                 # E(-t) = E(+t): one side is enough for the curvature
            elif cls == "zero":
                pts.clear()
        if order > 1:
            self.say("iteration %d: the centre has %d symmetry operations; directions: %s" % (
                k, order, ", ".join("%d %s" % (j, c) for j, c in enumerate(classes))))

    def _status_after_attention(self) -> str:
        return FINALIZING if self.state.get("final") else RUNNING_STATUS

    def _k_memory(self, j: int) -> Optional[Tuple[float, float]]:
        """Force constant of direction ``j`` that was measured at earlier centres and disagreed with the
        surrogate there (value, error), or None."""
        entry = (self.state.get("k_memory") or {}).get(str(j))
        return None if entry is None else (float(entry[0]), float(entry[1]))

    def _k_prior(self, it: dict, j: int, cv_ok: bool) -> Optional[Tuple[float, float]]:
        """What is known about the force constant of direction j before this scan, apart from the surrogate at
        this centre: the remembered value (:meth:`_k_memory`); or, at a centre where the surrogate is not used,
        the force constant that the surrogate has along this direction at its own stationary point, with an
        error of its whole size. Without it a measurement that is mostly noise would be used as it is there.
        (The value of the previous centre is NOT used for this: where that one had the wrong sign, the short
        scans that follow could never correct it.)"""
        memory = self._k_memory(j)
        if memory is None and not cv_ok:
            first = self._it(1)["lam"][j]
            if first is not None and np.isfinite(first) and first != 0:
                memory = (2.0 * float(first), 2.0 * abs(float(first)))
        return memory

    def _shape_q(self, j: int) -> Optional[float]:
        """Cubic part of the surrogate slope along direction j per squared span, kept from the first centre at
        which the surrogate was subtracted (its own stationary point), or None."""
        q = (self.state.get("shape_q") or {}).get(str(j))
        return None if q is None else float(q)

    def _remember_shape(self, it: dict, ests: List[dict]) -> None:
        if self.state.get("shape_q") is not None or not it.get("surrogate_ok"):
            return
        kept = {}
        for j, e in enumerate(ests):
            R = float(e.get("R") or 0.0)
            if e.get("control_variate") and np.isfinite(e.get("shape", float("nan"))) and R > 0.0:
                kept[str(j)] = float(e["shape"]) / (R * R)
        if kept:
            self.state["shape_q"] = kept

    def _remember_curvatures(self, ests: List[dict], gain: float) -> None:
        """After a decision: keep the measured force constants along the directions where the surrogate value
        was contradicted, and update the ones that are kept already.

        A force constant is a local property. It is kept only while the search is near the stationary point:
        the predicted gain of this centre is at most ``cv_window`` (the energy range of the scans), and the
        sign of the force constant fits the direction. Far from the stationary point, or beyond an inflection
        point, nothing is kept: a remembered value from there would be wrong at the next centres, and it could
        not be corrected by scans with short spans."""
        p = self.protocol
        memory = self.state.setdefault("k_memory", {})
        window = p.e_cap if p.cv_window is None else float(p.cv_window)
        if not np.isfinite(gain) or gain > window:
            memory.clear()
            self.state.pop("couplings", None)
            return
        for j, e in enumerate(ests):
            source = e.get("k_source")
            if not e.get("curv_ok"):
                memory.pop(str(j), None)
            elif source == "measured" and np.isfinite(e.get("sk_meas", float("inf"))):
                memory[str(j)] = [float(e["k_meas"]), float(e["sk_meas"])]
            elif source == "combined with earlier centres":
                memory[str(j)] = [float(e["k"]), float(e["sk"])]

    def _point_blocks(self) -> int:
        return int(self._it().get("point_blocks") or self.protocol.cap)

    # ------------------------------------------------------------------ points
    def _stop_iteration(self, k: int, it: dict) -> int:
        """Stop every point of an iteration that still runs. The reference runs of iteration 1 go on."""
        n = 0
        keys = [key for pts in it["points"].values() for key in pts.values()]
        if k != 1:
            keys += list(it["centre_keys"])
        for key in keys:
            if self.engine.read(key).state in (RUNNING, QUEUED):
                self.engine.stop(key); n += 1
        it["closed"] = True
        return n

    def blocks_needed(self, it: dict, sigma_block: float) -> float:
        """Usable blocks per point at which the joint bound is expected to equal the tolerance, for a noise of
        ``sigma_block`` kcal/mol per block: c * sum_j sigma^2 / (4 t_j^2 |k_j|) / eps_total. The sum runs over
        the directions that are scanned on both sides, with their spans t_j and force constants k_j. The factor c
        is ``bound_factor`` of the effective number of directions that carry the noise."""
        p = self.protocol
        classes = it.get("classes") or ["symmetric"] * self.dim
        total, squares = 0.0, 0.0
        for j in range(self.dim):
            lam, t = it["lam"][j], float(it["half_span"][j])
            if classes[j] in ("odd", "zero") or lam is None or lam == 0 or t <= 0:
                continue
            term = sigma_block ** 2 / (4.0 * t * t * abs(2.0 * float(lam)))
            total += term; squares += term * term
        return float(P2.bound_factor(total ** 2 / squares if squares > 0.0 else 1.0) * total / p.eps_total)

    def _trim_points(self, k: int, it: dict) -> None:
        """Stop a running point that holds more blocks than the decision can need.

        The decision is made for all directions together, so it waits until every direction has data. If a
        batch system starts only some of the points, those points would run on to the block limit while the
        others wait for cores. Here the noise of one block is measured on the points that have data (root
        mean square over the points), and a point is stopped when it holds ``plan_factor`` times the usable
        blocks that the precision target needs, plus ``drop_max`` blocks. The reference runs of iteration 1
        are not stopped."""
        p = self.protocol
        if not p.plan_factor or it.get("closed"):
            return
        keys = [key for pts in it["points"].values() for key in pts.values()]
        if k != 1:
            keys += list(it["centre_keys"])
        noise, running = [], []
        for key in keys:
            d = self.engine.read(key)
            st = self._stat(key)
            usable = 0
            if st is not None:
                usable = st[2] - self._drop(d.energy[:st[2]])
                # two estimates of the noise of one block, the larger one is used: the error bar of the point
                # (it knows the correlation between blocks, but the choice of the discarded blocks makes it
                # too small for a short trace) and the plain scatter of the blocks (no correlation)
                plain = float(np.std(d.energy[int(p.drop_min):st[2]], ddof=1))
                noise.append(max(st[1] * np.sqrt(max(usable, 1)), plain) * HARTREE_TO_KCAL)
            if d.state == RUNNING:
                running.append((key, d.nblocks, usable))
        if not noise or not running:
            return
        sigma = float(np.sqrt(np.mean(np.square(noise))))
        needed = self.blocks_needed(it, sigma)
        # after a scan that was too short, the number that the measured noise energy of that scan asks for counts too
        usable = max(int(np.ceil(p.plan_factor * needed)), int(p.min_blocks_after_drop) + 1, int(it.get("usable_wanted") or 0))
        # The two points of a pair are read at their common number of blocks, and the number of discarded
        # blocks of a trace can be anything up to drop_max there. So a point is stopped at a total number of
        # blocks that leaves enough usable blocks with the largest discard: then a pair of stopped points
        # always has usable data.
        limit = usable + int(p.drop_max)
        if limit >= self._point_blocks():
            return                                  # the block limit of the points comes first
        trimmed = it.setdefault("trimmed", {})
        for key, n, _usable in running:
            if n >= limit and key not in trimmed:
                self.engine.stop(key)
                trimmed[key] = int(n)
                self.say("iteration %d: point %s stopped at %d blocks (limit %d: %d times the %d usable blocks that the "
                         "decision is expected to need, plus %d that may be discarded; noise of one block %.2f kcal/mol)"
                         % (k, key, n, limit, int(round(p.plan_factor)), int(np.ceil(needed)), int(p.drop_max), sigma))

    def _centre_extension(self, it: dict) -> None:
        first = self.engine.read(it["centre_keys"][0])
        if first.nblocks == 0:
            return
        for s, key in enumerate(it["centre_keys"][1:], start=2):
            self.engine.submit(PointSpec(key=key, symbols=self.symbols, coords=np.array(it["centre"]),
                                         blocks=int(self.protocol.cap), role="centre", series=s,
                                         base_key=it["centre_keys"][0], settings=self.settings))

    # ------------------------------------------------------------------ one pass
    def tick(self) -> str:
        if self.state["status"] == FINALIZING:
            self._tick_final()
            self._save()
            return self.state["status"]
        if self.state["status"] != RUNNING_STATUS:
            return self.state["status"]
        k = self.state["iteration"]; it = self._it()
        self._ensure_points(k, it)
        self._centre_extension(it)
        self._trim_points(k, it)
        if not self._reject_step(k, it):
            data = self._gather(k, it)
            if data is not None:
                self._decide(k, it, data)
        self._save()
        return self.state["status"]

    def _gather(self, k: int, it: dict) -> Optional[dict]:
        """Estimates of all directions from the blocks written so far, or None if a direction has no usable data yet."""
        p = self.protocol
        sur = self._surrogate_now(it)
        if p.control_variate and self.surrogate is not None and sur is None:
            raw = self.surrogate.result(it["surrogate_tag"])
            if raw is not None and raw.failed:
                self.state["status"] = NEEDS_ATTENTION
                self.state["message"] = "iteration %d: the surrogate calculation at the centre failed: %s" % (k, raw.message)
                self.say(self.state["message"])
                return None
            self.say("iteration %d: waiting for the surrogate gradient and Hessian at the centre" % k)
            return None
        for key in it["centre_keys"]:
            cd = self.engine.read(key)
            if cd.state == FAILED:
                self.state["status"] = NEEDS_ATTENTION
                self.state["message"] = "iteration %d: the run at the centre failed (%s): %s" % (k, key, cd.message)
                self.say(self.state["message"])
                return None
        centre = self._centre_stat(it)
        e0_sur = self.engine.read(it["centre_keys"][0]).e_surrogate
        ests, blocks, running = [], 10 ** 9, False
        classes = it.get("classes") or ["symmetric"] * self.dim
        # is the surrogate close enough to its own stationary point to be used at this centre?
        cv_ok = True
        if p.control_variate and sur is not None and sur.gradient is not None:
            g_all = [float(self.vectors[j] @ sur.gradient.reshape(-1)) for j in range(self.dim)]
            k_all = [self._surrogate_curvature(sur, j) for j in range(self.dim)]
            cv_ok, height = P2.surrogate_in_window(g_all, k_all, p)
            if it.get("surrogate_ok") is None:
                it["surrogate_ok"], it["surrogate_height"] = bool(cv_ok), float(height)
                if not cv_ok:
                    self.say("iteration %d: the surrogate is %.1f kcal/mol above its own stationary point at this centre "
                             "(limit %.1f). It is not subtracted from the DMC energies here, and its curvatures and "
                             "couplings are not used" % (k, height, p.e_cap if p.cv_window is None else p.cv_window))
        for j in range(self.dim):
            pts = self._direction_points(it, j)
            if classes[j] in ("odd", "zero"):
                est = self._symmetry_estimate(k, it, j, classes[j], pts, sur, centre, e0_sur, cv_ok)
                if est is None:
                    return None
                if est.get("_running"):
                    running = True
                if "_blocks" in est:
                    blocks = min(blocks, est.pop("_blocks"))
                est.pop("_running", None)
                ests.append(est)
                continue
            ts = sorted(t for t in pts if t > 0 and -t in pts)
            shells = []
            for t in ts:
                dp, dm = self.engine.read(pts[t]), self.engine.read(pts[-t])
                for d in (dp, dm):
                    if d.state in (RUNNING, QUEUED):
                        running = True
                    if d.state == FAILED:
                        self.state["status"] = NEEDS_ATTENTION
                        self.state["message"] = "iteration %d direction %d: point %s failed" % (k, j, d.key)
                        self.say(self.state["message"]); return None
                n = min(dp.nblocks, dm.nblocks)
                sp, sm = self._stat(pts[t], n), self._stat(pts[-t], n)
                if sp is None or sm is None:
                    if dp.state in (DONE, STOPPED) and dm.state in (DONE, STOPPED):
                        self.state["status"] = NEEDS_ATTENTION
                        self.state["message"] = "iteration %d direction %d: points ended with too few blocks" % (k, j)
                        self.say(self.state["message"])
                    else:
                        self.say("iteration %d: waiting for blocks (direction %d has %d)" % (k, j, n))
                    return None
                blocks = min(blocks, n)
                shells.append(P2.Shell(t=t, e_plus=sp[0], s_plus=sp[1], e_minus=sm[0], s_minus=sm[1],
                                       sur_plus=dp.e_surrogate, sur_minus=dm.e_surrogate))
            u = self.vectors[j]
            g_sur = float(u @ sur.gradient.reshape(-1)) if (sur is not None and sur.gradient is not None) else None
            k_sur = self._surrogate_curvature(sur, j)
            if k_sur is None and it["lam"][j] is not None:
                k_sur = 2.0 * it["lam"][j]
            prior = None
            if p.pool and it.get("prior"):
                pr = it["prior"]
                if pr["kind"] == "d" and g_sur is not None:
                    prior = (g_sur + pr["value"][j], pr["error"][j])     # surrogate slope here + carried difference
                elif pr["kind"] == "g":
                    prior = (pr["value"][j], pr["error"][j])
            ests.append(P2.estimate(shells, centre=centre, sur_centre=e0_sur, g_sur=g_sur, k_sur=k_sur,
                                    prior_g=prior, role=self.state["roles"][j], protocol=p, surrogate_ok=cv_ok,
                                    k_memory=self._k_prior(it, j, cv_ok), shape_q=self._shape_q(j),
                                    shape_trusted=self._k_memory(j) is None))
        if blocks == 10 ** 9:                   # no direction has scan points (all slopes are zero by symmetry)
            blocks = 0
        return {"ests": ests, "blocks": int(blocks), "running": running}

    def _symmetry_estimate(self, k, it, j, cls, pts, sur, centre, e0_sur, cv_ok: bool = True) -> Optional[dict]:
        """Estimate of a direction whose slope is zero by symmetry. "odd": the curvature is measured with the
        one point at +R and the centre. "zero": the surrogate curvature is used."""
        p = self.protocol
        role = self.state["roles"][j]
        k_sur = self._surrogate_curvature(sur, j)
        if k_sur is None and it["lam"][j] is not None:
            k_sur = 2.0 * it["lam"][j]
        k_meas, sk_meas = float("nan"), float("inf")
        extra = {}
        R = float(it["half_span"][j])
        if cls == "odd" and pts:
            t = max(pts)
            d = self.engine.read(pts[t])
            if d.state == FAILED:
                self.state["status"] = NEEDS_ATTENTION
                self.state["message"] = "iteration %d direction %d: point %s failed" % (k, j, d.key)
                self.say(self.state["message"]); return None
            st = self._stat(pts[t])
            extra["_running"] = d.state in (RUNNING, QUEUED)
            if st is None or centre is None:
                if d.state in (DONE, STOPPED) and st is None:
                    self.state["status"] = NEEDS_ATTENTION
                    self.state["message"] = "iteration %d direction %d: point ended with too few blocks" % (k, j)
                    self.say(self.state["message"])
                else:
                    self.say("iteration %d: waiting for blocks (direction %d, one-sided)" % (k, j))
                return None
            extra["_blocks"] = st[2]
            ev = 2.0 * (st[0] - centre[0]) * HARTREE_TO_KCAL                   # E(+t) + E(-t) - 2 E(0) with E(-t) = E(+t)
            se = 2.0 * float(np.hypot(st[1], centre[1])) * HARTREE_TO_KCAL
            if p.control_variate and cv_ok and d.e_surrogate is not None and e0_sur is not None and k_sur is not None:
                ev = ev - 2.0 * (d.e_surrogate - e0_sur) * HARTREE_TO_KCAL + k_sur * t * t
            k_meas, sk_meas = ev / (t * t), se / (t * t)
        memory = self._k_prior(it, j, cv_ok)
        if memory is not None:
            kk, sk, source = P2.combine_curvature(k_meas, sk_meas, memory[0], p, s_prior=P2.memory_error(memory, p))
        else:
            k_prior = k_sur if (cv_ok or not (np.isfinite(k_meas) and np.isfinite(sk_meas))) else None
            kk, sk, source = P2.combine_curvature(k_meas, sk_meas, k_prior, p)
        curv_ok = bool(np.isfinite(kk) and kk != 0.0 and np.sign(kk) == np.sign(role))
        est = dict(g=0.0, sg=0.0, g_meas=0.0, sg_meas=0.0, pooled=False, k=float(kk), sk=float(sk),
                   k_meas=float(k_meas), sk_meas=float(sk_meas), k_source=source,
                   k_sur=float(k_sur) if k_sur is not None else float("nan"), g_sur=0.0, chi2=0.0, R=R,
                   npairs=0, control_variate=bool(p.control_variate), role=int(role), curv_ok=curv_ok, z=0.0,
                   m=0.0 if curv_ok else float("nan"), sm=0.0 if curv_ok else float("nan"), by_symmetry=cls)
        est.update(extra)
        return est

    # ------------------------------------------------------------------ model Hessian
    def _secant_update(self, k: int, it: dict, Kc: np.ndarray, g_meas: np.ndarray, sg_meas: np.ndarray,
                       roles: np.ndarray, quiet: bool = False) -> np.ndarray:
        """Correct the model Hessian with the slope changes that DMC measured along the last steps.

        For a past step s from centre a to centre b the measured slope change is y = g(b) - g(a). The
        model predicts Kc @ s. When the difference r = y - Kc @ s is larger than its statistical error
        allows, the symmetric rank-one update Kc += r r^T / (r^T s) makes the model reproduce y. An
        update that changes the number of negative eigenvalues is discarded."""
        p = self.protocol
        if not p.secant:
            return Kc
        n = self.dim
        pairs = []
        kk, cur_g, cur_sg, cur = k, g_meas, sg_meas, it
        while cur.get("base") is not None and len(pairs) < p.secant_memory:
            base = self._it(cur["base"])
            comp = base.get("composed")
            if not comp or comp.get("g_meas") is None:
                break
            step = np.array(comp["components"], dtype=float) * float(cur.get("step_scale", 1.0))
            y = cur_g - np.array(comp["g_meas"], dtype=float)
            sy = np.sqrt(cur_sg ** 2 + np.array(comp["sg_meas"], dtype=float) ** 2)
            pairs.append((step, y, sy, kk))
            kk, cur_g, cur_sg, cur = cur["base"], np.array(comp["g_meas"], dtype=float), np.array(comp["sg_meas"], dtype=float), base
        n_neg = int(np.sum(roles < 0))
        for step, y, sy, kk in reversed(pairs):              # oldest first
            if not (np.all(np.isfinite(y)) and np.all(np.isfinite(sy)) and np.linalg.norm(step) > 1e-6):
                continue
            r = y - Kc @ step
            free = sy > 0                                    # slopes fixed by symmetry carry no information here
            if not np.any(free):
                continue
            r = np.where(free, r, 0.0)
            chi2 = float(np.sum((r[free] / sy[free]) ** 2))
            nf = int(np.sum(free))
            if chi2 <= nf + 3.0 * np.sqrt(2.0 * nf):
                continue                                     # the model already explains the measured change
            denom = float(r @ step)
            if abs(denom) < 1e-8 * np.linalg.norm(r) * np.linalg.norm(step):
                continue
            trial = Kc + np.outer(r, r) / denom
            w = np.linalg.eigvalsh(0.5 * (trial + trial.T))
            if int(np.sum(w < 0)) != n_neg:
                continue
            Kc = trial
            if not quiet:
                self.say("  model Hessian corrected with the slope change of the step into iteration %d "
                         "(mismatch chi2 %.1f for %d directions)" % (kk, chi2, n))
        return Kc

    def _coupling_test(self, it: dict, coupled: np.ndarray, diagonal: np.ndarray, g_meas: np.ndarray,
                       sg_meas: np.ndarray) -> Optional[Tuple[float, float]]:
        """How well two step models predict the slope change that DMC measured along the last step.

        For the step s from the base centre to this centre the measured slope change is y = g(here) - g(base).
        A model Hessian K predicts K @ s. Returns (chi2 of the model with surrogate couplings, chi2 of the
        diagonal model), or None if this centre has no base, the step is negligible or no slope is free."""
        if it.get("base") is None:
            return None
        base = self._it(it["base"])
        comp = base.get("composed")
        if not comp or comp.get("g_meas") is None:
            return None
        step = np.array(comp["components"], dtype=float) * float(it.get("step_scale", 1.0))
        if not np.linalg.norm(step) > 1e-6:
            return None
        y = g_meas - np.array(comp["g_meas"], dtype=float)
        sy = np.sqrt(sg_meas ** 2 + np.array(comp["sg_meas"], dtype=float) ** 2)
        free = sy > 0                                       # slopes fixed by symmetry carry no information here
        if not np.any(free) or not (np.all(np.isfinite(y)) and np.all(np.isfinite(sy))):
            return None
        chi2 = [float(np.sum((((y - K @ step)[free]) / sy[free]) ** 2)) for K in (coupled, diagonal)]
        return chi2[0], chi2[1]

    def _slope_pairs(self, it: dict, ests: List[dict]) -> list:
        """The steps between centres that lead to this centre, newest first, as (step, measured slope change, its
        error, mean diagonal force constants of the two centres). Only steps that started near the stationary
        point are used (predicted gain at most ``cv_window``): farther away the slope changes also hold the
        anharmonic part of the surface."""
        p = self.protocol
        window = p.e_cap if p.cv_window is None else float(p.cv_window)
        pairs = []
        cur = it
        g_cur = np.array([e["g_meas"] for e in ests], dtype=float)
        s_cur = np.array([e.get("sg_meas_total", e["sg_meas"]) for e in ests], dtype=float)
        k_cur = np.array([e["k"] for e in ests], dtype=float)
        while cur.get("base") is not None and len(pairs) < int(p.coupling_fit_memory):
            base = self._it(cur["base"])
            comp = base.get("composed")
            if not comp or comp.get("g_meas") is None or comp.get("kdiag") is None:
                break
            if not np.isfinite(float(comp.get("G", float("inf")))) or float(comp["G"]) > window:
                break
            step = np.array(comp["components"], dtype=float) * float(cur.get("step_scale", 1.0))
            g_base = np.array(comp["g_meas"], dtype=float)
            s_base = np.hypot(np.array(comp["sg_meas"], dtype=float), np.array(comp.get("sg_shape") or 0.0, dtype=float))
            k_base = np.array(comp["kdiag"], dtype=float)
            if (np.linalg.norm(step) > 1e-6 and np.all(np.isfinite(g_cur)) and np.all(np.isfinite(g_base))
                    and np.all(np.isfinite(k_cur)) and np.all(np.isfinite(k_base))):
                pairs.append((step, g_cur - g_base, np.hypot(s_cur, s_base), 0.5 * (k_cur + k_base)))
            cur, g_cur, s_cur, k_cur = base, g_base, s_base, k_base
        return pairs

    def _model_hessian(self, k: int, it: dict, ests: List[dict], quiet: bool = False) -> Tuple[np.ndarray, str]:
        """Hessian of the step model in the direction basis, and a label of how it was built: the model from the
        scans and the surrogate (:meth:`_scan_model_hessian`), with couplings fitted to the measured slope
        changes where that model does not predict them (``protocol_v2.fit_couplings``). Couplings that were
        fitted at an earlier centre are kept while the search stays near the stationary point."""
        p = self.protocol
        Kh, rule = self._scan_model_hessian(k, it, ests, quiet=quiet)
        if not p.coupling_fit or self.dim < 2 or any(not e["curv_ok"] for e in ests):
            return Kh, rule
        kdiag = np.array([e["k"] for e in ests], dtype=float)
        if not np.all(np.isfinite(kdiag)) or np.any(kdiag == 0):
            return Kh, rule
        roles = np.array(self.state["roles"])
        fit = P2.fit_couplings(kdiag, Kh, self._slope_pairs(it, ests), roles, p)
        if fit is not None:
            if not quiet:
                self.state["couplings"] = {"C": np.asarray(fit["C"]).tolist(), "iteration": k}
                self.say("  couplings fitted to the slope changes of %d step(s): chi2 %.1f -> %.1f for %d values; largest "
                         "coupling %.2f of the diagonal" % (fit["steps"], fit["chi2_base"], fit["chi2_fit"], fit["dof"],
                                                           max_coupling(np.diag(kdiag) + fit["C"])))
            return np.diag(kdiag) + np.asarray(fit["C"]), "newton, fitted couplings"
        kept = self.state.get("couplings")
        if kept is not None:
            # no new evidence at this centre (the steps are short now): the couplings that were fitted before
            Kf = np.diag(kdiag) + np.array(kept["C"], dtype=float)
            scale = np.sqrt(np.abs(kdiag))
            mu = np.linalg.eigvalsh(Kf / np.outer(scale, scale))
            if int(np.sum(mu < 0)) == int(np.sum(roles < 0)) and float(np.abs(mu).min()) >= p.coupling_fit_floor:
                return Kf, "newton, fitted couplings"
        return Kh, rule

    def _scan_model_hessian(self, k: int, it: dict, ests: List[dict], quiet: bool = False) -> Tuple[np.ndarray, str]:
        """Hessian of the step model in the direction basis, and a label of how it was built.

        The diagonal holds the force constants of the estimator. The off-diagonal elements come from the
        surrogate Hessian at the centre, if (1) the surrogate reproduces the curvatures that DMC measured
        clearly, (2) the model with these couplings predicts the slope change measured along the last step at
        least as well as the model without them, and (3) the result has the expected number of negative
        eigenvalues. Otherwise the model is diagonal."""
        p = self.protocol
        n = self.dim
        roles = np.array(self.state["roles"])
        kdiag = np.array([e["k"] for e in ests], dtype=float)
        if not np.all(np.isfinite(kdiag)) or np.any(kdiag == 0):
            return np.diag(np.where(np.isfinite(kdiag) & (kdiag != 0), kdiag, roles * 1.0)), "diagonal"
        Kd = self._hessian_in_directions(it)
        if Kd is None or any(not e["curv_ok"] for e in ests) or it.get("surrogate_ok") is False:
            return np.diag(kdiag), "diagonal"
        ratios = []
        for e in ests:
            km, skm, ks = e["k_meas"], e["sk_meas"], e["k_sur"]
            if np.isfinite(km) and np.isfinite(skm) and np.isfinite(ks) and ks != 0 and abs(km) >= 3.0 * skm:
                ratios.append(abs(km / ks - 1.0))
        if ratios and max(ratios) > p.coupling_tolerance:
            if not quiet:
                self.say("  surrogate couplings not used at this centre: measured and surrogate curvature differ by a "
                         "factor of up to %.2f" % (1.0 + max(ratios)))
            return np.diag(kdiag), "diagonal"
        Kc = Kd.copy(); Kc[np.diag_indices(n)] = kdiag
        g_meas = np.array([e["g_meas"] for e in ests])
        sg_meas = np.array([e.get("sg_meas_total", e["sg_meas"]) for e in ests])        # statistical error and shape term
        if p.coupling_test:
            test = self._coupling_test(it, Kc, np.diag(kdiag), g_meas, sg_meas)
            if test is not None and test[0] > test[1]:
                if not quiet:
                    self.say("  surrogate couplings not used at this centre: the model without couplings predicts the slope "
                             "change of the last step better (chi2 %.1f against %.1f)" % (test[1], test[0]))
                return np.diag(kdiag), "diagonal"
        Kc = self._secant_update(k, it, Kc, g_meas, sg_meas, roles, quiet=quiet)
        w = np.linalg.eigvalsh(0.5 * (Kc + Kc.T))
        if int(np.sum(w < 0)) != int(np.sum(roles < 0)) or np.any(np.abs(w) < 1e-8):
            if not quiet:
                self.say("  surrogate couplings not used at this centre: the combined Hessian has the wrong number "
                         "of negative eigenvalues")
            return np.diag(kdiag), "diagonal"
        return Kc, "newton"

    # ------------------------------------------------------------------ decision and step
    def _step(self, it: dict, ests: List[dict], Kh: np.ndarray, rule: str, spans: np.ndarray) -> Tuple[np.ndarray, str, bool]:
        """Step in the direction basis, its label, and whether it was limited by the trusted span."""
        p = self.protocol
        n = self.dim
        trust = p.trust * self.state["trust_scale"]
        roles = np.array(self.state["roles"])
        g = np.array([e["g"] for e in ests]); kdiag = np.array([e["k"] for e in ests])
        wrong = [j for j in range(n) if not ests[j]["curv_ok"]]
        limited = False
        if wrong:
            # diagonal step; a direction whose curvature has the wrong sign moves downhill (uphill for a reaction
            # coordinate) by the trusted span
            s = np.zeros(n)
            for j in range(n):
                if j in wrong:
                    sign = -roles[j] * np.sign(g[j]) if g[j] != 0 else 1.0
                    s[j] = sign * trust * spans[j]
                else:
                    s[j] = float(np.clip(-g[j] / kdiag[j], -trust * spans[j], trust * spans[j]))
                    limited = limited or abs(g[j] / kdiag[j]) > trust * spans[j]
            return s, "diagonal+downhill", True
        s = -np.linalg.solve(Kh, g)
        over = np.abs(s) / (trust * spans)
        if over.max() > 1.0 + 1e-9:
            s = s / over.max(); limited = True; rule += "+trust"
        if rule.startswith("newton") and np.all(roles > 0) and float(g @ s + 0.5 * s @ Kh @ s) >= 0.0:
            # a minimum search must not take a step that the model itself predicts to raise the energy
            sd = -g / kdiag
            over = np.abs(sd) / (trust * spans)
            limited = over.max() > 1.0 + 1e-9
            s = sd / max(1.0, over.max()); rule = "diagonal" + ("+trust" if limited else "")
        return s, rule, limited

    def _decide(self, k: int, it: dict, data: dict) -> None:
        p = self.protocol
        n = self.dim
        ests, blocks, running = data["ests"], data["blocks"], data["running"]
        final = (not running) or blocks >= self._point_blocks()
        g = np.array([e["g"] for e in ests]); sg = np.array([e["sg"] for e in ests])
        Kh, rule = self._model_hessian(k, it, ests, quiet=True)
        stats = P2.joint_statistics(g, sg, Kh)
        cn = P2.bound_factor(stats["nu"])
        # the part of the noise energy N that comes from the shape term of the slopes: blocks do not reduce it
        sg_shape = np.array([float(e.get("sg_shape", 0.0)) for e in ests])
        n_shape = float(P2.joint_statistics(np.zeros(n), sg_shape, Kh)["N"]) if np.any(sg_shape > 0.0) else 0.0
        shape_limited = bool(cn * n_shape >= p.eps_total)
        if shape_limited and not final and stats["N"] <= 2.0 * n_shape:
            final = True                        # the statistical part is already below the shape part
        wrong = [j for j in range(n) if not ests[j]["curv_ok"]]
        if wrong:
            resolved = [j for j in wrong if ests[j]["z"] >= p.z_resolved]
            clear = [j for j in wrong if np.isfinite(ests[j]["sk_meas"]) and abs(ests[j]["k_meas"]) >= 3.0 * ests[j]["sk_meas"]]
            if clear and not resolved and all(ests[j].get("by_symmetry") for j in clear):
                resolved = clear                # zero slope by symmetry, clearly wrong curvature: the symmetric geometry is unstable
            if resolved or (final and clear):
                verdict = "MOVE"
                note = "the curvature has the wrong sign along direction(s) %s: step to the edge of the trusted span" % wrong
            elif final:
                verdict = "FINAL"; note = "no more blocks; the curvature along direction(s) %s is not determined" % wrong
            else:
                verdict, note = "CONTINUE", ""
        else:
            verdict, note = P2.joint_decision(stats, n, final, p)

            def shorter_spans() -> Tuple[str, str]:
                energies = self._shape_energies(ests)
                target = self._shape_target()
                factors = self.state.get("span_factor") or [1.0] * n
                can = [j for j in range(n) if energies[j] > target and factors[j] > p.span_factor_min + 1e-9
                       and float(it["half_span"][j]) > p.r_min + 1e-9]
                if not can:
                    return verdict, note
                return "MOVE", ("along direction(s) %s the surrogate can differ in shape from the DMC surface by more "
                                "than the tolerance allows at these spans (shape part of the joint bound %.3f kcal/mol): "
                                "the scan is repeated with shorter spans" % (can, cn * n_shape))

            if shape_limited and verdict == "FINAL" and k < p.max_iter:
                verdict, note = shorter_spans()
            if verdict == "FINAL" and cn * stats["N"] >= p.eps_total and not shape_limited and k < p.max_iter:
                # the blocks of this scan were too few for the precision target (the block limit was too low, or
                # points were stopped early). The statistical part of the noise energy falls like 1 / blocks:
                # ask for the number that brings the bound below the target
                usable_now = max(blocks - int(p.drop_min), 1)
                stat = cn * max(stats["N"] - n_shape, 0.0)
                room = max(p.eps_total - cn * n_shape, 0.25 * p.eps_total)
                usable_wanted = int(np.ceil(p.rescan_margin * usable_now * stat / room))
                allowed = int(np.floor(max(p.cap_scan_factor, 1.0) * p.cap))
                want = int(min(usable_wanted + int(p.drop_max), allowed))
                if want >= 1.2 * blocks:
                    verdict = "MOVE"
                    note = ("the %d blocks of this scan do not reach the precision target (joint bound %.3f, target %.2f "
                            "kcal/mol): the step is taken and the scan is repeated with up to %d blocks per point"
                            % (blocks, cn * stats["N"], p.eps_total, max(want, int(p.cap))))
                    self.state["scan_blocks"] = max(want, int(p.cap))
                    self.state["usable_wanted"] = int(min(usable_wanted, want - int(p.drop_min)))
                elif cn * n_shape >= 0.5 * p.eps_total:
                    # More blocks cannot bring the bound below the target (the statistical part is small already),
                    # and the shape part is most of the bound: shorter spans reduce it.
                    verdict, note = shorter_spans()
        self.say("iteration %d at %d blocks: predicted gain G %.3f, noise N %.3f (joint bound %.3f%s), sqrt(V) %.3f kcal/mol -> %s"
                 % (k, blocks, stats["G"], stats["N"], cn * stats["N"],
                    "" if cn * n_shape < 0.25 * p.eps_total else ", shape part %.3f" % (cn * n_shape),
                    np.sqrt(stats["V"]), verdict))
        if verdict == "CONTINUE":
            return
        # ---- the centre is decided: stop its points and compose the step
        stopped = self._stop_iteration(k, it)
        if it.get("accepted") is None and it.get("base") is not None:
            if self._reject_step(k, it):
                return
            it["accepted"] = True
        self._adapt_trust(k, it)
        Kh, rule = self._model_hessian(k, it, ests, quiet=False)
        spans = np.array(it["half_span"], dtype=float)
        s, rule, limited = self._step(it, ests, Kh, rule, spans)
        if verdict == "FINAL" and (limited or wrong):
            verdict = "MOVE"; note += "; the step is limited by the trusted span, so the centre moves"
        shrink = 1.0
        if verdict == "FINAL" and p.shrink_final:
            shrink = max(0.0, 1.0 - stats["N"] / stats["G"]) if stats["G"] > 0 else 0.0
        change = float(g @ (shrink * s) + 0.5 * (shrink * s) @ Kh @ (shrink * s))
        it["decided"] = {str(j): {"status": verdict, "m": float(shrink * s[j]), "blocks": blocks, "note": note,
                                  "res": _clean(ests[j])} for j in range(n)}
        it["composed"] = {
            "rule": rule, "verdict": verdict, "components": [float(v) for v in shrink * s], "shrink": float(shrink),
            "step_norm": float(np.linalg.norm(shrink * s)), "predicted_change": change, "limited": bool(limited),
            "G": stats["G"], "N": stats["N"], "V": stats["V"], "joint_bound": cn * stats["N"], "blocks": blocks,
            "Kh": Kh.tolist(), "g_all": g.tolist(), "sg": sg.tolist(),
            "g_meas": [e["g_meas"] for e in ests], "sg_meas": [e["sg_meas"] for e in ests],
            "g_sur": [e["g_sur"] if np.isfinite(e["g_sur"]) else None for e in ests],
            "kdiag": [e["k"] for e in ests], "spans": spans.tolist(),
            "max_coupling": max_coupling(Kh) if rule.startswith("newton") else 0.0,
            "N_shape": n_shape, "shape": [e.get("shape") if np.isfinite(e.get("shape", float("nan"))) else None for e in ests],
            "sg_shape": [float(v) for v in sg_shape],
        }
        self._remember_curvatures(ests, float(stats["G"]))
        self._remember_shape(it, ests)
        window = p.e_cap if p.cv_window is None else float(p.cv_window)
        if stats["G"] > window:
            # Far from the stationary point the task is to get there, not to be precise: full spans. Shorter
            # spans (for a smaller shape term of the slopes) would only limit the steps.
            self.state["span_factor"] = [1.0] * n
            self.state["span_wanted"] = {}
            shortened = []
        else:
            shortened = self._update_span_factors(it, ests)
        it["composed"]["span_factor_next"] = list(self.state["span_factor"])
        if shortened and verdict == "MOVE":
            self.say("  the spans of direction(s) %s are shortened at the next centre (to %s A)" % (
                shortened, [round(float(self.state["span_wanted"][str(j)]), 4) for j in shortened]))
        it["decision"] = {"verdict": verdict, "note": note}
        self.say("iteration %d decided: %s [%s]; stopped %d point(s)" % (k, verdict, note, stopped))
        self.say("  step (%s%s): %s, |s| = %.5f A, predicted change %+.3f kcal/mol"
                 % (rule, "" if shrink == 1.0 else ", multiplied by %.2f" % shrink, np.round(shrink * s, 4).tolist(),
                    float(np.linalg.norm(shrink * s)), change))
        if verdict == "FINAL":
            self._start_final(k, it, cn * stats["N"] < p.eps_total and not wrong)
            return
        if k >= p.max_iter:
            # the iteration limit is reached: no further step. The energy of this last centre is still measured
            # with full-length runs, so that the report can compare it with the starting geometry.
            self.say("iteration limit %d reached (joint bound %.3f, predicted gain %.3f kcal/mol): the search ends at "
                     "this centre without the last step" % (k, cn * stats["N"], stats["G"]))
            it["composed"]["components"] = [0.0] * n
            it["composed"]["step_norm"] = 0.0
            self._start_final(k, it, False, reason="the iteration limit of %d was reached" % p.max_iter)
            return
        self._build_next(k + 1, k, 1.0)

    # ------------------------------------------------------------------ safeguards
    def _reject_step(self, k: int, it: dict) -> bool:
        """Step acceptance test (minimum searches). The centre energy of this iteration is compared with the
        centre energy of the iteration it was built from. A rise of more than ``accept_sigma`` standard errors
        (and more than half the tolerance) rejects the step: the points of this iteration are stopped and a new
        iteration is built with half the step from the same base. Returns True on a rejection."""
        p = self.protocol
        if not p.accept_test or it.get("accepted") is not None or it.get("base") is None:
            return False
        if any(r < 0 for r in self.state["roles"]):
            it["accepted"] = True             # a saddle is not an energy minimum: no monotonic test
            return False
        base = self._it(it["base"])
        e_new, e_old = self._centre_stat(it), self._centre_stat(base)
        if e_new is None or e_old is None:
            return False
        de = (e_new[0] - e_old[0]) * HARTREE_TO_KCAL
        sd = float(np.hypot(e_new[1], e_old[1]) * HARTREE_TO_KCAL)
        it["energy_change"] = [float(de), sd]
        if not (de > p.accept_sigma * sd and de > 0.5 * p.eps_total):
            return False
        scale = float(it.get("step_scale", 1.0)) * 0.5
        if np.linalg.norm(np.array(base["composed"]["components"])) * scale < 2e-4:
            it["accepted"] = True             # the step is already negligible
            return False
        self._stop_iteration(k, it)
        it["accepted"] = False
        it["composed"] = None
        self.state["rejected"] += 1
        self.state["trust_scale"] = max(p.trust_min, self.state["trust_scale"] * 0.5)
        self.say("iteration %d REJECTED: the centre energy rose by %+.3f +/- %.3f kcal/mol. New iteration with %.3g of "
                 "the step from iteration %d; trust factor now %.2f" % (k, de, sd, scale, it["base"], self.state["trust_scale"]))
        if k >= p.max_iter:
            # No iteration is left. The final geometry is the centre that the rejected step started from: it gets
            # the full-length runs, so that the report has an energy of the same quality as in every other case.
            self.say("not converged after %d iterations (the last step was rejected)" % k)
            self._start_final(int(it["base"]), base, certified=False, at_centre=True,
                              reason="the iteration limit was reached and the last step was rejected; the final "
                                     "geometry is the centre of iteration %d" % int(it["base"]))
            return True
        self._build_next(k + 1, it["base"], scale)
        return True

    def _adapt_trust(self, k: int, it: dict) -> None:
        """Widen or narrow the trust factor from the ratio of measured to predicted energy change of the last step."""
        p = self.protocol
        if it.get("base") is None or it.get("energy_change") is None or it.get("predicted_change") is None:
            return
        de, sd = it["energy_change"]; pred = it["predicted_change"]
        if pred >= 0 or abs(pred) < 3.0 * sd:
            return                              # the change is too small to judge the model
        rho = de / pred
        old = self.state["trust_scale"]
        if rho > 0.75 and self._it(it["base"])["composed"].get("limited"):
            self.state["trust_scale"] = min(p.trust_max, old * 2.0)
        elif rho < 0.25:
            self.state["trust_scale"] = max(p.trust_min, old * 0.5)
        if self.state["trust_scale"] != old:
            self.say("  measured / predicted energy change of the last step = %.2f: trust factor %.2f -> %.2f"
                     % (rho, old, self.state["trust_scale"]))

    def _build_next(self, k_new: int, k_base: int, scale: float) -> None:
        """Build iteration ``k_new`` at (centre of iteration ``k_base``) + scale * (its step)."""
        p = self.protocol
        base = self._it(k_base); comp = base["composed"]
        s = np.array(comp["components"], dtype=float) * scale
        new_centre = np.array(base["centre"]) + (s @ self.vectors).reshape(-1, 3)
        self._new_iteration(k_new, new_centre, lam=[float(v) / 2.0 for v in comp["kdiag"]], half_span_prev=list(comp["spans"]))
        nxt = self._it(k_new)
        Kh = np.array(comp["Kh"]); g_all = np.array(comp["g_all"]); sg = np.array(comp["sg"])
        nxt["base"] = k_base; nxt["step_scale"] = float(scale)
        nxt["predicted_change"] = float(g_all @ s + 0.5 * s @ Kh @ s)
        if p.pool:
            # model error of a carried-over slope: relative curvature error times the step
            kd = np.abs(np.diag(Kh))
            model = np.sqrt((p.pool_model_error * kd * np.abs(s)) ** 2
                            + (p.pool_model_error * (np.abs(Kh - np.diag(np.diag(Kh))) @ np.abs(s))) ** 2)
            err = np.sqrt(sg ** 2 + model ** 2)
            g_sur = comp.get("g_sur")
            if p.control_variate and g_sur is not None and all(v is not None for v in g_sur):
                # carry the difference "DMC slope minus surrogate slope"; the surrogate slope at the new
                # centre is computed there exactly
                nxt["prior"] = {"kind": "d", "value": (g_all - np.array(g_sur, dtype=float)).tolist(), "error": err.tolist()}
            else:
                nxt["prior"] = {"kind": "g", "value": (g_all + Kh @ s).tolist(), "error": err.tolist()}
        # The decision and the new centre go to disk BEFORE any job of the new iteration is submitted. If this
        # process is stopped during the submissions, the next process finds the same centre and submits what is
        # missing. Without this, it would decide again with other data, and the jobs that exist already (they
        # keep their keys) would belong to another geometry.
        self._save()
        self._ensure_points(k_new, nxt)
        self.say("iteration %d built: %d scan points about the new centre" % (
            k_new, sum(len(v) for v in nxt["points"].values())))

    # ------------------------------------------------------------------ the end of the search
    def _start_final(self, k: int, it: dict, certified: bool, reason: str = "", at_centre: bool = False) -> None:
        """Start the full-length runs at the final geometry: the centre of iteration ``k`` plus its composed step,
        or the centre itself with ``at_centre`` (used when no step from this centre can be trusted)."""
        p = self.protocol
        if at_centre:
            final_centre = np.array(it["centre"], dtype=float)
            bound = None
        else:
            s = np.array(it["composed"]["components"], dtype=float)
            final_centre = np.array(it["centre"]) + (s @ self.vectors).reshape(-1, 3)
            bound = it["composed"]["joint_bound"]
        start = np.array(self._it(1)["centre"])
        same_as_start = float(np.abs(final_centre - start).max()) < 1e-6
        if same_as_start:
            keys = list(self._it(1)["centre_keys"])        # the reference runs of the start are the final runs
        else:
            keys = [self._key(k, "final" if i == 1 else "final_s%d" % i) for i in range(1, int(p.final_series) + 1)]
        self.state["final"] = {"iteration": k, "centre": final_centre.tolist(), "keys": keys, "certified": bool(certified),
                               "joint_bound": bound, "same_as_start": same_as_start, "reason": reason,
                               "at_centre": bool(at_centre)}
        self.state["status"] = FINALIZING
        self._save()                                # as in _build_next: on disk before the job is submitted
        if not same_as_start:
            self.engine.submit(PointSpec(key=keys[0], symbols=self.symbols, coords=final_centre, blocks=int(p.cap),
                                         role="final", series=1, settings=self.settings))
        self.say("search ended at iteration %d (%s). %s" % (
            k, "precision target reached" if certified else (reason or "precision target NOT reached"),
            "The final geometry is the starting geometry." if same_as_start
            else "Full-length runs requested at the final geometry."))

    def _tick_final(self) -> None:
        p = self.protocol
        fin = self.state["final"]
        if not fin.get("same_as_start"):
            # asked for on every pass: the process may have ended between the saved decision and the submission
            self.engine.submit(PointSpec(key=fin["keys"][0], symbols=self.symbols, coords=np.array(fin["centre"]),
                                         blocks=int(p.cap), role="final", series=1, settings=self.settings))
        first = self.engine.read(fin["keys"][0])
        if first.nblocks > 0 and not fin.get("same_as_start"):
            for i, key in enumerate(fin["keys"][1:], start=2):
                self.engine.submit(PointSpec(key=key, symbols=self.symbols, coords=np.array(fin["centre"]),
                                             blocks=int(p.cap), role="final", series=i, base_key=fin["keys"][0],
                                             settings=self.settings))
        it1 = self._it(1)
        self._centre_extension(it1)
        states = [self.engine.read(key).state for key in dict.fromkeys(list(fin["keys"]) + list(it1["centre_keys"]))]
        if any(s == FAILED for s in states):
            self.state["status"] = NEEDS_ATTENTION
            self.state["message"] = "a full-length run at the start or final geometry failed"
            self.say(self.state["message"]); return
        if not all(s in (DONE, STOPPED) for s in states):
            return
        ef, e0 = self.final_energy(), self.start_energy()
        if ef is not None and e0 is not None:
            de = (ef[0] - e0[0]) * HARTREE_TO_KCAL; sd = float(np.hypot(ef[1], e0[1]) * HARTREE_TO_KCAL)
            fin["energy_change"] = [float(de), sd]
            self.say("energy of the final geometry minus energy of the starting geometry: %+.3f +/- %.3f kcal/mol" % (de, sd))
        if fin["certified"]:
            self.state["status"] = CONVERGED
            self.state["message"] = "converged: joint 95 %% bound %.3f < %.2f kcal/mol" % (fin["joint_bound"], p.eps_total)
        else:
            self.state["status"] = NOT_CONVERGED
            bound = "no joint bound" if fin["joint_bound"] is None else "joint bound %.3f kcal/mol" % fin["joint_bound"]
            self.state["message"] = "ended without reaching the precision target (%s, target %.2f)%s" % (
                bound, p.eps_total, ": " + fin["reason"] if fin.get("reason") else "")
        self.say(self.state["message"])

    # ------------------------------------------------------------------ results
    def _combined(self, keys) -> Optional[Tuple[float, float]]:
        vals = []
        for key in keys:
            st = self._stat(key)
            if st is not None:
                vals.append(st)
        if not vals:
            return None
        return combine_series(vals)

    def final_geometry(self) -> np.ndarray:
        if self.state.get("final"):
            return np.array(self.state["final"]["centre"], dtype=float)
        return self.centre(self.final_iteration())

    def final_energy(self) -> Optional[Tuple[float, float]]:
        """Combined DMC energy (hartree) and error of the full-length runs at the final geometry."""
        if not self.state.get("final"):
            return None
        return self._combined(self.state["final"]["keys"])

    def start_energy(self) -> Optional[Tuple[float, float]]:
        """Combined DMC energy (hartree) and error of the full-length runs at the starting geometry."""
        return self._combined(self._it(1)["centre_keys"])

    def total_blocks(self) -> int:
        n = super().total_blocks()
        if self.state.get("final") and not self.state["final"].get("same_as_start"):
            n += sum(self.engine.read(key).nblocks for key in self.state["final"]["keys"])
        return n

    def search_blocks(self) -> int:
        """Blocks spent on the search alone: scan points and centre runs, without the full-length runs at the
        starting geometry and at the final geometry."""
        n = 0
        for ks, it in self.state["iters"].items():
            if int(ks) != 1:
                n += sum(self.engine.read(key).nblocks for key in it["centre_keys"])
            for pts in it["points"].values():
                n += sum(self.engine.read(key).nblocks for key in pts.values())
        return n

    def run(self, max_ticks: int = 10000) -> str:
        for _ in range(max_ticks):
            self.engine.advance()
            if self.tick() not in (RUNNING_STATUS, FINALIZING):
                break
        return self.state["status"]
