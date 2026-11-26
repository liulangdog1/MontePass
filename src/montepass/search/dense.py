"""The dense-scan driver: optimizes one geometry on the DMC surface by repeated symmetric scans.

The driver is independent of the QMC program and of the batch system. It talks
to an :class:`~montepass.engine.base.Engine` (block energies of geometries) and
to a :class:`~montepass.surrogate.base.Surrogate` (DFT gradient and Hessian at
a scan centre).

One call of :meth:`DenseScanSearch.tick` is one pass:

1. make sure every point of the current iteration is submitted;
2. read the blocks written so far, estimate slope and curvature of every open direction,
   and apply the stopping rules of the protocol;
3. stop the jobs of a direction as soon as it is decided;
4. when every direction is decided: test convergence, or compose the step, move the
   centre and build the next iteration.

The state is a JSON file, written after every pass. A driver that is started again
continues where the previous one stopped. No decision depends on anything but the
surrogate and the block energies that the driver itself requested.

Two protocols exist (see ``protocol_v1`` and ``protocol_v2``). The rules of protocol v1
are those of an earlier application; ``protocol_v1`` is a line-by-line port of them. The test
that replays the block traces of that application needs data that are not part of this
repository (docs/protocols.md, section 2).
"""
from __future__ import annotations

import json
import os
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..engine.base import ABSENT, DONE, FAILED, QUEUED, RUNNING, STOPPED, Engine, PointSpec
from ..stats import reblock
from ..subspace import max_coupling
from ..surrogate.base import Surrogate
from ..units import HARTREE_TO_KCAL
from . import protocol_v1 as P1
from . import protocol_v2 as P2

RUNNING_STATUS = "RUNNING"
CONVERGED = "CONVERGED"
NOT_CONVERGED = "NOT_CONVERGED"          # the iteration limit was reached
NEEDS_ATTENTION = "NEEDS_ATTENTION"      # a calculation failed; a person must look


def _tkey(t: float) -> str:
    return "%+.3f" % t


def _point_name(j: int, t: float) -> str:
    return "d%d_%s%03d" % (j, "p" if t > 0 else "m", int(round(abs(t) * 1000)))


def combine_series(vals: Sequence[Tuple[float, float, int]]) -> Tuple[float, float]:
    """Mean and error of independent runs of one geometry, weighted by their inverse variances. A run with an
    error of exactly zero (a model without noise) is exact: its value is returned with the error zero."""
    exact = [v for v in vals if not v[1] > 0.0]
    if exact:
        return float(exact[0][0]), 0.0
    w = np.array([1.0 / v[1] ** 2 for v in vals])
    return float(np.sum(w * np.array([v[0] for v in vals])) / w.sum()), float(1.0 / np.sqrt(w.sum()))


class DenseScanSearch:
    """Dense-scan optimization of one geometry (a minimum or a saddle).

    Parameters
    ----------
    name        short label of the target, used in keys and file names (for example ``min``).
    engine      source of DMC block energies.
    surrogate   source of the DFT Hessian and gradient at scan centres. May be None: then the
                step is composed from the one-dimensional results alone.
    symbols, start
                element symbols and the starting geometry (n_atoms, 3), angstrom.
    vectors     search directions, shape (d, 3 n), unit norm, Cartesian. They stay fixed.
    curvature   signed surrogate force constants k_j along the directions at the start
                (kcal/mol/A^2, E = 0.5 k t^2). Used for the grid of the first iteration.
    protocol    a :class:`ProtocolV1` or :class:`ProtocolV2` object.
    state_dir   folder for the state file ``<name>_state.json`` and the log ``<name>.log``.
    """

    def __new__(cls, *args, **kwargs):
        protocol = kwargs.get("protocol", args[7] if len(args) > 7 else P1.V1)
        if cls is DenseScanSearch and isinstance(protocol, P2.ProtocolV2):
            from .dense_v2 import DenseScanSearchV2
            return super().__new__(DenseScanSearchV2)
        return super().__new__(cls)

    def __init__(self, name: str, engine: Engine, surrogate: Optional[Surrogate], symbols: Sequence[str],
                 start: np.ndarray, vectors: np.ndarray, curvature: Sequence[float], protocol=P1.V1,
                 state_dir: str = ".", settings: Optional[Dict[str, object]] = None,
                 centre_series: int = 2, log: Optional[Callable[[str], None]] = None):
        self.name = name
        self.engine = engine
        self.surrogate = surrogate
        self.protocol = protocol
        self.is_v2 = isinstance(protocol, P2.ProtocolV2)
        self.state_dir = state_dir
        self.settings = dict(settings or {})
        self.centre_series = int(centre_series)
        self._log_sink = log
        os.makedirs(state_dir, exist_ok=True)
        self.state_file = os.path.join(state_dir, "%s_state.json" % name)
        self.log_file = os.path.join(state_dir, "%s.log" % name)
        if os.path.exists(self.state_file):
            with open(self.state_file) as fh:
                self.state = json.load(fh)
            self.vectors = np.array(self.state["vectors"], dtype=float)
            self.symbols = list(self.state["symbols"])
        else:
            vectors = np.atleast_2d(np.asarray(vectors, dtype=float))
            norms = np.linalg.norm(vectors, axis=1)
            if np.abs(norms - 1.0).max() > 1e-6:
                raise ValueError("search directions must have unit norm (found %s)" % np.round(norms, 6))
            start = np.asarray(start, dtype=float).reshape(-1, 3)
            if vectors.shape[1] != start.size:
                raise ValueError("direction length %d does not match %d atoms" % (vectors.shape[1], len(start)))
            curvature = np.asarray(curvature, dtype=float)
            if len(curvature) != len(vectors):
                raise ValueError("%d curvatures for %d directions" % (len(curvature), len(vectors)))
            self.vectors = vectors
            self.symbols = list(symbols)
            self.state = {
                "name": name, "protocol": "v2" if self.is_v2 else "v1", "status": RUNNING_STATUS,
                "symbols": self.symbols, "vectors": vectors.tolist(), "iteration": 1, "iters": {},
                "message": "",
                # role of every direction: +1 = the energy is minimized along it, -1 = maximized (the reaction
                # coordinate of a saddle search). Taken from the sign of the surrogate curvature at the start.
                "roles": [1 if float(c) > 0 else -1 for c in curvature],
                "trust_scale": 1.0, "rejected": 0,
            }
            self._new_iteration(1, start, lam=[float(k) / 2.0 for k in curvature], half_span_prev=None)
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

    # ------------------------------------------------------------------ building an iteration
    def _half_span(self, lam: Optional[float], previous: Optional[float]) -> float:
        """Outer displacement R of a direction: sqrt(e_cap / |Lambda|). Falls back to the previous
        half-span when no curvature is known."""
        if lam is None or lam == 0 or not np.isfinite(lam):
            if previous is None:
                raise ValueError("no curvature and no previous half-span for a direction")
            return float(previous)
        if self.is_v2:
            return self.protocol.half_span(lam)
        return float((self.protocol.e_cap / abs(lam)) ** 0.5)

    def _scaled_span(self, j: int, R: float) -> float:
        """The span that is used for direction j when the rule of the protocol gives ``R``."""
        return float(R)

    def _direction_span(self, j: int, lam: Optional[float], previous: Optional[float]) -> float:
        """Span of direction j for a new iteration."""
        return self._scaled_span(j, self._half_span(lam, previous))

    def _drop(self, energy: np.ndarray) -> int:
        """Number of start-up blocks discarded from a trace."""
        return int(self.protocol.drop)

    def _n_centre_series(self, k: int) -> int:
        """Independent runs at the centre of iteration k."""
        return self.centre_series

    def _new_iteration(self, k: int, centre: np.ndarray, lam: List[Optional[float]],
                       half_span_prev: Optional[List[float]]) -> None:
        pts: Dict[str, Dict[str, str]] = {}
        spans = []
        for j in range(self.dim):
            R = self._direction_span(j, lam[j], None if half_span_prev is None else half_span_prev[j])
            spans.append(R)
            pts[str(j)] = {}
            for f in self.protocol.shells:
                t = round(f * R, 3)
                if t <= 0:
                    raise ValueError("direction %d: shell displacement %.4f A is not positive" % (j, t))
                for sgn in (1.0, -1.0):
                    pts[str(j)][_tkey(sgn * t)] = self._key(k, _point_name(j, sgn * t))
        self.state["iters"][str(k)] = {
            "centre": np.asarray(centre, dtype=float).reshape(-1, 3).tolist(),
            "lam": [None if v is None else float(v) for v in lam], "half_span": spans, "points": pts,
            "decided": {}, "rescued": {}, "stopped": [],
            "centre_keys": [self._key(k, "centre" if s == 1 else "centre_s%d" % s)
                            for s in range(1, self._n_centre_series(k) + 1)],
            "surrogate_tag": self._key(k, "surrogate"), "composed": None, "prior_g": None,
            "base": None, "accepted": None,
        }
        self.state["iteration"] = k

    def _geometry(self, it: dict, j: int, t: float) -> np.ndarray:
        return np.array(it["centre"], dtype=float) + t * self.vectors[j].reshape(-1, 3)

    def _point_blocks(self) -> int:
        return int(self.protocol.cap)

    def _status_after_attention(self) -> str:
        return RUNNING_STATUS

    def clear_attention(self) -> str:
        """Set a ``NEEDS_ATTENTION`` status back to running, after the cause was repaired (for example a failed
        point that was submitted again). The next :meth:`tick` checks everything again."""
        if self.state["status"] == NEEDS_ATTENTION:
            self.state["status"] = self._status_after_attention()
            self.state["message"] = ""
            self.say("the NEEDS_ATTENTION status was cleared by request; checking again")
            self._save()
        return self.state["status"]

    def _ensure_points(self, k: int, it: dict) -> None:
        if self.surrogate is not None:
            self.surrogate.request(it["surrogate_tag"], np.array(it["centre"]), hessian=True, gradient=True)
        self.engine.submit(PointSpec(key=it["centre_keys"][0], symbols=self.symbols, coords=np.array(it["centre"]),
                                     blocks=int(self.protocol.cap), role="centre", series=1, settings=self.settings))
        scan_settings = self.settings
        if self.settings.get("jastrow_reuse") == "centre":
            # scan points start their Jastrow optimization from the Jastrow factor of the centre of this iteration
            scan_settings = dict(self.settings, jastrow_start_key=it["centre_keys"][0])
        for j in range(self.dim):
            if str(j) in it["decided"] or it.get("closed"):
                continue
            for tk, key in it["points"][str(j)].items():
                self.engine.submit(PointSpec(key=key, symbols=self.symbols, coords=self._geometry(it, j, float(tk)),
                                             blocks=self._point_blocks(), role="scan", series=1, settings=scan_settings))

    def _centre_extension(self, it: dict) -> None:
        """Further independent series at the centre, started once the first one writes blocks."""
        first = self.engine.read(it["centre_keys"][0])
        if first.nblocks == 0:
            return
        for s, key in enumerate(it["centre_keys"][1:], start=2):
            self.engine.submit(PointSpec(key=key, symbols=self.symbols, coords=np.array(it["centre"]),
                                         blocks=int(self.protocol.cap), role="centre", series=s,
                                         base_key=it["centre_keys"][0], settings=self.settings))

    # ------------------------------------------------------------------ reading data
    def _stat(self, key: str, n: Optional[int] = None) -> Optional[Tuple[float, float, int]]:
        """Reblocked mean and error (hartree) of a point after the equilibration blocks, or None."""
        d = self.engine.read(key)
        m = d.nblocks if n is None else min(n, d.nblocks)
        drop = self._drop(d.energy[:m])
        if m <= drop + self.protocol.min_blocks_after_drop:
            return None
        mean, err = reblock(d.energy[drop:m], d.weight[drop:m])
        return mean, err, m

    def _stop_points(self, it: dict, j: int) -> int:
        n = 0
        for key in it["points"][str(j)].values():
            if key in it["stopped"]:
                continue
            st = self.engine.read(key).state
            if st in (RUNNING, QUEUED):
                self.engine.stop(key); it["stopped"].append(key); n += 1
        return n

    # ------------------------------------------------------------------ one pass
    def tick(self) -> str:
        if self.state["status"] != RUNNING_STATUS:
            return self.state["status"]
        k = self.state["iteration"]; it = self._it()
        self._ensure_points(k, it)
        n_dec = 0
        for j in range(self.dim):
            if str(j) in it["decided"]:
                n_dec += 1
                self._stop_points(it, j)
                continue
            if self._evaluate_direction(k, it, j):
                n_dec += 1
        self._centre_extension(it)
        self.say("iteration %d: %d of %d directions decided" % (k, n_dec, self.dim))
        if n_dec == self.dim and self.state["status"] == RUNNING_STATUS:
            self._compose(k, it)
        self._save()
        return self.state["status"]

    def _direction_points(self, it: dict, j: int) -> Dict[float, str]:
        return {float(tk): key for tk, key in it["points"][str(j)].items()}

    def _evaluate_direction(self, k: int, it: dict, j: int) -> bool:
        """Apply the stopping rules to one open direction. Returns True when it is now decided."""
        p = self.protocol
        pts = self._direction_points(it, j)
        ts = sorted(t for t in pts if t > 0 and -t in pts)
        pairs, blocks, missing = [], 10 ** 9, []
        states = {t: self.engine.read(pts[t]).state for t in pts}
        for t in ts:
            dp, dm = self.engine.read(pts[t]), self.engine.read(pts[-t])
            n = min(dp.nblocks, dm.nblocks)
            if n <= p.drop + p.min_blocks_after_drop:
                missing.append(t); continue
            blocks = min(blocks, n)
            mp, sp = reblock(dp.energy[p.drop:n], dp.weight[p.drop:n])
            mm, sm = reblock(dm.energy[p.drop:n], dm.weight[p.drop:n])
            pairs.append((t, mp, sp, mm, sm))
        running = [t for t in ts for s in (t, -t) if states[s] in (RUNNING, QUEUED)]
        dead = [s for s in pts if states[s] == FAILED or
                (states[s] in (DONE, STOPPED) and self.engine.read(pts[s]).nblocks <= p.drop + p.min_blocks_after_drop
                 and pts[s] not in it["stopped"])]
        if dead:
            self.state["status"] = NEEDS_ATTENTION
            self.state["message"] = "iteration %d direction %d: point(s) %s ended without usable data" % (
                k, j, ", ".join(pts[s] for s in dead))
            self.say(self.state["message"])
            return False
        if len(pairs) < 2:
            self.say("d%d: waiting (%d of %d shells have more than %d blocks on both sides)"
                     % (j, len(pairs), len(ts), p.drop + p.min_blocks_after_drop))
            return False
        res = P1.analyse(pairs)
        eps_e = p.eps_total / self.dim
        st, note = P1.decide(res, blocks, eps_e=eps_e, protocol=p)
        if not running and not missing and st == "CONTINUE":
            # no more blocks will come (the jobs ended or were stopped): the data are final -> cap verdict
            st, note = P1.decide(res, p.cap, eps_e=eps_e, protocol=p)
            note += " [points no longer running: final data at %d blocks]" % blocks
        bound = res["lam"] * (abs(res["m"]) + 2 * res["sm"]) ** 2
        if st == "CONTINUE" or (blocks < p.cap and running and st not in ("DISPLACED", "ZERO")):
            self.say("d%d: CONTINUE at %d blocks (%.1f sigma, chi2/dof %.2f, m %+.4f +/- %.4f, bound %.3f)%s"
                     % (j, blocks, res["z"], res["chi2"], res["m"], res["sm"], bound,
                        "" if not missing else "; shells not yet usable: %s" % missing))
            return False
        if st == "RESCUE":
            # A rejected linear model or a failed bracket at the cap ADDS a shell at 1.5 R and refits.
            t15 = round(1.5 * res["R"], 3)
            done = it["rescued"].get(str(j))
            exhausted = False
            if done or t15 in pts or -t15 in pts:
                if running or missing:
                    self.say("d%d: RESCUE at %d blocks [%s]; the shell at %.3f A exists, waiting for it"
                             % (j, blocks, note, done["t"] if done else t15))
                else:
                    exhausted = True
                    if res["z"] >= 2.0 and not np.isnan(res["m"]) and abs(res["m"]) < res["R"]:
                        st = "DISPLACED"
                        note = "rescue exhausted: linear 2 sigma verdict (%.1f sigma, chi2/dof %.2f) [%s]" % (
                            res["z"], res["chi2"], note)
                    else:
                        st = "ZERO"
                        note = "rescue exhausted: not resolved (%.1f sigma, chi2/dof %.2f, bound %.4f) [%s]" % (
                            res["z"], res["chi2"], bound if not np.isnan(res["m"]) else float("nan"), note)
            else:
                for sgn in (1.0, -1.0):
                    key = self._key(k, _point_name(j, sgn * t15))
                    it["points"][str(j)][_tkey(sgn * t15)] = key
                    self.engine.submit(PointSpec(key=key, symbols=self.symbols, coords=self._geometry(it, j, sgn * t15),
                                                 blocks=self._point_blocks(), role="scan", series=1, settings=self.settings))
                it["rescued"][str(j)] = {"t": t15, "note": note}
                self.say("d%d: RESCUE at %d blocks [%s] -> built the shell at %.3f A; the direction stays open"
                         % (j, blocks, note, t15))
            if not exhausted:
                return False
        if st == "DISPLACED_CUBIC":
            m = res.get("m_cubic", res["m"])
        elif st.startswith("DISPLACED"):
            m = res["m"]
        else:
            m = 0.0
        it["decided"][str(j)] = {
            "status": st, "m": float(m), "blocks": int(blocks), "note": note,
            "res": {key: (None if (isinstance(v, float) and np.isnan(v)) else v) for key, v in res.items()},
        }
        stopped = self._stop_points(it, j)
        self.say("d%d: %s -> step %+.4f A  [%s]  stopped %d running point(s)" % (j, st, m, note, stopped))
        return True

    def _surrogate_now(self, it: dict):
        if self.surrogate is None:
            return None
        res = self.surrogate.result(it["surrogate_tag"])
        if res is None or res.failed:
            return None
        return res

    def _centre_stat(self, it: dict) -> Optional[Tuple[float, float]]:
        """Combined mean and error (hartree) of the centre energy over its series, or None."""
        vals = []
        for key in it["centre_keys"]:
            s = self._stat(key)
            if s is not None:
                vals.append(s)
        if not vals:
            return None
        return combine_series(vals)

    # ------------------------------------------------------------------ composing the step
    def _hessian_in_directions(self, it: dict) -> Optional[np.ndarray]:
        sur = self._surrogate_now(it)
        if sur is None:
            return None
        if sur.hessian_directions is not None:
            K = np.asarray(sur.hessian_directions, dtype=float)
            return 0.5 * (K + K.T)
        if sur.hessian is None:
            return None
        H = np.asarray(sur.hessian, dtype=float)
        return self.vectors @ (0.5 * (H + H.T)) @ self.vectors.T

    def _surrogate_curvature(self, sur, j: int) -> Optional[float]:
        """Surrogate force constant along direction j at a centre (kcal/mol/A^2), or None."""
        if sur is None:
            return None
        if sur.hessian_directions is not None:
            return float(np.asarray(sur.hessian_directions)[j, j])
        if sur.hessian is None:
            return None
        u = self.vectors[j]
        return float(u @ np.asarray(sur.hessian) @ u)

    def _compose(self, k: int, it: dict) -> None:
        p = self.protocol
        n = self.dim
        dec = [it["decided"][str(j)] for j in range(n)]
        lam: List[Optional[float]] = []
        step = np.zeros(self.vectors.shape[1]); bound = 0.0; any_disp = False
        for j, d in enumerate(dec):
            r = d["res"]
            lj = r["lam"] if (r["lam"] and r["lam"] > 0) else None
            lam.append(lj)
            if r["m"] is not None and r["sm"] is not None and lj:
                bound += lj * (abs(r["m"]) + 2 * r["sm"]) ** 2
            if d["status"].startswith("DISPLACED"):
                any_disp = True
                step = step + d["m"] * self.vectors[j]
        rule = "diagonal"
        comp = [float(d["m"]) if d["status"].startswith("DISPLACED") else 0.0 for d in dec]
        self.say("iteration %d composed (diagonal rule): |step| = %.5f A, remaining-energy bound %.3f kcal/mol, components %s"
                 % (k, float(np.linalg.norm(step)), bound, [round(c, 4) for c in comp]))
        Kd = self._hessian_in_directions(it)
        spans = [max(t for t in self._direction_points(it, j) if t > 0) for j in range(n)]
        if Kd is not None:
            # Coupled Newton step: DMC diagonal (2 Lambda_j) and surrogate off-diagonal elements.
            Kh = Kd.copy(); g = np.zeros(n)
            for j, d in enumerate(dec):
                if lam[j]:
                    Kh[j, j] = 2 * lam[j]
                if d["status"].startswith("DISPLACED"):
                    g[j] = -2.0 * lam[j] * d["m"] if lam[j] else d["res"]["b"] / 2.0
            s = -np.linalg.solve(Kh, g)
            self.say("  surrogate Hessian at the centre found: largest normalized coupling %.3f; coupled Newton step %s, |s| = %.5f A"
                     % (max_coupling(Kd), np.round(s, 4).tolist(), float(np.linalg.norm(s))))
            if np.all(np.abs(s) < np.array(spans)):
                step = s @ self.vectors; rule = "coupled-newton"; comp = [float(v) for v in s]
                sd = -g / np.diag(Kh)
                self.say("  -> using the coupled step (predicted gain %+.3f kcal/mol against %+.3f for the diagonal step)"
                         % (g @ s + 0.5 * s @ Kh @ s, g @ sd + 0.5 * sd @ Kh @ sd))
            else:
                self.say("  -> the coupled step leaves the scanned span on some direction; keeping the diagonal step")
        else:
            self.say("  no surrogate Hessian at this centre: diagonal step")
        it["composed"] = {"rule": rule, "components": comp, "step_norm": float(np.linalg.norm(step)),
                          "bound": float(bound), "any_displaced": bool(any_disp)}
        if (not any_disp) and bound < p.eps_total:
            self.state["status"] = CONVERGED
            self.state["message"] = "converged at iteration %d: no direction displaced and bound %.3f < %.2f kcal/mol" % (
                k, bound, p.eps_total)
            self.say(self.state["message"]); return
        if k >= p.max_iter:
            self.state["status"] = NOT_CONVERGED
            self.state["message"] = "not converged after %d iterations (bound %.3f kcal/mol)" % (k, bound)
            self.say(self.state["message"]); return
        new_centre = np.array(it["centre"]) + step.reshape(-1, 3)
        self._new_iteration(k + 1, new_centre, lam=lam, half_span_prev=spans)
        self._ensure_points(k + 1, self._it(k + 1))
        self.say("iteration %d built: %d scan points about the new centre" % (
            k + 1, sum(len(v) for v in self._it(k + 1)["points"].values())))

    # ------------------------------------------------------------------ results
    def final_iteration(self) -> int:
        """Number of the last iteration whose centre was not rejected."""
        ks = sorted((int(x) for x in self.state["iters"]), reverse=True)
        for k in ks:
            if self.state["iters"][str(k)].get("accepted") is not False:
                return k
        return ks[-1]

    def final_geometry(self) -> np.ndarray:
        """Centre of the last accepted iteration (the converged geometry when the status is CONVERGED)."""
        return self.centre(self.final_iteration())

    def trajectory(self) -> List[np.ndarray]:
        """Centres of all iterations, in order."""
        return [self.centre(k) for k in sorted(int(s) for s in self.state["iters"])]

    def centre_energy(self, k: Optional[int] = None) -> Optional[Tuple[float, float]]:
        """Combined DMC energy (hartree) and error of the centre of iteration k, or None."""
        return self._centre_stat(self._it(k))

    def total_blocks(self) -> int:
        """All DMC blocks acquired by this search so far (scan points and centres)."""
        n = 0
        for it in self.state["iters"].values():
            for key in it["centre_keys"]:
                n += self.engine.read(key).nblocks
            for pts in it["points"].values():
                for key in pts.values():
                    n += self.engine.read(key).nblocks
        return n

    def run(self, max_ticks: int = 10000) -> str:
        """Tick until the search ends (for the mock engine and for tests)."""
        for _ in range(max_ticks):
            self.engine.advance()
            if self.tick() != RUNNING_STATUS:
                break
        return self.state["status"]
