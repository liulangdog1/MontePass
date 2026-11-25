"""Reads a project folder and writes the tables of the results folder.

:func:`collect` reads everything that the report needs into one :class:`ReportData`
object: the project state, the path, the subspace, and for every target the start
geometry, the search state and the DMC energies of its points. It only reads. A part
that is missing or cannot be read is recorded in ``ReportData.missing`` with the
reason, and the rest is still collected.

The other functions turn that object into files: ``summary.json``, ``summary.md``,
``energies.csv``, the xyz files and ``bond_table.csv``.

Conventions of every table
--------------------------
* The energy of one geometry is in hartree. Energy differences are in kcal/mol.
  Lengths are in angstrom, angles in degrees.
* Every error bar is 1 standard error. A number without statistical error is marked as exact.
* The DMC energy of a geometry is computed in the same way as in the search drivers: the
  start-up blocks of each run are discarded, the mean and its reblocked error are taken
  (``stats.reblock``), and independent runs of one geometry are combined with weights
  1 / error^2.
"""
from __future__ import annotations

import configparser
import csv
import json
import os
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .. import geometry as G
from ..engine.base import ABSENT, DONE, FAILED, STOPPED, PointData
from ..search import protocol_v1 as P1
from ..search import protocol_v2 as P2
from ..stats import reblock
from ..subspace import Subspace
from ..units import HARTREE_TO_KCAL

ENDED = ("CONVERGED", "NOT_CONVERGED", "DONE")   # states of a search that has ended (DONE: the classic line search)
CLASSIC_MIN_USABLE = 8                           # the classic search uses a point once it has this many blocks after the drop
SIGMA_LIMIT = 2.0                                # an energy change counts as significant above this many standard errors
IMAGE_MATCH_LIMIT = 0.15                         # angstrom: a target farther than this (RMSD) from every image is not on the path
NUMERICAL_ZERO = 1e-12                           # explained variance ratios below this are rounding noise


class Skip(Exception):
    """An item of the report cannot be made. The message says why. This is not an error of the program."""


# ============================================================================= small tools
def ignore_library_deprecations() -> None:
    """Add warning filters (in front of the filters now in force) that hide deprecation warnings raised
    inside matplotlib, its parser library pyparsing, and Pillow, and the grammar diagnostics of pyparsing.

    Such a warning is about how one library uses another one (for example matplotlib 3.9 calls functions
    that pyparsing 3.3 has renamed). It says nothing about a figure, and it must not stop a report whose
    caller turns warnings into errors. Warnings from the code of this package are not hidden.
    Call this inside ``warnings.catch_warnings()``, so that the filters are removed again."""
    import warnings

    categories = [DeprecationWarning, PendingDeprecationWarning]
    try:                                         # pyparsing 3.3 has its own class for these warnings
        from pyparsing.warnings import PyparsingDeprecationWarning
        categories.append(PyparsingDeprecationWarning)
    except Exception:
        pass
    for module in ("matplotlib", "pyparsing", "PIL"):
        for category in categories:
            warnings.filterwarnings("ignore", category=category, module=r"%s(\.|$)" % module)
    # pyparsing switches its own diagnostics on when Python runs with a -W option. matplotlib 3.7 then gets
    # messages such as "warn_name_set_on_empty_Forward: ..." about the grammar of its text parser while it
    # draws the first text. They are about that grammar and say nothing about a figure.
    warnings.filterwarnings("ignore", message=r"warn_[a-z_]+: ")


def atomic_write(path: str, text: str) -> None:
    """Write a text file through a temporary file, so that a reader never sees a half-written file."""
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        fh.write(text)
    os.replace(tmp, path)


def _num(value) -> Optional[float]:
    """A finite float, or None for None, NaN, infinity and things that are not numbers."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def jsonable(obj):
    """Copy of a nested structure that ``json.dump`` accepts: numpy types become Python types, and numbers
    that are not finite become None (JSON has no NaN)."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj) if np.isfinite(obj) else None
    return obj


def fmt_pm(value: Optional[float], error: Optional[float], digits: int = 2) -> str:
    """``value +/- error`` with as many decimals as ``digits`` significant digits of the error need.
    An error of exactly 0 is written as ``(exact)``. None gives ``n/a``."""
    if value is None:
        return "n/a"
    if error is None:
        return "%.6g" % value
    if error <= 0.0:
        return "%.6f (exact)" % value
    decimals = int(min(10, max(0, digits - 1 - int(np.floor(np.log10(error))))))
    return "%.*f +/- %.*f" % (decimals, value, decimals, error)


def fmt_num(value: Optional[float], spec: str = "%.4f") -> str:
    return "n/a" if value is None else spec % value


# ============================================================================= statistics of the DMC traces
@dataclass(frozen=True)
class DropRule:
    """How many start-up blocks of a DMC trace are discarded, and how many blocks a point needs to be used.

    ``drop`` is a fixed number of blocks or ``"auto"``. With ``"auto"`` the number is found for every trace
    by the MSER-5 rule (``protocol_v2.mser_drop``) inside ``drop_min`` .. ``drop_max``. These are the rules
    of the two dense-scan protocols, so the report gives the same energies as the search drivers.
    """

    drop: object = "auto"
    drop_min: int = 10
    drop_max: int = 50
    min_blocks_after_drop: int = 30

    @staticmethod
    def of(protocol) -> "DropRule":
        """The rule of a protocol object, or of a protocol label (``"v1"``; any other label gives the v2 rule)."""
        if isinstance(protocol, DropRule):
            return protocol
        if protocol == "v1":
            protocol = P1.V1
        elif protocol is None or isinstance(protocol, str):
            protocol = P2.V2
        return DropRule(drop=getattr(protocol, "drop", "auto"), drop_min=int(getattr(protocol, "drop_min", 10)),
                        drop_max=int(getattr(protocol, "drop_max", 50)),
                        min_blocks_after_drop=int(getattr(protocol, "min_blocks_after_drop", 30)))

    def discard(self, energy: np.ndarray) -> int:
        if self.drop == "auto":
            return P2.mser_drop(energy, self.drop_min, self.drop_max)
        return int(self.drop)


def trace_stat(energy: np.ndarray, weight: np.ndarray, rule: DropRule) -> Optional[Tuple[float, float, int]]:
    """Mean (hartree), reblocked standard error and number of discarded blocks of one trace.
    None if the trace is too short to be used."""
    n = len(energy)
    drop = rule.discard(energy)
    if n <= drop + rule.min_blocks_after_drop:
        return None
    mean, err = reblock(energy[drop:], weight[drop:])
    return mean, err, drop


def combine(parts: Sequence[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
    """Mean of independent runs with weights 1 / error^2, and its standard error. A run with error 0
    (a model without noise) is exact: then the mean of the exact runs is returned with error 0."""
    if not parts:
        return None
    m = np.array([p[0] for p in parts], dtype=float)
    e = np.array([p[1] for p in parts], dtype=float)
    if np.any(e <= 0.0):
        return float(m[e <= 0.0].mean()), 0.0
    w = 1.0 / e ** 2
    return float(np.sum(w * m) / w.sum()), float(1.0 / np.sqrt(w.sum()))


@dataclass
class Stat:
    """DMC energy of one geometry."""

    value: float            # hartree
    error: float            # hartree, 1 standard error
    blocks: int             # blocks written by all runs of this geometry
    used: int               # blocks left after the start-up blocks were discarded
    series: int             # number of independent runs that were combined
    complete: bool          # every run of this geometry has ended

    def as_dict(self) -> dict:
        return {"value": self.value, "error": self.error, "blocks": self.blocks, "blocks_used": self.used,
                "runs": self.series, "complete": self.complete}


class Reader:
    """Reads points from the engine once and keeps them. A read that fails is recorded, not raised."""

    def __init__(self, engine):
        self.engine = engine
        self._data: Dict[str, PointData] = {}
        self.problems: List[str] = []

    def read(self, key: str) -> PointData:
        if key not in self._data:
            if self.engine is None:
                self._data[key] = PointData(key=key, state=ABSENT)
            else:
                try:
                    self._data[key] = self.engine.read(key)
                except Exception as exc:                     # one unreadable point must not stop the report
                    self.problems.append("point %s could not be read: %s: %s" % (key, type(exc).__name__, exc))
                    self._data[key] = PointData(key=key, state=FAILED, message=str(exc))
        return self._data[key]

    def cost(self, key: str) -> float:
        if self.engine is None:
            return 0.0
        try:
            return float(self.engine.cost(key))
        except Exception:
            return 0.0

    def stat(self, keys: Sequence[str], rule: DropRule) -> Optional[Stat]:
        """Combined energy of the runs ``keys`` of one geometry, or None if no run has enough blocks."""
        parts, blocks, used, complete = [], 0, 0, True
        for key in dict.fromkeys(keys):
            d = self.read(key)
            blocks += d.nblocks
            if d.state not in (DONE, STOPPED):
                complete = False
            st = trace_stat(d.energy, d.weight, rule)
            if st is not None:
                parts.append(st[:2])
                used += d.nblocks - st[2]
        both = combine(parts)
        if both is None:
            return None
        return Stat(value=both[0], error=both[1], blocks=blocks, used=used, series=len(parts), complete=complete)


# ============================================================================= the data of one report
@dataclass
class ScanPoint:
    t: float                            # displacement along the direction, angstrom
    key: str
    stat: Optional[Stat]
    e_surrogate: Optional[float]        # hartree


@dataclass
class IterationData:
    number: int
    centre: np.ndarray
    accepted: Optional[bool]            # False = the step into this centre was rejected
    base: Optional[int]                 # iteration from whose centre this one was built
    energy: Optional[Stat]              # DMC energy of the centre
    verdict: str
    rule: Optional[str]
    blocks: Optional[int]               # blocks per point when the iteration was decided
    G: Optional[float]                  # predicted gain, kcal/mol (protocol v2)
    G_error: Optional[float]            # standard deviation of G, kcal/mol (protocol v2: sqrt(V))
    N: Optional[float]                  # expected noise energy, kcal/mol (protocol v2)
    joint_bound: Optional[float]        # joint 95 % bound c_n N, kcal/mol (protocol v2)
    bound_v1: Optional[float]           # remaining-energy bound, kcal/mol (protocol v1)
    step_length: Optional[float]        # norm of the step composed at this centre, angstrom
    step_error: Optional[float]         # statistical error of the step length, angstrom
    components: Optional[List[float]]   # the step in the direction basis, angstrom
    energy_change: Optional[Tuple[float, float]]    # centre energy minus the energy of the base centre, kcal/mol
    note: str
    scans: Dict[int, List[ScanPoint]] = field(default_factory=dict)
    estimates: Dict[int, dict] = field(default_factory=dict)
    classes: Optional[List[str]] = None

    @property
    def has_scan_data(self) -> bool:
        return any(p.stat is not None for pts in self.scans.values() for p in pts)


@dataclass
class PointRow:
    """One line of ``energies.csv``."""

    key: str
    role: str
    state: str
    blocks: int
    energy: Optional[float]
    error: Optional[float]
    e_surrogate: Optional[float]
    target: str
    iteration: Optional[int]
    direction: Optional[int]
    t: Optional[float]
    discarded: Optional[int]
    core_hours: float


@dataclass
class TargetData:
    name: str
    kind: str = "unknown"               # "minimum", "saddle" or "unknown"
    protocol: str = "unknown"
    status: str = "NO SEARCH STATE"
    message: str = ""
    symbols: Optional[List[str]] = None
    start: Optional[np.ndarray] = None
    vectors: Optional[np.ndarray] = None
    curvature: Optional[np.ndarray] = None
    roles: Optional[List[int]] = None
    state: Optional[dict] = None
    rule: DropRule = field(default_factory=DropRule)
    eps_total: Optional[float] = None
    iterations: List[IterationData] = field(default_factory=list)
    rejected: int = 0
    ended: bool = False
    final: Optional[np.ndarray] = None              # final geometry; None while the search runs
    final_iteration: Optional[int] = None
    final_source: str = ""
    moved: Optional[bool] = None                    # False = the final geometry is the starting geometry
    start_energy: Optional[Stat] = None
    final_energy: Optional[Stat] = None
    difference: Optional[Tuple[float, float]] = None        # final minus start, kcal/mol
    driver_difference: Optional[Tuple[float, float]] = None   # the same number as stored by the search driver
    points: List[PointRow] = field(default_factory=list)
    blocks_search: int = 0
    blocks_total: int = 0
    core_hours_search: float = 0.0
    core_hours_total: float = 0.0
    position_cov: Optional[np.ndarray] = None       # (d, d) covariance of the final geometry in the direction basis, A^2
    image: Optional[int] = None                     # number of the nearest path image (from 0)
    image_rmsd: Optional[float] = None
    problems: List[str] = field(default_factory=list)

    def iteration(self, number: int) -> Optional[IterationData]:
        for it in self.iterations:
            if it.number == number:
                return it
        return None

    def trajectory(self) -> List[np.ndarray]:
        """Centres of all iterations, in order."""
        return [it.centre for it in self.iterations]

    def last_scan_iteration(self) -> Optional[IterationData]:
        """The last iteration that has DMC data on its scan points."""
        for it in reversed(self.iterations):
            if it.has_scan_data:
                return it
        return None


@dataclass
class PathData:
    symbols: List[str]
    frames: List[np.ndarray]
    energies: Optional[np.ndarray]      # surrogate energy of each image in hartree, or None if not all are known


@dataclass
class ReportData:
    project_dir: str
    title: str
    project: dict
    path: Optional[PathData] = None
    subspace: Optional[Subspace] = None
    targets: Dict[str, TargetData] = field(default_factory=dict)
    settings: Optional[dict] = None
    missing: List[Tuple[str, str]] = field(default_factory=list)     # (what, why) for parts that could not be read
    problems: List[str] = field(default_factory=list)

    @property
    def results_dir(self) -> str:
        return os.path.join(self.project_dir, "results")


# ============================================================================= reading the project
_ENERGY_IN_COMMENT = re.compile(r"(?:^|\s)E\s+([-+]?(?:\d+\.?\d*|\.\d+)(?:[eEdD][-+]?\d+)?)(?=\s|$)")


def comment_energy(comment: str) -> Optional[float]:
    """The number after ``E`` in the comment line of an xyz frame (hartree), or None."""
    m = _ENERGY_IN_COMMENT.search(comment)
    if not m:
        return None
    return float(m.group(1).replace("d", "e").replace("D", "e"))


def _read_json(path: str) -> dict:
    with open(path) as fh:
        text = fh.read()
    if not text.strip():
        raise ValueError("the file is empty")
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("the file does not hold a JSON object")
    return data


def _input_tolerance(project_dir: str) -> Optional[float]:
    """``eps_total`` of section ``[search]`` of the input file, if the file names it."""
    path = os.path.join(project_dir, "input.ini")
    if not os.path.exists(path):
        return None
    cp = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=(";",))
    try:
        cp.read(path)
        return float(cp.get("search", "eps_total"))
    except (configparser.Error, ValueError):
        return None


def final_iteration(state: dict) -> int:
    """Number of the last iteration whose centre was not rejected (the rule of the search drivers)."""
    ks = sorted((int(k) for k in state["iters"]), reverse=True)
    for k in ks:
        if state["iters"][str(k)].get("accepted") is not False:
            return k
    return ks[-1]


def _estimate(protocol: str, dec: dict) -> dict:
    """The estimate of one direction in one common form.

    Keys: slope ``g`` with error ``sg`` (kcal/mol/A), force constant ``k`` with error ``sk``
    (kcal/mol/A^2, E = 0.5 k t^2), ``k_source``, stationary point ``m`` with error ``sm`` (A), the step
    component ``step`` that the search took along the direction (A), ``status`` and ``by_symmetry``.
    Protocol v1 stores the slope b of E(+t) - E(-t) and Lambda = k / 2; they are converted here.
    """
    res = dec.get("res") or {}
    out = {"status": dec.get("status", ""), "step": _num(dec.get("m")), "note": dec.get("note", ""),
           "by_symmetry": res.get("by_symmetry"), "k_source": res.get("k_source"), "coefficients": None}
    if "coefficients" in res:                # classic line search: a cubic fit; slope at t = 0, curvature at the step
        coefficients = [_num(c) for c in res.get("coefficients") or []]
        if len(coefficients) >= 3 and all(c is not None for c in coefficients):
            out["coefficients"] = coefficients
        out.update(g=_num(res.get("slope")), sg=None, k=_num(res.get("k")), sk=None)
        out["k_source"] = "cubic fit"
    elif "g" in res or "k" in res:           # protocol v2
        out.update(g=_num(res.get("g")), sg=_num(res.get("sg")), k=_num(res.get("k")), sk=_num(res.get("sk")))
        # cubic part of the surrogate along the scan (kcal/mol/A) and the span: the search corrects the slope
        # of the DMC pair with it, so a curve through the points needs it too
        if res.get("control_variate"):
            out.update(shape=_num(res.get("shape")), span=_num(res.get("R")))
    else:                                    # protocol v1
        b, sb, lam = _num(res.get("b")), _num(res.get("sb")), _num(res.get("lam"))
        out.update(g=None if b is None else b / 2.0, sg=None if sb is None else sb / 2.0,
                   k=None if lam is None else 2.0 * lam, sk=None)
        out["k_source"] = "measured"
    out.update(m=_num(res.get("m")), sm=_num(res.get("sm")), z=_num(res.get("z")))
    return out


def step_covariance(it: dict) -> Optional[np.ndarray]:
    """Covariance (A^2), in the basis of the search directions, of the stationary point that one iteration
    found relative to its centre. It comes from the slope errors of the scan; the curvatures of the step
    model are taken as exact. ``it`` is one entry of ``state["iters"]``. None if the entry does not hold
    what is needed.

    Protocol v2: the step is s = -K^-1 g, so cov(s) = K^-1 diag(sg^2) K^-T (K: the model Hessian). For a
    diagonal step it is diag((sg_j / k_j)^2).
    Protocol v1: the minimum along direction j is known to ``sm_j``, and the directions are independent."""
    try:
        comp = it.get("composed") or {}
        if "sg" in comp:
            sg = np.array(comp["sg"], dtype=float)
            if str(comp.get("rule", "")).startswith("newton"):
                inv = np.linalg.inv(np.array(comp["Kh"], dtype=float))
                cov = np.dot(np.dot(inv, np.diag(sg ** 2)), inv.T)
            else:
                cov = np.diag((sg / np.array(comp["kdiag"], dtype=float)) ** 2)
        else:
            dec = it.get("decided") or {}
            if not dec:
                return None
            sm = [dec[str(j)]["res"]["sm"] for j in range(len(dec))]
            if any(v is None for v in sm):
                return None
            cov = np.diag(np.array(sm, dtype=float) ** 2)
    except (KeyError, TypeError, ValueError, IndexError, np.linalg.LinAlgError):
        return None
    return cov if np.all(np.isfinite(cov)) else None


def _step_error(it: dict) -> Optional[float]:
    """Statistical error (1 standard error, A) of the length of the step composed in one iteration:
    the standard deviation of the step along its own direction. None for a step of length 0."""
    comp = it.get("composed") or {}
    s = np.array(comp.get("components") or [], dtype=float)
    cov = step_covariance(it)
    norm = float(np.linalg.norm(s)) if s.size else 0.0
    if cov is None or norm <= 0.0 or cov.shape[0] != s.size:
        return None
    u = s / norm
    shrink = _num(comp.get("shrink"))
    return float(np.sqrt(max(float(np.dot(u, np.dot(cov, u))), 0.0))) * (1.0 if shrink is None else shrink)


def _position_covariance(protocol: str, state: dict, status: str) -> Optional[np.ndarray]:
    """Covariance (A^2) of the final geometry in the basis of the search directions: the covariance of the
    stationary point found by the last scan (:func:`step_covariance`). None if the search did not end with a
    final step (protocol v2) or did not converge (protocol v1), or if the state does not hold what is needed."""
    try:
        fin = state.get("final")
        if fin and fin.get("at_centre"):
            return None                         # the final geometry is a centre, not the end of a measured step
        if fin:
            return step_covariance(state["iters"][str(fin["iteration"])])
        if protocol == "v1" and status == "CONVERGED":
            return step_covariance(state["iters"][str(final_iteration(state))])
    except (KeyError, TypeError, ValueError):
        return None
    return None


def _read_target(name: str, info: dict, tdir: str, reader: Reader, protocol_object, tolerance: Optional[float],
                 path: Optional[PathData]) -> TargetData:
    t = TargetData(name=name, kind=str(info.get("kind", "unknown")), protocol=str(info.get("protocol", "unknown")),
                   status=str(info.get("status", "NO SEARCH STATE")), message=str(info.get("message", "") or ""))
    # ---- start geometry and directions
    start_file = os.path.join(tdir, "start.xyz")
    if os.path.exists(start_file):
        try:
            symbols, start = G.read_xyz(start_file)
            t.symbols, t.start = list(symbols), start
        except (ValueError, OSError) as exc:
            t.problems.append("start.xyz could not be read: %s" % exc)
    else:
        t.problems.append("start.xyz is missing")
    dfile = os.path.join(tdir, "directions.npz")
    if os.path.exists(dfile):
        try:
            with np.load(dfile, allow_pickle=False) as z:
                t.vectors, t.curvature = np.array(z["vectors"], dtype=float), np.array(z["curvature"], dtype=float)
        except (ValueError, OSError, KeyError) as exc:
            t.problems.append("directions.npz could not be read: %s" % exc)
    # ---- search state
    sfile = os.path.join(tdir, "search", "%s_state.json" % name)
    if os.path.exists(sfile):
        try:
            t.state = _read_json(sfile)
        except (ValueError, OSError) as exc:
            t.problems.append("search state %s could not be read: %s" % (os.path.basename(sfile), exc))
    state = t.state
    if state is not None:
        t.status = str(state.get("status", t.status))
        t.message = str(state.get("message", "") or t.message)
        if t.protocol in ("unknown", "auto") and state.get("protocol"):
            t.protocol = str(state["protocol"])
        if t.symbols is None and state.get("symbols"):
            t.symbols = [G.normalize_symbol(s) for s in state["symbols"]]
        if state.get("vectors") is not None:
            t.vectors = np.array(state["vectors"], dtype=float)          # the directions that the search used
        if state.get("roles"):
            t.roles = [int(r) for r in state["roles"]]
            if t.kind == "unknown":
                t.kind = "saddle" if any(r < 0 for r in t.roles) else "minimum"
        t.rejected = int(state.get("rejected", 0) or 0)
        for text in state.get("warnings") or []:
            t.problems.append("warning of the search: %s" % text)
    options = (state or {}).get("options")
    if protocol_object is not None:
        t.rule = DropRule.of(protocol_object)
    elif (state or {}).get("protocol") == "classic" and isinstance(options, dict) and options.get("drop") is not None:
        # the classic line search keeps its own rule for the start-up blocks in its state file
        t.rule = DropRule(drop=int(options["drop"]), min_blocks_after_drop=CLASSIC_MIN_USABLE - 1)
    else:
        t.rule = DropRule.of(t.protocol)
    t.eps_total = _num(getattr(protocol_object, "eps_total", None)) if protocol_object is not None else tolerance
    t.ended = t.status in ENDED
    if state is not None and isinstance(state.get("iters"), dict) and state["iters"]:
        try:
            _read_search(t, state, reader)
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            # nothing half-read may stay: a table built from a part of the state would look complete
            t.iterations, t.points, t.final, t.final_iteration, t.moved = [], [], None, None, None
            t.start_energy = t.final_energy = t.difference = t.driver_difference = t.position_cov = None
            t.blocks_search = t.blocks_total = 0
            t.core_hours_search = t.core_hours_total = 0.0
            t.problems.append("the search state has a layout that this report cannot read (%s: %s)"
                              % (type(exc).__name__, exc))
    elif state is not None:
        t.problems.append("the search state holds no iterations that this report can read (protocol %s)" % t.protocol)
    if t.start is None and t.iterations:
        t.start = t.iterations[0].centre.copy()
    # ---- place on the path
    if path is not None and t.start is not None and t.symbols == path.symbols:
        rmsds = [G.kabsch(t.start, f)[1] for f in path.frames]
        t.image = int(np.argmin(rmsds))
        t.image_rmsd = float(rmsds[t.image])
    return t


def _read_search(t: TargetData, state: dict, reader: Reader) -> None:
    """Fill the search part of a target from the state of a dense-scan driver."""
    rule = t.rule
    iters = state["iters"]
    numbers = sorted(int(k) for k in iters)
    seen_keys: Dict[str, bool] = {}        # key -> counts as search (True) or as a full-length reference run (False)

    def row(key: str, role: str, k: Optional[int], j: Optional[int], disp: Optional[float]) -> PointRow:
        d = reader.read(key)
        st = trace_stat(d.energy, d.weight, rule)
        if d.state == FAILED:
            t.problems.append("point %s failed%s" % (key, (": " + d.message) if d.message else ""))
        return PointRow(key=key, role=role, state=d.state, blocks=d.nblocks, energy=None if st is None else st[0],
                        error=None if st is None else st[1], e_surrogate=_num(d.e_surrogate), target=t.name, iteration=k,
                        direction=j, t=disp, discarded=None if st is None else st[2], core_hours=reader.cost(key))

    for k in numbers:
        it = iters[str(k)]
        centre = np.array(it["centre"], dtype=float).reshape(-1, 3)
        centre_keys = list(it.get("centre_keys") or [])
        for key in centre_keys:
            if key not in seen_keys:
                seen_keys[key] = (k != 1)
                t.points.append(row(key, "centre", k, None, None))
        scans: Dict[int, List[ScanPoint]] = {}
        for js, pts in (it.get("points") or {}).items():
            j = int(js)
            scans[j] = []
            for tk, key in sorted(pts.items(), key=lambda kv: float(kv[0])):
                if key not in seen_keys:
                    seen_keys[key] = True
                    t.points.append(row(key, "scan", k, j, float(tk)))
                d = reader.read(key)
                scans[j].append(ScanPoint(t=float(tk), key=key, stat=reader.stat([key], rule), e_surrogate=_num(d.e_surrogate)))
        decided = it.get("decided") or {}
        comp = it.get("composed") or None
        estimates = {int(js): _estimate(t.protocol, dec) for js, dec in decided.items()}
        accepted = it.get("accepted")
        blocks_decided = [int(dec["blocks"]) for dec in decided.values() if dec.get("blocks") is not None]
        if accepted is False:
            verdict = "REJECTED"
        elif comp is None:
            verdict = "open" if t.status in ("RUNNING", "FINALIZING") else "not decided"
        elif comp.get("verdict"):
            verdict = str(comp["verdict"])
        else:                                    # protocol v1: one verdict per direction
            counts = Counter(str(dec.get("status", "?")) for dec in decided.values())
            verdict = ", ".join("%d %s" % (n, s) for s, n in sorted(counts.items()))
        note = ""
        if isinstance(it.get("decision"), dict):
            note = str(it["decision"].get("note", "") or "")
        audit = it.get("audit")
        if isinstance(audit, list) and audit and all(isinstance(a, dict) and "verdict" in a for a in audit):
            note = (note + "; " if note else "") + "stability audit: " + ", ".join(
                "direction %d %s" % (j, a["verdict"]) for j, a in enumerate(audit))
        change = it.get("energy_change")
        t.iterations.append(IterationData(
            number=k, centre=centre, accepted=accepted, base=it.get("base"),
            energy=reader.stat(centre_keys, rule) if centre_keys else None, verdict=verdict,
            rule=None if comp is None else comp.get("rule"),
            blocks=(None if comp is None else comp.get("blocks")) or (max(blocks_decided) if blocks_decided else None),
            G=None if comp is None else _num(comp.get("G")),
            G_error=None if (comp is None or _num(comp.get("V")) is None) else float(np.sqrt(max(_num(comp.get("V")), 0.0))),
            N=None if comp is None else _num(comp.get("N")),
            joint_bound=None if comp is None else _num(comp.get("joint_bound")),
            bound_v1=None if comp is None else _num(comp.get("bound")),
            step_length=None if comp is None else _num(comp.get("step_norm")),
            step_error=None if comp is None else _step_error(it),
            components=None if comp is None else [float(v) for v in comp.get("components", [])],
            energy_change=(float(change[0]), float(change[1])) if change else None, note=note, scans=scans,
            estimates=estimates, classes=it.get("classes")))

    # ---- final geometry and its energy
    fin = state.get("final") or None
    start_keys = list(iters[str(numbers[0])].get("centre_keys") or [])
    final_keys: List[str] = []
    if fin:
        t.final = np.array(fin["centre"], dtype=float).reshape(-1, 3)
        t.final_iteration = int(fin.get("iteration", numbers[-1]))
        final_keys = list(fin.get("keys") or [])
        t.final_source = ("the runs at the starting geometry (the final geometry is the starting geometry)"
                          if fin.get("same_as_start") else "full-length runs at the final geometry")
        for key in final_keys:
            if key not in seen_keys:
                seen_keys[key] = False
                t.points.append(row(key, "final", t.final_iteration, None, None))
        if fin.get("energy_change"):
            t.driver_difference = (float(fin["energy_change"][0]), float(fin["energy_change"][1]))
    elif t.ended:
        kf = final_iteration(state)
        t.final = np.array(iters[str(kf)]["centre"], dtype=float).reshape(-1, 3)
        t.final_iteration = kf
        final_keys = list(iters[str(kf)].get("centre_keys") or [])
        t.final_source = "centre run of iteration %d" % kf
    t.start_energy = reader.stat(start_keys, rule) if start_keys else None
    if t.final is not None:
        first = np.array(iters[str(numbers[0])]["centre"], dtype=float).reshape(-1, 3)
        t.moved = bool(np.abs(t.final - first).max() > 1e-6) and set(final_keys) != set(start_keys)
        t.final_energy = reader.stat(final_keys, rule) if final_keys else None
        if t.start_energy is not None and t.final_energy is not None:
            if not t.moved:
                t.difference = (0.0, 0.0)            # the same geometry and the same runs: the difference is exactly 0
            else:
                t.difference = ((t.final_energy.value - t.start_energy.value) * HARTREE_TO_KCAL,
                                float(np.hypot(t.final_energy.error, t.start_energy.error)) * HARTREE_TO_KCAL)
        t.position_cov = _position_covariance(t.protocol, state, t.status) if t.moved else None
    # ---- totals: every point is counted once
    for key, is_search in seen_keys.items():
        n = reader.read(key).nblocks
        cost = reader.cost(key)
        t.blocks_total += n
        t.core_hours_total += cost
        if is_search:
            t.blocks_search += n
            t.core_hours_search += cost
    if state.get("protocol") == "classic" and start_keys:
        # the classic search uses the first blocks of the first run at the start as the centre point of
        # iteration 1; its driver counts these blocks as search blocks
        used = min(reader.read(start_keys[0]).nblocks, int(iters[str(numbers[0])].get("centre_blocks") or 0))
        t.blocks_search += used


def collect(project_dir: str, engine, title: Optional[str] = None, protocols: Optional[dict] = None) -> ReportData:
    """Read a project folder. ``engine`` gives the DMC blocks of the points (``engine.read(key)``).

    ``protocols`` may map a target name to the protocol object that its search used. It is needed only if
    the search changed the rule for the start-up blocks (``drop``, ``drop_min``, ``drop_max``,
    ``min_blocks_after_drop``); without it the default rule of the protocol named in the project state is used.

    Raises FileNotFoundError if the folder holds no project at all (neither ``work/project.json`` nor ``work/targets``).
    """
    project_dir = os.path.abspath(project_dir)
    work = os.path.join(project_dir, "work")
    pfile = os.path.join(work, "project.json")
    tfolder = os.path.join(work, "targets")
    if not os.path.exists(pfile) and not os.path.isdir(tfolder):
        raise FileNotFoundError("no project found in %s: work/project.json and work/targets are both missing" % project_dir)
    data = ReportData(project_dir=project_dir, title=title or os.path.basename(project_dir), project={})
    if os.path.exists(pfile):
        try:
            data.project = _read_json(pfile)
        except (ValueError, OSError) as exc:
            data.missing.append(("project state", "work/project.json could not be read: %s" % exc))
    else:
        data.missing.append(("project state", "work/project.json is missing; the targets were taken from work/targets"))
    if title is None and data.project.get("name"):
        data.title = str(data.project["name"])
    # ---- path
    path_file = os.path.join(work, "path.xyz")
    if os.path.exists(path_file):
        try:
            frames = G.read_xyz_frames(path_file)
            symbols = list(frames[0][0])
            if any(list(f[0]) != symbols for f in frames):
                raise ValueError("the images do not all have the same atoms in the same order")
            energies = [comment_energy(f[2]) for f in frames]
            known = all(e is not None for e in energies)
            data.path = PathData(symbols=symbols, frames=[f[1] for f in frames],
                                 energies=np.array(energies, dtype=float) if known else None)
            if not known:
                data.missing.append(("path energies", "%d of %d images of work/path.xyz have no energy in the comment line"
                                     % (sum(e is None for e in energies), len(frames))))
        except (ValueError, OSError) as exc:
            data.missing.append(("path", "work/path.xyz could not be read: %s" % exc))
    else:
        data.missing.append(("path", "this project has no path (work/path.xyz does not exist)"))
    # ---- subspace
    sub_file = os.path.join(work, "subspace.npz")
    if os.path.exists(sub_file):
        try:
            data.subspace = Subspace.load(sub_file)
        except Exception as exc:                       # numpy raises several types for a damaged file
            data.missing.append(("subspace", "work/subspace.npz could not be read: %s: %s" % (type(exc).__name__, exc)))
    else:
        data.missing.append(("subspace", "work/subspace.npz does not exist"))
    # ---- QMC settings
    set_file = os.path.join(work, "autotune", "qmc_settings.json")
    if os.path.exists(set_file):
        try:
            data.settings = _read_json(set_file)
        except (ValueError, OSError) as exc:
            data.missing.append(("QMC settings", "work/autotune/qmc_settings.json could not be read: %s" % exc))
    # ---- targets
    listed = data.project.get("targets") if isinstance(data.project.get("targets"), dict) else {}
    names = list(listed)
    if os.path.isdir(tfolder):
        names += [n for n in sorted(os.listdir(tfolder)) if n not in listed and os.path.isdir(os.path.join(tfolder, n))]
    reader = Reader(engine)
    tolerance = _input_tolerance(project_dir)
    for name in names:
        info = listed.get(name) if isinstance(listed.get(name), dict) else {}
        tdir = os.path.join(tfolder, name)
        if not os.path.isdir(tdir):
            data.missing.append(("target %s" % name, "the folder work/targets/%s does not exist" % name))
            continue
        data.targets[name] = _read_target(name, info, tdir, reader, (protocols or {}).get(name), tolerance, data.path)
    if not data.targets:
        data.missing.append(("targets", "the project has no target folder yet"))
    data.problems.extend(reader.problems)
    return data


# ============================================================================= derived numbers
def geometry_change(t: TargetData) -> Optional[dict]:
    """RMSD after rigid alignment and the largest displacement of one atom between start and final geometry."""
    if t.start is None or t.final is None:
        return None
    moved, rmsd, _ = G.kabsch(t.final, t.start)
    disp = np.linalg.norm(moved - t.start, axis=1)
    i = int(np.argmax(disp))
    out = {"rmsd": float(rmsd), "max_displacement": float(disp[i]),
           "max_displacement_atom": "%s%d" % (t.symbols[i], i + 1) if t.symbols else str(i + 1),
           "position_error_rms": None}
    if t.position_cov is not None and t.vectors is not None:
        V = t.vectors
        var = float(np.trace(np.dot(np.dot(V.T, t.position_cov), V)))
        out["position_error_rms"] = float(np.sqrt(max(var, 0.0) / len(t.start)))
    return out


def internal_table(t: TargetData) -> List[dict]:
    """Every bond and every angle of a target at the start and at the final geometry, with the change.

    The bonds are those of the start geometry or of the final geometry (``geometry.bond_list``). ``error`` is
    the statistical error (1 standard error) of the final value, propagated from the covariance of the final
    geometry; None where that covariance is not known. The start geometry is an input and has no error.
    """
    if t.start is None or t.final is None or t.symbols is None:
        return []
    bonds = sorted(set(G.bond_list(t.symbols, t.start)) | set(G.bond_list(t.symbols, t.final)))
    names = ["%s%d" % (s, i + 1) for i, s in enumerate(t.symbols)]
    items = [("bond", (i, j)) for i, j in bonds] + [("angle", a) for a in G.angle_list(bonds)]

    def value(kind, atoms, X):
        return G.distance(X, *atoms) if kind == "bond" else G.angle(X, *atoms)

    cov, V = t.position_cov, t.vectors
    rows = []
    for kind, atoms in items:
        a, b = value(kind, atoms, t.start), value(kind, atoms, t.final)
        err = None
        if cov is not None and V is not None and V.shape[1] == t.final.size:
            h = 1e-4
            J = np.array([(value(kind, atoms, t.final + h * v.reshape(-1, 3)) - value(kind, atoms, t.final - h * v.reshape(-1, 3)))
                          / (2.0 * h) for v in V])
            err = float(np.sqrt(max(float(np.dot(J, np.dot(cov, J))), 0.0)))
        rows.append({"kind": kind, "atoms": "-".join(names[i] for i in atoms), "start": float(a), "final": float(b),
                     "change": float(b - a), "error": err, "unit": "angstrom" if kind == "bond" else "degree"})
    return rows


def path_internal(path: "PathData") -> dict:
    """Bond lengths and angles of every image of the path.

    The bonds are the pairs of atoms that are bonded in at least one image (``geometry.bond_list``), so a
    bond that breaks or forms along the path is followed over the whole path. The angles are those between
    two such bonds at a common atom. Returns a dict: ``names`` (for example ``N1-H2``, ``H2-N1-H3``; the
    number is the position of the atom in the user's file, from 1), ``kinds`` (``bond`` or ``angle``),
    ``atoms`` (index lists, from 0) and ``values``, one row per image and one column per bond or angle
    (angstrom and degrees).
    """
    symbols, frames = path.symbols, path.frames
    pairs = set()
    for X in frames:
        pairs.update(G.bond_list(symbols, X))
    bonds = sorted(pairs)
    labels = ["%s%d" % (s, i + 1) for i, s in enumerate(symbols)]
    items = [("bond", tuple(b)) for b in bonds] + [("angle", tuple(a)) for a in G.angle_list(bonds)]
    values = np.zeros((len(frames), len(items)))
    for r, X in enumerate(frames):
        for c, (kind, atoms) in enumerate(items):
            values[r, c] = G.distance(X, *atoms) if kind == "bond" else G.angle(X, *atoms)
    return {"names": ["-".join(labels[i] for i in atoms) for _kind, atoms in items],
            "kinds": [kind for kind, _atoms in items], "atoms": [list(atoms) for _kind, atoms in items],
            "values": values}


EQUAL_TOLERANCE = {"bond": 1e-4, "angle": 1e-2}          # angstrom and degrees: values closer than this are equal
CHANGE_FLOOR = {"bond": 0.002, "angle": 0.2}             # a smaller change along the path is not worth a row or a line
COORDINATE_UNIT = {"bond": "angstrom", "angle": "degrees"}


def coordinate_groups(internal: dict, kind: str, limit: int = 4, merge: float = 0.0) -> Tuple[List[dict], int]:
    """The bonds (``kind = "bond"``) or angles (``"angle"``) that change most along the path.

    Columns whose values agree at every image (within 1e-4 angstrom or 0.01 degree) form one group: in a
    plot they would lie exactly on top of each other. With ``merge`` > 0, curves that are closer than
    ``merge`` times the largest range of this kind at every image are one group as well: a figure could not
    show them apart. The groups are sorted by the range of their values along the path, largest first, and
    the first ``limit`` are returned together with the number of all groups. Each group is a dict with
    ``members`` (column numbers), ``atoms`` (of the first member), ``label``, ``values``, ``range``,
    ``delta`` (values minus the value at image 0) and ``extreme`` (the change from image 0 with the largest
    size, with its sign) at image ``extreme_image``.
    """
    exact = EQUAL_TOLERANCE[kind]
    V = internal["values"]
    columns = [j for j, k in enumerate(internal["kinds"]) if k == kind]
    largest = max([float(V[:, j].max() - V[:, j].min()) for j in columns] or [0.0])
    tolerance = max(exact, float(merge) * largest)
    groups: List[dict] = []
    # the columns with the largest range come first, so that a group is named after the coordinate that changes
    # most; columns with the same range (within the tolerance of equal values) keep the order of the atoms
    for j in sorted(columns, key=lambda c: (-np.floor(float(V[:, c].max() - V[:, c].min()) / (2.0 * exact) + 0.5), c)):
        for g in groups:
            deviation = float(np.abs(V[:, j] - V[:, g["members"][0]]).max())
            if deviation < tolerance:
                g["members"].append(j)
                g["exact"] = g["exact"] and deviation < exact
                break
        else:
            groups.append({"members": [j], "exact": True})
    for g in groups:
        if g["exact"]:
            g["members"].sort()             # equal values: the group is named after its first coordinate
        first = g["members"][0]
        g["members"] = [first] + sorted(g["members"][1:])
        g["values"] = V[:, first].copy()
        g["range"] = float(g["values"].max() - g["values"].min())
        g["delta"] = g["values"] - g["values"][0]
        g["extreme_image"] = int(np.argmax(np.abs(g["delta"])))
        g["extreme"] = float(g["delta"][g["extreme_image"]])
        g["atoms"] = list(internal["atoms"][first])
        g["name"] = internal["names"][first]
        more = len(g["members"]) - 1
        if more == 0:
            g["label"] = g["name"]
        elif g["exact"]:
            g["label"] = "%s and %d equal %s%s" % (g["name"], more, kind, "s" if more > 1 else "")
        else:
            g["label"] = "%s and %d similar" % (g["name"], more)          # the same curve at the scale of a figure
    groups.sort(key=lambda g: (-g["range"], g["members"][0]))
    return groups[:limit], len(groups)


def path_changes(path: "PathData", rows: int = 8) -> Optional[dict]:
    """The bonds and angles that change most along the path, for the summary.

    For each kind: the ``rows`` groups of equal coordinates with the largest change from image 0, the number
    of all groups, and how many of them change by less than ``CHANGE_FLOOR``. None for a path of one image."""
    if path is None or len(path.frames) < 2:
        return None
    internal = path_internal(path)
    out = {"reference_image": 0, "last_image": len(path.frames) - 1}
    for kind in ("bond", "angle"):
        groups, total = coordinate_groups(internal, kind, limit=10 ** 6)
        groups.sort(key=lambda g: -abs(g["extreme"]))
        out[kind] = {
            "unit": COORDINATE_UNIT[kind], "count": int(sum(len(g["members"]) for g in groups)), "groups": total,
            "floor": CHANGE_FLOOR[kind],
            "below_floor": int(sum(len(g["members"]) for g in groups if abs(g["extreme"]) < CHANGE_FLOOR[kind])),
            "largest": [{"name": g["name"], "equal_coordinates": len(g["members"]),
                         "value_at_image_0": float(g["values"][0]), "largest_change": g["extreme"],
                         "image_of_largest_change": g["extreme_image"], "value_at_last_image": float(g["values"][-1])}
                        for g in groups[:rows]],
        }
    return out


def write_path_internal_csv(data: "ReportData", path: str) -> None:
    """One row per path image: every bond length (angstrom) and angle (degrees) of :func:`path_internal`,
    and the surrogate energy relative to image 0 (kcal/mol) if the path file holds the energies."""
    if data.path is None:
        raise Skip("the project has no path")
    internal = path_internal(data.path)
    if not internal["names"]:
        raise Skip("no pair of atoms of the path is close enough to be a bond")
    head = ["image"] + ["%s_%s_%s" % (k, n, "angstrom" if k == "bond" else "degree")
                        for k, n in zip(internal["kinds"], internal["names"])]
    energies = data.path.energies
    if energies is not None:
        head.append("surrogate_energy_minus_image_0_kcal_per_mol")
    lines = [",".join(head)]
    for r, row in enumerate(internal["values"]):
        cells = ["%d" % r] + ["%.5f" % v if k == "bond" else "%.3f" % v for k, v in zip(internal["kinds"], row)]
        if energies is not None:
            cells.append("%.4f" % ((energies[r] - energies[0]) * HARTREE_TO_KCAL))
        lines.append(",".join(cells))
    atomic_write(path, "\n".join(lines) + "\n")


def statement(t: TargetData) -> Tuple[Optional[bool], str]:
    """Whether the energy changed by more than ``SIGMA_LIMIT`` standard errors, and one sentence in plain words."""
    if t.state is None:
        return None, "no search state file: the search of this target has not started"
    if not t.iterations:
        return None, "the search state holds no iteration that this report can read"
    if t.final is None:
        return None, "the search has not ended (status %s): there is no final geometry yet" % t.status
    if t.moved is False:
        return False, ("the search did not move the geometry: the final geometry is the starting geometry, "
                       "so the energy difference is exactly 0")
    if t.difference is None:
        return None, "the DMC energies of the start and the final geometry are not both available yet"
    de, sd = t.difference
    text = "%s kcal/mol" % fmt_pm(abs(de), sd)
    preliminary = ""
    if not (t.start_energy.complete and t.final_energy.complete):
        preliminary = " (preliminary: the DMC runs at these two geometries have not all ended)"
    if abs(de) <= SIGMA_LIMIT * sd:
        return False, ("no significant change: the optimized geometry is %s %s than the starting geometry, "
                       "which is not more than %.0f standard errors%s" % (text, "lower" if de < 0 else "higher",
                                                                          SIGMA_LIMIT, preliminary))
    if t.kind == "saddle":
        return True, ("the DMC energy of the optimized saddle geometry is %s %s than that of the starting geometry%s"
                      % (text, "lower" if de < 0 else "higher", preliminary))
    if t.kind == "image" and de < 0:
        return True, ("the path image moved across the path: the optimized geometry is %s lower than the starting "
                      "image%s" % (text, preliminary))
    if de < 0:
        return True, ("the starting geometry is not the DMC minimum: the optimized geometry is %s lower%s"
                      % (text, preliminary))
    return True, ("warning: the optimized geometry is %s HIGHER than the starting geometry; the search did not "
                  "lower the energy%s" % (text, preliminary))


def barriers(data: ReportData) -> List[dict]:
    """One entry for every pair of a saddle target and a minimum target.

    ``start``: barrier between the two starting geometries (fixed-geometry DMC). ``optimized``: barrier
    between the two optimized geometries. ``change``: optimized minus start. ``surrogate_path``: difference
    of the surrogate energies of the path images nearest to the two targets (exact, no statistical error).
    """
    out = []
    saddles = [t for t in data.targets.values() if t.kind == "saddle"]
    minima = [t for t in data.targets.values() if t.kind == "minimum"]
    for s in saddles:
        for m in minima:
            entry = {"saddle": s.name, "minimum": m.name, "start": None, "optimized": None, "change": None,
                     "surrogate_path": None, "notes": []}
            if s.start_energy and m.start_energy:
                entry["start"] = {"value": (s.start_energy.value - m.start_energy.value) * HARTREE_TO_KCAL,
                                  "error": float(np.hypot(s.start_energy.error, m.start_energy.error)) * HARTREE_TO_KCAL}
            else:
                entry["notes"].append("the DMC energies of both starting geometries are not available yet")
            if s.final_energy and m.final_energy and s.final is not None and m.final is not None:
                entry["optimized"] = {"value": (s.final_energy.value - m.final_energy.value) * HARTREE_TO_KCAL,
                                      "error": float(np.hypot(s.final_energy.error, m.final_energy.error)) * HARTREE_TO_KCAL,
                                      "complete": bool(s.final_energy.complete and m.final_energy.complete
                                                       and s.ended and m.ended)}
                if entry["start"]:
                    # a target that did not move uses the same runs in both barriers: its noise cancels in the change
                    var = sum((x.start_energy.error ** 2 + x.final_energy.error ** 2) for x in (s, m) if x.moved)
                    entry["change"] = {"value": entry["optimized"]["value"] - entry["start"]["value"],
                                       "error": float(np.sqrt(var)) * HARTREE_TO_KCAL}
            else:
                entry["notes"].append("the DMC energies of both optimized geometries are not available yet")
            if data.path is None:
                entry["notes"].append("no path: no surrogate barrier")
            elif data.path.energies is None:
                entry["notes"].append("the path file holds no energies: no surrogate barrier")
            elif s.image is None or m.image is None:
                entry["notes"].append("the targets could not be compared with the path images")
            elif max(s.image_rmsd, m.image_rmsd) > IMAGE_MATCH_LIMIT:
                entry["notes"].append("a target is more than %.2f A (RMSD) from every path image: no surrogate barrier"
                                      % IMAGE_MATCH_LIMIT)
            else:
                e = data.path.energies
                entry["surrogate_path"] = {"value": float(e[s.image] - e[m.image]) * HARTREE_TO_KCAL, "exact": True,
                                           "saddle_image": s.image, "minimum_image": m.image,
                                           "saddle_rmsd_to_image": s.image_rmsd, "minimum_rmsd_to_image": m.image_rmsd}
            out.append(entry)
    return out


def history_rows(t: TargetData) -> List[dict]:
    rows = []
    e0 = t.start_energy
    for it in t.iterations:
        rel = None
        if it.energy is not None and e0 is not None:
            rel = {"value": (it.energy.value - e0.value) * HARTREE_TO_KCAL, "error": it.energy.error * HARTREE_TO_KCAL}
        rows.append({"iteration": it.number, "blocks": it.blocks, "verdict": it.verdict, "G": it.G, "G_error": it.G_error,
                     "N": it.N, "joint_bound": it.joint_bound, "bound_v1": it.bound_v1, "step_length": it.step_length,
                     "step_length_error": it.step_error, "rule": it.rule, "accepted": it.accepted, "built_from": it.base,
                     "centre_energy": None if it.energy is None else it.energy.as_dict(),
                     "centre_energy_minus_start_kcal": rel, "note": it.note})
    return rows


def summarize(data: ReportData) -> dict:
    """Everything that ``summary.json`` holds, except the lists of written and skipped files."""
    targets = {}
    for name, t in data.targets.items():
        significant, text = statement(t)
        last = t.last_scan_iteration()
        directions = []
        if last is not None:
            for j in sorted(last.estimates):
                e = last.estimates[j]
                directions.append({"direction": j, "slope": e.get("g"), "slope_error": e.get("sg"),
                                   "force_constant": e.get("k"), "force_constant_error": e.get("sk"),
                                   "force_constant_source": e.get("k_source"), "step": e.get("step"),
                                   "stationary_point": e.get("m"), "stationary_point_error": e.get("sm"),
                                   "by_symmetry": e.get("by_symmetry")})
        targets[name] = {
            "kind": t.kind, "protocol": t.protocol, "status": t.status, "message": t.message,
            "iterations": len(t.iterations), "rejected_steps": t.rejected, "search_ended": t.ended,
            "blocks": {"search": t.blocks_search, "total": t.blocks_total},
            "core_hours": {"search": t.core_hours_search, "total": t.core_hours_total},
            "start_energy_hartree": None if t.start_energy is None else t.start_energy.as_dict(),
            "final_energy_hartree": None if t.final_energy is None else t.final_energy.as_dict(),
            "final_energy_source": t.final_source,
            "energy_change_kcal": None if t.difference is None else {"value": t.difference[0], "error": t.difference[1]},
            "energy_change_kcal_stored_by_driver": None if t.driver_difference is None else
            {"value": t.driver_difference[0], "error": t.driver_difference[1]},
            "significant": significant, "sigma_limit": SIGMA_LIMIT, "statement": text,
            "geometry_change_angstrom": geometry_change(t), "moved": t.moved,
            "path_image": t.image, "rmsd_to_path_image_angstrom": t.image_rmsd,
            "tolerance_kcal": t.eps_total, "history": history_rows(t),
            "last_scan": None if last is None else {"iteration": last.number, "directions": directions},
            "problems": list(t.problems),
        }
    sub = None
    if data.subspace is not None:
        evr = data.subspace.explained_variance_ratio
        sub = {"kind": data.subspace.kind, "dim": data.subspace.dim,
               "explained_variance_ratio": None if evr is None else [float(v) for v in evr if v > NUMERICAL_ZERO],
               "captured_fraction": (data.project.get("subspace") or {}).get("captured")
               if isinstance(data.project.get("subspace"), dict) else None}
    path = None
    if data.path is not None:
        e = data.path.energies
        path = {"images": len(data.path.frames), "energies_known": e is not None,
                "surrogate_energy_minus_first_image_kcal": None if e is None else [float(v) for v in (e - e[0]) * HARTREE_TO_KCAL],
                "geometry_changes": path_changes(data.path)}
    return {
        "title": data.title, "written": time.strftime("%Y-%m-%d %H:%M:%S"),
        "units": {"energy_of_one_geometry": "hartree", "energy_difference": "kcal/mol", "length": "angstrom",
                  "angle": "degree", "error_bars": "1 standard error"},
        "project": {"name": data.project.get("name"), "status": data.project.get("status"),
                    "stages": data.project.get("stages")},
        "targets": targets, "barriers": barriers(data), "path": path, "subspace": sub, "qmc_settings": data.settings,
        "not_available": [{"item": a, "reason": b} for a, b in data.missing], "problems": list(data.problems),
    }


# ============================================================================= writers
def write_energies_csv(data: ReportData, path: str) -> None:
    """One row per point that has DMC blocks."""
    rows = [r for t in data.targets.values() for r in t.points if r.blocks > 0]
    if not rows:
        raise Skip("no point of any target has DMC blocks yet")
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["key", "role", "state", "blocks", "energy_hartree", "error_hartree", "surrogate_energy_hartree",
                    "target", "iteration", "direction", "displacement_angstrom", "blocks_discarded", "core_hours"])
        for r in rows:
            w.writerow([r.key, r.role, r.state, r.blocks,
                        "" if r.energy is None else "%.10f" % r.energy, "" if r.error is None else "%.10f" % r.error,
                        "" if r.e_surrogate is None else "%.10f" % r.e_surrogate, r.target,
                        "" if r.iteration is None else r.iteration, "" if r.direction is None else r.direction,
                        "" if r.t is None else "%.3f" % r.t, "" if r.discarded is None else r.discarded,
                        "%.4f" % r.core_hours])
    os.replace(tmp, path)


def _energy_comment(stat: Optional[Stat]) -> str:
    return "" if stat is None else "; DMC energy %s hartree" % fmt_pm(stat.value, stat.error)


def write_geometries(data: ReportData, folder: str) -> Tuple[List[str], List[Tuple[str, str]]]:
    """``<target>_start.xyz``, ``<target>_final.xyz`` and ``<target>_trajectory.xyz`` for every target.
    Returns the written files and the list of (file, reason) that were not written."""
    os.makedirs(folder, exist_ok=True)
    written, skipped = [], []
    for name, t in data.targets.items():
        files = {kind: os.path.join(folder, "%s_%s.xyz" % (name, kind)) for kind in ("start", "final", "trajectory")}
        if t.start is None or t.symbols is None:
            for f in files.values():
                skipped.append((f, "target %s has no readable start geometry" % name))
            continue
        try:
            tmp = files["start"] + ".tmp"
            G.write_xyz(tmp, t.symbols, t.start, "target %s: starting geometry%s" % (name, _energy_comment(t.start_energy)))
            os.replace(tmp, files["start"]); written.append(files["start"])
        except (OSError, ValueError) as exc:
            skipped.append((files["start"], "%s: %s" % (type(exc).__name__, exc)))
        if t.final is not None:
            try:
                tmp = files["final"] + ".tmp"
                G.write_xyz(tmp, t.symbols, t.final, "target %s: final geometry, status %s%s"
                            % (name, t.status, _energy_comment(t.final_energy)))
                os.replace(tmp, files["final"]); written.append(files["final"])
            except (OSError, ValueError) as exc:
                skipped.append((files["final"], "%s: %s" % (type(exc).__name__, exc)))
        else:
            skipped.append((files["final"], "the search of target %s has not ended (status %s); the latest centre is "
                            "the last frame of %s" % (name, t.status, os.path.basename(files["trajectory"]))))
        if t.iterations:
            comments = []
            for it in t.iterations:
                comments.append("target %s: centre of iteration %d%s%s" % (
                    name, it.number, " (the step into this centre was rejected)" if it.accepted is False else "",
                    _energy_comment(it.energy)))
            try:
                tmp = files["trajectory"] + ".tmp"
                G.write_xyz_frames(tmp, t.symbols, [it.centre for it in t.iterations], comments)
                if len(G.read_xyz_frames(tmp)) != len(t.iterations):
                    raise ValueError("%s: the number of frames changed on write" % tmp)
                os.replace(tmp, files["trajectory"]); written.append(files["trajectory"])
            except (OSError, ValueError) as exc:
                skipped.append((files["trajectory"], "%s: %s" % (type(exc).__name__, exc)))
        else:
            skipped.append((files["trajectory"], "target %s has no search iterations yet" % name))
    return written, skipped


def write_bond_table(data: ReportData, path: str) -> None:
    rows = [(name, r) for name, t in data.targets.items() for r in internal_table(t)]
    if not rows:
        raise Skip("no target has both a start and a final geometry")
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["target", "kind", "atoms", "start", "final", "change", "statistical_error_of_final", "unit"])
        for name, r in rows:
            w.writerow([name, r["kind"], r["atoms"], "%.5f" % r["start"], "%.5f" % r["final"], "%.5f" % r["change"],
                        "" if r["error"] is None else "%.5f" % r["error"], r["unit"]])
    os.replace(tmp, path)


def write_summary_json(summary: dict, path: str) -> None:
    atomic_write(path, json.dumps(jsonable(summary), indent=1, allow_nan=False) + "\n")


def _settings_lines(settings: dict) -> List[str]:
    """The section "QMC settings" of the summary: the values, and how the pilot runs chose them."""
    L = ["## QMC settings", ""]
    for k, v in settings.items():
        if k != "autotune":
            L.append("- %s: %s" % (k, json.dumps(v) if isinstance(v, (dict, list)) else v))
    L.append("")
    record = settings.get("autotune")
    if not isinstance(record, dict):
        return L
    L += ["How the values were chosen (mode %s):" % record.get("mode", "not recorded"), ""]
    rows = []
    for name, choice in (record.get("choices") or {}).items():
        if not isinstance(choice, dict):
            continue
        value = choice.get("value")
        rows.append([name, json.dumps(value) if isinstance(value, (dict, list)) else str(value),
                     str(choice.get("source", "")), str(choice.get("basis", ""))])
    if rows:
        L += _md_table(["setting", "value", "source", "basis"], rows)
    cost = (record.get("measurements") or {}).get("pilot_cost_core_hours")
    if isinstance(cost, (int, float)):
        L += ["Cost of the pilot runs: %.1f core-hours." % cost, ""]
    for label, key in (("First guesses, not measured", "guesses"), ("Rules that could not be checked", "not_checked"),
                       ("Warnings of the pilot", "warnings")):
        items = [str(x) for x in (record.get(key) or [])]
        if items:
            L += ["%s:" % label, ""] + ["- %s" % x for x in items] + [""]
    L += ["The full record of the pilot runs is in work/autotune/qmc_settings.json.", ""]
    return L


def _md_table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> List[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(" --- " for _ in header) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c).replace("|", "/") for c in r) + " |")
    return out + [""]


def _pm_dict(d: Optional[dict], exact: bool = False) -> str:
    if not d:
        return "n/a"
    if exact:
        return "%.3f (exact)" % d["value"]
    return fmt_pm(d["value"], d["error"])


def write_summary_md(summary: dict, path: str) -> None:
    """The summary for a person: the same content as ``summary.json`` in tables and plain sentences."""
    s = jsonable(summary)
    L: List[str] = ["# %s: results" % s["title"], "",
                    "Written %s. Every error bar is 1 standard error. The energy of one geometry is in hartree. "
                    "Energy differences are in kcal/mol, lengths in angstrom, angles in degrees." % s["written"], ""]
    stages = (s.get("project") or {}).get("stages")
    if isinstance(stages, dict) and stages:
        L += ["## State of the project", ""]
        L += _md_table(["stage", "status", "message"],
                       [[k, (v or {}).get("status", ""), (v or {}).get("message", "")] for k, v in stages.items()])
    T = s["targets"]
    L += ["## Targets", ""]
    if not T:
        L += ["The project has no target yet.", ""]
    else:
        L += _md_table(["target", "kind", "protocol", "status", "iterations", "rejected steps",
                        "DMC blocks: search", "DMC blocks: total", "core-hours: total"],
                       [[n, t["kind"], t["protocol"], t["status"], t["iterations"], t["rejected_steps"],
                         t["blocks"]["search"], t["blocks"]["total"],
                         "%.1f" % t["core_hours"]["total"] if t["core_hours"]["total"] > 0 else "not known"]
                        for n, t in T.items()])
        L += ["\"DMC blocks: search\" counts the scan points and the centre runs. \"total\" adds the full-length runs "
              "at the starting geometry and at the final geometry.", ""]
        for n, t in T.items():
            if t["message"]:
                L.append("- Target %s: %s" % (n, t["message"]))
        L += ["", "### DMC energies", ""]
        L += _md_table(["target", "starting geometry (hartree)", "final geometry (hartree)", "final minus start (kcal/mol)"],
                       [[n, _pm_dict(t["start_energy_hartree"]), _pm_dict(t["final_energy_hartree"]),
                         _pm_dict(t["energy_change_kcal"])] for n, t in T.items()])
        for n, t in T.items():
            L.append("- Target %s: %s." % (n, t["statement"]))
        L += ["", "An energy change counts as significant if it is larger than %.0f standard errors." % SIGMA_LIMIT, ""]
        rows = []
        for n, t in T.items():
            g = t["geometry_change_angstrom"]
            if g:
                rows.append([n, "%.4f" % g["rmsd"], "%.4f" % g["max_displacement"], g["max_displacement_atom"],
                             fmt_num(g["position_error_rms"])])
        if rows:
            L += ["### Geometry change from the starting geometry to the final geometry", ""]
            L += _md_table(["target", "RMSD after rigid alignment (A)", "largest displacement of one atom (A)", "atom",
                            "statistical error of an atom position (A)"], rows)
            L += ["The RMSD and the largest displacement are exact numbers for the two geometries. The last column is "
                  "the statistical error (1 standard error, root mean square over the atoms) of the final geometry. It "
                  "comes from the slope errors of the last scan; the curvatures of the step model are taken as exact. "
                  "\"n/a\" means that the search state does not allow this estimate. The file geometries/bond_table.csv "
                  "lists every bond and angle.", ""]
    B = s["barriers"]
    if B:
        L += ["## Barriers", ""]
        L += _md_table(["saddle", "minimum", "DMC at the starting geometries", "DMC at the optimized geometries",
                        "change by the optimization", "surrogate, from the path energies"],
                       [[b["saddle"], b["minimum"], _pm_dict(b["start"]), _pm_dict(b["optimized"]), _pm_dict(b["change"]),
                         _pm_dict(b["surrogate_path"], exact=True)] for b in B])
        L += ["All barriers are in kcal/mol: energy of the saddle target minus energy of the minimum target. "
              "\"DMC at the starting geometries\" is the fixed-geometry DMC barrier. The surrogate barrier is the "
              "difference of the path energies of the images nearest to the two targets; it has no statistical error.", ""]
        for b in B:
            for note in b["notes"]:
                L.append("- %s -> %s: %s." % (b["minimum"], b["saddle"], note))
            if b["optimized"] and not b["optimized"].get("complete", True):
                L.append("- %s -> %s: the barrier at the optimized geometries is preliminary (a search or a "
                         "full-length run has not ended)." % (b["minimum"], b["saddle"]))
        L.append("")
    for n, t in T.items():
        if not t["history"]:
            continue
        L += ["## Search history of target %s" % n, ""]
        v1 = any(h["bound_v1"] is not None for h in t["history"])
        head = ["iteration", "blocks per point", "verdict", "predicted gain G", "noise N", "joint bound",
                "step length (A)", "rule", "centre energy minus start"]
        if v1:
            head.insert(6, "remaining-energy bound")
        rows = []
        for h in t["history"]:
            r = [h["iteration"], fmt_num(h["blocks"], "%d"), h["verdict"], fmt_pm(h["G"], h["G_error"]),
                 fmt_num(h["N"], "%.4f"), fmt_num(h["joint_bound"], "%.4f"),
                 fmt_pm(h["step_length"], h["step_length_error"]), h["rule"] or "n/a",
                 _pm_dict(h["centre_energy_minus_start_kcal"])]
            if v1:
                r.insert(6, fmt_num(h["bound_v1"], "%.4f"))
            rows.append(r)
        L += _md_table(head, rows)
        L += ["G, N and the bounds are in kcal/mol. \"n/a\" marks a quantity that the protocol of this search does not "
              "have, or an iteration that is not decided. The error bar of the step length is its statistical error "
              "along the step, from the slope errors of the scan. The error bar of \"centre energy minus start\" is "
              "that of the centre energy alone.", ""]
        if t["last_scan"] and t["last_scan"]["directions"]:
            L += ["Estimates of the last scan (iteration %d):" % t["last_scan"]["iteration"], ""]
            L += _md_table(["direction", "slope (kcal/mol/A)", "force constant (kcal/mol/A^2)", "source of the force constant",
                            "step taken (A)", "note"],
                           [[d["direction"], fmt_pm(d["slope"], d["slope_error"]),
                             fmt_pm(d["force_constant"], d["force_constant_error"]), d["force_constant_source"] or "n/a",
                             fmt_num(d["step"], "%+.5f"),
                             "slope is zero by symmetry (%s)" % d["by_symmetry"] if d["by_symmetry"] else ""]
                            for d in t["last_scan"]["directions"]])
        for p in t["problems"]:
            L.append("- Problem: %s" % p)
        if t["problems"]:
            L.append("")
    changes = (s.get("path") or {}).get("geometry_changes")
    if changes:
        L += ["## Geometry along the path", "",
              "Bond lengths and angles of the %d path images. The change is counted from image 0. A bond that breaks "
              "or forms along the path is followed over the whole path." % s["path"]["images"], ""]
        for kind, digits in (("bond", 4), ("angle", 2)):
            c = changes.get(kind)
            if not c or not c["largest"]:
                continue
            unit = "A" if kind == "bond" else "degrees"
            L += ["%s%s: %d, in %d group(s) of equal values. The largest changes:" % (kind.capitalize(), "s", c["count"], c["groups"]), ""]
            L += _md_table([kind, "equal %ss" % kind, "at image 0 (%s)" % unit, "largest change (%s)" % unit, "at image",
                            "at image %d (%s)" % (changes["last_image"], unit)],
                           [[r["name"], r["equal_coordinates"], "%.*f" % (digits, r["value_at_image_0"]),
                             "%+.*f" % (digits, r["largest_change"]), r["image_of_largest_change"],
                             "%.*f" % (digits, r["value_at_last_image"])] for r in c["largest"]])
            if c["below_floor"]:
                L.append("%d of the %d %ss change by less than %g %s along the path." % (
                    c["below_floor"], c["count"], kind, c["floor"], unit))
            L.append("")
        L += ["All values: `geometries/path_internal_coordinates.csv`. Figures: `plots/path_geometry` (the ones that "
              "change most) and `plots/path_geometry_map` (all of them, when there are more than a few).", ""]
    if s.get("subspace"):
        sub = s["subspace"]
        L += ["## Search subspace", "", "Kind: %s. Number of search coordinates: %d." % (sub["kind"], sub["dim"]), ""]
        evr = sub.get("explained_variance_ratio")
        if evr:
            cum = np.cumsum(evr)
            L += _md_table(["principal component", "explained variance ratio", "cumulative", "kept"],
                           [[i + 1, "%.6f" % v, "%.6f" % cum[i], "yes" if i < sub["dim"] else "no"]
                            for i, v in enumerate(evr[:12])])
            L += ["These ratios are exact numbers of the path images (no statistical error).", ""]
        if sub.get("captured_fraction"):
            L.append("Fraction of the displacement between the targets that lies in the subspace: " +
                     ", ".join("%s %.4f" % (k, v) for k, v in sub["captured_fraction"].items()) + ".")
            L.append("")
    if s.get("qmc_settings"):
        L += _settings_lines(s["qmc_settings"])
    if s.get("files"):
        L += ["## Files of this report", ""]
        L += ["- %s" % f for f in s["files"]]
        L.append("")
    notes = list(s.get("not_available") or []) + list(s.get("skipped") or [])
    L += ["## What was not made, and why", ""]
    if notes:
        L += ["- %s: %s" % (x["item"], x["reason"]) for x in notes]
    else:
        L.append("Nothing was skipped.")
    L.append("")
    if s.get("warnings") or s.get("problems"):
        L += ["## Warnings", ""]
        L += ["- %s" % w for w in list(s.get("problems") or []) + list(s.get("warnings") or [])]
        L.append("")
    atomic_write(path, "\n".join(L))
