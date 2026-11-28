"""The stage ``trial``: DMC energies with a multi-determinant trial wave function, after the geometry search.

The geometry search of every target uses one Slater determinant. When the input file holds
``[trial] kind = multidet``, the workflow has one more stage between ``search`` and ``report``. This
module runs that stage. For every target it takes two geometries, the start geometry and the final
geometry of the search. For every geometry and every cutoff of the input it asks the engine for
``series`` independent DMC runs with the trial wave function of that cutoff.

The engine does the work of a point (module :mod:`montepass.engine.multidet`): the Quantum Package 2
(QP2) jobs of the geometry, the quality checks and the QMC job. This module decides which points
exist, asks for them on every pass, and keeps a record in ``<state folder>/state.json``.

Keys of the points (a key is also the folder of the point below the root of the engine)
---------------------------------------------------------------------------------------
    trial/<target>/<start|final>/<label>              series 1
    trial/<target>/<start|final>/<label>_s<n>         series n = 2, 3, ...
    trial/<target>/<start|final>/qp2_n<n_det_max>     the QP2 calculations of that geometry, shared by all its
                                                      cutoffs (``qp2`` when every cutoff of the project is hf)

``label`` is ``n<n_det_max>_<name of the cutoff>`` for a numeric cutoff, for example ``n10000_c1e-03``, and
``hf`` for the single RHF determinant of QP2.

Rule for a geometry that the search did not move: if the final geometry of a target equals its start
geometry within ``SAME_GEOMETRY`` angstrom, only the start geometry is computed, and the record says so.

Order of the requests: for every geometry the numeric cutoffs are asked for before ``hf``. The first request
of a geometry decides what its first QP2 job holds: a numeric cutoff needs RHF and CIPSI, ``hf`` needs RHF
only. So one job holds both. The record and the tables keep the order of the input.

When the plan changes (a cutoff, ``n_det_max`` or ``series`` changed in the input), every point of the old plan
that is not in the new plan and has not ended is stopped (``engine.stop``), and the log names it. The stop comes
after the first pass of the new plan has asked for its points, so that the engine keeps every QP2 job that a
point of the new plan needs. The blocks of a stopped point stay on disk. A stopped point is not continued later:
delete its folder to compute it again. The record keeps the keys of the points that left the plan
(``left_plan``), so that the report can count their core-hours.

The automatic mode (``[trial] cutoff = auto``; then ``trial["auto"]`` is ``{"ladder": [...]}``)
-----------------------------------------------------------------------------------------------
In the manual mode the plan of the stage is the input: the cutoffs and the number of series that the user
wrote. In the automatic mode the plan grows. It starts with the first two cutoffs of the ladder and ``hf``,
with ``series`` DMC runs for each row. When every run of the plan is DONE, the stage applies the rule of
module :mod:`montepass.trial_rule` to the energies of :func:`results` and makes one decision:

    a row is not complete although its runs are DONE    the stage ends with NO_VERDICT
    CONVERGED or NO_DIFFERENCE                          the stage ends with that verdict
    a cutoff is above the single determinant by at      one more series for every row, while ``series_max``
      most 3 standard errors, at no geometry by more,     allows it (the result is tested again with more data)
      and fewer than two usable cutoffs are smaller;
      only when two cutoffs can then be compared
      (``trial_rule.above_to_test_again``)
    NOT_CONVERGED, or fewer than two usable cutoffs     the next cutoff of the ladder is added; without
                                                          one the stage ends with NOT_CONVERGED
    NOT_DECIDED                                         one more series for every row. When ``series_max``
                                                          allows no further series: the stage ends with
                                                          NOT_DECIDED if the errors are too large for the
                                                          verdict CONVERGED also with a smaller cutoff
                                                          (``trial_rule.errors_allow_convergence``); else the
                                                          next cutoff of the ladder is added to the plan;
                                                          without one the stage ends with NOT_DECIDED

The plan keeps one number of series: every cutoff of the plan, ``hf`` included, has the same number of series at every
geometry. After the final decision the status of the stage is DONE for every verdict, because a verdict is a
result and not a failure. The message of the stage then ends with the verdict.

Everything that the stage adds while it runs is in the record entry ``auto``. The entry ``rule.trial`` stays
the input, so that the workflow can see a changed input.

    ladder, tolerance, series_max    the values of the input that the decisions use
    cutoffs, series                  the plan now: its numeric cutoffs, and the series of every row
    final, verdict, reason           True after the last decision, with its verdict word and one sentence
    decisions                        one entry per decision: time, series, cutoffs, ladder, pair, verdict, action,
                                     and for each tested difference its name, S, e, bound and verdict; also what
                                     was measured against ``hf``

A decision and the plan that follows from it are written to the record in one step. So a new driver process
continues exactly: the same plan, and no decision is made twice. When the input changed but ``n_det_max`` and
the QP2 files are the same, and the cutoffs that the plan reached are still the first cutoffs of the ladder (the
ladder is the same, or it changed only after them, for example it got one more cutoff), the plan that was
reached is kept, no run leaves the plan, and the decision is made again with the new values. When a cutoff of
the plan that was reached is not at the same place in the new ladder (another cutoff stands there, or the ladder
is now shorter than the plan), when ``n_det_max`` changed, or when the record was written by the manual mode,
the plan starts from the input again. A point of that plan that exists is never computed again. A point of the
old plan that the plan of the input does not hold, and that has not ended, is stopped as in the manual mode (see
above). A later decision can add such a point to the plan again. The point is then STOPPED, and the stage needs
attention: a stopped point is not continued, and its folder must be deleted to compute it again. So these three
changes of the input are safe only while no run of the stage is running or waiting.
"""
from __future__ import annotations

import copy
import json
import os
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import trial_rule as R
from .engine import base as B
from .engine import multidet as MD
from .engine.qmcpack import max_series
from .project import DONE, NEEDS_ATTENTION, RUNNING
from .search import protocol_v2 as P2
from .search.dense import combine_series
from .stats import plain, reblock

SAME_GEOMETRY = 1e-6                    # angstrom: a final geometry closer than this to the start geometry is the start geometry
GEOMETRIES = ("start", "final")
PREFIX = "trial"                        # first part of every key of the stage
REFUSAL_SECONDS = 3600.0                # a request that the batch system refuses longer than this is a problem
STATE_NAME = "state.json"
RULE = ("After the geometry search of every target has ended, DMC energies with a multi-determinant trial wave function "
        "are computed at the start geometry and at the final geometry of every target. A final geometry that equals the "
        "start geometry within %g angstrom is not computed again." % SAME_GEOMETRY)


# ============================================================================= names
def entry_label(cutoff, n_det_max) -> str:
    """Name of a cutoff in keys and tables: ``hf``, or ``n<n_det_max>_<name of the cutoff>`` (``n10000_c1e-03``)."""
    if cutoff == MD.HF_LABEL:
        return MD.HF_LABEL
    return "n%d_%s" % (int(n_det_max), MD.cut_label(cutoff))


def qp2_key(target: str, geometry: str, trial: dict) -> str:
    """Key of the folder of the QP2 calculations of one geometry. All cutoffs of the geometry share it."""
    if all(c == MD.HF_LABEL for c in trial["cutoffs"]):
        return "trial/%s/%s/qp2" % (target, geometry)
    return "trial/%s/%s/qp2_n%d" % (target, geometry, int(trial["n_det_max"]))


def point_keys(target: str, geometry: str, label: str, series: int) -> List[str]:
    """Keys of the DMC runs of one geometry and one cutoff: series 1 first."""
    first = "trial/%s/%s/%s" % (target, geometry, label)
    return [first] + ["%s_s%d" % (first, n) for n in range(2, int(series) + 1)]


def is_stage_key(key: str) -> bool:
    """True if ``key`` names a DMC run of this stage: ``trial/<target>/<start|final>/<label>``. The keys of the
    search (``<target>/it<k>/<leaf>``) and of the pilot runs (``<name>/r<k>/tau<t>``) have 3 parts, so a target
    named ``trial`` cannot give such a key."""
    parts = str(key).split("/")
    return len(parts) == 4 and parts[0] == PREFIX and parts[2] in GEOMETRIES


def record_keys(state: Optional[dict]) -> List[str]:
    """Keys of all DMC runs that a record of the stage names, in the order of the record."""
    return [key for rec in ((state or {}).get("targets") or {}).values() for part in (rec.get("geometries") or {}).values()
            for cut in (part.get("cutoffs") or {}).values() for key in cut.get("keys") or []]


def stop_keys(engine: B.Engine, keys: Sequence[str], seen: Optional[dict] = None,
              log: Optional[Callable[[str], None]] = None, why: str = "") -> List[str]:
    """Stop the points ``keys``. A point that the last pass saw as DONE, FAILED or STOPPED is left alone
    (``engine.stop`` would do nothing for it). A point that no pass has seen, or that the last pass saw as ABSENT,
    is stopped too: its request can have reached the engine after the record was written. (The automatic mode
    names the runs of a larger plan in its record one pass before it asks for them, and a pass can end between
    its requests and its record.) The engine does nothing for a key that it does not hold. Only a point that the
    last pass saw running or waiting gets a line in the log. Returns the keys for which ``engine.stop`` was
    called."""
    seen = seen or {}
    out: List[str] = []
    for key in keys:
        last = (seen.get(key) or {}).get("state")
        if last in B.FINAL_STATES:
            continue
        engine.stop(key)
        out.append(key)
        if log is not None and last not in (None, B.ABSENT):
            log("point %s: stopped%s. Its blocks stay on disk; a stopped point is not continued later"
                % (key, (", because " + why) if why else ""))
    return out


def stop_points(engine: B.Engine, state: Optional[dict], keep: Sequence[str] = (),
                log: Optional[Callable[[str], None]] = None, why: str = "") -> List[str]:
    """Stop every point of a record that has not ended, except the keys in ``keep`` (see :func:`stop_keys`).
    Returns the keys for which ``engine.stop`` was called."""
    keep = set(keep)
    keys = [k for k in record_keys(state) + list((state or {}).get("to_stop") or []) if k not in keep]
    return stop_keys(engine, list(dict.fromkeys(keys)), (state or {}).get("points"), log, why)


# ============================================================================= the record on disk
def read_state(state_dir: str) -> Optional[dict]:
    """The record of the stage (``state.json`` in ``state_dir``), or None if the stage never wrote one."""
    path = os.path.join(state_dir, STATE_NAME)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as fh:
            state = json.load(fh)
    except (OSError, ValueError) as exc:
        raise ValueError("%s could not be read (%s). Delete the file: the next pass writes it again from the input "
                         "and from the points on disk" % (path, exc))
    if not isinstance(state, dict) or not isinstance(state.get("targets"), dict):
        raise ValueError("%s is not a record of the stage trial. Delete the file: the next pass writes it again" % path)
    return state


def stored_trial(state_dir: str) -> Optional[dict]:
    """The values of ``[trial]`` that the record of the stage was written with, or None if there is no record."""
    state = read_state(state_dir)
    return None if state is None else (state.get("rule") or {}).get("trial")


# ============================================================================= energies
def run_energy(data: B.PointData) -> Optional[Tuple[float, float, int]]:
    """Mean energy (hartree), reblocked standard error and number of used blocks of one DMC run. None if the run
    holds too few blocks.

    The rule is the one of the full-length runs of the dense-scan search (protocol v2, method ``_stat`` of
    ``search/dense_v2.py``). The number of start-up blocks that are not used comes from the MSER-5 rule, inside
    ``drop_min`` .. ``drop_max`` of the protocol. A run that has ended with few blocks gets a smaller number, not
    below ``drop_min``."""
    p = P2.V2
    energy = np.asarray(data.energy, dtype=float)
    weight = np.asarray(data.weight, dtype=float)
    n = len(energy)
    drop = P2.mser_drop(energy, p.drop_min, p.drop_max)
    keep = int(p.min_blocks_after_drop)
    if n <= drop + keep and data.state in (B.DONE, B.STOPPED):
        drop = max(int(p.drop_min), n - keep - 1)
    if n <= drop + keep:
        return None
    mean, err = reblock(energy[drop:], weight[drop:])
    return float(mean), float(err), int(n - drop)


def results(engine: B.Engine, state: dict) -> dict:
    """What the report needs, read from the engine now.

    Returns ``{"rule", "targets", "rows"}``. ``targets`` is ``{name: {"same_geometry": bool}}``. ``rows`` is a list
    with one entry per target, geometry and cutoff, in the order of the record:

        target, geometry, label, cutoff, n_det_max    which wave function at which geometry
        keys, states, messages                        the DMC runs (series 1 first) and what the engine says
        done                                          True if every series is DONE
        energy, error                                 DMC energy in hartree of all series together (weights
                                                      1 / error^2, ``combine_series`` of the search), or None
        series, series_planned                        runs with an energy; runs that the stage asked for
        blocks, blocks_used                           DMC blocks of all series; blocks left after the start-up blocks
        trial, checks                                 ``meta["trial"]`` and ``meta["checks"]`` of the series-1 point
        cost                                          core-hours of all series. The engine adds the QP2 jobs of the
                                                      cutoff to the series-1 point, and the QP2 parent calculation
                                                      of a geometry to the first point that asked for it.
    """
    rows = []
    info = getattr(engine, "qp2_info", None)
    left = list(state.get("left_plan") or [])
    moved: Dict[str, float] = {}            # point of an older plan -> core-hours of its parent that a row took over
    for target, record in state["targets"].items():
        for geometry, part in record["geometries"].items():
            first_row = len(rows)
            keys_here = {key for cut in part["cutoffs"].values() for key in cut["keys"]}
            for label, cut in part["cutoffs"].items():
                datas = [engine.read(key) for key in cut["keys"]]
                stats = [s for s in (run_energy(d) for d in datas) if s is not None]
                energy = combine_series(stats) if stats else None
                meta = datas[0].meta or {}
                rows.append({
                    "target": target, "geometry": geometry, "label": label, "cutoff": cut["cutoff"],
                    "n_det_max": cut["n_det_max"], "keys": list(cut["keys"]),
                    "states": [d.state for d in datas], "messages": [d.message for d in datas],
                    "done": all(d.state == B.DONE for d in datas),
                    "energy": None if energy is None else energy[0], "error": None if energy is None else energy[1],
                    "series": len(stats), "series_planned": len(datas),
                    "blocks": int(sum(d.nblocks for d in datas)), "blocks_used": int(sum(s[2] for s in stats)),
                    "trial": dict(meta.get("trial") or {}), "checks": dict(meta.get("checks") or {}),
                    "cost": float(sum(engine.cost(key) for key in cut["keys"])),
                })
            # the engine counts the shared parent calculation (RHF and CIPSI) with the first point that asked for
            # it; when that point left the plan, the first row of the geometry gets it
            parent = info(part["qp2_key"]) if callable(info) else None
            if parent and parent.get("owner") and parent["owner"] not in keys_here and len(rows) > first_row:
                hours = float(parent.get("parent_core_hours") or 0.0)
                rows[first_row]["cost"] += hours
                moved[str(parent["owner"])] = hours
    # a parent calculation is counted once. The engine counts it with the point that asked for it first
    # (engine.cost of that point). When a row took it over, it is taken out of the cost of the older plan.
    cost_left = float(sum(max(0.0, engine.cost(key) - moved.get(key, 0.0)) for key in left))
    return {"rule": state["rule"], "rows": rows,
            "targets": {name: {"same_geometry": bool(record["same_geometry"])} for name, record in state["targets"].items()},
            "left_plan": left, "cost_left_plan": cost_left}


# ============================================================================= what can be said before any job runs
def machine_problems(machine, trial: dict) -> List[str]:
    """What a machine profile lacks for this stage (an empty list means nothing): the lines of
    ``Machine.check_qp2()``, and one line if the DMC series of one geometry do not all fit.

    QMCPACK has 1024 seed offsets, and a run on R MPI ranks uses 2 R of them. The engine refuses a series that
    would share random numbers with another series of its geometry (``max_series`` of ``engine/qmcpack.py``).
    The manual mode runs ``series`` DMC runs for each row. The automatic mode can add series up to
    ``series_max``, so it is that number which must fit."""
    problems = list(machine.check_qp2())
    ranks = int(machine.tasks("qmc"))
    key = "series_max" if trial.get("auto") is not None else "series"
    wanted = int(trial.get(key, trial["series"]))
    if ranks >= 1 and wanted > max_series(ranks):
        problems.append("[trial] %s = %d, but the QMC jobs of this profile use %d MPI ranks ([job.qmc] tasks), and "
                        "then at most %d DMC runs of one geometry can have random numbers of their own"
                        % (key, wanted, ranks, max_series(ranks)))
    return problems


def plan_lines(trial: dict, targets: Sequence[str]) -> List[str]:
    """The plan of the stage in plain lines, for the command ``check``: geometries, QP2 parent calculations,
    cutoffs and DMC series. The numbers of geometries and runs are upper limits, because a final geometry that
    equals its start geometry is not computed again. In the automatic mode the lines name the ladder, the
    tolerance and the two numbers of series, and they give the number of DMC runs at the start and at most."""
    cutoffs = list(trial["cutoffs"])
    numeric = [c for c in cutoffs if c != MD.HF_LABEL]
    geometries = 2 * len(targets)
    parent = ("RHF orbitals and a CIPSI wave function of %d determinants, used by all cutoffs of the geometry"
              % int(trial["n_det_max"]) if numeric else "RHF orbitals only, because every cutoff is hf")
    head = [
        "stage trial (after the geometry search): DMC energies with several determinants from Quantum Package 2",
        "  geometries: up to %d (the start geometry and the final geometry of %d target(s); a final geometry that "
        "equals its start geometry is not computed again)" % (geometries, len(targets)),
        "  Quantum Package parent calculations: up to %d, one per geometry (%s; basis file %s, pseudopotential file %s)"
        % (geometries, parent, trial["qp2_basis"], trial["qp2_pseudo"]),
    ]
    if trial.get("auto") is not None:
        ladder = [float(c) for c in trial["auto"]["ladder"]]
        series, series_max = int(trial["series"]), int(trial.get("series_max", trial["series"]))
        return head + [
            "  cutoffs: the automatic mode runs ([trial] cutoff = auto). It starts with the cutoffs %s and hf. Its "
            "ladder is %s ([trial] auto_cutoffs)" % (", ".join(R.cut_text(c) for c in ladder[:2]),
                                                    ", ".join(R.cut_text(c) for c in ladder)),
            "  rule: each cut is tested against hf, and every energy difference is tested between two cutoffs with "
            "the tolerance %s kcal/mol ([trial] tolerance). The stage adds DMC series or the next cutoff of the "
            "ladder when the test needs them" % R.number_text(float(trial.get("tolerance", R.DEFAULT_TOLERANCE))),
            "  DMC series: %d at the start and at most %d for each geometry and each cutoff ([trial] series and "
            "series_max)" % (series, series_max),
            "  DMC runs: up to %d at the start (3 cutoffs with hf, %d series) and at most %d (%d cutoffs with hf, "
            "%d series)" % (geometries * 3 * series, series, geometries * (len(ladder) + 1) * series_max,
                           len(ladder) + 1, series_max),
        ]
    return head + [
        "  cutoffs: %d (%s)" % (len(cutoffs), ", ".join("%g" % c if c != MD.HF_LABEL else MD.HF_LABEL for c in cutoffs)),
        "  DMC series: %d for each geometry and each cutoff, up to %d DMC run(s) in total"
        % (int(trial["series"]), geometries * len(cutoffs) * int(trial["series"])),
    ]


# ============================================================================= the stage
class TrialRuns:
    """The points of the stage ``trial`` and their record.

    Parameters
    ----------
    engine      source of the DMC blocks. Only ``submit``, ``read``, ``cost`` and, if it exists, ``retry`` are called.
    symbols     element symbols of the molecule, in the atom order of the user.
    geometries  ``{target name: (start coordinates, final coordinates)}``, each (n_atoms, 3) in angstrom. The
                order of the targets is the order of the points.
    trial       the dictionary ``Config.trial``: ``kind``, ``n_det_max``, ``cutoffs``, ``series``, ``qp2_basis``,
                ``qp2_pseudo``, and for the rule ``tolerance``, ``series_max`` and ``auto`` (None or missing: the
                manual mode; ``{"ladder": [...]}``: the automatic mode, see the module text).
    settings    QMC settings that every point gets. The key ``trial`` is added here for every point.
    blocks      number of DMC blocks of every run.
    state_dir   folder of the record ``state.json`` (``work/trial`` of a project).
    log         function that takes one line of text, or None.
    clock       function that gives the time in seconds, or None for the clock of the computer.
    kinds       ``{target name: "minimum" | "saddle" | ...}``, or None. The automatic mode tests the barrier of
                every pair of a saddle target and a minimum target; without kinds no barrier exists, and only
                the energy change of every target (final minus start) is tested.

    :meth:`tick` is one pass. The object keeps nothing but the record in memory, so a new object on the same
    folder continues: the engine does nothing for a point that exists. A new object whose plan differs from the
    record stops the points of the record that the plan no longer holds (see the module text).
    """

    def __init__(self, engine: B.Engine, symbols: Sequence[str], geometries: Dict[str, Tuple[np.ndarray, np.ndarray]],
                 trial: dict, settings: Optional[Dict[str, object]], blocks: int, state_dir: str,
                 log: Optional[Callable[[str], None]] = None, clock: Optional[Callable[[], float]] = None,
                 kinds: Optional[Dict[str, str]] = None):
        self.engine = engine
        self.clock = clock or time.time
        self.symbols = [str(s) for s in symbols]
        self.trial = copy.deepcopy(dict(trial))
        self.settings = copy.deepcopy(dict(settings or {}))
        self.blocks = int(blocks)
        self.state_dir = state_dir
        self._log = log
        self.kinds = {str(name): str(kind) for name, kind in (kinds or {}).items()}
        t = self.trial
        if t.get("kind") != "multidet":
            raise ValueError("the stage trial needs kind = multidet, not %r" % (t.get("kind"),))
        if not t.get("cutoffs"):
            raise ValueError("the stage trial needs at least one cutoff ([trial] cutoff)")
        if int(t["series"]) < 1 or self.blocks < 1:
            raise ValueError("the stage trial needs at least 1 series and 1 block (found %s and %d)" % (t["series"], self.blocks))
        if any(c != MD.HF_LABEL for c in t["cutoffs"]) and t.get("n_det_max") is None:
            raise ValueError("the stage trial needs the size of the CIPSI wave function for a numeric cutoff ([trial] n_det_max)")
        self.automatic = t.get("auto") is not None
        if self.automatic:
            self._check_automatic()
        os.makedirs(state_dir, exist_ok=True)
        self.state_file = os.path.join(state_dir, STATE_NAME)
        old = read_state(state_dir)
        self._same_files(old)
        # the geometries stay in memory: the automatic mode writes the plan again when it grows
        self._geometries = {str(name): (np.array(start, dtype=float).reshape(-1, 3), np.array(final, dtype=float).reshape(-1, 3))
                            for name, (start, final) in geometries.items()}
        cutoffs, series, auto, note = list(t["cutoffs"]), int(t["series"]), None, ""
        if self.automatic:                          # the plan that the record reached, or the plan of the input
            auto, note = self._automatic_record(old)
            cutoffs, series = list(auto["cutoffs"]) + [MD.HF_LABEL], int(auto["series"])
        self.state = {
            "version": 1, "status": RUNNING, "message": "",
            "rule": {"text": RULE, "trial": self.trial, "blocks": self.blocks, "same_geometry_angstrom": SAME_GEOMETRY,
                     "energy": {"drop": P2.V2.drop, "drop_min": int(P2.V2.drop_min), "drop_max": int(P2.V2.drop_max),
                                "min_blocks_after_drop": int(P2.V2.min_blocks_after_drop)}},
            "symbols": self.symbols, "targets": self._plan(self._geometries, cutoffs, series), "points": {},
            "to_stop": [], "stop_seen": {}, "left_plan": [], "refusals": {},
        }
        if auto is not None:                        # the manual mode keeps the record fields that it always had
            self.state["auto"] = auto
            self.state["rule"]["cutoff_rule"] = {
                "text": R.RULE_TEXT, "above_sigmas": R.ABOVE_SIGMAS, "above_sure_sigmas": R.ABOVE_SURE_SIGMAS,
                "verdict_sigmas": R.VERDICT_SIGMAS, "kinds": dict(self.kinds)}
        self.stopped: List[str] = []
        self._unlogged: List[str] = []              # lines of decisions whose record is not written yet
        if old is not None:                         # what the last pass saw: a change of state is logged once
            known, seen = set(self.keys()), old.get("points") or {}
            self.state["points"] = {k: v for k, v in seen.items() if k in known}
            if set(seen) == known:                  # the same points as before: the last status still holds
                self.state["status"], self.state["message"] = old.get("status", RUNNING), old.get("message", "")
            # points of the old plan that the new plan does not hold must not go on (their jobs cost core-hours). They
            # are stopped in the next pass, after the points of the new plan were asked for (see the module text).
            gone = [k for k in record_keys(old) if k not in known]
            pending = [k for k in old.get("to_stop") or [] if k not in known]
            self.state["to_stop"] = list(dict.fromkeys(pending + gone))
            self.state["stop_seen"] = {k: v for k, v in dict(old.get("stop_seen") or {}, **seen).items()
                                       if k in self.state["to_stop"]}
            self.state["left_plan"] = list(dict.fromkeys([k for k in old.get("left_plan") or [] if k not in known] + gone))
            self.state["refusals"] = {k: v for k, v in (old.get("refusals") or {}).items() if k in known}
        if auto is not None and not auto["final"] and self.state["status"] == DONE:
            # every run of this plan is DONE, but the decision about it has not been made (or must be made again)
            self.state["status"], self.state["message"] = RUNNING, ""
        self._save()
        if old is None:
            parts = sum(len(rec["geometries"]) for rec in self.state["targets"].values())
            self.say("plan: %d DMC run(s) of %d blocks. Targets: %d. Geometries: %d. Cutoffs: %d. Series: %d."
                     % (len(self.keys()), self.blocks, len(self.state["targets"]), parts, len(t["cutoffs"]),
                        int(t["series"])))
            for name, rec in self.state["targets"].items():
                if rec["same_geometry"]:
                    self.say("target %s: %s" % (name, rec["note"]))
        if note:
            self.say(note)

    # ------------------------------------------------------------------ the plan
    def _check_automatic(self) -> None:
        """Refuse values of the automatic mode that the stage cannot use. The input section makes the same checks
        (``config._trial_settings``); this method is for a caller that builds the dictionary itself."""
        t = self.trial
        ladder = t["auto"].get("ladder") if isinstance(t["auto"], dict) else None
        try:
            values = [float(c) for c in ladder]
        except (TypeError, ValueError):
            values = []
        if len(values) < 2 or any(not 0.0 < c < 1.0 for c in values) or any(b >= a for a, b in zip(values, values[1:])):
            raise ValueError("the automatic mode of the stage trial needs a ladder of at least 2 cutoffs between 0 and 1, "
                             "each smaller than the one before it ([trial] auto_cutoffs; found %r)" % (ladder,))
        # two cutoffs with the same short name would share their keys and folders: the plan would hold one of them
        names = [MD.cut_label(c) for c in values]
        if len(set(names)) != len(names):
            raise ValueError("the automatic mode of the stage trial needs a ladder whose cutoffs differ in their first %d "
                             "significant digits ([trial] auto_cutoffs; found %r)" % (MD.CUT_DIGITS, ladder))
        if list(t["cutoffs"]) != values[:2] + [MD.HF_LABEL]:
            raise ValueError("the automatic mode of the stage trial starts with the first two cutoffs of its ladder and "
                             "hf (%s), but the key cutoffs holds %r" % (values[:2] + [MD.HF_LABEL], t["cutoffs"]))
        if int(t.get("series_max", t["series"])) < int(t["series"]):
            raise ValueError("the automatic mode of the stage trial needs series_max of at least series (found %s and %s)"
                             % (t.get("series_max"), t["series"]))
        tolerance = float(t.get("tolerance", R.DEFAULT_TOLERANCE))
        if not 0.0 < tolerance < float("inf"):
            raise ValueError("the stage trial needs a tolerance larger than 0 kcal/mol ([trial] tolerance; found %r)"
                             % (t.get("tolerance"),))

    def _automatic_record(self, old: Optional[dict]) -> Tuple[dict, str]:
        """The record entry ``auto`` of this object, and one line for the log (empty: nothing to say).

        No record: the plan of the input (the first two cutoffs of the ladder, ``series`` runs each). A record of
        the automatic mode with the same input: exactly its entry, so that nothing is decided twice. The input
        changed, but ``n_det_max`` is the same and the cutoffs that the plan reached are still the first cutoffs
        of the ladder (the ladder is the same, or it changed only after them): the plan that was reached is kept
        (its cutoffs, and its number of series unless the input now starts with more), no run leaves the plan,
        and the decision is made again. In every other case (a cutoff of the plan that was reached is not at the
        same place in the new ladder, because another cutoff stands there or the ladder is now shorter than the
        plan; another ``n_det_max``; a record of the manual mode) the plan of the input."""
        t = self.trial
        ladder = [float(c) for c in t["auto"]["ladder"]]
        fresh = {"ladder": ladder, "tolerance": float(t.get("tolerance", R.DEFAULT_TOLERANCE)),
                 "series_max": int(t.get("series_max", t["series"])), "cutoffs": ladder[:2], "series": int(t["series"]),
                 "final": False, "verdict": None, "reason": "", "decisions": []}
        if old is None:
            return fresh, ("automatic cutoff: the ladder is %s; tolerance %s kcal/mol; %d DMC series at the start and at "
                           "most %d" % (", ".join(R.cut_text(c) for c in ladder), R.number_text(fresh["tolerance"]),
                                        fresh["series"], fresh["series_max"]))
        was, stored = old.get("auto"), (old.get("rule") or {}).get("trial") or {}
        start = "the plan starts from the input again; a point that exists is not computed again"
        if not isinstance(was, dict):
            return fresh, ("automatic cutoff: the record was written by the manual mode ([trial] cutoff was a list); the "
                           "plan of the automatic mode starts from the input, and a point that exists is not computed again")
        try:
            reached = [float(c) for c in was.get("cutoffs") or []]
            series = int(was.get("series"))
            same_ladder = [float(c) for c in was.get("ladder") or []] == ladder
        except (TypeError, ValueError):
            reached, series, same_ladder = [], 0, True
        if stored.get("n_det_max") != t.get("n_det_max"):
            return fresh, "automatic cutoff: [trial] n_det_max changed; " + start
        if series < 1 or len(reached) < 2 or (same_ladder and reached != ladder[:len(reached)]):
            # the plan of a record is always the first cutoffs of its ladder: this record was changed by hand or damaged
            return fresh, "automatic cutoff: the plan of the record does not fit its ladder; " + start
        if reached != ladder[:len(reached)]:
            return fresh, ("automatic cutoff: [trial] auto_cutoffs changed, and the cutoffs of the plan that was reached "
                           "(%s) are not the first cutoffs of the new ladder; %s"
                           % (", ".join(R.cut_text(c) for c in reached), start))
        if stored == t:                             # the same input: continue exactly where the record stands
            kept = dict(fresh, cutoffs=reached, series=series, final=bool(was.get("final")), verdict=was.get("verdict"),
                        reason=str(was.get("reason") or ""), decisions=copy.deepcopy(list(was.get("decisions") or [])))
            if kept["final"] and not kept["decisions"]:      # a final verdict needs the decision that gave it
                kept.update(final=False, verdict=None, reason="")
            return kept, ""
        # A decision of a record that an earlier development code wrote holds no ladder. It keeps none: the ladder of
        # the record can be newer than the decision, so a ladder given to it here could be false. The report then
        # says only that the input changed after a final decision.
        decisions = copy.deepcopy(list(was.get("decisions") or []))
        kept = dict(fresh, cutoffs=reached, series=max(series, int(t["series"])), decisions=decisions)
        return kept, ("automatic cutoff: the values of [trial] changed; the plan that was reached is kept (cutoffs %s and "
                      "hf, %d DMC series), and the decision is made again with the new values"
                      % (", ".join(R.cut_text(c) for c in reached), kept["series"]))

    def _same_files(self, old: Optional[dict]) -> None:
        """Refuse a record that was written with other QP2 files. A point that exists keeps its files, and no key
        carries the name of a file, so the stage would mix two basis sets without a message."""
        was = ((old or {}).get("rule") or {}).get("trial") or {}
        for key in ("qp2_basis", "qp2_pseudo"):
            if was.get(key) is not None and was[key] != self.trial[key]:
                raise ValueError("%s was written with [trial] %s = %s, and the input now gives %s. The points that exist "
                                 "keep their files, so two results could not be compared. Set the key back, or delete "
                                 "the folder of this record and the points of the stage (the folder trial of the QMC "
                                 "points) to compute the stage again" % (self.state_file, key, was[key], self.trial[key]))

    def _plan(self, geometries: Dict[str, Tuple[np.ndarray, np.ndarray]], cutoffs: Sequence[object],
              series: int) -> Dict[str, dict]:
        """The record of every target: its geometries, and for each geometry the keys of every cutoff.

        ``cutoffs`` and ``series`` are the plan: in the manual mode the values of the input, in the automatic
        mode the cutoffs (``hf`` last) and the number of series that the record entry ``auto`` holds now."""
        t = self.trial
        out: Dict[str, dict] = {}
        for name, (start, final) in geometries.items():
            start = np.asarray(start, dtype=float).reshape(-1, 3)
            final = np.asarray(final, dtype=float).reshape(-1, 3)
            if len(start) != len(self.symbols) or final.shape != start.shape:
                raise ValueError("target %s: %d symbols, %d atoms in the start geometry and %d in the final geometry"
                                 % (name, len(self.symbols), len(start), len(final)))
            distance = float(np.abs(final - start).max())
            same = distance < SAME_GEOMETRY
            parts: Dict[str, dict] = {}
            for geometry, coords in zip(GEOMETRIES, (start,) if same else (start, final)):
                cuts: Dict[str, dict] = {}
                for cutoff in cutoffs:
                    n_det_max = None if cutoff == MD.HF_LABEL else int(t["n_det_max"])
                    label = entry_label(cutoff, n_det_max)
                    cuts[label] = {"cutoff": cutoff, "n_det_max": n_det_max,
                                   "keys": point_keys(name, geometry, label, int(series))}
                parts[geometry] = {"coords": coords.tolist(), "qp2_key": qp2_key(name, geometry, t), "cutoffs": cuts}
            note = ("the final geometry equals the start geometry within %g angstrom: only the start geometry is computed"
                    % SAME_GEOMETRY) if same else ""
            out[str(name)] = {"same_geometry": same, "largest_displacement_angstrom": distance, "note": note,
                              "geometries": parts}
        return out

    @staticmethod
    def _cuts_in_order(part: dict) -> List[dict]:
        """The cutoffs of one geometry in the order of the requests: the numeric cutoffs first, then ``hf`` (see the
        module text)."""
        cuts = list(part["cutoffs"].values())
        return [c for c in cuts if c["cutoff"] != MD.HF_LABEL] + [c for c in cuts if c["cutoff"] == MD.HF_LABEL]

    def keys(self) -> List[str]:
        """Keys of all DMC runs of the stage, in the order in which they are submitted."""
        return [key for rec in self.state["targets"].values() for part in rec["geometries"].values()
                for cut in self._cuts_in_order(part) for key in cut["keys"]]

    def specs(self) -> List[B.PointSpec]:
        """The request for every DMC run, in the order of :meth:`keys`. A further series follows its series-1 point."""
        t = self.trial
        out = []
        for rec in self.state["targets"].values():
            for part in rec["geometries"].values():
                coords = np.array(part["coords"], dtype=float)
                for cut in self._cuts_in_order(part):
                    settings = copy.deepcopy(self.settings)
                    settings["trial"] = {"kind": "multidet", "cutoff": cut["cutoff"], "n_det_max": cut["n_det_max"],
                                         "qp2_key": part["qp2_key"], "qp2_basis": t["qp2_basis"],
                                         "qp2_pseudo": t["qp2_pseudo"]}
                    for n, key in enumerate(cut["keys"], start=1):
                        out.append(B.PointSpec(key=key, symbols=list(self.symbols), coords=coords.copy(),
                                               blocks=self.blocks, role="final", series=n,
                                               base_key=None if n == 1 else cut["keys"][0],
                                               settings=copy.deepcopy(settings)))
        return out

    # ------------------------------------------------------------------ small helpers
    @property
    def status(self) -> str:
        return self.state["status"]

    @property
    def message(self) -> str:
        return self.state["message"]

    def say(self, text: str) -> None:
        if self._log:
            self._log("%s  [trial] %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), text))

    def _save(self) -> None:
        tmp = self.state_file + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(self.state, fh, indent=1)
        os.replace(tmp, self.state_file)

    # ------------------------------------------------------------------ one pass
    def tick(self) -> str:
        """One pass: ask the engine for every point, read every point, write the record.

        A point has a *problem* when it is FAILED, was stopped before its last block, or its request was refused
        by the engine (a ``ValueError``: the request does not fit the point that exists, for example; it fails
        again on every pass). A point without a problem that is not DONE can still make progress. A refusal by
        the batch system or the file system (``RuntimeError``, ``OSError``) is no problem of the point: the
        request is made again on the next pass, and the engine starts the job of an existing point by itself.

        Returns ``DONE`` when every point is DONE and the engine accepted the request of every point in this pass.
        Returns ``NEEDS_ATTENTION`` when no point can make progress any more and at least one has a problem; the
        message names every such point with the reason. Else it returns ``RUNNING``: while other points can still
        make progress, the points with a problem wait, and the message names them as well.

        A point that is DONE in the engine but whose request was refused has a problem too: the engine refuses a
        request for a point that exists when the point holds another geometry or another trial wave function. Its
        energy then does not belong to the row of the plan, so the stage does not use it and needs attention. A
        point that is DONE and whose request the batch system or the file system refused in this pass does not
        count as DONE in this pass: the pass returns ``RUNNING``, and the request is made again.

        In the automatic mode the pass returns ``DONE`` only when every point is DONE and the final decision
        exists. A pass in which every point is DONE and the decision adds runs (one more series, or the next
        cutoff) returns ``RUNNING``: the next pass asks the engine for the new runs."""
        refused: Dict[str, str] = {}
        transient: Dict[str, str] = {}
        for spec in self.specs():
            if spec.base_key in refused:
                refused[spec.key] = "its series-1 point %s could not be submitted" % spec.base_key
                continue
            if spec.base_key in transient:
                transient[spec.key] = ("the request is made again on the next pass, after its series-1 point %s"
                                       % spec.base_key)
                continue
            try:
                self.engine.submit(spec)            # does nothing for a point that exists
            except (ValueError, FileNotFoundError, PermissionError) as exc:      # the request itself cannot succeed
                refused[spec.key] = "the point could not be submitted: %s" % exc
            except (OSError, RuntimeError) as exc:  # the batch system or the file system refused it this time
                transient[spec.key] = "the request is made again on the next pass: %s" % exc
        # a refusal of the batch system that lasts longer than REFUSAL_SECONDS is a problem of the point
        now = self.clock()
        refusals = self.state.setdefault("refusals", {})
        for key in list(refusals):
            if key not in transient:
                del refusals[key]
        for key, text in transient.items():
            if key not in refusals:
                refusals[key] = now
                self.say("point %s: %s" % (key, text))
            elif now - refusals[key] > REFUSAL_SECONDS:
                refused[key] = "refused for %.0f minutes: %s" % ((now - refusals[key]) / 60.0, text)
        for spec in self.specs():                   # a further series of such a point waits for nothing any more
            if spec.base_key in refused and spec.key not in refused:
                refused[spec.key] = "its series-1 point %s could not be submitted" % spec.base_key
                transient.pop(spec.key, None)
        # the points that left the plan are stopped now, after the points of the plan were asked for
        if self.state.get("to_stop"):
            stopped = stop_keys(self.engine, self.state["to_stop"], self.state.get("stop_seen"), log=self.say,
                                why="the plan of the stage no longer holds it (the values of [trial] changed)")
            self.stopped += stopped
            self.state["to_stop"], self.state["stop_seen"] = [], {}
        seen = self.state["points"]
        points: Dict[str, dict] = {}
        problems: List[str] = []
        count = {B.DONE: 0, B.RUNNING: 0}
        for key in self.keys():
            d = self.engine.read(key)
            message = refused.get(key) or transient.get(key) or d.message
            if key in refused:
                problems.append("%s: %s" % (key, message))
            elif d.state == B.FAILED:
                problems.append("%s: %s" % (key, message or "FAILED, and the engine gives no reason"))
            elif d.state == B.STOPPED:
                problems.append("%s: stopped after %d of %d blocks; the stage needs every run complete"
                                % (key, d.nblocks, self.blocks))
            # A point counts as DONE only when the engine accepted its request in this pass. A refused request is a
            # problem of the point (it holds another geometry or another trial wave function), and a request that
            # the batch system or the file system refused this time is made again on the next pass. A refused
            # point is not counted as writing blocks either: it is counted once, as a problem.
            if d.state == B.DONE and key not in refused and key not in transient:
                count[B.DONE] += 1
            elif d.state == B.RUNNING and key not in refused:
                count[B.RUNNING] += 1
            before = (seen.get(key) or {}).get("state")
            if d.state != before and d.state in B.FINAL_STATES:
                # the message of a point that is DONE is logged only when its request was refused
                tail = ": " + message if (message and (d.state != B.DONE or key in refused)) else ""
                self.say("point %s: %s after %d block(s)%s" % (key, d.state, d.nblocks, tail))
            elif key in refused and (seen.get(key) or {}).get("message") != message:
                self.say("point %s: %s" % (key, message))
            points[key] = {"state": d.state, "message": message, "blocks": d.nblocks}
        total = len(points)
        waiting = total - count[B.DONE] - count[B.RUNNING] - len(problems)
        progress = "%d of %d DMC run(s) are DONE, %d write blocks, %d wait" % (count[B.DONE], total, count[B.RUNNING], waiting)
        if count[B.DONE] == total:
            status, text = DONE, "all %d DMC run(s) are DONE" % total
        elif problems and count[B.DONE] + len(problems) == total:
            status, text = NEEDS_ATTENTION, "; ".join(problems)          # nothing can make progress any more
        elif problems:
            status = RUNNING
            text = "%s; %d run(s) need attention when the others have ended: %s" % (progress, len(problems), "; ".join(problems))
        else:
            status, text = RUNNING, progress
        waiting_refusals = [k for k in transient if k not in refused]
        if status == RUNNING and waiting_refusals:
            text += "; the batch system refused %d request(s), made again on every pass: %s" % (
                len(waiting_refusals), "; ".join("%s: %s" % (k, transient[k]) for k in waiting_refusals))
        if self.automatic:
            status, text, points, lines = self._automatic_pass(status, text, points)
            self._unlogged += lines
        self.state["points"], self.state["status"], self.state["message"] = points, status, text
        self._save()
        # a decision is logged after the record holds it, so that the log never names a decision that the record
        # does not hold; a line whose record could not be written waits for the next pass
        waiting_lines, self._unlogged = self._unlogged, []
        for line in waiting_lines:
            self.say(line)
        return status

    # ------------------------------------------------------------------ the automatic mode
    def _automatic_pass(self, status: str, text: str, points: Dict[str, dict]) -> Tuple[str, str, Dict[str, dict], List[str]]:
        """The part of one pass that only the automatic mode has. ``status``, ``text`` and ``points`` are what the
        pass found for the plan as it stands. Returns them as they must be written, and the lines for the log.

        When every run of the plan is DONE and no final decision exists, one decision is made (:meth:`_decide`).
        A decision that ends the stage keeps the status DONE. A decision that adds runs gives RUNNING; the next
        pass asks the engine for the new points. The message of the stage ends with the verdict, or with the plan
        while there is no final verdict."""
        auto = self.state["auto"]
        lines: List[str] = []
        if status == DONE and not auto["final"]:
            entry = self._decide()
            lines.append("automatic cutoff, decision %d: %s" % (len(auto["decisions"]), R.decision_text(entry)))
            if not auto["final"]:
                # the plan grew: its new points are in the record at once, so that a new process sees the same plan
                grown: Dict[str, dict] = {}
                for key in self.keys():
                    if key in points:
                        grown[key] = points[key]
                    else:
                        d = self.engine.read(key)
                        grown[key] = {"state": d.state, "message": d.message, "blocks": d.nblocks}
                done = sum(1 for p in grown.values() if p["state"] == B.DONE)
                running = sum(1 for p in grown.values() if p["state"] == B.RUNNING)
                # a run of an older plan that is in the plan again can have ended as FAILED or STOPPED (the stage
                # stops the runs that leave its plan). Such a run does not wait: the next pass names it as a problem
                ended = [key for key, p in grown.items() if p["state"] in (B.FAILED, B.STOPPED)]
                text = "%d of %d DMC run(s) are DONE, %d write blocks, %d wait" % (
                    done, len(grown), running, len(grown) - done - running - len(ended))
                if ended:
                    text += ("; %d run(s) that the plan holds again ended earlier as FAILED or STOPPED and need "
                             "attention: %s" % (len(ended), ", ".join(ended)))
                text += "; automatic cutoff: verdict %s with %d series, so %s" % (
                    entry["verdict"], int(entry["series"]), R.action_text(entry))
                return RUNNING, text, grown, lines
        if status == DONE:
            text += "; automatic cutoff: %s" % R.final_text(auto["decisions"][-1])
        elif status == RUNNING and auto["final"]:
            # a request for a run that is DONE was refused in this pass, after the final decision: the verdict stands
            text += "; automatic cutoff: the record holds the final verdict %s" % R.final_text(auto["decisions"][-1])
        elif status == RUNNING:
            text += "; automatic cutoff: no verdict yet (the plan holds the cutoffs %s and hf with %d series, at most %d)" % (
                ", ".join(R.cut_text(c) for c in auto["cutoffs"]), int(auto["series"]), int(auto["series_max"]))
        return status, text, points, lines

    def _decide(self) -> dict:
        """Make one decision of the automatic mode and put it into the record in memory: the entry in
        ``decisions``, and either the final verdict or the plan with one more series or one more cutoff. The
        caller writes the record. Returns the entry.

        The decision uses the energies of :func:`results` now. Nothing of the record is changed before every
        value of the decision is known, so an error while the points are read leaves the record as it was."""
        auto = self.state["auto"]
        res = results(self.engine, self.state)
        same = {name: bool(info["same_geometry"]) for name, info in res["targets"].items()}
        ev = R.evaluate(res["rows"], same, self.kinds, float(auto["tolerance"]))
        step = R.decide(ev, int(auto["series"]), int(auto["series_max"]), auto["cutoffs"], auto["ladder"])
        entry = dict(ev, time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.clock())),
                     series=int(auto["series"]), series_max=int(auto["series_max"]),
                     cutoffs=[float(c) for c in auto["cutoffs"]], ladder=[float(c) for c in auto["ladder"]],
                     action=step["action"], cause=step["cause"], added=step["added"])
        entry = plain(entry)
        if step["action"] == R.FINAL:
            auto.update(final=True, verdict=step["verdict"],
                        reason=entry["reason"] + R.final_tail(step["cause"], auto["series_max"], auto["series"],
                                                                   R.estimate_failures(ev)))
        elif step["action"] == R.MORE_SERIES:
            auto["series"] = int(step["added"])
        else:
            auto["cutoffs"] = [float(c) for c in auto["cutoffs"]] + [float(step["added"])]
        auto["decisions"].append(entry)
        if step["action"] != R.FINAL:
            self.state["targets"] = self._plan(self._geometries, list(auto["cutoffs"]) + [MD.HF_LABEL], int(auto["series"]))
            known = set(self.keys())                # a point that is in the plan again is no point of an older plan
            self.state["left_plan"] = [k for k in self.state.get("left_plan") or [] if k not in known]
            self.state["to_stop"] = [k for k in self.state.get("to_stop") or [] if k not in known]
            self.state["stop_seen"] = {k: v for k, v in (self.state.get("stop_seen") or {}).items()
                                       if k in self.state["to_stop"]}
        return entry

    def retry(self) -> List[str]:
        """Give every failed point of the plan a new set of attempts (``engine.retry``). A series-1 point comes
        before its further series, because they wait for it. A failed point of an older plan is not submitted
        again. Returns the keys that are active again."""
        retry = getattr(self.engine, "retry", None)
        out: List[str] = []
        if not callable(retry):
            return out
        for key in self.keys():
            if self.engine.read(key).state == B.FAILED and retry(key):
                out.append(key)
        if out:
            self.say("submitted again by request: %s" % ", ".join(out))
        return out

    def results(self) -> dict:
        """The energies and the facts of every geometry and cutoff, for the report (function :func:`results`)."""
        return results(self.engine, self.state)
