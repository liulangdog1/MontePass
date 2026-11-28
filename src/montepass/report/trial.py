"""The tables of the stage ``trial``: DMC energies with a multi-determinant trial wave function.

    trial_wavefunctions.csv     one line per target, geometry and cutoff; then one line per energy difference;
                                then the lines of the two tests of the size of the expansion
    trial_wavefunctions.md      the same numbers in tables for a person

Both files are written only for a project whose state holds the stage ``trial`` and that has a record of it
(``work/trial/state.json``). The numbers come from ``montepass.trial.results``: the facts of every wave
function as the engine reports them, and the DMC energy of all series of a geometry.

Rules of these tables
---------------------
* A difference of DMC energies is printed only if all its rows have the same cutoff and the same ``n_det_max``,
  every series of every row is DONE, and every row passed the VMC check without Jastrow factor. Else the cell
  reads ``NOT COMPARABLE:`` and the reason.
* An energy difference of the molecule is always between rows of the same cutoff. The error of a difference is
  the square root of the sum of the squared errors of its rows (the rows are independent runs). A row that a
  difference uses twice with opposite signs (the final geometry of a target that the search did not move is its
  start geometry) cancels exactly.
* No difference between the single-determinant energy of the search and a multi-determinant energy is printed.
  The two use different trial wave functions and so have different fixed-node errors; their difference is not an
  energy difference of the molecule. The single-determinant energy of the search stands in a column of its own.
  Both kinds of runs use the same basis set and pseudopotentials: for every geometry the engine checks that the
  RHF energies of PySCF and of Quantum Package 2 (QP2) agree (step ``mdc4q``); the tables show that difference.
* Every row and every difference that uses the final geometry of a target whose search ended NOT_CONVERGED is
  marked (input key ``[trial] not_converged = run``).

The size of the expansion (rule: module :mod:`montepass.trial_rule`; both modes of ``[trial] cutoff``)
-----------------------------------------------------------------------------------------------------
* Section "DMC energy against the single RHF determinant" (a plan with ``hf`` and a numeric cutoff): the DMC
  energy of every cut minus that of the row ``hf`` of this stage at the same geometry. Both rows use the RHF
  orbitals of QP2 and the same chain of programs, so the lower energy has the better nodes. This is the one
  place where rows of two cutoffs are subtracted, and it is no energy difference of the molecule.
* Section "Convergence with the size of the expansion" (at least two numeric cutoffs): every printed energy
  difference at the two smallest usable cutoffs, its change, and a verdict.
* Section "Automatic choice of the cutoff" (only with ``cutoff = auto``): the rule, the ladder, every decision
  of the stage from its record, the final verdict and the result.
"""
from __future__ import annotations

import csv
import os
import time
from typing import Dict, List, Optional, Sequence

from .. import trial as T
from .. import trial_rule as R
from ..engine.base import DONE, FINAL_STATES
from ..engine.multidet import HF_LABEL
from ..project import NEEDS_ATTENTION, TRIAL
# the rule of a printed difference is the one that the stage itself uses to test the size of the expansion
from ..trial_rule import NOT_COMPARABLE, not_comparable  # noqa: F401  (part of this module for its callers)
from ..trial_rule import combination as _combination
from ..trial_rule import used as _used
from .tables import ReportData, Skip, _md_table, atomic_write, fmt_pm

STATE_FOLDER = os.path.join("work", "trial")
NOT_COMPUTED = "not computed"
NOT_CONVERGED = "NOT_CONVERGED"
MARK = " (search NOT_CONVERGED)"
ORBITALS = ("The rows of this table use RHF orbitals of Quantum Package 2 (QP2); the single-determinant column uses "
            "B3LYP orbitals of PySCF. Both use the same basis set and the same pseudopotentials: for every geometry "
            "the engine checks that the RHF energies of the two programs agree (table \"Checks of the trial wave "
            "function\").")
NO_SUBTRACTION = ("So the two energy columns must not be subtracted from each other, and this report prints no such "
                  "difference: the two trial wave functions have different nodes, so their fixed-node errors differ, and "
                  "their difference is not an energy difference of the molecule.")
ONE_CUTOFF = "One cutoff cannot show whether the energy difference is converged with the size of the expansion."
AGAINST_HF = (
    "What this table tests. At every geometry the DMC energy with a cut is compared with the DMC energy with the "
    "single RHF determinant of Quantum Package 2. That is the row hf of this stage: it has the same RHF orbitals and "
    "the same chain of programs as the cut. Both are fixed-node DMC energies, so the lower energy has the better "
    "nodes. A cut whose energy is higher than that of the single determinant by more than %g standard errors has "
    "worse nodes than one determinant at that geometry: the cut is \"above\", and its cutoff is NOT USABLE. The "
    "measured cause for water and ammonia: the coefficients of the determinants come from Quantum Package without a "
    "Jastrow factor, and they are not optimized together with the Jastrow factor (docs/multireference.md, "
    "section 9). This table uses the row hf of this stage. It is no difference to the single-determinant energy of "
    "the search, which has the Kohn-Sham orbitals of PySCF: this report prints no such difference." % R.ABOVE_SIGMAS)
NO_HF_ROW = ("The stage holds no row hf, so no cut was tested against the single RHF determinant: every numeric cutoff "
             "counts as \"%s\". Add hf to [trial] cutoff, or use cutoff = auto, to get this test." % R.NOT_TESTED)
MANUAL_MODE = ("The cutoffs of this project were chosen by hand ([trial] cutoff is a list). So this section gives a "
               "verdict only, and the package adds no DMC run by itself. With [trial] cutoff = auto the package adds DMC "
               "series or the next cutoff of its ladder when the test needs them.")
ATTENTION_NOTE = ("At its last pass the stage had the status %s, with runs that need attention: %s. The stage does not "
                  "use such a run. A run whose request the engine refused is still printed in the tables of this file "
                  "with its energy, and the two tests of the size of the expansion still use it: do not use a value of a "
                  "row that the message names. A run that failed or was stopped makes its row not complete.")
CONVERGENCE_COLUMNS = (
    "The change is the value at the fine cutoff minus the value at the coarse cutoff, with its standard error. The "
    "bound is the size of the change plus %(n)g standard errors. CONVERGED: the bound is at most the tolerance. "
    "NOT_CONVERGED: the size of the change minus %(n)g standard errors is above the tolerance. NOT_DECIDED: neither. "
    "The change of a barrier (final minus start) is not tested: it is the difference of two tested values."
    % {"n": R.VERDICT_SIGMAS})

CSV_HEADER = [
    "row", "target", "kind", "geometry", "minus_target", "cutoff", "n_det_max", "n_det_cipsi", "n_det",
    "n_det_cipsi_over_n_det", "e_rhf_qp2_hartree", "e_var_cipsi_hartree", "pt2_cipsi_hartree", "pt2_cipsi_error_hartree",
    "e_var_plus_pt2_cipsi_hartree", "e_ref_hartree", "qp2_basis", "qp2_pseudo", "qp2_version",
    "rhf_qp2_minus_pyscf_millihartree", "vmc_check_energy_hartree", "vmc_check_error_hartree",
    "vmc_check_difference_millihartree", "vmc_check_limit_millihartree", "vmc_check_verdict", "vmc_check_runs",
    "dmc_states", "dmc_energy_hartree", "dmc_error_hartree", "dmc_series_with_energy", "dmc_series_planned", "dmc_blocks",
    "core_hours", "dmc_messages",
    "single_determinant_dmc_energy_hartree", "single_determinant_dmc_error_hartree",
    "search_status", "minus_target_search_status", "uses_not_converged_geometry",
    "difference_kcal_per_mol", "difference_error_kcal_per_mol",
    # the columns of the lines "against hf" and "convergence" (added after version 0.1.0, so they stand at the end)
    "dmc_minus_hf_millihartree", "dmc_minus_hf_error_millihartree", "against_hf", "cutoff_usable",
    "coarse_cutoff", "fine_cutoff", "difference_coarse_kcal_per_mol", "difference_coarse_error_kcal_per_mol",
    "difference_fine_kcal_per_mol", "difference_fine_error_kcal_per_mol", "change_fine_minus_coarse_kcal_per_mol",
    "change_error_kcal_per_mol", "bound_kcal_per_mol", "tolerance_kcal_per_mol", "verdict", "project_verdict",
    "project_verdict_reason", "automatic_final_verdict",
]


def available(project_dir: str) -> bool:
    """True if the project holds a record of the stage ``trial``."""
    return os.path.exists(os.path.join(project_dir, STATE_FOLDER, T.STATE_NAME))


def wanted(data: ReportData) -> bool:
    """True if the report must hold the two files: the project state holds the stage ``trial`` (the input asks
    for it) and the stage has a record. A project whose option was switched off gets no such files, and the
    report then reads none of the points of the stage."""
    return TRIAL in ((data.project or {}).get("stages") or {}) and available(data.project_dir)


# ============================================================================= numbers
def cutoff_text(cutoff) -> str:
    return HF_LABEL if cutoff == HF_LABEL else "%g" % float(cutoff)


def _difference(a: Optional[dict], b: Optional[dict]) -> dict:
    """Energy of row ``a`` minus energy of row ``b`` in kcal/mol, or the reason why it is not printed
    (``trial_rule.combination`` holds the rule and the arithmetic)."""
    return _combination([(a, 1.0), (b, -1.0)])


def _uses_not_converged(d: dict, statuses: Dict[str, str]) -> bool:
    """True if a tested difference of the rule uses the final geometry of a target whose search ended
    NOT_CONVERGED. A barrier at the start geometries uses no final geometry."""
    if d["geometry"] == "start":
        return False
    return NOT_CONVERGED in (statuses.get(d["target"]), statuses.get(d.get("minus_target")))


def collect(data: ReportData, engine) -> dict:
    """Everything that the two files hold.

    ``rows``: the entries of ``trial.results`` with three more keys: ``kind`` and ``search_status`` of the target,
    and ``single``, the single-determinant DMC energy of the same geometry from the search (``{"value", "error"}``
    or None). ``changes``: final minus start for every target and cutoff. ``barriers``: saddle minus minimum at
    the final geometries, ``start_barriers`` the same at the start geometries, and ``barrier_changes`` the
    difference of the two, for every pair of a saddle target and a minimum target and every cutoff.

    ``evaluation``: the rule for the size of the expansion on these rows (``trial_rule.evaluate``, with the
    tolerance of the record): each cut against the row ``hf``, and the verdict of every tested difference.
    Its lines against ``hf`` and its differences get the key ``marked``. ``auto``: the record entry of the
    automatic mode with its decisions, or None for a project of the manual mode. ``stage_status`` and
    ``stage_message``: the status and the message of the stage at its last pass (the file says so when the stage
    has runs that need attention, because the rows still hold every energy that is on disk)."""
    if engine is None:
        raise Skip("the report was built without an engine, so the points of the stage trial cannot be read")
    state = T.read_state(os.path.join(data.project_dir, STATE_FOLDER))
    if state is None:
        raise Skip("the project has no record of the stage trial")
    res = T.results(engine, state)
    rows = res["rows"]
    kinds = {name: (data.targets[name].kind if name in data.targets else "unknown") for name in res["targets"]}
    statuses = {name: (str(data.targets[name].status) if name in data.targets else "unknown") for name in res["targets"]}
    same = {name: bool(info["same_geometry"]) for name, info in res["targets"].items()}
    for r in rows:
        r["kind"], r["search_status"] = kinds[r["target"]], statuses[r["target"]]
        t = data.targets.get(r["target"])
        stat = None if t is None else (t.start_energy if r["geometry"] == "start" else t.final_energy)
        r["single"] = None if stat is None else {"value": stat.value, "error": stat.error}
    index = {(r["target"], r["geometry"], r["label"]): r for r in rows}
    labels = list(dict.fromkeys(r["label"] for r in rows))

    def start_row(name: str, label: str) -> Optional[dict]:
        return index.get((name, "start", label))

    def final_row(name: str, label: str) -> Optional[dict]:
        return index.get((name, "start" if same[name] else "final", label))

    def named(label: str, extra: dict) -> dict:
        r = next(x for x in rows if x["label"] == label)        # the cutoff and the size that the record names with this label
        return dict(extra, label=label, cutoff=r["cutoff"], n_det_max=r["n_det_max"])

    changes, barriers, start_barriers, barrier_changes = [], [], [], []
    for name in res["targets"]:
        for label in labels:
            diff = _combination([(final_row(name, label), 1.0), (start_row(name, label), -1.0)])
            changes.append(named(label, dict(diff, target=name, kind=kinds[name], same_geometry=same[name],
                                              search_status=statuses[name], marked=statuses[name] == NOT_CONVERGED)))
    for saddle in [n for n in res["targets"] if kinds[n] == "saddle"]:
        for minimum in [n for n in res["targets"] if kinds[n] == "minimum"]:
            pair = dict(saddle=saddle, minimum=minimum, saddle_status=statuses[saddle], minimum_status=statuses[minimum])
            marked = NOT_CONVERGED in (statuses[saddle], statuses[minimum])
            for label in labels:
                barriers.append(named(label, dict(_combination([(final_row(saddle, label), 1.0),
                                                                (final_row(minimum, label), -1.0)]),
                                                  geometry="final", marked=marked, **pair)))
                start_barriers.append(named(label, dict(_combination([(start_row(saddle, label), 1.0),
                                                                      (start_row(minimum, label), -1.0)]),
                                                        geometry="start", marked=False, **pair)))
                barrier_changes.append(named(label, dict(_combination([
                    (final_row(saddle, label), 1.0), (final_row(minimum, label), -1.0),
                    (start_row(saddle, label), -1.0), (start_row(minimum, label), 1.0)]),
                    geometry="final minus start", marked=marked, **pair)))
    for r in rows:
        r["marked"] = r["geometry"] == "final" and r["search_status"] == NOT_CONVERGED
    notes = {name: str(rec.get("note") or "") for name, rec in state["targets"].items()}
    # the size of the expansion: the same rule that the automatic mode of the stage applies. A record of version
    # 0.1.0 names no tolerance; it gets the default of the input file.
    tolerance = float((res["rule"].get("trial") or {}).get("tolerance") or R.DEFAULT_TOLERANCE)
    evaluation = R.evaluate(rows, same, kinds, tolerance, text=cutoff_text)
    for line in evaluation["against_hf"]:
        line["kind"], line["search_status"] = kinds[line["target"]], statuses[line["target"]]
        line["marked"] = line["geometry"] == "final" and line["search_status"] == NOT_CONVERGED
    for d in evaluation["differences"]:
        d["marked"] = _uses_not_converged(d, statuses)
    auto = state.get("auto") if isinstance(state.get("auto"), dict) else None
    return {"title": data.title, "written": time.strftime("%Y-%m-%d %H:%M:%S"), "rule": res["rule"], "rows": rows,
            "left_plan": list(res.get("left_plan") or []), "cost_left_plan": float(res.get("cost_left_plan") or 0.0),
            "changes": changes, "barriers": barriers, "start_barriers": start_barriers, "barrier_changes": barrier_changes,
            "notes": notes, "xc": (data.settings or {}).get("xc"), "statuses": statuses, "kinds": kinds,
            "one_cutoff": len((res["rule"].get("trial") or {}).get("cutoffs") or labels) == 1,
            "evaluation": evaluation, "auto": auto,
            "stage_status": str(state.get("status") or ""), "stage_message": str(state.get("message") or "")}


# ============================================================================= cells
def _fact(row: dict, name: str):
    """A fact of the wave function, or None while the engine does not report it."""
    return row["trial"].get(name)


def _ratio(row: dict) -> Optional[float]:
    parent, n = _fact(row, "n_det_parent"), _fact(row, "n_det")
    return None if (not parent or not n) else float(parent) / float(n)


def _ratio_cut(row: dict) -> str:
    """The ratio of :func:`_ratio` with 1 decimal that is cut and never rounded up: 3280 / 329 = 9.97 gives ``9.9``.
    So a ratio below a limit is printed below that limit. The two numbers of determinants are whole numbers, and
    the division is made with whole numbers."""
    tenths = (10 * int(_fact(row, "n_det_parent"))) // int(_fact(row, "n_det"))
    return "%d.%d" % divmod(tenths, 10)


def _check(row: dict) -> dict:
    """The VMC check without Jastrow factor of a row: energy and error in hartree, difference to ``e_ref`` and
    limit in millihartree, verdict, number of VMC runs. Every value is None while the check was not made.
    A check that failed inside the job of the point is in the record too (verdict FAIL)."""
    check = row["checks"].get("mdvmc") or {}
    out = {"energy": check.get("energy"), "error": check.get("error"), "verdict": check.get("verdict"),
           "runs": len(check["runs"]) if isinstance(check.get("runs"), (list, tuple)) else check.get("runs"),
           "difference": None, "limit": None}
    for name in ("difference", "limit"):
        if check.get(name) is not None:
            out[name] = 1000.0 * float(check[name])                         # hartree -> millihartree
    return out


def _rhf_difference(row: dict) -> Optional[float]:
    """RHF energy of QP2 minus RHF energy of PySCF in millihartree (check of step ``mdc4q``), or None."""
    value = (row["checks"].get("mdc4q") or {}).get("rhf_difference")
    return None if value is None else 1000.0 * float(value)


def _first_ended(row: dict) -> bool:
    """True if the series-1 run of a row has ended (DONE, FAILED or STOPPED)."""
    return bool(row["states"]) and row["states"][0] in FINAL_STATES


def _verdict(row: dict) -> str:
    """The verdict of the VMC check, or why there is none: "not made" for a run that ended without it, "not made
    yet" for a run that has not reached it."""
    verdict = _check(row)["verdict"]
    if verdict:
        return verdict
    return "not made" if (row["done"] or _first_ended(row)) else "not made yet"


def _cipsi(row: dict, name: str, spec: str) -> str:
    """A number of the CIPSI wave function. ``not computed`` for the cutoff hf (it has no CIPSI wave function) and
    for a number that Quantum Package did not print; empty while the facts are not known."""
    if row["cutoff"] == HF_LABEL:
        return NOT_COMPUTED
    if "n_det" not in row["trial"]:
        return ""
    value = _fact(row, name)
    return NOT_COMPUTED if value is None else spec % value


def _pt2(row: dict) -> str:
    """The second-order energy (PT2) of the CIPSI wave function with its statistical error, for a person."""
    value = _fact(row, "pt2_parent")
    if row["cutoff"] == HF_LABEL or "n_det" not in row["trial"] or value is None:
        return _cipsi(row, "pt2_parent", "%.6f")
    return fmt_pm(value, _fact(row, "pt2_error_parent"))


def _number(value, spec: str) -> str:
    return "" if value is None else spec % value


def _states(row: dict) -> str:
    return DONE if row["done"] else ", ".join(row["states"])


def _messages(row: dict) -> str:
    """The message of every run of a row that is not DONE, with its key."""
    return " / ".join("%s: %s" % (key, message or state) for key, state, message
                      in zip(row["keys"], row["states"], row["messages"]) if state != DONE)


# ============================================================================= writers
def write_csv(table: dict, path: str) -> None:
    """One line per target, geometry and cutoff (``row`` = ``energy``). Then one line per target and cutoff with
    the energy change final minus start (``row`` = ``final minus start``), one line per saddle, minimum, geometry
    and cutoff with the barrier (``row`` = ``barrier``, ``geometry`` = ``start`` or ``final``; ``target`` is the
    saddle, ``minus_target`` the minimum), and one line per saddle, minimum and cutoff with the change of the
    barrier (``row`` = ``barrier change``).

    Then the two tests of the size of the expansion (``trial_rule``). ``row`` = ``against hf``: one line per
    target, geometry and numeric cutoff with the DMC energy of the cut minus that of the row ``hf``, the result
    (``above``, ``not above`` or ``not available:`` and the reason) and what follows for the cutoff. ``row`` =
    ``convergence``: one line per tested energy difference with its values at the coarse and at the fine cutoff,
    their change, the bound and the verdict; every such line repeats the verdict of the project. When no
    difference was tested, one line holds the verdict of the project and its reason."""
    lines: List[Dict[str, object]] = []
    for r in table["rows"]:
        check, single = _check(r), r["single"]
        lines.append({
            "row": "energy", "target": r["target"], "kind": r["kind"], "geometry": r["geometry"],
            "cutoff": cutoff_text(_used(r, "cutoff")), "n_det_max": _number(_used(r, "n_det_max"), "%d"),
            "n_det_cipsi": _cipsi(r, "n_det_parent", "%d"), "n_det": _number(_fact(r, "n_det"), "%d"),
            "n_det_cipsi_over_n_det": _number(_ratio(r), "%.2f"), "e_rhf_qp2_hartree": _number(_fact(r, "e_scf"), "%.10f"),
            "e_var_cipsi_hartree": _cipsi(r, "e_var_parent", "%.10f"), "pt2_cipsi_hartree": _cipsi(r, "pt2_parent", "%.10f"),
            "pt2_cipsi_error_hartree": _cipsi(r, "pt2_error_parent", "%.10f"),
            "e_var_plus_pt2_cipsi_hartree": _cipsi(r, "e_pt2_parent", "%.10f"),
            "e_ref_hartree": _number(_fact(r, "e_ref"), "%.10f"),
            "qp2_basis": _fact(r, "qp2_basis") or "", "qp2_pseudo": _fact(r, "qp2_pseudo") or "",
            "qp2_version": _fact(r, "qp2_version") or "",
            "rhf_qp2_minus_pyscf_millihartree": _number(_rhf_difference(r), "%.6f"),
            "vmc_check_energy_hartree": _number(check["energy"], "%.10f"),
            "vmc_check_error_hartree": _number(check["error"], "%.10f"),
            "vmc_check_difference_millihartree": _number(check["difference"], "%.4f"),
            "vmc_check_limit_millihartree": _number(check["limit"], "%.4f"),
            "vmc_check_verdict": check["verdict"] or "", "vmc_check_runs": _number(check["runs"], "%d"),
            "dmc_states": _states(r), "dmc_energy_hartree": _number(r["energy"], "%.10f"),
            "dmc_error_hartree": _number(r["error"], "%.10f"), "dmc_series_with_energy": r["series"],
            "dmc_series_planned": r["series_planned"], "dmc_blocks": r["blocks"], "core_hours": "%.4f" % r["cost"],
            "dmc_messages": _messages(r),
            "single_determinant_dmc_energy_hartree": "" if single is None else "%.10f" % single["value"],
            "single_determinant_dmc_error_hartree": "" if single is None else "%.10f" % single["error"],
            "search_status": r["search_status"], "uses_not_converged_geometry": "yes" if r["marked"] else "",
        })
    for d in table["changes"]:
        lines.append(_difference_line("final minus start", d, d["target"], d["kind"], "final minus start", "",
                                      d["search_status"], ""))
    for name, entries in (("barrier", table["start_barriers"] + table["barriers"]), ("barrier change", table["barrier_changes"])):
        for d in entries:
            lines.append(_difference_line(name, d, d["saddle"], "saddle", d["geometry"], d["minimum"],
                                          d["saddle_status"], d["minimum_status"]))
    lines += _rule_lines(table)
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(CSV_HEADER)
        for line in lines:
            w.writerow([line.get(name, "") for name in CSV_HEADER])
    os.replace(tmp, path)


def _difference_line(row: str, d: dict, target: str, kind: str, geometry: str, minus: str, status: str,
                     minus_status: str) -> Dict[str, object]:
    return {"row": row, "target": target, "kind": kind, "geometry": geometry, "minus_target": minus,
            "cutoff": cutoff_text(d["cutoff"]), "n_det_max": _number(d["n_det_max"], "%d"),
            "search_status": status, "minus_target_search_status": minus_status,
            "uses_not_converged_geometry": "yes" if d["marked"] else "",
            "difference_kcal_per_mol": d["text"] if d["text"] else "%.6f" % d["value"],
            "difference_error_kcal_per_mol": "" if d["text"] else "%.6f" % d["error"]}


def _rule_lines(table: dict) -> List[Dict[str, object]]:
    """The CSV lines of the two tests of the size of the expansion: ``against hf`` and ``convergence``."""
    ev, statuses, kinds = table["evaluation"], table["statuses"], table["kinds"]
    lines: List[Dict[str, object]] = []
    usable = {cut["label"]: cut["status"] for cut in ev["usability"]}
    for line in ev["against_hf"]:
        available = line["state"] != R.NOT_AVAILABLE
        lines.append({
            "row": "against hf", "target": line["target"], "kind": line["kind"], "geometry": line["geometry"],
            "cutoff": cutoff_text(line["cutoff"]), "n_det_max": _number(line["n_det_max"], "%d"),
            "n_det_cipsi": _number(line["n_det_parent"], "%d"), "n_det": _number(line["n_det"], "%d"),
            "search_status": line["search_status"], "uses_not_converged_geometry": "yes" if line["marked"] else "",
            "dmc_minus_hf_millihartree": "%.6f" % line["delta"] if available else "",
            "dmc_minus_hf_error_millihartree": "%.6f" % line["error"] if available else "",
            "against_hf": line["state"] if available else "%s: %s" % (R.NOT_AVAILABLE, line["reason"]),
            "cutoff_usable": usable[line["label"]]})
    if len(ev["usability"]) < 2:                # one numeric cutoff cannot show convergence
        return lines
    auto = table.get("auto") or {}
    project = {"row": "convergence", "tolerance_kcal_per_mol": "%.6f" % ev["tolerance"], "project_verdict": ev["verdict"],
               "project_verdict_reason": ev["reason"],
               "automatic_final_verdict": str(auto.get("verdict") or "") if auto.get("final") else ""}
    if ev["pair"]:
        project.update(coarse_cutoff=cutoff_text(ev["pair"][0]), fine_cutoff=cutoff_text(ev["pair"][1]))
    for d in ev["differences"]:
        line = dict(project, target=d["target"], kind=kinds.get(d["target"], ""), geometry=d["geometry"],
                    minus_target=d["minus_target"] or "", search_status=statuses.get(d["target"], ""),
                    minus_target_search_status=statuses.get(d["minus_target"], "") if d["minus_target"] else "",
                    uses_not_converged_geometry="yes" if d["marked"] else "")
        if d["verdict"] is None:
            line["verdict"] = NOT_COMPARABLE + str(d["reason"])
        else:
            line.update(difference_coarse_kcal_per_mol="%.6f" % d["coarse"],
                        difference_coarse_error_kcal_per_mol="%.6f" % d["coarse_error"],
                        difference_fine_kcal_per_mol="%.6f" % d["fine"],
                        difference_fine_error_kcal_per_mol="%.6f" % d["fine_error"],
                        change_fine_minus_coarse_kcal_per_mol="%.6f" % d["S"], change_error_kcal_per_mol="%.6f" % d["e"],
                        bound_kcal_per_mol="%.6f" % d["bound"], verdict=d["verdict"])
        lines.append(line)
    if not ev["differences"]:
        lines.append(project)
    return lines


def _pm(value: Optional[float], error: Optional[float]) -> str:
    return "not available" if value is None else fmt_pm(value, error)


def _diff_cell(d: dict) -> str:
    return (d["text"] if d["text"] else fmt_pm(d["value"], d["error"])) + (MARK if d["marked"] else "")


def _files_sentence(rows: Sequence[dict]) -> str:
    """One sentence with the QP2 files and the QP2 version of the rows (every distinct value is named)."""
    parts = []
    for name, what in (("qp2_basis", "basis file"), ("qp2_pseudo", "pseudopotential file"),
                       ("qp2_version", "QP2 version (git commit)")):
        values = list(dict.fromkeys(str(r["trial"][name]) for r in rows if r["trial"].get(name)))
        parts.append("%s %s" % (what, ", ".join(values) if values else "not reported yet"))
    return "Quantum Package 2 (QP2) of these rows: %s." % "; ".join(parts)


def _against_hf_md(table: dict) -> List[str]:
    """The section "DMC energy against the single RHF determinant" (step 1 of the rule). Empty for a stage
    without a row ``hf`` or without a numeric cutoff."""
    ev = table["evaluation"]
    if not ev["hf"]:
        return []
    L: List[str] = ["## DMC energy against the single RHF determinant", ""]
    body = []
    for line in ev["against_hf"]:
        available = line["state"] != R.NOT_AVAILABLE
        body.append([line["target"], line["geometry"] + (MARK if line["marked"] else ""), cutoff_text(line["cutoff"]),
                     _number(line["n_det"], "%d") or "not available",
                     R.pm_text(line["delta"], line["error"]) if available else "not available",
                     line["state"] if available else "%s: %s" % (R.NOT_AVAILABLE, line["reason"])])
    L += _md_table(["target", "geometry", "cutoff", "N_det",
                    "DMC energy minus that of the single RHF determinant (millihartree)", "result"], body)
    for cut in ev["usability"]:
        mine = [line for line in ev["against_hf"] if line["label"] == cut["label"]]
        where = {state: ", ".join("%s %s" % (line["target"], line["geometry"]) for line in mine if line["state"] == state)
                 for state in (R.ABOVE, R.NOT_AVAILABLE)}
        if cut["status"] == R.USABLE:
            detail = "not above at %s" % ("its geometry" if len(mine) == 1 else "any of its %d geometries" % len(mine))
        elif cut["status"] == R.NOT_USABLE:
            detail = "above at %s" % where[R.ABOVE]
        else:
            detail = "not available at %s" % (where[R.NOT_AVAILABLE] or "every geometry")
        L.append("- Cutoff %s: %s (%s)." % (cutoff_text(cut["cutoff"]), cut["status"], detail))
    L += ["", AGAINST_HF, ""]
    notes = []
    for r in table["rows"]:
        ratio = _ratio(r)
        if r["cutoff"] != HF_LABEL and ratio is not None and ratio < R.PARENT_RATIO_NOTE:
            notes.append("- Note: at %s %s the cut %s holds %d of the %d determinants of its CIPSI wave function. The CIPSI "
                         "wave function is only %s times larger (less than %g times), so this cut depends on n_det_max."
                         % (r["target"], r["geometry"], cutoff_text(_used(r, "cutoff")), _fact(r, "n_det"),
                            _fact(r, "n_det_parent"), _ratio_cut(r), R.PARENT_RATIO_NOTE))
    return L + (notes + [""] if notes else [])


def _convergence_md(table: dict) -> List[str]:
    """The section "Convergence with the size of the expansion" (steps 2 to 4 of the rule). Empty for a stage
    with fewer than two numeric cutoffs."""
    ev = table["evaluation"]
    if len(ev["usability"]) < 2:
        return []
    L: List[str] = ["## Convergence with the size of the expansion", ""]
    if not ev["hf"]:                            # the second sentence only when two cutoffs are in fact compared
        L += [NO_HF_ROW + (" So the two smallest cutoffs of the list are compared." if ev["pair"] else ""), ""]
    if ev["pair"]:
        coarse, fine = cutoff_text(ev["pair"][0]), cutoff_text(ev["pair"][1])
        L += ["The two %scutoffs with the smallest values are compared: %s (coarse: fewer determinants) and %s (fine). "
              "Tolerance: %s kcal/mol ([trial] tolerance)."
              % ("usable " if ev["hf"] else "", coarse, fine, R.number_text(ev["tolerance"])), ""]
        body = []
        for d in ev["differences"]:
            name = d["name"] + (MARK if d["marked"] else "")
            if d["verdict"] is None:
                body.append([name] + ["not available"] * 4 + [NOT_COMPARABLE + str(d["reason"])])
            else:
                body.append([name, fmt_pm(d["coarse"], d["coarse_error"]), fmt_pm(d["fine"], d["fine_error"]),
                             R.pm_text(d["S"], d["e"]), R.bound_text(d), d["verdict"]])
        L += _md_table(["energy difference", "at the cutoff %s (kcal/mol)" % coarse, "at the cutoff %s (kcal/mol)" % fine,
                        "change: fine minus coarse (kcal/mol)", "bound (kcal/mol)", "verdict"], body)
        L += [CONVERGENCE_COLUMNS, ""]
    L += ["Verdict of the project: %s. Reason: %s." % (ev["verdict"], ev["reason"]), "",
          R.MEANING_TEXT, "", R.TOTAL_ENERGIES_TEXT, ""]
    if not table.get("auto"):
        L += [MANUAL_MODE, ""]
    return L


def _automatic_md(table: dict) -> List[str]:
    """The section "Automatic choice of the cutoff": the rule, the values of the input, every decision of the
    record, the final verdict and the result. Empty for a project of the manual mode."""
    auto = table.get("auto")
    if not auto:
        return []
    t = table["rule"].get("trial") or {}
    L: List[str] = ["## Automatic choice of the cutoff", "", R.RULE_TEXT, ""]
    L += ["Ladder ([trial] auto_cutoffs): %s. Tolerance: %s kcal/mol. DMC series: %s at the start and at most %s. The plan "
          "of the stage now holds the cutoffs %s and hf with %d DMC series for each geometry and each cutoff."
          % (", ".join(cutoff_text(c) for c in auto.get("ladder") or []), R.number_text(float(auto.get("tolerance") or 0.0)),
             t.get("series"), auto.get("series_max"), ", ".join(cutoff_text(c) for c in auto.get("cutoffs") or []),
             int(auto.get("series") or 0)), ""]
    decisions = list(auto.get("decisions") or [])
    if decisions:
        L += ["Decisions of the stage, in the order in which they were made:", ""]
        now = [float(c) for c in auto.get("ladder") or []]

        def then(entry: dict) -> str:
            """One sentence for a decision that was made with another ladder than the one of the input now."""
            was = entry.get("ladder")
            if not was or [float(c) for c in was] == now:     # no ladder, or an empty one: nothing to say
                return ""
            return " The ladder of the input was then %s." % ", ".join(cutoff_text(c) for c in was)

        def after(i: int, entry: dict) -> str:
            """One sentence for a final decision that another decision follows: the input changed in between."""
            if entry.get("action") != R.FINAL or i == len(decisions):
                return ""
            return " After this decision the values of [trial] changed, and the stage decided again."
        L += ["%d. %s: %s%s%s" % (i, entry.get("time", ""), R.decision_text(entry, cutoff_text), then(entry), after(i, entry))
              for i, entry in enumerate(decisions, start=1)]
        L.append("")
    else:
        L += ["The stage has made no decision yet: it decides when every DMC run of its plan is DONE.", ""]
    if not (auto.get("final") and decisions):
        changed = bool(decisions) and decisions[-1].get("action") == R.FINAL
        L += ["No final verdict yet: the stage has not ended.%s"
              % (" The values of [trial] changed after the last decision, so the stage decides again." if changed else ""), ""]
        return L
    last = decisions[-1]
    L += ["Final verdict: %s." % R.final_text(last, cutoff_text), ""]
    if not last.get("pair"):
        return L + ["Result: this verdict gives no energy difference at a fine cutoff.", ""]
    L += ["Result: the energy differences of the project are those at the fine cutoff %s." % cutoff_text(last["pair"][1]), ""]
    for d in last["differences"]:
        name = d["name"] + (MARK if _uses_not_converged(d, table["statuses"]) else "")
        if d["verdict"] is None:
            L.append("- %s: no value (%s%s)" % (name, NOT_COMPARABLE, d["reason"]))
        else:
            L.append("- %s: %s kcal/mol (%s; bound %s kcal/mol)" % (
                name, fmt_pm(d["fine"], d["fine_error"]), d["verdict"], R.bound_text(d)))
    return L + [""]


def write_md(table: dict, path: str) -> None:
    """The same content as the CSV file in tables, with the rules in plain sentences."""
    rule = table["rule"]
    t = rule.get("trial") or {}
    rows = table["rows"]
    L: List[str] = ["# %s: DMC energies with several determinants" % table["title"], "",
                    "Written %s. Every error bar is 1 standard error. The energy of one geometry is in hartree. "
                    "Energy differences are in kcal/mol." % table["written"], "", str(rule.get("text", "")), ""]
    auto = table.get("auto")
    tolerance = R.number_text(table["evaluation"]["tolerance"])
    if t and auto:
        L += ["Values of the input section [trial]: n_det_max = %s; cutoff = auto with the ladder %s (auto_cutoffs); "
              "tolerance = %s kcal/mol; %s DMC series at the start and at most %s (series and series_max), of %s blocks "
              "each, for each geometry and each cutoff; basis file %s and pseudopotential file %s of Quantum Package; "
              "not_converged = %s. The plan of the stage now holds the cutoffs %s and hf with %d DMC series."
              % (t.get("n_det_max"), ", ".join(cutoff_text(c) for c in auto.get("ladder") or []), tolerance,
                 t.get("series"), auto.get("series_max"), rule.get("blocks"), t.get("qp2_basis"), t.get("qp2_pseudo"),
                 t.get("not_converged", "stop"), ", ".join(cutoff_text(c) for c in auto.get("cutoffs") or []),
                 int(auto.get("series") or 0)), ""]
    elif t:
        L += ["Values of the input section [trial]: n_det_max = %s; cutoff = %s; %s DMC series of %s blocks for each "
              "geometry and each cutoff; basis file %s and pseudopotential file %s of Quantum Package; not_converged "
              "= %s; tolerance = %s kcal/mol."
              % (t.get("n_det_max"), ", ".join(cutoff_text(c) for c in t.get("cutoffs") or []), t.get("series"),
                 rule.get("blocks"), t.get("qp2_basis"), t.get("qp2_pseudo"), t.get("not_converged", "stop"), tolerance), ""]
    L += ["Search status of every target: %s." % ", ".join("%s %s" % kv for kv in table["statuses"].items()), ""]
    # the note is printed also while other runs still write blocks: the message of the stage then names the runs
    # that need attention when the others have ended
    stage_status, stage_message = str(table.get("stage_status") or ""), str(table.get("stage_message") or "")
    if stage_status == NEEDS_ATTENTION or (stage_status != DONE and "need attention" in stage_message):
        L += [ATTENTION_NOTE % (stage_status, (stage_message or "no message").rstrip(".")), ""]
    for name, note in table["notes"].items():
        if note:
            L.append("- Target %s: %s." % (name, note))
    if any(table["notes"].values()):
        L.append("")

    def key(r: dict) -> List[str]:
        return [r["target"], r["geometry"] + (MARK if r["marked"] else ""), cutoff_text(_used(r, "cutoff"))]

    waiting = "not available"
    L += ["## Trial wave functions", "", _files_sentence(rows), ""]
    L += _md_table(["target", "geometry", "cutoff", "n_det_max", "N_det of the CIPSI wave function", "N_det",
                    "N_det (CIPSI) / N_det", "RHF energy of Quantum Package", "variational energy of the CIPSI wave function",
                    "second-order energy (PT2) of the CIPSI wave function", "variational energy + PT2", "e_ref"],
                   [key(r) + [_number(_used(r, "n_det_max"), "%d") or "none", _cipsi(r, "n_det_parent", "%d") or waiting,
                              _number(_fact(r, "n_det"), "%d") or waiting, _number(_ratio(r), "%.1f") or "n/a",
                              _number(_fact(r, "e_scf"), "%.6f") or waiting, _cipsi(r, "e_var_parent", "%.6f") or waiting,
                              _pt2(r) or waiting, _cipsi(r, "e_pt2_parent", "%.6f") or waiting,
                              _number(_fact(r, "e_ref"), "%.6f") or waiting] for r in rows])
    L += ["N_det is the number of determinants of the trial wave function: the CIPSI wave function without the "
          "determinants that the cutoff removes. e_ref is the energy that Quantum Package printed for that wave function. "
          "The cutoff hf is the single RHF determinant of Quantum Package: it has no CIPSI wave function "
          "(\"%s\"), and its e_ref is the RHF energy. \"%s\" marks a number of a calculation that has not ended. "
          "All energies of this table are in hartree." % (NOT_COMPUTED, waiting), ""]

    L += ["## Checks of the trial wave function", ""]
    body = []
    for r in rows:
        c = _check(r)
        body.append(key(r) + [_number(_rhf_difference(r), "%+.6f") or waiting,
                              _pm(c["energy"], c["error"]), _number(_fact(r, "e_ref"), "%.6f") or waiting,
                              _number(c["difference"], "%+.3f") or waiting, _number(c["limit"], "%.3f") or waiting,
                              _verdict(r), _number(c["runs"], "%d") or "0"])
    L += _md_table(["target", "geometry", "cutoff", "RHF of QP2 minus RHF of PySCF (millihartree)", "VMC energy (hartree)",
                    "e_ref (hartree)", "VMC energy minus e_ref (millihartree)", "limit (millihartree)", "verdict",
                    "VMC runs"], body)
    L += ["The RHF energies of QP2 and of PySCF at the same geometry show that the two programs use the same basis set "
          "and pseudopotentials. The VMC energy of the determinants alone, with no Jastrow factor, must equal e_ref "
          "within the limit. This check finds a wrong orbital file, a lost determinant, a wrong atom order and a wrong "
          "pseudopotential. \"not made yet\" marks a check of a run that has not reached it.", ""]

    L += ["## DMC energies", ""]
    L += _md_table(["target", "geometry", "cutoff", "state of the DMC runs", "DMC energy (hartree)", "series with an energy",
                    "DMC blocks", "core-hours", "single-determinant DMC energy of the same geometry (hartree)",
                    "message of every run that is not DONE"],
                   [key(r) + [_states(r), _pm(r["energy"], r["error"]), "%d of %d" % (r["series"], r["series_planned"]),
                              r["blocks"], "%.1f" % r["cost"] if r["cost"] > 0 else "not known",
                              waiting if r["single"] is None else fmt_pm(r["single"]["value"], r["single"]["error"]),
                              _messages(r) or "-"]
                    for r in rows])
    L += [ORBITALS, ""]
    xc = str(table.get("xc") or "")
    if xc and xc.lower() != "b3lyp":            # the sentence above names the default functional of the package
        L += ["This project set [qmc] xc = %s: its single-determinant column uses the Kohn-Sham orbitals of that "
              "functional." % xc, ""]
    L += [NO_SUBTRACTION + " The single-determinant energy is the energy of the geometry search: its start energy in a "
          "row of a start geometry, its final energy in a row of a final geometry. The core-hours of a row hold the "
          "Quantum Package jobs of its cutoff. The Quantum Package calculation that all cutoffs of one geometry share "
          "(RHF and CIPSI) is counted once, in the first row of that geometry whose point asked for it, or in the "
          "first row of the geometry when that point is no longer in the plan.", ""]
    if table["left_plan"]:
        L += ["%d DMC run(s) of an older plan of the stage (a value of [trial] changed) are not in these tables; they "
              "used %.1f core-hours." % (len(table["left_plan"]), table["cost_left_plan"]), ""]

    # the size of the expansion: each cut against the single determinant, the convergence of every difference,
    # and (automatic mode) what the stage decided
    L += _against_hf_md(table) + _convergence_md(table) + _automatic_md(table)
    if len(table["evaluation"]["usability"]) == 1:
        # one numeric cutoff: no convergence section exists. What that section says in every report stands here:
        # whether the cut was tested against the single determinant, and the sentence about total energies
        if not table["evaluation"]["hf"]:
            L += [NO_HF_ROW, ""]
        L += [R.TOTAL_ENERGIES_TEXT, ""]

    not_converged = [n for n, s in table["statuses"].items() if s == NOT_CONVERGED]
    warning = ("The search of target(s) %s ended NOT_CONVERGED: it did not reach its precision target. A row or a value "
               "marked \"%s\" uses the final geometry of such a search, and that geometry is not a converged stationary "
               "point." % (", ".join(not_converged), MARK.strip())) if not_converged else ""
    L += ["## Energy change of every target: final geometry minus start geometry", ""]
    L += _md_table(["target", "kind", "search status", "cutoff", "n_det_max", "final minus start (kcal/mol)"],
                   [[d["target"], d["kind"], d["search_status"], cutoff_text(d["cutoff"]),
                     _number(d["n_det_max"], "%d") or "none",
                     _diff_cell(d) + (" (the final geometry is the start geometry)" if d["same_geometry"] else "")]
                    for d in table["changes"]])
    if warning and any(d["marked"] for d in table["changes"]):
        L += [warning, ""]
    if table["barriers"]:
        L += ["## Barriers: saddle minus minimum", ""]
        start = {(d["saddle"], d["minimum"], d["label"]): d for d in table["start_barriers"]}
        change = {(d["saddle"], d["minimum"], d["label"]): d for d in table["barrier_changes"]}
        L += _md_table(["saddle", "minimum", "search of the saddle", "search of the minimum", "cutoff", "n_det_max",
                        "barrier at the start geometries (kcal/mol)", "barrier at the final geometries (kcal/mol)",
                        "final minus start (kcal/mol)"],
                       [[d["saddle"], d["minimum"], d["saddle_status"], d["minimum_status"], cutoff_text(d["cutoff"]),
                         _number(d["n_det_max"], "%d") or "none",
                         _diff_cell(start[d["saddle"], d["minimum"], d["label"]]), _diff_cell(d),
                         _diff_cell(change[d["saddle"], d["minimum"], d["label"]])] for d in table["barriers"]])
        L += ["The barrier at the start geometries uses the start geometry of both targets; the barrier at the final "
              "geometries uses their final geometries. A target that the search did not move has the same row in both.", ""]
        if warning and any(d["marked"] for d in table["barriers"]):
            L += [warning, ""]
    if table["one_cutoff"]:
        L += [ONE_CUTOFF, ""]
    L += ["A difference is printed only if all its rows have the same cutoff and the same n_det_max, every DMC series "
          "of every row is DONE, and every row passed the VMC check. Else the cell reads \"%s\" and the reason. The "
          "error of a difference is the square root of the sum of the squared errors of its rows." % NOT_COMPARABLE.strip(), ""]
    atomic_write(path, "\n".join(L))


def write(data: ReportData, engine, results_dir: str) -> List[str]:
    """Write the two files into ``results_dir`` and return their paths."""
    table = collect(data, engine)
    files = [os.path.join(results_dir, "trial_wavefunctions.csv"), os.path.join(results_dir, "trial_wavefunctions.md")]
    write_csv(table, files[0])
    write_md(table, files[1])
    return files
