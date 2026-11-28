"""The rule of the stage ``trial`` for the size of the determinant expansion.

The stage ``trial`` (module :mod:`montepass.trial`) computes DMC energies with CIPSI wave functions that a
cutoff has cut. A short expansion can give a DMC energy ABOVE that of one determinant, and an energy
difference can still change when the expansion grows. This module holds the rule that tests both. It
holds pure functions only: every function takes plain values (the rows of ``trial.results``, numbers,
lists) and returns plain values. No function reads a file, and no function calls an engine. The three
names that this module takes from the engine package are constants: the state ``DONE``, the label ``hf``,
and the number of digits that tell two cutoffs apart.

Who uses the rule
-----------------
* The stage, in the automatic mode (``[trial] cutoff = auto``): after all DMC runs of its plan are done it
  calls :func:`evaluate` and :func:`decide`, and then ends, adds a DMC series, or adds the next cutoff.
* The report, in both modes: it prints the test of each cut against the single determinant and the test
  of every energy difference (module :mod:`montepass.report.trial`).

The rule in five steps (the text for a reader is :data:`RULE_TEXT`)
-------------------------------------------------------------------
1. Each cut against the single determinant (:func:`against_hf`). For every numeric cutoff and every geometry
   that has a row ``hf``: ``delta = E(cut) - E(hf)`` and ``error = sqrt(err(cut)^2 + err(hf)^2)``. The cut is
   "above" at that geometry if ``delta > ABOVE_SIGMAS * error``. A cutoff is USABLE if it is above at no
   geometry. A row that is not complete gives "not available" with the reason.
2. The pair: of the usable cutoffs, the two with the smallest values. "coarse" is the larger value (fewer
   determinants), "fine" the smaller one. A plan without a row ``hf`` tests no cut, and then the pair is taken
   from all numeric cutoffs.
3. Every tested difference ``D`` (:func:`difference_specs`): ``S = D(fine) - D(coarse)``,
   ``e = sqrt(err(D fine)^2 + err(D coarse)^2)``, ``bound = |S| + VERDICT_SIGMAS * e``. ``CONVERGED`` if
   ``bound <= tolerance``; ``NOT_CONVERGED`` if ``|S| - VERDICT_SIGMAS * e > tolerance``; else ``NOT_DECIDED``.
4. The verdict of the project (:func:`evaluate`).
5. The result: the energy differences at the fine cutoff of the pair.

Units: an energy of one geometry is in hartree. ``delta`` and its error are in millihartree. Energy
differences, ``S``, ``e``, ``bound`` and the tolerance are in kcal/mol.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .engine.base import DONE
from .engine.multidet import CUT_DIGITS, HF_LABEL
from .units import HARTREE_TO_KCAL

ABOVE_SIGMAS = 2.0                      # a cut is "above" the single determinant beyond this many standard errors
ABOVE_SURE_SIGMAS = 3.0                 # an "above" result up to this many standard errors can be tested again (above_to_test_again)
VERDICT_SIGMAS = 2.0                    # margin of the verdict of an energy difference, in standard errors
PARENT_RATIO_NOTE = 10.0                # a cut that holds more than 1 / this of its CIPSI wave function gets a note
DEFAULT_TOLERANCE = 1.0                 # kcal/mol: [trial] tolerance of an input or a record that names none

# verdict words
CONVERGED = "CONVERGED"
NOT_CONVERGED = "NOT_CONVERGED"
NOT_DECIDED = "NOT_DECIDED"
NO_DIFFERENCE = "NO_DIFFERENCE"         # the project prints no energy difference, so nothing can be tested
NO_VERDICT = "NO_VERDICT"               # a row is not complete, or a difference cannot be compared
VERDICTS = (CONVERGED, NOT_CONVERGED, NOT_DECIDED, NO_DIFFERENCE, NO_VERDICT)

# a cut at one geometry, against the single determinant
ABOVE = "above"
NOT_ABOVE = "not above"
NOT_AVAILABLE = "not available"

# a cutoff, over all geometries
USABLE = "USABLE"
NOT_USABLE = "NOT USABLE"
NOT_TESTED = "not tested"               # the plan holds no row hf
NO_VERDICT_YET = "no verdict yet"       # a row of the cutoff, or the row hf of a geometry, is not complete

# what the stage does after a decision (automatic mode)
FINAL = "final"
MORE_SERIES = "more series"
NEXT_CUTOFF = "next cutoff"

NOT_COMPARABLE = "NOT COMPARABLE: "

RULE_TEXT = (
    "Rule for the size of the determinant expansion. "
    "Step 1, each cut against the single determinant: at every geometry the DMC energy with a cut is compared with "
    "the DMC energy with the single RHF determinant (cutoff hf, same orbitals). A cut is \"above\" at a geometry if "
    "its energy is higher by more than %(above)g standard errors of the difference. A cutoff is USABLE if it is "
    "above at no geometry. "
    "Step 2, the pair: of the usable cutoffs, the two with the smallest values are compared. The larger value is the "
    "coarse cutoff (fewer determinants), the smaller value is the fine cutoff. "
    "Step 3, every tested energy difference D: final minus start of every target that the search moved, and the "
    "barrier at the start geometries and at the final geometries of every pair of a saddle and a minimum. With "
    "S = D(fine) - D(coarse) and its standard error e, the difference is CONVERGED if |S| + %(verdict)g e is at most "
    "the tolerance, NOT_CONVERGED if |S| - %(verdict)g e is larger than the tolerance, and NOT_DECIDED otherwise. "
    "Step 4, the project: NOT_CONVERGED if one difference is NOT_CONVERGED or if fewer than two cutoffs are usable; "
    "else NOT_DECIDED if one difference is NOT_DECIDED; else CONVERGED. "
    "Step 5, the result: the energy differences of the project are those at the fine cutoff. "
    "In the automatic mode ([trial] cutoff = auto) the stage starts with the first two cutoffs of its ladder and "
    "with hf, and it applies the rule when all DMC runs of its plan are done. CONVERGED ends the stage, also when a "
    "smaller cutoff of the plan is above by at most %(sure)g standard errors. "
    "NO_DIFFERENCE (the project prints no energy difference) and NO_VERDICT (a row is not complete, or a difference "
    "cannot be compared) also end the stage. With NOT_CONVERGED or NOT_DECIDED the stage takes the first of the next "
    "three steps that applies. First step: a cutoff that is above by at most %(sure)g standard errors is tested again "
    "with more data, and every row gets one more DMC series while the number of series is below [trial] series_max. "
    "This step applies only to a cutoff that is above by more than %(sure)g standard errors at no geometry and that "
    "has fewer than two usable cutoffs with a smaller value, and only when these cutoffs and the usable cutoffs are "
    "at least two together: more data can change the two cutoffs that are compared only in this case. Second step: "
    "with NOT_CONVERGED the next cutoff of the ladder is added; when the ladder holds no further cutoff, the stage "
    "ends with NOT_CONVERGED. Third step: with NOT_DECIDED every row gets one more DMC series. When no further series "
    "is allowed ([trial] series_max), the stage estimates the error of every tested difference for the next pair of "
    "cutoffs: sqrt(2) times its error at the fine cutoff. If %(verdict)g times this estimate is larger than the "
    "tolerance for a difference, a smaller cutoff cannot give the verdict CONVERGED with the allowed series: the stage "
    "ends with NOT_DECIDED, and its message says to raise series_max. Else the next cutoff of the ladder is added; when "
    "the ladder holds no further cutoff, the stage ends with NOT_DECIDED, and its message says to add a smaller cutoff."
) % {"above": ABOVE_SIGMAS, "verdict": VERDICT_SIGMAS, "sure": ABOVE_SURE_SIGMAS}

MEANING_TEXT = (
    "What the verdict means. CONVERGED: every tested energy difference changed by less than the tolerance between "
    "the two largest usable expansions that were computed, with %(verdict)g standard errors of margin. It is a "
    "statement about these two cutoffs. It does not prove that a much larger expansion gives the same value. "
    "NOT_CONVERGED: at least one difference changed by more than the tolerance, with the same margin, or fewer than "
    "two cutoffs are usable. NOT_DECIDED: the error bars are too large to say either. NO_DIFFERENCE: the project "
    "prints no energy difference, so nothing can be tested. NO_VERDICT: a row of the stage is not complete, or a "
    "difference cannot be compared, so no test is possible."
) % {"verdict": VERDICT_SIGMAS}

TOTAL_ENERGIES_TEXT = (
    "Total energies are not converged with the size of the expansion at the cutoffs of the default ladder: for water "
    "and ammonia the DMC energy of one geometry fell by 3.10 +/- 0.41 and 1.24 +/- 0.38 millihartree between the "
    "cutoffs 2.5e-4 and 1e-4 (docs/multireference.md, section 9). The tests of this report are about energy "
    "differences only; they say nothing about the total energy of one geometry."
)


# ============================================================================= texts of numbers
def cut_text(cutoff) -> str:
    """A cutoff as a person writes it in the input file: ``2.5e-4``, ``1e-4``; ``hf`` for the single RHF
    determinant. The number keeps the digits that tell two cutoffs apart (``CUT_DIGITS`` of the engine)."""
    if cutoff == HF_LABEL:
        return HF_LABEL
    mantissa, exponent = ("%.*e" % (CUT_DIGITS - 1, float(cutoff))).split("e")
    return "%se%d" % (mantissa.rstrip("0").rstrip("."), int(exponent))


def number_text(value: float) -> str:
    """A number with at least one decimal: ``1.0``, ``0.5``, ``0.25``."""
    text = "%g" % float(value)
    return text if ("." in text or "e" in text or "n" in text) else text + ".0"


def decimals_of(error: Optional[float]) -> int:
    """The number of decimals that 2 significant digits of ``error`` need (2 when the error is not above 0)."""
    if error is None or not error > 0.0 or not math.isfinite(error):
        return 2
    return int(min(10, max(0, 1 - int(math.floor(math.log10(error))))))


def pm_text(value: float, error: float) -> str:
    """``value +/- error`` with the sign of the value. An error of exactly 0 is written as ``(exact)``."""
    if not error > 0.0:
        return "%+.6f (exact)" % value
    decimals = decimals_of(error)
    return "%+.*f +/- %.*f" % (decimals, value, decimals, error)


def bound_text(difference: dict) -> str:
    """The bound of a tested difference, with the decimals of its change ``S +/- e``."""
    return "%.*f" % (decimals_of(difference["e"]), difference["bound"])


# ============================================================================= rows and their differences
def used(row: dict, name: str):
    """The cutoff or the ``n_det_max`` that a point was computed with: the value that the engine reports, or the
    value of the record while the engine has none yet."""
    got = row["trial"].get(name)
    return row[name] if got is None else got


def row_reasons(row: dict) -> List[str]:
    """Why a row is not complete; an empty list for a complete row. A complete row has every series DONE, a
    DMC energy, and a VMC check without Jastrow factor that passed."""
    reasons: List[str] = []
    check = row["checks"].get("mdvmc")
    if not row["done"]:
        reasons.append("%d of %d series are DONE (%s)" % (row["states"].count(DONE), len(row["states"]),
                                                          ", ".join(row["states"])))
    if check and not check.get("ok"):
        reasons.append("the VMC check without Jastrow factor did not pass (verdict %s)" % check.get("verdict", "FAIL"))
    if row["done"] and not check:               # for a row that is not DONE this follows from the first reason
        reasons.append("the VMC check without Jastrow factor was not made")
    if row["done"] and (row["energy"] is None or row["error"] is None):
        reasons.append("no DMC energy")
    return reasons


def not_comparable(rows: Sequence[Optional[dict]]) -> Optional[str]:
    """None if a difference of the DMC energies of these rows may be printed. Else the reason in plain words."""
    if any(r is None for r in rows):
        return "a geometry has no row for this cutoff"
    reasons: List[str] = []
    first = rows[0]
    for r in rows[1:]:
        for name in ("cutoff", "n_det_max"):
            if used(r, name) != used(first, name):
                reasons.append("%s differs (%s at %s %s, %s at %s %s)" % (
                    name, used(first, name), first["target"], first["geometry"], used(r, name), r["target"], r["geometry"]))
    for r in rows:
        reasons += ["%s %s: %s" % (r["target"], r["geometry"], why) for why in row_reasons(r)]
    return "; ".join(reasons) or None


def combination(terms: Sequence[Tuple[Optional[dict], float]]) -> dict:
    """The energy sum ``sum(c * E(row))`` in kcal/mol, or the reason why it is not printed.

    Returns ``{"value", "error", "text"}``; ``text`` is None for a printed value. A row that appears twice (the
    final row of a target that did not move is its start row) gets the sum of its coefficients; a coefficient
    of 0 drops the row from the value but not from the rule: every row that the difference names must be
    complete. The error is ``sqrt(sum(c**2 * error(row)**2))``: the rows are independent runs. When every
    coefficient cancels, the value is 0 and exact."""
    if any(row is None for row, _c in terms):
        return {"value": None, "error": None, "text": NOT_COMPARABLE + "a geometry has no row for this cutoff"}
    rows: List[dict] = []
    coef: Dict[int, float] = {}
    for row, c in terms:
        if id(row) not in coef:
            rows.append(row)
            coef[id(row)] = 0.0
        coef[id(row)] += float(c)
    why = not_comparable(rows)
    if why:
        return {"value": None, "error": None, "text": NOT_COMPARABLE + why}
    kept = [r for r in rows if coef[id(r)] != 0.0]
    if not kept:
        return {"value": 0.0, "error": 0.0, "text": None}
    value = sum(coef[id(r)] * r["energy"] for r in kept) * HARTREE_TO_KCAL
    error = math.sqrt(sum((coef[id(r)] * r["error"]) ** 2 for r in kept)) * HARTREE_TO_KCAL
    return {"value": float(value), "error": float(error), "text": None}


# ============================================================================= step 1
def numeric_cutoffs(rows: Sequence[dict]) -> List[dict]:
    """The numeric cutoffs of the rows, in the order of the rows: ``[{"label", "cutoff", "n_det_max"}]``."""
    out: List[dict] = []
    seen = set()
    for r in rows:
        if r["cutoff"] != HF_LABEL and r["label"] not in seen:
            seen.add(r["label"])
            out.append({"label": r["label"], "cutoff": float(r["cutoff"]), "n_det_max": r["n_det_max"]})
    return out


def against_hf(rows: Sequence[dict]) -> dict:
    """Step 1: each cut against the single RHF determinant of the same geometry.

    Returns ``{"tested", "lines", "usability"}``. ``tested`` is False when the rows hold no numeric cutoff or
    no row ``hf``; then ``lines`` is empty and every cutoff is "not tested". ``lines`` has one entry per
    geometry and numeric cutoff, in the order of the rows:

        target, geometry, label, cutoff, n_det_max    which cut at which geometry
        n_det, n_det_parent                           determinants of the cut and of its CIPSI wave function, or None
        delta, error                                  E(cut) - E(hf) and its standard error in millihartree, or None
        state                                         "above", "not above" or "not available"
        sure                                          for "above": True if delta > ABOVE_SURE_SIGMAS * error
        reason                                        for "not available": what is not complete

    ``usability`` has one entry per numeric cutoff: its ``label``, ``cutoff``, ``n_det_max`` and ``status``
    (USABLE, NOT USABLE, "no verdict yet" or "not tested")."""
    index = {(r["target"], r["geometry"], r["label"]): r for r in rows}
    places = list(dict.fromkeys((r["target"], r["geometry"]) for r in rows))
    numeric = numeric_cutoffs(rows)
    tested = bool(numeric) and any(r["cutoff"] == HF_LABEL for r in rows)
    lines: List[dict] = []
    for target, geometry in (places if tested else []):
        hf = index.get((target, geometry, HF_LABEL))
        if hf is None:
            continue
        for cut in numeric:
            row = index.get((target, geometry, cut["label"]))
            if row is None:
                continue
            line = {"target": target, "geometry": geometry, "label": cut["label"], "cutoff": cut["cutoff"],
                    "n_det_max": cut["n_det_max"], "n_det": row["trial"].get("n_det"),
                    "n_det_parent": row["trial"].get("n_det_parent"), "delta": None, "error": None,
                    "state": NOT_AVAILABLE, "sure": None, "reason": None}
            why = row_reasons(row) + ["the row hf: %s" % w for w in row_reasons(hf)]
            if why:
                line["reason"] = "; ".join(why)
            else:
                delta = 1000.0 * (float(row["energy"]) - float(hf["energy"]))
                error = 1000.0 * math.hypot(float(row["error"]), float(hf["error"]))
                above = delta > ABOVE_SIGMAS * error
                line.update(delta=delta, error=error, state=ABOVE if above else NOT_ABOVE,
                            sure=bool(delta > ABOVE_SURE_SIGMAS * error) if above else None)
            lines.append(line)
    usability: List[dict] = []
    for cut in numeric:
        mine = [line for line in lines if line["label"] == cut["label"]]
        if not tested:
            status = NOT_TESTED
        elif any(line["state"] == ABOVE for line in mine):
            status = NOT_USABLE
        elif not mine or any(line["state"] == NOT_AVAILABLE for line in mine):
            status = NO_VERDICT_YET
        else:
            status = USABLE
        usability.append(dict(cut, status=status))
    return {"tested": tested, "lines": lines, "usability": usability}


# ============================================================================= step 3
def difference_specs(same: Dict[str, bool], kinds: Optional[Dict[str, str]] = None) -> List[dict]:
    """The energy differences that the rule tests, in the order of the report.

    ``same``: ``{target: True if the search did not move it}``, in the order of the targets. ``kinds``:
    ``{target: "minimum" | "saddle" | ...}``; without kinds no barrier exists. The list holds (a) final minus
    start of every target that the search moved, then for every pair of a saddle target and a minimum target
    (b) the barrier at the start geometries and (c) the barrier at the final geometries. The change of a
    barrier (final minus start) is not in the list: it is the difference of (b) and (c).

    Each entry: ``name``, ``kind`` ("change" or "barrier"), ``target`` (for a barrier: the saddle),
    ``minus_target`` (the minimum, or None), ``geometry`` ("final minus start", "start" or "final") and
    ``terms``, a list of ``((target, geometry of the row), coefficient)``."""
    kinds = kinds or {}
    final = {name: ("start" if did_not_move else "final") for name, did_not_move in same.items()}
    out: List[dict] = []
    for name, did_not_move in same.items():
        if not did_not_move:
            out.append({"name": "final minus start of %s" % name, "kind": "change", "target": name,
                        "minus_target": None, "geometry": "final minus start",
                        "terms": [((name, "final"), 1.0), ((name, "start"), -1.0)]})
    for saddle in [n for n in same if kinds.get(n) == "saddle"]:
        for minimum in [n for n in same if kinds.get(n) == "minimum"]:
            for geometry, rows in (("start", ("start", "start")), ("final", (final[saddle], final[minimum]))):
                out.append({"name": "barrier %s minus %s at the %s geometries" % (saddle, minimum, geometry),
                            "kind": "barrier", "target": saddle, "minus_target": minimum, "geometry": geometry,
                            "terms": [((saddle, rows[0]), 1.0), ((minimum, rows[1]), -1.0)]})
    return out


def difference_verdict(coarse: dict, fine: dict, tolerance: float) -> dict:
    """Step 3 for one energy difference. ``coarse`` and ``fine`` are results of :func:`combination` for the
    two cutoffs of the pair. Returns ``{"S", "e", "bound", "verdict", "reason"}`` in kcal/mol. A difference that
    is NOT COMPARABLE at one of the two cutoffs has no verdict (None), and ``reason`` says why."""
    why = ["at the %s cutoff: %s" % (side, str(d["text"])[len(NOT_COMPARABLE):] if str(d["text"]).startswith(NOT_COMPARABLE)
                                     else d["text"]) for side, d in (("coarse", coarse), ("fine", fine)) if d["text"]]
    if why:
        return {"S": None, "e": None, "bound": None, "verdict": None, "reason": "; ".join(why)}
    change = float(fine["value"]) - float(coarse["value"])
    error = math.hypot(float(fine["error"]), float(coarse["error"]))
    bound = abs(change) + VERDICT_SIGMAS * error
    if bound <= tolerance:
        verdict = CONVERGED
    elif abs(change) - VERDICT_SIGMAS * error > tolerance:
        verdict = NOT_CONVERGED
    else:
        verdict = NOT_DECIDED
    return {"S": change, "e": error, "bound": bound, "verdict": verdict, "reason": None}


# ============================================================================= steps 1 to 4 together
def evaluate(rows: Sequence[dict], same: Dict[str, bool], kinds: Optional[Dict[str, str]] = None,
             tolerance: float = DEFAULT_TOLERANCE, text: Callable[[object], str] = cut_text) -> dict:
    """Steps 1 to 4 on the rows of ``trial.results``.

    ``rows``: one entry per target, geometry and cutoff (see ``trial.results``). ``same`` and ``kinds``: see
    :func:`difference_specs`. ``tolerance`` in kcal/mol. ``text`` writes a cutoff in the sentence ``reason``.

    Returns a dictionary that ``json`` can write:

        tolerance            the tolerance that was used
        hf                   True if the cuts were tested against a row hf
        against_hf           step 1: one line per geometry and numeric cutoff (the ``lines`` of :func:`against_hf`)
        usability            step 1: one entry per numeric cutoff with its ``status``
        incomplete           the rows that are not complete: target, geometry, label, cutoff, reasons
        names                the names of all differences that the rule tests for this project
        pair, pair_labels    [coarse cutoff, fine cutoff] and their labels, or None when no pair exists
        differences          step 3: one entry per tested difference with name, kind, target, minus_target,
                             geometry, coarse, coarse_error, fine, fine_error, S, e, bound, verdict, reason
        verdict              step 4: a verdict word, or None when the rows hold fewer than two numeric cutoffs
        reason               one sentence that says why the project has this verdict

    The verdict of the project: ``NO_DIFFERENCE`` when the project prints no energy difference; else
    ``NO_VERDICT`` when a row is not complete or a tested difference cannot be compared; else
    ``NOT_CONVERGED`` when fewer than two cutoffs are usable or a difference is NOT_CONVERGED; else
    ``NOT_DECIDED`` when a difference is NOT_DECIDED; else ``CONVERGED``."""
    step1 = against_hf(rows)
    specs = difference_specs(same, kinds)
    out = {"tolerance": float(tolerance), "hf": step1["tested"], "against_hf": step1["lines"],
           "usability": step1["usability"],
           "incomplete": [{"target": r["target"], "geometry": r["geometry"], "label": r["label"], "cutoff": r["cutoff"],
                           "reasons": row_reasons(r)} for r in rows if row_reasons(r)],
           "names": [s["name"] for s in specs], "pair": None, "pair_labels": None, "differences": [],
           "verdict": None, "reason": ""}
    if len(step1["usability"]) < 2:
        out["reason"] = ("the stage holds fewer than two numeric cutoffs, so the convergence with the size of the "
                         "expansion cannot be tested")
        return out
    if not specs:
        out["verdict"] = NO_DIFFERENCE
    elif out["incomplete"]:
        out["verdict"] = NO_VERDICT
    else:
        candidates = sorted((c for c in step1["usability"] if c["status"] in (USABLE, NOT_TESTED)),
                            key=lambda c: c["cutoff"])
        if len(candidates) < 2:
            out["verdict"] = NOT_CONVERGED
        else:
            fine, coarse = candidates[0], candidates[1]
            out["pair"], out["pair_labels"] = [coarse["cutoff"], fine["cutoff"]], [coarse["label"], fine["label"]]
            index = {(r["target"], r["geometry"], r["label"]): r for r in rows}
            for spec in specs:
                at = {side: combination([(index.get((target, geometry, cut["label"])), c)
                                         for (target, geometry), c in spec["terms"]])
                      for side, cut in (("coarse", coarse), ("fine", fine))}
                entry = {k: v for k, v in spec.items() if k != "terms"}
                entry.update(coarse=at["coarse"]["value"], coarse_error=at["coarse"]["error"],
                             fine=at["fine"]["value"], fine_error=at["fine"]["error"])
                entry.update(difference_verdict(at["coarse"], at["fine"], float(tolerance)))
                out["differences"].append(entry)
            verdicts = [d["verdict"] for d in out["differences"]]
            if None in verdicts:
                out["verdict"] = NO_VERDICT
            elif NOT_CONVERGED in verdicts:
                out["verdict"] = NOT_CONVERGED
            elif NOT_DECIDED in verdicts:
                out["verdict"] = NOT_DECIDED
            else:
                out["verdict"] = CONVERGED
    out["reason"] = reason_text(out, text)
    return out


def reason_text(ev: dict, text: Callable[[object], str] = cut_text) -> str:
    """One sentence (no full stop at its end) that says why an evaluation, or a decision of the record, has its
    verdict. ``text`` writes a cutoff."""
    verdict = ev["verdict"]
    tolerance = number_text(ev["tolerance"])
    if verdict == NO_DIFFERENCE:
        return ("the project prints no energy difference (the search moved no target, and there is no pair of a "
                "saddle target and a minimum target), so nothing can be tested")
    if verdict == NO_VERDICT and ev["incomplete"]:
        return "%d row(s) of the stage are not complete: %s" % (len(ev["incomplete"]), "; ".join(
            "%s %s, cutoff %s: %s" % (i["target"], i["geometry"], text(i["cutoff"]), "; ".join(i["reasons"]))
            for i in ev["incomplete"]))
    if verdict == NO_VERDICT:
        return "; ".join("%s cannot be compared (%s)" % (d["name"], d["reason"])
                         for d in ev["differences"] if d["verdict"] is None)
    if verdict == NOT_CONVERGED and not ev["pair"]:
        above = ["the cutoff %s is above the single RHF determinant at %s %s by %s millihartree" % (
            text(line["cutoff"]), line["target"], line["geometry"], pm_text(line["delta"], line["error"]))
            for line in ev["against_hf"] if line["state"] == ABOVE]
        return "fewer than two cutoffs are usable: %s" % "; ".join(above)
    if not ev["pair"]:
        return str(ev.get("reason") or "")
    coarse, fine = text(ev["pair"][0]), text(ev["pair"][1])
    mine = [d for d in ev["differences"] if d["verdict"] == verdict]
    if verdict == CONVERGED:
        return ("all %d tested energy difference(s) changed by at most %s kcal/mol between the cutoffs %s and %s, with "
                "%g standard errors of margin (largest bound %s kcal/mol)"
                % (len(mine), tolerance, coarse, fine, VERDICT_SIGMAS, bound_text(max(mine, key=lambda d: d["bound"]))))
    if verdict == NOT_CONVERGED:
        return "; ".join("%s changed by %s kcal/mol between the cutoffs %s and %s, more than the tolerance of %s kcal/mol "
                         "with %g standard errors of margin" % (d["name"], pm_text(d["S"], d["e"]), coarse, fine,
                                                                 tolerance, VERDICT_SIGMAS) for d in mine)
    return "; ".join("%s changed by %s kcal/mol between the cutoffs %s and %s (bound %s kcal/mol): the errors are too "
                     "large to say whether the change is within the tolerance of %s kcal/mol"
                     % (d["name"], pm_text(d["S"], d["e"]), coarse, fine, bound_text(d), tolerance) for d in mine)


# ============================================================================= the automatic mode
def above_to_test_again(ev: dict) -> List[float]:
    """The cutoffs whose result "above the single determinant" is tested again with one more DMC series.

    A cutoff is in the list when all three conditions hold:

    (a) It is above at one geometry or more, and at none of them by more than ``ABOVE_SURE_SIGMAS`` standard
        errors. The stage takes a result above by more than that as final: it adds no series to test it again,
        so the cutoff stays NOT USABLE, whatever the results at the other geometries are.
    (b) Fewer than two usable cutoffs have a smaller value. The pair of the rule is the two usable cutoffs with
        the smallest values, so a cutoff that has two usable cutoffs with smaller values would not be in the pair
        even if it were usable.
    (c) The usable cutoffs and the cutoffs that meet (a) and (b) are at least two together. With fewer, no pair
        exists even if no such result is above after more data, so more data cannot change the verdict.

    ``ev`` is the result of :func:`evaluate`. The order is that of ``ev["usability"]``."""
    usable = [float(c["cutoff"]) for c in ev["usability"] if c["status"] in (USABLE, NOT_TESTED)]
    out: List[float] = []
    for cut in ev["usability"]:
        above = [line for line in ev["against_hf"] if line["label"] == cut["label"] and line["state"] == ABOVE]
        if not above or any(line["sure"] for line in above):
            continue
        if sum(1 for value in usable if value < float(cut["cutoff"])) < 2:
            out.append(float(cut["cutoff"]))
    if len(usable) + len(out) < 2:
        return []
    return out


def errors_allow_convergence(ev: dict) -> bool:
    """True if the next pair of cutoffs (the fine cutoff and a smaller one) can give the verdict CONVERGED with the
    series that are allowed, judged by the errors alone.

    The next pair compares the fine cutoff with a new one, and the new rows get the same number of DMC series. So
    the error of a difference for that pair is estimated as ``sqrt(2) * fine_error``: the new rows are assumed to
    have the error of the fine rows. The estimate is not a measurement. The condition is
    ``VERDICT_SIGMAS * sqrt(2) * fine_error <= tolerance`` for every tested difference that has a verdict. With a
    larger estimate the bound ``|S| + VERDICT_SIGMAS * e`` of the next pair is above the tolerance whatever its
    change ``S`` is: only more DMC series can help. Returns True when no difference has a verdict."""
    return not estimate_failures(ev)


def estimate_failures(ev: dict) -> List[Tuple[str, float]]:
    """The tested differences that fail the test of :func:`errors_allow_convergence`: ``(name, estimate)`` with
    ``estimate = VERDICT_SIGMAS * sqrt(2) * fine_error`` in kcal/mol, in the order of the differences."""
    tolerance = float(ev["tolerance"])
    out: List[Tuple[str, float]] = []
    for d in ev["differences"]:
        if d["verdict"] is None:
            continue
        estimate = VERDICT_SIGMAS * math.sqrt(2.0) * float(d["fine_error"])
        if estimate > tolerance:
            out.append((str(d["name"]), estimate))
    return out


def decide(ev: dict, series: int, series_max: int, cutoffs: Sequence[float], ladder: Sequence[float]) -> dict:
    """What the stage does next in the automatic mode, when every DMC run of its plan is DONE.

    ``ev``: the result of :func:`evaluate` for the rows of the plan. ``series``: DMC series of every row now.
    ``cutoffs``: the numeric cutoffs of the plan now; they are the first entries of ``ladder``.

    Returns ``{"action", "cause", "verdict", "added"}``. The branches, in this order:

        a row is not complete (NO_VERDICT)                      final, verdict NO_VERDICT
        CONVERGED or NO_DIFFERENCE                              final, with that verdict
        a cutoff of :func:`above_to_test_again` exists          one more series for every row
          (above by at most ABOVE_SURE_SIGMAS standard            (added = the new number of series)
          errors, and more data can change the pair),
          and series < series_max
        (CONVERGED comes first: the stage ends with it also when a smaller cutoff of the plan is above by at
        most ABOVE_SURE_SIGMAS standard errors, because the two smallest USABLE cutoffs are converged.)
        NOT_CONVERGED (a difference, or fewer than two          the next cutoff of the ladder (added = the cutoff);
          usable cutoffs)                                         without one: final, verdict NOT_CONVERGED
        NOT_DECIDED                                             one more series while series < series_max; then
                                                                  final with the cause "errors above the
                                                                  tolerance" unless :func:`errors_allow_convergence`;
                                                                  else the next cutoff of the ladder (added = the
                                                                  cutoff); without one: final, verdict NOT_DECIDED

    So every decision ends the stage, adds a series (at most ``series_max``) or adds a cutoff (at most the
    ladder): the stage cannot go on for ever."""
    verdict = ev["verdict"]
    if verdict not in VERDICTS:
        raise ValueError("the rows have no verdict (%s): the automatic mode needs at least two numeric cutoffs"
                         % (ev.get("reason") or verdict))

    def step(action: str, cause: str, final: Optional[str] = None, added=None) -> dict:
        return {"action": action, "cause": cause, "verdict": final, "added": added}

    if verdict == NO_VERDICT:
        return step(FINAL, "no verdict", NO_VERDICT)
    if verdict == CONVERGED:
        return step(FINAL, "converged", CONVERGED)
    if verdict == NO_DIFFERENCE:
        return step(FINAL, "no difference", NO_DIFFERENCE)
    more = int(series) < int(series_max)
    if more and above_to_test_again(ev):
        return step(MORE_SERIES, "above not sure", added=int(series) + 1)
    if verdict == NOT_CONVERGED:
        if len(cutoffs) < len(ladder):
            return step(NEXT_CUTOFF, "not converged", added=float(ladder[len(cutoffs)]))
        return step(FINAL, "ladder ended", NOT_CONVERGED)
    if more:
        return step(MORE_SERIES, "not decided", added=int(series) + 1)
    # No further series is allowed. A smaller cutoff is the step that is left, and it can decide only when the
    # errors are small enough (see errors_allow_convergence). The errors are tested first, so that the message of
    # the end says what can help: more series, or a smaller cutoff.
    if not errors_allow_convergence(ev):
        return step(FINAL, "errors above the tolerance", NOT_DECIDED)
    if len(cutoffs) < len(ladder):
        return step(NEXT_CUTOFF, "not decided at series_max", added=float(ladder[len(cutoffs)]))
    return step(FINAL, "series_max reached", NOT_DECIDED)


def headline(verdict: str, pair: Optional[Sequence[float]], tolerance: float, series: Optional[int] = None,
             text: Callable[[object], str] = cut_text) -> str:
    """The verdict of the project with its two cutoffs, for example
    ``CONVERGED within 1.0 kcal/mol between the cutoffs 2.5e-4 and 1e-4 (3 series)``."""
    tail = "" if series is None else " (%d series)" % int(series)
    if pair and verdict == CONVERGED:
        return "%s within %s kcal/mol between the cutoffs %s and %s%s" % (
            verdict, number_text(tolerance), text(pair[0]), text(pair[1]), tail)
    if pair and verdict in (NOT_CONVERGED, NOT_DECIDED):
        return "%s between the cutoffs %s and %s with the tolerance %s kcal/mol%s" % (
            verdict, text(pair[0]), text(pair[1]), number_text(tolerance), tail)
    return "%s%s" % (verdict, tail)


def final_tail(cause: str, series_max: int, series: int, failures: Sequence[Tuple[str, float]] = ()) -> str:
    """The words that follow the reason of a final verdict: why the stage could take no further step, and what
    can help. ``series``: the DMC series of every row of the plan. ``failures``: the result of
    :func:`estimate_failures` (named in the words of the cause "errors above the tolerance"). Empty for a cause that
    needs none."""
    limit = "no further series is allowed ([trial] series_max = %d)" % int(series_max)
    smaller = "to continue, add a smaller cutoff at the end of [trial] auto_cutoffs"
    named = (" for %s" % ", ".join("%s (%.2f kcal/mol)" % (name, value) for name, value in failures)) if failures else ""
    return {"ladder ended": "; the ladder holds no further cutoff: %s" % smaller,
            "series_max reached": "; %s, and the ladder holds no further cutoff: %s" % (limit, smaller),
            "errors above the tolerance": ("; %s, and the errors are too large for the verdict CONVERGED also with a "
                                           "smaller cutoff (%g x sqrt(2) x the error at the fine cutoff is larger than "
                                           "the tolerance%s): to continue, raise [trial] series_max above %d"
                                           % (limit, VERDICT_SIGMAS, named, int(series)))}.get(cause, "")


def final_text(entry: dict, text: Callable[[object], str] = cut_text) -> str:
    """The final verdict of the automatic mode in one sentence (no full stop at its end), from the last decision
    of the record. CONVERGED needs no reason: the headline says everything."""
    head = headline(entry["verdict"], entry["pair"], entry["tolerance"], entry["series"], text)
    if entry["verdict"] == CONVERGED:
        return head
    return "%s: %s%s" % (head, reason_text(entry, text), final_tail(entry["cause"], entry["series_max"], entry["series"],
                                                                    estimate_failures(entry)))


def measured_text(entry: dict, text: Callable[[object], str] = cut_text) -> str:
    """What a decision measured, in sentences: the rows that are not complete, the cuts above the single
    determinant, and the change of every tested difference."""
    out: List[str] = []
    if entry["incomplete"]:
        out.append("Not complete: %s." % "; ".join(
            "%s %s, cutoff %s: %s" % (i["target"], i["geometry"], text(i["cutoff"]), "; ".join(i["reasons"]))
            for i in entry["incomplete"]))
    above = [line for line in entry["against_hf"] if line["state"] == ABOVE]
    if above:
        out.append("Above the single RHF determinant: %s." % "; ".join(
            "the cutoff %s at %s %s by %s millihartree%s" % (
                text(line["cutoff"]), line["target"], line["geometry"], pm_text(line["delta"], line["error"]),
                "" if line["sure"] else " (not more than %g standard errors)" % ABOVE_SURE_SIGMAS) for line in above))
    elif entry["hf"] and not any(line["state"] == NOT_AVAILABLE for line in entry["against_hf"]):
        out.append("No cut is above the single RHF determinant.")
    if entry["pair"]:
        parts = []
        for d in entry["differences"]:
            if d["verdict"] is None:
                parts.append("%s cannot be compared (%s)" % (d["name"], d["reason"]))
            else:
                parts.append("%s changed by %s kcal/mol (bound %s kcal/mol, %s)" % (
                    d["name"], pm_text(d["S"], d["e"]), bound_text(d), d["verdict"]))
        out.append("Between the cutoffs %s and %s: %s." % (text(entry["pair"][0]), text(entry["pair"][1]), "; ".join(parts)))
    return " ".join(out)


def action_text(entry: dict, text: Callable[[object], str] = cut_text) -> str:
    """What a decision did and why, in one sentence (no full stop at its end)."""
    if entry["action"] == MORE_SERIES:
        why = ("a cut is above the single RHF determinant by at most %g standard errors, and more data test this result "
               "again" % ABOVE_SURE_SIGMAS) if entry["cause"] == "above not sure" else "the errors are too large to decide"
        return "one more DMC series is added for every row (%d series now, at most %d), because %s" % (
            int(entry["added"]), int(entry["series_max"]), why)
    if entry["action"] == NEXT_CUTOFF:
        if entry["cause"] == "not decided at series_max":
            why = ("an energy difference is NOT_DECIDED, no further series is allowed ([trial] series_max = %d), and "
                   "the errors are small enough for a verdict with a smaller cutoff" % int(entry["series_max"]))
        else:
            why = "an energy difference is NOT_CONVERGED" if entry["pair"] else "fewer than two cutoffs are usable"
        return "the next cutoff of the ladder, %s, is added with %d DMC series, because %s" % (
            text(entry["added"]), int(entry["series"]), why)
    tail = final_tail(entry["cause"], entry["series_max"], entry["series"], estimate_failures(entry))
    return "the stage ends with the verdict %s%s" % (entry["verdict"], " (%s)" % tail[2:] if tail else "")


def decision_text(entry: dict, text: Callable[[object], str] = cut_text) -> str:
    """One decision of the automatic mode in plain sentences: the plan and the tolerance at that time, what was
    measured, the verdict, and what was added and why."""
    measured = measured_text(entry, text)
    return "cutoffs %s and hf with %d DMC series; tolerance %s kcal/mol. %sVerdict of this decision: %s. Action: %s." % (
        ", ".join(text(c) for c in entry["cutoffs"]), int(entry["series"]), number_text(entry["tolerance"]),
        measured + " " if measured else "", entry["verdict"], action_text(entry, text))
