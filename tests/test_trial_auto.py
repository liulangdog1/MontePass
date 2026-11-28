"""The automatic cutoff mode of the stage ``trial`` (``[trial] cutoff = auto``) and the rule behind it.

Nothing here needs a cluster program. The engine is the stand-in of ``test_trial_stage.py`` with one change:
the block energies of a run are its mean plus and minus a fixed amplitude, in turn. So the mean of every run is
the number that a test gives, the error of every run follows from the amplitude, and no result depends on
random numbers. A test sets the energy of every row itself, so every branch of the decision is reached on
purpose.

    TestInput          the keys of the input section: cutoff = auto, auto_cutoffs, tolerance, series, series_max
    TestRule           the pure functions of ``montepass/trial_rule.py`` with numbers
    TestDecisions      every branch of the decision, with the stage and the stand-in engine
    TestRecord         the growth of the plan, the record, a second object, a changed input, a switch of the mode
    TestManualMode     the manual mode is what it was
    TestPlanAndProfile the lines of the command ``check`` and the test of the machine profile
    TestWorkflowAuto   the stage in the workflow, the commands ``check`` and ``status``, a record of version 0.1.0
    TestRealEngineAuto the first plan and a grown plan on the real engine with a fake batch system
    TestReportAuto     the three report sections and the CSV lines
"""
import contextlib
import csv
import io
import json
import math
import os
import re
import shutil
import sys
import tempfile
import time
import unittest

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
sys.path.insert(0, HERE)

import test_trial_stage as TS  # noqa: E402  (the stand-in engine, the input texts and the helpers of the stage tests)
from montepass import cli, geometry as G, model as M, trial as T, trial_rule as R  # noqa: E402
from montepass.config import AUTO_N_DET_MAX, Config, TEMPLATE  # noqa: E402
from montepass.engine import multidet as MD  # noqa: E402
from montepass.engine.base import DONE, FAILED, QUEUED, RUNNING, STOPPED  # noqa: E402
from montepass.project import Project  # noqa: E402
from montepass.report import build_report, tables, trial as RT  # noqa: E402
from montepass.report.demo import make_demo_project  # noqa: E402
from montepass.units import HARTREE_TO_KCAL as H  # noqa: E402
from montepass.workflow import ModelBackend, Workflow, make_backend  # noqa: E402

LADDER = [2.5e-4, 1e-4]
AUTO = dict(TS.TRIAL_DICT, n_det_max=100000, cutoffs=LADDER + ["hf"], not_converged="stop", tolerance=1.0,
            series_max=4, auto={"ladder": list(LADDER)})
DEFAULT_LADDER = [2.5e-4, 1e-4, 5e-5]       # the default of [trial] auto_cutoffs
AUTO_DEFAULT = dict(AUTO, auto={"ladder": list(DEFAULT_LADDER)})    # Config.trial of "cutoff = auto" with every default
AUTO_KEYS = ["cutoffs", "decisions", "final", "ladder", "reason", "series", "series_max", "tolerance", "verdict"]
E_HF = -17.2000                     # hartree: the energy of the rows hf in most tests
E_CUT = -17.2050                    # hartree: the energy of the cut rows; 5 millihartree below hf, so no cut is above
COARSE, FINE, THIRD = "n100000_c2.5e-04", "n100000_c1e-04", "n100000_c5e-05"


class Exact(TS.StandIn):
    """The stand-in engine with block energies that a test controls completely.

    The blocks of a run are ``mean + noise`` and ``mean - noise`` in turn. The rule for the start-up blocks drops
    the first 10 blocks of such a trace, and an even number of blocks is left. So the energy of the run is
    ``mean`` (within rounding), and its error is ``noise / sqrt(blocks - 11)``."""

    def finish(self, key, mean=-17.2, noise=1.0e-3, **kw):
        super().finish(key, mean=mean, noise=noise, **kw)
        n = self.specs[key].blocks
        self.points[key]["energy"] = mean + noise * np.where(np.arange(n) % 2 == 0, 1.0, -1.0)


def split(key):
    """``(target, geometry, label, series)`` of a key of the stage."""
    target, geometry, leaf = key.split("/")[1:4]
    match = re.fullmatch(r"(.+)_s(\d+)", leaf)
    return (target, geometry, match.group(1), int(match.group(2))) if match else (target, geometry, leaf, 1)


def auto_input(extra="", n_det_max=""):
    """A model input whose section [trial] asks for the automatic mode."""
    return TS.model_input("kind = multidet\ncutoff = auto\n%s%s" % ("n_det_max = %s\n" % n_det_max if n_det_max else "", extra))


def rule_row(target, geometry, cutoff, energy, error, n_det_max=100000, states=("DONE", "DONE"), check="PASS", facts=None):
    """One row as ``trial.results`` gives it, with the values that the rule reads."""
    hf = cutoff == "hf"
    checks = {} if check is None else {"mdvmc": {"ok": check == "PASS", "verdict": check}}
    return {"target": target, "geometry": geometry, "label": T.entry_label(cutoff, n_det_max), "cutoff": cutoff,
            "n_det_max": None if hf else n_det_max, "states": list(states), "done": all(s == "DONE" for s in states),
            "energy": energy, "error": error, "trial": dict(facts or {}), "checks": checks}


# ===================================================================================== the input section
class TestInput(unittest.TestCase):
    def test_cutoff_auto_with_the_defaults(self):
        for word in ("auto", "AUTO", " Auto "):
            cfg = TS.load(TS.model_input("kind = multidet\ncutoff = %s\n" % word))
            self.assertEqual(cfg.trial, {
                "kind": "multidet", "n_det_max": 100000, "cutoffs": [2.5e-4, 1e-4, "hf"], "series": 2,
                "qp2_basis": "cc-pvtz_ecp_ccecp", "qp2_pseudo": "ccecp", "not_converged": "stop",
                "tolerance": 1.0, "series_max": 4, "auto": {"ladder": [2.5e-4, 1e-4, 5e-5]}}, word)
        self.assertEqual(AUTO_N_DET_MAX, 100000)
        self.assertEqual(cfg.summary()["trial"], cfg.trial)
        self.assertEqual(cfg.trial, dict(AUTO_DEFAULT))
        # the dictionary that most other tests use is the same with a ladder of 2 cutoffs
        self.assertEqual(TS.load(TS.model_input("kind = multidet\ncutoff = auto\nauto_cutoffs = 2.5e-4, 1e-4\n")).trial,
                         dict(AUTO))
        # the engine accepts every cutoff that the stage can ask for
        for cutoff in cfg.trial["auto"]["ladder"] + ["hf"]:
            MD.validate_trial(MD.trial_of({"trial": {"kind": "multidet", "cutoff": cutoff, "n_det_max": 100000}}))

    def test_every_key_of_the_automatic_mode(self):
        cfg = TS.load(auto_input("auto_cutoffs = 1e-3, 5e-4, 1e-4\ntolerance = 0.5\nseries = 3\nseries_max = 6\n", 50000))
        t = cfg.trial
        self.assertEqual(t["auto"], {"ladder": [1e-3, 5e-4, 1e-4]})
        self.assertEqual(t["cutoffs"], [1e-3, 5e-4, "hf"])              # the plan of the first pass: two cutoffs, then hf
        self.assertEqual((t["n_det_max"], t["tolerance"], t["series"], t["series_max"]), (50000, 0.5, 3, 6))
        self.assertIsInstance(t["tolerance"], float)
        self.assertIsInstance(t["series_max"], int)
        # series_max may equal series: then the stage never adds a series
        self.assertEqual(TS.load(auto_input("series = 3\nseries_max = 3\n")).trial["series_max"], 3)
        # a ladder of exactly 2 cutoffs is the smallest one
        self.assertEqual(TS.load(auto_input("auto_cutoffs = 5e-4, 4e-4\n")).trial["cutoffs"], [5e-4, 4e-4, "hf"])

    def test_the_manual_mode_reads_tolerance_and_not_the_keys_of_the_automatic_mode(self):
        base = "kind = multidet\nn_det_max = 10000\ncutoff = 1e-3, hf\n"
        cfg = TS.load(TS.model_input(base + "tolerance = 0.25\nseries = 3\n"))
        self.assertEqual((cfg.trial["tolerance"], cfg.trial["series"], cfg.trial["series_max"], cfg.trial["auto"]),
                         (0.25, 3, 3, None))
        # auto_cutoffs and series_max are read only with cutoff = auto: whatever they hold is no problem here
        cfg = TS.load(TS.model_input(base + "auto_cutoffs = x, hf, 7\nseries_max = 0\n"))
        self.assertEqual((cfg.trial["cutoffs"], cfg.trial["series_max"], cfg.trial["auto"]), ([1e-3, "hf"], 2, None))
        # the manual mode still needs n_det_max for a numeric cutoff: only the automatic mode has a default
        self.assertIn("[trial] n_det_max is needed for a numeric cutoff",
                      TS.problems(TS.model_input("kind = multidet\ncutoff = 1e-3\n")))
        # kind = single: no key of the section is read
        single = TS.load(TS.model_input("kind = single\ncutoff = auto, 3\nauto_cutoffs = hf\ntolerance = -1\nseries_max = x\n"))
        self.assertIsNone(single.trial)

    def test_every_problem_line(self):
        cases = [
            # the word auto must stand alone
            ("cutoff = auto, hf\n", "[trial] cutoff = auto, hf: the word auto must stand alone"),
            ("cutoff = 1e-3, auto\nn_det_max = 100\n", "[trial] cutoff = 1e-3, auto: the word auto must stand alone"),
            ("cutoff = auto, auto\n", "[trial] cutoff = auto, auto: the word auto must stand alone"),
            # the ladder
            ("cutoff = auto\nauto_cutoffs =\n",
             "[trial] auto_cutoffs = (empty) holds 0 cutoff(s); the automatic mode needs at least 2"),
            ("cutoff = auto\nauto_cutoffs = 1e-4\n",
             "[trial] auto_cutoffs = 1e-4 holds 1 cutoff(s); the automatic mode needs at least 2"),
            ("cutoff = auto\nauto_cutoffs = 1e-4, 2.5e-4\n",
             "[trial] auto_cutoffs = 1e-4, 2.5e-4 must be strictly decreasing: 0.00025 comes after 0.0001"),
            ("cutoff = auto\nauto_cutoffs = 1e-3, 1e-4, 5e-4\n", "must be strictly decreasing: 0.0005 comes after 0.0001"),
            ("cutoff = auto\nauto_cutoffs = 2.5e-4, 2.5e-4\n", "[trial] auto_cutoffs lists the same cutoff twice: 2.5e-4 and 2.5e-4"),
            ("cutoff = auto\nauto_cutoffs = 1.0000001e-4, 1e-4\n", "[trial] auto_cutoffs lists the same cutoff twice"),
            ("cutoff = auto\nauto_cutoffs = 2.5e-4, hf\n", "[trial] auto_cutoffs = hf: the word hf is not allowed here"),
            ("cutoff = auto\nauto_cutoffs = 2.5e-4, small\n", "[trial] auto_cutoffs = 'small' is not a number"),
            ("cutoff = auto\nauto_cutoffs = 2.5e-4, 0\n", "[trial] auto_cutoffs = 0 must be between 0 and 1"),
            ("cutoff = auto\nauto_cutoffs = 1.5, 1e-4\n", "[trial] auto_cutoffs = 1.5 must be between 0 and 1"),
            ("cutoff = auto\nauto_cutoffs = 2.5e-4, -1e-4\n", "[trial] auto_cutoffs = -1e-4 must be between 0 and 1"),
            # the numbers of series
            ("cutoff = auto\nseries = 5\n", "[trial] series_max = 4 must be at least series = 5"),
            ("cutoff = auto\nseries_max = 1\n", "[trial] series_max = 1 must be at least series = 2"),
            ("cutoff = auto\nseries_max = 0\n", "[trial] series_max = 0 must be at least 1"),
            ("cutoff = auto\nseries_max = many\n", "[trial] series_max = 'many' is not a number"),
            ("cutoff = auto\nseries = 0\n", "[trial] series = 0 must be at least 1"),
            # n_det_max: a number or empty; auto is still not built
            ("cutoff = auto\nn_det_max = auto\n", "[trial] n_det_max = auto is not built yet"),
            ("cutoff = auto\nn_det_max = 1\n", "[trial] n_det_max = 1 must be at least 2"),
        ]
        for text, expected in cases:
            self.assertIn(expected, TS.problems(TS.model_input("kind = multidet\n" + text)), text)
        # the tolerance is read in both modes
        for mode in ("cutoff = auto\n", "cutoff = 1e-3, hf\nn_det_max = 100\n"):
            for value, expected in (("0", "[trial] tolerance = 0 must be larger than 0"),
                                    ("-0.5", "[trial] tolerance = -0.5 must be larger than 0"),
                                    ("nan", "[trial] tolerance = nan must be larger than 0"),
                                    ("inf", "[trial] tolerance = inf must be larger than 0"),
                                    ("tight", "[trial] tolerance = 'tight' is not a number")):
                text = TS.problems(TS.model_input("kind = multidet\n" + mode + "tolerance = %s\n" % value))
                self.assertIn(expected, text, (mode, value))
                self.assertEqual(text.count("[trial] tolerance"), 1, text)      # one line for one mistake
        # one input with several mistakes: every one of them is reported
        text = TS.problems(TS.model_input("kind = multidet\ncutoff = auto\nauto_cutoffs = hf, 3\ntolerance = 0\nseries_max = 1\n"))
        for expected in ("auto_cutoffs = hf", "auto_cutoffs = 3 must be between", "tolerance = 0 must", "series_max = 1 must"):
            self.assertIn(expected, text)
        self.assertNotIn("holds", text)                                 # the rules of the whole list wait for usable words
        # a charge or a spin is refused as in the manual mode
        self.assertIn("[trial] kind = multidet needs [system] charge = 0 and multiplicity = 1",
                      TS.problems("[system]\ncharge = 1\n" + auto_input()))

    def test_the_template_names_every_key_with_its_default(self):
        section = TEMPLATE[TEMPLATE.index("[trial]"):TEMPLATE.index("[search]")]
        for line in ("\nkind = single\n", "\nn_det_max =\n", "\ncutoff =\n", "\nauto_cutoffs = 2.5e-4, 1e-4, 5e-5\n",
                     "\ntolerance = 1.0\n", "\nseries = 2\n", "\nseries_max = 4\n", "\nnot_converged = stop\n"):
            self.assertIn(line, section)
        self.assertIn("; auto = the package chooses the cutoffs", section)
        self.assertIn("With cutoff = auto an empty value means 100000", section)
        self.assertNotIn("not built", section)
        self.assertIn("experimental", section)                          # the stage keeps this label
        # the template with the option switched on is a valid input of the automatic mode
        with tempfile.TemporaryDirectory() as tmp:
            text = TEMPLATE.replace("kind = single", "kind = multidet").replace("\ncutoff =\n", "\ncutoff = auto\n")
            text = text.replace("mode = path", "mode = single").replace("targets = reactant, saddle", "targets = geometry")
            TS.write(os.path.join(tmp, "input.ini"), text + "[model]\nmolecule = water\n")
            cfg = Config.load(os.path.join(tmp, "input.ini"))
        self.assertEqual(cfg.trial, dict(AUTO_DEFAULT))


# ===================================================================================== the rule
WATER_HF = (-17.239747, 0.000214)           # measured DMC energies in hartree (cutoff ladder of 2026-11-27)
WATER = {1e-3: (-17.237608, 0.000315, 73), 5e-4: (-17.237300, 0.000301, 439),
         2.5e-4: (-17.240700, 0.000322, 1006), 1e-4: (-17.243796, 0.000261, 2362)}
AMMONIA_HF = (-11.729473, 0.000234)
AMMONIA = {1e-3: (-11.726324, 0.000322, 196), 5e-4: (-11.726978, 0.000315, 508),
           2.5e-4: (-11.730250, 0.000269, 1610), 1e-4: (-11.731488, 0.000272, 4105)}


def ladder_rows(hf, cuts, target="geometry"):
    """The rows of one geometry of a measured cutoff ladder."""
    rows = [rule_row(target, "start", c, e, s, facts={"n_det": n}) for c, (e, s, n) in cuts.items()]
    return rows + [rule_row(target, "start", "hf", hf[0], hf[1], facts={"n_det": 1})]


def project_rows(energy, error=1.0e-5, cutoffs=(2.5e-4, 1e-4), targets=("reactant", "saddle"), hf=True, same=()):
    """Rows of a project: ``energy(target, geometry, cutoff)`` in hartree for every row, one error for all."""
    rows = []
    for target in targets:
        for geometry in (("start",) if target in same else ("start", "final")):
            for cutoff in list(cutoffs) + (["hf"] if hf else []):
                rows.append(rule_row(target, geometry, cutoff, energy(target, geometry, cutoff), error))
    return rows


class TestRule(unittest.TestCase):
    KINDS = {"reactant": "minimum", "saddle": "saddle"}
    SAME = {"reactant": False, "saddle": False}

    def test_constants_and_words(self):
        self.assertEqual((R.ABOVE_SIGMAS, R.ABOVE_SURE_SIGMAS, R.VERDICT_SIGMAS, R.PARENT_RATIO_NOTE), (2.0, 3.0, 2.0, 10.0))
        self.assertEqual((R.CONVERGED, R.NOT_CONVERGED, R.NOT_DECIDED, R.NO_DIFFERENCE, R.NO_VERDICT),
                         ("CONVERGED", "NOT_CONVERGED", "NOT_DECIDED", "NO_DIFFERENCE", "NO_VERDICT"))
        self.assertEqual(R.DEFAULT_TOLERANCE, 1.0)
        # the rule text names every number of the rule and every verdict of a difference
        for word in ("more than 2 standard errors", "USABLE", "coarse", "fine", "|S| + 2 e", "|S| - 2 e", "CONVERGED",
                     "NOT_CONVERGED", "NOT_DECIDED", "at most 3 standard errors", "next cutoff of the ladder", "series_max"):
            self.assertIn(word, R.RULE_TEXT)
        # the automatic mode: every verdict that ends the stage, the order of the three steps, and the condition
        # of the series that tests an "above" result again
        for word in ("NO_DIFFERENCE (the project prints no energy difference) and NO_VERDICT (a row is not complete, or a "
                     "difference cannot be compared) also end the stage",
                     "With NOT_CONVERGED or NOT_DECIDED the stage takes the first of the next three steps that applies",
                     "every row gets one more DMC series while the number of series is below [trial] series_max",
                     "only to a cutoff that is above by more than 3 standard errors at no geometry and that has fewer than "
                     "two usable cutoffs with a smaller value",
                     "and only when these cutoffs and the usable cutoffs are at least two together",
                     "more data can change the two cutoffs that are compared only in this case",
                     "CONVERGED ends the stage, also when a smaller cutoff of the plan is above by at most 3 standard errors"):
            self.assertIn(word, R.RULE_TEXT)
        for word in ("It is a statement about these two cutoffs", "does not prove that a much larger expansion"):
            self.assertIn(word, R.MEANING_TEXT)
        self.assertIn("Total energies are not converged", R.TOTAL_ENERGIES_TEXT)

    def test_texts_of_numbers(self):
        self.assertEqual([R.cut_text(c) for c in (2.5e-4, 1e-4, 1e-3, 5e-5, 0.5, "hf")],
                         ["2.5e-4", "1e-4", "1e-3", "5e-5", "5e-1", "hf"])
        self.assertEqual([R.number_text(v) for v in (1.0, 0.5, 0.25, 2, 10.0)], ["1.0", "0.5", "0.25", "2.0", "10.0"])
        self.assertEqual(R.pm_text(2.139, 0.381), "+2.14 +/- 0.38")
        self.assertEqual(R.pm_text(-0.953, 0.387), "-0.95 +/- 0.39")
        self.assertEqual(R.pm_text(0.3, 1.9), "+0.3 +/- 1.9")
        self.assertEqual(R.pm_text(0.0, 0.0), "+0.000000 (exact)")
        self.assertEqual(R.bound_text({"e": 0.042, "bound": 1.2345}), "1.234")

    def test_step_1_with_the_eight_measured_values(self):
        """Water and ammonia, 4 cutoffs each: DMC energy minus that of the single RHF determinant in millihartree."""
        expected = {"water": {1e-3: "+2.14 +/- 0.38", 5e-4: "+2.45 +/- 0.37", 2.5e-4: "-0.95 +/- 0.39", 1e-4: "-4.05 +/- 0.34"},
                    "ammonia": {1e-3: "+3.15 +/- 0.40", 5e-4: "+2.49 +/- 0.39", 2.5e-4: "-0.78 +/- 0.36", 1e-4: "-2.02 +/- 0.36"}}
        for name, hf, cuts in (("water", WATER_HF, WATER), ("ammonia", AMMONIA_HF, AMMONIA)):
            found = R.against_hf(ladder_rows(hf, cuts))
            self.assertTrue(found["tested"])
            lines = {line["cutoff"]: line for line in found["lines"]}
            self.assertEqual({c: R.pm_text(line["delta"], line["error"]) for c, line in lines.items()}, expected[name])
            self.assertEqual({c: line["state"] for c, line in lines.items()},
                             {1e-3: "above", 5e-4: "above", 2.5e-4: "not above", 1e-4: "not above"})
            # the two short expansions are above by 5.6 to 7.9 standard errors: more than the 3 of a sure result
            self.assertEqual([lines[c]["sure"] for c in (1e-3, 5e-4, 2.5e-4, 1e-4)], [True, True, None, None])
            for c in (1e-3, 5e-4):
                self.assertTrue(5.5 < lines[c]["delta"] / lines[c]["error"] < 8.0)
            self.assertEqual({c["cutoff"]: c["status"] for c in found["usability"]},
                             {1e-3: "NOT USABLE", 5e-4: "NOT USABLE", 2.5e-4: "USABLE", 1e-4: "USABLE"})
            self.assertEqual([line["n_det"] for line in found["lines"]], [cuts[c][2] for c in cuts])
        # the cutoffs of version 0.1.0 of the example (1e-3 and hf): the one cut is not usable
        found = R.against_hf(ladder_rows(WATER_HF, {1e-3: WATER[1e-3]}))
        self.assertEqual([c["status"] for c in found["usability"]], ["NOT USABLE"])

    def test_step_1_thresholds(self):
        def state(sigmas, error_cut=3.0e-4, error_hf=4.0e-4):
            delta = sigmas * math.hypot(error_cut, error_hf)
            line = R.against_hf([rule_row("a", "start", 1e-3, E_HF + delta, error_cut),
                                 rule_row("a", "start", "hf", E_HF, error_hf)])["lines"][0]
            self.assertAlmostEqual(line["delta"], 1000.0 * delta, places=9)        # millihartree
            self.assertAlmostEqual(line["error"], 0.5, places=12)                  # sqrt(0.3^2 + 0.4^2) millihartree
            return line["state"], line["sure"]

        self.assertEqual(state(-4.0), ("not above", None))
        self.assertEqual(state(0.0), ("not above", None))
        self.assertEqual(state(1.999), ("not above", None))             # above means MORE than 2 standard errors
        self.assertEqual(state(2.001), ("above", False))
        self.assertEqual(state(2.999), ("above", False))                # up to 3 standard errors: not sure
        self.assertEqual(state(3.001), ("above", True))
        self.assertEqual(state(9.0), ("above", True))

    def test_step_1_a_cutoff_is_usable_only_if_it_is_above_at_no_geometry(self):
        def energy(target, geometry, cutoff):
            if cutoff == "hf":
                return E_HF
            return E_HF + (1.0e-3 if (target, geometry, cutoff) == ("saddle", "final", 2.5e-4) else -1.0e-3)
        found = R.against_hf(project_rows(energy))
        self.assertEqual(len(found["lines"]), 8)                        # 4 geometries, 2 numeric cutoffs
        self.assertEqual([(line["target"], line["geometry"], line["cutoff"]) for line in found["lines"]][:3],
                         [("reactant", "start", 2.5e-4), ("reactant", "start", 1e-4), ("reactant", "final", 2.5e-4)])
        self.assertEqual({c["cutoff"]: c["status"] for c in found["usability"]}, {2.5e-4: "NOT USABLE", 1e-4: "USABLE"})
        self.assertEqual([(line["target"], line["geometry"]) for line in found["lines"] if line["state"] == "above"],
                         [("saddle", "final")])

    def test_step_1_rows_that_are_not_complete(self):
        good = rule_row("a", "start", 1e-3, E_CUT, 1.0e-4)
        hf = rule_row("a", "start", "hf", E_HF, 1.0e-4)
        cases = [
            (dict(good, states=["DONE", "RUNNING"], done=False), hf, "1 of 2 series are DONE (DONE, RUNNING)"),
            (dict(good, states=["FAILED", "DONE"], done=False), hf, "1 of 2 series are DONE (FAILED, DONE)"),
            (dict(good, energy=None, error=None), hf, "no DMC energy"),
            (dict(good, checks={"mdvmc": {"ok": False, "verdict": "FAIL"}}), hf,
             "the VMC check without Jastrow factor did not pass (verdict FAIL)"),
            (dict(good, checks={}), hf, "the VMC check without Jastrow factor was not made"),
            (good, dict(hf, states=["DONE", "QUEUED"], done=False), "the row hf: 1 of 2 series are DONE (DONE, QUEUED)"),
            (good, dict(hf, checks={}), "the row hf: the VMC check without Jastrow factor was not made"),
        ]
        for cut, single, reason in cases:
            found = R.against_hf([cut, single])
            line = found["lines"][0]
            self.assertEqual((line["state"], line["reason"], line["delta"], line["error"], line["sure"]),
                             ("not available", reason, None, None, None))
            self.assertEqual(found["usability"][0]["status"], "no verdict yet")
        self.assertEqual(R.row_reasons(good), [])
        # a cut that is above at one geometry is NOT USABLE, whatever is missing at another geometry
        rows = [rule_row("a", "start", 1e-3, E_HF + 1.0e-2, 1.0e-4), hf,
                rule_row("a", "final", 1e-3, E_CUT, 1.0e-4, check=None), rule_row("a", "final", "hf", E_HF, 1.0e-4)]
        self.assertEqual(R.against_hf(rows)["usability"][0]["status"], "NOT USABLE")

    def test_without_a_row_hf_no_cut_is_tested(self):
        rows = project_rows(lambda t, g, c: E_CUT, hf=False)
        found = R.against_hf(rows)
        self.assertEqual((found["tested"], found["lines"]), (False, []))
        self.assertEqual([c["status"] for c in found["usability"]], ["not tested", "not tested"])
        ev = R.evaluate(rows, self.SAME, self.KINDS, 1.0)
        self.assertEqual((ev["hf"], ev["pair"], ev["verdict"]), (False, [2.5e-4, 1e-4], "CONVERGED"))
        # hf alone, or one numeric cutoff: nothing to test, and no verdict
        only_hf = [rule_row("a", "start", "hf", E_HF, 1.0e-4)]
        self.assertEqual(R.against_hf(only_hf), {"tested": False, "lines": [], "usability": []})
        ev = R.evaluate(project_rows(lambda t, g, c: E_CUT, cutoffs=(1e-3,)), self.SAME, self.KINDS, 1.0)
        self.assertEqual((ev["verdict"], ev["pair"], ev["differences"]), (None, None, []))
        self.assertIn("fewer than two numeric cutoffs", ev["reason"])
        with self.assertRaises(ValueError):
            R.decide(ev, 2, 4, [1e-3], [1e-3])

    def test_step_2_the_pair_is_the_two_smallest_usable_cutoffs(self):
        above = set()

        def energy(target, geometry, cutoff):
            if cutoff == "hf":
                return E_HF
            return E_HF + (2.0e-3 if cutoff in above else -2.0e-3)
        cutoffs = (1e-3, 5e-4, 2.5e-4, 1e-4)
        for bad, pair in (((), [2.5e-4, 1e-4]), ((1e-4,), [5e-4, 2.5e-4]), ((2.5e-4,), [5e-4, 1e-4]),
                          ((1e-3, 5e-4), [2.5e-4, 1e-4]), ((2.5e-4, 1e-4), [1e-3, 5e-4])):
            above.clear(); above.update(bad)
            ev = R.evaluate(project_rows(energy, cutoffs=cutoffs), self.SAME, self.KINDS, 1.0)
            self.assertEqual(ev["pair"], pair, bad)                     # [coarse, fine]: the larger value is the coarse one
            self.assertEqual(ev["pair_labels"], [T.entry_label(c, 100000) for c in pair])
            self.assertGreater(ev["pair"][0], ev["pair"][1])
        # the order of the cutoffs in the input does not matter
        ev = R.evaluate(project_rows(lambda t, g, c: E_HF if c == "hf" else E_CUT, cutoffs=(1e-4, 1e-3, 2.5e-4)),
                        self.SAME, self.KINDS, 1.0)
        self.assertEqual(ev["pair"], [2.5e-4, 1e-4])
        # fewer than two usable cutoffs: no pair, NOT_CONVERGED, and the reason names every cut that is above
        above.clear(); above.update((1e-3, 5e-4, 2.5e-4))
        ev = R.evaluate(project_rows(energy, cutoffs=cutoffs, targets=("reactant",)), {"reactant": False}, {}, 1.0)
        self.assertEqual((ev["pair"], ev["verdict"], ev["differences"]), (None, "NOT_CONVERGED", []))
        self.assertTrue(ev["reason"].startswith("fewer than two cutoffs are usable: the cutoff 1e-3 is above the single RHF "
                                                "determinant at reactant start by +2.000 +/- 0.014 millihartree; "), ev["reason"])
        for cutoff in ("1e-3", "5e-4", "2.5e-4"):
            for geometry in ("start", "final"):
                self.assertIn("the cutoff %s is above the single RHF determinant at reactant %s by" % (cutoff, geometry), ev["reason"])
        self.assertNotIn("the cutoff 1e-4", ev["reason"])

    def test_step_3_the_tested_differences(self):
        specs = R.difference_specs({"reactant": False, "saddle": False, "product": False},
                                   {"reactant": "minimum", "saddle": "saddle", "product": "minimum"})
        self.assertEqual([s["name"] for s in specs], [
            "final minus start of reactant", "final minus start of saddle", "final minus start of product",
            "barrier saddle minus reactant at the start geometries", "barrier saddle minus reactant at the final geometries",
            "barrier saddle minus product at the start geometries", "barrier saddle minus product at the final geometries"])
        self.assertEqual(specs[0]["terms"], [(("reactant", "final"), 1.0), (("reactant", "start"), -1.0)])
        self.assertEqual(specs[3]["terms"], [(("saddle", "start"), 1.0), (("reactant", "start"), -1.0)])
        self.assertEqual(specs[4]["terms"], [(("saddle", "final"), 1.0), (("reactant", "final"), -1.0)])
        self.assertEqual([(s["kind"], s["target"], s["minus_target"], s["geometry"]) for s in specs[:1] + specs[3:5]],
                         [("change", "reactant", None, "final minus start"), ("barrier", "saddle", "reactant", "start"),
                          ("barrier", "saddle", "reactant", "final")])
        self.assertFalse([s for s in specs if "change" in s["name"]])   # the change of a barrier is not tested
        # a target that the search did not move: no change of its own, and its start row stands for its final row
        specs = R.difference_specs({"reactant": False, "saddle": True}, self.KINDS)
        self.assertEqual([s["name"] for s in specs], ["final minus start of reactant",
                                                      "barrier saddle minus reactant at the start geometries",
                                                      "barrier saddle minus reactant at the final geometries"])
        self.assertEqual(specs[2]["terms"], [(("saddle", "start"), 1.0), (("reactant", "final"), -1.0)])
        # without kinds no barrier exists; a kind that is neither saddle nor minimum gives none either
        self.assertEqual([s["name"] for s in R.difference_specs(self.SAME)],
                         ["final minus start of reactant", "final minus start of saddle"])
        self.assertEqual(len(R.difference_specs({"0": False, "1": False}, {"0": "minimum", "1": "image"})), 2)
        self.assertEqual(R.difference_specs({"geometry": True}), [])

    def test_step_3_numbers_of_one_difference(self):
        def energy(target, geometry, cutoff):
            # every wave function sees the same change with the geometry, except the fine cut at one geometry
            place = {("reactant", "final"): -0.0010, ("saddle", "start"): 0.0150, ("saddle", "final"): 0.0140}.get(
                (target, geometry), 0.0)
            if cutoff == "hf":
                return E_HF + place
            return E_CUT + place + (0.0004 if (target, geometry, cutoff) == ("reactant", "final", 1e-4) else 0.0)
        ev = R.evaluate(project_rows(energy, error=1.0e-4), self.SAME, self.KINDS, 1.0)
        by_name = {d["name"]: d for d in ev["differences"]}
        self.assertEqual(list(by_name), ev["names"])
        self.assertEqual(len(by_name), 4)
        d = by_name["final minus start of reactant"]
        self.assertAlmostEqual(d["coarse"], -0.0010 * H, places=9)      # at 2.5e-4
        self.assertAlmostEqual(d["fine"], -0.0006 * H, places=9)        # at 1e-4
        self.assertAlmostEqual(d["coarse_error"], math.sqrt(2.0) * 1.0e-4 * H, places=9)
        self.assertAlmostEqual(d["S"], 0.0004 * H, places=9)            # fine minus coarse
        self.assertAlmostEqual(d["e"], 2.0 * 1.0e-4 * H, places=9)      # four independent rows
        self.assertAlmostEqual(d["bound"], abs(d["S"]) + 2.0 * d["e"], places=12)
        self.assertEqual(d["verdict"], "CONVERGED")                     # 0.251 + 2 x 0.126 = 0.50 kcal/mol
        # the barrier at the final geometries uses the same changed row with the other sign
        self.assertAlmostEqual(by_name["barrier saddle minus reactant at the final geometries"]["S"], -0.0004 * H, places=9)
        self.assertAlmostEqual(by_name["barrier saddle minus reactant at the start geometries"]["S"], 0.0, places=9)
        self.assertAlmostEqual(by_name["final minus start of saddle"]["fine"], -0.0010 * H, places=9)

    def test_step_3_thresholds_of_the_verdict(self):
        def verdict(change, error, tolerance=1.0):
            found = R.difference_verdict({"value": 10.0, "error": 0.0, "text": None},
                                         {"value": 10.0 + change, "error": error, "text": None}, tolerance)
            self.assertAlmostEqual(found["S"], change, places=12)
            self.assertEqual(found["e"], error)
            self.assertAlmostEqual(found["bound"], abs(change) + 2.0 * error, places=12)
            return found["verdict"]

        for sign in (1.0, -1.0):
            self.assertEqual(verdict(sign * 0.0, 0.25), "CONVERGED")
            self.assertEqual(verdict(sign * 0.49, 0.25), "CONVERGED")       # bound 0.99
            self.assertEqual(verdict(sign * 0.5, 0.25), "CONVERGED")        # bound exactly 1.0: at most the tolerance
            self.assertEqual(verdict(sign * 0.51, 0.25), "NOT_DECIDED")     # bound 1.01
            self.assertEqual(verdict(sign * 1.0, 0.25), "NOT_DECIDED")
            self.assertEqual(verdict(sign * 1.5, 0.25), "NOT_DECIDED")      # |S| - 2 e exactly 1.0: not above the tolerance
            self.assertEqual(verdict(sign * 1.51, 0.25), "NOT_CONVERGED")   # |S| - 2 e = 1.01
            self.assertEqual(verdict(sign * 9.0, 0.25), "NOT_CONVERGED")
        # an exact difference: only its size counts
        self.assertEqual([verdict(0.99, 0.0), verdict(1.0, 0.0), verdict(1.01, 0.0)], ["CONVERGED", "CONVERGED", "NOT_CONVERGED"])
        # the tolerance of the input is used
        self.assertEqual([verdict(0.3, 0.05, tolerance=0.5), verdict(0.3, 0.05, tolerance=0.35), verdict(0.3, 0.05, tolerance=0.1)],
                         ["CONVERGED", "NOT_DECIDED", "NOT_CONVERGED"])
        # the errors of the two cutoffs add in squares
        found = R.difference_verdict({"value": 1.0, "error": 0.3, "text": None}, {"value": 1.2, "error": 0.4, "text": None}, 1.0)
        self.assertAlmostEqual(found["e"], 0.5, places=12)
        # a difference that is NOT COMPARABLE at one cutoff has no verdict, with the reason
        found = R.difference_verdict({"value": None, "error": None, "text": "NOT COMPARABLE: n_det_max differs (1 at a, 2 at b)"},
                                     {"value": 1.2, "error": 0.4, "text": None}, 1.0)
        self.assertEqual(found, {"S": None, "e": None, "bound": None, "verdict": None,
                                 "reason": "at the coarse cutoff: n_det_max differs (1 at a, 2 at b)"})

    def evaluation(self, shift_fine, error=1.0e-5, **kw):
        """A project whose difference "final minus start of reactant" is ``shift_fine`` kcal/mol larger at the fine cutoff."""
        def energy(target, geometry, cutoff):
            if cutoff == "hf":
                return E_HF
            return E_CUT + (shift_fine / H if (target, geometry, cutoff) == ("reactant", "final", 1e-4) else 0.0)
        return R.evaluate(project_rows(energy, error=error), kw.get("same", self.SAME), kw.get("kinds", self.KINDS),
                          kw.get("tolerance", 1.0))

    def test_step_4_the_verdict_of_the_project(self):
        ev = self.evaluation(0.0)
        self.assertEqual((ev["verdict"], [d["verdict"] for d in ev["differences"]]), ("CONVERGED", ["CONVERGED"] * 4))
        self.assertEqual(ev["reason"], "all 4 tested energy difference(s) changed by at most 1.0 kcal/mol between the cutoffs "
                                       "2.5e-4 and 1e-4, with 2 standard errors of margin (largest bound 0.025 kcal/mol)")
        # one difference NOT_DECIDED, the others CONVERGED: NOT_DECIDED
        ev = self.evaluation(0.9, error=1.0e-4, kinds={})
        self.assertEqual(([d["verdict"] for d in ev["differences"]], ev["verdict"]), (["NOT_DECIDED", "CONVERGED"], "NOT_DECIDED"))
        self.assertEqual(ev["reason"], "final minus start of reactant changed by +0.90 +/- 0.13 kcal/mol between the cutoffs "
                                       "2.5e-4 and 1e-4 (bound 1.15 kcal/mol): the errors are too large to say whether the "
                                       "change is within the tolerance of 1.0 kcal/mol")
        # one difference NOT_CONVERGED wins over CONVERGED
        ev = self.evaluation(3.0)
        verdicts = {d["name"]: d["verdict"] for d in ev["differences"]}
        self.assertEqual(verdicts, {"final minus start of reactant": "NOT_CONVERGED", "final minus start of saddle": "CONVERGED",
                                    "barrier saddle minus reactant at the start geometries": "CONVERGED",
                                    "barrier saddle minus reactant at the final geometries": "NOT_CONVERGED"})
        self.assertEqual(ev["verdict"], "NOT_CONVERGED")
        self.assertIn("final minus start of reactant changed by +3.000 +/- 0.013 kcal/mol between the cutoffs 2.5e-4 and 1e-4, "
                      "more than the tolerance of 1.0 kcal/mol with 2 standard errors of margin", ev["reason"])
        self.assertIn("barrier saddle minus reactant at the final geometries changed by -3.000", ev["reason"])
        # the same numbers with a tolerance of 5: CONVERGED
        self.assertEqual(self.evaluation(3.0, tolerance=5.0)["verdict"], "CONVERGED")
        # one difference NOT_CONVERGED and one NOT_DECIDED: NOT_CONVERGED wins. The errors are 0.13 kcal/mol: a change
        # of 3.0 is NOT_CONVERGED (3.0 - 0.25 is above 1.0), a change of 0.9 is NOT_DECIDED (bound 1.15, and 0.65 below)
        shift = {("reactant", "final", 1e-4): 3.0 / H, ("saddle", "final", 1e-4): 0.9 / H}
        rows = project_rows(lambda t, g, c: E_HF if c == "hf" else E_CUT + shift.get((t, g, c), 0.0), error=1.0e-4)
        ev = R.evaluate(rows, self.SAME, {}, 1.0)
        self.assertEqual([(d["name"], d["verdict"]) for d in ev["differences"]],
                         [("final minus start of reactant", "NOT_CONVERGED"), ("final minus start of saddle", "NOT_DECIDED")])
        self.assertEqual(ev["verdict"], "NOT_CONVERGED")
        self.assertIn("final minus start of reactant changed by +3.00 +/- 0.13 kcal/mol", ev["reason"])
        self.assertNotIn("saddle", ev["reason"])                        # the reason names the difference that decides
        # so the stage adds the next cutoff, and not a series that NOT_DECIDED alone would add
        self.assertEqual(self.decide(ev, ladder=(2.5e-4, 1e-4, 5e-5)),
                         {"action": "next cutoff", "cause": "not converged", "verdict": None, "added": 5e-5})
        self.assertEqual(self.decide(ev), {"action": "final", "cause": "ladder ended", "verdict": "NOT_CONVERGED", "added": None})
        # the report writes the cutoffs in its own way
        ev = R.evaluate(project_rows(lambda t, g, c: E_HF if c == "hf" else E_CUT), self.SAME, self.KINDS, 1.0,
                        text=lambda c: "<%g>" % c)
        self.assertIn("between the cutoffs <0.00025> and <0.0001>", ev["reason"])

    def test_step_4_no_difference_and_no_verdict(self):
        # one target that the search did not move: the project prints no energy difference
        rows = project_rows(lambda t, g, c: E_HF + 1.0e-2 if c == 2.5e-4 else E_HF, targets=("geometry",), same=("geometry",))
        ev = R.evaluate(rows, {"geometry": True}, {"geometry": "minimum"}, 1.0)
        self.assertEqual((ev["verdict"], ev["names"], ev["pair"]), ("NO_DIFFERENCE", [], None))
        self.assertEqual(ev["reason"], "the project prints no energy difference (the search moved no target, and there is no "
                                       "pair of a saddle target and a minimum target), so nothing can be tested")
        self.assertEqual(ev["usability"][0]["status"], "NOT USABLE")    # step 1 is still made
        # a row that is not complete: no verdict, and the reason names the row
        rows = project_rows(lambda t, g, c: E_HF if c == "hf" else E_CUT)
        rows[4]["checks"] = {}                                          # reactant final, cutoff 1e-4
        rows[11]["states"], rows[11]["done"] = ["DONE", "FAILED"], False   # saddle final, hf
        ev = R.evaluate(rows, self.SAME, self.KINDS, 1.0)
        self.assertEqual((ev["verdict"], ev["pair"], ev["differences"]), ("NO_VERDICT", None, []))
        self.assertEqual(ev["reason"], "2 row(s) of the stage are not complete: reactant final, cutoff 1e-4: the VMC check "
                                       "without Jastrow factor was not made; saddle final, cutoff hf: 1 of 2 series are DONE "
                                       "(DONE, FAILED)")
        self.assertEqual([(i["target"], i["geometry"], i["cutoff"]) for i in ev["incomplete"]],
                         [("reactant", "final", 1e-4), ("saddle", "final", "hf")])
        # complete rows whose facts differ: the difference cannot be compared, so the project has no verdict
        rows = project_rows(lambda t, g, c: E_HF if c == "hf" else E_CUT)
        rows[3]["trial"]["n_det_max"] = 5000                            # reactant final, cutoff 2.5e-4
        ev = R.evaluate(rows, self.SAME, {}, 1.0)
        self.assertEqual(([d["verdict"] for d in ev["differences"]], ev["verdict"]), ([None, "CONVERGED"], "NO_VERDICT"))
        self.assertEqual(ev["reason"], "final minus start of reactant cannot be compared (at the coarse cutoff: n_det_max "
                                       "differs (5000 at reactant final, 100000 at reactant start))")

    def test_the_rule_of_a_printed_difference_is_the_one_of_the_report(self):
        # the report takes these functions from the rule module: one rule for the tables and for the decisions
        self.assertIs(RT.not_comparable, R.not_comparable)
        self.assertIs(RT._combination, R.combination)
        self.assertEqual(RT.NOT_COMPARABLE, "NOT COMPARABLE: ")
        a, b = rule_row("a", "final", 1e-3, -17.21, 3.0e-4), rule_row("a", "start", 1e-3, -17.20, 4.0e-4)
        d = R.combination([(a, 1.0), (b, -1.0)])
        self.assertAlmostEqual(d["value"], -0.01 * H, places=9)
        self.assertAlmostEqual(d["error"], 5.0e-4 * H, places=9)
        self.assertIsNone(d["text"])
        self.assertEqual(R.combination([(a, 1.0), (a, -1.0)]), {"value": 0.0, "error": 0.0, "text": None})
        self.assertEqual(R.combination([(a, 1.0), (None, -1.0)])["text"], "NOT COMPARABLE: a geometry has no row for this cutoff")
        self.assertEqual(R.combination([(a, 1.0), (dict(b, checks={}), -1.0)])["text"],
                         "NOT COMPARABLE: a start: the VMC check without Jastrow factor was not made")

    # ------------------------------------------------------------------ the decision (pure)
    def decide(self, ev, series=2, series_max=4, cutoffs=(2.5e-4, 1e-4), ladder=(2.5e-4, 1e-4)):
        return R.decide(ev, series, series_max, list(cutoffs), list(ladder))

    def above(self, sigmas, cutoffs=(2.5e-4, 1e-4), bad=2.5e-4, same=None, kinds=None):
        """An evaluation in which the cut ``bad`` is above hf at reactant start by ``sigmas`` standard errors."""
        error = 1.0e-5

        def energy(target, geometry, cutoff):
            if cutoff == "hf":
                return E_HF
            if (target, geometry, cutoff) == ("reactant", "start", bad):
                return E_HF + sigmas * math.sqrt(2.0) * error
            return E_CUT
        return R.evaluate(project_rows(energy, error=error, cutoffs=cutoffs), same or self.SAME, kinds or {}, 100.0)

    def test_decide_every_branch(self):
        def final(verdict, cause):
            return {"action": "final", "cause": cause, "verdict": verdict, "added": None}

        def series(n, cause):
            return {"action": "more series", "cause": cause, "verdict": None, "added": n}

        def cutoff(c, cause="not converged"):
            return {"action": "next cutoff", "cause": cause, "verdict": None, "added": c}

        long = (2.5e-4, 1e-4, 5e-5)
        # 1. CONVERGED or NO_DIFFERENCE: final, whatever the ladder and the series still allow
        self.assertEqual(self.decide(self.evaluation(0.0), ladder=long), final("CONVERGED", "converged"))
        ev = self.above(9.0, same={"reactant": True, "saddle": True})
        self.assertEqual(ev["verdict"], "NO_DIFFERENCE")
        self.assertEqual(self.decide(ev, ladder=long), final("NO_DIFFERENCE", "no difference"))
        # 2. a cut is above by at most 3 standard errors, and a series is left: one more series for every row
        ev = self.above(2.5)
        self.assertEqual((ev["verdict"], ev["pair"]), ("NOT_CONVERGED", None))
        self.assertEqual(self.decide(ev, ladder=long), series(3, "above not sure"))
        self.assertEqual(self.decide(ev, series=3, ladder=long), series(4, "above not sure"))
        #    ... no series is left: the rule goes on with branch 3
        self.assertEqual(self.decide(ev, series=4, ladder=long), cutoff(5e-5))
        self.assertEqual(self.decide(ev, series=4), final("NOT_CONVERGED", "ladder ended"))
        #    ... the cut is above by more than 3 standard errors: more data are not needed
        self.assertEqual(self.decide(self.above(3.5), ladder=long), cutoff(5e-5))
        #    ... the verdict is CONVERGED with two other cutoffs: branch 1 comes first
        ev = self.above(2.5, cutoffs=long)
        self.assertEqual((ev["verdict"], ev["pair"]), ("CONVERGED", [1e-4, 5e-5]))
        self.assertEqual(self.decide(ev, cutoffs=long, ladder=long), final("CONVERGED", "converged"))
        #    ... the series is added only when more data can change the two cutoffs that are compared.
        #    (a) the same cutoff is above by more than 3 standard errors at another geometry: it is not usable in any
        #    case, so no series is added, and branch 3 decides
        sigma = math.sqrt(2.0) * 1.0e-5                                 # the error of a cut against hf, in hartree

        def lifted(lift, shift=None):
            """Cut rows ``lift`` standard errors above hf; every other cut row is 5 millihartree below hf."""
            return lambda t, g, c: E_HF if c == "hf" else (E_HF + lift[t, g, c] * sigma if (t, g, c) in lift
                                                           else E_CUT + (shift or {}).get((t, g, c), 0.0))
        ev = R.evaluate(project_rows(lifted({("reactant", "start", 2.5e-4): 6.0, ("reactant", "final", 2.5e-4): 2.5}),
                                     error=1.0e-5), self.SAME, {}, 100.0)
        self.assertEqual([(line["geometry"], line["sure"]) for line in ev["against_hf"] if line["state"] == "above"],
                         [("start", True), ("final", False)])
        self.assertEqual((ev["verdict"], ev["pair"]), ("NOT_CONVERGED", None))
        self.assertEqual(R.above_to_test_again(ev), [])
        self.assertEqual(self.decide(ev, ladder=long), cutoff(5e-5))
        self.assertEqual(self.decide(ev), final("NOT_CONVERGED", "ladder ended"))
        #    ... the same two results the other way round (not sure at the start geometry, sure at the final one)
        ev = R.evaluate(project_rows(lifted({("reactant", "start", 2.5e-4): 2.5, ("reactant", "final", 2.5e-4): 6.0}),
                                     error=1.0e-5), self.SAME, {}, 100.0)
        self.assertEqual(R.above_to_test_again(ev), [])
        self.assertEqual(self.decide(ev, ladder=long), cutoff(5e-5))
        #    ... not sure at both geometries: more data can make the cutoff usable, so the series is added
        ev = R.evaluate(project_rows(lifted({("reactant", "start", 2.5e-4): 2.5, ("reactant", "final", 2.5e-4): 2.9}),
                                     error=1.0e-5), self.SAME, {}, 100.0)
        self.assertEqual(R.above_to_test_again(ev), [2.5e-4])
        self.assertEqual(self.decide(ev, ladder=long), series(3, "above not sure"))
        #    (b) two usable cutoffs have a smaller value: the cutoff would not be compared even if it were usable, so
        #    no series is added for it
        rows = project_rows(lifted({("reactant", "start", 2.5e-4): 2.5}, shift={("reactant", "final", 5e-5): 3.0 / H}),
                            error=1.0e-5, cutoffs=long)
        ev = R.evaluate(rows, self.SAME, {}, 1.0)
        self.assertEqual((ev["verdict"], ev["pair"]), ("NOT_CONVERGED", [1e-4, 5e-5]))
        self.assertEqual(R.above_to_test_again(ev), [])
        self.assertEqual(self.decide(ev, cutoffs=long, ladder=long + (2e-5,)), cutoff(2e-5))
        self.assertEqual(self.decide(ev, cutoffs=long, ladder=long), final("NOT_CONVERGED", "ladder ended"))
        #    ... when the two compared cutoffs are NOT_DECIDED, the series is added for that reason (branch 4)
        rows = project_rows(lifted({("reactant", "start", 2.5e-4): 2.5}, shift={("reactant", "final", 5e-5): 1.0 / H}),
                            error=1.0e-5, cutoffs=long)
        ev = R.evaluate(rows, self.SAME, {}, 1.0)
        self.assertEqual((ev["verdict"], ev["pair"]), ("NOT_DECIDED", [1e-4, 5e-5]))
        self.assertEqual(self.decide(ev, cutoffs=long, ladder=long), series(3, "not decided"))
        #    (c) the cutoff has fewer than two usable cutoffs below it: it would be one of the two compared cutoffs
        rows = project_rows(lifted({("reactant", "start", 1e-4): 2.5}, shift={("reactant", "final", 5e-5): 3.0 / H}),
                            error=1.0e-5, cutoffs=long)
        ev = R.evaluate(rows, self.SAME, {}, 1.0)
        self.assertEqual((ev["verdict"], ev["pair"]), ("NOT_CONVERGED", [2.5e-4, 5e-5]))
        self.assertEqual(R.above_to_test_again(ev), [1e-4])
        self.assertEqual(self.decide(ev, cutoffs=long, ladder=long), series(3, "above not sure"))
        self.assertEqual(self.decide(ev, series=4, cutoffs=long, ladder=long), final("NOT_CONVERGED", "ladder ended"))
        rows = project_rows(lifted({("reactant", "start", 5e-5): 2.5}, shift={("reactant", "final", 1e-4): 3.0 / H}),
                            error=1.0e-5, cutoffs=long)
        ev = R.evaluate(rows, self.SAME, {}, 1.0)
        self.assertEqual((ev["verdict"], ev["pair"]), ("NOT_CONVERGED", [2.5e-4, 1e-4]))
        self.assertEqual(R.above_to_test_again(ev), [5e-5])
        self.assertEqual(self.decide(ev, cutoffs=long, ladder=long), series(3, "above not sure"))
        #    ... two cutoffs with such a result: both are named, in the order of the plan
        rows = project_rows(lifted({("reactant", "start", 2.5e-4): 2.5, ("saddle", "final", 1e-4): 2.2}), error=1.0e-5, cutoffs=long)
        ev = R.evaluate(rows, self.SAME, {}, 1.0)
        self.assertEqual((ev["verdict"], ev["pair"]), ("NOT_CONVERGED", None))
        self.assertEqual(R.above_to_test_again(ev), [2.5e-4, 1e-4])
        self.assertEqual(self.decide(ev, cutoffs=long, ladder=long), series(3, "above not sure"))
        #    (d) more data must be able to give two cutoffs to compare. No cutoff is usable, one cutoff is above and
        #    not sure, and the other one is above by more than 3 standard errors: no pair can result, so no series
        ev = R.evaluate(project_rows(lifted({("reactant", "start", 2.5e-4): 9.0, ("reactant", "start", 1e-4): 2.1}),
                                     error=1.0e-5), self.SAME, {}, 100.0)
        self.assertEqual({c["cutoff"]: c["status"] for c in ev["usability"]}, {2.5e-4: "NOT USABLE", 1e-4: "NOT USABLE"})
        self.assertEqual(R.above_to_test_again(ev), [])
        self.assertEqual(self.decide(ev), final("NOT_CONVERGED", "ladder ended"))
        self.assertEqual(self.decide(ev, ladder=long), cutoff(5e-5))
        #    ... the same with three cutoffs, two of them above by more than 3 standard errors
        rows = project_rows(lifted({("reactant", "start", 2.5e-4): 9.0, ("reactant", "start", 1e-4): 9.0,
                                    ("reactant", "start", 5e-5): 2.1}), error=1.0e-5, cutoffs=long)
        ev = R.evaluate(rows, self.SAME, {}, 100.0)
        self.assertEqual(R.above_to_test_again(ev), [])
        self.assertEqual(self.decide(ev, cutoffs=long, ladder=long), final("NOT_CONVERGED", "ladder ended"))
        #    ... with one usable cutoff the series is added: the cutoff that is not sure can join it in a pair
        rows = project_rows(lifted({("reactant", "start", 2.5e-4): 9.0, ("reactant", "start", 1e-4): 2.1}),
                            error=1.0e-5, cutoffs=long)
        ev = R.evaluate(rows, self.SAME, {}, 100.0)
        self.assertEqual({c["cutoff"]: c["status"] for c in ev["usability"]},
                         {2.5e-4: "NOT USABLE", 1e-4: "NOT USABLE", 5e-5: "USABLE"})
        self.assertEqual(R.above_to_test_again(ev), [1e-4])
        self.assertEqual(self.decide(ev, cutoffs=long, ladder=long), series(3, "above not sure"))
        #    ... and with no usable cutoff and two cutoffs that are not sure: both can become usable
        ev = R.evaluate(project_rows(lifted({("reactant", "start", 2.5e-4): 2.5, ("reactant", "final", 1e-4): 2.1}),
                                     error=1.0e-5), self.SAME, {}, 100.0)
        self.assertEqual(R.above_to_test_again(ev), [2.5e-4, 1e-4])
        self.assertEqual(self.decide(ev), series(3, "above not sure"))
        #    (e) CONVERGED comes first: the stage ends also when a SMALLER cutoff of the plan is above and not sure,
        #    because the two smallest usable cutoffs are converged
        ev = self.above(2.5, cutoffs=long, bad=5e-5)
        self.assertEqual((ev["verdict"], ev["pair"]), ("CONVERGED", [2.5e-4, 1e-4]))
        self.assertEqual(R.above_to_test_again(ev), [5e-5])
        self.assertEqual(self.decide(ev, cutoffs=long, ladder=long), final("CONVERGED", "converged"))
        # 3. fewer than two usable cutoffs, or a difference NOT_CONVERGED: the next cutoff of the ladder, or final
        ev = self.above(9.0)
        self.assertEqual(self.decide(ev, ladder=long), cutoff(5e-5))
        self.assertEqual(self.decide(ev, ladder=long + (2e-5,)), cutoff(5e-5))      # the NEXT one, not the last one
        self.assertEqual(self.decide(ev, cutoffs=long, ladder=long + (2e-5,)), cutoff(2e-5))
        self.assertEqual(self.decide(ev), final("NOT_CONVERGED", "ladder ended"))
        ev = self.evaluation(3.0)
        self.assertEqual((ev["verdict"], ev["pair"]), ("NOT_CONVERGED", [2.5e-4, 1e-4]))
        self.assertEqual(self.decide(ev, ladder=long), cutoff(5e-5))
        self.assertEqual(self.decide(ev, series=4, series_max=4, ladder=long), cutoff(5e-5))
        self.assertEqual(self.decide(ev), final("NOT_CONVERGED", "ladder ended"))
        # 4. NOT_DECIDED: one more series. At the largest number of series the next cutoff of the ladder is added,
        # because more series are not allowed; without a further cutoff the stage ends
        ev = self.evaluation(0.9, error=1.0e-4)
        self.assertEqual(ev["verdict"], "NOT_DECIDED")
        self.assertEqual(self.decide(ev, ladder=long), series(3, "not decided"))
        self.assertEqual(self.decide(ev, series=3, ladder=long), series(4, "not decided"))
        self.assertEqual(self.decide(ev, series=4, ladder=long), cutoff(5e-5, "not decided at series_max"))
        self.assertEqual(self.decide(ev, series=2, series_max=2, ladder=long), cutoff(5e-5, "not decided at series_max"))
        self.assertEqual(self.decide(ev, series=4, ladder=long + (2e-5,)), cutoff(5e-5, "not decided at series_max"))
        self.assertEqual(self.decide(ev, series=4), final("NOT_DECIDED", "series_max reached"))
        self.assertEqual(self.decide(ev, series=2, series_max=2), final("NOT_DECIDED", "series_max reached"))
        self.assertEqual(self.decide(ev, series=4, cutoffs=long, ladder=long), final("NOT_DECIDED", "series_max reached"))
        #    ... the cutoff is added only when the errors are small enough for the verdict CONVERGED with a smaller
        #    cutoff (2 x sqrt(2) x the error at the fine cutoff at most the tolerance): the error of the next pair is
        #    estimated as sqrt(2) times the error at the fine cutoff. With a larger estimate the stage ends, also when
        #    the ladder holds a further cutoff
        self.assertTrue(R.errors_allow_convergence(ev))                 # fine error 0.0887 kcal/mol, tolerance 1.0
        loose = self.evaluation(0.0, error=5.0e-4)                      # no change at all, but e = 0.63 kcal/mol
        self.assertEqual((loose["verdict"], R.errors_allow_convergence(loose)), ("NOT_DECIDED", False))
        self.assertAlmostEqual(loose["differences"][0]["e"], 2.0 * 5.0e-4 * H, places=9)
        self.assertEqual(self.decide(loose, ladder=long), series(3, "not decided"))
        self.assertEqual(self.decide(loose, series=4, ladder=long), final("NOT_DECIDED", "errors above the tolerance"))
        self.assertEqual(self.decide(loose, series=4), final("NOT_DECIDED", "errors above the tolerance"))
        edge = self.evaluation(0.9, error=1.0e-4)
        fine = edge["differences"][0]["fine_error"]
        self.assertAlmostEqual(fine, math.sqrt(2.0) * 1.0e-4 * H, places=9)
        edge["tolerance"] = 2.0 * math.sqrt(2.0) * fine * (1.0 + 1.0e-12)     # 2 sqrt(2) fine error at the tolerance
        self.assertTrue(R.errors_allow_convergence(edge))
        edge["tolerance"] = 2.0 * math.sqrt(2.0) * fine * (1.0 - 1.0e-9)
        self.assertFalse(R.errors_allow_convergence(edge))
        #    ... the estimate uses the fine cutoff only: a large error at the coarse cutoff does not stop the stage
        coarse_noisy = R.evaluate(project_rows(lambda t, g, c: E_HF if c == "hf" else E_CUT, error=1.0e-5), self.SAME,
                                  self.KINDS, 1.0)
        for d in coarse_noisy["differences"]:
            d.update(verdict="NOT_DECIDED", coarse_error=0.9, e=math.hypot(0.9, d["fine_error"]))
        coarse_noisy["verdict"] = "NOT_DECIDED"
        self.assertTrue(R.errors_allow_convergence(coarse_noisy))
        self.assertEqual(self.decide(coarse_noisy, series=4, ladder=long), cutoff(5e-5, "not decided at series_max"))
        #    ... a difference without a verdict is not tested, and no difference at all allows the next cutoff
        self.assertTrue(R.errors_allow_convergence(dict(loose, differences=[dict(d, verdict=None) for d in loose["differences"]])))
        self.assertTrue(R.errors_allow_convergence(dict(loose, differences=[])))
        # 5. a row is not complete: final with NO_VERDICT, before every other branch
        rows = project_rows(lambda t, g, c: E_HF if c == "hf" else (E_HF + 2.5 * math.sqrt(2.0) * 1.0e-5 if c == 2.5e-4 else E_CUT))
        rows[0]["checks"] = {}
        ev = R.evaluate(rows, self.SAME, self.KINDS, 1.0)
        self.assertEqual(ev["verdict"], "NO_VERDICT")
        self.assertTrue(any(line["state"] == "above" and not line["sure"] for line in ev["against_hf"]))
        self.assertEqual(self.decide(ev, ladder=long), final("NO_VERDICT", "no verdict"))

    def test_texts_of_a_decision(self):
        entry = dict(self.evaluation(0.9, error=1.0e-4, kinds={}), time="2026-11-27 10:00:00", series=2, series_max=4,
                     cutoffs=[2.5e-4, 1e-4], action="more series", cause="not decided", added=3)
        self.assertEqual(R.headline("CONVERGED", [2.5e-4, 1e-4], 1.0, 3),
                         "CONVERGED within 1.0 kcal/mol between the cutoffs 2.5e-4 and 1e-4 (3 series)")
        self.assertEqual(R.headline("NOT_DECIDED", [2.5e-4, 1e-4], 0.5, 4),
                         "NOT_DECIDED between the cutoffs 2.5e-4 and 1e-4 with the tolerance 0.5 kcal/mol (4 series)")
        self.assertEqual(R.headline("NOT_CONVERGED", None, 1.0, 2), "NOT_CONVERGED (2 series)")
        self.assertEqual(R.headline("NO_DIFFERENCE", None, 1.0), "NO_DIFFERENCE")
        self.assertEqual(R.action_text(entry), "one more DMC series is added for every row (3 series now, at most 4), because "
                                               "the errors are too large to decide")
        self.assertEqual(R.decision_text(entry), (
            "cutoffs 2.5e-4, 1e-4 and hf with 2 DMC series; tolerance 1.0 kcal/mol. No cut is above the single RHF "
            "determinant. Between the cutoffs 2.5e-4 and 1e-4: final minus start of reactant changed by +0.90 +/- 0.13 "
            "kcal/mol (bound 1.15 kcal/mol, NOT_DECIDED); final minus start of saddle changed by +0.00 +/- 0.13 kcal/mol "
            "(bound 0.25 kcal/mol, CONVERGED). Verdict of this decision: NOT_DECIDED. Action: one more DMC series is added "
            "for every row (3 series now, at most 4), because the errors are too large to decide."))
        final = dict(entry, series=4, action="final", cause="series_max reached", added=None)
        self.assertEqual(R.final_text(final), (
            "NOT_DECIDED between the cutoffs 2.5e-4 and 1e-4 with the tolerance 1.0 kcal/mol (4 series): final minus start "
            "of reactant changed by +0.90 +/- 0.13 kcal/mol between the cutoffs 2.5e-4 and 1e-4 (bound 1.15 kcal/mol): the "
            "errors are too large to say whether the change is within the tolerance of 1.0 kcal/mol; no further series is "
            "allowed ([trial] series_max = 4), and the ladder holds no further cutoff: to continue, add a smaller cutoff at "
            "the end of [trial] auto_cutoffs"))
        self.assertEqual(R.action_text(final), "the stage ends with the verdict NOT_DECIDED (no further series is allowed "
                                               "([trial] series_max = 4), and the ladder holds no further cutoff: to continue, "
                                               "add a smaller cutoff at the end of [trial] auto_cutoffs)")
        finer = dict(entry, series=4, action="next cutoff", cause="not decided at series_max", added=5e-5)
        self.assertEqual(R.action_text(finer), "the next cutoff of the ladder, 5e-5, is added with 4 DMC series, because an "
                                               "energy difference is NOT_DECIDED, no further series is allowed ([trial] "
                                               "series_max = 4), and the errors are small enough for a verdict with a "
                                               "smaller cutoff")
        stopped = dict(entry, series=4, action="final", cause="errors above the tolerance", added=None)
        self.assertEqual(R.action_text(stopped), "the stage ends with the verdict NOT_DECIDED (no further series is allowed "
                                                 "([trial] series_max = 4), and the errors are too large for the verdict "
                                                 "CONVERGED also with a smaller cutoff (2 x sqrt(2) x the error at the fine "
                                                 "cutoff is larger than the tolerance): to continue, raise [trial] series_max "
                                                 "above 4)")
        self.assertTrue(R.final_text(stopped).endswith(
            "; no further series is allowed ([trial] series_max = 4), and the errors are too large for the verdict CONVERGED "
            "also with a smaller cutoff (2 x sqrt(2) x the error at the fine cutoff is larger than the tolerance): to "
            "continue, raise [trial] series_max above 4"))
        # the number to raise above is the number of series of the plan, also when series_max is lower
        self.assertTrue(R.final_text(dict(stopped, series_max=3)).endswith(
            "[trial] series_max = 3), and the errors are too large for the verdict CONVERGED also with a smaller cutoff (2 x "
            "sqrt(2) x the error at the fine cutoff is larger than the tolerance): to continue, raise [trial] series_max "
            "above 4"))
        # the differences that fail the estimate are named, CONVERGED ones too: here a tolerance of 0.2 kcal/mol is below
        # 2 sqrt(2) x 0.0887 = 0.251 kcal/mol for both differences
        tight = dict(stopped, tolerance=0.2)
        self.assertEqual([name for name, _value in R.estimate_failures(tight)],
                         ["final minus start of reactant", "final minus start of saddle"])
        self.assertTrue(R.final_text(tight).endswith(
            "(2 x sqrt(2) x the error at the fine cutoff is larger than the tolerance for final minus start of reactant "
            "(0.25 kcal/mol), final minus start of saddle (0.25 kcal/mol)): to continue, raise [trial] series_max above 4"))
        self.assertEqual((R.final_tail("converged", 4, 4), R.final_tail("ladder ended", 4, 4)),
                         ("", "; the ladder holds no further cutoff: to continue, add a smaller cutoff at the end of [trial] auto_cutoffs"))
        # a cut that is above, and the next cutoff
        ev = self.above(2.5, kinds=self.KINDS)
        entry = dict(ev, time="t", series=4, series_max=4, cutoffs=[2.5e-4, 1e-4], action="next cutoff", cause="not converged",
                     added=5e-5)
        text = R.decision_text(entry)
        self.assertIn("Above the single RHF determinant: the cutoff 2.5e-4 at reactant start by +0.035 +/- 0.014 millihartree "
                      "(not more than 3 standard errors).", text)
        self.assertIn("Action: the next cutoff of the ladder, 5e-5, is added with 4 DMC series, because fewer than two cutoffs "
                      "are usable.", text)
        self.assertEqual(R.final_text(dict(entry, action="final", cause="ladder ended")),
                         "NOT_CONVERGED (4 series): fewer than two cutoffs are usable: the cutoff 2.5e-4 is above the single "
                         "RHF determinant at reactant start by +0.035 +/- 0.014 millihartree; the ladder holds no further cutoff: to continue, add a smaller cutoff at the end of [trial] auto_cutoffs")
        self.assertIn("because an energy difference is NOT_CONVERGED",
                      R.action_text(dict(self.evaluation(3.0), series=2, series_max=4, action="next cutoff", cause="not converged",
                                         added=5e-5)))
        self.assertIn("because a cut is above the single RHF determinant by at most 3 standard errors",
                      R.action_text(dict(ev, series=2, series_max=4, action="more series", cause="above not sure", added=3)))
        # CONVERGED: the headline alone; and the report writes cutoffs in its own way
        done = dict(self.evaluation(0.0), series=3, series_max=4, cutoffs=[2.5e-4, 1e-4], action="final", cause="converged",
                    added=None)
        self.assertEqual(R.final_text(done), "CONVERGED within 1.0 kcal/mol between the cutoffs 2.5e-4 and 1e-4 (3 series)")
        self.assertEqual(R.final_text(done, text=RT.cutoff_text),
                         "CONVERGED within 1.0 kcal/mol between the cutoffs 0.00025 and 0.0001 (3 series)")
        self.assertEqual(R.action_text(done), "the stage ends with the verdict CONVERGED")


# ===================================================================================== the stage with the stand-in engine
class StageCase(unittest.TestCase):
    """A stage of the automatic mode on the stand-in engine. One target, which the search moved, unless a test
    gives other geometries. Every run has 120 blocks with the amplitude ``NOISE``: the error of one run is
    ``NOISE / sqrt(109)`` (9.58e-5 hartree), and the error of a row with n series is that divided by sqrt(n)."""

    BLOCKS = 120
    NOISE = 1.0e-3

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = tmp.name
        self.folder = os.path.join(tmp.name, "work", "trial")
        self.symbols = ["O", "H", "H"]
        x = M.water_geometry()
        moved = x.copy(); moved[1] += 0.01
        self.geometries = {"reactant": (x, moved)}
        self.two_targets = {"reactant": (x, moved), "saddle": (x + 0.2, moved + 0.2)}
        self.engine = Exact()
        self.log = []
        self.energy = {}                # (target, geometry, label) or a key -> energy in hartree; default: E_HF or E_CUT
        self.noise = {}                 # (target, geometry, label) or a key -> amplitude in hartree; default: NOISE

    def again(self, name):
        """A new folder, a new engine and no energies: the next case of a test."""
        self.folder = os.path.join(self.tmp, name, "work", "trial")
        self.engine, self.log, self.energy, self.noise = Exact(), [], {}, {}

    def runs(self, trial=None, **kw):
        return T.TrialRuns(kw.get("engine", self.engine), self.symbols, kw.get("geometries", self.geometries),
                           trial or AUTO, TS.SETTINGS, self.BLOCKS, self.folder, log=self.log.append,
                           clock=kw.get("clock"), kinds=kw.get("kinds"))

    def state(self):
        return json.loads(TS.read(os.path.join(self.folder, "state.json")))

    def auto(self):
        return self.state()["auto"]

    def row_error(self, series=2, noise=None):
        """The error of a row in hartree."""
        return (self.NOISE if noise is None else noise) / math.sqrt(self.BLOCKS - 11) / math.sqrt(series)

    def finish(self, runs):
        """Every point of the plan that waits gets its blocks: the energy of its row (or of its key) and the
        amplitude. Returns the keys."""
        done = []
        for key in runs.keys():
            point = self.engine.points.get(key)
            if point is None or point["state"] != QUEUED:
                continue
            target, geometry, label, _series = split(key)
            row = (target, geometry, label)
            mean = self.energy.get(key, self.energy.get(row, E_HF if label == "hf" else E_CUT))
            self.engine.finish(key, mean=mean, noise=self.noise.get(key, self.noise.get(row, self.NOISE)))
            done.append(key)
        return done

    def start(self, trial=None, **kw):
        """A stage whose first plan is asked for and finished. The next pass makes the first decision."""
        runs = self.runs(trial, **kw)
        self.assertEqual(runs.tick(), "RUNNING")
        self.finish(runs)
        return runs

    def decision_lines(self):
        return [line for line in self.log if "automatic cutoff, decision" in line]

    def rows(self, runs):
        return {(r["target"], r["geometry"], r["label"]): r for r in runs.results()["rows"]}


class TestDecisions(StageCase):
    LONG = dict(AUTO, auto={"ladder": [2.5e-4, 1e-4, 5e-5]})

    def test_the_first_plan(self):
        runs = self.runs()
        self.assertTrue(runs.automatic)
        keys = runs.keys()
        self.assertEqual(keys, ["trial/reactant/%s/%s%s" % (g, label, s) for g in ("start", "final")
                                for label in (COARSE, FINE, "hf") for s in ("", "_s2")])
        self.assertEqual(runs.tick(), "RUNNING")
        self.assertEqual(self.engine.calls, keys)                       # the numeric cutoffs are asked for before hf
        self.assertEqual(runs.message, "0 of 12 DMC run(s) are DONE, 0 write blocks, 12 wait; automatic cutoff: no verdict yet "
                                       "(the plan holds the cutoffs 2.5e-4, 1e-4 and hf with 2 series, at most 4)")
        for key, spec in self.engine.specs.items():
            hf = "/hf" in key
            self.assertEqual(spec.settings["trial"], {
                "kind": "multidet", "cutoff": "hf" if hf else (2.5e-4 if "c2.5e-04" in key else 1e-4),
                "n_det_max": None if hf else 100000, "qp2_key": "/".join(key.split("/")[:3]) + "/qp2_n100000",
                "qp2_basis": "cc-pvtz_ecp_ccecp", "qp2_pseudo": "ccecp"})
            MD.validate_trial(MD.trial_of(spec.settings), spec.settings["charge"], spec.settings["spin"])
        self.assertTrue(any("[trial] plan: 12 DMC run(s) of 120 blocks. Targets: 1. Geometries: 2. Cutoffs: 3. Series: 2."
                            in line for line in self.log), self.log)
        self.assertTrue(any("[trial] automatic cutoff: the ladder is 2.5e-4, 1e-4; tolerance 1.0 kcal/mol; 2 DMC series at "
                            "the start and at most 4" in line for line in self.log), self.log)
        # while a run is not DONE nothing is decided
        self.finish(runs)
        last = keys[-1]
        self.engine.points[last].update(state=RUNNING, message="40 of 120 blocks")
        self.assertEqual(runs.tick(), "RUNNING")
        self.assertEqual(self.auto()["decisions"], [])
        self.assertTrue(runs.message.startswith("11 of 12 DMC run(s) are DONE, 1 write blocks, 0 wait; automatic cutoff: no "
                                                "verdict yet"))

    def test_branch_1_converged_ends_the_stage(self):
        runs = self.start(self.LONG, clock=lambda: 1.0e9)               # a third cutoff and two more series are allowed
        self.assertEqual(runs.tick(), "DONE")
        self.assertEqual((runs.status, runs.message), ("DONE", "all 12 DMC run(s) are DONE; automatic cutoff: CONVERGED within "
                                                               "1.0 kcal/mol between the cutoffs 2.5e-4 and 1e-4 (2 series)"))
        auto = self.auto()
        self.assertEqual(sorted(auto), AUTO_KEYS)
        self.assertEqual((auto["final"], auto["verdict"], auto["cutoffs"], auto["series"]), (True, "CONVERGED", [2.5e-4, 1e-4], 2))
        self.assertEqual(auto["reason"], "all 1 tested energy difference(s) changed by at most 1.0 kcal/mol between the cutoffs "
                                         "2.5e-4 and 1e-4, with 2 standard errors of margin (largest bound 0.170 kcal/mol)")
        self.assertEqual(len(auto["decisions"]), 1)
        entry = auto["decisions"][0]
        self.assertEqual((entry["action"], entry["cause"], entry["added"], entry["verdict"], entry["pair"], entry["series"],
                          entry["cutoffs"]), ("final", "converged", None, "CONVERGED", [2.5e-4, 1e-4], 2, [2.5e-4, 1e-4]))
        self.assertEqual(entry["time"], time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(1.0e9)))    # the clock of the stage
        d = entry["differences"][0]
        self.assertEqual((d["name"], d["verdict"]), ("final minus start of reactant", "CONVERGED"))
        self.assertAlmostEqual(d["S"], 0.0, places=9)
        self.assertAlmostEqual(d["e"], 2.0 * self.row_error() * H, places=9)        # four rows with the same error
        self.assertAlmostEqual(d["bound"], abs(d["S"]) + 2.0 * d["e"], places=12)
        self.assertEqual(len(self.engine.points), 12)                   # nothing was added
        self.assertEqual(len(self.decision_lines()), 1)
        self.assertIn("automatic cutoff, decision 1: cutoffs 2.5e-4, 1e-4 and hf with 2 DMC series; tolerance 1.0 kcal/mol. "
                      "No cut is above the single RHF determinant. Between the cutoffs 2.5e-4 and 1e-4: final minus start of "
                      "reactant changed by", self.decision_lines()[0])
        self.assertTrue(self.decision_lines()[0].endswith("Verdict of this decision: CONVERGED. Action: the stage ends with the "
                                                          "verdict CONVERGED."))
        # further passes: the same status, the same message, no new decision, no new point
        before = self.state()
        for _ in range(3):
            self.assertEqual(runs.tick(), "DONE")
        self.assertEqual(self.state(), before)
        self.assertEqual(len(self.decision_lines()), 1)

    def test_branch_1_no_difference_ends_the_stage(self):
        x = self.geometries["reactant"][0]
        self.energy["geometry", "start", COARSE] = E_HF + 1.0e-2       # far above hf: step 1 would ask for the next cutoff
        runs = self.start(self.LONG, geometries={"geometry": (x, x.copy())})
        self.assertEqual(len(runs.keys()), 6)                           # one geometry: the search did not move the target
        self.assertEqual(runs.tick(), "DONE")
        self.assertEqual(runs.message, "all 6 DMC run(s) are DONE; automatic cutoff: NO_DIFFERENCE (2 series): the project "
                                       "prints no energy difference (the search moved no target, and there is no pair of a "
                                       "saddle target and a minimum target), so nothing can be tested")
        auto = self.auto()
        self.assertEqual((auto["final"], auto["verdict"], auto["cutoffs"], auto["series"]), (True, "NO_DIFFERENCE", [2.5e-4, 1e-4], 2))
        self.assertEqual((auto["decisions"][0]["action"], auto["decisions"][0]["differences"]), ("final", []))
        self.assertEqual(len(self.engine.points), 6)
        # two targets that did not move, a saddle and a minimum: their barrier is a difference, so the rule runs
        self.again("barrier")
        geometries = {"reactant": (x, x.copy()), "saddle": (x + 0.2, x + 0.2)}
        runs = self.start(geometries=geometries, kinds={"reactant": "minimum", "saddle": "saddle"})
        self.assertEqual(runs.tick(), "DONE")
        self.assertEqual(self.auto()["verdict"], "CONVERGED")
        self.assertEqual([d["name"] for d in self.auto()["decisions"][0]["differences"]],
                         ["barrier saddle minus reactant at the start geometries",
                          "barrier saddle minus reactant at the final geometries"])
        # ... and without the kinds there is no barrier: nothing can be tested
        self.again("no_kinds")
        runs = self.start(geometries=geometries)
        self.assertEqual(runs.tick(), "DONE")
        self.assertEqual(self.auto()["verdict"], "NO_DIFFERENCE")

    def above(self, sigmas, series=2):
        """Put the coarse cut at the start geometry ``sigmas`` standard errors above hf (rows with ``series`` runs)."""
        self.energy["reactant", "start", COARSE] = E_HF + sigmas * math.sqrt(2.0) * self.row_error(series)
        # the energy change of the target must not depend on this: the final geometry gets the same shift
        self.energy["reactant", "final", COARSE] = self.energy["reactant", "start", COARSE]

    def test_branch_2_thresholds_of_above(self):
        # 1.9 standard errors: not above, so both cutoffs are usable and the differences decide
        self.above(1.9)
        runs = self.start()
        self.assertEqual(runs.tick(), "DONE")
        self.assertEqual(self.auto()["verdict"], "CONVERGED")
        # 2.5 standard errors: above, but not sure: one more series for every row
        self.again("unsure")
        self.above(2.5)
        runs = self.start()
        self.assertEqual(runs.tick(), "RUNNING")
        auto = self.auto()
        self.assertEqual((auto["final"], auto["verdict"], auto["series"], auto["cutoffs"]), (False, None, 3, [2.5e-4, 1e-4]))
        entry = auto["decisions"][0]
        self.assertEqual((entry["action"], entry["cause"], entry["added"], entry["verdict"], entry["pair"]),
                         ("more series", "above not sure", 3, "NOT_CONVERGED", None))
        lines = [line for line in entry["against_hf"] if line["state"] == "above"]
        self.assertEqual([(line["target"], line["geometry"], line["cutoff"], line["sure"]) for line in lines],
                         [("reactant", "start", 2.5e-4, False), ("reactant", "final", 2.5e-4, False)])
        self.assertAlmostEqual(lines[0]["delta"] / lines[0]["error"], 2.5, places=6)
        self.assertAlmostEqual(lines[0]["error"], 1000.0 * math.sqrt(2.0) * self.row_error(), places=9)    # millihartree
        self.assertEqual(runs.message, "12 of 18 DMC run(s) are DONE, 0 write blocks, 6 wait; automatic cutoff: verdict "
                                       "NOT_CONVERGED with 2 series, so one more DMC series is added for every row (3 series "
                                       "now, at most 4), because a cut is above the single RHF determinant by at most 3 "
                                       "standard errors, and more data test this result again")
        # the plan is rectangular: every cutoff, hf included, has 3 series at every geometry
        keys = runs.keys()
        self.assertEqual(keys, ["trial/reactant/%s/%s%s" % (g, label, s) for g in ("start", "final")
                                for label in (COARSE, FINE, "hf") for s in ("", "_s2", "_s3")])
        self.assertEqual(len(self.engine.points), 12)                   # the new runs are asked for on the next pass
        self.assertEqual(runs.tick(), "RUNNING")
        self.assertEqual(sorted(self.engine.points), sorted(keys))
        for key in keys:
            if key.endswith("_s3"):
                spec = self.engine.specs[key]
                self.assertEqual((spec.series, spec.base_key, spec.blocks), (3, key[:-3], self.BLOCKS))
        # 3.5 standard errors: above and sure. More data are not asked for: the cutoff is not usable (branch 3)
        self.again("sure")
        self.above(3.5)
        runs = self.start()
        self.assertEqual(runs.tick(), "DONE")
        auto = self.auto()
        self.assertEqual((auto["verdict"], auto["series"], auto["decisions"][0]["cause"]), ("NOT_CONVERGED", 2, "ladder ended"))
        self.assertEqual(len(self.engine.points), 12)

    def test_branch_2_the_result_is_tested_again_with_more_data(self):
        # case A: the third series shows that the cut is not above: the stage ends CONVERGED
        self.above(2.5)
        runs = self.start()
        self.assertEqual(runs.tick(), "RUNNING")                        # decision 1: one more series
        self.assertEqual(runs.tick(), "RUNNING")                        # the 6 new runs are asked for
        for geometry in ("start", "final"):                             # a precise third run, 5 millihartree below hf
            key = "trial/reactant/%s/%s_s3" % (geometry, COARSE)
            self.energy[key], self.noise[key] = E_CUT, 1.0e-6
        self.assertEqual(len(self.finish(runs)), 6)
        self.assertEqual(runs.tick(), "DONE")
        self.assertEqual(runs.message, "all 18 DMC run(s) are DONE; automatic cutoff: CONVERGED within 1.0 kcal/mol between "
                                       "the cutoffs 2.5e-4 and 1e-4 (3 series)")
        auto = self.auto()
        self.assertEqual([(d["action"], d["cause"], d["series"]) for d in auto["decisions"]],
                         [("more series", "above not sure", 2), ("final", "converged", 3)])
        self.assertFalse([line for line in auto["decisions"][1]["against_hf"] if line["state"] == "above"])
        self.assertEqual(len(self.decision_lines()), 2)
        # case B: the third series has the same energy: now the cut is above by 2.5 x sqrt(3/2) = 3.06 standard errors
        self.again("still_above")
        self.above(2.5)
        runs = self.start()
        runs.tick(); runs.tick()
        self.finish(runs)
        self.assertEqual(runs.tick(), "DONE")
        auto = self.auto()
        self.assertEqual((auto["verdict"], auto["series"]), ("NOT_CONVERGED", 3))
        line = [line for line in auto["decisions"][1]["against_hf"] if line["state"] == "above"][0]
        self.assertAlmostEqual(line["delta"] / line["error"], 2.5 * math.sqrt(1.5), places=6)
        self.assertTrue(line["sure"])
        self.assertEqual([(d["action"], d["cause"]) for d in auto["decisions"]],
                         [("more series", "above not sure"), ("final", "ladder ended")])
        self.assertTrue(runs.message.startswith(
            "all 18 DMC run(s) are DONE; automatic cutoff: NOT_CONVERGED (3 series): fewer than two cutoffs are usable: the "
            "cutoff 2.5e-4 is above the single RHF determinant at reactant start by +0.239 +/- 0.078 millihartree; "), runs.message)
        self.assertTrue(runs.message.endswith("; the ladder holds no further cutoff: to continue, add a smaller cutoff at the end of [trial] auto_cutoffs"))

    def test_branch_2_no_series_when_no_pair_of_cutoffs_can_result(self):
        # the coarse cut is above by 9 standard errors and the fine cut by 2.1: also if more data showed that the fine
        # cut is not above, only one cutoff would be usable. The ladder holds no further cutoff, so the stage ends at
        # once, with the 12 runs of its first plan
        self.above(9.0)
        self.energy["reactant", "start", FINE] = E_HF + 2.1 * math.sqrt(2.0) * self.row_error()
        self.energy["reactant", "final", FINE] = self.energy["reactant", "start", FINE]
        runs = self.start()
        self.assertEqual(runs.tick(), "DONE")
        auto = self.auto()
        self.assertEqual((auto["verdict"], auto["series"], len(auto["decisions"]), auto["decisions"][0]["cause"]),
                         ("NOT_CONVERGED", 2, 1, "ladder ended"))
        self.assertEqual({line["cutoff"]: line["sure"] for line in auto["decisions"][0]["against_hf"]}, {2.5e-4: True, 1e-4: False})
        self.assertEqual(len(self.engine.points), 12)
        # with a further cutoff in the ladder that cutoff is added first. Then one usable cutoff exists, and the
        # result that is not sure is tested again with more data
        self.again("longer_ladder")
        self.above(9.0)
        self.energy["reactant", "start", FINE] = E_HF + 2.1 * math.sqrt(2.0) * self.row_error()
        self.energy["reactant", "final", FINE] = self.energy["reactant", "start", FINE]
        runs = self.start(self.LONG)
        self.assertEqual(runs.tick(), "RUNNING")
        self.assertEqual((self.auto()["cutoffs"], self.auto()["series"], self.auto()["decisions"][0]["action"]),
                         ([2.5e-4, 1e-4, 5e-5], 2, "next cutoff"))
        self.assertEqual(runs.tick(), "RUNNING")
        self.assertEqual(len(self.finish(runs)), 4)
        self.assertEqual(runs.tick(), "RUNNING")
        entry = self.auto()["decisions"][1]
        self.assertEqual((entry["action"], entry["cause"], entry["added"]), ("more series", "above not sure", 3))

    def test_branch_2_without_a_series_left_the_next_branch_decides(self):
        self.above(2.5)
        runs = self.start(dict(self.LONG, series_max=2))                # series = series_max = 2
        self.assertEqual(runs.tick(), "RUNNING")
        auto = self.auto()
        self.assertEqual((auto["series"], auto["cutoffs"]), (2, [2.5e-4, 1e-4, 5e-5]))
        self.assertEqual((auto["decisions"][0]["action"], auto["decisions"][0]["added"]), ("next cutoff", 5e-5))

    def test_branch_3_fewer_than_two_usable_cutoffs_adds_the_next_cutoff(self):
        self.above(9.0)
        runs = self.start(self.LONG)
        self.assertEqual(runs.tick(), "RUNNING")
        auto = self.auto()
        self.assertEqual((auto["final"], auto["cutoffs"], auto["series"]), (False, [2.5e-4, 1e-4, 5e-5], 2))
        entry = auto["decisions"][0]
        self.assertEqual((entry["action"], entry["cause"], entry["added"], entry["verdict"], entry["pair"], entry["differences"]),
                         ("next cutoff", "not converged", 5e-5, "NOT_CONVERGED", None, []))
        self.assertEqual({c["cutoff"]: c["status"] for c in entry["usability"]}, {2.5e-4: "NOT USABLE", 1e-4: "USABLE"})
        self.assertEqual(runs.message, "12 of 16 DMC run(s) are DONE, 0 write blocks, 4 wait; automatic cutoff: verdict "
                                       "NOT_CONVERGED with 2 series, so the next cutoff of the ladder, 5e-5, is added with 2 DMC "
                                       "series, because fewer than two cutoffs are usable")
        # the new cutoff stands in front of hf at every geometry, with the series of the other rows
        record = self.state()["targets"]["reactant"]["geometries"]
        for geometry in ("start", "final"):
            self.assertEqual(list(record[geometry]["cutoffs"]), [COARSE, FINE, THIRD, "hf"])
            self.assertEqual(record[geometry]["cutoffs"][THIRD], {
                "cutoff": 5e-5, "n_det_max": 100000,
                "keys": ["trial/reactant/%s/%s" % (geometry, THIRD), "trial/reactant/%s/%s_s2" % (geometry, THIRD)]})
        self.assertEqual(runs.tick(), "RUNNING")
        new = sorted(k for k in self.engine.specs if THIRD in k)
        self.assertEqual(len(new), 4)
        self.assertEqual(self.engine.specs[new[0]].settings["trial"]["cutoff"], 5e-5)
        self.assertEqual(self.engine.specs[new[0]].settings["trial"]["qp2_key"], "trial/reactant/final/qp2_n100000")
        # the two smallest usable cutoffs are now 1e-4 and 5e-5
        self.finish(runs)
        self.assertEqual(runs.tick(), "DONE")
        self.assertEqual(runs.message, "all 16 DMC run(s) are DONE; automatic cutoff: CONVERGED within 1.0 kcal/mol between "
                                       "the cutoffs 1e-4 and 5e-5 (2 series)")
        auto = self.auto()
        self.assertEqual((auto["verdict"], auto["decisions"][1]["pair"], auto["decisions"][1]["cutoffs"]),
                         ("CONVERGED", [1e-4, 5e-5], [2.5e-4, 1e-4, 5e-5]))
        self.assertEqual(len(self.decision_lines()), 2)
        self.assertIn("Above the single RHF determinant: the cutoff 2.5e-4 at reactant start by +0.862 +/- 0.096 millihartree; "
                      "the cutoff 2.5e-4 at reactant final by +0.862 +/- 0.096 millihartree.", self.decision_lines()[0])

    def test_branch_3_fewer_than_two_usable_cutoffs_and_no_cutoff_left(self):
        self.above(9.0)
        runs = self.start()                                             # the ladder holds 2 cutoffs
        self.assertEqual(runs.tick(), "DONE")                           # a verdict is a result, not a failure
        self.assertEqual(runs.message, (
            "all 12 DMC run(s) are DONE; automatic cutoff: NOT_CONVERGED (2 series): fewer than two cutoffs are usable: the "
            "cutoff 2.5e-4 is above the single RHF determinant at reactant start by +0.862 +/- 0.096 millihartree; the cutoff "
            "2.5e-4 is above the single RHF determinant at reactant final by +0.862 +/- 0.096 millihartree; the ladder holds "
            "no further cutoff: to continue, add a smaller cutoff at the end of [trial] auto_cutoffs"))
        auto = self.auto()
        self.assertEqual((auto["final"], auto["verdict"], auto["series"]), (True, "NOT_CONVERGED", 2))
        self.assertTrue(auto["reason"].endswith("; the ladder holds no further cutoff: to continue, add a smaller cutoff at "
                                                "the end of [trial] auto_cutoffs"))
        self.assertEqual(len(self.engine.points), 12)                   # series are not added for a result that is sure

    def test_branch_3_a_difference_that_is_not_converged(self):
        self.energy["reactant", "final", FINE] = E_CUT - 3.0 / H        # the energy change is 3 kcal/mol lower at 1e-4
        runs = self.start(self.LONG)
        self.assertEqual(runs.tick(), "RUNNING")
        entry = self.auto()["decisions"][0]
        self.assertEqual((entry["action"], entry["added"], entry["verdict"], entry["pair"]),
                         ("next cutoff", 5e-5, "NOT_CONVERGED", [2.5e-4, 1e-4]))
        d = entry["differences"][0]
        self.assertAlmostEqual(d["S"], -3.0, places=8)
        self.assertEqual(d["verdict"], "NOT_CONVERGED")
        self.assertGreater(abs(d["S"]) - 2.0 * d["e"], 1.0)
        self.assertTrue(runs.message.endswith("so the next cutoff of the ladder, 5e-5, is added with 2 DMC series, because an "
                                              "energy difference is NOT_CONVERGED"))
        self.assertEqual(self.auto()["series"], 2)                      # no series is added for NOT_CONVERGED
        # case A: the new cutoff agrees with 1e-4: CONVERGED between 1e-4 and 5e-5
        runs.tick()
        self.energy["reactant", "final", THIRD] = E_CUT - 3.0 / H
        self.finish(runs)
        self.assertEqual(runs.tick(), "DONE")
        self.assertTrue(runs.message.endswith("automatic cutoff: CONVERGED within 1.0 kcal/mol between the cutoffs 1e-4 and 5e-5 "
                                              "(2 series)"))
        # case B: the new cutoff does not agree either, and the ladder ends: NOT_CONVERGED
        self.again("ladder_ends")
        self.energy["reactant", "final", FINE] = E_CUT - 3.0 / H
        runs = self.start(self.LONG)
        runs.tick(); runs.tick()
        self.finish(runs)
        self.assertEqual(runs.tick(), "DONE")
        self.assertEqual(runs.message, (
            "all 16 DMC run(s) are DONE; automatic cutoff: NOT_CONVERGED between the cutoffs 1e-4 and 5e-5 with the tolerance "
            "1.0 kcal/mol (2 series): final minus start of reactant changed by +3.000 +/- 0.085 kcal/mol between the cutoffs "
            "1e-4 and 5e-5, more than the tolerance of 1.0 kcal/mol with 2 standard errors of margin; the ladder holds no "
            "further cutoff: to continue, add a smaller cutoff at the end of [trial] auto_cutoffs"))
        auto = self.auto()
        self.assertEqual((auto["final"], auto["verdict"]), (True, "NOT_CONVERGED"))
        self.assertEqual([(d["action"], d["cause"]) for d in auto["decisions"]],
                         [("next cutoff", "not converged"), ("final", "ladder ended")])
        # a ladder of 2 cutoffs: NOT_CONVERGED at once
        self.again("short_ladder")
        self.energy["reactant", "final", FINE] = E_CUT - 3.0 / H
        runs = self.start()
        self.assertEqual(runs.tick(), "DONE")
        self.assertEqual((self.auto()["verdict"], len(self.engine.points)), ("NOT_CONVERGED", 12))

    def test_branch_4_not_decided_adds_series_up_to_series_max(self):
        self.energy["reactant", "final", FINE] = E_CUT + 1.0 / H        # the change is exactly at the tolerance
        runs = self.start()                                             # the ladder holds no third cutoff
        expected_e = {2: 2.0 * self.row_error(2) * H, 3: 2.0 * self.row_error(3) * H, 4: 2.0 * self.row_error(4) * H}
        for series in (2, 3):
            self.assertEqual(runs.tick(), "RUNNING")                    # the decision
            auto = self.auto()
            entry = auto["decisions"][-1]
            self.assertEqual((entry["action"], entry["cause"], entry["added"], entry["verdict"], entry["series"]),
                             ("more series", "not decided", series + 1, "NOT_DECIDED", series))
            d = entry["differences"][0]
            self.assertAlmostEqual(d["S"], 1.0, places=8)
            self.assertAlmostEqual(d["e"], expected_e[series], places=9)        # more series, smaller error
            self.assertGreater(d["bound"], 1.0)
            self.assertLessEqual(abs(d["S"]) - 2.0 * d["e"], 1.0)
            self.assertEqual((auto["series"], auto["cutoffs"], auto["final"]), (series + 1, [2.5e-4, 1e-4], False))
            self.assertTrue(runs.message.endswith("verdict NOT_DECIDED with %d series, so one more DMC series is added for "
                                                  "every row (%d series now, at most 4), because the errors are too large to "
                                                  "decide" % (series, series + 1)), runs.message)
            self.assertEqual(runs.tick(), "RUNNING")                    # the new runs are asked for
            self.assertEqual(len(self.finish(runs)), 6)
        self.assertEqual(runs.tick(), "DONE")                           # 4 series: the largest number
        self.assertEqual(runs.message, (
            "all 24 DMC run(s) are DONE; automatic cutoff: NOT_DECIDED between the cutoffs 2.5e-4 and 1e-4 with the tolerance "
            "1.0 kcal/mol (4 series): final minus start of reactant changed by +1.000 +/- 0.060 kcal/mol between the cutoffs "
            "2.5e-4 and 1e-4 (bound 1.120 kcal/mol): the errors are too large to say whether the change is within the "
            "tolerance of 1.0 kcal/mol; no further series is allowed ([trial] series_max = 4), and the ladder holds no "
            "further cutoff: to continue, add a smaller cutoff at the end of [trial] auto_cutoffs"))
        auto = self.auto()
        self.assertEqual((auto["final"], auto["verdict"], auto["series"], auto["cutoffs"]), (True, "NOT_DECIDED", 4, [2.5e-4, 1e-4]))
        self.assertEqual([(d["action"], d["cause"]) for d in auto["decisions"]],
                         [("more series", "not decided"), ("more series", "not decided"), ("final", "series_max reached")])
        self.assertAlmostEqual(auto["decisions"][2]["differences"][0]["e"], expected_e[4], places=9)
        self.assertEqual(len(self.engine.points), 24)                   # 2 geometries x 3 cutoffs x 4 series, no fifth one
        self.assertEqual(len(self.decision_lines()), 3)
        for _ in range(3):
            self.assertEqual(runs.tick(), "DONE")
        self.assertEqual(len(self.engine.points), 24)

    def test_branch_4_at_the_largest_number_of_series_the_next_cutoff_is_added(self):
        # the change is exactly at the tolerance between 2.5e-4 and 1e-4, and it is 0 between 1e-4 and 5e-5
        self.energy["reactant", "final", FINE] = E_CUT + 1.0 / H
        self.energy["reactant", "final", THIRD] = E_CUT + 1.0 / H
        runs = self.start(self.LONG)
        for _series in (2, 3):                                          # NOT_DECIDED: one more series, twice
            self.assertEqual(runs.tick(), "RUNNING")
            self.assertEqual(runs.tick(), "RUNNING")
            self.assertEqual(len(self.finish(runs)), 6)
        # 4 series, still NOT_DECIDED: more series are not allowed, so the next cutoff of the ladder joins the plan
        self.assertEqual(runs.tick(), "RUNNING")
        auto = self.auto()
        entry = auto["decisions"][-1]
        self.assertEqual((entry["action"], entry["cause"], entry["added"], entry["verdict"], entry["series"], entry["pair"]),
                         ("next cutoff", "not decided at series_max", 5e-5, "NOT_DECIDED", 4, [2.5e-4, 1e-4]))
        self.assertEqual((auto["final"], auto["series"], auto["cutoffs"]), (False, 4, [2.5e-4, 1e-4, 5e-5]))
        self.assertTrue(runs.message.endswith(
            "automatic cutoff: verdict NOT_DECIDED with 4 series, so the next cutoff of the ladder, 5e-5, is added with 4 DMC "
            "series, because an energy difference is NOT_DECIDED, no further series is allowed ([trial] series_max = 4), and "
            "the errors are small enough for a verdict with a smaller cutoff"), runs.message)
        self.assertEqual(runs.tick(), "RUNNING")                        # the 8 runs of the new cutoff are asked for
        self.assertEqual(len(self.finish(runs)), 8)                     # 2 geometries x 4 series
        self.assertEqual(runs.tick(), "DONE")
        self.assertEqual(runs.message, "all 32 DMC run(s) are DONE; automatic cutoff: CONVERGED within 1.0 kcal/mol between "
                                       "the cutoffs 1e-4 and 5e-5 (4 series)")
        self.assertEqual([(d["action"], d["cause"]) for d in self.auto()["decisions"]],
                         [("more series", "not decided"), ("more series", "not decided"),
                          ("next cutoff", "not decided at series_max"), ("final", "converged")])
        self.assertEqual((self.engine.stopped, len(self.engine.points)), ([], 32))
        # the last cutoff of the ladder is also NOT_DECIDED: the stage ends, because no step is left
        self.again("no_step_left")
        self.energy["reactant", "final", FINE] = E_CUT + 1.0 / H        # +1.0 kcal/mol between 2.5e-4 and 1e-4
        runs = self.start(self.LONG)                                    # ... and -1.0 kcal/mol between 1e-4 and 5e-5
        for _ in range(12):
            if runs.tick() != "RUNNING":
                break
            self.finish(runs)
        auto = self.auto()
        self.assertEqual((runs.status, auto["verdict"], auto["series"], auto["cutoffs"]),
                         ("DONE", "NOT_DECIDED", 4, [2.5e-4, 1e-4, 5e-5]))
        self.assertEqual([(d["action"], d["cause"]) for d in auto["decisions"]],
                         [("more series", "not decided"), ("more series", "not decided"),
                          ("next cutoff", "not decided at series_max"), ("final", "series_max reached")])
        self.assertEqual((auto["decisions"][-1]["pair"], len(self.engine.points), self.engine.stopped), ([1e-4, 5e-5], 32, []))
        self.assertTrue(runs.message.endswith("; no further series is allowed ([trial] series_max = 4), and the ladder holds "
                                              "no further cutoff: to continue, add a smaller cutoff at the end of [trial] "
                                              "auto_cutoffs"), runs.message)
        # a stage that ended NOT_DECIDED with a ladder of 2 cutoffs goes on when the ladder gets one more cutoff
        self.again("longer_ladder_later")
        self.energy["reactant", "final", FINE] = E_CUT + 1.0 / H
        self.energy["reactant", "final", THIRD] = E_CUT + 1.0 / H
        runs = self.start()
        for _ in range(8):
            if runs.tick() != "RUNNING":
                break
            self.finish(runs)
        self.assertEqual((runs.status, self.auto()["verdict"], self.auto()["decisions"][-1]["cause"], len(self.engine.points)),
                         ("DONE", "NOT_DECIDED", "series_max reached", 24))
        again = self.runs(self.LONG)
        self.assertEqual((again.status, self.auto()["final"], self.auto()["series"]), ("RUNNING", False, 4))
        self.assertEqual(again.tick(), "RUNNING")
        entry = self.auto()["decisions"][-1]
        self.assertEqual((entry["action"], entry["cause"], entry["added"], entry["ladder"]),
                         ("next cutoff", "not decided at series_max", 5e-5, [2.5e-4, 1e-4, 5e-5]))
        self.assertEqual(again.tick(), "RUNNING")
        self.assertEqual(len(self.finish(again)), 8)
        self.assertEqual(again.tick(), "DONE")
        self.assertEqual((self.auto()["verdict"], self.auto()["decisions"][-1]["pair"], self.engine.stopped, len(self.engine.points)),
                         ("CONVERGED", [1e-4, 5e-5], [], 32))
        # the new cutoff is above the single determinant: it is not usable, the old pair is compared again, and the
        # stage ends NOT_DECIDED between the cutoffs 2.5e-4 and 1e-4
        self.again("new_cutoff_not_usable")
        self.energy["reactant", "final", FINE] = E_CUT + 1.0 / H
        for geometry in ("start", "final"):
            self.energy["reactant", geometry, THIRD] = E_HF + 1.0e-3
        runs = self.start(self.LONG)
        for _ in range(12):
            if runs.tick() != "RUNNING":
                break
            self.finish(runs)
        auto = self.auto()
        self.assertEqual((runs.status, auto["verdict"], auto["cutoffs"], auto["decisions"][-1]["pair"]),
                         ("DONE", "NOT_DECIDED", [2.5e-4, 1e-4, 5e-5], [2.5e-4, 1e-4]))
        self.assertEqual({c["cutoff"]: c["status"] for c in auto["decisions"][-1]["usability"]},
                         {2.5e-4: "USABLE", 1e-4: "USABLE", 5e-5: "NOT USABLE"})
        self.assertEqual([d["cause"] for d in auto["decisions"]],
                         ["not decided", "not decided", "not decided at series_max", "series_max reached"])

    def test_branch_4_no_smaller_cutoff_when_the_errors_are_too_large_also_for_a_smaller_cutoff(self):
        # no difference changes between any two cutoffs, but the tolerance 0.08 kcal/mol is below 2 x sqrt(2) x the
        # fine error (0.12 kcal/mol with 4 series), so a smaller cutoff could not give CONVERGED: the stage ends after
        # the last series, with the 24 runs of the first two cutoffs
        tight = dict(self.LONG, tolerance=0.08)
        runs = self.start(tight)
        for _ in range(8):
            if runs.tick() != "RUNNING":
                break
            self.finish(runs)
        auto = self.auto()
        self.assertEqual((runs.status, auto["verdict"], auto["series"], auto["cutoffs"], len(self.engine.points)),
                         ("DONE", "NOT_DECIDED", 4, [2.5e-4, 1e-4], 24))
        self.assertEqual([(d["action"], d["cause"]) for d in auto["decisions"]],
                         [("more series", "not decided"), ("more series", "not decided"), ("final", "errors above the tolerance")])
        last = auto["decisions"][-1]["differences"][0]
        self.assertAlmostEqual(last["S"], 0.0, places=8)
        self.assertGreater(2.0 * last["e"], 0.08)
        self.assertTrue(runs.message.endswith("; no further series is allowed ([trial] series_max = 4), and the errors are too "
                                              "large for the verdict CONVERGED also with a smaller cutoff (2 x sqrt(2) x the "
                                              "error at the fine cutoff is larger than the tolerance for final minus start of "
                                              "reactant (0.12 kcal/mol)): to continue, raise [trial] series_max above 4"),
                        runs.message)
        self.assertEqual(auto["reason"], runs.message.split(": ", 2)[2])
        # raised series_max: the stage goes on with more series, as the message says
        again = self.runs(dict(tight, series_max=6))
        self.assertEqual(again.tick(), "RUNNING")
        self.assertEqual((self.auto()["series"], self.auto()["decisions"][-1]["action"]), (5, "more series"))

    def test_branch_4_a_kept_plan_with_more_series_than_series_max_and_a_longer_ladder(self):
        # the default ladder ends NOT_DECIDED with 4 series and small errors (the change is at the tolerance)
        self.energy["reactant", "final", FINE] = E_CUT + 1.0 / H
        runs = self.start()
        for _ in range(8):
            if runs.tick() != "RUNNING":
                break
            self.finish(runs)
        self.assertEqual((self.auto()["verdict"], self.auto()["series"], self.auto()["decisions"][-1]["cause"]),
                         ("NOT_DECIDED", 4, "series_max reached"))
        # a longer ladder and series_max lowered to 2: the plan keeps its 4 series, no series is added, and the new
        # cutoff gets the 4 series of the other rows
        again = self.runs(dict(self.LONG, series_max=2))
        self.assertEqual((self.auto()["series"], self.auto()["series_max"], len(again.keys())), (4, 2, 24))
        self.assertEqual(again.tick(), "RUNNING")
        entry = self.auto()["decisions"][-1]
        self.assertEqual((entry["action"], entry["cause"], entry["added"], entry["series"]),
                         ("next cutoff", "not decided at series_max", 5e-5, 4))
        self.assertEqual(again.tick(), "RUNNING")
        self.assertEqual(len(self.finish(again)), 8)
        self.assertEqual(again.tick(), "DONE")
        self.assertEqual((len(self.engine.points), self.engine.stopped, self.auto()["series"]), (32, [], 4))

    def test_branch_4_more_data_can_decide(self):
        self.energy["reactant", "final", FINE] = E_CUT + 1.0 / H
        runs = self.start()
        self.assertEqual(runs.tick(), "RUNNING")                        # NOT_DECIDED: one more series
        runs.tick()
        key = "trial/reactant/final/%s_s3" % FINE                       # a precise third run without the shift
        self.energy[key], self.noise[key] = E_CUT, 1.0e-6
        self.finish(runs)
        self.assertEqual(runs.tick(), "DONE")
        self.assertEqual(runs.message, "all 18 DMC run(s) are DONE; automatic cutoff: CONVERGED within 1.0 kcal/mol between "
                                       "the cutoffs 2.5e-4 and 1e-4 (3 series)")
        # with series_max = series the stage cannot add a series: NOT_DECIDED at once
        self.again("no_series_left")
        self.energy["reactant", "final", FINE] = E_CUT + 1.0 / H
        runs = self.start(dict(AUTO, series_max=2))
        self.assertEqual(runs.tick(), "DONE")
        auto = self.auto()
        self.assertEqual((auto["verdict"], auto["series"], auto["cutoffs"], len(self.engine.points)),
                         ("NOT_DECIDED", 2, [2.5e-4, 1e-4], 12))
        self.assertTrue(runs.message.endswith("; no further series is allowed ([trial] series_max = 2), and the ladder holds "
                                              "no further cutoff: to continue, add a smaller cutoff at the end of [trial] "
                                              "auto_cutoffs"), runs.message)

    def test_branch_5_a_row_that_is_not_complete_gives_no_verdict(self):
        key = "trial/reactant/final/%s" % FINE
        cases = [
            (lambda p: p["checks"].pop("mdvmc"), "the VMC check without Jastrow factor was not made"),
            (lambda p: p["checks"]["mdvmc"].update(ok=False, verdict="FAIL"),
             "the VMC check without Jastrow factor did not pass (verdict FAIL)"),
        ]
        for n, (change, reason) in enumerate(cases):
            self.again("case%d" % n)
            self.above(2.5)                                             # branch 2 would add a series, branch 3 a cutoff
            runs = self.start(self.LONG)
            change(self.engine.points[key])
            self.assertEqual(runs.tick(), "DONE")
            self.assertEqual(runs.message, "all 12 DMC run(s) are DONE; automatic cutoff: NO_VERDICT (2 series): 1 row(s) of the "
                                           "stage are not complete: reactant final, cutoff 1e-4: %s" % reason)
            auto = self.auto()
            self.assertEqual((auto["final"], auto["verdict"], auto["series"], auto["cutoffs"]),
                             (True, "NO_VERDICT", 2, [2.5e-4, 1e-4]))
            entry = auto["decisions"][0]
            self.assertEqual((entry["action"], entry["cause"], entry["pair"], entry["differences"]), ("final", "no verdict", None, []))
            self.assertEqual(entry["incomplete"], [{"target": "reactant", "geometry": "final", "label": FINE, "cutoff": 1e-4,
                                                    "reasons": [reason]}])
            self.assertEqual(len(self.engine.points), 12)
            self.assertIn("Not complete: reactant final, cutoff 1e-4: %s." % reason, self.decision_lines()[0])
        # runs that are DONE with too few blocks for an energy
        self.again("no_energy")
        runs = self.start()
        for k in (key, key + "_s2"):
            self.engine.points[k]["energy"] = self.engine.points[k]["energy"][:20]
        self.assertEqual(runs.tick(), "DONE")
        self.assertEqual(self.auto()["verdict"], "NO_VERDICT")
        self.assertIn("reactant final, cutoff 1e-4: no DMC energy", runs.message)
        # the row hf counts as every other row
        self.again("hf")
        runs = self.start()
        self.engine.points["trial/reactant/start/hf"]["checks"].pop("mdvmc")
        self.assertEqual(runs.tick(), "DONE")
        self.assertIn("NO_VERDICT (2 series): 1 row(s) of the stage are not complete: reactant start, cutoff hf: the VMC check "
                      "without Jastrow factor was not made", runs.message)

    def test_a_failed_or_stopped_run_needs_attention_and_no_decision_is_made(self):
        runs = self.start()
        bad = "trial/reactant/final/hf_s2"
        self.engine.points[bad].update(state=FAILED, message="job 7 ended in state FAILED")
        self.assertEqual(runs.tick(), "NEEDS_ATTENTION")
        self.assertEqual(runs.message, "%s: job 7 ended in state FAILED" % bad)     # the message of today, nothing added
        self.assertEqual((self.auto()["decisions"], self.auto()["final"]), ([], False))
        # after the repair the decision is made
        self.assertEqual(runs.retry(), [bad])
        self.assertEqual(runs.tick(), "RUNNING")
        self.finish(runs)
        self.assertEqual(runs.tick(), "DONE")
        self.assertEqual(self.auto()["verdict"], "CONVERGED")
        # a run that was stopped by hand
        self.again("stopped")
        runs = self.start()
        self.engine.points[bad].update(state=STOPPED, energy=np.full(50, -17.2))
        self.assertEqual(runs.tick(), "NEEDS_ATTENTION")
        self.assertEqual(self.auto()["decisions"], [])
        self.assertIn("stopped after 50 of 120 blocks", runs.message)

    def test_a_run_that_is_done_but_whose_request_is_refused_needs_attention_and_no_decision_is_made(self):
        """The engine refuses a request for a point that exists with another geometry or another trial wave function.
        The energy of such a point does not belong to the row of the plan, so no decision may use it."""
        runs = self.start()
        key = "trial/reactant/final/%s" % FINE
        refusal = "point %s exists with another geometry (largest difference 1.00e-02 angstrom)" % key
        self.engine.refuse[key] = refusal
        self.assertEqual(runs.tick(), "NEEDS_ATTENTION")                # every run is DONE in the engine
        self.assertEqual({p["state"] for p in self.engine.points.values()}, {DONE})
        # the second series of the row waits for its series-1 point, so it has the problem too
        self.assertEqual(runs.message, "%s: the point could not be submitted: %s; %s_s2: its series-1 point %s could not be "
                                       "submitted" % (key, refusal, key, key))
        self.assertEqual((self.auto()["decisions"], self.auto()["final"]), ([], False))
        self.assertEqual(runs.tick(), "NEEDS_ATTENTION")
        # while another run still writes blocks the stage goes on, and its message names the two runs
        other = "trial/reactant/start/hf_s2"
        self.engine.points[other].update(state=RUNNING)
        self.assertEqual(runs.tick(), "RUNNING")
        self.assertIn("2 run(s) need attention when the others have ended: %s: the point could not be submitted" % key,
                      runs.message)
        self.assertTrue(runs.message.startswith("9 of 12 DMC run(s) are DONE, 1 write blocks, 0 wait;"), runs.message)
        self.engine.points[other].update(state=DONE)
        # the cause is repaired: the request is accepted again, and the decision is made
        del self.engine.refuse[key]
        self.assertEqual(runs.tick(), "DONE")
        self.assertEqual((self.auto()["verdict"], len(self.auto()["decisions"])), ("CONVERGED", 1))
        # a refusal after the final decision: the stage needs attention again; the record keeps the decision
        self.engine.refuse[key] = refusal
        self.assertEqual(runs.tick(), "NEEDS_ATTENTION")
        self.assertEqual((self.auto()["verdict"], len(self.auto()["decisions"])), ("CONVERGED", 1))
        # a refused run that still writes blocks is counted once, as a problem: not as a run that writes blocks
        self.again("refused_and_running")
        runs = self.start()
        self.engine.refuse[key] = refusal
        self.engine.points[key].update(state=RUNNING)
        self.engine.points[other].update(state=RUNNING)
        self.assertEqual(runs.tick(), "RUNNING")
        self.assertTrue(runs.message.startswith("9 of 12 DMC run(s) are DONE, 1 write blocks, 0 wait; 2 run(s) need attention "
                                                "when the others have ended: "), runs.message)
        self.engine.points[other].update(state=DONE)
        self.assertEqual(runs.tick(), "NEEDS_ATTENTION")                # the refused run is the only one that is not DONE
        self.assertEqual(self.auto()["decisions"], [])
        # the log names the refusal also when the run becomes DONE in the pass that refuses its request
        self.again("refused_when_done")
        runs = self.runs()
        self.assertEqual(runs.tick(), "RUNNING")
        self.finish(runs)
        self.engine.refuse[key] = refusal
        self.assertEqual(runs.tick(), "NEEDS_ATTENTION")
        self.assertEqual(sum(("point %s: DONE after 120 block(s): the point could not be submitted: %s" % (key, refusal))
                             in line for line in self.log), 1)

    def test_a_request_that_the_batch_system_refuses_for_a_run_that_is_done_is_made_again_before_a_decision(self):
        """A request that fails with RuntimeError or OSError is made again on the next pass. In such a pass the engine
        did not confirm the geometry and the trial wave function of the point, so the point does not count as DONE."""
        clock = [1000.0]
        runs = self.start(clock=lambda: clock[0])
        key = "trial/reactant/final/%s" % FINE
        self.engine.transient[key] = "sbatch: error: Socket timed out on send/recv operation"
        self.assertEqual(runs.tick(), "RUNNING")
        self.assertEqual(self.auto()["decisions"], [])
        # the second series of the row waits for its series-1 point
        self.assertTrue(runs.message.startswith("10 of 12 DMC run(s) are DONE, 0 write blocks, 2 wait; the batch system refused "
                                                "2 request(s), made again on every pass: "), runs.message)
        del self.engine.transient[key]
        self.assertEqual(runs.tick(), "DONE")
        self.assertEqual(self.auto()["verdict"], "CONVERGED")
        # the same refusal after the final decision: the pass returns RUNNING, and the message keeps the verdict
        self.engine.transient[key] = "sbatch: error: Socket timed out on send/recv operation"
        self.assertEqual(runs.tick(), "RUNNING")
        self.assertTrue(runs.message.endswith("automatic cutoff: the record holds the final verdict CONVERGED within 1.0 "
                                              "kcal/mol between the cutoffs 2.5e-4 and 1e-4 (2 series)"), runs.message)
        self.assertEqual((self.auto()["final"], len(self.auto()["decisions"])), (True, 1))
        del self.engine.transient[key]
        self.assertEqual(runs.tick(), "DONE")
        # a refusal that lasts longer than 1 hour is a problem of the point: the stage needs attention
        self.again("long_refusal")
        runs = self.start(clock=lambda: clock[0])
        self.engine.transient[key] = "sbatch: error: Socket timed out on send/recv operation"
        self.assertEqual(runs.tick(), "RUNNING")
        clock[0] += T.REFUSAL_SECONDS + 60.0
        self.assertEqual(runs.tick(), "NEEDS_ATTENTION")
        self.assertIn("refused for 61 minutes", runs.message)
        self.assertEqual(self.auto()["decisions"], [])

    def test_barriers_are_tested_when_the_kinds_are_known(self):
        kinds = {"reactant": "minimum", "saddle": "saddle"}
        for name, given, verdict, names in (
                ("with", kinds, "NOT_CONVERGED", ["final minus start of reactant", "final minus start of saddle",
                                                  "barrier saddle minus reactant at the start geometries",
                                                  "barrier saddle minus reactant at the final geometries"]),
                ("without", None, "CONVERGED", ["final minus start of reactant", "final minus start of saddle"])):
            self.again(name)
            for geometry in ("start", "final"):                         # the fine cut lifts the saddle by 2 kcal/mol at both
                self.energy["saddle", geometry, FINE] = E_CUT + 2.0 / H  # geometries: its energy change stays, the barrier not
            runs = self.start(geometries=self.two_targets, kinds=given)
            self.assertEqual(len(runs.keys()), 24)
            self.assertEqual(runs.tick(), "DONE")
            entry = self.auto()["decisions"][0]
            self.assertEqual(([d["name"] for d in entry["differences"]], self.auto()["verdict"]), (names, verdict))
            found = {d["name"]: (round(d["S"], 6), d["verdict"]) for d in entry["differences"]}
            self.assertEqual(found["final minus start of saddle"], (0.0, "CONVERGED"))
            if given:
                self.assertEqual(found["barrier saddle minus reactant at the start geometries"], (2.0, "NOT_CONVERGED"))
                self.assertEqual(found["barrier saddle minus reactant at the final geometries"], (2.0, "NOT_CONVERGED"))
                self.assertEqual(self.state()["rule"]["cutoff_rule"]["kinds"], kinds)
                self.assertIn("barrier saddle minus reactant at the start geometries changed by +2.000 +/- 0.085 kcal/mol",
                              runs.message)

    def test_the_stage_cannot_go_on_for_ever(self):
        """The longest way: one more series, then every cutoff of a ladder of 4, then the end."""
        ladder = [2.5e-4, 1e-4, 5e-5, 2e-5]
        labels = [T.entry_label(c, 100000) for c in ladder]
        self.above(2.5)                                                 # 2 series: not sure; 3 series: sure
        for i, label in enumerate(labels[1:], start=1):                 # every cutoff lowers the energy difference by 3 kcal/mol
            self.energy["reactant", "final", label] = E_CUT - 3.0 * i / H
        runs = self.runs(dict(AUTO, auto={"ladder": ladder}))
        statuses = []
        for _ in range(40):
            statuses.append(runs.tick())
            self.finish(runs)
        auto = self.auto()
        self.assertEqual([(d["action"], d["added"]) for d in auto["decisions"]],
                         [("more series", 3), ("next cutoff", 5e-5), ("next cutoff", 2e-5), ("final", None)])
        self.assertLessEqual(len(auto["decisions"]), (4 - 2) + (len(ladder) - 2) + 1)       # series, cutoffs, the end
        self.assertEqual((auto["final"], auto["verdict"], auto["series"], auto["cutoffs"]), (True, "NOT_CONVERGED", 3, ladder))
        self.assertEqual(auto["decisions"][-1]["pair"], [5e-5, 2e-5])
        self.assertEqual(statuses.count("RUNNING"), statuses.index("DONE"))        # once DONE, always DONE
        self.assertEqual(set(statuses[statuses.index("DONE"):]), {"DONE"})
        self.assertEqual(len(self.engine.points), 2 * 5 * 3)            # 2 geometries, 4 cutoffs and hf, 3 series
        self.assertEqual(len(self.decision_lines()), 4)                 # one line in the log for each decision
        # a cutoff that joined the plan later has the series of the other rows at once
        record = self.state()["targets"]["reactant"]["geometries"]["start"]["cutoffs"]
        self.assertEqual(list(record), labels + ["hf"])
        self.assertEqual({len(cut["keys"]) for cut in record.values()}, {3})


# ===================================================================================== the record and a changed input
class TestRecord(StageCase):
    LONG = dict(AUTO, auto={"ladder": [2.5e-4, 1e-4, 5e-5]})

    def not_decided(self):
        """The energy change is 1.0 kcal/mol larger at the fine cutoff: NOT_DECIDED with 2, 3 and 4 series."""
        self.energy["reactant", "final", FINE] = E_CUT + 1.0 / H

    def same_again(self, runs):
        """A new object on the same folder with the same input: the same record, the same plan, the same status,
        nothing in the log and no request to the engine. Returns the new object."""
        before, lines, calls = self.state(), len(self.log), len(self.engine.calls)
        again = self.runs(runs.trial, kinds=runs.kinds)
        self.assertEqual(self.state(), before)
        self.assertEqual(again.keys(), runs.keys())
        self.assertEqual((again.status, again.message), (runs.status, runs.message))
        self.assertEqual((len(self.log), len(self.engine.calls)), (lines, calls))
        return again

    def test_the_record_keeps_the_input_and_the_plan_apart(self):
        self.not_decided()
        runs = self.runs()
        state = self.state()                                            # the record exists before the first pass
        self.assertEqual(state["auto"], {"ladder": [2.5e-4, 1e-4], "tolerance": 1.0, "series_max": 4, "cutoffs": [2.5e-4, 1e-4],
                                         "series": 2, "final": False, "verdict": None, "reason": "", "decisions": []})
        self.assertEqual(state["rule"]["trial"], AUTO)                  # exactly the values of the input
        self.assertEqual(T.stored_trial(self.folder), AUTO)             # what the workflow compares with the input
        self.assertEqual(state["rule"]["cutoff_rule"], {"text": R.RULE_TEXT, "above_sigmas": 2.0, "above_sure_sigmas": 3.0,
                                                        "verdict_sigmas": 2.0, "kinds": {}})
        self.assertEqual(sorted(state), ["auto", "left_plan", "message", "points", "refusals", "rule", "status", "stop_seen",
                                         "symbols", "targets", "to_stop", "version"])
        runs.tick(); self.finish(runs)
        self.assertEqual(runs.tick(), "RUNNING")                        # the plan grows: 3 series
        state = self.state()
        self.assertEqual(state["rule"]["trial"], AUTO)                  # the input is not touched: 2 series, 2 cutoffs and hf
        self.assertEqual((state["rule"]["trial"]["series"], state["rule"]["trial"]["cutoffs"]), (2, [2.5e-4, 1e-4, "hf"]))
        self.assertEqual(T.stored_trial(self.folder), AUTO)
        self.assertEqual((state["auto"]["series"], state["auto"]["cutoffs"], len(state["auto"]["decisions"])), (3, [2.5e-4, 1e-4], 1))
        self.assertEqual(sorted(state["auto"]), AUTO_KEYS)
        # the record names every run of the grown plan, also those that are not asked for yet
        self.assertEqual(sorted(T.record_keys(state)), sorted(runs.keys()))
        self.assertEqual(len(runs.keys()), 18)
        self.assertEqual(sorted(state["points"]), sorted(runs.keys()))
        self.assertEqual(state["points"]["trial/reactant/start/hf_s3"], {"state": "ABSENT", "message": "", "blocks": 0})
        for part in state["targets"]["reactant"]["geometries"].values():
            self.assertEqual({len(cut["keys"]) for cut in part["cutoffs"].values()}, {3})       # rectangular
        # every entry of a decision that the specification names
        entry = state["auto"]["decisions"][0]
        for name in ("time", "series", "cutoffs", "pair", "verdict", "action", "differences"):
            self.assertIn(name, entry)
        for name in ("name", "S", "e", "bound", "verdict"):
            self.assertIn(name, entry["differences"][0])
        json.dumps(state)                                               # plain values only
        self.assertEqual([f for f in os.listdir(self.folder) if f.endswith(".tmp")], [])

    def test_a_new_object_continues_at_every_point_of_the_growth(self):
        self.not_decided()
        runs = self.same_again(self.runs())                             # before the first pass
        self.assertEqual(runs.tick(), "RUNNING")
        runs = self.same_again(runs)                                    # the first plan is asked for
        self.finish(runs)
        runs = self.same_again(runs)                                    # every run is DONE, the decision is not made
        for series in (2, 3):
            self.assertEqual(runs.tick(), "RUNNING")                    # the new object makes the decision
            self.assertEqual((len(self.auto()["decisions"]), self.auto()["series"]), (series - 1, series + 1))
            points = len(self.engine.points)
            runs = self.same_again(runs)                                # the plan grew; its new runs are not asked for yet
            self.assertEqual(len(self.engine.points), points)
            specs = dict(self.engine.specs)
            self.assertEqual(runs.tick(), "RUNNING")                    # another object asks for exactly the 6 new runs
            self.assertEqual(len(self.engine.points), points + 6)
            self.assertEqual({k: v for k, v in self.engine.specs.items() if k in specs}, specs)     # no point twice
            self.assertEqual(len(self.auto()["decisions"]), series - 1)     # and no decision twice
            runs = self.same_again(runs)                                # the new runs wait
            self.assertEqual(runs.tick(), "RUNNING")
            self.assertEqual(len(self.auto()["decisions"]), series - 1)
            self.finish(runs)
            runs = self.same_again(runs)
        self.assertEqual(runs.tick(), "DONE")                           # 4 series: the final decision
        runs = self.same_again(runs)                                    # after the end
        self.assertEqual((runs.status, runs.tick()), ("DONE", "DONE"))
        auto = self.auto()
        self.assertEqual([d["action"] for d in auto["decisions"]], ["more series", "more series", "final"])
        self.assertEqual((auto["final"], auto["verdict"]), (True, "NOT_DECIDED"))
        self.assertEqual(len(self.decision_lines()), 3)                 # each decision once in the log
        self.assertEqual(sorted(self.engine.specs), sorted(runs.keys()))
        self.assertEqual(len(self.engine.specs), 24)

    def test_a_decision_whose_record_could_not_be_written_is_made_once(self):
        self.not_decided()
        runs = self.start()
        save = runs._save

        def broken():
            raise OSError("no space left on device")
        runs._save = broken
        with self.assertRaises(OSError):
            runs.tick()
        self.assertEqual((self.auto()["decisions"], self.auto()["series"]), ([], 2))    # the record holds no decision
        self.assertEqual(self.decision_lines(), [])                     # and the log names none
        # a new process decides from the same energies: one decision in the record, one line in the log
        again = self.runs()
        self.assertEqual(again.tick(), "RUNNING")
        self.assertEqual((len(self.auto()["decisions"]), self.auto()["series"]), (1, 3))
        self.assertEqual(len(self.decision_lines()), 1)
        # the same process, when the disk works again: the decision that it made is written, and logged then
        self.again("same_process")
        self.not_decided()
        runs = self.start()
        save = runs._save
        runs._save = broken
        with self.assertRaises(OSError):
            runs.tick()
        runs._save = save
        self.assertEqual(self.decision_lines(), [])
        self.assertEqual(runs.tick(), "RUNNING")
        self.assertEqual((len(self.auto()["decisions"]), self.auto()["series"]), (1, 3))
        self.assertEqual(len(self.decision_lines()), 1)
        self.assertEqual(len(self.engine.points), 18)
        self.finish(runs)
        self.assertEqual(runs.tick(), "RUNNING")
        self.assertEqual(len(self.auto()["decisions"]), 2)

    def test_a_changed_tolerance_keeps_the_plan_and_decides_again(self):
        runs = self.start()
        self.assertEqual(runs.tick(), "DONE")                           # CONVERGED within 1.0 kcal/mol; bound 0.17
        first = self.auto()["decisions"][0]
        tight = dict(AUTO, tolerance=0.1)
        again = self.runs(tight)
        auto = self.auto()
        self.assertEqual((auto["final"], auto["verdict"], auto["reason"], auto["tolerance"]), (False, None, "", 0.1))
        self.assertEqual((auto["cutoffs"], auto["series"], auto["decisions"]), ([2.5e-4, 1e-4], 2, [first]))
        self.assertEqual((again.status, again.message), ("RUNNING", ""))           # the stage has to decide again
        self.assertEqual(self.state()["rule"]["trial"], tight)
        self.assertEqual((self.state()["to_stop"], self.state()["left_plan"]), ([], []))
        self.assertEqual(sum("automatic cutoff: the values of [trial] changed; the plan that was reached is kept (cutoffs "
                             "2.5e-4, 1e-4 and hf, 2 DMC series), and the decision is made again with the new values" in line
                             for line in self.log), 1)
        self.assertEqual(again.tick(), "RUNNING")                       # 0.17 is above 0.1: NOT_DECIDED, one more series
        auto = self.auto()
        self.assertEqual([(d["verdict"], d["tolerance"], d["action"]) for d in auto["decisions"]],
                         [("CONVERGED", 1.0, "final"), ("NOT_DECIDED", 0.1, "more series")])
        self.assertEqual((auto["series"], self.engine.stopped, len(self.engine.points)), (3, [], 12))
        # a third object with the new input continues exactly: the note is not written again
        third = self.same_again(again)
        self.assertEqual(sum("the values of [trial] changed" in line for line in self.log), 1)
        self.assertEqual(third.tick(), "RUNNING")
        self.assertEqual(len(self.engine.points), 18)
        # back to the old tolerance while the third series runs: the plan of 3 series is kept
        back = self.runs(AUTO)
        self.assertEqual((self.auto()["series"], self.auto()["tolerance"], len(back.keys())), (3, 1.0, 18))
        self.finish(back)
        self.assertEqual(back.tick(), "DONE")
        self.assertEqual((self.auto()["verdict"], self.engine.stopped), ("CONVERGED", []))
        self.assertTrue(back.message.endswith("CONVERGED within 1.0 kcal/mol between the cutoffs 2.5e-4 and 1e-4 (3 series)"))

    def test_a_changed_number_of_series(self):
        # series_max raised after NOT_DECIDED at the old limit: the stage goes on
        self.not_decided()
        runs = self.start(dict(AUTO, series_max=2))
        self.assertEqual(runs.tick(), "DONE")
        self.assertEqual(self.auto()["verdict"], "NOT_DECIDED")
        again = self.runs(AUTO)
        self.assertEqual((again.status, self.auto()["final"], self.auto()["series_max"]), ("RUNNING", False, 4))
        self.assertEqual(again.tick(), "RUNNING")
        self.assertEqual((self.auto()["series"], self.auto()["decisions"][-1]["action"]), (3, "more series"))
        # series raised above the series that the plan reached: every row gets that number at once
        self.again("more_at_the_start")
        runs = self.start()
        self.assertEqual(runs.tick(), "DONE")
        more = self.runs(dict(AUTO, series=3))
        self.assertEqual((self.auto()["series"], len(more.keys()), self.auto()["final"]), (3, 18, False))
        self.assertEqual(more.tick(), "RUNNING")
        self.assertEqual(len(self.engine.points), 18)
        # series_max lowered below the series that the plan reached: no run is stopped, and no series is added
        self.again("lower_limit")
        self.not_decided()
        runs = self.runs()
        for _ in range(8):
            runs.tick(); self.finish(runs)
        self.assertEqual((self.auto()["verdict"], self.auto()["series"]), ("NOT_DECIDED", 4))
        lower = self.runs(dict(AUTO, series_max=3))
        self.assertEqual((self.auto()["series"], len(lower.keys()), self.state()["to_stop"]), (4, 24, []))
        self.assertEqual(lower.tick(), "DONE")
        self.assertEqual((self.auto()["verdict"], self.auto()["series"], self.engine.stopped), ("NOT_DECIDED", 4, []))

    def test_a_longer_ladder_keeps_the_plan_and_decides_again(self):
        self.not_decided()
        runs = self.start()
        self.assertEqual(runs.tick(), "RUNNING")                        # NOT_DECIDED: 3 series
        runs.tick()
        key = "trial/reactant/final/%s_s3" % FINE                       # the third series decides: CONVERGED
        self.energy[key], self.noise[key] = E_CUT, 1.0e-6
        self.finish(runs)
        self.assertEqual(runs.tick(), "DONE")
        self.assertEqual((self.auto()["verdict"], self.auto()["series"], len(self.engine.specs)), ("CONVERGED", 3, 18))
        specs = dict(self.engine.specs)
        decisions = self.auto()["decisions"]
        # one more cutoff at the end of the ladder: the cutoffs of the plan are still the first cutoffs of the ladder,
        # so the plan that was reached is kept (2 cutoffs, 3 series), no run leaves it, and the decision is made again
        again = self.runs(self.LONG)
        self.assertEqual(self.auto(), {"ladder": [2.5e-4, 1e-4, 5e-5], "tolerance": 1.0, "series_max": 4,
                                       "cutoffs": [2.5e-4, 1e-4], "series": 3, "final": False, "verdict": None, "reason": "",
                                       "decisions": decisions})
        self.assertEqual((len(again.keys()), again.status), (18, "RUNNING"))
        self.assertEqual((self.state()["to_stop"], self.state()["left_plan"]), ([], []))
        self.assertEqual(self.state()["rule"]["trial"], self.LONG)
        self.assertEqual(sum("automatic cutoff: the values of [trial] changed; the plan that was reached is kept (cutoffs "
                             "2.5e-4, 1e-4 and hf, 3 DMC series), and the decision is made again with the new values" in line
                             for line in self.log), 1)
        self.assertEqual(again.tick(), "DONE")                          # the same energies give the same verdict
        self.assertEqual((self.engine.specs, self.engine.stopped), (specs, []))
        self.assertEqual([(d["verdict"], d["action"]) for d in self.auto()["decisions"]],
                         [("NOT_DECIDED", "more series"), ("CONVERGED", "final"), ("CONVERGED", "final")])
        # a stage that ended NOT_CONVERGED because its ladder held no further cutoff goes on with the new cutoff
        self.again("ladder_ended")
        self.energy["reactant", "final", FINE] = E_CUT + 3.0 / H        # the change differs by 3.0 kcal/mol
        self.energy["reactant", "final", THIRD] = E_CUT + 3.0 / H       # ... and not between the cutoffs 1e-4 and 5e-5
        runs = self.start()
        self.assertEqual(runs.tick(), "DONE")
        self.assertEqual((self.auto()["verdict"], self.auto()["decisions"][0]["cause"]), ("NOT_CONVERGED", "ladder ended"))
        again = self.runs(self.LONG)
        self.assertEqual((self.auto()["final"], self.auto()["cutoffs"], again.status), (False, [2.5e-4, 1e-4], "RUNNING"))
        self.assertEqual(again.tick(), "RUNNING")
        auto = self.auto()
        self.assertEqual((auto["cutoffs"], auto["series"], auto["decisions"][-1]["action"], auto["decisions"][-1]["added"]),
                         ([2.5e-4, 1e-4, 5e-5], 2, "next cutoff", 5e-5))
        self.assertEqual(again.tick(), "RUNNING")                       # the 4 runs of the new cutoff are asked for
        self.assertEqual((len(self.engine.points), self.engine.stopped), (16, []))
        self.assertEqual(len(self.finish(again)), 4)
        self.assertEqual(again.tick(), "DONE")
        self.assertTrue(again.message.endswith("CONVERGED within 1.0 kcal/mol between the cutoffs 1e-4 and 5e-5 (2 series)"),
                        again.message)
        self.assertEqual([(d["verdict"], d["cause"]) for d in self.auto()["decisions"]],
                         [("NOT_CONVERGED", "ladder ended"), ("NOT_CONVERGED", "not converged"), ("CONVERGED", "converged")])
        self.assertEqual([d["ladder"] for d in self.auto()["decisions"]],
                         [[2.5e-4, 1e-4], [2.5e-4, 1e-4, 5e-5], [2.5e-4, 1e-4, 5e-5]])
        # the first decision as an older code wrote it (no ladder), the later ones with a ladder: a change of [trial]
        # that keeps the plan must not give the first decision the ladder of the record, which is newer than it
        state_file = os.path.join(self.folder, "state.json")
        record = self.state()
        del record["auto"]["decisions"][0]["ladder"]
        TS.write(state_file, json.dumps(record))
        self.runs(dict(self.LONG, tolerance=2.0))
        self.assertEqual([d.get("ladder") for d in self.auto()["decisions"]],
                         [None, [2.5e-4, 1e-4, 5e-5], [2.5e-4, 1e-4, 5e-5]])
        # no decision with a ladder: no decision gets one either (the ladder of the record can be newer than them)
        record = self.state()
        for entry in record["auto"]["decisions"]:
            entry.pop("ladder", None)
        TS.write(state_file, json.dumps(record))
        self.runs(dict(self.LONG, tolerance=3.0))
        self.assertEqual([d.get("ladder") for d in self.auto()["decisions"]], [None] * 3)
        # a ladder that changed only after the cutoffs of the plan is handled in the same way, also while runs write blocks
        self.again("changed_after_the_plan")
        self.not_decided()
        runs = self.start(self.LONG)
        self.assertEqual(runs.tick(), "RUNNING")                        # NOT_DECIDED: the plan has 3 series now
        self.assertEqual(runs.tick(), "RUNNING")                        # the 6 runs of the third series are asked for
        third = [k for k in runs.keys() if k.endswith("_s3")]
        for key in third:                                               # ... and they write blocks
            self.engine.points[key].update(state=RUNNING, energy=np.full(40, E_CUT), message="40 of 120 blocks")
        other = dict(AUTO, auto={"ladder": [2.5e-4, 1e-4, 2e-5]})
        again = self.runs(other)
        self.assertEqual((self.auto()["ladder"], self.auto()["cutoffs"], self.auto()["series"], len(again.keys())),
                         ([2.5e-4, 1e-4, 2e-5], [2.5e-4, 1e-4], 3, 18))
        self.assertEqual(self.state()["to_stop"], [])
        self.assertEqual(again.tick(), "RUNNING")
        self.assertEqual(self.engine.stopped, [])                       # no run is stopped
        self.assertEqual({self.engine.points[k]["state"] for k in third}, {RUNNING})

    def test_a_ladder_that_is_now_shorter_than_the_plan_and_two_values_changed_in_one_step(self):
        self.energy["reactant", "final", FINE] = E_CUT + 3.0 / H        # NOT_CONVERGED between 2.5e-4 and 1e-4
        self.energy["reactant", "final", THIRD] = E_CUT + 3.0 / H       # ... and CONVERGED between 1e-4 and 5e-5
        runs = self.start(self.LONG)
        self.assertEqual(runs.tick(), "RUNNING")                        # the cutoff 5e-5 joins the plan
        self.assertEqual(runs.tick(), "RUNNING")
        self.finish(runs)
        self.assertEqual(runs.tick(), "DONE")
        self.assertEqual((self.auto()["verdict"], self.auto()["cutoffs"]), ("CONVERGED", [2.5e-4, 1e-4, 5e-5]))
        self.assertEqual([d["ladder"] for d in self.auto()["decisions"]], [[2.5e-4, 1e-4, 5e-5]] * 2)
        # the ladder loses its third cutoff: the plan holds a cutoff that the ladder does not hold, so the plan of
        # the input again. The tolerance changes in the same step
        again = self.runs(dict(AUTO, tolerance=5.0))
        auto = self.auto()
        self.assertEqual((auto["ladder"], auto["cutoffs"], auto["series"], auto["decisions"], auto["tolerance"]),
                         ([2.5e-4, 1e-4], [2.5e-4, 1e-4], 2, [], 5.0))
        self.assertEqual(sum("automatic cutoff: [trial] auto_cutoffs changed, and the cutoffs of the plan that was reached "
                             "(2.5e-4, 1e-4, 5e-5) are not the first cutoffs of the new ladder; the plan starts from the input "
                             "again" in line for line in self.log), 1)
        self.assertEqual(sorted(self.state()["left_plan"]), sorted(k for k in runs.keys() if THIRD in k))
        self.assertEqual(again.tick(), "DONE")                          # 3.0 kcal/mol is within the tolerance 5.0
        self.assertEqual((self.auto()["verdict"], self.engine.stopped, len(self.engine.points)), ("CONVERGED", [], 16))
        # a longer ladder and another tolerance in one step: the plan is kept, and the new decision uses both values
        self.again("two_changes")
        self.energy["reactant", "final", FINE] = E_CUT + 3.0 / H
        runs = self.start()
        self.assertEqual(runs.tick(), "DONE")
        self.assertEqual((self.auto()["verdict"], self.auto()["decisions"][0]["cause"]), ("NOT_CONVERGED", "ladder ended"))
        again = self.runs(dict(self.LONG, tolerance=5.0))
        auto = self.auto()
        self.assertEqual((auto["cutoffs"], auto["tolerance"], auto["ladder"], auto["final"]),
                         ([2.5e-4, 1e-4], 5.0, [2.5e-4, 1e-4, 5e-5], False))
        self.assertEqual(again.tick(), "DONE")
        self.assertEqual([(d["verdict"], d["tolerance"], d["ladder"]) for d in self.auto()["decisions"]],
                         [("NOT_CONVERGED", 1.0, [2.5e-4, 1e-4]), ("CONVERGED", 5.0, [2.5e-4, 1e-4, 5e-5])])
        self.assertEqual((self.engine.stopped, len(self.engine.points)), ([], 12))

    def test_a_ladder_with_other_first_cutoffs_starts_from_the_input_and_computes_nothing_again(self):
        self.not_decided()
        runs = self.start()
        self.assertEqual(runs.tick(), "RUNNING")                        # NOT_DECIDED: 3 series
        runs.tick()
        key = "trial/reactant/final/%s_s3" % FINE                       # the third series decides: CONVERGED
        self.energy[key], self.noise[key] = E_CUT, 1.0e-6
        self.finish(runs)
        self.assertEqual(runs.tick(), "DONE")
        self.assertEqual((self.auto()["verdict"], self.auto()["series"], len(self.engine.specs)), ("CONVERGED", 3, 18))
        specs = dict(self.engine.specs)
        old = runs.keys()
        # another second cutoff: the cutoffs of the plan are not the first cutoffs of the new ladder. The plan of the
        # input again (2 cutoffs, 2 series), and no decision of the old ladder
        other = dict(AUTO, cutoffs=[2.5e-4, 5e-5, "hf"], auto={"ladder": [2.5e-4, 5e-5]})
        again = self.runs(other)
        self.assertEqual(self.auto(), {"ladder": [2.5e-4, 5e-5], "tolerance": 1.0, "series_max": 4, "cutoffs": [2.5e-4, 5e-5],
                                       "series": 2, "final": False, "verdict": None, "reason": "", "decisions": []})
        keys = again.keys()
        self.assertEqual(len(keys), 12)
        self.assertEqual(sorted(self.state()["left_plan"]), sorted(k for k in old if k not in keys))
        self.assertEqual(len(self.state()["left_plan"]), 10)            # 6 runs of the cutoff 1e-4 and 4 runs of a third series
        self.assertEqual(sum("automatic cutoff: [trial] auto_cutoffs changed, and the cutoffs of the plan that was reached "
                             "(2.5e-4, 1e-4) are not the first cutoffs of the new ladder; the plan starts from the input "
                             "again; a point that exists is not computed again" in line for line in self.log), 1)
        self.assertEqual(again.tick(), "RUNNING")                       # the 4 runs of the cutoff 5e-5 are asked for
        self.assertEqual(self.engine.stopped, [])                       # the runs that left the plan had ended
        self.assertEqual(len(self.engine.points), 22)
        for key, spec in specs.items():                                 # no run that exists is asked for a second time
            self.assertIs(self.engine.specs[key], spec)
        self.assertEqual(len(self.finish(again)), 4)
        self.assertEqual(again.tick(), "DONE")
        self.assertTrue(again.message.endswith("CONVERGED within 1.0 kcal/mol between the cutoffs 2.5e-4 and 5e-5 (2 series)"),
                        again.message)

    def test_a_changed_n_det_max_starts_from_the_input_and_stops_the_runs_of_the_old_plan(self):
        self.not_decided()
        runs = self.start()
        self.assertEqual(runs.tick(), "RUNNING")                        # NOT_DECIDED: the plan has 3 series now
        self.assertEqual(runs.tick(), "RUNNING")                        # the 6 runs of the third series wait
        old = runs.keys()
        waiting = sorted(k for k in old if k.endswith("_s3"))
        self.assertEqual((len(old), len(waiting), len(self.auto()["decisions"])), (18, 6, 1))
        other = dict(AUTO, n_det_max=50000)
        again = self.runs(other)
        # the plan of the input: 2 series, and no decision of the plan with the other CIPSI wave function
        self.assertEqual(self.auto(), {"ladder": [2.5e-4, 1e-4], "tolerance": 1.0, "series_max": 4, "cutoffs": [2.5e-4, 1e-4],
                                       "series": 2, "final": False, "verdict": None, "reason": "", "decisions": []})
        self.assertEqual(sum("automatic cutoff: [trial] n_det_max changed; the plan starts from the input again" in line
                             for line in self.log), 1)
        keys = again.keys()
        self.assertEqual(sorted(k for k in keys if "/hf" not in k),
                         sorted("trial/reactant/%s/n50000_%s%s" % (g, c, s) for g in ("start", "final")
                                for c in ("c2.5e-04", "c1e-04") for s in ("", "_s2")))
        gone = sorted(k for k in old if k not in keys)                  # the 12 runs of the old cuts and 2 runs of hf
        self.assertEqual(len(gone), 14)
        self.assertEqual(sorted(self.state()["to_stop"]), gone)
        self.assertEqual(self.engine.stopped, [])                       # nothing is stopped before the new plan is asked for
        calls = len(self.engine.calls)
        self.assertEqual(again.tick(), "RUNNING")
        self.assertGreater(len(self.engine.calls), calls)
        self.assertEqual(sorted(self.engine.stopped), waiting)          # only the runs that had not ended are stopped
        for key in keys:
            if "/hf" in key:                                            # the rows hf are the same points: not computed again
                self.assertEqual(self.engine.points[key]["state"], DONE)
                self.assertNotIn(key, self.engine.stopped)
        self.assertEqual({self.engine.specs[k].settings["trial"]["n_det_max"] for k in keys if "/hf" not in k}, {50000})
        self.assertEqual({self.engine.specs[k].settings["trial"]["qp2_key"] for k in keys if "/hf" not in k},
                         {"trial/reactant/start/qp2_n50000", "trial/reactant/final/qp2_n50000"})
        # no run goes on that no plan holds
        self.assertEqual({self.engine.points[k]["state"] for k in self.engine.points if k not in keys}, {DONE, STOPPED})
        self.assertEqual({self.engine.points[k]["state"] for k in waiting}, {STOPPED})

    def test_a_ladder_with_other_first_cutoffs_while_added_runs_write_blocks_stops_them_and_the_stage_needs_attention(self):
        """The plan of the input does not hold the third series, so its runs are stopped, as in the manual mode. A
        later decision can add the same runs to the plan again, and a stopped run is not continued. The documents
        say this, and they say that such a change is safe only while no run of the stage is running or waiting."""
        self.not_decided()
        self.energy["reactant", "final", THIRD] = E_CUT + 1.0 / H       # NOT_DECIDED also with the cutoff 5e-5
        runs = self.start()
        self.assertEqual(runs.tick(), "RUNNING")                        # NOT_DECIDED: the plan has 3 series now
        self.assertEqual(runs.tick(), "RUNNING")                        # the 6 runs of the third series are asked for
        third = [k for k in runs.keys() if k.endswith("_s3")]
        for key in third:                                               # ... and they write blocks
            self.engine.points[key].update(state=RUNNING, energy=np.full(40, E_CUT), message="40 of 120 blocks")
        self.assertEqual(runs.tick(), "RUNNING")
        self.assertTrue(runs.message.startswith("12 of 18 DMC run(s) are DONE, 6 write blocks, 0 wait;"), runs.message)
        # another second cutoff: the plan of the input has the cutoffs 2.5e-4 and 5e-5 with 2 series. It holds neither
        # the cutoff 1e-4 nor a third series
        other = dict(AUTO, cutoffs=[2.5e-4, 5e-5, "hf"], auto={"ladder": [2.5e-4, 5e-5]})
        again = self.runs(other)
        self.assertEqual((self.auto()["series"], self.auto()["cutoffs"], len(again.keys())), (2, [2.5e-4, 5e-5], 12))
        self.assertTrue(set(third) <= set(self.state()["to_stop"]))
        self.assertEqual(again.tick(), "RUNNING")                       # the 4 runs of the cutoff 5e-5 are asked for
        self.assertEqual(sorted(self.engine.stopped), sorted(third))    # only the runs that had not ended are stopped
        self.assertEqual({self.engine.points[k]["state"] for k in third}, {STOPPED})
        self.assertEqual(sum("stopped, because the plan of the stage no longer holds it" in line for line in self.log), 6)
        self.assertEqual(len(self.finish(again)), 4)
        # the decision is NOT_DECIDED, so the plan gets a third series. Its runs for the cutoff 2.5e-4 and for hf are
        # the runs that were stopped. The message does not count them as waiting
        again_third = [k for k in third if FINE not in k]
        self.assertEqual(again.tick(), "RUNNING")
        self.assertEqual((self.auto()["series"], self.auto()["decisions"][0]["verdict"]), (3, "NOT_DECIDED"))
        self.assertEqual(again.message, (
            "12 of 18 DMC run(s) are DONE, 0 write blocks, 2 wait; 4 run(s) that the plan holds again ended earlier as "
            "FAILED or STOPPED and need attention: %s; automatic cutoff: verdict NOT_DECIDED with 2 series, so one more DMC "
            "series is added for every row (3 series now, at most 4), because the errors are too large to decide"
            % ", ".join(again_third)))
        # the next passes: the 2 new runs of the cutoff 5e-5 are asked for and end. A stopped run is not continued, so
        # a person must look. The message names every such run
        self.assertEqual(again.tick(), "RUNNING")
        self.assertEqual(len(self.finish(again)), 2)
        self.assertEqual(again.tick(), "NEEDS_ATTENTION")
        self.assertEqual(again.message, "; ".join("%s: stopped after 40 of 120 blocks; the stage needs every run complete" % k
                                                 for k in again_third))
        self.assertEqual(len(self.auto()["decisions"]), 1)              # no decision is made from stopped runs
        self.assertEqual(again.retry(), [])                             # retry is for failed runs only
        self.assertEqual(again.tick(), "NEEDS_ATTENTION")
        # the person deletes the folders of the stopped runs: they are asked for again and start from their first block
        for key in again_third:
            del self.engine.points[key], self.engine.specs[key]
        self.assertEqual(again.tick(), "RUNNING")
        self.assertEqual({(self.engine.points[k]["state"], len(self.engine.points[k]["energy"])) for k in again_third}, {(QUEUED, 0)})
        key = "trial/reactant/final/%s_s3" % COARSE                     # a precise third series decides: the change
        self.energy[key], self.noise[key] = E_CUT + 1.0 / H, 1.0e-6     # is 1.0 kcal/mol at both cutoffs, so CONVERGED
        self.assertEqual(len(self.finish(again)), 4)
        self.assertEqual(again.tick(), "DONE")
        auto = self.auto()
        self.assertEqual((auto["verdict"], auto["series"], auto["cutoffs"]), ("CONVERGED", 3, [2.5e-4, 5e-5]))

    def test_runs_that_the_record_calls_absent_are_stopped_too(self):
        """The record names the runs of a larger plan one pass before the stage asks the engine for them. A pass that
        ends between its requests and its record leaves runs in the queue that the record calls ABSENT. They must
        not go on when the plan no longer holds them, or when the option is switched off."""
        def killed():
            raise OSError("the driver job reached its time limit")

        for case in ("another plan", "switched off"):
            self.again(case.replace(" ", "_"))
            self.not_decided()
            runs = self.start()
            self.assertEqual(runs.tick(), "RUNNING")                    # NOT_DECIDED: the plan has 3 series now
            new = sorted(k for k in runs.keys() if k.endswith("_s3"))
            self.assertEqual({self.state()["points"][k]["state"] for k in new}, {"ABSENT"})
            runs._save = killed
            with self.assertRaises(OSError):
                runs.tick()                                             # the 6 runs are asked for; the record is not written
            self.assertEqual({self.engine.points[k]["state"] for k in new}, {QUEUED})
            self.assertEqual({self.state()["points"][k]["state"] for k in new}, {"ABSENT"})
            if case == "another plan":                                  # the input is now a list of the same cutoffs: 2 series
                back = self.runs(dict(TS.TRIAL_DICT, n_det_max=100000, cutoffs=[2.5e-4, 1e-4, "hf"]))
                self.assertEqual(sorted(self.state()["to_stop"]), new)
                self.assertEqual(back.tick(), "DONE")
            else:                                                       # what the workflow does when [trial] kind is single
                T.stop_points(self.engine, self.state(), log=self.log.append, why="the option was switched off")
            self.assertEqual(sorted(self.engine.stopped), new, case)
            self.assertEqual({self.engine.points[k]["state"] for k in new}, {STOPPED}, case)
            # only a run that a pass saw running or waiting gets a line in the log
            self.assertFalse([line for line in self.log if ": stopped" in line], case)
        # a run that the engine does not hold is asked to stop as well, and nothing happens
        self.again("not_asked_for")
        self.not_decided()
        runs = self.start()
        self.assertEqual(runs.tick(), "RUNNING")
        new = sorted(k for k in runs.keys() if k.endswith("_s3"))
        self.assertEqual(sorted(T.stop_points(self.engine, self.state())), new)
        self.assertEqual(sorted(self.engine.points), sorted(k for k in runs.keys() if k not in new))

    def test_from_the_manual_mode_to_the_automatic_mode_and_back(self):
        manual = dict(TS.TRIAL_DICT, n_det_max=100000, cutoffs=[1e-3, 2.5e-4, "hf"])       # a dictionary of version 0.1.0
        runs = self.runs(manual)
        self.assertFalse(runs.automatic)
        runs.tick()
        short = sorted(k for k in runs.keys() if "c1e-03" in k)
        for key in runs.keys():                                         # the cut 1e-3 still runs; the other rows have ended
            if key not in short:
                target, geometry, label, _n = split(key)
                self.engine.finish(key, mean=E_HF if label == "hf" else E_CUT)
        runs.tick()
        self.assertNotIn("auto", self.state())

        def nothing_runs_outside(plan):
            return {self.engine.points[k]["state"] for k in self.engine.points if k not in plan.keys()} <= {DONE, STOPPED, FAILED}

        # ---- to the automatic mode: the rows of 2.5e-4 and hf are in its plan, the cut 1e-3 is not
        auto_runs = self.runs(AUTO)
        self.assertEqual(self.auto()["decisions"], [])
        self.assertEqual(sum("automatic cutoff: the record was written by the manual mode" in line for line in self.log), 1)
        self.assertEqual(sorted(self.state()["to_stop"]), short)
        kept = {k: self.engine.points[k]["energy"].copy() for k in auto_runs.keys() if k in self.engine.points}
        self.assertEqual(len(kept), 8)
        self.assertEqual(auto_runs.tick(), "RUNNING")
        self.assertEqual(sorted(self.engine.stopped), short)            # stopped after the new plan was asked for
        self.assertTrue(nothing_runs_outside(auto_runs))
        for key, energy in kept.items():                                # no point was lost or computed again
            np.testing.assert_array_equal(self.engine.points[key]["energy"], energy)
        self.assertEqual(sorted(k for k in self.engine.points if k not in kept and k not in short),
                         sorted("trial/reactant/%s/%s%s" % (g, FINE, s) for g in ("start", "final") for s in ("", "_s2")))
        self.finish(auto_runs)
        self.assertEqual(auto_runs.tick(), "DONE")
        self.assertEqual(self.auto()["verdict"], "CONVERGED")
        points = len(self.engine.points)
        # ---- back to the manual mode: the record has the fields of the manual mode, and its runs exist
        back = self.runs(dict(manual, cutoffs=[1e-4, "hf"]))
        state = self.state()
        self.assertNotIn("auto", state)
        self.assertNotIn("cutoff_rule", state["rule"])
        self.assertEqual(back.tick(), "DONE")
        self.assertEqual(back.message, "all 8 DMC run(s) are DONE")
        self.assertEqual(len(self.engine.points), points)
        self.assertTrue(nothing_runs_outside(back))
        # ---- and to the automatic mode again: it starts from the input, and every run of its plan is on disk
        auto_again = self.runs(AUTO)
        self.assertEqual((self.auto()["decisions"], self.auto()["final"]), ([], False))
        self.assertEqual(auto_again.tick(), "DONE")
        self.assertEqual((self.auto()["verdict"], len(self.engine.points)), ("CONVERGED", points))

    def test_the_automatic_mode_switched_off_while_added_runs_wait(self):
        self.not_decided()
        runs = self.start()
        self.assertEqual(runs.tick(), "RUNNING")                        # 3 series
        self.assertEqual(runs.tick(), "RUNNING")                        # the 6 new runs wait in the queue
        waiting = sorted(k for k in runs.keys() if k.endswith("_s3"))
        manual = dict(TS.TRIAL_DICT, n_det_max=100000, cutoffs=[2.5e-4, 1e-4, "hf"])
        back = self.runs(manual)
        self.assertEqual(len(back.keys()), 12)
        self.assertEqual(back.tick(), "DONE")                           # the 12 runs of the manual plan had ended
        self.assertEqual(sorted(self.engine.stopped), waiting)          # and the runs that only the automatic plan held stop
        self.assertEqual({self.engine.points[k]["state"] for k in waiting}, {STOPPED})
        self.assertEqual(back.message, "all 12 DMC run(s) are DONE")

    def test_a_record_that_does_not_fit_its_ladder_starts_from_the_input(self):
        runs = self.start()
        self.assertEqual(runs.tick(), "DONE")
        state = self.state()
        state["auto"]["cutoffs"] = [1e-4, 2.5e-4]                       # not the first cutoffs of the ladder
        TS.write(os.path.join(self.folder, "state.json"), json.dumps(state))
        again = self.runs()
        self.assertEqual((self.auto()["cutoffs"], self.auto()["decisions"], self.auto()["final"]), ([2.5e-4, 1e-4], [], False))
        self.assertEqual(sum("automatic cutoff: the plan of the record does not fit its ladder; the plan starts from the "
                             "input again" in line for line in self.log), 1)
        self.assertEqual(again.tick(), "DONE")
        self.assertEqual(self.auto()["verdict"], "CONVERGED")
        # a record whose entry holds no usable numbers is handled in the same way
        state = self.state()
        state["auto"].update(series="many", final=True)
        TS.write(os.path.join(self.folder, "state.json"), json.dumps(state))
        again = self.runs()
        self.assertEqual((self.auto()["series"], self.auto()["final"], self.auto()["decisions"]), (2, False, []))
        self.assertEqual(again.tick(), "DONE")

    def test_wrong_values_of_the_automatic_mode_are_refused(self):
        for bad in (dict(AUTO, auto={"ladder": [2.5e-4]}, cutoffs=[2.5e-4, "hf"]),
                    dict(AUTO, auto={"ladder": [1e-4, 2.5e-4]}, cutoffs=[1e-4, 2.5e-4, "hf"]),
                    dict(AUTO, auto={"ladder": [2.5e-4, 2.5e-4]}, cutoffs=[2.5e-4, 2.5e-4, "hf"]),
                    dict(AUTO, auto={"ladder": [2.5e-4, 1.5]}, cutoffs=[2.5e-4, 1.5, "hf"]),
                    # two cutoffs that agree in their first 6 significant digits have one name and one folder
                    dict(AUTO, auto={"ladder": [1.0000001e-4, 1e-4]}, cutoffs=[1.0000001e-4, 1e-4, "hf"]),
                    dict(AUTO, auto={"ladder": [2.5e-4, 1.0000001e-4, 1e-4]}, cutoffs=[2.5e-4, 1.0000001e-4, "hf"]),
                    dict(AUTO, auto={}), dict(AUTO, auto={"ladder": "2.5e-4"}),
                    dict(AUTO, cutoffs=[2.5e-4, "hf"]),                # not the first plan of this ladder
                    dict(AUTO, cutoffs=[2.5e-4, 1e-4]),
                    dict(AUTO, series=3, series_max=2), dict(AUTO, tolerance=0.0), dict(AUTO, tolerance=float("inf")),
                    dict(AUTO, n_det_max=None)):
            with self.assertRaises(ValueError, msg=str(bad)):
                self.runs(bad)
        with self.assertRaises(ValueError) as ctx:                      # the input reader refuses the same ladder
            self.runs(dict(AUTO, auto={"ladder": [1.0000001e-4, 1e-4]}, cutoffs=[1.0000001e-4, 1e-4, "hf"]))
        self.assertIn("a ladder whose cutoffs differ in their first 6 significant digits", str(ctx.exception))
        self.assertFalse(os.path.exists(os.path.join(self.folder, "state.json")))      # nothing was written


# ===================================================================================== the manual mode
class TestManualMode(StageCase):
    OLD = dict(TS.TRIAL_DICT, n_det_max=100000, cutoffs=[2.5e-4, 1e-4, "hf"], not_converged="stop")   # version 0.1.0
    NEW = dict(OLD, tolerance=1.0, series_max=2, auto=None)             # what Config.trial holds for the same input
    STATE_KEYS = ["left_plan", "message", "points", "refusals", "rule", "status", "stop_seen", "symbols", "targets", "to_stop",
                  "version"]
    ROW_KEYS = ["blocks", "blocks_used", "checks", "cost", "cutoff", "done", "energy", "error", "geometry", "keys", "label",
                "messages", "n_det_max", "series", "series_planned", "states", "target", "trial"]

    def test_the_dictionary_of_the_input_is_the_manual_dictionary(self):
        cfg = TS.load(TS.model_input("kind = multidet\nn_det_max = 100000\ncutoff = 2.5e-4, 1e-4, hf\n"))
        self.assertEqual(cfg.trial, self.NEW)

    def test_requests_record_and_messages_are_those_of_version_0_1_0(self):
        found = {}
        for name, trial in (("old", self.OLD), ("new", self.NEW)):
            self.again(name)
            self.energy["reactant", "start", COARSE] = E_HF + 1.0e-2   # far above hf, and the differences of the two
            self.energy["reactant", "final", FINE] = E_CUT - 5.0 / H    # cutoffs disagree by 5 kcal/mol
            runs = self.runs(trial)
            self.assertFalse(runs.automatic)
            self.assertEqual(runs.tick(), "RUNNING")
            self.assertEqual(runs.message, "0 of 12 DMC run(s) are DONE, 0 write blocks, 12 wait")
            self.finish(runs)
            self.assertEqual(runs.tick(), "DONE")                       # no verdict, no growth: the plan is the input
            self.assertEqual((runs.status, runs.message), ("DONE", "all 12 DMC run(s) are DONE"))
            for _ in range(3):
                self.assertEqual(runs.tick(), "DONE")
            self.assertEqual(len(self.engine.points), 12)
            state = self.state()
            self.assertEqual(sorted(state), self.STATE_KEYS)            # no entry "auto"
            self.assertEqual(sorted(state["rule"]), ["blocks", "energy", "same_geometry_angstrom", "text", "trial"])
            self.assertEqual(state["rule"]["trial"], trial)
            self.assertFalse([line for line in self.log if "automatic" in line])
            res = runs.results()
            self.assertEqual(sorted(res), ["cost_left_plan", "left_plan", "rows", "rule", "targets"])
            self.assertEqual(sorted(res["rows"][0]), self.ROW_KEYS)
            found[name] = (runs.keys(), list(self.engine.calls), {k: s.settings for k, s in self.engine.specs.items()},
                           {k: (s.series, s.base_key, s.blocks, s.role) for k, s in self.engine.specs.items()},
                           {k: v for k, v in state.items() if k != "rule"},
                           [line.split("  [trial] ", 1)[1] for line in self.log])
        self.assertEqual(found["old"], found["new"])                    # the three new keys change nothing of the runs
        keys = found["new"][0]
        self.assertEqual(keys, ["trial/reactant/%s/%s%s" % (g, label, s) for g in ("start", "final")
                                for label in (COARSE, FINE, "hf") for s in ("", "_s2")])
        self.assertEqual(found["new"][1][:12], keys)                    # the order of the requests

    def test_a_record_of_version_0_1_0_is_read_and_nothing_is_computed_again(self):
        runs = self.start(self.OLD)                                     # the record as version 0.1.0 wrote it
        self.assertEqual(runs.tick(), "DONE")
        record = self.state()
        self.assertEqual(sorted(record["rule"]["trial"]), sorted(list(TS.TRIAL_DICT) + ["not_converged"]))
        self.assertNotEqual(T.stored_trial(self.folder), self.NEW)      # so the workflow sees a changed input
        specs, lines = dict(self.engine.specs), len(self.log)
        new = self.runs(self.NEW)
        self.assertEqual((new.status, new.message), ("DONE", "all 12 DMC run(s) are DONE"))
        self.assertEqual(new.tick(), "DONE")
        self.assertEqual(self.engine.specs, specs)                      # the same 12 points, none of them new
        self.assertEqual((self.engine.stopped, self.state()["to_stop"], self.state()["left_plan"]), ([], [], []))
        self.assertEqual(T.stored_trial(self.folder), self.NEW)         # the record now holds the values of the input
        self.assertEqual(len(self.log), lines)
        # the same record, read by the automatic mode with the same cutoffs and n_det_max: its 12 runs are the first plan
        self.again("to_auto")
        runs = self.start(self.OLD)
        self.assertEqual(runs.tick(), "DONE")
        specs = dict(self.engine.specs)
        auto = self.runs(AUTO)
        self.assertEqual((auto.status, auto.message), ("RUNNING", ""))  # every run is DONE, but nothing is decided yet
        self.assertEqual(auto.tick(), "DONE")
        self.assertEqual(self.engine.specs, specs)
        self.assertEqual(self.auto()["verdict"], "CONVERGED")

    def test_a_run_that_is_done_but_whose_request_is_refused_needs_attention(self):
        # the manual mode has the same check: the energy of a refused point is not the energy of the row of the plan
        runs = self.start(self.OLD)
        self.assertEqual(runs.tick(), "DONE")
        key = "trial/reactant/start/hf"
        self.engine.refuse[key] = "point %s exists with another geometry" % key
        self.assertEqual(runs.tick(), "NEEDS_ATTENTION")
        self.assertEqual(runs.message, "%s: the point could not be submitted: point %s exists with another geometry; %s_s2: "
                                       "its series-1 point %s could not be submitted" % (key, key, key, key))
        del self.engine.refuse[key]
        self.assertEqual(runs.tick(), "DONE")
        self.assertEqual(runs.message, "all 12 DMC run(s) are DONE")


# ===================================================================================== check and the machine profile
class Profile:
    """What ``machine_problems`` asks a machine profile."""

    def __init__(self, ranks, lacks=()):
        self.ranks, self.lacks = ranks, list(lacks)

    def check_qp2(self):
        return list(self.lacks)

    def tasks(self, kind):
        return self.ranks


class TestPlanAndProfile(unittest.TestCase):
    def test_plan_lines_of_the_automatic_mode(self):
        lines = T.plan_lines(AUTO, ["reactant", "saddle"])
        self.assertEqual(lines, [
            "stage trial (after the geometry search): DMC energies with several determinants from Quantum Package 2",
            "  geometries: up to 4 (the start geometry and the final geometry of 2 target(s); a final geometry that equals "
            "its start geometry is not computed again)",
            "  Quantum Package parent calculations: up to 4, one per geometry (RHF orbitals and a CIPSI wave function of "
            "100000 determinants, used by all cutoffs of the geometry; basis file cc-pvtz_ecp_ccecp, pseudopotential file ccecp)",
            "  cutoffs: the automatic mode runs ([trial] cutoff = auto). It starts with the cutoffs 2.5e-4, 1e-4 and hf. Its "
            "ladder is 2.5e-4, 1e-4 ([trial] auto_cutoffs)",
            "  rule: each cut is tested against hf, and every energy difference is tested between two cutoffs with the "
            "tolerance 1.0 kcal/mol ([trial] tolerance). The stage adds DMC series or the next cutoff of the ladder when the "
            "test needs them",
            "  DMC series: 2 at the start and at most 4 for each geometry and each cutoff ([trial] series and series_max)",
            "  DMC runs: up to 24 at the start (3 cutoffs with hf, 2 series) and at most 48 (3 cutoffs with hf, 4 series)",
        ])
        # a longer ladder, other numbers of series, one target, another tolerance
        long = dict(AUTO, auto={"ladder": [1e-3, 5e-4, 2.5e-4, 1e-4]}, cutoffs=[1e-3, 5e-4, "hf"], series=3, series_max=5,
                    tolerance=0.25)
        text = "\n".join(T.plan_lines(long, ["geometry"]))
        for part in ("It starts with the cutoffs 1e-3, 5e-4 and hf. Its ladder is 1e-3, 5e-4, 2.5e-4, 1e-4",
                     "the tolerance 0.25 kcal/mol", "DMC series: 3 at the start and at most 5",
                     "DMC runs: up to 18 at the start (3 cutoffs with hf, 3 series) and at most 50 (5 cutoffs with hf, 5 series)"):
            self.assertIn(part, text)

    def test_plan_lines_of_the_manual_mode_are_those_of_version_0_1_0(self):
        expected = [
            "stage trial (after the geometry search): DMC energies with several determinants from Quantum Package 2",
            "  geometries: up to 4 (the start geometry and the final geometry of 2 target(s); a final geometry that equals "
            "its start geometry is not computed again)",
            "  Quantum Package parent calculations: up to 4, one per geometry (RHF orbitals and a CIPSI wave function of "
            "10000 determinants, used by all cutoffs of the geometry; basis file cc-pvtz_ecp_ccecp, pseudopotential file ccecp)",
            "  cutoffs: 2 (0.001, hf)",
            "  DMC series: 2 for each geometry and each cutoff, up to 16 DMC run(s) in total",
        ]
        self.assertEqual(T.plan_lines(TS.TRIAL_DICT, ["reactant", "saddle"]), expected)
        self.assertEqual(T.plan_lines(dict(TS.TRIAL_DICT, tolerance=1.0, series_max=2, auto=None), ["reactant", "saddle"]), expected)

    def test_the_profile_must_allow_the_largest_number_of_series(self):
        # 256 MPI ranks: at most 2 DMC runs of one geometry have random numbers of their own
        found = T.machine_problems(Profile(256), AUTO)
        self.assertEqual(found, ["[trial] series_max = 4, but the QMC jobs of this profile use 256 MPI ranks ([job.qmc] tasks), "
                                 "and then at most 2 DMC runs of one geometry can have random numbers of their own"])
        self.assertEqual(T.machine_problems(Profile(256), dict(AUTO, series_max=2)), [])
        self.assertEqual(T.machine_problems(Profile(32), AUTO), [])     # 16 runs fit on 32 ranks
        self.assertEqual(len(T.machine_problems(Profile(32), dict(AUTO, series_max=17))), 1)
        self.assertEqual(T.machine_problems(Profile(32, ["section [job.qp2] is missing"]), AUTO), ["section [job.qp2] is missing"])
        # the manual mode tests series, as before, whatever series_max holds
        manual = dict(TS.TRIAL_DICT, tolerance=1.0, series_max=2, auto=None)
        self.assertEqual(T.machine_problems(Profile(256), manual), [])
        self.assertEqual(T.machine_problems(Profile(256), dict(manual, series_max=9)), [])
        found = T.machine_problems(Profile(256), dict(manual, series=3, series_max=3))
        self.assertEqual(len(found), 1)
        self.assertIn("[trial] series = 3, but the QMC jobs", found[0])


# ===================================================================================== the stage in the workflow
class Scripted(Exact):
    """The stand-in engine whose points end by themselves, one pass after their request. The energy and the
    amplitude of a run come from two dictionaries of the test: by key, else by row, else the default of its kind."""

    def __init__(self, inner, energy, noise):
        super().__init__(inner, auto=1)
        self.energy, self.noise = energy, noise

    def advance(self):
        self.advances += 1
        if self.inner is not None:
            self.inner.advance()
        for key, p in self.points.items():
            p["age"] += 1
            if p["state"] == QUEUED and p["age"] >= self.auto:
                target, geometry, label, _series = split(key)
                row = (target, geometry, label)
                self.finish(key, mean=self.energy.get(key, self.energy.get(row, E_HF if label == "hf" else E_CUT)),
                            noise=self.noise.get(key, self.noise.get(row, 1.0e-3)))


class ScriptedBackend(ModelBackend):
    """The backend of a model project whose stage ``trial`` talks to :class:`Scripted`."""

    def __init__(self, cfg, energy=None, noise=None, engine=None):
        super().__init__(cfg)
        self.energy, self.noise, self.stand_in = energy or {}, noise or {}, engine

    def engine(self, wf):
        if self.stand_in is None:
            self.stand_in = Scripted(super().engine(wf), self.energy, self.noise)
        return self.stand_in


AUTO_INPUT = TS.WATER + "[report]\ngif = no\n[trial]\nkind = multidet\ncutoff = auto\n"
WATER_FINE = "trial/geometry/final/%s" % FINE


class TestWorkflowAuto(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.folder = tmp.name
        self.input = os.path.join(self.folder, "input.ini")
        # the energy change of the target is 1.0 kcal/mol larger at the fine cutoff in the first two series (NOT_DECIDED),
        # and a precise third series of that row shows no such shift (CONVERGED)
        self.energy = {("geometry", "final", FINE): E_CUT + 1.0 / H, WATER_FINE + "_s3": E_CUT}
        self.noise = {WATER_FINE + "_s3": 1.0e-6}

    def workflow(self, text, engine=None):
        TS.write(self.input, text)
        return Workflow(self.folder, backend=ScriptedBackend(Config.load(self.input), self.energy, self.noise, engine))

    def project(self):
        return json.loads(TS.read(os.path.join(self.folder, "work", "project.json")))

    def record(self):
        return json.loads(TS.read(os.path.join(self.folder, "work", "trial", "state.json")))

    def command(self, *argv):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = cli.main(list(argv) + [self.folder])
        return code, out.getvalue()

    def test_the_stage_grows_its_plan_and_ends_with_a_verdict(self):
        wf = self.workflow(AUTO_INPUT)
        self.assertEqual(wf.cfg.trial, dict(AUTO_DEFAULT))
        status, passes = TS.advance_to_end(wf)
        self.assertEqual(status, "DONE", self.project()["stages"])
        engine = wf.backend.engine(wf)
        verdict = "CONVERGED within 1.0 kcal/mol between the cutoffs 2.5e-4 and 1e-4 (3 series)"
        stages = self.project()["stages"]
        self.assertEqual(stages["trial"], {"status": "DONE", "message": "all 18 DMC run(s) are DONE; automatic cutoff: " + verdict})
        self.assertEqual({s["status"] for s in stages.values()}, {"DONE"})
        # 2 geometries (the search moved the target), 2 cutoffs and hf, 3 series; every run has the length of a final run
        self.assertEqual(sorted(engine.specs), sorted("trial/geometry/%s/%s%s" % (g, label, s) for g in ("start", "final")
                                                      for label in (COARSE, FINE, "hf") for s in ("", "_s2", "_s3")))
        self.assertEqual({spec.blocks for spec in engine.specs.values()}, {400})
        self.assertEqual({spec.settings["jastrow_reuse"] for spec in engine.specs.values()}, {"none"})
        record = self.record()
        self.assertEqual(record["rule"]["trial"], wf.cfg.trial)         # so a new process sees no changed input
        self.assertEqual(record["rule"]["cutoff_rule"]["kinds"], {"geometry": "minimum"})      # from the project state
        auto = record["auto"]
        self.assertEqual((auto["final"], auto["verdict"], auto["series"], auto["cutoffs"]), (True, "CONVERGED", 3, [2.5e-4, 1e-4]))
        self.assertEqual([(d["verdict"], d["action"], d["cause"], d["series"]) for d in auto["decisions"]],
                         [("NOT_DECIDED", "more series", "not decided", 2), ("CONVERGED", "final", "converged", 3)])
        first = auto["decisions"][0]["differences"][0]
        self.assertEqual(first["name"], "final minus start of geometry")
        self.assertAlmostEqual(first["S"], 1.0, places=8)
        self.assertAlmostEqual(first["e"], 2.0 * 1.0e-3 / math.sqrt(389.0) / math.sqrt(2.0) * H, places=9)
        # the stage ran in passes, with engine.advance() in every pass in which it was RUNNING
        running = [p for p in passes if p[1] == "RUNNING"]
        self.assertGreaterEqual(len(running), 3)                        # decision 1, the new runs, decision 2
        self.assertTrue(all(p[2] == 1 for p in running), passes)
        # the log: each decision once
        log = TS.read(os.path.join(self.folder, "work", "driver.log"))
        self.assertEqual(log.count("[trial] automatic cutoff, decision 1: "), 1)
        self.assertEqual(log.count("[trial] automatic cutoff, decision 2: "), 1)
        self.assertNotIn("decision 3", log)
        self.assertIn("stage trial: DONE (all 18 DMC run(s) are DONE; automatic cutoff: %s)" % verdict, log)
        # the command status and the summary of the report show the verdict
        code, text = self.command("status")
        self.assertEqual(code, 0)
        self.assertIn("automatic cutoff: " + verdict, text)
        summary = TS.read(os.path.join(self.folder, "results", "summary.md"))
        self.assertIn("| trial | DONE | all 18 DMC run(s) are DONE; automatic cutoff: %s |" % verdict, summary)
        # the report of the stage has the three sections, in this order, after the DMC energies
        text = TS.read(os.path.join(self.folder, "results", "trial_wavefunctions.md"))
        places = [text.index(title) for title in (
            "## DMC energies", "## DMC energy against the single RHF determinant",
            "## Convergence with the size of the expansion", "## Automatic choice of the cutoff",
            "## Energy change of every target: final geometry minus start geometry")]
        self.assertEqual(places, sorted(places))
        self.assertIn("Final verdict: CONVERGED within 1.0 kcal/mol between the cutoffs 0.00025 and 0.0001 (3 series).", text)
        with open(os.path.join(self.folder, "results", "trial_wavefunctions.csv")) as fh:
            rows = list(csv.DictReader(fh))
        self.assertEqual([r["row"] for r in rows].count("against hf"), 4)
        self.assertEqual([(r["target"], r["geometry"], r["verdict"], r["project_verdict"], r["automatic_final_verdict"])
                          for r in rows if r["row"] == "convergence"],
                         [("geometry", "final minus start", "CONVERGED", "CONVERGED", "CONVERGED")])

        # ---- a new process with the same input: the stage is DONE, and it is not opened again
        wf = self.workflow(AUTO_INPUT)
        self.assertEqual(self.project()["stages"]["trial"]["status"], "DONE")
        self.assertEqual(wf.advance(), "DONE")
        self.assertEqual(wf.backend.engine(wf).calls, [])

        # ---- a tighter tolerance: the input changed, the plan is kept, and the stage decides again
        wf = self.workflow(AUTO_INPUT + "tolerance = 0.01\n", engine=engine)
        stages = self.project()["stages"]
        self.assertEqual(stages["trial"], {"status": "PENDING", "message": "the values of [trial] changed"})
        self.assertEqual(stages["report"]["status"], "PENDING")
        specs = dict(engine.specs)
        status, _passes = TS.advance_to_end(wf)
        self.assertEqual(status, "DONE", self.project()["stages"])
        self.assertEqual({k: v for k, v in engine.specs.items() if k in specs}, specs)     # no run was computed again
        self.assertEqual(len(engine.specs), 24)                         # a fourth series: the largest number
        message = self.project()["stages"]["trial"]["message"]
        self.assertTrue(message.startswith("all 24 DMC run(s) are DONE; automatic cutoff: NOT_DECIDED between the cutoffs "
                                           "2.5e-4 and 1e-4 with the tolerance 0.01 kcal/mol (4 series): final minus start of "
                                           "geometry changed by "), message)
        self.assertTrue(message.endswith("the tolerance of 0.01 kcal/mol; no further series is allowed ([trial] series_max = "
                                         "4), and the errors are too large for the verdict CONVERGED also with a smaller "
                                         "cutoff (2 x sqrt(2) x the error at the fine cutoff is larger than the tolerance "
                                         "for final minus start of geometry (0.04 kcal/mol)): to continue, raise [trial] "
                                         "series_max above 4"), message)
        auto = self.record()["auto"]
        self.assertEqual([(d["tolerance"], d["verdict"], d["action"]) for d in auto["decisions"]],
                         [(1.0, "NOT_DECIDED", "more series"), (1.0, "CONVERGED", "final"),
                          (0.01, "NOT_DECIDED", "more series"), (0.01, "NOT_DECIDED", "final")])
        text = TS.read(os.path.join(self.folder, "results", "trial_wavefunctions.md"))
        self.assertIn("Final verdict: NOT_DECIDED between the cutoffs 0.00025 and 0.0001 with the tolerance 0.01 kcal/mol "
                      "(4 series): ", text)
        self.assertIn("\n4. ", text)                                    # the four decisions, in order
        code, text = self.command("status")
        self.assertIn("automatic cutoff: NOT_DECIDED between the cutoffs 2.5e-4 and 1e-4", text)

    def test_a_project_of_version_0_1_0_and_a_switch_to_the_automatic_mode(self):
        manual = TS.WATER + "[report]\ngif = no\n" + TS.TRIAL % "1e-3, hf"
        self.energy.clear(); self.noise.clear()
        wf = self.workflow(manual)
        status, _passes = TS.advance_to_end(wf)
        self.assertEqual(status, "DONE", self.project()["stages"])
        engine = wf.backend.engine(wf)
        self.assertEqual(self.project()["stages"]["trial"]["message"], "all 8 DMC run(s) are DONE")   # the manual mode
        self.assertNotIn("auto", self.record())
        # ---- make the record the one of version 0.1.0: its values of [trial] hold none of the three new keys
        path = os.path.join(self.folder, "work", "trial", "state.json")
        record = self.record()
        for key in ("tolerance", "series_max", "auto"):
            del record["rule"]["trial"][key]
        TS.write(path, json.dumps(record))
        specs = dict(engine.specs)
        wf = self.workflow(manual, engine=engine)
        stages = self.project()["stages"]
        self.assertEqual(stages["trial"], {"status": "PENDING", "message": "the values of [trial] changed"})
        self.assertEqual((stages["report"]["status"], self.project()["status"]), ("PENDING", "RUNNING"))
        status, _passes = TS.advance_to_end(wf)
        self.assertEqual(status, "DONE", self.project()["stages"])
        self.assertEqual(engine.specs, specs)                           # every point existed: nothing was computed again
        self.assertEqual(engine.stopped, [])
        self.assertEqual(self.record()["rule"]["trial"], wf.cfg.trial)
        self.assertEqual(self.project()["stages"]["trial"]["message"], "all 8 DMC run(s) are DONE")
        text = TS.read(os.path.join(self.folder, "results", "trial_wavefunctions.md"))
        self.assertIn("## DMC energy against the single RHF determinant", text)
        self.assertIn("- Cutoff 0.001: USABLE (not above at any of its 2 geometries).", text)
        self.assertNotIn("## Convergence with the size of the expansion", text)    # one numeric cutoff shows no convergence
        self.assertNotIn("## Automatic choice of the cutoff", text)
        wf = self.workflow(manual, engine=engine)                       # the next process has nothing to do
        self.assertEqual(self.project()["stages"]["trial"]["status"], "DONE")
        # ---- the input now asks for the automatic mode: a new plan with n_det_max = 100000; the rows hf are kept
        calls = len(engine.calls)
        wf = self.workflow(AUTO_INPUT, engine=engine)
        self.assertEqual(self.project()["stages"]["trial"]["status"], "PENDING")
        status, _passes = TS.advance_to_end(wf)
        self.assertEqual(status, "DONE", self.project()["stages"])
        self.assertGreater(len(engine.calls), calls)
        self.assertEqual({k: v for k, v in engine.specs.items() if k in specs}, specs)
        self.assertEqual(sorted(k for k in engine.specs if k not in specs),
                         sorted("trial/geometry/%s/%s%s" % (g, label, s) for g in ("start", "final") for label in (COARSE, FINE)
                                for s in ("", "_s2")))
        self.assertEqual(self.project()["stages"]["trial"]["message"], "all 12 DMC run(s) are DONE; automatic cutoff: CONVERGED "
                                                                       "within 1.0 kcal/mol between the cutoffs 2.5e-4 and 1e-4 "
                                                                       "(2 series)")
        self.assertEqual(sorted(self.record()["left_plan"]), sorted(k for k in specs if "c1e-03" in k))
        text = TS.read(os.path.join(self.folder, "results", "trial_wavefunctions.md"))
        self.assertIn("4 DMC run(s) of an older plan of the stage", text)
        self.assertIn("## Automatic choice of the cutoff", text)

    def grown(self):
        """A workflow whose stage has just made its first decision (NOT_DECIDED: one more series). The record names
        the 6 runs of the third series, and the engine was not asked for them yet. Returns the workflow, the engine
        and the 6 keys."""
        wf = self.workflow(AUTO_INPUT)
        path = os.path.join(self.folder, "work", "trial", "state.json")
        for _ in range(4000):
            if wf.advance() != "RUNNING" or (os.path.exists(path) and self.record()["auto"]["series"] == 3):
                break
        record = self.record()
        new = sorted(k for k in T.record_keys(record) if k.endswith("_s3"))
        self.assertEqual((record["auto"]["series"], len(new)), (3, 6))
        self.assertEqual({record["points"][k]["state"] for k in new}, {"ABSENT"})
        engine = wf.backend.engine(wf)
        self.assertFalse([k for k in new if k in engine.points])
        return wf, engine, new

    def test_switching_the_option_off_stops_the_runs_that_the_stage_added(self):
        wf, engine, new = self.grown()
        self.assertEqual(wf.advance(), "RUNNING")                       # the 6 runs are asked for, and they wait
        self.assertEqual({self.record()["points"][k]["state"] for k in new}, {"QUEUED"})
        self.workflow(TS.WATER + "[report]\ngif = no\n", engine=engine)  # the input has no section [trial] any more
        self.assertEqual(sorted(engine.stopped), new)                   # the 12 runs that had ended are left alone
        self.assertEqual({engine.points[k]["state"] for k in new}, {STOPPED})
        self.assertNotIn("trial", self.project()["stages"])
        log = TS.read(os.path.join(self.folder, "work", "driver.log"))
        self.assertEqual(log.count("stopped, because the option [trial] kind = multidet was switched off"), 6)

    def test_switching_the_option_off_stops_runs_that_the_record_calls_absent(self):
        wf, engine, new = self.grown()
        # the next pass asks the engine for the 6 runs, and the driver process ends before the pass writes its record
        for spec in wf.trial_runs().specs():
            engine.submit(spec)
        self.assertEqual({engine.points[k]["state"] for k in new}, {QUEUED})
        self.assertEqual({self.record()["points"][k]["state"] for k in new}, {"ABSENT"})
        self.workflow(TS.WATER + "[report]\ngif = no\n", engine=engine)
        self.assertEqual(sorted(engine.stopped), new)
        self.assertEqual({engine.points[k]["state"] for k in new}, {STOPPED})
        self.assertNotIn("trial", self.project()["stages"])

    def cluster_input(self, trial, ranks=32):
        """A project that needs a machine profile (no section [model]), with the profile next to the input file."""
        os.makedirs(os.path.join(self.folder, "pp"), exist_ok=True)
        G.write_xyz(os.path.join(self.folder, "geometry.xyz"), ["O", "H", "H"], M.water_geometry())
        TS.write(os.path.join(self.folder, "testmachine.ini"), TS.QP2_PROFILE % {
            "pp": os.path.join(self.folder, "pp"), "ranks": ranks, "setup": TS.QP2_SETUP, "job": TS.QP2_JOB, "tools": TS.QP2_TOOLS})
        TS.write(self.input, "[geometry]\nmode = single\ngeometry = geometry.xyz\n[surrogate]\nprogram = none\n"
                             "[machine]\nprofile = testmachine\n[trial]\nkind = multidet\n" + trial)
        return Config.load(self.input)

    def test_check_prints_the_plan_of_the_automatic_mode_and_tests_series_max(self):
        self.cluster_input("cutoff = auto\n")
        code, text = self.command("check")
        self.assertEqual(code, 0, text)
        self.assertNotIn("PROBLEM", text)
        for part in ("stage trial (after the geometry search)", "geometries: up to 2",
                     "a CIPSI wave function of 100000 determinants",
                     "cutoffs: the automatic mode runs ([trial] cutoff = auto). It starts with the cutoffs 2.5e-4, 1e-4 and hf. "
                     "Its ladder is 2.5e-4, 1e-4, 5e-5 ([trial] auto_cutoffs)", "the tolerance 1.0 kcal/mol ([trial] tolerance)",
                     "DMC series: 2 at the start and at most 4",
                     "DMC runs: up to 12 at the start (3 cutoffs with hf, 2 series) and at most 32 (4 cutoffs with hf, 4 series)"):
            self.assertIn(part, text)
        # 256 MPI ranks allow 2 DMC runs of one geometry: the 4 of series_max do not fit, the 2 of series would
        cfg = self.cluster_input("cutoff = auto\n", ranks=256)
        code, text = self.command("check")
        self.assertEqual(code, 1)
        self.assertIn("PROBLEM: [trial] series_max = 4, but the QMC jobs of this profile use 256 MPI ranks", text)
        with self.assertRaises(ValueError) as ctx:
            make_backend(cfg)
        self.assertIn("[trial] series_max = 4", str(ctx.exception))
        code, text = self.command("submit")
        self.assertEqual(code, 1)
        self.assertIn("PROBLEM: [trial] series_max = 4", text)
        self.assertIn("no driver job was submitted", text)
        self.cluster_input("cutoff = auto\nseries_max = 2\n", ranks=256)
        code, text = self.command("check")
        self.assertEqual(code, 0, text)
        self.assertIn("DMC runs: up to 12 at the start (3 cutoffs with hf, 2 series) and at most 16 (4 cutoffs with hf, 2 "
                      "series)", text)
        # a problem of the input is printed with its key
        TS.write(self.input, TS.read(self.input) + "auto_cutoffs = 1e-4, 2.5e-4\n")
        code, text = self.command("check")
        self.assertEqual(code, 1)
        self.assertIn("[trial] auto_cutoffs = 1e-4, 2.5e-4 must be strictly decreasing", text)


# ===================================================================================== the real engine
class TestRealEngineAuto(TS.TM.Qp2Case):
    """The plan of the automatic mode on ``QmcpackEngine``, with the fake batch system of the engine tests. The
    engine must accept every request of the first plan, and every request of a plan that grew."""

    LONG = dict(AUTO, auto={"ladder": [2.5e-4, 1e-4, 5e-5]})

    def runs(self, eng, geometries):
        return T.TrialRuns(eng, self.symbols, geometries, self.LONG, TS.TestRealEngine.WORKFLOW_SETTINGS, 200,
                           os.path.join(self.tmp, "project", "work", "trial"))

    def test_the_engine_accepts_the_first_plan_and_a_plan_that_grew(self):
        eng = self.engine()
        moved = self.coords.copy(); moved[1] += 0.01
        geometries = {"min": (self.coords, moved)}
        runs = self.runs(eng, geometries)
        self.assertEqual(runs.tick(), "RUNNING")
        keys = runs.keys()
        self.assertEqual(len(keys), 12)                                 # 2 geometries, 2 cutoffs and hf, 2 series
        self.assertEqual(sorted(eng.keys()), sorted(keys))
        self.assertNotIn("need attention", runs.message)
        # one Quantum Package job per geometry: RHF and a CIPSI wave function of 100000 determinants for all cutoffs
        parents = [os.path.join(self.root, "trial", "min", g, "qp2_n100000", "job_parent.sh") for g in ("start", "final")]
        self.assertEqual([j["script"] for j in self.sched.submitted], parents)
        self.assertIn("begin_step cipsi", self.sched.submitted[0]["text"])
        self.assertIn("n_det_max 100000", self.sched.submitted[0]["text"])
        for key in keys:
            p = self.read_json(key, "point.json")
            hf = "/hf" in key
            self.assertEqual((p["trial"]["cutoff"], p["trial"]["n_det_max"]),
                             ("hf", None) if hf else (2.5e-4 if "c2.5e-04" in key else 1e-4, 100000))
            self.assertEqual((p["series"], p.get("base_key")), (2, key[:-3]) if key.endswith("_s2") else (1, None))
        # the record of a plan that grew, as the stage writes it: a third series for every row, and the third cutoff
        path = os.path.join(self.tmp, "project", "work", "trial", "state.json")
        state = json.loads(TS.read(path))
        state["auto"].update(series=3, cutoffs=[2.5e-4, 1e-4, 5e-5])
        TS.write(path, json.dumps(state))
        again = self.engine()                                           # a new driver process: a new engine object
        grown = self.runs(again, geometries)
        self.assertEqual(len(grown.keys()), 2 * 4 * 3)
        self.assertEqual(grown.tick(), "RUNNING")
        self.assertNotIn("need attention", grown.message)               # the engine refused no request
        self.assertNotIn("refused", grown.message)
        self.assertEqual(sorted(again.keys()), sorted(grown.keys()))
        self.assertEqual([j["script"] for j in self.sched.submitted], parents)     # the new cutoff needs no new CIPSI job
        for key in grown.keys():
            p = self.read_json(key, "point.json")
            if key.endswith("_s3"):
                self.assertEqual((p["series"], p["base_key"]), (3, key[:-3]))
            if THIRD in key and not key.endswith(("_s2", "_s3")):
                self.assertEqual((p["trial"]["cutoff"], p["qp2_key"]),
                                 (5e-5, "/".join(key.split("/")[:3]) + "/qp2_n100000"))
        record = json.loads(TS.read(path))
        self.assertEqual({v["state"] for v in record["points"].values()}, {"QUEUED"})
        self.assertEqual((record["auto"]["series"], record["auto"]["cutoffs"]), (3, [2.5e-4, 1e-4, 5e-5]))


# ===================================================================================== the report
OLD_CSV_HEADER = [                      # the columns of version 0.1.0: their places must stay
    "row", "target", "kind", "geometry", "minus_target", "cutoff", "n_det_max", "n_det_cipsi", "n_det",
    "n_det_cipsi_over_n_det", "e_rhf_qp2_hartree", "e_var_cipsi_hartree", "pt2_cipsi_hartree", "pt2_cipsi_error_hartree",
    "e_var_plus_pt2_cipsi_hartree", "e_ref_hartree", "qp2_basis", "qp2_pseudo", "qp2_version",
    "rhf_qp2_minus_pyscf_millihartree", "vmc_check_energy_hartree", "vmc_check_error_hartree",
    "vmc_check_difference_millihartree", "vmc_check_limit_millihartree", "vmc_check_verdict", "vmc_check_runs",
    "dmc_states", "dmc_energy_hartree", "dmc_error_hartree", "dmc_series_with_energy", "dmc_series_planned", "dmc_blocks",
    "core_hours", "dmc_messages", "single_determinant_dmc_energy_hartree", "single_determinant_dmc_error_hartree",
    "search_status", "minus_target_search_status", "uses_not_converged_geometry", "difference_kcal_per_mol",
    "difference_error_kcal_per_mol"]


class TestReportAuto(unittest.TestCase):
    """The demonstration project (ammonia inversion: a minimum ``reactant`` and a ``saddle``, both moved by their
    search) with a record of the stage. Every run has 200 blocks with the amplitude 4.0e-4 hartree: the error of
    a row of 2 series is 4.0e-4 / sqrt(189) / sqrt(2) = 2.06e-5 hartree."""

    ENERGY = {("reactant", "start"): -11.7300, ("reactant", "final"): -11.7310, ("saddle", "start"): -11.7200,
              ("saddle", "final"): -11.7205}
    HF_SHIFT = 0.0400                   # the single RHF determinant is this much higher at every geometry
    NOISE = 4.0e-4
    LONG = dict(AUTO, auto={"ladder": [2.5e-4, 1e-4, 5e-5]})

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.base = os.path.join(cls.tmp, "demo")
        cls.demo = make_demo_project(cls.base)
        data = tables.collect(cls.base, cls.demo.engine)
        cls.geometries = {name: (t.start, t.final) for name, t in data.targets.items()}
        cls.kinds = {name: t.kind for name, t in data.targets.items()}
        cls.symbols = cls.demo.symbols

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def stage(self, name, trial=None, energy=None, noise=None, passes=40, kinds="project"):
        """A copy of the project with the record of the stage. The stage runs up to ``passes`` times: one pass, then
        every run that waits gets its blocks; ``passes = 0`` asks for the first plan and gives no run its blocks.
        ``energy`` and ``noise``: by key or by row. Returns the folder, the engine and the stage."""
        energy, noise = energy or {}, noise or {}
        folder = os.path.join(self.tmp, name)
        shutil.copytree(self.base, folder)
        Project(folder).add_stage("trial", before="report")             # the state of a project whose input asks for the stage
        engine = Exact(self.demo.engine)
        runs = T.TrialRuns(engine, self.symbols, self.geometries, trial or AUTO, TS.SETTINGS, 200,
                           os.path.join(folder, "work", "trial"), kinds=self.kinds if kinds == "project" else kinds)
        if passes == 0:
            runs.tick()
        for _ in range(passes):
            if runs.tick() != "RUNNING":
                break
            for key in runs.keys():
                if key in engine.points and engine.points[key]["state"] == QUEUED:
                    target, geometry, label, _series = split(key)
                    row = (target, geometry, label)
                    default = self.ENERGY[target, geometry] + (self.HF_SHIFT if label == "hf" else 0.0)
                    engine.finish(key, mean=energy.get(key, energy.get(row, default)),
                                  noise=noise.get(key, noise.get(row, self.NOISE)))
        return folder, engine, runs

    def table(self, folder, engine):
        return RT.collect(tables.collect(folder, engine), engine)

    def files(self, folder, engine, table=None):
        """The two files of the report as text and as CSV lines."""
        table = table or self.table(folder, engine)
        RT.write_md(table, os.path.join(folder, "t.md"))
        RT.write_csv(table, os.path.join(folder, "t.csv"))
        with open(os.path.join(folder, "t.csv")) as fh:
            return TS.read(os.path.join(folder, "t.md")), list(csv.DictReader(fh))

    def section(self, text, title):
        """The text of one section of the file (from its title to the next title)."""
        start = text.index("## " + title + "\n")
        end = text.find("\n## ", start + 1)
        return text[start:end if end > 0 else len(text)]

    def test_kinds_of_the_demonstration_project(self):
        self.assertEqual(self.kinds, {"reactant": "minimum", "saddle": "saddle"})
        self.assertEqual(RT.CSV_HEADER[:len(OLD_CSV_HEADER)], OLD_CSV_HEADER)       # new columns stand at the end
        self.assertEqual(RT.CSV_HEADER[len(OLD_CSV_HEADER):], [
            "dmc_minus_hf_millihartree", "dmc_minus_hf_error_millihartree", "against_hf", "cutoff_usable", "coarse_cutoff",
            "fine_cutoff", "difference_coarse_kcal_per_mol", "difference_coarse_error_kcal_per_mol",
            "difference_fine_kcal_per_mol", "difference_fine_error_kcal_per_mol", "change_fine_minus_coarse_kcal_per_mol",
            "change_error_kcal_per_mol", "bound_kcal_per_mol", "tolerance_kcal_per_mol", "verdict", "project_verdict",
            "project_verdict_reason", "automatic_final_verdict"])
        self.assertEqual(len(set(RT.CSV_HEADER)), len(RT.CSV_HEADER))

    def test_the_three_sections_of_a_converged_project(self):
        folder, engine, runs = self.stage("converged")
        self.assertEqual(runs.status, "DONE")
        files = build_report(folder, engine, gifs=False)
        self.assertTrue(any(f.endswith("trial_wavefunctions.md") for f in files))
        text = TS.read(os.path.join(folder, "results", "trial_wavefunctions.md"))
        titles = ["## Trial wave functions", "## Checks of the trial wave function", "## DMC energies",
                  "## DMC energy against the single RHF determinant", "## Convergence with the size of the expansion",
                  "## Automatic choice of the cutoff", "## Energy change of every target: final geometry minus start geometry",
                  "## Barriers: saddle minus minimum"]
        self.assertEqual(re.findall(r"^## .*$", text, flags=re.M), titles)          # the order of the sections
        self.assertIn("Values of the input section [trial]: n_det_max = 100000; cutoff = auto with the ladder 0.00025, 0.0001 "
                      "(auto_cutoffs); tolerance = 1.0 kcal/mol; 2 DMC series at the start and at most 4 (series and "
                      "series_max), of 200 blocks each, for each geometry and each cutoff;", text)
        self.assertIn("The plan of the stage now holds the cutoffs 0.00025, 0.0001 and hf with 2 DMC series.", text)
        row_error = self.NOISE / math.sqrt(189.0) / math.sqrt(2.0)      # hartree

        # ---- section 1: each cut against the row hf
        part = self.section(text, "DMC energy against the single RHF determinant")
        delta = R.pm_text(-1000.0 * self.HF_SHIFT, 1000.0 * math.sqrt(2.0) * row_error)
        self.assertEqual(delta, "-40.000 +/- 0.029")                    # millihartree
        for target in ("reactant", "saddle"):
            for geometry in ("start", "final"):
                for cutoff in ("0.00025", "0.0001"):
                    self.assertIn("| %s | %s | %s | 55 | %s | not above |" % (target, geometry, cutoff, delta), part)
        self.assertEqual(part.count("| not above |"), 8)
        self.assertIn("- Cutoff 0.00025: USABLE (not above at any of its 4 geometries).", part)
        self.assertIn("- Cutoff 0.0001: USABLE (not above at any of its 4 geometries).", part)
        self.assertIn(RT.AGAINST_HF, part)
        for words in ("more than 2 standard errors", "worse nodes than one determinant", "not optimized together with the "
                      "Jastrow factor", "the row hf of this stage", "this report prints no such difference"):
            self.assertIn(words, RT.AGAINST_HF)
        self.assertNotIn("Note:", part)                                 # the CIPSI wave function is 59.6 times larger

        # ---- section 2: the convergence of every printed difference
        part = self.section(text, "Convergence with the size of the expansion")
        self.assertIn("The two usable cutoffs with the smallest values are compared: 0.00025 (coarse: fewer determinants) and "
                      "0.0001 (fine). Tolerance: 1.0 kcal/mol ([trial] tolerance).", part)
        self.assertIn("| energy difference | at the cutoff 0.00025 (kcal/mol) | at the cutoff 0.0001 (kcal/mol) | change: fine "
                      "minus coarse (kcal/mol) | bound (kcal/mol) | verdict |", part)
        table = self.table(folder, engine)
        ev = table["evaluation"]
        expected = {"final minus start of reactant": -0.0010, "final minus start of saddle": -0.0005,
                    "barrier saddle minus reactant at the start geometries": 0.0100,
                    "barrier saddle minus reactant at the final geometries": 0.0105}
        self.assertEqual([d["name"] for d in ev["differences"]], list(expected))
        for d in ev["differences"]:
            self.assertAlmostEqual(d["coarse"], expected[d["name"]] * H, places=8)
            self.assertAlmostEqual(d["fine"], expected[d["name"]] * H, places=8)
            self.assertAlmostEqual(d["coarse_error"], math.sqrt(2.0) * row_error * H, places=9)
            self.assertAlmostEqual(d["e"], 2.0 * row_error * H, places=9)
            self.assertEqual(d["verdict"], "CONVERGED")
            value = tables.fmt_pm(expected[d["name"]] * H, math.sqrt(2.0) * row_error * H)
            self.assertIn("| %s | %s | %s | %s | %s | CONVERGED |" % (d["name"], value, value, R.pm_text(d["S"], d["e"]),
                                                                    R.bound_text(d)), part)
        self.assertIn("| final minus start of reactant | -0.628 +/- 0.018 | -0.628 +/- 0.018 |", part)
        self.assertIn("| 0.052 | CONVERGED |", part)                    # the bound: 2 x 0.0258 kcal/mol
        self.assertIn(RT.CONVERGENCE_COLUMNS, part)
        self.assertIn("Verdict of the project: CONVERGED. Reason: all 4 tested energy difference(s) changed by at most 1.0 "
                      "kcal/mol between the cutoffs 0.00025 and 0.0001, with 2 standard errors of margin (largest bound 0.052 "
                      "kcal/mol).", part)
        self.assertIn(R.MEANING_TEXT, part)
        self.assertIn(R.TOTAL_ENERGIES_TEXT, part)
        self.assertNotIn(RT.MANUAL_MODE, text)                          # the automatic mode adds runs by itself
        self.assertNotIn(RT.NO_HF_ROW, text)
        self.assertEqual(text.count(R.TOTAL_ENERGIES_TEXT), 1)

        # ---- section 3: what the stage decided
        part = self.section(text, "Automatic choice of the cutoff")
        self.assertIn(R.RULE_TEXT, part)
        self.assertIn("Ladder ([trial] auto_cutoffs): 0.00025, 0.0001. Tolerance: 1.0 kcal/mol. DMC series: 2 at the start and "
                      "at most 4. The plan of the stage now holds the cutoffs 0.00025, 0.0001 and hf with 2 DMC series for each "
                      "geometry and each cutoff.", part)
        self.assertIn("Decisions of the stage, in the order in which they were made:", part)
        decision = runs.state["auto"]["decisions"][0]
        self.assertIn("1. %s: cutoffs 0.00025, 0.0001 and hf with 2 DMC series; tolerance 1.0 kcal/mol. No cut is above the "
                      "single RHF determinant. Between the cutoffs 0.00025 and 0.0001: final minus start of reactant changed by "
                      % decision["time"], part)
        self.assertIn("Verdict of this decision: CONVERGED. Action: the stage ends with the verdict CONVERGED.", part)
        self.assertNotIn("\n2. ", part)
        self.assertIn("Final verdict: CONVERGED within 1.0 kcal/mol between the cutoffs 0.00025 and 0.0001 (2 series).", part)
        self.assertIn("Result: the energy differences of the project are those at the fine cutoff 0.0001.", part)
        self.assertIn("- final minus start of reactant: -0.628 +/- 0.018 kcal/mol (CONVERGED; bound 0.052 kcal/mol)", part)
        self.assertIn("- barrier saddle minus reactant at the final geometries: 6.589 +/- 0.018 kcal/mol (CONVERGED; bound "
                      "0.052 kcal/mol)", part)
        self.assertEqual(part.count("\n- "), 4)                         # one result line for each tested difference
        # the rest of the file is what it was
        for needle in (RT.ORBITALS, RT.NO_SUBTRACTION, "this report prints no such difference"):
            self.assertIn(needle, text)
        self.assertNotIn("NOT COMPARABLE: ", text)
        self.assertNotIn(RT.ONE_CUTOFF, text)
        self.assertIsNone(re.search(r"\b(nan|inf|None)\b", text))      # every cell holds a number or plain words

        # ---- the CSV file
        with open(os.path.join(folder, "results", "trial_wavefunctions.csv")) as fh:
            rows = list(csv.DictReader(fh))
        self.assertEqual(list(rows[0]), RT.CSV_HEADER)
        kinds = [r["row"] for r in rows]
        self.assertEqual([kinds.count(k) for k in ("energy", "final minus start", "barrier", "barrier change", "against hf",
                                                   "convergence")], [12, 6, 6, 3, 8, 4])
        self.assertEqual(kinds, sorted(kinds, key=["energy", "final minus start", "barrier", "barrier change", "against hf",
                                                   "convergence"].index))          # the new lines come last
        against = [r for r in rows if r["row"] == "against hf"]
        self.assertEqual([(r["target"], r["geometry"], r["cutoff"]) for r in against],
                         [(t, g, c) for t in ("reactant", "saddle") for g in ("start", "final") for c in ("0.00025", "0.0001")])
        for r in against:
            self.assertEqual((r["kind"], r["n_det_max"], r["n_det"], r["n_det_cipsi"], r["against_hf"], r["cutoff_usable"],
                              r["uses_not_converged_geometry"]),
                             ("saddle" if r["target"] == "saddle" else "minimum", "100000", "55", "3280", "not above", "USABLE", ""))
            self.assertAlmostEqual(float(r["dmc_minus_hf_millihartree"]), -40.0, places=5)
            self.assertAlmostEqual(float(r["dmc_minus_hf_error_millihartree"]), 1000.0 * math.sqrt(2.0) * row_error, places=6)
            self.assertEqual((r["dmc_energy_hartree"], r["difference_kcal_per_mol"], r["verdict"]), ("", "", ""))
        convergence = [r for r in rows if r["row"] == "convergence"]
        self.assertEqual([(r["target"], r["minus_target"], r["geometry"], r["kind"]) for r in convergence],
                         [("reactant", "", "final minus start", "minimum"), ("saddle", "", "final minus start", "saddle"),
                          ("saddle", "reactant", "start", "saddle"), ("saddle", "reactant", "final", "saddle")])
        old = {(r["row"], r["target"], r["geometry"], r["cutoff"]): r for r in rows}
        for r, d in zip(convergence, ev["differences"]):
            self.assertEqual((r["coarse_cutoff"], r["fine_cutoff"], r["verdict"], r["project_verdict"], r["automatic_final_verdict"],
                              r["tolerance_kcal_per_mol"], r["cutoff"]),
                             ("0.00025", "0.0001", "CONVERGED", "CONVERGED", "CONVERGED", "1.000000", ""))
            self.assertEqual(r["project_verdict_reason"], ev["reason"])
            for column, value in (("difference_coarse_kcal_per_mol", d["coarse"]), ("difference_fine_kcal_per_mol", d["fine"]),
                                  ("difference_fine_error_kcal_per_mol", d["fine_error"]),
                                  ("change_fine_minus_coarse_kcal_per_mol", d["S"]), ("change_error_kcal_per_mol", d["e"]),
                                  ("bound_kcal_per_mol", d["bound"])):
                self.assertAlmostEqual(float(r[column]), value, places=5)
            # the value at the fine cutoff is the value of the old line of that difference and cutoff
            kind = "final minus start" if d["kind"] == "change" else "barrier"
            line = old[kind, r["target"], r["geometry"], "0.0001"]
            self.assertEqual(r["difference_fine_kcal_per_mol"], line["difference_kcal_per_mol"])
            self.assertEqual(r["difference_fine_error_kcal_per_mol"], line["difference_error_kcal_per_mol"])

    def test_a_cut_above_the_single_determinant_and_a_longer_ladder(self):
        # the coarse cut is 1 millihartree above hf at the final geometry of the saddle: 34 standard errors
        above = self.ENERGY["saddle", "final"] + self.HF_SHIFT + 1.0e-3
        folder, engine, runs = self.stage("above", self.LONG, energy={("saddle", "final", COARSE): above})
        self.assertEqual(runs.status, "DONE")
        auto = runs.state["auto"]
        self.assertEqual((auto["verdict"], auto["cutoffs"], [d["action"] for d in auto["decisions"]]),
                         ("CONVERGED", [2.5e-4, 1e-4, 5e-5], ["next cutoff", "final"]))
        engine.points["trial/reactant/start/%s" % FINE]["facts"].update(n_det=400)      # 3280 / 400 = 8.2: below 10
        text, rows = self.files(folder, engine)
        part = self.section(text, "DMC energy against the single RHF determinant")
        self.assertIn("| saddle | final | 0.00025 | 55 | +1.000 +/- 0.029 | above |", part)
        self.assertEqual((part.count("| above |"), part.count("| not above |")), (1, 11))
        self.assertIn("- Cutoff 0.00025: NOT USABLE (above at saddle final).", part)
        self.assertIn("- Cutoff 0.0001: USABLE (not above at any of its 4 geometries).", part)
        self.assertIn("- Cutoff 5e-05: USABLE (not above at any of its 4 geometries).", part)
        self.assertIn("- Note: at reactant start the cut 0.0001 holds 400 of the 3280 determinants of its CIPSI wave function. "
                      "The CIPSI wave function is only 8.2 times larger (less than 10 times), so this cut depends on n_det_max.",
                      part)
        self.assertEqual(part.count("- Note:"), 1)
        part = self.section(text, "Convergence with the size of the expansion")
        self.assertIn("The two usable cutoffs with the smallest values are compared: 0.0001 (coarse: fewer determinants) and "
                      "5e-05 (fine).", part)
        self.assertIn("Verdict of the project: CONVERGED.", part)
        part = self.section(text, "Automatic choice of the cutoff")
        self.assertIn("Ladder ([trial] auto_cutoffs): 0.00025, 0.0001, 5e-05.", part)
        self.assertIn("The plan of the stage now holds the cutoffs 0.00025, 0.0001, 5e-05 and hf with 2 DMC series", part)
        self.assertIn("1. %s: cutoffs 0.00025, 0.0001 and hf with 2 DMC series; tolerance 1.0 kcal/mol. Above the single RHF "
                      "determinant: the cutoff 0.00025 at saddle final by +1.000 +/- 0.029 millihartree. Verdict of this "
                      "decision: NOT_CONVERGED. Action: the next cutoff of the ladder, 5e-05, is added with 2 DMC series, "
                      "because fewer than two cutoffs are usable." % auto["decisions"][0]["time"], part)
        self.assertIn("2. %s: cutoffs 0.00025, 0.0001, 5e-05 and hf with 2 DMC series;" % auto["decisions"][1]["time"], part)
        self.assertIn("Final verdict: CONVERGED within 1.0 kcal/mol between the cutoffs 0.0001 and 5e-05 (2 series).", part)
        self.assertIn("Result: the energy differences of the project are those at the fine cutoff 5e-05.", part)
        against = [r for r in rows if r["row"] == "against hf"]
        self.assertEqual(len(against), 12)
        self.assertEqual({r["cutoff"]: r["cutoff_usable"] for r in against},
                         {"0.00025": "NOT USABLE", "0.0001": "USABLE", "5e-05": "USABLE"})
        self.assertEqual([(r["target"], r["geometry"], r["cutoff"]) for r in against if r["against_hf"] == "above"],
                         [("saddle", "final", "0.00025")])
        self.assertEqual({(r["coarse_cutoff"], r["fine_cutoff"]) for r in rows if r["row"] == "convergence"}, {("0.0001", "5e-05")})
        # the ratio of the note is cut, not rounded: 3280 / 329 = 9.97 is below 10, and it is not printed as 10.0
        engine.points["trial/reactant/start/%s" % FINE]["facts"].update(n_det=329)
        text, rows = self.files(folder, engine)
        self.assertIn("the cut 0.0001 holds 329 of the 3280 determinants of its CIPSI wave function. The CIPSI wave function "
                      "is only 9.9 times larger (less than 10 times), so this cut depends on n_det_max.", text)
        self.assertNotIn("10.0 times larger", text)
        self.assertEqual(text.count("- Note:"), 1)
        for parent, n_det, expected in ((3280, 400, "8.2"), (3280, 329, "9.9"), (9999, 1000, "9.9"), (2000, 1000, "2.0"),
                                        (1999, 1000, "1.9")):
            self.assertEqual(RT._ratio_cut({"trial": {"n_det_parent": parent, "n_det": n_det}}), expected)

    def test_decisions_made_with_two_ladders_and_a_stage_that_needs_attention(self):
        # the reactant change differs by 3.0 kcal/mol between the cutoffs 2.5e-4 and 1e-4, and not between 1e-4 and 5e-5
        energy = {("reactant", "final", FINE): self.ENERGY["reactant", "final"] + 3.0 / H,
                  ("reactant", "final", THIRD): self.ENERGY["reactant", "final"] + 3.0 / H}
        folder, engine, runs = self.stage("two_ladders", energy=energy)
        self.assertEqual((runs.status, runs.state["auto"]["verdict"]), ("DONE", "NOT_CONVERGED"))
        # one more cutoff at the end of the ladder: the stage goes on, and its record holds decisions of both ladders
        longer = T.TrialRuns(engine, self.symbols, self.geometries, self.LONG, TS.SETTINGS, 200,
                             os.path.join(folder, "work", "trial"), kinds=self.kinds)
        for _ in range(10):
            if longer.tick() != "RUNNING":
                break
            for key in longer.keys():
                if key in engine.points and engine.points[key]["state"] == QUEUED:
                    target, geometry, label, _series = split(key)
                    engine.finish(key, mean=energy.get((target, geometry, label), self.ENERGY[target, geometry]), noise=self.NOISE)
        auto = longer.state["auto"]
        self.assertEqual((longer.status, auto["verdict"], [d["ladder"] for d in auto["decisions"]]),
                         ("DONE", "CONVERGED", [[2.5e-4, 1e-4], [2.5e-4, 1e-4, 5e-5], [2.5e-4, 1e-4, 5e-5]]))
        text, _lines = self.files(folder, engine)
        part = self.section(text, "Automatic choice of the cutoff")
        first, second, third = [line for line in part.splitlines() if re.match(r"\d\. ", line)]
        self.assertTrue(first.endswith("Action: the stage ends with the verdict NOT_CONVERGED (the ladder holds no further "
                                       "cutoff: to continue, add a smaller cutoff at the end of [trial] auto_cutoffs). The ladder of the input was then 0.00025, 0.0001. After this decision the "
                                       "values of [trial] changed, and the stage decided again."), first)
        self.assertNotIn("The ladder of the input was then", second + third)
        self.assertNotIn("After this decision", second + third)
        # a record that an older code wrote: its decisions hold no ladder. The report still says that the input
        # changed after a final decision, and a new object gives the old decisions the ladder of the record
        state_file = os.path.join(folder, "work", "trial", "state.json")
        record = json.loads(TS.read(state_file))
        for entry in record["auto"]["decisions"]:
            del entry["ladder"]
        TS.write(state_file, json.dumps(record))
        text, _lines = self.files(folder, engine)
        lines = [line for line in self.section(text, "Automatic choice of the cutoff").splitlines() if re.match(r"\d\. ", line)]
        self.assertTrue(lines[0].endswith("(the ladder holds no further cutoff: to continue, add a smaller cutoff at the end of [trial] auto_cutoffs). After this decision the values of [trial] "
                                          "changed, and the stage decided again."), lines[0])
        self.assertNotIn("The ladder of the input was then", "".join(lines))
        record["auto"]["ladder"] = [2.5e-4, 1e-4]                       # the record as the older code left it
        record["rule"]["trial"] = dict(AUTO)
        record["auto"]["decisions"] = record["auto"]["decisions"][:1]
        record["auto"].update(cutoffs=[2.5e-4, 1e-4], final=True, verdict="NOT_CONVERGED")
        TS.write(state_file, json.dumps(record))
        T.TrialRuns(engine, self.symbols, self.geometries, self.LONG, TS.SETTINGS, 200, os.path.join(folder, "work", "trial"),
                    kinds=self.kinds)
        kept = json.loads(TS.read(state_file))["auto"]
        self.assertEqual(([d.get("ladder") for d in kept["decisions"]], kept["ladder"], kept["final"]),
                         ([None], [2.5e-4, 1e-4, 5e-5], False))
        self.assertNotIn("The stage needed attention", text)
        # a run whose request the engine refuses: the stage needs attention, and the file says so above its tables
        key = "trial/saddle/final/%s" % FINE
        engine.refuse[key] = "point %s exists with another geometry" % key
        self.assertEqual(longer.tick(), "NEEDS_ATTENTION")
        text, _lines = self.files(folder, engine)
        start = "At its last pass the stage had the status NEEDS_ATTENTION, with runs that need attention: "
        note = [line for line in text.splitlines() if line.startswith(start)]
        self.assertEqual(len(note), 1)
        self.assertIn("%s: the point could not be submitted: point %s exists with another geometry" % (key, key), note[0])
        self.assertIn("A run whose request the engine refused is still printed in the tables of this file with its energy, "
                      "and the two tests of the size of the expansion still use it: do not use a value of a row that the "
                      "message names.", note[0])
        self.assertTrue(note[0].endswith("A run that failed or was stopped makes its row not complete."), note[0])
        self.assertLess(text.index(note[0]), text.index("## Trial wave functions"))
        # while another run still writes blocks the status is RUNNING, and the file has the note as well
        other = "trial/reactant/start/hf_s2"
        engine.points[other].update(state=RUNNING)
        self.assertEqual(longer.tick(), "RUNNING")
        text, _lines = self.files(folder, engine)
        note = [line for line in text.splitlines() if line.startswith("At its last pass the stage had the status RUNNING, ")]
        self.assertEqual(len(note), 1)
        self.assertIn("need attention when the others have ended: %s: the point could not be submitted" % key, note[0])
        # a stage that only waits for its runs has no note
        del engine.refuse[key]
        self.assertEqual(longer.tick(), "RUNNING")
        text, _lines = self.files(folder, engine)
        self.assertNotIn("At its last pass the stage had the status", text)

    def test_a_final_verdict_whose_errors_are_too_large_also_for_a_smaller_cutoff(self):
        # every difference changes by 0, but the errors of 0.7 kcal/mol are too large for the tolerance 1.0: the stage
        # ends after the last series, and the report says to raise series_max above the series of the plan
        folder, engine, runs = self.stage("errors_too_large", dict(self.LONG, series_max=3), noise={
            (t, g, label): 1.2e-2 for t in ("reactant", "saddle") for g in ("start", "final") for label in (COARSE, FINE)})
        auto = runs.state["auto"]
        self.assertEqual((runs.status, auto["verdict"], auto["series"], auto["cutoffs"]), ("DONE", "NOT_DECIDED", 3, [2.5e-4, 1e-4]))
        self.assertEqual(auto["decisions"][-1]["cause"], "errors above the tolerance")
        text, _rows = self.files(folder, engine)
        part = self.section(text, "Automatic choice of the cutoff")
        self.assertIn("and the errors are too large for the verdict CONVERGED also with a smaller cutoff (2 x sqrt(2) x the "
                      "error at the fine cutoff is larger than the tolerance for final minus start of reactant (", part)
        self.assertIn("): to continue, raise [trial] series_max above 3.", part)
        # no run of the third cutoff: 4 geometries x (2 cutoffs and hf) x 3 series
        self.assertEqual(len([k for k in engine.points if k.startswith("trial/")]), 36)
        self.assertFalse([k for k in engine.points if THIRD in k])

    def test_final_verdicts_that_are_not_converged(self):
        # NOT_DECIDED at the largest number of series: the barrier changes by 1.0 kcal/mol, and the errors are 0.3 kcal/mol
        shift = {("saddle", g, FINE): self.ENERGY["saddle", g] + 1.0 / H for g in ("start", "final")}
        folder, engine, runs = self.stage("not_decided", dict(AUTO, series_max=3), energy=shift, noise={
            (t, g, label): 5.0e-3 for t in ("reactant", "saddle") for g in ("start", "final") for label in (COARSE, FINE)})
        auto = runs.state["auto"]
        self.assertEqual((runs.status, auto["verdict"], auto["series"]), ("DONE", "NOT_DECIDED", 3))
        text, rows = self.files(folder, engine)
        part = self.section(text, "Automatic choice of the cutoff")
        self.assertIn("Action: one more DMC series is added for every row (3 series now, at most 3), because the errors are too "
                      "large to decide.", part)
        self.assertIn("Final verdict: NOT_DECIDED between the cutoffs 0.00025 and 0.0001 with the tolerance 1.0 kcal/mol "
                      "(3 series): barrier saddle minus reactant at the start geometries changed by +1.00 +/- 0.26 kcal/mol "
                      "between the cutoffs 0.00025 and 0.0001 (bound 1.53 kcal/mol): the errors are too large", part)
        self.assertIn("; no further series is allowed ([trial] series_max = 3), and the ladder holds no further cutoff: to "
                      "continue, add a smaller cutoff at the end of [trial] auto_cutoffs.", part)
        self.assertIn("- barrier saddle minus reactant at the start geometries: 7.28 +/- 0.19 kcal/mol (NOT_DECIDED; bound 1.53 "
                      "kcal/mol)", part)
        self.assertIn("- final minus start of saddle: -0.31 +/- 0.19 kcal/mol (CONVERGED; bound 0.53 kcal/mol)", part)
        part = self.section(text, "Convergence with the size of the expansion")
        self.assertIn("Verdict of the project: NOT_DECIDED. Reason: barrier saddle minus reactant at the start geometries "
                      "changed by +1.00 +/- 0.26 kcal/mol", part)
        self.assertEqual([r["verdict"] for r in rows if r["row"] == "convergence"],
                         ["CONVERGED", "CONVERGED", "NOT_DECIDED", "NOT_DECIDED"])
        self.assertEqual({r["automatic_final_verdict"] for r in rows if r["row"] == "convergence"}, {"NOT_DECIDED"})
        # NOT_CONVERGED without a usable pair: both cuts are above hf, and the ladder has no further cutoff
        energy = {(t, g, label): self.ENERGY[t, g] + self.HF_SHIFT + 2.0e-3 for t in ("reactant", "saddle")
                  for g in ("start", "final") for label in (COARSE, FINE)}
        folder, engine, runs = self.stage("not_usable", energy=energy)
        self.assertEqual((runs.status, runs.state["auto"]["verdict"]), ("DONE", "NOT_CONVERGED"))
        text, rows = self.files(folder, engine)
        part = self.section(text, "Convergence with the size of the expansion")
        self.assertNotIn("are compared", part)                          # no pair, so no table
        self.assertNotIn("| energy difference |", part)
        self.assertIn("Verdict of the project: NOT_CONVERGED. Reason: fewer than two cutoffs are usable: the cutoff 0.00025 is "
                      "above the single RHF determinant at reactant start by +2.000 +/- 0.029 millihartree; ", part)
        part = self.section(text, "Automatic choice of the cutoff")
        self.assertIn("Final verdict: NOT_CONVERGED (2 series): fewer than two cutoffs are usable: ", part)
        self.assertIn("; the ladder holds no further cutoff: to continue, add a smaller cutoff at the end of [trial] auto_cutoffs.", part)
        self.assertIn("Result: this verdict gives no energy difference at a fine cutoff.", part)
        convergence = [r for r in rows if r["row"] == "convergence"]
        self.assertEqual(len(convergence), 1)                           # one line with the verdict of the project
        self.assertEqual((convergence[0]["project_verdict"], convergence[0]["target"], convergence[0]["verdict"],
                          convergence[0]["automatic_final_verdict"], convergence[0]["coarse_cutoff"]),
                         ("NOT_CONVERGED", "", "", "NOT_CONVERGED", ""))
        self.assertIn("fewer than two cutoffs are usable", convergence[0]["project_verdict_reason"])
        self.assertEqual({r["cutoff_usable"] for r in rows if r["row"] == "against hf"}, {"NOT USABLE"})

    def test_a_stage_that_has_not_ended(self):
        # one pass: the runs wait, nothing is decided
        folder, engine, runs = self.stage("waiting", passes=0)
        text, rows = self.files(folder, engine)
        part = self.section(text, "DMC energy against the single RHF determinant")
        self.assertIn("| reactant | start | 0.00025 | not available | not available | not available: 0 of 2 series are DONE "
                      "(QUEUED, QUEUED); the row hf: 0 of 2 series are DONE (QUEUED, QUEUED) |", part)
        self.assertIn("- Cutoff 0.00025: no verdict yet (not available at reactant start, reactant final, saddle start, saddle "
                      "final).", part)
        part = self.section(text, "Convergence with the size of the expansion")
        self.assertIn("Verdict of the project: NO_VERDICT. Reason: 12 row(s) of the stage are not complete: reactant start, "
                      "cutoff 0.00025: 0 of 2 series are DONE (QUEUED, QUEUED); ", part)
        self.assertNotIn("| energy difference |", part)
        part = self.section(text, "Automatic choice of the cutoff")
        self.assertIn("The stage has made no decision yet: it decides when every DMC run of its plan is DONE.", part)
        self.assertIn("No final verdict yet: the stage has not ended.", part)
        self.assertNotIn("Final verdict:", part)
        self.assertNotIn("The values of [trial] changed after the last decision", part)
        against = [r for r in rows if r["row"] == "against hf"]
        self.assertEqual({r["against_hf"] for r in against}, {"not available: 0 of 2 series are DONE (QUEUED, QUEUED); the row "
                                                              "hf: 0 of 2 series are DONE (QUEUED, QUEUED)"})
        self.assertEqual({(r["dmc_minus_hf_millihartree"], r["cutoff_usable"]) for r in against}, {("", "no verdict yet")})
        convergence = [r for r in rows if r["row"] == "convergence"]
        self.assertEqual([(r["project_verdict"], r["automatic_final_verdict"]) for r in convergence], [("NO_VERDICT", "")])
        self.assertIsNone(re.search(r"\b(nan|inf|None)\b", text))
        # the input changed after the final decision: the old decision is listed, and the file says what happens next
        folder, engine, runs = self.stage("changed")
        self.assertEqual(runs.status, "DONE")
        T.TrialRuns(engine, self.symbols, self.geometries, dict(AUTO, tolerance=0.01), TS.SETTINGS, 200,
                    os.path.join(folder, "work", "trial"), kinds=self.kinds)
        text, _rows = self.files(folder, engine)
        part = self.section(text, "Automatic choice of the cutoff")
        self.assertIn("1. ", part)
        self.assertIn("tolerance 1.0 kcal/mol. No cut is above", part)   # the decision names the tolerance of its time
        self.assertIn("Tolerance: 0.01 kcal/mol.", part)
        self.assertIn("No final verdict yet: the stage has not ended. The values of [trial] changed after the last decision, so "
                      "the stage decides again.", part)

    def test_the_manual_mode_gets_the_two_tests_and_no_decision(self):
        manual = dict(TS.TRIAL_DICT, n_det_max=100000, cutoffs=[2.5e-4, 1e-4, "hf"], not_converged="stop")
        # the barrier differs by 3 kcal/mol between the two cutoffs: the automatic mode would ask for more
        shift = {("saddle", g, FINE): self.ENERGY["saddle", g] - 3.0 / H for g in ("start", "final")}
        folder, engine, runs = self.stage("manual", manual, energy=shift)
        self.assertEqual((runs.status, runs.message, len(runs.keys())), ("DONE", "all 24 DMC run(s) are DONE", 24))
        text, rows = self.files(folder, engine)
        self.assertEqual(re.findall(r"^## .*$", text, flags=re.M), [
            "## Trial wave functions", "## Checks of the trial wave function", "## DMC energies",
            "## DMC energy against the single RHF determinant", "## Convergence with the size of the expansion",
            "## Energy change of every target: final geometry minus start geometry", "## Barriers: saddle minus minimum"])
        self.assertIn("n_det_max = 100000; cutoff = 0.00025, 0.0001, hf; 2 DMC series of 200 blocks for each geometry and each "
                      "cutoff; basis file cc-pvtz_ecp_ccecp and pseudopotential file ccecp of Quantum Package; not_converged = "
                      "stop; tolerance = 1.0 kcal/mol.", text)
        part = self.section(text, "Convergence with the size of the expansion")
        self.assertIn("| barrier saddle minus reactant at the start geometries | 6.275 +/- 0.018 | 3.275 +/- 0.018 | -3.000 +/- "
                      "0.026 | 3.052 | NOT_CONVERGED |", part)
        self.assertIn("| final minus start of saddle | -0.314 +/- 0.018 | -0.314 +/- 0.018 |", part)
        self.assertIn("Verdict of the project: NOT_CONVERGED. Reason: barrier saddle minus reactant at the start geometries "
                      "changed by -3.000 +/- 0.026 kcal/mol between the cutoffs 0.00025 and 0.0001, more than the tolerance of "
                      "1.0 kcal/mol with 2 standard errors of margin; barrier saddle minus reactant at the final geometries "
                      "changed by -3.000 +/- 0.026 kcal/mol", part)
        self.assertEqual(part.count(RT.MANUAL_MODE), 1)                 # a verdict only: nothing is added by itself
        self.assertIn(R.MEANING_TEXT, part)
        self.assertIn(R.TOTAL_ENERGIES_TEXT, part)
        self.assertEqual({r["automatic_final_verdict"] for r in rows}, {""})
        self.assertEqual([r["verdict"] for r in rows if r["row"] == "convergence"],
                         ["CONVERGED", "CONVERGED", "NOT_CONVERGED", "NOT_CONVERGED"])
        # the tolerance of the input decides the verdict
        loose = os.path.join(folder, "work", "trial", "state.json")
        state = json.loads(TS.read(loose))
        state["rule"]["trial"]["tolerance"] = 5.0
        TS.write(loose, json.dumps(state))
        text, rows = self.files(folder, engine)
        self.assertIn("Tolerance: 5.0 kcal/mol ([trial] tolerance).", text)
        self.assertIn("Verdict of the project: CONVERGED.", text)
        self.assertIn("tolerance = 5.0 kcal/mol.", text)
        self.assertEqual({r["tolerance_kcal_per_mol"] for r in rows if r["row"] == "convergence"}, {"5.000000"})

    def test_the_manual_mode_without_a_row_hf_and_with_too_few_cutoffs(self):
        base = dict(TS.TRIAL_DICT, n_det_max=100000, not_converged="stop")
        # two numeric cutoffs and no hf: no cut is tested, and the two smallest cutoffs are compared
        folder, engine, runs = self.stage("no_hf", dict(base, cutoffs=[1e-3, 2.5e-4, 1e-4]))
        text, rows = self.files(folder, engine)
        self.assertNotIn("## DMC energy against the single RHF determinant", text)
        part = self.section(text, "Convergence with the size of the expansion")
        self.assertIn(RT.NO_HF_ROW, part)
        self.assertIn(RT.NO_HF_ROW + " So the two smallest cutoffs of the list are compared.", part)
        self.assertIn("every numeric cutoff counts as \"not tested\"", part)
        self.assertIn("The two cutoffs with the smallest values are compared: 0.00025 (coarse: fewer determinants) and 0.0001 "
                      "(fine).", part)
        self.assertIn("Verdict of the project: CONVERGED.", part)
        self.assertEqual([r["row"] for r in rows].count("against hf"), 0)
        self.assertEqual([r["row"] for r in rows].count("convergence"), 4)
        # the same kind of list while its runs wait: no pair is compared yet, and the section does not say that one is
        folder, engine, runs = self.stage("no_hf_waiting", dict(base, cutoffs=[2.5e-4, 1e-4]), passes=0)
        text, rows = self.files(folder, engine)
        part = self.section(text, "Convergence with the size of the expansion")
        self.assertIn(RT.NO_HF_ROW, part)
        self.assertNotIn("are compared", part)
        self.assertNotIn("| energy difference |", part)
        self.assertIn("Verdict of the project: NO_VERDICT. Reason: 8 row(s) of the stage are not complete: ", part)
        # one numeric cutoff and hf (the input of version 0.1.0): the test against hf, and no convergence section
        folder, engine, runs = self.stage("one_cut", dict(base, cutoffs=[1e-3, "hf"]))
        text, rows = self.files(folder, engine)
        self.assertIn("## DMC energy against the single RHF determinant", text)
        self.assertIn("- Cutoff 0.001: USABLE (not above at any of its 4 geometries).", text)
        self.assertNotIn("## Convergence with the size of the expansion", text)
        self.assertNotIn(R.MEANING_TEXT, text)
        self.assertEqual(text.count(R.TOTAL_ENERGIES_TEXT), 1)          # without that section the sentence stands alone
        self.assertNotIn(RT.NO_HF_ROW, text)
        self.assertEqual(([r["row"] for r in rows].count("against hf"), [r["row"] for r in rows].count("convergence")), (4, 0))
        # hf alone, and one cutoff alone: neither section; one cutoff gets the old sentence
        folder, engine, runs = self.stage("only_hf", dict(base, cutoffs=["hf"], n_det_max=None))
        text, rows = self.files(folder, engine)
        self.assertNotIn("## DMC energy against the single RHF determinant", text)
        self.assertNotIn("## Convergence", text)
        self.assertEqual({r["row"] for r in rows}, {"energy", "final minus start", "barrier", "barrier change"})
        self.assertNotIn(R.TOTAL_ENERGIES_TEXT, text)                   # no expansion, so nothing to say about its size
        folder, engine, runs = self.stage("only_one", dict(base, cutoffs=[1e-3]))
        text, rows = self.files(folder, engine)
        self.assertEqual(text.count(RT.ONE_CUTOFF), 1)
        self.assertEqual(text.count(RT.NO_HF_ROW), 1)                   # the one cut was not tested against hf
        self.assertEqual(text.count(R.TOTAL_ENERGIES_TEXT), 1)
        self.assertNotIn("## Convergence", text)
        self.assertNotIn("## DMC energy against", text)

    def test_values_that_use_a_search_that_did_not_converge_are_marked(self):
        folder, engine, runs = self.stage("marked")
        data = tables.collect(folder, engine)
        data.targets["saddle"].status = "NOT_CONVERGED"
        text, rows = self.files(folder, engine, RT.collect(data, engine))
        part = self.section(text, "DMC energy against the single RHF determinant")
        self.assertEqual(part.count(RT.MARK.strip()), 2)                # the final geometry of the saddle, 2 cutoffs
        self.assertIn("| saddle | final (search NOT_CONVERGED) | 0.0001 |", part)
        part = self.section(text, "Convergence with the size of the expansion")
        self.assertIn("| final minus start of saddle (search NOT_CONVERGED) |", part)
        self.assertIn("| barrier saddle minus reactant at the final geometries (search NOT_CONVERGED) |", part)
        self.assertIn("| barrier saddle minus reactant at the start geometries | ", part)   # the start geometries are not marked
        self.assertEqual(part.count(RT.MARK.strip()), 2)
        part = self.section(text, "Automatic choice of the cutoff")
        self.assertIn("- final minus start of saddle (search NOT_CONVERGED): ", part)
        self.assertIn("- barrier saddle minus reactant at the final geometries (search NOT_CONVERGED): ", part)
        self.assertEqual(part.count(RT.MARK.strip()), 2)
        marked = {(r["row"], r["target"], r["geometry"]) for r in rows
                  if r["uses_not_converged_geometry"] == "yes" and r["row"] in ("against hf", "convergence")}
        self.assertEqual(marked, {("against hf", "saddle", "final"), ("convergence", "saddle", "final minus start"),
                                  ("convergence", "saddle", "final")})


if __name__ == "__main__":
    unittest.main()
