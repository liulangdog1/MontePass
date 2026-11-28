"""The stage ``trial``: DMC energies with a multi-determinant trial wave function after the geometry search.

Nothing here needs a cluster program. The engine is replaced by a stand-in that records every request and
lets a test set the state, the blocks, the facts and the checks of a point. The projects are model projects
(analytic surfaces), so the geometry search itself is not replaced: it runs as in every model project.

    TestConfig          the input section [trial]
    TestTrialRuns       the class that asks for the points and keeps the record (``montepass/trial.py``)
    TestRealEngine      the same class with the real engine and a fake batch system: the engine accepts every
                        request, and the report can read what the engine says about a finished point
    TestWorkflow        the stage in the workflow, the backends and the command ``check``
    TestReport          the two files of the report (``montepass/report/trial.py``)
"""
import contextlib
import csv
import io
import json
import os
import re
import shutil
import sys
import tempfile
import unittest

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
sys.path.insert(0, HERE)

import test_multidet as TM  # noqa: E402  (the fake Quantum Package files and the fake batch system of the engine tests)
from montepass import cli, geometry as G, model as M, trial as T  # noqa: E402
from montepass.config import Config, ConfigError, TEMPLATE, write_template  # noqa: E402
from montepass.engine import multidet as MD  # noqa: E402
from montepass.engine.base import ABSENT, DONE, FAILED, QUEUED, RUNNING, STOPPED, Engine, PointData, PointSpec  # noqa: E402
from montepass.project import STAGES, Project, stage_names  # noqa: E402
from montepass.report import build_report, tables, trial as RT  # noqa: E402
from montepass.report.demo import make_demo_project  # noqa: E402
from montepass.units import HARTREE_TO_KCAL as H  # noqa: E402
from montepass.workflow import MODEL_WITHOUT_QP2, ModelBackend, Workflow, make_backend  # noqa: E402

E_SCF, E_CUT = -16.943222, -16.980444          # RHF energy and energy of a cut wave function (water, hartree)

WATER = """[system]
name = model_water
[geometry]
mode = single
[model]
molecule = water
sigma_block = 2.0
seed = 3
[search]
method = dense-v2
eps_total = 0.05
"""

TRIAL = """[trial]
kind = multidet
n_det_max = 10000
cutoff = %s
series = 2
"""

TRIAL_DICT = {"kind": "multidet", "n_det_max": 10000, "cutoffs": [1e-3, "hf"], "series": 2,
              "qp2_basis": "cc-pvtz_ecp_ccecp", "qp2_pseudo": "ccecp"}
SETTINGS = {"charge": 0, "spin": 0, "xc": "b3lyp", "basis": "ccecpccpvtz", "ecp": "ccecp", "jastrow_reuse": "none",
            "dmc": {"timestep": 0.005, "walkers": 2048, "steps": 500, "warmup_steps": 2000}}

QP2_PROFILE = """[machine]
name = test-cluster
scheduler = slurm
[setup.qmc]
lines =
    module load qmc-tools
%(setup)s
[executables]
python = python3
%(tools)s
[pseudopotentials]
dir = %(pp)s
[slurm]
account = testaccount
[job.scf]
tasks = 1
cpus_per_task = 4
mem = 8G
time = 00:30:00
[job.qmc]
tasks = %(ranks)d
mem = 16G
time = 02:00:00
%(job)s
"""
QP2_SETUP = "[setup.qp2]\nlines =\n    set +u\n    module load qp2\n"
QP2_JOB = "[job.qp2]\ntasks = 1\ncpus_per_task = 8\nmem = 16G\ntime = 01:00:00\n"
QP2_TOOLS = "determinants_tools = /q/determinants_tools.py"


def write(path, text):
    with open(path, "w") as fh:
        fh.write(text)


def read(path):
    with open(path) as fh:
        return fh.read()


def load(text):
    """Config of an input text in a temporary folder."""
    with tempfile.TemporaryDirectory() as tmp:
        write(os.path.join(tmp, "input.ini"), text)
        return Config.load(os.path.join(tmp, "input.ini"))


def problems(text):
    """The message of the ConfigError that an input text raises."""
    try:
        load(text)
    except ConfigError as exc:
        return str(exc)
    raise AssertionError("the input was accepted")


def model_input(trial="", extra=""):
    return "[geometry]\nmode = single\n[model]\nmolecule = water\n" + extra + ("[trial]\n" + trial if trial else "")


# ===================================================================================== the engine stand-in
class StandIn(Engine):
    """Engine stand-in for the points of the stage ``trial`` (keys that start with ``trial/``).

    It records every request. A test sets the state, the blocks, the facts and the checks of a point with
    :meth:`finish` or by hand in ``points``. Every other key goes to ``inner`` (the mock engine of a model
    project); with ``inner = None`` every key is handled here. With ``auto = n`` a point becomes DONE by itself
    at the n-th call of ``advance`` after its submission, as the points of a real engine do with time."""

    def __init__(self, inner=None, auto=0):
        self.inner, self.auto = inner, int(auto)
        self.calls = []                 # key of every call of submit, in order
        self.specs = {}                 # key -> the PointSpec that created the point
        self.points = {}                # key -> state, energy, message, facts, checks, cost, age
        self.refuse = {}                # key -> text: submit raises ValueError with this text
        self.fail = {}                  # key -> text: in auto mode the point ends as FAILED with this message
        self.advances = 0
        self.retried = []
        self.stopped = []               # key of every call of stop for a point of the stage
        self.transient = {}             # key -> text: submit raises RuntimeError with this text (the batch system refused)

    def mine(self, key):
        return self.inner is None or key.startswith("trial/")

    def submit(self, spec):
        if not self.mine(spec.key):
            return self.inner.submit(spec)
        self.calls.append(spec.key)
        if spec.key in self.refuse:
            raise ValueError(self.refuse[spec.key])
        if spec.key in self.transient:
            raise RuntimeError(self.transient[spec.key])
        if spec.key in self.points:
            return                      # a point that exists is left as it is
        if spec.series > 1 and spec.base_key not in self.points:
            raise ValueError("point %s: its base point %s is not a submitted series-1 point" % (spec.key, spec.base_key))
        t = spec.settings["trial"]
        self.specs[spec.key] = spec
        self.points[spec.key] = {"state": QUEUED, "energy": np.zeros(0), "age": 0, "cost": 0.0, "checks": {},
                                 "message": "waits for the Quantum Package calculation (not started)",
                                 "facts": {"kind": "multidet", "cutoff": t["cutoff"], "orbitals": "qp2_rhf"}}

    def finish(self, key, mean=-17.2, noise=1.0e-3, state=DONE, check=True, message=None, cost=1.5):
        """Give a point all its blocks around ``mean``, the facts of its wave function and the result of its checks."""
        spec, p = self.specs[key], self.points[key]
        t = spec.settings["trial"]
        rng = np.random.default_rng(sum(key.encode()))
        hf = t["cutoff"] == "hf"
        facts = {"kind": "multidet", "cutoff": t["cutoff"], "orbitals": "qp2_rhf", "label": MD.cut_label(t["cutoff"]),
                 "n_det": 1 if hf else 55, "e_ref": E_SCF if hf else E_CUT, "e_scf": E_SCF, "mo_num": 57,
                 "qp2_basis": t["qp2_basis"], "qp2_pseudo": t["qp2_pseudo"], "qp2_version": "b65c9cf9"}
        if not hf:
            facts.update({"n_det_parent": 3280, "n_det_max": t["n_det_max"], "segments": 1, "cipsi_seconds": 300.0,
                          "e_var_parent": -17.164589, "pt2_parent": -0.053480, "pt2_error_parent": 7.8e-5,
                          "e_pt2_parent": -17.218069})
        e_ref = facts["e_ref"]
        p.update({"state": state, "energy": mean + noise * rng.normal(size=spec.blocks), "facts": facts, "cost": cost,
                  "message": message or "all %d blocks are written" % spec.blocks, "checks": {"seed": {"ok": True}}})
        if spec.series == 1:
            p["checks"]["mdc4q"] = {"check": "mdc4q", "ok": True, "e_rhf_pyscf": E_SCF, "e_rhf_qp2": E_SCF + 5.0e-9,
                                    "rhf_difference": 5.0e-9}
            p["checks"]["mdvmc"] = {"check": "mdvmc", "ok": bool(check), "verdict": "PASS" if check else "FAIL",
                                    "energy": e_ref + 2.0e-4, "error": 3.0e-4, "e_ref": e_ref, "difference": 2.0e-4,
                                    "limit": 1.0e-3, "runs": [{"run": 1, "blocks": 40}, {"run": 2, "blocks": 900}]}

    def read(self, key):
        if not self.mine(key):
            return self.inner.read(key)
        p = self.points.get(key)
        if p is None:
            return PointData(key=key, state=ABSENT)
        spec = self.specs[key]
        owner = p if spec.series == 1 else self.points[spec.base_key]     # a further series reports the facts of its base
        meta = {"requested_blocks": spec.blocks, "series": spec.series, "trial": dict(owner["facts"]),
                "checks": json.loads(json.dumps(p["checks"]))}
        return PointData(key=key, state=p["state"], energy=np.array(p["energy"]), weight=np.ones(len(p["energy"])),
                         message=p["message"], meta=meta)

    def stop(self, key):
        if not self.mine(key):
            self.inner.stop(key)
            return
        self.stopped.append(key)
        p = self.points.get(key)
        if p is not None and p["state"] not in (DONE, FAILED, STOPPED):
            p.update(state=STOPPED, message="stopped on request after %d blocks" % len(p["energy"]))

    def advance(self):
        self.advances += 1
        if self.inner is not None:
            self.inner.advance()
        for key, p in self.points.items():
            p["age"] += 1
            if self.auto and p["state"] == QUEUED and p["age"] >= self.auto:
                index = sorted(self.points).index(key)
                if key in self.fail:
                    p.update(state=FAILED, message=self.fail[key])
                else:
                    self.finish(key, mean=-17.2 - 0.001 * index)

    def cost(self, key):
        if not self.mine(key):
            return self.inner.cost(key)
        return self.points[key]["cost"] if key in self.points else 0.0

    def retry(self, key):
        p = self.points.get(key)
        if p is None or p["state"] != FAILED:
            return False
        self.fail.pop(key, None)
        p.update(state=QUEUED, message="the job waits in the queue", age=0)
        self.retried.append(key)
        return True


class StandInBackend(ModelBackend):
    """The backend of a model project whose engine also answers for the points of the stage ``trial``."""

    stand_in = None

    def engine(self, wf):
        if self.stand_in is None:
            self.stand_in = StandIn(super().engine(wf), auto=3)
        return self.stand_in


class TwoEngines(Engine):
    """The points of the stage ``trial`` go to one engine, every other point to another one."""

    def __init__(self, search, trial):
        self.search, self.trial = search, trial

    def of(self, key):
        return self.trial if key.startswith("trial/") else self.search

    def submit(self, spec):
        self.of(spec.key).submit(spec)

    def read(self, key):
        return self.of(key).read(key)

    def stop(self, key):
        self.of(key).stop(key)

    def advance(self):
        self.search.advance()
        self.trial.advance()

    def cost(self, key):
        return self.of(key).cost(key)

    def failed_keys(self):
        out = []
        for engine in (self.search, self.trial):
            failed = getattr(engine, "failed_keys", None)
            if callable(failed):
                out += failed()
        return out

    def retry(self, key):
        return self.of(key).retry(key)


# ===================================================================================== the input section
class TestConfig(unittest.TestCase):
    QMC = {"charge": 0, "spin": 0, "xc": "b3lyp", "basis": "ccecpccpvtz", "ecp": "ccecp", "jastrow_reuse": "centre",
           "dmc": {"timestep": 0.005, "walkers": 2048, "steps": 500, "warmup_steps": 2000}, "slow_factor": 3.0}
    SUMMARY_KEYS = ["charge", "experiment", "machine", "mode", "model", "multiplicity", "name", "qmc", "search", "selection",
                    "subspace", "surrogate", "targets"]

    def test_default_is_single_and_changes_nothing(self):
        plain = load(model_input())                                    # an input file from before this option
        self.assertIsNone(plain.trial)
        self.assertEqual(plain.qmc, self.QMC)
        self.assertEqual(sorted(plain.summary()), self.SUMMARY_KEYS)
        self.assertEqual(plain.summary()["qmc"], dict(self.QMC, mode="auto", sigma_target=0.2))
        # kind = single: the other keys of the section are not read, whatever they hold
        single = load(model_input("kind = single\nn_det_max = auto\ncutoff = 7, x\nseries = 0\nqp2_basis = a b\n",
                                  "[system]\ncharge = 1\nmultiplicity = 2\n"))
        self.assertIsNone(single.trial)
        self.assertNotIn("trial", single.summary())
        self.assertEqual(load(model_input("kind = single\n")).summary(), plain.summary())
        self.assertEqual(load(model_input("kind = single\n")).qmc, plain.qmc)

    def test_template_is_still_a_valid_input(self):
        self.assertIn("\n[trial]\n", TEMPLATE)
        self.assertLess(TEMPLATE.index("[qmc]"), TEMPLATE.index("[trial]"))
        self.assertLess(TEMPLATE.index("[trial]"), TEMPLATE.index("[search]"))
        with tempfile.TemporaryDirectory() as tmp:
            write_template(os.path.join(tmp, "input.ini"))
            for name in ("reactant.xyz", "product.xyz"):
                G.write_xyz(os.path.join(tmp, name), ["O", "H", "H"], M.water_geometry())
            cfg = Config.load(os.path.join(tmp, "input.ini"))
        self.assertIsNone(cfg.trial)
        self.assertEqual(cfg.qmc, self.QMC)
        self.assertEqual(sorted(cfg.summary()), self.SUMMARY_KEYS)

    def test_multidet_values(self):
        cfg = load(model_input("kind = multidet\nn_det_max = 10000\ncutoff = 1e-3, HF, 2.5e-4\n"))
        self.assertEqual(cfg.trial, {"kind": "multidet", "n_det_max": 10000, "cutoffs": [1e-3, "hf", 2.5e-4], "series": 2,
                                     "qp2_basis": "cc-pvtz_ecp_ccecp", "qp2_pseudo": "ccecp", "not_converged": "stop",
                                     # the keys of the rule for the size of the expansion: a list of cutoffs is the
                                     # manual mode (tests/test_trial_auto.py has the automatic mode)
                                     "tolerance": 1.0, "series_max": 2, "auto": None})
        self.assertEqual(load(model_input("kind = multidet\ncutoff = hf\nnot_converged = Run\n")).trial["not_converged"], "run")
        self.assertEqual(cfg.summary()["trial"], cfg.trial)
        self.assertEqual(sorted(cfg.summary()), sorted(self.SUMMARY_KEYS + ["trial"]))
        self.assertEqual(cfg.qmc, self.QMC)                             # the settings of the search points do not change
        # every cutoff is hf: no CIPSI wave function, so no size is needed
        cfg = load(model_input("kind = multidet\ncutoff = hf\nseries = 3\n"))
        self.assertEqual((cfg.trial["n_det_max"], cfg.trial["cutoffs"], cfg.trial["series"]), (None, ["hf"], 3))
        # the engine accepts what the input section accepts
        for cutoff in load(model_input("kind = multidet\nn_det_max = 2\ncutoff = 0.5, hf\n")).trial["cutoffs"]:
            MD.validate_trial(MD.trial_of({"trial": {"kind": "multidet", "cutoff": cutoff, "n_det_max": 2}}))

    def test_names_of_the_quantum_package_files(self):
        table = {"ccecpccpvdz": "cc-pvdz_ecp_ccecp", "ccecpccpvtz": "cc-pvtz_ecp_ccecp", "ccecpccpvqz": "cc-pvqz_ecp_ccecp",
                 "ccecpccpv5z": "cc-pv5z_ecp_ccecp", "ccECP-cc-pVTZ": "cc-pvtz_ecp_ccecp"}
        for basis, expected in table.items():
            cfg = load(model_input("kind = multidet\ncutoff = hf\n", "[qmc]\nbasis = %s\n" % basis))
            self.assertEqual((cfg.trial["qp2_basis"], cfg.trial["qp2_pseudo"]), (expected, "ccecp"), basis)
        text = problems(model_input("kind = multidet\ncutoff = hf\n", "[qmc]\nbasis = def2-tzvp\necp = stuttgart\n"))
        self.assertIn("[trial] qp2_basis = auto", text)
        self.assertIn("[qmc] basis = def2-tzvp", text)
        self.assertIn("Set [trial] qp2_basis", text)
        self.assertIn("[trial] qp2_pseudo = auto", text)
        self.assertIn("Set [trial] qp2_pseudo", text)
        cfg = load(model_input("kind = multidet\ncutoff = hf\nqp2_basis = def2-tzvp_x\nqp2_pseudo = my.pp\n",
                               "[qmc]\nbasis = def2-tzvp\necp = stuttgart\n"))
        self.assertEqual((cfg.trial["qp2_basis"], cfg.trial["qp2_pseudo"]), ("def2-tzvp_x", "my.pp"))
        self.assertIn("[trial] qp2_basis = 'cc pvtz; rm' is not a file name",
                      problems(model_input("kind = multidet\ncutoff = hf\nqp2_basis = cc pvtz; rm\n")))

    def test_every_error_names_the_section_and_the_key(self):
        cases = [
            ("kind = multidet\nn_det_max = 100\n", "[trial] cutoff is needed for kind = multidet"),
            ("kind = multidet\ncutoff = 1e-3\n", "[trial] n_det_max is needed for a numeric cutoff"),
            ("kind = multidet\nn_det_max = auto\ncutoff = 1e-3\n", "[trial] n_det_max = auto is not built yet"),
            ("kind = multidet\nn_det_max = 100\ncutoff = auto, 1e-3\n", "[trial] cutoff = auto, 1e-3: the word auto must stand alone"),
            ("kind = multidet\nn_det_max = 100\ncutoff = 1.5\n", "[trial] cutoff = 1.5 must be between 0 and 1"),
            ("kind = multidet\nn_det_max = 100\ncutoff = 0\n", "[trial] cutoff = 0 must be between 0 and 1"),
            ("kind = multidet\nn_det_max = 100\ncutoff = 1\n", "[trial] cutoff = 1 must be between 0 and 1"),
            ("kind = multidet\nn_det_max = 100\ncutoff = -1e-3\n", "[trial] cutoff = -1e-3 must be between 0 and 1"),
            ("kind = multidet\nn_det_max = 100\ncutoff = small\n", "[trial] cutoff = 'small' is not a number"),
            ("kind = multidet\nn_det_max = 1\ncutoff = 1e-3\n", "[trial] n_det_max = 1 must be at least 2"),
            ("kind = multidet\nn_det_max = 1e4\ncutoff = 1e-3\n", "[trial] n_det_max = '1e4' is not a whole number"),
            ("kind = multidet\nn_det_max = 100\ncutoff = 1e-3, 0.001\n", "[trial] cutoff lists the same cutoff twice: 1e-3 and 0.001"),
            ("kind = multidet\nn_det_max = 100\ncutoff = 1e-3, 1.0000001e-3\n",
             "apart by their first %d significant digits" % MD.CUT_DIGITS),
            ("kind = multidet\ncutoff = hf\nnot_converged = maybe\n", "[trial] not_converged = 'maybe'; allowed: stop, run"),
            ("kind = multidet\ncutoff = hf, HF\n", "[trial] cutoff lists the same cutoff twice: hf and HF"),
            ("kind = multidet\nn_det_max = 100\ncutoff = 1e-3\nseries = 0\n", "[trial] series = 0 must be at least 1"),
            ("kind = casscf\n", "[trial] kind = 'casscf'; allowed: single, multidet"),
            ("kind = multidet\ncutoff = hf\ncolour = 1\n", "[trial] unknown key 'colour'"),
        ]
        for text, expected in cases:
            self.assertIn(expected, problems(model_input(text)), text)
        for system in ("charge = 1\n", "multiplicity = 3\n"):
            self.assertIn("[trial] kind = multidet needs [system] charge = 0 and multiplicity = 1",
                          problems(model_input("kind = multidet\ncutoff = hf\n", "[system]\n" + system)))
        # one input with several mistakes: every one of them is reported
        text = problems(model_input("kind = multidet\nn_det_max = 1\ncutoff = 2, auto\nseries = 0\n"))
        for expected in ("n_det_max = 1 must", "cutoff = 2 must", "the word auto must stand alone", "series = 0 must"):
            self.assertIn(expected, text)


# ===================================================================================== the points and their record
class TestTrialRuns(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.folder = os.path.join(tmp.name, "work", "trial")
        self.symbols = ["O", "H", "H"]
        x = M.water_geometry()
        moved = x.copy(); moved[1] += 0.01
        # the search moved the first target; the second one stayed where it started (within 1e-6 angstrom)
        self.geometries = {"reactant": (x, moved), "saddle": (x + 0.2, x + 0.2 + 4.0e-7)}
        self.engine = StandIn()
        self.log = []

    def runs(self, trial=None, engine=None, blocks=120, **kw):
        return T.TrialRuns(engine or self.engine, self.symbols, kw.get("geometries", self.geometries),
                           trial or TRIAL_DICT, SETTINGS, blocks, self.folder, log=self.log.append)

    def state(self):
        return json.loads(read(os.path.join(self.folder, "state.json")))

    def finish_all(self, engine=None):
        engine = engine or self.engine
        for i, key in enumerate(engine.points):
            engine.finish(key, mean=-17.2 - 0.001 * i)

    def test_names(self):
        self.assertEqual(T.entry_label(1e-3, 10000), "n10000_c1e-03")
        self.assertEqual(T.entry_label(2.5e-4, 50000), "n50000_c2.5e-04")
        self.assertEqual(T.entry_label("hf", 10000), "hf")
        self.assertEqual(T.entry_label("hf", None), "hf")
        self.assertEqual(T.qp2_key("min", "final", TRIAL_DICT), "trial/min/final/qp2_n10000")
        self.assertEqual(T.qp2_key("min", "start", dict(TRIAL_DICT, cutoffs=["hf"], n_det_max=None)), "trial/min/start/qp2")
        self.assertEqual(T.point_keys("min", "start", "hf", 3),
                         ["trial/min/start/hf", "trial/min/start/hf_s2", "trial/min/start/hf_s3"])

    def test_keys_settings_and_order_of_the_requests(self):
        runs = self.runs()
        self.assertEqual(self.engine.calls, [])                        # nothing is asked for before the first pass
        self.assertEqual(runs.tick(), "RUNNING")
        expected = []
        for target, geometry in (("reactant", "start"), ("reactant", "final"), ("saddle", "start")):
            for label in ("n10000_c1e-03", "hf"):
                expected += ["trial/%s/%s/%s" % (target, geometry, label), "trial/%s/%s/%s_s2" % (target, geometry, label)]
        self.assertEqual(self.engine.calls, expected)                  # a further series follows its series-1 point
        self.assertEqual(runs.keys(), expected)
        self.assertEqual(len(self.engine.specs), 12)
        for key, spec in self.engine.specs.items():
            target, geometry = key.split("/")[1:3]
            second = key.endswith("_s2")
            self.assertEqual((spec.role, spec.blocks, spec.symbols), ("final", 120, self.symbols))
            self.assertEqual((spec.series, spec.base_key), (2, key[:-3]) if second else (1, None))
            np.testing.assert_array_equal(spec.coords, self.geometries[target][0 if geometry == "start" else 1])
            hf = "/hf" in key
            self.assertEqual(spec.settings["trial"], {
                "kind": "multidet", "cutoff": "hf" if hf else 1e-3, "n_det_max": None if hf else 10000,
                "qp2_key": "trial/%s/%s/qp2_n10000" % (target, geometry),      # one folder for all cutoffs of a geometry
                "qp2_basis": "cc-pvtz_ecp_ccecp", "qp2_pseudo": "ccecp"})
            self.assertEqual({k: v for k, v in spec.settings.items() if k != "trial"}, SETTINGS)
            # the engine accepts the settings of every point
            MD.validate_trial(MD.trial_of(spec.settings), spec.settings["charge"], spec.settings["spin"])
        self.assertNotIn("trial", SETTINGS)                            # the settings that were given are not changed

    def test_record_on_disk(self):
        runs = self.runs()
        s = self.state()                                               # the plan is on disk before any point exists
        self.assertEqual(s["points"], {})
        runs.tick()
        s = self.state()
        self.assertEqual((s["version"], s["status"]), (1, "RUNNING"))
        self.assertEqual(s["rule"]["trial"], TRIAL_DICT)
        self.assertEqual(s["rule"]["blocks"], 120)
        self.assertIn("start geometry and at the final geometry of every target", s["rule"]["text"])
        self.assertEqual(s["rule"]["energy"], {"drop": "auto", "drop_min": 10, "drop_max": 50, "min_blocks_after_drop": 30})
        self.assertEqual(list(s["targets"]), ["reactant", "saddle"])
        r = s["targets"]["reactant"]
        self.assertFalse(r["same_geometry"])
        self.assertEqual(list(r["geometries"]), ["start", "final"])
        self.assertEqual(list(r["geometries"]["final"]["cutoffs"]), ["n10000_c1e-03", "hf"])
        cut = r["geometries"]["final"]["cutoffs"]["n10000_c1e-03"]
        self.assertEqual(cut, {"cutoff": 1e-3, "n_det_max": 10000,
                               "keys": ["trial/reactant/final/n10000_c1e-03", "trial/reactant/final/n10000_c1e-03_s2"]})
        self.assertEqual(r["geometries"]["final"]["qp2_key"], "trial/reactant/final/qp2_n10000")
        np.testing.assert_allclose(r["geometries"]["final"]["coords"], self.geometries["reactant"][1], atol=0)
        self.assertEqual(sorted(s["points"]), sorted(runs.keys()))
        self.assertEqual(s["points"]["trial/reactant/start/hf"],
                         {"state": "QUEUED", "message": "waits for the Quantum Package calculation (not started)", "blocks": 0})
        self.assertEqual(runs.message, "0 of 12 DMC run(s) are DONE, 0 write blocks, 12 wait")
        self.assertTrue(any("[trial] plan: 12 DMC run(s) of 120 blocks" in line for line in self.log), self.log)
        self.assertEqual([f for f in os.listdir(self.folder) if f.endswith(".tmp")], [])

    def test_a_geometry_that_did_not_move_is_computed_once(self):
        runs = self.runs()
        runs.tick()
        s = self.state()["targets"]["saddle"]
        self.assertTrue(s["same_geometry"])
        self.assertEqual(list(s["geometries"]), ["start"])
        self.assertIn("only the start geometry is computed", s["note"])
        self.assertLess(s["largest_displacement_angstrom"], 1e-6)
        self.assertFalse(any(key.startswith("trial/saddle/final") for key in self.engine.calls))
        self.assertTrue(any("target saddle: the final geometry equals the start geometry" in line for line in self.log))
        self.assertTrue(runs.results()["targets"]["saddle"]["same_geometry"])
        # a displacement just above the limit is a second geometry
        x = self.geometries["saddle"][0]
        other = T.TrialRuns(StandIn(), self.symbols, {"saddle": (x, x + 2.0e-6)}, TRIAL_DICT, SETTINGS, 120,
                            os.path.join(self.folder, "other"))
        self.assertEqual(list(other.state["targets"]["saddle"]["geometries"]), ["start", "final"])

    def test_every_cutoff_hf_needs_no_cipsi_folder(self):
        trial = dict(TRIAL_DICT, cutoffs=["hf"], n_det_max=None, series=1)
        runs = self.runs(trial)
        runs.tick()
        self.assertEqual(self.engine.calls, ["trial/reactant/start/hf", "trial/reactant/final/hf", "trial/saddle/start/hf"])
        self.assertEqual({s.settings["trial"]["qp2_key"] for s in self.engine.specs.values()},
                         {"trial/reactant/start/qp2", "trial/reactant/final/qp2", "trial/saddle/start/qp2"})

    def test_done_running_and_needs_attention(self):
        runs = self.runs()
        self.assertEqual(runs.tick(), "RUNNING")
        keys = runs.keys()
        for key in keys[:-1]:
            self.engine.finish(key)
        self.engine.points[keys[-1]].update(state=RUNNING, energy=np.full(40, -17.2), message="40 of 120 blocks")
        self.assertEqual(runs.tick(), "RUNNING")
        self.assertEqual(runs.message, "11 of 12 DMC run(s) are DONE, 1 write blocks, 0 wait")
        self.assertEqual(self.state()["points"][keys[-1]], {"state": "RUNNING", "message": "40 of 120 blocks", "blocks": 40})
        self.engine.finish(keys[-1])
        self.assertEqual(runs.tick(), "DONE")
        self.assertEqual((runs.status, runs.message), ("DONE", "all 12 DMC run(s) are DONE"))
        self.assertEqual(self.state()["status"], "DONE")
        self.assertEqual(sum("DONE after 120 block(s)" in line for line in self.log), 12)      # every end is logged once
        # two points fail: the message names each one with its reason, and the other points are still read
        self.engine.points[keys[0]].update(state=FAILED,
                                           message="the Quantum Package calculation failed: step cipsi: RUN: exit code 1")
        self.engine.points[keys[1]].update(state=FAILED, message="the base point %s ended without orbitals" % keys[0])
        self.assertEqual(runs.tick(), "NEEDS_ATTENTION")
        self.assertEqual(runs.message, "%s: the Quantum Package calculation failed: step cipsi: RUN: exit code 1; "
                                       "%s: the base point %s ended without orbitals" % (keys[0], keys[1], keys[0]))
        self.assertEqual(self.state()["points"][keys[5]]["state"], "DONE")
        self.assertEqual(runs.tick(), "NEEDS_ATTENTION")               # it stays until the point is repaired
        # a run that was stopped by hand can never be complete: a person must look
        self.finish_all()
        self.engine.points[keys[3]].update(state=STOPPED, energy=np.full(50, -17.2))
        self.assertEqual(runs.tick(), "NEEDS_ATTENTION")
        self.assertEqual(runs.message, "%s: stopped after 50 of 120 blocks; the stage needs every run complete" % keys[3])

    def test_a_refused_request_waits_until_the_other_points_have_ended(self):
        first = "trial/reactant/final/hf"
        self.engine.refuse[first] = "the QP2 folder holds another geometry"
        runs = self.runs()
        problems = ("%s: the point could not be submitted: the QP2 folder holds another geometry; "
                    "%s_s2: its series-1 point %s could not be submitted" % (first, first, first))
        # the other 10 points can make progress: the stage goes on, and its message names the 2 problems
        self.assertEqual(runs.tick(), "RUNNING")
        self.assertEqual(runs.message, "0 of 12 DMC run(s) are DONE, 0 write blocks, 10 wait; 2 run(s) need attention "
                                       "when the others have ended: " + problems)
        self.assertEqual(len(self.engine.points), 10)                  # the other 10 points exist
        self.assertEqual(self.state()["points"][first]["state"], "ABSENT")
        self.assertNotIn(first + "_s2", self.engine.calls)             # no request that must fail
        # when nothing else can make progress, a person must look
        self.finish_all()
        self.assertEqual(runs.tick(), "NEEDS_ATTENTION")
        self.assertEqual(runs.message, problems)
        del self.engine.refuse[first]
        self.assertEqual(runs.tick(), "RUNNING")                       # the next pass asks again
        self.assertEqual(len(self.engine.points), 12)
        self.assertEqual(runs.message, "10 of 12 DMC run(s) are DONE, 0 write blocks, 2 wait")

    def test_a_failed_point_does_not_stop_the_points_that_can_still_run(self):
        runs = self.runs()
        runs.tick()
        keys = runs.keys()
        self.engine.points[keys[0]].update(state=FAILED, message="job 5 ended in state FAILED")
        self.assertEqual(runs.tick(), "RUNNING")                       # 11 points can still make progress
        self.assertIn("1 run(s) need attention when the others have ended: %s: job 5 ended in state FAILED" % keys[0],
                      runs.message)
        for key in keys[1:]:
            self.engine.finish(key)
        self.assertEqual(runs.tick(), "NEEDS_ATTENTION")
        self.assertEqual(runs.message, "%s: job 5 ended in state FAILED" % keys[0])

    def test_a_refusal_of_the_batch_system_is_no_problem_of_the_point(self):
        first = "trial/reactant/start/n10000_c1e-03"
        self.engine.transient[first] = "sbatch: error: Socket timed out on send/recv operation"
        runs = self.runs()
        self.assertEqual(runs.tick(), "RUNNING")
        self.assertEqual(self.state()["points"][first],
                         {"state": "ABSENT", "blocks": 0, "message": "the request is made again on the next pass: "
                                                                   "sbatch: error: Socket timed out on send/recv operation"})
        self.assertNotIn("need attention", runs.message)
        self.finish_all()                                              # every other point ends: the stage still waits
        self.assertEqual(runs.tick(), "RUNNING")
        del self.engine.transient[first]
        self.assertEqual(runs.tick(), "RUNNING")                       # the request is made again and succeeds
        self.assertIn(first, self.engine.points)

    def test_the_cost_of_the_shared_parent_stays_in_the_tables_after_a_plan_change(self):
        class WithParent(StandIn):
            def qp2_info(self, qp2_key):
                return {"owner": qp2_key.rsplit("/", 1)[0] + "/n10000_c1e-03", "parent_core_hours": 7.0}
        engine = WithParent()
        runs = self.runs(engine=engine)
        runs.tick()
        self.finish_all(engine)
        res = runs.results()
        rows = [r for r in res["rows"] if (r["target"], r["geometry"]) == ("reactant", "start")]
        self.assertEqual([r["cost"] for r in rows], [3.0, 3.0])        # the owner row holds the parent itself (engine)
        # the cutoff 1e-3 leaves the plan: its old point owned the parent, so the first row of the geometry gets it
        other = self.runs(dict(TRIAL_DICT, cutoffs=["hf", 2e-4]), engine=engine)
        other.tick()
        self.finish_all(engine)
        rows = [r for r in other.results()["rows"] if (r["target"], r["geometry"]) == ("reactant", "start")]
        self.assertEqual([r["label"] for r in rows], ["hf", "n10000_c2e-04"])
        self.assertEqual([r["cost"] for r in rows], [3.0 + 7.0, 3.0])

    def test_the_shared_parent_is_counted_once_when_its_owner_left_the_plan(self):
        class WithParent(StandIn):
            """As the real engine: the point that asked for the parent calculation first carries its core-hours."""

            def qp2_info(self, qp2_key):
                return {"owner": qp2_key.rsplit("/", 1)[0] + "/n10000_c1e-03", "parent_core_hours": 7.0}

            def cost(self, key):
                return super().cost(key) + (7.0 if key in self.points and key.endswith("/n10000_c1e-03") else 0.0)
        engine = WithParent()
        runs = self.runs(engine=engine)
        runs.tick()
        self.finish_all(engine)
        res = runs.results()
        rows = [r for r in res["rows"] if (r["target"], r["geometry"]) == ("reactant", "start")]
        self.assertEqual([r["cost"] for r in rows], [3.0 + 7.0, 3.0])  # the row of the owner holds the parent
        self.assertEqual((res["left_plan"], res["cost_left_plan"]), ([], 0.0))
        # 12 runs of 1.5 core-hours, and the parent of each of the 3 geometries once
        self.assertAlmostEqual(sum(r["cost"] for r in res["rows"]) + res["cost_left_plan"], 12 * 1.5 + 3 * 7.0)
        # the cutoff 1e-3 leaves the plan. Its series-1 points own the parents, and they are now points of an older plan
        other = self.runs(dict(TRIAL_DICT, cutoffs=["hf", 2e-4]), engine=engine)
        other.tick()
        self.finish_all(engine)
        res = other.results()
        rows = [r for r in res["rows"] if (r["target"], r["geometry"]) == ("reactant", "start")]
        self.assertEqual([r["cost"] for r in rows], [3.0 + 7.0, 3.0])  # the first row of the geometry holds the parent
        self.assertEqual(len(res["left_plan"]), 6)
        self.assertAlmostEqual(res["cost_left_plan"], 6 * 1.5)         # the 6 old runs, without the 3 parents
        # every parent is counted once: 18 runs, 3 parents
        self.assertAlmostEqual(sum(r["cost"] for r in res["rows"]) + res["cost_left_plan"], 18 * 1.5 + 3 * 7.0)
        # an owner whose folder was deleted has no cost of its own: nothing is taken from the other old runs
        owner = "trial/reactant/start/n10000_c1e-03"
        del engine.points[owner], engine.specs[owner]
        self.assertAlmostEqual(other.results()["cost_left_plan"], 5 * 1.5)

    def test_a_refusal_that_lasts_more_than_an_hour_is_a_problem(self):
        now = [1000.0]
        first = "trial/reactant/start/n10000_c1e-03"
        self.engine.transient[first] = "sbatch: error: Requested time limit is invalid (missing or exceeds some limit)"
        runs = T.TrialRuns(self.engine, self.symbols, self.geometries, TRIAL_DICT, SETTINGS, 120, self.folder,
                           log=self.log.append, clock=lambda: now[0])
        self.assertEqual(runs.tick(), "RUNNING")
        self.assertIn("the batch system refused 2 request(s)", runs.message)
        self.assertEqual(sum("sbatch: error: Requested time limit" in line for line in self.log), 1)   # logged once
        self.finish_all()
        now[0] += T.REFUSAL_SECONDS - 1
        self.assertEqual(runs.tick(), "RUNNING")                       # not yet: the batch system may come back
        now[0] += 2
        self.assertEqual(runs.tick(), "NEEDS_ATTENTION")               # every other point ended; this one is stuck
        self.assertIn("%s: refused for 60 minutes: the request is made again on the next pass: sbatch" % first, runs.message)
        self.assertIn("%s_s2: refused for 60 minutes: the request is made again on the next pass, after its series-1 "
                      "point %s" % (first, first), runs.message)
        # 2 lines in the log: the first refusal, and the pass in which it became a problem
        self.assertEqual(sum("sbatch: error: Requested time limit" in line for line in self.log), 2)

    def test_a_missing_file_is_a_permanent_refusal(self):
        first = "trial/reactant/start/n10000_c1e-03"

        class Missing(StandIn):
            def submit(self, spec):
                if spec.key == first:
                    raise FileNotFoundError("pseudopotential file missing for element O")
                return super().submit(spec)
        engine = Missing()
        runs = self.runs(engine=engine)
        runs.tick()
        self.assertIn("%s: the point could not be submitted: pseudopotential file missing" % first, runs.message)
        for key in list(engine.points):
            engine.finish(key)
        self.assertEqual(runs.tick(), "NEEDS_ATTENTION")

    def test_numeric_cutoffs_are_asked_for_before_hf(self):
        runs = self.runs(dict(TRIAL_DICT, cutoffs=["hf", 1e-3]))
        runs.tick()
        start = [k for k in self.engine.calls if k.startswith("trial/reactant/start/")]
        self.assertEqual(start, ["trial/reactant/start/n10000_c1e-03", "trial/reactant/start/n10000_c1e-03_s2",
                                 "trial/reactant/start/hf", "trial/reactant/start/hf_s2"])
        self.assertEqual(runs.keys(), self.engine.calls)
        # the record and the tables keep the order of the input
        self.assertEqual(list(self.state()["targets"]["reactant"]["geometries"]["start"]["cutoffs"]), ["hf", "n10000_c1e-03"])
        self.assertEqual([r["label"] for r in runs.results()["rows"]][:2], ["hf", "n10000_c1e-03"])

    def test_points_that_leave_the_plan_are_stopped(self):
        runs = self.runs(dict(TRIAL_DICT, cutoffs=[1e-3, "hf", 1e-4]))
        runs.tick()
        keys = runs.keys()
        done = "trial/reactant/start/n10000_c1e-04"
        self.engine.finish(done)                                       # an ended point is left alone
        runs.tick()
        # the cutoff 1e-4 leaves the input, and the series go from 2 to 1
        fewer = self.runs(dict(TRIAL_DICT, cutoffs=[1e-3, "hf"], series=1))
        gone = [k for k in keys if k not in fewer.keys()]
        self.assertEqual(len(gone), 12)
        self.assertEqual(self.engine.stopped, [])                     # nothing is stopped before the new plan is asked for
        self.assertEqual(sorted(fewer.state["to_stop"]), sorted(gone))
        calls = len(self.engine.calls)
        fewer.tick()
        self.assertGreater(len(self.engine.calls), calls)
        self.assertEqual(sorted(self.engine.stopped), sorted(k for k in gone if k != done))
        self.assertEqual(sorted(fewer.state["left_plan"]), sorted(gone))
        self.assertEqual(fewer.state["to_stop"], [])
        self.assertEqual(sorted(fewer.stopped), sorted(self.engine.stopped))
        self.assertEqual(self.engine.points[done]["state"], DONE)
        for key in fewer.keys():                                       # the points of the new plan go on
            self.assertNotIn(key, self.engine.stopped)
        self.assertEqual(sum("stopped, because the plan of the stage no longer holds it" in line for line in self.log), 11)
        # a second object with the same plan stops nothing
        self.engine.stopped.clear()
        self.runs(dict(TRIAL_DICT, cutoffs=[1e-3, "hf"], series=1)).tick()
        self.assertEqual(self.engine.stopped, [])
        # the report counts the core-hours of the points that left the plan
        res = fewer.results()
        self.assertEqual(sorted(res["left_plan"]), sorted(gone))
        self.assertAlmostEqual(res["cost_left_plan"], 1.5)            # the one finished point of the stand-in

    def test_retry(self):
        runs = self.runs()
        runs.tick()
        keys = runs.keys()
        self.finish_all()
        self.assertEqual(runs.retry(), [])                             # nothing has failed
        for key in (keys[5], keys[4]):                                 # a series-2 point and its series-1 point
            self.engine.points[key].update(state=FAILED, message="job 7 ended in state FAILED")
        self.assertEqual(runs.tick(), "NEEDS_ATTENTION")
        self.assertEqual(runs.retry(), [keys[4], keys[5]])             # the series-1 point first
        self.assertEqual(self.engine.retried, [keys[4], keys[5]])
        self.assertEqual(runs.tick(), "RUNNING")
        self.assertEqual(runs.message, "10 of 12 DMC run(s) are DONE, 0 write blocks, 2 wait")
        # an engine without the method retry: nothing can be done, and nothing breaks
        class Plain(StandIn):
            retry = None
        plain = T.TrialRuns(Plain(), self.symbols, self.geometries, TRIAL_DICT, SETTINGS, 120, os.path.join(self.folder, "p"))
        self.assertEqual(plain.retry(), [])

    def test_a_second_object_on_the_same_folder_submits_nothing_new(self):
        runs = self.runs()
        runs.tick()
        self.finish_all()
        self.assertEqual(runs.tick(), "DONE")
        before, specs, lines = self.state(), dict(self.engine.specs), len(self.log)
        again = self.runs()                                            # a new process: only the files are shared
        self.assertEqual(again.status, "DONE")                         # what the last pass saw is kept
        self.assertEqual(again.tick(), "DONE")
        self.assertEqual(self.engine.specs, specs)                     # the same 12 points, the same requests
        self.assertEqual(len(self.engine.points), 12)
        self.assertEqual(self.state(), before)
        self.assertEqual(len(self.log), lines)                         # nothing new happened, nothing new is logged
        # a cutoff more: only its points are new; a point that exists is not computed again
        more = self.runs(dict(TRIAL_DICT, cutoffs=[1e-3, "hf", 1e-4]))
        self.assertEqual((more.status, more.message), ("RUNNING", ""))   # other points than before: the old status is gone
        self.assertEqual(more.tick(), "RUNNING")
        new = sorted(set(self.engine.specs) - set(specs))
        self.assertEqual(len(new), 6)
        self.assertTrue(all("n10000_c1e-04" in key for key in new))
        self.assertEqual(more.message, "12 of 18 DMC run(s) are DONE, 0 write blocks, 6 wait")

    def test_other_quantum_package_files_are_refused(self):
        self.runs().tick()
        with self.assertRaises(ValueError) as ctx:
            self.runs(dict(TRIAL_DICT, qp2_basis="cc-pvqz_ecp_ccecp"))
        self.assertIn("was written with [trial] qp2_basis = cc-pvtz_ecp_ccecp", str(ctx.exception))
        self.assertIn("the input now gives cc-pvqz_ecp_ccecp", str(ctx.exception))
        with self.assertRaises(ValueError):
            self.runs(dict(TRIAL_DICT, qp2_pseudo="other"))
        write(os.path.join(self.folder, "state.json"), "{ not json")
        with self.assertRaises(ValueError) as ctx:
            self.runs()
        self.assertIn("could not be read", str(ctx.exception))

    def test_wrong_arguments(self):
        for bad in (dict(TRIAL_DICT, kind="single"), dict(TRIAL_DICT, cutoffs=[]), dict(TRIAL_DICT, series=0),
                    dict(TRIAL_DICT, n_det_max=None)):
            with self.assertRaises(ValueError, msg=str(bad)):
                self.runs(bad)
        with self.assertRaises(ValueError):
            self.runs(blocks=0)
        with self.assertRaises(ValueError):
            self.runs(geometries={"a": (np.zeros((2, 3)), np.zeros((2, 3)))})       # 3 symbols, 2 atoms

    def test_results_use_the_estimator_of_the_final_energies(self):
        runs = self.runs()
        runs.tick()
        key = "trial/reactant/final/n10000_c1e-03"
        self.engine.finish(key, mean=-17.2100, noise=2.0e-3, cost=3.0)
        self.engine.finish(key + "_s2", mean=-17.2110, noise=1.0e-3, cost=1.0)
        res = runs.results()
        self.assertEqual(sorted(res), ["cost_left_plan", "left_plan", "rows", "rule", "targets"])
        self.assertEqual([(r["target"], r["geometry"], r["label"]) for r in res["rows"]],
                         [("reactant", "start", "n10000_c1e-03"), ("reactant", "start", "hf"),
                          ("reactant", "final", "n10000_c1e-03"), ("reactant", "final", "hf"),
                          ("saddle", "start", "n10000_c1e-03"), ("saddle", "start", "hf")])
        row = res["rows"][2]
        self.assertEqual(row["keys"], [key, key + "_s2"])
        self.assertEqual((row["states"], row["done"]), (["DONE", "DONE"], True))
        self.assertEqual((row["cutoff"], row["n_det_max"], row["series"], row["series_planned"]), (1e-3, 10000, 2, 2))
        # the report computes the final energies of the search with Reader.stat: the same numbers come out here
        stat = tables.Reader(self.engine).stat([key, key + "_s2"], tables.DropRule())
        self.assertAlmostEqual(row["energy"], stat.value, places=12)
        self.assertAlmostEqual(row["error"], stat.error, places=12)
        self.assertEqual((row["blocks"], row["blocks_used"]), (stat.blocks, stat.used))
        # and the two runs are weighted with 1 / error^2
        parts = [T.run_energy(self.engine.read(k)) for k in (key, key + "_s2")]
        w = [1.0 / p[1] ** 2 for p in parts]
        self.assertAlmostEqual(row["energy"], (w[0] * parts[0][0] + w[1] * parts[1][0]) / (w[0] + w[1]), places=12)
        self.assertAlmostEqual(row["error"], 1.0 / np.sqrt(w[0] + w[1]), places=12)
        self.assertLess(row["error"], min(p[1] for p in parts))
        self.assertEqual(row["trial"]["n_det"], 55)
        self.assertEqual(row["trial"]["n_det_parent"], 3280)
        self.assertEqual(row["checks"]["mdvmc"]["verdict"], "PASS")
        self.assertAlmostEqual(row["cost"], 4.0)
        # a point that waits: no energy, the facts that are known, no check
        wait = res["rows"][0]
        self.assertEqual((wait["done"], wait["energy"], wait["error"], wait["series"], wait["blocks"]), (False, None, None, 0, 0))
        self.assertEqual(wait["trial"], {"kind": "multidet", "cutoff": 1e-3, "orbitals": "qp2_rhf"})
        self.assertEqual(wait["checks"], {})
        self.assertIsNone(T.run_energy(PointData(key="x", state=RUNNING, energy=np.zeros(35), weight=np.ones(35))))

    def test_plan_lines_and_machine_problems(self):
        lines = T.plan_lines(TRIAL_DICT, ["reactant", "saddle"])
        text = "\n".join(lines)
        for word in ("geometries: up to 4", "Quantum Package parent calculations: up to 4",
                     "CIPSI wave function of 10000 determinants", "basis file cc-pvtz_ecp_ccecp", "cutoffs: 2 (0.001, hf)",
                     "DMC series: 2 for each geometry and each cutoff, up to 16 DMC run(s) in total"):
            self.assertIn(word, text)
        self.assertIn("RHF orbitals only", "\n".join(T.plan_lines(dict(TRIAL_DICT, cutoffs=["hf"]), ["geometry"])))

        class Profile:                                                  # what machine_problems asks a machine profile
            def __init__(self, ranks, lacks=()):
                self.ranks, self.lacks = ranks, list(lacks)

            def check_qp2(self):
                return list(self.lacks)

            def tasks(self, kind):
                return self.ranks

        self.assertEqual(T.machine_problems(Profile(32), TRIAL_DICT), [])
        self.assertEqual(T.machine_problems(Profile(32, ["section [job.qp2] is missing"]), TRIAL_DICT),
                         ["section [job.qp2] is missing"])
        self.assertEqual(T.machine_problems(Profile(256), TRIAL_DICT), [])           # 2 series fit on 256 ranks
        found = T.machine_problems(Profile(256), dict(TRIAL_DICT, series=3))
        self.assertEqual(len(found), 1)
        self.assertIn("[trial] series = 3", found[0])
        self.assertIn("at most 2 DMC runs", found[0])


# ===================================================================================== the real engine
class TestRealEngine(TM.Qp2Case):
    """``TrialRuns`` on ``QmcpackEngine``. The batch system is the fake one of the engine tests; the files that the
    Quantum Package jobs and the QMC jobs would write are written by the test (real outputs for a water molecule)."""

    WORKFLOW_SETTINGS = {"charge": 0, "spin": 0, "xc": "b3lyp", "basis": "ccecpccpvtz", "ecp": "ccecp", "jastrow_reuse": "none",
                         "dmc": {"timestep": 0.005, "walkers": 2048, "steps": 100, "warmup_steps": 2000},
                         "sigma_block": 1.3}                            # what Workflow.trial_settings() gives

    def runs(self, eng, geometries, trial):
        return T.TrialRuns(eng, self.symbols, geometries, trial, self.WORKFLOW_SETTINGS, 200,
                           os.path.join(self.tmp, "project", "work", "trial"))

    def test_the_engine_accepts_every_request_of_the_stage(self):
        eng = self.engine()
        moved = self.coords.copy(); moved[1] += 0.01
        runs = self.runs(eng, {"min": (self.coords, moved)}, dict(TRIAL_DICT, cutoffs=[1e-3, "hf"]))
        self.assertEqual(runs.tick(), "RUNNING")
        self.assertEqual(sorted(eng.keys()), sorted(runs.keys()))       # 8 points; Quantum Package folders are no points
        self.assertEqual(len(runs.keys()), 8)
        # one Quantum Package job per geometry: the two cutoffs share its folder, and no QMC job exists yet
        self.assertEqual([j["script"] for j in self.sched.submitted],
                         [os.path.join(self.root, "trial", "min", g, "qp2_n10000", "job_parent.sh") for g in ("start", "final")])
        self.assertIn("begin_step cipsi", self.sched.submitted[0]["text"])
        for key in runs.keys():
            p = self.read_json(key, "point.json")
            data_ = eng.read(key)
            self.assertEqual((p["role"], p["blocks"], data_.state), ("final", 200, "QUEUED"))
            self.assertEqual((p["settings"]["jastrow_reuse"], p["settings"]["slow_factor"]), ("none", 3.0))
            self.assertEqual(p["trial"]["qp2_basis"], "cc-pvtz_ecp_ccecp")
            if key.endswith("_s2"):
                self.assertEqual((p["series"], p["base_key"], p["steps"]), (2, key[:-3], ["dmc"]))
                self.assertIn("waits for the orbitals and the Jastrow factor of " + key[:-3], data_.message)
            else:
                self.assertEqual(p["steps"], ["scf", "mdc4q", "mdvmc", "opt", "dmc"])
                self.assertEqual(p["qp2_key"], "/".join(key.split("/")[:3]) + "/qp2_n10000")
                self.assertIn("waits for the Quantum Package calculation", data_.message)
        record = json.loads(read(os.path.join(self.tmp, "project", "work", "trial", "state.json")))
        self.assertEqual({v["state"] for v in record["points"].values()}, {"QUEUED"})
        self.assertIn("waits for the Quantum Package calculation", record["points"]["trial/min/final/hf"]["message"])
        # the next pass and a new process ask again: no new job and no new point
        self.assertEqual(runs.tick(), "RUNNING")
        self.assertEqual(self.runs(self.engine(), {"min": (self.coords, moved)}, dict(TRIAL_DICT, cutoffs=[1e-3, "hf"])).tick(),
                         "RUNNING")
        self.assertEqual(len(self.sched.submitted), 2)
        # the same key with another geometry is refused by the engine, and the stage says so
        other = self.runs(self.engine(), {"min": (self.coords, moved + 0.05)}, dict(TRIAL_DICT, cutoffs=[1e-3, "hf"]))
        self.assertEqual(other.tick(), "RUNNING")                      # the points of the start geometry can still run
        self.assertIn("need attention when the others have ended", other.message)
        self.assertIn("trial/min/final/hf: the point could not be submitted: point trial/min/final/hf exists with "
                      "another geometry", other.message)

    def test_the_workflow_gives_the_real_engine_requests_that_it_accepts(self):
        """A model project (its search runs on the mock engine) whose stage ``trial`` talks to the real engine."""
        folder = os.path.join(self.tmp, "water")
        os.makedirs(folder)
        write(os.path.join(folder, "input.ini"), WATER + TRIAL % "1e-3, hf")
        real = self.engine()

        class Backend(ModelBackend):
            both = None

            def engine(self, wf):
                if self.both is None:
                    self.both = TwoEngines(ModelBackend.engine(self, wf), real)
                return self.both

        wf = Workflow(folder, backend=Backend(Config.load(os.path.join(folder, "input.ini"))))
        for _ in range(4000):
            if wf.advance() != "RUNNING" or wf.project.stage("trial")["status"] != "PENDING":
                break
        stages = wf.project.state["stages"]
        self.assertEqual(stages["search"]["status"], "DONE", stages)
        self.assertEqual(stages["trial"], {"status": "RUNNING", "message": "0 of 8 DMC run(s) are DONE, 0 write blocks, 8 wait"})
        self.assertNotIn("restore", stages["trial"])                   # the stage has a record: nothing to give back
        keys = wf.trial_runs().keys()
        self.assertEqual(sorted(real.keys()), sorted(keys))
        # one Quantum Package job for the start geometry and one for the final geometry of the search
        self.assertEqual([j["script"] for j in self.sched.submitted],
                         [os.path.join(self.root, "trial", "geometry", g, "qp2_n10000", "job_parent.sh") for g in ("start", "final")])
        final = wf.driver("geometry").final_geometry()
        for key in keys:
            p = self.read_json(key, "point.json")
            self.assertEqual((p["blocks"], p["role"]), (400, "final"))
            s = p["settings"]
            self.assertEqual((s["jastrow_reuse"], s["slow_factor"], s["charge"], s["spin"], s["basis"], s["ecp"]),
                             ("none", wf.point_settings()["slow_factor"], 0, 0, "ccecpccpvtz", "ccecp"))
            self.assertEqual(s["dmc"], wf.cfg.qmc["dmc"])
            self.assertEqual(p["trial"]["qp2_basis"], "cc-pvtz_ecp_ccecp")
        _symbols, coords = G.read_xyz(self.path("trial/geometry/final/hf", "geom.xyz"))
        np.testing.assert_allclose(coords, final, atol=1e-8)
        # more passes: the engine is asked again and again, and nothing new is submitted
        for _ in range(3):
            self.assertEqual(wf.advance(), "RUNNING")
        self.assertEqual(len(self.sched.submitted), 2)
        self.assertEqual(wf.project.stage("report")["status"], "PENDING")

    # ------------------------------------------------------------------ helpers that play the jobs of one point
    def job_in(self, folder):
        """The last job whose script is in ``folder``."""
        return [j for j in self.sched.submitted if os.path.dirname(j["script"]) == folder][-1]

    def play_qp2(self, eng, qp2_folder, cut="c1e-03"):
        """The Quantum Package jobs of one geometry end: the parent (RHF and CIPSI), then the cut ``cut``."""
        self.folder = qp2_folder
        self.fake_scf(); self.fake_cipsi()
        self.sched.state_of[self.job_in(qp2_folder)["id"]] = TM.S.COMPLETED
        eng.advance()
        self.fake_cut(cut)
        self.sched.state_of[self.job_in(os.path.join(qp2_folder, "cut_" + cut))["id"]] = TM.S.COMPLETED
        eng.advance()

    def play_qmc_until_mdvmc(self, key):
        """The QMC job of the series-1 point ``key`` runs; its steps scf and mdc4q are done. Returns the job id."""
        job = self.job_in(os.path.dirname(self.path(key, "job.sh")))["id"]
        self.sched.state_of[job] = TM.S.RUNNING
        system = self.system(key)
        system["e_rhf"] = TM.E_SCF
        self.write(key, "system.json", json.dumps(system))
        self.write(key, "energy.dat", "%.18e\n0.0\n" % TM.TE.E_SCF)
        self.mark(key, "scf", 1000, 1060)
        block, _facts = MD.read_determinantset(os.path.join(TM.DATA, "md_h2o_c4q.wfj.xml"), "trial.h5")
        # the export of Quantum Package holds the atoms of THIS geometry (in the order O H H of the stored file)
        structure = TM.data("md_h2o_c4q.structure.xml")
        first = structure.index('<attrib name="position" datatype="posArray">') + len('<attrib name="position" datatype="posArray">')
        last = structure.index("</attrib>", first)
        self.assertEqual(system["symbols"], ["O", "H", "H"])
        positions = "\n" + "".join("%18.10e %18.10e %18.10e\n" % tuple(r) for r in system["coords_bohr"])
        structure = structure[:first] + positions + structure[last:]
        for name, text in (("mdc4q.wfj.xml", TM.data("md_h2o_c4q.wfj.xml")), ("trial.h5", "h5\n"),
                           ("mdc4q.structure.xml", structure), ("md_determinantset.xml", block + "\n"),
                           ("mdvmc1.out", "Found 55 terms in the MSD expansion.\n")):
            self.write(key, name, text)
        self.mark(key, "mdc4q", 1060, 1070)
        return job

    def play_qmc_to_the_end(self, eng, key, shift=0.0):
        """The QMC job of the series-1 point ``key`` ends with 200 DMC blocks; ``shift`` (hartree) is added to every
        block energy."""
        job = self.play_qmc_until_mdvmc(key)
        TM.vmc_trace(self.path(key, "mdvmc1.s000.scalar.dat"), TM.E_CUT, seed=1)
        self.mark(key, "mdvmc", 1070, 1100)
        self.fake_opt(key)
        self.fake_rows(key, 200)
        if shift:
            path = self.path(key, "dmc.s001.scalar.dat")
            lines = read(path).split("\n")
            out = [lines[0]]
            for line in lines[1:]:
                cols = line.split()
                if cols:
                    cols[1] = "%.10e" % (float(cols[1]) + shift)
                    line = "  ".join(cols)
                out.append(line)
            write(path, "\n".join(out))
        self.write(key, "dmc.done", "1320\n")
        self.sched.state_of[job] = TM.S.COMPLETED
        return eng.read(key)

    def test_a_new_cutoff_keeps_the_cipsi_job_that_runs(self):
        eng = self.engine()
        trial = dict(TRIAL_DICT, cutoffs=[1e-3], n_det_max=20000, series=2)
        geometries = {"min": (self.coords, self.coords.copy())}
        self.runs(eng, geometries, trial).tick()
        folder = os.path.join(self.root, "trial", "min", "start", "qp2_n20000")
        parent = self.job_in(folder)["id"]
        self.sched.state_of[parent] = TM.S.RUNNING
        # the user replaces the cutoff while the CIPSI job runs
        new = self.runs(eng, geometries, dict(trial, cutoffs=[1e-4]))
        self.assertEqual(new.tick(), "RUNNING")
        self.assertNotIn(parent, self.sched.cancelled)                # the new cutoff needs that job
        old = "trial/min/start/n20000_c1e-03"
        self.assertEqual(sorted(new.stopped), [old, old + "_s2"])
        self.assertEqual(eng.read(old).state, "STOPPED")
        # the job ends: the new cutoff gets its cut job, and nothing failed
        self.folder = folder
        self.fake_scf(); self.fake_cipsi()
        self.sched.state_of[parent] = TM.S.COMPLETED
        eng.advance()
        self.assertEqual(os.path.dirname(self.sched.submitted[-1]["script"]), os.path.join(folder, "cut_c1e-04"))
        self.assertEqual(eng.read("trial/min/start/n20000_c1e-04").state, "QUEUED")

    def test_numeric_cutoffs_first_give_one_parent_job_per_geometry(self):
        eng = self.engine()
        moved = self.coords.copy(); moved[1] += 0.01
        runs = self.runs(eng, {"min": (self.coords, moved)}, dict(TRIAL_DICT, cutoffs=["hf", 1e-3]))
        runs.tick()
        parents = [j for j in self.sched.submitted if j["script"].endswith("job_parent.sh")]
        self.assertEqual(len(parents), 2)                              # one per geometry, not two
        for job in parents:
            self.assertIn("begin_step cipsi", job["text"])             # the first request of the geometry needs CIPSI

    def test_final_minus_start_from_two_finished_geometries(self):
        eng = self.engine()
        moved = self.coords.copy(); moved[1] += 0.01
        trial = dict(TRIAL_DICT, cutoffs=[1e-3], n_det_max=20000, series=1)
        runs = self.runs(eng, {"min": (self.coords, moved)}, trial)
        self.assertEqual(runs.tick(), "RUNNING")
        keys = {g: "trial/min/%s/n20000_c1e-03" % g for g in ("start", "final")}
        shift = 0.0016                                                 # hartree: about 1 kcal/mol
        for geometry, extra in (("start", 0.0), ("final", shift)):
            self.play_qp2(eng, os.path.join(self.root, "trial", "min", geometry, "qp2_n20000"))
            done = self.play_qmc_to_the_end(eng, keys[geometry], shift=extra)
            self.assertEqual((done.state, done.nblocks), ("DONE", 200), done.message)
        self.assertEqual(runs.tick(), "DONE")
        data_ = tables.ReportData(project_dir=os.path.join(self.tmp, "project"), title="water", project={})
        table = RT.collect(data_, eng)
        change = table["changes"][0]
        self.assertIsNone(change["text"])
        start, final = (T.run_energy(eng.read(keys[g])) for g in ("start", "final"))
        self.assertAlmostEqual(change["value"], (final[0] - start[0]) * H, places=9)
        self.assertAlmostEqual(change["error"], float(np.hypot(final[1], start[1])) * H, places=9)
        self.assertAlmostEqual(change["value"], shift * H, places=6)   # the same trace, shifted: exactly the shift

    def test_a_vmc_check_that_failed_in_the_job_is_in_the_report(self):
        eng = self.engine()
        trial = dict(TRIAL_DICT, cutoffs=[1e-3], n_det_max=20000, series=1)
        runs = self.runs(eng, {"min": (self.coords, self.coords.copy())}, trial)
        key = "trial/min/start/n20000_c1e-03"
        runs.tick()
        self.play_qp2(eng, os.path.join(self.root, "trial", "min", "start", "qp2_n20000"))
        job = self.play_qmc_until_mdvmc(key)
        verdict = {"check": "mdvmc", "ok": False, "verdict": "FAIL", "energy": -16.9700, "error": 4.0e-4, "e_ref": TM.E_CUT,
                   "difference": 0.0104, "limit": 1.3e-3, "runs": [{"run": 1, "blocks": 40}],
                   "message": "the difference is above 5 standard errors"}
        self.write(key, "check_mdvmc.json", json.dumps(verdict))
        self.write(key, "mdvmc.failed", "QUALITY: the difference is above 5 standard errors\n")
        self.sched.state_of[job] = TM.S.FAILED
        self.assertEqual(eng.read(key).state, "FAILED")
        self.assertEqual(runs.tick(), "NEEDS_ATTENTION")
        data_ = tables.ReportData(project_dir=os.path.join(self.tmp, "project"), title="water", project={})
        table = RT.collect(data_, eng)
        row = table["rows"][0]
        self.assertEqual(row["checks"]["mdvmc"]["verdict"], "FAIL")
        RT.write_csv(table, os.path.join(self.tmp, "f.csv"))
        RT.write_md(table, os.path.join(self.tmp, "f.md"))
        with open(os.path.join(self.tmp, "f.csv")) as fh:
            line = list(csv.DictReader(fh))[0]
        self.assertEqual((line["vmc_check_verdict"], line["vmc_check_difference_millihartree"]), ("FAIL", "10.4000"))
        self.assertIn("step mdvmc: the difference is above 5 standard errors", line["dmc_messages"])
        text = read(os.path.join(self.tmp, "f.md"))
        self.assertIn("| FAIL | 1 |", text)
        self.assertIn("step mdvmc: the difference is above 5 standard errors", text)

    def test_retry_submits_only_the_failed_points_of_the_plan(self):
        folder = os.path.join(self.tmp, "water")
        os.makedirs(folder)
        write(os.path.join(folder, "input.ini"), WATER + TRIAL % "1e-3, hf")
        real = self.engine()

        class Backend(ModelBackend):
            both = None

            def engine(self, wf):
                if self.both is None:
                    self.both = TwoEngines(ModelBackend.engine(self, wf), real)
                return self.both

        wf = Workflow(folder, backend=Backend(Config.load(os.path.join(folder, "input.ini"))))
        for _ in range(4000):
            if wf.advance() != "RUNNING" or wf.project.stage("trial")["status"] != "PENDING":
                break
        self.assertEqual(wf.project.stage("trial")["status"], "RUNNING")
        start = os.path.join(self.root, "trial", "geometry", "start", "qp2_n10000")
        # the parent job of the start geometry fails; a point of an older plan uses the same parent
        self.folder = start
        self.put("qpcreate.failed", "QUALITY: basis/ao_normalized of the EZFIO is not T\n")
        self.sched.state_of[self.job_in(start)["id"]] = TM.S.FAILED
        spec = wf.trial_runs().specs()[0]
        old = PointSpec(key="trial/geometry/start/n10000_c5e-04", symbols=spec.symbols, coords=spec.coords,
                        blocks=spec.blocks, role="final",
                        settings=dict(spec.settings, trial=dict(spec.settings["trial"], cutoff=5e-4)))
        real.submit(old)
        self.assertEqual(wf.advance(), "RUNNING")                      # the final geometry can still make progress
        self.assertEqual(real.read(old.key).state, "FAILED")
        failed = sorted(k for k in real.failed_keys())
        self.assertIn(old.key, failed)
        self.assertIn("trial/geometry/start/n10000_c1e-03", failed)
        jobs = len(self.sched.submitted)
        os.remove(os.path.join(start, "qpcreate.failed"))
        lines = wf.retry()
        self.assertIn("QMC point trial/geometry/start/n10000_c1e-03: submitted again", lines)
        self.assertIn("QMC point trial/geometry/start/hf: submitted again", lines)
        self.assertFalse([line for line in lines if old.key in line])  # the point of the older plan is not retried
        self.assertEqual(real.read(old.key).state, "FAILED")
        self.assertEqual(len(self.sched.submitted), jobs + 1)          # one new parent job for the start geometry
        self.assertEqual(os.path.dirname(self.sched.submitted[-1]["script"]), start)

    def test_a_finished_geometry_gives_the_facts_and_checks_that_the_report_reads(self):
        eng = self.engine()
        trial = dict(TRIAL_DICT, cutoffs=[1e-3], n_det_max=20000)
        runs = self.runs(eng, {"min": (self.coords, self.coords.copy())}, trial)      # the search did not move the geometry
        key = "trial/min/start/n20000_c1e-03"
        self.assertEqual(runs.keys(), [key, key + "_s2"])
        self.assertEqual(runs.tick(), "RUNNING")
        # the Quantum Package jobs of the geometry: parent (RHF and CIPSI), then the cut
        self.folder = os.path.join(self.root, "trial", "min", "start", "qp2_n20000")
        self.fake_scf(); self.fake_cipsi(); self.end()
        eng.advance()
        self.fake_cut(); self.end()
        eng.advance()
        self.assertEqual(self.sched.submitted[-1]["script"], self.path(key, "job.sh"))
        self.assertEqual(runs.tick(), "RUNNING")
        # what the QMC job of the point writes
        job = self.sched.submitted[-1]["id"]
        self.sched.state_of[job] = TM.S.RUNNING
        system = self.system(key)
        system["e_rhf"] = TM.E_SCF
        self.write(key, "system.json", json.dumps(system))
        self.write(key, "energy.dat", "%.18e\n0.0\n" % TM.TE.E_SCF)
        self.mark(key, "scf", 1000, 1060)
        block, _facts = MD.read_determinantset(os.path.join(TM.DATA, "md_h2o_c4q.wfj.xml"), "trial.h5")
        for name, text in (("mdc4q.wfj.xml", TM.data("md_h2o_c4q.wfj.xml")), ("trial.h5", "h5\n"),
                           ("mdc4q.structure.xml", TM.data("md_h2o_c4q.structure.xml")), ("md_determinantset.xml", block + "\n"),
                           ("mdvmc1.out", "Found 55 terms in the MSD expansion.\n")):
            self.write(key, name, text)
        self.mark(key, "mdc4q", 1060, 1070)
        TM.vmc_trace(self.path(key, "mdvmc1.s000.scalar.dat"), TM.E_CUT, seed=1)
        self.mark(key, "mdvmc", 1070, 1100)
        self.fake_opt(key)
        self.fake_rows(key, 200)
        self.write(key, "dmc.done", "1320\n")
        self.sched.state_of[job] = TM.S.COMPLETED
        first = eng.read(key)
        self.assertEqual((first.state, first.nblocks), ("DONE", 200), first.message)
        # the second series starts when the first one has its wave function, and writes its own trace
        eng.advance()
        self.assertEqual(self.sched.submitted[-1]["script"], self.path(key + "_s2", "job.sh"))
        self.fake_rows(key + "_s2", 200, project="dmc2")
        self.write(key + "_s2", "dmc.done", "1400\n")
        self.sched.state_of[self.sched.submitted[-1]["id"]] = TM.S.COMPLETED
        self.assertEqual(runs.tick(), "DONE")
        # the facts and the checks, as the engine reports them
        row = runs.results()["rows"][0]
        self.assertEqual((row["done"], row["states"], row["series"], row["blocks"]), (True, ["DONE", "DONE"], 2, 400))
        facts = row["trial"]
        for name in ("kind", "cutoff", "orbitals", "label", "n_det", "e_ref", "e_scf", "mo_num", "n_det_parent", "n_det_max",
                     "segments", "cipsi_seconds", "e_var_parent", "pt2_parent", "pt2_error_parent", "e_pt2_parent"):
            self.assertIn(name, facts)
        self.assertEqual((facts["cutoff"], facts["n_det"], facts["n_det_parent"], facts["n_det_max"]), (1e-3, 55, 3280, 20000))
        check = row["checks"]["mdvmc"]
        for name in ("ok", "verdict", "energy", "error", "e_ref", "difference", "limit", "runs"):
            self.assertIn(name, check)
        self.assertEqual((check["ok"], check["verdict"], len(check["runs"])), (True, "PASS", 1))
        self.assertTrue(all(row["checks"][name]["ok"] for name in ("system", "mdc4q", "mdvmc", "jastrow")))
        self.assertGreater(row["cost"], 0.0)
        stat = tables.Reader(eng).stat([key, key + "_s2"], tables.DropRule())
        self.assertAlmostEqual(row["energy"], stat.value, places=12)
        self.assertAlmostEqual(row["error"], stat.error, places=12)
        # the report on these rows: the rule allows a difference, and the cells hold the numbers of Quantum Package
        self.assertIsNone(RT.not_comparable([row, row]))
        data_ = tables.ReportData(project_dir=os.path.join(self.tmp, "project"), title="water", project={})
        table = RT.collect(data_, eng)
        self.assertEqual([(d["value"], d["error"], d["same_geometry"]) for d in table["changes"]], [(0.0, 0.0, True)])
        self.assertEqual(table["barriers"], [])
        RT.write_csv(table, os.path.join(self.tmp, "t.csv"))
        RT.write_md(table, os.path.join(self.tmp, "t.md"))
        with open(os.path.join(self.tmp, "t.csv")) as fh:
            line = list(csv.DictReader(fh))[0]
        self.assertEqual((line["target"], line["kind"], line["geometry"], line["cutoff"], line["n_det_max"], line["n_det_cipsi"],
                          line["n_det"], line["vmc_check_verdict"], line["vmc_check_runs"], line["dmc_states"],
                          line["dmc_series_with_energy"], line["dmc_series_planned"]),
                         ("min", "unknown", "start", "0.001", "20000", "3280", "55", "PASS", "1", "DONE", "2", "2"))
        self.assertAlmostEqual(float(line["e_rhf_qp2_hartree"]), TM.E_SCF, places=9)
        self.assertAlmostEqual(float(line["e_ref_hartree"]), TM.E_CUT, places=9)
        self.assertAlmostEqual(float(line["e_var_cipsi_hartree"]), -17.164589337131737, places=9)
        self.assertAlmostEqual(float(line["pt2_cipsi_error_hartree"]), 7.7816158601717874e-5, places=9)
        self.assertAlmostEqual(float(line["e_var_plus_pt2_cipsi_hartree"]), -17.218069583638240, places=9)
        self.assertLess(abs(float(line["vmc_check_difference_millihartree"])), float(line["vmc_check_limit_millihartree"]))
        self.assertEqual((line["single_determinant_dmc_energy_hartree"], line["difference_kcal_per_mol"]), ("", ""))
        self.assertIn("| min | start | 0.001 | 20000 | 3280 | 55 | 59.6 |", read(os.path.join(self.tmp, "t.md")))


# ===================================================================================== the stage in the workflow
def advance_to_end(wf, max_passes=4000):
    """Advance until the workflow stops. Returns the status and, for every pass, the status of the stages search
    and trial before the pass and the number of calls of engine.advance in the pass."""
    passes, status = [], "RUNNING"
    engine = wf.backend.engine(wf)
    for _ in range(max_passes):
        stages = wf.project.state["stages"]
        before = (stages["search"]["status"], stages.get("trial", {}).get("status"), engine.advances)
        status = wf.advance()
        passes.append((before[0], before[1], engine.advances - before[2]))
        if status != "RUNNING":
            break
    return status, passes


class TestWorkflow(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.folder = tmp.name
        self.input = os.path.join(self.folder, "input.ini")

    def workflow(self, text):
        write(self.input, text)
        return Workflow(self.folder, backend=StandInBackend(Config.load(self.input)))

    def project(self):
        return json.loads(read(os.path.join(self.folder, "work", "project.json")))

    def results(self):
        return sorted(os.path.relpath(os.path.join(d, f), os.path.join(self.folder, "results"))
                      for d, _s, names in os.walk(os.path.join(self.folder, "results")) for f in names)

    DEFAULT_FILES = sorted(
        ["summary.md", "summary.json", "energies.csv", os.path.join("gifs", "geometry_optimization.gif"),
         os.path.join("geometries", "bond_table.csv")]
        + [os.path.join("geometries", "geometry_%s.xyz" % k) for k in ("start", "final", "trajectory")]
        + [os.path.join("plots", "geometry_%s%s" % (k, ext)) for k in ("convergence", "scans", "geometry")
           for ext in (".png", ".pdf")])

    def test_stage_names(self):
        self.assertEqual(STAGES, ("path", "subspace", "directions", "autotune", "search", "report"))
        self.assertEqual(stage_names(), STAGES)
        self.assertEqual(stage_names(False), STAGES)
        self.assertEqual(stage_names(True), ("path", "subspace", "directions", "autotune", "search", "trial", "report"))

    def test_default_project_then_the_option_is_switched_on_changed_and_off(self):
        # ---- 1. no section [trial]: the project of today
        write(self.input, WATER)
        wf = Workflow(self.folder)                                     # the backend of every model project
        self.assertEqual(wf.stages, STAGES)
        for _ in range(4000):
            if wf.advance() != "RUNNING":
                break
        state = self.project()
        self.assertEqual(sorted(state), ["created", "n_images", "name", "stages", "status", "subspace", "symbols", "targets"])
        self.assertEqual(list(state["stages"]), list(STAGES))
        self.assertEqual({s["status"] for s in state["stages"].values()}, {"DONE"})
        self.assertEqual(state["status"], "DONE")
        self.assertEqual(self.results(), self.DEFAULT_FILES)
        summary = json.loads(read(os.path.join(self.folder, "results", "summary.json")))
        self.assertEqual(sorted(summary["files"]), self.DEFAULT_FILES)
        self.assertFalse(os.path.exists(os.path.join(self.folder, "work", "trial")))
        self.assertNotIn("trial", read(os.path.join(self.folder, "work", "driver.log")))
        searched = len(wf.backend.engine(wf).points)

        # ---- 2. the option is switched on later: the stage is added, and the report is written again
        wf = self.workflow(WATER + "[report]\ngif = no\n" + TRIAL % "1e-3")
        self.assertEqual(wf.stages, stage_names(True))
        state = self.project()
        self.assertEqual(list(state["stages"]), ["path", "subspace", "directions", "autotune", "search", "trial", "report"])
        trial_entry = dict(state["stages"]["trial"])
        restore = trial_entry.pop("restore")                           # what the option switched off again would give back
        self.assertEqual((trial_entry, state["stages"]["report"]["status"], state["status"]),
                         ({"status": "PENDING", "message": ""}, "PENDING", "RUNNING"))
        self.assertEqual((restore["report"]["status"], restore["status"]), ("DONE", "DONE"))
        self.assertEqual(state["stages"]["search"]["status"], "DONE")
        status, passes = advance_to_end(wf)
        self.assertEqual(status, "DONE", self.project()["stages"])
        engine = wf.backend.engine(wf)
        self.assertEqual(len(engine.inner.points), searched)           # no point of the search was computed again
        keys = ["trial/geometry/%s/n10000_c1e-03%s" % (g, s) for g in ("start", "final") for s in ("", "_s2")]
        self.assertEqual(sorted(engine.specs), sorted(keys))
        start = G.read_xyz(os.path.join(self.folder, "work", "targets", "geometry", "start.xyz"))[1]
        final = wf.driver("geometry").final_geometry()
        np.testing.assert_allclose(engine.specs[keys[0]].coords, start, atol=1e-12)
        np.testing.assert_allclose(engine.specs[keys[2]].coords, final, atol=1e-12)
        self.assertGreater(float(np.abs(final - start).max()), 1e-4)
        search = wf.point_settings()                                   # what every point of the search got
        self.assertEqual((search["jastrow_reuse"], search["slow_factor"]), ("centre", 3.0))
        for spec in engine.specs.values():
            self.assertEqual(spec.blocks, 400)                         # the length of a final run: Workflow.cap_blocks()
            self.assertEqual(spec.settings["jastrow_reuse"], "none")
            self.assertNotIn("jastrow_start_key", spec.settings)
            self.assertEqual(spec.settings["slow_factor"], search["slow_factor"])    # the engine compares like with like
            self.assertEqual(spec.settings["trial"]["qp2_basis"], "cc-pvtz_ecp_ccecp")
            # everything else is the setting of the search points
            self.assertEqual({k: v for k, v in spec.settings.items() if k not in ("trial", "jastrow_reuse", "slow_factor")},
                             {k: v for k, v in search.items() if k not in ("jastrow_reuse", "slow_factor")})
        self.assertEqual(wf.trial_settings(), dict(search, jastrow_reuse="none"))
        state = self.project()
        self.assertEqual(state["stages"]["trial"], {"status": "DONE", "message": "all 4 DMC run(s) are DONE"})
        self.assertEqual(state["stages"]["report"]["status"], "DONE")
        self.assertTrue(os.path.exists(os.path.join(self.folder, "work", "trial", "state.json")))
        files = self.results()
        self.assertIn("trial_wavefunctions.csv", files)
        self.assertIn("trial_wavefunctions.md", files)
        summary = json.loads(read(os.path.join(self.folder, "results", "summary.json")))
        self.assertIn("trial_wavefunctions.md", summary["files"])
        self.assertEqual(list(summary["project"]["stages"]), list(stage_names(True)))
        log = read(os.path.join(self.folder, "work", "driver.log"))
        for word in ("stage trial: added to the project", "[trial] plan: 4 DMC run(s) of 400 blocks",
                     "stage trial: DONE (all 4 DMC run(s) are DONE)"):
            self.assertIn(word, log)
        # engine.advance() is called in every pass in which the stage runs, and not in its first pass (PENDING)
        self.assertEqual(passes[0], ("DONE", "PENDING", 0))
        running = [p for p in passes if p[1] == "RUNNING"]
        self.assertGreaterEqual(len(running), 3)                       # the stand-in needs 3 calls to finish a point
        self.assertTrue(all(p[2] == 1 for p in running), passes)

        # ---- 3. a new process with the same input: nothing to do
        wf = self.workflow(read(self.input))
        self.assertEqual(self.project()["stages"]["trial"]["status"], "DONE")
        self.assertEqual(wf.advance(), "DONE")
        self.assertEqual(wf.backend.engine(wf).calls, [])              # the stage was not opened again

        # ---- 4. one more cutoff: the stage and the report are PENDING again, and only the new points are computed
        wf = self.workflow(WATER + "[report]\ngif = no\n" + TRIAL % "1e-3, hf")
        state = self.project()
        self.assertEqual(state["stages"]["trial"], {"status": "PENDING", "message": "the values of [trial] changed"})
        self.assertEqual((state["stages"]["report"]["status"], state["status"]), ("PENDING", "RUNNING"))
        engine = wf.backend.engine(wf)
        for key in keys:                                               # the points of the first cutoff exist already
            engine.submit(wf.trial_runs().specs()[wf.trial_runs().keys().index(key)])
            engine.finish(key, mean=-17.21)
        status, passes = advance_to_end(wf)
        self.assertEqual(status, "DONE", self.project()["stages"])
        self.assertEqual(self.project()["stages"]["trial"]["message"], "all 8 DMC run(s) are DONE")
        self.assertEqual(len(engine.specs), 8)
        for key in keys:                                               # the old points were left as they were
            self.assertAlmostEqual(float(engine.points[key]["energy"].mean()), -17.21, delta=1.0e-3)
        with open(os.path.join(self.folder, "results", "trial_wavefunctions.csv")) as fh:
            rows = [r for r in csv.DictReader(fh) if r["row"] == "energy"]
        self.assertEqual([(r["geometry"], r["cutoff"]) for r in rows],
                         [("start", "0.001"), ("start", "hf"), ("final", "0.001"), ("final", "hf")])

        # ---- 5. the option is switched off: the stage leaves the project state, the files stay
        write(self.input, WATER)
        wf = Workflow(self.folder)
        self.assertEqual(list(self.project()["stages"]), list(STAGES))
        self.assertEqual(wf.advance(), "DONE")
        self.assertTrue(os.path.exists(os.path.join(self.folder, "work", "trial", "state.json")))
        self.assertIn("stage trial: taken out of the project state", read(os.path.join(self.folder, "work", "driver.log")))

    def finished_default_project(self):
        """A model project without the option, run to its end. Returns the project state."""
        write(self.input, WATER)
        wf = Workflow(self.folder)
        for _ in range(4000):
            if wf.advance() != "RUNNING":
                break
        state = self.project()
        self.assertEqual(state["status"], "DONE")
        return state

    def test_the_option_switched_on_and_off_before_the_stage_started(self):
        before = self.finished_default_project()
        self.workflow(WATER + TRIAL % "1e-3")
        state = self.project()
        self.assertEqual((state["stages"]["trial"]["status"], state["stages"]["report"]["status"], state["status"]),
                         ("PENDING", "PENDING", "RUNNING"))
        write(self.input, WATER)
        wf = Workflow(self.folder)
        state = self.project()
        self.assertEqual(list(state["stages"]), list(STAGES))
        self.assertEqual(state["stages"]["report"], before["stages"]["report"])
        self.assertEqual(state["status"], "DONE")
        self.assertIn("stage trial: it never started, so the stage report and the project got the old status again",
                      read(os.path.join(self.folder, "work", "driver.log")))
        summary = os.path.join(self.folder, "results", "summary.json")
        written = os.path.getmtime(summary)
        self.assertEqual(wf.advance(), "DONE")
        self.assertEqual(os.path.getmtime(summary), written)          # the report was not written again

    def test_switching_off_gives_back_only_a_status_that_nothing_else_changed(self):
        self.finished_default_project()
        self.workflow(WATER + TRIAL % "1e-3")                          # report PENDING, project RUNNING
        project = Project(self.folder)
        project.set_stage("search", "NEEDS_ATTENTION", "a point failed")      # something else happened meanwhile
        project.state["status"] = "NEEDS_ATTENTION"
        project.save()
        write(self.input, WATER)
        Workflow(self.folder)
        state = self.project()
        self.assertEqual(state["status"], "NEEDS_ATTENTION")           # not set back to DONE
        self.assertEqual(state["stages"]["report"]["status"], "DONE")  # the stage report is what the switch-on changed
        self.assertIn("got the old status again", read(os.path.join(self.folder, "work", "driver.log")))

    def test_retry_opens_no_stage_that_the_not_converged_rule_holds_back(self):
        self.finished_default_project()
        project = Project(self.folder)
        project.target("geometry")["status"] = "NOT_CONVERGED"
        project.save()
        wf = self.workflow(WATER + "[report]\ngif = no\n" + TRIAL % "1e-3")
        self.assertEqual(wf.advance(), "NEEDS_ATTENTION")
        lines = wf.retry()
        self.assertFalse([line for line in lines if "QMC point" in line])
        self.assertFalse(os.path.exists(os.path.join(self.folder, "work", "trial", "state.json")))
        self.assertNotIn("[trial] plan:", read(os.path.join(self.folder, "work", "driver.log")))

    def test_an_error_of_a_project_without_the_option_is_raised_once(self):
        self.finished_default_project()
        write(os.path.join(self.folder, "work", "project.json"), "{ not json")
        with self.assertRaises(ValueError) as ctx:
            with contextlib.redirect_stdout(io.StringIO()):
                cli.main(["run", self.folder, "--once"])
        self.assertIsNone(ctx.exception.__context__)                  # no second error while the first was handled

    def test_the_profile_check_refuses_a_qp2_time_limit_of_15_minutes(self):
        from montepass.machine import Machine
        self.cluster_input(TRIAL % "hf", job=QP2_JOB.replace("time = 01:00:00", "time = 00:15:00"))
        m = Machine(os.path.join(self.folder, "testmachine.ini"))
        self.assertTrue(any("[job.qp2] time = 00:15:00 must be more than 15 minutes" in p for p in m.check_qp2()))
        self.cluster_input(TRIAL % "hf")
        self.assertEqual(Machine(os.path.join(self.folder, "testmachine.ini")).check_qp2(), [])

    def test_switching_the_option_off_stops_the_points_of_the_stage(self):
        wf = self.workflow(WATER + "[report]\ngif = no\n" + TRIAL % "1e-3, hf")
        engine = wf.backend.engine(wf)
        engine.auto = 0                                                # the points of the stage do not end by themselves
        for _ in range(4000):
            if wf.advance() != "RUNNING" or self.project()["stages"]["trial"]["status"] == "RUNNING":
                break
        self.assertEqual(self.project()["stages"]["trial"]["status"], "RUNNING")
        keys = wf.trial_runs().keys()
        self.assertEqual(len(keys), 8)
        engine.finish(keys[0])                                         # an ended point is left alone
        self.assertEqual(wf.advance(), "RUNNING")
        calls = len(engine.calls)
        write(self.input, WATER + "[report]\ngif = no\n")
        backend = StandInBackend(Config.load(self.input))
        backend.stand_in = engine                                      # the same engine, as on the cluster
        wf = Workflow(self.folder, backend=backend)
        self.assertEqual(sorted(engine.stopped), sorted(keys[1:]))
        self.assertEqual({engine.points[k]["state"] for k in keys[1:]}, {STOPPED})
        log = read(os.path.join(self.folder, "work", "driver.log"))
        self.assertEqual(log.count("stopped, because the option [trial] kind = multidet was switched off"), 7)
        self.assertEqual(list(self.project()["stages"]), list(STAGES))
        status, _passes = advance_to_end(wf)
        self.assertEqual(status, "DONE", self.project()["stages"])
        self.assertEqual(len(engine.calls), calls)                     # nothing of the stage was asked for again
        self.assertNotIn("trial_wavefunctions.csv", self.results())   # the report reads no point of the stage
        self.assertTrue(os.path.exists(os.path.join(self.folder, "work", "trial", "state.json")))

    def test_a_search_that_did_not_converge_stops_the_stage_unless_the_input_allows_it(self):
        self.finished_default_project()
        # the search of the target ended without reaching its precision target (the project and the search record say so)
        project = Project(self.folder)
        project.target("geometry")["status"] = "NOT_CONVERGED"
        project.save()
        record = os.path.join(self.folder, "work", "targets", "geometry", "search", "geometry_state.json")
        state = json.loads(read(record))
        state["status"] = "NOT_CONVERGED"
        write(record, json.dumps(state))
        wf = self.workflow(WATER + "[report]\ngif = no\n" + TRIAL % "1e-3")
        status, _passes = advance_to_end(wf)
        self.assertEqual(status, "NEEDS_ATTENTION")
        stage = self.project()["stages"]["trial"]
        self.assertEqual(stage["status"], "NEEDS_ATTENTION")
        for word in ("the search of target(s) geometry ended NOT_CONVERGED", "Set [trial] not_converged = run"):
            self.assertIn(word, stage["message"])
        self.assertEqual(wf.backend.engine(wf).calls, [])              # no point of the stage was asked for
        self.assertFalse(os.path.exists(os.path.join(self.folder, "work", "trial", "state.json")))
        # the input allows it: the stage runs, and the tables mark every value that uses the final geometry
        wf = self.workflow(WATER + "[report]\ngif = no\n" + TRIAL % "1e-3" + "not_converged = run\n")
        status, _passes = advance_to_end(wf)
        self.assertEqual(status, "DONE", self.project()["stages"])
        text = read(os.path.join(self.folder, "results", "trial_wavefunctions.md"))
        self.assertIn("Search status of every target: geometry NOT_CONVERGED.", text)
        self.assertIn("(search NOT_CONVERGED)", text)
        self.assertIn("The search of target(s) geometry ended NOT_CONVERGED", text)

    def test_the_commands_refuse_a_profile_that_cannot_run_the_stage(self):
        self.cluster_input(TRIAL % "1e-3", setup="", job="", tools="")
        for argv in (["submit", self.folder], ["run", self.folder, "--once"], ["report", self.folder], ["retry", self.folder]):
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = cli.main(argv)
            self.assertEqual(code, 1, argv)
            self.assertIn("PROBLEM: section [job.qp2] is missing" if argv[0] == "submit" else "cannot be opened", out.getvalue())
            self.assertIn("section [job.qp2] is missing", out.getvalue())
        self.assertFalse(os.path.exists(os.path.join(self.folder, "driver.sh")))     # no driver job was written or submitted
        # [trial] together with [model]: submit refuses it too
        write(self.input, WATER + TRIAL % "hf")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(cli.main(["submit", self.folder]), 1)
        self.assertIn("PROBLEM: " + MODEL_WITHOUT_QP2, out.getvalue())
        # a project without the option: an error that is not of its input file is raised as before
        self.cluster_input("", setup="", job="", tools="")
        write(os.path.join(self.folder, "testmachine.ini"), "[machine]\nscheduler = nothing-like-this\n")
        with self.assertRaises(Exception) as ctx:
            with contextlib.redirect_stdout(io.StringIO()):
                cli.main(["run", self.folder, "--once"])
        self.assertNotIsInstance(ctx.exception, SystemExit)

    def test_multidet_from_the_start_with_a_failed_point_and_retry(self):
        wf = self.workflow(WATER + "[report]\ngif = no\n" + TRIAL % "1e-3, hf")
        self.assertEqual(list(self.project()["stages"]), list(stage_names(True)))
        engine = wf.backend.engine(wf)
        bad = "trial/geometry/final/hf"
        engine.fail[bad] = "the Quantum Package calculation failed: step qpscf: RUN: exit code 1; attempt 2 of 2"
        status, passes = advance_to_end(wf)
        self.assertEqual(status, "NEEDS_ATTENTION")
        state = self.project()
        self.assertEqual(state["stages"]["search"]["status"], "DONE")
        self.assertEqual(state["stages"]["report"]["status"], "PENDING")          # no report before the stage is done
        self.assertEqual(state["stages"]["trial"]["status"], "NEEDS_ATTENTION")
        message = state["stages"]["trial"]["message"]
        self.assertIn("%s: the Quantum Package calculation failed: step qpscf" % bad, message)
        self.assertNotIn("trial/geometry/start", message)              # only the failed point is named
        # the search ran before the stage, and the stage did not start before the search was DONE
        self.assertTrue(all(p[1] == "PENDING" for p in passes if p[0] != "DONE"))
        self.assertTrue(all(p[2] == 1 for p in passes if p[1] in ("RUNNING", "NEEDS_ATTENTION")))
        self.assertEqual(wf.advance(), "NEEDS_ATTENTION")              # it stays, and the other points are still read
        # after the repair: retry gives the failed point new attempts and clears the status of the stage
        wf = self.workflow(read(self.input))                           # a new process
        engine = wf.backend.engine(wf)
        self.assertEqual(engine.points, {})                            # the stand-in keeps nothing on disk:
        for spec in wf.trial_runs().specs():                           # put its points back as the first process left them
            engine.submit(spec)
            if spec.key == bad:
                engine.points[bad].update(state=FAILED, message="failed")
            elif spec.key != bad + "_s2":
                engine.finish(spec.key)
        lines = wf.retry()
        self.assertIn("QMC point %s: submitted again" % bad, lines)
        self.assertIn("stage trial: the NEEDS_ATTENTION status was cleared", lines)
        self.assertEqual(engine.retried, [bad])
        self.assertEqual(self.project()["stages"]["trial"]["status"], "RUNNING")
        status, passes = advance_to_end(wf)
        self.assertEqual(status, "DONE", self.project()["stages"])
        self.assertEqual({s["status"] for s in self.project()["stages"].values()}, {"DONE"})
        self.assertIn("trial_wavefunctions.csv", self.results())

    def test_an_error_of_the_stage_is_visible_in_the_project_state(self):
        text = WATER + "[report]\ngif = no\n" + TRIAL % "hf"
        wf = self.workflow(text)
        status, _passes = advance_to_end(wf)
        self.assertEqual(status, "DONE")
        # 1. the record of the stage cannot be read
        record = os.path.join(self.folder, "work", "trial", "state.json")
        write(record, "{ not json")
        wf = self.workflow(text)                                       # this must not stop the program at its start
        self.assertEqual(self.project()["stages"]["trial"]["status"], "PENDING")
        self.assertEqual(wf.advance(), "NEEDS_ATTENTION")
        stage = self.project()["stages"]["trial"]
        self.assertIn("state.json could not be read", stage["message"])
        self.assertIn("Delete the file", stage["message"])
        os.remove(record)                                              # what the message says
        status, _passes = advance_to_end(wf)
        self.assertEqual(status, "DONE", self.project()["stages"])
        self.assertEqual(json.loads(read(record))["status"], "DONE")
        # 2. the record was written with one basis file, the input now names another one
        wf = self.workflow(text + "qp2_basis = cc-pvqz_ecp_ccecp\n")
        self.assertEqual(self.project()["stages"]["trial"]["status"], "PENDING")
        self.assertEqual(wf.advance(), "NEEDS_ATTENTION")
        stage = self.project()["stages"]["trial"]
        self.assertEqual(stage["status"], "NEEDS_ATTENTION")
        self.assertIn("ValueError", stage["message"])
        self.assertIn("was written with [trial] qp2_basis = cc-pvtz_ecp_ccecp", stage["message"])
        self.assertIn("the stage trial could not be opened", " ".join(wf.retry()))

    def test_the_model_engine_cannot_run_quantum_package(self):
        write(self.input, WATER + TRIAL % "hf")
        cfg = Config.load(self.input)                                  # the input itself is valid
        self.assertIsNotNone(cfg.trial)
        with self.assertRaises(ValueError) as ctx:
            make_backend(cfg)
        self.assertIn(MODEL_WITHOUT_QP2, str(ctx.exception))
        self.assertIn("[trial] kind = multidet cannot be used together with a section [model]", str(ctx.exception))
        with self.assertRaises(ValueError):
            Workflow(self.folder)
        self.assertIsInstance(make_backend(load(WATER)), ModelBackend)           # without the option nothing changes
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(cli.main(["check", self.folder]), 1)
        self.assertIn("PROBLEM: " + MODEL_WITHOUT_QP2, out.getvalue())

    def cluster_input(self, trial, ranks=32, setup=QP2_SETUP, job=QP2_JOB, tools=QP2_TOOLS):
        """A project that needs a machine profile (no section [model]), with the profile next to the input file."""
        os.makedirs(os.path.join(self.folder, "pp"), exist_ok=True)
        G.write_xyz(os.path.join(self.folder, "geometry.xyz"), ["O", "H", "H"], M.water_geometry())
        write(os.path.join(self.folder, "testmachine.ini"), QP2_PROFILE % {
            "pp": os.path.join(self.folder, "pp"), "ranks": ranks, "setup": setup, "job": job, "tools": tools})
        write(self.input, "[geometry]\nmode = single\ngeometry = geometry.xyz\n[surrogate]\nprogram = none\n"
                          "[machine]\nprofile = testmachine\n" + trial)
        return Config.load(self.input)

    def check(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = cli.main(["check", self.folder])
        return code, out.getvalue()

    def test_cluster_backend_and_check_need_a_profile_for_quantum_package(self):
        from montepass.cluster import ClusterBackend

        # a complete profile
        cfg = self.cluster_input(TRIAL % "1e-3, hf")
        self.assertIsInstance(make_backend(cfg), ClusterBackend)
        code, text = self.check()
        self.assertEqual(code, 0, text)
        self.assertNotIn("PROBLEM", text)
        for word in ("stage trial (after the geometry search)", "geometries: up to 2", "Quantum Package parent calculations: up to 2",
                     "cutoffs: 2 (0.001, hf)", "DMC series: 2 for each geometry and each cutoff, up to 8 DMC run(s) in total"):
            self.assertIn(word, text)
        # a profile without the sections and the key for Quantum Package
        cfg = self.cluster_input(TRIAL % "1e-3", setup="", job="", tools="")
        with self.assertRaises(ValueError) as ctx:
            make_backend(cfg)
        message = str(ctx.exception)
        for word in ("[trial] kind = multidet", "testmachine.ini", "section [setup.qp2] has no lines",
                     "section [job.qp2] is missing", "[executables] determinants_tools is not set"):
            self.assertIn(word, message)
        code, text = self.check()
        self.assertEqual(code, 1)
        for word in ("PROBLEM: section [setup.qp2] has no lines", "PROBLEM: section [job.qp2] is missing",
                     "PROBLEM: [executables] determinants_tools is not set", "stage trial (after the geometry search)"):
            self.assertIn(word, text)
        # more series than the random numbers of QMCPACK allow on 256 MPI ranks
        cfg = self.cluster_input((TRIAL % "hf").replace("series = 2", "series = 3"), ranks=256)
        with self.assertRaises(ValueError) as ctx:
            make_backend(cfg)
        self.assertIn("[trial] series = 3", str(ctx.exception))
        # without the option the same profiles are accepted, and "check" prints nothing about the stage
        cfg = self.cluster_input("", setup="", job="", tools="")
        self.assertIsNone(cfg.trial)
        self.assertIsInstance(make_backend(cfg), ClusterBackend)
        code, text = self.check()
        self.assertEqual(code, 0, text)
        self.assertNotIn("stage trial", text)
        self.assertNotIn("Quantum Package", text)

    def test_the_example_profile_has_the_sections_for_quantum_package(self):
        from montepass.machine import Machine

        m = Machine(os.path.join(os.path.dirname(HERE), "machines", "example_slurm.ini"))
        self.assertEqual(m.check_qp2(), [])
        self.assertEqual(T.machine_problems(m, TRIAL_DICT), [])
        lines = m.setup_lines("qp2")
        self.assertEqual(lines[0], "set +u")                           # the rc file reads variables that are not set
        self.assertTrue(lines[-1].startswith("source ") and lines[-1].endswith("quantum_package.rc"), lines)
        self.assertEqual((m.tasks("qp2"), m.threads("qp2"), m.job("qp2")["mem"], m.job("qp2")["time"]), (1, 16, "64G", "12:00:00"))
        self.assertEqual(m.job("qmc")["time_multidet"], "12:00:00")
        self.assertTrue(m.executable("determinants_tools").endswith("utils/determinants_tools.py"))
        self.assertTrue(m.cp.get("executables", "qp2_python_bin"))
        header = m.header("qp2", "qp2_x", "/work/q")
        for needle in ("#SBATCH --ntasks=1", "#SBATCH --cpus-per-task=16", "#SBATCH --mem=64G", "#SBATCH --time=12:00:00",
                       "set +u || {", "quantum_package.rc || {", "export OMP_NUM_THREADS=1"):
            self.assertIn(needle, header)
        # what the jobs of a project without the option read from this profile is what it was
        self.assertEqual((m.tasks("qmc"), m.threads("scf"), m.job("qmc")["time"], m.job("qmc")["mem"]), (32, 4, "04:00:00", "32G"))
        self.assertEqual((m.job("orca")["tasks"], m.job("orca")["time"], m.job("driver")["time"]), ("8", "04:00:00", "48:00:00"))
        self.assertEqual(m.setup_lines("orca"), ["module load openmpi/4.1.8"])
        header = m.header("qmc", "test", "/work/x")
        self.assertIn("#SBATCH --time=04:00:00", header)
        self.assertNotIn("12:00:00", header)
        self.assertNotIn("quantum_package", header)


# ===================================================================================== the report
class TestReport(unittest.TestCase):
    """The demonstration project (ammonia inversion: a minimum ``reactant`` and a ``saddle``) with a stage record."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.base = os.path.join(cls.tmp, "demo")
        cls.demo = make_demo_project(cls.base)
        data = tables.collect(cls.base, cls.demo.engine)
        cls.geometries = {name: (t.start, t.final) for name, t in data.targets.items()}
        cls.symbols = cls.demo.symbols
        cls.plain_files = build_report(cls.base, cls.demo.engine, gifs=False)         # a project without the stage

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    ENERGY = {("reactant", "start"): -11.7300, ("reactant", "final"): -11.7310, ("saddle", "start"): -11.7200,
              ("saddle", "final"): -11.7205}
    HF_SHIFT = 0.0400                   # the single RHF determinant is this much higher at every geometry

    def stage(self, name, trial=None, geometries=None, finish=True):
        """A copy of the project with the record of the stage and an engine that holds its points."""
        folder = os.path.join(self.tmp, name)
        shutil.copytree(self.base, folder)
        Project(folder).add_stage("trial", before="report")             # the state of a project whose input asks for the stage
        engine = StandIn(self.demo.engine)
        runs = T.TrialRuns(engine, self.symbols, geometries or self.geometries, trial or TRIAL_DICT, SETTINGS, 200,
                           os.path.join(folder, "work", "trial"))
        runs.tick()
        if finish:
            for key in runs.keys():
                target, geometry, label = key.split("/")[1:4]
                shift = self.HF_SHIFT if label.startswith("hf") else 0.0
                engine.finish(key, mean=self.ENERGY[target, geometry] + shift, noise=4.0e-4)
            self.assertEqual(runs.tick(), "DONE")
        return folder, engine, runs

    def table(self, folder, engine):
        return RT.collect(tables.collect(folder, engine), engine)

    def csv_rows(self, folder):
        with open(os.path.join(folder, "results", "trial_wavefunctions.csv")) as fh:
            return list(csv.DictReader(fh))

    def test_without_the_record_the_report_is_the_report_of_today(self):
        expected = ["summary.md", "summary.json", "energies.csv", os.path.join("geometries", "bond_table.csv"),
                    os.path.join("geometries", "path_internal_coordinates.csv")]
        for stem in ("pca_variance", "path_profile", "path_geometry"):
            expected += [os.path.join("plots", stem + ext) for ext in (".png", ".pdf")]
        for target in ("reactant", "saddle"):
            expected += [os.path.join("geometries", "%s_%s.xyz" % (target, k)) for k in ("start", "final", "trajectory")]
            expected += [os.path.join("plots", "%s_%s%s" % (target, k, ext)) for k in ("convergence", "scans", "geometry")
                         for ext in (".png", ".pdf")]
        results = os.path.join(self.base, "results")
        self.assertEqual(sorted(os.path.relpath(f, results) for f in self.plain_files), sorted(expected))
        self.assertFalse(RT.available(self.base))
        summary = json.loads(read(os.path.join(results, "summary.json")))
        self.assertEqual(sorted(summary["files"]), sorted(expected))
        self.assertEqual(sorted(x["item"] for x in summary["skipped"]), ["gifs", "plots/path_geometry_map"])
        self.assertFalse(any("trial" in f for f in os.listdir(results)))

    def test_files_and_numbers(self):
        folder, engine, runs = self.stage("full")
        self.assertTrue(RT.available(folder))
        files = build_report(folder, engine, gifs=False)
        results = os.path.join(folder, "results")
        rel = sorted(os.path.relpath(f, results) for f in files)
        plain = sorted(os.path.relpath(f, os.path.join(self.base, "results")) for f in self.plain_files)
        self.assertEqual(rel, sorted(plain + ["trial_wavefunctions.csv", "trial_wavefunctions.md"]))     # 2 files more
        summary = json.loads(read(os.path.join(results, "summary.json")))
        self.assertIn("trial_wavefunctions.csv", summary["files"])
        self.assertEqual(sorted(x["item"] for x in summary["skipped"]), ["gifs", "plots/path_geometry_map"])
        rows = self.csv_rows(folder)
        self.assertEqual(list(rows[0]), RT.CSV_HEADER)
        energy = [r for r in rows if r["row"] == "energy"]
        self.assertEqual([(r["target"], r["geometry"], r["cutoff"]) for r in energy],
                         [(t, g, c) for t in ("reactant", "saddle") for g in ("start", "final") for c in ("0.001", "hf")])
        res = {(r["target"], r["geometry"], r["label"]): r for r in runs.results()["rows"]}
        search = summary["targets"]
        for r in energy:
            hf = r["cutoff"] == "hf"
            mine = res[r["target"], r["geometry"], "hf" if hf else "n10000_c1e-03"]
            self.assertEqual(r["kind"], "saddle" if r["target"] == "saddle" else "minimum")
            self.assertEqual((r["n_det_max"], r["n_det"], r["n_det_cipsi"], r["n_det_cipsi_over_n_det"]),
                             ("", "1", "not computed", "") if hf else ("10000", "55", "3280", "59.64"))
            self.assertAlmostEqual(float(r["e_rhf_qp2_hartree"]), E_SCF, places=9)
            self.assertAlmostEqual(float(r["e_ref_hartree"]), E_SCF if hf else E_CUT, places=9)
            if hf:
                self.assertEqual({r[c] for c in ("e_var_cipsi_hartree", "pt2_cipsi_hartree", "pt2_cipsi_error_hartree",
                                                 "e_var_plus_pt2_cipsi_hartree")}, {"not computed"})
            else:
                self.assertAlmostEqual(float(r["e_var_cipsi_hartree"]), -17.164589, places=9)
                self.assertAlmostEqual(float(r["pt2_cipsi_hartree"]), -0.053480, places=9)
                self.assertAlmostEqual(float(r["pt2_cipsi_error_hartree"]), 7.8e-5, places=9)
                self.assertAlmostEqual(float(r["e_var_plus_pt2_cipsi_hartree"]), -17.218069, places=9)
            self.assertAlmostEqual(float(r["vmc_check_energy_hartree"]), (E_SCF if hf else E_CUT) + 2.0e-4, places=9)
            self.assertAlmostEqual(float(r["vmc_check_error_hartree"]), 3.0e-4, places=9)
            self.assertEqual((r["vmc_check_difference_millihartree"], r["vmc_check_limit_millihartree"],
                              r["vmc_check_verdict"], r["vmc_check_runs"]), ("0.2000", "1.0000", "PASS", "2"))
            self.assertEqual((r["dmc_states"], r["dmc_series_with_energy"], r["dmc_series_planned"], r["dmc_blocks"],
                              r["core_hours"]), ("DONE", "2", "2", "400", "3.0000"))
            self.assertAlmostEqual(float(r["dmc_energy_hartree"]), mine["energy"], places=9)
            self.assertAlmostEqual(float(r["dmc_error_hartree"]), mine["error"], places=9)
            self.assertLess(abs(mine["energy"] - self.ENERGY[r["target"], r["geometry"]] - (self.HF_SHIFT if hf else 0.0)),
                            5.0 * mine["error"])
            # its own column: the single-determinant energy of the search at the same geometry
            single = search[r["target"]]["start_energy_hartree" if r["geometry"] == "start" else "final_energy_hartree"]
            self.assertAlmostEqual(float(r["single_determinant_dmc_energy_hartree"]), single["value"], places=9)
            self.assertAlmostEqual(float(r["single_determinant_dmc_error_hartree"]), single["error"], places=9)
            self.assertEqual((r["difference_kcal_per_mol"], r["difference_error_kcal_per_mol"]), ("", ""))
            # the files of Quantum Package, its version, and the check that both programs use the same basis set
            self.assertEqual((r["qp2_basis"], r["qp2_pseudo"], r["qp2_version"]), ("cc-pvtz_ecp_ccecp", "ccecp", "b65c9cf9"))
            self.assertAlmostEqual(float(r["rhf_qp2_minus_pyscf_millihartree"]), 5.0e-6, places=9)
            self.assertEqual((r["dmc_messages"], r["search_status"]), ("", search[r["target"]]["status"]))
        # the differences: final minus start for every target and cutoff, then the barrier for every cutoff
        changes = [r for r in rows if r["row"] == "final minus start"]
        self.assertEqual([(r["target"], r["cutoff"]) for r in changes],
                         [("reactant", "0.001"), ("reactant", "hf"), ("saddle", "0.001"), ("saddle", "hf")])
        for r in changes:
            label = "hf" if r["cutoff"] == "hf" else "n10000_c1e-03"
            a, b = res[r["target"], "final", label], res[r["target"], "start", label]
            self.assertAlmostEqual(float(r["difference_kcal_per_mol"]), (a["energy"] - b["energy"]) * H, places=5)
            self.assertAlmostEqual(float(r["difference_error_kcal_per_mol"]), float(np.hypot(a["error"], b["error"])) * H, places=5)
        barriers = [r for r in rows if r["row"] == "barrier"]
        self.assertEqual([(r["target"], r["minus_target"], r["geometry"], r["cutoff"]) for r in barriers],
                         [("saddle", "reactant", g, c) for g in ("start", "final") for c in ("0.001", "hf")])
        for r in barriers:
            label = "hf" if r["cutoff"] == "hf" else "n10000_c1e-03"
            a, b = res["saddle", r["geometry"], label], res["reactant", r["geometry"], label]
            self.assertAlmostEqual(float(r["difference_kcal_per_mol"]), (a["energy"] - b["energy"]) * H, places=5)
            self.assertAlmostEqual(float(r["difference_error_kcal_per_mol"]), float(np.hypot(a["error"], b["error"])) * H, places=5)
            expected = (self.ENERGY["saddle", r["geometry"]] - self.ENERGY["reactant", r["geometry"]]) * H
            self.assertLess(abs(float(r["difference_kcal_per_mol"]) - expected), 5.0 * float(r["difference_error_kcal_per_mol"]))
            self.assertEqual((r["search_status"], r["minus_target_search_status"]),
                             (search["saddle"]["status"], search["reactant"]["status"]))
        # the change of the barrier: four independent rows
        changes_of_barrier = [r for r in rows if r["row"] == "barrier change"]
        self.assertEqual([(r["target"], r["minus_target"], r["geometry"], r["cutoff"]) for r in changes_of_barrier],
                         [("saddle", "reactant", "final minus start", c) for c in ("0.001", "hf")])
        for r in changes_of_barrier:
            label = "hf" if r["cutoff"] == "hf" else "n10000_c1e-03"
            sf, mf, ss, ms = (res[name, g, label] for name, g in (("saddle", "final"), ("reactant", "final"),
                                                                    ("saddle", "start"), ("reactant", "start")))
            self.assertAlmostEqual(float(r["difference_kcal_per_mol"]),
                                   (sf["energy"] - mf["energy"] - ss["energy"] + ms["energy"]) * H, places=5)
            self.assertAlmostEqual(float(r["difference_error_kcal_per_mol"]),
                                   float(np.sqrt(sum(x["error"] ** 2 for x in (sf, mf, ss, ms)))) * H, places=5)
        # and one line "against hf" for each of the 4 geometries: the cut minus the row hf of this stage
        self.assertEqual(len(rows), 8 + 4 + 4 + 2 + 4)
        self.assertEqual({r["uses_not_converged_geometry"] for r in rows}, {""})
        # no line of the file subtracts the single-determinant energy of the search from a multi-determinant energy
        self.assertEqual({r["row"] for r in rows}, {"energy", "final minus start", "barrier", "barrier change", "against hf"})
        self.assertEqual({r["single_determinant_dmc_energy_hartree"] for r in rows if r["row"] != "energy"}, {""})
        text = read(os.path.join(results, "trial_wavefunctions.md"))
        for needle in (
                "# model_ammonia_inversion: DMC energies with several determinants", "1 standard error",
                RT.ORBITALS, RT.NO_SUBTRACTION, "same basis set",
                "## Trial wave functions", "## Checks of the trial wave function", "## DMC energies",
                "## Energy change of every target: final geometry minus start geometry",
                "## Barriers: saddle minus minimum", "barrier at the start geometries (kcal/mol)",
                "Quantum Package 2 (QP2) of these rows: basis file cc-pvtz_ecp_ccecp; pseudopotential file ccecp; "
                "QP2 version (git commit) b65c9cf9.", "| +0.000005 |",
                "| reactant | final | 0.001 | 10000 | 3280 | 55 | 59.6 |", "| saddle | start | hf | none | not computed | 1 | n/a |",
                "| +0.200 | 1.000 | PASS | 2 |", "n_det_max = 10000; cutoff = 0.001, hf; 2 DMC series of 200 blocks",
                "this report prints no such difference"):
            self.assertIn(needle, text)
        self.assertEqual(text.count(RT.ORBITALS), 1)
        self.assertNotIn("another basis", text)
        self.assertNotIn("its basis file", text)
        self.assertNotIn(RT.MARK.strip(), text)                          # no search ended NOT_CONVERGED
        self.assertNotIn(RT.ONE_CUTOFF, text)                           # this project has two cutoffs
        self.assertNotIn("NOT COMPARABLE: ", text)
        self.assertIsNone(re.search(r"\b(nan|inf|None)\b", text))      # every cell holds a number or plain words
        self.assertEqual([f for f in os.listdir(results) if f.endswith(".tmp")], [])

    def test_one_cutoff_gets_the_sentence_about_convergence(self):
        folder, engine, _runs = self.stage("one", dict(TRIAL_DICT, cutoffs=[1e-3]))
        table = self.table(folder, engine)
        self.assertTrue(table["one_cutoff"])
        path = os.path.join(folder, "one.md")
        RT.write_md(table, path)
        text = read(path)
        self.assertEqual(text.count("One cutoff cannot show whether the energy difference is converged with the size of "
                                    "the expansion."), 1)
        self.assertEqual(text.count(RT.ORBITALS), 1)
        self.assertEqual(len(table["changes"]), 2)
        self.assertEqual(len(table["barriers"]), 1)
        # the fixed sentence names B3LYP, the default of [qmc] xc: a project with another functional gets one more sentence
        self.assertIsNone(table["xc"])                                 # the settings of the demonstration project name none
        self.assertNotIn("[qmc] xc", text)
        for xc, expected in (("b3lyp", 0), ("B3LYP", 0), ("pbe0", 1)):
            RT.write_md(dict(table, xc=xc), path)
            self.assertEqual(read(path).count("This project set [qmc] xc = %s" % xc), expected)

    def test_a_geometry_that_did_not_move(self):
        x = self.geometries["saddle"][0]
        geometries = dict(self.geometries, saddle=(x, x.copy()))
        folder, engine, _runs = self.stage("same", geometries=geometries)
        table = self.table(folder, engine)
        self.assertEqual([(r["target"], r["geometry"]) for r in table["rows"] if r["label"] == "hf"],
                         [("reactant", "start"), ("reactant", "final"), ("saddle", "start")])
        change = [d for d in table["changes"] if d["target"] == "saddle"]
        self.assertEqual([(d["value"], d["error"], d["text"], d["same_geometry"]) for d in change], [(0.0, 0.0, None, True)] * 2)
        # the barrier uses the start geometry of the saddle: it is its final geometry
        res = {(r["target"], r["geometry"], r["label"]): r for r in table["rows"]}
        barrier = table["barriers"][0]
        self.assertAlmostEqual(barrier["value"], (res["saddle", "start", "n10000_c1e-03"]["energy"]
                                                  - res["reactant", "final", "n10000_c1e-03"]["energy"]) * H, places=9)
        path = os.path.join(folder, "same.md")
        RT.write_md(table, path)
        text = read(path)
        self.assertIn("- Target saddle: the final geometry equals the start geometry within 1e-06 angstrom: only the start "
                      "geometry is computed.", text)
        self.assertIn("0.000000 (exact) (the final geometry is the start geometry)", text)

    def test_a_geometry_that_did_not_move_but_whose_run_failed(self):
        x = self.geometries["saddle"][0]
        folder, engine, _runs = self.stage("same_failed", geometries=dict(self.geometries, saddle=(x, x.copy())))
        engine.points["trial/saddle/start/n10000_c1e-03_s2"].update(state=FAILED, message="job 9 ended in state FAILED")
        table = self.table(folder, engine)
        change = [d for d in table["changes"] if (d["target"], d["label"]) == ("saddle", "n10000_c1e-03")][0]
        self.assertEqual((change["value"], change["error"]), (None, None))
        self.assertEqual(change["text"], "NOT COMPARABLE: saddle start: 1 of 2 series are DONE (DONE, FAILED)")
        other = [d for d in table["changes"] if (d["target"], d["label"]) == ("saddle", "hf")][0]
        self.assertEqual((other["value"], other["error"], other["text"]), (0.0, 0.0, None))     # complete: 0 and exact
        # the change of the barrier: the saddle rows cancel, but they must be complete too
        change = [d for d in table["barrier_changes"] if d["label"] == "n10000_c1e-03"][0]
        self.assertTrue(change["text"].startswith("NOT COMPARABLE: "))
        res = {(r["target"], r["geometry"], r["label"]): r for r in table["rows"]}
        hf = [d for d in table["barrier_changes"] if d["label"] == "hf"][0]
        mf, ms = res["reactant", "final", "hf"], res["reactant", "start", "hf"]
        self.assertAlmostEqual(hf["value"], -(mf["energy"] - ms["energy"]) * H, places=9)
        self.assertAlmostEqual(hf["error"], float(np.hypot(mf["error"], ms["error"])) * H, places=9)
        # the message of the failed run is in both files
        RT.write_csv(table, os.path.join(folder, "t.csv"))
        RT.write_md(table, os.path.join(folder, "t.md"))
        with open(os.path.join(folder, "t.csv")) as fh:
            row = [r for r in csv.DictReader(fh) if r["row"] == "energy"
                   and (r["target"], r["geometry"], r["cutoff"]) == ("saddle", "start", "0.001")][0]
        self.assertEqual(row["dmc_messages"], "trial/saddle/start/n10000_c1e-03_s2: job 9 ended in state FAILED")
        self.assertIn("trial/saddle/start/n10000_c1e-03_s2: job 9 ended in state FAILED", read(os.path.join(folder, "t.md")))

    def test_values_that_use_a_search_that_did_not_converge_are_marked(self):
        folder, engine, _runs = self.stage("not_converged")
        data = tables.collect(folder, engine)
        data.targets["saddle"].status = "NOT_CONVERGED"
        table = RT.collect(data, engine)
        self.assertEqual(table["statuses"]["saddle"], "NOT_CONVERGED")
        self.assertEqual({d["marked"] for d in table["changes"] if d["target"] == "saddle"}, {True})
        self.assertEqual({d["marked"] for d in table["changes"] if d["target"] == "reactant"}, {False})
        self.assertEqual({d["marked"] for d in table["barriers"] + table["barrier_changes"]}, {True})
        self.assertEqual({d["marked"] for d in table["start_barriers"]}, {False})        # the start geometries do not depend on it
        RT.write_csv(table, os.path.join(folder, "t.csv"))
        RT.write_md(table, os.path.join(folder, "t.md"))
        with open(os.path.join(folder, "t.csv")) as fh:
            rows = list(csv.DictReader(fh))
        marked = {(r["row"], r["target"], r["geometry"]) for r in rows if r["uses_not_converged_geometry"] == "yes"}
        self.assertEqual(marked, {("energy", "saddle", "final"), ("final minus start", "saddle", "final minus start"),
                                  ("barrier", "saddle", "final"), ("barrier change", "saddle", "final minus start"),
                                  ("against hf", "saddle", "final")})
        text = read(os.path.join(folder, "t.md"))
        self.assertIn("Search status of every target: reactant ", text)
        self.assertIn("saddle NOT_CONVERGED", text)
        self.assertEqual(text.count("The search of target(s) saddle ended NOT_CONVERGED"), 2)     # under the 2 tables
        # the geometry cell of the 2 final rows of the saddle in 3 tables, 2 cells of changes, 2 x 2 barrier cells
        # (final geometries and change), 1 in each of the 2 sentences, and the geometry cell of the cut in the table
        # "DMC energy against the single RHF determinant"
        self.assertEqual(text.count(RT.MARK.strip()), 6 + 2 + 4 + 2 + 1)
        self.assertIn("| saddle | final (search NOT_CONVERGED) | 0.001 |", text)

    def test_a_project_whose_option_was_switched_off_gets_no_trial_files(self):
        folder, engine, _runs = self.stage("off")
        project = Project(folder)
        project.remove_stage("trial")                                  # what the workflow does when [trial] is removed
        reads = []
        original = engine.read

        def counting(key):
            reads.append(key)
            return original(key)
        engine.read = counting
        files = build_report(folder, engine, gifs=False)
        self.assertFalse(any("trial_wavefunctions" in f for f in files))
        self.assertFalse([k for k in reads if k.startswith("trial/")])  # no point of the stage was read
        self.assertTrue(RT.available(folder))                          # the record is still on disk

    def test_every_case_that_is_not_comparable(self):
        folder, engine, runs = self.stage("cases")
        final, start = "trial/reactant/final/n10000_c1e-03", "trial/reactant/start/n10000_c1e-03"
        good = {key: dict(p, checks=json.loads(json.dumps(p["checks"])), facts=dict(p["facts"])) for key, p in engine.points.items()}

        def cells(change):
            """Apply a change to the points, and return the two cells that use the changed geometry and one that does not."""
            engine.points = {key: dict(p, checks=json.loads(json.dumps(p["checks"])), facts=dict(p["facts"]))
                             for key, p in good.items()}
            change(engine.points)
            table = self.table(folder, engine)
            first = [d for d in table["changes"] if (d["target"], d["label"]) == ("reactant", "n10000_c1e-03")][0]
            other = [d for d in table["changes"] if (d["target"], d["label"]) == ("reactant", "hf")][0]
            barrier = [d for d in table["barriers"] if d["label"] == "n10000_c1e-03"][0]
            self.assertIsNone(other["text"])                           # the other cutoff is not touched
            self.assertIsNotNone(other["value"])
            return table, first, barrier

        _table, first, barrier = cells(lambda points: None)
        self.assertEqual((first["text"], barrier["text"]), (None, None))
        self.assertLess(abs(first["value"] - (-11.7310 + 11.7300) * H), 5.0 * first["error"])

        # 1. one series still runs
        table, first, barrier = cells(lambda points: points[final + "_s2"].update(state=RUNNING, message="90 of 200 blocks"))
        self.assertEqual(first["text"], "NOT COMPARABLE: reactant final: 1 of 2 series are DONE (DONE, RUNNING)")
        self.assertEqual((first["value"], first["error"]), (None, None))
        self.assertEqual(barrier["text"], first["text"])               # the barrier uses the same geometry
        # the files show the reason in the cell of the difference, and still show the energy of the row itself
        RT.write_csv(table, os.path.join(folder, "t.csv"))
        RT.write_md(table, os.path.join(folder, "t.md"))
        with open(os.path.join(folder, "t.csv")) as fh:
            rows = list(csv.DictReader(fh))
        bad = [r for r in rows if r["row"] == "final minus start" and (r["target"], r["cutoff"]) == ("reactant", "0.001")][0]
        self.assertEqual((bad["difference_kcal_per_mol"], bad["difference_error_kcal_per_mol"]), (first["text"], ""))
        row = [r for r in rows if r["row"] == "energy"
               and (r["target"], r["geometry"], r["cutoff"]) == ("reactant", "final", "0.001")][0]
        self.assertEqual(row["dmc_states"], "DONE, RUNNING")
        self.assertNotEqual(row["dmc_energy_hartree"], "")
        text = read(os.path.join(folder, "t.md"))
        # the cell of the change, the barrier at the final geometries, and the change of the barrier
        self.assertEqual(text.count("NOT COMPARABLE: reactant final: 1 of 2 series are DONE (DONE, RUNNING)"), 3)

        # 2. a point failed
        _t, first, _b = cells(lambda points: points[start].update(state=FAILED, message="job 5 ended in state FAILED"))
        self.assertEqual(first["text"], "NOT COMPARABLE: reactant start: 1 of 2 series are DONE (FAILED, DONE)")

        # 3. the VMC check did not pass
        def failed_check(points):
            points[start]["checks"]["mdvmc"].update(ok=False, verdict="FAIL")
        _t, first, barrier = cells(failed_check)
        self.assertEqual(first["text"], "NOT COMPARABLE: reactant start: the VMC check without Jastrow factor did not pass "
                                        "(verdict FAIL)")
        self.assertIsNone(barrier["text"])                             # the barrier does not use the start geometry

        # 4. the VMC check was not made
        _t, first, _b = cells(lambda points: points[final]["checks"].pop("mdvmc"))
        self.assertEqual(first["text"], "NOT COMPARABLE: reactant final: the VMC check without Jastrow factor was not made")

        # 5. the two geometries were computed with another size of the CIPSI wave function
        _t, first, _b = cells(lambda points: points[start]["facts"].update(n_det_max=5000))
        self.assertEqual(first["text"], "NOT COMPARABLE: n_det_max differs (10000 at reactant final, 5000 at reactant start)")

        # 6. ... or with another cutoff
        _t, first, _b = cells(lambda points: points[final]["facts"].update(cutoff=5e-4))
        self.assertEqual(first["text"], "NOT COMPARABLE: cutoff differs (0.0005 at reactant final, 0.001 at reactant start)")

        # 7. no energy: the runs are DONE but hold too few blocks
        def short(points):
            for key in (final, final + "_s2"):
                points[key]["energy"] = points[key]["energy"][:20]
        _t, first, _b = cells(short)
        self.assertEqual(first["text"], "NOT COMPARABLE: reactant final: no DMC energy")

        # 8. several reasons are all named
        def two(points):
            points[final + "_s2"].update(state=RUNNING)
            points[start]["checks"]["mdvmc"].update(ok=False, verdict="FAIL")
        _t, first, _b = cells(two)
        self.assertEqual(first["text"], "NOT COMPARABLE: reactant final: 1 of 2 series are DONE (DONE, RUNNING); "
                                        "reactant start: the VMC check without Jastrow factor did not pass (verdict FAIL)")

        # the rule itself, on rows of the stage
        engine.points = good
        by_key = {(r["target"], r["geometry"], r["label"]): r for r in runs.results()["rows"]}
        self.assertIsNone(RT.not_comparable([by_key["reactant", "final", "hf"], by_key["reactant", "start", "hf"]]))
        self.assertEqual(RT.not_comparable([by_key["reactant", "final", "hf"], by_key["reactant", "final", "n10000_c1e-03"]]),
                         "cutoff differs (hf at reactant final, 0.001 at reactant final); "
                         "n_det_max differs (None at reactant final, 10000 at reactant final)")
        self.assertEqual(RT.not_comparable([by_key["reactant", "final", "hf"], None]), "a geometry has no row for this cutoff")

    def test_a_stage_that_has_not_started_and_a_report_without_engine(self):
        folder, engine, _runs = self.stage("waiting", finish=False)
        table = self.table(folder, engine)
        self.assertEqual({r["energy"] for r in table["rows"]}, {None})
        self.assertTrue(all(d["text"].startswith("NOT COMPARABLE: ") for d in table["changes"] + table["barriers"]))
        self.assertEqual(table["changes"][0]["text"], "NOT COMPARABLE: reactant final: 0 of 2 series are DONE (QUEUED, QUEUED); "
                                                      "reactant start: 0 of 2 series are DONE (QUEUED, QUEUED)")
        files = build_report(folder, engine, gifs=False)               # the two files can be written at any time
        self.assertTrue(any(f.endswith("trial_wavefunctions.md") for f in files))
        text = read(os.path.join(folder, "results", "trial_wavefunctions.md"))
        self.assertIn("not available", text)
        self.assertIn("| not made yet | 0 |", text)
        # without an engine the points cannot be read: the two files are not written, and the summary says why
        files = build_report(folder, None, gifs=False)
        self.assertFalse(any("trial_wavefunctions" in f for f in files))
        summary = json.loads(read(os.path.join(folder, "results", "summary.json")))
        skipped = {x["item"]: x["reason"] for x in summary["skipped"]}
        self.assertIn("without an engine", skipped["trial_wavefunctions"])
        self.assertIn("trial_wavefunctions.csv", summary["removed_outdated_files"])
        self.assertFalse(os.path.exists(os.path.join(folder, "results", "trial_wavefunctions.csv")))


if __name__ == "__main__":
    unittest.main()
