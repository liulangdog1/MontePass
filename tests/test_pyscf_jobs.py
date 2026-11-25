"""PySCF surrogate jobs: the job script with a model in place of PySCF, and the handling of the batch jobs."""
import json
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from montepass.machine import Machine                                  # noqa: E402
from montepass.scheduler import base as sched                          # noqa: E402
from montepass.scheduler.base import JobTable, Scheduler               # noqa: E402
from montepass.surrogate import pyscf_jobs as PJ                       # noqa: E402
from montepass.units import BOHR_TO_ANGSTROM, HARTREE_TO_KCAL          # noqa: E402

SYMBOLS = ["O", "H", "H"]
X0 = np.array([[0.0, 0.0, 0.117], [0.0, 0.757, -0.469], [0.0, -0.757, -0.469]])


class FakeScheduler(Scheduler):
    """Records the submissions. A new job waits; the test sets the state of the job of a folder."""

    def __init__(self):
        self.submitted = []
        self.cancelled = []
        self.states = {}

    def submit(self, script, workdir, name="job", after=None):
        self.submitted.append({"script": script, "workdir": workdir, "name": name, "after": after})
        return "F%d" % len(self.submitted)

    def state(self, job_id):
        return self.states.get(job_id, sched.PENDING)

    def set(self, workdir, state):
        """Set the state of the newest job that was submitted for a folder."""
        newest = [i for i, job in enumerate(self.submitted, start=1) if job["workdir"] == workdir][-1]
        self.states["F%d" % newest] = state

    def cancel(self, job_id):
        self.cancelled.append(job_id)


def job_script():
    """The functions of the job script, without starting it (it has a ``__main__`` guard and loads PySCF late)."""
    ns = {"__name__": "surrogate_job_under_test"}
    exec(compile(PJ.JOB_SCRIPT, "surrogate_job.py", "exec"), ns)
    return ns


class Quadratic:
    """E = e0 + 0.5 (x - x0)^T A (x - x0), x in angstrom, E in hartree; the gradient is returned in hartree/bohr,
    as PySCF returns it. Counts its calls and can stop after a number of them."""

    def __init__(self, seed=0, stop_after=None):
        rng = np.random.default_rng(seed)
        B = rng.normal(size=(9, 9))
        self.A = B @ B.T / 9.0 + np.eye(9)
        self.e0 = -17.2
        self.calls = 0
        self.stop_after = stop_after

    def energy(self, x):
        d = (np.asarray(x, dtype=float) - X0).reshape(-1)
        return self.e0 + 0.5 * float(d @ self.A @ d)

    def __call__(self, coords, need_gradient=True):
        if self.stop_after is not None and self.calls >= self.stop_after:
            raise KeyboardInterrupt("the job reached its time limit")
        self.calls += 1
        d = (np.asarray(coords, dtype=float) - X0).reshape(-1)
        gradient = (self.A @ d).reshape(-1, 3) * BOHR_TO_ANGSTROM if need_gradient else None
        return self.energy(coords), gradient


def directions(seed=1, d=3):
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(9, d)))
    return q.T


class TestJobScript(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.ns = job_script()
        self.V = directions()
        self.x = X0 + 0.02 * np.arange(9).reshape(3, 3) / 9.0
        self.task = {"kind": "point", "symbols": SYMBOLS, "settings": {"xc": "b3lyp"}, "run": {"threads": 4},
                     "coords": self.x.tolist(), "vectors": self.V.tolist(), "hessian": True, "gradient": True,
                     "fd_step": 0.005}

    def test_point_task_gives_energy_gradient_and_hessian(self):
        f = Quadratic()
        lines = []
        res = self.ns["run_point"](self.task, f, self.tmp, lines.append)
        self.assertEqual(f.calls, 1 + 2 * 3)                                  # the centre and two per direction
        self.assertAlmostEqual(res["energy"], f.energy(self.x), places=12)
        want_g = f.A @ (self.x - X0).reshape(-1)                              # hartree/angstrom
        np.testing.assert_allclose(np.array(res["gradient"]).reshape(-1) / BOHR_TO_ANGSTROM, want_g, atol=1e-12)
        np.testing.assert_allclose(res["gradient_directions"], self.V @ want_g, atol=1e-12)
        want_K = self.V @ f.A @ self.V.T                                      # exact for a quadratic function
        np.testing.assert_allclose(res["hessian_directions"], want_K, atol=1e-9)
        self.assertLess(res["hessian_asymmetry"], 1e-9)
        np.testing.assert_allclose(res["curvature_from_energies"], np.diag(want_K), atol=1e-6)
        with open(os.path.join(self.tmp, PJ.RESULT_FILE)) as fh:
            stored = json.load(fh)
        self.assertEqual(stored["kind"], "point"); self.assertEqual(stored["n_directions"], 3)
        self.assertEqual(PJ.finished_calculations(self.tmp), 7)
        # the same folder again: nothing is computed twice
        again = Quadratic()
        res2 = self.ns["run_point"](self.task, again, self.tmp, lines.append)
        self.assertEqual(again.calls, 0)
        np.testing.assert_allclose(res2["hessian_directions"], res["hessian_directions"], atol=0)

    def test_a_second_start_continues_after_the_last_finished_calculation(self):
        f = Quadratic(stop_after=4)
        with self.assertRaises(KeyboardInterrupt):
            self.ns["run_point"](self.task, f, self.tmp, lambda _t: None)
        self.assertEqual(PJ.finished_calculations(self.tmp), 4)
        self.assertFalse(os.path.exists(os.path.join(self.tmp, PJ.RESULT_FILE)))
        g = Quadratic()
        res = self.ns["run_point"](self.task, g, self.tmp, lambda _t: None)
        self.assertEqual(g.calls, 3)                                          # 7 planned, 4 were in the checkpoint
        np.testing.assert_allclose(res["hessian_directions"], self.V @ g.A @ self.V.T, atol=1e-9)
        # a checkpoint of another task (another geometry) is refused
        other = dict(self.task, coords=(self.x + 0.01).tolist())
        with self.assertRaises(RuntimeError) as ctx:
            self.ns["run_point"](other, Quadratic(), self.tmp, lambda _t: None)
        self.assertIn("belongs to another task", str(ctx.exception))
        # a change of the run settings (threads) is not another task
        same = dict(self.task, run={"threads": 8})
        h = Quadratic()
        self.ns["run_point"](same, h, self.tmp, lambda _t: None)
        self.assertEqual(h.calls, 0)

    def test_energy_only_and_scan_tasks(self):
        f = Quadratic()
        res = self.ns["run_point"](dict(self.task, hessian=False, gradient=False), f, self.tmp, lambda _t: None)
        self.assertEqual(f.calls, 1)
        self.assertIsNone(res["gradient"]); self.assertIsNone(res["hessian_directions"])
        folder = os.path.join(self.tmp, "scan"); os.makedirs(folder)
        geoms = [X0, X0 + 0.01, self.x]
        g = Quadratic()
        scan = self.ns["run_scan"]({"kind": "scan", "symbols": SYMBOLS, "settings": {}, "run": {},
                                    "geometries": [x.tolist() for x in geoms]}, g, folder, lambda _t: None)
        self.assertEqual(g.calls, 3)
        np.testing.assert_allclose(scan["energies"], [g.energy(x) for x in geoms], atol=1e-12)
        with self.assertRaises(RuntimeError):
            self.ns["run_point"](dict(self.task, vectors=[]), Quadratic(), os.path.join(self.tmp, "scan"), lambda _t: None)


class TestSurrogateJobs(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.machine = Machine(os.path.join(ROOT, "machines", "example_slurm.ini"))
        self.sched = FakeScheduler()
        self.V = directions()
        self.table = JobTable(os.path.join(self.tmp, "jobs.json"))
        self.sur = PJ.PyscfSurrogate(os.path.join(self.tmp, "pyscf"), self.machine, self.sched, SYMBOLS, self.V,
                                     job_table=self.table)
        self.ns = job_script()

    def finish(self, tag, model=None):
        """Do what the batch job does: run the task of the folder with a model and write the marker file."""
        folder = self.sur.folder(tag)
        with open(os.path.join(folder, PJ.TASK_FILE)) as fh:
            task = json.load(fh)
        model = model or Quadratic()
        runner = self.ns["run_point"] if task["kind"] == "point" else self.ns["run_scan"]
        runner(task, model, folder, lambda _t: None)
        open(os.path.join(folder, PJ.STEP + ".done"), "w").close()
        self.sched.set(folder, sched.COMPLETED)
        return model

    def mark_failed(self, tag, reason="exit code 1", state=sched.FAILED):
        with open(os.path.join(self.sur.folder(tag), PJ.STEP + ".failed"), "w") as fh:
            fh.write(reason + "\n")
        self.sched.set(self.sur.folder(tag), state)

    def test_request_writes_the_folder_and_submits_once(self):
        tag = "min/it1/surrogate"
        self.assertIsNone(self.sur.result(tag))                               # never requested
        self.sur.request(tag, X0)
        folder = self.sur.folder(tag)
        for name in (PJ.TASK_FILE, PJ.SCRIPT_FILE, "job.sh", PJ.ATTEMPTS_FILE):
            self.assertTrue(os.path.exists(os.path.join(folder, name)), name)
        with open(os.path.join(folder, "job.sh")) as fh:
            script = fh.read()
        for needle in ("#SBATCH", "surrogate_job.py > surrogate.out 2> surrogate.err", "OMP_NUM_THREADS=",
                       "[ -s result.json ]"):
            self.assertIn(needle, script)
        with open(os.path.join(folder, PJ.TASK_FILE)) as fh:
            task = json.load(fh)
        self.assertEqual(task["kind"], "point"); self.assertTrue(task["hessian"])
        self.assertEqual(task["settings"]["xc"], "b3lyp"); self.assertEqual(task["settings"]["ecp"], "ccecp")
        self.assertEqual(len(self.sched.submitted), 1)
        self.sur.request(tag, X0 + 1e-12)                                     # the same geometry: nothing happens
        self.assertEqual(len(self.sched.submitted), 1)
        with self.assertRaises(ValueError) as ctx:                            # a tag names one geometry
            self.sur.request(tag, X0 + 0.01)
        self.assertIn("already used for another calculation", str(ctx.exception))
        self.assertIsNone(self.sur.result(tag))                               # waits
        self.sched.set(folder, sched.RUNNING)
        self.assertIsNone(self.sur.result(tag))
        self.assertEqual(self.sur.tags(), [tag])
        with self.assertRaises(ValueError):
            self.sur.folder("../outside")

    def test_result_units(self):
        tag = "min/it1/surrogate"
        x = X0 + 0.01 * np.arange(9).reshape(3, 3) / 9.0
        self.sur.request(tag, x)
        model = self.finish(tag)
        res = self.sur.result(tag)
        self.assertFalse(res.failed)
        self.assertAlmostEqual(res.energy, model.energy(x), places=12)        # hartree
        want_g = (model.A @ (x - X0).reshape(-1)).reshape(-1, 3) * HARTREE_TO_KCAL       # kcal/mol/angstrom
        np.testing.assert_allclose(res.gradient, want_g, atol=1e-7)
        np.testing.assert_allclose(res.hessian_directions, self.V @ model.A @ self.V.T * HARTREE_TO_KCAL, atol=1e-5)
        self.assertIn("largest asymmetry", res.message)
        self.assertEqual(self.sur.details(tag)["units"]["gradient"], "hartree/bohr")
        # a result file that cannot be used is reported, not read as "no result"
        with open(os.path.join(self.sur.folder(tag), PJ.RESULT_FILE), "w") as fh:
            fh.write("")
        bad = self.sur.result(tag)
        self.assertTrue(bad.failed); self.assertIn("empty", bad.message)

    def test_scan_energies(self):
        geoms = [X0, X0 + 0.02]
        self.assertIsNone(self.sur.scan("moved/a"))
        self.sur.request_scan("moved/a", geoms)
        self.assertIsNone(self.sur.scan("moved/a"))
        model = self.finish("moved/a")
        np.testing.assert_allclose(self.sur.scan("moved/a"), [model.energy(x) for x in geoms], atol=1e-12)
        with self.assertRaises(ValueError):
            self.sur.request_scan("moved/b", [])
        self.sur.request_scan("moved/c", geoms)
        self.mark_failed("moved/c", "exit code 137")
        with self.assertRaises(RuntimeError) as ctx:
            self.sur.scan("moved/c")
        self.assertIn("exit code 137", str(ctx.exception))

    def test_failed_job_is_reported_and_retry_submits_it_again(self):
        tag = "min/it2/surrogate"
        self.sur.request(tag, X0)
        self.sur.request("min/it1/surrogate", X0 + 0.01)
        self.finish("min/it1/surrogate")
        self.mark_failed(tag, "exit code 137 (out of memory)")
        res = self.sur.result(tag)
        self.assertTrue(res.failed); self.assertIn("out of memory", res.message)
        self.assertEqual(len(self.sched.submitted), 2)
        self.assertTrue(self.sur.result(tag).failed)                           # asking again changes nothing
        self.assertEqual(len(self.sched.submitted), 2)
        # the command "retry": only the failed job is submitted again
        self.assertEqual(self.sur.retry_failed(), [tag])
        self.assertEqual(len(self.sched.submitted), 3)
        self.assertEqual(self.sched.submitted[-1]["workdir"], self.sur.folder(tag))
        self.assertFalse(os.path.exists(os.path.join(self.sur.folder(tag), PJ.STEP + ".failed")))
        self.sched.set(self.sur.folder(tag), sched.RUNNING)
        self.assertIsNone(self.sur.result(tag))                               # runs again
        self.assertEqual(self.sur.retry_failed(), [])                         # nothing failed now
        self.finish(tag)
        self.assertFalse(self.sur.result(tag).failed)
        self.assertFalse(self.sur.result("min/it1/surrogate").failed)

    def test_time_limit_continues_while_calculations_finish(self):
        tag = "min/it1/surrogate"
        self.sur.request(tag, X0)
        folder = self.sur.folder(tag)
        with open(os.path.join(folder, PJ.TASK_FILE)) as fh:
            task = json.load(fh)

        def time_limit(calls):
            try:
                self.ns["run_point"](task, Quadratic(stop_after=calls), folder, lambda _t: None)
            except KeyboardInterrupt:
                pass
            self.mark_failed(tag, "time limit of 13968 s reached", state=sched.TIMEOUT)

        time_limit(3)                                                         # 3 of 7 calculations finished
        self.assertIsNone(self.sur.result(tag))                               # submitted again by itself
        self.assertEqual(len(self.sched.submitted), 2)
        time_limit(0)                                                         # nothing new in the second attempt
        res = self.sur.result(tag)
        self.assertTrue(res.failed)
        self.assertIn("no calculation finished in this attempt", res.message)
        self.assertEqual(len(self.sched.submitted), 2)
        # by request the job gets a new set of attempts
        self.assertEqual(self.sur.retry_failed(), [tag])
        self.assertEqual(len(self.sched.submitted), 3)
        self.finish(tag)
        self.assertFalse(self.sur.result(tag).failed)

    def test_settings_are_checked(self):
        with self.assertRaises(ValueError):
            PJ.PyscfSurrogate(self.tmp, self.machine, self.sched, SYMBOLS, self.V, settings={"functional": "pbe"})
        with self.assertRaises(ValueError):
            PJ.PyscfSurrogate(self.tmp, self.machine, self.sched, SYMBOLS, 2.0 * self.V)
        with self.assertRaises(ValueError):
            PJ.PyscfSurrogate(self.tmp, self.machine, self.sched, SYMBOLS, self.V[:, :6])
        no_dirs = PJ.PyscfSurrogate(os.path.join(self.tmp, "e"), self.machine, self.sched, SYMBOLS, None)
        with self.assertRaises(ValueError):
            no_dirs.request("a", X0)                                          # a Hessian needs directions
        no_dirs.request("a", X0, hessian=False)
        with self.assertRaises(ValueError):
            self.sur.request("b", X0[:2])


if __name__ == "__main__":
    unittest.main()
