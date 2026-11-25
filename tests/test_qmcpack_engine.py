"""The engine for real QMC runs, tested without a cluster.

A fake scheduler records the submitted jobs and lets a test set their states. The outputs of a
job (marker files, energy.dat, system.json, jastrow.json, a growing dmc.s001.scalar.dat) are
written by the test into the point folder. The DMC trace, the zero-Jastrow VMC trace and the
Jastrow file come from a real run of the chain for a water molecule (``tests/data/h2o_*``).

The last test class runs the generated job script with ``bash`` and small stand-in programs
for python, convert4qmc, mpirun and qmcpack.
"""
import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

from montepass.engine import qmcpack as E  # noqa: E402
from montepass.engine import qmcpack_inputs as Q  # noqa: E402
from montepass.engine.base import ABSENT, DONE, FAILED, QUEUED, RUNNING, STOPPED, PointSpec  # noqa: E402
from montepass.geometry import read_xyz  # noqa: E402
from montepass.scheduler import base as S  # noqa: E402
from montepass.scheduler.base import JobTable, Scheduler  # noqa: E402
from montepass.stats import read_scalar  # noqa: E402
from montepass.units import ANGSTROM_TO_BOHR  # noqa: E402

DATA = os.path.join(HERE, "data")
TRACE = os.path.join(DATA, "h2o_dmc.s001.scalar.dat")          # 200 DMC blocks
VMC0 = os.path.join(DATA, "h2o_vmc0.s000.scalar.dat")          # 100 VMC blocks without Jastrow factor
JASTROW = os.path.join(DATA, "h2o_jastrow.json")

# water with the atoms NOT grouped by element
SYMBOLS = ["H", "O", "H"]
COORDS = np.array([[0.0, 0.7631265780, -0.5855676186], [0.0, 0.0, 0.0], [0.0, -0.7631265780, -0.5855676186]])
E_SCF = -17.23711747822735
E_DET = -16.98            # energy of the determinant that fits the zero-Jastrow trace (-16.9759 +/- 0.0075 Ha)

PROFILE = """
[machine]
name = test-cluster
scheduler = slurm

[setup.qmc]
lines =
    module load qmc-tools

[executables]
python = python3
qmcpack = qmcpack_real
convert4qmc = convert4qmc

[launch]
qmc = mpirun -np {ranks} --bind-to core

[pseudopotentials]
dir = %(pp)s

[slurm]
account = testaccount

[job.scf]
nodes = 1
tasks = 1
cpus_per_task = 4
mem = 8G
time = 00:30:00

[job.qmc]
nodes = 1
tasks = 32
mem = 16G
time = 02:00:00
"""


def write_pp(folder, element, zval):
    with open(os.path.join(folder, "%s.ccECP.xml" % element), "w") as fh:
        fh.write('<?xml version="1.0"?>\n<pseudo>\n  <header symbol="%s" atomic-number="1" zval="%s" '
                 'creator="x" flavor="y"/>\n</pseudo>\n' % (element, zval))


def trace_lines():
    with open(TRACE) as fh:
        return fh.read().split("\n")[:-1]           # the header and 200 rows, without line breaks


class FakeScheduler(Scheduler):
    """Records submissions. A test sets the state of a job by hand."""

    def __init__(self):
        self.submitted = []
        self.state_of = {}
        self.cancelled = []
        self.questions = 0
        self.refuse = None
        self._n = 1000

    def submit(self, script, workdir, name="job", after=None):
        if self.refuse:
            raise RuntimeError(self.refuse)
        self._n += 1
        job_id = str(self._n)
        with open(script) as fh:
            text = fh.read()
        self.submitted.append({"id": job_id, "script": script, "workdir": workdir, "name": name,
                               "after": list(after or []), "text": text})
        self.state_of[job_id] = S.PENDING
        return job_id

    def state(self, job_id):
        return self.state_of.get(job_id, S.UNKNOWN)

    def states(self, job_ids):
        self.questions += 1
        return {j: self.state(j) for j in job_ids}

    def cancel(self, job_id):
        self.cancelled.append(job_id)
        if self.state_of.get(job_id) in (S.PENDING, S.RUNNING):
            self.state_of[job_id] = S.CANCELLED


class EngineCase(unittest.TestCase):
    """A temporary project folder, a machine profile, a fake scheduler, a clock, and fake job outputs."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = os.path.realpath(tmp.name)
        self.pp = os.path.join(self.tmp, "pseudo")
        os.makedirs(self.pp)
        write_pp(self.pp, "H", 1)
        write_pp(self.pp, "O", 6)
        self.profile = os.path.join(self.tmp, "machine.ini")
        with open(self.profile, "w") as fh:
            fh.write(PROFILE % {"pp": self.pp})
        self.sched = FakeScheduler()
        self.now = 1000.0
        self.root = os.path.join(self.tmp, "points")

    def engine(self, **settings):
        base = {"poll_seconds": 0, "settle_seconds": 0, "unknown_seconds": 0, "dmc": {"steps": 100}}
        return E.QmcpackEngine(self.root, self.profile, self.sched, E.merge_settings(base, settings),
                               clock=lambda: self.now)

    def spec(self, key="min/it1/centre", blocks=200, **kw):
        return PointSpec(key=key, symbols=list(SYMBOLS), coords=COORDS.copy(), blocks=blocks, **kw)

    # ----------------------------------------------------------------- files of a point
    def path(self, key, name):
        return os.path.join(self.root, *(key.split("/") + [name]))

    def write(self, key, name, text):
        with open(self.path(key, name), "w") as fh:
            fh.write(text)

    def read_text(self, key, name):
        with open(self.path(key, name)) as fh:
            return fh.read()

    def read_json(self, key, name):
        return json.loads(self.read_text(key, name))

    def mark(self, key, step, start, end=None):
        self.write(key, step + ".started", "%d\n" % start)
        if end is not None:
            self.write(key, step + ".done", "%d\n" % end)

    def system(self, key, e_det=E_DET, zeff=None):
        symbols, coords = read_xyz(self.path(key, "geom.xyz"))
        order = self.read_json(key, "order.json")["order"]
        grouped = [symbols[i] for i in order]
        charge = dict({"H": 1, "O": 6}, **(zeff or {}))
        return {"symbols": grouped, "coords_bohr": (coords[order] * ANGSTROM_TO_BOHR).tolist(),
                "zeff": [charge[s] for s in grouped], "nup": 4, "ndn": 4, "nao": 58, "energy": E_SCF,
                "e_determinant": e_det, "xc": "b3lyp", "basis": "ccecpccpvtz", "ecp": "ccecp", "restricted": True}

    def fake_scf(self, key, t0=1000, **kw):
        self.write(key, "system.json", json.dumps(self.system(key, **kw)))
        self.write(key, "energy.dat", "%.18e\n%.18e\n" % (E_SCF, 0.0))
        self.write(key, "scf.h5", "orbital file\n")
        self.mark(key, "scf", t0, t0 + 60)

    def fake_vmc0(self, key, t0=1070):
        shutil.copyfile(VMC0, self.path(key, "vmc0.s000.scalar.dat"))
        self.mark(key, "vmc0", t0, t0 + 30)

    def fake_opt(self, key, t0=1100, jastrow=None):
        if jastrow is None:
            shutil.copyfile(JASTROW, self.path(key, "jastrow.json"))
        else:
            self.write(key, "jastrow.json", json.dumps(jastrow))
        self.write(key, "dmc.in.xml", "<simulation/>\n")
        self.mark(key, "opt", t0, t0 + 100)

    def fake_setup(self, key, **kw):
        """All steps before the DMC are done."""
        self.fake_scf(key, **kw)
        self.mark(key, "c4q", 1060, 1070)
        if "vmc0" in self.read_json(key, "point.json")["steps"]:
            self.fake_vmc0(key)
        self.fake_opt(key)

    def fake_rows(self, key, n, tail="", project="dmc"):
        lines = trace_lines()
        self.write(key, "%s.s001.scalar.dat" % project, "\n".join(lines[:n + 1]) + "\n" + tail)
        if not os.path.exists(self.path(key, "dmc.started")):
            self.write(key, "dmc.started", "1200\n")

    def running_point(self, eng, key="min/it1/centre", rows=50, **kw):
        """Submit a point and bring it to the DMC step with ``rows`` blocks written."""
        eng.submit(self.spec(key, **kw))
        job = self.sched.submitted[-1]["id"]
        self.sched.state_of[job] = S.RUNNING
        self.fake_setup(key)
        self.fake_rows(key, rows)
        return job


# ===================================================================================== tests
class TestSeedRule(unittest.TestCase):
    def test_prime_ranges_printed_by_qmcpack(self):
        # (offset, MPI ranks) -> range printed by five real QMCPACK runs
        seen = {(363, 32): (2467, 2719), (264, 64): (1699, 2213), (24, 64): (101, 463), (928, 64): (7297, 7873),
                (653, 64): (4903, 5441)}
        for (offset, ranks), expected in seen.items():
            self.assertEqual(E.printed_prime_range(offset, ranks), expected)
        self.assertEqual(E.printed_prime_range(10264, 64), (101, 463))        # QMCPACK takes the seed modulo 1024

    def test_window_and_overlap(self):
        self.assertEqual(E.seed_width(32), 64)
        self.assertEqual(E.seed_window(1024 + 5, 32), (5, 68))
        self.assertTrue(E.windows_overlap((5, 68), (68, 131)))
        self.assertFalse(E.windows_overlap((5, 68), (69, 132)))
        self.assertEqual([E.max_series(r) for r in (32, 64, 128)], [16, 8, 4])

    def test_all_series_of_one_geometry_are_disjoint(self):
        for ranks in (32, 48, 64, 128):
            n = E.max_series(ranks)
            windows = [E.seed_window(E.point_seeds("min/it1/centre", s, ranks)["dmc"], ranks) for s in range(1, n + 1)]
            for i in range(n):
                for j in range(i + 1, n):
                    self.assertFalse(E.windows_overlap(windows[i], windows[j]), (ranks, i + 1, j + 1))
            with self.assertRaises(ValueError) as ctx:
                E.point_seeds("min/it1/centre", n + 1, ranks)
            self.assertIn("at most %d runs" % n, str(ctx.exception))

    def test_seeds_of_one_point(self):
        seeds = E.point_seeds("min/it1/centre", 1, 32)
        self.assertEqual(sorted(seeds), ["dmc", "opt", "vmc0"])
        self.assertEqual(len(set(seeds.values())), 3)
        self.assertTrue(all(0 <= v < 1024 for v in seeds.values()))
        self.assertEqual(seeds, E.point_seeds("min/it1/centre", 1, 32))                 # the same in every process
        self.assertNotEqual(seeds["dmc"], E.point_seeds("min/it1/d0_p091", 1, 32)["dmc"])
        self.assertEqual(sorted(E.point_seeds("min/it1/centre", 2, 32)), ["dmc"])       # a further series runs only DMC
        self.assertEqual((seeds["dmc"] + 7) % 1024, E.point_seeds("min/it1/centre", 1, 32, shift=7)["dmc"])

    def test_seed_report_of_an_output_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "dmc.out")
            self.assertIsNone(E.read_seed_report(path))
            with open(path, "w") as fh:
                fh.write(" Random Number\n -------------\n  Offset for the random number seeds from input file (mod 1024): 363\n\n"
                         "  Range of prime numbers to use as seeds over processors and threads = 2467-2719\n")
            self.assertEqual(E.read_seed_report(path), {"offset": 363, "from_input": True, "prime_range": [2467, 2719]})
            with open(path, "w") as fh:
                fh.write("  Offset for the random number seeds based on time: 21\n")
            self.assertEqual(E.read_seed_report(path), {"offset": 21, "from_input": False, "prime_range": None})


class TestSmallTools(unittest.TestCase):
    def test_time_limits(self):
        for text, seconds in (("30", 1800), ("10:30", 630), ("02:00:00", 7200), ("1-12", 129600), ("1-00:30", 88200),
                              ("2-01:02:03", 176523)):
            self.assertEqual(E.parse_time(text), seconds)
        self.assertIsNone(E.parse_time(None))
        self.assertIsNone(E.parse_time(" "))
        with self.assertRaises(ValueError):
            E.parse_time("two hours")

    def test_settings_are_merged_key_by_key(self):
        merged = E.merge_settings(E.DEFAULT_SETTINGS, {"dmc": {"steps": 20}, "charge": 1})
        self.assertEqual(merged["dmc"], {"timestep": 0.005, "steps": 20, "warmup_steps": 2000, "walkers": 2048})
        self.assertEqual(merged["charge"], 1)
        self.assertEqual(E.DEFAULT_SETTINGS["dmc"]["steps"], 500)                     # the defaults are not changed

    def test_wrong_settings_are_refused(self):
        for wrong in ({"vmc0_check": "sometimes"}, {"jastrow_reuse": "all"}, {"dmc": {"blocks": 100}},
                      {"dmc": {"time_step": 0.01}}, {"opt": {"cycle": 4}}, {"time_limits": {"jastrow": 10}},
                      {"max_attempts": 0}, {"job_time": "soon"}, {"pseudo_files": "move"}, {"dmc": {"walkers": 0}}):
            with self.assertRaises(ValueError, msg=str(wrong)):
                E.validate_settings(E.merge_settings(E.DEFAULT_SETTINGS, wrong))
        E.validate_settings(E.merge_settings(E.DEFAULT_SETTINGS, {"dmc": {"checkpoint": 50}, "opt": {"cycles_reuse": 3}}))

    def test_trace_reader_drops_a_line_without_line_break(self):
        lines = trace_lines()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "dmc.s001.scalar.dat")
            with open(path, "w") as fh:
                fh.write("\n".join(lines[:31]) + "\n" + lines[31][:-5])              # all 12 fields, the last one cut short
            self.assertEqual(len(read_scalar(path)["LocalEnergy"]), 31)             # the plain reader accepts the cut line
            cols = E.read_trace(path)
            self.assertEqual(len(cols["LocalEnergy"]), 30)
            self.assertEqual(len(cols["AcceptRatio"]), 30)
            with open(path, "w") as fh:
                fh.write(lines[0] + "\n")
            self.assertEqual(len(E.read_trace(path)["LocalEnergy"]), 0)


class TestNewPoint(EngineCase):
    def test_folder_contents(self):
        eng = self.engine()
        key = "min/it1/centre"
        eng.submit(self.spec(key))
        folder = eng.folder(key)
        self.assertEqual(folder, os.path.join(self.root, "min", "it1", "centre"))
        for name in ("geom.xyz", "order.json", "scf.py", "qmc_settings.json", "H.ccECP.xml", "O.ccECP.xml", "job.sh",
                     "point.json"):
            self.assertTrue(os.path.exists(os.path.join(folder, name)), name)
        self.assertFalse(os.path.exists(os.path.join(folder, "job_scf.sh")))
        for el in ("H", "O"):
            link = os.path.join(folder, "%s.ccECP.xml" % el)
            self.assertTrue(os.path.islink(link))
            self.assertEqual(os.path.realpath(link), os.path.join(self.pp, "%s.ccECP.xml" % el))
        # one job, written into the job table
        self.assertEqual(len(self.sched.submitted), 1)
        sub = self.sched.submitted[0]
        self.assertEqual(sub["script"], os.path.join(folder, "job.sh"))
        self.assertEqual(sub["workdir"], folder)
        self.assertEqual(sub["after"], [])
        table = JobTable(os.path.join(self.root, "jobs.json"))
        self.assertEqual(table.current(key)["id"], sub["id"])
        self.assertEqual(table.current(key)["kind"], "qmc")
        # settings read inside the job
        cfg = self.read_json(key, "qmc_settings.json")
        record = self.read_json(key, "point.json")
        self.assertEqual(cfg["pseudo_dir"], "")
        self.assertEqual(cfg["dmc"], {"timestep": 0.005, "steps": 100, "warmup_steps": 2000, "walkers": 2048,
                                      "blocks": 200, "seed": record["seeds"]["dmc"], "project": "dmc"})
        self.assertEqual(cfg["opt"], {"cycles": 8, "samples": 100000, "seed": record["seeds"]["opt"]})
        self.assertEqual(cfg["vmc0"], {"walkers": 2048, "seed": record["seeds"]["vmc0"]})
        self.assertEqual(cfg["jastrow"], {"j1_size": 6, "j1_rcut": 6.0, "j2_size": 8, "j2_rcut": 8.0})
        self.assertEqual((cfg["charge"], cfg["spin"]), (0, 0))
        self.assertNotIn("jastrow_start", cfg)
        # the seeds: explicit, recorded, different for the three runs
        self.assertEqual(record["seeds"], E.point_seeds(key, 1, 32))
        self.assertEqual(len(set(record["seeds"].values())), 3)
        self.assertEqual(record["seed_windows"]["dmc"], list(E.seed_window(record["seeds"]["dmc"], 32)))
        data = eng.read(key)
        self.assertEqual(data.state, QUEUED)
        self.assertEqual(data.nblocks, 0)
        self.assertIsNone(data.e_surrogate)
        self.assertEqual(data.meta["seed"], record["seeds"]["dmc"])
        self.assertEqual(data.meta["requested_blocks"], 200)
        self.assertEqual((data.meta["steps"], data.meta["timestep"], data.meta["walkers"]), (100, 0.005, 2048))
        self.assertEqual(data.meta["job_ids"], [sub["id"]])
        self.assertEqual(data.meta["attempts"], 1)
        self.assertEqual(eng.read("no/such/point").state, ABSENT)

    def test_job_script(self):
        eng = self.engine()
        eng.submit(self.spec("min/it1/centre"))
        text = self.sched.submitted[0]["text"]
        folder = eng.folder("min/it1/centre")
        for needle in ("#!/bin/bash", "#SBATCH --job-name=q_min_it1_centre", "#SBATCH --chdir=%s" % folder,
                       "#SBATCH --output=%s" % os.path.join(folder, "job.out"), "#SBATCH --ntasks=32",
                       "#SBATCH --time=02:00:00", "#SBATCH --account=testaccount", "module load qmc-tools",
                       # thread variables
                       "export OMP_NUM_THREADS=1", "export MKL_NUM_THREADS=1",
                       # marker files
                       'mark "$STEP.started"', 'mark "$STEP.done"', 'mark "$STEP.failed"', 'if [ -e "$STEP.done" ]',
                       "stop.requested", "dmc.stopped",
                       # every command runs under a time limit
                       'timeout -k 30 "$LIMIT" "$@"',
                       "    run_limited 7200 scf.out - python3 -u scf.py",
                       "    run_limited 600 c4q.out - convert4qmc -prefix c4q -orbitals scf.h5",
                       "    run_limited 1800 vmc0.out vmc0.err mpirun -np 32 --bind-to core qmcpack_real vmc0.in.xml",
                       "    run_limited 7200 opt.out opt.err mpirun -np 32 --bind-to core qmcpack_real opt.in.xml",
                       "    run_limited 7200 dmc.out dmc.err mpirun -np 32 --bind-to core qmcpack_real dmc.in.xml",
                       # success texts and the block count
                       "need_text scf.out SCF_DONE", 'need_text c4q.out "QMCGaussianParserBase::dump"',
                       'need_text opt.out "QMCPACK execution completed successfully"',
                       'need_text dmc.out "QMCPACK execution completed successfully"',
                       "dmc.s001.scalar.dat holds $blocks of 200 blocks",
                       # checks made inside the job
                       "package check-system", "package check-vmc0", "package write-opt", "package write-dmc",
                       "python3 -m montepass.engine.qmcpack", "MPI start failure"):
            self.assertIn(needle, text)
        order = [text.index("if begin_step %s; then" % step) for step in ("scf", "c4q", "vmc0", "opt", "dmc")]
        self.assertEqual(order, sorted(order))
        self.assertTrue(text.rstrip().endswith("exit 0"))
        self.assertEqual(text, self.read_text("min/it1/centre", "job.sh"))

    def test_time_limits_follow_the_settings(self):
        eng = self.engine(job_time="00:20:00", time_limits={"opt": 900, "scf": 100000})
        eng.submit(self.spec("a"))
        text = self.sched.submitted[0]["text"]
        self.assertIn("#SBATCH --time=00:20:00", text)
        self.assertIn("run_limited 1200 scf.out", text)             # never longer than the job itself
        self.assertIn("run_limited 600 c4q.out", text)
        self.assertIn("run_limited 900 opt.out", text)
        self.assertIn("run_limited 1200 dmc.out", text)

    def test_atoms_are_grouped_for_the_scf_only(self):
        eng = self.engine()
        eng.submit(self.spec("a"))
        symbols, coords = read_xyz(self.path("a", "geom.xyz"))
        self.assertEqual(symbols, ["H", "O", "H"])                   # the order of the user
        np.testing.assert_allclose(coords, COORDS, atol=1e-10)
        order = self.read_json("a", "order.json")
        self.assertEqual(order["order"], Q.species_order(SYMBOLS))
        self.assertEqual(order["order"], [0, 2, 1])
        self.assertEqual(order["grouped_symbols"], ["H", "H", "O"])
        scf = self.read_text("a", "scf.py")
        compile(scf, "scf.py", "exec")
        block = scf.split("mol.atom = '''")[1].split("'''")[0].strip().split("\n")
        self.assertEqual([line.split()[0] for line in block], ["H", "H", "O"])
        np.testing.assert_allclose([[float(x) for x in line.split()[1:]] for line in block], COORDS[[0, 2, 1]], atol=1e-10)
        for needle in ("mol.basis = 'ccecpccpvtz'", "mol.ecp = 'ccecp'", "mf.xc = 'b3lyp'", "mf.grids.level = 3",
                       "lib.num_threads(4)", "e_determinant", "print('SCF_DONE')"):
            self.assertIn(needle, scf)

    def test_submit_twice_does_nothing_the_second_time(self):
        eng = self.engine()
        eng.submit(self.spec("a"))
        before = {name: self.read_text("a", name) for name in ("geom.xyz", "scf.py", "job.sh", "qmc_settings.json")}
        eng.submit(PointSpec(key="a", symbols=list(SYMBOLS), coords=COORDS.copy(), blocks=77))    # other block count: ignored
        eng.submit(self.spec("a"))
        self.assertEqual(len(self.sched.submitted), 1)
        # the same key with another geometry is an error: the caller's state does not fit the points on disk
        moved = PointSpec(key="a", symbols=list(SYMBOLS), coords=COORDS + 0.3, blocks=77)
        with self.assertRaises(ValueError) as ctx:
            eng.submit(moved)
        self.assertIn("exists with another geometry", str(ctx.exception))
        self.assertEqual(len(self.sched.submitted), 1)
        self.assertEqual(before, {name: self.read_text("a", name) for name in before})
        self.assertEqual(eng.read("a").meta["requested_blocks"], 200)
        other = self.engine()                                        # a new process finds the point in its files
        other.submit(self.spec("a"))
        self.assertEqual(len(self.sched.submitted), 1)
        self.assertEqual(JobTable(os.path.join(self.root, "jobs.json")).attempts("a"), 1)

    def test_walkers_become_a_multiple_of_the_ranks(self):
        eng = self.engine(dmc={"walkers": 1000})
        eng.submit(self.spec("a"))
        self.assertEqual(self.read_json("a", "qmc_settings.json")["dmc"]["walkers"], 1024)         # 32 x 32
        meta = eng.read("a").meta
        self.assertEqual((meta["walkers"], meta["walkers_requested"], meta["ranks"]), (1024, 1000, 32))
        eng.submit(self.spec("b", settings={"ranks": 48, "dmc": {"walkers": 2048}}))
        self.assertEqual(self.read_json("b", "qmc_settings.json")["dmc"]["walkers"], 2064)         # 43 x 48
        self.assertEqual(self.read_json("b", "qmc_settings.json")["vmc0"]["walkers"], 2064)
        self.assertEqual(eng.read("b").meta["walkers"], 2064)
        self.assertIn("mpirun -np 48 --bind-to core qmcpack_real dmc.in.xml", self.sched.submitted[-1]["text"])
        self.assertIn("#SBATCH --ntasks=48", self.sched.submitted[-1]["text"])
        eng.submit(self.spec("c", settings={"dmc": {"walkers": 2048}}))
        self.assertEqual(eng.read("c").meta["walkers"], 2048)                                      # already a multiple

    def test_settings_of_one_point_override_the_project(self):
        eng = self.engine(opt={"samples": 50000})
        eng.submit(self.spec("a", blocks=64, settings={"dmc": {"steps": 20, "checkpoint": 50}, "charge": 0}))
        cfg = self.read_json("a", "qmc_settings.json")
        self.assertEqual((cfg["dmc"]["steps"], cfg["dmc"]["blocks"], cfg["dmc"]["checkpoint"]), (20, 64, 50))
        self.assertEqual(cfg["dmc"]["warmup_steps"], 2000)           # project value kept
        self.assertEqual(cfg["opt"]["samples"], 50000)

    def test_missing_pseudopotential_file(self):
        os.remove(os.path.join(self.pp, "O.ccECP.xml"))
        eng = self.engine()
        with self.assertRaises(FileNotFoundError) as ctx:
            eng.submit(self.spec("a"))
        text = str(ctx.exception)
        self.assertIn("pseudopotential file missing for element O", text)
        self.assertIn(os.path.join(self.pp, "O.ccECP.xml"), text)
        self.assertIn(self.profile, text)
        self.assertEqual(self.sched.submitted, [])
        self.assertEqual(eng.read("a").state, ABSENT)

    def test_pseudopotential_copies(self):
        eng = self.engine(pseudo_files="copy")
        eng.submit(self.spec("a"))
        self.assertFalse(os.path.islink(self.path("a", "O.ccECP.xml")))
        self.assertIn('zval="6"', self.read_text("a", "O.ccECP.xml"))

    def test_wrong_requests(self):
        eng = self.engine()
        for key in ("", "../a", "a//b", "a b", "/a"):
            with self.assertRaises(ValueError, msg=key):
                eng.submit(self.spec(key))
        with self.assertRaises(ValueError):
            eng.submit(self.spec("a", blocks=0))
        with self.assertRaises(ValueError):
            eng.submit(self.spec("a", series=2))                    # no base point named
        with self.assertRaises(ValueError):
            eng.submit(self.spec("a", series=2, base_key="never"))   # the base point does not exist
        with self.assertRaises(ValueError):
            eng.submit(self.spec("a", settings={"dmc": {"blocks": 5}}))
        self.assertEqual(self.sched.submitted, [])

    def test_a_refused_submission_is_tried_again(self):
        eng = self.engine()
        self.sched.refuse = "sbatch: error: QOSMaxSubmitJobPerUserLimit"
        with self.assertRaises(RuntimeError):
            eng.submit(self.spec("a"))
        data = eng.read("a")
        self.assertEqual(data.state, QUEUED)
        self.assertIn("QOSMaxSubmitJobPerUserLimit", data.message)
        self.sched.refuse = None
        self.assertEqual(eng.read("a").state, QUEUED)                # this read started the job
        self.assertEqual(len(self.sched.submitted), 1)
        self.assertEqual(eng.read("a").message, "the job waits in the queue")


class TestStates(EngineCase):
    def test_queued_running_done_with_a_growing_trace(self):
        eng = self.engine()
        key = "min/it1/centre"
        eng.submit(self.spec(key))
        job = self.sched.submitted[0]["id"]
        self.assertEqual(eng.read(key).state, QUEUED)
        self.sched.state_of[job] = S.RUNNING
        self.fake_scf(key)
        data = eng.read(key)
        self.assertEqual((data.state, data.nblocks), (QUEUED, 0))    # no DMC block yet
        self.assertAlmostEqual(data.e_surrogate, E_SCF, places=12)
        self.assertEqual(data.meta["step"], "c4q")
        self.assertTrue(data.meta["checks"]["system"]["ok"])
        self.mark(key, "c4q", 1060, 1070)
        self.fake_vmc0(key)
        self.fake_opt(key)
        self.assertEqual(eng.read(key).meta["step"], "dmc")
        reference = read_scalar(TRACE)
        for n in (40, 120):
            self.fake_rows(key, n)
            data = eng.read(key)
            self.assertEqual((data.state, data.nblocks), (RUNNING, n))
            np.testing.assert_array_equal(data.energy, reference["LocalEnergy"][:n])
            np.testing.assert_array_equal(data.weight, reference["BlockWeight"][:n])
        self.fake_rows(key, 200)
        self.assertEqual(eng.read(key).state, RUNNING)               # all rows, but the job did not write dmc.done yet
        self.write(key, "dmc.done", "1320\n")
        self.sched.state_of[job] = S.COMPLETED
        data = eng.read(key)
        self.assertEqual((data.state, data.nblocks), (DONE, 200))
        self.assertTrue(data.final)
        meta = data.meta
        self.assertAlmostEqual(meta["acceptance"], 0.99727, places=4)
        self.assertAlmostEqual(meta["seconds_per_block"], 0.5837, places=3)
        self.assertAlmostEqual(meta["vmc_energy"], -17.2062896255, places=9)
        self.assertAlmostEqual(meta["vmc_variance"], 0.3803101890, places=9)
        self.assertEqual(meta["jastrow_source"], "./opt.s002.opt.xml")
        self.assertTrue(meta["zero_jastrow"]["ok"])
        self.assertAlmostEqual(meta["zero_jastrow"]["energy"], -16.975903, places=5)
        self.assertTrue(meta["checks"]["jastrow"]["ok"])
        self.assertIsNone(meta["step"])
        # a final point is not looked up in the batch system again
        asked = self.sched.questions
        self.assertEqual(eng.read(key).state, DONE)
        self.assertEqual(self.sched.questions, asked)
        # a new process reads the same result from the files
        again = self.engine().read(key)
        self.assertEqual((again.state, again.nblocks), (DONE, 200))
        np.testing.assert_array_equal(again.energy, data.energy)

    def test_half_written_last_line(self):
        eng = self.engine()
        self.running_point(eng, "a", rows=30)
        lines = trace_lines()
        self.fake_rows("a", 30, tail=" ".join(lines[31].split()[:5]))            # 5 of 12 fields
        self.assertEqual(eng.read("a").nblocks, 30)
        self.fake_rows("a", 30, tail=lines[31][:-5])                             # 12 fields, the last one cut, no line break
        data = eng.read("a")
        self.assertEqual((data.state, data.nblocks), (RUNNING, 30))
        self.assertAlmostEqual(data.meta["acceptance"], float(np.mean(read_scalar(TRACE)["AcceptRatio"][:30])), places=12)
        self.fake_rows("a", 31)
        self.assertEqual(eng.read("a").nblocks, 31)

    def test_done_marker_with_too_few_blocks_is_a_failure(self):
        eng = self.engine()
        job = self.running_point(eng, "a", rows=150)
        self.write("a", "dmc.done", "1300\n")
        self.sched.state_of[job] = S.COMPLETED
        data = eng.read("a")
        self.assertEqual(data.state, FAILED)
        self.assertIn("150 of 200 blocks", data.message)
        self.assertEqual(data.nblocks, 150)                          # the blocks stay readable
        self.assertEqual(len(self.sched.submitted), 1)

    def test_the_batch_system_is_asked_once_per_poll_time(self):
        eng = self.engine(poll_seconds=30)
        self.running_point(eng, "a", rows=10)
        eng.submit(self.spec("b"))
        eng.read("a")
        asked = self.sched.questions
        for _ in range(5):
            eng.read("a")
            eng.read("b")
        self.assertEqual(self.sched.questions, asked)
        self.now += 31
        eng.read("a")
        eng.read("b")
        self.assertEqual(self.sched.questions, asked + 1)

    def test_cost(self):
        eng = self.engine()
        self.assertEqual(eng.cost("a"), 0.0)
        job = self.running_point(eng, "a", rows=50)
        self.now = 1250.0
        # scf 60 s, c4q 10 s, vmc0 30 s, opt 100 s, dmc 50 s so far (started at 1200); 32 cores
        self.assertAlmostEqual(eng.cost("a"), (60 + 10 + 30 + 100 + 50) * 32 / 3600.0, places=9)
        self.fake_rows("a", 200)
        self.write("a", "dmc.done", "1320\n")
        self.sched.state_of[job] = S.COMPLETED
        self.assertEqual(eng.read("a").state, DONE)
        self.now = 9999.0
        self.assertAlmostEqual(eng.cost("a"), (60 + 10 + 30 + 100 + 120) * 32 / 3600.0, places=9)
        self.assertAlmostEqual(eng.total_cost(), eng.cost("a"), places=12)


class TestStop(EngineCase):
    def test_stop_request_then_stopped(self):
        eng = self.engine()
        job = self.running_point(eng, "a", rows=50)
        self.assertEqual(eng.read("a").state, RUNNING)
        eng.stop("a")
        data = eng.read("a")
        self.assertEqual((data.state, data.nblocks), (STOPPED, 50))
        self.assertIn("after 50 of 200 blocks", data.message)
        self.assertTrue(data.meta["stop_requested"])
        for name in ("stop.requested", "dmc.STOP", "dmc.s001.STOP"):
            self.assertTrue(os.path.exists(self.path("a", name)), name)
        self.assertEqual(self.read_text("a", "stop.requested").strip(), "120")
        self.assertEqual(self.sched.cancelled, [])                   # QMCPACK gets the time to end its block
        # QMCPACK ends two more blocks: the data of the stopped point do not change
        self.fake_rows("a", 52)
        data = eng.read("a")
        self.assertEqual((data.state, data.nblocks, data.meta["blocks_on_disk"]), (STOPPED, 50, 52))
        self.now += 119
        eng.read("a")
        self.assertEqual(self.sched.cancelled, [])
        self.now += 2                                                # 121 s after the request: the grace time is over
        eng.submit(self.spec("b"))
        eng.read("b")                                                # any read does the late duty
        self.assertEqual(self.sched.cancelled, [job])
        eng.read("a")
        eng.advance()
        self.assertEqual(self.sched.cancelled, [job])
        self.assertEqual(eng.read("a").state, STOPPED)
        self.assertEqual(self.engine().read("a").nblocks, 50)        # also for a new process
        self.assertEqual(len(self.sched.submitted), 2)               # a stopped point is never submitted again

    def test_no_cancel_when_the_job_ends_in_time(self):
        eng = self.engine()
        job = self.running_point(eng, "a", rows=50)
        eng.stop("a")
        self.sched.state_of[job] = S.COMPLETED
        self.now += 500
        eng.advance()
        self.assertEqual(self.sched.cancelled, [])
        self.assertEqual(eng.read("a").state, STOPPED)

    def test_grace_time_follows_the_block_time(self):
        eng = self.engine()
        job = self.running_point(eng, "a", rows=1)
        rows = "".join("%d -17.2 297.0 204800 100.0 0.997\n" % i for i in range(10))            # 100 s per block
        self.write("a", "dmc.s001.scalar.dat", "#   index   LocalEnergy   LocalEnergy_sq   BlockWeight   BlockCPU   AcceptRatio\n" + rows)
        eng.stop("a")
        self.assertEqual(self.read_text("a", "stop.requested").strip(), "300")                  # 3 block times
        self.now += 290
        eng.advance()
        self.assertEqual(self.sched.cancelled, [])
        self.now += 11
        eng.advance()
        self.assertEqual(self.sched.cancelled, [job])
        # a new process continues the duty from the files
        eng2 = self.engine()
        job2 = self.running_point(eng2, "b", rows=20)
        eng2.stop("b")
        eng3 = self.engine()
        self.now += 121
        eng3.advance()
        self.assertEqual(self.sched.cancelled, [job, job2])

    def test_a_point_without_blocks_is_cancelled_at_once(self):
        eng = self.engine()
        eng.submit(self.spec("a"))
        job = self.sched.submitted[0]["id"]
        eng.stop("a")
        self.assertEqual(self.sched.cancelled, [job])
        data = eng.read("a")
        self.assertEqual((data.state, data.nblocks), (STOPPED, 0))
        self.assertFalse(os.path.exists(self.path("a", "dmc.STOP")))
        self.assertEqual(len(self.sched.submitted), 1)

    def test_stop_of_a_complete_or_unknown_point_does_nothing(self):
        eng = self.engine()
        eng.stop("never/submitted")
        job = self.running_point(eng, "a", rows=200)                 # all blocks on disk, dmc.done not yet
        eng.stop("a")
        self.assertFalse(os.path.exists(self.path("a", "stop.requested")))
        self.write("a", "dmc.done", "1320\n")
        self.sched.state_of[job] = S.COMPLETED
        self.assertEqual(eng.read("a").state, DONE)
        eng.stop("a")
        self.assertEqual(eng.read("a").state, DONE)
        self.assertEqual(self.sched.cancelled, [])


class TestFailures(EngineCase):
    def fail_in_opt(self, key, job):
        self.write(key, "opt.started", "1100\n")
        self.write(key, "opt.failed", "RUN: exit code 1: mpirun -np 32 --bind-to core qmcpack_real opt.in.xml "
                                      "[MPI start failure, see opt.err]\n")
        self.sched.state_of[job] = S.FAILED

    def test_failed_job_is_submitted_again_once(self):
        eng = self.engine()
        key = "min/it1/d0_p091"
        eng.submit(self.spec(key))
        first = self.sched.submitted[0]["id"]
        self.sched.state_of[first] = S.RUNNING
        self.fake_scf(key)
        self.mark(key, "c4q", 1060, 1070)
        self.fake_vmc0(key)
        self.fail_in_opt(key, first)
        data = eng.read(key)
        self.assertEqual(data.state, QUEUED)                         # submitted again
        self.assertEqual(len(self.sched.submitted), 2)
        second = self.sched.submitted[1]
        self.assertEqual(second["script"], self.sched.submitted[0]["script"])
        self.assertEqual(data.meta["attempts"], 2)
        self.assertEqual(data.meta["job_ids"], [first, second["id"]])
        self.assertIn("attempt 1 failed (step opt: RUN: exit code 1", data.meta["notes"][-1])
        self.assertFalse(os.path.exists(self.path(key, "opt.failed")))
        self.assertTrue(os.path.exists(self.path(key, "opt.failed.attempt1")))
        self.assertFalse(os.path.exists(self.path(key, "opt.started")))
        self.assertTrue(os.path.exists(self.path(key, "scf.done")))  # done steps are kept: the new job skips them
        self.assertEqual(eng.read(key).state, QUEUED)
        self.assertEqual(len(self.sched.submitted), 2)
        # the second attempt fails in the same way: now the point is FAILED, with the reason
        self.fail_in_opt(key, second["id"])
        data = eng.read(key)
        self.assertEqual(data.state, FAILED)
        for needle in ("step opt", "RUN: exit code 1", "MPI start failure", "job %s ended in state FAILED" % second["id"],
                       "attempt 2 of 2"):
            self.assertIn(needle, data.message)
        self.assertEqual(len(self.sched.submitted), 2)
        self.assertEqual(eng.read(key).state, FAILED)
        self.assertEqual(JobTable(os.path.join(self.root, "jobs.json")).attempts(key), 2)

    def test_retry_gives_a_failed_point_new_attempts(self):
        eng = self.engine()
        key = "min/it1/d0_p091"
        eng.submit(self.spec(key))
        first = self.sched.submitted[0]["id"]
        self.sched.state_of[first] = S.RUNNING
        self.fake_scf(key)
        self.mark(key, "c4q", 1060, 1070)
        self.fake_vmc0(key)
        self.fail_in_opt(key, first)
        eng.read(key)                                                # the engine submits it again by itself ...
        self.fail_in_opt(key, self.sched.submitted[1]["id"])
        self.assertEqual(eng.read(key).state, FAILED)                # ... and the second failure is final
        self.assertEqual(eng.failed_keys(), [key])
        self.assertFalse(eng.retry("no/such/point"))
        # a person repaired the cause and asks for a retry
        self.assertEqual(eng.retry_failed(), [key])
        self.assertEqual(len(self.sched.submitted), 3)
        data = eng.read(key)
        self.assertEqual((data.state, data.meta["attempts"]), (QUEUED, 1))
        self.assertTrue(any("submitted again by request" in note for note in data.meta["notes"]))
        self.assertTrue(os.path.exists(self.path(key, "opt.failed.retry1")))
        self.assertFalse(os.path.exists(self.path(key, "opt.failed")))
        self.assertFalse(os.path.exists(self.path(key, "opt.started")))
        self.assertTrue(os.path.exists(self.path(key, "scf.done")))  # done steps are kept
        self.assertEqual(eng.failed_keys(), [])
        self.assertFalse(eng.retry(key))                             # the point is not failed any more
        # the new attempt works
        third = self.sched.submitted[2]["id"]
        self.sched.state_of[third] = S.RUNNING
        self.fake_opt(key)
        self.fake_rows(key, 200)
        self.write(key, "dmc.done", "1500\n")
        self.sched.state_of[third] = S.COMPLETED
        self.assertEqual((eng.read(key).state, eng.read(key).nblocks), (DONE, 200))

    def test_retry_after_a_time_limit_sets_the_partial_trace_aside(self):
        eng = self.engine()
        job = self.running_point(eng, "a", rows=150)
        self.sched.state_of[job] = S.TIMEOUT
        self.assertEqual(eng.read("a").state, FAILED)
        self.assertTrue(eng.retry("a"))
        data = eng.read("a")
        self.assertEqual((data.state, data.nblocks), (QUEUED, 0))
        self.assertTrue(os.path.exists(self.path("a", "dmc.s001.scalar.dat.retry1")))
        self.assertTrue(os.path.exists(self.path("a", "opt.done")))
        # a point that waits for a failed point is tried after it
        eng.submit(self.spec("a_s2", series=2, base_key="a"))
        self.assertEqual(eng.failed_keys(), [])

    def test_a_job_that_vanished_without_a_reason(self):
        eng = self.engine(max_attempts=1)
        eng.submit(self.spec("a"))
        self.sched.state_of[self.sched.submitted[0]["id"]] = S.COMPLETED          # ended, but no step ran
        data = eng.read("a")
        self.assertEqual(data.state, FAILED)
        self.assertIn("step scf did not finish and left no reason", data.message)
        self.assertIn("attempt 1 of 1", data.message)
        self.assertEqual(len(self.sched.submitted), 1)

    def test_the_judgement_waits_for_the_settle_time(self):
        eng = self.engine(settle_seconds=60)
        eng.submit(self.spec("a"))
        first = self.sched.submitted[0]["id"]
        self.sched.state_of[first] = S.FAILED
        self.assertEqual(eng.read("a").state, QUEUED)
        self.now += 59
        self.assertEqual(eng.read("a").state, QUEUED)
        self.assertEqual(len(self.sched.submitted), 1)               # the marker files may be late on a shared disk
        self.now += 2
        eng.read("a")
        self.assertEqual(len(self.sched.submitted), 2)

    def test_a_job_unknown_to_the_batch_system_is_not_judged_at_once(self):
        eng = self.engine(unknown_seconds=600)
        job = self.running_point(eng, "a", rows=20)
        del self.sched.state_of[job]                                 # the batch system answers UNKNOWN
        self.assertEqual(eng.read("a").state, RUNNING)
        self.now += 300
        self.assertEqual(eng.read("a").state, RUNNING)
        self.sched.state_of[job] = S.RUNNING                         # it was a failed question, the job is there
        self.assertEqual(eng.read("a").state, RUNNING)
        self.now += 400
        self.assertEqual(eng.read("a").state, RUNNING)
        self.assertEqual(len(self.sched.submitted), 1)
        del self.sched.state_of[job]
        eng.read("a")
        self.now += 601
        eng.read("a")
        self.assertEqual(len(self.sched.submitted), 2)               # unknown for 10 minutes: submitted again

    def test_time_limit_of_the_job_during_dmc(self):
        eng = self.engine()
        job = self.running_point(eng, "a", rows=150)
        self.sched.state_of[job] = S.TIMEOUT
        data = eng.read("a")
        self.assertEqual((data.state, data.nblocks), (FAILED, 150))
        self.assertIn("time limit after 150 of 200 blocks", data.message)
        self.assertEqual(len(self.sched.submitted), 1)               # a second attempt would end in the same way

    def test_a_job_cancelled_by_hand_is_not_submitted_again(self):
        eng = self.engine()
        job = self.running_point(eng, "a", rows=20)
        self.sched.state_of[job] = S.CANCELLED
        data = eng.read("a")
        self.assertEqual(data.state, FAILED)
        self.assertIn("cancelled outside this program", data.message)
        self.assertEqual(len(self.sched.submitted), 1)

    def test_quality_failure_reported_by_the_job(self):
        eng = self.engine()
        eng.submit(self.spec("a"))
        job = self.sched.submitted[0]["id"]
        self.sched.state_of[job] = S.RUNNING
        self.fake_scf("a")
        self.mark("a", "c4q", 1060, 1070)
        self.write("a", "vmc0.started", "1070\n")
        self.write("a", "vmc0.failed", "QUALITY: zero-Jastrow VMC energy -68.011619 +/- 0.010000 Ha, energy of the "
                                       "determinant -88.183700 Ha\n")
        data = eng.read("a")
        self.assertEqual(data.state, FAILED)
        self.assertIn("step vmc0: zero-Jastrow VMC energy -68.011619", data.message)
        self.assertEqual(self.sched.cancelled, [job])
        self.assertEqual(len(self.sched.submitted), 1)               # a second attempt cannot repair a wrong result

    def test_zero_jastrow_check(self):
        eng = self.engine()
        eng.submit(self.spec("a"))
        job = self.sched.submitted[0]["id"]
        self.sched.state_of[job] = S.RUNNING
        self.fake_scf("a", e_det=-15.5)                              # not the determinant that the VMC run sampled
        self.mark("a", "c4q", 1060, 1070)
        self.fake_vmc0("a")
        data = eng.read("a")
        self.assertEqual(data.state, FAILED)
        self.assertIn("vmc0 check: zero-Jastrow VMC energy -16.975903", data.message)
        self.assertIn("energy of the determinant -15.500000 Ha", data.message)
        self.assertIn("does not describe the determinant of the SCF", data.message)
        self.assertFalse(data.meta["zero_jastrow"]["ok"])
        self.assertAlmostEqual(data.meta["zero_jastrow"]["difference"], -16.975903 + 15.5, places=5)
        self.assertAlmostEqual(data.meta["zero_jastrow"]["limit"], 4 * 0.007544 + 0.005, places=5)
        self.assertEqual(self.sched.cancelled, [job])
        self.assertEqual(len(self.sched.submitted), 1)

    def test_zero_jastrow_check_runs_for_the_first_point_only(self):
        eng = self.engine()
        eng.submit(self.spec("a"))
        eng.submit(self.spec("b"))
        self.assertEqual(self.read_json("a", "point.json")["steps"], ["scf", "c4q", "vmc0", "opt", "dmc"])
        self.assertEqual(self.read_json("b", "point.json")["steps"], ["scf", "c4q", "opt", "dmc"])
        self.assertNotIn("begin_step vmc0", self.sched.submitted[1]["text"])
        self.engine().submit(self.spec("c"))                         # a new process remembers the first point
        self.assertEqual(self.read_json("c", "point.json")["steps"], ["scf", "c4q", "opt", "dmc"])
        ammonia = PointSpec(key="d", symbols=["O", "H", "H"], coords=COORDS[[1, 0, 2]], blocks=10)
        eng.submit(ammonia)                                          # another atom order is another first point
        self.assertIn("vmc0", self.read_json("d", "point.json")["steps"])
        self.engine(vmc0_check="always").submit(self.spec("e"))
        self.assertIn("vmc0", self.read_json("e", "point.json")["steps"])
        root2 = self.root
        self.root = os.path.join(self.tmp, "other")
        self.engine(vmc0_check="never").submit(self.spec("a"))
        self.assertNotIn("vmc0", self.read_json("a", "point.json")["steps"])
        self.root = root2

    def test_pseudopotential_charge_mismatch(self):
        eng = self.engine()
        eng.submit(self.spec("a"))
        job = self.sched.submitted[0]["id"]
        self.sched.state_of[job] = S.RUNNING
        self.fake_scf("a", zeff={"O": 8})                            # the SCF ran with all electrons of oxygen
        data = eng.read("a")
        self.assertEqual(data.state, FAILED)
        self.assertIn("system check: element O: the pseudopotential file has zval 6 but the SCF used charge [8]", data.message)
        self.assertEqual(self.sched.cancelled, [job])

    def test_jastrow_with_numbers_that_are_not_finite(self):
        eng = self.engine(vmc0_check="never")
        eng.submit(self.spec("a"))
        self.sched.state_of[self.sched.submitted[0]["id"]] = S.RUNNING
        self.fake_scf("a")
        self.mark("a", "c4q", 1060, 1070)
        with open(JASTROW) as fh:
            jastrow = json.load(fh)
        jastrow["coefficients"]["uu"][3] = float("nan")
        self.fake_opt("a", jastrow=jastrow)
        data = eng.read("a")
        self.assertEqual(data.state, FAILED)
        self.assertIn("jastrow check: the Jastrow coefficients uu hold numbers that are not finite", data.message)
        # a cycle with an energy that is not finite
        eng.submit(self.spec("b"))
        self.sched.state_of[self.sched.submitted[1]["id"]] = S.RUNNING
        self.fake_scf("b")
        self.mark("b", "c4q", 1060, 1070)
        with open(JASTROW) as fh:
            jastrow = json.load(fh)
        jastrow["selection"]["cycles"][5]["energy"] = float("inf")
        self.fake_opt("b", jastrow=jastrow)
        self.assertIn("optimization cycle(s) [5]", eng.read("b").message)

    def test_jastrow_that_does_not_lower_the_energy(self):
        eng = self.engine(vmc0_check="never")
        eng.submit(self.spec("a"))
        self.sched.state_of[self.sched.submitted[0]["id"]] = S.RUNNING
        self.fake_scf("a", e_det=-17.30)                             # below the VMC energy -17.2063 of the Jastrow file
        self.mark("a", "c4q", 1060, 1070)
        self.fake_opt("a")
        data = eng.read("a")
        self.assertEqual(data.state, FAILED)
        self.assertIn("jastrow check: VMC energy with the Jastrow factor -17.206290", data.message)
        self.assertIn("does not lower the energy", data.message)

    def test_seed_reported_by_qmcpack(self):
        eng = self.engine()
        self.running_point(eng, "a", rows=10)
        seed = self.read_json("a", "point.json")["seeds"]["dmc"]
        lo, hi = E.printed_prime_range(seed, 32)
        self.write("a", "dmc.out", "  Offset for the random number seeds from input file (mod 1024): %d\n\n"
                                   "  Range of prime numbers to use as seeds over processors and threads = %d-%d\n" % (seed, lo, hi))
        data = eng.read("a")
        self.assertEqual(data.state, RUNNING)
        self.assertEqual(data.meta["prime_range"], [lo, hi])
        self.assertTrue(data.meta["checks"]["seed"]["ok"])
        job = self.running_point(eng, "b", rows=10)
        self.write("b", "dmc.out", "  Offset for the random number seeds based on time: 363\n")
        data = eng.read("b")
        self.assertEqual(data.state, FAILED)
        self.assertIn("seed check: QMCPACK used the seed offset 363 (from the clock)", data.message)
        self.assertIn(job, self.sched.cancelled)

    def test_unreadable_scf_energy(self):
        eng = self.engine()
        eng.submit(self.spec("a"))
        self.sched.state_of[self.sched.submitted[0]["id"]] = S.RUNNING
        self.fake_scf("a")
        self.write("a", "energy.dat", "")
        data = eng.read("a")
        self.assertEqual(data.state, FAILED)
        self.assertIn("energy.dat", data.message)
        self.assertIn("scf.done exists but the SCF energy cannot be read", data.message)


class TestSlowRun(EngineCase):
    """A run whose blocks take much longer than those of the other points of the molecule."""

    def slow_rows(self, key, n, factor=10.0):
        """The usual trace with the seconds per block multiplied by ``factor``."""
        lines = trace_lines()
        col = lines[0].split()[1:].index("BlockCPU")
        rows = []
        for ln in lines[1:n + 1]:
            parts = ln.split()
            parts[col] = "%.10e" % (float(parts[col]) * factor)
            rows.append("  ".join(parts))
        self.write(key, "dmc.s001.scalar.dat", "\n".join([lines[0]] + rows) + "\n")
        if not os.path.exists(self.path(key, "dmc.started")):
            self.write(key, "dmc.started", "1200\n")

    def others(self, eng, names=("a", "b", "c")):
        for name in names:
            self.running_point(eng, name, rows=20)
            self.assertEqual(eng.read(name).state, RUNNING)

    def test_a_slow_run_is_cancelled_and_submitted_again(self):
        eng = self.engine()
        self.others(eng)
        job = self.running_point(eng, "d", rows=20)
        self.slow_rows("d", 20)
        self.write("d", "job.info", "1000 job=%s nodes=n0001\n" % job)
        data = eng.read("d")
        self.assertEqual(self.sched.cancelled, [job])                 # cancelled by the engine ...
        self.assertIn("a block takes 10.0 times longer than in the other runs", data.meta["notes"][-1])
        self.assertIn("n0001", data.meta["notes"][-1])
        data = eng.read("d")                                         # ... and, once the job is gone, submitted again
        self.assertEqual(len(self.sched.submitted), 5)
        again = self.sched.submitted[-1]
        self.assertIn("#SBATCH --exclude=n0001", again["text"])
        self.assertNotIn("--exclude", self.sched.submitted[0]["text"])
        self.assertEqual((data.state, data.nblocks, data.meta["attempts"]), (QUEUED, 0, 2))
        self.assertIn("the run was 10.0 times slower than the other runs on n0001", data.meta["notes"][-1])
        self.assertTrue(os.path.exists(self.path("d", "dmc.s001.scalar.dat.attempt1")))   # the slow blocks are set aside
        self.assertFalse(os.path.exists(self.path("d", "dmc.s001.scalar.dat")))
        self.assertFalse(os.path.exists(self.path("d", "dmc.started")))
        self.assertTrue(os.path.exists(self.path("d", "opt.done")))                        # done steps are kept
        # the second attempt is not judged again: it runs to its end, whatever its speed
        self.sched.state_of[again["id"]] = S.RUNNING
        self.slow_rows("d", 30)
        data = eng.read("d")
        self.assertEqual((data.state, data.nblocks), (RUNNING, 30))
        self.assertEqual(self.sched.cancelled, [job])
        self.assertEqual(len(self.sched.submitted), 5)
        # the other points were not touched
        for name in ("a", "b", "c"):
            self.assertEqual(eng.read(name).state, RUNNING)

    def test_a_second_engine_finds_the_reference_in_the_files(self):
        """The driver job was started again: the speed of the earlier points is read from their traces."""
        eng = self.engine()
        self.others(eng)
        job = self.running_point(eng, "d", rows=20)
        self.slow_rows("d", 20)
        eng2 = self.engine()
        eng2.read("d")
        self.assertEqual(self.sched.cancelled, [job])

    def test_no_verdict_without_other_points_or_with_the_check_off(self):
        eng = self.engine()
        self.others(eng, names=("a", "b"))                           # 2 other points: the comparison needs 3
        self.running_point(eng, "d", rows=20)
        self.slow_rows("d", 20)
        self.assertEqual(eng.read("d").state, RUNNING)
        self.assertEqual(self.sched.cancelled, [])
        eng = self.engine(slow_factor=0)
        self.running_point(eng, "c", rows=20)
        eng.read("c")
        self.running_point(eng, "e", rows=20)
        self.slow_rows("e", 20)
        self.assertEqual(eng.read("e").state, RUNNING)
        self.assertEqual(self.sched.cancelled, [])
        # a run that is 2 times slower is not "slow": nodes of different types differ by that much
        eng = self.engine()
        self.running_point(eng, "f", rows=20)
        self.slow_rows("f", 20, factor=2.0)
        self.assertEqual(eng.read("f").state, RUNNING)
        self.assertEqual(self.sched.cancelled, [])
        # too few blocks to judge
        self.running_point(eng, "g", rows=3)
        self.slow_rows("g", 3)
        self.assertEqual(eng.read("g").state, RUNNING)
        self.assertEqual(self.sched.cancelled, [])

    def test_time_limit_of_a_slow_run_gives_one_more_attempt(self):
        eng = self.engine()
        self.others(eng)
        job = self.running_point(eng, "d", rows=3)
        self.assertEqual(eng.read("d").state, RUNNING)               # too few blocks: not judged yet
        self.slow_rows("d", 150)
        self.write("d", "job.info", "1000 job=%s nodes=n[0001-0002]\n" % job)
        self.sched.state_of[job] = S.TIMEOUT                         # the job reached its time limit
        data = eng.read("d")
        self.assertEqual((data.state, data.meta["attempts"]), (QUEUED, 2))
        self.assertEqual(len(self.sched.submitted), 5)
        self.assertIn("#SBATCH --exclude=n[0001-0002]", self.sched.submitted[-1]["text"])
        self.assertIn("10.0 times slower", data.meta["notes"][-1])
        # the second attempt is slow as well and reaches the limit: now the point fails, with the blocks kept
        second = self.sched.submitted[-1]["id"]
        self.sched.state_of[second] = S.RUNNING
        self.slow_rows("d", 150)
        self.sched.state_of[second] = S.TIMEOUT
        data = eng.read("d")
        self.assertEqual((data.state, data.nblocks), (FAILED, 150))
        self.assertIn("time limit after 150 of 200 blocks", data.message)
        self.assertEqual(len(self.sched.submitted), 5)


class TestSeries(EngineCase):
    def test_series_two_waits_for_the_base_point(self):
        eng = self.engine()
        base, second = "min/it1/centre", "min/it1/centre_s2"
        eng.submit(self.spec(base, role="centre"))
        eng.submit(self.spec(second, role="centre", series=2, base_key=base))
        self.assertEqual(len(self.sched.submitted), 1)               # only the base point has a job
        data = eng.read(second)
        self.assertEqual(data.state, QUEUED)
        self.assertIn("waits for the orbitals and the Jastrow factor of min/it1/centre", data.message)
        self.sched.state_of[self.sched.submitted[0]["id"]] = S.RUNNING
        self.fake_scf(base)
        self.assertEqual(eng.read(second).state, QUEUED)             # orbitals alone are not enough
        self.assertEqual(len(self.sched.submitted), 1)
        self.assertAlmostEqual(eng.read(second).e_surrogate, E_SCF, places=12)
        self.mark(base, "c4q", 1060, 1070)
        self.fake_vmc0(base)
        self.fake_opt(base)
        self.fake_rows(base, 30)
        eng.advance()                                                # the base point has orbitals and a Jastrow factor
        self.assertEqual(len(self.sched.submitted), 2)
        sub = self.sched.submitted[1]
        self.assertEqual(sub["script"], self.path(second, "job.sh"))
        text = sub["text"]
        self.assertIn("if begin_step dmc; then", text)
        for absent in ("begin_step scf", "begin_step c4q", "begin_step vmc0", "begin_step opt", "package write-dmc",
                       "scf.py"):
            self.assertNotIn(absent, text)
        self.assertIn("run_limited 7200 dmc2.out dmc2.err mpirun -np 32 --bind-to core qmcpack_real dmc2.in.xml", text)
        self.assertIn("dmc2.s001.scalar.dat holds $blocks of 200 blocks", text)
        for name in ("scf.py", "qmc_settings.json", "scf.h5"):
            self.assertFalse(os.path.exists(self.path(second, name)), name)
        # the DMC input: another project id, another seed, orbitals and Jastrow factor of the base point
        root = ET.parse(self.path(second, "dmc2.in.xml")).getroot()
        rec_base, rec_second = self.read_json(base, "point.json"), self.read_json(second, "point.json")
        self.assertEqual(root.find("project").get("id"), "dmc2")
        self.assertEqual(int(root.find("random").get("seed")), rec_second["seeds"]["dmc"])
        self.assertNotEqual(rec_second["seeds"]["dmc"], rec_base["seeds"]["dmc"])
        self.assertFalse(E.windows_overlap(rec_base["seed_windows"]["dmc"], rec_second["seed_windows"]["dmc"]))
        self.assertEqual(root.find(".//determinantset").get("href"), os.path.join("..", "centre", "scf.h5"))
        self.assertEqual([q.get("method") for q in root.iter("qmc")], ["vmc", "dmc"])
        dmc = {p.get("name"): p.text.strip() for p in root.findall("qmc")[1].findall("parameter")}
        self.assertEqual((int(dmc["blocks"]), int(dmc["steps"]), int(dmc["warmupSteps"])), (200, 100, 2000))
        with open(JASTROW) as fh:
            coeff = json.load(fh)["coefficients"]
        for c in root.iter("coefficients"):
            np.testing.assert_allclose([float(v) for v in c.text.split()], coeff[c.get("id")], rtol=1e-9)
        self.assertEqual([g.get("name") for g in root.findall(".//particleset[@name='ion0']/group")], ["H", "O"])
        self.assertEqual(sorted(p.get("href") for p in root.iter("pseudo")), ["H.ccECP.xml", "O.ccECP.xml"])
        # its own trace in its own folder
        self.sched.state_of[sub["id"]] = S.RUNNING
        self.fake_rows(second, 25, project="dmc2")
        data = eng.read(second)
        self.assertEqual((data.state, data.nblocks), (RUNNING, 25))
        self.assertEqual((data.meta["project"], data.meta["series"], data.meta["base_key"]), ("dmc2", 2, base))
        self.assertAlmostEqual(data.meta["vmc_energy"], -17.2062896255, places=9)
        self.assertAlmostEqual(data.e_surrogate, E_SCF, places=12)
        self.assertEqual(eng.read(base).nblocks, 30)
        self.assertEqual(JobTable(os.path.join(self.root, "jobs.json")).current(second)["id"], sub["id"])

    def test_series_rules(self):
        eng = self.engine()
        eng.submit(self.spec("c"))
        eng.submit(self.spec("c2", series=2, base_key="c"))
        with self.assertRaises(ValueError):                           # the same series twice under another key
            eng.submit(self.spec("c2b", series=2, base_key="c"))
        with self.assertRaises(ValueError):                           # another geometry
            eng.submit(PointSpec(key="c3", symbols=list(SYMBOLS), coords=COORDS + 0.01, blocks=10, series=3, base_key="c"))
        with self.assertRaises(ValueError):                           # a base point must be a series-1 point
            eng.submit(self.spec("c4", series=3, base_key="c2"))
        with self.assertRaises(ValueError) as ctx:                    # 32 ranks: 16 series have prime numbers of their own
            eng.submit(self.spec("c17", series=17, base_key="c"))
        self.assertIn("at most 16 runs", str(ctx.exception))
        eng.submit(self.spec("c3", series=3, base_key="c"))
        windows = [self.read_json(k, "point.json")["seed_windows"]["dmc"] for k in ("c", "c2", "c3")]
        self.assertFalse(any(E.windows_overlap(windows[i], windows[j]) for i in range(3) for j in range(i + 1, 3)))
        self.assertEqual(len(self.sched.submitted), 1)
        # A base run on 96 ranks uses 192 positions. A second series on 32 ranks would start only 64 positions
        # further and share primes with it: refused.
        wide = next(k for k in ("w", "x", "y", "z") if E.point_seeds(k, 1, 96)["dmc"] < 800)
        eng.submit(self.spec(wide, settings={"ranks": 96}))
        with self.assertRaises(ValueError) as ctx:
            eng.submit(self.spec(wide + "2", series=2, base_key=wide))
        self.assertIn("overlap", str(ctx.exception))
        eng.submit(self.spec(wide + "2", series=2, base_key=wide, settings={"ranks": 96}))
        both = [self.read_json(k, "point.json")["seed_windows"]["dmc"] for k in (wide, wide + "2")]
        self.assertFalse(E.windows_overlap(both[0], both[1]))

    def test_series_of_a_failed_base_point_fails(self):
        eng = self.engine(max_attempts=1)
        eng.submit(self.spec("c"))
        eng.submit(self.spec("c2", series=2, base_key="c"))
        self.sched.state_of[self.sched.submitted[0]["id"]] = S.FAILED
        self.write("c", "scf.started", "1000\n")
        self.write("c", "scf.failed", "RUN: exit code 1: python3 -u scf.py\n")
        data = eng.read("c2")
        self.assertEqual(data.state, FAILED)
        self.assertIn("the base point c ended without orbitals and a checked Jastrow factor", data.message)
        self.assertIn("step scf: RUN: exit code 1", data.message)
        self.assertEqual(len(self.sched.submitted), 1)

    def test_stop_of_a_waiting_series(self):
        eng = self.engine()
        eng.submit(self.spec("c"))
        eng.submit(self.spec("c2", series=2, base_key="c"))
        eng.stop("c2")
        self.assertEqual(eng.read("c2").state, STOPPED)
        self.fake_setup("c")
        eng.advance()
        self.assertEqual(len(self.sched.submitted), 1)               # the stopped series never starts


class TestSplitScf(EngineCase):
    def test_two_jobs(self):
        eng = self.engine(split_scf=True)
        eng.submit(self.spec("a"))
        self.assertEqual(len(self.sched.submitted), 2)
        scf, qmc = self.sched.submitted
        self.assertEqual(scf["script"], self.path("a", "job_scf.sh"))
        self.assertEqual(qmc["script"], self.path("a", "job.sh"))
        self.assertEqual(scf["after"], [])
        self.assertEqual(qmc["after"], [scf["id"]])                  # the QMC job starts after the SCF job
        for needle in ("#SBATCH --job-name=q_a_scf", "#SBATCH --cpus-per-task=4", "#SBATCH --ntasks=1",
                       "#SBATCH --time=00:30:00", "export OMP_NUM_THREADS=4", "job_scf.out",
                       "run_limited 1800 scf.out - python3 -u scf.py", "package check-system"):
            self.assertIn(needle, scf["text"])
        self.assertNotIn("begin_step c4q", scf["text"])
        self.assertNotIn("begin_step dmc", scf["text"])
        self.assertIn("#SBATCH --ntasks=32", qmc["text"])
        self.assertIn("if begin_step scf; then", qmc["text"])        # skipped at run time because scf.done exists
        self.assertIn("if begin_step dmc; then", qmc["text"])
        self.assertIn("lib.num_threads(4)", self.read_text("a", "scf.py"))
        table = JobTable(os.path.join(self.root, "jobs.json"))
        self.assertEqual([e["kind"] for e in table.jobs["a"]], ["scf", "qmc"])
        self.assertEqual(eng.read("a").state, QUEUED)
        # the SCF job fails: the QMC job would wait for ever, so it is cancelled, and both are submitted again
        self.sched.state_of[scf["id"]] = S.FAILED
        self.write("a", "scf.started", "1000\n")
        self.write("a", "scf.failed", "RUN: exit code 1: python3 -u scf.py\n")
        os.utime(self.path("a", "scf.failed"), (1030, 1030))         # the failed attempt ran for 30 s
        self.assertEqual(eng.read("a").state, QUEUED)
        self.assertEqual(self.sched.cancelled, [qmc["id"]])
        self.assertEqual(len(self.sched.submitted), 4)
        self.assertEqual(self.sched.submitted[3]["after"], [self.sched.submitted[2]["id"]])
        self.assertEqual(self.sched.submitted[2]["script"], self.path("a", "job_scf.sh"))
        # the second SCF job works; the QMC job runs
        self.sched.state_of[self.sched.submitted[2]["id"]] = S.COMPLETED
        self.sched.state_of[self.sched.submitted[3]["id"]] = S.RUNNING
        self.fake_setup("a")
        self.fake_rows("a", 12)
        data = eng.read("a")
        self.assertEqual((data.state, data.nblocks, data.meta["attempts"]), (RUNNING, 12, 2))
        # cost: the SCF step counts the 4 cores of its own job, the other steps the 32 ranks;
        # the 30 s of the failed SCF attempt are kept
        self.now = 1212.0
        self.assertAlmostEqual(eng.cost("a"), ((30 + 60) * 4 + (10 + 30 + 100 + 12) * 32) / 3600.0, places=9)


class TestJastrowReuse(EngineCase):
    def test_scan_point_starts_from_the_jastrow_of_the_centre(self):
        eng = self.engine(jastrow_reuse="centre", opt={"cycles": 8, "cycles_reuse": 3})
        eng.submit(self.spec("it1/centre", role="centre"))
        eng.submit(self.spec("it1/d0_p091", role="scan", settings={"jastrow_start_key": "it1/centre"}))
        self.assertEqual(len(self.sched.submitted), 1)               # the scan point waits
        data = eng.read("it1/d0_p091")
        self.assertEqual(data.state, QUEUED)
        self.assertIn("waits for the Jastrow factor of it1/centre", data.message)
        self.sched.state_of[self.sched.submitted[0]["id"]] = S.RUNNING
        self.fake_setup("it1/centre")
        self.assertEqual(eng.read("it1/d0_p091").state, QUEUED)      # this read started the job
        self.assertEqual(len(self.sched.submitted), 2)
        cfg = self.read_json("it1/d0_p091", "qmc_settings.json")
        self.assertEqual(cfg["jastrow_start"], "jastrow_start.xml")
        self.assertEqual(cfg["opt"]["cycles"], 3)
        self.assertEqual(self.read_json("it1/centre", "qmc_settings.json")["opt"]["cycles"], 8)
        start = Q.read_jastrow(self.path("it1/d0_p091", "jastrow_start.xml"))
        with open(JASTROW) as fh:
            coeff = json.load(fh)["coefficients"]
        self.assertEqual(sorted(start), sorted(coeff))
        for cid in coeff:
            np.testing.assert_allclose(start[cid], coeff[cid], rtol=1e-9)
        self.assertEqual(eng.read("it1/d0_p091").meta["jastrow_start"], "it1/centre")
        self.assertIn("begin_step opt", self.sched.submitted[1]["text"])          # it still optimizes, from that start

    def test_without_a_start_key_or_a_usable_start(self):
        eng = self.engine(jastrow_reuse="centre", max_attempts=1)
        eng.submit(self.spec("c", role="centre"))
        eng.submit(self.spec("s1", role="scan"))                     # no key: starts at once, from zero
        self.assertEqual(len(self.sched.submitted), 2)
        self.assertIn("names no jastrow_start_key", eng.read("s1").meta["notes"][0])
        with self.assertRaises(ValueError):
            eng.submit(self.spec("s2", role="scan", settings={"jastrow_start_key": "unknown"}))
        eng.submit(self.spec("s3", role="scan", settings={"jastrow_start_key": "c"}))
        self.assertEqual(len(self.sched.submitted), 2)
        self.sched.state_of[self.sched.submitted[0]["id"]] = S.FAILED            # the centre fails
        data = eng.read("s3")
        self.assertEqual(len(self.sched.submitted), 3)               # the scan point runs all the same, from zero
        self.assertNotIn("jastrow_start", self.read_json("s3", "qmc_settings.json"))
        self.assertIn("no usable Jastrow factor", eng.read("s3").meta["notes"][-1])
        self.assertIsNone(data.meta["jastrow_start"])


class TestCommandsOfTheJob(EngineCase):
    """The commands that job.sh runs in the point folder (``python -m montepass.engine.qmcpack ...``)."""

    def run_command(self, key, command):
        old = os.getcwd()
        os.chdir(os.path.join(self.root, *key.split("/")))
        out = io.StringIO()
        try:
            with contextlib.redirect_stdout(out):
                code = E.main([command])
        finally:
            os.chdir(old)
        return code, out.getvalue().strip().split("\n")[-1]

    def write_cycle(self, key, n, energy, variance, coeff=None):
        rng = np.random.default_rng(n)
        rows = ["#   index    LocalEnergy   LocalEnergy_sq   BlockWeight"]
        for i in range(40):
            e = energy + 0.001 * rng.normal()
            rows.append("%d %.10f %.10f 1000" % (i, e, variance + e * e))
        self.write(key, "opt.s%03d.scalar.dat" % n, "\n".join(rows) + "\n")
        if coeff is not None:
            body = "".join('<coefficients id="%s" type="Array"> %s </coefficients>\n' % (cid, " ".join("%.8f" % v for v in vals))
                           for cid, vals in coeff.items())
            self.write(key, "opt.s%03d.opt.xml" % n, "<qmcsystem>\n%s</qmcsystem>\n" % body)

    def test_check_system(self):
        eng = self.engine()
        eng.submit(self.spec("a"))
        self.fake_scf("a")
        code, last = self.run_command("a", "check-system")
        self.assertEqual(code, 0)
        self.assertIn("system.json agrees with the geometry", last)
        self.assertTrue(self.read_json("a", "check_system.json")["ok"])
        self.fake_scf("a", zeff={"O": 8})
        code, last = self.run_command("a", "check-system")
        self.assertEqual(code, E.QUALITY_EXIT)
        self.assertIn("element O: the pseudopotential file has zval 6", last)
        self.assertFalse(self.read_json("a", "check_system.json")["ok"])
        system = self.system("a")
        system["coords_bohr"][0][2] += 0.01                          # the SCF ran another geometry
        self.write("a", "system.json", json.dumps(system))
        code, last = self.run_command("a", "check-system")
        self.assertEqual(code, E.QUALITY_EXIT)
        self.assertIn("bohr away from its place in geom.xyz", last)
        system = self.system("a")
        system["nup"], system["ndn"] = 5, 3
        self.write("a", "system.json", json.dumps(system))
        self.assertIn("requested spin (2S) is 0", self.run_command("a", "check-system")[1])

    def test_inputs_and_checks(self):
        eng = self.engine(opt={"cycles": 4})
        eng.submit(self.spec("a", blocks=64))
        self.fake_scf("a")
        record = self.read_json("a", "point.json")
        # zero-Jastrow run: input and check
        self.assertEqual(self.run_command("a", "write-vmc0")[0], 0)
        root = ET.parse(self.path("a", "vmc0.in.xml")).getroot()
        self.assertEqual(int(root.find("random").get("seed")), record["seeds"]["vmc0"])
        self.assertTrue(all(float(v) == 0.0 for c in root.iter("coefficients") for v in c.text.split()))
        shutil.copyfile(VMC0, self.path("a", "vmc0.s000.scalar.dat"))
        code, last = self.run_command("a", "check-vmc0")
        self.assertEqual(code, 0)
        self.assertIn("zero-Jastrow VMC energy -16.975903 +/- 0.007544 Ha", last)
        self.fake_scf("a", e_det=-15.5)
        code, last = self.run_command("a", "check-vmc0")
        self.assertEqual(code, E.QUALITY_EXIT)
        self.assertIn("does not describe the determinant of the SCF", last)
        self.fake_scf("a")
        # optimization input
        self.assertEqual(self.run_command("a", "write-opt")[0], 0)
        root = ET.parse(self.path("a", "opt.in.xml")).getroot()
        self.assertEqual(int(root.find("random").get("seed")), record["seeds"]["opt"])
        self.assertEqual(root.find("loop").get("max"), "4")
        self.assertEqual([g.get("name") for g in root.findall(".//particleset[@name='ion0']/group")], ["H", "O"])
        # DMC input: no measured cycle yet
        code, last = self.run_command("a", "write-dmc")
        self.assertEqual(code, E.QUALITY_EXIT)
        self.assertIn("no optimization cycle with measured parameters", last)
        coeff = {"eH": [0.1] * 6, "eO": [0.2] * 6, "uu": [0.3] * 8, "ud": [0.4] * 8}
        self.write_cycle("a", 0, -16.98, 1.50, coeff={k: [0.5 * x for x in v] for k, v in coeff.items()})
        self.write_cycle("a", 1, -17.19, 0.40, coeff=coeff)
        self.write_cycle("a", 2, -17.21, 0.38, coeff={k: [2 * x for x in v] for k, v in coeff.items()})
        code, last = self.run_command("a", "write-dmc")
        self.assertEqual(code, 0)
        self.assertIn("VMC energy with the Jastrow factor -17.21", last)
        root = ET.parse(self.path("a", "dmc.in.xml")).getroot()
        self.assertEqual(root.find("project").get("id"), "dmc")
        self.assertEqual(int(root.find("random").get("seed")), record["seeds"]["dmc"])
        dmc = {p.get("name"): p.text.strip() for p in root.findall("qmc")[1].findall("parameter")}
        self.assertEqual((int(dmc["blocks"]), int(dmc["steps"]), float(dmc["timestep"])), (64, 100, 0.005))
        got = {c.get("id"): [float(v) for v in c.text.split()] for c in root.iter("coefficients")}
        np.testing.assert_allclose(got["uu"], coeff["uu"])           # the parameters that cycle 2 measured
        self.assertTrue(self.read_json("a", "check_jastrow.json")["ok"])
        self.assertEqual(self.read_json("a", "jastrow.json")["coefficients"]["eO"], coeff["eO"])
        # a Jastrow factor that does not lower the energy stops the job before the DMC run
        self.fake_scf("a", e_det=-17.5)
        code, last = self.run_command("a", "write-dmc")
        self.assertEqual(code, E.QUALITY_EXIT)
        self.assertIn("does not lower the energy", last)
        self.assertEqual(self.run_command("a", "no-such-command")[0], 2)

    def test_optimization_starts_from_another_jastrow(self):
        eng = self.engine(jastrow_reuse="centre", vmc0_check="never")
        eng.submit(self.spec("c", role="centre"))
        self.fake_setup("c")
        eng.submit(self.spec("s", role="scan", settings={"jastrow_start_key": "c"}))
        self.fake_scf("s")
        self.assertEqual(self.run_command("s", "write-opt")[0], 0)
        with open(JASTROW) as fh:
            coeff = json.load(fh)["coefficients"]
        got = {c.get("id"): [float(v) for v in c.text.split()] for c in ET.parse(self.path("s", "opt.in.xml")).getroot().iter("coefficients")}
        for cid in coeff:
            np.testing.assert_allclose(got[cid], coeff[cid], rtol=1e-9)


# ===================================================================================== the job script, run with bash
STUB_PYTHON = r'''#!/bin/bash
# stands in for python: the SCF script and the commands of the package
if [ "$1" = "-u" ]; then
    printf -- '-1.723711747822735418e+01\n0.000000000000000000e+00\n' > energy.dat
    cp stub_system.json system.json
    echo orbitals > scf.h5
    echo SCF_DONE
    exit 0
fi
case "$3" in
    check-system) echo "system.json agrees with the geometry" ;;
    write-vmc0) echo "<simulation/>" > vmc0.in.xml ;;
    check-vmc0)
        if [ -e QUALITY_vmc0 ]; then echo "zero-Jastrow VMC energy is 20 Ha above the energy of the determinant"; exit 3; fi
        echo "zero-Jastrow check passed" ;;
    write-opt) echo "<simulation/>" > opt.in.xml ;;
    write-dmc) cp "$STUB_DATA/h2o_jastrow.json" jastrow.json; echo "<simulation/>" > dmc.in.xml ;;
    *) echo "unknown command $3"; exit 2 ;;
esac
exit 0
'''

STUB_QMCPACK = r'''#!/bin/bash
# stands in for qmcpack: writes the files of the run named by the input file
project="${1%.in.xml}"
if [ -e "FAIL_$project" ]; then
    rm -f "FAIL_$project"
    echo "UCX ERROR: MPI_INIT has failed" >&2
    exit 1
fi
if [ -e "SLOW_$project" ]; then rm -f "SLOW_$project"; sleep 20; fi
case "$project" in
    vmc0) cp "$STUB_DATA/h2o_vmc0.s000.scalar.dat" vmc0.s000.scalar.dat ;;
    opt)
        for k in 0 1 2; do
            cp "$STUB_DATA/h2o_vmc0.s000.scalar.dat" "opt.s00$k.scalar.dat"
            echo "<opt/>" > "opt.s00$k.opt.xml"
        done ;;
    dmc*)
        n="$STUB_BLOCKS"
        if [ -e STOP_dmc ]; then n=20; echo 120 > stop.requested; fi
        head -n $((n + 1)) "$STUB_DATA/h2o_dmc.s001.scalar.dat" > "$project.s001.scalar.dat" ;;
esac
echo "QMCPACK execution completed successfully"
'''

STUB_CONVERT = r'''#!/bin/bash
echo "QMCGaussianParserBase::dump"
echo structure > c4q.structure.xml
echo wavefunction > c4q.wfj.xml
'''

STUB_MPIRUN = r'''#!/bin/bash
# mpirun -np N program arguments ...
shift 2
exec "$@"
'''

LOCAL_PROFILE = """
[machine]
name = workstation
scheduler = local

[setup.qmc]
lines =
    export STUB_DATA=%(data)s
    export STUB_BLOCKS=50

[executables]
python = %(bin)s/python
qmcpack = %(bin)s/qmcpack
convert4qmc = %(bin)s/convert4qmc

[launch]
qmc = %(bin)s/mpirun -np {ranks}

[pseudopotentials]
dir = %(pp)s

[job.qmc]
tasks = 4

[job.scf]
cpus_per_task = 2
"""


class BashScheduler(Scheduler):
    """Runs a submitted script with bash when the test asks for it."""

    def __init__(self):
        self.jobs = {}
        self._n = 0

    def submit(self, script, workdir, name="job", after=None):
        self._n += 1
        job_id = "B%d" % self._n
        self.jobs[job_id] = {"script": script, "workdir": workdir, "state": S.PENDING, "output": ""}
        return job_id

    def run(self, job_id):
        job = self.jobs[job_id]
        res = subprocess.run(["bash", job["script"]], cwd=job["workdir"], stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, universal_newlines=True, timeout=120)
        job["output"] = res.stdout
        job["state"] = S.COMPLETED if res.returncode == 0 else S.FAILED
        return res.returncode

    def state(self, job_id):
        return self.jobs[job_id]["state"] if job_id in self.jobs else S.UNKNOWN

    def cancel(self, job_id):
        if job_id in self.jobs and self.jobs[job_id]["state"] == S.PENDING:
            self.jobs[job_id]["state"] = S.CANCELLED


@unittest.skipUnless(all(shutil.which(tool) for tool in ("bash", "timeout", "awk", "grep")),
                     "bash, timeout, awk and grep are needed to run the job script")
class TestJobScriptWithStandInPrograms(EngineCase):
    def setUp(self):
        super().setUp()
        self.bin = os.path.join(self.tmp, "bin")
        os.makedirs(self.bin)
        for name, text in (("python", STUB_PYTHON), ("qmcpack", STUB_QMCPACK), ("convert4qmc", STUB_CONVERT),
                           ("mpirun", STUB_MPIRUN)):
            path = os.path.join(self.bin, name)
            with open(path, "w") as fh:
                fh.write(text)
            os.chmod(path, 0o755)
        with open(self.profile, "w") as fh:
            fh.write(LOCAL_PROFILE % {"data": DATA, "bin": self.bin, "pp": self.pp})
        self.sched = BashScheduler()

    def start(self, eng, key="a", blocks=50, flags=()):
        eng.submit(self.spec(key, blocks=blocks))
        self.write(key, "stub_system.json", json.dumps(self.system(key)))
        for flag in flags:
            self.write(key, flag, "")
        return sorted(self.sched.jobs)[-1]

    def test_whole_chain_and_a_second_start(self):
        eng = self.engine()
        job = self.start(eng)
        self.assertEqual(eng.read("a").state, QUEUED)
        self.assertEqual(self.sched.run(job), 0, self.sched.jobs[job]["output"])
        out = self.sched.jobs[job]["output"]
        started = [out.index("step %s: started" % step) for step in ("scf", "c4q", "vmc0", "opt", "dmc")]
        self.assertEqual(started, sorted(started))
        self.assertIn("all steps of this job are done", out)
        for step in ("scf", "c4q", "vmc0", "opt", "dmc"):
            self.assertTrue(os.path.exists(self.path("a", step + ".started")), step)
            self.assertTrue(os.path.exists(self.path("a", step + ".done")), step)
            self.assertFalse(os.path.exists(self.path("a", step + ".failed")), step)
            self.assertGreater(float(self.read_text("a", step + ".done")), 1.6e9)        # the time of the end
        # "local" outside a batch job; the number of the batch job when the tests themselves run inside one
        self.assertRegex(self.read_text("a", "job.info"), r"job=(local|\d+) nodes=")
        data = eng.read("a")
        self.assertEqual((data.state, data.nblocks), (DONE, 50))
        np.testing.assert_array_equal(data.energy, read_scalar(TRACE)["LocalEnergy"][:50])
        self.assertAlmostEqual(data.e_surrogate, E_SCF, places=12)
        self.assertTrue(data.meta["zero_jastrow"]["ok"])
        # the same script started again: every step is skipped, nothing is computed twice
        ends = {step: self.read_text("a", step + ".done") for step in ("scf", "c4q", "vmc0", "opt", "dmc")}
        res = subprocess.run(["bash", self.path("a", "job.sh")], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             universal_newlines=True, timeout=120)
        self.assertEqual(res.returncode, 0, res.stdout)
        self.assertEqual(res.stdout.count("done before, skipped"), 5)
        self.assertNotIn(": started", res.stdout)
        self.assertEqual(ends, {step: self.read_text("a", step + ".done") for step in ends})

    def test_run_failure_and_second_attempt(self):
        eng = self.engine()
        first = self.start(eng, flags=("FAIL_opt",))
        self.assertEqual(self.sched.run(first), 1)
        line = self.read_text("a", "opt.failed").strip()
        self.assertTrue(line.startswith("RUN: exit code 1: %s/mpirun -np 4 %s/qmcpack opt.in.xml" % (self.bin, self.bin)), line)
        self.assertIn("[MPI start failure, see opt.err]", line)
        self.assertFalse(os.path.exists(self.path("a", "opt.done")))
        self.assertFalse(os.path.exists(self.path("a", "dmc.started")))                 # the job stopped at the failed step
        self.assertEqual(eng.read("a").state, QUEUED)                                   # the engine submitted it again
        second = sorted(self.sched.jobs)[-1]
        self.assertNotEqual(second, first)
        self.assertEqual(self.sched.run(second), 0, self.sched.jobs[second]["output"])
        out = self.sched.jobs[second]["output"]
        self.assertEqual(out.count("done before, skipped"), 3)                          # scf, c4q and vmc0 are not repeated
        self.assertIn("step opt: started", out)
        data = eng.read("a")
        self.assertEqual((data.state, data.nblocks, data.meta["attempts"]), (DONE, 50, 2))

    def test_quality_failure_stops_the_job(self):
        eng = self.engine()
        job = self.start(eng, flags=("QUALITY_vmc0",))
        self.assertEqual(self.sched.run(job), 1)
        self.assertEqual(self.read_text("a", "vmc0.failed").strip(),
                         "QUALITY: zero-Jastrow VMC energy is 20 Ha above the energy of the determinant")
        self.assertFalse(os.path.exists(self.path("a", "opt.started")))
        data = eng.read("a")
        self.assertEqual(data.state, FAILED)
        self.assertIn("step vmc0: zero-Jastrow VMC energy is 20 Ha above", data.message)
        self.assertEqual(len(self.sched.jobs), 1)

    def test_time_limit_of_a_step(self):
        eng = self.engine(time_limits={"vmc0": 1})
        job = self.start(eng, flags=("SLOW_vmc0",))
        self.assertEqual(self.sched.run(job), 1)
        line = self.read_text("a", "vmc0.failed").strip()
        self.assertTrue(line.startswith("TIMEOUT: no end within 1 s: %s/mpirun" % self.bin), line)

    def test_too_few_blocks_and_missing_text(self):
        eng = self.engine()
        job = self.start(eng, blocks=60)                             # the stand-in program writes 50 blocks
        self.assertEqual(self.sched.run(job), 1)
        self.assertEqual(self.read_text("a", "dmc.failed").strip(), "RUN: dmc.s001.scalar.dat holds 50 of 60 blocks")
        self.assertFalse(os.path.exists(self.path("a", "dmc.done")))
        with open(os.path.join(self.bin, "convert4qmc"), "w") as fh:
            fh.write("#!/bin/bash\necho converted\necho x > c4q.structure.xml\necho x > c4q.wfj.xml\n")
        job = self.start(eng, key="b")
        self.assertEqual(self.sched.run(job), 1)
        self.assertEqual(self.read_text("b", "c4q.failed").strip(),
                         "RUN: the text 'QMCGaussianParserBase::dump' is not in c4q.out")

    def test_stop_request_during_and_before_a_step(self):
        eng = self.engine()
        job = self.start(eng, flags=("STOP_dmc",))                   # the stop request arrives during the DMC run
        self.assertEqual(self.sched.run(job), 0)
        self.assertEqual(self.read_text("a", "dmc.stopped").strip(), "20")
        self.assertFalse(os.path.exists(self.path("a", "dmc.done")))
        self.assertFalse(os.path.exists(self.path("a", "dmc.failed")))
        self.assertIn("step dmc: stopped on request after 20 blocks", self.sched.jobs[job]["output"])
        job = self.start(eng, key="b")
        self.write("b", "stop.requested", "120\n")                   # the request is there before the job starts
        self.assertEqual(self.sched.run(job), 0)
        self.assertIn("step scf: not started, a stop was requested", self.sched.jobs[job]["output"])
        self.assertFalse(os.path.exists(self.path("b", "scf.started")))


if __name__ == "__main__":
    unittest.main()
