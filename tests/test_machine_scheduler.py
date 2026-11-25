"""Machine profiles, the local scheduler and the job table."""
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from montepass.machine import Machine, find_profile  # noqa: E402
from montepass.scheduler import COMPLETED, FAILED, CANCELLED, JobTable  # noqa: E402
from montepass.scheduler.local import LocalScheduler  # noqa: E402
from montepass.scheduler.slurm import parse_state  # noqa: E402


class TestMachine(unittest.TestCase):
    def setUp(self):
        self.m = Machine(os.path.join(ROOT, "machines", "example_slurm.ini"))

    def test_values(self):
        self.assertEqual(self.m.name, "example-slurm")
        self.assertEqual(self.m.tasks("qmc"), 32); self.assertEqual(self.m.threads("scf"), 4)
        self.assertEqual(self.m.launch("qmc"), "mpirun -np 32 --map-by core --bind-to core")
        self.assertEqual(self.m.launch("qmc", ranks=16), "mpirun -np 16 --map-by core --bind-to core")
        self.assertEqual(self.m.job("qmc")["account"], "myaccount")          # common value inherited
        self.assertIn("module load openmpi hdf5 boost", self.m.setup_lines("qmc"))

    def test_header(self):
        h = self.m.header("qmc", "test", "/work/x", time="00:30:00")
        for needle in ("#SBATCH --job-name=test", "#SBATCH --chdir=/work/x", "#SBATCH --ntasks=32", "#SBATCH --time=00:30:00",
                       "#SBATCH --account=myaccount", "#SBATCH --no-requeue", "export OMP_NUM_THREADS=1",
                       "export MKL_NUM_THREADS=1", "exit 96", "PYTHONPATH"):
            self.assertIn(needle, h)
        self.assertIn("export OMP_NUM_THREADS=4", self.m.header("scf", "s", "/work/y"))

    def test_find_profile_and_check(self):
        self.assertTrue(find_profile("example_slurm").endswith("example_slurm.ini"))
        with self.assertRaises(FileNotFoundError):
            find_profile("no_such_machine")
        self.assertTrue(any("pseudopotential folder" in p for p in self.m.check()))     # the example path does not exist


class TestLocalSchedulerAndTable(unittest.TestCase):
    def test_run_dependency_and_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            def script(name, body):
                p = os.path.join(tmp, name)
                with open(p, "w") as fh:
                    fh.write("#!/bin/bash\n" + body + "\n")
                return p
            s = LocalScheduler(max_running=1)
            a = s.submit(script("a.sh", "echo one > a.txt"), tmp, "a")
            b = s.submit(script("b.sh", "cat a.txt > b.txt"), tmp, "b", after=[a])
            c = s.submit(script("c.sh", "exit 3"), tmp, "c")
            d = s.submit(script("d.sh", "echo never > d.txt"), tmp, "d", after=[c])
            s.wait_all()
            self.assertEqual(s.state(a), COMPLETED); self.assertEqual(s.state(b), COMPLETED)
            self.assertEqual(s.state(c), FAILED); self.assertEqual(s.state(d), CANCELLED)
            with open(os.path.join(tmp, "b.txt")) as fh:
                self.assertEqual(fh.read().strip(), "one")
            self.assertFalse(os.path.exists(os.path.join(tmp, "d.txt")))

    def test_job_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "jobs.json")
            t = JobTable(p); t.add("min/it1/centre", "101", "/w/a", "c", "qmc"); t.add("min/it1/centre", "102", "/w/a", "c", "qmc")
            t2 = JobTable(p)
            self.assertEqual(t2.current("min/it1/centre")["id"], "102"); self.assertEqual(t2.attempts("min/it1/centre"), 2)
            self.assertIsNone(t2.current("other")); self.assertEqual(t2.all_ids(), ["101", "102"])

    def test_slurm_state_words(self):
        self.assertEqual(parse_state("COMPLETED"), COMPLETED); self.assertEqual(parse_state("CANCELLED by 1234"), CANCELLED)
        self.assertEqual(parse_state("OUT_OF_MEMORY"), FAILED); self.assertEqual(parse_state("TIMEOUT"), "TIMEOUT")
        self.assertEqual(parse_state(""), "UNKNOWN")


if __name__ == "__main__":
    unittest.main()
