"""ORCA input writers, output readers and jobs.

The files ``tests/data/orca_*`` are real ORCA results for ammonia (B3LYP/def2-SVP): Hessian files of
ORCA 5.0.3 and 6.1.0, the path and saddle files of an NEB-TS job, and excerpts of outputs (exact
lines of the outputs; a marker line stands for the lines that were left out).
"""
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(HERE, "data")
sys.path.insert(0, os.path.join(ROOT, "src"))

from montepass.geometry import read_xyz  # noqa: E402
from montepass.machine import Machine  # noqa: E402
from montepass.scheduler import base as sched  # noqa: E402
from montepass.scheduler.base import JobTable, Scheduler  # noqa: E402
from montepass.scheduler.local import LocalScheduler  # noqa: E402
from montepass.surrogate import orca  # noqa: E402
from montepass.surrogate.orca import ABSENT, DONE, FAILED, PENDING, RUNNING, OrcaJob  # noqa: E402
from montepass.units import BOHR_TO_ANGSTROM, HARTREE_TO_KCAL  # noqa: E402

NH3 = ["N", "H", "H", "H"]
PYRAMID = np.array([[0.0, 0.0, 0.1173], [0.0, 0.9377, -0.2737], [0.8121, -0.4689, -0.2737], [-0.8121, -0.4689, -0.2737]])


def data(name):
    return os.path.join(DATA, name)


def text_of(name):
    with open(data(name)) as fh:
        return fh.read()


class FakeScheduler(Scheduler):
    """Records the submissions; the test sets the state that the batch system reports."""

    def __init__(self):
        self.submitted = []
        self.cancelled = []
        self.answer = sched.PENDING

    def submit(self, script, workdir, name="job", after=None):
        self.submitted.append({"script": script, "workdir": workdir, "name": name, "after": after})
        return "F%d" % len(self.submitted)

    def state(self, job_id):
        return self.answer

    def cancel(self, job_id):
        self.cancelled.append(job_id)


class TestInputs(unittest.TestCase):
    """The writers must reproduce inputs that ran with ORCA 5.0.3 (PBE0/cc-pVTZ)."""

    def test_optimization_of_a_minimum(self):
        self.assertEqual(orca.optimization_input(xyz_file="start.xyz", method="PBE0", nprocs=1),
                         "! PBE0 cc-pVTZ TightOpt TightSCF\n%maxcore 3000\n%pal nprocs 1 end\n* xyzfile 0 1 start.xyz\n")
        self.assertEqual(orca.optimization_input(xyz_file="sym.xyz", method="PBE0", nprocs=8, frequencies=True),
                         "! PBE0 cc-pVTZ TightOpt TightSCF Freq\n%maxcore 3000\n%pal nprocs 8 end\n* xyzfile 0 1 sym.xyz\n")
        self.assertIn("! B3LYP cc-pVTZ Opt TightSCF\n", orca.optimization_input(xyz_file="a.xyz", tight=False))

    def test_frequencies(self):
        self.assertEqual(orca.frequency_input(xyz_file="minimum.xyz", method="PBE0", nprocs=8),
                         "! PBE0 cc-pVTZ TightSCF Freq\n%maxcore 3000\n%pal nprocs 8 end\n* xyzfile 0 1 minimum.xyz\n")
        self.assertIn(" NumFreq\n", orca.frequency_input(xyz_file="a.xyz", numerical=True))

    def test_saddle_optimization(self):
        text = orca.optimization_input(xyz_file="saddle_start.xyz", method="PBE0", nprocs=8, saddle=True, frequencies=True,
                                       tight=False, maxcore=None)
        self.assertEqual(text, "! PBE0 cc-pVTZ TightSCF OptTS Freq\n%pal nprocs 8 end\n%geom\n  Calc_Hess true\nend\n"
                               "* xyzfile 0 1 saddle_start.xyz\n")
        text = orca.optimization_input(xyz_file="a.xyz", saddle=True, recalc_hess=5)
        self.assertIn("! B3LYP cc-pVTZ TightSCF OptTS TightOpt\n", text)            # ran with ORCA 6.1.0
        self.assertIn("%geom\n  Calc_Hess true\n  Recalc_Hess 5\nend\n", text)
        self.assertNotIn("%geom", orca.optimization_input(xyz_file="a.xyz", saddle=True, calc_hess=False))

    def test_neb(self):
        text = orca.neb_input("product.xyz", xyz_file="reactant.xyz", ts_xyz_file="saddle_guess.xyz", images=5, method="PBE0", nprocs=10)
        self.assertEqual(text, '! PBE0 cc-pVTZ TightSCF NEB-TS\n%maxcore 3000\n%pal nprocs 10 end\n%neb\n'
                               '  NEB_END_XYZFILE "product.xyz"\n  NEB_TS_XYZFILE "saddle_guess.xyz"\n  NImages 5\n'
                               '  Preopt false\nend\n* xyzfile 0 1 reactant.xyz\n')
        text = orca.neb_input("b.xyz", xyz_file="a.xyz", kind="neb-ci", images=8, preopt=True, keys="orca6")
        self.assertIn("! B3LYP cc-pVTZ TightSCF NEB-CI\n", text)
        self.assertIn('%neb\n  Product "b.xyz"\n  NImages 8\n  PreOpt true\nend\n', text)
        text = orca.neb_input("b.xyz", xyz_file="a.xyz", ts_xyz_file="t.xyz",
                              keys={"end": "E_KEY", "ts": "T_KEY", "images": "N_KEY", "preopt": "P_KEY"})
        for needle in ('  E_KEY "b.xyz"\n', '  T_KEY "t.xyz"\n', "  N_KEY 8\n", "  P_KEY false\n"):
            self.assertIn(needle, text)

    def test_geometry_in_the_input_and_extras(self):
        text = orca.single_point_input(symbols=["o", "H", "h"], coords=[[0, 0, 0.1173], [0, 0.7572, -0.4692], [0, -0.7572, -0.4692]],
                                       charge=1, multiplicity=2, nprocs=2, maxcore=1500, extra_keywords=["RIJCOSX", "D3BJ"],
                                       extra_blocks="%scf\n  MaxIter 300\nend")
        lines = text.splitlines()
        self.assertEqual(lines[0], "! B3LYP cc-pVTZ TightSCF RIJCOSX D3BJ")
        self.assertEqual(lines[1:6], ["%maxcore 1500", "%pal nprocs 2 end", "%scf", "  MaxIter 300", "end"])
        self.assertEqual(lines[6], "* xyz 1 2")
        self.assertEqual(lines[7].split(), ["O", "0.0000000000", "0.0000000000", "0.1173000000"])
        self.assertEqual(lines[10], "*")
        self.assertTrue(text.endswith("*\n"))

    def test_wrong_arguments(self):
        with self.assertRaises(ValueError):
            orca.neb_input("b.xyz", xyz_file="a.xyz", kind="NEB-XY")
        with self.assertRaises(ValueError):
            orca.neb_input("b.xyz", xyz_file="a.xyz", images=0)
        with self.assertRaises(ValueError):
            orca.neb_input("b.xyz", xyz_file="a.xyz", keys="orca4")
        with self.assertRaises(ValueError):
            orca.neb_input("b.xyz", xyz_file="a.xyz", keys={"end": "X"})
        with self.assertRaises(ValueError):
            orca.neb_input("my file.xyz", xyz_file="a.xyz")
        with self.assertRaises(ValueError):
            orca.optimization_input(xyz_file="a.xyz", symbols=NH3, coords=PYRAMID)
        with self.assertRaises(ValueError):
            orca.optimization_input()
        with self.assertRaises(ValueError):
            orca.optimization_input(symbols=NH3, coords=PYRAMID[:3])
        with self.assertRaises(ValueError):
            orca.frequency_input(xyz_file="a.xyz", multiplicity=0)
        with self.assertRaises(ValueError):
            orca.frequency_input(xyz_file="a.xyz", nprocs=0)
        with self.assertRaises(ValueError):
            orca.optimization_input(symbols=["Xx", "H"], coords=np.zeros((2, 3)))


class TestOutputText(unittest.TestCase):
    def test_optimization_with_frequencies(self):
        t = text_of("orca_610_nh3_opt.out")
        self.assertEqual(orca.program_version(t), "6.1.0")
        self.assertTrue(orca.terminated_normally(t))
        self.assertTrue(orca.optimization_converged(t))
        self.assertEqual(orca.final_energy(t), -56.473098845005)          # the last of the energies in the file
        self.assertFalse(orca.neb_path_converged(t)); self.assertFalse(orca.neb_saddle_converged(t))
        self.assertEqual(orca.error_lines(t), [])

    def test_orca_5_frequencies(self):
        t = text_of("orca_503_nh3_freq.out")
        self.assertEqual(orca.program_version(t), "5.0.3")
        self.assertTrue(orca.terminated_normally(t))
        self.assertEqual(orca.final_energy(t), -56.473099141148)
        self.assertIn("THE CP-SCF HAS CONVERGED", t)                        # must not count as an optimization
        self.assertFalse(orca.optimization_converged(t))

    def test_neb_ts_two_separate_results(self):
        t = text_of("orca_610_nh3_neb.out")
        self.assertTrue(orca.neb_path_converged(t)); self.assertTrue(orca.neb_saddle_converged(t))
        self.assertTrue(orca.terminated_normally(t))
        self.assertEqual(orca.final_energy(t), -56.463646825546)
        # a run that was stopped inside the saddle optimization: the path is converged, the saddle is not
        t = text_of("orca_610_nh3_neb_stopped.out")
        self.assertTrue(orca.neb_path_converged(t)); self.assertFalse(orca.neb_saddle_converged(t))
        self.assertFalse(orca.terminated_normally(t))

    def test_optimization_that_did_not_converge(self):
        t = text_of("orca_610_nh3_not_converged.out")
        self.assertTrue(orca.terminated_normally(t))                        # ORCA ends normally all the same
        self.assertFalse(orca.optimization_converged(t))
        self.assertTrue(any("did not converge" in ln for ln in orca.error_lines(t)))

    def test_missing_values_are_errors(self):
        with self.assertRaises(ValueError):
            orca.final_energy("no energy here")
        self.assertIsNone(orca.program_version("no banner"))
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                orca.read_output(os.path.join(tmp, "none.out"))
            empty = os.path.join(tmp, "empty.out")
            open(empty, "w").close()
            with self.assertRaises(ValueError):
                orca.read_output(empty)


class TestPathAndSaddle(unittest.TestCase):
    def test_path_of_orca_6(self):
        symbols, frames, energies = orca.read_path(data("orca_610_nh3_neb_MEP_trj.xyz"))
        self.assertEqual(symbols, NH3)
        self.assertEqual(len(frames), 7)                                   # 5 images and 2 end points
        self.assertEqual(frames[3].shape, (4, 3))
        self.assertEqual(energies[0], -56.473099016534); self.assertEqual(energies[3], -56.463516088937)
        self.assertEqual(int(np.argmax(energies)), 3)
        barrier = (energies.max() - energies[0]) * HARTREE_TO_KCAL
        self.assertAlmostEqual(barrier, 6.01, places=2)                     # inversion barrier of ammonia on the path
        self.assertAlmostEqual(frames[0][1][1], 0.937817, places=5)

    def test_frame_without_energy(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "x_MEP_trj.xyz")
            with open(p, "w") as fh:
                fh.write("1\nCoordinates from ORCA-job x_MEP E -1.5\nH 0 0 0\n1\nno energy\nH 0 0 1\n")
            with self.assertRaises(ValueError) as cm:
                orca.read_path(p)
            self.assertIn("frame 1", str(cm.exception))
            with self.assertRaises(FileNotFoundError):
                orca.read_path(os.path.join(tmp, "none.xyz"))

    def test_saddle_files(self):
        path, kind = orca.saddle_file(DATA, "orca_610_nh3_neb")
        self.assertEqual(kind, "TS"); self.assertTrue(path.endswith("orca_610_nh3_neb_NEB-TS_converged.xyz"))
        symbols, coords, kind = orca.read_saddle(DATA, "orca_610_nh3_neb")
        self.assertEqual(symbols, NH3)
        self.assertLess(np.abs(coords[:, 2]).max(), 2e-5)                   # the saddle of the inversion is planar
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(orca.saddle_file(tmp, "neb"), (None, "NONE"))
            with self.assertRaises(FileNotFoundError):
                orca.read_saddle(tmp, "neb")
            shutil.copy(data("orca_610_nh3_neb_NEB-CI_converged.xyz"), os.path.join(tmp, "neb_NEB-CI_converged.xyz"))
            self.assertEqual(orca.read_saddle(tmp, "neb")[2], "CI")         # only the climbing image exists
            open(os.path.join(tmp, "neb_NEB-TS_converged.xyz"), "w").close()
            self.assertEqual(orca.saddle_file(tmp, "neb")[1], "CI")         # an empty file is not a result


class TestHessianFile(unittest.TestCase):
    def test_both_versions_against_their_own_frequencies(self):
        """Mass-weight the parsed matrix, diagonalize, and compare with the frequencies of the same file."""
        for name, lowest in (("orca_503_nh3_min.hess", 1061.88), ("orca_610_nh3_min.hess", 1062.14),
                             ("orca_610_nh3_saddle.hess", -830.18)):
            h = orca.read_hess(data(name))
            self.assertEqual(h.symbols, NH3)
            self.assertEqual(h.hessian.shape, (12, 12)); self.assertEqual(h.coords.shape, (4, 3))
            self.assertEqual(len(h.frequencies), 6)
            self.assertAlmostEqual(h.frequencies[0], lowest, places=2)
            self.assertEqual(float(np.abs(h.hessian - h.hessian.T).max()), 0.0)
            computed = orca.harmonic_frequencies(h.hessian, h.masses, h.coords)
            self.assertLess(np.abs(computed - h.frequencies).max(), 1.0, name)          # 1/cm
            self.assertAlmostEqual(float(h.masses[0]), 14.007); self.assertAlmostEqual(float(h.masses[1]), 1.008)
            self.assertAlmostEqual(float(np.linalg.norm(h.coords[1] - h.coords[0])), 1.02, places=1)

    def test_units(self):
        h = orca.read_hess(data("orca_610_nh3_min.hess"))
        # first number of the file: 6.1421125418E-01 hartree/bohr^2
        self.assertAlmostEqual(h.hessian[0, 0], 6.1421125418e-01 * HARTREE_TO_KCAL / BOHR_TO_ANGSTROM ** 2, places=6)
        self.assertAlmostEqual(h.hessian[0, 0], 1376.36, places=1)

    def test_imaginary_frequencies(self):
        minimum = orca.read_hess(data("orca_610_nh3_min.hess"))
        saddle = orca.read_hess(data("orca_610_nh3_saddle.hess"))
        self.assertEqual(minimum.n_imaginary(), 0); self.assertEqual(saddle.n_imaginary(), 1)
        self.assertEqual(minimum.problem("minimum"), ""); self.assertEqual(saddle.problem("saddle"), "")
        self.assertIn("found 1", saddle.problem("minimum")); self.assertIn("-830.2", saddle.problem("minimum"))
        self.assertIn("found 0", minimum.problem("saddle"))
        self.assertEqual(orca.count_imaginary([-30.0, -2.0, 100.0]), 2)
        self.assertEqual(orca.count_imaginary([-30.0, -2.0, 100.0], tolerance=5.0), 1)
        self.assertEqual(orca.stationary_point_problem([-30.0, -2.0, 100.0], "saddle", tolerance=5.0), "")
        with self.assertRaises(ValueError):
            orca.stationary_point_problem([1.0], "maximum")

    def test_version_differences(self):
        h5 = orca.read_hess(data("orca_503_nh3_min.hess")); h6 = orca.read_hess(data("orca_610_nh3_min.hess"))
        symbols, xyz = read_xyz(data("orca_610_nh3_min.xyz"))               # the geometry both jobs were given
        self.assertLess(np.abs(h5.coords - xyz).max(), 1e-6)                # 5.0.3: the positions of the input
        shift = h6.coords - xyz                                             # 6.1.0: centre of mass at the origin
        self.assertGreater(np.abs(shift).max(), 0.05)
        self.assertLess(np.abs(shift - shift.mean(axis=0)).max(), 1e-6)
        centre = (h6.masses[:, None] * h6.coords).sum(axis=0) / h6.masses.sum()
        self.assertLess(np.abs(centre).max(), 1e-6)
        self.assertGreater(h5.asymmetry, 1e-7); self.assertEqual(h6.asymmetry, 0.0)
        # same geometry, same method: the two programs agree to 0.4 1/cm and 0.1 % of the largest element
        self.assertLess(np.abs(h5.frequencies - h6.frequencies).max(), 0.4)
        self.assertLess(np.abs(h5.hessian - h6.hessian).max(), 1e-3 * np.abs(h6.hessian).max())

    def test_hessian_in_another_frame(self):
        h = orca.read_hess(data("orca_610_nh3_min.hess"))
        symbols, xyz = read_xyz(data("orca_610_nh3_min.xyz"))
        same = h.in_frame_of(xyz)                                           # a shift only: the matrix stays
        self.assertLess(np.abs(same - h.hessian).max(), 1e-3)
        a = np.radians(40.0)
        rot = np.array([[np.cos(a), np.sin(a), 0.0], [-np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]]) @ \
            np.array([[1.0, 0.0, 0.0], [0.0, np.cos(1.0), np.sin(1.0)], [0.0, -np.sin(1.0), np.cos(1.0)]])
        moved = np.dot(xyz, rot) + np.array([0.3, -1.0, 2.0])
        turned = h.in_frame_of(moved)
        self.assertGreater(np.abs(turned - h.hessian).max(), 100.0)
        rng = np.random.RandomState(3)
        d = rng.normal(size=(4, 3))                                         # a displacement in the frame of the file
        d_new = np.dot(d, rot)                                              # the same displacement in the new frame
        e_old = float(np.dot(d.reshape(-1), np.dot(h.hessian, d.reshape(-1))))
        e_new = float(np.dot(d_new.reshape(-1), np.dot(turned, d_new.reshape(-1))))
        self.assertAlmostEqual(e_old, e_new, places=4)
        np.testing.assert_allclose(np.linalg.eigvalsh(turned), np.linalg.eigvalsh(h.hessian), atol=1e-6)
        self.assertLess(np.abs(orca.harmonic_frequencies(turned, h.masses, moved) - h.frequencies).max(), 1.0)
        other = xyz.copy(); other[1, 1] += 0.2
        with self.assertRaises(ValueError):
            h.in_frame_of(other)
        with self.assertRaises(ValueError):
            h.in_frame_of(xyz[:3])

    def _write_small(self, tmp, hessian_lines=None, rigid="0.000000", drop=None):
        """A Hessian file written by hand: 2 atoms, element [i][j] = 10 i + j (not symmetric)."""
        lines = ["", "$orca_hessian_file", "", "$act_atom", "  0", "", "$hessian", "6"]
        if hessian_lines is None:
            hessian_lines = ["                  0          1          2          3          4"]
            hessian_lines += ["    %d  " % i + "  ".join("%.10E" % (10 * i + j) for j in range(5)) for i in range(6)]
            hessian_lines += ["                  5"]
            hessian_lines += ["    %d  %.10E" % (i, 10 * i + 5) for i in range(6)]
        lines += hessian_lines
        lines += ["", "$vibrational_frequencies", "6"] + ["    %d     %s" % (i, rigid) for i in range(5)] + ["    5     4401.210000"]
        lines += ["", "#", "# The atoms: label  mass x y z (in bohrs)", "#", "$atoms", "2",
                  " H      1.00800      0.000000000000     0.000000000000     0.000000000000",
                  " F     18.99800      0.000000000000     0.000000000000     1.732000000000", "", "$end", ""]
        text = "\n".join(lines)
        if drop:
            text = text.replace(drop, "$dropped")
        p = os.path.join(tmp, "small.hess")
        with open(p, "w") as fh:
            fh.write(text)
        return p

    def test_index_order_and_symmetrization(self):
        with tempfile.TemporaryDirectory() as tmp:
            h = orca.read_hess(self._write_small(tmp))
            raw = np.array([[10.0 * i + j for j in range(6)] for i in range(6)])
            factor = HARTREE_TO_KCAL / BOHR_TO_ANGSTROM ** 2
            np.testing.assert_allclose(h.hessian, 0.5 * (raw + raw.T) * factor, rtol=1e-12)
            self.assertAlmostEqual(h.asymmetry, 45.0)                       # |H[0][5] - H[5][0]| = |5 - 50|
            self.assertEqual(h.symbols, ["H", "F"])
            self.assertAlmostEqual(h.coords[1, 2], 1.732 * BOHR_TO_ANGSTROM)
            np.testing.assert_allclose(h.frequencies, [4401.21])            # a linear molecule: 5 rigid motions

    def test_broken_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                orca.read_hess(os.path.join(tmp, "none.hess"))
            for drop in ("$atoms", "$hessian", "$vibrational_frequencies"):
                with self.assertRaises(ValueError) as cm:
                    orca.read_hess(self._write_small(tmp, drop=drop))
                self.assertIn(drop, str(cm.exception)); self.assertIn("small.hess", str(cm.exception))
            cut = ["                  0          1          2          3          4"]
            cut += ["    %d  " % i + "  ".join("%.10E" % (10 * i + j) for j in range(5)) for i in range(6)]
            with self.assertRaises(ValueError) as cm:                       # the block of column 5 is missing
                orca.read_hess(self._write_small(tmp, hessian_lines=cut))
            self.assertIn("small.hess", str(cm.exception))
            with self.assertRaises(ValueError):                             # rigid motions that are not zero
                orca.read_hess(self._write_small(tmp, rigid="12.500000"))
            p = os.path.join(tmp, "text.hess")
            with open(p, "w") as fh:
                fh.write("this is not a Hessian file\n")
            with self.assertRaises(ValueError):
                orca.read_hess(p)

    def test_harmonic_frequencies_of_a_diatomic(self):
        k = 500.0                                                           # kcal/mol/angstrom^2 along the bond
        H = np.zeros((6, 6))
        for a, b, s in ((2, 2, 1), (5, 5, 1), (2, 5, -1), (5, 2, -1)):
            H[a, b] = s * k
        f = orca.harmonic_frequencies(H, [1.0, 1.0], [[0, 0, 0], [0, 0, 1.0]])
        self.assertEqual(len(f), 1)
        self.assertAlmostEqual(f[0], 108.5913 * np.sqrt(k / 0.5), places=1)   # reduced mass 0.5
        with self.assertRaises(ValueError):
            orca.harmonic_frequencies(H, [1.0], [[0, 0, 0], [0, 0, 1.0]])


class TestSmallTools(unittest.TestCase):
    def test_time_and_memory(self):
        for text, seconds in (("30", 1800), ("30:10", 1810), ("01:00:00", 3600), ("1-00", 86400), ("1-02:03", 93780),
                              ("2-01:01:01", 176461)):
            self.assertEqual(orca.time_seconds(text), seconds)
        with self.assertRaises(ValueError):
            orca.time_seconds("soon")
        self.assertEqual(orca.memory_mb("32G"), 32768); self.assertEqual(orca.memory_mb("32000M"), 32000)
        self.assertEqual(orca.memory_mb("512"), 512); self.assertEqual(orca.memory_mb("16GB"), 16384)
        with self.assertRaises(ValueError):
            orca.memory_mb("much")
        self.assertIsNone(orca.step_limit(None))
        self.assertEqual(orca.step_limit("04:00:00"), 14400 - 432)          # 3 % before the end
        self.assertEqual(orca.step_limit("00:20:00"), 1200 - 120)           # at least 120 s before the end
        self.assertEqual(orca.step_limit("00:02:00"), 60)

    def _run(self, tmp, body, wait=True):
        script = os.path.join(tmp, "s.sh")
        with open(script, "w") as fh:
            fh.write("#!/bin/bash\nset -u\ncd %s\n" % tmp + orca.SCRIPT_PREAMBLE + body)
        proc = subprocess.Popen(["bash", script], cwd=tmp, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                start_new_session=True)
        if not wait:
            return proc
        out = proc.communicate()[0].decode()
        return proc.returncode, out

    def test_step_markers(self):
        with tempfile.TemporaryDirectory() as tmp:
            block = orca.step_block("one", "echo run >> calls.txt; echo result > out.txt",
                                    after=[("[ -s out.txt ]", "out.txt is missing or empty")], clean=["out.txt"])
            rc, out = self._run(tmp, block)
            self.assertEqual(rc, 0, out)
            for name in ("one.started", "one.done"):
                self.assertTrue(os.path.exists(os.path.join(tmp, name)), name)
            self.assertFalse(os.path.exists(os.path.join(tmp, "one.failed")))
            rc, out = self._run(tmp, block)                                 # a second start skips the finished step
            self.assertEqual(rc, 0); self.assertIn("skipped", out)
            with open(os.path.join(tmp, "calls.txt")) as fh:
                self.assertEqual(fh.read().split(), ["run"])

    def test_step_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            cases = [(orca.step_block("a", "bash -c 'exit 7'"), "exit code 7"),      # a program that ends with code 7
                     (orca.step_block("b", "true", after=[("[ -s none.txt ]", "none.txt is missing or empty")]),
                      "none.txt is missing or empty"),
                     (orca.step_block("c", "echo never > c.txt", before=[("[ -x /no/such/program ]", "program not found")]),
                      "program not found")]
            for block, reason in cases:
                rc, out = self._run(tmp, block)
                step = block.split()[3]
                self.assertEqual(rc, 1, out)
                with open(os.path.join(tmp, step + ".failed")) as fh:
                    self.assertEqual(fh.read().strip(), reason)
                self.assertFalse(os.path.exists(os.path.join(tmp, step + ".done")))
            self.assertFalse(os.path.exists(os.path.join(tmp, "c.txt")))    # the command did not start
            # an old result file cannot pass the check of a new attempt
            with open(os.path.join(tmp, "old.txt"), "w") as fh:
                fh.write("from an earlier attempt\n")
            rc, out = self._run(tmp, orca.step_block("d", "true", after=[("[ -s old.txt ]", "old.txt is missing or empty")],
                                                     clean=["old.txt"]))
            self.assertEqual(rc, 1)
            with self.assertRaises(ValueError):
                orca.step_block("bad name", "true")

    @unittest.skipUnless(shutil.which("timeout"), "needs the program timeout")
    def test_step_time_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            start = time.time()
            rc, out = self._run(tmp, orca.step_block("slow", "sleep 20", limit_seconds=1))
            self.assertEqual(rc, 1, out); self.assertLess(time.time() - start, 10.0)
            with open(os.path.join(tmp, "slow.failed")) as fh:
                self.assertEqual(fh.read().strip(), "time limit of 1 s reached")

    def test_step_stopped_by_a_signal(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = self._run(tmp, orca.step_block("long", "sleep 30"), wait=False)
            started = os.path.join(tmp, "long.started")
            for _ in range(100):
                if os.path.exists(started):
                    break
                time.sleep(0.05)
            self.assertTrue(os.path.exists(started))
            os.killpg(proc.pid, signal.SIGTERM)                             # the batch system signals the whole job
            proc.communicate()
            self.assertEqual(proc.returncode, 143)
            with open(os.path.join(tmp, "long.failed")) as fh:
                self.assertIn("stopped by a signal", fh.read())


FAKE_ORCA = """#!/bin/bash
# Stand-in for the ORCA program in the tests: prints a saved output and writes saved result files.
name=${1%%.inp}
echo run >> calls.txt
mode=$(cat mode.txt 2>/dev/null || echo ok)
case "$mode" in
    ok)            cat %(data)s/orca_610_nh3_opt.out; cp %(data)s/orca_610_nh3_min.xyz $name.xyz
                   cp %(data)s/orca_610_nh3_min.hess $name.hess ;;
    no_hess)       cat %(data)s/orca_610_nh3_opt.out; cp %(data)s/orca_610_nh3_min.xyz $name.xyz ;;
    not_converged) cat %(data)s/orca_610_nh3_not_converged.out; cp %(data)s/orca_610_nh3_min.xyz $name.xyz ;;
    crash)         echo "the program stopped" >&2; exit 3 ;;
    neb)           cat %(data)s/orca_610_nh3_neb.out; cp %(data)s/orca_610_nh3_neb_MEP_trj.xyz ${name}_MEP_trj.xyz
                   cp %(data)s/orca_610_nh3_neb_NEB-TS_converged.xyz ${name}_NEB-TS_converged.xyz ;;
    neb_stopped)   cat %(data)s/orca_610_nh3_neb_stopped.out; cp %(data)s/orca_610_nh3_neb_MEP_trj.xyz ${name}_MEP_trj.xyz
                   cp %(data)s/orca_610_nh3_neb_NEB-CI_converged.xyz ${name}_NEB-CI_converged.xyz; exit 9 ;;
esac
"""


def local_machine(tmp, orca_program=None, extra=""):
    """A machine profile that runs jobs on this computer with the stand-in ORCA program."""
    program = os.path.join(tmp, "fake_orca.sh")
    with open(program, "w") as fh:
        fh.write(FAKE_ORCA % {"data": DATA})
    os.chmod(program, 0o755)
    profile = os.path.join(tmp, "machine.ini")
    with open(profile, "w") as fh:
        fh.write("[machine]\nname = test\nscheduler = local\n[executables]\norca = %s\n[job.orca]\ntasks = 2\nmem = 4G\n%s"
                 % (orca_program or program, extra))
    return Machine(profile)


class TestOrcaJobRuns(unittest.TestCase):
    """The real job script, run on this computer with a stand-in for the ORCA program."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.machine = local_machine(self.tmp)
        self.sched = LocalScheduler(max_running=1)
        self.table = JobTable(os.path.join(self.tmp, "jobs.json"))

    def job(self, name="opt", folder="opt", **kw):
        return OrcaJob(os.path.join(self.tmp, folder), self.machine, self.sched, self.table, name=name, **kw)

    def mode(self, job, mode):
        with open(job.file("mode.txt"), "w") as fh:
            fh.write(mode + "\n")

    def calls(self, job):
        with open(job.file("calls.txt")) as fh:
            return len(fh.read().split())

    def test_optimization_done(self):
        job = self.job()
        job.write_optimization(NH3, PYRAMID, method="B3LYP", basis="def2-SVP")
        self.assertEqual(job.state(), ABSENT)
        with open(job.file("opt.inp")) as fh:
            self.assertEqual(fh.read(), "! B3LYP def2-SVP TightOpt TightSCF Freq\n%maxcore 1536\n%pal nprocs 2 end\n"
                                        "* xyzfile 0 1 opt_start.xyz\n")
        first = job.submit()
        self.assertEqual(job.submit(), first)                               # submitted once
        self.sched.wait_all()
        self.assertEqual(job.state(), DONE, job.message)
        self.assertEqual(self.table.attempts(job.key), 1); self.assertEqual(self.calls(job), 1)
        self.assertEqual(job.energy(), -56.473098845005); self.assertEqual(job.version(), "6.1.0")
        symbols, coords = job.geometry()
        self.assertEqual(symbols, NH3); self.assertAlmostEqual(coords[0, 2], 0.130217, places=5)
        self.assertEqual(job.hessian().n_imaginary(), 0)
        # the start geometry is kept apart from the result file opt.xyz
        self.assertAlmostEqual(read_xyz(job.file("opt_start.xyz"))[1][0, 2], 0.1173)
        # a new process: same folder, same table file, no knowledge of the first object
        again = OrcaJob(job.folder, self.machine, LocalScheduler(), JobTable(self.table.path), name="opt")
        self.assertEqual(again.state(), DONE); self.assertEqual(again.require, [orca.OPT_CONVERGED])
        self.assertEqual(again.submit(), first); self.assertEqual(again.resubmit(), first)
        # the job script started a second time does not run ORCA again
        self.sched.submit(job.file("job.sh"), job.folder)
        self.sched.wait_all()
        self.assertEqual(self.calls(job), 1)

    def test_not_converged_then_submitted_again(self):
        job = self.job()
        job.write_optimization(NH3, PYRAMID)
        self.mode(job, "not_converged")
        job.submit(); self.sched.wait_all()
        st = job.status()
        self.assertEqual(st.state, FAILED); self.assertFalse(st.timed_out)
        self.assertIn("no line 'THE OPTIMIZATION HAS CONVERGED' in opt.out", st.message)
        self.assertIn("did not converge", st.message)                       # the error line of the output
        self.assertEqual(job.message, st.message)
        self.mode(job, "ok")
        second = job.resubmit(); self.sched.wait_all()
        self.assertEqual(job.state(), DONE, job.message)
        self.assertEqual(self.table.attempts(job.key), 2); self.assertEqual(self.table.current(job.key)["id"], second)
        self.assertEqual(self.calls(job), 2)

    def test_crash_and_missing_result(self):
        job = self.job()
        job.write_optimization(NH3, PYRAMID)
        self.mode(job, "crash")
        job.submit(); self.sched.wait_all()
        self.assertEqual(job.state(), FAILED)
        self.assertIn("exit code 3", job.message); self.assertIn("the program stopped", job.message)
        other = self.job(folder="opt2")
        other.write_optimization(NH3, PYRAMID)
        self.mode(other, "no_hess")
        other.submit(); self.sched.wait_all()
        self.assertEqual(other.state(), FAILED)
        self.assertIn("the result file opt.hess is missing or empty", other.message)

    def test_program_not_found(self):
        machine = local_machine(self.tmp, orca_program=os.path.join(self.tmp, "no_such_orca"))
        job = OrcaJob(os.path.join(self.tmp, "x"), machine, self.sched, self.table, name="opt")
        job.write_frequencies(NH3, PYRAMID)
        job.submit(); self.sched.wait_all()
        self.assertEqual(job.state(), FAILED)
        self.assertIn("the ORCA program was not found", job.message)

    def test_neb_ts(self):
        job = self.job(name="neb", folder="neb")
        mirror = PYRAMID * np.array([1.0, 1.0, -1.0])
        planar = PYRAMID * np.array([1.0, 1.0, 0.0])
        job.write_neb(NH3, PYRAMID, mirror, saddle_guess=planar, images=5, method="B3LYP", basis="def2-SVP")
        for name in ("reactant.xyz", "product.xyz", "saddle_guess.xyz", "neb.inp", "job.sh", "orca_job.json"):
            self.assertTrue(os.path.exists(job.file(name)), name)
        with open(job.file("neb.inp")) as fh:
            text = fh.read()
        self.assertIn('  NEB_END_XYZFILE "product.xyz"\n  NEB_TS_XYZFILE "saddle_guess.xyz"\n  NImages 5\n', text)
        self.assertTrue(text.endswith("* xyzfile 0 1 reactant.xyz\n"))
        self.mode(job, "neb")
        job.submit(); self.sched.wait_all()
        self.assertEqual(job.state(), DONE, job.message)
        symbols, frames, energies = job.path()
        self.assertEqual(len(frames), 7)
        self.assertEqual(job.saddle()[2], "TS")
        self.assertEqual(job.neb_result(), {"path_converged": True, "saddle_converged": True, "saddle_kind": "TS",
                                            "terminated_normally": True})

    def test_neb_path_converged_saddle_not(self):
        job = self.job(name="neb", folder="neb")
        job.write_neb(NH3, PYRAMID, PYRAMID * np.array([1.0, 1.0, -1.0]), images=5)
        self.mode(job, "neb_stopped")
        job.submit(); self.sched.wait_all()
        self.assertEqual(job.state(), FAILED)
        self.assertIn("exit code 9", job.message)
        self.assertIn("path search converged: yes", job.message)
        self.assertIn("saddle optimization converged: no", job.message)
        res = job.neb_result()                                              # the two results stay readable
        self.assertTrue(res["path_converged"]); self.assertFalse(res["saddle_converged"])
        self.assertEqual(res["saddle_kind"], "CI"); self.assertEqual(len(job.path()[1]), 7)


class TestOrcaJobStates(unittest.TestCase):
    """States and messages, with a batch system that only records what it is asked."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.machine = Machine(os.path.join(ROOT, "machines", "example_slurm.ini"))
        self.sched = FakeScheduler()
        self.table = JobTable(os.path.join(self.tmp, "jobs.json"))
        self.job = OrcaJob(os.path.join(self.tmp, "path", "neb"), self.machine, self.sched, self.table, name="neb")

    def touch(self, name, text="x\n"):
        with open(self.job.file(name), "w") as fh:
            fh.write(text)

    def test_script_text(self):
        job = self.job
        self.assertEqual(job.nprocs, 8); self.assertEqual(job.maxcore, 3072)       # 75 % of 32 GB over 8 processes
        self.assertEqual(job.key, "path/neb")
        job.write_neb(NH3, PYRAMID, PYRAMID * np.array([1.0, 1.0, -1.0]), kind="NEB-CI")
        with open(job.file("neb.inp")) as fh:
            text = fh.read()
        self.assertIn("%pal nprocs 8 end\n", text); self.assertIn("%maxcore 3072\n", text)
        with open(job.file("job.sh")) as fh:
            script = fh.read()
        for needle in ("#SBATCH --ntasks=8", "#SBATCH --time=04:00:00", "#SBATCH --job-name=orca_neb",
                       "#SBATCH --chdir=%s" % job.folder, "module load openmpi/4.1.8", "export OMP_NUM_THREADS=1",
                       "ORCA=/path/to/orca/bin/orca", '"$ORCA" neb.inp > neb.out 2> neb.err', "timeout -k 30 13968",
                       "grep -q -F -- 'ORCA TERMINATED NORMALLY' neb.out",
                       "grep -q -F -- 'THE NEB OPTIMIZATION HAS CONVERGED' neb.out", "[ -s neb_MEP_trj.xyz ]",
                       "orca.started", "neb.out neb_MEP_trj.xyz", "orca.done", "orca.failed", "trap stopped TERM INT"):
            self.assertIn(needle, script)
        self.assertNotIn("THE TS OPTIMIZATION HAS CONVERGED", script)       # NEB-CI has no saddle optimization
        self.assertEqual(subprocess.run(["bash", "-n", job.file("job.sh")]).returncode, 0)     # the syntax is valid

    def test_neb_keys_from_the_profile_and_saddle_requirement(self):
        profile = os.path.join(self.tmp, "m.ini")
        with open(os.path.join(ROOT, "machines", "example_slurm.ini")) as fh:
            text = fh.read()
        with open(profile, "w") as fh:
            fh.write(text + "\n[orca]\nneb_keys = orca6\n")
        job = OrcaJob(os.path.join(self.tmp, "k"), Machine(profile), self.sched, self.table, name="neb", nprocs=4,
                      maxcore=2000, time="00:30:00")
        job.write_neb(NH3, PYRAMID, PYRAMID * np.array([1.0, 1.0, -1.0]))
        with open(job.file("neb.inp")) as fh:
            text = fh.read()
        self.assertIn('  Product "product.xyz"\n', text); self.assertIn("%maxcore 2000\n%pal nprocs 4 end\n", text)
        self.assertEqual(job.require, [orca.NEB_CONVERGED, orca.TS_CONVERGED])
        with open(job.file("job.sh")) as fh:
            script = fh.read()
        self.assertIn("#SBATCH --ntasks=4", script); self.assertIn("#SBATCH --time=00:30:00", script)
        self.assertIn("timeout -k 30 1680", script)
        loose = OrcaJob(os.path.join(self.tmp, "l"), self.machine, self.sched, self.table, name="neb")
        loose.write_neb(NH3, PYRAMID, PYRAMID * np.array([1.0, 1.0, -1.0]), require_saddle=False)
        self.assertEqual(loose.require, [orca.NEB_CONVERGED])

    def test_states(self):
        job, s = self.job, self.sched
        with self.assertRaises(FileNotFoundError):
            job.submit()                                                    # nothing was written yet
        job.write_neb(NH3, PYRAMID, PYRAMID * np.array([1.0, 1.0, -1.0]), kind="NEB-CI")
        self.assertEqual(job.state(), ABSENT)
        self.assertEqual(job.submit(), "F1"); self.assertEqual(job.submit(), "F1")
        self.assertEqual(len(s.submitted), 1)
        self.assertEqual(s.submitted[0]["workdir"], job.folder); self.assertEqual(s.submitted[0]["name"], "orca_neb")
        entry = JobTable(self.table.path).current("path/neb")
        self.assertEqual((entry["id"], entry["kind"], entry["workdir"]), ("F1", "orca", job.folder))
        self.assertEqual(job.state(), PENDING)
        s.answer = sched.RUNNING
        self.assertEqual(job.state(), RUNNING)
        with self.assertRaises(RuntimeError):
            job.resubmit()                                                  # never while the job runs
        s.answer = sched.UNKNOWN
        st = job.status()
        self.assertEqual(st.state, PENDING); self.assertIn("no record of job F1", st.message)
        self.touch("orca.started")
        self.assertEqual(job.state(), RUNNING)
        # the batch job ended without a marker file
        for answer, timed_out in ((sched.FAILED, False), (sched.CANCELLED, False), (sched.TIMEOUT, True),
                                  (sched.COMPLETED, False)):
            s.answer = answer
            st = job.status()
            self.assertEqual(st.state, FAILED); self.assertEqual(st.timed_out, timed_out)
            self.assertIn("batch job F1 ended in state %s after the step started" % answer, st.message)
        self.touch("job.out", "SETUP_FAILED: module load openmpi/4.1.8\n")
        self.assertIn("SETUP_FAILED: module load openmpi/4.1.8", job.status().message)
        # the step wrote its own verdict
        self.touch("orca.failed", "time limit of 13968 s reached\n")
        st = job.status()
        self.assertEqual(st.state, FAILED); self.assertTrue(st.timed_out)
        self.assertIn("time limit of 13968 s reached", st.message)
        # submitted again: the old markers are removed, the table holds both jobs
        self.assertEqual(job.resubmit(), "F2")
        self.assertFalse(os.path.exists(job.file("orca.failed"))); self.assertFalse(os.path.exists(job.file("orca.started")))
        self.assertEqual(self.table.attempts(job.key), 2)
        s.answer = sched.PENDING
        self.assertEqual(job.state(), PENDING)
        # a done marker alone is not enough: the results are checked
        s.answer = sched.COMPLETED
        self.touch("orca.done")
        st = job.status()
        self.assertEqual(st.state, FAILED); self.assertIn("orca.done exists", st.message); self.assertIn("neb.out", st.message)
        shutil.copy(data("orca_610_nh3_neb.out"), job.file("neb.out"))
        self.assertIn("neb_MEP_trj.xyz is missing or empty", job.status().message)
        shutil.copy(data("orca_610_nh3_neb_MEP_trj.xyz"), job.file("neb_MEP_trj.xyz"))
        self.assertEqual(job.state(), DONE)
        job.cancel()
        self.assertEqual(s.cancelled, ["F2"])

    def test_input_is_written_once(self):
        job = self.job
        job.write_frequencies(NH3, PYRAMID)
        job.write_frequencies(NH3, PYRAMID)                                 # the same again: nothing happens
        job.write_frequencies(NH3, PYRAMID + 0.01)                          # not submitted yet: may be replaced
        job.submit()
        job.write_frequencies(NH3, PYRAMID + 0.01)
        with self.assertRaises(ValueError):
            job.write_frequencies(NH3, PYRAMID, method="PBE0")              # another calculation needs another folder
        with self.assertRaises(ValueError):
            OrcaJob(job.folder, self.machine, self.sched, self.table, name="other")
        self.assertEqual(len(self.sched.submitted), 1)


if __name__ == "__main__":
    unittest.main()
