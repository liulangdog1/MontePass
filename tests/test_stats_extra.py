"""Additions to the statistics module: the error bar from the autocorrelation time, the start-up
blocks of a run, the combination of independent series, and the random seeds of a QMCPACK run.

``tests/data/trace_h2o_dmc.scalar.dat`` is the DMC trace of one water run (8 valence electrons,
2048 walkers, time step 0.005 Ha^-1, 100 steps per block, 200 blocks); ``trace_h2o_vmc.scalar.dat``
is the VMC trace that made its walkers.
"""
import json
import math
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

from montepass import stats as S  # noqa: E402
from montepass.units import HARTREE_TO_KCAL  # noqa: E402

DATA = os.path.join(HERE, "data")
WATER = os.path.join(DATA, "trace_h2o_dmc.scalar.dat")
WATER_VMC = os.path.join(DATA, "trace_h2o_vmc.scalar.dat")
WATER_OUT = os.path.join(DATA, "trace_h2o_output.txt")


def ar1(n, rho, rng):
    """Stationary Gaussian AR(1) series: unit variance, lag-1 correlation rho, exact tau = (1 + rho) / (2 (1 - rho))."""
    from scipy.signal import lfilter
    xi = rng.normal(size=n)
    y, _ = lfilter([math.sqrt(1.0 - rho * rho)], [1.0, -rho], xi[1:], zi=[rho * xi[0]])
    return np.concatenate([[xi[0]], y])


class TestAutocorrelation(unittest.TestCase):
    def test_function_by_hand(self):
        rho = S.autocorrelation([1.0, 2.0, 3.0, 4.0])
        # deviations -1.5 -0.5 0.5 1.5: c0 = 5/4, c1 = (0.75 - 0.25 + 0.75) / 3, c2 = (-0.75 - 0.75) / 2
        self.assertAlmostEqual(rho[0], 1.0)
        self.assertAlmostEqual(rho[1], 1.0 / 3.0)
        self.assertAlmostEqual(rho[2], -0.6)
        np.testing.assert_allclose(S.autocorrelation([2.0, 2.0, 2.0, 2.0]), [1.0, 0.0, 0.0])
        with self.assertRaises(ValueError):
            S.autocorrelation([1.0])

    def test_white_noise(self):
        x = np.random.default_rng(11).normal(size=4000)
        tau, window, err = S.autocorrelation_time(x)
        self.assertGreaterEqual(tau, 0.5)
        self.assertLess(tau, 0.65)
        self.assertLessEqual(window, 6)
        self.assertLess(err, 0.05)

    def test_ar1_known_correlation_time(self):
        rng = np.random.default_rng(12)
        tau, window, err = S.autocorrelation_time(ar1(20000, 0.6, rng))       # exact 2.0
        self.assertAlmostEqual(tau, 2.0, delta=0.3)
        self.assertTrue(0.04 < err < 0.2)
        self.assertGreaterEqual(window, 5 * tau - 1)                           # the window is 5 tau
        tau9, _, _ = S.autocorrelation_time(ar1(40000, 0.9, rng))              # exact 9.5
        self.assertAlmostEqual(tau9, 9.5, delta=2.0)
        tau3, _, _ = S.autocorrelation_time(ar1(20000, 0.3, rng))              # exact 0.929
        self.assertAlmostEqual(tau3, 0.929, delta=0.12)

    def test_limits(self):
        self.assertEqual(S.autocorrelation_time(np.ones(50))[0], 0.5)          # constant series
        tau, window, err = S.autocorrelation_time([1.0, 2.0, 3.0])             # too short for an estimate
        self.assertEqual((tau, window), (0.5, 0))
        self.assertTrue(math.isnan(err))
        alternating = np.tile([1.0, -1.0], 100)                                # anti-correlated: never below 0.5
        self.assertEqual(S.autocorrelation_time(alternating)[0], 0.5)


class TestEstimate(unittest.TestCase):
    def test_equal_weights(self):
        x = np.random.default_rng(1).normal(size=64)
        est = S.estimate(x)
        self.assertAlmostEqual(est.mean, x.mean(), places=12)
        self.assertAlmostEqual(est.error_naive, x.std(ddof=1) / 8.0, places=12)
        self.assertAlmostEqual(est.error, est.error_naive * math.sqrt(2.0 * est.tau), places=12)
        self.assertEqual((est.n_blocks, est.drop), (64, 0))
        same = S.estimate(x, 3.0 * np.ones(64))                               # a common factor of the weights changes nothing
        self.assertAlmostEqual(same.error_naive, est.error_naive, places=12)
        self.assertAlmostEqual(est.error_reblock, S.reblock(x, np.ones(64))[1], places=14)

    def test_weights_and_drop(self):
        e = np.array([5.0, 1.0, 2.0, 3.0, 6.0])
        w = np.array([9.0, 1.0, 1.0, 2.0, 4.0])
        est = S.estimate(e, w, drop=1)
        self.assertAlmostEqual(est.mean, (1.0 + 2.0 + 6.0 + 24.0) / 8.0)
        self.assertEqual((est.n_blocks, est.drop), (4, 1))
        self.assertEqual(est.tau, 0.5)                                         # fewer than 8 blocks: no estimate of tau
        self.assertTrue(math.isnan(est.error_reblock))
        d = est.as_dict()
        self.assertIsNone(d["error_reblock"]); self.assertIsNone(d["tau_error"])
        json.dumps(d)

    def test_failures_are_errors(self):
        with self.assertRaises(ValueError):
            S.estimate([1.0, 2.0, 3.0], drop=2)                                # one block left
        with self.assertRaises(ValueError):
            S.estimate([1.0, float("nan"), 3.0])
        with self.assertRaises(ValueError):
            S.estimate([1.0, 2.0, 3.0], [1.0, 1.0])
        with self.assertRaises(ValueError):
            S.estimate([1.0, 2.0, 3.0], drop=-1)
        with self.assertRaises(ValueError):
            S.estimate([1.0, 2.0, 3.0], [0.0, 0.0, 0.0])

    def test_correlated_blocks_get_a_larger_error(self):
        rng = np.random.default_rng(21)
        est = S.estimate(ar1(4000, 0.6, rng))
        ratio = est.error / est.error_naive                                    # exact sqrt(2 tau) = 2.0
        self.assertTrue(1.7 < ratio < 2.3, ratio)
        # The reblocking rule takes the largest standard error over the block sizes. At the largest sizes
        # only 8 groups remain, so that maximum scatters upwards: 0.99 to 1.48 for six seeds of this series.
        self.assertTrue(0.7 < est.error_reblock / est.error < 1.8, est.error_reblock / est.error)

    def test_error_bar_is_calibrated_for_independent_blocks(self):
        rng = np.random.default_rng(22)
        z = []
        for _ in range(200):
            est = S.estimate(rng.normal(size=256))
            z.append(est.mean / est.error)
        rms = float(np.sqrt(np.mean(np.square(z))))
        self.assertTrue(0.82 < rms < 1.14, rms)

    def test_water_trace(self):
        e, w = S.energy_weight(WATER)
        self.assertEqual(len(e), 200)
        est = S.estimate(e, w, drop=10)
        self.assertAlmostEqual(est.mean, -17.24009, places=5)
        self.assertAlmostEqual(est.error_reblock, 0.000602, delta=2e-6)        # rule of protocol v1
        self.assertAlmostEqual(est.error, 0.000681, delta=2e-6)               # from the autocorrelation time
        self.assertAlmostEqual(est.error * HARTREE_TO_KCAL, 0.427, delta=0.002)
        self.assertAlmostEqual(est.tau, 1.108, delta=0.005)                    # blocks of 0.5 Ha^-1 are correlated
        self.assertEqual(est.n_blocks, 190)
        self.assertAlmostEqual(float(np.std(e[10:], ddof=1)) * HARTREE_TO_KCAL, 3.96, delta=0.01)

    def test_variance_of_the_local_energy(self):
        self.assertAlmostEqual(S.local_energy_variance([1.0, 3.0], [2.0, 12.0]), 7.0 - 4.0)
        self.assertAlmostEqual(S.local_energy_variance([1.0, 3.0], [2.0, 12.0], [3.0, 1.0]), 4.5 - 2.25)
        cols = S.read_scalar(WATER)
        v = S.local_energy_variance(cols["LocalEnergy"], cols["LocalEnergy_sq"], cols["BlockWeight"], drop=10)
        self.assertAlmostEqual(v, 0.379, places=3)
        vmc = S.read_scalar(WATER_VMC)
        vv = S.local_energy_variance(vmc["LocalEnergy"], vmc["LocalEnergy_sq"], vmc["BlockWeight"])
        self.assertAlmostEqual(vv, 0.37068, places=4)                          # the value that QMCPACK printed for this run
        self.assertAlmostEqual(float(cols["AcceptRatio"][10:].mean()), 0.9973, places=4)
        self.assertAlmostEqual(float(cols["BlockCPU"].mean()), 0.585, delta=0.005)
        with self.assertRaises(ValueError):
            S.local_energy_variance([1.0, 2.0], [1.0])


class TestEquilibration(unittest.TestCase):
    def test_transient_blocks(self):
        self.assertEqual(S.transient_blocks(500, 0.005), 4)                    # 10 Ha^-1 / 2.5 Ha^-1
        self.assertEqual(S.transient_blocks(100, 0.005), 20)
        self.assertEqual(S.transient_blocks(250, 0.01), 4)
        self.assertEqual(S.transient_blocks(125, 0.02), 4)
        self.assertEqual(S.transient_blocks(10, 0.005), 200)
        self.assertEqual(S.transient_blocks(568, 0.0044), 5)                   # 2.4992 Ha^-1 per block
        self.assertEqual(S.transient_blocks(500, 0.005, transient=4.0), 2)
        self.assertEqual(S.transient_blocks(500, 0.005, transient=0.0), 0)
        with self.assertRaises(ValueError):
            S.transient_blocks(0, 0.005)

    def test_mser_finds_a_transient(self):
        rng = np.random.default_rng(31)
        x = rng.normal(size=300) + 30.0 * np.exp(-np.arange(300) / 3.0)
        d = S.mser(x)
        self.assertTrue(5 <= d <= 25, d)
        d5 = S.mser(x, batch=5)
        self.assertEqual(d5 % 5, 0)
        self.assertTrue(5 <= d5 <= 30, d5)

    def test_mser_on_stationary_data(self):
        rng = np.random.default_rng(32)
        cuts = [S.mser(rng.normal(size=300)) for _ in range(40)]
        self.assertLessEqual(float(np.median(cuts)), 2.0)
        self.assertLessEqual(max(cuts), 150)                                   # never more than half of the series
        self.assertEqual(S.mser(np.arange(9.0)), 0)                            # too short

    def test_mser_against_the_definition(self):
        x = np.random.default_rng(33).normal(size=60) + 5.0 * np.exp(-np.arange(60) / 4.0)
        best, best_d = None, 0
        for d in range(0, 31):
            tail = x[d:]
            val = float(np.sum((tail - tail.mean()) ** 2)) / len(tail) ** 2
            if best is None or val < best:
                best, best_d = val, d
        self.assertEqual(S.mser(x), best_d)

    def test_drift_test(self):
        rng = np.random.default_rng(34)
        x = rng.normal(size=400)
        diff, err, z = S.drift_test(x)
        self.assertLess(abs(z), 3.0)
        self.assertAlmostEqual(diff, x[:200].mean() - x[200:].mean(), places=12)
        y = x.copy(); y[:200] += 1.0                                           # the first half lies 1 sigma higher
        diff, err, z = S.drift_test(y)
        self.assertGreater(z, 5.0)
        self.assertGreater(diff, 0.0)
        with self.assertRaises(ValueError):
            S.drift_test([1.0, 2.0, 3.0])

    def test_equilibration_combines_the_three_pieces(self):
        rng = np.random.default_rng(35)
        x = rng.normal(size=300) + 30.0 * np.exp(-np.arange(300) / 3.0)        # transient of about 10 blocks
        eq = S.equilibration(x, steps=500, timestep=0.005)
        self.assertEqual(eq.drop_physical, 4)
        self.assertGreater(eq.drop_mser, 4)
        self.assertEqual(eq.drop, eq.drop_mser)                                # the data ask for more: the larger number is used
        self.assertIn("more than", eq.message)
        self.assertEqual(eq.n_kept, 300 - eq.drop)
        self.assertTrue(eq.stationary)
        eq2 = S.equilibration(x)                                               # no steps, no time step: data only
        self.assertIsNone(eq2.drop_physical)
        self.assertEqual(eq2.drop, eq2.drop_mser)
        json.dumps(eq.as_dict())

    def test_equilibration_reports_a_drift(self):
        rng = np.random.default_rng(36)
        x = rng.normal(size=400) + np.linspace(3.0, 0.0, 400)                  # the whole run still falls
        eq = S.equilibration(x, steps=500, timestep=0.005)
        self.assertFalse(eq.stationary)
        self.assertGreater(eq.drift_z, 3.0)
        self.assertIn("drift", eq.message)

    def test_equilibration_of_a_short_series(self):
        eq = S.equilibration([1.0, 2.0, 3.0], steps=500, timestep=0.005)
        self.assertEqual((eq.drop, eq.n_kept), (4, 0))
        self.assertIsNone(eq.stationary)
        self.assertIsNone(eq.drift_z)

    def test_water_trace(self):
        e, w = S.energy_weight(WATER)
        eq = S.equilibration(e, w, steps=100, timestep=0.005)
        self.assertEqual((eq.drop_physical, eq.drop, eq.n_kept), (20, 20, 180))
        self.assertLessEqual(eq.drop_mser, 5)
        self.assertTrue(eq.stationary)
        self.assertLess(abs(eq.drift_z), 1.0)


class TestCombine(unittest.TestCase):
    def test_known_numbers(self):
        c = S.combine([1.0, 1.2, 0.9], [0.1, 0.2, 0.1])
        self.assertAlmostEqual(c.mean, 0.977778, places=6)
        self.assertAlmostEqual(c.error, 0.0666667, places=6)
        self.assertAlmostEqual(c.chi2, 1.888889, places=6)
        self.assertEqual((c.dof, c.n), (2, 3))
        self.assertAlmostEqual(c.p_value, math.exp(-c.chi2 / 2.0), places=10)  # exact for 2 degrees of freedom
        self.assertAlmostEqual(c.error_scaled, c.error)                        # chi2 / dof below 1: no scaling
        json.dumps(c.as_dict())

    def test_two_equal_series(self):
        c = S.combine([2.0, 2.0], [0.3, 0.3])
        self.assertAlmostEqual(c.mean, 2.0)
        self.assertAlmostEqual(c.error, 0.3 / math.sqrt(2.0))
        self.assertAlmostEqual(c.chi2, 0.0)
        self.assertAlmostEqual(c.p_value, 1.0)

    def test_series_that_disagree(self):
        c = S.combine([0.0, 1.0], [0.1, 0.1])                                  # 7 sigma apart
        self.assertAlmostEqual(c.chi2, 50.0)
        self.assertLess(c.p_value, 1e-10)
        self.assertAlmostEqual(c.error_scaled, c.error * math.sqrt(50.0))

    def test_one_series_and_bad_input(self):
        c = S.combine([1.5], [0.2])
        self.assertAlmostEqual(c.mean, 1.5)
        self.assertAlmostEqual(c.error, 0.2)
        self.assertEqual(c.dof, 0)
        self.assertIsNone(c.p_value)
        for means, errors in (([], []), ([1.0, 2.0], [0.1]), ([1.0, 2.0], [0.1, 0.0]), ([1.0, float("nan")], [0.1, 0.1])):
            with self.assertRaises(ValueError):
                S.combine(means, errors)


class TestSeeds(unittest.TestCase):
    def test_reading_the_range(self):
        with open(WATER_OUT) as fh:
            text = fh.read()
        self.assertEqual(S.seed_ranges(text), [(2467, 2719)])
        self.assertEqual(S.read_seed_ranges(WATER_OUT), [(2467, 2719)])
        self.assertEqual(S.read_seed_ranges(os.path.join(DATA, "no_such_file.out")), [])
        self.assertEqual(S.seed_ranges("QMCPACK output without the seed line\n"), [])
        line = "  Range of prime numbers to use as seeds over processors and threads = %s\n"
        self.assertEqual(S.seed_ranges(line % "83-449" + "text\n" + line % "193-593" + line % "83-449"),
                         [(83, 449), (193, 593)])                              # an output appended by two runs
        self.assertEqual(S.seed_ranges(line % "83-449" + line % "12-"), [])    # one unreadable line: nothing is known
        self.assertEqual(S.seed_ranges(line % "449-83"), [])

    def test_unknown_is_never_called_disjoint(self):
        for a, b in ((None, (83, 449)), ((83, 449), None), ([], [(83, 449)]), (None, None), ((449, 83), (1000, 2000)),
                     ("83-449", (1000, 2000)), ([(83, 449), None], (1000, 2000))):
            self.assertEqual(S.seed_overlap(a, b), "unknown", (a, b))
            self.assertFalse(S.seeds_independent(a, b))

    def test_overlap_and_disjoint(self):
        self.assertEqual(S.seed_overlap((83, 449), (193, 593)), "overlap")     # two runs with 64 ranks that shared 44 primes
        self.assertEqual(S.seed_overlap((83, 449), (83, 449)), "overlap")
        self.assertEqual(S.seed_overlap((1699, 2213), (4903, 5441)), "disjoint")
        self.assertEqual(S.seed_overlap((101, 463), (7297, 7873)), "disjoint")
        self.assertEqual(S.seed_overlap([(101, 463)], [(1699, 2213), (7297, 7873)]), "disjoint")
        self.assertEqual(S.seed_overlap([(101, 463)], [(1699, 2213), (401, 800)]), "overlap")
        self.assertTrue(S.seeds_independent((1699, 2213), (4903, 5441)))
        self.assertFalse(S.seeds_independent((83, 449), (193, 593)))
        self.assertEqual(S.seed_overlap([101, 463], [7297, 7873]), "disjoint")  # lists, as read back from a JSON file

    def test_safety_margin(self):
        # 101-463 holds 65 primes (64 ranks). The margin adds the next 65 primes: up to 907.
        self.assertEqual(S.seed_overlap((101, 463), (467, 900), margin=0.0), "disjoint")
        self.assertEqual(S.seed_overlap((101, 463), (467, 900)), "overlap")
        self.assertEqual(S.seed_overlap((101, 463), (907, 1300)), "overlap")
        self.assertEqual(S.seed_overlap((101, 463), (911, 1300)), "disjoint")
        self.assertEqual(S.seed_overlap((911, 1300), (101, 463)), "disjoint")  # the order of the two runs does not matter


class TestPlain(unittest.TestCase):
    def test_json_values(self):
        out = S.plain({"a": np.float64(1.5), "b": np.int64(3), "c": float("nan"), "d": np.array([1.0, float("inf")]),
                       "e": (np.bool_(True), "text", None)})
        self.assertEqual(out, {"a": 1.5, "b": 3, "c": None, "d": [1.0, None], "e": [True, "text", None]})
        json.dumps(out)


if __name__ == "__main__":
    unittest.main()
