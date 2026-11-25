"""The monitors: each one fires on a trace made to trigger it and stays silent on healthy traces.

The healthy traces are the DMC trace of a water run (``tests/data/trace_h2o_dmc.scalar.dat``:
2048 walkers, time step 0.005 Ha^-1, 100 steps per block, 200 blocks) and synthetic traces with
the same statistics.
"""
import json
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

from montepass import monitor as Mo, stats as S  # noqa: E402

DATA = os.path.join(HERE, "data")
WATER = os.path.join(DATA, "trace_h2o_dmc.scalar.dat")
WATER_VMC = os.path.join(DATA, "trace_h2o_vmc.scalar.dat")
WATER_OUT = os.path.join(DATA, "trace_h2o_output.txt")

SIGMA = 0.006          # hartree: noise of one block of the synthetic traces
WALKERS, STEPS = 2048, 500
BASE_SEED = 0                          # the synthetic trace that the tests change
HEALTHY_SEEDS = (0, 1, 2, 3, 4, 5)


def healthy(n=400, seed=0):
    """A synthetic trace of a healthy run: independent blocks, population within 1.2 %, variance 0.38 Ha^2."""
    r = np.random.default_rng(seed)
    e = -17.24 + SIGMA * r.normal(size=n)
    return {"LocalEnergy": e, "LocalEnergy_sq": e * e + 0.38 * (1.0 + 0.03 * r.normal(size=n)),
            "BlockWeight": WALKERS * STEPS * (1.0 + 0.012 * r.normal(size=n)),
            "BlockCPU": 0.58 * (1.0 + 0.02 * r.normal(size=n)), "AcceptRatio": 0.9973 + 5e-5 * r.normal(size=n)}


def add_energy(trace, delta):
    """Add ``delta`` (a number or one value per block) to the block energies. The column of the squared
    energy is changed with it, so that the variance inside every block stays what it was."""
    e = trace["LocalEnergy"]
    new = e + delta
    trace["LocalEnergy_sq"] = trace["LocalEnergy_sq"] + new * new - e * e
    trace["LocalEnergy"] = new


def one_block(n, index, value):
    out = np.zeros(n)
    out[index] = value
    return out


def check(trace, **kw):
    args = dict(steps=STEPS, timestep=0.005, walkers=WALKERS, vmc_variance=0.375, expected_block_seconds=0.58)
    args.update(kw)
    return Mo.check_trace(trace, **args)


def tests_of(findings, code):
    return sorted(str(f.data.get("test")) for f in findings if f.code == code)


class TestHealthyTraces(unittest.TestCase):
    def test_water_trace_is_silent(self):
        trace = S.read_scalar(WATER)
        vmc = S.read_scalar(WATER_VMC)
        vmc_variance = S.local_energy_variance(vmc["LocalEnergy"], vmc["LocalEnergy_sq"], vmc["BlockWeight"])
        seeds = {"series 1": S.read_seed_ranges(WATER_OUT), "series 2": [(4903, 5441)]}
        found = Mo.check_trace(trace, steps=100, timestep=0.005, walkers=2048, vmc_variance=vmc_variance,
                               expected_block_seconds=0.585, seed_ranges=seeds)
        self.assertEqual(found.as_dicts(), [])
        self.assertIsNone(found.worst())
        for code in (Mo.M2, Mo.M3, Mo.M4, Mo.M5, Mo.M6, Mo.M7, Mo.M8, Mo.M9):
            self.assertIn(code, found.checked)
        self.assertEqual(list(found.skipped), [Mo.M1])                         # the run is finished: no progress test

    def test_water_trace_as_a_running_job(self):
        trace = S.read_scalar(WATER)
        found = Mo.check_trace(trace, steps=100, timestep=0.005, walkers=2048, finished=False, seconds_since_last_row=1.5,
                               expected_block_seconds=0.585, blocks_requested=400, wall_seconds_left=600.0)
        self.assertEqual(found.as_dicts(), [])
        self.assertIn(Mo.M1, found.checked)

    def test_synthetic_traces_are_silent(self):
        for seed in HEALTHY_SEEDS:
            found = check(healthy(seed=seed))
            self.assertEqual(found.as_dicts(), [], "seed %d" % seed)

    def test_a_trace_without_qmc_columns(self):
        # an engine that gives only block energies: the monitors that need more are listed as skipped
        r = np.random.default_rng(1)
        found = Mo.check_trace({"LocalEnergy": r.normal(size=200), "BlockWeight": np.ones(200)}, drop=10)
        self.assertEqual(found.as_dicts(), [])
        self.assertEqual(sorted(found.checked), [Mo.M3, Mo.M5])
        for code in (Mo.M1, Mo.M2, Mo.M4, Mo.M6, Mo.M7, Mo.M8, Mo.M9):
            self.assertIn(code, found.skipped)
        with self.assertRaises(ValueError):
            Mo.check_trace({"BlockWeight": np.ones(10)})
        with self.assertRaises(ValueError):
            Mo.acceptance({"LocalEnergy": np.zeros(10)})


class TestProgressAndSpeed(unittest.TestCase):
    def test_m1_no_progress(self):
        self.assertEqual(Mo.no_progress(40.0, 10.0), [])
        f = Mo.no_progress(100.0, 10.0)
        self.assertEqual([x.code for x in f], [Mo.M1])
        self.assertEqual(f[0].severity, Mo.ERROR)
        self.assertIn("100", f[0].message)
        found = check(healthy(seed=BASE_SEED), finished=False, seconds_since_last_row=30.0)
        self.assertEqual(found.codes(), [Mo.M1])
        self.assertEqual(check(healthy(seed=BASE_SEED), finished=False, seconds_since_last_row=2.0).codes(), [])
        with self.assertRaises(ValueError):
            Mo.no_progress(10.0, 0.0)

    def test_m2_slow_job(self):
        t = healthy(seed=BASE_SEED)
        t["BlockCPU"] = 4.0 * t["BlockCPU"]
        found = check(t)
        self.assertEqual(found.codes(), [Mo.M2])
        self.assertEqual(found[0].severity, Mo.WARNING)
        self.assertAlmostEqual(found[0].data["ratio"], 4.0, delta=0.1)
        self.assertEqual(Mo.slow_job(healthy(seed=BASE_SEED), 0.58), [])

    def test_m2_wall_time(self):
        t = healthy(seed=BASE_SEED)                                            # 400 blocks written, 0.58 s per block
        f = Mo.slow_job(t, 0.58, blocks_requested=1000, wall_seconds_left=100.0)
        self.assertEqual(len(f), 1)
        self.assertTrue(f[0].data["cannot_finish"])
        self.assertEqual(f[0].data["missing_blocks"], 600)
        self.assertEqual(Mo.slow_job(t, 0.58, blocks_requested=1000, wall_seconds_left=1000.0), [])


class TestTransient(unittest.TestCase):
    def test_m3_first_blocks(self):
        t = healthy(seed=BASE_SEED)
        add_energy(t, np.where((np.arange(400) >= 4) & (np.arange(400) < 7), 5.0 * SIGMA, 0.0))                                  # the first 3 kept blocks lie 5 sigma high
        f = Mo.transient(t, drop=4)
        self.assertIn("first_blocks", tests_of(f, Mo.M3))
        self.assertTrue(all(x.severity == Mo.WARNING for x in f))
        self.assertEqual(Mo.transient(healthy(seed=BASE_SEED), drop=4), [])

    def test_m3_data_ask_for_a_longer_discard(self):
        t = healthy(seed=BASE_SEED)
        add_energy(t, 10.0 * SIGMA * np.exp(-np.arange(400) / 20.0))            # a transient of about 60 blocks
        f = Mo.transient(t, drop=4)
        self.assertIn("mser", tests_of(f, Mo.M3))
        mser = [x for x in f if x.data.get("test") == "mser"][0]
        self.assertGreater(mser.data["mser_blocks"], 12)

    def test_m3_drift(self):
        t = healthy(seed=BASE_SEED)
        add_energy(t, np.linspace(0.7 * SIGMA, -0.7 * SIGMA, 400))              # the energy falls during the whole run
        self.assertIn("drift", tests_of(Mo.transient(t, drop=4), Mo.M3))

    def test_m3_discard_fraction(self):
        t = healthy(seed=BASE_SEED)
        self.assertEqual(tests_of(Mo.transient(t, drop=120, finished=True), Mo.M3), ["discard_fraction"])
        self.assertEqual(Mo.transient(t, drop=120, finished=False), [])        # a running job is not judged by its length
        self.assertEqual(tests_of(Mo.transient(t, drop=120, finished=False, blocks_requested=400), Mo.M3),
                         ["discard_fraction"])
        self.assertEqual(Mo.transient(t, drop=120, finished=False, blocks_requested=1000), [])


class TestPopulation(unittest.TestCase):
    def test_m4_mean_off_target(self):
        t = healthy(seed=BASE_SEED)
        t["BlockWeight"] = 0.95 * t["BlockWeight"]
        f = Mo.population(t, WALKERS, STEPS)
        self.assertEqual(tests_of(f, Mo.M4), ["mean"])
        self.assertEqual(f[0].severity, Mo.ERROR)
        self.assertEqual(Mo.population(t), [])                                 # without walkers and steps the mean cannot be judged

    def test_m4_spread(self):
        t = healthy(seed=BASE_SEED)
        r = np.random.default_rng(7)
        t["BlockWeight"] = WALKERS * STEPS * (1.0 + 0.05 * r.normal(size=400))
        self.assertEqual(tests_of(Mo.population(t, WALKERS, STEPS), Mo.M4), ["spread"])
        # the limit grows like 1 / sqrt(walkers): 6 % for 512 walkers
        t["BlockWeight"] = 512 * STEPS * (1.0 + 0.05 * r.normal(size=400))
        self.assertEqual(Mo.population(t, 512, STEPS), [])

    def test_m4_drift(self):
        t = healthy(seed=BASE_SEED)
        t["BlockWeight"][:200] *= 1.03
        self.assertEqual(tests_of(Mo.population(t, WALKERS, STEPS), Mo.M4), ["drift"])

    def test_m4_needs_a_recorded_population(self):
        t = healthy(seed=BASE_SEED)
        t["BlockWeight"] = np.ones(400)
        self.assertEqual(Mo.population(t, WALKERS, STEPS), [])
        self.assertIn(Mo.M4, check(t).skipped)


class TestOutliers(unittest.TestCase):
    def test_m5_one_and_two_outliers(self):
        t = healthy(seed=BASE_SEED)
        add_energy(t, one_block(400, 200, 13.0 * SIGMA))
        f = Mo.outlier_blocks(t, drop=4)
        self.assertEqual([(x.code, x.severity) for x in f], [(Mo.M5, Mo.WARNING)])
        self.assertEqual(f[0].data["blocks"], [200])
        add_energy(t, one_block(400, 300, -13.0 * SIGMA))
        f = Mo.outlier_blocks(t, drop=4)
        self.assertEqual([(x.code, x.severity) for x in f], [(Mo.M5, Mo.ERROR)])
        self.assertEqual(f[0].data["blocks"], [200, 300])
        self.assertEqual(Mo.outlier_blocks(t, drop=301), [])                   # both outliers are in the discarded part

    def test_m5_with_a_population_finding_is_an_error(self):
        t = healthy(seed=BASE_SEED)
        add_energy(t, one_block(400, 200, 13.0 * SIGMA))
        self.assertEqual([(x.code, x.severity) for x in check(t)], [(Mo.M5, Mo.WARNING)])
        t["BlockWeight"] = 0.95 * t["BlockWeight"]
        found = check(t)
        self.assertEqual(sorted(found.codes()), [Mo.M4, Mo.M5])
        self.assertEqual([x.severity for x in found if x.code == Mo.M5], [Mo.ERROR])
        self.assertEqual(found.worst(), Mo.ERROR)
        self.assertEqual(len(found.errors()), 2)


class TestAcceptance(unittest.TestCase):
    def test_m6_acceptance_drop(self):
        t = healthy(seed=BASE_SEED)
        t["AcceptRatio"][250] -= 0.002
        f = Mo.acceptance(t, drop=4)
        self.assertEqual([(x.code, x.severity) for x in f], [(Mo.M6, Mo.ERROR)])
        self.assertEqual(f[0].data["blocks"], [250])

    def test_m6_scatter_of_a_pilot_run_is_not_a_fall(self):
        """512 walkers at time step 0.02: the acceptance of one block scatters by 0.0005 (measured). In 1600 blocks
        about 30 are more than 0.001 below the first blocks. That is not a walker that stopped moving."""
        r = np.random.default_rng(BASE_SEED)
        t = healthy(n=1600, seed=BASE_SEED)
        t["AcceptRatio"] = 0.9903 + 5e-4 * r.normal(size=1600)
        self.assertGreater(int(np.sum(t["AcceptRatio"][10:] < t["AcceptRatio"][:10].mean() - 0.001)), 10)
        self.assertEqual([x.code for x in Mo.acceptance(t, drop=4)], [])

    def test_m6_a_fall_that_lasts(self):
        """One of 512 walkers stops moving at block 800: the acceptance is lower by 0.002 from there on."""
        r = np.random.default_rng(BASE_SEED)
        t = healthy(n=1600, seed=BASE_SEED)
        a = 0.9953 + 5e-4 * r.normal(size=1600)
        a[800:] -= 0.002
        t["AcceptRatio"] = a
        f = Mo.acceptance(t, drop=4)
        self.assertEqual([(x.code, x.severity) for x in f], [(Mo.M6, Mo.ERROR)])
        self.assertEqual(f[0].data["blocks"], [])                  # no single block is far enough below ...
        self.assertTrue(785 <= f[0].data["rows"][0] <= 800)        # ... but the rows of 10 blocks are, from block 800 on
        self.assertIn("in a row", f[0].message)

    def test_m7_acceptance_low(self):
        t = healthy(seed=BASE_SEED)
        t["AcceptRatio"] = t["AcceptRatio"] - 0.0123                           # mean 0.985
        f = Mo.acceptance(t, drop=4)
        self.assertEqual([(x.code, x.severity) for x in f], [(Mo.M7, Mo.ERROR)])
        self.assertAlmostEqual(f[0].data["acceptance"], 0.985, places=3)
        self.assertEqual(check(t).codes(), [Mo.M7])


class TestVariance(unittest.TestCase):
    def test_m8_ratio(self):
        t = healthy(seed=BASE_SEED)
        f = Mo.variance(t, vmc_variance=0.30, drop=4)                          # DMC 0.38 against VMC 0.30
        self.assertEqual([(x.code, x.severity) for x in f], [(Mo.M8, Mo.ERROR)])
        self.assertAlmostEqual(f[0].data["ratio"], 0.38 / 0.30, delta=0.01)
        self.assertEqual(Mo.variance(t, vmc_variance=0.375, drop=4), [])
        self.assertEqual(Mo.variance(t, drop=4), [])                           # no VMC variance: only the change is tested

    def test_m8_one_extreme_sample_is_not_a_wave_function_problem(self):
        t = healthy(seed=BASE_SEED)
        t["LocalEnergy_sq"][150] += 60.0 * 0.38                                # one block with 61 times the usual variance
        f = Mo.variance(t, vmc_variance=0.375, drop=4)
        self.assertEqual([(x.code, x.severity) for x in f], [(Mo.M8, Mo.INFO)])
        self.assertEqual(f[0].data["extreme_blocks"], [150])
        self.assertGreater(f[0].data["ratio"], 1.10)
        self.assertLess(f[0].data["ratio_without_extreme_blocks"], 1.05)

    def test_m8_change_during_the_run(self):
        t = healthy(seed=BASE_SEED)
        e = t["LocalEnergy"]
        v = t["LocalEnergy_sq"] - e * e
        v[200:] *= 1.25
        t["LocalEnergy_sq"] = e * e + v
        f = Mo.variance(t, drop=0)
        self.assertEqual(tests_of(f, Mo.M8), ["change"])
        self.assertEqual(f[0].severity, Mo.ERROR)


class TestSeedsAndCheckpoint(unittest.TestCase):
    def test_m9_seeds(self):
        self.assertEqual(Mo.seeds({"dmc": [(1699, 2213)], "dmce": [(4903, 5441)], "dmcf": [(101, 463)]}), [])
        f = Mo.seeds({"dmc": [(83, 449)], "dmce": [(193, 593)]})
        self.assertEqual([(x.code, x.severity, x.data["verdict"]) for x in f], [(Mo.M9, Mo.ERROR, "overlap")])
        self.assertIn("83-449", f[0].message)
        f = Mo.seeds({"dmc": [(83, 449)], "dmce": None, "dmcf": [(4903, 5441)]})
        self.assertEqual(sorted(tuple(x.data["series"]) for x in f), [("dmc", "dmce"), ("dmce", "dmcf")])
        self.assertTrue(all(x.data["verdict"] == "unknown" and x.severity == Mo.ERROR for x in f))
        self.assertIn("could not be read", f[0].message)
        found = check(healthy(seed=BASE_SEED), seed_ranges={"a": [(83, 449)], "b": [(193, 593)]})
        self.assertEqual(found.codes(), [Mo.M9])

    def test_m10_checkpoint(self):
        f = Mo.checkpoint(-1, 100, False)
        self.assertEqual([(x.code, x.severity) for x in f], [(Mo.M10, Mo.INFO)])
        f = Mo.checkpoint(50, 120, False)
        self.assertEqual([(x.code, x.severity) for x in f], [(Mo.M10, Mo.WARNING)])
        self.assertEqual(Mo.checkpoint(50, 120, True), [])
        self.assertEqual(Mo.checkpoint(50, 20, False), [])                     # the first checkpoint is not due yet


class TestFindings(unittest.TestCase):
    def test_findings_are_plain_data(self):
        t = healthy(seed=BASE_SEED)
        t["AcceptRatio"] = t["AcceptRatio"] - 0.0123
        add_energy(t, one_block(400, 200, 13.0 * SIGMA))
        found = check(t)
        self.assertEqual(sorted(found.codes()), [Mo.M5, Mo.M7])
        self.assertEqual(found.worst(), Mo.ERROR)
        self.assertEqual([x.code for x in found.errors()], [Mo.M7])
        text = json.dumps(found.as_dicts())
        self.assertIn("M7_ACCEPTANCE_LOW", text)
        for x in found:
            self.assertTrue(any(ch.isdigit() for ch in x.message))            # every message carries its numbers

    def test_limits_can_be_changed(self):
        t = healthy(seed=BASE_SEED)
        strict = Mo.Limits(acceptance_min=0.999)
        self.assertEqual(Mo.acceptance(t, limits=strict)[0].code, Mo.M7)
        self.assertEqual(check(t, limits=strict).codes(), [Mo.M7])


if __name__ == "__main__":
    unittest.main()
