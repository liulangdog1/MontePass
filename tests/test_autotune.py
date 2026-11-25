"""Automatic choice of the QMC settings: the planning formulas against worked numbers, the
time-step rules on hand-made pilot results, and the pilot driver on the mock engine.

Worked numbers of a made-up model molecule with 36 valence electrons ("the large molecule" below).
These values are not measurements: SCF energy -55.0 hartree, variance 1.2 Ha^2, effective correlation
time 0.13 Ha^-1, 8.0e-4 core-seconds for one DMC step of one walker, slope of the DMC energy against
the time step -300 kcal/mol per Ha^-1, DMC acceptance 0.9917 / 0.9966 / 0.9986 at the time steps
0.02 / 0.01 / 0.005 Ha^-1 (a rejected fraction of 0.0014 at 0.005 Ha^-1 that grows like tau^1.28).
The expected numbers that follow from them are the output of the package functions for these inputs.
"""
import json
import math
import os
import sys
import tempfile
import unittest
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

from montepass import autotune as A, model as M, stats as S  # noqa: E402
from montepass.engine import FAILED, PointData  # noqa: E402
from montepass.engine.mock import MockEngine  # noqa: E402
from montepass.engine.qmcpack_inputs import DEFAULT_JASTROW  # noqa: E402
from montepass.search import protocol_v2 as P2  # noqa: E402
from montepass.units import HARTREE_TO_KCAL as H  # noqa: E402

DATA = os.path.join(HERE, "data")
WATER = os.path.join(DATA, "trace_h2o_dmc.scalar.dat")
WATER_VMC = os.path.join(DATA, "trace_h2o_vmc.scalar.dat")

NE_BIG, E_BIG = 36, -55.0             # the large molecule (a model): valence electrons, SCF energy in hartree
V_BIG, TC_BIG = 1.2, 0.13             # its variance (Ha^2) and effective correlation time (Ha^-1)
STEP_BIG = 8.0e-4                     # its core-seconds for one step of one walker
SLOPE_BIG = -300.0                    # its slope of the DMC energy against the time step, kcal/mol per Ha^-1
ACCEPTANCE_BIG = (0.9917, 0.9966, 0.9986)       # its DMC acceptance at the time steps 0.02, 0.01, 0.005
REJECTED_BIG = 0.0014                 # 1 - acceptance at 0.005


class TestFormulas(unittest.TestCase):
    def test_f1_worked_number(self):
        f = A.forecast(NE_BIG, E_BIG, 0.2, walkers=2048, timestep=0.005, variance=V_BIG, t_corr=TC_BIG)
        self.assertAlmostEqual(f.time, 1499.7, delta=1.5)                      # Ha^-1 for 0.2 kcal/mol
        self.assertEqual((f.steps, f.discard), (500, 4))
        self.assertEqual(f.blocks, 4 + math.ceil(1.2 * f.time / 2.5))          # F2
        self.assertEqual(f.blocks, 724)
        self.assertAlmostEqual(f.sigma_block, 4.90, delta=0.01)                # error of one block of 2.5 Ha^-1, kcal/mol
        self.assertEqual(f.guesses, ["step_seconds"])

    def test_f1_short_run(self):
        # 30 Ha^-1 of data give 1.41 kcal/mol (a short scan run of the large molecule)
        p = A.plan_dmc(0.005, 2048, blocks=400, variance=V_BIG, t_corr=TC_BIG, block_time=0.1)
        self.assertEqual((p.steps, p.discard), (20, 100))
        self.assertAlmostEqual(p.time, 30.0, places=9)
        self.assertAlmostEqual(p.sigma_expected, 1.414, delta=0.005)
        self.assertIsNone(p.sigma_target)

    def test_f2_f3_f4_first_guesses(self):
        f = A.forecast(NE_BIG, E_BIG, 0.2)
        self.assertEqual(f.guesses, ["variance", "t_corr", "step_seconds"])
        self.assertAlmostEqual(f.variance, 0.022 * abs(E_BIG), places=12)
        self.assertEqual(f.t_corr, 0.14)
        self.assertAlmostEqual(f.step_seconds, 5.4e-4 * (NE_BIG / 28.0) ** 1.9, places=12)
        need = 2.0 * 0.14 * 0.022 * abs(E_BIG) / (2048 * (0.2 / H) ** 2)
        self.assertAlmostEqual(f.time, need, places=6)
        # the forecast is within 15 % of the need computed with the variance and correlation time of the model
        self.assertLess(abs(f.time / 1499.7 - 1.0), 0.15)
        self.assertEqual(f.blocks, 4 + math.ceil(1.2 * need / 2.5))
        self.assertAlmostEqual(f.core_hours, f.blocks * 500 * 2048 * f.step_seconds / 3600.0, places=6)
        self.assertAlmostEqual(A.step_seconds_guess(28), 5.4e-4, places=12)
        self.assertLess(abs(A.step_seconds_guess(NE_BIG) / STEP_BIG - 1.0), 0.10)    # the model value 8.0e-4 core-seconds
        # water: the guessed variance is the measured one (0.379 Ha^2)
        self.assertAlmostEqual(A.forecast(8, -17.2371, 0.1).variance, 0.379, delta=0.002)
        with self.assertRaises(ValueError):
            A.forecast(8, 0.0)

    def test_plan_dmc_rules(self):
        for tau, steps, warmup, discard in ((0.005, 500, 2000, 4), (0.01, 250, 1000, 4), (0.02, 125, 500, 4),
                                            (0.0044, 568, 2273, 5)):
            p = A.plan_dmc(tau, 2048, sigma_target=0.2, variance=V_BIG, t_corr=TC_BIG)
            self.assertEqual((p.steps, p.warmup_steps, p.discard), (steps, warmup, discard), tau)
            self.assertAlmostEqual(p.block_time, 2.5, delta=0.001)
        p = A.plan_dmc(0.005, 2048, sigma_target=0.2, variance=V_BIG, t_corr=TC_BIG, safety=1.0)
        self.assertEqual(p.blocks, 4 + 600)
        self.assertTrue(0.1995 < p.sigma_expected <= 0.2)
        self.assertEqual(p.notes, [])
        self.assertAlmostEqual(p.noise, 2.0 * TC_BIG * V_BIG, places=12)

    def test_plan_dmc_keeps_enough_blocks(self):
        p = A.plan_dmc(0.005, 2048, sigma_target=5.0, variance=V_BIG, t_corr=TC_BIG)      # needs 2 blocks
        self.assertEqual((p.blocks, p.discard), (68, 4))
        self.assertEqual(len(p.notes), 1)
        self.assertAlmostEqual(p.sigma_expected, p.sigma_block / 8.0, places=9)

    def test_plan_dmc_from_blocks_and_cost(self):
        p = A.plan_dmc(0.005, 2048, blocks=404, noise=2.0 * TC_BIG * V_BIG, step_seconds=STEP_BIG, ranks=64, setup_seconds=600.0)
        self.assertAlmostEqual(p.sigma_expected, p.sigma_block / 20.0, places=9)
        self.assertAlmostEqual(p.sigma_expected, 0.245, delta=0.001)
        core_seconds = 404 * 500 * 2048 * STEP_BIG
        self.assertAlmostEqual(p.core_hours, core_seconds / 3600.0, places=6)
        self.assertAlmostEqual(p.wall_seconds, 1.3 * core_seconds / 64 + 600.0, places=6)
        self.assertIsNone(A.plan_dmc(0.005, 2048, blocks=404, noise=0.6).core_hours)
        self.assertIsNone(A.plan_dmc(0.005, 2048, blocks=404, noise=0.6, step_seconds=1e-3).wall_seconds)
        short = A.plan_dmc(0.005, 2048, blocks=12, noise=0.6)
        self.assertEqual(len(short.notes), 2)                                  # fewer than 64 kept blocks; shorter than 5 transients
        for kw in (dict(), dict(sigma_target=0.2, blocks=400), dict(blocks=4), dict(sigma_target=0.2, noise=None)):
            args = dict(variance=V_BIG, t_corr=TC_BIG)
            if "noise" in kw:
                args = {}
            args.update({k: v for k, v in kw.items() if k != "noise"})
            with self.assertRaises(ValueError):
                A.plan_dmc(0.005, 2048, **args)

    def test_scan_point_error(self):
        # protocol v1, shells (0.5, 1): 0.43 kcal/mol for 6 directions
        self.assertAlmostEqual(A.scan_point_sigma(6, 5.0, 0.2, (0.5, 1.0), "rule_b"), 0.430, places=3)
        # a point error of 1.29 kcal/mol costs 0.2 kcal/mol for 6 directions (F7)
        self.assertAlmostEqual(A.scan_noise_energy(6, 1.29, 5.0, (0.5, 1.0)), 0.2, delta=0.001)
        # protocol v2, one shell: the joint 95 % bound c_n * N equals the tolerance at this point error
        s = A.scan_point_sigma(6, 5.0, 0.2, (1.0,), "joint")
        self.assertAlmostEqual(s, math.sqrt(8 * 5.0 * 0.2 / 12.592), places=9)
        self.assertAlmostEqual(P2.joint_factor(6) * A.scan_noise_energy(6, s, 5.0, (1.0,)), 0.2, places=9)
        # the same condition as the block plan of the protocol, for a block noise of 5 kcal/mol
        v2 = P2.ProtocolV2()
        self.assertEqual(v2.planned_blocks(6, 5.0), math.ceil((5.0 / s) ** 2) + v2.drop_min + v2.min_blocks_after_drop)
        self.assertLess(A.scan_point_sigma(12), A.scan_point_sigma(3))         # more directions need better points
        # a saddle search: with a reaction coordinate the points must be 1.86 times better (3.5 times the blocks)
        saddle = A.scan_point_sigma(6, 5.0, 0.2, (1.0,), "joint", n_negative=1)
        total, factor = v2.plan_noise(6, 1)
        self.assertAlmostEqual(saddle, math.sqrt(0.2 / (factor * total)), places=9)
        self.assertAlmostEqual(s / saddle, 1.864, delta=0.002)
        self.assertEqual(v2.__class__(cap=10 ** 6).planned_blocks(6, 5.0, n_negative=1),
                         math.ceil((5.0 / saddle) ** 2) + v2.drop_min + v2.min_blocks_after_drop)
        self.assertEqual(A.scan_point_sigma(6, 5.0, 0.2, (1.0,), "joint", n_negative=0), s)
        with self.assertRaises(ValueError):
            A.scan_point_sigma(6, rule="other")

    def test_jastrow_defaults(self):
        water = M.water_geometry(r=0.9619, theta=105.0)                        # largest distance 2.88 bohr
        self.assertEqual(A.jastrow_defaults(water), {"j1_size": 5, "j1_rcut": 4.9, "j2_size": 7, "j2_rcut": 6.9})
        long_molecule = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 5.0]])
        self.assertEqual(A.jastrow_defaults(long_molecule), DEFAULT_JASTROW)   # 6 and 8 bohr, the production values
        self.assertEqual(A.jastrow_defaults(np.zeros((1, 3))), {"j1_size": 4, "j1_rcut": 4.0, "j2_size": 5, "j2_rcut": 5.0})

    def test_walkers_are_a_multiple_of_the_ranks(self):
        self.assertEqual(A.walkers_for(64, "final"), 2048)
        self.assertEqual(A.walkers_for(32, "scan"), 2048)
        self.assertEqual(A.walkers_for(48, "centre"), 2064)
        self.assertEqual(A.walkers_for(48, "pilot"), 528)
        self.assertEqual(A.walkers_for(16, "pilot"), 512)
        self.assertEqual(A.walkers_for(200, "pilot"), 800)                     # at least 4 pilot walkers per rank
        self.assertEqual(A.walkers_for(3000, "final"), 3000)
        self.assertEqual(A.walkers_for(32, "scan", target=1000), 1024)
        for ranks in (1, 7, 24, 36, 96, 128):
            for role in ("final", "centre", "scan", "pilot"):
                n = A.walkers_for(ranks, role)
                self.assertEqual(n % ranks, 0)
                self.assertGreaterEqual(n, A.WALKER_TARGET[role])
        with self.assertRaises(ValueError):
            A.walkers_for(32, "other")
        with self.assertRaises(ValueError):
            A.walkers_for(0, "final")

    def test_f5_time_step_of_99_percent_acceptance(self):
        self.assertAlmostEqual(A.timestep_for_acceptance(0.005, ACCEPTANCE_BIG[2]), 0.0232, delta=0.0005)   # the large molecule
        # a second model molecule with a lower acceptance, 0.9978 (made up). The expected value is the output of the
        # package function for this input.
        self.assertAlmostEqual(A.timestep_for_acceptance(0.005, 0.9978), 0.0163, delta=0.0005)
        self.assertAlmostEqual(A.timestep_for_acceptance(0.01, 0.99), 0.01, places=12)
        self.assertEqual(A.candidate_timesteps(0.0163), (0.02, 0.01, 0.005))     # the two model molecules
        self.assertEqual(A.candidate_timesteps(0.0232), (0.02, 0.01, 0.005))
        self.assertEqual(A.candidate_timesteps(0.0061), (0.006, 0.003, 0.0015))
        with self.assertRaises(ValueError):
            A.timestep_for_acceptance(0.005, 1.0)

    def test_fit_slope(self):
        taus = [0.02, 0.01, 0.005]
        fit = A.fit_slope(taus, [E_BIG + SLOPE_BIG * t / H for t in taus], [1e-4] * 3)
        self.assertAlmostEqual(fit["slope"], SLOPE_BIG, places=6)
        self.assertAlmostEqual(fit["intercept"], E_BIG, places=9)
        self.assertAlmostEqual(fit["chi2"], 0.0, places=6)
        self.assertEqual(fit["dof"], 1)
        spread = math.sqrt(sum((t - np.mean(taus)) ** 2 for t in taus))
        self.assertAlmostEqual(fit["slope_error"], 1e-4 * H / spread, places=6)
        # model energies relative to the value at 0.005 (kcal/mol): a slope of -300 plus made-up scatter of up to
        # 0.05 kcal/mol. The expected slope is the output of the package function for these inputs.
        fit = A.fit_slope([0.0025, 0.005, 0.01, 0.02], [x / H for x in (0.80, 0.0, -1.45, -4.55)], [0.3 / H] * 4)
        self.assertAlmostEqual(fit["slope"], -304.7, delta=1.0)
        self.assertEqual(fit["dof"], 2)
        with self.assertRaises(ValueError):
            A.fit_slope([0.01, 0.01], [-1.0, -1.0], [0.1, 0.1])


def results(acceptance=ACCEPTANCE_BIG, slope=SLOPE_BIG, error=0.02, ratio=(1.01, 1.01, 1.01),
            spread=(0.017, 0.017, 0.017)):
    """Hand-made pilot results at the time steps 0.02, 0.01, 0.005. ``slope`` in kcal/mol per Ha^-1, ``error`` in kcal/mol."""
    return [{"timestep": tau, "acceptance": acceptance[i], "variance_ratio": ratio[i], "population_spread": spread[i],
             "walkers": 2048, "energy": E_BIG + slope * tau / H, "error": error / H}
            for i, tau in enumerate((0.02, 0.01, 0.005))]


class TestChooseTimestep(unittest.TestCase):
    def test_r4_worked_case(self):
        # slope 300: with a tolerance of 0.3 kcal/mol the limit is 0.0067, so 0.005 is chosen
        c = A.choose_timestep(results(), eps_tau=0.3)
        self.assertEqual((c.timestep, c.rule), (0.005, "R4"))
        self.assertAlmostEqual(c.limit, 0.3 / (0.15 * 300.0), places=6)
        self.assertAlmostEqual(c.slope, SLOPE_BIG, places=3)
        self.assertEqual(c.slope_source, "measured")
        self.assertIn("R4", c.text); self.assertIn("0.01", c.text)
        self.assertAlmostEqual(c.tau_99, 0.0232, delta=0.0005)
        self.assertEqual(c.not_checked, [])
        self.assertEqual([row["R4"] for row in c.table], ["fail", "fail", "pass"])
        json.dumps(c.as_dict())

    def test_f6_when_no_candidate_passes_r4(self):
        # with a tolerance of 0.2 kcal/mol the limit is 0.0044: below the smallest candidate
        c = A.choose_timestep(results(), eps_tau=0.2)
        self.assertEqual((c.timestep, c.rule), (0.0044, "F6"))
        self.assertFalse(c.extrapolate)                                        # 0.0044 is above tau_99 / 9 = 0.0026
        self.assertIn("F6", c.text)
        # a much steeper slope: the limit falls below tau_99 / 9 and two time steps with extrapolation cost less
        c = A.choose_timestep(results(slope=-3000.0), eps_tau=0.2)
        self.assertEqual(c.rule, "F6")
        self.assertAlmostEqual(c.timestep, 0.00044, places=6)
        self.assertTrue(c.extrapolate)

    def test_r1_acceptance_decides(self):
        c = A.choose_timestep(results(acceptance=(0.985, 0.994, 0.998), slope=-20.0), eps_tau=0.2)
        self.assertEqual((c.timestep, c.rule), (0.01, "R1"))
        self.assertIn("acceptance 0.9850", c.text)
        self.assertEqual([row["R1"] for row in c.table], ["fail", "pass", "pass"])

    def test_r2_variance_ratio_decides(self):
        c = A.choose_timestep(results(slope=-20.0, ratio=(1.15, 1.02, 1.02)), eps_tau=0.2)
        self.assertEqual((c.timestep, c.rule), (0.01, "R2"))
        # the ratio can also be given as two variances
        rows = results(slope=-20.0)
        for row, v in zip(rows, (1.45, 1.26, 1.26)):                          # ratios 1.21, 1.05, 1.05
            del row["variance_ratio"]
            row["variance"], row["vmc_variance"] = v, V_BIG
        c = A.choose_timestep(rows, eps_tau=0.2)
        self.assertEqual((c.timestep, c.rule), (0.01, "R2"))

    def test_r3_population_spread_decides(self):
        c = A.choose_timestep(results(slope=-20.0, spread=(0.07, 0.02, 0.02)), eps_tau=0.2)
        self.assertEqual((c.timestep, c.rule), (0.01, "R3"))
        # 512 walkers: the limit is 5 % * sqrt(2048 / 512) = 10 %, so a spread of 7 % passes
        rows = results(slope=-20.0, spread=(0.07, 0.02, 0.02))
        for row in rows:
            row["walkers"] = 512
        c = A.choose_timestep(rows, eps_tau=0.2)
        self.assertEqual((c.timestep, c.rule), (0.02, "largest"))

    def test_largest_candidate_when_all_rules_pass(self):
        c = A.choose_timestep(results(slope=-20.0), eps_tau=0.2)
        self.assertEqual((c.timestep, c.rule), (0.02, "largest"))
        self.assertEqual(c.table[0]["failed"], [])

    def test_two_rules_fail_for_the_next_candidate(self):
        c = A.choose_timestep(results(acceptance=(0.985, 0.994, 0.998), slope=-100.0), eps_tau=0.2)
        self.assertEqual((c.timestep, c.rule), (0.01, "R1+R4"))

    def test_no_valid_time_step(self):
        c = A.choose_timestep(results(acceptance=(0.95, 0.97, 0.985), slope=-20.0), eps_tau=0.2)
        self.assertIsNone(c.timestep)
        self.assertEqual(c.rule, "R1")
        self.assertIn("F5", c.text)
        self.assertLess(c.tau_99, 0.005)
        c = A.choose_timestep(results(slope=-20.0, ratio=(1.3, 1.3, 1.3)), eps_tau=0.2)
        self.assertIsNone(c.timestep)
        self.assertEqual(c.rule, "R2")
        self.assertIn("Jastrow", c.text)

    def test_slope_that_is_not_resolved(self):
        # slope -20 +/- 46: the rule uses |slope| + 2 sigma = 113, so 0.02 fails R4 (0.34 > 0.2) and 0.01 passes (0.17)
        c = A.choose_timestep(results(slope=-20.0, error=0.5), eps_tau=0.2)
        self.assertEqual(c.slope_source, "measured, not resolved")
        self.assertAlmostEqual(c.slope_used, abs(c.slope) + 2.0 * c.slope_error, places=9)
        self.assertEqual((c.timestep, c.rule), (0.01, "R4"))

    def test_energy_that_is_not_linear_in_the_time_step(self):
        # the energies at 0.01 and 0.005 give a slope of -300; the energy at 0.02 is 2 kcal/mol above that line
        rows = results(slope=-300.0, error=0.05)
        rows[0]["energy"] += 2.0 / H
        c = A.choose_timestep(rows, eps_tau=0.3)
        self.assertAlmostEqual(c.slope, -300.0, places=3)                     # the two smallest time steps
        self.assertIn("left out of the slope", c.text); self.assertIn("0.02", c.text)
        self.assertEqual(c.timestep, 0.005)                                    # 0.15 x 300 x 0.005 = 0.225 < 0.3
        # with all three points on one line nothing is left out, and the error of the slope is smaller
        c3 = A.choose_timestep(results(slope=-300.0, error=0.05), eps_tau=0.3)
        self.assertNotIn("left out", c3.text)
        self.assertLess(c3.slope_error, 0.5 * c.slope_error)

    def test_missing_quantities_are_not_checked(self):
        rows = [{"timestep": t, "energy": -17.2, "error": 0.02 / H} for t in (0.02, 0.01, 0.005)]
        c = A.choose_timestep(rows, eps_tau=0.2)                               # an engine that gives energies only
        self.assertEqual((c.timestep, c.rule), (0.02, "largest"))
        self.assertEqual(c.not_checked, ["R1", "R2", "R3"])
        self.assertEqual([c.table[0][name] for name in ("R1", "R2", "R3")], ["not checked"] * 3)
        self.assertIsNone(c.tau_99)
        # nothing measured at all: the first guess of the slope decides, and this is recorded
        c = A.choose_timestep([{"timestep": t} for t in (0.02, 0.01, 0.005)], eps_tau=0.3)
        self.assertEqual((c.timestep, c.rule), (0.005, "R4"))
        self.assertEqual(c.slope_source, "first guess")
        self.assertIsNone(c.slope)
        self.assertIn("R4 (slope is a first guess)", c.not_checked)
        self.assertIn("first guess", c.text)

    def test_given_slope_and_unusable_candidate(self):
        c = A.choose_timestep([{"timestep": t} for t in (0.02, 0.01, 0.005)], eps_tau=0.2, slope=(-60.0, 5.0))
        self.assertEqual((c.timestep, c.rule, c.slope_source), (0.02, "largest", "given"))
        rows = results(slope=-20.0)
        rows[0]["usable"], rows[0]["reason"] = False, "the population drifts"
        c = A.choose_timestep(rows, eps_tau=0.2)
        self.assertEqual((c.timestep, c.rule), (0.01, "data"))
        self.assertIn("the population drifts", c.text)

    def test_bad_input(self):
        with self.assertRaises(ValueError):
            A.choose_timestep([], eps_tau=0.2)
        with self.assertRaises(ValueError):
            A.choose_timestep([{"timestep": 0.01}, {"timestep": 0.01}], eps_tau=0.2)
        with self.assertRaises(ValueError):
            A.choose_timestep([{"acceptance": 0.99}], eps_tau=0.2)
        with self.assertRaises(ValueError):
            A.choose_timestep(results(), eps_tau=0.0)


class TestCheckSettings(unittest.TestCase):
    GOOD = {"dmc": {"timestep": 0.005, "walkers": 2048, "steps": 500, "warmup_steps": 2000, "checkpoint": 50}}

    def test_good_settings_give_no_warning(self):
        self.assertEqual(A.check_settings(self.GOOD, ranks=64, blocks=400, sigma_target=0.5, e_scf=E_BIG), [])

    def test_one_warning_per_broken_rule(self):
        def warn(dmc=None, **kw):
            s = {"dmc": dict(self.GOOD["dmc"], **(dmc or {}))}
            return A.check_settings(s, **kw)
        w = warn({"walkers": 2000}, ranks=64)
        self.assertEqual(len(w), 1); self.assertIn("not a multiple", w[0]); self.assertIn("2048", w[0])
        w = warn({"timestep": 0.02, "steps": 125})
        self.assertEqual(len(w), 1); self.assertIn("first guess", w[0])
        w = warn({"timestep": 0.02, "steps": 125}, acceptance=0.9861)
        self.assertEqual(len(w), 1); self.assertIn("0.9861", w[0])
        self.assertEqual(warn({"timestep": 0.02, "steps": 125}, acceptance=0.9950), [])
        w = warn({"steps": 10}, blocks=400)                                    # 20 Ha^-1: 200 of 400 blocks are transient
        self.assertEqual(len(w), 1); self.assertIn("shorter than", w[0])
        w = warn(blocks=40)                                                    # 36 kept blocks
        self.assertEqual(len(w), 1); self.assertIn("36 of 40", w[0])
        w = warn(blocks=400, sigma_target=0.2, e_scf=E_BIG)                    # expected 0.26 kcal/mol
        self.assertEqual(len(w), 1); self.assertIn("above the wanted", w[0]); self.assertIn("first guesses", w[0])
        w = warn(blocks=400, sigma_target=0.2, variance=V_BIG, t_corr=TC_BIG)
        self.assertEqual(len(w), 1); self.assertNotIn("first guesses", w[0])
        w = warn({"checkpoint": -1})
        self.assertEqual(len(w), 1); self.assertIn("checkpoint", w[0])

    def test_defaults_of_the_input_writer_are_judged(self):
        w = A.check_settings({})                                               # nothing given: the writer uses checkpoint = -1
        self.assertEqual(len(w), 1); self.assertIn("checkpoint", w[0])
        w = A.check_settings({"dmc": {"checkpoint": 50}, "opt": {"nonlocalpp": "yes"}})
        self.assertEqual(len(w), 1); self.assertIn("obsolete", w[0])


class TestMeasurePoint(unittest.TestCase):
    def test_water_run(self):
        e, w = S.energy_weight(WATER)
        data = PointData(key="water", state="DONE", energy=e, weight=w, e_surrogate=-17.2371,
                         meta={"scalar_file": WATER, "vmc_scalar_file": WATER_VMC})
        m = A.measure_point(data, timestep=0.005, steps=100, walkers=2048, ranks=32)
        self.assertEqual((m["blocks"], m["drop"], m["drop_planned"], m["kept"]), (200, 20, 20, 180))
        self.assertAlmostEqual(m["energy"], -17.2401, places=4)
        self.assertAlmostEqual(m["acceptance"], 0.9973, places=4)
        self.assertAlmostEqual(m["variance"], 0.379, places=3)
        self.assertAlmostEqual(m["vmc_variance"], 0.3707, places=4)
        self.assertAlmostEqual(m["population_spread"], 0.009, delta=0.001)
        self.assertAlmostEqual(m["seconds_per_block"], 0.58, delta=0.01)
        self.assertAlmostEqual(m["step_seconds"], m["seconds_per_block"] * 32 / 204800.0, places=12)
        self.assertAlmostEqual(m["data_time"], 90.0, places=9)
        self.assertAlmostEqual(m["noise"], m["error"] ** 2 * 2048 * 90.0, places=12)
        self.assertTrue(0.10 < m["t_corr"] < 0.16, m["t_corr"])               # near the first guess of the package, 0.14
        self.assertEqual(m["findings"], [])
        self.assertEqual(m["not_measured"], [])
        self.assertEqual(m["mismatch"], [])
        json.dumps(m)
        # the plan that follows: 0.1 kcal/mol needs about 740 blocks of 2.5 Ha^-1 before the safety factor
        plan = A.plan_dmc(0.005, 2048, sigma_target=0.1, noise=m["noise"], safety=1.0, step_seconds=m["step_seconds"], ranks=32)
        self.assertTrue(600 < plan.blocks < 900, plan.blocks)
        self.assertAlmostEqual(plan.sigma_block, 2.7, delta=0.2)

    def test_engine_with_energies_only(self):
        r = np.random.default_rng(3)
        data = PointData(key="p", state="DONE", energy=-17.2 + 0.002 * r.normal(size=150), weight=np.ones(150))
        m = A.measure_point(data, timestep=0.01, steps=50, walkers=512)
        self.assertEqual(m["drop_planned"], 20)
        self.assertEqual(sorted(m["not_measured"]), ["acceptance", "population_spread", "seconds_per_block", "step_seconds",
                                                    "variance", "vmc_variance"])
        for name in ("acceptance", "variance", "vmc_variance", "population_spread", "seconds_per_block", "t_corr"):
            self.assertIsNone(m[name])
        self.assertGreater(m["noise"], 0.0)                                    # the noise is always measured

    def test_plain_numbers_in_meta_and_mismatch(self):
        r = np.random.default_rng(4)
        data = PointData(key="p", state="DONE", energy=-17.2 + 0.002 * r.normal(size=150), weight=np.ones(150),
                         meta={"acceptance": 0.996, "variance": 0.38, "vmc_variance": 0.37, "seconds_per_block": 2.0,
                               "ranks": 16, "timestep": 0.005})
        m = A.measure_point(data, timestep=0.01, steps=50, walkers=512)
        self.assertEqual((m["acceptance"], m["variance"], m["vmc_variance"]), (0.996, 0.38, 0.37))
        self.assertAlmostEqual(m["step_seconds"], 2.0 * 16 / (512 * 50), places=12)
        self.assertEqual(len(m["mismatch"]), 1)                                # the run reports another time step
        self.assertIn("timestep", m["mismatch"][0])

    def test_too_few_blocks(self):
        data = PointData(key="short", state="DONE", energy=np.zeros(25), weight=np.ones(25))
        with self.assertRaises(ValueError):
            A.measure_point(data, timestep=0.01, steps=50, walkers=512)


class QmcLike(MockEngine):
    """Mock engine that behaves like DMC in the settings of a point: the energy has a slope against the
    time step, the rejected fraction grows like (time step)^1.28, and the trace has all QMCPACK columns."""

    def __init__(self, *args, slope=SLOPE_BIG, rejected=REJECTED_BIG, variance=V_BIG, vmc_ratio=1.01, ranks=32, core_seconds=1e-4,
                 honour=True, report=True, bad_weight=(), fail=(), **kw):
        super().__init__(*args, **kw)
        self.slope, self.rejected, self.variance, self.vmc_ratio = slope, rejected, variance, vmc_ratio
        self.ranks, self.core_seconds, self.honour, self.report = ranks, core_seconds, honour, report
        self.bad_weight, self.fail = set(bad_weight), set(fail)
        self.specs = {}

    def submit(self, spec):
        if spec.key not in self.points:
            self.specs[spec.key] = spec
        super().submit(spec)

    def read(self, key):
        d = super().read(key)
        if key not in self.specs:
            return d
        if key in self.fail:
            return PointData(key=key, state=FAILED, message="the job ended with an error")
        if d.nblocks == 0:
            return d
        dmc = self.specs[key].settings["dmc"]
        n, total = d.nblocks, self.points[key]["blocks"]
        tau = dmc["timestep"] if self.honour else 0.005
        r = np.random.default_rng([zlib.crc32(key.encode()), 11])
        e = d.energy + self.slope * tau / H
        spread = 0.10 if key in self.bad_weight else 0.01                      # 10 % is above the limit of the population monitor
        weight = dmc["walkers"] * dmc["steps"] * (1.0 + spread * r.normal(size=total))[:n]
        block_seconds = self.core_seconds * dmc["walkers"] * dmc["steps"] / self.ranks
        trace = {"LocalEnergy": e, "LocalEnergy_sq": e * e + self.variance * (1.0 + 0.02 * r.normal(size=total))[:n],
                 "BlockWeight": weight, "BlockCPU": block_seconds * (1.0 + 0.02 * r.normal(size=total))[:n],
                 "AcceptRatio": 1.0 - self.rejected * (tau / 0.005) ** 1.28 + 2e-5 * r.normal(size=total)[:n]}
        d.energy, d.weight = e, weight
        d.meta.update({"trace": trace, "vmc_variance": self.variance / self.vmc_ratio, "ranks": self.ranks})
        if self.report:
            d.meta.update({"timestep": dmc["timestep"], "steps": dmc["steps"], "walkers": dmc["walkers"]})
        return d


USER = {"charge": 0, "spin": 0, "xc": "b3lyp", "jastrow_reuse": "centre",
        "dmc": {"timestep": 0.005, "walkers": 2048, "steps": 500, "warmup_steps": 2000}}


class TestPilot(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.x = M.water_geometry(r=0.9619, theta=105.0)
        self.symbols = ["O", "H", "H"]

    def tearDown(self):
        self.tmp.cleanup()

    def folder(self, name="pilot"):
        return os.path.join(self.tmp.name, name)

    def engine(self, sigma=0.2, transient=20.0, equil_blocks=2.0, **kw):
        return QmcLike(M.water(True), M.water(False), sigma_block=sigma, blocks_per_tick=60, seed=5, transient=transient,
                       equil_blocks=equil_blocks, e_offset_hartree=E_BIG, **kw)

    def pilot(self, engine, name="pilot", **options):
        options.setdefault("ranks", 32)
        return A.Pilot(engine, self.symbols, self.x, USER, self.folder(name), **options)

    # ------------------------------------------------------------------ mode fixed
    def test_fixed_mode_changes_nothing(self):
        user = dict(USER, dmc=dict(USER["dmc"], walkers=2000))
        p = A.Pilot(None, self.symbols, self.x, user, self.folder(), mode="fixed", ranks=32, final_blocks=400,
                    e_scf=-17.2371)
        self.assertEqual(p.tick(), A.DONE)
        s = p.settings()
        record = s.pop("autotune")
        self.assertEqual(s, user)                                              # the user's values, unchanged
        self.assertEqual(record["mode"], "fixed")
        self.assertEqual(len(record["warnings"]), 2)                           # walkers not a multiple of 32; no checkpoint
        self.assertTrue(any("not a multiple" in w for w in record["warnings"]))
        self.assertEqual(p.warnings, record["warnings"])
        self.assertEqual(record["forecast_first_guess"]["n_electrons"], 8)
        self.assertEqual(record["analysis"]["discard_blocks"], 4)
        self.assertEqual(p.tick(), A.DONE)                                     # a second pass does nothing

    def test_options_are_checked(self):
        with self.assertRaises(ValueError):
            A.Pilot(None, self.symbols, self.x, USER, self.folder("a"), mode="fixed", rank=32)      # misspelled option
        with self.assertRaises(ValueError):
            A.Pilot(None, self.symbols, self.x, USER, self.folder("b"), mode="automatic")
        with self.assertRaises(ValueError):
            A.Pilot(None, self.symbols, self.x, USER, self.folder("c"), mode="auto")                # a pilot needs an engine
        with self.assertRaises(ValueError):
            A.Pilot(None, ["O", "H"], self.x, USER, self.folder("d"), mode="fixed")

    # ------------------------------------------------------------------ mode auto
    def test_auto_on_the_mock_engine_ends_with_a_settings_file(self):
        # the plain mock engine gives block energies only: no acceptance, no variance, no timing
        eng = MockEngine(M.water(True), M.water(False), sigma_block=1.0, blocks_per_tick=50, seed=3, transient=20.0,
                         equil_blocks=2.0, e_offset_hartree=-17.2)
        p = self.pilot(eng, n_directions=3)
        self.assertEqual(p.run(max_ticks=100), A.DONE)
        self.assertTrue(os.path.exists(os.path.join(self.folder(), "qmc_settings.json")))
        s = p.settings()
        a = s["autotune"]
        tau = s["dmc"]["timestep"]
        self.assertEqual(tau, 0.02)                                            # the mock has no time-step error
        self.assertEqual(a["choices"]["timestep"]["rule"], "largest")
        self.assertEqual(s["dmc"], {"timestep": 0.02, "walkers": 2048, "steps": 125, "warmup_steps": 500})
        self.assertEqual(s["jastrow"], {"j1_size": 5, "j1_rcut": 4.9, "j2_size": 7, "j2_rcut": 6.9})
        self.assertEqual({k: s[k] for k in ("charge", "spin", "xc", "jastrow_reuse")},
                         {k: USER[k] for k in ("charge", "spin", "xc", "jastrow_reuse")})
        # every chosen value has its basis; what was not measured is said
        for name in ("timestep", "steps", "warmup_steps", "walkers", "discard_blocks", "jastrow", "final_blocks"):
            self.assertIn("basis", a["choices"][name]); self.assertIn("value", a["choices"][name])
        self.assertEqual(a["not_checked"], ["R1", "R2", "R3"])
        self.assertEqual(len(a["guesses"]), 2)                                 # variance and time per step are first guesses
        self.assertEqual(len(a["measurements"]["points"]), 3)
        self.assertGreater(a["measurements"]["noise"], 0.0)
        self.assertEqual(a["plan"]["final"]["blocks"], a["choices"]["final_blocks"]["value"])
        self.assertEqual(a["plan"]["final"]["discard"], 4)
        self.assertEqual(a["plan"]["scan"]["n_directions"], 3)
        self.assertAlmostEqual(a["plan"]["scan"]["sigma_point"], A.scan_point_sigma(3), places=9)
        self.assertLess(eng.total_blocks(), 3 * 400)                           # the runs were stopped early: short pilot
        with open(os.path.join(self.folder(), "pilot.log")) as fh:
            self.assertIn("DONE", fh.read())

    def test_pilot_continues_from_its_files(self):
        def make(folder):
            eng = MockEngine(M.water(True), M.water(False), sigma_block=1.0, blocks_per_tick=50, seed=3, transient=20.0,
                             equil_blocks=2.0, e_offset_hartree=-17.2, state_file=os.path.join(self.folder(folder), "engine.json"))
            return eng, A.Pilot(eng, self.symbols, self.x, USER, self.folder(folder), ranks=32, pilot_time=100.0)
        os.makedirs(self.folder("one")); os.makedirs(self.folder("two"))
        eng, p = make("one")
        self.assertEqual(p.run(max_ticks=100), A.DONE)
        eng, p2 = make("two")
        eng.advance(); self.assertEqual(p2.tick(), A.RUNNING)
        eng.advance(); self.assertEqual(p2.tick(), A.RUNNING)
        eng, p3 = make("two")                                                  # new objects: the state comes from the files
        self.assertEqual(p3.status, A.RUNNING)
        self.assertEqual(p3.run(max_ticks=100), A.DONE)
        a, b = p.settings(), p3.settings()
        self.assertEqual(a["dmc"], b["dmc"])
        self.assertEqual(a["autotune"]["measurements"], b["autotune"]["measurements"])
        eng, p4 = make("two")
        self.assertEqual(p4.tick(), A.DONE)                                    # a finished pilot stays finished

    def test_auto_with_a_time_step_error(self):
        # slope -300 kcal/mol per Ha^-1, acceptance 0.9917 / 0.9966 / 0.9986: the worked case
        p = self.pilot(self.engine(), eps_tau=0.3)
        self.assertEqual(p.run(max_ticks=100), A.DONE)
        s = p.settings(); a = s["autotune"]; m = a["measurements"]
        self.assertEqual(s["dmc"], {"timestep": 0.005, "walkers": 2048, "steps": 500, "warmup_steps": 2000})
        self.assertEqual(a["choices"]["timestep"]["rule"], "R4")
        self.assertAlmostEqual(m["slope"], SLOPE_BIG, delta=3.0 * m["slope_error"] + 1e-9)
        self.assertLess(m["slope_error"], 0.25 * abs(SLOPE_BIG))
        self.assertAlmostEqual(m["acceptance"], 1.0 - REJECTED_BIG, places=4)
        self.assertAlmostEqual(m["acceptance_exponent"], 1.28, delta=0.02)
        self.assertAlmostEqual(m["variance"], V_BIG, delta=0.02)
        self.assertAlmostEqual(m["step_seconds"], 1e-4, delta=2e-6)            # measured, not guessed
        self.assertEqual(a["guesses"], [])
        self.assertEqual(a["not_checked"], [])
        self.assertEqual(a["repeats"], [])
        final = a["plan"]["final"]
        self.assertAlmostEqual(final["core_hours"], final["blocks"] * 500 * 2048 * m["step_seconds"] / 3600.0, places=6)
        self.assertAlmostEqual(final["wall_seconds"], 1.3 * final["core_hours"] * 3600.0 / 32, places=3)
        self.assertLessEqual(final["sigma_expected"], 0.2)

    def test_auto_bias_limit_below_the_candidates(self):
        p = self.pilot(self.engine(), eps_tau=0.2)                             # limit 0.0044, below 0.005
        self.assertEqual(p.run(max_ticks=100), A.DONE)
        s = p.settings()
        self.assertEqual(s["autotune"]["choices"]["timestep"]["rule"], "F6")
        self.assertAlmostEqual(s["dmc"]["timestep"], 0.0044, delta=0.0002)
        self.assertEqual(s["dmc"]["steps"], A.plan_dmc(s["dmc"]["timestep"], 2048, blocks=100, noise=1.0).steps)
        self.assertTrue(any("no pilot run exists at the chosen time step" in w for w in s["autotune"]["warnings"]))

    def test_auto_stops_when_no_time_step_is_valid(self):
        p = self.pilot(self.engine(rejected=0.02, slope=-20.0))                # acceptance 0.98 at 0.005
        self.assertEqual(p.run(max_ticks=100), A.NEEDS_ATTENTION)
        self.assertIn("R1", p.message)
        self.assertIsNone(p.settings())
        self.assertFalse(os.path.exists(os.path.join(self.folder(), "qmc_settings.json")))

    def test_auto_refuses_runs_that_ignore_the_time_step(self):
        # an engine that runs every point at the same time step and does not say so
        p = self.pilot(self.engine(honour=False, report=False))
        self.assertEqual(p.run(max_ticks=100), A.NEEDS_ATTENTION)
        self.assertIn("requested time steps", p.message)
        # an engine that reports the settings it used: the mismatch is named
        eng = self.engine()
        p = self.pilot(eng, name="reported")
        real_read = eng.read

        def read(key):
            d = real_read(key)
            if "timestep" in d.meta:
                d.meta["timestep"] = 0.005
            return d
        eng.read = read
        self.assertEqual(p.run(max_ticks=100), A.NEEDS_ATTENTION)
        self.assertIn("did not use the requested settings", p.message)

    def test_auto_reports_a_failed_run(self):
        p = self.pilot(self.engine(fail=["pilot/r1/tau0.01"]))
        self.assertEqual(p.run(max_ticks=100), A.NEEDS_ATTENTION)
        self.assertIn("pilot/r1/tau0.01", p.message)
        self.assertIn("the job ended with an error", p.message)

    def test_auto_excludes_a_run_with_a_bad_population(self):
        p = self.pilot(self.engine(bad_weight=["pilot/r1/tau0.02"], slope=-20.0))
        self.assertEqual(p.run(max_ticks=100), A.DONE)
        s = p.settings()
        self.assertEqual(s["dmc"]["timestep"], 0.01)
        self.assertEqual(s["autotune"]["choices"]["timestep"]["rule"], "data")
        self.assertTrue(any("block weight" in w for w in s["autotune"]["warnings"]))

    # ------------------------------------------------------------------ mode iterative
    def test_iterative_repeats_with_smaller_time_steps(self):
        eng = self.engine(rejected=0.02, slope=-20.0)
        p = self.pilot(eng, mode="iterative")
        self.assertEqual(p.run(max_ticks=100), A.DONE)
        s = p.settings(); a = s["autotune"]
        self.assertEqual(s["dmc"]["timestep"], 0.002)                          # F5 gave 0.0029; the repeat ran 0.002 and 0.001
        self.assertEqual(len(a["repeats"]), 1)
        self.assertIn("M7", a["repeats"][0]["reason"])
        self.assertEqual(len(a["measurements"]["points"]), 5)
        self.assertIn("pilot/r2/tau0.002", eng.points)

    def test_iterative_runs_a_failed_point_again(self):
        eng = self.engine(fail=["pilot/r1/tau0.01"], slope=-20.0)
        p = self.pilot(eng, mode="iterative")
        self.assertEqual(p.run(max_ticks=100), A.DONE)
        a = p.settings()["autotune"]
        self.assertEqual(len(a["repeats"]), 1)
        self.assertIn("pilot/r2/tau0.01", eng.points)
        self.assertEqual(sorted(pt["key"] for pt in a["measurements"]["points"]),
                         ["pilot/r1/tau0.005", "pilot/r1/tau0.02", "pilot/r2/tau0.01"])

    def test_iterative_repeats_a_run_with_a_bad_population(self):
        eng = self.engine(bad_weight=["pilot/r1/tau0.02"], slope=-20.0)
        p = self.pilot(eng, mode="iterative")
        self.assertEqual(p.run(max_ticks=100), A.DONE)
        s = p.settings(); a = s["autotune"]
        self.assertEqual(s["dmc"]["timestep"], 0.02)                           # the new run at 0.02 is healthy
        self.assertEqual(len(a["repeats"]), 1)
        self.assertIn("pilot/r2/tau0.02", eng.points)
        self.assertFalse(any("block weight" in w for w in a["warnings"]))      # the warning of the replaced run is gone

    def test_iterative_doubles_the_transient(self):
        # a start-up part of about 60 blocks = 30 Ha^-1: not over after the planned 10 Ha^-1, nor after 20 Ha^-1
        eng = self.engine(slope=-20.0, transient=100.0, equil_blocks=12.0)
        p = self.pilot(eng, mode="iterative", max_repeats=2)
        self.assertEqual(p.run(max_ticks=200), A.DONE)
        s = p.settings(); a = s["autotune"]
        self.assertEqual(len(a["repeats"]), 2)
        self.assertTrue(all("M3" in r["reason"] for r in a["repeats"]))
        self.assertEqual(a["analysis"]["transient"], 40.0)                     # 10 -> 20 -> 40 Ha^-1
        self.assertEqual(a["analysis"]["discard_blocks"], 16)
        tau = s["dmc"]["timestep"]
        self.assertEqual(s["dmc"]["warmup_steps"], int(round(40.0 / tau)))
        self.assertEqual(eng.specs["pilot/r3/tau0.02"].settings["dmc"]["warmup_steps"], 2000)

    def test_auto_discards_more_when_the_transient_is_long(self):
        eng = self.engine(slope=-20.0, transient=100.0, equil_blocks=12.0)
        p = self.pilot(eng)
        self.assertEqual(p.run(max_ticks=200), A.DONE)
        a = p.settings()["autotune"]
        self.assertEqual(a["repeats"], [])
        self.assertEqual(a["analysis"]["transient"], 10.0)                     # mode auto does not change the plan
        self.assertTrue(any("more than the planned 20" in w for w in a["warnings"]))
        for point in a["measurements"]["points"]:
            self.assertGreater(point["drop"], point["drop_planned"])

    def test_runs_go_to_their_full_length_without_an_early_stop(self):
        eng = self.engine()
        p = self.pilot(eng, eps_tau=0.3, pilot_time=50.0, pilot_min_blocks=1000)
        self.assertEqual(p.run(max_ticks=100), A.DONE)
        self.assertIsNone(p.state["rounds"][0]["stopped_early"])
        for key in eng.points:
            d = eng.read(key)
            self.assertEqual((d.state, d.nblocks), ("DONE", 120))              # 20 blocks of transient and 100 of data
        self.assertEqual(p.settings()["dmc"]["timestep"], 0.005)

    def test_iterative_gives_up_after_max_repeats(self):
        # the DMC variance is 1.3 times the VMC variance at every time step, also after new optimizations
        eng = self.engine(vmc_ratio=1.3, slope=-20.0)
        p = self.pilot(eng, mode="iterative", max_repeats=2)
        self.assertEqual(p.run(max_ticks=100), A.NEEDS_ATTENTION)
        self.assertEqual(p.state["repeats"], 2)
        self.assertIn("R2", p.message)
        doubled = eng.specs["pilot/r2/tau0.02"].settings["opt"]["samples"]
        self.assertEqual(doubled, 200000)                                      # twice the default of 100000
        self.assertEqual(eng.specs["pilot/r3/tau0.02"].settings["opt"]["samples"], 200000)

    # ------------------------------------------------------------------ what the points receive
    def test_points_get_role_settings_and_keys(self):
        eng = self.engine(slope=-20.0)
        p = self.pilot(eng)
        self.assertEqual(p.run(max_ticks=100), A.DONE)
        self.assertEqual(sorted(eng.specs), ["pilot/r1/tau0.005", "pilot/r1/tau0.01", "pilot/r1/tau0.02"])
        for key, spec in eng.specs.items():
            tau = float(key.split("tau")[1])
            self.assertEqual(spec.role, "pilot")
            self.assertEqual((spec.series, spec.base_key), (1, None))
            self.assertEqual(spec.settings["dmc"], {"timestep": tau, "steps": int(round(0.5 / tau)),
                                                    "warmup_steps": int(round(10.0 / tau)), "walkers": 512})
            self.assertEqual(spec.settings["jastrow"]["j1_rcut"], 4.9)
            self.assertEqual(spec.settings["xc"], "b3lyp")
            self.assertEqual(spec.blocks, 20 + 1600)                           # 10 Ha^-1 of transient and 800 Ha^-1 of data

    def test_shared_wave_function(self):
        eng = self.engine(slope=-20.0)
        p = self.pilot(eng, share_wavefunction=True)
        eng.advance(); p.tick()
        self.assertEqual(sorted(eng.specs), ["pilot/r1/tau0.02"])              # the further series wait for the first run
        self.assertEqual(p.run(max_ticks=100), A.DONE)
        self.assertEqual([(eng.specs[k].series, eng.specs[k].base_key) for k in sorted(eng.specs)],
                         [(3, "pilot/r1/tau0.02"), (2, "pilot/r1/tau0.02"), (1, None)])

    def test_user_jastrow_and_untested_element(self):
        user = dict(USER, jastrow={"j1_rcut": 5.0, "j1_size": 5})
        eng = QmcLike(lambda x: 0.0, lambda x: 0.0, sigma_block=0.2, blocks_per_tick=60, seed=5, e_offset_hartree=-24.0,
                      slope=-20.0)
        p = A.Pilot(eng, ["F", "H"], np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.92]]), user, self.folder(), ranks=32)
        self.assertEqual(p.run(max_ticks=100), A.DONE)
        s = p.settings()
        self.assertEqual(s["jastrow"], dict(DEFAULT_JASTROW, j1_rcut=5.0, j1_size=5))
        self.assertEqual(s["autotune"]["choices"]["jastrow"]["source"], "user")
        self.assertTrue(any("outside the tested set" in w for w in s["autotune"]["warnings"]))


if __name__ == "__main__":
    unittest.main()
