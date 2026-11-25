"""The classic line search: fit rule, window rules, the sizer, and the driver on model surfaces (mock engine)."""
import json
import os
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

from montepass import geometry as G, model as M, subspace as S  # noqa: E402
from montepass.engine import FAILED  # noqa: E402
from montepass.engine.mock import MockEngine  # noqa: E402
from montepass.search import classic as C  # noqa: E402
from montepass.search.classic import ClassicLineSearch  # noqa: E402
from montepass.surrogate.analytic import AnalyticSurrogate  # noqa: E402
from montepass.surrogate.base import Surrogate  # noqa: E402
from montepass.units import HARTREE_TO_KCAL  # noqa: E402


def polyfit_step(t, e, sign=1):
    """The step rule written a second time with numpy.polyfit and numpy.roots (as a reference in the tests):
    real roots of the derivative with the wanted sign of the second derivative, the one closest to zero;
    without such a root the end of the grid with the lower (sign * energy)."""
    pf = np.polyfit(t, e, 3)
    pfd = np.polyder(pf)
    r = np.roots(pfd)
    d = np.polyval(np.polyder(pfd), r)
    x = r[(r.imag == 0) & (sign * d > 0)].real
    if len(x):
        return float(x[np.argmin(np.abs(x))])
    ends = np.array([min(t), max(t)])
    return float(ends[np.argmin(sign * np.polyval(pf, ends))])


def water_case():
    """True surface, surrogate surface, start geometry (surrogate minimum), true minimum, directions."""
    ft, fs = M.water(True), M.water(False)
    xt = M.minimize(ft, M.water_geometry())
    xs = M.minimize(fs, M.water_geometry())
    Hs = M.hessian(fs, xs); sub = S.mode_subspace(Hs, xs)
    d = S.search_directions(S.project_hessian(Hs, sub.basis), sub)
    return ft, fs, xs, xt, d


def make(case, tmp, seed=1, sigma_block=10.0, classic=None, with_sigma=True, name="geo", engine=None, surrogate="analytic"):
    ft, fs, xs, xt, d = case
    eng = engine or MockEngine(ft, fs, sigma_block=sigma_block, blocks_per_tick=20, seed=seed, transient=20.0, equil_blocks=0.6)
    settings = {"classic": dict(classic or {})}
    if with_sigma:
        settings["sigma_block"] = sigma_block
    sur = AnalyticSurrogate(fs) if surrogate == "analytic" else surrogate
    return eng, ClassicLineSearch(name, eng, sur, list(ft.symbols), xs, d.vectors, d.curvature, settings=settings, state_dir=tmp)


def line_case(curve):
    """A model with one search direction: two atoms, and an energy that depends only on the displacement
    along the unit vector of their stretch."""
    x0 = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.2]])
    v = np.array([0.0, 0.0, -1.0, 0.0, 0.0, 1.0]) / np.sqrt(2.0)

    def surface(x):
        return float(curve(float(np.dot(np.asarray(x, dtype=float).reshape(-1) - x0.reshape(-1), v))))

    return x0, v[None, :], surface


class TestFitRule(unittest.TestCase):
    def test_same_step_as_polyfit_and_roots(self):
        rng = np.random.default_rng(7)
        seen = set()
        for trial in range(600):
            R = rng.uniform(0.02, 1.2)
            t = np.linspace(-R, R, 7)
            sign = 1 if trial % 2 == 0 else -1
            kind = trial % 4
            if kind == 0:
                e = sign * 150.0 * (t - rng.uniform(-2, 2) * R) ** 2 + rng.normal(size=7) * rng.uniform(0, 5)
            elif kind == 1:
                e = rng.normal(size=7) * 3.0
            elif kind == 2:
                e = 50.0 * t + 30.0 * t ** 3 + rng.normal(size=7) * 0.1
            else:
                e = -59000.0 + sign * 110.0 * (1 - np.exp(-2.0 * (t - 0.3 * R))) ** 2 + rng.normal(size=7) * 0.5
            r = C.fit_step(t, e, sign=sign)
            ref = polyfit_step(t, e, sign)
            self.assertAlmostEqual(r["m"], ref, delta=1e-7 * max(1.0, abs(ref)))
            seen.add(r["status"])
        self.assertEqual(seen, {C.INSIDE, C.OUTSIDE, C.NO_ROOT})

    def test_shrink_factor_of_a_quartic_curve(self):
        """E = Lam (x - d)^2 + c4 (x - d)^4 on a 7-point grid of half-width R: the fitted step is the true
        offset d times s = 1 / (1 + 1.0635 (R / L4)^2), L4 = sqrt(Lam / c4). The step is biased and the
        bracket check sees nothing."""
        lam, R = 100.0, 0.40
        t = np.linspace(-R, R, 7)
        for ratio, expected in ((0.5, 0.790), (1.0, 0.485), (2.0, 0.190), (3.0, 0.095)):
            c4 = lam * (ratio / R) ** 2
            d = 0.01 * R
            r = C.fit_step(t, lam * (t - d) ** 2 + c4 * (t - d) ** 4)
            self.assertAlmostEqual(r["m"] / d, 1.0 / (1.0 + 1.0635 * ratio ** 2), delta=2e-4)
            self.assertAlmostEqual(r["m"] / d, expected, delta=6e-4)
            self.assertEqual(r["status"], C.INSIDE)

    def test_step_is_never_clipped_and_the_bracket_check_records_it(self):
        lam, R = 100.0, 0.40
        t = np.linspace(-R, R, 7)
        r = C.fit_step(t, lam * (t - 5 * R) ** 2)                      # exact parabola, minimum at 5 R
        self.assertAlmostEqual(r["m"], 2.0, places=8)
        self.assertEqual((r["status"], r["root"], r["inside"]), (C.OUTSIDE, True, False))
        r = C.fit_step(t, 110.0 * (1 - np.exp(-2.0 * (t + 0.60))) ** 2)      # Morse curve, minimum at -0.60 A
        self.assertAlmostEqual(r["m"], -1.4565, delta=5e-4)            # extrapolated far past the minimum
        self.assertEqual(r["status"], C.OUTSIDE)
        e = 50.0 * t + 30.0 * t ** 3                                    # rises everywhere
        r = C.fit_step(t, e)
        self.assertEqual((r["m"], r["status"], r["root"]), (-R, C.NO_ROOT, False))    # the end with the lower energy
        r = C.fit_step(t, e, sign=-1)
        self.assertEqual((r["m"], r["status"]), (R, C.NO_ROOT))                       # the end with the higher energy

    def test_maximum_along_negative_curvature(self):
        t = np.linspace(-0.3, 0.3, 7)
        e = 7.0 - 0.5 * 180.0 * (t - 0.05) ** 2
        r = C.fit_step(t, e, sign=-1)
        self.assertAlmostEqual(r["m"], 0.05, places=9)
        self.assertEqual(r["status"], C.INSIDE)
        self.assertAlmostEqual(r["k"], -180.0, places=6)
        self.assertAlmostEqual(r["gain"], 0.5 * 180.0 * 0.05 ** 2, places=8)        # the fitted energy rises to the maximum
        r = C.fit_step(t, e, sign=+1)                                    # asked for a minimum: an exact parabola has none
        self.assertEqual((r["status"], r["m"]), (C.NO_ROOT, -0.3))       # the end of the grid with the lower energy
        r = C.fit_step(t, e + 40.0 * t ** 3, sign=+1)                    # with a cubic term the minimum exists, far outside
        self.assertEqual(r["status"], C.OUTSIDE)
        self.assertAlmostEqual(r["m"], (180.0 + np.sqrt(180.0 ** 2 - 4 * 120.0 * 9.0)) / 240.0, places=8)      # 1.448 A

    def test_error_bar_is_one_standard_error(self):
        """Parabola, cubic fit, 7 points: sigma_x0 = 1.086 sigma / sqrt(W k). The second number is the 95 %
        half-width that the old method printed; it is about twice as large. Model values: force constant k
        (kcal/mol/A^2), window W (kcal/mol) and noise sigma of one point (kcal/mol)."""
        for k, W, sigma in ((300.0, 25.0, 0.8), (1600.0, 400.0, 2.5)):
            R = C.window_half_width(W, k)
            t = np.linspace(-R, R, 7)
            r = C.fit_step(t, 0.5 * k * t ** 2, errors=np.full(7, sigma), resamples=8000, rng=np.random.default_rng(1))
            expected = 1.086 * sigma / np.sqrt(W * k)
            self.assertAlmostEqual(r["sm"] / expected, 1.0, delta=0.05)
            self.assertAlmostEqual(r["sm95"] / r["sm"], 1.96, delta=0.15)
            self.assertEqual(r["inside_fraction"], 1.0)
        r = C.fit_step(t, 0.5 * k * t ** 2)                              # no error bars given
        self.assertEqual((r["sm"], r["sm95"]), (0.0, 0.0))

    def test_bad_input_is_refused(self):
        t = np.linspace(-0.1, 0.1, 7)
        with self.assertRaises(ValueError):
            C.fit_step(t[:3], t[:3] ** 2)
        with self.assertRaises(ValueError):
            C.fit_step(t, np.array([0.0, 1.0, np.nan, 0.0, 1.0, 2.0, 3.0]))
        with self.assertRaises(ValueError):
            C.fit_step(np.zeros(7), t)
        with self.assertRaises(ValueError):
            C.fit_polynomial(t, t ** 2, 2, sigma=np.zeros(7))


class TestWindowRules(unittest.TestCase):
    # six model force constants in kcal/mol/A^2. The expected numbers in this class are the output of the
    # package functions for these model inputs (computed once and written here), unless a comment gives a formula.
    K = [300.0, 700.0, 1300.0, 1600.0, 2300.0, 2700.0]

    def test_scan_half_width(self):
        self.assertAlmostEqual(C.scan_half_width(300.0), 1.2026, places=4)
        self.assertAlmostEqual(C.scan_half_width(-300.0), 1.2026, places=4)       # the sign is ignored
        self.assertAlmostEqual(C.scan_half_width(2700.0), 0.6943, places=4)
        # the window of the whole scan is 0.5 sqrt(Lambda) hartree, Lambda = 1300 / 627.509474 hartree/A^2
        k = 1300.0
        self.assertAlmostEqual(0.5 * k * C.scan_half_width(k) ** 2 / HARTREE_TO_KCAL, 0.719667, places=6)
        with self.assertRaises(ValueError):
            C.scan_half_width(0.0)

    def test_window_half_width(self):
        # model force constants (kcal/mol/A^2), windows (kcal/mol) and the grid half-widths (A) that the package
        # function gives for them: R = sqrt(2 W / k)
        for k, w, r in ((300.0, 10.0, 0.2582), (700.0, 40.0, 0.3381), (1300.0, 450.0, 0.8321)):
            self.assertAlmostEqual(C.window_half_width(w, k), r, places=4)
        # a window of zero is replaced by 0.0001 hartree
        self.assertAlmostEqual(C.window_half_width(0.0, 0.5 * HARTREE_TO_KCAL), np.sqrt(2 * 1e-4 / 0.5), places=12)

    def test_default_tolerance(self):
        eps = [C.default_tolerance(k) for k in self.K]
        np.testing.assert_allclose(eps, [0.01826, 0.01195, 0.00877, 0.00791, 0.00659, 0.00609], atol=5e-5)
        k = 400.0                                                        # the harmonic energy at epsilon is 0.05 kcal/mol
        self.assertAlmostEqual(0.5 * k * C.default_tolerance(k) ** 2, 0.05, places=12)

    def test_blocks_for_noise(self):
        self.assertEqual(C.blocks_for_noise(2.0, 0.5, 50, 30, 400), (80, False))       # 16 blocks needed, lower limit 30
        self.assertEqual(C.blocks_for_noise(6.3, 0.42, 50, 30, 400), (275, False))     # (6.3 / 0.42)^2 = 225
        self.assertEqual(C.blocks_for_noise(6.3, 0.2, 50, 30, 400), (400, True))       # 993 needed: upper limit
        self.assertEqual(C.blocks_for_noise(6.3, 0.0, 50, 30, 400), (400, True))
        self.assertEqual(C.blocks_for_noise(2.0, 0.25, 0, 8, 400), (64, False))

    def test_min_distance(self):
        self.assertAlmostEqual(C.min_distance(M.water_geometry(0.96, 104.5)), 0.96, places=12)
        self.assertEqual(C.min_distance(np.zeros((1, 3))), float("inf"))


class TestSizer(unittest.TestCase):
    """The reference numbers come from the old library, run on the same curves with 5 of its own random
    seeds and 500 noise draws. Its choice sits on a mesh, so different seeds give a few discrete values.
    With the same noise draws this module and the old library agree to rounding (checked outside the tests)."""

    @staticmethod
    def scan(curve, k):
        R = C.scan_half_width(k)
        t = np.linspace(-R, R, 17)
        return t, np.array([float(curve(x)) for x in t])

    def test_quadratic_curve(self):
        k = 400.0
        t, e = self.scan(lambda x: 0.5 * k * x * x, k)
        eps = C.default_tolerance(k)                                      # 0.0158 A
        self.assertAlmostEqual(0.5 * k * C.scan_half_width(k) ** 2, 250.501, places=3)     # window of the whole scan
        for seed in range(3):
            # settings of the old library: one scale, 10 x 10 mesh.
            # Old library, 5 seeds: window 222.67 or 250.50 kcal/mol, noise 2.088 or 2.435 kcal/mol
            r = C.size_direction(t, e, k, eps, scales=(1,), mesh=10, rng=np.random.default_rng(seed))
            self.assertGreater(r["window"], 222.6); self.assertLess(r["window"], 250.6)
            self.assertGreater(r["noise"], 2.08); self.assertLess(r["noise"], 2.44)
            self.assertLess(r["error"], eps)
            self.assertLess(r["bias"], 1e-9)                              # a parabola has no fit bias
            self.assertAlmostEqual(r["half_width"], np.sqrt(2 * r["window"] / k), places=12)
            # the error of a parabola is the noise term alone: 1.96 * 1.086 sigma / sqrt(W k) < epsilon
            self.assertLess(1.96 * 1.086 * r["noise"] / np.sqrt(r["window"] * k), 1.05 * eps)
            # the default settings: four scales, 25 x 25 mesh.
            # Old library, 5 seeds: window 219.19, 229.63 or 240.06 kcal/mol, noise 2.088 kcal/mol
            r = C.size_direction(t, e, k, eps, rng=np.random.default_rng(seed))
            self.assertGreater(r["window"], 208.0); self.assertLess(r["window"], 250.6)
            self.assertGreater(r["noise"], 1.8); self.assertLess(r["noise"], 2.44)
            self.assertEqual(r["scale"], 1.0)

    def test_morse_curve(self):
        k, a = 1100.0, 2.2                                               # the size of a water stretch
        de = k / (2 * a * a)
        t, e = self.scan(lambda x: de * (1 - np.exp(-a * x)) ** 2, k)
        eps = C.default_tolerance(k)                                      # 0.0095 A
        for seed in range(3):
            # Old library, one scale, 5 seeds: window 46.157 kcal/mol every time, noise 0.865 or 1.010 kcal/mol
            r = C.size_direction(t, e, k, eps, scales=(1,), mesh=10, rng=np.random.default_rng(seed))
            self.assertAlmostEqual(r["window"], 46.157, delta=0.001)      # one ninth of the scan window 415.41
            self.assertGreater(r["noise"], 0.86); self.assertLess(r["noise"], 1.02)
            self.assertLess(r["error"], eps)
            self.assertLess(abs(r["x_ref"]), 5e-4)                         # the scan is centred on the minimum
            # Old library, four scales, 5 seeds: window 51.93 or 69.23 kcal/mol, noise 0.865 or 0.974 kcal/mol
            r = C.size_direction(t, e, k, eps, rng=np.random.default_rng(seed))
            self.assertGreater(r["window"], 46.0); self.assertLess(r["window"], 70.0)
            self.assertGreater(r["noise"], 0.75); self.assertLess(r["noise"], 1.02)
            self.assertAlmostEqual(r["half_width"], np.sqrt(2 * r["window"] / k), places=12)
            # the window is far narrower than for the parabola of test_quadratic_curve: the bias limits it
            self.assertLess(r["half_width"], 0.36)

    def test_same_seed_same_result_and_refinement(self):
        k = 220.0
        t, e = self.scan(lambda x: 0.5 * k * x * x - 50.0 * x ** 3 + 100.0 * x ** 4, k)
        eps = C.default_tolerance(k)
        a = C.size_direction(t, e, k, eps, rng=np.random.default_rng(11))
        b = C.size_direction(t, e, k, eps, rng=np.random.default_rng(11))
        self.assertEqual((a["window"], a["noise"], a["scale"]), (b["window"], b["noise"], b["scale"]))
        self.assertEqual([tr["scale"] for tr in a["trials"]], [1.0, 4.0, 16.0, 64.0])
        coarse = C.size_direction(t, e, k, eps, scales=(1,), mesh=10, refine=False, rng=np.random.default_rng(11))
        fine = C.size_direction(t, e, k, eps, scales=(1,), mesh=10, refine=True, rng=np.random.default_rng(11))
        self.assertGreaterEqual(fine["noise"], coarse["noise"])           # refinement never lowers the noise

    def test_maximum_of_a_double_well(self):
        """Reaction coordinate of a symmetric double well (barrier 5.12 kcal/mol, minima at +-0.33 A): the window
        must stay inside the region where the curve has a maximum."""
        eb, q0 = 5.12, 0.33
        k = -4 * eb / q0 ** 2
        t, e = self.scan(lambda x: eb * ((x / q0) ** 2 - 1.0) ** 2, k)
        r = C.size_direction(t, e, k, C.default_tolerance(k), sign=-1, rng=np.random.default_rng(1))
        self.assertLess(r["half_width"], 0.33)
        self.assertGreater(r["noise"], 0.1); self.assertLess(r["noise"], 0.35)
        self.assertLess(abs(r["x_ref"]), 1e-6)

    def test_no_pair_is_an_error(self):
        k, a = 1100.0, 2.2
        de = k / (2 * a * a)
        t, e = self.scan(lambda x: de * (1 - np.exp(-a * (x - 0.05))) ** 2, k)      # the scan is not centred on the minimum
        with self.assertRaises(C.SizerError) as cm:
            C.size_direction(t, e, k, 1e-7)
        self.assertIn("epsilon", str(cm.exception))
        with self.assertRaises(ValueError):
            C.size_direction(t, e, k, 0.0)
        e[3] = np.nan
        with self.assertRaises(ValueError):
            C.size_direction(t, e, k, 0.01)


class TestDriver(unittest.TestCase):
    def test_full_search_on_water_ends_within_the_tolerance(self):
        """Model water from the surrogate minimum, block noise 10 kcal/mol. The start is 0.20 kcal/mol above
        the true minimum and 0.0136 A away along the second direction (tolerance 0.0068 A)."""
        case = water_case(); ft, fs, xs, xt, d = case
        with tempfile.TemporaryDirectory() as tmp:
            eng, s = make(case, tmp, seed=1)
            self.assertEqual(s.run(), C.DONE_STATUS)                      # never CONVERGED: a fixed number of iterations
            st = s.state
            eps = np.array(st["epsilon"])
            np.testing.assert_allclose(eps, np.sqrt(0.1 / np.abs(d.curvature)), rtol=1e-12)
            self.assertEqual(sorted(st["iters"]), ["1", "2", "3", "4"])
            self.assertEqual(len(s.trajectory()), 5)
            # position along the directions, relative to the true minimum: inside the tolerance at the end
            z_true = d.vectors.dot((xt - xs).reshape(-1))
            z = s.coordinates()
            self.assertGreater(abs(z[0] - z_true)[1], 1.9 * eps[1])       # the start was outside
            self.assertTrue(np.all(np.abs(z[-1] - z_true) < eps), (z[-1] - z_true, eps))
            self.assertLess(ft(s.final_geometry()) - ft(xt), 0.05)
            self.assertGreater(ft(xs) - ft(xt), 0.20)
            np.testing.assert_allclose(s.final_geometry(), xs + z[-1].dot(d.vectors).reshape(-1, 3), atol=1e-12)
            np.testing.assert_allclose(s.trajectory()[0], xs, atol=1e-14)
            # the error bars of the steps are 1 standard error: about half of the 95 % half-width, which the
            # sizer keeps below the tolerance
            for j in range(3):
                res = st["iters"]["4"]["decided"][str(j)]["res"]
                self.assertLess(res["sm"], 0.62 * eps[j]); self.assertGreater(res["sm95"], 1.6 * res["sm"])
            # the bracket check: every step was inside its grid
            self.assertEqual([b["status"] for b in s.brackets()], [C.INSIDE] * 12)
            # the audit is stored for every iteration. The windows of the sizer are 0.28 to 0.92 A wide, far more
            # than the range in which this surface is a cubic: the two totally symmetric directions are UNRESOLVED.
            self.assertGreater(st["sizer"][0]["half_width"], 0.9)
            for k in "1234":
                audit = st["iters"][k]["audit"]
                self.assertEqual([r["verdict"] for r in audit[:2]], ["UNRESOLVED", "UNRESOLVED"])
                self.assertGreater(audit[0]["spread"], 2.0 * audit[0]["threshold"])
            self.assertIn("UNRESOLVED", st["message"])
            # the scan and two of the lines hold geometries with atoms closer than 0.6 A: reported, not stopped
            self.assertTrue(any("surrogate scan of direction 0" in w for w in st["warnings"]))
            self.assertTrue(any("line of direction 0" in w for w in st["warnings"]))
            self.assertEqual(len(st["warnings"]), 5)
            # full-length runs at the start and at the final geometry
            fin = st["final"]
            self.assertEqual(len(fin["keys"]), 2); self.assertFalse(fin["certified"])
            self.assertEqual(eng.read(fin["keys"][1]).nblocks, 400)
            de, sd = fin["energy_change"]
            self.assertLess(abs(de - (ft(s.final_geometry()) - ft(xs))), 3.0 * sd)
            self.assertAlmostEqual(sd, 10.0 * np.sqrt(2.0 / 700.0), delta=0.15)          # 2 x 2 runs of 350 usable blocks
            self.assertAlmostEqual((s.final_energy()[0] - s.start_energy()[0]) * HARTREE_TO_KCAL, de, places=9)
            # the mean of the last three centres, as the old method reported it
            mean = st["mean"]
            self.assertEqual(mean["centres"], [2, 3, 4])
            np.testing.assert_allclose(mean["coordinates"], z[2:5].mean(axis=0), atol=1e-12)
            np.testing.assert_allclose(s.mean_geometry(), xs + z[2:5].mean(axis=0).dot(d.vectors).reshape(-1, 3), atol=1e-12)
            self.assertLess(ft(s.mean_geometry()) - ft(xt), 0.05)
            # block numbers follow the noise of the sizer: (10 / sigma_j)^2 usable blocks, at least 30, plus 50
            plan = st["plan"]
            self.assertEqual(plan["blocks"], [86, 140, 80]); self.assertEqual(plan["capped"], [False] * 3)
            for j in range(3):
                self.assertEqual(plan["blocks"][j], C.blocks_for_noise(10.0, st["sizer"][j]["noise"], 50, 30, 400)[0])
                for key in st["iters"]["3"]["points"][str(j)].values():
                    self.assertEqual(eng.read(key).nblocks, plan["blocks"][j])
            self.assertEqual(plan["centre_blocks"], 140)                  # the centre gets the smallest noise
            self.assertEqual(eng.read("geo/it2/centre").nblocks, 140)
            self.assertEqual(eng.read("geo/it1/centre").nblocks, 400)     # the first centre is a full-length run
            self.assertEqual(s.total_blocks(), eng.total_blocks())
            self.assertEqual(s.search_blocks(), 4 * (6 * (86 + 140 + 80) + 140))
            json.dumps(st, allow_nan=False)                               # the state file holds no NaN
            with open(os.path.join(tmp, "geo_state.json")) as fh:
                self.assertEqual(json.load(fh)["status"], "DONE")
            self.assertFalse(os.path.exists(os.path.join(tmp, "geo_state.json.tmp")))

    def test_settings_of_the_old_library_give_the_same_quality(self):
        """One window scale and a 10 x 10 mesh (the sizer of the first applications) on the same case."""
        case = water_case(); ft, fs, xs, xt, d = case
        with tempfile.TemporaryDirectory() as tmp:
            _eng, s = make(case, tmp, seed=2, classic={"window_scales": [1], "mesh": 10})
            self.assertEqual(s.run(), C.DONE_STATUS)
            z_true = d.vectors.dot((xt - xs).reshape(-1))
            self.assertTrue(np.all(np.abs(s.coordinates()[-1] - z_true) < np.array(s.state["epsilon"])))
            self.assertEqual([len(z["trials"]) for z in s.state["sizer"]], [1, 1, 1])

    def test_state_file_resume_gives_the_same_result(self):
        case = water_case(); ft, fs, xs, xt, d = case
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            _e, ref = make(case, a, seed=4); ref.run()
            eng, s = make(case, b, seed=4)
            for _ in range(500):                                          # stop in the middle of the search
                if s.state["iteration"] == 2:
                    break
                eng.advance(); s.tick()
            self.assertEqual(s.state["iteration"], 2)
            self.assertEqual(s.status, "RUNNING")
            # a new driver object on the same state file and engine; its other arguments are not used
            s2 = ClassicLineSearch("geo", eng, None, [], np.zeros((1, 3)), np.zeros((1, 3)), [1.0], state_dir=b)
            self.assertEqual(s2.state["iteration"], s.state["iteration"])
            s2.run()
            self.assertEqual(s2.status, ref.status)
            np.testing.assert_allclose(s2.final_geometry(), ref.final_geometry(), atol=1e-12)
            self.assertEqual(s2.state["final"]["energy_change"], ref.state["final"]["energy_change"])
            self.assertEqual(s2.state["sizer"], ref.state["sizer"])

    def test_block_noise_is_measured_when_the_settings_do_not_give_it(self):
        case = water_case(); ft, xt = case[0], case[3]
        with tempfile.TemporaryDirectory() as tmp:
            _eng, s = make(case, tmp, seed=2, sigma_block=2.0, with_sigma=False)
            self.assertIsNone(s.state["sigma_block"])
            self.assertEqual(s.run(), C.DONE_STATUS)
            self.assertAlmostEqual(s.state["sigma_block"], 2.0, delta=0.6)
            self.assertIn("measured", s.state["sigma_block_source"])
            self.assertLess(ft(s.final_geometry()) - ft(xt), 0.15)

    def test_failed_point_needs_attention_and_can_be_cleared(self):
        case = water_case(); ft, fs, xs, xt, d = case

        class Broken(MockEngine):
            broken = True

            def read(self, key):
                data = super().read(key)
                if self.broken and "/it2/d1_p" in key:
                    data.state = FAILED; data.message = "walkers lost"
                return data

        with tempfile.TemporaryDirectory() as tmp:
            eng = Broken(ft, fs, sigma_block=2.0, blocks_per_tick=40, seed=1)
            _e, s = make(case, tmp, engine=eng)
            self.assertEqual(s.run(max_ticks=200), "NEEDS_ATTENTION")
            self.assertIn("/it2/d1_p", s.state["message"]); self.assertIn("walkers lost", s.state["message"])
            self.assertEqual(s.tick(), "NEEDS_ATTENTION")                 # stays there until a person acts
            eng.broken = False                                            # the point was repaired
            self.assertEqual(s.clear_attention(), "RUNNING")
            self.assertEqual(s.run(), C.DONE_STATUS)

    def test_surrogate_scan_failures_are_visible(self):
        case = water_case(); ft, fs, xs, xt, d = case

        class NanScan(AnalyticSurrogate):
            def scan(self, tag):
                e = super().scan(tag).copy(); e[20] = np.nan
                return e

        class NoScan(Surrogate):
            pass

        with tempfile.TemporaryDirectory() as tmp:
            _e, s = make(case, tmp, surrogate=NanScan(fs))
            self.assertEqual(s.run(max_ticks=5), "NEEDS_ATTENTION")
            self.assertIn("no finite energy", s.state["message"]); self.assertIn("direction 1", s.state["message"])
        with tempfile.TemporaryDirectory() as tmp:
            _e, s = make(case, tmp, surrogate=NoScan())
            self.assertEqual(s.run(max_ticks=5), "NEEDS_ATTENTION")
            self.assertIn("request_scan", s.state["message"])
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):                           # no surrogate and no windows
                make(case, tmp, surrogate=None)

    def test_tolerance_that_cannot_be_met_needs_attention(self):
        case = water_case()
        with tempfile.TemporaryDirectory() as tmp:
            _e, s = make(case, tmp, classic={"epsilon": 1e-8})
            self.assertEqual(s.run(max_ticks=5), "NEEDS_ATTENTION")
            self.assertIn("no window and noise meet the tolerance", s.state["message"])
            self.assertIn("direction 1", s.state["message"])
            self.assertIsNone(s.state["sizer"])
            self.assertEqual(s.tick(), "NEEDS_ATTENTION")

    def test_options_are_checked(self):
        case = water_case()
        with tempfile.TemporaryDirectory() as tmp:
            for bad in ({"iteratons": 3}, {"points": 6}, {"windows": [1.0, 1.0, 1.0]}, {"epsilon": [0.01, 0.01]},
                        {"max_blocks": 60}, {"window_scales": [0.5]}):
                with self.assertRaises(ValueError, msg=str(bad)):
                    make(case, tmp, classic=bad)
            with self.assertRaises(ValueError):
                ft, fs, xs, xt, d = case
                ClassicLineSearch("geo", MockEngine(ft), AnalyticSurrogate(fs), list(ft.symbols), xs, 2.0 * d.vectors,
                                  d.curvature, state_dir=tmp)
            self.assertEqual(os.listdir(tmp), [])                         # nothing was written by a refused search

    def test_given_windows_and_capped_blocks(self):
        """Windows and noises from the settings replace the scan and the sizer. A noise that needs more blocks
        than the upper limit is reported."""
        case = water_case(); ft, fs, xs, xt, d = case
        with tempfile.TemporaryDirectory() as tmp:
            eng, s = make(case, tmp, seed=3, sigma_block=2.0, surrogate=None,
                          classic={"windows": [4.0, 20.0, 20.0], "noises": [0.25, 0.5, 0.1], "iterations": 2, "warmup": 1,
                                   "final_series": 1})
            self.assertEqual(s.run(), C.DONE_STATUS)
            st = s.state
            self.assertIsNone(st["scan"])
            self.assertEqual(st["plan"]["blocks"], [114, 80, 400]); self.assertEqual(st["plan"]["capped"], [False, False, True])
            self.assertTrue(any("upper limit of 400 blocks" in w for w in st["warnings"]))
            self.assertIn("warning", st["message"])
            self.assertEqual(sorted(st["iters"]), ["1", "2"]); self.assertEqual(st["mean"]["centres"], [1, 2])
            self.assertEqual(len(st["final"]["keys"]), 1)
            R = [C.window_half_width(w, k) for w, k in zip([4.0, 20.0, 20.0], d.curvature)]
            np.testing.assert_allclose([g[-1] for g in st["iters"]["1"]["grid"]], R, rtol=1e-12)
            self.assertEqual(len(st["iters"]["1"]["points"]["0"]), 6)      # 7 points, the centre is shared

    def test_wide_window_is_recorded_by_the_bracket_check(self):
        """One direction, a Morse curve whose minimum is far outside the grid: the step is an extrapolation.
        The driver takes it unchanged (as the old method did) and records that it was not bracketed."""
        curve = lambda x: 110.0 * (1 - np.exp(-2.0 * (x + 0.60))) ** 2          # noqa: E731  minimum at -0.60 A
        x0, vectors, surface = line_case(curve)
        k = 2 * 110.0 * 4.0
        with tempfile.TemporaryDirectory() as tmp:
            eng = MockEngine(surface, sigma_block=0.2, blocks_per_tick=50, seed=1)
            s = ClassicLineSearch("line", eng, None, ["H", "H"], x0, vectors, [k], state_dir=tmp, settings={
                "sigma_block": 0.2, "classic": {"windows": [0.5 * k * 0.4 ** 2], "noises": [0.05], "iterations": 1,
                                                 "warmup": 1, "final_series": 1, "min_distance": 0.1}})
            self.assertEqual(s.run(), C.DONE_STATUS)
            b = s.brackets()[0]
            self.assertEqual(b["status"], C.OUTSIDE)
            self.assertAlmostEqual(b["m"], -1.4565, delta=0.08)             # the noise-free value; far outside +-0.40 A
            self.assertAlmostEqual(s.coordinates()[-1, 0], b["m"], places=12)       # the step was applied as it is
            self.assertIn("1 of 1 fitted steps were not bracketed", s.state["message"])
            self.assertEqual(s.state["iters"]["1"]["composed"]["not_bracketed"], [0])

    def test_saddle_search_follows_the_sign_rule(self):
        ft, fs = M.ammonia(True), M.ammonia(False)
        xs = M.refine_stationary(fs, M.ammonia_geometry(theta=120.0, r=1.0))
        xt = M.refine_stationary(ft, M.ammonia_geometry(theta=120.0, r=1.0))
        Hs = M.hessian(fs, xs); sub = S.mode_subspace(Hs, xs)
        d = S.search_directions(S.project_hessian(Hs, sub.basis), sub)
        self.assertEqual(d.n_negative, 1)
        with tempfile.TemporaryDirectory() as tmp:
            _eng, s = make((ft, fs, xs, xt, d), tmp, seed=5, sigma_block=1.0, name="saddle")
            self.assertEqual(s.state["roles"], [-1, 1, 1, 1, 1, 1])
            self.assertEqual(s.run(), C.DONE_STATUS)
            xf = s.final_geometry()
            self.assertAlmostEqual(G.distance(xf, 0, 1), G.distance(xt, 0, 1), delta=0.008)
            self.assertLess(abs(G.distance(xf, 0, 1) - G.distance(xt, 0, 1)), abs(G.distance(xs, 0, 1) - G.distance(xt, 0, 1)))
            self.assertAlmostEqual(G.angle(xf, 1, 0, 2), 120.0, delta=1.5)          # still planar: a saddle, not a minimum
            self.assertEqual(s.state["iters"]["1"]["decided"]["0"]["res"]["role"], -1)
            self.assertLess(s.state["iters"]["1"]["decided"]["0"]["res"]["k"], 0.0)  # a maximum was fitted


if __name__ == "__main__":
    unittest.main()
