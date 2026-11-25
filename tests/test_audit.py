"""The stability audit of a classic line-search iteration."""
import os
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

from montepass.engine.mock import MockEngine  # noqa: E402
from montepass.search import audit as A  # noqa: E402
from montepass.search import classic as C  # noqa: E402


def reference_extremum(t, e, s, order, leave_out=None):
    """The fit of the audit written a second time with numpy.polynomial: weighted normal equations, roots of
    the derivative, a minimum inside the range of the points that were used, the one closest to zero."""
    idx = [i for i in range(len(t)) if i != leave_out]
    T, E, S = np.array(t)[idx], np.array(e)[idx], np.array(s)[idx]
    if len(T) < order + 2:
        return None
    X = np.vstack([T ** p for p in range(order + 1)]).T
    W = np.diag(1.0 / S ** 2)
    beta = np.linalg.solve(X.T.dot(W).dot(X), X.T.dot(W).dot(E))
    der = np.polynomial.polynomial.polyder(beta)
    der2 = np.polynomial.polynomial.polyder(der)
    roots = np.polynomial.polynomial.polyroots(der)
    real = [x.real for x in roots if abs(x.imag) < 1e-9 and T.min() <= x.real <= T.max()
            and np.polynomial.polynomial.polyval(x.real, der2) > 0]
    return min(real, key=abs) if real else None


class TestAuditRule(unittest.TestCase):
    def test_clean_direction_is_resolved(self):
        # a parabola with its minimum at +0.02 A inside a window of +-0.30 A, noise 0.3 kcal/mol per point
        k = 400.0
        t = np.linspace(-0.3, 0.3, 7)
        eps = C.default_tolerance(k)                                   # 0.0158 A
        for seed in range(4):
            e = 0.5 * k * (t - 0.02) ** 2 + 0.3 * np.random.default_rng(seed).normal(size=7)
            r = A.audit_direction(t, e, np.full(7, 0.3), eps)
            self.assertEqual(r["verdict"], A.RESOLVED, r)
            self.assertEqual(len(r["candidates"]), 9)                 # quadratic, cubic and 7 leave-one-out fits
            self.assertAlmostEqual(r["threshold"], 2 * eps)
            self.assertLessEqual(r["spread"], r["threshold"])
            self.assertAlmostEqual(r["cubic"], 0.02, delta=0.012)
            self.assertAlmostEqual(r["spread"], max(r["candidates"]) - min(r["candidates"]))

    def test_strongly_quartic_curve_in_a_wide_window_is_unresolved(self):
        # E = Lam x^2 + c4 x^4 about a minimum at +0.08 A; window +-0.40 A is twice the length sqrt(Lam / c4)
        lam, R = 100.0, 0.40
        c4 = lam * (2.0 / R) ** 2
        t = np.linspace(-R, R, 7)
        eps = C.default_tolerance(2.0 * lam)                           # 0.0224 A
        for seed in range(4):
            e = lam * (t - 0.08) ** 2 + c4 * (t - 0.08) ** 4 + 0.3 * np.random.default_rng(seed).normal(size=7)
            r = A.audit_direction(t, e, np.full(7, 0.3), eps)
            self.assertEqual(r["verdict"], A.UNRESOLVED, r)
            self.assertGreater(r["spread"], r["threshold"])
            # the classic fit of the same data returns a small step and reports nothing unusual
            fit = C.fit_step(t, e, errors=np.full(7, 0.3))
            self.assertEqual(fit["status"], C.INSIDE)
            self.assertLess(abs(fit["m"]), 0.04)                       # the true offset is 0.08 A

    def test_same_numbers_as_an_independent_implementation(self):
        rng = np.random.default_rng(3)
        for trial in range(300):
            R = rng.uniform(0.05, 1.0)
            t = np.linspace(-R, R, 7)
            k = rng.uniform(50.0, 2000.0)
            d = rng.uniform(-1.5, 1.5) * R
            c4 = rng.uniform(0.0, 5.0) * k / R ** 2 * (trial % 3 == 0)
            s = rng.uniform(0.05, 2.0, size=7)
            e = 0.5 * k * (t - d) ** 2 + c4 * (t - d) ** 4 + s * rng.normal(size=7) - 59000.0
            r = A.audit_direction(t, e, s, 0.01)
            ref = [reference_extremum(t, e, s, 2), reference_extremum(t, e, s, 3)] + \
                  [reference_extremum(t, e, s, 3, leave_out=i) for i in range(7)]
            got = [r["quadratic"], r["cubic"]] + list(r["leave_one_out"])
            for a, b in zip(got, ref):
                self.assertEqual(a is None, b is None, (trial, got, ref))
                if a is not None:
                    self.assertAlmostEqual(a, b, delta=1e-7)

    def test_fewer_than_three_candidates_is_unresolved(self):
        t = np.linspace(-0.3, 0.3, 7)
        e = 40.0 * t + 25.0 * t ** 3                                    # rises everywhere: no minimum
        r = A.audit_direction(t, e, np.full(7, 0.2), 0.02)
        self.assertEqual(r["verdict"], A.UNRESOLVED)
        self.assertIsNone(r["spread"])
        self.assertEqual(r["candidates"], [])
        self.assertIn("0 of 9 fits", r["note"])

    def test_threshold_has_a_floor_of_one_hundredth_angstrom(self):
        t = np.linspace(-0.1, 0.1, 7)
        e = 0.5 * 2000.0 * t ** 2
        r = A.audit_direction(t, e, np.full(7, 0.05), 0.002)
        self.assertAlmostEqual(r["threshold"], 0.01)
        self.assertEqual(r["verdict"], A.RESOLVED)
        r = A.audit_direction(t, e, np.full(7, 0.05), 0.002, floor=0.0)
        self.assertAlmostEqual(r["threshold"], 0.004)

    def test_maximum_along_a_direction_of_negative_curvature(self):
        t = np.linspace(-0.25, 0.25, 7)
        e = -0.5 * 188.0 * (t - 0.03) ** 2 + 0.1 * np.random.default_rng(5).normal(size=7)
        s = np.full(7, 0.1)
        r = A.audit_direction(t, e, s, 0.023, sign=-1)
        self.assertEqual(r["verdict"], A.RESOLVED)
        self.assertAlmostEqual(r["cubic"], 0.03, delta=0.01)
        # the same data searched for a minimum: no fit has one
        r = A.audit_direction(t, e, s, 0.023, sign=+1)
        self.assertEqual(r["verdict"], A.UNRESOLVED)
        self.assertEqual(len(r["candidates"]), 0)

    def test_range_of_a_fit_without_an_end_point(self):
        """The minimum must lie inside the points that the fit used. A minimum at +0.25 A lies between the
        last two points of the grid; the cubic that leaves out the last point (+0.30 A) gives no candidate."""
        t = np.linspace(-0.3, 0.3, 7)
        e = 0.5 * 300.0 * (t - 0.25) ** 2
        r = A.audit_direction(t, e, np.zeros(7), 0.02)                  # no noise: equal weights
        self.assertAlmostEqual(r["quadratic"], 0.25, places=9)
        self.assertAlmostEqual(r["cubic"], 0.25, places=7)
        self.assertIsNone(r["leave_one_out"][6])
        self.assertAlmostEqual(r["leave_one_out"][0], 0.25, places=7)
        self.assertEqual(len(r["candidates"]), 8)
        self.assertEqual(r["verdict"], A.RESOLVED)

    def test_too_few_points_and_bad_input(self):
        t = np.linspace(-0.2, 0.2, 4)
        r = A.audit_direction(t, 100.0 * t ** 2, np.full(4, 0.1), 0.02)
        self.assertEqual(r["verdict"], A.SKIPPED)
        t = np.linspace(-0.2, 0.2, 7)
        e = 100.0 * t ** 2
        with self.assertRaises(ValueError):                             # zero next to non-zero error bars
            A.audit_direction(t, e, np.array([0.1, 0.1, 0.0, 0.1, 0.1, 0.1, 0.1]), 0.02)
        with self.assertRaises(ValueError):
            A.audit_direction(t, e, np.full(7, 0.1), 0.0)
        with self.assertRaises(ValueError):
            A.audit_direction(t, e, np.full(7, np.nan), 0.02)
        with self.assertRaises(ValueError):
            A.audit_iteration([t], [e], [np.full(7, 0.1)], [0.02, 0.02])

    def test_iteration_with_mixed_roles(self):
        t = np.linspace(-0.2, 0.2, 7)
        s = np.full(7, 0.05)
        e_min = 0.5 * 500.0 * (t + 0.01) ** 2
        e_max = -0.5 * 200.0 * (t - 0.02) ** 2
        res = A.audit_iteration([t, t], [e_max, e_min], [s, s], [0.02, 0.014], signs=[-1, 1])
        self.assertEqual([r["verdict"] for r in res], [A.RESOLVED, A.RESOLVED])
        self.assertEqual(A.unresolved(res), [])
        self.assertAlmostEqual(res[0]["cubic"], 0.02, places=6)
        self.assertAlmostEqual(res[1]["cubic"], -0.01, places=6)
        res = A.audit_iteration([t, t], [e_max, e_min], [s, s], [0.02, 0.014])      # both searched for a minimum
        self.assertEqual(A.unresolved(res), [0])


def line_case(curve):
    """A model with one search direction: two atoms, and an energy that depends only on the displacement
    along the unit vector of their stretch."""
    x0 = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.2]])
    v = np.array([0.0, 0.0, -1.0, 0.0, 0.0, 1.0]) / np.sqrt(2.0)

    def surface(x):
        return float(curve(float(np.dot(np.asarray(x, dtype=float).reshape(-1) - x0.reshape(-1), v))))

    return x0, v[None, :], surface


class TestAuditOfTheDriver(unittest.TestCase):
    def test_driver_records_the_audit_of_every_iteration(self):
        """A model curve with a large quartic term, E = 150 (x - 0.1)^2 + 15000 (x - 0.1)^4 kcal/mol (force
        constant 300 kcal/mol/A^2 at the minimum; the quartic term equals the harmonic term 0.1 A from the
        minimum), through the driver with a wide window (25 kcal/mol, half-width 0.4082 A): the classic search
        ends near +0.04 A although the minimum of the curve is at +0.10 A, and the audit of every iteration says
        UNRESOLVED. The numbers of the asserts were taken from a run of this test, with margins."""
        curve = lambda x: 150.0 * (x - 0.1) ** 2 + 15000.0 * (x - 0.1) ** 4     # noqa: E731
        k = 300.0
        x0, vectors, surface = line_case(curve)
        with tempfile.TemporaryDirectory() as tmp:
            eng = MockEngine(surface, sigma_block=1.0, blocks_per_tick=50, seed=2)
            s = C.ClassicLineSearch("line", eng, None, ["H", "H"], x0, vectors, [k], state_dir=tmp, settings={
                "sigma_block": 1.0, "classic": {"windows": [25.0], "noises": [0.1], "final_series": 1}})
            self.assertEqual(s.run(), C.DONE_STATUS)
            z = s.coordinates()[:, 0]
            self.assertAlmostEqual(s.state["sizer"][0]["half_width"], 0.4082, places=4)    # sqrt(2 x 25 / 300)
            self.assertGreater(z[-1], 0.02); self.assertLess(z[-1], 0.05)          # far from the minimum at +0.10 A
            self.assertGreater(curve(z[-1]) - curve(0.1), 0.6)                     # of 3.00 kcal/mol to gain
            for kk in range(1, 5):
                stored = s.state["iters"][str(kk)]["audit"]
                self.assertEqual(stored[0]["verdict"], A.UNRESOLVED)
                again = s.audit(kk)
                self.assertEqual(again[0]["verdict"], stored[0]["verdict"])
                self.assertAlmostEqual(again[0]["spread"], stored[0]["spread"])
            self.assertEqual(s.audit()[0]["spread"], s.audit(4)[0]["spread"])       # default: the last iteration
            self.assertIn("UNRESOLVED", s.state["message"])
            self.assertEqual([b["status"] for b in s.brackets()], [C.INSIDE] * 4)   # the bracket check sees nothing

    def test_audit_before_the_first_fit_is_refused(self):
        x0, vectors, surface = line_case(lambda x: 100.0 * x * x)
        with tempfile.TemporaryDirectory() as tmp:
            eng = MockEngine(surface, sigma_block=1.0, blocks_per_tick=50, seed=1)
            s = C.ClassicLineSearch("line", eng, None, ["H", "H"], x0, vectors, [200.0], state_dir=tmp, settings={
                "sigma_block": 1.0, "classic": {"windows": [5.0], "noises": [0.2]}})
            with self.assertRaises(ValueError):
                s.audit()


if __name__ == "__main__":
    unittest.main()
