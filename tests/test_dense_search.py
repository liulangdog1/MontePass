"""The dense-scan driver on model surfaces (mock engine): protocols v1 and v2."""
import os
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

from montepass import geometry as G, model as M, subspace as S  # noqa: E402
from montepass.engine import PointSpec, FAILED  # noqa: E402
from montepass.units import HARTREE_TO_KCAL  # noqa: E402
from montepass.linalg import product  # noqa: E402
from montepass.engine.mock import MockEngine  # noqa: E402
from montepass.search import protocol_v1 as P1, protocol_v2 as P2  # noqa: E402
from montepass.search.dense import CONVERGED, NEEDS_ATTENTION, NOT_CONVERGED, RUNNING_STATUS, DenseScanSearch  # noqa: E402
from montepass.surrogate.analytic import AnalyticSurrogate  # noqa: E402


def water_case(shift=None, direction=2):
    """True surface, surrogate surface, start geometry (surrogate minimum), true minimum, directions."""
    ft, fs = M.water(True), M.water(False)
    xt = M.minimize(ft, M.water_geometry())
    if shift is None:
        xs = M.minimize(fs, M.water_geometry())
    else:
        Ht = M.hessian(ft, xt); sub = S.mode_subspace(Ht, xt)
        d0 = S.search_directions(S.project_hessian(Ht, sub.basis), sub)
        delta = (shift * d0.vectors[direction]).reshape(-1, 3)
        fs = ft.shifted(delta); xs = xt + delta
    Hs = M.hessian(fs, xs); sub = S.mode_subspace(Hs, xs)
    d = S.search_directions(S.project_hessian(Hs, sub.basis), sub)
    return ft, fs, xs, xt, d


def make(case, proto, tmp, seed=1, sigma=5.0, name="min", surrogate=True):
    ft, fs, xs, xt, d = case
    eng = MockEngine(ft, fs, sigma_block=sigma, blocks_per_tick=10, seed=seed, transient=20.0, equil_blocks=0.6)
    sur = AnalyticSurrogate(fs) if surrogate else None
    return eng, DenseScanSearch(name, eng, sur, list(ft.symbols), xs, d.vectors, d.curvature, protocol=proto, state_dir=tmp)


class TestProtocolV1Driver(unittest.TestCase):
    def test_converges_near_the_minimum(self):
        case = water_case(); ft, xt = case[0], case[3]
        with tempfile.TemporaryDirectory() as tmp:
            _eng, s = make(case, P1.V1, tmp, seed=1)
            self.assertEqual(s.run(), CONVERGED)
            self.assertLess(ft(s.final_geometry()) - ft(xt), 0.15)
            self.assertLess(ft(s.final_geometry()) - ft(xt), ft(case[2]) - ft(xt))     # better than the start
            it1 = s.state["iters"]["1"]
            self.assertEqual(len(it1["points"]["0"]), 4)                              # two shells, two sides
            self.assertIn(it1["composed"]["rule"], ("coupled-newton", "diagonal"))

    def test_state_file_resume_gives_the_same_result(self):
        case = water_case()
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            _e, ref = make(case, P1.V1, a, seed=4); ref.run()
            eng, s = make(case, P1.V1, b, seed=4)
            for _ in range(12):
                eng.advance(); s.tick()
            ft, fs, xs, xt, d = case                    # a new driver object on the same state file and engine
            s2 = DenseScanSearch("min", eng, AnalyticSurrogate(fs), list(ft.symbols), xs, d.vectors, d.curvature,
                                 protocol=P1.V1, state_dir=b)
            self.assertEqual(s2.state["iteration"], s.state["iteration"])
            s2.run()
            self.assertEqual(s2.status, ref.status)
            np.testing.assert_allclose(s2.final_geometry(), ref.final_geometry(), atol=1e-12)

    def test_far_start_is_not_recovered_by_v1(self):
        """A minimum outside 1.5 times the scanned span is recorded as not resolved: v1 does not move there."""
        case = water_case(shift=0.30); ft, xt = case[0], case[3]
        with tempfile.TemporaryDirectory() as tmp:
            _eng, s = make(case, P1.V1, tmp, seed=2)
            self.assertEqual(s.run(), NOT_CONVERGED)
            self.assertGreater(ft(s.final_geometry()) - ft(xt), 50.0)
            notes = " ".join(d["note"] for it in s.state["iters"].values() for d in it["decided"].values())
            self.assertIn("rescue exhausted", notes)

    def test_without_surrogate_the_step_is_diagonal(self):
        case = water_case()
        with tempfile.TemporaryDirectory() as tmp:
            _eng, s = make(case, P1.V1, tmp, seed=3, surrogate=False)
            s.run()
            self.assertEqual(s.state["iters"]["1"]["composed"]["rule"], "diagonal")

    def test_failed_point_needs_attention(self):
        case = water_case()

        class Broken(MockEngine):
            def read(self, key):
                d = super().read(key)
                if key.endswith("d1_p%03d" % 0) or "d1_p" in key:
                    d.state = FAILED; d.energy = d.energy[:0]; d.weight = d.weight[:0]
                return d

        ft, fs, xs, xt, d = case
        with tempfile.TemporaryDirectory() as tmp:
            eng = Broken(ft, fs, sigma_block=5.0, blocks_per_tick=50, seed=1)
            s = DenseScanSearch("min", eng, AnalyticSurrogate(fs), list(ft.symbols), xs, d.vectors, d.curvature,
                                protocol=P1.V1, state_dir=tmp)
            self.assertEqual(s.run(max_ticks=50), NEEDS_ATTENTION)
            self.assertIn("without usable data", s.state["message"])


class TestRestartAndFailures(unittest.TestCase):
    def test_decision_is_saved_before_the_next_points_are_submitted(self):
        """The process is stopped while it submits the points of iteration 2. The next process must find the
        decision of iteration 1 in the state file, and must not decide again."""
        case = water_case(shift=0.10); ft, fs, xs, xt, d = case

        class Stop(Exception):
            pass

        class Breaks(MockEngine):
            armed = True

            def submit(self, spec):
                if self.armed and "/it2/" in spec.key and spec.role == "scan":
                    raise Stop("the process dies here")
                super().submit(spec)

        with tempfile.TemporaryDirectory() as tmp:
            state_file = os.path.join(tmp, "engine.json")
            eng = Breaks(ft, fs, sigma_block=5.0, blocks_per_tick=10, seed=3, transient=20.0, equil_blocks=0.6, state_file=state_file)
            s = DenseScanSearch("min", eng, AnalyticSurrogate(fs), list(ft.symbols), xs, d.vectors, d.curvature,
                                protocol=P2.V2, state_dir=tmp)
            with self.assertRaises(Stop):
                s.run()
            # a new process: new engine object, new search object, the same files
            eng2 = MockEngine(ft, fs, sigma_block=5.0, blocks_per_tick=10, seed=3, transient=20.0, equil_blocks=0.6, state_file=state_file)
            s2 = DenseScanSearch("min", eng2, AnalyticSurrogate(fs), list(ft.symbols), xs, d.vectors, d.curvature,
                                 protocol=P2.V2, state_dir=tmp)
            self.assertEqual(s2.state["iteration"], 2)                       # the decision of iteration 1 was on disk
            first = s2.state["iters"]["1"]
            self.assertEqual(first["composed"]["verdict"], "MOVE")
            step = np.array(first["composed"]["components"])
            centre2 = np.array(first["centre"]) + (step @ d.vectors).reshape(-1, 3)
            np.testing.assert_allclose(np.array(s2.state["iters"]["2"]["centre"]), centre2, atol=1e-12)
            self.assertEqual(s2.run(), CONVERGED)
            self.assertLess(ft(s2.final_geometry()) - ft(xt), P2.V2.eps_total)
            # every scan point of iteration 2 sits where the saved state says
            it2 = s2.state["iters"]["2"]
            for j, pts in it2["points"].items():
                for t, key in pts.items():
                    want = np.array(it2["centre"]) + float(t) * d.vectors[int(j)].reshape(-1, 3)
                    np.testing.assert_allclose(np.array(eng2.points[key]["coords"]), want, atol=1e-9)

    def test_failed_centre_and_failed_surrogate_need_attention(self):
        case = water_case(); ft, fs, xs, xt, d = case

        class BrokenCentre(MockEngine):
            def read(self, key):
                data = super().read(key)
                if key.endswith("it1/centre"):
                    data.state = FAILED; data.energy = data.energy[:0]; data.weight = data.weight[:0]
                    data.message = "the Jastrow check failed"
                return data

        with tempfile.TemporaryDirectory() as tmp:
            eng = BrokenCentre(ft, fs, sigma_block=5.0, blocks_per_tick=10, seed=1)
            s = DenseScanSearch("min", eng, AnalyticSurrogate(fs), list(ft.symbols), xs, d.vectors, d.curvature,
                                protocol=P2.V2, state_dir=tmp)
            self.assertEqual(s.run(max_ticks=40), NEEDS_ATTENTION)
            self.assertIn("the run at the centre failed", s.state["message"])
            self.assertIn("the Jastrow check failed", s.state["message"])

        class BrokenSurrogate(AnalyticSurrogate):
            def result(self, tag):
                res = super().result(tag)
                if res is not None:
                    res.failed = True; res.message = "the calculation ran out of memory"
                return res

        with tempfile.TemporaryDirectory() as tmp:
            eng = MockEngine(ft, fs, sigma_block=5.0, blocks_per_tick=10, seed=1)
            s = DenseScanSearch("min", eng, BrokenSurrogate(fs), list(ft.symbols), xs, d.vectors, d.curvature,
                                protocol=P2.V2, state_dir=tmp)
            self.assertEqual(s.run(max_ticks=40), NEEDS_ATTENTION)
            self.assertIn("the surrogate calculation at the centre failed", s.state["message"])
            self.assertIn("out of memory", s.state["message"])

    def test_model_without_noise(self):
        """All blocks are equal, so every error bar is exactly zero: the search must not divide by it."""
        case = water_case(); ft, fs, xs, xt, d = case
        with tempfile.TemporaryDirectory() as tmp:
            eng = MockEngine(ft, fs, sigma_block=0.0, blocks_per_tick=10, seed=1)
            s = DenseScanSearch("min", eng, AnalyticSurrogate(fs), list(ft.symbols), xs, d.vectors, d.curvature,
                                protocol=P2.V2, state_dir=tmp)
            self.assertIn(s.run(max_ticks=400), (CONVERGED, NOT_CONVERGED))
            self.assertLess(ft(s.final_geometry()) - ft(xt), 0.05)


class TestEndOfSearch(unittest.TestCase):
    def test_final_run_is_submitted_again_after_a_stop_between_decision_and_submission(self):
        """The process ends after the final decision was saved and before the final run was submitted."""
        case = water_case(shift=0.05); ft, fs, xs, xt, d = case

        class Stop(Exception):
            pass

        class Breaks(MockEngine):
            def submit(self, spec):
                if spec.role == "final":
                    raise Stop("the process dies here")
                super().submit(spec)

        with tempfile.TemporaryDirectory() as tmp:
            state_file = os.path.join(tmp, "engine.json")
            kw = dict(sigma_block=5.0, blocks_per_tick=10, seed=2, transient=20.0, equil_blocks=0.6, state_file=state_file)
            s = DenseScanSearch("min", Breaks(ft, fs, **kw), AnalyticSurrogate(fs), list(ft.symbols), xs, d.vectors,
                                d.curvature, protocol=P2.V2, state_dir=tmp)
            with self.assertRaises(Stop):
                s.run()
            eng2 = MockEngine(ft, fs, **kw)
            s2 = DenseScanSearch("min", eng2, AnalyticSurrogate(fs), list(ft.symbols), xs, d.vectors, d.curvature,
                                 protocol=P2.V2, state_dir=tmp)
            fin = s2.state["final"]
            self.assertIsNotNone(fin)                                        # the decision was on disk
            self.assertNotIn(fin["keys"][0], eng2.points)                    # the run was not
            self.assertEqual(s2.run(), CONVERGED)
            np.testing.assert_allclose(np.array(eng2.points[fin["keys"][0]]["coords"]), np.array(fin["centre"]), atol=1e-12)
            self.assertIsNotNone(s2.final_energy())

    def test_rejected_step_at_the_iteration_limit_still_gets_final_runs(self):
        """Iteration 3 is the last one that is allowed, and its centre is higher in energy than the centre of
        iteration 2. The search must end at the centre of iteration 2, with full-length runs there."""
        case = water_case(shift=0.10); ft, fs, xs, xt, d = case

        class HighThird(MockEngine):
            def _trace(self, key, p):
                trace = super()._trace(key, p)
                if "/it3/" in key and not getattr(self, "_raised_" + str(hash(key)), False):
                    trace += 40.0 / HARTREE_TO_KCAL                          # the whole third scan sits 40 kcal/mol higher
                    setattr(self, "_raised_" + str(hash(key)), True)
                return trace

        with tempfile.TemporaryDirectory() as tmp:
            eng = HighThird(ft, fs, sigma_block=5.0, blocks_per_tick=10, seed=3, transient=20.0, equil_blocks=0.6)
            s = DenseScanSearch("min", eng, AnalyticSurrogate(fs), list(ft.symbols), xs, d.vectors, d.curvature,
                                protocol=P2.ProtocolV2(max_iter=3), state_dir=tmp)
            self.assertEqual(s.run(), NOT_CONVERGED)
            self.assertEqual(s.state["rejected"], 1)
            self.assertIs(s.state["iters"]["3"]["accepted"], False)
            fin = s.state["final"]
            self.assertTrue(fin["at_centre"]); self.assertFalse(fin["certified"]); self.assertIsNone(fin["joint_bound"])
            self.assertEqual(fin["iteration"], 2)
            np.testing.assert_allclose(s.final_geometry(), np.array(s.state["iters"]["2"]["centre"]), atol=1e-12)
            self.assertIn("the last step was rejected", s.state["message"])
            # full-length runs exist at that geometry, and the energy change is reported
            self.assertEqual(len(fin["keys"]), 2)
            for key in fin["keys"]:
                self.assertEqual(eng.read(key).nblocks, P2.V2.cap)
            ef = s.final_energy()
            self.assertIsNotNone(ef)
            self.assertLess(ef[1] * HARTREE_TO_KCAL, 0.3)
            self.assertLess(fin["energy_change"][0], -5.0)                   # iteration 2 was far below the start

    def test_short_runs_that_ended_keep_usable_data(self):
        """Block limit 50: the equilibration rule may ask to discard 20 or 25 blocks, which would leave fewer
        than the 31 usable blocks that a statistic needs. A run that has ended gets a smaller discard."""
        case = water_case(); ft, fs, xs, xt, d = case
        for seed in range(6):
            with tempfile.TemporaryDirectory() as tmp:
                # a slow start-up (8 blocks) makes the rule choose a large discard
                eng = MockEngine(ft, fs, sigma_block=5.0, blocks_per_tick=10, seed=seed, transient=40.0, equil_blocks=8.0)
                s = DenseScanSearch("min", eng, AnalyticSurrogate(fs), list(ft.symbols), xs, d.vectors, d.curvature,
                                    protocol=P2.ProtocolV2(cap=50, cap_scan_factor=1.0, plan_factor=0.0), state_dir=tmp)
                status = s.run()
                self.assertIn(status, (CONVERGED, NOT_CONVERGED), s.state.get("message"))
                large = 0
                for key, p in eng.points.items():
                    data = eng.read(key)
                    if data.nblocks == 50 and P2.mser_drop(data.energy, 10, 50) >= 20:
                        large += 1
                        self.assertIsNotNone(s._stat(key), key)
                if seed == 0:
                    self.assertGreater(large, 0)                             # the case that the test is about occurs

    def test_shape_part_close_to_the_tolerance_gives_shorter_spans(self):
        """All blocks are used, the joint bound is just above the tolerance, and most of it is the shape part.
        More blocks cannot help much, so the scan is repeated with shorter spans."""
        case = water_case(); ft, fs, xs, xt, d = case
        with tempfile.TemporaryDirectory() as tmp:
            eng, s = make(case, P2.V2, tmp, seed=1)
            for _ in range(200):
                s.tick(); eng.advance()
                data = s._gather(1, s._it(1))
                if data is not None and data["blocks"] >= 60:
                    break
            self.assertIsNotNone(data)
            it = s._it(1)
            n = s.dim
            Kh, _rule = s._model_hessian(1, it, data["ests"], quiet=True)
            k = np.abs(np.diag(Kh))
            # slopes of zero; errors that give a shape part of 86 % and a joint bound of 104 % of the tolerance
            cn = P2.joint_factor(n)
            for j, e in enumerate(data["ests"]):
                shape_energy = 0.86 * P2.V2.eps_total / (cn * n)
                total_energy = 1.04 * P2.V2.eps_total / (cn * n)
                e["g"] = 0.0; e["g_meas"] = 0.0; e["z"] = 0.0; e["curv_ok"] = True
                e["sg_shape"] = float(np.sqrt(2.0 * k[j] * shape_energy))
                e["sg"] = float(np.sqrt(2.0 * k[j] * total_energy))
                e["k"] = float(k[j])
            data["blocks"] = P2.V2.cap; data["running"] = False
            s._decide(1, it, data)
            self.assertEqual(it["composed"]["verdict"], "MOVE")
            self.assertIn("repeated with shorter spans", it["decision"]["note"])
            self.assertIsNone(s.state.get("scan_blocks"))                    # not the rule "more blocks"
            self.assertTrue(all(f < 1.0 for f in s.state["span_factor"]))


class TestRetry(unittest.TestCase):
    def test_search_continues_after_the_failed_point_was_repaired(self):
        case = water_case(); ft, fs, xs, xt, d = case

        class Broken(MockEngine):
            broken = True

            def read(self, key):
                data = super().read(key)
                if self.broken and "d1_p" in key:
                    data.state = FAILED; data.energy = data.energy[:0]; data.weight = data.weight[:0]
                return data

        for proto in (P1.V1, P2.V2):
            with tempfile.TemporaryDirectory() as tmp:
                eng = Broken(ft, fs, sigma_block=5.0, blocks_per_tick=10, seed=1, transient=20.0, equil_blocks=0.6)
                s = DenseScanSearch("min", eng, AnalyticSurrogate(fs), list(ft.symbols), xs, d.vectors, d.curvature,
                                    protocol=proto, state_dir=tmp)
                self.assertEqual(s.run(max_ticks=60), NEEDS_ATTENTION)
                self.assertEqual(s.run(max_ticks=5), NEEDS_ATTENTION)          # the status stays until it is cleared
                eng.broken = False                                              # the point was run again
                self.assertEqual(s.clear_attention(), RUNNING_STATUS)
                self.assertEqual(s.state["message"], "")
                self.assertEqual(s.run(), CONVERGED)
                self.assertLess(ft(s.final_geometry()) - ft(xt), 0.2)
                self.assertEqual(s.clear_attention(), CONVERGED)                # nothing to clear: no change


def fixed_signs(d):
    """The same search directions with signs that do not depend on the machine.

    ``search_directions`` makes the largest component of a direction positive. A symmetric direction of a
    symmetric molecule has two components of equal size and opposite sign, so rounding (2e-16) decides which of
    them counts as the largest, and the sign of the direction then differs between machines. The model case
    below needs one start and one set of scan points everywhere: here the LAST of the largest components is made
    positive."""
    vec = np.array(d.vectors, dtype=float); coeff = np.array(d.coefficients, dtype=float)
    for i in range(vec.shape[0]):
        a = np.abs(vec[i])
        k = int(np.flatnonzero(a >= (1.0 - 1e-6) * a.max())[-1])
        if vec[i, k] < 0:
            vec[i] = -vec[i]; coeff[i] = -coeff[i]
    return S.Directions(vectors=vec, curvature=d.curvature, hessian=d.hessian, coefficients=coeff)


def tilted_case(angle=34.0, energy=1.0):
    """Water with a surrogate that has the right curvatures along the wrong directions: the bend and the
    symmetric stretch of the true minimum are turned into each other by ``angle`` degrees, and the surrogate
    minimum is ``energy`` kcal/mol above the true minimum."""
    ft = M.water(True)
    xt = M.minimize(ft, M.water_geometry())
    Ht = M.hessian(ft, xt); sub = S.mode_subspace(Ht, xt)
    d0 = fixed_signs(S.search_directions(S.project_hessian(Ht, sub.basis), sub))
    u, v = d0.vectors[0], d0.vectors[1]
    w = (u + v) / np.sqrt(2.0)
    delta = (np.sqrt(2.0 * energy / (0.5 * (d0.curvature[0] + d0.curvature[1]))) * w).reshape(-1, 3)
    xs = xt + delta
    fs = M.RotatedSurface(ft, xt, xs, M.plane_rotation(u, v, angle))
    Hs = M.hessian(fs, xs); sub = S.mode_subspace(Hs, xs)
    d = fixed_signs(S.search_directions(S.project_hessian(Hs, sub.basis), sub))
    return ft, fs, xs, xt, d


class TestTurnedSurrogates(unittest.TestCase):
    def test_the_two_models_of_a_turned_hessian(self):
        """Both models have the same Hessian at the surrogate minimum. Only the model "surface" also turns the
        anharmonic part."""
        ft = M.water(True)
        xt = M.minimize(ft, M.water_geometry())
        Ht = M.hessian(ft, xt); sub = S.mode_subspace(Ht, xt)
        d0 = S.search_directions(S.project_hessian(Ht, sub.basis), sub)
        u, v = d0.vectors[0], d0.vectors[1]
        A = M.plane_rotation(u, v, 34.0)
        xs = xt + 0.02 * (u + v).reshape(-1, 3)
        whole = M.RotatedSurface(ft, xt, xs, A)
        harmonic = M.TurnedHessianSurface(ft, xt, xs, A)
        want = product(A.T, Ht, A)
        for f in (whole, harmonic):
            self.assertAlmostEqual(f(xs), ft(xt), places=9)                  # the stationary point was moved to xs
            np.testing.assert_allclose(M.hessian(f, xs), want, atol=0.5)
        # along the soft direction the third derivative of "harmonic" is that of the true surface; "whole" differs
        def third(f, x, w, h=0.02):
            w = w.reshape(-1, 3)
            return (f(x + 2 * h * w) - 2 * f(x + h * w) + 2 * f(x - h * w) - f(x - 2 * h * w)) / (2 * h ** 3)
        t_true, t_h, t_w = third(ft, xt, u), third(harmonic, xs, u), third(whole, xs, u)
        self.assertAlmostEqual(t_h, t_true, delta=1e-3 * max(1.0, abs(t_true)))
        self.assertGreater(abs(t_w - t_true), 0.2 * abs(t_true))


class TestWrongSurrogateHessian(unittest.TestCase):
    """The surrogate Hessian is rotated against the true one: the search directions are not the normal
    directions of the true surface, and the surrogate curvature along the soft direction is 2.5 times too small."""

    def test_the_model_case(self):
        ft, fs, xs, xt, d = tilted_case()
        self.assertAlmostEqual(ft(xs) - ft(xt), 1.0, delta=0.05)
        Ht, Hs = M.hessian(ft, xt), M.hessian(fs, xs)
        # the same force constants ...
        np.testing.assert_allclose(np.sort(np.linalg.eigvalsh(Hs))[-3:], np.sort(np.linalg.eigvalsh(Ht))[-3:], rtol=2e-3)
        # ... along other directions: the true Hessian is not diagonal in the search directions
        K = d.vectors @ M.hessian(ft, xs) @ d.vectors.T
        coupling = abs(K[0, 1]) / np.sqrt(K[0, 0] * K[1, 1])
        self.assertGreater(coupling, 0.5)
        self.assertGreater(K[0, 0], 2.0 * d.curvature[0])                     # the soft direction is much stiffer in truth
        A = M.plane_rotation(np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), 90.0)
        np.testing.assert_allclose(A @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(A @ A.T, np.eye(3), atol=1e-12)

    def test_search_does_not_oscillate(self):
        """Before the measured force constants were remembered, every second scan fell back to the surrogate
        curvature, the steps along the soft direction were 2.5 times too long, and the search oscillated
        (seed 11 ended 3.1 kcal/mol above the minimum)."""
        case = tilted_case(); ft, xt = case[0], case[3]
        for seed in (11, 2, 5):
            with tempfile.TemporaryDirectory() as tmp:
                _eng, s = make(case, P2.V2, tmp, seed=seed)
                self.assertEqual(s.run(), CONVERGED, "seed %d" % seed)
                self.assertLess(ft(s.final_geometry()) - ft(xt), P2.V2.eps_total, "seed %d" % seed)
                self.assertLessEqual(s.state["iteration"], 6)
                k, sk = s.state["k_memory"]["0"]                              # the soft direction: measured, not 378
                self.assertGreater(k, 1.6 * case[4].curvature[0])
                energies = [ft(np.array(it["centre"])) - ft(xt) for _k, it in sorted(s.state["iters"].items(), key=lambda kv: int(kv[0]))]
                self.assertLess(max(energies[1:] or [0.0]), 1.0)             # no centre is above the starting energy

    def test_remembered_force_constant(self):
        p = P2.V2
        # a weak measurement that agrees with the surrogate value within 3 sigma is pulled to it ...
        k, sk, source = P2.combine_curvature(700.0, 165.0, 378.0, p)
        self.assertEqual(source, "combined"); self.assertLess(k, 480.0)
        # ... but not when an earlier centre measured 853 +/- 62 along this direction
        k, sk, source = P2.combine_curvature(700.0, 165.0, 853.0, p, s_prior=P2.memory_error((853.0, 62.0), p))
        self.assertEqual(source, "combined with earlier centres")
        self.assertTrue(700.0 < k < 853.0)
        self.assertAlmostEqual(P2.memory_error((853.0, 62.0), p), p.curvature_prior * 853.0)   # at least 20 %
        self.assertAlmostEqual(P2.memory_error((853.0, 300.0), p), 300.0)
        # a clear measurement far from the remembered value replaces it
        k, sk, source = P2.combine_curvature(2000.0, 100.0, 853.0, p, s_prior=170.0)
        self.assertEqual((k, source), (2000.0, "measured"))
        # no usable measurement: the remembered value
        k, sk, source = P2.combine_curvature(float("nan"), float("inf"), 853.0, p, s_prior=170.0)
        self.assertEqual((k, sk, source), (853.0, 170.0, "earlier centres"))

    def test_the_last_step_must_be_small(self):
        """With the precision reached, the search ends only if the gain of the last step is at most the tolerance:
        that step is not checked by a scan, and its gain is only as good as the model Hessian."""
        p = P2.V2
        n = 3
        stats = {"G": 0.35, "N": 0.02, "V": 0.01}                           # c_n N = 0.052 < 0.2; G - N = 0.33
        self.assertLess(P2.joint_factor(n) * stats["N"], p.eps_total)
        self.assertEqual(P2.joint_decision(stats, n, False, p)[0], "MOVE")
        self.assertEqual(P2.joint_decision(stats, n, False, P2.ProtocolV2(final_gain_factor=0.0))[0], "FINAL")
        self.assertEqual(P2.joint_decision({"G": 0.15, "N": 0.02, "V": 0.01}, n, False, p)[0], "FINAL")
        # a tight tolerance asks for a small last step as well
        tight = P2.ProtocolV2(eps_total=0.02)
        self.assertEqual(P2.joint_decision({"G": 0.020, "N": 0.007, "V": 0.001}, n, False, tight)[0], "FINAL")
        # A gain above the limit that is not resolved (G - N - sqrt(V) below the limit): the search does not move on
        # what may be noise. It takes more blocks, and moves when no more blocks will come.
        noisy = {"G": 0.056, "N": 0.007, "V": 0.001}                        # G - N = 0.049, sqrt(V) = 0.032
        self.assertEqual(P2.joint_decision(noisy, n, False, tight)[0], "CONTINUE")
        self.assertEqual(P2.joint_decision(noisy, n, True, tight)[0], "MOVE")
        self.assertEqual(P2.joint_decision({"G": 0.056, "N": 0.007, "V": 0.0001}, n, False, tight)[0], "MOVE")   # resolved
        self.assertEqual(P2.joint_decision(noisy, n, False, P2.ProtocolV2(eps_total=0.02, endgame_sig=0.0))[0], "MOVE")


class SlotEngine(MockEngine):
    """A batch system with a limited number of job slots: a point starts when a slot is free, in the order of
    submission. A point that is stopped or complete gives its slot back."""

    def __init__(self, *args, slots=2, **kw):
        super().__init__(*args, **kw)
        self.slots = int(slots)

    def _available(self, p):
        if p.get("started") is None:
            return 0
        n = min(max(0, self.tick - p["started"]) * self.blocks_per_tick, p["blocks"])
        if p["stopped_at"] is not None:
            n = min(n, p["stopped_at"])
        return int(n)

    def advance(self, ticks=1):
        for _ in range(int(ticks)):
            busy = sum(1 for p in self.points.values()
                       if p.get("started") is not None and p["stopped_at"] is None and self._available(p) < p["blocks"])
            for p in self.points.values():
                if busy >= self.slots:
                    break
                if p.get("started") is None and p["stopped_at"] is None:
                    p["started"] = self.tick; busy += 1
            self.tick += 1
        self.save()


class TestProtocolV2Driver(unittest.TestCase):
    def test_plan_for_a_reaction_coordinate(self):
        """A reaction coordinate is scanned at 0.5 kcal/mol in place of 5: it adds 10 times the noise energy of
        another direction, and the 95 % factor is that of fewer effective directions."""
        p = P2.V2
        total, factor = p.plan_noise(6, 0)
        self.assertAlmostEqual(total, 6 / 40.0); self.assertAlmostEqual(factor, P2.joint_factor(6))
        total, factor = p.plan_noise(6, 1)
        self.assertAlmostEqual(total, 5 / 40.0 + 1 / 4.0)
        nu = (5 / 40.0 + 1 / 4.0) ** 2 / (5 / 1600.0 + 1 / 16.0)
        self.assertAlmostEqual(nu, 2.1429, places=3)
        self.assertAlmostEqual(factor, P2.bound_factor(nu))
        self.assertGreater(factor, P2.joint_factor(3)); self.assertLess(factor, P2.joint_factor(2))
        # usable blocks for 5 kcal/mol per block: 12.592 x 0.15 x 25 / (6 x 0.2) = 39.4 without a reaction
        # coordinate, 2.917 x 0.375 x 25 / 0.2 = 136.8 with one (3.5 times more)
        wide = P2.ProtocolV2(cap=100000)
        discard = wide.drop_min + wide.min_blocks_after_drop
        self.assertEqual(wide.planned_blocks(6, 5.0) - discard, 40)
        self.assertEqual(wide.planned_blocks(6, 5.0, n_negative=1) - discard, 137)
        self.assertEqual(wide.planned_blocks(6, 5.0, n_negative=0), wide.planned_blocks(6, 5.0))
        with self.assertRaises(ValueError):
            p.plan_noise(3, 4)
        # the factor for a number of directions that is not whole
        self.assertAlmostEqual(P2.bound_factor(1.0), 3.841); self.assertAlmostEqual(P2.bound_factor(0.3), 3.841)
        self.assertAlmostEqual(P2.bound_factor(2.0), 5.991 / 2); self.assertAlmostEqual(P2.bound_factor(6.0), P2.joint_factor(6))
        self.assertAlmostEqual(P2.bound_factor(2.5), (5.991 + 0.5 * (7.815 - 5.991)) / 2.5)
        self.assertAlmostEqual(P2.bound_factor(20.0), P2.joint_factor(20))

    def test_joint_bound_follows_the_directions_that_carry_the_noise(self):
        """One direction carries all the noise energy: the 95 % point is 3.84 N, not c_3 N = 2.6 N."""
        Kh = np.diag([100.0, 100.0, 100.0])
        even = P2.joint_statistics(np.zeros(3), np.array([1.0, 1.0, 1.0]), Kh)
        one = P2.joint_statistics(np.zeros(3), np.array([1.0, 0.0, 0.0]), Kh)
        self.assertAlmostEqual(even["nu"], 3.0); self.assertAlmostEqual(one["nu"], 1.0)
        rng = np.random.default_rng(0)
        z = rng.normal(size=(200000, 3))
        for stats, sg in ((even, np.array([1.0, 1.0, 1.0])), (one, np.array([1.0, 0.0, 0.0]))):
            energy = 0.5 * (z * sg) ** 2 / 100.0
            bound = P2.bound_factor(stats["nu"]) * stats["N"]
            self.assertAlmostEqual(float(np.mean(energy.sum(axis=1) < bound)), 0.95, delta=0.004)
        # unequal shares: the bound must cover at least 95 % (the interpolation is on the safe side or close to it)
        sg = np.array([1.0, 0.5, 0.2])
        mixed = P2.joint_statistics(np.zeros(3), sg, Kh)
        energy = (0.5 * (z * sg) ** 2 / 100.0).sum(axis=1)
        self.assertGreater(float(np.mean(energy < P2.bound_factor(mixed["nu"]) * mixed["N"])), 0.94)
        self.assertLess(float(np.mean(energy < P2.joint_factor(3) * mixed["N"])), 0.93)      # the old factor was too low

    def test_span_limits_after_the_first_scan(self):
        """A force constant that was measured far too low must not give a scan that is much longer than the first
        one; and a shortened span is the span that the rule asked for, also when the normal span changed."""
        case = water_case(); ft, fs, xs, xt, d = case
        with tempfile.TemporaryDirectory() as tmp:
            _eng, s = make(case, P2.V2, tmp, seed=1)
            first = list(s._it(1)["half_span"])
            lam = [float(v) for v in s._it(1)["lam"]]
            for j in range(s.dim):
                self.assertAlmostEqual(first[j], P2.V2.half_span(lam[j]))
                # a force constant 100 times lower would give 10 times the span (or r_max): the limit is 1.5
                self.assertAlmostEqual(s._direction_span(j, lam[j] / 100.0, first[j]), 1.5 * first[j])
                self.assertAlmostEqual(s._direction_span(j, lam[j] * 4.0, first[j]), 0.5 * first[j])      # shorter is free
                self.assertAlmostEqual(s._direction_span(j, lam[j], first[j]), first[j])
            free = DenseScanSearch("free", MockEngine(ft, fs, sigma_block=5.0, seed=1), AnalyticSurrogate(fs),
                                   list(ft.symbols), xs, d.vectors, d.curvature,
                                   protocol=P2.ProtocolV2(span_growth_max=0.0), state_dir=os.path.join(tmp, "free"))
            self.assertAlmostEqual(free._direction_span(0, lam[0] / 100.0, first[0]), P2.V2.r_max)
            # the shape rule: energy 16 times the share of a direction -> half of the span of THIS scan
            target = s._shape_target()
            it = s._it(1)
            ests = [{"k": 2.0 * lam[j], "sg_shape": float(np.sqrt(2.0 * 2.0 * lam[j] * (16.0 * target if j == 0 else 0.0)))}
                    for j in range(s.dim)]
            self.assertEqual(s._update_span_factors(it, ests), [0])
            self.assertAlmostEqual(s.state["span_wanted"]["0"], 0.5 * first[0])
            # the next centre has a force constant 4 times larger: its normal span is half as long already
            self.assertAlmostEqual(s._direction_span(0, lam[0] * 4.0, first[0]), 0.5 * first[0])
            self.assertAlmostEqual(s.state["span_factor"][0], 1.0)
            self.assertEqual(s.state["span_wanted"], {})
            # the same rule with an unchanged force constant: the factor is 0.5
            self.assertEqual(s._update_span_factors(it, ests), [0])
            self.assertAlmostEqual(s._direction_span(0, lam[0], first[0]), 0.5 * first[0])
            self.assertAlmostEqual(s.state["span_factor"][0], 0.5)

    def test_planned_blocks_formula(self):
        """With spans at their normal length, the blocks that the driver expects to need are the planned blocks
        of the protocol."""
        case = water_case(); p = P2.ProtocolV2(symmetry=False)
        with tempfile.TemporaryDirectory() as tmp:
            _eng, s = make(case, p, tmp, seed=1)
            s.tick()
            it = s.state["iters"]["1"]
            for sigma in (2.0, 5.0, 8.0):
                need = s.blocks_needed(it, sigma)
                self.assertAlmostEqual(need, P2.joint_factor(3) * 3 * sigma ** 2 / (8.0 * p.e_cap * p.eps_total), places=9)
                self.assertEqual(int(np.ceil(need)) + p.drop_min + p.min_blocks_after_drop, p.planned_blocks(3, sigma))

    def test_points_do_not_hold_cores_longer_than_the_decision_needs(self):
        """A batch system that runs 2 points at a time. Without the limit, a scan point runs to the block limit of
        the search while the other points wait. With it, a point is stopped once it holds 3 times the usable
        blocks that the precision target needs, and the search ends much earlier with the same quality."""
        case = water_case(); ft, fs, xs, xt, d = case
        used = {}
        for label, factor in (("limit", 3.0), ("no limit", 0.0)):
            proto = P2.ProtocolV2(plan_factor=factor, cap=1500)
            with tempfile.TemporaryDirectory() as tmp:
                eng = SlotEngine(ft, fs, slots=2, sigma_block=5.0, blocks_per_tick=10, seed=5, transient=20.0, equil_blocks=0.6)
                s = DenseScanSearch("min", eng, AnalyticSurrogate(fs), list(ft.symbols), xs, d.vectors, d.curvature,
                                    protocol=proto, state_dir=tmp)
                self.assertEqual(s.run(max_ticks=20000), CONVERGED, label)
                self.assertLess(ft(s.final_geometry()) - ft(xt), 0.2, label)
                it = s.state["iters"]["1"]
                scan = [key for pts in it["points"].values() for key in pts.values()]
                used[label] = {"ticks": eng.tick, "scan": max(eng.read(key).nblocks for key in scan),
                               "trimmed": dict(it.get("trimmed", {})), "blocks": it["composed"]["blocks"],
                               "bound": it["composed"]["joint_bound"]}
                if factor:
                    # the scan points that ran first were stopped at the limit (the decision came before the
                    # last one reached it), and none of the reference runs
                    self.assertTrue(set(used[label]["trimmed"]) <= set(scan))
                    self.assertGreaterEqual(len(used[label]["trimmed"]), len(scan) - 1)
                    need = s.blocks_needed(it, 5.0)
                    for key, n in used[label]["trimmed"].items():
                        self.assertLess(n, proto.cap)
                        self.assertGreaterEqual(n, 3.0 * 0.8 * need + proto.drop_max)   # noise estimate within 0.9 .. 1.2
                        self.assertLessEqual(n, 3.0 * 1.5 * need + proto.drop_max + 10)
                    for key in it["centre_keys"]:
                        self.assertNotIn(key, used[label]["trimmed"])
                        self.assertEqual(eng.read(key).nblocks, proto.cap)
                else:
                    self.assertEqual(used[label]["trimmed"], {})
        if os.environ.get("MONTEPASS_TEST_VERBOSE"):
            print(used)
        self.assertLess(used["limit"]["bound"], 0.6 * proto.eps_total)        # 3 times the needed blocks: a clear margin
        self.assertLess(used["limit"]["scan"], 0.25 * used["no limit"]["scan"])
        self.assertLess(used["limit"]["ticks"], 0.75 * used["no limit"]["ticks"])

    def test_stopped_points_of_a_pair_always_have_usable_data(self):
        """Ammonia at its surrogate minimum, 16 random seeds. Points are stopped at different block counts, and
        the number of discarded blocks differs from trace to trace. No search may end for lack of usable blocks
        (a rule that counted the usable blocks of each point alone did, for seed 2)."""
        fs = M.ammonia(False); ft = M.ammonia(True)
        x0 = M.ammonia_geometry()
        xt, xs = M.minimize(ft, x0), M.minimize(fs, x0)
        Hs = M.hessian(fs, xs); sub = S.mode_subspace(Hs, xs)
        d = S.search_directions(S.project_hessian(Hs, sub.basis), sub)
        for seed in range(16):
            with tempfile.TemporaryDirectory() as tmp:
                eng = MockEngine(ft, fs, sigma_block=5.0, blocks_per_tick=10, seed=seed, transient=20.0, equil_blocks=0.6)
                s = DenseScanSearch("t", eng, AnalyticSurrogate(fs), list(ft.symbols), xs, d.vectors, d.curvature,
                                    protocol=P2.V2, state_dir=tmp)
                self.assertEqual(s.run(), CONVERGED, "seed %d: %s" % (seed, s.state.get("message")))

    def test_a_scan_that_is_too_short_is_repeated_with_more_blocks(self):
        """The block limit is too low for the tolerance. The search does not end as NOT_CONVERGED: it takes the
        step and scans again with the number of blocks that the measured noise asks for."""
        case = water_case(); ft, xt = case[0], case[3]
        with tempfile.TemporaryDirectory() as tmp:
            proto = P2.ProtocolV2(cap=60, eps_total=0.05)
            _eng, s = make(case, proto, tmp, seed=4)
            self.assertEqual(s.run(), CONVERGED)
            first = s.state["iters"]["1"]
            self.assertEqual(first["composed"]["verdict"], "MOVE")
            self.assertGreaterEqual(first["composed"]["joint_bound"], proto.eps_total)
            self.assertIn("repeated with up to", first["decision"]["note"])
            self.assertEqual(first["point_blocks"], 60)
            more = s.state["scan_blocks"]
            self.assertTrue(1.2 * 60 <= more <= 4 * 60, more)
            last = s.state["iters"][str(s.state["iteration"])]
            self.assertEqual(last["point_blocks"], more)
            self.assertLess(last["composed"]["joint_bound"], proto.eps_total)
            self.assertGreater(last["composed"]["blocks"], 60)
            self.assertLess(ft(s.final_geometry()) - ft(xt), 0.05)
        # when more blocks are not allowed, the search ends and says that the target was not reached
        with tempfile.TemporaryDirectory() as tmp:
            _eng, s = make(case, P2.ProtocolV2(cap=60, eps_total=0.05, cap_scan_factor=1.0), tmp, seed=4)
            self.assertEqual(s.run(), NOT_CONVERGED)
            self.assertEqual(s.state["iteration"], 1)
            self.assertIsNone(s.state.get("scan_blocks"))

    def test_converges_with_half_the_points(self):
        case = water_case(); ft, xt = case[0], case[3]
        with tempfile.TemporaryDirectory() as tmp:
            _eng, s = make(case, P2.V2, tmp, seed=1)
            self.assertEqual(s.run(), CONVERGED)
            self.assertLess(ft(s.final_geometry()) - ft(xt), P2.V2.eps_total)
            self.assertEqual(len(s.state["iters"]["1"]["points"]["0"]), 2)             # one shell, two sides
            est = s.state["iters"]["1"]["decided"]["0"]["res"]
            self.assertTrue(est["control_variate"])

    def test_far_start_converges(self):
        case = water_case(shift=0.20); ft, xt = case[0], case[3]
        self.assertGreater(ft(case[2]) - ft(xt), 50.0)                                 # the start is 59 kcal/mol up
        with tempfile.TemporaryDirectory() as tmp:
            _eng, s = make(case, P2.V2, tmp, seed=3)
            status = s.run()
            self.assertIn(status, (CONVERGED, NOT_CONVERGED))
            self.assertLess(ft(s.final_geometry()) - ft(xt), 0.5)

    def test_pooling_uses_fewer_blocks(self):
        case = water_case(shift=0.05)
        totals = {}
        for label, proto in (("plain", P2.ProtocolV2(pool=False)), ("pool", P2.ProtocolV2(pool=True))):
            blocks = []
            for seed in range(4):
                with tempfile.TemporaryDirectory() as tmp:
                    _eng, s = make(case, proto, tmp, seed=seed)
                    s.run(); blocks.append(s.search_blocks())
            totals[label] = float(np.mean(blocks))
        self.assertLess(totals["pool"], totals["plain"] * 1.10)

    def test_final_runs_and_energy_change(self):
        case = water_case(shift=0.05); ft, xt, xs = case[0], case[3], case[2]
        with tempfile.TemporaryDirectory() as tmp:
            _eng, s = make(case, P2.V2, tmp, seed=2)
            self.assertEqual(s.run(), CONVERGED)
            self.assertEqual(len(s.state["final"]["keys"]), 2)
            ef, e0 = s.final_energy(), s.start_energy()
            true_change = ft(s.final_geometry()) - ft(xs)
            de, sd = s.state["final"]["energy_change"]
            self.assertLess(abs(de - true_change), 4.0 * sd)                # the measured change agrees with the true one
            self.assertLess(de, -2.0)                                         # the start was 3.4 kcal/mol above the minimum
            self.assertLess(ef[1], 0.01); self.assertLess(e0[1], 0.01)       # full-length runs: small error bars (hartree)

    def test_saddle_search(self):
        ft, fs = M.ammonia(True), M.ammonia(False)
        xs = M.refine_stationary(fs, M.ammonia_geometry(theta=120.0, r=1.0))
        xt = M.refine_stationary(ft, M.ammonia_geometry(theta=120.0, r=1.0))
        Hs = M.hessian(fs, xs); sub = S.mode_subspace(Hs, xs)
        d = S.search_directions(S.project_hessian(Hs, sub.basis), sub)
        self.assertEqual(d.n_negative, 1)
        with tempfile.TemporaryDirectory() as tmp:
            eng = MockEngine(ft, fs, sigma_block=5.0, blocks_per_tick=10, seed=5, transient=20.0, equil_blocks=0.6)
            # the energy to gain at this saddle is 0.3 kcal/mol, so a tight tolerance is needed to see the geometry move
            s = DenseScanSearch("saddle", eng, AnalyticSurrogate(fs), list(ft.symbols), xs, d.vectors, d.curvature,
                                protocol=P2.ProtocolV2(eps_total=0.05, g_move=0.1, e_cap_negative=2.0), state_dir=tmp)
            self.assertEqual(s.state["roles"], [-1, 1, 1, 1, 1, 1])
            self.assertEqual(s.run(), CONVERGED)
            xf = s.final_geometry()
            self.assertAlmostEqual(G.distance(xf, 0, 1), G.distance(xt, 0, 1), delta=0.006)
            self.assertAlmostEqual(G.angle(xf, 1, 0, 2), 120.0, delta=1.0)             # still planar: a saddle, not a minimum

    def test_rejected_step_is_halved(self):
        """A surrogate with a wrong coupling makes a bad coupled step; the energy test must catch it."""
        case = water_case(shift=0.10); ft, xt = case[0], case[3]
        with tempfile.TemporaryDirectory() as tmp:
            _eng, s = make(case, P2.ProtocolV2(secant=False, max_iter=14), tmp, seed=0)
            s.run()
            self.assertLess(ft(s.final_geometry()) - ft(xt), 1.0)
            if s.state["rejected"]:
                rejected = [it for it in s.state["iters"].values() if it.get("accepted") is False]
                self.assertTrue(rejected)
                self.assertLess(s.state["trust_scale"], 1.0 + 1e-12)


class TestV2Estimator(unittest.TestCase):
    def test_control_variate_removes_the_cubic_bias(self):
        # E(t) = 0.5 k t^2 + c3 t^3, minimum at t = 0; the surrogate has the same cubic term and a slope
        k, c3, R, Hk = 1000.0, -900.0, 0.1, 627.509474
        e = lambda t: (0.5 * k * t * t + c3 * t ** 3) / Hk
        sur = lambda t: (0.5 * k * t * t + c3 * t ** 3 + 3.0 * t) / Hk
        sh = [P2.Shell(t=R, e_plus=e(R), s_plus=1e-5, e_minus=e(-R), s_minus=1e-5, sur_plus=sur(R), sur_minus=sur(-R))]
        with_cv = P2.estimate(sh, centre=(e(0.0), 1e-5), sur_centre=sur(0.0), g_sur=3.0, k_sur=k)
        no_cv = P2.estimate(sh, centre=(e(0.0), 1e-5), sur_centre=None, g_sur=None, k_sur=k,
                            protocol=P2.ProtocolV2(control_variate=False))
        self.assertAlmostEqual(with_cv["g"], 0.0, delta=0.05)              # exact slope recovered
        self.assertAlmostEqual(no_cv["g"], c3 * R * R, delta=0.05)         # biased by c3 R^2 = -9 kcal/mol/A
        self.assertAlmostEqual(with_cv["k"], k, delta=5.0)

    def test_remembered_force_constant_is_replaced_only_by_a_better_measurement(self):
        p = P2.V2
        # weak measurement far above the remembered value (an upward fluctuation): the remembered value stays
        k, sk, source = P2.combine_curvature(8138.0, 2213.0, 1086.0, p, s_prior=217.0)
        self.assertEqual((k, sk, source), (1086.0, 217.0, "earlier centres"))
        k, sk, source = P2.combine_curvature(2000.0, 300.0, 853.0, p, s_prior=171.0)
        self.assertEqual((k, source), (853.0, "earlier centres"))
        # a clear measurement far BELOW the remembered value is taken, also when it is less precise: the smaller
        # force constant makes the bound larger
        k, sk, source = P2.combine_curvature(853.0, 300.0, 2000.0, p, s_prior=171.0)
        self.assertEqual((k, sk, source), (853.0, 300.0, "measured"))
        # but not a measurement of the other sign that is less precise
        k, sk, source = P2.combine_curvature(-700.0, 300.0, 2000.0, p, s_prior=171.0)
        self.assertEqual((k, source), (2000.0, "earlier centres"))
        # a measurement that is more precise than the remembered value replaces it
        k, sk, source = P2.combine_curvature(2400.0, 150.0, 1086.0, p, s_prior=217.0)
        self.assertEqual((k, sk, source), (2400.0, 150.0, "measured"))
        # a clear force constant against a value that is known only to 100 %: the measurement
        k, sk, source = P2.combine_curvature(9321.0, 469.0, 2003.0, p, s_prior=2003.0)
        self.assertEqual((k, source), (9321.0, "measured"))
        # agreement within 3 sigma: the weighted mean
        k, sk, source = P2.combine_curvature(1300.0, 300.0, 1086.0, p, s_prior=217.0)
        self.assertEqual(source, "combined with earlier centres")
        self.assertTrue(1086.0 < k < 1300.0); self.assertLess(sk, 217.0)

    def test_coupling_fit(self):
        """The optional fit of the couplings: exact slope changes give the true coupling back; slope changes that
        the model predicts give no fit; a fit that would make the Hessian unsafe is reduced."""
        p = P2.ProtocolV2(coupling_fit=True)
        K = np.array([[1000.0, 600.0, 0.0], [600.0, 2000.0, 0.0], [0.0, 0.0, 2500.0]])
        kd = np.diag(K).copy()
        roles = np.array([1, 1, 1])
        steps = [np.array([0.03, 0.02, 0.01]), np.array([-0.01, 0.03, 0.0])]
        err = np.array([2.0, 2.0, 2.0])
        pairs = [(s, K @ s, err, kd) for s in steps]
        fit = P2.fit_couplings(kd, np.diag(kd), pairs, roles, p)
        self.assertIsNotNone(fit)
        self.assertAlmostEqual(fit["C"][0, 1], 600.0, delta=30.0)            # the prior pulls it down a little
        self.assertAlmostEqual(fit["C"][0, 1], fit["C"][1, 0])
        self.assertLess(abs(fit["C"][0, 2]), 60.0); self.assertLess(abs(fit["C"][1, 2]), 60.0)
        self.assertEqual(fit["dof"], 6); self.assertEqual(fit["shrink"], 1.0)
        self.assertLess(fit["chi2_fit"], 0.5 * fit["chi2_base"])
        np.testing.assert_allclose(np.diag(fit["C"]), 0.0)
        # the model with the right couplings predicts the slope changes: nothing to fit
        self.assertIsNone(P2.fit_couplings(kd, K, pairs, roles, p))
        # slope changes of a diagonal surface with large errors: no evidence against the diagonal model
        weak = [(s, np.diag(kd) @ s + np.array([1.0, -1.0, 0.5]), err, kd) for s in steps]
        self.assertIsNone(P2.fit_couplings(kd, np.diag(kd), weak, roles, p))
        # a coupling larger than the diagonal allows (the matrix would not be positive definite): reduced
        big = np.array([[1000.0, 1500.0], [1500.0, 2000.0]])
        s2 = [np.array([0.03, 0.02]), np.array([-0.02, 0.03])]
        fit2 = P2.fit_couplings(np.diag(big).copy(), np.diag(np.diag(big)), [(s, big @ s, err[:2], np.diag(big).copy()) for s in s2],
                                np.array([1, 1]), p)
        if fit2 is not None:
            mu = np.linalg.eigvalsh((np.diag(np.diag(big)) + fit2["C"]) / np.sqrt(np.outer(np.diag(big), np.diag(big))))
            self.assertGreaterEqual(mu.min(), p.coupling_fit_floor - 1e-9)
            self.assertLess(fit2["shrink"], 1.0)
        # a slope fixed by symmetry (error 0) is left out; one direction alone has no coupling
        sym = [(s, K @ s, np.array([2.0, 2.0, 0.0]), kd) for s in steps]
        self.assertEqual(P2.fit_couplings(kd, np.diag(kd), sym, roles, p)["dof"], 4)
        self.assertIsNone(P2.fit_couplings(kd[:1], np.diag(kd[:1]), [(np.array([0.1]), np.array([1.0]), np.array([1.0]), kd[:1])],
                                           np.array([1]), p))
        self.assertAlmostEqual(P2.chi2_point(1, 1.645), 3.84, delta=0.1)
        self.assertAlmostEqual(P2.chi2_point(6, 3.09), 22.46, delta=0.3)       # exact 22.46; the formula gives 22.67
        self.assertFalse(P2.V2.coupling_fit); self.assertFalse(P2.V2.shape_distrust)        # both are off by default

    def test_error_bars_of_exactly_zero(self):
        """A model without noise: pair and centre have an error of exactly zero. No division by zero, no warning."""
        import warnings
        f = lambda t: 300.0 * t * t + 2.0 * t                   # kcal/mol: k = 600, slope 2 at the centre
        H = HARTREE_TO_KCAL
        shell = P2.Shell(t=0.05, e_plus=f(0.05) / H, s_plus=0.0, e_minus=f(-0.05) / H, s_minus=0.0,
                         sur_plus=f(0.05) / H, sur_minus=f(-0.05) / H)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            e = P2.estimate([shell], centre=(0.0, 0.0), sur_centre=0.0, g_sur=2.0, k_sur=600.0, prior_g=None,
                            role=1, protocol=P2.V2)
        self.assertAlmostEqual(e["g"], 2.0, places=9)
        self.assertAlmostEqual(e["k"], 600.0, places=6)
        self.assertTrue(np.isfinite(e["sg"]))

    def test_wrong_curvature_sign_is_flagged(self):
        Hk = 627.509474
        e = lambda t: (-0.5 * 300.0 * t * t + 40.0 * t) / Hk             # negative curvature, rising to +t
        sh = [P2.Shell(t=0.1, e_plus=e(0.1), s_plus=2e-5, e_minus=e(-0.1), s_minus=2e-5)]
        est = P2.estimate(sh, centre=(e(0.0), 2e-5), k_sur=400.0, role=1)
        self.assertFalse(est["curv_ok"]); self.assertEqual(est["k_source"], "measured")
        self.assertAlmostEqual(est["k"], -300.0, delta=5.0)
        est = P2.estimate(sh, centre=(e(0.0), 2e-5), k_sur=-280.0, role=-1)    # the same curve as a reaction coordinate
        self.assertTrue(est["curv_ok"]); self.assertAlmostEqual(est["m"], 40.0 / 300.0, delta=0.01)

    def test_joint_statistics_and_decision(self):
        Hm = np.diag([400.0, 2000.0, 2500.0])
        sg = np.array([2.0, 4.0, 5.0])
        st = P2.joint_statistics(np.zeros(3), sg, Hm)
        self.assertAlmostEqual(st["G"], 0.0)
        self.assertAlmostEqual(st["N"], 0.5 * (4.0 / 400 + 16.0 / 2000 + 25.0 / 2500))
        self.assertEqual(P2.joint_decision(st, 3, final=False)[0], "FINAL")     # 2.6 * 0.014 = 0.036 < 0.2
        st = P2.joint_statistics(np.array([40.0, 0.0, 0.0]), sg, Hm)             # G = 2.0 kcal/mol, well resolved
        self.assertAlmostEqual(st["G"], 2.0)
        self.assertEqual(P2.joint_decision(st, 3, final=False)[0], "MOVE")
        st = P2.joint_statistics(np.zeros(3), sg * 8.0, Hm)                       # too noisy to certify
        self.assertEqual(P2.joint_decision(st, 3, final=False)[0], "CONTINUE")
        self.assertEqual(P2.joint_decision(st, 3, final=True)[0], "FINAL")
        # a negative eigenvalue (saddle) enters by its absolute value
        st2 = P2.joint_statistics(np.array([0.0, 10.0, 0.0]), sg, np.diag([400.0, -200.0, 2500.0]))
        self.assertAlmostEqual(st2["G"], 0.5 * 100.0 / 200.0)
        self.assertAlmostEqual(P2.joint_factor(6), 12.592 / 6, places=6)

    def test_equilibration_rule(self):
        rng = np.random.default_rng(0)
        trace = rng.normal(size=300); trace[:3] += np.array([40.0, 8.0, 1.5])
        self.assertEqual(P2.mser_drop(trace, 10, 50), 10)                    # a short transient: the lower limit
        slow = rng.normal(size=300) + 15.0 * np.exp(-np.arange(300) / 12.0)
        self.assertGreaterEqual(P2.mser_drop(slow, 10, 50), 30)              # a slow transient: more blocks dropped
        self.assertLessEqual(P2.mser_drop(slow, 10, 50), 50)

    def test_inconsistent_prior_slope_is_ignored(self):
        Hk = 627.509474
        e = lambda t: (0.5 * 800.0 * t * t + 20.0 * t) / Hk
        sh = [P2.Shell(t=0.1, e_plus=e(0.1), s_plus=1e-4, e_minus=e(-0.1), s_minus=1e-4)]
        est = P2.estimate(sh, centre=(e(0.0), 1e-4), k_sur=800.0, prior_g=(-50.0, 0.2))
        self.assertFalse(est["pooled"]); self.assertAlmostEqual(est["g"], 20.0, delta=0.5)
        est = P2.estimate(sh, centre=(e(0.0), 1e-4), k_sur=800.0, prior_g=(20.3, 0.2))
        self.assertTrue(est["pooled"]); self.assertLess(est["sg"], est["sg_meas"])


if __name__ == "__main__":
    unittest.main()
