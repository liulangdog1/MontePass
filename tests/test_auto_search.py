"""The automatic search: classic line search first, dense-scan search when the first iteration cannot be trusted."""
import json
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from montepass import model as M                     # noqa: E402
from montepass import subspace as S        # noqa: E402
from montepass.engine.mock import MockEngine        # noqa: E402
from montepass.search import classic as C        # noqa: E402
from montepass.search import protocol_v2 as P2        # noqa: E402
from montepass.search.auto import CLASSIC, DENSE, AutoSearch        # noqa: E402
from montepass.surrogate.analytic import AnalyticSurrogate        # noqa: E402


def water_case():
    ft, fs = M.water(True), M.water(False)
    xt = M.minimize(ft, M.water_geometry())
    xs = M.minimize(fs, M.water_geometry())
    Hs = M.hessian(fs, xs); sub = S.mode_subspace(Hs, xs)
    d = S.search_directions(S.project_hessian(Hs, sub.basis), sub)
    return ft, fs, xs, xt, d


class TestAutoSearch(unittest.TestCase):
    def test_wide_windows_hand_over_to_the_dense_search(self):
        """Model water, block noise 10 kcal/mol. The sizer of the classic search chooses windows of 0.28 to
        0.92 A, the audit does not resolve the two symmetric directions, and the dense-scan search takes over."""
        ft, fs, xs, xt, d = water_case()
        with tempfile.TemporaryDirectory() as tmp:
            eng = MockEngine(ft, fs, sigma_block=10.0, blocks_per_tick=20, seed=1, transient=20.0, equil_blocks=0.6)
            logged = []
            s = AutoSearch("geo", eng, AnalyticSurrogate(fs), list(ft.symbols), xs, d.vectors, d.curvature,
                           protocol=P2.ProtocolV2(), state_dir=tmp, settings={"sigma_block": 10.0}, log=logged.append)
            self.assertEqual(s.phase, CLASSIC)
            status = s.run()
            self.assertEqual(s.phase, DENSE)
            self.assertIn(status, ("CONVERGED", "NOT_CONVERGED"))
            self.assertIn("stability audit did not resolve", s.meta["reason"])
            self.assertTrue(any("dense-scan search starts" in line for line in logged))
            # the record of the stopped classic search, and the state of the search that is active now
            with open(os.path.join(tmp, "geo_classic_state.json")) as fh:
                old = json.load(fh)
            self.assertEqual((old["protocol"], old["status"], sorted(old["iters"])), ("classic", C.UNRESOLVED_STATUS, ["1"]))
            with open(os.path.join(tmp, "geo_state.json")) as fh:
                new = json.load(fh)
            self.assertEqual(new["protocol"], "v2")
            # the full-length runs at the start are used again: same keys, and each ran once
            self.assertEqual(new["iters"]["1"]["centre_keys"], old["iters"]["1"]["centre_keys"])
            np.testing.assert_allclose(new["iters"]["1"]["centre"], xs, atol=1e-12)
            # no scan point of the dense search has the name of a point of the classic search
            classic_points = {k for pts in old["iters"]["1"]["points"].values() for k in pts.values()}
            dense_points = {k for it in new["iters"].values() for pts in it["points"].values() for k in pts.values()}
            self.assertEqual(classic_points & dense_points, set())
            # the result: close to the true minimum, and not above the start
            self.assertLess(ft(s.final_geometry()) - ft(xt), 0.15)
            self.assertGreater(ft(xs) - ft(xt), 0.20)
            # a new object on the same folder continues with the dense search and gives the same answer
            s2 = AutoSearch("geo", eng, AnalyticSurrogate(fs), list(ft.symbols), xs, d.vectors, d.curvature,
                            protocol=P2.ProtocolV2(), state_dir=tmp, settings={"sigma_block": 10.0})
            self.assertEqual((s2.phase, s2.status), (DENSE, status))
            np.testing.assert_allclose(s2.final_geometry(), s.final_geometry(), atol=1e-12)

    def test_a_resolved_first_iteration_keeps_the_classic_search(self):
        """One direction, a parabola, a narrow window: the audit resolves it and the classic search runs to its end."""
        x0 = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.2]])
        v = np.array([0.0, 0.0, -1.0, 0.0, 0.0, 1.0]) / np.sqrt(2.0)
        k = 880.0

        def surface(x):
            t = float(np.dot(np.asarray(x, dtype=float).reshape(-1) - x0.reshape(-1), v))
            return 0.5 * k * (t - 0.02) ** 2

        with tempfile.TemporaryDirectory() as tmp:
            eng = MockEngine(surface, sigma_block=0.2, blocks_per_tick=50, seed=3)
            settings = {"sigma_block": 0.2, "classic": {"windows": [0.5 * k * 0.08 ** 2], "noises": [0.05], "iterations": 2,
                                                        "warmup": 1, "final_series": 1, "min_distance": 0.1}}
            s = AutoSearch("line", eng, None, ["H", "H"], x0, v[None, :], [k], protocol=P2.ProtocolV2(), state_dir=tmp,
                           settings=settings)
            self.assertEqual(s.run(), C.DONE_STATUS)
            self.assertEqual(s.phase, CLASSIC)
            self.assertFalse(os.path.exists(os.path.join(tmp, "line_classic_state.json")))
            self.assertEqual(s.state["protocol"], "classic")
            self.assertEqual(s.state["iters"]["1"]["audit"][0]["verdict"], "RESOLVED")
            self.assertAlmostEqual(float(s.coordinates()[-1, 0]), 0.02, delta=0.004)

    def test_an_interrupted_switch_is_finished_by_the_next_object(self):
        ft, fs, xs, xt, d = water_case()
        with tempfile.TemporaryDirectory() as tmp:
            eng = MockEngine(ft, fs, sigma_block=10.0, blocks_per_tick=20, seed=1, transient=20.0, equil_blocks=0.6)
            args = ("geo", eng, AnalyticSurrogate(fs), list(ft.symbols), xs, d.vectors, d.curvature)
            kw = dict(protocol=P2.ProtocolV2(), state_dir=tmp, settings={"sigma_block": 10.0})
            s = AutoSearch(*args, **kw)
            for _ in range(2000):
                if s.active.state["status"] == C.UNRESOLVED_STATUS:
                    break
                eng.advance()
                if s.active.tick() == C.UNRESOLVED_STATUS:          # the classic search alone: no switch yet
                    break
            self.assertEqual(s.active.state["status"], C.UNRESOLVED_STATUS)
            # the process stops after it wrote "switching" and before it moved any file
            s.meta.update(phase="switching", reason="test"); s._save_meta()
            s2 = AutoSearch(*args, **kw)
            self.assertEqual(s2.phase, DENSE)
            self.assertTrue(os.path.exists(os.path.join(tmp, "geo_classic_state.json")))
            self.assertEqual(s2.state["protocol"], "v2")
            self.assertIn(s2.run(), ("CONVERGED", "NOT_CONVERGED"))


if __name__ == "__main__":
    unittest.main()
