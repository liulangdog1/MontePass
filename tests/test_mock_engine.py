"""The mock engine and the model surfaces."""
import os
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

from montepass import geometry as G, model as M, subspace as S  # noqa: E402
from montepass.engine import PointSpec, ABSENT, QUEUED, RUNNING, DONE, STOPPED  # noqa: E402
from montepass.engine.mock import MockEngine  # noqa: E402
from montepass.stats import reblock  # noqa: E402
from montepass.units import HARTREE_TO_KCAL  # noqa: E402


class TestModels(unittest.TestCase):
    def test_water_minimum_and_surrogate_offset(self):
        ff, fs = M.water(True), M.water(False)
        xt = M.minimize(ff, M.water_geometry()); xs = M.minimize(fs, M.water_geometry())
        self.assertAlmostEqual(G.distance(xt, 0, 1), 0.9584, places=4)
        self.assertAlmostEqual(G.angle(xt, 1, 0, 2), 104.48, places=2)
        self.assertAlmostEqual(G.distance(xs, 0, 1), 0.9704, places=4)
        self.assertGreater(ff(xs) - ff(xt), 0.1)             # the surrogate minimum is not the true minimum
        self.assertLess(ff(xs) - ff(xt), 0.5)

    def test_ammonia_inversion(self):
        ff = M.ammonia(True)
        up = M.minimize(ff, M.ammonia_geometry(up=True))
        planar = M.refine_stationary(ff, M.ammonia_geometry(theta=120.0, r=1.0))
        barrier = ff(planar) - ff(up)
        self.assertGreater(barrier, 4.5); self.assertLess(barrier, 6.5)
        T = M.internal_basis(planar)
        w = np.linalg.eigvalsh(T @ M.hessian(ff, planar) @ T.T)
        self.assertEqual(int(np.sum(w < -1.0)), 1)           # exactly one negative curvature: a first-order saddle
        self.assertLess(G.distance(planar, 0, 1), G.distance(up, 0, 1))     # the bond is shorter at the planar saddle

    def test_band_finds_the_surrogate_barrier(self):
        fs = M.ammonia(False)
        a = M.minimize(fs, M.ammonia_geometry(up=True)); b = M.minimize(fs, M.ammonia_geometry(up=False))
        path, E = M.nudged_elastic_band(fs, M.interpolate_path(a, b, 9))
        saddle = M.refine_stationary(fs, M.ammonia_geometry(theta=120.0, r=1.0))
        self.assertAlmostEqual(E.max() - E[0], fs(saddle) - fs(a), delta=0.02)
        spacing = [np.linalg.norm(path[i + 1] - path[i]) for i in range(8)]
        self.assertLess(max(spacing) / min(spacing), 1.1)    # images evenly spread
        sub = S.pca_subspace(path, variance=0.999)
        self.assertLessEqual(sub.dim, 3)
        self.assertGreater(sub.explained_variance_ratio[0], 0.95)

    def test_shifted_surface(self):
        ff = M.water(True); x = M.water_geometry()
        d = np.zeros((3, 3)); d[1, 1] = 0.05
        self.assertAlmostEqual(ff.shifted(d)(x + d), ff(x), places=12)


class TestMockEngine(unittest.TestCase):
    def setUp(self):
        self.ff = M.water(True); self.x = M.minimize(self.ff, M.water_geometry())
        self.spec = PointSpec(key="p/a", symbols=["O", "H", "H"], coords=self.x, blocks=400)

    def test_lifecycle(self):
        eng = MockEngine(self.ff, M.water(False), sigma_block=1.0, blocks_per_tick=100, queue_ticks=1, seed=3)
        self.assertEqual(eng.read("p/a").state, ABSENT)
        eng.submit(self.spec)
        self.assertEqual(eng.read("p/a").state, QUEUED)
        eng.advance(); self.assertEqual(eng.read("p/a").state, QUEUED)       # still in the queue
        eng.advance(); d = eng.read("p/a")
        self.assertEqual(d.state, RUNNING); self.assertEqual(d.nblocks, 100)
        eng.advance(5); d = eng.read("p/a")
        self.assertEqual(d.state, DONE); self.assertEqual(d.nblocks, 400)
        self.assertIsNotNone(d.e_surrogate)
        eng.submit(self.spec)                                                # second submit: no effect
        self.assertEqual(eng.read("p/a").nblocks, 400)

    def test_statistics(self):
        eng = MockEngine(self.ff, sigma_block=2.0, blocks_per_tick=4000, seed=11)
        eng.submit(PointSpec(key="p/b", symbols=["O", "H", "H"], coords=self.x, blocks=4000))
        eng.advance(); d = eng.read("p/b")
        mean, err = reblock(d.energy, d.weight)
        expected_err = 2.0 / np.sqrt(4000) / HARTREE_TO_KCAL
        self.assertAlmostEqual(err / expected_err, 1.0, delta=0.35)
        self.assertLess(abs(mean - self.ff(self.x) / HARTREE_TO_KCAL), 5 * expected_err)

    def test_stop_keeps_partial_data_and_prefix_is_stable(self):
        eng = MockEngine(self.ff, sigma_block=1.0, blocks_per_tick=100, seed=5)
        eng.submit(self.spec); eng.advance()
        first = eng.read("p/a").energy.copy()
        eng.advance(); eng.stop("p/a"); eng.advance(3)
        d = eng.read("p/a")
        self.assertEqual(d.state, STOPPED); self.assertEqual(d.nblocks, 200)
        np.testing.assert_array_equal(d.energy[:100], first)                # reading more never changes earlier blocks
        self.assertAlmostEqual(eng.cost("p/a"), 200.0)

    def test_transient_and_series(self):
        eng = MockEngine(self.ff, sigma_block=0.1, transient=20.0, equil_blocks=10.0, blocks_per_tick=400, seed=7)
        eng.submit(self.spec)
        eng.submit(PointSpec(key="p/a2", symbols=["O", "H", "H"], coords=self.x, blocks=400, series=2))
        eng.advance()
        e1 = eng.read("p/a").energy * HARTREE_TO_KCAL; e2 = eng.read("p/a2").energy * HARTREE_TO_KCAL
        self.assertGreater(e1[0] - e1[200:].mean(), 15.0)                    # starts high, relaxes
        self.assertFalse(np.allclose(e1, e2))                                # a second series is independent

    def test_state_file_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            sf = os.path.join(tmp, "mock.json")
            eng = MockEngine(self.ff, sigma_block=1.0, blocks_per_tick=100, seed=5, state_file=sf)
            eng.submit(self.spec); eng.advance(2); before = eng.read("p/a").energy.copy()
            eng2 = MockEngine(self.ff, sigma_block=1.0, blocks_per_tick=100, seed=5, state_file=sf)
            d = eng2.read("p/a")
            self.assertEqual(d.nblocks, 200); np.testing.assert_array_equal(d.energy, before)


if __name__ == "__main__":
    unittest.main()
