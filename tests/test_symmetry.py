"""Point-group symmetry and its use by the v2 search."""
import os
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

from montepass import model as M, subspace as S  # noqa: E402
from montepass.engine.mock import MockEngine  # noqa: E402
from montepass.search import protocol_v2 as P2, symmetry as SYM  # noqa: E402
from montepass.search.dense import CONVERGED, DenseScanSearch  # noqa: E402
from montepass.surrogate.analytic import AnalyticSurrogate  # noqa: E402


def directions(surface, x):
    Hs = M.hessian(surface, x); sub = S.mode_subspace(Hs, x)
    return S.search_directions(S.project_hessian(Hs, sub.basis), sub)


class TestOperations(unittest.TestCase):
    def test_group_orders(self):
        water = M.minimize(M.water(True), M.water_geometry())
        self.assertEqual(len(SYM.find_operations(["O", "H", "H"], water)) + 1, 4)            # C2v
        nh3 = M.minimize(M.ammonia(True), M.ammonia_geometry())
        self.assertEqual(len(SYM.find_operations(["N", "H", "H", "H"], nh3)) + 1, 6)         # C3v
        planar = M.refine_stationary(M.ammonia(True), M.ammonia_geometry(theta=120.0, r=1.0))
        self.assertEqual(len(SYM.find_operations(["N", "H", "H", "H"], planar)) + 1, 12)     # D3h
        bent = water.copy(); bent[1, 0] += 0.05; bent[2, 1] += 0.03
        self.assertEqual(len(SYM.find_operations(["O", "H", "H"], bent)), 1)                # three atoms always share a plane
        twisted = nh3.copy(); twisted[1, 0] += 0.05; twisted[2, 2] += 0.04; twisted[3, 1] -= 0.03
        self.assertEqual(len(SYM.find_operations(["N", "H", "H", "H"], twisted)), 0)        # no symmetry left

    def test_direction_classes(self):
        ff = M.water(True); x = M.minimize(ff, M.water_geometry())
        d = directions(ff, x)
        classes, order = SYM.classify_directions(d.vectors, ["O", "H", "H"], x)
        self.assertEqual(order, 4)
        self.assertEqual(sorted(classes), ["odd", "symmetric", "symmetric"])                 # the antisymmetric stretch is odd
        fa = M.ammonia(True); planar = M.refine_stationary(fa, M.ammonia_geometry(theta=120.0, r=1.0))
        d = directions(fa, planar)
        classes, order = SYM.classify_directions(d.vectors, ["N", "H", "H", "H"], planar)
        self.assertEqual(classes.count("symmetric"), 1)                                      # only the breathing mode
        self.assertEqual(classes[0], "odd")                                                  # the umbrella mode (reaction coordinate)
        g = M.gradient(fa, planar + 0.0).reshape(-1)
        for j, c in enumerate(classes):
            if c != "symmetric":
                self.assertAlmostEqual(float(d.vectors[j] @ g), 0.0, delta=1e-4)

    def test_slope_is_zero_along_non_symmetric_directions_away_from_the_minimum(self):
        # a symmetric but not stationary geometry: stretched water. Only symmetric directions carry a slope.
        ff = M.water(True); x = M.water_geometry(r=1.02, theta=100.0)
        d = directions(ff, M.minimize(ff, M.water_geometry()))
        classes, _ = SYM.classify_directions(d.vectors, ["O", "H", "H"], x)
        g = M.gradient(ff, x).reshape(-1)
        proj = d.vectors @ g
        self.assertGreater(np.abs(proj[[c == "symmetric" for c in classes]]).min(), 5.0)
        self.assertLess(np.abs(proj[[c == "odd" for c in classes]]).max(), 1e-3)


class TestSearchWithSymmetry(unittest.TestCase):
    def run_case(self, proto, seed=2):
        ft, fs = M.water(True), M.water(False)
        xs, xt = M.minimize(fs, M.water_geometry()), M.minimize(ft, M.water_geometry())
        d = directions(fs, xs)
        with tempfile.TemporaryDirectory() as tmp:
            eng = MockEngine(ft, fs, sigma_block=5.0, blocks_per_tick=10, seed=seed, transient=20.0, equil_blocks=0.6)
            s = DenseScanSearch("min", eng, AnalyticSurrogate(fs), ["O", "H", "H"], xs, d.vectors, d.curvature, protocol=proto, state_dir=tmp)
            status = s.run()
            return status, s.state["iters"]["1"], s.search_blocks(), ft(s.final_geometry()) - ft(xt)

    def test_fewer_points_same_answer(self):
        st_on, it_on, blocks_on, e_on = self.run_case(P2.ProtocolV2(symmetry=True))
        st_off, it_off, blocks_off, e_off = self.run_case(P2.ProtocolV2(symmetry=False))
        self.assertEqual(st_on, CONVERGED); self.assertEqual(st_off, CONVERGED)
        n_on = sum(len(v) for v in it_on["points"].values()); n_off = sum(len(v) for v in it_off["points"].values())
        self.assertEqual(n_off, 6); self.assertEqual(n_on, 5)               # the odd direction keeps one point
        self.assertEqual(sorted(it_on["classes"]), ["odd", "symmetric", "symmetric"])
        self.assertLess(e_on, 0.2); self.assertLess(e_off, 0.2)
        self.assertLess(blocks_on, blocks_off)

    def test_unstable_symmetric_geometry_is_left(self):
        """Start at the planar saddle of ammonia but search for a MINIMUM with a surrogate that calls the planar
        form stable. The measured curvature along the umbrella direction is negative: the search must leave the plane."""
        ft = M.ammonia(True)
        planar = M.refine_stationary(ft, M.ammonia_geometry(theta=120.0, r=1.0))
        Ht = M.hessian(ft, planar); sub = S.mode_subspace(Ht, planar)
        K = S.project_hessian(Ht, sub.basis)
        w, V = np.linalg.eigh(K)
        fake = (V * np.abs(w)) @ V.T                              # the surrogate curvature with the wrong sign made positive
        d = S.search_directions(fake, sub)
        self.assertEqual(d.n_negative, 0)
        fs = lambda x: 0.5 * float((sub.basis @ (np.asarray(x) - planar).reshape(-1)) @ fake @ (sub.basis @ (np.asarray(x) - planar).reshape(-1)))
        with tempfile.TemporaryDirectory() as tmp:
            eng = MockEngine(ft, fs, sigma_block=2.0, blocks_per_tick=10, seed=1, transient=20.0, equil_blocks=0.6)
            s = DenseScanSearch("min", eng, AnalyticSurrogate(fs), ["N", "H", "H", "H"], planar, d.vectors, d.curvature,
                                protocol=P2.ProtocolV2(max_iter=14), state_dir=tmp)
            s.run()
            xf = s.final_geometry()
            e_min = ft(M.minimize(ft, M.ammonia_geometry()))
            self.assertGreater(ft(planar) - ft(xf), 3.0)          # it left the plane (the planar form is 5.4 kcal/mol above the pyramid)
            self.assertLess(ft(xf) - e_min, 2.5)
            self.assertFalse(s.state["iters"]["1"]["decided"]["0"]["res"]["curv_ok"])     # found from the measured curvature


if __name__ == "__main__":
    unittest.main()
