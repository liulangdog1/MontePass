"""The moved surrogate ("right shape, wrong place"): the wrappers of ``shifted.py``."""
import os
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

from montepass import model as M                                  # noqa: E402
from montepass import subspace as S                               # noqa: E402
from montepass.engine.base import DONE, FAILED, RUNNING, PointSpec        # noqa: E402
from montepass.engine.mock import MockEngine                      # noqa: E402
from montepass.search import protocol_v2 as P2                    # noqa: E402
from montepass.search.dense import DenseScanSearch                # noqa: E402
from montepass.shifted import ShiftedEngine, ShiftedSurrogate     # noqa: E402
from montepass.surrogate.analytic import AnalyticSurrogate        # noqa: E402
from montepass.units import HARTREE_TO_KCAL                       # noqa: E402


class SlowSource:
    """Surrogate energies of single geometries that arrive only after ``release()``."""

    def __init__(self, surface):
        self.surface, self.requests, self.ready = surface, {}, False

    def request_scan(self, tag, geometries):
        self.requests.setdefault(tag, [np.array(g, dtype=float) for g in geometries])

    def release(self):
        self.ready = True

    def scan(self, tag):
        if not self.ready or tag not in self.requests:
            return None
        return np.array([self.surface(x) / HARTREE_TO_KCAL for x in self.requests[tag]])


def shifted_case(shift=0.10, direction=1):
    ft = M.water(True)
    xt = M.minimize(ft, M.water_geometry())
    Ht = M.hessian(ft, xt); sub = S.mode_subspace(Ht, xt)
    d0 = S.search_directions(S.project_hessian(Ht, sub.basis), sub)
    delta = (shift * d0.vectors[direction]).reshape(-1, 3)
    return ft, xt, d0, delta


class TestShiftedEngine(unittest.TestCase):
    def test_the_surrogate_energy_of_the_true_geometry_never_reaches_the_search(self):
        ft, xt, d0, delta = shifted_case()
        x = xt + delta                                              # a point of the search
        inner = MockEngine(ft, ft, sigma_block=1.0, blocks_per_tick=10, seed=3)       # surrogate = true surface, not moved
        source = SlowSource(ft)
        eng = ShiftedEngine(inner, delta, source=source)
        eng.submit(PointSpec(key="a", symbols=list(ft.symbols), coords=x, blocks=40))
        np.testing.assert_allclose(source.requests["moved/a"][0], x - delta, atol=1e-14)      # the moved geometry
        eng.advance(); eng.advance()
        raw = inner.read("a")
        self.assertEqual(raw.state, RUNNING); self.assertGreater(raw.nblocks, 0)
        self.assertAlmostEqual(raw.e_surrogate * HARTREE_TO_KCAL, ft(x), places=6)             # true geometry
        # while the energy of the moved surrogate is missing: no blocks, no surrogate energy
        d = eng.read("a")
        self.assertEqual((d.state, d.nblocks), (RUNNING, 0)); self.assertIsNone(d.e_surrogate)
        source.release()
        # a running point shows its blocks with the energy of the moved surrogate
        d = eng.read("a")
        self.assertEqual(d.state, RUNNING); self.assertEqual(d.nblocks, raw.nblocks)
        self.assertAlmostEqual(d.e_surrogate * HARTREE_TO_KCAL, ft(x - delta), places=6)
        self.assertGreater(abs(ft(x) - ft(x - delta)), 5.0)                                      # the two differ clearly
        for _ in range(10):
            eng.advance()
        d = eng.read("a")
        self.assertEqual((d.state, d.nblocks), (DONE, 40))
        self.assertAlmostEqual(d.e_surrogate * HARTREE_TO_KCAL, ft(x - delta), places=6)

    def test_a_failed_point_never_shows_the_energy_of_the_true_geometry(self):
        """A point that failed can keep its blocks. Its surrogate energy must be that of the moved surrogate,
        or none; never the energy of the surrogate that was not moved."""
        ft, xt, d0, delta = shifted_case()

        class Fails(MockEngine):
            def read(self, key):
                data = super().read(key)
                if key == "bad":
                    data.state = FAILED
                return data

        inner = Fails(ft, ft, sigma_block=5.0, blocks_per_tick=10, seed=1)
        source = SlowSource(ft)
        eng = ShiftedEngine(inner, delta, source=source)
        x = xt + delta + 0.03
        eng.submit(PointSpec(key="bad", symbols=list(ft.symbols), coords=x, blocks=40))
        inner.advance(2)
        unmoved = inner.read("bad").e_surrogate
        first = eng.read("bad")
        self.assertEqual(first.state, FAILED); self.assertIsNone(first.e_surrogate)       # the moved energy is not there yet
        self.assertEqual(first.nblocks, inner.read("bad").nblocks)
        source.release()
        later = eng.read("bad")
        self.assertEqual(later.state, FAILED)
        self.assertAlmostEqual(later.e_surrogate, ft(x - delta) / HARTREE_TO_KCAL, places=12)
        self.assertGreater(abs(later.e_surrogate - unmoved) * HARTREE_TO_KCAL, 1.0)

    def test_a_finished_point_waits_for_the_moved_energy(self):
        ft, xt, d0, delta = shifted_case()
        inner = MockEngine(ft, ft, sigma_block=1.0, blocks_per_tick=50, seed=4)
        source = SlowSource(ft)
        eng = ShiftedEngine(inner, delta, source=source)
        eng.submit(PointSpec(key="b", symbols=list(ft.symbols), coords=xt + delta, blocks=40))
        for _ in range(5):
            eng.advance()
        self.assertEqual(inner.read("b").state, DONE)
        self.assertEqual(eng.read("b").state, RUNNING)               # not final for the search yet
        source.release()
        self.assertEqual(eng.read("b").state, DONE)

    def test_search_from_a_shifted_start_with_both_ways_of_moving_the_surrogate(self):
        """The search must find the true minimum from a start 0.10 A away, whether the engine computes the
        moved surrogate energy itself (mock engine) or a separate source does (real engines)."""
        ft, xt, d0, delta = shifted_case()
        fs = ft.shifted(delta)                                       # the moved surrogate as one surface
        xs = xt + delta
        Hs = M.hessian(fs, xs); sub = S.mode_subspace(Hs, xs)
        d = S.search_directions(S.project_hessian(Hs, sub.basis), sub)
        results = []
        for way in ("native", "source"):
            with tempfile.TemporaryDirectory() as tmp:
                if way == "native":
                    eng = MockEngine(ft, fs, sigma_block=5.0, blocks_per_tick=10, seed=5, transient=20.0, equil_blocks=0.6)
                    sur = AnalyticSurrogate(fs)
                else:
                    inner = MockEngine(ft, ft, sigma_block=5.0, blocks_per_tick=10, seed=5, transient=20.0, equil_blocks=0.6)
                    eng = ShiftedEngine(inner, delta, source=AnalyticSurrogate(ft))
                    sur = ShiftedSurrogate(AnalyticSurrogate(ft), delta)
                s = DenseScanSearch("t", eng, sur, list(ft.symbols), xs, d.vectors, d.curvature, protocol=P2.V2, state_dir=tmp)
                status = s.run()
                first = s.state["iters"]["1"]["composed"]
                results.append((status, ft(s.final_geometry()) - ft(xt), first["G"], first["step_norm"]))
        for status, e_final, gain_first, step_first in results:
            self.assertEqual(status, "CONVERGED")
            self.assertLess(e_final, 0.2)
            self.assertGreater(gain_first, 5.0)                      # the first scan sees that the start is wrong
            self.assertGreater(step_first, 0.04)
        self.assertGreater(ft(xs) - ft(xt), 8.0)                     # the start is 8 kcal/mol or more above the minimum


if __name__ == "__main__":
    unittest.main()
