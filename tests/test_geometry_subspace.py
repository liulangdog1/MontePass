"""Geometry utilities and subspace construction."""
import os
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

from montepass import geometry as G, model as M  # noqa: E402
from montepass import subspace as S  # noqa: E402
from montepass.units import HARTREE_TO_KCAL  # noqa: E402

WATER = (["O", "H", "H"], np.array([[0.0, 0.0, 0.1173], [0.0, 0.7572, -0.4692], [0.0, -0.7572, -0.4692]]))
NH3_PATH = os.path.join(HERE, "data", "orca_610_nh3_neb_MEP_trj.xyz")      # the ammonia inversion path written by ORCA


def rotation(axis, angle):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


class TestXyz(unittest.TestCase):
    def test_roundtrip_and_empty_comment(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "w.xyz")
            G.write_xyz(p, *WATER)                       # empty comment line
            with open(p) as fh:
                lines = fh.read().split("\n")
            self.assertEqual(lines[0], "3"); self.assertEqual(lines[1], "")
            s, x = G.read_xyz(p)
            self.assertEqual(s, WATER[0]); np.testing.assert_allclose(x, WATER[1], atol=1e-10)

    def test_truncated_file_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "bad.xyz")
            with open(p, "w") as fh:
                fh.write("3\n\nO 0 0 0\nH 0 0 1\n")       # announces 3 atoms, holds 2
            with self.assertRaises(ValueError):
                G.read_xyz(p)

    def test_blank_comment_not_skipped(self):
        """A parser that drops blank lines would read the first atom as the comment."""
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "w.xyz")
            with open(p, "w") as fh:
                fh.write("3\n\nO 0.0 0.0 0.1\nH 0.0 0.7 -0.4\nH 0.0 -0.7 -0.4\n")
            s, x = G.read_xyz(p)
            self.assertEqual(s, ["O", "H", "H"]); self.assertAlmostEqual(x[0, 2], 0.1)


class TestAlignment(unittest.TestCase):
    def test_kabsch_recovers_rotation(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(7, 3))
        Y = X @ rotation([1, 2, 3], 0.7) + np.array([1.0, -2.0, 0.5])
        moved, rmsd, refl = G.kabsch(Y, X)
        self.assertLess(rmsd, 1e-10); self.assertFalse(refl)
        np.testing.assert_allclose(moved, X, atol=1e-9)

    def test_mirror_needs_reflection(self):
        rng = np.random.default_rng(2)
        X = rng.normal(size=(6, 3))
        Y = X * np.array([1.0, 1.0, -1.0])
        _m, rmsd_proper, _r = G.kabsch(Y, X)
        _m, rmsd_mirror, refl = G.kabsch(Y, X, allow_reflection=True)
        self.assertGreater(rmsd_proper, 0.1); self.assertLess(rmsd_mirror, 1e-10); self.assertTrue(refl)

    def test_rigid_body_basis_counts(self):
        self.assertEqual(len(G.rigid_body_basis(WATER[1])), 6)
        linear = np.array([[0, 0, -1.06], [0, 0, 0.0], [0, 0, 1.16]])       # H-C-N on a line
        self.assertEqual(len(G.rigid_body_basis(linear)), 5)
        self.assertEqual(G.n_internal(WATER[1]), 3); self.assertEqual(G.n_internal(linear), 4)

    def test_remove_rigid_motion(self):
        X = WATER[1]
        trans = np.tile([0.3, -0.1, 0.2], 3)
        rot = np.cross([0.0, 0.0, 1.0], X - X.mean(0)).reshape(-1)
        cleaned = G.remove_rigid_motion(np.array([trans, rot]), X)
        self.assertLess(np.abs(cleaned).max(), 1e-12)

    def test_bonds_and_angles(self):
        bonds = G.bond_list(*WATER)
        self.assertEqual(bonds, [(0, 1), (0, 2)])
        self.assertEqual(G.angle_list(bonds), [(1, 0, 2)])
        self.assertAlmostEqual(G.angle(WATER[1], 1, 0, 2), 104.5, delta=0.5)
        self.assertEqual(G.valence_electrons(WATER[0]), 8)


class TestSubspace(unittest.TestCase):
    def test_mode_subspace_and_directions(self):
        # model Hessian: two internal springs of water plus an angle term, built from internal vectors
        X = WATER[1]
        b1 = np.zeros(9); b1[0:3] = (X[0] - X[1]) / np.linalg.norm(X[0] - X[1]); b1[3:6] = -b1[0:3]
        b2 = np.zeros(9); b2[0:3] = (X[0] - X[2]) / np.linalg.norm(X[0] - X[2]); b2[6:9] = -b2[0:3]
        H = 1200.0 * np.outer(b1, b1) + 1200.0 * np.outer(b2, b2)
        bend = G.remove_rigid_motion(np.array([[0, 0, 0, 0, 0, 1.0, 0, 0, 1.0]]), X)[0]
        bend -= (bend @ b1) * b1 / (b1 @ b1); bend -= (bend @ b2) * b2 / (b2 @ b2)
        H += 160.0 * np.outer(bend, bend) / (bend @ bend)
        sub = S.mode_subspace(H, X)
        self.assertEqual(sub.dim, 3)
        K = S.project_hessian(H, sub.basis)
        d = S.search_directions(K, sub)
        self.assertTrue(np.all(np.diff(d.curvature) >= 0))            # signed ascending order
        self.assertEqual(d.n_negative, 0)
        np.testing.assert_allclose(np.linalg.norm(d.vectors, axis=1), 1.0, atol=1e-12)
        Kd = S.hessian_in_directions(H, d)
        self.assertLess(S.max_coupling(Kd), 1e-10)
        np.testing.assert_allclose(np.diag(Kd), d.curvature, rtol=1e-10)

    def test_saddle_direction_is_first(self):
        sub = S.Subspace(basis=np.eye(3), reference=np.zeros((1, 3)), kind="span")
        K = np.diag([500.0, -80.0, 30.0])
        d = S.search_directions(K, sub)
        self.assertEqual(d.n_negative, 1); self.assertAlmostEqual(d.curvature[0], -80.0)
        np.testing.assert_allclose(np.abs(d.vectors[0]), [0, 1, 0], atol=1e-12)

    def test_excluded_displacement_leaves_the_directions_across_it(self):
        rng = np.random.default_rng(7)
        basis = np.linalg.qr(rng.normal(size=(12, 4)))[0].T                  # 4 orthonormal vectors in 12 dimensions
        sub = S.Subspace(basis=basis, reference=np.zeros((4, 3)), kind="span")
        A = rng.normal(size=(4, 4)); K = A @ A.T + 50.0 * np.eye(4)
        t = 0.7 * basis[0] + 0.5 * basis[2] + 0.3 * rng.normal(size=12)      # partly outside the subspace
        d = S.search_directions(K, sub, exclude=t)
        self.assertEqual(d.dim, 3); self.assertEqual(d.coefficients.shape, (3, 4))
        np.testing.assert_allclose(d.vectors @ t, 0.0, atol=1e-12)                    # every direction is across t
        np.testing.assert_allclose(d.vectors @ d.vectors.T, np.eye(3), atol=1e-12)    # orthonormal
        np.testing.assert_allclose(d.coefficients @ sub.basis, d.vectors, atol=1e-12)
        Kd = d.coefficients @ K @ d.coefficients.T                           # the Hessian along the directions
        np.testing.assert_allclose(np.diag(Kd), d.curvature, rtol=1e-10)
        self.assertLess(np.abs(Kd - np.diag(np.diag(Kd))).max(), 1e-9)       # they do not couple
        self.assertTrue(np.all(np.diff(d.curvature) >= 0))
        # the curvatures of a restricted search lie between those of the full subspace (interlacing)
        full = S.search_directions(K, sub).curvature
        self.assertTrue(np.all(d.curvature >= full[:-1] - 1e-9) and np.all(d.curvature <= full[1:] + 1e-9))
        # the result does not depend on the length or the sign of the excluded displacement
        again = S.search_directions(K, sub, exclude=-3.0 * t)
        np.testing.assert_allclose(again.vectors, d.vectors, atol=1e-10)
        np.testing.assert_allclose(again.curvature, d.curvature, rtol=1e-10)

    def test_a_saddle_without_its_reaction_coordinate_is_a_minimum(self):
        sub = S.Subspace(basis=np.eye(3), reference=np.zeros((1, 3)), kind="span")
        K = np.diag([500.0, -80.0, 30.0])
        d = S.search_directions(K, sub, exclude=np.array([0.0, 1.0, 0.0]))
        self.assertEqual(d.dim, 2); self.assertEqual(d.n_negative, 0)
        np.testing.assert_allclose(d.curvature, [30.0, 500.0])
        np.testing.assert_allclose(np.abs(d.vectors), [[0, 0, 1], [1, 0, 0]], atol=1e-12)

    def test_exclusion_that_is_not_possible_is_an_error(self):
        sub = S.Subspace(basis=np.eye(3)[:2], reference=np.zeros((1, 3)), kind="span")
        with self.assertRaises(ValueError) as cm:                            # no part inside the subspace
            S.search_directions(np.eye(2), sub, exclude=np.array([0.0, 0.0, 1.0]))
        self.assertIn("no part inside the subspace", str(cm.exception))
        one = S.Subspace(basis=np.eye(3)[:1], reference=np.zeros((1, 3)), kind="span")
        with self.assertRaises(ValueError) as cm:                            # no direction would be left
            S.search_directions(np.eye(1), one, exclude=np.array([1.0, 0.0, 0.0]))
        self.assertIn("none is left", str(cm.exception))

    def test_without_exclusion_the_directions_are_the_eigenvectors(self):
        rng = np.random.default_rng(11)
        basis = np.linalg.qr(rng.normal(size=(9, 3)))[0].T
        sub = S.Subspace(basis=basis, reference=np.zeros((3, 3)), kind="span")
        A = rng.normal(size=(3, 3)); K = A + A.T
        d = S.search_directions(K, sub)
        w, V = np.linalg.eigh(K)
        np.testing.assert_allclose(d.curvature, w, rtol=1e-12)
        self.assertEqual(d.coefficients.shape, (3, 3))
        for j in range(3):                                                   # equal up to the sign
            self.assertAlmostEqual(abs(float(d.coefficients[j] @ V[:, j])), 1.0, places=10)


class TestPathTargets(unittest.TestCase):
    def test_energy_in_a_comment_line(self):
        self.assertEqual(G.comment_energy("Coordinates from ORCA-job neb_MEP_ALL E -56.473099016534"), -56.473099016534)
        self.assertEqual(G.comment_energy("image 3 E -56.5411304047"), -56.5411304047)
        self.assertEqual(G.comment_energy("E 1.5e-3 hartree"), 1.5e-3)
        for text in ("", "start geometry of target 6", "ENERGY -1.0", "point 6/it2/centre", "E = -1.0", "E -1.0a"):
            self.assertIsNone(G.comment_energy(text), text)

    def test_a_path_without_a_saddle_between_its_ends(self):
        for energies in ([0.0, 1.0, 2.0], [3.0, 1.0, 2.0], [0.0, None, 1.0], [0.0, 1.0]):
            with self.assertRaises(ValueError):
                S.highest_image(energies)

    def test_barrier_mep_and_lists(self):
        self.assertEqual(S.path_targets("barrier", 7, 3), {0: "minimum", 3: "saddle"})
        self.assertEqual(list(S.path_targets("mep", 7, 3).items()),
                         [(0, "minimum"), (1, "image"), (2, "image"), (3, "saddle"), (4, "image"), (5, "image"), (6, "minimum")])
        self.assertEqual(list(S.path_targets([6, 3, 4], 7, 3).items()), [(6, "minimum"), (3, "saddle"), (4, "image")])
        # a path without a saddle: every image between the ends is a path image
        self.assertEqual(S.path_targets("mep", 4, None), {0: "minimum", 1: "image", 2: "image", 3: "minimum"})
        for args in (("barrier", 7, None), ("mep", 7, 0), ("mep", 7, 6), ([7], 7, 3), ([2, 2], 7, 3), ("mep", 1, None)):
            with self.assertRaises(ValueError):
                S.path_targets(*args)


class TestAmmoniaPath(unittest.TestCase):
    """The path of the ammonia inversion in ``tests/data``: 7 images from ORCA with the planar saddle in the middle.
    The two ends are mirror images: the N atom is above the plane of the H atoms at one end and below it at the
    other end. The threefold axis of the molecule is the z axis of the file."""

    def setUp(self):
        frames = G.read_xyz_frames(NH3_PATH)
        self.symbol_lists = [f[0] for f in frames]
        self.frames = [np.asarray(f[1]) for f in frames]
        self.energies = [G.comment_energy(f[2]) for f in frames]

    def test_frames_and_energies(self):
        self.assertEqual(len(self.frames), 7)
        self.assertTrue(all(s == ["N", "H", "H", "H"] for s in self.symbol_lists))
        self.assertTrue(all(e is not None for e in self.energies))
        self.assertEqual(S.highest_image(self.energies), 3)
        self.assertAlmostEqual((self.energies[3] - self.energies[0]) * HARTREE_TO_KCAL, 6.013, places=3)    # barrier along this path
        self.assertLess(abs(self.energies[6] - self.energies[0]) * HARTREE_TO_KCAL, 1e-6)                    # the path is symmetric

    def test_align_path_removes_only_rigid_motion(self):
        rng = np.random.default_rng(5)
        moves = [(rotation(rng.normal(size=3), rng.uniform(0.2, 3.0)), 2.0 * rng.normal(size=3)) for _ in self.frames]
        moved = [x @ R.T + t for x, (R, t) in zip(self.frames, moves)]         # every image turned and shifted on its own
        aligned = S.align_path(moved)
        self.assertEqual(len(aligned), 7)
        np.testing.assert_allclose(aligned[0], moved[0], atol=1e-12)          # the reference image does not move
        before = S.align_path(self.frames)
        R0, t0 = moves[0]
        for k, (a, x) in enumerate(zip(aligned, self.frames)):
            for i in range(4):                                                 # the shape of every image is kept
                for j in range(i):
                    self.assertAlmostEqual(G.distance(a, i, j), G.distance(x, i, j), places=10)
            # the image sits on the reference with the best rotation: the same result as for the path before it was
            # moved, carried along with the reference image
            np.testing.assert_allclose(a, before[k] @ R0.T + t0, atol=1e-9)
            rmsd = float(np.sqrt(np.mean(np.sum((a - aligned[0]) ** 2, axis=1))))
            self.assertAlmostEqual(rmsd, G.kabsch(x, self.frames[0])[1], places=9)
        # the last image is the mirror image of the first: a rotation cannot put it on the first, a reflection can
        self.assertGreater(G.kabsch(aligned[6], aligned[0])[1], 0.3)
        mirrored = S.align_path(moved, allow_reflection=True)
        self.assertLess(float(np.sqrt(np.mean(np.sum((mirrored[6] - mirrored[0]) ** 2, axis=1)))), 1e-5)

    def test_principal_components_of_the_inversion(self):
        sub = S.pca_subspace(self.frames, d=2)
        evr = sub.explained_variance_ratio
        self.assertAlmostEqual(float(evr.sum()), 1.0, places=12)
        self.assertTrue(np.all(np.diff(evr) <= 0.0))
        self.assertGreater(evr[0], 0.95); self.assertGreater(evr[1], 0.03); self.assertLess(float(evr[2:].max()), 1e-6)
        np.testing.assert_allclose(sub.basis @ sub.basis.T, np.eye(2), atol=1e-12)
        rigid = G.rigid_body_basis(sub.reference)
        self.assertLess(np.abs(sub.basis @ rigid.T).max(), 1e-10)             # no translation or rotation is left in them
        self.assertLess(max(abs(f) for f in sub.info["rigid_fraction"]), 1e-6)    # they were internal motions already
        # component 1 is the inversion: N moves along the threefold axis and the three H atoms move the other way
        v = sub.basis[0].reshape(4, 3)
        self.assertGreater(abs(v[0, 2]), 0.85)
        np.testing.assert_allclose(v[1:, 2], v[1, 2], atol=1e-5)
        self.assertLess(v[0, 2] * v[1, 2], 0.0)
        self.assertLess(np.abs(v[:, :2]).max(), 1e-4)
        # component 2 moves the three H atoms outward, across the axis: the H triangle grows on the way to the
        # planar saddle
        w = sub.basis[1].reshape(4, 3)
        self.assertLess(np.abs(w[0]).max(), 1e-3); self.assertLess(np.abs(w[:, 2]).max(), 1e-3)
        for h in (1, 2, 3):
            out = sub.reference[h, :2] - sub.reference[0, :2]
            self.assertGreater(float(w[h, :2] @ out) / np.linalg.norm(out), 0.55)
        # the number of components from the explained variance, and the limits
        self.assertEqual(S.pca_subspace(self.frames, variance=0.9).dim, 1)
        self.assertEqual(S.pca_subspace(self.frames, variance=0.99).dim, 2)
        with self.assertRaises(ValueError):
            S.pca_subspace(self.frames, d=7)                                   # 7 images give at most 6 components
        with self.assertRaises(ValueError):
            S.pca_subspace(self.frames[:1])

    def test_captured_fraction(self):
        aligned = S.align_path(self.frames)
        two = S.pca_subspace(self.frames, d=2, remove_rigid=False)
        for i in range(7):                                                     # every step along the path lies in
            for j in range(i):                                                 # the plane of the first two components
                self.assertGreater(two.captured_fraction(aligned[i] - aligned[j]), 1.0 - 1e-6)
        one = S.pca_subspace(self.frames, d=1, remove_rigid=False)
        self.assertGreater(one.captured_fraction(aligned[6] - aligned[0]), 1.0 - 1e-6)     # from end to end: the inversion
        half = one.captured_fraction(aligned[3] - aligned[0])                 # to the saddle the H atoms also move outward
        self.assertGreater(half, 0.8); self.assertLess(half, 0.95)
        self.assertEqual(one.captured_fraction(np.zeros((4, 3))), 1.0)
        self.assertAlmostEqual(one.captured_fraction(one.basis[0]), 1.0, places=12)
        internal = S.pca_subspace(self.frames, d=2)                            # rigid motion removed
        self.assertLess(internal.captured_fraction(np.tile([0.3, -0.2, 0.1], (4, 1))), 1e-12)   # a shift of the molecule

    def test_save_and_load(self):
        sub = S.pca_subspace(self.frames, d=2)
        span = S.Subspace(basis=np.eye(12)[:3], reference=self.frames[3], kind="span")
        with tempfile.TemporaryDirectory() as tmp:
            sub.save(os.path.join(tmp, "pca.npz")); span.save(os.path.join(tmp, "span.npz"))
            back = S.Subspace.load(os.path.join(tmp, "pca.npz"))
            back_span = S.Subspace.load(os.path.join(tmp, "span.npz"))
        self.assertEqual((back.kind, back.dim), ("pca", 2))
        np.testing.assert_array_equal(back.basis, sub.basis)
        np.testing.assert_array_equal(back.reference, sub.reference)
        np.testing.assert_array_equal(back.explained_variance_ratio, sub.explained_variance_ratio)
        np.testing.assert_array_equal(back.singular_values, sub.singular_values)
        # a subspace that was not made by a principal component analysis has no variance ratios, also after loading
        self.assertEqual((back_span.kind, back_span.dim), ("span", 3))
        self.assertIsNone(back_span.explained_variance_ratio); self.assertIsNone(back_span.singular_values)
        np.testing.assert_array_equal(back_span.reference, self.frames[3])


class TestPathTangent(unittest.TestCase):
    def frames(self):
        # a bent water molecule whose angle opens along the path, each image moved and turned as a whole
        out = []
        for k, angle in enumerate((100.0, 102.0, 104.0, 106.0, 108.0)):
            a = np.radians(angle) / 2.0
            x = np.array([[0.0, 0.0, 0.0], [0.96 * np.sin(a), 0.0, 0.96 * np.cos(a)], [-0.96 * np.sin(a), 0.0, 0.96 * np.cos(a)]])
            c, s = np.cos(0.3 * k), np.sin(0.3 * k)
            R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
            out.append(x @ R.T + np.array([0.1 * k, -0.2 * k, 0.05 * k]))
        return out

    def test_tangent_is_the_internal_motion_between_the_neighbours(self):
        frames = self.frames()
        t = S.path_tangent(frames, 2)
        self.assertAlmostEqual(float(np.linalg.norm(t)), 1.0, places=12)
        # the rigid motion between the images is no part of the tangent
        inner = G.remove_rigid_motion(t[None, :], frames[2])[0]
        self.assertGreater(float(inner @ inner), 1.0 - 1e-4)
        # the tangent opens the angle: both H atoms move apart along their connecting line
        x = frames[2]
        hh = (x[1] - x[2]) / np.linalg.norm(x[1] - x[2])
        self.assertGreater(float((t.reshape(3, 3)[1] - t.reshape(3, 3)[2]) @ hh), 0.5)

    def test_tangent_does_not_depend_on_how_the_neighbours_are_placed(self):
        frames = self.frames()
        t = S.path_tangent(frames, 2)
        c, s = np.cos(1.1), np.sin(1.1)
        R = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
        moved = list(frames)
        moved[1] = frames[1] @ R.T + 3.0
        moved[3] = frames[3] @ R + np.array([-1.0, 2.0, 0.5])
        np.testing.assert_allclose(S.path_tangent(moved, 2), t, atol=1e-10)

    def test_the_ends_of_a_path_have_no_tangent(self):
        frames = self.frames()
        for index in (0, 4, 7, -1):
            with self.assertRaises(ValueError):
                S.path_tangent(frames, index)
        with self.assertRaises(ValueError):                                  # two equal neighbours
            S.path_tangent([frames[0], frames[1], frames[0]], 1)


class TestRandomDisplacement(unittest.TestCase):
    def test_size_seed_and_no_rigid_motion(self):
        x = M.ammonia_geometry()
        d = G.random_displacement(x, 0.15, seed=7)
        self.assertEqual(d.shape, x.shape)
        self.assertAlmostEqual(float(np.sqrt(np.mean(np.sum(d * d, axis=1)))), 0.15, places=12)
        np.testing.assert_allclose(d, G.random_displacement(x, 0.15, seed=7))           # the same seed, the same vector
        self.assertGreater(float(np.abs(d - G.random_displacement(x, 0.15, seed=8)).max()), 0.01)
        np.testing.assert_allclose(d.sum(axis=0), 0.0, atol=1e-12)                      # no translation of the molecule
        self.assertLess(float(np.abs(G.rigid_body_basis(x) @ d.reshape(-1)).max()), 1e-12)   # and no rotation
        # the displaced molecule can be put back on the original only up to the displacement itself
        self.assertAlmostEqual(G.kabsch(x + d, x)[1], 0.15, delta=0.002)
        np.testing.assert_allclose(G.random_displacement(x, 0.0, seed=1), 0.0)


class TestLinearMolecules(unittest.TestCase):
    def test_an_optimized_linear_molecule_is_linear(self):
        # geometry of HCN as a program wrote it after an optimization: 1e-7 angstrom off the axis
        x = np.array([[0.00000006312748, -0.00000011054340, -1.06348218042268],
                      [-0.00000012625401, 0.00000022108436, 0.00200158195359],
                      [0.00000006312653, -0.00000011054095, 1.14848059846910]])
        self.assertEqual(G.n_internal(x), 4)                         # 3 n - 5
        self.assertEqual(len(G.rigid_body_basis(x)), 5)
        bent = x.copy(); bent[0, 0] += 0.05                           # a bend of 2.7 degrees is not linear
        self.assertEqual(G.n_internal(bent), 3)
        # the two bends are not rigid motions: projecting the rigid motions out must keep them
        bend = np.array([[0.0, 1.0, 0.0], [0.0, -2.0, 0.0], [0.0, 1.0, 0.0]]).reshape(-1)
        bend /= np.linalg.norm(bend)
        left = G.remove_rigid_motion(bend[None, :], x)[0]
        self.assertGreater(np.linalg.norm(left), 0.5)


class TestFrames(unittest.TestCase):
    def test_hessian_follows_a_rotated_and_shifted_geometry(self):
        from montepass import model as M
        surface = M.ammonia(True)
        x = M.minimize(surface, M.ammonia_geometry())
        H = M.hessian(surface, x)
        a, b = 0.7, -0.4
        rz = np.array([[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]])
        rx = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(b), -np.sin(b)], [0.0, np.sin(b), np.cos(b)]])
        y = x @ (rz @ rx) + np.array([0.3, -1.1, 2.0])                 # the same molecule in another frame
        rot, cx, cy, rmsd = G.superposition(x, y)
        self.assertLess(rmsd, 1e-10)
        np.testing.assert_allclose((x - cx) @ rot + cy, y, atol=1e-10)
        np.testing.assert_allclose(rot, rz @ rx, atol=1e-10)
        # the model Hessians come from finite differences: they agree to 0.04 of about 1300 kcal/mol/A^2
        np.testing.assert_allclose(G.rotate_hessian(H, rot), M.hessian(surface, y), atol=0.1, rtol=0.0)
        self.assertGreater(np.abs(H - M.hessian(surface, y)).max(), 100.0)      # without the rotation they differ
        with self.assertRaises(ValueError):
            G.rotate_hessian(np.zeros((4, 4)), rot)


if __name__ == "__main__":
    unittest.main()
