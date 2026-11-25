"""The whole workflow on model molecules (no cluster program): input file -> stages -> search results."""
import json
import os
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

from montepass import geometry as G, model as M, subspace as S  # noqa: E402
from montepass.config import Config, ConfigError, write_template  # noqa: E402
from montepass.project import DONE, Project  # noqa: E402
from montepass.workflow import ModelBackend, Workflow  # noqa: E402

WATER = """[system]
name = model_water
[geometry]
mode = single
[model]
molecule = water
sigma_block = 2.0
seed = 3
[search]
method = %s
eps_total = 0.05
"""

AMMONIA = """[system]
name = model_ammonia
[geometry]
mode = path
targets = reactant, saddle
[surrogate]
neb_images = 8
[model]
molecule = ammonia
sigma_block = 1.0
seed = 5
[search]
method = dense-v2
eps_total = 0.05
"""

# a path image between the reactant (image 0) and the saddle (image 4) of the model path of 9 images
AMMONIA_IMAGE = """[system]
name = model_ammonia_image
[geometry]
mode = images
targets = %s
kinds = %s
[surrogate]
neb_images = 8
[subspace]
dim = 2
[model]
molecule = ammonia
sigma_block = 1.0
seed = 5
[search]
method = dense-v2
eps_total = 0.05
"""

# the targets are taken from the path: "barrier" or "mep" (mode path or images), or a list with kinds = auto
AMMONIA_CHOICE = """[system]
name = model_ammonia_choice
[geometry]
mode = %s
targets = %s
%s
[surrogate]
neb_images = 4
[subspace]
dim = 2
[model]
molecule = ammonia
sigma_block = 1.0
seed = 5
[search]
method = dense-v2
eps_total = 0.05
"""


def run(folder, text, max_passes=4000):
    with open(os.path.join(folder, "input.ini"), "w") as fh:
        fh.write(text)
    wf = Workflow(folder)
    for _ in range(max_passes):
        if wf.advance() != "RUNNING":
            break
    return wf


class TestConfig(unittest.TestCase):
    def test_template_is_a_valid_input_once_the_files_exist(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_template(os.path.join(tmp, "input.ini"))
            with self.assertRaises(ConfigError) as cm:           # the geometry files of the template do not exist yet
                Config.load(os.path.join(tmp, "input.ini"))
            self.assertIn("file not found", str(cm.exception))
            G.write_xyz(os.path.join(tmp, "reactant.xyz"), ["O", "H", "H"], M.water_geometry())
            G.write_xyz(os.path.join(tmp, "product.xyz"), ["O", "H", "H"], M.water_geometry())
            cfg = Config.load(os.path.join(tmp, "input.ini"))
            self.assertEqual(cfg.mode, "path"); self.assertEqual(cfg.targets, ["reactant", "saddle"])
            self.assertEqual(cfg.qmc["dmc"]["timestep"], 0.005); self.assertEqual(cfg.search_method, "dense-v2")

    def test_problems_are_reported_with_section_and_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "input.ini"), "w") as fh:
                fh.write("[geometry]\nmode = single\n[model]\nmolecule = water\n[search]\nmethod = fastest\neps_total = -1\n[qmc]\nwalkrs = 5\n")
            with self.assertRaises(ConfigError) as cm:
                Config.load(os.path.join(tmp, "input.ini"))
            msg = str(cm.exception)
            for needle in ("[search] method", "[search] eps_total", "[qmc] unknown key 'walkrs'"):
                self.assertIn(needle, msg)


class TestModelWorkflow(unittest.TestCase):
    def test_single_geometry_water(self):
        with tempfile.TemporaryDirectory() as tmp:
            wf = run(tmp, WATER % "dense-v2")
            p = Project(tmp)
            self.assertEqual(p.state["status"], DONE, p.state["stages"])
            self.assertEqual(p.state["subspace"], {"kind": "modes", "dim": 3})
            self.assertEqual(p.target("geometry")["status"], "CONVERGED")
            drv = wf.driver("geometry")
            b = wf.backend
            xt = M.minimize(b.true, M.water_geometry())
            start = G.read_xyz(os.path.join(p.target_dir("geometry"), "start.xyz"))[1]
            self.assertLess(b.true_energy(drv.final_geometry()) - b.true_energy(xt), 0.06)
            self.assertGreater(b.true_energy(start) - b.true_energy(xt), 0.15)       # the search improved the geometry
            de, sd = drv.state["final"]["energy_change"]
            self.assertLess(de, 0.0)
            self.assertTrue(os.path.exists(os.path.join(tmp, "work", "driver.log")))

    def test_second_process_continues(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "input.ini"), "w") as fh:
                fh.write(WATER % "dense-v2")
            wf = Workflow(tmp)
            for _ in range(6):
                wf.advance()
            self.assertEqual(Project(tmp).state["status"], "RUNNING")
            wf2 = Workflow(tmp)                                  # a new process: only the files are shared
            for _ in range(4000):
                if wf2.advance() != "RUNNING":
                    break
            self.assertEqual(Project(tmp).state["status"], DONE)

    def test_registered_protocol_runs_too(self):
        with tempfile.TemporaryDirectory() as tmp:
            run(tmp, WATER % "dense-v1")
            p = Project(tmp)
            self.assertEqual(p.state["stages"]["search"]["status"], DONE, p.state["stages"])
            self.assertIn(p.target("geometry")["status"], ("CONVERGED", "NOT_CONVERGED"))

    def test_path_with_minimum_and_saddle(self):
        with tempfile.TemporaryDirectory() as tmp:
            wf = run(tmp, AMMONIA)
            p = Project(tmp)
            self.assertEqual(p.state["stages"]["search"]["status"], DONE, p.state["stages"])
            self.assertEqual(p.state["subspace"]["kind"], "pca")
            self.assertEqual(len(G.read_xyz_frames(p.path_file)), 9)
            self.assertEqual(p.target("saddle")["kind"], "saddle")
            b = wf.backend
            barrier_true = b.true_energy(M.refine_stationary(b.true, M.ammonia_geometry(theta=120.0, r=1.0))) - \
                b.true_energy(M.minimize(b.true, M.ammonia_geometry()))
            e_min = wf.driver("reactant").final_energy(); e_sad = wf.driver("saddle").final_energy()
            barrier = (e_sad[0] - e_min[0]) * 627.509474
            err = np.hypot(e_sad[1], e_min[1]) * 627.509474
            # the optimized barrier is the true one within the tolerance of both searches and the noise
            self.assertLess(abs(barrier - barrier_true), 4 * err + 0.15, "barrier %.3f +/- %.3f, true %.3f" % (barrier, err, barrier_true))
            with np.load(os.path.join(p.target_dir("saddle"), "directions.npz")) as z:
                self.assertEqual(int(np.sum(z["curvature"] < 0)), 1)

    def test_wrong_start_with_a_moved_surrogate(self):
        """The start is 0.10 A off along the stiffest direction and the surrogate is moved with it (right shape,
        wrong place). Only the DMC energies know where the minimum is; the search must still find it."""
        text = WATER % "dense-v2" + "[experiment]\nshift = 0.10\ndirection = 2\n"
        with tempfile.TemporaryDirectory() as tmp:
            wf = run(tmp, text.replace("eps_total = 0.05", "eps_total = 0.2"))
            p = Project(tmp)
            self.assertEqual(p.state["stages"]["search"]["status"], DONE, p.state["stages"])
            b = wf.backend
            start = G.read_xyz(os.path.join(p.target_dir("geometry"), "start.xyz"))[1]
            unshifted = G.read_xyz(os.path.join(p.target_dir("geometry"), "start_unshifted.xyz"))[1]
            self.assertAlmostEqual(float(np.linalg.norm(start - unshifted)), 0.10, places=6)
            xt = M.minimize(b.true, M.water_geometry())
            e_start = b.true_energy(start) - b.true_energy(xt)
            e_final = b.true_energy(wf.driver("geometry").final_geometry()) - b.true_energy(xt)
            self.assertGreater(e_start, 8.0)                    # the start is far up the wall
            self.assertLess(e_final, 0.5)                       # the search came back to the minimum
            # the surrogate itself calls the moved start a stationary point: its slope there is zero
            est = wf.driver("geometry").state["iters"]["1"]["decided"]["2"]["res"]
            self.assertLess(abs(est["g_sur"]), 1.0); self.assertGreater(abs(est["g"]), 50.0)

    def test_random_displacement_of_all_atoms(self):
        """Every atom is moved at random (rms 0.10 A per atom) and the surrogate is moved with the start. Both the
        dense-scan search and the classic line search must come back to the minimum."""
        for method in ("dense-v2", "classic"):
            text = WATER % method + "[experiment]\nrandom_rms = 0.10\nrandom_seed = 3\n"
            with tempfile.TemporaryDirectory() as tmp:
                wf = run(tmp, text.replace("eps_total = 0.05", "eps_total = 0.2"))
                p = Project(tmp)
                self.assertEqual(p.state["stages"]["search"]["status"], DONE, (method, p.state["stages"]))
                b = wf.backend
                start = G.read_xyz(os.path.join(p.target_dir("geometry"), "start.xyz"))[1]
                unshifted = G.read_xyz(os.path.join(p.target_dir("geometry"), "start_unshifted.xyz"))[1]
                moved = start - unshifted
                self.assertAlmostEqual(float(np.sqrt(np.mean(np.sum(moved ** 2, axis=1)))), 0.10, places=6)   # rms per atom
                self.assertAlmostEqual(G.kabsch(start, unshifted)[1], float(np.sqrt(np.mean(np.sum(moved ** 2, axis=1)))), places=4)
                info = p.target("geometry")["experiment"]
                self.assertEqual((info["random_rms"], info["random_seed"]), (0.10, 3))
                self.assertGreater(info["fraction_in_search_directions"], 0.999)     # water: the 3 directions span everything
                xt = M.minimize(b.true, M.water_geometry())
                e_start = b.true_energy(start) - b.true_energy(xt)
                e_final = b.true_energy(wf.driver("geometry").final_geometry()) - b.true_energy(xt)
                self.assertGreater(e_start, 5.0, method)
                self.assertLess(e_final, 0.05 * e_start, method)                    # at least 95 % of the error is removed
                self.assertLess(e_final, 0.5, method)

    def test_retry_clears_needs_attention(self):
        """A QMC point fails: the project needs attention. After the repair, ``retry`` lets the workflow continue."""
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "input.ini"), "w") as fh:
                fh.write(WATER % "dense-v2")
            wf = Workflow(tmp)
            self.assertEqual(wf.retry(), [])                         # nothing has failed: nothing to do
            engine_class = type(wf.backend.engine(wf))
            original = engine_class.read
            broken = {"on": True}

            def read(self, key):
                data = original(self, key)
                if broken["on"] and "d1_p" in key:
                    data.state = "FAILED"; data.energy = data.energy[:0]; data.weight = data.weight[:0]
                return data

            engine_class.read = read
            try:
                status = "RUNNING"
                for _ in range(400):
                    status = wf.advance()
                    if status != "RUNNING":
                        break
                self.assertEqual(status, "NEEDS_ATTENTION")
                self.assertEqual(wf.project.stage("search")["status"], "NEEDS_ATTENTION")
                self.assertEqual(wf.advance(), "NEEDS_ATTENTION")    # it stays, also in a new process
                broken["on"] = False
                wf2 = Workflow(tmp)
                lines = wf2.retry()
                self.assertTrue(any("search of target geometry" in ln for ln in lines), lines)
                self.assertTrue(any("stage search" in ln for ln in lines), lines)
                self.assertEqual(wf2.project.stage("search")["status"], "RUNNING")
                for _ in range(4000):
                    status = wf2.advance()
                    if status != "RUNNING":
                        break
                self.assertEqual(status, DONE)
                self.assertEqual(wf2.project.target("geometry")["status"], "CONVERGED")
            finally:
                engine_class.read = original

    def test_path_image_is_searched_across_the_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            wf = run(tmp, AMMONIA_IMAGE % ("2", "image"))
            p = Project(tmp)
            self.assertEqual(p.state["status"], DONE, p.state["stages"])
            t = p.target("2")
            self.assertEqual((t["kind"], t["image"], t["status"]), ("image", 2, "CONVERGED"))
            with np.load(os.path.join(p.target_dir("2"), "directions.npz")) as z:
                vectors, tangent, curvature = z["vectors"].copy(), z["tangent"].copy(), z["curvature"].copy()
                self.assertEqual(z["coefficients"].shape, (1, 2))
            # the path direction is left out: one direction less than the subspace has, and it is across the path
            self.assertEqual(p.state["subspace"]["dim"], 2); self.assertEqual(vectors.shape[0], 1)
            np.testing.assert_allclose(vectors @ tangent, 0.0, atol=1e-10)
            self.assertTrue(np.all(curvature > 0))
            frames = [f[1] for f in G.read_xyz_frames(p.path_file)]
            np.testing.assert_allclose(tangent, S.path_tangent(frames, 2), atol=1e-8)
            b = wf.backend; drv = wf.driver("2")
            start = G.read_xyz(os.path.join(p.target_dir("2"), "start.xyz"))[1]
            final = np.asarray(drv.final_geometry())
            move = (final - start).reshape(-1)
            self.assertLess(abs(float(move @ tangent)), 1e-9)                 # the image did not move along the path
            self.assertGreater(float(np.linalg.norm(move)), 0.01)             # and it did move across the path

            def energy(z):                                                    # exact energy across the path
                return b.true_energy(start + (z * vectors[0]).reshape(-1, 3))
            z, h = 0.0, 1e-3
            for _ in range(40):                                               # Newton steps with numerical derivatives
                g = (energy(z + h) - energy(z - h)) / (2 * h)
                k = (energy(z + h) - 2 * energy(z) + energy(z - h)) / h ** 2
                z -= g / k
                if abs(g / k) < 1e-9:
                    break
            lowest = energy(z)
            self.assertLess(b.true_energy(final) - lowest, 0.06)              # the search found the lowest point across the path
            self.assertGreater(b.true_energy(start) - lowest, 0.15)           # which the starting image was not
            # the image did not slide down the path: it is still far above the minimum of the true surface
            e_min = b.true_energy(M.minimize(b.true, M.ammonia_geometry()))
            self.assertGreater(b.true_energy(final) - e_min, 2.0)
            de, _sd = drv.state["final"]["energy_change"]
            self.assertLess(de, 0.0)
            with open(os.path.join(tmp, "work", "driver.log")) as fh:
                self.assertIn("path image 2. The path direction (difference of the images 1 and 3", fh.read())

    def test_mirror_images_of_a_symmetric_path_end_at_the_same_energy(self):
        ends = []
        for image in ("2", "6"):                                              # the model path is symmetric about image 4
            with tempfile.TemporaryDirectory() as tmp:
                wf = run(tmp, AMMONIA_IMAGE % (image, "image"))
                self.assertEqual(Project(tmp).target(image)["status"], "CONVERGED")
                ends.append(wf.backend.true_energy(wf.driver(image).final_geometry()))
        self.assertLess(abs(ends[0] - ends[1]), 0.02)

    def test_an_end_of_the_path_cannot_be_a_path_image(self):
        for image in ("0", "8"):
            with tempfile.TemporaryDirectory() as tmp:
                run(tmp, AMMONIA_IMAGE % (image, "image"), max_passes=20)
                st = Project(tmp).state
                self.assertEqual(st["status"], "NEEDS_ATTENTION")
                self.assertIn("two neighbours", st["stages"]["path"]["message"])

    def test_images_as_minimum_and_saddle_targets_still_work(self):
        # the kinds minimum and saddle in mode images: all directions of the subspace, no path direction removed
        with tempfile.TemporaryDirectory() as tmp:
            run(tmp, AMMONIA_IMAGE % ("0, 4", "minimum, saddle"))
            p = Project(tmp)
            self.assertEqual(p.state["stages"]["search"]["status"], DONE, p.state["stages"])
            for name, negative in (("0", 0), ("4", 1)):
                self.assertNotIn("image", p.target(name))
                with np.load(os.path.join(p.target_dir(name), "directions.npz")) as z:
                    self.assertEqual(z["vectors"].shape[0], 2); self.assertNotIn("tangent", z.files)
                    self.assertEqual(int(np.sum(z["curvature"] < 0)), negative)

    def test_kind_image_is_checked_in_the_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "input.ini"), "w") as fh:               # a single geometry has no path
                fh.write("[geometry]\nmode = single\nkinds = image\n[model]\nmolecule = water\n")
            with self.assertRaises(ConfigError) as cm:
                Config.load(os.path.join(tmp, "input.ini"))
            self.assertIn("[geometry] kinds: allowed values are minimum and saddle", str(cm.exception))
            with open(os.path.join(tmp, "input.ini"), "w") as fh:
                fh.write(AMMONIA_IMAGE % ("2", "bead"))
            with self.assertRaises(ConfigError) as cm:
                Config.load(os.path.join(tmp, "input.ini"))
            self.assertIn("[geometry] kinds: allowed values are minimum, saddle and image", str(cm.exception))
            with open(os.path.join(tmp, "input.ini"), "w") as fh:
                fh.write(AMMONIA_IMAGE % ("2, 4", "image, saddle"))
            cfg = Config.load(os.path.join(tmp, "input.ini"))
            self.assertEqual(cfg.kinds, {"2": "image", "4": "saddle"})


    def test_mep_of_a_given_path_searches_every_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            wf = run(tmp, AMMONIA_CHOICE % ("images", "mep", ""))
            p = Project(tmp)
            self.assertEqual(p.state["status"], DONE, p.state["stages"])
            self.assertEqual(p.targets(), ["0", "1", "2", "3", "4"])                    # the model path has 5 images
            self.assertEqual([p.target(n)["kind"] for n in p.targets()], ["minimum", "image", "saddle", "image", "minimum"])
            self.assertTrue(all(p.target(n)["status"] == "CONVERGED" for n in p.targets()), p.state["targets"])
            b = wf.backend
            e = {n: b.true_energy(wf.driver(n).final_geometry()) for n in p.targets()}
            self.assertLess(e["0"], e["1"]); self.assertLess(e["1"], e["2"])          # the profile rises to the saddle
            self.assertLess(abs(e["1"] - e["3"]), 0.03); self.assertLess(abs(e["0"] - e["4"]), 0.03)   # mirror images
            for n, tangent in (("0", False), ("1", True), ("2", False), ("3", True)):
                with np.load(os.path.join(p.target_dir(n), "directions.npz")) as z:
                    self.assertEqual("tangent" in z.files, tangent)
                    self.assertEqual(z["vectors"].shape[0], 1 if tangent else 2)
            with open(os.path.join(tmp, "work", "driver.log")) as fh:
                log = fh.read()
            self.assertIn("targets = mep: 5 search(es) on the path of 5 images", log)
            self.assertIn("Saddle image: 2 (the image with the highest energy)", log)

    def test_barrier_of_a_given_path_is_two_searches(self):
        with tempfile.TemporaryDirectory() as tmp:
            wf = run(tmp, AMMONIA_CHOICE % ("images", "barrier", ""))
            p = Project(tmp)
            self.assertEqual(p.state["status"], DONE, p.state["stages"])
            self.assertEqual({n: p.target(n)["kind"] for n in p.targets()}, {"0": "minimum", "2": "saddle"})
            b = wf.backend
            barrier_true = b.true_energy(M.refine_stationary(b.true, M.ammonia_geometry(theta=120.0, r=1.0))) - \
                b.true_energy(M.minimize(b.true, M.ammonia_geometry()))
            e_min = wf.driver("0").final_energy(); e_sad = wf.driver("2").final_energy()
            barrier = (e_sad[0] - e_min[0]) * 627.509474
            err = np.hypot(e_sad[1], e_min[1]) * 627.509474
            self.assertLess(abs(barrier - barrier_true), 4 * err + 0.15, "barrier %.3f +/- %.3f, true %.3f" % (barrier, err, barrier_true))

    def test_mep_when_the_package_makes_the_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            run(tmp, AMMONIA_CHOICE % ("path", "mep", ""))
            p = Project(tmp)
            self.assertEqual(p.state["status"], DONE, p.state["stages"])
            self.assertEqual(p.targets(), ["reactant", "image1", "saddle", "image3", "product"])
            self.assertEqual([p.target(n)["kind"] for n in p.targets()], ["minimum", "image", "saddle", "image", "minimum"])
            self.assertEqual((p.target("image1")["image"], p.target("image3")["image"]), (1, 3))
            with np.load(os.path.join(p.target_dir("image3"), "directions.npz")) as z:
                self.assertIn("tangent", z.files)

    def test_the_words_and_automatic_kinds_in_the_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            def load(mode, targets, extra=""):
                with open(os.path.join(tmp, "input.ini"), "w") as fh:
                    fh.write(AMMONIA_CHOICE % (mode, targets, extra))
                return Config.load(os.path.join(tmp, "input.ini"))
            cfg = load("path", "barrier")
            self.assertEqual((cfg.selection, cfg.targets), ("barrier", ["reactant", "saddle"]))
            cfg = load("path", "MEP")
            self.assertEqual((cfg.selection, cfg.targets), ("mep", ["reactant", "saddle", "product"]))
            self.assertEqual(cfg.kinds, {"reactant": "minimum", "saddle": "saddle", "product": "minimum"})
            cfg = load("images", "mep", "saddle_image = 2")
            self.assertEqual((cfg.selection, cfg.targets, cfg.kinds_auto, cfg.saddle_image), ("mep", [], True, 2))
            cfg = load("images", "1, 2", "kinds = auto")
            self.assertEqual((cfg.selection, cfg.targets, cfg.kinds_auto, cfg.saddle_image), ("list", ["1", "2"], True, None))
            cfg = load("images", "1, 2", "kinds = image, saddle")                # a list with its kinds, as before
            self.assertEqual((cfg.kinds_auto, cfg.kinds), (False, {"1": "image", "2": "saddle"}))
            cfg = load("path", "reactant, saddle")
            self.assertEqual((cfg.selection, cfg.kinds_auto), ("list", False))
            for mode, targets, extra, needle in (
                    ("single", "barrier", "", "[geometry] targets = barrier needs a path"),
                    ("images", "mep", "kinds = minimum", "targets = mep chooses the kinds itself"),
                    ("images", "mep", "saddle_image = top", "[geometry] saddle_image = 'top' is not an image number or auto"),
                    ("path", "barrier, saddle", "", "are not allowed in mode path")):
                with self.assertRaises(ConfigError) as cm:
                    load(mode, targets, extra)
                self.assertIn(needle, str(cm.exception))

    def test_automatic_kinds_for_a_list_and_a_named_saddle_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            run(tmp, AMMONIA_CHOICE % ("images", "1, 2", "kinds = auto"))
            p = Project(tmp)
            self.assertEqual(p.state["stages"]["search"]["status"], DONE, p.state["stages"])
            self.assertEqual({n: p.target(n)["kind"] for n in p.targets()}, {"1": "image", "2": "saddle"})
        with tempfile.TemporaryDirectory() as tmp:                               # a saddle image that is an end of the path
            run(tmp, AMMONIA_CHOICE % ("images", "barrier", "saddle_image = 4"), max_passes=10)
            st = Project(tmp).state
            self.assertEqual(st["status"], "NEEDS_ATTENTION")
            self.assertIn("the saddle image must lie between the two ends", st["stages"]["path"]["message"])

    def test_wrong_number_of_negative_curvatures_needs_attention(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "input.ini"), "w") as fh:
                fh.write(AMMONIA)
            wf = Workflow(tmp)
            real = wf.backend.geometries

            def wrong(w):                                     # call the saddle a minimum
                g = real(w); g["targets"]["saddle"]["kind"] = "minimum"; return g
            wf.backend.geometries = wrong
            for _ in range(50):
                st = wf.advance()
                if st != "RUNNING":
                    break
            self.assertEqual(st, "NEEDS_ATTENTION")
            self.assertIn("negative curvature", Project(tmp).state["stages"]["directions"]["message"])

    def test_targets_from_the_energies_of_a_path_file(self):
        """The ammonia inversion path written by ORCA: 7 images with their energies, the planar saddle is image 3."""
        from types import SimpleNamespace
        from montepass.workflow import chosen_path_targets
        frames = G.read_xyz_frames(os.path.join(HERE, "data", "orca_610_nh3_neb_MEP_trj.xyz"))
        energies = [G.comment_energy(f[2]) for f in frames]
        cfg = SimpleNamespace(selection="barrier", saddle_image=None, targets=[], kinds_auto=True)
        chosen, note = chosen_path_targets(cfg, energies, "the path file")
        self.assertEqual(chosen, {0: "minimum", 3: "saddle"})
        self.assertIn("targets = barrier: 2 search(es) on the path of 7 images", note)
        self.assertIn("Saddle image: 3 (the image with the highest energy)", note)
        cfg.selection = "mep"
        chosen, _note = chosen_path_targets(cfg, energies, "the path file")
        self.assertEqual(list(chosen.values()), ["minimum", "image", "image", "saddle", "image", "image", "minimum"])
        cfg = SimpleNamespace(selection="list", saddle_image=None, targets=["3", "4", "5", "6"], kinds_auto=True)
        chosen, note = chosen_path_targets(cfg, energies, "the path file")
        self.assertEqual(chosen, {3: "saddle", 4: "image", 5: "image", 6: "minimum"})
        self.assertIn("kinds = auto: 4 search(es)", note)
        # a file without energies: the saddle image must be named
        cfg = SimpleNamespace(selection="mep", saddle_image=None, targets=[], kinds_auto=True)
        with self.assertRaises(ValueError) as cm:
            chosen_path_targets(cfg, [None] * 7, "the path file p.xyz")
        self.assertIn("name it with [geometry] saddle_image", str(cm.exception))
        cfg.saddle_image = 3
        chosen, note = chosen_path_targets(cfg, [None] * 7, "the path file p.xyz")
        self.assertEqual(chosen[3], "saddle"); self.assertIn("Saddle image: 3 ([geometry] saddle_image)", note)
        # a path that only goes up has no saddle: a barrier is not possible, the whole path is
        cfg = SimpleNamespace(selection="barrier", saddle_image=None, targets=[], kinds_auto=True)
        with self.assertRaises(ValueError):
            chosen_path_targets(cfg, [0.0, 1.0, 2.0, 3.0], "the path file")
        cfg.selection = "mep"
        chosen, note = chosen_path_targets(cfg, [0.0, 1.0, 2.0, 3.0], "the path file")
        self.assertEqual(list(chosen.values()), ["minimum", "image", "image", "minimum"]); self.assertIn("Saddle image: none", note)


if __name__ == "__main__":
    unittest.main()
