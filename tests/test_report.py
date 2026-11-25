"""The results folder: tables, plots and animations, built from model projects (no cluster program)."""
import csv
import json
import os
import re
import shutil
import sys
import tempfile
import unittest

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

from montepass import geometry as G, model as M, subspace as S  # noqa: E402
from montepass.engine.mock import MockEngine  # noqa: E402
from montepass.report import anim, build_report, molview, plots, tables  # noqa: E402
from montepass.report.demo import make_demo_project  # noqa: E402
from montepass.search import protocol_v1 as P1, protocol_v2 as P2  # noqa: E402
from montepass.search.dense import DenseScanSearch  # noqa: E402
from montepass.stats import reblock  # noqa: E402
from montepass.surrogate.analytic import AnalyticSurrogate  # noqa: E402
from montepass.units import HARTREE_TO_KCAL as H  # noqa: E402

from PIL import Image  # noqa: E402


NH3_PATH = os.path.join(HERE, "data", "orca_610_nh3_neb_MEP_trj.xyz")      # the ammonia inversion path written by ORCA


def read_summary(folder):
    with open(os.path.join(folder, "results", "summary.json")) as fh:
        return json.load(fh)


def path_report_data(symbols, frames, energies, title):
    """Report data that hold a path and nothing else."""
    data = tables.ReportData(project_dir=HERE, title=title, project={})
    data.path = tables.PathData(symbols=list(symbols), frames=[np.asarray(x, dtype=float) for x in frames],
                                energies=None if energies is None else np.asarray(energies, dtype=float))
    return data


def pdf_size_inches(path):
    """Page size of a PDF written by matplotlib, from its MediaBox entry."""
    with open(path, "rb") as fh:
        m = re.search(rb"/MediaBox\s*\[\s*0\s+0\s+([0-9.]+)\s+([0-9.]+)\s*\]", fh.read())
    return float(m.group(1)) / 72.0, float(m.group(2)) / 72.0


def driver_of(folder, name, engine, protocol):
    """A search driver object on the state file of a finished target (it only reads)."""
    tdir = os.path.join(folder, "work", "targets", name)
    symbols, start = G.read_xyz(os.path.join(tdir, "start.xyz"))
    with np.load(os.path.join(tdir, "directions.npz")) as z:
        vectors, curvature = z["vectors"].copy(), z["curvature"].copy()
    return DenseScanSearch(name, engine, None, symbols, start, vectors, curvature, protocol=protocol,
                           state_dir=os.path.join(tdir, "search"))


def water_project(folder, protocol, label, passes=None, seed=3, sigma=1.0):
    """A project with one target (model water, its 3 internal modes, no path), searched with ``protocol``.
    ``passes`` stops the search after that many passes (None: run to the end)."""
    ft, fs = M.water(True), M.water(False)
    xs = M.minimize(fs, M.water_geometry())
    Hs = M.hessian(fs, xs)
    sub = S.mode_subspace(Hs, xs)
    d = S.search_directions(S.project_hessian(Hs, sub.basis), sub)
    tdir = os.path.join(folder, "work", "targets", "geometry")
    os.makedirs(tdir)
    G.write_xyz(os.path.join(tdir, "start.xyz"), list(ft.symbols), xs, "start geometry of target geometry")
    np.savez(os.path.join(tdir, "directions.npz"), vectors=d.vectors, curvature=d.curvature)
    sub.save(os.path.join(folder, "work", "subspace.npz"))
    engine = MockEngine(ft, fs, sigma_block=sigma, blocks_per_tick=20, seed=seed, transient=20.0, equil_blocks=0.6)
    drv = DenseScanSearch("geometry", engine, AnalyticSurrogate(fs), list(ft.symbols), xs, d.vectors, d.curvature,
                          protocol=protocol, state_dir=os.path.join(tdir, "search"))
    if passes is None:
        drv.run()
    else:
        for _ in range(passes):
            engine.advance()
            drv.tick()
    with open(os.path.join(folder, "work", "project.json"), "w") as fh:
        json.dump({"name": "model_water", "stages": {"search": {"status": "RUNNING", "message": ""}},
                   "targets": {"geometry": {"kind": "minimum", "protocol": label, "status": drv.status,
                                            "message": drv.state.get("message", "")}}}, fh)
    return engine, drv


class TestDemoReport(unittest.TestCase):
    """The demonstration project: ammonia inversion, a path, two targets (minimum and saddle), protocol v2."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.folder = os.path.join(cls.tmp, "demo")
        cls.demo = make_demo_project(cls.folder)
        cls.files = build_report(cls.folder, cls.demo.engine, title="ammonia inversion (model)")
        cls.results = os.path.join(cls.folder, "results")
        cls.summary = read_summary(cls.folder)
        cls.protocol = P2.ProtocolV2(cap=120, eps_total=0.05, g_move=0.1)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def path(self, *parts):
        return os.path.join(self.results, *parts)

    def test_bond_lengths_and_angles_along_the_path(self):
        path = tables.collect(self.folder, self.demo.engine).path
        internal = tables.path_internal(path)
        self.assertEqual(internal["names"], ["N1-H2", "N1-H3", "N1-H4", "H2-N1-H3", "H2-N1-H4", "H3-N1-H4"])
        self.assertEqual(internal["values"].shape, (len(path.frames), 6))
        bonds, n_bonds = tables.coordinate_groups(internal, "bond")
        angles, n_angles = tables.coordinate_groups(internal, "angle")
        # the three bonds are equal at every image, and so are the three angles: one line each
        self.assertEqual((n_bonds, n_angles, len(bonds), len(angles)), (1, 1, 1, 1))
        self.assertEqual(bonds[0]["label"], "N1-H2 and 2 equal bonds")
        self.assertEqual(angles[0]["label"], "H2-N1-H3 and 2 equal angles")
        self.assertAlmostEqual(float(angles[0]["values"].max()), 120.0, delta=0.05)       # planar at the saddle
        self.assertLess(float(angles[0]["values"][0]), 110.0)                           # a pyramid at the ends
        with open(self.path("geometries", "path_internal_coordinates.csv")) as fh:
            rows = list(csv.reader(fh))
        self.assertEqual(rows[0][0], "image")
        self.assertEqual(rows[0][1], "bond_N1-H2_angstrom"); self.assertEqual(rows[0][4], "angle_H2-N1-H3_degree")
        self.assertEqual(len(rows), 1 + len(path.frames))
        frames = path.frames
        self.assertAlmostEqual(float(rows[1][1]), G.distance(frames[0], 0, 1), places=4)
        self.assertAlmostEqual(float(rows[-1][4]), G.angle(frames[-1], 1, 0, 2), places=2)
        self.assertAlmostEqual(float(rows[1][-1]), 0.0, places=6)                       # energy relative to image 0

    def test_searches_ended(self):
        self.assertEqual(self.demo.status, {"reactant": "CONVERGED", "saddle": "CONVERGED"})

    def test_files_exist_and_are_not_empty(self):
        expected = ["summary.md", "summary.json", "energies.csv", os.path.join("geometries", "bond_table.csv"),
                    os.path.join("geometries", "path_internal_coordinates.csv"), os.path.join("gifs", "path.gif")]
        for stem in ("pca_variance", "path_profile", "path_geometry"):
            expected += [os.path.join("plots", stem + ext) for ext in (".png", ".pdf")]
        for target in ("reactant", "saddle"):
            expected += [os.path.join("geometries", "%s_%s.xyz" % (target, k)) for k in ("start", "final", "trajectory")]
            expected += [os.path.join("plots", "%s_%s%s" % (target, k, ext)) for k in ("convergence", "scans", "geometry")
                         for ext in (".png", ".pdf")]
            expected.append(os.path.join("gifs", "%s_optimization.gif" % target))
        for rel in expected:
            self.assertTrue(os.path.isfile(self.path(rel)), rel)
            self.assertGreater(os.path.getsize(self.path(rel)), 0, rel)
        self.assertEqual(sorted(os.path.relpath(f, self.results) for f in self.files), sorted(expected))
        self.assertEqual(sorted(self.summary["files"]), sorted(expected))
        # ammonia has 3 equal bonds and 3 equal angles: the map of all bonds and angles is not needed
        self.assertEqual(self.summary["skipped"], [{"item": "plots/path_geometry_map", "reason":
                                                   "every bond and angle of this molecule is a line in plots/path_geometry"}])
        self.assertEqual(self.summary["warnings"], [])
        self.assertEqual(self.summary["problems"], [])
        leftovers = [f for _d, _s, names in os.walk(self.results) for f in names if ".tmp" in f]
        self.assertEqual(leftovers, [])

    def test_energies_agree_with_the_driver(self):
        for name in ("reactant", "saddle"):
            t = self.summary["targets"][name]
            drv = driver_of(self.folder, name, self.demo.engine, self.protocol)
            de, sd = drv.state["final"]["energy_change"]
            self.assertAlmostEqual(t["energy_change_kcal"]["value"], de, places=9)
            self.assertAlmostEqual(t["energy_change_kcal"]["error"], sd, places=9)
            self.assertAlmostEqual(t["energy_change_kcal_stored_by_driver"]["value"], de, places=12)
            e0, ef = drv.start_energy(), drv.final_energy()
            self.assertAlmostEqual(t["start_energy_hartree"]["value"], e0[0], places=12)
            self.assertAlmostEqual(t["start_energy_hartree"]["error"], e0[1], places=12)
            self.assertAlmostEqual(t["final_energy_hartree"]["value"], ef[0], places=12)
            self.assertAlmostEqual(t["final_energy_hartree"]["error"], ef[1], places=12)
            self.assertEqual(t["blocks"], {"search": drv.search_blocks(), "total": drv.total_blocks()})
            self.assertEqual(t["iterations"], len(drv.state["iters"]))
            self.assertEqual(t["status"], "CONVERGED")
            self.assertTrue(t["start_energy_hartree"]["complete"] and t["final_energy_hartree"]["complete"])
            for row, k in zip(t["history"], sorted(int(x) for x in drv.state["iters"])):
                comp = drv.state["iters"][str(k)]["composed"]
                self.assertEqual(row["iteration"], k)
                self.assertEqual(row["verdict"], comp["verdict"])
                self.assertAlmostEqual(row["G"], comp["G"], places=12)
                self.assertAlmostEqual(row["joint_bound"], comp["joint_bound"], places=12)
                self.assertAlmostEqual(row["step_length"], comp["step_norm"], places=12)
                ce = drv.centre_energy(k)
                self.assertAlmostEqual(row["centre_energy"]["value"], ce[0], places=12)

    def test_statement_in_plain_words(self):
        for name, phrase in (("reactant", "the starting geometry is not the DMC minimum"),
                             ("saddle", "the DMC energy of the optimized saddle geometry is")):
            t = self.summary["targets"][name]
            de, sd = t["energy_change_kcal"]["value"], t["energy_change_kcal"]["error"]
            self.assertEqual(t["significant"], abs(de) > 2.0 * sd)
            if t["significant"]:
                self.assertIn(phrase, t["statement"])
                self.assertIn("lower" if de < 0 else "higher", t["statement"].lower())
            else:
                self.assertIn("no significant change", t["statement"])
        # with the noise of this project the gain of 0.24 kcal/mol at the minimum is resolved
        self.assertTrue(self.summary["targets"]["reactant"]["significant"])
        self.assertLess(self.summary["targets"]["reactant"]["energy_change_kcal"]["value"], 0.0)

    def test_barriers_agree_with_the_model(self):
        true, sur = self.demo.true_surface, self.demo.surrogate_surface
        self.assertEqual(len(self.summary["barriers"]), 1)
        b = self.summary["barriers"][0]
        self.assertEqual((b["saddle"], b["minimum"]), ("saddle", "reactant"))
        geo = {}
        for name in ("reactant", "saddle"):
            for kind in ("start", "final"):
                geo[name, kind] = G.read_xyz(self.path("geometries", "%s_%s.xyz" % (name, kind)))[1]
        exact_start = true(geo["saddle", "start"]) - true(geo["reactant", "start"])
        exact_final = true(geo["saddle", "final"]) - true(geo["reactant", "final"])
        self.assertLess(abs(b["start"]["value"] - exact_start), 4.0 * b["start"]["error"])
        self.assertLess(abs(b["optimized"]["value"] - exact_final), 4.0 * b["optimized"]["error"])
        self.assertLess(abs(b["change"]["value"] - (exact_final - exact_start)), 4.0 * b["change"]["error"])
        self.assertLess(b["start"]["error"], 0.1); self.assertLess(b["optimized"]["error"], 0.1)
        # the barrier of the true surface between its own stationary points (5.42 kcal/mol): the optimized
        # geometries are there within the tolerance of the searches
        x_min = M.minimize(true, M.ammonia_geometry())
        x_sad = M.refine_stationary(true, M.ammonia_geometry(theta=120.0, r=1.0))
        self.assertLess(abs(b["optimized"]["value"] - (true(x_sad) - true(x_min))), 4.0 * b["optimized"]["error"] + 0.05)
        # the surrogate barrier comes from the path file and has no noise
        frames = G.read_xyz_frames(os.path.join(self.folder, "work", "path.xyz"))
        e = [sur(f[1]) for f in frames]
        self.assertTrue(b["surrogate_path"]["exact"])
        self.assertAlmostEqual(b["surrogate_path"]["value"], max(e) - e[0], places=5)
        self.assertEqual((b["surrogate_path"]["minimum_image"], b["surrogate_path"]["saddle_image"]), (0, len(frames) // 2))
        self.assertLess(b["surrogate_path"]["value"], b["start"]["value"] - 4.0 * b["start"]["error"])   # the model surrogate is too low

    def test_geometry_files_and_bond_table(self):
        for name in ("reactant", "saddle"):
            drv = driver_of(self.folder, name, self.demo.engine, self.protocol)
            symbols, final = G.read_xyz(self.path("geometries", "%s_final.xyz" % name))
            self.assertEqual(symbols, ["N", "H", "H", "H"])
            np.testing.assert_allclose(final, drv.final_geometry(), atol=1e-9)
            _s, start = G.read_xyz(self.path("geometries", "%s_start.xyz" % name))
            np.testing.assert_allclose(start, G.read_xyz(os.path.join(self.folder, "work", "targets", name, "start.xyz"))[1], atol=1e-9)
            traj = G.read_xyz_frames(self.path("geometries", "%s_trajectory.xyz" % name))
            self.assertEqual(len(traj), len(drv.trajectory()))
            for frame, centre in zip(traj, drv.trajectory()):
                np.testing.assert_allclose(frame[1], centre, atol=1e-9)
            g = self.summary["targets"][name]["geometry_change_angstrom"]
            self.assertAlmostEqual(g["rmsd"], G.kabsch(final, start)[1], places=9)
            self.assertGreater(g["max_displacement"], g["rmsd"])
            self.assertGreater(g["position_error_rms"], 0.0)
            self.assertLess(g["position_error_rms"], g["rmsd"])          # the geometry change is larger than its noise
        with open(self.path("geometries", "bond_table.csv")) as fh:
            rows = list(csv.DictReader(fh))
        self.assertEqual(len(rows), 12)                                   # 3 bonds and 3 angles for each of 2 targets
        bond = [r for r in rows if r["target"] == "reactant" and r["atoms"] == "N1-H2"][0]
        _s, start = G.read_xyz(self.path("geometries", "reactant_start.xyz"))
        _s, final = G.read_xyz(self.path("geometries", "reactant_final.xyz"))
        self.assertAlmostEqual(float(bond["start"]), G.distance(start, 0, 1), places=4)
        self.assertAlmostEqual(float(bond["change"]), G.distance(final, 0, 1) - G.distance(start, 0, 1), places=4)
        self.assertLess(float(bond["change"]), 0.0)                       # the surrogate N-H bond is 0.010 A too long
        self.assertGreater(float(bond["statistical_error_of_final"]), 0.0)
        self.assertEqual(bond["unit"], "angstrom")
        angle = [r for r in rows if r["target"] == "saddle" and r["atoms"] == "H2-N1-H3"][0]
        self.assertAlmostEqual(float(angle["final"]), 120.0, delta=0.5)   # the saddle stays planar
        self.assertEqual(angle["unit"], "degree")

    def test_energies_csv(self):
        with open(self.path("energies.csv")) as fh:
            rows = list(csv.DictReader(fh))
        engine = self.demo.engine
        with_blocks = [k for k in engine.points if engine.read(k).nblocks > 0]
        self.assertEqual(sorted(r["key"] for r in rows), sorted(with_blocks))
        self.assertEqual({r["role"] for r in rows}, {"centre", "scan", "final"})
        for r in rows[:6] + rows[-3:]:
            d = engine.read(r["key"])
            self.assertEqual(int(r["blocks"]), d.nblocks)
            drop = P2.mser_drop(d.energy, 10, 50)
            mean, err = reblock(d.energy[drop:], d.weight[drop:])
            self.assertAlmostEqual(float(r["energy_hartree"]), mean, places=9)
            self.assertAlmostEqual(float(r["error_hartree"]), err, places=9)
            self.assertAlmostEqual(float(r["surrogate_energy_hartree"]), d.e_surrogate, places=9)
            self.assertEqual(int(r["blocks_discarded"]), drop)
        scan = [r for r in rows if r["role"] == "scan" and r["target"] == "saddle" and r["direction"] == "0"]
        self.assertTrue(all(float(r["displacement_angstrom"]) > 0 for r in scan))     # one-sided: slope zero by symmetry

    def test_plots_have_the_declared_size(self):
        sizes = {"pca_variance": (plots.MEDIUM, 2.9), "path_profile": (plots.MEDIUM, 3.4),
                 "path_geometry": (plots.MEDIUM, 0.55 + 2 * 2.25),
                 "reactant_convergence": (plots.MEDIUM, 5.3), "reactant_scans": (plots.WIDE, 3.05),
                 "saddle_geometry": (plots.WIDE, 3.0)}
        for stem, (w, h) in sizes.items():
            with Image.open(self.path("plots", stem + ".png")) as im:
                im.load()
                self.assertEqual(im.format, "PNG")
                self.assertEqual(im.size, plots.png_size(w, h), stem)
                self.assertEqual(im.size, (int(round(w * 200)), int(round(h * 200))), stem)
            pw, ph = pdf_size_inches(self.path("plots", stem + ".pdf"))
            self.assertAlmostEqual(pw, w, places=3); self.assertAlmostEqual(ph, h, places=3)
        with Image.open(self.path("plots", "path_profile.png")) as im:
            colours = im.convert("RGB").getcolors(maxcolors=1 << 24)
        self.assertGreater(len(colours), 20)                               # a drawing, not an empty canvas

    def test_gifs(self):
        for name in ("path.gif", "reactant_optimization.gif", "saddle_optimization.gif"):
            p = self.path("gifs", name)
            self.assertLess(os.path.getsize(p), 3000000, name)
            with Image.open(p) as im:
                self.assertEqual(im.format, "GIF")
                self.assertGreater(im.n_frames, 5, name)
                # the path animation has 3 panels (energy, bond lengths, angles); a search has 1 or 2 views
                heights = [anim.path_figure_height(3)] if name == "path.gif" else \
                    [anim.optimization_figure_height(1), anim.optimization_figure_height(2)]
                self.assertEqual(im.size[0], int(round(plots.WIDE * 120)))
                self.assertIn(im.size[1], [int(round(h * 120)) for h in heights], name)
                im.seek(im.n_frames - 1)                                   # every frame can be decoded
                im.load()

    def test_summary_text(self):
        with open(self.path("summary.md")) as fh:
            text = fh.read()
        for needle in ("# ammonia inversion (model): results", "## Barriers", "## Search history of target reactant",
                       "## Search history of target saddle", "the starting geometry is not the DMC minimum",
                       "slope is zero by symmetry", "## What was not made, and why",
                       "plots/path_geometry_map: every bond and angle of this molecule is a line in plots/path_geometry",
                       "## Geometry along the path", "| N1-H2 | 3 |", "1 standard error"):
            self.assertIn(needle, text)
        self.assertNotIn("nan", text.lower().replace("minan", ""))
        last = self.summary["targets"]["saddle"]["last_scan"]["directions"]
        self.assertEqual(last[0]["by_symmetry"], "odd")
        self.assertEqual(last[0]["slope"], 0.0)

    def test_report_still_builds_without_path_and_with_one_target(self):
        folder = os.path.join(self.tmp, "reduced")
        shutil.copytree(self.folder, folder)                               # the old results folder is copied too
        os.remove(os.path.join(folder, "work", "path.xyz"))
        shutil.rmtree(os.path.join(folder, "work", "targets", "saddle"))
        pfile = os.path.join(folder, "work", "project.json")
        with open(pfile) as fh:
            project = json.load(fh)
        del project["targets"]["saddle"]
        with open(pfile, "w") as fh:
            json.dump(project, fh)
        files = build_report(folder, self.demo.engine)
        rel = sorted(os.path.relpath(f, os.path.join(folder, "results")) for f in files)
        s = read_summary(folder)
        self.assertEqual(s["title"], "model_ammonia_inversion")            # no title given: the name of the project
        self.assertEqual(list(s["targets"]), ["reactant"])
        self.assertEqual(s["barriers"], [])
        self.assertIsNone(s["path"])
        skipped = {x["item"]: x["reason"] for x in s["skipped"]}
        self.assertIn("no path", skipped["plots/path_profile"])
        self.assertIn("no path", skipped["gifs/path.gif"])
        self.assertIn(os.path.join("plots", "reactant_convergence.png"), rel)
        self.assertIn(os.path.join("plots", "pca_variance.png"), rel)      # the subspace file is still there
        self.assertIn(os.path.join("gifs", "reactant_optimization.gif"), rel)
        self.assertEqual(s["warnings"], [])
        # files of the first report that this report did not write again are gone
        for old in (os.path.join("gifs", "path.gif"), os.path.join("plots", "saddle_scans.png"),
                    os.path.join("geometries", "saddle_final.xyz"), os.path.join("plots", "path_profile.pdf")):
            self.assertFalse(os.path.exists(os.path.join(folder, "results", old)), old)
            self.assertIn(old, s["removed_outdated_files"])
        self.assertEqual(s["targets"]["reactant"]["energy_change_kcal"],
                         self.summary["targets"]["reactant"]["energy_change_kcal"])


class TestOtherProjects(unittest.TestCase):
    def test_running_search_then_finished_search(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine, drv = water_project(tmp, P2.ProtocolV2(cap=120, eps_total=0.05, g_move=0.1), "v2", passes=2)
            self.assertEqual(drv.status, "RUNNING")
            files = build_report(tmp, engine, gifs=False)
            s = read_summary(tmp)
            t = s["targets"]["geometry"]
            self.assertEqual(t["status"], "RUNNING")
            self.assertFalse(t["search_ended"])
            self.assertIsNone(t["energy_change_kcal"]); self.assertIsNone(t["significant"])
            self.assertIn("has not ended", t["statement"])
            skipped = {x["item"]: x["reason"] for x in s["skipped"]}
            self.assertIn("has not ended", skipped[os.path.join("geometries", "geometry_final.xyz")])
            self.assertIn("has not ended", skipped["plots/geometry_geometry"])
            self.assertIn("kind 'modes'", skipped["plots/pca_variance"])
            self.assertIn("switched off", skipped["gifs"])
            self.assertTrue(any(f.endswith("geometry_start.xyz") for f in files))
            self.assertTrue(any("no path" in x["reason"] for x in s["not_available"]))
            # the same project after the search has ended
            drv.run()
            self.assertEqual(drv.status, "CONVERGED")
            build_report(tmp, engine, gifs=False)
            t = read_summary(tmp)["targets"]["geometry"]
            de, sd = drv.state["final"]["energy_change"]
            self.assertAlmostEqual(t["energy_change_kcal"]["value"], de, places=9)
            self.assertAlmostEqual(t["energy_change_kcal"]["error"], sd, places=9)
            self.assertTrue(os.path.isfile(os.path.join(tmp, "results", "plots", "geometry_scans.png")))
            self.assertTrue(os.path.isfile(os.path.join(tmp, "results", "geometries", "geometry_final.xyz")))

    def test_protocol_v1(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine, drv = water_project(tmp, P1.V1, "v1", sigma=3.0)
            self.assertIn(drv.status, ("CONVERGED", "NOT_CONVERGED"))
            build_report(tmp, engine, gifs=False)
            s = read_summary(tmp)
            t = s["targets"]["geometry"]
            self.assertEqual(t["protocol"], "v1")
            kf = drv.final_iteration()
            e1, ef = drv.centre_energy(1), drv.centre_energy(kf)
            self.assertAlmostEqual(t["start_energy_hartree"]["value"], e1[0], places=12)
            self.assertAlmostEqual(t["final_energy_hartree"]["value"], ef[0], places=12)
            if kf > 1:
                self.assertAlmostEqual(t["energy_change_kcal"]["value"], (ef[0] - e1[0]) * H, places=9)
                self.assertAlmostEqual(t["energy_change_kcal"]["error"], float(np.hypot(ef[1], e1[1])) * H, places=9)
            else:
                self.assertEqual(t["energy_change_kcal"], {"value": 0.0, "error": 0.0})
                self.assertIn("did not move", t["statement"])
            self.assertEqual(t["blocks"]["total"], drv.total_blocks())
            self.assertEqual(len(t["history"]), len(drv.state["iters"]))
            self.assertIsNotNone(t["history"][0]["bound_v1"])
            self.assertIsNone(t["history"][0]["G"])
            self.assertEqual(s["warnings"], [])
            with open(os.path.join(tmp, "results", "summary.md")) as fh:
                self.assertIn("remaining-energy bound", fh.read())
            for stem in ("geometry_convergence", "geometry_scans"):
                self.assertTrue(os.path.isfile(os.path.join(tmp, "results", "plots", stem + ".png")), stem)

    def test_more_than_nine_directions_give_two_scan_figures(self):
        """A quadratic model surface with 11 search directions (4 atoms, no symmetry, no surrogate)."""
        rng = np.random.default_rng(7)
        symbols = ["C", "O", "N", "H"]
        x0 = np.array([[0.0, 0.0, 0.0], [1.21, 0.05, 0.02], [-0.61, 1.17, -0.08], [-0.52, -0.93, 0.31]])
        Q = np.linalg.qr(rng.normal(size=(12, 12)))[0].T[:11]              # 11 orthonormal directions
        k = np.linspace(300.0, 1500.0, 11)
        shift = rng.normal(size=11) * 0.02                                  # the minimum is displaced from the start

        def surface(X):
            z = Q @ (np.asarray(X).reshape(-1) - x0.reshape(-1)) - shift
            return float(0.5 * np.sum(k * z * z))

        with tempfile.TemporaryDirectory() as tmp:
            tdir = os.path.join(tmp, "work", "targets", "min")
            os.makedirs(tdir)
            G.write_xyz(os.path.join(tdir, "start.xyz"), symbols, x0)
            engine = MockEngine(surface, None, sigma_block=0.5, blocks_per_tick=40, seed=2)
            drv = DenseScanSearch("min", engine, None, symbols, x0, Q, k, protocol=P2.ProtocolV2(cap=120, symmetry=False),
                                  state_dir=os.path.join(tdir, "search"))
            drv.run()
            files = build_report(tmp, engine, title="quadratic model", gifs=False)     # no project.json: targets from the folders
            s = read_summary(tmp)
            self.assertEqual(s["warnings"], [])
            self.assertEqual(s["targets"]["min"]["kind"], "minimum")        # from the roles in the search state
            names = [os.path.basename(f) for f in files]
            self.assertIn("min_scans.png", names); self.assertIn("min_scans_2.png", names)
            self.assertNotIn("min_scans_3.png", names)
            with Image.open(os.path.join(tmp, "results", "plots", "min_scans.png")) as im:
                self.assertEqual(im.size, plots.png_size(plots.WIDE, 0.4 + 0.75 + 3 * 1.9))    # 9 panels in 3 rows
            with Image.open(os.path.join(tmp, "results", "plots", "min_scans_2.png")) as im:
                self.assertEqual(im.size, plots.png_size(plots.WIDE, 0.4 + 0.75 + 1.9))        # 2 panels in 1 row
            self.assertEqual(len(s["targets"]["min"]["last_scan"]["directions"]), 11)
            self.assertTrue(any("project.json is missing" in x["reason"] for x in s["not_available"]))

    def test_failures_are_visible(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                build_report(tmp, None)                                     # no project in the folder
            engine, drv = water_project(tmp, P2.ProtocolV2(cap=120), "v2")
            sfile = os.path.join(tmp, "work", "targets", "geometry", "search", "geometry_state.json")
            with open(sfile) as fh:
                good = fh.read()
            with open(sfile, "w") as fh:
                fh.write(good[: len(good) // 2])                            # a damaged state file
            build_report(tmp, engine, gifs=False)
            t = read_summary(tmp)["targets"]["geometry"]
            self.assertTrue(any("could not be read" in p for p in t["problems"]), t["problems"])
            self.assertIsNone(t["energy_change_kcal"])
            with open(sfile, "w") as fh:
                fh.write(good)

            class Broken(MockEngine):                                       # an engine whose read fails for one point
                def read(self, key):
                    if key.endswith("it1/centre_s2"):
                        raise IOError("scalar file is damaged")
                    return super().read(key)

            broken = Broken(engine.true_surface, engine.surrogate_surface, sigma_block=1.0, blocks_per_tick=20, seed=3,
                            transient=20.0, equil_blocks=0.6)
            broken.points, broken.tick = engine.points, engine.tick
            build_report(tmp, broken, gifs=False)
            s = read_summary(tmp)
            self.assertTrue(any("could not be read" in p and "damaged" in p for p in s["problems"]), s["problems"])
            self.assertEqual(s["targets"]["geometry"]["start_energy_hartree"]["runs"], 1)      # the other run is still used
            self.assertFalse(s["targets"]["geometry"]["start_energy_hartree"]["complete"])
            # without an engine the geometries are still written
            files = build_report(tmp, None, gifs=False)
            self.assertTrue(any(f.endswith("geometry_final.xyz") for f in files))
            self.assertIsNone(read_summary(tmp)["targets"]["geometry"]["start_energy_hartree"])


class TestPieces(unittest.TestCase):
    def test_number_formats(self):
        self.assertEqual(tables.fmt_pm(-0.2357, 0.0489), "-0.236 +/- 0.049")
        self.assertEqual(tables.fmt_pm(5.4, 1.3), "5.4 +/- 1.3")
        self.assertEqual(tables.fmt_pm(-56.5012345, 0.000123), "-56.50123 +/- 0.00012")
        self.assertEqual(tables.fmt_pm(0.0, 0.0), "0.000000 (exact)")
        self.assertEqual(tables.fmt_pm(None, 0.1), "n/a")
        self.assertEqual(tables.comment_energy("image 3 E -56.5123456789"), -56.5123456789)
        self.assertEqual(tables.comment_energy("Coordinates from a job E -40.5d0"), -40.5)
        self.assertIsNone(tables.comment_energy("ENERGY unknown"))
        self.assertIsNone(tables.comment_energy(""))
        self.assertEqual(json.dumps(tables.jsonable({"a": np.float64("nan"), "b": np.arange(2), "c": (np.int64(3), True)})),
                         '{"a": null, "b": [0, 1], "c": [3, true]}')

    def test_statistics(self):
        self.assertEqual(tables.combine([]), None)
        m, e = tables.combine([(1.0, 0.1), (2.0, 0.1)])
        self.assertAlmostEqual(m, 1.5); self.assertAlmostEqual(e, 0.1 / np.sqrt(2.0))
        self.assertEqual(tables.combine([(1.0, 0.0), (3.0, 0.2)]), (1.0, 0.0))        # a run without noise is exact
        self.assertEqual(tables.DropRule.of("v1"), tables.DropRule(drop=50, min_blocks_after_drop=30))
        self.assertEqual(tables.DropRule.of("classic").drop, "auto")
        self.assertEqual(tables.DropRule.of(P2.ProtocolV2(drop=20)).drop, 20)
        rng = np.random.default_rng(0)
        trace = rng.normal(size=25)
        self.assertIsNone(tables.trace_stat(trace, np.ones(25), tables.DropRule()))    # too short to be used
        trace = rng.normal(size=200)
        mean, err, drop = tables.trace_stat(trace, np.ones(200), tables.DropRule(drop=50))
        self.assertEqual(drop, 50); self.assertAlmostEqual(mean, trace[50:].mean())

    def test_two_plain_views_show_the_true_geometry(self):
        # the inversion of ammonia leaves the best-fit plane: two views, from above and from the side
        up, down = M.ammonia_geometry(up=True), M.ammonia_geometry(up=False)
        frames = M.interpolate_path(up, down, 7, align=False)
        views = molview.View.fit_views(["N", "H", "H", "H"], frames)
        self.assertEqual([v.tilt for v in views], [0.0, 90.0])
        self.assertIn("from above", views[0].describe()); self.assertIn("from the side", views[1].describe())
        self.assertLess(max(len(v.describe()) for v in views), 40)           # the line must fit under a narrow drawing
        for v in views:
            np.testing.assert_allclose(v.axes @ v.axes.T, np.eye(3), atol=1e-12)
            self.assertAlmostEqual(float(np.linalg.det(v.axes)), 1.0, places=9)          # a rotation: no mirror image
            for X in frames:
                xy, depth = v.project(X)
                for i in range(4):
                    for j in range(i):
                        # screen distance and depth difference together give the true distance of two atoms
                        self.assertAlmostEqual(float(np.hypot(np.linalg.norm(xy[i] - xy[j]), depth[i] - depth[j])),
                                               G.distance(X, i, j), places=9)
        # from above, the three N-H bonds of every image have the same length on the screen
        for X in frames:
            xy, _depth = views[0].project(X)
            lengths = [np.linalg.norm(xy[k] - xy[0]) for k in (1, 2, 3)]
            self.assertLess(max(lengths) - min(lengths), 1e-6)
        # from the side, the nitrogen atom is below the plane of the hydrogen atoms at one end and above at the other
        side = [views[1].project(X) for X in (frames[0], frames[-1])]
        height = [xy[0, 1] - xy[1:, 1].mean() for xy, _depth in side]
        self.assertLess(height[0] * height[-1], 0.0)
        self.assertGreater(abs(height[0]), 0.25)
        # a motion inside the plane needs one view
        water = M.water_geometry()
        stretched = water.copy(); stretched[1] += 0.2 * (water[1] - water[0])
        self.assertEqual(len(molview.View.fit_views(["O", "H", "H"], [water, stretched])), 1)
        self.assertEqual(len(molview.View.fit_views(["O", "H", "H"], [water])), 1)

    def test_frame_label(self):
        labels = ["iteration 1", "iteration 2: centre of a rejected step", "final geometry (centre of iteration 2)"]
        self.assertEqual(anim.frame_label(labels, 0.0), "iteration 1")
        self.assertEqual(anim.frame_label(labels, 1.0), "iteration 2: centre of a rejected step")
        self.assertEqual(anim.frame_label(labels, 2.0), "final geometry (centre of iteration 2)")
        self.assertEqual(anim.frame_label(labels, 0.5), "iteration 1 to iteration 2")
        self.assertEqual(anim.frame_label(labels, 1.25), "iteration 2 to final geometry")
        self.assertEqual(anim.frame_label(labels, 2.4), "final geometry (centre of iteration 2)")   # past the last one

    def test_magnification(self):
        self.assertEqual(anim.magnification(0.3), 1)
        self.assertEqual(anim.magnification(0.15), 1)
        self.assertEqual(anim.magnification(0.012), 20)
        self.assertEqual(anim.magnification(0.0031), 50)
        self.assertEqual(anim.magnification(1e-9), 10000)

    def test_change_colours(self):
        self.assertEqual([plots.nice_limit(v) for v in (0.0432, 0.0019, 1.13, 26.0, 0.5)], [0.05, 0.002, 1.5, 30.0, 0.5])
        self.assertEqual(plots.nice_limit(0.001, smallest=0.005), 0.005)
        norm, cmap = plots.change_scale(3.6, cap=0.30)
        self.assertEqual((norm.vmin, norm.vmax), (-0.30, 0.30))
        self.assertEqual(cmap(norm(3.6)), cmap(norm(0.30)))                   # a change beyond the scale: the end colour
        water = M.water_geometry()
        longer = water.copy(); longer[1] += 0.05 * (water[1] - water[0]) / np.linalg.norm(water[1] - water[0])
        norm, cmap, largest = plots.bond_change_scale(["O", "H", "H"], [water, longer], [(0, 1), (0, 2)])
        self.assertAlmostEqual(largest, 0.05, places=9)
        colours = molview.bond_change_colors([(0, 1), (0, 2)], longer, water, norm, cmap)
        r, g, b, _a = colours[(0, 1)]
        self.assertGreater(b, r)                                             # a longer bond is purple
        r0, g0, b0, _a = colours[(0, 2)]
        self.assertGreater(min(r0, g0, b0), 0.9)                             # a bond that did not change is white
        r, g, b, _a = molview.bond_change_colors([(0, 1)], water, longer, norm, cmap)[(0, 1)]
        self.assertGreater(r, b)                                             # a shorter bond is orange
        # a drawing with coloured bonds: a dark line with a narrower coloured line on top of it
        with plots.style():
            fig = plots.new_figure(plots.NARROW, 2.0)
            ax = fig.add_subplot(1, 1, 1)
            art = molview.draw_molecule(ax, ["O", "H", "H"], longer, pairs=[(0, 1), (0, 2)], colored_bonds=True,
                                        bond_colors=colours)
            self.assertEqual(sorted(art.fills), [(0, 1), (0, 2)])
            self.assertTrue(all(f.get_visible() for f in art.fills.values()))
            self.assertLess(art.fills[(0, 1)].get_linewidth(), art.lines[(0, 1)].get_linewidth())
            art.update(longer)                                               # no colours given: plain bonds
            self.assertFalse(any(f.get_visible() for f in art.fills.values()))
            plain = molview.draw_molecule(fig.add_subplot(2, 1, 2), ["O", "H", "H"], water)
            self.assertEqual(plain.fills, {})

    def test_view(self):
        water = M.water_geometry()
        v = molview.View.fit(["O", "H", "H"], [water])
        self.assertEqual(v.tilt, 0.0)                                       # a planar molecule: its plane is the view
        xy, depth = v.project(water)
        self.assertLess(np.abs(depth).max(), 1e-9)
        np.testing.assert_allclose(np.linalg.norm(xy[1] - xy[0]), 0.9584, atol=1e-9)        # true lengths in the plane
        np.testing.assert_allclose(v.axes @ v.axes.T, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(float(np.linalg.det(v.axes)), 1.0, places=9)                  # not a mirror image
        # the inversion of ammonia moves the atoms out of the best-fit plane: the view is tilted
        up, down = M.ammonia_geometry(up=True), M.ammonia_geometry(up=False)
        frames = M.interpolate_path(up, down, 7, align=False)
        v = molview.View.fit(["N", "H", "H", "H"], frames)
        self.assertEqual(v.tilt, 60.0)
        self.assertIn("tilted 60", v.describe())
        np.testing.assert_allclose(v.axes @ v.axes.T, np.eye(3), atol=1e-12)
        heights = [v.project(f)[0][1, 1] - v.project(f)[0][0, 1] for f in frames]     # screen height of H2 above N1
        self.assertGreater(abs(heights[0] - heights[-1]), 0.4)               # the inversion is visible
        for f in frames:                                                     # no atom hides another one
            xy = v.project(f)[0]
            d = [np.linalg.norm(xy[i] - xy[j]) for i in range(4) for j in range(i)]
            self.assertGreater(min(d), 0.35)
        self.assertEqual(molview.atom_labels(["N", "H", "H"]), ["N1", "H2", "H3"])
        self.assertEqual(molview.union_bonds(["N", "H", "H", "H"], frames), [(0, 1), (0, 2), (0, 3)])

    def test_text_check_knows_a_logarithmic_axis(self):
        """A tick far below the lower limit of a log axis has a label object in some matplotlib versions. It is
        not drawn, so it must not be reported."""
        with plots.style():
            fig = plots.new_figure(plots.NARROW, 2.0)
            ax = fig.add_subplot(1, 1, 1)
            ax.plot([1, 2, 3], [1.0, 1e-3, 1e-9], color=plots.PALETTE["dmc"])
            ax.set_yscale("log")
            ax.set_yticks([1e-13, 1e-9, 1e-5, 1e-1])
            ax.set_ylim(3e-11, 3.0)                                           # after this the first tick is outside the limits
            fig.canvas.draw()
            texts = [t.get_text() for t in plots._axis_texts(ax)]
            self.assertFalse(any("13" in t for t in texts), texts)
            self.assertTrue(any("-9" in t or "9" in t for t in texts), texts)
            self.assertEqual(plots.text_problems(fig), [])

    def test_text_check_finds_a_collision(self):
        with plots.style():
            fig = plots.new_figure(plots.NARROW, 2.0)
            ax = fig.add_subplot(1, 1, 1)
            ax.plot([0, 1], [0, 1], color=plots.PALETTE["dmc"])
            ax.set_xlabel("displacement")
            fig.canvas.draw()
            self.assertEqual(plots.text_problems(fig), [])
            ax.text(0.5, 0.5, "first label", ha="center")
            ax.text(0.5, 0.5, "second label", ha="center")
            ax.text(1.2, 0.5, "a text that reaches far over the right edge of the figure")
            drawing_warnings = []
            with plots.recorded_warnings(drawing_warnings):                  # the layout engine may warn about this figure
                fig.canvas.draw()
                found = plots.text_problems(fig)
        self.assertTrue(any("cover each other" in p for p in found), found)
        self.assertTrue(any("over the edge" in p for p in found), found)
        with self.assertRaises(ValueError):
            plots.new_figure(4.0, 3.0)                                       # not one of the named widths


class TestPathGeometryMap(unittest.TestCase):
    """The bonds and angles along a path, and the map ``plots/path_geometry_map``. The map is drawn only when a
    molecule has more than 4 different bond curves or more than 4 different angle curves (more than the lines of
    ``plots/path_geometry``). Ammonia has 1 of each, so the map itself is drawn for a made-up molecule."""

    def test_ammonia_path_needs_no_map(self):
        frames = G.read_xyz_frames(NH3_PATH)
        data = path_report_data(frames[0][0], [x for _s, x, _c in frames], [tables.comment_energy(c) for _s, _x, c in frames],
                                "ammonia inversion")
        internal = tables.path_internal(data.path)
        self.assertEqual(internal["names"], ["N1-H2", "N1-H3", "N1-H4", "H2-N1-H3", "H2-N1-H4", "H3-N1-H4"])
        for kind, label, members in (("bond", "N1-H2 and 2 equal bonds", [0, 1, 2]),
                                     ("angle", "H2-N1-H3 and 2 equal angles", [3, 4, 5])):
            groups, n_groups = tables.coordinate_groups(internal, kind, 10 ** 6)
            self.assertEqual((n_groups, groups[0]["label"]), (1, label))     # the 3 bonds (angles) are equal at every image
            self.assertEqual(groups[0]["extreme_image"], 3)                   # the largest change is at the planar saddle
            rows, small, cut = plots._map_rows(internal, kind)
            self.assertEqual((len(rows), rows[0]["members"], small, cut), (1, members, 0, 0))
            self.assertAlmostEqual(rows[0]["extreme"], float(rows[0]["delta"][3]), places=12)
        with self.assertRaises(tables.Skip) as cm:
            plots.plot_path_geometry_map(data)
        self.assertIn("every bond and angle of this molecule is a line in plots/path_geometry", str(cm.exception))
        changes = tables.path_changes(data.path)
        self.assertEqual((changes["bond"]["count"], changes["angle"]["count"]), (3, 3))
        self.assertEqual((changes["bond"]["below_floor"], changes["angle"]["below_floor"]), (0, 0))
        bond, angle = changes["bond"]["largest"][0], changes["angle"]["largest"][0]
        self.assertEqual((bond["image_of_largest_change"], angle["image_of_largest_change"]), (3, 3))
        X0, X3 = data.path.frames[0], data.path.frames[3]
        self.assertAlmostEqual(bond["largest_change"], G.distance(X3, 0, 1) - G.distance(X0, 0, 1), places=12)
        self.assertLess(bond["largest_change"], -0.01)                        # the N-H bond is shortest at the saddle
        self.assertAlmostEqual(angle["value_at_image_0"] + angle["largest_change"], 120.0, delta=0.01)   # planar there
        with plots.style(), tempfile.TemporaryDirectory() as tmp:
            files, problems = plots.save_figure(plots.plot_path_geometry(data), os.path.join(tmp, "lines"))
            self.assertEqual(problems, [])
            self.assertEqual(len(files), 2)

    def test_map_rows_of_made_up_coordinates(self):
        """Hand-made bond lengths (angstrom) and one angle (degrees) at 4 images of a path."""
        values = np.array([[1.10, 1.10, 1.40, 1.000, 1.50, 0.97, 105.0],
                           [1.12, 1.12, 1.38, 1.0005, 1.45, 0.98, 106.0],
                           [1.15, 1.15, 1.35, 1.001, 1.42, 0.99, 107.5],
                           [1.13, 1.13, 1.37, 1.000, 1.44, 0.985, 106.5]])
        internal = {"names": ["C1-C2", "C2-C3", "C3-C4", "C4-C5", "C5-C6", "C6-H7", "C1-C2-C3"],
                    "kinds": ["bond"] * 6 + ["angle"], "atoms": [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [0, 1, 2]],
                    "values": values}
        rows, small, cut = plots._map_rows(internal, "bond")
        # from the largest increase to the largest decrease; the two bonds with equal values are one row
        self.assertEqual([(g["name"], g["members"]) for g in rows],
                         [("C1-C2", [0, 1]), ("C6-H7", [5]), ("C3-C4", [2]), ("C5-C6", [4])])
        np.testing.assert_allclose([g["extreme"] for g in rows], [0.05, 0.02, -0.05, -0.08], atol=1e-12)
        self.assertEqual([g["extreme_image"] for g in rows], [2, 2, 2, 2])
        self.assertEqual((small, cut), (1, 0))                               # C4-C5 changes by 0.001 A: below the floor
        self.assertLess(0.001, tables.CHANGE_FLOOR["bond"])
        rows, small, cut = plots._map_rows(internal, "angle")                 # the angle has its own panel
        self.assertEqual(([g["name"] for g in rows], small, cut), (["C1-C2-C3"], 0, 0))
        self.assertAlmostEqual(rows[0]["extreme"], 2.5, places=12)
        # more rows than one panel holds: the bonds with the largest changes are kept, and the others are counted
        n = plots.MAP_ROWS + 6
        change = 0.004 * np.arange(1, n + 1) * np.where(np.arange(n) % 2 == 0, 1.0, -1.0)   # +0.004, -0.008, +0.012, ...
        internal = {"names": ["C%d-C%d" % (j + 1, j + 2) for j in range(n)], "kinds": ["bond"] * n,
                    "atoms": [[j, j + 1] for j in range(n)], "values": np.vstack([np.ones(n), 1.0 + change])}
        rows, small, cut = plots._map_rows(internal, "bond")
        self.assertEqual((len(rows), small, cut), (plots.MAP_ROWS, 0, 6))
        self.assertAlmostEqual(min(abs(g["extreme"]) for g in rows), 0.004 * 7, places=12)    # the 6 smallest have no row
        extremes = [g["extreme"] for g in rows]
        self.assertEqual(extremes, sorted(extremes, reverse=True))

    def test_map_of_a_made_up_molecule(self):
        """A made-up path of 5 images of a molecule C2H4 in which all 5 bonds and all 6 angles change differently."""
        start = np.array([[0.0, 0.0, 0.667], [0.0, 0.0, -0.667], [0.0, 0.923, 1.238], [0.0, -0.923, 1.238],
                          [0.0, 0.923, -1.238], [0.0, -0.923, -1.238]])
        step = np.array([[0.0, 0.0, 0.06], [0.0, 0.0, -0.04], [0.0, 0.05, 0.03], [0.02, -0.01, 0.0],
                         [0.0, 0.10, -0.02], [0.0, 0.0, 0.0]])
        data = path_report_data(["C", "C", "H", "H", "H", "H"], [start + s * step for s in np.linspace(0.0, 1.0, 5)],
                                [0.0, 0.004, 0.006, 0.003, 0.001], "made-up molecule")
        internal = tables.path_internal(data.path)
        self.assertEqual((internal["kinds"].count("bond"), internal["kinds"].count("angle")), (5, 6))
        self.assertEqual(tables.coordinate_groups(internal, "bond", 1)[1], 5)  # 5 different curves: more than 4 lines
        rows, small, cut = plots._map_rows(internal, "bond")
        self.assertEqual(([g["name"] for g in rows], small, cut), (["C1-C2", "C2-H5", "C1-H3", "C2-H6", "C1-H4"], 0, 0))
        self.assertAlmostEqual(rows[0]["extreme"], 0.10, places=9)            # the C-C bond: 0.06 + 0.04 A longer
        with plots.style(), tempfile.TemporaryDirectory() as tmp:
            fig = plots.plot_path_geometry_map(data)
            self.assertAlmostEqual(fig.get_size_inches()[0], plots.WIDE, delta=0.01)
            files, problems = plots.save_figure(fig, os.path.join(tmp, "map"))
            self.assertEqual(problems, [])
            self.assertEqual(len(files), 2)


if __name__ == "__main__":
    unittest.main()
