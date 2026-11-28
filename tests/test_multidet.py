"""Trial wave functions of several determinants (module engine/multidet.py), tested without a cluster.

The small text files ``tests/data/qp2_h2o_*`` and ``tests/data/md_h2o_*`` are real outputs of Quantum
Package 2 and of convert4qmc for one water molecule (one run on 2026-11-24: 24,383 determinants in
the CIPSI wave function, 55 after the cut at 1e-3). One value was changed: the network address of the
task server in ``qp2_h2o_truncate.log`` is a documentation address. The Quantum Package jobs themselves
are replaced by files that a test writes, and the batch system by the fake scheduler of
``test_qmcpack_engine``.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
sys.path.insert(0, HERE)

import test_qmcpack_engine as TE  # noqa: E402
from montepass.engine import multidet as MD  # noqa: E402
from montepass.engine import qmcpack as E  # noqa: E402
from montepass.engine import qmcpack_inputs as Q  # noqa: E402
from montepass.engine.base import DONE, FAILED, QUEUED, PointSpec  # noqa: E402
from montepass.machine import Machine  # noqa: E402
from montepass.scheduler import base as S  # noqa: E402

DATA = os.path.join(HERE, "data")
E_SCF = -16.9432220412203            # RHF energy of Quantum Package for the water molecule of the test files
E_CUT = -16.980443615992346          # energy that Quantum Package printed for the 55 determinants

QP2_PROFILE = """
[setup.qp2]
lines =
    set +u
    module load qp2

[job.qp2]
nodes = 1
tasks = 1
cpus_per_task = 8
mem = 16G
time = 01:00:00
"""


def read(path):
    with open(path) as fh:
        return fh.read()


def data(name):
    return read(os.path.join(DATA, name))


def trial(**kw):
    return MD.trial_of({"trial": dict({"kind": "multidet", "cutoff": 1e-3, "n_det_max": 20000}, **kw)})


def vmc_trace(path, mean, n=140, noise=0.004, seed=1):
    """A VMC trace whose block energies scatter around ``mean``."""
    r = np.random.default_rng(seed)
    lines = ["#   index    LocalEnergy    LocalEnergy_sq    BlockWeight"]
    for i, e in enumerate(mean + noise * r.normal(size=n)):
        lines.append("%10d  %.10e  %.10e  %.10e" % (i, e, e * e + 3.0, 40960.0))
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


# ===================================================================================== settings
class TestSettings(unittest.TestCase):
    def test_no_setting_and_single_mean_the_default_chain(self):
        self.assertIsNone(MD.trial_of(None))
        self.assertIsNone(MD.trial_of({}))
        self.assertIsNone(MD.trial_of({"trial": None}))
        self.assertIsNone(MD.trial_of({"trial": {"kind": "single", "cutoff": 1e-3}}))
        MD.validate_trial(None)

    def test_trial_settings_without_a_kind_are_refused(self):
        with self.assertRaises(ValueError):                           # the point would run with one determinant, silently
            MD.trial_of({"trial": {"cutoff": 1e-3, "n_det_max": 100}})

    def test_defaults_are_filled_in(self):
        t = trial(time_limits={"cipsi": 600})
        self.assertEqual(t["qp2_basis"], "cc-pvtz_ecp_ccecp")
        self.assertEqual(t["time_limits"]["cipsi"], 600)
        self.assertEqual(t["time_limits"]["qpexport"], 1800)       # the other limits keep their defaults
        MD.validate_trial(t)
        self.assertEqual(MD.trial_of({"trial": {"kind": "multidet", "cutoff": " HF "}})["cutoff"], "hf")
        self.assertEqual(MD.trial_of({"trial": {"kind": "multidet", "cutoff": "2.5e-4", "n_det_max": 9}})["cutoff"], 2.5e-4)

    def test_labels(self):
        self.assertEqual([MD.cut_label(c) for c in (1e-3, 2.5e-4, 1e-5, "hf", 7.5e-4)],
                         ["c1e-03", "c2.5e-04", "c1e-05", "hf", "c7.5e-04"])

    def test_wrong_settings_are_refused(self):
        for bad in ({"cutoff": 0.0}, {"cutoff": 1.5}, {"cutoff": "small"}, {"n_det_max": 1}, {"n_det_max": None},
                    {"n_det_max": 10.5}, {"orbitals": "scf"}, {"kind": "casscf"}, {"colour": 1}, {"check_sigma_max": 0},
                    {"check_pilot_blocks": 30}, {"check_fail_sigmas": 2.0}, {"min_parent_ratio": 0.5},
                    {"time_limits": {"dmc": 5}}, {"qp2_basis": "cc pvtz; rm"}, {"max_segments": 0}):
            with self.assertRaises(ValueError, msg=str(bad)):
                MD.validate_trial(trial(**bad))
        MD.validate_trial(trial(cutoff="hf", n_det_max=None))       # one determinant needs no CIPSI size
        with self.assertRaises(ValueError):
            MD.validate_trial(trial(), charge=1)
        with self.assertRaises(ValueError):
            MD.validate_trial(trial(), spin=2)

    def test_the_engine_refuses_wrong_trial_settings(self):
        s = E.merge_settings(E.DEFAULT_SETTINGS, {"trial": {"kind": "multidet", "cutoff": 2.0, "n_det_max": 100}})
        with self.assertRaises(ValueError):
            E.validate_settings(s)
        E.validate_settings(E.merge_settings(E.DEFAULT_SETTINGS, {"trial": {"kind": "single"}}))
        self.assertNotIn("trial", E.DEFAULT_SETTINGS)               # the record of a default point stays as it was
        self.assertNotIn("determinantset", E.OPT_KEYS + E.DMC_KEYS)


# ===================================================================================== outputs of Quantum Package
class TestReaders(unittest.TestCase):
    def test_scf_energy(self):
        self.assertAlmostEqual(MD.parse_scf_log(data("qp2_h2o_scf.log")), -16.94322204121373, places=12)
        with self.assertRaises(ValueError):
            MD.parse_scf_log("no energy here\n")

    def test_cipsi_summaries(self):
        found = MD.parse_cipsi_log(data("qp2_h2o_fci_two_iterations.log"))
        self.assertEqual([s["n_det"] for s in found], [1641, 3280])
        last = found[-1]
        self.assertAlmostEqual(last["e_var"], -17.164589337131737, places=12)
        self.assertAlmostEqual(last["pt2"], -5.3480246506501898e-2, places=14)
        self.assertAlmostEqual(last["pt2_error"], 7.7816158601717874e-5, places=14)
        self.assertAlmostEqual(last["e_pt2"], -17.218069583638240, places=12)
        self.assertAlmostEqual(last["e_var"] + last["pt2"], last["e_pt2"], places=10)
        self.assertEqual(MD.parse_cipsi_log("nothing\n"), [])

    def test_truncation_takes_the_result_line_not_the_start_line(self):
        text = data("qp2_h2o_truncate.log")
        self.assertIn("* Number of determinants                                     24383", text)   # the parent
        n, e = MD.parse_truncate_log(text)
        self.assertEqual(MD.truncate_parent_size(text), 24383)         # the size of the wave function that was read
        self.assertIsNone(MD.truncate_parent_size("no such line\n"))
        self.assertEqual(n, 55)
        self.assertAlmostEqual(e, E_CUT, places=13)
        with self.assertRaises(ValueError):
            MD.parse_truncate_log("* Number of determinants    24383\n")

    def test_print_energy(self):
        self.assertAlmostEqual(MD.parse_print_energy_log(data("qp2_h2o_print_energy.log")), -16.980443615992332, places=13)
        with self.assertRaises(ValueError):
            MD.parse_print_energy_log(" E( 1 ) = -1.0\n")


# ===================================================================================== files of convert4qmc
class TestDeterminantSet(unittest.TestCase):
    def test_several_determinants(self):
        block, facts = MD.read_determinantset(os.path.join(DATA, "md_h2o_c4q.wfj.xml"), "trial.h5")
        self.assertEqual(facts, {"kind": "multidet", "ndet": 55, "nstates": 57})
        self.assertNotIn("jastrow", block.lower())
        self.assertNotIn("QP2QMCPACK.h5", block)
        self.assertEqual(block.count('href="trial.h5"'), 2)          # the orbital file and the list of determinants
        self.assertIn('<multideterminant optimize="no" spo_up="spo-up" spo_dn="spo-dn">', block)
        self.assertTrue(block.startswith("         <determinantset "))
        self.assertTrue(block.rstrip().endswith("</determinantset>"))

    def test_one_determinant(self):
        block, facts = MD.read_determinantset(os.path.join(DATA, "md_h2o_c4q_1det.wfj.xml"), "trial.h5")
        self.assertEqual((facts["kind"], facts["ndet"]), ("single", 1))
        self.assertIn("<slaterdeterminant>", block)

    def test_the_block_fits_into_the_inputs_of_the_package(self):
        import xml.etree.ElementTree as ET

        block, _facts = MD.read_determinantset(os.path.join(DATA, "md_h2o_c4q.wfj.xml"), "trial.h5")
        system = {"symbols": ["O", "H", "H"], "coords_bohr": [[0, 0, 0], [0, 1.44, -1.11], [0, -1.44, -1.11]],
                  "zeff": [6, 1, 1], "nup": 4, "ndn": 4, "nao": 58, "restricted": True}
        coeff = {"eO": [0.1] * 6, "eH": [0.2] * 6, "uu": [0.3] * 8, "ud": [0.4] * 8}
        for text in (Q.opt_input(system, determinantset=block), Q.dmc_input(system, coeff, determinantset=block),
                     MD.md_vmc_input(system, block, seed=3, blocks=70)):
            root = ET.fromstring(text)
            wf = root.find(".//wavefunction")
            self.assertEqual(len(wf.findall("determinantset")), 1)
            self.assertEqual(wf.find(".//detlist").get("size"), "55")
            self.assertIsNone(wf.find(".//slaterdeterminant"))
            self.assertNotIn("scf.h5", text)
        self.assertIsNone(ET.fromstring(MD.md_vmc_input(system, block)).find(".//jastrow"))      # NO Jastrow element
        self.assertEqual(len(ET.fromstring(Q.dmc_input(system, coeff, determinantset=block)).findall(".//jastrow")), 2)
        with self.assertRaises(ValueError):
            Q.opt_input(system, determinantset="<sposet/>")

    def write(self, text):
        import tempfile
        tmp = tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False)
        self.addCleanup(os.remove, tmp.name)
        tmp.write(text); tmp.close()
        return tmp.name

    def test_unusable_files_are_refused(self):
        good = data("md_h2o_c4q.wfj.xml")
        for bad in (good.replace('optimize="no"', 'optimize="yes"'),            # coefficients would be optimized
                    good.replace('cutoff="1e-20"', 'cutoff="1e-3"'),            # QMCPACK would drop determinants
                    good.replace('type="DETS"', 'type="CSF"'),
                    good.replace("<determinantset", "<other").replace("</determinantset>", "</other>"),
                    good.replace("</wavefunction>", "</wavefunction><wavefunction></wavefunction>")):
            with self.assertRaises(ValueError):
                MD.read_determinantset(self.write(bad), "trial.h5")

    def test_orbital_sets_outside_the_determinant_set_are_kept(self):
        """A QMCPACK version that writes the orbital sets as their own element: everything that is no Jastrow
        element belongs to the block."""
        text = ('<qmcsystem><wavefunction name="psi0" target="e">\n'
                '<sposet_collection type="MolecularOrbital" href="x.h5"><sposet name="spo-up" size="5"/></sposet_collection>\n'
                '<determinantset><multideterminant optimize="no" spo_up="spo-up" spo_dn="spo-up">'
                '<detlist size="7" type="DETS" nstates="5" cutoff="1e-20" href="x.h5"/></multideterminant></determinantset>\n'
                '<jastrow name="J2" type="Two-Body"><correlation/></jastrow></wavefunction></qmcsystem>')
        block, facts = MD.read_determinantset(self.write(text), "trial.h5")
        self.assertEqual(facts["ndet"], 7)
        self.assertIn("<sposet_collection", block)
        self.assertEqual(block.count('href="trial.h5"'), 2)
        self.assertNotIn("jastrow", block)

    def test_structure_file(self):
        st = MD.read_structure(os.path.join(DATA, "md_h2o_c4q.structure.xml"))
        self.assertEqual(st["symbols"], ["O", "H", "H"])
        self.assertEqual(st["charges"], [6.0, 1.0, 1.0])
        self.assertEqual((st["nup"], st["ndn"]), (4, 4))
        self.assertAlmostEqual(st["coords_bohr"][1][1], 1.4421002353, places=9)


class TestExportCheck(TE.EngineCase):
    def folder(self, ndet=55, e_rhf=E_SCF, symbols=("O", "H", "H"), wfj="md_h2o_c4q.wfj.xml"):
        os.makedirs(self.root, exist_ok=True)
        st = MD.read_structure(os.path.join(DATA, "md_h2o_c4q.structure.xml"))
        system = {"symbols": list(symbols), "coords_bohr": st["coords_bohr"], "zeff": [6, 1, 1], "nup": 4, "ndn": 4,
                  "nao": 58, "e_rhf": e_rhf}
        for name, text in (("system.json", json.dumps(system)), ("trial_facts.json", json.dumps({"n_det": ndet, "e_scf": E_SCF})),
                           ("mdc4q.wfj.xml", data(wfj)), ("mdc4q.structure.xml", data("md_h2o_c4q.structure.xml"))):
            with open(os.path.join(self.root, name), "w") as fh:
                fh.write(text)
        return self.root

    def test_pass(self):
        res = MD.check_export(self.folder())
        self.assertTrue(res["ok"], res["message"])
        self.assertEqual(res["ndet"], 55)
        self.assertLess(res["max_shift_bohr"], 1e-9)
        self.assertTrue(os.path.exists(os.path.join(self.root, "md_determinantset.xml")))

    def test_the_engine_reads_without_writing(self):
        self.assertTrue(MD.check_export(self.folder(), write=False)["ok"])
        self.assertFalse(os.path.exists(os.path.join(self.root, "md_determinantset.xml")))

    def test_a_lost_determinant_is_found(self):
        # what the option -multidets (with s) does: one determinant with no message
        res = MD.check_export(self.folder(wfj="md_h2o_c4q_1det.wfj.xml"))
        self.assertFalse(res["ok"])
        self.assertIn("holds 1 determinant(s), QP2 printed 55", res["message"])
        self.assertFalse(os.path.exists(os.path.join(self.root, "md_determinantset.xml")))

    def test_another_atom_order_is_found(self):
        res = MD.check_export(self.folder(symbols=("H", "O", "H")))
        self.assertFalse(res["ok"])
        self.assertIn("are not the atoms of system.json", res["message"])

    def test_another_molecule_or_basis_is_found_by_the_rhf_energy(self):
        res = MD.check_export(self.folder(e_rhf=E_SCF + 3e-4))
        self.assertFalse(res["ok"])
        self.assertIn("do not describe the same molecule, basis or pseudopotentials", res["message"])
        res = MD.check_export(self.folder(e_rhf=E_SCF + 5e-9))            # the difference measured for water
        self.assertTrue(res["ok"])
        self.assertAlmostEqual(res["rhf_difference"], -5e-9, places=12)
        self.assertTrue(MD.check_export(self.folder(e_rhf=E_SCF + 3e-4), rhf_tolerance=1e-3)["ok"])
        self.assertTrue(MD.check_export(self.folder(e_rhf=None))["ok"])          # no RHF energy: the test is skipped


# ===================================================================================== the VMC check
class TestVmcCheck(TE.EngineCase):
    def setUp(self):
        super().setUp()
        os.makedirs(self.root)
        with open(os.path.join(self.root, "trial_facts.json"), "w") as fh:
            json.dump({"n_det": 55, "e_ref": E_CUT}, fh)

    def run_files(self, r, mean, text=None, **kw):
        vmc_trace(os.path.join(self.root, "mdvmc%d.s000.scalar.dat" % r), mean, seed=r, **kw)
        with open(os.path.join(self.root, "mdvmc%d.out" % r), "w") as fh:
            fh.write("Found 55 terms in the MSD expansion.\n" if text is None else text)

    def test_pass(self):
        self.run_files(1, E_CUT)
        res = MD.check_md_vmc(self.root)
        self.assertEqual((res["ok"], res["verdict"]), (True, "PASS"), res["message"])
        self.assertLess(res["error"], 6e-4)
        self.assertEqual(res["runs"][0]["blocks"], 120)             # 140 blocks, the first 20 are not used

    def test_the_known_export_error_fails_at_once(self):
        self.run_files(1, E_CUT + 0.007)                            # 7 mHa too high
        res = MD.check_md_vmc(self.root)
        self.assertEqual((res["ok"], res["verdict"]), (False, "FAIL"))
        self.assertIn("does not describe the wave function of QP2", res["message"])

    def test_a_wide_error_bar_is_no_pass(self):
        self.run_files(1, E_CUT, noise=0.02)                        # error bar about 1.8 mHa
        res = MD.check_md_vmc(self.root)
        self.assertEqual((res["ok"], res["verdict"]), (False, "MORE"))
        self.assertIn("is above the largest accepted one", res["message"])
        self.run_files(2, E_CUT, noise=0.02, n=3000)                # a long second run brings the error down
        res = MD.check_md_vmc(self.root)
        self.assertEqual(res["verdict"], "PASS", res["message"])
        self.assertEqual(len(res["runs"]), 2)

    def test_no_pass_after_the_last_run(self):
        for r in (1, 2):
            self.run_files(r, E_CUT, noise=0.02)
        self.assertEqual(MD.check_md_vmc(self.root, max_runs=2)["verdict"], "FAIL")
        self.assertEqual(MD.check_md_vmc(self.root, max_runs=3)["verdict"], "MORE")

    def test_qmcpack_must_read_all_determinants(self):
        self.run_files(1, E_CUT, text="Found 12 terms in the MSD expansion.\n")
        res = MD.check_md_vmc(self.root)
        self.assertEqual(res["verdict"], "FAIL")
        self.assertIn("does not say 'Found 55 terms in the MSD expansion'", res["message"])

    def test_runs_are_combined_with_their_weights(self):
        mean, err = MD.combine_runs([(-1.0, 0.1), (-2.0, 0.1)])
        self.assertAlmostEqual(mean, -1.5)
        self.assertAlmostEqual(err, 0.1 / np.sqrt(2.0))
        mean, err = MD.combine_runs([(-1.0, 0.1), (-2.0, 1e9)])
        self.assertAlmostEqual(mean, -1.0, places=6)

    def test_length_of_the_next_run(self):
        # first run: 40 used blocks, error 3 mHa. Wanted: 0.6 / 1.3 mHa.
        blocks = MD.plan_check_blocks(3e-3, 40, 3e-3, 6e-4, 100000)
        target = 6e-4 / 1.3
        combined = 1.0 / np.sqrt(1.0 / 3e-3 ** 2 + (blocks - MD.CHECK_DROP) / (3e-3 ** 2 * 40))
        self.assertLessEqual(combined, target * 1.0001)
        self.assertGreater(1.0 / np.sqrt(1.0 / 3e-3 ** 2 + (blocks - MD.CHECK_DROP - 2) / (3e-3 ** 2 * 40)), target)
        self.assertEqual(MD.plan_check_blocks(3e-3, 40, 3e-3, 6e-4, 500), 500)          # never above the largest run
        self.assertEqual(MD.plan_check_blocks(1e-4, 40, 1e-4, 6e-4, 500), 40 + MD.CHECK_DROP)   # small enough already

    def test_a_small_error_bar_with_an_undecided_difference_doubles_the_run(self):
        # error bar 0.2 mHa (below the target), difference undecided: the next run is as long as all runs so far
        self.assertEqual(MD.plan_check_blocks(2e-4, 40, 3e-3, 6e-4, 100000, blocks_total=3000), 3000 + MD.CHECK_DROP)
        self.assertEqual(MD.plan_check_blocks(2e-4, 40, 3e-3, 6e-4, 100000), 40 + MD.CHECK_DROP)

    def test_a_block_that_is_not_a_number_fails_the_check(self):
        self.run_files(1, E_CUT)
        path = os.path.join(self.root, "mdvmc1.s000.scalar.dat")
        lines = read(path).split("\n")
        lines[60] = lines[60].replace(lines[60].split()[1], "nan")
        with open(path, "w") as fh:
            fh.write("\n".join(lines))
        res = MD.check_md_vmc(self.root)
        self.assertEqual(res["verdict"], "FAIL")
        self.assertIn("not a finite number", res["message"])

    def test_number_of_runs_that_do_not_share_random_numbers(self):
        self.assertEqual([MD.check_run_limit(4, r) for r in (8, 32, 64, 128, 256, 512, 1024)], [4, 4, 4, 4, 2, 1, 1])

    def test_seeds_of_the_runs_do_not_overlap(self):
        seeds = [MD.check_seed(10, r, 32) for r in (1, 2, 3)]
        self.assertEqual(seeds, [10, 970, 906])
        windows = [E.seed_window(s, 32) for s in seeds]
        self.assertFalse(E.windows_overlap(windows[0], windows[1]) or E.windows_overlap(windows[1], windows[2]))
        self.assertIsNone(MD.check_seed(None, 2, 32))


# ===================================================================================== the job scripts of Quantum Package
class TestScripts(unittest.TestCase):
    def bash_accepts(self, text):
        res = subprocess.run(["bash", "-n"], input="#!/bin/bash\n" + text, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, res.stderr)

    def test_parent_script(self):
        text = MD.parent_script(MD.PARENT_STEPS, trial(), threads=8, job_seconds=3600)
        self.bash_accepts(text)
        self.assertIn("qp_create_ezfio -q -b cc-pvtz_ecp_ccecp -p ccecp -o rhf.ezfio geom.xyz", text)
        self.assertIn("qp_set cipsi.ezfio determinants n_det_max 20000", text)
        self.assertIn("QP_THREADS=8", text)
        self.assertIn("export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1", text)
        # threads only on the qp_run line of the function qp_program
        lines = [ln for ln in text.splitlines() if "OMP_NUM_THREADS=$QP_USE_THREADS" in ln]
        self.assertEqual(len(lines), 1)
        self.assertIn("qp_run \"$@\"", lines[0])
        self.assertNotIn("OMP_NUM_THREADS=$QP_THREADS", text)
        # a stop request of an earlier run is removed before every QP2 program
        self.assertIn('rm -f "$last/work/qpstop" "$last/work/qpkill"', text)
        # the step is done only at the wanted size, or when QP2 stopped by its own rule
        self.assertIn('elif [ "$now" -lt 20000 ]; then', text)
        self.assertIn("is not below $pt2max", text)
        # a continued run gets its copy before QP2 starts to write; the copy is renewed by one function
        self.assertIn('keep_copy "$have" ||', text)
        self.assertIn('usable cipsi.ezfio && keep_copy "$now"; then last=$now; fi', text)
        self.assertLess(text.index('keep_copy "$have" ||'), text.index("fci cipsi.ezfio"))
        self.assertIn("manifest cipsi.ezfio > cipsi.manifest.tmp", text)
        self.assertIn("the diagonalization changed the number of determinants", text)
        for step in MD.PARENT_STEPS:
            self.assertIn("if begin_step %s; then" % step, text)
        self.assertIn("qp_set cipsi.ezfio determinants read_wf true", text)       # a later job continues
        self.assertIn("cipsi.bak.ezfio", text)
        self.assertIn("diagonalize_h cipsi.ezfio", text)
        self.assertLess(text.index("begin_step qpscf"), text.index("begin_step cipsi"))

    def test_parent_script_without_cipsi(self):
        text = MD.parent_script(MD.PARENT_STEPS[:2], trial(cutoff="hf", n_det_max=None), threads=4, job_seconds=600 + 900)
        self.bash_accepts(text)
        self.assertNotIn("begin_step cipsi", text)
        with self.assertRaises(ValueError):
            MD.parent_script(["scf"], trial(), 4, 3600)

    def test_cut_script(self):
        text = MD.cut_script(1e-3, trial(), threads=8, job_seconds=3600, python_bin="/opt/py/bin")
        self.bash_accepts(text)
        self.assertIn("qp_set t.ezfio qmcpack ci_threshold 0.001", text)
        self.assertIn("truncate_wf_spin t.ezfio", text)
        self.assertIn("print_energy pe.ezfio", text)
        self.assertIn("PATH=/opt/py/bin:$PATH", text)
        self.assertIn("save.log save_for_qmcpack x.ezfio", text)
        self.assertIn("copy_ezfio pre.ezfio export/x.ezfio", text)      # the export runs on a copy
        self.assertIn('"0/0/0 "*) ;;', text)                            # the only fingerprint that passes
        self.assertIn('"0/0/0 0"|"") fail_step', text)                  # an empty basis file is no pass
        self.assertIn("20210101", text)
        self.assertIn("same_manifest ../cipsi.ezfio ../cipsi.manifest", text)     # the stored parent is unchanged
        # save_for_qmcpack runs with 1 thread, and the thread number of the job comes back afterwards
        lines = text.splitlines()
        i = [k for k, ln in enumerate(lines) if "save.log save_for_qmcpack x.ezfio" in ln][0]
        self.assertEqual(lines[i - 1].strip(), "QP_USE_THREADS=1")
        self.assertIn("QP_USE_THREADS=$QP_THREADS", lines[i + 1])

    def test_cut_script_for_one_determinant(self):
        text = MD.cut_script("hf", trial(cutoff="hf", n_det_max=None), threads=8, job_seconds=3600)
        self.bash_accepts(text)
        self.assertNotIn("truncate_wf_spin", text)
        self.assertNotIn("read_wf", text.split("begin_step qpexport")[1])
        self.assertIn("copy_ezfio ../rhf.ezfio export/x.ezfio", text)

    def test_steps_of_the_point_job(self):
        limits = {"mdc4q": 900, "mdvmc": 7200}
        blocks = MD.qmc_steps_text(["mdc4q", "mdvmc"], limits, "mpirun -np 32",
                                   h5_source="/work/qp2/cut_c1e-03/export/QP2QMCPACK.h5", ndet=55)
        text = "\n".join(blocks)
        self.bash_accepts(text)
        self.assertIn("convert4qmc -orbitals trial.h5 -prefix mdc4q -multidet trial.h5", text)
        self.assertNotIn("-multidets", text)                           # the plural option gives 1 determinant with no message
        # the package makes the determinant strings unsigned itself, before convert4qmc reads them
        self.assertIn("md_package unsigned-dets", text)
        self.assertLess(text.index("md_package unsigned-dets"), text.index("-multidet trial.h5"))
        self.assertNotIn("determinants_tools", text)
        self.assertIn("cp /work/qp2/cut_c1e-03/export/QP2QMCPACK.h5 trial.h5", text)
        one = "\n".join(MD.qmc_steps_text(["mdc4q"], limits, "mpirun", h5_source="x.h5", ndet=1))
        self.assertNotIn("-multidet", one)
        self.assertNotIn("unsigned-dets", one)

    def test_unsigned_determinant_strings(self):
        # 200 orbitals: 4 integers of 64 bits per string. An electron in orbital 64, 128 or 192 sets the highest bit
        # of an integer, and QP2 writes that integer as a negative number
        top = -(1 << 63)
        signed = np.array([[7, 0, 0, 0], [7, top, 0, 0], [5, 0, top + 1, top], [-1, 3, 0, 0]], dtype=np.int64)
        unsigned = MD.unsigned_strings(signed)
        self.assertEqual(unsigned.dtype, np.uint64)
        self.assertEqual(unsigned.shape, signed.shape)
        self.assertEqual(unsigned.tobytes(), signed.tobytes())                    # no bit changes
        # the values are those of the two's complement of determinants_tools.py --transform: s + 2**64 for s < 0
        expected = [[int(s) + (1 << 64) if s < 0 else int(s) for s in row] for row in signed.tolist()]
        self.assertEqual([[int(v) for v in row] for row in unsigned], expected)
        # a file of the other byte order (h5py keeps the byte order of the file): the same values, native bytes
        other = signed.astype(signed.dtype.newbyteorder("S"))
        self.assertFalse(other.dtype.isnative)
        swapped = MD.unsigned_strings(other)
        self.assertTrue(swapped.dtype.isnative)
        self.assertEqual(swapped.dtype.kind, "u")
        self.assertEqual([[int(v) for v in row] for row in swapped], expected)
        self.assertEqual([[int(v) for v in row] for row in MD.unsigned_strings(unsigned.astype(">u8"))], expected)
        # a 1-D array (one word per determinant) is converted in the same way
        self.assertEqual([int(v) for v in MD.unsigned_strings(signed[:, 1])], [row[1] for row in expected])
        self.assertEqual(int(unsigned[1, 1]), 1 << 63)
        self.assertEqual(int(unsigned[3, 0]), (1 << 64) - 1)
        # unsigned strings stay as they are; other kinds are refused
        self.assertIs(MD.unsigned_strings(unsigned), unsigned)
        with self.assertRaises(ValueError):
            MD.unsigned_strings(np.array([[1.0]]))
        self.assertIn("unsigned-dets", MD._COMMANDS)

    def test_unsigned_determinants_of_an_h5_file(self):
        try:
            import h5py
        except ImportError:
            self.skipTest("h5py is not installed (the QMC jobs of several determinants need it)")
        top = -(1 << 63)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "trial.h5")
            alpha = np.array([[7, top], [11, 0]], dtype=np.int64)
            beta = np.array([[7, 0], [7, top + 5]], dtype=np.int64)
            with h5py.File(path, "w") as f:
                f.create_dataset("MultiDet/CI_Alpha", data=alpha)
                f.create_dataset("MultiDet/CI_Beta", data=beta)
                f.create_dataset("MultiDet/Coeff", data=np.array([0.9, -0.1]))
            self.assertEqual(MD.unsigned_determinants(path), {"MultiDet/CI_Alpha": 1, "MultiDet/CI_Beta": 1})
            with h5py.File(path, "r") as f:
                self.assertEqual(f["MultiDet/CI_Alpha"].dtype, np.uint64)
                self.assertEqual(f["MultiDet/CI_Alpha"][...].tobytes(), alpha.tobytes())
                self.assertEqual(f["MultiDet/CI_Beta"][...].tobytes(), beta.tobytes())
                self.assertEqual(list(f["MultiDet/Coeff"][...]), [0.9, -0.1])     # the rest of the file is kept
            self.assertEqual(MD.unsigned_determinants(path), {"MultiDet/CI_Alpha": 0, "MultiDet/CI_Beta": 0})
            # the group names of older exports
            old = os.path.join(tmp, "old.h5")
            with h5py.File(old, "w") as f:
                f.create_dataset("MultiDet/CI_0", data=alpha)
                f.create_dataset("MultiDet/CI_1", data=beta)
            self.assertEqual(MD.unsigned_determinants(old), {"MultiDet/CI_0": 1, "MultiDet/CI_1": 1})
            with h5py.File(os.path.join(tmp, "none.h5"), "w") as f:
                f.create_dataset("MultiDet/Coeff", data=np.array([1.0]))
            with self.assertRaises(ValueError):
                MD.unsigned_determinants(os.path.join(tmp, "none.h5"))


# ===================================================================================== the Quantum Package jobs of one geometry
class Qp2Case(TE.EngineCase):
    def setUp(self):
        super().setUp()
        with open(self.profile, "a") as fh:
            fh.write(QP2_PROFILE)
            fh.write("\n[executables2]\n")
        # the key determinants_tools is not read since version 0.2.1; it stays here on purpose: a profile of an earlier
        # version must still work (the job script then must not call the script of QMCPACK)
        text = read(self.profile).replace("convert4qmc = convert4qmc",
                                                 "convert4qmc = convert4qmc\ndeterminants_tools = /q/determinants_tools.py")
        with open(self.profile, "w") as fh:
            fh.write(text)
        self.folder = os.path.join(self.tmp, "qp2")
        self.symbols = ["O", "H", "H"]
        self.coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.7631265780, -0.5855676186], [0.0, -0.7631265780, -0.5855676186]])

    def run_(self, **kw):
        return MD.Qp2Run(self.folder, Machine(self.profile), self.sched, clock=lambda: self.now, poll_seconds=0,
                         settle_seconds=0, unknown_seconds=0, **kw)

    def put(self, name, text):
        path = os.path.join(self.folder, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            fh.write(text)

    def fake_scf(self):
        for name, value in (("ezfio_files/ezfio_convention", "20210101"), ("hartree_fock/energy", "  -0.169432220412203E+02"),
                            ("mo_basis/mo_num", "57"), ("ao_basis/ao_num", "64"), ("electrons/elec_alpha_num", "4"),
                            ("electrons/elec_beta_num", "4")):
            self.put("rhf.ezfio/" + name, value + "\n")
        self.put("scf.log", data("qp2_h2o_scf.log"))
        for step in ("qpcreate", "qpscf"):
            self.put(step + ".started", "1000\n"); self.put(step + ".done", "1010\n")

    def fake_cipsi(self, n_det=3280, done=True, segment=1, start=1, rc=0, seconds=300):
        self.put("cipsi.ezfio/determinants/n_det", "%d\n" % n_det)
        self.put("cipsi.%d.log" % segment, data("qp2_h2o_fci_two_iterations.log"))
        with open(os.path.join(self.folder, "cipsi.segments"), "a") as fh:
            fh.write("start %d 1100 %d\nend %d %d %d %d\n" % (segment, start, segment, 1100 + seconds, n_det, rc))
        self.put("cipsi.started", "1100\n")
        if done:
            self.put("cipsi.done", "1400\n")

    def fake_cut(self, label="c1e-03", n_det=55, e_print=None, fingerprint="0/0/0 64", h5=True, n_read=3280):
        d = "cut_%s/" % label
        if label != "hf":
            text = data("qp2_h2o_truncate.log")
            star = "* Number of determinants                                     24383"
            assert text.count(star) == 1
            self.put(d + "truncate.log", text.replace(star, star.replace("24383", "%5d" % n_read)))
            self.put(d + "pre.ezfio/determinants/n_det", "%d\n" % n_det)
            text = data("qp2_h2o_print_energy.log")
            if e_print is not None:
                text = text.replace("-16.980443615992332", "%.15f" % e_print)
            self.put(d + "print_energy.log", text)
        if h5:
            self.put(d + "export/QP2QMCPACK.h5", "h5\n")
        self.put(d + "export/fingerprint", fingerprint + "\n")
        for step in MD.CUT_STEPS:
            self.put(d + step + ".started", "1500\n"); self.put(d + step + ".done", "1510\n")

    def end(self, state=S.COMPLETED):
        self.sched.state_of[self.sched.submitted[-1]["id"]] = state


class TestQp2Run(Qp2Case):
    def test_from_the_geometry_to_the_export(self):
        run, t = self.run_(), trial()
        with self.assertRaises(ValueError):
            run.advance(t)                                             # the folder was not created
        run.ensure(self.symbols, self.coords, t)
        self.assertEqual(read(os.path.join(self.folder, "geom.xyz")).split("\n")[0], "3")
        state, message = run.advance(t)
        self.assertEqual(state, MD.RUNNING)
        job = self.sched.submitted[-1]
        self.assertEqual(job["script"], os.path.join(self.folder, "job_parent.sh"))
        self.assertIn("#SBATCH --cpus-per-task=8", job["text"])
        self.assertIn("#SBATCH --ntasks=1", job["text"])
        self.assertIn("QP_THREADS=8", job["text"])
        self.assertIn("JOB_SECONDS=3600", job["text"])
        self.assertIn("begin_step cipsi", job["text"])
        self.assertEqual(run.advance(t)[0], MD.RUNNING)                # the job waits in the queue
        self.assertEqual(len(self.sched.submitted), 1)
        self.fake_scf(); self.fake_cipsi(); self.end()
        state, message = run.advance(t)                                # the parent is done: the cut job starts
        self.assertEqual(state, MD.RUNNING)
        self.assertEqual(len(self.sched.submitted), 2)
        cut = self.sched.submitted[-1]
        self.assertEqual(cut["workdir"], os.path.join(self.folder, "cut_c1e-03"))
        self.assertIn("qp_set t.ezfio qmcpack ci_threshold 0.001", cut["text"])
        self.fake_cut(); self.end()
        state, message = run.advance(t)
        self.assertEqual(state, MD.READY, message)
        facts = run.facts(t)
        self.assertEqual((facts["n_det"], facts["n_det_parent"], facts["label"]), (55, 3280, "c1e-03"))
        self.assertEqual((facts["qp2_basis"], facts["qp2_pseudo"]), ("cc-pvtz_ecp_ccecp", "ccecp"))
        self.assertEqual(facts["qp2_version"], None)                   # the short test log holds no version line
        self.assertAlmostEqual(facts["e_ref"], E_CUT, places=12)
        self.assertAlmostEqual(facts["e_scf"], E_SCF, places=12)
        self.assertAlmostEqual(facts["e_pt2_parent"], -17.218069583638240, places=10)
        self.assertTrue(facts["h5"].endswith("cut_c1e-03/export/QP2QMCPACK.h5"))
        self.assertEqual(json.loads(read(os.path.join(self.folder, "cut_c1e-03", "facts.json")))["n_det"], 55)
        # a new process finds everything in the files and submits nothing
        self.assertEqual(self.run_().advance(t)[0], MD.READY)
        self.assertEqual(len(self.sched.submitted), 2)
        self.assertGreater(run.cost(), 0.0)

    def test_a_second_cutoff_needs_no_new_cipsi_run(self):
        run, t = self.run_(), trial()
        run.ensure(self.symbols, self.coords, t)
        run.advance(t); self.fake_scf(); self.fake_cipsi(); self.end()
        run.advance(t); self.fake_cut(); self.end()
        self.assertEqual(run.advance(t)[0], MD.READY)
        t2 = trial(cutoff=1e-4)
        run.ensure(self.symbols, self.coords, t2)
        self.assertEqual(run.advance(t2)[0], MD.RUNNING)
        self.assertEqual(len(self.sched.submitted), 3)
        self.assertEqual(self.sched.submitted[-1]["workdir"], os.path.join(self.folder, "cut_c1e-04"))
        self.assertNotIn("qp_run fci", self.sched.submitted[-1]["text"].replace("qp_program", ""))

    def test_one_determinant_needs_no_cipsi_and_a_later_cut_adds_it(self):
        run, t = self.run_(), trial(cutoff="hf", n_det_max=None)
        run.ensure(self.symbols, self.coords, t)
        run.advance(t)
        self.assertNotIn("begin_step cipsi", self.sched.submitted[-1]["text"])
        self.fake_scf(); self.end()
        run.advance(t)                                                 # the export of the RHF determinant
        self.assertIn("../rhf.ezfio", self.sched.submitted[-1]["text"])
        self.fake_cut("hf"); self.end()
        self.assertEqual(run.advance(t)[0], MD.READY)
        facts = run.facts(t)
        self.assertEqual(facts["n_det"], 1)
        self.assertAlmostEqual(facts["e_ref"], E_SCF, places=12)
        t2 = trial()                                                   # now a real cut on the same geometry
        run.ensure(self.symbols, self.coords, t2)
        self.assertEqual(run.advance(t2)[0], MD.RUNNING)
        self.assertIn("begin_step cipsi", self.sched.submitted[-1]["text"])
        self.assertEqual(self.sched.submitted[-1]["script"], os.path.join(self.folder, "job_parent.sh"))

    def test_two_requests_never_start_two_parent_jobs(self):
        """The cut ``hf`` and another cut share one parent. While the CIPSI job of the other cut runs, a request for
        ``hf`` finds the RHF step done. That request must not make the program forget the running job."""
        run, hf, cut = self.run_(), trial(cutoff="hf", n_det_max=None), trial()
        run.ensure(self.symbols, self.coords, hf)
        run.ensure(self.symbols, self.coords, cut)
        run.advance(hf)                                                # job 1: RHF only
        self.assertEqual(len(self.sched.submitted), 1)
        self.assertEqual(run.advance(cut)[0], MD.RUNNING)              # the cut waits for job 1
        self.assertEqual(len(self.sched.submitted), 1)
        self.fake_scf(); self.end()
        self.assertEqual(run.advance(cut)[0], MD.RUNNING)              # job 2: the CIPSI run
        self.assertEqual(len(self.sched.submitted), 2)
        self.assertIn("begin_step cipsi", self.sched.submitted[-1]["text"])
        cipsi_job = self.sched.submitted[-1]["id"]
        self.sched.state_of[cipsi_job] = S.RUNNING
        self.assertEqual(run.advance(hf)[0], MD.RUNNING)               # job 3: the export of the RHF determinant
        self.assertEqual(len(self.sched.submitted), 3)
        for _ in range(3):                                             # both requests ask again and again
            self.assertEqual(run.advance(cut)[0], MD.RUNNING)
            run.advance(hf)
        self.assertEqual(len(self.sched.submitted), 3)                 # no second CIPSI job
        self.fake_cut("hf"); self.sched.state_of[self.sched.submitted[-1]["id"]] = S.COMPLETED
        self.assertEqual(run.advance(hf)[0], MD.READY)
        self.assertEqual(run.advance(cut)[0], MD.RUNNING)
        self.assertEqual(len(self.sched.submitted), 3)
        self.fake_cipsi(); self.sched.state_of[cipsi_job] = S.COMPLETED
        self.assertEqual(run.advance(cut)[0], MD.RUNNING)              # job 4: the cut
        self.assertEqual(len(self.sched.submitted), 4)
        self.assertEqual(self.sched.submitted[-1]["workdir"], os.path.join(self.folder, "cut_c1e-03"))

    def test_cipsi_continues_after_a_time_limit(self):
        run, t = self.run_(), trial(max_segments=3)
        run.ensure(self.symbols, self.coords, t)
        run.advance(t); self.fake_scf()
        self.fake_cipsi(n_det=1641, done=False, segment=1, start=1, rc=124)
        self.put("cipsi.failed", "TIMEOUT: no end within 3000 s: qp_run fci cipsi.ezfio\n")
        self.end(S.FAILED)
        state, message = run.advance(t)
        self.assertEqual(state, MD.RUNNING)
        self.assertIn("the next job continues", message)
        self.assertEqual(len(self.sched.submitted), 2)
        self.assertFalse(os.path.exists(os.path.join(self.folder, "cipsi.failed")))
        self.assertTrue(os.path.exists(os.path.join(self.folder, "cipsi.failed.segment1")))
        # the batch system itself ends the second job; the wave function grew again
        self.fake_cipsi(n_det=3280, done=False, segment=2, start=1641, rc=124)
        self.end(S.TIMEOUT)
        self.assertEqual(run.advance(t)[0], MD.RUNNING)
        self.assertEqual(len(self.sched.submitted), 3)
        # the third job is the last one that max_segments allows
        self.fake_cipsi(n_det=6000, done=False, segment=3, start=3280, rc=124)
        self.end(S.TIMEOUT)
        state, message = run.advance(t)
        self.assertEqual(state, MD.FAILED)
        self.assertIn("reached the time limit in 3 jobs", message)
        self.assertEqual(len(self.sched.submitted), 3)

    def test_cipsi_that_does_not_grow_is_a_failure(self):
        run, t = self.run_(), trial()
        run.ensure(self.symbols, self.coords, t)
        run.advance(t); self.fake_scf()
        self.fake_cipsi(n_det=1641, done=False, segment=1, start=1641, rc=124, seconds=3000)
        self.end(S.TIMEOUT)
        state, message = run.advance(t)
        self.assertEqual(state, MD.FAILED)
        self.assertIn("did not grow in a whole job", message)
        self.assertEqual(len(self.sched.submitted), 1)
        # after a repair a person asks for new attempts
        self.assertEqual(run.retry(), ["parent"])
        self.assertEqual(run.advance(t)[0], MD.RUNNING)
        self.assertEqual(len(self.sched.submitted), 2)

    def test_a_run_failure_gets_a_second_attempt_and_a_quality_failure_none(self):
        run, t = self.run_(), trial()
        run.ensure(self.symbols, self.coords, t)
        run.advance(t)
        self.put("qpscf.failed", "RUN: the task server of QP2 did not start: qp_run scf rhf.ezfio\n")
        self.put("qpcreate.done", "1001\n")
        self.end(S.FAILED)
        self.assertEqual(run.advance(t)[0], MD.RUNNING)
        self.assertEqual(len(self.sched.submitted), 2)
        self.put("qpscf.failed", "RUN: exit code 1: qp_run scf rhf.ezfio\n")
        self.end(S.FAILED)
        state, message = run.advance(t)
        self.assertEqual(state, MD.FAILED)
        self.assertIn("attempt 2 of 2", message)
        run2 = MD.Qp2Run(os.path.join(self.tmp, "other"), Machine(self.profile), self.sched, clock=lambda: self.now,
                         poll_seconds=0, settle_seconds=0, unknown_seconds=0)
        run2.ensure(self.symbols, self.coords, t)
        run2.advance(t)
        with open(os.path.join(self.tmp, "other", "qpcreate.failed"), "w") as fh:
            fh.write("QUALITY: the EZFIO has the convention 20250211; save_for_qmcpack needs 20210101\n")
        self.end(S.FAILED)
        n = len(self.sched.submitted)
        state, message = run2.advance(t)
        self.assertEqual(state, MD.FAILED)
        self.assertIn("20250211", message)
        self.assertEqual(len(self.sched.submitted), n)                 # no second attempt for a wrong result

    def test_a_job_cancelled_by_hand_is_not_submitted_again(self):
        run, t = self.run_(), trial()
        run.ensure(self.symbols, self.coords, t)
        run.advance(t)
        self.end(S.CANCELLED)
        state, message = run.advance(t)
        self.assertEqual(state, MD.FAILED)
        self.assertIn("cancelled outside this program", message)

    def test_the_judgement_waits_for_the_settle_time(self):
        run, t = self.run_(), trial()
        run.settle_seconds = 60.0
        run.ensure(self.symbols, self.coords, t)
        run.advance(t)
        self.end(S.FAILED)
        self.assertEqual(run.advance(t)[0], MD.RUNNING)                # the files of the job may arrive late
        self.fake_scf(); self.fake_cipsi()
        self.now += 61
        self.assertEqual(run.advance(t)[0], MD.RUNNING)                # the parent is done after all: the cut job starts
        self.assertEqual(self.sched.submitted[-1]["workdir"], os.path.join(self.folder, "cut_c1e-03"))

    def cut_ready(self, t=None, **kw):
        run, t = self.run_(), t or trial()
        run.ensure(self.symbols, self.coords, t)
        run.advance(t); self.fake_scf(); self.fake_cipsi(); self.end()
        run.advance(t); self.fake_cut(**kw); self.end()
        return run.advance(t)

    def test_wrong_results_of_the_cut_are_refused(self):
        for kw, word in (({"fingerprint": "42/24/4 64"}, "factors of the known export error"),
                         ({"fingerprint": "0/0/0 0"}, "the test could not read it"),      # an empty basis file is no pass
                         ({"fingerprint": "0/0/0"}, "the test could not read it"),
                         ({"e_print": E_CUT + 1e-5}, "print_energy finds"),
                         ({"n_read": 24383}, "does not come from the recorded parent"),
                         ({"n_det": 54}, "printed 55 determinants, the stored wave function holds 54"),
                         ({"h5": False}, "wrote no file")):
            state, message = self.cut_ready(**kw)
            self.assertEqual(state, MD.FAILED, kw)
            self.assertIn(word, message)
            import shutil
            shutil.rmtree(self.folder)

    def test_a_parent_that_is_too_small_is_refused(self):
        state, message = self.cut_ready(t=trial(min_parent_ratio=100.0))
        self.assertEqual(state, MD.FAILED)
        self.assertIn("keeps 55 of the 3280 determinants", message)
        self.assertIn("raise n_det_max", message)

    def test_a_step_with_little_time_is_no_proof_that_cipsi_cannot_grow(self):
        run, t = self.run_(), trial()
        run.ensure(self.symbols, self.coords, t)
        run.advance(t); self.fake_scf()
        self.fake_cipsi(n_det=1641, done=False, segment=1, start=1641, rc=124, seconds=120)   # 2 of 60 minutes
        self.end(S.TIMEOUT)
        self.assertEqual(run.advance(t)[0], MD.RUNNING)
        self.assertEqual(len(self.sched.submitted), 2)

    def test_a_job_that_the_batch_system_cannot_see_but_that_is_alive_is_not_replaced(self):
        run, t = self.run_(), trial()
        run.unknown_seconds = 600.0
        run.ensure(self.symbols, self.coords, t)
        run.advance(t)
        job = self.sched.submitted[-1]["id"]
        del self.sched.state_of[job]                                   # squeue and sacct give no answer: UNKNOWN
        for minute in range(30):                                       # half an hour without an answer
            self.put(MD.HEARTBEAT, "%d %s\n" % (self.now, job))        # the job writes its sign of life
            self.assertEqual(run.advance(t)[0], MD.RUNNING)
            self.now += 60
        self.assertEqual(len(self.sched.submitted), 1)
        self.assertEqual(self.sched.cancelled, [])
        # the sign of life stops: after the waiting time the job is cancelled (in case it still exists) and replaced
        self.now += 601
        self.assertEqual(run.advance(t)[0], MD.RUNNING)                # first look: the waiting time starts
        self.now += 601
        self.assertEqual(run.advance(t)[0], MD.RUNNING)
        self.assertEqual(self.sched.cancelled, [job])
        self.assertEqual(len(self.sched.submitted), 2)
        self.assertEqual(run.rec["parent"]["attempts"], 1)

    def test_a_submission_that_fails_is_repeated_with_the_same_decision_and_counted_once(self):
        run, t = self.run_(), trial()
        run.ensure(self.symbols, self.coords, t)
        run.advance(t)
        self.put("qpcreate.failed", "RUN: exit code 1: qp_create_ezfio\n")
        self.end(S.FAILED)
        self.sched.refuse = "sbatch: error: the controller does not answer"
        for _ in range(3):
            with self.assertRaises(RuntimeError):
                run.advance(t)
        self.assertEqual(run.rec["parent"]["attempts"], 1)             # counted once, not three times
        self.assertEqual(run.rec["parent"]["next"]["kind"], "retry")
        self.assertEqual(len(self.sched.submitted), 1)
        # a new process (the driver was started again) finds the decision in the record
        self.sched.refuse = None
        again = self.run_()
        self.assertEqual(again.advance(t)[0], MD.RUNNING)
        self.assertEqual(len(self.sched.submitted), 2)
        self.assertEqual(again.rec["parent"]["attempts"], 1)
        self.assertIsNone(again.rec["parent"]["next"])
        self.assertEqual(again.advance(t)[0], MD.RUNNING)              # the new job waits in the queue
        self.assertEqual(len(self.sched.submitted), 2)

    def test_a_submission_whose_answer_was_lost_is_found_by_its_sign_of_life(self):
        run, t = self.run_(), trial()
        run.ensure(self.symbols, self.coords, t)
        run.advance(t)
        self.put("qpcreate.failed", "RUN: exit code 1: qp_create_ezfio\n")
        self.end(S.FAILED)
        self.sched.refuse = "sbatch: error: Socket timed out on send/recv operation"
        with self.assertRaises(RuntimeError):
            run.advance(t)                                             # the decision (second attempt) is in the record
        self.sched.refuse = None
        # the batch system had accepted that job all the same: it works in the folder and writes its sign of life
        self.put(MD.HEARTBEAT, "%d 77777\n" % self.now)
        self.sched.state_of["77777"] = S.RUNNING
        state, message = run.advance(t)
        self.assertEqual(state, MD.RUNNING)
        self.assertIn("77777", message)
        par = run.rec["parent"]
        self.assertEqual((par["jobs"][-1]["id"], par["jobs"][-1]["kind"]), ("77777", "adopted"))
        self.assertIsNone(par["next"])
        self.assertEqual(par["attempts"], 1)
        self.assertEqual(len(self.sched.submitted), 1)                 # no second job next to the one that works
        self.assertEqual(run.advance(t)[0], MD.RUNNING)                # the job is followed like every other one
        self.assertEqual(len(self.sched.submitted), 1)
        # a sign of life of a job that the record holds already is no unknown job
        self.assertIsNone(run._foreign_job(par, self.folder))
        self.put(MD.HEARTBEAT, "0 88888\n")                            # a job that ended wrote the time 0
        self.assertIsNone(run._foreign_job(par, self.folder))

    def test_a_job_that_ended_because_another_one_holds_the_folder_costs_no_attempt(self):
        run, t = self.run_(), trial()
        run.ensure(self.symbols, self.coords, t)
        run.advance(t)
        self.put(MD.HEARTBEAT, "%d 88888\n" % self.now)
        self.sched.state_of["88888"] = S.RUNNING
        self.end(S.FAILED)                                             # the job of the record ended with exit code 98
        state, _message = run.advance(t)
        self.assertEqual(state, MD.RUNNING)
        par = run.rec["parent"]
        self.assertEqual(par["jobs"][-1]["id"], "88888")
        self.assertTrue(par["jobs"][-2]["judged"])
        self.assertEqual((par["attempts"], len(self.sched.submitted)), (0, 1))

    def test_the_job_script_lets_only_one_job_work_in_a_folder(self):
        for text in (MD.parent_script(MD.PARENT_STEPS, trial(), 8, 3600), MD.cut_script(1e-3, trial(), 8, 3600),
                     MD.import_script("/a/x.ezfio", trial(), 8, 3600)):
            for word in ('ME=${SLURM_JOB_ID:-local$$}', 'while [ "$(beat_age)" -lt 180 ]; do', "exit 98", 'beat "$(date +%s)"',
                         "|| beat 0' EXIT"):
                self.assertIn(word, text)
            self.assertNotIn("HEARTBEAT_FRESH_VALUE", text)
            self.assertLess(text.rindex("exit 98"), text.index("if begin_step"))  # before any work in the folder

    def test_a_record_of_the_first_version_is_read(self):
        run, t = self.run_(), trial()
        run.ensure(self.symbols, self.coords, t)
        run.advance(t); self.fake_scf(); self.fake_cipsi(); self.end()
        run.advance(t); self.fake_cut(); self.end()
        self.assertEqual(run.advance(t)[0], MD.READY)
        rec = json.loads(read(run.record_file))
        del rec["settings"]
        for part in [rec["parent"]] + list(rec["cuts"].values()):
            del part["next"]
        del rec["cuts"]["c1e-03"]["cutoff"], rec["cuts"]["c1e-03"]["import_ezfio"]
        with open(run.record_file, "w") as fh:
            json.dump(rec, fh)
        old = self.run_()
        self.assertEqual(old.rec["settings"]["max_segments"], MD.DEFAULT_TRIAL["max_segments"])
        self.assertEqual(old.rec["cuts"]["c1e-03"]["cutoff"], 1e-3)
        self.assertEqual(old.advance(t)[0], MD.READY)
        self.assertEqual(old.facts(t)["n_det"], 55)

    def test_retry_cancels_a_job_that_can_still_run(self):
        run, t = self.run_(), trial()
        run.ensure(self.symbols, self.coords, t)
        run.advance(t)
        job = self.sched.submitted[-1]["id"]
        run._fail(run.rec["parent"], "set to failed by hand")          # for example by a wrong judgement
        self.sched.state_of[job] = S.RUNNING
        self.assertEqual(run.retry(), ["parent"])
        self.assertEqual(self.sched.cancelled, [job])
        self.assertEqual(run.advance(t)[0], MD.RUNNING)
        self.assertEqual(len(self.sched.submitted), 2)

    def test_cancel_keeps_the_jobs_that_another_point_needs_and_frees_the_folder(self):
        run, t = self.run_(), trial()
        run.ensure(self.symbols, self.coords, t)
        run.advance(t); self.fake_scf(); self.fake_cipsi(); self.end()
        run.advance(t)                                                 # the cut job of c1e-03 is in the queue
        t4 = trial(cutoff=1e-4)
        run.advance(t4)                                                # and the cut job of c1e-04
        jobs = {j["id"]: j["workdir"] for j in self.sched.submitted}
        cut3 = [i for i, w in jobs.items() if w.endswith("cut_c1e-03")][0]
        cut4 = [i for i, w in jobs.items() if w.endswith("cut_c1e-04")][0]
        self.assertEqual(run.cancel(keep_cuts=["c1e-04"]), [cut3])     # the point of c1e-04 still needs its job
        self.assertEqual(self.sched.cancelled, [cut3])
        cut = run.rec["cuts"]["c1e-03"]
        self.assertEqual((cut["status"], cut["jobs"][-1]["judged"]), ("WAITING", True))
        # a point that asks for the cut again gets a new job; the cancelled one is no failure
        self.assertEqual(run.advance(t)[0], MD.RUNNING)
        self.assertEqual(self.sched.submitted[-1]["workdir"], os.path.join(self.folder, "cut_c1e-03"))
        self.assertNotIn(cut4, self.sched.cancelled)

    def test_a_cancelled_parent_job_is_submitted_again_for_the_next_request(self):
        run, t = self.run_(), trial()
        run.ensure(self.symbols, self.coords, t)
        run.advance(t)
        job = self.sched.submitted[-1]["id"]
        self.sched.state_of[job] = S.RUNNING
        self.assertEqual(run.cancel(), [job])
        self.assertEqual(run.rec["parent"]["status"], "WAITING")
        state, message = run.advance(trial(cutoff=1e-4))               # a new cutoff uses the same folder
        self.assertEqual(state, MD.RUNNING, message)
        self.assertEqual(len(self.sched.submitted), 2)
        self.assertNotIn("cancelled outside this program", message)

    def test_a_finished_parent_job_is_judged_and_the_parent_is_done(self):
        run, t = self.run_(), trial()
        run.ensure(self.symbols, self.coords, t)
        run.advance(t); self.fake_scf(); self.fake_cipsi(); self.end()
        run.advance(t)
        par = run.rec["parent"]
        self.assertEqual((par["status"], par["jobs"][0]["judged"], par["message"]), ("DONE", True, ""))

    def test_cancel_ends_the_jobs_in_the_queue(self):
        run, t = self.run_(), trial()
        run.ensure(self.symbols, self.coords, t)
        run.advance(t)
        job = self.sched.submitted[-1]["id"]
        self.assertEqual(run.cancel(), [job])
        self.assertEqual(self.sched.cancelled, [job])

    def test_two_wave_functions_cannot_share_one_label(self):
        run, t = self.run_(), trial()
        run.ensure(self.symbols, self.coords, t)
        run.advance(t); self.fake_scf(); self.fake_cipsi(); self.end()
        run.advance(t)
        with self.assertRaises(ValueError):
            run._advance_cut("c1e-03", trial(cutoff=2e-3))
        self.assertNotEqual(MD.cut_label(1.23451e-4), MD.cut_label(1.23452e-4))

    def test_a_cut_must_lie_between_its_parent_and_the_rhf_energy(self):
        run, t = self.run_(), trial()
        run.ensure(self.symbols, self.coords, t)
        run.advance(t); self.fake_scf()
        # an RHF energy BELOW the energy of the cut: the cut cannot come from these orbitals
        self.put("rhf.ezfio/hartree_fock/energy", "  -0.171000000000000E+02\n")
        log = data("qp2_h2o_scf.log")
        self.assertEqual(log.count("-16.94322204121373"), 1)
        self.put("scf.log", log.replace("-16.94322204121373", "-17.10000000000000"))
        self.fake_cipsi(); self.end()
        self.assertEqual(run.advance(t)[0], MD.RUNNING)                # the parent is accepted, the cut job starts
        self.fake_cut(); self.end()
        state, message = run.advance(t)
        self.assertEqual(state, MD.FAILED)
        self.assertIn("is not between the energy of the CIPSI wave function", message)

    def test_a_folder_of_a_point_is_refused(self):
        os.makedirs(self.folder)
        self.put("point.json", "{}")
        with self.assertRaises(ValueError):
            self.run_().ensure(self.symbols, self.coords, trial())

    def test_an_existing_wave_function_is_imported(self):
        t = MD.trial_of({"trial": {"kind": "multidet", "import_ezfio": "/data/store/A_t1e-3/pre.ezfio", "label": "imp_c1e-03"}})
        MD.validate_trial(t)
        self.assertEqual(MD.cut_id(t), "imp_c1e-03")
        run = self.run_()
        run.ensure(self.symbols, self.coords, t)
        self.assertEqual(run.advance(t)[0], MD.RUNNING)
        job = self.sched.submitted[-1]
        self.assertEqual(job["workdir"], os.path.join(self.folder, "cut_imp_c1e-03"))     # no parent job at all
        for word in ("begin_step qpimport", "copy_ezfio /data/store/A_t1e-3/pre.ezfio pre.ezfio", "print_energy pe.ezfio",
                     "-ge 20250211", 'sed -i "s/$conv/20210101/"', "primitives_normalized", "prim_normalization_factor",
                     "if [ -e export/x.ezfio/determinants/psi_det.gz ]; then echo T > export/x.ezfio/determinants/read_wf; fi"):
            self.assertIn(word, job["text"])
        self.assertNotIn("truncate_wf_spin", job["text"])
        self.assertNotIn("qp_run fci", job["text"])
        res = subprocess.run(["bash", "-n", job["script"]], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, res.stderr)
        d = "cut_imp_c1e-03/"
        self.put(d + "pre.ezfio/hartree_fock/energy", "  -0.169432220412203E+02\n")
        self.put(d + "pre.ezfio/mo_basis/mo_num", "57\n")
        self.put(d + "pre.ezfio/determinants/n_det", "55\n")
        self.put(d + "pre.ezfio/determinants/psi_det.gz", "x")
        self.put(d + "print_energy.log", data("qp2_h2o_print_energy.log"))
        self.put(d + "export/QP2QMCPACK.h5", "h5\n")
        self.put(d + "export/fingerprint", "0/0/0 64\n")
        for step in MD.IMPORT_STEPS:
            self.put(d + step + ".done", "1510\n")
        self.end()
        state, message = run.advance(t)
        self.assertEqual(state, MD.READY, message)
        facts = run.facts(t)
        self.assertEqual((facts["n_det"], facts["label"], facts["imported_from"]), (55, "imp_c1e-03", "/data/store/A_t1e-3/pre.ezfio"))
        self.assertAlmostEqual(facts["e_ref"], -16.980443615992332, places=12)
        for bad in ({"import_ezfio": "relative/x.ezfio", "label": "a"}, {"import_ezfio": "/a b/x.ezfio", "label": "a"},
                    {"import_ezfio": "/a/x.ezfio"}, {"import_ezfio": "/a/x.ezfio", "label": "a b"},
                    {"import_ezfio": "/a/x.ezfio", "label": "a", "cutoff": 1e-3}):
            with self.assertRaises(ValueError, msg=str(bad)):
                MD.validate_trial(MD.trial_of({"trial": dict({"kind": "multidet"}, **bad)}))

    def test_another_geometry_or_another_size_needs_another_folder(self):
        run, t = self.run_(), trial()
        run.ensure(self.symbols, self.coords, t)
        run.ensure(self.symbols, self.coords, t)                       # the same request again is fine
        with self.assertRaises(ValueError):
            run.ensure(self.symbols, self.coords + 0.01, t)
        with self.assertRaises(ValueError):
            run.ensure(self.symbols, self.coords, trial(n_det_max=50000))
        with self.assertRaises(ValueError):
            run.ensure(self.symbols, self.coords, trial(qp2_basis="cc-pvdz_ecp_ccecp"))


# ===================================================================================== a point with several determinants
class TestEnginePoint(Qp2Case):
    SETTINGS = {"trial": {"kind": "multidet", "cutoff": 1e-3, "n_det_max": 20000}}

    def setUp(self):
        super().setUp()
        self.key = "trial/min/start"
        self.folder = os.path.join(self.root, "trial", "min", "start", "qp2")

    def point(self, eng):
        eng.submit(self.spec(self.key, blocks=200, role="final", settings=self.SETTINGS))
        return self.read_json(self.key, "point.json")

    def finish_qp2(self, eng):
        self.fake_scf(); self.fake_cipsi(); self.end()
        eng.advance()                                                  # the parent is done: the cut job starts
        self.fake_cut(); self.end()
        eng.advance()                                                  # the export is ready: the QMC job starts

    def test_the_point_waits_for_quantum_package(self):
        eng = self.engine()
        p = self.point(eng)
        self.assertEqual(p["steps"], ["scf", "mdc4q", "mdvmc", "opt", "dmc"])
        self.assertEqual(p["wait"], {"for": "qp2", "key": "trial/min/start/qp2"})
        self.assertEqual(p["trial"]["cutoff"], 1e-3)
        self.assertEqual(len(self.sched.submitted), 1)                 # only the Quantum Package job
        self.assertTrue(self.sched.submitted[0]["script"].endswith("qp2/job_parent.sh"))
        self.assertFalse(os.path.exists(self.path(self.key, "job.sh")))
        data_ = eng.read(self.key)
        self.assertEqual(data_.state, QUEUED)
        self.assertIn("waits for the Quantum Package calculation", data_.message)
        self.assertIn("e_rhf", self.read_text(self.key, "scf.py"))     # the comparison of the two RHF energies
        # the geometry of the Quantum Package run is the grouped geometry of the SCF
        self.assertEqual(read(os.path.join(self.folder, "geom.xyz")).split("\n")[2].split()[0], "H")
        self.assertEqual(sorted(eng.keys()), [self.key])               # Quantum Package jobs are no points

    def test_the_qmc_job_starts_when_the_export_is_ready(self):
        eng = self.engine()
        self.point(eng)
        self.finish_qp2(eng)
        self.assertEqual(len(self.sched.submitted), 3)
        job = self.sched.submitted[-1]
        self.assertTrue(job["script"].endswith("trial/min/start/job.sh"))
        text = job["text"]
        for word in ("begin_step scf", "begin_step mdc4q", "begin_step mdvmc", "begin_step opt", "begin_step dmc",
                     "md_package()", "-multidet trial.h5", "md_package unsigned-dets",
                     os.path.join(self.folder, "cut_c1e-03", "export", "QP2QMCPACK.h5")):
            self.assertIn(word, text)
        for word in ("begin_step c4q", "begin_step vmc0", "-orbitals scf.h5", "determinants_tools", "--transform"):
            self.assertNotIn(word, text)
        self.assertLess(text.index("begin_step mdvmc"), text.index("begin_step opt"))    # the check comes before the Jastrow
        res = subprocess.run(["bash", "-n", job["script"]], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, res.stderr)
        cfg = self.read_json(self.key, "qmc_settings.json")
        self.assertEqual(cfg["determinantset_file"], "md_determinantset.xml")
        self.assertEqual(cfg["trial"]["check_sigma_max"], 6e-4)
        self.assertEqual(cfg["mdvmc"]["ranks"], 32)
        self.assertEqual(self.read_json(self.key, "trial_facts.json")["n_det"], 55)
        p = self.read_json(self.key, "point.json")
        self.assertIsNone(p["wait"])
        self.assertEqual(p["status"], "ACTIVE")
        self.assertEqual(eng.read(self.key).meta["trial"]["n_det"], 55)

    def test_a_failed_quantum_package_job_fails_the_point_and_retry_starts_again(self):
        eng = self.engine()
        self.point(eng)
        self.put("qpcreate.failed", "QUALITY: basis/ao_normalized of the EZFIO is not T\n")
        self.end(S.FAILED)
        data_ = eng.read(self.key)
        self.assertEqual(data_.state, FAILED)
        self.assertIn("the Quantum Package calculation failed", data_.message)
        self.assertEqual(eng.failed_keys(), [self.key])
        os.remove(os.path.join(self.folder, "qpcreate.failed"))
        self.assertTrue(eng.retry(self.key))
        self.assertEqual(eng.read(self.key).state, QUEUED)
        self.assertEqual(len(self.sched.submitted), 2)

    def test_second_series_reads_the_determinants_of_the_base(self):
        eng = self.engine()
        self.point(eng)
        self.finish_qp2(eng)
        key = self.key
        block, _f = MD.read_determinantset(os.path.join(DATA, "md_h2o_c4q.wfj.xml"), "trial.h5")
        self.write(key, "md_determinantset.xml", block + "\n")
        self.write(key, "trial.h5", "h5\n")
        self.write(key, "system.json", json.dumps(self.system(key)))
        self.write(key, "energy.dat", "%.18e\n0.0\n" % TE.E_SCF)
        self.write(key, "jastrow.json", read(TE.JASTROW))
        eng.submit(self.spec(key + "_s2", blocks=200, role="final", series=2, base_key=key, settings=self.SETTINGS))
        p2 = self.read_json(key + "_s2", "point.json")
        self.assertEqual(p2["steps"], ["dmc"])
        self.assertEqual(p2["trial"]["cutoff"], 1e-3)
        text = self.read_text(key + "_s2", "dmc2.in.xml")
        self.assertIn('<detlist size="55"', text)
        self.assertEqual(text.count('href="../start/trial.h5"'), 2)
        self.assertNotIn("scf.h5", text)
        self.assertNotIn("slaterdeterminant", text)
        job = self.sched.submitted[-1]["text"]
        self.assertNotIn("begin_step mdc4q", job)
        self.assertIn("begin_step dmc", job)
        # a further series that names no trial wave function gets the one of its base point
        eng.submit(self.spec(key + "_s3", blocks=200, role="final", series=3, base_key=key))
        self.assertEqual(self.read_json(key + "_s3", "point.json")["trial"]["cutoff"], 1e-3)
        # one that names ANOTHER wave function is refused: its energies would not be those of the request
        for other in ({"trial": {"kind": "multidet", "cutoff": 1e-4, "n_det_max": 20000}}, {"trial": {"kind": "single"}}):
            with self.assertRaises(ValueError, msg=str(other)) as ctx:
                eng.submit(self.spec(key + "_s4", blocks=200, role="final", series=4, base_key=key, settings=other))
            self.assertIn("another trial wave function", str(ctx.exception))
        self.assertFalse(os.path.exists(self.path(key + "_s4", "point.json")))

    def test_a_point_with_an_imported_wave_function(self):
        eng = self.engine()
        settings = {"trial": {"kind": "multidet", "import_ezfio": "/data/store/A_t1e-3/pre.ezfio", "label": "wf_A_t1e-3"}}
        eng.submit(self.spec(self.key, blocks=200, role="final", settings=settings))
        p = self.read_json(self.key, "point.json")
        self.assertEqual(p["steps"], ["scf", "mdc4q", "mdvmc", "opt", "dmc"])
        self.assertEqual(len(self.sched.submitted), 1)                 # the import job; no parent job, no CIPSI
        job = self.sched.submitted[0]
        self.assertTrue(job["script"].endswith("qp2/cut_wf_A_t1e-3/job_cut.sh"))
        self.assertIn("copy_ezfio /data/store/A_t1e-3/pre.ezfio pre.ezfio", job["text"])
        d = "cut_wf_A_t1e-3/"
        self.put(d + "pre.ezfio/hartree_fock/energy", "  -0.169432220412203E+02\n")
        self.put(d + "pre.ezfio/mo_basis/mo_num", "57\n")
        self.put(d + "pre.ezfio/determinants/n_det", "55\n")
        self.put(d + "pre.ezfio/determinants/psi_det.gz", "x")
        self.put(d + "print_energy.log", data("qp2_h2o_print_energy.log"))
        self.put(d + "export/QP2QMCPACK.h5", "h5\n")
        self.put(d + "export/fingerprint", "0/0/0 64\n")
        for step in MD.IMPORT_STEPS:
            self.put(d + step + ".started", "1500\n"); self.put(d + step + ".done", "1510\n")
        self.end()
        eng.advance()                                                  # the export is ready: the QMC job starts
        self.assertEqual(len(self.sched.submitted), 2)
        text = self.sched.submitted[-1]["text"]
        self.assertIn(os.path.join(self.folder, "cut_wf_A_t1e-3", "export", "QP2QMCPACK.h5"), text)
        self.assertIn("-multidet trial.h5", text)
        facts = self.read_json(self.key, "trial_facts.json")
        self.assertEqual((facts["n_det"], facts["label"]), (55, "wf_A_t1e-3"))
        self.assertAlmostEqual(facts["e_ref"], -16.980443615992332, places=12)
        meta = eng.read(self.key).meta["trial"]
        self.assertEqual((meta["n_det"], meta["imported_from"]), (55, "/data/store/A_t1e-3/pre.ezfio"))
        self.assertGreater(eng.cost(self.key), 0.0)                    # the import steps are counted (8 threads, 30 s)

    def test_an_existing_point_keeps_its_wave_function(self):
        eng = self.engine()
        self.point(eng)
        eng.submit(self.spec(self.key, blocks=200, role="final", settings=self.SETTINGS))        # the same request again
        for other in ({"trial": {"kind": "multidet", "cutoff": 1e-4, "n_det_max": 20000}},
                      {"trial": {"kind": "multidet", "cutoff": "hf"}}, None):
            with self.assertRaises(ValueError, msg=str(other)) as ctx:
                eng.submit(self.spec(self.key, blocks=200, role="final", settings=other))
            self.assertIn("another trial wave function", str(ctx.exception))
        self.assertEqual(len(self.sched.submitted), 1)
        # the same holds for a point with one determinant that is asked again with several
        eng.submit(self.spec("min/it1/centre"))
        with self.assertRaises(ValueError):
            eng.submit(self.spec("min/it1/centre", settings=self.SETTINGS))

    def test_a_new_process_finds_a_point_that_failed_in_the_quantum_package_part(self):
        eng = self.engine()
        self.point(eng)
        self.put("qpcreate.failed", "QUALITY: basis/ao_normalized of the EZFIO is not T\n")
        self.end(S.FAILED)
        self.assertEqual(eng.read(self.key).state, FAILED)
        new = self.engine()                                            # the driver was started again
        self.assertEqual(new.keys(), [self.key])
        self.assertEqual(new.failed_keys(), [self.key])
        os.remove(os.path.join(self.folder, "qpcreate.failed"))
        self.assertEqual(new.retry_failed(), [self.key])
        self.assertEqual(len(self.sched.submitted), 2)
        # and a point that still waits for Quantum Package is found too
        waiting = self.engine()
        self.assertEqual(waiting.read(self.key).state, QUEUED)

    def test_a_check_that_failed_inside_the_job_is_in_the_record(self):
        eng = self.engine()
        self.point(eng)
        self.finish_qp2(eng)
        key = self.key
        self.sched.state_of[self.sched.submitted[-1]["id"]] = S.RUNNING
        verdict = {"check": "mdvmc", "ok": False, "verdict": "FAIL", "energy": -16.97, "error": 4e-4, "e_ref": E_CUT,
                   "difference": 0.0104, "message": "VMC energy ... the difference is above 5 standard errors"}
        self.write(key, "check_mdvmc.json", json.dumps(verdict))
        self.write(key, "mdvmc.failed", "QUALITY: VMC energy ... the difference is above 5 standard errors\n")
        self.sched.state_of[self.sched.submitted[-1]["id"]] = S.FAILED
        data_ = eng.read(key)
        self.assertEqual(data_.state, FAILED)
        self.assertIn("step mdvmc", data_.message)
        self.assertEqual(data_.meta["checks"]["mdvmc"]["verdict"], "FAIL")
        self.assertAlmostEqual(data_.meta["checks"]["mdvmc"]["difference"], 0.0104)

    def test_stop_of_a_waiting_point_cancels_its_quantum_package_job(self):
        eng = self.engine()
        self.point(eng)
        job = self.sched.submitted[-1]["id"]
        eng.stop(self.key)
        self.assertEqual(self.sched.cancelled, [job])
        self.assertEqual(len(self.sched.submitted), 1)                 # the stop submits nothing
        self.assertEqual(eng.read(self.key).state, "STOPPED")

    def test_stop_ignores_a_point_whose_qmc_job_already_runs(self):
        """The RHF determinant point of a geometry runs its QMC job; the CIPSI job serves only the numeric cutoff.
        Stopping the numeric point cancels the CIPSI job, whatever the order of the stops."""
        eng = self.engine()
        shared = "trial/min/shared_qp2"
        hf = {"trial": {"kind": "multidet", "cutoff": "hf", "qp2_key": shared}}
        num = {"trial": {"kind": "multidet", "cutoff": 1e-3, "n_det_max": 20000, "qp2_key": shared}}
        eng.submit(self.spec("trial/min/hf", blocks=200, role="final", settings=hf))
        parent = self.sched.submitted[-1]["id"]                         # RHF only, for the hf point
        self.folder = os.path.join(self.root, *shared.split("/"))
        self.fake_scf(); self.sched.state_of[parent] = S.COMPLETED
        eng.advance()                                                   # the hf cut job
        self.fake_cut("hf"); self.sched.state_of[self.sched.submitted[-1]["id"]] = S.COMPLETED
        eng.submit(self.spec("trial/min/num", blocks=200, role="final", settings=num))
        eng.advance()                                                   # hf: QMC job; num: the CIPSI job
        cipsi = [j for j in self.sched.submitted if j["script"].endswith("job_parent.sh")][-1]["id"]
        self.assertNotEqual(cipsi, parent)
        self.assertEqual(self.read_json("trial/min/hf", "point.json")["status"], "ACTIVE")
        eng.stop("trial/min/num")
        self.assertIn(cipsi, self.sched.cancelled)

    def test_retry_makes_a_deleted_quantum_package_folder_again(self):
        eng = self.engine()
        self.point(eng)
        self.put("qpcreate.failed", "QUALITY: basis/ao_normalized of the EZFIO is not T\n")
        self.end(S.FAILED)
        self.assertEqual(eng.read(self.key).state, FAILED)
        shutil.rmtree(self.folder)                                      # what the documentation says to do
        new = self.engine()                                             # a new process (the command retry)
        self.assertTrue(new.retry(self.key))
        self.assertEqual(new.read(self.key).state, QUEUED)
        self.assertEqual(len(self.sched.submitted), 2)                  # a new parent job in the new folder
        self.assertTrue(self.sched.submitted[-1]["script"].endswith("qp2/job_parent.sh"))
        self.assertTrue(os.path.exists(os.path.join(self.folder, "geom.xyz")))

    def test_a_further_series_asked_again_without_a_trial_setting(self):
        eng = self.engine()
        self.point(eng)
        self.finish_qp2(eng)
        key = self.key
        block, _f = MD.read_determinantset(os.path.join(DATA, "md_h2o_c4q.wfj.xml"), "trial.h5")
        self.write(key, "md_determinantset.xml", block + "\n")
        self.write(key, "trial.h5", "h5\n")
        self.write(key, "system.json", json.dumps(self.system(key)))
        self.write(key, "energy.dat", "%.18e\n0.0\n" % TE.E_SCF)
        self.write(key, "jastrow.json", read(TE.JASTROW))
        spec = self.spec(key + "_s2", blocks=200, role="final", series=2, base_key=key)
        eng.submit(spec)
        eng.submit(spec)                                                # the same request again: accepted
        self.assertEqual(self.read_json(key + "_s2", "point.json")["trial"]["cutoff"], 1e-3)

    def test_stop_leaves_a_quantum_package_job_that_another_point_needs(self):
        eng = self.engine()
        shared = "trial/min/shared_qp2"
        a = {"trial": {"kind": "multidet", "cutoff": 1e-3, "n_det_max": 20000, "qp2_key": shared}}
        b = {"trial": {"kind": "multidet", "cutoff": 1e-4, "n_det_max": 20000, "qp2_key": shared}}
        eng.submit(self.spec("trial/min/a", blocks=200, role="final", settings=a))
        eng.submit(self.spec("trial/min/b", blocks=200, role="final", settings=b))
        self.assertEqual(len(self.sched.submitted), 1)                 # one parent job serves both points
        eng.stop("trial/min/a")
        self.assertEqual(self.sched.cancelled, [])
        eng.stop("trial/min/b")
        self.assertEqual(self.sched.cancelled, [self.sched.submitted[0]["id"]])

    def test_the_speed_rule_compares_only_points_with_the_same_kind_of_wave_function(self):
        eng = self.engine()
        p = self.point(eng)
        kind = "md:trial/min/start/qp2:c1e-03"                           # one wave function at one geometry
        self.assertEqual(eng._wave_kind(p), kind)
        self.assertEqual(eng._wave_kind({"trial": None}), "")
        eng._speed_scanned = True
        eng._speed = {"x/1": 1.0, "x/2": 1.2, "y/1": 9.0, "z/1": 30.0}  # core-seconds per walker and step
        eng._speed_kind = {"x/1": "", "x/2": "", "y/1": kind,
                           "z/1": "md:trial/saddle/start/qp2:c1e-03"}      # the same cutoff at another geometry
        self.assertAlmostEqual(eng._reference_cost(self.key, 1), 9.0)  # not the fast points with one determinant
        self.assertIsNone(eng._reference_cost(self.key, 2))
        self.assertAlmostEqual(eng._reference_cost("unknown/point", 2), 1.1)

    def test_the_job_time_of_such_a_point_comes_from_the_profile(self):
        text = read(self.profile).replace("[job.qmc]\nnodes = 1",
                                          "[job.qmc]\ntime_multidet = 20:00:00\nmem_multidet = 96G\nnodes = 1")
        with open(self.profile, "w") as fh:
            fh.write(text)
        eng = self.engine()
        self.point(eng)
        self.finish_qp2(eng)
        self.assertIn("#SBATCH --time=20:00:00", self.sched.submitted[-1]["text"])
        self.assertIn("#SBATCH --mem=96G", self.sched.submitted[-1]["text"])
        eng2 = self.engine()                                           # a default point keeps the usual limits
        eng2.submit(self.spec("min/it1/centre"))
        self.assertIn("#SBATCH --time=02:00:00", self.sched.submitted[-1]["text"])
        self.assertNotIn("#SBATCH --mem=96G", self.sched.submitted[-1]["text"])

    def test_a_profile_without_quantum_package_is_refused(self):
        text = read(self.profile).replace("[job.qp2]", "[job.none]")
        with open(self.profile, "w") as fh:
            fh.write(text)
        with self.assertRaises(ValueError) as ctx:
            self.engine().submit(self.spec(self.key, role="final", settings=self.SETTINGS))
        self.assertIn("section [job.qp2] is missing", str(ctx.exception))

    def test_quality_checks_of_the_engine(self):
        eng = self.engine()
        self.point(eng)
        self.finish_qp2(eng)
        key = self.key
        self.sched.state_of[self.sched.submitted[-1]["id"]] = S.RUNNING
        system = self.system(key)                                      # grouped as H H O
        system["e_rhf"] = E_SCF
        self.write(key, "system.json", json.dumps(system))
        self.write(key, "energy.dat", "%.18e\n0.0\n" % TE.E_SCF)
        self.mark(key, "scf", 1000, 1060)
        # the export of Quantum Package has the atoms in ANOTHER order than the SCF of this test: the check must see it
        self.write(key, "mdc4q.wfj.xml", data("md_h2o_c4q.wfj.xml"))
        self.write(key, "mdc4q.structure.xml", data("md_h2o_c4q.structure.xml"))
        self.mark(key, "mdc4q", 1060, 1070)
        data_ = eng.read(key)
        self.assertEqual(data_.state, FAILED)
        self.assertIn("mdc4q check", data_.message)
        self.assertIn("are not the atoms of system.json", data_.message)
        self.assertIn(self.sched.submitted[-1]["id"], self.sched.cancelled)


if __name__ == "__main__":
    unittest.main()
