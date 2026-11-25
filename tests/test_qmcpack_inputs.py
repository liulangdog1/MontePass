"""The QMCPACK input writer: the order of the atoms, the SCF script, the checks of a system and of the
VMC input without a Jastrow factor, and the choice of the Jastrow factor of an optimization.
"""
import os
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

from montepass.engine import qmcpack_inputs as Q  # noqa: E402


class TestOrderAndScript(unittest.TestCase):
    def test_species_order(self):
        syms = ["N", "C", "C", "N", "H", "C", "H"]
        perm = Q.species_order(syms)
        self.assertEqual([syms[i] for i in perm], ["N", "N", "C", "C", "C", "H", "H"])
        self.assertEqual(perm, [0, 3, 1, 2, 5, 4, 6])
        self.assertTrue(Q.is_grouped(["O", "H", "H"])); self.assertFalse(Q.is_grouped(["H", "O", "H"]))

    def test_scf_script_refuses_ungrouped_atoms(self):
        with self.assertRaises(ValueError):
            Q.scf_script(["H", "O", "H"], np.zeros((3, 3)))

    def test_scf_script_is_valid_python(self):
        text = Q.scf_script(["O", "H", "H"], [[0, 0, 0.1173], [0, 0.7572, -0.4692], [0, -0.7572, -0.4692]], gradient=True)
        compile(text, "scf.py", "exec")
        for needle in ("mol.basis = 'ccecpccpvtz'", "mol.ecp = 'ccecp'", "mf.xc = 'b3lyp'", "savetoqmcpack(mol, mf, 'scf')",
                       "system.json", "gradient.dat", "mf.conv_tol = 1e-10", "e_determinant"):
            self.assertIn(needle, text)

    def test_ungrouped_system_file_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "system.json")
            with open(p, "w") as fh:
                fh.write('{"symbols": ["H", "O", "H"], "coords_bohr": [[0,0,0],[0,0,1],[0,1,0]], "zeff": [1,6,1], "nup": 4, "ndn": 4, "nao": 58}')
            with self.assertRaises(ValueError):
                Q.load_system(p)


class TestChecks(unittest.TestCase):
    SYSTEM = {"symbols": ["O", "H", "H"], "coords_bohr": [[0, 0, 0], [0, 1.4, -1.1], [0, -1.4, -1.1]],
              "zeff": [6, 1, 1], "nup": 4, "ndn": 4, "nao": 58, "charge": 0, "restricted": True}

    def write_pp(self, folder, el, zval):
        with open(os.path.join(folder, "%s.ccECP.xml" % el), "w") as fh:
            fh.write('<?xml version="1.0"?>\n<pseudo>\n  <header symbol="%s" atomic-number="1" zval="%s" '
                     'creator="x" flavor="y"/>\n</pseudo>\n' % (el, zval))

    def test_pseudopotential_charges(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.write_pp(tmp, "O", 6); self.write_pp(tmp, "H", 1)
            Q.check_pseudopotentials(self.SYSTEM, tmp)                      # consistent: no error
            self.write_pp(tmp, "O", 8)                                      # an all-electron charge by mistake
            with self.assertRaises(ValueError):
                Q.check_pseudopotentials(self.SYSTEM, tmp)
            os.remove(os.path.join(tmp, "H.ccECP.xml")); self.write_pp(tmp, "O", 6)
            with self.assertRaises(ValueError):
                Q.check_pseudopotentials(self.SYSTEM, tmp)                  # missing file

    def test_zero_jastrow_input(self):
        root = ET.fromstring(Q.vmc_check_input(self.SYSTEM))
        self.assertEqual([q.get("method") for q in root.iter("qmc")], ["vmc"])
        # the determinant alone: a Jastrow element with zero coefficients would still add the electron-electron cusp
        self.assertEqual(len(list(root.iter("jastrow"))), 0)
        self.assertEqual(len(list(root.iter("determinant"))), 2)


class TestJastrowSelection(unittest.TestCase):
    def write_cycle(self, folder, n, energy, variance, params=None):
        rng = np.random.default_rng(n)
        with open(os.path.join(folder, "opt.s%03d.scalar.dat" % n), "w") as fh:
            fh.write("#   index    LocalEnergy   LocalEnergy_sq   BlockWeight\n")
            for i in range(40):
                e = energy + 0.001 * rng.normal()
                fh.write("%d %.10f %.10f 1000\n" % (i, e, variance + e * e))
        if params is not None:
            with open(os.path.join(folder, "opt.s%03d.opt.xml" % n), "w") as fh:
                fh.write('<qmcsystem><wavefunction><jastrow><correlation><coefficients id="uu" type="Array"> %s</coefficients>'
                         '</correlation></jastrow></wavefunction></qmcsystem>' % " ".join("%.6f" % v for v in params))

    def test_selection_uses_measured_parameters(self):
        with tempfile.TemporaryDirectory() as tmp:
            # cycle 0 measures the zero Jastrow; parameters of cycle n are measured by cycle n + 1
            self.write_cycle(tmp, 0, -17.00, 1.00, params=[0.1])
            self.write_cycle(tmp, 1, -17.20, 0.40, params=[0.2])
            self.write_cycle(tmp, 2, -17.23, 0.35, params=[0.3])      # measures the parameters written by cycle 1
            self.write_cycle(tmp, 3, -17.21, 0.36, params=[0.4])      # the parameters of cycle 3 are never measured
            cycles = Q.read_optimization(tmp)
            self.assertEqual([c["series"] for c in cycles], [0, 1, 2, 3])
            self.assertIsNone(cycles[0]["parameters_from"])
            best = Q.select_jastrow(cycles)
            self.assertEqual(best["series"], 2)
            self.assertEqual(Q.read_jastrow(best["parameters_from"])["uu"], [0.2])

    def test_high_variance_cycle_is_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.write_cycle(tmp, 0, -17.00, 1.00, params=[0.1])
            self.write_cycle(tmp, 1, -17.20, 0.40, params=[0.2])
            self.write_cycle(tmp, 2, -17.60, 3.00, params=[0.3])      # lower energy but 7.5 times the variance: a broken cycle
            self.write_cycle(tmp, 3, -17.22, 0.41, params=[0.4])
            best = Q.select_jastrow(Q.read_optimization(tmp))
            self.assertEqual(best["series"], 3)

    def test_no_measured_cycle(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.write_cycle(tmp, 0, -17.0, 1.0)
            with self.assertRaises(ValueError):
                Q.select_jastrow(Q.read_optimization(tmp))


if __name__ == "__main__":
    unittest.main()
