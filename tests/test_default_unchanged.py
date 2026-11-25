"""The files of a point with the default trial wave function (one determinant) must not change.

The option for several determinants (``settings["trial"]``, module ``engine/multidet.py``) touches the
engine and the input writer. This test proves that a point WITHOUT that option gets the same files as
before, byte for byte: the job script, the PySCF script, the settings file, the three QMCPACK inputs,
and the job script and the DMC input of a second series.

The reference files in ``tests/data/ref_default`` were written by this same file with the version of the
package before the option for several determinants was added:

    MONTEPASS_SRC=<src folder of that version> python tests/test_default_unchanged.py --write-reference

Folder names that differ from run to run are replaced by the words @ROOT@, @PKG@ and @TMP@.
"""
import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.environ.get("MONTEPASS_SRC") or os.path.join(os.path.dirname(HERE), "src")
sys.path.insert(0, SRC)

import numpy as np  # noqa: E402

from montepass.engine import qmcpack as E  # noqa: E402
from montepass.engine import qmcpack_inputs as Q  # noqa: E402
from montepass.engine.base import PointSpec  # noqa: E402
from montepass.scheduler import base as S  # noqa: E402
from montepass.scheduler.base import Scheduler  # noqa: E402
from montepass.units import ANGSTROM_TO_BOHR  # noqa: E402

REF = os.path.join(HERE, "data", "ref_default")
SYMBOLS = ["H", "O", "H"]                       # not grouped by element, as a user may give them
COORDS = np.array([[0.0, 0.7631265780, -0.5855676186], [0.0, 0.0, 0.0], [0.0, -0.7631265780, -0.5855676186]])

PROFILE = """
[machine]
name = test-cluster
scheduler = slurm

[setup.scf]
lines =
    module load scf-tools

[setup.qmc]
lines =
    module load qmc-tools
    export PATH=/opt/qmcpack/bin:$PATH

[executables]
python = python3
qmcpack = qmcpack_real
convert4qmc = convert4qmc
pyscf_qmctools = /opt/qmcpack/src/QMCTools

[launch]
qmc = mpirun -np {ranks} --map-by core --bind-to core

[pseudopotentials]
dir = %(pp)s

[slurm]
account = testaccount
qos = testqos
partition = testpartition

[job.scf]
nodes = 1
tasks = 1
cpus_per_task = 4
mem = 8G
time = 00:30:00

[job.qmc]
nodes = 1
tasks = 32
mem = 16G
time = 02:00:00
constraint = testfeature
"""


class _Scheduler(Scheduler):
    def __init__(self):
        self.n = 5000

    def submit(self, script, workdir, name="job", after=None):
        self.n += 1
        return str(self.n)

    def states(self, job_ids):
        return {j: S.PENDING for j in job_ids}

    def state(self, job_id):
        return S.PENDING

    def cancel(self, job_id):
        pass


def _scalar(path, energy, variance, n=40):
    """A small VMC trace with fixed numbers: the mean energy and the variance of the local energy are as given."""
    lines = ["#   index    LocalEnergy    LocalEnergy_sq    BlockWeight"]
    for i in range(n):
        e = energy + 0.001 * ((i * 7) % 11 - 5)
        lines.append("%10d  %.10e  %.10e  %.10e" % (i, e, e * e + variance, 3200.0))
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def _opt_xml(path, shift):
    ids = {"eH": 6, "eO": 6, "uu": 8, "ud": 8}
    lines = ['<?xml version="1.0"?>', "<qmcsystem>", '<wavefunction name="psi0" target="e">']
    for k, (cid, n) in enumerate(sorted(ids.items())):
        lines.append('<coefficients id="%s" type="Array"> %s </coefficients>'
                     % (cid, " ".join("%.6f" % (0.01 * (i + 1) * (k + 1) + shift) for i in range(n))))
    with open(path, "w") as fh:
        fh.write("\n".join(lines + ["</wavefunction>", "</qmcsystem>", ""]))


def make_files(tmp):
    """Write the files of a default point and of its second series; return ``{name: text}`` with the changing
    folder names replaced."""
    tmp = os.path.realpath(tmp)
    root, pp = os.path.join(tmp, "points"), os.path.join(tmp, "pseudo")
    os.makedirs(pp)
    for el, z in (("H", 1), ("O", 6)):
        with open(os.path.join(pp, "%s.ccECP.xml" % el), "w") as fh:
            fh.write('<?xml version="1.0"?>\n<pseudo>\n  <header symbol="%s" atomic-number="1" zval="%s" '
                     'creator="x" flavor="y"/>\n</pseudo>\n' % (el, z))
    profile = os.path.join(tmp, "machine.ini")
    with open(profile, "w") as fh:
        fh.write(PROFILE % {"pp": pp})
    settings = {"poll_seconds": 0, "settle_seconds": 0, "unknown_seconds": 0, "jastrow_reuse": "none"}
    eng = E.QmcpackEngine(root, profile, _Scheduler(), settings, clock=lambda: 1000.0)
    key = "min/it1/centre"
    eng.submit(PointSpec(key=key, symbols=list(SYMBOLS), coords=COORDS.copy(), blocks=400, role="centre"))
    folder = eng.folder(key)
    out = {}

    def take(prefix, where, names):
        for name in names:
            with open(os.path.join(where, name)) as fh:
                out[prefix + name] = fh.read()

    take("s1_", folder, ["job.sh", "scf.py", "qmc_settings.json", "order.json", "geom.xyz"])
    # what the job would write: the SCF record, then the three QMCPACK inputs
    with open(os.path.join(folder, "order.json")) as fh:
        order = json.load(fh)["order"]
    grouped = [SYMBOLS[i] for i in order]
    system = {"symbols": grouped, "coords_bohr": (COORDS[order] * ANGSTROM_TO_BOHR).tolist(),
              "zeff": [{"H": 1, "O": 6}[s] for s in grouped], "nup": 4, "ndn": 4, "nao": 58, "energy": -17.2371,
              "e_determinant": -16.98, "xc": "b3lyp", "basis": "ccecpccpvtz", "ecp": "ccecp", "restricted": True}
    with open(os.path.join(folder, "system.json"), "w") as fh:
        json.dump(system, fh, indent=1)
    here = os.getcwd()
    os.chdir(folder)
    try:
        with open(os.devnull, "w") as null:
            stdout, sys.stdout = sys.stdout, null
            try:
                Q.main(["vmc0", "qmc_settings.json"])
                Q.main(["opt", "qmc_settings.json"])
                _scalar("opt.s000.scalar.dat", -17.00, 0.90)
                _scalar("opt.s001.scalar.dat", -17.20, 0.40)
                _scalar("opt.s002.scalar.dat", -17.21, 0.38)
                _opt_xml("opt.s000.opt.xml", 0.0)
                _opt_xml("opt.s001.opt.xml", 0.5)
                Q.main(["dmc", "qmc_settings.json"])
            finally:
                sys.stdout = stdout
    finally:
        os.chdir(here)
    take("s1_", folder, ["vmc0.in.xml", "opt.in.xml", "dmc.in.xml", "jastrow.json"])
    # a second series of the same geometry: it needs the orbital file and the Jastrow factor of the first
    with open(os.path.join(folder, "scf.h5"), "w") as fh:
        fh.write("orbital file\n")
    key2 = key + "_s2"
    eng.submit(PointSpec(key=key2, symbols=list(SYMBOLS), coords=COORDS.copy(), blocks=400, role="centre",
                         series=2, base_key=key))
    take("s2_", eng.folder(key2), ["job.sh", "dmc2.in.xml"])
    pkg = os.path.dirname(os.path.dirname(os.path.abspath(E.__file__)))
    pkg = os.path.dirname(pkg)                      # the folder that holds the package montepass
    for name in out:
        out[name] = out[name].replace(root, "@ROOT@").replace(pkg, "@PKG@").replace(tmp, "@TMP@")
    return out


class TestDefaultPointIsUnchanged(unittest.TestCase):
    def test_every_file_equals_the_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            files = make_files(tmp)
        self.assertEqual(sorted(files), sorted(os.listdir(REF)))
        for name, text in sorted(files.items()):
            with open(os.path.join(REF, name)) as fh:
                self.assertEqual(text, fh.read(), "the file %s of a default point differs from the reference" % name)

    def test_the_reference_holds_no_step_of_the_other_chain(self):
        for name in ("s1_job.sh", "s2_job.sh"):
            with open(os.path.join(REF, name)) as fh:
                text = fh.read()
            for word in ("mdc4q", "mdvmc", "qp_run", "multidet"):
                self.assertNotIn(word, text)


if __name__ == "__main__":
    if "--write-reference" in sys.argv:
        os.makedirs(REF, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            made = make_files(tmp)
        for name, text in sorted(made.items()):
            with open(os.path.join(REF, name), "w") as fh:
                fh.write(text)
        print("wrote %d reference files to %s with the package in %s" % (len(made), REF, SRC))
    else:
        unittest.main()
