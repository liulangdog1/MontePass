"""PySCF surrogate jobs: Kohn-Sham energy, gradient and Hessian in the search directions.

The local surrogate of the geometry search is the same Kohn-Sham calculation that makes the QMC
trial orbitals (B3LYP with ccECP pseudopotentials by default). Its energy comes with every QMC
point. Its gradient and its Hessian need extra calculations, and this module runs them as batch jobs.

    surrogate = PyscfSurrogate(root, machine, scheduler, symbols, vectors)
    surrogate.request("min/it1/surrogate", coords)       # writes the job folder and submits it once
    res = surrogate.result("min/it1/surrogate")          # None while the job waits or runs

One request is one job in the folder ``<root>/<tag>/``:

    task.json           what to compute (written here)
    surrogate_job.py    the calculation script; it imports numpy and PySCF only
    job.sh              the job script with marker files (surrogate.started, .done, .failed)
    checkpoint.json     finished calculations; a job that is started again continues from it
    result.json         the results

Cost of a request with the Hessian: 1 + 2 d energy-and-gradient calculations for d directions.
The Hessian element (i, j) is the central difference of the gradient component along direction
i for a displacement of +-``fd_step`` along direction j. The matrix is symmetrized, and the
largest asymmetry before that is reported as a measure of the numerical noise.

A job that stops at its time limit is submitted again while each attempt finishes at least one
more calculation, up to ``max_attempts`` attempts.
"""
from __future__ import annotations

import json
import os
import shlex
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..geometry import normalize_symbol
from ..scheduler.base import FAILED, JobTable, Scheduler
from ..units import BOHR_TO_ANGSTROM, HARTREE_TO_KCAL
from .base import Surrogate, SurrogateResult
from .orca import ABSENT, DONE, SCRIPT_PREAMBLE, JobStatus, StepJob, step_block, step_limit, write_json, write_text

TASK_FILE = "task.json"
SCRIPT_FILE = "surrogate_job.py"
CHECKPOINT_FILE = "checkpoint.json"
RESULT_FILE = "result.json"
ATTEMPTS_FILE = "attempts.json"
STEP = "surrogate"

DEFAULT_SETTINGS: Dict[str, object] = {
    "charge": 0,
    "spin": 0,                  # number of unpaired electrons (2S), the PySCF convention
    "xc": "b3lyp",
    "basis": "ccecpccpvtz",
    "ecp": "ccecp",             # "" for an all-electron calculation
    "grid_level": 3,
    "conv_tol": 1e-10,          # SCF convergence in hartree
    "max_cycle": 300,
    "fd_step": 0.005,           # displacement of the finite differences, angstrom
    "grid_response": False,     # include the response of the integration grid in the gradient
    "reuse_density": True,      # start each SCF from the density matrix of the calculation before it
    "threads": None,            # None: cpus_per_task of the kind "scf" in the machine profile
    "time": None,               # batch time limit, for example "04:00:00"; None: the value of the profile
    "max_memory_mb": None,      # memory that PySCF may use; None: the PySCF default
    "verbose": 4,               # print level of PySCF
    "max_attempts": 6,          # how often a job that stops at its time limit is submitted in total
}
_PHYSICS = ("charge", "spin", "xc", "basis", "ecp", "grid_level", "conv_tol", "max_cycle", "grid_response")
_RUN = ("threads", "max_memory_mb", "verbose", "reuse_density")


# The script that runs in the job folder. It is kept as text because it must work without this package.
JOB_SCRIPT = r'''#!/usr/bin/env python3
"""Kohn-Sham surrogate calculations of one job folder.

Start it in a folder that holds task.json:    python surrogate_job.py

kind "point"   energy and analytic nuclear gradient at one geometry and, if asked, the Hessian in
               a set of directions by central differences of the projected gradient
kind "scan"    energies of a list of geometries

The script needs numpy and PySCF, nothing else. After every finished calculation it rewrites
checkpoint.json. A second start in the same folder continues after the last finished calculation.
The results go to result.json. Both files are written to a temporary name first and then renamed.

Units: coordinates and displacements in angstrom, energies in hartree, gradients in hartree/bohr
(as PySCF returns them), the Hessian in the direction basis in hartree/angstrom^2.
"""
import hashlib
import json
import os
import sys
import time

import numpy as np

BOHR_TO_ANGSTROM = 0.529177210903
TASK_FILE = "task.json"
CHECKPOINT_FILE = "checkpoint.json"
RESULT_FILE = "result.json"


def write_json(path, data):
    """Write a JSON file atomically: write <path>.tmp, then rename."""
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(data, fh, indent=1)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def read_json(path):
    """Read a JSON file. A missing, empty or broken file is an error that names the file."""
    if not os.path.exists(path):
        raise RuntimeError("file not found: %s" % path)
    with open(path) as fh:
        text = fh.read()
    if not text.strip():
        raise RuntimeError("file is empty: %s" % path)
    try:
        return json.loads(text)
    except ValueError as exc:
        raise RuntimeError("%s is not valid JSON: %s" % (path, exc))


def fingerprint(task):
    """Identity of the numerical task. Settings that only change the speed (the entry "run") are left out."""
    core = {k: v for k, v in task.items() if k != "run"}
    return hashlib.sha256(json.dumps(core, sort_keys=True).encode()).hexdigest()


def load_checkpoint(path, mark):
    """Finished calculations of an earlier start, or an empty record. A checkpoint of another task is an error."""
    if not os.path.exists(path):
        return {"fingerprint": mark, "evaluations": {}}
    data = read_json(path)
    if data.get("fingerprint") != mark:
        raise RuntimeError("%s belongs to another task (other geometry, directions or settings); "
                           "remove it, or use a new folder" % path)
    if not isinstance(data.get("evaluations"), dict):
        raise RuntimeError("%s has no table of finished calculations" % path)
    return data


def point_plan(n_directions, hessian, gradient):
    """Calculations of a "point" task in the order they run: (label, direction, sign, gradient needed)."""
    plan = [("centre", None, 0, bool(gradient))]
    if hessian:
        for j in range(n_directions):
            plan.append(("d%d_plus" % j, j, +1, True))
            plan.append(("d%d_minus" % j, j, -1, True))
    return plan


def displaced(coords, vectors, direction, shift):
    """Geometry (n, 3) moved by ``shift`` angstrom along a direction (a flat unit vector of length 3 n)."""
    coords = np.asarray(coords, dtype=float).reshape(-1, 3)
    if direction is None:
        return coords.copy()
    return coords + shift * np.asarray(vectors, dtype=float)[direction].reshape(-1, 3)


def evaluate(label, coords, need_gradient, energy_gradient, checkpoint, checkpoint_path, log):
    """One calculation, or its stored result. The checkpoint file is rewritten after a new calculation."""
    done = checkpoint["evaluations"]
    old = done.get(label)
    if old is not None and (old.get("gradient") is not None or not need_gradient):
        return old
    start = time.time()
    log("calculation %s started" % label)
    energy, grad = energy_gradient(coords, need_gradient)
    energy = float(energy)
    if not np.isfinite(energy):
        raise RuntimeError("calculation %s: the energy is not a finite number" % label)
    record = {"energy": energy, "gradient": None, "seconds": 0.0}
    if need_gradient:
        g = np.asarray(grad, dtype=float)
        if g.shape != np.asarray(coords).shape or not np.all(np.isfinite(g)):
            raise RuntimeError("calculation %s: the gradient has shape %s or holds a value that is not finite"
                               % (label, g.shape))
        record["gradient"] = g.tolist()
    record["seconds"] = time.time() - start
    done[label] = record
    write_json(checkpoint_path, checkpoint)
    log("calculation %s finished: E = %.10f hartree, %.1f s" % (label, energy, record["seconds"]))
    return record


def project(vectors, gradient):
    """Gradient components along the directions in hartree/angstrom (the gradient is in hartree/bohr)."""
    return np.asarray(vectors, dtype=float) @ (np.asarray(gradient, dtype=float).reshape(-1) / BOHR_TO_ANGSTROM)


def hessian_in_directions(vectors, gradients_plus, gradients_minus, step):
    """Hessian in the direction basis by central differences of the projected gradient.

    raw[i, j] = (g_i at +step along direction j  -  g_i at -step along direction j) / (2 step),
    with g_i the gradient component along direction i. Returns (symmetrized matrix, largest
    |raw[i, j] - raw[j, i]|, raw), all in hartree/angstrom^2.
    """
    d = len(vectors)
    raw = np.zeros((d, d))
    for j in range(d):
        raw[:, j] = (project(vectors, gradients_plus[j]) - project(vectors, gradients_minus[j])) / (2.0 * step)
    asymmetry = float(np.abs(raw - raw.T).max()) if d else 0.0
    return 0.5 * (raw + raw.T), asymmetry, raw


def run_point(task, energy_gradient, folder=".", log=print):
    """Energy, gradient and Hessian in the directions for one geometry. Returns the result dictionary
    and writes it to result.json."""
    coords = np.asarray(task["coords"], dtype=float).reshape(-1, 3)
    vectors = np.asarray(task.get("vectors") or np.zeros((0, coords.size)), dtype=float).reshape(-1, coords.size)
    step = float(task["fd_step"])
    want_hessian, want_gradient = bool(task["hessian"]), bool(task["gradient"])
    if want_hessian and len(vectors) == 0:
        raise RuntimeError("a Hessian is asked for but the task has no directions")
    if want_hessian and not step > 0.0:
        raise RuntimeError("fd_step must be positive, got %r" % step)
    checkpoint_path = os.path.join(folder, CHECKPOINT_FILE)
    checkpoint = load_checkpoint(checkpoint_path, fingerprint(task))
    plan = point_plan(len(vectors), want_hessian, want_gradient)
    known = sum(1 for p in plan if p[0] in checkpoint["evaluations"])
    log("%d calculations planned, %d of them are in %s" % (len(plan), known, CHECKPOINT_FILE))
    records = {}
    for label, direction, sign, need_gradient in plan:
        x = displaced(coords, vectors, direction, sign * step)
        records[label] = evaluate(label, x, need_gradient, energy_gradient, checkpoint, checkpoint_path, log)

    centre = records["centre"]
    result = {"kind": "point", "fingerprint": checkpoint["fingerprint"], "symbols": task["symbols"],
              "energy": centre["energy"], "gradient": centre["gradient"], "gradient_directions": None,
              "hessian_directions": None, "hessian_directions_raw": None, "hessian_asymmetry": None,
              "curvature_from_energies": None, "energies_plus": None, "energies_minus": None,
              "fd_step": step, "n_directions": int(len(vectors)),
              "units": {"energy": "hartree", "gradient": "hartree/bohr", "gradient_directions": "hartree/angstrom",
                        "hessian_directions": "hartree/angstrom^2", "fd_step": "angstrom", "seconds": "s"}}
    if centre["gradient"] is not None and len(vectors):
        result["gradient_directions"] = project(vectors, centre["gradient"]).tolist()
    if want_hessian:
        d = len(vectors)
        plus = [records["d%d_plus" % j] for j in range(d)]
        minus = [records["d%d_minus" % j] for j in range(d)]
        sym, asymmetry, raw = hessian_in_directions(vectors, [p["gradient"] for p in plus],
                                                    [m["gradient"] for m in minus], step)
        result["hessian_directions"] = sym.tolist()
        result["hessian_directions_raw"] = raw.tolist()
        result["hessian_asymmetry"] = asymmetry
        result["energies_plus"] = [p["energy"] for p in plus]
        result["energies_minus"] = [m["energy"] for m in minus]
        # second difference of the energies: an independent, less precise value of the diagonal
        result["curvature_from_energies"] = [(p["energy"] + m["energy"] - 2.0 * centre["energy"]) / step ** 2
                                             for p, m in zip(plus, minus)]
    result["seconds"] = {label: records[label]["seconds"] for label, _d, _s, _g in plan}
    result["seconds_total"] = float(sum(result["seconds"].values()))
    write_json(os.path.join(folder, RESULT_FILE), result)
    return result


def run_scan(task, energy_gradient, folder=".", log=print):
    """Energies of a list of geometries, in the order of the list. Writes result.json."""
    geometries = [np.asarray(g, dtype=float).reshape(-1, 3) for g in task["geometries"]]
    if not geometries:
        raise RuntimeError("the scan task holds no geometry")
    checkpoint_path = os.path.join(folder, CHECKPOINT_FILE)
    checkpoint = load_checkpoint(checkpoint_path, fingerprint(task))
    known = sum(1 for i in range(len(geometries)) if ("g%d" % i) in checkpoint["evaluations"])
    log("%d calculations planned, %d of them are in %s" % (len(geometries), known, CHECKPOINT_FILE))
    records = [evaluate("g%d" % i, x, False, energy_gradient, checkpoint, checkpoint_path, log)
               for i, x in enumerate(geometries)]
    result = {"kind": "scan", "fingerprint": checkpoint["fingerprint"], "symbols": task["symbols"],
              "energies": [r["energy"] for r in records], "seconds": [r["seconds"] for r in records],
              "seconds_total": float(sum(r["seconds"] for r in records)),
              "units": {"energies": "hartree", "seconds": "s"}}
    write_json(os.path.join(folder, RESULT_FILE), result)
    return result


def pyscf_energy_gradient(symbols, settings, run, log=print):
    """The function ``energy_gradient(coords, need_gradient)`` backed by PySCF.

    ``coords`` is (n, 3) in angstrom in the atom order of ``symbols``. It returns the Kohn-Sham
    energy in hartree and the analytic nuclear gradient (n, 3) in hartree/bohr, or None in place
    of the gradient if it is not needed.
    """
    import pyscf
    from pyscf import dft, gto, lib

    if run.get("threads"):
        lib.num_threads(int(run["threads"]))
    log("PySCF %s, numpy %s, %d threads" % (pyscf.__version__, np.__version__, lib.num_threads()))
    log("functional %s, basis %s, pseudopotential %s, grid level %s, SCF convergence %s hartree"
        % (settings["xc"], settings["basis"], settings.get("ecp") or "none", settings["grid_level"],
           settings["conv_tol"]))
    last = {"dm": None}

    def energy_gradient(coords, need_gradient=True):
        mol = gto.Mole()
        mol.atom = [(s, (float(r[0]), float(r[1]), float(r[2]))) for s, r in zip(symbols, coords)]
        mol.unit = "A"
        mol.basis = settings["basis"]
        if settings.get("ecp"):
            mol.ecp = settings["ecp"]
        mol.charge = int(settings["charge"])
        mol.spin = int(settings["spin"])
        mol.symmetry = False
        mol.verbose = int(run.get("verbose", 4))
        if run.get("max_memory_mb"):
            mol.max_memory = float(run["max_memory_mb"])
        mol.build(dump_input=False)        # with dump_input PySCF prints this whole script before every calculation

        def scf(dm0):
            mf = dft.RKS(mol) if mol.spin == 0 else dft.UKS(mol)
            mf.xc = settings["xc"]
            mf.grids.level = int(settings["grid_level"])
            mf.conv_tol = float(settings["conv_tol"])
            mf.max_cycle = int(settings["max_cycle"])
            mf.kernel(dm0=dm0)
            return mf

        guess = last["dm"] if run.get("reuse_density", True) else None
        mf = scf(guess)
        if not mf.converged and guess is not None:
            log("SCF not converged from the density of the calculation before; starting again from the default guess")
            mf = scf(None)
        if not mf.converged:
            raise RuntimeError("the SCF did not converge in %d cycles" % int(settings["max_cycle"]))
        last["dm"] = mf.make_rdm1()
        gradient = None
        if need_gradient:
            method = mf.nuc_grad_method()
            if settings.get("grid_response"):
                method.grid_response = True
            gradient = np.asarray(method.kernel(), dtype=float)
        return float(mf.e_tot), gradient

    return energy_gradient


def main(folder="."):
    start = time.time()
    task = read_json(os.path.join(folder, TASK_FILE))

    def log(text):
        print("[%8.1f s] %s" % (time.time() - start, text), flush=True)

    engine = pyscf_energy_gradient(task["symbols"], task["settings"], task.get("run", {}), log)
    if task["kind"] == "point":
        result = run_point(task, engine, folder, log)
        if result["hessian_directions"] is not None:
            log("Hessian in the directions (hartree/angstrom^2), largest asymmetry %.2e:" % result["hessian_asymmetry"])
            print(np.array2string(np.array(result["hessian_directions"]), precision=5, suppress_small=True), flush=True)
    elif task["kind"] == "scan":
        run_scan(task, engine, folder, log)
    else:
        raise RuntimeError("%s: unknown kind %r" % (TASK_FILE, task["kind"]))
    log("SURROGATE_DONE")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
'''


class _PyscfJob(StepJob):
    """The batch job of one request."""

    def verify(self) -> str:
        path = self.file(RESULT_FILE)
        try:
            _read_result(path)
        except (OSError, ValueError) as exc:
            return str(exc)
        return ""

    def explain(self) -> str:
        facts = []
        n = finished_calculations(self.folder)
        if n is not None:
            facts.append("%d calculations are finished in %s" % (n, CHECKPOINT_FILE))
        for name in (STEP + ".err", STEP + ".out"):
            p = self.file(name)
            if os.path.exists(p) and os.path.getsize(p) > 0:
                with open(p, errors="replace") as fh:
                    rows = [ln.strip()[:200] for ln in fh.read().splitlines() if ln.strip()]
                facts.append("last lines of %s: %s" % (name, " | ".join(rows[-2:])))
                break
        return "; ".join(facts)


def _read_result(path: str) -> dict:
    """Content of a result file. A missing, empty or broken file is an error that names the file."""
    if not os.path.exists(path):
        raise FileNotFoundError("result file not found: %s" % path)
    with open(path) as fh:
        text = fh.read()
    if not text.strip():
        raise ValueError("result file is empty: %s" % path)
    try:
        data = json.loads(text)
    except ValueError as exc:
        raise ValueError("%s is not valid JSON: %s" % (path, exc))
    if data.get("kind") not in ("point", "scan"):
        raise ValueError("%s: unknown kind %r" % (path, data.get("kind")))
    return data


def finished_calculations(folder: str) -> Optional[int]:
    """Number of finished calculations in the checkpoint of a job folder; None if there is no checkpoint."""
    path = os.path.join(folder, CHECKPOINT_FILE)
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        data = json.load(fh)
    return len(data.get("evaluations", {}))


def _finite_array(value, shape, what: str, path: str) -> np.ndarray:
    a = np.asarray(value, dtype=float)
    if a.shape != shape:
        raise ValueError("%s: %s has shape %s, expected %s" % (path, what, a.shape, shape))
    if not np.all(np.isfinite(a)):
        raise ValueError("%s: %s holds a value that is not a finite number" % (path, what))
    return a


class PyscfSurrogate(Surrogate):
    """Surrogate energies, gradients and direction Hessians from PySCF batch jobs.

    ``root``       folder of the job folders; a request with the tag ``a/b`` uses ``<root>/a/b/``
    ``machine``    machine profile; the jobs are of the kind ``scf``
    ``scheduler``  batch system
    ``symbols``    element symbols in the atom order of the user
    ``vectors``    search directions, shape (d, 3 n), each of unit norm, same atom order
    ``settings``   values that replace entries of :data:`DEFAULT_SETTINGS`
    ``job_table``  record of the submitted jobs; default ``<root>/jobs.json``

    Results (:meth:`result`): energy in hartree, gradient in kcal/mol/angstrom with shape (n, 3)
    in the atom order of the user, ``hessian_directions`` in kcal/mol/angstrom^2 with shape (d, d).
    """

    def __init__(self, root: str, machine, scheduler: Scheduler, symbols: Sequence[str], vectors: np.ndarray,
                 settings: Optional[Dict[str, object]] = None, job_table: Optional[JobTable] = None):
        self.root = os.path.abspath(root)
        self.machine = machine
        self.scheduler = scheduler
        self.symbols = [normalize_symbol(s) for s in symbols]
        n3 = 3 * len(self.symbols)
        V = np.zeros((0, n3)) if vectors is None else np.atleast_2d(np.asarray(vectors, dtype=float))
        if V.size == 0:
            V = np.zeros((0, n3))
        if V.shape[1] != n3:
            raise ValueError("the directions have length %d, expected 3 x %d atoms = %d" % (V.shape[1], len(self.symbols), n3))
        norms = np.linalg.norm(V, axis=1)
        if len(V) and np.abs(norms - 1.0).max() > 1e-6:
            raise ValueError("every direction must have unit norm; found norms %s" % np.round(norms, 6))
        self.vectors = V
        self.settings = dict(DEFAULT_SETTINGS)
        unknown = [k for k in (settings or {}) if k not in DEFAULT_SETTINGS]
        if unknown:
            raise ValueError("unknown surrogate settings: %s; known: %s" % (", ".join(sorted(unknown)),
                                                                           ", ".join(sorted(DEFAULT_SETTINGS))))
        self.settings.update(settings or {})
        available = machine.threads("scf")
        if self.settings["threads"] is None:
            self.settings["threads"] = available
        if int(self.settings["threads"]) > available:
            raise ValueError("%d threads are asked for, but the machine profile gives a job of the kind scf %d CPUs"
                             % (int(self.settings["threads"]), available))
        if self.settings["time"] is None:
            self.settings["time"] = machine.job("scf").get("time")
        if not float(self.settings["fd_step"]) > 0.0:
            raise ValueError("fd_step must be positive, got %r" % self.settings["fd_step"])
        os.makedirs(self.root, exist_ok=True)
        self.job_table = job_table if job_table is not None else JobTable(os.path.join(self.root, "jobs.json"))

    # ------------------------------------------------------------------ job folders
    def folder(self, tag: str) -> str:
        parts = [p for p in str(tag).replace("\\", "/").split("/") if p]
        if not parts or str(tag).startswith("/") or any(p in (".", "..") for p in parts):
            raise ValueError("tag %r is not usable as a folder below %s" % (tag, self.root))
        return os.path.join(self.root, *parts)

    def _job(self, tag: str) -> _PyscfJob:
        return _PyscfJob(self.folder(tag), STEP, self.scheduler, self.job_table, key="surrogate/%s" % tag,
                         kind="scf", job_name="sur_" + str(tag).replace("/", "_"))

    def script_text(self, tag: str) -> str:
        """Text of the job script of a tag."""
        q = shlex.quote
        job = self._job(tag)
        threads = int(self.settings["threads"])
        head = self.machine.header("scf", job.job_name, job.folder, time=self.settings["time"])
        body = [SCRIPT_PREAMBLE.rstrip("\n"),
                "export OMP_NUM_THREADS=%d MKL_NUM_THREADS=%d OPENBLAS_NUM_THREADS=%d" % (threads, threads, threads), ""]
        command = "%s -u %s > %s.out 2> %s.err" % (q(self.machine.executable("python")), SCRIPT_FILE, STEP, STEP)
        before = [("[ -s %s ]" % TASK_FILE, "the file %s is missing or empty" % TASK_FILE),
                  ("[ -s %s ]" % SCRIPT_FILE, "the file %s is missing or empty" % SCRIPT_FILE)]
        after = [("[ -s %s ]" % RESULT_FILE, "the result file %s is missing or empty" % RESULT_FILE)]
        body.append(step_block(STEP, command, step_limit(self.settings["time"]), before=before, after=after,
                               clean=[RESULT_FILE]))
        return head + "\n".join(body)

    def _task(self, kind: str) -> dict:
        s = self.settings
        return {"kind": kind, "symbols": list(self.symbols),
                "settings": {k: s[k] for k in _PHYSICS}, "run": {k: s[k] for k in _RUN}}

    @staticmethod
    def _same(old: dict, new: dict) -> bool:
        """True if two tasks ask for the same numbers (coordinates compared to 1e-9 angstrom)."""
        for key in set(old) | set(new):
            if key == "run":
                continue
            a, b = old.get(key), new.get(key)
            if key in ("coords", "vectors", "geometries"):
                A, B = np.asarray(a if a is not None else [], dtype=float), np.asarray(b if b is not None else [], dtype=float)
                if A.shape != B.shape or (A.size and np.abs(A - B).max() > 1e-9):
                    return False
            elif a != b:
                return False
        return True

    def _start(self, tag: str, task: dict) -> None:
        job = self._job(tag)
        path = job.file(TASK_FILE)
        if os.path.exists(path):
            with open(path) as fh:
                old = json.load(fh)
            if not self._same(old, task):
                raise ValueError("the tag %r was already used for another calculation (%s); "
                                 "a tag names one geometry and one set of settings" % (tag, path))
        else:
            os.makedirs(job.folder, exist_ok=True)
            write_text(job.file(SCRIPT_FILE), JOB_SCRIPT)
            write_text(job.file(job.SCRIPT), self.script_text(tag))
            write_json(path, task)              # written last: its presence says that the folder is complete
        self._submit(job)

    def _submit(self, job: _PyscfJob) -> None:
        if job.submitted():
            return
        job.submit()
        write_json(job.file(ATTEMPTS_FILE), {"finished_at_submit": [finished_calculations(job.folder) or 0]})

    # ------------------------------------------------------------------ state, with continuation after a time limit
    def _status(self, job: _PyscfJob) -> JobStatus:
        st = job.status()
        if st.state == ABSENT:                  # the folder was written but the job was never submitted
            self._submit(job)
            return job.status()
        if st.state != FAILED or not st.timed_out:
            return st
        path = job.file(ATTEMPTS_FILE)
        history = [0]
        if os.path.exists(path):
            with open(path) as fh:
                history = list(json.load(fh).get("finished_at_submit", [0]))
        now = finished_calculations(job.folder) or 0
        limit = int(self.settings["max_attempts"])
        if now <= history[-1]:
            return JobStatus(FAILED, st.message + "; no calculation finished in this attempt, so the job was not "
                             "submitted again: raise the time limit (setting 'time', now %s)" % self.settings["time"],
                             timed_out=True)
        if len(history) >= limit:
            return JobStatus(FAILED, st.message + "; this was attempt %d of at most %d (setting 'max_attempts')"
                             % (len(history), limit), timed_out=True)
        job.resubmit()
        write_json(path, {"finished_at_submit": history + [now]})
        return job.status()

    # ------------------------------------------------------------------ repair by request
    def tags(self) -> List[str]:
        """The tags of all requests below the root (folders that hold a task file)."""
        out = []
        for folder, _dirs, files in os.walk(self.root):
            if TASK_FILE in files:
                out.append(os.path.relpath(folder, self.root).replace(os.sep, "/"))
        return sorted(out)

    def retry_failed(self) -> List[str]:
        """Submit again every job whose state is FAILED (after the cause was repaired, or after a failure of the
        cluster). A job that waits, runs or is done is not touched. The count of attempts after a time limit
        starts again. Returns the tags of the jobs that were submitted."""
        again = []
        for tag in self.tags():
            job = self._job(tag)
            if job.status().state != FAILED:
                continue
            job.resubmit()
            write_json(job.file(ATTEMPTS_FILE), {"finished_at_submit": [finished_calculations(job.folder) or 0]})
            again.append(tag)
        return again

    # ------------------------------------------------------------------ the surrogate interface
    def request(self, tag: str, coords: np.ndarray, hessian: bool = True, gradient: bool = True) -> None:
        """Start the calculation of a geometry (angstrom, atom order of ``symbols``). Does nothing if the
        tag exists with the same geometry; a tag that exists with another geometry is an error."""
        X = np.asarray(coords, dtype=float).reshape(-1, 3)
        if len(X) != len(self.symbols):
            raise ValueError("%d coordinate rows for %d atoms" % (len(X), len(self.symbols)))
        if not np.all(np.isfinite(X)):
            raise ValueError("the coordinates hold a value that is not a finite number")
        if hessian and len(self.vectors) == 0:
            raise ValueError("a Hessian in the search directions was asked for, but no direction was given")
        task = self._task("point")
        task.update({"coords": X.tolist(), "vectors": self.vectors.tolist(), "hessian": bool(hessian),
                     "gradient": bool(gradient), "fd_step": float(self.settings["fd_step"])})
        self._start(tag, task)

    def result(self, tag: str) -> Optional[SurrogateResult]:
        """The finished result, None while the job waits or runs (and for a tag that was never requested),
        or a result with ``failed=True`` and a message."""
        job = self._job(tag)
        if not os.path.exists(job.file(TASK_FILE)):
            return None
        st = self._status(job)
        if st.state == FAILED:
            return SurrogateResult(failed=True, message=st.message)
        if st.state != DONE:
            return None
        path = job.file(RESULT_FILE)
        n, d = len(self.symbols), len(self.vectors)
        try:
            data = _read_result(path)
            if data["kind"] != "point":
                raise ValueError("%s holds a %s result, not the result of a point" % (path, data["kind"]))
            res = SurrogateResult(energy=float(_finite_array(data["energy"], (), "energy", path)))
            if data.get("gradient") is not None:
                g = _finite_array(data["gradient"], (n, 3), "gradient", path)
                res.gradient = g * (HARTREE_TO_KCAL / BOHR_TO_ANGSTROM)
            if data.get("hessian_directions") is not None:
                K = _finite_array(data["hessian_directions"], (d, d), "hessian_directions", path)
                res.hessian_directions = 0.5 * (K + K.T) * HARTREE_TO_KCAL
            if data.get("hessian_asymmetry") is not None:
                res.message = ("largest asymmetry of the finite-difference Hessian before it was symmetrized: "
                               "%.3g kcal/mol/angstrom^2" % (float(data["hessian_asymmetry"]) * HARTREE_TO_KCAL))
        except (KeyError, TypeError, ValueError, OSError) as exc:
            return SurrogateResult(failed=True, message="cannot use %s: %s" % (path, exc))
        return res

    def details(self, tag: str) -> dict:
        """The raw content of the result file of a finished request (units as written by the job script:
        hartree, hartree/bohr, hartree/angstrom^2, seconds)."""
        return _read_result(self._job(tag).file(RESULT_FILE))

    def request_scan(self, tag: str, geometries: Sequence[np.ndarray]) -> None:
        """Start one job that computes the energy of every geometry of a list."""
        geoms: List[list] = []
        for i, g in enumerate(geometries):
            X = np.asarray(g, dtype=float).reshape(-1, 3)
            if len(X) != len(self.symbols) or not np.all(np.isfinite(X)):
                raise ValueError("geometry %d of the scan does not fit %d atoms or holds a value that is not finite"
                                 % (i, len(self.symbols)))
            geoms.append(X.tolist())
        if not geoms:
            raise ValueError("a scan needs at least one geometry")
        task = self._task("scan")
        task["geometries"] = geoms
        self._start(tag, task)

    def scan(self, tag: str) -> Optional[np.ndarray]:
        """Energies in hartree in the order of the request, or None while the job waits or runs (and for a
        tag that was never requested). A failed job raises RuntimeError with the reason."""
        job = self._job(tag)
        if not os.path.exists(job.file(TASK_FILE)):
            return None
        st = self._status(job)
        if st.state == FAILED:
            raise RuntimeError("surrogate scan %r failed: %s" % (tag, st.message))
        if st.state != DONE:
            return None
        path = job.file(RESULT_FILE)
        data = _read_result(path)
        if data["kind"] != "scan":
            raise RuntimeError("%s holds a %s result, not the result of a scan" % (path, data["kind"]))
        with open(job.file(TASK_FILE)) as fh:
            n = len(json.load(fh)["geometries"])
        return _finite_array(data.get("energies"), (n,), "energies", path)
