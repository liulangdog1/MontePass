"""The engine that runs the real QMC chain: PySCF, convert4qmc and QMCPACK through a batch system.

One point (one geometry) lives in the folder ``<root>/<key>``. :meth:`QmcpackEngine.submit`
writes the folder and starts a batch job. :meth:`QmcpackEngine.read` looks at the files of the
folder and at the state of the job, and returns the DMC blocks written so far. The engine never
waits: the caller calls ``read`` again later. Everything the engine knows is in files, so a new
process continues where the old one stopped.

Files of a point
----------------
    geom.xyz             the geometry in the atom order of the user
    order.json           the permutation that groups the atoms by element (QMCPACK needs grouped atoms)
    scf.py               PySCF script; its atoms are grouped by element
    qmc_settings.json    the values that the input writer reads inside the job
    <Element>.ccECP.xml  pseudopotential files (links or copies)
    job.sh               the batch script (``job_scf.sh`` too when the SCF runs as its own job)
    point.json           the record of the engine: settings, seeds, jobs, stop request, final state

Steps and marker files
----------------------
``job.sh`` runs the steps ``scf``, ``c4q``, ``vmc0`` (optional), ``opt`` and ``dmc`` in this
order. A step is skipped when ``<step>.done`` exists, so a job that is started again continues
with the first unfinished step. A step writes ``<step>.started``, runs under a time limit, checks
the exit code and the expected text of the output, and writes ``<step>.done`` or
``<step>.failed``. The file ``<step>.failed`` holds one line. It starts with ``RUN:`` or
``TIMEOUT:`` for a failure that a second attempt can repair (the engine submits the job again,
``max_attempts`` attempts in total), or with ``QUALITY:`` for a wrong result (the point is FAILED
at once).

Quality checks
--------------
* after ``scf``: the atoms of ``system.json`` are the grouped atoms of ``geom.xyz``, the electron
  count fits the charge and the spin, and every pseudopotential file has the valence charge that
  the SCF used;
* after ``vmc0``: the VMC energy with no Jastrow factor equals the energy of the determinant that
  PySCF computed (``e_determinant`` in ``system.json``) within 4 standard errors plus 0.005 Ha.
  A wrong atom order, orbital file or pseudopotential breaks this test;
* before ``dmc``: all numbers of the Jastrow optimization are finite, and the VMC energy with the
  chosen Jastrow factor is below the energy of the determinant by more than 2 standard errors;
* during ``dmc``: QMCPACK reports the seed offset that the input gave.

The job makes the first three checks itself and stops before it spends more computer time. The
engine makes the same checks again from the files when it reads the point.

Random seeds
------------
QMCPACK reduces the seed of the input modulo 1024 and calls the result the offset. It keeps a
table of the odd prime numbers 3, 5, 7, 11, ... MPI rank ``r`` seeds its main generator with the
prime at position ``offset + r``. Thread ``t`` of rank ``r`` seeds its generator with the prime at
position ``offset + ranks + threads * r + t`` (QMCPACK source file
``src/Utilities/RandomNumberControl.cpp``, functions ``put`` and ``make_children``). A run on R
ranks with T threads per rank therefore uses the W = R (T + 1) positions from ``offset`` to
``offset + W - 1``. The line "Range of prime numbers to use as seeds" of the QMCPACK output shows
only the first R T + 1 of them.

The engine gives every QMCPACK run an explicit seed, equal to its offset:

* a geometry with key K has the base offset h = (CRC-32 of K + ``settings["seed"]``) mod 1024;
* the DMC run of series s (1, 2, ...) of that geometry has the offset (h + (s - 1) W) mod 1024;
* the Jastrow optimization has (h - W) mod 1024, the zero-Jastrow check (h - 2 W) mod 1024.

Two series of one geometry are therefore at least W positions apart and share no prime. At most
floor(1024 / W) series fit: 16 on 32 ranks, 8 on 64 ranks, 4 on 128 ranks. A request for one more
series raises ValueError. The two set-up runs can share primes only with the last two possible
series; a set-up run gives no energy that is averaged with a DMC series. The seeds are recorded
in ``point.json``, and the engine compares the offset that QMCPACK prints with the seed it gave.

Stopping a run
--------------
:meth:`QmcpackEngine.stop` fixes the data of the point at the blocks that are complete at that
moment and sets the state STOPPED. If no block exists yet, the job is cancelled at once. If
blocks exist, the engine writes the stop files of QMCPACK into the folder, and QMCPACK ends
cleanly after its current block. The legacy DMC driver looks for ``<project id>.s001.STOP`` (its
file root plus ``.STOP``), the batched driver for ``<project id>.STOP``; the engine writes both.
The block that ends after the request is not used, so the data of a stopped point never change.
If the job is still in the queue after a grace time (3 expected block times, at least 120 s), a
later call of ``read`` or ``advance`` cancels it through the batch system.

Further runs of one geometry
----------------------------
A point with ``series > 1`` names its ``base_key``. It runs no SCF and no optimization. It waits
(state QUEUED) until the base point has ``scf.h5``, ``system.json`` and ``jastrow.json``, and then
runs only a DMC with its own project id (``dmc2``, ``dmc3``, ...) and its own seed in its own
folder. It reads the orbital file of the base point.

Points with several determinants
--------------------------------
With ``settings["trial"]["kind"] = "multidet"`` a point uses a trial wave function of several
determinants from Quantum Package 2 (module :mod:`multidet`). The point first waits (state QUEUED)
for the Quantum Package jobs of its geometry, which run in their own folder with their own batch
jobs. Then its job runs the steps ``scf``, ``mdc4q``, ``mdvmc``, ``opt`` and ``dmc``. Two quality
checks are added: the files of convert4qmc agree with Quantum Package and with the SCF, and the VMC
energy without Jastrow factor equals the energy that Quantum Package printed for the same wave
function. Without that setting nothing of this is used, and every file is written as before.
"""
from __future__ import annotations

import copy
import inspect
import json
import math
import os
import re
import shutil
import sys
import time
import zlib
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..geometry import normalize_symbol, read_xyz, write_xyz
from ..machine import Machine
from ..scheduler import base as sched
from ..scheduler.base import JobTable, Scheduler
from ..stats import read_scalar, reblock
from ..units import ANGSTROM_TO_BOHR
from . import multidet as MD
from . import qmcpack_inputs as Q
from .base import ABSENT, DONE, FAILED, QUEUED, RUNNING, STOPPED, Engine, PointData, PointSpec

SEED_MODULUS = 1024                      # QMCPACK uses the seed modulo this number
QMCPACK_SUCCESS = "QMCPACK execution completed successfully"
CONVERT_SUCCESS = "QMCGaussianParserBase::dump"
SCF_SUCCESS = "SCF_DONE"
QUALITY_EXIT = 3                         # exit code of a command of this module when a quality check fails
STEPS = ("scf", "c4q", "vmc0", "mdc4q", "mdvmc", "opt", "dmc")
_LONG_LIMIT = 30 * 86400                 # step limit in seconds when neither the step nor the job has one

DEFAULT_SETTINGS: Dict[str, object] = {
    # the molecule and its trial wave function
    "charge": 0, "spin": 0, "xc": "b3lyp", "basis": "ccecpccpvtz", "ecp": "ccecp", "grid_level": 3,
    "jastrow": {"j1_size": 6, "j1_rcut": 6.0, "j2_size": 8, "j2_rcut": 8.0},
    # the runs. "opt" may also hold "cycles_reuse": the number of cycles when the optimization starts
    # from the Jastrow factor of another point.
    "opt": {"cycles": 8, "samples": 100000},
    "dmc": {"timestep": 0.005, "steps": 500, "warmup_steps": 2000, "walkers": 2048},
    "vmc0": {},
    "vmc0_check": "first",               # first = only the first point of each molecule; always; never
    "jastrow_reuse": "none",             # centre = start from the Jastrow of the point settings["jastrow_start_key"]
    # jobs
    "max_attempts": 2,                   # a job that ends with a run failure is submitted again up to this count
    "slow_factor": 3.0,                  # a DMC run whose blocks take this many times longer than those of the other
                                         # points is cancelled and submitted again, once, away from its nodes (0: never)
    "slow_min_blocks": 5,                # blocks that a run must hold before its speed is judged
    "slow_min_points": 3,                # other points with blocks that the comparison needs
    "time_limits": {"scf": 7200, "c4q": 600, "vmc0": 1800, "opt": 7200, "dmc": 0},      # seconds; 0 = the job limit
    "job_time": None,                    # time limit of the QMC job (batch format); None = value of the machine profile
    "scf_job_time": None,                # the same for the SCF job of split_scf
    "split_scf": False,                  # run the SCF as its own small job before the QMC job
    "ranks": None,                       # MPI ranks of the QMCPACK runs; None = tasks of the machine profile
    "scf_threads": None,                 # threads of the SCF; None = cpus_per_task of the SCF job kind
    "pseudo_files": "symlink",           # symlink or copy
    "seed": 0,                           # shifts the seeds of all points
    # stopping and watching
    "stop_grace_blocks": 3.0, "stop_grace_seconds": 120.0,
    "poll_seconds": 30.0,                # the batch system is asked at most once in this time
    "settle_seconds": 60.0,              # a job that left the queue is judged only after this time
    "unknown_seconds": 600.0,            # a job that the batch system does not know is judged only after this time
    # limits of the quality checks
    "vmc0_sigmas": 4.0, "vmc0_tolerance": 0.005, "jastrow_sigmas": 2.0,
}


def _parameters(func, exclude: Sequence[str]) -> List[str]:
    return [n for n in inspect.signature(func).parameters if n not in exclude]


_ENGINE_SET = ("system", "coefficients", "jastrow", "start", "orbitals", "pseudo_dir", "seed", "project", "determinantset")
OPT_KEYS = _parameters(Q.opt_input, _ENGINE_SET)
DMC_KEYS = _parameters(Q.dmc_input, _ENGINE_SET + ("blocks",))
VMC0_KEYS = _parameters(Q.vmc_check_input, _ENGINE_SET + ("walkers",))


# ============================================================================= small tools
def merge_settings(*layers: Optional[dict]) -> dict:
    """Overlay dictionaries from left to right. Dictionaries inside them are merged key by key."""
    out: dict = {}
    for layer in layers:
        for key, value in (layer or {}).items():
            if isinstance(value, dict) and isinstance(out.get(key), dict):
                out[key] = merge_settings(out[key], value)
            else:
                out[key] = copy.deepcopy(value)
    return out


def parse_time(text: Optional[str]) -> Optional[int]:
    """Seconds of a batch time limit: ``MM``, ``MM:SS``, ``HH:MM:SS``, ``D-HH``, ``D-HH:MM`` or ``D-HH:MM:SS``.
    None for an empty value."""
    if text is None or not str(text).strip():
        return None
    t = str(text).strip()
    try:
        days = 0
        if "-" in t:
            d, t = t.split("-", 1)
            days = int(d)
            parts = [int(x) for x in t.split(":")]
            if not 1 <= len(parts) <= 3:
                raise ValueError(t)
            hours, minutes, seconds = (parts + [0, 0])[:3]
        else:
            parts = [int(x) for x in t.split(":")]
            if len(parts) == 1:
                hours, minutes, seconds = 0, parts[0], 0
            elif len(parts) == 2:
                hours, minutes, seconds = 0, parts[0], parts[1]
            elif len(parts) == 3:
                hours, minutes, seconds = parts
            else:
                raise ValueError(t)
    except ValueError:
        raise ValueError("cannot read the time limit %r" % (text,))
    return ((days * 24 + hours) * 60 + minutes) * 60 + seconds


def validate_settings(s: dict) -> None:
    """Raise ValueError for a setting that the engine cannot use."""
    for name in ("jastrow", "opt", "dmc", "vmc0", "time_limits"):
        if not isinstance(s.get(name), dict):
            raise ValueError("settings[%r] must be a dictionary" % name)
    if s["vmc0_check"] not in ("first", "always", "never"):
        raise ValueError("vmc0_check = %r; allowed: first, always, never" % (s["vmc0_check"],))
    if s["jastrow_reuse"] not in ("none", "centre"):
        raise ValueError("jastrow_reuse = %r; allowed: none, centre" % (s["jastrow_reuse"],))
    if s["pseudo_files"] not in ("symlink", "copy"):
        raise ValueError("pseudo_files = %r; allowed: symlink, copy" % (s["pseudo_files"],))
    if int(s["max_attempts"]) < 1:
        raise ValueError("max_attempts must be at least 1")
    if "blocks" in s["dmc"]:
        raise ValueError("the number of DMC blocks is PointSpec.blocks; remove 'blocks' from settings['dmc']")
    for name, allowed in (("opt", OPT_KEYS + ["cycles_reuse"]), ("dmc", DMC_KEYS), ("vmc0", VMC0_KEYS)):
        unknown = sorted(set(s[name]) - set(allowed))
        if unknown:
            raise ValueError("settings[%r] has unknown keys %s; allowed: %s" % (name, unknown, ", ".join(allowed)))
    unknown = sorted(set(s["time_limits"]) - set(STEPS))
    if unknown:
        raise ValueError("settings['time_limits'] has unknown steps %s; allowed: %s" % (unknown, ", ".join(STEPS)))
    for key in ("timestep", "steps", "warmup_steps", "walkers"):
        if key not in s["dmc"]:
            raise ValueError("settings['dmc'] lacks %r" % key)
    if int(s["dmc"]["walkers"]) < 1 or int(s["dmc"]["steps"]) < 1 or float(s["dmc"]["timestep"]) <= 0:
        raise ValueError("settings['dmc']: walkers, steps and timestep must be positive")
    parse_time(s["job_time"])
    parse_time(s["scf_job_time"])
    MD.validate_trial(MD.trial_of(s), s.get("charge", 0), s.get("spin", 0))


def _jsonable(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError("cannot write an object of type %s into a JSON file" % type(obj).__name__)


def _write_text(path: str, text: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        fh.write(text)
    os.replace(tmp, path)


def _write_json(path: str, obj) -> None:
    _write_text(path, json.dumps(obj, indent=1, default=_jsonable) + "\n")


def _read_json(path: str):
    with open(path) as fh:
        return json.load(fh)


def _first_line(path: str) -> Optional[str]:
    """First line of a small file, or None when the file does not exist."""
    try:
        with open(path) as fh:
            return fh.readline().strip()
    except OSError:
        return None


def _marker_time(path: str) -> Optional[float]:
    """Time written into a ``.started`` or ``.done`` file (seconds since 1970); the time of the file itself
    when it holds no number; None when the file does not exist."""
    line = _first_line(path)
    if line is None:
        return None
    try:
        return float(line.split()[0])
    except (ValueError, IndexError):
        try:
            return os.path.getmtime(path)
        except OSError:
            return None


def _finite(value) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


# ============================================================================= random seeds
def seed_width(ranks: int, threads: int = 1) -> int:
    """Number of consecutive prime positions that one QMCPACK run uses."""
    return int(ranks) * (int(threads) + 1)


def seed_window(seed: int, ranks: int, threads: int = 1) -> Tuple[int, int]:
    """First and last position in the prime table that a run with this seed uses."""
    first = int(seed) % SEED_MODULUS
    return first, first + seed_width(ranks, threads) - 1


def windows_overlap(a: Sequence[int], b: Sequence[int]) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


def max_series(ranks: int, threads: int = 1) -> int:
    """Largest number of runs of one geometry that can have prime numbers of their own."""
    return SEED_MODULUS // seed_width(ranks, threads)


def point_seeds(base_key: str, series: int, ranks: int, threads: int = 1, shift: int = 0) -> Dict[str, int]:
    """Seeds of the QMCPACK runs of one point (see "Random seeds" in the module text). ``base_key`` is the
    key of the series-1 point of the geometry."""
    width = seed_width(ranks, threads)
    if series < 1:
        raise ValueError("series must be 1 or larger, not %d" % series)
    if series > SEED_MODULUS // width:
        raise ValueError("series %d of %s: QMCPACK has %d seed offsets and one run on %d ranks uses %d of them, so at "
                         "most %d runs of one geometry can have prime numbers of their own"
                         % (series, base_key, SEED_MODULUS, ranks, width, SEED_MODULUS // width))
    h = (zlib.crc32(base_key.encode("utf-8")) + int(shift)) % SEED_MODULUS
    seeds = {"dmc": (h + (series - 1) * width) % SEED_MODULUS}
    if series == 1:
        seeds["opt"] = (h - width) % SEED_MODULUS
        seeds["vmc0"] = (h - 2 * width) % SEED_MODULUS
    return seeds


def prime_table(count: int) -> List[int]:
    """The first ``count`` odd primes: the table QMCPACK takes its seeds from."""
    out: List[int] = []
    cand = 3
    while len(out) < count:
        if all(cand % p for p in out if p * p <= cand):
            out.append(cand)
        cand += 2
    return out


def printed_prime_range(seed: int, ranks: int, threads: int = 1) -> Tuple[int, int]:
    """The two numbers of the QMCPACK output line "Range of prime numbers to use as seeds" for this seed."""
    offset = int(seed) % SEED_MODULUS
    table = prime_table(offset + ranks * threads + 1)
    return table[offset], table[offset + ranks * threads]


_OFFSET_RE = re.compile(r"Offset for the random number seeds (from input file \(mod 1024\)|based on time):\s*(\d+)")
_RANGE_RE = re.compile(r"Range of prime numbers to use as seeds over processors and threads\s*=\s*(\d+)-(\d+)")


def read_seed_report(path: str) -> Optional[dict]:
    """What a QMCPACK output file says about its seeds: ``{"offset", "from_input", "prime_range"}``.
    None when the file does not exist or does not hold the offset line yet."""
    try:
        with open(path, errors="replace") as fh:
            head = fh.read(20000)
    except OSError:
        return None
    m = _OFFSET_RE.search(head)
    if not m:
        return None
    out = {"offset": int(m.group(2)), "from_input": m.group(1).startswith("from input"), "prime_range": None}
    r = _RANGE_RE.search(head)
    if r:
        out["prime_range"] = [int(r.group(1)), int(r.group(2))]
    return out


# ============================================================================= traces
def read_trace(path: str) -> Dict[str, np.ndarray]:
    """Columns of a ``scalar.dat`` file that may still be growing.

    Only lines that end with a line break are used. A last line that is still being written can
    have the right number of fields and a last field that is cut short; such a line is dropped.
    """
    with open(path, "rb") as fh:
        raw = fh.read()
    complete = sum(1 for ln in raw.split(b"\n")[:-1] if ln.strip() and not ln.lstrip().startswith(b"#"))
    cols = read_scalar(path)
    return {name: values[:complete] for name, values in cols.items()}


def trace_energy(cols: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Block energies (hartree) and block weights of a trace."""
    energy = cols["LocalEnergy"]
    weight = cols["BlockWeight"] if "BlockWeight" in cols else np.ones_like(energy)
    return energy, weight


# ============================================================================= quality checks
def check_system(folder: str = ".", charge: int = 0, spin: int = 0) -> dict:
    """Check ``system.json`` (written by the SCF) against the geometry, the charge, the spin and the
    pseudopotential files of the folder. Returns ``{"ok": bool, "message": str, ...}``."""
    system = Q.load_system(os.path.join(folder, "system.json"))
    order = _read_json(os.path.join(folder, "order.json"))
    symbols, coords = read_xyz(os.path.join(folder, "geom.xyz"))
    perm = [int(i) for i in order["order"]]
    grouped = [symbols[i] for i in perm]
    out: Dict[str, object] = {"check": "system", "ok": False}
    problems = []
    if list(system["symbols"]) != grouped:
        problems.append("the atoms of system.json (%s) are not the grouped atoms of geom.xyz (%s)"
                        % (" ".join(system["symbols"]), " ".join(grouped)))
    else:
        shift = float(np.abs(np.asarray(system["coords_bohr"], dtype=float) - coords[perm] * ANGSTROM_TO_BOHR).max())
        out["max_shift_bohr"] = shift
        if shift > 1e-5:
            problems.append("an atom of system.json is %.2e bohr away from its place in geom.xyz" % shift)
    if int(system["nup"]) - int(system["ndn"]) != int(spin):
        problems.append("system.json has %d up and %d down electrons, but the requested spin (2S) is %d"
                        % (system["nup"], system["ndn"], spin))
    try:
        Q.check_pseudopotentials(dict(system, charge=int(charge)), folder)
    except ValueError as exc:
        problems.append(str(exc))
    out["ok"] = not problems
    out["message"] = ("; ".join(problems) if problems else
                      "system.json agrees with the geometry, the charge, the spin and the pseudopotential files")
    return out


def check_zero_jastrow(folder: str = ".", sigmas: float = 4.0, tolerance: float = 0.005,
                       trace: str = "vmc0.s000.scalar.dat") -> dict:
    """Compare the VMC energy without Jastrow factor with the energy of the determinant from PySCF.
    The check fails when they differ by more than ``sigmas`` standard errors plus ``tolerance`` hartree."""
    out: Dict[str, object] = {"check": "vmc0", "ok": False}
    system = _read_json(os.path.join(folder, "system.json"))
    if not _finite(system.get("e_determinant")):
        out["message"] = "system.json holds no e_determinant, so the zero-Jastrow check cannot be made"
        return out
    energy, weight = trace_energy(read_trace(os.path.join(folder, trace)))
    if len(energy) < 8 or not np.all(np.isfinite(energy)):
        out["message"] = "%s holds %d blocks with finite energies; 8 are needed for an error bar" % (
            trace, int(np.sum(np.isfinite(energy))))
        return out
    mean, err = reblock(energy, weight)
    e_det = float(system["e_determinant"])
    limit = float(sigmas) * err + float(tolerance)
    out.update({"energy": mean, "error": err, "e_determinant": e_det, "difference": mean - e_det, "limit": limit,
                "blocks": int(len(energy)), "ok": bool(abs(mean - e_det) <= limit)})
    text = ("zero-Jastrow VMC energy %.6f +/- %.6f Ha, energy of the determinant %.6f Ha, difference %+.6f Ha, "
            "limit %.6f Ha" % (mean, err, e_det, mean - e_det, limit))
    if not out["ok"]:
        text += ": the QMCPACK input does not describe the determinant of the SCF (atom order, orbital file or pseudopotentials)"
    out["message"] = text
    return out


def check_jastrow(folder: str = ".", sigmas: float = 2.0, reference: Optional[float] = None) -> dict:
    """Check the result of the Jastrow optimization in ``jastrow.json``: all numbers finite, and the VMC
    energy below the energy of the determinant by more than ``sigmas`` standard errors. ``reference`` is the
    energy without Jastrow factor of a trial function that is not the determinant of ``system.json``."""
    out: Dict[str, object] = {"check": "jastrow", "ok": False}
    data = _read_json(os.path.join(folder, "jastrow.json"))
    bad = [cid for cid, values in data["coefficients"].items() if not all(_finite(v) for v in values)]
    if bad:
        out["message"] = "the Jastrow coefficients %s hold numbers that are not finite" % ", ".join(sorted(bad))
        return out
    sel = data.get("selection", {})
    broken = [c.get("series") for c in sel.get("cycles", [])
              if not (_finite(c.get("energy")) and _finite(c.get("variance")))]
    if broken:
        out["message"] = "optimization cycle(s) %s gave an energy or a variance that is not finite" % broken
        return out
    if "energy" not in sel:
        out.update({"ok": True, "message": "the Jastrow coefficients are finite (taken from %s)" % sel.get("source")})
        return out
    if not (_finite(sel["energy"]) and _finite(sel.get("error")) and _finite(sel.get("variance")) and sel["variance"] > 0):
        out["message"] = "the chosen optimization cycle has an energy, an error or a variance that is not finite"
        return out
    out.update({"energy": float(sel["energy"]), "error": float(sel["error"]), "variance": float(sel["variance"])})
    system_file = os.path.join(folder, "system.json")
    e_det = _read_json(system_file).get("e_determinant") if os.path.exists(system_file) else None
    if reference is not None:
        e_det = reference
    if not _finite(e_det):
        out.update({"ok": True, "message": "all numbers of the optimization are finite; VMC energy %.6f +/- %.6f Ha "
                                           "(system.json holds no e_determinant to compare with)"
                                           % (sel["energy"], sel["error"])})
        return out
    gain = float(e_det) - float(sel["energy"])
    out.update({"e_determinant": float(e_det), "gain": gain, "ok": bool(gain > float(sigmas) * float(sel["error"]))})
    text = ("VMC energy with the Jastrow factor %.6f +/- %.6f Ha, energy of the determinant %.6f Ha, gain %.6f Ha"
            % (sel["energy"], sel["error"], float(e_det), gain))
    if not out["ok"]:
        text += ": the Jastrow factor does not lower the energy by more than %g standard errors" % sigmas
    out["message"] = text
    return out


# ============================================================================= the batch script
_SHELL_TOOLS = r'''
# ---------------------------------------------------------------------------------------------
# Steps of this point. A step is skipped when <step>.done exists. A step writes <step>.started,
# runs under a time limit, checks the exit code and the expected output, and then writes
# <step>.done or <step>.failed (one line that says why).
# ---------------------------------------------------------------------------------------------
say() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*"; }
mark() { echo "$2" > "$1.tmp" && mv -f "$1.tmp" "$1"; }
begin_step() {
    STEP="$1"
    if [ -e "$STEP.done" ]; then say "step $STEP: done before, skipped"; return 1; fi
    if [ -e stop.requested ]; then say "step $STEP: not started, a stop was requested"; exit 0; fi
    rm -f "$STEP.failed"
    mark "$STEP.started" "$(date +%s)"
    say "step $STEP: started"
    return 0
}
end_step() { mark "$STEP.done" "$(date +%s)"; say "step $STEP: done"; }
fail_step() {
    mark "$STEP.failed" "$*"
    say "step $STEP: FAILED: $*"
    exit 1
}
# run_limited SECONDS OUTPUT ERRORS COMMAND ...   (ERRORS "-" = into OUTPUT). The exit code is left in RC.
run_limited() {
    LIMIT="$1"; run_out="$2"; run_err="$3"; shift 3
    WHAT="$*"
    if [ "$run_err" = "-" ]; then
        run_err="$run_out"
        TIMEOUT_COMMAND -k 30 "$LIMIT" "$@" > "$run_out" 2>&1
        RC=$?
    else
        TIMEOUT_COMMAND -k 30 "$LIMIT" "$@" > "$run_out" 2> "$run_err"
        RC=$?
    fi
}
# check_run: the step fails unless the last command ended with exit code 0
check_run() {
    if [ "$RC" -eq 124 ] || [ "$RC" -eq 137 ]; then fail_step "TIMEOUT: no end within $LIMIT s: $WHAT"; fi
    if [ "$RC" -ne 0 ]; then
        hint=""
        if grep -q -E "UCX|MPI_INIT" "$run_err" 2>/dev/null; then hint=" [MPI start failure, see $run_err]"; fi
        fail_step "RUN: exit code $RC: $WHAT$hint"
    fi
}
need_text() { grep -q -- "$2" "$1" 2>/dev/null || fail_step "RUN: the text '$2' is not in $1"; }
need_file() { [ -s "$1" ] || fail_step "RUN: the file $1 is missing or empty"; }
# package COMMAND: a command of the package. Exit code QUALITY_EXIT = a quality check failed; the last line of its output says why.
package() {
    run_limited 900 "$1.out" - PYTHON_COMMAND -m montepass.engine.qmcpack "$1"
    if [ "$RC" -eq QUALITY_EXIT ]; then fail_step "QUALITY: $(tail -n 1 "$1.out")"; fi
    check_run
}
count_blocks() { awk '!/^#/ && NF > 0 { n++ } END { print n + 0 }' "$1" 2>/dev/null; }
say "job ${SLURM_JOB_ID:-local} on $(hostname), nodes ${SLURM_JOB_NODELIST:-$(hostname)}"
echo "$(date +%s) job=${SLURM_JOB_ID:-local} nodes=${SLURM_JOB_NODELIST:-$(hostname)}" >> job.info
'''


def _dmc_step(project: str, blocks: int, limit: int, launch: str, qmcpack: str, write_input: bool) -> str:
    trace = "%s.s001.scalar.dat" % project
    lines = ["if begin_step dmc; then",
             "    rm -f %s.s000.scalar.dat %s      # blocks of an earlier attempt are not blocks of this run" % (project, trace)]
    if write_input:
        lines += ["    package write-dmc",
                  "    need_file jastrow.json"]
    lines += ["    need_file %s.in.xml" % project,
              "    run_limited %d %s.out %s.err %s %s %s.in.xml" % (limit, project, project, launch, qmcpack, project),
              "    blocks=$(count_blocks %s); blocks=${blocks:-0}" % trace,
              "    if [ -e stop.requested ]; then",
              "        mark dmc.stopped \"$blocks\"",
              "        say \"step dmc: stopped on request after $blocks blocks\"",
              "        exit 0",
              "    fi",
              "    check_run",
              "    need_text %s.out \"%s\"" % (project, QMCPACK_SUCCESS),
              "    if [ \"$blocks\" -lt %d ]; then fail_step \"RUN: %s holds $blocks of %d blocks\"; fi" % (blocks, trace, blocks),
              "    end_step",
              "fi"]
    return "\n".join(lines)


def steps_script(steps: Sequence[str], project: str, blocks: int, limits: Dict[str, int], launch: str,
                 python: str = "python", qmcpack: str = "qmcpack", convert4qmc: str = "convert4qmc",
                 timeout: str = "timeout", write_dmc_input: bool = True, md: Optional[dict] = None) -> str:
    """Shell text of the steps of one job: everything after the header of the machine profile.

    ``limits`` gives the time limit of every step in seconds. ``launch`` is the prefix that starts
    QMCPACK in parallel. With ``write_dmc_input=False`` the DMC input file must exist already
    (further series of a geometry). ``md`` holds what the steps ``mdc4q`` and ``mdvmc`` need: the h5 file
    that Quantum Package exported (``h5``), its number of determinants (``ndet``) and the command
    ``determinants_tools``."""
    for step in steps:
        if step not in STEPS:
            raise ValueError("unknown step %r; allowed: %s" % (step, ", ".join(STEPS)))
    several = any(step in MD.QMC_STEPS for step in steps)
    if several and not md:
        raise ValueError("the steps mdc4q and mdvmc need the facts of the Quantum Package export")
    parts = [(_SHELL_TOOLS + (MD.MD_PACKAGE_FUNCTION if several else "")).replace("TIMEOUT_COMMAND", timeout)
             .replace("PYTHON_COMMAND", python).replace("QUALITY_EXIT", str(QUALITY_EXIT))]
    if "scf" in steps:
        parts.append("\n".join([
            "if begin_step scf; then",
            "    run_limited %d scf.out - %s -u scf.py" % (limits["scf"], python),
            "    check_run",
            "    need_text scf.out %s" % SCF_SUCCESS,
            "    need_file scf.h5; need_file energy.dat; need_file system.json",
            "    package check-system",
            "    end_step",
            "fi"]))
    if "c4q" in steps:
        parts.append("\n".join([
            "if begin_step c4q; then",
            "    run_limited %d c4q.out - %s -prefix c4q -orbitals scf.h5" % (limits["c4q"], convert4qmc),
            "    check_run",
            "    need_text c4q.out \"%s\"" % CONVERT_SUCCESS,
            "    need_file c4q.structure.xml; need_file c4q.wfj.xml",
            "    end_step",
            "fi"]))
    if "vmc0" in steps:
        parts.append("\n".join([
            "if begin_step vmc0; then",
            "    rm -f vmc0.s000.scalar.dat",
            "    package write-vmc0",
            "    need_file vmc0.in.xml",
            "    run_limited %d vmc0.out vmc0.err %s %s vmc0.in.xml" % (limits["vmc0"], launch, qmcpack),
            "    check_run",
            "    need_text vmc0.out \"%s\"" % QMCPACK_SUCCESS,
            "    need_file vmc0.s000.scalar.dat",
            "    package check-vmc0",
            "    end_step",
            "fi"]))
    if several:
        parts += MD.qmc_steps_text(steps, limits, launch, python=python, qmcpack=qmcpack, convert4qmc=convert4qmc,
                                   determinants_tools=md["determinants_tools"], h5_source=md["h5"], ndet=int(md["ndet"]))
    if "opt" in steps:
        parts.append("\n".join([
            "if begin_step opt; then",
            "    rm -f opt.s[0-9][0-9][0-9].scalar.dat opt.s[0-9][0-9][0-9].opt.xml      # cycles of an earlier attempt",
            "    package write-opt",
            "    need_file opt.in.xml",
            "    run_limited %d opt.out opt.err %s %s opt.in.xml" % (limits["opt"], launch, qmcpack),
            "    check_run",
            "    need_text opt.out \"%s\"" % QMCPACK_SUCCESS,
            "    end_step",
            "fi"]))
    if "dmc" in steps:
        parts.append(_dmc_step(project, blocks, limits["dmc"], launch, qmcpack, write_dmc_input))
    parts.append('say "all steps of this job are done"\nexit 0\n')
    return "\n".join(parts)


# ============================================================================= the engine
_KEY_PART = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+\-]*$")
WAITING, ACTIVE, FINAL = "WAITING", "ACTIVE", "FINAL"          # status of the record of a point
_IN_QUEUE = (sched.PENDING, sched.RUNNING)


def _check_key(key: str) -> None:
    if not key or any(not _KEY_PART.match(part) for part in str(key).split("/")):
        raise ValueError("point key %r: use letters, digits and the characters _ . + - in parts separated by /" % (key,))


def _job_name(key: str) -> str:
    return ("q_" + key.replace("/", "_"))[-50:]


class QmcpackEngine(Engine):
    """Engine for real QMC runs. The module text describes the files, the steps, the seeds and the stop rule.

    Parameters
    ----------
    root        folder of all points; the point with key K lives in ``<root>/<K>``.
    machine     a :class:`~montepass.machine.Machine` or the path of a machine profile.
    scheduler   a :class:`~montepass.scheduler.base.Scheduler`; None = the scheduler of the machine profile.
    settings    project-wide QMC settings, laid over :data:`DEFAULT_SETTINGS`. ``PointSpec.settings`` is laid
                over them for one point. ``PointSpec.blocks`` is the number of DMC blocks.
    job_table   record of all submitted jobs; None = the file ``<root>/jobs.json``.
    clock       function that returns the time in seconds; None = ``time.time`` (tests pass their own).
    """

    def __init__(self, root: str, machine, scheduler: Optional[Scheduler] = None, settings: Optional[dict] = None,
                 job_table: Optional[JobTable] = None, clock: Optional[Callable[[], float]] = None):
        self.root = os.path.abspath(root)
        self.machine = machine if isinstance(machine, Machine) else Machine(str(machine))
        self.scheduler = scheduler if scheduler is not None else self.machine.scheduler()
        self.settings = merge_settings(DEFAULT_SETTINGS, settings)
        validate_settings(self.settings)
        for kind in ("qmc", "scf"):
            parse_time(self.machine.job(kind).get("time"))          # a time limit that cannot be read fails here
        os.makedirs(self.root, exist_ok=True)
        self.jobs = job_table if job_table is not None else JobTable(os.path.join(self.root, "jobs.json"))
        self.clock = clock or time.time
        self._points: Dict[str, dict] = {}
        self._trace_cache: Dict[str, tuple] = {}
        self._snapshot: Dict[str, str] = {}
        self._snapshot_time: Optional[float] = None
        self._fresh: set = set()            # jobs submitted after the last question to the batch system
        self._watch: set = set()            # jobs whose state the engine still needs
        self._preloaded = False
        self._speed: Dict[str, float] = {}  # core-seconds per walker and step of the points that ran at a normal speed
        self._speed_kind: Dict[str, str] = {}       # kind of wave function of these points (see _wave_kind)
        self._speed_scanned = False
        self._qp2_runs: Dict[str, MD.Qp2Run] = {}      # Quantum Package calculations of points with several determinants
        self._qp2_jobs: Optional[JobTable] = None
        self._state_file = os.path.join(self.root, "engine.json")
        self._state = {"version": 1, "vmc0": {}, "waiting": [], "stops": []}
        if os.path.exists(self._state_file):
            self._state.update(_read_json(self._state_file))

    # ------------------------------------------------------------------ files and records
    def folder(self, key: str) -> str:
        """Folder of a point."""
        return os.path.join(self.root, *str(key).split("/"))

    def keys(self) -> List[str]:
        """Keys of all points that this engine has a job or a record for."""
        return sorted(set(self.jobs.jobs) | set(self._points) | set(self._state.get("trial_points", [])))

    def _save_state(self) -> None:
        _write_json(self._state_file, self._state)

    def _list(self, name: str, key: str, present: bool) -> None:
        """Put a key into, or take it out of, one of the lists of ``engine.json``."""
        if present != (key in self._state[name]):
            if present:
                self._state[name].append(key)
            else:
                self._state[name].remove(key)
            self._save_state()

    def _load(self, key: str) -> Optional[dict]:
        if key in self._points:
            return self._points[key]
        path = os.path.join(self.folder(key), "point.json")
        if not os.path.exists(path):
            return None
        p = _read_json(path)
        self._points[key] = p
        if p["status"] != FINAL or (p.get("stop") and not p["stop"].get("settled")):
            self._watch.update(j for j in p["jobs"].values() if j)
        return p

    def _save(self, p: dict) -> None:
        _write_json(os.path.join(self.folder(p["key"]), "point.json"), p)

    def _exists(self, p: dict, name: str) -> bool:
        return os.path.exists(os.path.join(self.folder(p["key"]), name))

    def _note(self, p: dict, text: str) -> None:
        stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.clock()))
        p.setdefault("notes", []).append("%s %s" % (stamp, text))

    # ------------------------------------------------------------------ the batch system
    def _poll(self, force: bool = False) -> None:
        """Ask the batch system for the states of all watched jobs, at most once per ``poll_seconds``."""
        now = self.clock()
        if not self._preloaded:
            self._preloaded = True
            for key in sorted(set(self.jobs.jobs) | set(self._state.get("trial_points", []))):
                self._load(key)
        if (not force and self._snapshot_time is not None
                and now - self._snapshot_time < float(self.settings["poll_seconds"])):
            return
        ids = sorted(self._watch)
        self._snapshot = dict(self.scheduler.states(ids)) if ids else {}
        self._snapshot_time = now
        self._fresh.clear()

    def _job_state(self, job_id: Optional[str]) -> str:
        if not job_id:
            return sched.UNKNOWN
        if job_id not in self._snapshot:
            if job_id in self._fresh:
                return sched.PENDING            # submitted after the last question: it waits in the queue
            self._watch.add(job_id)
            self._poll(force=True)
            self._snapshot.setdefault(job_id, sched.UNKNOWN)
        return self._snapshot[job_id]

    def _submit_job(self, p: dict, script: str, kind: str, after: Optional[List[str]]) -> str:
        folder = self.folder(p["key"])
        name = _job_name(p["key"]) + ("_scf" if kind == "scf" else "")
        job_id = str(self.scheduler.submit(os.path.join(folder, script), folder, name, after or None))
        self.jobs.add(p["key"], job_id, folder, name, kind)
        if p["jobs"].get(kind):
            self._watch.discard(p["jobs"][kind])
        p["jobs"][kind] = job_id
        p["job_ids"].append(job_id)
        self._fresh.add(job_id)
        self._watch.add(job_id)
        self._save(p)
        return job_id

    def _cancel_jobs(self, p: dict) -> List[str]:
        """Cancel every job of the point that is still in the queue. Returns the cancelled job ids."""
        cancelled = []
        for job_id in p["jobs"].values():
            if job_id and self._job_state(job_id) in _IN_QUEUE:
                self.scheduler.cancel(job_id)
                self._snapshot[job_id] = sched.CANCELLED
                self._fresh.discard(job_id)
                cancelled.append(job_id)
        return cancelled

    # ------------------------------------------------------------------ submit
    def submit(self, spec: PointSpec) -> None:
        """Create the folder of the point and start its job. A key that exists already is left as it is; the
        request must then name the same geometry and the same trial wave function (one determinant, or the same
        cut of the same multi-determinant wave function), because both decide what the energy of the key means."""
        p = self._load(str(spec.key))
        if p is None:
            p = self._prepare(spec)
        else:
            self._same_geometry(spec)
            self._same_trial(spec, p.get("trial"), "exists", inherit=int(p.get("series", 1)) > 1)
        if p["status"] == WAITING:
            self._try_launch(p, raise_errors=True)

    def _same_geometry(self, spec: PointSpec) -> None:
        """A point that exists is asked for again: it must be the same geometry. A search whose state does not
        fit the points on disk (for example a state file from another run in this folder) would otherwise read
        the energies of other geometries without a message."""
        path = os.path.join(self.folder(str(spec.key)), "geom.xyz")
        if not os.path.exists(path):
            return
        _symbols, old = read_xyz(path)
        new = np.asarray(spec.coords, dtype=float).reshape(-1, 3)
        if old.shape != new.shape or float(np.abs(old - new).max()) > 1e-5:
            raise ValueError("point %s exists with another geometry (largest difference %.2e angstrom): the state of "
                             "the search does not fit the points in %s" % (
                                 spec.key, float(np.abs(old - new).max()) if old.shape == new.shape else float("nan"),
                                 self.root))

    _TRIAL_RULE = ("kind", "orbitals", "cutoff", "n_det_max", "qp2_basis", "qp2_pseudo", "import_ezfio", "label")

    def _same_trial(self, spec: PointSpec, stored: Optional[dict], what: str, inherit: bool = False) -> None:
        """A point keeps the trial wave function it was created with. A request for another one (another cutoff,
        one determinant instead of several) must not get the old energies without a message.

        ``inherit``: the request is a further series. Such a request need not name a trial wave function (it reads
        the files of its base point); only a request that names one itself is compared."""
        if inherit and not (spec.settings or {}).get("trial"):
            return
        asked = MD.trial_of(merge_settings(self.settings, spec.settings))

        def rule(t):
            return None if t is None else {k: t.get(k) for k in self._TRIAL_RULE}
        if rule(asked) != rule(stored):
            raise ValueError("point %s %s with another trial wave function (stored: %s; asked: %s): a new wave function "
                             "needs a new key" % (spec.key, what, rule(stored), rule(asked)))

    def _prepare(self, spec: PointSpec) -> dict:
        """Check the request, write the files that do not depend on other points, and write the record."""
        key = str(spec.key)
        _check_key(key)
        symbols = [normalize_symbol(sym) for sym in spec.symbols]
        coords = np.asarray(spec.coords, dtype=float).reshape(-1, 3)
        if len(symbols) != len(coords) or not symbols:
            raise ValueError("point %s: %d symbols and %d coordinate rows" % (key, len(symbols), len(coords)))
        if not np.all(np.isfinite(coords)):
            raise ValueError("point %s: a coordinate is not finite" % key)
        blocks, series = int(spec.blocks), int(spec.series)
        if blocks < 1:
            raise ValueError("point %s: blocks = %d; at least 1 is needed" % (key, blocks))
        s = merge_settings(self.settings, spec.settings)
        validate_settings(s)
        trial = MD.trial_of(s)                      # None for the single determinant of the default chain
        ranks = int(s["ranks"] or self.machine.tasks("qmc"))
        if ranks < 1:
            raise ValueError("point %s: %d MPI ranks" % (key, ranks))
        requested = int(s["dmc"]["walkers"])
        walkers = int(math.ceil(requested / float(ranks))) * ranks        # the same number of walkers on every rank
        base, base_key, signature = None, None, None
        if series > 1:
            if not spec.base_key or str(spec.base_key) == key:
                raise ValueError("point %s: series %d needs the key of its series-1 point in base_key" % (key, series))
            base_key = str(spec.base_key)
            base = self._load(base_key)
            if base is None or base["series"] != 1:
                raise ValueError("point %s: its base point %s is not a submitted series-1 point" % (key, base_key))
            base_symbols, base_coords = read_xyz(os.path.join(self.folder(base_key), "geom.xyz"))
            if list(base_symbols) != symbols or np.abs(base_coords - coords).max() > 1e-6:
                raise ValueError("point %s: a further series must have the geometry of its base point %s" % (key, base_key))
            other = base.get("series_keys", {}).get(str(series))
            if other and other != key:
                raise ValueError("point %s: series %d of %s exists already as %s" % (key, series, base_key, other))
            self._same_trial(spec, base.get("trial"), "is a further series of %s, which was made" % base_key, inherit=True)
            trial = copy.deepcopy(base.get("trial"))    # a further series uses the trial wave function of its base point
        elif spec.base_key and str(spec.base_key) != key:
            raise ValueError("point %s: base_key is given but series is 1" % key)
        seeds = point_seeds(base_key or key, series, ranks, 1, int(s["seed"]))
        if series > 1:
            mine = seed_window(seeds["dmc"], ranks)
            siblings = [base_key] + [k for n, k in base.get("series_keys", {}).items() if int(n) != series]
            for other_key in siblings:
                q = self._load(other_key)
                if q is not None and windows_overlap(mine, seed_window(q["seeds"]["dmc"], q["ranks"])):
                    raise ValueError("point %s: its prime numbers (positions %d to %d) overlap with those of %s; use the "
                                     "same number of MPI ranks for all series of one geometry" % (key, mine[0], mine[1], other_key))
        if not self.machine.pseudo_dir:
            raise FileNotFoundError("the machine profile %s names no pseudopotential folder ([pseudopotentials] dir)"
                                    % self.machine.path)
        sources = {}
        for el in symbols:
            src = os.path.join(self.machine.pseudo_dir, "%s.ccECP.xml" % el)
            if el not in sources and not os.path.isfile(src):
                raise FileNotFoundError("pseudopotential file missing for element %s: %s (folder [pseudopotentials] dir "
                                        "of the machine profile %s)" % (el, src, self.machine.path))
            sources[el] = src
        # which steps this point runs, and from which Jastrow factor its optimization starts
        start_key, notes = None, []
        if series > 1:
            steps = ["dmc"]
        elif trial is not None:
            problems = self.machine.check_qp2()
            if problems:
                raise ValueError("point %s asks for a multi-determinant trial wave function, but the machine profile %s "
                                 "cannot run Quantum Package 2: %s" % (key, self.machine.path, "; ".join(problems)))
            steps = ["scf", "mdc4q", "mdvmc", "opt", "dmc"]
        else:
            signature = json.dumps([symbols, int(s["charge"]), int(s["spin"]), s["xc"], s["basis"], s["ecp"]])
            check = s["vmc0_check"] == "always" or (s["vmc0_check"] == "first" and signature not in self._state["vmc0"])
            steps = ["scf", "c4q"] + (["vmc0"] if check else []) + ["opt", "dmc"]
            if s["jastrow_reuse"] == "centre":
                start_key = s.get("jastrow_start_key")
                if start_key in (None, "", key):
                    if start_key is None and spec.role == "scan":
                        notes.append("jastrow_reuse = centre but the point names no jastrow_start_key: "
                                     "the optimization starts from zero")
                    start_key = None
                elif self._load(str(start_key)) is None:
                    raise ValueError("point %s: jastrow_start_key %s is not a submitted point; submit that point first"
                                     % (key, start_key))
        # the files
        folder = self.folder(key)
        if os.path.exists(os.path.join(folder, "qp2.json")):
            raise ValueError("point %s: its folder %s holds Quantum Package calculations of another point" % (key, folder))
        os.makedirs(folder, exist_ok=True)
        perm = Q.species_order(symbols)
        grouped = [symbols[i] for i in perm]
        write_xyz(os.path.join(folder, "geom.xyz"), symbols, coords, comment="point %s" % key)
        _write_json(os.path.join(folder, "order.json"), {"order": perm, "symbols": symbols, "grouped_symbols": grouped})
        split = bool(s["split_scf"]) and series == 1
        if series == 1:
            if s["scf_threads"]:
                threads = int(s["scf_threads"])
            elif split:
                threads = self.machine.threads("scf")
            else:                               # the SCF runs inside the QMC job, on the cores of its first node
                nodes = max(1, int(self.machine.job("qmc").get("nodes", "1")))
                threads = max(1, min(self.machine.threads("scf"), ranks // nodes))
            _write_text(os.path.join(folder, "scf.py"), Q.scf_script(
                grouped, coords[perm], charge=int(s["charge"]), spin=int(s["spin"]), xc=s["xc"], basis=s["basis"],
                ecp=s["ecp"], grid_level=int(s["grid_level"]), threads=threads,
                **({"rhf_energy": True} if trial is not None else {})))
        for src in sources.values():
            dst = os.path.join(folder, os.path.basename(src))
            if os.path.lexists(dst):
                os.remove(dst)
            if s["pseudo_files"] == "symlink":
                os.symlink(os.path.abspath(src), dst)
            else:
                shutil.copyfile(src, dst)
        p = {
            "key": key, "role": spec.role, "series": series, "base_key": base_key, "blocks": blocks,
            "project": "dmc" if series == 1 else "dmc%d" % series, "ranks": ranks, "threads": 1,
            "walkers": walkers, "walkers_requested": requested, "steps": steps, "seeds": seeds,
            "seed_windows": {name: list(seed_window(seed, ranks)) for name, seed in seeds.items()},
            "settings": s, "split_scf": split, "status": WAITING, "wait": None, "jastrow_start": None,
            "jobs": {}, "job_ids": [], "attempts": 0, "stop": None, "final": None, "pending_since": None,
            "pending_what": None, "checks": {}, "notes": notes, "cost_previous": 0.0, "launch_error": None,
            "created": self.clock(),
        }
        if series > 1:
            p["wait"] = {"for": "base", "key": base_key}
        elif start_key:
            p["wait"] = {"for": "jastrow", "key": str(start_key)}
        if trial is not None:
            p["trial"] = trial
            if series == 1:
                p["qp2_key"] = str(trial.get("qp2_key") or (key + "/qp2"))
                _check_key(p["qp2_key"])
                p["wait"] = {"for": "qp2", "key": p["qp2_key"]}
                run = self._qp2(p)
                run.ensure(grouped, coords[perm], trial)
                if not run.rec.get("owner"):        # the point that carries the cost of the shared parent
                    run.rec["owner"] = key
                    run._save()
                if MD.cut_id(trial) not in run.rec.setdefault("cut_owners", {}):   # and the one that carries the cost of this cut
                    run.rec["cut_owners"][MD.cut_id(trial)] = key
                    run._save()
                # such a point has no QMC job for a long time: the list lets a new process find it
                if key not in self._state.setdefault("trial_points", []):
                    self._state["trial_points"].append(key)
                    self._save_state()
        self._points[key] = p
        self._save(p)
        if signature is not None and "vmc0" in steps:
            self._state["vmc0"][signature] = key
            self._save_state()
        if base is not None:
            base.setdefault("series_keys", {})[str(series)] = key
            self._save(base)
        return p

    # ------------------------------------------------------------------ Quantum Package
    def _ensure_qp2(self, p: dict) -> MD.Qp2Run:
        """The QP2 calculations of the geometry of a series-1 multi-determinant point. If their folder was deleted
        (to compute them again), it is made again from the geometry of the point."""
        run = self._qp2(p)
        if run.rec is None:
            folder = self.folder(p["key"])
            symbols, coords = read_xyz(os.path.join(folder, "geom.xyz"))
            perm = [int(i) for i in _read_json(os.path.join(folder, "order.json"))["order"]]
            run.ensure([symbols[i] for i in perm], np.asarray(coords, dtype=float)[perm], p["trial"])
            run.rec["owner"] = p["key"]
            run.rec.setdefault("cut_owners", {})[MD.cut_id(p["trial"])] = p["key"]
            run._save()
        return run

    def qp2_info(self, qp2_key: str) -> Optional[dict]:
        """``{"owner", "parent_core_hours"}`` of a QP2 folder: the point that carries the cost of the shared parent
        calculation, and that cost. None if the folder holds no record."""
        if not os.path.exists(os.path.join(self.folder(qp2_key), "qp2.json")):
            return None
        run = self._qp2_runs.get(qp2_key)
        if run is None:
            run = MD.Qp2Run(self.folder(qp2_key), self.machine, self.scheduler, clock=self.clock)
        return {"owner": run.rec.get("owner"), "parent_core_hours": run.cost_parent()}

    def _qp2(self, p: dict) -> MD.Qp2Run:
        """The Quantum Package calculations of the geometry of a multi-determinant point."""
        key = p.get("qp2_key") or self._load(p["base_key"])["qp2_key"]
        run = self._qp2_runs.get(key)
        if run is None:
            s = p["settings"]
            if self._qp2_jobs is None:
                self._qp2_jobs = JobTable(os.path.join(self.root, "qp2_jobs.json"))
            run = MD.Qp2Run(self.folder(key), self.machine, self.scheduler, clock=self.clock,
                            poll_seconds=float(s["poll_seconds"]), settle_seconds=float(s["settle_seconds"]),
                            unknown_seconds=float(s["unknown_seconds"]), max_attempts=int(s["max_attempts"]),
                            job_table=self._qp2_jobs, name="qp2_" + key.replace("/", "_"))
            self._qp2_runs[key] = run
        return run

    # ------------------------------------------------------------------ launch
    def _job_time(self, p: dict) -> Optional[str]:
        """Time limit of the QMC job of a point (batch format), or None for the value of the machine profile. A
        point with several determinants is several times slower per step: its limit is ``[job.qmc] time_multidet``
        of the profile, read whenever the job script is written, so that it can be raised after a failure."""
        s = p["settings"]
        if s["job_time"]:
            return s["job_time"]
        if p.get("trial"):
            return self.machine.job("qmc").get("time_multidet") or None
        return None

    def _job_mem(self, p: dict) -> Optional[str]:
        """Memory of the QMC job of a point, or None for the value of the machine profile. A point with several
        determinants can need more (QMCPACK keeps tables for every determinant and walker): ``[job.qmc] mem_multidet``."""
        if p.get("trial"):
            return self.machine.job("qmc").get("mem_multidet") or None
        return None

    def _limits(self, p: dict, kind: str = "qmc") -> Dict[str, int]:
        """Time limit of every step in seconds, never longer than the time limit of the job."""
        s = p["settings"]
        job_seconds = parse_time((s["scf_job_time"] if kind == "scf" else self._job_time(p)) or self.machine.job(kind).get("time"))
        out = {}
        for step in STEPS:
            limit = int(s["time_limits"].get(step) or 0) or job_seconds or _LONG_LIMIT
            out[step] = min(limit, job_seconds) if job_seconds else limit
        if p.get("trial") and int(s["time_limits"].get("opt") or 0) == int(DEFAULT_SETTINGS["time_limits"]["opt"]):
            out["opt"] = job_seconds or _LONG_LIMIT     # the default of 2 hours was chosen for one determinant
        return out

    def _script(self, p: dict, kind: str) -> str:
        """Text of ``job.sh`` (kind qmc) or ``job_scf.sh`` (kind scf)."""
        s = p["settings"]
        folder = self.folder(p["key"])
        name = _job_name(p["key"])
        m = self.machine
        common = dict(project=p["project"], blocks=p["blocks"], launch=m.launch("qmc", p["ranks"]),
                      python=m.executable("python"), qmcpack=m.executable("qmcpack"),
                      convert4qmc=m.executable("convert4qmc"), timeout=m.executable("timeout"))
        if kind == "scf":
            head = m.header("scf", name + "_scf", folder, log="job_scf.out", time=s["scf_job_time"])
            return head + steps_script(["scf"], limits=self._limits(p, "scf"), **common)
        head = m.header("qmc", name, folder, log="job.out", time=self._job_time(p), tasks=p["ranks"],
                        exclude=p.get("exclude_nodes"), mem=self._job_mem(p))
        if p.get("trial") and p["series"] == 1:
            facts = p.get("trial_facts") or {}
            common["md"] = {"h5": facts.get("h5", ""), "ndet": int(facts.get("n_det", 2)),
                            "determinants_tools": m.executable("determinants_tools")}
        return head + steps_script(p["steps"], limits=self._limits(p, "qmc"), write_dmc_input=p["series"] == 1, **common)

    def _write_start_file(self, p: dict, start_key: str) -> bool:
        """Write ``jastrow_start.xml`` from the Jastrow factor of another point. False if that factor does not fit."""
        s = p["settings"]
        folder = self.folder(p["key"])
        coeff = _read_json(os.path.join(self.folder(start_key), "jastrow.json"))["coefficients"]
        grouped = _read_json(os.path.join(folder, "order.json"))["grouped_symbols"]
        need = {"e" + el: int(s["jastrow"]["j1_size"]) for el in grouped}
        need["uu"] = int(s["jastrow"]["j2_size"])
        if "ud" in coeff:
            need["ud"] = int(s["jastrow"]["j2_size"])
        for cid, size in need.items():
            if cid not in coeff or len(coeff[cid]) != size or not all(_finite(v) for v in coeff[cid]):
                return False
        lines = ['<?xml version="1.0"?>', '<jastrow_start source="%s">' % start_key]
        lines += ['   <coefficients id="%s" type="Array"> %s </coefficients>' % (cid, " ".join("%.10g" % v for v in coeff[cid]))
                  for cid in need]
        _write_text(os.path.join(folder, "jastrow_start.xml"), "\n".join(lines + ["</jastrow_start>", ""]))
        return True

    def _write_job_files(self, p: dict) -> None:
        """Write the files of the job: settings of the input writer and scripts, or the DMC input of a further series."""
        s = p["settings"]
        folder = self.folder(p["key"])
        dmc = {k: v for k, v in s["dmc"].items() if k in DMC_KEYS}
        dmc.update({"walkers": p["walkers"], "blocks": p["blocks"], "seed": p["seeds"]["dmc"], "project": p["project"]})
        if p["series"] == 1:
            opt = {k: v for k, v in s["opt"].items() if k in OPT_KEYS}
            if p["jastrow_start"] and s["opt"].get("cycles_reuse"):
                opt["cycles"] = int(s["opt"]["cycles_reuse"])
            opt["seed"] = p["seeds"]["opt"]
            vmc0 = {k: v for k, v in s["vmc0"].items() if k in VMC0_KEYS}
            vmc0.update({"walkers": p["walkers"], "seed": p["seeds"]["vmc0"]})
            cfg = {"pseudo_dir": "", "jastrow": s["jastrow"], "vmc0": vmc0, "opt": opt, "dmc": dmc,
                   "charge": int(s["charge"]), "spin": int(s["spin"]),
                   "limits": {"vmc0_sigmas": s["vmc0_sigmas"], "vmc0_tolerance": s["vmc0_tolerance"],
                              "jastrow_sigmas": s["jastrow_sigmas"]}}
            if p["jastrow_start"]:
                cfg["jastrow_start"] = "jastrow_start.xml"
            if p.get("trial"):
                t = p["trial"]
                cfg["determinantset_file"] = "md_determinantset.xml"
                cfg["trial"] = {k: t[k] for k in ("check_sigmas", "check_fail_sigmas", "check_tolerance", "check_sigma_max",
                                                  "check_pilot_blocks", "check_max_blocks", "check_max_runs", "check_steps",
                                                  "check_walkers", "rhf_tolerance")}
                cfg["mdvmc"] = {"seed": p["seeds"]["vmc0"], "ranks": p["ranks"]}
                _write_json(os.path.join(folder, "trial_facts.json"), p["trial_facts"])
            _write_json(os.path.join(folder, "qmc_settings.json"), cfg)
            if p["split_scf"]:
                _write_text(os.path.join(folder, "job_scf.sh"), self._script(p, "scf"))
                os.chmod(os.path.join(folder, "job_scf.sh"), 0o755)
        else:
            base_folder = self.folder(p["base_key"])
            system = Q.load_system(os.path.join(base_folder, "system.json"))
            Q.check_pseudopotentials(dict(system, charge=int(s["charge"])), folder)
            coeff = _read_json(os.path.join(base_folder, "jastrow.json"))["coefficients"]
            several = {}
            if p.get("trial"):                      # the determinants of the base point, read from its h5 file
                with open(os.path.join(base_folder, "md_determinantset.xml")) as fh:
                    block = fh.read().rstrip("\n")
                several["determinantset"] = block.replace(
                    'href="trial.h5"', 'href="%s"' % os.path.relpath(os.path.join(base_folder, "trial.h5"), folder))
            text = Q.dmc_input(system, coeff, jastrow=s["jastrow"], pseudo_dir="",
                               orbitals=os.path.relpath(os.path.join(base_folder, "scf.h5"), folder), **dmc, **several)
            _write_text(os.path.join(folder, "%s.in.xml" % p["project"]), text)
        _write_text(os.path.join(folder, "job.sh"), self._script(p, "qmc"))
        os.chmod(os.path.join(folder, "job.sh"), 0o755)

    def _jastrow_ready(self, key: str) -> Optional[bool]:
        """True when the point has orbitals and a checked Jastrow factor, False when it ended without them,
        None while that is still open."""
        q = self._load(key)
        if q is None:
            return False
        if q["status"] == ACTIVE:
            self._update(q)                         # also makes the quality checks of that point
        if q["status"] == FINAL and q["final"]["state"] == FAILED:
            return False
        folder = self.folder(key)
        names = (("trial.h5", "md_determinantset.xml") if q.get("trial") else ("scf.h5",)) + ("system.json", "jastrow.json")
        files = [os.path.join(folder, name) for name in names]
        if all(os.path.exists(f) and os.path.getsize(f) > 0 for f in files):
            try:
                _read_json(files[-2])
                if _read_json(files[-1]).get("coefficients"):
                    return True
            except (OSError, ValueError):
                pass                                # a file is being written: look again later
        return False if q["status"] == FINAL else None

    def _try_launch(self, p: dict, raise_errors: bool = False) -> None:
        """Start the job of a waiting point if nothing holds it back."""
        key = p["key"]
        wait = p.get("wait")
        try:
            if wait and wait["for"] == "base":
                ready = self._jastrow_ready(wait["key"])
                if ready is None:
                    self._list("waiting", key, True)
                    return
                if not ready:
                    base = self._load(wait["key"])
                    why = base["final"]["message"] if (base and base.get("final")) else "it has no record"
                    self._finalize(p, FAILED, "the base point %s ended without orbitals and a checked Jastrow factor (%s)"
                                   % (wait["key"], why))
                    return
            elif wait and wait["for"] == "jastrow":
                if p["split_scf"] and not p["jobs"].get("scf") and not self._exists(p, "scf.done"):
                    self._write_job_files(p)
                    p["attempts"] = max(1, p["attempts"])
                    self._submit_job(p, "job_scf.sh", "scf", None)      # the SCF does not need the other point
                ready = self._jastrow_ready(wait["key"])
                if ready is None:
                    self._list("waiting", key, True)
                    return
                if ready and self._write_start_file(p, wait["key"]):
                    p["jastrow_start"] = wait["key"]
                else:
                    self._note(p, "the point %s has no usable Jastrow factor: the optimization starts from zero" % wait["key"])
            elif wait and wait["for"] == "qp2":
                run = self._ensure_qp2(p)
                state, message = run.advance(p["trial"])
                if state == MD.RUNNING:
                    if p.get("qp2_message") != message or p.get("launch_error"):
                        p["qp2_message"], p["launch_error"] = message, None
                        self._save(p)
                    self._list("waiting", key, True)
                    return
                if state == MD.FAILED:
                    self._finalize(p, FAILED, "the Quantum Package calculation failed: %s" % message)
                    return
                p["trial_facts"] = run.facts(p["trial"])
                p["qp2_message"] = None
            try:
                self._write_job_files(p)
            except ValueError as exc:               # for example pseudopotential files that do not fit the SCF of the base point
                self._finalize(p, FAILED, "the input files could not be written: %s" % exc)
                return
            p["wait"] = None
            self._submit_attempt(p)
        except Exception as exc:                    # a refused submission must not lose the point: it is tried again
            p["launch_error"] = "the job could not be started: %s" % exc
            self._save(p)
            self._list("waiting", key, True)
            if raise_errors:
                raise
            return
        p["status"] = ACTIVE
        p["launch_error"] = None
        self._save(p)
        self._list("waiting", key, False)

    def _submit_attempt(self, p: dict) -> None:
        """Submit the job, or the two jobs, of the current attempt."""
        p["attempts"] = max(1, p["attempts"])
        after: List[str] = []
        if p["split_scf"] and not self._exists(p, "scf.done"):
            scf_job = p["jobs"].get("scf")
            if not (scf_job and self._job_state(scf_job) in _IN_QUEUE):
                scf_job = self._submit_job(p, "job_scf.sh", "scf", None)
            after = [scf_job]
        self._submit_job(p, "job.sh", "qmc", after)

    # ------------------------------------------------------------------ reading files
    def _trace_path(self, p: dict) -> str:
        return os.path.join(self.folder(p["key"]), "%s.s001.scalar.dat" % p["project"])

    def _trace(self, p: dict) -> Dict[str, np.ndarray]:
        """Complete rows of the DMC trace. An empty dictionary when no trace exists yet."""
        path = self._trace_path(p)
        try:
            st = os.stat(path)
        except OSError:
            return {}
        signature = (st.st_size, st.st_mtime_ns)
        cached = self._trace_cache.get(path)
        if cached and cached[0] == signature:
            return cached[1]
        try:
            cols = read_trace(path)
        except ValueError:                          # the file exists but has no header line yet
            cols = {}
        if cols and "LocalEnergy" not in cols:
            raise ValueError("%s has no column LocalEnergy" % path)
        self._trace_cache[path] = (signature, cols)
        return cols

    def _rows(self, p: dict) -> int:
        cols = self._trace(p)
        return int(len(cols["LocalEnergy"])) if cols else 0

    def _surrogate_energy(self, p: dict) -> Optional[float]:
        """SCF energy of the geometry in hartree, once the SCF step is done."""
        if p.get("e_surrogate") is not None:
            return p["e_surrogate"]
        folder = self.folder(p["base_key"] or p["key"])
        if not os.path.exists(os.path.join(folder, "scf.done")):
            return None
        path = os.path.join(folder, "energy.dat")
        try:
            with open(path) as fh:
                value = float(fh.read().split()[0])
        except (OSError, ValueError, IndexError) as exc:
            raise ValueError("%s: scf.done exists but the SCF energy cannot be read (%s)" % (path, exc))
        if not np.isfinite(value):
            raise ValueError("%s: the SCF energy is not finite" % path)
        p["e_surrogate"] = value
        self._save(p)
        return value

    # ------------------------------------------------------------------ quality
    def _quality(self, p: dict) -> Optional[str]:
        """Message of the first failed quality check, or None. A check is made once, when its files are complete."""
        folder = self.folder(p["key"])
        s = p["settings"]
        checks = p["checks"]
        for step in p["steps"]:
            line = _first_line(os.path.join(folder, step + ".failed"))
            if line and line.startswith("QUALITY:"):
                if step in ("mdc4q", "mdvmc") and step not in checks:
                    # the job made this check itself and stopped: its numbers go into the record, so that a report
                    # can show the verdict and not "check not made"
                    try:
                        with open(os.path.join(folder, "check_%s.json" % step)) as fh:
                            checks[step] = json.load(fh)
                        self._save(p)
                    except (OSError, ValueError):
                        pass
                return "step %s: %s" % (step, line[len("QUALITY:"):].strip())
        todo = []
        if p["series"] == 1:
            if self._exists(p, "scf.done"):
                todo.append(("system", lambda: check_system(folder, s["charge"], s["spin"])))
            if "vmc0" in p["steps"] and self._exists(p, "vmc0.done"):
                todo.append(("vmc0", lambda: check_zero_jastrow(folder, s["vmc0_sigmas"], s["vmc0_tolerance"])))
            if p.get("trial"):
                t = p["trial"]
                if self._exists(p, "mdc4q.done"):
                    todo.append(("mdc4q", lambda: MD.check_export(folder, rhf_tolerance=t["rhf_tolerance"], write=False)))
                if self._exists(p, "mdvmc.done"):
                    todo.append(("mdvmc", lambda: MD.check_md_vmc(folder, t["check_sigmas"], t["check_tolerance"],
                                                                  t["check_sigma_max"], t["check_fail_sigmas"],
                                                                  t["check_max_runs"])))
            if self._exists(p, "opt.done") and self._exists(p, "jastrow.json"):
                if p.get("trial"):
                    todo.append(("jastrow", lambda: check_jastrow(folder, s["jastrow_sigmas"],
                                                                  reference=(p.get("trial_facts") or {}).get("e_ref"))))
                else:
                    todo.append(("jastrow", lambda: check_jastrow(folder, s["jastrow_sigmas"])))
        for name, func in todo:
            if name in checks:
                continue
            try:
                res = func()
            except (OSError, ValueError, KeyError) as exc:
                if name == "jastrow" and not self._exists(p, "dmc.done") and not self._exists(p, "dmc.in.xml"):
                    continue                        # jastrow.json is being written: look again later
                res = {"check": name, "ok": False, "message": "the check could not be made: %s" % exc}
            checks[name] = res
            self._save(p)
            if not res["ok"]:
                return "%s check: %s" % (name, res["message"])
        if "seed" not in checks:
            report = read_seed_report(os.path.join(folder, "%s.out" % p["project"]))
            if report is not None:
                ok = bool(report["from_input"] and report["offset"] == p["seeds"]["dmc"] % SEED_MODULUS)
                checks["seed"] = dict(report, ok=ok, check="seed")
                self._save(p)
                if not ok:
                    return ("seed check: QMCPACK used the seed offset %d (%s) but the input gave the seed %d"
                            % (report["offset"], "from the input" if report["from_input"] else "from the clock",
                               p["seeds"]["dmc"]))
        if p.get("trial") and "msd" not in checks and self._rows(p) > 0:
            owner = p if p["series"] == 1 else (self._load(p["base_key"]) or {})
            ndet = int((owner.get("trial_facts") or {}).get("n_det", 0))
            if ndet > 1:
                try:
                    with open(os.path.join(folder, "%s.out" % p["project"]), errors="replace") as fh:
                        ok = (MD.MSD_TEXT % ndet) in fh.read()
                except OSError:
                    return None                     # the output is not there yet: look again later
                checks["msd"] = {"check": "msd", "ok": ok, "ndet": ndet,
                                 "message": "the DMC run %s %d determinants" % ("read" if ok else "did NOT read", ndet)}
                self._save(p)
                if not ok:
                    return ("determinant check: the QMCPACK output of the DMC run does not say '%s'" % (MD.MSD_TEXT % ndet))
        return None

    # ------------------------------------------------------------------ state
    def _finalize(self, p: dict, state: str, message: str, rows: Optional[int] = None) -> None:
        p["final"] = {"state": state, "message": message, "rows": self._rows(p) if rows is None else int(rows),
                      "time": self.clock()}
        p["status"] = FINAL
        p["pending_since"] = None
        p["pending_what"] = None
        if not (p.get("stop") and not p["stop"].get("settled")):
            self._watch.difference_update(j for j in p["jobs"].values() if j)
        self._save(p)
        self._list("waiting", p["key"], False)

    def _pending(self, p: dict, what: str, wait: float) -> bool:
        """True when the condition ``what`` has held for ``wait`` seconds. The first sighting starts the count."""
        now = self.clock()
        if p.get("pending_what") != what:
            p["pending_what"] = what
            p["pending_since"] = now
            self._save(p)
        return now - p["pending_since"] >= wait

    def _clear_pending(self, p: dict) -> None:
        if p.get("pending_what") is not None:
            p["pending_what"] = None
            p["pending_since"] = None
            self._save(p)

    def _open_step(self, p: dict) -> Optional[str]:
        """The first step of the point that is not done."""
        for step in p["steps"]:
            if not self._exists(p, step + ".done"):
                return step
        return None

    def _update(self, p: dict) -> None:
        """Bring the record of a point up to date: start its job, judge it, submit it again or finish it."""
        if p["status"] == FINAL:
            return
        if p["status"] == WAITING:
            self._try_launch(p)
            if p["status"] != WAITING or not p["jobs"].get("scf"):
                return                              # with split_scf the SCF job of a waiting point is judged below
        s = p["settings"]
        problem = self._quality(p)
        if problem:
            self._cancel_jobs(p)
            self._finalize(p, FAILED, problem)
            return
        rows = self._rows(p)
        if self._exists(p, "dmc.done"):
            if rows >= p["blocks"]:
                self._finalize(p, DONE, "all %d blocks are written" % p["blocks"])
            elif self._pending(p, "short", float(s["settle_seconds"])):
                self._finalize(p, FAILED, "dmc.done exists but %s holds %d of %d blocks"
                               % (os.path.basename(self._trace_path(p)), rows, p["blocks"]))
            return
        scf_job, qmc_job = p["jobs"].get("scf"), p["jobs"].get("qmc")
        scf_state = self._job_state(scf_job) if (scf_job and not self._exists(p, "scf.done")) else None
        if scf_state is not None and scf_state not in _IN_QUEUE and scf_state != sched.COMPLETED:
            state, job_id = scf_state, scf_job      # the SCF job ended badly: the QMC job can never start
        elif qmc_job:
            state, job_id = self._job_state(qmc_job), qmc_job
        elif scf_state is not None and scf_state not in _IN_QUEUE:
            state, job_id = scf_state, scf_job      # the SCF job ended and scf.done does not exist
        else:
            return                                  # only the SCF job exists, and it is in the queue or done
        if state in _IN_QUEUE:
            self._clear_pending(p)
            if state == sched.RUNNING and job_id == qmc_job and rows > 0:
                self._check_speed(p)
            return
        wait = float(s["unknown_seconds"] if state == sched.UNKNOWN else s["settle_seconds"])
        if self._pending(p, "ended:%s:%s" % (job_id, state), wait):
            self._attempt_ended(p, state, job_id, rows)

    # ------------------------------------------------------------------ speed of a run
    def _block_cost(self, p: dict) -> Optional[float]:
        """Core-seconds per walker and step of the DMC run of a point (median over its blocks), or None while
        the run holds fewer than ``slow_min_blocks`` blocks. The number can be compared between the points of
        one molecule, whatever their time step, walker count and block length."""
        cols = self._trace(p)
        if not cols or "BlockCPU" not in cols or len(cols["BlockCPU"]) < int(p["settings"].get("slow_min_blocks", 5)):
            return None
        work = float(p["walkers"]) * float(p["settings"]["dmc"]["steps"])
        if not work > 0.0:
            return None
        return float(np.median(cols["BlockCPU"])) * float(p["ranks"]) / work

    def _wave_kind(self, p: dict) -> str:
        """Points whose DMC steps cost the same: all points with one determinant (the empty word), or the series of
        one multi-determinant wave function at one geometry (the same cut of the same QP2 folder). A step with
        several determinants costs several times more, and one cutoff keeps another number of determinants at
        another geometry or with another ``n_det_max``, so the speed rule compares only runs of the same kind."""
        if not p.get("trial"):
            return ""
        qp2_key = p.get("qp2_key")
        if not qp2_key and p.get("base_key"):                 # a further series: the folder of its base point
            qp2_key = (self._load(p["base_key"]) or {}).get("qp2_key")
        return "md:%s:%s" % (qp2_key, MD.cut_id(p["trial"]))

    def _reference_cost(self, key: str, need: int) -> Optional[float]:
        """Median block cost of the other points with the same kind of wave function that ran at a normal speed.
        None if fewer than ``need`` exist."""
        if not self._speed_scanned:
            self._speed_scanned = True              # once per program start: the points of earlier driver jobs
            for other in self.keys():
                if other != key and other not in self._speed:
                    q = self._load(other)
                    if q is not None and not q.get("slow"):
                        cost = self._block_cost(q)
                        if cost is not None:
                            self._speed[other] = cost
                            self._speed_kind[other] = self._wave_kind(q)
        mine = self._wave_kind(self._load(key) or {})
        values = [v for k, v in self._speed.items() if k != key and self._speed_kind.get(k, "") == mine]
        return float(np.median(values)) if len(values) >= max(1, int(need)) else None

    def _job_nodes(self, p: dict) -> Optional[str]:
        """Nodes of the last job of the point, as its script wrote them to ``job.info``."""
        try:
            with open(os.path.join(self.folder(p["key"]), "job.info")) as fh:
                lines = [ln for ln in fh.read().splitlines() if "nodes=" in ln]
        except OSError:
            return None
        nodes = lines[-1].split("nodes=", 1)[1].strip() if lines else ""
        return nodes if re.fullmatch(r"[A-Za-z0-9_.,\[\]-]+", nodes or "") else None

    def _slow(self, p: dict) -> Optional[Tuple[float, float]]:
        """(block cost, reference) if the run of the point is ``slow_factor`` times slower than the other points."""
        factor = float(p["settings"].get("slow_factor") or 0.0)
        cost = self._block_cost(p)
        if cost is None or not factor > 0.0:
            return None
        ref = self._reference_cost(p["key"], int(p["settings"].get("slow_min_points", 3)))
        if ref is not None and cost > factor * ref:
            return cost, ref
        if not p.get("slow"):
            self._speed[p["key"]] = cost            # a run of normal speed is a reference for the others
            self._speed_kind[p["key"]] = self._wave_kind(p)
        return None

    def _check_speed(self, p: dict) -> None:
        """A running job whose blocks take much longer than those of the other points (a node that is shared
        with a job that overloads it, for example) is cancelled. :meth:`_attempt_ended` then submits it again,
        away from the nodes it ran on. This happens once per point."""
        if p.get("slow"):
            return
        found = self._slow(p)
        if found is None:
            return
        cost, ref = found
        rows, nodes = self._rows(p), self._job_nodes(p)
        again = p["attempts"] < int(p["settings"]["max_attempts"])
        p["slow"] = {"cost": cost, "reference": ref, "rows": rows, "nodes": nodes, "time": self.clock(), "pending": again}
        text = ("a block takes %.1f times longer than in the other runs (%.3g against %.3g core-seconds per walker and "
                "step) on %s after %d blocks" % (cost / ref, cost, ref, nodes or "unknown nodes", rows))
        if again:
            self._cancel_jobs(p)
            self._note(p, text + ": the job was cancelled and is submitted again")
        else:
            self._note(p, text + ": no attempt is left, the job goes on")
        self._save(p)

    def _attempt_ended(self, p: dict, state: str, job_id: str, rows: int) -> None:
        """The job of the current attempt left the queue and ``dmc.done`` does not exist."""
        s = p["settings"]
        folder = self.folder(p["key"])
        self._cancel_jobs(p)                        # a QMC job that waits for a failed SCF job
        step = self._open_step(p)
        line = _first_line(os.path.join(folder, "%s.failed" % step)) if step else None
        where = "step %s" % step if step else "the job"
        reason = "%s: %s" % (where, line) if line else "%s did not finish and left no reason" % where
        tail = "job %s ended in state %s" % (job_id, state)
        if p.get("trial") and (state == sched.TIMEOUT or (line or "").startswith("TIMEOUT:")):
            tail += (". A point with several determinants needs more time for every step: raise [job.qmc] time_multidet "
                     "of the machine profile, then use retry")
        slow = p.get("slow") or {}
        ours = bool(slow.get("pending"))            # this program cancelled the job because it was slow
        if state == sched.CANCELLED and not ours:
            self._finalize(p, FAILED, "%s; %s. The job was cancelled outside this program, so it is not submitted again"
                           % (reason, tail))
            return
        if state == sched.TIMEOUT and rows > 0 and not ours:
            found = None if slow else self._slow(p)
            if found is None or p["attempts"] >= int(s["max_attempts"]):
                extra = "" if found is None else (" The run was %.1f times slower than the other runs, and no attempt is left"
                                                  % (found[0] / found[1]))
                self._finalize(p, FAILED, "the job reached its time limit after %d of %d blocks; %s. The blocks are kept. "
                                          "A second attempt would reach the same limit%s" % (rows, p["blocks"], tail,
                                                                                            "." + extra if extra else ""))
                return
            # the limit was reached because the run was slow, not because the limit is too short: one more attempt
            slow = p["slow"] = {"cost": found[0], "reference": found[1], "rows": rows, "nodes": self._job_nodes(p),
                                "time": self.clock(), "pending": True}
            ours = True
        if ours:
            reason = ("the run was %.1f times slower than the other runs%s" % (
                slow["cost"] / slow["reference"], " on %s" % slow["nodes"] if slow.get("nodes") else ""))
        if p["attempts"] >= int(s["max_attempts"]):
            self._finalize(p, FAILED, "%s; %s; attempt %d of %d" % (reason, tail, p["attempts"], int(s["max_attempts"])))
            return
        # one more attempt: the steps that are done are skipped
        spent = self._open_step_cost(p)
        for name in p["steps"]:
            old = os.path.join(folder, name + ".failed")
            if os.path.exists(old):
                os.replace(old, "%s.attempt%d" % (old, p["attempts"]))
            started = os.path.join(folder, name + ".started")
            if os.path.exists(started) and not os.path.exists(os.path.join(folder, name + ".done")):
                os.remove(started)
        if not self._exists(p, "scf.done"):
            p["jobs"]["scf"] = None
        if ours:
            # the new run writes the same file names: the blocks of the slow run are set aside, and the new job
            # stays away from the nodes of the slow one
            trace = self._trace_path(p)
            if os.path.exists(trace):
                os.replace(trace, "%s.attempt%d" % (trace, p["attempts"]))
            self._trace_cache.pop(trace, None)
            slow["pending"] = False
            p["slow"] = slow
            p["exclude_nodes"] = slow.get("nodes")
            _write_text(os.path.join(folder, "job.sh"), self._script(p, "qmc"))
        try:
            if p["status"] == WAITING:              # split_scf: the QMC job still waits for another point
                self._submit_job(p, "job_scf.sh", "scf", None)
            else:
                self._submit_attempt(p)
        except Exception as exc:
            p["launch_error"] = "the job could not be submitted again: %s" % exc
            p["pending_what"] = None
            self._save(p)
            return
        self._note(p, "attempt %d failed (%s; %s): the job was submitted again" % (p["attempts"], reason, tail))
        p["attempts"] += 1
        p["cost_previous"] = float(p.get("cost_previous", 0.0)) + spent
        p["launch_error"] = None
        p["pending_what"] = None
        p["pending_since"] = None
        self._save(p)

    # ------------------------------------------------------------------ retry by request
    def failed_keys(self) -> List[str]:
        """Keys of the points that ended as FAILED. The points that other points wait for come first."""
        failed = []
        for key in self.keys():
            p = self._load(key)
            if p is not None and p["status"] == FINAL and (p.get("final") or {}).get("state") == FAILED:
                waits = p.get("base_key") or (p.get("wait") or {}).get("for") not in (None, "qp2")
                failed.append((1 if waits else 0, key))
        return [key for _rank, key in sorted(failed)]

    def retry(self, key: str) -> bool:
        """Give a failed point a new set of attempts: after a person repaired the cause, or after a failure of
        the cluster. Steps that are done are kept; the files of the failed step and a partial DMC trace are
        set aside with the ending ``.retry<n>``. Returns True if the point is active again."""
        p = self._load(str(key))
        if p is None or p["status"] != FINAL or (p.get("final") or {}).get("state") != FAILED:
            return False
        folder = self.folder(p["key"])
        n = int(p.get("retries", 0)) + 1
        for name in p["steps"]:
            old = os.path.join(folder, name + ".failed")
            if os.path.exists(old):
                os.replace(old, "%s.retry%d" % (old, n))
            started = os.path.join(folder, name + ".started")
            if os.path.exists(started) and not os.path.exists(os.path.join(folder, name + ".done")):
                os.remove(started)
        if not self._exists(p, "dmc.done"):
            trace = self._trace_path(p)
            if os.path.exists(trace):
                os.replace(trace, "%s.retry%d" % (trace, n))
            self._trace_cache.pop(trace, None)
        self._note(p, "submitted again by request; before that: %s" % p["final"]["message"])
        p.update({"retries": n, "final": None, "attempts": 0, "stop": None, "launch_error": None,
                  "pending_since": None, "pending_what": None, "status": WAITING})
        p.pop("slow", None)
        if not self._exists(p, "scf.done"):
            p["jobs"]["scf"] = None
        if p.get("trial") and p["series"] == 1 and (p.get("wait") or {}).get("for") == "qp2":
            self._ensure_qp2(p).retry()             # the Quantum Package jobs of this geometry get new attempts too
        self._save(p)
        self._list("waiting", p["key"], True)
        self._try_launch(p)                         # a point that waits for another point starts when that one is ready
        return True

    def retry_failed(self) -> List[str]:
        """:meth:`retry` for every failed point. Returns the keys that are active again."""
        return [key for key in self.failed_keys() if self.retry(key)]

    def _current_state(self, p: dict) -> Tuple[str, str]:
        """State and message of a point that is not final."""
        rows = self._rows(p)
        if p["status"] == WAITING:
            wait = p.get("wait")
            if p.get("launch_error"):
                return QUEUED, p["launch_error"]
            if wait and wait["for"] == "base":
                return QUEUED, "waits for the orbitals and the Jastrow factor of %s" % wait["key"]
            if wait and wait["for"] == "qp2":
                return QUEUED, "waits for the Quantum Package calculation (%s)" % (p.get("qp2_message") or "not started")
            if wait:
                return QUEUED, "waits for the Jastrow factor of %s" % wait["key"]
            return QUEUED, "the job is not started yet"
        if rows > 0:
            return RUNNING, "%d of %d blocks" % (rows, p["blocks"])
        if p.get("launch_error"):
            return QUEUED, p["launch_error"]
        step = self._open_step(p)
        job_id = p["jobs"].get("qmc") or p["jobs"].get("scf")
        if step and (self._exists(p, step + ".started") or self._job_state(job_id) == sched.RUNNING):
            return QUEUED, "the job runs; no DMC block yet (step %s)" % step
        return QUEUED, "the job waits in the queue"

    # ------------------------------------------------------------------ read
    def read(self, key: str) -> PointData:
        """State and data of a point right now. An unknown key gives the state ABSENT."""
        key = str(key)
        p = self._load(key)
        if p is None:
            return PointData(key=key, state=ABSENT)
        self._poll()
        self._housekeeping()
        if p["status"] != FINAL:
            self._update(p)
        e_sur = None
        try:
            e_sur = self._surrogate_energy(p)
        except ValueError as exc:
            if p["status"] != FINAL:
                self._cancel_jobs(p)
                self._finalize(p, FAILED, str(exc))
        if p["status"] == FINAL:
            state, message, limit = p["final"]["state"], p["final"]["message"], p["final"]["rows"]
        else:
            (state, message), limit = self._current_state(p), None
        cols = self._trace(p)
        if cols:
            energy, weight = trace_energy(cols)
            energy, weight = energy[:limit].copy(), weight[:limit].copy()
        else:
            energy, weight = np.zeros(0), np.zeros(0)
        return PointData(key=key, state=state, energy=energy, weight=weight, e_surrogate=e_sur, message=message,
                         meta=self._meta(p, cols, len(energy)))

    def _meta(self, p: dict, cols: Dict[str, np.ndarray], n: int) -> Dict[str, object]:
        s = p["settings"]
        meta: Dict[str, object] = {
            "requested_blocks": p["blocks"], "steps": int(s["dmc"]["steps"]), "timestep": float(s["dmc"]["timestep"]),
            "warmup_steps": int(s["dmc"]["warmup_steps"]), "walkers": p["walkers"],
            "walkers_requested": p["walkers_requested"], "ranks": p["ranks"], "seed": p["seeds"]["dmc"],
            "seeds": dict(p["seeds"]), "seed_window": list(p["seed_windows"]["dmc"]), "project": p["project"],
            "series": p["series"], "base_key": p["base_key"], "role": p["role"], "attempts": p["attempts"],
            "job_ids": list(p["job_ids"]), "folder": self.folder(p["key"]), "chain_steps": list(p["steps"]),
            "step": self._open_step(p), "stop_requested": bool(p.get("stop")), "jastrow_start": p.get("jastrow_start"),
            "notes": list(p.get("notes", [])), "blocks_on_disk": int(len(cols["LocalEnergy"])) if cols else 0,
            "acceptance": None, "seconds_per_block": None, "vmc_energy": None, "vmc_error": None,
            "vmc_variance": None, "jastrow_source": None, "zero_jastrow": p["checks"].get("vmc0"),
            "checks": copy.deepcopy(p["checks"]), "prime_range": (p["checks"].get("seed") or {}).get("prime_range"),
        }
        if cols and n:
            if "AcceptRatio" in cols:
                meta["acceptance"] = float(np.mean(cols["AcceptRatio"][:n]))
            if "BlockCPU" in cols:
                meta["seconds_per_block"] = float(np.mean(cols["BlockCPU"][:n]))
            # the columns of the DMC trace, cut to the blocks that count for this point. The pilot runs read the
            # variance of the local energy, the time per block and the acceptance of every block from them.
            meta["trace"] = {name: cols[name][:n].copy() for name in
                             ("LocalEnergy", "LocalEnergy_sq", "BlockWeight", "BlockCPU", "AcceptRatio") if name in cols}
        if p.get("trial"):
            owner = p if p["series"] == 1 else (self._load(p["base_key"]) or {})    # a further series uses the files of its base
            meta["trial"] = dict(owner.get("trial_facts") or {}, kind="multidet", cutoff=p["trial"]["cutoff"],
                                 orbitals=p["trial"]["orbitals"])
        if p.get("vmc") is None:
            path = os.path.join(self.folder(p["base_key"] or p["key"]), "jastrow.json")
            if os.path.exists(path):
                try:
                    sel = _read_json(path).get("selection", {})
                except (OSError, ValueError):
                    sel = {}                        # the file is being written
                if "energy" in sel:
                    p["vmc"] = {"energy": sel["energy"], "error": sel.get("error"), "variance": sel.get("variance"),
                                "source": sel.get("source")}
                    self._save(p)
        if p.get("vmc"):
            meta.update({"vmc_energy": p["vmc"]["energy"], "vmc_error": p["vmc"]["error"],
                         "vmc_variance": p["vmc"]["variance"], "jastrow_source": p["vmc"]["source"]})
        return meta

    # ------------------------------------------------------------------ stop
    def stop(self, key: str) -> None:
        """Stop a point. The blocks that are complete now are its final data (see "Stopping a run")."""
        p = self._load(str(key))
        if p is None or p["status"] == FINAL:
            return
        self._poll()
        waits_qp2 = p["status"] == WAITING and (p.get("wait") or {}).get("for") == "qp2"
        if not waits_qp2:                           # an update of a waiting point could submit its next Quantum Package job
            self._update(p)
        if p["status"] == FINAL:
            return
        cols = self._trace(p)
        rows = int(len(cols["LocalEnergy"])) if cols else 0
        if rows >= p["blocks"]:
            return                                  # all blocks are on disk: there is nothing to stop
        s = p["settings"]
        folder = self.folder(p["key"])
        grace = float(s["stop_grace_seconds"])
        if rows and "BlockCPU" in cols:
            grace = max(grace, float(s["stop_grace_blocks"]) * float(np.mean(cols["BlockCPU"])))
        stop = {"time": self.clock(), "rows": rows, "grace": grace, "cancelled": [], "settled": False}
        _write_text(os.path.join(folder, "stop.requested"), "%d\n" % int(round(grace)))
        if rows > 0:
            for name in ("%s.STOP" % p["project"], "%s.s001.STOP" % p["project"]):
                _write_text(os.path.join(folder, name), "")
        else:
            stop["cancelled"] = self._cancel_jobs(p)        # no block can be lost: cancel at once
            stop["settled"] = True
        p["stop"] = stop
        self._finalize(p, STOPPED, "stopped on request after %d of %d blocks" % (rows, p["blocks"]), rows=rows)
        if not stop["settled"]:
            self._list("stops", p["key"], True)
        if waits_qp2:
            # the Quantum Package jobs of this geometry that no other waiting point needs are cancelled. A point that
            # waits for the QP2 calculations needs the jobs of its own cut, and the parent job unless it is the RHF
            # determinant and the RHF step is done; a point whose QMC job runs needs no QP2 job any more.
            run = self._qp2(p)
            users = [q for q in (self._load(k) for k in self.keys())
                     if q is not None and q["key"] != p["key"] and q.get("qp2_key") == p["qp2_key"]
                     and q["status"] == WAITING and (q.get("wait") or {}).get("for") == "qp2"]
            rhf_done = os.path.exists(os.path.join(run.folder, "qpscf.done"))
            keep_parent = any(not q["trial"].get("import_ezfio") and (q["trial"]["cutoff"] != MD.HF_LABEL or not rhf_done)
                              for q in users)
            cancelled = run.cancel(keep_parent=keep_parent, keep_cuts=[MD.cut_id(q["trial"]) for q in users])
            if cancelled:
                self._note(p, "Quantum Package job(s) %s cancelled with the stop" % ", ".join(cancelled))
                self._save(p)

    def _housekeeping(self) -> None:
        """Cancel the jobs of stopped points that are still in the queue after their grace time."""
        if not self._state["stops"]:
            return
        now = self.clock()
        for key in list(self._state["stops"]):
            p = self._load(key)
            stop = p.get("stop") if p else None
            if stop and not stop.get("settled"):
                in_queue = [j for j in p["jobs"].values() if j and self._job_state(j) in _IN_QUEUE]
                if in_queue and now - stop["time"] < stop["grace"]:
                    continue
                if in_queue:
                    stop["cancelled"] = self._cancel_jobs(p)
                    self._note(p, "the job did not end within %.0f s after the stop request: cancelled" % stop["grace"])
                stop["settled"] = True
                self._watch.difference_update(j for j in p["jobs"].values() if j)
                self._save(p)
            self._list("stops", key, False)

    def advance(self) -> None:
        """Do the duties that no single ``read`` owns: cancel stopped jobs whose grace time is over, and
        start points that waited for another point."""
        self._poll()
        self._housekeeping()
        for key in list(self._state["waiting"]):
            p = self._load(key)
            if p is None or p["status"] != WAITING:
                self._list("waiting", key, False)
            else:
                self._update(p)

    # ------------------------------------------------------------------ cost
    def _cores(self, p: dict, step: str) -> int:
        if step == "scf" and p["split_scf"]:
            return self.machine.tasks("scf") * self.machine.threads("scf")
        return p["ranks"]

    def _step_seconds(self, p: dict, step: str, open_only: bool = False) -> float:
        """Run time of a step from its marker files. A step that still runs counts up to now.
        With ``open_only`` a step that is done counts 0."""
        folder = self.folder(p["key"])
        start = _marker_time(os.path.join(folder, step + ".started"))
        if start is None:
            return 0.0
        end = _marker_time(os.path.join(folder, step + ".done"))
        if end is not None and open_only:
            return 0.0
        if end is None:
            for name in (step + ".stopped", step + ".failed"):
                path = os.path.join(folder, name)
                if os.path.exists(path):
                    end = os.path.getmtime(path)
                    break
        if end is None:
            end = p["final"]["time"] if p.get("final") else self.clock()
        return max(0.0, float(end) - float(start))

    def _open_step_cost(self, p: dict) -> float:
        """Core-hours of the steps of the current attempt that did not finish."""
        return sum(self._step_seconds(p, step, open_only=True) * self._cores(p, step) for step in p["steps"]) / 3600.0

    def cost(self, key: str) -> float:
        """Core-hours spent on a point so far: the run time of every step (from its marker files) times the
        cores of its job, plus the unfinished steps of failed attempts. For the DMC step the sum of the column
        BlockCPU is used when it is larger. 0 if nothing is known."""
        p = self._load(str(key))
        if p is None:
            return 0.0
        total = float(p.get("cost_previous", 0.0))
        for step in p["steps"]:
            seconds = self._step_seconds(p, step)
            if step == "dmc":
                cols = self._trace(p)
                if cols and "BlockCPU" in cols:
                    seconds = max(seconds, float(np.sum(cols["BlockCPU"])))
            total += seconds * self._cores(p, step) / 3600.0
        if p.get("trial") and p["series"] == 1:
            run = self._qp2(p)                      # the Quantum Package part is counted once: for the first point of a
            if run.rec is not None:                 # cut, and the shared parent for the first point of the geometry
                if run.rec.get("cut_owners", {}).get(MD.cut_id(p["trial"]), p["key"]) == p["key"]:
                    total += run.cost_cut(p["trial"])
                if run.rec.get("owner") == p["key"]:
                    total += run.cost_parent()
        return total

    def total_cost(self) -> float:
        """Core-hours of all points."""
        return float(sum(self.cost(key) for key in self.keys()))


# ============================================================================= commands used inside the job
_COMMANDS = {"check-system": "check_system.json", "write-vmc0": None, "check-vmc0": "check_vmc0.json",
             "write-opt": None, "write-dmc": "check_jastrow.json"}


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Commands that ``job.sh`` runs in the folder of a point.

        python -m montepass.engine.qmcpack check-system    system.json against geometry, charge, pseudopotentials
        python -m montepass.engine.qmcpack write-vmc0      write vmc0.in.xml
        python -m montepass.engine.qmcpack check-vmc0      zero-Jastrow VMC energy against the PySCF determinant
        python -m montepass.engine.qmcpack write-opt       write opt.in.xml
        python -m montepass.engine.qmcpack write-dmc       choose the Jastrow factor, check it, write dmc.in.xml

    Exit code 0: done. Exit code 3: a quality check failed; the last line of the output says why, and a
    second attempt cannot repair it. Any other exit code: the command itself could not run.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1 or args[0] not in _COMMANDS:
        print(main.__doc__)
        return 2
    command = args[0]
    cfg = _read_json("qmc_settings.json") if os.path.exists("qmc_settings.json") else {}
    limits = cfg.get("limits", {})

    def verdict(result: dict) -> int:
        _write_json(_COMMANDS[command], result)
        print(result["message"])
        return 0 if result["ok"] else QUALITY_EXIT

    if command == "check-system":
        return verdict(check_system(".", cfg.get("charge", 0), cfg.get("spin", 0)))
    if command == "check-vmc0":
        return verdict(check_zero_jastrow(".", limits.get("vmc0_sigmas", 4.0), limits.get("vmc0_tolerance", 0.005)))
    what = command.split("-")[1]
    try:
        rc = Q.main([what, "qmc_settings.json"])
    except (ValueError, KeyError, SystemExit) as exc:
        # no usable optimization cycle, missing coefficients, or pseudopotential files that do not fit the SCF
        print("the %s input could not be written: %s" % (what, exc))
        return QUALITY_EXIT
    if rc != 0:
        return int(rc)
    if command == "write-dmc":
        if cfg.get("determinantset_file") and os.path.exists("trial_facts.json"):
            return verdict(check_jastrow(".", limits.get("jastrow_sigmas", 2.0),
                                         reference=_read_json("trial_facts.json").get("e_ref")))
        return verdict(check_jastrow(".", limits.get("jastrow_sigmas", 2.0)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
