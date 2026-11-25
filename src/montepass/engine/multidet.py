"""Multi-determinant trial wave functions built with Quantum Package 2 (QP2).

The search for a geometry uses ONE Slater determinant (module :mod:`qmcpack_inputs`). After a
geometry is fixed, its DMC energy can be computed again with a trial wave function of several
determinants. This module holds everything of that chain that is not QMCPACK input text:

    QP2 jobs of one geometry (class :class:`Qp2Run`; job kind ``qp2`` of the machine profile)
        parent job   qp_create_ezfio -q  ->  qp_run scf  ->  qp_run fci (CIPSI; continued in further jobs)
        cut job      qp_run truncate_wf_spin on a copy  ->  qp_run print_energy  ->  qp_run save_for_qmcpack
    QMC job of the point (job kind ``qmc``; steps of :mod:`qmcpack`)
        scf     PySCF as for every point (``system.json``, surrogate energy, RHF energy for a comparison)
        mdc4q   determinants_tools.py --transform, convert4qmc -multidet, check of the files
        mdvmc   VMC run WITHOUT Jastrow factor; its energy must equal the QP2 energy of the same wave function
        opt     Jastrow optimization; the coefficients of the determinants stay fixed
        dmc     DMC

Words used here
---------------
parent      the CIPSI wave function of a geometry (QP2 program ``fci``), as large as ``n_det_max`` asks
cut         the wave function used in QMC: the parent with every determinant removed that holds an alpha or
            beta string whose summed weight is below ``cutoff`` (QP2 program ``truncate_wf_spin``, setting
            ``qmcpack ci_threshold``). The coefficients are not optimized again. The label ``hf`` means the
            single RHF determinant of QP2.
e_ref       the energy that QP2 prints for the cut wave function (line ``E( 1 ) =``); for ``hf`` the SCF energy

Rules that this module enforces (each one cost time before; see docs/multireference.md)
-------------------------------------------------------------------------------------
* QP2 is ONE process with OpenMP threads. Only the ``qp_run`` lines get more than one thread; every
  other line of a job keeps ``OMP_NUM_THREADS=1``.
* The EZFIO files are written with ``qp_create_ezfio -q`` (convention 20210101). The export program
  ``save_for_qmcpack`` gives wrong orbitals for the newer convention, with no error message. The export
  is refused for any other convention, and the exported basis is tested for the factors that the error leaves.
* ``save_for_qmcpack`` rewrites the EZFIO it runs on: the export always runs on a copy.
* ``convert4qmc`` is called with ``-multidet`` (the option ``-multidets`` is ignored without a message),
  and the number of determinants in its output is compared with the number that QP2 printed.
* Before any Jastrow optimization the VMC energy without Jastrow factor must equal ``e_ref`` within
  ``check_sigmas`` standard errors plus ``check_tolerance``, with an error bar of at most ``check_sigma_max``.
"""
from __future__ import annotations

import copy
import json
import math
import os
import re
import shutil
import sys
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..scheduler import base as sched
from ..scheduler.base import JobTable
from ..units import ANGSTROM_TO_BOHR

QUALITY_EXIT = 3                         # exit code of a command of this module when a quality check fails
CONVENTION = 20210101                    # the EZFIO convention that save_for_qmcpack handles
MSD_TEXT = "Found %d terms in the MSD expansion"
QMCPACK_SUCCESS = "QMCPACK execution completed successfully"
CI_FIXED_TEXT = "CI coefficients are not optimizable"
HF_LABEL = "hf"

PARENT_STEPS = ("qpcreate", "qpscf", "cipsi")
CUT_STEPS = ("trunc", "qpenergy", "qpexport")
IMPORT_STEPS = ("qpimport", "qpenergy", "qpexport")      # a wave function that exists already (setting import_ezfio)
HEARTBEAT_FRESH = 180.0                  # seconds: a job whose sign of life is younger than this is alive
QMC_STEPS = ("mdc4q", "mdvmc")           # the steps that this module adds to the QMC job of a point

DEFAULT_TRIAL: Dict[str, object] = {
    "kind": "single",                    # single = one Kohn-Sham determinant (the default chain); multidet = this module
    "orbitals": "qp2_rhf",               # orbitals of the determinants: the RHF orbitals that QP2 computes
    "cutoff": None,                      # QP2 setting "qmcpack ci_threshold" (0 < cutoff < 1), or "hf"
    "n_det_max": None,                   # QP2 setting "determinants n_det_max" of the CIPSI run
    "qp2_basis": "cc-pvtz_ecp_ccecp",    # option -b of qp_create_ezfio
    "qp2_pseudo": "ccecp",               # option -p of qp_create_ezfio
    "qp2_key": None,                     # folder of the QP2 runs of this geometry, below the engine root; None = <key>/qp2
    "import_ezfio": None,                # absolute path of an EZFIO folder whose wave function is used as it is (no CIPSI,
                                         # no cut); the geometry of that folder must be the geometry of the point
    "label": None,                       # name of an imported wave function (needed with import_ezfio)
    # the VMC run without Jastrow factor
    "check_sigmas": 3.0, "check_tolerance": 1.0e-4, "check_sigma_max": 6.0e-4,
    "check_fail_sigmas": 5.0,            # a difference above this many standard errors (plus the tolerance) fails at once
    "check_pilot_blocks": 60,            # short first run that measures the noise; it sets the length of the next run
    "check_max_blocks": 6000,            # no run is longer than this
    "check_max_runs": 4,                 # runs with new random numbers that are combined (the first one is the short run)
    "check_steps": 40,                   # VMC steps per block
    "check_walkers": 32,                 # VMC walkers of every MPI rank
    # comparisons without statistics
    "rhf_tolerance": 1.0e-5,             # largest difference of the RHF energies of PySCF and QP2, hartree. Measured
                                         # for water with ccECP and cc-pVTZ on 2026-11-24: 5e-9. A larger difference
                                         # means another molecule, basis or pseudopotential, or another SCF solution
    "energy_tolerance": 1.0e-6,          # largest difference of truncate_wf_spin and print_energy, hartree
    "min_parent_ratio": 2.0,             # the parent must hold at least this many times the determinants of the cut
    # jobs
    "max_segments": 4,                   # QP2 jobs of the CIPSI run of one geometry (a job at its time limit continues)
    "margin_seconds": 600,               # the CIPSI step ends this long before the time limit of its job
    "time_limits": {"qpcreate": 600, "qpscf": 0, "cipsi": 0, "trunc": 0, "qpenergy": 0, "qpexport": 1800, "qpimport": 1800},
}


# ============================================================================= settings
def trial_of(settings: Optional[dict]) -> Optional[dict]:
    """The complete trial settings of a point, or None for a single-determinant point.

    ``settings["trial"]`` is laid over :data:`DEFAULT_TRIAL`. No section ``trial``, an empty one, or
    ``kind = single`` mean the default chain, and the engine then behaves exactly as without this module. A
    section with other keys but without ``kind`` is an error (ValueError)."""
    given = (settings or {}).get("trial") or {}
    if given and "kind" not in given:
        raise ValueError("settings['trial'] holds %s but no key 'kind' (single or multidet): the point would run with "
                         "one determinant without a message" % sorted(given))
    if str(given.get("kind", "single")).lower() == "single":
        return None
    out = copy.deepcopy(DEFAULT_TRIAL)
    for key, value in given.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key].update(value)
        else:
            out[key] = copy.deepcopy(value)
    out["kind"] = str(out["kind"]).lower()
    if isinstance(out.get("cutoff"), str):
        text = out["cutoff"].strip().lower()
        out["cutoff"] = HF_LABEL if text == HF_LABEL else text
        if out["cutoff"] != HF_LABEL:
            try:
                out["cutoff"] = float(text)
            except ValueError:
                pass                    # validate_trial reports it
    return out


CUT_DIGITS = 6                           # two cutoffs are told apart by this number of significant digits


def cut_label(cutoff) -> str:
    """Short name of a cut, used in folder names: ``hf``, ``c1e-03``, ``c2.5e-04`` (``CUT_DIGITS`` significant digits)."""
    if cutoff == HF_LABEL:
        return HF_LABEL
    mantissa, exponent = ("%.*e" % (CUT_DIGITS - 1, float(cutoff))).split("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    return "c%se%+03d" % (mantissa, int(exponent))


def cut_id(trial: dict) -> str:
    """Name of the cut folder of a point: the label of an imported wave function, else the label of the cutoff."""
    return str(trial["label"]) if trial.get("import_ezfio") else cut_label(trial["cutoff"])


def validate_trial(trial: Optional[dict], charge: int = 0, spin: int = 0) -> None:
    """Raise ValueError for trial settings that this module cannot use."""
    if trial is None:
        return
    unknown = sorted(set(trial) - set(DEFAULT_TRIAL))
    if unknown:
        raise ValueError("settings['trial'] has unknown keys %s; allowed: %s" % (unknown, ", ".join(sorted(DEFAULT_TRIAL))))
    if trial["kind"] != "multidet":
        raise ValueError("trial kind = %r; allowed: single, multidet" % (trial["kind"],))
    if trial["orbitals"] != "qp2_rhf":
        raise ValueError("trial orbitals = %r; this version knows only qp2_rhf (the RHF orbitals of QP2)" % (trial["orbitals"],))
    if int(charge) != 0 or int(spin) != 0:
        raise ValueError("a multi-determinant trial function needs charge 0 and multiplicity 1 in this version "
                         "(charge %d, %d unpaired electrons were given)" % (int(charge), int(spin)))
    cutoff = trial["cutoff"]
    if trial.get("import_ezfio"):
        path = str(trial["import_ezfio"])
        if not re.fullmatch(r"/[A-Za-z0-9_./+\-]+", path):
            raise ValueError("trial import_ezfio = %r; give the absolute path of an EZFIO folder (letters, digits and "
                             "the characters _ . / + - only)" % (trial["import_ezfio"],))
        if not trial.get("label") or not re.fullmatch(r"[A-Za-z0-9_.+\-]+", str(trial["label"])):
            raise ValueError("trial label = %r; an imported wave function needs a short name (letters, digits and the "
                             "characters _ . + -)" % (trial.get("label"),))
        if cutoff is not None:
            raise ValueError("trial cutoff = %r together with import_ezfio: an imported wave function is used as it is" % (cutoff,))
    elif cutoff != HF_LABEL:
        if isinstance(cutoff, bool) or not isinstance(cutoff, (int, float)) or not 0.0 < float(cutoff) < 1.0:
            raise ValueError("trial cutoff = %r; give a number between 0 and 1 (QP2 ci_threshold) or hf" % (cutoff,))
        n = trial["n_det_max"]
        if isinstance(n, bool) or not isinstance(n, (int, float)) or int(n) != n or int(n) < 2:
            raise ValueError("trial n_det_max = %r; give the size of the CIPSI wave function (a whole number, at least 2)" % (n,))
    for name in ("check_sigmas", "check_fail_sigmas", "check_tolerance", "check_sigma_max", "rhf_tolerance", "energy_tolerance"):
        if not float(trial[name]) > 0.0:
            raise ValueError("trial %s must be positive" % name)
    for name in ("check_pilot_blocks", "check_max_blocks", "check_max_runs", "check_steps", "check_walkers", "max_segments"):
        if int(trial[name]) < 1:
            raise ValueError("trial %s must be at least 1" % name)
    if int(trial["check_pilot_blocks"]) < 40:
        raise ValueError("trial check_pilot_blocks must be at least 40: the first 20 blocks are not used, and fewer give no error bar")
    if float(trial["check_fail_sigmas"]) < float(trial["check_sigmas"]):
        raise ValueError("trial check_fail_sigmas must not be smaller than check_sigmas")
    if not float(trial["min_parent_ratio"]) >= 1.0:
        raise ValueError("trial min_parent_ratio must be at least 1")
    if not isinstance(trial["time_limits"], dict):
        raise ValueError("trial time_limits must be a dictionary")
    known = PARENT_STEPS + CUT_STEPS + ("qpimport",)
    unknown = sorted(set(trial["time_limits"]) - set(known))
    if unknown:
        raise ValueError("trial time_limits has unknown steps %s; allowed: %s" % (unknown, ", ".join(known)))
    for name in ("qp2_basis", "qp2_pseudo"):
        if not re.fullmatch(r"[A-Za-z0-9_.+\-]+", str(trial[name])):
            raise ValueError("trial %s = %r is not the name of a QP2 data file" % (name, trial[name]))


# ============================================================================= reading what QP2 wrote
_NUMBER = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[EeDd][-+]?\d+)?"


def _number(text: str) -> float:
    return float(text.replace("D", "E").replace("d", "e"))


def ezfio_value(path: str) -> str:
    """Content of a one-value EZFIO file (``ezfio_files/ezfio_convention``, ``determinants/n_det``, ...)."""
    with open(path) as fh:
        return fh.read().strip()


def parse_scf_log(text: str) -> float:
    """SCF energy of a ``qp_run scf`` output (line ``* SCF energy``)."""
    found = re.findall(r"^\*\s*SCF energy\s+(%s)\s*$" % _NUMBER, text, re.M)
    if not found:
        raise ValueError("no line '* SCF energy' in the output of qp_run scf")
    return _number(found[-1])


def parse_cipsi_log(text: str) -> List[dict]:
    """The summaries that ``qp_run fci`` prints, one per CIPSI iteration: number of determinants,
    variational energy, second-order energy (PT2) with its statistical error, and their sum."""
    out = []
    for chunk in re.split(r"^\s*Summary at N_det\s*=", text, flags=re.M)[1:]:
        entry: Dict[str, float] = {}
        m = re.search(r"^\s*N_det\s*=\s*(\d+)", chunk, re.M)
        if not m:
            m = re.match(r"\s*(\d+)", chunk)
        if not m:
            continue
        entry["n_det"] = int(m.group(1))
        for key, name in (("e_var", r"E"), ("pt2", r"PT2"), ("e_pt2", r"E\+PT2")):
            m = re.search(r"^\s*%s\s*=\s*(%s)(?:\s*\+/-\s*(%s))?" % (name, _NUMBER, _NUMBER), chunk, re.M)
            if m:
                entry[key] = _number(m.group(1))
                if m.group(2) and key != "e_var":
                    entry[key + "_error"] = _number(m.group(2))
        if "e_var" in entry:
            out.append(entry)
    return out


def parse_truncate_log(text: str) -> Tuple[int, float]:
    """Number of determinants and energy of the cut wave function from a ``qp_run truncate_wf_spin`` output.

    The output holds two kinds of lines with the words "Number of determinants". The line that starts with
    a star is printed at the start and gives the size of the PARENT. The line with a colon gives the result."""
    sizes = re.findall(r"^\s*Number of determinants:\s*(\d+)\s*$", text, re.M)
    energies = re.findall(r"^\s*E\(\s*1\s*\)\s*=\s*(%s)\s*$" % _NUMBER, text, re.M)
    if not sizes or not energies:
        raise ValueError("the output of truncate_wf_spin holds no line 'Number of determinants:' or no line 'E( 1 ) ='")
    return int(sizes[-1]), _number(energies[-1])


def truncate_parent_size(text: str) -> Optional[int]:
    """Number of determinants of the wave function that ``qp_run truncate_wf_spin`` READ (the line with a star,
    printed at its start), or None if the output holds no such line."""
    found = re.findall(r"^\s*\*\s*Number of determinants\s+(\d+)\s*$", text, re.M)
    return int(found[0]) if found else None


def parse_print_energy_log(text: str) -> float:
    """Energy of the stored wave function from a ``qp_run print_energy`` output (line ``E(CI) =``)."""
    found = re.findall(r"^\s*E\(CI\)\s*=\s*(%s)\s*$" % _NUMBER, text, re.M)
    if not found:
        raise ValueError("the output of print_energy holds no line 'E(CI) ='")
    return _number(found[-1])


# ============================================================================= the files of convert4qmc
_WF_RE = re.compile(r"<wavefunction\b[^>]*>(.*?)</wavefunction>", re.S)
_JASTROW_RE = re.compile(r"<jastrow\b[^>]*/>|<jastrow\b.*?</jastrow>", re.S)
_ATTR_RE = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)\s*=\s*"([^"]*)"')


def _attributes(tag: str) -> Dict[str, str]:
    return dict(_ATTR_RE.findall(tag))


def read_determinantset(path: str, orbitals: str, indent: str = "         ") -> Tuple[str, dict]:
    """The orbitals and determinants of a wave function file of convert4qmc, ready for an input of this package.

    The text returned is everything inside ``<wavefunction>`` that is no Jastrow element. For the QMCPACK
    version of 2026 that is one ``<determinantset>`` element; a version that writes the orbital sets as their
    own element next to it is covered by the same rule. The facts are ``{"ndet", "nstates", "kind"}``. Every
    ``href`` is set to ``orbitals`` (the h5 file as the QMCPACK run will see it). Raises ValueError if the text
    is not one that this package accepts: for several determinants the coefficients must be fixed
    (``optimize="no"``), the list must be of type DETS, and no determinant may be dropped by a cutoff of
    QMCPACK (``cutoff`` at most 1e-12)."""
    with open(path) as fh:
        text = fh.read()
    bodies = _WF_RE.findall(text)
    if len(bodies) != 1:
        raise ValueError("%s holds %d wavefunction elements, 1 is needed" % (path, len(bodies)))
    block = _JASTROW_RE.sub("", bodies[0]).strip()
    if block.count("<determinantset") != 1 or "</determinantset>" not in block:
        raise ValueError("%s holds %d determinantset elements, 1 is needed" % (path, block.count("<determinantset")))
    multi = re.findall(r"<multideterminant\b[^>]*>", block)
    lists = re.findall(r"<detlist\b[^>]*>", block)
    facts: Dict[str, object] = {"kind": "multidet" if multi else "single"}
    if multi:
        if len(multi) != 1 or len(lists) != 1:
            raise ValueError("%s: %d multideterminant and %d detlist elements, 1 of each is needed" % (path, len(multi), len(lists)))
        m, d = _attributes(multi[0]), _attributes(lists[0])
        if m.get("optimize", "no").lower() != "no":
            raise ValueError("%s: the coefficients of the determinants are marked as optimizable (optimize=\"%s\")"
                             % (path, m.get("optimize")))
        if d.get("type", "").upper() != "DETS":
            raise ValueError("%s: detlist type %r, DETS is needed" % (path, d.get("type")))
        try:
            ndet, cutoff = int(d["size"]), float(d.get("cutoff", "0"))
        except (KeyError, ValueError):
            raise ValueError("%s: the detlist element has no usable size or cutoff" % path)
        if cutoff > 1e-12:
            raise ValueError("%s: detlist cutoff %g would drop determinants inside QMCPACK; the cut is made by QP2 alone"
                             % (path, cutoff))
        facts.update({"ndet": ndet, "nstates": int(d["nstates"]) if d.get("nstates", "").isdigit() else None})
    else:
        if "<slaterdeterminant" not in block:
            raise ValueError("%s: the determinantset holds neither a multideterminant nor a slaterdeterminant" % path)
        facts.update({"ndet": 1, "nstates": None})
    block = re.sub(r'href\s*=\s*"[^"]*"', 'href="%s"' % orbitals, block)
    lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
    depth, out = 0, []
    for ln in lines:
        if ln.startswith("</"):
            depth -= 1
        out.append(indent + "   " * max(depth, 0) + ln)
        if ln.startswith("<") and not ln.startswith("</") and not ln.endswith("/>") and "</" not in ln:
            depth += 1
    return "\n".join(out), facts


def read_structure(path: str) -> dict:
    """Ions and electrons of a structure file of convert4qmc: ``{"symbols", "coords_bohr", "charges", "nup", "ndn"}``."""
    import xml.etree.ElementTree as ET

    root = ET.parse(path).getroot()
    ions = [ps for ps in root.iter("particleset") if ps.get("name") == "ion0"]
    elec = [ps for ps in root.iter("particleset") if ps.get("name") == "e"]
    if len(ions) != 1 or len(elec) != 1:
        raise ValueError("%s: the particle sets ion0 and e are needed once each" % path)
    charge = {}
    for group in ions[0].findall("group"):
        for par in group.findall("parameter"):
            if par.get("name") == "charge":
                charge[group.get("name")] = float(par.text)
    symbols, coords = None, None
    for attrib in ions[0].findall("attrib"):
        if attrib.get("name") == "ionid":
            symbols = attrib.text.split()
        elif attrib.get("name") == "position":
            coords = [float(v) for v in attrib.text.split()]
    if not symbols or coords is None or len(coords) != 3 * len(symbols):
        raise ValueError("%s: positions or names of the ions are missing" % path)
    sizes = {g.get("name"): int(g.get("size")) for g in elec[0].findall("group")}
    return {"symbols": symbols, "coords_bohr": np.asarray(coords, dtype=float).reshape(-1, 3).tolist(),
            "charges": [charge.get(s) for s in symbols], "nup": sizes.get("u", 0), "ndn": sizes.get("d", 0)}


def check_export(folder: str = ".", facts_file: str = "trial_facts.json", rhf_tolerance: float = 1.0e-5,
                 write: bool = True) -> dict:
    """Check the files of the step ``mdc4q`` against QP2 and against the SCF of the point.

    * the wave function file of convert4qmc holds the number of determinants that QP2 printed;
    * the structure file of convert4qmc has the atoms of ``system.json`` in the same order at the same
      places (1e-5 bohr), the same valence charges and the same numbers of electrons;
    * the RHF energy of QP2 equals the RHF energy that PySCF computed for the same geometry, basis and
      pseudopotentials (``e_rhf`` of ``system.json``) within ``rhf_tolerance``. The two programs then
      describe the same molecule. The test is skipped when ``system.json`` holds no ``e_rhf``.
    With ``write`` the determinant set is written to ``md_determinantset.xml`` for the later inputs.
    Returns ``{"ok", "message", ...}``."""
    out: Dict[str, object] = {"check": "mdc4q", "ok": False}
    facts = _read_json(os.path.join(folder, facts_file))
    with open(os.path.join(folder, "system.json")) as fh:
        system = json.load(fh)
    problems = []
    try:
        block, found = read_determinantset(os.path.join(folder, "mdc4q.wfj.xml"), "trial.h5")
    except ValueError as exc:
        out["message"] = str(exc)
        return out
    out.update({"ndet": found["ndet"], "ndet_qp2": int(facts["n_det"])})
    if int(found["ndet"]) != int(facts["n_det"]):
        problems.append("the wave function file holds %d determinant(s), QP2 printed %d" % (found["ndet"], int(facts["n_det"])))
    structure = read_structure(os.path.join(folder, "mdc4q.structure.xml"))
    symbols = [str(s).capitalize() for s in system["symbols"]]
    if [str(s).capitalize() for s in structure["symbols"]] != symbols:
        problems.append("the atoms of the QP2 export (%s) are not the atoms of system.json (%s)"
                        % (" ".join(structure["symbols"]), " ".join(symbols)))
    else:
        shift = float(np.abs(np.asarray(structure["coords_bohr"]) - np.asarray(system["coords_bohr"], dtype=float)).max())
        out["max_shift_bohr"] = shift
        if shift > 1e-5:
            problems.append("an atom of the QP2 export is %.2e bohr away from its place in system.json" % shift)
        if any(c is None or abs(float(c) - float(z)) > 1e-8 for c, z in zip(structure["charges"], system["zeff"])):
            problems.append("valence charges of the QP2 export %s differ from those of the SCF %s"
                            % (structure["charges"], list(system["zeff"])))
    if (structure["nup"], structure["ndn"]) != (int(system["nup"]), int(system["ndn"])):
        problems.append("the QP2 export has %d up and %d down electrons, the SCF has %d and %d"
                        % (structure["nup"], structure["ndn"], system["nup"], system["ndn"]))
    extra = ""
    if system.get("e_rhf") is not None and facts.get("e_scf") is not None:
        diff = float(facts["e_scf"]) - float(system["e_rhf"])
        out.update({"e_rhf_pyscf": float(system["e_rhf"]), "e_rhf_qp2": float(facts["e_scf"]), "rhf_difference": diff})
        extra = "; RHF energy of QP2 minus PySCF %+.2e Ha" % diff
        if abs(diff) > float(rhf_tolerance):
            problems.append("the RHF energy of QP2 (%.8f Ha) differs from the RHF energy of PySCF (%.8f Ha) by %+.2e Ha "
                            "(limit %.1e): the two programs do not describe the same molecule, basis or pseudopotentials"
                            % (float(facts["e_scf"]), float(system["e_rhf"]), diff, float(rhf_tolerance)))
    if not problems and write:
        _write_text(os.path.join(folder, "md_determinantset.xml"), block + "\n")
    out["ok"] = not problems
    out["message"] = ("; ".join(problems) if problems else
                      "%d determinant(s) as in QP2; atoms, charges and electrons agree with system.json%s" % (found["ndet"], extra))
    return out


# ============================================================================= the VMC run without Jastrow factor
CHECK_DROP = 20                          # blocks at the start of every VMC run that are not used
MORE_EXIT = 4                            # exit code of check-mdvmc: no verdict yet, one more run is needed


def _vmc_stats(trace: str, drop: int = CHECK_DROP) -> Tuple[float, float, int]:
    """Mean energy, reblocked error and number of used blocks of one VMC run (the first ``drop`` blocks are not used)."""
    from ..stats import read_scalar, reblock

    cols = read_scalar(trace)
    energy = np.asarray(cols["LocalEnergy"], dtype=float)[int(drop):]
    weight = (np.asarray(cols["BlockWeight"], dtype=float)[int(drop):] if "BlockWeight" in cols else np.ones_like(energy))
    if not np.all(np.isfinite(energy)):
        raise ValueError("%s holds %d block(s) whose energy is not a finite number" % (trace, int(np.sum(~np.isfinite(energy)))))
    if len(energy) < 16:
        raise ValueError("%s holds %d usable blocks; 16 are needed for an error bar" % (trace, len(energy)))
    mean, err = reblock(energy, weight)
    if not (math.isfinite(mean) and math.isfinite(err) and err > 0.0):
        raise ValueError("%s gives no finite energy and error bar" % trace)
    return float(mean), float(err), int(len(energy))


def combine_runs(runs: Sequence[Tuple[float, float]]) -> Tuple[float, float]:
    """Mean and error of independent runs (each ``(mean, error)``), weighted with the inverse variances."""
    w = np.array([1.0 / (e * e) for _m, e in runs])
    m = np.array([m for m, _e in runs])
    return float(np.sum(w * m) / np.sum(w)), float(1.0 / math.sqrt(np.sum(w)))


def plan_check_blocks(error_now: float, blocks_first: int, error_first: float, sigma_max: float, max_blocks: int,
                      safety: float = 1.3, drop: int = CHECK_DROP, blocks_total: int = 0) -> int:
    """Blocks of the next VMC run so that the combined error bar becomes ``sigma_max / safety``.

    ``error_now`` is the combined error of the runs made so far. The noise of one block is taken from the first
    run (``error_first`` with ``blocks_first`` used blocks): the error of a run falls with the square root of its
    number of blocks. If the error bar is small enough already, the run is asked for another reason (a
    difference between the pass limit and the fail limit): the next run is then as long as all runs so far
    (``blocks_total`` used blocks), so that the combined result can decide. The answer counts the ``drop``
    blocks that are not used, is at least 40 + ``drop`` and at most ``max_blocks``."""
    target = float(sigma_max) / float(safety)
    if not (error_now > 0.0 and error_first > 0.0 and math.isfinite(error_now) and math.isfinite(error_first)):
        return int(max_blocks)
    if error_now <= target:
        need = max(40, int(blocks_total))
    else:
        # 1 / target^2 = 1 / error_now^2 + 1 / error_next^2 ;  error_next^2 = error_first^2 * blocks_first / blocks_next
        inverse = 1.0 / target ** 2 - 1.0 / error_now ** 2
        need = int(math.ceil(error_first ** 2 * blocks_first * inverse))
    return int(min(max(need, 40) + int(drop), int(max_blocks)))


def check_md_vmc(folder: str = ".", sigmas: float = 3.0, tolerance: float = 1.0e-4, sigma_max: float = 6.0e-4,
                 fail_sigmas: float = 5.0, max_runs: int = 4, facts_file: str = "trial_facts.json") -> dict:
    """Compare the VMC energy without Jastrow factor with the energy that QP2 printed for the same wave function.

    All runs ``mdvmc<r>`` (r = 1, 2, ...) in the folder are combined. The verdict (key ``verdict``):

    PASS   the energies differ by at most ``sigmas`` standard errors plus ``tolerance``, the standard error is at
           most ``sigma_max`` (so that an error of a few millihartree cannot hide in the noise), and every run
           read as many determinants as QP2 wrote;
    FAIL   the energies differ by more than ``fail_sigmas`` standard errors plus ``tolerance``; or a run read
           another number of determinants; or ``max_runs`` runs gave no PASS;
    MORE   neither: one more run with new random numbers is needed.
    ``ok`` is True only for PASS."""
    out: Dict[str, object] = {"check": "mdvmc", "ok": False, "verdict": "FAIL"}
    facts = _read_json(os.path.join(folder, facts_file))
    e_ref, ndet = float(facts["e_ref"]), int(facts["n_det"])
    runs, problems = [], []
    r = 1
    while os.path.exists(os.path.join(folder, "mdvmc%d.s000.scalar.dat" % r)):
        try:
            mean, err, blocks = _vmc_stats(os.path.join(folder, "mdvmc%d.s000.scalar.dat" % r))
        except (OSError, ValueError, KeyError) as exc:
            out["message"] = "run %d of the VMC check without Jastrow factor gave no usable energy: %s" % (r, exc)
            return out
        runs.append({"run": r, "energy": mean, "error": err, "blocks": blocks})
        if ndet > 1:
            try:
                with open(os.path.join(folder, "mdvmc%d.out" % r)) as fh:
                    log = fh.read()
            except OSError:
                log = ""
            if (MSD_TEXT % ndet) not in log:
                problems.append("the QMCPACK output of run %d does not say '%s'" % (r, MSD_TEXT % ndet))
        r += 1
    if not runs:
        out["message"] = "no run of the VMC check without Jastrow factor exists"
        return out
    mean, err = combine_runs([(x["energy"], x["error"]) for x in runs])
    diff = mean - e_ref
    limit = float(sigmas) * err + float(tolerance)
    out.update({"energy": mean, "error": err, "e_ref": e_ref, "difference": diff, "limit": limit, "ndet": ndet,
                "sigma_max": float(sigma_max), "runs": runs})
    text = ("VMC energy without Jastrow factor %.6f +/- %.6f Ha (%d run(s)), QP2 energy of the same wave function %.6f Ha, "
            "difference %+.6f Ha, limit %.6f Ha" % (mean, err, len(runs), e_ref, diff, limit))
    if problems:
        out["message"] = text + ": " + "; ".join(problems)
        return out
    if abs(diff) > float(fail_sigmas) * err + float(tolerance):
        out["message"] = text + (": the difference is above %g standard errors: the QMCPACK input does not describe the wave "
                                 "function of QP2 (orbitals, determinants, atom order or pseudopotentials)" % float(fail_sigmas))
        return out
    if abs(diff) <= limit and err <= float(sigma_max):
        out.update({"ok": True, "verdict": "PASS", "message": text})
        return out
    why = ("the error bar %.6f Ha is above the largest accepted one, %.6f Ha" % (err, float(sigma_max)) if abs(diff) <= limit
           else "the difference is between %g and %g standard errors" % (float(sigmas), float(fail_sigmas)))
    if len(runs) >= int(max_runs):
        out["message"] = text + ": %s after %d runs: no pass" % (why, len(runs))
        return out
    out.update({"verdict": "MORE", "message": text + ": %s; one more run is needed" % why})
    return out


# ============================================================================= small tools
def _write_text(path: str, text: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        fh.write(text)
    os.replace(tmp, path)


def _write_json(path: str, obj) -> None:
    def clean(o):
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    _write_text(path, json.dumps(obj, indent=1, default=clean))


def _read_json(path: str):
    with open(path) as fh:
        return json.load(fh)


# ============================================================================= the batch scripts of QP2
_QP2_TOOLS = r'''
# ---------------------------------------------------------------------------------------------
# Steps of the QP2 calculations of one geometry. A step is skipped when <step>.done exists. A step
# writes <step>.started, runs under a time limit, and then writes <step>.done or <step>.failed
# (one line that says why: RUN: or TIMEOUT: can be repaired by a new job, QUALITY: cannot).
# QP2 is one process with OpenMP threads: only the qp_run lines get QP_THREADS threads.
# ---------------------------------------------------------------------------------------------
QP_THREADS=THREADS_VALUE
QP_USE_THREADS=$QP_THREADS
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
if [ -n "${SLURM_MEM_PER_NODE:-}" ]; then
    QP_MAXMEM=$(( SLURM_MEM_PER_NODE * 9 / 10240 )); [ "$QP_MAXMEM" -ge 1 ] || QP_MAXMEM=1
    export QP_MAXMEM                     # QP2 plans its memory with this number of GB (90 % of the job)
fi
JOB_SECONDS=JOB_SECONDS_VALUE
say() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*"; }
mark() { echo "$2" > "$1.tmp" && mv -f "$1.tmp" "$1"; }
begin_step() {
    STEP="$1"
    if [ -e "$STEP.done" ]; then say "step $STEP: done before, skipped"; return 1; fi
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
need_file() { [ -s "$1" ] || fail_step "RUN: the file $1 is missing or empty"; }
need_text() { grep -q -- "$2" "$1" 2>/dev/null || fail_step "RUN: the text '$2' is not in $1"; }
value_of() { if [ -r "$1" ]; then tr -d ' \n' < "$1"; fi; }
# step_limit SECONDS : the limit of a step, never longer than the time that the job has left (the answer can be 0 or negative)
step_limit() {
    left=$(( JOB_SECONDS - SECONDS - MARGIN_VALUE ))
    if [ "$1" -gt 0 ] && [ "$1" -lt "$left" ]; then echo "$1"; else echo "$left"; fi
}
# qp_set EZFIO GROUP KEY VALUE : set a value of an EZFIO and read it back (numbers are compared as numbers)
qp_set() {
    ( set +u; qp set_file "$1" && qp set "$2" "$3" "$4" ) > qp_set.log 2>&1
    got=$(value_of "$1/$2/$3")
    case "$4" in
        true) want=T ;;
        false) want=F ;;
        *) want="$4" ;;
    esac
    if [ "$got" != "$want" ] && ! awk -v a="$got" -v b="$want" 'BEGIN { if (a !~ /^[-+0-9.]/ || b !~ /^[-+0-9.]/) exit 1
            d = a - b; if (d < 0) d = -d; m = (b < 0 ? -b : b); exit !(d <= 1e-12 * m) }'; then
        fail_step "RUN: qp set $2 $3 $4 left the value '$got' in $1"
    fi
}
# copy_ezfio FROM TO : a copy without the scratch folder work/ (it can hold a stop request of an earlier run)
copy_ezfio() {
    rm -rf "$2" && mkdir "$2" && ( set -o pipefail; ( cd "$1" && tar cf - --exclude=./work . ) | ( cd "$2" && tar xf - ) )
}
# usable EZFIO : true if the stored wave function is complete: every file of determinants/ unzips, and the number of
# determinants is the size of the coefficient file and of the determinant file
usable() {
    [ -s "$1/determinants/n_det" ] && [ -s "$1/determinants/psi_det.gz" ] && [ -s "$1/determinants/psi_coef.gz" ] || return 1
    for gz in "$1"/determinants/*.gz; do gzip -t "$gz" 2>/dev/null || return 1; done
    n=$(value_of "$1/determinants/n_det")
    [ "$n" = "$(zcat "$1/determinants/psi_coef.gz" | awk 'NR == 2 { print $1; exit }')" ] || return 1
    [ "$n" = "$(zcat "$1/determinants/psi_det.gz" | awk 'NR == 2 { print $NF; exit }')" ]
}
# keep_copy N : make cipsi.bak.ezfio a copy of cipsi.ezfio. The copy is taken only if it is complete and if cipsi.ezfio
# held N determinants before and after the copying (QP2 did not save meanwhile). The old copy is removed only after
# the new one has its name; a job that ends in between leaves cipsi.bak.old, which the next job takes.
keep_copy() {
    if copy_ezfio cipsi.ezfio cipsi.bak.tmp && usable cipsi.bak.tmp \
           && [ "$(value_of cipsi.bak.tmp/determinants/n_det)" = "$1" ] \
           && [ "$(value_of cipsi.ezfio/determinants/n_det)" = "$1" ]; then
        rm -rf cipsi.bak.old
        if [ -d cipsi.bak.ezfio ]; then mv cipsi.bak.ezfio cipsi.bak.old; fi
        mv cipsi.bak.tmp cipsi.bak.ezfio && rm -rf cipsi.bak.old
    else
        rm -rf cipsi.bak.tmp; return 1
    fi
}
# manifest EZFIO : one line per file with its checksum and size (without work/). A stored wave function is used again
# later (another cutoff); files that the file system removed in between must not go unseen.
manifest() { ( cd "$1" && find . -type f ! -path './work/*' | LC_ALL=C sort | xargs cksum ); }
same_manifest() { [ -s "$2" ] && [ "$(manifest "$1")" = "$(cat "$2")" ]; }
# qp_program SECONDS OUTPUT PROGRAM EZFIO : qp_run under a time limit, with QP_USE_THREADS threads. The exit code is
# left in RC. A QP2 program whose task server does not start never returns; such a run is ended with RC=75.
# A stop request that an earlier run left in the EZFIO would end the program at once; it is removed first.
qp_program() {
    LIMIT="$1"; run_out="$2"; shift 2
    WHAT="qp_run $*"
    [ "$LIMIT" -gt 0 ] || { RC=124; return; }
    for last; do :; done
    rm -f "$last/work/qpstop" "$last/work/qpkill"
    OMP_NUM_THREADS=$QP_USE_THREADS MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TIMEOUT_COMMAND -k 30 "$LIMIT" qp_run "$@" > "$run_out" 2>&1 &
    qp_pid=$!
    rm -f qp_server.failed
    ( waited=0
      while kill -0 "$qp_pid" 2>/dev/null; do
          sleep 15; waited=$(( waited + 15 ))
          if grep -q "Unable to bind" "$run_out" 2>/dev/null || { [ "$waited" -ge 300 ] && ! grep -q "Task server running" "$run_out" 2>/dev/null; }; then
              echo "$waited" > qp_server.failed; kill "$qp_pid" 2>/dev/null; break
          fi
          grep -q "Task server running" "$run_out" 2>/dev/null && break
      done ) &
    watch_pid=$!
    wait "$qp_pid"; RC=$?
    kill "$watch_pid" 2>/dev/null; wait "$watch_pid" 2>/dev/null
    if [ -e qp_server.failed ]; then RC=75; fi
}
check_qp() {
    if [ "$RC" -eq 75 ]; then fail_step "RUN: the task server of QP2 did not start: $WHAT"; fi
    if [ "$RC" -eq 124 ] || [ "$RC" -eq 137 ]; then fail_step "TIMEOUT: no end within $LIMIT s: $WHAT"; fi
    if [ "$RC" -ne 0 ]; then fail_step "RUN: exit code $RC: $WHAT"; fi
}
say "job ${SLURM_JOB_ID:-local} on $(hostname), $QP_THREADS threads for QP2, QP_MAXMEM=${QP_MAXMEM:-unset}"
echo "$(date +%s) job=${SLURM_JOB_ID:-local} nodes=${SLURM_JOB_NODELIST:-$(hostname)}" >> job.info
command -v qp_run > /dev/null 2>&1 || { echo "SETUP_FAILED: qp_run is not on the PATH (section [setup.qp2] of the machine profile)"; exit 96; }
# Two jobs must never work in this folder at the same time. A second job can exist: the batch system accepted a
# submission but its answer was lost, or it could not cancel a job. So every job writes a sign of life once a minute
# (the time and its job number, file job.heartbeat). A job that finds the young sign of life of ANOTHER job waits:
# if the sign gets 3 minutes old, the other job has ended and this job goes on; if the sign stays young for
# 4 minutes, this job ends. The program that watches the jobs reads the same file (it is not always able to ask
# the batch system) and takes a job that it finds there into its record.
ME=${SLURM_JOB_ID:-local$$}
beat_age() {    # seconds since the sign of life of another job; 999999 if there is none
    beat_time=""; beat_job=""
    if [ -r job.heartbeat ]; then read -r beat_time beat_job < job.heartbeat; fi
    case "$beat_time" in ''|*[!0-9]*) echo 999999; return ;; esac
    if [ "${beat_job:-}" = "$ME" ]; then echo 999999; else echo $(( $(date +%s) - beat_time )); fi
}
beat() { echo "$1 $ME" > "job.heartbeat.$ME" && mv -f "job.heartbeat.$ME" job.heartbeat; }
waited=0
while [ "$(beat_age)" -lt HEARTBEAT_FRESH_VALUE ]; do
    if [ "$waited" -ge 240 ]; then
        echo "SETUP_FAILED: another job works in this folder (sign of life: $(cat job.heartbeat 2>/dev/null)); this job ends"
        exit 98
    fi
    sleep 20; waited=$(( waited + 20 ))
done
beat "$(date +%s)"
sleep 10                                 # two jobs that start in the same second: the one that wrote last keeps the folder
if [ "$(beat_age)" -lt HEARTBEAT_FRESH_VALUE ]; then
    echo "SETUP_FAILED: another job took this folder at the same time (sign of life: $(cat job.heartbeat 2>/dev/null)); this job ends"
    exit 98
fi
( while true; do sleep 60 & wait $!; beat "$(date +%s)"; done ) &
HEART_PID=$!
# at the end the sign of life gets the time 0, so that the next job of this folder does not wait for it
trap 'kill "$HEART_PID" 2>/dev/null; wait "$HEART_PID" 2>/dev/null; [ "$(beat_age)" -lt 999999 ] || beat 0' EXIT
'''


def _tools(threads: int, job_seconds: int, margin: int, timeout: str) -> str:
    return (_QP2_TOOLS.replace("THREADS_VALUE", str(int(threads))).replace("JOB_SECONDS_VALUE", str(int(job_seconds)))
            .replace("MARGIN_VALUE", str(int(margin))).replace("TIMEOUT_COMMAND", timeout)
            .replace("HEARTBEAT_FRESH_VALUE", str(int(HEARTBEAT_FRESH))))


def parent_script(steps: Sequence[str], trial: dict, threads: int, job_seconds: int, timeout: str = "timeout",
                  qp_create: str = "qp_create_ezfio") -> str:
    """Shell text of the parent job of one geometry (everything after the header of the machine profile):
    ``qpcreate`` (the EZFIO from ``geom.xyz``), ``qpscf`` (RHF orbitals), ``cipsi`` (the CIPSI wave function).

    The CIPSI step continues from the stored wave function when one exists, so a job that reached its time
    limit loses only the iteration that was running. While QP2 runs, the step keeps a copy of the last
    complete wave function (``cipsi.bak.ezfio``); a wave function that was damaged by the end of a job is
    replaced by that copy. The step is done only when the wave function holds at least ``n_det_max``
    determinants, or when QP2 stopped because the second-order energy fell below its own limit."""
    for step in steps:
        if step not in PARENT_STEPS:
            raise ValueError("unknown step %r of the parent job; allowed: %s" % (step, ", ".join(PARENT_STEPS)))
    limits = dict(DEFAULT_TRIAL["time_limits"]); limits.update(trial.get("time_limits") or {})
    parts = [_tools(threads, job_seconds, int(trial["margin_seconds"]), timeout)]
    if "qpcreate" in steps:
        parts.append("\n".join([
            "if begin_step qpcreate; then",
            "    rm -rf rhf.ezfio",
            "    LIMIT=$(step_limit %d); WHAT=\"%s\"" % (int(limits["qpcreate"]), qp_create),
            "    [ \"$LIMIT\" -gt 0 ] || fail_step \"TIMEOUT: no time is left in this job for $WHAT\"",
            "    %s -k 30 \"$LIMIT\" %s -q -b %s -p %s -o rhf.ezfio geom.xyz > create.log 2>&1; RC=$?" % (
                timeout, qp_create, trial["qp2_basis"], trial["qp2_pseudo"]),
            "    check_qp",
            "    need_file rhf.ezfio/ezfio_files/ezfio_convention",
            "    conv=$(value_of rhf.ezfio/ezfio_files/ezfio_convention)",
            "    [ \"$conv\" = \"%d\" ] || fail_step \"QUALITY: the EZFIO has the convention $conv; save_for_qmcpack needs %d\"" % (
                CONVENTION, CONVENTION),
            "    [ \"$(value_of rhf.ezfio/basis/ao_normalized)\" = \"T\" ] || fail_step \"QUALITY: basis/ao_normalized of the EZFIO is not T\"",
            "    end_step",
            "fi"]))
    if "qpscf" in steps:
        parts.append("\n".join([
            "if begin_step qpscf; then",
            "    qp_set rhf.ezfio scf_utils thresh_scf 1e-10",
            "    qp_program \"$(step_limit %d)\" scf.log scf rhf.ezfio" % int(limits["qpscf"]),
            "    check_qp",
            "    need_text scf.log \"SCF energy\"",
            "    need_file rhf.ezfio/hartree_fock/energy",
            "    manifest rhf.ezfio > rhf.manifest.tmp && mv -f rhf.manifest.tmp rhf.manifest",
            "    end_step",
            "fi"]))
    if "cipsi" in steps:
        n = int(trial["n_det_max"])
        parts.append("\n".join([
            "if begin_step cipsi; then",
            "    seg=$(( $(ls cipsi.[0-9]*.log 2>/dev/null | wc -l) + 1 ))",
            "    rm -rf cipsi.bak.tmp",
            "    if [ ! -d cipsi.bak.ezfio ] && [ -d cipsi.bak.old ]; then mv cipsi.bak.old cipsi.bak.ezfio; fi",
            "    if [ -d cipsi.ezfio ] && ! usable cipsi.ezfio; then",
            "        say \"the stored CIPSI wave function is damaged\"; rm -rf cipsi.ezfio",
            "    fi",
            "    if [ ! -d cipsi.ezfio ] && [ -d cipsi.bak.ezfio ] && usable cipsi.bak.ezfio; then",
            "        say \"the copy of the last complete CIPSI wave function is used\"; copy_ezfio cipsi.bak.ezfio cipsi.ezfio",
            "    fi",
            "    if [ -d cipsi.ezfio ]; then",
            "        have=$(value_of cipsi.ezfio/determinants/n_det)",
            "        qp_set cipsi.ezfio determinants read_wf true",
            "    else",
            "        same_manifest rhf.ezfio rhf.manifest || fail_step \"QUALITY: files of rhf.ezfio are missing or changed since the SCF\"",
            "        copy_ezfio rhf.ezfio cipsi.ezfio || fail_step \"RUN: rhf.ezfio could not be copied\"",
            "        have=1",
            "    fi",
            "    echo \"start $seg $(date +%s) $have\" >> cipsi.segments",
            "    if [ \"$have\" -gt %d ]; then" % n,
            "        # the stored wave function is large enough already; fci would cut it down to n_det_max",
            "        say \"the stored wave function holds $have determinants: only the diagonalization runs\"",
            "        qp_program \"$(step_limit %d)\" cipsi.$seg.log diagonalize_h cipsi.ezfio" % int(limits["cipsi"]),
            "    else",
            "        qp_set cipsi.ezfio determinants n_det_max %d" % n,
            "        LIMIT=$(step_limit %d)" % int(limits["cipsi"]),
            "        # copy of the last complete wave function. A continued run gets it before QP2 starts (QP2 writes the",
            "        # stored wave function again at its start); while QP2 runs, the copy is renewed after every larger save.",
            "        if [ \"$have\" -gt 1 ] && [ \"$(value_of cipsi.bak.ezfio/determinants/n_det)\" != \"$have\" ]; then",
            "            keep_copy \"$have\" || say \"no copy of the stored wave function could be made before the run\"",
            "        fi",
            "        ( last=$(value_of cipsi.bak.ezfio/determinants/n_det)",
            "          while true; do",
            "              sleep 60 & wait $!",
            "              now=$(value_of cipsi.ezfio/determinants/n_det)",
            "              if [ -n \"$now\" ] && [ \"$now\" != \"$last\" ] && usable cipsi.ezfio && keep_copy \"$now\"; then last=$now; fi",
            "          done ) &",
            "        keep_pid=$!",
            "        qp_program \"$LIMIT\" cipsi.$seg.log fci cipsi.ezfio",
            "        kill \"$keep_pid\" 2>/dev/null; wait \"$keep_pid\" 2>/dev/null",
            "    fi",
            "    now=$(value_of cipsi.ezfio/determinants/n_det)",
            "    echo \"end $seg $(date +%s) ${now:-0} $RC\" >> cipsi.segments",
            "    check_qp",
            "    usable cipsi.ezfio || fail_step \"RUN: the CIPSI wave function is not complete after the run\"",
            "    if [ \"$have\" -gt %d ]; then" % n,
            "        [ \"$now\" = \"$have\" ] || fail_step \"RUN: the diagonalization changed the number of determinants from $have to $now\"",
            "    elif [ \"$now\" -lt %d ]; then" % n,
            "        # QP2 ended by itself below the wanted size: right only if the second-order energy is below its limit",
            "        pt2max=$(value_of cipsi.ezfio/perturbation/pt2_max); pt2max=${pt2max:-1e-4}",
            "        pt2=$(awk '$1 == \"PT2\" && $2 == \"=\" { v = $3 } END { print v }' cipsi.$seg.log)",
            "        awk -v p=\"$pt2\" -v m=\"$pt2max\" 'BEGIN { if (p == \"\") exit 1; p += 0; if (p < 0) p = -p; exit !(p <= m + 0) }' \\",
            "            || fail_step \"RUN: qp_run fci ended with $now determinants (wanted: at least %d) and the second-order energy '$pt2' is not below $pt2max\"" % n,
            "        say \"CIPSI ended with $now determinants: the second-order energy $pt2 is below $pt2max\"",
            "    fi",
            "    manifest cipsi.ezfio > cipsi.manifest.tmp && mv -f cipsi.manifest.tmp cipsi.manifest",
            "    end_step",
            "fi"]))
    parts.append('say "all steps of this job are done"\nexit 0\n')
    return "\n".join(parts)


_FINGERPRINT = (r"""zcat x.ezfio/ao_basis/ao_coef.gz | awk 'NR==2{n=$1} NR>2 && NR<=2+n {m++; v=$1+0; """
                r"""if (v>1.73204 && v<1.73207) a++; else if (v>2.23605 && v<2.23608) b++; else if (v>3.87296 && v<3.87300) c++} """
                r"""END{printf "%d/%d/%d %d", a, b, c, m}'""")


def _export_lines(source: str, read_wf: str, limit: int, python_bin: str, before: Sequence[str] = (),
                  change_convention: bool = False) -> List[str]:
    """Shell lines of the step ``qpexport``: ``save_for_qmcpack`` on a COPY of ``source`` (the program rewrites the
    EZFIO it runs on), then the test of the exported basis. ``read_wf`` says how the copy learns that it holds
    several determinants: ``yes``, ``no``, or ``if_stored`` (decided in the job from the files).

    The export program handles only the EZFIO convention 20210101. With ``change_convention`` a copy of the newer
    convention gets that number, which is right only when the basis functions are stored in normalized form
    (three conditions, tested here); without it any other convention stops the step."""
    lines = ["if begin_step qpexport; then", "    rm -rf export; mkdir export"] + ["    " + ln for ln in before]
    lines += ["    copy_ezfio %s export/x.ezfio || fail_step \"RUN: the wave function could not be copied for the export\"" % source,
              "    conv=$(value_of export/x.ezfio/ezfio_files/ezfio_convention)"]
    if change_convention:
        lines += [
            "    if [ \"$conv\" != \"%d\" ]; then" % CONVENTION,
            "        [ \"$conv\" -ge 20250211 ] 2>/dev/null || fail_step \"QUALITY: the EZFIO has the unknown convention '$conv'\"",
            "        if [ \"$(value_of export/x.ezfio/basis/ao_normalized)\" != \"T\" ] || [ \"$(value_of export/x.ezfio/basis/primitives_normalized)\" != \"T\" ] \\",
            "               || ls export/x.ezfio/basis/prim_normalization_factor* > /dev/null 2>&1; then",
            "            fail_step \"QUALITY: the EZFIO has the convention $conv and its basis is not stored in normalized form: save_for_qmcpack cannot export it\"",
            "        fi",
            "        sed -i \"s/$conv/%d/\" export/x.ezfio/ezfio_files/ezfio_convention" % CONVENTION,
            "        [ \"$(value_of export/x.ezfio/ezfio_files/ezfio_convention)\" = \"%d\" ] || fail_step \"RUN: the convention of the copy could not be set\"" % CONVENTION,
            "        say \"the copy for the export got the convention %d (the source has $conv)\"" % CONVENTION,
            "    fi"]
    else:
        lines.append("    [ \"$conv\" = \"%d\" ] || fail_step \"QUALITY: the EZFIO has the convention $conv; save_for_qmcpack needs %d\"" % (
            CONVENTION, CONVENTION))
    if read_wf == "yes":
        lines.append("    echo T > export/x.ezfio/determinants/read_wf")
    elif read_wf == "if_stored":
        lines.append("    if [ -e export/x.ezfio/determinants/psi_det.gz ]; then echo T > export/x.ezfio/determinants/read_wf; fi")
    lines += [
        "    # save_for_qmcpack starts a Python converter: it runs with 1 thread and needs a Python with h5py",
        "    path_before=$PATH; %scd export || fail_step \"RUN: the folder export is missing\"" % (
            ("PATH=%s:$PATH; " % python_bin.rstrip("/")) if python_bin else ""),
        "    QP_USE_THREADS=1",
        "    qp_program \"$(step_limit %d)\" save.log save_for_qmcpack x.ezfio" % int(limit),
        "    QP_USE_THREADS=$QP_THREADS; PATH=$path_before; cd ..",
        "    check_qp",
        "    h5=$(ls export/*.h5 2>/dev/null | head -n 1)",
        "    [ -n \"$h5\" ] || fail_step \"RUN: save_for_qmcpack wrote no h5 file (see export/save.log)\"",
        "    [ \"$h5\" = export/QP2QMCPACK.h5 ] || mv \"$h5\" export/QP2QMCPACK.h5",
        "    gzip -t export/x.ezfio/ao_basis/ao_coef.gz 2>/dev/null || fail_step \"RUN: the basis file of the exported copy cannot be read\"",
        "    fp=$( cd export && %s )" % _FINGERPRINT,
        "    echo \"$fp\" > export/fingerprint",
        "    case \"$fp\" in",
        "        \"0/0/0 0\"|\"\") fail_step \"RUN: the basis file of the exported copy holds no value\" ;;",
        "        \"0/0/0 \"*) ;;",
        "        *) fail_step \"QUALITY: the exported basis holds the factors of the known export error ($fp): the orbitals of the h5 file are wrong\" ;;",
        "    esac",
        "    rm -rf export/x.ezfio",
        "    end_step",
        "fi"]
    return lines


def cut_script(cutoff, trial: dict, threads: int, job_seconds: int, timeout: str = "timeout", python_bin: str = "") -> str:
    """Shell text of the cut job of one geometry and one cutoff. It runs in the folder ``cut_<label>`` below the
    folder of the parent:

        trunc     truncate_wf_spin on a copy of the parent (not for ``hf``)
        qpenergy  print_energy on a copy of the cut wave function: QP2 must find the same energy again (not for ``hf``)
        qpexport  save_for_qmcpack on a copy; the basis of the copy is then tested for the factors that the
                  known export error leaves (square roots of 3, 5 and 15)

    The parent is used only if its files are those that its own job recorded (``cipsi.manifest``,
    ``rhf.manifest``): a stored wave function can be days old when another cutoff is asked for."""
    limits = dict(DEFAULT_TRIAL["time_limits"]); limits.update(trial.get("time_limits") or {})
    hf = cutoff == HF_LABEL
    parts = [_tools(threads, job_seconds, int(trial["margin_seconds"]), timeout)]
    if not hf:
        parts.append("\n".join([
            "if begin_step trunc; then",
            "    rm -rf t.ezfio pre.ezfio",
            "    usable ../cipsi.ezfio || fail_step \"RUN: the CIPSI wave function of the parent is not complete\"",
            "    same_manifest ../cipsi.ezfio ../cipsi.manifest || fail_step \"QUALITY: files of the stored CIPSI wave function are missing or changed since its job ended\"",
            "    copy_ezfio ../cipsi.ezfio t.ezfio || fail_step \"RUN: the parent could not be copied\"",
            "    qp_set t.ezfio qmcpack ci_threshold %r" % float(cutoff),
            "    qp_set t.ezfio determinants read_wf true",
            "    qp_program \"$(step_limit %d)\" truncate.log truncate_wf_spin t.ezfio" % int(limits["trunc"]),
            "    check_qp",
            "    need_text truncate.log \"Number of determinants:\"",
            "    usable t.ezfio || fail_step \"RUN: the cut wave function is not complete\"",
            "    copy_ezfio t.ezfio pre.ezfio || fail_step \"RUN: the cut wave function could not be copied\"",
            "    end_step",
            "fi",
            "if begin_step qpenergy; then",
            "    copy_ezfio pre.ezfio pe.ezfio || fail_step \"RUN: the cut wave function could not be copied\"",
            "    qp_program \"$(step_limit %d)\" print_energy.log print_energy pe.ezfio" % int(limits["qpenergy"]),
            "    check_qp",
            "    need_text print_energy.log \"E(CI)\"",
            "    rm -rf pe.ezfio",
            "    end_step",
            "fi"]))
        parts.append("\n".join(_export_lines("pre.ezfio", "yes", limits["qpexport"], python_bin)))
    else:
        check = ["same_manifest ../rhf.ezfio ../rhf.manifest || fail_step \"QUALITY: files of rhf.ezfio are missing or changed since the SCF\""]
        parts.append("\n".join(_export_lines("../rhf.ezfio", "no", limits["qpexport"], python_bin, before=check)))
    parts.append('say "all steps of this job are done"\nexit 0\n')
    return "\n".join(parts)


def import_script(path: str, trial: dict, threads: int, job_seconds: int, timeout: str = "timeout", python_bin: str = "") -> str:
    """Shell text of the job that takes a wave function that exists already (an EZFIO folder at ``path``) instead
    of a CIPSI run and a cut. It runs in the folder ``cut_<label>``:

        qpimport  copy of the folder (the source is never changed); the stored wave function must be complete
        qpenergy  print_energy on a copy: the energy that QP2 finds for this wave function is the reference of the
                  VMC check (a folder that holds only RHF orbitals needs no run: its SCF energy is the reference)
        qpexport  as for a cut; a copy of the newer EZFIO convention gets the number that the export program
                  handles, under the three conditions that make this change right
    """
    limits = dict(DEFAULT_TRIAL["time_limits"]); limits.update(trial.get("time_limits") or {})
    parts = [_tools(threads, job_seconds, int(trial["margin_seconds"]), timeout)]
    parts.append("\n".join([
        "if begin_step qpimport; then",
        "    rm -rf pre.ezfio",
        "    [ -d %s ] || fail_step \"QUALITY: the wave function folder %s does not exist\"" % (path, path),
        "    copy_ezfio %s pre.ezfio || fail_step \"RUN: the wave function folder could not be copied\"" % path,
        "    need_file pre.ezfio/ezfio_files/ezfio_convention",
        "    need_file pre.ezfio/hartree_fock/energy",
        "    need_file pre.ezfio/mo_basis/mo_coef.gz",
        "    if [ -e pre.ezfio/determinants/psi_det.gz ]; then",
        "        usable pre.ezfio || fail_step \"QUALITY: the wave function of %s is not complete\"" % path,
        "    fi",
        "    end_step",
        "fi",
        "if begin_step qpenergy; then",
        "    if [ -e pre.ezfio/determinants/psi_det.gz ]; then",
        "        copy_ezfio pre.ezfio pe.ezfio || fail_step \"RUN: the wave function could not be copied\"",
        "        qp_program \"$(step_limit %d)\" print_energy.log print_energy pe.ezfio" % int(limits["qpenergy"]),
        "        check_qp",
        "        need_text print_energy.log \"E(CI)\"",
        "        rm -rf pe.ezfio",
        "    fi",
        "    end_step",
        "fi"]))
    parts.append("\n".join(_export_lines("pre.ezfio", "if_stored", limits["qpexport"], python_bin, change_convention=True)))
    parts.append('say "all steps of this job are done"\nexit 0\n')
    return "\n".join(parts)


# ============================================================================= the QP2 calculations of one geometry
WAITING, ACTIVE, DONE, FAILED = "WAITING", "ACTIVE", "DONE", "FAILED"      # status of the parent and of a cut
READY, RUNNING = "READY", "RUNNING"                                        # what advance() returns besides FAILED
_IN_QUEUE = (sched.PENDING, sched.RUNNING)
HEARTBEAT = "job.heartbeat"              # a running QP2 job writes the time and its job number into this file once a minute


class Qp2Run:
    """The QP2 calculations of one geometry, in one folder.

    One *parent* (RHF orbitals and the CIPSI wave function) serves every *cut* of that geometry, so a
    second cutoff costs only a truncation and an export. The parent job is submitted again while its
    CIPSI run reaches the time limit and the wave function still grows (at most ``max_segments`` jobs);
    every other failure that a new job can repair gets ``max_attempts`` attempts. Everything this class
    knows is in files of the folder (``qp2.json``), so a new process continues where the old one stopped.

    Two jobs must never work in one folder at the same time. Three rules keep that:
    * the state of a job is kept with the job (key ``judged``), not derived from what a request needs;
    * what follows the end of a job (a further CIPSI job, a second attempt) is written to the record BEFORE
      the new job is submitted (key ``next``), so a submission that fails, or a driver that stops in
      between, is repeated with the same decision and counted once;
    * a job that the batch system does not know (state UNKNOWN) counts as ended only when its own sign of
      life (the file ``job.heartbeat``) is older than ``unknown_seconds``.

    Files of the folder
        geom.xyz, qp2.json, job_parent.sh, rhf.ezfio, cipsi.ezfio, cipsi.bak.ezfio, cipsi.segments, *.manifest, *.log
        cut_<label>/   job_cut.sh, t.ezfio, pre.ezfio, truncate.log, print_energy.log, export/QP2QMCPACK.h5,
                       export/fingerprint, facts.json
    """

    def __init__(self, folder: str, machine, scheduler, clock: Optional[Callable[[], float]] = None,
                 poll_seconds: float = 30.0, settle_seconds: float = 60.0, unknown_seconds: float = 600.0,
                 max_attempts: int = 2, job_table: Optional[JobTable] = None, name: str = "qp2"):
        self.folder = os.path.abspath(folder)
        self.machine, self.scheduler = machine, scheduler
        self.clock = clock or time.time
        self.poll_seconds, self.settle_seconds, self.unknown_seconds = float(poll_seconds), float(settle_seconds), float(unknown_seconds)
        self.max_attempts = int(max_attempts)
        self.jobs = job_table
        self.name = re.sub(r"[^A-Za-z0-9_.+\-]", "_", name)[-40:]
        self.record_file = os.path.join(self.folder, "qp2.json")
        self.rec: Optional[dict] = _read_json(self.record_file) if os.path.exists(self.record_file) else None
        if self.rec is not None and "settings" not in self.rec:       # a record of the first version of this class
            self.rec["settings"] = {"max_segments": int(DEFAULT_TRIAL["max_segments"]),
                                    "margin_seconds": int(DEFAULT_TRIAL["margin_seconds"]),
                                    "time_limits": dict(DEFAULT_TRIAL["time_limits"])}
            for label, cut in self.rec.get("cuts", {}).items():
                cut.setdefault("cutoff", HF_LABEL if label == HF_LABEL else (cut.get("facts") or {}).get("cutoff", float(label[1:])))
                cut.setdefault("import_ezfio", None)
            for part in [self.rec["parent"]] + list(self.rec.get("cuts", {}).values()):
                part.setdefault("next", None)
        self._snapshot: Dict[str, str] = {}
        self._snapshot_time: Optional[float] = None

    # ------------------------------------------------------------------ files
    def _path(self, *parts: str) -> str:
        return os.path.join(self.folder, *parts)

    def _exists(self, *parts: str) -> bool:
        return os.path.exists(self._path(*parts))

    def _save(self) -> None:
        _write_json(self.record_file, self.rec)

    def _text(self, *parts: str) -> str:
        with open(self._path(*parts)) as fh:
            return fh.read()

    @staticmethod
    def _first_line(path: str) -> Optional[str]:
        try:
            with open(path) as fh:
                line = fh.readline().strip()
        except OSError:
            return None
        return line or None

    @staticmethod
    def _new_part() -> dict:
        return {"status": WAITING, "jobs": [], "attempts": 0, "segments": 0, "message": "", "pending_what": None,
                "pending_since": None, "next": None, "facts": {}}

    # ------------------------------------------------------------------ set-up
    def ensure(self, symbols: Sequence[str], coords: np.ndarray, trial: dict) -> None:
        """Create the folder for this geometry, or check that an existing folder holds the same geometry and
        the same settings of the parent. ``symbols`` must be grouped by element (the order of the QMCPACK
        input); ``coords`` in angstrom."""
        coords = np.asarray(coords, dtype=float).reshape(-1, 3)
        symbols = [str(s) for s in symbols]
        n_det_max = None if (trial["cutoff"] == HF_LABEL or trial.get("import_ezfio")) else int(trial["n_det_max"])
        if self.rec is None:
            if os.path.exists(self._path("point.json")):
                raise ValueError("the folder %s belongs to a QMC point; the QP2 calculations need their own folder "
                                 "(setting qp2_key)" % self.folder)
            os.makedirs(self.folder, exist_ok=True)
            lines = ["%d" % len(symbols), "geometry of the QP2 calculations (atoms grouped by element)"]
            lines += ["%-2s %18.10f %18.10f %18.10f" % (s, r[0], r[1], r[2]) for s, r in zip(symbols, coords)]
            _write_text(self._path("geom.xyz"), "\n".join(lines) + "\n")
            limits = dict(DEFAULT_TRIAL["time_limits"]); limits.update(trial.get("time_limits") or {})
            self.rec = {"version": 2, "created": self.clock(), "symbols": symbols, "coords": coords.tolist(),
                        "basis": trial["qp2_basis"], "pseudo": trial["qp2_pseudo"], "n_det_max": n_det_max,
                        # the limits of the jobs of this folder are those of the request that made the folder
                        "settings": {"max_segments": int(trial["max_segments"]), "margin_seconds": int(trial["margin_seconds"]),
                                     "time_limits": limits},
                        "parent": self._new_part(), "cuts": {}}
            self._save()
            return
        r = self.rec
        if list(r["symbols"]) != symbols or float(np.abs(np.asarray(r["coords"]) - coords).max()) > 1e-6:
            raise ValueError("the QP2 folder %s holds another geometry" % self.folder)
        if (r["basis"], r["pseudo"]) != (trial["qp2_basis"], trial["qp2_pseudo"]):
            raise ValueError("the QP2 folder %s was made with the basis %s and the pseudopotential %s, not %s and %s"
                             % (self.folder, r["basis"], r["pseudo"], trial["qp2_basis"], trial["qp2_pseudo"]))
        if n_det_max is not None:
            if r["n_det_max"] is None:
                r["n_det_max"] = n_det_max
                self._save()
            elif int(r["n_det_max"]) != n_det_max:
                raise ValueError("the QP2 folder %s holds a CIPSI run with n_det_max = %d; a cut with n_det_max = %d "
                                 "needs its own folder (setting qp2_key)" % (self.folder, int(r["n_det_max"]), n_det_max))

    def _job_trial(self, trial: dict) -> dict:
        """The trial settings with the job limits of this folder."""
        t = dict(trial)
        t.update({"margin_seconds": self.rec["settings"]["margin_seconds"], "time_limits": dict(self.rec["settings"]["time_limits"])})
        return t

    # ------------------------------------------------------------------ the batch system
    def _parts(self) -> List[Tuple[str, str, dict]]:
        out = [("parent", self.folder, self.rec["parent"])]
        out += [("cut " + label, self._path("cut_" + label), cut) for label, cut in self.rec["cuts"].items()]
        return out

    def _open_jobs(self) -> List[Tuple[str, dict]]:
        """(folder, job) of every job whose end was not judged yet."""
        return [(folder, part["jobs"][-1]) for _name, folder, part in self._parts()
                if part["jobs"] and not part["jobs"][-1].get("judged")]

    def _state(self, job_id: str) -> str:
        now = self.clock()
        if (job_id not in self._snapshot or self._snapshot_time is None
                or now - self._snapshot_time >= self.poll_seconds):
            ids = sorted({job["id"] for _f, job in self._open_jobs()} | {job_id})
            self._snapshot = dict(self.scheduler.states(ids))
            self._snapshot_time = now
        return self._snapshot.get(job_id, sched.UNKNOWN)

    @staticmethod
    def _heartbeat(folder: str) -> Optional[Tuple[float, str]]:
        """(time, job number) of the last sign of life in a folder, or None."""
        try:
            with open(os.path.join(folder, HEARTBEAT)) as fh:
                words = fh.read().split()
            return float(words[0]), (words[1] if len(words) > 1 else "")
        except (OSError, ValueError, IndexError):
            return None

    def _alive(self, folder: str, job_id: str) -> bool:
        """True if the job ``job_id`` wrote its sign of life into the folder less than 3 minutes ago."""
        beat = self._heartbeat(folder)
        return beat is not None and beat[1] == str(job_id) and self.clock() - beat[0] < HEARTBEAT_FRESH

    def _foreign_job(self, part: dict, folder: str) -> Optional[str]:
        """Number of a job that works in the folder now and that the record does not hold: a submission whose
        answer was lost. It is found by its sign of life."""
        beat = self._heartbeat(folder)
        if beat is None or not beat[1] or self.clock() - beat[0] >= HEARTBEAT_FRESH:
            return None
        return None if beat[1] in {job["id"] for job in part["jobs"]} else beat[1]

    def _adopt(self, part: dict, folder: str, job_id: str, steps: Sequence[str], what: str) -> Tuple[str, str]:
        """Take a job that works in the folder into the record, in place of a submission that is still to do or of
        a job that ended without work because it found the other one."""
        if part["jobs"]:
            part["jobs"][-1]["judged"] = True
        part["jobs"].append({"id": str(job_id), "kind": "adopted", "steps": list(steps), "time": self.clock()})
        part["status"], part["next"] = ACTIVE, None
        part["pending_what"], part["pending_since"] = None, None
        part["message"] = "the job %s was found in the folder by its sign of life and taken into the record" % job_id
        self._save()
        if self.jobs is not None:
            self.jobs.add(os.path.relpath(folder, os.path.dirname(self.jobs.path)), str(job_id), folder, self.name, "qp2")
        return RUNNING, "the QP2 job %s of %s runs (the record did not know it; found by its sign of life)" % (job_id, what)

    def _pending(self, part: dict, what: str, wait: float) -> bool:
        """True when the condition ``what`` has held for ``wait`` seconds (files of an ended job can arrive late)."""
        now = self.clock()
        if part.get("pending_what") != what:
            part["pending_what"], part["pending_since"] = what, now
            self._save()
        return now - part["pending_since"] >= wait

    def _clear_pending(self, part: dict) -> None:
        if part.get("pending_what"):
            part["pending_what"], part["pending_since"] = None, None
            self._save()

    def _job_seconds(self) -> int:
        from .qmcpack import parse_time
        seconds = parse_time(self.machine.job("qp2").get("time"))
        if not seconds and self.machine.scheduler_kind != "slurm":
            return 30 * 86400                        # no batch system, no limit: the steps get a very long one
        if not seconds:
            raise ValueError("the machine profile %s gives no time limit for QP2 jobs ([job.qp2] time)" % self.machine.path)
        if seconds <= int(self.rec["settings"]["margin_seconds"]) + 300:
            raise ValueError("the time limit of QP2 jobs ([job.qp2] time = %d s) must be more than 300 s above the margin "
                             "of %d s" % (seconds, int(self.rec["settings"]["margin_seconds"])))
        return int(seconds)

    def _submit(self, part: dict, folder: str, script_name: str, text: str, name: str, kind: str, steps: Sequence[str]) -> str:
        """Write the job script again (the header then holds the limits of the profile as they are now) and submit it.
        The record is saved before anything else, so that a new process knows the job."""
        head = self.machine.header("qp2", name, folder, log=script_name.replace(".sh", ".out"))
        path = os.path.join(folder, script_name)
        _write_text(path, head + text)
        os.chmod(path, 0o755)
        job_id = str(self.scheduler.submit(path, folder, name, None))
        part["jobs"].append({"id": job_id, "kind": kind, "steps": list(steps), "time": self.clock()})
        part["status"], part["next"] = ACTIVE, None
        part["pending_what"], part["pending_since"] = None, None
        self._save()
        self._snapshot[job_id] = sched.PENDING
        if self.jobs is not None:
            self.jobs.add(os.path.relpath(folder, os.path.dirname(self.jobs.path)), job_id, folder, name, "qp2")
        return job_id

    def _fail(self, part: dict, message: str) -> Tuple[str, str]:
        part["status"], part["message"], part["next"] = FAILED, message, None
        part["pending_what"], part["pending_since"] = None, None
        self._save()
        return FAILED, message

    @staticmethod
    def _open_step(folder: str, steps: Sequence[str]) -> Optional[str]:
        for step in steps:
            if not os.path.exists(os.path.join(folder, step + ".done")):
                return step
        return None

    @staticmethod
    def _set_aside(folder: str, steps: Sequence[str], tag: str) -> None:
        """Markers of an attempt that ended: a failed step keeps its reason under another name."""
        for step in steps:
            old = os.path.join(folder, step + ".failed")
            if os.path.exists(old):
                os.replace(old, "%s.%s" % (old, tag))
            started = os.path.join(folder, step + ".started")
            if os.path.exists(started) and not os.path.exists(os.path.join(folder, step + ".done")):
                os.remove(started)

    def _look(self, part: dict, folder: str, needed: Sequence[str], what: str, cipsi: bool) -> Optional[Tuple[str, str]]:
        """Look at the last job of a part whose result is not there yet. Returns ``(RUNNING | FAILED, message)``,
        or None when a job must be submitted now; ``part["next"]`` then says which one.

        ``needed`` are the steps that the present request needs. A job that finished all ITS steps without
        the needed one (the RHF job, when a cut now needs CIPSI) is no failure: a new job follows."""
        job = part["jobs"][-1] if part["jobs"] else None
        if part.get("next") or job is None or job.get("judged"):
            # a submission is to do. If a job works in the folder already (a submission whose answer was lost), it is taken
            lost = self._foreign_job(part, folder)
            if lost:
                return self._adopt(part, folder, lost, (part.get("next") or {}).get("steps") or needed, what)
            if not part.get("next"):
                part["next"] = {"kind": "first", "steps": list(needed), "tag": None}
                self._save()
            return None
        state = self._state(job["id"])
        if state in _IN_QUEUE:
            self._clear_pending(part)
            step = self._open_step(folder, job["steps"])
            return RUNNING, "the QP2 job %s of %s %s (step %s)" % (job["id"], what, "runs" if state == sched.RUNNING else "waits", step)
        if state == sched.UNKNOWN and self._alive(folder, job["id"]):
            self._clear_pending(part)                # the batch system gives no answer, but the job writes its sign of life
            return RUNNING, "the QP2 job %s of %s runs (the batch system gave no answer; the job is alive)" % (job["id"], what)
        lost = self._foreign_job(part, folder)
        if lost:                                     # the job of the record ended because another job holds the folder
            return self._adopt(part, folder, lost, job["steps"], what)
        wait = self.unknown_seconds if state == sched.UNKNOWN else self.settle_seconds
        if not self._pending(part, "ended:%s:%s" % (job["id"], state), wait):
            return RUNNING, "the QP2 job %s of %s left the queue (%s); its files are judged after %.0f s" % (job["id"], what, state, wait)
        if state == sched.UNKNOWN:
            self.scheduler.cancel(job["id"])         # a job that no one can see must not run next to a new one
        step = self._open_step(folder, job["steps"])
        if step is None:
            job["judged"] = True
            part["next"] = {"kind": "first", "steps": list(needed), "tag": None}
            self._save()
            return None
        line = self._first_line(os.path.join(folder, step + ".failed"))
        reason = "step %s: %s" % (step, line) if line else "step %s did not finish and left no reason" % step
        tail = "job %s ended in state %s" % (job["id"], state)
        job["judged"] = True
        if line and line.startswith("QUALITY:"):
            return self._fail(part, "%s; %s" % (reason, tail))
        if state == sched.CANCELLED:
            return self._fail(part, "%s; %s. The job was cancelled outside this program, so it is not submitted again" % (reason, tail))
        timed_out = state == sched.TIMEOUT or bool(line and line.startswith("TIMEOUT:"))
        if cipsi and step == "cipsi" and timed_out:
            segs = self._segments()
            before = segs[-1]["n_start"] if segs else 1
            now = self._stored_n_det()
            ran = (segs[-1].get("end_time", self.clock()) - segs[-1]["start_time"]) if segs else 0.0
            limit = int(self.rec["settings"]["max_segments"])
            if now <= before and ran >= 0.25 * self._job_seconds():
                return self._fail(part, "the CIPSI wave function did not grow in a whole job (%d determinants before and "
                                        "after; %s): raise the time limit of the QP2 jobs ([job.qp2] time) or lower n_det_max"
                                        % (before, tail))
            if part["segments"] + 1 >= limit:
                return self._fail(part, "the CIPSI run reached the time limit in %d jobs and holds %d of the %d wanted "
                                        "determinants (%s): raise [job.qp2] time or max_segments, or lower n_det_max"
                                        % (part["segments"] + 1, now, int(self.rec["n_det_max"]), tail))
            part["segments"] += 1
            part["message"] = "CIPSI job %d reached its time limit at %d determinants; the next job continues" % (part["segments"], now)
            part["next"] = {"kind": "segment", "steps": list(job["steps"]), "tag": "segment%d" % part["segments"]}
        else:
            if part["attempts"] + 1 >= self.max_attempts:
                return self._fail(part, "%s; %s; attempt %d of %d" % (reason, tail, part["attempts"] + 1, self.max_attempts))
            part["attempts"] += 1
            part["message"] = "attempt %d failed (%s; %s): the job was submitted again" % (part["attempts"], reason, tail)
            part["next"] = {"kind": "retry", "steps": list(job["steps"]), "tag": "attempt%d" % part["attempts"]}
        self._save()
        return None

    # ------------------------------------------------------------------ the parent
    def _segments(self) -> List[dict]:
        """The lines of ``cipsi.segments``: one entry per CIPSI job with the number of determinants at its start
        and, if the job wrote it, at its end."""
        out: Dict[int, dict] = {}
        if self._exists("cipsi.segments"):
            for line in self._text("cipsi.segments").splitlines():
                parts = line.split()
                try:
                    if parts[0] == "start" and len(parts) >= 4:
                        out[int(parts[1])] = {"segment": int(parts[1]), "start_time": float(parts[2]), "n_start": int(parts[3])}
                    elif parts[0] == "end" and len(parts) >= 5 and int(parts[1]) in out:
                        out[int(parts[1])].update({"end_time": float(parts[2]), "n_end": int(parts[3]), "rc": int(parts[4])})
                except (ValueError, IndexError):
                    continue
        return [out[k] for k in sorted(out)]

    def _stored_n_det(self) -> int:
        """Determinants of the stored CIPSI wave function, or of its last complete copy; 0 if none can be read."""
        best = 0
        for name in ("cipsi.ezfio", "cipsi.bak.ezfio"):
            try:
                best = max(best, int(ezfio_value(self._path(name, "determinants", "n_det"))))
            except (OSError, ValueError):
                pass
        return best

    def _parent_facts(self, need_cipsi: bool) -> dict:
        facts = dict(self.rec["parent"].get("facts") or {})
        if "e_scf" not in facts:
            log = self._text("scf.log")
            e_log = parse_scf_log(log)
            e_file = _number(ezfio_value(self._path("rhf.ezfio", "hartree_fock", "energy")))
            # 1e-6 Ha. Measured: in one qp_run scf, the value in hartree_fock/energy and the last "* SCF energy"
            # line of the log differed by 1.55e-8 Ha; the file value lies between the energies of two SCF
            # iterations. A wrong or old folder differs by far more than 1e-6 Ha.
            if abs(e_log - e_file) > 1e-6:
                raise ValueError("the SCF energy of the QP2 output (%.10f) differs from the stored one (%.10f)" % (e_log, e_file))
            version = re.findall(r"^Git SHA1\s*:\s*([0-9a-f]+)", log, re.M)
            facts.update({"e_scf": e_file, "mo_num": int(ezfio_value(self._path("rhf.ezfio", "mo_basis", "mo_num"))),
                          "ao_num": int(ezfio_value(self._path("rhf.ezfio", "ao_basis", "ao_num"))),
                          "nup": int(ezfio_value(self._path("rhf.ezfio", "electrons", "elec_alpha_num"))),
                          "ndn": int(ezfio_value(self._path("rhf.ezfio", "electrons", "elec_beta_num"))),
                          "qp2_version": version[-1] if version else None})
        if need_cipsi and "n_det_parent" not in facts:
            n_det = int(ezfio_value(self._path("cipsi.ezfio", "determinants", "n_det")))
            if n_det < 2:
                raise ValueError("the CIPSI wave function holds %d determinant(s)" % n_det)
            summary = None
            logs = sorted((f for f in os.listdir(self.folder) if re.fullmatch(r"cipsi\.\d+\.log", f)),
                          key=lambda f: int(f.split(".")[1]))
            for name in reversed(logs):
                found = [s for s in parse_cipsi_log(self._text(name)) if s["n_det"] == n_det]
                if found:
                    summary = found[-1]
                    break
            segs = self._segments()
            facts.update({"n_det_parent": n_det, "n_det_max": int(self.rec["n_det_max"]), "segments": len(segs),
                          "cipsi_seconds": float(sum(s.get("end_time", s["start_time"]) - s["start_time"] for s in segs)),
                          "e_var_parent": None if summary is None else summary.get("e_var"),
                          "pt2_parent": None if summary is None else summary.get("pt2"),
                          "pt2_error_parent": None if summary is None else summary.get("pt2_error"),
                          "e_pt2_parent": None if summary is None else summary.get("e_pt2")})
        return facts

    def _advance_parent(self, need_cipsi: bool, trial: dict) -> Tuple[str, str]:
        """The parent as far as this request needs it: the RHF orbitals for the cut ``hf``, the CIPSI wave function
        for every other cut. Two requests can ask at the same time (``hf`` while the CIPSI job of another cut runs)."""
        par = self.rec["parent"]
        steps = list(PARENT_STEPS) if need_cipsi else list(PARENT_STEPS[:2])
        if self._exists(steps[-1] + ".done"):
            try:
                facts = self._parent_facts(need_cipsi)
            except (OSError, ValueError, KeyError, IndexError) as exc:
                return self._fail(par, "the result of the QP2 parent job cannot be used: %s" % exc)
            changed = facts != par.get("facts")
            last = par["jobs"][-1] if par["jobs"] else None
            if last is not None and not last.get("judged") and all(self._exists(st + ".done") for st in last["steps"]):
                last["judged"] = True               # the job did all its steps: nothing more to judge
                changed = True
            if (last is None or last.get("judged")) and par["status"] != DONE:
                par.update(status=DONE, message="", next=None, pending_what=None, pending_since=None)
                changed = True
            if changed:
                par["facts"] = facts
                self._save()
            return DONE, ""
        if par["status"] == FAILED:
            return FAILED, par["message"]
        seen = self._look(par, self.folder, steps, "the parent", cipsi=True)
        if seen is not None:
            return seen
        if self._exists(steps[-1] + ".done"):        # the files of the ended job arrived in the meantime
            par["next"] = None
            self._save()
            return self._advance_parent(need_cipsi, trial)
        nxt = par["next"]
        if "cipsi" in nxt["steps"] and self.rec["n_det_max"] is None:
            return self._fail(par, "the QP2 folder %s has no size of the CIPSI wave function (n_det_max)" % self.folder)
        t = self._job_trial(trial)
        t["n_det_max"] = self.rec["n_det_max"]
        text = parent_script(nxt["steps"], t, self.machine.threads("qp2"), self._job_seconds(),
                             timeout=self.machine.executable("timeout"), qp_create=self.machine.executable("qp_create_ezfio"))
        if nxt.get("tag"):
            self._set_aside(self.folder, nxt["steps"], nxt["tag"])
        kind = nxt["kind"]
        job_id = self._submit(par, self.folder, "job_parent.sh", text, self.name + "_p", kind, nxt["steps"])
        return RUNNING, par["message"] if kind != "first" else "the QP2 job %s of the parent was submitted" % job_id

    # ------------------------------------------------------------------ a cut
    def _cut_facts(self, label: str, trial: dict) -> dict:
        folder = self._path("cut_" + label)
        cutoff, imported = trial["cutoff"], trial.get("import_ezfio")
        h5 = os.path.join(folder, "export", "QP2QMCPACK.h5")
        if not (os.path.exists(h5) and os.path.getsize(h5) > 0):
            raise ValueError("the export wrote no file %s" % h5)
        fingerprint = (self._first_line(os.path.join(folder, "export", "fingerprint")) or "").strip()
        counts = fingerprint.split()
        if len(counts) != 2 or counts[0] != "0/0/0" or not counts[1].isdigit() or int(counts[1]) < 1:
            raise ValueError("the exported basis holds the factors of the known export error, or the test could not read it "
                             "(%s)" % (fingerprint or "no record"))
        facts: Dict[str, object] = {"label": label, "cutoff": cutoff, "h5": h5, "fingerprint": fingerprint,
                                    "threads": self.machine.threads("qp2"),
                                    "qp2_basis": self.rec["basis"], "qp2_pseudo": self.rec["pseudo"]}
        if imported:
            pre = os.path.join(folder, "pre.ezfio")
            e_scf = _number(ezfio_value(os.path.join(pre, "hartree_fock", "energy")))
            facts.update({"imported_from": imported, "e_scf": e_scf,
                          "mo_num": int(ezfio_value(os.path.join(pre, "mo_basis", "mo_num")))})
            if os.path.exists(os.path.join(pre, "determinants", "psi_det.gz")):
                n_file = int(ezfio_value(os.path.join(pre, "determinants", "n_det")))
                e_ref = parse_print_energy_log(self._text("cut_" + label, "print_energy.log"))
            else:
                n_file, e_ref = 1, e_scf
            if n_file > 1 and e_ref > e_scf + 1e-8:
                raise ValueError("the imported wave function of %d determinants has the energy %.8f Ha, above the RHF energy "
                                 "%.8f Ha of the same file" % (n_file, e_ref, e_scf))
            facts.update({"n_det": n_file, "e_ref": e_ref})
            return facts
        par = self.rec["parent"]["facts"]
        facts.update({"e_scf": par["e_scf"], "mo_num": par["mo_num"], "qp2_version": par.get("qp2_version")})
        if cutoff == HF_LABEL:
            facts.update({"n_det": 1, "e_ref": par["e_scf"]})
            return facts
        truncate_log = self._text("cut_" + label, "truncate.log")
        n_log, e_trunc = parse_truncate_log(truncate_log)
        n_file = int(ezfio_value(os.path.join(folder, "pre.ezfio", "determinants", "n_det")))
        e_print = parse_print_energy_log(self._text("cut_" + label, "print_energy.log"))
        n_parent = int(par["n_det_parent"])
        n_read = truncate_parent_size(truncate_log)
        if n_read is not None and n_read != n_parent:
            raise ValueError("truncate_wf_spin read a wave function of %d determinants, but the recorded CIPSI wave function "
                             "holds %d: the cut does not come from the recorded parent" % (n_read, n_parent))
        if n_log != n_file:
            raise ValueError("truncate_wf_spin printed %d determinants, the stored wave function holds %d" % (n_log, n_file))
        if abs(e_trunc - e_print) > float(trial["energy_tolerance"]):
            raise ValueError("print_energy finds %.10f Ha for the cut wave function, truncate_wf_spin printed %.10f Ha "
                             "(difference %.2e, limit %.1e)" % (e_print, e_trunc, e_print - e_trunc, float(trial["energy_tolerance"])))
        if n_file < 2:
            raise ValueError("the cutoff %g leaves %d determinant: use a smaller cutoff, or cutoff = hf for one determinant"
                             % (float(cutoff), n_file))
        if n_parent < float(trial["min_parent_ratio"]) * n_file:
            raise ValueError("the cutoff %g keeps %d of the %d determinants of the CIPSI wave function; the CIPSI wave function "
                             "must be at least %g times larger than the cut: raise n_det_max"
                             % (float(cutoff), n_file, n_parent, float(trial["min_parent_ratio"])))
        # a cut wave function lies between its parent and the RHF determinant; anything else means that the cut was
        # made from other orbitals or another wave function than the record says
        low = par.get("e_var_parent")
        if e_trunc > float(par["e_scf"]) + 1e-8 or (low is not None and e_trunc < float(low) - 1e-8):
            raise ValueError("the energy of the cut wave function (%.8f Ha) is not between the energy of the CIPSI wave "
                             "function (%s Ha) and the RHF energy (%.8f Ha)"
                             % (e_trunc, "%.8f" % low if low is not None else "unknown", float(par["e_scf"])))
        facts.update({"n_det": n_file, "e_ref": e_trunc, "print_energy_difference": e_print - e_trunc,
                      "n_det_parent": n_parent, "n_det_max": par.get("n_det_max"), "segments": par.get("segments"),
                      "cipsi_seconds": par.get("cipsi_seconds"), "e_var_parent": low,
                      "pt2_parent": par.get("pt2_parent"), "pt2_error_parent": par.get("pt2_error_parent"),
                      "e_pt2_parent": par.get("e_pt2_parent")})
        return facts

    def _advance_cut(self, label: str, trial: dict) -> Tuple[str, str]:
        cutoff, imported = trial["cutoff"], trial.get("import_ezfio")
        identity = {"cutoff": cutoff, "import_ezfio": imported}
        cut = self.rec["cuts"].get(label)
        if cut is None:
            cut = self.rec["cuts"][label] = dict(self._new_part(), **identity)
            self._save()
        same = cut.get("import_ezfio") == imported and (
            cut["cutoff"] == cutoff or (not isinstance(cutoff, str) and not isinstance(cut["cutoff"], str)
                                        and abs(float(cut["cutoff"]) - float(cutoff)) <= 1e-12 * abs(float(cutoff))))
        if not same:
            raise ValueError("the cut %s of the QP2 folder %s was made for cutoff %r (import: %s), not for cutoff %r "
                             "(import: %s): two different wave functions must have two labels"
                             % (label, self.folder, cut["cutoff"], cut.get("import_ezfio"), cutoff, imported))
        folder = self._path("cut_" + label)
        if imported:
            steps = list(IMPORT_STEPS)
        else:
            steps = ["qpexport"] if cutoff == HF_LABEL else list(CUT_STEPS)
        if cut["status"] == DONE:
            return READY, ""
        if cut["status"] == FAILED:
            return FAILED, cut["message"]
        if os.path.exists(os.path.join(folder, "qpexport.done")):
            try:
                facts = self._cut_facts(label, trial)
            except (OSError, ValueError, KeyError, IndexError) as exc:
                return self._fail(cut, "the cut %s cannot be used: %s" % (label, exc))
            cut["facts"], cut["status"], cut["message"], cut["next"] = facts, DONE, "", None
            cut["pending_what"], cut["pending_since"] = None, None
            if cut["jobs"]:
                cut["jobs"][-1]["judged"] = True
            _write_json(os.path.join(folder, "facts.json"), facts)
            self._save()
            return READY, ""
        seen = self._look(cut, folder, steps, "the cut %s" % label, cipsi=False)
        if seen is not None:
            return seen
        if os.path.exists(os.path.join(folder, "qpexport.done")):
            cut["next"] = None
            self._save()
            return self._advance_cut(label, trial)
        nxt = cut["next"]
        os.makedirs(folder, exist_ok=True)
        t = self._job_trial(trial)
        args = dict(threads=self.machine.threads("qp2"), job_seconds=self._job_seconds(), timeout=self.machine.executable("timeout"),
                    python_bin=self.machine.cp.get("executables", "qp2_python_bin", fallback="").strip())
        text = import_script(imported, t, **args) if imported else cut_script(cutoff, t, **args)
        if nxt.get("tag"):
            self._set_aside(folder, steps, nxt["tag"])
        kind = nxt["kind"]
        job_id = self._submit(cut, folder, "job_cut.sh", text, "%s_%s" % (self.name, label), kind, steps)
        return RUNNING, cut["message"] if kind != "first" else "the QP2 job %s of the cut %s was submitted" % (job_id, label)

    # ------------------------------------------------------------------ public
    def advance(self, trial: dict) -> Tuple[str, str]:
        """Bring the QP2 calculations for the cut of ``trial`` one step further: start a job, judge an ended job,
        submit it again, or read its result. Returns ``(READY | RUNNING | FAILED, message)``. It never waits."""
        if self.rec is None:
            raise ValueError("the QP2 folder %s was not created (call ensure first)" % self.folder)
        if not trial.get("import_ezfio"):
            state, message = self._advance_parent(trial["cutoff"] != HF_LABEL, trial)
            if state != DONE:
                return state, message
        return self._advance_cut(cut_id(trial), trial)

    def facts(self, trial: dict) -> dict:
        """What is known about the cut of ``trial`` once it is READY (see :meth:`_cut_facts`)."""
        return dict(self.rec["cuts"][cut_id(trial)]["facts"])

    def cancel(self, keep_parent: bool = False, keep_cuts: Sequence[str] = ()) -> List[str]:
        """Cancel the jobs of this folder that are in the queue, except the parent job when ``keep_parent`` and the
        jobs of the cuts whose label is in ``keep_cuts`` (another point still needs them). A cancelled job counts as
        judged, and its part waits for the next request: a point that asks for the folder later gets a new job,
        which continues from the files of the cancelled one. Returns the ids of the cancelled jobs."""
        out: List[str] = []
        if self.rec is None:
            return out
        keep = set(keep_cuts)
        for name, _folder, part in self._parts():
            if (name == "parent" and keep_parent) or (name != "parent" and name[len("cut "):] in keep):
                continue
            if not part["jobs"] or part["jobs"][-1].get("judged"):
                continue
            job = part["jobs"][-1]
            state = self._state(job["id"])
            if state in _IN_QUEUE or state == sched.UNKNOWN:
                self.scheduler.cancel(job["id"])
                self._snapshot[job["id"]] = sched.CANCELLED
                out.append(job["id"])
            job["judged"] = True
            if part["status"] != FAILED:
                part.update(status=WAITING, next=None, pending_what=None, pending_since=None,
                            message="the job %s was cancelled because no point needs it any more" % job["id"])
        self._save()
        return out

    def retry(self) -> List[str]:
        """Give every failed part a new set of attempts (after a person repaired the cause). A job of such a part
        that could still run is cancelled first. Returns the names of the parts."""
        out = []
        if self.rec is None:
            return out
        for name, folder, part in self._parts():
            if part["status"] != FAILED:
                continue
            if part["jobs"]:
                last = part["jobs"][-1]
                if self._state(last["id"]) in _IN_QUEUE or self._alive(folder, last["id"]):
                    self.scheduler.cancel(last["id"])
                    self._snapshot[last["id"]] = sched.CANCELLED
                last["judged"] = True
            n = int(part.get("retries", 0)) + 1
            self._set_aside(folder, PARENT_STEPS + CUT_STEPS + IMPORT_STEPS, "retry%d" % n)
            part.update({"status": WAITING, "attempts": 0, "segments": 0, "message": "", "retries": n, "next": None,
                         "pending_what": None, "pending_since": None})
            out.append(name)
        if out:
            self._save()
        return out

    def _step_hours(self, folder: str, steps: Sequence[str]) -> float:
        seconds = 0.0
        for step in steps:
            start, end = _marker(os.path.join(folder, step + ".started")), _marker(os.path.join(folder, step + ".done"))
            if start is not None and end is not None:
                seconds += max(0.0, end - start)
        return seconds * self.machine.threads("qp2") / 3600.0

    def cost_parent(self) -> float:
        """Core-hours of the parent so far: the finished steps from their marker files, and every CIPSI job from
        ``cipsi.segments``. A CIPSI job that wrote no end line counts up to the start of the next one, or up to
        now while it runs."""
        if self.rec is None:
            return 0.0
        par = self.rec["parent"]
        running = (bool(par["jobs"]) and not par["jobs"][-1].get("judged") and par["status"] != FAILED
                   and not self._exists("cipsi.done"))
        segs = self._segments()
        seconds = 0.0
        for i, seg in enumerate(segs):
            if "end_time" in seg:
                end = seg["end_time"]
            elif i + 1 < len(segs):
                end = segs[i + 1]["start_time"]
            else:
                end = self.clock() if running else seg["start_time"]
            seconds += max(0.0, end - seg["start_time"])
        return self._step_hours(self.folder, ("qpcreate", "qpscf")) + seconds * self.machine.threads("qp2") / 3600.0

    def cost_cut(self, trial: dict) -> float:
        """Core-hours of the finished steps of the cut of ``trial``."""
        if self.rec is None:
            return 0.0
        return self._step_hours(self._path("cut_" + cut_id(trial)), tuple(CUT_STEPS) + ("qpimport",))

    def cost(self) -> float:
        """Core-hours of all QP2 steps of this geometry so far."""
        if self.rec is None:
            return 0.0
        return self.cost_parent() + sum(self._step_hours(self._path("cut_" + label), tuple(CUT_STEPS) + ("qpimport",))
                                        for label in self.rec["cuts"])


def _marker(path: str) -> Optional[float]:
    try:
        with open(path) as fh:
            return float(fh.read().split()[0])
    except (OSError, ValueError, IndexError):
        return None


# ============================================================================= the steps of the QMC job
def md_vmc_input(system: dict, determinantset: str, pseudo_dir: str = "", seed: Optional[int] = None,
                 project: str = "mdvmc1", walkers: int = 32, blocks: int = 60, steps: int = 40, timestep: float = 0.3,
                 substeps: int = 1, warmup: int = 100) -> str:
    """VMC run of the determinants alone (NO Jastrow element), for the comparison with the QP2 energy.
    ``walkers`` is the number of walkers of every MPI rank, so a block holds ranks x walkers x steps samples."""
    from . import qmcpack_inputs as Q

    parts = [Q._head(project, seed), '   <qmcsystem>', Q._particlesets(system),
             Q._wavefunction(system, None, None, "trial.h5", determinantset=determinantset),
             Q._hamiltonian(system, pseudo_dir), '   </qmcsystem>',
             '   <qmc method="vmc" move="pbyp" checkpoint="-1">',
             '      <parameter name="walkers"             >    %d              </parameter>' % int(walkers),
             '      <parameter name="warmupSteps"         >    %d              </parameter>' % int(warmup),
             '      <parameter name="blocks"              >    %d              </parameter>' % int(blocks),
             '      <parameter name="steps"               >    %d              </parameter>' % int(steps),
             '      <parameter name="subSteps"            >    %d               </parameter>' % int(substeps),
             '      <parameter name="timestep"            >    %s             </parameter>' % timestep,
             '      <parameter name="useDrift"            >    yes             </parameter>',
             '   </qmc>', '</simulation>', '']
    return "\n".join(parts)


def qmc_steps_text(steps: Sequence[str], limits: Dict[str, int], launch: str, python: str = "python", qmcpack: str = "qmcpack",
                   convert4qmc: str = "convert4qmc", determinants_tools: str = "determinants_tools.py", h5_source: str = "",
                   ndet: int = 2) -> List[str]:
    """Shell blocks of the steps ``mdc4q`` and ``mdvmc`` for the job script of a point (see ``steps_script`` of
    the engine, which supplies the shell functions used here). ``h5_source`` is the h5 file that QP2 exported,
    ``ndet`` the number of determinants that QP2 wrote into it (1 = the RHF determinant alone)."""
    parts = []
    if "mdc4q" in steps:
        lines = ["if begin_step mdc4q; then",
                 "    rm -f trial.h5 mdc4q.structure.xml mdc4q.wfj.xml md_determinantset.xml",
                 "    need_file trial_facts.json",
                 "    cp %s trial.h5 || fail_step \"RUN: the h5 file of QP2 could not be copied\"" % h5_source]
        if int(ndet) > 1:
            lines += ["    # QP2 writes the determinants as signed numbers; QMCPACK needs unsigned ones",
                      "    run_limited %d mdtransform.out - %s %s --transform -s trial.h5" % (limits["mdc4q"], python, determinants_tools),
                      "    check_run",
                      "    run_limited %d mdc4q.out - %s -orbitals trial.h5 -prefix mdc4q -multidet trial.h5" % (limits["mdc4q"], convert4qmc)]
        else:
            lines += ["    run_limited %d mdc4q.out - %s -orbitals trial.h5 -prefix mdc4q" % (limits["mdc4q"], convert4qmc)]
        lines += ["    check_run",
                  "    need_file mdc4q.structure.xml; need_file mdc4q.wfj.xml",
                  "    md_package check-mdc4q",
                  "    need_file md_determinantset.xml",
                  "    end_step",
                  "fi"]
        parts.append("\n".join(lines))
    if "mdvmc" in steps:
        parts.append("\n".join([
            "if begin_step mdvmc; then",
            "    # runs that an earlier attempt of this job finished are kept (a run is long); everything after them is removed",
            "    run=0",
            "    while [ -s mdvmc$(( run + 1 )).s000.scalar.dat ] && grep -q -- \"%s\" mdvmc$(( run + 1 )).out 2>/dev/null; do" % QMCPACK_SUCCESS,
            "        run=$(( run + 1 ))",
            "    done",
            "    for old in mdvmc[0-9]*.in.xml mdvmc[0-9]*.out mdvmc[0-9]*.s000.scalar.dat; do",
            "        [ -e \"$old\" ] || continue",
            "        n=${old#mdvmc}; n=${n%%.*}",
            "        if [ \"$n\" -gt \"$run\" ]; then rm -f mdvmc$n.*; fi",
            "    done",
            "    rm -f check_mdvmc.json",
            "    if [ \"$run\" -gt 0 ]; then say \"$run finished run(s) of the VMC check are kept\"; fi",
            "    while true; do",
            "        if [ \"$run\" -gt 0 ]; then",
            "            run_limited 900 check-mdvmc.out - %s -m montepass.engine.multidet check-mdvmc" % python,
            "            if [ \"$RC\" -eq %d ]; then fail_step \"QUALITY: $(tail -n 1 check-mdvmc.out)\"; fi" % QUALITY_EXIT,
            "            if [ \"$RC\" -ne %d ]; then check_run; break; fi" % MORE_EXIT,
            "        fi",
            "        run=$(( run + 1 ))",
            "        md_package write-mdvmc",
            "        need_file mdvmc$run.in.xml",
            "        run_limited %d mdvmc$run.out mdvmc$run.err %s %s mdvmc$run.in.xml" % (limits["mdvmc"], launch, qmcpack),
            "        check_run",
            "        need_text mdvmc$run.out \"%s\"" % QMCPACK_SUCCESS,
            "        need_file mdvmc$run.s000.scalar.dat",
            "    done",
            "    end_step",
            "fi"]))
    return parts


MD_PACKAGE_FUNCTION = r'''
# md_package COMMAND: a command of the module for several determinants. Exit code QUALITY_EXIT = a quality check failed.
md_package() {
    run_limited 900 "$1.out" - PYTHON_COMMAND -m montepass.engine.multidet "$1"
    if [ "$RC" -eq QUALITY_EXIT ]; then fail_step "QUALITY: $(tail -n 1 "$1.out")"; fi
    check_run
}
'''

_COMMANDS = {"check-mdc4q": "check_mdc4q.json", "write-mdvmc": None, "check-mdvmc": "check_mdvmc.json"}


def check_seed(first: Optional[int], run: int, ranks: int) -> Optional[int]:
    """Seed of run ``run`` (1, 2, ...) of the VMC check: every run gets its own window of prime numbers below the
    window of the first one (a QMCPACK run on R ranks with 1 thread uses 2 R positions of the table, modulo 1024)."""
    if first is None:
        return None
    return int((int(first) - (int(run) - 1) * 2 * int(ranks)) % 1024)


def check_run_limit(max_runs: int, ranks: int) -> int:
    """Largest number of runs of the VMC check: the table of QMCPACK has 1024 positions, a run uses 2 x ranks of them,
    and two runs that are combined must not share one (2 runs on 256 ranks, 4 on 128, ``max_runs`` on fewer)."""
    return max(1, min(int(max_runs), 1024 // (2 * max(1, int(ranks)))))


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Commands that the job script of a point runs in the folder of the point.

        python -m montepass.engine.multidet check-mdc4q     files of convert4qmc against QP2 and the SCF
        python -m montepass.engine.multidet write-mdvmc     input of the next VMC run without Jastrow factor
        python -m montepass.engine.multidet check-mdvmc     the combined energy of all runs against the QP2 energy

    Exit code 0: done. Exit code 3: a quality check failed; the last line of the output says why. Exit code 4
    (check-mdvmc only): no verdict yet, one more VMC run is needed. Any other exit code: the command itself
    could not run."""
    from . import qmcpack_inputs as Q

    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1 or args[0] not in _COMMANDS:
        print(main.__doc__)
        return 2
    command = args[0]
    cfg = _read_json("qmc_settings.json") if os.path.exists("qmc_settings.json") else {}
    trial = copy.deepcopy(DEFAULT_TRIAL); trial.update(cfg.get("trial") or {})

    if command == "check-mdc4q":
        result = check_export(".", rhf_tolerance=float(trial["rhf_tolerance"]))
        _write_json(_COMMANDS[command], result)
        print(result["message"])
        return 0 if result["ok"] else QUALITY_EXIT
    if command == "check-mdvmc":
        result = check_md_vmc(".", trial["check_sigmas"], trial["check_tolerance"], trial["check_sigma_max"],
                              trial["check_fail_sigmas"],
                              check_run_limit(trial["check_max_runs"], (cfg.get("mdvmc") or {}).get("ranks", 1)))
        _write_json(_COMMANDS[command], result)
        print(result["message"])
        return 0 if result["ok"] else (MORE_EXIT if result["verdict"] == "MORE" else QUALITY_EXIT)
    # write-mdvmc: the first run is short and measures the noise; a later run is as long as the wanted error bar needs
    system = Q.load_system("system.json")
    Q.check_pseudopotentials(system, cfg.get("pseudo_dir", ""))
    with open("md_determinantset.xml") as fh:
        block = fh.read().rstrip("\n")
    check = dict(cfg.get("mdvmc") or {})
    done = []
    while os.path.exists("mdvmc%d.s000.scalar.dat" % (len(done) + 1)):
        try:
            done.append(_vmc_stats("mdvmc%d.s000.scalar.dat" % (len(done) + 1)))
        except (OSError, ValueError, KeyError) as exc:
            print("run %d of the VMC check gave no usable energy: %s" % (len(done) + 1, exc))
            return QUALITY_EXIT
    run = len(done) + 1
    allowed = check_run_limit(trial["check_max_runs"], check.get("ranks", 1))
    if run > allowed:
        print("the VMC check has made its %d runs already" % allowed)
        return QUALITY_EXIT
    if not done:
        blocks = int(trial["check_pilot_blocks"])
        note = "the short first run"
    else:
        _mean, err_now = combine_runs([(m, e) for m, e, _n in done])
        blocks = plan_check_blocks(err_now, done[0][2], done[0][1], float(trial["check_sigma_max"]), int(trial["check_max_blocks"]),
                                   blocks_total=sum(n for _m, _e, n in done))
        note = "the runs so far have the error %.6f Ha" % err_now
    text = md_vmc_input(system, block, pseudo_dir=cfg.get("pseudo_dir", ""),
                        seed=check_seed(check.get("seed"), run, int(check.get("ranks", 1))), project="mdvmc%d" % run,
                        walkers=int(trial["check_walkers"]), blocks=blocks, steps=int(trial["check_steps"]))
    _write_text("mdvmc%d.in.xml" % run, text)
    print("wrote mdvmc%d.in.xml (%d blocks; %s)" % (run, blocks, note))
    return 0


if __name__ == "__main__":
    sys.exit(main())
