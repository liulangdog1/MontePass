"""The backend that runs real calculations through a batch system.

    ORCA      the surrogate path: optimized end points, the path between them, the saddle
    PySCF     the local surrogate of the search: Kohn-Sham energy, gradient and Hessian in the search
              directions, at the same level of theory as the QMC trial orbitals
    QMCPACK   the DMC energies (the engine)

Everything that depends on the computer comes from the machine profile that the input file names.
Folders of a project:

    work/surrogate/<job>/        ORCA calculations                   (job table work/surrogate/jobs.json)
    work/pyscf/<label>/<tag>/    PySCF surrogate calculations        (job table work/pyscf/jobs.json)
    work/qmc/<point>/            QMC points                          (job table work/qmc/jobs.json)
    work/qmc/trial/...           points of the stage "trial", and the Quantum Package 2 calculations of their
                                 geometries                          (job table work/qmc/qp2_jobs.json)
    work/autotune/pilot/         state of the pilot runs that choose the QMC settings

Why two programs give surrogate data. The path needs a program with a path search, and ORCA has
one. The search on the DMC surface subtracts the surrogate energy of every DMC point from its DMC
energy (a control variate), and it adds the surrogate slope and curvature back. That is exact only
if energy, slope and curvature belong to the same function. The Kohn-Sham energy of a DMC point is
a by-product of its trial orbitals (PySCF, pseudopotentials), so the slope and the curvature are
computed with the same PySCF settings.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import geometry as G
from .config import Config
from .machine import Machine, find_profile
from .scheduler.base import JobTable
from .workflow import Backend, chosen_path_targets, with_path_images

MAX_ORCA_ATTEMPTS = 2           # a batch job that ends without writing a result is submitted once more


class ClusterBackend(Backend):
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.machine = Machine(find_profile(cfg.machine_profile, [cfg.folder]))
        if cfg.trial is not None:
            # the stage "trial" starts after the whole search: a profile that cannot run it is refused now
            from .trial import machine_problems
            problems = machine_problems(self.machine, cfg.trial)
            if problems:
                from .config import TrialSetupError
                raise TrialSetupError("%s asks for [trial] kind = multidet, but the machine profile %s cannot run that "
                                      "stage:\n  %s" % (cfg.path, self.machine.path, "\n  ".join(problems)))
        self._scheduler = None
        self._engine = None
        self._pilot = None
        self._surrogates: Dict[str, object] = {}
        self._tables: Dict[str, JobTable] = {}

    # ------------------------------------------------------------------ shared objects
    @property
    def scheduler(self):
        if self._scheduler is None:
            self._scheduler = self.machine.scheduler()
        return self._scheduler

    @staticmethod
    def _dir(wf, *parts: str) -> str:
        """A folder below ``work/``; it is created."""
        d = os.path.join(wf.project.work, *parts)
        os.makedirs(d, exist_ok=True)
        return d

    def _table(self, wf, part: str) -> JobTable:
        """One job table per part of the project. Every object of that part shares the same table object."""
        if part not in self._tables:
            self._tables[part] = JobTable(os.path.join(self._dir(wf, part), "jobs.json"))
        return self._tables[part]

    def _scf_settings(self) -> Dict[str, object]:
        q = self.cfg.qmc
        return {"charge": q["charge"], "spin": q["spin"], "xc": q["xc"], "basis": q["basis"], "ecp": q["ecp"]}

    # ------------------------------------------------------------------ ORCA jobs
    def _orca_job(self, wf, folder: str, name: str):
        from .surrogate.orca import OrcaJob
        return OrcaJob(self._dir(wf, "surrogate", folder), self.machine, self.scheduler,
                       self._table(wf, "surrogate"), name=name, job_name="mpass_%s" % folder)

    def _ready(self, wf, job, what: str) -> bool:
        """Submit the job if that was not done, and say whether its result can be read.
        A failure raises RuntimeError with the reason."""
        from .surrogate import orca as O
        first = not job.submitted()
        job.submit()
        if first:
            wf.project.log("%s: ORCA job %s submitted (%s)" % (what, job.job_id(), job.folder))
        st = job.status()
        if st.state == O.DONE:
            return True
        if st.state == O.FAILED:
            wrote_reason = os.path.exists(job.file(job.step + ".failed")) or os.path.exists(job.file(job.step + ".done"))
            if not wrote_reason and job.job_table.attempts(job.key) < MAX_ORCA_ATTEMPTS:
                # the batch job ended without a verdict of its own (a node failure, for example): once more
                wf.project.log("%s: %s. The job is submitted once more." % (what, st.message))
                job.resubmit()
                return False
            asked = wf.project.state.get("retry_request")
            last = (job.job_table.current(job.key) or {}).get("time")
            if asked and last is not None and float(last) < float(asked):
                # the command "retry" was given after this job was submitted: once more
                wf.project.log("%s: submitted again by request (it had failed: %s)" % (what, st.message))
                job.resubmit()
                return False
            raise RuntimeError("%s failed: %s" % (what, st.message))
        return False

    def _orca_options(self) -> Dict[str, object]:
        cfg = self.cfg
        return {"charge": cfg.charge, "multiplicity": cfg.multiplicity, "method": cfg.surrogate_method,
                "basis": cfg.surrogate_basis}

    # ------------------------------------------------------------------ geometries
    def geometries(self, wf) -> Optional[dict]:
        mode = self.cfg.mode
        if mode == "single":
            return self._single(wf)
        if mode == "images":
            return self._images(wf)
        return self._path(wf)

    def _check_frequencies(self, wf, job, label: str, kind: str) -> None:
        """Write a warning into the log when the frequencies do not fit the kind of stationary point."""
        try:
            problem = job.hessian().problem(kind)
        except (OSError, ValueError) as exc:
            problem = "the Hessian file could not be read (%s)" % exc
        notes = wf.project.state.setdefault("surrogate_notes", {})
        if problem and notes.get(label) != problem:
            notes[label] = problem
            wf.project.log("WARNING, %s: %s" % (label, problem))

    def _single(self, wf) -> Optional[dict]:
        cfg = self.cfg
        symbols, x = G.read_xyz(cfg.geometry)
        kind = cfg.kinds["geometry"]
        if cfg.surrogate_program == "orca" and cfg.surrogate_optimize:
            job = self._orca_job(wf, "opt_geometry", "opt")
            job.write_optimization(symbols, x, saddle=(kind == "saddle"), frequencies=True, **self._orca_options())
            if not self._ready(wf, job, "optimization of the geometry"):
                return None
            symbols, x = job.geometry()
            self._check_frequencies(wf, job, "geometry", kind)
        return {"symbols": list(symbols), "frames": None, "energies": None,
                "targets": {"geometry": {"coords": x, "kind": kind}}}

    def _images(self, wf) -> Optional[dict]:
        cfg = self.cfg
        frames = G.read_xyz_frames(cfg.images)
        symbols = [G.normalize_symbol(s) for s in frames[0][0]]
        for i, f in enumerate(frames):
            if [G.normalize_symbol(s) for s in f[0]] != symbols:
                raise ValueError("%s: image %d has other atoms or another atom order than image 0" % (cfg.images, i))
        coords = [f[1] for f in frames]
        targets = {}
        if cfg.selection == "list" and not cfg.kinds_auto:
            for t in cfg.targets:
                i = int(t)
                if i >= len(coords):
                    raise ValueError("[geometry] targets: image %d does not exist; %s holds the images 0 to %d"
                                     % (i, cfg.images, len(coords) - 1))
                targets[t] = {"coords": coords[i], "kind": cfg.kinds[t]}
        else:
            # the package takes the targets (barrier, mep) or their kinds (kinds = auto) from the path
            chosen, note = chosen_path_targets(cfg, [G.comment_energy(f[2]) for f in frames], "the path file %s" % cfg.images)
            for i, kind in chosen.items():
                targets[str(i)] = {"coords": coords[i], "kind": kind}
            notes = wf.project.state.setdefault("surrogate_notes", {})
            if notes.get("targets") != note:
                notes["targets"] = note
                wf.project.log(note)
        return {"symbols": symbols, "frames": coords if len(coords) >= 2 else None, "energies": None, "targets": targets}

    def _path(self, wf) -> Optional[dict]:
        cfg = self.cfg
        sym, xr = G.read_xyz(cfg.reactant)
        sym = [G.normalize_symbol(s) for s in sym]
        sym_p, xp = G.read_xyz(cfg.product)
        if [G.normalize_symbol(s) for s in sym_p] != sym:
            raise ValueError("the reactant and the product must have the same atoms in the same order (%s, %s)"
                             % (cfg.reactant, cfg.product))
        guess = None
        if cfg.saddle_guess:
            sym_g, guess = G.read_xyz(cfg.saddle_guess)
            if [G.normalize_symbol(s) for s in sym_g] != sym:
                raise ValueError("the saddle guess must have the atoms of the reactant in the same order (%s)" % cfg.saddle_guess)
        opts = self._orca_options()
        if cfg.surrogate_optimize:
            jobs = []
            for label, x in (("reactant", xr), ("product", xp)):
                job = self._orca_job(wf, "opt_" + label, "opt")
                job.write_optimization(sym, x, saddle=False, frequencies=cfg.surrogate_frequencies, **opts)
                jobs.append((label, job))
            ready = [self._ready(wf, job, "optimization of the %s" % label) for label, job in jobs]     # start both
            if not all(ready):
                return None
            xr, xp = jobs[0][1].geometry()[1], jobs[1][1].geometry()[1]
            if cfg.surrogate_frequencies:
                for label, job in jobs:
                    self._check_frequencies(wf, job, label, "minimum")
        neb = self._orca_job(wf, "neb", "neb")
        neb.write_neb(sym, xr, xp, saddle_guess=guess, images=cfg.neb_images, kind=cfg.neb_kind, require_saddle=False, **opts)
        if not self._ready(wf, neb, "path search"):
            return None
        _s, frames, energies = neb.path()
        targets: Dict[str, dict] = {}
        if "reactant" in cfg.targets:
            targets["reactant"] = {"coords": frames[0], "kind": "minimum"}
        if "product" in cfg.targets:
            targets["product"] = {"coords": frames[-1], "kind": "minimum"}
        if "saddle" in cfg.targets:
            _s2, xs, saddle_kind = neb.saddle()
            result = neb.neb_result()
            note = {"TS": "the saddle is the optimized transition state of the path search",
                    "CI": "the saddle is the climbing image of the path (no converged saddle optimization)"}.get(saddle_kind)
            if note is None:
                raise RuntimeError("the path search in %s gave no saddle geometry" % neb.folder)
            notes = wf.project.state.setdefault("surrogate_notes", {})
            if notes.get("saddle") != note:
                notes["saddle"] = note
                wf.project.log("saddle: %s%s" % (note, "" if result.get("saddle_converged") or saddle_kind == "CI"
                                                 else "; the output does not report a converged saddle optimization"))
            targets["saddle"] = {"coords": xs, "kind": "saddle"}
        ordered = {t: targets[t] for t in cfg.targets}
        if cfg.selection == "mep":
            ordered = with_path_images(ordered, list(frames))
            notes = wf.project.state.setdefault("surrogate_notes", {})
            note = "targets = mep: %d searches (%s)" % (len(ordered), ", ".join("%s: %s" % (n, t["kind"]) for n, t in ordered.items()))
            if notes.get("targets") != note:
                notes["targets"] = note
                wf.project.log(note)
        return {"symbols": sym, "frames": list(frames), "energies": [float(e) for e in energies], "targets": ordered}

    # ------------------------------------------------------------------ surrogate of the search
    def surrogate(self, wf, vectors: np.ndarray, label: str):
        if label not in self._surrogates:
            from .surrogate.pyscf_jobs import PyscfSurrogate
            self._surrogates[label] = PyscfSurrogate(self._dir(wf, "pyscf", label), self.machine, self.scheduler,
                                                     wf.project.state["symbols"], vectors, settings=self._scf_settings(),
                                                     job_table=self._table(wf, "pyscf"))
        return self._surrogates[label]

    def retry_surrogates(self, wf) -> List[str]:
        """Submit again every failed PySCF job of the surrogate (command ``retry``). Returns one name per job."""
        from .surrogate.pyscf_jobs import PyscfSurrogate
        root = os.path.join(wf.project.work, "pyscf")
        out: List[str] = []
        if not os.path.isdir(root):
            return out
        for label in sorted(os.listdir(root)):
            if not os.path.isdir(os.path.join(root, label)):
                continue
            sur = self._surrogates.get(label)
            if sur is None:                     # only the jobs are needed here, not the directions
                sur = PyscfSurrogate(os.path.join(root, label), self.machine, self.scheduler,
                                     wf.project.state["symbols"], None, settings=self._scf_settings(),
                                     job_table=self._table(wf, "pyscf"))
            out.extend("%s/%s" % (label, tag) for tag in sur.retry_failed())
        return out

    def moved_energy_source(self, wf):
        """Surrogate energies of single geometries, for the test with a moved surrogate (``shifted.py``)."""
        n3 = 3 * len(wf.project.state["symbols"])
        return self.surrogate(wf, np.zeros((0, n3)), "moved")

    @staticmethod
    def _hessian_at(h, x: np.ndarray, tolerance: float = 1e-4) -> Optional[np.ndarray]:
        """The Hessian of an ORCA Hessian file in the frame of the geometry ``x``, or None if the file belongs
        to another geometry. ORCA writes the file in its own frame (shifted to the centre of mass), so the
        geometry of the file is first put on ``x`` and the Hessian is rotated with it."""
        if h.coords.shape != x.shape:
            return None
        rot, centre_file, centre_x, _rmsd = G.superposition(h.coords, x)
        aligned = (h.coords - centre_file) @ rot + centre_x
        if np.abs(aligned - x).max() > tolerance:
            return None
        return G.rotate_hessian(h.hessian, rot)

    def cartesian_hessian(self, wf, tag: str, coords: np.ndarray) -> Optional[np.ndarray]:
        """Full Cartesian Hessian at a target geometry (used to choose soft modes as the search subspace).

        It is taken from the ORCA optimization of that geometry when the job computed frequencies and
        its final geometry is the geometry asked for. Else ORCA computes frequencies at the fixed
        geometry. With ``[surrogate] program = none`` PySCF differentiates its gradient in all 3 N
        Cartesian directions.
        """
        cfg = self.cfg
        x = np.asarray(coords, dtype=float).reshape(-1, 3)
        symbols = wf.project.state["symbols"]
        if cfg.surrogate_program == "orca":
            from .surrogate import orca as O
            if cfg.mode == "single" and cfg.surrogate_optimize:
                job = self._orca_job(wf, "opt_geometry", "opt")
                if job.submitted() and job.state() == O.DONE:
                    try:
                        H = self._hessian_at(job.hessian(), x)
                    except (OSError, ValueError):
                        H = None
                    if H is not None:
                        return H
            job = self._orca_job(wf, "freq_" + tag, "freq")
            job.write_frequencies(symbols, x, **self._orca_options())
            if not self._ready(wf, job, "frequencies of %s" % tag):
                return None
            H = self._hessian_at(job.hessian(), x)
            if H is None:
                raise RuntimeError("the Hessian file of %s belongs to another geometry than the one asked for" % job.folder)
            return H
        sur = self.surrogate(wf, np.eye(x.size), "cartesian")
        sur.request(tag, x, hessian=True, gradient=True)
        res = sur.result(tag)
        if res is None:
            return None
        if res.failed:
            raise RuntimeError("the Cartesian surrogate Hessian of %s failed: %s" % (tag, res.message))
        return np.asarray(res.hessian_directions, dtype=float)

    # ------------------------------------------------------------------ QMC
    def engine(self, wf):
        if self._engine is None:
            from .engine.qmcpack import QmcpackEngine
            self._engine = QmcpackEngine(self._dir(wf, "qmc"), self.machine, self.scheduler,
                                         settings=dict(self.cfg.qmc))
        return self._engine

    def qmc_settings(self, wf) -> Optional[dict]:
        """The QMC settings of the project. Mode ``fixed`` checks the given values; the other modes run pilot
        DMC jobs at the starting geometry of the first target and choose the values from what they measure."""
        p = wf.project
        from . import autotune
        pilot = self.pilot(wf)
        status = pilot.tick()
        if status == autotune.DONE:
            for w in pilot.warnings:
                p.log("QMC settings, warning: %s" % w)
            return pilot.settings()
        if status == autotune.NEEDS_ATTENTION:
            raise RuntimeError("the pilot runs could not choose the QMC settings: %s" % pilot.message)
        return None

    def pilot(self, wf):
        """The object that chooses the QMC settings (created once per process; its state is on disk)."""
        cfg = self.cfg
        p = wf.project
        if self._pilot is None:
            from .autotune import Pilot
            first = p.targets()[0]
            x = G.read_xyz(os.path.join(p.target_dir(first), "start.xyz"))[1]
            with np.load(os.path.join(p.target_dir(first), "directions.npz")) as z:
                n_dir = int(z["vectors"].shape[0])
            # the scan points are planned for the target that needs the smallest error bar: the one with the
            # most reaction coordinates (a saddle)
            n_neg = 0
            for name in p.targets():
                path = os.path.join(p.target_dir(name), "directions.npz")
                if os.path.exists(path):
                    with np.load(path) as z:
                        if "curvature" in z.files:
                            n_neg = max(n_neg, int(np.sum(np.asarray(z["curvature"], dtype=float) < 0.0)))
            fixed = cfg.qmc_mode == "fixed"
            options = dict(mode=cfg.qmc_mode, max_repeats=cfg.qmc_max_repeats, sigma_target=cfg.sigma_target,
                           ranks=self.machine.tasks("qmc"), charge=cfg.charge, n_directions=n_dir,
                           n_negative=min(n_neg, n_dir), e_cap=cfg.e_cap,
                           eps_total=cfg.eps_total, scan_rule="rule_b" if cfg.search_method == "dense-v1" else "joint")
            if fixed:
                options["final_blocks"] = cfg.cap_blocks
            self._pilot = Pilot(None if fixed else self.engine(wf), p.state["symbols"], x, dict(cfg.qmc),
                                self._dir(wf, "autotune", "pilot"), log=p.log, **options)
        return self._pilot
