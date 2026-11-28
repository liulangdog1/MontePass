"""The workflow: the stages of a project and their order.

``Workflow(project_dir).advance()`` is one pass. It does what can be done now
(writes inputs, submits jobs, reads finished results, decides), writes the state
and returns. Calling it again later continues. Nothing is kept in memory between
passes, so the process can be stopped at any time.

Stages, in order:

    path        geometries of the targets and, if asked for, the path between the end points
    subspace    the search subspace (principal components of the path, or soft Hessian modes)
    directions  search directions and their surrogate curvatures at every target
    autotune    the QMC settings
    search      the geometry search of every target
    trial       only with ``[trial] kind = multidet``: DMC energies with several determinants at the start
                geometry and at the final geometry of every target (module ``trial``)
    report      tables, plots and animations

Where the numbers come from is decided by a *backend*: ``ClusterBackend`` runs ORCA,
PySCF and QMCPACK through a batch system; ``ModelBackend`` uses analytic model molecules
and needs no other program (demonstrations and tests).
"""
from __future__ import annotations

import json
import os
import traceback
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import geometry as G
from . import subspace as S
from .config import Config, TrialSetupError
from .project import DONE, FAILED, NEEDS_ATTENTION, PENDING, RUNNING, TRIAL, Project, stage_names
from .search import protocol_v1 as P1
from .search import protocol_v2 as P2


# ============================================================================= backends
PCA_MAX_DIM = 6          # [subspace] max_dim = auto: principal components of a path
MODES_MAX_DIM = 12       # [subspace] max_dim = auto: internal coordinates of a single geometry


def chosen_path_targets(cfg: Config, energies, source: str):
    """Mode images with ``targets = barrier``, ``targets = mep`` or ``kinds = auto``: the images that are
    optimized and their kinds, ``({image number: kind}, one line for the log)``.

    ``energies``: energy of every image of the path, or None where it is not known. The saddle image is
    ``[geometry] saddle_image``, or else the image with the highest energy. ``source`` names the path in messages.
    """
    n = len(energies)
    saddle, why = cfg.saddle_image, "[geometry] saddle_image"
    if saddle is None:
        if any(e is None for e in energies):
            raise ValueError("%s gives no energy for every image, so the saddle image cannot be found: name it with "
                             "[geometry] saddle_image, or list the targets and their kinds" % source)
        try:
            saddle, why = S.highest_image(energies), "the image with the highest energy"
        except ValueError as exc:
            if cfg.selection == "barrier":
                raise ValueError("[geometry] targets = barrier: %s" % exc)
            saddle, why = None, str(exc)          # a path without a saddle: every image between the ends is a path image
    selection = cfg.selection if cfg.selection != "list" else [int(t) for t in cfg.targets]
    chosen = S.path_targets(selection, n, saddle)
    what = "targets = %s" % cfg.selection if cfg.selection != "list" else "kinds = auto"
    note = "%s: %d search(es) on the path of %d images: %s. Saddle image: %s" % (
        what, len(chosen), n, ", ".join("image %d as %s" % (i, "a path image" if k == "image" else "a " + k)
                                         for i, k in chosen.items()),
        "none (%s)" % why if saddle is None else "%d (%s)" % (saddle, why))
    return chosen, note


def with_path_images(targets: Dict[str, dict], frames: List[np.ndarray]) -> Dict[str, dict]:
    """Mode path with ``targets = mep``: the targets reactant, saddle and product plus every image between the
    ends as a path image (names ``image1``, ``image2``, ...), in the order of the path. The image nearest to the
    saddle geometry is left out: the saddle target stands for it."""
    saddle = targets.get("saddle")
    nearest = None
    if saddle is not None:
        nearest = int(np.argmin([G.kabsch(np.asarray(saddle["coords"], dtype=float).reshape(-1, 3),
                                          np.asarray(f, dtype=float).reshape(-1, 3))[1] for f in frames]))
    out: Dict[str, dict] = {}
    if "reactant" in targets:
        out["reactant"] = targets["reactant"]
    for i in range(1, len(frames) - 1):
        if i == nearest:
            out["saddle"] = saddle
        else:
            out["image%d" % i] = {"coords": frames[i], "kind": "image"}
    if saddle is not None and "saddle" not in out:
        out["saddle"] = saddle
    if "product" in targets:
        out["product"] = targets["product"]
    return out


class Backend:
    """Source of geometries, energies and surrogate data for a workflow."""

    def geometries(self, wf: "Workflow") -> Optional[dict]:
        """None while calculations run. Else a dict with ``symbols``, ``frames`` (list of path images or
        None), ``energies`` (surrogate energy of each image in hartree, or None), and ``targets``:
        ``{name: {"coords": array, "kind": "minimum" or "saddle"}}``."""
        raise NotImplementedError

    def engine(self, wf: "Workflow"):
        raise NotImplementedError

    def surrogate(self, wf: "Workflow", vectors: np.ndarray, label: str):
        """A surrogate object for gradients and Hessians in the basis ``vectors``."""
        raise NotImplementedError

    def cartesian_hessian(self, wf: "Workflow", tag: str, coords: np.ndarray) -> Optional[np.ndarray]:
        """Full Cartesian surrogate Hessian (kcal/mol/A^2), or None while it is being computed."""
        raise NotImplementedError

    def qmc_settings(self, wf: "Workflow") -> Optional[dict]:
        """The QMC settings of the project, or None while pilot runs are going."""
        return dict(wf.cfg.qmc)

    def moved_energy_source(self, wf: "Workflow"):
        """For the test with a moved surrogate: an object with ``request_scan`` / ``scan`` that gives surrogate
        energies of single geometries, or None if the engine computes them itself."""
        return None


class ModelBackend(Backend):
    """Analytic model molecules (``model.py``): the true surface stands for DMC, the surrogate for DFT.

    Input section ``[model]``:
        molecule     water or ammonia
        sigma_block  noise of one DMC block in kcal/mol (default 2.0)
        seed         seed of the noise (default 0)
        shift        if not 0: the surrogate is the true surface moved by this many angstrom along
                     ``shift_direction`` (a normal direction of the true minimum, counted from 0, softest first)
    """

    def __init__(self, cfg: Config):
        from . import model as M
        m = cfg.model or {}
        self.M = M
        self.molecule = m.get("molecule", "water").strip().lower()
        if self.molecule not in ("water", "ammonia"):
            raise ValueError("[model] molecule = %r; allowed: water, ammonia" % self.molecule)
        self.sigma_block = float(m.get("sigma_block", 2.0))
        self.seed = int(m.get("seed", 0))
        self.blocks_per_tick = int(m.get("blocks_per_tick", 20))
        shift = float(m.get("shift", 0.0))
        if self.molecule == "water":
            self.true, self.sur, guess = M.water(True), M.water(False), M.water_geometry()
        else:
            self.true, self.sur, guess = M.ammonia(True), M.ammonia(False), M.ammonia_geometry()
        self.guess = guess
        if shift != 0.0:
            xt = M.minimize(self.true, guess)
            Ht = M.hessian(self.true, xt)
            sub = S.mode_subspace(Ht, xt)
            d0 = S.search_directions(S.project_hessian(Ht, sub.basis), sub)
            delta = (shift * d0.vectors[int(m.get("shift_direction", 0))]).reshape(-1, 3)
            self.sur = self.true.shifted(delta)
            self.guess = xt + delta
        self._engine = None
        self._surrogates: Dict[str, object] = {}

    def geometries(self, wf):
        M = self.M
        cfg = wf.cfg
        symbols = list(self.true.symbols)
        if cfg.mode == "single":
            x = M.minimize(self.sur, self.guess)
            return {"symbols": symbols, "frames": None, "energies": None,
                    "targets": {"geometry": {"coords": x, "kind": "minimum"}}}
        if self.molecule != "ammonia":
            raise ValueError("the model path needs molecule = ammonia (its inversion); water has mode = single only")
        a = M.minimize(self.sur, M.ammonia_geometry(up=True))
        b = M.minimize(self.sur, M.ammonia_geometry(up=False))
        frames, e = M.nudged_elastic_band(self.sur, M.interpolate_path(a, b, max(5, cfg.neb_images + 1) | 1))
        top = int(np.argmax(e))
        saddle = M.refine_stationary(self.sur, frames[top])
        from .units import HARTREE_TO_KCAL
        all_targets = {"reactant": {"coords": frames[0], "kind": "minimum"},
                       "product": {"coords": frames[-1], "kind": "minimum"},
                       "saddle": {"coords": saddle, "kind": "saddle"}}
        if cfg.mode == "images":                # targets are image numbers of the model path
            if cfg.selection != "list" or cfg.kinds_auto:
                chosen, note = chosen_path_targets(cfg, [float(v) for v in e], "the model path")
                wf.project.log(note)
                picked = {str(i): {"coords": frames[i], "kind": kind} for i, kind in chosen.items()}
            else:
                for t in cfg.targets:
                    if int(t) >= len(frames):
                        raise ValueError("[geometry] targets: image %s does not exist; the model path has the images 0 to %d"
                                         % (t, len(frames) - 1))
                picked = {t: {"coords": frames[int(t)], "kind": cfg.kinds[t]} for t in cfg.targets}
            return {"symbols": symbols, "frames": frames, "energies": [float(v) / HARTREE_TO_KCAL for v in e],
                    "targets": picked}
        picked = {t: all_targets[t] for t in cfg.targets}
        if cfg.selection == "mep":
            picked = with_path_images(picked, frames)
            wf.project.log("targets = mep: %d searches (%s)" % (len(picked), ", ".join("%s: %s" % (n, t["kind"]) for n, t in picked.items())))
        return {"symbols": symbols, "frames": frames, "energies": [float(v) / HARTREE_TO_KCAL for v in e],
                "targets": picked}

    def engine(self, wf):
        if self._engine is None:
            from .engine.mock import MockEngine
            self._engine = MockEngine(self.true, self.sur, sigma_block=self.sigma_block, blocks_per_tick=self.blocks_per_tick,
                                      seed=self.seed, transient=20.0, equil_blocks=0.6,
                                      state_file=wf.project.path("mock_engine.json"))
        return self._engine

    def surrogate(self, wf, vectors, label):
        if label not in self._surrogates:
            from .surrogate.analytic import AnalyticSurrogate
            self._surrogates[label] = AnalyticSurrogate(self.sur)
        return self._surrogates[label]

    def cartesian_hessian(self, wf, tag, coords):
        return self.M.hessian(self.sur, coords)

    def true_energy(self, coords: np.ndarray) -> float:
        """Energy of the true model surface in kcal/mol (for demonstrations: the exact answer is known)."""
        return float(self.true(coords))


MODEL_WITHOUT_QP2 = ("[trial] kind = multidet cannot be used together with a section [model]: the model engine cannot "
                     "run Quantum Package 2. Remove the section [model], or set [trial] kind = single")


def make_backend(cfg: Config) -> Backend:
    if cfg.model is not None:
        if cfg.trial is not None:
            raise TrialSetupError("%s: %s" % (cfg.path, MODEL_WITHOUT_QP2))
        return ModelBackend(cfg)
    from .cluster import ClusterBackend          # imported late: needs the machine profile
    return ClusterBackend(cfg)


# ============================================================================= workflow
class Workflow:
    def __init__(self, project_dir: str, backend: Optional[Backend] = None):
        self.project = Project(project_dir)
        self.cfg = Config.load(self.project.input_file)
        self.project.state["name"] = self.cfg.name
        self.backend = backend if backend is not None else make_backend(self.cfg)
        self._drivers: Dict[str, object] = {}
        self._trial_runs = None
        self._sync_trial_stage()

    @property
    def stages(self) -> Tuple[str, ...]:
        """The stages of this project in their order. The stage ``trial`` exists only with ``[trial] kind = multidet``."""
        return stage_names(self.cfg.trial is not None)

    def _sync_trial_stage(self) -> None:
        """Make the project state fit the section ``[trial]`` of the input file.

        Without the option the state holds no stage ``trial`` (a project that never had the option is not
        touched). With the option the stage stands between ``search`` and ``report``. When the stage is new, or
        when it is DONE and the values of ``[trial]`` are no longer those of its record (a cutoff was added, for
        example, or the record was written by an older version that did not know the key ``tolerance``), the
        stage and the stage ``report`` are set to PENDING, so that the next passes compute what is missing and
        write the report again. Points that exist are never computed again. The record keeps the values of the
        input as they are: what the automatic mode of the stage adds while it runs (cutoffs, DMC series) is in
        another entry of the record, so it never looks like a changed input.

        When the option is switched off, the stage leaves the project state, and every point of its record that
        has not ended is stopped (the log names each one). If the stage never started (no record), the stage
        ``report`` and the project get back the status that they had before the option was switched on."""
        p = self.project
        if self.cfg.trial is None:
            entry = p.state["stages"].get(TRIAL)
            if entry is None:
                return
            from .trial import STATE_NAME, read_state, stop_points
            started = os.path.exists(os.path.join(self.trial_dir, STATE_NAME))
            try:
                record = read_state(self.trial_dir)
            except ValueError as exc:               # a damaged record: its points cannot be named
                record = None
                p.log("stage trial: its record could not be read (%s); no point of the stage was stopped" % exc)
            if record is not None:
                try:
                    stop_points(self.backend.engine(self), record, log=lambda line: p.log("[trial] " + line),
                                why="the option [trial] kind = multidet was switched off")
                except Exception as exc:            # the engine cannot be opened: say so, and go on
                    p.log("stage trial: the points of the stage could not be stopped (%s: %s)" % (type(exc).__name__, exc))
            restore = None if started else entry.get("restore")
            p.remove_stage(TRIAL)
            if restore:
                # give back only what the switch-on changed, and only while it still holds the value written then
                back = []
                if restore["report"]["status"] != PENDING and p.stage("report")["status"] == PENDING:
                    p.set_stage("report", restore["report"]["status"], restore["report"]["message"])
                    back.append("the stage report")
                if restore["status"] == DONE and p.state.get("status") == RUNNING:
                    p.state["status"] = DONE
                    back.append("the project")
                if back:
                    p.log("stage trial: it never started, so %s got the old status again" % " and ".join(back))
                p.save()
            return
        before = {"report": dict(p.stage("report")), "status": p.state.get("status")}
        if p.add_stage(TRIAL, before="report"):
            p.state["stages"][TRIAL]["restore"] = before          # read if the option is switched off before the stage starts
            p.save()
        else:
            if p.stage(TRIAL)["status"] != DONE:
                return                              # the stage has not ended: its next pass uses the values of the input
            from .trial import stored_trial
            try:
                if stored_trial(self.trial_dir) == self.cfg.trial:
                    return
            except ValueError:                      # a record that cannot be read: the next pass of the stage says so
                pass
            p.set_stage(TRIAL, PENDING, "the values of [trial] changed")
        if p.stage("report")["status"] != PENDING:
            p.set_stage("report", PENDING, "")
        if p.state.get("status") == DONE:           # the project has work to do again
            p.state["status"] = RUNNING
            p.save()

    @property
    def trial_dir(self) -> str:
        """Folder of the record of the stage ``trial``."""
        return os.path.join(self.project.work, "trial")

    # ------------------------------------------------------------------ public
    def advance(self) -> str:
        """One pass over the stages. Returns the status of the project: RUNNING, DONE or NEEDS_ATTENTION."""
        p = self.project
        status = DONE
        try:
            if (p.stage("search")["status"] not in (DONE, PENDING) or p.stage("autotune")["status"] == RUNNING
                    or (TRIAL in self.stages and p.stage(TRIAL)["status"] not in (DONE, PENDING))):
                self.backend.engine(self).advance()
            for stage in self.stages:
                if p.stage(stage)["status"] == DONE:
                    continue
                result = getattr(self, "_stage_" + stage)()
                if result != DONE:
                    status = result
                    break
        except Exception as exc:                      # a failure must be visible in the state, not only on a terminal
            status = NEEDS_ATTENTION
            p.state["error"] = traceback.format_exc()
            for stage in self.stages:
                if p.stage(stage)["status"] != DONE:
                    p.set_stage(stage, NEEDS_ATTENTION, "%s: %s" % (type(exc).__name__, exc))
                    break
        p.state["status"] = status
        p.save()
        return status

    def engine(self):
        return self.backend.engine(self)

    def retry(self) -> List[str]:
        """Give every failed calculation a new set of attempts and clear the ``NEEDS_ATTENTION`` states.

        For use after the cause was repaired (a module that was missing, a full disk, a time limit that was
        too short) or after a failure of the cluster. The driver must be started again afterwards (``submit``
        or ``run``). Returns one line per action."""
        import time
        p = self.project
        out: List[str] = []
        p.state["retry_request"] = time.time()      # failed batch jobs of the surrogate are submitted again when read
        searching = p.stage("directions")["status"] == DONE
        engine = None
        if searching or p.stage("subspace")["status"] == DONE:
            try:
                engine = self.backend.engine(self)
            except Exception as exc:                # no QMC settings yet, for example
                out.append("the QMC engine could not be opened (%s: %s)" % (type(exc).__name__, exc))
        if engine is not None:
            out += self._retry_points(engine)
        if hasattr(self.backend, "retry_surrogates"):
            for tag in self.backend.retry_surrogates(self):
                out.append("surrogate calculation %s: submitted again" % tag)
        if searching and p.stage("autotune")["status"] not in (DONE, PENDING) and hasattr(self.backend, "pilot"):
            pilot = self.backend.pilot(self)
            if pilot.status == NEEDS_ATTENTION:
                pilot.clear_attention()
                out.append("pilot runs: the NEEDS_ATTENTION status was cleared")
        if searching:
            for name in p.targets():
                t = p.target(name)
                if t.get("status") != NEEDS_ATTENTION:
                    continue
                state_dir = os.path.join(p.target_dir(name), "search")
                if os.path.isdir(state_dir):
                    drv = self.driver(name)
                    if hasattr(drv, "clear_attention"):
                        t["status"] = drv.clear_attention()
                        t["message"] = ""
                        out.append("search of target %s: the NEEDS_ATTENTION status was cleared" % name)
                else:
                    t["status"], t["message"] = PENDING, ""
                    out.append("target %s: the NEEDS_ATTENTION status was cleared" % name)
        if (TRIAL in self.stages and p.stage("search")["status"] == DONE and not self.trial_held_back()
                and p.stage(TRIAL)["status"] in (RUNNING, NEEDS_ATTENTION)):
            # the failed points of the plan of the stage trial (the points of an older plan are not submitted again)
            try:
                for key in self.trial_runs().retry():
                    out.append("QMC point %s: submitted again" % key)
            except Exception as exc:                # the record of the stage does not fit the input, for example
                out.append("the stage trial could not be opened (%s: %s)" % (type(exc).__name__, exc))
        for stage in self.stages:
            if p.stage(stage)["status"] == NEEDS_ATTENTION:
                p.set_stage(stage, RUNNING, "continued by request")
                out.append("stage %s: the NEEDS_ATTENTION status was cleared" % stage)
        p.state.pop("error", None)
        if out:
            p.state["status"] = RUNNING
            p.log("retry: " + "; ".join(out))
        p.save()
        return out

    def _retry_points(self, engine) -> List[str]:
        """Give the failed QMC points of the search and of the pilot runs new attempts. The points of the stage
        ``trial`` are left to :meth:`TrialRuns.retry` (only the points of its current plan). Returns one line per
        point."""
        from .trial import is_stage_key
        out: List[str] = []
        failed = getattr(engine, "failed_keys", None)
        retry = getattr(engine, "retry", None)
        if callable(failed) and callable(retry):
            for key in failed():
                if is_stage_key(key):
                    continue
                if retry(key):
                    out.append("QMC point %s: submitted again" % key)
        elif hasattr(engine, "retry_failed"):     # an engine that can only retry all its failed points together
            for key in engine.retry_failed():
                out.append("QMC point %s: submitted again" % key)
        return out

    # ------------------------------------------------------------------ stage: path
    def _stage_path(self) -> str:
        p, cfg = self.project, self.cfg
        g = self.backend.geometries(self)
        if g is None:
            p.set_stage("path", RUNNING, "surrogate calculations are running")
            return RUNNING
        symbols = [G.normalize_symbol(s) for s in g["symbols"]]
        p.state["symbols"] = symbols
        frames = g.get("frames")
        if frames is not None:
            frames = S.align_path([np.asarray(f, dtype=float).reshape(-1, 3) for f in frames], 0)
            energies = g.get("energies")
            comments = ["image %d" % i + (" E %.10f" % energies[i] if energies is not None else "") for i in range(len(frames))]
            G.write_xyz_frames(p.path_file, symbols, frames, comments)
        protocol = {"dense-v1": "v1", "dense-v2": "v2", "classic": "classic", "auto": "auto"}[cfg.search_method]
        for name, t in g["targets"].items():
            x = np.asarray(t["coords"], dtype=float).reshape(-1, 3)
            if len(x) != len(symbols):
                raise ValueError("target %s has %d atoms, the path has %d" % (name, len(x), len(symbols)))
            nearest = None
            if frames is not None:
                # the subspace vectors are defined in the frame of the aligned path: put the target there,
                # on top of its nearest image
                fits = [G.kabsch(x, f) for f in frames]
                nearest = int(np.argmin([r[1] for r in fits]))
                x = fits[nearest][0]
            if t["kind"] == "image":
                # a path image is searched across the path only: it needs the image before and the image after
                if frames is None:
                    raise ValueError("target %s: the kind image needs a path, and this project has none" % name)
                if not 0 < nearest < len(frames) - 1:
                    raise ValueError("target %s: the kind image needs an image with two neighbours on the path; "
                                     "this target is image %d of the images 0 to %d (an end of the path is a "
                                     "minimum)" % (name, nearest, len(frames) - 1))
            G.write_xyz(os.path.join(p.target_dir(name), "start.xyz"), symbols, x, "start geometry of target %s" % name)
            p.add_target(name, t["kind"], protocol)
            if t["kind"] == "image":
                p.target(name)["image"] = nearest
        p.state["n_images"] = 0 if frames is None else len(frames)
        p.set_stage("path", DONE, "%d target(s)%s" % (len(g["targets"]), "" if frames is None else ", path of %d images" % len(frames)))
        return DONE

    def _start(self, target: str) -> np.ndarray:
        return G.read_xyz(os.path.join(self.project.target_dir(target), "start.xyz"))[1]

    def _frames(self) -> Optional[List[np.ndarray]]:
        if not os.path.exists(self.project.path_file):
            return None
        return [f[1] for f in G.read_xyz_frames(self.project.path_file)]

    # ------------------------------------------------------------------ stage: subspace
    def _stage_subspace(self) -> str:
        p, cfg = self.project, self.cfg
        frames = self._frames()
        kind = cfg.subspace_kind
        if kind == "auto":
            kind = "pca" if (frames is not None and len(frames) >= 3) else "modes"
        if kind == "pca":
            if frames is None:
                raise ValueError("[subspace] kind = pca needs a path, and this project has none")
            sub = S.pca_subspace(frames, d=cfg.subspace_dim, variance=cfg.subspace_variance,
                                 max_dim=cfg.subspace_max_dim if cfg.subspace_max_dim is not None else PCA_MAX_DIM)
            note = "PCA of %d images: %d component(s), explained variance %s" % (
                len(frames), sub.dim, ", ".join("%.4f" % v for v in sub.explained_variance_ratio[:sub.dim]))
        else:
            first = p.targets()[0]
            x = self._start(first)
            H = self.backend.cartesian_hessian(self, "modes_" + first, x)
            if H is None:
                p.set_stage("subspace", RUNNING, "the surrogate Hessian of target %s is being computed" % first)
                return RUNNING
            n_int = G.n_internal(x)
            limit = cfg.subspace_max_dim if cfg.subspace_max_dim is not None else MODES_MAX_DIM
            k = cfg.subspace_dim if cfg.subspace_dim is not None else min(n_int, limit)
            if k > n_int:
                raise ValueError("[subspace] dim = %d, but the molecule has only %d internal coordinates" % (k, n_int))
            sub = S.mode_subspace(H, x, k)
            note = "the %d softest of %d internal Hessian modes of target %s" % (k, n_int, first)
            if k < n_int:
                p.log("WARNING: the search covers %d of the %d internal coordinates of the molecule (the softest "
                      "modes of the surrogate Hessian). The other %d keep their surrogate values. Set [subspace] dim "
                      "or max_dim to search more." % (k, n_int, n_int - k))
        sub.save(p.subspace_file)
        p.state["subspace"] = {"kind": sub.kind, "dim": sub.dim}
        # how much of the displacement between the targets can the subspace represent?
        names = p.targets()
        if len(names) > 1:
            x0 = self._start(names[0])
            p.state["subspace"]["captured"] = {n: sub.captured_fraction(self._start(n) - x0) for n in names[1:]}
        p.set_stage("subspace", DONE, note)
        return DONE

    # ------------------------------------------------------------------ stage: directions
    def _stage_directions(self) -> str:
        p = self.project
        sub = S.Subspace.load(p.subspace_file)
        sur = self.backend.surrogate(self, sub.basis, "subspace")
        waiting = []
        for name in p.targets():
            f = os.path.join(p.target_dir(name), "directions.npz")
            if os.path.exists(f):
                continue
            tag = "directions_" + name
            x = self._start(name)
            sur.request(tag, x, hessian=True, gradient=True)
            res = sur.result(tag)
            if res is None:
                waiting.append(name); continue
            if res.failed:
                raise RuntimeError("surrogate Hessian of target %s failed: %s" % (name, res.message))
            K = res.hessian_directions if res.hessian_directions is not None else S.project_hessian(res.hessian, sub.basis)
            kind = p.target(name)["kind"]
            tangent = None
            if kind == "image":
                # A path image is no stationary point: along the path the energy has a slope. The search is made
                # across the path only. The path direction (from the two neighbour images) is removed from the
                # subspace, and the image is then treated as a minimum in the directions that are left.
                image = int(p.target(name)["image"])
                tangent = S.path_tangent(self._frames(), image, at=x)
                inside = sub.captured_fraction(tangent)
                if inside <= 1e-12:
                    p.log("target %s: the path direction at image %d has no part in the search subspace; all %d "
                          "directions of the subspace are searched" % (name, image, sub.dim))
                    d = S.search_directions(np.asarray(K, dtype=float), sub)
                else:
                    try:
                        d = S.search_directions(np.asarray(K, dtype=float), sub, exclude=tangent)
                    except ValueError as exc:
                        p.target(name).update(status=NEEDS_ATTENTION, message="path image %d: %s" % (image, exc))
                        p.set_stage("directions", NEEDS_ATTENTION, "target %s: %s" % (name, p.target(name)["message"]))
                        return NEEDS_ATTENTION
                    p.log("target %s: path image %d. The path direction (difference of the images %d and %d; %.1f %% of "
                          "its squared length lies in the subspace) is removed: %d of the %d directions of the "
                          "subspace are searched, all of them across the path" % (
                              name, image, image - 1, image + 1, 100.0 * inside, d.dim, sub.dim))
            else:
                d = S.search_directions(np.asarray(K, dtype=float), sub)
            expected = 1 if kind == "saddle" else 0
            if d.n_negative != expected:
                p.target(name).update(status=NEEDS_ATTENTION,
                                      message="the surrogate Hessian in the subspace has %d negative curvature(s); a %s needs %d"
                                      % (d.n_negative, "path image (across the path)" if kind == "image" else kind, expected))
                p.set_stage("directions", NEEDS_ATTENTION, "target %s: %s" % (name, p.target(name)["message"]))
                return NEEDS_ATTENTION
            g_sub = None if res.gradient is None else sub.basis @ np.asarray(res.gradient).reshape(-1)
            shift = np.zeros_like(x)
            if self.cfg.shift != 0.0:
                if self.cfg.shift_direction >= d.dim:
                    raise ValueError("[experiment] direction = %d, but target %s has only %d search directions"
                                     % (self.cfg.shift_direction, name, d.dim))
                shift = (self.cfg.shift * d.vectors[self.cfg.shift_direction]).reshape(-1, 3)
                symbols = p.state["symbols"]
                G.write_xyz(os.path.join(p.target_dir(name), "start_unshifted.xyz"), symbols, x, "surrogate geometry of target %s" % name)
                G.write_xyz(os.path.join(p.target_dir(name), "start.xyz"), symbols, x + shift,
                            "start of target %s, moved by %.4f A along search direction %d" % (name, self.cfg.shift, self.cfg.shift_direction))
                p.log("target %s: the start is moved by %.4f A along direction %d (test of a wrong starting geometry)%s" % (
                    name, self.cfg.shift, self.cfg.shift_direction, "; the surrogate is moved with it" if self.cfg.move_surrogate else ""))
            elif self.cfg.random_rms > 0.0:
                # a random displacement of all atoms. The part of it that lies outside the search directions
                # cannot be removed by the search: the log says how large that part is.
                shift = G.random_displacement(x, self.cfg.random_rms, self.cfg.random_seed)
                if self.cfg.random_space == "subspace":
                    # only the part inside the search directions, scaled back to the wanted size
                    inside = (d.vectors.T @ (d.vectors @ shift.reshape(-1))).reshape(-1, 3)
                    shift = inside * (self.cfg.random_rms / float(np.sqrt(np.mean(np.sum(inside ** 2, axis=1)))))
                inside = d.vectors @ shift.reshape(-1)
                fraction = float(inside @ inside) / float(shift.reshape(-1) @ shift.reshape(-1))
                symbols = p.state["symbols"]
                G.write_xyz(os.path.join(p.target_dir(name), "start_unshifted.xyz"), symbols, x, "surrogate geometry of target %s" % name)
                G.write_xyz(os.path.join(p.target_dir(name), "start.xyz"), symbols, x + shift,
                            "start of target %s, every atom moved at random (rms %.4f A per atom, seed %d)" % (
                                name, self.cfg.random_rms, self.cfg.random_seed))
                p.target(name)["experiment"] = {"random_rms": self.cfg.random_rms, "random_seed": self.cfg.random_seed,
                                                "displacement_norm": float(np.linalg.norm(shift)),
                                                "fraction_in_search_directions": fraction}
                p.log("target %s: every atom is moved at random, rms %.4f A per atom, seed %d (test of a wrong starting "
                      "geometry)%s. %.1f %% of the squared displacement lies in the %d search directions" % (
                          name, self.cfg.random_rms, self.cfg.random_seed,
                          "; the surrogate is moved with it" if self.cfg.move_surrogate else "", 100.0 * fraction, d.dim))
            extra = {} if tangent is None else {"tangent": tangent}          # path image: the direction that is not searched
            np.savez(f, vectors=d.vectors, curvature=d.curvature, hessian=np.asarray(K), coefficients=d.coefficients,
                     gradient=np.array([]) if g_sub is None else d.coefficients @ g_sub, shift=shift, **extra)
            p.log("target %s: %d search directions, surrogate curvatures %s kcal/mol/A^2" % (
                name, d.dim, np.round(d.curvature, 1).tolist()))
        if waiting:
            p.set_stage("directions", RUNNING, "surrogate Hessian running for: %s" % ", ".join(waiting))
            return RUNNING
        p.set_stage("directions", DONE)
        return DONE

    # ------------------------------------------------------------------ stage: autotune
    def _stage_autotune(self) -> str:
        p = self.project
        f = p.path("autotune", "qmc_settings.json")
        if not os.path.exists(f):
            settings = self.backend.qmc_settings(self)
            if settings is None:
                p.set_stage("autotune", RUNNING, "pilot runs are going")
                return RUNNING
            tmp = f + ".tmp"
            with open(tmp, "w") as fh:
                json.dump(settings, fh, indent=1)
            os.replace(tmp, f)
        p.set_stage("autotune", DONE)
        return DONE

    def qmc_settings(self) -> dict:
        """The QMC settings as the autotune stage wrote them (with its record under ``"autotune"``)."""
        with open(self.project.path("autotune", "qmc_settings.json")) as fh:
            return json.load(fh)

    def point_settings(self) -> dict:
        """The settings that the search drivers pass on with every point: the QMC settings without the record
        of the pilot runs, plus the noise of one DMC block if the pilot runs measured it."""
        s = self.qmc_settings()
        record = s.pop("autotune", None) or {}
        sigma = (record.get("analysis") or {}).get("sigma_block_kcal")
        if sigma and "sigma_block" not in s:
            s["sigma_block"] = float(sigma)
        return s

    def cap_blocks(self) -> int:
        """Largest number of DMC blocks of one point, which is also the length of a final run. With
        ``[qmc] mode = fixed`` it is ``[search] cap_blocks``. In the other modes it is the number of blocks
        that the pilot runs planned for the wanted error bar of a final energy."""
        if self.cfg.qmc_mode != "fixed":
            record = self.qmc_settings().get("autotune") or {}
            planned = ((record.get("choices") or {}).get("final_blocks") or {}).get("value")
            if planned:
                return int(planned)
        return int(self.cfg.cap_blocks)

    # ------------------------------------------------------------------ stage: search
    def _protocol(self, name: str):
        cfg = self.cfg
        method = self.project.target(name)["protocol"]
        cap = self.cap_blocks()
        if method == "v1":
            return P1.ProtocolV1(e_cap=cfg.e_cap, cap=cap, eps_total=cfg.eps_total, max_iter=cfg.max_iter)
        # with fixed QMC settings ``cap_blocks`` is the limit that the user set: a repeated scan never asks for more
        factor = 1.0 if cfg.qmc_mode == "fixed" else P2.V2.cap_scan_factor
        return P2.ProtocolV2(e_cap=cfg.e_cap, cap=cap, eps_total=cfg.eps_total, max_iter=cfg.max_iter,
                             cap_scan_factor=factor)

    def driver(self, name: str):
        """The search driver of a target (created once per process; its state is on disk)."""
        if name in self._drivers:
            return self._drivers[name]
        p = self.project
        symbols = p.state["symbols"]
        with np.load(os.path.join(p.target_dir(name), "directions.npz")) as z:
            vectors, curvature = z["vectors"].copy(), z["curvature"].copy()
            shift = z["shift"].copy() if "shift" in z.files else None
        state_dir = os.path.join(p.target_dir(name), "search")
        method = p.target(name)["protocol"]
        engine = self.backend.engine(self)
        surrogate = self.backend.surrogate(self, vectors, "search_" + name)
        if shift is not None and np.abs(shift).max() > 0 and self.cfg.move_surrogate:
            from .shifted import ShiftedEngine, ShiftedSurrogate
            surrogate = ShiftedSurrogate(surrogate, shift)
            if method in ("v2", "auto"):          # only protocol v2 reads the surrogate energy of a DMC point
                engine = ShiftedEngine(engine, shift, source=self.backend.moved_energy_source(self))
        settings = self.point_settings()
        if method == "classic":
            from .search.classic import ClassicLineSearch
            drv = ClassicLineSearch(name, engine, surrogate, symbols, self._start(name), vectors, curvature,
                                    settings=settings, state_dir=state_dir, log=p.log)
        elif method == "auto":
            from .search.auto import AutoSearch
            drv = AutoSearch(name, engine, surrogate, symbols, self._start(name), vectors, curvature,
                             protocol=self._protocol(name), state_dir=state_dir, settings=settings, log=p.log)
        else:
            from .search.dense import DenseScanSearch
            drv = DenseScanSearch(name, engine, surrogate, symbols, self._start(name), vectors, curvature,
                                  protocol=self._protocol(name), state_dir=state_dir, settings=settings, log=p.log)
        self._drivers[name] = drv
        return drv

    def _stage_search(self) -> str:
        p = self.project
        statuses = {}
        for name in p.targets():
            t = p.target(name)
            if t["status"] in ("CONVERGED", "NOT_CONVERGED", "DONE"):
                statuses[name] = t["status"]; continue
            drv = self.driver(name)
            st = drv.tick()
            t["status"] = st; t["message"] = drv.state.get("message", "")
            statuses[name] = st
        p.save()
        if any(s == NEEDS_ATTENTION for s in statuses.values()):
            bad = [n for n, s in statuses.items() if s == NEEDS_ATTENTION]
            p.set_stage("search", NEEDS_ATTENTION, "; ".join("%s: %s" % (n, p.target(n)["message"]) for n in bad))
            return NEEDS_ATTENTION
        if all(s in ("CONVERGED", "NOT_CONVERGED", "DONE") for s in statuses.values()):
            p.set_stage("search", DONE, ", ".join("%s %s" % kv for kv in statuses.items()))
            return DONE
        p.set_stage("search", RUNNING, ", ".join("%s %s" % kv for kv in statuses.items()))
        return RUNNING

    # ------------------------------------------------------------------ stage: trial
    def trial_settings(self) -> dict:
        """The settings of every point of the stage ``trial``: those of the search points, with one change.

        Every point optimizes its own Jastrow factor from zero: the Jastrow factor of a search point belongs to
        another trial wave function. The check of slow runs (``slow_factor``) stays as it is: the engine compares
        a run only with runs of the same kind of wave function (one determinant, or the same cut), so the longer
        steps of a multi-determinant run do not count as slow."""
        s = self.point_settings()
        s.pop("jastrow_start_key", None)
        s["jastrow_reuse"] = "none"
        return s

    def trial_runs(self):
        """The object that runs the stage ``trial`` (created once per process; its record is on disk). The two
        geometries of every target come from its search: the first centre and the final geometry. The kinds of
        the targets come from the project state: the automatic mode of the stage tests the barrier of every
        pair of a saddle target and a minimum target."""
        if self._trial_runs is None:
            from .trial import TrialRuns
            p = self.project
            geometries = {}
            for name in p.targets():
                drv = self.driver(name)
                geometries[name] = (drv.trajectory()[0], drv.final_geometry())
            kinds = {name: str(p.target(name).get("kind", "")) for name in p.targets()}
            self._trial_runs = TrialRuns(self.backend.engine(self), p.state["symbols"], geometries, self.cfg.trial,
                                         self.trial_settings(), self.cap_blocks(), self.trial_dir, log=p.log, kinds=kinds)
        return self._trial_runs

    def not_converged_targets(self) -> List[str]:
        """Targets whose geometry search ended without reaching its precision target (status NOT_CONVERGED)."""
        p = self.project
        return [name for name in p.targets() if p.target(name).get("status") == "NOT_CONVERGED"]

    def trial_held_back(self) -> List[str]:
        """Targets that keep the stage ``trial`` from starting: those whose search ended NOT_CONVERGED, unless the
        input says ``[trial] not_converged = run``. An empty list means the stage may start."""
        if self.cfg.trial is None or self.cfg.trial.get("not_converged", "stop") == "run":
            return []
        return self.not_converged_targets()

    def _stage_trial(self) -> str:
        p = self.project
        bad = self.trial_held_back()
        if bad:
            # multi-determinant energies are meant for a geometry that the search has fixed
            p.set_stage(TRIAL, NEEDS_ATTENTION, "the search of target(s) %s ended NOT_CONVERGED, so the stage does not "
                        "start. Set [trial] not_converged = run to compute it anyway; the tables then mark every value "
                        "that uses the final geometry of such a target" % ", ".join(bad))
            return NEEDS_ATTENTION
        runs = self.trial_runs()
        if p.stage(TRIAL).pop("restore", None) is not None:      # the stage has a record now: nothing to give back
            p.save()
        status = runs.tick()
        p.set_stage(TRIAL, status, runs.message)
        return status

    # ------------------------------------------------------------------ stage: report
    def _stage_report(self) -> str:
        p = self.project
        try:
            from .report import build_report
        except ImportError as exc:
            p.set_stage("report", DONE, "no report module: %s" % exc)
            return DONE
        # the summary shows the table of the stages: mark this stage first, so that the table is the final one
        p.set_stage("report", DONE, "written to results/")
        try:
            files = build_report(p.folder, self.backend.engine(self), title=self.cfg.name, gifs=self.cfg.gif)
        except Exception:
            p.set_stage("report", PENDING, "")
            raise
        p.set_stage("report", DONE, "%d file(s) in results/" % len(files))
        return DONE
