"""A complete small project on a model molecule, built without any cluster program.

    from montepass.report import build_report
    from montepass.report.demo import make_demo_project
    demo = make_demo_project("demo_ammonia")
    files = build_report(demo.folder, demo.engine, title=demo.title)

The molecule is the analytic ammonia model of ``montepass.model``. Its "true" surface plays the role of
the DMC energy (the mock engine adds statistical noise to it), its "surrogate" surface the role of DFT.
The steps are those of a real project:

1. the two pyramidal minima of the surrogate and the inversion path between them (a small nudged elastic band);
2. the principal components of the path images as the search subspace;
3. search directions and surrogate curvatures at two targets: the minimum (``reactant``) and the planar
   saddle point (``saddle``);
4. a dense-scan search (protocol v2) of each target on the mock engine.

The function writes the folder layout of a project (``input.ini``, ``work/project.json``, ``work/path.xyz``,
``work/subspace.npz``, ``work/targets/<target>/...``, ``work/autotune/qmc_settings.json``). The whole run takes
about 2 seconds on one core.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np

from .. import geometry as G
from .. import model as M
from .. import subspace as S
from ..engine.mock import MockEngine
from ..search import protocol_v2 as P2
from ..search.dense import RUNNING_STATUS, DenseScanSearch
from ..search.dense_v2 import FINALIZING
from ..surrogate.analytic import AnalyticSurrogate
from ..units import HARTREE_TO_KCAL

INPUT_TEXT = """\
; Demonstration project on a model molecule. It needs no cluster program.
; It was built by montepass.report.demo.make_demo_project with the values below.
; The searches used protocol dense-v2 with g_move = {g_move} kcal/mol.

[system]
name = {name}

[geometry]
mode = path
targets = reactant, saddle

[surrogate]
neb_images = {neb_images}

[model]
molecule = ammonia
sigma_block = {sigma_block}
seed = {seed}
blocks_per_tick = {blocks_per_tick}

[search]
method = dense-v2
eps_total = {eps_total}
cap_blocks = {cap_blocks}
"""


@dataclass
class DemoProject:
    """What :func:`make_demo_project` built."""

    folder: str                                 # the project folder
    title: str
    engine: MockEngine                          # holds the DMC blocks of every point; give it to build_report
    symbols: List[str]
    true_surface: Callable[[np.ndarray], float]         # kcal/mol; the exact energy behind the mock DMC
    surrogate_surface: Callable[[np.ndarray], float]    # kcal/mol
    status: Dict[str, str]                      # final status of the search of every target


def _write_json(path: str, data: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(data, fh, indent=1)
    os.replace(tmp, path)


def make_demo_project(folder: str, sigma_block: float = 0.5, seed: int = 1, cap_blocks: int = 120,
                      eps_total: float = 0.05, g_move: float = 0.1, n_images: int = 9, blocks_per_tick: int = 20,
                      name: str = "model_ammonia_inversion", max_passes: int = 5000) -> DemoProject:
    """Build the demonstration project in ``folder`` (created if needed) and run its two searches to the end.

    ``sigma_block`` is the noise of one mock DMC block in kcal/mol, ``cap_blocks`` the largest number of blocks
    per point, ``eps_total`` the tolerance of the searches in kcal/mol, ``g_move`` the predicted gain (kcal/mol)
    above which a search scans again at a new centre, ``n_images`` the number of path images (odd, at least 5).

    If the folder already holds this project, the searches continue from their state files; nothing is run twice.
    Raises RuntimeError if a search does not end within ``max_passes`` passes.
    """
    if n_images < 5 or n_images % 2 == 0:
        raise ValueError("n_images must be an odd number of at least 5, got %d" % n_images)
    folder = os.path.abspath(folder)
    work = os.path.join(folder, "work")
    os.makedirs(os.path.join(work, "autotune"), exist_ok=True)
    true, sur = M.ammonia(True), M.ammonia(False)
    symbols = list(true.symbols)

    # 1. path on the surrogate
    a = M.minimize(sur, M.ammonia_geometry(up=True))
    b = M.minimize(sur, M.ammonia_geometry(up=False))
    frames, e_path = M.nudged_elastic_band(sur, M.interpolate_path(a, b, n_images))
    frames = S.align_path(frames, 0)
    top = int(np.argmax(e_path))
    saddle = G.kabsch(M.refine_stationary(sur, frames[top]), frames[top])[0]
    G.write_xyz_frames(os.path.join(work, "path.xyz"), symbols, frames,
                       ["image %d E %.10f" % (i, e / HARTREE_TO_KCAL) for i, e in enumerate(e_path)])

    # 2. subspace
    sub = S.pca_subspace(frames, variance=0.999, max_dim=6)
    sub.save(os.path.join(work, "subspace.npz"))

    # 3. targets: start geometry and search directions
    targets = {"reactant": {"coords": frames[0], "kind": "minimum"}, "saddle": {"coords": saddle, "kind": "saddle"}}
    directions = {}
    for target, info in targets.items():
        tdir = os.path.join(work, "targets", target)
        os.makedirs(tdir, exist_ok=True)
        G.write_xyz(os.path.join(tdir, "start.xyz"), symbols, info["coords"], "start geometry of target %s" % target)
        K = S.project_hessian(M.hessian(sur, info["coords"]), sub.basis)
        d = S.search_directions(K, sub)
        expected = 1 if info["kind"] == "saddle" else 0
        if d.n_negative != expected:
            raise RuntimeError("target %s: the surrogate Hessian in the subspace has %d negative curvature(s), "
                               "a %s needs %d" % (target, d.n_negative, info["kind"], expected))
        np.savez(os.path.join(tdir, "directions.npz"), vectors=d.vectors, curvature=d.curvature, hessian=K,
                 coefficients=d.coefficients)
        directions[target] = d

    # 4. the searches, on one engine and one clock
    settings = {"engine": "mock", "sigma_block_kcal_per_mol": sigma_block, "blocks_per_tick": blocks_per_tick, "seed": seed}
    _write_json(os.path.join(work, "autotune", "qmc_settings.json"), settings)
    engine = MockEngine(true, sur, sigma_block=sigma_block, blocks_per_tick=blocks_per_tick, seed=seed,
                        transient=20.0, equil_blocks=0.6, state_file=os.path.join(work, "mock_engine.json"))
    protocol = P2.ProtocolV2(cap=cap_blocks, eps_total=eps_total, g_move=g_move)
    drivers = {}
    for target, info in targets.items():
        d = directions[target]
        drivers[target] = DenseScanSearch(target, engine, AnalyticSurrogate(sur), symbols, info["coords"], d.vectors,
                                          d.curvature, protocol=protocol,
                                          state_dir=os.path.join(work, "targets", target, "search"))
    for _ in range(max_passes):
        if all(drv.status not in (RUNNING_STATUS, FINALIZING) for drv in drivers.values()):
            break
        engine.advance()
        for drv in drivers.values():
            drv.tick()
    else:
        raise RuntimeError("the demonstration searches did not end within %d passes: %s"
                           % (max_passes, {t: drv.status for t, drv in drivers.items()}))

    # 5. input file and project state
    with open(os.path.join(folder, "input.ini"), "w") as fh:
        fh.write(INPUT_TEXT.format(name=name, neb_images=n_images - 1, sigma_block=sigma_block, seed=seed,
                                   blocks_per_tick=blocks_per_tick, eps_total=eps_total, cap_blocks=cap_blocks,
                                   g_move=g_move))
    status = {t: drv.status for t, drv in drivers.items()}
    x0 = targets["reactant"]["coords"]
    project = {
        "name": name, "status": "RUNNING", "symbols": symbols, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_images": len(frames),
        "stages": {
            "path": {"status": "DONE", "message": "%d target(s), path of %d images" % (len(targets), len(frames))},
            "subspace": {"status": "DONE", "message": "PCA of %d images: %d component(s), explained variance %s" % (
                len(frames), sub.dim, ", ".join("%.4f" % v for v in sub.explained_variance_ratio[:sub.dim]))},
            "directions": {"status": "DONE", "message": ""},
            "autotune": {"status": "DONE", "message": ""},
            "search": {"status": "DONE", "message": ", ".join("%s %s" % kv for kv in status.items())},
            "report": {"status": "PENDING", "message": ""},
        },
        "targets": {t: {"kind": targets[t]["kind"], "protocol": "v2", "status": status[t],
                        "message": drivers[t].state.get("message", "")} for t in targets},
        "subspace": {"kind": sub.kind, "dim": sub.dim,
                     "captured": {"saddle": sub.captured_fraction(targets["saddle"]["coords"] - x0)}},
    }
    _write_json(os.path.join(work, "project.json"), project)
    return DemoProject(folder=folder, title=name, engine=engine, symbols=symbols, true_surface=true,
                       surrogate_surface=sur, status=status)
