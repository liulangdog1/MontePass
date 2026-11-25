"""Mock engine: an analytic energy surface plus statistical noise.

It behaves like a batch system with DMC jobs. A submitted point waits in a
queue for a few ticks, then writes a fixed number of blocks per tick until it
reaches the requested number of blocks. Every call of :meth:`MockEngine.advance`
is one tick. The block energies are

    E_block(i) = E_true(x) + transient * exp(-i / equil_blocks) + noise(i)

where the noise is a Gaussian AR(1) series with standard deviation
``sigma_block`` and lag-1 correlation ``rho``. The transient models the
relaxation of a DMC run from the variational energy. The trace of a point
depends only on the seed, the key and the series number, so a run is
reproducible and does not depend on how often the data are read.

The state (tick counter and submitted points) can be saved to a JSON file, so
that a driver that is started again continues the same simulated campaign.
"""
from __future__ import annotations

import json
import os
import zlib
from typing import Callable, Dict, Optional

import numpy as np

from ..units import HARTREE_TO_KCAL
from .base import ABSENT, DONE, QUEUED, RUNNING, STOPPED, Engine, PointData, PointSpec

Surface = Callable[[np.ndarray], float]      # energy in kcal/mol of a geometry in angstrom


class MockEngine(Engine):
    def __init__(self, true_surface: Surface, surrogate_surface: Optional[Surface] = None,
                 sigma_block: float = 1.0, rho: float = 0.0, transient: float = 0.0, equil_blocks: float = 10.0,
                 blocks_per_tick: int = 50, queue_ticks: int = 0, seed: int = 0,
                 core_hours_per_block: float = 1.0, state_file: Optional[str] = None,
                 e_offset_hartree: float = 0.0):
        """``sigma_block`` and ``transient`` are in kcal/mol. ``core_hours_per_block`` is the pretended
        cost of one block, used to compare the cost of search protocols."""
        self.true_surface = true_surface
        self.surrogate_surface = surrogate_surface
        self.sigma_block = float(sigma_block)
        self.rho = float(rho)
        self.transient = float(transient)
        self.equil_blocks = float(equil_blocks)
        self.blocks_per_tick = int(blocks_per_tick)
        self.queue_ticks = int(queue_ticks)
        self.seed = int(seed)
        self.core_hours_per_block = float(core_hours_per_block)
        self.e_offset = float(e_offset_hartree)
        self.state_file = state_file
        self.tick = 0
        self.points: Dict[str, dict] = {}
        self._trace_cache: Dict[str, np.ndarray] = {}
        if state_file and os.path.exists(state_file):
            self._load()

    # ------------------------------------------------------------------ persistence
    def _load(self) -> None:
        with open(self.state_file) as fh:
            st = json.load(fh)
        self.tick = int(st["tick"])
        self.points = st["points"]

    def save(self) -> None:
        if not self.state_file:
            return
        tmp = self.state_file + ".tmp"
        with open(tmp, "w") as fh:
            json.dump({"tick": self.tick, "points": self.points}, fh)
        os.replace(tmp, self.state_file)

    # ------------------------------------------------------------------ engine interface
    def submit(self, spec: PointSpec) -> None:
        if spec.key in self.points:
            return
        coords = np.asarray(spec.coords, dtype=float).reshape(-1, 3)
        self.points[spec.key] = {
            "coords": coords.tolist(), "blocks": int(spec.blocks), "series": int(spec.series),
            "submitted": self.tick, "stopped_at": None, "role": spec.role,
            # geometry at which the surrogate energy is taken (differs from the point only in tests of a moved surrogate)
            "sur_coords": spec.settings.get("surrogate_coords"),
        }
        self.save()

    def _available(self, p: dict) -> int:
        start = p["submitted"] + self.queue_ticks
        n = max(0, self.tick - start) * self.blocks_per_tick
        n = min(n, p["blocks"])
        if p["stopped_at"] is not None:
            n = min(n, p["stopped_at"])
        return int(n)

    def _trace(self, key: str, p: dict) -> np.ndarray:
        if key not in self._trace_cache:
            rng = np.random.default_rng([self.seed, zlib.crc32(key.encode()), p["series"]])
            n = p["blocks"]
            xi = rng.normal(size=n)
            noise = np.empty(n)
            scale = np.sqrt(max(0.0, 1.0 - self.rho ** 2))
            prev = xi[0] if n else 0.0
            for i in range(n):
                prev = xi[i] if i == 0 else self.rho * prev + scale * xi[i]
                noise[i] = prev
            e_true = self.true_surface(np.array(p["coords"]))
            trace = e_true + self.transient * np.exp(-np.arange(n) / self.equil_blocks) + self.sigma_block * noise
            self._trace_cache[key] = trace / HARTREE_TO_KCAL + self.e_offset
        return self._trace_cache[key]

    def read(self, key: str) -> PointData:
        p = self.points.get(key)
        if p is None:
            return PointData(key=key, state=ABSENT)
        n = self._available(p)
        if p["stopped_at"] is not None:
            state = STOPPED
        elif n >= p["blocks"]:
            state = DONE
        elif n > 0:
            state = RUNNING
        else:
            state = QUEUED
        e_sur = None
        if self.surrogate_surface is not None:
            xs = p.get("sur_coords") or p["coords"]
            e_sur = self.surrogate_surface(np.array(xs)) / HARTREE_TO_KCAL + self.e_offset
        energy = self._trace(key, p)[:n].copy()
        return PointData(key=key, state=state, energy=energy, weight=np.ones(n), e_surrogate=e_sur,
                         meta={"submitted_tick": p["submitted"], "requested_blocks": p["blocks"]})

    def stop(self, key: str) -> None:
        p = self.points.get(key)
        if p is None or p["stopped_at"] is not None:
            return
        n = self._available(p)
        if n >= p["blocks"]:
            return                      # already complete
        p["stopped_at"] = n
        self.save()

    def advance(self, ticks: int = 1) -> None:
        self.tick += int(ticks)
        self.save()

    def cost(self, key: str) -> float:
        p = self.points.get(key)
        return 0.0 if p is None else self._available(p) * self.core_hours_per_block

    def total_cost(self) -> float:
        """Core-hours of all points so far."""
        return float(sum(self.cost(k) for k in self.points))

    def total_blocks(self) -> int:
        return int(sum(self._available(p) for p in self.points.values()))
