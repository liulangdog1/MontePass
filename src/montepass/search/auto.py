"""The automatic search: the classic line search first, the dense-scan search when it is needed.

The classic line search is the cheaper method when its cubic fits are a good model of the energy
inside its windows. When the windows are much wider than the range in which a cubic holds, the
fitted step depends on the model and the error bar of the fit does not show it. The stability
audit (module ``audit``) measures that dependence with the energies of the iteration itself, and
the bracket check says whether the fitted stationary point lies inside the scanned range.

Rule of the automatic search:

1. Run the first iteration of the classic line search from the starting geometry.
2. If every direction of that iteration is ``RESOLVED`` by the audit and bracketed by its grid,
   the classic search continues to its end. Its later audits are recorded and reported.
3. Otherwise the classic search stops, and the dense-scan search (protocol v2) starts from the
   same starting geometry. The full-length runs at the starting geometry, which the classic
   search already asked for, are used again; the scan points of the classic iteration are not.

Files in the state folder: ``<name>_auto.json`` (which phase is active and why),
``<name>_state.json`` (the state of the active search) and, after a switch,
``<name>_classic_state.json`` (the state of the stopped classic search, kept for the record).
"""
from __future__ import annotations

import json
import os
from typing import Callable, Dict, Optional, Sequence

import numpy as np

from ..engine.base import Engine
from ..surrogate.base import Surrogate
from .classic import UNRESOLVED_STATUS, ClassicLineSearch
from .dense import DenseScanSearch

CLASSIC, SWITCHING, DENSE = "classic", "switching", "dense"


class AutoSearch:
    """Search of one geometry with the rule above. The interface is the one of the two searches it runs:
    ``tick()``, ``state``, ``status`` and the result methods of the active search."""

    def __init__(self, name: str, engine: Engine, surrogate: Optional[Surrogate], symbols: Sequence[str],
                 start: np.ndarray, vectors: np.ndarray, curvature: Sequence[float], protocol=None,
                 state_dir: str = ".", settings: Optional[Dict[str, object]] = None,
                 log: Optional[Callable[[str], None]] = None):
        self.name = name
        self.state_dir = state_dir
        self._args = dict(engine=engine, surrogate=surrogate, symbols=list(symbols),
                          start=np.asarray(start, dtype=float).reshape(-1, 3),
                          vectors=np.atleast_2d(np.asarray(vectors, dtype=float)),
                          curvature=np.asarray(curvature, dtype=float).reshape(-1))
        self._protocol = protocol
        self._settings = dict(settings or {})
        self._log = log
        os.makedirs(state_dir, exist_ok=True)
        self.meta_file = os.path.join(state_dir, "%s_auto.json" % name)
        self.state_file = os.path.join(state_dir, "%s_state.json" % name)
        self.classic_record = os.path.join(state_dir, "%s_classic_state.json" % name)
        if os.path.exists(self.meta_file):
            with open(self.meta_file) as fh:
                self.meta = json.load(fh)
        else:
            self.meta = {"phase": CLASSIC, "reason": ""}
            self._save_meta()
        self.active = None
        if self.meta["phase"] == CLASSIC:
            self.active = self._classic()
        else:
            self._finish_switch()

    # ------------------------------------------------------------------ the two searches
    def _save_meta(self) -> None:
        tmp = self.meta_file + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(self.meta, fh, indent=1)
        os.replace(tmp, self.meta_file)

    def _classic(self) -> ClassicLineSearch:
        a = self._args
        settings = dict(self._settings)
        settings["classic"] = dict(settings.get("classic") or {}, stop_if_unresolved=True)
        return ClassicLineSearch(self.name, a["engine"], a["surrogate"], a["symbols"], a["start"], a["vectors"],
                                 a["curvature"], settings=settings, state_dir=self.state_dir, log=self._log)

    def _dense(self):
        a = self._args
        settings = {k: v for k, v in self._settings.items() if k != "classic"}
        return DenseScanSearch(self.name, a["engine"], a["surrogate"], a["symbols"], a["start"], a["vectors"],
                               a["curvature"], protocol=self._protocol, state_dir=self.state_dir, settings=settings,
                               log=self._log)

    def _finish_switch(self) -> None:
        """Move the state of the stopped classic search aside and start (or load) the dense-scan search.
        Every step can be repeated, so a process that stops in the middle continues here."""
        if os.path.exists(self.state_file):
            with open(self.state_file) as fh:
                protocol = json.load(fh).get("protocol")
            if protocol == "classic":
                os.replace(self.state_file, self.classic_record)
        self.active = self._dense()
        if self.meta["phase"] != DENSE:
            self.meta["phase"] = DENSE
            self._save_meta()

    def _switch(self, classic: ClassicLineSearch) -> None:
        self.meta.update(phase=SWITCHING, reason=classic.state.get("message", ""))
        self._save_meta()
        if self._log:
            self._log("target %s: the classic line search stopped after its first iteration (%s). "
                      "The dense-scan search starts from the same geometry." % (self.name, self.meta["reason"]))
        self._finish_switch()

    # ------------------------------------------------------------------ interface of a search
    @property
    def phase(self) -> str:
        return self.meta["phase"]

    @property
    def state(self) -> dict:
        return self.active.state

    @property
    def status(self) -> str:
        return self.active.state["status"]

    def tick(self) -> str:
        status = self.active.tick()
        if self.meta["phase"] == CLASSIC and status == UNRESOLVED_STATUS:
            self._switch(self.active)
            status = self.active.tick()
        return status

    def clear_attention(self) -> str:
        return self.active.clear_attention()

    def run(self, max_ticks: int = 10000) -> str:
        status = self.status
        for _ in range(max_ticks):
            status = self.tick()
            if status in ("CONVERGED", "NOT_CONVERGED", "DONE", "NEEDS_ATTENTION"):
                break
            self._args["engine"].advance()
        return status

    def __getattr__(self, item):
        # result methods (final_geometry, final_energy, trajectory, ...) are those of the active search
        if item.startswith("_") or item in ("active", "meta"):
            raise AttributeError(item)
        return getattr(self.active, item)
