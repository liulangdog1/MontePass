"""The engine interface.

The geometry search never talks to QMCPACK or to the batch system directly. It
asks an *engine* for the DMC energy trace of a geometry:

    engine.submit(PointSpec(...))     # create the point and start its job; does nothing if it exists
    data = engine.read(key)           # block energies acquired so far (the job may still be running)
    engine.stop(key)                  # stop the job early; the blocks written so far stay usable

A point is identified by its ``key``, a short path-like string that is unique
inside one project, for example ``search/min/it1/d0_p091``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

ABSENT = "ABSENT"        # the point was never submitted
QUEUED = "QUEUED"        # submitted, no DMC block written yet
RUNNING = "RUNNING"      # DMC blocks are being written
DONE = "DONE"            # all requested blocks are written
STOPPED = "STOPPED"      # stopped early by the driver; the blocks written so far are final
FAILED = "FAILED"        # the job ended with an error

FINAL_STATES = (DONE, STOPPED, FAILED)


@dataclass
class PointSpec:
    """Request for the DMC energy of one geometry."""

    key: str
    symbols: List[str]
    coords: np.ndarray                      # (n_atoms, 3), angstrom
    blocks: int                             # number of DMC blocks requested (upper limit of this point)
    role: str = "scan"                      # "scan", "centre", "final" or "pilot"
    series: int = 1                         # 1 = first run; 2, 3, ... = further runs with disjoint random seeds
    base_key: Optional[str] = None          # for series > 1: key of the series-1 point of the same geometry. The
                                            # further run uses its orbitals and its Jastrow factor, with new seeds.
    settings: Dict[str, object] = field(default_factory=dict)      # QMC settings chosen for this point


@dataclass
class PointData:
    """What is known about a point right now."""

    key: str
    state: str
    energy: np.ndarray = field(default_factory=lambda: np.zeros(0))    # DMC block energies, hartree
    weight: np.ndarray = field(default_factory=lambda: np.zeros(0))    # DMC block weights
    e_surrogate: Optional[float] = None     # surrogate (DFT) energy at the same geometry, hartree
    message: str = ""
    meta: Dict[str, object] = field(default_factory=dict)

    @property
    def nblocks(self) -> int:
        return int(len(self.energy))

    @property
    def final(self) -> bool:
        return self.state in FINAL_STATES


class Engine:
    """Base class of all engines."""

    def submit(self, spec: PointSpec) -> None:
        """Create the point and start its calculation. Calling it again for an existing key does nothing."""
        raise NotImplementedError

    def read(self, key: str) -> PointData:
        """Current data of a point. Returns state ABSENT for an unknown key."""
        raise NotImplementedError

    def stop(self, key: str) -> None:
        """Stop the calculation of a point. Its blocks written so far remain readable."""
        raise NotImplementedError

    def advance(self) -> None:
        """Let time pass. Real engines do nothing here; the mock engine produces more blocks."""
        return None

    def retry(self, key: str) -> bool:
        """Give a FAILED point a new set of attempts. Returns True if the point is active again. An engine that
        cannot do this returns False."""
        return False

    def cost(self, key: str) -> float:
        """Computer time spent on a point so far, in core-hours (0 if unknown)."""
        return 0.0
