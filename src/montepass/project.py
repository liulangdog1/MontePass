"""One project on disk: its folders and its state file.

```
<project>/
  input.ini                      the input file
  work/project.json              state of the workflow (stages, targets)
  work/jobs.json                 every job that was submitted
  work/driver.log                what the workflow did, one line per event
  work/surrogate/                ORCA calculations
  work/path.xyz                  the path images (when there is a path)
  work/subspace.npz              the search subspace
  work/targets/<target>/         one folder per optimized geometry: start.xyz, directions.npz, search/
  work/qmc/                      QMC points (one folder per point)
  work/pyscf/                    surrogate gradient and Hessian jobs
  work/autotune/                 pilot runs and the chosen QMC settings
  work/trial/state.json          record of the stage "trial" (only with [trial] kind = multidet in the input file)
  results/                       tables, plots and animations
```
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Dict, List, Optional, Tuple

PENDING = "PENDING"
RUNNING = "RUNNING"
DONE = "DONE"
FAILED = "FAILED"
NEEDS_ATTENTION = "NEEDS_ATTENTION"

STAGES = ("path", "subspace", "directions", "autotune", "search", "report")
TRIAL = "trial"                 # a stage between "search" and "report" that only some projects have


def stage_names(trial: bool = False) -> Tuple[str, ...]:
    """The stages of a project in their order. The stage ``trial`` (DMC energies with several determinants at
    the start and the final geometry of every target) exists only when the input file asks for it."""
    if not trial:
        return STAGES
    i = STAGES.index("report")
    return STAGES[:i] + (TRIAL,) + STAGES[i:]


_STAMPED = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}  ")


class Project:
    def __init__(self, folder: str):
        self.folder = os.path.abspath(folder)
        self.work = os.path.join(self.folder, "work")
        self.results = os.path.join(self.folder, "results")
        self.state_file = os.path.join(self.work, "project.json")
        self.log_file = os.path.join(self.work, "driver.log")
        os.makedirs(self.work, exist_ok=True)
        if os.path.exists(self.state_file):
            with open(self.state_file) as fh:
                self.state = json.load(fh)
        else:
            self.state = {"name": os.path.basename(self.folder), "status": PENDING,
                          "stages": {s: {"status": PENDING, "message": ""} for s in STAGES},
                          "targets": {}, "symbols": None, "created": time.strftime("%Y-%m-%d %H:%M:%S")}
            self.save()

    # ------------------------------------------------------------------ paths
    @property
    def input_file(self) -> str:
        return os.path.join(self.folder, "input.ini")

    def path(self, *parts: str) -> str:
        """A path below ``work/``; the folder of the path is created."""
        p = os.path.join(self.work, *parts)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        return p

    def target_dir(self, target: str) -> str:
        d = os.path.join(self.work, "targets", target)
        os.makedirs(d, exist_ok=True)
        return d

    @property
    def path_file(self) -> str:
        return os.path.join(self.work, "path.xyz")

    @property
    def subspace_file(self) -> str:
        return os.path.join(self.work, "subspace.npz")

    # ------------------------------------------------------------------ state
    def save(self) -> None:
        tmp = self.state_file + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(self.state, fh, indent=1)
        os.replace(tmp, self.state_file)

    def stage(self, name: str) -> Dict[str, str]:
        return self.state["stages"][name]

    def set_stage(self, name: str, status: str, message: str = "") -> None:
        st = self.state["stages"][name]
        changed = st["status"] != status or st["message"] != message
        st["status"], st["message"] = status, message
        if changed:
            self.log("stage %s: %s%s" % (name, status, (" (" + message + ")") if message else ""))
        self.save()

    def add_stage(self, name: str, before: str) -> bool:
        """Put a stage that the state does not hold yet in front of the stage ``before``, with the status
        PENDING. Returns True if the stage was added, False if the state held it already."""
        stages = self.state["stages"]
        if name in stages:
            return False
        ordered: Dict[str, Dict[str, str]] = {}
        for key, value in stages.items():
            if key == before:
                ordered[name] = {"status": PENDING, "message": ""}
            ordered[key] = value
        ordered.setdefault(name, {"status": PENDING, "message": ""})       # no stage ``before``: at the end
        self.state["stages"] = ordered
        self.log("stage %s: added to the project, in front of the stage %s" % (name, before))
        self.save()
        return True

    def remove_stage(self, name: str) -> bool:
        """Take a stage out of the state. Returns True if the state held it. No file of the stage is deleted."""
        if name not in self.state["stages"]:
            return False
        del self.state["stages"][name]
        self.log("stage %s: taken out of the project state" % name)
        self.save()
        return True

    def log(self, text: str) -> None:
        """Append one line to the log of the project. A line that already starts with a time stamp (the
        search drivers and the pilot write such lines) is written as it is."""
        line = text if _STAMPED.match(text) else "%s  %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), text)
        with open(self.log_file, "a") as fh:
            fh.write(line + "\n")

    def targets(self) -> List[str]:
        return list(self.state["targets"])

    def target(self, name: str) -> dict:
        return self.state["targets"][name]

    def add_target(self, name: str, kind: str, protocol: str) -> None:
        if name not in self.state["targets"]:
            self.state["targets"][name] = {"kind": kind, "protocol": protocol, "status": PENDING, "message": ""}
            self.save()

    def last_log_lines(self, n: int = 20) -> List[str]:
        if not os.path.exists(self.log_file):
            return []
        with open(self.log_file) as fh:
            return fh.read().splitlines()[-n:]
