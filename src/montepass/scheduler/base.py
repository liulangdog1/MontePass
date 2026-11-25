"""The scheduler interface and the job table.

Every calculation of the package is a shell script in its own folder. A
scheduler starts such a script and reports its state. The package never finds a
job by its name: every submitted job is written into a job table (a JSON file)
with its id, its folder and its purpose.
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional

PENDING = "PENDING"          # waiting in the queue
RUNNING = "RUNNING"
COMPLETED = "COMPLETED"      # ended with exit code 0
FAILED = "FAILED"            # ended with an error (also: node failure, out of memory)
CANCELLED = "CANCELLED"
TIMEOUT = "TIMEOUT"          # stopped at its time limit
UNKNOWN = "UNKNOWN"          # the batch system has no record of the job

FINISHED_STATES = (COMPLETED, FAILED, CANCELLED, TIMEOUT)


class Scheduler:
    """Base class. ``submit`` returns a job id (a string)."""

    def submit(self, script: str, workdir: str, name: str = "job", after: Optional[List[str]] = None) -> str:
        """Start ``script`` (a path) with ``workdir`` as its working folder. ``after`` lists job ids that
        must end successfully before this job starts."""
        raise NotImplementedError

    def state(self, job_id: str) -> str:
        raise NotImplementedError

    def states(self, job_ids: List[str]) -> Dict[str, str]:
        """States of several jobs. Subclasses can answer with one call to the batch system."""
        return {j: self.state(j) for j in job_ids}

    def cancel(self, job_id: str) -> None:
        raise NotImplementedError

    def cores_in_use(self) -> Optional[int]:
        """CPUs that all jobs of the user hold or wait for right now, or None if unknown."""
        return None


class JobTable:
    """Record of every job that the package submitted: ``{key: [{"id", "workdir", "name", "kind", "time"}, ...]}``.

    ``key`` identifies what the job computes (for example a point key). The newest entry of a key is
    its current job. The file is rewritten after every change, so a new process sees all earlier jobs.
    """

    def __init__(self, path: str):
        self.path = path
        self.jobs: Dict[str, list] = {}
        if os.path.exists(path):
            with open(path) as fh:
                self.jobs = json.load(fh)

    def _save(self) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(self.jobs, fh, indent=1)
        os.replace(tmp, self.path)

    def add(self, key: str, job_id: str, workdir: str, name: str = "", kind: str = "") -> None:
        self.jobs.setdefault(key, []).append({"id": str(job_id), "workdir": workdir, "name": name, "kind": kind,
                                              "time": time.strftime("%Y-%m-%d %H:%M:%S")})
        self._save()

    def current(self, key: str) -> Optional[dict]:
        entries = self.jobs.get(key)
        return entries[-1] if entries else None

    def attempts(self, key: str) -> int:
        return len(self.jobs.get(key, []))

    def all_ids(self) -> List[str]:
        return [e["id"] for entries in self.jobs.values() for e in entries]
