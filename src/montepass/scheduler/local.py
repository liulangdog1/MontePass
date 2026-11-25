"""Run job scripts on this computer, one after the other (workstations, tests)."""
from __future__ import annotations

import os
import subprocess
from typing import Dict, List, Optional

from .base import CANCELLED, COMPLETED, FAILED, PENDING, RUNNING, UNKNOWN, Scheduler


class LocalScheduler(Scheduler):
    """``submit`` starts the script with ``bash`` as a background process. At most ``max_running`` scripts run
    at the same time; the others wait. A job that names ``after`` jobs waits for them."""

    def __init__(self, max_running: int = 1):
        self.max_running = int(max_running)
        self._jobs: Dict[str, dict] = {}
        self._n = 0

    def submit(self, script: str, workdir: str, name: str = "job", after: Optional[List[str]] = None) -> str:
        self._n += 1
        job_id = "L%d" % self._n
        self._jobs[job_id] = {"script": script, "workdir": workdir, "after": list(after or []), "proc": None,
                              "state": PENDING}
        self._start_waiting()
        return job_id

    @staticmethod
    def _close(job: dict) -> None:
        if job.get("out") is not None:
            job["out"].close(); job["out"] = None

    def _poll(self) -> None:
        for job in self._jobs.values():
            if job["state"] == RUNNING and job["proc"].poll() is not None:
                job["state"] = COMPLETED if job["proc"].returncode == 0 else FAILED
                self._close(job)

    def _start_waiting(self) -> None:
        self._poll()
        running = sum(1 for j in self._jobs.values() if j["state"] == RUNNING)
        for job in self._jobs.values():
            if job["state"] != PENDING:
                continue
            deps = [self._jobs[a]["state"] for a in job["after"] if a in self._jobs]
            if any(d in (FAILED, CANCELLED) for d in deps):
                job["state"] = CANCELLED; continue
            if any(d != COMPLETED for d in deps) or running >= self.max_running:
                continue
            job["out"] = open(os.path.join(job["workdir"], "local_job.out"), "a")
            job["proc"] = subprocess.Popen(["bash", job["script"]], cwd=job["workdir"], stdout=job["out"], stderr=subprocess.STDOUT)
            job["state"] = RUNNING; running += 1

    def state(self, job_id: str) -> str:
        self._start_waiting()
        job = self._jobs.get(job_id)
        return job["state"] if job else UNKNOWN

    def cancel(self, job_id: str) -> None:
        job = self._jobs.get(job_id)
        if not job:
            return
        if job["state"] == RUNNING:
            job["proc"].terminate()
            job["proc"].wait()
            self._close(job)
        if job["state"] in (RUNNING, PENDING):
            job["state"] = CANCELLED

    def wait_all(self) -> None:
        """Block until every job has ended (for tests)."""
        while True:
            self._start_waiting()
            procs = [j["proc"] for j in self._jobs.values() if j["state"] == RUNNING]
            if not procs:
                if not any(j["state"] == PENDING for j in self._jobs.values()):
                    return
                continue
            procs[0].wait()
