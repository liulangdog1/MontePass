"""SLURM: ``sbatch``, ``squeue``, ``sacct``, ``scancel``."""
from __future__ import annotations

import os
import subprocess
from typing import Dict, List, Optional

from .base import CANCELLED, COMPLETED, FAILED, PENDING, RUNNING, TIMEOUT, UNKNOWN, Scheduler

_MAP = {
    "PENDING": PENDING, "CONFIGURING": PENDING, "REQUEUED": PENDING, "RESIZING": PENDING, "SUSPENDED": PENDING,
    "RUNNING": RUNNING, "COMPLETING": RUNNING, "STAGE_OUT": RUNNING,
    "COMPLETED": COMPLETED,
    "FAILED": FAILED, "NODE_FAIL": FAILED, "OUT_OF_MEMORY": FAILED, "BOOT_FAIL": FAILED, "DEADLINE": FAILED,
    "PREEMPTED": FAILED,
    "CANCELLED": CANCELLED, "REVOKED": CANCELLED,
    "TIMEOUT": TIMEOUT,
}


def _run(cmd: List[str], timeout: int = 60) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ""


def parse_state(text: str) -> str:
    """Map a SLURM state word (for example ``CANCELLED by 1234`` or ``COMPLETED``) to a package state."""
    word = text.strip().split()[0].rstrip("+") if text.strip() else ""
    return _MAP.get(word, UNKNOWN)


class SlurmScheduler(Scheduler):
    def __init__(self, user: Optional[str] = None):
        self.user = user or os.environ.get("USER", "")

    def submit(self, script: str, workdir: str, name: str = "job", after: Optional[List[str]] = None) -> str:
        cmd = ["sbatch", "--parsable", "--chdir=%s" % workdir, "--job-name=%s" % name]
        if after:
            cmd.append("--dependency=afterok:%s" % ":".join(str(a) for a in after))
        cmd.append(script)
        res = subprocess.run(cmd, capture_output=True, text=True)
        out = res.stdout.strip()
        job_id = out.split(";")[0].strip() if out else ""
        if res.returncode != 0 or not job_id.isdigit():
            raise RuntimeError("sbatch failed for %s: %s %s" % (script, out, res.stderr.strip()))
        return job_id

    def states(self, job_ids: List[str]) -> Dict[str, str]:
        ids = [str(j) for j in job_ids]
        out: Dict[str, str] = {}
        if not ids:
            return out
        # jobs in the queue
        for line in _run(["squeue", "-h", "-j", ",".join(ids), "-o", "%i|%T"]).splitlines():
            parts = line.split("|")
            if len(parts) == 2 and parts[0] in ids:
                out[parts[0]] = parse_state(parts[1])
        # jobs that left the queue: the accounting record
        missing = [j for j in ids if j not in out]
        if missing:
            for line in _run(["sacct", "-X", "-n", "-P", "-j", ",".join(missing), "-o", "JobID,State"]).splitlines():
                parts = line.split("|")
                if len(parts) == 2 and parts[0] in missing:
                    out[parts[0]] = parse_state(parts[1])
        for j in ids:
            out.setdefault(j, UNKNOWN)
        return out

    def state(self, job_id: str) -> str:
        return self.states([job_id])[str(job_id)]

    def cancel(self, job_id: str) -> None:
        _run(["scancel", str(job_id)])

    def cores_in_use(self) -> Optional[int]:
        text = _run(["squeue", "-h", "-u", self.user, "-o", "%C"])
        try:
            return sum(int(x) for x in text.split())
        except ValueError:
            return None
