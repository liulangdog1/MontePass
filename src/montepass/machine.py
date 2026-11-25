"""Machine profile: everything that depends on the computer, read from one INI file.

A profile names the programs, the lines that prepare the environment of a job
(module loads, exports), the launch command of parallel programs, the folder of
the pseudopotential files, and the batch options of each kind of job. The rest
of the package contains no path and no module name.

Kinds of job used by the package:

    scf      PySCF calculations (orbitals, surrogate gradients and Hessians); threads, no MPI
    qmc      QMCPACK runs (Jastrow optimization, VMC, DMC); MPI
    orca     ORCA calculations (optimization, path, frequencies)
    driver   the one-core job that advances the workflow
    qp2      Quantum Package 2 (orbitals and determinants of a multi-determinant trial wave function); one
             process with ``cpus_per_task`` OpenMP threads. Needed only for ``[trial] kind = multidet``.

See ``machines/example_slurm.ini`` for a commented example.
"""
from __future__ import annotations

import configparser
import os
from typing import Dict, List, Optional

KINDS = ("scf", "qmc", "orca", "driver", "qp2")


class Machine:
    def __init__(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError("machine profile not found: %s" % path)
        cp = configparser.ConfigParser(interpolation=None)
        cp.optionxform = str
        cp.read(path)
        self.path = os.path.abspath(path)
        self.cp = cp
        self.name = cp.get("machine", "name", fallback=os.path.basename(path))
        self.scheduler_kind = cp.get("machine", "scheduler", fallback="slurm").strip().lower()
        if self.scheduler_kind not in ("slurm", "local"):
            raise ValueError("%s: scheduler must be slurm or local, not %r" % (path, self.scheduler_kind))
        self.pseudo_dir = os.path.expanduser(cp.get("pseudopotentials", "dir", fallback="").strip())

    # ------------------------------------------------------------------ simple values
    def executable(self, name: str) -> str:
        """Command of a program: python, qmcpack, convert4qmc or orca."""
        return self.cp.get("executables", name, fallback=name).strip()

    def setup_lines(self, kind: str) -> List[str]:
        """Shell lines that prepare the environment of a job of this kind."""
        text = self.cp.get("setup.%s" % kind, "lines", fallback="")
        return [ln.strip() for ln in text.splitlines() if ln.strip()]

    def job(self, kind: str) -> Dict[str, str]:
        """Batch options of a kind of job: the common ``[slurm]`` values overlaid with ``[job.<kind>]``."""
        out: Dict[str, str] = {}
        for sec in ("slurm", "job.%s" % kind):
            if self.cp.has_section(sec):
                out.update({k: v.strip() for k, v in self.cp.items(sec)})
        return out

    def tasks(self, kind: str) -> int:
        return int(self.job(kind).get("tasks", "1"))

    def threads(self, kind: str) -> int:
        return int(self.job(kind).get("cpus_per_task", "1"))

    def launch(self, kind: str = "qmc", ranks: Optional[int] = None) -> str:
        """Command prefix that starts a parallel program, for example ``mpirun -np 64 --bind-to core``."""
        j = self.job(kind)
        ranks = int(ranks or j.get("tasks", "1"))
        nodes = int(j.get("nodes", "1"))
        template = self.cp.get("launch", kind, fallback="mpirun -np {ranks}")
        return template.format(ranks=ranks, nodes=nodes, ranks_per_node=max(1, ranks // max(1, nodes)))

    def scheduler(self):
        if self.scheduler_kind == "slurm":
            from .scheduler.slurm import SlurmScheduler
            return SlurmScheduler()
        from .scheduler.local import LocalScheduler
        return LocalScheduler(max_running=int(self.cp.get("machine", "max_running", fallback="1")))

    # ------------------------------------------------------------------ job scripts
    def header(self, kind: str, name: str, workdir: str, log: str = "job.out", time: Optional[str] = None,
               tasks: Optional[int] = None, exclude: Optional[str] = None, mem: Optional[str] = None) -> str:
        """First part of a job script: interpreter line, batch options, environment.

        The script stops at the first failing setup line (``module load`` of a missing module, for example).
        ``OMP_NUM_THREADS`` and ``MKL_NUM_THREADS`` are always set: 1 for MPI jobs, the thread count for PySCF.
        A job of kind ``qp2`` also starts with 1: its script gives the threads to the ``qp_run`` lines only.
        """
        j = self.job(kind)
        lines = ["#!/bin/bash"]
        if self.scheduler_kind == "slurm":
            opts = [("--job-name", name[:60]), ("--chdir", workdir), ("--output", os.path.join(workdir, log)),
                    ("--nodes", j.get("nodes", "1")), ("--ntasks", str(tasks or j.get("tasks", "1")))]
            if j.get("tasks_per_node"):
                opts.append(("--ntasks-per-node", j["tasks_per_node"]))
            if j.get("cpus_per_task"):
                opts.append(("--cpus-per-task", j["cpus_per_task"]))
            opts += [("--mem", mem or j.get("mem", "8G")), ("--time", time or j.get("time", "01:00:00"))]
            for key in ("account", "qos", "partition", "constraint"):
                if j.get(key):
                    opts.append(("--" + key, j[key]))
            lines += ["#SBATCH %s=%s" % kv for kv in opts]
            lines.append("#SBATCH --no-requeue")
            if exclude:
                lines.append("#SBATCH --exclude=%s" % exclude)          # nodes on which an earlier attempt ran too slowly
            for extra in j.get("extra", "").splitlines():
                if extra.strip():
                    lines.append("#SBATCH %s" % extra.strip())
        threads = j.get("cpus_per_task", "1") if kind == "scf" else "1"
        lines += ["set -u", "cd %s || exit 97" % workdir]
        for ln in self.setup_lines(kind):
            lines.append("%s || { echo SETUP_FAILED: %s; exit 96; }" % (ln, ln.replace("'", "")))
        lines += ["export OMP_NUM_THREADS=%s" % threads, "export MKL_NUM_THREADS=%s" % threads]
        tools = self.cp.get("executables", "pyscf_qmctools", fallback="").strip()
        if tools:
            lines.append('export PYTHONPATH=%s:${PYTHONPATH:-}' % tools)
        pkg = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # folder that holds montepass/
        lines.append('export PYTHONPATH=%s:${PYTHONPATH:-}' % pkg)
        return "\n".join(lines) + "\n"

    def check(self) -> List[str]:
        """Problems of the profile that can be seen without running a job (an empty list means none)."""
        problems = []
        for sec in ("executables", "slurm" if self.scheduler_kind == "slurm" else "machine"):
            if not self.cp.has_section(sec):
                problems.append("section [%s] is missing" % sec)
        if not self.pseudo_dir:
            problems.append("[pseudopotentials] dir is not set")
        elif not os.path.isdir(self.pseudo_dir):
            problems.append("pseudopotential folder does not exist: %s" % self.pseudo_dir)
        for kind in ("scf", "qmc"):
            if self.scheduler_kind == "slurm" and not self.cp.has_section("job.%s" % kind):
                problems.append("section [job.%s] is missing" % kind)
        return problems

    def check_qp2(self) -> List[str]:
        """Problems of the profile for Quantum Package 2 jobs (an empty list means none). The package asks
        this only when a multi-determinant trial wave function is requested."""
        problems = []
        if not self.setup_lines("qp2"):
            problems.append("section [setup.qp2] has no lines (they must put qp_run and qp_create_ezfio on the PATH)")
        if self.scheduler_kind == "slurm":
            if not self.cp.has_section("job.qp2"):
                problems.append("section [job.qp2] is missing (1 task with cpus_per_task threads, mem, time)")
            else:
                j = self.job("qp2")
                if int(j.get("tasks", "1")) != 1:
                    problems.append("[job.qp2] tasks must be 1: Quantum Package 2 is one process with threads")
                for key in ("cpus_per_task", "mem", "time"):
                    if not self.cp.has_option("job.qp2", key):
                        problems.append("[job.qp2] %s is not set" % key)
                if self.cp.has_option("job.qp2", "time"):
                    from .engine.qmcpack import parse_time     # imported late: machine.py is read before the engine
                    seconds = parse_time(j.get("time"))
                    if not seconds or seconds <= 900:
                        problems.append("[job.qp2] time = %s must be more than 15 minutes: the CIPSI step ends 10 minutes "
                                        "before the time limit of its job" % j.get("time"))
        if not self.cp.get("executables", "determinants_tools", fallback="").strip():
            problems.append("[executables] determinants_tools is not set (path of utils/determinants_tools.py of QMCPACK)")
        return problems


def find_profile(name_or_path: str, search: Optional[List[str]] = None) -> str:
    """Path of a profile given as a path or as a name. Names are looked up as ``<name>.ini`` in the folders
    of ``search``, then in ``$MONTEPASS_MACHINES``, ``~/.config/montepass/machines`` and the ``machines``
    folders of the source tree (``machines/local`` first)."""
    if os.path.exists(name_or_path):
        return os.path.abspath(name_or_path)
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    folders = list(search or [])
    if os.environ.get("MONTEPASS_MACHINES"):
        folders.append(os.environ["MONTEPASS_MACHINES"])
    folders += [os.path.expanduser("~/.config/montepass/machines"), os.path.join(root, "machines", "local"),
                os.path.join(root, "machines")]
    for folder in folders:
        cand = os.path.join(folder, name_or_path if name_or_path.endswith(".ini") else name_or_path + ".ini")
        if os.path.exists(cand):
            return os.path.abspath(cand)
    raise FileNotFoundError("machine profile %r not found in: %s" % (name_or_path, ", ".join(folders)))
