"""Command line.

    montepass init   [folder]          write a commented input file
    montepass check  [folder]          read the input and the machine profile, print the plan
    montepass run    [folder]          advance the workflow until it ends (or once, with --once)
    montepass submit [folder]          start "run" as a small batch job
    montepass status [folder]          what is done, what runs, what needs attention
    montepass report [folder]          write the results folder again
    montepass demo   water|ammonia [folder]   a complete run on a model molecule, on this computer

The same commands work as ``python -m montepass <command> ...``.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List, Optional

from .config import Config, ConfigError, TrialSetupError, write_template
from .project import DONE, NEEDS_ATTENTION, Project

DEMO_INPUT = """\
[system]
name = {name}

[geometry]
mode = {mode}
targets = {targets}

[model]
molecule = {molecule}
sigma_block = 2.0
seed = 1

[search]
method = dense-v2
eps_total = 0.05
"""


def _print_status(folder: str) -> None:
    p = Project(folder)
    print("project %s   status %s" % (p.state.get("name", ""), p.state.get("status", "")))
    for stage, st in p.state["stages"].items():
        print("  %-11s %-16s %s" % (stage, st["status"], st["message"]))
    for name, t in p.state["targets"].items():
        print("  target %-10s %-8s %-16s %s" % (name, t["kind"], t["status"], t.get("message", "")))
    lines = p.last_log_lines(8)
    if lines:
        print("last log lines:")
        for ln in lines:
            print("  " + ln)


def cmd_init(args) -> int:
    os.makedirs(args.folder, exist_ok=True)
    path = os.path.join(args.folder, "input.ini")
    try:
        write_template(path)
    except FileExistsError as exc:
        print(exc); return 1
    print("wrote %s . Edit it, put the geometry files next to it, then run:  montepass check %s" % (path, args.folder))
    return 0


def cmd_check(args) -> int:
    try:
        cfg = Config.load(os.path.join(args.folder, "input.ini"))
    except ConfigError as exc:
        print(exc); return 1
    s = cfg.summary()
    print("input file is valid: %s" % cfg.path)
    print("  mode %s, targets %s, search method %s, tolerance %.3g kcal/mol" % (
        s["mode"], ", ".join(s["targets"]), s["search"]["method"], s["search"]["eps_total"]))
    if cfg.model is not None:
        print("  model run (no cluster programs): %s" % cfg.model)
        if cfg.trial is not None:
            from .workflow import MODEL_WITHOUT_QP2
            print("  PROBLEM: " + MODEL_WITHOUT_QP2)
            return 1
        return 0
    from .machine import Machine, find_profile
    try:
        m = Machine(find_profile(cfg.machine_profile, [cfg.folder]))
    except FileNotFoundError as exc:
        print(exc); return 1
    problems = m.check()
    plan: List[str] = []
    if cfg.trial is not None:                   # the stage "trial": what the profile lacks for it, and its plan
        from .trial import machine_problems, plan_lines
        problems += machine_problems(m, cfg.trial)
        plan = plan_lines(cfg.trial, cfg.targets)
    print("machine profile %s (%s)" % (m.name, m.path))
    for pr in problems:
        print("  PROBLEM: " + pr)
    for line in plan:
        print(line)
    return 1 if problems else 0


def _open_workflow(folder: str):
    """The workflow of a project, or a message and None when [trial] kind = multidet cannot run with this project
    (a machine profile that cannot run the stage trial, or [trial] together with [model]). Every other error is
    raised as before: main() prints an error of the input file, any other error ends with its traceback."""
    from .workflow import Workflow
    try:
        return Workflow(folder)
    except TrialSetupError as exc:
        print("the project %s cannot be opened: %s" % (folder, exc))
        return None


def cmd_run(args) -> int:
    wf = _open_workflow(args.folder)
    if wf is None:
        return 1
    passes = 0
    while True:
        status = wf.advance()
        passes += 1
        if status in (DONE, NEEDS_ATTENTION) or args.once or (args.max_passes and passes >= args.max_passes):
            break
        if args.interval > 0:
            time.sleep(args.interval)
    _print_status(args.folder)
    return 0 if status == DONE else (2 if status == NEEDS_ATTENTION else 3)


def cmd_submit(args) -> int:
    from .machine import Machine, find_profile
    cfg = Config.load(os.path.join(args.folder, "input.ini"))
    if cfg.trial is not None and cfg.model is not None:
        from .workflow import MODEL_WITHOUT_QP2
        print("PROBLEM: " + MODEL_WITHOUT_QP2)
        print("no driver job was submitted")
        return 1
    m = Machine(find_profile(cfg.machine_profile, [cfg.folder]))
    if cfg.trial is not None:
        # the stage trial runs after the whole search: a profile that cannot run it is refused before the first job
        from .trial import machine_problems
        problems = machine_problems(m, cfg.trial)
        if problems:
            print("%s asks for [trial] kind = multidet, but the machine profile %s cannot run that stage:" % (cfg.path, m.path))
            for pr in problems:
                print("  PROBLEM: " + pr)
            print("no driver job was submitted")
            return 1
    p = Project(args.folder)
    script = p.path("driver.sh")
    body = m.header("driver", "mpass_" + cfg.name[:20], p.folder, log=os.path.join("work", "driver.out"))
    body += "%s -m montepass run %s --interval %d\n" % (m.executable("python"), p.folder, args.interval)
    with open(script, "w") as fh:
        fh.write(body)
    job_id = m.scheduler().submit(script, p.folder, "mpass_" + cfg.name[:20])
    p.log("driver job submitted: %s" % job_id)
    print("driver job %s submitted. Follow it with:  montepass status %s" % (job_id, args.folder))
    return 0


def cmd_status(args) -> int:
    _print_status(args.folder)
    return 0


def cmd_retry(args) -> int:
    wf = _open_workflow(args.folder)
    if wf is None:
        return 1
    lines = wf.retry()
    if not lines:
        print("nothing to do: no calculation has failed and no stage needs attention")
        return 0
    for line in lines:
        print(line)
    print("Now start the driver again:  montepass submit %s" % args.folder)
    return 0


def cmd_report(args) -> int:
    from .report import build_report
    wf = _open_workflow(args.folder)
    if wf is None:
        return 1
    files = build_report(wf.project.folder, wf.engine(), title=wf.cfg.name)
    print("%d file(s) written to %s" % (len(files), wf.project.results))
    return 0


def cmd_demo(args) -> int:
    folder = args.folder or ("demo_" + args.molecule)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "input.ini")
    if not os.path.exists(path):
        if args.molecule == "ammonia":
            text = DEMO_INPUT.format(name="model_ammonia_inversion", mode="path", targets="reactant, saddle", molecule="ammonia")
        else:
            text = DEMO_INPUT.format(name="model_water", mode="single", targets="geometry", molecule="water")
        with open(path, "w") as fh:
            fh.write(text)
    args.folder, args.once, args.max_passes, args.interval = folder, False, 5000, 0
    return cmd_run(args)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="montepass", description="Geometry and reaction-path optimization on the DMC energy surface.")
    sub = ap.add_subparsers(dest="command")
    for name, fn, text in (("init", cmd_init, "write a commented input file"), ("check", cmd_check, "check input and machine profile"),
                           ("status", cmd_status, "show the state of a project"), ("report", cmd_report, "write the results folder"),
                           ("retry", cmd_retry, "submit failed calculations again and clear NEEDS_ATTENTION")):
        sp = sub.add_parser(name, help=text); sp.add_argument("folder", nargs="?", default="."); sp.set_defaults(fn=fn)
    sp = sub.add_parser("run", help="advance the workflow until it ends")
    sp.add_argument("folder", nargs="?", default=".")
    sp.add_argument("--once", action="store_true", help="one pass only")
    sp.add_argument("--interval", type=int, default=60, help="seconds between passes (default 60)")
    sp.add_argument("--max-passes", type=int, default=0, help="stop after this many passes (0 = no limit)")
    sp.set_defaults(fn=cmd_run)
    sp = sub.add_parser("submit", help='start "run" as a batch job')
    sp.add_argument("folder", nargs="?", default="."); sp.add_argument("--interval", type=int, default=60); sp.set_defaults(fn=cmd_submit)
    sp = sub.add_parser("demo", help="a complete run on a model molecule (no cluster programs)")
    sp.add_argument("molecule", choices=("water", "ammonia")); sp.add_argument("folder", nargs="?", default=None); sp.set_defaults(fn=cmd_demo)
    args = ap.parse_args(argv)
    if not getattr(args, "fn", None):
        ap.print_help(); return 2
    try:
        return args.fn(args)
    except ConfigError as exc:
        print(exc); return 1


if __name__ == "__main__":
    sys.exit(main())
