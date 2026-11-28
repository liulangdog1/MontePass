"""The results folder of a project: tables, plots and animations.

    from montepass.report import build_report
    files = build_report(project_dir, engine, title="my molecule")

``build_report`` reads the project folder (it never changes ``work/``) and writes ``results/``:

    summary.md, summary.json     energies, energy changes, barriers, search history, what was skipped and why
    energies.csv                 one row per point that has DMC blocks
    trial_wavefunctions.csv, trial_wavefunctions.md
                                 only for a project whose input asks for the stage "trial" and that has a record of
                                 it: DMC energies with several determinants at the start and the final geometry of
                                 every target, each cut against the single RHF determinant, and the convergence of
                                 the energy differences with the size of the expansion
    geometries/                  <target>_start.xyz, <target>_final.xyz, <target>_trajectory.xyz, bond_table.csv,
                                 path_internal_coordinates.csv (every bond length and angle of every path image)
    plots/                       pca_variance, path_profile, path_geometry (bond lengths and angles along the path),
                                 <target>_convergence, <target>_scans, <target>_geometry
                                 (each as PNG at 200 dots per inch and as PDF)
    gifs/                        path.gif (the molecule along the path, with its bond lengths and angles),
                                 <target>_optimization.gif

Parts that cannot be made (no path, a search that still runs, a missing file) are skipped. The summary
lists every skipped item with the reason. One failing plot never stops the report.

Modules: ``tables`` (reading the project, numbers, text files), ``trial`` (the tables of the stage "trial"),
``plots`` (figure style and static plots), ``molview`` (drawing of a molecule), ``anim`` (GIF files), ``demo``
(a complete small project on a model molecule, for demonstrations and tests).
"""
from __future__ import annotations

import json
import os
import warnings
from typing import Callable, List, Optional, Tuple

from . import tables                                # noqa: E402  (tables does not import matplotlib)

with warnings.catch_warnings():
    # Some matplotlib releases use names that their parser library has marked as deprecated, and that library
    # then warns while matplotlib is imported. Such a warning says nothing about this package. Without this
    # block a caller that turns warnings into errors could not import the report at all.
    tables.ignore_library_deprecations()
    import matplotlib                               # noqa: F401
    import matplotlib.animation                     # noqa: F401
    import matplotlib.backends.backend_agg          # noqa: F401
    import matplotlib.backends.backend_pdf          # noqa: F401
    import matplotlib.figure                        # noqa: F401
    import PIL.Image                                # noqa: F401

from . import anim, plots, trial                    # noqa: E402
from .tables import Skip                            # noqa: E402

__all__ = ["build_report", "Skip"]


def _previous_files(results: str) -> List[str]:
    """Files that the last report wrote into this results folder (from its ``summary.json``)."""
    path = os.path.join(results, "summary.json")
    try:
        with open(path) as fh:
            listed = json.load(fh).get("files", [])
    except (OSError, ValueError, AttributeError):
        return []
    return [f for f in listed if isinstance(f, str)]


def _remove_outdated(results: str, previous: List[str], current: List[str]) -> List[str]:
    """Delete files of the previous report that this report did not write again. A file that stayed would
    look like a current result. Only files inside the results folder that the previous report listed are removed."""
    removed = []
    root = os.path.realpath(results)
    for rel in previous:
        if rel in current or os.path.isabs(rel):
            continue
        full = os.path.realpath(os.path.join(results, rel))
        if not full.startswith(root + os.sep) or not os.path.isfile(full):
            continue
        os.remove(full)
        removed.append(rel)
    return removed


def build_report(project_dir: str, engine, title: Optional[str] = None, gifs: bool = True,
                 protocols: Optional[dict] = None) -> List[str]:
    """Write the results folder of a project and return the list of written files (absolute paths).

    Parameters
    ----------
    project_dir   the project folder (it holds ``work/``; ``results/`` is created).
    engine        an engine object. Only ``engine.read(key)`` and ``engine.cost(key)`` are called.
    title         name of the project in titles (default: the name in ``work/project.json``).
    gifs          False leaves out the animations (they take most of the time).
    protocols     optional ``{target name: protocol object}``. Needed only if a search used another rule for
                  the start-up blocks than the default of its protocol (see ``tables.DropRule``).

    Raises FileNotFoundError if the folder holds no project. Every other problem is written into the summary.
    """
    data = tables.collect(project_dir, engine, title=title, protocols=protocols)
    results = data.results_dir
    folders = {name: os.path.join(results, name) for name in ("geometries", "plots", "gifs")}
    for folder in [results] + list(folders.values()):
        os.makedirs(folder, exist_ok=True)
    previous = _previous_files(results)
    written: List[str] = []
    skipped: List[Tuple[str, str]] = []
    notes: List[str] = []

    def rel(path: str) -> str:
        return os.path.relpath(path, results)

    def attempt(item: str, action: Callable[[], List[str]]) -> None:
        """Run one item. A failure of the item is recorded with its reason and the report goes on."""
        try:
            written.extend(action())
        except Skip as why:
            skipped.append((item, str(why)))
        except Exception as exc:
            skipped.append((item, "failed with %s: %s" % (type(exc).__name__, exc)))

    def figure(item: str, stem: str, make: Callable[[], object]) -> None:
        def action() -> List[str]:
            out = []
            with plots.style():
                figs = make()
                for i, fig in enumerate(figs if isinstance(figs, list) else [figs]):
                    files, problems = plots.save_figure(fig, stem if i == 0 else "%s_%d" % (stem, i + 1))
                    out += files
                    notes.extend("%s: %s" % (rel(files[0]), p) for p in problems)
            return out
        attempt(item, action)

    def animation(item: str, path: str, make: Callable[[str], dict]) -> None:
        def action() -> List[str]:
            with plots.style():
                info = make(path)
            notes.extend("%s: %s" % (rel(path), p) for p in info["problems"])
            return [path]
        attempt(item, action)

    # ---- geometries and tables
    def geometries() -> List[str]:
        files, missed = tables.write_geometries(data, folders["geometries"])
        skipped.extend((rel(f), why) for f, why in missed)
        return files
    attempt("geometries", geometries)

    def bond_table() -> List[str]:
        path = os.path.join(folders["geometries"], "bond_table.csv")
        tables.write_bond_table(data, path)
        return [path]
    attempt(os.path.join("geometries", "bond_table.csv"), bond_table)

    def path_table() -> List[str]:
        path = os.path.join(folders["geometries"], "path_internal_coordinates.csv")
        tables.write_path_internal_csv(data, path)
        return [path]
    attempt(os.path.join("geometries", "path_internal_coordinates.csv"), path_table)

    def energies() -> List[str]:
        path = os.path.join(results, "energies.csv")
        tables.write_energies_csv(data, path)
        return [path]
    attempt("energies.csv", energies)

    if trial.wanted(data):                          # only a project whose state holds the stage "trial" gets these two files
        attempt("trial_wavefunctions", lambda: trial.write(data, engine, results))

    # ---- plots
    stem = lambda name: os.path.join(folders["plots"], name)          # noqa: E731
    figure("plots/pca_variance", stem("pca_variance"), lambda: plots.plot_pca_variance(data))
    figure("plots/path_profile", stem("path_profile"), lambda: plots.plot_path_profile(data))
    figure("plots/path_geometry", stem("path_geometry"), lambda: plots.plot_path_geometry(data))
    figure("plots/path_geometry_map", stem("path_geometry_map"), lambda: plots.plot_path_geometry_map(data))
    for name in data.targets:
        figure("plots/%s_convergence" % name, stem("%s_convergence" % name), lambda n=name: plots.plot_convergence(data, n))
        figure("plots/%s_scans" % name, stem("%s_scans" % name), lambda n=name: plots.plot_scans(data, n))
        figure("plots/%s_geometry" % name, stem("%s_geometry" % name), lambda n=name: plots.plot_geometry(data, n))

    # ---- animations
    if gifs:
        animation("gifs/path.gif", os.path.join(folders["gifs"], "path.gif"), lambda p: anim.write_path_gif(data, p))
        for name in data.targets:
            animation("gifs/%s_optimization.gif" % name, os.path.join(folders["gifs"], "%s_optimization.gif" % name),
                      lambda p, n=name: anim.write_optimization_gif(data, n, p))
    else:
        skipped.append(("gifs", "the animations were switched off for this report"))

    # ---- summary (written last: it lists the files and what was skipped)
    summary = tables.summarize(data)
    summary_files = [os.path.join(results, "summary.md"), os.path.join(results, "summary.json")]
    summary["files"] = [rel(f) for f in summary_files + written]
    summary["skipped"] = [{"item": item, "reason": why} for item, why in skipped]
    summary["warnings"] = list(notes)
    removed = _remove_outdated(results, previous, summary["files"])
    summary["removed_outdated_files"] = removed
    tables.write_summary_md(summary, summary_files[0])
    tables.write_summary_json(summary, summary_files[1])
    return summary_files + written
