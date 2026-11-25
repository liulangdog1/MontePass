"""The animations of the results folder: GIF files made with matplotlib and Pillow only.

Two animations exist:

* ``path.gif``: the molecule moves along the images of the path. Next to it: the surrogate energy profile
  (if the path file holds the energies) with a ring on the current image, and the bond lengths and angles
  that change most along the path with a line at the current image. Under the molecule the values of
  these bond lengths and angles at the current image are printed.
* ``<target>_optimization.gif``: the molecule moves from the starting geometry of a search through the
  centres of its iterations to the final geometry. The starting geometry stays visible as a dashed outline.
  If the largest displacement of an atom is below 0.15 angstrom it is hard to see; then all displacements
  from the start are multiplied by one factor, and the frame says which factor.

Every frame uses the same views of the molecule (``molview.View.fit_views``: the best-fit plane of the atoms
and, if the motion leaves that plane, a second view from the side) and the same axes, so only the atoms,
the ring, the line and the printed values change. A file must stay below 3 MB: if it does not, it is written again with fewer dots per inch
and fewer frames, and an error is raised if it is still too large.
"""
from __future__ import annotations

import os
import re
from typing import Callable, List, Sequence

import numpy as np
from matplotlib.animation import PillowWriter
from PIL import Image

from .. import geometry as G
from ..units import HARTREE_TO_KCAL
from . import molview
from .plots import (COORD_STYLES, FS, NARROW, PALETTE, WIDE, _kcal, _short, _title, coordinate_label, coordinate_lines,
                    freeze_layout, new_figure, recorded_warnings, text_problems, LINE_MERGE, bond_change_scale,
                    change_colorbar)
from .tables import ReportData, Skip, coordinate_groups, path_internal

FPS = 10                                    # frames per second
MAX_BYTES = 3000000                         # largest allowed size of one GIF file (3 MB)
ATTEMPTS = ((120, 1), (96, 2), (72, 3))     # (dots per inch, keep every n-th frame), tried in this order
HOLD_START, HOLD_END = 3, 8                 # the first and the last frame are repeated this many times
VISIBLE = 0.15                              # angstrom: a smaller largest displacement is magnified
NICE_FACTORS = (2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000)
FRAMES_WANTED = 36                          # about this many moving frames per animation
RING = dict(linestyle="none", marker="o", markersize=11.0, markerfacecolor="none", markeredgecolor="black",
            markeredgewidth=1.4)


def _write_gif(fig, positions: Sequence[float], update: Callable[[float], None], out_path: str) -> dict:
    """Draw one frame per entry of ``positions`` (``update(position)`` moves the artists) and write a GIF.

    Returns a dict with the number of frames in the file, its size in bytes and the dots per inch used.
    Raises RuntimeError if the file is not below ``MAX_BYTES`` at the smallest setting."""
    if not out_path.endswith(".gif"):
        raise ValueError("the name of an animation must end in .gif: %s" % out_path)
    tmp = out_path[:-4] + ".tmp.gif"            # Pillow chooses the file format from the ending
    size = 0
    notes: List[str] = []
    for dpi, stride in ATTEMPTS:
        seq = list(positions[::stride])
        if seq[-1] != positions[-1]:
            seq.append(positions[-1])
        seq = [seq[0]] * HOLD_START + seq + [seq[-1]] * HOLD_END
        notes = []
        with recorded_warnings(notes):
            writer = PillowWriter(fps=FPS)
            with writer.saving(fig, tmp, dpi=dpi):
                for pos in seq:
                    update(pos)
                    writer.grab_frame()
        size = os.path.getsize(tmp)
        if size < MAX_BYTES:
            with Image.open(tmp) as im:
                n_frames = int(getattr(im, "n_frames", 1))
            if n_frames < 2:
                os.remove(tmp)
                raise RuntimeError("%s would hold only %d frame: nothing moves" % (os.path.basename(out_path), n_frames))
            os.replace(tmp, out_path)
            return {"file": out_path, "frames": n_frames, "bytes": int(size), "dpi": dpi, "problems": notes}
    os.remove(tmp)
    raise RuntimeError("%s would be %d bytes at %d dots per inch; the limit is %d bytes"
                       % (os.path.basename(out_path), size, ATTEMPTS[-1][0], MAX_BYTES))


def _note_below(ax, text: str):
    """A note under a molecule panel (the axes of a molecule has no axis labels)."""
    return ax.text(0.5, -0.01, text, transform=ax.transAxes, ha="center", va="top", fontsize=FS.ANNOT)


# ----------------------------------------------------------------------------- the path
PATH_PANEL = 1.55           # inches: height of one panel on the right side of the path animation
PATH_LINES = 3              # at most this many bond lines and this many angle lines
BAR_ROW = 0.21              # height of the row of the colour bar of the bonds, relative to the molecule drawing


def path_figure_height(panels: int) -> float:
    """Height in inches of the figure of the path animation with ``panels`` panels on the right."""
    return max(3.0, 0.55 + PATH_PANEL * panels)


def write_path_gif(data: ReportData, out_path: str) -> dict:
    """The molecule moving along the path images. Between two images the geometry is interpolated linearly,
    so that the motion is smooth; the ring and the line move with it, and the printed bond lengths and
    angles are those of the interpolated geometry."""
    p = data.path
    if p is None:
        raise Skip("the project has no path")
    frames, n = p.frames, len(p.frames)
    if n < 2:
        raise Skip("the path has only one image")
    if n - 1 >= FRAMES_WANTED:
        positions = [float(i) for i in np.unique(np.round(np.linspace(0, n - 1, FRAMES_WANTED + 12)).astype(int))]
    else:
        sub = int(np.ceil(FRAMES_WANTED / float(n - 1)))
        positions = [i / float(sub) for i in range((n - 1) * sub + 1)]

    def geometry(s: float) -> np.ndarray:
        i = min(int(np.floor(s)), n - 2)
        f = s - i
        return (1.0 - f) * frames[i] + f * frames[i + 1]

    internal = path_internal(p)
    bond_groups, all_bonds = coordinate_groups(internal, "bond", PATH_LINES, merge=LINE_MERGE)
    angle_groups, all_angles = coordinate_groups(internal, "angle", PATH_LINES, merge=LINE_MERGE)
    pairs = molview.union_bonds(p.symbols, frames)
    # every bond of the drawing has the colour of its change of length from image 0
    norm, cmap, largest = bond_change_scale(p.symbols, frames, pairs)
    kinds = (["energy"] if p.energies is not None else []) + (["bond"] if bond_groups else []) \
        + (["angle"] if angle_groups else [])
    fig = new_figure(WIDE if kinds else NARROW, path_figure_height(len(kinds)))
    x = np.arange(n)
    ring, y = None, None
    guides, values = [], []
    if kinds:
        gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.2])
        with_table = bool(bond_groups or angle_groups)
        # left column: the molecule, the colour bar of its bonds (with room for its numbers and its label), the table
        left = gs[0, 0].subgridspec(3 if with_table else 2, 1, height_ratios=[1.0, BAR_ROW, 0.40] if with_table else [1.0, BAR_ROW])
        molecule_cell = left[0, 0]
        bar_cell = left[1, 0].subgridspec(1, 3, width_ratios=[0.14, 0.72, 0.14])[0, 1]
        right = gs[0, 1].subgridspec(len(kinds), 1)
        axes = []
        for r, kind in enumerate(kinds):
            ax = fig.add_subplot(right[r, 0], sharex=axes[0] if axes else None)
            axes.append(ax)
            if kind == "energy":
                y = (p.energies - p.energies[0]) * HARTREE_TO_KCAL
                ax.plot(x, y, color=PALETTE["surrogate"], marker="s" if n <= 40 else None, markersize=3.5, zorder=2)
                ring, = ax.plot([0.0], [y[0]], zorder=4, **RING)
                ax.set_ylabel("surrogate energy minus\nimage 0 (kcal/mol)")
            else:
                groups, total = (bond_groups, all_bonds) if kind == "bond" else (angle_groups, all_angles)
                coordinate_lines(ax, x, groups, markers=n <= 40)
                guides.append(ax.axvline(0.0, color=PALETTE["guide"], linewidth=0.8, zorder=0.5))
                ax.set_ylabel("bond length (Å)" if kind == "bond" else "angle (degrees)")
            ax.margins(x=0.07, y=0.15)
            if r < len(kinds) - 1:
                ax.tick_params(labelbottom=False)
        axes[-1].set_xlabel("path image (numbered from 0)")
        if with_table:
            # the values at the current image. The line samples make this table the legend of the panels too.
            ax_t = fig.add_subplot(left[2, 0])
            ax_t.set_axis_off(); ax_t.set_xlim(0.0, 1.0); ax_t.set_ylim(0.0, 1.0)
            head = "values at this image (lines: panels on the right)"
            if all_bonds > len(bond_groups) or all_angles > len(angle_groups):
                # a molecule with many bonds: the lines are the curves that change most
                head += "\nshown: %d of %d bond curves, %d of %d angle curves" % (
                    len(bond_groups), all_bonds, len(angle_groups), all_angles)
            # The texts of the table do not take part in the layout: a text that is wider than the column would
            # make the layout engine shrink the table. The text check reports a text that does not fit.
            ax_t.text(0.0, 0.97, head, ha="left", va="top", fontsize=FS.ANNOT).set_in_layout(False)
            shown = [(g, "bond", i) for i, g in enumerate(bond_groups)] + [(g, "angle", i) for i, g in enumerate(angle_groups)]
            first_row = 0.66 if "\n" in head else 0.76
            step = first_row / len(shown)
            for row, (g, kind, i) in enumerate(shown):
                yy = first_row - (row + 0.5) * step
                style, marker, filled = COORD_STYLES[i]
                ax_t.plot([0.0, 0.05, 0.10], [yy, yy, yy], color=PALETTE["coordinate"], linestyle=style, linewidth=0.9,
                          marker=marker, markevery=[1], markersize=3.5,
                          markerfacecolor=PALETTE["coordinate"] if filled else "white",
                          markeredgecolor=PALETTE["coordinate"], clip_on=False)
                ax_t.text(0.13, yy, ("bond " if kind == "bond" else "angle ") + coordinate_label(g), ha="left",
                          va="center", fontsize=FS.ANNOT).set_in_layout(False)
                value = ax_t.text(1.0, yy, "", ha="right", va="center", fontsize=FS.ANNOT)
                value.set_in_layout(False)
                values.append((value, g["atoms"], kind))
    else:
        alone = fig.add_gridspec(2, 1, height_ratios=[1.0, BAR_ROW])
        molecule_cell = alone[0, 0]
        bar_cell = alone[1, 0].subgridspec(1, 3, width_ratios=[0.14, 0.72, 0.14])[0, 1]
    change_colorbar(fig, fig.add_subplot(bar_cell), norm, cmap, largest,
                    "colour of a bond: its length minus\nits length at image 0 (Å)", orientation="horizontal")
    fig.suptitle(_short("%s: the path" % data.title))

    def show_values(X: np.ndarray) -> None:
        for text, atoms, kind in values:
            text.set_text("%.3f Å" % G.distance(X, *atoms) if kind == "bond" else "%.1f°" % G.angle(X, *atoms))

    show_values(frames[0])
    panel = molview.MoleculePanel(fig, molecule_cell, p.symbols, frames, pairs, first=frames[0],
                                  bonds=G.bond_list(p.symbols, frames[0]), fontsize=FS.ANNOT, colored_bonds=True,
                                  bond_colors=molview.bond_change_colors(pairs, frames[0], frames[0], norm, cmap))
    title = panel.axes[0].set_title("path image 0 (images 0 to %d)" % (n - 1), loc="left")
    freeze_layout(fig)
    problems = text_problems(fig)

    def update(s: float) -> None:
        X = geometry(s)
        panel.update(X, bonds=G.bond_list(p.symbols, X),
                     bond_colors=molview.bond_change_colors(pairs, X, frames[0], norm, cmap))
        title.set_text("path image %d (images 0 to %d)" % (int(np.floor(s + 0.5)), n - 1))
        if ring is not None:
            ring.set_data([s], [float(np.interp(s, x, y))])
        for line in guides:
            line.set_xdata([s, s])
        show_values(X)

    info = _write_gif(fig, positions, update, out_path)
    info["problems"] = problems + info["problems"]
    return info


# ----------------------------------------------------------------------------- one search
def magnification(largest_displacement: float) -> int:
    """Factor by which displacements are multiplied in the animation of a search: 1 if the largest
    displacement of an atom is at least ``VISIBLE`` angstrom, else the smallest of 2, 5, 10, 20, 50, ...
    that makes it so."""
    if largest_displacement >= VISIBLE:
        return 1
    for f in NICE_FACTORS:
        if f * largest_displacement >= VISIBLE:
            return f
    return NICE_FACTORS[-1]


def optimization_figure_height(views: int) -> float:
    """Height in inches of the figure of the animation of one search, with one or two views of the molecule."""
    return 3.2 if views <= 1 else 4.4


def frame_label(labels: Sequence[str], s: float) -> str:
    """Text for position ``s`` of an animation through the geometries ``labels``: the label of a geometry when
    ``s`` is a whole number, and "<a> to <b>" (short forms) for a frame between two geometries."""
    lo = int(np.floor(s + 1e-9))
    if abs(s - round(s)) < 1e-6 or lo + 1 >= len(labels):
        return labels[min(int(round(s)), len(labels) - 1)]
    short = [re.split(r":| \(", labels[i])[0] for i in (lo, lo + 1)]
    return "%s to %s" % (short[0], short[1])


def write_optimization_gif(data: ReportData, name: str, out_path: str) -> dict:
    """The molecule of one target moving from the starting geometry through the centres of the iterations
    to the final geometry, next to the DMC energies of these geometries."""
    t = data.targets[name]
    if t.symbols is None or not t.iterations:
        raise Skip("target %s has no search iteration" % name)
    centres = [it.centre for it in t.iterations]
    labels = ["iteration %d%s" % (it.number, ": centre of a rejected step" if it.accepted is False else "")
              for it in t.iterations]
    stats = [it.energy for it in t.iterations]
    ticks = ["%d" % it.number for it in t.iterations]
    rejected = [it.accepted is False for it in t.iterations]
    if t.final is not None:
        if float(np.abs(t.final - centres[-1]).max()) > 1e-9:
            centres.append(t.final); labels.append("final geometry"); stats.append(t.final_energy)
            ticks.append("final"); rejected.append(False)
        else:
            labels[-1] = "final geometry (centre of iteration %d)" % t.iterations[-1].number
    start = centres[0]
    largest = max(float(np.linalg.norm(c - start, axis=1).max()) for c in centres)
    if largest < 1e-6:
        raise Skip("the search did not move the geometry of target %s" % name)
    factor = magnification(largest)
    shown = [start + factor * (c - start) for c in centres]
    segments = len(shown) - 1
    sub = int(min(12, max(3, round(FRAMES_WANTED / float(segments)))))
    positions = [i / float(sub) for i in range(segments * sub + 1)]

    def geometry(s: float) -> np.ndarray:
        i = min(int(np.floor(s)), segments - 1)
        f = s - i
        return (1.0 - f) * shown[i] + f * shown[i + 1]

    known = [i for i, st in enumerate(stats) if st is not None]
    with_energy = t.start_energy is not None and len(known) >= 2
    n_views = len(molview.View.fit_views(t.symbols, shown))
    pairs = molview.union_bonds(t.symbols, [centres[0], centres[-1]])      # bonds of the true geometries
    fig = new_figure(WIDE if with_energy else NARROW, optimization_figure_height(n_views))
    ring, xs, ys = None, None, None
    if with_energy:
        gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.2])
        molecule_cell, ax_e = gs[0, 0], fig.add_subplot(gs[0, 1])
        e0 = t.start_energy.value
        vals = {i: _kcal(stats[i], e0) for i in known}
        good = [i for i in known if not rejected[i]]
        bad = [i for i in known if rejected[i]]
        ax_e.axhline(0.0, color=PALETTE["guide"], linewidth=0.6, linestyle=":", zorder=0.5)
        if good:
            ax_e.errorbar(good, [vals[i][0] for i in good], yerr=[vals[i][1] for i in good], color=PALETTE["dmc"],
                          marker="o", linewidth=0.8, zorder=3)
        if bad:
            ax_e.errorbar(bad, [vals[i][0] for i in bad], yerr=[vals[i][1] for i in bad], linestyle="none", marker="X",
                          markersize=5.5, markerfacecolor="white", markeredgecolor=PALETTE["rejected"],
                          ecolor=PALETTE["rejected"], zorder=3)
        xs = np.array(known, dtype=float)           # the ring visits every geometry that has an energy
        ys = np.array([vals[i][0] for i in known])
        ring, = ax_e.plot([xs[0]], [ys[0]], zorder=4, **RING)
        ax_e.set_xticks(range(len(ticks)))
        ax_e.set_xticklabels(ticks)
        ax_e.set_xlim(-0.6, len(ticks) - 0.4)
        ax_e.margins(y=0.15)
        ax_e.set_xlabel("iteration" + ("  (grey cross: centre of a rejected step)" if bad else ""))
        ax_e.set_ylabel("DMC energy minus start\n(kcal/mol; bars: 1 standard error)")
    else:
        molecule_cell = fig.add_gridspec(1, 1)[0, 0]
    size = ("displacements from the start are multiplied by %d" % factor) if factor > 1 else \
        "displacements are shown at their true size"
    _title(fig, "%s: search of target %s" % (data.title, name), more=size + "; dashed outline: starting geometry")
    panel = molview.MoleculePanel(fig, molecule_cell, t.symbols, shown, pairs, first=shown[0], ghost=start,
                                  fontsize=FS.ANNOT)
    title = panel.axes[0].set_title(labels[0], loc="left")
    freeze_layout(fig)
    problems = text_problems(fig)

    def update(s: float) -> None:
        panel.update(geometry(s))
        title.set_text(frame_label(labels, s))
        if ring is not None:
            ring.set_data([s], [float(np.interp(s, xs, ys))])

    info = _write_gif(fig, positions, update, out_path)
    info["factor"] = factor
    info["problems"] = problems + info["problems"]
    return info
