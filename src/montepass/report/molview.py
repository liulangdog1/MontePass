"""Two-dimensional ball-and-stick drawings of a molecule (matplotlib only).

A drawing has three parts:

* a :class:`View`: one fixed orthographic projection, fitted to all geometries that
  will be shown (the frames of an animation). Every frame uses the same view, so
  that only the atoms move.
* atoms as circles in the element colour, with a label ``<element><number>``. The
  number is the position of the atom in the user's file, counted from 1.
* bonds as lines, from :func:`montepass.geometry.bond_list`.

The view is the best-fit plane of the atoms. If more than half of the motion
between the frames is perpendicular to that plane, the motion would be invisible
in that view. :meth:`View.fit_views` then gives two views with no oblique angle:
the plane seen from above and the molecule seen from the side, along the plane.
(:meth:`View.fit` gives one view that is tilted by 60 degrees instead. In such a
view equal bonds look unequal, so the figures of the report do not use it.)

Every view is a rotation of the molecule followed by a projection: no view shows
a mirror image, and two atoms are drawn at their true distance when the line
between them lies in the drawing plane.

Lengths are in angstrom. One unit of the axes is one angstrom in both directions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

from ..geometry import COVALENT_RADIUS, bond_list

# Element colours of the Jmol colour table (a common convention; the labels carry the same information).
ELEMENT_COLOR = {
    "H": "#FFFFFF", "He": "#D9FFFF", "Li": "#CC80FF", "Be": "#C2FF00", "B": "#FFB5B5", "C": "#909090",
    "N": "#3050F8", "O": "#FF0D0D", "F": "#90E050", "Ne": "#B3E3F5", "Na": "#AB5CF2", "Mg": "#8AFF00",
    "Al": "#BFA6A6", "Si": "#F0C8A0", "P": "#FF8000", "S": "#FFFF30", "Cl": "#1FF01F", "Ar": "#80D1E3",
    "K": "#8F40D4", "Ca": "#3DFF00", "Sc": "#E6E6E6", "Ti": "#BFC2C7", "V": "#A6A6AB", "Cr": "#8A99C7",
    "Mn": "#9C7AC7", "Fe": "#E06633", "Co": "#F090A0", "Ni": "#50D050", "Cu": "#C88033", "Zn": "#7D80B0",
    "Ga": "#C28F8F", "Ge": "#668F8F", "As": "#BD80E3", "Se": "#FFA100", "Br": "#A62929", "Kr": "#5CB8D1",
}
UNKNOWN_ELEMENT_COLOR = "#C0C0C0"
EDGE_COLOR = "#000000"          # outline of an atom
BOND_COLOR = "#4D4D4D"
GHOST_COLOR = "#A0A0A0"         # outline drawing of a reference geometry

LABEL_PT = 7.0                  # size of the atom labels in points
ATOM_EDGE_PT = 0.8              # line widths in points
BOND_PT = 2.6
BOND_OUTLINE_PT = 4.4           # a bond that carries a colour: dark line of this width ...
BOND_FILL_PT = 2.8              # ... with the coloured line of this width on top
GHOST_PT = 0.9
TILT_DEGREES = 60.0             # tilt of the view when the motion is perpendicular to the best-fit plane
MARGIN = 0.45                   # free space around the molecule in angstrom (room for labels next to small atoms)


def atom_radius(symbol: str) -> float:
    """Radius of the circle of an atom in angstrom: 0.42 times the covalent radius plus 0.07."""
    return 0.42 * COVALENT_RADIUS.get(symbol, 0.75) + 0.07


def atom_labels(symbols: Sequence[str]) -> List[str]:
    """Labels ``N1, H2, H3, ...``: element symbol and the number of the atom in the user's order, from 1."""
    return ["%s%d" % (s, i + 1) for i, s in enumerate(symbols)]


def _fix_sign(v: np.ndarray) -> np.ndarray:
    """Make the largest component positive, so that the same molecule always gets the same view."""
    k = int(np.argmax(np.abs(v)))
    return -v if v[k] < 0 else v


def _is_dark(color: str) -> bool:
    """True if black text on this colour would be hard to read (relative luminance below 0.35)."""
    rgb = [int(color[i:i + 2], 16) / 255.0 for i in (1, 3, 5)]
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2] < 0.35


@dataclass
class View:
    """A fixed orthographic projection of a molecule on the drawing plane."""

    centre: np.ndarray                  # (3,) point that is drawn at the origin
    axes: np.ndarray                    # (3, 3): rows are the screen x axis, the screen y axis, and the axis to the viewer
    tilt: float                         # degrees between the viewing direction and the normal of the best-fit plane
    xlim: Tuple[float, float]
    ylim: Tuple[float, float]
    name: str = ""                      # what the view shows, when a drawing has more than one view

    def project(self, coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Screen positions (n_atoms, 2) and depths (n_atoms,) of a geometry. A larger depth is nearer to the viewer."""
        X = np.dot(np.asarray(coords, dtype=float).reshape(-1, 3) - self.centre, self.axes.T)
        return X[:, :2], X[:, 2]

    def apply(self, ax) -> None:
        """Give an axes the limits of this view, an equal scale on both axes, and no frame.

        The limits stay exactly as given here: the box of the axes is made smaller to fit them (and kept
        in the middle of its place in the figure). So every frame of an animation shows the same region."""
        ax.set_autoscale_on(False)
        ax.set_xlim(*self.xlim)
        ax.set_ylim(*self.ylim)
        ax.set_aspect("equal", adjustable="box", anchor="C")
        ax.set_axis_off()

    def describe(self) -> str:
        """One short sentence about the view, for a figure."""
        if self.name:
            return self.name
        if self.tilt == 0.0:
            return "view: best-fit plane of the atoms"
        return "view: tilted %.0f° out of the best-fit plane" % self.tilt

    @staticmethod
    def fit_views(symbols: Sequence[str], frames: Sequence[np.ndarray], margin: float = MARGIN) -> List["View"]:
        """The views for a set of geometries of one molecule, with no oblique angle.

        One view, the best-fit plane of all atom positions, if the motion between the frames can be seen in
        that plane (or if there is one frame). Two views if more than half of the squared displacement
        between the frames is perpendicular to the plane: the plane seen from above, and the molecule seen
        from the side, along the plane. In the side view the normal of the plane points up, and the viewing
        direction in the plane is the one that keeps the atoms farthest apart on the screen.
        """
        F = [np.asarray(f, dtype=float).reshape(-1, 3) for f in frames]
        if not F:
            raise ValueError("View.fit_views needs at least one geometry")
        n = len(F[0])
        if n != len(symbols) or any(len(f) != n for f in F):
            raise ValueError("View.fit_views: the frames and the symbols do not have the same number of atoms")
        P = np.concatenate(F)
        centre = P.mean(axis=0)
        _u, _s, vt = np.linalg.svd(P - centre, full_matrices=True)
        a1, a2 = _fix_sign(vt[0]), _fix_sign(vt[1])
        a3 = np.cross(a1, a2)
        radii = np.array([atom_radius(s) for s in symbols])
        rr = np.tile(radii, len(F))

        def view(axes: np.ndarray, tilt: float, name: str) -> "View":
            xy = np.concatenate([np.dot(f - centre, axes[:2].T) for f in F])
            xlim = (float((xy[:, 0] - rr).min() - margin), float((xy[:, 0] + rr).max() + margin))
            ylim = (float((xy[:, 1] - rr).min() - margin), float((xy[:, 1] + rr).max() + margin))
            return View(centre=centre, axes=axes, tilt=tilt, xlim=xlim, ylim=ylim, name=name)

        out_of_plane = False
        if len(F) > 1 and n > 1:
            D = np.array(F) - np.mean(F, axis=0)
            total = float((D ** 2).sum())
            out_of_plane = total > 1e-10 and float((np.dot(D, a3) ** 2).sum()) > 0.5 * total
        if not out_of_plane:
            return [view(np.array([a1, a2, a3]), 0.0, "")]
        side = View._tilted_axes(a1, a2, a3, np.radians(90.0), F, centre, radii)
        return [view(np.array([a1, a2, a3]), 0.0, "seen from above the plane of the atoms"),
                view(side, 90.0, "seen from the side, along that plane")]

    @staticmethod
    def fit(symbols: Sequence[str], frames: Sequence[np.ndarray], margin: float = MARGIN) -> "View":
        """The view for a set of geometries of one molecule (same atom order in every frame).

        The drawing plane is the best-fit plane of all atom positions of all frames; the direction of
        the largest extent is horizontal. If more than half of the squared displacement between the
        frames is perpendicular to that plane, the viewing direction is tilted by ``TILT_DEGREES``
        towards the plane. The tilt direction is the one that keeps the atoms farthest apart on the screen.
        """
        F = [np.asarray(f, dtype=float).reshape(-1, 3) for f in frames]
        if not F:
            raise ValueError("View.fit needs at least one geometry")
        n = len(F[0])
        if n != len(symbols) or any(len(f) != n for f in F):
            raise ValueError("View.fit: the frames and the symbols do not have the same number of atoms")
        P = np.concatenate(F)
        centre = P.mean(axis=0)
        _u, _s, vt = np.linalg.svd(P - centre, full_matrices=True)
        a1, a2 = _fix_sign(vt[0]), _fix_sign(vt[1])
        a3 = np.cross(a1, a2)
        axes, tilt = np.array([a1, a2, a3]), 0.0
        radii = np.array([atom_radius(s) for s in symbols])
        if len(F) > 1 and n > 1:
            D = np.array(F) - np.mean(F, axis=0)
            total = float((D ** 2).sum())
            hidden = float((np.dot(D, a3) ** 2).sum())
            if total > 1e-10 and hidden > 0.5 * total:
                tilt = TILT_DEGREES
                axes = View._tilted_axes(a1, a2, a3, np.radians(tilt), F, centre, radii)
        xy = np.concatenate([np.dot(f - centre, axes[:2].T) for f in F])
        rr = np.tile(radii, len(F))
        xlim = (float((xy[:, 0] - rr).min() - margin), float((xy[:, 0] + rr).max() + margin))
        ylim = (float((xy[:, 1] - rr).min() - margin), float((xy[:, 1] + rr).max() + margin))
        return View(centre=centre, axes=axes, tilt=tilt, xlim=xlim, ylim=ylim)

    @staticmethod
    def _tilted_axes(a1, a2, a3, theta, frames, centre, radii) -> np.ndarray:
        """Screen axes of the tilted view. The tilt direction in the plane is scanned in steps of 5 degrees; the
        one with the largest smallest distance between two atoms on the screen (in units of the sum of their
        radii) is taken."""
        sample = frames[:: max(1, len(frames) // 12)]
        iu = np.triu_indices(len(radii), k=1)
        rsum = (radii[:, None] + radii[None, :])[iu]
        best_axes, best_score = None, -1.0
        for phi in np.radians(np.arange(0.0, 360.0, 5.0)):
            u = np.cos(phi) * a1 + np.sin(phi) * a2             # in-plane direction of the tilt
            h = -np.sin(phi) * a1 + np.cos(phi) * a2            # screen x: in the plane, perpendicular to u
            w = np.cos(theta) * a3 + np.sin(theta) * u          # to the viewer
            v = np.cross(w, h)                                  # screen y
            score = np.inf
            for f in sample:
                xy = np.dot(f - centre, np.array([h, v]).T)
                d = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=2)[iu]
                score = min(score, float((d / rsum).min()))
            if score > best_score + 1e-9:
                best_axes, best_score = np.array([h, v, w]), score
        return best_axes


class MoleculeArtist:
    """The matplotlib objects of one drawn molecule. :meth:`update` moves them to a new geometry."""

    def __init__(self, ax, symbols: Sequence[str], view: View, pairs: Sequence[Tuple[int, int]],
                 labels: bool = True, ghost: bool = False, fontsize: float = LABEL_PT, zorder: float = 3.0,
                 colored_bonds: bool = False):
        self.ax = ax
        self.symbols = list(symbols)
        self.view = view
        self.pairs = [(int(i), int(j)) for i, j in pairs]
        self.ghost = bool(ghost)
        self.zorder = float(zorder)
        self.radii = np.array([atom_radius(s) for s in self.symbols])
        self.circles: List[Circle] = []
        self.lines: Dict[Tuple[int, int], Line2D] = {}
        self.texts: List[Optional[object]] = []
        self._outside: List[bool] = []
        self.fontsize = float(fontsize)
        # clip_on=False: an atom at the edge of the view is drawn whole, never cut by the box of the axes
        for s, r in zip(self.symbols, self.radii):
            if self.ghost:
                c = Circle((0.0, 0.0), r, facecolor="none", edgecolor=GHOST_COLOR, linewidth=GHOST_PT, linestyle="--",
                           clip_on=False)
            else:
                c = Circle((0.0, 0.0), r, facecolor=ELEMENT_COLOR.get(s, UNKNOWN_ELEMENT_COLOR), edgecolor=EDGE_COLOR,
                           linewidth=ATOM_EDGE_PT, clip_on=False)
            ax.add_patch(c)
            self.circles.append(c)
        # A bond can carry a colour (for example its change of length along a path). The colour is a second,
        # narrower line on top of a dark one: a bond whose colour is close to white stays visible.
        self.colored = bool(colored_bonds) and not self.ghost
        self.fills: Dict[Tuple[int, int], Line2D] = {}
        for pair in self.pairs:
            width = GHOST_PT if self.ghost else (BOND_OUTLINE_PT if self.colored else BOND_PT)
            line = Line2D([0.0, 0.0], [0.0, 0.0], color=GHOST_COLOR if self.ghost else BOND_COLOR,
                          linewidth=width, solid_capstyle="butt", dash_capstyle="butt",
                          linestyle="--" if self.ghost else "-", clip_on=False)
            ax.add_line(line)
            self.lines[pair] = line
            if self.colored:
                fill = Line2D([0.0, 0.0], [0.0, 0.0], color="#FFFFFF", linewidth=BOND_FILL_PT, solid_capstyle="butt",
                              clip_on=False)
                ax.add_line(fill)
                self.fills[pair] = fill
        self._want_labels = bool(labels) and not self.ghost

    # ------------------------------------------------------------------ labels
    def _make_labels(self, xy: np.ndarray) -> None:
        """Create the atom labels. A label is written inside its circle if it fits there; else next to the atom,
        on the side away from its bonded neighbours. The figure must have been drawn once before, so that the
        size of the axes on the page is known."""
        fig = self.ax.figure
        renderer = fig.canvas.get_renderer()
        px_per_angstrom = float(abs(self.ax.transData.transform((1.0, 0.0))[0] - self.ax.transData.transform((0.0, 0.0))[0]))
        px_per_point = fig.dpi / 72.0
        neighbours: Dict[int, List[int]] = {}
        for i, j in self.pairs:
            neighbours.setdefault(i, []).append(j)
            neighbours.setdefault(j, []).append(i)
        middle = xy.mean(axis=0)
        for i, text in enumerate(atom_labels(self.symbols)):
            face = ELEMENT_COLOR.get(self.symbols[i], UNKNOWN_ELEMENT_COLOR)
            t = self.ax.text(xy[i, 0], xy[i, 1], text, ha="center", va="center", fontsize=self.fontsize,
                             color="#FFFFFF" if _is_dark(face) else "#000000", clip_on=False)
            box = t.get_window_extent(renderer)
            half_diagonal = 0.5 * float(np.hypot(box.width, box.height))
            r_px = self.radii[i] * px_per_angstrom
            outside = half_diagonal > 0.98 * r_px
            if outside:
                t.remove()
                away = xy[i] - (xy[neighbours[i]].mean(axis=0) if i in neighbours else middle)
                norm = float(np.linalg.norm(away))
                away = away / norm if norm > 1e-9 else np.array([0.0, 1.0])
                # distance of the label centre from the atom centre, in points: the radius of the circle, half the
                # size of the label box along this direction, and 1.5 points of space
                half = 0.5 * (abs(away[0]) * box.width + abs(away[1]) * box.height) / px_per_point
                reach = r_px / px_per_point + half + 1.5
                t = self.ax.annotate(text, xy=(xy[i, 0], xy[i, 1]), xytext=(away[0] * reach, away[1] * reach),
                                     textcoords="offset points", ha="center", va="center", fontsize=self.fontsize,
                                     color="#000000", annotation_clip=False)
            t._atom_label = True                 # read by plots.text_problems: atom labels may hide each other
            self.texts.append(t)
            self._outside.append(outside)

    # ------------------------------------------------------------------ geometry
    def update(self, coords: np.ndarray, bonds: Optional[Sequence[Tuple[int, int]]] = None,
               bond_colors: Optional[Dict[Tuple[int, int], object]] = None) -> None:
        """Move the drawing to a geometry. ``bonds`` are the bonds to show (default: all pairs of this drawing).
        ``bond_colors`` gives a colour to bonds of a drawing that was made with ``colored_bonds=True``."""
        xy, depth = self.view.project(coords)
        n = len(self.symbols)
        rank = np.empty(n)
        rank[np.argsort(depth, kind="stable")] = np.arange(n)
        z_atom = self.zorder + rank / (n + 1.0)
        if self._want_labels and not self.texts:
            self._make_labels(xy)
        for i, c in enumerate(self.circles):
            c.center = (float(xy[i, 0]), float(xy[i, 1]))
            c.set_zorder(z_atom[i])
        shown = None if bonds is None else {(min(i, j), max(i, j)) for i, j in bonds}
        for (i, j), line in self.lines.items():
            # a bond is drawn from the edge of one circle to the edge of the other one, so that no line
            # runs under an atom label. If the two circles touch on the screen, nothing is left to draw.
            step = xy[j] - xy[i]
            length = float(np.linalg.norm(step))
            free = length - self.radii[i] - self.radii[j]
            if free > 0.0:
                a = xy[i] + step * (self.radii[i] / length)
                b = xy[j] - step * (self.radii[j] / length)
                line.set_data([a[0], b[0]], [a[1], b[1]])
            line.set_zorder(min(z_atom[i], z_atom[j]) - 0.4 / (n + 1.0))
            visible = free > 0.0 and (shown is None or (min(i, j), max(i, j)) in shown)
            line.set_visible(visible)
            fill = self.fills.get((i, j))
            if fill is not None:
                colour = None if bond_colors is None else bond_colors.get((i, j), bond_colors.get((j, i)))
                if free > 0.0:
                    # the coloured line is a little shorter, so that the dark line closes it at both ends
                    cut = min(0.03, 0.2 * free)
                    a2, b2 = a + step * (cut / length), b - step * (cut / length)
                    fill.set_data([a2[0], b2[0]], [a2[1], b2[1]])
                if colour is not None:
                    fill.set_color(colour)
                fill.set_zorder(min(z_atom[i], z_atom[j]) - 0.3 / (n + 1.0))
                fill.set_visible(visible and colour is not None)
        for i, t in enumerate(self.texts):
            if self._outside[i]:
                t.xy = (float(xy[i, 0]), float(xy[i, 1]))
            else:
                t.set_position((float(xy[i, 0]), float(xy[i, 1])))
            t.set_zorder(z_atom[i] + 0.2 / (n + 1.0))

    def artists(self) -> list:
        return (list(self.circles) + list(self.lines.values()) + list(self.fills.values())
                + [t for t in self.texts if t is not None])


class MoleculePanel:
    """One molecule drawn in one or two views (:meth:`View.fit_views`) inside one cell of a figure layout.

    ``cell`` is a ``SubplotSpec`` (for example ``gridspec[0, 0]``). Two views are put one above the other,
    each with a line that says what it shows. ``ghost`` is an optional reference geometry that is drawn
    behind the molecule as a dashed outline. :meth:`update` moves the molecule in every view.
    The figure is drawn once inside the constructor (see :func:`draw_molecule`): add the other content first.
    """

    def __init__(self, fig, cell, symbols: Sequence[str], frames: Sequence[np.ndarray],
                 pairs: Sequence[Tuple[int, int]], first: Optional[np.ndarray] = None,
                 bonds: Optional[Sequence[Tuple[int, int]]] = None, ghost: Optional[np.ndarray] = None,
                 fontsize: float = LABEL_PT, colored_bonds: bool = False,
                 bond_colors: Optional[Dict[Tuple[int, int], object]] = None):
        self.views = View.fit_views(symbols, frames)
        if len(self.views) > 1:
            # the same scale in both views: the same width of the shown region, and heights of the two axes in
            # the ratio of the shown heights. An atom then has the same size in both drawings.
            half = 0.5 * max(v.xlim[1] - v.xlim[0] for v in self.views)
            for v in self.views:
                middle = 0.5 * (v.xlim[0] + v.xlim[1])
                v.xlim = (middle - half, middle + half)
        grid = cell.subgridspec(len(self.views), 1, height_ratios=[v.ylim[1] - v.ylim[0] for v in self.views])
        self.axes = [fig.add_subplot(grid[i, 0]) for i in range(len(self.views))]
        self.notes = []
        start = np.asarray(frames[0] if first is None else first, dtype=float).reshape(-1, 3)
        self.arts: List[MoleculeArtist] = []
        for ax, view in zip(self.axes, self.views):
            if view.name:
                self.notes.append(ax.text(0.5, -0.01, view.name, transform=ax.transAxes, ha="center", va="top",
                                          fontsize=fontsize))
        for ax, view in zip(self.axes, self.views):
            if ghost is not None:
                draw_molecule(ax, symbols, ghost, view=view, pairs=pairs, labels=False, ghost=True, zorder=1.0)
            self.arts.append(draw_molecule(ax, symbols, start, view=view, pairs=pairs, bonds=bonds, fontsize=fontsize,
                                           colored_bonds=colored_bonds, bond_colors=bond_colors))

    @property
    def two_views(self) -> bool:
        return len(self.views) > 1

    def update(self, coords: np.ndarray, bonds: Optional[Sequence[Tuple[int, int]]] = None,
               bond_colors: Optional[Dict[Tuple[int, int], object]] = None) -> None:
        for art in self.arts:
            art.update(coords, bonds=bonds, bond_colors=bond_colors)


def union_bonds(symbols: Sequence[str], frames: Sequence[np.ndarray]) -> List[Tuple[int, int]]:
    """Every pair of atoms that is bonded in at least one of the geometries, sorted."""
    pairs = set()
    for f in frames:
        pairs.update(bond_list(symbols, f))
    return sorted(pairs)


def draw_molecule(ax, symbols: Sequence[str], coords: np.ndarray, view: Optional[View] = None,
                  pairs: Optional[Sequence[Tuple[int, int]]] = None, bonds: Optional[Sequence[Tuple[int, int]]] = None,
                  labels: bool = True, ghost: bool = False, fontsize: float = LABEL_PT,
                  zorder: float = 3.0, colored_bonds: bool = False,
                  bond_colors: Optional[Dict[Tuple[int, int], object]] = None) -> MoleculeArtist:
    """Draw one geometry in an axes and return the drawing.

    ``view`` is the projection (default: fitted to this geometry). ``pairs`` are the atom pairs that can be
    bonded in this drawing (default: the bonds of this geometry); ``bonds`` are the pairs shown now (default:
    all of ``pairs``). ``ghost=True`` draws grey dashed outlines without labels; it is used for a reference
    geometry behind the moving one.

    The figure is drawn once inside this function, because the size of an atom on the page decides whether
    its label fits inside the circle. Add the other content of the figure first.
    """
    X = np.asarray(coords, dtype=float).reshape(-1, 3)
    if view is None:
        view = View.fit(symbols, [X])
    view.apply(ax)
    if pairs is None:
        pairs = bond_list(symbols, X)
    art = MoleculeArtist(ax, symbols, view, pairs, labels=labels, ghost=ghost, fontsize=fontsize, zorder=zorder,
                         colored_bonds=colored_bonds)
    if art._want_labels:
        ax.figure.canvas.draw()             # fixes the layout: the scale of the axes is known after this
    art.update(X, bonds=bonds, bond_colors=bond_colors)
    return art


def bond_change_colors(pairs: Sequence[Tuple[int, int]], coords: np.ndarray, reference: np.ndarray, norm, cmap) -> Dict[Tuple[int, int], object]:
    """Colour of every bond for its change of length: ``length in coords - length in reference`` (angstrom)
    through the colour scale ``norm`` and the colour map ``cmap`` of matplotlib."""
    X = np.asarray(coords, dtype=float).reshape(-1, 3)
    X0 = np.asarray(reference, dtype=float).reshape(-1, 3)
    out = {}
    for i, j in pairs:
        change = float(np.linalg.norm(X[i] - X[j]) - np.linalg.norm(X0[i] - X0[j]))
        out[(int(i), int(j))] = cmap(norm(change))
    return out
