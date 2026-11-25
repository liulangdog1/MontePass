"""The static plots of the results folder, and the figure style of all plots and animations.

Rules that every figure of this package follows
-----------------------------------------------
* The width is one of three named sizes (``NARROW``, ``MEDIUM``, ``WIDE``, in inches). The saved file has
  exactly the declared size: the layout is computed by matplotlib ("constrained"), nothing is cropped.
* One font (DejaVu Sans, which ships with matplotlib) for text and numbers. Sizes come from the named
  scale ``FS``; no text is smaller than 7 points.
* Colours come from ``PALETTE`` (colour-blind-safe Okabe-Ito colours). A colour means one quantity in
  every figure. Categories also differ by marker shape or by filled against open markers.
* A stochastic number is drawn with its error bar (1 standard error), and the figure says so.
* Legends are placed in a strip above the panels, never on the data.
* The style is set inside a context (:func:`style`); the settings of the calling program are not changed.

Every ``plot_*`` function returns a ``matplotlib.figure.Figure`` (or a list of them) and raises
:class:`~montepass.report.tables.Skip` with the reason if the plot cannot be made from the data.
Call them inside ``with style():`` and save with :func:`save_figure`.
"""
from __future__ import annotations

import contextlib
import os
import textwrap
import warnings
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import matplotlib
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Text
from matplotlib.ticker import FuncFormatter, MaxNLocator, NullFormatter, ScalarFormatter
from matplotlib.transforms import ScaledTranslation

from ..units import HARTREE_TO_KCAL
from . import molview
from .tables import (CHANGE_FLOOR, IMAGE_MATCH_LIMIT, NUMERICAL_ZERO, IterationData, ReportData, Skip, TargetData,
                     coordinate_groups, ignore_library_deprecations, internal_table, path_internal)

# ----------------------------------------------------------------------------- style
NARROW = 3.33           # figure widths in inches
MEDIUM = 5.0
WIDE = 6.5
PNG_DPI = 200
FONT = "DejaVu Sans"


@dataclass(frozen=True)
class _TypeScale:
    """Text sizes in points. Plot code uses these names, never a number."""

    TITLE: float = 8.5      # title of a figure
    PANEL: float = 9.0      # panel letters (a), (b)
    LABEL: float = 8.0      # axis labels, panel titles
    TICK: float = 7.0       # tick labels
    LEGEND: float = 7.0
    ANNOT: float = 7.0      # notes inside a figure, atom labels
    SMALL: float = 6.0      # numbers inside the cells of a map (the smallest size that is used)


FS = _TypeScale()

# One colour per quantity (Okabe-Ito colours).
PALETTE = {
    "dmc": "#0072B2",           # DMC energies, and what follows from them (geometry changes)
    "surrogate": "#D55E00",     # surrogate (DFT) energies
    "model": "#000000",         # the quadratic model of the search
    "step": "#009E73",          # the step that the search took
    "gain": "#E69F00",          # predicted gain G
    "bound": "#CC79A7",         # bounds on the energy that is left
    "variance": "#56B4E9",      # explained variance of a principal component
    "cumulative": "#000000",    # cumulative explained variance
    "rejected": "#999999",      # a rejected step
    "guide": "#666666",         # reference lines (zero, tolerance)
    "coordinate": "#000000",    # bond lengths and angles of a geometry (the lines differ by style and marker)
}

RC = {
    "font.family": [FONT], "font.size": FS.TICK, "mathtext.fontset": "dejavusans",
    "axes.labelsize": FS.LABEL, "axes.titlesize": FS.LABEL, "xtick.labelsize": FS.TICK, "ytick.labelsize": FS.TICK,
    "legend.fontsize": FS.LEGEND, "figure.titlesize": FS.TITLE,
    "axes.linewidth": 0.6, "lines.linewidth": 1.0, "lines.markersize": 4.5, "patch.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.minor.width": 0.5, "ytick.minor.width": 0.5,
    "xtick.major.size": 3.0, "ytick.major.size": 3.0, "xtick.minor.size": 1.8, "ytick.minor.size": 1.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False, "legend.frameon": False,
    "errorbar.capsize": 2.0, "hatch.linewidth": 0.6,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "figure.dpi": 100.0, "savefig.bbox": "standard", "savefig.transparent": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,
}

_FONT_CHECKED = False


def _check_font() -> None:
    """matplotlib replaces a font that it cannot find without saying so. Make that an error."""
    global _FONT_CHECKED
    if _FONT_CHECKED:
        return
    try:
        findfont(FontProperties(family=[FONT]), fallback_to_default=False)
    except Exception as exc:
        raise RuntimeError("the font %s, which ships with matplotlib, was not found (%s)" % (FONT, exc))
    _FONT_CHECKED = True


@contextlib.contextmanager
def style():
    """Context in which every figure of the report is built and saved: matplotlib's own defaults (not the
    settings file of the user) plus ``RC``. The settings and the warning filters of the caller are restored
    on exit."""
    with matplotlib.rc_context(), warnings.catch_warnings():
        ignore_library_deprecations()
        matplotlib.rcdefaults()
        matplotlib.rcParams.update(RC)
        _check_font()
        yield


@contextlib.contextmanager
def recorded_warnings(sink: List[str]):
    """Inside this context a warning is not printed and not raised: its text is appended to ``sink`` once.
    Deprecation warnings from inside the plotting libraries are left out (see ``ignore_library_deprecations``)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ignore_library_deprecations()
        try:
            yield
        finally:
            for w in caught:
                text = "warning while drawing: %s" % w.message
                if text not in sink:
                    sink.append(text)


def _whole_pixels(inches: float) -> float:
    """The size nearest to ``inches`` that is a whole number of pixels at ``PNG_DPI``.

    matplotlib cuts the product "size times dots per inch" to an integer, and a product such as
    2.85 x 200 can come out as 569.9999999999999 in floating-point arithmetic. The returned size
    always gives the rounded number of pixels."""
    pixels = int(round(inches * PNG_DPI))
    value = pixels / float(PNG_DPI)
    while int(value * PNG_DPI) < pixels:
        value = float(np.nextafter(value, np.inf))
    return value


def new_figure(width: float, height: float) -> Figure:
    """A figure of ``width`` x ``height`` inches (a whole number of pixels at ``PNG_DPI``) with the constrained
    layout and an Agg canvas. No global figure list is used, so nothing has to be closed."""
    if width not in (NARROW, MEDIUM, WIDE):
        raise ValueError("figure width %.3f in is not one of the named sizes %s" % (width, (NARROW, MEDIUM, WIDE)))
    fig = Figure(figsize=(_whole_pixels(width), _whole_pixels(height)), layout="constrained")
    FigureCanvasAgg(fig)
    return fig


def freeze_layout(fig: Figure) -> None:
    """Compute the layout once and keep it. Used before the frames of an animation are drawn, so that
    no axes moves between frames."""
    fig.canvas.draw()
    if hasattr(fig, "set_layout_engine"):
        fig.set_layout_engine("none")
    else:                                        # matplotlib 3.5
        fig.set_constrained_layout(False)


def _axis_texts(ax) -> List[Text]:
    """The texts of one axes that are really drawn."""
    out: List[Text] = [c for c in ax.get_children() if isinstance(c, Text)]
    if ax.axison:
        for axis, limits in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
            out += [axis.label, axis.get_offset_text()]
            # a tick label is drawn only if its tick is inside the limits. The test is made in the scale of
            # the axis (the logarithm for a log axis): there a tick at 1e-13 is far below a limit of 1e-11.
            scale = axis.get_transform()

            def scaled(values):
                with np.errstate(divide="ignore", invalid="ignore"):
                    return np.asarray(scale.transform(np.asarray(values, dtype=float).reshape(-1, 1)), dtype=float).reshape(-1)

            lo, hi = sorted(scaled(limits))
            span = hi - lo
            for tick in list(axis.get_major_ticks()) + list(axis.get_minor_ticks()):
                loc = scaled([tick.get_loc()])[0]
                if np.isfinite(loc) and lo - 1e-9 * span <= loc <= hi + 1e-9 * span:
                    out += [tick.label1, tick.label2]
    legend = ax.get_legend()
    if legend is not None:
        out += list(legend.get_texts()) + [legend.get_title()]
    for child in getattr(ax, "child_axes", []):
        out += _axis_texts(child)
    return out


def text_problems(fig: Figure, overlap: float = 0.25, pad: float = 1.0) -> List[str]:
    """Check of a drawn figure: text that leaves the canvas, and two texts that cover each other.

    Returns one sentence per problem (an empty list for a clean figure). Two texts count as colliding if
    their boxes share a rectangle that is deeper, in both directions, than ``overlap`` times the height of
    the smaller font. (Boxes of neighbouring tick labels may touch by a pixel; that is not a collision.)
    ``pad`` is the number of pixels that a text may reach over the edge of the canvas. Atom labels of a
    molecule drawing are not compared with each other: one atom can be behind another one.
    """
    renderer = fig.canvas.get_renderer()
    texts: List[Text] = list(fig.texts)
    for ax in fig.axes:
        texts += _axis_texts(ax)
    boxes = []
    seen = set()
    for t in texts:
        if id(t) in seen or not t.get_visible() or not t.get_text().strip():
            continue
        seen.add(id(t))
        try:
            b = t.get_window_extent(renderer)
        except Exception:
            continue
        if b.width > 0 and b.height > 0:
            boxes.append((t, b))
    problems = []
    canvas = fig.bbox
    for i, (t1, b1) in enumerate(boxes):
        if b1.x0 < canvas.x0 - pad or b1.x1 > canvas.x1 + pad or b1.y0 < canvas.y0 - pad or b1.y1 > canvas.y1 + pad:
            problems.append("text reaches over the edge of the figure: %r" % t1.get_text())
        for t2, b2 in boxes[i + 1:]:
            if getattr(t1, "_atom_label", False) and getattr(t2, "_atom_label", False):
                continue
            w = min(b1.x1, b2.x1) - max(b1.x0, b2.x0)
            h = min(b1.y1, b2.y1) - max(b1.y0, b2.y0)
            limit = overlap * min(t1.get_fontsize(), t2.get_fontsize()) * fig.dpi / 72.0
            if min(w, h) > limit:
                problems.append("two texts cover each other: %r and %r" % (t1.get_text(), t2.get_text()))
    return sorted(set(problems))


def save_figure(fig: Figure, stem: str) -> Tuple[List[str], List[str]]:
    """Write ``<stem>.png`` (200 dpi) and ``<stem>.pdf`` at the declared size of the figure.

    Returns ``(files, problems)``. ``problems`` holds the findings of :func:`text_problems` and every warning
    that matplotlib gave while drawing. The files are written through temporary files."""
    files: List[str] = []
    problems: List[str] = []
    with recorded_warnings(problems):
        fig.canvas.draw()
        found = text_problems(fig)
        for ext, options in (("png", {"dpi": PNG_DPI}), ("pdf", {})):
            tmp = "%s.tmp.%s" % (stem, ext)
            fig.savefig(tmp, format=ext, bbox_inches=None, **options)
            os.replace(tmp, "%s.%s" % (stem, ext))
            files.append("%s.%s" % (stem, ext))
    return files, found + problems


def png_size(width: float, height: float) -> Tuple[int, int]:
    """Pixel size of the PNG that :func:`save_figure` writes for ``new_figure(width, height)``."""
    return int(round(width * PNG_DPI)), int(round(height * PNG_DPI))


# ----------------------------------------------------------------------------- shared pieces
BARS_NOTE = "every error bar is 1 standard error"


def _short(text: str, limit: int = 70) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _title(fig: Figure, text: str, bars: bool = False, more: str = "") -> None:
    """Title of a figure: one line with the project and the subject (cut to 70 characters). ``bars=True``
    adds a line that says what the error bars of the figure mean; ``more`` adds one more line."""
    lines = [_short(text)] + ([BARS_NOTE] if bars else []) + ([more] if more else [])
    fig.suptitle("\n".join(lines))


def _plain_log_axis(axis) -> None:
    """Tick labels of a logarithmic axis as plain numbers (0.01, not 10 with a small raised -2)."""
    axis.set_major_formatter(FuncFormatter(lambda value, _position: "%g" % value))
    axis.set_minor_formatter(NullFormatter())


def _legend_strip(ax, handles: Sequence, labels: Sequence[str], ncol: int, handler_map=None) -> None:
    """Put a legend in an axes of its own (above the panels), so that it can never cover data."""
    ax.set_axis_off()
    ax.legend(handles, labels, loc="center", ncol=ncol, borderaxespad=0.0, handlelength=1.8, handletextpad=0.5,
              columnspacing=1.2, labelspacing=0.3, handler_map=handler_map)


def _panel_letter(ax, letter: str) -> None:
    """Panel letter outside the data area, above the left end of the axes."""
    ax.text(0.0, 1.03, letter, transform=ax.transAxes, ha="left", va="bottom", fontsize=FS.PANEL, fontweight="bold")


def _marker(color: str, marker: str, filled: bool = True, size: float = 4.5) -> Line2D:
    """Legend handle of a marker without a line."""
    return Line2D([], [], linestyle="none", marker=marker, markersize=size, markerfacecolor=color if filled else "white",
                  markeredgecolor=color, color=color)


def _kcal(stat, reference: float) -> Tuple[float, float]:
    """Energy of a point relative to ``reference`` (hartree) and its own error bar, both in kcal/mol."""
    return (stat.value - reference) * HARTREE_TO_KCAL, stat.error * HARTREE_TO_KCAL


# ----------------------------------------------------------------------------- principal components
MAX_COMPONENTS = 12


def plot_pca_variance(data: ReportData) -> Figure:
    """Explained variance ratio of the principal components of the path, and its cumulative sum.
    Panel (a) has a linear scale, panel (b) shows the same ratios on a logarithmic scale."""
    sub = data.subspace
    if sub is None:
        raise Skip("the project has no subspace file")
    if sub.explained_variance_ratio is None:
        raise Skip("the subspace is of kind %r, not a principal component analysis of a path: it has no "
                   "explained variance" % sub.kind)
    evr = np.asarray(sub.explained_variance_ratio, dtype=float)
    evr = evr[evr > NUMERICAL_ZERO]
    if evr.size == 0:
        raise Skip("the subspace file holds no explained variance ratio above %.0e" % NUMERICAL_ZERO)
    n_all = int(evr.size)
    shown = evr[:MAX_COMPONENTS]
    n = int(shown.size)
    d = min(sub.dim, n)
    x = np.arange(1, n + 1)
    cum = np.cumsum(evr)[:n]
    kept = x <= d

    fig = new_figure(MEDIUM, 2.9)
    gs = fig.add_gridspec(2, 2, height_ratios=[0.14, 1.0])
    strip, a, b = fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    a.bar(x[kept], shown[kept], width=0.6, facecolor=PALETTE["variance"], edgecolor="black", zorder=2)
    a.bar(x[~kept], shown[~kept], width=0.6, facecolor="white", edgecolor=PALETTE["variance"], hatch="////", zorder=2)
    a.plot(x, cum, color=PALETTE["cumulative"], marker="D", markersize=3.5, zorder=3)
    a.set_xlim(0.4, n + 0.6)
    a.set_ylim(0.0, 1.08)
    a.set_xticks(x)
    a.set_xlabel("principal component" if n == n_all else "principal component (first %d of %d)" % (n, n_all))
    a.set_ylabel("fraction of the path variance")
    b.set_yscale("log")
    b.plot(x[kept], shown[kept], linestyle="none", marker="o", markerfacecolor=PALETTE["variance"],
           markeredgecolor="black", zorder=2)
    b.plot(x[~kept], shown[~kept], linestyle="none", marker="o", markerfacecolor="white",
           markeredgecolor=PALETTE["variance"], zorder=2)
    b.set_xlim(0.4, n + 0.6)
    b.set_ylim(float(shown.min()) / 3.0, 3.0)
    b.set_xticks(x)
    b.set_xlabel("principal component")
    b.set_ylabel("explained variance ratio (log scale)")
    _panel_letter(a, "(a)")
    _panel_letter(b, "(b)")
    handles = [(Patch(facecolor=PALETTE["variance"], edgecolor="black"),
                Line2D([], [], linestyle="none", marker="o", markerfacecolor=PALETTE["variance"], markeredgecolor="black"))]
    labels = ["explained variance ratio, kept in the subspace (%d)" % d]
    if not kept.all():
        handles.append((Patch(facecolor="white", edgecolor=PALETTE["variance"], hatch="////"),
                        Line2D([], [], linestyle="none", marker="o", markerfacecolor="white",
                               markeredgecolor=PALETTE["variance"])))
        labels.append("not kept")
    handles.append(Line2D([], [], color=PALETTE["cumulative"], marker="D", markersize=3.5))
    labels.append("cumulative sum")
    _legend_strip(strip, handles, labels, ncol=len(handles), handler_map={tuple: HandlerTuple(ndivide=None)})
    fig.suptitle(_short("%s: principal components of the path" % data.title))
    return fig


# ----------------------------------------------------------------------------- path profile
DODGE_PT = 5.0          # horizontal shift, in points, between markers that belong to the same path image


def plot_path_profile(data: ReportData) -> Figure:
    """Surrogate energy along the path, with the DMC energies of the targets at their path images.

    Every method is shown relative to its own energy at one reference image: the image of the first minimum
    target that has a DMC energy (image 0 if no target has one). DMC points are drawn at the starting geometry
    (open circles) and at the optimized geometry (filled circles) of a target."""
    p = data.path
    if p is None:
        raise Skip("the project has no path")
    if p.energies is None:
        raise Skip("the path file holds no surrogate energy for some images")
    e = p.energies
    n = len(e)
    on_path = [t for t in data.targets.values() if t.image is not None and t.image_rmsd <= IMAGE_MATCH_LIMIT]
    with_dmc = [t for t in on_path if t.start_energy is not None]
    ref = next((t for t in with_dmc if t.kind == "minimum"), with_dmc[0] if with_dmc else None)
    i_ref = ref.image if ref is not None else 0
    y = (e - e[i_ref]) * HARTREE_TO_KCAL

    fig = new_figure(MEDIUM, 3.4)
    gs = fig.add_gridspec(2, 1, height_ratios=[0.1, 1.0])
    strip, ax = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    ax.plot(np.arange(n), y, color=PALETTE["surrogate"], marker="s" if n <= 40 else None, markersize=3.5, zorder=2)
    handles = [Line2D([], [], color=PALETTE["surrogate"], marker="s" if n <= 40 else None, markersize=3.5)]
    labels = ["surrogate, path images"]
    if ref is not None:
        e_ref = ref.start_energy.value
        left = ax.transData + ScaledTranslation(-DODGE_PT / 72.0, 0.0, fig.dpi_scale_trans)
        right = ax.transData + ScaledTranslation(DODGE_PT / 72.0, 0.0, fig.dpi_scale_trans)
        v0 = [_kcal(t.start_energy, e_ref) for t in with_dmc]
        ax.errorbar([t.image for t in with_dmc], [v[0] for v in v0], yerr=[v[1] for v in v0], transform=left,
                    linestyle="none", marker="o", markerfacecolor="white", markeredgecolor=PALETTE["dmc"],
                    ecolor=PALETTE["dmc"], zorder=3)
        handles.append(_marker(PALETTE["dmc"], "o", filled=False))
        labels.append("DMC, starting geometry")
        done = [t for t in with_dmc if t.final_energy is not None and t.moved]
        if done:
            v1 = [_kcal(t.final_energy, e_ref) for t in done]
            ax.errorbar([t.image for t in done], [v[0] for v in v1], yerr=[v[1] for v in v1], transform=right,
                        linestyle="none", marker="o", markerfacecolor=PALETTE["dmc"], markeredgecolor=PALETTE["dmc"],
                        ecolor=PALETTE["dmc"], zorder=3)
            handles.append(_marker(PALETTE["dmc"], "o", filled=True))
            labels.append("DMC, optimized geometry")
    ax.set_xlim(-0.6, n - 0.4)
    ax.margins(y=0.08)
    ax.set_xlabel("path image (numbered from 0)")
    label = "energy minus energy at image %d,\nsame method (kcal/mol)" % i_ref
    ax.set_ylabel(label + ("\nDMC bars: 1 standard error" if ref is not None else ""))
    if on_path:
        names = {}
        for t in on_path:
            names.setdefault(t.image, []).append(t.name)
        top = ax.secondary_xaxis("top")
        top.set_xticks(sorted(names))
        top.set_xticklabels([", ".join(names[i]) for i in sorted(names)])
        top.set_xlabel("targets")
    _legend_strip(strip, handles, labels, ncol=len(handles))
    fig.suptitle(_short("%s: energy profile of the path" % data.title))
    return fig


# ----------------------------------------------------------------------------- convergence of one search
def plot_convergence(data: ReportData, name: str) -> Figure:
    """Centre energy, step length and predicted gain of one search against the iteration number.

    Panel (a): DMC energy of every scan centre minus the DMC energy of the starting geometry. The error bar of a
    point is that of its own energy. Panel (b): length of the step composed at each centre. Panel (c): predicted
    gain G and the bound on the energy that is left."""
    t = data.targets[name]
    its = t.iterations
    if not its:
        raise Skip("target %s has no search iteration" % name)
    if t.start_energy is None:
        raise Skip("target %s: the DMC energy of the starting geometry is not available yet" % name)
    e0 = t.start_energy.value
    with_steps = [it for it in its if it.step_length is not None]
    with_gain = [it for it in its if it.G is not None or it.joint_bound is not None or it.bound_v1 is not None]
    separate_final = bool(t.final_energy is not None and t.moved and (t.state or {}).get("final"))
    last = max(it.number for it in its)
    rows = 1 + bool(with_steps) + bool(with_gain)

    fig = new_figure(MEDIUM, {1: 2.7, 2: 4.0, 3: 5.3}[rows])
    gs = fig.add_gridspec(rows + 1, 1, height_ratios=[0.3] + [1.0] * rows)
    strip = fig.add_subplot(gs[0])
    axes = [fig.add_subplot(gs[1])]
    for r in range(1, rows):
        axes.append(fig.add_subplot(gs[r + 1], sharex=axes[0]))
    handles, labels = [], []

    # (a) centre energies
    ax = axes[0]
    ax.axhline(0.0, color=PALETTE["guide"], linewidth=0.6, linestyle=":", zorder=0.5)
    good = [it for it in its if it.energy is not None and it.accepted is not False]
    bad = [it for it in its if it.energy is not None and it.accepted is False]
    if good:
        v = [_kcal(it.energy, e0) for it in good]
        ax.errorbar([it.number for it in good], [a[0] for a in v], yerr=[a[1] for a in v], color=PALETTE["dmc"],
                    marker="o", linewidth=0.8, zorder=3)
        handles.append(Line2D([], [], color=PALETTE["dmc"], marker="o", linewidth=0.8))
        labels.append("scan centre")
    if bad:
        v = [_kcal(it.energy, e0) for it in bad]
        ax.errorbar([it.number for it in bad], [a[0] for a in v], yerr=[a[1] for a in v], linestyle="none", marker="X",
                    markersize=5.5, markerfacecolor="white", markeredgecolor=PALETTE["rejected"],
                    ecolor=PALETTE["rejected"], zorder=3)
        handles.append(_marker(PALETTE["rejected"], "X", filled=False, size=5.5))
        labels.append("centre of a rejected step")
    if separate_final:
        v = _kcal(t.final_energy, e0)
        ax.errorbar([last + 1], [v[0]], yerr=[v[1]], linestyle="none", marker="D", color=PALETTE["dmc"], zorder=3)
        handles.append(_marker(PALETTE["dmc"], "D"))
        labels.append("final geometry, full-length runs")
    ax.margins(y=0.12)
    ax.set_ylabel("DMC energy \u2212 start\n(kcal/mol)")
    letters = iter(["(a)", "(b)", "(c)"])
    _panel_letter(ax, next(letters))

    # (b) step length
    row = 1
    if with_steps:
        ax = axes[row]; row += 1
        errors = [0.0 if it.step_error is None else it.step_error for it in with_steps]
        ax.bar([it.number for it in with_steps], [it.step_length for it in with_steps], width=0.5,
               facecolor=PALETTE["step"], edgecolor="black", zorder=2)
        known = [(it.number, it.step_length, it.step_error) for it in with_steps if it.step_error is not None]
        if known:
            ax.errorbar([k[0] for k in known], [k[1] for k in known], yerr=[k[2] for k in known], linestyle="none",
                        ecolor="black", elinewidth=0.8, zorder=3)
        ax.set_ylim(0.0, 1.15 * max(a + b for a, b in zip([it.step_length for it in with_steps], errors)) or 1.0)
        ax.set_ylabel("step length (Å)")
        _panel_letter(ax, next(letters))

    # (c) predicted gain and bounds
    if with_gain:
        ax = axes[row]
        g = [it for it in with_gain if it.G is not None]
        jb = [it for it in with_gain if it.joint_bound is not None]
        b1 = [it for it in with_gain if it.bound_v1 is not None]
        ge = [0.0 if it.G_error is None else it.G_error for it in g]
        values = [it.G for it in g] + [it.joint_bound for it in jb] + [it.bound_v1 for it in b1]
        if t.eps_total is not None:
            values.append(t.eps_total)
        log = bool(values) and min(values) > 0.0
        cut = []                    # points whose error bar would reach zero or below: it cannot be drawn on a log scale
        if log:
            low = min(values) / 3.0
            ax.set_yscale("log")
            ax.set_ylim(low, max(values + [it.G + e for it, e in zip(g, ge)]) * 3.0)
            ax.set_ylabel("energy (kcal/mol)")
            cut = [it.G - e < 1.3 * low for it, e in zip(g, ge)]
        else:
            ax.margins(y=0.12)
            ax.set_ylabel("energy (kcal/mol)")
        if g:
            down = [it.G - 1.3 * low if c else e for it, e, c in zip(g, ge, cut)] if log else ge
            ax.errorbar([it.number for it in g], [it.G for it in g], yerr=[down, ge], color=PALETTE["gain"], marker="^",
                        linewidth=0.8, zorder=3)
            handles.append(Line2D([], [], color=PALETTE["gain"], marker="^", linewidth=0.8))
            labels.append("predicted gain G")
            if any(cut):
                ax.plot([it.number for it, c in zip(g, cut) if c], [1.3 * low] * sum(cut), linestyle="none", marker=7,
                        markersize=5.0, color=PALETTE["gain"], zorder=3)
                handles.append(Line2D([], [], linestyle="none", marker=7, markersize=5.0, color=PALETTE["gain"]))
                labels.append("the error bar of G reaches zero")
        if jb:
            ax.plot([it.number for it in jb], [it.joint_bound for it in jb], color=PALETTE["bound"], marker="s",
                    linewidth=0.8, linestyle="--", zorder=3)
            handles.append(Line2D([], [], color=PALETTE["bound"], marker="s", linewidth=0.8, linestyle="--"))
            labels.append("joint 95 % bound on the energy left")
        if b1:
            ax.plot([it.number for it in b1], [it.bound_v1 for it in b1], color=PALETTE["bound"], marker="s",
                    linewidth=0.8, linestyle="--", zorder=3)
            handles.append(Line2D([], [], color=PALETTE["bound"], marker="s", linewidth=0.8, linestyle="--"))
            labels.append("bound on the energy left")
        if t.eps_total is not None:
            ax.axhline(t.eps_total, color=PALETTE["guide"], linewidth=0.8, linestyle="-.", zorder=0.5)
            handles.append(Line2D([], [], color=PALETTE["guide"], linewidth=0.8, linestyle="-."))
            labels.append("tolerance %g kcal/mol" % t.eps_total)
        _panel_letter(ax, next(letters))

    ticks = [it.number for it in its] + ([last + 1] if separate_final else [])
    axes[-1].set_xticks(ticks)
    axes[-1].set_xticklabels(["%d" % it.number for it in its] + (["final"] if separate_final else []))
    axes[-1].set_xlim(min(ticks) - 0.6, max(ticks) + 0.6)
    axes[-1].set_xlabel("iteration")
    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)
    _legend_strip(strip, handles, labels, ncol=2 if len(handles) > 3 else max(1, len(handles)))
    # The axis labels of stacked panels must be shorter than a panel is high, so the meaning of the error
    # bars is said once, under the title.
    _title(fig, "%s: search of target %s" % (data.title, name), bars=True)
    return fig


# ----------------------------------------------------------------------------- bond lengths and angles along the path
# line style, marker and "marker filled" of the first, second, ... line of a panel. All lines are black, so
# they can be told apart in a grey print too.
COORD_STYLES = (("-", "o", True), ("--", "s", False), ("-.", "^", True), (":", "D", False))


def coordinate_label(group: dict) -> str:
    """Label of a group of equal bonds or angles for a figure, for example ``N1–H2 and 2 equal bonds``."""
    return group["label"].replace("-", "\u2013")


def coordinate_lines(ax, x, groups: Sequence[dict], markers: bool = True) -> List[Line2D]:
    """One line per group of ``tables.coordinate_groups`` against the path image. Returns the lines."""
    lines = []
    for g, (style, marker, filled) in zip(groups, COORD_STYLES):
        line, = ax.plot(x, g["values"], color=PALETTE["coordinate"], linestyle=style, linewidth=0.9,
                        marker=marker if markers else None, markersize=3.5,
                        markerfacecolor=PALETTE["coordinate"] if filled else "white",
                        markeredgecolor=PALETTE["coordinate"], zorder=2)
        lines.append(line)
    return lines


LINE_MERGE = 0.03            # curves closer than this fraction of the largest range are drawn as one line
CHANGE_CMAP = "PuOr"         # change of a bond length or angle: orange = smaller, white = no change, purple = larger.
                             # Diverging, readable with the common colour-vision deficiencies, and none of its
                             # colours is the colour of an element in the drawings
MAP_ROWS = 24                # at most this many rows in one panel of the map of geometry changes
NOTE_WIDTH = 42              # characters in one line of the note under the molecule of the map figure
BOND_CHANGE_MIN = 0.005      # angstrom: smallest full scale of the colours of a bond (a smaller change stays pale)
CHANGE_CAP = {"bond": 0.30, "angle": 30.0}      # largest full scale of the colours (angstrom, degrees). A bond that
                             # breaks changes by several angstrom; with a scale that wide, the bonds that change their
                             # order (0.05 to 0.2 angstrom) would all look white. Larger changes get the end colour.
NICE_STEPS = (1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0)


def nice_limit(value: float, smallest: float = 0.0) -> float:
    """``value`` rounded up to a round number (1, 1.5, 2, 3, 4, 5, 6 or 8 times a power of ten), at least ``smallest``."""
    v = max(float(value), float(smallest))
    if not v > 0.0:
        return 1.0
    power = 10.0 ** np.floor(np.log10(v))
    for step in NICE_STEPS:
        if v <= step * power * (1.0 + 1e-9):
            return float(step * power)
    return float(10.0 * power)


def change_scale(largest: float, smallest: float = 0.0, cap: Optional[float] = None):
    """Colour scale for a change that can have both signs: ``(norm, colour map)`` with zero in the middle and
    the limits at plus and minus :func:`nice_limit` of ``largest``, but not beyond ``cap``. ``norm.clip`` is
    set, so a larger change gets the end colour."""
    limit = nice_limit(largest, smallest)
    if cap is not None:
        limit = min(limit, float(cap))
    return Normalize(vmin=-limit, vmax=limit, clip=True), matplotlib.colormaps[CHANGE_CMAP]


def bond_change_scale(symbols: Sequence[str], frames: Sequence[np.ndarray], pairs: Sequence[Tuple[int, int]]):
    """Colour scale of the bonds of a path: the largest change of a bond length from the first frame decides,
    up to ``CHANGE_CAP``. Returns ``(norm, colour map, largest change)``."""
    X0 = np.asarray(frames[0], dtype=float)
    largest = 0.0
    for X in frames:
        for i, j in pairs:
            largest = max(largest, abs(float(np.linalg.norm(X[i] - X[j]) - np.linalg.norm(X0[i] - X0[j]))))
    norm, cmap = change_scale(largest, BOND_CHANGE_MIN, CHANGE_CAP["bond"])
    return norm, cmap, largest


def change_colorbar(fig: Figure, cax, norm, cmap, largest: float, label: str, orientation: str = "vertical"):
    """Colour bar of a change scale. Pointed ends say that changes beyond the scale exist (they have the end colour)."""
    beyond = float(largest) > float(norm.vmax) * (1.0 + 1e-9)
    bar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation=orientation,
                       extend="both" if beyond else "neither")
    bar.set_label(label)
    if orientation == "horizontal":
        bar.locator = MaxNLocator(nbins=4, symmetric=True)          # few numbers: the bar is short
        bar.update_ticks()
    return bar


def _map_rows(internal: dict, kind: str) -> Tuple[List[dict], int, int]:
    """Rows of one panel of the map of geometry changes: the groups of equal coordinates whose largest change
    from image 0 is at least ``CHANGE_FLOOR``, at most ``MAP_ROWS`` of them (those with the largest change),
    sorted from the largest increase to the largest decrease. Also returned: the number of coordinates below
    the floor, and the number of coordinates left out because of the row limit."""
    groups, _total = coordinate_groups(internal, kind, limit=10 ** 6)
    big = sorted([g for g in groups if abs(g["extreme"]) >= CHANGE_FLOOR[kind]], key=lambda g: -abs(g["extreme"]))
    small = int(sum(len(g["members"]) for g in groups if abs(g["extreme"]) < CHANGE_FLOOR[kind]))
    cut = int(sum(len(g["members"]) for g in big[MAP_ROWS:]))
    rows = sorted(big[:MAP_ROWS], key=lambda g: (-g["extreme"], g["members"][0]))
    return rows, small, cut


def _row_label(g: dict) -> str:
    n = len(g["members"])
    return g["name"].replace("-", "\u2013") + ("" if n == 1 else " (%d equal)" % n)


def plot_path_geometry_map(data: ReportData) -> Figure:
    """Every bond length and angle that changes along the path, for a molecule with many of them.

    (a) The molecule at the image where the bonds differ most from image 0; every bond has the colour of its
    change of length. (b) One row per bond and one column per path image; the colour is the bond length minus
    its value at image 0. (c) The same for the angles. Rows are sorted from the largest increase to the largest
    decrease. Bonds or angles with equal values are one row. A coordinate that changes by less than
    ``CHANGE_FLOOR`` has no row; the figure says how many these are. The figure is not made when every bond and
    angle is already a line in ``plots/path_geometry``."""
    p = data.path
    if p is None:
        raise Skip("the project has no path")
    n = len(p.frames)
    if n < 2:
        raise Skip("the path has only one image")
    internal = path_internal(p)
    _b, n_bond_groups = coordinate_groups(internal, "bond", 1)
    _a, n_angle_groups = coordinate_groups(internal, "angle", 1)
    if n_bond_groups == 0:
        raise Skip("no pair of atoms of the path is close enough to be a bond")
    if n_bond_groups <= len(COORD_STYLES) and n_angle_groups <= len(COORD_STYLES):
        raise Skip("every bond and angle of this molecule is a line in plots/path_geometry")
    bonds, bonds_small, bonds_cut = _map_rows(internal, "bond")
    angles, angles_small, angles_cut = _map_rows(internal, "angle")
    if not bonds and not angles:
        raise Skip("no bond changes by %g angstrom and no angle by %g degrees along the path"
                   % (CHANGE_FLOOR["bond"], CHANGE_FLOOR["angle"]))
    panels = [(kind, rows) for kind, rows in (("bond", bonds), ("angle", angles)) if rows]
    n_rows = sum(len(rows) for _kind, rows in panels)
    fig = new_figure(WIDE, float(min(9.0, max(3.6, 1.25 + 0.17 * n_rows + 0.55 * len(panels)))))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.85, 1.25])
    left = gs[0, 0].subgridspec(2, 1, height_ratios=[1.0, 0.42])
    right = gs[0, 1].subgridspec(len(panels), 2, width_ratios=[1.0, 0.045],
                                 height_ratios=[len(rows) + 1.5 for _kind, rows in panels])
    pairs = molview.union_bonds(p.symbols, p.frames)
    norm_bond, cmap, largest_bond = bond_change_scale(p.symbols, p.frames, pairs)
    x_edges = np.arange(n + 1) - 0.5
    top = None
    maps = []
    for r, (kind, rows) in enumerate(panels):
        ax = fig.add_subplot(right[r, 0])
        top = top if top is not None else ax
        cax = fig.add_subplot(right[r, 1])
        Z = np.array([g["delta"] for g in rows])
        largest = float(np.abs(Z).max())
        if kind == "bond":
            norm, largest = norm_bond, max(largest, largest_bond)
        else:
            norm, _cmap = change_scale(largest, cap=CHANGE_CAP["angle"])
        ax.pcolormesh(x_edges, np.arange(len(rows) + 1) - 0.5, Z, cmap=cmap, norm=norm, edgecolors="face", linewidth=0.0)
        ax.invert_yaxis()
        ax.set_yticks(np.arange(len(rows)))
        ax.set_yticklabels([_row_label(g) for g in rows])
        ax.tick_params(axis="y", length=0)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))
        change_colorbar(fig, cax, norm, cmap, largest,
                        "bond length minus its value\nat image 0 (Å)" if kind == "bond"
                        else "angle minus its value\nat image 0 (degrees)")
        _panel_letter(ax, "(%s)" % "bc"[r])
        if r < len(panels) - 1:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("path image (numbered from 0)")
        maps.append((ax, kind, rows, norm))
    # the image of the highest surrogate energy: a triangle above the first map
    k_top = None
    if p.energies is not None and 0 < int(np.argmax(p.energies)) < n - 1:
        k_top = int(np.argmax(p.energies))
        top.plot([k_top], [1.035], transform=top.get_xaxis_transform(), marker="v", markersize=5.0,
                 color=PALETTE["coordinate"], clip_on=False, zorder=5, linestyle="none")
    # (a) the molecule at the image whose bonds differ most from image 0
    X0 = np.asarray(p.frames[0], dtype=float)
    spread = [sum((np.linalg.norm(X[i] - X[j]) - np.linalg.norm(X0[i] - X0[j])) ** 2 for i, j in pairs) for X in p.frames]
    k_show = int(np.argmax(spread)) if max(spread) > 0.0 else n - 1
    note = fig.add_subplot(left[1, 0])
    note.set_axis_off()
    lines = ["(a) Image %d. Bond colours: change of the bond length from image 0, on the scale of panel (b)." % k_show,
             "Rows: the largest increase at the top, the largest decrease at the bottom.",
             "(2 equal): 2 bonds or angles with equal values."]
    if k_top is not None:
        lines.append("Triangle above (b): the image with the highest surrogate energy (image %d)." % k_top)
    for kind, small, cut in (("bond", bonds_small, bonds_cut), ("angle", angles_small, angles_cut)):
        unit = "Å" if kind == "bond" else "degrees"
        if small:
            lines.append("No row: %d %s%s that change%s by less than %g %s." % (
                small, kind, "s" if small > 1 else "", "" if small > 1 else "s", CHANGE_FLOOR[kind], unit))
        if cut:
            lines.append("No row: %d more %s%s (the panel shows the %d largest)." % (cut, kind, "s" if cut > 1 else "", MAP_ROWS))
    note.text(0.0, 0.95, "\n".join(textwrap.fill(line, width=NOTE_WIDTH) for line in lines), ha="left", va="top",
              fontsize=FS.ANNOT, transform=note.transAxes)
    fig.suptitle(_short("%s: change of the bond lengths and angles along the path" % data.title))
    panel = molview.MoleculePanel(fig, left[0, 0], p.symbols, p.frames, pairs, first=p.frames[k_show],
                                  bonds=molview.bond_list(p.symbols, p.frames[k_show]), fontsize=FS.ANNOT,
                                  colored_bonds=True,
                                  bond_colors=molview.bond_change_colors(pairs, p.frames[k_show], X0, norm_bond, cmap))
    _panel_letter(panel.axes[0], "(a)")
    # the numbers of the changes are written into the cells where they fit (a path with few images)
    fig.canvas.draw()
    for ax, kind, rows, norm in maps:
        fmt = "%+.3f" if kind == "bond" else "%+.1f"
        sample = max((fmt % g["delta"][k] for g in rows for k in range(n)), key=len)
        cell = ax.get_window_extent().width / fig.dpi / float(n)                 # inches
        if 0.62 * FS.SMALL * len(sample) / 72.0 > 0.92 * cell:
            continue
        for i, g in enumerate(rows):
            for k in range(n):
                rgba = cmap(norm(g["delta"][k]))
                dark = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2] < 0.5
                ax.text(k, i, fmt % g["delta"][k], ha="center", va="center", fontsize=FS.SMALL,
                        color="#FFFFFF" if dark else "#000000")
    return fig


def plot_path_geometry(data: ReportData) -> Figure:
    """The bond lengths and the angles that change most along the path, against the path image.

    Bonds or angles that have the same value at every image are one line (the label says how many), and so
    are curves that the figure could not show apart. A bond that breaks or forms along the path is followed
    over the whole path. All values are in the file ``geometries/path_internal_coordinates.csv``; a molecule
    with many bonds also gets the map ``plots/path_geometry_map``."""
    p = data.path
    if p is None:
        raise Skip("the project has no path")
    if len(p.frames) < 2:
        raise Skip("the path has only one image")
    internal = path_internal(p)
    bonds, n_bonds = coordinate_groups(internal, "bond", len(COORD_STYLES), merge=LINE_MERGE)
    angles, n_angles = coordinate_groups(internal, "angle", len(COORD_STYLES), merge=LINE_MERGE)
    if not bonds:
        raise Skip("no pair of atoms of the path is close enough to be a bond")
    panels = [("bond length (Å)", bonds, n_bonds)] + ([("angle (degrees)", angles, n_angles)] if angles else [])
    fig = new_figure(MEDIUM, 0.55 + 2.25 * len(panels))
    gs = fig.add_gridspec(2 * len(panels), 1, height_ratios=[0.42, 1.8] * len(panels))
    x = np.arange(len(p.frames))
    first = None
    ax = None
    for r, (ylabel, groups, total) in enumerate(panels):
        strip = fig.add_subplot(gs[2 * r, 0])
        ax = fig.add_subplot(gs[2 * r + 1, 0], sharex=first)
        first = first if first is not None else ax
        lines = coordinate_lines(ax, x, groups, markers=len(x) <= 40)
        _legend_strip(strip, lines, [coordinate_label(g) for g in groups], ncol=2 if len(groups) > 1 else 1)
        ax.margins(x=0.04, y=0.15)
        more = "\nthe %d curves that change most, of %d" % (len(groups), total) if total > len(groups) else ""
        ax.set_ylabel(ylabel + more)
        _panel_letter(ax, "(%s)" % "ab"[r])
        if r < len(panels) - 1:
            ax.tick_params(labelbottom=False)
    ax.set_xlabel("path image (numbered from 0)")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))
    fig.suptitle(_short("%s: bond lengths and angles along the path" % data.title))
    return fig


# ----------------------------------------------------------------------------- scans of one iteration
MAX_PANELS = 9


MODEL_LABEL = {"search": "quadratic model of the search", "cubic": "cubic fit of the search",
               "fit": "quadratic fit of these points",
               "search+shape": "model of the search: slope, curvature, cubic term of the surrogate"}


def _model_of(points: List[Tuple[float, float, float]], estimate: Optional[dict]):
    """The curve of one direction: ``(function of t, origin)``, or None if no curve can be given.

    * origin ``"cubic"``: the cubic fit of the classic line search (its coefficients are in the state file);
    * origin ``"search"``: E(t) = c + g t + k t^2 / 2 with the slope g and the force constant k of the
      estimator of the dense-scan search;
    * origin ``"search+shape"``: the same plus the cubic term of the surrogate, when the search subtracted
      the surrogate (protocol v2 with the control variate);
    * origin ``"fit"``: without these numbers, a weighted least-squares parabola through at least 3 points.

    In the first two cases the shape is that of the search and only the constant c is fitted to the points
    (weights 1 / error^2), because the search and this figure do not use the same zero of energy.
    ``points`` are (t, energy, error) in angstrom and kcal/mol."""
    tt = np.array([p[0] for p in points]); yy = np.array([p[1] for p in points])
    ee = np.array([p[2] for p in points])
    w = 1.0 / np.maximum(ee, 1e-12) ** 2
    shape, origin = None, ""
    if estimate and estimate.get("coefficients"):
        higher = [float(c) for c in estimate["coefficients"][1:]]
        shape, origin = (lambda x: sum(c * np.asarray(x, dtype=float) ** (p + 1) for p, c in enumerate(higher))), "cubic"
    elif estimate and estimate.get("g") is not None and estimate.get("k") is not None:
        g, k = float(estimate["g"]), float(estimate["k"])
        c, R = estimate.get("shape"), estimate.get("span")
        if c is not None and R:
            # The search takes the slope at the centre from the pair of points and corrects it with the cubic
            # term of the surrogate, c = g_surrogate - (slope of the surrogate pair). A curve with this slope
            # passes through the points only with the same cubic term: -c t^3 / R^2.
            c3 = -float(c) / float(R) ** 2
            shape, origin = (lambda x: g * np.asarray(x, dtype=float) + 0.5 * k * np.asarray(x, dtype=float) ** 2
                             + c3 * np.asarray(x, dtype=float) ** 3), "search+shape"
        else:
            shape, origin = (lambda x: g * np.asarray(x, dtype=float) + 0.5 * k * np.asarray(x, dtype=float) ** 2), "search"
    if shape is not None:
        c0 = float(np.sum(w * (yy - shape(tt))) / w.sum())
        return (lambda x: c0 + shape(x)), origin
    if len(set(np.round(tt, 6))) >= 3:
        A = np.vstack([np.ones_like(tt), tt, 0.5 * tt ** 2]).T * np.sqrt(w)[:, None]
        c, g, k = [float(v) for v in np.linalg.lstsq(A, yy * np.sqrt(w), rcond=None)[0]]
        return (lambda x: c + g * np.asarray(x, dtype=float) + 0.5 * k * np.asarray(x, dtype=float) ** 2), "fit"
    return None


def _scan_panel(ax, t: TargetData, it: IterationData, j: int) -> dict:
    """Draw one direction. Returns which kinds of marks were drawn (for the legend)."""
    drawn = {"model": None, "step": False, "mirror": False}
    pts = [p for p in it.scans[j] if p.stat is not None]
    est = it.estimates.get(j)
    if it.energy is not None:
        ref = it.energy.value
    else:
        w = np.array([1.0 / max(p.stat.error, 1e-12) ** 2 for p in pts])
        ref = float(np.sum(w * np.array([p.stat.value for p in pts])) / w.sum())
    rows = [(p.t,) + _kcal(p.stat, ref) for p in pts]
    if it.energy is not None:
        rows.append((0.0, 0.0, it.energy.error * HARTREE_TO_KCAL))
    rows.sort()
    span = max(abs(r[0]) for r in rows) or 0.05
    model = _model_of(rows, est)
    if model is not None:
        curve, drawn["model"] = model
        x = np.linspace(-1.2 * span, 1.2 * span, 121)
        ax.plot(x, curve(x), color=PALETTE["model"], zorder=1.5)
    ax.errorbar([r[0] for r in rows], [r[1] for r in rows], yerr=[r[2] for r in rows], linestyle="none", marker="o",
                color=PALETTE["dmc"], zorder=3)
    one_sided = [r for r in rows if r[0] != 0.0 and not any(abs(q[0] + r[0]) < 1e-9 for q in rows)]
    if est and est.get("by_symmetry") and one_sided:
        ax.plot([-r[0] for r in one_sided], [r[1] for r in one_sided], linestyle="none", marker="o",
                markerfacecolor="white", markeredgecolor=PALETTE["dmc"], zorder=3)
        drawn["mirror"] = True
    step = None if est is None else est.get("step")
    notes = []
    if t.roles and j < len(t.roles) and t.roles[j] < 0:
        notes.append("energy maximized")
    if est and est.get("by_symmetry"):
        notes.append("slope zero by symmetry")
    if step is not None and abs(step) <= 1.2 * span:
        ax.axvline(step, color=PALETTE["step"], linestyle="--", zorder=0.5)
        drawn["step"] = True
    elif step is not None:
        notes.append("step %+.3f Å, outside the plot" % step)
    ax.set_xlim(-1.3 * span, 1.3 * span)
    ax.margins(y=0.12)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
    ax.set_title("\n".join(["direction %d" % j] + notes), loc="left")
    return drawn


def plot_scans(data: ReportData, name: str) -> List[Figure]:
    """The scans of the last iteration that has scan data: one small panel per direction, at most 9 per figure.

    Each panel shows the DMC energies of the scan points and of the centre against the displacement along the
    direction, relative to the centre energy, the quadratic model, and the step that the search took."""
    t = data.targets[name]
    it = t.last_scan_iteration()
    if it is None:
        raise Skip("target %s has no scan point with enough DMC blocks yet" % name)
    scanned = [j for j in sorted(it.scans) if any(p.stat is not None for p in it.scans[j])]
    empty = [j for j in sorted(it.scans) if not it.scans[j]]
    pages = [scanned[i:i + MAX_PANELS] for i in range(0, len(scanned), MAX_PANELS)]
    figures = []
    for number, page in enumerate(pages, start=1):
        n = len(page)
        ncols = min(3, n)
        nrows = int(np.ceil(n / float(ncols)))
        width, strip_height = (NARROW, 0.7) if n == 1 else (WIDE, 0.4)
        fig = new_figure(width, strip_height + 0.75 + 1.9 * nrows)
        gs = fig.add_gridspec(nrows + 1, ncols, height_ratios=[strip_height] + [1.9] * nrows)
        strip = fig.add_subplot(gs[0, :])
        seen = {"model": None, "step": False, "mirror": False}
        for i, j in enumerate(page):
            ax = fig.add_subplot(gs[1 + i // ncols, i % ncols])
            drawn = _scan_panel(ax, t, it, j)
            seen["model"] = seen["model"] or drawn["model"]
            seen["step"] = seen["step"] or drawn["step"]
            seen["mirror"] = seen["mirror"] or drawn["mirror"]
        handles = [Line2D([], [], color=PALETTE["dmc"], marker="o", linestyle="none")]
        labels = ["DMC energy (bars: 1 standard error)"]
        if seen["mirror"]:
            handles.append(_marker(PALETTE["dmc"], "o", filled=False))
            labels.append("same energy by symmetry (not computed)")
        if seen["model"]:
            handles.append(Line2D([], [], color=PALETTE["model"]))
            labels.append(MODEL_LABEL[seen["model"]])
        if seen["step"]:
            handles.append(Line2D([], [], color=PALETTE["step"], linestyle="--"))
            labels.append("step taken")
        _legend_strip(strip, handles, labels, ncol=1 if n == 1 else 2)
        fig.supxlabel("displacement along the direction (Å)", fontsize=FS.LABEL)
        fig.supylabel("DMC energy minus energy of the scan centre\n(kcal/mol)" if it.energy is not None else
                      "DMC energy minus mean of the points\n(kcal/mol)", fontsize=FS.LABEL)
        title = "%s: target %s, scans of iteration %d" % (data.title, name, it.number)
        if len(pages) > 1:
            title += " (figure %d of %d)" % (number, len(pages))
        if empty and number == 1:
            title = _short(title) + "\nnot scanned, slope zero by symmetry: direction %s" % ", ".join("%d" % j for j in empty)
            fig.suptitle(title)
        else:
            fig.suptitle(_short(title))
        figures.append(fig)
    return figures


# ----------------------------------------------------------------------------- geometry change
MAX_BARS = 10


def _bar_panel(ax, rows: List[dict], unit: str, title: str, resolution: float) -> None:
    """Horizontal bars of the largest changes, sorted by size, each named on the axis.

    ``resolution`` is the smallest change of this quantity that means anything (rounding of the files).
    The axis never shows less than 10 times this value on each side of zero, so that rounding noise is
    not drawn as if it were a change."""
    rows = sorted(rows, key=lambda r: -abs(r["change"]))
    shown = rows[:MAX_BARS]
    y = np.arange(len(shown))
    ax.axvline(0.0, color=PALETTE["guide"], linewidth=0.6, zorder=0.5)
    ax.barh(y, [r["change"] for r in shown], height=0.6, facecolor=PALETTE["dmc"], edgecolor="black", zorder=2)
    known = [(i, r) for i, r in enumerate(shown) if r["error"] is not None]
    if known:
        ax.errorbar([r["change"] for _i, r in known], [i for i, _r in known], xerr=[r["error"] for _i, r in known],
                    linestyle="none", ecolor="black", elinewidth=0.8, zorder=3)
    ends = [r["change"] + s * (r["error"] or 0.0) for r in shown for s in (-1.0, 1.0)]
    lo, hi = min(0.0, min(ends)), max(0.0, max(ends))
    pad = 0.12 * (hi - lo)
    ax.set_xlim(min(lo - pad, -10.0 * resolution), max(hi + pad, 10.0 * resolution))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
    formatter = ScalarFormatter(useOffset=False)
    formatter.set_scientific(False)
    ax.xaxis.set_major_formatter(formatter)
    ax.set_yticks(y)
    ax.set_yticklabels([r["atoms"].replace("-", "–") for r in shown])
    ax.set_ylim(len(shown) - 0.4, -0.6)
    ax.set_title(title)
    ax.set_xlabel("final minus start (%s)" % unit + ("\nbars: 1 standard error" if known else "")
                  + ("\nthe %d largest of %d" % (len(shown), len(rows)) if len(rows) > len(shown) else ""))


def plot_geometry(data: ReportData, name: str) -> Figure:
    """Change of the bond lengths and of the angles from the starting geometry to the final geometry.

    Panel (a) is a drawing of the final geometry with the atom labels that the other panels use. The bars are
    sorted by size; the largest changes are named on the axis. Error bars are the statistical errors of the
    final geometry (see ``tables.internal_table``)."""
    t = data.targets[name]
    if t.final is None or t.start is None or t.symbols is None:
        raise Skip("the search of target %s has not ended: there is no final geometry" % name)
    if t.moved is False:
        raise Skip("the search did not move the geometry of target %s" % name)
    rows = internal_table(t)
    bonds = [r for r in rows if r["kind"] == "bond"]
    angles = [r for r in rows if r["kind"] == "angle"]
    if not bonds:
        raise Skip("target %s has no bond (no pair of atoms is close enough)" % name)
    panels = 3 if angles else 2
    fig = new_figure(WIDE, 3.0)
    gs = fig.add_gridspec(1, panels, width_ratios=[0.85, 1.0, 1.0][:panels])
    ax_m, ax_b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    _bar_panel(ax_b, bonds, "Å", "bond lengths", 1e-4)
    _panel_letter(ax_b, "(b)")
    if angles:
        ax_a = fig.add_subplot(gs[0, 2])
        _bar_panel(ax_a, angles, "degrees", "angles", 1e-3)
        _panel_letter(ax_a, "(c)")
    fig.suptitle(_short("%s: geometry change of target %s" % (data.title, name)))
    ax_m.set_title("final geometry")
    molview.draw_molecule(ax_m, t.symbols, t.final, fontsize=FS.ANNOT)
    _panel_letter(ax_m, "(a)")
    return fig
