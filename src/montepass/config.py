"""The input file of a project (``input.ini``).

The file has a few sections with ``key = value`` lines. Every key has a default,
so a minimal input names only the geometry files. ``Config.load`` checks the
values and reports every problem with the section and key that caused it.
"""
from __future__ import annotations

import configparser
import os
import re
from typing import Callable, Dict, List, Optional

TEMPLATE = """\
; Input of one montepass project. Lines that start with ";" are comments.
; Every key has a default: delete what you do not need.

[system]
name = my_molecule
charge = 0
; spin multiplicity 2S+1: 1 = singlet, 2 = doublet, 3 = triplet
multiplicity = 1

[geometry]
; mode = path    reactant and product (and, if you have one, a saddle guess); the path between them is searched
; mode = images  a path or a few images that you already have, as one multi-frame xyz file
; mode = single  one geometry, no path
mode = path
reactant = reactant.xyz
product = product.xyz
saddle_guess =
images =
geometry =
; Geometries that are optimized on the DMC surface.
;   mode path:   any of  reactant, product, saddle
;   mode images: image numbers, counted from 0 (for example  0, 3)
;   mode single: geometry
; The modes path and images also take one of two words in place of a list:
;   barrier  the first minimum and the saddle of the path: 2 searches, which give the barrier height
;   mep      every image of the path: the two ends as minima, the saddle as a saddle, and every other
;            image across the path (kind image, see below). One search per image.
targets = reactant, saddle
; Only for mode images and mode single: what each target is, in the order of "targets": minimum or saddle.
; Empty = every target is a minimum. Mode images has a third kind, image: an image between the ends of the
; path that is no stationary point. Its search leaves out the path direction (the difference of the two
; neighbour images) and treats the image as a minimum in the other directions of the subspace.
; Mode images also takes  kinds = auto : the first and the last image are minima, the saddle image is a
; saddle, and every other image has the kind image. The words barrier and mep always use these kinds.
kinds =
; Only for mode images with barrier, mep or kinds = auto: the number of the saddle image. auto = the image
; with the highest energy in the comment lines of the path file (ORCA writes them: "... E -56.4730990").
saddle_image = auto

[surrogate]
; Program for the path search and the first optimizations: orca, or none (use the given geometries as they are)
program = orca
method = B3LYP
basis = cc-pVTZ
; optimize the given end points (or the single geometry) before anything else
optimize = yes
; compute frequencies of the optimized end points of a path, to check that they are minima
frequencies = yes
neb_images = 8
; NEB-TS (path, then saddle optimization) or NEB-CI (climbing image only)
neb_kind = NEB-TS

[subspace]
; pca = principal components of the path images; modes = softest modes of the surrogate Hessian;
; auto = pca when a path exists, else modes
kind = auto
; number of search coordinates: a number, or auto
dim = auto
; with dim = auto and pca: the smallest number of components that explains this fraction of the path variance
variance = 0.999
; largest number of search coordinates with dim = auto. auto = 6 principal components of a path; for a single
; geometry all internal coordinates of the molecule up to 12 (then the softest 12, and the log says so)
max_dim = auto

[qmc]
; auto = short pilot runs choose the settings; fixed = the values below are used as they are, except that the
; walker count is rounded up to a multiple of the MPI ranks;
; iterative = as auto, and a pilot is repeated (at most max_repeats times) when a check fails
mode = auto
max_repeats = 2
; trial wave function: Kohn-Sham determinant with these settings, ccECP pseudopotentials
xc = b3lyp
basis = ccecpccpvtz
ecp = ccecp
; wanted error bar of the final DMC energy of one geometry, kcal/mol (modes auto and iterative)
sigma_target = 0.2
; DMC settings. In the modes auto and iterative values chosen from the pilot runs replace timestep, steps and
; warmup_steps. walkers is used in every mode (the pilot runs use their own walker count), rounded up to a
; multiple of the MPI ranks
timestep = 0.005
walkers = 2048
steps = 500
warmup_steps = 2000
; a DMC run whose blocks take this many times longer than those of the other points is cancelled and
; submitted again once, away from its nodes (0 = never)
slow_factor = 3.0
; none = every point optimizes its own Jastrow factor from zero; centre = scan points start from the Jastrow of their centre
jastrow_reuse = centre

[trial]
; Energies that are computed AFTER the geometry search. The geometry search itself always uses one determinant.
; This stage is experimental: see docs/multireference.md, section 9.
; single   = nothing more is computed
; multidet = when the search of every target has ended: DMC energies with a trial wave function of several
;            determinants (from Quantum Package 2) at the start geometry and at the final geometry of every target
kind = single
; size of the CIPSI wave function of each geometry: a whole number, at least 2. With a list of cutoffs it is
; needed, unless every cutoff is hf. With cutoff = auto an empty value means 100000.
n_det_max =
; Needed for multidet. Either the word auto, or a list that you choose.
; auto = the package chooses the cutoffs. It starts with the first two cutoffs of auto_cutoffs and with hf, tests
;        each cut against the single determinant, tests whether every tested energy difference is converged with the
;        size of the expansion, and adds DMC series or the next cutoff of auto_cutoffs when the test needs them
;        (docs/multireference.md, section 10).
; A list = one value, or several values separated by commas. Each value is a number between 0 and 1 (Quantum
;        Package setting ci_threshold: a smaller number keeps more determinants of the CIPSI wave function), or
;        the word hf (the single RHF determinant of Quantum Package).
cutoff =
; only with cutoff = auto: the cutoffs that the package may use, in the order of their use. At least 2 numbers
; between 0 and 1, each smaller than the one before it
auto_cutoffs = 2.5e-4, 1e-4, 5e-5
; kcal/mol, larger than 0: an energy difference counts as converged between two cutoffs when the size of its
; change plus 2 standard errors is at most this value
tolerance = 1.0
; independent DMC runs for each geometry and each cutoff (at least 1). With cutoff = auto: the number at the start
series = 2
; only with cutoff = auto: the largest number of DMC runs for each geometry and each cutoff (at least series)
series_max = 4
; names of the basis file and of the pseudopotential file of Quantum Package. auto = chosen from [qmc] basis and ecp
qp2_basis = auto
qp2_pseudo = auto
; a target whose geometry search ended NOT_CONVERGED (it did not reach its precision target):
; stop = the stage does not start, and the project needs attention; run = the stage runs, and the tables mark
; every value that uses the final geometry of such a target
not_converged = stop

[search]
; classic  = line search with fitted windows (the method of the first applications)
; dense-v1 = dense symmetric scans with the registered rules of an earlier application
; dense-v2 = dense scans with the joint stopping rule and the safeguards (recommended)
; auto     = classic first; the stability audit decides whether dense-v2 continues
method = dense-v2
; tolerance on the energy of the final geometry, kcal/mol (joint 95 % statement in dense-v2)
eps_total = 0.2
; energy rise at the outer scan points, kcal/mol
e_cap = 5.0
max_iter = 20
; largest number of DMC blocks per point. Used with [qmc] mode = fixed; in the other modes the pilot runs
; set this number from sigma_target.
cap_blocks = 400

[experiment]
; Test of a wrong starting geometry (leave shift = 0 for a normal run). The search starts "shift" angstrom
; away from the surrogate minimum, along search direction "direction" (a number, softest first) of each
; target. With move_surrogate = yes the surrogate is moved with the start, so the start still looks like
; a stationary point to the surrogate: right shape, wrong place.
shift = 0.0
direction = 0
; Or: move every atom by a random displacement (no rotation or translation of the whole molecule), with a
; root mean square displacement of "random_rms" angstrom per atom. "random_seed" chooses the displacement.
random_rms = 0.0
random_seed = 0
; all = the random displacement uses every internal coordinate of the molecule; subspace = only the search
; directions (for a molecule whose search does not cover all coordinates)
random_space = all
move_surrogate = yes

[machine]
; name of a machine profile, or a path to one. A name is looked up as <name>.ini in this order: the project
; folder, $MONTEPASS_MACHINES, ~/.config/montepass/machines, machines/local/ and machines/ of the clone.
; example_slurm is the example profile, whose paths are placeholders: write the name of your own profile here.
profile = example_slurm

[report]
gif = yes
"""

_MODES = ("path", "images", "single")
_METHODS = ("classic", "dense-v1", "dense-v2", "auto")
_QMC_MODES = ("auto", "fixed", "iterative")
_TRIAL_KINDS = ("single", "multidet")
_NOT_CONVERGED = ("stop", "run")
AUTO_N_DET_MAX = 100000        # [trial] cutoff = auto with an empty n_det_max: the size of the CIPSI wave function
# [trial] qp2_basis = auto and qp2_pseudo = auto: the file of Quantum Package 2 for a PySCF name of [qmc]
_QP2_BASIS = {"ccecpccpvdz": "cc-pvdz_ecp_ccecp", "ccecpccpvtz": "cc-pvtz_ecp_ccecp",
              "ccecpccpvqz": "cc-pvqz_ecp_ccecp", "ccecpccpv5z": "cc-pv5z_ecp_ccecp"}
_QP2_PSEUDO = {"ccecp": "ccecp"}
_QP2_NAME = re.compile(r"[A-Za-z0-9_.+\-]+")


class TrialSetupError(ValueError):
    """[trial] kind = multidet cannot run with this project: a section [model], or a machine profile without what
    Quantum Package 2 needs. The commands print the text and end with exit code 1."""


class ConfigError(ValueError):
    pass


def _truth(text: str) -> bool:
    t = text.strip().lower()
    if t in ("yes", "true", "1", "on"):
        return True
    if t in ("no", "false", "0", "off", ""):
        return False
    raise ValueError("expected yes or no, found %r" % text)


def _plain_name(name: str) -> str:
    """A basis or pseudopotential name as PySCF compares names: lower case, without ``-``, ``_`` and blanks."""
    return name.lower().replace("-", "").replace("_", "").replace(" ", "")


def _trial_settings(get: Callable[[str, str], str], number: Callable[..., float], problems: List[str], charge: int,
                    multiplicity: int, basis: str, ecp: str) -> Dict[str, object]:
    """The values of section ``[trial]`` for ``kind = multidet``. Every problem is added to ``problems``.

    Returns ``{"kind": "multidet", "n_det_max", "cutoffs", "series", "qp2_basis", "qp2_pseudo", "not_converged",
    "tolerance", "series_max", "auto"}``. ``get`` and ``number`` are the readers of :class:`Config`; ``basis`` and
    ``ecp`` are the values of ``[qmc]``.

    The manual mode (``cutoff`` is a list): ``cutoffs`` is that list in the order of the input; each entry is a
    number between 0 and 1 or the word ``hf``. ``n_det_max`` is None when the input gives none. ``series_max``
    equals ``series`` and ``auto`` is None; the keys ``auto_cutoffs`` and ``series_max`` of the input are not read.

    The automatic mode (``cutoff = auto``; the word must stand alone): ``auto`` is ``{"ladder": [...]}`` with
    the numbers of ``auto_cutoffs``. ``cutoffs`` is the plan of the first pass of the stage: the first two
    cutoffs of the ladder, then ``hf``. An empty ``n_det_max`` means ``AUTO_N_DET_MAX``. ``series`` is the
    number of DMC series at the start, ``series_max`` the largest number.

    ``tolerance`` (kcal/mol) is read in both modes: the report tests the convergence with the size of the
    expansion in both."""
    from .engine import multidet as MD          # imported late: an input without this option does not need the engine

    known = len(problems)
    if charge != 0 or multiplicity != 1:
        problems.append("[trial] kind = multidet needs [system] charge = 0 and multiplicity = 1 in this version "
                        "(found charge = %d, multiplicity = %d)" % (charge, multiplicity))
    raw = get("trial", "n_det_max")
    n_det_max: Optional[int] = None
    if raw.lower() == "auto":
        problems.append("[trial] n_det_max = auto is not built yet: give a whole number (at least 2)")
    elif raw:
        try:
            n_det_max = int(raw)
        except ValueError:
            problems.append("[trial] n_det_max = %r is not a whole number" % raw)
        else:
            if n_det_max < 2:
                problems.append("[trial] n_det_max = %d must be at least 2" % n_det_max)
    words = [w.strip() for w in get("trial", "cutoff").split(",") if w.strip()]
    if not words:
        problems.append("[trial] cutoff is needed for kind = multidet: the word auto, or a list of numbers between 0 "
                        "and 1 and the word hf")
    automatic = any(w.lower() == "auto" for w in words)
    if automatic and len(words) > 1:
        problems.append("[trial] cutoff = %s: the word auto must stand alone. With auto the package chooses the cutoffs "
                        "from [trial] auto_cutoffs; a list of numbers and hf is the manual mode"
                        % get("trial", "cutoff"))

    def cutoff_list(key: str, given: List[str], hf_allowed: bool) -> List[object]:
        """The cutoffs that the words of one key name. Every word that cannot be used is a problem line."""
        out: List[object] = []
        names: Dict[str, str] = {}              # short name of a cutoff (it names folders) -> the word of the input
        for word in given:
            if word.lower() == "auto":          # the problem line for this word is written above
                continue
            if word.lower() == MD.HF_LABEL:
                if not hf_allowed:
                    problems.append("[trial] %s = %s: the word hf is not allowed here. The automatic mode always "
                                    "computes hf, the single RHF determinant" % (key, word))
                    continue
                value: object = MD.HF_LABEL
            else:
                try:
                    number_of_word = float(word)
                except ValueError:
                    problems.append("[trial] %s = %r is not a number%s" % (key, word, " and not the word hf" if hf_allowed else ""))
                    continue
                if not 0.0 < number_of_word < 1.0:
                    problems.append("[trial] %s = %s must be between 0 and 1 (0 and 1 themselves are not allowed)" % (key, word))
                    continue
                value = number_of_word
            label = MD.cut_label(value)
            if label in names:
                problems.append("[trial] %s lists the same cutoff twice: %s and %s (this package tells two cutoffs "
                                "apart by their first %d significant digits)" % (key, names[label], word, MD.CUT_DIGITS))
                continue
            names[label] = word
            out.append(value)
        return out

    auto: Optional[Dict[str, object]] = None
    if automatic and len(words) == 1:
        given = [w.strip() for w in get("trial", "auto_cutoffs").split(",") if w.strip()]
        before = len(problems)
        ladder = [float(c) for c in cutoff_list("auto_cutoffs", given, hf_allowed=False)]
        if len(problems) == before:             # every word is a usable number: now the rules for the whole list
            if len(ladder) < 2:
                problems.append("[trial] auto_cutoffs = %s holds %d cutoff(s); the automatic mode needs at least 2, "
                                "because it compares the energy differences of two cutoffs"
                                % (get("trial", "auto_cutoffs") or "(empty)", len(ladder)))
            for larger, smaller in zip(ladder, ladder[1:]):
                if not smaller < larger:
                    problems.append("[trial] auto_cutoffs = %s must be strictly decreasing: %g comes after %g, but it "
                                    "is not smaller" % (get("trial", "auto_cutoffs"), smaller, larger))
                    break
        if not raw:
            n_det_max = AUTO_N_DET_MAX
        auto = {"ladder": ladder}
        cutoffs: List[object] = ladder[:2] + [MD.HF_LABEL]
        tested: List[object] = ladder + [MD.HF_LABEL]       # every cutoff that the stage can ask the engine for
    else:
        cutoffs = cutoff_list("cutoff", words, hf_allowed=True)
        tested = list(cutoffs)
        if not raw and any(c != MD.HF_LABEL for c in cutoffs):
            problems.append("[trial] n_det_max is needed for a numeric cutoff: the size of the CIPSI wave function, a "
                            "whole number of at least 2")
    series = int(number("trial", "series", int, low=1))
    series_max = series
    if auto is not None:
        series_max = int(number("trial", "series_max", int, low=1))
        if 1 <= series_max < series:
            problems.append("[trial] series_max = %d must be at least series = %d (series is the number of DMC series "
                            "at the start, series_max the largest number)" % (series_max, series))
    raw_tolerance = get("trial", "tolerance")
    tolerance = 0.0
    try:
        tolerance = float(raw_tolerance)
    except ValueError:
        problems.append("[trial] tolerance = %r is not a number" % raw_tolerance)
    else:
        if not 0.0 < tolerance < float("inf"):
            problems.append("[trial] tolerance = %s must be larger than 0 (kcal/mol: the largest change of an energy "
                            "difference between two cutoffs that counts as converged)" % raw_tolerance)
    not_converged = get("trial", "not_converged").lower()
    if not_converged not in _NOT_CONVERGED:
        problems.append("[trial] not_converged = %r; allowed: %s" % (get("trial", "not_converged"), ", ".join(_NOT_CONVERGED)))
    files: Dict[str, str] = {}
    for key, table, source_key, source in (("qp2_basis", _QP2_BASIS, "basis", basis),
                                           ("qp2_pseudo", _QP2_PSEUDO, "ecp", ecp)):
        chosen = get("trial", key)
        if chosen.lower() in ("", "auto"):
            chosen = table.get(_plain_name(source), "")
            if not chosen:
                problems.append("[trial] %s = auto: this package knows no file of Quantum Package for [qmc] %s = %s "
                                "(it knows: %s). Set [trial] %s to the name of the file of Quantum Package"
                                % (key, source_key, source, ", ".join(sorted(table)), key))
        elif not _QP2_NAME.fullmatch(chosen):
            problems.append("[trial] %s = %r is not a file name of Quantum Package (allowed: letters, digits and "
                            "the characters _ . + -)" % (key, chosen))
        files[key] = chosen
    if len(problems) == known:
        # the check of the engine itself: an input that is accepted here must be accepted there
        for cutoff in tested:
            try:
                MD.validate_trial(MD.trial_of({"trial": dict(files, kind="multidet", cutoff=cutoff, n_det_max=n_det_max)}),
                                  charge, multiplicity - 1)
            except ValueError as exc:
                problems.append("[trial] %s" % exc)
                break
    return {"kind": "multidet", "n_det_max": n_det_max, "cutoffs": cutoffs, "series": series,
            "qp2_basis": files["qp2_basis"], "qp2_pseudo": files["qp2_pseudo"], "not_converged": not_converged,
            "tolerance": tolerance, "series_max": series_max, "auto": auto}


class Config:
    """Checked values of an input file. File names are absolute paths."""

    def __init__(self, path: str):
        self.path = os.path.abspath(path)
        self.folder = os.path.dirname(self.path)
        default = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=(";",))
        default.read_string(TEMPLATE)
        cp = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=(";",))
        cp.read_string(TEMPLATE)
        if not os.path.exists(self.path):
            raise ConfigError("input file not found: %s" % self.path)
        user = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=(";",))
        try:
            user.read(self.path)
        except configparser.Error as exc:
            raise ConfigError("%s: %s" % (self.path, exc))
        problems: List[str] = []
        for sec in user.sections():
            if not default.has_section(sec) and sec != "model":
                problems.append("unknown section [%s]" % sec); continue
            if sec == "model" and not cp.has_section("model"):
                cp.add_section("model")
            for key, val in user.items(sec):
                if sec != "model" and not default.has_option(sec, key):
                    problems.append("[%s] unknown key %r" % (sec, key)); continue
                cp.set(sec, key, val)
        self.cp = cp

        def get(sec, key):
            return cp.get(sec, key).strip()

        def number(sec, key, kind=float, low=None):
            try:
                v = kind(get(sec, key))
            except ValueError:
                problems.append("[%s] %s = %r is not a number" % (sec, key, get(sec, key))); return kind(0)
            if low is not None and v < low:
                problems.append("[%s] %s = %s must be at least %s" % (sec, key, v, low))
            return v

        def flag(sec, key):
            try:
                return _truth(get(sec, key))
            except ValueError as exc:
                problems.append("[%s] %s: %s" % (sec, key, exc)); return False

        def choice(sec, key, allowed):
            v = get(sec, key).lower()
            if v not in [a.lower() for a in allowed]:
                problems.append("[%s] %s = %r; allowed: %s" % (sec, key, get(sec, key), ", ".join(allowed)))
            return v

        def file(sec, key, needed):
            v = get(sec, key)
            if is_model:
                return None                     # a model run builds its own geometries
            if not needed and v == default.get(sec, key).strip() and not os.path.exists(os.path.join(self.folder, v)):
                return None                     # an unused default name
            if not v:
                if needed:
                    problems.append("[%s] %s is needed for mode %s" % (sec, key, self.mode))
                return None
            p = v if os.path.isabs(v) else os.path.join(self.folder, v)
            if not os.path.exists(p):
                problems.append("[%s] %s: file not found: %s" % (sec, key, p))
            return p

        self.name = get("system", "name") or "project"
        self.charge = number("system", "charge", int)
        self.multiplicity = number("system", "multiplicity", int, low=1)
        self.spin = self.multiplicity - 1                      # number of unpaired electrons (2S)

        self.model = dict(cp.items("model")) if cp.has_section("model") else None
        is_model = self.model is not None
        self.mode = choice("geometry", "mode", _MODES)
        self.reactant = file("geometry", "reactant", self.mode == "path" and not is_model)
        self.product = file("geometry", "product", self.mode == "path" and not is_model)
        self.saddle_guess = file("geometry", "saddle_guess", False)
        self.images = file("geometry", "images", self.mode == "images" and not is_model)
        self.geometry = file("geometry", "geometry", self.mode == "single" and not is_model)
        self.targets = [t.strip() for t in get("geometry", "targets").split(",") if t.strip()]
        if not self.targets:
            problems.append("[geometry] targets is empty")
        # "list": the targets are named one by one. "barrier" and "mep": the package takes them from the path.
        self.selection = "list"
        if len(self.targets) == 1 and self.targets[0].lower() in ("barrier", "mep"):
            self.selection = self.targets[0].lower()
        if self.selection != "list":
            if self.mode == "path":
                # the images between the ends are added when the path exists (selection mep)
                self.targets = ["reactant", "saddle"] if self.selection == "barrier" else ["reactant", "saddle", "product"]
            elif self.mode == "images":
                self.targets = []                   # image numbers: known when the path file is read
            else:
                problems.append("[geometry] targets = %s needs a path: use mode path or mode images" % self.selection)
                self.targets = ["geometry"]
        elif self.mode == "path":
            bad = [t for t in self.targets if t not in ("reactant", "product", "saddle")]
            if bad:
                problems.append("[geometry] targets %s are not allowed in mode path (use reactant, product, saddle)" % bad)
        elif self.mode == "images":
            bad = [t for t in self.targets if not t.isdigit()]
            if bad:
                problems.append("[geometry] targets %s must be image numbers in mode images" % bad)
        else:
            if self.targets != ["geometry"]:
                self.targets = ["geometry"]
        kinds = [k.strip().lower() for k in get("geometry", "kinds").split(",") if k.strip()]
        self.kinds_auto = False                     # mode images: the kinds follow from the place on the path
        self.saddle_image: Optional[int] = None     # None = the image with the highest energy in the path file
        raw_saddle = get("geometry", "saddle_image").lower()
        if raw_saddle not in ("", "auto"):
            if raw_saddle.isdigit():
                self.saddle_image = int(raw_saddle)
            else:
                problems.append("[geometry] saddle_image = %r is not an image number or auto" % get("geometry", "saddle_image"))
        if self.mode == "path":
            self.kinds = {t: ("saddle" if t == "saddle" else "minimum") for t in self.targets}
        elif self.mode == "images" and (self.selection != "list" or kinds == ["auto"]):
            if self.selection != "list" and kinds not in ([], ["auto"]):
                problems.append("[geometry] kinds: targets = %s chooses the kinds itself; leave kinds empty" % self.selection)
            self.kinds_auto = True
            self.kinds = {}
        else:
            if kinds and len(kinds) != len(self.targets):
                problems.append("[geometry] kinds has %d entries for %d targets" % (len(kinds), len(self.targets)))
            if self.mode == "images":
                if any(k not in ("minimum", "saddle", "image") for k in kinds):
                    problems.append("[geometry] kinds: allowed values are minimum, saddle and image")
            elif any(k not in ("minimum", "saddle") for k in kinds):
                problems.append("[geometry] kinds: allowed values are minimum and saddle")
            self.kinds = {t: (kinds[i] if i < len(kinds) else "minimum") for i, t in enumerate(self.targets)}

        self.surrogate_program = choice("surrogate", "program", ("orca", "none"))
        self.surrogate_method = get("surrogate", "method")
        self.surrogate_basis = get("surrogate", "basis")
        self.surrogate_optimize = flag("surrogate", "optimize")
        self.surrogate_frequencies = flag("surrogate", "frequencies")
        self.neb_images = number("surrogate", "neb_images", int, low=3)
        self.neb_kind = get("surrogate", "neb_kind").upper()
        if self.neb_kind not in ("NEB-TS", "NEB-CI"):
            problems.append("[surrogate] neb_kind = %r; allowed: NEB-TS, NEB-CI" % get("surrogate", "neb_kind"))
        if self.mode == "path" and self.surrogate_program == "none" and not is_model:
            problems.append("[surrogate] program = none cannot build a path: use mode = images with your own path")

        self.subspace_kind = choice("subspace", "kind", ("auto", "pca", "modes"))
        dim = get("subspace", "dim").lower()
        self.subspace_dim: Optional[int] = None
        if dim != "auto":
            try:
                self.subspace_dim = int(dim)
                if self.subspace_dim < 1:
                    problems.append("[subspace] dim must be at least 1")
            except ValueError:
                problems.append("[subspace] dim = %r is not a number or auto" % dim)
        self.subspace_variance = number("subspace", "variance", float, low=0.0)
        raw_max = get("subspace", "max_dim").strip().lower()
        self.subspace_max_dim: Optional[int] = None                 # None = auto: 6 for pca, 12 for modes
        if raw_max != "auto":
            self.subspace_max_dim = number("subspace", "max_dim", int, low=1)
        if self.mode == "single" and self.subspace_kind == "pca":
            problems.append("[subspace] kind = pca needs a path; mode single has none")

        self.qmc_mode = choice("qmc", "mode", _QMC_MODES)
        self.qmc = {
            "charge": self.charge, "spin": self.spin,
            "xc": get("qmc", "xc"), "basis": get("qmc", "basis"), "ecp": get("qmc", "ecp"),
            "jastrow_reuse": choice("qmc", "jastrow_reuse", ("none", "centre")),
            "dmc": {"timestep": number("qmc", "timestep", float, low=1e-6), "walkers": number("qmc", "walkers", int, low=1),
                    "steps": number("qmc", "steps", int, low=1), "warmup_steps": number("qmc", "warmup_steps", int, low=0)},
            "slow_factor": number("qmc", "slow_factor", float, low=0.0),
        }
        self.qmc_max_repeats = number("qmc", "max_repeats", int, low=0)
        self.sigma_target = number("qmc", "sigma_target", float, low=1e-6)

        # energies after the search. None = no such energies; then the other keys of [trial] are not read.
        self.trial: Optional[Dict[str, object]] = None
        if choice("trial", "kind", _TRIAL_KINDS) == "multidet":
            self.trial = _trial_settings(get, number, problems, self.charge, self.multiplicity,
                                         self.qmc["basis"], self.qmc["ecp"])

        self.search_method = choice("search", "method", _METHODS)
        self.eps_total = number("search", "eps_total", float, low=1e-6)
        self.e_cap = number("search", "e_cap", float, low=1e-3)
        self.max_iter = number("search", "max_iter", int, low=1)
        self.cap_blocks = number("search", "cap_blocks", int, low=50)

        self.shift = number("experiment", "shift", float)
        self.shift_direction = number("experiment", "direction", int, low=0)
        self.random_rms = number("experiment", "random_rms", float, low=0.0)
        self.random_seed = number("experiment", "random_seed", int, low=0)
        self.random_space = choice("experiment", "random_space", ("all", "subspace"))
        if self.shift != 0.0 and self.random_rms > 0.0:
            problems.append("[experiment] shift and random_rms are both set: use one of them")
        self.move_surrogate = flag("experiment", "move_surrogate")

        self.machine_profile = get("machine", "profile")
        self.gif = flag("report", "gif")
        if problems:
            raise ConfigError("%s:\n  " % self.path + "\n  ".join(problems))

    @staticmethod
    def load(path: str) -> "Config":
        return Config(path)

    def summary(self) -> Dict[str, object]:
        """The values that define the calculation, for the record in the results. The key ``trial`` exists only
        with ``[trial] kind = multidet``."""
        out: Dict[str, object] = {
            "name": self.name, "charge": self.charge, "multiplicity": self.multiplicity, "mode": self.mode,
            "targets": self.targets, "selection": self.selection, "surrogate": {"program": self.surrogate_program, "method": self.surrogate_method,
                                                   "basis": self.surrogate_basis, "optimize": self.surrogate_optimize,
                                                   "frequencies": self.surrogate_frequencies,
                                                   "neb_images": self.neb_images, "neb_kind": self.neb_kind},
            "subspace": {"kind": self.subspace_kind, "dim": self.subspace_dim, "variance": self.subspace_variance,
                         "max_dim": self.subspace_max_dim},
            "qmc": dict(self.qmc, mode=self.qmc_mode, sigma_target=self.sigma_target),
            "search": {"method": self.search_method, "eps_total": self.eps_total, "e_cap": self.e_cap,
                       "max_iter": self.max_iter, "cap_blocks": self.cap_blocks},
            "experiment": {"shift": self.shift, "direction": self.shift_direction, "random_rms": self.random_rms,
                           "random_seed": self.random_seed, "random_space": self.random_space,
                           "move_surrogate": self.move_surrogate},
            "machine": self.machine_profile, "model": self.model,
        }
        if self.trial is not None:
            out["trial"] = dict(self.trial, cutoffs=list(self.trial["cutoffs"]))
        return out


def write_template(path: str, overwrite: bool = False) -> None:
    if os.path.exists(path) and not overwrite:
        raise FileExistsError("%s exists; it was not overwritten" % path)
    with open(path, "w") as fh:
        fh.write(TEMPLATE)
