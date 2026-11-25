"""Automatic choice of the QMC settings.

The module answers one question: which time step, how many walkers, how many steps per block and
how many blocks does a DMC run of this molecule need for a wanted error bar? It has two layers.

1. Planning functions. Plain arithmetic, usable before any QMC run:
   :func:`forecast`, :func:`jastrow_defaults`, :func:`walkers_for`, :func:`choose_timestep`,
   :func:`plan_dmc`, :func:`scan_point_sigma`, :func:`check_settings`.
2. :class:`Pilot`, a small driver. It runs short DMC runs at a few time steps through an engine,
   measures what the formulas need, applies the planning functions and writes the result to
   ``qmc_settings.json`` together with the measurement behind every value.

Notation: N_e valence electrons, R MPI ranks, N_w walkers, tau time step (Ha^-1), s steps per
block, B blocks, T_b = s * tau the length of a block in imaginary time, V variance of the local
energy (Ha^2), sigma the wanted error bar. T_c is the *effective correlation time*, defined by

    SEM^2 = 2 T_c V / (N_w T)          (SEM = standard error of the mean after T Ha^-1 of data)

Formulas

    F1  imaginary time for an error bar:   T = 2 T_c V / (N_w sigma^2)                (sigma in hartree)
    F2  blocks to request:                 B = n_discard + 1.2 T / T_b
    F3  first guesses before any QMC run:  V = 0.022 |E_SCF|,  T_c = 0.14 Ha^-1,
                                           t_ws = 5.4e-4 s * (N_e / 28)^1.9,  transient 10 Ha^-1
    F4  cost:                              core-hours = B s N_w t_ws / 3600
    F5  time step of 99 % acceptance:      tau_99 = tau_0 * (0.01 / (1 - a_0))^(1 / 1.28)
    F6  time step allowed by the bias:     tau <= eps_tau / (f_geo |k_E|)
    F7  dense scan, shells at fractions f_i of the span: a point error sigma_pt costs
        sigma_pt^2 / (8 E_cap sum f_i^2) of energy per direction

t_ws is the computer time for one step of one walker in core-seconds. k_E is the slope of the
total DMC energy against the time step (kcal/mol per Ha^-1). f_geo = 0.15 is the measured ratio
between the time-step slope of an energy difference of two nearby geometries and k_E.

Where the defaults come from. They were measured on 188 production DMC traces of 9 closed-shell
molecules with 28 to 62 valence electrons (elements H, C, N, O, S; ccECP pseudopotentials; one
B3LYP determinant with a one-body and two-body Jastrow factor; 2048 walkers; legacy DMC driver
with T-moves). Measured ranges: V / |E| = 0.0204 to 0.0228 Ha; T_c = 0.114 to 0.158 Ha^-1; DMC
acceptance at tau = 0.005: 0.9976 to 0.9984; no start-up excess after 10 Ha^-1; k_E = -270 to
-380 kcal/mol per Ha^-1. One run of water (8 electrons) agrees with F3 for the variance (0.379 Ha^2
against 0.379) and gives T_c = 0.13 Ha^-1; its t_ws was 1.8 times the guess. Rules that have no
measurement behind them yet are marked "not tested" where they are defined.

What an engine should report. The pilot reads the DMC block energies and weights from
``PointData.energy`` and ``PointData.weight``. Everything else is optional and is taken from
``PointData.meta`` when present:

    scalar_file / trace           path of the DMC ``scalar.dat`` file, or its columns as a dict
                                  (gives the acceptance, the variance and the seconds per block)
    vmc_scalar_file / vmc_trace   the same for the VMC run that made the walkers (VMC variance)
    acceptance, variance, vmc_variance, seconds_per_block
                                  the same quantities as plain numbers (used when no trace is given)
    timestep, steps, walkers      the settings that the run really used (compared with the request)

A quantity that is missing is never invented without a note: the rule that needs it is recorded
as "not checked", or the first guess F3 is used and recorded as a guess.
"""
from __future__ import annotations

import copy
import inspect
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import monitor, stats
from .engine.base import FAILED, Engine, PointData, PointSpec
from .engine.qmcpack_inputs import DEFAULT_JASTROW, dmc_input
from .geometry import normalize_symbol, valence_electrons
from .search.protocol_v2 import joint_factor
from .units import ANGSTROM_TO_BOHR, HARTREE_TO_KCAL

# ----------------------------------------------------------------------------- defaults (see the module text)
VARIANCE_PER_HARTREE = 0.022        # F3: variance / |E_SCF| in hartree
T_CORR = 0.14                       # F3: effective correlation time, Ha^-1
STEP_SECONDS_28 = 5.4e-4            # F3: core-seconds for one step of one walker at 28 electrons
STEP_SECONDS_EXPONENT = 1.9         # F3: growth of that time with the electron count
TRANSIENT = 10.0                    # Ha^-1 of start-up that is discarded
BLOCK_TIME = 2.5                    # Ha^-1 per block (20 to 30 fast correlation times)
SAFETY = 1.2                        # F2: factor on the blocks (a pilot gives the noise to about 10 %)
MIN_TRANSIENTS = 5.0                # a run is at least this many transients long
MIN_KEPT_BLOCKS = 64                # a run keeps at least this many blocks, so that its error bar can be estimated
WALL_FACTOR = 1.3                   # wall time to request = this factor times the expected run time
ACCEPTANCE_MIN = 0.99               # R1
ACCEPTANCE_EXPONENT = 1.28          # F5: 1 - acceptance is proportional to tau^1.28 (measured 1.27 to 1.31)
REJECTED_AT_0005 = 0.0025           # first guess of 1 - acceptance at tau = 0.005 (measured 0.0016 to 0.0027)
VARIANCE_RATIO_MAX = 1.10           # R2: DMC variance / VMC variance (measured 0.995 to 1.025)
SPREAD_MAX = 0.05                   # R3: relative spread of the population at 2048 walkers (measured 0.016 to 0.019)
F_GEO = 0.15                        # R4, F6 (measured 0.14 for the rms of 5 energy differences; largest single ratio 0.28)
SLOPE_RELATIVE_ERROR = 0.25         # R4: a slope with a larger relative error counts as not resolved
SLOPE_GUESS = 340.0                 # kcal/mol per Ha^-1: |k_E| used when no slope was measured (largest measured mean)
WALKERS_REFERENCE = 2048            # walker count of all measurements
WALKER_TARGET = {"final": 2048, "centre": 2048, "scan": 2048, "pilot": 512}
PILOT_WALKERS_PER_RANK = 4          # smallest number of pilot walkers per rank
CANDIDATES = (0.02, 0.01, 0.005)    # time steps of the pilot, Ha^-1
TESTED_ELEMENTS = ("H", "C", "N", "O", "S")
MODES = ("fixed", "auto", "iterative")

RUNNING = "RUNNING"
DONE = "DONE"
NEEDS_ATTENTION = "NEEDS_ATTENTION"

PASS, FAIL, NOT_CHECKED = "pass", "fail", "not checked"


def _round_sig(x: float, digits: int = 1, down: bool = False) -> float:
    """``x`` with ``digits`` significant digits, rounded to the nearest value or down."""
    if not x > 0.0 or not math.isfinite(x):
        raise ValueError("cannot round %r to significant digits" % x)
    exp = math.floor(math.log10(x)) - (digits - 1)
    f = x / 10.0 ** exp
    f = math.floor(f + 1e-9) if down else math.floor(f + 0.5)
    return float("%.12g" % (f * 10.0 ** exp))


def _steps_for(block_time: float, timestep: float) -> int:
    return max(1, int(math.floor(block_time / timestep + 0.5)))


# ----------------------------------------------------------------------------- stage 0: facts and first guesses
def step_seconds_guess(n_electrons: int) -> float:
    """F3: core-seconds for one DMC step of one walker, 5.4e-4 * (N_e / 28)^1.9.

    Fitted for 28 to 62 electrons (scatter 5 %). For water (8 electrons) the measured value was
    1.8 times this guess, because fixed costs do not shrink with the molecule."""
    if n_electrons < 1:
        raise ValueError("the number of electrons must be positive, got %r" % n_electrons)
    return STEP_SECONDS_28 * (float(n_electrons) / 28.0) ** STEP_SECONDS_EXPONENT


def jastrow_defaults(coords: np.ndarray) -> Dict[str, object]:
    """Sizes and cutoffs of the one-body (J1) and two-body (J2) B-spline Jastrow factor from the
    size of the molecule.

    With D = the largest distance between two atoms in bohr:

        J1 cutoff = min(6, max(4, D + 2)) bohr,   J2 cutoff = min(8, max(5, D + 4)) bohr,

    rounded to 0.1 bohr, and one knot per bohr (rounded up). Molecules with D of 4 bohr or more
    get 6 and 8 bohr with 6 and 8 knots: the values of all production runs. Water (D = 2.9 bohr)
    gets 4.9 and 6.9 bohr with 5 and 7 knots. The shorter cutoffs of small molecules are NOT
    tested: the one water run so far used 6 and 8 bohr. The keys are those of
    ``engine.qmcpack_inputs.DEFAULT_JASTROW``.
    """
    X = np.asarray(coords, dtype=float).reshape(-1, 3) * ANGSTROM_TO_BOHR
    if len(X) == 0:
        raise ValueError("jastrow_defaults: no atoms")
    diff = X[:, None, :] - X[None, :, :]
    d = float(np.sqrt((diff ** 2).sum(axis=2)).max())
    r1 = round(min(6.0, max(4.0, d + 2.0)), 1)
    r2 = round(min(8.0, max(5.0, d + 4.0)), 1)
    return {"j1_size": int(math.ceil(r1 - 1e-9)), "j1_rcut": r1, "j2_size": int(math.ceil(r2 - 1e-9)), "j2_rcut": r2}


def walkers_for(ranks: int, role: str = "final", target: Optional[int] = None) -> int:
    """DMC target population for a job with ``ranks`` MPI ranks: the smallest multiple of the rank
    count that is not below the target of the role.

    Targets: 2048 for the roles ``final``, ``centre`` and ``scan``; 512 for ``pilot``, with at
    least 4 walkers per rank. A population that the rank count does not divide makes QMCPACK
    change the population on its own. Energies of scan, centre and final points are subtracted
    from each other, so these roles share one walker count. 2048 is the only walker count with
    measurements behind it; 512 for pilots is not tested for a population-control bias, and a
    pilot result is never subtracted from a production energy.
    """
    ranks = int(ranks)
    if ranks < 1:
        raise ValueError("walkers_for: the number of ranks must be at least 1, got %d" % ranks)
    if target is None:
        if role not in WALKER_TARGET:
            raise ValueError("walkers_for: unknown role %r (allowed: %s)" % (role, ", ".join(sorted(WALKER_TARGET))))
        target = WALKER_TARGET[role]
    if int(target) < 1:
        raise ValueError("walkers_for: the target population must be at least 1, got %r" % target)
    per_rank = int(math.ceil(int(target) / float(ranks)))
    if role == "pilot":
        per_rank = max(per_rank, PILOT_WALKERS_PER_RANK)
    return per_rank * ranks


@dataclass
class DmcPlan:
    """Sizes of one DMC run (result of :func:`plan_dmc`)."""

    timestep: float                     # Ha^-1
    walkers: int
    steps: int                          # steps per block
    block_time: float                   # Ha^-1 per block = steps * timestep
    warmup_steps: int                   # value of the DMC input: the transient in steps
    blocks: int                         # blocks to request, the discarded ones included
    discard: int                        # start-up blocks that the analysis discards
    time: float                         # Ha^-1 of data in the kept blocks
    sigma_target: Optional[float]       # kcal/mol; None when the blocks were given
    sigma_expected: float               # kcal/mol: error bar expected from the kept blocks
    sigma_block: float                  # kcal/mol: error of one block (correlation between blocks included)
    noise: float                        # 2 T_c V in hartree, the measured input of F1
    core_hours: Optional[float]         # F4; None when the time per step is not known
    wall_seconds: Optional[float]       # wall time to request; None when the time per step or the ranks are not known
    notes: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, object]:
        return stats.plain(asdict(self))


def plan_dmc(timestep: float, walkers: int, sigma_target: Optional[float] = None, blocks: Optional[int] = None,
             variance: Optional[float] = None, t_corr: Optional[float] = None, noise: Optional[float] = None,
             block_time: float = BLOCK_TIME, transient: float = TRANSIENT, safety: float = SAFETY,
             step_seconds: Optional[float] = None, ranks: Optional[int] = None, setup_seconds: float = 0.0,
             wall_factor: float = WALL_FACTOR) -> DmcPlan:
    """Size one DMC run.

    Give exactly one of

    * ``sigma_target`` (kcal/mol): the blocks follow from F1 and F2;
    * ``blocks`` (for example the block limit of a search protocol): the expected error bar follows.

    The statistical input is the product 2 T_c V. Give ``variance`` (Ha^2) and ``t_corr`` (Ha^-1),
    or give ``noise`` = 2 T_c V in hartree directly. A pilot run measures it without knowing V:
    noise = SEM^2 * walkers * (imaginary time of the data).

    Rules: steps per block = ``block_time`` / time step (2.5 Ha^-1); warm-up steps = ``transient``
    / time step (10 Ha^-1); discarded blocks = ceil(transient / block length). With
    ``sigma_target`` the run is never shorter than 5 transients and never keeps fewer than 64
    blocks; a note says when one of these limits made the run longer than the error bar needs.
    Cost (F4) needs ``step_seconds`` (core-seconds per walker and step). The wall time is
    ``wall_factor`` (1.3) times blocks * steps * walkers * step_seconds / ranks, plus ``setup_seconds``.
    """
    if (sigma_target is None) == (blocks is None):
        raise ValueError("plan_dmc: give exactly one of sigma_target and blocks")
    if not timestep > 0.0 or walkers < 1:
        raise ValueError("plan_dmc: time step %r and walkers %r must be positive" % (timestep, walkers))
    if noise is None:
        if variance is None or t_corr is None:
            raise ValueError("plan_dmc: give noise (2 T_c V), or both variance and t_corr")
        noise = 2.0 * float(t_corr) * float(variance)
    if not noise > 0.0 or not math.isfinite(noise):
        raise ValueError("plan_dmc: the noise 2 T_c V must be a positive number, got %r" % noise)
    steps = _steps_for(block_time, timestep)
    tb = steps * timestep
    warmup = int(math.ceil(transient / timestep - 1e-9))
    discard = stats.transient_blocks(steps, timestep, transient)
    notes: List[str] = []
    if sigma_target is not None:
        if not sigma_target > 0.0:
            raise ValueError("plan_dmc: sigma_target must be positive, got %r" % sigma_target)
        s_ha = sigma_target / HARTREE_TO_KCAL
        need = noise / (walkers * s_ha ** 2)                                   # F1
        n_data = int(math.ceil(safety * need / tb - 1e-9))                     # F2
        low = max(MIN_KEPT_BLOCKS, int(math.ceil(MIN_TRANSIENTS * transient / tb - 1e-9)) - discard)
        if n_data < low:
            notes.append("the error bar needs %d kept blocks; the run keeps %d (at least %d kept blocks and %g "
                         "transients). Fewer walkers would reach the same error bar at a lower cost."
                         % (n_data, low, MIN_KEPT_BLOCKS, MIN_TRANSIENTS))
            n_data = low
        blocks = discard + n_data
    else:
        blocks = int(blocks)
        n_data = blocks - discard
        if n_data < 1:
            raise ValueError("plan_dmc: %d blocks, but %d are discarded as transient (%g Ha^-1 at %g Ha^-1 per block)"
                             % (blocks, discard, transient, tb))
        if n_data < MIN_KEPT_BLOCKS:
            notes.append("only %d blocks are kept; below %d the error bar of the run is not reliable" % (n_data, MIN_KEPT_BLOCKS))
        if blocks * tb < MIN_TRANSIENTS * transient:
            notes.append("the run is %.1f Ha^-1 long, shorter than %g transients (%g Ha^-1)"
                         % (blocks * tb, MIN_TRANSIENTS, MIN_TRANSIENTS * transient))
    data_time = n_data * tb
    sigma_expected = math.sqrt(noise / (walkers * data_time)) * HARTREE_TO_KCAL
    sigma_block = math.sqrt(noise / (walkers * tb)) * HARTREE_TO_KCAL
    core_hours = wall = None
    if step_seconds is not None:
        core_seconds = blocks * steps * walkers * float(step_seconds)
        core_hours = core_seconds / 3600.0                                     # F4
        if ranks:
            wall = wall_factor * core_seconds / int(ranks) + setup_seconds
    return DmcPlan(timestep=float(timestep), walkers=int(walkers), steps=steps, block_time=tb, warmup_steps=warmup,
                   blocks=int(blocks), discard=discard, time=data_time, sigma_target=sigma_target,
                   sigma_expected=sigma_expected, sigma_block=sigma_block, noise=float(noise), core_hours=core_hours,
                   wall_seconds=wall, notes=notes)


@dataclass
class Forecast:
    """Result of :func:`forecast`. ``guesses`` names the inputs that are first guesses (F3)."""

    n_electrons: int
    variance: float                     # Ha^2
    t_corr: float                       # Ha^-1
    step_seconds: float                 # core-seconds per walker and step
    sigma_target: float                 # kcal/mol
    walkers: int
    timestep: float
    steps: int                          # steps per block
    time: float                         # Ha^-1 of data needed (F1)
    discard: int
    blocks: int                         # F2
    sigma_block: float                  # kcal/mol
    core_hours: float                   # F4
    guesses: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, object]:
        return stats.plain(asdict(self))


def forecast(n_electrons: int, e_scf: float, sigma_target: float = 0.2, walkers: int = 2048, timestep: float = 0.005,
             block_time: float = BLOCK_TIME, transient: float = TRANSIENT, safety: float = SAFETY,
             variance: Optional[float] = None, t_corr: Optional[float] = None,
             step_seconds: Optional[float] = None) -> Forecast:
    """Size and cost of a DMC run before any QMC run (F1 to F4).

    ``e_scf`` is the SCF energy of the pseudopotential calculation in hartree (the valence
    energy, for example -17.2 Ha for water with ccECP; NOT an all-electron energy).
    ``sigma_target`` is in kcal/mol. A measured ``variance``, ``t_corr`` or ``step_seconds``
    replaces the first guess of F3. On the 9 molecules of the measurements the forecast of the
    blocks was within about 15 % of the measured need. ``time`` is the imaginary time of F1
    without the safety factor; ``blocks`` includes the safety factor and the discarded blocks.
    """
    guesses = []
    if variance is None:
        variance = VARIANCE_PER_HARTREE * abs(float(e_scf)); guesses.append("variance")
    if t_corr is None:
        t_corr = T_CORR; guesses.append("t_corr")
    if step_seconds is None:
        step_seconds = step_seconds_guess(n_electrons); guesses.append("step_seconds")
    if not variance > 0.0:
        raise ValueError("forecast: the variance must be positive (e_scf = %r)" % e_scf)
    plan = plan_dmc(timestep, walkers, sigma_target=sigma_target, variance=variance, t_corr=t_corr,
                    block_time=block_time, transient=transient, safety=safety, step_seconds=step_seconds)
    need = 2.0 * t_corr * variance / (walkers * (sigma_target / HARTREE_TO_KCAL) ** 2)
    return Forecast(n_electrons=int(n_electrons), variance=float(variance), t_corr=float(t_corr),
                    step_seconds=float(step_seconds), sigma_target=float(sigma_target), walkers=int(walkers),
                    timestep=float(timestep), steps=plan.steps, time=need, discard=plan.discard, blocks=plan.blocks,
                    sigma_block=plan.sigma_block, core_hours=float(plan.core_hours), guesses=guesses)


# ----------------------------------------------------------------------------- dense scan: error bar of a point
def scan_noise_energy(n_directions: int, sigma_point: float, e_cap: float = 5.0,
                      shells: Sequence[float] = (0.5, 1.0)) -> float:
    """F7: energy (kcal/mol) that the statistical error of the composed step is expected to cost,
    for ``n_directions`` directions scanned with a point error of ``sigma_point`` kcal/mol:

        n * sigma_point^2 / (8 * e_cap * sum f_i^2)

    ``shells`` holds the positions f_i of the scan points as fractions of the span
    R = sqrt(e_cap / Lambda). The result does not depend on the curvature. For the shells
    (0.5, 1) it is n * sigma_point^2 / (10 * e_cap)."""
    f2 = float(sum(float(f) ** 2 for f in shells))
    if n_directions < 1 or not f2 > 0.0 or not e_cap > 0.0:
        raise ValueError("scan_noise_energy: need at least 1 direction, a positive e_cap and shells")
    return n_directions * sigma_point ** 2 / (8.0 * e_cap * f2)


def scan_point_sigma(n_directions: int, e_cap: float = 5.0, eps_total: float = 0.2, shells: Sequence[float] = (1.0,),
                     rule: str = "joint", q: float = 1.0, n_negative: int = 0, e_cap_negative: float = 0.5) -> float:
    """Error bar (kcal/mol) that every point of a dense scan needs, so that the search can end
    with the energy tolerance ``eps_total`` for ``n_directions`` directions.

    rule = "joint" (protocol v2). The search ends when c_n * N < eps_total, where N is the noise
    energy of F7 and c_n = chi2(0.95, n) / n turns it into a joint 95 % bound:

        sigma_pt = sqrt(8 * e_cap * sum f_i^2 * eps_total / (c_n * n))

    For one shell at f = 1 this is the same condition as ``ProtocolV2.planned_blocks``. ``n_negative`` of the
    directions are reaction coordinates (negative curvature, a saddle search). Protocol v2 scans them at the
    energy ``e_cap_negative`` in place of ``e_cap``, so each adds e_cap / e_cap_negative times the noise energy
    of another direction; the factor c is then that of the effective number of directions
    (``ProtocolV2.plan_noise``).

    rule = "rule_b" (protocol v1). A direction with no displacement stops when its bound
    Lambda * (2 sigma_m)^2 is below eps_total / n. With a margin of ``q`` sigma (q = 1: 84 % of
    such directions stop):

        sigma_pt = sqrt(8 * e_cap * sum f_i^2 * eps_total / n) / (2 + q)

    Example: n = 6, e_cap = 5, eps_total = 0.2, shells (0.5, 1), rule_b: 0.43 kcal/mol.
    """
    f2 = float(sum(float(f) ** 2 for f in shells))
    if n_directions < 1 or not f2 > 0.0 or not e_cap > 0.0 or not eps_total > 0.0:
        raise ValueError("scan_point_sigma: need at least 1 direction and positive e_cap, eps_total and shells")
    base = 8.0 * e_cap * f2 * eps_total / n_directions
    if rule == "joint":
        if n_negative:
            from .search.protocol_v2 import ProtocolV2
            total, factor = ProtocolV2(e_cap=float(e_cap), e_cap_negative=float(e_cap_negative),
                                       eps_total=float(eps_total)).plan_noise(n_directions, n_negative)
            return math.sqrt(f2 * eps_total / (factor * total))
        return math.sqrt(base / joint_factor(n_directions))
    if rule == "rule_b":
        return math.sqrt(base) / (2.0 + q)
    raise ValueError("scan_point_sigma: unknown rule %r (allowed: joint, rule_b)" % rule)


# ----------------------------------------------------------------------------- stage 3: the time step
def timestep_for_acceptance(timestep0: float, acceptance0: float, acceptance: float = ACCEPTANCE_MIN,
                            exponent: float = ACCEPTANCE_EXPONENT) -> float:
    """F5: the time step at which the DMC acceptance is ``acceptance`` (0.99), from one run at
    ``timestep0`` with acceptance ``acceptance0``. The rejected fraction grows like tau^1.28."""
    if not (0.0 < acceptance0 < 1.0) or not (0.0 < acceptance < 1.0) or not timestep0 > 0.0:
        raise ValueError("timestep_for_acceptance: time step %r and acceptances %r, %r must lie between 0 and 1"
                         % (timestep0, acceptance0, acceptance))
    return float(timestep0 * ((1.0 - acceptance) / (1.0 - acceptance0)) ** (1.0 / exponent))


def candidate_timesteps(tau_99: float) -> Tuple[float, float, float]:
    """The three time steps of a pilot: tau_99 rounded to one digit, half of it and a quarter of it.
    For tau_99 between 0.015 and 0.025 this is 0.02, 0.01, 0.005."""
    t = _round_sig(tau_99, 1)
    return t, float("%.12g" % (t / 2.0)), float("%.12g" % (t / 4.0))


CHI2_95 = {1: 3.841, 2: 5.991, 3: 7.815, 4: 9.488}       # 95 % points of chi^2 for 1 to 4 degrees of freedom


def fit_slope(timesteps: Sequence[float], energies: Sequence[float], errors: Sequence[float]) -> Dict[str, float]:
    """Weighted straight-line fit E(tau) = E_0 + k tau of total energies (hartree) with their error bars.

    Returns ``slope`` and ``slope_error`` in kcal/mol per Ha^-1, ``intercept`` and
    ``intercept_error`` in hartree (the energy at zero time step), ``chi2`` and ``dof``.
    Needs at least 2 different time steps."""
    x = np.asarray(timesteps, dtype=float); y = np.asarray(energies, dtype=float); s = np.asarray(errors, dtype=float)
    if not (len(x) == len(y) == len(s)) or len(x) < 2 or len(set(x.tolist())) < 2:
        raise ValueError("fit_slope: need energies at 2 or more different time steps, got %d point(s)" % len(x))
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(s)) or np.any(s <= 0):
        raise ValueError("fit_slope: energies and error bars must be finite, error bars positive")
    w = 1.0 / s ** 2
    sw, sx, sy = w.sum(), (w * x).sum(), (w * y).sum()
    sxx, sxy = (w * x * x).sum(), (w * x * y).sum()
    det = sw * sxx - sx * sx
    k = (sw * sxy - sx * sy) / det
    e0 = (sxx * sy - sx * sxy) / det
    chi2 = float(np.sum(w * (y - e0 - k * x) ** 2))
    return {"slope": float(k * HARTREE_TO_KCAL), "slope_error": float(math.sqrt(sw / det) * HARTREE_TO_KCAL),
            "intercept": float(e0), "intercept_error": float(math.sqrt(sxx / det)), "chi2": chi2, "dof": int(len(x) - 2)}


@dataclass
class TimestepChoice:
    """Result of :func:`choose_timestep`."""

    timestep: Optional[float]           # the chosen time step; None when no valid choice exists
    rule: str                           # what decided: "largest", "R1" ... "R4" (also joined, "R1+R4"), "F6", or "data"
    text: str                           # the decision in words, with the numbers
    slope: Optional[float]              # measured k_E, kcal/mol per Ha^-1 (None: not measured)
    slope_error: Optional[float]
    slope_used: float                   # |k_E| that R4 used
    slope_source: str                   # "measured", "measured, not resolved", "given" or "first guess"
    limit: float                        # F6: eps_tau / (f_geo * slope_used)
    tau_99: Optional[float]             # F5, from the smallest candidate with a measured acceptance
    extrapolate: bool                   # True: two time steps with extrapolation would cost less
    table: List[Dict[str, object]]      # one row per candidate, largest time step first
    not_checked: List[str]              # rules that could not be checked for the chosen time step

    def as_dict(self) -> Dict[str, object]:
        return stats.plain(asdict(self))


def _why(row: Dict[str, object], name: str, f_geo: float, slope_used: float, eps_tau: float, acceptance_min: float,
         variance_ratio_max: float) -> str:
    if name == "R1":
        return "R1 (acceptance %.4f, below %.2f)" % (row["acceptance"], acceptance_min)
    if name == "R2":
        return "R2 (DMC variance / VMC variance = %.3f, above %.2f)" % (row["variance_ratio"], variance_ratio_max)
    if name == "R3":
        return "R3 (population spread %.1f %%, above %.1f %%)" % (100 * row["population_spread"], 100 * row["spread_limit"])
    if name == "R4":
        return ("R4 (predicted bias of an energy difference %.2f x %.0f x %g = %.3f kcal/mol, above %.2f)"
                % (f_geo, slope_used, row["timestep"], row["predicted_bias"], eps_tau))
    return "the data check (%s)" % (row.get("reason") or "run not usable")


def choose_timestep(pilot_results: Sequence[Dict[str, object]], eps_tau: float, f_geo: float = F_GEO,
                    acceptance_min: float = ACCEPTANCE_MIN, variance_ratio_max: float = VARIANCE_RATIO_MAX,
                    spread_max: float = SPREAD_MAX, slope: Optional[Tuple[float, float]] = None,
                    slope_guess: float = SLOPE_GUESS, resolved: float = SLOPE_RELATIVE_ERROR) -> TimestepChoice:
    """Choose the DMC time step from pilot runs at several candidate time steps.

    ``pilot_results`` has one dict per candidate. Key ``timestep`` is needed. The other keys are
    optional; a rule whose input is missing (or None) is "not checked" and does not exclude a candidate:

        acceptance           mean DMC acceptance                                   -> R1
        variance_ratio       DMC variance / VMC variance (or ``variance`` and ``vmc_variance``) -> R2
        population_spread    relative standard deviation of the walker count or of the block weight;
                             ``walkers`` scales the limit by sqrt(2048 / walkers)   -> R3
        energy, error        total DMC energy and its error bar in hartree          -> slope for R4
        usable, reason       usable = False excludes the candidate (bad data); its energy is not fitted

    A candidate is allowed when all of these hold:

        R1  acceptance >= 0.99
        R2  DMC variance <= 1.10 * VMC variance
        R3  relative spread of the population <= 5 % (at 2048 walkers)
        R4  predicted time-step bias of an energy difference, f_geo * |k_E| * tau, <= ``eps_tau`` (kcal/mol)

    k_E is the slope of the energies against the time step (:func:`fit_slope`), or ``slope`` =
    (value, error) when given. If its relative error is above 25 % the rule uses |k_E| + 2 sigma.
    The fit uses every usable candidate, also one that fails R1 to R3. If a straight line does not
    describe the energies (chi^2 above its 95 % point), the largest time step is left out of the fit.
    If no slope can be measured (fewer than 2 energies) the first guess ``slope_guess`` is used.

    The largest allowed candidate is chosen, because the cost of a run is proportional to 1 / tau.
    If no candidate is allowed and the smallest one fails R4 only, the time step is the limit F6,
    eps_tau / (f_geo * |k_E|), rounded down to 2 digits; ``extrapolate`` is set when that limit is
    below tau_99 / 9 (then two time steps, tau and 4 tau with 8/9 of the run time at tau, and an
    extrapolation of every energy difference cost less). If the smallest candidate fails another
    rule, ``timestep`` is None and ``text`` says what to do.
    """
    if not eps_tau > 0.0 or not f_geo > 0.0:
        raise ValueError("choose_timestep: eps_tau and f_geo must be positive")
    rows_in = []
    for r in pilot_results:
        if "timestep" not in r or not float(r["timestep"]) > 0.0:
            raise ValueError("choose_timestep: every pilot result needs a positive 'timestep'")
        rows_in.append(dict(r, timestep=float(r["timestep"])))
    if not rows_in:
        raise ValueError("choose_timestep: no pilot result")
    taus = [r["timestep"] for r in rows_in]
    if len(set(taus)) != len(taus):
        raise ValueError("choose_timestep: a time step appears twice: %s" % sorted(taus))
    rows_in.sort(key=lambda r: -r["timestep"])

    def number(r, key):
        v = r.get(key)
        return None if v is None or not math.isfinite(float(v)) else float(v)

    # ---- slope of the total energy against the time step
    pts = [(r["timestep"], number(r, "energy"), number(r, "error")) for r in rows_in if r.get("usable", True)]
    pts = [p for p in pts if p[1] is not None and p[2] is not None and p[2] > 0.0]
    k = sk = None
    dropped: List[float] = []
    if slope is not None:
        k, sk = float(slope[0]), float(slope[1])
        source = "given"
    elif len(pts) >= 2:
        fit = fit_slope([p[0] for p in pts], [p[1] for p in pts], [p[2] for p in pts])
        # a straight line must describe the energies. If it does not (chi^2 above the 95 % point), the largest
        # time step is left out: the rule needs the slope at small time steps
        while len(pts) > 2 and fit["chi2"] > CHI2_95.get(fit["dof"], 3.0 * fit["dof"]):
            dropped.append(pts[0][0])
            pts = pts[1:]
            fit = fit_slope([p[0] for p in pts], [p[1] for p in pts], [p[2] for p in pts])
        k, sk = fit["slope"], fit["slope_error"]
        source = "measured"
    if k is None:
        slope_used, source = float(slope_guess), "first guess"
    elif sk > resolved * abs(k):
        slope_used = abs(k) + 2.0 * sk
        source += ", not resolved"
    else:
        slope_used = abs(k)
    limit = eps_tau / (f_geo * slope_used) if slope_used > 0.0 else float("inf")

    # ---- F5 from the smallest candidate that has a measured acceptance below 1
    tau_99 = None
    for r in reversed(rows_in):
        a = number(r, "acceptance")
        if a is not None and 0.0 < a < 1.0:
            tau_99 = timestep_for_acceptance(r["timestep"], a, acceptance_min)
            break

    table: List[Dict[str, object]] = []
    for r in rows_in:
        tau = r["timestep"]
        a = number(r, "acceptance")
        ratio = number(r, "variance_ratio")
        if ratio is None and number(r, "variance") is not None and number(r, "vmc_variance"):
            ratio = number(r, "variance") / number(r, "vmc_variance")
        spread = number(r, "population_spread")
        walkers = r.get("walkers")
        spread_limit = spread_max * (math.sqrt(WALKERS_REFERENCE / float(walkers)) if walkers else 1.0)
        bias = f_geo * slope_used * tau
        row = {"timestep": tau, "acceptance": a, "variance_ratio": ratio, "population_spread": spread,
               "spread_limit": spread_limit, "predicted_bias": bias,
               "R1": NOT_CHECKED if a is None else (PASS if a >= acceptance_min else FAIL),
               "R2": NOT_CHECKED if ratio is None else (PASS if ratio <= variance_ratio_max else FAIL),
               "R3": NOT_CHECKED if spread is None else (PASS if spread <= spread_limit else FAIL),
               "R4": PASS if bias <= eps_tau else FAIL,
               "usable": bool(r.get("usable", True)), "reason": r.get("reason")}
        row["failed"] = [name for name in ("R1", "R2", "R3", "R4") if row[name] == FAIL] + ([] if row["usable"] else ["data"])
        table.append(row)

    def why(row, names):
        return ", ".join(_why(row, n, f_geo, slope_used, eps_tau, acceptance_min, variance_ratio_max) for n in names)

    if k is None:
        slope_text = "No slope of the energy against the time step was measured; R4 uses the first guess %.0f kcal/mol per Ha^-1." % slope_used
    elif source.endswith("not resolved"):
        slope_text = ("Slope of the energy against the time step: %+.0f +/- %.0f kcal/mol per Ha^-1, not resolved to %.0f %%; "
                      "R4 uses |slope| + 2 sigma = %.0f." % (k, sk, 100 * resolved, slope_used))
    else:
        slope_text = "Slope of the energy against the time step: %+.0f +/- %.0f kcal/mol per Ha^-1 (%s)." % (k, sk, source)

    allowed = [row for row in table if not row["failed"]]
    extrapolate = False
    if allowed:
        chosen = allowed[0]
        timestep = chosen["timestep"]
        larger = [row for row in table if row["timestep"] > timestep]
        if not larger:
            rule = "largest"
            text = "Time step %g: the largest candidate. It passes every rule that could be checked." % timestep
        else:
            nxt = larger[-1]
            rule = "+".join(nxt["failed"])
            text = ("Time step %g: the largest candidate that passes R1 to R4. The next larger candidate, %g, fails %s."
                    % (timestep, nxt["timestep"], why(nxt, nxt["failed"])))
        not_checked = [name for name in ("R1", "R2", "R3") if chosen[name] == NOT_CHECKED]
    else:
        smallest = table[-1]
        not_checked = [name for name in ("R1", "R2", "R3") if smallest[name] == NOT_CHECKED]
        if smallest["failed"] == ["R4"]:
            timestep = _round_sig(limit, 2, down=True)
            rule = "F6"
            text = ("No candidate passes R4: the smallest one, %g, fails %s. The time step is the limit F6 = eps_tau / "
                    "(f_geo x slope) = %.2f / (%.2f x %.0f) = %.5f, rounded down to %g. R1 to R3 hold at %g and are "
                    "assumed to hold below it." % (smallest["timestep"], why(smallest, ["R4"]), eps_tau, f_geo, slope_used,
                                                   limit, timestep, smallest["timestep"]))
            if tau_99 is not None and limit < tau_99 / 9.0:
                extrapolate = True
                text += (" The limit is below tau_99 / 9 = %.5f: runs at two time steps (tau and 4 tau, with 8/9 of the run "
                         "time at tau) and an extrapolation of every energy difference would cost less." % (tau_99 / 9.0))
        else:
            timestep = None
            rule = "+".join(smallest["failed"])
            text = ("No time step can be chosen: the smallest candidate, %g, fails %s."
                    % (smallest["timestep"], why(smallest, smallest["failed"])))
            if "R1" in smallest["failed"] and tau_99 is not None:
                text += " F5 predicts an acceptance of %.2f at a time step of %.4f: repeat the pilot below it." % (acceptance_min, tau_99)
            if "R2" in smallest["failed"]:
                text += " A DMC variance far above the VMC variance points to the trial wave function: repeat the Jastrow optimization."
    if k is None:
        not_checked.append("R4 (slope is a first guess)")
    if dropped:
        slope_text += (" The energies at the time step(s) %s are not on the straight line of the smaller ones and were left "
                       "out of the slope." % ", ".join("%g" % t for t in dropped))
    return TimestepChoice(timestep=timestep, rule=rule, text=text + " " + slope_text, slope=k, slope_error=sk,
                          slope_used=float(slope_used), slope_source=source, limit=float(limit), tau_99=tau_99,
                          extrapolate=extrapolate, table=table, not_checked=not_checked)


# ----------------------------------------------------------------------------- mode "fixed": checks of given values
def _dmc_defaults() -> Dict[str, object]:
    """Defaults of the DMC input writer, so that a missing setting is judged by the value that will be used."""
    return {name: p.default for name, p in inspect.signature(dmc_input).parameters.items()
            if p.default is not inspect.Parameter.empty}


def check_settings(settings: Dict[str, object], ranks: Optional[int] = None, blocks: Optional[int] = None,
                   sigma_target: Optional[float] = None, e_scf: Optional[float] = None,
                   acceptance: Optional[float] = None, variance: Optional[float] = None,
                   t_corr: Optional[float] = None, transient: float = TRANSIENT) -> List[str]:
    """Warnings for QMC settings that the user fixed. Nothing is changed.

    ``settings`` has the layout of the QMC settings of a point: the DMC values are in
    ``settings["dmc"]`` (``timestep``, ``steps``, ``walkers``, ``warmup_steps``, ``checkpoint``).
    A value that is not given is judged by the default of the input writer. One warning per
    broken rule:

    * the walker count is not a multiple of ``ranks``;
    * the acceptance is below 0.99 (the measured ``acceptance`` when given; else a first guess from
      the time step alone, which is marked as a guess);
    * with ``blocks``: the run is shorter than 5 transients; fewer than 64 blocks are kept;
    * with ``blocks`` and ``sigma_target`` (kcal/mol): the expected error bar is above the wanted
      one. This needs the variance: ``variance`` when given, else the guess 0.022 * abs(``e_scf``);
    * no checkpoint is written;
    * keywords that QMCPACK no longer uses.
    """
    out: List[str] = []
    dmc = dict(_dmc_defaults())
    dmc.update(settings.get("dmc") or {})
    tau, steps, walkers = float(dmc["timestep"]), int(dmc["steps"]), int(dmc["walkers"])
    if not tau > 0.0 or steps < 1 or walkers < 1:
        raise ValueError("check_settings: time step %r, steps %r and walkers %r must be positive" % (tau, steps, walkers))
    tb = steps * tau
    if ranks and walkers % int(ranks) != 0:
        out.append("walkers = %d is not a multiple of the %d MPI ranks: QMCPACK will change the population. "
                   "The next multiple is %d." % (walkers, ranks, walkers_for(ranks, target=walkers)))
    if acceptance is not None:
        if acceptance < ACCEPTANCE_MIN:
            out.append("measured DMC acceptance %.4f at time step %g is below %.2f: the time step is too large."
                       % (acceptance, tau, ACCEPTANCE_MIN))
    else:
        guess = timestep_for_acceptance(0.005, 1.0 - REJECTED_AT_0005)
        if tau > guess:
            out.append("time step %g: the acceptance is expected below %.2f (first guess: %.2f is reached near a time "
                       "step of %.3f; not measured for this molecule)." % (tau, ACCEPTANCE_MIN, ACCEPTANCE_MIN, guess))
    if blocks is not None:
        discard = stats.transient_blocks(steps, tau, transient)
        kept = int(blocks) - discard
        if blocks * tb < MIN_TRANSIENTS * transient:
            out.append("the run is %g Ha^-1 long (%d blocks x %d steps x %g). That is shorter than %g transients of %g "
                       "Ha^-1: a large part of the run is start-up." % (blocks * tb, blocks, steps, tau, MIN_TRANSIENTS, transient))
        if kept < MIN_KEPT_BLOCKS:
            out.append("%d of %d blocks are kept after the transient (%d discarded): fewer than %d, so the error bar of "
                       "the run is not reliable." % (max(kept, 0), blocks, discard, MIN_KEPT_BLOCKS))
        if sigma_target is not None and kept > 0:
            guessed = variance is None
            if variance is None and e_scf is not None:
                variance = VARIANCE_PER_HARTREE * abs(float(e_scf))
            if variance:
                plan = plan_dmc(tau, walkers, blocks=int(blocks), variance=variance, t_corr=T_CORR if t_corr is None else t_corr,
                                block_time=tb, transient=transient)
                if plan.sigma_expected > sigma_target:
                    out.append("expected error bar %.3f kcal/mol with %d blocks, above the wanted %.3f kcal/mol%s."
                               % (plan.sigma_expected, blocks, sigma_target,
                                  " (from first guesses of the variance and the correlation time)" if guessed else ""))
    if int(dmc.get("checkpoint", -1)) <= 0:
        out.append("checkpoint = %s: no checkpoint is written. A run that is stopped cannot be continued and must pass "
                   "the transient again." % dmc.get("checkpoint", -1))
    obsolete = ("nonlocalpp", "use_nonlocalpp_deriv")
    for section in ("opt", "dmc"):
        for key in (settings.get(section) or {}):
            if key in obsolete:
                out.append("%s setting %r is obsolete: QMCPACK ignores it and a later version will stop on it." % (section, key))
    return out


# ----------------------------------------------------------------------------- reading one pilot point
def _raise_discard(energy: np.ndarray, weight: np.ndarray, drop: int) -> int:
    """Action of monitor M3: while the blocks kept after ``drop`` fail a transient test (first blocks
    against the rest, first half against second half), double the discard. The discard stays
    below 25 % of the run."""
    n = len(energy)
    trace = {"LocalEnergy": energy, "BlockWeight": weight}
    for _ in range(10):
        bad = [f for f in monitor.transient(trace, drop, finished=True) if f.data.get("test") in ("first_blocks", "drift")]
        larger = max(2 * drop, drop + 1)
        if not bad or larger > monitor.LIMITS.discard_fraction * n:
            break
        drop = larger
    return int(drop)


def _trace_from(meta: Dict[str, object], name: str, file_key: str) -> Optional[Dict[str, np.ndarray]]:
    if isinstance(meta.get(name), dict):
        return {k: np.asarray(v, dtype=float) for k, v in meta[name].items()}
    if meta.get(file_key):
        return stats.read_scalar(str(meta[file_key]))          # raises with the file name when it cannot be read
    return None


def measure_point(data: PointData, timestep: float, steps: int, walkers: int, ranks: Optional[int] = None,
                  transient: float = TRANSIENT) -> Dict[str, object]:
    """Everything that a finished or running pilot point tells, as a plain dict.

    ``timestep``, ``steps`` (per block) and ``walkers`` are the settings of the run. Always
    measured: the energy with its error bars, the autocorrelation time of the blocks, the discard,
    and ``noise`` = SEM^2 * walkers * imaginary time of the data (= 2 T_c V, the input of F1).
    Measured only when the engine reports it (see the module text): ``acceptance``, ``variance``,
    ``vmc_variance``, ``population_spread``, ``seconds_per_block``. A value that is not available
    is None and its name is in ``not_measured``.

    Discard. ``drop_planned`` is the transient in blocks. ``findings`` holds the transient test (M3)
    for that planned discard and all other monitors for the discard that is used. ``drop`` is the
    discard that the numbers above use: the
    planned one, raised to the data-driven value (rule MSER) when that is larger, and doubled while
    the kept blocks still fail a transient test (action of monitor M3), up to 25 % of the run.
    Raises ValueError when fewer than 8 blocks are left after the discard.
    """
    meta = dict(data.meta or {})
    trace = _trace_from(meta, "trace", "scalar_file")
    if trace is not None and "LocalEnergy" in trace:
        energy = np.asarray(trace["LocalEnergy"], dtype=float)
        weight = np.asarray(trace["BlockWeight"], dtype=float) if "BlockWeight" in trace else np.ones(len(energy))
    else:
        energy, weight = np.asarray(data.energy, dtype=float), np.asarray(data.weight, dtype=float)
        trace = {"LocalEnergy": energy, "BlockWeight": weight}
    eq = stats.equilibration(energy, weight, steps=steps, timestep=timestep, transient=transient)
    if eq.n_kept < 8:
        raise ValueError("point %s: %d blocks, %d are discarded as transient: fewer than 8 blocks are left"
                         % (data.key, len(energy), eq.drop))
    planned = int(eq.drop_physical)
    drop = _raise_discard(energy, weight, eq.drop)
    est = stats.estimate(energy, weight, drop)
    data_time = est.n_blocks * steps * timestep
    out: Dict[str, object] = {
        "key": data.key, "timestep": float(timestep), "steps": int(steps), "walkers": int(walkers),
        "blocks": int(len(energy)), "drop": int(drop), "drop_planned": planned, "drop_mser": eq.drop_mser,
        "kept": est.n_blocks, "energy": est.mean, "error": est.error, "error_reblock": est.error_reblock,
        "tau_blocks": est.tau, "drift_z": eq.drift_z, "data_time": data_time,
        "block_sd_kcal": float(np.std(energy[drop:], ddof=1) * HARTREE_TO_KCAL),
        "noise": est.error ** 2 * walkers * data_time, "e_scf": data.e_surrogate,
    }
    not_measured = []
    if "AcceptRatio" in trace:
        out["acceptance"] = float(np.mean(trace["AcceptRatio"][drop:]))
    elif meta.get("acceptance") is not None:
        out["acceptance"] = float(meta["acceptance"])
    else:
        out["acceptance"] = None; not_measured.append("acceptance")
    if "LocalEnergy_sq" in trace:
        out["variance"] = stats.local_energy_variance(energy, trace["LocalEnergy_sq"], weight, drop)
    elif meta.get("variance") is not None:
        out["variance"] = float(meta["variance"])
    else:
        out["variance"] = None; not_measured.append("variance")
    vmc = _trace_from(meta, "vmc_trace", "vmc_scalar_file")
    if vmc is not None and "LocalEnergy_sq" in vmc and len(vmc["LocalEnergy"]) >= 2:
        out["vmc_variance"] = stats.local_energy_variance(vmc["LocalEnergy"], vmc["LocalEnergy_sq"], vmc.get("BlockWeight"))
    elif meta.get("vmc_variance") is not None:
        out["vmc_variance"] = float(meta["vmc_variance"])
    else:
        out["vmc_variance"] = None; not_measured.append("vmc_variance")
    w = weight[drop:]
    if float(np.ptp(w)) > 0.0:
        out["population_spread"] = float(np.std(w, ddof=1) / np.mean(w))
    else:
        out["population_spread"] = None; not_measured.append("population_spread")
    if "BlockCPU" in trace:
        out["seconds_per_block"] = float(np.median(trace["BlockCPU"][drop:]))
    elif meta.get("seconds_per_block") is not None:
        out["seconds_per_block"] = float(meta["seconds_per_block"])
    else:
        out["seconds_per_block"] = None; not_measured.append("seconds_per_block")
    out["t_corr"] = out["noise"] / (2.0 * out["variance"]) if out["variance"] else None
    use_ranks = meta.get("ranks") or ranks
    if out["seconds_per_block"] is not None and use_ranks:
        out["step_seconds"] = out["seconds_per_block"] * int(use_ranks) / (float(walkers) * steps)
    else:
        out["step_seconds"] = None; not_measured.append("step_seconds")
    out["not_measured"] = not_measured
    found = monitor.check_trace(trace, drop=planned, steps=steps, timestep=timestep, walkers=walkers,
                                vmc_variance=out["vmc_variance"], finished=data.final)
    if drop > planned:
        # The transient test (M3) belongs to the planned discard: it says whether the plan was long enough.
        # Every other test is made on the blocks that the numbers above use. With the planned discard the
        # start-up blocks that are discarded anyway would be reported as outlier blocks.
        kept = monitor.check_trace(trace, drop=drop, steps=steps, timestep=timestep, walkers=walkers,
                                   vmc_variance=out["vmc_variance"], finished=data.final)
        merged = monitor.Findings([f for f in found if f.code == monitor.M3] + [f for f in kept if f.code != monitor.M3])
        merged.checked = list(kept.checked)
        merged.skipped = dict(kept.skipped)
        found = merged
    out["findings"] = found.as_dicts()
    out["monitors_skipped"] = dict(found.skipped)
    mismatch = []
    for name, want in (("timestep", timestep), ("steps", steps), ("walkers", walkers)):
        got = meta.get(name)
        if got is not None and abs(float(got) - float(want)) > 1e-9 * max(1.0, abs(float(want))):
            mismatch.append("%s: requested %g, the run used %g" % (name, want, float(got)))
    out["mismatch"] = mismatch
    return stats.plain(out)


# ----------------------------------------------------------------------------- the pilot driver
class Pilot:
    """Pilot runs that choose the QMC settings of a project.

    Parameters
    ----------
    engine      source of DMC block energies (``engine.base.Engine``). Pilot points are submitted
                with role ``pilot`` and their own settings.
    symbols, coords
                the geometry of the pilot runs (angstrom). Use the first target of the search.
    settings    the QMC settings of the project, in the layout that points receive:
                ``{"dmc": {"timestep": ..., "walkers": ..., "steps": ..., "warmup_steps": ...},
                "jastrow": {...}, "opt": {...}, ...}``. Other keys (functional, basis, charge, ...)
                are passed on unchanged.
    state_dir   folder for ``pilot_state.json``, ``pilot.log`` and, at the end, ``qmc_settings.json``.
    options     keyword arguments; see ``Pilot.OPTIONS`` for the names and defaults.

    Modes (option ``mode``)

    fixed       no pilot run. The given settings are checked (:func:`check_settings`) and written
                unchanged, with the warnings.
    auto        one pilot: one short DMC run per candidate time step (0.02, 0.01, 0.005 Ha^-1).
                From the runs: acceptance, variance, population spread, block noise, seconds per
                block, and the slope of the energy against the time step. Then
                :func:`choose_timestep` and :func:`plan_dmc`.
    iterative   as ``auto``; when a check fires, the affected runs are repeated with a changed
                setting, at most ``max_repeats`` times: an acceptance below 0.99 at every candidate
                -> smaller time steps; a transient that is not over -> twice the transient; a
                population, outlier or acceptance-drop finding, or a failed job -> a new run
                (new seed); a DMC variance above 1.10 times the VMC variance -> a new Jastrow
                optimization with twice the samples.

    ``tick()`` is one pass and returns ``RUNNING``, ``DONE`` or ``NEEDS_ATTENTION``. The state is in
    ``pilot_state.json``; a new object with the same folder continues the same pilot.
    ``qmc_settings.json`` is written only when the status is ``DONE``. It holds the settings for
    all later points (same layout as ``settings``) and, under ``"autotune"``, every chosen value
    with its basis, the measurements, the plans for final energies and scan points, the names of
    quantities that are first guesses, the rules that were not checked, and all warnings.

    Length of the pilot runs. Each run has the transient (10 Ha^-1) and at most ``pilot_time``
    (800 Ha^-1) of data in blocks of ``pilot_block_time`` (0.5 Ha^-1). The runs are stopped
    together as soon as every run keeps ``pilot_min_blocks`` blocks and either the slope of the
    energy is known to 25 %, or the largest candidate that passes R1 to R3 also passes R4 with
    |slope| + 2 sigma (then more data cannot change the choice).
    """

    OPTIONS: Dict[str, object] = {
        "mode": "auto",
        "max_repeats": 2,               # mode iterative: largest number of repeat rounds
        "sigma_target": 0.2,            # kcal/mol: wanted error bar of the final energy of one geometry
        "eps_tau": None,                # kcal/mol: tolerated time-step bias of an energy difference (default: sigma_target)
        "ranks": None,                  # MPI ranks of one QMC job (walker counts, time per step, wall time)
        "charge": 0,                    # molecular charge (electron count for the first guess of the cost)
        "candidates": CANDIDATES,       # time steps of the first pilot round, Ha^-1
        "pilot_time": 800.0,            # Ha^-1 of data per pilot run (upper limit)
        "pilot_block_time": 0.5,        # Ha^-1 per pilot block
        "pilot_min_blocks": 100,        # kept blocks per run before the runs may be stopped early
        "pilot_walkers": None,          # walkers of a pilot run (default: walkers_for(ranks, "pilot"))
        "transient": TRANSIENT,         # Ha^-1
        "block_time": BLOCK_TIME,       # Ha^-1 per block of the production runs
        "share_wavefunction": False,    # True: one Jastrow factor for all pilot runs (they are further series of the first)
        "final_blocks": None,           # blocks of a final run, if fixed elsewhere; default: from sigma_target
        "n_directions": None,           # dense scan: number of search directions (plan of the scan points)
        "n_negative": 0,                # dense scan: how many of them are reaction coordinates (saddle search)
        "e_cap": 5.0,                   # dense scan: energy rise at the outer shell, kcal/mol
        "eps_total": 0.2,               # dense scan: tolerance on the energy of the final geometry, kcal/mol
        "shells": (1.0,),               # dense scan: shell positions as fractions of the span
        "scan_rule": "joint",           # "joint" (protocol v2) or "rule_b" (protocol v1)
        "e_scf": None,                  # hartree: SCF energy with pseudopotentials, if known (mode fixed: forecast)
        "core_hour_limit": None,        # warn when one final run is planned above this cost
        "setup_seconds": 0.0,           # time of a job before the DMC section starts (added to the wall time)
        "name": "pilot",                # first part of the keys of the pilot points
    }

    def __init__(self, engine: Optional[Engine], symbols: Sequence[str], coords: np.ndarray, settings: Dict[str, object],
                 state_dir: str, log: Optional[Callable[[str], None]] = None, **options):
        unknown = sorted(set(options) - set(self.OPTIONS))
        if unknown:
            raise ValueError("Pilot: unknown option(s) %s. Allowed: %s" % (", ".join(unknown), ", ".join(sorted(self.OPTIONS))))
        self.engine = engine
        self.state_dir = state_dir
        self._log_sink = log
        os.makedirs(state_dir, exist_ok=True)
        self.state_file = os.path.join(state_dir, "pilot_state.json")
        self.settings_file = os.path.join(state_dir, "qmc_settings.json")
        self.log_file = os.path.join(state_dir, "pilot.log")
        if os.path.exists(self.state_file):
            with open(self.state_file) as fh:
                self.state = json.load(fh)
            return
        opts = dict(self.OPTIONS)
        opts.update(options)
        if opts["mode"] not in MODES:
            raise ValueError("Pilot: mode %r; allowed: %s" % (opts["mode"], ", ".join(MODES)))
        if opts["eps_tau"] is None:
            opts["eps_tau"] = opts["sigma_target"]
        cands = sorted({float(t) for t in opts["candidates"]}, reverse=True)
        if not cands or min(cands) <= 0.0:
            raise ValueError("Pilot: candidates must be positive time steps, got %r" % (opts["candidates"],))
        opts["candidates"] = cands
        opts["shells"] = [float(f) for f in opts["shells"]]
        for key in ("sigma_target", "eps_tau", "pilot_time", "pilot_block_time", "transient", "block_time"):
            if not float(opts[key]) > 0.0:
                raise ValueError("Pilot: option %s must be positive, got %r" % (key, opts[key]))
        if opts["mode"] != "fixed" and engine is None:
            raise ValueError("Pilot: mode %s needs an engine" % opts["mode"])
        syms = [normalize_symbol(s) for s in symbols]
        X = np.asarray(coords, dtype=float).reshape(-1, 3)
        if len(X) != len(syms):
            raise ValueError("Pilot: %d symbols and %d coordinate rows" % (len(syms), len(X)))
        user = copy.deepcopy(dict(settings or {}))
        user.pop("autotune", None)
        if user.get("jastrow"):
            jastrow = {"value": dict(DEFAULT_JASTROW, **user["jastrow"]), "source": "user",
                       "basis": "given in the settings"}
        else:
            jastrow = {"value": jastrow_defaults(X), "source": "rule",
                       "basis": "cutoffs from the largest distance between two atoms; not tested for molecules below 4 bohr"}
        try:
            n_el = int(valence_electrons(syms, int(opts["charge"])))
        except KeyError:
            n_el = None
        self.state = {
            "status": RUNNING, "message": "", "options": stats.plain(opts), "symbols": syms, "coords": X.tolist(),
            "settings": stats.plain(user), "jastrow": jastrow, "n_electrons": n_el, "rounds": [], "repeats": 0,
            "warnings": [], "base_warnings": [], "history": [],
        }
        untested = sorted({s for s in syms if s not in TESTED_ELEMENTS})
        if untested:
            self._warn("element(s) %s are outside the tested set (%s): every default of the automatic choice is untested "
                       "for them" % (", ".join(untested), ", ".join(TESTED_ELEMENTS)))
        self.state["base_warnings"] = list(self.state["warnings"])
        self._save()

    # ------------------------------------------------------------------ small helpers
    @property
    def status(self) -> str:
        return self.state["status"]

    @property
    def message(self) -> str:
        return self.state["message"]

    @property
    def warnings(self) -> List[str]:
        return list(self.state["warnings"])

    @property
    def options(self) -> Dict[str, object]:
        return self.state["options"]

    def say(self, text: str) -> None:
        line = "%s  [pilot] %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), text)
        with open(self.log_file, "a") as fh:
            fh.write(line + "\n")
        if self._log_sink:
            self._log_sink(line)

    def _warn(self, text: str) -> None:
        if text not in self.state["warnings"]:
            self.state["warnings"].append(text)

    def _save(self) -> None:
        tmp = self.state_file + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(stats.plain(self.state), fh, indent=1)
        os.replace(tmp, self.state_file)

    def _stop(self, message: str) -> str:
        self.state["status"] = NEEDS_ATTENTION
        self.state["message"] = message
        self.say("NEEDS ATTENTION: " + message)
        return NEEDS_ATTENTION

    def clear_attention(self) -> str:
        """Set a ``NEEDS_ATTENTION`` status back to running, after the cause was repaired (for example a failed
        pilot run that was submitted again). The next :meth:`tick` looks at the pilot runs again."""
        if self.state["status"] == NEEDS_ATTENTION:
            self.state["status"] = RUNNING
            self.state["message"] = ""
            self.say("the NEEDS_ATTENTION status was cleared by request; checking again")
            self._save()
        return self.state["status"]

    def settings(self) -> Optional[Dict[str, object]]:
        """The chosen settings (content of ``qmc_settings.json``), or None while the pilot is not done."""
        if self.state["status"] != DONE:
            return None
        if not os.path.exists(self.settings_file):
            raise FileNotFoundError("the pilot is DONE but its result file is missing: %s" % self.settings_file)
        with open(self.settings_file) as fh:
            return json.load(fh)

    def _write_settings(self, settings: Dict[str, object]) -> None:
        tmp = self.settings_file + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(stats.plain(settings), fh, indent=1)
        os.replace(tmp, self.settings_file)

    def _user_walkers(self) -> int:
        return int((self.state["settings"].get("dmc") or {}).get("walkers") or WALKER_TARGET["final"])

    def _pilot_walkers(self) -> int:
        o = self.options
        if o["pilot_walkers"]:
            return int(o["pilot_walkers"])
        if o["ranks"]:
            return walkers_for(int(o["ranks"]), "pilot")
        return WALKER_TARGET["pilot"]

    # ------------------------------------------------------------------ rounds and points
    def _new_round(self, timesteps: Sequence[float], reason: str, transient: Optional[float] = None,
                   opt_samples: Optional[int] = None) -> None:
        o = self.options
        index = len(self.state["rounds"]) + 1
        transient = float(o["transient"] if transient is None else transient)
        walkers = self._pilot_walkers()
        points = []
        for tau in sorted({float(t) for t in timesteps}, reverse=True):
            steps = _steps_for(float(o["pilot_block_time"]), tau)
            tb = steps * tau
            discard = stats.transient_blocks(steps, tau, transient)
            blocks = discard + int(math.ceil(float(o["pilot_time"]) / tb - 1e-9))
            points.append({"timestep": tau, "key": "%s/r%d/tau%g" % (o["name"], index, tau), "blocks": blocks,
                           "dmc": {"timestep": tau, "steps": steps, "warmup_steps": int(math.ceil(transient / tau - 1e-9)),
                                   "walkers": walkers},
                           "series": 1, "base_key": None, "opt_samples": opt_samples, "submitted": False})
        if o["share_wavefunction"]:
            for i, p in enumerate(points[1:], start=2):
                p["series"], p["base_key"] = i, points[0]["key"]
        self.state["rounds"].append({"index": index, "reason": reason, "transient": transient, "points": points,
                                     "stopped_early": None})
        self.say("round %d (%s): pilot runs at time steps %s, %d walkers, up to %g Ha^-1 of data each"
                 % (index, reason, ", ".join("%g" % p["timestep"] for p in points), walkers, o["pilot_time"]))

    def _point_settings(self, point: Dict[str, object]) -> Dict[str, object]:
        s = copy.deepcopy(self.state["settings"])
        dmc = dict(s.get("dmc") or {})
        dmc.update(point["dmc"])
        dmc.pop("blocks", None)                         # the number of blocks is given by PointSpec.blocks
        s["dmc"] = dmc
        s["jastrow"] = dict(self.state["jastrow"]["value"])
        if point.get("opt_samples"):
            opt = dict(s.get("opt") or {})
            opt["samples"] = int(point["opt_samples"])
            s["opt"] = opt
        return s

    def _submit(self, rnd: Dict[str, object]) -> None:
        coords = np.array(self.state["coords"], dtype=float)
        first = rnd["points"][0]
        for p in rnd["points"]:
            if p["submitted"]:
                continue
            if p["base_key"] is not None and self.engine.read(first["key"]).nblocks == 0:
                continue                                # a further series starts once the first run writes blocks
            self.engine.submit(PointSpec(key=p["key"], symbols=list(self.state["symbols"]), coords=coords,
                                         blocks=int(p["blocks"]), role="pilot", series=int(p["series"]),
                                         base_key=p["base_key"], settings=self._point_settings(p)))
            p["submitted"] = True

    def _latest(self) -> Dict[float, Tuple[Dict[str, object], Dict[str, object]]]:
        """For every time step the point of the latest round that ran it."""
        out: Dict[float, Tuple[Dict[str, object], Dict[str, object]]] = {}
        for rnd in self.state["rounds"]:
            for p in rnd["points"]:
                out[float(p["timestep"])] = (rnd, p)
        return out

    # ------------------------------------------------------------------ one pass
    def tick(self) -> str:
        """One pass: submit what is missing, read what exists, decide, write the state."""
        if self.state["status"] != RUNNING:
            return self.state["status"]
        try:
            if self.options["mode"] == "fixed":
                self._finish_fixed()
            else:
                self._tick_pilot()
        finally:
            self._save()
        return self.state["status"]

    def run(self, max_ticks: int = 10000) -> str:
        """Call ``engine.advance()`` and ``tick()`` until the pilot ends (for engines that simulate time)."""
        for _ in range(max_ticks):
            if self.engine is not None:
                self.engine.advance()
            if self.tick() != RUNNING:
                break
        return self.state["status"]

    def _tick_pilot(self) -> None:
        if not self.state["rounds"]:
            self._new_round(self.options["candidates"], "first pilot")
        rnd = self.state["rounds"][-1]
        self._submit(rnd)
        watched = list(self._latest().values())         # the latest run of every time step, over all rounds
        datas = {p["key"]: self.engine.read(p["key"]) for _, p in watched if p["submitted"]}
        failed = [d for d in datas.values() if d.state == FAILED]
        if failed:
            text = "; ".join("%s: %s" % (d.key, d.message or "no message") for d in failed)
            keys = [d.key for d in failed]
            taus = [p["timestep"] for _, p in watched if p["key"] in keys]
            if self._may_repeat():
                self._repeat(taus, "pilot run failed (%s)" % text, transient=rnd["transient"])
            else:
                self._stop("pilot run failed: %s" % text)
            return
        if len(datas) < len(watched) or not all(d.final for d in datas.values()):
            if len(datas) == len(watched) and rnd["stopped_early"] is None:
                self._stop_early(rnd, watched, datas)
            return
        self._analyse()

    def _may_repeat(self) -> bool:
        return self.options["mode"] == "iterative" and self.state["repeats"] < int(self.options["max_repeats"])

    def _repeat(self, timesteps: Sequence[float], reason: str, transient: Optional[float] = None,
                opt_samples: Optional[int] = None) -> None:
        self.state["repeats"] += 1
        self.state["history"].append({"repeat": self.state["repeats"], "reason": reason, "timesteps": list(timesteps)})
        self.say("repeat %d of %d: %s" % (self.state["repeats"], self.options["max_repeats"], reason))
        self._new_round(timesteps, reason, transient=transient, opt_samples=opt_samples)
        self._submit(self.state["rounds"][-1])

    # ------------------------------------------------------------------ early stop of a round
    def _stop_early(self, rnd: Dict[str, object], watched, datas: Dict[str, PointData]) -> None:
        """Stop the runs that are still going when more data cannot change the choice of the time step."""
        o = self.options
        results = []
        for r, p in watched:
            d = datas[p["key"]]
            dmc = p["dmc"]
            drop = stats.transient_blocks(dmc["steps"], dmc["timestep"], r["transient"])
            if d.nblocks - drop < int(o["pilot_min_blocks"]):
                return
            est = stats.estimate(d.energy, d.weight, drop)
            row = {"timestep": dmc["timestep"], "energy": est.mean, "error": est.error, "walkers": dmc["walkers"]}
            meta = d.meta or {}
            if isinstance(meta.get("trace"), dict) and "AcceptRatio" in meta["trace"]:
                row["acceptance"] = float(np.mean(np.asarray(meta["trace"]["AcceptRatio"], dtype=float)[drop:]))
            elif meta.get("acceptance") is not None:
                row["acceptance"] = float(meta["acceptance"])
            results.append(row)
        if len(results) < 2:
            return
        choice = choose_timestep(results, float(o["eps_tau"]))
        reason = None
        if choice.slope is not None and choice.slope_error <= SLOPE_RELATIVE_ERROR * abs(choice.slope):
            reason = "the slope of the energy is known to %.0f %%" % (100 * choice.slope_error / abs(choice.slope))
        else:
            ok123 = [r for r in choice.table if not [f for f in r["failed"] if f != "R4"]]
            if ok123 and choice.timestep is not None and choice.timestep >= ok123[0]["timestep"]:
                reason = "the largest candidate that passes R1 to R3 passes R4 even with |slope| + 2 sigma"
        if reason is None:
            return
        n = 0
        for _, p in watched:
            if not datas[p["key"]].final:
                self.engine.stop(p["key"]); n += 1
        rnd["stopped_early"] = reason
        self.say("round %d: %s. %d run(s) stopped." % (rnd["index"], reason, n))

    # ------------------------------------------------------------------ analysis of the finished runs
    def _analyse(self) -> None:
        o = self.options
        self.state["warnings"] = list(self.state["base_warnings"])
        latest = self._latest()
        measured: Dict[float, Dict[str, object]] = {}
        for tau in sorted(latest, reverse=True):
            rnd, p = latest[tau]
            d = self.engine.read(p["key"])
            try:
                m = measure_point(d, tau, int(p["dmc"]["steps"]), int(p["dmc"]["walkers"]), ranks=o["ranks"],
                                  transient=float(rnd["transient"]))
            except ValueError as exc:
                if self._may_repeat():
                    self._repeat([tau], "pilot run without usable data (%s)" % exc, transient=rnd["transient"])
                else:
                    self._stop("pilot run without usable data: %s" % exc)
                return
            m["round"] = rnd["index"]
            m["cost_core_hours"] = float(self.engine.cost(p["key"]))
            measured[tau] = m
        mismatch = ["%s: %s" % (m["key"], "; ".join(m["mismatch"])) for m in measured.values() if m["mismatch"]]
        if mismatch:
            self._stop("a pilot run did not use the requested settings: %s" % " | ".join(mismatch))
            return

        # ---- what the monitors found
        redo: List[Tuple[float, str]] = []
        transient_hit: List[float] = []
        variance_hit: List[float] = []
        for tau, m in measured.items():
            m["usable"], m["reason"] = True, None
            for f in m["findings"]:
                code, sev = f["code"], f["severity"]
                if code == monitor.M3 and f["data"].get("test") != "discard_fraction":
                    transient_hit.append(tau)
                elif code in (monitor.M4, monitor.M6) or (code == monitor.M5 and sev == monitor.ERROR):
                    redo.append((tau, f["message"])); m["usable"], m["reason"] = False, f["message"]
                elif code == monitor.M8 and sev == monitor.ERROR:
                    variance_hit.append(tau)
                    if f["data"].get("test") == "change":
                        m["usable"], m["reason"] = False, f["message"]
                if code != monitor.M7 and (sev != monitor.INFO or code == monitor.M8):
                    self._warn("pilot run at time step %g (%s): %s" % (tau, m["key"], f["message"]))
            if m["drop"] > m["drop_planned"]:
                self._warn("pilot run at time step %g (%s): %d start-up blocks were discarded, more than the planned %d"
                           % (tau, m["key"], m["drop"], m["drop_planned"]))
        if self._may_repeat():
            if transient_hit:
                T = 2.0 * max(float(latest[t][0]["transient"]) for t in transient_hit)
                self.state["options"]["transient"] = T
                self._repeat(sorted(set(transient_hit)), "the transient is not over after the planned discard (monitor M3): "
                             "transient doubled to %g Ha^-1" % T, transient=T)
                return
            if redo:
                self._repeat(sorted({t for t, _ in redo}), "bad pilot data: %s" % redo[0][1],
                             transient=float(o["transient"]))
                return
            if variance_hit:
                samples = 2 * int(self._opt_samples())
                self._repeat(sorted(set(variance_hit)), "DMC variance above the limit (monitor M8): new Jastrow optimization "
                             "with %d samples" % samples, transient=float(o["transient"]), opt_samples=samples)
                return

        # ---- the acceptance must change with the time step as F5 says; else the runs did not use the requested steps
        acc = [(t, m["acceptance"]) for t, m in sorted(measured.items()) if m["acceptance"] is not None and m["acceptance"] < 1.0]
        exponent = None
        if len(acc) >= 2:
            (t_small, a_small), (t_big, a_big) = acc[0], acc[-1]
            exponent = math.log((1.0 - a_big) / (1.0 - a_small)) / math.log(t_big / t_small)
            if not 0.8 <= exponent <= 2.0:
                self._stop("the rejected fraction of the DMC moves changes like (time step)^%.2f between %g (acceptance "
                           "%.5f) and %g (acceptance %.5f). The measured exponent of 5 systems is 1.27 to 1.31. The runs may "
                           "not have used the requested time steps." % (exponent, t_small, a_small, t_big, a_big))
                return

        results = [{"timestep": t, "acceptance": m["acceptance"], "variance": m["variance"],
                    "vmc_variance": m["vmc_variance"], "population_spread": m["population_spread"],
                    "walkers": m["walkers"], "energy": m["energy"], "error": m["error"], "usable": m["usable"],
                    "reason": m["reason"]} for t, m in measured.items()]
        choice = choose_timestep(results, float(o["eps_tau"]))
        self.say(choice.text)
        if choice.timestep is None:
            if "R1" in choice.rule and choice.tau_99 is not None and self._may_repeat():
                smallest = min(measured)
                top = min(_round_sig(choice.tau_99, 1, down=True), float("%.12g" % (smallest / 2.0)))
                self._repeat([top, float("%.12g" % (top / 2.0))], "acceptance below %.2f at every candidate (monitor M7): "
                             "smaller time steps" % ACCEPTANCE_MIN, transient=float(o["transient"]))
                return
            if ("data" in choice.rule or "R2" in choice.rule or "R3" in choice.rule) and self._may_repeat():
                self._repeat([min(measured)], "no usable pilot run: %s" % choice.text, transient=float(o["transient"]))
                return
            self._stop(choice.text)
            return
        self._finish(choice, measured, exponent)

    def _opt_samples(self) -> int:
        from .engine.qmcpack_inputs import opt_input
        default = inspect.signature(opt_input).parameters["samples"].default
        return int((self.state["settings"].get("opt") or {}).get("samples") or default)

    # ------------------------------------------------------------------ results
    def _finish(self, choice: TimestepChoice, measured: Dict[float, Dict[str, object]], exponent: Optional[float]) -> None:
        o = self.options
        tau = float(choice.timestep)
        ref_tau = tau if tau in measured else min(measured)
        m = measured[ref_tau]
        guesses: List[str] = []
        ranks = int(o["ranks"]) if o["ranks"] else None
        target = self._user_walkers()
        walkers = walkers_for(ranks, "final", target=target) if ranks else target
        if ranks is None:
            self._warn("the number of MPI ranks is not given: the walker count %d was not made a multiple of it, and no "
                       "wall time is planned" % walkers)
        n_el = self.state["n_electrons"]
        e_scf = m["e_scf"] if m["e_scf"] is not None else o["e_scf"]
        variance = m["variance"]
        if variance is None:
            e_ref = e_scf if e_scf is not None else m["energy"]
            if abs(e_ref) > 0.0:
                variance = VARIANCE_PER_HARTREE * abs(e_ref)
                guesses.append("variance (first guess %.3g x |E| = %.4f Ha^2; not measured)" % (VARIANCE_PER_HARTREE, variance))
        t_corr = m["noise"] / (2.0 * variance) if variance else None
        step_seconds = m["step_seconds"]
        if step_seconds is None and n_el:
            step_seconds = step_seconds_guess(n_el)
            guesses.append("time per walker and step (first guess %.3g core-seconds for %d electrons; not measured): "
                           "core-hours and wall times are estimates" % (step_seconds, n_el))
        common = dict(noise=m["noise"], block_time=float(o["block_time"]), transient=float(o["transient"]),
                      step_seconds=step_seconds, ranks=ranks, setup_seconds=float(o["setup_seconds"]))
        if o["final_blocks"]:
            final = plan_dmc(tau, walkers, blocks=int(o["final_blocks"]), **common)
            if final.sigma_expected > float(o["sigma_target"]):
                self._warn("a final run of %d blocks is expected to give an error bar of %.3f kcal/mol, above the wanted "
                           "%.3f kcal/mol" % (final.blocks, final.sigma_expected, float(o["sigma_target"])))
        else:
            final = plan_dmc(tau, walkers, sigma_target=float(o["sigma_target"]), **common)
        plans: Dict[str, object] = {"final": final.as_dict()}
        if o["n_directions"]:
            sigma_pt = scan_point_sigma(int(o["n_directions"]), float(o["e_cap"]), float(o["eps_total"]), o["shells"],
                                        o["scan_rule"], n_negative=int(o.get("n_negative") or 0))
            scan = plan_dmc(tau, walkers, sigma_target=sigma_pt, **common)
            plans["scan"] = dict(scan.as_dict(), sigma_point=sigma_pt, n_directions=int(o["n_directions"]),
                                 rule=o["scan_rule"], shells=list(o["shells"]))
        for name, plan in plans.items():
            for note in plan["notes"]:
                self._warn("plan of the %s runs: %s" % (name, note))
        if o["core_hour_limit"] and final.core_hours is not None and final.core_hours > float(o["core_hour_limit"]):
            self._warn("one final run is planned at %.1f core-hours, above the limit of %.1f"
                       % (final.core_hours, float(o["core_hour_limit"])))
        if choice.extrapolate:
            self._warn("the time step %g is set by the bias limit and lies below tau_99 / 9: runs at two time steps with "
                       "an extrapolation would cost less. The package does not do this on its own." % tau)
        if ref_tau != tau:
            self._warn("no pilot run exists at the chosen time step %g: noise, variance and time per step are those of "
                       "the run at %g" % (tau, ref_tau))
        for name in choice.not_checked:
            self._warn("time step rule %s was not checked: the pilot data do not hold the quantity it needs" % name)
        fc = None
        if n_el and e_scf is not None and abs(e_scf) > 0.0:
            fc = forecast(n_el, e_scf, float(o["sigma_target"]), walkers, tau, float(o["block_time"]),
                          float(o["transient"])).as_dict()

        out = copy.deepcopy(self.state["settings"])
        dmc = dict(out.get("dmc") or {})
        dmc.pop("blocks", None)
        dmc.update({"timestep": tau, "steps": final.steps, "warmup_steps": final.warmup_steps, "walkers": walkers})
        out["dmc"] = dmc
        out["jastrow"] = dict(self.state["jastrow"]["value"])
        basis_noise = ("pilot run %s: error bar %.5f Ha from %d kept blocks (%g Ha^-1 of data, %d walkers), "
                       "autocorrelation time %.2f blocks" % (m["key"], m["error"], m["kept"], m["data_time"], m["walkers"],
                                                             m["tau_blocks"]))
        choices = {
            "timestep": {"value": tau, "source": "pilot", "rule": choice.rule, "basis": choice.text},
            "steps": {"value": final.steps, "source": "rule",
                      "basis": "block length %g Ha^-1 / time step %g" % (float(o["block_time"]), tau)},
            "warmup_steps": {"value": final.warmup_steps, "source": "rule",
                             "basis": "transient %g Ha^-1 / time step %g" % (float(o["transient"]), tau)},
            "walkers": {"value": walkers, "source": "rule",
                        "basis": ("target %d, rounded up to a multiple of %d ranks" % (target, ranks)) if ranks
                        else "target %d (ranks not given)" % target},
            "discard_blocks": {"value": final.discard, "source": "rule",
                               "basis": "transient %g Ha^-1 / block length %g Ha^-1" % (float(o["transient"]), final.block_time)},
            "jastrow": dict(self.state["jastrow"]),
            "final_blocks": {"value": final.blocks, "source": "pilot" if not o["final_blocks"] else "given",
                             "basis": "F1 and F2 for %.3g kcal/mol; noise 2 T_c V = %.4g Ha from %s"
                                      % (float(o["sigma_target"]), m["noise"], basis_noise) if not o["final_blocks"]
                             else "given; expected error bar %.3f kcal/mol from %s" % (final.sigma_expected, basis_noise)},
        }
        record = {
            "version": 1, "mode": o["mode"], "status": DONE, "choices": choices, "plan": plans,
            "analysis": {"discard_blocks": final.discard, "transient": float(o["transient"]),
                         "sigma_block_kcal": final.sigma_block, "error_bar": "autocorrelation time (stats.estimate)"},
            "measurements": {
                "reference_run": m["key"], "noise": m["noise"], "variance": variance, "t_corr": t_corr,
                "step_seconds": step_seconds, "acceptance": m["acceptance"], "acceptance_exponent": exponent,
                "slope": choice.slope, "slope_error": choice.slope_error, "slope_used": choice.slope_used,
                "slope_source": choice.slope_source, "limit_f6": choice.limit, "tau_99": choice.tau_99,
                "timestep_table": choice.table, "points": [measured[t] for t in sorted(measured, reverse=True)],
                "pilot_cost_core_hours": float(sum(self.engine.cost(p["key"]) for r in self.state["rounds"] for p in r["points"])),
            },
            "forecast_first_guess": fc, "guesses": guesses, "not_checked": list(choice.not_checked),
            "repeats": self.state["history"], "warnings": list(self.state["warnings"]),
            "options": dict(o),
        }
        out["autotune"] = record
        self._write_settings(out)
        self.state["status"] = DONE
        self.state["message"] = "time step %g, %d walkers, %d steps per block; a final run has %d blocks (%d discarded)" % (
            tau, walkers, final.steps, final.blocks, final.discard)
        self.say("DONE: " + self.state["message"])

    def _finish_fixed(self) -> None:
        o = self.options
        user = copy.deepcopy(self.state["settings"])
        for text in check_settings(user, ranks=o["ranks"], blocks=o["final_blocks"], sigma_target=o["sigma_target"],
                                   e_scf=o["e_scf"], transient=float(o["transient"])):
            self._warn(text)
        dmc = dict(_dmc_defaults()); dmc.update(user.get("dmc") or {})
        fc = None
        if self.state["n_electrons"] and o["e_scf"] is not None:
            fc = forecast(self.state["n_electrons"], float(o["e_scf"]), float(o["sigma_target"]), int(dmc["walkers"]),
                          float(dmc["timestep"]), int(dmc["steps"]) * float(dmc["timestep"]), float(o["transient"])).as_dict()
        user["autotune"] = {
            "version": 1, "mode": "fixed", "status": DONE, "choices": {}, "plan": {},
            "analysis": {"discard_blocks": stats.transient_blocks(int(dmc["steps"]), float(dmc["timestep"]), float(o["transient"])),
                         "transient": float(o["transient"]), "error_bar": "autocorrelation time (stats.estimate)"},
            "measurements": {}, "forecast_first_guess": fc, "guesses": [], "not_checked": [], "repeats": [],
            "warnings": list(self.state["warnings"]), "options": dict(o),
        }
        self._write_settings(user)
        self.state["status"] = DONE
        self.state["message"] = "mode fixed: the given settings are used unchanged (%d warning(s))" % len(self.state["warnings"])
        self.say("DONE: " + self.state["message"])
        for text in self.state["warnings"]:
            self.say("warning: " + text)
