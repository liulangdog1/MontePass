"""Checks on a running or finished QMC trace.

A *trace* is the dict of columns that :func:`montepass.stats.read_scalar` returns for a QMCPACK
``scalar.dat`` file (``LocalEnergy``, ``LocalEnergy_sq``, ``BlockWeight``, ``BlockCPU``,
``AcceptRatio``; energies in hartree, one row per block). Every monitor is a pure function: it
gets a trace and the values that are expected for the run, and returns a list of
:class:`Finding` objects. An empty list means that the monitor found nothing.

    M1   no_progress      no new block for 5 expected block times
    M2   slow_job         a block takes more than 2.5 times the expected time
    M3   transient        the start-up part is not over after the planned discard
    M4   population       the block weight is off target, spreads too much, or drifts
    M5   outlier_blocks   a block energy more than 6 robust sigma from the median
    M6   acceptance       the acceptance falls below that of the first blocks (one block, or 10 blocks in a row)
    M7   acceptance       the mean acceptance is below 0.99 (time step too large)
    M8   variance         the DMC variance is more than 1.10 times the VMC variance, or changes during the run
    M9   seeds            two series of one geometry share random seeds, or this cannot be excluded
    M10  checkpoint       no checkpoint is written

:func:`check_trace` runs every monitor whose input is available and says which ones it could
not run.

Severity of a finding:

    info      for the record; the data are usable
    warning   the data are usable, but an action is advised (discard more blocks, extend the run,
              ask for more wall time)
    error     the data of this run must not enter a result as they are (re-run, or do not combine)

The default limits come from 188 production traces of 9 molecules with 28 to 62 valence
electrons (2048 walkers, time step 0.005 Ha^-1, 500 steps per block). The measured range of each
quantity is given next to its limit in :class:`Limits`. They were not tested for other walker
counts; see the note on ``walkers_reference``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import stats
from .units import HARTREE_TO_KCAL

INFO = "info"
WARNING = "warning"
ERROR = "error"
_RANK = {INFO: 0, WARNING: 1, ERROR: 2}

M1 = "M1_NO_PROGRESS"
M2 = "M2_SLOW_JOB"
M3 = "M3_TRANSIENT"
M4 = "M4_POPULATION"
M5 = "M5_OUTLIER_BLOCKS"
M6 = "M6_ACCEPTANCE_DROP"
M7 = "M7_ACCEPTANCE_LOW"
M8 = "M8_VARIANCE"
M9 = "M9_SEEDS"
M10 = "M10_CHECKPOINT"


@dataclass
class Finding:
    """One result of a monitor."""

    code: str                   # which monitor, for example "M7_ACCEPTANCE_LOW"
    severity: str               # "info", "warning" or "error"
    message: str                # one sentence with the numbers
    data: Dict[str, object] = field(default_factory=dict)       # the same numbers, for programs

    def as_dict(self) -> Dict[str, object]:
        return {"code": self.code, "severity": self.severity, "message": self.message, "data": stats.plain(self.data)}


class Findings(list):
    """List of :class:`Finding` objects returned by :func:`check_trace`.

    Two more attributes record what was done: ``checked`` (codes of the monitors that ran) and
    ``skipped`` (code -> reason, for the monitors that could not run because an input was missing).
    A monitor that did not run has found nothing; it has not shown that the run is healthy.
    """

    def __init__(self, items: Sequence[Finding] = ()):
        super().__init__(items)
        self.checked: List[str] = []
        self.skipped: Dict[str, str] = {}

    def codes(self) -> List[str]:
        return [f.code for f in self]

    def worst(self) -> Optional[str]:
        """Highest severity among the findings, or None when there is none."""
        return max((f.severity for f in self), key=lambda s: _RANK[s]) if len(self) else None

    def errors(self) -> List[Finding]:
        return [f for f in self if f.severity == ERROR]

    def as_dicts(self) -> List[Dict[str, object]]:
        return [f.as_dict() for f in self]


@dataclass(frozen=True)
class Limits:
    """Limits of the monitors. "Measured" is the range seen in healthy production runs."""

    stall_factor: float = 5.0           # M1: no new block for this many expected block times
    slow_factor: float = 2.5            # M2: seconds per block against the expected value (measured between jobs: up to 4)
    transient_head: int = 3             # M3: this many first kept blocks are compared with the rest
    transient_sigma: float = 3.0        # M3: limit of that difference and of the drift between the two halves, in sigma
    mser_factor: float = 3.0            # M3: the data-driven discard may be this many times the planned discard
    discard_fraction: float = 0.25      # M3: largest share of a run that may be discarded
    weight_mean: float = 0.02           # M4: mean block weight against walkers * steps (measured: 0.36 %)
    weight_spread: float = 0.03         # M4: relative standard deviation of the block weight (measured: at most 1.5 %)
    weight_drift: float = 0.02          # M4: first half against second half of the block weight (measured: at most 0.5 %)
    walkers_reference: int = 2048       # M4: walker count of the measurements. For another count the three limits are
                                        #     multiplied by sqrt(2048 / walkers): the relative fluctuation of a population
                                        #     grows like 1 / sqrt(walkers). This scaling is an assumption, not a measurement.
    outlier_sigma: float = 6.0          # M5: robust sigma (measured: no block beyond 5 in 74,259 blocks)
    acceptance_drop: float = 0.001      # M6: smallest fall of the acceptance that counts (measured range inside a
                                        #     production run with 2048 walkers at time step 0.005: 0.00009)
    acceptance_block_sigma: float = 8.0 # M6: one block counts only if it is also this many robust sigma of the block
                                        #     acceptance below the reference
    acceptance_window: int = 10         # M6: a fall that lasts: the median of this many blocks in a row ...
    acceptance_window_sigma: float = 3.0  # M6: ... is this many robust sigma of the block acceptance below the reference
    acceptance_min: float = 0.99        # M7: mean acceptance (measured at time step 0.005: 0.9976 to 0.9984)
    variance_ratio: float = 1.10        # M8: DMC variance / VMC variance (measured: 0.995 to 1.025)
    variance_change: float = 0.10       # M8: change of the block variance between the two halves of a run
    extreme_block: float = 20.0         # M8: a block variance above this many medians is "one extreme sample"


LIMITS = Limits()


def _column(trace: Dict[str, np.ndarray], name: str) -> np.ndarray:
    if name not in trace:
        raise ValueError("the trace has no column %s (columns: %s)" % (name, ", ".join(sorted(trace))))
    return np.asarray(trace[name], dtype=float).reshape(-1)


# ----------------------------------------------------------------------------- M1, M2: progress and speed
def no_progress(seconds_since_last_row: float, expected_block_seconds: float, limits: Limits = LIMITS) -> List[Finding]:
    """M1. The job wrote no new block for ``stall_factor`` (5) expected block times.

    ``seconds_since_last_row`` is the time since the trace file last changed. Use it only for a job
    that is still running. Action: read the error file of the job, resubmit once, then report."""
    if not expected_block_seconds > 0.0:
        raise ValueError("no_progress: the expected time of a block must be positive, got %r" % expected_block_seconds)
    limit = limits.stall_factor * expected_block_seconds
    if seconds_since_last_row > limit:
        return [Finding(M1, ERROR, "no new block for %.0f s. One block is expected every %.1f s; the limit is %.0f s "
                                   "(%g block times)." % (seconds_since_last_row, expected_block_seconds, limit,
                                                          limits.stall_factor),
                        {"seconds_since_last_row": seconds_since_last_row, "expected_block_seconds": expected_block_seconds,
                         "limit_seconds": limit})]
    return []


def slow_job(trace: Dict[str, np.ndarray], expected_block_seconds: float, blocks_requested: Optional[int] = None,
             wall_seconds_left: Optional[float] = None, limits: Limits = LIMITS) -> List[Finding]:
    """M2. The median time of a block (column ``BlockCPU``) is more than ``slow_factor`` (2.5) times
    the expected value.

    ``expected_block_seconds`` must be the expectation for the block size of THIS run (the time per
    walker and step of a pilot run times walkers times steps, divided by the ranks). With
    ``blocks_requested`` and ``wall_seconds_left`` the monitor also says whether the blocks that
    are missing fit into the wall time that is left."""
    if not expected_block_seconds > 0.0:
        raise ValueError("slow_job: the expected time of a block must be positive, got %r" % expected_block_seconds)
    cpu = _column(trace, "BlockCPU")
    if len(cpu) == 0:
        return []
    med = float(np.median(cpu))
    ratio = med / expected_block_seconds
    out = []
    if ratio > limits.slow_factor:
        out.append(Finding(M2, WARNING, "a block takes %.2f s (median of %d blocks). Expected: %.2f s. The job is %.1f "
                                        "times slower; the limit is %.1f." % (med, len(cpu), expected_block_seconds, ratio,
                                                                              limits.slow_factor),
                           {"median_block_seconds": med, "expected_block_seconds": expected_block_seconds, "ratio": ratio}))
    if blocks_requested is not None and wall_seconds_left is not None:
        missing = max(0, int(blocks_requested) - len(cpu))
        needed = missing * med
        if needed > wall_seconds_left:
            out.append(Finding(M2, WARNING, "the run cannot finish inside its wall time: %d blocks are missing, they need "
                                            "%.0f s at %.2f s per block, and %.0f s are left."
                               % (missing, needed, med, wall_seconds_left),
                               {"cannot_finish": True, "missing_blocks": missing, "seconds_needed": needed,
                                "seconds_left": float(wall_seconds_left), "median_block_seconds": med}))
    return out


# ----------------------------------------------------------------------------- M3: transient
def transient(trace: Dict[str, np.ndarray], drop: int, blocks_requested: Optional[int] = None, finished: bool = False,
              limits: Limits = LIMITS) -> List[Finding]:
    """M3. The start-up part of the run is not over after the planned discard of ``drop`` blocks.

    Four tests:

    * the mean of the first 3 kept blocks is more than 3 sigma from the mean of the other kept
      blocks (the sigma takes the correlation between neighbouring blocks into account);
    * the data-driven rule (:func:`montepass.stats.mser`) asks for more than 3 times the planned discard;
    * the first half of the kept blocks differs from the second half by more than 3 sigma;
    * the discard is more than 25 % of the run (tested for a finished run, or against
      ``blocks_requested``): the run is short for its transient.

    Action: discard more blocks; if more than 25 % of the run is discarded, extend the run."""
    e = _column(trace, "LocalEnergy")
    w = _column(trace, "BlockWeight") if "BlockWeight" in trace else np.ones(len(e))
    n = len(e)
    drop = int(drop)
    if drop < 0:
        raise ValueError("transient: the number of discarded blocks is negative (%d)" % drop)
    kept, wk = e[drop:], w[drop:]
    out = []
    k = int(limits.transient_head)
    if len(kept) >= k + 16:
        head, rest = kept[:k], kept[k:]
        est = stats.estimate(rest, wk[k:])
        rho = stats.autocorrelation(rest, max_lag=k - 1)
        factor = 1.0 + 2.0 * sum((1.0 - t / k) * float(rho[t]) for t in range(1, k))
        factor = min(float(k), max(1.0, factor))            # variance of a mean of k correlated blocks / (variance / k)
        sd = float(rest.std(ddof=1))
        err = math.sqrt(sd * sd * factor / k + est.error ** 2)
        if err > 0.0:
            z = (float(head.mean()) - est.mean) / err
            if abs(z) > limits.transient_sigma:
                out.append(Finding(M3, WARNING, "the first %d kept blocks (from block %d) are %+.3f kcal/mol from the mean of "
                                                "the other %d blocks: %.1f sigma, the limit is %.1f. The transient is not over "
                                                "after %d discarded blocks."
                                   % (k, drop, (float(head.mean()) - est.mean) * HARTREE_TO_KCAL, len(rest), abs(z),
                                      limits.transient_sigma, drop),
                                   {"test": "first_blocks", "z": z, "difference_hartree": float(head.mean()) - est.mean,
                                    "drop": drop}))
    if n >= 10:
        d = stats.mser(e)
        if d > limits.mser_factor * max(drop, 1):
            out.append(Finding(M3, WARNING, "the data ask for a discard of %d blocks (rule MSER). That is more than %g times "
                                            "the planned discard of %d blocks." % (d, limits.mser_factor, drop),
                               {"test": "mser", "mser_blocks": d, "drop": drop}))
    if len(kept) >= 16:
        diff, err, z = stats.drift_test(kept, wk)
        if abs(z) > limits.transient_sigma:
            out.append(Finding(M3, WARNING, "the first half of the %d kept blocks is %+.3f +/- %.3f kcal/mol from the second "
                                            "half: %.1f sigma, the limit is %.1f. The energy still drifts."
                               % (len(kept), diff * HARTREE_TO_KCAL, err * HARTREE_TO_KCAL, abs(z), limits.transient_sigma),
                               {"test": "drift", "z": z, "difference_hartree": diff, "error_hartree": err, "drop": drop}))
    total = n if finished else (int(blocks_requested) if blocks_requested else None)
    if total and drop > limits.discard_fraction * total:
        out.append(Finding(M3, WARNING, "%d of %d blocks are discarded (%.0f %%; the limit is %.0f %%). The run is short for "
                                        "its transient: extend it." % (drop, total, 100.0 * drop / total,
                                                                      100.0 * limits.discard_fraction),
                           {"test": "discard_fraction", "drop": drop, "blocks": total, "fraction": drop / float(total)}))
    return out


# ----------------------------------------------------------------------------- M4: population
def population(trace: Dict[str, np.ndarray], walkers: Optional[int] = None, steps: Optional[int] = None, drop: int = 0,
               limits: Limits = LIMITS) -> List[Finding]:
    """M4. The walker population is not under control. The column ``BlockWeight`` is the sum of the
    population weight over the steps of a block, so its target is walkers * steps.

    Three tests on the blocks after ``drop``:

    * the mean block weight is more than 2 % from walkers * steps (needs ``walkers`` and ``steps``);
    * the relative standard deviation between blocks is above 3 %;
    * the first half differs from the second half by more than 2 %.

    The first and the third test also need 3 standard errors, so that a short run does not fire by
    chance. With ``walkers`` given, the limits are multiplied by sqrt(2048 / walkers) (see
    :class:`Limits`). Action: keep the data out of every result and run again with a new seed.
    Fewer than 8 blocks, or block weights that are all equal (a source that does not record the
    population), give no finding."""
    w = _column(trace, "BlockWeight")[int(drop):]
    n = len(w)
    if n < 8 or float(np.ptp(w)) == 0.0:
        return []
    mean = float(w.mean())
    if not mean > 0.0:
        return [Finding(M4, ERROR, "the mean block weight is %g: it must be positive." % mean, {"mean_weight": mean})]
    scale = math.sqrt(limits.walkers_reference / float(walkers)) if walkers else 1.0
    est = stats.estimate(w)
    rel_sd = float(w.std(ddof=1)) / mean
    out = []
    if walkers and steps:
        target = float(walkers) * float(steps)
        dev = mean / target - 1.0
        if abs(dev) > limits.weight_mean * scale and abs(dev) > 3.0 * est.error / target:
            out.append(Finding(M4, ERROR, "the mean block weight %.0f is %+.2f %% from walkers x steps = %d x %d = %.0f. "
                                          "The limit is %.2f %%." % (mean, 100 * dev, walkers, steps, target,
                                                                    100 * limits.weight_mean * scale),
                               {"test": "mean", "mean_weight": mean, "target": target, "deviation": dev,
                                "limit": limits.weight_mean * scale}))
    if rel_sd > limits.weight_spread * scale:
        out.append(Finding(M4, ERROR, "the block weight spreads by %.2f %% between blocks (relative standard deviation of %d "
                                      "blocks). The limit is %.2f %%." % (100 * rel_sd, n, 100 * limits.weight_spread * scale),
                           {"test": "spread", "relative_sd": rel_sd, "limit": limits.weight_spread * scale}))
    diff, err, z = stats.drift_test(w)
    if abs(diff) / mean > limits.weight_drift * scale and abs(z) > 3.0:
        out.append(Finding(M4, ERROR, "the block weight drifts: first half minus second half = %+.2f %% (%.1f sigma). The "
                                      "limit is %.2f %%." % (100 * diff / mean, abs(z), 100 * limits.weight_drift * scale),
                           {"test": "drift", "relative_drift": diff / mean, "z": z, "limit": limits.weight_drift * scale}))
    return out


# ----------------------------------------------------------------------------- M5: outlier blocks
def outlier_blocks(trace: Dict[str, np.ndarray], drop: int = 0, limits: Limits = LIMITS) -> List[Finding]:
    """M5. Block energies far from the others, after ``drop`` blocks.

    The scale is robust: 1.4826 times the median absolute deviation from the median, which one bad
    block cannot change. A block more than ``outlier_sigma`` (6) such sigma from the median is an
    outlier. One outlier is a warning (mark the run in the record). Two or more are an error
    (stop and run again). Fewer than 8 blocks give no finding."""
    e = _column(trace, "LocalEnergy")[int(drop):]
    if len(e) < 8:
        return []
    med = float(np.median(e))
    sigma = 1.4826 * float(np.median(np.abs(e - med)))
    if not sigma > 0.0:
        return []
    dev = np.abs(e - med) / sigma
    idx = np.flatnonzero(dev > limits.outlier_sigma)
    if len(idx) == 0:
        return []
    blocks = [int(i) + int(drop) for i in idx]
    worst = int(idx[np.argmax(dev[idx])])
    return [Finding(M5, WARNING if len(idx) == 1 else ERROR,
                    "%d block(s) are more than %g robust sigma from the median: block(s) %s. The largest deviation is "
                    "%.1f sigma = %+.2f kcal/mol at block %d (robust sigma of a block: %.2f kcal/mol)."
                    % (len(idx), limits.outlier_sigma, ", ".join(str(b) for b in blocks[:10]) + (" ..." if len(blocks) > 10 else ""),
                       float(dev[worst]), (float(e[worst]) - med) * HARTREE_TO_KCAL, worst + int(drop), sigma * HARTREE_TO_KCAL),
                    {"blocks": blocks, "robust_sigma_hartree": sigma, "largest_sigma": float(dev[worst])})]


# ----------------------------------------------------------------------------- M6, M7: acceptance
def acceptance(trace: Dict[str, np.ndarray], drop: int = 0, limits: Limits = LIMITS) -> List[Finding]:
    """M7 and M6, from the column ``AcceptRatio``.

    M7: the mean acceptance after ``drop`` blocks is below ``acceptance_min`` (0.99). The time step is
    too large and the run must not be used for results.

    M6: the acceptance falls during the run. A walker may have stopped moving. Action: stop and run
    again with a new seed. The reference is the median of the first quarter of the run (at least 10
    blocks), and ``s`` is the robust standard deviation of the acceptance of one block (the larger of
    two estimates: from the reference part, and from the differences of neighbouring blocks). Two tests:

    - one block after the first 10 is more than ``max(acceptance_drop, 8 s)`` below the reference;
    - the median of 10 blocks in a row is more than ``max(acceptance_drop, 3 s)`` below the reference
      (a fall that lasts, which is what a walker that stopped moving gives).

    The scale ``s`` matters for short blocks, few walkers or a large time step: there the acceptance of
    one block scatters by more than 0.001 in a healthy run (measured: 0.0005 for 512 walkers at time
    step 0.02), and a fixed limit of 0.001 would report 2 % of the blocks."""
    a = _column(trace, "AcceptRatio")
    out = []
    kept = a[int(drop):]
    if len(kept):
        mean = float(kept.mean())
        if mean < limits.acceptance_min:
            out.append(Finding(M7, ERROR, "the mean acceptance is %.4f (%d blocks). The limit is %.2f: the time step is "
                                          "too large." % (mean, len(kept), limits.acceptance_min),
                               {"acceptance": mean, "limit": limits.acceptance_min}))
    if len(a) > 10:
        n_ref = max(10, len(a) // 4)
        ref = float(np.median(a[:n_ref]))
        rest = a[10:]
        # scatter of the acceptance of one block. Two robust estimates, the larger one is used: from the
        # reference part, and from the differences of neighbouring blocks (a fall changes one difference only)
        s_ref = 1.4826 * float(np.median(np.abs(a[:n_ref] - ref)))
        s_diff = 1.4826 * float(np.median(np.abs(np.diff(a) - np.median(np.diff(a))))) / np.sqrt(2.0)
        s = max(s_ref, s_diff)
        limit_one = max(limits.acceptance_drop, limits.acceptance_block_sigma * s)
        limit_row = max(limits.acceptance_drop, limits.acceptance_window_sigma * s)
        low = np.flatnonzero(rest < ref - limit_one) + 10
        w = int(limits.acceptance_window)
        rows = np.zeros(0, dtype=int)
        if len(rest) >= w:
            med = np.median(np.lib.stride_tricks.sliding_window_view(rest, w), axis=1)
            rows = np.flatnonzero(med < ref - limit_row) + 10          # first block of every low row of w blocks
        if len(low) or len(rows):
            parts = []
            if len(low):
                worst = int(low[np.argmin(a[low])])
                parts.append("%d block(s) have an acceptance more than %.2g below the first %d blocks (%.5f); the lowest "
                             "is %.5f at block %d" % (len(low), limit_one, n_ref, ref, float(a[worst]), worst))
            if len(rows):
                parts.append("from block %d on, the median acceptance of %d blocks in a row is more than %.2g below the "
                             "first %d blocks (%.5f)" % (int(rows[0]), w, limit_row, n_ref, ref))
            out.append(Finding(M6, ERROR, "; ".join(parts)[0].upper() + "; ".join(parts)[1:] + ". A walker may have "
                                                                                             "stopped moving.",
                               {"blocks": [int(i) for i in low], "rows": [int(i) for i in rows[:50]], "reference": ref,
                                "robust_sigma": s, "lowest": float(a.min())}))
    return out


# ----------------------------------------------------------------------------- M8: variance
def variance(trace: Dict[str, np.ndarray], vmc_variance: Optional[float] = None, drop: int = 0,
             limits: Limits = LIMITS) -> List[Finding]:
    """M8. The variance of the local energy shows a problem of the trial wave function.

    * With ``vmc_variance``: the DMC variance (after ``drop`` blocks) is more than ``variance_ratio``
      (1.10) times the VMC variance. If the excess comes from single blocks whose variance is above
      20 times the median block variance, the finding is only "info": one extreme sample of the
      local energy, not a wave-function problem.
    * The median block variance of the second half of the run differs from that of the first half
      by more than ``variance_change`` (10 %).

    Action for an error: stop, and repeat the Jastrow optimization of this point."""
    e = _column(trace, "LocalEnergy")
    e2 = _column(trace, "LocalEnergy_sq")
    w = _column(trace, "BlockWeight") if "BlockWeight" in trace else np.ones(len(e))
    drop = int(drop)
    e, e2, w = e[drop:], e2[drop:], w[drop:]
    if len(e) < 2:
        return []
    out = []
    v_block = e2 - e * e
    med = float(np.median(v_block))
    total = stats.local_energy_variance(e, e2, w)
    if vmc_variance is not None:
        if not vmc_variance > 0.0:
            raise ValueError("variance: the VMC variance must be positive, got %r" % vmc_variance)
        ratio = total / vmc_variance
        if ratio > limits.variance_ratio:
            extreme = v_block > limits.extreme_block * med if med > 0.0 else np.zeros(len(v_block), dtype=bool)
            ratio_rest = None
            if extreme.any() and (~extreme).sum() >= 2:
                ratio_rest = stats.local_energy_variance(e[~extreme], e2[~extreme], w[~extreme]) / vmc_variance
            if ratio_rest is not None and ratio_rest <= limits.variance_ratio:
                blocks = [int(i) + drop for i in np.flatnonzero(extreme)]
                out.append(Finding(M8, INFO, "DMC variance %.4f = %.2f times the VMC variance %.4f, caused by extreme "
                                             "samples in block(s) %s (block variance up to %.1f, median %.3f). Without "
                                             "these blocks the ratio is %.3f, inside the limit of %.2f."
                                   % (total, ratio, vmc_variance, ", ".join(str(b) for b in blocks), float(v_block.max()),
                                      med, ratio_rest, limits.variance_ratio),
                                   {"test": "ratio", "ratio": ratio, "ratio_without_extreme_blocks": ratio_rest,
                                    "extreme_blocks": blocks, "dmc_variance": total, "vmc_variance": vmc_variance}))
            else:
                out.append(Finding(M8, ERROR, "DMC variance %.4f = %.3f times the VMC variance %.4f. The limit is %.2f."
                                   % (total, ratio, vmc_variance, limits.variance_ratio),
                                   {"test": "ratio", "ratio": ratio, "dmc_variance": total, "vmc_variance": vmc_variance}))
    if len(v_block) >= 16 and med > 0.0:
        h = len(v_block) // 2
        first, second = float(np.median(v_block[:h])), float(np.median(v_block[h:]))
        change = (second - first) / med
        if abs(change) > limits.variance_change:
            out.append(Finding(M8, ERROR, "the block variance changed by %+.0f %% during the run (median of the first half "
                                          "%.4f, of the second half %.4f). The limit is %.0f %%."
                               % (100 * change, first, second, 100 * limits.variance_change),
                               {"test": "change", "change": change, "first_half": first, "second_half": second}))
    return out


# ----------------------------------------------------------------------------- M9: seeds
def seeds(ranges: Dict[str, object], margin: float = 1.0) -> List[Finding]:
    """M9. Random seeds of several series of one geometry.

    ``ranges`` maps the name of each series to its seed ranges (the result of
    :func:`montepass.stats.read_seed_ranges` for the QMCPACK output of the series; None or an empty
    list when the output could not be read). Every pair is tested with
    :func:`montepass.stats.seed_overlap`. A pair that overlaps, and a pair that cannot be tested,
    both give an error: series are combined only when their seeds are proven to differ."""
    names = list(ranges)
    out = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            verdict = stats.seed_overlap(ranges[a], ranges[b], margin)
            if verdict == stats.DISJOINT:
                continue
            ra, rb = stats.as_seed_ranges(ranges[a]), stats.as_seed_ranges(ranges[b])
            if verdict == stats.UNKNOWN:
                missing = [name for name, r in ((a, ra), (b, rb)) if not r]
                text = ("the seed range of series %s could not be read, so it is not proven that series %s and %s used "
                        "different random seeds. Do not combine them." % (" and ".join(missing), a, b))
            else:
                text = ("series %s (seed primes %s) and series %s (seed primes %s) share random seeds or are closer than "
                        "the safety margin. They are not independent: do not combine them; run one again with another seed."
                        % (a, _ranges_text(ra), b, _ranges_text(rb)))
            out.append(Finding(M9, ERROR, text, {"series": [a, b], "verdict": verdict, "ranges": [ra, rb], "margin": margin}))
    return out


def _ranges_text(ranges) -> str:
    return ", ".join("%d-%d" % (lo, hi) for lo, hi in ranges)


# ----------------------------------------------------------------------------- M10: checkpoint
def checkpoint(interval: int, blocks_written: int, file_exists: bool) -> List[Finding]:
    """M10. ``interval`` is the ``checkpoint`` value of the DMC input (blocks between two checkpoint
    files; -1 or 0 = none). Without a checkpoint a run that is stopped cannot be continued: a new
    run must pass the transient again."""
    if interval is None or int(interval) <= 0:
        return [Finding(M10, INFO, "checkpoint = %s: the run writes no checkpoint. A stopped run cannot be continued and "
                                   "needs a new transient." % interval, {"interval": interval})]
    if blocks_written >= int(interval) and not file_exists:
        return [Finding(M10, WARNING, "no checkpoint file after %d blocks; one was expected every %d blocks."
                        % (blocks_written, int(interval)), {"interval": int(interval), "blocks_written": int(blocks_written)})]
    return []


# ----------------------------------------------------------------------------- all monitors of one trace
def check_trace(trace: Dict[str, np.ndarray], drop: Optional[int] = None, steps: Optional[int] = None,
                timestep: Optional[float] = None, walkers: Optional[int] = None, vmc_variance: Optional[float] = None,
                expected_block_seconds: Optional[float] = None, seconds_since_last_row: Optional[float] = None,
                finished: bool = True, blocks_requested: Optional[int] = None, wall_seconds_left: Optional[float] = None,
                seed_ranges: Optional[Dict[str, object]] = None, transient_time: float = 10.0,
                limits: Limits = LIMITS) -> Findings:
    """Run every monitor that applies to a trace and return all findings.

    trace                    dict of columns (``stats.read_scalar``). ``LocalEnergy`` is needed; a monitor
                             whose column is missing is skipped and listed in ``result.skipped``.
    drop                     planned number of discarded start-up blocks. Default: from ``steps`` and
                             ``timestep`` (``transient_time`` Ha^-1); without them the data-driven rule.
    steps, timestep, walkers settings of the run (steps per block, time step in Ha^-1, target population).
    vmc_variance             variance of the VMC run that made the walkers (hartree^2), for M8.
    expected_block_seconds   expected time of one block of this run, for M1 and M2.
    seconds_since_last_row   time since the trace last changed, for M1 (only for a running job).
    finished                 False while the job runs.
    blocks_requested, wall_seconds_left
                             for the wall-time test of M2 and the discard test of M3.
    seed_ranges              name of a series -> its seed ranges, for M9.

    The result is a list (:class:`Findings`) with the attributes ``checked`` and ``skipped``. One
    outlier block together with a population finding becomes an error, as two outlier blocks are."""
    res = Findings()
    if "LocalEnergy" not in trace:
        raise ValueError("check_trace: the trace has no column LocalEnergy (columns: %s)" % ", ".join(sorted(trace)))
    n = len(_column(trace, "LocalEnergy"))
    if drop is None:
        if steps is not None and timestep is not None:
            drop = stats.transient_blocks(steps, timestep, transient_time)
        else:
            drop = stats.mser(_column(trace, "LocalEnergy"))
    drop = int(drop)

    def run(code: str, ok: bool, reason: str, call) -> None:
        if not ok:
            res.skipped[code] = reason
            return
        res.checked.append(code)
        res.extend(call())

    block_seconds = expected_block_seconds
    run(M1, (not finished) and seconds_since_last_row is not None and block_seconds is not None,
        "the run is finished" if finished else "the time since the last block or the expected block time is not given",
        lambda: no_progress(seconds_since_last_row, block_seconds, limits))
    run(M2, "BlockCPU" in trace and block_seconds is not None and n > 0,
        "no column BlockCPU" if "BlockCPU" not in trace else "the expected block time is not given",
        lambda: slow_job(trace, block_seconds, blocks_requested if not finished else None,
                         wall_seconds_left if not finished else None, limits))
    run(M3, n > drop, "only %d block(s), %d are discarded" % (n, drop),
        lambda: transient(trace, drop, blocks_requested, finished, limits))
    has_weight = "BlockWeight" in trace and n - drop >= 8 and float(np.ptp(_column(trace, "BlockWeight")[drop:])) > 0.0
    run(M4, has_weight, "no column BlockWeight" if "BlockWeight" not in trace else
        ("fewer than 8 kept blocks" if n - drop < 8 else "the block weights are all equal: no population is recorded"),
        lambda: population(trace, walkers, steps, drop, limits))
    run(M5, n - drop >= 8, "fewer than 8 kept blocks", lambda: outlier_blocks(trace, drop, limits))
    for code in (M6, M7):
        if "AcceptRatio" not in trace:
            res.skipped[code] = "no column AcceptRatio"
        else:
            res.checked.append(code)
    if "AcceptRatio" in trace:
        res.extend(acceptance(trace, drop, limits))
    run(M8, "LocalEnergy_sq" in trace and n - drop >= 2,
        "no column LocalEnergy_sq" if "LocalEnergy_sq" not in trace else "fewer than 2 kept blocks",
        lambda: variance(trace, vmc_variance, drop, limits))
    if "LocalEnergy_sq" in trace and vmc_variance is None:
        res.skipped[M8 + "_RATIO"] = "the VMC variance is not given: only the change of the variance during the run was tested"
    run(M9, seed_ranges is not None and len(seed_ranges) >= 2, "fewer than 2 series with seed ranges given",
        lambda: seeds(seed_ranges))
    if any(f.code == M4 for f in res):
        for f in res:
            if f.code == M5 and f.severity == WARNING:
                f.severity = ERROR
                f.message += " A population finding is present too: stop and run again."
    return res
