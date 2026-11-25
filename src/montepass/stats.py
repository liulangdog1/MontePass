"""Statistics of QMCPACK block traces (``*.scalar.dat`` files).

The module has five parts:

1. reading a trace by column name (:func:`read_scalar`, :func:`energy_weight`);
2. error bars of a weighted mean: the reblocking rule of protocol v1 (:func:`reblock`) and the
   error bar from the integrated autocorrelation time (:func:`autocorrelation_time`,
   :func:`estimate`);
3. the start-up part of a DMC run (:func:`transient_blocks`, :func:`mser`, :func:`drift_test`,
   :func:`equilibration`);
4. the combination of independent runs (:func:`combine`);
5. the random seeds of a QMCPACK run (:func:`seed_ranges`, :func:`seed_overlap`).

All functions of parts 2 to 4 work on plain arrays and do not care about the unit of the
energies. A "block" is one row of a trace.
"""
from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


def read_scalar(path: str) -> Dict[str, np.ndarray]:
    """Read a QMCPACK ``scalar.dat`` file into a dict of columns, keyed by the header names.

    The file may still be growing while its job runs, so a last line that is
    incomplete (wrong number of fields, or a field that is not a number) is skipped.
    """
    names = None
    rows = []
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if not parts:
                continue
            if parts[0].startswith("#"):
                names = parts[1:] if parts[0] == "#" else [parts[0][1:]] + parts[1:]
                continue
            if names is None or len(parts) != len(names):
                continue
            try:
                rows.append([float(p) for p in parts])
            except ValueError:
                continue
    if names is None:
        raise ValueError("no header line in %s" % path)
    data = np.array(rows, dtype=float).reshape(len(rows), len(names))
    return {name: data[:, i] for i, name in enumerate(names)}


def energy_weight(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Block energies (hartree) and block weights of one trace."""
    cols = read_scalar(path)
    energy = cols["LocalEnergy"]
    weight = cols["BlockWeight"] if "BlockWeight" in cols else np.ones_like(energy)
    return energy, weight


def reblock(energy: np.ndarray, weight: np.ndarray) -> Tuple[float, float]:
    """Weighted mean of the block energies and its reblocked standard error.

    The blocks are merged into groups of 1, 2, 4, ... blocks while at least 8
    groups remain. The standard error is computed at every group size and the
    largest one is returned. This rule protects against serial correlation
    between blocks. It is the rule of protocol v1.
    """
    mean = float(np.sum(energy * weight) / np.sum(weight))
    n = len(energy)
    best, size = 0.0, 1
    while n // size >= 8:
        nb = n // size
        b = np.array([np.sum(energy[k * size:(k + 1) * size] * weight[k * size:(k + 1) * size]) /
                      np.sum(weight[k * size:(k + 1) * size]) for k in range(nb)])
        best = max(best, float(np.sqrt(b.var(ddof=1) / nb)))
        size *= 2
    return mean, best


# ----------------------------------------------------------------------------- small helpers
def plain(value):
    """A value that ``json`` can write: numpy numbers become Python numbers, and a number that
    is not finite becomes None."""
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return [plain(v) for v in value.tolist()]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    return value


def _arrays(energy, weight, drop: int, who: str) -> Tuple[np.ndarray, np.ndarray]:
    """Checked copies of the block energies and weights after the first ``drop`` blocks."""
    e = np.asarray(energy, dtype=float).reshape(-1)
    w = np.ones(len(e)) if weight is None else np.asarray(weight, dtype=float).reshape(-1)
    if len(w) != len(e):
        raise ValueError("%s: %d block energies but %d block weights" % (who, len(e), len(w)))
    drop = int(drop)
    if drop < 0:
        raise ValueError("%s: the number of discarded blocks is negative (%d)" % (who, drop))
    e, w = e[drop:], w[drop:]
    if not np.all(np.isfinite(e)) or not np.all(np.isfinite(w)):
        raise ValueError("%s: a block energy or a block weight is not a finite number" % who)
    if np.any(w < 0) or (len(w) and not w.sum() > 0):
        raise ValueError("%s: block weights must not be negative and their sum must be positive" % who)
    return e, w


# ----------------------------------------------------------------------------- autocorrelation and error bars
def autocorrelation(x: np.ndarray, max_lag: Optional[int] = None) -> np.ndarray:
    """Normalized autocorrelation function rho(0), rho(1), ..., rho(max_lag) of a series.

    rho(t) = c(t) / c(0). c(t) is the mean of d[i] * d[i + t] over the n - t pairs that exist, and
    d is the deviation from the mean of the series. ``max_lag`` defaults to n // 2. A series whose
    values are all equal returns 1, 0, 0, ...
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    n = len(x)
    if n < 2:
        raise ValueError("autocorrelation: the series has %d value(s); at least 2 are needed" % n)
    max_lag = n // 2 if max_lag is None else int(max_lag)
    max_lag = max(0, min(max_lag, n - 1))
    d = x - x.mean()
    c0 = float(np.dot(d, d)) / n
    rho = np.zeros(max_lag + 1)
    rho[0] = 1.0
    if c0 <= 0.0:
        return rho
    for t in range(1, max_lag + 1):
        rho[t] = float(np.dot(d[:n - t], d[t:])) / (n - t) / c0
    return rho


def autocorrelation_time(x: np.ndarray, c: float = 5.0, max_lag: int = 200) -> Tuple[float, int, float]:
    """Integrated autocorrelation time of a block series, with an automatic window.

        tau(M) = 1/2 + rho(1) + rho(2) + ... + rho(M)

    The window M is the smallest lag with M >= c * tau(M) (c = 5). The sum stops there, because
    later terms add noise and no signal. Returns ``(tau, M, error of tau)`` in blocks.

    * tau = 0.5 means that the blocks are independent. The variance of the mean of n blocks is
      2 * tau times the variance of independent blocks ("statistical inefficiency" g = 2 tau).
    * tau is never returned below 0.5, so the error bar built from it is never smaller than the
      error bar of independent blocks.
    * The error of tau is tau * sqrt(2 * (2 M + 1) / n).
    * A series of fewer than 8 blocks is too short for an estimate: (0.5, 0, nan) is returned.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    n = len(x)
    if n < 8:
        return 0.5, 0, float("nan")
    rho = autocorrelation(x, max_lag=min(n // 2, int(max_lag)))
    tau, window = 0.5, 0
    for t in range(1, len(rho)):
        tau += float(rho[t])
        window = t
        if t >= c * max(tau, 0.5):
            break
    tau = max(tau, 0.5)
    return float(tau), int(window), float(tau * math.sqrt(2.0 * (2 * window + 1) / n))


@dataclass
class Estimate:
    """Mean of a block series with its error bars (same unit as the block energies)."""

    mean: float             # weighted mean of the blocks used
    error: float            # standard error from the autocorrelation time: error_naive * sqrt(2 * tau)
    error_reblock: float    # standard error of function ``reblock`` (largest value over block sizes); nan below 8 blocks
    error_naive: float      # standard error if the blocks were independent
    tau: float              # integrated autocorrelation time in blocks (0.5 = independent blocks)
    tau_error: float        # statistical error of tau; nan below 8 blocks
    window: int             # summation window of tau, in blocks
    n_blocks: int           # number of blocks used (after the discard)
    drop: int               # number of blocks discarded at the start

    def as_dict(self) -> Dict[str, object]:
        return plain(asdict(self))


def estimate(energy: np.ndarray, weight: Optional[np.ndarray] = None, drop: int = 0) -> Estimate:
    """Weighted mean of the blocks after the first ``drop`` blocks, with two error bars.

    ``error`` uses the measured autocorrelation time of the blocks:

        error = error_naive * sqrt(2 * tau)

    ``error_reblock`` is the rule of :func:`reblock` (largest standard error over block sizes 1, 2,
    4, ...). On 188 production traces of 9 molecules (28 to 62 valence electrons) ``error`` was
    calibrated: the difference of the two halves of a run, divided by its error bar, had a
    root-mean-square value of 0.98 (1.00 is exact). ``error_reblock`` was about 12 % larger than
    that. Use ``error`` to size and to stop runs. ``error_reblock`` is kept because protocol v1
    is defined with it.

    Limit of both error bars: the run must be long compared with the correlation time. On
    synthetic series of n correlated blocks (3000 series per case) the mean square of
    deviation / ``error`` was 0.92 to 0.97 for independent blocks, 1.14 for n = 200 tau, 1.5 for
    n = 50 tau and 2.1 for n = 20 tau (1.00 is exact). So with fewer than about 200 tau blocks the true
    error is larger than the reported one: by 7 % at 200 tau, by 24 % at 50 tau. Production blocks
    of 2.5 Ha^-1 have tau = 0.58 blocks, so about 120 blocks are enough.

    With unequal weights the naive error is that of a weighted mean; it equals
    ``std(ddof=1) / sqrt(n)`` when all weights are equal. Below 8 blocks the correlation time is
    not estimated (tau = 0.5), so ``error`` is the naive error and is only a rough value.
    Raises ValueError when fewer than 2 blocks remain or a value is not finite.
    """
    n_all = len(np.asarray(energy).reshape(-1))
    e, w = _arrays(energy, weight, drop, "estimate")
    n = len(e)
    if n < 2:
        raise ValueError("estimate: %d blocks, %d discarded: fewer than 2 blocks remain" % (n_all, int(drop)))
    sw = float(w.sum())
    mean = float(np.sum(w * e) / sw)
    n_eff = sw * sw / float(np.sum(w * w))
    var = float(np.sum(w * (e - mean) ** 2) / sw) * n / (n - 1.0)
    naive = math.sqrt(var / n_eff)
    tau, window, tau_err = autocorrelation_time(e)
    err_reblock = float(reblock(e, w)[1]) if n >= 8 else float("nan")
    return Estimate(mean=mean, error=float(naive * math.sqrt(2.0 * tau)), error_reblock=err_reblock,
                    error_naive=float(naive), tau=tau, tau_error=tau_err, window=window, n_blocks=n, drop=int(drop))


def local_energy_variance(energy: np.ndarray, energy_sq: np.ndarray, weight: Optional[np.ndarray] = None,
                          drop: int = 0) -> float:
    """Variance of the local energy from the columns ``LocalEnergy`` and ``LocalEnergy_sq`` of a
    trace: weighted mean of E^2 minus the square of the weighted mean of E, after ``drop`` blocks.
    For energies in hartree the result is in hartree^2."""
    e, w = _arrays(energy, weight, drop, "local_energy_variance")
    e2 = np.asarray(energy_sq, dtype=float).reshape(-1)[int(drop):]
    if len(e2) != len(e):
        raise ValueError("local_energy_variance: %d energies but %d squared energies" % (len(e), len(e2)))
    if len(e) < 1:
        raise ValueError("local_energy_variance: no block is left after the discard of %d blocks" % int(drop))
    sw = float(w.sum())
    return float(np.sum(w * e2) / sw - (np.sum(w * e) / sw) ** 2)


# ----------------------------------------------------------------------------- equilibration (start-up blocks)
def transient_blocks(steps: int, timestep: float, transient: float = 10.0) -> int:
    """Number of start-up blocks to discard for a transient of ``transient`` Ha^-1 of imaginary time:
    ceil(transient / (steps * timestep)).

    The default of 10 Ha^-1 is the longest transient measured on 188 production traces (no excess
    energy was left after 10 Ha^-1 for 28 to 62 valence electrons). Example: 500 steps per block at
    a time step of 0.005 Ha^-1 is 2.5 Ha^-1 per block, so 4 blocks are discarded.
    """
    block_time = float(steps) * float(timestep)
    if not block_time > 0.0:
        raise ValueError("transient_blocks: steps * timestep must be positive (steps %r, timestep %r)" % (steps, timestep))
    if transient <= 0.0:
        return 0
    return int(math.ceil(float(transient) / block_time - 1e-9))


def mser(energy: np.ndarray, batch: int = 1, max_fraction: float = 0.5) -> int:
    """Data-driven number of start-up blocks to discard (rule MSER-m, m = ``batch``).

    The series is averaged in groups of ``batch`` blocks. The rule returns the truncation d (a
    multiple of ``batch``, at most ``max_fraction`` of the series) for which the estimated variance
    of the mean of the remaining values, var(y[d:]) / (n - d), is smallest. A series of fewer than
    10 groups returns 0.

    ``batch = 1`` reacts to a short, strong transient and removes little else (on stationary data
    of 300 blocks it removed 1.2 blocks on average). It does not see a transient that is weaker
    than the noise of one block. Use it as a check next to :func:`transient_blocks`, not alone.
    """
    x = np.asarray(energy, dtype=float).reshape(-1)
    batch = max(1, int(batch))
    nb = len(x) // batch
    if nb < 10:
        return 0
    y = x[:nb * batch].reshape(nb, batch).mean(axis=1)
    y = y - y.mean()
    s1 = np.cumsum(y[::-1])[::-1]                       # sums of y[d:]
    s2 = np.cumsum((y * y)[::-1])[::-1]                 # sums of y[d:]**2
    d = np.arange(0, min(int(max_fraction * nb), nb - 2) + 1)
    kept = (nb - d).astype(float)
    value = (s2[d] - s1[d] ** 2 / kept) / kept ** 2
    return int(np.argmin(value)) * batch


def drift_test(energy: np.ndarray, weight: Optional[np.ndarray] = None) -> Tuple[float, float, float]:
    """Compare the first half of a series with the second half.

    Returns ``(difference, error, z)``: mean of the first half minus mean of the second half, the
    error of that difference (from the autocorrelation error bars of the two halves) and their
    ratio z. For a stationary series z is a standard normal number, so abs(z) > 3 happens by
    chance in about 0.3 % of the cases. A positive z means that the energy still falls.
    Raises ValueError below 4 blocks.
    """
    e, w = _arrays(energy, weight, 0, "drift_test")
    n = len(e)
    if n < 4:
        raise ValueError("drift_test: %d blocks; at least 4 are needed" % n)
    h = n // 2
    a, b = estimate(e[:h], w[:h]), estimate(e[h:], w[h:])
    diff = a.mean - b.mean
    err = math.hypot(a.error, b.error)
    if err > 0.0:
        z = diff / err
    else:
        z = 0.0 if diff == 0.0 else math.copysign(float("inf"), diff)
    return float(diff), float(err), float(z)


@dataclass
class Equilibration:
    """Result of :func:`equilibration`."""

    drop: int                       # blocks to discard: the larger of drop_physical and drop_mser
    drop_physical: Optional[int]    # from the transient time; None when steps or timestep were not given
    drop_mser: int                  # from the data (rule MSER)
    n_blocks: int                   # blocks in the series
    n_kept: int                     # blocks left after the discard
    drift: Optional[float]          # first half minus second half of the kept blocks; None below 4 kept blocks
    drift_error: Optional[float]
    drift_z: Optional[float]
    stationary: Optional[bool]      # abs(drift_z) <= z_max; None when the drift could not be tested
    message: str

    def as_dict(self) -> Dict[str, object]:
        return plain(asdict(self))


def equilibration(energy: np.ndarray, weight: Optional[np.ndarray] = None, steps: Optional[int] = None,
                  timestep: Optional[float] = None, transient: float = 10.0, batch: int = 1,
                  z_max: float = 3.0) -> Equilibration:
    """Number of start-up blocks to discard from a DMC trace, from three pieces.

    1. Physical rule: the transient lasts ``transient`` Ha^-1 of imaginary time (default 10), so
       ceil(transient / (steps * timestep)) blocks are discarded. Used when ``steps`` (steps per
       block) and ``timestep`` are given.
    2. Data check: the rule :func:`mser` on the whole series. When it asks for more blocks than
       the physical rule, the larger number is used.
    3. Drift test on the blocks that are kept: first half against second half
       (:func:`drift_test`). ``stationary`` is False when abs(z) > ``z_max``. Then the discard is
       not enough, or the run has another problem; the caller must decide (discard more, or
       extend the run).

    The function never raises for a short series: with fewer than 4 kept blocks the drift is not
    tested and ``stationary`` is None.
    """
    e, w = _arrays(energy, weight, 0, "equilibration")
    n = len(e)
    physical = None
    if steps is not None and timestep is not None:
        physical = transient_blocks(steps, timestep, transient)
    data = mser(e, batch=batch) if n else 0
    drop = max(physical or 0, data)
    kept = max(0, n - drop)
    notes = []
    if physical is None:
        notes.append("no physical rule (steps per block and time step not given): discard from the data only")
    if physical is not None and data > physical:
        notes.append("the data ask for %d discarded blocks, more than the %d of the %g Ha^-1 transient"
                     % (data, physical, transient))
    drift = drift_err = z = None
    stationary = None
    if kept >= 4:
        drift, drift_err, z = drift_test(e[drop:], w[drop:])
        stationary = bool(abs(z) <= z_max)
        if not stationary:
            notes.append("the kept blocks drift: first half minus second half = %.3g +/- %.3g (%.1f sigma)"
                         % (drift, drift_err, z))
    else:
        notes.append("%d block(s) left after the discard of %d: the drift was not tested" % (kept, drop))
    return Equilibration(drop=int(drop), drop_physical=physical, drop_mser=int(data), n_blocks=int(n), n_kept=int(kept),
                         drift=drift, drift_error=drift_err, drift_z=z, stationary=stationary, message="; ".join(notes))


# ----------------------------------------------------------------------------- independent series
@dataclass
class Combined:
    """Result of :func:`combine`."""

    mean: float                 # inverse-variance weighted mean
    error: float                # 1 / sqrt(sum of 1 / error_i^2)
    chi2: float                 # sum of ((mean_i - mean) / error_i)^2
    dof: int                    # number of series - 1
    p_value: Optional[float]    # probability of a chi2 at least this large when the series agree; None for one series
    error_scaled: float         # error * sqrt(chi2 / dof) when chi2 / dof > 1, else error
    n: int                      # number of series

    def as_dict(self) -> Dict[str, object]:
        return plain(asdict(self))


def combine(means: Sequence[float], errors: Sequence[float]) -> Combined:
    """Combine independent estimates of one quantity by inverse-variance weights.

    ``chi2`` with ``dof`` degrees of freedom measures whether the series agree. A small
    ``p_value`` (for example below 0.01) says that they differ by more than their error bars
    allow: then the error bars are too small, or the series do not measure the same quantity.
    ``error_scaled`` is the error multiplied by sqrt(chi2 / dof) when that factor is above 1.

    The series must be statistically independent. For QMCPACK runs check the random seeds first
    (:func:`seed_overlap`). Raises ValueError for an empty input or an error bar that is not a
    positive finite number.
    """
    m = np.asarray(means, dtype=float).reshape(-1)
    s = np.asarray(errors, dtype=float).reshape(-1)
    if len(m) == 0 or len(m) != len(s):
        raise ValueError("combine: %d means and %d errors" % (len(m), len(s)))
    bad = [i for i in range(len(m)) if not (np.isfinite(m[i]) and np.isfinite(s[i]) and s[i] > 0)]
    if bad:
        raise ValueError("combine: series %s have a mean or an error bar that is not a positive finite number"
                         % ", ".join(str(i) for i in bad))
    w = 1.0 / s ** 2
    mean = float(np.sum(w * m) / w.sum())
    error = float(1.0 / math.sqrt(w.sum()))
    chi2 = float(np.sum(((m - mean) / s) ** 2))
    dof = len(m) - 1
    p_value = None
    scale = 1.0
    if dof > 0:
        from scipy.special import gammaincc           # survival function of the chi-square distribution
        p_value = float(gammaincc(0.5 * dof, 0.5 * chi2))
        scale = max(1.0, math.sqrt(chi2 / dof))
    return Combined(mean=mean, error=error, chi2=chi2, dof=dof, p_value=p_value, error_scaled=error * scale, n=len(m))


# ----------------------------------------------------------------------------- random seeds of a QMCPACK run
SeedRange = Tuple[int, int]
_SEED_LINE = "Range of prime numbers"
_SEED_RANGE_RE = re.compile(
    r"Range of prime numbers to use as seeds over processors and threads\s*=\s*(\d+)\s*-\s*(\d+)")

OVERLAP = "overlap"
DISJOINT = "disjoint"
UNKNOWN = "unknown"


def seed_ranges(text: str) -> List[SeedRange]:
    """Seed ranges printed in the text of a QMCPACK output file.

    QMCPACK prints one line per run::

        Range of prime numbers to use as seeds over processors and threads = 2467-2719

    Returns the list of ``(lo, hi)`` pairs, without repeats, in the order they appear. An output
    that was appended by several runs holds several lines; all of them are returned. The list is
    EMPTY when the text has no such line, or when one such line cannot be read. An empty list
    means "the seeds are not known", never "no seeds".
    """
    out: List[SeedRange] = []
    for line in text.splitlines():
        if _SEED_LINE not in line:
            continue
        m = _SEED_RANGE_RE.search(line)
        if m is None or int(m.group(1)) > int(m.group(2)):
            return []
        r = (int(m.group(1)), int(m.group(2)))
        if r not in out:
            out.append(r)
    return out


def read_seed_ranges(path: str) -> List[SeedRange]:
    """:func:`seed_ranges` of a file. A file that is missing or cannot be read gives an empty
    list, which :func:`seed_overlap` answers with "unknown"."""
    try:
        with open(path, errors="replace") as fh:
            return seed_ranges(fh.read())
    except OSError:
        return []


def _primes_up_to(n: int) -> np.ndarray:
    sieve = np.ones(max(n, 1) + 1, dtype=bool)
    sieve[:2] = False
    for i in range(2, int(math.isqrt(max(n, 1))) + 1):
        if sieve[i]:
            sieve[i * i::i] = False
    return np.flatnonzero(sieve)


def _widen(r: SeedRange, margin: float) -> SeedRange:
    """The range with its upper end moved up by ceil(margin * c) primes, c = number of primes in the range."""
    lo, hi = r
    if margin <= 0.0:
        return lo, hi
    if hi > 20000000:                                   # far above any range QMCPACK prints: widen by value
        return lo, hi + int(math.ceil(2.0 * margin * (hi - lo + 100)))
    bound = hi + int(4 * margin * (hi - lo + 100)) + 100
    while True:
        primes = _primes_up_to(bound)
        inside = int(np.count_nonzero((primes >= lo) & (primes <= hi)))
        need = int(math.ceil(margin * max(inside, 1)))
        above = primes[primes > hi]
        if len(above) >= need:
            return lo, int(above[need - 1])
        bound *= 2


def as_seed_ranges(x) -> List[SeedRange]:
    """Normalize one range, a list of ranges or None to a list of ranges; an entry that is not a
    valid range makes the whole list empty (= unknown)."""
    if x is None:
        return []
    try:
        if len(x) == 2 and all(isinstance(v, (int, np.integer)) and not isinstance(v, bool) for v in x):
            x = [x]
        out = []
        for r in x:
            lo, hi = int(r[0]), int(r[1])
            if len(r) != 2 or lo > hi or lo < 0:
                return []
            out.append((lo, hi))
        return out
    except (TypeError, ValueError, IndexError):
        return []


def seed_overlap(a: Union[None, SeedRange, Sequence[SeedRange]], b: Union[None, SeedRange, Sequence[SeedRange]],
                 margin: float = 1.0) -> str:
    """Do two QMCPACK runs share random seeds? Returns "overlap", "disjoint" or "unknown".

    ``a`` and ``b`` are the seed ranges of the two runs: one ``(lo, hi)`` pair, a list of pairs
    (as returned by :func:`seed_ranges`), or None / an empty list when the range could not be read.

    * "unknown"  at least one side has no readable range. The test REFUSES to answer: two runs
                 are never called independent without proof.
    * "overlap"  the ranges share a prime, or they are closer than the safety margin.
    * "disjoint" every range of ``a`` lies apart from every range of ``b`` by more than the margin.

    Why a margin. Every generator of a run takes one prime of the list of primes as its seed, and
    two generators with the same prime produce the same random numbers. The printed range holds
    ranks * threads + 1 primes (checked on outputs of runs with 32 and with 64 ranks). Whether a
    run also takes primes just above the printed range (one more per rank for the generators of
    the threads) was not checked against the QMCPACK source. ``margin = 1`` therefore treats
    every run as using its printed range plus as many primes above it as the range contains.
    ``margin = 0`` compares the printed ranges only.

    Only "disjoint" allows two series to be combined. Use :func:`seeds_independent` for a yes/no answer.
    """
    ra, rb = as_seed_ranges(a), as_seed_ranges(b)
    if not ra or not rb:
        return UNKNOWN
    wa = [_widen(r, margin) for r in ra]
    wb = [_widen(r, margin) for r in rb]
    for xl, xh in wa:
        for yl, yh in wb:
            if not (xh < yl or yh < xl):
                return OVERLAP
    return DISJOINT


def seeds_independent(a, b, margin: float = 1.0) -> bool:
    """True only when :func:`seed_overlap` proves that the two runs used different seeds."""
    return seed_overlap(a, b, margin) == DISJOINT
