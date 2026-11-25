#!/usr/bin/env python3
"""Compare the search methods on analytic model surfaces (no cluster, no QMC).

The "true" surface stands for the DMC energy; the mock engine adds statistical noise
to it. The surrogate stands for DFT. Every run starts at the minimum (or saddle) of
the surrogate, as a real calculation starts at the DFT geometry.

    python3 compare_protocols.py near    16        # start = surrogate minimum, 16 random seeds
    python3 compare_protocols.py shift   8 water 2 # surrogate = true surface moved along direction 2
    python3 compare_protocols.py saddle  8         # ammonia inversion saddle
    python3 compare_protocols.py shape   8         # surrogate with a wrong anharmonic shape
    python3 compare_protocols.py tilt    8 water 34 1.0   # surrogate Hessian turned by 34 degrees, start 1 kcal/mol up
    python3 compare_protocols.py tilt    8 water 34 1.0 hessian   # the same with only the harmonic part turned
    python3 compare_protocols.py random  16 water 0.05 0.10 0.20   # every atom moved at random, rms per atom in angstrom

Method names after the arguments choose what is run (default: v1 v2):

    v1            dense scan with the rules of the first application
    v2            improved dense scan with its default tolerance of 0.2 kcal/mol
    v2-tight      the same with a tolerance of 0.05 kcal/mol (for a comparison at the accuracy of the classic search)
    v2-nocv, v2-fit, v2-distrust     variants of v2 (see docs/protocols.md)
    classic       the classic line search with its default options: 50 blocks discarded and at least 30 usable
                  blocks per point. With the block noise of this study these two lower limits, not the noise rule
                  of the search, set the number of blocks of most points.
    classic-lean  the classic line search with 10 blocks discarded (as v2 discards at least) and at least 8 usable
                  blocks per point, so that its own noise rule sets the number of blocks
    auto          classic first, the dense-scan search if the audit of the first iteration fails

Results are printed and appended to ``results.jsonl`` in this folder. The cost is counted in DMC
blocks. "all" is every block that the method ran. "search" leaves out the full-length runs at the
starting geometry and at the final geometry, which every method needs for its reported energies. For the
dense scan with the rules of the first application these are the two runs at the first and at the last centre.

The error of a run is the true energy of its final geometry above the true stationary point. For a saddle
that number can be negative along the reaction coordinate and would cancel an error in the other
directions. So for a saddle the error is the harmonic energy sum_j |k_j| z_j^2 / 2 of the final displacement
z in the normal directions of the true saddle.
"""
import json
import os
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))

from montepass import geometry as G, model as M, subspace as S  # noqa: E402
from montepass.engine.mock import MockEngine  # noqa: E402
from montepass.search import protocol_v1 as P1, protocol_v2 as P2  # noqa: E402
from montepass.search.auto import AutoSearch  # noqa: E402
from montepass.search.classic import ClassicLineSearch  # noqa: E402
from montepass.search.dense import DenseScanSearch  # noqa: E402
from montepass.surrogate.analytic import AnalyticSurrogate  # noqa: E402

PROTOCOLS = {
    "v1": P1.V1,
    "v2": P2.V2,
    "v2-nocv": P2.ProtocolV2(control_variate=False),
    "v2-tight": P2.ProtocolV2(eps_total=0.05),
    "v2-fit": P2.ProtocolV2(coupling_fit=True),
    "v2-distrust": P2.ProtocolV2(shape_distrust=True),
}
METHODS = dict(PROTOCOLS, classic="classic", auto="auto")
METHODS["classic-lean"] = "classic-lean"
CLASSIC_LEAN = {"drop": 10, "min_blocks": 8}
SIGMA_BLOCK = 5.0          # kcal/mol per block: the noise of one block of the model DMC
ENDED = ("CONVERGED", "NOT_CONVERGED", "DONE")


def directions_at(surface, x):
    Hs = M.hessian(surface, x)
    sub = S.mode_subspace(Hs, x)
    return S.search_directions(S.project_hessian(Hs, sub.basis), sub)


def molecule(name):
    if name == "water":
        return M.water(True), M.water(False), M.water_geometry()
    if name == "ammonia":
        return M.ammonia(True), M.ammonia(False), M.ammonia_geometry()
    raise ValueError(name)


def case_near(name):
    ft, fs, x0 = molecule(name)
    xt, xs = M.minimize(ft, x0), M.minimize(fs, x0)
    return ft, fs, xs, xt, directions_at(fs, xs)


def case_shift(name, shift, direction):
    """Surrogate = the true surface moved rigidly by ``shift`` angstrom along one normal direction of
    the true minimum: right shape, wrong place."""
    ft, _fs, x0 = molecule(name)
    xt = M.minimize(ft, x0)
    d0 = directions_at(ft, xt)
    if direction == "random":
        v = np.random.default_rng(12345).normal(size=d0.dim); v /= np.linalg.norm(v)
        vec = v @ d0.vectors
    else:
        vec = d0.vectors[int(direction)]
    delta = (shift * vec).reshape(-1, 3)
    fs = ft.shifted(delta)
    xs = xt + delta
    return ft, fs, xs, xt, directions_at(fs, xs)


def case_saddle():
    ft, fs = M.ammonia(True), M.ammonia(False)
    xs = M.refine_stationary(fs, M.ammonia_geometry(theta=120.0, r=1.0))
    xt = M.refine_stationary(ft, M.ammonia_geometry(theta=120.0, r=1.0))
    return ft, fs, xs, xt, directions_at(fs, xs)


def case_shape(name):
    """Surrogate with a clearly different anharmonicity: Morse range parameter 25 % smaller."""
    ft, fs, x0 = molecule(name)
    terms = tuple(M.Morse(t.i, t.j, t.de * 1.5625, t.a * 0.8, t.re) if isinstance(t, M.Morse) else t for t in fs.terms)
    fs = M.ForceField(symbols=fs.symbols, terms=terms)      # same force constant 2 de a^2, different cubic term
    xt, xs = M.minimize(ft, x0), M.minimize(fs, x0)
    return ft, fs, xs, xt, directions_at(fs, xs)


def case_tilt(name, angle=34.0, energy=1.0, kind="surface"):
    """Surrogate with the right curvatures along the wrong directions, and a minimum at the wrong place.

    The two softest normal directions of the true minimum that have different curvatures are turned into
    each other by ``angle`` degrees. The surrogate minimum is moved along their sum, so far that the true
    energy there is ``energy`` kcal/mol above the true minimum (harmonic estimate). The search directions
    (the normal directions of the surrogate) are then not the normal directions of the true surface.

    kind = "surface": the whole surrogate surface is turned, so its anharmonic part is wrong as well.
    kind = "hessian": only the harmonic part is turned; the anharmonic part is that of the true surface."""
    ft, _fs, x0 = molecule(name)
    xt = M.minimize(ft, x0)
    d0 = directions_at(ft, xt)
    i, j = next((a, b) for a in range(d0.dim) for b in range(a + 1, d0.dim)
                if abs(d0.curvature[b] - d0.curvature[a]) > 0.05 * abs(d0.curvature[b]))
    u, v = d0.vectors[i], d0.vectors[j]
    w = (u + v) / np.sqrt(2.0)
    k_w = 0.5 * (d0.curvature[i] + d0.curvature[j])
    delta = (np.sqrt(2.0 * energy / k_w) * w).reshape(-1, 3)
    xs = xt + delta
    if kind == "hessian":
        fs = M.TurnedHessianSurface(ft, xt, xs, M.plane_rotation(u, v, angle))
    elif kind == "surface":
        fs = M.RotatedSurface(ft, xt, xs, M.plane_rotation(u, v, angle))
    else:
        raise ValueError("case_tilt: kind must be 'surface' or 'hessian', got %r" % kind)
    return ft, fs, xs, xt, directions_at(fs, xs)


def case_random(name, rms, seed):
    """The usual surrogate (slightly wrong force constants and geometry) moved rigidly by a random displacement
    of all atoms with a root mean square displacement of ``rms`` angstrom per atom: a DFT geometry that is
    wrong in a random direction. The search starts at the minimum of that moved surrogate."""
    ft, fs0, x0 = molecule(name)
    xt, xs0 = M.minimize(ft, x0), M.minimize(fs0, x0)
    delta = G.random_displacement(xs0, rms, seed)
    fs = fs0.shifted(delta)
    xs = xs0 + delta
    return ft, fs, xs, xt, directions_at(fs, xs)


_SCORE_CACHE = {}


def error_energy(ft, xt, xf):
    """Error of a final geometry in kcal/mol: the true energy above the stationary point ``xt``; for a saddle
    the harmonic energy sum_j |k_j| z_j^2 / 2 of the displacement in the normal directions of the true surface."""
    key = (id(ft), xt.tobytes())
    if key not in _SCORE_CACHE:
        H = M.hessian(ft, xt)
        w, V = np.linalg.eigh(0.5 * (H + H.T))
        internal = np.abs(w) > 1e-3 * np.abs(w).max()          # rigid motions have force constants of zero
        _SCORE_CACHE[key] = (w[internal], V[:, internal], bool(np.any(w[internal] < 0)))
    w, V, saddle = _SCORE_CACHE[key]
    if not saddle:
        return float(ft(xf) - ft(xt))
    rot, c_f, c_t, _rmsd = G.superposition(xf, xt)
    z = (((xf - c_f) @ rot + c_t) - xt).reshape(-1) @ V
    return float(0.5 * np.sum(np.abs(w) * z * z))


def run(case, protocol, seed, sigma=SIGMA_BLOCK):
    ft, fs, xs, xt, d = case
    with tempfile.TemporaryDirectory() as tmp:
        # start-up transient as measured on real traces: +20 kcal/mol in block 0, gone after block 2
        eng = MockEngine(ft, fs, sigma_block=sigma, blocks_per_tick=10, seed=seed, transient=20.0, equil_blocks=0.6)
        args = ("t", eng, AnalyticSurrogate(fs), list(ft.symbols), xs, d.vectors, d.curvature)
        if protocol == "classic":
            s = ClassicLineSearch(*args, settings={"sigma_block": sigma}, state_dir=tmp)
        elif protocol == "classic-lean":
            s = ClassicLineSearch(*args, settings={"sigma_block": sigma, "classic": dict(CLASSIC_LEAN)}, state_dir=tmp)
        elif protocol == "auto":
            s = AutoSearch(*args, protocol=P2.V2, state_dir=tmp, settings={"sigma_block": sigma})
        else:
            s = DenseScanSearch(*args, protocol=protocol, state_dir=tmp)
        status = s.run()
        xf = s.final_geometry()
        total = int(sum(eng.read(key).nblocks for key in list(eng.points)))       # every block that was run
        state = s.state
        # the same rule for every method: all blocks, without the runs at the starting geometry and at the final
        # geometry. Protocol v1 has no separate final runs: the runs at its last centre are its final energies.
        reference = set(state["iters"]["1"]["centre_keys"]) | set((state.get("final") or {}).get("keys", []))
        if state.get("protocol") == "v1":
            reference |= set(state["iters"][str(state["iteration"])]["centre_keys"])
        search = total - int(sum(eng.read(key).nblocks for key in reference))
        tolerance = float(protocol.eps_total) if hasattr(protocol, "eps_total") else 0.2
        out = dict(status=status, iterations=int(state["iteration"]), blocks=total, search_blocks=int(search),
                   rejected=int(state.get("rejected", 0)), e_final=error_energy(ft, xt, xf), e_start=error_energy(ft, xt, xs),
                   tolerance=tolerance,
                   d_start=float(G.kabsch(xs, xt)[1]), d_final=float(G.kabsch(xf, xt)[1]))
        if protocol == "auto":
            out["switched"] = bool(s.phase == "dense")
        if protocol in ("classic", "classic-lean"):
            out["unresolved_first"] = bool(s.untrusted(1)) if hasattr(s, "untrusted") else bool(s.active.untrusted(1))
            out["not_bracketed"] = int(sum(1 for b in (s.brackets() if hasattr(s, "brackets") else s.active.brackets())
                                           if b["status"] != "INSIDE"))
        return out


def summarize(label, runs, out):
    n = len(runs)
    ef = np.array([r["e_final"] for r in runs]); bl = np.array([r["blocks"] for r in runs])
    sb = np.array([r["search_blocks"] for r in runs])
    row = dict(label=label, runs=n, converged=int(sum(r["status"] == "CONVERGED" for r in runs)),
               ended=int(sum(r["status"] in ENDED for r in runs)),
               statuses={st: int(sum(r["status"] == st for r in runs)) for st in sorted({r["status"] for r in runs})},
               # a run that ends far below the minimum has left the molecule (the model surface is not bounded there)
               within_tolerance=int(sum(abs(r["e_final"]) < 0.2 for r in runs)),
               tolerance=float(runs[0].get("tolerance", 0.2)),
               within_own_tolerance=int(sum(abs(r["e_final"]) < r.get("tolerance", 0.2) for r in runs)),
               e_final_p95=float(np.percentile(np.abs(ef), 95)),
               left_molecule=int(sum(r["e_final"] < -0.5 or r["d_final"] > 1.0 for r in runs)),
               switched=int(sum(bool(r.get("switched")) for r in runs)),
               unresolved_first=int(sum(bool(r.get("unresolved_first")) for r in runs)),
               iterations=float(np.mean([r["iterations"] for r in runs])),
               blocks_mean=float(bl.mean()), blocks_sem=float(bl.std(ddof=1) / np.sqrt(n)),
               search_blocks_mean=float(sb.mean()), search_blocks_sem=float(sb.std(ddof=1) / np.sqrt(n)),
               e_start=float(np.mean([r["e_start"] for r in runs])), e_final_mean=float(ef.mean()),
               e_final_sem=float(ef.std(ddof=1) / np.sqrt(n)), e_final_max=float(ef.max()),
               # how much of the starting error each run removed: 1 = all of it, 0 = nothing, below 0 = made it worse
               energy_recovered_median=float(np.median([1.0 - abs(r["e_final"]) / r["e_start"] for r in runs])),
               energy_recovered_min=float(np.min([1.0 - abs(r["e_final"]) / r["e_start"] for r in runs])),
               recovered_90_percent=int(sum(abs(r["e_final"]) < 0.1 * r["e_start"] for r in runs)),
               distance_start_mean=float(np.mean([r["d_start"] for r in runs])),
               distance_final_mean=float(np.mean([r["d_final"] for r in runs])),
               distance_final_max=float(np.max([r["d_final"] for r in runs])))
    own = "" if row["tolerance"] == 0.2 else "  E_final < %.2f: %d/%d" % (row["tolerance"], row["within_own_tolerance"], n)
    print("%-38s %s  E_final < 0.2: %2d/%-2d%s  iterations %4.1f  blocks: search %6.0f +/- %4.0f, all %6.0f  E_start %7.3f  E_final mean %6.3f +/- %.3f, 95th percentile %.3f, max %.3f%s" % (
        label, " ".join("%s %d" % kv for kv in row["statuses"].items()), row["within_tolerance"], n, own, row["iterations"],
        row["search_blocks_mean"], row["search_blocks_sem"], row["blocks_mean"], row["e_start"], row["e_final_mean"],
        row["e_final_sem"], row["e_final_p95"], row["e_final_max"],
        ("  switched to dense %d/%d" % (row["switched"], n)) if label.endswith(" auto") else
        (("  first audit failed %d/%d" % (row["unresolved_first"], n)) if " classic" in label else "")), flush=True)
    if label.startswith("random"):
        print("%-34s    start: energy %.1f kcal/mol above the minimum, RMSD %.3f A; recovered energy: median %.1f %%, worst run %.1f %%; runs that removed 90 %% of the error: %d/%d; runs that left the molecule: %d; RMSD at the end: mean %.4f A, largest %.4f A" % (
            "", row["e_start"], row["distance_start_mean"], 100.0 * row["energy_recovered_median"], 100.0 * row["energy_recovered_min"],
            row["recovered_90_percent"], n, row["left_molecule"], row["distance_final_mean"], row["distance_final_max"]), flush=True)
    out.write(json.dumps(row) + "\n"); out.flush()


def main():
    exp, n = sys.argv[1], int(sys.argv[2])
    names = [a for a in sys.argv[3:] if a in METHODS] or ["v1", "v2"]
    rest = [a for a in sys.argv[3:] if a not in METHODS]
    t0 = time.time()
    with open(os.path.join(HERE, "results.jsonl"), "a") as out:
        if exp == "near":
            for mol in (rest or ["water", "ammonia"]):
                case = case_near(mol)
                for p in names:
                    summarize("near %s %s" % (mol, p), [run(case, METHODS[p], s) for s in range(n)], out)
        elif exp == "shift":
            mol, direction = rest[0], rest[1]
            for shift in [float(v) for v in rest[2:]] or [0.05, 0.10, 0.20, 0.30]:
                case = case_shift(mol, shift, direction)        # a negative shift moves the start the other way
                for p in names:
                    summarize("shift %s dir %s %+.2f A %s" % (mol, direction, shift, p),
                              [run(case, METHODS[p], s) for s in range(n)], out)
        elif exp == "saddle":
            case = case_saddle()
            for p in names:
                summarize("saddle ammonia %s" % p, [run(case, METHODS[p], s) for s in range(n)], out)
        elif exp == "shape":
            for mol in (rest or ["water", "ammonia"]):
                case = case_shape(mol)
                for p in names:
                    summarize("shape %s %s" % (mol, p), [run(case, METHODS[p], s) for s in range(n)], out)
        elif exp == "random":
            # every run has its own random displacement (seed) and its own noise
            mol = rest[0] if rest else "water"
            for rms in [float(v) for v in rest[1:]] or [0.05, 0.10, 0.20]:
                cases = [case_random(mol, rms, s) for s in range(n)]
                for p in names:
                    summarize("random %s rms %.2f A %s" % (mol, rms, p), [run(cases[s], METHODS[p], s) for s in range(n)], out)
        elif exp == "tilt":
            mol = rest[0] if rest else "water"
            angle = float(rest[1]) if len(rest) > 1 else 34.0
            energy = float(rest[2]) if len(rest) > 2 else 1.0
            kind = rest[3] if len(rest) > 3 else "surface"
            case = case_tilt(mol, angle, energy, kind)
            for p in names:
                summarize("tilt %s %g deg %g kcal %s %s" % (mol, angle, energy, kind, p),
                          [run(case, METHODS[p], s) for s in range(n)], out)
        else:
            raise SystemExit(__doc__)
    print("time %.0f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
