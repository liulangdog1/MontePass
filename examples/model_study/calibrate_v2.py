#!/usr/bin/env python3
"""How often is the statement CONVERGED of protocol v2 true? (model surfaces, no cluster, no QMC)

    python3 calibrate_v2.py 96            # 96 runs per case, default tolerance (0.2 kcal/mol)
    python3 calibrate_v2.py 96 0.05       # the same with a tolerance of 0.05 kcal/mol

For every case and the given number of random seeds, the script counts the runs that end as
CONVERGED and, among them, the runs whose final geometry is less than the tolerance away (in
energy) from the true stationary point. The cases are the ones of ``compare_protocols.py``:

* a start at the stationary point of the surrogate (water, ammonia, the ammonia inversion saddle);
* a surrogate that is the true surface moved rigidly along one direction;
* a surrogate whose Hessian is turned by 34 degrees (the whole surface turned, or only its harmonic part);
* every atom moved at random (a new displacement for every run).

docs/protocols.md quotes the result for 96 seeds.
"""
import sys

import numpy as np

import compare_protocols as CP
from montepass.search import protocol_v2 as P2


def main():
    nseed = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    prot = P2.V2 if len(sys.argv) < 3 else P2.ProtocolV2(eps_total=float(sys.argv[2]))
    same = lambda case: (lambda seed: case)             # noqa: E731  one case for all seeds
    cases = [("water, start at the surrogate minimum", same(CP.case_near("water"))),
             ("ammonia, start at the surrogate minimum", same(CP.case_near("ammonia"))),
             ("ammonia inversion saddle", same(CP.case_saddle()))]
    for shift in (0.05, 0.10, 0.20, 0.30):
        cases.append(("water, %.2f A along the antisymmetric stretch" % shift, same(CP.case_shift("water", shift, "2"))))
    for shift in (0.10, -0.10, 0.20, -0.20):
        cases.append(("water, %+.2f A along the symmetric stretch" % shift, same(CP.case_shift("water", shift, "1"))))
    cases.append(("water, 0.30 A along the bend", same(CP.case_shift("water", 0.30, "0"))))
    for mol in ("water", "ammonia"):
        cases.append(("%s, surrogate surface turned by 34 deg" % mol, same(CP.case_tilt(mol, 34.0, 1.0, "surface"))))
        cases.append(("%s, only the surrogate Hessian turned by 34 deg" % mol, same(CP.case_tilt(mol, 34.0, 1.0, "hessian"))))
    for mol in ("water", "ammonia"):
        for rms in (0.05, 0.10, 0.20):
            cases.append(("%s, every atom moved at random, rms %.2f A" % (mol, rms),
                          (lambda seed, mol=mol, rms=rms: CP.case_random(mol, rms, seed))))
    total = [0, 0, 0]
    print("tolerance %.2f kcal/mol, %d runs per case" % (prot.eps_total, nseed), flush=True)
    for label, make in cases:
        runs = [CP.run(make(seed), prot, seed) for seed in range(nseed)]
        conv = [r for r in runs if r["status"] == "CONVERGED"]
        ok = sum(abs(r["e_final"]) < prot.eps_total for r in conv)
        ef = np.abs(np.array([r["e_final"] for r in runs]))
        sb = np.array([r["search_blocks"] for r in runs])
        print("%-52s CONVERGED %3d of %d; of these within %.2f kcal/mol: %3d (%.0f %%); error of the final geometry: "
              "mean %.3f, 95th percentile %.3f, largest %.3f kcal/mol; start %.1f; iterations %.1f; search blocks %.0f"
              % (label, len(conv), nseed, prot.eps_total, ok, 100.0 * ok / max(1, len(conv)), ef.mean(),
                 np.percentile(ef, 95), ef.max(), np.mean([r["e_start"] for r in runs]),
                 np.mean([r["iterations"] for r in runs]), sb.mean()), flush=True)
        total[0] += len(conv); total[1] += ok; total[2] += nseed
    print("all cases: CONVERGED %d of %d runs; within the tolerance %d of the converged runs (%.1f %%)"
          % (total[0], total[2], total[1], 100.0 * total[1] / max(1, total[0])))


if __name__ == "__main__":
    main()
