"""Engine-clustered bootstrap 95% bands for the whole-trajectory signed-error curves (Fig. 2).

Usage:
    python scripts/trajectory_bands.py      # reads results/cv_SUBSET.json, writes results/curve_ci.json
"""

import json, numpy as np

rng = np.random.default_rng(7)
out = {}
for s in ("FD001", "FD002", "FD003", "FD004"):
    R = json.load(open(f"results/cv_{s}.json"))
    tu = np.array(R["te_unit"])
    tc = np.array(R["te_cycle"])
    rr = np.array(R["rul_raw"])
    lastc = {u: tc[tu == u].max() for u in np.unique(tu)}
    true = np.minimum(np.array([rr[u - 1] + lastc[u] - c for u, c in zip(tu, tc)], float), 125)
    units = np.unique(tu)
    bins = np.arange(0, 130, 10)
    out[s] = {}
    for m in ("LR", "Ridge", "RF", "GB"):
        e = np.array(R["test_all"][m]).mean(0) - true
        # per engine, per bin: sum and count
        S = np.zeros((len(units), len(bins)))
        N = np.zeros_like(S)
        b = np.minimum(np.digitize(true, bins) - 1, len(bins) - 1)
        ui = np.searchsorted(units, tu)
        np.add.at(S, (ui, b), e)
        np.add.at(N, (ui, b), 1)
        mean = S.sum(0) / np.maximum(N.sum(0), 1)
        bs = []
        for _ in range(1000):
            j = rng.integers(0, len(units), len(units))
            bs.append(S[j].sum(0) / np.maximum(N[j].sum(0), 1))
        lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
        out[s][m] = [
            [
                float(b0 + 5),
                float(mean[i]),
                float(lo[i]),
                float(hi[i]),
                int(N[:, i].sum()),
                int((N[:, i] > 0).sum()),
            ]
            for i, b0 in enumerate(bins)
        ]
json.dump(out, open("results/curve_ci.json", "w"))
for s in out:
    print(
        s, "GB crit", [(r[0], round(r[1], 1), round(r[2], 1), round(r[3], 1), r[5]) for r in out[s]["GB"][:3]]
    )
