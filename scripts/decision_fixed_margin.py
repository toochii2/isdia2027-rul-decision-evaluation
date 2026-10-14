"""Cross-fitted decision lens with margins held fixed in the bootstrap (non-nested sensitivity design).

Margins are calibrated on four folds of out-of-fold predictions and evaluated on the held-out fold.
Used only for the comparison in Sect. 4.4 (fixed versus re-estimated margins).

Usage:
    python scripts/decision_fixed_margin.py RESULTS_DIR FD001 FD002 FD003 FD004
Reads RESULTS_DIR/cv_SUBSET.json and writes RESULTS_DIR/decision_fixed_margin.json.
"""

import json, sys, os, numpy as np
from sklearn.model_selection import GroupKFold
from scipy.stats import beta

rng = np.random.default_rng(11)
M = ["LR", "Ridge", "RF", "GB"]
MARG = np.arange(0, 121)


def cp_upper(k, n, a=0.05):
    return 1.0 if k >= n else float(beta.ppf(1 - a, k + 1, n - k))


def run(path, leads=(10, 15, 30), targets=(0.0, 0.02), B=2000):
    R = json.load(open(path))
    unit = np.array(R["tr_unit"])
    rul = np.array(R["tr_rul"], float)
    y = np.array(R["tr_y"], float)
    M = [m for m in ("LR", "Ridge", "RF", "GB", "CNN") if m in R["oof"]]
    OF = {m: np.array(R["oof"][m]) for m in M}
    units = np.unique(unit)
    gi = {u: np.where(unit == u)[0] for u in units}
    folds = (
        [np.array(f) for f in R["folds"]]
        if R.get("folds")
        else [np.unique(unit[b]) for _, b in GroupKFold(n_splits=5).split(np.zeros(len(unit)), groups=unit)]
    )
    uidx = {u: i for i, u in enumerate(units)}
    out = {"oof_rmse": {m: float(np.sqrt(np.mean((OF[m] - y) ** 2))) for m in M}, "n": int(len(units))}
    for L in leads:
        # trigger RUL matrix: engines x margins (-1 = no alarm)
        RT = {}
        for m in M:
            mat = np.full((len(units), len(MARG)), -1.0)
            for i, u in enumerate(units):
                p = OF[m][gi[u]]
                r = rul[gi[u]]
                for j, mg in enumerate(MARG):
                    t = np.where(p <= L + mg)[0]
                    if len(t):
                        mat[i, j] = r[t[0]]
            RT[m] = mat
        res = {}
        for a in targets:
            res[a] = {}
            per_engine = {}
            for m in M:
                miss_e = np.zeros(len(units), bool)
                unused_e = np.full(len(units), np.nan)
                mh = []
                for f in folds:
                    test = np.array([uidx[u] for u in f])
                    cal = np.setdiff1d(np.arange(len(units)), test)
                    allowed = int(np.floor(a * len(cal)))
                    misses = (RT[m][cal] < L).sum(0)
                    ok = np.where(misses <= allowed)[0]
                    j = ok[0] if len(ok) else len(MARG) - 1
                    mh.append(int(MARG[j]))
                    rt = RT[m][test, j]
                    miss_e[test] = rt < L
                    unused_e[test] = np.where(rt >= L, rt - L, np.nan)
                k = int(miss_e.sum())
                n = len(units)
                res[a][m] = dict(
                    margins=mh,
                    misses=k,
                    miss_rate=k / n,
                    cp_upper=cp_upper(k, n),
                    unused=float(np.nanmean(unused_e)),
                )
                per_engine[m] = (miss_e, unused_e)

            # paired bootstrap of unused-life differences
            def pdiff(m1, m2):
                u1, u2 = per_engine[m1][1], per_engine[m2][1]
                both = ~np.isnan(u1) & ~np.isnan(u2)
                d = (u1 - u2)[both]
                bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(B)]
                return dict(
                    mean=float(d.mean()),
                    ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                    n=int(both.sum()),
                )

            res[a]["pairs"] = {
                "RF-GB": pdiff("RF", "GB"),
                "RF-LR": pdiff("RF", "LR"),
                "GB-LR": pdiff("GB", "LR"),
            }
            if "CNN" in M:
                res[a]["pairs"].update(
                    {"CNN-GB": pdiff("CNN", "GB"), "CNN-LR": pdiff("CNN", "LR"), "CNN-RF": pdiff("CNN", "RF")}
                )
        out[L] = res
    # time-based baseline cross-fitted: fixed age set on calibration folds (zero misses)
    life = np.array([rul[gi[u]].max() for u in units])
    tb = {}
    for L in leads:
        miss = 0
        un = []
        for f in folds:
            test = np.array([uidx[u] for u in f])
            cal = np.setdiff1d(np.arange(len(units)), test)
            T = life[cal].min() - L
            lt = life[test]
            miss += int((lt - T < L).sum())
            un += list((lt - T - L)[lt - T >= L])
        tb[L] = dict(misses=miss, unused=float(np.mean(un)))
    out["time"] = tb
    return out


if __name__ == "__main__":
    D = sys.argv[1]
    subs = sys.argv[2:]
    allres = {}
    for s in subs:
        r = run(f"{D}/cv_{s}.json" if not s.endswith(".json") else f"{D}/{s}")
        allres[s] = r
        print("=====", s, "OOF RMSE", {m: round(v, 2) for m, v in r["oof_rmse"].items()}, "n", r["n"])
        for L in (10, 15, 30):
            for a in (0.0, 0.02):
                x = r[L][a]
                print(
                    f" L={L} a={a}",
                    {
                        m: (x[m]["misses"], round(x[m]["cp_upper"] * 100, 1), round(x[m]["unused"], 1))
                        for m in x
                        if m != "pairs"
                    },
                    {
                        k: (round(v["mean"], 1), [round(c, 1) for c in v["ci95"]])
                        for k, v in x["pairs"].items()
                    },
                    "TIME",
                    r["time"][L],
                )
    json.dump(
        {s: {str(k): v for k, v in r.items()} for s, r in allres.items()},
        open(os.environ.get("D2OUT", f"{D}/decision_fixed_margin.json"), "w"),
        default=lambda o: str(o),
    )
