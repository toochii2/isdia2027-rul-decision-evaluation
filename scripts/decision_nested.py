"""Decision lens L5 with nested (or non-nested) margin calibration (Sect. 3.4).

Alarm at the first cycle with predicted RUL <= L + m; a miss is no timely alarm (true RUL < L).
Margins are set per outer fold as the smallest meeting the calibration target (zero misses or
at most floor(0.02 n) misses) on inner predictions; held-out misses are reported with one-sided
95% Clopper-Pearson bounds. Paired differences in unused life use a fold-stratified engine
bootstrap (B = 2000) that re-estimates every fold's margin in each replicate (models are not
refitted); p = min{1, 2 min[P(D* <= 0), P(D* >= 0)]}.

Usage:
    python scripts/decision_nested.py MODE OUT_JSON SUBSET [SUBSET ...]
    MODE = nested            reads results/nested_SUBSET.json (+ results/nestedcnn_SUBSET.json)
    MODE = nonnested:PREFIX  reads results/PREFIXSUBSET.json, e.g. nonnested:cv_P1_
"""

import sys, json, os, numpy as np
from scipy.stats import beta
from sklearn.model_selection import GroupKFold

LEADS = (10, 15, 30)
TARGETS = (0.0, 0.02)
MMAX = 200
B = 2000
COSTS = (25, 100, 400)
rng = np.random.default_rng(2027)


def cp_upper(k, n, a=0.05):
    return 1.0 if k >= n else float(beta.ppf(1 - a, k + 1, n - k))


def load(mode, s):
    if mode == "nested":
        R = json.load(open(f"results/nested_{s}.json"))
        outer = {m: np.array(v, float) for m, v in R["outer"].items()}
        inner = {
            int(k): {m: np.array([np.nan if x is None else x for x in v], float) for m, v in d.items()}
            for k, d in R["inner"].items()
        }
        if os.path.exists(f"results/nestedcnn_{s}.json"):
            C = json.load(open(f"results/nestedcnn_{s}.json"))
            assert C["folds"] == R["folds"]
            outer["CNN"] = np.array(C["outer"]["CNN"], float)
            for k in range(5):
                inner[k]["CNN"] = np.array(
                    [np.nan if x is None else x for x in C["inner"][str(k)]["CNN"]], float
                )
        folds = [np.array(f) for f in R["folds"]]
    else:
        R = json.load(open(f'results/{mode.split(":")[1]}{s}.json'))
        outer = {m: np.array(v, float) for m, v in R["oof"].items()}
        inner = {k: outer for k in range(5)}
        unit = np.array(R["tr_unit"])
        folds = (
            [np.array(f) for f in R["folds"]]
            if R.get("folds")
            else [np.unique(unit[b]) for _, b in GroupKFold(5).split(unit, unit, groups=unit)]
        )
    return R, outer, inner, folds


def trigger(p, rul, units, ulist, L):
    """RT[u, m]: true RUL at first cycle with predicted RUL <= L + m (-1 if never). Monotone non-decreasing in m."""
    T = L + np.arange(MMAX + 1)
    RT = np.full((len(ulist), MMAX + 1), -1.0)
    for i, u in enumerate(ulist):
        ix = np.where(units == u)[0]
        pu = p[ix]
        if np.isnan(pu).any():
            continue
        cm = np.minimum.accumulate(pu)
        j = np.searchsorted(-cm, -T, side="left")
        ok = j < len(cm)
        RT[i, ok] = rul[ix][j[ok]]
    return RT


def req_margin(RT, L):
    """Smallest margin with timely alarm per engine (MMAX + 1 if none)."""
    ok = RT >= L
    r = np.where(ok.any(1), ok.argmax(1), MMAX + 1)
    return r


def run(mode, s):
    R, outer, inner, folds = load(mode, s)
    units = np.array(R["tr_unit"])
    rul = np.array(R["tr_rul"], float)
    y = np.array(R["tr_y"], float)
    ulist = np.unique(units)
    uidx = {u: i for i, u in enumerate(ulist)}
    n = len(ulist)
    fold_idx = [np.array([uidx[u] for u in f]) for f in folds]
    models = [m for m in ("LR", "Ridge", "RF", "GB", "CNN") if m in outer]
    out = {
        "keeps": R.get("keep"),
        "n": n,
        "models": models,
        "oof_rmse": {m: float(np.sqrt(np.mean((outer[m] - y) ** 2))) for m in models},
    }
    # engine-level accuracy on the predictions used by the decision lens (matched population)
    emae = {
        m: np.array([np.mean(np.abs(outer[m][units == u] - y[units == u])) for u in ulist]) for m in models
    }
    eq = {}
    for a_, b_ in (("RF", "GB"), ("GB", "RF"), ("CNN", "GB"), ("LR", "GB")):
        if a_ in models and b_ in models and (a_, b_) != ("GB", "RF"):
            d = emae[a_] - emae[b_]
            bs = np.array([d[rng.integers(0, n, n)].mean() for _ in range(B)])
            lo, hi = np.percentile(bs, [5, 95])
            eq[f"{a_}-{b_}"] = dict(
                mean=float(d.mean()),
                ci90=[float(lo), float(hi)],
                equiv={str(e): bool(-e < lo and hi < e) for e in (1, 2, 3)},
            )
    out["eq_oof_engineMAE"] = eq
    life = np.array([rul[units == u].max() for u in ulist])
    for L in LEADS:
        RTo = {m: trigger(outer[m], rul, units, ulist, L) for m in models}
        REQin = {}
        for m in models:
            REQin[m] = {}
            for k in range(5):
                RTi = trigger(inner[k][m], rul, units, ulist, L)
                REQin[m][k] = req_margin(RTi, L)

        def evaluate(samp, a):
            """samp: list per fold of resampled engine indices. Returns per-model per-instance (alarm RUL) and margins."""
            res = {}
            for m in models:
                rr = []
                mg = []
                for k in range(5):
                    cal = np.concatenate([samp[j] for j in range(5) if j != k])
                    req = np.sort(REQin[m][k][cal])[::-1]
                    allowed = int(np.floor(a * len(cal)))
                    mk = int(min(req[allowed], MMAX)) if allowed < len(req) else 0
                    mg.append(mk)
                    rr.append(RTo[m][samp[k], mk])
                res[m] = (np.concatenate(rr), mg)
            return res

        def summarise(res):
            S = {}
            for m, (r, mg) in res.items():
                miss = r < L
                S[m] = dict(miss=miss, unused=np.where(miss, np.nan, r - L), margins=mg)
            return S

        out[str(L)] = {"miss_m0": {m: float((RTo[m][:, 0] < L).mean()) for m in models}}
        for a in TARGETS:
            base = summarise(evaluate(fold_idx, a))
            pairs = [("RF", "GB"), ("GB", "LR"), ("RF", "LR"), ("CNN", "GB"), ("CNN", "LR"), ("CNN", "RF")]
            pairs = [p for p in pairs if p[0] in models and p[1] in models]

            def pdiff(S, p):
                u1, u2 = S[p[0]]["unused"], S[p[1]]["unused"]
                both = ~np.isnan(u1) & ~np.isnan(u2)
                return (u1 - u2)[both].mean(), int(both.sum())

            def cost(S, m, c):
                return float((np.nansum(S[m]["unused"]) + c * S[m]["miss"].sum()) / len(S[m]["miss"]))

            boots = {p: [] for p in pairs}
            bcost = {(p, c): [] for p in pairs for c in COSTS}
            for _ in range(B):
                samp = [fi[rng.integers(0, len(fi), len(fi))] for fi in fold_idx]
                Sb = summarise(evaluate(samp, a))
                for p in pairs:
                    boots[p].append(pdiff(Sb, p)[0])
                    for c in COSTS:
                        bcost[(p, c)].append(cost(Sb, p[0], c) - cost(Sb, p[1], c))
            md = {}
            for m in models:
                k = int(base[m]["miss"].sum())
                md[m] = dict(
                    margins=base[m]["margins"],
                    misses=k,
                    n=n,
                    cp_upper=cp_upper(k, n),
                    unused=float(np.nanmean(base[m]["unused"])),
                    n_timely=int((~base[m]["miss"]).sum()),
                    cost={str(c): cost(base, m, c) for c in COSTS},
                )
            pr = {}
            for p in pairs:
                mean, nb = pdiff(base, p)
                lo, hi = np.percentile(boots[p], [2.5, 97.5])
                bb = np.array(boots[p])
                pval = float(min(1.0, 2 * min((bb <= 0).mean(), (bb >= 0).mean())))
                pr[f"{p[0]}-{p[1]}"] = dict(
                    mean=float(mean),
                    ci95=[float(lo), float(hi)],
                    n=nb,
                    sig=int(0 if lo <= 0 <= hi else np.sign(mean)),
                    p=pval,
                    cost={
                        str(c): dict(
                            mean=cost(base, p[0], c) - cost(base, p[1], c),
                            ci95=[float(x) for x in np.percentile(bcost[(p, c)], [2.5, 97.5])],
                        )
                        for c in COSTS
                    },
                )
            out[str(L)][str(a)] = dict(models=md, pairs=pr)
        # cross-fitted fixed-age baseline (zero-miss): replace at min training life - L
        miss = 0
        un = []
        for k in range(5):
            cal = np.concatenate([fold_idx[j] for j in range(5) if j != k])
            T = life[cal].min() - L
            lt = life[fold_idx[k]]
            miss += int((lt - T < L).sum())
            un += list((lt - T - L)[lt - T >= L])
        out[str(L)]["time"] = dict(misses=miss, n=n, unused=float(np.mean(un)))
        # first-alarm timing check: earliest alarm cycle index per model at the zero-miss margins (padding relevance)
    return out


if __name__ == "__main__":
    mode, outf = sys.argv[1], sys.argv[2]
    res = {}
    for s in sys.argv[3:]:
        res[s] = run(mode, s)
        r = res[s]
        print(
            "=====",
            s,
            mode,
            "OOF RMSE",
            {m: round(v, 2) for m, v in r["oof_rmse"].items()},
            "eq",
            {
                k: (round(v["mean"], 2), [round(x, 2) for x in v["ci90"]], v["equiv"])
                for k, v in r["eq_oof_engineMAE"].items()
            },
            flush=True,
        )
        for L in LEADS:
            for a in TARGETS:
                x = r[str(L)][str(a)]
                print(
                    f" L={L} a={a}",
                    {
                        m: (v["misses"], round(v["unused"], 1), f"m{min(v['margins'])}-{max(v['margins'])}")
                        for m, v in x["models"].items()
                    },
                    {
                        k: (round(v["mean"], 1), [round(c, 1) for c in v["ci95"]], v["n"])
                        for k, v in x["pairs"].items()
                    },
                    flush=True,
                )
            print(" time", r[str(L)]["time"])
    json.dump(res, open(outf, "w"))
