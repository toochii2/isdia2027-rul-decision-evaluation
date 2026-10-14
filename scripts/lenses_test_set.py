"""Test-set lenses L1-L4: accuracy and fold stability, paired separability (Wilcoxon with Holm
correction, bootstrap equivalence intervals), error direction and life-stage (whole-trajectory) error.

Usage:
    python scripts/lenses_test_set.py RESULTS_DIR FD001 FD002 FD003 FD004
Reads RESULTS_DIR/cv_SUBSET.json and writes RESULTS_DIR/analysis.json.
"""

import json, sys, numpy as np
from scipy.stats import wilcoxon

rng = np.random.default_rng(7)
M = ["LR", "Ridge", "RF", "GB"]
CAP, LEAD = 125, 15


def nasa(e):
    return float(np.sum(np.where(e < 0, np.exp(-e / 13) - 1, np.exp(e / 10) - 1)))


def metr(p, y):
    e = p - y
    return dict(
        MAE=float(np.mean(np.abs(e))),
        RMSE=float(np.sqrt(np.mean(e**2))),
        R2=float(1 - np.sum(e**2) / np.sum((y - y.mean()) ** 2)),
        NASA=nasa(e),
        late=int(np.sum(e > 0)),
        early=int(np.sum(e < 0)),
        sl10=int(np.sum(e > 10)),
        sl20=int(np.sum(e > 20)),
        sl30=int(np.sum(e > 30)),
        se20=int(np.sum(e < -20)),
        mlate=float(e[e > 0].mean()) if (e > 0).any() else 0.0,
        mearly=float(e[e < 0].mean()) if (e < 0).any() else 0.0,
    )


def holm(ps):
    o = np.argsort(ps)
    adj = np.empty(len(ps))
    run = 0
    for i, k in enumerate(o):
        run = max(run, (len(ps) - i) * ps[k])
        adj[k] = min(1, run)
    return adj.tolist()


def analyse(path):
    R = json.load(open(path))
    out = {"sub": R["sub"]}
    yt = np.array(R["yt"], float)
    raw = np.array(R["rul_raw"], float)
    n = len(yt)
    TP = {m: np.array(R["test_preds"][m]) for m in M}  # folds x n
    # A. per-fold metrics and rank stability
    fold = {m: [metr(TP[m][k], yt) for k in range(TP[m].shape[0])] for m in M}
    summ = {}
    for m in M:
        summ[m] = {
            k: (float(np.mean([f[k] for f in fold[m]])), float(np.std([f[k] for f in fold[m]], ddof=1)))
            for k in ("MAE", "RMSE", "R2", "NASA", "sl20")
        }
    wins = {}
    for key in ("RMSE", "NASA", "sl20", "MAE"):
        w = {m: 0 for m in M}
        for k in range(5):
            vals = {m: fold[m][k][key] for m in M}
            best = min(vals.values())
            ties = [m for m in M if vals[m] == best]
            for m in ties:
                w[m] += 1 / len(ties)
        wins[key] = w
    out["fold_summary"] = summ
    out["wins"] = wins
    # B. fold-ensemble
    E = {m: TP[m].mean(0) for m in M}
    out["ens"] = {m: metr(E[m], yt) for m in M}
    out["ens_uncapped"] = {m: metr(E[m], raw) for m in M}
    best = min(M, key=lambda m: out["ens"][m]["RMSE"])
    out["best_rmse"] = best
    others = [m for m in M if m != best]
    ae = {m: np.abs(E[m] - yt) for m in M}
    tests = {}
    for m in others:
        w = wilcoxon(ae[best], ae[m], method="approx")
        z = abs(w.zstatistic) if hasattr(w, "zstatistic") else np.nan
        d = ae[m] - ae[best]
        bs = np.array([d[rng.integers(0, n, n)].mean() for _ in range(10000)])
        en = E[m] - yt
        eb = E[best] - yt
        bn = []
        for _ in range(4000):
            i = rng.integers(0, n, n)
            bn.append(nasa(en[i]) - nasa(eb[i]))
        tests[m] = dict(
            W=float(w.statistic),
            p=float(w.pvalue),
            r=float(z / np.sqrt(n)),
            dMAE=float(d.mean()),
            ci90=[float(np.percentile(bs, 5)), float(np.percentile(bs, 95))],
            ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            dNASA_ci95=[float(np.percentile(bn, 2.5)), float(np.percentile(bn, 97.5))],
        )
    adj = holm([tests[m]["p"] for m in others])
    for m, a in zip(others, adj):
        tests[m]["p_holm"] = a
        tests[m]["equiv_2cyc"] = bool(-2 < tests[m]["ci90"][0] and tests[m]["ci90"][1] < 2)
    out["tests"] = tests
    # C. whole-trajectory test evaluation
    tu = np.array(R["te_unit"])
    tc = np.array(R["te_cycle"])
    maxc = {u: tc[tu == u].max() for u in np.unique(tu)}
    units = np.unique(tu)
    uidx = {u: i for i, u in enumerate(units)}
    true_all = np.array([raw[uidx[u]] + maxc[u] - c for u, c in zip(tu, tc)])
    true_cap = np.minimum(true_all, CAP)
    TA = {m: np.array(R["test_all"][m]).mean(0) for m in M}
    bands = [(0, 30, "critical"), (31, 60, "warning"), (61, 90, "moderate"), (91, 125, "healthy")]
    groups = {u: np.where(tu == u)[0] for u in units}

    def clustered(stat_fn, B=1000):
        vals = []
        for _ in range(B):
            samp = rng.choice(units, len(units), replace=True)
            idx = np.concatenate([groups[u] for u in samp])
            vals.append(stat_fn(idx))
        return [float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))]

    stage = {}
    for m in M:
        e = TA[m] - true_cap
        stage[m] = {}
        for lo, hi, nm in bands:
            mask = (true_cap >= lo) & (true_cap <= hi)
            f_mae = lambda idx: np.mean(np.abs(e[idx][mask[idx]])) if mask[idx].any() else np.nan
            f_bias = lambda idx: np.mean(e[idx][mask[idx]]) if mask[idx].any() else np.nan
            stage[m][nm] = dict(
                n_cycles=int(mask.sum()),
                n_engines=int(len(np.unique(tu[mask]))),
                MAE=float(np.mean(np.abs(e[mask]))),
                bias=float(np.mean(e[mask])),
                MAE_ci=clustered(f_mae, 400),
                bias_ci=clustered(f_bias, 400),
            )
    out["stage_traj"] = stage
    # binned signed error curve (10-cycle bins) for figures
    bins = np.arange(0, 130, 10)
    curve = {}
    for m in M:
        e = TA[m] - true_cap
        curve[m] = []
        for b0 in bins:
            mask = (true_cap >= b0) & (true_cap < b0 + 10) if b0 < 120 else (true_cap >= 120)
            curve[m].append([float(b0 + 5), float(e[mask].mean()), float(np.mean(np.abs(e[mask])))])
    out["curve"] = curve
    # D. decision analysis on out-of-fold run-to-failure trajectories
    ou = np.array(R["tr_unit"])
    orul = np.array(R["tr_rul"], float)
    OF = {m: np.array(R["oof"][m]) for m in M}
    tunits = np.unique(ou)
    gi = {u: np.where(ou == u)[0] for u in tunits}
    life = np.array(
        [orul[gi[u]].max() for u in tunits]
    )  # failure cycle minus first cycle (approx lifetime-1)
    margins = np.arange(0, 101, 1)
    dec = {}
    for m in M:
        miss, waste = [], []
        for mg in margins:
            ms, ws = 0, []
            for u in tunits:
                idx = gi[u]
                p = OF[m][idx]
                r = orul[idx]
                trig = np.where(p <= LEAD + mg)[0]
                if len(trig) == 0 or r[trig[0]] < LEAD:
                    ms += 1
                else:
                    ws.append(r[trig[0]] - LEAD)
            miss.append(ms)
            waste.append(float(np.mean(ws)) if ws else np.nan)
        miss = np.array(miss)
        waste = np.array(waste)
        z = np.where(miss == 0)[0]
        o1 = np.where(miss <= 1)[0]
        dec[m] = dict(
            miss0=int(miss[0]),
            waste0=float(waste[0]),
            m_star=int(margins[z[0]]) if len(z) else None,
            waste_star=float(waste[z[0]]) if len(z) else None,
            m_1=int(margins[o1[0]]) if len(o1) else None,
            waste_1=float(waste[o1[0]]) if len(o1) else None,
            curve=[[int(a), int(b), float(c)] for a, b, c in zip(margins, miss, waste)],
        )
    # bootstrap over engines for zero-miss margin and unused life
    RT = {}
    for m in M:
        mat = np.full((len(tunits), len(margins)), np.nan)
        for i, u in enumerate(tunits):
            idx = gi[u]
            p = OF[m][idx]
            r = orul[idx]
            for j, mg in enumerate(margins):
                trig = np.where(p <= LEAD + mg)[0]
                mat[i, j] = r[trig[0]] if len(trig) else -1
        RT[m] = mat

    def star(mat, rows):
        sub = mat[rows]
        ok = (sub >= LEAD).all(0)
        if not ok.any():
            return np.nan, np.nan
        j = np.argmax(ok)
        return margins[j], float(np.mean(sub[:, j] - LEAD))

    boots = {m: [] for m in M}
    for _ in range(1000):
        rows = rng.integers(0, len(tunits), len(tunits))
        for m in M:
            boots[m].append(star(RT[m], rows))
    for m in M:
        b = np.array(boots[m], float)
        dec[m]["m_star_ci"] = [float(np.nanpercentile(b[:, 0], 2.5)), float(np.nanpercentile(b[:, 0], 97.5))]
        dec[m]["waste_star_ci"] = [
            float(np.nanpercentile(b[:, 1], 2.5)),
            float(np.nanpercentile(b[:, 1], 97.5)),
        ]
    bw = {m: np.array(boots[m], float)[:, 1] for m in M}
    dec["pairs"] = {f"{a}<{b}": float(np.nanmean(bw[a] < bw[b])) for a in M for b in M if a != b}
    dec["best_decision"] = min(
        M, key=lambda m: dec[m]["waste_star"] if dec[m]["waste_star"] is not None else 1e9
    )
    # lead-time sensitivity
    sens = {}
    for Ld in (10, 30):
        sens[Ld] = {}
        for m in M:
            res_m = None
            for mg in margins:
                ok, ws = True, []
                for u in tunits:
                    idx = gi[u]
                    p = OF[m][idx]
                    r = orul[idx]
                    trig = np.where(p <= Ld + mg)[0]
                    if len(trig) == 0 or r[trig[0]] < Ld:
                        ok = False
                        break
                    ws.append(r[trig[0]] - Ld)
                if ok:
                    res_m = (int(mg), float(np.mean(ws)))
                    break
            sens[Ld][m] = res_m
        sens[Ld]["TIME"] = float(np.mean(life - (life.min() - Ld) - Ld))
    dec["lead_sens"] = sens
    # time-based baseline: replace at fixed age T (chosen in-sample to give zero misses)
    lif = life  # RUL at first cycle = lifetime-1
    T0 = lif.min() - LEAD
    dec["TIME"] = dict(
        T_star=float(T0),
        waste_star=float(np.mean(lif - T0 - LEAD)),
        m_1=None,
        waste_1=float(np.mean(lif - (np.sort(lif)[1] - LEAD) - LEAD)),
    )
    out["decision"] = dec
    out["n_train_engines"] = int(len(tunits))
    # OOF accuracy at all cycles (capped truth) for reference
    oy = np.array(R["tr_y"], float)
    out["oof_rmse"] = {m: float(np.sqrt(np.mean((OF[m] - oy) ** 2))) for m in M}
    return out


if __name__ == "__main__":
    res = {}
    for s in sys.argv[2:]:
        res[s] = analyse(f"{sys.argv[1]}/cv_{s}.json")
        r = res[s]
        print("=====", s, "best", r["best_rmse"])
        for m in M:
            f = r["fold_summary"][m]
            e = r["ens"][m]
            d = r["decision"][m]
            print(
                f"{m:6s} fold RMSE {f['RMSE'][0]:.2f}±{f['RMSE'][1]:.2f} NASA {f['NASA'][0]:.0f}±{f['NASA'][1]:.0f} | ens MAE {e['MAE']:.2f} RMSE {e['RMSE']:.2f} NASA {e['NASA']:.0f} late {e['late']} sl20 {e['sl20']} | unc RMSE {r['ens_uncapped'][m]['RMSE']:.2f} | miss0 {d['miss0']} m* {d['m_star']} waste* {d['waste_star']} m1 {d['m_1']} waste1 {d['waste_1']}"
            )
        print("wins", r["wins"])
        print(
            "tests",
            {
                k: (
                    round(v["p_holm"], 4),
                    round(v["dMAE"], 2),
                    [round(x, 2) for x in v["ci90"]],
                    v["equiv_2cyc"],
                )
                for k, v in r["tests"].items()
            },
        )
        print("TIME", r["decision"]["TIME"])
        print(
            "dec CI",
            {
                m: (r["decision"][m]["m_star_ci"], [round(x, 1) for x in r["decision"][m]["waste_star_ci"]])
                for m in M
            },
        )
        print(
            "pairs",
            {
                k: round(v, 2)
                for k, v in r["decision"]["pairs"].items()
                if k in ("RF<GB", "GB<RF", "RF<LR", "GB<LR")
            },
        )
        print("best_decision", r["decision"]["best_decision"])
        print("lead_sens", r["decision"]["lead_sens"])
        for m in M:
            print(
                m,
                {
                    b: (
                        v["n_engines"],
                        round(v["MAE"], 1),
                        round(v["bias"], 1),
                        [round(x, 1) for x in v["bias_ci"]],
                    )
                    for b, v in r["stage_traj"][m].items()
                },
            )
    json.dump(res, open(f"{sys.argv[1]}/analysis.json", "w"))
