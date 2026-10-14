"""Diagnostics behind Sect. 4.3 and 4.5 (Supplementary Tables S2 and S4).

(1) Distribution of nested out-of-fold predicted RUL near failure (CNN floor diagnostic).
(2) Margin decomposition by lead time and model: zero-miss margins, 95th-percentile required
    margins, tail gap, error spread at true RUL = L and alarm timing.

Usage:
    python scripts/diagnostics.py           # writes results/diagnostics.json
"""

import json, numpy as np
from decision_nested import load, trigger, req_margin, MMAX

SUBS = ("FD001", "FD002", "FD003", "FD004")
RS = (0, 10, 15, 30, 50)
out = {}
D = json.load(open("results/decision_nested.json"))
for s in SUBS:
    R, outer, inner, folds = load("nested", s)
    units = np.array(R["tr_unit"])
    rul = np.array(R["tr_rul"], float)
    ulist = np.unique(units)
    uidx = {u: i for i, u in enumerate(ulist)}
    fold_idx = [np.array([uidx[u] for u in f]) for f in folds]
    out[s] = {"dist": {}, "healthy": {}, "margin": {}}
    for m in ("LR", "RF", "GB", "CNN"):
        p = outer[m]
        out[s]["dist"][m] = {}
        for r in RS:
            v = p[rul == r]
            q = np.percentile(v, [25, 50, 75])
            out[s]["dist"][m][str(r)] = dict(
                n=int(len(v)),
                mean_err=float(v.mean() - r),
                q25=float(q[0]),
                med=float(q[1]),
                q75=float(q[2]),
                share_le_r=float((v <= r).mean()),
                sd=float(v.std(ddof=1)),
            )
        out[s]["healthy"][m] = dict(
            mean_pred_rul_ge150=float(p[rul >= 150].mean()), max_pred=float(p.max()), min_pred=float(p.min())
        )
        # minimum prediction ever reached before true RUL < 15: does the CNN ever cross 15 in time?
        out[s]["dist"][m]["min_pred_by_rul15"] = float(
            np.median([p[(units == u) & (rul >= 15)].min() for u in ulist])
        )
    for L in (10, 15, 30):
        out[s]["margin"][str(L)] = {}
        for m in ("LR", "RF", "GB", "CNN"):
            req_all = []
            zero = []
            p95 = []
            gap = []
            ntop = []
            for k in range(5):
                RTi = trigger(inner[k][m], rul, units, ulist, L)
                req = req_margin(RTi, L)
                cal = np.concatenate([fold_idx[j] for j in range(5) if j != k])
                r = np.sort(req[cal])[::-1]
                zero.append(r[0])
                p95.append(r[int(np.floor(0.05 * len(r)))])
                gap.append(r[0] - r[1])
                ntop.append(int((r > np.median(r) + 3 * (np.percentile(r, 75) - np.percentile(r, 25))).sum()))
                req_all.append(req[cal])
            # held-out alarm timing at the zero-miss margins
            RTo = trigger(outer[m], rul, units, ulist, L)
            mg = D[s][str(L)]["0.0"]["models"][m]["margins"]
            at = np.concatenate([RTo[fold_idx[k], mg[k]] for k in range(5)])
            # alarm timing if the margin were the 95th-percentile requirement (5% target), for contrast
            at95 = np.concatenate([RTo[fold_idx[k], p95[k]] for k in range(5)])
            # local error at true RUL = L (what the margin has to cover): held-out engines
            pe = outer[m][rul == L] - L
            out[s]["margin"][str(L)][m] = dict(
                zero=[int(x) for x in zero],
                p95=[int(x) for x in p95],
                gap_top2=[int(x) for x in gap],
                outliers=ntop,
                med_alarm_rul=float(np.median(at[at >= L])),
                unused=float(np.mean(at[at >= L] - L)),
                unused_p95margin=float(np.mean(at95[at95 >= L] - L)),
                err_at_L_mean=float(pe.mean()),
                err_at_L_p95=float(np.percentile(pe, 95)),
                err_at_L_sd=float(pe.std(ddof=1)),
            )
json.dump(out, open("results/diagnostics.json", "w"), indent=1)
for s in SUBS:
    print("=====", s)
    for m in ("LR", "RF", "GB", "CNN"):
        d = out[s]["dist"][m]
        print(
            f" {m:4s}",
            " ".join(
                f"r{r}: med {d[str(r)]['med']:.1f} [{d[str(r)]['q25']:.1f},{d[str(r)]['q75']:.1f}] le {d[str(r)]['share_le_r']:.2f} bias {d[str(r)]['mean_err']:+.1f}"
                for r in RS
            ),
            "healthy",
            {k: round(v, 1) for k, v in out[s]["healthy"][m].items()},
            "minpred>=15",
            round(d["min_pred_by_rul15"], 1),
        )
    for L in ("10", "15", "30"):
        for m in ("LR", "RF", "GB", "CNN"):
            x = out[s]["margin"][L][m]
            print(
                f"  L{L} {m:4s} zero {x['zero']} p95 {x['p95']} gap {x['gap_top2']} out {x['outliers']} alarmRUL {x['med_alarm_rul']:.0f} unused {x['unused']:.1f} (p95m {x['unused_p95margin']:.1f}) errL mean {x['err_at_L_mean']:+.1f} p95 {x['err_at_L_p95']:+.1f} sd {x['err_at_L_sd']:.1f}"
            )
