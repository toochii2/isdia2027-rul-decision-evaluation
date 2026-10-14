"""Supplementary analyses: out-of-fold miss-rate curves (Table S5), significant-contrast counts
across partitions (Table S6) and sensitivity to a penalty per missed engine (Table S7).

Usage:
    python scripts/supplementary_analyses.py   # writes results/supplementary_analyses.json
"""

import json, numpy as np
from decision_nested import load, trigger, req_margin

SUBS = ("FD001", "FD002", "FD003", "FD004")
MODELS = ("LR", "Ridge", "RF", "GB", "CNN")
QS = (0.0, 0.01, 0.02, 0.05, 0.10)
res = {"curve": {}}
for s in SUBS:
    R, outer, inner, folds = load("nested", s)
    units = np.array(R["tr_unit"])
    rul = np.array(R["tr_rul"], float)
    ulist = np.unique(units)
    n = len(ulist)
    res["curve"][s] = {}
    for L in (10, 15, 30):
        res["curve"][s][str(L)] = {}
        for m in MODELS:
            RT = trigger(outer[m], rul, units, ulist, L)
            req = np.sort(req_margin(RT, L))[::-1]
            row = {}
            for q in QS:
                mg = int(req[int(np.floor(q * n))])
                r = RT[:, mg]
                ok = r >= L
                row[str(q)] = dict(margin=mg, misses=int((~ok).sum()), unused=float(np.mean(r[ok] - L)))
            res["curve"][s][str(L)][m] = row
# (b) significant contrasts across partitions
P = {
    "nested": json.load(open("results/decision_nested.json")),
    "P1": json.load(open("results/decision_P1.json")),
    "P2": json.load(open("results/decision_P2.json")),
}
cnt = {}
for part, D in P.items():
    for s in SUBS:
        for L in ("10", "15", "30"):
            for a in ("0.0", "0.02"):
                pr = D[s][L][a]["pairs"]
                for pair in ("GB-LR", "RF-LR", "RF-GB"):
                    sg = pr[pair]["sig"]
                    if sg == 0:
                        continue
                    if pair == "RF-GB":
                        who = "RF" if sg < 0 else "GB"
                    else:
                        who = "ensemble" if sg < 0 else "linear"
                    key = f"{'ens' if pair == 'RF-GB' else 'linens'}|{L}|{who}"
                    cnt.setdefault(key, {}).setdefault(part, 0)
                    cnt[key][part] += 1
res["counts"] = cnt
# (c) cost sensitivity: significant contrasts (nested) that lose significance or reverse when misses are charged
D = P["nested"]
cs = {}
for c in ("25", "100", "400"):
    kept = lost = rev = new = 0
    for s in SUBS:
        for L in ("10", "15", "30"):
            for a in ("0.0", "0.02"):
                for pair, v in D[s][L][a]["pairs"].items():
                    lo, hi = v["cost"][c]["ci95"]
                    sc = 0 if lo <= 0 <= hi else int(np.sign(v["cost"][c]["mean"]))
                    if v["sig"] != 0 and sc == v["sig"]:
                        kept += 1
                    elif v["sig"] != 0 and sc == 0:
                        lost += 1
                    elif v["sig"] != 0 and sc == -v["sig"]:
                        rev += 1
                    elif v["sig"] == 0 and sc != 0:
                        new += 1
    cs[c] = dict(kept=kept, lost=lost, reversed=rev, new=new)
res["cost"] = cs
json.dump(res, open("results/supplementary_analyses.json", "w"), indent=1)
for s in SUBS:
    for L in ("10", "15", "30"):
        c = res["curve"][s][L]
        print(
            s,
            L,
            " | ".join(
                f"{m}: "
                + " ".join(f"{q}:{c[m][q]['unused']:.1f}(m{c[m][q]['margin']})" for q in ("0.05", "0.0"))
                for m in MODELS
            ),
        )
print(json.dumps(cnt, indent=0))
print(cs)
