"""Print the decision-lens numbers quoted in the paper from the nested and sensitivity results.

Usage:
    python scripts/report_numbers.py
"""

import json, os, numpy as np

O = "results"
subs = [s for s in ("FD001", "FD002", "FD003", "FD004")]
N = {}
for s in subs:
    f = f"{O}/decision_nested_{s}.json"
    if os.path.exists(f):
        N.update(json.load(open(f)))
P0 = json.load(open(f"{O}/decision_P0.json"))
P1 = json.load(open(f"{O}/decision_P1.json"))
P2 = json.load(open(f"{O}/decision_P2.json"))
C150 = json.load(open(f"{O}/decision_cap150.json"))
have = [s for s in subs if s in N]
print("subsets with nested results:", have)
rng = lambda v: f"{min(v):.1f} to {max(v):.1f}"
# accuracy
print("nested OOF RMSE", {s: {m: round(v, 2) for m, v in N[s]["oof_rmse"].items()} for s in have})
print("nonnested OOF RMSE", {s: {m: round(v, 2) for m, v in P0[s]["oof_rmse"].items()} for s in have})
print(
    "delta nested-nonnested",
    {s: {m: round(N[s]["oof_rmse"][m] - P0[s]["oof_rmse"][m], 2) for m in ("LR", "RF", "GB")} for s in have},
)
print(
    "OOF eq RF-GB",
    {
        s: (
            round(N[s]["eq_oof_engineMAE"]["RF-GB"]["mean"], 2),
            [round(x, 2) for x in N[s]["eq_oof_engineMAE"]["RF-GB"]["ci90"]],
        )
        for s in have
    },
)
if all("CNN-GB" in N[s]["eq_oof_engineMAE"] for s in have):
    print(
        "OOF eq CNN-GB",
        {
            s: (
                round(N[s]["eq_oof_engineMAE"]["CNN-GB"]["mean"], 2),
                [round(x, 2) for x in N[s]["eq_oof_engineMAE"]["CNN-GB"]["ci90"]],
            )
            for s in have
        },
    )
print("keeps identical", {s: len({tuple(v) for v in (N[s]["keeps"] or {}).values()}) for s in have})
# decision, primary
for L in ("10", "15", "30"):
    for a in ("0.0", "0.02"):
        print(f"--- L={L} a={a}")
        for s in have:
            x = N[s][L][a]
            md = x["models"]
            print(
                " ",
                s,
                {
                    m: (
                        md[m]["misses"],
                        round(md[m]["unused"], 1),
                        f"{min(md[m]['margins'])}-{max(md[m]['margins'])}",
                        round(100 * md[m]["cp_upper"], 1),
                    )
                    for m in md
                },
                {
                    k: (
                        round(v["mean"], 1),
                        [round(c, 1) for c in v["ci95"]],
                        v["n"],
                        v["sig"],
                        round(v["p"], 3),
                    )
                    for k, v in x["pairs"].items()
                },
                "time",
                round(N[s][L]["time"]["unused"], 1),
                N[s][L]["time"]["misses"],
            )
print("m0 miss (L=15)", {s: {m: round(v, 2) for m, v in N[s]["15"]["miss_m0"].items()} for s in have})


# Holm on primary family
def holm(ps):
    o = np.argsort(ps)
    m = len(ps)
    adj = np.empty(m)
    run = 0
    for r, i in enumerate(o):
        run = max(run, min(1, (m - r) * ps[i]))
        adj[i] = run
    return adj


for s in have:
    x = N[s]["15"]["0.0"]["pairs"]
    fam = [k for k in ("RF-GB", "GB-LR", "CNN-GB") if k in x]
    ps = [x[k]["p"] for k in fam]
    adj = holm(ps)
    print("Holm", s, {k: (round(p, 3), round(q, 3)) for k, p, q in zip(fam, ps, adj)})
# costs at L15 zero-miss
for s in have:
    x = N[s]["15"]["0.0"]["pairs"]
    print(
        "cost",
        s,
        {
            k: {
                c: (round(v["cost"][c]["mean"], 1), [round(z, 1) for z in v["cost"][c]["ci95"]])
                for c in ("25", "100", "400")
            }
            for k, v in x.items()
            if k in ("RF-GB", "GB-LR", "CNN-GB")
        },
    )
# within-class significance count & class-level
for name, R in (("nested", N), ("P0nn", P0), ("P1", P1), ("P2", P2)):
    cnt = 0
    tot = 0
    gl = {}
    for s in have if name == "nested" else subs:
        for L in ("10", "15", "30"):
            for a in ("0.0", "0.02"):
                p = R[s][L][a]["pairs"]
                tot += 1
                cnt += p["RF-GB"]["sig"] != 0
                gl[(s, L, a)] = p["GB-LR"]["sig"]
    print(
        name,
        "RF-GB significant cells",
        cnt,
        "of",
        tot,
        "| GB-LR sig",
        {k: v for k, v in gl.items() if k[2] == "0.0"},
    )
print(
    "cap150",
    {
        s: {
            L: {k: (round(v["mean"], 1), v["sig"]) for k, v in C150[s][L]["0.0"]["pairs"].items()}
            for L in ("15", "30")
        }
        for s in C150
    },
)
