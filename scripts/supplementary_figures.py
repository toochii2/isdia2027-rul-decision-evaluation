"""Supplementary Figs. S1 (predicted vs true RUL near failure) and S2 (miss-rate curves).

Usage:
    python scripts/supplementary_figures.py    # writes results/figures/figS1_*.png, figS2_*.png
"""

import json, numpy as np, matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from decision_nested import load, trigger, req_margin

plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 9})
SUBS = ("FD001", "FD002", "FD003", "FD004")
col = {"LR": "#b35806", "Ridge": "#e08214", "RF": "#1f4e79", "GB": "#4a90c2", "CNN": "#000000"}
ls = {"LR": "-", "Ridge": ":", "RF": "--", "GB": "-", "CNN": "-"}
cache = {s: load("nested", s) for s in SUBS}
# Fig S1: predicted vs true RUL near failure
fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.4), dpi=300, sharey=True)
for ax, s in zip(axs, SUBS):
    R, outer, inner, folds = cache[s]
    rul = np.array(R["tr_rul"])
    x = np.arange(0, 101)
    for m in ("LR", "GB", "CNN"):
        q = np.array([np.percentile(outer[m][rul == r], [25, 50, 75]) for r in x])
        ax.fill_between(x, q[:, 0], q[:, 2], color=col[m], alpha=0.18, lw=0)
        ax.plot(
            x,
            q[:, 1],
            color=col[m],
            lw=1.3,
            label={"LR": "Linear regression", "GB": "Gradient boosting", "CNN": "1D-CNN"}[m],
        )
    ax.plot(x, x, color="grey", lw=0.8, ls="--")
    ax.axhline(15, color="grey", lw=0.6, ls=":")
    ax.set_title(s)
    ax.set_xlabel("True RUL (cycles)")
    ax.set_xlim(0, 100)
    ax.set_ylim(-20, 125)
    ax.grid(lw=0.3, alpha=0.5)
axs[0].set_ylabel("Predicted RUL (cycles)")
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.08))
plt.tight_layout()
plt.savefig("results/figures/figS1_cnn_floor.png", bbox_inches="tight")
plt.close()
# Fig S2: OOF miss-rate curves
QS = np.arange(0, 0.1001, 0.01)
curves = {}
fig, axs = plt.subplots(3, 4, figsize=(7.2, 6.0), dpi=300)
for j, s in enumerate(SUBS):
    R, outer, inner, folds = cache[s]
    units = np.array(R["tr_unit"])
    rul = np.array(R["tr_rul"], float)
    ulist = np.unique(units)
    n = len(ulist)
    curves[s] = {}
    for i, L in enumerate((10, 15, 30)):
        ax = axs[i, j]
        curves[s][L] = {}
        for m in ("LR", "Ridge", "RF", "GB", "CNN"):
            RT = trigger(outer[m], rul, units, ulist, L)
            req = np.sort(req_margin(RT, L))[::-1]
            ys = []
            for q in QS:
                mg = int(req[int(np.floor(q * n))])
                r = RT[:, mg]
                ok = r >= L
                ys.append(float(np.mean(r[ok] - L)))
            curves[s][L][m] = ys
            ax.plot(
                100 * QS,
                ys,
                ls[m],
                color=col[m],
                lw=1.3,
                label=m if m != "CNN" else "1D-CNN",
                marker="o" if m == "CNN" else None,
                ms=2.5,
            )
        ax.set_title(f"{s}, L = {L}", fontsize=8.5)
        ax.grid(lw=0.3, alpha=0.5)
        if i == 2:
            ax.set_xlabel("Allowed misses (%)")
        if j == 0:
            ax.set_ylabel("Unused life (cycles)")
axs[0, 0].legend(fontsize=6.5, frameon=False, ncol=1)
plt.tight_layout()
plt.savefig("results/figures/figS2_miss_rate_curves.png", bbox_inches="tight")
plt.close()
json.dump(curves, open("results/miss_rate_curves.json", "w"))
print("ok")
