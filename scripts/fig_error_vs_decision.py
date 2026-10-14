"""Fig. 3: unused life relative to gradient boosting against the difference in nested out-of-fold RMSE.

Usage:
    python scripts/fig_error_vs_decision.py [results/decision_nested.json]
"""

import json, sys, numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 10})
D = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "results/decision_nested.json"))
subs = ["FD001", "FD002", "FD003", "FD004"]
scol = {"FD001": "#000000", "FD002": "#1f4e79", "FD003": "#8c8c8c", "FD004": "#b35806"}
off = {"FD001": -0.18, "FD002": -0.06, "FD003": 0.06, "FD004": 0.18}
mk = {"RF": "^", "LR": "s", "CNN": "D"}
fig, axs = plt.subplots(1, 3, figsize=(6.5, 2.6), dpi=600, sharey=True)
ylo, yhi = 0, 0
for ax, Ls in zip(axs, ["10", "15", "30"]):
    ax.axhspan(-200, 0, color="#f0f0f0", zorder=0)
    ax.axhline(0, color="black", lw=0.7)
    ax.axvline(0, color="black", lw=0.7)
    for s in subs:
        x = D[s][Ls]["0.0"]["pairs"]
        r = D[s]["oof_rmse"]
        items = (
            [("RF", x["RF-GB"], 1)]
            + [("LR", x["GB-LR"], -1)]
            + ([("CNN", x["CNN-GB"], 1)] if "CNN-GB" in x else [])
        )
        for m, p, sign in items:
            y = sign * p["mean"]
            lo, hi = sorted([sign * p["ci95"][0], sign * p["ci95"][1]])
            xx = r[m] - r["GB"] + off[s]
            ylo, yhi = min(ylo, lo), max(yhi, hi)
            filled = m == "CNN"
            ax.errorbar(
                xx,
                y,
                yerr=[[y - lo], [hi - y]],
                fmt=mk[m],
                ms=5.5 if m != "CNN" else 5.5,
                mfc=scol[s] if filled else "white",
                mec=scol[s],
                ecolor=scol[s],
                elinewidth=1.1,
                capsize=2,
                mew=1.3,
                zorder=3,
            )
    ax.plot(0, 0, "o", ms=5, mfc="white", mec="black", mew=1.3, zorder=6)
    ax.set_title(f"Lead time L = {Ls} cycles", fontsize=10)
    ax.grid(lw=0.3, alpha=0.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
for ax in axs:
    ax.set_ylim(ylo - 3, yhi + 3)
axs[0].text(
    0.98,
    0.03,
    "shaded: less life\ngiven up than GB",
    transform=axs[0].transAxes,
    ha="right",
    va="bottom",
    fontsize=8,
    style="italic",
    color="#404040",
)
axs[0].set_ylabel("Unused life minus that of\ngradient boosting (cycles)")
fig.supxlabel("Nested out-of-fold RMSE minus that of gradient boosting (cycles)", fontsize=9.5, y=0.0)
h = [Line2D([], [], color=scol[s], lw=2, label=s) for s in subs] + [
    Line2D([], [], marker="o", ls="", mfc="white", mec="black", mew=1.3, ms=6, label="GB (origin)"),
    Line2D([], [], marker="^", ls="", mfc="white", mec="black", mew=1.5, ms=7, label="Random forest"),
    Line2D([], [], marker="s", ls="", mfc="white", mec="black", mew=1.5, ms=7, label="Linear"),
    Line2D([], [], marker="D", ls="", color="black", ms=6.5, label="1D-CNN"),
]
fig.legend(
    handles=h,
    loc="upper center",
    ncol=4,
    frameon=False,
    fontsize=9,
    bbox_to_anchor=(0.5, 1.16),
    handlelength=1.3,
    columnspacing=0.9,
)
plt.tight_layout(w_pad=0.8)
plt.savefig("results/figures/fig3_error_vs_decision.png", bbox_inches="tight")
plt.close()
print("ok", round(ylo), round(yhi))
