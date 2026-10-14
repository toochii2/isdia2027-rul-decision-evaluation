"""Fig. 2: mean signed error over all test cycles by true RUL, with engine-clustered 95% bands.

Usage:
    python scripts/fig_trajectory_error.py   # needs results/curve_ci.json (trajectory_bands.py)
"""

import json, numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 9})
OUT = "results"
subs = ["FD001", "FD002", "FD003", "FD004"]
lab = {"LR": "Linear", "Ridge": "Ridge", "RF": "Random forest", "GB": "Gradient boosting"}
sty = {
    "LR": dict(color="#8c8c8c", ls=":", lw=1.6),
    "Ridge": dict(color="#5a5a5a", ls="-.", lw=1.2),
    "RF": dict(color="#1f4e79", ls="--", lw=1.6),
    "GB": dict(color="#000000", ls="-", lw=1.6),
}
mk = {"LR": "s", "RF": "^", "GB": "o"}
CC = json.load(open(f"{OUT}/curve_ci.json"))

fig, axs = plt.subplots(1, 4, figsize=(6.2, 2.05), dpi=600, sharey=True)
for ax, s in zip(axs, subs):
    for m in ["LR", "Ridge", "RF", "GB"]:
        c = np.array(CC[s][m])
        ax.plot(c[:, 0], c[:, 1], label=lab[m], **sty[m])
        if m in ("LR", "GB"):
            ax.fill_between(
                c[:, 0], c[:, 2], c[:, 3], color={"LR": "#d9a400", "GB": "#1f4e79"}[m], alpha=0.22, lw=0
            )
    ax.axhline(0, color="black", lw=0.6)
    ax.axvline(30, color="#555555", lw=0.8, ls=(0, (4, 2)))
    ax.set_title(s, fontsize=9)
    ax.set_xlabel("True RUL (cycles)")
    ax.set_xticks([0, 40, 80, 120])
    ax.grid(lw=0.3, alpha=0.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
axs[0].set_ylabel("Mean signed error\n(cycles, > 0 = late)")
h, l = axs[0].get_legend_handles_labels()
from matplotlib.patches import Patch

h += [Patch(color="#d9a400", alpha=0.35), Patch(color="#1f4e79", alpha=0.35)]
l += ["95% band, linear", "95% band, GB"]
fig.legend(h, l, loc="upper center", ncol=3, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 1.17))
plt.tight_layout(w_pad=0.5)
plt.savefig("results/figures/fig2_trajectory_error.png", bbox_inches="tight")
plt.close()
