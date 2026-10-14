"""Summarise the CNN near-failure floor check (Supplementary Table S3).

Reads the three runs of cnn_floor_check.py on FD001 (dropout 0.5 / 150 epochs, dropout 0,
300 epochs) and reports median predictions at true RUL 0, 15 and 30, the minimum prediction and
RMSE, separately for in-sample (folds 1-4) and held-out (fold 0) engines.

Usage:
    python scripts/cnn_floor_summary.py     # writes results/cnn_floor_summary.json
"""

import json
import numpy as np

R = json.load(open("results/nested_FD001.json"))
rul = np.array(R["tr_rul"])
y = np.array(R["tr_y"])
runs = {
    "base": "results/cnn_floor_FD001.json",
    "dropout0": "results/cnn_floor_drop0_FD001.json",
    "epochs300": "results/cnn_floor_ep300_FD001.json",
}
res = {}
for name, path in runs.items():
    I = json.load(open(path))
    p = np.array(I["pred"])
    held = np.array(I["heldout"])
    res[name] = {}
    for lab, mask in (("in", ~held), ("out", held)):
        d = {str(r): [round(float(np.median(p[mask & (rul == r)])), 1)] for r in (0, 15, 30)}
        d["min"] = round(float(p[mask].min()), 1)
        d["rmse"] = round(float(np.sqrt(np.mean((p[mask] - y[mask]) ** 2))), 2)
        res[name][lab] = d
    print(name, res[name])
json.dump(res, open("results/cnn_floor_summary.json", "w"), indent=1)
