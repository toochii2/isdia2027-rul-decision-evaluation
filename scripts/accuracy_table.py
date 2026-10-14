"""Official-test accuracy of every model (Table 1, Supplementary Table S1).

Fold-model MAE/RMSE/NASA (mean and SD over the five fold models), fold-averaged prediction
accuracy and nested out-of-fold RMSE.

Usage:
    python scripts/accuracy_table.py        # writes results/accuracy_table.json
"""

import json, numpy as np

D = json.load(open("results/decision_nested.json"))
out = {}


def nasa(e):
    return float(np.sum(np.where(e < 0, np.exp(-e / 13) - 1, np.exp(e / 10) - 1)))


for s in ("FD001", "FD002", "FD003", "FD004"):
    R = json.load(open(f"results/cv_{s}.json"))
    C = json.load(open(f"results/cnn_{s}.json"))
    yt = np.array(R["yt"], float)
    out[s] = {}
    preds = {m: np.array(R["test_preds"][m]) for m in ("LR", "Ridge", "RF", "GB")}
    preds["CNN"] = np.array(C["test_preds"])
    assert C["yt"] == R["yt"]
    for m, P in preds.items():
        E = P - yt
        f = dict(MAE=np.abs(E).mean(1), RMSE=np.sqrt((E**2).mean(1)), NASA=np.array([nasa(e) for e in E]))
        ens = P.mean(0) - yt
        out[s][m] = {k: [float(v.mean()), float(v.std(ddof=1))] for k, v in f.items()}
        out[s][m]["fold_RMSE"] = [float(x) for x in f["RMSE"]]
        out[s][m]["ens"] = dict(
            MAE=float(np.abs(ens).mean()), RMSE=float(np.sqrt((ens**2).mean())), NASA=nasa(ens)
        )
        out[s][m]["oof_RMSE"] = float(D[s]["oof_rmse"][m])
json.dump(out, open("results/accuracy_table.json", "w"), indent=1)
for s in out:
    print("==", s)
    for m, v in out[s].items():
        print(
            f"  {m:5s} MAE {v['MAE'][0]:.2f}±{v['MAE'][1]:.2f} RMSE {v['RMSE'][0]:.2f}±{v['RMSE'][1]:.2f} NASA {v['NASA'][0]:.0f}±{v['NASA'][1]:.0f} | ens RMSE {v['ens']['RMSE']:.2f} | OOF {v['oof_RMSE']:.2f} | folds {[round(x,2) for x in v['fold_RMSE']]}"
        )
