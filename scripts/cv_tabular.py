"""Engine-grouped 5-fold cross-validation of the four tabular regressors (LR, Ridge, RF, GB).

For each fold, the model is trained on four folds of training engines and used to predict
(i) the official test set (fold models, later averaged) and (ii) the full run-to-failure
trajectories of the held-out training engines (out-of-fold predictions).

Usage:
    python scripts/cv_tabular.py DATA_DIR SUBSET OUT_JSON [CAP=125] [PARTITION_SEED]

    PARTITION_SEED omitted -> original deterministic, unshuffled GroupKFold partition (P0).
    PARTITION_SEED given   -> shuffled GroupKFold repeat (P1: 101, P2: 102).
    CAP=150                -> RUL-cap sensitivity analysis.

Note: in this script regime clustering, regime scaling and the variance screen are fitted on
all training engines (the nested pipeline in nested_tabular.py refits them per split).
"""

import sys, json, time, numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.cluster import KMeans

D, SUB, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
CAP = int(sys.argv[4]) if len(sys.argv) > 4 else 125
REP = (
    int(sys.argv[5]) if len(sys.argv) > 5 else None
)  # partition seed for repeated CV (None = original unshuffled partition)
cols = ["unit", "cycle", "setting_1", "setting_2", "setting_3"] + [f"sensor_{i}" for i in range(1, 22)]
tr = pd.read_csv(f"{D}/train_{SUB}.txt", sep=r"\s+", header=None, names=cols)
te = pd.read_csv(f"{D}/test_{SUB}.txt", sep=r"\s+", header=None, names=cols)
rul = pd.read_csv(f"{D}/RUL_{SUB}.txt", sep=r"\s+", header=None, names=["RUL"]).RUL.values
tr["RUL"] = tr.groupby("unit").cycle.transform("max") - tr.cycle
tr["y"] = tr.RUL.clip(upper=CAP)
sens = [f"sensor_{i}" for i in range(1, 22)]
multi = SUB in ("FD002", "FD004")
if multi:  # operating-condition-wise standardisation
    km = KMeans(6, n_init=10, random_state=0).fit(tr[["setting_1", "setting_2", "setting_3"]].round(2))
    tr["oc"] = km.predict(tr[["setting_1", "setting_2", "setting_3"]].round(2))
    te["oc"] = km.predict(te[["setting_1", "setting_2", "setting_3"]].round(2))
    within_sd = tr.groupby("oc")[sens].std().mean()
    keep = [s for s in sens if within_sd[s] > 0.002]
    mu = tr.groupby("oc")[keep].mean()
    sd = tr.groupby("oc")[keep].std().replace(0, 1)
    for df in (tr, te):
        df[keep] = (df[keep] - mu.loc[df.oc].values) / sd.loc[df.oc].values
else:
    keep = [s for s in sens if tr[s].std() > 0.002]


def feats(df):
    out = df[["unit", "cycle"] + keep].copy()
    g = df.groupby("unit")
    for c in keep:
        for w in (5, 10):
            out[f"{c}_rm{w}"] = g[c].transform(lambda s: s.rolling(w, min_periods=1).mean())
            out[f"{c}_rs{w}"] = g[c].transform(lambda s: s.rolling(w, min_periods=1).std()).fillna(0)
    return out


Ftr, Fte = feats(tr), feats(te)
fc = [c for c in Ftr.columns if c not in ("unit", "cycle")]
last = Fte.groupby("unit").tail(1)
yt = np.minimum(rul, CAP)


def make(n, seed):
    return {
        "LR": make_pipeline(StandardScaler(), LinearRegression()),
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "RF": RandomForestRegressor(
            n_estimators=300, max_depth=14, min_samples_leaf=4, random_state=seed, n_jobs=-1
        ),
        "GB": GradientBoostingRegressor(n_estimators=250, learning_rate=0.05, max_depth=3, random_state=seed),
    }[n]


models = ["LR", "Ridge", "RF", "GB"]
gkf = GroupKFold(n_splits=5) if REP is None else GroupKFold(n_splits=5, shuffle=True, random_state=REP)
folds_units = []
oof = {m: np.full(len(tr), np.nan) for m in models}
testp = {m: [] for m in models}
testall = {m: [] for m in models}
t0 = time.time()
for k, (a, b) in enumerate(gkf.split(Ftr, tr.y, groups=tr.unit)):
    folds_units.append(sorted(set(int(u) for u in tr.unit.iloc[b])))
    for m in models:
        mdl = make(m, 42 + k).fit(Ftr.iloc[a][fc], tr.y.iloc[a])
        oof[m][b] = mdl.predict(Ftr.iloc[b][fc])
        testp[m].append(np.round(mdl.predict(last[fc]), 3).tolist())
        testall[m].append(np.round(mdl.predict(Fte[fc]), 3).tolist())
    print(SUB, "fold", k, round(time.time() - t0), "s", flush=True)
res = {
    "sub": SUB,
    "rep": REP,
    "folds": folds_units,
    "keep": keep,
    "n_test": int(len(yt)),
    "yt": yt.tolist(),
    "test_preds": testp,
    "test_all": testall,
    "te_unit": te.unit.tolist(),
    "te_cycle": te.cycle.tolist(),
    "rul_raw": rul.tolist(),
    "oof": {m: np.round(oof[m], 3).tolist() for m in models},
    "tr_unit": tr.unit.tolist(),
    "tr_rul": tr.RUL.tolist(),
    "tr_y": tr.y.tolist(),
}
json.dump(res, open(OUT, "w"))
print("done", SUB, round(time.time() - t0), "s")
