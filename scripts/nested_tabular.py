"""Nested engine-grouped validation of the tabular models for the decision lens (Sect. 3.4).
Outer 5-fold GroupKFold (original unshuffled partition). For each outer evaluation fold k:
  - inner predictions on every training engine (folds != k) come from models trained on the
    other three training folds (pair split excluding {k, c}), i.e. an inner 4-fold loop;
  - the evaluated predictor is refitted on all four training folds.
Preprocessing (regime clustering, regime scaling, variance screen) is refitted on the training
engines of every split, so evaluation engines inform neither preprocessing nor the models that
set their margin. Usage: python scripts/nested_tabular.py DATA_DIR SUBSET OUT_JSON [MODELS=LR,Ridge,RF,GB]"""

import sys, json, time, itertools, numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.cluster import KMeans

D, SUB, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
MODELS = sys.argv[4].split(",") if len(sys.argv) > 4 else ["LR", "Ridge", "RF", "GB"]
CAP = 125
cols = ["unit", "cycle", "setting_1", "setting_2", "setting_3"] + [f"sensor_{i}" for i in range(1, 22)]
tr = pd.read_csv(f"{D}/train_{SUB}.txt", sep=r"\s+", header=None, names=cols)
tr["RUL"] = tr.groupby("unit").cycle.transform("max") - tr.cycle
tr["y"] = tr.RUL.clip(upper=CAP)
sens = [f"sensor_{i}" for i in range(1, 22)]
multi = SUB in ("FD002", "FD004")
folds = [np.unique(tr.unit.values[b]) for _, b in GroupKFold(n_splits=5).split(tr, tr.y, groups=tr.unit)]
fold_of = {u: k for k, f in enumerate(folds) for u in f}
rowfold = tr.unit.map(fold_of).values


def prep(train_units):
    """Fit preprocessing on training engines only; return feature matrix for all rows and feature names."""
    df = tr.copy()
    m = df.unit.isin(train_units).values
    if multi:
        S = df[["setting_1", "setting_2", "setting_3"]].round(2)
        km = KMeans(6, n_init=10, random_state=0).fit(S[m])
        df["oc"] = km.predict(S)
        sd_w = df[m].groupby("oc")[sens].std().mean()
        keep = [s for s in sens if sd_w[s] > 0.002]
        mu = df[m].groupby("oc")[keep].mean()
        sd = df[m].groupby("oc")[keep].std().replace(0, 1).fillna(1)
        df[keep] = (df[keep] - mu.loc[df.oc].values) / sd.loc[df.oc].values
    else:
        keep = [s for s in sens if df.loc[m, s].std() > 0.002]
    out = df[keep].copy()
    g = df.groupby("unit")
    for c in keep:
        for w in (5, 10):
            r = g[c].rolling(w, min_periods=1)
            out[f"{c}_rm{w}"] = r.mean().reset_index(level=0, drop=True)
            out[f"{c}_rs{w}"] = r.std().reset_index(level=0, drop=True).fillna(0)
    return out.values.astype(np.float64), keep


def make(n, seed):
    return {
        "LR": make_pipeline(StandardScaler(), LinearRegression()),
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "RF": RandomForestRegressor(
            n_estimators=300, max_depth=14, min_samples_leaf=4, random_state=seed, n_jobs=1
        ),
        "GB": GradientBoostingRegressor(n_estimators=250, learning_rate=0.05, max_depth=3, random_state=seed),
    }[n]


def job(excl, seed):
    """Train on all folds except `excl`; predict rows of the excluded folds."""
    t0 = time.time()
    tu = np.concatenate([folds[j] for j in range(5) if j not in excl])
    X, keep = prep(tu)
    m = np.isin(tr.unit.values, tu)
    p = np.isin(rowfold, list(excl))
    res = {}
    for mn in MODELS:
        mdl = make(mn, seed).fit(X[m], tr.y.values[m])
        res[mn] = mdl.predict(X[p]).round(3).tolist()
    return excl, np.where(p)[0].tolist(), res, keep, round(time.time() - t0)


splits = [((k,), 42 + k) for k in range(5)] + [
    ((k, c), 142 + 10 * k + c) for k, c in itertools.combinations(range(5), 2)
]
t0 = time.time()
outs = Parallel(n_jobs=2, verbose=0)(delayed(job)(e, s) for e, s in splits)
outer = {mn: np.full(len(tr), np.nan) for mn in MODELS}
inner = {k: {mn: np.full(len(tr), np.nan) for mn in MODELS} for k in range(5)}
keeps = {}
for excl, idx, res, keep, dt in outs:
    idx = np.array(idx)
    keeps[str(excl)] = keep
    for mn in MODELS:
        pr = np.array(res[mn])
        if len(excl) == 1:
            outer[mn][idx] = pr
        else:
            k, c = excl
            # rows of fold c predicted by model excluding {k,c} -> inner prediction for outer fold k (and vice versa)
            inner[k][mn][idx[rowfold[idx] == c]] = pr[rowfold[idx] == c]
            inner[c][mn][idx[rowfold[idx] == k]] = pr[rowfold[idx] == k]
    print(SUB, excl, dt, "s", flush=True)
for k in range(5):
    for mn in MODELS:
        assert np.isnan(inner[k][mn][rowfold == k]).all() and not np.isnan(inner[k][mn][rowfold != k]).any()
for mn in MODELS:
    assert not np.isnan(outer[mn]).any()
json.dump(
    {
        "sub": SUB,
        "folds": [f.tolist() for f in folds],
        "tr_unit": tr.unit.tolist(),
        "tr_rul": tr.RUL.tolist(),
        "tr_y": tr.y.tolist(),
        "outer": {mn: v.round(3).tolist() for mn, v in outer.items()},
        "inner": {
            str(k): {mn: [None if np.isnan(x) else round(float(x), 3) for x in v] for mn, v in d.items()}
            for k, d in inner.items()
        },
        "keep": keeps,
    },
    open(OUT, "w"),
)
print("done", SUB, round(time.time() - t0), "s", flush=True)
