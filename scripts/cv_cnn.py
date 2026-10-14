"""1D-CNN reference model (architecture after Li et al., 2018) under the protocol of cv_tabular.py.

Same retained sensors, regime scaling, RUL cap, unshuffled engine-grouped 5-fold partition and
fold seeds 42-46. Inputs are standardised 30-cycle windows; the first 29 windows of each engine
repeat its first observation. Saves out-of-fold trajectory predictions and fold test predictions.

Usage:
    python scripts/cv_cnn.py DATA_DIR SUBSET OUT_JSON [EPOCHS=150]
"""

import sys, json, time, numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.model_selection import GroupKFold
from sklearn.cluster import KMeans

torch.set_num_threads(2)
D, SUB, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
EPOCHS = int(sys.argv[4]) if len(sys.argv) > 4 else 150
CAP, W = 125, 30
cols = ["unit", "cycle", "setting_1", "setting_2", "setting_3"] + [f"sensor_{i}" for i in range(1, 22)]
tr = pd.read_csv(f"{D}/train_{SUB}.txt", sep=r"\s+", header=None, names=cols)
te = pd.read_csv(f"{D}/test_{SUB}.txt", sep=r"\s+", header=None, names=cols)
rul = pd.read_csv(f"{D}/RUL_{SUB}.txt", sep=r"\s+", header=None, names=["RUL"]).RUL.values
tr["RUL"] = tr.groupby("unit").cycle.transform("max") - tr.cycle
tr["y"] = tr.RUL.clip(upper=CAP)
sens = [f"sensor_{i}" for i in range(1, 22)]
if SUB in ("FD002", "FD004"):
    km = KMeans(6, n_init=10, random_state=0).fit(tr[["setting_1", "setting_2", "setting_3"]].round(2))
    tr["oc"] = km.predict(tr[["setting_1", "setting_2", "setting_3"]].round(2))
    te["oc"] = km.predict(te[["setting_1", "setting_2", "setting_3"]].round(2))
    keep = [s for s in sens if tr.groupby("oc")[sens].std().mean()[s] > 0.002]
    mu = tr.groupby("oc")[keep].mean()
    sd = tr.groupby("oc")[keep].std().replace(0, 1)
    for df in (tr, te):
        df[keep] = (df[keep] - mu.loc[df.oc].values) / sd.loc[df.oc].values
    scale_in_fold = False
else:
    keep = [s for s in sens if tr[s].std() > 0.002]
    scale_in_fold = True


def windows(df):
    X, idx = [], []
    for u, g in df.groupby("unit", sort=False):
        a = g[keep].values.astype(np.float32)
        pad = np.vstack([np.repeat(a[:1], W - 1, 0), a])
        X.append(np.stack([pad[i : i + W] for i in range(len(a))]))
        idx.append(g.index.values)
    return np.concatenate(X), np.concatenate(idx)


Xtr, itr = windows(tr)
order = np.argsort(itr)
Xtr = Xtr[order]  # row-aligned with tr
Xte, ite = windows(te)
Xte = Xte[np.argsort(ite)]
last = te.groupby("unit").tail(1).index.values
yt = np.minimum(rul, CAP)


class Net(nn.Module):
    def __init__(s, F):
        super().__init__()
        L = []
        c = F
        for _ in range(4):
            L += [nn.Conv1d(c, 10, 10, padding="same"), nn.Tanh()]
            c = 10
        L += [
            nn.Conv1d(10, 1, 3, padding="same"),
            nn.Tanh(),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(W, 100),
            nn.Tanh(),
            nn.Linear(100, 1),
        ]
        s.f = nn.Sequential(*L)

    def forward(s, x):
        return s.f(x.transpose(1, 2)).squeeze(1)


def fit_predict(Xa, ya, preds, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    m = Net(Xa.shape[2])
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    Xa = torch.tensor(Xa)
    ya = torch.tensor(ya / CAP, dtype=torch.float32)  # target scaled to [0, 1]
    n = len(Xa)
    g = torch.Generator().manual_seed(seed)
    for ep in range(EPOCHS):
        if ep == int(EPOCHS * 0.75):
            for pg in opt.param_groups:
                pg["lr"] = 1e-4
        m.train()
        perm = torch.randperm(n, generator=g)
        for b in range(0, n, 512):
            j = perm[b : b + 512]
            opt.zero_grad()
            loss = ((m(Xa[j]) - ya[j]) ** 2).mean()
            loss.backward()
            opt.step()
    m.eval()
    out = []
    with torch.no_grad():
        for P in preds:
            out.append(
                CAP
                * np.concatenate([m(torch.tensor(P[i : i + 4096])).numpy() for i in range(0, len(P), 4096)])
            )
    return out


oof = np.full(len(tr), np.nan)
testp, testall, folds_units = [], [], []
t0 = time.time()
for k, (a, b) in enumerate(GroupKFold(n_splits=5).split(Xtr, tr.y, groups=tr.unit)):
    folds_units.append(sorted(set(int(u) for u in tr.unit.iloc[b])))
    Xa, Xb, Xt = Xtr[a], Xtr[b], Xte
    if scale_in_fold:
        mu_, sd_ = Xa[:, -1, :].mean(0), Xa[:, -1, :].std(0) + 1e-6
        Xa, Xb, Xt = (Xa - mu_) / sd_, (Xb - mu_) / sd_, (Xt - mu_) / sd_
    pb, pt = fit_predict(
        Xa.astype(np.float32), tr.y.values[a], [Xb.astype(np.float32), Xt.astype(np.float32)], 42 + k
    )
    oof[b] = pb
    testp.append(np.round(pt[np.searchsorted(np.arange(len(te)), last)], 3).tolist())
    testall.append(np.round(pt, 3).tolist())
    r = np.sqrt(np.mean((pt[last] - yt) ** 2))
    ro = np.sqrt(np.mean((pb - tr.y.values[b]) ** 2))
    print(
        SUB,
        "fold",
        k,
        "test RMSE",
        round(r, 2),
        "OOF RMSE",
        round(ro, 2),
        round(time.time() - t0),
        "s",
        flush=True,
    )
json.dump(
    {
        "sub": SUB,
        "folds": folds_units,
        "yt": yt.tolist(),
        "test_preds": testp,
        "test_all": testall,
        "oof": np.round(oof, 3).tolist(),
        "tr_unit": tr.unit.tolist(),
        "tr_rul": tr.RUL.tolist(),
        "tr_y": tr.y.tolist(),
        "epochs": EPOCHS,
    },
    open(OUT, "w"),
)
print("done", SUB, round(time.time() - t0), "s")
