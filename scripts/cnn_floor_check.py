"""CNN near-failure floor check (Supplementary Sect. S2, Table S3).

Fits one CNN on folds 1-4 of a subset (seed 42) and predicts every training engine, so that
in-sample (folds 1-4) and held-out (fold 0) predictions near failure can be compared.
The dropout rate before the dense layer is read from the DROP environment variable.

Usage:
    python scripts/cnn_floor_check.py DATA_DIR SUBSET OUT_JSON [EPOCHS=150]
    DROP=0 python scripts/cnn_floor_check.py DATA_DIR FD001 results/cnn_floor_drop0.json
"""

import sys, json, time, itertools, numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.model_selection import GroupKFold
from sklearn.cluster import KMeans

torch.set_num_threads(1)
D, SUB, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
EPOCHS = int(sys.argv[4]) if len(sys.argv) > 4 else 150
CAP, W = 125, 30
cols = ["unit", "cycle", "setting_1", "setting_2", "setting_3"] + [f"sensor_{i}" for i in range(1, 22)]
tr = pd.read_csv(f"{D}/train_{SUB}.txt", sep=r"\s+", header=None, names=cols)
tr["RUL"] = tr.groupby("unit").cycle.transform("max") - tr.cycle
tr["y"] = tr.RUL.clip(upper=CAP)
sens = [f"sensor_{i}" for i in range(1, 22)]
multi = SUB in ("FD002", "FD004")
folds = [np.unique(tr.unit.values[b]) for _, b in GroupKFold(n_splits=5).split(tr, tr.y, groups=tr.unit)]
fold_of = {u: k for k, f in enumerate(folds) for u in f}
rowfold = tr.unit.map(fold_of).values


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
            nn.Dropout(float(__import__("os").environ.get("DROP", "0.5"))),
            nn.Linear(W, 100),
            nn.Tanh(),
            nn.Linear(100, 1),
        ]
        s.f = nn.Sequential(*L)

    def forward(s, x):
        return s.f(x.transpose(1, 2)).squeeze(1)


def windows(A, units):
    out = np.empty((len(A), W, A.shape[1]), np.float32)
    for u in np.unique(units):
        ix = np.where(units == u)[0]
        a = A[ix]
        pad = np.vstack([np.repeat(a[:1], W - 1, 0), a])
        out[ix] = np.stack([pad[i : i + W] for i in range(len(a))])
    return out


def job(excl, seed):
    t0 = time.time()
    tu = np.concatenate([folds[j] for j in range(5) if j not in excl])
    df = tr.copy()
    m = df.unit.isin(tu).values
    if multi:
        S = df[["setting_1", "setting_2", "setting_3"]].round(2)
        km = KMeans(6, n_init=10, random_state=0).fit(S[m])
        df["oc"] = km.predict(S)
        sd_w = df[m].groupby("oc")[sens].std().mean()
        keep = [s for s in sens if sd_w[s] > 0.002]
        mu = df[m].groupby("oc")[keep].mean()
        sd = df[m].groupby("oc")[keep].std().replace(0, 1).fillna(1)
        A = ((df[keep] - mu.loc[df.oc].values) / sd.loc[df.oc].values).values
    else:
        keep = [s for s in sens if df.loc[m, s].std() > 0.002]
        A = df[keep].values
        A = (A - A[m].mean(0)) / (A[m].std(0) + 1e-6)
    X = windows(A.astype(np.float32), df.unit.values)
    p = np.isin(rowfold, list(excl))
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = Net(X.shape[2])
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    Xa = torch.tensor(X[m])
    ya = torch.tensor(df.y.values[m] / CAP, dtype=torch.float32)
    n = len(Xa)
    g = torch.Generator().manual_seed(seed)
    for ep in range(EPOCHS):
        if ep == int(EPOCHS * 0.75):
            for pg in opt.param_groups:
                pg["lr"] = 1e-4
        net.train()
        perm = torch.randperm(n, generator=g)
        for b in range(0, n, 512):
            j = perm[b : b + 512]
            opt.zero_grad()
            loss = ((net(Xa[j]) - ya[j]) ** 2).mean()
            loss.backward()
            opt.step()
    net.eval()
    Xp = X
    with torch.no_grad():
        pr = CAP * np.concatenate(
            [net(torch.tensor(Xp[i : i + 4096])).numpy() for i in range(0, len(Xp), 4096)]
        )
    print(SUB, excl, round(time.time() - t0), "s", flush=True)
    return excl, None, pr


excl, idx, pr = job((0,), 42)
json.dump(
    {"sub": SUB, "pred": pr.round(3).tolist(), "heldout": np.isin(rowfold, [0]).tolist()}, open(OUT, "w")
)
print("done", flush=True)
