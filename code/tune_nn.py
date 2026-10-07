import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from config import EDGE_ML, TON_NET, SEED
from baseline import prep, ID_COLS
from nn_hier import signed_log1p, fixed_partition

torch.manual_seed(SEED); np.random.seed(SEED)
DEVICE = "cpu"


class Net(nn.Module):
    def __init__(self, d, ncls, hidden, p=0.3, bn=False):
        super().__init__()
        self.gate = nn.Linear(d, d)
        L, prev = [], d
        for h in hidden:
            L.append(nn.Linear(prev, h))
            if bn:
                L.append(nn.BatchNorm1d(h))
            L += [nn.ReLU(), nn.Dropout(p)]
            prev = h
        L.append(nn.Linear(prev, ncls))
        self.mlp = nn.Sequential(*L)

    def forward(self, x):
        return self.mlp(x * torch.sigmoid(self.gate(x)))


def weights(y, ncls, mode):
    if mode == "none":
        return None
    cnt = np.bincount(y, minlength=ncls).astype(np.float32)
    w = len(y) / (ncls * np.maximum(cnt, 1))
    if mode == "sqrt":
        w = np.sqrt(w)
    return torch.tensor(w, dtype=torch.float32)


def train(X, y, ncls, hidden, epochs, mode, bn):
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.15, stratify=y, random_state=SEED)
    xt = torch.tensor(Xtr, dtype=torch.float32); yt = torch.tensor(ytr, dtype=torch.long)
    xv = torch.tensor(Xva, dtype=torch.float32); yv = torch.tensor(yva, dtype=torch.long)
    crit = nn.CrossEntropyLoss(weight=weights(ytr, ncls, mode))
    m = Net(X.shape[1], ncls, hidden, bn=bn)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    n = len(xt); best, wait, state = 1e9, 0, None
    for ep in range(epochs):
        m.train(); perm = torch.randperm(n)
        for i in range(0, n, 512):
            idx = perm[i:i + 512]; opt.zero_grad()
            loss = crit(m(xt[idx]), yt[idx]); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            vl = crit(m(xv), yv).item()
        if vl < best - 1e-4:
            best, wait, state = vl, 0, {k: v.clone() for k, v in m.state_dict().items()}
        else:
            wait += 1
            if wait >= 8:
                break
    m.load_state_dict(state); m.eval()
    return m


def evaluate(name, path, label_col, configs):
    print("=" * 78); print(name); print("=" * 78)
    df = pd.read_csv(path, low_memory=False)
    X, y = prep(df, label_col, ID_COLS)
    Xtr_all, Xte, ytr_all, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=SEED)
    Xtr_all = Xtr_all.reset_index(drop=True); Xte = Xte.reset_index(drop=True)
    ytr_all = ytr_all.reset_index(drop=True); yte = yte.reset_index(drop=True)
    vc = ytr_all.value_counts(); cls = vc.index.tolist(); C = len(cls)
    pm = {c: k for k, c in enumerate(cls)}; p = ytr_all.map(pm).values
    a = signed_log1p(Xtr_all.values.astype(np.float64)); b = signed_log1p(Xte.values.astype(np.float64))
    mu, sd = a.mean(0), a.std(0); sd[sd == 0] = 1
    Xs = ((a - mu) / sd).astype(np.float32); Xts = ((b - mu) / sd).astype(np.float32)
    intervals = fixed_partition(C, 4)
    for cfg in configs:
        models = []
        for (j, i) in intervals:
            mask = p >= j; Xl = Xs[mask]; pl = p[mask]
            yl = np.where(pl <= i, pl - j, i - j + 1).astype(np.int64)
            models.append((train(Xl, yl, (i - j + 1) + 1, cfg["hidden"], cfg["epochs"], cfg["mode"], cfg["bn"]), j, i))
        n = len(Xts); pred = np.array([None] * n, dtype=object); rem = np.ones(n, dtype=bool); ip = np.arange(n)
        with torch.no_grad():
            for (m, j, i) in models:
                if not rem.any():
                    break
                ridx = ip[rem]; out = m(torch.tensor(Xts[ridx])).argmax(1).numpy()
                sel = ridx[out <= (i - j)]
                pred[sel] = [cls[j + int(v)] for v in out[out <= (i - j)]]
                rem[sel] = False
        if rem.any():
            pred[rem] = cls[-1]
        f1m = f1_score(yte, pred, average="macro", zero_division=0)
        per = f1_score(yte, pred, average=None, labels=cls, zero_division=0)
        print("  {:<28} macro-F1={:.4f} worst={:.4f} acc={:.4f}".format(
            cfg["tag"], f1m, float(np.min(per)), accuracy_score(yte, pred)))


if __name__ == "__main__":
    configs = [
        {"tag": "noW h512 bn ep80", "hidden": (512, 256, 128), "epochs": 80, "mode": "none", "bn": True},
        {"tag": "sqrtW h512 bn ep80", "hidden": (512, 256, 128), "epochs": 80, "mode": "sqrt", "bn": True},
        {"tag": "invW h512 bn ep80", "hidden": (512, 256, 128), "epochs": 80, "mode": "inv", "bn": True},
    ]
    evaluate("Edge-IIoTset(ML)", EDGE_ML, "Attack_type", configs)
    evaluate("ToN-IoT(network)", TON_NET, "type", configs)
