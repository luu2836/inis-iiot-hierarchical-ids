import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from config import EDGE_ML, TON_NET, CIC, SEED
from baseline import prep, ID_COLS

torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = "cpu"


class LevelNet(nn.Module):
    def __init__(self, d, ncls, hidden=(256, 128, 64), p=0.3):
        super().__init__()
        self.gate = nn.Linear(d, d)
        layers = []
        prev = d
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(p)]
            prev = h
        layers += [nn.Linear(prev, ncls)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        g = torch.sigmoid(self.gate(x))
        return self.mlp(x * g)


def signed_log1p(X):
    return np.sign(X) * np.log1p(np.abs(X))


def train_level(X, y, ncls, epochs=60, patience=8, batch=512, lr=1e-3):
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.15, stratify=y, random_state=SEED)
    xt = torch.tensor(Xtr, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.long)
    xv = torch.tensor(Xva, dtype=torch.float32)
    yv = torch.tensor(yva, dtype=torch.long)
    cnt = np.bincount(ytr, minlength=ncls).astype(np.float32)
    w = len(ytr) / (ncls * np.maximum(cnt, 1))
    crit = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32))
    model = LevelNet(X.shape[1], ncls).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(xt)
    best, best_state, wait = 1e9, None, 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss = crit(model(xt[idx]), yt[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = crit(model(xv), yv).item()
        if vl < best - 1e-4:
            best, wait = vl, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model


def load(path, label_col, dedup=False):
    df = pd.read_csv(path, low_memory=False)
    if dedup:
        feat = [c for c in df.columns if c != label_col]
        df = df.drop_duplicates(subset=feat).reset_index(drop=True)
    X, y = prep(df, label_col, ID_COLS)
    return X, y


def fixed_partition(C, k=4):
    base, rem, idx, out = C // k, C % k, 0, []
    for lv in range(k):
        sz = base + (1 if lv < rem else 0)
        out.append((idx, idx + sz - 1))
        idx += sz
    return out


def run(name, path, label_col, dedup=False):
    print("=" * 78)
    print(name)
    print("=" * 78)
    X, y = load(path, label_col, dedup)
    Xtr_all, Xte, ytr_all, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=SEED)
    Xtr_all = Xtr_all.reset_index(drop=True); Xte = Xte.reset_index(drop=True)
    ytr_all = ytr_all.reset_index(drop=True); yte = yte.reset_index(drop=True)
    vc = ytr_all.value_counts()
    cls = vc.index.tolist()
    C = len(cls)
    pos_map = {c: k for k, c in enumerate(cls)}
    ptr = ytr_all.map(pos_map).values

    # preprocessing: signed log1p + standardize (fit on train)
    ptr_all = signed_log1p(Xtr_all.values.astype(np.float64))
    pte = signed_log1p(Xte.values.astype(np.float64))
    mu, sd = ptr_all.mean(0), ptr_all.std(0)
    sd[sd == 0] = 1
    Xtr_s = ((ptr_all - mu) / sd).astype(np.float32)
    Xte_s = ((pte - mu) / sd).astype(np.float32)
    Xte_s = Xte_s  # numpy
    intervals = fixed_partition(C, 4)

    models = []
    total_params = 0
    for (j, i) in intervals:
        mask = ptr >= j
        Xl = Xtr_s[mask]
        pl = ptr[mask]
        tgt = pl <= i
        yl = np.where(tgt, pl - j, i - j + 1).astype(np.int64)
        ncls = (i - j + 1) + 1
        m = train_level(Xl, yl, ncls)
        models.append((m, j, i, i - j + 1))
        total_params += sum(p.numel() for p in m.parameters())
        print("  level [{}-{}] classes={} train_samples={} params={}".format(
            j, i, i - j + 1, len(Xl), sum(p.numel() for p in m.parameters())))

    # cascade inference
    n = len(Xte_s)
    pred = np.array([None] * n, dtype=object)
    rem = np.ones(n, dtype=bool)
    ipos = np.arange(n)
    t0 = time.perf_counter()
    with torch.no_grad():
        for (m, j, i, other_idx) in models:
            if not rem.any():
                break
            ridx = ipos[rem]
            out = m(torch.tensor(Xte_s[ridx])).argmax(1).numpy()
            local = np.where(out == other_idx, -1, out)
            assigned = local >= 0
            sel = ridx[assigned]
            pred[sel] = [cls[j + int(v)] for v in local[assigned]]
            rem[sel] = False
    if rem.any():
        pred[rem] = cls[-1]
    lat = (time.perf_counter() - t0) / n * 1000
    acc = accuracy_score(yte, pred)
    f1m = f1_score(yte, pred, average="macro", zero_division=0)
    f1w = f1_score(yte, pred, average="weighted", zero_division=0)
    per = f1_score(yte, pred, average=None, labels=cls, zero_division=0)
    print("[NN-Hier] acc={:.4f} macro-F1={:.4f} weighted-F1={:.4f} worst={:.4f}".format(
        acc, f1m, f1w, float(np.min(per))))
    print("  total params={:,}  latency={:.4f} ms/sample".format(total_params, lat))
    print()


if __name__ == "__main__":
    run("Edge-IIoTset(ML)", EDGE_ML, "Attack_type")
    run("ToN-IoT(network)", TON_NET, "type")
    run("CIC-IDS2017(dedup)", CIC, "Label", dedup=True)
