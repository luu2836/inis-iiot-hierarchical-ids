import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from config import EDGE_ML, TON_NET, CIC, UNSW
from baseline import prep, ID_COLS
from nn_hier import signed_log1p
from search_partition import sort_classes

SEEDS = [1, 2, 3]
torch.manual_seed(0)


class CNN1D(nn.Module):
    def __init__(self, d, ncls, ch=64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, ch, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(ch, ch, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1))
        self.fc = nn.Linear(ch, ncls)

    def forward(self, x):
        x = x.unsqueeze(1)
        return self.fc(self.features(x).squeeze(-1))


def train_cnn(Xtr, ytr, ncls, seed, epochs=50, batch=512, lr=1e-3):
    torch.manual_seed(seed)
    xt = torch.tensor(Xtr); yt = torch.tensor(ytr, dtype=torch.long)
    m = CNN1D(Xtr.shape[1], ncls)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    n = len(xt)
    for ep in range(epochs):
        m.train(); perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]; opt.zero_grad()
            loss = crit(m(xt[idx]), yt[idx]); loss.backward(); opt.step()
    m.eval()
    return m


def run(name, path, label_col, dedup=False):
    print("=" * 70); print(name); print("=" * 70)
    df = pd.read_csv(path, low_memory=False)
    if dedup:
        feat = [c for c in df.columns if c != label_col]
        df = df.drop_duplicates(subset=feat).reset_index(drop=True)
    X, y = prep(df, label_col, ID_COLS)
    m, w, a = [], [], []
    params = 0
    for s in SEEDS:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=s)
        Xtr = Xtr.reset_index(drop=True); Xte = Xte.reset_index(drop=True)
        ytr = ytr.reset_index(drop=True); yte = yte.reset_index(drop=True)
        cls, _ = sort_classes(ytr)
        a1 = signed_log1p(Xtr.values.astype(float)); b1 = signed_log1p(Xte.values.astype(float))
        mu, sd = a1.mean(0), a1.std(0); sd[sd == 0] = 1
        Xs = ((a1 - mu) / sd).astype(np.float32); Xts = ((b1 - mu) / sd).astype(np.float32)
        pm = {c: k for k, c in enumerate(cls)}
        yc = np.array([pm[c] for c in ytr])
        model = train_cnn(Xs, yc, len(cls), s)
        params = sum(p.numel() for p in model.parameters())
        with torch.no_grad():
            out = model(torch.tensor(Xts)).argmax(1).numpy()
        pred = np.array([cls[v] for v in out], dtype=object)
        m.append(f1_score(yte, pred, average="macro", zero_division=0))
        w.append(float(np.min(f1_score(yte, pred, average=None, labels=cls, zero_division=0))))
        a.append(accuracy_score(yte, pred))
        print("  seed {} done".format(s))
    print("[flat-CNN] acc={:.4f} macro-F1={:.4f} worst={:.4f} params={}".format(
        np.mean(a), np.mean(m), np.mean(w), params))
    print()


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "unsw"
    if which == "edge":
        run("Edge-IIoTset", EDGE_ML, "Attack_type")
    elif which == "ton":
        run("ToN-IoT", TON_NET, "type")
    elif which == "unsw":
        run("UNSW-NB15", UNSW, "attack_cat")
    elif which == "cic":
        run("CIC-IDS2017", CIC, "Label", True)
