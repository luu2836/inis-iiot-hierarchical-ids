import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from imblearn.over_sampling import SMOTE
from config import EDGE_ML, TON_NET, CIC, UNSW
from baseline import prep, ID_COLS
from nn_hier import signed_log1p, fixed_partition
from search_partition import sort_classes, build_interval_model
from tune_nn import Net

SEED = 1
SMOTE_TARGET = 5000


def metrics(yte, pred, cls):
    per = f1_score(yte, pred, average=None, labels=cls, zero_division=0)
    return (accuracy_score(yte, pred), f1_score(yte, pred, average="macro", zero_division=0),
            f1_score(yte, pred, average="weighted", zero_division=0), float(np.min(per)))


def focal_mlp(Xtr, ytr, Xte, ncls, gamma=2.0, epochs=80):
    torch.manual_seed(SEED)
    y = np.array(ytr, dtype=np.int64)
    xt = torch.tensor(Xtr); yt = torch.tensor(y)
    m = Net(Xtr.shape[1], ncls, (512, 256, 128), p=0.3, bn=True)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    n = len(xt)
    for ep in range(epochs):
        m.train(); perm = torch.randperm(n)
        for i in range(0, n, 512):
            idx = perm[i:i + 512]; opt.zero_grad()
            logits = m(xt[idx]); logp = F.log_softmax(logits, 1)
            pt = logp.gather(1, yt[idx].unsqueeze(1)).exp()
            loss = (-(1 - pt) ** gamma * logp.gather(1, yt[idx].unsqueeze(1))).mean()
            loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        return m(torch.tensor(Xte)).argmax(1).numpy()


def run(name, path, label_col, dedup=False):
    print("=" * 78); print(name); print("=" * 78)
    df = pd.read_csv(path, low_memory=False)
    if dedup:
        feat = [c for c in df.columns if c != label_col]
        df = df.drop_duplicates(subset=feat).reset_index(drop=True)
    X, y = prep(df, label_col, ID_COLS)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=SEED)
    Xtr = Xtr.reset_index(drop=True); Xte = Xte.reset_index(drop=True)
    ytr = ytr.reset_index(drop=True); yte = yte.reset_index(drop=True)
    cls, counts = sort_classes(ytr)
    C = len(cls)
    cls_arr = np.array(cls, dtype=object)
    ypos = pd.Series(ytr).map({c: k for k, c in enumerate(cls)}).values

    clf = HistGradientBoostingClassifier(random_state=SEED, max_iter=200).fit(Xtr, ytr)
    print("[flat-GBDT]        ", " ".join("{}={:.3f}".format(*z) for z in zip(["acc", "macF1", "wF1", "worst"], metrics(yte, clf.predict(Xte), cls))))

    cnt = ytr.value_counts()
    w = ytr.map(lambda c: len(ytr) / (C * cnt[c])).values
    clf = HistGradientBoostingClassifier(random_state=SEED, max_iter=200).fit(Xtr, ytr, sample_weight=w)
    print("[flat-GBDT+weight] ", " ".join("{}={:.3f}".format(*z) for z in zip(["acc", "macF1", "wF1", "worst"], metrics(yte, clf.predict(Xte), cls))))

    strat = {c: SMOTE_TARGET for c in cnt.index if cnt[c] < SMOTE_TARGET}
    if strat:
        sm = SMOTE(sampling_strategy=strat, k_neighbors=1, random_state=SEED)
        Xr, yr = sm.fit_resample(Xtr.values, ytr.values)
        clf = HistGradientBoostingClassifier(random_state=SEED, max_iter=200).fit(Xr, yr)
        print("[flat-GBDT+SMOTE]  ", " ".join("{}={:.3f}".format(*z) for z in zip(["acc", "macF1", "wF1", "worst"], metrics(yte, clf.predict(Xte), cls))))

    a = signed_log1p(Xtr.values.astype(float)); b = signed_log1p(Xte.values.astype(float))
    mu, sd = a.mean(0), a.std(0); sd[sd == 0] = 1
    Xs = ((a - mu) / sd).astype(np.float32); Xts = ((b - mu) / sd).astype(np.float32)
    pm = {c: k for k, c in enumerate(cls)}
    yc = np.array([pm[c] for c in ytr], dtype=np.int64)
    out = focal_mlp(Xs, ytr.map(pm).values, Xts, C)
    pred = np.array([cls[v] for v in out], dtype=object)
    print("[flat-MLP+focal]   ", " ".join("{}={:.3f}".format(*z) for z in zip(["acc", "macF1", "wF1", "worst"], metrics(yte, pred, cls))))

    intervals = fixed_partition(C, 4)
    pred = np.array([None] * len(Xte), dtype=object); rem = np.ones(len(Xte), dtype=bool); ip = np.arange(len(Xte))
    for (j, i) in intervals:
        if not rem.any():
            break
        m = build_interval_model(Xtr, ypos, cls_arr, j, i, 200, 10 ** 9)
        ridx = ip[rem]; o = m.predict(Xte.iloc[ridx])
        for aa, pp in enumerate(o):
            if pp != "Other":
                pred[ridx[aa]] = pp; rem[ridx[aa]] = False
    if rem.any():
        pred[rem] = cls[-1]
    print("[hier-GBDT]        ", " ".join("{}={:.3f}".format(*z) for z in zip(["acc", "macF1", "wF1", "worst"], metrics(yte, pred, cls))))
    print()


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "unsw"
    if which == "edge":
        run("Edge-IIoTset(ML)", EDGE_ML, "Attack_type")
    elif which == "ton":
        run("ToN-IoT(network)", TON_NET, "type")
    elif which == "cic":
        run("CIC-IDS2017(dedup)", CIC, "Label", True)
    elif which == "unsw":
        run("UNSW-NB15", UNSW, "attack_cat")
