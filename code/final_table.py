import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from config import EDGE_ML, TON_NET, CIC, UNSW
from baseline import prep, ID_COLS
from nn_hier import signed_log1p, fixed_partition
from search_partition import build_interval_model, sort_classes
from tune_nn import Net

SEEDS = [1, 2, 3, 4, 5]
HIDDEN = (512, 256, 128)
EPOCHS = 80


def train_nn(X, y, ncls, seed, hidden=HIDDEN, epochs=EPOCHS, bn=True, p=0.3):
    torch.manual_seed(seed)
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.15, stratify=y, random_state=seed)
    xt = torch.tensor(Xtr); yt = torch.tensor(ytr, dtype=torch.long)
    xv = torch.tensor(Xva); yv = torch.tensor(yva, dtype=torch.long)
    crit = nn.CrossEntropyLoss()
    m = Net(X.shape[1], ncls, hidden, p=p, bn=bn)
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


def metrics(yte, pred, cls):
    return (accuracy_score(yte, pred),
            f1_score(yte, pred, average="macro", zero_division=0),
            f1_score(yte, pred, average="weighted", zero_division=0),
            float(np.min(f1_score(yte, pred, average=None, labels=cls, zero_division=0))))


def flat_hist(Xtr, ytr, Xte, seed):
    m = HistGradientBoostingClassifier(random_state=seed, max_iter=200).fit(Xtr, ytr)
    return m.predict(Xte)


def hier_hist(Xtr, ytr, Xte, cls, seed):
    cls_arr = np.array(cls, dtype=object)
    ypos = pd.Series(ytr).map({c: k for k, c in enumerate(cls)}).values
    intervals = fixed_partition(len(cls), 4)
    n = len(Xte); pred = np.array([None] * n, dtype=object); rem = np.ones(n, dtype=bool); ip = np.arange(n)
    for (j, i) in intervals:
        if not rem.any():
            break
        m = build_interval_model(Xtr, ypos, cls_arr, j, i, 200, 10 ** 9)
        ridx = ip[rem]; out = m.predict(Xte.iloc[ridx])
        for a, pp in enumerate(out):
            if pp != "Other":
                pred[ridx[a]] = pp; rem[ridx[a]] = False
    if rem.any():
        pred[rem] = cls[-1]
    return pred


def flat_nn(Xtr_s, ytr, Xte_s, cls, seed):
    pm = {c: k for k, c in enumerate(cls)}
    y = np.array([pm[c] for c in ytr], dtype=np.int64)
    m = train_nn(Xtr_s, y, len(cls), seed)
    with torch.no_grad():
        out = m(torch.tensor(Xte_s)).argmax(1).numpy()
    return np.array([cls[v] for v in out], dtype=object)


def hier_nn(Xtr_s, ytr, Xte_s, cls, seed):
    pm = {c: k for k, c in enumerate(cls)}
    ytr_codes = np.array([pm[c] for c in ytr], dtype=np.int64)
    intervals = fixed_partition(len(cls), 4)
    models = []
    for (j, i) in intervals:
        mask = ytr_codes >= j
        yl = np.where(ytr_codes[mask] <= i, ytr_codes[mask] - j, i - j + 1).astype(np.int64)
        models.append((train_nn(Xtr_s[mask], yl, (i - j + 1) + 1, seed), j, i))
    n = len(Xte_s); pred = np.array([None] * n, dtype=object); rem = np.ones(n, dtype=bool); ip = np.arange(n)
    with torch.no_grad():
        for (m, j, i) in models:
            if not rem.any():
                break
            ridx = ip[rem]; out = m(torch.tensor(Xte_s[ridx])).argmax(1).numpy()
            ok = out <= (i - j); sel = ridx[ok]
            pred[sel] = [cls[j + int(v)] for v in out[ok]]
            rem[sel] = False
    if rem.any():
        pred[rem] = cls[-1]
    return pred


def run(name, path, label_col, dedup):
    print("=" * 78); print(name, "| seeds", SEEDS); print("=" * 78)
    df = pd.read_csv(path, low_memory=False)
    if dedup:
        feat = [c for c in df.columns if c != label_col]
        df = df.drop_duplicates(subset=feat).reset_index(drop=True)
        print("dedup rows:", len(df))
    X, y = prep(df, label_col, ID_COLS)
    res = {m: [] for m in ["flat-HistGB", "hier-HistGB", "flat-NN", "hier-NN"]}
    for s in SEEDS:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=s)
        Xtr = Xtr.reset_index(drop=True); Xte = Xte.reset_index(drop=True)
        ytr = ytr.reset_index(drop=True); yte = yte.reset_index(drop=True)
        cls, _ = sort_classes(ytr)
        a = signed_log1p(Xtr.values.astype(np.float64)); b = signed_log1p(Xte.values.astype(np.float64))
        mu, sd = a.mean(0), a.std(0); sd[sd == 0] = 1
        Xtr_s = ((a - mu) / sd).astype(np.float32); Xte_s = ((b - mu) / sd).astype(np.float32)

        res["flat-HistGB"].append(metrics(yte, flat_hist(Xtr, ytr, Xte, s), cls))
        res["hier-HistGB"].append(metrics(yte, hier_hist(Xtr, ytr, Xte, cls, s), cls))
        res["flat-NN"].append(metrics(yte, flat_nn(Xtr_s, ytr, Xte_s, cls, s), cls))
        res["hier-NN"].append(metrics(yte, hier_nn(Xtr_s, ytr, Xte_s, cls, s), cls))
        print("  seed {} done".format(s))

    print("\n{:<14} {:>16} {:>17} {:>17} {:>16}".format("method", "acc", "macro-F1", "weighted-F1", "worst-class-F1"))
    for m in res:
        a = np.array(res[m])
        print("{:<14} {:.4f}+-{:.4f}   {:.4f}+-{:.4f}    {:.4f}+-{:.4f}    {:.4f}+-{:.4f}".format(
            m, a[:, 0].mean(), a[:, 0].std(), a[:, 1].mean(), a[:, 1].std(),
            a[:, 2].mean(), a[:, 2].std(), a[:, 3].mean(), a[:, 3].std()))

    print("\nsignificance on macro-F1 (paired, n={}):".format(len(SEEDS)))
    for x, yv in [("hier-HistGB", "flat-HistGB"), ("hier-NN", "flat-NN"), ("hier-HistGB", "hier-NN")]:
        a = np.array(res[x])[:, 1]; b = np.array(res[yv])[:, 1]
        try:
            _, pv = stats.ttest_rel(a, b)
        except Exception:
            pv = float("nan")
        try:
            _, wp = stats.wilcoxon(a, b)
        except Exception:
            wp = float("nan")
        print("  {:<12} vs {:<12} diff={:+.4f}  t-test p={:.4f}  wilcoxon p={:.4f}".format(
            x, yv, (a - b).mean(), pv, wp))
    print()


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "edge"
    if which == "edge":
        run("Edge-IIoTset(ML)", EDGE_ML, "Attack_type", False)
    elif which == "ton":
        run("ToN-IoT(network)", TON_NET, "type", False)
    elif which == "cic":
        run("CIC-IDS2017(dedup)", CIC, "Label", True)
    elif which == "unsw":
        run("UNSW-NB15", UNSW, "attack_cat", False)
