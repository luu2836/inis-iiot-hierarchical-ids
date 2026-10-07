import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from imblearn.over_sampling import SMOTE
from config import EDGE_ML, TON_NET, CIC, UNSW
from baseline import prep, ID_COLS
from nn_hier import signed_log1p, fixed_partition
from search_partition import sort_classes
from tune_nn import Net

SEEDS = [1, 2, 3, 4, 5]
SMOTE_TARGET = 5000
METHODS = ["flat", "flat+weight", "flat+SMOTE", "flat+focal", "hier", "hier+weight"]


def metrics(yte, pred, cls):
    per = f1_score(yte, pred, average=None, labels=cls, zero_division=0)
    return (accuracy_score(yte, pred), f1_score(yte, pred, average="macro", zero_division=0),
            f1_score(yte, pred, average="weighted", zero_division=0), float(np.min(per)))


def focal_mlp(Xtr, ytr, Xte, ncls, seed, gamma=2.0, epochs=80):
    torch.manual_seed(seed)
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


def level_model(X, ypos, cls_arr, j, i, seed, weight):
    mask = ypos >= j
    Xk = X[mask]; p = ypos[mask]
    labs = np.full(len(p), "Other", dtype=object)
    sel = (p >= j) & (p <= i)
    labs[sel] = cls_arr[p[sel]]
    m = HistGradientBoostingClassifier(random_state=seed, max_iter=200)
    if weight:
        cnt = pd.Series(labs).value_counts()
        w = np.array([len(labs) / (len(cnt) * cnt[l]) for l in labs])
        m.fit(Xk, labs, sample_weight=w)
    else:
        m.fit(Xk, labs)
    return m


def cascade(Xtr, ypos, cls_arr, cls, Xte, seed, weight):
    intervals = fixed_partition(len(cls), 4)
    pred = np.array([None] * len(Xte), dtype=object); rem = np.ones(len(Xte), dtype=bool); ip = np.arange(len(Xte))
    for (j, i) in intervals:
        if not rem.any():
            break
        m = level_model(Xtr, ypos, cls_arr, j, i, seed, weight)
        ridx = ip[rem]; out = m.predict(Xte.iloc[ridx])
        for a, pp in enumerate(out):
            if pp != "Other":
                pred[ridx[a]] = pp; rem[ridx[a]] = False
    if rem.any():
        pred[rem] = cls[-1]
    return pred


def run_ds(name, path, label_col, dedup):
    print("=" * 78); print(name, "| seeds", SEEDS); print("=" * 78)
    df = pd.read_csv(path, low_memory=False)
    if dedup:
        feat = [c for c in df.columns if c != label_col]
        df = df.drop_duplicates(subset=feat).reset_index(drop=True)
    X, y = prep(df, label_col, ID_COLS)
    res = {m: [] for m in METHODS}
    for s in SEEDS:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=s)
        Xtr = Xtr.reset_index(drop=True); Xte = Xte.reset_index(drop=True)
        ytr = ytr.reset_index(drop=True); yte = yte.reset_index(drop=True)
        cls, _ = sort_classes(ytr); C = len(cls)
        cls_arr = np.array(cls, dtype=object)
        ypos = pd.Series(ytr).map({c: k for k, c in enumerate(cls)}).values

        res["flat"].append(metrics(yte, HistGradientBoostingClassifier(random_state=s, max_iter=200).fit(Xtr, ytr).predict(Xte), cls))
        cnt = ytr.value_counts(); w = ytr.map(lambda c: len(ytr) / (C * cnt[c])).values
        res["flat+weight"].append(metrics(yte, HistGradientBoostingClassifier(random_state=s, max_iter=200).fit(Xtr, ytr, sample_weight=w).predict(Xte), cls))
        strat = {c: SMOTE_TARGET for c in cnt.index if cnt[c] < SMOTE_TARGET}
        if strat:
            Xr, yr = SMOTE(sampling_strategy=strat, k_neighbors=1, random_state=s).fit_resample(Xtr.values, ytr.values)
            res["flat+SMOTE"].append(metrics(yte, HistGradientBoostingClassifier(random_state=s, max_iter=200).fit(Xr, yr).predict(Xte), cls))
        else:
            res["flat+SMOTE"].append(res["flat"][-1])
        a = signed_log1p(Xtr.values.astype(float)); b = signed_log1p(Xte.values.astype(float))
        mu, sd = a.mean(0), a.std(0); sd[sd == 0] = 1
        Xs = ((a - mu) / sd).astype(np.float32); Xts = ((b - mu) / sd).astype(np.float32)
        pm = {c: k for k, c in enumerate(cls)}
        out = focal_mlp(Xs, ytr.map(pm).values, Xts, C, s)
        res["flat+focal"].append(metrics(yte, np.array([cls[v] for v in out], dtype=object), cls))
        res["hier"].append(metrics(yte, cascade(Xtr, ypos, cls_arr, cls, Xte, s, False), cls))
        res["hier+weight"].append(metrics(yte, cascade(Xtr, ypos, cls_arr, cls, Xte, s, True), cls))
        print("  seed {} done".format(s))

    print("\n{:<14} {:>16} {:>17} {:>16}".format("method", "acc", "macro-F1", "worst-class-F1"))
    for m in METHODS:
        a = np.array(res[m])
        print("{:<14} {:.4f}+-{:.4f}   {:.4f}+-{:.4f}    {:.4f}+-{:.4f}".format(
            m, a[:, 0].mean(), a[:, 0].std(), a[:, 1].mean(), a[:, 1].std(), a[:, 3].mean(), a[:, 3].std()))
    print("\nsignificance on macro-F1 vs flat (paired, n=%d):" % len(SEEDS))
    base = np.array(res["flat"])[:, 1]
    for m in METHODS[1:]:
        a = np.array(res[m])[:, 1]
        try:
            _, pv = stats.ttest_rel(a, base)
        except Exception:
            pv = float("nan")
        print("  {:<14} diff={:+.4f}  t-test p={:.4f}".format(m, (a - base).mean(), pv))
    print()


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "unsw"
    if which == "edge":
        run_ds("Edge-IIoTset(ML)", EDGE_ML, "Attack_type", False)
    elif which == "ton":
        run_ds("ToN-IoT(network)", TON_NET, "type", False)
    elif which == "cic":
        run_ds("CIC-IDS2017(dedup)", CIC, "Label", True)
    elif which == "unsw":
        run_ds("UNSW-NB15", UNSW, "attack_cat", False)
