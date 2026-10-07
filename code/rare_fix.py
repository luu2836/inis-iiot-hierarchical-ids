import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from config import EDGE_ML, TON_NET, CIC
from baseline import prep, ID_COLS
from nn_hier import signed_log1p, fixed_partition
from search_partition import sort_classes

MAXIT = 200


def build_level(X, ypos, j, i, mode, cls_arr, alpha=1.0, k=3):
    mask = ypos >= j
    Xk = X[mask]; p = ypos[mask]
    labs = np.full(len(p), "Other", dtype=object)
    sel = (p >= j) & (p <= i)
    labs[sel] = cls_arr[p[sel]]
    if mode == "knn":
        return KNeighborsClassifier(n_neighbors=min(k, max(1, len(Xk) // 2)), weights="distance").fit(Xk, labs)
    m = HistGradientBoostingClassifier(random_state=42, max_iter=MAXIT)
    if mode == "plain":
        m.fit(Xk, labs)
    else:
        cnt = pd.Series(labs).value_counts()
        w = np.array([(1.0 / cnt[l]) ** alpha for l in labs])
        w = w / w.mean()
        m.fit(Xk, labs, sample_weight=w)
    return m


def cascade(models, intervals, Xte, cls):
    n = len(Xte)
    pred = np.array([None] * n, dtype=object); rem = np.ones(n, dtype=bool); ip = np.arange(n)
    for (m, j, i) in models:
        if not rem.any():
            break
        ridx = ip[rem]; out = m.predict(Xte[ridx])
        for a, pp in enumerate(out):
            if pp != "Other":
                pred[ridx[a]] = pp; rem[ridx[a]] = False
    if rem.any():
        pred[rem] = cls[-1]
    return pred


def run(name, path, label_col, dedup=False):
    print("=" * 78); print(name); print("=" * 78)
    df = pd.read_csv(path, low_memory=False)
    if dedup:
        feat = [c for c in df.columns if c != label_col]
        df = df.drop_duplicates(subset=feat).reset_index(drop=True)
    X, y = prep(df, label_col, ID_COLS)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=1)
    a = signed_log1p(Xtr.values.astype(float)); b = signed_log1p(Xte.values.astype(float))
    mu, sd = a.mean(0), a.std(0); sd[sd == 0] = 1
    Xtr = ((a - mu) / sd).astype(np.float32); Xte = ((b - mu) / sd).astype(np.float32)
    ytr = ytr.reset_index(drop=True); yte = yte.reset_index(drop=True)
    cls, counts = sort_classes(ytr)
    C = len(cls)
    cls_arr = np.array(cls, dtype=object)
    ypos = pd.Series(ytr).map({c: k for k, c in enumerate(cls)}).values
    intervals = fixed_partition(C, 4)

    flat = HistGradientBoostingClassifier(random_state=1, max_iter=MAXIT).fit(Xtr, ytr)
    pf = flat.predict(Xte)
    per = f1_score(yte, pf, average=None, labels=cls, zero_division=0)
    print("[Flat-Hist]   macro-F1={:.4f} worst={:.4f} acc={:.4f}".format(
        f1_score(yte, pf, average="macro", zero_division=0), float(np.min(per)), accuracy_score(yte, pf)))

    variants = [("plain", "plain"), ("weighted sqrt", "sqrt"), ("weighted inv", "inv"),
                ("knn tail", "knn")]
    for tag, mode in variants:
        models = []
        for (j, i) in intervals:
            is_tail = (i == C - 1)
            if is_tail and mode == "knn":
                m = build_level(Xtr, ypos, j, i, "knn", cls_arr, k=3)
            elif is_tail and mode in ("sqrt", "inv"):
                m = build_level(Xtr, ypos, j, i, "weighted", cls_arr, alpha=(0.5 if mode == "sqrt" else 1.0))
            else:
                m = build_level(Xtr, ypos, j, i, "plain", cls_arr)
            models.append((m, j, i))
        pred = cascade(models, intervals, Xte, cls)
        per = f1_score(yte, pred, average=None, labels=cls, zero_division=0)
        rare = {cls[i]: round(float(per[i]), 3) for i in range(C) if counts[i] < 100}
        print("[Hier {}] macro-F1={:.4f} worst={:.4f} acc={:.4f} | rare: {}".format(
            tag, f1_score(yte, pred, average="macro", zero_division=0),
            float(np.min(per)), accuracy_score(yte, pred), rare))
    print()


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "cic"
    if which == "cic":
        run("CIC-IDS2017(dedup)", CIC, "Label", dedup=True)
    elif which == "edge":
        run("Edge-IIoTset(ML)", EDGE_ML, "Attack_type")
    elif which == "ton":
        run("ToN-IoT(network)", TON_NET, "type")
