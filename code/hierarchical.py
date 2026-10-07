import math
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from config import EDGE_ML, TON_NET, CIC, SEED
from baseline import prep, ID_COLS

BASE = HistGradientBoostingClassifier(random_state=SEED, max_iter=200)
LAMBDA = 0.5


def sort_classes(y):
    vc = y.value_counts()
    return vc.index.tolist(), vc.values.tolist()


def block_cost(counts, j, i):
    seg = counts[j:i + 1]
    return math.log2(max(seg) / min(seg))


def adaptive_partition(classes, counts, lam=0.3, kmax=8, gamma=0.03, min_size=2, theta=0.5):
    C = len(classes)
    INF = float("inf")
    Ntot = sum(counts)
    cost = [[0.0] * C for _ in range(C)]
    for j in range(C):
        for i in range(j, C):
            nblk = sum(counts[j:i + 1])
            cost[j][i] = (block_cost(counts, j, i)
                          + gamma * (i - j + 1) ** 2
                          + theta * math.log2(Ntot / nblk))

    f = [[INF] * C for _ in range(kmax + 1)]
    prev = [[-1] * C for _ in range(kmax + 1)]
    for i in range(C):
        if (i + 1) >= min_size:
            f[1][i] = cost[0][i]
    for k in range(2, kmax + 1):
        for i in range(C):
            best = INF
            bj = -1
            for j in range(1, i + 1):
                if (i - j + 1) < min_size:
                    continue
                if f[k - 1][j - 1] == INF:
                    continue
                v = f[k - 1][j - 1] + cost[j][i]
                if v < best:
                    best = v
                    bj = j
            f[k][i] = best
            prev[k][i] = bj

    best_obj, best_k = INF, 1
    for k in range(1, kmax + 1):
        if f[k][C - 1] == INF:
            continue
        obj = f[k][C - 1] + lam * k
        if obj < best_obj:
            best_obj, best_k = obj, k

    cuts = []
    i = C - 1
    k = best_k
    while k >= 1:
        j = prev[k][i]
        start = 0 if j == -1 else j
        cuts.append((start, i))
        i = start - 1
        k -= 1
    cuts = cuts[::-1]
    levels = [[classes[t] for t in range(s, e + 1)] for s, e in cuts]
    return levels, best_k


def fixed_partition(classes, counts, k=4):
    C = len(classes)
    base = C // k
    rem = C % k
    levels = []
    idx = 0
    for lv in range(k):
        sz = base + (1 if lv < rem else 0)
        levels.append(classes[idx:idx + sz])
        idx += sz
    return levels


def build_partition(Xtr, ytr, mode, k_fixed=4):
    classes, counts = sort_classes(ytr)
    if mode == "fixed":
        return fixed_partition(classes, counts, k_fixed)
    if mode == "adaptive":
        levels, _ = adaptive_partition(classes, counts)
        return levels
    raise ValueError(mode)


def train_hier(models_built, Xtr, ytr, levels):
    all_classes = [c for lv in levels for c in lv]
    for k, lv in enumerate(levels):
        deeper = set(c for l in levels[k + 1:] for c in l)
        mask = ytr.isin(lv) | ytr.isin(deeper)
        Xk, yk = Xtr[mask], ytr[mask]
        yk2 = yk.where(yk.isin(lv), "Other")
        m = clone(BASE)
        m.fit(Xk, yk2)
        models_built.append(m)
    return all_classes


def predict_hier(models, levels, Xte):
    n = len(Xte)
    y = np.array([None] * n, dtype=object)
    remaining = np.ones(n, dtype=bool)
    for k, lv in enumerate(levels):
        if not remaining.any():
            break
        pos = np.where(remaining)[0]
        pred = models[k].predict(Xte.iloc[pos])
        for local, p in enumerate(pred):
            if p != "Other":
                y[pos[local]] = p
                remaining[pos[local]] = False
    y[remaining] = levels[-1][0]
    return y


def eval_levels(name, yte, pred, classes):
    acc = accuracy_score(yte, pred)
    f1m = f1_score(yte, pred, average="macro", zero_division=0)
    f1w = f1_score(yte, pred, average="weighted", zero_division=0)
    per = f1_score(yte, pred, average=None, labels=classes, zero_division=0)
    worst = classes[int(np.argmin(per))]
    print("[{}] acc={:.4f} macro-F1={:.4f} weighted-F1={:.4f} | worst-class F1={:.4f} ({})".format(
        name, acc, f1m, f1w, float(np.min(per)), worst))
    return acc, f1m, f1w, per


def run(name, path, label_col):
    print("=" * 78)
    print(name)
    print("=" * 78)
    df = pd.read_csv(path, low_memory=False)
    X, y = prep(df, label_col, ID_COLS)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=SEED)
    Xtr = Xtr.reset_index(drop=True)
    Xte = Xte.reset_index(drop=True)
    ytr = ytr.reset_index(drop=True)
    yte = yte.reset_index(drop=True)
    classes, counts = sort_classes(ytr)

    flat = clone(BASE).fit(Xtr, ytr)
    p_flat = flat.predict(Xte)
    eval_levels("Flat", yte, p_flat, classes)

    for mode, tag in [("fixed", "Fixed-4"), ("adaptive", "Adaptive")]:
        levels = build_partition(Xtr, ytr, mode)
        k = len(levels)
        sizes = [len(lv) for lv in levels]
        print("\n{} partition: K={} level sizes={}".format(tag, k, sizes))
        for lv in levels:
            print("   level:", lv)
        models = []
        train_hier(models, Xtr, ytr, levels)
        pred = predict_hier(models, levels, Xte)
        eval_levels(tag, yte, pred, classes)
    print()


if __name__ == "__main__":
    run("Edge-IIoTset(ML)", EDGE_ML, "Attack_type")
    run("ToN-IoT(network)", TON_NET, "type")
    run("CIC-IDS2017", CIC, "Label")
