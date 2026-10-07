import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from config import EDGE_ML, TON_NET, CIC, SEED
from baseline import prep, ID_COLS

SEARCH_MAX_ITER = 100
FINAL_MAX_ITER = 200
PER_CLASS_CAP = 10000


def sort_classes(y):
    vc = y.value_counts()
    return vc.index.tolist(), vc.values.tolist()


def subsample_by_label(labs, cap, seed):
    idx = np.arange(len(labs))
    keep = []
    df = pd.DataFrame({"i": idx, "l": labs})
    for l, g in df.groupby("l"):
        if len(g) > cap:
            keep.append(g.sample(n=cap, random_state=seed)["i"].values)
        else:
            keep.append(g["i"].values)
    return np.concatenate(keep)


def build_interval_model(X, ypos, cls_arr, j, i, max_iter, cap):
    mask = ypos >= j
    Xk = X[mask]
    p = ypos[mask]
    labs = np.full(len(p), "Other", dtype=object)
    sel = (p >= j) & (p <= i)
    labs[sel] = cls_arr[p[sel]]
    idx = subsample_by_label(labs, cap, SEED)
    m = HistGradientBoostingClassifier(random_state=SEED, max_iter=max_iter)
    m.fit(Xk.iloc[idx], labs[idx])
    return m


def eval_partition(intervals, Xval, yval, cache, cls, cls_arr, max_iter, cap, Xtr, ypos):
    n = len(Xval)
    pred = np.full(n, None, dtype=object)
    rem = np.ones(n, dtype=bool)
    pos = np.arange(n)
    for (j, i) in intervals:
        if not rem.any():
            break
        key = (j, i)
        if key not in cache:
            cache[key] = build_interval_model(Xtr, ypos, cls_arr, j, i, max_iter, cap)
        m = cache[key]
        ridx = pos[rem]
        pr = m.predict(Xval.iloc[ridx])
        for a, pp in enumerate(pr):
            if pp != "Other":
                pred[ridx[a]] = pp
                rem[ridx[a]] = False
    if rem.any():
        pred[rem] = cls[-1]
    return f1_score(yval, pred, average="macro", zero_division=0), pred


def greedy_merge(Xtr, ypos, cls, cls_arr, Xval, yval, max_iter, cap):
    cache = {}
    C = len(cls)
    intervals = [(p, p) for p in range(C)]
    best_score, best = eval_partition(intervals, Xval, yval, cache, cls, cls_arr, max_iter, cap, Xtr, ypos)
    print("  start K={} val macro-F1={:.4f}".format(C, best_score))
    while len(intervals) > 1:
        cands = []
        for t in range(len(intervals) - 1):
            merged = intervals[:t] + [(intervals[t][0], intervals[t + 1][1])] + intervals[t + 2:]
            s, _ = eval_partition(merged, Xval, yval, cache, cls, cls_arr, max_iter, cap, Xtr, ypos)
            cands.append((s, merged))
        s, merged = max(cands, key=lambda x: x[0])
        intervals = merged
        print("  K={} val macro-F1={:.4f}  levels={}".format(len(intervals), s, merged))
        if s > best_score:
            best_score, best = s, merged[:]
    return best, best_score, len(cache)


def final_eval(name, Xtr, ytr, Xte, yte, intervals, cls):
    models = []
    cls_arr = np.array(cls, dtype=object)
    ypos = pd.Series(ytr).map({c: k for k, c in enumerate(cls)}).values
    for (j, i) in intervals:
        models.append(build_interval_model(Xtr, ypos, cls_arr, j, i, FINAL_MAX_ITER, 10 ** 9))
    n = len(Xte)
    pred = np.full(n, None, dtype=object)
    rem = np.ones(n, dtype=bool)
    pos = np.arange(n)
    for (j, i), m in zip(intervals, models):
        if not rem.any():
            break
        ridx = pos[rem]
        pr = m.predict(Xte.iloc[ridx])
        for a, pp in enumerate(pr):
            if pp != "Other":
                pred[ridx[a]] = pp
                rem[ridx[a]] = False
    if rem.any():
        pred[rem] = cls[-1]
    acc = accuracy_score(yte, pred)
    f1m = f1_score(yte, pred, average="macro", zero_division=0)
    f1w = f1_score(yte, pred, average="weighted", zero_division=0)
    print("[Searched] acc={:.4f} macro-F1={:.4f} weighted-F1={:.4f}  K={}".format(acc, f1m, f1w, len(intervals)))
    return acc, f1m, f1w


def run(name, path, label_col):
    print("=" * 78)
    print(name)
    print("=" * 78)
    df = pd.read_csv(path, low_memory=False)
    X, y = prep(df, label_col, ID_COLS)
    Xtr_all, Xte, ytr_all, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=SEED)
    Xtr, Xval, ytr, yval = train_test_split(Xtr_all, ytr_all, test_size=0.2, stratify=ytr_all, random_state=SEED)
    Xtr = Xtr.reset_index(drop=True); Xval = Xval.reset_index(drop=True)
    Xte = Xte.reset_index(drop=True)
    ytr = ytr.reset_index(drop=True); yval = yval.reset_index(drop=True); yte = yte.reset_index(drop=True)

    cls, counts = sort_classes(ytr)
    cls_arr = np.array(cls, dtype=object)
    ypos = pd.Series(ytr).map({c: k for k, c in enumerate(cls)}).values

    flat = HistGradientBoostingClassifier(random_state=SEED, max_iter=FINAL_MAX_ITER).fit(Xtr_all, ytr_all)
    pf = flat.predict(Xte)
    print("[Flat]     acc={:.4f} macro-F1={:.4f} weighted-F1={:.4f}".format(
        accuracy_score(yte, pf), f1_score(yte, pf, average="macro", zero_division=0),
        f1_score(yte, pf, average="weighted", zero_division=0)))

    C = len(cls)
    base = C // 4
    rem = C % 4
    fixed4 = []
    idx = 0
    for lv in range(4):
        sz = base + (1 if lv < rem else 0)
        fixed4.append((idx, idx + sz - 1))
        idx += sz
    final_eval_fixed = fixed4
    n = len(Xte)
    pred = np.full(n, None, dtype=object); rem_mask = np.ones(n, dtype=bool); pos = np.arange(n)
    for (j, i) in fixed4:
        if not rem_mask.any():
            break
        m = build_interval_model(Xtr, ypos, cls_arr, j, i, FINAL_MAX_ITER, 10 ** 9)
        ridx = pos[rem_mask]
        pr = m.predict(Xte.iloc[ridx])
        for a, pp in enumerate(pr):
            if pp != "Other":
                pred[ridx[a]] = pp; rem_mask[ridx[a]] = False
    if rem_mask.any():
        pred[rem_mask] = cls[-1]
    print("[Fixed-4]  acc={:.4f} macro-F1={:.4f} weighted-F1={:.4f}".format(
        accuracy_score(yte, pred), f1_score(yte, pred, average="macro", zero_division=0),
        f1_score(yte, pred, average="weighted", zero_division=0)))

    print("Searching partition (greedy merge, val macro-F1)...")
    best, score, ncache = greedy_merge(Xtr, ypos, cls, cls_arr, Xval, yval, SEARCH_MAX_ITER, PER_CLASS_CAP)
    print("best val partition:", best, "val macro-F1={:.4f} ({} cached models)".format(score, ncache))
    final_eval(name, Xtr, ytr, Xte, yte, best, cls)
    print()


if __name__ == "__main__":
    run("CIC-IDS2017", CIC, "Label")
    run("Edge-IIoTset(ML)", EDGE_ML, "Attack_type")
    run("ToN-IoT(network)", TON_NET, "type")
