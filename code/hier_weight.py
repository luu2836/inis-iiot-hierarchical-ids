import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from config import EDGE_ML, TON_NET, CIC, UNSW
from baseline import prep, ID_COLS
from nn_hier import fixed_partition
from search_partition import sort_classes

SEED = 1


def wmodel(X, ypos, cls_arr, j, i):
    mask = ypos >= j
    Xk = X[mask]; p = ypos[mask]
    labs = np.full(len(p), "Other", dtype=object)
    sel = (p >= j) & (p <= i)
    labs[sel] = cls_arr[p[sel]]
    cnt = pd.Series(labs).value_counts()
    w = np.array([len(labs) / (len(cnt) * cnt[l]) for l in labs])
    return HistGradientBoostingClassifier(random_state=SEED, max_iter=200).fit(Xk, labs, sample_weight=w)


def metrics(yte, pred, cls):
    per = f1_score(yte, pred, average=None, labels=cls, zero_division=0)
    return accuracy_score(yte, pred), f1_score(yte, pred, average="macro", zero_division=0), \
        f1_score(yte, pred, average="weighted", zero_division=0), float(np.min(per))


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
    cls, _ = sort_classes(ytr)
    cls_arr = np.array(cls, dtype=object)
    ypos = pd.Series(ytr).map({c: k for k, c in enumerate(cls)}).values
    intervals = fixed_partition(len(cls), 4)
    pred = np.array([None] * len(Xte), dtype=object); rem = np.ones(len(Xte), dtype=bool); ip = np.arange(len(Xte))
    for (j, i) in intervals:
        if not rem.any():
            break
        m = wmodel(Xtr, ypos, cls_arr, j, i)
        ridx = ip[rem]; out = m.predict(Xte.iloc[ridx])
        for a, pp in enumerate(out):
            if pp != "Other":
                pred[ridx[a]] = pp; rem[ridx[a]] = False
    if rem.any():
        pred[rem] = cls[-1]
    print("[hier-GBDT+weight] ", " ".join("{}={:.3f}".format(*z) for z in zip(["acc", "macF1", "wF1", "worst"], metrics(yte, pred, cls))))
    print()


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "cic"
    if which == "edge":
        run("Edge-IIoTset(ML)", EDGE_ML, "Attack_type")
    elif which == "ton":
        run("ToN-IoT(network)", TON_NET, "type")
    elif which == "cic":
        run("CIC-IDS2017(dedup)", CIC, "Label", True)
    elif which == "unsw":
        run("UNSW-NB15", UNSW, "attack_cat")
