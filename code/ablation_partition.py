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

SEEDS = [1, 2, 3]
N_RANDOM = 5
SEED = 1


def train_cascade(Xtr, ytr, cls_order, Xte, K=4):
    # position of each class in cls_order
    order_pos = {c: k for k, c in enumerate(cls_order)}
    pos = np.array([order_pos[c] for c in ytr])
    C = len(cls_order)
    intervals = fixed_partition(C, K)
    cls_order_arr = np.array(cls_order, dtype=object)
    n = len(Xte)
    pred = np.array([None] * n, dtype=object)
    rem = np.ones(n, dtype=bool)
    ip = np.arange(n)
    for (j, i) in intervals:
        mask = pos >= j
        Xk = Xtr[mask]; pk = pos[mask]
        labs = np.full(len(pk), "Other", dtype=object)
        sel = (pk >= j) & (pk <= i)
        labs[sel] = cls_order_arr[pk[sel]]
        m = HistGradientBoostingClassifier(random_state=1, max_iter=200).fit(Xk, labs)
        if not rem.any():
            break
        ridx = ip[rem]; out = m.predict(Xte.iloc[ridx])
        for a, pp in enumerate(out):
            if pp != "Other":
                pred[ridx[a]] = pp; rem[ridx[a]] = False
    if rem.any():
        pred[rem] = cls_order[-1]
    return pred


def run(name, path, label_col, dedup=False):
    print("=" * 78); print(name); print("=" * 78)
    df = pd.read_csv(path, low_memory=False)
    if dedup:
        feat = [c for c in df.columns if c != label_col]
        df = df.drop_duplicates(subset=feat).reset_index(drop=True)
    X, y = prep(df, label_col, ID_COLS)
    freq_m, freq_w, rand_m, rand_w, sem_m, sem_w = [], [], [], [], [], []
    for s in SEEDS:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=s)
        Xtr = Xtr.reset_index(drop=True); Xte = Xte.reset_index(drop=True)
        ytr = ytr.reset_index(drop=True); yte = yte.reset_index(drop=True)
        cls, counts = sort_classes(ytr)
        labels_all = sorted(ytr.unique())

        # (1) frequency-sorted partition
        p = train_cascade(Xtr, ytr, cls, Xte)
        freq_m.append(f1_score(yte, p, average="macro", zero_division=0))
        freq_w.append(float(np.min(f1_score(yte, p, average=None, labels=labels_all, zero_division=0))))

        # (2) random permutations
        rm, rw = [], []
        for r in range(N_RANDOM):
            rng = np.random.RandomState(s * 100 + r)
            perm = list(cls); rng.shuffle(perm)
            p = train_cascade(Xtr, ytr, perm, Xte)
            rm.append(f1_score(yte, p, average="macro", zero_division=0))
            rw.append(float(np.min(f1_score(yte, p, average=None, labels=labels_all, zero_division=0))))
        rand_m.append(np.mean(rm)); rand_w.append(np.mean(rw))

        # (3) reverse-frequency (ascending counts) as a structured alternative
        p = train_cascade(Xtr, ytr, list(reversed(cls)), Xte)
        sem_m.append(f1_score(yte, p, average="macro", zero_division=0))
        sem_w.append(float(np.min(f1_score(yte, p, average=None, labels=labels_all, zero_division=0))))
        print("  seed {} done".format(s))

    print("{:<28} macro-F1={:.4f}  worst-F1={:.4f}".format("frequency-desc (ours)", np.mean(freq_m), np.mean(freq_w)))
    print("{:<28} macro-F1={:.4f}  worst-F1={:.4f}".format("random order (avg %d)" % N_RANDOM, np.mean(rand_m), np.mean(rand_w)))
    print("{:<28} macro-F1={:.4f}  worst-F1={:.4f}".format("frequency-asc (reverse)", np.mean(sem_m), np.mean(sem_w)))
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
