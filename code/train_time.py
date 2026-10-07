import sys
import time
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from config import EDGE_ML, TON_NET, CIC, UNSW
from baseline import prep, ID_COLS
from nn_hier import signed_log1p, fixed_partition, train_level
from search_partition import sort_classes
from compare_all import level_model

SEED = 1
HIDDEN = (512, 256, 128)
EPOCHS = 80


def timeit(fn):
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def run(name, path, label_col, dedup=False):
    df = pd.read_csv(path, low_memory=False)
    if dedup:
        feat = [c for c in df.columns if c != label_col]
        df = df.drop_duplicates(subset=feat).reset_index(drop=True)
    X, y = prep(df, label_col, ID_COLS)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=SEED)
    Xtr = Xtr.reset_index(drop=True); ytr = ytr.reset_index(drop=True)
    cls, _ = sort_classes(ytr)
    cls_arr = np.array(cls, dtype=object)
    ypos = pd.Series(ytr).map({c: k for k, c in enumerate(cls)}).values
    intervals = fixed_partition(len(cls), 4)

    t_flat_gbdt = timeit(lambda: HistGradientBoostingClassifier(random_state=SEED, max_iter=200).fit(Xtr, ytr))

    def hier_gbdt():
        for (j, i) in intervals:
            level_model(Xtr, ypos, cls_arr, j, i, SEED, False)
    t_hier_gbdt = timeit(hier_gbdt)

    a = signed_log1p(Xtr.values.astype(float))
    mu, sd = a.mean(0), a.std(0); sd[sd == 0] = 1
    Xs = ((a - mu) / sd).astype(np.float32)
    pm = {c: k for k, c in enumerate(cls)}
    ycode = np.array([pm[c] for c in ytr], dtype=np.int64)

    t_flat_nn = timeit(lambda: train_level(Xs, ycode, len(cls), epochs=EPOCHS))

    def hier_nn():
        for (j, i) in intervals:
            mask = ypos >= j
            yl = np.where(ypos[mask] <= i, ypos[mask] - j, i - j + 1).astype(np.int64)
            train_level(Xs[mask], yl, (i - j + 1) + 1, epochs=EPOCHS)
    t_hier_nn = timeit(hier_nn)

    print("{:<14} flat-GBDT={:6.1f}s  hier-GBDT={:6.1f}s ({:.2f}x)  flat-MLP={:6.1f}s  hier-MLP={:6.1f}s ({:.2f}x)".format(
        name, t_flat_gbdt, t_hier_gbdt, t_hier_gbdt / t_flat_gbdt, t_flat_nn, t_hier_nn, t_hier_nn / t_flat_nn))


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "edge"):
        run("Edge-IIoTset", EDGE_ML, "Attack_type")
    if which in ("all", "ton"):
        run("ToN-IoT", TON_NET, "type")
    if which in ("all", "unsw"):
        run("UNSW-NB15", UNSW, "attack_cat")
    if which in ("all", "cic"):
        run("CIC-IDS2017", CIC, "Label", True)
