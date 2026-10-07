import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from scipy import stats
from config import EDGE_ML, TON_NET
from baseline import prep, ID_COLS
from hierarchical import BASE, build_partition, train_hier, predict_hier, sort_classes

SEEDS = [1, 2, 3, 4, 5]
MODES = ["flat", "fixed", "adaptive"]


def metrics(yte, pred, classes):
    acc = accuracy_score(yte, pred)
    f1m = f1_score(yte, pred, average="macro", zero_division=0)
    f1w = f1_score(yte, pred, average="weighted", zero_division=0)
    per = f1_score(yte, pred, average=None, labels=classes, zero_division=0)
    return acc, f1m, f1w, float(np.min(per))


def run_ds(name, path, label_col):
    print("=" * 78)
    print(name)
    print("=" * 78)
    df = pd.read_csv(path, low_memory=False)
    X, y = prep(df, label_col, ID_COLS)
    res = {m: [] for m in MODES}
    for s in SEEDS:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=s)
        Xtr = Xtr.reset_index(drop=True); Xte = Xte.reset_index(drop=True)
        ytr = ytr.reset_index(drop=True); yte = yte.reset_index(drop=True)
        classes, _ = sort_classes(ytr)

        flat = clone(BASE).fit(Xtr, ytr)
        res["flat"].append(metrics(yte, flat.predict(Xte), classes))

        for mode in ["fixed", "adaptive"]:
            levels = build_partition(Xtr, ytr, mode)
            models = []
            train_hier(models, Xtr, ytr, levels)
            pred = predict_hier(models, levels, Xte)
            res[mode].append(metrics(yte, pred, classes))
        print("  seed {} done".format(s))

    print("\n{:<10} {:>16} {:>17} {:>17} {:>16}".format(
        "method", "acc", "macro-F1", "weighted-F1", "worst-class-F1"))
    for m in MODES:
        a = np.array(res[m])
        print("{:<10} {:.4f}+-{:.4f}   {:.4f}+-{:.4f}    {:.4f}+-{:.4f}    {:.4f}+-{:.4f}".format(
            m, a[:, 0].mean(), a[:, 0].std(), a[:, 1].mean(), a[:, 1].std(),
            a[:, 2].mean(), a[:, 2].std(), a[:, 3].mean(), a[:, 3].std()))

    print("\nsignificance on macro-F1 (paired, n={}):".format(len(SEEDS)))
    for x, yv in [("adaptive", "fixed"), ("adaptive", "flat"), ("fixed", "flat")]:
        a = np.array(res[x])[:, 1]
        b = np.array(res[yv])[:, 1]
        d = a - b
        try:
            t, pv = stats.ttest_rel(a, b)
        except Exception:
            t, pv = float("nan"), float("nan")
        try:
            w, wp = stats.wilcoxon(a, b)
        except Exception:
            w, wp = float("nan"), float("nan")
        print("  {:<9} vs {:<9} mean_diff={:+.4f}  t-test p={:.4f}  wilcoxon p={:.4f}".format(
            x, yv, d.mean(), pv, wp))
    print()


if __name__ == "__main__":
    run_ds("Edge-IIoTset(ML)", EDGE_ML, "Attack_type")
    run_ds("ToN-IoT(network)", TON_NET, "type")
