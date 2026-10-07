import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from config import EDGE_ML, TON_NET, CIC, UNSW
from baseline import prep, ID_COLS
from nn_hier import fixed_partition
from search_partition import sort_classes
from compare_all import level_model

SEEDS = [1, 2, 3]
OUT = r"C:\Users\24540\Desktop\EI\results"
DS = [("Edge-IIoTset", EDGE_ML, "Attack_type", False), ("ToN-IoT", TON_NET, "type", False),
      ("UNSW-NB15", UNSW, "attack_cat", False), ("CIC-IDS2017", CIC, "Label", True)]


def cascade(Xtr, ypos, cls_arr, cls, Xte, seed, K):
    intervals = fixed_partition(len(cls), K)
    pred = np.array([None] * len(Xte), dtype=object); rem = np.ones(len(Xte), dtype=bool); ip = np.arange(len(Xte))
    for (j, i) in intervals:
        if not rem.any():
            break
        m = level_model(Xtr, ypos, cls_arr, j, i, seed, False)
        ridx = ip[rem]; out = m.predict(Xte.iloc[ridx])
        for a, pp in enumerate(out):
            if pp != "Other":
                pred[ridx[a]] = pp; rem[ridx[a]] = False
    if rem.any():
        pred[rem] = cls[-1]
    return pred


def run(name, path, label_col, dedup):
    df = pd.read_csv(path, low_memory=False)
    if dedup:
        feat = [c for c in df.columns if c != label_col]
        df = df.drop_duplicates(subset=feat).reset_index(drop=True)
    X, y = prep(df, label_col, ID_COLS)
    ks = [1, 2, 3, 4, 5, 6, 8]
    macro = {k: [] for k in ks}; worst = {k: [] for k in ks}
    for s in SEEDS:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=s)
        Xtr = Xtr.reset_index(drop=True); Xte = Xte.reset_index(drop=True)
        ytr = ytr.reset_index(drop=True); yte = yte.reset_index(drop=True)
        cls, _ = sort_classes(ytr); cls_arr = np.array(cls, dtype=object)
        ypos = pd.Series(ytr).map({c: k for k, c in enumerate(cls)}).values
        for k in ks:
            if k > len(cls):
                continue
            p = cascade(Xtr, ypos, cls_arr, cls, Xte, s, k)
            macro[k].append(f1_score(yte, p, average="macro", zero_division=0))
            worst[k].append(float(np.min(f1_score(yte, p, average=None, labels=cls, zero_division=0))))
    xs = [k for k in ks if k <= len(cls)]
    print(name)
    for k in xs:
        print("  K={} macro-F1={:.4f} worst={:.4f}".format(k, np.mean(macro[k]), np.mean(worst[k])))
    return xs, [np.mean(macro[k]) for k in xs], [np.mean(worst[k]) for k in xs]


if __name__ == "__main__":
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, path, lab, dd in DS:
        xs, m, w = run(name, path, lab, dd)
        axes[0].plot(xs, m, marker="o", label=name)
        axes[1].plot(xs, w, marker="o", label=name)
    axes[0].set_xlabel("K (number of levels)"); axes[0].set_ylabel("Macro-F1"); axes[0].set_title("Sensitivity to K")
    axes[1].set_xlabel("K (number of levels)"); axes[1].set_ylabel("Worst-class F1"); axes[1].set_title("Sensitivity to K")
    for ax in axes:
        ax.grid(alpha=0.3); ax.legend(fontsize=10)
    plt.tight_layout()
    p = OUT + r"\fig_k_sensitivity.png"
    plt.savefig(p, dpi=200); plt.close()
    print("saved", p)

