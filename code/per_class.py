import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from config import EDGE_ML, TON_NET, CIC, UNSW
from baseline import prep, ID_COLS
from nn_hier import signed_log1p, fixed_partition
from search_partition import sort_classes, build_interval_model

OUT = r"C:\Users\24540\Desktop\EI\results"
DATASETS = [("Edge-IIoTset", EDGE_ML, "Attack_type", False), ("ToN-IoT", TON_NET, "type", False),
            ("CIC-IDS2017", CIC, "Label", True), ("UNSW-NB15", UNSW, "attack_cat", False)]


def load(path, label_col, dedup):
    df = pd.read_csv(path, low_memory=False)
    if dedup:
        feat = [c for c in df.columns if c != label_col]
        df = df.drop_duplicates(subset=feat).reset_index(drop=True)
    X, y = prep(df, label_col, ID_COLS)
    return X, y


print("=== dataset stats ===")
for name, path, lab, dd in DATASETS:
    X, y = load(path, lab, dd)
    vc = y.value_counts()
    print("{:<14} rows={:<8} classes={:<3} IR(max/min)={:.1f}".format(name, len(y), len(vc), vc.max() / vc.min()))


def per_class(name, path, label_col, dedup, fname):
    X, y = load(path, label_col, dedup)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=1)
    Xtr = Xtr.reset_index(drop=True); Xte = Xte.reset_index(drop=True)
    ytr = ytr.reset_index(drop=True); yte = yte.reset_index(drop=True)
    cls, counts = sort_classes(ytr)
    C = len(cls)
    cls_arr = np.array(cls, dtype=object)
    ypos = pd.Series(ytr).map({c: k for k, c in enumerate(cls)}).values
    intervals = fixed_partition(C, 4)

    flat = HistGradientBoostingClassifier(random_state=1, max_iter=200).fit(Xtr, ytr)
    pf = f1_score(yte, flat.predict(Xte), average=None, labels=cls, zero_division=0)

    pred = np.array([None] * len(Xte), dtype=object); rem = np.ones(len(Xte), dtype=bool); ip = np.arange(len(Xte))
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
    ph = f1_score(yte, pred, average=None, labels=cls, zero_division=0)

    x = np.arange(C); w = 0.4
    fig, ax = plt.subplots(figsize=(8, max(4, C * 0.42)))
    ax.barh(x - w / 2, pf, w, label="flat", color="#b0b0b0", edgecolor="black", linewidth=0.4)
    ax.barh(x + w / 2, ph, w, label="hierarchical", color="#2c7fb8", edgecolor="black", linewidth=0.4)
    ax.set_yticks(x)
    ax.set_yticklabels([str(c)[:22] for c in cls], fontsize=12)
    ax.invert_yaxis()
    ax.set_xlabel("per-class F1")
    ax.set_title("Per-class F1: flat vs hierarchical - {}".format(name))
    ax.legend(fontsize=12)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    p = OUT + "\\" + fname
    plt.savefig(p, dpi=200)
    plt.close()
    print("saved", p)
    for c, a, b in zip(cls, pf, ph):
        print("   {:<26} flat={:.3f} hier={:.3f}".format(str(c), a, b))


per_class("UNSW-NB15", UNSW, "attack_cat", False, "fig_perclass_unsw.png")
per_class("CIC-IDS2017", CIC, "Label", True, "fig_perclass_cic.png")


