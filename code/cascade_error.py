import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from config import EDGE_ML, TON_NET, CIC, UNSW
from baseline import prep, ID_COLS
from nn_hier import fixed_partition
from search_partition import sort_classes
from compare_all import level_model

SEED = 1


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
    cls, _ = sort_classes(ytr); C = len(cls)
    cls_arr = np.array(cls, dtype=object)
    ypos = pd.Series(ytr).map({c: k for k, c in enumerate(cls)}).values
    # true test positions
    pm = {c: k for k, c in enumerate(cls)}
    tpos = yte.map(pm).values
    intervals = fixed_partition(C, 4)

    n = len(Xte)
    pred = np.array([None] * n, dtype=object)
    rem = np.ones(n, dtype=bool)
    ip = np.arange(n)
    rows = []
    for li, (j, i) in enumerate(intervals):
        ridx = ip[rem]
        if len(ridx) == 0:
            break
        m = level_model(Xtr, ypos, cls_arr, j, i, SEED, False)
        out = m.predict(Xte.iloc[ridx])
        reach = len(ridx)
        premature = inlevel_err = over_routed = routed = assigned_ok = 0
        for a, pp in enumerate(out):
            p = tpos[ridx[a]]
            in_here = (j <= p <= i)
            if pp == "Other":
                if in_here:
                    over_routed += 1
                else:
                    routed += 1
            else:
                q = cls.index(pp)
                if in_here:
                    if q == p:
                        assigned_ok += 1
                    else:
                        inlevel_err += 1
                    pred[ridx[a]] = pp; rem[ridx[a]] = False
                else:
                    premature += 1
                    pred[ridx[a]] = pp; rem[ridx[a]] = False
        rows.append((li + 1, reach, assigned_ok, premature, inlevel_err, over_routed, routed))
    # last level handles remaining
    if rem.any():
        ridx = ip[rem]
        m = level_model(Xtr, ypos, cls_arr, intervals[-1][0], intervals[-1][1], SEED, False)
        out = m.predict(Xte.iloc[ridx])
        ok = err = 0
        for a, pp in enumerate(out):
            p = tpos[ridx[a]]
            if cls.index(pp) == p:
                ok += 1
            else:
                err += 1
            pred[ridx[a]] = pp
        rows.append(("last", len(ridx), ok, 0, err, 0, 0))

    print("{:<5} {:>8} {:>9} {:>11} {:>11} {:>11} {:>8}".format("Level", "Reached", "Correct", "Premature", "InLevelErr", "OverRoute", "Routed"))
    for r in rows:
        print("{:<5} {:>8} {:>9} {:>11} {:>11} {:>11} {:>8}".format(*[str(x) for x in r]))
    acc = accuracy_score(yte, pred)
    total_err = int((pred != yte).sum())
    print("final acc={:.4f}  total errors={}".format(acc, total_err))
    print()


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "cic"
    if which == "edge":
        run("Edge-IIoTset(ML)", EDGE_ML, "Attack_type")
    elif which == "ton":
        run("ToN-IoT(network)", TON_NET, "type")
    elif which == "unsw":
        run("UNSW-NB15", UNSW, "attack_cat")
    elif which == "cic":
        run("CIC-IDS2017(dedup)", CIC, "Label", True)
