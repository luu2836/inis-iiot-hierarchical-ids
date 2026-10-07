import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from config import CIC, SEED
from baseline import prep, ID_COLS
from search_partition import build_interval_model, sort_classes, eval_partition

df = pd.read_csv(CIC, low_memory=False)
feat = [c for c in df.columns if c != "Label"]
df = df.drop_duplicates(subset=feat).reset_index(drop=True)
print("after exact dedup:", len(df))
print(df["Label"].value_counts().to_string())

X, y = prep(df, "Label", ID_COLS)
Xtr_all, Xte, ytr_all, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=SEED)
Xtr_all = Xtr_all.reset_index(drop=True); Xte = Xte.reset_index(drop=True)
ytr_all = ytr_all.reset_index(drop=True); yte = yte.reset_index(drop=True)
Xtr, Xval, ytr, yval = train_test_split(Xtr_all, ytr_all, test_size=0.2, stratify=ytr_all, random_state=SEED)
Xtr = Xtr.reset_index(drop=True); ytr = ytr.reset_index(drop=True)
Xval = Xval.reset_index(drop=True); yval = yval.reset_index(drop=True)
cls, counts = sort_classes(ytr)
cls_arr = np.array(cls, dtype=object)
ypos = pd.Series(ytr).map({c: k for k, c in enumerate(cls)}).values

flat = HistGradientBoostingClassifier(random_state=SEED, max_iter=200).fit(Xtr_all, ytr_all)
pf = flat.predict(Xte)
print("[Flat]    macro-F1={:.4f}".format(f1_score(yte, pf, average="macro", zero_division=0)))


def run_partition(intervals, tag):
    n = len(Xte)
    pred = np.full(n, None, dtype=object); rem = np.ones(n, dtype=bool); pos = np.arange(n)
    for (j, i) in intervals:
        if not rem.any():
            break
        m = build_interval_model(Xtr, ypos, cls_arr, j, i, 200, 10 ** 9)
        ridx = pos[rem]
        pr = m.predict(Xte.iloc[ridx])
        for a, pp in enumerate(pr):
            if pp != "Other":
                pred[ridx[a]] = pp; rem[ridx[a]] = False
    if rem.any():
        pred[rem] = cls[-1]
    per = f1_score(yte, pred, average=None, labels=cls, zero_division=0)
    print("[{}] macro-F1={:.4f}".format(tag, f1_score(yte, pred, average="macro", zero_division=0)))
    print("   ", {c: round(float(v), 3) for c, v in zip(cls, per) if v < 0.9})


run_partition([(0, 3), (4, 7), (8, 11), (12, 14)], "Fixed-4")
run_partition([(0, 9), (10, 14)], "Searched(leaky-selected)")
run_partition([(p, p) for p in range(len(cls))], "All-singleton")
