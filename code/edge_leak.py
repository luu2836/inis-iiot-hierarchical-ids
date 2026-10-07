import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from config import EDGE_ML, SEED
from baseline import prep, ID_COLS

LEAK_SUBSTR = ["qry.name.len", "conack", "protoname", "mqtt.topic"]

df = pd.read_csv(EDGE_ML, low_memory=False)
cand = [c for c in df.columns if any(s in c for s in LEAK_SUBSTR)]
print("leakage candidate columns found:", cand)


def run(tag, drop_leak):
    d = df.drop(columns=[c for c in cand if drop_leak], errors="ignore")
    X, y = prep(d, "Attack_type", ID_COLS)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=SEED)
    clf = HistGradientBoostingClassifier(random_state=SEED, max_iter=200).fit(Xtr, ytr)
    pred = clf.predict(Xte)
    f1m = f1_score(yte, pred, average="macro", zero_division=0)
    f1w = f1_score(yte, pred, average="weighted", zero_division=0)
    per = f1_score(yte, pred, average=None, labels=sorted(y.unique()), zero_division=0)
    print("[{}] acc={:.4f} macro-F1={:.4f} weighted-F1={:.4f} worst={:.4f}".format(
        tag, accuracy_score(yte, pred), f1m, f1w, float(np.min(per))))


run("with-leak", False)
run("no-leak", True)
