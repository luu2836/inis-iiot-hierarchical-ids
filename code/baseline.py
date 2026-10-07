import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from config import EDGE_ML, TON_NET, SEED

ID_COLS = ["id", "src_ip", "dst_ip", "src_port", "dst_port", "frame.time", "ip.src_host",
           "ip.dst_host", "tcp.srcport", "tcp.dstport", "udp.port", "arp.src.proto_ipv4",
           "arp.dst.proto_ipv4"]


def prep(df, label_col, drop_cols):
    y = df[label_col].astype(str)
    X = df.drop(columns=[c for c in drop_cols + [label_col] if c in df.columns])
    obj = X.select_dtypes(include="object").columns.tolist()
    for c in obj:
        X[c] = X[c].astype("category").cat.codes
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(-999)
    return X, y


def run(name, path, label_col, drop_cols):
    print("=" * 70)
    print(name, "| label:", label_col)
    print("=" * 70)
    df = pd.read_csv(path, low_memory=False)
    X, y = prep(df, label_col, drop_cols)
    print("features:", X.shape[1], "| classes:", y.nunique())
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=SEED)

    for clf_name, clf in [
        ("RandomForest", RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=SEED, class_weight="balanced_subsample")),
        ("HistGB", HistGradientBoostingClassifier(random_state=SEED, max_iter=200)),
    ]:
        clf.fit(Xtr, ytr)
        pred = clf.predict(Xte)
        acc = accuracy_score(yte, pred)
        f1m = f1_score(yte, pred, average="macro")
        f1w = f1_score(yte, pred, average="weighted")
        print("\n[{}] acc={:.4f} macro-F1={:.4f} weighted-F1={:.4f}".format(clf_name, acc, f1m, f1w))
        print(classification_report(yte, pred, digits=4, zero_division=0))


if __name__ == "__main__":
    run("Edge-IIoTset(ML)", EDGE_ML, "Attack_type", ID_COLS)
    print()
    run("ToN-IoT(network)", TON_NET, "type", ID_COLS)
