import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from config import EDGE_ML, TON_NET, RESULT_DIR
import os

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def describe(name, df, class_col):
    print("=" * 70)
    print(name)
    print("=" * 70)
    print("shape:", df.shape)
    print("dtypes:\n", df.dtypes.value_counts())
    print("\nmissing values (top 10):")
    miss = df.isna().sum()
    print(miss[miss > 0].sort_values(ascending=False).head(10))
    print("\nclass distribution ({}):".format(class_col))
    vc = df[class_col].value_counts()
    print(vc)
    ir = vc.max() / vc.min()
    print("imbalance ratio (max/min): {:.1f}".format(ir))

    fig, ax = plt.subplots(figsize=(10, 5))
    vc.plot(kind="bar", ax=ax, color="steelblue")
    ax.set_title("{} - class distribution ({})".format(name, class_col))
    ax.set_ylabel("count")
    for i, v in enumerate(vc.values):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=7)
    plt.tight_layout()
    out = os.path.join(RESULT_DIR, "dist_{}.png".format(name.replace("/", "_").replace(" ", "_")))
    plt.savefig(out, dpi=120)
    plt.close()
    print("saved plot ->", out)
    print()


if __name__ == "__main__":
    edge = pd.read_csv(EDGE_ML, low_memory=False)
    describe("Edge-IIoTset(ML)", edge, "Attack_type")

    ton = pd.read_csv(TON_NET, low_memory=False)
    describe("ToN-IoT(network)", ton, "type")
