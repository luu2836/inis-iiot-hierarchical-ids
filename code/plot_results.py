import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

OUT = r"C:\Users\24540\Desktop\EI\results"
datasets = ["Edge-IIoTset", "ToN-IoT", "CIC-IDS2017", "UNSW-NB15"]
methods = ["flat-GBDT", "hier-GBDT", "flat-MLP", "hier-MLP"]
colors = ["#b0b0b0", "#2c7fb8", "#f0a0a0", "#d62728"]

macro = {
    "flat-GBDT": [0.9694, 0.9376, 0.7253, 0.5883],
    "hier-GBDT": [0.9805, 0.9561, 0.9018, 0.6491],
    "flat-MLP":  [0.9711, 0.9466, 0.8396, 0.5928],
    "hier-MLP":  [0.9736, 0.9460, 0.8495, 0.6206],
}
mstd = {
    "flat-GBDT": [0.0080, 0.0068, 0.0041, 0.0106],
    "hier-GBDT": [0.0007, 0.0010, 0.0141, 0.0047],
    "flat-MLP":  [0.0013, 0.0009, 0.0056, 0.0080],
    "hier-MLP":  [0.0028, 0.0012, 0.0071, 0.0051],
}
worst = {
    "flat-GBDT": [0.6548, 0.6063, 0.0000, 0.0733],
    "hier-GBDT": [0.7977, 0.7572, 0.3684, 0.2040],
    "flat-MLP":  [0.7963, 0.7007, 0.0000, 0.0842],
    "hier-MLP":  [0.7982, 0.6937, 0.0277, 0.1662],
}
wstd = {
    "flat-GBDT": [0.1156, 0.0612, 0.0000, 0.0131],
    "hier-GBDT": [0.0109, 0.0116, 0.0652, 0.0154],
    "flat-MLP":  [0.0135, 0.0101, 0.0000, 0.0087],
    "hier-MLP":  [0.0096, 0.0103, 0.0245, 0.0309],
}


def grouped_bar(data, err, title, fname):
    x = np.arange(len(datasets)); w = 0.2
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, m in enumerate(methods):
        ax.bar(x + (i - 1.5) * w, data[m], w, yerr=err[m], capsize=2, label=m,
               color=colors[i], edgecolor="black", linewidth=0.4)
    ax.set_xticks(x); ax.set_xticklabels(datasets, fontsize=12)
    ax.set_ylabel(title.split(":")[0]); ax.set_ylim(0, 1.02)
    ax.set_title(title); ax.legend(fontsize=12, ncol=4); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    p = os.path.join(OUT, fname); plt.savefig(p, dpi=200); plt.close(); print("saved", p)


grouped_bar(macro, mstd, "Macro-F1: flat vs hierarchical (5 seeds)", "fig_macroF1.png")
grouped_bar(worst, wstd, "Worst-class F1: flat vs hierarchical (5 seeds)", "fig_worstF1.png")
print("done")


