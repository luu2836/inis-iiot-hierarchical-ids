import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

OUT = r"C:\Users\24540\Desktop\EI\results"
datasets = ["Edge-IIoTset", "ToN-IoT", "UNSW-NB15", "CIC-IDS2017"]
methods = ["flat", "+weight", "+SMOTE", "+focal", "hier", "hier+weight"]

macro = {
    "flat":        [0.9694, 0.9376, 0.5883, 0.7253],
    "+weight":     [0.9645, 0.9536, 0.6235, 0.9162],
    "+SMOTE":      [0.9661, 0.9564, 0.6491, 0.8925],
    "+focal":      [0.9783, 0.9473, 0.6002, 0.8632],
    "hier":        [0.9805, 0.9570, 0.6551, 0.9022],
    "hier+weight": [0.9760, 0.9508, 0.6342, 0.9021],
}
std = {
    "flat":        [0.0080, 0.0068, 0.0106, 0.0041],
    "+weight":     [0.0011, 0.0017, 0.0037, 0.0134],
    "+SMOTE":      [0.0027, 0.0018, 0.0106, 0.0593],
    "+focal":      [0.0031, 0.0025, 0.0071, 0.0169],
    "hier":        [0.0005, 0.0012, 0.0041, 0.0146],
    "hier+weight": [0.0012, 0.0032, 0.0045, 0.0140],
}
colors = ["#b0b0b0", "#7fb3d5", "#f5b041", "#e59866", "#2c7fb8", "#1a5276"]

x = np.arange(len(datasets)); w = 0.13
fig, ax = plt.subplots(figsize=(9.5, 4.5))
for i, m in enumerate(methods):
    ax.bar(x + (i - 2.5) * w, macro[m], w, yerr=std[m], capsize=2, label=m,
           color=colors[i], edgecolor="black", linewidth=0.4)
ax.set_xticks(x); ax.set_xticklabels(datasets, fontsize=12)
ax.set_ylabel("Macro-F1"); ax.set_ylim(0.5, 1.0)
ax.set_title("Macro-F1: hierarchy vs imbalance-handling methods (5 seeds)")
ax.legend(fontsize=12, ncol=6); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
p = os.path.join(OUT, "fig_compare.png")
plt.savefig(p, dpi=200); plt.close()
print("saved", p)


