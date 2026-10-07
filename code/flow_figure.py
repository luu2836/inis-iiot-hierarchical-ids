import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

OUT = r"C:\Users\24540\Desktop\EI\results"
# CIC-IDS2017 cascade (from cascade_error.py): level -> [correct, premature, in-level, over, passed]
levels = ["Level 1", "Level 2", "Level 3", "Level 4"]
data = np.array([
    [52885, 22, 23, 18, 10235],
    [8017, 2, 15, 3, 2216],
    [1960, 20, 220, 3, 16],
    [16, 3, 0, 0, 0],
], dtype=float)
cats = ["Correct", "Premature", "In-level error", "Over-routed", "Passed down"]
colors = ["#2c7fb8", "#f39c12", "#d62728", "#8e44ad", "#95a5a6"]

reached = data.sum(1)
frac = data / reached[:, None]

fig, ax = plt.subplots(figsize=(8, 3.6))
left = np.zeros(len(levels))
y = np.arange(len(levels))
for c in range(5):
    ax.barh(y, frac[:, c] * 100, left=left, color=colors[c], label=cats[c], edgecolor="black", linewidth=0.3)
    left += frac[:, c] * 100
ax.set_yticks(y)
ax.set_yticklabels(["{} (n={:,})".format(l, int(r)) for l, r in zip(levels, reached)], fontsize=12)
ax.invert_yaxis()
ax.set_xlabel("share of samples reaching the level (%)")
ax.set_xlim(0, 100)
ax.set_title("Cascade routing flow on CIC-IDS2017 (deduplicated)")
ax.legend(fontsize=12, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.30))
plt.tight_layout()
p = os.path.join(OUT, "fig_flow.png")
plt.savefig(p, dpi=200, bbox_inches="tight")
plt.close()
print("saved", p)
print("reached:", dict(zip(levels, reached.astype(int))))


