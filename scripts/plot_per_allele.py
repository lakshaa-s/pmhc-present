"""Per-allele AUROC distribution plot, coloured by HLA locus (A/B/C).
Shows the full 123-allele spread and the HLA-C tail. Reads results/per_allele_auroc.csv.
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

csv = sys.argv[1] if len(sys.argv) > 1 else "results/per_allele_auroc.csv"
df = pd.read_csv(csv)

# locus from allele string: HLA-A*.., HLA-B*.., HLA-C*..
df["locus"] = df["allele"].str.extract(r"HLA-([ABC])")
INK = "#12343B"
COL = {"A": "#028090", "B": "#F4A259", "C": "#F07167"}  # C in coral to pop

fig, ax = plt.subplots(figsize=(9, 5.6))

# jittered strip plot by locus
rng = np.random.default_rng(0)
xpos = {"A": 0, "B": 1, "C": 2}
for loc in ["A", "B", "C"]:
    sub = df[df.locus == loc]
    x = xpos[loc] + rng.uniform(-0.28, 0.28, len(sub))
    ax.scatter(x, sub["auroc"], s=42, color=COL[loc], edgecolor=INK,
               linewidth=0.6, alpha=0.85, zorder=3,
               label=f"HLA-{loc}  (n={len(sub)}, median {sub.auroc.median():.3f})")
    # median bar
    ax.plot([xpos[loc]-0.32, xpos[loc]+0.32], [sub.auroc.median()]*2,
            color=INK, lw=2, zorder=4)

# annotate worst few
worst = df.nsmallest(4, "auroc")
for _, r in worst.iterrows():
    loc = r["locus"]
    ax.annotate(r["allele"], xy=(xpos[loc], r["auroc"]),
                xytext=(xpos[loc]+0.38, r["auroc"]), fontsize=8.5,
                color=INK, va="center",
                arrowprops=dict(arrowstyle="-", color="#AAB9BC", lw=0.7))

ax.set_xticks([0, 1, 2])
ax.set_xticklabels(["HLA-A", "HLA-B", "HLA-C"], fontsize=12)
ax.set_ylabel("Per-allele test AUROC", fontsize=12.5)
ax.set_title("Per-allele performance by locus — HLA-C trails",
             fontsize=13.5, fontweight="bold", color=INK, pad=12)
ax.set_ylim(min(0.87, df.auroc.min()-0.01), 1.0)
ax.legend(frameon=False, fontsize=9.5, loc="lower left")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.yaxis.grid(True, color="#E1EAEA", lw=0.7)
ax.set_axisbelow(True)
fig.text(0.5, -0.02,
         "Each point = one allele. HLA-C is motif-distinct from the A/B-heavy training "
         "data, consistent with the orphan-allele effect.",
         ha="center", fontsize=9.5, style="italic", color="#5B7B80")
plt.tight_layout()
plt.savefig("per_allele_dist.png", dpi=200, bbox_inches="tight")
print("saved per_allele_dist.png")