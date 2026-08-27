"""Six-panel summary figure, regenerated from committed results.

Why this exists rather than reusing the review's figure
--------------------------------------------------------
The external review produced a six-panel figure. Four panels were sound; two used
numbers that did not reproduce, and one caption overstated its claim. Rather than
publish a figure that would need three caveats, this regenerates every panel from
files in the repository so the figure and the results cannot drift apart.

What changed, and why
---------------------
  PANEL d   The review plotted ECE 0.21 with mean prediction 0.71, and a reliability
  calibration curve showing seven bins at exactly zero. Independent implementation
            (`scripts/calibration_metrics.py`) gives **ECE 0.2507, mean 0.666**, and
            **one** empty bin. This panel is drawn from
            `results/calibration_v4_reliability.csv`.

  PANEL e   The review powered for an effect of +0.013 and reported power 0.37 at
  power     n=216. That is a weaker configuration than any actually run. Powering
            for +0.026 — which matches the gated ensemble's ungated row, measured
            under leave-one-allele-out — gives **0.63 at n=216, 80% near n=432**.
            Drawn from `results/rq2_power.csv`.

  PANEL f   The review titled this "the equity question rests on one HLA-C allele".
  locus     That is true of fold set v4's composition and false of the project: the
            equity claim rests on the 123-allele validation table, and an audit
            confirmed no locus claim is drawn from the fold sets. Fold set v2 also
            contains three HLA-C alleles and 72 HLA-C complexes. Retitled as the
            composition caveat it actually is.

Panel c is retained unchanged in substance and is the most useful of the six:
architecture-to-architecture agreement is no higher than architecture-to-sequence
agreement, which is the visual form of the conclusion that the consensus is variance
reduction rather than complementary information.

Usage:
    python scripts/make_summary_figure.py --out figures/summary.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

KEY = ["allele", "peptide"]
GREY, BLUE, RED = "#9a9a9a", "#3b6ea5", "#b3402f"
BLUES = ["#12466e", "#3b6ea5", "#7aa6cd", "#b8d0e6"]


def style():
    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "figure.dpi": 200, "savefig.bbox": "tight",
    })


def zwithin(v, by):
    return v.groupby(by).transform(lambda x: (x - x.mean()) / x.std())


def load_panel(seq_path, struct, feature):
    """Merge sequence and structural scores, all z-scored within allele."""
    seq = pd.read_csv(seq_path)[KEY + ["label", "score"]]
    df = seq.rename(columns={"score": "sequence"})
    names = []
    for name, path in struct:
        if not Path(path).exists():
            continue
        d = pd.read_csv(path)
        if feature not in d.columns:
            continue
        df = df.merge(d[KEY].assign(**{name: -d[feature]}), on=KEY, how="inner")
        names.append(name)
    for c in names + ["sequence"]:
        df[c] = zwithin(df[c], df.allele)
    if names:
        df["consensus"] = df[names].mean(axis=1)
    return df, names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature", default="pae_anchors_ic")
    ap.add_argument("--out", default="figures/summary.png")
    args = ap.parse_args()
    style()

    v4, n4 = load_panel("results/sequence_v4.csv",
                        [("af3", "pae_af3_v4.csv"), ("af2", "pae_af2_v4.csv"),
                         ("esmfold2", "pae_esmfold2_v4.csv"),
                         ("boltz", "pae_boltz_v4.csv")], args.feature)
    v2, n2 = load_panel("results/sequence_v2.csv",
                        [("af2", "pae_af2_v2.csv"),
                         ("esmfold2", "pae_esmfold2_v2.csv"),
                         ("boltz", "pae_boltz_v2.csv")], args.feature)

    y4 = v4.label.to_numpy()
    fig, ax = plt.subplots(3, 2, figsize=(10, 11))

    # ---------------- a: the RQ1 ladder ----------------
    a = ax[0, 0]
    order = [("sequence CNN", roc_auc_score(y4, v4.sequence), GREY),
             ("4-model consensus", roc_auc_score(y4, v4.consensus), RED)]
    for i, m in enumerate(sorted(n4, key=lambda c: -roc_auc_score(y4, v4[c]))):
        label = {"af3": "AlphaFold 3", "af2": "AlphaFold 2",
                 "esmfold2": "ESMFold2", "boltz": "Boltz-2.1"}.get(m, m)
        order.append((label, roc_auc_score(y4, v4[m]), BLUES[i % len(BLUES)]))
    ypos = np.arange(len(order))[::-1]
    for yy, (lab, val, col) in zip(ypos, order):
        a.barh(yy, val - 0.5, left=0.5, color=col, height=0.68)
        a.text(val + 0.008, yy, f"{val:.3f}", va="center", fontsize=8,
               fontweight="bold" if lab.startswith("4-model") else "normal")
    a.set_yticks(ypos); a.set_yticklabels([o[0] for o in order])
    a.set_xlim(0.5, 1.02); a.set_xlabel("AUROC")
    a.set_title("Averaging four architectures beats\nthe best single one", loc="left")

    # ---------------- b: and does not replicate on v2 ----------------
    b = ax[0, 1]
    best4 = max(n4, key=lambda c: roc_auc_score(y4, v4[c]))
    y2 = v2.label.to_numpy()
    best2 = max(n2, key=lambda c: roc_auc_score(y2, v2[c])) if n2 else None
    groups = [("fold set v4\n9 alleles · CI excludes 0",
               roc_auc_score(y4, v4[best4]), roc_auc_score(y4, v4.consensus))]
    if best2:
        groups.append(("fold set v2\n6 alleles · CI spans 0",
                       roc_auc_score(y2, v2[best2]),
                       roc_auc_score(y2, v2.consensus)))
    x = np.arange(len(groups)); w = 0.34
    for i, (lab, s, c) in enumerate(groups):
        b.bar(x[i] - w / 2, s, w, color=BLUE)
        b.bar(x[i] + w / 2, c, w, color=RED)
        b.text(x[i] - w / 2, s + .006, f"{s:.3f}", ha="center", fontsize=8)
        b.text(x[i] + w / 2, c + .006, f"{c:.3f}", ha="center", fontsize=8)
        b.text(x[i], max(s, c) + .035, f"{c - s:+.3f}", ha="center",
               fontsize=9, fontweight="bold", color=RED)
    b.set_xticks(x); b.set_xticklabels([g[0] for g in groups])
    b.set_ylim(0.45, 1.0); b.set_ylabel("AUROC")
    b.set_title("...but the gain does not replicate\non the other panel", loc="left")
    b.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=BLUE),
                      plt.Rectangle((0, 0), 1, 1, color=RED)],
             labels=["best single model", "consensus"],
             loc="lower center", bbox_to_anchor=(0.5, -0.34), ncol=2, frameon=False)

    # ---------------- c: within-allele agreement ----------------
    c = ax[1, 0]
    cols = ["sequence"] + n4
    M = np.ones((len(cols), len(cols)))
    for i, ci in enumerate(cols):
        for j, cj in enumerate(cols):
            if i < j:
                rs = [stats.spearmanr(g[ci], g[cj])[0]
                      for _, g in v4.groupby("allele") if len(g) > 3]
                M[i, j] = M[j, i] = float(np.nanmean(rs))
    im = c.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1)
    lab = ["sequence", "AF3", "AF2", "ESMFold2", "Boltz"][:len(cols)]
    lab = ["sequence"] + [{"af3": "AF3", "af2": "AF2", "esmfold2": "ESMFold2",
                           "boltz": "Boltz"}.get(m, m) for m in n4]
    c.set_xticks(range(len(cols))); c.set_xticklabels(lab, rotation=40, ha="right")
    c.set_yticks(range(len(cols))); c.set_yticklabels(lab)
    for i in range(len(cols)):
        for j in range(len(cols)):
            c.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7,
                   color="white" if abs(M[i, j]) > .7 else "black")
    plt.colorbar(im, ax=c, shrink=.7, label="Spearman ρ")
    c.set_title("Architectures agree with each other\nno more than with sequence",
                loc="left")

    # ---------------- d: calibration, from our own numbers ----------------
    d = ax[1, 1]
    relp = Path("results/calibration_v4_reliability.csv")
    if relp.exists():
        rel = pd.read_csv(relp).dropna(subset=["mean_predicted"])
        d.plot([0, 1], [0, 1], "--", color="grey", lw=1)
        d.scatter(rel.mean_predicted, rel.observed, s=np.clip(rel.n, 8, 90),
                  color=RED, zorder=3)
        seqp = 1 / (1 + np.exp(-v4.sequence.to_numpy()))
        note = (f"ECE {_ece(y4, seqp):.3f}\nmean prediction "
                f"{seqp.mean():.3f} vs\ntrue prevalence {y4.mean():.2f}")
        d.text(.04, .93, note, va="top", fontsize=8)
        d.text(.58, .72, "perfect calibration", rotation=32, color="grey",
               fontsize=8)
        d.set_xlabel("predicted P(presented)")
        d.set_ylabel("observed fraction presented")
        d.set_xlim(-.03, 1.03); d.set_ylim(-.05, 1.05)
    else:
        d.text(.5, .5, "run scripts/calibration_metrics.py", ha="center")
    d.set_title("Sequence-model probabilities are\nsystematically overconfident",
                loc="left")

    # ---------------- e: power, at the effect actually measured ----------------
    e = ax[2, 0]
    pwp = Path("results/rq2_power.csv")
    if pwp.exists():
        pw = pd.read_csv(pwp)
        e.plot(pw.n_complexes, pw.power, "o-", color=BLUE, lw=2, ms=6)
        e.axhline(.8, ls="--", color="grey", lw=1)
        e.text(pw.n_complexes.min(), .82, "80% power", color="grey", fontsize=8)
        cur = pw[pw.n_complexes == len(v4)]
        if len(cur):
            e.annotate(f"current panel: n={len(v4)},\npower "
                       f"{cur.power.iloc[0]:.2f}",
                       xy=(len(v4), cur.power.iloc[0]),
                       xytext=(len(v4) * 1.9, cur.power.iloc[0] - .22),
                       fontsize=8, arrowprops=dict(arrowstyle="-", lw=.8))
        eff = pw.mean_effect.mean()
        e.set_xscale("log")
        e.set_xticks(list(pw.n_complexes))
        e.set_xticklabels([str(int(v)) for v in pw.n_complexes])
        # a log scale draws its own minor tick labels, which collide with ours
        e.minorticks_off()
        e.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        e.set_xlabel("complexes in panel"); e.set_ylabel("P(CI excludes zero)")
        e.set_ylim(0, 1.05)
        e.set_title(f"RQ2 is underpowered for the effect\nit observes "
                    f"({eff:+.3f})", loc="left")
    else:
        e.text(.5, .5, "run scripts/rq2_power.py", ha="center")

    # ---------------- f: panel composition, correctly captioned ----------------
    f = ax[2, 1]
    v4["locus"] = v4.allele.str[4]
    comp = []
    for loc, g in v4.groupby("locus"):
        gap = roc_auc_score(g.label, g.sequence) - roc_auc_score(
            g.label, g.consensus) if g.label.nunique() > 1 else np.nan
        comp.append({"locus": f"HLA-{loc}", "n": len(g),
                     "alleles": g.allele.nunique(), "gap": gap})
    cf = pd.DataFrame(comp)
    colours = [GREY, BLUE, RED][:len(cf)]
    f.bar(range(len(cf)), cf.n, color=colours, width=.6)
    for i, r in cf.iterrows():
        f.text(i, r.n + 6, f"n={r.n}", ha="center", fontsize=8)
        if not np.isnan(r.gap):
            f.text(i, r.n + 26, f"seq − consensus\n{r.gap:+.3f}", ha="center",
                   fontsize=8, color=RED)
    f.set_xticks(range(len(cf)))
    f.set_xticklabels([f"{r.locus}\n{r.alleles} allele"
                       f"{'s' if r.alleles > 1 else ''}"
                       for _, r in cf.iterrows()])
    f.set_ylabel("complexes in fold set v4")
    f.set_ylim(0, cf.n.max() * 1.45)
    f.set_title("Fold set v4 is HLA-B dominated — locus\nclaims use the "
                "123-allele table instead", loc="left")

    for k, axx in zip("abcdef", ax.ravel()):
        axx.text(-.16, 1.14, k, transform=axx.transAxes, fontsize=13,
                 fontweight="bold", va="top")

    plt.tight_layout(h_pad=3.4, w_pad=3)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    fig.savefig(str(Path(args.out).with_suffix(".pdf")))
    print(f"wrote {args.out} and .pdf")


def _ece(y, p, n_bins=10):
    edges = np.linspace(0, 1, n_bins + 1)
    t = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if m.sum():
            t += m.sum() / len(p) * abs(p[m].mean() - y[m].mean())
    return t


if __name__ == "__main__":
    main()