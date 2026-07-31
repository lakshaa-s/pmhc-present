"""Is structure *quality* the bottleneck for binder/decoy discrimination?

King et al. (arXiv:2512.06592) asked this for affinity regression by retraining
Boltz-2-PPI on experimentally determined structures instead of predicted ones. It
did not improve performance, which rules out poor structure prediction as the
limitation and points at the representations themselves.

We cannot swap in experimental structures -- there are none for most of these
complexes -- but we can ask the equivalent question from the other side: within a
fold set, are the complexes the structural feature gets *wrong* the ones the
folding model was least confident about?

If yes, better folds would mean better discrimination and the ceiling is fold
quality. If no, confidence and discrimination are decoupled, and the limitation is
what the representations encode rather than how well they are predicted.

Method
------
Two definitions of "wrong", because a threshold-based one conflates ranking with
classification:

  MARGIN   For each complex, the fraction of opposite-class complexes *within the
           same allele* that it is correctly ordered against. This is exactly the
           per-complex contribution to that allele's AUROC, so it needs no
           threshold and degrades gracefully.

  BINARY   Margin < 0.5, i.e. the complex is on the wrong side of more than half
           its opposite-class comparisons. Reported for readability only.

Confidence metrics are then compared between correctly and incorrectly ordered
complexes, and correlated against the continuous margin.

Usage:
    python scripts/fold_quality_control.py \
        --conf conf_af2_v2.csv --pae pae_af2_v2.csv \
        --feature pae_anchors --label af2
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy import stats

META = {"allele", "peptide", "kind"}


def per_complex_margin(g: pd.DataFrame, score: str) -> pd.Series:
    """Fraction of opposite-class complexes this one is correctly ordered against.

    Mean over all complexes equals the allele's AUROC, so this decomposes AUROC
    into a per-complex contribution.
    """
    pos = g.loc[g.kind == "binder", score].to_numpy()
    neg = g.loc[g.kind == "decoy", score].to_numpy()
    out = np.empty(len(g))
    for i, (s, k) in enumerate(zip(g[score], g.kind)):
        other = neg if k == "binder" else pos
        if len(other) == 0:
            out[i] = np.nan
        elif k == "binder":
            out[i] = np.mean((s > other) + 0.5 * (s == other))
        else:
            out[i] = np.mean((s < other) + 0.5 * (s == other))
    return pd.Series(out, index=g.index)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", required=True, help="confidence feature CSV")
    ap.add_argument("--pae", required=True, help="PAE feature CSV")
    ap.add_argument("--feature", default="pae_anchors",
                    help="discriminating feature to compute margins from")
    ap.add_argument("--label", default="model")
    args = ap.parse_args()

    conf = pd.read_csv(args.conf)
    pae = pd.read_csv(args.pae)
    if args.feature not in pae.columns:
        raise SystemExit(f"{args.feature} not in {args.pae}: {list(pae.columns)}")

    d = conf.merge(pae[["allele", "peptide", args.feature]],
                   on=["allele", "peptide"], how="inner")
    # PAE: lower means more binder-like, so negate for "higher = binder"
    d["_score"] = -d[args.feature]
    print(f"{args.label}: {len(d)} complexes, {d.allele.nunique()} alleles, "
          f"feature {args.feature}\n")

    d["margin"] = (d.groupby("allele", group_keys=False)
                    .apply(lambda g: per_complex_margin(g, "_score"),
                           include_groups=False))
    d = d.dropna(subset=["margin"])
    d["correct"] = (d.margin > 0.5).astype(int)

    print(f"mean margin (= mean per-allele AUROC): {d.margin.mean():.3f}")
    print(f"{int(d.correct.sum())}/{len(d)} complexes correctly ordered "
          f"(margin > 0.5)\n")

    feats = [c for c in conf.columns if c not in META]
    print("=== confidence: correctly vs incorrectly ordered ===")
    print(f"{'feature':<22} {'correct':>9} {'incorrect':>10} {'diff':>9} "
          f"{'p (MWU)':>9}")
    rows = []
    for f in feats:
        a = d.loc[d.correct == 1, f].dropna()
        b = d.loc[d.correct == 0, f].dropna()
        if len(a) < 5 or len(b) < 5:
            continue
        p = stats.mannwhitneyu(a, b).pvalue
        print(f"{f:<22} {a.mean():>9.4f} {b.mean():>10.4f} "
              f"{a.mean() - b.mean():>+9.4f} {p:>9.3f}")
        rows.append((f, a.mean() - b.mean(), p))

    print("\n=== confidence vs continuous margin (Spearman) ===")
    print("positive rho = more confident folds are better discriminated")
    print(f"{'feature':<22} {'rho':>7} {'p':>9}")
    for f in feats:
        sub = d[[f, "margin"]].dropna()
        if len(sub) < 10 or sub[f].nunique() < 3:
            continue
        rho, p = stats.spearmanr(sub[f], sub.margin)
        print(f"{f:<22} {rho:>+7.3f} {p:>9.3f}")

    print("\n=== confidence vs the discriminating score (circularity check) ===")
    print("high |rho| means the feature is a proxy for the score defining margin,")
    print("so any margin difference is feature correlation, not fold quality")
    for f in feats:
        sub = d[[f, "_score"]].dropna()
        if len(sub) < 10 or sub[f].nunique() < 3:
            continue
        rho, _ = stats.spearmanr(sub[f], sub._score)
        print(f"  {f:<22} rho={rho:+.3f}")

    # only count a POSITIVE difference as evidence for the fold-quality story
    sig = [r for r in rows if r[2] < 0.05 and r[1] > 0]
    print()
    if sig:
        print(f"{len(sig)} confidence feature(s) differ significantly: "
              f"{[r[0] for r in sig]}")
        print("-> fold quality may be part of the limitation.")
    else:
        print("No confidence feature distinguishes correctly from incorrectly")
        print("ordered complexes. Confidence and discrimination are decoupled,")
        print("so better folds would not obviously mean better discrimination.")
        print("Same conclusion as King et al. reached for affinity regression by")
        print("retraining on experimental structures.")


if __name__ == "__main__":
    main()