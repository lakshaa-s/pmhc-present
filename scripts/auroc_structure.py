"""Turn structural features into a classifier and score them against the sequence model.

The per-allele "gap" tables show whether binders and decoys differ on average. That is
not directly comparable to the sequence model, which is reported as AUROC. This script
treats each structural feature as a binding score and computes AUROC over the
binder/decoy labels — the same metric, so the comparison is like-for-like.

Features are directional: for PAE-type features LOWER means a more confident placement,
so the score is negated before computing AUROC. For contact-type features higher is
assumed better. AUROC is reported per allele and pooled.

Inputs: the CSVs written by analyse_pae.py and extract_geometry.py (both have
allele/peptide/kind columns, so they are merged on those).

  python auroc_structure.py --pae pae_analysis_k6.csv --geometry geometry_k6.csv

A note on interpretation: with a handful of peptides per allele the per-allele AUROCs
are very noisy — a single swapped pair moves them a lot. The pooled figure and the
consistency of direction across alleles carry more weight than any individual value.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# feature -> True if LOWER values indicate binding (so the score is negated)
LOWER_IS_BINDING = {
    "pae_anchors": True,
    "pae_anchor2": True,
    "pae_anchorC": True,
    "pae_pep_mhc": True,
    "n_contacts": False,
    "n_contacts_close": False,
    "contacts_per_res": False,
    "anchor2_contacts": False,
    "anchorC_contacts": False,
    "min_anchor_dist2": True,
    "min_anchor_distC": True,
}


def auroc_for(df, feature):
    """AUROC of one feature over binder(1)/decoy(0) labels, orientation-corrected."""
    sub = df[["kind", feature]].dropna()
    y = (sub["kind"] == "binder").astype(int).to_numpy()
    if y.sum() == 0 or y.sum() == len(y):
        return np.nan, len(y)
    s = sub[feature].to_numpy(dtype=float)
    if LOWER_IS_BINDING.get(feature, False):
        s = -s
    return roc_auc_score(y, s), len(y)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pae", help="output of analyse_pae.py")
    ap.add_argument("--geometry", help="output of extract_geometry.py")
    ap.add_argument("--out", default="structure_auroc.csv")
    args = ap.parse_args()

    frames = []
    if args.pae:
        frames.append(pd.read_csv(args.pae))
    if args.geometry:
        frames.append(pd.read_csv(args.geometry))
    if not frames:
        raise SystemExit("give --pae and/or --geometry")

    if len(frames) == 2:
        df = frames[0].merge(frames[1], on=["allele", "peptide", "kind"], how="outer",
                             suffixes=("", "_geom"))
    else:
        df = frames[0]

    features = [c for c in df.columns if c in LOWER_IS_BINDING]
    print(f"{len(df)} folds, {int((df.kind == 'binder').sum())} binders / "
          f"{int((df.kind == 'decoy').sum())} decoys")
    print(f"features: {features}\n")

    rows = []

    print("=== pooled AUROC (all alleles together) ===")
    for f in features:
        a, n = auroc_for(df, f)
        rows.append({"scope": "pooled", "allele": "ALL", "feature": f,
                     "auroc": a, "n": n})
        print(f"  {f:20s}  AUROC {a:.3f}  (n={n})")

    print("\n=== per-allele AUROC ===")
    for allele, sub in df.groupby("allele"):
        n_b = int((sub.kind == "binder").sum())
        n_d = int((sub.kind == "decoy").sum())
        if n_b == 0 or n_d == 0:
            continue
        line = [f"  {allele:14s} ({n_b}b/{n_d}d)"]
        for f in features:
            a, n = auroc_for(sub, f)
            rows.append({"scope": "per_allele", "allele": allele, "feature": f,
                         "auroc": a, "n": n})
            if f in ("pae_anchors", "n_contacts"):
                line.append(f"{f}={a:.3f}")
        print("  ".join(line))

    res = pd.DataFrame(rows)
    res.to_csv(args.out, index=False)

    best = res[(res.scope == "pooled")].sort_values("auroc", ascending=False).iloc[0]
    print(f"\nbest pooled feature: {best.feature} (AUROC {best.auroc:.3f})")
    print("Sequence-model baseline for comparison: ~0.97 (pooled, full test set).")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
