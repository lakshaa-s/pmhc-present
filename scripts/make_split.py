"""Compute the canonical train/validation split ONCE and write it to disk.

Why this exists
---------------
Several scripts need to know which (allele, peptide) pairs are in the validation split —
fold-set selection (so structural benchmarks use unseen peptides) and the sequence-model
scorer (so it can flag leakage). Each was reconstructing the split by calling
`hamming_cluster` itself.

That is unsafe. `hamming_cluster` assigns cluster ids by walking its input in order, so
clustering a *filtered subset* (e.g. positives only, 9mers only) yields completely
different ids than clustering the full table — and therefore a different split, even with
an identical seed and fraction. In practice this meant a "held-out only" fold set drew
~80% of its peptides from the training split.

So: derive the split once here, from the full labelled table, and have every other script
read the resulting file. One definition, no drift.

    python scripts/make_split.py \
        --data data/processed/atlas_labelled.csv \
        --out data/processed/split_val.csv

Output is a CSV of `allele,peptide` for the validation side only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from pmhcpresent.eval.splits import hamming_cluster


def compute_val_mask(df, allele_col, peptide_col, frac_val, seed):
    """Boolean mask over df rows: True = validation. Must match training exactly."""
    clusters = hamming_cluster(df[peptide_col].tolist(), df[allele_col].tolist())
    rng = np.random.default_rng(seed)
    uniq = np.unique(clusters)
    n_val = max(1, round(len(uniq) * frac_val))
    val_clusters = set(rng.choice(uniq, size=n_val, replace=False))
    return np.array([c in val_clusters for c in clusters])


def load_val_pairs(path):
    """Read the saved split back as a set of (allele, peptide) tuples."""
    d = pd.read_csv(path)
    return set(zip(d["allele"], d["peptide"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--allele-col", default="allele")
    ap.add_argument("--peptide-col", default="peptide")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/processed/split_val.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    print(f"{len(df)} rows, {df[args.allele_col].nunique()} alleles")

    print(f"clustering (Hamming, frac_val={args.val_frac}, seed={args.seed})...")
    is_val = compute_val_mask(df, args.allele_col, args.peptide_col,
                              args.val_frac, args.seed)

    val = df.loc[is_val, [args.allele_col, args.peptide_col]].drop_duplicates()
    val.columns = ["allele", "peptide"]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    val.to_csv(args.out, index=False)

    n_val, n_train = int(is_val.sum()), int((~is_val).sum())
    print(f"  validation: {n_val} rows ({100 * n_val / len(df):.1f}%), "
          f"{len(val)} unique (allele, peptide) pairs")
    print(f"  training:   {n_train} rows")
    print(f"\nWrote {args.out}")
    print("All scripts that need the split should read this file rather than "
          "recomputing it — see the module docstring for why.")


if __name__ == "__main__":
    main()
