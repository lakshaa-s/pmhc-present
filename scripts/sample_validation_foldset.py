"""Sample a balanced per-allele slice of the validation split, in fold-set format.

Why
---
The claim that motif isolation predicts per-allele performance is well powered for
this project's own model — rho -0.291 across 123 alleles — and badly powered for the
external predictors, which have only been scored on benchmark fold sets of six and
nine alleles. At fifteen alleles the attenuation ceiling on any observable
correlation is about 0.30, so the observed trends (NetMHCpan -0.638, MHCflurry
-0.284) cannot be distinguished from noise however suggestive their agreement.

The fix is more alleles, not more peptides per allele. Sampling k pairs from each of
the 123 validation alleles gives 123 points instead of 15. At k=200 the per-allele
standard error falls from roughly 0.075 to near 0.03, which lifts the attenuation
ceiling from about 0.30 to about 0.60 — enough to detect an effect of the size
already measured.

Output is written in **fold-set format** (`tag,locus,allele_slug,peptide,note`), so
`score_netmhcpan.py`, `score_mhcflurry.py` and `score_mixmhcpred.py` consume it with
no modification via their existing `--fold-set` argument.

What this sample is and is not
-------------------------------
These are the *validation split's* negatives — peptides observed for other alleles,
drawn from the peptide pool — not the anchor-matched decoys of fold sets v2 and v4.
Three consequences follow and all should be stated wherever the result is used.

The task is easier than the fold-set task, so absolute AUROCs here are not comparable
with fold-set AUROCs. Only the *between-allele* pattern is the quantity of interest.

These negatives carry the crossover artefact documented in the project's methods: the
same peptide can appear as a positive for one allele and a negative for another. That
affects all models equally, including the external ones, but it means an absolute
figure from this sample is not a clean estimate of anything.

And this is exactly the data the project's own -0.291 was computed on, which is the
point: it makes the external comparison like-for-like rather than comparing a
fold-set correlation against a validation-split one.

Sampling is balanced within allele and seeded, so the same sample is reproducible.
Alleles with fewer than `--min-per-class` of either class are dropped rather than
silently unbalanced, since a per-allele AUROC from three positives is not usable.

Usage:
    python scripts/sample_validation_foldset.py \
        --data data/processed/atlas_labelled_v2.csv \
        --split data/processed/split_val_v2.csv \
        --k 200 --length 9 \
        --out fold_sets/validation_sample_123.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd


def slug(allele: str) -> str:
    """HLA-A*02:01 -> hla_a_02_01, matching the fold-set convention."""
    locus, rest = allele.split("*")
    f1, f2 = rest.split(":")[:2]
    return f"hla_{locus[-1].lower()}_{f1}_{f2}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--k", type=int, default=200,
                    help="pairs per allele, split evenly between classes")
    ap.add_argument("--length", type=int, default=9)
    ap.add_argument("--min-per-class", type=int, default=15,
                    help="drop alleles with fewer than this of either class")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="fold_sets/validation_sample_123.csv")
    args = ap.parse_args()

    d = pd.read_csv(args.data)
    v = pd.read_csv(args.split)
    d = d.merge(v, on=["allele", "peptide"])
    d = d[d.peptide.str.len() == args.length]
    print(f"{len(d):,} validation {args.length}mers across "
          f"{d.allele.nunique()} alleles")

    rng = np.random.default_rng(args.seed)
    half = args.k // 2
    rows, kept, dropped = [], [], []

    for allele, g in d.groupby("allele"):
        pos = g[g.label == 1]
        neg = g[g.label == 0]
        if len(pos) < args.min_per_class or len(neg) < args.min_per_class:
            dropped.append((allele, len(pos), len(neg)))
            continue
        npos = min(half, len(pos))
        nneg = min(half, len(neg))
        n = min(npos, nneg)          # keep it balanced within allele
        ps = pos.sample(n=n, random_state=int(rng.integers(1e9)))
        ns = neg.sample(n=n, random_state=int(rng.integers(1e9)))
        s = slug(allele)
        loc = f"hla-{allele.split('*')[0][-1].lower()}"
        for p in ps.peptide:
            rows.append(["NA", loc, s, p, "val_pos"])
        for p in ns.peptide:
            rows.append(["hard", loc, s, p, "val_neg"])
        kept.append((allele, n))

    if dropped:
        print(f"\ndropped {len(dropped)} alleles with <{args.min_per_class} of a "
              f"class: {[a for a, _, _ in dropped[:6]]}"
              f"{' ...' if len(dropped) > 6 else ''}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        csv.writer(fh, lineterminator="\n").writerows(rows)

    ns = [n for _, n in kept]
    print(f"\nwrote {len(rows):,} rows to {args.out}")
    print(f"  {len(kept)} alleles, {min(ns)}-{max(ns)} pairs per class each")
    print(f"  median {int(np.median(ns))} per class")

    # what this buys, in the terms that matter for the analysis
    se_here = 0.5 / np.sqrt(np.median(ns))       # rough SE of AUROC at this n
    sd_signal = 0.024
    ceil_here = sd_signal / np.sqrt(sd_signal ** 2 + se_here ** 2)
    ceil_fold = sd_signal / np.sqrt(sd_signal ** 2 + 0.075 ** 2)
    print(f"\n  approximate per-allele AUROC SE: {se_here:.3f} "
          f"(fold sets: 0.075)")
    print(f"  attenuation ceiling on a between-allele correlation: "
          f"~{ceil_here:.2f} (fold sets: ~{ceil_fold:.2f})")
    print(f"  and {len(kept)} points rather than 15")

    print(f"""
Score it with the existing baseline scripts, unchanged:

  python scripts/score_netmhcpan.py --fold-set {args.out} \\
      --out results/netmhcpan_val123.csv
  python scripts/score_mhcflurry.py --fold-set {args.out} \\
      --score presentation --out results/mhcflurry_val123.csv

Then correlate against motif isolation with external_vs_isolation.py.

NOTE these are peptide-pool negatives, not anchor-matched decoys, so absolute AUROCs
are NOT comparable with fold-set figures. The between-allele pattern is the quantity
of interest. NetMHCpan's training data is not public, so its overlap with this sample
cannot be quantified the way MHCflurry's can — state that wherever the two are
compared.""")


if __name__ == "__main__":
    main()