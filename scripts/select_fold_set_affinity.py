"""Build a fold set from experimentally measured binding affinities.

Why this exists
---------------
The existing fold set (fold_sets/fold_set_v2.csv) is selected by MHC Motif Atlas
PWM score: binders from the top decile, decoys from below the 25th percentile of
the same score. The consequence is that a position weight matrix alone separates
it at AUROC 1.000, and across five sequence models the correlation between a
model's score and the PWM score tracks its AUROC almost perfectly:

    MixMHCpred           rho 0.909    AUROC 0.999
    NetMHCpan-4.1        rho 0.773    AUROC 0.961
    MHCflurry affinity   rho 0.718    AUROC 0.911
    ours                 rho 0.690    AUROC 0.921
    MHCflurry present.   rho 0.621    AUROC 0.841

So the sequence-model comparison on that benchmark largely measures how closely
each model resembles the selection criterion. This builds an alternative with
none of that structure.

Three properties the PWM-selected set does not have
---------------------------------------------------
NO PWM            Selection uses measured affinity only. No motif scoring at any
                  point, so no predictor is advantaged by resembling the criterion.

REAL NEGATIVES    Decoys are peptides experimentally assayed against the allele and
                  found not to bind (>5000 nM), rather than constructed by matching
                  anchors. This removes the assumption, built into the anchor-matched
                  decoys, about what makes a negative hard.

ATLAS-DISJOINT    Every peptide is absent from the MHC Motif Atlas, so none appears
                  in our model's training data under the target allele.

Coverage limitation
-------------------
Affinity data is not evenly distributed across loci. Peptides measured by affinity
and absent from the Atlas, by allele:

    HLA-A*02:01   17,348        HLA-C*03:04      95
    HLA-B*07:02    5,416        HLA-C*15:05       0
    HLA-B*27:05    3,344        HLA-C*16:02       0

So this benchmark is buildable for HLA-A and HLA-B, marginal for C*03:04, and
impossible for the two rare HLA-C alleles. That is the coverage finding again:
mass spectrometry is untargeted and covers all loci comparably, whereas affinity
assays require deliberate selection and have followed research attention toward
HLA-A. **Results on this fold set therefore cannot address the equity question**,
and it should be reported alongside the PWM-selected set rather than replacing it.

Not comparable to the v2 numbers
--------------------------------
The v2 decoys are anchor-matched and deliberately hard. Experimental non-binders
are whatever happened to be assayed, and are probably easier on average since
researchers tend to test peptides they expect to bind. AUROCs from the two sets
measure different things and should not be compared directly.

Usage (run in the mhcflurry env, which has the curated data):
    python scripts/select_fold_set_affinity.py \
        --binder-nm 50 --decoy-nm 5000 --k 12 \
        --alleles "HLA-A*02:01" "HLA-B*07:02" "HLA-B*27:05" "HLA-C*03:04" \
        --out fold_sets/fold_set_affinity.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd


def allele_to_slug(allele: str) -> str:
    """HLA-C*03:04 -> hla_c_03_04"""
    a = allele.replace("HLA-", "")
    locus, rest = a.split("*")
    field1, field2 = rest.split(":")[:2]
    return f"hla_{locus.lower()}_{field1}_{field2}"


def allele_to_locus(allele: str) -> str:
    return f"hla-{allele.replace('HLA-', '').split('*')[0].lower()}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlas", default="data/processed/atlas_labelled.csv")
    ap.add_argument("--alleles", nargs="+", required=True)
    ap.add_argument("--binder-nm", type=float, default=50.0,
                    help="measured affinity below this is a binder")
    ap.add_argument("--decoy-nm", type=float, default=5000.0,
                    help="measured affinity above this is a non-binder")
    ap.add_argument("--k", type=int, default=12, help="binders and decoys per allele")
    ap.add_argument("--length", type=int, default=9)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="fold_sets/fold_set_affinity.csv")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    from mhcflurry.downloads import get_path
    mf = pd.read_csv(get_path("data_curated", "curated_training_data.csv.bz2",
                              test_exists=False))
    mf = mf[(mf.peptide.str.len() == args.length)
            & (mf.measurement_kind == "affinity")]

    atlas = pd.read_csv(args.atlas)
    atlas_peps = set(atlas[(atlas.label == 1)
                           & (atlas.length == args.length)].peptide)
    print(f"MHCflurry affinity {args.length}mers: {len(mf)}")
    print(f"Atlas {args.length}mer positives: {len(atlas_peps)}\n")

    rows, report = [], []
    for allele in args.alleles:
        g = mf[(mf.allele == allele) & (~mf.peptide.isin(atlas_peps))]

        # binders: measured below the threshold, either exactly or as an upper bound
        b = g[((g.measurement_inequality == "=") & (g.measurement_value < args.binder_nm))
              | ((g.measurement_inequality == "<")
                 & (g.measurement_value <= args.binder_nm))]
        # non-binders: measured above the threshold, exactly or as a lower bound
        d = g[((g.measurement_inequality == "=") & (g.measurement_value > args.decoy_nm))
              | ((g.measurement_inequality == ">")
                 & (g.measurement_value >= args.decoy_nm))]

        b = b.drop_duplicates("peptide")
        d = d.drop_duplicates("peptide")
        # a peptide measured both ways is ambiguous; drop it from both
        both = set(b.peptide) & set(d.peptide)
        if both:
            b = b[~b.peptide.isin(both)]
            d = d[~d.peptide.isin(both)]

        report.append((allele, len(b), len(d), len(both)))
        if len(b) < args.k or len(d) < args.k:
            print(f"  {allele}: only {len(b)} binders / {len(d)} non-binders "
                  f"available, need {args.k} each -- SKIPPED")
            continue

        take_b = b.sample(args.k, random_state=int(rng.integers(1 << 31)))
        take_d = d.sample(args.k, random_state=int(rng.integers(1 << 31)))
        slug, locus = allele_to_slug(allele), allele_to_locus(allele)
        for p in take_b.peptide:
            rows.append(["NA", locus, slug, p, "NA"])
        for p in take_d.peptide:
            rows.append(["hard", locus, slug, p, "NA"])

    print(f"\n{'allele':<14} {'binders':>9} {'non-binders':>12} {'ambiguous':>10}")
    for allele, nb, nd, amb in report:
        print(f"{allele:<14} {nb:>9} {nd:>12} {amb:>10}")

    if not rows:
        raise SystemExit("\nNo allele had enough measurements. Relax the thresholds.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    # csv.writer defaults to \r\n line endings regardless of the newline= setting,
    # and the downstream parsers expect Unix newlines
    with open(args.out, "w", newline="") as fh:
        csv.writer(fh, lineterminator="\n").writerows(rows)

    n_alleles = len({r[2] for r in rows})
    print(f"\nWrote {len(rows)} complexes across {n_alleles} alleles -> {args.out}")
    print(f"  binders  < {args.binder_nm:g} nM")
    print(f"  decoys   > {args.decoy_nm:g} nM")
    print("  no PWM used in selection; all peptides absent from the Atlas")
    print("\nNOTE: AUROCs from this set are not comparable to fold_set_v2, whose")
    print("decoys are anchor-matched and deliberately hard. Report separately.")


if __name__ == "__main__":
    main()