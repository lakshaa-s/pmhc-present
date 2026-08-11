"""Answer two questions about the labelled table that the repository does not record.

1. WHICH NEGATIVE MODE PRODUCED THIS FILE?
   `prepare_atlas.py` can generate negatives two ways, and they are different
   experiments. `proteome` draws random windows from a human proteome FASTA;
   `peptide-pool` draws real eluted ligands observed for *other* alleles. The
   README calls proteome the intended set but its quickstart uses peptide-pool,
   and REPRODUCE.md does not record which was run.

   They are distinguishable after the fact: peptide-pool negatives are, by
   construction, peptides present somewhere in the Atlas. Proteome negatives are
   almost never in the Atlas. Membership rate settles it.

2. HOW MUCH CROSS-ALLELE LEAKAGE DOES THE SPLIT PERMIT?
   `hamming_cluster` clusters within allele, so a peptide can sit in training
   under one allele and validation under another. This is a defensible design
   choice -- a different groove is a different prediction problem -- but the
   magnitude should be stated rather than left open. Reports both exact and
   near-duplicate (Hamming) crossover.

Usage
-----
    python scripts/audit_dataset.py \
        --data data/processed/atlas_labelled.csv \
        --atlas data/raw/all_peptides.txt \
        --split data/processed/split_val.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pmhcpresent.eval.splits import _hamming_le_threshold


def load_atlas_peptides(path: Path) -> set[str]:
    df = pd.read_csv(path, sep="\t")
    df.columns = [c.strip().lower() for c in df.columns]
    return set(df["peptide"].astype(str).str.strip().str.upper())


def report_negative_mode(data: pd.DataFrame, atlas: set[str]) -> None:
    neg = data[data.label == 0]["peptide"]
    pos = data[data.label == 1]["peptide"]
    in_atlas = neg.isin(atlas).mean()
    pos_in_atlas = pos.isin(atlas).mean()

    print("=" * 70)
    print("1. NEGATIVE MODE")
    print("=" * 70)
    print(f"  negatives:                {len(neg):,}")
    print(f"  negatives found in Atlas: {in_atlas:.1%}")
    print(f"  positives found in Atlas: {pos_in_atlas:.1%}  (sanity check, expect ~100%)")

    if in_atlas > 0.90:
        verdict = "peptide-pool"
        note = ("Negatives are real eluted ligands of other alleles. This is a HARDER "
                "and differently confounded benchmark than proteome decoys: a model can "
                "succeed by learning cross-allele motif discrimination rather than "
                "presented-versus-not. Describe it as such in Methods.")
    elif in_atlas < 0.05:
        verdict = "proteome"
        note = ("Negatives are random proteome windows, as the README intends. Some may "
                "carry canonical anchors by chance and thus be false negatives; state "
                "this as a limitation.")
    else:
        verdict = "AMBIGUOUS"
        note = ("Membership rate sits between the two expected regimes. The file may be "
                "a mixture, or built from a different Atlas release than the one given "
                "to --atlas. Investigate before writing Methods 3.1.3.")

    print(f"\n  VERDICT: {verdict}")
    print(f"  {note}")


def report_cross_allele_leakage(data: pd.DataFrame, val_pairs: pd.DataFrame,
                                threshold: float, check_near: bool) -> None:
    val_keys = set(zip(val_pairs["allele"], val_pairs["peptide"]))
    is_val = [(a, p) in val_keys for a, p in zip(data.allele, data.peptide)]
    val = data[pd.Series(is_val, index=data.index)]
    train = data[~pd.Series(is_val, index=data.index)]

    train_peps = set(train.peptide)
    val_peps = val.peptide.unique()

    exact = sum(p in train_peps for p in val_peps)

    print()
    print("=" * 70)
    print("2. CROSS-ALLELE LEAKAGE")
    print("=" * 70)
    print(f"  training rows:            {len(train):,}")
    print(f"  validation rows:          {len(val):,}")
    print(f"  unique validation peptides: {len(val_peps):,}")
    print("\n  EXACT: validation peptides also appearing in training under ANY allele:")
    print(f"    {exact:,} / {len(val_peps):,} = {exact / len(val_peps):.1%}")

    if check_near:
        by_len: dict[int, list[str]] = {}
        for p in train_peps:
            by_len.setdefault(len(p), []).append(p)
        near = 0
        for p in val_peps:
            if p in train_peps:
                near += 1
                continue
            max_diffs = round((1.0 - threshold) * len(p))
            if any(_hamming_le_threshold(p, q, max_diffs)
                   for q in by_len.get(len(p), ())):
                near += 1
        print(f"\n  NEAR-DUPLICATE (Hamming identity >= {threshold}):")
        print(f"    {near:,} / {len(val_peps):,} = {near / len(val_peps):.1%}")
        print("    (this is the O(n*m) pass -- slow; omit with --no-near for a quick run)")

    print("\n  Report the exact figure in Methods 3.1.4. It converts an unbounded")
    print("  limitation into a bounded one, which is the point of stating it.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=Path,
                    help="atlas_labelled.csv")
    ap.add_argument("--atlas", type=Path,
                    help="all_peptides.txt (needed for the negative-mode check)")
    ap.add_argument("--split", type=Path,
                    help="split_val.csv from make_split.py")
    ap.add_argument("--threshold", type=float, default=0.8)
    ap.add_argument("--no-near", action="store_true",
                    help="skip the slow near-duplicate pass")
    args = ap.parse_args()

    data = pd.read_csv(args.data)
    print(f"{len(data):,} rows, {data.allele.nunique()} alleles, "
          f"{(data.label == 1).sum():,} positives / {(data.label == 0).sum():,} negatives\n")

    if args.atlas:
        report_negative_mode(data, load_atlas_peptides(args.atlas))
    else:
        print("(skipping negative-mode check -- pass --atlas)")

    if args.split:
        report_cross_allele_leakage(data, pd.read_csv(args.split),
                                    args.threshold, not args.no_near)
    else:
        print("\n(skipping leakage check -- pass --split)")


if __name__ == "__main__":
    main()