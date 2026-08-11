"""How much of the 85.3% cross-allele crossover is actually exploitable?

Context
-------
`audit_dataset.py` reports that 85.3% of validation peptides also appear in the
training partition under some other allele. Taken alone that number looks severe.
It is not, and the reason is the negative construction.

Under `--neg-mode peptide-pool`, negatives are real eluted ligands of *other*
alleles. So a peptide that is a positive for its own allele is simultaneously a
negative for the alleles it was sampled against. A model that memorises peptide
identity therefore learns nothing usable: the same peptide carries both labels,
and only the allele pseudosequence resolves which applies.

What remains exploitable is *imbalance*. If a validation positive's peptide appears
in training predominantly as a positive (under related alleles), "this peptide is
usually presented" is a usable prior that does not require the allele input. This
script measures that.

Reported per validation row, split by validation label:
  n_train_pos / n_train_neg   occurrences of the same peptide in training, by label
  prior                       n_train_pos / (n_train_pos + n_train_neg)
  AUROC of the prior alone    the headline: how well does peptide-identity-only
                              memorisation score on the validation set?

Interpretation. A prior-alone AUROC near 0.5 means cross-allele crossover carries
essentially no exploitable signal, and the 85.3% figure can be reported as benign
with evidence. Substantially above 0.5 means the trained model's validation AUROC
is partly attributable to peptide memorisation and must be discounted accordingly.

Usage
-----
    python scripts/crossover_label_balance.py \
        --data  data/processed/atlas_labelled.csv \
        --split data/processed/split_val.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--split", required=True, type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    data = pd.read_csv(args.data)
    val_keys = set(map(tuple, pd.read_csv(args.split)[["allele", "peptide"]].to_numpy()))
    is_val = np.fromiter(
        ((a, p) in val_keys for a, p in zip(data.allele, data.peptide)),
        dtype=bool, count=len(data),
    )
    val, train = data[is_val], data[~is_val]
    print(f"train {len(train):,} rows | validation {len(val):,} rows\n")

    counts = (train.groupby(["peptide", "label"]).size()
                   .unstack(fill_value=0).rename(columns={0: "n_neg", 1: "n_pos"}))
    for col in ("n_pos", "n_neg"):
        if col not in counts:
            counts[col] = 0

    v = val.join(counts, on="peptide")
    v[["n_pos", "n_neg"]] = v[["n_pos", "n_neg"]].fillna(0)
    v["n_train"] = v.n_pos + v.n_neg
    seen = v[v.n_train > 0]

    print("=" * 70)
    print("CROSSOVER COMPOSITION")
    print("=" * 70)
    print(f"  validation rows whose peptide occurs in training: "
          f"{len(seen):,} / {len(v):,} = {len(seen) / len(v):.1%}")

    for label, name in ((1, "validation POSITIVES"), (0, "validation NEGATIVES")):
        s = seen[seen.label == label]
        if not len(s):
            continue
        print(f"\n  {name} (n={len(s):,})")
        print(f"    mean training occurrences of the same peptide: {s.n_train.mean():.2f}")
        print(f"    as positive: {s.n_pos.mean():.2f}   as negative: {s.n_neg.mean():.2f}")
        print(f"    mean positive fraction: {(s.n_pos / s.n_train).mean():.3f}")

    print()
    print("=" * 70)
    print("EXPLOITABILITY: peptide-identity-only prior")
    print("=" * 70)
    prior = (seen.n_pos / seen.n_train).to_numpy()
    y = seen.label.to_numpy()
    if len(np.unique(y)) < 2:
        print("  only one class present; cannot score")
    else:
        auroc = roc_auc_score(y, prior)
        print(f"  AUROC of the prior alone: {auroc:.4f}  (n={len(seen):,})")
        if auroc < 0.55:
            print("\n  BENIGN. Peptide identity alone barely separates the classes, so the")
            print("  85.3% crossover is not an exploitable leak. The peptide-pool negative")
            print("  construction is what neutralises it: the same peptide carries both")
            print("  labels, so only the allele input resolves the case. Report this figure")
            print("  in Methods 3.1.4 as the evidence that the limitation is bounded.")
        elif auroc < 0.70:
            print("\n  PARTIAL. Some exploitable imbalance exists. Report the figure and")
            print("  discount the validation AUROC accordingly in Chapter 5.")
        else:
            print("\n  MATERIAL. Peptide identity alone is substantially predictive. The")
            print("  validation AUROC cannot be read as allele-specific discrimination;")
            print("  cross-allele cluster assignment is needed.")

    print("\n  Per-allele prior AUROC (worst 10 alleles):")
    rows = []
    for allele, g in seen.groupby("allele"):
        if g.label.nunique() == 2 and len(g) >= 30:
            rows.append({
                "allele": allele,
                "prior_auroc": roc_auc_score(g.label, g.n_pos / g.n_train),
                "n": len(g),
            })
    per = pd.DataFrame(rows).sort_values("prior_auroc", ascending=False)
    print(per.head(10).to_string(index=False))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        per.to_csv(args.out, index=False)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()