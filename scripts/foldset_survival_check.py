"""Would the existing fold sets survive a regenerated split?

Decision this answers
---------------------
Fixing the negative-pool sampling artefact means regenerating the labelled table,
which changes the clusters, which changes the split. Fold-set binders were selected
*from the validation partition* of the old split, so under a new split some of them
may land in training -- and a binder the model trained on cannot be used to evaluate
that model.

Structural features do not change (the folds are the same coordinates), so
regeneration costs a retrain and a rescore, not a refold. The only real risk is
attrition: if too many fold-set peptides move into training, the structural
comparison loses power and regeneration is not worth it.

Run this BEFORE retraining. It reports, per fold set and per allele, how many
complexes survive.

Workflow
--------
    # 1. apply the one-line dedup fix to negatives_peptide_pool
    # 2. regenerate to a NEW path -- never overwrite the existing table
    python scripts/prepare_atlas.py --atlas data/raw/all_peptides.txt \\
        --neg-mode peptide-pool --out data/processed/atlas_labelled_v2.csv
    python scripts/make_split.py --data data/processed/atlas_labelled_v2.csv \\
        --out data/processed/split_val_v2.csv
    # 3. check survival
    python scripts/foldset_survival_check.py \\
        --split data/processed/split_val_v2.csv \\
        --fold-sets fold_sets/fold_set_v2.csv fold_sets/fold_set_v4.csv \\
                    fold_sets/fold_set_affinity.csv

Reading the output
------------------
Binders must be in the validation partition. Decoys are not drawn from the Atlas
positives in the same way, so decoy survival is reported for information only --
hard decoys are non-ligands of the target allele by construction and are unaffected
by which partition the target's ligands fall into.

  >90% binder survival   regeneration is cheap; do it
  75-90%                 viable but note the reduced n in Methods
  <75%                   attrition is material; keep the current fold sets and
                         report the confound as analysed instead
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

FOLD_COLS = ["kind", "locus", "allele", "peptide", "extra"]


def normalise_allele(raw: str) -> str:
    """fold-set 'hla_a_02_01' / split 'HLA-A*02:01' -> canonical form."""
    s = str(raw).strip().upper().replace("HLA-", "").replace("HLA_", "")
    s = s.replace("*", "_").replace(":", "_")
    m = re.match(r"^([ABC])_?(\d{2})_?(\d{2,3})", s)
    return f"HLA-{m.group(1)}*{m.group(2)}:{m.group(3)}" if m else str(raw)


def load_fold_set(path: Path) -> pd.DataFrame:
    head = pd.read_csv(path, nrows=1, header=None)
    headered = str(head.iloc[0, 3]).lower() in {"peptide", "sequence"}
    df = (pd.read_csv(path) if headered
          else pd.read_csv(path, header=None, names=FOLD_COLS))
    df.columns = [str(c).strip().lower() for c in df.columns]
    df["allele_norm"] = df["allele"].map(normalise_allele)
    df["is_binder"] = ~df["kind"].astype(str).str.lower().eq("hard")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True, type=Path,
                    help="the NEW split_val.csv")
    ap.add_argument("--fold-sets", required=True, nargs="+", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    split = pd.read_csv(args.split)
    split.columns = [c.strip().lower() for c in split.columns]
    split["allele_norm"] = split["allele"].map(normalise_allele)
    val_keys = set(zip(split.allele_norm, split.peptide.astype(str).str.upper()))
    val_peptides = set(split.peptide.astype(str).str.upper())
    print(f"new split: {len(split):,} validation rows, "
          f"{len(val_peptides):,} unique peptides\n")

    records = []
    for path in args.fold_sets:
        df = load_fold_set(path)
        df["peptide"] = df.peptide.astype(str).str.upper()
        df["in_val"] = [
            (a, p) in val_keys for a, p in zip(df.allele_norm, df.peptide)
        ]
        binders = df[df.is_binder]
        n, kept = len(binders), int(binders.in_val.sum())
        pct = 100 * kept / n if n else float("nan")

        print("=" * 70)
        print(f"{path.name}")
        print("=" * 70)
        print(f"  binders: {kept}/{n} still in validation = {pct:.1f}%")
        decoys = df[~df.is_binder]
        if len(decoys):
            print(f"  decoys:  {len(decoys)} (unaffected by construction; "
                  f"{int(decoys.in_val.sum())} happen to appear in the new split)")

        per = (binders.groupby("allele_norm")["in_val"]
                      .agg(["sum", "count"]).astype(int))
        per["pct"] = (100 * per["sum"] / per["count"]).round(1)
        print("\n  per allele (binders surviving):")
        print(per.rename(columns={"sum": "kept", "count": "n"}).to_string())

        losers = per[per.pct < 75]
        if len(losers):
            print(f"\n  ⚠ alleles below 75%: {list(losers.index)}")

        print("\n  VERDICT: ", end="")
        if pct >= 90:
            print("regeneration is cheap -- attrition is minimal, proceed.")
        elif pct >= 75:
            print("viable; note the reduced complex count in Methods.")
        else:
            print("attrition is MATERIAL. Keep the current fold sets and report\n"
                  "           the confound as analysed rather than regenerating.")
        print()

        records.append({"fold_set": path.name, "n_binders": n,
                        "kept": kept, "pct_kept": round(pct, 1)})

    summary = pd.DataFrame(records)
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(summary.to_string(index=False))
    overall = 100 * summary.kept.sum() / summary.n_binders.sum()
    print(f"\n  overall binder survival: {overall:.1f}%")
    print("\n  Note: even at high survival, the retrained model must be rescored on\n"
          "  these fold sets -- structural features are unchanged, but every\n"
          "  sequence-model number in Chapter 4 moves.")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.out, index=False)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()