"""Score fold set v2 with MHCflurry 2.x as an independent sequence baseline.

Why this exists
---------------
Our own network gives 0.921 pooled on fold set v2, with HLA-C occupying the bottom
three positions. That could be a property of sequence-based prediction, or a
property of our implementation. A second, independently developed pan-allele
sequence model distinguishes the two: if MHCflurry also underperforms on HLA-C,
the deficit is about the approach rather than about us.

MHCflurry is a fair comparator for this. On held-out mass-spectrometry data the
MHCflurry 2.0 integrated model outperformed NetMHCpan 4.0 and MixMHCpred 2.0.2,
and an independent benchmark of 18 predictors found it best for class I 9mers.

Training-data overlap
---------------------
MHCflurry's presentation models date from 2020 and its training data is public,
unlike NetMHCpan's. Overlap with our fold set is therefore likely but checkable,
and is reported here where the curated training data is available locally
(`mhcflurry-downloads fetch data_curated`).

Output schema matches results/sequence_v2.csv (allele, peptide, label, in_train,
score) so scripts/auroc_structure.py --sequence-csv works against it unchanged.

Run in the `mhcflurry` conda env:
    conda activate mhcflurry
    python scripts/score_mhcflurry.py \
        --fold-set fold_sets/fold_set_v2.csv \
        --out results/mhcflurry_v2.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score


def slug_to_allele(slug: str) -> str:
    b = slug.split("_")
    if len(b) < 4:
        return slug
    return f"HLA-{b[1].upper()}*{b[2]}:{b[3]}"


def load_fold_set(path: str) -> pd.DataFrame:
    rows = []
    with open(path) as fh:
        for row in csv.reader(fh):
            if len(row) < 4:
                continue
            tag, _locus, slug, peptide = row[0], row[1], row[2], row[3]
            rows.append({"allele": slug_to_allele(slug), "peptide": peptide,
                         "label": 0 if tag in ("decoy", "hard") else 1})
    return pd.DataFrame(rows)


def training_peptides() -> set[str] | None:
    """MHCflurry's curated training peptides, if that download is present."""
    try:
        from mhcflurry.downloads import get_path
        p = Path(get_path("data_curated", "curated_training_data.csv.bz2",
                          test_exists=False))
        if not p.exists():
            return None
        d = pd.read_csv(p)
        col = next((c for c in ("peptide", "Peptide") if c in d.columns), None)
        return set(d[col]) if col else None
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold-set", default="fold_sets/fold_set_v2.csv")
    ap.add_argument("--out", default="results/mhcflurry_v2.csv")
    ap.add_argument("--score", default="presentation",
                    choices=["presentation", "affinity"],
                    help="presentation_score (higher=binder) or affinity (nM, "
                         "lower=binder; negated so higher=binder)")
    args = ap.parse_args()

    fs = load_fold_set(args.fold_set)
    print(f"{len(fs)} complexes, {int(fs.label.sum())} binders / "
          f"{int((1 - fs.label).sum())} decoys, "
          f"{fs.allele.nunique()} alleles")

    from mhcflurry import Class1PresentationPredictor
    predictor = Class1PresentationPredictor.load()
    print(f"MHCflurry presentation predictor loaded\n")

    scores = []
    for allele, g in fs.groupby("allele"):
        res = predictor.predict(peptides=list(g.peptide), alleles=[allele],
                                verbose=0)
        if args.score == "presentation":
            col = next(c for c in res.columns if "presentation_score" in c)
            s = res[col].to_numpy()
        else:
            col = next(c for c in res.columns if "affinity" in c
                       and "percentile" not in c)
            s = -res[col].to_numpy()   # nM: lower is a better binder
        sub = g.copy()
        sub["score"] = s
        scores.append(sub)

    out = pd.concat(scores, ignore_index=True)

    train = training_peptides()
    if train is None:
        out["in_train"] = ""
        print("NOTE: MHCflurry curated training data not downloaded, so overlap "
              "is unknown. `mhcflurry-downloads fetch data_curated` to enable.\n")
    else:
        out["in_train"] = out.peptide.isin(train)
        n = int(out.in_train.sum())
        print(f"{n}/{len(out)} fold-set peptides appear in MHCflurry's curated "
              f"training data ({n / len(out):.0%})\n")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out[["allele", "peptide", "label", "in_train", "score"]].to_csv(
        args.out, index=False)

    print(f"=== MHCflurry on fold set ({args.score}) ===")
    print(f"  pooled         {roc_auc_score(out.label, out.score):.3f}  "
          f"(n={len(out)})")
    for allele, g in out.groupby("allele"):
        if g.label.nunique() > 1:
            print(f"  {allele:<14} {roc_auc_score(g.label, g.score):.3f}  "
                  f"(n={len(g)})")

    if train is not None and out.in_train.nunique() > 1:
        print("\n=== split by MHCflurry training-data overlap ===")
        for seen in [True, False]:
            g = out[out.in_train == seen]
            if len(g) and g.label.nunique() > 1:
                print(f"  in_train={seen}: n={len(g)}  "
                      f"AUROC {roc_auc_score(g.label, g.score):.3f}")

    print(f"\nWrote {args.out}")
    print("Compare against results/sequence_v2.csv (our model, 0.921 pooled).")


if __name__ == "__main__":
    main()