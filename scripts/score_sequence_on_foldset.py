"""Score the sequence model on the SAME complexes used for the structural evaluation.

Why this exists
---------------
The two arms of RQ1 have so far been measured on different data:

  sequence   AUROC ~0.974, over a large held-out set with pooled negatives
  structure  AUROC 0.700-0.911, over 60 designed complexes with constructed decoys

Those numbers are not comparable, so "0.974 vs 0.700" says nothing about whether
structure beats sequence. This script closes that gap: it runs the trained sequence
model over the exact peptide/allele pairs that were folded, producing an AUROC on
identical data to the structural one.

Training-set leakage
--------------------
The fold set was drawn from the atlas, so some of these peptides were very likely in the
sequence model's training split — which would flatter it. Every pair is checked against
the training split and flagged; AUROC is reported both over all pairs and over held-out
pairs only. **The held-out figure is the one to compare against the structural AUROC.**
If too few pairs are held out, the comparison is not usable and the script says so.

Usage
-----
    python scripts/score_sequence_on_foldset.py \
        --data data/processed/atlas_labelled.csv \
        --pseudoseq data/pseudoseq/hla_{a,b,c}.json \
        --model models/rq1_baseline_hamming.pt \
        --fold-set fold_sets/fold_set_60.csv \
        --out results/sequence_on_foldset.csv

Decoy labelling follows the fold-set convention: `pdb_code` of 'decoy' or 'hard' marks a
negative, anything else a positive.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pmhcpresent.io.pseudoseq import load_pseudosequences_json
from pmhcpresent.models.nn import NetConfig, PresentationNet
from pmhcpresent.train import PeptideMHCDataset, TrainConfig
from sklearn.metrics import roc_auc_score


def slug_to_allele(slug: str) -> str:
    """hla_c_15_05 -> HLA-C*15:05"""
    parts = slug.split("_")
    if len(parts) < 4:
        return slug
    return f"HLA-{parts[1].upper()}*{parts[2]}:{parts[3]}"


def load_val_pairs(path):
    """(allele, peptide) pairs on the validation side, from scripts/make_split.py.

    Read from disk rather than recomputed: `hamming_cluster` assigns ids by input
    order, so reconstructing the split from a filtered subset gives a different
    answer than the one the model was actually trained against.
    """
    d = pd.read_csv(path)
    return set(zip(d["allele"], d["peptide"]))


def load_fold_set(path):
    rows = []
    with open(path) as fh:
        for row in csv.reader(fh):
            if len(row) < 4:
                continue
            tag, _locus, slug, peptide = row[0], row[1], row[2], row[3]
            rows.append({
                "allele": slug_to_allele(slug),
                "peptide": peptide,
                "label": 0 if tag in ("decoy", "hard") else 1,
            })
    return pd.DataFrame(rows).drop_duplicates(subset=["allele", "peptide", "label"])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--val-split", required=True,
                    help="CSV of validation pairs from scripts/make_split.py")
    ap.add_argument("--pseudoseq", required=True, nargs="+")
    ap.add_argument("--model", required=True)
    ap.add_argument("--fold-set", required=True, nargs="+",
                    help="one or more fold-set CSVs (binders + decoys)")
    ap.add_argument("--peptide-col", default="peptide")
    ap.add_argument("--allele-col", default="allele")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--out", default="results/sequence_on_foldset.csv")
    args = ap.parse_args()

    fs = pd.concat([load_fold_set(p) for p in args.fold_set], ignore_index=True)
    fs = fs.drop_duplicates(subset=["allele", "peptide", "label"]).reset_index(drop=True)
    print(f"{len(fs)} fold-set pairs "
          f"({int((fs.label == 1).sum())} binders / {int((fs.label == 0).sum())} decoys)")

    # which of these did the sequence model train on?
    val_pairs = load_val_pairs(args.val_split)
    val_peptides = {p for _a, p in val_pairs}
    # Binders: was this exact (allele, peptide) pairing trained on?
    # Decoys: a decoy is a validated peptide of a *different* allele, so the pairing
    # never appears in the split file — what matters is whether the model saw the
    # peptide at all, under any allele.
    fs["in_train"] = [
        (a, p) not in val_pairs if lab == 1 else p not in val_peptides
        for a, p, lab in zip(fs["allele"], fs["peptide"], fs["label"])
    ]
    n_leak = int(fs["in_train"].sum())
    print(f"\n{n_leak}/{len(fs)} pairs are NOT in the validation split "
          f"({100 * n_leak / len(fs):.0f}% leakage)")

    print(f"\nloading model {args.model} ...")
    pseudo = load_pseudosequences_json(args.pseudoseq)
    model = PresentationNet(NetConfig())
    model.load_state_dict(torch.load(args.model, map_location="cpu"))
    model.eval()

    ds = PeptideMHCDataset.from_frame(
        fs, pseudo, peptide_col="peptide", allele_col="allele",
        label_col="label", stratum_col=None,
    )
    from pmhcpresent.train.trainer import _predict_proba, select_device

    cfg = TrainConfig()
    device = select_device(cfg.device)
    model.to(device)
    probs = np.asarray(_predict_proba(model, ds, device, cfg.batch_size)).ravel()

    # dataset may drop rows with no pseudosequence; align lengths defensively
    if len(probs) != len(fs):
        print(f"  WARNING: model scored {len(probs)} of {len(fs)} pairs "
              "(rows without a pseudosequence are dropped)")
        fs = fs.iloc[: len(probs)].copy()
    fs["score"] = probs

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fs.to_csv(args.out, index=False)

    def auroc(sub, label):
        y = sub["label"].to_numpy()
        if y.sum() == 0 or y.sum() == len(y):
            print(f"  {label:28s} n/a (only one class present, n={len(sub)})")
            return
        a = roc_auc_score(y, sub["score"].to_numpy())
        print(f"  {label:28s} AUROC {a:.3f}  "
              f"(n={len(sub)}, {int(y.sum())}b/{int(len(y) - y.sum())}d)")

    print("\n=== sequence model on the fold set ===")
    auroc(fs, "all pairs")
    held = fs[~fs["in_train"]]
    auroc(held, "held-out pairs only")

    print("\n=== per allele (held-out pairs) ===")
    for allele, sub in held.groupby("allele"):
        auroc(sub, allele)

    print("\nCompare the HELD-OUT figure against the structural AUROC on the same set.")
    print("Pairs seen in training inflate the sequence model and should be excluded.")
    if len(held) < 20:
        print("WARNING: too few held-out pairs for a meaningful comparison — "
              "consider selecting a fold set restricted to validation-split peptides.")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
