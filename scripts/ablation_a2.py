"""A*02:01 data-ablation experiment (Condition A: starve A*02:01 only).

Tests the RQ1 premise directly: how much does an allele's performance fall as its
training data is starved, holding everything else fixed?

Design
------
- Split once (cluster-based holdout) into train / val. The A*02:01 rows in *val* are
  the FIXED test set — never subsampled, evaluated identically every run.
- For each dose d (number of A*02:01 training examples), subsample A*02:01's train rows
  to d positives + d negatives (1:1, so class balance is held constant). All OTHER
  alleles keep their full training data — this mirrors a rare allele against a rich
  background, and lets the pan-allele model borrow from similar alleles (the thing we
  want to measure).
- Repeat each dose with several random subsets → mean ± std AUROC, so the drop (or
  lack of one) can be judged against noise. This is the answer to "is the gap
  significant?": the error bars come from the experiment itself.
- Retrain a fresh model each run; early-stop on the full val set; report AUROC on the
  A*02:01-only val subset.

Phase 2 (later): repeat at the starved end with structure features added — does
structure recover the lost performance? That is the actual RQ1 hypothesis; this script
establishes the sequence-only baseline curve it will be compared against.

Output: a CSV of (dose, repeat, auroc) plus a printed mean±std summary table.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pmhcpresent.io.pseudoseq import load_pseudosequences_json
from pmhcpresent.train import PeptideMHCDataset, TrainConfig, train_model, evaluate
from pmhcpresent.models.nn import PresentationNet, NetConfig
from pmhcpresent.eval.splits import hamming_cluster


def make_split(df, allele_col, peptide_col, frac_val=0.2, seed=42):
    """Cluster-based holdout: whole clusters go to val so similar peptides don't leak."""
    clusters = hamming_cluster(df[peptide_col].tolist(), df[allele_col].tolist())
    rng = np.random.default_rng(seed)
    uniq = np.unique(clusters)
    n_val = max(1, int(round(len(uniq) * frac_val)))
    val_clusters = set(rng.choice(uniq, size=n_val, replace=False))
    is_val = np.array([c in val_clusters for c in clusters])
    return is_val


def ablate_allele(df_train, target, dose, allele_col, label_col, seed):
    """Return df_train with `target` allele subsampled to `dose` pos + `dose` neg.

    Other alleles untouched. If dose >= available, keeps all of that class.
    """
    is_target = df_train[allele_col] == target
    other = df_train[~is_target]
    tgt = df_train[is_target]
    pos = tgt[tgt[label_col] == 1]
    neg = tgt[tgt[label_col] == 0]
    rng = np.random.RandomState(seed)
    pos_s = pos.sample(n=min(dose, len(pos)), random_state=rng)
    neg_s = neg.sample(n=min(dose, len(neg)), random_state=rng)
    return pd.concat([other, pos_s, neg_s], ignore_index=True)


def build_ds(df, pseudo, peptide_col, allele_col, label_col):
    return PeptideMHCDataset.from_frame(
        df, pseudo, peptide_col=peptide_col, allele_col=allele_col,
        label_col=label_col, stratum_col=None,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--pseudoseq", required=True, nargs="+")
    ap.add_argument("--allele", default="HLA-A*02:01", help="allele to ablate")
    ap.add_argument("--peptide-col", default="peptide")
    ap.add_argument("--allele-col", default="allele")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--doses", type=int, nargs="+",
                    default=[115, 500, 2000, 8000],
                    help="A*02:01 training sizes (per class); FULL is appended automatically")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--out", default="results/ablation_a2.csv")
    args = ap.parse_args()

    print("Loading data + pseudosequences...")
    df = pd.read_csv(args.data)
    pseudo = load_pseudosequences_json(args.pseudoseq)

    # ---- split ONCE; A*02:01 val rows are the fixed test set ----
    print("Building cluster-based split (once)...")
    is_val = make_split(df, args.allele_col, args.peptide_col)
    df_train_full = df[~is_val].reset_index(drop=True)
    df_val = df[is_val].reset_index(drop=True)

    df_val_a2 = df_val[df_val[args.allele_col] == args.allele].reset_index(drop=True)
    n_a2_pos_train = int(((df_train_full[args.allele_col] == args.allele) &
                          (df_train_full[args.label_col] == 1)).sum())
    print(f"  {args.allele}: {n_a2_pos_train} train positives available, "
          f"{len(df_val_a2)} val rows ({int((df_val_a2[args.label_col]==1).sum())} pos) as fixed test set")

    # build the fixed val datasets ONCE
    val_ds_full = build_ds(df_val, pseudo, args.peptide_col, args.allele_col, args.label_col)
    val_ds_a2 = build_ds(df_val_a2, pseudo, args.peptide_col, args.allele_col, args.label_col)

    # dose ladder: requested doses below the cap, then FULL
    doses = sorted([d for d in args.doses if d < n_a2_pos_train]) + [n_a2_pos_train]
    print(f"  dose ladder: {doses}")

    records = []
    cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size)
    total = len(doses) * args.repeats
    run = 0
    for dose in doses:
        for rep in range(args.repeats):
            run += 1
            seed = 1000 * dose + rep
            df_abl = ablate_allele(df_train_full, args.allele, dose,
                                   args.allele_col, args.label_col, seed)
            train_ds = build_ds(df_abl, pseudo, args.peptide_col, args.allele_col, args.label_col)
            model = PresentationNet(NetConfig())
            model, _ = train_model(model, train_ds, val_ds_full, cfg)
            auroc = evaluate(model, val_ds_a2, cfg)["overall"]["auroc"]
            records.append({"dose": dose, "repeat": rep, "auroc": auroc})
            print(f"  [{run}/{total}] dose={dose:6d} rep={rep}  A2 AUROC={auroc:.4f}")

    res = pd.DataFrame(records)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.out, index=False)

    print("\n=== A*02:01 ablation dose-response (mean ± std over repeats) ===")
    summary = res.groupby("dose")["auroc"].agg(["mean", "std", "count"])
    for dose, row in summary.iterrows():
        std = 0.0 if pd.isna(row["std"]) else row["std"]
        print(f"  dose={dose:6d}   AUROC = {row['mean']:.4f} ± {std:.4f}   (n={int(row['count'])})")
    print(f"\nWrote per-run results to {args.out}")


if __name__ == "__main__":
    main()