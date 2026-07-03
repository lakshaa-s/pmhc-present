"""A*02:01 ablation — Condition B: does performance depend on similar alleles?

Condition A (ablation_a2.py) starved A*02:01 only, keeping its A*02 siblings in
training, and found AUROC barely moved — suggesting the pan-allele model borrows from
related alleles. This script tests that directly with a 2x2:

    A*02:01 data   |  A*02 family in training  |  question
    ---------------+---------------------------+------------------------------------
    full           |  present                  |  normal baseline
    starved        |  present                  |  Condition A (transfer available)
    full           |  removed                  |  does losing siblings hurt on its own?
    starved        |  removed                  |  starved AND no transfer -> should crater

"Family" = all HLA-A*02:xx alleles (a tight motif-similarity block per MHCMotifAtlas).
The A*02:01 *test set* is always the same held-out rows, so the four cells are directly
comparable. If only the starved+removed cell collapses, that isolates the mechanism:
A*02:01 survives scarcity ONLY when a motif-similar allele is represented in training.

(HLA-E, also motif-similar to A*02, was filtered out of this dataset as non-classical,
so it isn't in the removal set here — noted as a cross-locus follow-up.)
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
    clusters = hamming_cluster(df[peptide_col].tolist(), df[allele_col].tolist())
    rng = np.random.default_rng(seed)
    uniq = np.unique(clusters)
    n_val = max(1, int(round(len(uniq) * frac_val)))
    val_clusters = set(rng.choice(uniq, size=n_val, replace=False))
    return np.array([c in val_clusters for c in clusters])


def build_ds(df, pseudo, peptide_col, allele_col, label_col):
    return PeptideMHCDataset.from_frame(
        df, pseudo, peptide_col=peptide_col, allele_col=allele_col,
        label_col=label_col, stratum_col=None,
    )


def prepare_train(df_train, target, family, dose, remove_family,
                  allele_col, label_col, seed):
    """Build a training frame for one 2x2 cell.

    - target rows (A*02:01) subsampled to `dose` pos + `dose` neg (1:1), or kept full
      if dose is None.
    - if remove_family: drop every allele in `family` (except target) from training.
    """
    df = df_train
    if remove_family:
        drop = set(family) - {target}
        df = df[~df[allele_col].isin(drop)]

    is_target = df[allele_col] == target
    other = df[~is_target]
    tgt = df[is_target]

    if dose is None:  # full
        return pd.concat([other, tgt], ignore_index=True)

    pos = tgt[tgt[label_col] == 1]
    neg = tgt[tgt[label_col] == 0]
    rng = np.random.RandomState(seed)
    pos_s = pos.sample(n=min(dose, len(pos)), random_state=rng)
    neg_s = neg.sample(n=min(dose, len(neg)), random_state=rng)
    return pd.concat([other, pos_s, neg_s], ignore_index=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--pseudoseq", required=True, nargs="+")
    ap.add_argument("--target", default="HLA-A*02:01")
    ap.add_argument("--family-prefix", default="HLA-A*02:",
                    help="alleles starting with this are the 'family' removed in Condition B")
    ap.add_argument("--starve-dose", type=int, default=115,
                    help="training size (per class) for the 'starved' rows")
    ap.add_argument("--peptide-col", default="peptide")
    ap.add_argument("--allele-col", default="allele")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--out", default="results/ablation_a2_condB.csv")
    args = ap.parse_args()

    print("Loading data + pseudosequences...")
    df = pd.read_csv(args.data)
    pseudo = load_pseudosequences_json(args.pseudoseq)

    family = sorted(df[df[args.allele_col].str.startswith(args.family_prefix)]
                    [args.allele_col].unique())
    print(f"A*02 family in data ({len(family)}): {family}")

    print("Building cluster-based split (once)...")
    is_val = make_split(df, args.allele_col, args.peptide_col)
    df_train_full = df[~is_val].reset_index(drop=True)
    df_val = df[is_val].reset_index(drop=True)
    df_val_t = df_val[df_val[args.allele_col] == args.target].reset_index(drop=True)
    print(f"  {args.target}: fixed test set = {len(df_val_t)} rows "
          f"({int((df_val_t[args.label_col]==1).sum())} pos)")

    val_ds_full = build_ds(df_val, pseudo, args.peptide_col, args.allele_col, args.label_col)
    val_ds_t = build_ds(df_val_t, pseudo, args.peptide_col, args.allele_col, args.label_col)

    # the 2x2 cells: (label, dose, remove_family)
    cells = [
        ("full__family_present",    None,             False),
        ("starved__family_present", args.starve_dose, False),
        ("full__family_removed",    None,             True),
        ("starved__family_removed", args.starve_dose, True),
    ]

    records = []
    cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size)
    total = len(cells) * args.repeats
    run = 0
    for name, dose, remove_family in cells:
        for rep in range(args.repeats):
            run += 1
            seed = 7000 + 100 * cells.index((name, dose, remove_family)) + rep
            df_tr = prepare_train(df_train_full, args.target, family, dose,
                                  remove_family, args.allele_col, args.label_col, seed)
            train_ds = build_ds(df_tr, pseudo, args.peptide_col, args.allele_col, args.label_col)
            model = PresentationNet(NetConfig())
            model, _ = train_model(model, train_ds, val_ds_full, cfg)
            auroc = evaluate(model, val_ds_t, cfg)["overall"]["auroc"]
            n_target = int((df_tr[args.allele_col] == args.target).sum())
            records.append({"cell": name, "repeat": rep, "auroc": auroc,
                            "target_rows": n_target})
            print(f"  [{run}/{total}] {name:26s} rep={rep}  "
                  f"A2 AUROC={auroc:.4f}  (target rows={n_target})")

    res = pd.DataFrame(records)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.out, index=False)

    print("\n=== Condition B 2x2 (mean ± std over repeats) ===")
    summary = res.groupby("cell")["auroc"].agg(["mean", "std", "count"])
    # print in a sensible order
    order = ["full__family_present", "starved__family_present",
             "full__family_removed", "starved__family_removed"]
    for name in order:
        if name in summary.index:
            row = summary.loc[name]
            std = 0.0 if pd.isna(row["std"]) else row["std"]
            print(f"  {name:26s}  AUROC = {row['mean']:.4f} ± {std:.4f}  (n={int(row['count'])})")

    print("\nRead: if only 'starved__family_removed' drops sharply, A*02:01 survives "
          "scarcity only when a motif-similar allele is in training.")
    print(f"Wrote per-run results to {args.out}")


if __name__ == "__main__":
    main()