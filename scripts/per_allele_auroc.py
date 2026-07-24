"""Per-allele AUROC distribution for the trained sequence baseline.

Benny asked to see performance across ALL alleles, not just the 5 representation
bins. This loads the saved baseline model, scores the validation set once, and
computes AUROC separately for each allele — giving the full 123-point distribution
plus a per-bin summary. Writes a CSV (allele, n, n_pos, auroc, peptide_count, bin)
and prints a summary.

Reuses the same split as training so the val set matches the baseline's.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from pmhcpresent.io.pseudoseq import load_pseudosequences_json
from pmhcpresent.train import PeptideMHCDataset, TrainConfig
from pmhcpresent.models.nn import PresentationNet, NetConfig
from pmhcpresent.eval.splits import hamming_cluster
from pmhcpresent.eval.stratified import assign_frequency_bins


def make_split(df, allele_col, peptide_col, frac_val=0.2, seed=42):
    clusters = hamming_cluster(df[peptide_col].tolist(), df[allele_col].tolist())
    rng = np.random.default_rng(seed)
    uniq = np.unique(clusters)
    n_val = max(1, int(round(len(uniq) * frac_val)))
    val_clusters = set(rng.choice(uniq, size=n_val, replace=False))
    return np.array([c in val_clusters for c in clusters])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--pseudoseq", required=True, nargs="+")
    ap.add_argument("--model", required=True, help="path to saved baseline .pt")
    ap.add_argument("--peptide-col", default="peptide")
    ap.add_argument("--allele-col", default="allele")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--count-bins", type=int, nargs="+",
                    default=[0, 1000, 2500, 6500, 12000, 100000])
    ap.add_argument("--count-labels", nargs="+",
                    default=["rare", "low", "medium", "high", "very_high"])
    ap.add_argument("--min-pos", type=int, default=10,
                    help="skip alleles with fewer than this many positive val rows "
                         "(AUROC unstable below this)")
    ap.add_argument("--out", default="results/per_allele_auroc.csv")
    args = ap.parse_args()

    print("Loading data + pseudosequences...")
    df = pd.read_csv(args.data)
    pseudo = load_pseudosequences_json(args.pseudoseq)

    # per-allele positive counts (for binning), same axis as training
    pos_counts = df[df[args.label_col] == 1].groupby(args.allele_col).size()

    print("Rebuilding the split (same seed as training)...")
    is_val = make_split(df, args.allele_col, args.peptide_col)
    df_val = df[is_val].reset_index(drop=True)

    print(f"Loading model from {args.model} ...")
    model = PresentationNet(NetConfig())
    state = torch.load(args.model, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    # score the whole val set once, then split scores by allele
    val_ds = PeptideMHCDataset.from_frame(
        df_val, pseudo, peptide_col=args.peptide_col, allele_col=args.allele_col,
        label_col=args.label_col, stratum_col=None,
    )

    # get per-row predicted probabilities. evaluate() returns overall metrics; here
    # we need raw scores, so run the model directly over the dataset.
    print("Scoring validation set...")
    probs = []
    with torch.no_grad():
        # PeptideMHCDataset is expected to be indexable / batchable like in training;
        # if train.py exposes a predict helper, prefer that. Fallback: simple loop.
        try:
            from pmhcpresent.train import predict_proba
            probs = predict_proba(model, val_ds, TrainConfig())
        except Exception:
            # generic fallback: iterate the dataset's tensors
            loader = torch.utils.data.DataLoader(val_ds, batch_size=512)
            for batch in loader:
                # assume batch is (features..., label) or a dict; take model output
                out = model(batch) if not isinstance(batch, (list, tuple)) else model(*batch[:-1])
                p = torch.sigmoid(out).squeeze(-1)
                probs.append(p.cpu().numpy())
            probs = np.concatenate(probs)
    probs = np.asarray(probs).ravel()

    df_val = df_val.copy()
    df_val["_prob"] = probs

    print("Computing per-allele AUROC...")
    rows = []
    for allele, sub in df_val.groupby(args.allele_col):
        y = sub[args.label_col].to_numpy()
        n_pos = int((y == 1).sum())
        n_neg = int((y == 0).sum())
        if n_pos < args.min_pos or n_neg < args.min_pos:
            continue  # AUROC not meaningful with too few of either class
        auroc = roc_auc_score(y, sub["_prob"].to_numpy())
        pc = int(pos_counts.get(allele, 0))
        rows.append({"allele": allele, "n": len(sub), "n_pos": n_pos,
                     "auroc": auroc, "peptide_count": pc})

    res = pd.DataFrame(rows).sort_values("peptide_count", ascending=False)
    res["bin"] = assign_frequency_bins(
        res["peptide_count"].to_numpy(), args.count_bins, args.count_labels)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.out, index=False)

    print(f"\n=== Per-allele AUROC ({len(res)} alleles with >= {args.min_pos} of each class) ===")
    print(f"  overall spread: min {res.auroc.min():.3f}  median {res.auroc.median():.3f}  "
          f"max {res.auroc.max():.3f}")
    print("\n  by representation bin:")
    for b in args.count_labels:
        s = res[res.bin == b]["auroc"]
        if len(s):
            print(f"    {b:10s}  n={len(s):3d}  AUROC median {s.median():.3f}  "
                  f"[{s.min():.3f} – {s.max():.3f}]")
    print("\n  worst 5 alleles:")
    for _, r in res.nsmallest(5, "auroc").iterrows():
        print(f"    {r.allele:14s}  AUROC {r.auroc:.3f}  (peptides={r.peptide_count})")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()