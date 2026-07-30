"""Select an allele panel spanning the sequence-model performance range.

Why this exists
---------------
On fold set v2 (6 alleles) the benefit of adding structure to sequence is perfectly
monotone in sequence performance: -0.125 for HLA-B*27:05 (sequence 1.000) up to
+0.125 for HLA-C*15:05 (sequence 0.806). That is the King et al. complementarity
claim, but it rests on six points with per-allele CIs 0.325 wide.

To test it properly the panel has to span the performance range with enough alleles
to correlate against. This selects that panel.

Two constraints
---------------
CANDIDATE COUNT   The v2 panel's HLA-C*15:05 and C*16:02 had only 44 and 39
                  held-out 9mers, so their "top decile by motif score" was the whole
                  pool and their binders were not canonical (69th-77th percentile).
                  --min-candidates enforces a real selection pool.

SPREAD, NOT COUNT Per-allele AUROC does not correlate with peptide count
                  (rho -0.020 across 123 alleles); it correlates with anchor
                  information content (rho 0.660). So stratify on AUROC directly.
                  The range is narrow (~0.86-1.00) and the low end is sparse, so
                  the bottom stratum is deliberately oversampled.

Usage:
    python scripts/select_allele_panel.py \
        --auroc results/per_allele_auroc.csv \
        --atlas data/processed/atlas_labelled.csv \
        --split data/processed/split_val.csv \
        --n-alleles 15 --min-candidates 120 \
        --out fold_sets/panel_v3.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--auroc", default="results/per_allele_auroc.csv")
    ap.add_argument("--atlas", default="data/processed/atlas_labelled.csv")
    ap.add_argument("--split", default="data/processed/split_val.csv")
    ap.add_argument("--n-alleles", type=int, default=15)
    ap.add_argument("--min-candidates", type=int, default=120,
                    help="minimum held-out 9mers, so the top decile is a real subset")
    ap.add_argument("--keep", nargs="*",
                    default=["HLA-A*02:01", "HLA-B*07:02", "HLA-B*27:05",
                             "HLA-C*03:04", "HLA-C*15:05", "HLA-C*16:02"],
                    help="alleles already folded, kept regardless")
    ap.add_argument("--out", default="fold_sets/panel_v3.txt")
    args = ap.parse_args()

    perf = pd.read_csv(args.auroc)[["allele", "auroc", "peptide_count"]]

    atlas = pd.read_csv(args.atlas)
    atlas = atlas[(atlas.label == 1) & (atlas.length == 9)]
    val = set(map(tuple, pd.read_csv(args.split).values))
    atlas["is_val"] = [(a, p) in val for a, p in zip(atlas.allele, atlas.peptide)]
    held = atlas[atlas.is_val].groupby("allele").size().rename("heldout_9mers")

    d = perf.merge(held, left_on="allele", right_index=True, how="left")
    d["heldout_9mers"] = d.heldout_9mers.fillna(0).astype(int)

    print(f"{len(d)} alleles with a sequence AUROC")
    print(f"AUROC range {d.auroc.min():.3f} - {d.auroc.max():.3f}")
    print(f"held-out 9mers: median {int(d.heldout_9mers.median())}, "
          f"max {int(d.heldout_9mers.max())}\n")

    eligible = d[d.heldout_9mers >= args.min_candidates].copy()
    print(f"{len(eligible)}/{len(d)} alleles have >= {args.min_candidates} "
          f"held-out 9mers")
    if len(eligible) < args.n_alleles:
        print(f"  WARNING: fewer eligible alleles than requested. Lower "
              f"--min-candidates or --n-alleles.")

    already = [a for a in args.keep if a in set(d.allele)]
    print(f"keeping {len(already)} already-folded alleles: {already}\n")

    # how much of the AUROC range do the kept alleles already cover?
    kept = d[d.allele.isin(already)]
    print("already folded:")
    for _, r in kept.sort_values("auroc").iterrows():
        el = "eligible" if r.heldout_9mers >= args.min_candidates else "SPARSE"
        print(f"  {r.allele:<14} auroc {r.auroc:.3f}  "
              f"held-out {r.heldout_9mers:>5}  {el}")

    # stratify the remaining picks across the AUROC range, oversampling the
    # low end where alleles are scarce and the effect is expected to be largest
    need = max(0, args.n_alleles - len(already))
    pool = eligible[~eligible.allele.isin(already)].copy()
    if need and len(pool):
        lo, hi = d.auroc.min(), d.auroc.max()
        # three strata, weighted 2:1:1 toward the low end
        edges = [lo, lo + 0.35 * (hi - lo), lo + 0.7 * (hi - lo), hi + 1e-9]
        weights = [2, 1, 1]
        alloc = np.array(weights, dtype=float) / sum(weights) * need
        alloc = np.floor(alloc).astype(int)
        while alloc.sum() < need:
            alloc[int(np.argmax(weights))] += 1

        picked = []
        for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
            stratum = pool[(pool.auroc >= a) & (pool.auroc < b)]
            k = min(alloc[i], len(stratum))
            # within a stratum, prefer the largest candidate pools
            take = stratum.nlargest(k, "heldout_9mers")
            picked.append(take)
            print(f"\nstratum {a:.3f}-{b:.3f}: {len(stratum)} eligible, "
                  f"taking {len(take)}")
            for _, r in take.sort_values("auroc").iterrows():
                print(f"  {r.allele:<14} auroc {r.auroc:.3f}  "
                      f"held-out {r.heldout_9mers:>5}")
        new = pd.concat(picked) if picked else pool.head(0)
    else:
        new = pool.head(0)

    panel = pd.concat([kept, new]).sort_values("auroc")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(panel.allele) + "\n")

    print(f"\n=== panel: {len(panel)} alleles ===")
    print(f"{'allele':<14} {'seq AUROC':>10} {'held-out':>9} {'status':>10}")
    for _, r in panel.iterrows():
        st = "folded" if r.allele in already else "to fold"
        print(f"{r.allele:<14} {r.auroc:>10.3f} {r.heldout_9mers:>9} {st:>10}")

    n_new = len(panel) - len(already)
    print(f"\nAUROC span {panel.auroc.min():.3f} - {panel.auroc.max():.3f} "
          f"({panel.auroc.max() - panel.auroc.min():.3f} wide)")
    print(f"{n_new} new alleles x 24 complexes = {n_new * 24} folds")
    print(f"  ESMFold2 at ~20s/fold: ~{n_new * 24 * 20 / 3600:.1f} h")
    print(f"  AF2 at ~90s/fold:      ~{n_new * 24 * 90 / 3600:.1f} h")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
