"""Select an allele panel spanning both sequence performance and anchor IC.

Design history, because two earlier versions failed for instructive reasons
-------------------------------------------------------------------------
v1 stratified on sequence AUROC over *all* alleles. The bottom stratum came out
empty: alleles with weak sequence performance are also data-sparse, so none had
enough held-out 9mers to build a canonical fold set.

v2 stratified on anchor information content instead, on the reasoning that IC is
the causal driver (rho 0.660 with AUROC across 123 alleles) and is well populated
at the low end among eligible alleles. It filled all strata, but the nine selected
alleles spanned only 0.964-0.983 in AUROC. That was a selection artefact, not a
property of the pool: taking `nlargest(held_out)` within each IC stratum favours
data-rich alleles, which cluster at high AUROC.

The eligible pool is in fact fine on both axes. At min-candidates=120 it spans
AUROC 0.922-0.999 and IC 1.47-3.57, with rho(IC, AUROC) = 0.659 -- stable across
thresholds (0.660 / 0.677 / 0.666 / 0.659 at 0 / 40 / 80 / 120), so the
correlation is not an artefact of including sparse alleles.

This version therefore stratifies on **sequence AUROC**, because that is the
scarcer axis: only about a dozen eligible alleles sit below 0.945, whereas IC is
plentiful throughout. Within each AUROC stratum it picks alleles that spread the
IC range rather than maximising held-out count, so both axes vary. IC will vary
regardless given the 0.66 correlation, but spreading it explicitly guards against
a stratum happening to contain only high-IC alleles.

Usage:
    python scripts/select_allele_panel.py \
        --auroc results/per_allele_auroc.csv \
        --anchors data/processed/anchors.json \
        --atlas data/processed/atlas_labelled.csv \
        --split data/processed/split_val.csv \
        --n-alleles 15 --min-candidates 120 \
        --out fold_sets/panel_v3.txt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def spread_pick(stratum: pd.DataFrame, k: int, col: str) -> pd.DataFrame:
    """Pick k rows spreading `col` as widely as possible (greedy max-min)."""
    if k <= 0 or stratum.empty:
        return stratum.head(0)
    if k >= len(stratum):
        return stratum
    s = stratum.sort_values(col).reset_index(drop=True)
    chosen = [0, len(s) - 1][:k]          # always take both extremes
    while len(chosen) < k:
        best, best_gap = None, -1.0
        for i in range(len(s)):
            if i in chosen:
                continue
            gap = min(abs(s.loc[i, col] - s.loc[j, col]) for j in chosen)
            if gap > best_gap:
                best, best_gap = i, gap
        chosen.append(best)
    return s.loc[sorted(chosen)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--auroc", default="results/per_allele_auroc.csv")
    ap.add_argument("--anchors", default="data/processed/anchors.json")
    ap.add_argument("--atlas", default="data/processed/atlas_labelled.csv")
    ap.add_argument("--split", default="data/processed/split_val.csv")
    ap.add_argument("--n-alleles", type=int, default=15)
    ap.add_argument("--min-candidates", type=int, default=120)
    ap.add_argument("--n-strata", type=int, default=4)
    ap.add_argument("--keep", nargs="*",
                    default=["HLA-A*02:01", "HLA-B*07:02", "HLA-B*27:05",
                             "HLA-C*03:04", "HLA-C*15:05", "HLA-C*16:02"])
    ap.add_argument("--out", default="fold_sets/panel_v3.txt")
    args = ap.parse_args()

    perf = pd.read_csv(args.auroc)[["allele", "auroc", "peptide_count"]]
    with open(args.anchors) as fh:
        anchors = json.load(fh)["alleles"]
    perf["anchor_ic"] = perf.allele.map(
        {a: (r["ic"][1] + r["ic"][-1]) / 2 for a, r in anchors.items()})

    atlas = pd.read_csv(args.atlas)
    atlas = atlas[(atlas.label == 1) & (atlas.length == 9)]
    val = set(map(tuple, pd.read_csv(args.split).values))
    atlas["is_val"] = [(a, p) in val for a, p in zip(atlas.allele, atlas.peptide)]
    held = atlas[atlas.is_val].groupby("allele").size().rename("heldout")
    d = perf.merge(held, left_on="allele", right_index=True, how="left")
    d["heldout"] = d.heldout.fillna(0).astype(int)
    d = d.dropna(subset=["anchor_ic"])

    eligible = d[d.heldout >= args.min_candidates].copy()
    print(f"{len(d)} alleles; {len(eligible)} with >= {args.min_candidates} "
          f"held-out 9mers")
    print(f"eligible AUROC {eligible.auroc.min():.3f}-{eligible.auroc.max():.3f}, "
          f"IC {eligible.anchor_ic.min():.2f}-{eligible.anchor_ic.max():.2f}\n")

    already = [a for a in args.keep if a in set(d.allele)]
    kept = d[d.allele.isin(already)]
    print("already folded:")
    for _, r in kept.sort_values("auroc").iterrows():
        flag = "" if r.heldout >= args.min_candidates else "   SPARSE"
        print(f"  {r.allele:<14} auroc {r.auroc:.3f}  IC {r.anchor_ic:>5.2f}  "
              f"held-out {r.heldout:>5}{flag}")

    need = max(0, args.n_alleles - len(already))
    pool = eligible[~eligible.allele.isin(already)].copy()

    picked, taken, carry = [], set(), 0
    if need and len(pool):
        lo, hi = eligible.auroc.min(), eligible.auroc.max()
        edges = np.linspace(lo, hi + 1e-9, args.n_strata + 1)
        base, rem = divmod(need, args.n_strata)
        quota = [base + (1 if i < rem else 0) for i in range(args.n_strata)]

        print()
        for i in range(args.n_strata):
            a, b = edges[i], edges[i + 1]
            stratum = pool[(pool.auroc >= a) & (pool.auroc < b)
                           & ~pool.allele.isin(taken)]
            want = quota[i] + carry
            take = spread_pick(stratum, min(want, len(stratum)), "anchor_ic")
            carry = want - len(take)
            taken |= set(take.allele)
            picked.append(take)
            short = f"  (short {carry}, carried)" if carry else ""
            print(f"AUROC {a:.3f}-{b:.3f}: {len(stratum)} eligible, "
                  f"taking {len(take)}{short}")
            for _, r in take.sort_values("anchor_ic").iterrows():
                print(f"    {r.allele:<14} auroc {r.auroc:.3f}  "
                      f"IC {r.anchor_ic:>5.2f}  held-out {r.heldout:>5}")

        if carry:
            rest = pool[~pool.allele.isin(taken)].nsmallest(carry, "auroc")
            if len(rest):
                print(f"\n{carry} unfilled, taking lowest-AUROC remaining:")
                for _, r in rest.iterrows():
                    print(f"    {r.allele:<14} auroc {r.auroc:.3f}  "
                          f"IC {r.anchor_ic:>5.2f}  held-out {r.heldout:>5}")
                picked.append(rest)

    new = pd.concat(picked) if picked else pool.head(0)
    panel = pd.concat([kept, new]).sort_values("auroc")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(panel.allele) + "\n")

    print(f"\n=== panel: {len(panel)} alleles ===")
    print(f"{'allele':<14} {'seq AUROC':>10} {'anchor IC':>10} "
          f"{'held-out':>9} {'status':>9}")
    for _, r in panel.iterrows():
        st = "folded" if r.allele in already else "to fold"
        print(f"{r.allele:<14} {r.auroc:>10.3f} {r.anchor_ic:>10.2f} "
              f"{r.heldout:>9} {st:>9}")

    new_only = panel[~panel.allele.isin(already)]
    print(f"\nfull panel:  AUROC {panel.auroc.min():.3f}-{panel.auroc.max():.3f}"
          f"   IC {panel.anchor_ic.min():.2f}-{panel.anchor_ic.max():.2f}")
    if len(new_only):
        print(f"new alleles: AUROC "
              f"{new_only.auroc.min():.3f}-{new_only.auroc.max():.3f}"
              f"   IC {new_only.anchor_ic.min():.2f}-"
              f"{new_only.anchor_ic.max():.2f}")
    print(f"by locus:    {dict(panel.allele.str.split('*').str[0].value_counts())}")
    n_new = len(new_only)
    print(f"\n{n_new} new alleles x 24 complexes = {n_new * 24} folds")
    print(f"  ESMFold2 ~20s/fold: {n_new * 24 * 20 / 3600:.1f} h")
    print(f"  AF2      ~90s/fold: {n_new * 24 * 90 / 3600:.1f} h")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()