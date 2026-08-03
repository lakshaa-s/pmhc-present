"""Select an allele panel by motif isolation (Jensen-Shannon nearest-neighbour distance).

Why this axis
-------------
Two earlier designs were rejected. Stratifying on sequence-model AUROC fails
because weak performance and data sparsity are confounded: among alleles with
enough held-out 9mers for a fair fold set, AUROC spans only 0.922-0.999.
Stratifying on anchor information content works statistically but is a downstream
consequence rather than the property of interest.

Both supervisors independently favoured motif dissimilarity. Benny Chain: "this
will test the influence of structural modelling on the most different alleles,
which targets the objective of increasing the coverage of the HLA space. I don't
think performance or data quality should be the primary criteria in the context of
your project." Chris Thorpe selects his MSA alleles on the same basis.

There is also direct evidence from this project that it is the right axis. Across
123 alleles, motif nearest-neighbour distance predicts per-allele AUROC (rho
-0.363, p 3.7e-5) while pseudosequence distance does not (rho -0.021, p 0.82), and
motif distance survives controlling for pseudosequence distance (-0.417). Isolated
alleles perform worse because they cannot borrow signal from well-characterised
neighbours — which is precisely the population that a structural model, needing no
per-allele training data, might help.

Design
------
STRATIFY, DON'T TRUNCATE   Taking the fifteen most isolated alleles would maximise
                           contrast but leave no variation to correlate against.
                           Strata across the nn_dist range preserve the ability to
                           ask whether structural benefit tracks isolation, which is
                           the mechanistic question. Strata are weighted toward the
                           isolated end, where the hypothesis lives.

CANDIDATE FLOOR            An allele still needs enough held-out 9mers for the top
                           decile to be a real subset. The v2 panel's C*15:05 and
                           C*16:02 had 44 and 39, so their "canonical" binders sat
                           at the 69th-77th percentile. Kept as a constraint, not a
                           criterion.

PWM NOISE                  nn_dist is computed from PWMs, and sparse alleles have
                           noisier PWMs, which inflates apparent distance. The
                           subsampling control (data/processed/subsample_pwm_noise.csv)
                           put this at a median 0.013 inflation for n<=400, about 9%
                           of the observed 0.143 spread. The candidate floor also
                           limits exposure. Reported per allele so it can be judged.

Usage:
    python scripts/select_allele_panel_motif.py \
        --motif data/processed/motif_distinctiveness.csv \
        --atlas data/processed/atlas_labelled.csv \
        --split data/processed/split_val.csv \
        --n-alleles 15 --min-candidates 120 \
        --out fold_sets/panel_v4.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--motif", default="data/processed/motif_distinctiveness.csv")
    ap.add_argument("--atlas", default="data/processed/atlas_labelled.csv")
    ap.add_argument("--split", default="data/processed/split_val.csv")
    ap.add_argument("--n-alleles", type=int, default=15)
    ap.add_argument("--min-candidates", type=int, default=120)
    ap.add_argument("--n-strata", type=int, default=4)
    ap.add_argument("--weights", nargs="*", type=float, default=[3, 2, 1, 1],
                    help="stratum quotas from most to least isolated")
    ap.add_argument("--keep", nargs="*",
                    default=["HLA-A*02:01", "HLA-B*07:02", "HLA-B*27:05",
                             "HLA-C*03:04", "HLA-C*15:05", "HLA-C*16:02"],
                    help="already folded; kept regardless of eligibility")
    ap.add_argument("--out", default="fold_sets/panel_v4.txt")
    args = ap.parse_args()

    d = pd.read_csv(args.motif)

    atlas = pd.read_csv(args.atlas)
    atlas = atlas[(atlas.label == 1) & (atlas.length == 9)]
    val = set(map(tuple, pd.read_csv(args.split).values))
    atlas["is_val"] = [(a, p) in val for a, p in zip(atlas.allele, atlas.peptide)]
    held = atlas[atlas.is_val].groupby("allele").size().rename("heldout")
    d = d.merge(held, left_on="allele", right_index=True, how="left")
    d["heldout"] = d.heldout.fillna(0).astype(int)

    eligible = d[d.heldout >= args.min_candidates].copy()
    print(f"{len(d)} alleles; {len(eligible)} with >= {args.min_candidates} "
          f"held-out 9mers")
    print(f"motif nn_dist, all alleles: {d.nn_dist.min():.3f}-{d.nn_dist.max():.3f}")
    print(f"           among eligible: {eligible.nn_dist.min():.3f}-"
          f"{eligible.nn_dist.max():.3f}\n")

    already = [a for a in args.keep if a in set(d.allele)]
    kept = d[d.allele.isin(already)]
    print("already folded:")
    for _, r in kept.sort_values("nn_dist", ascending=False).iterrows():
        flag = "" if r.heldout >= args.min_candidates else "   SPARSE"
        print(f"  {r.allele:<14} nn_dist {r.nn_dist:.3f} (nearest {r.nn_allele:<12}) "
              f"auroc {r.auroc:.3f}  held-out {r.heldout:>5}{flag}")

    need = max(0, args.n_alleles - len(already))
    pool = eligible[~eligible.allele.isin(already)].copy()

    picked, taken, carry = [], set(), 0
    if need and len(pool):
        # strata run from most isolated (high nn_dist) to least
        lo, hi = eligible.nn_dist.min(), eligible.nn_dist.max()
        edges = np.linspace(hi, lo, args.n_strata + 1)   # descending
        w = np.array(args.weights[:args.n_strata], dtype=float)
        if len(w) < args.n_strata:
            w = np.append(w, np.ones(args.n_strata - len(w)))
        quota = np.floor(w / w.sum() * need).astype(int)
        while quota.sum() < need:
            quota[int(np.argmax(w))] += 1

        print()
        for i in range(args.n_strata):
            hi_i, lo_i = edges[i], edges[i + 1]
            stratum = pool[(pool.nn_dist <= hi_i) & (pool.nn_dist > lo_i)
                           & ~pool.allele.isin(taken)]
            want = quota[i] + carry
            # within a stratum, prefer the most isolated
            take = stratum.nlargest(min(want, len(stratum)), "nn_dist")
            carry = want - len(take)
            taken |= set(take.allele)
            picked.append(take)
            short = f"  (short {carry}, carried)" if carry else ""
            print(f"nn_dist {lo_i:.3f}-{hi_i:.3f}: {len(stratum)} eligible, "
                  f"taking {len(take)}{short}")
            for _, r in take.sort_values("nn_dist", ascending=False).iterrows():
                print(f"    {r.allele:<14} nn_dist {r.nn_dist:.3f} "
                      f"(nearest {r.nn_allele:<12}) auroc {r.auroc:.3f}  "
                      f"held-out {r.heldout:>5}")

        if carry:
            rest = pool[~pool.allele.isin(taken)].nlargest(carry, "nn_dist")
            if len(rest):
                print(f"\n{carry} unfilled, taking most isolated remaining:")
                for _, r in rest.iterrows():
                    print(f"    {r.allele:<14} nn_dist {r.nn_dist:.3f}  "
                          f"auroc {r.auroc:.3f}  held-out {r.heldout:>5}")
                picked.append(rest)

    new = pd.concat(picked) if picked else pool.head(0)
    panel = pd.concat([kept, new]).sort_values("nn_dist", ascending=False)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(panel.allele) + "\n")

    print(f"\n=== panel: {len(panel)} alleles, ordered by motif isolation ===")
    print(f"{'allele':<14} {'nn_dist':>8} {'nearest':<14} {'seq AUROC':>10} "
          f"{'IC':>6} {'held-out':>9} {'status':>9}")
    for _, r in panel.iterrows():
        st = "folded" if r.allele in already else "to fold"
        print(f"{r.allele:<14} {r.nn_dist:>8.3f} {r.nn_allele:<14} "
              f"{r.auroc:>10.3f} {r.ic_p2_pomega:>6.2f} {r.heldout:>9} {st:>9}")

    new_only = panel[~panel.allele.isin(already)]
    print(f"\nfull panel:  nn_dist {panel.nn_dist.min():.3f}-{panel.nn_dist.max():.3f}"
          f"  AUROC {panel.auroc.min():.3f}-{panel.auroc.max():.3f}")
    if len(new_only):
        print(f"new alleles: nn_dist {new_only.nn_dist.min():.3f}-"
              f"{new_only.nn_dist.max():.3f}"
              f"  AUROC {new_only.auroc.min():.3f}-{new_only.auroc.max():.3f}")
    print(f"by locus:    {dict(panel.allele.str.split('*').str[0].value_counts())}")

    # how much of the panel's isolation could be PWM noise?
    sparse = panel[panel.heldout < 400]
    if len(sparse):
        print(f"\n{len(sparse)} panel alleles have <400 held-out 9mers, where the "
              f"subsampling control puts PWM noise at ~0.013 of nn_dist:")
        for _, r in sparse.iterrows():
            print(f"    {r.allele:<14} nn_dist {r.nn_dist:.3f}  "
                  f"held-out {r.heldout:>5}")

    n_new = len(new_only)
    print(f"\n{n_new} new alleles x 24 complexes = {n_new * 24} folds")
    print(f"  ESMFold2 ~20s/fold: {n_new * 24 * 20 / 3600:.1f} h")
    print(f"  AF2      ~90s/fold: {n_new * 24 * 90 / 3600:.1f} h")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()