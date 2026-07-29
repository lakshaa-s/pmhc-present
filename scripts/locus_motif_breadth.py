"""Compare motif structure across HLA loci, with a sample-size-matched control.

Chris's observation was that HLA-C has several overlapping motifs. If that is so,
HLA-C alleles should show *broader* motifs than A or B: information spread over
more positions, with a lower peak, rather than two sharp anchors.

The confound is that HLA-C alleles have systematically fewer observed ligands, and
information content is upward-biased at small n. So every statistic here is also
computed with all alleles subsampled to a common n, repeated --trials times.

Metrics per allele (9mers):
  n_informative : positions with IC >= --min-ic     (breadth)
  peak_ic       : highest single-position IC        (sharpness)
  total_ic      : summed IC across all 9 positions  (total constraint)
  frac_in_peak  : peak_ic / total_ic                (concentration)

A sharp two-anchor motif: low n_informative, high peak, high frac_in_peak.
A broad or overlapping motif: the reverse.

Usage:
    python scripts/locus_motif_breadth.py \
        --atlas data/processed/atlas_labelled.csv \
        --min-ic 1.0 --match-n 200 --trials 20 \
        --out data/processed/locus_motif_breadth.csv
"""

from __future__ import annotations

import argparse
import collections
import math
import random
import statistics as st

import pandas as pd

LENGTH = 9


def ic_profile(peptides: list[str]) -> list[float]:
    out = []
    for i in range(LENGTH):
        counts = collections.Counter(p[i] for p in peptides)
        n = sum(counts.values())
        out.append(math.log2(20) + sum((v / n) * math.log2(v / n) for v in counts.values()))
    return out


def summarise(profile: list[float], min_ic: float) -> dict:
    peak = max(profile)
    total = sum(profile)
    return {
        "n_informative": sum(1 for v in profile if v >= min_ic),
        "peak_ic": peak,
        "total_ic": total,
        "frac_in_peak": peak / total if total else float("nan"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlas", default="data/processed/atlas_labelled.csv")
    ap.add_argument("--out", default="data/processed/locus_motif_breadth.csv")
    ap.add_argument("--min-ic", type=float, default=1.0)
    ap.add_argument("--match-n", type=int, default=200,
                    help="common sample size for the matched-n control")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    df = pd.read_csv(args.atlas)
    df = df[(df.label == 1) & (df.length == LENGTH)]

    rows = []
    for allele, grp in df.groupby("allele"):
        peps = [p for p in grp.peptide if len(p) == LENGTH]
        if len(peps) < args.match_n:
            continue
        rec = {"allele": allele, "locus": allele.split("*")[0], "n_peptides": len(peps)}
        rec.update({f"full_{k}": v for k, v in summarise(ic_profile(peps), args.min_ic).items()})

        # matched-n: same statistics, every allele cut to the same sample size
        trials = [summarise(ic_profile(random.sample(peps, args.match_n)), args.min_ic)
                  for _ in range(args.trials)]
        for k in trials[0]:
            rec[f"matched_{k}"] = st.mean(t[k] for t in trials)
        rows.append(rec)

    out = pd.DataFrame(rows).sort_values(["locus", "allele"])
    out.to_csv(args.out, index=False)

    print(f"{len(out)} alleles with >= {args.match_n} {LENGTH}mers")
    print(f"matched-n control: {args.match_n} peptides, {args.trials} trials per allele\n")

    for label, prefix in [("AS OBSERVED (confounded by n)", "full"),
                          (f"MATCHED n={args.match_n} (the comparison that counts)", "matched")]:
        print(f"=== {label} ===")
        print(f"{'locus':<8} {'alleles':>7} {'median n':>9} "
              f"{'n_informative':>14} {'peak_ic':>9} {'total_ic':>9} {'frac_in_peak':>13}")
        for locus, g in out.groupby("locus"):
            print(f"{locus:<8} {len(g):>7} {int(g.n_peptides.median()):>9} "
                  f"{g[f'{prefix}_n_informative'].median():>14.2f} "
                  f"{g[f'{prefix}_peak_ic'].median():>9.2f} "
                  f"{g[f'{prefix}_total_ic'].median():>9.2f} "
                  f"{g[f'{prefix}_frac_in_peak'].median():>13.3f}")
        print()

    # is the locus difference bigger than the spread within a locus?
    print("Spread within locus (matched n_informative, min / median / max):")
    for locus, g in out.groupby("locus"):
        v = g["matched_n_informative"]
        print(f"  {locus}: {v.min():.1f} / {v.median():.1f} / {v.max():.1f}")

    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
