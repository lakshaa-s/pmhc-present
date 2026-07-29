"""Derive anchor positions per allele from motif information content.

The pipeline has assumed anchors sit at P2 and the C-terminus. That holds for
HLA-A*02:01 but not universally -- HLA-B*08:01 carries a 1.93-bit R/K preference
at P5 that the fixed scheme ignores entirely.

Anchors are instead defined as peptide positions whose information content
clears --min-ic bits, derived from the allele's own 9mer ligands.

Positions are written in the indexing convention analyse_pae.py already uses:
0-based from the N-terminus, with -1 denoting the C-terminal residue, so the
existing default is exactly [1, -1].

Usage:
    python scripts/derive_anchors.py \
        --atlas data/processed/atlas_labelled.csv \
        --out data/processed/anchors.json \
        --min-ic 1.0
"""

from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path

import pandas as pd

DEFAULT_ANCHORS = [1, -1]  # P2 and C-terminus -- the previous hardcoded scheme
MIN_PEPTIDES = 50  # below this the PWM is too unstable to trust


def position_ic(peptides: list[str], i: int) -> float:
    """Information content in bits at position i, max log2(20) = 4.32."""
    counts = collections.Counter(p[i] for p in peptides)
    n = sum(counts.values())
    return math.log2(20) + sum((v / n) * math.log2(v / n) for v in counts.values())


def derive(peptides: list[str], min_ic: float, length: int = 9) -> tuple[list[int], list[float]]:
    """Return (anchor positions, per-position IC) for one allele."""
    ics = [position_ic(peptides, i) for i in range(length)]
    pos = [i for i, ic in enumerate(ics) if ic >= min_ic]
    # re-express the final position as -1 so the convention survives other lengths
    return [-1 if i == length - 1 else i for i in pos], ics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlas", default="data/processed/atlas_labelled.csv")
    ap.add_argument("--out", default="data/processed/anchors.json")
    ap.add_argument("--min-ic", type=float, default=1.0)
    ap.add_argument("--length", type=int, default=9)
    args = ap.parse_args()

    df = pd.read_csv(args.atlas)
    df = df[(df.label == 1) & (df.length == args.length)]

    table: dict[str, dict] = {}
    skipped: list[str] = []

    for allele, grp in df.groupby("allele"):
        peps = [p for p in grp.peptide if len(p) == args.length]
        if len(peps) < MIN_PEPTIDES:
            skipped.append(allele)
            continue
        anchors, ics = derive(peps, args.min_ic, args.length)
        table[allele] = {
            "anchors": anchors,
            "ic": [round(x, 3) for x in ics],
            "n_peptides": len(peps),
        }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(
            {"min_ic": args.min_ic, "length": args.length, "alleles": table},
            fh,
            indent=2,
            sort_keys=True,
        )

    # ---- survey: does the fixed P2/C-terminus scheme actually fit? ----
    same, extra, missing_p2, none_found = [], [], [], []
    for allele, rec in table.items():
        a = set(rec["anchors"])
        if a == set(DEFAULT_ANCHORS):
            same.append(allele)
        if not a:
            none_found.append(allele)
        if a - set(DEFAULT_ANCHORS):
            extra.append((allele, sorted(a - set(DEFAULT_ANCHORS)), rec["n_peptides"]))
        if 1 not in a and a:
            missing_p2.append(allele)

    n = len(table)
    print(f"{n} alleles with >= {MIN_PEPTIDES} {args.length}mers "
          f"({len(skipped)} skipped as too sparse)")
    print(f"  matches the hardcoded [P2, C-term]:  {len(same):>4}  ({len(same)/n:.0%})")
    print(f"  has an anchor OUTSIDE that scheme:   {len(extra):>4}  ({len(extra)/n:.0%})")
    print(f"  P2 is NOT an anchor:                 {len(missing_p2):>4}  ({len(missing_p2)/n:.0%})")
    print(f"  no position clears {args.min_ic} bits:        {len(none_found):>4}")

    if extra:
        print(f"\nAlleles whose anchors the current code misses "
              f"(showing up to 25 of {len(extra)}):")
        for allele, ex, npep in sorted(extra, key=lambda r: -r[2])[:25]:
            shown = ", ".join(f"P{i+1}" for i in ex)
            print(f"  {allele:<14} extra: {shown:<14} (n={npep})")

    by_locus: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    for allele in table:
        locus = allele.split("*")[0]
        by_locus[locus][1] += 1
        if any(allele == e[0] for e in extra):
            by_locus[locus][0] += 1
    print("\nBy locus (alleles with anchors outside the fixed scheme):")
    for locus, (bad, tot) in sorted(by_locus.items()):
        print(f"  {locus}: {bad}/{tot} ({bad/tot:.0%})")

    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
