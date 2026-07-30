"""Score fold set v2 with MixMHCpred 3.0 as a second eluted-ligand baseline.

Why this exists
---------------
MHCflurry gave a split result: its *presentation* predictor reproduced our model's
HLA-C deficit (C alleles bottom three), while its *affinity* predictor did not
(C*03:04 third of six at 0.986). That suggests the deficit tracks training-data
type -- eluted-ligand/immunopeptidomics versus biochemical binding affinity --
rather than the allele itself.

MixMHCpred is the test. It is trained on immunopeptidomics by the Gfeller lab,
who also produce the MHC Motif Atlas our own model trains on. So it sits on the
presentation side, and the hypothesis predicts it should show the deficit too.

Coverage note
-------------
MixMHCpred 3.0 reports a "Closest Allele" line. For our panel it maps five alleles
to themselves at distance 0, but HLA-C*03:04 to C*03:03 -- i.e. it has no model for
C*03:04 and silently substitutes its nearest neighbour. (Our own motif analysis
found C*03:03/C*03:04 to be the closest pair in the 123-allele panel, JSD 0.0048,
so the substitution is reasonable -- but it should be reported, not hidden.)

Score column
------------
%Rank is used by default: it is the percentile against random peptides, normalised
per allele, which is the right choice when comparing across alleles. Lower means a
better binder, so it is negated to keep "higher = binder" consistent with the other
baselines. --score raw uses the unnormalised Score_<allele> column instead.

Usage:
    python scripts/score_mixmhcpred.py \
        --mix-out /tmp/foldset_mix.out \
        --fold-set fold_sets/fold_set_v2.csv \
        --out results/mixmhcpred_v2.csv
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

# fold-set allele slug -> MixMHCpred allele code
SLUG_TO_MIX = {
    "hla_a_02_01": "A0201", "hla_b_07_02": "B0702", "hla_b_27_05": "B2705",
    "hla_c_03_04": "C0304", "hla_c_15_05": "C1505", "hla_c_16_02": "C1602",
}


def slug_to_allele(slug: str) -> str:
    b = slug.split("_")
    return f"HLA-{b[1].upper()}*{b[2]}:{b[3]}" if len(b) >= 4 else slug


def read_mix(path: str) -> tuple[pd.DataFrame, dict[str, str]]:
    """MixMHCpred output table, plus the allele -> closest-allele mapping."""
    closest = {}
    header_idx = None
    lines = Path(path).read_text().splitlines()
    for i, ln in enumerate(lines):
        if ln.startswith("# Closest Allele"):
            for m in re.finditer(r"([A-Z]\d{4})\s*\(([\d.]+)\)", ln):
                closest[m.group(1)] = m.group(2)
        if ln.startswith("Peptide\t") or ln.startswith("Peptide "):
            header_idx = i
            break
    if header_idx is None:
        raise SystemExit(f"No header row found in {path}")
    df = pd.read_csv(path, sep="\t", skiprows=header_idx)
    return df, closest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mix-out", required=True, help="MixMHCpred output file")
    ap.add_argument("--fold-set", default="fold_sets/fold_set_v2.csv")
    ap.add_argument("--out", default="results/mixmhcpred_v2.csv")
    ap.add_argument("--score", default="rank", choices=["rank", "raw"])
    args = ap.parse_args()

    mix, closest_raw = read_mix(args.mix_out)
    allele_cols = [c for c in mix.columns
                   if c.startswith("Score_") and c != "Score_bestAllele"]
    print(f"MixMHCpred output: {len(mix)} peptides, {len(allele_cols)} alleles")

    # the header lists alleles in order but names the *closest* one, so re-pair
    # by position against the requested list
    print("\nallele coverage (closest allele used by MixMHCpred):")
    requested = [c.replace("Score_", "") for c in allele_cols]
    closest_list = list(closest_raw.items())
    for i, req in enumerate(requested):
        if i < len(closest_list):
            got, dist = closest_list[i]
            flag = "" if got == req else f"   <-- SUBSTITUTED (distance {dist})"
            print(f"  {req:<8} -> {got:<8}{flag}")

    rows = []
    with open(args.fold_set) as fh:
        for row in csv.reader(fh):
            if len(row) < 4:
                continue
            tag, _locus, slug, peptide = row[0], row[1], row[2], row[3]
            code = SLUG_TO_MIX.get(slug)
            if code is None:
                raise SystemExit(f"No MixMHCpred code mapped for slug {slug}")
            col = (f"%Rank_{code}" if args.score == "rank" else f"Score_{code}")
            if col not in mix.columns:
                raise SystemExit(f"Column {col} not in MixMHCpred output; "
                                 f"available: {list(mix.columns)[:8]}")
            hit = mix.loc[mix.Peptide == peptide, col]
            if hit.empty:
                raise SystemExit(f"Peptide {peptide} missing from MixMHCpred output")
            val = float(hit.iloc[0])
            rows.append({
                "allele": slug_to_allele(slug),
                "peptide": peptide,
                "label": 0 if tag in ("decoy", "hard") else 1,
                "in_train": "",
                # %Rank: lower = better binder, so negate for "higher = binder"
                "score": -val if args.score == "rank" else val,
            })

    out = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    print(f"\n{len(out)} complexes, {int(out.label.sum())} binders / "
          f"{int((1 - out.label).sum())} decoys")
    print(f"\n=== MixMHCpred on fold set ({args.score}) ===")
    print(f"  pooled         {roc_auc_score(out.label, out.score):.3f}  "
          f"(n={len(out)})")
    for allele, g in out.groupby("allele"):
        if g.label.nunique() > 1:
            print(f"  {allele:<14} {roc_auc_score(g.label, g.score):.3f}  "
                  f"(n={len(g)})")

    print(f"\nWrote {args.out}")
    print("Compare: our model 0.921 | MHCflurry presentation 0.841 | "
          "MHCflurry affinity 0.911")


if __name__ == "__main__":
    main()