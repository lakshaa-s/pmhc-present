"""Build a DECOY fold set to test whether Boltz confidence discriminates binders.

For each target allele, we already folded its real binders. This picks DECOYS:
peptides that are genuine binders of a DIFFERENT, motif-distant allele — so they are
real, well-formed peptides that simply should not fit the target's binding groove.

RQ1 test: fold these decoys, then compare (per allele) the target's real-binder
confidence vs its decoy confidence. If binders score clearly higher than decoys, the
structural confidence discriminates. The key question is whether that gap holds for the
ORPHAN alleles (HLA-C) as well as the covered ones — if so, structure discriminates even
where the sequence model failed.

Decoys are drawn from the allele that is MAXIMALLY motif-distant (largest pocket-
pseudosequence Hamming distance) from the target, so this is the easiest, clearest first
test. Output is the Boltz CSV format; the folder names get a '__decoy' tag so they don't
collide with the real-binder folds and are easy to separate at analysis time.

Run on Beta, then copy the CSV to the Mac Boltz folder's complexes/ directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def hamming(a: str, b: str) -> int:
    if len(a) != len(b):
        return max(len(a), len(b))
    return sum(x != y for x, y in zip(a, b))


def allele_to_slug(allele: str) -> str:
    locus = allele.split("-")[1].split("*")[0].lower()
    grp, prot = allele.split("*")[1].split(":")
    return f"hla_{locus}_{grp}_{prot}"


def locus_of(allele: str) -> str:
    return "hla-" + allele.split("-")[1].split("*")[0].lower()


def load_pseudoseqs(paths):
    out = {}
    for p in paths:
        d = json.load(open(p))
        for entry in (d.values() if isinstance(d, dict) else d):
            try:
                name = entry["canonical_allele"]["protein_allele_name"]
                out[name] = entry["pocket_pseudosequence"]
            except (KeyError, TypeError):
                continue
    return out


def diverse_pick(peps, k):
    """k maximally-diverse peptides (greedy max-min Hamming)."""
    if len(peps) <= k:
        return peps
    chosen = [peps[0]]
    pool = peps[1:]
    while len(chosen) < k and pool:
        best, best_d = None, -1
        for x in pool:
            d = min(hamming(x, c) for c in chosen)
            if d > best_d:
                best, best_d = x, d
        chosen.append(best)
        pool.remove(best)
    return chosen


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--pseudoseq", required=True, nargs="+")
    ap.add_argument("--targets", nargs="+",
                    default=["HLA-B*27:05", "HLA-A*02:01", "HLA-B*07:02",
                             "HLA-C*15:05", "HLA-C*16:02"],
                    help="alleles to generate decoys for (the pilot set)")
    ap.add_argument("--k-decoys", type=int, default=2)
    ap.add_argument("--allele-col", default="allele")
    ap.add_argument("--peptide-col", default="peptide")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--out", default="decoy_set.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    pos = df[df[args.label_col] == 1]
    pseudo = load_pseudoseqs(args.pseudoseq)

    candidates = [a for a in pos[args.allele_col].unique() if a in pseudo]

    rows = []
    for target in args.targets:
        if target not in pseudo:
            print(f"  skip {target}: no pseudosequence")
            continue
        # find the most motif-distant allele from the target (max pseudoseq Hamming)
        others = [a for a in candidates if a != target]
        donor = max(others, key=lambda a: hamming(pseudo[target], pseudo[a]))
        dist = hamming(pseudo[target], pseudo[donor])
        # pull that donor's real binders as decoys (diverse among themselves)
        donor_peps = pos[pos[args.allele_col] == donor][args.peptide_col].unique().tolist()
        decoys = diverse_pick(donor_peps, args.k_decoys)
        print(f"{target}: decoys from {donor} (pseudoseq dist {dist}) -> {decoys}")
        for pep in decoys:
            rows.append({
                "pdb_code": "decoy",           # tags the fold as a decoy
                "locus": locus_of(target),
                "allele_slug": allele_to_slug(target),
                "peptide_sequence": pep,
                "resolution": "NA",
            })

    out = pd.DataFrame(rows, columns=["pdb_code", "locus", "allele_slug",
                                      "peptide_sequence", "resolution"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False, header=False)
    print(f"\nWrote {len(out)} decoy folds -> {args.out}")
    print(f"Estimated cost: {len(out)} x $0.05 = ${len(out) * 0.05:.2f}")
    print("Note: decoy folds are the TARGET allele + a WRONG (motif-distant) peptide.")


if __name__ == "__main__":
    main()