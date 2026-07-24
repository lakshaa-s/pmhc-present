"""Select a maximally-diverse allele x peptide set to fold with Boltz.

Chris's instruction: pick alleles far apart from each other and peptides diverse from
each other, to make the most of limited API credits. This does that quantitatively:

  ALLELES  greedy max-min selection by pocket-pseudosequence Hamming distance, over
           alleles present in BOTH the atlas data (so real peptides exist) and the
           foldable set. A few "anchor" alleles are force-included so the
           covered-vs-orphan contrast that RQ1 needs is guaranteed to be present.
  PEPTIDES per allele, greedy max-min selection by Hamming distance among that allele's
           known binders (atlas positives), so no two folded peptides are near-dups.

Output: Boltz CSV  ->  pdb_code,locus,allele_slug,peptide_sequence,resolution
(pdb_code/resolution are placeholders; the folding only uses locus+allele+peptide.)

Run on Beta (needs atlas_labelled.csv + pseudoseq JSONs), then copy the CSV to the
Mac Boltz folder's complexes/ directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def hamming(a: str, b: str) -> int:
    """Hamming distance; unequal lengths -> large distance so they count as 'far'."""
    if len(a) != len(b):
        return max(len(a), len(b))
    return sum(x != y for x, y in zip(a, b))


def greedy_maxmin(items, dist, k, forced=None):
    """Greedy max-min diversity: repeatedly add the item farthest (in min-distance)
    from those already chosen. `forced` items are seeded first."""
    chosen = list(forced or [])
    pool = [x for x in items if x not in chosen]
    if not chosen and pool:  # seed with an arbitrary point
        chosen.append(pool.pop(0))
    while len(chosen) < k and pool:
        # pick the pool item whose distance to the nearest chosen is largest
        best, best_d = None, -1
        for x in pool:
            d = min(dist(x, c) for c in chosen)
            if d > best_d:
                best, best_d = x, d
        chosen.append(best)
        pool.remove(best)
    return chosen[:k]


def allele_to_slug(allele: str) -> str:
    """HLA-C*15:05 -> hla_c_15_05"""
    locus = allele.split("-")[1].split("*")[0].lower()
    grp, prot = allele.split("*")[1].split(":")
    return f"hla_{locus}_{grp}_{prot}"


def locus_of(allele: str) -> str:
    return "hla-" + allele.split("-")[1].split("*")[0].lower()


def load_pseudoseqs(paths):
    """Return {allele_name: pseudosequence} from Chris's pocket-pseudoseq JSONs."""
    out = {}
    for p in paths:
        d = json.load(open(p))
        for entry in d.values() if isinstance(d, dict) else d:
            try:
                name = entry["canonical_allele"]["protein_allele_name"]
                seq = entry["pocket_pseudosequence"]
                out[name] = seq
            except (KeyError, TypeError):
                continue
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="atlas_labelled.csv")
    ap.add_argument("--pseudoseq", required=True, nargs="+",
                    help="pocket-pseudoseq JSONs (for allele-distance)")
    ap.add_argument("--n-alleles", type=int, default=20)
    ap.add_argument("--k-peptides", type=int, default=5)
    ap.add_argument("--force-alleles", nargs="*",
                    default=["HLA-B*27:05", "HLA-A*02:01", "HLA-B*07:02",
                             "HLA-C*15:05", "HLA-C*16:02", "HLA-C*12:04",
                             "HLA-C*12:03", "HLA-C*16:01"],
                    help="alleles guaranteed in the set (covered + orphan anchors)")
    ap.add_argument("--allele-col", default="allele")
    ap.add_argument("--peptide-col", default="peptide")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--out", default="complexes/hla_class_i.csv")
    ap.add_argument("--peptide-length", type=int, default=None,
                    help="restrict to peptides of exactly this length (e.g. 9 for HISTOFold)")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    pos = df[df[args.label_col] == 1]
    if args.peptide_length:
        pos = pos[pos[args.peptide_col].str.len() == args.peptide_length]
        print(f"restricted to {args.peptide_length}mers: {len(pos)} positives")
    pseudo = load_pseudoseqs(args.pseudoseq)

    # candidate alleles: have atlas positives AND a pseudosequence
    atlas_alleles = sorted(pos[args.allele_col].unique())
    candidates = [a for a in atlas_alleles if a in pseudo]
    print(f"{len(candidates)} candidate alleles (atlas positives + pseudoseq)")

    forced = [a for a in args.force_alleles if a in candidates]
    missing = [a for a in args.force_alleles if a not in candidates]
    if missing:
        print(f"  note: forced alleles not in candidate pool (skipped): {missing}")

    # select N maximally-distant alleles by pseudosequence Hamming, seeding the forced ones
    sel_alleles = greedy_maxmin(
        candidates, lambda x, y: hamming(pseudo[x], pseudo[y]),
        args.n_alleles, forced=forced,
    )
    print(f"selected {len(sel_alleles)} alleles:")
    for a in sel_alleles:
        print(f"    {a}  (slug {allele_to_slug(a)})")

    # for each allele, K maximally-diverse known-binder peptides
    rows = []
    for allele in sel_alleles:
        peps = pos[pos[args.allele_col] == allele][args.peptide_col].unique().tolist()
        if not peps:
            continue
        chosen = greedy_maxmin(peps, hamming, args.k_peptides)
        for pep in chosen:
            rows.append({
                "pdb_code": "NA",
                "locus": locus_of(allele),
                "allele_slug": allele_to_slug(allele),
                "peptide_sequence": pep,
                "resolution": "NA",
            })

    out = pd.DataFrame(rows, columns=["pdb_code", "locus", "allele_slug",
                                      "peptide_sequence", "resolution"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False, header=False)  # Chris's code reads headerless rows
    print(f"\nWrote {len(out)} fold rows ({len(sel_alleles)} alleles x ~{args.k_peptides} peptides) -> {args.out}")
    print(f"Estimated cost: {len(out)} x $0.05 = ${len(out) * 0.05:.2f}")


if __name__ == "__main__":
    main()