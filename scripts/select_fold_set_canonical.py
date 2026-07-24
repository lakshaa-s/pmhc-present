"""Select CANONICAL binders (+ diverse alleles) for a binder-vs-decoy discrimination test.

Why this exists
---------------
`select_fold_set.py` picks maximally-DIVERSE peptides per allele. That is right for
building a training set (cover the space), but wrong for a discrimination test: having
picked one peptide it then picks the most *different* one, which in practice is the least
motif-typical peptide available. You end up testing "atypical/borderline binder" vs
"canonical binder of another allele", which muddies the contrast — and can even make the
decoys better-formed peptides than the binders.

For a discrimination test you want UNAMBIGUOUS positives: peptides that clearly match the
allele's binding motif. This script:

  ALLELES  same as before — greedy max-min on pocket pseudosequence, so the allele set
           still spans the covered->orphan spectrum (anchors can be force-included).
  PEPTIDES per allele, build a position weight matrix (PWM) from that allele's own atlas
           positives, score every candidate peptide by PWM log-odds, keep the top
           `--top-frac` most motif-typical, then take a max-min DIVERSE subset of those.
           => canonical but not near-duplicates.

Output: Boltz/HISTOFold CSV -> pdb_code,locus,allele_slug,peptide_sequence,resolution

Use --peptide-length 9 for HISTOFold (9mers only).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_IDX = {a: i for i, a in enumerate(AA)}


def hamming(a: str, b: str) -> int:
    if len(a) != len(b):
        return max(len(a), len(b))
    return sum(x != y for x, y in zip(a, b))


def greedy_maxmin(items, dist, k, forced=None):
    chosen = list(forced or [])
    pool = [x for x in items if x not in chosen]
    if not chosen and pool:
        chosen.append(pool.pop(0))
    while len(chosen) < k and pool:
        best, best_d = None, -1
        for x in pool:
            d = min(dist(x, c) for c in chosen)
            if d > best_d:
                best, best_d = x, d
        chosen.append(best)
        pool.remove(best)
    return chosen[:k]


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
                out[entry["canonical_allele"]["protein_allele_name"]] = entry["pocket_pseudosequence"]
            except (KeyError, TypeError):
                continue
    return out


def build_pwm(peptides, length, pseudocount=0.5):
    """Position frequency matrix -> log-odds vs background, from an allele's own binders."""
    counts = np.full((length, len(AA)), pseudocount, dtype=float)
    n = 0
    for pep in peptides:
        if len(pep) != length:
            continue
        ok = all(c in AA_IDX for c in pep)
        if not ok:
            continue
        for i, c in enumerate(pep):
            counts[i, AA_IDX[c]] += 1
        n += 1
    if n == 0:
        return None
    freqs = counts / counts.sum(axis=1, keepdims=True)
    background = np.full(len(AA), 1.0 / len(AA))
    return np.log(freqs / background)  # (length, 20) log-odds


def score_peptide(pep, pwm):
    """Sum of PWM log-odds. Higher = more typical of this allele's motif."""
    if pwm is None or len(pep) != pwm.shape[0]:
        return -math.inf
    s = 0.0
    for i, c in enumerate(pep):
        j = AA_IDX.get(c)
        if j is None:
            return -math.inf
        s += pwm[i, j]
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--pseudoseq", required=True, nargs="+")
    ap.add_argument("--peptide-length", type=int, default=9,
                    help="peptide length to use (9 for HISTOFold)")
    ap.add_argument("--n-alleles", type=int, default=5)
    ap.add_argument("--k-peptides", type=int, default=2)
    ap.add_argument("--top-frac", type=float, default=0.10,
                    help="keep this top fraction by motif score before diversifying")
    ap.add_argument("--force-alleles", nargs="*",
                    default=["HLA-B*27:05", "HLA-A*02:01", "HLA-B*07:02",
                             "HLA-C*15:05", "HLA-C*16:02"])
    ap.add_argument("--allele-col", default="allele")
    ap.add_argument("--peptide-col", default="peptide")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--out", default="fold_set_canonical.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    pos = df[df[args.label_col] == 1]
    pos = pos[pos[args.peptide_col].str.len() == args.peptide_length]
    print(f"{len(pos)} positives at length {args.peptide_length}")

    pseudo = load_pseudoseqs(args.pseudoseq)
    candidates = [a for a in sorted(pos[args.allele_col].unique()) if a in pseudo]
    forced = [a for a in args.force_alleles if a in candidates]

    sel_alleles = greedy_maxmin(
        candidates, lambda x, y: hamming(pseudo[x], pseudo[y]),
        args.n_alleles, forced=forced,
    )
    print(f"selected {len(sel_alleles)} alleles: {sel_alleles}\n")

    rows = []
    for allele in sel_alleles:
        peps = pos[pos[args.allele_col] == allele][args.peptide_col].unique().tolist()
        if not peps:
            continue
        pwm = build_pwm(peps, args.peptide_length)
        scored = sorted(((score_peptide(p, pwm), p) for p in peps), reverse=True)
        n_top = max(args.k_peptides, int(len(scored) * args.top_frac))
        top_pool = [p for _, p in scored[:n_top]]
        chosen = greedy_maxmin(top_pool, hamming, args.k_peptides)

        smap = {p: s for s, p in scored}
        med = np.median([s for s, _ in scored])
        print(f"{allele}: {len(peps)} candidates, top {n_top} by motif score")
        for p in chosen:
            pct = 100.0 * sum(1 for s, _ in scored if s < smap[p]) / len(scored)
            print(f"    {p}   score {smap[p]:+.2f}  (median {med:+.2f}, {pct:.0f}th pctile)")
            rows.append({"pdb_code": "NA", "locus": locus_of(allele),
                         "allele_slug": allele_to_slug(allele),
                         "peptide_sequence": p, "resolution": "NA"})

    out = pd.DataFrame(rows, columns=["pdb_code", "locus", "allele_slug",
                                      "peptide_sequence", "resolution"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False, header=False)
    print(f"\nWrote {len(out)} canonical-binder folds -> {args.out}")
    print(f"Estimated cost: {len(out)} x $0.05 = ${len(out) * 0.05:.2f}")


if __name__ == "__main__":
    main()
