"""Select DECOYS that genuinely do not fit the target allele's groove.

Why this version exists
-----------------------
The first `select_decoys.py` picked peptides from the most motif-DISTANT allele, assuming
a peptide presented by a distant allele won't fit the target. That is mostly true but not
reliable: allele-level distance says nothing about whether an *individual* peptide happens
to carry the target's anchor residues. In practice this leaked likely-binders into the
decoy set (e.g. a peptide with Arg at P2 selected as a "decoy" for HLA-B*27:05, whose
defining anchor is exactly P2-Arg) — i.e. false negatives that blunt the discrimination
test.

This version adds a motif-rejection step:
  1. Build a PWM for the TARGET allele from its own atlas positives.
  2. Take candidate peptides from the most motif-distant donor allele.
  3. REJECT any candidate scoring above `--max-pctile` of the target's real binders — so
     surviving decoys look genuinely atypical for the groove they'll be placed in.
  4. Diversify among the survivors.

Pairs with `select_fold_set_canonical.py` (canonical binders) to give a clean
binder-vs-decoy discrimination set.

Output: Boltz/HISTOFold CSV -> pdb_code,locus,allele_slug,peptide_sequence,resolution
`pdb_code` is set to 'decoy' so fold directories are distinguishable at analysis time.
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


def greedy_maxmin(items, k):
    if len(items) <= k:
        return list(items)
    chosen = [items[0]]
    pool = list(items[1:])
    while len(chosen) < k and pool:
        best, best_d = None, -1
        for x in pool:
            d = min(hamming(x, c) for c in chosen)
            if d > best_d:
                best, best_d = x, d
        chosen.append(best)
        pool.remove(best)
    return chosen


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
    counts = np.full((length, len(AA)), pseudocount, dtype=float)
    n = 0
    for pep in peptides:
        if len(pep) != length or not all(c in AA_IDX for c in pep):
            continue
        for i, c in enumerate(pep):
            counts[i, AA_IDX[c]] += 1
        n += 1
    if n == 0:
        return None
    freqs = counts / counts.sum(axis=1, keepdims=True)
    return np.log(freqs / np.full(len(AA), 1.0 / len(AA)))


def score_peptide(pep, pwm):
    if pwm is None or len(pep) != pwm.shape[0]:
        return -math.inf
    s = 0.0
    for i, c in enumerate(pep):
        j = AA_IDX.get(c)
        if j is None:
            return -math.inf
        s += pwm[i, j]
    return s


def anchor_residues(pwm, positions, top_n=4):
    """Preferred residues at each anchor position (highest PWM log-odds).

    Binding is anchor-dominated, so a decoy must not carry the target's preferred anchor
    residues. The total PWM score alone is too permissive: a strong anchor match gets
    diluted across the other seven positions, so anchors are checked explicitly.
    """
    out = {}
    for p in positions:
        order = np.argsort(pwm[p])[::-1][:top_n]
        out[p] = {AA[i] for i in order}
    return out


def carries_target_anchors(pep, anchors):
    """True if the peptide carries any of the target's preferred residues at its anchors."""
    return any(pep[p] in res for p, res in anchors.items() if p < len(pep))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--pseudoseq", required=True, nargs="+")
    ap.add_argument("--peptide-length", type=int, default=9)
    ap.add_argument("--targets", nargs="+",
                    default=["HLA-B*27:05", "HLA-A*02:01", "HLA-B*07:02",
                             "HLA-C*15:05", "HLA-C*16:02"])
    ap.add_argument("--k-decoys", type=int, default=2)
    ap.add_argument("--max-pctile", type=float, default=25.0,
                    help="reject candidates scoring above this percentile of the TARGET "
                         "allele's real binders (lower = stricter decoys)")
    ap.add_argument("--anchor-top-n", type=int, default=4,
                    help="how many top residues per anchor position count as 'the motif'")
    ap.add_argument("--n-donors", type=int, default=3,
                    help="how many motif-distant donor alleles to draw candidates from")
    ap.add_argument("--allele-col", default="allele")
    ap.add_argument("--peptide-col", default="peptide")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--out", default="decoy_set_clean.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    pos = df[df[args.label_col] == 1]
    pos = pos[pos[args.peptide_col].str.len() == args.peptide_length]
    print(f"{len(pos)} positives at length {args.peptide_length}\n")

    pseudo = load_pseudoseqs(args.pseudoseq)
    candidates_alleles = [a for a in sorted(pos[args.allele_col].unique()) if a in pseudo]

    rows = []
    for target in args.targets:
        if target not in pseudo:
            print(f"  skip {target}: no pseudosequence")
            continue

        # target's own motif + the score distribution of its real binders
        tgt_peps = pos[pos[args.allele_col] == target][args.peptide_col].unique().tolist()
        pwm = build_pwm(tgt_peps, args.peptide_length)
        if pwm is None:
            print(f"  skip {target}: no {args.peptide_length}mers to build PWM")
            continue
        tgt_scores = np.array([score_peptide(p, pwm) for p in tgt_peps])
        cutoff = float(np.percentile(tgt_scores, args.max_pctile))

        # candidate pool: peptides from the N most motif-distant donor alleles
        others = [a for a in candidates_alleles if a != target]
        others.sort(key=lambda a: hamming(pseudo[target], pseudo[a]), reverse=True)
        donors = others[:args.n_donors]

        pool = []
        for d in donors:
            pool.extend(pos[pos[args.allele_col] == d][args.peptide_col].unique().tolist())
        pool = list(dict.fromkeys(pool))  # dedupe, keep order

        # anchor positions: P2 (index 1) and the C-terminus
        anchor_pos = [1, args.peptide_length - 1]
        anchors = anchor_residues(pwm, anchor_pos, top_n=args.anchor_top_n)

        # REJECT candidates that look like binders for the TARGET groove:
        #   (a) carrying the target's preferred anchor residues, or
        #   (b) scoring above the total-score cutoff
        kept = [(score_peptide(p, pwm), p) for p in pool]
        survivors = [p for s, p in kept
                     if s <= cutoff and not carries_target_anchors(p, anchors)]
        rej_anchor = sum(1 for _, p in kept if carries_target_anchors(p, anchors))
        rejected = len(kept) - len(survivors)

        chosen = greedy_maxmin(survivors, args.k_decoys)
        smap = {p: s for s, p in kept}

        anchor_desc = ", ".join(f"P{p+1}:{{{''.join(sorted(r))}}}" for p, r in anchors.items())
        print(f"{target}: donors {donors}")
        print(f"    target anchors {anchor_desc}")
        print(f"    score cutoff ({args.max_pctile:.0f}th pctile) = {cutoff:+.2f}; "
              f"pool {len(pool)}, rejected {rejected} "
              f"({rej_anchor} for carrying target anchors), {len(survivors)} survive")
        for p in chosen:
            print(f"    decoy {p}  target-motif score {smap[p]:+.2f}  (cutoff {cutoff:+.2f})")
            rows.append({"pdb_code": "decoy", "locus": locus_of(target),
                         "allele_slug": allele_to_slug(target),
                         "peptide_sequence": p, "resolution": "NA"})

    out = pd.DataFrame(rows, columns=["pdb_code", "locus", "allele_slug",
                                      "peptide_sequence", "resolution"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False, header=False)
    print(f"\nWrote {len(out)} decoy folds -> {args.out}")
    print(f"Estimated cost: {len(out)} x $0.05 = ${len(out) * 0.05:.2f}")


if __name__ == "__main__":
    main()
