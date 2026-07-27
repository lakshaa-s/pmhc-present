"""Select HARD decoys: real peptides that MATCH the target's anchors but aren't its ligands.

Why this exists
---------------
`select_decoys_clean.py` builds *easy* decoys — it rejects any candidate carrying the
target's preferred anchor residues. That is the right control for a first pass, but it
creates a circularity risk: if a structure model's discrimination is really just "does
this peptide have the right anchor residues", then a high AUROC on anchor-rejected decoys
measures the decoy filter, not binding prediction.

This script builds the adversarial case. Candidates must:
  1. CARRY the target's preferred residues at BOTH anchor positions (P2 and C-terminus),
  2. still score LOW overall against the target's PWM (i.e. the non-anchor positions are
     wrong for this groove),
  3. be real eluted ligands of some *other* allele — never observed on the target.

Real peptides are used rather than synthetic anchor-preserving scrambles because
ESMFold2's backbone is a protein language model: a scrambled sequence is out of
distribution, so high predicted error could reflect implausibility rather than
non-binding. Every peptide here is a genuine MHC ligand; only the groove differs.

Interpretation: if anchor PAE still separates binders from these decoys, the structural
signal is more than anchor matching. If it collapses toward 0.5, the easier result was
largely the decoy filter reflected back.

Caveat: "not observed on the target allele" is weaker than "does not bind the target" —
immunopeptidomics is incomplete, so a few of these may be genuine binders. That makes
this a conservative (pessimistic) test.

Output: Boltz/HISTOFold CSV -> pdb_code,locus,allele_slug,peptide_sequence,resolution
with `pdb_code` set to 'hard' so folds are distinguishable from the easy-decoy set.
"""

from __future__ import annotations

import argparse
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



def load_val_pairs(path):
    """(allele, peptide) pairs on the validation side, from scripts/make_split.py.

    The split is read from disk rather than recomputed. `hamming_cluster` assigns
    cluster ids by walking its input in order, so clustering a filtered subset
    (positives only, one length only) produces a *different* split than clustering
    the full table — same seed or not. Recomputing here silently drew ~80% of
    "held-out" peptides from the training split.
    """
    d = pd.read_csv(path)
    return set(zip(d["allele"], d["peptide"]))


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
    return {p: {AA[i] for i in np.argsort(pwm[p])[::-1][:top_n]} for p in positions}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data", required=True)
    ap.add_argument("--peptide-length", type=int, default=9)
    ap.add_argument(
        "--targets",
        nargs="+",
        default=[
            "HLA-B*27:05",
            "HLA-A*02:01",
            "HLA-B*07:02",
            "HLA-C*15:05",
            "HLA-C*16:02",
        ],
    )
    ap.add_argument("--k-decoys", type=int, default=6)
    ap.add_argument("--anchor-top-n", type=int, default=4)
    ap.add_argument(
        "--max-pctile",
        type=float,
        default=50.0,
        help="candidate must score below this percentile of the target's real binders; "
             "looser than the easy-decoy default because anchor matches raise the score",
    )
    ap.add_argument("--allele-col", default="allele")
    ap.add_argument("--peptide-col", default="peptide")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--val-split",
                    help="CSV of validation (allele, peptide) pairs from "
                         "scripts/make_split.py; restricts selection to unseen peptides")
    ap.add_argument("--out", default="decoy_set_hard.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    pos = df[df[args.label_col] == 1]
    pos = pos[pos[args.peptide_col].str.len() == args.peptide_length]
    print(f"{len(pos)} positives at length {args.peptide_length}\n")

    if args.val_split:
        held = load_val_pairs(args.val_split)
        before = len(pos)
        pos = pos[[(a, p) in held for a, p in
                   zip(pos[args.allele_col], pos[args.peptide_col])]]
        print(f"validation split only: {before} -> {len(pos)} positives "
              f"(from {args.val_split})")

    anchor_pos = [1, args.peptide_length - 1]

    rows = []
    for target in args.targets:
        tgt_peps = pos[pos[args.allele_col] == target][args.peptide_col].unique().tolist()
        pwm = build_pwm(tgt_peps, args.peptide_length)
        if pwm is None:
            print(f"  skip {target}: no {args.peptide_length}mers")
            continue

        anchors = anchor_residues(pwm, anchor_pos, top_n=args.anchor_top_n)
        tgt_scores = np.array([score_peptide(p, pwm) for p in tgt_peps])
        cutoff = float(np.percentile(tgt_scores, args.max_pctile))
        tgt_set = set(tgt_peps)

        # candidates: every 9mer observed on some OTHER allele
        pool = pos[pos[args.allele_col] != target][args.peptide_col].unique().tolist()
        pool = [p for p in pool if p not in tgt_set]

        # REQUIRE both anchors to match the target, but overall score to stay low
        survivors = []
        n_anchor_ok = 0
        for p in pool:
            if not all(p[i] in anchors[i] for i in anchor_pos):
                continue
            n_anchor_ok += 1
            if score_peptide(p, pwm) <= cutoff:
                survivors.append(p)

        chosen = greedy_maxmin(survivors, args.k_decoys)
        anchor_desc = ", ".join(
            f"P{p + 1}:{{{''.join(sorted(r))}}}" for p, r in anchors.items()
        )
        print(f"{target}: anchors {anchor_desc}")
        print(
            f"    pool {len(pool)} other-allele peptides; {n_anchor_ok} match both anchors; "
            f"{len(survivors)} of those also score <= {cutoff:+.2f} "
            f"({args.max_pctile:.0f}th pctile)"
        )
        if len(chosen) < args.k_decoys:
            print(f"    WARNING: only {len(chosen)} hard decoys available")
        for p in chosen:
            print(
                f"    hard decoy {p}  P2={p[1]} P{args.peptide_length}={p[-1]}  "
                f"target-motif score {score_peptide(p, pwm):+.2f}"
            )
            rows.append({
                "pdb_code": "hard",
                "locus": locus_of(target),
                "allele_slug": allele_to_slug(target),
                "peptide_sequence": p,
                "resolution": "NA",
            })

    out = pd.DataFrame(
        rows,
        columns=["pdb_code", "locus", "allele_slug", "peptide_sequence", "resolution"],
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False, header=False)
    print(f"\nWrote {len(out)} hard decoy folds -> {args.out}")


if __name__ == "__main__":
    main()
