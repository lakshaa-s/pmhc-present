"""Motif distance vs pseudosequence distance: which predicts per-allele performance?

The pan-allele literature (Hoof 2009; Karosiene 2012; NetMHCIIpan-3.0) reports that
predictive performance falls with distance to the nearest neighbour in the training
set, where distance is measured between MHC *pseudosequences* -- the pocket residues
of the protein, following Nielsen et al. 2008.

scripts/motif_distinctiveness.py found the same relationship using distance between
*motifs* (PWMs over eluted ligands). Those are different objects: a pseudosequence is
a property of the molecule, a motif is a property of its observed binding behaviour.

This script computes both and asks whether motif distance carries information that
pseudosequence distance does not:

  - if the two are near-identical and neither survives controlling for the other,
    the motif result is a re-derivation of the published one;
  - if motif distance predicts AUROC after controlling for pseudosequence distance,
    it adds something -- observed specificity diverging from pocket chemistry.

Pseudosequence distance follows the Nielsen convention: BLOSUM62 similarity, summed
over positions, normalised by the self-similarities, so distance = 1 - s_ab/sqrt(s_aa*s_bb).

Usage:
    python scripts/pseudoseq_vs_motif_distance.py \
        --pseudoseq data/pseudoseq/hla_a.json data/pseudoseq/hla_b.json data/pseudoseq/hla_c.json \
        --distinctiveness data/processed/motif_distinctiveness.csv \
        --out data/processed/pseudoseq_vs_motif.csv
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from scipy import stats

# BLOSUM62, standard 20 amino acids in this order
AA = "ARNDCQEGHILKMFPSTWYV"
_B62 = """
 4 -1 -2 -2  0 -1 -1  0 -2 -1 -1 -1 -1 -2 -1  1  0 -3 -2  0
-1  5  0 -2 -3  1  0 -2  0 -3 -2  2 -1 -3 -2 -1 -1 -3 -2 -3
-2  0  6  1 -3  0  0  0  1 -3 -3  0 -2 -3 -2  1  0 -4 -2 -3
-2 -2  1  6 -3  0  2 -1 -1 -3 -4 -1 -3 -3 -1  0 -1 -4 -3 -3
 0 -3 -3 -3  9 -3 -4 -3 -3 -1 -1 -3 -1 -2 -3 -1 -1 -2 -2 -1
-1  1  0  0 -3  5  2 -2  0 -3 -2  1  0 -3 -1  0 -1 -2 -1 -2
-1  0  0  2 -4  2  5 -2  0 -3 -3  1 -2 -3 -1  0 -1 -3 -2 -2
 0 -2  0 -1 -3 -2 -2  6 -2 -4 -4 -2 -3 -3 -2  0 -2 -2 -3 -3
-2  0  1 -1 -3  0  0 -2  8 -3 -3 -1 -2 -1 -2 -1 -2 -2  2 -3
-1 -3 -3 -3 -1 -3 -3 -4 -3  4  2 -3  1  0 -3 -2 -1 -3 -1  3
-1 -2 -3 -4 -1 -2 -3 -4 -3  2  4 -2  2  0 -3 -2 -1 -2 -1  1
-1  2  0 -1 -3  1  1 -2 -1 -3 -2  5 -1 -3 -1  0 -1 -3 -2 -2
-1 -1 -2 -3 -1  0 -2 -3 -2  1  2 -1  5  0 -2 -1 -1 -1 -1  1
-2 -3 -3 -3 -2 -3 -3 -3 -1  0  0 -3  0  6 -4 -2 -2  1  3 -1
-1 -2 -2 -1 -3 -1 -1 -2 -2 -3 -3 -1 -2 -4  7 -1 -1 -4 -3 -2
 1 -1  1  0 -1  0  0  0 -1 -2 -2  0 -1 -2 -1  4  1 -3 -2 -2
 0 -1  0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1  1  5 -2 -2  0
-3 -3 -4 -4 -2 -2 -3 -2 -2 -3 -2 -3 -1  1 -4 -3 -2 11  2 -3
-2 -2 -2 -3 -2 -1 -2 -3  2 -1 -1 -2 -1  3 -3 -2 -2  2  7 -1
 0 -3 -3 -3 -1 -2 -2 -3 -3  3  1 -2  1 -1 -2 -2  0 -3 -1  4
"""
BL = {(a, b): int(v)
      for a, row in zip(AA, _B62.strip().split("\n"))
      for b, v in zip(AA, row.split())}


def pseudo_distance(a: str, b: str) -> float:
    """Nielsen-convention distance between two pseudosequences, in [0, ~1]."""
    if len(a) != len(b):
        return float("nan")
    sab = sum(BL.get((x, y), 0) for x, y in zip(a, b))
    saa = sum(BL.get((x, x), 0) for x in a)
    sbb = sum(BL.get((y, y), 0) for y in b)
    if saa <= 0 or sbb <= 0:
        return float("nan")
    return 1.0 - sab / np.sqrt(saa * sbb)


def partial(x, y, z):
    rx, ry, rz = (stats.rankdata(v) for v in (x, y, z))
    ex = rx - np.polyval(np.polyfit(rz, rx, 1), rz)
    ey = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
    return stats.pearsonr(ex, ey)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pseudoseq", nargs="+", required=True)
    ap.add_argument("--distinctiveness",
                    default="data/processed/motif_distinctiveness.csv")
    ap.add_argument("--out", default="data/processed/pseudoseq_vs_motif.csv")
    args = ap.parse_args()

    pseudo: dict[str, str] = {}
    for path in args.pseudoseq:
        with open(path) as fh:
            pseudo.update(json.load(fh))

    d = pd.read_csv(args.distinctiveness)
    have = [a for a in d.allele if a in pseudo]
    missing = len(d) - len(have)
    d = d[d.allele.isin(have)].reset_index(drop=True)
    print(f"{len(d)} alleles with both a pseudosequence and a motif "
          f"({missing} dropped for missing pseudosequence)")
    if len(d) < 20:
        print("  too few to interpret -- check allele naming matches between files")
        return

    alleles = d.allele.tolist()
    n = len(alleles)
    P = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            P[i, j] = P[j, i] = pseudo_distance(pseudo[alleles[i]], pseudo[alleles[j]])

    d["pseudo_nn_dist"] = [float(np.min(np.delete(P[i], i))) for i in range(n)]
    d["pseudo_nn_allele"] = [alleles[int(np.argmin(np.where(
        np.arange(n) == i, np.inf, P[i])))] for i in range(n)]
    d["pseudo_mean_dist"] = [float(np.mean(np.delete(P[i], i))) for i in range(n)]
    d.to_csv(args.out, index=False)

    print("\n=== how related are the two distance measures? ===")
    rho, pv = stats.spearmanr(d.nn_dist, d.pseudo_nn_dist)
    print(f"motif nn_dist vs pseudoseq nn_dist:  rho={rho:.3f}  p={pv:.2e}")
    agree = (d.nn_allele == d.pseudo_nn_allele).sum()
    print(f"same nearest neighbour by both:      {agree}/{n} ({agree/n:.0%})")

    print("\n=== each on its own (Spearman vs AUROC) ===")
    for c in ["nn_dist", "pseudo_nn_dist", "mean_dist_all", "pseudo_mean_dist"]:
        rho, pv = stats.spearmanr(d[c], d.auroc)
        print(f"  {c:<18} rho={rho:>7.3f}  p={pv:.2e}")

    print("\n=== does either survive the other? ===")
    r, p = partial(d.nn_dist.values, d.auroc.values, d.pseudo_nn_dist.values)
    print(f"  motif dist vs AUROC, controlling for pseudoseq dist: rho={r:>6.3f}  p={p:.2e}")
    r, p = partial(d.pseudo_nn_dist.values, d.auroc.values, d.nn_dist.values)
    print(f"  pseudoseq dist vs AUROC, controlling for motif dist: rho={r:>6.3f}  p={p:.2e}")

    if "ic_p2_pomega" in d.columns:
        r, p = partial(d.pseudo_nn_dist.values, d.auroc.values, d.ic_p2_pomega.values)
        print(f"  pseudoseq dist vs AUROC, controlling for anchor IC : rho={r:>6.3f}  p={p:.2e}")

    print("\n=== within locus ===")
    print(f"{'locus':<8} {'n':>4} {'rho(motif)':>11} {'rho(pseudo)':>12} {'NN agree':>9}")
    for locus, g in d.groupby("locus"):
        if len(g) >= 5:
            r1, _ = stats.spearmanr(g.nn_dist, g.auroc)
            r2, _ = stats.spearmanr(g.pseudo_nn_dist, g.auroc)
            ag = (g.nn_allele == g.pseudo_nn_allele).mean()
            print(f"{locus:<8} {len(g):>4} {r1:>11.3f} {r2:>12.3f} {ag:>9.0%}")

    print("\n=== alleles where motif and pseudosequence most disagree ===")
    d["rank_gap"] = (stats.rankdata(d.nn_dist) - stats.rankdata(d.pseudo_nn_dist))
    cols = ["allele", "nn_allele", "nn_dist", "pseudo_nn_allele",
            "pseudo_nn_dist", "auroc"]
    print("  motif-isolated but pseudosequence-typical:")
    print(d.nlargest(6, "rank_gap")[cols].to_string(index=False))
    print("\n  pseudosequence-isolated but motif-typical:")
    print(d.nsmallest(6, "rank_gap")[cols].to_string(index=False))

    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
