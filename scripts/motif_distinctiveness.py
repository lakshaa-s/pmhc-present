"""Does an allele's motif similarity to OTHER alleles predict its performance?

Anchor information content predicts per-allele AUROC across loci (rho 0.66) but not
within HLA-C (rho 0.07), even though HLA-C has comparable IC spread and the largest
AUROC variance of the three loci. So something else orders the HLA-C alleles.

The candidate: motif *distinctiveness*. HLA-C*16:02 (P2 = A75/S12/T6) and HLA-C*16:01
(P2 = A50/S18/T7) have near-identical preferences. A model has no basis to separate
them however sharply each is individually constrained -- so an allele can carry high
IC and still be unlearnable if a neighbour shares its motif. IC measures how
constrained an allele is; it says nothing about whether that constraint is unique.

Distance is Jensen-Shannon divergence between position weight matrices, averaged over
positions (bounded [0, 1]). Computed over the full 9mer profile and over anchor
positions only, since anchors carry most of the binding signal.

Reports, per allele: distance to nearest neighbour, mean distance to the 3 nearest,
and which allele is nearest. Then tests whether those predict AUROC, overall and
within locus, controlling for anchor IC and sample size.

Usage:
    python scripts/motif_distinctiveness.py \
        --atlas data/processed/atlas_labelled.csv \
        --ic-perf data/processed/ic_vs_performance.csv \
        --out data/processed/motif_distinctiveness.csv
"""

from __future__ import annotations

import argparse
import collections

import numpy as np
import pandas as pd
from scipy import stats

AA = "ACDEFGHIKLMNPQRSTVWY"
LENGTH = 9
PSEUDOCOUNT = 0.5  # keeps JSD stable for the ~200-peptide alleles
ANCHORS = (1, 8)  # P2 and P-omega, 0-based


def pwm(peptides: list[str]) -> np.ndarray:
    """(LENGTH, 20) position weight matrix, pseudocounted and row-normalised."""
    m = np.full((LENGTH, 20), PSEUDOCOUNT)
    idx = {a: i for i, a in enumerate(AA)}
    for p in peptides:
        for i, ch in enumerate(p):
            j = idx.get(ch)
            if j is not None:
                m[i, j] += 1
    return m / m.sum(axis=1, keepdims=True)


def jsd(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence, base 2, in [0, 1]."""
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def profile_distance(a: np.ndarray, b: np.ndarray, positions=None) -> float:
    pos = range(LENGTH) if positions is None else positions
    return float(np.mean([jsd(a[i], b[i]) for i in pos]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlas", default="data/processed/atlas_labelled.csv")
    ap.add_argument("--ic-perf", default="data/processed/ic_vs_performance.csv",
                    help="output of scripts/ic_vs_performance.py (allele, auroc, ic, n)")
    ap.add_argument("--out", default="data/processed/motif_distinctiveness.csv")
    ap.add_argument("--min-peptides", type=int, default=50)
    args = ap.parse_args()

    df = pd.read_csv(args.atlas)
    df = df[(df.label == 1) & (df.length == LENGTH)]

    mats, counts = {}, {}
    for allele, grp in df.groupby("allele"):
        peps = [p for p in grp.peptide if len(p) == LENGTH]
        if len(peps) >= args.min_peptides:
            mats[allele] = pwm(peps)
            counts[allele] = len(peps)

    alleles = sorted(mats)
    n = len(alleles)
    full = np.zeros((n, n))
    anch = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            a, b = mats[alleles[i]], mats[alleles[j]]
            full[i, j] = full[j, i] = profile_distance(a, b)
            anch[i, j] = anch[j, i] = profile_distance(a, b, ANCHORS)

    rows = []
    for i, allele in enumerate(alleles):
        others = [j for j in range(n) if j != i]
        fd = full[i, others]
        ad = anch[i, others]
        order = np.argsort(fd)
        rows.append({
            "allele": allele,
            "locus": allele.split("*")[0],
            "n_peptides": counts[allele],
            "nn_dist": float(fd[order[0]]),
            "nn_allele": alleles[others[order[0]]],
            "mean_dist_3": float(np.mean(fd[order[:3]])),
            "mean_dist_all": float(np.mean(fd)),
            "nn_dist_anchor": float(np.min(ad)),
            "nn_allele_anchor": alleles[others[int(np.argmin(ad))]],
        })
    dist = pd.DataFrame(rows)

    perf = pd.read_csv(args.ic_perf)
    keep = [c for c in ["allele", "auroc", "ic_p2_pomega", "log_n"] if c in perf.columns]
    d = dist.merge(perf[keep], on="allele", how="inner").dropna(subset=["auroc"])
    if "log_n" not in d.columns:
        d["log_n"] = np.log10(d.n_peptides)
    d.to_csv(args.out, index=False)

    print(f"{n} alleles with >= {args.min_peptides} {LENGTH}mers; "
          f"{len(d)} matched to AUROC\n")

    print("=== nearest-neighbour motif distance by locus ===")
    print(f"{'locus':<8} {'n':>4} {'median nn':>10} {'median nn(anchor)':>18} "
          f"{'median AUROC':>13}")
    for locus, g in d.groupby("locus"):
        print(f"{locus:<8} {len(g):>4} {g.nn_dist.median():>10.4f} "
              f"{g.nn_dist_anchor.median():>18.4f} {g.auroc.median():>13.3f}")

    print("\n=== does distinctiveness predict AUROC? (Spearman) ===")
    print(f"{'predictor':<18} {'rho':>7} {'p':>10}")
    for c in ["nn_dist", "nn_dist_anchor", "mean_dist_3", "mean_dist_all"]:
        rho, pv = stats.spearmanr(d[c], d.auroc)
        print(f"{c:<18} {rho:>7.3f} {pv:>10.2e}")

    if "ic_p2_pomega" in d.columns:
        print("\n=== partial correlations (competing explanations) ===")

        def partial(x, y, z):
            rx, ry, rz = (stats.rankdata(v) for v in (x, y, z))
            ex = rx - np.polyval(np.polyfit(rz, rx, 1), rz)
            ey = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
            return stats.pearsonr(ex, ey)

        r, p = partial(d.nn_dist.values, d.auroc.values, d.ic_p2_pomega.values)
        print(f"nn_dist vs AUROC, controlling for anchor IC : rho={r:>6.3f}  p={p:.2e}")
        r, p = partial(d.ic_p2_pomega.values, d.auroc.values, d.nn_dist.values)
        print(f"anchor IC vs AUROC, controlling for nn_dist : rho={r:>6.3f}  p={p:.2e}")
        r, p = partial(d.nn_dist.values, d.auroc.values, d.log_n.values)
        print(f"nn_dist vs AUROC, controlling for sample size: rho={r:>6.3f}  p={p:.2e}")

    print("\n=== within locus ===")
    print(f"{'locus':<8} {'n':>4} {'rho(nn_dist, AUROC)':>21} {'rho(IC, AUROC)':>16}")
    for locus, g in d.groupby("locus"):
        if len(g) >= 5:
            r1, _ = stats.spearmanr(g.nn_dist, g.auroc)
            r2 = (stats.spearmanr(g.ic_p2_pomega, g.auroc)[0]
                  if "ic_p2_pomega" in g.columns else np.nan)
            print(f"{locus:<8} {len(g):>4} {r1:>21.3f} {r2:>16.3f}")

    print("\n=== least distinctive alleles (closest motif neighbours) ===")
    cols = ["allele", "nn_allele", "nn_dist", "auroc", "n_peptides"]
    print(d.nsmallest(12, "nn_dist")[cols].to_string(index=False))

    print("\n=== lowest-AUROC alleles, with their nearest neighbour ===")
    print(d.nsmallest(12, "auroc")[cols].to_string(index=False))

    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()