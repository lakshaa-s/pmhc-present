"""Does motif information content predict per-allele model performance?

The locus survey falsified the "HLA-C is data-poor" explanation -- HLA-C alleles
carry a slightly *higher* median 9mer count than A or B. But their anchors are
lower-contrast: HLA-B*27:05 holds 4.04 bits at P2, while HLA-C*03:04 holds 1.75
despite having more peptides.

This tests the resulting hypothesis: performance tracks how much information the
motif actually carries, not how many examples were available. Sample size is
included as a competing predictor so the two can be separated rather than assumed.

Reports Spearman correlations, partial correlations (each predictor with the other
residualised out), and a standardised linear model, overall and per locus.

Usage:
    python scripts/ic_vs_performance.py \
        --anchors data/processed/anchors.json \
        --auroc <per-allele auroc csv> \
        --out data/processed/ic_vs_performance.csv
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from scipy import stats

ALLELE_COLS = ["allele", "Allele", "hla", "HLA"]
AUROC_COLS = ["auroc", "AUROC", "auc", "AUC", "roc_auc", "auroc_mean"]


def pick(df: pd.DataFrame, candidates: list[str], what: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise SystemExit(
        f"Could not find a {what} column. Looked for {candidates}.\n"
        f"Columns present: {list(df.columns)}\n"
        f"Re-run with --allele-col / --auroc-col to name it explicitly."
    )


def partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[float, float]:
    """Spearman correlation of x and y with z residualised out of both."""
    rx, ry, rz = (stats.rankdata(v) for v in (x, y, z))
    ex = rx - np.polyval(np.polyfit(rz, rx, 1), rz)
    ey = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
    return stats.pearsonr(ex, ey)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchors", default="data/processed/anchors.json")
    ap.add_argument("--auroc", required=True)
    ap.add_argument("--out", default="data/processed/ic_vs_performance.csv")
    ap.add_argument("--allele-col")
    ap.add_argument("--auroc-col")
    args = ap.parse_args()

    with open(args.anchors) as fh:
        meta = json.load(fh)["alleles"]
    rows = []
    for allele, rec in meta.items():
        ic = rec["ic"]
        anchors = [i if i >= 0 else len(ic) - 1 for i in rec["anchors"]]
        rows.append({
            "allele": allele,
            "locus": allele.split("*")[0],
            "n_peptides": rec["n_peptides"],
            "ic_p2": ic[1],
            "ic_pomega": ic[-1],
            "ic_anchor_mean": float(np.mean([ic[i] for i in anchors])) if anchors else np.nan,
            "ic_p2_pomega": (ic[1] + ic[-1]) / 2,
            "ic_total": float(sum(ic)),
        })
    motif = pd.DataFrame(rows)

    perf = pd.read_csv(args.auroc)
    acol = args.allele_col or pick(perf, ALLELE_COLS, "allele")
    ycol = args.auroc_col or pick(perf, AUROC_COLS, "AUROC")
    perf = perf[[acol, ycol]].rename(columns={acol: "allele", ycol: "auroc"})
    perf = perf.groupby("allele", as_index=False).auroc.mean()

    df = motif.merge(perf, on="allele", how="inner").dropna(subset=["auroc"])
    df["log_n"] = np.log10(df.n_peptides)
    df.to_csv(args.out, index=False)

    print(f"{len(df)} alleles matched between motif table and AUROC file")
    if len(df) < 20:
        print("  (too few to interpret -- check the allele naming matches between files)")
        return
    print(f"AUROC: median {df.auroc.median():.3f}, range {df.auroc.min():.3f}-{df.auroc.max():.3f}\n")

    preds = ["ic_p2", "ic_pomega", "ic_p2_pomega", "ic_anchor_mean", "ic_total", "log_n"]
    print("=== Spearman correlation with AUROC ===")
    print(f"{'predictor':<18} {'rho':>7} {'p':>10}")
    for p in preds:
        rho, pv = stats.spearmanr(df[p], df.auroc)
        print(f"{p:<18} {rho:>7.3f} {pv:>10.2e}")

    print("\n=== Partial correlations (the competing explanations) ===")
    rho, pv = partial_spearman(df.ic_p2_pomega.values, df.auroc.values, df.log_n.values)
    print(f"anchor IC vs AUROC, controlling for sample size : rho={rho:>6.3f}  p={pv:.2e}")
    rho, pv = partial_spearman(df.log_n.values, df.auroc.values, df.ic_p2_pomega.values)
    print(f"sample size vs AUROC, controlling for anchor IC : rho={rho:>6.3f}  p={pv:.2e}")

    # standardised coefficients: comparable effect sizes on the same scale
    X = np.column_stack([
        stats.zscore(df.ic_p2_pomega), stats.zscore(df.log_n), np.ones(len(df))
    ])
    beta, *_ = np.linalg.lstsq(X, stats.zscore(df.auroc), rcond=None)
    pred = X @ beta
    r2 = 1 - ((stats.zscore(df.auroc) - pred) ** 2).sum() / (stats.zscore(df.auroc) ** 2).sum()
    print("\nStandardised model: AUROC ~ anchor_IC + log10(n)")
    print(f"  beta(anchor IC) = {beta[0]:+.3f}")
    print(f"  beta(log n)     = {beta[1]:+.3f}")
    print(f"  R^2             = {r2:.3f}")

    print("\n=== Per locus ===")
    print(f"{'locus':<8} {'n':>4} {'med AUROC':>10} {'med anchor IC':>14} {'med n_pep':>10} "
          f"{'rho(IC,AUROC)':>14}")
    for locus, g in df.groupby("locus"):
        if len(g) >= 5:
            rho, _ = stats.spearmanr(g.ic_p2_pomega, g.auroc)
        else:
            rho = np.nan
        print(f"{locus:<8} {len(g):>4} {g.auroc.median():>10.3f} "
              f"{g.ic_p2_pomega.median():>14.2f} {int(g.n_peptides.median()):>10} {rho:>14.3f}")

    print("\n=== Lowest-AUROC alleles ===")
    cols = ["allele", "auroc", "ic_p2", "ic_pomega", "n_peptides"]
    print(df.nsmallest(12, "auroc")[cols].to_string(index=False))

    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()