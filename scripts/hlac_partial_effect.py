"""Does the HLA-C deficit survive controlling for negative-sampling confound?

Context
-------
`confound_vs_per_allele.py` found rho +0.302 (p 0.0007) between per-allele confound
strength and per-allele model AUROC, and a locus breakdown showing HLA-C is the
*least* confounded locus (0.202) while also the lowest performing (0.940). Because
the correlation is positive, the confound inflates HLA-A and HLA-B more than HLA-C,
so some of the apparent HLA-C deficit is differential inflation elsewhere rather
than depression of HLA-C itself.

The question is whether any HLA-C effect remains once that is accounted for. Three
tests, increasing in strength:

  1. Partial Spearman correlation of locus-C indicator with AUROC, controlling for
     confound strength.
  2. OLS of AUROC on locus dummies plus confound strength -- reports whether the
     HLA-C coefficient survives with the confound in the model.
  3. Matched comparison: HLA-C alleles against A/B alleles of comparable confound
     strength, which makes no functional-form assumption.

Usage
-----
    python scripts/hlac_partial_effect.py \
        --merged results/confound_vs_per_allele.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def partial_spearman(x, y, z):
    """Spearman correlation of x and y with z partialled out, via rank residuals."""
    rx, ry, rz = (stats.rankdata(v) for v in (x, y, z))
    def resid(a):
        slope, intercept, *_ = stats.linregress(rz, a)
        return a - (slope * rz + intercept)
    return stats.pearsonr(resid(rx), resid(ry))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--merged", required=True, type=Path,
                    help="confound_vs_per_allele.csv")
    ap.add_argument("--tolerance", type=float, default=0.03,
                    help="confound-strength window for the matched comparison")
    args = ap.parse_args()

    df = pd.read_csv(args.merged)
    df["locus"] = df.allele.str[4]
    df["is_C"] = (df.locus == "C").astype(int)
    print(f"{len(df)} alleles: " +
          ", ".join(f"{k} {v}" for k, v in df.locus.value_counts().items()) + "\n")

    print("=" * 70)
    print("1. PARTIAL CORRELATION (locus-C vs AUROC, confound partialled out)")
    print("=" * 70)
    r_raw, p_raw = stats.spearmanr(df.is_C, df.model_auroc)
    r_par, p_par = partial_spearman(df.is_C, df.model_auroc, df.confound_strength)
    print(f"  raw:      rho {r_raw:+.3f}   p {p_raw:.4f}")
    print(f"  partial:  rho {r_par:+.3f}   p {p_par:.4f}")
    print(f"  attenuation: {100 * (1 - abs(r_par) / abs(r_raw)):.1f}% of the raw "
          f"association is attributable to confound strength")

    print()
    print("=" * 70)
    print("2. OLS: AUROC ~ locus + confound_strength")
    print("=" * 70)
    X = pd.get_dummies(df.locus, prefix="locus", drop_first=True).astype(float)
    X["confound"] = df.confound_strength.to_numpy()
    X.insert(0, "const", 1.0)
    y = df.model_auroc.to_numpy()
    Xa = X.to_numpy()
    beta, *_ = np.linalg.lstsq(Xa, y, rcond=None)
    resid = y - Xa @ beta
    dof = len(y) - Xa.shape[1]
    mse = resid @ resid / dof
    se = np.sqrt(np.diag(mse * np.linalg.pinv(Xa.T @ Xa)))
    print(f"  {'term':<16}{'coef':>10}{'se':>10}{'t':>8}{'p':>10}")
    for name, b, s in zip(X.columns, beta, se):
        t = b / s if s else np.nan
        p = 2 * (1 - stats.t.cdf(abs(t), dof))
        print(f"  {name:<16}{b:>10.4f}{s:>10.4f}{t:>8.2f}{p:>10.4f}")
    print("\n  Read the locus_C row (or the omitted-category comparison): if its")
    print("  coefficient stays negative and significant with confound in the model,")
    print("  the HLA-C effect is not explained by negative sampling.")

    print()
    print("=" * 70)
    print(f"3. MATCHED COMPARISON (confound strength within +/-{args.tolerance})")
    print("=" * 70)
    C = df[df.locus == "C"]
    AB = df[df.locus != "C"]
    diffs = []
    for _, row in C.iterrows():
        m = AB[(AB.confound_strength - row.confound_strength).abs() <= args.tolerance]
        if len(m):
            diffs.append(row.model_auroc - m.model_auroc.mean())
    if diffs:
        d = np.array(diffs)
        t, p = stats.ttest_1samp(d, 0.0)
        print(f"  {len(d)} HLA-C alleles matched to A/B controls")
        print(f"  mean AUROC difference (C minus matched A/B): {d.mean():+.4f}")
        print(f"  95% CI: [{d.mean() - 1.96 * d.std(ddof=1) / np.sqrt(len(d)):+.4f}, "
              f"{d.mean() + 1.96 * d.std(ddof=1) / np.sqrt(len(d)):+.4f}]")
        print(f"  t {t:+.2f}   p {p:.4f}")
        print("\n  This makes no functional-form assumption and is the figure to")
        print("  report in Chapter 5 alongside the fold-set v2 HLA-C result, which")
        print("  is independent of peptide-pool negatives entirely.")
    else:
        print("  no matches within tolerance; widen --tolerance")


if __name__ == "__main__":
    main()