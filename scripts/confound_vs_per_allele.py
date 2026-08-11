"""Does the negative-sampling confound reach the per-allele analysis?

Context
-------
`negatives_peptide_pool` builds its sampling pool as a multiset: a peptide observed
for twelve alleles is appended twelve times, so promiscuous peptides are drawn as
negatives far more often than motif-restricted ones. The consequence, measured by
`crossover_label_balance.py`, is that peptide identity alone scores AUROC 0.248 on
the validation set -- 0.25 from chance, inverted.

That inflates the pooled validation figure. The question this script answers is
narrower and more important: does it also distort the *per-allele* AUROC
distribution, on which the HLA-C motif-isolation finding rests?

The worry is a specific coupling. The confound's strength varies by allele, and it
should be strongest for alleles whose ligands are promiscuous -- which is to say
alleles with broad, permissive motifs. Motif breadth is precisely the variable the
HLA-C argument turns on. If per-allele confound strength correlates with per-allele
model AUROC, the two cannot be disentangled without regenerating the negatives.

Reads the per-allele prior AUROCs written by `crossover_label_balance.py --out` and
correlates them against per-allele model AUROC.

Interpretation
--------------
  |rho| < 0.2, p > 0.05   Per-allele analysis is safe. Report this as the evidence
                          that the pooled confound does not propagate.
  |rho| 0.2-0.4           Partial. Report the correlation and caveat the HLA-C
                          mechanism claim in Chapter 5.
  |rho| > 0.4             Material. The per-allele distribution is confounded with
                          negative sampling; the motif-isolation claim needs either
                          regenerated negatives or restatement as provisional.

Usage
-----
    python scripts/confound_vs_per_allele.py \
        --prior  results/crossover_prior_per_allele.csv \
        --auroc  results/per_allele_auroc.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr


def load_auroc(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "allele" not in df.columns:
        raise SystemExit(f"no 'allele' column in {path}; found {sorted(df.columns)}")
    cand = [c for c in df.columns if "auroc" in c or c in ("auc", "score")]
    if not cand:
        raise SystemExit(f"no AUROC-like column in {path}; found {sorted(df.columns)}")
    return df.rename(columns={cand[0]: "model_auroc"})[["allele", "model_auroc"]]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prior", required=True, type=Path,
                    help="crossover_prior_per_allele.csv")
    ap.add_argument("--auroc", required=True, type=Path,
                    help="per-allele model AUROC CSV")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    prior = pd.read_csv(args.prior)
    prior.columns = [c.strip().lower() for c in prior.columns]
    # Distance from chance is the quantity of interest: 0.25 and 0.75 are equally
    # confounded, in opposite directions.
    prior["confound_strength"] = (prior["prior_auroc"] - 0.5).abs()

    merged = load_auroc(args.auroc).merge(prior, on="allele", how="inner")
    print(f"matched {len(merged)} alleles\n")
    if len(merged) < 10:
        print("too few alleles matched to correlate; check allele-string formats")
        return

    print("=" * 70)
    print("CONFOUND STRENGTH vs MODEL PERFORMANCE")
    print("=" * 70)
    for col, desc in (
        ("confound_strength", "|prior AUROC - 0.5|  (how confounded the allele is)"),
        ("prior_auroc", "raw prior AUROC       (signed, for reference)"),
    ):
        rho, p = spearmanr(merged[col], merged["model_auroc"])
        print(f"  {desc}\n      rho {rho:+.3f}   p {p:.4f}")

    rho, p = spearmanr(merged["confound_strength"], merged["model_auroc"])
    print()
    if abs(rho) < 0.2 and p > 0.05:
        verdict = ("SAFE. Per-allele model AUROC is not associated with per-allele "
                   "confound strength, so the negative-sampling artefact inflates the "
                   "pooled figure without distorting the per-allele distribution. "
                   "Report this correlation in Methods 3.1.3 as the evidence that the "
                   "HLA-C motif-isolation finding is unaffected.")
    elif abs(rho) < 0.4:
        verdict = ("PARTIAL. Some association. Report the correlation and add an "
                   "explicit caveat to the HLA-C mechanism claim in Chapter 5.")
    else:
        verdict = ("MATERIAL. Per-allele performance tracks confound strength. The "
                   "motif-isolation claim cannot be separated from negative sampling "
                   "without regenerated negatives; restate it as provisional.")
    print(f"  VERDICT: {verdict}")

    print("\n  By locus (mean confound strength):")
    merged["locus"] = merged.allele.str[4]
    print(merged.groupby("locus")[["confound_strength", "model_auroc"]]
                .agg(["mean", "count"]).round(3).to_string())

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(args.out, index=False)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()