"""Does allele-specific expression predict per-allele performance?

The question
------------
Three per-allele predictors are established in this work: anchor information content,
motif nearest-neighbour distance, and — from the AFND analysis — global population
frequency, together with repertoire overlap. A fourth candidate is surface
expression. HLA-C is expressed roughly ten-fold lower than HLA-A and HLA-B, and the
per-locus means (0.968 / 0.976 / 0.941 for A / B / C) are consistent with expression
mattering. But that is three points, and locus membership is confounded with almost
everything else distinguishing these genes.

D'Antonio et al. (2019, eLife 8:e48476) quantified allele-specific expression at
eight-digit resolution in 419 individuals, finding more than four-fold differences
between the least- and most-expressed alleles of HLA-A and HLA-C. Collapsed to
four-digit, 92 of this project's 123 alleles are covered — enough for an allele-level
test rather than a locus-level observation.

Design
------
Expression differs systematically between loci, so a raw correlation across all
alleles would recover the locus effect rather than an allele-level one. Expression is
therefore **standardised within gene** before correlating, which asks the sharper
question: among HLA-B alleles, do the higher-expressed ones score better?

Eight-digit types are collapsed to four by averaging, weighted by the number of
individuals carrying each type, since a type seen in two people gives a far less
reliable estimate than one seen in two hundred. Alleles whose four-digit estimate
rests on fewer than `--min-carriers` individuals in total are dropped rather than
included at face value.

The locus-level comparison is reported alongside, because it is the observation the
allele-level test is meant to improve on and the two should be read together.

Four limitations, all of which weaken a positive finding
---------------------------------------------------------
Expression is measured in induced pluripotent stem cells. The authors note iPSCs have
a distinct regulatory landscape from somatic tissues, and immunopeptidomics is
performed on cell lines and tissues, not iPSCs. This is a proxy for the relevant
quantity, not the quantity.

The cohort is approximately 80% of European descent (HipSci's 146 donors entirely so;
iPSCORE's 273 comprising 190 European against 30 Asian, 18 Hispanic, 7 African
American, 5 Indian and 3 Middle Eastern). The motif-isolated and less-studied alleles
central to this project's equity argument are therefore those estimated from fewest
individuals — the same bias that limited the AFND analysis.

Collapsing eight-digit to four-digit averages over exactly the distinctions the
source finding rests on.

And expression here is steady-state transcript abundance, not surface protein
density, which is what would actually constrain peptide loading.

Usage:
    python scripts/expression_vs_performance.py \
        --supp data/raw/dantonio_supp5.xlsx \
        --auroc results/per_allele_auroc_v3.csv \
        --motif data/processed/motif_distinctiveness.csv \
        --afnd results/afnd_frequency_per_allele.csv \
        --out results/expression_vs_performance.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def partial_spearman(x, y, z):
    xr, yr, zr = (stats.rankdata(v) for v in (x, y, z))
    Z = np.column_stack([np.ones_like(zr), zr])
    rx = xr - Z @ np.linalg.lstsq(Z, xr, rcond=None)[0]
    ry = yr - Z @ np.linalg.lstsq(Z, yr, rcond=None)[0]
    r, p = stats.pearsonr(rx, ry)
    return float(r), float(p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--supp", required=True, help="D'Antonio supplementary file 5")
    ap.add_argument("--auroc", required=True)
    ap.add_argument("--motif", default="data/processed/motif_distinctiveness.csv")
    ap.add_argument("--afnd", default="results/afnd_frequency_per_allele.csv")
    ap.add_argument("--min-carriers", type=int, default=4,
                    help="drop four-digit alleles estimated from fewer carriers")
    ap.add_argument("--out", default="results/expression_vs_performance.csv")
    args = ap.parse_args()

    e = pd.read_excel(args.supp)
    e.columns = [c.strip() for c in e.columns]
    ncol = next(c for c in e.columns if c.lower().startswith("n allele"))
    e = e.rename(columns={"HLA type": "type", "Mean expression": "expr",
                          ncol: "n_carriers"})
    e = e[e.Gene.isin(["HLA-A", "HLA-B", "HLA-C"])].copy()
    e["allele"] = "HLA-" + e.type.astype(str).str.extract(
        r"^([ABC]\*\d+:\d+)")[0]
    e = e.dropna(subset=["allele", "expr", "n_carriers"])
    print(f"{len(e)} class I eight-digit types, "
          f"{e.allele.nunique()} distinct four-digit")

    # carrier-weighted mean across the eight-digit types of each four-digit allele
    e["w"] = e.expr * e.n_carriers
    g = e.groupby(["allele", "Gene"]).agg(
        wsum=("w", "sum"), carriers=("n_carriers", "sum"),
        n_types=("type", "nunique")).reset_index()
    g["expr"] = g.wsum / g.carriers
    before = len(g)
    g = g[g.carriers >= args.min_carriers]
    print(f"{before - len(g)} alleles dropped for <{args.min_carriers} carriers; "
          f"{len(g)} remain")

    a = pd.read_csv(args.auroc)
    col = next(c for c in ("auroc", "val_auroc", "model_auroc") if c in a.columns)
    d = a[["allele", col]].rename(columns={col: "auroc"}).merge(g, on="allele")
    print(f"\n{len(d)} of {len(a)} alleles matched to expression data")
    if len(d) < 25:
        raise SystemExit("too few matched alleles for an allele-level test")

    # expression differs between loci, so standardise within gene: the question is
    # whether higher-expressed alleles OF THE SAME LOCUS score better
    d["expr_z"] = d.groupby("Gene").expr.transform(
        lambda x: (x - x.mean()) / x.std() if len(x) > 2 and x.std() > 0 else np.nan)
    d = d.dropna(subset=["expr_z"])

    print("\n=== the locus-level observation, for comparison ===")
    loc = d.groupby("Gene").agg(n=("allele", "size"),
                                mean_expr=("expr", "mean"),
                                mean_auroc=("auroc", "mean")).round(3)
    print(loc.to_string())
    print("  (three points; locus membership is confounded with motif breadth,")
    print("   allele counts and assay history, so this cannot carry weight alone)")

    print("\n=== the allele-level test ===")
    r_raw = stats.spearmanr(d.expr, d.auroc)
    r_z = stats.spearmanr(d.expr_z, d.auroc)
    print(f"  raw expression vs AUROC              rho {r_raw[0]:+.3f}  "
          f"p {r_raw[1]:.4f}   (recovers the locus effect)")
    print(f"  within-locus expression vs AUROC     rho {r_z[0]:+.3f}  "
          f"p {r_z[1]:.4f}   <- the question")
    for gene, sub in d.groupby("Gene"):
        if len(sub) >= 8:
            rr = stats.spearmanr(sub.expr, sub.auroc)
            print(f"    within {gene:<8} n={len(sub):<3} rho {rr[0]:+.3f}  "
                  f"p {rr[1]:.4f}")

    # against the predictors already established
    extra = {}
    if Path(args.motif).exists():
        m = pd.read_csv(args.motif)[["allele", "nn_dist"]]
        d = d.merge(m, on="allele", how="left")
        extra["motif isolation"] = "nn_dist"
    if Path(args.afnd).exists():
        f = pd.read_csv(args.afnd)
        if "global_freq" in f.columns:
            d = d.merge(f[["allele", "global_freq"]], on="allele", how="left")
            extra["global frequency"] = "global_freq"

    if extra:
        print("\n=== is it independent of the established predictors? ===")
        for label, c in extra.items():
            s = d[[c, "expr_z", "auroc"]].dropna()
            if len(s) < 20:
                continue
            base = stats.spearmanr(s[c], s.auroc)
            pe = partial_spearman(s.expr_z, s.auroc, s[c])
            po = partial_spearman(s[c], s.auroc, s.expr_z)
            print(f"  {label:<20} alone {base[0]:+.3f} (p {base[1]:.4f})")
            print(f"  {'':20} expression | {label:<18} {pe[0]:+.3f} "
                  f"(p {pe[1]:.4f})")
            print(f"  {'':20} {label} | expression{'':<8} {po[0]:+.3f} "
                  f"(p {po[1]:.4f})")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(args.out, index=False)

    print(f"""
INTERPRETATION, decided before seeing the result.

A within-locus correlation is the test; the raw one recovers the locus effect and is
reported only to show that it does. If the within-locus correlation is null, the
locus-level pattern is not evidence of an expression mechanism and should be reported
as the confounded observation it is. If it holds, expression joins the list of
per-allele predictors — but subject to the four limitations in this script's
docstring, principally that this is iPSC transcript abundance rather than surface
protein in the cells immunopeptidomics uses.

Wrote {args.out}""")


if __name__ == "__main__":
    main()