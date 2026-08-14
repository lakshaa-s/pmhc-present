"""Do the per-allele predictors survive the negative-sampling confound?

The question
------------
`hlac_partial_effect.py` established that the HLA-C locus effect survives controlling
for confound strength (-0.0267 in OLS, -0.0269 matched). But three other per-allele
findings rest on the same AUROC column and have not been checked:

    anchor information content          rho +0.660   p 1.1e-16
    motif nearest-neighbour distance    rho -0.363   p 3.7e-05
    log10(peptide count)                rho -0.020   p 0.82

These are reported as well-powered findings across 123 alleles, and the third is used
to argue that data volume does not drive performance. If confound strength correlates
with any of them, the corresponding result is contaminated.

There is specific reason to worry. Confound strength correlates with per-allele AUROC
at rho +0.302 (p 0.0007), so it is a live contaminant of anything computed from that
column. And it is not obviously independent of the predictors: an allele with a
sharply determined motif presents a more restricted peptide set, which plausibly
affects how often its peptides appear as other alleles' negatives.

What is computed
----------------
For each predictor, three things:

  RAW              the correlation as currently reported, recomputed here so the
                   comparison is like-for-like.

  PARTIAL          Spearman partial correlation controlling for confound strength.
                   If this holds near the raw value, the finding is not explained by
                   the artefact.

  CONFOUND LINK    the predictor's own correlation with confound strength. A
                   predictor uncorrelated with the confound cannot be contaminated by
                   it, whatever the partial says, so this is the cleanest single
                   check.

Interpretation is stated in advance rather than fitted afterwards: a predictor whose
partial correlation retains most of its raw magnitude, and which is itself weakly
correlated with confound strength, is safe to report. One that loses most of its
magnitude is not.

Usage
-----
    python scripts/predictors_vs_confound.py \
        --prior results/crossover_prior_per_allele.csv \
        --auroc results/per_allele_auroc.csv \
        --anchors data/processed/anchors.json \
        --motif data/processed/motif_distinctiveness.csv \
        --out results/predictors_vs_confound.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def partial_spearman(x, y, z):
    """Spearman partial correlation of x and y controlling for z.

    Ranks first, then residualises both against z by least squares, then correlates
    the residuals. Equivalent to the usual formula but reports n and p directly.
    """
    xr, yr, zr = (stats.rankdata(v) for v in (x, y, z))
    Z = np.column_stack([np.ones_like(zr), zr])
    bx = np.linalg.lstsq(Z, xr, rcond=None)[0]
    by = np.linalg.lstsq(Z, yr, rcond=None)[0]
    rx, ry = xr - Z @ bx, yr - Z @ by
    r, p = stats.pearsonr(rx, ry)
    return float(r), float(p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior", required=True,
                    help="crossover_prior_per_allele.csv")
    ap.add_argument("--auroc", required=True, help="per_allele_auroc.csv")
    ap.add_argument("--anchors", default="data/processed/anchors.json")
    ap.add_argument("--motif", default="data/processed/motif_distinctiveness.csv")
    ap.add_argument("--out", default="results/predictors_vs_confound.csv")
    args = ap.parse_args()

    prior = pd.read_csv(args.prior)
    auroc = pd.read_csv(args.auroc)

    # column names differ between the two files; find them rather than assume
    def pick(df, *names):
        for n in names:
            if n in df.columns:
                return n
        raise SystemExit(f"none of {names} in {list(df.columns)}")

    pa = pick(prior, "allele")
    pv = pick(prior, "prior_auroc", "auroc", "confound_auroc")
    aa = pick(auroc, "allele")
    av = pick(auroc, "auroc", "val_auroc", "model_auroc")

    d = (prior[[pa, pv]].rename(columns={pa: "allele", pv: "prior"})
         .merge(auroc[[aa, av]].rename(columns={aa: "allele", av: "auroc"}),
                on="allele"))
    d["confound"] = (d.prior - 0.5).abs()

    # the three predictors
    anch = json.loads(Path(args.anchors).read_text()).get("alleles", {})
    d["anchor_ic"] = d.allele.map(
        lambda a: float(np.mean([anch[a]["ic"][i] for i in anch[a]["anchors"]]))
        if a in anch and anch[a].get("anchors") else np.nan)

    if Path(args.motif).exists():
        m = pd.read_csv(args.motif)
        cols = [c for c in ("nn_dist", "n_peptides") if c in m.columns]
        d = d.merge(m[["allele"] + cols], on="allele", how="left")
    if "n_peptides" in d.columns:
        d["log_n"] = np.log10(d.n_peptides.clip(lower=1))

    preds = [c for c in ("anchor_ic", "nn_dist", "log_n") if c in d.columns]
    print(f"{len(d)} alleles matched; predictors: {preds}")
    print(f"confound vs AUROC: rho "
          f"{stats.spearmanr(d.confound, d.auroc)[0]:+.3f} "
          f"p {stats.spearmanr(d.confound, d.auroc)[1]:.4f}\n")

    print(f"{'predictor':<14} {'raw':>16} {'partial':>16} "
          f"{'vs confound':>16}  verdict")
    rows = []
    for c in preds:
        s = d[[c, "auroc", "confound"]].dropna()
        raw = stats.spearmanr(s[c], s.auroc)
        par = partial_spearman(s[c], s.auroc, s.confound)
        link = stats.spearmanr(s[c], s.confound)

        # a predictor barely correlated with the confound cannot be much
        # contaminated by it, whatever the partial happens to be
        retained = abs(par[0]) / abs(raw[0]) if raw[0] else np.nan
        if abs(link[0]) < 0.15:
            v = "safe (independent of the confound)"
        elif retained > 0.8:
            v = "survives"
        elif retained > 0.5:
            v = "attenuated — report the partial"
        else:
            v = "LARGELY EXPLAINED BY THE CONFOUND"

        rows.append({"predictor": c, "n": len(s),
                     "raw_rho": round(raw[0], 3), "raw_p": raw[1],
                     "partial_rho": round(par[0], 3), "partial_p": par[1],
                     "rho_with_confound": round(link[0], 3),
                     "fraction_retained": round(retained, 3), "verdict": v})
        print(f"{c:<14} {raw[0]:>+8.3f} (p{raw[1]:.0e}) "
              f"{par[0]:>+8.3f} (p{par[1]:.0e}) "
              f"{link[0]:>+8.3f} (p{link[1]:.0e})  {v}")

    out = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    print("\n  raw          the correlation as currently reported")
    print("  partial      controlling for confound strength")
    print("  vs confound  the predictor's own link to the confound; below ~0.15")
    print("               it cannot be much contaminated whatever the partial says")

    bad = out[out.verdict.str.startswith("LARGELY")]
    if bad.empty:
        print("\n  All predictors survive. The per-allele findings can be reported")
        print("  from the current data with the confound noted, and regenerating")
        print("  is not required to make them defensible.")
    else:
        print(f"\n  {len(bad)} predictor(s) do not survive: "
              f"{list(bad.predictor)}")
        print("  Those findings need the regenerated data before they can be")
        print("  reported, or must be reported as partial correlations only.")

    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()