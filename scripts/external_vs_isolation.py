"""Does motif isolation predict per-allele performance for the field's predictors too?

The question, and why it is worth a chapter claim
--------------------------------------------------
Across 123 alleles, motif nearest-neighbour distance predicts this project's
per-allele AUROC (rho -0.291) while sample size does not (-0.118). That is a finding
about *one* model, and the obvious objection is that a 30,465-parameter CNN might
simply be too small to handle isolated alleles well.

If NetMHCpan, MixMHCpred and MHCflurry degrade the same way, the objection fails and
the claim changes character: the coverage deficit becomes a property of
sequence-based presentation prediction rather than of this implementation. That is a
substantially stronger thing to put in a discussion.

The obstacle this addresses
----------------------------
The external baselines were scored on fold set v2 (six alleles, chosen by
pseudosequence max-min) and every structural result on fold set v4 (nine alleles,
chosen by motif isolation). **The two share no alleles at all.** So the comparison
that would answer the question does not currently exist, and motif isolation — the
organising variable — varies only on the panel with no external baselines.

Pooling both fold sets gives up to fifteen alleles spanning a wider isolation range
than either alone, which is the best-powered version available without new folding.

What is computed
----------------
For each model with per-allele results on either fold set: per-allele AUROC, then the
Spearman correlation of that against the allele's motif nearest-neighbour distance.
Models are compared on the alleles they share, and the shared-allele count is printed
for every pair, because a correlation over six alleles and one over fifteen are not
the same evidence.

The power problem, stated up front
-----------------------------------
Between-allele variation in per-allele AUROC has a standard deviation near 0.024
across 123 alleles, while a fold-set AUROC from 12 binders and 12 decoys has a
standard error near 0.075. Attenuation therefore caps any observable between-allele
correlation near 0.3 even if the underlying relationship were perfect. At n=15 a null
is uninformative and only a strong effect is detectable, so:

  - a clear negative correlation across several independent predictors is
    informative, because agreement across models is not something attenuation
    produces;
  - a null for any single model is not evidence of absence and must not be reported
    as one.

The script prints the attenuation ceiling alongside every correlation for this
reason.

Usage:
    python scripts/external_vs_isolation.py \
        --results results/netmhcpan_v2.csv=NetMHCpan-4.1 \
                  results/netmhcpan_v4.csv=NetMHCpan-4.1 \
                  results/mixmhcpred_v2.csv=MixMHCpred \
                  results/mixmhcpred_v4.csv=MixMHCpred \
                  results/mhcflurry_v2.csv="MHCflurry (pres)" \
                  results/mhcflurry_v4.csv="MHCflurry (pres)" \
                  results/sequence_v2.csv=ours \
                  results/sequence_v4.csv=ours \
        --motif data/processed/motif_distinctiveness.csv \
        --out results/external_vs_isolation.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score


def per_allele(df: pd.DataFrame) -> pd.DataFrame:
    """Per-allele AUROC, skipping any allele without both classes."""
    rows = []
    for a, g in df.groupby("allele"):
        if g.label.nunique() < 2:
            continue
        rows.append({"allele": a, "auroc": roc_auc_score(g.label, g.score),
                     "n": len(g)})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", required=True,
                    help="path=modelname, repeated; several paths may share a name")
    ap.add_argument("--motif", default="data/processed/motif_distinctiveness.csv")
    ap.add_argument("--held-out-only", action="store_true",
                    help="restrict to rows with in_train False, where present")
    ap.add_argument("--out", default="results/external_vs_isolation.csv")
    args = ap.parse_args()

    motif = pd.read_csv(args.motif)
    if "nn_dist" not in motif.columns:
        raise SystemExit(f"no nn_dist column in {args.motif}: {list(motif.columns)}")

    frames: dict[str, list[pd.DataFrame]] = {}
    for spec in args.results:
        path, name = spec.rsplit("=", 1)
        if not Path(path).exists():
            print(f"  missing, skipped: {path}")
            continue
        d = pd.read_csv(path)
        if not {"allele", "label", "score"} <= set(d.columns):
            print(f"  {path}: needs allele/label/score, has {list(d.columns)}")
            continue
        if args.held_out_only and "in_train" in d.columns:
            before = len(d)
            d = d[~d.in_train.astype(str).str.lower().isin(["true", "1"])]
            print(f"  {path}: {before} -> {len(d)} rows after dropping training pairs")
        frames.setdefault(name, []).append(d)

    if not frames:
        raise SystemExit("nothing loaded")

    # per-allele AUROC per model, pooled across whichever fold sets it was scored on
    per = {}
    for name, ds in frames.items():
        d = pd.concat(ds, ignore_index=True).drop_duplicates(["allele", "peptide"])
        p = per_allele(d).merge(motif[["allele", "nn_dist"]], on="allele", how="left")
        per[name] = p.dropna(subset=["nn_dist"])

    print("\n=== allele coverage per model ===")
    for name, p in per.items():
        print(f"  {name:<20} {len(p):>2} alleles   nn_dist "
              f"{p.nn_dist.min():.3f}-{p.nn_dist.max():.3f}   "
              f"AUROC {p.auroc.min():.3f}-{p.auroc.max():.3f}")

    names = list(per)
    print("\n=== shared alleles between models ===")
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = set(per[a].allele) & set(per[b].allele)
            print(f"  {a:<20} vs {b:<20} {len(shared):>2} shared")

    # the attenuation ceiling, so no correlation is read without it
    sd_signal, se_meas = 0.024, 0.075
    ceiling = sd_signal / np.sqrt(sd_signal ** 2 + se_meas ** 2)
    print(f"\n  attenuation ceiling on any observable correlation: ~{ceiling:.2f}")
    print("  (between-allele sd 0.024 against per-allele SE 0.075)")

    print("\n=== does motif isolation predict per-allele AUROC? ===")
    print(f"{'model':<20} {'n':>3} {'rho':>8} {'p':>8}   interpretation")
    rows = []
    for name, p in per.items():
        if len(p) < 4:
            print(f"{name:<20} {len(p):>3}      too few alleles")
            continue
        r = stats.spearmanr(p.nn_dist, p.auroc)
        if r[1] < 0.05 and r[0] < 0:
            verdict = "degrades with isolation"
        elif r[0] < -0.2:
            verdict = "trend, underpowered at this n"
        else:
            verdict = "no detectable relationship"
        rows.append({"model": name, "n_alleles": len(p),
                     "rho": round(r[0], 3), "p": round(r[1], 4),
                     "verdict": verdict})
        print(f"{name:<20} {len(p):>3} {r[0]:>+8.3f} {r[1]:>8.3f}   {verdict}")

    out = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    # significance matters more than count once the panel is large enough for any
    # single correlation to be interpretable on its own
    sig = out[(out.rho < 0) & (out.p < 0.05)]
    neg = out[out.rho < -0.2]
    print()
    if len(sig) >= 2 and out.n_alleles.min() >= 50:
        print(f"  {len(sig)} of {len(out)} predictors show a significant negative")
        print("  relationship at a panel size where each is interpretable alone.")
        print("  Agreement across independently developed models on different")
        print("  training data makes this a property of the approach rather than")
        print("  of any one implementation.")
    elif len(neg) >= 3:
        print(f"  {len(neg)} of {len(out)} predictors show a negative trend. Agreement")
        print("  across independently developed models is not something attenuation")
        print("  produces, so this supports the coverage deficit being a property of")
        print("  sequence-based presentation prediction rather than of one model.")
    elif len(neg) >= 1:
        print(f"  {len(neg)} of {len(out)} show a negative trend — suggestive but not")
        print("  enough models agreeing to rule out chance at this allele count.")
    else:
        print("  No predictor shows the relationship. Given the attenuation ceiling")
        print("  above, this is uninformative rather than negative: report it as a")
        print("  limitation of the panel size, not as evidence of absence.")

    print("\n  NOTE: the fold sets differ in how binders were selected (top-decile PWM")
    print("  score), so every model's absolute AUROC here is inflated by coupling to")
    print("  that criterion. The correlation with isolation is the quantity of")
    print("  interest, not the level.")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()