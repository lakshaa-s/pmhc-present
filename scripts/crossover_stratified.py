"""Is the motif-isolation effect an artefact of cross-allele crossover?

The concern
-----------
Three predictors degrade with motif isolation at n=123 (MHCflurry -0.393, NetMHCpan
-0.248, ours -0.196 on the matched sample). But 92.2% of the sample's unique peptides
also appear in training under some other allele, because clustering is within allele
by design. A peptide presented by twelve alleles is a positive for those twelve and a
negative for every other, so the labels are contradictory across alleles and a model
can learn "this peptide is usually a negative" independently of the groove. That is
the crossover artefact, measured at 0.3596 by the peptide-identity prior.

If per-allele performance were driven by how promiscuous an allele's peptides are
rather than by its motif, the isolation correlation could be that artefact wearing a
different name.

Why not simply filter
---------------------
Restricting to peptides unseen in training leaves 9,118 of 104,804 validation 9mers
(8.7%), giving 53 alleles at 15 pairs per class — a per-allele standard error near
0.13 against 0.050 on the full sample, so *less* power, not more. Worse, the filter is
not neutral: negatives are drawn from other alleles' repertoires and are therefore
promiscuous by construction, so filtering removes them preferentially and leaves a
positive-heavy set whose surviving negatives are systematically unusual. A null under
those conditions would be uninterpretable.

What this does instead
----------------------
Every peptide is assigned a **training promiscuity** — the number of distinct alleles
presenting it in the training split, with zero meaning unseen. All 123 alleles are
retained and the analysis is run within promiscuity strata, so each stratum is a
separate test at full allele coverage rather than one filtered set at reduced
coverage.

  PER STRATUM   Within each promiscuity band, per-allele AUROC against motif
                isolation. If the effect holds where crossover is low, it is not the
                artefact.

  ALLELE-LEVEL  Does an allele's *mean* peptide promiscuity predict its AUROC? This is
                the direct form of the worry: if promiscuous-repertoire alleles score
                worse, the per-allele variation may be crossover rather than motif.

  PARTIAL       Motif isolation against AUROC controlling for mean promiscuity, and
                the reverse. If isolation survives and promiscuity does not, the
                concern is answered.

Interpretation, fixed before running
-------------------------------------
The isolation effect survives if it holds in the low-promiscuity strata and in the
partial. It is compromised if it appears only where crossover is high, or if
controlling for mean promiscuity removes it. A stratum with too few alleles to
correlate is reported as such rather than folded into a neighbour.

Note that a shared artefact cannot easily explain agreement across NetMHCpan and
MHCflurry, which were trained by other groups on their own corpora and never saw this
split — so the cross-model concordance is itself evidence against the artefact
reading. This script tests the concern directly rather than relying on that argument.

Usage:
    python scripts/crossover_stratified.py \
        --sample fold_sets/validation_sample_123.csv \
        --data data/processed/atlas_labelled_v2.csv \
        --split data/processed/split_val_v2.csv \
        --results results/sequence_val123.csv=ours \
                  results/mhcflurry_val123.csv="MHCflurry" \
                  results/netmhcpan_val123.csv=NetMHCpan \
        --motif data/processed/motif_distinctiveness.csv \
        --out results/crossover_stratified.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score


def partial_spearman(x, y, z):
    xr, yr, zr = (stats.rankdata(v) for v in (x, y, z))
    Z = np.column_stack([np.ones_like(zr), zr])
    rx = xr - Z @ np.linalg.lstsq(Z, xr, rcond=None)[0]
    ry = yr - Z @ np.linalg.lstsq(Z, yr, rcond=None)[0]
    r, p = stats.pearsonr(rx, ry)
    return float(r), float(p)


def per_allele_auroc(df, score="score"):
    rows = []
    for a, g in df.groupby("allele"):
        if g.label.nunique() < 2 or len(g) < 8:
            continue
        rows.append({"allele": a, "auroc": roc_auc_score(g.label, g[score]),
                     "n": len(g)})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--results", nargs="+", required=True, help="path=name")
    ap.add_argument("--motif", default="data/processed/motif_distinctiveness.csv")
    ap.add_argument("--length", type=int, default=9)
    ap.add_argument("--min-alleles", type=int, default=25,
                    help="minimum alleles for a stratum to be correlated")
    ap.add_argument("--out", default="results/crossover_stratified.csv")
    args = ap.parse_args()

    # training promiscuity: how many alleles present this peptide in TRAINING
    a = pd.read_csv(args.data)
    a = a[a.peptide.str.len() == args.length]
    v = set(map(tuple, pd.read_csv(args.split).values))
    in_val = np.array([(x, y) in v for x, y in zip(a.allele, a.peptide)])
    train_pos = a[(~in_val) & (a.label == 1)]
    promisc = train_pos.groupby("peptide").allele.nunique()
    print(f"{len(promisc):,} distinct peptides presented in training, "
          f"by {promisc.min()}-{promisc.max()} alleles each")

    motif = pd.read_csv(args.motif)[["allele", "nn_dist"]]
    rows = []

    for spec in args.results:
        path, name = spec.rsplit("=", 1)
        if not Path(path).exists():
            print(f"  missing, skipped: {path}")
            continue
        d = pd.read_csv(path)
        if not {"allele", "peptide", "label", "score"} <= set(d.columns):
            print(f"  {path}: wrong columns, skipped")
            continue
        d["promisc"] = d.peptide.map(promisc).fillna(0).astype(int)

        print(f"\n{'=' * 62}\n{name}  ({len(d):,} pairs)")
        share = (d.promisc > 0).mean()
        print(f"  {share:.1%} of pairs use a peptide seen in training under some allele")
        print(f"  median training promiscuity: {int(d.promisc.median())} alleles")

        # ---- allele-level: does mean promiscuity predict AUROC? ----
        pa = per_allele_auroc(d)
        mp = d.groupby("allele").promisc.mean().rename("mean_promisc")
        pa = pa.merge(mp, on="allele").merge(motif, on="allele", how="left").dropna()

        r_iso = stats.spearmanr(pa.nn_dist, pa.auroc)
        r_pro = stats.spearmanr(pa.mean_promisc, pa.auroc)
        r_ip = stats.spearmanr(pa.nn_dist, pa.mean_promisc)
        pi = partial_spearman(pa.nn_dist, pa.auroc, pa.mean_promisc)
        pp = partial_spearman(pa.mean_promisc, pa.auroc, pa.nn_dist)

        print(f"\n  allele level, n={len(pa)}")
        print(f"    motif isolation  vs AUROC          {r_iso[0]:+.3f}  p {r_iso[1]:.4f}")
        print(f"    mean promiscuity vs AUROC          {r_pro[0]:+.3f}  p {r_pro[1]:.4f}")
        print(f"    isolation vs promiscuity           {r_ip[0]:+.3f}  p {r_ip[1]:.4f}")
        print(f"    isolation | promiscuity            {pi[0]:+.3f}  p {pi[1]:.4f}   <-")
        print(f"    promiscuity | isolation            {pp[0]:+.3f}  p {pp[1]:.4f}")

        rows.append({"model": name, "level": "allele", "stratum": "all",
                     "n_alleles": len(pa), "rho_isolation": round(r_iso[0], 3),
                     "p_isolation": round(r_iso[1], 4),
                     "rho_isolation_partial": round(pi[0], 3),
                     "p_isolation_partial": round(pi[1], 4),
                     "rho_promiscuity": round(r_pro[0], 3),
                     "p_promiscuity": round(r_pro[1], 4)})

        # ---- stratified by peptide promiscuity ----
        # bands chosen on the distribution, not on the outcome
        q = d.promisc.quantile([.25, .5, .75]).astype(int).tolist()
        edges = sorted(set([-1, 0] + q + [d.promisc.max()]))
        labels = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            labels.append((lo, hi, f"{lo+1}-{hi}" if hi > lo + 1 else f"{hi}"))
        print(f"\n  by peptide promiscuity (alleles presenting it in training)")
        print(f"  {'stratum':<12} {'pairs':>7} {'alleles':>8} {'rho':>8} {'p':>8}")
        for lo, hi, lab in labels:
            sub = d[(d.promisc > lo) & (d.promisc <= hi)]
            if len(sub) < 200:
                continue
            ps = per_allele_auroc(sub).merge(motif, on="allele",
                                             how="left").dropna()
            if len(ps) < args.min_alleles:
                print(f"  {lab:<12} {len(sub):>7,} {len(ps):>8}   "
                      f"too few alleles to correlate")
                continue
            rr = stats.spearmanr(ps.nn_dist, ps.auroc)
            print(f"  {lab:<12} {len(sub):>7,} {len(ps):>8} {rr[0]:>+8.3f} "
                  f"{rr[1]:>8.4f}")
            rows.append({"model": name, "level": "stratum", "stratum": lab,
                         "n_alleles": len(ps), "rho_isolation": round(rr[0], 3),
                         "p_isolation": round(rr[1], 4),
                         "rho_isolation_partial": np.nan,
                         "p_isolation_partial": np.nan,
                         "rho_promiscuity": np.nan, "p_promiscuity": np.nan})

    out = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    print(f"\n{'=' * 62}")
    al = out[out.level == "allele"]
    survives = al[(al.rho_isolation_partial < 0) & (al.p_isolation_partial < 0.05)]
    print(f"isolation survives controlling for mean promiscuity in "
          f"{len(survives)} of {len(al)} models")
    lowest = out[(out.level == "stratum") & (out.stratum.isin(["0", "1"]))]
    if len(lowest):
        print(f"in the lowest-promiscuity strata: "
              f"{(lowest.rho_isolation < 0).sum()} of {len(lowest)} negative, "
              f"{((lowest.rho_isolation < 0) & (lowest.p_isolation < 0.05)).sum()} "
              f"significant")
    print("""
Read the partial as the test. If isolation survives controlling for mean promiscuity,
the per-allele effect is not the cross-allele crossover artefact under another name.
The strata are supporting evidence: a correlation that holds where crossover is low
cannot be produced by crossover.

Strata with fewer than the minimum alleles are reported as such rather than merged,
because merging them would reintroduce the high-crossover pairs the stratum exists to
exclude.""")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()