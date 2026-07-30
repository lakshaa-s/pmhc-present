"""Bootstrap confidence intervals for the RQ1 comparison.

Every AUROC reported so far is a bare point estimate. This resamples complexes with
replacement to put intervals on them, and does two things the point estimates cannot:

  PAIRED DIFFERENCES  sequence and structure are scored on the *same* complexes, so
                      the difference can be bootstrapped on paired resamples. That is
                      a much tighter test than comparing two independent intervals.

  PER-ALLELE WIDTH    at n=24 per allele the intervals should be wide enough to
                      explain why per-allele structural AUROCs disagree across
                      architectures (C*16:02 spans 0.576 to 0.944). Quantifying that
                      turns "underpowered" from an assertion into a measurement.

Input files are the per-peptide score CSVs (allele, peptide, label, score) written by
score_sequence_on_foldset.py and score_mhcflurry.py, plus the per-fold feature CSVs
from analyse_pae.py / analyse_pae_af2.py (allele, peptide, kind, pae_*).

Usage:
    python scripts/bootstrap_auroc.py \
        --scores sequence=results/sequence_v2.csv \
                 mhcflurry_pres=results/mhcflurry_v2.csv \
                 mhcflurry_aff=results/mhcflurry_v2_affinity.csv \
        --pae esmfold2=pae_esmfold2_v2.csv boltz=pae_boltz_v2.csv af2=pae_af2_v2.csv \
        --feature pae_anchors \
        --out results/bootstrap_ci.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def load_scores(path: str) -> pd.DataFrame:
    d = pd.read_csv(path)
    return d[["allele", "peptide", "label", "score"]]


def load_pae(path: str, feature: str) -> pd.DataFrame:
    d = pd.read_csv(path)
    if feature not in d.columns:
        raise SystemExit(f"{path}: no column {feature}; has {list(d.columns)}")
    out = d[["allele", "peptide"]].copy()
    out["label"] = (d.kind == "binder").astype(int)
    out["score"] = -d[feature]        # PAE: lower = more binder-like
    return out


def boot_auroc(label, score, n_boot, rng):
    """Bootstrap AUROC, resampling complexes with replacement."""
    label, score = np.asarray(label), np.asarray(score)
    n = len(label)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(label[idx])) < 2:
            continue
        vals.append(roc_auc_score(label[idx], score[idx]))
    return np.array(vals)


def ci(vals, lo=2.5, hi=97.5):
    return (float(np.percentile(vals, lo)), float(np.percentile(vals, hi)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", nargs="*", default=[],
                    help="name=path per-peptide score CSVs")
    ap.add_argument("--pae", nargs="*", default=[],
                    help="name=path per-fold PAE feature CSVs")
    ap.add_argument("--feature", default="pae_anchors")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--baseline", default="sequence",
                    help="model to take paired differences against")
    ap.add_argument("--out", default="results/bootstrap_ci.csv")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    models: dict[str, pd.DataFrame] = {}
    for spec in args.scores:
        name, path = spec.split("=", 1)
        models[name] = load_scores(path)
    for spec in args.pae:
        name, path = spec.split("=", 1)
        models[name] = load_pae(path, args.feature)

    if not models:
        raise SystemExit("No models given.")

    # align every model on the same (allele, peptide) complexes
    keys = None
    for name, d in models.items():
        k = set(zip(d.allele, d.peptide))
        keys = k if keys is None else keys & k
    keys = sorted(keys)
    print(f"{len(keys)} complexes common to all {len(models)} models "
          f"({args.n_boot} bootstrap resamples)\n")

    aligned = {}
    for name, d in models.items():
        d = d.set_index(["allele", "peptide"]).loc[keys].reset_index()
        aligned[name] = d
    labels = aligned[next(iter(aligned))].label.to_numpy()

    rows = []
    print(f"{'model':<18} {'pooled AUROC':>13}  {'95% CI':>16}  {'width':>6}")
    boots = {}
    for name, d in aligned.items():
        pt = roc_auc_score(d.label, d.score)
        b = boot_auroc(d.label, d.score, args.n_boot, np.random.default_rng(args.seed))
        boots[name] = b
        lo, hi = ci(b)
        print(f"{name:<18} {pt:>13.3f}  [{lo:.3f}, {hi:.3f}]  {hi - lo:>6.3f}")
        rows.append({"model": name, "scope": "pooled", "allele": "ALL",
                     "auroc": pt, "ci_lo": lo, "ci_hi": hi, "n": len(d)})

    if args.baseline in aligned:
        print(f"\n=== paired difference vs {args.baseline} (same complexes) ===")
        base = aligned[args.baseline].score.to_numpy()
        n = len(labels)
        for name, d in aligned.items():
            if name == args.baseline:
                continue
            other = d.score.to_numpy()
            r = np.random.default_rng(args.seed)
            diffs = []
            for _ in range(args.n_boot):
                idx = r.integers(0, n, n)
                if len(np.unique(labels[idx])) < 2:
                    continue
                diffs.append(roc_auc_score(labels[idx], base[idx])
                             - roc_auc_score(labels[idx], other[idx]))
            diffs = np.array(diffs)
            lo, hi = ci(diffs)
            sig = "yes" if lo > 0 or hi < 0 else "NO — interval spans zero"
            print(f"  {args.baseline} - {name:<16} {diffs.mean():+.3f}  "
                  f"[{lo:+.3f}, {hi:+.3f}]   differs: {sig}")
            rows.append({"model": f"{args.baseline}-minus-{name}", "scope": "paired_diff",
                         "allele": "ALL", "auroc": float(diffs.mean()),
                         "ci_lo": lo, "ci_hi": hi, "n": n})

    print(f"\n=== per allele (n=24 each) ===")
    alleles = sorted(aligned[next(iter(aligned))].allele.unique())
    print(f"{'allele':<14} " + "  ".join(f"{m:<20}" for m in aligned))
    for allele in alleles:
        cells = []
        for name, d in aligned.items():
            g = d[d.allele == allele]
            pt = roc_auc_score(g.label, g.score)
            b = boot_auroc(g.label, g.score, args.n_boot,
                           np.random.default_rng(args.seed))
            lo, hi = ci(b)
            cells.append(f"{pt:.2f} [{lo:.2f},{hi:.2f}]")
            rows.append({"model": name, "scope": "per_allele", "allele": allele,
                         "auroc": pt, "ci_lo": lo, "ci_hi": hi, "n": len(g)})
        print(f"{allele:<14} " + "  ".join(f"{c:<20}" for c in cells))

    res = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.out, index=False)

    pa = res[res.scope == "per_allele"]
    print(f"\nmedian per-allele CI width: {(pa.ci_hi - pa.ci_lo).median():.3f}")
    po = res[res.scope == "pooled"]
    print(f"median pooled CI width:     {(po.ci_hi - po.ci_lo).median():.3f}")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()