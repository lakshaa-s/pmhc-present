"""RQ2: does structure add anything to sequence?

Stacks the features already computed on fold set v2 -- sequence model score, PAE
features and confidence features from three folding architectures -- and asks
whether a combined model beats sequence alone.

Design decisions that matter more than the model choice
------------------------------------------------------
LEAVE-ONE-ALLELE-OUT   Not random k-fold. Train on five alleles, test on the sixth,
                       rotate. Random folds would let the classifier learn allele
                       identity, which is trivially available in structural features
                       (the MHC chain dominates the input) and would inflate
                       everything. LOAO also matches the question the project asks:
                       does this generalise to an allele the model has not seen?

STRONG REGULARISATION  144 samples. Logistic regression with heavy L2 and
                       standardised inputs; nothing with capacity to memorise.

PAIRED COMPARISON      Sequence-only, structure-only and combined are all evaluated
                       on the same held-out predictions, so differences can be
                       bootstrapped on paired resamples rather than compared as
                       independent intervals.

Reported per model: pooled AUROC over all held-out predictions (each complex
predicted exactly once, by a model that never saw its allele), plus per-allele.

Usage:
    python scripts/rq2_stack.py \
        --sequence results/sequence_v2.csv \
        --features esmfold2_pae=pae_esmfold2_v2.csv \
                   boltz_pae=pae_boltz_v2.csv \
                   af2_pae=pae_af2_v2.csv \
                   esmfold2_conf=conf_esmfold2_v2.csv \
                   af2_conf=conf_af2_v2.csv \
                   boltz_conf=conf_boltz_v2.csv \
        --out results/rq2_stack.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

KEY = ["allele", "peptide"]
# features where a LOWER value indicates binding; sign-flipped so all point the
# same way before standardisation
LOWER_IS_BINDING = {
    "pae_pep_mhc", "pae_anchor2", "pae_anchorC", "pae_anchors", "pae_anchors_ic",
    "complex_pde", "complex_ipde",
}


def load_feature_table(path: str, prefix: str) -> pd.DataFrame:
    d = pd.read_csv(path)
    drop = [c for c in ("kind", "label", "in_train", "score") if c in d.columns]
    feats = [c for c in d.columns if c not in KEY + drop]
    out = d[KEY].copy()
    for c in feats:
        v = d[c]
        out[f"{prefix}__{c}"] = -v if c in LOWER_IS_BINDING else v
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequence", required=True,
                    help="per-peptide sequence scores (allele,peptide,label,score)")
    ap.add_argument("--features", nargs="+", required=True,
                    help="name=path structural feature CSVs")
    ap.add_argument("--C", type=float, default=0.05,
                    help="inverse L2 strength; small = heavy regularisation")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default="results/rq2_stack.csv")
    args = ap.parse_args()

    seq = pd.read_csv(args.sequence)[KEY + ["label", "score"]]
    seq = seq.rename(columns={"score": "sequence__score"})
    df = seq
    for spec in args.features:
        name, path = spec.split("=", 1)
        df = df.merge(load_feature_table(path, name), on=KEY, how="inner")

    struct_cols = [c for c in df.columns
                   if c not in KEY + ["label", "sequence__score"]]
    df = df.dropna(subset=struct_cols + ["sequence__score", "label"])
    print(f"{len(df)} complexes, {int(df.label.sum())} binders / "
          f"{int((1 - df.label).sum())} decoys")
    print(f"{len(struct_cols)} structural features, "
          f"{df.allele.nunique()} alleles\n")
    if len(df) < 100:
        print("WARNING: merge lost complexes -- check the feature files cover "
              "the same fold set")

    def clf():
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=args.C, max_iter=5000, solver="lbfgs"))

    blocks = {
        "sequence only": ["sequence__score"],
        "structure only": struct_cols,
        "sequence + structure": ["sequence__score"] + struct_cols,
    }

    preds = {}
    for name, cols in blocks.items():
        p = np.full(len(df), np.nan)
        for allele in df.allele.unique():
            te = df.allele == allele
            tr = ~te
            if df.loc[tr, "label"].nunique() < 2:
                continue
            m = clf().fit(df.loc[tr, cols], df.loc[tr, "label"])
            p[te.to_numpy()] = m.predict_proba(df.loc[te, cols])[:, 1]
        preds[name] = p

    y = df.label.to_numpy()
    rng_seed = 0

    def boot(a, b=None):
        """AUROC CI for a, or paired difference a-b."""
        r = np.random.default_rng(rng_seed)
        n, vals = len(y), []
        for _ in range(args.n_boot):
            i = r.integers(0, n, n)
            if len(np.unique(y[i])) < 2:
                continue
            v = roc_auc_score(y[i], a[i])
            if b is not None:
                v -= roc_auc_score(y[i], b[i])
            vals.append(v)
        v = np.array(vals)
        return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))

    print("=== leave-one-allele-out, pooled over held-out predictions ===")
    print(f"{'model':<22} {'AUROC':>7}  {'95% CI':>16}")
    rows = []
    for name, p in preds.items():
        auc = roc_auc_score(y, p)
        lo, hi = boot(p)
        print(f"{name:<22} {auc:>7.3f}  [{lo:.3f}, {hi:.3f}]")
        rows.append({"model": name, "scope": "pooled", "allele": "ALL",
                     "auroc": auc, "ci_lo": lo, "ci_hi": hi, "n": len(df)})

    print("\n=== paired differences ===")
    for a, b in [("sequence + structure", "sequence only"),
                 ("sequence + structure", "structure only"),
                 ("sequence only", "structure only")]:
        d = roc_auc_score(y, preds[a]) - roc_auc_score(y, preds[b])
        lo, hi = boot(preds[a], preds[b])
        sig = "yes" if lo > 0 or hi < 0 else "NO - spans zero"
        print(f"  {a} - {b:<22} {d:+.3f}  [{lo:+.3f}, {hi:+.3f}]  differs: {sig}")
        rows.append({"model": f"{a} minus {b}", "scope": "paired_diff",
                     "allele": "ALL", "auroc": d, "ci_lo": lo, "ci_hi": hi,
                     "n": len(df)})

    print("\n=== per held-out allele ===")
    print(f"{'allele':<14} " + "  ".join(f"{n:<22}" for n in preds))
    for allele in sorted(df.allele.unique()):
        m = (df.allele == allele).to_numpy()
        cells = []
        for name, p in preds.items():
            if len(np.unique(y[m])) < 2:
                cells.append("n/a")
                continue
            a = roc_auc_score(y[m], p[m])
            cells.append(f"{a:.3f}")
            rows.append({"model": name, "scope": "per_allele", "allele": allele,
                         "auroc": a, "ci_lo": np.nan, "ci_hi": np.nan,
                         "n": int(m.sum())})
        print(f"{allele:<14} " + "  ".join(f"{c:<22}" for c in cells))

    # which structural features the combined model leans on, fit on everything
    full = clf().fit(df[["sequence__score"] + struct_cols], y)
    coefs = pd.Series(full[-1].coef_[0],
                      index=["sequence__score"] + struct_cols)
    print("\n=== standardised coefficients, combined model on all data ===")
    print("(direction only -- fitted in-sample, not a performance estimate)")
    for k, v in coefs.reindex(coefs.abs().sort_values(ascending=False).index)[:12].items():
        print(f"  {k:<34} {v:+.3f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
