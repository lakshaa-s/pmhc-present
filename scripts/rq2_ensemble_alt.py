"""RQ2: do non-linear or rank-based combinations of sequence and structure help?

Why this exists
---------------
Five linear-stacking configurations across two panels have all returned intervals
spanning zero:

    v2, all models, 21 features        -0.046 [-0.105, +0.011]
    v2, AF2 PAE only, 5 features       +0.006 [-0.043, +0.056]
    v4, three architectures' PAE       -0.022 [-0.061, +0.017]
    v4, same but per-allele z-scored   -0.001 [-0.037, +0.037]
    v4, mixed output types             -0.024 [-0.063, +0.013]

The z-scored run is the most telling: structure alone reaches 0.857, within 0.064
of sequence, and combining still gives exactly nothing. That points to redundancy
rather than weakness.

Two combination strategies remain untested, and both are things "ensemble" often
means in practice:

  RANK AVERAGE   Normalise each model's scores to ranks within an allele and
                 average. No fitting at all, so no overfitting and no
                 leave-one-allele-out needed. This is the simplest possible
                 ensemble and the one most likely to be used in practice.

  GRADIENT BOOSTING  A small tree ensemble can represent interactions a linear
                 stacker cannot -- for instance "trust structure only when
                 sequence is uncertain", which is the gated behaviour the
                 per-allele pattern originally suggested. At n=216 it will
                 probably overfit; heavy constraints (shallow trees, few
                 estimators, strong subsampling) mitigate but do not remove that.

Both are evaluated leave-one-allele-out for the fitted model, and by direct
computation for the rank average, against the same paired bootstrap used
throughout.

A weighted rank average is also swept across weights, which answers a slightly
different question: is there *any* fixed trade-off between the two scores that
beats sequence alone? If the optimum sits at weight 1.0 (all sequence), that is
about as clean a demonstration of redundancy as this design allows.

Usage:
    python scripts/rq2_ensemble_alt.py \
        --sequence results/sequence_v4.csv \
        --structure pae_af2_v4.csv --feature pae_anchors_ic \
        --out results/rq2_ensemble_alt_v4.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

KEY = ["allele", "peptide"]


def per_allele_rank(df: pd.DataFrame, col: str) -> pd.Series:
    """Rank within allele, scaled to [0, 1]. Higher = more binder-like."""
    return df.groupby("allele")[col].transform(
        lambda x: x.rank(pct=True))


def paired_boot(y, a, b, n_boot=2000, seed=0):
    r = np.random.default_rng(seed)
    n, vals = len(y), []
    for _ in range(n_boot):
        i = r.integers(0, n, n)
        if len(np.unique(y[i])) < 2:
            continue
        vals.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
    v = np.array(vals)
    return v.mean(), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequence", required=True)
    ap.add_argument("--structure", nargs="+", required=True,
                    help="one or more PAE feature CSVs")
    ap.add_argument("--feature", default="pae_anchors_ic",
                    help="which PAE column to use from each structure file")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default="results/rq2_ensemble_alt.csv")
    args = ap.parse_args()

    seq = pd.read_csv(args.sequence)[KEY + ["label", "score"]]
    seq = seq.rename(columns={"score": "seq_score"})
    df = seq
    struct_cols = []
    for i, path in enumerate(args.structure):
        d = pd.read_csv(path)
        if args.feature not in d.columns:
            raise SystemExit(f"{path}: no column {args.feature}")
        name = Path(path).stem.replace("pae_", "").replace("_v4", "")
        # PAE is lower-is-binding, so negate
        d = d[KEY].assign(**{name: -d[args.feature]})
        df = df.merge(d, on=KEY, how="inner")
        struct_cols.append(name)

    y = df.label.to_numpy()
    print(f"{len(df)} complexes, {int(y.sum())} binders / {int((1-y).sum())} decoys")
    print(f"{df.allele.nunique()} alleles | structure: {struct_cols} "
          f"({args.feature})\n")

    # ---- ranks within allele, which also removes the between-allele scale
    df["seq_rank"] = per_allele_rank(df, "seq_score")
    for c in struct_cols:
        df[f"{c}_rank"] = per_allele_rank(df, c)
    df["struct_rank"] = df[[f"{c}_rank" for c in struct_cols]].mean(axis=1)

    seq_auc = roc_auc_score(y, df.seq_score)
    str_auc = roc_auc_score(y, df.struct_rank)
    print(f"sequence alone            {seq_auc:.3f}")
    print(f"structure alone (ranked)  {str_auc:.3f}\n")

    rows = [{"model": "sequence", "auroc": seq_auc},
            {"model": "structure_rank", "auroc": str_auc}]

    # ---- weighted rank average, swept
    print("=== weighted rank average: w * sequence + (1-w) * structure ===")
    best_w, best_auc = None, -1
    for w in np.arange(0.0, 1.01, 0.1):
        comb = w * df.seq_rank + (1 - w) * df.struct_rank
        a = roc_auc_score(y, comb)
        flag = ""
        if a > best_auc:
            best_auc, best_w = a, w
        print(f"  w={w:.1f}  AUROC {a:.3f}{flag}")
        rows.append({"model": f"rank_avg_w{w:.1f}", "auroc": a})
    print(f"\nbest weight {best_w:.1f} at AUROC {best_auc:.3f}")
    if best_w >= 0.999:
        print("-> the optimum is pure sequence: no fixed trade-off with structure helps")

    comb50 = 0.5 * df.seq_rank + 0.5 * df.struct_rank
    m, lo, hi = paired_boot(y, comb50.to_numpy(), df.seq_score.to_numpy(),
                            args.n_boot)
    sig = "yes" if lo > 0 or hi < 0 else "NO - spans zero"
    print(f"\nequal-weight rank average vs sequence: {m:+.3f} [{lo:+.3f}, {hi:+.3f}]"
          f"   differs: {sig}")

    combbest = best_w * df.seq_rank + (1 - best_w) * df.struct_rank
    m, lo, hi = paired_boot(y, combbest.to_numpy(), df.seq_score.to_numpy(),
                            args.n_boot)
    sig = "yes" if lo > 0 or hi < 0 else "NO - spans zero"
    print(f"best-weight rank average vs sequence:  {m:+.3f} [{lo:+.3f}, {hi:+.3f}]"
          f"   differs: {sig}")
    print("(the best weight is chosen on the same data, so this is optimistic)")

    # ---- gradient boosting, leave-one-allele-out
    feat_cols = ["seq_score"] + struct_cols
    pred = np.full(len(df), np.nan)
    for allele in df.allele.unique():
        te = (df.allele == allele).to_numpy()
        tr = ~te
        if len(np.unique(y[tr])) < 2:
            continue
        m_ = HistGradientBoostingClassifier(
            max_depth=2, max_iter=60, learning_rate=0.06,
            min_samples_leaf=20, l2_regularization=1.0, random_state=0)
        m_.fit(df.loc[tr, feat_cols], y[tr])
        pred[te] = m_.predict_proba(df.loc[te, feat_cols])[:, 1]

    gb_auc = roc_auc_score(y, pred)
    m, lo, hi = paired_boot(y, pred, df.seq_score.to_numpy(), args.n_boot)
    sig = "yes" if lo > 0 or hi < 0 else "NO - spans zero"
    print(f"\n=== gradient boosting, leave-one-allele-out ===")
    print(f"  AUROC {gb_auc:.3f}")
    print(f"  vs sequence: {m:+.3f} [{lo:+.3f}, {hi:+.3f}]   differs: {sig}")
    rows.append({"model": "gradient_boosting", "auroc": gb_auc})

    # sequence-only gradient boosting, to separate the model class from the features
    pred_s = np.full(len(df), np.nan)
    for allele in df.allele.unique():
        te = (df.allele == allele).to_numpy()
        tr = ~te
        if len(np.unique(y[tr])) < 2:
            continue
        m_ = HistGradientBoostingClassifier(
            max_depth=2, max_iter=60, learning_rate=0.06,
            min_samples_leaf=20, l2_regularization=1.0, random_state=0)
        m_.fit(df.loc[tr, ["seq_score"]], y[tr])
        pred_s[te] = m_.predict_proba(df.loc[te, ["seq_score"]])[:, 1]
    m, lo, hi = paired_boot(y, pred, pred_s, args.n_boot)
    sig = "yes" if lo > 0 or hi < 0 else "NO - spans zero"
    print(f"  vs the same model on sequence alone ({roc_auc_score(y, pred_s):.3f}): "
          f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]   differs: {sig}")

    print("\n=== per allele ===")
    print(f"{'allele':<14} {'sequence':>9} {'struct':>8} {'rank avg':>9} {'GB':>8}")
    for allele, g in df.groupby("allele"):
        mask = (df.allele == allele).to_numpy()
        if len(np.unique(y[mask])) < 2:
            continue
        print(f"{allele:<14} {roc_auc_score(y[mask], g.seq_score):>9.3f} "
              f"{roc_auc_score(y[mask], g.struct_rank):>8.3f} "
              f"{roc_auc_score(y[mask], comb50[mask]):>9.3f} "
              f"{roc_auc_score(y[mask], pred[mask]):>8.3f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()