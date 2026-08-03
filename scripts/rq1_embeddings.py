"""Do AF2's learned representations carry signal its confidence metrics do not?

Why this exists
---------------
Every structural feature tested so far is a model self-assessment: PAE (best 0.804),
confidence metrics (0.753) and geometry computed from coordinates (0.492), against
a sequence model at 0.921. That supports a negative RQ1, but only for *off-the-shelf
structural confidence*. It says nothing about whether the folding model internally
encodes binding-relevant information that its confidence outputs fail to expose.

Motmaen et al. showed that fine-tuning AlphaFold closes most of the gap to sequence
models, and the 2025 AF3 work feeds learned representations to a downstream
classifier rather than reading confidence. This tests the same hypothesis at a
fraction of the cost: freeze the folding model, take the representations it already
produced, and train a light classifier on them.

If the representations beat 0.804, the information is present but not exposed by the
confidence outputs, and RQ1's conclusion must be stated as being about confidence
metrics specifically. If they land near 0.804 too, the representations genuinely
lack the signal, and only fine-tuning could distinguish model from readout.

Design decisions that matter more than the classifier
------------------------------------------------------
PEPTIDE POOLING     The representation is (n_residues, 256) over the concatenated
                    MHC + b2m + peptide chains. Only the final `peptide_len` rows
                    are the peptide. Mean-pooling the whole complex would be
                    dominated by the ~373 MHC residues, which are identical for
                    every complex of a given allele and therefore encode allele
                    identity rather than binding.

LEAVE-ONE-ALLELE-OUT  Not random k-fold. The MHC portion makes allele identity
                    trivially recoverable, and random folds would reward learning it.
                    LOAO also matches the question the project asks.

PCA THEN L2         256 dimensions against 144 samples. PCA is fitted on the
                    training alleles only, inside each fold, so the held-out allele
                    does not inform the projection.

PAIRED BOOTSTRAP    Against sequence (0.921) and AF2 `pae_anchors` (0.804) on the
                    same complexes, so the differences can be tested directly.

Usage:
    python scripts/rq1_embeddings.py \
        third_party/HISTOFold/outputs/experiments/fold_set_v2emb_v3b \
        --fold-set fold_sets/fold_set_v2.csv \
        --sequence results/sequence_v2.csv \
        --pae pae_af2_v3b.csv \
        --out results/embeddings_af2.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def slug_to_allele(slug: str) -> str:
    b = slug.split("_")
    return f"HLA-{b[1].upper()}*{b[2]}:{b[3]}" if len(b) >= 4 else slug


def load_fold_set(path: str) -> dict:
    """Accept both HISTOFold naming schemes as a lookup key."""
    out = {}
    with open(path) as fh:
        for row in csv.reader(fh):
            if len(row) < 4:
                continue
            tag, _locus, slug, peptide = row[0], row[1], row[2], row[3]
            meta = (slug_to_allele(slug), peptide, tag in ("decoy", "hard"))
            out[f"{slug}_{peptide.lower()}"] = meta          # v2 naming
            out[f"{tag}__{slug}__{peptide.lower()}"] = meta   # v3 naming
    return out


def load_repr(fold: Path, peptide_len: int, rank: str, pool: str):
    hits = sorted(fold.glob(f"*_single_repr_rank_{rank}_*.npy"))
    if not hits:
        return None
    a = np.load(hits[0]).astype(np.float32)
    if a.ndim != 2 or a.shape[0] <= peptide_len:
        return None
    pep = a[-peptide_len:]          # peptide is the final chain
    if pool == "mean":
        return pep.mean(axis=0)
    if pool == "concat":
        return pep.reshape(-1)      # position-specific, peptide_len * 256 dims
    if pool == "meanmax":
        return np.concatenate([pep.mean(axis=0), pep.max(axis=0)])
    raise ValueError(pool)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--fold-set", required=True)
    ap.add_argument("--sequence", help="per-peptide sequence scores, for comparison")
    ap.add_argument("--pae", help="PAE feature CSV, for comparison")
    ap.add_argument("--pae-feature", default="pae_anchors")
    ap.add_argument("--rank", default="001")
    ap.add_argument("--pool", default="mean", choices=["mean", "concat", "meanmax"])
    ap.add_argument("--n-components", type=int, default=15)
    ap.add_argument("--C", type=float, default=0.05)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default="results/embeddings_af2.csv")
    args = ap.parse_args()

    fold_set = load_fold_set(args.fold_set)
    rows, X, missing = [], [], []
    for fold in sorted(Path(args.root).iterdir()):
        if not fold.is_dir():
            continue
        meta = fold_set.get(fold.name)
        if meta is None:
            missing.append(fold.name)
            continue
        allele, peptide, is_decoy = meta
        v = load_repr(fold, len(peptide), args.rank, args.pool)
        if v is None:
            missing.append(fold.name)
            continue
        rows.append({"allele": allele, "peptide": peptide,
                     "label": 0 if is_decoy else 1})
        X.append(v)

    if missing:
        print(f"WARNING: skipped {len(missing)}: {missing[:3]}")
    if not rows:
        raise SystemExit("No representations loaded.")

    df = pd.DataFrame(rows)
    X = np.vstack(X)
    y = df.label.to_numpy()
    print(f"{len(df)} complexes, {int(y.sum())} binders / {int((1-y).sum())} decoys")
    print(f"{df.allele.nunique()} alleles | representation {X.shape} "
          f"(pool={args.pool}, rank={args.rank})\n")

    n_comp = min(args.n_components, X.shape[1], len(df) - df.allele.value_counts().max())
    if n_comp < args.n_components:
        print(f"reducing PCA components to {n_comp} to fit the training folds\n")

    pred = np.full(len(df), np.nan)
    for allele in df.allele.unique():
        te = (df.allele == allele).to_numpy()
        tr = ~te
        if len(np.unique(y[tr])) < 2:
            continue
        # PCA fitted on training alleles only, inside the fold
        m = make_pipeline(StandardScaler(),
                          PCA(n_components=n_comp, random_state=0),
                          LogisticRegression(C=args.C, max_iter=5000))
        m.fit(X[tr], y[tr])
        pred[te] = m.predict_proba(X[te])[:, 1]

    df["score"] = pred
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    def boot(a, b=None, seed=0):
        r = np.random.default_rng(seed)
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

    auc = roc_auc_score(y, pred)
    lo, hi = boot(pred)
    print("=== AF2 single representations, leave-one-allele-out ===")
    print(f"  pooled AUROC {auc:.3f}  [{lo:.3f}, {hi:.3f}]\n")

    print("=== per held-out allele ===")
    for allele, g in df.groupby("allele"):
        if g.label.nunique() > 1:
            print(f"  {allele:<14} {roc_auc_score(g.label, g.score):.3f}  (n={len(g)})")

    # comparisons on the same complexes
    others = {}
    if args.sequence and Path(args.sequence).exists():
        s = pd.read_csv(args.sequence)[["allele", "peptide", "score"]]
        others["sequence"] = s
    if args.pae and Path(args.pae).exists():
        p = pd.read_csv(args.pae)
        if args.pae_feature in p.columns:
            p = p[["allele", "peptide", args.pae_feature]].copy()
            p["score"] = -p[args.pae_feature]
            others[f"AF2 {args.pae_feature}"] = p[["allele", "peptide", "score"]]

    if others:
        print("\n=== paired differences (same complexes) ===")
        for name, o in others.items():
            m = df.merge(o, on=["allele", "peptide"], how="inner",
                         suffixes=("", "_other"))
            if len(m) != len(df):
                print(f"  {name}: only {len(m)}/{len(df)} matched, skipping")
                continue
            a, b = m.score.to_numpy(), m.score_other.to_numpy()
            yy = m.label.to_numpy()
            r = np.random.default_rng(0)
            diffs = []
            for _ in range(args.n_boot):
                i = r.integers(0, len(yy), len(yy))
                if len(np.unique(yy[i])) < 2:
                    continue
                diffs.append(roc_auc_score(yy[i], a[i]) - roc_auc_score(yy[i], b[i]))
            diffs = np.array(diffs)
            d_lo, d_hi = np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)
            sig = "yes" if d_lo > 0 or d_hi < 0 else "NO - spans zero"
            print(f"  embeddings - {name:<20} {diffs.mean():+.3f}  "
                  f"[{d_lo:+.3f}, {d_hi:+.3f}]   differs: {sig}")

    print(f"\nWrote {args.out}")
    print("\nInterpretation: if this beats AF2's best confidence feature (0.804 for")
    print("pae_anchors on v3b), the representations encode binding signal the")
    print("confidence outputs do not expose, and RQ1's negative result should be")
    print("stated as being about confidence metrics specifically. If it lands near")
    print("0.804, the representations lack the signal too.")


if __name__ == "__main__":
    main()