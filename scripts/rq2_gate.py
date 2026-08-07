"""RQ2: does gating structure on sequence uncertainty help?

The idea, and why it is worth testing despite eight prior nulls
---------------------------------------------------------------
Every combination tried so far applies structure uniformly: a fixed weight, or a
single set of learned coefficients. The per-allele table suggests that is wrong.
Across the fifteen benchmarked alleles, structure beats sequence on exactly two --
HLA-C*16:02 (gap -0.090) and HLA-C*15:05 (-0.014) -- and loses on the other
thirteen. A model that could tell those cases apart might do better than one that
cannot.

The gate has to be computable at inference time, so it cannot be "how well does
sequence do on this allele" -- that is what you are trying to predict. Three
candidates that are available per complex or per allele:

  MARGIN        |sequence score - allele median|. Small means the sequence model is
                near its own decision boundary for that allele, so it is uncertain
                about this particular peptide.

  ENTROPY       Binary entropy of the sequence model's probability. Same idea via a
                different route, and independent of the allele's score distribution.

  ANCHOR IC     The allele's anchor information content, from derive_anchors.py.
                Fixed per allele, known before any prediction, and the strongest
                predictor of per-allele sequence performance across 123 alleles
                (rho 0.660). Low IC means a weakly determined motif.

Why this is likely to fail, stated up front
--------------------------------------------
The gate must be learned from eight training alleles under leave-one-allele-out.
Per-allele AUROC at n=24 has a standard error near 0.075 against between-allele
variation of sd 0.024, so per-allele signal in this benchmark is mostly noise. A
gate fitted to that is fitting noise, and any gain it shows is likely to be the
same artefact caught in the rank-average run, where per-allele ranking alone lifted
sequence from 0.930 to 0.946 and made a null look like +0.026.

Two guards against fooling ourselves:

  LIKE FOR LIKE   The baseline is the same transformation applied to sequence alone,
                  not the raw sequence score. If ranking or z-scoring is used for
                  the gated model, the baseline gets it too.

  PERMUTATION     The gate is also evaluated with its values shuffled within the
                  fold. If a shuffled gate does as well as the real one, the gain is
                  from the machinery rather than from the gating signal.

Usage:
    python scripts/rq2_gate.py \
        --sequence results/sequence_v4.csv \
        --structure pae_af3_v4.csv --feature pae_anchors_ic \
        --anchors data/processed/anchors.json \
        --out results/rq2_gate_v4.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

KEY = ["allele", "peptide"]


def zscore(df, col):
    return df.groupby("allele")[col].transform(lambda x: (x - x.mean()) / x.std())


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
    ap.add_argument("--structure", nargs="+", required=True)
    ap.add_argument("--feature", default="pae_anchors_ic")
    ap.add_argument("--anchors", default="data/processed/anchors.json")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default="results/rq2_gate.csv")
    args = ap.parse_args()

    seq = pd.read_csv(args.sequence)[KEY + ["label", "score"]]
    df = seq.rename(columns={"score": "seq"})
    scols = []
    for path in args.structure:
        d = pd.read_csv(path)
        if args.feature not in d.columns:
            raise SystemExit(f"{path}: no column {args.feature}")
        name = Path(path).stem.replace("pae_", "").replace("_v4", "")
        df = df.merge(d[KEY].assign(**{name: -d[args.feature]}), on=KEY, how="inner")
        scols.append(name)

    y = df.label.to_numpy()
    print(f"{len(df)} complexes, {df.allele.nunique()} alleles, structure {scols}\n")

    # both sides standardised per allele, so the comparison is like for like
    df["seq_z"] = zscore(df, "seq")
    for c in scols:
        df[f"{c}_z"] = zscore(df, c)
    df["struct_z"] = df[[f"{c}_z" for c in scols]].mean(axis=1)

    # ---- gates, all computable without knowing the label ----
    # 1. distance from the allele's median sequence score, inverted: high when the
    #    sequence model is near its own boundary for this allele
    med = df.groupby("allele").seq.transform("median")
    mad = df.groupby("allele").seq.transform(lambda x: (x - x.median()).abs().median())
    df["gate_margin"] = 1.0 / (1.0 + ((df.seq - med).abs() / mad.replace(0, np.nan)))
    df["gate_margin"] = df.gate_margin.fillna(0.5)

    # 2. binary entropy of the sequence probability
    p = 1.0 / (1.0 + np.exp(-df.seq))
    p = p.clip(1e-6, 1 - 1e-6)
    df["gate_entropy"] = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))

    # 3. anchor information content, fixed per allele and known in advance
    if Path(args.anchors).exists():
        with open(args.anchors) as fh:
            anch = json.load(fh).get("alleles", {})
        ic = {a: (r["ic"][1] + r["ic"][-1]) / 2 for a, r in anch.items()}
        df["allele_ic"] = df.allele.map(ic)
        if df.allele_ic.notna().all():
            lo, hi = df.allele_ic.min(), df.allele_ic.max()
            # low IC -> weakly determined motif -> lean on structure
            df["gate_ic"] = 1.0 - (df.allele_ic - lo) / (hi - lo) if hi > lo else 0.5
        else:
            df["gate_ic"] = np.nan
    else:
        df["gate_ic"] = np.nan

    gates = [g for g in ("gate_margin", "gate_entropy", "gate_ic")
             if df[g].notna().all()]
    print(f"gates available: {gates}\n")

    base_auc = roc_auc_score(y, df.seq_z)
    print(f"sequence alone (z-scored, the like-for-like baseline)  {base_auc:.3f}")
    print(f"structure alone (z-scored)                             "
          f"{roc_auc_score(y, df.struct_z):.3f}\n")

    rows = [{"model": "sequence_z", "auroc": base_auc},
            {"model": "structure_z", "auroc": roc_auc_score(y, df.struct_z)}]

    def loao_gated(gate_col, shuffle=False, seed=0):
        """Learn a per-complex mix weight from the gate, leave-one-allele-out."""
        pred = np.full(len(df), np.nan)
        r = np.random.default_rng(seed)
        for allele in df.allele.unique():
            te = (df.allele == allele).to_numpy()
            tr = ~te
            if len(np.unique(y[tr])) < 2:
                continue
            g_tr = df.loc[tr, gate_col].to_numpy().copy()
            g_te = df.loc[te, gate_col].to_numpy().copy()
            if shuffle:
                r.shuffle(g_tr)
                r.shuffle(g_te)
            # interaction model: sequence, structure, and structure x gate.
            # A positive interaction coefficient means structure is weighted more
            # where the gate is high.
            X_tr = np.column_stack([df.loc[tr, "seq_z"], df.loc[tr, "struct_z"],
                                    df.loc[tr, "struct_z"] * g_tr])
            X_te = np.column_stack([df.loc[te, "seq_z"], df.loc[te, "struct_z"],
                                    df.loc[te, "struct_z"] * g_te])
            m = LogisticRegression(C=0.1, max_iter=5000).fit(X_tr, y[tr])
            pred[te] = m.predict_proba(X_te)[:, 1]
        return pred

    print("=== gated combination, leave-one-allele-out ===")
    print("(baseline is z-scored sequence alone, so the transformation is not "
          "credited to the gate)")
    for g in gates:
        pred = loao_gated(g)
        auc = roc_auc_score(y, pred)
        m, lo, hi = paired_boot(y, pred, df.seq_z.to_numpy(), args.n_boot)
        sig = "yes" if lo > 0 or hi < 0 else "NO - spans zero"
        print(f"  {g:<15} {auc:.3f}   vs sequence {m:+.3f} "
              f"[{lo:+.3f}, {hi:+.3f}]   differs: {sig}")
        rows.append({"model": g, "auroc": auc, "diff": m, "lo": lo, "hi": hi})

        # permutation control: does a meaningless gate do just as well?
        shuf = [roc_auc_score(y, loao_gated(g, shuffle=True, seed=s))
                for s in range(5)]
        print(f"  {'':15} shuffled gate: {np.mean(shuf):.3f} "
              f"(range {min(shuf):.3f}-{max(shuf):.3f})")
        if np.mean(shuf) >= auc - 0.005:
            print(f"  {'':15} -> the real gate does no better than a shuffled one")
        print()

    # ungated interaction-free control, to isolate what the gate adds
    pred_plain = np.full(len(df), np.nan)
    for allele in df.allele.unique():
        te = (df.allele == allele).to_numpy()
        tr = ~te
        if len(np.unique(y[tr])) < 2:
            continue
        m = LogisticRegression(C=0.1, max_iter=5000).fit(
            df.loc[tr, ["seq_z", "struct_z"]], y[tr])
        pred_plain[te] = m.predict_proba(df.loc[te, ["seq_z", "struct_z"]])[:, 1]
    auc_plain = roc_auc_score(y, pred_plain)
    m, lo, hi = paired_boot(y, pred_plain, df.seq_z.to_numpy(), args.n_boot)
    print(f"ungated sequence + structure  {auc_plain:.3f}   "
          f"vs sequence {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    rows.append({"model": "ungated", "auroc": auc_plain})

    print("\n=== per allele ===")
    print(f"{'allele':<14} {'seq':>7} {'struct':>8} " +
          " ".join(f"{g.replace('gate_',''):>9}" for g in gates))
    preds = {g: loao_gated(g) for g in gates}
    for allele in sorted(df.allele.unique()):
        mask = (df.allele == allele).to_numpy()
        if len(np.unique(y[mask])) < 2:
            continue
        cells = " ".join(f"{roc_auc_score(y[mask], preds[g][mask]):>9.3f}"
                         for g in gates)
        print(f"{allele:<14} {roc_auc_score(y[mask], df.seq_z[mask]):>7.3f} "
              f"{roc_auc_score(y[mask], df.struct_z[mask]):>8.3f} {cells}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()