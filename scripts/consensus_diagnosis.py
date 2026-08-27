"""Is the structural consensus complementary information, or variance reduction?

The finding being tested
------------------------
Averaging the per-allele z-scored anchor PAE across four folding architectures gives
AUROC 0.905 against 0.858 for the best single architecture, paired difference
+0.046 [+0.003, +0.091]. Sequence minus consensus then spans zero, where sequence
minus any individual architecture does not.

That is real and it was a genuine gap — the project compared each architecture
against sequence and never against the others. But "the architectures add to each
other" admits two readings with different consequences, and the difference matters
for what RQ1 can claim.

  COMPLEMENTARY     Each architecture carries binding-relevant information the
                    others lack, so averaging recovers signal no single model has.
                    If so, structural consensus is a genuinely better readout and
                    the RQ1 comparison should arguably use it.

  VARIANCE          The four are noisy measurements of one underlying quantity, so
  REDUCTION         averaging cancels independent error without adding information.
                    The consensus then approaches what a single noiseless
                    architecture would give, and no more. RQ1's conclusion is
                    unaffected, because the ceiling is unchanged.

Three tests
-----------
  PRINCIPAL         Take the first principal component of the four z-scored scores.
  COMPONENT         Under variance reduction the consensus is essentially the shared
                    component, so PC1 alone should score about as well as the mean.
                    Under complementarity the mean should beat PC1, because the mean
                    retains the architecture-specific residuals PC1 discards.

  HEADROOM          Per allele, regress the consensus gain against the best single
                    architecture's AUROC on that allele. The project has already
                    established (ceiling-effect analysis, 3-4 August) that ensemble
                    benefit tracks how much room there was to improve rather than
                    how good the added model is. If the consensus gain shows the
                    same pattern, it is the same phenomenon.

  SUBSET            Report every subset, not the best one. `drop boltz` reaches
  SELECTION         0.923 against the four-way 0.905, but it is the best of eleven
                    subsets and quoting it alone would be selection. The honest
                    headline is the pre-specified four-way mean.

A caveat that applies throughout
---------------------------------
All scores here are per-allele z-scored, which is transductive: it uses the held-out
set's own mean and standard deviation over 12 binders and 12 decoys. Both sides of
every comparison carry it equally, so the paired differences are fair, but the
absolute AUROCs are upper bounds. This is the same caveat the project attaches to
every z-scored figure, and it applies to the consensus no less than to AF3.

Usage:
    python scripts/consensus_diagnosis.py \
        --sequence results/sequence_v4.csv \
        --structure af3=pae_af3_v4.csv af2=pae_af2_v4.csv \
                    esmfold2=pae_esmfold2_v4.csv boltz=pae_boltz_v4.csv \
        --feature pae_anchors_ic \
        --out results/consensus_diagnosis_v4.csv
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

KEY = ["allele", "peptide"]


def zwithin(v: pd.Series, by: pd.Series) -> pd.Series:
    return v.groupby(by).transform(lambda x: (x - x.mean()) / x.std())


def cluster_boot(y, a, b, alleles, n_boot=2000, seed=11):
    """Paired AUROC difference resampling ALLELES, since complexes nest in them."""
    rng = np.random.default_rng(seed)
    groups = np.array(sorted(pd.unique(alleles)))
    idx = {g: np.flatnonzero(alleles.values == g) for g in groups}
    out = []
    for _ in range(n_boot):
        pick = rng.choice(groups, size=len(groups), replace=True)
        sel = np.concatenate([idx[g] for g in pick])
        if len(np.unique(y[sel])) < 2:
            continue
        out.append(roc_auc_score(y[sel], a[sel]) - roc_auc_score(y[sel], b[sel]))
    v = np.array(out)
    return float(v.mean()), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequence", required=True)
    ap.add_argument("--structure", nargs="+", required=True, help="name=path.csv")
    ap.add_argument("--feature", default="pae_anchors_ic")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default="results/consensus_diagnosis.csv")
    args = ap.parse_args()

    seq = pd.read_csv(args.sequence)[KEY + ["label", "score"]]
    df = seq.rename(columns={"score": "sequence"})
    arch = []
    for spec in args.structure:
        name, path = spec.split("=", 1)
        d = pd.read_csv(path)
        if args.feature not in d.columns:
            print(f"  {path}: no {args.feature}, skipped")
            continue
        df = df.merge(d[KEY].assign(**{name: -d[args.feature]}), on=KEY, how="inner")
        arch.append(name)

    # everything z-scored within allele, sequence included, so like for like
    for c in arch + ["sequence"]:
        df[c] = zwithin(df[c], df.allele)
    df["consensus"] = df[arch].mean(axis=1)

    y = df.label.to_numpy()
    al = df.allele
    print(f"{len(df)} complexes, {df.allele.nunique()} alleles, "
          f"architectures {arch}\n")

    singles = {a: roc_auc_score(y, df[a]) for a in arch}
    best = max(singles, key=singles.get)
    auc_cons = roc_auc_score(y, df.consensus)
    auc_seq = roc_auc_score(y, df.sequence)
    print("=== the finding, reproduced ===")
    for a in sorted(singles, key=singles.get, reverse=True):
        print(f"  {a:<12} {singles[a]:.4f}")
    print(f"  {'consensus':<12} {auc_cons:.4f}")
    print(f"  {'sequence':<12} {auc_seq:.4f}  (z-scored, so like for like)")

    m, lo, hi = cluster_boot(y, df.consensus.values, df[best].values, al, args.n_boot)
    print(f"\n  consensus - {best}:  {m:+.4f} [{lo:+.4f}, {hi:+.4f}]"
          f"  {'excludes zero' if lo > 0 or hi < 0 else 'spans zero'}")
    m2, lo2, hi2 = cluster_boot(y, df.sequence.values, df.consensus.values, al,
                                args.n_boot)
    print(f"  sequence - consensus: {m2:+.4f} [{lo2:+.4f}, {hi2:+.4f}]"
          f"  {'excludes zero' if lo2 > 0 or hi2 < 0 else 'spans zero'}")

    # ---- test 1: is the consensus just the shared component? ----
    X = df[arch].to_numpy()
    Xc = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    pc1 = Xc @ Vt[0]
    if roc_auc_score(y, pc1) < 0.5:
        pc1 = -pc1
    var_explained = float(S[0] ** 2 / (S ** 2).sum())
    auc_pc1 = roc_auc_score(y, pc1)

    print("\n=== test 1: consensus vs the shared component ===")
    print(f"  PC1 explains {var_explained:.1%} of variance across the four")
    print(f"  PC1 alone        {auc_pc1:.4f}")
    print(f"  four-way mean    {auc_cons:.4f}   difference {auc_cons - auc_pc1:+.4f}")
    if abs(auc_cons - auc_pc1) < 0.01:
        print("  -> the mean is essentially the shared component. The gain is")
        print("     variance reduction: averaging cancels independent error in four")
        print("     noisy readings of one quantity, rather than combining distinct")
        print("     information. RQ1's ceiling is unchanged.")
    else:
        print("  -> the mean beats the shared component, so the architecture-specific")
        print("     residuals carry signal PC1 discards. That is complementarity,")
        print("     and the consensus is a genuinely better structural readout.")

    # ---- test 2: does the gain track headroom, as the ceiling effect predicts? ----
    rows = []
    for a, g in df.groupby("allele"):
        yy = g.label.to_numpy()
        if len(np.unique(yy)) < 2:
            continue
        b = max(roc_auc_score(yy, g[c]) for c in arch)
        rows.append({"allele": a, "best_single": b,
                     "consensus": roc_auc_score(yy, g.consensus),
                     "sequence": roc_auc_score(yy, g.sequence)})
    per = pd.DataFrame(rows)
    per["gain"] = per.consensus - per.best_single

    r_head = stats.spearmanr(per.best_single, per.gain)
    r_seq = stats.spearmanr(per.sequence, per.gain)
    print("\n=== test 2: does the gain track headroom? ===")
    print(f"  gain vs best single architecture AUROC   rho {r_head[0]:+.3f}  "
          f"p {r_head[1]:.3f}")
    print(f"  gain vs sequence AUROC on that allele    rho {r_seq[0]:+.3f}  "
          f"p {r_seq[1]:.3f}")
    print("  (the project's ceiling-effect analysis found ensemble benefit tracks")
    print("   headroom, not the quality of what is added; a strong negative here is")
    print("   the same phenomenon)")

    # ---- test 3: every subset, so the best one is not quoted as if pre-specified ----
    print("\n=== test 3: all subsets, not the best one ===")
    subs = []
    for k in range(2, len(arch) + 1):
        for c in combinations(arch, k):
            subs.append({"members": " + ".join(c), "k": k,
                         "auroc": roc_auc_score(y, df[list(c)].mean(axis=1))})
    sub = pd.DataFrame(subs).sort_values("auroc", ascending=False)
    print(sub.to_string(index=False))
    top = sub.iloc[0]
    print(f"\n  best subset is {top.members} at {top.auroc:.4f}, chosen from "
          f"{len(sub)} — that is selection, not a result.")
    print(f"  the pre-specified four-way mean is {auc_cons:.4f} and is the honest")
    print("  headline.")

    out = {"n": len(df), "best_single": best, "auroc_best_single": singles[best],
           "auroc_consensus": auc_cons, "auroc_sequence": auc_seq,
           "auroc_pc1": auc_pc1, "pc1_var_explained": var_explained,
           "cons_minus_best": m, "cons_minus_best_lo": lo, "cons_minus_best_hi": hi,
           "seq_minus_cons": m2, "seq_minus_cons_lo": lo2, "seq_minus_cons_hi": hi2,
           "rho_gain_vs_headroom": r_head[0], "p_gain_vs_headroom": r_head[1],
           "n_subsets": len(sub), "best_subset": top.members,
           "best_subset_auroc": top.auroc}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([out]).to_csv(args.out, index=False)
    per.to_csv(str(Path(args.out).with_name(
        Path(args.out).stem + "_per_allele.csv")), index=False)
    sub.to_csv(str(Path(args.out).with_name(
        Path(args.out).stem + "_subsets.csv")), index=False)
    print(f"\nWrote {args.out} plus per-allele and subset tables")


if __name__ == "__main__":
    main()