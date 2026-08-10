"""Do sequence and structure fail on different complexes, or the same ones?

The claim being tested
-----------------------
A 2024 pMHC-II study (biorxiv 2024.10.06.616783) reports that structural methods are
better at identifying binders while sequence models are better at filtering
non-binders, and argues that this asymmetry motivates hybrid consensus approaches.

If that holds for class I, it would supply the mechanism RQ2 lacks. Nine combination
strategies found no synergy, but every one of them asked whether combining raises
AUROC. None asked whether the two model types fail on *different complexes*, which is
the precondition for combining to help at all. Two models that are individually
mediocre but wrong about different things can be combined usefully; two that are
wrong about the same things cannot, however sophisticated the combiner.

What is computed
----------------
AUROC pools both directions into one number, so it hides the asymmetry. Three
decompositions:

  SENSITIVITY /    At a fixed operating point, the fraction of binders correctly
  SPECIFICITY      called and the fraction of decoys correctly rejected. This needs
                   a threshold: a rank-based decomposition cannot answer the
                   question, because the mean binder margin and the mean decoy
                   margin count the same pairwise comparisons grouped differently
                   and are identical by construction whenever the classes are
                   balanced. Verified empirically before this was written.

                   Scores are standardised within allele first, so one operating
                   point is meaningful across alleles with different scales — the
                   same correction that lifted pooled PAE figures by up to 0.15.
                   The threshold is set per model to the value that maximises
                   Youden's J, so each is judged at its own best operating point
                   rather than at an arbitrary shared one.

  ERROR OVERLAP    Which complexes each model gets wrong, and whether those sets
                   coincide. This is the decisive quantity. Reported as the
                   correlation of per-complex margins and as the Jaccard overlap of
                   the worst-ranked complexes.

The margin here is the same per-complex quantity used in fold_quality_control.py: the
fraction of opposite-class complexes within the same allele that this complex is
correctly ordered against. It averages to the allele's AUROC, and it is computed
within allele so that between-allele scale differences do not contaminate it — the
same correction that lifted pooled PAE figures by up to 0.15.

Interpretation, stated in advance so it is not fitted after the fact
--------------------------------------------------------------------
  High error overlap    The models fail on the same complexes. RQ2's null has a
                        mechanism: there is no complementary information to combine,
                        which is redundancy rather than weakness.

  Low error overlap     They fail on different complexes, so combining *should*
                        help and does not. That makes the null a sharper puzzle and
                        points at the combiner or at the sample size rather than at
                        the features.

  Asymmetry             If structure ranks binders well but fails to reject decoys
                        while sequence does the reverse, the pMHC-II claim carries
                        over to class I and a consensus method targeted at that
                        asymmetry would be worth building.

Usage:
    python scripts/rq2_error_overlap.py \
        --sequence results/sequence_v4.csv \
        --structure af3=pae_af3_v4.csv af2=pae_af2_v4.csv \
                    esmfold2=pae_esmfold2_v4.csv boltz=pae_boltz_v4.csv \
        --feature pae_anchors_ic \
        --out results/rq2_error_overlap.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve

KEY = ["allele", "peptide"]


def margins(df: pd.DataFrame, score: str) -> pd.Series:
    """Per-complex margin: fraction of opposite-class complexes correctly ordered.

    Computed within allele, so between-allele scale differences cannot contaminate
    it. Averages to the allele's AUROC by construction.
    """
    out = pd.Series(np.nan, index=df.index)
    for _, g in df.groupby("allele"):
        b = g[g.label == 1][score].to_numpy()
        d = g[g.label == 0][score].to_numpy()
        if len(b) == 0 or len(d) == 0:
            continue
        for i in g.index[g.label == 1]:
            v = df.at[i, score]
            out[i] = ((v > d).sum() + 0.5 * (v == d).sum()) / len(d)
        for i in g.index[g.label == 0]:
            v = df.at[i, score]
            out[i] = ((v < b).sum() + 0.5 * (v == b).sum()) / len(b)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequence", required=True)
    ap.add_argument("--structure", nargs="+", required=True,
                    help="name=path.csv")
    ap.add_argument("--feature", default="pae_anchors_ic")
    ap.add_argument("--worst-frac", type=float, default=0.25,
                    help="fraction of complexes counted as 'wrong' for the overlap")
    ap.add_argument("--out", default="results/rq2_error_overlap.csv")
    args = ap.parse_args()

    seq = pd.read_csv(args.sequence)[KEY + ["label", "score"]]
    df = seq.rename(columns={"score": "sequence"})
    names = ["sequence"]

    for spec in args.structure:
        name, path = spec.split("=", 1)
        d = pd.read_csv(path)
        if args.feature not in d.columns:
            print(f"  {path}: no {args.feature}, skipped")
            continue
        # PAE is lower-is-binding, so negate to make every score higher-is-binding
        d = d[KEY].assign(**{name: -d[args.feature]})
        df = df.merge(d, on=KEY, how="inner")
        names.append(name)

    print(f"{len(df)} complexes, {int(df.label.sum())} binders / "
          f"{int((1 - df.label).sum())} decoys")
    print(f"models: {names}\n")

    M = pd.DataFrame({n: margins(df, n) for n in names})
    M["label"] = df.label.values
    M["allele"] = df.allele.values

    print("=== is the failure on binders or on decoys? ===")
    print("(at each model's own best operating point, scores standardised within")
    print(" allele so one threshold is meaningful across them)\n")
    print(f"{'model':<12} {'AUROC':>7} {'sens':>7} {'spec':>7} {'gap':>8}")
    rows = []
    y = df.label.to_numpy()
    for n in names:
        z = df.groupby("allele")[n].transform(lambda x: (x - x.mean()) / x.std())
        auc = roc_auc_score(y, z)
        fpr, tpr, thr = roc_curve(y, z)
        j = int(np.argmax(tpr - fpr))          # Youden's J
        sens, spec = float(tpr[j]), float(1 - fpr[j])
        rows.append({"model": n, "auroc": auc, "sensitivity": sens,
                     "specificity": spec, "asymmetry": sens - spec,
                     "threshold": float(thr[j])})
        print(f"{n:<12} {auc:>7.3f} {sens:>7.3f} {spec:>7.3f} {sens - spec:>+8.3f}")
    print("\n  sens = binders correctly called; spec = decoys correctly rejected")
    print("  a positive gap means the model is better at recognising binders than")
    print("  at rejecting decoys; negative is the reverse")

    asym = pd.DataFrame(rows)
    seq_gap = float(asym.loc[asym.model == "sequence", "asymmetry"].iloc[0])
    st_gaps = asym[asym.model != "sequence"].asymmetry
    if len(st_gaps) and ((st_gaps > seq_gap + 0.05).all()
                         or (st_gaps < seq_gap - 0.05).all()):
        direction = "better at binders" if st_gaps.mean() > seq_gap \
            else "better at decoys"
        print(f"\n  -> structure is consistently {direction} relative to sequence,")
        print("     which is the asymmetry the pMHC-II study reports. A consensus")
        print("     method targeted at it would be worth building.")
    else:
        print("\n  -> no consistent asymmetry between sequence and structure, so the")
        print("     pMHC-II claim does not carry over to class I here.")

    print("\n=== do they fail on the same complexes? ===")
    k = max(1, int(len(M) * args.worst_frac))
    worst = {n: set(M[n].nsmallest(k).index) for n in names}
    over = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            r = stats.spearmanr(M[a], M[b])[0]
            j = len(worst[a] & worst[b]) / len(worst[a] | worst[b])
            over.append({"model_a": a, "model_b": b, "margin_rho": round(r, 3),
                         "worst_jaccard": round(j, 3)})
            print(f"  {a:<10} vs {b:<10} margin rho {r:+.3f}   "
                  f"worst-{int(args.worst_frac*100)}% overlap {j:.3f}")

    ov = pd.DataFrame(over)
    ss = ov[(ov.model_a == "sequence") | (ov.model_b == "sequence")]
    st = ov[(ov.model_a != "sequence") & (ov.model_b != "sequence")]

    # a random-overlap baseline, so the Jaccard has something to be judged against
    exp_j = args.worst_frac / (2 - args.worst_frac)
    print(f"\n  Jaccard expected by chance at this threshold: {exp_j:.3f}")
    if not ss.empty:
        print(f"  sequence vs structure: median rho "
              f"{ss.margin_rho.median():+.3f}, median overlap "
              f"{ss.worst_jaccard.median():.3f}")
    if not st.empty:
        print(f"  structure vs structure: median rho "
              f"{st.margin_rho.median():+.3f}, median overlap "
              f"{st.worst_jaccard.median():.3f}")

    if not ss.empty:
        m = ss.margin_rho.median()
        if m > 0.5:
            print("\n  -> sequence and structure fail on largely the same complexes.")
            print("     RQ2's null has a mechanism: there is little complementary")
            print("     information to combine. Redundancy, not weakness.")
        elif m > 0.2:
            print("\n  -> partial overlap. Some complementary signal exists but not")
            print("     enough for the combiners tested to exploit at this n.")
        else:
            print("\n  -> they fail on different complexes, so combining should help")
            print("     and does not. That points at the combiner or the sample")
            print("     size rather than at the features, and makes RQ2's null a")
            print("     sharper puzzle than a redundancy account would.")

    if not st.empty and not ss.empty:
        if st.margin_rho.median() > ss.margin_rho.median() + 0.15:
            print("\n  Note the structural models agree with each other more than")
            print("  any of them agrees with sequence, so the two families are")
            print("  making distinct kinds of error even where combining fails.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.concat([asym.assign(kind="asymmetry"),
               ov.assign(kind="overlap")]).to_csv(args.out, index=False)
    M.to_csv(str(Path(args.out).with_name(
        Path(args.out).stem + "_margins.csv")), index=False)
    print(f"\nWrote {args.out} and the per-complex margins alongside it")


if __name__ == "__main__":
    main()