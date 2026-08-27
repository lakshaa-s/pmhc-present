"""Calibration and decision-relevant metrics — what AUROC cannot tell you.

Why this exists
---------------
Every metric reported in this project so far is rank-based. AUROC is invariant to
any monotone transform of the score, so nothing measured to date constrains whether
the model's output means anything *as a probability*, or how it behaves at the
operating point anyone would actually use.

Both matter here. The stated motivation is prioritising candidate peptides, and a
prioritisation pipeline takes the top k or applies a threshold — it does not care
about the middle of the ROC curve, and it does care whether "0.9" means ninety per
cent.

What is computed
----------------
  CALIBRATION      Brier score, expected calibration error, and a reliability table.
                   Then a single temperature parameter fitted **leave-one-allele-out**,
                   so the correction is never fitted on the allele it is evaluated on.
                   Temperature scaling is monotone, so AUROC is unchanged by
                   construction — that is the point. It changes what the number
                   means, not how well it ranks.

  DECISION METRICS Partial AUROC restricted to FPR <= 0.10 and <= 0.20, and precision
                   at the top k. These are the quantities a screening pipeline is
                   actually judged on, and they do not always preserve the AUROC
                   ordering.

Two caveats that must ship with any calibration figure
-------------------------------------------------------
The fold sets are balanced 50/50 by construction. Real presentation is closer to 1
in 1,000, so a temperature fitted here does not transfer to a deployment setting
without recalibrating against realistic prevalence. What it does establish is
whether the scores are *internally* coherent, which is a weaker but still absent
claim.

And the structural scores are per-allele z-scored, which is transductive — it uses
the held-out set's own mean and standard deviation. Calibrating a transductive score
compounds that, so structural calibration figures are reported for completeness but
should not be quoted as deployment-ready.

Usage:
    python scripts/calibration_metrics.py \
        --sequence results/sequence_v4.csv \
        --structure af3=pae_af3_v4.csv af2=pae_af2_v4.csv \
                    esmfold2=pae_esmfold2_v4.csv boltz=pae_boltz_v4.csv \
        --feature pae_anchors_ic \
        --out results/calibration_v4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import roc_auc_score, roc_curve

KEY = ["allele", "peptide"]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def brier(y, p):
    return float(np.mean((p - y) ** 2))


def ece(y, p, n_bins: int = 10):
    """Expected calibration error: mean |confidence - accuracy| over equal-width bins."""
    edges = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if m.sum():
            total += m.sum() / len(p) * abs(p[m].mean() - y[m].mean())
    return float(total)


def reliability(y, p, n_bins: int = 10) -> pd.DataFrame:
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        rows.append({"bin_lo": round(lo, 2), "bin_hi": round(hi, 2),
                     "n": int(m.sum()),
                     "mean_predicted": round(float(p[m].mean()), 4) if m.sum() else None,
                     "observed": round(float(y[m].mean()), 4) if m.sum() else None})
    return pd.DataFrame(rows)


def fit_temperature(logits, y) -> float:
    """Single scalar T minimising NLL of sigmoid(logit / T)."""
    def nll(t):
        if t <= 0:
            return 1e9
        p = np.clip(sigmoid(logits / t), 1e-7, 1 - 1e-7)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    r = minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded")
    return float(r.x)


def pauc(y, s, max_fpr: float) -> float:
    """Partial AUROC below a false-positive-rate ceiling, McClish-standardised.

    Standardising maps the partial area onto [0.5, 1] so it is comparable with the
    full AUROC; without it a pAUC at FPR<=0.1 is bounded above by 0.1 and cannot be
    read alongside the headline number.
    """
    fpr, tpr, _ = roc_curve(y, s)
    keep = fpr <= max_fpr
    if keep.sum() < 2:
        return float("nan")
    f, t = fpr[keep], tpr[keep]
    if f[-1] < max_fpr:                       # interpolate to the exact ceiling
        t_end = np.interp(max_fpr, fpr, tpr)
        f, t = np.append(f, max_fpr), np.append(t, t_end)
    area = np.trapezoid(t, f)
    minarea, maxarea = max_fpr ** 2 / 2, max_fpr
    return float(0.5 * (1 + (area - minarea) / (maxarea - minarea)))


def ppv_at_k(y, s, k: int) -> float:
    idx = np.argsort(-s)[:k]
    return float(y[idx].mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequence", required=True)
    ap.add_argument("--structure", nargs="*", default=[], help="name=path.csv")
    ap.add_argument("--feature", default="pae_anchors_ic")
    ap.add_argument("--out", default="results/calibration")
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
        # negate: PAE is lower-is-binding. z-score per allele, as everywhere else.
        v = -d[args.feature]
        z = v.groupby(d.allele).transform(lambda x: (x - x.mean()) / x.std())
        df = df.merge(d[KEY].assign(**{name: z.values}), on=KEY, how="inner")
        names.append(name)

    if len(names) > 1:
        df["consensus"] = df[names[1:]].mean(axis=1)
        names.append("consensus")

    y = df.label.to_numpy().astype(float)
    print(f"{len(df)} complexes, {int(y.sum())} binders, "
          f"prevalence {y.mean():.2f}\n")

    # ---------- calibration, sequence model only ----------
    # The structural scores are transductive z-scores; calibrating them compounds
    # that, so the headline calibration is the sequence model's.
    logit = df.sequence.to_numpy()
    p_raw = sigmoid(logit)

    print("=== calibration, sequence model ===")
    print(f"  mean predicted P   {p_raw.mean():.3f}   (true prevalence {y.mean():.3f})")
    print(f"  Brier              {brier(y, p_raw):.4f}")
    print(f"  ECE                {ece(y, p_raw):.4f}")
    print(f"  AUROC              {roc_auc_score(y, p_raw):.4f}")

    # leave-one-allele-out temperature, so T is never fitted on the allele it scales
    p_cal = np.empty_like(p_raw)
    temps = {}
    for a in df.allele.unique():
        te = (df.allele == a).to_numpy()
        t = fit_temperature(logit[~te], y[~te])
        temps[a] = round(t, 3)
        p_cal[te] = sigmoid(logit[te] / t)

    print(f"\n  after leave-one-allele-out temperature scaling "
          f"(T {min(temps.values()):.2f}-{max(temps.values()):.2f}):")
    print(f"  mean predicted P   {p_cal.mean():.3f}")
    print(f"  Brier              {brier(y, p_raw):.4f} -> {brier(y, p_cal):.4f}")
    print(f"  ECE                {ece(y, p_raw):.4f} -> {ece(y, p_cal):.4f}")
    print(f"  AUROC              {roc_auc_score(y, p_raw):.4f} -> "
          f"{roc_auc_score(y, p_cal):.4f}")
    print("  (temperature scaling is monotone within an allele, but leave-one-out")
    print("   gives each allele its own T, so the pooled AUROC can shift slightly.")
    print("   A large shift would mean the per-allele temperatures differ enough to")
    print("   be reordering complexes across alleles — worth checking if it does.)")

    rel = reliability(y, p_raw)
    rel_cal = reliability(y, p_cal)
    rel["calibrated_mean_predicted"] = rel_cal.mean_predicted
    rel["calibrated_observed"] = rel_cal.observed
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    rel.to_csv(f"{args.out}_reliability.csv", index=False)

    empty = rel[(rel.n > 0) & (rel.observed == 0)]
    if len(empty):
        print(f"\n  {len(empty)} occupied bins contain no binders at all "
              f"(up to predicted {empty.mean_predicted.max():.2f}) — the model is "
              f"never confidently wrong at the low end; the miscalibration is "
              f"overconfidence at the top.")

    # ---------- decision-relevant metrics, all models ----------
    print("\n=== decision-relevant metrics ===")
    n_pos = int(y.sum())
    print(f"{'model':<12} {'AUROC':>7} {'pAUC.10':>8} {'pAUC.20':>8} "
          f"{'PPV@20':>7} {'PPV@n+':>7}")
    rows = []
    for n in names:
        s = df[n].to_numpy()
        r = {"model": n, "auroc": round(roc_auc_score(y, s), 4),
             "pauc_fpr10": round(pauc(y, s, 0.10), 4),
             "pauc_fpr20": round(pauc(y, s, 0.20), 4),
             "ppv_top20": round(ppv_at_k(y, s, 20), 4),
             "ppv_top_npos": round(ppv_at_k(y, s, n_pos), 4)}
        rows.append(r)
        print(f"{n:<12} {r['auroc']:>7.3f} {r['pauc_fpr10']:>8.3f} "
              f"{r['pauc_fpr20']:>8.3f} {r['ppv_top20']:>7.2f} "
              f"{r['ppv_top_npos']:>7.3f}")

    dm = pd.DataFrame(rows)
    dm.to_csv(f"{args.out}_decision_metrics.csv", index=False)

    # does the AUROC ordering survive the other metrics?
    print("\n=== does the AUROC ordering hold? ===")
    by_auroc = list(dm.sort_values("auroc", ascending=False).model)
    for m in ("pauc_fpr10", "ppv_top20"):
        by_m = list(dm.sort_values(m, ascending=False).model)
        same = by_auroc == by_m
        print(f"  {m:<12} {'same order' if same else 'REORDERS'}: "
              f"{' > '.join(by_m)}")
    print("\n  Where the ordering changes, no single model is 'best' without naming")
    print("  the operating point — worth a sentence wherever the RQ1 table is read")
    print("  as a ranking.")

    print(f"\nWrote {args.out}_reliability.csv and {args.out}_decision_metrics.csv")


if __name__ == "__main__":
    main()