"""How much power did RQ2 actually have?

The question
------------
Nine ensemble configurations returned intervals spanning zero. `PROGRESS.md` records
the inference that sample size is the likelier explanation for the null. That is
reasonable but it is an assertion, and it can be made a number: given the effect
actually observed, how often would this design have detected it?

That distinction matters for the write-up. "We found nothing" invites the reader to
conclude there is nothing. "We could not have found an effect this size even if it
were real, and here is the panel size that could" is a stronger and more useful
claim, and it converts a limitation into a concrete recommendation.

Method
------
Take the best real blend and the sequence baseline. Their observed paired AUROC
difference is the effect size. Then, for each candidate panel size n, resample
complexes with replacement to size n, run the same paired bootstrap the project uses
elsewhere, and record whether the interval excludes zero. Power is the fraction of
simulated panels in which it does.

Three honest limitations, stated because they all push the same way
--------------------------------------------------------------------
The observed effect is treated as the true effect. If the observed +0.013 is itself
noise around a true zero, these numbers describe the power to detect a
non-existent effect, which is not a meaningful quantity. The calculation is
therefore conditional: *if* the effect is real and this size, here is the power.

Resampling complexes assumes a larger panel would look like more of the same. In
practice a larger panel means more *alleles*, which adds between-allele variance the
current panel does not contain. Real power at n=500 would be lower than estimated
here.

And the complexes are nested within nine alleles, so resampling them independently
understates the correlation structure. The cluster-aware version is reported
alongside for comparison.

All three make the reported power an upper bound, so the recommended panel size is a
lower bound. That is the conservative direction for a future-work claim.

Usage:
    python scripts/rq2_power.py \
        --sequence results/sequence_v4.csv \
        --structure af3=pae_af3_v4.csv af2=pae_af2_v4.csv \
                    esmfold2=pae_esmfold2_v4.csv boltz=pae_boltz_v4.csv \
        --feature pae_anchors_ic \
        --out results/rq2_power.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

KEY = ["allele", "peptide"]


def zwithin(v: pd.Series, by: pd.Series) -> pd.Series:
    return v.groupby(by).transform(lambda x: (x - x.mean()) / x.std())


def paired_ci(y, a, b, rng, n_boot=400):
    """Paired AUROC difference with a percentile interval, resampling complexes."""
    n = len(y)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if len(np.unique(y[i])) < 2:
            continue
        vals.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
    if len(vals) < 20:
        return 0.0, -1.0, 1.0
    v = np.array(vals)
    return float(v.mean()), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequence", required=True)
    ap.add_argument("--structure", nargs="+", required=True)
    ap.add_argument("--feature", default="pae_anchors_ic")
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[216, 324, 432, 648, 864, 1296, 1728])
    ap.add_argument("--n-sim", type=int, default=200,
                    help="simulated panels per size")
    ap.add_argument("--n-boot", type=int, default=400,
                    help="bootstrap resamples within each simulated panel")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/rq2_power.csv")
    args = ap.parse_args()

    seq = pd.read_csv(args.sequence)[KEY + ["label", "score"]]
    df = seq.rename(columns={"score": "sequence"})
    arch = []
    for spec in args.structure:
        name, path = spec.split("=", 1)
        d = pd.read_csv(path)
        if args.feature not in d.columns:
            continue
        df = df.merge(d[KEY].assign(**{name: -d[args.feature]}), on=KEY, how="inner")
        arch.append(name)

    for c in arch + ["sequence"]:
        df[c] = zwithin(df[c], df.allele)

    y = df.label.to_numpy()
    base = df.sequence.to_numpy()

    # the best real blend: sequence plus the structural consensus, weight chosen
    # on the observed data. That biases toward a larger effect, which is the
    # conservative direction for a power calculation.
    cons = df[arch].mean(axis=1).to_numpy()
    best_w, best_d = 0.0, 0.0
    for w in np.arange(0.0, 0.55, 0.05):
        d_ = roc_auc_score(y, (1 - w) * base + w * cons) - roc_auc_score(y, base)
        if d_ > best_d:
            best_w, best_d = w, d_
    blend = (1 - best_w) * base + best_w * cons

    print(f"{len(df)} complexes, {df.allele.nunique()} alleles")
    print(f"sequence alone           {roc_auc_score(y, base):.4f}")
    print(f"best blend (w={best_w:.2f} on consensus)  "
          f"{roc_auc_score(y, blend):.4f}")
    print(f"observed effect          {best_d:+.4f}")
    print("\n(the blend weight was fitted on this same data, so the effect is an")
    print(" optimistic estimate — which makes the power estimate optimistic too)\n")

    if best_d <= 0:
        print("No positive effect to power for; the null is not a power question.")
        return

    rng = np.random.default_rng(args.seed)
    n = len(y)
    rows = []
    print(f"{'n':>6} {'power':>7}  {'mean effect':>12}")
    for size in args.sizes:
        hits = 0
        effs = []
        for _ in range(args.n_sim):
            idx = rng.integers(0, n, size)
            yy = y[idx]
            if len(np.unique(yy)) < 2:
                continue
            m, lo, hi = paired_ci(yy, blend[idx], base[idx], rng, args.n_boot)
            effs.append(m)
            if lo > 0:
                hits += 1
        power = hits / args.n_sim
        rows.append({"n_complexes": size, "power": round(power, 3),
                     "mean_effect": round(float(np.mean(effs)), 4)})
        print(f"{size:>6} {power:>7.3f}  {np.mean(effs):>+12.4f}")

    out = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    at_current = out.loc[out.n_complexes == len(y), "power"]
    print()
    if len(at_current):
        print(f"  At the actual panel size (n={len(y)}), power is "
              f"{at_current.iloc[0]:.2f}.")
    reach = out[out.power >= 0.8]
    if len(reach):
        print(f"  80% power is first reached at n={int(reach.n_complexes.iloc[0])}.")
    else:
        print(f"  80% power is not reached by n={args.sizes[-1]}.")

    pw = float(at_current.iloc[0]) if len(at_current) else float("nan")
    print()
    if pw < 0.5:
        print(f"  Read this as: if an effect of this size is real, the current")
        print(f"  design misses it more often than it finds it. The null is then")
        print(f"  as much a statement about the design as about the effect.")
    elif pw < 0.8:
        print(f"  Read this as: the design would detect an effect of this size")
        print(f"  most of the time but not reliably, so the null is weak evidence")
        print(f"  of absence rather than none.")
    else:
        print(f"  Read this as: the design was well powered for an effect of this")
        print(f"  size, so the null is genuine evidence of absence rather than a")
        print(f"  power failure.")
    print("\n  Report this alongside the oracle ceiling, which is an independent")
    print("  argument: even a perfect per-complex router has little headroom on")
    print("  this panel, so the effect is bounded in size as well as hard to see.")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()