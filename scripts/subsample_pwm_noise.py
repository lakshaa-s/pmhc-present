"""Is the motif-distance result an artefact of PWM noise at low sample size?

`motif_distinctiveness.py` found that alleles with distant motif neighbours perform
worse (rho -0.363 across 123 alleles). The threat to that: a PWM built from 200
peptides is noisier than one built from 5,000, and noise pushes a profile away from
every other profile. Data-poor alleles would then look isolated for purely statistical
reasons -- and they also have noisier AUROCs, so the correlation could be manufactured.

Controlling for log(n) strengthened the effect rather than weakening it, but log(n) is
a crude proxy for estimation error. This is the direct test: take alleles with plenty
of data, subsample them to the sizes the sparse alleles actually have, and measure how
much their nearest-neighbour distance inflates.

Interpretation:
  - nn_dist stable under subsampling  -> the observed isolation is real, and the
    correlation stands as reported;
  - nn_dist inflates by an amount comparable to the observed spread between alleles
    -> part of the correlation is measurement artefact and needs a noise correction.

The comparison that matters is the inflation at n ~= 200 against the gap between the
most and least isolated real alleles (~0.005 to ~0.113 in the observed data).

Usage:
    python scripts/subsample_pwm_noise.py \
        --atlas data/processed/atlas_labelled.csv \
        --distinctiveness data/processed/motif_distinctiveness.csv \
        --sizes 100 200 400 800 1600 \
        --trials 10
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

AA = "ACDEFGHIKLMNPQRSTVWY"
LENGTH = 9
PSEUDOCOUNT = 0.5


def pwm(peptides: list[str]) -> np.ndarray:
    m = np.full((LENGTH, 20), PSEUDOCOUNT)
    idx = {a: i for i, a in enumerate(AA)}
    for p in peptides:
        for i, ch in enumerate(p):
            j = idx.get(ch)
            if j is not None:
                m[i, j] += 1
    return m / m.sum(axis=1, keepdims=True)


def jsd(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def profile_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean([jsd(a[i], b[i]) for i in range(LENGTH)]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlas", default="data/processed/atlas_labelled.csv")
    ap.add_argument("--distinctiveness",
                    default="data/processed/motif_distinctiveness.csv")
    ap.add_argument("--sizes", type=int, nargs="+", default=[100, 200, 400, 800, 1600])
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--n-test", type=int, default=8,
                    help="how many data-rich alleles to subsample")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/processed/subsample_pwm_noise.csv")
    args = ap.parse_args()

    random.seed(args.seed)
    df = pd.read_csv(args.atlas)
    df = df[(df.label == 1) & (df.length == LENGTH)]

    peps_by_allele, mats = {}, {}
    for allele, grp in df.groupby("allele"):
        peps = [p for p in grp.peptide if len(p) == LENGTH]
        if len(peps) >= 50:
            peps_by_allele[allele] = peps
            mats[allele] = pwm(peps)

    obs = pd.read_csv(args.distinctiveness)
    print(f"observed nn_dist across {len(obs)} alleles: "
          f"min {obs.nn_dist.min():.4f}, median {obs.nn_dist.median():.4f}, "
          f"max {obs.nn_dist.max():.4f}")
    print(f"  (the spread these results have to beat: "
          f"{obs.nn_dist.max() - obs.nn_dist.min():.4f})\n")

    # data-rich alleles to subsample, spread across loci
    rich = sorted(peps_by_allele, key=lambda a: -len(peps_by_allele[a]))
    chosen, seen = [], set()
    for a in rich:
        locus = a.split("*")[0]
        if len(peps_by_allele[a]) >= max(args.sizes) * 2:
            if seen.count(locus) if isinstance(seen, list) else True:
                chosen.append(a)
        if len(chosen) >= args.n_test:
            break

    obs_map = dict(zip(obs.allele, obs.nn_dist))
    rows = []

    print(f"{'allele':<14} {'full n':>7} {'full nn':>9} " +
          " ".join(f"{f'n={s}':>16}" for s in args.sizes))

    for allele in chosen:
        peps = peps_by_allele[allele]
        others = {b: m for b, m in mats.items() if b != allele}
        full_nn = min(profile_distance(mats[allele], m) for m in others.values())
        line = f"{allele:<14} {len(peps):>7} {full_nn:>9.4f} "
        for size in args.sizes:
            if size > len(peps):
                line += f"{'-':>17}"
                continue
            vals = []
            for _ in range(args.trials):
                sub = pwm(random.sample(peps, size))
                vals.append(min(profile_distance(sub, m) for m in others.values()))
            mean, sd = float(np.mean(vals)), float(np.std(vals))
            line += f"{mean:>10.4f}±{sd:.3f} "
            rows.append({"allele": allele, "n_full": len(peps), "nn_full": full_nn,
                         "n_sub": size, "nn_sub_mean": mean, "nn_sub_sd": sd,
                         "inflation": mean - full_nn})
        print(line)

    res = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.out, index=False)

    print("\n=== inflation (subsampled nn_dist minus full-data nn_dist) ===")
    print(f"{'n':>6} {'median':>9} {'mean':>9} {'max':>9}")
    for size, g in res.groupby("n_sub"):
        print(f"{size:>6} {g.inflation.median():>9.4f} {g.inflation.mean():>9.4f} "
              f"{g.inflation.max():>9.4f}")

    sparse = obs.nsmallest(10, "n_peptides")
    print(f"\nThe sparse alleles this is meant to mimic (n and observed nn_dist):")
    print(sparse[["allele", "n_peptides", "nn_dist", "auroc"]].to_string(index=False))

    small = [s for s in args.sizes if s <= 400]
    if small:
        infl = res[res.n_sub.isin(small)].inflation
        spread = obs.nn_dist.max() - obs.nn_dist.min()
        print(f"\nAt n <= 400, median inflation is {infl.median():.4f} "
              f"({infl.median()/spread:.0%} of the observed between-allele spread).")
        if infl.median() > 0.2 * spread:
            print("  -> LARGE relative to the spread. The correlation needs a noise "
                  "correction before it can be reported as biological.")
        else:
            print("  -> small relative to the spread. Sampling noise is unlikely to "
                  "explain the observed relationship.")

    # does inflation track n closely enough to mimic the real correlation?
    if len(res) > 10:
        rho, pv = stats.spearmanr(res.n_sub, res.inflation)
        print(f"\ninflation vs subsample size: rho={rho:.3f} p={pv:.2e} "
              f"(strong negative = noise does scale the way the confound requires)")

    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()