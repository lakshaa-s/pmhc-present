"""Select alleles for RQ3 as representatives of distinct binding-motif classes.

The brief, from the 6 August meeting
------------------------------------
Benny: which alleles matters less than making sure they span the space. Include
A*02:01, then alleles whose motifs are far from it — B*08:01 was suggested — and if
possible "pick a representative from each class of binder".

That is a clustering problem rather than a ranking one, and it differs from how
panel v4 was chosen. Panel v4 took the most motif-isolated alleles, which selects
outliers. Here we want coverage: partition the alleles into motif classes and take
one well-supported representative from each, so the saturation-mutagenesis
landscapes span the range of binding chemistry rather than sampling its extremes.

Method
------
Average-linkage agglomerative clustering on the pairwise Jensen-Shannon distance
between per-allele position weight matrices, cutting the tree at k clusters. Within
each cluster the representative is the allele with the most held-out 9mers, since
RQ3 needs a well-determined motif to mutate away from — a noisy PWM would make the
landscape comparison meaningless.

A*02:01 is forced in and its cluster is then excluded, so the remaining slots go to
genuinely different chemistry.

Why data sufficiency matters more here than for panel v4
--------------------------------------------------------
RQ3 mutates a canonical binder at every position. If the starting peptide is not
robustly canonical the landscape is measuring noise, so the candidate floor is
higher than the 120 used for the fold sets.

Usage:
    python scripts/select_rq3_alleles.py \
        --atlas data/processed/atlas_labelled.csv \
        --split data/processed/split_val.csv \
        --k 6 --min-heldout 250 \
        --out fold_sets/rq3_alleles.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

AA = "ACDEFGHIKLMNPQRSTVWY"


def pwm(peptides, length: int, pseudo: float = 0.5) -> np.ndarray:
    idx = {a: i for i, a in enumerate(AA)}
    m = np.full((length, 20), pseudo)
    for p in peptides:
        for i, c in enumerate(p):
            if c in idx:
                m[i, idx[c]] += 1
    return m / m.sum(axis=1, keepdims=True)


def js_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Mean Jensen-Shannon distance across positions."""
    m = 0.5 * (p + q)
    def kl(a, b):
        return np.sum(a * np.log2(np.clip(a, 1e-12, None) /
                                 np.clip(b, 1e-12, None)), axis=1)
    js = 0.5 * kl(p, m) + 0.5 * kl(q, m)
    return float(np.sqrt(np.clip(js, 0, None)).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlas", default="data/processed/atlas_labelled.csv")
    ap.add_argument("--split", default="data/processed/split_val.csv")
    ap.add_argument("--length", type=int, default=9)
    ap.add_argument("--k", type=int, default=6, help="number of motif classes")
    ap.add_argument("--min-peptides", type=int, default=200,
                    help="minimum total 9mers for a usable PWM")
    ap.add_argument("--min-heldout", type=int, default=250,
                    help="minimum held-out 9mers, so binders are truly canonical")
    ap.add_argument("--force", nargs="*", default=["HLA-A*02:01"])
    ap.add_argument("--out", default="fold_sets/rq3_alleles.txt")
    args = ap.parse_args()

    atlas = pd.read_csv(args.atlas)
    atlas = atlas[(atlas.label == 1) & (atlas.length == args.length)]
    val = set(map(tuple, pd.read_csv(args.split).values))
    atlas["heldout"] = [(a, p) in val for a, p in zip(atlas.allele, atlas.peptide)]

    counts = atlas.groupby("allele").size()
    held = atlas[atlas.heldout].groupby("allele").size()
    keep = counts[counts >= args.min_peptides].index
    print(f"{len(counts)} alleles, {len(keep)} with >= {args.min_peptides} "
          f"{args.length}mers")

    mats = {a: pwm(atlas[atlas.allele == a].peptide, args.length) for a in keep}
    alleles = sorted(mats)
    n = len(alleles)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = js_distance(mats[alleles[i]], mats[alleles[j]])

    Z = linkage(squareform(D, checks=False), method="average")
    labels = fcluster(Z, t=args.k, criterion="maxclust")
    clusters = {}
    for a, c in zip(alleles, labels):
        clusters.setdefault(c, []).append(a)

    print(f"\n{args.k} motif classes:")
    for c in sorted(clusters):
        members = clusters[c]
        loci = pd.Series([m.split("*")[0] for m in members]).value_counts().to_dict()
        print(f"  class {c}: {len(members):>3} alleles  {loci}")

    idx = {a: i for i, a in enumerate(alleles)}
    chosen, used = [], set()

    for a in args.force:
        if a not in idx:
            print(f"\nWARNING: forced allele {a} has too few peptides, skipping")
            continue
        c = labels[idx[a]]
        chosen.append((a, c, "forced"))
        used.add(c)

    # remaining classes, largest first, representative = most held-out peptides
    order = sorted((c for c in clusters if c not in used),
                   key=lambda c: -len(clusters[c]))
    for c in order:
        if len(chosen) >= args.k:
            break
        cands = [(held.get(a, 0), a) for a in clusters[c]
                 if held.get(a, 0) >= args.min_heldout]
        if not cands:
            print(f"  class {c}: no allele with >= {args.min_heldout} held-out, "
                  f"skipped")
            continue
        cands.sort(reverse=True)
        chosen.append((cands[0][1], c, f"{cands[0][0]} held-out"))
        used.add(c)

    a0 = args.force[0] if args.force and args.force[0] in idx else chosen[0][0]
    print(f"\n=== selected {len(chosen)} alleles ===")
    print(f"{'allele':<14} {'class':>6} {'held-out':>9} {'JS from '+a0:>14}  note")
    rows = []
    for a, c, note in chosen:
        d = D[idx[a], idx[a0]]
        print(f"{a:<14} {c:>6} {held.get(a, 0):>9} {d:>14.3f}  {note}")
        rows.append({"allele": a, "motif_class": c,
                     "heldout": int(held.get(a, 0)),
                     f"js_from_{a0}": round(d, 4)})

    if len(chosen) > 1:
        pair = [D[idx[x[0]], idx[y[0]]] for i, x in enumerate(chosen)
                for y in chosen[i + 1:]]
        print(f"\npairwise JS among selected: {min(pair):.3f}-{max(pair):.3f} "
              f"(all alleles: {D[D > 0].min():.3f}-{D.max():.3f})")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(a for a, _, _ in chosen) + "\n")
    pd.DataFrame(rows).to_csv(str(Path(args.out).with_suffix(".csv")), index=False)
    print(f"\nWrote {args.out}")

    n_var = len(chosen) * args.length * 19
    print(f"\n{len(chosen)} alleles x {args.length} positions x 19 substitutions "
          f"= {n_var} variants")
    print(f"  sequence model: instant")
    print(f"  ESMFold2 at ~20 s: {n_var * 20 / 3600:.1f} h")


if __name__ == "__main__":
    main()