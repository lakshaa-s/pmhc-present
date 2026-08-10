"""Build the saturation-mutagenesis variant set for RQ3's structural half.

The question
------------
The sequence half is done: across seven alleles the model's landscape agrees with
the Motif Atlas PWM strongly on *which positions matter* (median rho 0.817) and only
moderately on *which residues go there* (median rho 0.541), and its top-two positions
are a subset of the IC-derived anchors for six of seven alleles.

This builds the variants needed to ask the same of the folding models, so the two
landscapes can be compared directly. That comparison is what RQ2 lacks: eight
configurations found no synergy, and if the landscapes agree the redundancy has a
mechanism rather than just an observation behind it.

Scope, and why it is seeds before alleles
------------------------------------------
A landscape built from one starting peptide confounds the allele's chemistry with
that peptide's idiosyncrasies, and there is no way to tell whether that matters
without running more than one seed. Four alleles at three seeds answers both "do the
landscapes agree" and "are single-seed landscapes stable enough to trust"; seven
alleles at one seed answers only the first and leaves any disagreement
uninterpretable.

    4 alleles x 3 seeds x 9 positions x 19 substitutions = 2,052 folds
    ESMFold2 at ~20 s ≈ 11 h, an overnight run

ESMFold2 rather than AF3 (72 s/fold, ~41 h) or AF2 via HISTOFold. ESMFold2 is the
weakest of the four architectures on RQ1, which is a limitation to state — but it is
the only one fast enough at this scale, and the question here is *where* the model
responds rather than how well it discriminates.

A specific prediction to test
------------------------------
HLA-B*08:01 has IC-derived anchors at P2, P5, P8 and P9, and the sequence model
picked only P2 and P9 — it does not capture the secondary anchors that Chris
Thorpe's P5 MSA rebalancing targets. If the structural landscape does peak at P5
there, the two model types have found different things and RQ2's redundancy account
needs qualifying. That is why B*08:01 is in the default panel.

Wild-type folds are included once per seed so each variant's score can be expressed
as a change rather than an absolute, which removes per-peptide offsets the same way
per-allele z-scoring removes per-allele ones.

Usage:
    python scripts/build_rq3_variants.py \
        --alleles 'HLA-A*02:01' 'HLA-B*08:01' 'HLA-C*03:04' 'HLA-B*57:01' \
        --fold-sets fold_sets/fold_set_v2.csv fold_sets/fold_set_v4.csv \
                    fold_sets/binders_rq3.csv \
        --n-seeds 3 --out fold_sets/rq3_variants.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

AA = "ACDEFGHIKLMNPQRSTVWY"


def slug_to_allele(slug: str) -> str:
    b = slug.split("_")
    return f"HLA-{b[1].upper()}*{b[2]}:{b[3]}" if len(b) >= 4 else slug


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alleles", nargs="+", required=True)
    ap.add_argument("--fold-sets", nargs="+", required=True)
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--length", type=int, default=9)
    ap.add_argument("--out", default="fold_sets/rq3_variants.csv")
    args = ap.parse_args()

    # canonical binders, in fold-set order — that order is by motif score, so the
    # first n are the most motif-typical and make the best starting points
    starts: dict[str, list[tuple[str, str]]] = {}
    for fs in args.fold_sets:
        for r in csv.reader(open(fs)):
            if len(r) < 4 or r[0] in ("hard", "decoy"):
                continue
            a = slug_to_allele(r[2])
            if a in args.alleles and len(r[3]) == args.length:
                starts.setdefault(a, []).append((r[2], r[3].upper()))

    rows, missing = [], []
    for allele in args.alleles:
        seeds = starts.get(allele, [])[:args.n_seeds]
        if len(seeds) < args.n_seeds:
            missing.append(f"{allele} ({len(seeds)} of {args.n_seeds})")
        if not seeds:
            continue
        locus = f"hla-{allele.split('*')[0][-1].lower()}"
        for si, (slug, wt) in enumerate(seeds):
            # wild type once per seed, as the reference for every variant from it
            rows.append(["wt", locus, slug, wt, f"seed{si}"])
            for pos in range(args.length):
                for aa in AA:
                    if aa == wt[pos]:
                        continue
                    var = wt[:pos] + aa + wt[pos + 1:]
                    rows.append(["mut", locus, slug, var,
                                 f"seed{si}_p{pos + 1}{aa}"])

    if missing:
        print(f"fewer seeds than requested: {missing}")
    if not rows:
        raise SystemExit("no variants built — check the allele names match the "
                         "fold sets")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        csv.writer(fh, lineterminator="\n").writerows(rows)

    n_wt = sum(1 for r in rows if r[0] == "wt")
    n_mut = len(rows) - n_wt
    per = {}
    for r in rows:
        per[slug_to_allele(r[2])] = per.get(slug_to_allele(r[2]), 0) + 1

    print(f"\nwrote {len(rows)} folds to {args.out}")
    print(f"  {n_wt} wild-type references, {n_mut} variants")
    for a, n in sorted(per.items()):
        print(f"    {a:<14} {n}")
    print(f"\n  ESMFold2 at ~20 s: {len(rows) * 20 / 3600:.1f} h")
    print(f"  AF3 at ~72 s:      {len(rows) * 72 / 3600:.1f} h")

    print(f"""
Fold with:
  conda activate esmfold2
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup python -u \\
    scripts/fold_esmfold2.py --csv {args.out} \\
    --sequences data/sequences --out esmfold2-rq3 \\
    > /tmp/fold_rq3.log 2>&1 &

Note the tag column is 'wt'/'mut' rather than 'NA'/'hard', so any downstream script
that infers binder status from the tag will need telling — these are not
binder/decoy pairs and no AUROC is defined over them. The analysis is a landscape
comparison, not a discrimination task.""")


if __name__ == "__main__":
    main()