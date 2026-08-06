"""Does the Motmaen et al. fine-tuning set overlap our benchmark?

Why this runs before anything else
-----------------------------------
Motmaen et al. (PNAS 2023) fine-tuned AlphaFold on peptide-MHC binding data and
published both the fine-tuned parameters and the training/validation splits. That
makes it the one fine-tuned model we can evaluate — and, unlike NetMHCpan-4.1, the
one whose training data we can actually inspect.

That matters because the NetMHCpan comparison failed twice for exactly this reason.
On fold set v2 it was advantaged by resembling the PWM selection criterion; on the
affinity fold set its score rose to 0.992, consistent with those peptides being in
its training data, which is not public so the suspicion could not be settled. A
fine-tuned comparison that repeats that mistake is worth nothing, so the overlap is
checked first and the result decides whether the run is worth doing.

Three levels of overlap, in decreasing severity:

  PAIR      the same (allele, peptide) pair appears in training. Directly
            invalidating for that complex.
  PEPTIDE   the peptide appears under a different allele. Weaker, and our earlier
            cross-allele analysis found training exposure DEPRESSES scores on this
            benchmark rather than inflating them — hard decoys are real ligands the
            model has seen as positives. Still worth quantifying.
  ALLELE    the allele appears at all. Not leakage, but tells us whether the
            fine-tuning ever saw the allele, which bears on the rare HLA-C cases.

Also reports near-matches within a Hamming distance of 1 or 2, since a peptide one
substitution from a training example is not independent either.

Usage:
    python scripts/check_motmaen_overlap.py \
        --train datasets_alphafold_finetune/combo_1and2_train.tsv \
                datasets_alphafold_finetune/combo_1and2_valid.tsv \
        --fold-sets fold_sets/fold_set_v2.csv fold_sets/fold_set_v4.csv
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import pandas as pd


def slug_to_allele(slug: str) -> str:
    b = slug.split("_")
    return f"HLA-{b[1].upper()}*{b[2]}:{b[3]}" if len(b) >= 4 else slug


def normalise_allele(a: str) -> str:
    """Motmaen's tables use several spellings; reduce to HLA-A*02:01 form."""
    a = str(a).strip().replace("_", "").replace(" ", "")
    if a.upper().startswith("HLA-"):
        a = a[4:]
    a = a.replace("*", "").replace(":", "")
    # A0201 -> A*02:01 ; longer allele groups keep their extra fields
    if len(a) >= 5 and a[0].isalpha():
        locus, digits = a[0], a[1:]
        if digits[:4].isdigit():
            return f"HLA-{locus.upper()}*{digits[:2]}:{digits[2:4]}"
    return f"HLA-{a.upper()}"


def load_fold_sets(paths) -> pd.DataFrame:
    rows = []
    for p in paths:
        for r in csv.reader(open(p)):
            if len(r) < 4:
                continue
            rows.append({"fold_set": Path(p).stem, "kind": r[0],
                         "allele": slug_to_allele(r[2]), "peptide": r[3].upper()})
    return pd.DataFrame(rows).drop_duplicates(["allele", "peptide", "fold_set"])


def load_training(paths) -> pd.DataFrame:
    frames = []
    for p in paths:
        d = pd.read_csv(p, sep="\t")
        cols = {c.lower(): c for c in d.columns}
        pcol = next((cols[c] for c in ("peptide", "pep", "peptide_sequence")
                     if c in cols), None)
        acol = next((cols[c] for c in ("allele", "mhc", "mhc_allele", "hla")
                     if c in cols), None)
        if pcol is None:
            raise SystemExit(f"{p}: no peptide column. Columns: {list(d.columns)}")
        out = pd.DataFrame({"peptide": d[pcol].astype(str).str.upper()})
        out["allele"] = (d[acol].map(normalise_allele) if acol
                         else "UNKNOWN")
        out["source"] = Path(p).name
        frames.append(out)
        print(f"  {Path(p).name}: {len(d)} rows"
              f"{'' if acol else '  (no allele column found)'}")
    return pd.concat(frames, ignore_index=True)


def hamming_le(a: str, b: str, k: int) -> bool:
    if len(a) != len(b):
        return False
    n = 0
    for x, y in zip(a, b):
        if x != y:
            n += 1
            if n > k:
                return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--fold-sets", nargs="+", required=True)
    ap.add_argument("--max-hamming", type=int, default=2)
    ap.add_argument("--out", default="results/motmaen_overlap.csv")
    args = ap.parse_args()

    print("training files:")
    train = load_training(args.train)
    fs = load_fold_sets(args.fold_sets)
    print(f"\nfold sets: {len(fs)} complexes, "
          f"{fs.peptide.nunique()} distinct peptides, "
          f"{fs.allele.nunique()} alleles\n")

    train_pairs = set(zip(train.allele, train.peptide))
    train_peps = set(train.peptide)
    train_alleles = set(train.allele)
    by_len = defaultdict(list)
    for p in train_peps:
        by_len[len(p)].append(p)

    rows = []
    for _, r in fs.iterrows():
        pair_hit = (r.allele, r.peptide) in train_pairs
        pep_hit = r.peptide in train_peps
        near = 0
        if not pep_hit and args.max_hamming:
            for k in range(1, args.max_hamming + 1):
                if any(hamming_le(r.peptide, t, k) for t in by_len[len(r.peptide)]):
                    near = k
                    break
        rows.append({**r.to_dict(), "pair_in_train": pair_hit,
                     "peptide_in_train": pep_hit,
                     "allele_in_train": r.allele in train_alleles,
                     "nearest_hamming": 0 if pep_hit else (near or None)})
    d = pd.DataFrame(rows)

    print("=" * 66)
    for fset, g in d.groupby("fold_set"):
        n = len(g)
        print(f"\n{fset}  ({n} complexes)")
        print(f"  exact (allele, peptide) pairs in training   "
              f"{g.pair_in_train.sum():>4}  ({g.pair_in_train.mean():.1%})")
        print(f"  peptide in training under any allele        "
              f"{g.peptide_in_train.sum():>4}  ({g.peptide_in_train.mean():.1%})")
        for k in range(1, args.max_hamming + 1):
            c = (g.nearest_hamming == k).sum()
            print(f"  within {k} substitution{'s' if k > 1 else ' '} of a training peptide     "
                  f"{c:>4}  ({c / n:.1%})")
        # the direction that matters most: leakage concentrated in binders
        for kind, gg in g.groupby("kind"):
            lab = "decoys " if kind in ("hard", "decoy") else "binders"
            print(f"    {lab}: {gg.pair_in_train.sum()}/{len(gg)} pairs, "
                  f"{gg.peptide_in_train.sum()}/{len(gg)} peptides")

    print("\n=== allele coverage in the fine-tuning set ===")
    for allele, g in d.groupby("allele"):
        seen = "yes" if g.allele_in_train.iloc[0] else "NO"
        print(f"  {allele:<14} {seen}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(args.out, index=False)
    print(f"\nWrote {args.out}")

    worst = d.pair_in_train.mean()
    print("\n=== verdict ===")
    if worst == 0 and d.peptide_in_train.mean() < 0.05:
        print("  Clean. The fine-tuned comparison is worth running.")
    elif worst == 0:
        print(f"  No exact pairs, but {d.peptide_in_train.mean():.1%} of peptides")
        print("  appear under other alleles. Report a seen/unseen split, as was")
        print("  done for MHCflurry (0.824 seen vs 0.937 unseen).")
    else:
        print(f"  {worst:.1%} of complexes are exact training pairs. Either exclude")
        print("  them and report on the remainder, or state that the comparison")
        print("  cannot be made cleanly — the same conclusion reached for NetMHCpan.")


if __name__ == "__main__":
    main()