"""Prepare MHC Motif Atlas peptides into a labelled, train-ready table.

The atlas file (all_peptides.txt) is *positives only*: every row is a peptide
observed as presented by an allele. A presentation classifier needs negatives
too, so this script:

  1. Filters to classical human HLA (A/B/C) and to peptide lengths 8-11.
  2. Normalises the allele string ("A0201" -> "HLA-A*02:01").
  3. Labels every surviving atlas row as a positive (label=1).
  4. Generates negatives (label=0), per allele, length-matched to that
     allele's positives, at a configurable ratio.
  5. Writes a shuffled CSV with columns: peptide, allele, label, length.

Negative-generation modes
-------------------------
  proteome      Sample random sub-peptides from a human proteome FASTA that the
                allele does not present. This is the field-standard, biologically
                honest decoy set. Requires --proteome.
  peptide-pool  Sample from the pool of peptides observed for *other* alleles
                (real presented peptides, presumed not presented by this allele).
                Needs no external data, so it lets you train immediately while a
                proteome FASTA / Chris's guidance is pending. Swap to `proteome`
                later by re-running -- nothing downstream changes.

NOTE ON ALLELE FORMAT: the normalised form here is "HLA-A*02:01", matching the
canonical key produced by the pseudosequence loader's normalize_allele. The
pseudosequences themselves come from Chris's JSONs, keyed by protein_allele_name
(also "HLA-A*02:01"), so the atlas and pseudoseq tables join cleanly.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd

STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")
MIN_LEN_DEFAULT = 8
MAX_LEN_DEFAULT = 11


def normalise_allele(raw: str) -> str | None:
    """'A0201' -> 'HLA-A*02:01'. Returns None if not a classical HLA code.

    This matches the canonical key format used by the pseudosequence loader
    (PseudoSequenceMap.normalize_allele), so atlas alleles join cleanly against
    the pocket_pseudosequence lookup built from Chris's JSONs.
    """
    raw = raw.strip()
    if len(raw) < 5 or raw[0] not in "ABC" or not raw[1:].isdigit():
        return None
    locus, digits = raw[0], raw[1:]
    group, protein = digits[:2], digits[2:]
    return f"HLA-{locus}*{group}:{protein}"


def is_valid_peptide(pep: str) -> bool:
    return bool(pep) and set(pep) <= STANDARD_AA


def load_positives(path: Path, min_len: int, max_len: int) -> pd.DataFrame:
    """Load the atlas file, filter, normalise. Returns df[peptide, allele]."""
    df = pd.read_csv(path, sep="\t")
    df.columns = [c.strip().lower() for c in df.columns]
    if not {"allele", "peptide"} <= set(df.columns):
        raise ValueError(f"expected 'Allele'/'Peptide' columns, got {list(df.columns)}")

    df["peptide"] = df["peptide"].astype(str).str.strip().str.upper()
    df["allele"] = df["allele"].map(normalise_allele)

    n0 = len(df)
    df = df[df["allele"].notna()]
    df = df[df["peptide"].map(is_valid_peptide)]
    plen = df["peptide"].str.len()
    df = df[(plen >= min_len) & (plen <= max_len)]
    df = df.drop_duplicates(subset=["allele", "peptide"]).reset_index(drop=True)
    print(f"  positives: {n0} raw -> {len(df)} after filtering "
          f"(classical HLA, {min_len}-{max_len}mers, deduped)")
    return df[["peptide", "allele"]]


def parse_fasta(path: Path) -> list[str]:
    """Minimal FASTA reader. Returns clean protein sequences (standard AAs only)."""
    seqs, cur = [], []
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                if cur:
                    seqs.append("".join(cur))
                    cur = []
            else:
                cur.append(line.strip().upper())
    if cur:
        seqs.append("".join(cur))
    # keep only the part of each protein usable for clean windows is handled at
    # sample time; here just drop empties.
    return [s for s in seqs if s]


class ProteomeSampler:
    """Samples random valid peptides of a requested length from a proteome."""

    def __init__(self, sequences: list[str], rng: random.Random):
        self.rng = rng
        # bucket sequences by minimum usable length lazily; keep all, filter at draw
        self.sequences = sequences

    def sample(self, length: int, exclude: set[str], max_tries: int = 50) -> str | None:
        for _ in range(max_tries):
            seq = self.rng.choice(self.sequences)
            if len(seq) < length:
                continue
            start = self.rng.randint(0, len(seq) - length)
            pep = seq[start:start + length]
            if is_valid_peptide(pep) and pep not in exclude:
                return pep
        return None


def negatives_proteome(pos: pd.DataFrame, sampler: ProteomeSampler,
                       ratio: float, rng: random.Random) -> pd.DataFrame:
    rows = []
    for allele, grp in pos.groupby("allele"):
        present = set(grp["peptide"])
        for length, n_pos in grp["peptide"].str.len().value_counts().items():
            n_neg = round(n_pos * ratio)
            drawn = set()
            attempts = 0
            while len(drawn) < n_neg and attempts < n_neg * 20 + 100:
                attempts += 1
                pep = sampler.sample(int(length), present | drawn)
                if pep is not None:
                    drawn.add(pep)
            rows.extend((pep, allele) for pep in drawn)
    return pd.DataFrame(rows, columns=["peptide", "allele"])


def negatives_peptide_pool(pos: pd.DataFrame, ratio: float,
                           rng: random.Random) -> pd.DataFrame:
    """Negatives = peptides seen for OTHER alleles, bucketed by length."""
    by_len: dict[int, list[str]] = {}
    for pep in pos["peptide"]:
        by_len.setdefault(len(pep), []).append(pep)

    rows = []
    for allele, grp in pos.groupby("allele"):
        present = set(grp["peptide"])
        for length, n_pos in grp["peptide"].str.len().value_counts().items():
            pool = by_len.get(int(length), [])
            n_neg = round(n_pos * ratio)
            drawn = set()
            attempts = 0
            while len(drawn) < n_neg and attempts < n_neg * 20 + 100:
                attempts += 1
                pep = rng.choice(pool)
                if pep not in present:
                    drawn.add(pep)
            rows.extend((pep, allele) for pep in drawn)
    return pd.DataFrame(rows, columns=["peptide", "allele"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, type=Path, help="all_peptides.txt")
    ap.add_argument("--output", required=True, type=Path, help="output .csv")
    ap.add_argument("--neg-mode", choices=["proteome", "peptide-pool"],
                    default="proteome")
    ap.add_argument("--proteome", type=Path, help="human proteome FASTA (proteome mode)")
    ap.add_argument("--ratio", type=float, default=1.0,
                    help="negatives per positive (default 1.0)")
    ap.add_argument("--min-len", type=int, default=MIN_LEN_DEFAULT)
    ap.add_argument("--max-len", type=int, default=MAX_LEN_DEFAULT)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    print("Loading positives...")
    pos = load_positives(args.input, args.min_len, args.max_len)

    print(f"Generating negatives (mode={args.neg_mode}, ratio={args.ratio})...")
    if args.neg_mode == "proteome":
        if not args.proteome:
            ap.error("--proteome FASTA is required for proteome mode")
        seqs = parse_fasta(args.proteome)
        print(f"  proteome: {len(seqs)} sequences loaded")
        sampler = ProteomeSampler(seqs, rng)
        neg = negatives_proteome(pos, sampler, args.ratio, rng)
    else:
        neg = negatives_peptide_pool(pos, args.ratio, rng)
    print(f"  negatives: {len(neg)} generated")

    pos = pos.assign(label=1)
    neg = neg.assign(label=0)
    out = pd.concat([pos, neg], ignore_index=True)
    out["length"] = out["peptide"].str.len()
    out = out.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    print(f"\nWrote {len(out)} rows to {args.output}")
    print(f"  positives: {(out.label == 1).sum()}  negatives: {(out.label == 0).sum()}")
    print(f"  alleles:   {out.allele.nunique()}")
    print("  length distribution:")
    for length, n in out["length"].value_counts().sort_index().items():
        print(f"    {length}mer: {n}")


if __name__ == "__main__":
    main()