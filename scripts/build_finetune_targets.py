"""Build an alphafold_finetune targets file from our fold sets.

Why this exists
---------------
Motmaen et al. (PNAS 2023) published fine-tuned AlphaFold parameters and their
training splits. The overlap check against our benchmark came back at zero exact
allele-peptide pairs across all 360 complexes, with a single peptide appearing
anywhere in their fine-tuning data — cleaner than any external baseline we have
tested, since MHCflurry overlapped on 121 of 144 peptides and NetMHCpan's training
data is not public at all.

What it buys, for RQ2 more than RQ1
------------------------------------
The standing objection to RQ2's null is that structure sits at 0.745-0.858 against
sequence at 0.930, so of course combining a weaker signal with a stronger one does
not help. The per-allele z-scored run partly answers that — structure reaches 0.857,
within 0.064 — but z-scoring is transductive, so it is not airtight.

A fine-tuned model that lands near sequence and *still* contributes nothing when
combined would settle it: two comparably strong predictors with zero synergy is
unambiguous redundancy rather than a strength mismatch.

It is also worth running on its own terms. Motmaen's Class I test set covers five of
our fifteen alleles and no HLA-C whatsoever, so our panel evaluates their model on
exactly the alleles it has never been assessed against.

The 181-residue truncation
--------------------------
alphafold_finetune expects a two-chain target, MHC and peptide, with no
beta-2-microglobulin — unlike HISTOFold, which uses three chains and truncates the
MHC to 274. The alignment files map target positions to template positions by pure
identity, so the target length must match the template exactly:

    1k5n_alignments.tsv:  target_len = template_len = 190 = 181 MHC + 9 peptide

Their own 9mer training rows use 181, and their 10mer example uses 175 + 10 = 185
against a different alignment file. Getting this wrong does not error — it silently
misaligns every complex against the template — so the script checks the arithmetic
against the alignment file rather than trusting the constant.

Usage, run from the alphafold_finetune repo root with datasets_alphafold_finetune
symlinked in (the template paths inside the alignment file are relative to it):

    python build_finetune_targets.py \
        --fold-sets fold_sets/fold_set_v2.csv fold_sets/fold_set_v4.csv \
        --sequences data/sequences \
        --alignfile datasets_alphafold_finetune/pmhc_finetune/alignments/1k5n_alignments.tsv \
        --out targets_pmhcpresent.tsv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def slug_to_allele(slug: str) -> str:
    b = slug.split("_")
    return f"HLA-{b[1].upper()}*{b[2]}:{b[3]}" if len(b) >= 4 else slug


def allele_short(slug: str) -> str:
    """hla_a_02_01 -> A*02:01, the form their `mhc` column uses."""
    b = slug.split("_")
    return f"{b[1].upper()}*{b[2]}:{b[3]}" if len(b) >= 4 else slug


def alignment_target_len(path: Path) -> int:
    """Read target_len from the alignment file and verify it is self-consistent."""
    with open(path) as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    if not rows:
        raise SystemExit(f"{path}: empty")
    lens = {int(r["target_len"]) for r in rows}
    tlens = {int(r["template_len"]) for r in rows}
    if len(lens) != 1:
        raise SystemExit(f"{path}: templates disagree on target_len: {lens}")
    if lens != tlens:
        raise SystemExit(f"{path}: target_len {lens} != template_len {tlens}; the "
                         f"identity mapping would not hold")
    # confirm the mapping really is positional identity
    pairs = rows[0]["target_to_template_alignstring"].split(";")
    if any(a != b for a, b in (p.split(":") for p in pairs)):
        raise SystemExit(f"{path}: alignstring is not a pure identity mapping, so "
                         f"it cannot be reused across targets")
    if len(pairs) != lens.pop():
        raise SystemExit(f"{path}: alignstring length does not match target_len")
    return int(rows[0]["target_len"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold-sets", nargs="+", required=True)
    ap.add_argument("--sequences", default="data/sequences")
    ap.add_argument("--alignfile", required=True)
    ap.add_argument("--peptide-length", type=int, default=9)
    ap.add_argument("--out", default="targets_pmhcpresent.tsv")
    args = ap.parse_args()

    align = Path(args.alignfile)
    total = alignment_target_len(align)
    mhc_len = total - args.peptide_length
    print(f"alignment file declares target_len {total}")
    print(f"-> MHC chain truncated to {mhc_len}, peptide {args.peptide_length}\n")

    seqdir = Path(args.sequences)
    alleles = {}
    for locus in ("a", "b", "c"):
        f = seqdir / f"hla_{locus}.json"
        if f.exists():
            alleles.update(json.loads(f.read_text()))

    rows, skipped, seen = [], [], set()
    for fs in args.fold_sets:
        for r in csv.reader(open(fs)):
            if len(r) < 4:
                continue
            tag, _locus, slug, peptide = r[0], r[1], r[2], r[3]
            if len(peptide) != args.peptide_length:
                skipped.append(f"{peptide} (length {len(peptide)})")
                continue
            rec = alleles.get(slug)
            if rec is None:
                skipped.append(f"{slug} (no sequence)")
                continue
            full = rec["canonical_sequence"]
            if len(full) < mhc_len:
                skipped.append(f"{slug} (sequence only {len(full)} residues)")
                continue
            mhc = full[:mhc_len]

            targetid = f"{tag}__{slug}__{peptide.lower()}"
            if targetid in seen:      # v2 and v4 share no alleles, but be safe
                continue
            seen.add(targetid)
            rows.append({
                "mhc": allele_short(slug),
                "start": 0,
                "peptide": peptide,
                "targetid": targetid,
                "target_chainseq": f"{mhc}/{peptide}",
                "templates_alignfile": str(align),
            })

    if skipped:
        print(f"skipped {len(skipped)}: {skipped[:5]}")
    if not rows:
        raise SystemExit("no targets built")

    # every chainseq must be exactly the alignment length, or the template mapping
    # silently misaligns rather than failing
    bad = [r["targetid"] for r in rows
           if len(r["target_chainseq"].replace("/", "")) != total]
    if bad:
        raise SystemExit(f"{len(bad)} targets are not {total} residues: {bad[:3]}")

    fields = ["mhc", "start", "peptide", "targetid", "target_chainseq",
              "templates_alignfile"]
    # DictWriter defaults to \r\n regardless of the newline= setting, and a stray
    # carriage return would corrupt the final column
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, delimiter="\t",
                           lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    n_b = sum(1 for r in rows if not r["targetid"].startswith(("hard__", "decoy__")))
    print(f"wrote {len(rows)} targets to {args.out}")
    print(f"  {n_b} binders / {len(rows) - n_b} decoys, "
          f"{len({r['mhc'] for r in rows})} alleles")
    print(f"  all chainseqs verified at {total} residues\n")

    print("Run both models so the vanilla/fine-tuned contrast is on identical")
    print("complexes — that contrast is the point, not the fine-tuned score alone:\n")
    print("  export AFDD=$HOME/colabfold_cache/colabfold")
    print(f"  python run_prediction.py --targets {args.out} \\")
    print("      --data_dir $AFDD --outfile_prefix pmhc_vanilla \\")
    print("      --model_names model_2_ptm --ignore_identities\n")
    print(f"  python run_prediction.py --targets {args.out} \\")
    print("      --data_dir $AFDD --outfile_prefix pmhc_finetuned \\")
    print("      --model_names model_2_ptm_ft \\")
    print("      --model_params_files datasets_alphafold_finetune/params/"
          "mixed_mhc_pae_run6_af_mhc_params_20640.pkl \\")
    print("      --ignore_identities")


if __name__ == "__main__":
    main()