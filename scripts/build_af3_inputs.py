"""Build AlphaFold 3 JSON inputs for a pMHC fold set, reusing HISTOFold's MSAs.

Why this exists
---------------
AlphaFold 3's standard pipeline needs roughly 630 GB of genetic databases. That is
not available here, but AF3 accepts pre-computed alignments per chain, and setting
`unpairedMsa` and `pairedMsa` to the empty string suppresses the database search
entirely. Reusing Chris Thorpe's tuned MSAs rather than folding single-sequence also
keeps the AF2/AF3 comparison controlled: both models then see the same alignment.

Matching HISTOFold's construction
---------------------------------
HISTOFold builds one a3m per prediction, prepending the *target's own* sequence as
the query row (`functions.py:create_combined_sequence`, then
`prediction_msa.replace('###', prediction_string)` in run_msa_predictions.py). The
query is `allele_sequence[0:274] + b2m + peptide`, i.e. the MHC chain truncated to
274 residues.

AF3 requires the same thing but per chain: `Msa.from_a3m` raises if the first row is
not exactly the query sequence. So this writes one a3m per complex containing that
allele's MHC chain (truncated to 274) followed by the template rows' MHC columns.

Two consequences worth noting:

  TRUNCATION   Chain A must be `mhc[:274]`, not the full canonical sequence. Ours
               are 275 residues, and passing the untruncated sequence produces a
               query/MSA mismatch error from AF3.

  PER-COMPLEX  216 small a3m files rather than one shared alignment. They are
               about 450 KB each -- roughly 100 MB in total.

The MSA split
-------------
HISTOFold a3m files hold all three chains concatenated per row, with lengths in the
header (`#274,99,9`). The alignment is gapped-only -- every row in len9_v3b.a3m is
exactly 382 characters and no row contains lowercase -- so a positional split is
exact. A3M uses lowercase for insertions, which do not occupy alignment columns;
this script checks that invariant and aborts if it does not hold, because a naive
slice would otherwise silently misalign every row.

Only the MHC chain gets an alignment. beta-2-microglobulin is effectively invariant
and a 9mer has no meaningful homologs, so both are given empty MSAs -- which is also
what suppresses the database search for those chains. Empty `templates` lists are
required too: omitting the field means "not yet computed" rather than "none", and
AF3 refuses to featurise without it when --norun_data_pipeline is set.

Usage:
    python scripts/build_af3_inputs.py \
        --fold-set fold_sets/fold_set_v4.csv \
        --msa third_party/HISTOFold/inputs/msa_templates/len9_v3b.a3m \
        --sequences data/sequences \
        --out-dir /tmp/af3/inputs \
        --msa-dir /tmp/af3/msas
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

MHC_TRUNCATE = 274   # matches HISTOFold's create_combined_sequence(..., 274)


def parse_a3m(path: Path):
    """Return (chain_lengths, [(header, sequence)]) from a HISTOFold a3m."""
    lines = path.read_text().splitlines()
    lengths, entries, header = None, [], None
    for ln in lines:
        if ln.startswith("#") and lengths is None:
            lengths = [int(x) for x in ln[1:].split()[0].split(",")]
            continue
        if ln.startswith("###") or not ln.strip():
            continue
        if ln.startswith(">"):
            header = ln
            continue
        if header is not None:
            entries.append((header, ln))
            header = None
    if lengths is None:
        raise SystemExit(f"{path}: no '#lengths' header found")
    return lengths, entries


def split_msa(path: Path):
    """MHC-chain columns from a concatenated HISTOFold a3m."""
    lengths, entries = parse_a3m(path)
    total = sum(lengths)
    ragged = [s for _, s in entries if len(s) != total]
    if ragged:
        raise SystemExit(
            f"{path}: {len(ragged)} rows are not {total} characters. A positional "
            f"split would be wrong. Aborting.")
    lower = [s for _, s in entries if any(c.islower() for c in s)]
    if lower:
        raise SystemExit(
            f"{path}: {len(lower)} rows contain lowercase (a3m insertion) "
            f"characters, which do not occupy alignment columns. A positional "
            f"split would be wrong. Aborting.")

    mhc_len = lengths[0]
    if mhc_len != MHC_TRUNCATE:
        print(f"NOTE: MSA declares {mhc_len} MHC columns, truncating chain A to "
              f"match rather than the usual {MHC_TRUNCATE}")
    out = []
    for h, s in entries:
        seg = s[:mhc_len]
        if set(seg) <= {"-"}:            # all-gap rows carry nothing
            continue
        out.append((h, seg))
    return mhc_len, out


def slug_to_allele(slug: str) -> str:
    b = slug.split("_")
    return f"HLA-{b[1].upper()}*{b[2]}:{b[3]}" if len(b) >= 4 else slug


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold-set", required=True)
    ap.add_argument("--msa", required=True)
    ap.add_argument("--sequences", default="data/sequences")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--msa-dir", help="default: <out-dir>/msas")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--peptide-length", type=int, default=9)
    args = ap.parse_args()

    seqdir = Path(args.sequences)
    with open(seqdir / "human_b2m.json") as fh:
        b2m = json.load(fh)["canonical_sequence"]
    alleles = {}
    for locus in ("a", "b", "c"):
        with open(seqdir / f"hla_{locus}.json") as fh:
            alleles[locus] = json.load(fh)

    mhc_len, template_rows = split_msa(Path(args.msa))
    print(f"MSA: {len(template_rows)} template rows, MHC columns 1-{mhc_len}")

    out_dir = Path(args.out_dir)
    msa_dir = Path(args.msa_dir) if args.msa_dir else out_dir / "msas"
    out_dir.mkdir(parents=True, exist_ok=True)
    msa_dir.mkdir(parents=True, exist_ok=True)

    template_text = "\n".join(f"{h}\n{s}" for h, s in template_rows)

    n, skipped, msa_cache = 0, [], {}
    with open(args.fold_set) as fh:
        for row in csv.reader(fh):
            if len(row) < 4:
                continue
            tag, _locus_field, slug, peptide = row[0], row[1], row[2], row[3]
            if len(peptide) != args.peptide_length:
                skipped.append(f"{peptide} (length)")
                continue
            locus = slug.split("_")[1]
            table = alleles.get(locus)
            if table is None or slug not in table:
                skipped.append(f"{slug} (no sequence)")
                continue

            mhc_full = table[slug]["canonical_sequence"]
            mhc = mhc_full[:mhc_len]
            if len(mhc) != mhc_len:
                skipped.append(f"{slug} (sequence shorter than {mhc_len})")
                continue

            # one a3m per allele: query row first, as AF3 requires, then templates
            if slug not in msa_cache:
                path = msa_dir / f"{slug}_chainA.a3m"
                path.write_text(f">query_{slug}\n{mhc}\n{template_text}\n")
                msa_cache[slug] = path
            msa_path = msa_cache[slug]

            name = f"{tag}__{slug}__{peptide.lower()}"
            spec = {
                "name": name,
                "sequences": [
                    {"protein": {
                        "id": "A",
                        "sequence": mhc,
                        "description": f"{slug_to_allele(slug)} heavy chain "
                                       f"(truncated to {mhc_len})",
                        "unpairedMsaPath": str(msa_path.resolve()),
                        "pairedMsa": "",
                        "templates": [],
                    }},
                    {"protein": {
                        "id": "B",
                        "sequence": b2m,
                        "description": "beta-2-microglobulin",
                        "unpairedMsa": "",
                        "pairedMsa": "",
                        "templates": [],
                    }},
                    {"protein": {
                        "id": "C",
                        "sequence": peptide,
                        "description": f"peptide {peptide}",
                        "unpairedMsa": "",
                        "pairedMsa": "",
                        "templates": [],
                    }},
                ],
                "modelSeeds": [args.seed],
                "dialect": "alphafold3",
                "version": 4,
            }
            (out_dir / f"{name}.json").write_text(json.dumps(spec, indent=2))
            n += 1

    if skipped:
        print(f"skipped {len(skipped)}: {skipped[:5]}")
    print(f"\nwrote {n} JSON inputs to {out_dir}")
    print(f"wrote {len(msa_cache)} per-allele MSAs to {msa_dir}")
    print(f"""
Smoke-test one before the batch:

  mkdir -p /tmp/af3/output
  singularity exec --nv \\
    --bind {out_dir}:/root/af_input \\
    --bind /tmp/af3/output:/root/af_output \\
    --bind /tmp/af3/models:/root/models \\
    --bind {msa_dir}:{msa_dir} \\
    /tmp/af3/alphafold3.sif \\
    /alphafold3_venv/bin/python3 /app/alphafold/run_alphafold.py \\
    --json_path=/root/af_input/<one>.json \\
    --model_dir=/root/models \\
    --output_dir=/root/af_output \\
    --flash_attention_implementation=xla \\
    --norun_data_pipeline

Use `singularity exec ... python3 run_alphafold.py` rather than `singularity run`:
the image's runscript calls `uv run`, which tries to reinstall the package into the
read-only container and fails.""")


if __name__ == "__main__":
    main()
