"""Confidence metrics (ipTM, pTM, pLDDT and relatives) from all three folding models.

Why this exists
---------------
Every structural number reported so far is derived from PAE. The claim that
confidence metrics are uninformative was made on the old 60-complex easy-decoy set
at n=6 per allele, before the split fix. This retests it on fold set v2 across
ESMFold2, Boltz and AF2.

Two features here have never been tested and are the reason this is worth running:

  iptm_pep_mhc     ESMFold2 writes a 3x3 per-chain-pair ipTM matrix. Chains are
                   A=MHC, B=b2m, C=peptide, so entries [0][2] and [2][0] are the
                   peptide-groove pair specifically. On a spot check these sat at
                   0.87 and 0.79 where whole-complex ipTM was 0.971 -- i.e. real
                   spread where the global figure was pinned near 1.

  complex_iplddt   Boltz writes interface pLDDT and interface PDE. Both are
  complex_ipde     interface-localised, which is the property that made anchor PAE
                   work where whole-complex confidence did not.

The three models write different fields, so the output is deliberately ragged:
each row carries whatever its model provides. scripts/auroc_structure.py scores
only the columns it recognises, so absent features are simply skipped.

Direction: ipTM / pTM / pLDDT / structure_confidence are higher-is-better.
ipde / pde are error measures, lower-is-better. Both are emitted as-is; the
direction is declared in auroc_structure.py's LOWER_IS_BINDING table.

Usage:
    python scripts/extract_confidence.py esmfold2-v2 --model esmfold2 \
        --out conf_esmfold2_v2.csv
    python scripts/extract_confidence.py \
        third_party/HISTOFold/outputs/experiments/fold_set_v2_v2 --model af2 \
        --fold-set fold_sets/fold_set_v2.csv --out conf_af2_v2.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

PEPTIDE_CHAIN = 2   # A=0 MHC, B=1 b2m, C=2 peptide


def slug_to_allele(slug: str) -> str:
    b = slug.split("_")
    return f"HLA-{b[1].upper()}*{b[2]}:{b[3]}" if len(b) >= 4 else slug


def parse_tagged_name(name: str):
    """'{tag}__{allele_slug}__{peptide}' -> (allele, peptide, is_decoy)."""
    parts = name.split("__")
    if len(parts) < 3:
        return None
    tag, slug, peptide = parts[0], parts[1], parts[-1]
    return slug_to_allele(slug), peptide, tag in ("decoy", "hard")


def load_fold_set(path: str) -> dict:
    """prediction_code -> (allele, peptide, is_decoy), for HISTOFold names."""
    out = {}
    with open(path) as fh:
        for row in csv.reader(fh):
            if len(row) < 4:
                continue
            tag, _locus, slug, peptide = row[0], row[1], row[2], row[3]
            meta = (slug_to_allele(slug), peptide, tag in ("decoy", "hard"))
            # HISTOFold has produced four directory naming schemes; the v4 input
            # format (three-column header) writes the pdb_id column into the name,
            # which for our inputs is the literal "NA" and carries no label
            out[f"{slug}_{peptide.lower()}"] = meta
            out[f"{slug}__{peptide.lower()}"] = meta
            out[f"{tag}__{slug}__{peptide.lower()}"] = meta
            out[f"NA__{slug}__{peptide.lower()}"] = meta
    return out


def from_metrics_json(fold: Path) -> dict:
    """ESMFold2 and Boltz: outputs/files/prediction/metrics.json."""
    p = fold / "outputs" / "files" / "prediction" / "metrics.json"
    if not p.exists():
        return {}
    with open(p) as fh:
        d = json.load(fh)
    m = d.get("best_sample", {}).get("metrics", {})
    if not m:
        return {}

    out = {}
    for k in ("iptm", "ptm", "complex_plddt", "complex_iplddt", "complex_pde",
              "complex_ipde", "structure_confidence", "protein_iptm"):
        if k in m and isinstance(m[k], (int, float)):
            out[k] = float(m[k])

    # ESMFold2 only: per-chain-pair ipTM. Peptide-vs-MHC is the off-diagonal
    # A/C pair, which is asymmetric, so keep both directions and the mean.
    pc = m.get("pair_chains_iptm")
    if isinstance(pc, list) and len(pc) > PEPTIDE_CHAIN:
        try:
            a = float(pc[0][PEPTIDE_CHAIN])
            b = float(pc[PEPTIDE_CHAIN][0])
            out["iptm_mhc_pep"] = a
            out["iptm_pep_mhc"] = b
            out["iptm_pep_mhc_mean"] = (a + b) / 2
            out["iptm_pep_self"] = float(pc[PEPTIDE_CHAIN][PEPTIDE_CHAIN])
        except (IndexError, TypeError, ValueError):
            pass
    return out


def from_af2_scores(fold: Path, peptide_len: int) -> dict:
    """AF2/ColabFold: the rank_001 scores json."""
    hits = sorted(fold.glob("*_scores_rank_001_*.json"))
    if not hits:
        return {}
    with open(hits[0]) as fh:
        d = json.load(fh)
    out = {}
    for k in ("iptm", "ptm"):
        if k in d and isinstance(d[k], (int, float)):
            out[k] = float(d[k])
    pl = d.get("plddt")
    if isinstance(pl, list) and pl:
        arr = np.asarray(pl, dtype=float)
        out["complex_plddt"] = float(arr.mean()) / 100.0
        if 0 < peptide_len < len(arr):
            out["plddt_peptide"] = float(arr[-peptide_len:].mean()) / 100.0
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--model", required=True,
                    choices=["esmfold2", "boltz", "af2"])
    ap.add_argument("--fold-set",
                    help="required for --model af2 (HISTOFold names carry no tag)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    fold_set = load_fold_set(args.fold_set) if args.fold_set else None
    if args.model == "af2" and fold_set is None:
        raise SystemExit("--fold-set is required for --model af2")

    rows, skipped = [], []
    for fold in sorted(Path(args.root).iterdir()):
        if not fold.is_dir():
            continue
        if args.model == "af2":
            meta = fold_set.get(fold.name)
            if meta is None:
                skipped.append(fold.name)
                continue
            allele, peptide, is_decoy = meta
            feats = from_af2_scores(fold, len(peptide))
        else:
            meta = parse_tagged_name(fold.name)
            if meta is None:
                skipped.append(fold.name)
                continue
            allele, peptide, is_decoy = meta
            feats = from_metrics_json(fold)
        if not feats:
            skipped.append(fold.name)
            continue
        rows.append({"allele": allele, "peptide": peptide,
                     "kind": "decoy" if is_decoy else "binder", **feats})

    if skipped:
        print(f"skipped {len(skipped)} folds with no readable metrics: "
              f"{skipped[:3]}")

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No metrics extracted.")
    df.to_csv(args.out, index=False)

    n_b = (df.kind == "binder").sum()
    feats = [c for c in df.columns if c not in ("allele", "peptide", "kind")]
    print(f"\n{len(df)} folds ({n_b} binders / {len(df) - n_b} decoys)")
    print(f"features: {feats}\n")

    print("=== binder vs decoy, pooled (mean) ===")
    print(f"{'feature':<22} {'binder':>9} {'decoy':>9} {'gap':>9}")
    for f in feats:
        b = df[df.kind == "binder"][f].mean()
        d = df[df.kind == "decoy"][f].mean()
        print(f"{f:<22} {b:>9.4f} {d:>9.4f} {b - d:>+9.4f}")

    print(f"\nWrote {args.out}")
    print(f"Next: python scripts/auroc_structure.py --pae {args.out} "
          f"--out auroc_conf.csv")


if __name__ == "__main__":
    main()