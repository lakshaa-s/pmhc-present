"""PAE features from AlphaFold2 (ColabFold/HISTOFold) folds.

Same features and output schema as scripts/analyse_pae.py, so
scripts/auroc_structure.py runs against this unchanged. Two things differ from
the Boltz/ESMFold2 pipelines and are handled here:

  FORMAT   ColabFold writes PAE as JSON, not sample_0_pae.npz. Both the
           `_predicted_aligned_error_v1.json` file and the per-model
           `_scores_rank_00N_*.json` files carry it; the rank_001 scores file is
           preferred because it is the top-ranked model and also carries plddt,
           ptm and iptm.

  LABELS   HISTOFold names folders `{allele_slug}_{peptide.lower()}` with no
           binder/decoy tag, unlike `{tag}__{allele}__{peptide}` elsewhere. The
           label therefore has to come from the fold-set CSV, not the path.

Usage:
    python scripts/analyse_pae_af2.py \
        third_party/HISTOFold/outputs/experiments/fold_set_v2_v2 \
        --fold-set fold_sets/fold_set_v2.csv \
        --anchors data/processed/anchors.json \
        --out pae_af2_v2.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd


def slug_to_allele(slug: str) -> str:
    b = slug.split("_")
    if len(b) < 4:
        return slug
    return f"HLA-{b[1].upper()}*{b[2]}:{b[3]}"


def load_fold_set(path: str) -> dict[str, tuple[str, str, bool]]:
    """prediction_code -> (allele, peptide, is_decoy), from the fold-set CSV."""
    out = {}
    with open(path) as fh:
        for row in csv.reader(fh):
            if len(row) < 4:
                continue
            tag, _locus, slug, peptide = row[0], row[1], row[2], row[3]
            meta = (slug_to_allele(slug), peptide, tag in ("decoy", "hard"))
            # three naming schemes have been seen from HISTOFold:
            #   v2 input:  {slug}_{peptide}
            #   v3 output: {tag}__{slug}__{peptide}   (tag present)
            #   v4 input:  {slug}__{peptide}          (three-column header, no tag)
            out[f"{slug}_{peptide.lower()}"] = meta
            out[f"{slug}__{peptide.lower()}"] = meta
            out[f"{tag}__{slug}__{peptide.lower()}"] = meta
            # HISTOFold writes the pdb_id column into the directory name, which
            # for our inputs is the literal "NA" and carries no label. Key on
            # slug+peptide with any leading field.
            out[f"NA__{slug}__{peptide.lower()}"] = meta
    return out


def load_pae(fold_dir: Path) -> np.ndarray | None:
    """(L, L) PAE matrix from ColabFold output, preferring the rank_001 model."""
    ranked = sorted(fold_dir.glob("*_scores_rank_001_*.json"))
    for path in ranked:
        with open(path) as fh:
            d = json.load(fh)
        if "pae" in d:
            return np.asarray(d["pae"], dtype=float)
        if "predicted_aligned_error" in d:
            return np.asarray(d["predicted_aligned_error"], dtype=float)

    for path in sorted(fold_dir.glob("*predicted_aligned_error*.json")):
        with open(path) as fh:
            d = json.load(fh)
        if isinstance(d, list) and d:
            d = d[0]
        if isinstance(d, dict):
            if "predicted_aligned_error" in d:
                return np.asarray(d["predicted_aligned_error"], dtype=float)
            # legacy sparse format: residue1 / residue2 / distance
            if {"residue1", "residue2", "distance"} <= set(d):
                r1 = np.asarray(d["residue1"], dtype=int)
                r2 = np.asarray(d["residue2"], dtype=int)
                dist = np.asarray(d["distance"], dtype=float)
                n = int(max(r1.max(), r2.max()))
                m = np.zeros((n, n))
                m[r1 - 1, r2 - 1] = dist
                return m
    return None


def fold_features(pae: np.ndarray, peptide_len: int, anchor_pos=None) -> dict:
    """Identical to scripts/analyse_pae.py: peptide is the final residues."""
    n = pae.shape[0]
    if not (0 < peptide_len < n):
        return {}
    pep = slice(n - peptide_len, n)
    mhc = slice(0, n - peptide_len)
    pep_idx = np.arange(n - peptide_len, n)
    interface = np.concatenate([pae[pep, mhc].ravel(), pae[mhc, pep].ravel()])

    def res_vs_mhc(i):
        return np.concatenate([pae[i, mhc].ravel(), pae[mhc, i].ravel()]).mean()

    a2 = res_vs_mhc(pep_idx[1] if peptide_len >= 2 else pep_idx[0])
    ac = res_vs_mhc(pep_idx[-1])
    out = {
        "pae_pep_mhc": float(interface.mean()),
        "pae_anchor2": float(a2),
        "pae_anchorC": float(ac),
        "pae_anchors": float((a2 + ac) / 2),
    }
    if anchor_pos:
        vals = [res_vs_mhc(pep_idx[i]) for i in anchor_pos
                if -peptide_len <= i < peptide_len]
        if vals:
            out["pae_anchors_ic"] = float(np.mean(vals))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="HISTOFold experiment folder")
    ap.add_argument("--fold-set", required=True,
                    help="fold-set CSV the predictions were built from")
    ap.add_argument("--anchors", default="data/processed/anchors.json")
    ap.add_argument("--out", default="pae_af2.csv")
    args = ap.parse_args()

    fold_set = load_fold_set(args.fold_set)
    print(f"fold set: {len(fold_set)} complexes from {args.fold_set}")

    anchor_table = {}
    if args.anchors and Path(args.anchors).exists():
        with open(args.anchors) as fh:
            anchor_table = json.load(fh).get("alleles", {})
        print(f"anchors: {len(anchor_table)} alleles from {args.anchors}")

    rows, missing, unmatched = [], [], []
    for fold in sorted(Path(args.root).iterdir()):
        if not fold.is_dir():
            continue
        if fold.name in fold_set:
            allele, peptide, is_decoy = fold_set[fold.name]
        elif "__" in fold.name:
            # HISTOFold v3 writes {tag}__{allele_slug}__{peptide}, unlike v2's
            # {allele_slug}_{peptide}; the tag is present so no lookup is needed
            parts = fold.name.split("__")
            tag, slug, pep = parts[0], parts[1], parts[-1]
            allele = slug_to_allele(slug)
            is_decoy = tag in ("decoy", "hard")
            key = f"{slug}_{pep.lower()}"
            peptide = fold_set[key][1] if key in fold_set else pep.upper()
        else:
            unmatched.append(fold.name)
            continue
        pae = load_pae(fold)
        if pae is None:
            missing.append(fold.name)
            continue
        feats = fold_features(pae, len(peptide),
                              anchor_table.get(allele, {}).get("anchors"))
        if not feats:
            missing.append(fold.name)
            continue
        rows.append({"allele": allele, "peptide": peptide,
                     "kind": "decoy" if is_decoy else "binder", **feats})

    if unmatched:
        print(f"WARNING: {len(unmatched)} folders not in the fold set, skipped: "
              f"{unmatched[:3]}")
    if missing:
        print(f"WARNING: {len(missing)} folds with no readable PAE: {missing[:5]}")

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    n_b = (df.kind == "binder").sum()
    print(f"\n{len(df)} folds ({n_b} binders / {len(df) - n_b} decoys)")

    print("\n=== binder vs decoy anchor-PAE, per allele ===")
    for allele, sub in df.groupby("allele"):
        b = sub[sub.kind == "binder"].pae_anchors.mean()
        d = sub[sub.kind == "decoy"].pae_anchors.mean()
        flag = "binders lower (expected)" if d > b else "binders HIGHER"
        print(f"  {allele:<14} binder {b:.3f}  decoy {d:.3f}  "
              f"gap {d - b:+.3f}  <-- {flag}")

    print(f"\nWrote {args.out}")
    print(f"Next: python scripts/auroc_structure.py --pae {args.out} "
          f"--out auroc_af2_v2.csv --sequence-csv results/sequence_v2.csv")


if __name__ == "__main__":
    main()
