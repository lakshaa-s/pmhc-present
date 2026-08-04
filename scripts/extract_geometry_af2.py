"""Geometric interface features from AlphaFold2 (ColabFold/HISTOFold) structures.

Why this exists
---------------
Every structural feature tested so far -- PAE and confidence -- is the folding
model's own self-assessment. Both turned out to carry signal only when localised
to the peptide, and neither beat the sequence model. Geometry is different in kind:
contacts and distances are computed from coordinates, so unlike PAE and ipTM they
are not correlated with the model's own uncertainty. That makes them the one
structural category that might behave differently.

They have never been tested on a proper fold set. The earlier claim that geometry
features scored below 0.5 was measured at n=6 per allele on the old easy-decoy set,
before the split fix.

No refolding is needed: ColabFold already writes relaxed PDB structures for every
prediction, so the whole v2 fold set is on disk.

Features, peptide (chain C) against MHC heavy chain (chain A), heavy atoms only.
beta-2-microglobulin (chain B) is excluded -- it is not part of the groove.

  n_contacts          atom pairs within CONTACT_CUTOFF
  n_contacts_close    atom pairs within CLOSE_CUTOFF (tight packing)
  contacts_per_res    n_contacts / peptide length
  anchor2_contacts    contacts made by peptide position 2 (B-pocket anchor)
  anchorC_contacts    contacts made by the C-terminal residue (F-pocket anchor)
  anchor_ic_contacts  contacts made by the allele's high-information positions,
                      from scripts/derive_anchors.py, where available
  min_anchor_dist2    closest P2 -> MHC heavy-atom distance (low = seated)
  min_anchor_distC    closest C-terminus -> MHC distance
  mean_min_dist       mean over peptide residues of the closest MHC distance
  buried_frac         fraction of peptide atoms with at least one MHC contact

Direction is declared in auroc_structure.py's LOWER_IS_BINDING table: contact
counts are higher-is-binding, distances are lower-is-binding.

Usage:
    python scripts/extract_geometry_af2.py \
        third_party/HISTOFold/outputs/experiments/fold_set_v2_v2 \
        --fold-set fold_sets/fold_set_v2.csv \
        --anchors data/processed/anchors.json \
        --out geom_af2_v2.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from biotite.structure.io import pdb

CONTACT_CUTOFF = 4.5
CLOSE_CUTOFF = 3.5
MHC_CHAIN = "A"
PEPTIDE_CHAIN = "C"


def slug_to_allele(slug: str) -> str:
    b = slug.split("_")
    return f"HLA-{b[1].upper()}*{b[2]}:{b[3]}" if len(b) >= 4 else slug


def load_fold_set(path: str) -> dict:
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


def load_structure(pdb_path: Path):
    f = pdb.PDBFile.read(str(pdb_path))
    arr = f.get_structure(model=1)
    return arr[arr.element != "H"]


def geometry(arr, peptide_len: int, anchor_pos=None) -> dict:
    pep = arr[arr.chain_id == PEPTIDE_CHAIN]
    mhc = arr[arr.chain_id == MHC_CHAIN]
    if len(pep) == 0 or len(mhc) == 0:
        return {}

    # pairwise heavy-atom distances, peptide atoms x MHC atoms
    d = np.linalg.norm(pep.coord[:, None, :] - mhc.coord[None, :, :], axis=-1)

    res_ids = np.unique(pep.res_id)
    if len(res_ids) < 2:
        return {}
    res_ids = np.sort(res_ids)

    def res_mask(i):
        """i as a 0-based index into the sorted peptide residues; -1 = C-term."""
        return pep.res_id == res_ids[i]

    out = {
        "n_contacts": int((d < CONTACT_CUTOFF).sum()),
        "n_contacts_close": int((d < CLOSE_CUTOFF).sum()),
    }
    out["contacts_per_res"] = out["n_contacts"] / max(len(res_ids), 1)

    m2, mc = res_mask(1), res_mask(-1)
    out["anchor2_contacts"] = int((d[m2] < CONTACT_CUTOFF).sum())
    out["anchorC_contacts"] = int((d[mc] < CONTACT_CUTOFF).sum())
    out["min_anchor_dist2"] = float(d[m2].min())
    out["min_anchor_distC"] = float(d[mc].min())

    per_res_min = [float(d[res_mask(i)].min()) for i in range(len(res_ids))]
    out["mean_min_dist"] = float(np.mean(per_res_min))
    out["buried_frac"] = float((d.min(axis=1) < CONTACT_CUTOFF).mean())

    if anchor_pos:
        tot = 0
        used = False
        for i in anchor_pos:
            if -len(res_ids) <= i < len(res_ids):
                tot += int((d[res_mask(i)] < CONTACT_CUTOFF).sum())
                used = True
        if used:
            out["anchor_ic_contacts"] = tot
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="HISTOFold experiment folder")
    ap.add_argument("--fold-set", required=True)
    ap.add_argument("--anchors", default="data/processed/anchors.json")
    ap.add_argument("--rank", default="001", help="which ranked model to use")
    ap.add_argument("--relaxed", default="relaxed",
                    choices=["relaxed", "unrelaxed"])
    ap.add_argument("--out", default="geom_af2.csv")
    args = ap.parse_args()

    fold_set = load_fold_set(args.fold_set)
    print(f"fold set: {len(fold_set)} complexes")

    anchor_table = {}
    if args.anchors and Path(args.anchors).exists():
        with open(args.anchors) as fh:
            anchor_table = json.load(fh).get("alleles", {})
        print(f"anchors: {len(anchor_table)} alleles")

    rows, missing, unmatched = [], [], []
    for fold in sorted(Path(args.root).iterdir()):
        if not fold.is_dir():
            continue
        if fold.name not in fold_set:
            unmatched.append(fold.name)
            continue
        allele, peptide, is_decoy = fold_set[fold.name]
        hits = sorted(fold.glob(f"*_{args.relaxed}_rank_{args.rank}_*.pdb"))
        if not hits:
            missing.append(fold.name)
            continue
        try:
            arr = load_structure(hits[0])
            feats = geometry(arr, len(peptide),
                             anchor_table.get(allele, {}).get("anchors"))
        except Exception as e:  # noqa: BLE001
            print(f"  {fold.name}: {e}")
            missing.append(fold.name)
            continue
        if not feats:
            missing.append(fold.name)
            continue
        rows.append({"allele": allele, "peptide": peptide,
                     "kind": "decoy" if is_decoy else "binder", **feats})

    if unmatched:
        print(f"WARNING: {len(unmatched)} folders not in fold set: {unmatched[:3]}")
    if missing:
        print(f"WARNING: {len(missing)} folds unreadable: {missing[:5]}")

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No structures parsed.")
    df.to_csv(args.out, index=False)

    n_b = (df.kind == "binder").sum()
    feats = [c for c in df.columns if c not in ("allele", "peptide", "kind")]
    print(f"\n{len(df)} folds ({n_b} binders / {len(df) - n_b} decoys)")
    print(f"features: {feats}\n")

    print("=== binder vs decoy, pooled means ===")
    print(f"{'feature':<22} {'binder':>10} {'decoy':>10} {'gap':>10}")
    for f in feats:
        b = df[df.kind == "binder"][f].mean()
        dd = df[df.kind == "decoy"][f].mean()
        print(f"{f:<22} {b:>10.3f} {dd:>10.3f} {b - dd:>+10.3f}")

    print(f"\nWrote {args.out}")
    print(f"Next: python scripts/auroc_structure.py --pae {args.out} "
          f"--out auroc_geom_af2.csv --sequence-csv results/sequence_v2.csv")


if __name__ == "__main__":
    main()
