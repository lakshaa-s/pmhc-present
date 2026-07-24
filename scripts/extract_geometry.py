"""Geometric interface features from Boltz pMHC structures — the last structural signal.

Confidence (iptm) and per-residue PAE both failed to discriminate binders from decoys.
This tests the GEOMETRY itself: does a real binder make more/tighter contacts with the
MHC groove than a forced decoy, even when Boltz was equally confident placing both?

Per fold, from the predicted structure (.cif):
  - n_contacts        : peptide-MHC atom pairs within CONTACT_CUTOFF (heavy atoms)
  - n_contacts_close  : pairs within a tighter cutoff (tight packing)
  - contacts_per_res  : n_contacts normalised by peptide length
  - anchor2_contacts  : contacts made by peptide position 2 (B-pocket anchor)
  - anchorC_contacts  : contacts made by the peptide C-terminal residue (F-pocket anchor)
  - min_anchor_dist2  : closest peptide-P2 -> MHC heavy-atom distance (low = anchor seated)
  - min_anchor_distC  : closest C-terminus -> MHC distance

Chains: A = MHC heavy chain, B = beta-2-microglobulin, C = peptide (as submitted).
The MHC "groove" contacts are peptide(C) vs MHC(A); b2m(B) is ignored for interface.

Then, per allele, prints binder vs decoy contact counts so you can see if geometry
discriminates where confidence did not. Folder names: '{tag}__{allele_slug}__{peptide}',
tag 'decoy' => decoy, else binder.

Needs: biotite (`pip install biotite --break-system-packages`).
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

import biotite.structure.io.pdbx as pdbx

CONTACT_CUTOFF = 4.5      # Angstrom, heavy-atom contact
CLOSE_CUTOFF = 3.5        # tighter packing


def load_structure(cif_path):
    f = pdbx.CIFFile.read(str(cif_path))
    arr = pdbx.get_structure(f, model=1)
    return arr[arr.element != "H"]  # heavy atoms only


def chain_atoms(arr, chain_id):
    return arr[arr.chain_id == chain_id]


def parse_name(name):
    parts = name.split("__")
    tag = parts[0]
    allele_slug = parts[1] if len(parts) > 1 else ""
    peptide = parts[-1] if len(parts) > 1 else ""
    allele = allele_slug
    if allele_slug.startswith(("hla_a_", "hla_b_", "hla_c_")):
        b = allele_slug.split("_")
        allele = f"HLA-{b[1].upper()}*{b[2]}:{b[3]}"
    return allele, peptide, (tag == "decoy")


def interface_features(arr, peptide_len):
    # peptide is chain C; MHC groove is chain A
    pep = chain_atoms(arr, "C")
    mhc = chain_atoms(arr, "A")
    if pep.array_length() == 0 or mhc.array_length() == 0:
        return None

    pc = pep.coord            # (Npep_atoms, 3)
    mc = mhc.coord            # (Nmhc_atoms, 3)
    # pairwise distances peptide-atom x mhc-atom
    d = np.linalg.norm(pc[:, None, :] - mc[None, :, :], axis=-1)  # (Npep, Nmhc)

    n_contacts = int((d < CONTACT_CUTOFF).sum())
    n_close = int((d < CLOSE_CUTOFF).sum())

    # per-residue: map peptide atoms -> residue id, find P2 and C-terminal residue
    res_ids = pep.res_id
    uniq_res = np.unique(res_ids)
    uniq_res_sorted = np.sort(uniq_res)
    # position 2 = second residue, C-term = last
    p2_res = uniq_res_sorted[1] if len(uniq_res_sorted) >= 2 else uniq_res_sorted[0]
    pc_res = uniq_res_sorted[-1]

    def res_contacts(rid):
        mask = res_ids == rid
        dd = d[mask, :]
        return int((dd < CONTACT_CUTOFF).sum()), float(dd.min()) if dd.size else np.nan

    a2_c, a2_min = res_contacts(p2_res)
    ac_c, ac_min = res_contacts(pc_res)

    return {
        "n_contacts": n_contacts,
        "n_contacts_close": n_close,
        "contacts_per_res": n_contacts / max(peptide_len, 1),
        "anchor2_contacts": a2_c,
        "anchorC_contacts": ac_c,
        "min_anchor_dist2": a2_min,
        "min_anchor_distC": ac_min,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", nargs="?", default="boltz-experiments")
    ap.add_argument("--out", default="geometry_features.csv")
    args = ap.parse_args()

    rows = []
    for fold in sorted(Path(args.root).iterdir()):
        cif = fold / "outputs" / "files" / "prediction" / "sample_0_predicted_structure.cif"
        if not cif.exists():
            continue
        allele, peptide, is_decoy = parse_name(fold.name)
        try:
            arr = load_structure(cif)
            feats = interface_features(arr, len(peptide))
        except Exception as e:
            print(f"  error on {fold.name}: {e}")
            continue
        if feats is None:
            print(f"  no A/C chains in {fold.name}")
            continue
        rows.append({"allele": allele, "peptide": peptide,
                     "kind": "decoy" if is_decoy else "binder", **feats})

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)

    pd.set_option("display.width", 180)
    print("Per-fold interface geometry:\n")
    cols = ["allele", "kind", "peptide", "n_contacts", "contacts_per_res",
            "anchor2_contacts", "anchorC_contacts"]
    print(df.sort_values(["allele", "kind"])[cols].to_string(index=False))

    print("\n=== binder vs decoy contacts, per allele ===")
    for allele, sub in df.groupby("allele"):
        b = sub[sub.kind == "binder"]["n_contacts"]
        d = sub[sub.kind == "decoy"]["n_contacts"]
        if len(b) and len(d):
            gap = b.mean() - d.mean()
            flag = "  <-- binders MORE contacts (expected)" if gap > 0 else "  <-- NO separation"
            print(f"  {allele:14s}  binder {b.mean():6.1f}  decoy {d.mean():6.1f}  "
                  f"gap {gap:+.1f}{flag}")
        else:
            print(f"  {allele:14s}  (need both binder+decoy folds)")

    print("\nIf binders consistently make MORE contacts than decoys, geometry discriminates "
          "where confidence did not. If gaps are ~0 or mixed, it does not.")


if __name__ == "__main__":
    main()