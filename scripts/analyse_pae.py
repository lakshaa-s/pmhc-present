"""Per-residue PAE analysis of pMHC folds: does interface error discriminate binders?

iptm (a single number) didn't separate binders from decoys. This looks deeper: the PAE
matrix gives predicted alignment error for every residue PAIR. Binding is driven by ANCHOR
residues (peptide position 2 and the C-terminus) locking into the MHC B- and F-pockets.
So the sharp question is: for real binders, is PAE LOW at the anchor positions (Boltz is
confident those key residues sit correctly), while for decoys the anchors are UNCERTAIN
(high PAE) because they don't fit their pockets?

The peptide is chain C, submitted last, so it is the final `len(peptide)` rows/cols of
the PAE matrix. We report, per fold:
  - pae_pep_mhc      : mean PAE between peptide residues and all MHC residues (interface)
  - pae_anchor2      : mean PAE of peptide position 2 vs MHC (B-pocket anchor)
  - pae_anchorC      : mean PAE of the C-terminal residue vs MHC (F-pocket anchor)
  - pae_anchors      : mean of the two anchors (the key binding signal)

Then, per allele, it prints binder vs decoy anchor-PAE so you can see if a gap exists.
Reads boltz_features-style folder names: '{tag}__{allele_slug}__{peptide}'.
'decoy' tag => decoy; anything else => binder.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def load_pae(path):
    d = np.load(path)
    pae = np.asarray(d[list(d.keys())[0]], dtype=float)
    if pae.ndim == 3:
        pae = pae[0]
    return pae


def parse_name(name):
    parts = name.split("__")
    tag = parts[0]
    allele_slug = parts[1] if len(parts) > 1 else ""
    peptide = parts[-1] if len(parts) > 1 else ""
    allele = allele_slug
    if allele_slug.startswith(("hla_a_", "hla_b_", "hla_c_")):
        b = allele_slug.split("_")
        allele = f"HLA-{b[1].upper()}*{b[2]}:{b[3]}"
    is_decoy = tag == "decoy"
    return allele, peptide, is_decoy


def fold_features(pae, peptide_len):
    n = pae.shape[0]
    if not (0 < peptide_len < n):
        return {}
    pep = slice(n - peptide_len, n)          # peptide = last residues
    mhc = slice(0, n - peptide_len)
    pep_idx = np.arange(n - peptide_len, n)
    # interface = both off-diagonal blocks
    interface = np.concatenate([pae[pep, mhc].ravel(), pae[mhc, pep].ravel()])
    # anchor positions within the peptide: P2 (index 1) and C-terminus (last)
    p2 = pep_idx[1] if peptide_len >= 2 else pep_idx[0]
    pc = pep_idx[-1]
    def res_vs_mhc(i):
        return np.concatenate([pae[i, mhc].ravel(), pae[mhc, i].ravel()]).mean()
    a2, ac = res_vs_mhc(p2), res_vs_mhc(pc)
    return {
        "pae_pep_mhc": float(interface.mean()),
        "pae_anchor2": float(a2),
        "pae_anchorC": float(ac),
        "pae_anchors": float((a2 + ac) / 2),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", nargs="?", default="boltz-experiments")
    ap.add_argument("--out", default="pae_analysis.csv")
    args = ap.parse_args()

    rows = []
    for fold in sorted(Path(args.root).iterdir()):
        pae_path = fold / "outputs" / "files" / "prediction" / "sample_0_pae.npz"
        if not pae_path.exists():
            continue
        allele, peptide, is_decoy = parse_name(fold.name)
        feats = fold_features(load_pae(pae_path), len(peptide))
        if not feats:
            continue
        rows.append({"allele": allele, "peptide": peptide,
                     "kind": "decoy" if is_decoy else "binder", **feats})

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)

    pd.set_option("display.width", 160)
    print("Per-fold anchor PAE (lower = more confident placement):\n")
    print(df.sort_values(["allele", "kind"])[
        ["allele", "kind", "peptide", "pae_anchors", "pae_pep_mhc"]
    ].to_string(index=False))

    print("\n=== binder vs decoy anchor-PAE, per allele ===")
    for allele, sub in df.groupby("allele"):
        b = sub[sub.kind == "binder"]["pae_anchors"]
        d = sub[sub.kind == "decoy"]["pae_anchors"]
        if len(b) and len(d):
            gap = d.mean() - b.mean()
            flag = "  <-- binders lower (expected)" if gap > 0 else "  <-- NO separation"
            print(f"  {allele:14s}  binder {b.mean():.3f}  decoy {d.mean():.3f}  "
                  f"gap {gap:+.3f}{flag}")
        else:
            print(f"  {allele:14s}  (need both binder+decoy folds)")

    print("\nIf 'gap' is consistently positive (decoy PAE > binder PAE), anchor PAE "
          "discriminates where iptm did not. If gaps are ~0 or mixed, it does not.")


if __name__ == "__main__":
    main()
