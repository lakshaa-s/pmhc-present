"""Collect Boltz structural features across all completed folds into one CSV.

Walks boltz-experiments/, and for each fold that has a completed prediction, pulls:
  - the confidence metrics from metrics.json (structure_confidence, iptm, complex_plddt,
    complex_iplddt, complex_pde/ipde, ...)
  - summary statistics of the PAE matrix (sample_0_pae.npz) — mean PAE overall and, if
    chain boundaries can be inferred, interface PAE (peptide<->MHC)
  - the allele slug and peptide parsed from the folder name (pdb__allele__peptide)

Output: one row per fold, ready to join against the sequence-model per-allele results.
The interface-confidence columns (iptm, complex_iplddt) are the structural analogue of
"does this peptide bind", i.e. the RQ1 features to compare against the sequence model.

Usage:  python extract_boltz_features.py [boltz-experiments dir] [--out features.csv]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# metric keys we expect inside metrics.json -> best_sample -> metrics
METRIC_KEYS = [
    "structure_confidence", "ptm", "iptm", "ligand_iptm", "protein_iptm",
    "complex_plddt", "complex_iplddt", "complex_pde", "complex_ipde",
]


def parse_folder_name(name: str) -> dict:
    """Folder names are '{pdb}__{allele_slug}__{peptide}'."""
    parts = name.split("__")
    if len(parts) == 3:
        pdb, allele_slug, peptide = parts
    else:  # be tolerant of odd names
        pdb = parts[0] if parts else name
        allele_slug = parts[1] if len(parts) > 1 else ""
        peptide = parts[-1] if len(parts) > 1 else ""
    # turn hla_b_27_09 -> HLA-B*27:09 for joining with sequence-side results
    allele = allele_slug
    if allele_slug.startswith(("hla_a_", "hla_b_", "hla_c_")):
        locus = allele_slug.split("_")[1].upper()
        nums = allele_slug.split("_")[2:]
        if len(nums) >= 2:
            allele = f"HLA-{locus}*{nums[0]}:{nums[1]}"
    return {"pdb": pdb, "allele_slug": allele_slug, "allele": allele, "peptide": peptide}


def pae_summary(pae_path: Path, peptide_len: int | None) -> dict:
    """Mean PAE overall, and interface PAE (peptide vs rest) if we can locate the peptide.

    The peptide is chain C, submitted last, so it occupies the final `peptide_len` rows/cols
    of the PAE matrix. Interface PAE = mean of the peptide-vs-MHC blocks (lower = more
    confident docking).
    """
    out = {"pae_mean": np.nan, "pae_interface": np.nan}
    try:
        data = np.load(pae_path)
        # npz: take the first array (key name may vary)
        key = list(data.keys())[0]
        pae = np.asarray(data[key], dtype=float)
        if pae.ndim == 3:  # (samples, L, L) -> take first sample
            pae = pae[0]
        out["pae_mean"] = float(pae.mean())
        if peptide_len and 0 < peptide_len < pae.shape[0]:
            n = pae.shape[0]
            pep = slice(n - peptide_len, n)
            mhc = slice(0, n - peptide_len)
            # both off-diagonal blocks (peptide->MHC and MHC->peptide)
            block = np.concatenate([pae[pep, mhc].ravel(), pae[mhc, pep].ravel()])
            out["pae_interface"] = float(block.mean())
    except Exception as e:
        out["pae_error"] = str(e)
    return out


def collect(root: Path) -> pd.DataFrame:
    rows = []
    for fold_dir in sorted(root.iterdir()):
        if not fold_dir.is_dir():
            continue
        pred = fold_dir / "outputs" / "files" / "prediction"
        metrics_path = pred / "metrics.json"
        if not metrics_path.exists():
            continue  # not finished / no output
        row = parse_folder_name(fold_dir.name)
        with open(metrics_path) as f:
            m = json.load(f)
        best = m.get("best_sample", {}).get("metrics", {})
        for k in METRIC_KEYS:
            row[k] = best.get(k, np.nan)
        # PAE summary (peptide is chain C -> last len(peptide) residues)
        pae_path = pred / "sample_0_pae.npz"
        if pae_path.exists():
            row.update(pae_summary(pae_path, len(row.get("peptide", "")) or None))
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", nargs="?", default="boltz-experiments",
                    help="boltz-experiments directory")
    ap.add_argument("--out", default="boltz_features.csv")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"{root} not found")

    df = collect(root)
    if df.empty:
        raise SystemExit("No completed folds found (no metrics.json under any fold dir).")

    df = df.sort_values("iptm", ascending=False)
    df.to_csv(args.out, index=False)

    print(f"Collected {len(df)} folds -> {args.out}\n")
    show = ["allele", "peptide", "iptm", "complex_iplddt", "complex_plddt",
            "pae_interface", "pae_mean"]
    show = [c for c in show if c in df.columns]
    print(df[show].to_string(index=False))
    print("\nKey RQ1 features: iptm / complex_iplddt (interface confidence), "
          "pae_interface (peptide-MHC docking error).")


if __name__ == "__main__":
    main()
