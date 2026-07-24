"""Interface pLDDT from an AlphaFold PDB.

AlphaFold stores per-atom pLDDT in the B-factor column. The *interface* pLDDT is the
mean pLDDT over peptide residues that sit at the peptide–MHC interface (any heavy
atom within ``cutoff`` of the MHC chain). This is one of the refold-required features
in RQ3 — it depends on the folded model, so each mutant needs its own fold.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from pmhcpresent.structure._pdb import Residue, chains, min_interatomic_distance, parse_pdb


def interface_residues(
    peptide: list[Residue], mhc: list[Residue], cutoff: float
) -> list[Residue]:
    """Peptide residues with any heavy atom within ``cutoff`` Å of any MHC residue."""
    out = []
    for p in peptide:
        if any(min_interatomic_distance(p, m) <= cutoff for m in mhc):
            out.append(p)
    return out


def interface_plddt(
    pdb_path: str | Path,
    peptide_chain: str,
    mhc_chain: str,
    interface_cutoff: float = 5.0,
) -> dict:
    """Return interface and whole-peptide pLDDT summaries.

    Keys: ``interface_plddt`` (mean over interface residues), ``peptide_plddt``
    (mean over all peptide residues), ``n_interface``, ``n_peptide``.
    """
    residues = parse_pdb(pdb_path)
    by_chain = chains(residues)
    if peptide_chain not in by_chain or mhc_chain not in by_chain:
        raise KeyError(
            f"Chains {peptide_chain!r}/{mhc_chain!r} not found; present: {list(by_chain)}"
        )
    pep = by_chain[peptide_chain]
    mhc = by_chain[mhc_chain]

    iface = interface_residues(pep, mhc, interface_cutoff)
    pep_vals = np.array([r.mean_bfactor for r in pep])
    iface_vals = np.array([r.mean_bfactor for r in iface]) if iface else np.array([])

    return {
        "interface_plddt": float(iface_vals.mean()) if iface_vals.size else float("nan"),
        "peptide_plddt": float(pep_vals.mean()) if pep_vals.size else float("nan"),
        "n_interface": len(iface),
        "n_peptide": len(pep),
    }
