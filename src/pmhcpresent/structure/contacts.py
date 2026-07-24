"""Peptide–MHC contact maps.

A contact map records which peptide residues touch which MHC residues (minimum
heavy-atom distance below a cutoff). Unlike pLDDT/PAE/ipSAE, contacts can be
**recomputed on a fixed backbone** — for RQ3 saturation mutagenesis you can mutate a
side chain and recompute contacts without re-folding. The structure module marks
this feature ``refold_required = False``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from pmhcpresent.structure._pdb import Residue, chains, min_interatomic_distance, parse_pdb


def contact_matrix(
    peptide: list[Residue], mhc: list[Residue], cutoff: float = 4.0
) -> np.ndarray:
    """(n_peptide, n_mhc) float matrix of minimum heavy-atom distances."""
    out = np.empty((len(peptide), len(mhc)), dtype=float)
    for i, p in enumerate(peptide):
        for j, m in enumerate(mhc):
            out[i, j] = min_interatomic_distance(p, m)
    return out


def contact_features(
    pdb_path: str | Path,
    peptide_chain: str,
    mhc_chain: str,
    cutoff: float = 4.0,
) -> dict:
    """Summarise the contact map into scalar features.

    Returns counts and per-peptide-residue contact degree, plus the raw distance
    matrix (under ``_distance_matrix``) for callers that want it.
    """
    residues = parse_pdb(pdb_path)
    by_chain = chains(residues)
    pep = by_chain[peptide_chain]
    mhc = by_chain[mhc_chain]

    dist = contact_matrix(pep, mhc, cutoff)
    contacts = dist <= cutoff
    return {
        "n_contacts": int(contacts.sum()),
        "n_contacting_pep_residues": int(contacts.any(axis=1).sum()),
        "n_contacting_mhc_residues": int(contacts.any(axis=0).sum()),
        "mean_contact_degree": float(contacts.sum(axis=1).mean()),
        "min_pep_mhc_distance": float(dist.min()),
        "_distance_matrix": dist,
    }
