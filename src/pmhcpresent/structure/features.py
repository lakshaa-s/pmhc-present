"""Aggregate AlphaFold-derived structural features for one peptide–MHC model.

The key design decision for RQ3 lives here: every feature is tagged with whether it
needs a **re-fold** when a residue is mutated, or whether it can be recomputed on a
**fixed wild-type backbone**. The saturation-mutagenesis scorer uses these tags to
avoid re-folding for features that don't need it.

    refold_required = True   →  interface pLDDT, PAE/ipSAE   (depend on the fold)
    refold_required = False  →  contact map, shape compl.    (fixed-backbone OK)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from pmhcpresent.structure.contacts import contact_features
from pmhcpresent.structure.ipsae import run_ipsae
from pmhcpresent.structure.plddt import interface_plddt
from pmhcpresent.structure.shape import shape_complementarity_stub

# Which feature groups need a re-fold per mutant (RQ3 cost model).
REFOLD_REQUIRED = {
    "interface_plddt": True,
    "peptide_plddt": True,
    "ipsae": True,
    "n_contacts": False,
    "mean_contact_degree": False,
    "min_pep_mhc_distance": False,
    "shape_complementarity": False,
}


@dataclass
class StructureFeatures:
    features: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def fixed_backbone_subset(self) -> dict:
        """Features recomputable without re-folding (for cheap mutagenesis scoring)."""
        return {k: v for k, v in self.features.items() if REFOLD_REQUIRED.get(k) is False}

    def refold_subset(self) -> dict:
        return {k: v for k, v in self.features.items() if REFOLD_REQUIRED.get(k) is True}


def extract_structure_features(
    pdb_path: str | Path,
    peptide_chain: str,
    mhc_chain: str,
    *,
    pae_json: str | Path | None = None,
    ipsae_script: str | Path | None = None,
    interface_cutoff: float = 5.0,
    contact_cutoff: float = 4.0,
) -> StructureFeatures:
    """Run all available extractors over one folded peptide–MHC model.

    ipSAE is only computed if both ``pae_json`` and ``ipsae_script`` are given.
    Shape complementarity is a documented placeholder until a real Sc is wired in.
    """
    feats: dict = {}

    feats.update(interface_plddt(pdb_path, peptide_chain, mhc_chain, interface_cutoff))

    c = contact_features(pdb_path, peptide_chain, mhc_chain, contact_cutoff)
    c.pop("_distance_matrix", None)   # keep the scalar summary in the feature dict
    feats.update(c)

    feats.update(shape_complementarity_stub())

    if pae_json is not None and ipsae_script is not None:
        feats.update(run_ipsae(ipsae_script, pae_json, pdb_path))

    return StructureFeatures(
        features=feats,
        meta={
            "pdb_path": str(pdb_path),
            "peptide_chain": peptide_chain,
            "mhc_chain": mhc_chain,
            "ipsae_computed": pae_json is not None and ipsae_script is not None,
        },
    )
