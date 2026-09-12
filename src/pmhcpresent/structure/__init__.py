"""Structural feature extraction from folded peptide-MHC models (pLDDT, contacts,
ipSAE, shape). See `features.REFOLD_REQUIRED` for which need a re-fold per mutant."""
from pmhcpresent.structure.contacts import contact_features
from pmhcpresent.structure.features import (
    REFOLD_REQUIRED,
    StructureFeatures,
    extract_structure_features,
)
from pmhcpresent.structure.plddt import interface_plddt
