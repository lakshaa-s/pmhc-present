"""Shape complementarity between peptide side chains and the MHC groove.

Shape complementarity (Sc; Lawrence & Colman 1993) measures how well two molecular
surfaces interdigitate. A faithful Sc needs a molecular-surface representation
(dot surface + nearest-neighbour normals), which is heavier than the contact/pLDDT
maths — hence this module is a **scaffold with a documented approximation**, not a
finished metric.

Two caveats for the writeup / expert Q&A:
  1. AlphaFold does not model ordered water filling groove gaps, so any complementarity
     computed from an AlphaFold model treats the groove as dry — it will read as more
     "complementary" than the solvated reality.
  2. Sc is sensitive to side-chain placement, so on a fixed backbone it is still
     recomputable after a mutation (refold_required = False), but only if you repack
     the mutated side chain rather than leaving it as the wild-type rotamer.

The placeholder below returns a buried-surface-area proxy via ``freesasa`` if it's
installed (ΔSASA on complex formation), which correlates with interface size but is
**not** Sc. Swap in a proper Sc (e.g. a Python port of CCP4's ``sc``) before relying
on this for RQ1/RQ2 conclusions.
"""
from __future__ import annotations

from pathlib import Path


def buried_surface_area(
    pdb_path: str | Path,
    peptide_chain: str,
    mhc_chain: str,
) -> dict:
    """ΔSASA on complex formation (interface buried area), as an interim proxy.

    BSA = SASA(peptide alone) + SASA(MHC alone) − SASA(complex). Requires ``freesasa``.
    Returns ``{'bsa': float}`` or raises a clear error if freesasa is missing.
    """
    try:
        import freesasa  # noqa: F401
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "buried_surface_area needs freesasa (`pip install freesasa`). This is an "
            "interim proxy for shape complementarity — see module docstring."
        ) from e

    import freesasa
    from pmhcpresent.structure._pdb import parse_pdb, chains  # noqa: F401

    structure = freesasa.Structure(str(pdb_path))
    complex_area = freesasa.calc(structure).totalArea()

    # Per-chain SASA via freesasa selection. (A full implementation writes single-chain
    # PDBs and recomputes; kept explicit here as the TODO it is.)
    raise NotImplementedError(
        "BSA per-chain decomposition not implemented yet — write single-chain PDBs and "
        "recompute SASA, or port CCP4 `sc` for true shape complementarity. "
        f"(complex total SASA computed = {complex_area:.1f} A^2)"
    )


def shape_complementarity_stub(*_args, **_kwargs) -> dict:
    """Explicit placeholder so the feature aggregator has a stable signature."""
    return {"shape_complementarity": float("nan"), "_implemented": False}
