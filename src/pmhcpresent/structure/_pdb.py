"""Minimal PDB ATOM-record parser (numpy only, no Biopython needed).

AlphaFold writes per-atom pLDDT into the B-factor column, so the core structure
features only need ATOM coordinates + B-factors grouped by chain/residue. Keeping
this self-contained means ``plddt`` and ``contacts`` run with just numpy. For
mmCIF or anything heavier, fall back to Biopython (an optional dependency).

PDB column spec (1-indexed, fixed-width):
    record  1–6     atom name 13–16   chain 22    resSeq 23–26
    x 31–38   y 39–46   z 47–54   occupancy 55–60   bfactor 61–66
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Residue:
    chain: str
    resseq: int
    resname: str
    atom_names: list[str]
    coords: np.ndarray      # (n_atoms, 3)
    bfactors: np.ndarray    # (n_atoms,) — AlphaFold per-atom pLDDT lives here

    @property
    def mean_bfactor(self) -> float:
        return float(self.bfactors.mean())

    def ca(self) -> np.ndarray | None:
        if "CA" in self.atom_names:
            return self.coords[self.atom_names.index("CA")]
        return None


def parse_pdb(path: str | Path) -> list[Residue]:
    """Parse ATOM records into a list of Residue objects (HETATM ignored)."""
    by_key: dict[tuple[str, int], dict] = {}
    order: list[tuple[str, int]] = []

    for line in Path(path).read_text().splitlines():
        if not line.startswith("ATOM"):
            continue
        atom_name = line[12:16].strip()
        resname = line[17:20].strip()
        chain = line[21].strip() or "A"
        resseq = int(line[22:26])
        x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
        bfac = float(line[60:66]) if line[60:66].strip() else 0.0

        key = (chain, resseq)
        if key not in by_key:
            by_key[key] = {"resname": resname, "names": [], "coords": [], "b": []}
            order.append(key)
        by_key[key]["names"].append(atom_name)
        by_key[key]["coords"].append((x, y, z))
        by_key[key]["b"].append(bfac)

    residues = []
    for chain, resseq in order:
        d = by_key[(chain, resseq)]
        residues.append(
            Residue(
                chain=chain,
                resseq=resseq,
                resname=d["resname"],
                atom_names=d["names"],
                coords=np.asarray(d["coords"], dtype=float),
                bfactors=np.asarray(d["b"], dtype=float),
            )
        )
    return residues


def chains(residues: list[Residue]) -> dict[str, list[Residue]]:
    out: dict[str, list[Residue]] = {}
    for r in residues:
        out.setdefault(r.chain, []).append(r)
    return out


def min_interatomic_distance(a: Residue, b: Residue) -> float:
    """Minimum heavy-atom distance between two residues."""
    diff = a.coords[:, None, :] - b.coords[None, :, :]
    return float(np.sqrt((diff ** 2).sum(-1)).min())
