"""Load HLA pseudosequences for the sequence model.

The NN represents each allele by its NetMHCpan-style 34-residue MHC pseudosequence
(the polymorphic, peptide-contacting positions). That mapping is **not** fabricated
here — it ships with NetMHCpan-4.1 as a two-column file (allele <whitespace>
pseudosequence), typically ``MHC_pseudo.dat`` / ``pseudosequence.dat`` in the data
directory of the install. Point ``load_pseudosequences`` at that file.

Alternatively, ``load_pseudosequences_json`` reads the MHC Motif Atlas JSON files
(one per locus), where each allele entry carries its 34-residue pocket
pseudosequence at ``entry["pocket_pseudosequence"]`` and its canonical name at
``entry["canonical_allele"]["protein_allele_name"]``.

Allele naming is a common footgun: NetMHCpan output uses ``HLA-A*02:01`` while many
peptide tables use ``HLA-A02:01`` or ``A0201``. ``normalize_allele`` collapses these
to one canonical key so lookups don't silently miss.
"""

from __future__ import annotations

import json
from pathlib import Path


def normalize_allele(allele: str) -> str:
    """Canonicalise an allele label to ``HLA-A*02:01`` style.

    Handles ``HLA-A02:01``, ``A*02:01``, ``A0201``, ``HLA-A*02:01`` → ``HLA-A*02:01``.
    Non-HLA / unrecognised inputs are returned uppercased and stripped, unchanged
    otherwise, so nothing is silently dropped.
    """
    a = allele.strip().upper().replace(" ", "")
    a = a.removeprefix("HLA-")
    # a is now like "A*02:01" | "A02:01" | "A0201" | "DRB1*15:01"
    if "*" not in a and len(a) >= 5 and a[0].isalpha():
        # insert the star after the locus letters (A/B/C, or DRB1 etc.)
        i = 0
        while i < len(a) and (a[i].isalpha() or a[i].isdigit() and not a[max(i - 1, 0)].isdigit()):
            # stop at the first digit that begins the allele group
            if a[i].isdigit():
                break
            i += 1
        a = a[:i] + "*" + a[i:]
    if ":" not in a and "*" in a:
        locus, rest = a.split("*", 1)
        digits = "".join(ch for ch in rest if ch.isdigit())
        if len(digits) >= 4:
            a = f"{locus}*{digits[:2]}:{digits[2:4]}"
    return f"HLA-{a}"


class PseudoSequenceMap:
    """Allele → 34-residue pseudosequence, with name normalisation on lookup."""

    def __init__(self, mapping: dict[str, str]):
        self._raw = dict(mapping)
        self._norm = {normalize_allele(k): v for k, v in mapping.items()}
        lengths = {len(v) for v in mapping.values()}
        self.pseudoseq_len = next(iter(lengths)) if len(lengths) == 1 else None

    def get(self, allele: str) -> str | None:
        return self._norm.get(normalize_allele(allele)) or self._raw.get(allele)

    def __contains__(self, allele: str) -> bool:
        return self.get(allele) is not None

    def __len__(self) -> int:
        return len(self._raw)


def load_pseudosequences(path: str | Path) -> PseudoSequenceMap:
    """Parse a two-column ``allele  pseudosequence`` file (NetMHCpan format)."""
    mapping: dict[str, str] = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 2:
            mapping[parts[0]] = parts[1]
    if not mapping:
        raise ValueError(f"No allele/pseudosequence pairs parsed from {path}")
    return PseudoSequenceMap(mapping)


def load_pseudosequences_json(paths: list[str | Path]) -> PseudoSequenceMap:
    """Load pseudosequences from MHC Motif Atlas JSON files (one per locus).

    Each file maps an allele key to an entry dict; the 34-residue pocket
    pseudosequence is at ``entry["pocket_pseudosequence"]``, and the canonical
    ``HLA-A*02:01``-style name is at
    ``entry["canonical_allele"]["protein_allele_name"]`` (falling back to the
    raw key if that field is absent).
    """
    mapping: dict[str, str] = {}
    for path in paths:
        data = json.loads(Path(path).read_text())
        for key, entry in data.items():
            pseudoseq = entry.get("pocket_pseudosequence")
            if not pseudoseq:
                continue
            name = (entry.get("canonical_allele") or {}).get("protein_allele_name") or key
            mapping[name] = pseudoseq
    if not mapping:
        raise ValueError(f"No pseudosequences parsed from {paths}")
    return PseudoSequenceMap(mapping)