"""Parse NetMHCpan-4.1 output.

NetMHCpan-4.1 stdout is a whitespace-aligned block bounded by dashed separator
lines. The exact columns depend on the flags used (``-BA`` adds binding-affinity
columns), so this parser is **header-driven**: it locates the header row, uses its
tokens as keys, and maps each data row onto them. The trailing ``<= SB`` / ``<= WB``
BindLevel annotation (present only for binders) is captured separately.

The tool itself runs on Beta; this parser is developed and tested locally against
saved example output (the output is just text).

Alternative: ``netMHCpan -xls -xlsfile out.xls ...`` produces tab-separated output
that ``pandas.read_csv(sep='\\t', skiprows=1)`` reads directly. Use that path for
large runs; this parser exists for the human-readable stdout you get by default.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

# Tokens that, if all present on a line, mark the column header row.
_HEADER_MARKERS = ("Pos", "MHC", "Peptide")
_BIND_TOKENS = {"SB", "WB"}


@dataclass
class NetMHCpanRecord:
    """One peptide–allele prediction row, with raw fields preserved."""
    allele: str
    peptide: str
    pos: int | None = None
    score_el: float | None = None
    rank_el: float | None = None
    score_ba: float | None = None
    rank_ba: float | None = None
    aff_nm: float | None = None
    bind_level: str | None = None          # "SB", "WB", or None
    raw: dict = field(default_factory=dict)

    @property
    def is_binder(self) -> bool:
        return self.bind_level in _BIND_TOKENS


def _coerce(value: str) -> float | int | str:
    for caster in (int, float):
        try:
            return caster(value)
        except ValueError:
            continue
    return value


def _find_header(lines: list[str]) -> tuple[int, list[str]]:
    for i, line in enumerate(lines):
        toks = line.split()
        if all(m in toks for m in _HEADER_MARKERS):
            return i, toks
    raise ValueError(
        "No NetMHCpan header row found (expected a line containing "
        f"{_HEADER_MARKERS}). Is this NetMHCpan-4.1 stdout?"
    )


def _is_separator(line: str) -> bool:
    s = line.strip()
    return bool(s) and set(s) <= set("-")


def parse_netmhcpan_text(text: str) -> list[NetMHCpanRecord]:
    """Parse a NetMHCpan-4.1 stdout string into records."""
    lines = text.splitlines()
    header_idx, header = _find_header(lines)

    # BindLevel is an OPTIONAL trailing column: it appears as "<= SB" / "<= WB" only on
    # binder rows and is absent entirely on non-binders. So map the *core* columns
    # (everything except a trailing BindLevel header) and recover bind level separately.
    core_header = header[:-1] if header and header[-1] == "BindLevel" else header
    n_core = len(core_header)

    records: list[NetMHCpanRecord] = []
    for line in lines[header_idx + 1:]:
        if _is_separator(line):
            continue
        toks = line.split()
        if not toks:
            continue
        # End of the prediction block: NetMHCpan prints a "Protein ..." summary line.
        if toks[0].startswith("Protein") or toks[0].startswith("Number"):
            break

        # Strip the optional bind-level annotation ("<=", "SB"/"WB") from the row.
        bind_level = None
        core_toks = []
        for tok in toks:
            if tok in _BIND_TOKENS:
                bind_level = tok
            elif tok != "<=":
                core_toks.append(tok)

        if len(core_toks) != n_core:
            continue   # not a data row (or an unexpected format)

        row = dict(zip(core_header, core_toks))
        raw = {k: _coerce(v) for k, v in row.items()}
        records.append(
            NetMHCpanRecord(
                allele=str(row.get("MHC", "")),
                peptide=str(row.get("Peptide", "")),
                pos=raw.get("Pos") if isinstance(raw.get("Pos"), int) else None,
                score_el=_as_float(row.get("Score_EL") or row.get("Score")),
                rank_el=_as_float(row.get("%Rank_EL") or row.get("%Rank")),
                score_ba=_as_float(row.get("Score_BA")),
                rank_ba=_as_float(row.get("%Rank_BA")),
                aff_nm=_as_float(row.get("Aff(nM)") or row.get("Aff(nM)_BA")),
                bind_level=bind_level,
                raw=raw,
            )
        )
    return records


def _as_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_netmhcpan_file(path: str | Path) -> list[NetMHCpanRecord]:
    return parse_netmhcpan_text(Path(path).read_text())


def records_to_frame(records: Iterable[NetMHCpanRecord]):
    """Flatten records to a pandas DataFrame (one row per prediction)."""
    import pandas as pd

    rows = []
    for r in records:
        d = {
            "allele": r.allele,
            "peptide": r.peptide,
            "pos": r.pos,
            "score_el": r.score_el,
            "rank_el": r.rank_el,
            "score_ba": r.score_ba,
            "rank_ba": r.rank_ba,
            "aff_nm": r.aff_nm,
            "bind_level": r.bind_level,
            "is_binder": r.is_binder,
        }
        rows.append(d)
    return pd.DataFrame(rows)
