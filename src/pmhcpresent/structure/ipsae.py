"""Wrapper around the Dunbrack group's ipSAE tool (2025).

ipSAE is an interface-specific, PAE-derived score that outperforms ipTM for short
peptide chains — which is exactly the peptide–MHC regime here. The tool is a separate
script (set ``paths.ipsae_script`` in the config); this module shells out to it and
parses the result. It needs the AlphaFold PAE JSON and the model file, so it is a
**refold-required** feature for RQ3.

This is a thin wrapper, not a reimplementation. Point ``ipsae_script`` at the cloned
tool on Beta. The exact CLI/parsing below is intentionally defensive — confirm the
flag names and output format against the installed version before trusting the values.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path


def run_ipsae(
    ipsae_script: str | Path,
    pae_json: str | Path,
    model_path: str | Path,
    pae_cutoff: float = 10.0,
    dist_cutoff: float = 10.0,
    timeout: int = 600,
) -> dict:
    """Run the ipSAE script and return parsed scores.

    Args mirror the tool's typical interface: a PAE JSON, the predicted model
    (PDB/CIF), and PAE/distance cutoffs. Returns a dict of parsed fields plus the
    raw stdout under ``_stdout`` for debugging.
    """
    ipsae_script = Path(ipsae_script)
    if not ipsae_script.exists():
        raise FileNotFoundError(
            f"ipSAE script not found at {ipsae_script}. Clone the Dunbrack 2025 tool "
            "and set paths.ipsae_script in configs."
        )

    cmd = [
        "python", str(ipsae_script),
        str(pae_json), str(model_path),
        str(pae_cutoff), str(dist_cutoff),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"ipSAE failed (exit {proc.returncode}):\n{proc.stderr}")

    return {**_parse_ipsae_stdout(proc.stdout), "_stdout": proc.stdout}


def _parse_ipsae_stdout(text: str) -> dict:
    """Best-effort parse of ipSAE output.

    The tool writes a per-chain-pair table; values vary by version. This pulls the
    max ipSAE it can find as a scalar summary. Confirm against your installed version
    and tighten this parser once the exact format is fixed.
    """
    best = None
    for line in text.splitlines():
        toks = line.split()
        for tok in toks:
            try:
                val = float(tok)
            except ValueError:
                continue
            if 0.0 <= val <= 1.0:
                best = val if best is None else max(best, val)
    return {"ipsae": best}


def parse_pae_json(pae_json: str | Path) -> dict:
    """Light sanity-check / loader for an AlphaFold PAE JSON."""
    data = json.loads(Path(pae_json).read_text())
    if isinstance(data, list):
        data = data[0]
    return data
