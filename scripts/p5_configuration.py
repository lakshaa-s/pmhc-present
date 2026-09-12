#!/usr/bin/env python3
"""Recompute the B*08:01 P5 configuration over the *measured* contact positions.

`results/p5_anchor_residues.csv` was built over positions 9/69/74/97, chosen before the
contacts were measured. Direct measurement in PDB 4QRT (see `scripts/p5_contacts.py`
and REPRODUCE.md) shows the polar contacts to the P5 side chain are Asp9 (2.78 A),
Ser97 (2.81 A) and Asp74 (2.86 A), with Tyr116 packing at 3.03 A; Thr69 is 5.74 A away
and is not a contact.

This script recomputes the configuration over 9/74/97, reports 116 alongside, and
answers the question row 16 of PROGRESS.md now leaves open: across all 123 alleles, how
many carry B*08:01's set?

    python scripts/p5_configuration.py \
        --pseudoseq data/pseudoseq/hla_a.json data/pseudoseq/hla_b.json data/pseudoseq/hla_c.json \
        --alleles results/per_allele_auroc_v3.csv \
        --anchors data/processed/anchors.json

--anchors is optional; supply it to get P5 information content per allele.

IMPORTANT: this maps pseudosequence index -> HLA position using the NetMHCpan-4.1
position list. That mapping is asserted, not derived, so the script verifies it against
HLA-B*08:01, whose residues at these positions are known from the crystal structures
(9 D, 69 T, 70 N, 74 D, 97 S, 116 Y). If the check fails the mapping is wrong for your
pseudosequence file and the script aborts rather than emitting numbers.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pmhcpresent.io.pseudoseq import (  # noqa: E402
    load_pseudosequences,
    load_pseudosequences_json,
    normalize_allele,
)

# NetMHCpan-4.1 pseudosequence positions, in order, 1-based on the mature heavy chain.
NETMHCPAN_POSITIONS = [7, 9, 24, 45, 59, 62, 63, 66, 67, 69, 70, 73, 74, 76, 77, 80,
                       81, 84, 95, 97, 99, 114, 116, 118, 143, 147, 150, 152, 156,
                       158, 159, 163, 167, 171]

# from 4QRT / 8ESH, and from the chimera substitution list in Papadaki et al. Table 1
B0801_KNOWN = {9: "D", 69: "T", 70: "N", 74: "D", 97: "S", 116: "Y"}

CONTACTS = [9, 74, 97]          # measured polar contacts to the P5 side chain
REPORTED = [9, 69, 70, 74, 97, 116]   # emitted for the record


def residue_at(pseudo: str, position: int) -> str:
    if len(pseudo) != len(NETMHCPAN_POSITIONS):
        raise SystemExit(
            f"pseudosequence length {len(pseudo)} != {len(NETMHCPAN_POSITIONS)} "
            "expected positions; this script assumes the NetMHCpan-4.1 34-mer")
    return pseudo[NETMHCPAN_POSITIONS.index(position)]


def verify_mapping(lookup) -> None:
    pseudo = lookup("HLA-B*08:01")
    if pseudo is None:
        raise SystemExit("HLA-B*08:01 absent from the pseudosequence file; cannot verify "
                         "the position mapping, so refusing to continue")
    observed = {p: residue_at(pseudo, p) for p in B0801_KNOWN}
    if observed != B0801_KNOWN:
        raise SystemExit(
            "position mapping check FAILED for HLA-B*08:01\n"
            f"  expected {B0801_KNOWN}\n  observed {observed}\n"
            "The pseudosequence position list does not match this file. Nothing written.")
    print("position mapping verified against HLA-B*08:01: "
          + ", ".join(f"{p}{r}" for p, r in sorted(observed.items())))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pseudoseq", required=True, nargs="+",
                    help="MHC_pseudo.dat, or Motif Atlas JSON files (one per locus)")
    ap.add_argument("--alleles", default="results/per_allele_auroc_v3.csv",
                    help="CSV with an 'allele' column defining the panel")
    ap.add_argument("--anchors", default=None,
                    help="optional anchors.json, for P5 information content")
    ap.add_argument("--out", default="results/p5_configuration_123.csv")
    args = ap.parse_args()

    paths = [Path(p) for p in args.pseudoseq]
    if all(p.suffix == ".json" for p in paths):
        pmap = load_pseudosequences_json(paths)
    else:
        pmap = load_pseudosequences(paths[0])

    lookup = pmap.get   # PseudoSequenceMap.get() normalises the allele name itself

    verify_mapping(lookup)

    with open(args.alleles) as fh:
        panel = [normalize_allele(r["allele"]) for r in csv.DictReader(fh)]
    print(f"panel: {len(panel)} alleles from {args.alleles}")

    p5_ic = {}
    if args.anchors:
        anchors = json.load(open(args.anchors))["alleles"]
        for allele, entry in anchors.items():
            ic = entry.get("position_ic") or entry.get("ic") or entry.get("ic_by_position")
            if isinstance(ic, dict):
                v = ic.get("5") or ic.get(5)
            elif isinstance(ic, (list, tuple)) and len(ic) >= 5:
                v = ic[4]
            else:
                v = None
            if v is not None:
                p5_ic[normalize_allele(allele)] = float(v)
        print(f"P5 information content available for {len(p5_ic)} alleles")

    target = tuple(B0801_KNOWN[p] for p in CONTACTS)
    rows, carriers, missing = [], [], []
    for allele in panel:
        pseudo = lookup(allele)
        if not pseudo:
            missing.append(allele)
            continue
        res = {p: residue_at(pseudo, p) for p in REPORTED}
        match = tuple(res[p] for p in CONTACTS) == target
        rows.append({"allele": allele,
                     **{f"pos{p}": res[p] for p in REPORTED},
                     "carries_b0801_contacts": int(match),
                     "p5_ic": f"{p5_ic[allele]:.2f}" if allele in p5_ic else ""})
        if match:
            carriers.append(allele)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\nwrote {out}  ({len(rows)} alleles"
          + (f", {len(missing)} without a pseudosequence: {missing}" if missing else "")
          + ")")
    print(f"\nB*08:01 contact set = "
          + "/".join(f"{aa}{p}" for p, aa in zip(CONTACTS, target))
          + f"\ncarriers across the panel: {len(carriers)}")
    for a in carriers:
        ic = f"  (P5 IC {p5_ic[a]:.2f})" if a in p5_ic else ""
        print(f"  {a}{ic}")

    print("\nFor comparison, the superseded set over 9/69/74/97:")
    old_target = tuple(B0801_KNOWN[p] for p in (9, 69, 74, 97))
    old = [r["allele"] for r in rows
           if tuple(r[f"pos{p}"] for p in (9, 69, 74, 97)) == old_target]
    print(f"  carriers: {len(old)}  {old}")

    if len(carriers) == 1:
        print("\n=> B*08:01 is unique across the panel on the measured contacts. "
              "The 'exactly one of 123' claim in thesis 4.4.2 can be restored.")
    else:
        print(f"\n=> {len(carriers)} carriers. The 'exactly one of 123' claim CANNOT be "
              "restored as written; 4.4.2 and PROGRESS row 16 need the new number.")


if __name__ == "__main__":
    main()