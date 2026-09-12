#!/usr/bin/env python3
"""Which HLA-B*08:01 residues contact the P5 side chain?

Measures minimum heavy-atom distances from the P5 peptide side chain to a set of
candidate groove residues, in any pMHC-I crystal structure.

    curl -O https://files.rcsb.org/download/4QRT.pdb   # wild-type B*08:01 / ELNRKMIYM
    curl -O https://files.rcsb.org/download/8ESH.pdb   # B*08:01->A*02:01 chimera / CMV
    python p5_contacts.py 4QRT.pdb
    python p5_contacts.py 8ESH.pdb

No dependencies beyond the standard library. PDB format is fixed-width, so the
column offsets below are the format spec, not guesses.

Read the output as follows. A salt bridge or hydrogen bond is roughly 2.5-3.5 A
between heavy atoms; anything beyond about 4.5 A is not a contact. The residue
names printed alongside each position are the numbering check: if position 9
comes back ASP, 70 comes back ASN and 74 comes back ASP, the file uses the same
mature-protein numbering as the thesis and the comparison is valid. If they come
back as something else, the numbering is offset and nothing below can be trusted.
"""
import csv
import math
import os
import sys
from collections import defaultdict

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results")

CANDIDATES = [9, 62, 63, 66, 67, 69, 70, 73, 74, 76, 77, 80, 97, 99, 114, 116]
BACKBONE = {"N", "CA", "C", "O", "OXT"}
P5 = 5


def parse(path):
    """chain -> resseq -> (resname, [(atomname, x, y, z), ...])"""
    chains = defaultdict(dict)
    with open(path) as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            altloc = line[16]
            if altloc not in (" ", "A"):
                continue
            name = line[12:16].strip()
            if name.startswith("H"):
                continue
            resname = line[17:20].strip()
            chain = line[21]
            try:
                resseq = int(line[22:26])
            except ValueError:
                continue
            xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            entry = chains[chain].setdefault(resseq, (resname, []))
            entry[1].append((name, *xyz))
    return chains


def dist(a, b):
    return math.sqrt(sum((p - q) ** 2 for p, q in zip(a[1:], b[1:])))


def main(path):
    chains = parse(path)
    lengths = {c: len(r) for c, r in chains.items()}
    print(f"{path}: chains " + ", ".join(f"{c}({n} residues)" for c, n in lengths.items()))

    # the peptide is the shortest chain; the heavy chain is the longest
    pep_chain = min(lengths, key=lambda c: lengths[c])
    hc_chain = max(lengths, key=lambda c: lengths[c])
    pep = chains[pep_chain]
    seq1 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
            "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
            "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
            "TYR": "Y", "VAL": "V"}
    order = sorted(pep)
    print(f"peptide  = chain {pep_chain}, "
          + "".join(seq1.get(pep[r][0], "X") for r in order)
          + f"  (residues {order[0]}-{order[-1]})")
    print(f"heavy chain = chain {hc_chain}\n")

    if P5 not in pep:
        sys.exit(f"no residue {P5} in the peptide chain")
    p5_name, p5_atoms = pep[P5]
    side = [a for a in p5_atoms if a[0] not in BACKBONE]
    if not side:
        sys.exit(f"P5 ({p5_name}) has no side-chain atoms modelled")
    print(f"P5 is {p5_name}, {len(side)} side-chain atoms modelled\n")

    rows = []
    for pos in CANDIDATES:
        res = chains[hc_chain].get(pos)
        if res is None:
            rows.append((pos, "----", None, ""))
            continue
        name, atoms = res
        best, pair = min(
            ((dist(s, h), f"{s[0]}-{h[0]}") for s in side for h in atoms),
            key=lambda t: t[0])
        rows.append((pos, name, best, pair))

    print(f"{'pos':>5}  {'res':<5} {'min dist':>9}  atoms        verdict")
    print("-" * 58)
    for pos, name, d, pair in sorted(rows, key=lambda r: (r[2] is None, r[2])):
        if d is None:
            print(f"{pos:>5}  {name:<5} {'absent':>9}")
            continue
        verdict = ("CONTACT (H-bond / salt bridge range)" if d <= 3.5
                   else "close, not a bond" if d <= 4.5
                   else "")
        print(f"{pos:>5}  {name:<5} {d:9.2f}  {pair:<12} {verdict}")

    stem = os.path.splitext(os.path.basename(path))[0].lower()
    os.makedirs(OUT, exist_ok=True)
    dest = os.path.join(OUT, f"p5_contacts_{stem}.csv")
    with open(dest, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["structure", "position", "residue", "min_heavy_atom_dist_A",
                    "atom_pair", "contact_le_3.5A"])
        for pos, name, d, pair in sorted(rows, key=lambda r: (r[2] is None, r[2])):
            w.writerow([stem.upper(), pos, name,
                        "" if d is None else f"{d:.2f}", pair,
                        "" if d is None else int(d <= 3.5)])
    print(f"\nwrote {dest}")

    print("\nThesis claim: Asp9, Asn70, Asp74 are the P5 contacts.")
    print("Earlier claim: Asp9, Thr69, Asp74, Ser97.")
    print("Whichever set falls under 3.5 A is the one to keep.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "4QRT.pdb")