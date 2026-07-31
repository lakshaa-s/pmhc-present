"""Score fold set v2 with NetMHCpan-4.1 as a third sequence baseline.

Why this exists
---------------
Our own model gives 0.921 pooled on fold set v2 with HLA-C in the bottom three.
MHCflurry's presentation predictor reproduces that ordering (0.841 pooled); its
affinity predictor does not. NetMHCpan is the field standard and its eluted-ligand
output sits on the presentation side, so it is a third test of whether the HLA-C
deficit is a property of the approach or of one implementation.

Two things to know when interpreting the result
-----------------------------------------------
TRAINING DATA   NetMHCpan-4.1's training set is not public, so the in_train column
                cannot be filled the way it was for MHCflurry (121/144 overlap,
                AUROC 0.824 seen vs 0.937 unseen). Overlap here is likely and
                unquantifiable.

CIRCULARITY     NetMHCpan-4.1 trains on public immunopeptidomics, which is also
                what feeds the MHC Motif Atlas that our fold set was *selected*
                from by PWM score. MixMHCpred scored 0.999 on this fold set for
                exactly that reason (PWM score alone gives AUROC 1.000). Run the
                PWM-correlation check before treating a high number as meaningful.

Allele coverage
---------------
NetMHCpan reports "Distance to training data" and the nearest neighbour used, per
allele. On our panel it substitutes C*03:03 for C*03:04 (distance 0.000) and
C*16:01 for C*16:02 (distance 0.047) -- the same C*03:04 substitution MixMHCpred
makes. Those lines are captured and reported here, since an allele being scored by
proxy matters for a project about coverage. Note the distance is in pseudosequence
space, and pseudosequence distance does not predict per-allele performance in our
data (rho -0.021) whereas motif distance does (-0.363).

Score column: %Rank_EL, the eluted-ligand percentile. Lower means a better binder,
so it is negated to keep "higher = binder" consistent with the other baselines.

Usage:
    python scripts/score_netmhcpan.py \
        --fold-set fold_sets/fold_set_v2.csv \
        --out results/netmhcpan_v2.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pmhcpresent.io.netmhcpan import parse_netmhcpan_text, records_to_frame  # noqa: E402

RANK_COLS = ("%Rank_EL", "Rank_EL", "%Rank", "rank_el", "pct_rank_el")
PEP_COLS = ("Peptide", "peptide")


def slug_to_allele(slug: str) -> str:
    b = slug.split("_")
    return f"HLA-{b[1].upper()}*{b[2]}:{b[3]}" if len(b) >= 4 else slug


def slug_to_netmhc(slug: str) -> str:
    """hla_c_03_04 -> HLA-C03:04 (NetMHCpan drops the asterisk)."""
    b = slug.split("_")
    return f"HLA-{b[1].upper()}{b[2]}:{b[3]}"


def load_fold_set(path: str) -> list[dict]:
    rows = []
    with open(path) as fh:
        for row in csv.reader(fh):
            if len(row) < 4:
                continue
            tag, _locus, slug, peptide = row[0], row[1], row[2], row[3]
            rows.append({"slug": slug, "allele": slug_to_allele(slug),
                         "netmhc_allele": slug_to_netmhc(slug),
                         "peptide": peptide,
                         "label": 0 if tag in ("decoy", "hard") else 1})
    return rows


def run_netmhcpan(peptides: list[str], allele: str, binary: str) -> str:
    with tempfile.NamedTemporaryFile("w", suffix=".pep", delete=False) as fh:
        fh.write("\n".join(peptides) + "\n")
        pep_path = fh.name
    try:
        res = subprocess.run([binary, "-p", pep_path, "-a", allele, "-BA"],
                             capture_output=True, text=True, check=False)
        if res.returncode != 0 and not res.stdout.strip():
            raise SystemExit(f"netMHCpan failed for {allele}:\n{res.stderr[:500]}")
        return res.stdout
    finally:
        Path(pep_path).unlink(missing_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold-set", default="fold_sets/fold_set_v2.csv")
    ap.add_argument("--binary", default="netMHCpan")
    ap.add_argument("--out", default="results/netmhcpan_v2.csv")
    ap.add_argument("--raw-dir", help="optional dir to keep raw netMHCpan stdout")
    args = ap.parse_args()

    fs = load_fold_set(args.fold_set)
    df_fs = pd.DataFrame(fs)
    print(f"{len(df_fs)} complexes, {int(df_fs.label.sum())} binders / "
          f"{int((1 - df_fs.label).sum())} decoys, "
          f"{df_fs.allele.nunique()} alleles\n")

    if args.raw_dir:
        Path(args.raw_dir).mkdir(parents=True, exist_ok=True)

    out_rows, coverage = [], []
    for (allele, nm_allele), g in df_fs.groupby(["allele", "netmhc_allele"]):
        peptides = list(g.peptide)
        text = run_netmhcpan(peptides, nm_allele, args.binary)
        if args.raw_dir:
            (Path(args.raw_dir) / f"{nm_allele.replace(':', '')}.txt").write_text(text)

        for line in text.splitlines():
            m = re.search(r"Distance to training data\s+([\d.]+).*?"
                          r"nearest neighbor\s+(\S+?)\)", line)
            if m:
                coverage.append((allele, float(m.group(1)), m.group(2)))

        recs = parse_netmhcpan_text(text)
        frame = records_to_frame(recs)
        if frame is None or len(frame) == 0:
            raise SystemExit(f"No records parsed for {allele}")

        pcol = next((c for c in PEP_COLS if c in frame.columns), None)
        rcol = next((c for c in RANK_COLS if c in frame.columns), None)
        if pcol is None or rcol is None:
            raise SystemExit(
                f"Could not find peptide/rank columns for {allele}.\n"
                f"Available: {list(frame.columns)}")

        lookup = dict(zip(frame[pcol], frame[rcol]))
        missing = [p for p in peptides if p not in lookup]
        if missing:
            raise SystemExit(f"{allele}: {len(missing)} peptides missing from "
                             f"output, e.g. {missing[:3]}")
        sub = g.copy()
        # %Rank_EL: lower is a better binder, so negate
        sub["score"] = [-float(lookup[p]) for p in sub.peptide]
        out_rows.append(sub)

    print("=== allele coverage (NetMHCpan nearest neighbour) ===")
    for allele, dist, nn in sorted(set(coverage)):
        flag = "" if dist == 0 and nn.replace("*", "").replace(
            "HLA-", "") == allele.replace("*", "").replace("HLA-", "") \
            else "   <-- SUBSTITUTED"
        print(f"  {allele:<14} distance {dist:.3f}  nearest {nn}{flag}")

    out = pd.concat(out_rows, ignore_index=True)
    out["in_train"] = ""   # NetMHCpan-4.1 training data is not public
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out[["allele", "peptide", "label", "in_train", "score"]].to_csv(
        args.out, index=False)

    print(f"\n=== NetMHCpan-4.1 on fold set (%Rank_EL) ===")
    print(f"  pooled         {roc_auc_score(out.label, out.score):.3f}  "
          f"(n={len(out)})")
    for allele, g in out.groupby("allele"):
        if g.label.nunique() > 1:
            print(f"  {allele:<14} {roc_auc_score(g.label, g.score):.3f}  "
                  f"(n={len(g)})")

    print(f"\nWrote {args.out}")
    print("Compare: ours 0.921 | MHCflurry presentation 0.841 | "
          "MHCflurry affinity 0.911")
    print("NOTE: in_train is empty -- NetMHCpan-4.1's training data is not public.")


if __name__ == "__main__":
    main()