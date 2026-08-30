"""Does structural consistency carry signal the other readouts missed?

What this is, and what it is not
---------------------------------
Kim et al. (Sci Rep 2024, s41598-024-72784-3) report that peptide-MHC complex
*stability* — the experimentally measured half-life of the complex — correlates with
immunogenicity better than binding affinity does.

That quantity cannot be computed here. It is a thermodynamic property measured in a
dissociation assay, and measured stability data does not exist for the alleles this
project is about, in the same way that experimentally determined non-binders do not.
This script therefore does not test their claim.

What it does test is a weaker and clearly different question: whether *structural
consistency* — how much the model's own predictions vary across its ranked outputs
and along the peptide — carries binder/decoy signal that the four readouts already
tested did not. The reasoning is that a complex modelled the same way every time may
be one the model considers well-determined, and that this could differ from a
complex modelled confidently once.

Prior expectation, stated in advance
-------------------------------------
Probably not. Four readouts of these same folds converged — PAE 0.804,
representations 0.834, confidence 0.753, geometry 0.492, against sequence 0.921 —
and consistency is closer in kind to a confidence metric than to any of the others.
The value of running it is that a fifth readout landing in the same place makes the
RQ1 conclusion harder to attribute to feature choice; a fifth readout that differs
would be genuinely surprising and worth pursuing.

Features
--------
  plddt_peptide_std     variation in per-residue pLDDT along the peptide. Low means
                        uniformly well-modelled rather than confident on average
                        with a poorly-placed terminus.
  plddt_anchor_gap      mean pLDDT at the allele's IC-derived anchors minus the mean
                        at the other positions. Positive means the anchors are the
                        best-determined part of the peptide, which is what a
                        correctly seated peptide should look like.
  pae_peptide_std       variation in peptide-to-MHC PAE across peptide positions.
  pae_asymmetry         |mean PAE(pep->MHC) - mean PAE(MHC->pep)|. PAE is not
                        symmetric, and a large gap means the model is much more
                        certain about one direction of the relationship than the
                        other.

All are computed per complex from files already on disk. No refolding.

Usage:
    python scripts/structural_consistency.py \
        esmfold2-v4 --fold-set fold_sets/fold_set_v4.csv \
        --anchors data/processed/anchors.json \
        --sequence results/sequence_v4.csv \
        --out consistency_esmfold2_v4.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from biotite.structure.io import pdbx
from sklearn.metrics import roc_auc_score


PEPTIDE_CHAIN = "C"


def slug_to_allele(slug: str) -> str:
    b = slug.split("_")
    return f"HLA-{b[1].upper()}*{b[2]}:{b[3]}" if len(b) >= 4 else slug


def load_fold_set(path: str) -> dict:
    out = {}
    for r in csv.reader(open(path)):
        if len(r) < 4:
            continue
        tag, _locus, slug, pep = r[0], r[1], r[2], r[3]
        meta = (slug_to_allele(slug), pep, tag in ("decoy", "hard"))
        for key in (f"{slug}_{pep.lower()}", f"{slug}__{pep.lower()}",
                    f"{tag}__{slug}__{pep.lower()}", f"NA__{slug}__{pep.lower()}",
                    f"{tag}__{slug}__{pep}"):
            out[key] = meta
    return out


def features(fold: Path, peptide_len: int, anchors) -> dict:
    pred = fold / "outputs" / "files" / "prediction"
    out = {}

    # ESMFold2's metrics.json holds only scalar complex_plddt; the per-residue
    # values are in the CIF b_factor column, on a 0-100 scale
    cif = pred / "sample_0_predicted_structure.cif"
    if cif.exists():
        try:
            f = pdbx.CIFFile.read(str(cif))
            a = pdbx.get_structure(f, model=1, extra_fields=["b_factor"])
            c = a[a.chain_id == PEPTIDE_CHAIN]
            res = sorted(set(c.res_id))
            if len(res) == peptide_len:
                pep = np.array([float(c[c.res_id == r].b_factor.mean())
                                for r in res])
                out["plddt_peptide_std"] = float(pep.std())
                if anchors:
                    idx = sorted({a_ % peptide_len for a_ in anchors})
                    other = [i for i in range(peptide_len) if i not in idx]
                    if idx and other:
                        out["plddt_anchor_gap"] = float(pep[idx].mean()
                                                        - pep[other].mean())
        except Exception:
            pass

    pfile = pred / "sample_0_pae.npz"
    if pfile.exists():
        with np.load(pfile) as z:
            pae = z[list(z)[0]]
        if pae.ndim == 2 and pae.shape[0] > peptide_len:
            n = pae.shape[0]
            pep = slice(n - peptide_len, n)
            mhc = slice(0, n - peptide_len)
            fwd = pae[pep, mhc]                 # peptide row, MHC column
            rev = pae[mhc, pep]
            out["pae_peptide_std"] = float(fwd.mean(axis=1).std())
            out["pae_asymmetry"] = float(abs(fwd.mean() - rev.mean()))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--fold-set", required=True)
    ap.add_argument("--anchors", default="data/processed/anchors.json")
    ap.add_argument("--sequence", help="for the comparison line only")
    ap.add_argument("--length", type=int, default=9)
    ap.add_argument("--out", default="consistency.csv")
    args = ap.parse_args()

    fs = load_fold_set(args.fold_set)
    anch = {}
    if Path(args.anchors).exists():
        anch = json.loads(Path(args.anchors).read_text()).get("alleles", {})

    rows, skipped = [], 0
    for fold in sorted(Path(args.root).iterdir()):
        if not fold.is_dir():
            continue
        meta = fs.get(fold.name)
        if meta is None:
            skipped += 1
            continue
        allele, pep, is_decoy = meta
        f = features(fold, len(pep), anch.get(allele, {}).get("anchors"))
        if not f:
            skipped += 1
            continue
        rows.append({"allele": allele, "peptide": pep,
                     "kind": "decoy" if is_decoy else "binder", **f})

    if skipped:
        print(f"skipped {skipped} folds with no usable metrics")
    d = pd.DataFrame(rows)
    if d.empty:
        raise SystemExit("nothing extracted — check the metrics.json field names")

    y = (d.kind == "binder").astype(int)
    n_b = int(y.sum())
    print(f"{len(d)} complexes ({n_b} binders / {len(d) - n_b} decoys)\n")

    feats = [c for c in d.columns if c not in ("allele", "peptide", "kind")]
    print("=== structural consistency features ===")
    print(f"{'feature':<22} {'binder':>9} {'decoy':>9} {'AUROC':>7} {'z-scored':>9}")
    res = []
    for f in feats:
        if d[f].isna().all():
            continue
        b, k = d[y == 1][f].mean(), d[y == 0][f].mean()
        v = d[f].fillna(d[f].mean())
        # direction is not known in advance for these, so report the better
        # orientation and record which it was
        a1 = roc_auc_score(y, v)
        auc = max(a1, 1 - a1)
        sign = "+" if a1 >= 0.5 else "-"
        z = v.groupby(d.allele).transform(lambda x: (x - x.mean()) / x.std())
        z = z.fillna(0)
        az = roc_auc_score(y, z if a1 >= 0.5 else -z)
        res.append({"feature": f, "binder_mean": b, "decoy_mean": k,
                    "auroc": auc, "auroc_z": az, "direction": sign})
        print(f"{f:<22} {b:>9.4f} {k:>9.4f} {auc:>7.3f} {az:>9.3f}  ({sign})")

    r = pd.DataFrame(res)
    d.to_csv(args.out, index=False)
    r.to_csv(str(Path(args.out).with_name(
        Path(args.out).stem + "_auroc.csv")), index=False)

    best = r.loc[r.auroc_z.idxmax()] if not r.empty else None
    print("\n=== against the readouts already tested on these folds ===")
    # these are fold set v4 figures; printed for orientation only, and labelled
    # as such because this script also runs on v2 where they differ
    print("  (fold set v4 reference values — not recomputed here)")
    print("  PAE, anchor-localised        0.804")
    print("  learned representations      0.834")
    print("  confidence metrics           0.753")
    print("  interface geometry           0.492")
    if args.sequence and Path(args.sequence).exists():
        s = pd.read_csv(args.sequence)
        if {"label", "score"} <= set(s.columns):
            print(f"  sequence                     "
                  f"{roc_auc_score(s.label, s.score):.3f}")
    if best is not None:
        print(f"\n  best consistency feature     {best.auroc_z:.3f} "
              f"({best.feature})")
        if best.auroc_z > 0.85:
            print("\n  -> higher than every other structural readout. That would be")
            print("     genuinely surprising and worth pursuing: consistency would")
            print("     be carrying signal that confidence and PAE do not.")
        elif best.auroc_z > 0.75:
            print("\n  -> in the same band as the other structural readouts. A fifth")
            print("     independent way of reading these folds lands in the same")
            print("     place, which makes RQ1's conclusion harder to attribute to")
            print("     feature choice.")
        else:
            print("\n  -> below the other structural readouts. Consistency adds")
            print("     nothing, and the four already reported remain the best")
            print("     case for structure.")

    print("\nNote this is not the quantity Kim et al. measure. Their stability is an")
    print("experimental complex half-life; measured stability data does not exist")
    print("for these alleles, which is the coverage gap in a third form.")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()