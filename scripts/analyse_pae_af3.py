"""PAE and confidence features from AlphaFold 3 output.

Produces the same schema as scripts/analyse_pae_af2.py and
scripts/extract_confidence.py, so scripts/auroc_structure.py reads it unchanged.

AF3's output format differs from ColabFold's in ways that are mostly improvements:

  EXPLICIT CHAINS   `token_chain_ids` labels every token with its chain, so the
                    peptide is identified directly rather than assumed to be the
                    final N rows. That assumption held for the other pipelines but
                    was never checked; here it does not have to be made.

  ONE FILE          `<name>_confidences.json` holds `pae` (an n_tokens x n_tokens
                    matrix), `atom_plddts`, `atom_chain_ids`, `token_chain_ids` and
                    `token_res_ids`. The per-chain summary is in
                    `<name>_summary_confidences.json`.

  CHAIN PAIR PAE    `chain_pair_pae_min` is a per-chain-pair minimum PAE that none
                    of the other models expose. Extracted as `pae_min_pep_mhc`.

Features, matching the other extractors where they overlap:

  pae_pep_mhc        mean PAE over peptide-vs-MHC token pairs, both directions
  pae_anchor2        peptide position 2 against the MHC
  pae_anchorC        peptide C-terminal residue against the MHC
  pae_anchors        mean of the two above
  pae_anchors_ic     the allele's high-information positions from
                     scripts/derive_anchors.py, where available
  iptm, ptm          whole-complex
  iptm_pep_mhc       chain_pair_iptm[peptide][mhc], the peptide-groove pair
  iptm_pep_self      chain_pair_iptm[peptide][peptide]
  plddt_peptide      mean pLDDT over peptide atoms
  complex_plddt      mean pLDDT over all atoms
  pae_min_pep_mhc    chain_pair_pae_min[peptide][mhc]
  ranking_score      AF3's own ranking score
  has_clash, fraction_disordered

Chains follow the input JSON: A = MHC heavy chain, B = beta-2-microglobulin,
C = peptide.

Usage:
    python scripts/analyse_pae_af3.py /tmp/af3/output \
        --fold-set fold_sets/fold_set_v4.csv \
        --anchors data/processed/anchors.json \
        --out pae_af3_v4.csv --conf-out conf_af3_v4.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

MHC_CHAIN = "A"
PEPTIDE_CHAIN = "C"


def slug_to_allele(slug: str) -> str:
    b = slug.split("_")
    return f"HLA-{b[1].upper()}*{b[2]}:{b[3]}" if len(b) >= 4 else slug


def load_fold_set(path: str) -> dict:
    out = {}
    with open(path) as fh:
        for row in csv.reader(fh):
            if len(row) < 4:
                continue
            tag, _locus, slug, peptide = row[0], row[1], row[2], row[3]
            meta = (slug_to_allele(slug), peptide, tag in ("decoy", "hard"))
            out[f"{tag}__{slug}__{peptide.lower()}"] = meta
            out[f"{slug}__{peptide.lower()}"] = meta
            out[f"{slug}_{peptide.lower()}"] = meta
    return out


def strip_timestamp(name: str) -> str:
    """AF3 appends _YYYYMMDD_HHMMSS when the output directory already exists."""
    return re.sub(r"_\d{8}_\d{6}$", "", name)


def pae_features(pae, token_chain_ids, peptide_len, anchor_pos=None) -> dict:
    chains = np.asarray(token_chain_ids)
    pep = np.where(chains == PEPTIDE_CHAIN)[0]
    mhc = np.where(chains == MHC_CHAIN)[0]
    if len(pep) == 0 or len(mhc) == 0:
        return {}
    if len(pep) != peptide_len:
        # not fatal, but the anchor indices would be wrong
        return {}

    pae = np.asarray(pae, dtype=float)
    inter = np.concatenate([pae[np.ix_(pep, mhc)].ravel(),
                            pae[np.ix_(mhc, pep)].ravel()])

    def res_vs_mhc(i):
        t = pep[i]
        return float(np.concatenate([pae[t, mhc], pae[mhc, t]]).mean())

    a2 = res_vs_mhc(1) if peptide_len >= 2 else res_vs_mhc(0)
    ac = res_vs_mhc(-1)
    out = {
        "pae_pep_mhc": float(inter.mean()),
        "pae_anchor2": a2,
        "pae_anchorC": ac,
        "pae_anchors": (a2 + ac) / 2,
    }
    if anchor_pos:
        vals = [res_vs_mhc(i) for i in anchor_pos
                if -peptide_len <= i < peptide_len]
        if vals:
            out["pae_anchors_ic"] = float(np.mean(vals))
    return out


def conf_features(summary, conf) -> dict:
    out = {}
    for k in ("iptm", "ptm", "ranking_score", "has_clash",
              "fraction_disordered"):
        if k in summary and isinstance(summary[k], (int, float)):
            out[k] = float(summary[k])

    # chain order in the summary matrices follows the input JSON: A, B, C
    cp_iptm = summary.get("chain_pair_iptm")
    if isinstance(cp_iptm, list) and len(cp_iptm) >= 3:
        out["iptm_pep_mhc"] = float(cp_iptm[2][0])
        out["iptm_mhc_pep"] = float(cp_iptm[0][2])
        out["iptm_pep_mhc_mean"] = (out["iptm_pep_mhc"] + out["iptm_mhc_pep"]) / 2
        out["iptm_pep_self"] = float(cp_iptm[2][2])

    cp_pae = summary.get("chain_pair_pae_min")
    if isinstance(cp_pae, list) and len(cp_pae) >= 3:
        # lower is better, so this is declared in LOWER_IS_BINDING
        out["pae_min_pep_mhc"] = float(cp_pae[2][0])

    plddts = conf.get("atom_plddts")
    atom_chains = conf.get("atom_chain_ids")
    if isinstance(plddts, list) and isinstance(atom_chains, list):
        p = np.asarray(plddts, dtype=float)
        c = np.asarray(atom_chains)
        out["complex_plddt"] = float(p.mean()) / 100.0
        pepmask = c == PEPTIDE_CHAIN
        if pepmask.any():
            out["plddt_peptide"] = float(p[pepmask].mean()) / 100.0
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="AF3 output directory")
    ap.add_argument("--fold-set", required=True)
    ap.add_argument("--anchors", default="data/processed/anchors.json")
    ap.add_argument("--out", default="pae_af3.csv")
    ap.add_argument("--conf-out", help="also write confidence features here")
    args = ap.parse_args()

    fold_set = load_fold_set(args.fold_set)
    anchors = {}
    if args.anchors and Path(args.anchors).exists():
        with open(args.anchors) as fh:
            anchors = json.load(fh).get("alleles", {})
        print(f"anchors: {len(anchors)} alleles")

    pae_rows, conf_rows, skipped = [], [], []
    for d in sorted(Path(args.root).iterdir()):
        if not d.is_dir():
            continue
        key = strip_timestamp(d.name)
        meta = fold_set.get(key)
        if meta is None:
            skipped.append(d.name)
            continue
        allele, peptide, is_decoy = meta
        kind = "decoy" if is_decoy else "binder"

        conf_files = list(d.glob("*_confidences.json"))
        conf_files = [f for f in conf_files if "summary" not in f.name]
        summ_files = list(d.glob("*_summary_confidences.json"))
        if not conf_files or not summ_files:
            skipped.append(d.name)
            continue
        with open(conf_files[0]) as fh:
            conf = json.load(fh)
        with open(summ_files[0]) as fh:
            summary = json.load(fh)

        feats = pae_features(conf.get("pae"), conf.get("token_chain_ids"),
                             len(peptide),
                             anchors.get(allele, {}).get("anchors"))
        if feats:
            pae_rows.append({"allele": allele, "peptide": peptide,
                             "kind": kind, **feats})
        else:
            skipped.append(f"{d.name} (pae)")

        cf = conf_features(summary, conf)
        if cf:
            conf_rows.append({"allele": allele, "peptide": peptide,
                              "kind": kind, **cf})

    if skipped:
        print(f"skipped {len(skipped)}: {skipped[:5]}")

    df = pd.DataFrame(pae_rows)
    if df.empty:
        raise SystemExit("No PAE features extracted.")
    df.to_csv(args.out, index=False)
    n_b = (df.kind == "binder").sum()
    print(f"\n{len(df)} folds ({n_b} binders / {len(df) - n_b} decoys)")

    print("\n=== binder vs decoy anchor-PAE, per allele ===")
    for allele, sub in df.groupby("allele"):
        b = sub[sub.kind == "binder"].pae_anchors.mean()
        dd = sub[sub.kind == "decoy"].pae_anchors.mean()
        flag = "binders lower (expected)" if dd > b else "binders HIGHER"
        print(f"  {allele:<14} binder {b:.3f}  decoy {dd:.3f}  "
              f"gap {dd - b:+.3f}  <-- {flag}")
    print(f"\nWrote {args.out}")

    if args.conf_out and conf_rows:
        cdf = pd.DataFrame(conf_rows)
        cdf.to_csv(args.conf_out, index=False)
        feats = [c for c in cdf.columns if c not in ("allele", "peptide", "kind")]
        print(f"\nWrote {args.conf_out} ({len(cdf)} folds)")
        print(f"{'feature':<22} {'binder':>10} {'decoy':>10} {'gap':>10}")
        for f in feats:
            b = cdf[cdf.kind == "binder"][f].mean()
            dd = cdf[cdf.kind == "decoy"][f].mean()
            print(f"{f:<22} {b:>10.4f} {dd:>10.4f} {b - dd:>+10.4f}")

    print(f"\nNext: python scripts/auroc_structure.py --pae {args.out} "
          f"--out auroc_af3.csv --sequence-csv results/sequence_v4.csv")


if __name__ == "__main__":
    main()