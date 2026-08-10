"""RQ3: do the sequence and structural models respond to the same positions?

The question, and why it matters for RQ2
-----------------------------------------
Eight combination strategies found no synergy between sequence and structure, and a
ninth (gating) matched its own permutation control. That is a well-evidenced null,
but it is an observation without a mechanism. This supplies one.

If the two model types respond to the same positions and substitutions, the
redundancy is explained: they have learned the same binding biology by different
routes. If they respond to different things and combining still fails, the null
becomes a sharper puzzle and RQ2's account needs qualifying.

The sequence half is already done (`rq3_sequence_landscape.py`): across seven
alleles the model agrees with the Motif Atlas PWM strongly on which positions matter
(median rho 0.817) and only moderately on which residues go there (median rho 0.541),
with its top-two positions a subset of the IC-derived anchors for six of seven.

Three comparisons, in order of what they license
-------------------------------------------------
SEED STABILITY   Correlation between landscapes built from different starting
                 peptides of the same allele. This runs first because it decides
                 whether any of the rest is interpretable. A landscape from one seed
                 confounds the allele's chemistry with that peptide's idiosyncrasies;
                 if seeds disagree, single-seed landscapes elsewhere in the
                 literature — and any conclusion drawn here — are unreliable.

STRUCTURE vs     The actual RQ3 question, at two levels: the full position x residue
SEQUENCE         landscape, and the coarser per-position sensitivity profile. The
                 sequence half showed those two can diverge sharply, so both are
                 reported rather than one standing for the other.

ANCHOR RECOVERY  Whether each model's most-sensitive positions fall among the
                 IC-derived anchors from derive_anchors.py. There is a specific
                 prediction to test: HLA-B*08:01 has anchors at P2, P5, P8 and P9
                 and the sequence model picked only P2 and P9, so it does not capture
                 the secondary anchors that Chris Thorpe's P5 MSA rebalancing
                 targets. If the structural landscape peaks at P5 there, the two
                 model types have found different things.

Scoring
-------
Interface PAE between the peptide and the MHC, the same quantity used throughout
RQ1, differenced against the wild-type fold from the same seed. Differencing removes
the per-peptide offset the way per-allele z-scoring removed the per-allele one — an
absolute PAE is not comparable across starting peptides.

Sign convention: PAE is lower-is-better, so the reported delta is negated to make
positive mean "this substitution made the complex worse", matching the sequence
model's logit delta.

Usage:
    python scripts/rq3_compare_landscapes.py /tmp/rq3/esmfold2-rq3 \
        --variants fold_sets/rq3_variants.csv \
        --sequence-landscape results/rq3_sequence_landscape.csv \
        --anchors data/processed/anchors.json \
        --out results/rq3_structural
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

AA = "ACDEFGHIKLMNPQRSTVWY"


def slug_to_allele(slug: str) -> str:
    b = slug.split("_")
    return f"HLA-{b[1].upper()}*{b[2]}:{b[3]}" if len(b) >= 4 else slug


def load_variants(path: str) -> dict:
    """fold name -> (allele, peptide, seed, position, aa) ; position None for wt."""
    out = {}
    for r in csv.reader(open(path)):
        if len(r) < 5:
            continue
        tag, _locus, slug, pep, note = r[0], r[1], r[2], r[3], r[4]
        # fold_esmfold2.py keeps the peptide case from the CSV, unlike
        # HISTOFold which lowercases it
        name = f"{tag}__{slug}__{pep}"
        if tag == "wt":
            out[name] = (slug_to_allele(slug), pep, note, None, None)
        else:
            m = re.match(r"(seed\d+)_p(\d+)([A-Z])$", note)
            if not m:
                continue
            out[name] = (slug_to_allele(slug), pep, m.group(1),
                         int(m.group(2)), m.group(3))
    return out


def interface_pae(fold: Path, peptide_len: int) -> float | None:
    """Mean PAE between peptide tokens and MHC tokens, both directions."""
    f = fold / "outputs" / "files" / "prediction" / "sample_0_pae.npz"
    if not f.exists():
        return None
    with np.load(f) as z:
        pae = z[list(z)[0]]
    if pae.ndim != 2 or pae.shape[0] <= peptide_len:
        return None
    pep = slice(pae.shape[0] - peptide_len, pae.shape[0])
    mhc = slice(0, pae.shape[0] - peptide_len)
    return float(np.concatenate([pae[pep, mhc].ravel(),
                                 pae[mhc, pep].ravel()]).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--variants", required=True)
    ap.add_argument("--sequence-landscape")
    ap.add_argument("--anchors", default="data/processed/anchors.json")
    ap.add_argument("--length", type=int, default=9)
    ap.add_argument("--out", default="results/rq3_structural")
    args = ap.parse_args()

    variants = load_variants(args.variants)
    print(f"{len(variants)} variants declared\n")

    rows, missing = [], 0
    for fold in sorted(Path(args.root).iterdir()):
        if not fold.is_dir():
            continue
        meta = variants.get(fold.name)
        if meta is None:
            continue
        allele, pep, seed, pos, aa = meta
        v = interface_pae(fold, args.length)
        if v is None:
            missing += 1
            continue
        rows.append({"allele": allele, "seed": seed, "position": pos,
                     "aa": aa, "pae": v, "is_wt": pos is None})
    if missing:
        print(f"WARNING: {missing} folds had no readable PAE")
    d = pd.DataFrame(rows)
    if d.empty:
        raise SystemExit("nothing loaded")

    # difference each variant against its own seed's wild type
    wt = (d[d.is_wt].set_index(["allele", "seed"]).pae.to_dict())
    mut = d[~d.is_wt].copy()
    mut["wt_pae"] = [wt.get((a, s), np.nan)
                     for a, s in zip(mut.allele, mut.seed)]
    mut = mut.dropna(subset=["wt_pae"])
    # PAE is lower-is-better; negate so positive = substitution made it worse
    mut["delta"] = -(mut.pae - mut.wt_pae)
    print(f"{len(mut)} variants scored across {mut.allele.nunique()} alleles, "
          f"{mut.seed.nunique()} seeds\n")

    def landscape(sub):
        # `position` is float because the wild-type rows carry None, so the column
        # is nullable; cast before indexing
        m = np.full((args.length, 20), np.nan)
        for _, r in sub.iterrows():
            m[int(r.position) - 1, AA.index(r.aa)] = r.delta
        return m

    # ---- 1. seed stability, which licenses everything after it ----
    print("=== seed-to-seed agreement (does one starting peptide suffice?) ===")
    stab = []
    for allele, g in mut.groupby("allele"):
        seeds = sorted(g.seed.unique())
        L = {s: landscape(g[g.seed == s]) for s in seeds}
        rs = []
        for i in range(len(seeds)):
            for j in range(i + 1, len(seeds)):
                a, b = L[seeds[i]].ravel(), L[seeds[j]].ravel()
                ok = ~(np.isnan(a) | np.isnan(b))
                if ok.sum() > 20:
                    rs.append(stats.spearmanr(a[ok], b[ok])[0])
        if rs:
            stab.append({"allele": allele, "n_seeds": len(seeds),
                         "mean_seed_rho": float(np.mean(rs)),
                         "min_seed_rho": float(np.min(rs))})
            print(f"  {allele:<14} mean rho {np.mean(rs):+.3f}   "
                  f"min {np.min(rs):+.3f}   ({len(seeds)} seeds)")
    stab = pd.DataFrame(stab)
    if not stab.empty:
        m = stab.mean_seed_rho.median()
        print(f"\n  median across alleles: {m:+.3f}")
        if m < 0.3:
            print("  -> seeds disagree. Single-seed landscapes, here or elsewhere,")
            print("     are not interpretable, and the comparisons below are")
            print("     dominated by which peptide happened to be chosen.")
        elif m < 0.6:
            print("  -> moderate. Landscapes carry allele signal but a single seed")
            print("     is noisy; report averaged over seeds and say so.")
        else:
            print("  -> stable. Averaging over seeds is a refinement, not a")
            print("     necessity, and single-seed landscapes are defensible.")

    # seed-averaged landscape per allele
    avg = (mut.groupby(["allele", "position", "aa"]).delta.mean()
           .reset_index())
    avg.to_csv(f"{args.out}_landscape.csv", index=False)

    anch = {}
    if Path(args.anchors).exists():
        anch = json.loads(Path(args.anchors).read_text()).get("alleles", {})

    # ---- 2. structure vs sequence ----
    seq = None
    if args.sequence_landscape and Path(args.sequence_landscape).exists():
        seq = pd.read_csv(args.sequence_landscape)

    print("\n=== structural landscape vs sequence landscape ===")
    out = []
    for allele, g in avg.groupby("allele"):
        L = landscape(g)
        sens = np.nanstd(L, axis=1)
        top = [int(x) for x in np.argsort(-sens)[:2] + 1]

        rec = anch.get(allele, {})
        ic = rec.get("ic", [])
        n = len(ic) or args.length
        derived = sorted((x % n) + 1 for x in rec.get("anchors", []))

        row = {"allele": allele, "struct_top_positions": top,
               "derived_anchors": derived,
               "struct_top_in_anchors": set(top) <= set(derived) if derived else None}

        if seq is not None and allele in set(seq.allele):
            s = seq[seq.allele == allele]
            S = np.full((args.length, 20), np.nan)
            for _, r in s.iterrows():
                S[int(r.position) - 1, AA.index(r.aa)] = r.model_delta
            a, b = L.ravel(), S.ravel()
            ok = ~(np.isnan(a) | np.isnan(b))
            row["landscape_rho"] = round(stats.spearmanr(a[ok], b[ok])[0], 3)
            ssens = np.nanstd(S, axis=1)
            row["position_rho"] = round(stats.spearmanr(sens, ssens)[0], 3)
            row["seq_top_positions"] = [int(x) for x in np.argsort(-ssens)[:2] + 1]

        out.append(row)
        lr = row.get("landscape_rho")
        pr = row.get("position_rho")
        print(f"  {allele:<14} struct anchors {str(top):<8} derived {derived}")
        if lr is not None:
            print(f"  {'':14} landscape rho {lr:+.3f}   position rho {pr:+.3f}   "
                  f"seq anchors {row['seq_top_positions']}")

    res = pd.DataFrame(out)
    res.to_csv(f"{args.out}_summary.csv", index=False)
    if not stab.empty:
        stab.to_csv(f"{args.out}_seed_stability.csv", index=False)

    if "landscape_rho" in res and res.landscape_rho.notna().any():
        print(f"\n  median landscape agreement: "
              f"{res.landscape_rho.median():+.3f}")
        print(f"  median position agreement:  "
              f"{res.position_rho.median():+.3f}")
        print("\n  High agreement means the two model types respond to the same")
        print("  positions and substitutions, which would explain RQ2's redundancy")
        print("  mechanistically. Low agreement makes the failure to combine a")
        print("  sharper puzzle.")

    # ---- 3. the B*08:01 prediction ----
    b8 = res[res.allele == "HLA-B*08:01"]
    if not b8.empty:
        r = b8.iloc[0]
        L = landscape(avg[avg.allele == "HLA-B*08:01"])
        sens = np.nanstd(L, axis=1)
        rank5 = int(np.argsort(-sens).tolist().index(4)) + 1
        print(f"\n=== the HLA-B*08:01 prediction ===")
        print(f"  derived anchors {r.derived_anchors}; the sequence model picked "
              f"{r.get('seq_top_positions')}")
        print(f"  structural sensitivity ranks P5 at position {rank5} of "
              f"{args.length}")
        if rank5 <= 3:
            print("  -> the structural model does respond at P5 where the sequence")
            print("     model does not. The two have found different things, and")
            print("     RQ2's redundancy account needs qualifying.")
        else:
            print("  -> the structural model does not single out P5 either, so both")
            print("     miss the secondary anchors that Chris's P5 rebalancing")
            print("     targets.")

    print(f"\nWrote {args.out}_landscape.csv, {args.out}_summary.csv"
          + (", {}_seed_stability.csv".format(args.out) if not stab.empty else ""))


if __name__ == "__main__":
    main()