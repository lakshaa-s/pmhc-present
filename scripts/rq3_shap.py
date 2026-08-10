"""RQ3: Shapley attribution on the sequence model, against the mutational landscape.

Why run this when we already have a landscape
----------------------------------------------
A mutational landscape *is* per-position attribution, computed by direct perturbation.
So SHAP is not obviously additive — and Chris Thorpe suggested it as post-MSc work
partly for that reason. The reason to run it now is a specific and testable one.

The structural landscapes turned out to be seed-unstable: three starting peptides of
the same allele gave landscapes correlating at a median rho of only +0.168, with
negative minima for two of four alleles. A landscape is conditional on its seed
peptide by construction.

SHAP is not. It marginalises each position against a background distribution, so the
attribution is a property of the allele and the model rather than of one peptide. If
SHAP is more seed-stable than the landscape, it is the better primary analysis for
the sequence side, and the landscape becomes the supplement rather than the reverse.

That is the question here: not "what does SHAP say" but "is SHAP more stable than
what we already have, and do the two agree".

Method
------
Exact Shapley values are 2^9 = 512 coalitions per peptide, which is cheap enough to
compute exhaustively for a 9mer — no sampling approximation needed, so the numbers
are exact rather than estimated.

A position is "absent" from a coalition when it is resampled from the allele's own
held-out peptides at that position. That background matters: masking with a padding
token would measure the model's response to nonsense rather than to a typical
alternative, and would inflate every attribution.

For each peptide, phi[i] is the mean marginal contribution of position i across all
coalitions, and the phi sum to (score of the real peptide) - (mean score of the
background), which is the standard efficiency property and is checked.

Three comparisons, mirroring the landscape analysis
----------------------------------------------------
  SEED STABILITY   Agreement between attributions from different starting peptides
                   of the same allele. Directly comparable with the landscape's
                   +0.168, and the whole point of the exercise.

  vs LANDSCAPE     Does SHAP identify the same positions as direct perturbation? If
                   yes, the two methods validate each other. If no, at least one is
                   measuring something other than what we think.

  vs ANCHORS       Whether the top-attributed positions fall among the IC-derived
                   anchors, as the landscape achieved for 6 of 7 alleles.

Usage:
    python scripts/rq3_shap.py \
        --alleles fold_sets/rq3_alleles.txt \
        --fold-sets fold_sets/fold_set_v2.csv fold_sets/fold_set_v4.csv \
                    fold_sets/binders_rq3.csv \
        --atlas data/processed/atlas_labelled.csv \
        --split data/processed/split_val.csv \
        --model models/rq1_baseline_split_v2.pt \
        --pseudoseq data/pseudoseq/hla_a.json data/pseudoseq/hla_b.json \
                    data/pseudoseq/hla_c.json \
        --sequence-landscape results/rq3_sequence_landscape.csv \
        --n-seeds 6 --out results/rq3_shap
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from itertools import combinations
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

AA = "ACDEFGHIKLMNPQRSTVWY"


def slug_to_allele(slug: str) -> str:
    b = slug.split("_")
    return f"HLA-{b[1].upper()}*{b[2]}:{b[3]}" if len(b) >= 4 else slug


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alleles", required=True)
    ap.add_argument("--fold-sets", nargs="+", required=True)
    ap.add_argument("--atlas", default="data/processed/atlas_labelled.csv")
    ap.add_argument("--split", default="data/processed/split_val.csv")
    ap.add_argument("--model", required=True)
    ap.add_argument("--pseudoseq", nargs="+", required=True)
    ap.add_argument("--sequence-landscape")
    ap.add_argument("--anchors", default="data/processed/anchors.json")
    ap.add_argument("--length", type=int, default=9)
    ap.add_argument("--n-seeds", type=int, default=6)
    ap.add_argument("--n-background", type=int, default=64,
                    help="background peptides drawn per allele; each coalition is "
                         "averaged over this many resamplings")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/rq3_shap")
    args = ap.parse_args()

    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from pmhcpresent.models.nn import PresentationNet, NetConfig
    from pmhcpresent.io.peptides import encode_sequence

    rng = np.random.default_rng(args.seed)
    L = args.length

    panel = [l.strip() for l in open(args.alleles) if l.strip()]
    pseudo = {}
    for f in args.pseudoseq:
        pseudo.update(json.loads(Path(f).read_text()))

    starts: dict[str, list[tuple[str, str]]] = {}
    for fs in args.fold_sets:
        for r in csv.reader(open(fs)):
            if len(r) < 4 or r[0] in ("hard", "decoy"):
                continue
            a = slug_to_allele(r[2])
            if a in panel and len(r[3]) == L:
                starts.setdefault(a, []).append((r[2], r[3].upper()))

    atlas = pd.read_csv(args.atlas)
    atlas = atlas[(atlas.label == 1) & (atlas.length == L)]
    val = set(map(tuple, pd.read_csv(args.split).values))
    atlas = atlas[[(a, p) in val for a, p in zip(atlas.allele, atlas.peptide)]]

    cfg = NetConfig()
    model = PresentationNet(cfg)
    state = torch.load(args.model, map_location="cpu")
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()

    def score(peptides, slug):
        ps = pseudo[slug]["pocket_pseudosequence"]
        pep = torch.tensor(np.stack([encode_sequence(p, cfg.max_pep_len)
                                     for p in peptides]), dtype=torch.long)
        mhc = torch.tensor(np.stack([encode_sequence(ps, cfg.pseudoseq_len)]
                                    * len(peptides)), dtype=torch.long)
        with torch.no_grad():
            return model(pep, mhc).numpy()

    # exhaustive Shapley for L=9: 512 coalitions, so no sampling approximation
    weights = {k: 1.0 / (L * comb(L - 1, k)) for k in range(L)}

    rows, per_seed = [], []
    for allele in panel:
        seeds = starts.get(allele, [])[:args.n_seeds]
        if not seeds:
            print(f"  {allele}: no canonical binders, skipped")
            continue
        pool = atlas[atlas.allele == allele].peptide.tolist()
        if len(pool) < 20:
            print(f"  {allele}: only {len(pool)} background peptides, skipped")
            continue
        bg = [pool[i] for i in rng.integers(0, len(pool), args.n_background)]
        slug = seeds[0][0]

        phis = []
        for si, (_, wt) in enumerate(seeds):
            # value of every coalition, averaged over the background
            vals = {}
            for k in range(L + 1):
                for S in combinations(range(L), k):
                    Sset = set(S)
                    variants = []
                    for b in bg:
                        variants.append("".join(wt[i] if i in Sset else b[i]
                                                for i in range(L)))
                    vals[S] = float(score(variants, slug).mean())

            phi = np.zeros(L)
            for i in range(L):
                others = [j for j in range(L) if j != i]
                for k in range(L):
                    for S in combinations(others, k):
                        phi[i] += weights[k] * (vals[tuple(sorted(S + (i,)))]
                                                - vals[tuple(sorted(S))])
            # efficiency: the phi should sum to f(full) - f(empty)
            gap = abs(phi.sum() - (vals[tuple(range(L))] - vals[()]))
            if gap > 1e-3:
                print(f"    WARNING {allele} seed{si}: efficiency gap {gap:.4f}")

            phis.append(phi)
            for i in range(L):
                rows.append({"allele": allele, "seed": f"seed{si}",
                             "position": i + 1, "phi": phi[i],
                             "wt_residue": wt[i]})

        P = np.vstack(phis)
        rs = [stats.spearmanr(P[i], P[j])[0]
              for i in range(len(P)) for j in range(i + 1, len(P))]
        mean_phi = P.mean(axis=0)
        top = [int(x) for x in np.argsort(-np.abs(mean_phi))[:2] + 1]
        per_seed.append({"allele": allele, "n_seeds": len(seeds),
                         "mean_seed_rho": float(np.mean(rs)) if rs else np.nan,
                         "min_seed_rho": float(np.min(rs)) if rs else np.nan,
                         "top_positions": top})
        print(f"  {allele:<14} seed agreement {np.mean(rs):+.3f} "
              f"(min {np.min(rs):+.3f}, {len(seeds)} seeds)   top {top}")

    d = pd.DataFrame(rows)
    s = pd.DataFrame(per_seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(f"{args.out}_attributions.csv", index=False)

    print("\n=== seed stability: SHAP vs the mutational landscape ===")
    print(f"  SHAP, median across alleles:      "
          f"{s.mean_seed_rho.median():+.3f}")
    print(f"  structural landscape, for comparison:  +0.168")
    print("  (the sequence landscape was averaged over 12 seeds and so has no")
    print("   directly comparable per-seed figure)")
    if s.mean_seed_rho.median() > 0.6:
        print("\n  -> SHAP is substantially more stable. It marginalises over a")
        print("     background rather than depending on one starting peptide, so")
        print("     it is the better primary analysis for the sequence side.")
    elif s.mean_seed_rho.median() > 0.3:
        print("\n  -> more stable than the landscape but not decisively so.")
    else:
        print("\n  -> no more stable than the landscape. The instability is in the")
        print("     model or the task, not in the attribution method.")

    anch = {}
    if Path(args.anchors).exists():
        anch = json.loads(Path(args.anchors).read_text()).get("alleles", {})

    seq = None
    if args.sequence_landscape and Path(args.sequence_landscape).exists():
        seq = pd.read_csv(args.sequence_landscape)

    print("\n=== SHAP vs landscape vs derived anchors ===")
    out = []
    for _, r in s.iterrows():
        rec = anch.get(r.allele, {})
        ic = rec.get("ic", [])
        n = len(ic) or L
        derived = sorted((x % n) + 1 for x in rec.get("anchors", []))
        row = {"allele": r.allele, "shap_top": r.top_positions,
               "derived_anchors": derived,
               "shap_top_in_anchors": set(r.top_positions) <= set(derived)
               if derived else None,
               "mean_seed_rho": round(r.mean_seed_rho, 3)}

        if seq is not None and r.allele in set(seq.allele):
            g = seq[seq.allele == r.allele]
            sens = g.groupby("position").model_delta.std().reindex(
                range(1, L + 1)).to_numpy()
            phi = (d[d.allele == r.allele].groupby("position").phi.mean()
                   .reindex(range(1, L + 1)).to_numpy())
            row["shap_vs_landscape_rho"] = round(
                stats.spearmanr(np.abs(phi), sens)[0], 3)
            row["landscape_top"] = [int(x) for x in np.argsort(-sens)[:2] + 1]

        out.append(row)
        print(f"  {r.allele:<14} SHAP {str(r.top_positions):<8} "
              f"landscape {str(row.get('landscape_top', '-')):<8} "
              f"derived {derived}"
              + (f"   rho {row['shap_vs_landscape_rho']:+.3f}"
                 if "shap_vs_landscape_rho" in row else ""))

    res = pd.DataFrame(out)
    res.to_csv(f"{args.out}_summary.csv", index=False)
    if "shap_vs_landscape_rho" in res:
        print(f"\n  median SHAP-landscape agreement: "
              f"{res.shap_vs_landscape_rho.median():+.3f}")
    if "shap_top_in_anchors" in res and res.shap_top_in_anchors.notna().any():
        n_ok = int(res.shap_top_in_anchors.sum())
        print(f"  SHAP top-2 within the derived anchors: {n_ok}/{len(res)} "
              f"(the landscape managed 6/7)")

    print(f"\nWrote {args.out}_attributions.csv and {args.out}_summary.csv")


if __name__ == "__main__":
    main()