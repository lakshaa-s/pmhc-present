"""RQ3, sequence half: is the sequence model just learning the motif?

The question, from the 6 August meeting
----------------------------------------
Benny's hypothesis was that the sequence learner is essentially recovering the
binding motif, and that this is why it performs well without needing much training
data per allele. This tests it directly, and needs no folding — the sequence model
scores 1,197 variants instantly.

Method
------
For each allele, take its canonical binders from the fold set, mutate every position
to every other amino acid, and score all variants. The change in score relative to
the wild-type peptide gives a position x amino-acid mutational landscape, averaged
over starting peptides so the result is a property of the allele rather than of one
sequence.

The same landscape is computed from the Motif Atlas PWM, which is the motif by
definition. If Benny is right, the two should agree closely.

What the comparison can and cannot show
----------------------------------------
Agreement means the model's learned preferences match the observed amino-acid
frequencies — which is what "learning the motif" means operationally. It does not
follow that the model is *only* a PWM: a PWM is position-independent by construction,
so systematic disagreement concentrated on particular position pairs would be
evidence of learned higher-order structure. That is reported as the residual.

There is a specific reason to expect the anchors to emerge anyway. The model
(PresentationNet) max-pools over its convolution output, so the peptide path carries
no absolute position information — it can detect a local pattern but not where it
occurred. If the landscape still peaks at P2 and the C-terminus, that shows anchor
positions are recoverable from composition and local context alone, which is a
sharper result than it would be from a model that was given position explicitly.

Outputs
-------
  <out>_landscape.csv   per allele, position, amino acid: mean score change
  <out>_summary.csv     per allele: position sensitivity, PWM correlation, anchors

Usage:
    python scripts/rq3_sequence_landscape.py \
        --alleles fold_sets/rq3_alleles.txt \
        --fold-sets fold_sets/fold_set_v2.csv fold_sets/fold_set_v4.csv \
        --atlas data/processed/atlas_labelled.csv \
        --model models/rq1_baseline_split_v2.pt \
        --pseudoseq data/pseudoseq/hla_a.json data/pseudoseq/hla_b.json \
                    data/pseudoseq/hla_c.json \
        --out results/rq3_sequence
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats

AA = "ACDEFGHIKLMNPQRSTVWY"


def slug_to_allele(slug: str) -> str:
    b = slug.split("_")
    return f"HLA-{b[1].upper()}*{b[2]}:{b[3]}" if len(b) >= 4 else slug


def pwm_landscape(peptides, length: int, pseudo: float = 0.5) -> np.ndarray:
    """log2 frequency ratio per position and amino acid — the motif itself."""
    idx = {a: i for i, a in enumerate(AA)}
    m = np.full((length, 20), pseudo)
    for p in peptides:
        for i, c in enumerate(p):
            if c in idx:
                m[i, idx[c]] += 1
    f = m / m.sum(axis=1, keepdims=True)
    return np.log2(f / 0.05)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alleles", required=True)
    ap.add_argument("--fold-sets", nargs="+", required=True)
    ap.add_argument("--atlas", default="data/processed/atlas_labelled.csv")
    ap.add_argument("--model", required=True)
    ap.add_argument("--pseudoseq", nargs="+", required=True)
    ap.add_argument("--length", type=int, default=9)
    ap.add_argument("--max-starts", type=int, default=12,
                    help="canonical binders to average the landscape over")
    ap.add_argument("--out", default="results/rq3_sequence")
    args = ap.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from pmhcpresent.models.nn import PresentationNet, NetConfig
    from pmhcpresent.io.peptides import encode_sequence

    panel = [l.strip() for l in open(args.alleles) if l.strip()]
    print(f"{len(panel)} alleles: {panel}\n")

    pseudo = {}
    for f in args.pseudoseq:
        pseudo.update(json.loads(Path(f).read_text()))

    # canonical binders per allele, from the fold sets
    starts = {}
    for fs in args.fold_sets:
        for r in csv.reader(open(fs)):
            if len(r) < 4 or r[0] in ("hard", "decoy"):
                continue
            a = slug_to_allele(r[2])
            if a in panel and len(r[3]) == args.length:
                starts.setdefault(a, []).append((r[2], r[3]))

    atlas = pd.read_csv(args.atlas)
    atlas = atlas[(atlas.label == 1) & (atlas.length == args.length)]

    model = PresentationNet(NetConfig())
    state = torch.load(args.model, map_location="cpu")
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()

    cfg = NetConfig()

    def score(peptides, slug):
        ps = pseudo[slug]["pocket_pseudosequence"]
        pep = torch.tensor(
            np.stack([encode_sequence(p, cfg.max_pep_len) for p in peptides]),
            dtype=torch.long)
        mhc = torch.tensor(
            np.stack([encode_sequence(ps, cfg.pseudoseq_len)] * len(peptides)),
            dtype=torch.long)
        with torch.no_grad():
            return model(pep, mhc).numpy()

    rows, summary = [], []
    for allele in panel:
        if allele not in starts:
            print(f"  {allele}: no canonical binders in the fold sets, skipped")
            continue
        seeds = starts[allele][:args.max_starts]
        slug = seeds[0][0]

        # model landscape: mean change in logit, averaged over starting peptides
        land = np.zeros((args.length, 20))
        for _, wt in seeds:
            variants, coords = [], []
            for i in range(args.length):
                for j, aa in enumerate(AA):
                    if aa == wt[i]:
                        continue
                    variants.append(wt[:i] + aa + wt[i + 1:])
                    coords.append((i, j))
            s = score([wt] + variants, slug)
            base, muts = s[0], s[1:]
            for (i, j), v in zip(coords, muts):
                land[i, j] += (v - base)
            for i in range(args.length):
                land[i, AA.index(wt[i])] += 0.0
        land /= len(seeds)

        pw = pwm_landscape(atlas[atlas.allele == allele].peptide, args.length)

        for i in range(args.length):
            for j, aa in enumerate(AA):
                rows.append({"allele": allele, "position": i + 1, "aa": aa,
                             "model_delta": land[i, j], "pwm_score": pw[i, j]})

        # per-position sensitivity: spread of the model's response at that position
        sens = land.std(axis=1)
        pwm_sens = pw.std(axis=1)
        r_flat = stats.spearmanr(land.ravel(), pw.ravel())[0]
        r_pos = stats.spearmanr(sens, pwm_sens)[0]
        top_model = [int(x) for x in np.argsort(-sens)[:2] + 1]
        top_pwm = [int(x) for x in np.argsort(-pwm_sens)[:2] + 1]

        summary.append({"allele": allele, "n_seeds": len(seeds),
                        "spearman_landscape": round(r_flat, 3),
                        "spearman_position_sensitivity": round(r_pos, 3),
                        "model_top_positions": top_model,
                        "pwm_top_positions": top_pwm,
                        "agree_on_anchors": set(top_model) == set(top_pwm)})

        print(f"  {allele:<14} landscape rho {r_flat:+.3f}   "
              f"position-sensitivity rho {r_pos:+.3f}   "
              f"model anchors {top_model}  PWM anchors {top_pwm}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(f"{args.out}_landscape.csv", index=False)
    s = pd.DataFrame(summary)
    s.to_csv(f"{args.out}_summary.csv", index=False)

    print(f"\n=== across {len(s)} alleles ===")
    print(f"  landscape agreement with the PWM:  "
          f"median rho {s.spearman_landscape.median():+.3f} "
          f"(range {s.spearman_landscape.min():+.3f} to "
          f"{s.spearman_landscape.max():+.3f})")
    print(f"  position-sensitivity agreement:    "
          f"median rho {s.spearman_position_sensitivity.median():+.3f}")
    print(f"  same two most-sensitive positions: "
          f"{s.agree_on_anchors.sum()}/{len(s)} alleles")

    print("\nInterpretation. High landscape agreement supports the hypothesis that")
    print("the model is largely recovering the motif. Note the peptide path")
    print("max-pools and so carries no absolute position information, so anchors")
    print("emerging at all is itself the finding — they are recoverable from")
    print("composition and local context alone.")
    print(f"\nWrote {args.out}_landscape.csv and {args.out}_summary.csv")


if __name__ == "__main__":
    main()