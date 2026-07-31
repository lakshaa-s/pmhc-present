"""Fold peptide-MHC class I complexes with ESMFold2 (local, on Beta's GPU).

Cross-model replication of the Boltz experiment: the SAME fold-set CSV is used, the same
three-chain layout (A = MHC heavy chain, B = beta-2-microglobulin, C = peptide), and the
outputs are written in the SAME directory layout Boltz produced, so the existing
`analyse_pae.py`, `extract_geometry.py` and `auroc_structure.py` scripts run against it
unchanged.

Per fold it writes, under `{out}/{tag}__{allele_slug}__{peptide}/outputs/files/prediction/`:
  metrics.json                      ptm, iptm, mean plddt (+ per-chain-pair iptm)
  sample_0_pae.npz                  key 'pae' — the (L, L) predicted aligned error
  sample_0_predicted_structure.cif  the folded complex
  embeddings.npz                    sequence + pooled-pair embeddings (for RQ2)

Set --num-diffusion-samples > 1 to fold each complex several times; per-sample metrics are
recorded in metrics.json under "all_sample_results", so the spread of confidence can be
inspected rather than a single point estimate (ESMFold2 applies fresh LM dropout per fold,
so repeats are genuinely diverse).

Run in the `esmfold2` conda env:
    conda activate esmfold2
    python scripts/fold_esmfold2.py --csv fold_sets/fold_set_60.csv \
        --sequences ~/Downloads/boltz_prediction/sequences --out esmfold2-experiments
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch


def load_sequences(seq_dir: Path):
    """Chris's JSONs: hla_{a,b,c}.json keyed by slug, plus human_b2m.json."""
    with open(seq_dir / "human_b2m.json") as fh:
        b2m = json.load(fh)["canonical_sequence"]
    loci = {}
    for locus in ("a", "b", "c"):
        with open(seq_dir / f"hla_{locus}.json") as fh:
            loci[f"hla-{locus}"] = json.load(fh)
    return b2m, loci


def write_outputs(out_dir: Path, results, structures, model_name):
    """Mirror the Boltz output layout so downstream scripts work unchanged."""
    pred_dir = out_dir / "outputs" / "files" / "prediction"
    pred_dir.mkdir(parents=True, exist_ok=True)

    def sample_metrics(r):
        m = {
            "ptm": float(r.ptm),
            "iptm": float(r.iptm),
            "complex_plddt": float(np.mean(np.asarray(r.plddt))),
        }
        pc = getattr(r, "pair_chains_iptm", None)
        if pc is not None:
            m["pair_chains_iptm"] = np.asarray(pc).tolist()
        return {"metrics": m}

    best = results[0]
    with open(pred_dir / "metrics.json", "w") as fh:
        json.dump({
            "model": model_name,
            "best_sample": sample_metrics(best),
            "all_sample_results": [sample_metrics(r) for r in results],
        }, fh, indent=2)

    # PAE under the same key Boltz used, so analyse_pae.py reads it as-is
    np.savez_compressed(pred_dir / "sample_0_pae.npz",
                        pae=np.asarray(best.pae, dtype=np.float32))

    # embeddings for the RQ2 / King-style combined-representation work
    emb = {}
    for attr in ("output_embedding_sequence", "output_embedding_pair_pooled"):
        v = getattr(best, attr, None)
        if v is not None:
            emb[attr] = np.asarray(v, dtype=np.float32)
    if emb:
        np.savez_compressed(pred_dir / "embeddings.npz", **emb)

    if structures is not None:
        # to_mmcif() returns a string; it does not take a path. Not wrapped in a
        # try/except: a silent handler here hid two separate bugs (wrong attribute
        # name, then wrong call signature), so a failure should be loud.
        (pred_dir / "sample_0_predicted_structure.cif").write_text(
            structures[0].to_mmcif())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True,
                    help="fold set: pdb_code,locus,allele_slug,peptide,resolution")
    ap.add_argument("--sequences", required=True,
                    help="dir with hla_{a,b,c}.json and human_b2m.json")
    ap.add_argument("--out", default="esmfold2-experiments")
    ap.add_argument("--model", default="biohub/ESMFold2",
                    help="use biohub/ESMFold2-Fast for the fast variant "
                         "(Chris's benchmarking favoured it on pMHC)")
    ap.add_argument("--num-diffusion-samples", type=int, default=1)
    ap.add_argument("--num-loops", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from esm.models.esmfold2 import (
        ESMFold2InputBuilder,
        ProteinInput,
        StructurePredictionInput,
    )
    from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model

    seq_dir = Path(args.sequences).expanduser()
    b2m_seq, loci = load_sequences(seq_dir)

    rows = []
    with open(args.csv) as fh:
        for row in csv.reader(fh):
            if len(row) >= 4:
                rows.append(row[:5])
    print(f"{len(rows)} complexes in {args.csv}")

    print(f"loading {args.model} ...")
    model = ESMFold2Model.from_pretrained(args.model).cuda().eval()
    builder = ESMFold2InputBuilder()
    out_root = Path(args.out)

    done = skipped = failed = 0
    for i, (pdb, locus, slug, peptide, *_rest) in enumerate(rows, 1):
        name = f"{pdb}__{slug}__{peptide}"
        out_dir = out_root / name
        if (out_dir / "outputs" / "files" / "prediction" / "metrics.json").exists():
            print(f"[{i}/{len(rows)}] {name}: already folded, skipping")
            skipped += 1
            continue

        entry = loci.get(locus, {}).get(slug)
        if entry is None:
            print(f"[{i}/{len(rows)}] {name}: allele {slug} not in {locus} JSON, skipping")
            failed += 1
            continue
        mhc_seq = entry["canonical_sequence"]

        spi = StructurePredictionInput(sequences=[
            ProteinInput(id="A", sequence=mhc_seq),
            ProteinInput(id="B", sequence=b2m_seq),
            ProteinInput(id="C", sequence=peptide),
        ])

        print(f"[{i}/{len(rows)}] {name}: folding "
              f"({len(mhc_seq)}+{len(b2m_seq)}+{len(peptide)} residues)...", flush=True)
        try:
            with torch.no_grad():
                res = builder.fold(
                    model, spi,
                    num_loops=args.num_loops,
                    num_diffusion_samples=args.num_diffusion_samples,
                    seed=args.seed,
                    complex_id=name,
                )
        except Exception as e:  # noqa: BLE001 - one bad fold shouldn't kill the batch
            print(f"    FAILED: {e}")
            failed += 1
            continue

        results = res if isinstance(res, list) else [res]
        write_outputs(out_dir, results, [r.complex for r in results if getattr(r, "complex", None) is not None], args.model)
        m = results[0]
        print(f"    ok  iptm={float(m.iptm):.4f}  ptm={float(m.ptm):.4f}  "
              f"plddt={float(np.mean(np.asarray(m.plddt))):.4f}")
        done += 1

    print(f"\nfolded {done}, skipped {skipped}, failed {failed} -> {out_root}")
    print("Next: python scripts/analyse_pae.py", out_root, "--out pae_esmfold2.csv")


if __name__ == "__main__":
    main()