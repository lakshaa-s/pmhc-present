# REPRODUCE.md — how to regenerate every result

A map from each result to the exact script + command that produces it, plus where the
data and outputs live. Goal: nothing is a one-off; every number/figure is re-testable.

> **Verify before trusting:** some paths/filenames below are reconstructed from working
> notes — check the `# TODO/verify` tags and correct anything that doesn't match your tree.

---

## Environment

- **Compute:** Beta (Linux, RTX 4090, CUDA). Conda env `pmhcpresent`.
  `conda activate pmhcpresent` before running anything.
- **Repo:** `~/pmhc-present` (GitHub: `lakshaa-s/pmhc-present`).
- **Structure folding:** runs on the **Mac** via the Boltz API (folder
  `~/Downloads/boltz_prediction`), not on Beta. Outputs land in `boltz-experiments/`.

---

## Data pipeline

| Result | Script | Command | Output |
|---|---|---|---|
| Labelled dataset (838k rows, 123 alleles) | `scripts/prepare_atlas.py` | see command below | `data/processed/atlas_labelled.csv` |

```
python scripts/prepare_atlas.py \
  --input data/raw/all_peptides.txt \
  --output data/processed/atlas_labelled.csv \
  --neg-mode peptide-pool \
  --ratio 1.0 --min-len 8 --max-len 11 --seed 42
```

Flags: `--input` = raw Atlas `all_peptides.txt`; `--output` = labelled CSV;
`--neg-mode {proteome,peptide-pool}` (current dataset used **peptide-pool**; swap to
`proteome --proteome <human_proteome.fasta>` later — nothing downstream changes);
`--ratio` negatives per positive (default 1.0); `--min-len`/`--max-len` peptide length
window (8–11); `--seed` for reproducible negative sampling.
<!-- verify: exact --input path of the raw atlas file, and the seed you actually used -->

The script filters to classical HLA A/B/C, normalises alleles (`A0201` → `HLA-A*02:01`,
matching the pseudoseq loader's canonical key), labels atlas rows as positives, and
generates length-matched negatives per allele. Output columns: `peptide, allele, label, length`.

- Source: MHC Motif Atlas (`all_peptides.txt`), filtered to classical HLA A/B/C, 8–11mers;
  1:1 negatives (peptide-pool mode; proteome-sampled is the planned upgrade).
- Pseudosequences: `data/pseudoseq/hla_{a,b,c}.json` (Chris's pocket-pseudoseq JSONs).

---

## Sequence model — RQ1 baseline

| Result | How |
|---|---|
| **Baseline AUROC ~0.974, equity gap ~0.007** | `pmhcpresent train --data data/processed/atlas_labelled.csv --pseudoseq data/pseudoseq/hla_a.json data/pseudoseq/hla_b.json data/pseudoseq/hla_c.json --epochs 50 --save models/rq1_baseline_hamming.pt` |

- Split: near-duplicate-aware (`hamming_cluster` in `src/pmhcpresent/eval/splits.py`).
- Robustness: AUROC held (0.973 → 0.974) when tightening exact-dedup → Hamming split.
- Per-bin AUROC (Hamming): rare 0.965 / low 0.974 / medium 0.971 / high 0.978 / very_high 0.973.

---

## Per-allele distribution (the HLA-C equity result)

| Result | Script | Command |
|---|---|---|
| **Per-allele AUROC, all 123 alleles; HLA-C median 0.951 vs A/B ~0.98** | `scripts/per_allele_auroc.py` | `python scripts/per_allele_auroc.py --data data/processed/atlas_labelled.csv --pseudoseq data/pseudoseq/hla_a.json data/pseudoseq/hla_b.json data/pseudoseq/hla_c.json --model models/rq1_baseline_hamming.pt --out results/per_allele_auroc.csv` |
| **Locus distribution plot** | `scripts/plot_per_allele.py` | `python scripts/plot_per_allele.py results/per_allele_auroc.csv` → `per_allele_dist.png` |

- Key finding: HLA-C underperforms regardless of data volume (e.g. C\*12:03 has 3,026
  peptides and still low) → motif-isolation, not data quantity, drives the gap.
- Worst 5: C\*15:05 (0.889), B\*14:01 (0.902), C\*16:02 (0.902), C\*12:04 (0.924), C\*12:03 (0.927).

---

## Ablations — the orphan-allele mechanism

**Condition A — A\*02:01 dose-response** (starve one allele, keep the rest):
```
python scripts/ablation_a2.py \
  --data data/processed/atlas_labelled.csv \
  --pseudoseq data/pseudoseq/hla_a.json data/pseudoseq/hla_b.json data/pseudoseq/hla_c.json \
  --out results/ablation_a2.csv
```
Result: flat curve — AUROC ~0.96 even at 115 examples → cross-allele transfer.

**Condition B — 2×2 (starve × remove family).** A\*02 uses the fixed-prefix script;
A\*03 and B\*27 use the generalised regex script:
```
# A*02 (collapses):
python scripts/ablation_a2_condB.py \
  --data data/processed/atlas_labelled.csv \
  --pseudoseq data/pseudoseq/hla_a.json data/pseudoseq/hla_b.json data/pseudoseq/hla_c.json \
  --out results/ablation_a2_condB.csv

# A*03/A*11 supertype (does NOT collapse):
python scripts/ablation_family_condB.py \
  --data data/processed/atlas_labelled.csv \
  --pseudoseq data/pseudoseq/hla_a.json data/pseudoseq/hla_b.json data/pseudoseq/hla_c.json \
  --target "HLA-A*03:01" --family-regex '^HLA-A\*(03|11):' \
  --out results/ablation_a3_condB.csv

# B*27 family (does NOT collapse):
python scripts/ablation_family_condB.py \
  --data data/processed/atlas_labelled.csv \
  --pseudoseq data/pseudoseq/hla_a.json data/pseudoseq/hla_b.json data/pseudoseq/hla_c.json \
  --target "HLA-B*27:05" --family-regex '^HLA-B\*27:' \
  --out results/ablation_b27_condB.csv
```
Result across 4 alleles: only **A\*02:01 collapses** (0.96 → 0.90) when starved AND family
removed. A\*03, B\*27 don't. Refined claim: collapse needs subtle motif + genuine isolation.

Alleles removed per experiment: A\*02 = 9 family members; A\*03/A\*11 = 3; B\*27 = 2.

---

## Structure phase (Boltz) — RUNS ON THE MAC

Boltz folds via API; outputs in `~/Downloads/boltz_prediction/boltz-experiments/`.
API key in `.env` (gitignored — never commit).

**1. Select a diverse fold set** (on Beta — needs atlas + pseudoseqs):
```
python scripts/select_fold_set.py --data data/processed/atlas_labelled.csv \
  --pseudoseq data/pseudoseq/hla_a.json data/pseudoseq/hla_b.json data/pseudoseq/hla_c.json \
  --n-alleles 5 --k-peptides 2 --out fold_set_pilot.csv
```
Selects max-diversity alleles (seeded with covered anchors + orphan HLA-C).

**2. Select decoys** (motif-distant peptides as negatives):
```
python scripts/select_decoys.py \
  --data data/processed/atlas_labelled.csv \
  --pseudoseq data/pseudoseq/hla_a.json data/pseudoseq/hla_b.json data/pseudoseq/hla_c.json \
  --out decoy_set.csv
```

**3. Fold** (on Mac, in the Boltz folder): copy the CSV to `complexes/hla_class_i.csv`, then
`uv run boltz_pmhc_class_i.py`. Each pMHC ≈ $0.05.

**4. Extract features** (Mac): `python3 scripts/extract_boltz_features.py boltz-experiments --out boltz_features.csv`
→ per-fold iptm, complex_iplddt, complex_plddt, pae_interface, pae_mean.

**5. Per-residue anchor PAE analysis** (Mac): `python3 scripts/analyse_pae.py boltz-experiments --out pae_analysis.csv`
→ binder-vs-decoy anchor PAE per allele.

**Alleles folded (pilot, 6 total):** covered = B\*27:09, B\*27:05, A\*02:01, B\*07:02;
orphan = C\*15:05, C\*16:02. Each with real binders + motif-distant decoys.

**Result (negative, tested two ways):** Boltz confidence (iptm) and anchor PAE do **not**
discriminate binders from decoys — decoys score as confidently as binders, for covered and
orphan alleles alike. Fold confidence ≠ binding. Consistent with King et al. 2025
(arXiv:2512.06592): Boltz-2 underperforms sequence for affinity even fine-tuned.

---

## Where things live

- **Code:** all in `~/pmhc-present` (committed). Structure analysis scripts in `scripts/`.
- **Data:** `data/processed/atlas_labelled.csv`, `data/pseudoseq/*.json` (Beta; gitignored).
- **Models:** `models/*.pt` (Beta; gitignored).
- **Sequence results:** `results/*.csv` (Beta; gitignored — BACK UP separately).
- **Boltz outputs:** `boltz-experiments/` (Mac only — BACK UP to Drive; they cost credits).

---

## Known caveats / TODO

- Boltz pilot is small (5 alleles, ~2 binders + 2 decoys each) — pilot signal, not a benchmark.
- Boltz folds used `num_samples: 1` — meeting action: re-run with multiple samples, check the
  confidence *distribution* (single folds may be noisy).
- For a fair RQ1 comparison, structure & sequence should use the **same** positives/negatives.
- Still to test: geometric features (contacts/pockets from the .cif), AF2 (Chris's tuned-MSA
  code; needs NVIDIA Container Toolkit on Beta — not yet installed), ESMFold (free API).
- `data/` and `models/` are gitignored, so a fresh clone can't reproduce without them —
  document where to obtain/regenerate the Atlas download and pseudoseq JSONs.