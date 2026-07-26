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

**1. Select canonical binders** (on Beta — needs atlas + pseudoseqs):
```
python scripts/select_fold_set_canonical.py \
  --data data/processed/atlas_labelled.csv \
  --pseudoseq data/pseudoseq/hla_a.json data/pseudoseq/hla_b.json data/pseudoseq/hla_c.json \
  --peptide-length 9 --n-alleles 5 --k-peptides 6 \
  --out fold_sets/fold_set_9mer_canonical_k6.csv
```
Alleles are chosen by max-min pocket-pseudosequence distance (seeded with covered anchors
and orphan HLA-C). Peptides are scored against a PWM built from that allele's own atlas
positives, and the top decile is then diversified — so binders are motif-typical rather
than maximally diverse. **This supersedes `select_fold_set.py`** (max-diversity), which
selects motif-atypical peptides and is unsuitable for a discrimination test.

**2. Select decoys** (motif-mismatched, anchor-rejected):
```
python scripts/select_decoys_clean.py \
  --data data/processed/atlas_labelled.csv \
  --pseudoseq data/pseudoseq/hla_a.json data/pseudoseq/hla_b.json data/pseudoseq/hla_c.json \
  --peptide-length 9 --k-decoys 6 \
  --out fold_sets/decoy_set_9mer_clean_k6.csv
```
Candidates come from the 3 most motif-distant donor alleles, then are rejected if they
score above the 25th percentile of the target's real binders **or** carry any of the
target's top-4 preferred residues at P2 / C-terminus. **This supersedes
`select_decoys.py`**, which used allele-level distance only and leaked anchor-carrying
peptides into the decoy set.

Combine into one fold set:
```
cat fold_sets/fold_set_9mer_canonical_k6.csv fold_sets/decoy_set_9mer_clean_k6.csv \
  > fold_sets/fold_set_60.csv
```

**3. Fold** (on Mac, in the Boltz folder): copy the CSV to `complexes/hla_class_i.csv`, then
`uv run boltz_pmhc_class_i.py`. Each pMHC ≈ $0.05.

**4. Extract features** (Mac): `python3 scripts/extract_boltz_features.py boltz-experiments --out boltz_features.csv`
→ per-fold iptm, complex_iplddt, complex_plddt, pae_interface, pae_mean.

**5. Per-residue anchor PAE analysis** (Mac): `python3 scripts/analyse_pae.py boltz-experiments --out pae_analysis.csv`
→ binder-vs-decoy anchor PAE per allele.

**6. Interface geometry** (Mac): `python3 scripts/extract_geometry.py boltz-experiments --out geometry_features.csv`
→ peptide-MHC contact counts, anchor contacts, anchor-pocket distances (needs biotite:
`pip install biotite --break-system-packages`).

**7. AUROC vs the sequence baseline** (Mac): `python3 scripts/auroc_structure.py --pae pae_analysis_k6.csv --geometry geometry_k6.csv --out structure_auroc.csv`
→ treats each structural feature as a binding score and computes AUROC over binder/decoy
labels, so it is directly comparable to the sequence model's AUROC.

**Fold sets (all 9mers, for HISTOFold compatibility):** 5 alleles spanning covered→orphan
(B\*27:05, A\*02:01, B\*07:02, C\*15:05, C\*16:02), 6 canonical binders + 6 motif-mismatched
decoys each = 60 complexes. Binders are top-decile by the allele's own PWM; decoys are
rejected if they carry the target's preferred anchor residues (see script docstrings).

**Result — anchor-localised PAE discriminates; confidence and geometry do not.**

| Feature | Pooled AUROC (n=60) |
|---|---|
| `pae_anchors` (P2 + C-term) | **0.783** |
| `pae_anchorC` | 0.759 |
| `pae_anchor2` | 0.737 |
| `pae_pep_mhc` (whole interface) | 0.694 |
| contact/geometry features | 0.21 – 0.41 |
| `iptm` | ~flat 0.98–0.99 for binders and decoys alike |

Signal *increases* as the metric localises to the anchor positions (0.694 whole-interface
→ 0.783 anchors), which matches anchor-dominated binding biology. Geometry features sit
*below* 0.5 — decoys tend to make slightly more contacts — so contacts are not a binding
proxy.

Per-allele `pae_anchors` AUROC: C\*15:05 **0.972**, B\*07:02 0.917, C\*16:02 0.750,
B\*27:05 0.722, A\*02:01 **0.639**. Note the inverse relationship with the sequence model:
A\*02:01 is the sequence model's best allele and structure's worst; C\*15:05 is the
sequence model's worst (0.889) and structure's best. **Structure appears strongest exactly
where sequence is weakest** — the complementarity that RQ2 asks about, consistent with
King et al. 2025 (arXiv:2512.06592), who found structure and sequence embeddings combine
most usefully where the sequence model is weak.

**[CORRECTION — supersedes an earlier finding.]** An initial pilot concluded that *no*
Boltz signal discriminated. That was an artifact of the fold set, not of Boltz: the
max-diversity peptide selection had picked motif-atypical "binders" (e.g. `LVAKVRALD`
assigned to B\*07:02 with neither anchor), and allele-level distance alone let
anchor-carrying peptides into the decoy set (e.g. `QRSRFIVVV`, P2-Arg, as a B\*27:05
"decoy"). Rebuilt with canonical binders and anchor-rejected decoys, the anchor-PAE signal
appears clearly. The confidence (`iptm`) and geometry negatives survive the correction.

**Caveats.** 6 binders + 6 decoys per allele — the pooled figure and the consistency of
direction across five alleles are more reliable than any single per-allele AUROC. 0.783 is
well below the sequence model's ~0.97, so structure alone does **not** outperform sequence
(RQ1 remains negative); the interest is in complementarity (RQ2). The two AUROCs are also
not strictly comparable: the sequence figure is over a large held-out set with pooled
negatives, this one over 60 designed complexes with motif-mismatched decoys.

---

## Where things live

- **Code:** all in `~/pmhc-present` (committed). Structure analysis scripts in `scripts/`.
- **Data:** `data/processed/atlas_labelled.csv`, `data/pseudoseq/*.json` (Beta; gitignored).
- **Models:** `models/*.pt` (Beta; gitignored).
- **Sequence results:** `results/*.csv` (Beta; gitignored — BACK UP separately).
- **Boltz outputs:** `boltz-experiments/` (Mac only — BACK UP to Drive; they cost credits).

---

## Known caveats / TODO

- Fold sets are 6 binders + 6 decoys per allele across 5 alleles — enough to establish
  direction, not enough for tight per-allele estimates. Scaling to more peptides per
  allele (and more alleles) is the obvious next step.
- Boltz folds used `num_samples: 1` — meeting action: re-run with multiple samples, check
  the confidence *distribution* (single folds may be noisy).
- Binder/decoy selection is sensitive to how "binder" and "decoy" are defined: the first
  pilot's null result was caused by motif-atypical binders and anchor-carrying decoys.
  Any change to the selection scripts warrants re-checking the discrimination result.
- For a fair RQ1 comparison, structure & sequence should use the **same** positives/negatives.
  Currently they do not: the sequence AUROC is over a large held-out set with pooled
  negatives, the structural AUROC over 60 designed complexes with motif-mismatched decoys.
- Remaining structural avenues:
    - **AF2 / HISTOFold** (Chris's tuned MSAs, github.com/drchristhorpe/HISTOFold) — needs
      Docker + NVIDIA Container Toolkit on Beta (sudo available; not yet installed).
      Currently 9mer-only; Chris is extending it to 8-11mers.
    - **ESMFold2** — API key received from Chris; ~100 predictions/day. Use the **'fast'**
      variant (Chris's benchmarking found it better on pMHC, especially peptide confidence
      for unseen peptides). Mind the structure-tokens-per-minute limit; rate-limiting code
      at github.com/drchristhorpe/esmfold2_benchmarking.
    - **Structure embeddings** (King et al. lead) — do learned representations carry
      *complementary* signal to sequence in the weak/orphan regime (RQ2)?
- `data/` and `models/` are gitignored, so a fresh clone can't reproduce without them —
  document where to obtain/regenerate the Atlas download and pseudoseq JSONs.