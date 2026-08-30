# pmhcpresent — HLA class I presentation prediction, with an equity lens

COMP0190 / AI4BH 2025–26. Predicts which peptides are presented by HLA class I, and
asks whether structural methods help where sequence methods are weakest — that is,
for the ancestrally diverse alleles that the training data underrepresents.

This README is the usage guide. For results, the reasoning behind design decisions,
and the corrections to analyses that turned out wrong, see
**[REPRODUCE.md](REPRODUCE.md)**.

---

## Install

Beta or any Linux box with a CUDA GPU. Analysis runs on CPU; only folding needs the
card.

```bash
git clone https://github.com/lakshaa-s/pmhc-present.git
cd pmhc-present
conda env create -f environment.yml
conda activate pmhcpresent
pip install -e ".[dev,struct,ml]"
pytest -q                       # ~30 s, should be all green
```

**ESMFold2 needs a second environment**, because `esm` requires Python >=3.12,<3.13
while `pmhcpresent` runs 3.13:

```bash
conda create -n esmfold2 python=3.12 -y
conda activate esmfold2
pip install esm torch biotite
```

ESMC-6B (24 GB in the HuggingFace cache) is a required ESMFold2 dependency despite
appearing nowhere in this codebase. Deleting it triggers a silent 24 GB re-download
on the next fold.

### Data you must supply

None of it is in git. Put these in place before running anything:

| path | what | where from |
|---|---|---|
| `data/raw/all_peptides.txt` | MHC Motif Atlas class I peptides, TSV | mhcmotifatlas.org |
| `data/pseudoseq/hla_{a,b,c}.json` | 34-mer pocket pseudosequence per allele | IPD-IMGT/HLA, via `histo.fyi` |
| `data/sequences/hla_{a,b,c}.json` | full canonical HLA sequences, for folding | same |
| `data/sequences/human_b2m.json` | β2-microglobulin | same |

---

## Running the pipeline

Five stages. Each writes files the next reads, so run them in order the first time.

### 1. Build the labelled dataset

```bash
python scripts/prepare_atlas.py \
  --input data/raw/all_peptides.txt \
  --output data/processed/atlas_labelled.csv \
  --neg-mode peptide-pool
```

**In:** Atlas peptide TSV.
**Out:** `atlas_labelled.csv` — 838,654 rows, 123 alleles, columns
`peptide, allele, label, length`.

Filters to classical HLA-A/B/C and 8–11mers, normalises allele names to
`HLA-A*02:01` form, labels presented peptides `1`, and generates length-matched
negatives.

**Two negative modes, and they are different experiments.** `peptide-pool` draws real
eluted ligands of *other* alleles, so the benchmark tests cross-allele motif
discrimination. `proteome` draws random windows from a human proteome FASTA and tests
presented-versus-not. The committed data was built with `peptide-pool`, and that
turns out to be the better choice: on proteome negatives a classifier using peptide
identity alone reaches AUROC 0.8801, because proteome windows never appear as anyone's
positive so peptide identity cleanly separates the classes. On deduplicated
peptide-pool the same prior reaches only 0.3596 — the same peptide carries both labels
under different alleles, which largely neutralises it. See "Known issues" below.

### 2. Make the train/validation split

```bash
python scripts/make_split.py \
  --data data/processed/atlas_labelled.csv \
  --out data/processed/split_val.csv
```

**In:** labelled table.
**Out:** `split_val.csv` — the validation `(allele, peptide)` pairs, 167,655 of them.

Near-duplicate-aware: peptides are Hamming-clustered within allele and whole clusters
assigned to one side, so a near-identical peptide cannot straddle the split.

**Every script must read this file rather than recomputing a split.** Recomputing
independently caused up to 42% leakage concentrated in positives before this existed,
and a later hash-ordering bug made two identical runs disagree by 86 rows. Both are
fixed; `tests/test_split_determinism.py` guards the second.

### 3. Train and evaluate the sequence model

```bash
python -m pmhcpresent train \
  --data data/processed/atlas_labelled.csv \
  --split data/processed/split_val.csv \
  --out models/rq1_baseline_split_v2.pt

python scripts/per_allele_auroc.py \
  --model models/rq1_baseline_split_v2.pt \
  --out results/per_allele_auroc.csv
```

**In:** labelled table, split.
**Out:** checkpoint, and per-allele AUROC for all 123 alleles.

The model (`src/pmhcpresent/models/nn.py`) is a small pan-allele CNN — two branches,
peptide and 34-mer pseudosequence, 30,465 parameters. It is a baseline rather than a
contribution, and deliberately untuned.

### 4. Build a benchmark fold set

```bash
# canonical binders: PWM top decile, then diversified
python scripts/select_fold_set_canonical.py \
  --data data/processed/atlas_labelled.csv \
  --val-split data/processed/split_val.csv \
  --pseudoseq data/pseudoseq/hla_{a,b,c}.json \
  --n-alleles 9 --k-peptides 12 --top-frac 0.10 \
  --out fold_sets/binders_v4.csv

# anchor-matched decoys: carry the target's anchors, score low overall
python scripts/select_decoys_hard.py \
  --data data/processed/atlas_labelled.csv \
  --binders fold_sets/binders_v4.csv \
  --max-pctile 25 \
  --out fold_sets/fold_set_v4.csv
```

**In:** labelled table, split, pseudosequences.
**Out:** `fold_set_v4.csv` — five columns, `tag,locus,allele_slug,peptide,note`,
where `tag` is `NA` for binders and `hard` for decoys.

Decoys are real ligands of *other* alleles that carry the target's anchor residues
but score low against its overall motif, so they cannot be rejected on anchors alone.
Against motif-mismatched decoys ESMFold2 scores 0.911; against these, 0.700 — about
two-thirds of the apparent structural signal was anchor recognition.

### 5. Fold, extract features, score

```bash
# ESMFold2 — ~20 s per complex
conda activate esmfold2
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u \
  scripts/fold_esmfold2.py --csv fold_sets/fold_set_v4.csv \
  --sequences data/sequences --out esmfold2-v4

# features
conda activate pmhcpresent
python scripts/analyse_pae.py esmfold2-v4 \
  --fold-set fold_sets/fold_set_v4.csv \
  --anchors data/processed/anchors.json \
  --out pae_esmfold2_v4.csv

# AUROC against the sequence baseline
python scripts/auroc_structure.py --pae pae_esmfold2_v4.csv \
  --sequence-csv results/sequence_v4.csv --out auroc_esmfold2_v4.csv
```

**In:** fold set, HLA sequences.
**Out:** one directory per complex containing PAE, structure and metrics; then a
feature table; then AUROCs.

**Runtimes, per complex on an RTX 4090.** ESMFold2 ~18 s; AF2 via HISTOFold ~22 s
(~90 s with representations saved); AlphaFold 3 ~72 s; fine-tuned AF2 ~4 s after a 40 s
compilation. So fold set v4's 216 complexes take roughly 1 h on ESMFold2 and 4 h on AF3.
Boltz-2.1 runs through the cloud API at about $0.05 per complex.

**Disk.** ESMFold2 writes ~0.8 MB per complex, AF2 via HISTOFold ~8.5 MB, AF3 ~9 MB. A
2,000-fold saturation-mutagenesis run is therefore 1.6 GB on ESMFold2 but 17 GB on AF2 —
check free space before starting, and consider extracting the PAE and deleting each
complex as it completes.

AlphaFold 3 and the fine-tuned AlphaFold need extra setup — see
[REPRODUCE.md](REPRODUCE.md), sections for 5 August and 6–10 August.

---

## Known issues

**The validation AUROC is inflated.** `negatives_peptide_pool` built its sampling pool
as a multiset, so a peptide observed for twelve alleles was drawn as a negative twelve
times as often. Peptide identity alone scores AUROC **0.248** on the validation set —
0.25 from chance, inverted. The pooled figure of 0.9732 should therefore not be quoted
as a clean estimate.

The fix is one line, and the data **has** been regenerated — as
`atlas_labelled_v2.csv` with model `rq1_baseline_split_v3.pt`, which gives a pooled
AUROC of 0.9715 against the old 0.9732. The original files are kept because the fold
sets were built from the old split and only 10.3% of it survives regeneration; see the
next issue. Note the fix reduces but does not remove the confound — the
peptide-identity prior moves from 0.248 to 0.3596, the residual coming from
cross-allele crossover that deduplication cannot touch.

*This does not reach the fold-set results.* Fold-set decoys are anchor-matched or
affinity-measured, never drawn from the peptide pool, so RQ1, RQ2, RQ3, the fine-tuned
comparison and the external baselines are unaffected. The HLA-C finding also survives:
HLA-C is the *least* confounded locus (0.202 against HLA-A 0.235, HLA-B 0.249), so the
artefact inflated the others more, and the effect attenuates rather than disappearing —
OLS with confound in the model gives −0.0267 (p < 0.0001), a matched comparison gives
−0.0269 (p 0.0002), and it reproduces on fold set v2. Per-locus means are also
essentially unchanged between the two models (HLA-A 0.970→0.968, HLA-B 0.975→0.976,
HLA-C 0.940→0.941).

**The per-allele results use a different model from the fold-set results.**
`rq1_baseline_split_v3.pt` is trained on the deduplicated data and used for the pooled
and per-allele AUROC; `rq1_baseline_split_v2.pt` is used for everything computed on the
fold sets, which were built from the older split. Only 10.3% of the validation split
survives regeneration, so the two cannot be merged without rebuilding the benchmarks and
refolding 360 complexes across five architectures. Neither model is evaluated on its own
training data.

**85.3% of unique validation peptides also appear in training under some allele.** The
split clusters within allele, so a peptide can be a training example for one allele and
a validation example for another. Intentional — a different groove is a different
prediction problem — but worth knowing.

**The fold sets are PWM-separable by construction.** Binders are the PWM's top decile,
so a PWM alone separates fold set v2 at AUROC 1.000. Any sequence-model figure on these
sets must be read with that in mind, though our model sits 0.07–0.08 *below* the PWM on
both selected sets, which is the evidence it is not simply recovering the criterion.
The structural models never see the Atlas and are unaffected.

---

## Results in brief

Full tables, confidence intervals and caveats are in [REPRODUCE.md](REPRODUCE.md).

| | | |
|---|---|---|
| **RQ1** | Do structural models beat sequence models? | **No.** Sequence 0.930 against AF3 0.858, AF2 0.842, ESMFold2 0.805, Boltz 0.745 on fold set v4. Every paired margin excludes zero. Five feature readouts, five architectures including a fine-tuned one. |
| **RQ2** | Do they combine synergistically? | **No.** Nine configurations, every interval spanning zero. The models do fail on partly different complexes (margin rho +0.223 against a 0.143 chance baseline), so the null is not simple redundancy. |
| **RQ3** | Have the two learned the same binding biology? | **Partly.** Both recover anchor positions; the sequence model's top-2 fall inside the IC-derived anchor set for 6/7 alleles. For HLA-B\*08:01 both structural models rank P5 first where the sequence model does not — but the same analysis on HLA-B\*37:01, which has the *highest* P5 information content of the group, puts P5 sixth of nine. They detect one P5 mechanism, not P5 anchoring in general. |

Two findings independent of the research questions:

**The data gap is in negatives, not positives.** Across all of IEDB (5,770,781 assay
rows), a studied HLA-A allele has roughly one experimentally determined non-binder per
six positives; HLA-B and HLA-C have roughly one per 120. For HLA-C\*15:05 and
HLA-C\*16:02 there are **zero**. Constructed decoys are a necessity rather than a
convenience, and the equity question cannot currently be answered with experimentally
grounded negatives by anyone.

**Anchor conventions are too rigid.** 43% of alleles have a high-information position
outside the standard P2/PΩ scheme. Defining anchors per allele from information
content gives the best structural feature across six independent measurements.

---

## Script reference

**Data and sequence model**
- `prepare_atlas.py` — atlas positives → labelled table with generated negatives
- `make_split.py` — the shared train/validation split; every script reads this file
- `per_allele_auroc.py`, `plot_per_allele.py` — per-allele AUROC across 123 alleles
- `derive_anchors.py` — per-position information content; the 43% survey
- `ablation_a2.py`, `ablation_a2_condB.py`, `ablation_family_condB.py` — starvation
  and family-removal ablations

**Panel and fold-set construction**
- `select_allele_panel.py` — panel v1–v3 (AUROC and anchor-IC stratification)
- `select_allele_panel_motif.py` — panel v4, by Jensen-Shannon motif isolation
- `select_rq3_alleles.py` — RQ3 alleles as representatives of motif classes
- `select_fold_set_canonical.py` — motif-typical binders, PWM top decile
- `select_decoys_hard.py` — anchor-matched adversarial decoys (`--max-pctile 25`)
- `select_fold_set_affinity.py` — PWM-free fold set from measured affinities
- `build_rq3_variants.py` — saturation-mutagenesis variant sets

**Folding**
- `fold_esmfold2.py` — ESMFold2, writes structures and PAE
- `build_af3_inputs.py` — AlphaFold 3 JSON inputs with per-chain MSAs
- `build_finetune_targets.py` — `alphafold_finetune` targets file
- HISTOFold is a patched third-party fork; see `patches/`

**Structural features**
- `analyse_pae.py`, `analyse_pae_af2.py`, `analyse_pae_af3.py` — anchor-localised PAE
- `extract_confidence.py` — global and localised confidence, all architectures
- `extract_geometry.py`, `extract_geometry_af2.py` — interface geometry
- `rq1_embeddings.py` — learned representations with a light classifier
- `structural_consistency.py` — consistency features; a fifth readout
- `auroc_structure.py` — features as classifiers, with direction handling

**Baselines, ensembles and controls**
- `score_netmhcpan.py`, `score_mhcflurry.py`, `score_mixmhcpred.py`
- `score_sequence_on_foldset.py` — our model on a fold set, with leakage reporting
- `rq2_stack.py`, `rq2_ensemble_alt.py`, `rq2_gate.py` — the ensemble configurations
- `rq2_error_overlap.py` — do the model families fail on the same complexes?
- `bootstrap_auroc.py` — bootstrap CIs and paired differences
- `fold_quality_control.py` — does confidence predict which complexes are wrong?
- `check_motmaen_overlap.py` — training-set overlap for the fine-tuned comparison

**RQ3**
- `rq3_sequence_landscape.py` — sequence-model saturation mutagenesis vs the PWM
- `rq3_compare_landscapes.py` — structural landscapes, seed stability, anchor recovery
- `rq3_shap.py` — exact Shapley attribution, compared against the landscapes

**Dataset audit**
- `audit_dataset.py` — which negative mode produced the file; cross-allele crossover
- `crossover_label_balance.py` — the peptide-identity-only classifier
- `confound_vs_per_allele.py` — does the confound reach the per-allele distribution?
- `hlac_partial_effect.py` — does the HLA-C effect survive controlling for it?
- `foldset_survival_check.py` — does the finding reproduce on fold set v2?

Every script has a module docstring giving its purpose, method, inputs and outputs;
run any of them with `--help` for arguments.

---

## Where things run

| Stage | Machine | Environment |
|---|---|---|
| Data prep, sequence model, all analysis | Beta (RTX 4090) | `pmhcpresent` |
| ESMFold2 folding | Beta | `esmfold2` (Python 3.12) |
| AF2 via HISTOFold | Beta | ColabFold 1.5.5 Singularity image |
| AlphaFold 3 | Beta | Singularity image from `patches/af3.def` |
| Fine-tuned AF2 (Motmaen et al.) | Beta | ColabFold image + `patches/alphafold_finetune_modernise.patch` |
| Boltz-2.1 (cloud API) | Mac | `uv` project |
| Overflow and large downloads | CS lab machines | local `/tmp` (home quota is 10 GB) |

Beta's `/tmp` is tmpfs — RAM-backed, 63 GB, cleared on reboot. Useful when `/home`
is full, but copy anything you need to keep.

---

## Data governance

TRACERx is **controlled-access** (Data Access Committee) and is used as an
*illustrative application* only — never as a benchmark or evaluation cohort.
**Nothing TRACERx-derived enters git**; see `.gitignore`.

AlphaFold 3 output may not be used to train models intended for commercial
application under its weights terms of use. AF3 features therefore appear in
evaluation and in the RQ2 ensemble probes, which are fitted only to measure whether
structural and sequence signals combine, and are neither retained, deployed nor
distributed. No AF3-derived model leaves this repository.

NetMHCpan-4.1 is used through a colleague's licensed
installation; this project is not licensed independently.
