# pmhcpresent — HLA class I presentation prediction, with an equity lens

COMP0190 / AI4BH 2025–26. Predicts which peptides are presented by HLA class I, and
asks whether structural methods help where sequence methods are weakest — that is,
for the ancestrally diverse alleles that the training data underrepresents.

Exact commands for every result below are in **[REPRODUCE.md](REPRODUCE.md)**, which
also records the reasoning behind design decisions and the corrections to analyses
that turned out wrong.

## Research questions

| | | status |
|---|---|---|
| **RQ1** | Do structural models beat sequence models for pMHC presentation? | answered — no |
| **RQ2** | Do sequence and structure combine synergistically? | answered — no |
| **RQ3** | Does saturation mutagenesis show the two learned the same binding biology? | in progress |

Both answers are negative and both are better powered than the positive claims in
the literature they contradict.

## RQ1 — sequence beats every structural model

Fold set v4: 216 complexes, 9 motif-isolated alleles, 12 canonical binders and 12
anchor-matched decoys each. Per-allele z-scored anchor PAE, paired bootstrap.

| model | AUROC | sequence − structure |
|---|---|---|
| **sequence (ours)** | **0.930** | — |
| AlphaFold 3 | 0.858 | +0.073 [+0.022, +0.125] |
| AlphaFold 2 (HISTOFold) | 0.842 | +0.088 [+0.031, +0.149] |
| ESMFold2 | 0.805 | +0.124 [+0.066, +0.187] |
| Boltz-2.1 | 0.745 | +0.185 [+0.119, +0.257] |

Every interval excludes zero, and the ordering tracks model recency exactly — AF3
has roughly halved Boltz's deficit, but the newest and strongest still loses.
Per-allele z-scoring is transductive and so flatters structure; raw pooled figures
are 0.765 / 0.707 / 0.659 / 0.646, and the conclusion is unchanged.

**It is not an artefact of which output we read.** Four independent readouts of the
same folds, on fold set v2 (144 complexes, 6 alleles, sequence 0.921):

| readout | AUROC |
|---|---|
| learned representations (frozen AF2 embeddings + classifier) | 0.834 |
| predicted aligned error, anchor-localised | 0.804 |
| confidence metrics (peptide-region pLDDT) | 0.753 |
| interface geometry (contacts, burial, anchor distances) | 0.492 |

**Fold quality is not the ceiling**, on four independent lines: confidence does not
distinguish correctly from incorrectly ordered complexes within a model; accuracy
and discrimination are inversely ordered across models; improved MSAs give better
structures but no detectable change in discrimination; and King et al.
(arXiv:2512.06592) found the same by retraining on experimental structures.

**Fine-tuning does not rescue it on a hard benchmark.** Motmaen et al. (PNAS 2023)
published fine-tuned parameters; the overlap against our fold sets is clean (0 exact
allele–peptide pairs across 360 complexes).

| | our fold set v4 | their published test set |
|---|---|---|
| fine-tuned AlphaFold | 0.685 | 0.967 |
| vanilla AlphaFold | 0.698 | 0.877 |
| NetMHCpan-4.1 | — | 0.985 |

Their column reproduces at 0.967, so the difference is the benchmark rather than the
pipeline: their decoys differ from the binder by 4–5 substitutions in 91% of cases,
ours are anchor-matched. Fine-tuning gains +0.090 on their set and −0.013 on ours,
so its benefit is a function of decoy difficulty. Note also that in their own
evaluation a sequence method scores above the fine-tuned structural one.

## RQ2 — no combination strategy helps

Eight configurations, two panels, four architectures, four feature types. Every
paired-bootstrap interval spans zero.

| configuration | Δ vs sequence |
|---|---|
| linear stack, v2, 21 features | −0.046 [−0.105, +0.011] |
| linear stack, v2, AF2 PAE only | +0.006 [−0.043, +0.056] |
| linear stack, v4, three architectures | −0.022 [−0.061, +0.017] |
| **linear stack, v4, per-allele z-scored** | **−0.001 [−0.037, +0.037]** |
| linear stack, v4, mixed output types | −0.024 [−0.063, +0.013] |
| linear stack, v4, including AF3 | −0.019 [−0.057, +0.017] |
| rank average, like-for-like | +0.009 [−0.006, +0.024] |
| gradient boosting vs sequence-only GB | +0.003 [−0.018, +0.022] |

The z-scored run is decisive: structure alone reaches **0.857**, within 0.064 of
sequence, and combining still gives −0.001. The failure is **redundancy, not
weakness** — whatever the folding models encode, the sequence model already has.

Two confounds were found and controlled, either of which would have produced a false
positive:

- **Ceiling effect.** Per-allele benefit is perfectly monotone in sequence weakness —
  the pattern King et al. report as complementarity — but correlates with
  sequence-only AUROC (rho −0.899, p 0.015) and not with structure-only AUROC
  (rho +0.600, p 0.21). Benefit tracks headroom, not structural quality.
- **Rank-transformation confound.** A rank average appeared to give +0.026, the only
  positive across eight configurations. Per-allele ranking alone lifts the sequence
  model from 0.930 to 0.946; like-for-like the gain is +0.009 and spans zero.

The ensemble code recovers +0.042 from planted synergy on synthetic data, so this is
a demonstrated null rather than a blind test.

## Data coverage — a finding independent of either question

**The gap is in negatives, not positives.** Across all of IEDB (5,770,781 assay rows,
303 HLA alleles), restricted to alleles with ≥100 positive 9mers:

| locus | median negatives per positive | alleles studied |
|---|---|---|
| HLA-A | **0.166** | 44 |
| HLA-B | 0.0084 | 64 |
| HLA-C | 0.0074 | 21 |

HLA-C is *not* positive-poor — several alleles have thousands of known ligands and
the median positive count exceeds HLA-A's. Mass spectrometry is untargeted and yields
only positives; binding assays require choosing a peptide, and those choices followed
research attention toward HLA-A. The same disparity appears by data type: median
9mers per allele are 1583 / 1356 / 1760 by mass spectrometry against 1357 / 35 / 23
by binding affinity.

For HLA-C\*15:05 and HLA-C\*16:02 there are **zero** experimentally determined
non-binders in the whole of IEDB. Constructed decoys are therefore a necessity rather
than a convenience, and the equity question cannot currently be answered with
experimentally grounded negatives by anyone.

## What predicts per-allele performance

Across all 123 alleles, against validation-split AUROC:

| predictor | rho | p |
|---|---|---|
| anchor information content | **+0.660** | 1.1e-16 |
| motif nearest-neighbour distance | **−0.363** | 3.7e-05 |
| log₁₀(peptide count) | −0.020 | 0.82 |

Sample size does not predict performance; motif distinctiveness does. The
motif-distance correlation is stable under thresholding on peptide count, so it is
not confounded with sparsity.

**Anchor conventions.** 43% of alleles have a high-information position outside the
standard P2 / C-terminus scheme (HLA-A 75%, HLA-C 42%, HLA-B 25%). Defining anchors
per allele from information content yields the best structural feature for **six**
independent measurements — AF2 on two panels and two MSA versions, ESMFold2, Boltz
and AF3 — once between-allele scale is removed.

## Benchmark limitations

Both are quantified rather than merely noted, and both are in the write-up.

**Selection circularity.** Fold-set binders are the top decile of the Motif Atlas PWM
score, so a PWM alone separates fold set v2 at AUROC 1.000. Across five sequence
models, coupling to that criterion tracks AUROC almost perfectly (Spearman ≈ 0.9):

| model | rho(PWM, score) | AUROC |
|---|---|---|
| MixMHCpred 3.0 | 0.909 | 0.999 |
| NetMHCpan-4.1 | 0.773 | 0.961 |
| MHCflurry affinity | 0.718 | 0.911 |
| **ours** | **0.690** | **0.921** |
| MHCflurry presentation | 0.621 | 0.841 |

MixMHCpred's 0.999 is circular rather than impressive, and NetMHCpan's edge over our
model is largely that alignment. Ours is the only model meaningfully above the trend
line. This does not bite RQ1 the same way, since the structural models never see the
Atlas — if anything it biases toward sequence.

**Statistical power.** Between-allele variation in per-allele AUROC has sd 0.024
across 123 alleles; fold-set AUROC from 12 binders and 12 decoys has a standard error
near 0.075. Attenuation caps any observable between-allele correlation near 0.3, so
per-allele questions — including whether structural benefit tracks motif isolation —
are underpowered by roughly an order of magnitude in complexes per allele. Pooled
figures and within-allele paired comparisons are unaffected.

## Pipeline

```
MHC Motif Atlas ──► prepare_atlas.py ──► labelled set (838k rows, 123 alleles)
                          │
        ┌─────────────────┴───────────────────┐
        ▼                                     ▼
  sequence model                    panel + fold-set selection
  (PyTorch, pan-allele)             v2: 6 alleles · v4: 9 by motif isolation
        │                                     │
        │         ┌───────────┬───────────┬───┴───────┬──────────────┐
        │         ▼           ▼           ▼           ▼              ▼
        │    ESMFold2      Boltz-2.1   AF2/HISTO    AF3      fine-tuned AF2
        │         └───────────┴───────────┴───────────┴──────────────┘
        │                                 ▼
        │            PAE · confidence · geometry · representations
        └──────────────────► RQ2 ensembles ◄────────┘

  external baselines: NetMHCpan-4.1 · MHCflurry (affinity, presentation) · MixMHCpred
  coverage analysis:  MHC Motif Atlas × MHCflurry curation × full IEDB export
```

## Where things run

| Stage | Machine | Environment |
|---|---|---|
| Data prep, sequence model, all analysis | Beta (RTX 4090) | `pmhcpresent` |
| Fold-set and panel selection | Beta | `pmhcpresent` |
| ESMFold2 folding | Beta | `esmfold2` (Python 3.12) |
| AF2 via HISTOFold | Beta | ColabFold 1.5.5 Singularity image |
| AlphaFold 3 | Beta | Singularity image built from `patches/af3.def` |
| Fine-tuned AF2 (Motmaen et al.) | Beta | ColabFold image + `patches/alphafold_finetune_modernise.patch` |
| Boltz-2.1 (cloud API) | Mac | `uv` project |
| Overflow / large downloads | CS lab machines | local `/tmp` (home quota is 10 GB) |

ESMFold2 needs its own environment: `esm` requires Python >=3.12,<3.13 while
`pmhcpresent` runs 3.13. ESMC-6B (24 GB in the HuggingFace cache) is a required
dependency despite appearing nowhere in this codebase — deleting it triggers a silent
re-download.

## Scripts

**Data and sequence model**
- `prepare_atlas.py` — atlas positives → labelled table with generated negatives
- `make_split.py` — the shared train/validation split. **Every script must read this
  file**; recomputing it independently caused up to 42% leakage concentrated in
  positives before it existed
- `per_allele_auroc.py`, `plot_per_allele.py` — per-allele AUROC across 123 alleles
- `derive_anchors.py` — per-position information content; the 43% survey
- `ablation_a2.py`, `ablation_a2_condB.py`, `ablation_family_condB.py` — starvation
  and family-removal ablations. *These predate `make_split.py` and should be rerun
  before their numbers are quoted.*

**Panel and fold-set construction**
- `select_allele_panel.py` — panel v1–v3 (AUROC and anchor-IC stratification)
- `select_allele_panel_motif.py` — panel v4, by Jensen-Shannon motif isolation
- `select_rq3_alleles.py` — RQ3 alleles as representatives of motif classes
- `select_fold_set_canonical.py` — motif-typical binders, PWM top decile
- `select_decoys_hard.py` — anchor-matched adversarial decoys (`--max-pctile 25`)
- `select_fold_set_affinity.py` — PWM-free fold set from measured affinities

**Folding**
- `fold_esmfold2.py` — ESMFold2 on Beta, writes structures and PAE
- `build_af3_inputs.py` — AlphaFold 3 JSON inputs with per-chain MSAs
- `build_finetune_targets.py` — `alphafold_finetune` targets file
- HISTOFold is a patched third-party fork; see `patches/histofold_singularity.patch`

**Structural features**
- `analyse_pae.py`, `analyse_pae_af2.py`, `analyse_pae_af3.py` — anchor-localised PAE
- `extract_confidence.py` — global and localised confidence, all architectures
- `extract_geometry.py`, `extract_geometry_af2.py` — interface geometry
- `rq1_embeddings.py` — learned representations with a light classifier
- `auroc_structure.py` — features as classifiers, with direction handling

**Baselines, ensembles and controls**
- `score_netmhcpan.py`, `score_mhcflurry.py`, `score_mixmhcpred.py`
- `score_sequence_on_foldset.py` — our model on a fold set, with leakage reporting
- `rq2_stack.py`, `rq2_ensemble_alt.py`, `rq2_gate.py` — the eight configurations
- `bootstrap_auroc.py` — bootstrap CIs and paired differences
- `fold_quality_control.py` — does confidence predict which complexes are wrong?
- `check_motmaen_overlap.py` — training-set overlap for the fine-tuned comparison

## Data

| Source | Use | In git? |
|---|---|---|
| MHC Motif Atlas (class I MS peptides) | Training labels and fold-set binders | ❌ gitignored |
| Per-locus pseudosequence JSONs | 34-mer pocket pseudosequence per allele | ❌ gitignored |
| Per-locus canonical sequence JSONs | Full HLA and β2m sequences for folding | ❌ gitignored |
| MHCflurry curated data | Baselines and the data-type coverage analysis | ❌ via `mhcflurry-downloads` |
| IEDB `mhc_ligand_full` | Coverage analysis only — not a training source | ❌ 284 MB, processed off-repo |
| Fold outputs | Predicted structures, PAE, embeddings | ❌ gitignored |
| Extracted feature tables | `pae_*.csv`, `conf_*.csv`, `geom_*.csv`, `results/` | ✅ committed |

The Motif Atlas is the only training source. IEDB appears solely in the coverage
analysis, and MHCflurry's curation only as a baseline and for the affinity comparison.

`prepare_atlas.py` filters to classical HLA (A/B/C) and 8–11mers, normalises allele
names to `HLA-A*02:01` form, labels presented peptides `1`, and generates
length-matched negatives. `--neg-mode proteome` is the intended decoy set;
`--neg-mode peptide-pool` is a no-external-data fallback.

## RQ3 scoring constraint

pLDDT, PAE and ipSAE need a **re-fold per mutant**; contact maps and shape
complementarity can be recomputed on a **fixed wild-type backbone**. The structure
module tags every feature with `refold_required` so the saturation-mutagenesis scorer
can separate cheap from expensive features.

## Data governance

TRACERx is **controlled-access** (Data Access Committee) and is used as an
*illustrative application* only — never as a benchmark or evaluation cohort.
**Nothing TRACERx-derived enters git**; see `.gitignore`.

AlphaFold 3 output may not be used to train models intended for commercial
application under its weights terms of use, so AF3 is evaluation-only and excluded
from anything fitted. NetMHCpan-4.1 is used through a colleague's licensed
installation; this project is not licensed independently.

## Quickstart

```bash
conda env create -f environment.yml
conda activate pmhcpresent
pip install -e ".[dev,struct,ml]"
pytest -q

python scripts/prepare_atlas.py \
  --input data/raw/all_peptides.txt \
  --output data/processed/atlas_labelled.csv \
  --neg-mode peptide-pool

python scripts/make_split.py \
  --data data/processed/atlas_labelled.csv \
  --out data/processed/split_val.csv
```

For panels, folding, feature extraction and the ensemble experiments, see
[REPRODUCE.md](REPRODUCE.md).