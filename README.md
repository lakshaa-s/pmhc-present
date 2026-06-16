# pmhcpresent - Cancer-antigen discovery (HLA-I presentation, equity lens)

COMP0190 / AI4BH 2025–26. Predicts which tumour-derived peptides are presented by
HLA class I, and asks whether predictions hold up equitably across ancestrally
diverse populations.

## Research questions

- **RQ1** - Do AlphaFold-derived 3D structure models beat sequence models for
  underrepresented HLA alleles?
- **RQ2** - Do sequence and structure models combine synergistically in an ensemble?
- **RQ3** - Does in-silico saturation mutagenesis show the two model types learned
  the same binding biology? (Novel Extension)

## Pipeline

NetMHCpan-4.1 (sequence baseline) → custom PyTorch NN → AlphaFold 3D features → ensemble.

## Where things run

| Stage | Mac (local dev) | Beta (GPU) |
|---|---|---|
| Repo / configs / parsers / metrics | ✅ | ✅ |
| NN architecture + forward test (MPS dummy tensors) | ✅ | ✅ |
| Structure-feature extraction from example PDBs | ✅ | ✅ |
| Data prep (atlas → labelled set, pseudoseq join) | ✅ | ✅ |
| AlphaFold folding & per-mutant re-folds (RQ3) | ❌ | ✅ |
| NetMHCpan runs at scale | ❌ | ✅ |
| Full NN training | ❌ | ✅ |

## Data

| Source | Use | In git? |
|---|---|---|
| MHC Motif Atlas (class I MS peptides) | Presented-peptide labels (positives) | ❌ (gitignored) |
| Per-locus pseudosequence JSONs (`hla_a/b/c.json`) | 34-mer pocket pseudosequence per allele | ❌ (gitignored) |
| Generated decoys | Non-presented peptides (negatives) | ❌ (built by prep step) |

`scripts/prepare_atlas.py` turns the positives-only atlas file into a labelled
training table: it filters to classical HLA (A/B/C) and 8–11mers, normalises allele
names to `HLA-A*02:01` form, labels presented peptides `1`, and generates
length-matched negatives per allele (`label=0`) at a configurable ratio.

Pseudosequences are loaded via `load_pseudosequences_json` from the per-locus JSONs
(`entry["pocket_pseudosequence"]`, keyed by `protein_allele_name`).

> **Negative-sampling note:** `--neg-mode proteome` (random human-proteome peptides
> the allele does not present) is the intended decoy set; `--neg-mode peptide-pool`
> is a no-external-data fallback for quick baselines. Negative strategy lives
> upstream of the dataset, so it can be swapped without changing downstream code.

## RQ3 scoring constraint (designed into the structure module)

pLDDT, PAE and ipSAE need a **re-fold per mutant**; contact maps and shape
complementarity can be recomputed on a **fixed wild-type backbone**. The structure
module tags every feature with `refold_required` so the saturation-mutagenesis
scorer can split cheap vs expensive features.

## Data governance

TRACERx is **controlled-access** (Data Access Committee) and is used as the
*application* dataset only — never as a benchmark. Benchmark sets are kept separate
for evaluation. **Nothing TRACERx-derived enters git** (see `.gitignore`).

## Quickstart

```bash
conda env create -f environment.yml
conda activate pmhcpresent
pip install -e ".[dev,struct,ml]"
pytest -q

# Build the labelled training set from the atlas peptides
python scripts/prepare_atlas.py \
  --input data/raw/all_peptides.txt \
  --output data/processed/atlas_labelled.csv \
  --neg-mode peptide-pool
```