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
| Stage | Mac (now) | Beta (GPU) |
|---|---|---|
| Repo / configs / parsers / metrics | ✅ | ✅ |
| NN architecture + forward test (MPS dummy tensors) | ✅ | ✅ |
| Structure-feature extraction from example PDBs | ✅ | ✅ |
| AlphaFold folding & per-mutant re-folds (RQ3) | ❌ | ✅ |
| NetMHCpan runs at scale | ❌ | ✅ |
| Full NN training | ❌ | ✅ |

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
```
