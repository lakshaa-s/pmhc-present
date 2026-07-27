# pmhcpresent — Cancer-antigen discovery (HLA-I presentation, equity lens)

COMP0190 / AI4BH 2025–26. Predicts which tumour-derived peptides are presented by
HLA class I, and asks whether predictions hold up equitably across ancestrally
diverse populations.

For exact commands to regenerate every result below, see **[REPRODUCE.md](REPRODUCE.md)**.

## Research questions

- **RQ1** — Do 3D structure models beat sequence models for underrepresented HLA alleles?
- **RQ2** — Do sequence and structure models combine synergistically in an ensemble?
- **RQ3** — Does in-silico saturation mutagenesis show the two model types learned
  the same binding biology? (novel extension)

## Results so far

**Sequence baseline.** Pan-allele NN (peptide + 34-mer pocket pseudosequence),
near-duplicate-aware split via Hamming clustering: **AUROC ≈ 0.974**, equity gap
across representation bins ≈ 0.007. The score is robust to tightening the split from
exact-duplicate to near-duplicate removal, so it is not leakage-inflated.

**The equity gap is driven by motif isolation, not data volume.** Per-allele AUROC
across all 123 alleles shows HLA-C systematically trailing (median 0.951 vs 0.977 for
HLA-A and 0.980 for HLA-B). Data-rich HLA-C alleles underperform too — C\*12:03 has
3,026 peptides and still scores 0.927 — so the driver is that HLA-C is motif-distinct
from the HLA-A/B-dominated training data rather than simply scarce.

**Cross-allele transfer sustains rare alleles (the orphan-allele mechanism).** A 2×2
ablation on HLA-A\*02:01 (starve the allele's own data × remove its motif-similar
family) shows performance collapses only when *both* are removed: 0.967 → 0.904, with
error bars 20× wider. Starving alone or removing the family alone costs almost nothing.
So an allele with little data is fine *provided a motif-similar allele is represented*.
Repeating on HLA-A\*03:01 and HLA-B\*27:05 did **not** reproduce the collapse — those
alleles retain relatives outside the removed set — so this is a mechanism that operates
in specific cases, not a universal law.

**Structure: anchor-localised PAE carries binding signal, but much of it is anchor
recognition.** Folding a controlled set (5 alleles × 6 canonical binders + 6 decoys, all
9mers) and scoring the predicted aligned error at the peptide's anchor positions
(P2 and C-terminus). Two decoy classes were used, differing in difficulty:

- **Motif-mismatched decoys** — real ligands of other alleles, filtered to *exclude* the
  target's preferred anchor residues. Rejectable on anchors alone.
- **Anchor-matched decoys** — real ligands of other alleles that *carry* the target's
  anchors at both P2 and the C-terminus but score low against its overall motif. The
  anchor shortcut is removed, so discrimination must come from groove fit.

| Pooled AUROC (n=60) | Boltz-2.1 (mismatched) | ESMFold2 (mismatched) | ESMFold2 (anchor-matched) |
|---|---|---|---|
| `pae_anchors` (P2 + C-term) | 0.783 | **0.911** | **0.700** |
| `pae_anchor2` (P2 only) | 0.737 | 0.921 | 0.759 |
| `pae_pep_mhc` (whole interface) | 0.694 | 0.863 | 0.672 |
| interface geometry (contacts) | 0.21–0.41 | — | — |
| `iptm` (global confidence) | ~flat 0.98–0.99 | — | — |

Three things stand out. Signal **increases as the metric localises to the anchors** —
global confidence and raw contact counts carry nothing, matching anchor-dominated binding
biology. The effect **replicates across two independent architectures**, so it is a
property of structure prediction on pMHC rather than a quirk of one model. And roughly
**two-thirds of the apparent signal is anchor recognition**: against anchor-matched
decoys, discrimination falls from 0.911 to 0.700. It stays above chance and every allele
still points the right way, so there is residual sensitivity to groove fit beyond the
anchors — but the headline number depends heavily on how negatives are constructed.

Under the harder test, HLA-B\*07:02 (0.972) and HLA-B\*27:05 (0.889) retain most of their
discrimination while HLA-A\*02:01 (0.639), HLA-C\*15:05 (0.639) and HLA-C\*16:02 (0.667)
lose most of theirs. The apparent equity advantage seen against mismatched decoys —
C\*15:05 scoring 1.000 where the sequence model scores 0.889 — does **not** survive; it
was largely an artifact of easy negatives. Whether structure genuinely complements
sequence in the orphan-allele regime (RQ2) remains open.

*Caveats:* 6 binders + 6 decoys per allele, so per-allele figures are coarse. Neither
structural AUROC is directly comparable to the sequence model's 0.974, which is measured
over a large held-out set with pooled negatives rather than 60 designed complexes. The
anchor-matched decoys are also a conservative test: "not observed on this allele" is
weaker than "does not bind this allele", so some may be genuine binders.

## Pipeline

```
MHC Motif Atlas ──► prepare_atlas.py ──► labelled set (838k rows, 123 alleles)
                                              │
                    ┌─────────────────────────┴──────────────────────┐
                    ▼                                                ▼
        sequence model (PyTorch NN)                    fold-set selection (PWM-based)
        + per-allele AUROC                                           │
        + ablation studies                          ┌────────────────┼────────────────┐
                    │                               ▼                ▼                ▼
                    │                            Boltz-2.1       ESMFold2      HISTOFold/AF2
                    │                               └────────────────┼────────────────┘
                    │                                                ▼
                    │                                   PAE / geometry features
                    └──────────────────────► ensemble (RQ2) ◄────────┘
```

## Where things run

| Stage | Machine | Environment |
|---|---|---|
| Data prep, sequence training, ablations, per-allele AUROC | Beta (RTX 4090) | `pmhcpresent` |
| Fold-set selection (canonical binders, decoys) | Beta | `pmhcpresent` |
| Boltz folding (cloud API — no local GPU needed) | Mac | `.venv` via `uv` |
| ESMFold2 folding (local, ~13.7 GB VRAM) | Beta | `esmfold2` (Python 3.12) |
| HISTOFold / AF2 folding | Beta | Docker container *(not yet set up)* |

ESMFold2 needs its own conda environment: the `esm` package requires Python
>=3.12,<3.13, while `pmhcpresent` runs 3.13.

## Scripts

**Data & sequence model**
- `prepare_atlas.py` — atlas positives → labelled table with generated negatives
- `per_allele_auroc.py` / `plot_per_allele.py` — per-allele AUROC across all 123 alleles
- `ablation_a2.py` — dose-response starvation of a single allele (Condition A)
- `ablation_a2_condB.py` — 2×2 starve × remove-family for HLA-A\*02
- `ablation_family_condB.py` — the same, generalised via `--family-regex`

**Fold-set construction**
- `select_fold_set_canonical.py` — motif-typical binders (PWM top decile, then
  diversified). *Supersedes `select_fold_set.py`, which maximised diversity and so
  selected motif-atypical peptides — unsuitable for a discrimination test.*
- `select_decoys_clean.py` — decoys rejected for carrying the target's anchor
  residues. *Supersedes `select_decoys.py`, which used allele-level distance only and
  leaked anchor-carrying peptides into the decoy set.*
- `select_decoys_hard.py` — adversarial decoys that **do** carry the target's anchors
  but score low overall. Running these showed ~two-thirds of the apparent structural
  signal was anchor recognition (0.911 → 0.700), so both decoy classes should be
  reported together.

**Folding & structural analysis**
- `fold_esmfold2.py` — folds pMHC complexes with ESMFold2 on Beta; writes Boltz-compatible
  output plus embeddings
- `extract_boltz_features.py` — confidence metrics + PAE summaries per fold
- `analyse_pae.py` — per-residue anchor PAE, binder vs decoy
- `extract_geometry.py` — interface contacts from predicted structures
- `auroc_structure.py` — structural features as classifiers, AUROC vs the sequence baseline

## Data

| Source | Use | In git? |
|---|---|---|
| MHC Motif Atlas (class I MS peptides) | Presented-peptide labels (positives) | ❌ gitignored |
| Per-locus pseudosequence JSONs | 34-mer pocket pseudosequence per allele | ❌ gitignored |
| Per-locus canonical sequence JSONs | Full HLA + β2m sequences for folding | ❌ gitignored |
| Generated decoys | Non-presented peptides (negatives) | ❌ built by prep step |
| Fold outputs (`boltz-*`, `esmfold2-*`) | Predicted structures, PAE, embeddings | ❌ gitignored |

`scripts/prepare_atlas.py` filters to classical HLA (A/B/C) and 8–11mers, normalises
allele names to `HLA-A*02:01` form, labels presented peptides `1`, and generates
length-matched negatives per allele. `--neg-mode proteome` (random human-proteome
peptides) is the intended decoy set; `--neg-mode peptide-pool` is a no-external-data
fallback and is what the current dataset uses.

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

For the structural pipeline (fold-set selection → folding → PAE analysis → AUROC),
see [REPRODUCE.md](REPRODUCE.md).
