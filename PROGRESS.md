# Project state ledger

Same principle as `make_split.py`: derive state **once**, here, and have every
session read this file rather than reconstructing it. A session that re-infers
"where the project is" from scratch will drift, in exactly the way independently
recomputed splits drifted.

Rules for this file:

- Every result row cites the artefact it came from. **A number with no artefact is
  not a result** — it is a memory, and it goes in "Unconfirmed" until traced.
- Retractions stay visible. Do not delete a superseded claim; strike it and say why.
- Claude may propose edits to this file. Claude may not mark anything `done` or
  promote a row out of "Unconfirmed" without citing the artefact.

---

## Which model produced which number

**Read this before quoting any AUROC.** Two models are in play and they are not
interchangeable.

| Model | Data | Split | Use |
|---|---|---|---|
| `models/rq1_baseline_split_v2.pt` | `atlas_labelled.csv` | `split_val.csv` | **All fold-set results** — RQ1, RQ2, RQ3, external baselines |
| `models/rq1_baseline_split_v3.pt` | `atlas_labelled_v2.csv` | `split_val_v2.csv` | **All validation-split results** — pooled AUROC, per-allele AUROC, the three predictors |

v2's data has the multiset negative-sampling artefact; v3's does not. The fold sets
were built from v2's split, and only 10.3% of that split survives regeneration
(18/72 v2 binders and 12/108 v4 binders remain held out), so the fold sets cannot be
carried over to v3 without rebuilding and refolding 360 complexes across five
architectures. Neither model is evaluated on its own training data.

Artefact for the 10.3% figure: measured 11 Aug, recorded in the 11 August section of
`REPRODUCE.md`.

---

## Status: verified present in the repo

| Component | Evidence |
|---|---|
| Sequence CNN (`PresentationNet`) | `src/pmhcpresent/models/nn.py` — 30,465 params, verified by `count_parameters` |
| Within-allele cluster splitting, two-way | `src/pmhcpresent/eval/splits.py`, `scripts/make_split.py` |
| Shared canonical split file | `scripts/make_split.py` → `data/processed/split_val.csv` |
| Structure features: pLDDT, contacts, ipSAE, PAE | `src/pmhcpresent/structure/` |
| `REFOLD_REQUIRED` cost model for RQ3 | `structure/features.py:22` |
| Easy + hard decoy selection | `scripts/select_decoys_clean.py`, `select_decoys_hard.py` |
| Fold sets v2 / v3b / v4 / affinity | `fold_sets/` |
| Folding arms: Boltz, ESMFold2, AF2, AF3, fine-tuned AF2 | `auroc_*`, `pae_*`, `conf_*`, `geom_*` CSVs per arm |
| RQ2 ensemble scaffolding | `scripts/rq2_stack.py`, `rq2_gate.py`, `rq2_error_overlap.py`, `rq2_ensemble_alt.py` |
| RQ3 variants + landscape comparison | `scripts/build_rq3_variants.py`, `rq3_compare_landscapes.py`, `rq3_sequence_landscape.py`, `rq3_shap.py` |
| External baselines | `scripts/score_netmhcpan.py`, `score_mhcflurry.py`, `score_mixmhcpred.py` |
| Confound controls | `confound_vs_per_allele.py`, `hlac_partial_effect.py`, `foldset_survival_check.py`, `predictors_vs_confound.py` |
| Dataset audit | `audit_dataset.py`, `crossover_label_balance.py` |
| CI: ruff + pytest | `.github/workflows/ci.yml`, 9 test modules |

## Status: absent

| Component | Note |
|---|---|
| AFND integration | Still no script. `src/pmhcpresent/eval/stratified.py` has the *machinery* (`assign_frequency_bins`, `stratified_metrics`) but it is fed **peptide counts**, not population frequencies — so the reported "equity gap" is a data-representation gap, not an ancestry one. **Open decision unchanged: implement or amend the LR.** |
| Real shape complementarity | `shape.py` is a ΔSASA proxy, explicitly not Sc. Do not rely on it for RQ1/RQ2 conclusions without qualification. |
| Regenerated fold sets | The multiset fix is applied in `prepare_atlas.py` and used for v3, but the fold sets remain on the v2 split. First item in future work. |

---

## Results ledger

| # | Claim | Decoy set | Artefact | Status |
|---|---|---|---|---|
| 1 | Sequence 0.930 beats AF3 0.858, AF2 0.842, ESMFold2 0.805, Boltz 0.745; every paired margin excludes zero | fold set v4, anchor-matched | `auroc_af3_v4.csv`, `pae_af2_v4.csv`, `pae_esmfold2_v4.csv`, `pae_boltz_v4.csv`, `results/sequence_v4.csv` | verified |
| 2 | Five structural readouts all below sequence: representations 0.834, PAE 0.804, confidence 0.753, consistency 0.656, geometry 0.492 | fold set v2 | `results/embeddings_af2.csv`, `pae_af2_v3b.csv`, `conf_af2_v2.csv`, `consistency_esmfold2_v4.csv`, `geom_af2_v2.csv` | verified |
| 3 | RQ2 null across nine configurations, every interval spanning zero | v2 and v4 | `results/rq2_stack_v4.csv`, `rq2_ensemble_alt_v4.csv`, `rq2_gate_v4.csv`, `rq2_stack_v4_af3.csv` | verified |
| 4 | The two families fail on partly different complexes: margin rho +0.223, Jaccard 0.220 vs a 0.143 chance baseline | fold set v4 | `results/rq2_error_overlap.csv` | verified |
| 5 | Structural deficit is decoy rejection specifically — Boltz sens 0.898 / spec 0.556 | fold set v4 | `results/rq2_error_overlap.csv` | verified |
| 6 | IEDB gap is in negatives: HLA-A 0.166 neg/pos, HLA-B 0.0084, HLA-C 0.0074; zero for C\*15:05 and C\*16:02 | n/a | `data/processed/iedb_coverage.json` | verified |
| 7 | Anchor IC predicts per-allele AUROC (+0.533); motif distance does (−0.291); sample size does not (−0.118) | validation split, **v3 model** | `results/per_allele_auroc_v3.csv` | verified 12 Aug |
| 8 | HLA-C effect survives the negative-sampling confound: OLS −0.0267 (p<0.0001), matched −0.0269 (p 0.0002) | validation split, v2 model | `results/confound_vs_per_allele.csv`, `hlac_partial_effect.py` output | verified |
| 9 | Fine-tuning adds ~+0.09 where the allele is familiar and nothing on motif-isolated alleles; allele composition costs 0.193 against decoy construction's 0.038 | four benchmarks | `results/aff_finetuned.csv`, `aff_affinityset_ft.csv`, `aff_their_testset.csv` | verified |
| 10 | `pae_anchors_ic` is the best structural feature across six measurements once between-allele scale is removed | v2 and v4 | `auroc_af2_v*.csv`, `auroc_esmfold2_v4.csv`, `auroc_boltz_v4.csv`, `auroc_af3_v4.csv` | verified |
| 11 | Sequence model recovers anchor *positions* (top-2 inside IC anchors for 6/7 alleles) but residue preferences only partly (landscape rho +0.541) | RQ3 panel | `results/rq3_sequence_summary.csv` | verified |
| 12 | **B\*08:01 P5: both structural models rank P5 first; the sequence model does not** | RQ3 variants | `results/rq3_structural_6seed_summary.csv` (ESMFold2 [5,9]), `results/af2_b08_landscape.npy` (AF2 [5,9,2], P5 sens 0.489 vs P9 0.459, P2 0.295) | verified 12 Aug |
| 13 | Asp9 is in the model's input (pseudosequence position 2) but only 1 of 123 training alleles exhibits P5 anchoring — B\*08:01 at 1.93 bits, the four HLA-C\*07 Asp9-carriers at 0.25–0.61 | n/a | `data/pseudoseq/hla_*.json`, `data/processed/anchors.json` | verified 12 Aug |
| 14 | Attribution is peptide-specific: SHAP seed agreement +0.186 vs landscape +0.168, but the two methods agree with each other at +0.633 | RQ3 panel | `results/rq3_shap_summary.csv` | verified |

---

## Retracted — do not resurrect

- ~~Structure models are strongest precisely where sequence models are weakest
  (equity claim).~~ Did not survive anchor-matched hard decoys; roughly two-thirds
  of the apparent structural signal was anchor recognition. Retracted deliberately.
  If this sentence reappears in a draft, it is a regression, not a finding.

- ~~Validation AUROC 0.9732.~~ Inflated by the multiset negative-sampling artefact.
  The v3 model on regenerated data gives **0.9715** (`/tmp/train_v3.log`, 12 Aug).
  The difference is small because the model uses allele information the
  peptide-identity prior does not.

- ~~Anchor IC vs per-allele AUROC rho +0.660 / +0.603.~~ ~~Partial +0.648.~~ Both
  inflated by the same artefact, and the partial over-corrected. Clean value on the
  v3 model: **+0.533** (p 2e-10).

- ~~Motif nearest-neighbour distance rho −0.363.~~ ~~Partial −0.502.~~ Clean value:
  **−0.291** (p 1e-03). The effect is real but more modest than previously claimed.

- ~~log10(peptide count) rho −0.020.~~ Clean value **−0.118** (p 0.2) — still null,
  which is the load-bearing part.

- ~~Fine-tuned AlphaFold drops 0.24 on our benchmark because of decoy
  construction.~~ That compared v4 structural results against their test set,
  conflating two variables. Isolated: decoys cost **0.038**, alleles cost **0.193**.
  Fine-tuning works reliably and its benefit does not reach motif-isolated alleles.

- ~~RQ2's null is redundancy — whatever structure encodes, sequence already has.~~
  Error overlap gives margin rho +0.223 against a 0.143 chance baseline, so
  complementary signal exists and nine strategies failed to exploit it. Sample size
  is the likelier explanation.

- ~~The sequence model's 0.995 on the three-allele anchor-matched subset shows
  genuine discrimination.~~ A PWM alone scores 1.000 on those 72 complexes. The
  subset is an outlier; across full sets the model sits 0.07–0.08 *below* the PWM,
  which is the real evidence it is not riding the selection criterion.

- ~~The sequence model uses learned positional embeddings and attention pooling.~~
  Never implemented. It is max-pool over a kernel-3 convolution, therefore
  position-invariant in the peptide path. Verified by reading `nn.py`.

---

## Unconfirmed — carried from session notes, not yet traced to an artefact

- ~~HLA-C median AUROC ≈ 0.951 vs ≈ 0.977–0.980 for A/B~~ → traced.
  `confound_vs_per_allele.py` gives HLA-A 0.970, HLA-B 0.975, HLA-C 0.940 on the v2
  model. Recompute on v3 before quoting.

- ~~Partial rho ≈ −0.455 for the HLA-C effect after confound control~~ → superseded.
  The defensible figures are the OLS coefficient −0.0267 and the matched comparison
  −0.0269, both in row 8 above. The −0.455 does not correspond to either.

- ~~Prior-alone AUROC ≈ 0.248 from the negative-sampling artefact~~ → traced,
  `crossover_label_balance.py`. Note the deduplication fix only moves it to
  **0.3596**; the residual comes from 89.6% cross-allele crossover, which
  deduplication cannot touch. Proteome negatives are far worse (0.8801).

- ~~Boltz ≈ 0.783 pooled AUROC; ESMFold2 ≈ 0.911 against motif-mismatched decoys~~
  → traced to the 30–31 July section of `REPRODUCE.md`. The 0.911 is the easy-decoy
  figure and must always be quoted beside the anchor-matched 0.700.

- ~~Both arms ≈ 0.700 against anchor-matched hard decoys~~ → traced, same section.

- ~~Fold-set binder survival rate — session notes say ~13.9%, the git log says
  something else.~~ → **resolved 11 Aug.** Under regeneration, 18/72 fold set v2
  binders (25.0%) and 12/108 fold set v4 binders (11.1%) remain held out; the
  validation split itself retains 10.3%. The ~13.9% was neither figure and should
  not be used.

---

## Open decisions

- [ ] AFND: implement, or amend the literature review to drop the commitment.
      Note the existing "equity gap" is stratified by peptide count, not population
      frequency, so the LR commitment is genuinely unmet either way.
- [ ] Is AF3 in scope for §3.6, or reported as supplementary?
- [x] RQ2 stacker: logistic regression vs rank-average — **resolved.** Both are null
      and the error-overlap evidence shows partial complementarity, so neither is
      preferred on performance. Report all nine configurations rather than choosing.
- [ ] Whether to run AF2 on the remaining RQ3 alleles, or report B\*08:01 alone
- [ ] Introduction chapter — deprioritised pending results