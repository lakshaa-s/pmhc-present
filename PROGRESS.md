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
| `models/rq1_baseline_split_v3.pt` | `atlas_labelled_v2.csv` | `split_val_v2.csv` | **All validation-split results** — pooled AUROC, per-allele AUROC, the per-allele predictors |

v2's data has the multiset negative-sampling artefact; v3's does not. The fold sets
were built from v2's split, and only 10.3% of that split survives regeneration
(18/72 v2 binders and 12/108 v4 binders remain held out), so the fold sets cannot be
carried over to v3 without rebuilding and refolding 360 complexes across five
architectures. Neither model is evaluated on its own training data.

Artefact for the 10.3% figure: measured 11 Aug, recorded in the 11 August section of
`REPRODUCE.md`.

**One consequence to watch.** `results/predictors_vs_confound_v2model.csv` holds
figures computed on the v2 model and retracted below. The current file is
`predictors_vs_confound_v3.csv`. The old one was renamed rather than deleted because
its bare filename led an external review to recommend reconciling this ledger *to* the
stale value, which would have reintroduced a retraction.

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
| **AFND population-frequency analysis** | `scripts/afnd_frequency_analysis.py`, `data/raw/afnd/afnd.tsv`, `data/processed/afnd_country_regions.csv` |
| **Allele-specific expression test** | `scripts/expression_vs_performance.py`, `data/raw/dantonio_supp5.xlsx` |
| **123-allele validation sample + external scoring** | `scripts/sample_validation_foldset.py`, `external_vs_isolation.py` |
| **Calibration, decision metrics, power** | `scripts/calibration_metrics.py`, `rq2_power.py` |
| **Consensus diagnosis (PC1 test)** | `scripts/consensus_diagnosis.py` |
| **Summary figure** | `scripts/make_summary_figure.py` → `figures/summary.pdf` |
| **External review + assessment** | `review/REVIEW.md`, `review/ASSESSMENT.md`, `review/review_analysis.py` |
| CI: ruff + pytest | `.github/workflows/ci.yml`, 9 test modules |

## Status: absent

| Component | Note |
|---|---|
| Real shape complementarity | `shape.py` is a ΔSASA proxy, explicitly not Sc. Do not rely on it for RQ1/RQ2 conclusions without qualification. |
| Regenerated fold sets | The multiset fix is applied in `prepare_atlas.py` and used for v3, but the fold sets remain on the v2 split. First item in future work. |
| AF3 on fold set v2 | Image built and banked; folding blocked on infrastructure. See the section below. |
| Geometry for AF3 and Boltz | Permanently unavailable. AF3's v4 structures were written to Beta's tmpfs and lost when it cleared; Boltz ran through the cloud API and returned scores without coordinates; ESMFold2 v2 predates the `to_mmcif` fix and wrote no structures (0 CIFs against v4's 216). |
| Force-field energy features | Not attempted. STRUMP-I (bioRxiv, Sept 2025) reports force-field terms performing well on underrepresented alleles; this work tested folding-model outputs, not force-field scoring of the resulting structures. Stated limitation and the clearest next study. |

---

## Results ledger

| # | Claim | Decoy set | Artefact | Status |
|---|---|---|---|---|
| 1 | Sequence 0.930 beats AF3 0.858, AF2 0.842, ESMFold2 0.805, Boltz 0.745; every paired margin excludes zero | fold set v4, anchor-matched | `auroc_af3_v4.csv`, `pae_af2_v4.csv`, `pae_esmfold2_v4.csv`, `pae_boltz_v4.csv`, `results/sequence_v4.csv` | verified |
| 2 | Five structural readouts all below sequence: representations 0.834, PAE 0.804, confidence 0.753, consistency 0.656, geometry 0.492 | fold set v2 | `results/embeddings_af2.csv`, `pae_af2_v3b.csv`, `conf_af2_v2.csv`, `consistency_esmfold2_v4.csv`, `geom_af2_v2.csv` | verified |
| 2b | Self-consistency replicates across panels: `pae_asymmetry` 0.662 on v2 against 0.656 on v4, same ordering below every other readout | v2 and v4 | `consistency_esmfold2_v2.csv`, `consistency_esmfold2_v4.csv` | verified 30 Aug |
| 3 | RQ2 null across nine configurations, every interval spanning zero | v2 and v4 | `results/rq2_stack_v4.csv`, `rq2_ensemble_alt_v4.csv`, `rq2_gate_v4.csv`, `rq2_stack_v4_af3.csv` | verified |
| 4 | The two families fail on partly different complexes: margin rho +0.223, Jaccard 0.220 vs a 0.143 chance baseline | fold set v4 | `results/rq2_error_overlap.csv` | verified |
| 5 | Structural deficit is decoy rejection specifically — Boltz sens 0.898 / spec 0.556 | fold set v4 | `results/rq2_error_overlap.csv` | verified |
| 6 | IEDB gap is in negatives: HLA-A 0.166 neg/pos, HLA-B 0.0084, HLA-C 0.0074; zero for C\*15:05 and C\*16:02 | n/a | `data/processed/iedb_coverage.json` | verified |
| 7 | Anchor IC predicts per-allele AUROC (+0.533); motif distance does (−0.291); sample size does not (−0.118). All three survive controlling for confound strength | validation split, **v3 model** | `results/per_allele_auroc_v3.csv`, `predictors_vs_confound_v3.csv` | verified 12 Aug, artefact regenerated 30 Aug |
| 8 | HLA-C effect survives the negative-sampling confound: OLS −0.0267 (p<0.0001), matched −0.0269 (p 0.0002) | validation split, v2 model | `results/confound_vs_per_allele.csv`, `hlac_partial_effect.py` output | verified |
| 9 | Fine-tuning adds ~+0.09 where the allele is familiar and nothing on motif-isolated alleles; allele composition costs 0.193 against decoy construction's 0.038 | four benchmarks | `results/aff_finetuned.csv`, `aff_affinityset_ft.csv`, `aff_their_testset.csv` | verified |
| 10 | `pae_anchors_ic` is the best structural feature across six measurements once between-allele scale is removed | v2 and v4 | `auroc_af2_v*.csv`, `auroc_esmfold2_v4.csv`, `auroc_boltz_v4.csv`, `auroc_af3_v4.csv`, `pae_af3_v4_z.csv` | verified |
| 11 | Sequence model recovers anchor *positions* (top-2 inside IC anchors for 6/7 alleles) but residue preferences only partly (landscape rho +0.541) | RQ3 panel | `results/rq3_sequence_summary.csv` | verified |
| 12 | For **HLA-B\*08:01 specifically**, both structural models rank P5 first and the sequence model does not. Not evidence that they track P5 anchoring in general — see row 15 | RQ3 variants, 6 seeds (ESMFold2) / 3 seeds (AF2) | `results/rq3_structural_6seed_summary.csv`, `results/af2_b08_landscape.npy` (AF2 [5,9,2], P5 0.489 vs P9 0.459, P2 0.295) | verified 12 Aug, **bounded 21 Aug** |
| 13 | Asp9 is in the model's input (pseudosequence position 2) but only 1 of 123 training alleles exhibits P5 anchoring — B\*08:01 at 1.93 bits, the four HLA-C\*07 Asp9-carriers at 0.25–0.61 | n/a | `data/pseudoseq/hla_*.json`, `data/processed/anchors.json` | verified 12 Aug |
| 14 | Attribution is peptide-specific: SHAP seed agreement +0.186 vs landscape +0.168, but the two methods agree with each other at +0.633 | RQ3 panel | `results/rq3_shap_summary.csv` | verified |
| 15 | **The structural models do not find P5 for HLA-B\*37:01**, despite its P5 IC being the highest of the group (1.97 bits). P5 ranks 6th of 9 at 0.645; P4/P7/P8/P5 span only 0.05. Seed stability is the best measured anywhere in RQ3 (mean +0.557, min +0.511), so this is not noise. Prediction was recorded before the run and failed | RQ3 variants, 3 seeds | `results/rq3_structural_b37_landscape.csv`, `rq3_structural_b37_seed_stability.csv` | verified 21 Aug |
| 16 | P5 Arg chelation requires a four-residue configuration (Asp9, Thr69, Asp74, Ser97), not Asp9 alone. Exactly **one** of the 123 training alleles has B\*08:01's configuration. At least three distinct configurations achieve P5 anchoring: B\*08:01 (D/T/D/S), B\*37:01 (H/T/Y/R), B\*14:01–02 (Y/T/D/W) | n/a | `results/p5_anchor_residues.csv`; mechanism from Chris Thorpe, 21 Aug | verified 21 Aug |
| 17 | Per-locus mean AUROC is essentially unchanged between the confounded and clean models — HLA-A 0.970→0.968, HLA-B 0.975→0.976, HLA-C 0.940→0.941 — so the HLA-C deficit is not an artefact of negative sampling | validation split, both models | `results/per_allele_auroc.csv`, `per_allele_auroc_v3.csv` | verified 21 Aug |
| 18 | **No European-specific performance gap.** The pre-specified direct test — Europe-enrichment index against per-allele AUROC — is flat at **+0.013 (p 0.89)**, and European frequency controlling for global frequency gives +0.106 (p 0.24). What does hold is a *global* frequency effect in the opposite direction to the Europe-bias prediction: commoner alleles are predicted slightly worse, **−0.243 (p 0.007)** | validation split, v3 model, 123 alleles | `results/afnd_direct_tests.csv`, `afnd_frequency_per_allele.csv` | verified 30 Aug |
| 19 | The seven per-region correlations are **one collinear signal, not seven findings**: median pairwise Spearman between regional frequencies is **+0.590** (range +0.090 to +0.878). At Bonferroni alpha 0.0071 only South & Central Asia (p 0.0007) survives; Europe (−0.127, p 0.161) does not | as above | `results/afnd_direct_tests.csv` | verified 30 Aug |
| 20 | **Three independently developed predictors all degrade with motif isolation**, all on the same 123 alleles and the same peptides: MHCflurry **−0.393** (p<0.0001), NetMHCpan-4.1 **−0.248** (p 0.006), ours **−0.196** (p 0.030). Two were developed by other groups on their own training corpora, so the agreement is not a property of one implementation. For MHCflurry, where training overlap is checkable, the effect is not exposure: isolation and overlap correlate at only −0.113 (p 0.21) and partialling overlap moves isolation to −0.381 | 200 balanced pairs per allele from the validation split | `results/external_vs_isolation_matched.csv`, `sequence_val123.csv`, `mhcflurry_val123.csv`, `netmhcpan_val123.csv`, `fold_sets/validation_sample_123.csv` | verified 30 Aug, **matched 31 Aug** |
| 20b | Our model's own isolation correlation is **−0.291 on the full validation split** and **−0.196 on the 200-pairs-per-allele sample** — same alleles, fewer peptides each, so a noisier per-allele AUROC and an attenuated correlation. The matched figure is the one to present in any three-model table; the full-split figure is the better-powered estimate of our model alone. They must not appear together without this explanation | validation split, v3 model | `results/per_allele_auroc_v3.csv`, `external_vs_isolation_matched.csv` | verified 31 Aug |
| 21 | MixMHCpred is the **only** model to fall on the PWM-free affinity set (0.999 → 0.956); NetMHCpan and MHCflurry both improve, and ours is unchanged at 0.921. It is also the only one trained on immunopeptidomics from the group producing the Atlas against whose PWM the fold-set binders were selected | affinity set, 3 alleles, all modelled directly | `results/mixmhcpred_affinity.csv`, `mhcflurry_affinity.csv`, `netmhcpan_affinity.csv`, `sequence_affinity.csv` | verified 30 Aug |
| 22 | The four-architecture consensus (0.9047 vs AF3's 0.8575) is **variance reduction, not complementary information**: PC1 of the four z-scored scores alone gives 0.9033, a difference of 0.0014, with PC1 explaining 61.3% of variance. The gain does not track headroom (rho +0.203, p 0.60), so it is not the ceiling effect either. Diagnostic validated on synthetic data before use | fold set v4 | `results/consensus_diagnosis_v4.csv`, `_subsets.csv`, `_per_allele.csv` | verified 27 Aug |
| 23 | Calibration is a trade, not a free improvement. Leave-one-allele-out temperature scaling moves ECE 0.2507 → 0.1583 while Brier barely shifts (0.2274 → 0.2265), and pooled **AUROC falls 0.930 → 0.901** because differing per-allele temperatures reorder complexes across alleles | fold set v4 | `results/calibration_v4_reliability.csv` | verified 27 Aug |
| 24 | The AUROC ordering does not survive decision-relevant metrics. AF3 leads AF2 on AUROC (0.858 vs 0.842) but trails at high specificity (pAUC≤0.10 0.619 vs 0.633; PPV@20 0.90 vs 0.95). The consensus, statistically indistinguishable from sequence on AUROC, is clearly behind it at FPR≤0.10 (0.688 vs 0.743) | fold set v4 | `results/calibration_v4_decision_metrics.csv` | verified 27 Aug |
| 25 | RQ2's null is underpowered but bounded. Power is **0.63 at n=216** for the effect actually measured (+0.026, matching the gated ensemble's ungated row at +0.028), reaching 80% near n≈432. Independently, sequence gets only 5 of 216 wrong, 4 of which some structural model rescues, and a cheating per-complex oracle reaches 0.984 against sequence's 0.946 | fold set v4 | `results/rq2_power.csv`, `rq2_error_overlap_margins.csv` | verified 27 Aug |
| 26 | **Allele-specific expression does not predict per-allele performance.** Within-locus expression against AUROC gives +0.057 (p 0.63), and is null within every locus separately (HLA-A +0.098 n=25, HLA-B −0.001 n=31, HLA-C −0.090 n=18). Motif isolation is unaffected by controlling for it (−0.372 → −0.379) | 74 alleles matched from D'Antonio et al. 2019 | `results/expression_vs_performance.csv` | verified 30 Aug |
| 27 | **The isolation effect is not the cross-allele crossover artefact.** 92.2% of the sample's unique peptides also appear in training under some other allele, so peptide promiscuity was tested as a competing explanation. Isolation and promiscuity are independent (−0.018, p 0.84) and isolation survives controlling for it in all three models, strengthening in two: ours −0.196 → −0.229, MHCflurry −0.393 → −0.393, NetMHCpan −0.248 → −0.265. Filtering to unseen peptides was rejected as the control — it leaves 8.7% of validation 9mers, 53 alleles at 15 pairs per class (SE ≈ 0.13 against 0.050), and removes negatives preferentially because they are promiscuous by construction | 200 pairs per allele | `results/crossover_stratified.csv` | verified 31 Aug |
| 28 | **Peptide promiscuity predicts per-allele performance for our model and NetMHCpan but not MHCflurry** — ours −0.455 (p<0.0001), NetMHCpan −0.295 (p 0.0009), MHCflurry +0.055 (ns). MHCflurry's presentation predictor was trained on a different negative construction, so if the effect were biological it should appear there too. That it does not identifies this as a property of **this benchmark's construction** rather than of presentation: peptides shared across many repertoires appear as negatives for every other allele, so the labels are contradictory across alleles | 200 pairs per allele | `results/crossover_stratified.csv` | verified 31 Aug, promoted from Unconfirmed |

---

## Retracted — do not resurrect

- ~~Structure models are strongest precisely where sequence models are weakest
  (equity claim).~~ Did not survive anchor-matched hard decoys; roughly two-thirds
  of the apparent structural signal was anchor recognition. Retracted deliberately.
  If this sentence reappears in a draft, it is a regression, not a finding.

- ~~Validation AUROC 0.9732.~~ Inflated by the multiset negative-sampling artefact.
  The v3 model on regenerated data gives **0.9715**. The difference is small because
  the model uses allele information the peptide-identity prior does not.

- ~~Anchor IC vs per-allele AUROC rho +0.660 / +0.603.~~ ~~Partial +0.648.~~ Both
  inflated by the same artefact, and the partial over-corrected. Clean value on the
  v3 model: **+0.533** (p 2e-10).

- ~~Motif nearest-neighbour distance rho −0.363.~~ ~~Partial −0.502.~~ Clean value:
  **−0.291** (p 1e-03). The effect is real but more modest than previously claimed.
  Both retracted figures survive in `predictors_vs_confound_v2model.csv`; that file is
  kept for provenance and must not be quoted.

- ~~log10(peptide count) rho −0.020.~~ Clean value **−0.118** (p 0.2) — still null,
  which is the load-bearing part.

- ~~Fine-tuned AlphaFold drops 0.24 on our benchmark because of decoy
  construction.~~ That compared v4 structural results against their test set,
  conflating two variables. Isolated: decoys cost **0.038**, alleles cost **0.193**.
  Fine-tuning works reliably and its benefit does not reach motif-isolated alleles.

- ~~RQ2's null is redundancy — whatever structure encodes, sequence already has.~~
  Error overlap gives margin rho +0.223 against a 0.143 chance baseline, so
  complementary signal exists and nine strategies failed to exploit it. Sample size
  is the likelier explanation, now quantified in row 25.

- ~~The sequence model's 0.995 on the three-allele anchor-matched subset shows
  genuine discrimination.~~ A PWM alone scores 1.000 on those 72 complexes. The
  subset is an outlier; across full sets the model sits 0.07–0.08 *below* the PWM,
  which is the real evidence it is not riding the selection criterion.

- ~~The sequence model uses learned positional embeddings and attention pooling.~~
  Never implemented. It is max-pool over a kernel-3 convolution, therefore
  position-invariant in the peptide path. Verified by reading `nn.py`.

- ~~The structural models pick up P5 anchoring where the sequence model misses it.~~
  Too broad. They pick up **B\*08:01's** P5 anchoring. The same mutagenesis on
  HLA-B\*37:01 — which has the *highest* P5 information content of the P5-Arg group
  at 1.97 bits — puts P5 sixth of nine, undistinguished from P4, P7 and P8. Both
  alleles show near-identical P5 preference in their repertoires, so the difference is
  chemical: B\*08:01 chelates through an Asp9/Asp74 charged pair, B\*37:01 through
  His9/Tyr74/Arg97. The defensible claim is one mechanism in one allele.

- ~~Thr69 predicts P5 anchoring.~~ It separates the nine alleles in
  `p5_anchor_residues.csv` perfectly, but those nine were selected for being P5-Arg
  binders or Asp9 carriers. Across all 123 alleles Thr69 gives mean P5 IC 0.45 against
  0.34 for others, Mann-Whitney p 0.34, with 44 alleles carrying it. Not reported, and
  deliberately omitted from the correspondence with Chris.

- ~~Alleles common outside Europe are predicted worse.~~ Read off the per-region
  table (Africa −0.224, Americas −0.223, South & Central Asia −0.303 significant;
  Europe −0.127 not), this is the significant-here-not-there fallacy. The regional
  frequencies are collinear at median rho +0.590, the pre-specified direct test is
  flat at +0.013 (p 0.89), and only one of seven regional tests survives Bonferroni.
  The defensible claim is the global one in row 18, which runs *opposite* to the
  Europe-bias prediction.

- ~~The four folding architectures are substantially independent readouts, so
  averaging them recovers information no single model has.~~ PC1 alone gives 0.9033
  against the four-way mean's 0.9047. The consensus *is* the shared component;
  averaging cancels independent error rather than combining distinct information.

- ~~Temperature scaling leaves AUROC unchanged by construction.~~ True within an
  allele, false pooled: leave-one-allele-out gives each allele its own temperature and
  the pooled AUROC falls 0.930 → 0.901.

- ~~Surface expression explains the HLA-C deficit.~~ Untested when asserted, and now
  tested: within-locus allele expression does not predict per-allele AUROC (+0.057,
  p 0.63). The locus-level ordering remains consistent with an expression mechanism
  but is three points with locus membership confounded against motif breadth, allele
  counts and assay history. Note also that per-allele TPM from D'Antonio et al. does
  **not** reproduce the known surface hierarchy across loci (A 117, B 22, C 75),
  because per-allele transcript is diluted across the many more distinct HLA-B
  alleles — so that dataset cannot be cited for the locus-level claim, which should
  cite the surface-expression literature instead.

---

## Unconfirmed — carried from session notes, not yet traced to an artefact

- **Why common alleles are predicted worse.** Row 18's −0.243 is real and its
  direction is counterintuitive, since frequency correlates with peptide count at
  +0.570 — more training data and worse performance. Two explanations were tested and
  both failed: frequency does not track motif distinctiveness (−0.100, ns), and common
  alleles share *fewer* peptides with other repertoires (−0.189), so it is neither
  promiscuity nor decoy hardness. Report as an open question, not a finding. (Note the overlap figure this bullet refers to has since been promoted as row 28,
  with a sharper reading: the effect is benchmark construction rather than biology,
  evidenced by MHCflurry not showing it. That does not explain the frequency effect,
  which remains open.)

- ~~HLA-C median AUROC ≈ 0.951 vs ≈ 0.977–0.980 for A/B~~ → **resolved**, row 17.

- ~~Partial rho ≈ −0.455 for the HLA-C effect after confound control~~ → superseded.
  The defensible figures are the OLS coefficient −0.0267 and the matched comparison
  −0.0269, both in row 8. The −0.455 does not correspond to either.

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
  validation split itself retains 10.3%. The ~13.9% was neither figure.

---

## Analysis decisions changed after seeing a result

Recorded here rather than left in the git history, because a rule changed after the
numbers are in needs its reasoning visible.

**`external_vs_isolation.py`.** The verdict rule was changed from `len(neg) >= 3` —
three predictors trending negative — to `len(sig) >= 2` with a minimum panel size of
50 alleles, after the 123-allele results were computed. The original was written for
the 15-allele case, where no single correlation was significant (NetMHCpan −0.638
p 0.17, MHCflurry −0.284 p 0.31, ours −0.222 p 0.43) and agreement across models was
the only available evidence; at n=123 each correlation is individually interpretable,
so counting trends is the wrong test. The conclusion is unchanged under either rule:
all three predictors are both negative and significant. The verdict line is advisory
printing, not an analysis step — the correlations and p-values are computed
identically before and after.

**The crossover control was chosen over filtering, and the reasoning matters.** The
obvious control — restrict to peptides unseen in training — was rejected before being
run, on two grounds computed in advance: it leaves 8.7% of validation 9mers giving 53
alleles at a per-allele SE near 0.13, which is *less* power than the unfiltered sample;
and it is not a neutral filter, because negatives are drawn from other alleles'
repertoires and are therefore promiscuous by construction, so filtering removes them
preferentially and leaves a positive-heavy set whose surviving negatives are atypical.
Stratification retains all 123 alleles and tests the same concern. Recorded because
"why not just filter" is the obvious question and the answer is not obvious.

---

## AF3 on fold set v2 — image built, folding blocked

**Built 27 August** via the Sylabs remote builder from `patches/af3_remote.def`. Local
copy at `~/af3build/alphafold3.sif` on the CS filesystem (3.7 GB), also in the Sylabs
library, so it survives local disk problems.

Three build attempts. The first two were killed at `uv run build_data` — Sylabs' free
tier reclaimed the agent for running too long. Removing that step succeeded: AF3 builds
the components database on first run instead, so the first fold is slower and later ones
are unaffected. The image is 3.7 GB rather than the 4.8 GB built locally in August for
the same reason.

**AF3 source commit `c0f97eda2f1f482fd94d3a38bece18c7069b4a5c`.** The v4 folds came from
whatever `main` was on 5 August, probably a different commit — so any v2 result is a
robustness check, not a like-for-like comparison, and needs saying if both appear in one
table.

**Blocked on infrastructure, not on AF3.** Inputs are built and committed
(`fold_sets/af3v2_inputs/`, 144 JSONs and 6 MSAs); parameters download from anywhere;
folding is ~3 hours. What is missing is a machine that can run the container. Beta has a
GPU and under 1 GB free with no path to the CS filesystem. The lab machines have 589 GB
and free cards, but `getent passwd 19298` returns nothing on `gadwall-l`, `cackling-l`
and `smew-l`, so Singularity refuses to start with "Couldn't determine user account
information". Reported to CS support 28 Aug. Note the lab machines also reboot into
Windows on Monday and Thursday evenings (TSG, 28 Aug).

**This is a robustness check on a superseded panel.** Panel v4 is primary, all four
architectures ran on it, and v2 predates AF3's availability. The consensus finding it
would test has separately been shown to be variance reduction (row 22), so a v2
replication would tell us whether variance reduction replicates rather than whether the
architectures carry complementary information.

---

## Open decisions

- [x] **AFND: implement, or amend the literature review.** → **Implemented 30 Aug.**
      The commitment is discharged with a result: no Europe-specific effect (row 18),
      the regional table is one collinear signal (row 19). The literature review should
      report this rather than dropping the commitment.
- [ ] Is AF3 in scope for §3.6, or reported as supplementary?
- [x] RQ2 stacker: logistic regression vs rank-average — **resolved.** Both are null
      and the error-overlap evidence shows partial complementarity, so neither is
      preferred on performance. Report all nine configurations rather than choosing.
- [x] Whether to run AF2 on the remaining RQ3 alleles — **resolved.** AF2 confirmed
      B\*08:01 (P5 first, architecture-independent) and ESMFold2 on B\*37:01 returned
      the bounding negative. No further architecture runs would change the claim.
- [ ] Whether the B\*37:01 negative belongs in RQ3's results or in the discussion as a
      limitation on the B\*08:01 finding. It is arguably the more interesting of the
      two, so probably results.
- [x] Score the sequence model on `validation_sample_123.csv` — **done 31 Aug.** All
      three predictors now come from one dataset (row 20). Our figure moves from
      −0.291 to −0.196 for the reason recorded in row 20b.
- [ ] Introduction chapter — deprioritised pending results