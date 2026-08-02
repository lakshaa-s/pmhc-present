## 30-31 July — RQ1 completed, RQ2 answered, and two benchmark limitations

### Bootstrap confidence intervals

`scripts/bootstrap_auroc.py`, 2000 resamples over the 144 complexes of fold set v2,
all models aligned on the same complexes.

| model | pooled AUROC | 95% CI |
|---|---|---|
| sequence (ours) | 0.921 | [0.872, 0.957] |
| MHCflurry affinity | 0.911 | [0.857, 0.954] |
| MHCflurry presentation | 0.841 | [0.771, 0.901] |
| AF2 `pae_anchors` | 0.782 | [0.705, 0.852] |
| Boltz `pae_anchors` | 0.738 | [0.653, 0.819] |
| ESMFold2 `pae_anchors` | 0.689 | [0.601, 0.774] |

**Paired differences vs sequence**, bootstrapped on the same complexes:

- ESMFold2 +0.232 [0.151, 0.315] — differs
- Boltz +0.183 [0.099, 0.269] — differs
- AF2 +0.139 [0.062, 0.218] — differs
- MHCflurry presentation +0.079 [0.024, 0.140] — differs
- MHCflurry affinity +0.009 [-0.038, 0.061] — **does not differ**

Sequence significantly outperforms structural confidence across three independent
folding architectures. Our model is statistically indistinguishable from
MHCflurry's affinity predictor.

**Median per-allele CI width 0.325** against pooled 0.139. HLA-C\*16:02 spanning
0.576-0.944 across architectures fits inside a single interval, so cross-model
disagreement at n=24 is what sampling noise predicts. Quote this alongside any
per-allele table.

### Global confidence is uninformative; localised confidence is not

`scripts/extract_confidence.py`. The earlier claim that ipTM and pLDDT carry no
signal was measured on the 60-complex easy-decoy set at n=6, before the split fix.
Retested on fold set v2: it holds for *global* metrics and fails for localised ones.

| model | best global | best localised |
|---|---|---|
| ESMFold2 | ipTM 0.609 | ipTM peptide→MHC chain pair **0.653** |
| Boltz | PDE 0.610 | interface PDE **0.661** |
| AF2 | pLDDT 0.624 | peptide-region pLDDT **0.753** |

Boltz gives the cleanest version, both comparisons within a single forward pass:
interface pLDDT 0.564 vs global 0.532; interface PDE 0.661 vs global 0.610.

ESMFold2 full table: ipTM 0.609, pTM 0.571, complex pLDDT 0.598, ipTM MHC→pep
0.621, **ipTM pep→MHC 0.653**, ipTM pep–pep 0.648. Whole-complex ipTM has a
binder/decoy mean gap of 0.0014; the peptide→MHC pair has 0.0556, forty times
larger.

AF2 full table: ipTM 0.523, pTM 0.530, complex pLDDT 0.624, **peptide pLDDT 0.753**.
Whole-complex ipTM is indistinguishable from chance.

**Statement:** structural confidence carries binder/decoy signal only when
localised to the peptide or its interface. Global confidence is dominated by the
MHC fold, which is predicted well regardless of what occupies the groove. Four
quantities, three architectures, one direction — converging with the PAE result
(whole-interface 0.651-0.739, anchor-localised 0.689-0.804). This supersedes the
flat "confidence is uninformative" claim, which was true of whole-complex ipTM
specifically rather than of confidence generally.

Note `auroc_structure.py`'s per-allele table only prints `pae_anchors` and
`n_contacts`, so confidence runs show empty per-allele rows. Values are in the CSV.

### Geometry carries no signal

`scripts/extract_geometry_af2.py`, run on AF2's relaxed rank_001 PDB structures —
no refolding needed, ColabFold writes them for every prediction.

Best feature is `anchor2_contacts` at **0.492**, chance. Most features fall below
0.5: `n_contacts` 0.363, `contacts_per_res` 0.363, `n_contacts_close` 0.407,
`anchorC_contacts` 0.419, `min_anchor_dist2` 0.462, `min_anchor_distC` 0.466.

`n_contacts` is inverted — decoys make *more* contacts than binders (364 vs 350) —
and consistently so across five of six alleles, with C\*15:05 at 0.132 and C\*16:02
at 0.167. Plausible reading: hard decoys match the target's anchors by
construction, so AF2 seats them in the groove; lacking correct non-anchor
complementarity they may be modelled as compressed against the MHC surface, and
Amber relaxation pushes atoms into contact regardless of whether the pose is right.

**This completes RQ1's structural coverage.** Three feature categories on the same
144 complexes: PAE 0.804, confidence 0.753, geometry 0.492, against sequence 0.921.
Geometry is the only category computed from coordinates rather than from the
model's self-assessment, so if fold quality carried binding signal this is where it
would appear. It gives nothing.

### Fold quality is not the structural ceiling

`scripts/fold_quality_control.py`. AUROC is decomposed into a per-complex margin —
the fraction of opposite-class complexes within the same allele that a given
complex is correctly ordered against, which averages to the allele's AUROC — and
confidence is compared between correctly and incorrectly ordered complexes.

No confidence feature positively distinguishes them, in ESMFold2 or AF2. Against
the continuous margin, all |rho| < 0.19 and only AF2's `ptm` reaches p < 0.05
(rho +0.182, p = 0.029), which is about what eleven tests across two architectures
would produce by chance.

**An earlier version of this analysis was wrong and the correction matters.** A
binary correct/incorrect split initially showed three ESMFold2 features differing
significantly — but in the *wrong* direction, with incorrectly ordered complexes
scoring *higher* confidence. Those three features correlate 0.62-0.82 with the
score used to define the margin, so they were partly proxies for the classifier
itself rather than independent measures of fold quality. The script now reports
that correlation explicitly and only counts a positive difference as evidence.

Four independent lines now support the conclusion:

1. **Within a model** — confidence does not predict which complexes are correctly
   ordered (above).
2. **Across models** — from Chris's RMSD benchmark, Boltz is threefold more
   accurate (0.59 Å vs 1.75 Å) yet discriminates worse (0.738 vs 0.804).
3. **Varying the MSA within a model** — v3b gives better structures but no
   detectable change in discrimination (below).
4. **From the literature** — King et al. (arXiv:2512.06592) retrained Boltz-2 on
   experimentally determined structures with no improvement to affinity regression.

The limitation is in what the structural representations encode, not in how
accurately the fold is predicted.

### AF2 with v3b MSAs

Chris's P5-rebalanced MSAs, same 144 complexes, only the MSA changed.

| feature | v2 | v3b |
|---|---|---|
| `pae_pep_mhc` | 0.739 | 0.742 |
| `pae_anchor2` | 0.779 | 0.797 |
| **`pae_anchors`** | **0.782** | **0.804** |
| `pae_anchorC` | 0.723 | 0.764 |
| `pae_anchors_ic` | 0.723 | 0.771 |

Every feature improved, but the paired bootstrap difference is **-0.023
[-0.088, +0.040]** — not distinguishable at n=144. Report the direction, not a
claim of improvement.

Leakage: v3b is clean against the fold set (0 exact, 0 within two substitutions,
630 MSA peptides vs our 138). v3a has one exact match (TSDKPGSPY under
hla_a_36_01, one of our C\*16:02 decoys) and one near neighbour; both decoys,
neither sharing an allele, so minimal and conservative in direction.

Also checked against PDB after Chris noted Boltz's recent training cutoff: **0 of
138 fold-set peptides appear as a chain in `pdb_seqres`**, so structural leakage is
ruled out for all three folding models.

HISTOFold v3 changed its output directory naming from `{allele_slug}_{peptide}` to
`{tag}__{allele_slug}__{peptide}`; `analyse_pae_af2.py` handles both.

### RQ2: no synergy, and the per-allele pattern is a ceiling effect

`scripts/rq2_stack.py`, logistic regression with heavy L2, leave-one-allele-out CV.

| features | combined AUROC | vs sequence alone |
|---|---|---|
| 21 (all models) | 0.851 | -0.046 [-0.105, +0.011] |
| **5 (AF2 PAE only)** | **0.904** | **+0.006 [-0.043, +0.056]** |

The 21-feature version overfits at n=144; with a sensible feature set the
difference is nil. **RQ2's answer: no detectable synergy.**

The per-allele benefit is perfectly monotone in sequence performance — costing
0.125 on B\*27:05 (sequence 1.000) and gaining 0.125 on C\*15:05 (sequence 0.806) —
which is the King et al. complementarity pattern. But:

- gain vs sequence-only AUROC: **rho -0.899, p = 0.015**
- gain vs structure-only AUROC: **rho 0.600, p = 0.21**

Benefit tracks how much headroom exists, not how good structure is on that allele.
B\*27:05 has the *worst* structural score (0.632) and the largest loss; C\*16:02 has
the *best* (0.972) but gains less than C\*15:05, which has middling structure. This
is regression to the mean, and a gated ensemble built on it would be gating on
headroom rather than on when structure is trustworthy.

Note on citing King et al.: their evidence for the differential effect is two
models with no confidence intervals and no significance test, and against their
reported baselines the gains are near-identical (+0.017, +0.016). Cite as
"suggestive" rather than established. Controlling for the ceiling effect is not
something that literature does, and doing it properly would be a contribution.

### External sequence baselines, and why the comparison is compromised

Four external predictors on fold set v2:

| model | pooled AUROC | rho(PWM score, model score) |
|---|---|---|
| MixMHCpred 3.0 | 0.999 | 0.909 |
| NetMHCpan-4.1 | 0.961 | 0.773 |
| MHCflurry affinity | 0.911 | 0.718 |
| **ours** | **0.921** | **0.690** |
| MHCflurry presentation | 0.841 | 0.621 |

**PWM score alone separates fold set v2 at AUROC 1.000**, because binders are its
top decile and decoys are below its 25th percentile. Coupling to that criterion
tracks AUROC almost perfectly across the five models (Spearman ≈ 0.9). Models that
resemble a PWM score highest, close to mechanically.

So MixMHCpred's 0.999 is circular rather than impressive — it trains on Gfeller-lab
immunopeptidomics, the same lineage as the Atlas. And NetMHCpan's apparent
advantage over our model is largely attributable to closer alignment with the
selection criterion. Ours is the only model meaningfully above the trend line.

**The sequence-model comparison must be reported with the coupling figures
alongside**, or it is misleading.

Allele coverage is a second problem. Both NetMHCpan and MixMHCpred substitute
C\*03:03 for C\*03:04 — neither models it directly. NetMHCpan additionally uses
C\*16:01 for C\*16:02 (distance 0.047, the only non-zero in the panel). So
comparisons on those alleles are really comparisons on their neighbours.
NetMHCpan's distance is in pseudosequence space, and our own analysis found
pseudosequence distance does not predict per-allele performance (rho -0.021) while
motif distance does (-0.363).

MHCflurry's training data is public, so overlap is checkable: 121/144 fold-set
peptides appear in it, and AUROC is *lower* on those (0.824) than on the 23 unseen
(0.937). Training exposure depresses rather than inflates, because hard decoys are
real ligands the model scores high. Same direction as our own model (0.887 vs
0.951). NetMHCpan-4.1's training data is not public, so `in_train` cannot be
filled for it.

NetMHCpan is installed as a symlink to yjchoi's licensed copy at
`/home/yjchoi/dissertation/tools/netMHCpan/`; we are not licensed independently.

### The PWM-free fold set, and why it does not resolve the comparison

`scripts/select_fold_set_affinity.py` builds `fold_sets/fold_set_affinity.csv`:
72 complexes, 3 alleles, binders < 50 nM and non-binders > 5000 nM from MHCflurry's
curated affinity data, every peptide absent from the Atlas. No PWM anywhere in
selection, and the negatives are experimentally measured rather than constructed.

PWM alone scores **0.938** here against 1.000 on fold set v2, so the circularity is
reduced but not eliminated. Real binders do have canonical motifs; the difference
is that separation is now earned rather than guaranteed.

But it does not fix the NetMHCpan comparison:

| | fold set v2 | affinity set |
|---|---|---|
| PWM alone | 1.000 | 0.938 |
| NetMHCpan | 0.961 | **0.992** |
| ours | 0.921 | 0.921 |

Reducing the selection advantage made NetMHCpan *better*. NetMHCpan-4.1 trains on
both eluted-ligand and affinity data, and this set is drawn entirely from affinity
measurements, so these peptides are very likely in its training set — unverifiable,
since that data is not public. **Neither benchmark gives a fair comparison against
NetMHCpan**, for two different reasons.

Our model gives 0.921 on both, unchanged, which is the more interesting
observation: stable across two benchmarks built on completely different principles.

Caveat: `score_sequence_on_foldset.py` reports "100% leakage" on the affinity set.
This is a false alarm — those peptides are absent from the Atlas entirely, so they
are in neither the train nor the validation split, and the script has no category
for that. They are genuinely unseen.

The affinity set covers no HLA-C allele, so it says nothing about the equity
question. AUROCs from it are also not comparable to fold set v2, whose decoys are
anchor-matched and deliberately hard.

### The coverage gap is specific to data type

`data/processed/data_type_coverage.csv`. Median 9mers per allele across all 123
Atlas alleles:

| locus | mass spec (Atlas) | mass spec (MHCflurry) | binding affinity |
|---|---|---|---|
| HLA-A | 1,583 | 1,827 | 1,357 |
| HLA-B | 1,356 | 1,349 | 35 |
| HLA-C | 1,760 | 2,329 | 23 |

Mass spectrometry covers all three loci comparably — HLA-C is in fact best
represented, which is why the earlier "HLA-C is not data-poor" finding held.
Binding affinity shows a 40-60x disparity in the opposite direction.

| locus | alleles with no affinity data | median | best-covered allele |
|---|---|---|---|
| HLA-A | 7 (19%) | 1,357 | 13,222 |
| HLA-B | 18 (29%) | 35 | 4,593 |
| HLA-C | 10 (42%) | 23 | 519 |

The best-covered HLA-C allele falls below the median HLA-A allele. Six of the eight
most-measured alleles are HLA-A.

There is a further layer. Of C\*03:04's 99 affinity measurements, **100% are
placeholder-valued** — exactly 100 or 5000 nM, which encode qualitative
binds/does-not-bind calls rather than measured affinities — against roughly 30%
placeholder for HLA-A and HLA-B. No real dissociation constant has been measured
for any C\*03:04 peptide. This is why C\*03:04 was excluded from the affinity fold
set: mixing qualitative and quantitative measurements within one benchmark would
make its AUROC mean something different from the others'.

**Mechanism:** mass spectrometry is untargeted and captures whatever a cell line
presents; affinity assays require deliberate selection of peptide-allele pairs, and
those choices have followed research attention toward HLA-A. The coverage gap is
therefore not a property of HLA-C biology, nor of data volume in general, but is
specific to deliberately generated data. It also explains why NetMHCpan and
MixMHCpred substitute neighbours for HLA-C alleles, and why MHCflurry's affinity
and presentation predictors behave differently on HLA-C.

### Panel selection: two failed designs before a workable one

`scripts/select_allele_panel.py`. Recorded because the failures are informative.

**v1, stratified on sequence AUROC over all alleles.** The bottom stratum came out
empty. Among alleles with enough held-out 9mers for a canonical fold set (>=120),
sequence AUROC spans only 0.922-0.999: every weak-performing allele is also
data-sparse. **Weak performance and sparsity are confounded**, so the alleles the
project most wants to characterise cannot be properly benchmarked. That is a
finding in its own right and constrains what any panel can demonstrate.

**v2, stratified on anchor information content.** IC is the causal driver (rho 0.660
with AUROC across 123 alleles, against -0.020 for sample size) and is well populated
at the low end among eligible alleles (1.47-3.57 bits). All strata filled — but the
nine selected alleles spanned only 0.964-0.983 in AUROC. That was a *selection
artefact*: taking `nlargest(held_out)` within each IC stratum favours data-rich
alleles, which cluster at high AUROC. The eligible pool was fine; the sampling was
not. (I initially read this as the IC-AUROC correlation breaking down among
data-rich alleles, which was wrong — the correlation is stable at 0.660 / 0.677 /
0.666 / 0.659 across min-candidate thresholds of 0 / 40 / 80 / 120.)

**v3, stratified on sequence AUROC with IC spread within strata.** AUROC is the
scarcer axis — only about a dozen eligible alleles below 0.945 — so it drives the
strata, and a greedy max-min pick spreads IC within each. Result: 15 alleles
spanning AUROC 0.859-0.993 and IC 1.47-3.57, comprising 6 HLA-C, 5 HLA-B, 4 HLA-A,
including A\*30:01, A\*34:01, B\*81:01 and C\*17:01. Written to
`fold_sets/panel_v3.txt`. 9 new alleles × 24 complexes = 216 folds.

**Open question, raised with Benny and Chris:** the panel is selected on statistical
properties rather than population carriage. For a project about ancestral diversity
there is an argument for building it from AFND frequencies with statistical adequacy
as a constraint instead. Chris selects his MSA alleles on cosine dissimilarity of
bound repertoires, which is a third basis. Not folding until this is settled.

### Environment notes

- **ESMC-6B (24 GB in the HuggingFace cache) is a required ESMFold2 dependency**
  and must not be deleted, despite appearing nowhere in this codebase. Deleting it
  causes a silent 24 GB re-download on the next fold.
- `biotite` is required by the geometry scripts and was not previously installed.
- `fold_esmfold2.py` had three bugs in one code path, each masked by the next:
  structures were passed as `results[0].complexes` (the field is `.complex`), then
  written with `to_cif` (the method is `to_mmcif`), then `to_mmcif` was called with
  a path argument it silently discards — it returns a string. A `try/except` around
  the write printed to a log rather than raising, so none of it surfaced.
- HISTOFold has three issues, reported upstream and not yet patched: `os.system`'s
  return code is never checked so failed runs are logged `status=done` and skipped
  permanently on retry; the completeness check expects 26 output files where
  ColabFold 1.5.5 writes 24, so it never fires; and consequently a partial failure
  (9 of 24 files, died during model 3 of 5) was logged as complete. **Always run a
  per-directory file count before analysing a HISTOFold batch.**
- Beta's GPU is shared with two other users and `/home` reached 100% during this
  work. CS lab machines (`*-l` via `knuckles.cs.ucl.ac.uk`) offer RTX 3090 Ti cards
  with networked home directories, so a single setup serves all of them.