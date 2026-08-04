## 30-31 July — RQ1 completed and RQ2 answered

> **Read this first if you are looking at pooled structural AUROCs.** Every pooled
> figure in this section understates the signal by roughly 0.05, for reasons set out
> under "Per-allele scaling" in the 3-4 August section below. The per-allele figures
> and all paired comparisons within a model are unaffected. Corrected values are
> given there.

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

Geometry is the only feature category computed from coordinates rather than from
the model's self-assessment, so if fold quality carried binding signal this is
where it would appear. It gives nothing.

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

A caveat on line 2, which Chris raised directly: whole-peptide RMSD against an
experimental structure and binder/decoy discrimination are different quantities, and
a model could place a peptide accurately while encoding nothing about whether it
belongs in that groove. The observation is also three points, on different peptide
sets, with our Boltz being 2.1 via API against his Boltz-2. Read it as evidence that
the two are decoupled, not that worse structures cause better discrimination.

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
[-0.088, +0.040]** — not distinguishable at n=144. Report the direction as a trend,
not a claim of improvement. (Chris's own reading: "you can say there is a trend".)

Leakage: v3b is clean against the fold set (0 exact, 0 within two substitutions,
630 MSA peptides vs our 138). v3a has one exact match (TSDKPGSPY under
hla_a_36_01, one of our C\*16:02 decoys) and one near neighbour; both decoys,
neither sharing an allele, so minimal and conservative in direction.

Also checked against PDB after Chris noted Boltz's recent training cutoff: **0 of
138 fold-set peptides appear as a chain in `pdb_seqres`**, so structural leakage is
ruled out for all three folding models.

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

### AF2 learned representations

`scripts/rq1_embeddings.py`. Refolded fold set v2 with
`--save-single-representations`, giving a (382, 256) float16 array per model per
complex. The peptide is the final 9 rows; mean-pooled, PCA fitted inside each fold,
logistic regression with heavy L2, leave-one-allele-out.

- mean-pooled: **0.834 [0.765, 0.891]**; vs `pae_anchors` +0.031 [-0.060, +0.115]
  (does not differ); vs sequence -0.087 [-0.164, -0.014] (differs)
- concat-pooled (2304 dims): 0.741, worse, as expected at n=144

So the learned representations land in the same place as the confidence metrics.
The folding model does not appear to encode binding information that its confidence
outputs fail to expose — at least not information a linear classifier can extract
from 144 samples.

**RQ1's structural coverage is now four readouts of the same folds:** PAE 0.804,
representations 0.834, confidence 0.753, geometry 0.492, against sequence 0.921.
The negative result is not an artefact of reading the wrong output. The caveat is
that 144 samples with leave-one-allele-out limits what a linear classifier can
extract, so this shows the signal is not easily accessible rather than that it is
absent — fine-tuning remains untested, and Motmaen et al. suggest it would do better.

## 3 August — external baselines, benchmark circularity, and where the coverage gap actually is

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
those choices have followed research attention toward HLA-A.

An open question from Benny, worth pursuing: does this say anything about which HLA
is dominant in a cell? HLA-C is expressed at roughly ten-fold lower surface levels
than HLA-A and HLA-B, so the fact that mass spectrometry still covers it comparably
needs explaining — either immunopeptidomics protocols compensate, or the read of the
coverage needs qualifying. Checkable in the literature and not yet done.

### The gap is in negatives specifically (IEDB, primary source)

The section above uses MHCflurry's curated dataset. Going to IEDB directly — every
published pMHC assay rather than one curation — sharpens the finding and corrects
its emphasis.

Source: `mhc_ligand_full.zip` from iedb.org, 5,770,781 assay rows, 303 HLA alleles
with at least one 9mer. Summary in `data/processed/iedb_coverage.json`.

**HLA-C is not positive-poor.** Distinct 9mers with a positive qualitative call:

| allele | positives | negatives |
|---|---|---|
| HLA-A\*02:01 | 41,446 | 5,892 |
| HLA-B\*07:02 | 14,236 | 2,665 |
| HLA-C\*05:01 | 7,651 | 76 |
| HLA-C\*12:02 | 7,027 | 3 |
| HLA-C\*03:04 | 6,769 | 52 |
| HLA-C\*16:01 | 3,466 | 0 |
| HLA-C\*15:05 | 89 | 0 |
| HLA-C\*16:02 | 10 | 0 |

Several HLA-C alleles have thousands of known ligands, and the median positive
count is higher for HLA-C than for HLA-A or HLA-B — consistent with HLA-C being
well covered by mass spectrometry.

**What is missing is experimentally determined non-binders.** Restricting to
alleles with at least 100 positives, so barely-studied alleles do not distort the
comparison:

| locus | alleles | zero negatives | median negative:positive ratio |
|---|---|---|---|
| HLA-A | 44 | 11 (25%) | **0.166** |
| HLA-B | 64 | 21 (33%) | **0.0084** |
| HLA-C | 21 | 6 (29%) | **0.0074** |

The zero-negative rate is comparable across loci, so the disparity is not about
whether negatives exist at all — it is about the ratio. A studied HLA-A allele has
roughly one negative per six positives; HLA-B and HLA-C have roughly one per 120.

**The characterisation is therefore HLA-A versus everything else**, not HLA-C
specifically. That corrects the emphasis of the MHCflurry-derived section above.
Note that more HLA-B alleles have been studied (64) than HLA-A (44), so this is not
about attention to loci in general, but specifically about which alleles received
the assays that produce negatives.

**Mechanism.** Purified-MHC and cellular binding assays measure whether a chosen
peptide binds and therefore generate negatives. Mass spectrometry elutes what a
cell presents and by construction yields positives only. The binding assays were
developed for and applied predominantly to HLA-A\*02:01 and its relatives, driven
by vaccine and immunotherapy work; other alleles have been characterised mainly by
immunopeptidomics.

**Consequence for this project.** The anchor-matched constructed decoys in fold set
v2 were a necessity, not a convenience. For HLA-C\*15:05 and HLA-C\*16:02 there are
zero experimentally determined non-binders in the whole of IEDB, so no benchmark
for those alleles can use real negatives. This is also why
`select_fold_set_affinity.py` could build a PWM-free fold set for HLA-A and HLA-B
but not for HLA-C. It follows that the equity question cannot currently be answered
with experimentally grounded negatives for the rare HLA-C alleles by anyone, with
any method — a statement about the state of the field rather than a limitation of
this work.

Chris's view on how to use this: "If there is not enough data on ancestrally diverse
alleles then that should be called out explicitly in your write up. It is a proof
that we need to generate more data about these alleles for the purpose of ML
development."

**Caveat on the extracted file.** `iedb_coverage.json` has four fields per allele:
`pos`, `neg`, `ms`, `binding`. Only `pos` and `neg` are trustworthy. The method
classification used `('mass spectrometry' in m and ms or bind)`, which never
selects `ms` because an empty `defaultdict` is falsy, so every peptide was counted
as `binding` and `ms` is zero throughout. The positive and negative counts were
verified against an independent pass over the same file and match exactly.

Column indices in the IEDB export (two-row header, names from row 2): peptide 11,
assay method 90, qualitative measurement 94, measurement inequality 95,
quantitative measurement 96, MHC allele 107.

The download is 284 MB compressed and needs several GB to process. It was extracted
on a CS lab machine (`gadwall-l`) using local `/tmp`, since Beta's `/home` was at
99% and the CS networked home has a 10 GB quota. Only the 24 KB summary was copied
back.

## 3-4 August — panel v4 by motif isolation, and a scaling artefact in every pooled figure

### Panel design: three rejected criteria before the one both supervisors endorsed

`scripts/select_allele_panel.py` (v1-v3) and `scripts/select_allele_panel_motif.py`
(v4). The failures are recorded because each identifies a real property of the data.

**v1, stratified on sequence AUROC over all alleles.** The bottom stratum came out
empty. Among alleles with enough held-out 9mers for a canonical fold set (>=120),
sequence AUROC spans only 0.922-0.999: every weak-performing allele is also
data-sparse. **Weak performance and sparsity are confounded**, so the alleles the
project most wants to characterise cannot be given a fair fold set. That is a
finding in its own right and constrains what any panel can demonstrate.

**v2, stratified on anchor information content.** IC is the causal driver (rho 0.660
with AUROC across 123 alleles, against -0.020 for sample size) and is well populated
at the low end among eligible alleles (1.47-3.57 bits). All strata filled — but the
nine selected alleles spanned only 0.964-0.983 in AUROC. That was a *selection
artefact*: taking `nlargest(held_out)` within each IC stratum favours data-rich
alleles, which cluster at high AUROC. The eligible pool was fine; the sampling was
not. (This was initially misread as the IC-AUROC correlation breaking down among
data-rich alleles, which was wrong — the correlation is stable at 0.660 / 0.677 /
0.666 / 0.659 across min-candidate thresholds of 0 / 40 / 80 / 120.)

**v3, stratified on sequence AUROC with IC spread within strata.** Fixed the
artefact and gave 15 alleles spanning AUROC 0.859-0.993 and IC 1.47-3.57, 6 HLA-C /
5 HLA-B / 4 HLA-A. Written to `fold_sets/panel_v3.txt`. Not folded — the design
question was put to both supervisors first.

**v4, stratified on motif isolation.** Both supervisors independently favoured this.
Benny Chain: "this will test the influence of structural modelling on the most
different alleles, which targets the objective of increasing the coverage of the HLA
space. I don't think performance or data quality should be the primary criteria in
the context of your project." Chris Thorpe selects his MSA alleles on cosine
dissimilarity of bound repertoires and agreed.

There is direct support from earlier work here: across 123 alleles, motif
nearest-neighbour distance predicts per-allele AUROC (rho -0.363, p 3.7e-5) while
pseudosequence distance does not (rho -0.021, p 0.82), and motif distance survives
controlling for pseudosequence distance (-0.417).

`select_allele_panel_motif.py` stratifies on Jensen-Shannon nearest-neighbour
distance from `data/processed/motif_distinctiveness.csv`, weighted 3:2:1:1 toward
the isolated end, subject to at least 120 held-out 9mers.

Panel (`fold_sets/panel_v4.txt`), 15 alleles spanning nn_dist 0.005-0.138:

| allele | nn_dist | nearest | seq AUROC | held-out | status |
|---|---|---|---|---|---|
| HLA-B\*37:01 | 0.138 | B\*47:01 | 0.968 | 199 | to fold |
| HLA-B\*47:01 | 0.138 | B\*37:01 | 0.993 | 560 | to fold |
| HLA-B\*73:01 | 0.138 | B\*27:04 | 0.999 | 127 | to fold |
| HLA-B\*08:01 | 0.114 | B\*14:02 | 0.966 | 952 | to fold |
| HLA-C\*16:02 | 0.113 | C\*16:01 | 0.859 | 39 | folded |
| HLA-B\*39:06 | 0.110 | B\*39:24 | 0.991 | 245 | to fold |
| HLA-B\*15:18 | 0.102 | B\*38:01 | 0.978 | 200 | to fold |
| HLA-A\*29:02 | 0.095 | A\*30:02 | 0.959 | 254 | to fold |
| HLA-C\*15:05 | 0.074 | C\*15:02 | 0.889 | 44 | folded |
| HLA-B\*15:03 | 0.070 | B\*15:01 | 0.980 | 435 | to fold |
| HLA-C\*08:01 | 0.037 | C\*03:04 | 0.952 | 267 | to fold |
| HLA-A\*02:01 | 0.031 | A\*02:04 | 0.967 | 2946 | folded |
| HLA-B\*27:05 | 0.030 | B\*27:04 | 0.993 | 975 | folded |
| HLA-B\*07:02 | 0.026 | B\*42:01 | 0.975 | 1346 | folded |
| HLA-C\*03:04 | 0.005 | C\*03:03 | 0.951 | 986 | folded |

Checks on the selection. The three most isolated alleles all sit at nn_dist 0.138,
which could indicate a mutually-isolated cluster rather than individual
distinctiveness — but `mean_dist_3` is 0.139-0.146 for all three, barely above
`nn_dist`, so their second and third neighbours are about as far as their first.
They are genuinely isolated. Eight panel alleles have fewer than 400 held-out
9mers, where the subsampling control (`data/processed/subsample_pwm_noise.csv`)
puts PWM noise at roughly 0.013 of nn_dist, about 9% of the observed spread.

**A limitation to state.** The nine new alleles span sequence AUROC 0.952-0.999 —
narrower than v3's 0.926-0.983. Selecting on isolation did not pull in
poorly-performing alleles, because the candidate floor excludes exactly the
isolated-and-sparse alleles that drive the -0.363 correlation. This is the same
confound in a third guise. The panel answers "does structural benefit track motif
isolation" and not "does structure help where sequence is weak".

### Fold set v4

`fold_sets/fold_set_v4.csv`: 216 complexes, 9 new alleles, 12 canonical binders
(top decile by motif score) and 12 anchor-matched hard decoys (`--max-pctile 25`)
each. Same scripts as v2; the six already-folded alleles are reused.

Binder quality varies with pool size. B\*73:01 has 127 held-out candidates and a top
decile of exactly 12, so its binders are the entire decile with no selection
headroom and one sits at the 91st percentile — the same limitation C\*15:05 and
C\*16:02 carry. B\*37:01 (199) and B\*15:18 (200) are only slightly better. Decoy
pools are healthy throughout, the smallest being B\*39:06 at 1,257 candidates.

B\*73:01's anchors come out as P2:{RSWY}, P9:{ALPV}. The proline preference at the
C-terminus is atypical for class I and initially looked like a motif artefact from
127 peptides, but it appears in both binder and decoy anchor definitions, so it is
a genuine feature of this evolutionarily divergent allele.

**HISTOFold input format.** The v3 code reads its prediction list with a
`DictReader` and requires a three-column header:
`allele_slug,peptide_sequence,pdb_id`. Our five-column tagged format needs
converting, and the `pdb_id` column is written into the output directory name — so
with `pdb_id` set to "NA" the directories become `NA__{slug}__{peptide}` and carry
no binder/decoy label. This is the fourth naming scheme seen from HISTOFold;
`analyse_pae_af2.py` now handles all four, and labels must come from the fold-set
CSV rather than the directory name. Getting this wrong silently labels every
complex a binder, which shows up as `decoy nan` in the per-allele table.

### Per-allele scaling: every pooled structural figure was understated

On fold set v4, pooled `pae_anchorC` gives 0.707 while per-allele AUROCs run
0.708-0.917 with a mean near 0.81. The feature discriminates well *within* alleles
and the pooled figure is dragged down by between-allele scale differences: PAE
magnitude varies by allele (B\*73:01 binders average 4.96, B\*15:18 binders 3.52),
so pooling ranks a low-PAE allele's decoy above a high-PAE allele's binder on offset
alone.

Standardising PAE within each allele before pooling:

| model | pooled | per-allele z-scored | best feature (z) |
|---|---|---|---|
| AF2 v3b | 0.804 | **0.849** | `pae_anchors_ic` |
| AF2 v2 | 0.782 | 0.829 | `pae_anchors_ic` |
| ESMFold2 | 0.734 | 0.791 | `pae_anchor2` |
| Boltz | 0.738 | 0.761 | `pae_anchorC` |
| AF2 v4 panel | 0.707 | 0.832 | `pae_anchorC` |

Gains of 0.023-0.125 across every model, so this is a property of the metric rather
than of any one architecture.

**RQ1 survives but the margin narrows.** Sequence 0.921 against AF2 z-scored 0.849,
paired difference **+0.072 [+0.005, +0.143]**, stable across five bootstrap seeds
(lower bound 0.004-0.006). The raw comparison was +0.139 [0.062, 0.218]. The
direction holds either way, but with the best available structural configuration
the result is just significant rather than comfortably so.

**Caveat, and it matters.** Per-allele z-scoring uses the held-out set's own mean
and standard deviation over 12 binders and 12 decoys. Standardising within a
balanced set partly encodes the class structure, which makes this transductive
rather than a legitimate inference-time transformation. The defensible range is
therefore **0.804 to 0.849**, and both figures should be reported. A
non-transductive version — standardising each allele against an independent
reference distribution, for instance folds of training-split peptides for that
allele — is needed before the higher number can be claimed.

**A second observation worth pulling out.** Once between-allele scale is removed,
`pae_anchors_ic` becomes the best AF2 feature, where raw pooling favoured
`pae_anchors`. That is the IC-derived anchor definition from the 43% survey
outperforming the hardcoded P2/C-terminus scheme. The effect was previously buried
under scale differences. This connects to the anchor-convention finding and to
Chris Thorpe's P5 rebalancing work, and is worth reporting as a result in its own
right rather than as a feature-selection detail.

### Fold set v4 results (AF2, v3b MSAs)

All 216 folds complete, none logged as fast failures. Pooled: `pae_anchorC` 0.707,
`pae_anchors_ic` 0.698, `pae_anchors` 0.695, `pae_pep_mhc` 0.678, `pae_anchor2`
0.651. Per-allele `pae_anchors`:

| allele | AUROC | binder/decoy PAE gap |
|---|---|---|
| HLA-B\*47:01 | 0.917 | +0.838 |
| HLA-B\*39:06 | 0.875 | +0.443 |
| HLA-B\*15:03 | 0.854 | +0.490 |
| HLA-B\*15:18 | 0.819 | +0.340 |
| HLA-B\*08:01 | 0.806 | +0.518 |
| HLA-A\*29:02 | 0.799 | +0.498 |
| HLA-C\*08:01 | 0.764 | +0.477 |
| HLA-B\*73:01 | 0.757 | +1.353 |
| HLA-B\*37:01 | 0.708 | +0.333 |

B\*73:01 has by far the largest raw PAE gap (+1.353) but a middling AUROC (0.757) —
another illustration of why raw magnitudes and the ranking statistic diverge.

Still to run on this panel: confidence metrics, geometry, embeddings, and the
sequence baseline (`results/sequence_v2.csv` covers only the original six alleles).
The correlation the panel was built to test — does structural benefit track motif
isolation across the 15 alleles — needs the sequence baseline first.

### AlphaFold 3

The request form has been retired; parameters are a direct download from
`storage.googleapis.com/alphafold3/af3.bin.zst` (974 MB), subject to the weights
terms of use. The genetic databases are the obstacle at roughly 630 GB, but AF3
accepts pre-computed MSAs via `unpairedMsaPath`, so Chris's templates can be
supplied directly — which also keeps any AF2/AF3 comparison controlled, since both
would use identical MSAs.

On licensing: AF3 output may not be used to train models intended for commercial
application. Benny's view: "I am fairly relaxed about this. I doubt we will use the
output in a commercial application. But happy for you to keep AF3 for evaluation."
So AF3 stays evaluation-only and out of anything used for training.

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
- HISTOFold has three issues, reported upstream and being fixed: `os.system`'s
  return code is never checked so failed runs are logged `status=done` and skipped
  permanently on retry; the completeness check expects 26 output files where
  ColabFold 1.5.5 writes 24, so it never fires; and consequently a partial failure
  (9 of 24 files, died during model 3 of 5) was logged as complete. This bit three
  times, once invalidating an entire 144-complex run in 1.5 seconds per complex.
  **Always run a per-directory file count before analysing a HISTOFold batch** — 24
  files normally, 29 with `--save-single-representations`.
- Beta's GPU is shared with two other users and `/home` reached 100% during this
  work. CS lab machines (`*-l` via `knuckles.cs.ucl.ac.uk`) offer RTX 3090 Ti cards
  with networked home directories, but those homes have a 10 GB quota — too small
  for the ESMFold2 model cache. Local `/tmp` on each machine has several hundred GB.
  Note `/tmp` on Beta is tmpfs (RAM-backed, 63 GB), so writing there consumes memory
  and is cleared on reboot.
- Setting `OUTPUT_FOLDER` in HISTOFold's `local.toml` breaks the container path
  construction: the a3m is written to `{OUTPUT_FOLDER}/tmp/` but passed into the
  container as `/work/{item_path}`, where `/work` is the HISTOFold directory. The
  run then fails instantly for every complex — and, because of the `os.system` bug,
  logs all of them as done.

  ### Why the panel cannot test the hypothesis it was built for

The v4 panel was selected to test whether structural features contribute more for
motif-isolated alleles. On the nine new alleles, and on all fifteen once the
originals are included, the answer is a flat null:

| relationship (n=15) | rho | p |
|---|---|---|
| nn_dist vs structural AUROC | -0.138 | 0.624 |
| nn_dist vs (sequence - structure) gap | -0.050 | 0.859 |
| nn_dist vs sequence AUROC | -0.222 | 0.426 |

This is **not** evidence against the hypothesis. The benchmark cannot detect an
effect of this size, and the arithmetic is worth setting out because it applies to
every per-allele claim in this project.

**The signal is small.** Across 123 alleles, per-allele validation-split AUROC has a
standard deviation of **0.024** (range 0.859-0.999). That is the entire
between-allele variation available to correlate against.

**The measurement is noisy.** Fold-set AUROC is computed from 12 binders and 12
decoys. The bootstrap gives a median per-allele CI width of 0.325, so a standard
error near **0.075** — roughly three times the between-allele spread.

With signal sd 0.024 and measurement noise 0.075, attenuation caps the observable
correlation at around 0.3 even if the underlying relationship were perfect. The
observed validation-split vs fold-set AUROC correlation is **+0.431 (p 0.108,
n=15)**, which is at or above that ceiling — so the two measures are plausibly
tracking the same quantity and the shortfall is measurement error, not a difference
between the tasks.

**Consequences.**

1. The nn_dist nulls are uninformative rather than negative. Benny's hypothesis is
   untested, not refuted.
2. Detecting a between-allele effect would need hundreds of complexes per allele,
   not 24 — roughly an order of magnitude more folding than this project has done.
3. Every per-allele comparison here inherits the same limit: the C\*16:02 spread
   across architectures (0.576-0.944), the RQ2 ceiling-effect correlations, and the
   per-allele tables throughout. Pooled figures and within-allele paired
   comparisons are unaffected.

**A related trap, which caught this analysis before it was checked.** On the 15
panel alleles, log10(peptide count) correlates with fold-set sequence AUROC at
rho +0.564, p 0.028 — apparently supporting the equity claim that structure helps
where training data is thin. Across all 123 alleles the same correlation is
**-0.020, p 0.82**. The n=15 result is a sampling artefact: the panel was selected
for motif isolation subject to a candidate floor, and two low-count alleles
(C\*16:02 at 236 peptides, C\*15:05 at 215) anchor the bottom of the range. Do not
report it.

**What does hold, on the well-powered 123-allele analysis**, and which describes
validation-split performance rather than fold-set performance:

| predictor | rho vs per-allele AUROC | p |
|---|---|---|
| anchor information content | **+0.660** | 1.1e-16 |
| motif nearest-neighbour distance | **-0.363** | 3.7e-05 |
| log10(peptide count) | -0.020 | 0.82 |

The motif-distance correlation is stable under thresholding on peptide count
(-0.363 / -0.385 / -0.361 / -0.380 at thresholds 0 / 200 / 500 / 1000, n=123 down to
88), so it is not confounded with sparsity.

### Sequence baseline on fold set v4

`results/sequence_v4.csv`. Pooled 0.930 (n=216), against AF2 `pae_anchors` 0.695
raw and 0.832 per-allele z-scored.

| allele | sequence | AF2 `pae_anchors` | gap |
|---|---|---|---|
| HLA-B\*15:18 | 1.000 | 0.819 | +0.181 |
| HLA-B\*08:01 | 0.993 | 0.806 | +0.188 |
| HLA-B\*39:06 | 0.986 | 0.875 | +0.111 |
| HLA-B\*37:01 | 0.965 | 0.708 | +0.257 |
| HLA-C\*08:01 | 0.965 | 0.764 | +0.201 |
| HLA-B\*47:01 | 0.951 | 0.917 | +0.035 |
| HLA-B\*15:03 | 0.903 | 0.854 | +0.049 |
| HLA-A\*29:02 | 0.875 | 0.799 | +0.076 |
| HLA-B\*73:01 | 0.875 | 0.757 | +0.118 |

Sequence leads on every allele in this panel. Across all fifteen alleles the only
two where structure wins are HLA-C\*16:02 (gap -0.090) and HLA-C\*15:05 (-0.014) —
both from the v2 panel, and both the sparsest alleles in the study at 39 and 44
held-out 9mers. They sit mid-range on motif isolation (0.113 and 0.074), so what
distinguishes them is data scarcity rather than motif distinctiveness. Given the
noise limit above, two alleles is not evidence, but it is the pattern the equity
claim would predict and it is worth stating as an observation for future work with
a properly powered design.