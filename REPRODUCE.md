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

### Geometry carries no reliable signal

`scripts/extract_geometry_af2.py` (AF2, PDB) and `scripts/extract_geometry.py`
(ESMFold2, mmCIF). No refolding needed for AF2 — ColabFold writes relaxed PDBs for
every prediction. ESMFold2 structures only became available after the `to_mmcif`
fix on 31 July.

Three independent measurements, best feature in each:

| run | best feature | AUROC | `n_contacts` |
|---|---|---|---|
| AF2, fold set v2 | `anchor2_contacts` | 0.492 | 0.363 |
| AF2, fold set v4 | `anchor2_contacts` | 0.607 | 0.558 |
| ESMFold2, fold set v4 | `min_anchor_dist2` | 0.643 | 0.584 |

**The sign is not consistent.** On v2 decoys made more contacts than binders
(364 vs 350); on v4 binders made more, under both architectures. Per-allele
`n_contacts` within the ESMFold2 v4 run alone ranges 0.278 to 0.896, with six
alleles showing binders making more contacts and three the reverse. Per-allele
z-scoring changes nothing (v2 0.363 to 0.370, v4 0.558 unchanged), so this is not
a scaling artefact.

**Conclusion: geometry carries no reliable binder/decoy signal.** All three
measurements sit near chance, and the direction flips between panels.

An earlier version of this section reported the v2 result as a consistent inversion
and offered a mechanism for it — anchor-matched decoys seated in the groove but
compressed against the MHC surface, with Amber relaxation forcing contacts. That
explanation does not survive the v4 replication and is withdrawn. The v2 figure was
noise around 0.5.

Geometry is the only feature category computed from coordinates rather than from
the model's self-assessment, so if fold quality carried binding signal this is where
it would appear. It does not.

**A bug worth recording.** `extract_geometry.py`'s `parse_name` tested
`tag == "decoy"`, but every fold set from v2 onward uses `hard` for anchor-matched
decoys. The result is that all complexes are labelled binders and every AUROC comes
out `nan` — loud rather than silent, fortunately. `extract_geometry_af2.py` handled
both tags from the start, so no committed result was affected.

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

### ESMFold2 on fold set v4

216 folds, none failed. Structures written for the first time (the `to_mmcif` fix
landed after the v2 ESMFold2 run), so this is also the first ESMFold2 geometry.

PAE: pooled 0.659 raw, **0.805 per-allele z-scored**, `pae_anchors_ic` best in both.
That is the largest scaling correction yet (+0.146) and the fourth independent
measurement in which the IC-derived anchor definition beats the hardcoded
P2/C-terminus scheme after scale correction — two architectures, two panels.

Confidence: global metrics 0.497-0.550 (ipTM 0.550, pTM 0.497, complex pLDDT 0.544),
localised 0.594-0.639 (`iptm_pep_self` 0.639, `iptm_pep_mhc` 0.635). Replicates the
localisation pattern from v2 on nine different alleles.

Per-allele `pae_anchors`: B\*73:01 0.896, B\*15:18 0.854, B\*39:06 0.854, C\*08:01
0.833, B\*08:01 0.806, B\*15:03 0.792, B\*47:01 0.778, A\*29:02 0.611, B\*37:01 0.535.

One caution on B\*73:01. Its 0.896 and +2.636 mean PAE gap are driven by four decoys
placed very badly indeed (FYSNKEIFL 10.1, LWDLQDRVL 9.8, RSWAYRDSL 9.5, HSMSQPIMV
8.2, against binders all near 3). That is a handful of outliers rather than uniform
discrimination, and should not be read as strong per-allele signal.

**Structural summary across both panels** (best feature, per-allele z-scored where
PAE):

| | v2 (6 alleles) | v4 (9 alleles) |
|---|---|---|
| AF2 PAE | 0.849 | 0.832 |
| ESMFold2 PAE | 0.791 | 0.805 |
| AF2 confidence | 0.756 | 0.727 |
| ESMFold2 confidence | — | 0.639 |
| geometry | 0.492 | 0.607-0.643 |
| AF2 representations | 0.834 | 0.723 |
| **sequence** | **0.921** | **0.930** |

The ordering is stable: PAE and representations lead, confidence follows, geometry
is at chance, and sequence is above all of them on both panels.

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

## 5 August — AlphaFold 3, and the definitive RQ1 comparison

### Getting AlphaFold 3 running on a 24 GB card

The installation docs specify an 80 GB GPU, CUDA 12.6 and roughly 1 TB of disk.
None of that was available: Beta has a 24 GB RTX 4090, CUDA 13.0, and `/home` was
at 99%. It ran anyway, and the reasons each obstacle turned out to be surmountable
are worth recording.

**Memory.** The 80 GB figure is for large inputs. `docs/performance.md` states that
inputs up to 1,024 tokens fit on a single V100 16 GB with unified memory enabled,
and up to 5,120 tokens on an A100 40 GB. Our complexes are 382 tokens
(274 + 99 + 9), so they fit comfortably. A single fold takes **72 seconds** end to
end, 55 of it inference. The relevant settings, which are baked into the image:

    XLA_PYTHON_CLIENT_PREALLOCATE=false
    TF_FORCE_UNIFIED_MEMORY=true
    XLA_CLIENT_MEM_FRACTION=3.2

`--flash_attention_implementation=xla` is also required on anything that is not an
A100 or H100: the Triton kernel supports only those.

**Databases.** AF3 accepts pre-computed MSAs, so the ~630 GB genetic databases are
unnecessary. Passing `--norun_data_pipeline` with alignments supplied per chain
skips the search entirely. This also makes the AF2/AF3 comparison controlled, since
both then see Chris Thorpe's tuned MSAs.

**Container.** AF3 ships a Dockerfile and documents a Singularity route, but that
route requires building the Docker image first and converting it through a local
registry. Beta has no Docker and no root. `patches/af3.def` is a direct translation
of the Dockerfile to a Singularity definition; two Dockerfile features have no
equivalent and were worked around:

- `COPY --from=ghcr.io/astral-sh/uv:0.9.24` — a multi-stage copy. Replaced with
  uv's standalone installer pinned to the same version.
- `RUN --mount=type=cache,target=/root/.cache/uv` — a BuildKit cache mount. Purely
  a build-speed optimisation; dropped.

Everything else, including the HMMER 3.4 build with the jackhmmer sequence-limit
patch, follows upstream. Build with `--fakeroot`, which works because Beta has
subuid/subgid mappings configured. The image is 4.8 GB.

**Disk.** The first build failed at the final `mksquashfs` step: Singularity writes
its scratch to `/var/tmp`, which on Beta is the root filesystem at 99% capacity.
Forcing `TMPDIR`, `SINGULARITY_TMPDIR` and `APPTAINER_TMPDIR` to `/tmp` (63 GB
tmpfs) fixed it. Note tmpfs is RAM-backed, so the 4.8 GB image, 1 GB of parameters
and 1.9 GB of outputs consume memory and vanish on reboot.

**Two runtime gotchas.**

The image's `%runscript` calls `uv run`, which tries to re-sync the project into the
read-only container and fails with *"failed to remove file ... Read-only file
system"*. Use `singularity exec ... /alphafold3_venv/bin/python3 run_alphafold.py`
instead of `singularity run`.

Model parameters are read directly from `af3.bin.zst` in `--model_dir`; no
decompression needed.

### Input construction

`scripts/build_af3_inputs.py`. Two requirements that are not obvious from the docs
and each cost a failed run:

**The MSA's first row must be exactly the query sequence.** `Msa.from_a3m` raises
otherwise. HISTOFold does the same thing — `run_msa_predictions.py` prepends the
target's own sequence via `create_combined_sequence(...)` then
`prediction_msa.replace('###', prediction_string)` — but per complex rather than per
chain. So AF3 needs one a3m per allele, with that allele's MHC chain first, followed
by the template rows' MHC columns.

**The MHC chain must be truncated to 274 residues.** HISTOFold uses
`allele_sequences[slug][0:274]`; our canonical sequences are 275, and passing the
untruncated sequence produces a query/MSA mismatch.

**Empty `templates` lists are required.** Omitting the field means "not yet
computed" rather than "none", and AF3 refuses to featurise with
`ValueError: Protein chain 1 is missing Templates` under `--norun_data_pipeline`.
Likewise `unpairedMsa: ""` and `pairedMsa: ""` for beta-2-microglobulin and the
peptide, which suppresses the database search for those chains.

The MSA split is exact because HISTOFold's alignments are gapped-only: every row in
`len9_v3b.a3m` is 382 characters and no row contains lowercase. A3M uses lowercase
for insertions, which do not occupy alignment columns, so a positional split would
be wrong if any were present. The script checks this and aborts if the invariant
does not hold.

### AF3 output format

Better structured than ColabFold's. `<name>_confidences.json` holds `pae` as an
n_tokens x n_tokens matrix plus `token_chain_ids`, so the peptide is identified
explicitly rather than assumed to be the final rows — an assumption the other
pipelines rely on and which was never independently checked.
`<name>_summary_confidences.json` gives `chain_pair_iptm`, `chain_ptm` and
`chain_pair_pae_min`, the last of which no other model exposes.

`scripts/analyse_pae_af3.py` reads both into the same schema as the other
extractors.

### AF3 results on fold set v4

216 folds, no failures. Raw pooled:

| feature | AUROC |
|---|---|
| `plddt_peptide` | **0.785** |
| `pae_pep_mhc` | 0.765 |
| `iptm_pep_mhc` | 0.764 |
| `pae_anchors_ic` | 0.728 |
| `pae_anchors` | 0.725 |
| `pae_min_pep_mhc` | 0.719 |
| ipTM | 0.641 |
| complex pLDDT | 0.639 |
| pTM | 0.578 |

`plddt_peptide` at 0.785 is the best raw structural figure anywhere in this project.
AF3 is also the only model where whole-interface `pae_pep_mhc` beats the
anchor-localised features on raw pooling, which suggests its interface PAE is better
calibrated and needs the anchor restriction less.

The binder/decoy PAE gaps are much larger than any other model's: HLA-B\*73:01
+3.191, B\*39:06 +1.927, B\*47:01 +1.517, against AF2's typical +0.3 to +0.5 on the
same complexes.

### The definitive RQ1 comparison

All four architectures on fold set v4 (216 complexes, 9 motif-isolated alleles),
per-allele z-scored `pae_anchors_ic`, paired bootstrap against the sequence model:

| model | z-scored AUROC | sequence − structure |
|---|---|---|
| AlphaFold 3 | **0.858** | +0.073 [+0.022, +0.125] |
| AlphaFold 2 | 0.842 | +0.088 [+0.031, +0.149] |
| ESMFold2 | 0.805 | +0.124 [+0.066, +0.187] |
| Boltz-2.1 | 0.745 | +0.185 [+0.119, +0.257] |
| **sequence** | **0.930** | — |

Every margin excludes zero. The ordering tracks model recency exactly, and AF3 has
roughly halved Boltz's deficit — so structural methods are improving, but the newest
and strongest still loses to a sequence model trained on 838,654 eluted ligands.

Three things make this the version to report. It is a single panel, so the models
are compared on identical complexes. It is a single statistic, `pae_anchors_ic`
z-scored per allele, chosen because it is best for every model rather than
cherry-picked per model. And the intervals are tighter than the v2 equivalents
because 216 complexes beat 144.

`pae_anchors_ic` — the IC-derived anchor definition from the 43% survey — is now the
best feature for **six** independent measurements: AF2 v2, AF2 v3b, AF2 v4,
ESMFold2 v4, Boltz v4 and AF3 v4, after scale correction in each. That is a result
about the anchor convention in its own right, not a feature-selection detail.

**The z-scoring caveat still applies.** Per-allele standardisation uses the held-out
set's own mean and standard deviation over 12 binders and 12 decoys, which partly
encodes the class structure. It is transductive, so the structural figures are
optimistic — which makes the negative result stronger, not weaker, since sequence
wins anyway. Raw pooled figures are AF3 0.765, AF2 0.707, ESMFold2 0.659,
Boltz 0.646.

**None of these four models is fine-tuned.** The claim this table supports is that
*off-the-shelf structural confidence* does not outperform sequence. Motmaen et al.
report 0.97 on Class I by fine-tuning. That comparison has since been run directly —
see the 7-10 August section — and the short version is that fine-tuning adds a
reliable +0.09 wherever the allele is familiar and nothing at all on the
motif-isolated alleles of this panel. Note also that AF3's weights terms of use
prohibit using its output to train models intended for commercial application, so
AF3 stays evaluation-only regardless.

## 6-10 August — the fine-tuned comparison, RQ3, and two corrections

### RQ3 sequence half: anchor positions yes, residue preferences only partly

`scripts/rq3_sequence_landscape.py`. Saturation mutagenesis of the sequence model
across seven alleles chosen as representatives of distinct motif classes
(`scripts/select_rq3_alleles.py`, average-linkage clustering on pairwise
Jensen-Shannon PWM distance, k=8, one representative per class by held-out count).

| comparison | result |
|---|---|
| position-sensitivity agreement with the PWM | median rho **+0.817** |
| full landscape agreement | median rho **+0.541** (0.427-0.601) |
| model's top-2 positions within the IC-derived anchors | **6/7 alleles** |

Benny's hypothesis from the 6 August meeting — that the sequence learner is largely
recovering the motif — holds in a specific form: the model recovers the *positional
structure* of the motif very well and the *per-residue preferences* only partly.
That is consistent with 30,465 parameters shared across 123 alleles via a 34-mer
pseudosequence.

**The architecture makes this sharper than it looks.** `PresentationNet` max-pools
over its convolution output, so the peptide path carries no absolute position
information — it can detect a local pattern but not where it occurred. Anchors
emerging at P2 and the C-terminus for six of seven alleles therefore shows they are
recoverable from composition and local trigram context alone.

Two alleles are informative. HLA-A\*26:01 first looked like a disagreement, model
[1,2] against the PWM's [2,9] — but its IC profile is 1.89/1.88/1.90 at P1/P2/P9, a
three-way tie, and `derive_anchors.py` gives it three anchors. The model picked two
of them; the top-2 PWM comparison was the thing at fault. HLA-A\*03:01 is the one
genuine error, model [1,9] against derived [2,9], with IC 0.92 at P1 against 1.36 at
P2.

**A panel-selection note.** A\*02:01 and C\*03:04 are forced. Without the latter the
selection rule prefers data volume and takes A\*24:02 from the HLA-C-dominated
class, leaving no HLA-C at all — the confound in yet another guise. B\*08:01 emerged
independently, as Benny suggested. Note C\*03:04 has the lowest motif isolation in
the study (nn_dist 0.005), so it represents the locus but not the isolated-allele
problem.

### RQ3 structural half: seeds disagree, but the B\*08:01 prediction held

2,064 ESMFold2 folds — 4 alleles x 3 seeds x 9 positions x 19 substitutions plus
wild-type references, scored by interface PAE differenced against each seed's wild
type (`scripts/build_rq3_variants.py`, `scripts/rq3_compare_landscapes.py`).

**Seed stability is the governing result.** Median seed-to-seed agreement **+0.168**,
with negative minima for A\*02:01 (−0.086) and C\*03:04 (−0.137). Three landscapes of
the same allele from different starting peptides barely correlate, so most of what a
single-seed landscape contains is peptide-specific rather than allele-specific.

| allele | seed rho | struct anchors | derived | landscape rho |
|---|---|---|---|---|
| HLA-A\*02:01 | +0.144 | [9, 2] | [2, 9] | +0.262 |
| HLA-B\*08:01 | **+0.533** | **[5, 9]** | [2, 5, 8, 9] | +0.529 |
| HLA-B\*57:01 | +0.191 | [9, 2] | [2, 9] | +0.413 |
| HLA-C\*03:04 | +0.104 | [2, 3] | [2, 9] | +0.088 |

**The B\*08:01 prediction held.** The sequence half predicted that if the structural
landscape peaked at P5 there, the two model types have found different things. P5
ranks first in structural sensitivity, against the sequence model's [9, 2]. P5 is a
derived anchor and the position Chris Thorpe's MSA rebalancing targets. B\*08:01 is
also the only allele with usable seed stability, so the clearest signal comes from
the most reliable measurement rather than from noise.

Structural anchor recovery is worse than sequence: 2 of 4 alleles have their top-2
outside the derived anchors, against 6 of 7 for the sequence model.

**Report as preliminary.** Three seeds is not enough for three of the four alleles,
and the sequence landscapes were averaged over 12 seeds against the structural three,
so the cross-model correlations are not like-for-like. Seeds 4-6 were folded
subsequently; `build_rq3_variants.py` gained `--seed-offset` so the extension pools
without collision.

### SHAP: no more stable than a landscape, and the two agree

`scripts/rq3_shap.py`. Exact Shapley values — 512 coalitions per 9mer, so no sampling
approximation — with each position marginalised against 64 peptides drawn from the
allele's own held-out set rather than masked with a padding token.

The hypothesis was that SHAP would be more seed-stable than a mutational landscape,
since it marginalises over a background rather than conditioning on one seed peptide.
**It is not**: median seed agreement **+0.186** against the landscape's +0.168, with
negative minima on every allele (one at −0.833).

But the two methods agree well with each other — median rho **+0.633**, same top-2
positions for 5 of 7 alleles. So they measure the same thing consistently; that thing
is genuinely peptide-specific.

The reading: which position matters most depends on what sits at the others, which
fits a kernel-3 convolution with max-pooling, where local context is built in.
**Averaging over seeds is therefore necessary rather than optional, and
single-peptide attribution studies would be unreliable for this model class.** That
is a methodological result alongside the biological one.

SHAP recovers the derived anchors for 4/7 alleles against the landscape's 6/7, and
A\*02:01 gives [2,4] where P4 is not an anchor — but seed agreement there is +0.228,
so that top-2 is not stable enough to interpret.

### The fine-tuned comparison: allele composition, not decoy construction

Motmaen et al. published their fine-tuned parameters and training splits.
`scripts/check_motmaen_overlap.py` gives **0 exact allele-peptide pairs across all
360 complexes** and **1 peptide** appearing anywhere in `combo_1and2_train/valid` —
cleaner than any external baseline tested, since MHCflurry overlapped on 121/144 and
NetMHCpan's training data is not public. Leakage would also push the wrong way.

Their weights through our pipeline, across four benchmarks:

| benchmark | alleles | vanilla | fine-tuned | gain |
|---|---|---|---|---|
| their test set (published figures) | 32 | 0.877 | 0.967 | **+0.090** |
| affinity set, measured non-binders | 3 common | 0.813 | 0.916 | **+0.103** |
| fold set v2, anchor-matched | same 3 | 0.787 | 0.878 | **+0.091** |
| fold set v4, anchor-matched | 9 motif-isolated | 0.698 | 0.685 | **−0.013** |

Isolating each variable by holding the other fixed:

- **decoy construction**, alleles fixed: 0.916 → 0.878, a cost of **0.038**
- **allele composition**, decoys fixed: 0.878 → 0.685, a cost of **0.193**

**Allele composition matters roughly five times more than decoy construction.**
Fine-tuning delivers a consistent +0.09 to +0.10 wherever the allele family is
familiar and nothing on motif-isolated alleles.

That is corroborated directly: on our panel the fine-tuned model gives **0.894 on the
five alleles present in their Class I test set against 0.619 on the ten absent**.
Their evaluation covers 5 of our 15 and **no HLA-C at all**. This is the coverage
finding from a third independent direction — first the Atlas-versus-affinity
comparison, then all of IEDB, now the evaluation set of the leading fine-tuned
method.

Training exposure and template availability are partly confounded, since all folds
used templates from `1k5n_alignments.tsv` (B\*27:05, which scores 1.000). But four
alleles dissociate them and both favour exposure: A\*29:02 is in their test set with
no matching template and scores 0.993; B\*37:01 is absent with a template available
and scores 0.444.

**The pipeline reproduces**, so these are our numbers to trust: 400 nonamers from
their own test set give 0.927 through our pipeline against their published 0.967, the
0.04 shortfall attributable to our canonical sequences truncated to 181 rather than
their exact `chainseq`, a subsample, and one shared template alignment.

**And a sequence method beats it in their own evaluation.** NetMHCpan-4.1 scores
0.985 on their test set against the fine-tuned model's 0.967.

Decoy construction still differs — their `mismatches` column shows decoys differing
from the binder by 4-5 substitutions in 91% of cases, ours are anchor-matched — but
on matched alleles it costs 0.038, not the 0.23 first claimed.

### Reproduction notes for `alphafold_finetune`

Runs inside the ColabFold 1.5.5 Singularity image (JAX 0.4.20, haiku 0.0.10), not the
pinned requirements — those specify `jaxlib 0.1.75+cuda11.4`, predating Ada support,
plus `numpy==1.21.0` and `torch==1.10.1`, neither of which has a cp310 wheel.

Seven API-rename patches bridge roughly two years of JAX and haiku churn; all are
documented renames with exact equivalents and none alters numerics
(`patches/alphafold_finetune_modernise.patch`):

- `np.int/float/bool` → builtins (removed in NumPy 1.24)
- `Bio.Data.SCOPData` → `Bio.Data.PDBData` (Biopython 1.80)
- `jax.ops.index_add` → `.at[].add()` (JAX 0.2.22)
- `hk.vmap.require_split_rng = False` (haiku 0.0.10)
- `jax.tree_*` → `jax.tree_util.tree_*`, `tree_multimap` folded into `tree_map`

Targets are **two chains, MHC truncated to 181 plus the peptide**, unlike HISTOFold's
three chains at 274. The 181 comes from the alignment file's declared `target_len` of
190; getting it wrong does not error, it silently misaligns against the template. One
alignment file serves every 9mer target because the mapping is pure positional
identity, which is what `--ignore_identities` licenses.

`run_prediction.py` discards the `BinderClassifier` coefficients (slope −7.90,
intercepts 0.804/0.434), so the output is raw interface PAE rather than a calibrated
probability. Fine for AUROC. About 4 s per complex after a 40 s compilation.

### RQ2 configuration nine: the gated ensemble

`scripts/rq2_gate.py`. Three gates computable at inference time — distance from the
allele's median sequence score, binary entropy of the sequence probability, and
anchor information content — combined through an interaction term, leave-one-allele-
out, against a z-scored sequence baseline so the transformation is not credited to
the gate.

| gate | AUROC | vs sequence | shuffled control |
|---|---|---|---|
| margin | 0.957 | +0.027 [−0.006, +0.066] | 0.956 |
| entropy | 0.957 | +0.028 [−0.006, +0.065] | 0.957 |
| anchor IC | 0.962 | +0.033 [−0.001, +0.070] | 0.962 |
| ungated | 0.958 | +0.028 [−0.005, +0.065] | — |

**Every gate matches its own permutation control exactly**, and all three match
ungated stacking. Any gain comes from combining rather than from gating. The closest
approach to a positive in RQ2 arrives with its own mechanism disproved. Structure
alone reaches 0.916 here because AF3 and AF2 are z-scored together, which is why this
is the tightest configuration — not because gating works.

### Error overlap: the models fail on partly different complexes

`scripts/rq2_error_overlap.py`, 216 complexes, five models, margins computed within
allele.

Sequence versus structure gives a median margin correlation of **+0.223** and a
worst-quartile Jaccard of **0.220** against a chance baseline of 0.143. Above chance
but far from redundant.

**This corrects the earlier account of RQ2's null.** "Whatever structure encodes,
sequence already has" is not supported: complementary signal demonstrably exists and
nine combination strategies still extracted nothing from it. Sample size is the
likelier explanation, consistent with the gated ensemble matching its own permutation
control.

Structural models agree with each other more than with sequence — ESMFold2/Boltz
+0.524, AF3/ESMFold2 +0.398, against +0.169 to +0.263 for any structure-sequence
pair. The two families make distinct kinds of error.

**Asymmetry**, at each model's own best operating point (Youden's J, scores
standardised within allele):

| model | AUROC | sens | spec | gap |
|---|---|---|---|---|
| sequence | 0.930 | 0.944 | 0.852 | +0.093 |
| AF3 | 0.858 | 0.852 | 0.787 | +0.065 |
| AF2 | 0.842 | 0.880 | 0.704 | +0.176 |
| ESMFold2 | 0.805 | 0.861 | 0.676 | +0.185 |
| Boltz | 0.745 | 0.898 | 0.556 | +0.343 |

Every model is better at recognising binders than rejecting decoys, and the gap
tracks weakness — Boltz calls 90% of binders correctly but rejects only 56% of
decoys. **The structural deficit is specifically decoy rejection**, which follows
from anchor-matched decoys genuinely fitting the groove, and connects to the
fine-tuned result. This runs *opposite* to the pMHC-II claim
(biorxiv 2024.10.06.616783) that structure identifies binders better while sequence
filters non-binders better.

**A methodological trap worth recording.** The first version of this test decomposed
AUROC into mean binder and mean decoy margins. That cannot work: both count the same
pairwise comparisons grouped differently and are identical whenever the classes are
balanced. Verified empirically — all five models returned gaps of ±0.000. A
threshold-based decomposition is required.

### Structural consistency: a fifth readout, below the other four

`scripts/structural_consistency.py`, motivated by Kim et al. (Sci Rep 2024) reporting
that complex stability predicts immunogenicity better than affinity does.

| feature | AUROC (z-scored) |
|---|---|
| `pae_asymmetry` | 0.656 |
| `plddt_peptide_std` | 0.584 |
| `pae_peptide_std` | 0.565 |
| `plddt_anchor_gap` | 0.471 |

Against representations 0.834, PAE 0.804, confidence 0.753, geometry 0.492 and
sequence 0.930. A fifth independent way of reading these folds lands below the other
four, which makes RQ1's conclusion harder to attribute to feature choice.

`plddt_anchor_gap` is the informative failure: binders show an anchor-versus-rest
pLDDT gap of 8.03 and decoys 7.89 — both have their anchors modelled better than the
peptide middle, to the same degree, because the decoys satisfy those anchors by
construction.

**This is not the quantity Kim et al. measure.** Theirs is an experimentally measured
complex half-life, and measured stability data does not exist for these alleles —
the coverage gap in a third form. Per-residue pLDDT comes from the CIF `b_factor`
column; ESMFold2's `metrics.json` holds only scalar `complex_plddt` and a single
sample, so no cross-sample variance is available.

### How much of the sequence result is the selection criterion

The circularity documented on 3 August applies to our own model too, since fold sets
v2 and v4 are PWM-selected. Quantified:

| fold set | PWM alone | ours | rho | ours − PWM |
|---|---|---|---|---|
| v2 | 1.000 | 0.921 | 0.690 | −0.079 |
| v4 | 1.000 | 0.930 | 0.739 | −0.070 |
| affinity | 0.938 | 0.921 | 0.771 | −0.017 |
| 3-allele subset of v2 | 1.000 | 0.995 | 0.852 | −0.005 |

**A model recovering the selection criterion would match it.** Ours is 0.07-0.08
below on both PWM-selected sets, and gives the same 0.921 on the affinity set where
the PWM only reaches 0.938. Performance stable across benchmarks of very different
PWM separability is what one expects of a model not riding the selection rule.

The three-allele subset is the outlier and should not be quoted alone. This table
belongs in the Results chapter rather than the limitations, since it answers the
circularity objection directly.

### Verification of the headline numbers

Six independent checks, all passing:

| check | result |
|---|---|
| labels against the fold sets | 0 mislabelled across five models |
| identical complexes scored | 216 common, no duplicates |
| sequence model reproduces | 0.930 from a clean reload |
| PAE sign direction | binders lower for all four architectures |
| split integrity | 52,768 held-out 9mer positives, 0 fold-set binders outside |
| bootstrap stability | +0.073 [+0.022, +0.125] and +0.074 [+0.022, +0.129] |

### Corrections made in this period

Three, all from drafting a conclusion before verifying the diagnostic behind it. All
are in the git history rather than silently amended.

**The v4/v2 allele mismatch.** A decoy-construction comparison put the fine-tuned
model's drop at 0.231 by comparing v4 structural results against v2 sequence results.
Isolating the variables gives 0.038 for decoys and 0.193 for alleles — the opposite
emphasis, and a stronger equity result.

**The PWM diagnostic index bug.** PWM scores were computed per allele in group order
but assigned to rows in original order, inverting HLA-B\*07:02 (binders +3.11, decoys
+12.64) and producing a meaningless 0.779. The corrected figure is 1.000, which
reverses what that diagnostic had been used to argue.

**The rank-based asymmetry decomposition**, described above, which cannot separate
the two error directions at all.

### Naming schemes, now five

A running count, because this has caused silent mislabelling in three separate
scripts: HISTOFold v2 `{slug}_{peptide}`; v3 `{tag}__{slug}__{peptide}`; v4
`NA__{slug}__{peptide}` with the peptide lowercased; `fold_esmfold2.py`
`{tag}__{slug}__{PEPTIDE}` with the case preserved from the CSV; and AF3, which
appends `_YYYYMMDD_HHMMSS` when the output directory already exists. **Labels must
always come from the fold-set CSV, never from the directory name.**