# External review: analytical gaps in `pmhc-present`

Prepared by re-analysing the result tables already committed to the repository.
Every number below comes from files in git — no refolding, no retraining, no
access to the unpublished raw data. `review_analysis.py` reproduces all of it in
one run and writes `review_summary.json` plus the `review_*.csv` tables;
`review_figure.png` is the figure.

**Reproduction check first.** Before proposing anything I recomputed the headline
RQ1 comparison from `results/sequence_v4.csv` and `pae_*_v4.csv`, per-allele
z-scored, using `pae_anchors_ic` as the structural readout. I get sequence 0.930,
AF3 0.858, AF2 0.842, ESMFold2 0.805, Boltz 0.745 — matching the README. The rest
of this document rests on that verified footing.

Following the convention in `PROGRESS.md`, each finding cites the artefact it came
from, and nothing here should be promoted out of "unconfirmed" without the
follow-up named alongside it.

---

## 1. The one apparent missed finding: the architectures add to each other

The project evaluates four folding architectures **individually** and asks whether
structure adds to sequence. It never asks whether the four architectures add to
**each other**.

They do. Averaging the per-allele z-scored anchor-interface PAE across all four
gives AUROC **0.905**, against
**0.858** for the best single architecture (AF3).
Allele-cluster bootstrap on the paired difference:
**+0.0461 [+0.0029, +0.0910]** —
the interval excludes zero, which is more than can be said for any of the nine
sequence-plus-structure ensembles already tested.

It also moves an RQ1 conclusion. Sequence minus consensus is
**+0.0270 [-0.0135, +0.0808]** —
it spans zero. Against the same cluster bootstrap, sequence minus AF3 is
**+0.0731 [+0.0388, +0.1113]**, which does not. So on fold set v4 a
four-architecture structural consensus is *not distinguishable from* the sequence
model, whereas every individual architecture clearly is. "Sequence beats
structure" survives. "Sequence beats structure by a margin this panel can
resolve" does not survive replacing *structure* with *consensus structure*.

**Why it works.** Mean within-allele Spearman ρ between architectures is
0.52 — essentially the same as the
0.53 between structure and sequence
(`review_within_allele_agreement_v4.csv`). The four architectures are not four
measurements of one quantity; they are four substantially independent readouts, so
averaging cancels a real amount of independent error. That framing is itself a
result worth stating: it says something about folding models that the
per-architecture tables do not.

**Robustness** (`review_consensus_subsets_v4.csv`). Every three-architecture
subset except drop-AF2 and drop-AF3 beats the best single model, and AF2+AF3 alone
reaches 0.916. Not one architecture carrying it.

**The caveat that must ship with it.** The gain does **not** replicate on fold set
v2. Restricting both panels to the three architectures that were run on v2, the
consensus gain is
+0.0237
on v4 but +0.0098
on v2, with a cluster-bootstrap CI of
[-0.0797, +0.0870] spanning zero.
By the project's own standard this sits in "unconfirmed" until AF3 is run on v2 and
the four-architecture consensus is tested there.

**This is the highest-value remaining compute job in the project.** It is 144 AF3
predictions, on a panel whose inputs already exist, and it either confirms or kills
a finding that changes an RQ1 conclusion.

## 2. RQ2's null is underpowered — now quantified rather than asserted

`PROGRESS.md` states that sample size is the likelier explanation for the nine null
ensemble results. That was a reasonable inference; it can be made a number.
Simulating the observed effect (+0.013 AUROC for the best real
blend) at increasing panel sizes, using the same paired bootstrap the project
already uses, gives power **0.37** at the current n=216, crossing
80% near **n≈500**, and reaching 0.92 at n=864
(`review_rq2_power.csv`).

That converts "we found nothing" into "we could not have found this even if it were
real" — a materially stronger claim for a discussion chapter, and a defensible
number for future work.

There is a second, independent bound. From
`results/rq2_error_overlap_margins.csv`: sequence gets only **5 of
216** complexes wrong, and a structural model rescues **4** of
those 5. A per-complex oracle that cheats — picking the best of
all five models for every complex — reaches margin 0.984 against
sequence's 0.946. The entire headroom available to any
router on this panel is roughly four points of margin concentrated in five
complexes. Even a perfect gate cannot produce a large effect here.

That is a ceiling argument, not a power argument, and the two together close RQ2
much more firmly than either alone: the effect is too small for this panel to
detect *and* too small to exist on this panel.

## 3. Calibration is never reported, and the model is badly miscalibrated

The stated motivation is prioritising candidates. Every metric in the project is
rank-based, and AUROC is invariant to any monotone transform of the score — so
nothing reported so far constrains whether the output means anything as a
probability. It does not. On fold set v4 (true prevalence 0.50):
mean predicted P(presented) **0.71**, Brier **0.173**,
ECE **0.211**. The reliability curve
(`review_calibration_reliability_v4.csv`) lies entirely below the diagonal: 123 of
216 complexes score above 0.9, of which 82% are binders, and every bin below 0.7
contains zero binders.

This is genuine overconfidence rather than a prevalence artefact — the validation
split is 50/50 (83,822 positives of 167,642 in `results/per_allele_auroc_v3.csv`),
matching the fold set.

A single parameter fixes much of it. Fitted leave-one-allele-out, so honestly:
T≈2.02, Brier 0.173 → 0.155, ECE
0.211 → 0.177, AUROC essentially unchanged at
0.925. Twenty lines of code, one new results table, and a claim the
project cannot currently make: that its scores are usable as probabilities.

## 4. Report at least one decision-relevant metric alongside AUROC

AUROC weights all thresholds equally; nobody screening candidates operates at the
middle of the ROC curve. Two cheap additions (`review_decision_metrics_v4.csv`):

| model | AUROC | pAUC (FPR≤0.10) | PPV @ top 20 | PPV @ top 54 |
|---|---|---|---|---|
| sequence CNN | 0.930 | 0.697 | 1.00 | 0.889 |
| 4-arch consensus | 0.905 | 0.688 | 0.95 | 0.889 |
| AlphaFold3 | 0.858 | 0.619 | 0.90 | 0.796 |
| AlphaFold2 | 0.842 | 0.633 | 0.95 | 0.833 |
| ESMFold2 | 0.805 | 0.588 | 0.80 | 0.796 |
| Boltz-2.1 | 0.745 | 0.543 | 0.65 | 0.685 |

The ordering is not preserved. AF3 beats AF2 on AUROC
(0.858 vs 0.842) but AF2 beats AF3 in
the high-specificity regime (pAUC 0.633 vs
0.619; PPV@20 0.95 vs
0.90). "AF3 is the best structural model" is therefore
metric-dependent — worth a sentence in the RQ1 write-up, and a caution for anyone
reading the current table as a ranking.

## 5. Two smaller methodological notes

**Cluster resampling.** `scripts/bootstrap_auroc.py` resamples complexes;
complexes are nested within 9 alleles. I re-ran the load-bearing RQ1 interval
resampling *alleles* instead: sequence minus AF3 is +0.073 [+0.019, +0.128] under
complex resampling and **+0.073 [+0.039, +0.111]** under allele
resampling. The inference is unaffected. The honest finding is that the existing
intervals are fine, and it costs one line to say so — which is worth more than
silently leaving the question open to a viva examiner.

**The fold-set panel cannot carry a locus-level claim.** Fold set v4 is
168 HLA-B, 24 HLA-A
and 24 HLA-C complexes — HLA-C is a *single*
allele, C*08:01, and so is HLA-A. Per-locus sequence-minus-structure margins are
+0.118 (C),
+0.062 (B) and
-0.007 (A) —
but with n=24 and one allele at both A and C, those are single-allele observations,
not locus effects. Note also that v2 was HLA-C-heavy (72 of 144) and v4 reversed
that, so the two panels are not interchangeable for any locus-stratified claim —
a second reason the §1 replication matters.

The 123-allele validation table *does* support a locus statement, and it is the
stronger place to make one: mean per-allele AUROC is 0.968 (HLA-A,
n=36), 0.976 (HLA-B, n=63),
0.941 (HLA-C, n=24), with HLA-C carrying roughly
2.8× the standard deviation of HLA-B. Joining
`data/processed/iedb_coverage.json` shows why it is hard to fix: among alleles with
≥100 positives, median IEDB negatives per positive is 0.166 for HLA-A but
0.008 for HLA-B and 0.007 for HLA-C. The equity gap is a data-availability
gap, and that is a more defensible chapter claim than anything the 9-allele panel
can support.

---

## Suggested priority order

1. **Run AF3 on fold set v2** (144 predictions) and test the four-architecture consensus there. Confirms or kills §1. Highest value per unit compute remaining in the project.
2. **Promote the consensus to a first-class model** in RQ1 and RQ2, not a post-hoc ensemble. It answers a different question from "does structure add to sequence" — namely "do folding models disagree usefully" — and that question currently has an affirmative answer with an interval that excludes zero.
3. **Add calibration** (`scripts/calibration.py`: reliability curve, Brier, ECE, LOAO temperature). Cheap, and a real gap for a project motivated by candidate prioritisation.
4. **Add the power calculation** to the RQ2 discussion, paired with the oracle ceiling. Turns a null into a bound.
5. **Add pAUC and PPV@k** to the RQ1 table and note the AF2/AF3 ordering flip.
6. **Add the cluster-bootstrap sensitivity line**, and move any locus-level claim off the 9-allele panel onto the 123-allele validation table with the IEDB coverage join.

## What I deliberately did not recommend

More tests, more CI, type hints, docstrings, a Dockerfile, or refactoring
`scripts/` into a unified CLI. The engineering is not the bottleneck here — the
determinism guards, the shared-split rule, the retraction ledger and the
`REPRODUCE.md` discipline are already stronger than most published work. Nor did I
recommend anything requiring the controlled-access dataset or independent
NetMHCpan licensing, per the constraints stated in the README.

## Provenance

| artefact | what it contains |
|---|---|
| `review_analysis.py` | reproduces every number in this document from committed files |
| `review_summary.json` | machine-readable results |
| `review_consensus_paired_diffs.csv` | §1, §5 — cluster-bootstrap paired differences |
| `review_consensus_subsets_v4.csv` | §1 — leave-one-architecture-out and pairwise consensus |
| `review_within_allele_agreement_v4.csv` | §1 — within-allele Spearman between models |
| `review_per_allele_consensus_v4.csv` | §1 — per-allele AUROC for sequence, AF3, consensus |
| `review_rq2_power.csv` | §2 — power vs panel size |
| `review_decision_metrics_v4.csv` | §4 — pAUC and PPV@k |
| `review_calibration_reliability_v4.csv` | §3 — reliability bins |
| `review_figure.png` | six panels, one per finding |
