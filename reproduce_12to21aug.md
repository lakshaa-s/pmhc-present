# REPRODUCE.md — 12-21 August, plus README amendments

Part 1 appends to `REPRODUCE.md`. Part 2 lists the README changes.

---

## PART 1 — append to REPRODUCE.md

## 12-21 August — regeneration, the P5 mechanism, and a failed prediction

### Regeneration: what it cost and what it bought

The multiset fix was applied to a second dataset and a second model rather than
overwriting the originals, because the fold sets depend on the old split.

**The decision was tested rather than assumed.** Rebuilding the labelled table with
the deduplicated pool and rerunning `make_split.py` at the same seed gives a
validation set sharing only **10.3%** of its pairs with the current one (17,321 of
167,655) — the split is computed over the full table, so changing the negatives
changes the clustering throughout. Under the new split only **18 of 72** fold set v2
binders and **12 of 108** fold set v4 binders remain held out, so 89% of v4's binders
would become training data for a retrained model. Regeneration therefore requires
rebuilding both fold sets and refolding 360 complexes across five architectures.

**Two models are now in play, and they are not interchangeable:**

| Model | Data | Split | Use |
|---|---|---|---|
| `rq1_baseline_split_v2.pt` | `atlas_labelled.csv` | `split_val.csv` | all fold-set results — RQ1, RQ2, RQ3, baselines |
| `rq1_baseline_split_v3.pt` | `atlas_labelled_v2.csv` | `split_val_v2.csv` | all validation-split results — pooled and per-allele AUROC |

Neither is evaluated on its own training data.

**Pooled validation AUROC is 0.9715 on the clean data against 0.9732 before** — the
artefact was worth about 0.002 on the headline figure, far less than the
peptide-level signal suggested, because the model uses allele information the
peptide-identity prior does not.

**The three per-allele predictors on clean data**, against the old raw and partialled
values:

| predictor | clean (v3) | old partial | old raw |
|---|---|---|---|
| anchor information content | **+0.533** (p 2e-10) | +0.648 | +0.603 |
| motif nearest-neighbour distance | **−0.291** (p 1e-03) | −0.502 | −0.363 |
| log₁₀(peptide count) | **−0.118** (p 0.2) | −0.115 | −0.020 |

Both real effects are *weaker* on clean data than either prior version, so the
artefact was inflating them and the partial correction over-corrected. **The clean
figures are the ones to report.** Motif distance at −0.291 rather than −0.363 means
the effect is real but more modest than previously claimed. The sample-size null
holds, which is the load-bearing part for the equity argument.

**The per-locus means are unchanged**, which is the strongest confirmation the HLA-C
deficit is not an artefact of negative sampling:

| locus | v2 model | v3 model | n |
|---|---|---|---|
| HLA-A | 0.970 | 0.968 | 36 |
| HLA-B | 0.975 | 0.976 | 63 |
| HLA-C | 0.940 | **0.941** | 24 |

**The deduplication does not fully remove the confound.** Peptide-identity-only AUROC
moves from 0.248 to **0.3596**; the residual comes from 89.6% cross-allele crossover,
where the same peptide is legitimately positive for one allele and negative for
another. Deduplication cannot touch that.

**Proteome negatives were tested and are far worse.** The peptide-identity prior
reaches **0.8801** there, because proteome windows never appear as anyone's positive,
so peptide identity cleanly separates the classes — and in the *correct* direction,
which is the dangerous kind. Crossover drops to 26.2% as expected, but among peptides
that do cross over the positive fraction is 0.976 for validation positives against
0.228 for negatives. Peptide-pool's self-neutralising crossover is an accidental
virtue of the fallback choice, and should be described as such in Methods rather than
as a compromise.

### The orphan-allele ablation, restored and correctly bounded

`ablation_a2_condB.py` and `ablation_family_condB.py` gained a `--split` argument, so
the holdout comes from `make_split.py` rather than a fresh draw nothing else can
reproduce. Rerun on the clean data:

| condition | AUROC |
|---|---|
| full, family present | 0.9581 ± 0.0004 |
| full, family removed | 0.9593 ± 0.0004 |
| starved, family present | 0.9478 ± 0.0017 |
| **starved, family removed** | **0.8991 ± 0.0221** |

Removing the motif family alone costs nothing; starving alone costs 0.010; both
together cost 0.059, with the standard deviation rising 55-fold. A genuine
interaction rather than an additive effect. This supersedes the July figures
(0.967 → 0.904), which came from an unreproducible split; the structure of the result
is unchanged.

**A\*03:01 and B\*27:05 do not reproduce it, but they are underpowered rather than
negative:**

| target | family | rows | interaction |
|---|---|---|---|
| A\*02:01 | 10 alleles | 91,704 | **−0.049** |
| A\*03:01 | 2 alleles | 31,034 | −0.001 |
| B\*27:05 | 3 alleles | 25,914 | −0.003 |

Removing A\*02's family withdraws nine supporting alleles; A\*03's withdraws one and
B\*27's two. The interventions differ by an order of magnitude, so these are not
negative controls — **no other family in this dataset is large enough to test whether
the mechanism generalises.** The defensible claim: A\*02:01 survives starvation
because nine motif-similar alleles remain in training, and A\*02 is the only family in
the Atlas large enough to provide that buffer.

### The B\*08:01 P5 mechanism

Following the RQ3 finding that ESMFold2 ranks P5 first for HLA-B\*08:01 where the
sequence model ranks P9 and P2, Chris Thorpe supplied the structural explanation and
it turned out to be more specific than first assumed.

**P5 Arg chelation requires a four-residue configuration** — Asp9 and Asp74 provide
the charge, Ser97 hydrogen bonds to Asp9 without neutralising it, and Thr69 sits one
helical turn back from Asp74. Read from the NetMHCpan pseudosequences
(`results/p5_anchor_residues.csv`):

| allele | 9 | 69 | 74 | 97 | P5 IC |
|---|---|---|---|---|---|
| **HLA-B\*08:01** | **D** | **T** | **D** | **S** | **1.93** |
| HLA-B\*37:01 | H | T | Y | R | **1.97** |
| HLA-B\*14:01 | Y | T | D | W | 1.14 |
| HLA-B\*14:02 | Y | T | D | W | 1.08 |
| HLA-C\*16:01 | Y | R | D | W | 0.42 |
| HLA-C\*06:02 | D | R | D | W | 0.31 |
| HLA-C\*07:01 | D | R | D | R | 0.33 |
| HLA-C\*07:02 | D | R | D | R | 0.25 |
| HLA-C\*07:04 | D | R | D | R | 0.61 |

**Exactly one of the 123 training alleles has B\*08:01's configuration.** An earlier
version of this analysis used Asp9 alone as the marker and found five carriers; the
four HLA-C\*07 alleles share Asp9/Asp74 but carry Arg or Arg/Trp at 69 and 97, so the
positive charge sits where B\*08:01 has the hydrogen-bond donor and the P5 arginine
cannot be accommodated.

Asp9 is present in the model's input — it is the second residue of the 34-mer
pseudosequence — so the sequence model is not blind to it. What it lacks is examples:
a pan-allele model would have to learn a four-residue interaction from a single
training allele. **That also gives a complete account of the Gfeller
leave-one-allele-out failure for this allele**: remove B\*08:01 and no allele in the
training set has the configuration.

At least three distinct configurations reach P5 anchoring — B\*08:01 (D/T/D/S),
B\*37:01 (H/T/Y/R), B\*14:01–02 (Y/T/D/W) — which is Chris's superposition figure
quantified from the repertoires.

### AlphaFold 2 confirms the B\*08:01 result

515 AF2 folds via HISTOFold, 3 seeds × 9 positions × 19 substitutions:

  per-position sensitivity: **P5 0.489**, P9 0.459, P2 0.295, P7 0.236, P8 0.212

Top positions [5, 9, 2] with P5 first, matching ESMFold2's [5, 9] and differing from
the sequence model's [9, 2]. So the P5 response is not architecture-specific.

The run required draining PAE to `/tmp` during folding and deleting each complex
directory as it completed, because `/home` was at 100% throughout and 8.5 MB per fold
would not have fit. `results/af2_b08_pae/` holds the extracted matrices; note 511 of
515 use the key `pae` and 4 use `predicted_aligned_error`.

### The B\*37:01 prediction, recorded in advance, fails

**The prediction**, written before the run: B\*37:01 reaches P5 anchoring through
His9/Thr69/Tyr74/Arg97, sharing only Thr69 with B\*08:01. If the structural model
ranks P5 first or second, it is tracking geometry rather than a residue signature.

516 ESMFold2 folds, 3 seeds. **It does not.**

  P9 1.361, P2 1.195, P4 0.695, P7 0.685, P8 0.657, **P5 0.645**, P1 0.514,
  P3 0.444, P6 0.256

P5 ranks **sixth of nine**, and P4/P7/P8/P5 span only 0.05 — the model does not
distinguish P5 from any other non-terminal position. For B\*08:01, AF2 put P5 first
with a 0.194 margin over P2.

**This is not noise.** Seed-to-seed agreement for B\*37:01 is the best measured
anywhere in RQ3 — mean +0.557, minimum +0.511, no negative pairs — so the measurement
is more reliable here than for any of the four alleles in the earlier run.

Both alleles show near-identical P5 arginine preference in their eluted repertoires
(1.93 and 1.97 bits), so the difference is not in the biology of what is presented.
It is in the chemistry: B\*08:01 chelates through a charged Asp9/Asp74 pair, B\*37:01
through His9/Tyr74/Arg97.

**This bounds the RQ3 claim.** The structural models detect **B\*08:01's** P5
anchoring, not P5 anchoring in general. "Structure picks up what sequence misses" was
too broad — they picked up one mechanism, in one allele. A charged salt bridge may
perturb interface PAE more than an aromatic or hydrogen-bonding arrangement does,
but that is a conjecture and is flagged as such.

### A fourth selection artefact, caught and not reported

Across the nine alleles in the P5 table, Thr69 separates them perfectly: all four
with Thr69 anchor at P5 (1.08–1.97 bits), all five with Arg69 do not (0.25–0.61),
with no overlap.

It does not survive. Across all 123 alleles, Thr69 gives mean P5 IC **0.45** against
0.34 for others, **Mann-Whitney p 0.34**, with 44 alleles carrying it — including
B\*40:01, B\*40:02, B\*15:01, B\*44:02, B\*44:03 and B\*47:01, none of which anchors at
P5. The nine were selected for being P5-Arg binders or Asp9 carriers, so the
separation is selection.

Not reported, and deliberately omitted from the correspondence with Chris. This is
the fourth artefact of this kind caught in the project, after the withdrawn geometry
mechanism, the fold-quality features that were proxies for the classifier, and the
peptide-count correlation.

---

## PART 2 — README amendments

**The RQ3 status row** currently reads "in progress". Replace with:

> | **RQ3** | Does saturation mutagenesis show the two learned the same binding biology? | answered — partly |

**The results-in-brief RQ3 row** needs the bounding. Replace the existing text with:

> **Partly.** Both recover anchor positions; the sequence model's top-2 fall inside
> the IC-derived anchor set for 6/7 alleles. For HLA-B\*08:01 both structural models
> rank P5 first where the sequence model does not — but running the same analysis on
> HLA-B\*37:01, which has the highest P5 information content of the group, puts P5
> sixth of nine. The structural models detect one P5 mechanism, not P5 anchoring in
> general.

**Add to Known Issues**, after the negative-sampling entry:

> **The per-allele results use a different model from the fold-set results.**
> `rq1_baseline_split_v3.pt` is trained on the deduplicated data and used for the
> pooled and per-allele AUROC; `rq1_baseline_split_v2.pt` is used for everything
> computed on the fold sets, which were built from the older split. Only 10.3% of the
> validation split survives regeneration, so the two cannot be merged without
> rebuilding the benchmarks and refolding 360 complexes. Neither model is evaluated
> on its own training data.

**Remove the stale ablation caveat** — "These predate `make_split.py` and should be
rerun before their numbers are quoted" — since they now take `--split` and have been
rerun.

**Correct the negative-mode description.** The data section says `--neg-mode
proteome` is "the intended decoy set" and peptide-pool "a no-external-data fallback".
Testing shows proteome is substantially worse: the peptide-identity prior reaches
0.8801 there against 0.3596 for deduplicated peptide-pool. Describe peptide-pool as
the appropriate choice and say why.
