# Assessment of the external review

`REVIEW.md` was produced by an automated review over the committed results on
26 August. Every one of its findings was checked against the data before being
accepted, and two of its numbers did not reproduce.

**Read this file before quoting anything from `REVIEW.md`.** The review is retained
because its framing is useful and two of its findings were genuine gaps, not because
all of its numbers are right.

The same rule applies here as in `PROGRESS.md`: a claim with no artefact is not a
result, and corrections stay visible rather than being tidied away.

---

## 1. Structural consensus — genuine gap, reinterpreted

**The review's claim.** The four architectures were each tested against sequence but
never against each other. Averaging their per-allele z-scored anchor PAE gives 0.905
against 0.858 for the best single architecture, paired difference +0.046
[+0.003, +0.091]. Sequence minus consensus then spans zero where sequence minus any
individual architecture does not, so an RQ1 conclusion moves.

**Verdict: the gap was real, the numbers reproduce, the interpretation does not.**

The gap is genuine — this project had not compared the architectures with each
other, and it should have. All figures reproduce exactly
(`scripts/consensus_diagnosis.py`, `results/consensus_diagnosis_v4.csv`).

But the review reads the gain as the architectures being "four substantially
independent readouts". Three tests say otherwise:

| test | result |
|---|---|
| PC1 of the four z-scored scores | **0.9033** against the four-way mean's 0.9047 — a difference of 0.0014, with PC1 explaining 61.3% of variance |
| gain vs headroom, per allele | rho +0.203, p 0.60 — not the ceiling effect either |
| subset selection | AF3+AF2+ESMFold2 reaches 0.9233, but is the best of eleven subsets |

**A consensus equal to its own first principal component is variance reduction, not
complementarity.** Averaging cancels independent error across four noisy measurements
of one underlying quantity. The diagnostic was validated on synthetic data before
being applied: it returns +0.006 when four scores are noisy copies of one signal and
+0.036 when they carry genuinely different facets.

So the finding is worth reporting as a measurement and does not move RQ1's
conclusion. "Sequence minus consensus spans zero" holds, but because variance
reduction lifts a weak readout toward a strong one — not because structure caught up.

Two things the review did not note, both of which the project applies elsewhere:

- The z-scoring is **transductive**, using the held-out set's own mean and standard
  deviation. Both sides carry it so the paired difference is fair, but the absolute
  AUROCs are upper bounds — the same caveat attached to every z-scored figure here.
- At the operating point a screening pipeline would use, the consensus is clearly
  behind sequence: partial AUROC at FPR ≤ 0.10 is **0.688 against 0.743**. The AUROC
  closeness does not survive a decision-relevant metric.

One minor factual error: the review states "every three-architecture subset except
drop-AF2 and drop-AF3 beats the best single model". In fact all four do (0.8988,
0.8586, 0.8661, 0.9233 against AF3's 0.8575), so the review understates its own case.

---

## 2. Oracle and error-overlap numbers — the review's two outputs disagree

**The discrepancy.** `REVIEW.md` states sequence gets 5 of 216 complexes wrong with 4
rescued, and an oracle ceiling of 0.984 against sequence's 0.946. The accompanying
chat summary states 12 of 216, 5 rescued, and 0.972 against 0.930.

**Verdict: `REVIEW.md` is correct; the summary is wrong.** Recomputed directly from
`results/rq2_error_overlap_margins.csv`:

```
sequence margin < 0.5:            5 of 216
of those, rescued by structure:   4
sequence mean margin              0.946
oracle across all five models     0.984
```

Use the document's figures. The summary's should not be quoted.

---

## 3. Calibration — genuine gap, figures do not reproduce

**The review's claim.** Calibration is never reported anywhere and every metric used
is rank-based. At prevalence 0.50, mean predicted P is 0.71 with ECE 0.21; one
temperature parameter fitted leave-one-allele-out improves Brier 0.173 → 0.155 and
ECE 0.211 → 0.177 with AUROC unchanged.

**Verdict: the gap is real and important. The numbers do not reproduce, and the
conclusion changes.**

Independently implemented as `scripts/calibration_metrics.py`:

| | review | reproduced |
|---|---|---|
| mean predicted P | 0.71 | **0.666** |
| Brier | 0.173 → 0.155 | **0.2274 → 0.2265** |
| ECE | 0.211 → 0.177 | **0.2507 → 0.1583** |
| fitted temperature | ≈2.02 | **1.12–1.28** |
| AUROC | "essentially unchanged" | **0.930 → 0.901** |

Two differences matter. ECE improves far more than reported while Brier barely moves,
so temperature scaling fixes the confidence distribution without reducing error. And
**the AUROC cost is 0.029, not negligible** — leave-one-allele-out gives each allele
its own temperature, and differing temperatures reorder complexes across alleles.
Calibration here is a trade, not a free improvement, and must be reported as one.

The review's claim that "every reliability bin below 0.7 contains zero binders" also
does not hold: one such bin exists, topping out at predicted 0.53.

A caveat neither version states: the fold sets are balanced 50/50 by construction
while real presentation is nearer 1 in 1,000, so a temperature fitted here does not
transfer to deployment without recalibration against realistic prevalence.

---

## 4. Decision-relevant metrics — accepted in full

**The review's claim.** AF3 wins on AUROC (0.858 vs 0.842) but AF2 wins at low false
positive rate (pAUC 0.634 vs 0.619, PPV@20 0.95 vs 0.90), so "AF3 is the best
structural model" is metric-dependent.

**Verdict: reproduces, and is the most directly useful finding in the review.**

| model | AUROC | pAUC ≤0.10 | PPV@20 |
|---|---|---|---|
| sequence | 0.930 | 0.743 | 1.00 |
| consensus | 0.905 | 0.688 | 0.95 |
| af3 | 0.858 | 0.619 | 0.90 |
| af2 | 0.842 | **0.633** | **0.95** |
| esmfold2 | 0.805 | 0.588 | 0.80 |
| boltz | 0.745 | 0.543 | 0.65 |

Both alternative metrics reorder the models. Wherever the RQ1 table is read as a
ranking, the operating point has to be named.

---

## 5. Fold set v4 locus composition — accepted, no action needed

**The review's claim.** Fold set v4 has one HLA-A allele and one HLA-C allele, so any
locus claim from it is a single-allele observation and should move to the 123-allele
validation table.

**Verdict: correct, and already the case.** An audit of `REPRODUCE.md`, `README.md`
and `PROGRESS.md` found every locus claim already attributed to the validation split
or the per-allele tables. Nothing needed moving. Worth keeping as a standing check.

---

## 6. RQ2 power — adopted, with a different effect size

**The review's claim.** At the observed effect (+0.013), the paired bootstrap has
power 0.37 at n=216, crossing 80% near n≈500.

**Verdict: the approach is right and has been adopted; the specific numbers depend on
which configuration is powered for.**

`scripts/rq2_power.py` powers for **+0.026**, from blending sequence with the
four-architecture consensus — which matches the gated ensemble's ungated row
(+0.028 [−0.005, +0.065]), a configuration actually run under leave-one-allele-out.
That gives **power 0.63 at n=216, reaching 80% at n≈432**.

The review's 0.37 powers for a weaker configuration at +0.013. Both are defensible.
Neither should be quoted without naming the effect it refers to.

Three limitations apply to both and all push the same way, making the recommended
panel size a lower bound: the observed effect is treated as true, resampling
complexes assumes a larger panel would be more of the same rather than more alleles,
and the nesting within nine alleles is ignored.

---

## 7. Cluster bootstrap sensitivity — accepted

The review re-ran the load-bearing RQ1 interval resampling alleles rather than
complexes: +0.073 [+0.039, +0.111] against the reported +0.073 [+0.019, +0.128].
Same inference, tighter interval. Worth one sentence in Methods stating that the
choice of resampling unit does not change the conclusion.

---

## What the review did well, and what it did not

**Well.** It reproduced the headline RQ1 numbers from committed files before making
any claim. It found a genuine analytical gap that had been missed across three months
— the architectures were never compared with each other. It flagged its own
non-replication on fold set v2 rather than reporting the v4 result alone. And every
claim cited an artefact, which is what made checking it possible at all.

**Less well.** Two of its numeric findings did not reproduce, one of them changing a
conclusion. It read the consensus result as complementarity without testing the
variance-reduction alternative, which is the obvious competing explanation. It did
not apply the project's own transductive-z-scoring caveat to a z-scored result. And
its top recommendation — AlphaFold 3 on fold set v2 — is blocked by infrastructure it
had no way to know about.

**The general lesson**, which is worth stating in the discussion: an external review
of a well-documented project is useful in proportion to how checkable its claims are.
Every finding here could be verified because it named its artefact, and two of them
turned out to be wrong. A review whose claims could not have been checked would have
been worse than none.