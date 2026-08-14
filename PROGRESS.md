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

## Status: verified present in the repo

| Component | Evidence |
|---|---|
| Sequence CNN (`PresentationNet`) | `src/pmhcpresent/models/nn.py` |
| Within-allele cluster splitting, two-way | `src/pmhcpresent/eval/splits.py`, `scripts/make_split.py` |
| Shared canonical split file | `scripts/make_split.py` → `data/processed/split_val.csv` |
| Structure features: pLDDT, contacts, ipSAE, PAE | `src/pmhcpresent/structure/` |
| `REFOLD_REQUIRED` cost model for RQ3 | `structure/features.py:22` |
| Easy + hard decoy selection | `scripts/select_decoys_clean.py`, `select_decoys_hard.py` |
| Fold sets v2 / v3b / v4 | `fold_sets/` |
| Folding arms: Boltz, ESMFold2, AF2, AF3 | `auroc_*`, `pae_*`, `conf_*`, `geom_*` CSVs per arm |
| RQ2 ensemble scaffolding | `scripts/rq2_stack.py`, `rq2_gate.py`, `rq2_error_overlap.py`, `rq2_ensemble_alt.py` |
| RQ3 variants + landscape comparison | `scripts/build_rq3_variants.py`, `rq3_compare_landscapes.py`, `rq3_sequence_landscape.py`, `rq3_shap.py` |
| External baselines | `scripts/score_netmhcpan.py`, `score_mhcflurry.py`, `score_mixmhcpred.py` |
| Confound controls | `confound_vs_per_allele.py`, `hlac_partial_effect.py`, `foldset_survival_check.py` |
| CI: ruff + pytest | `.github/workflows/ci.yml`, 9 test modules |

## Status: absent

| Component | Note |
|---|---|
| AFND integration | No script, no import, no mention. Literature review commits to it. **Open decision: implement or amend the LR.** |
| Real shape complementarity | `shape.py` is a ΔSASA proxy, explicitly not Sc. Do not rely on it for RQ1/RQ2 conclusions without qualification. |

---

## Results ledger

> Populate one row per claim that will appear in the dissertation. Trace before writing.

| # | Claim | Decoy set | Artefact | Status |
|---|---|---|---|---|
| 1 | | | | |

## Retracted — do not resurrect

- ~~Structure models are strongest precisely where sequence models are weakest
  (equity claim).~~ Did not survive anchor-matched hard decoys; roughly two-thirds
  of the apparent structural signal was anchor recognition. Retracted deliberately.
  If this sentence reappears in a draft, it is a regression, not a finding.

## Unconfirmed — carried from session notes, not yet traced to an artefact

These are plausible but currently unevidenced in this file. Run `/audit-claim` on
each before it enters the write-up.

- HLA-C median AUROC ≈ 0.951 vs ≈ 0.977–0.980 for A/B
- Partial rho ≈ −0.455 for the HLA-C effect after confound control
- Prior-alone AUROC ≈ 0.248 from the negative-sampling artefact
- Boltz ≈ 0.783 pooled AUROC; ESMFold2 ≈ 0.911 against motif-mismatched decoys
- Both arms ≈ 0.700 against anchor-matched hard decoys
- Fold-set binder survival rate — session notes say ~13.9%, the git log says
  something else. **Trace this one first; it is a live discrepancy.**

---

## Open decisions

- [ ] AFND: implement, or amend the literature review to drop the commitment
- [ ] Is AF3 in scope for §3.6, or reported as supplementary?
- [ ] RQ2 stacker: logistic regression vs rank-average — pick on error-overlap evidence
- [ ] Introduction chapter — deprioritised pending results
