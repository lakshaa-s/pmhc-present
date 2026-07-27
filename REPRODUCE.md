# REPRODUCE.md — how to regenerate every result

A map from each result to the exact script + command that produces it, plus where the
data and outputs live. Goal: nothing is a one-off; every number/figure is re-testable.

> **Verify before trusting:** some paths/filenames below are reconstructed from working
> notes — check the `# TODO/verify` tags and correct anything that doesn't match your tree.

---

## Environment

- **Compute:** Beta (Linux, RTX 4090, CUDA). Conda env `pmhcpresent`.
  `conda activate pmhcpresent` before running anything.
- **Repo:** `~/pmhc-present` (GitHub: `lakshaa-s/pmhc-present`).
- **Structure folding:** runs on the **Mac** via the Boltz API (folder
  `~/Downloads/boltz_prediction`), not on Beta. Outputs land in `boltz-experiments/`.

---

## Data pipeline

| Result | Script | Command | Output |
|---|---|---|---|
| Labelled dataset (838k rows, 123 alleles) | `scripts/prepare_atlas.py` | see command below | `data/processed/atlas_labelled.csv` |

```
python scripts/prepare_atlas.py \
  --input data/raw/all_peptides.txt \
  --output data/processed/atlas_labelled.csv \
  --neg-mode peptide-pool \
  --ratio 1.0 --min-len 8 --max-len 11 --seed 42
```

Flags: `--input` = raw Atlas `all_peptides.txt`; `--output` = labelled CSV;
`--neg-mode {proteome,peptide-pool}` (current dataset used **peptide-pool**; swap to
`proteome --proteome <human_proteome.fasta>` later — nothing downstream changes);
`--ratio` negatives per positive (default 1.0); `--min-len`/`--max-len` peptide length
window (8–11); `--seed` for reproducible negative sampling.
<!-- verify: exact --input path of the raw atlas file, and the seed you actually used -->

The script filters to classical HLA A/B/C, normalises alleles (`A0201` → `HLA-A*02:01`,
matching the pseudoseq loader's canonical key), labels atlas rows as positives, and
generates length-matched negatives per allele. Output columns: `peptide, allele, label, length`.

- Source: MHC Motif Atlas (`all_peptides.txt`), filtered to classical HLA A/B/C, 8–11mers;
  1:1 negatives (peptide-pool mode; proteome-sampled is the planned upgrade).
- Pseudosequences: `data/pseudoseq/hla_{a,b,c}.json` (Chris's pocket-pseudoseq JSONs).

---

## Sequence model — RQ1 baseline

| Result | How |
|---|---|
| **Baseline AUROC ~0.974, equity gap ~0.007** | `pmhcpresent train --data data/processed/atlas_labelled.csv --pseudoseq data/pseudoseq/hla_a.json data/pseudoseq/hla_b.json data/pseudoseq/hla_c.json --epochs 50 --save models/rq1_baseline_hamming.pt` |

- Split: near-duplicate-aware (`hamming_cluster` in `src/pmhcpresent/eval/splits.py`).
- Robustness: AUROC held (0.973 → 0.974) when tightening exact-dedup → Hamming split.
- Per-bin AUROC (Hamming): rare 0.965 / low 0.974 / medium 0.971 / high 0.978 / very_high 0.973.

---

## Per-allele distribution (the HLA-C equity result)

| Result | Script | Command |
|---|---|---|
| **Per-allele AUROC, all 123 alleles; HLA-C median 0.951 vs A/B ~0.98** | `scripts/per_allele_auroc.py` | `python scripts/per_allele_auroc.py --data data/processed/atlas_labelled.csv --pseudoseq data/pseudoseq/hla_a.json data/pseudoseq/hla_b.json data/pseudoseq/hla_c.json --model models/rq1_baseline_hamming.pt --out results/per_allele_auroc.csv` |
| **Locus distribution plot** | `scripts/plot_per_allele.py` | `python scripts/plot_per_allele.py results/per_allele_auroc.csv` → `per_allele_dist.png` |

- Key finding: HLA-C underperforms regardless of data volume (e.g. C\*12:03 has 3,026
  peptides and still low) → motif-isolation, not data quantity, drives the gap.
- Worst 5: C\*15:05 (0.889), B\*14:01 (0.902), C\*16:02 (0.902), C\*12:04 (0.924), C\*12:03 (0.927).

---

## Ablations — the orphan-allele mechanism

**Condition A — A\*02:01 dose-response** (starve one allele, keep the rest):
```
python scripts/ablation_a2.py \
  --data data/processed/atlas_labelled.csv \
  --pseudoseq data/pseudoseq/hla_a.json data/pseudoseq/hla_b.json data/pseudoseq/hla_c.json \
  --out results/ablation_a2.csv
```
Result: flat curve — AUROC ~0.96 even at 115 examples → cross-allele transfer.

**Condition B — 2×2 (starve × remove family).** A\*02 uses the fixed-prefix script;
A\*03 and B\*27 use the generalised regex script:
```
# A*02 (collapses):
python scripts/ablation_a2_condB.py \
  --data data/processed/atlas_labelled.csv \
  --pseudoseq data/pseudoseq/hla_a.json data/pseudoseq/hla_b.json data/pseudoseq/hla_c.json \
  --out results/ablation_a2_condB.csv

# A*03/A*11 supertype (does NOT collapse):
python scripts/ablation_family_condB.py \
  --data data/processed/atlas_labelled.csv \
  --pseudoseq data/pseudoseq/hla_a.json data/pseudoseq/hla_b.json data/pseudoseq/hla_c.json \
  --target "HLA-A*03:01" --family-regex '^HLA-A\*(03|11):' \
  --out results/ablation_a3_condB.csv

# B*27 family (does NOT collapse):
python scripts/ablation_family_condB.py \
  --data data/processed/atlas_labelled.csv \
  --pseudoseq data/pseudoseq/hla_a.json data/pseudoseq/hla_b.json data/pseudoseq/hla_c.json \
  --target "HLA-B*27:05" --family-regex '^HLA-B\*27:' \
  --out results/ablation_b27_condB.csv
```
Result across 4 alleles: only **A\*02:01 collapses** (0.96 → 0.90) when starved AND family
removed. A\*03, B\*27 don't. Refined claim: collapse needs subtle motif + genuine isolation.

Alleles removed per experiment: A\*02 = 9 family members; A\*03/A\*11 = 3; B\*27 = 2.

---

## Structure phase (Boltz) — RUNS ON THE MAC

Boltz folds via API; outputs in `~/Downloads/boltz_prediction/boltz-experiments/`.
API key in `.env` (gitignored — never commit).

**1. Select canonical binders** (on Beta — needs atlas + pseudoseqs):
```
python scripts/select_fold_set_canonical.py \
  --data data/processed/atlas_labelled.csv \
  --pseudoseq data/pseudoseq/hla_a.json data/pseudoseq/hla_b.json data/pseudoseq/hla_c.json \
  --peptide-length 9 --n-alleles 5 --k-peptides 6 \
  --out fold_sets/fold_set_9mer_canonical_k6.csv
```
Alleles are chosen by max-min pocket-pseudosequence distance (seeded with covered anchors
and orphan HLA-C). Peptides are scored against a PWM built from that allele's own atlas
positives, and the top decile is then diversified — so binders are motif-typical rather
than maximally diverse. **This supersedes `select_fold_set.py`** (max-diversity), which
selects motif-atypical peptides and is unsuitable for a discrimination test.

**2. Select decoys** (motif-mismatched, anchor-rejected):
```
python scripts/select_decoys_clean.py \
  --data data/processed/atlas_labelled.csv \
  --pseudoseq data/pseudoseq/hla_a.json data/pseudoseq/hla_b.json data/pseudoseq/hla_c.json \
  --peptide-length 9 --k-decoys 6 \
  --out fold_sets/decoy_set_9mer_clean_k6.csv
```
Candidates come from the 3 most motif-distant donor alleles, then are rejected if they
score above the 25th percentile of the target's real binders **or** carry any of the
target's top-4 preferred residues at P2 / C-terminus. **This supersedes
`select_decoys.py`**, which used allele-level distance only and leaked anchor-carrying
peptides into the decoy set.

Combine into one fold set:
```
cat fold_sets/fold_set_9mer_canonical_k6.csv fold_sets/decoy_set_9mer_clean_k6.csv \
  > fold_sets/fold_set_60.csv
```

**2b. Select ANCHOR-MATCHED (hard) decoys** — the adversarial control:
```
python scripts/select_decoys_hard.py \
  --data data/processed/atlas_labelled.csv \
  --peptide-length 9 --k-decoys 6 \
  --out fold_sets/decoy_set_hard.csv
```
Inverts the filter: candidates must *carry* the target's top-4 preferred residues at both
P2 and the C-terminus, while still scoring below the 50th percentile of the target's real
binders. Real eluted ligands of other alleles are used rather than synthetic
anchor-preserving scrambles, because a scrambled sequence is out of distribution for a
protein-language-model backbone — high predicted error would then reflect implausibility
rather than non-binding.

These fold into a separate directory, with the binders copied in for comparison:
```
python scripts/fold_esmfold2.py --csv fold_sets/decoy_set_hard.csv \
  --sequences data/sequences --out esmfold2-hard
cp -r esmfold2-experiments/NA__* esmfold2-hard/     # 30 binders + 30 hard decoys
python scripts/analyse_pae.py esmfold2-hard --out pae_esmfold2_hard.csv
python scripts/auroc_structure.py --pae pae_esmfold2_hard.csv --out auroc_esmfold2_hard.csv
```
`analyse_pae.py` treats folder tags `decoy` and `hard` as negatives, anything else as a
binder.

**3. Fold** (on Mac, in the Boltz folder): copy the CSV to `complexes/hla_class_i.csv`, then
`uv run boltz_pmhc_class_i.py`. Each pMHC ≈ $0.05.

**4. Extract features** (Mac): `python3 scripts/extract_boltz_features.py boltz-experiments --out boltz_features.csv`
→ per-fold iptm, complex_iplddt, complex_plddt, pae_interface, pae_mean.

**5. Per-residue anchor PAE analysis** (Mac): `python3 scripts/analyse_pae.py boltz-experiments --out pae_analysis.csv`
→ binder-vs-decoy anchor PAE per allele.

**6. Interface geometry** (Mac): `python3 scripts/extract_geometry.py boltz-experiments --out geometry_features.csv`
→ peptide-MHC contact counts, anchor contacts, anchor-pocket distances (needs biotite:
`pip install biotite --break-system-packages`).

**7. AUROC vs the sequence baseline** (Mac): `python3 scripts/auroc_structure.py --pae pae_analysis_k6.csv --geometry geometry_k6.csv --out structure_auroc.csv`
→ treats each structural feature as a binding score and computes AUROC over binder/decoy
labels, so it is directly comparable to the sequence model's AUROC.

**Fold sets (all 9mers, for HISTOFold compatibility):** 5 alleles spanning covered→orphan
(B\*27:05, A\*02:01, B\*07:02, C\*15:05, C\*16:02), 6 canonical binders + 6 motif-mismatched
decoys each = 60 complexes. Binders are top-decile by the allele's own PWM; decoys are
rejected if they carry the target's preferred anchor residues (see script docstrings).

**Result — anchor-localised PAE discriminates; confidence and geometry do not.**

| Feature | Pooled AUROC (n=60) |
|---|---|
| `pae_anchors` (P2 + C-term) | **0.783** |
| `pae_anchorC` | 0.759 |
| `pae_anchor2` | 0.737 |
| `pae_pep_mhc` (whole interface) | 0.694 |
| contact/geometry features | 0.21 – 0.41 |
| `iptm` | ~flat 0.98–0.99 for binders and decoys alike |

Signal *increases* as the metric localises to the anchor positions (0.694 whole-interface
→ 0.783 anchors), which matches anchor-dominated binding biology. Geometry features sit
*below* 0.5 — decoys tend to make slightly more contacts — so contacts are not a binding
proxy.

Per-allele `pae_anchors` AUROC: C\*15:05 **0.972**, B\*07:02 0.917, C\*16:02 0.750,
B\*27:05 0.722, A\*02:01 **0.639**. Note the inverse relationship with the sequence model:
A\*02:01 is the sequence model's best allele and structure's worst; C\*15:05 is the
sequence model's worst (0.889) and structure's best. **Structure appears strongest exactly
where sequence is weakest** — the complementarity that RQ2 asks about, consistent with
King et al. 2025 (arXiv:2512.06592), who found structure and sequence embeddings combine
most usefully where the sequence model is weak.

**[CORRECTION — supersedes an earlier finding.]** An initial pilot concluded that *no*
Boltz signal discriminated. That was an artifact of the fold set, not of Boltz: the
max-diversity peptide selection had picked motif-atypical "binders" (e.g. `LVAKVRALD`
assigned to B\*07:02 with neither anchor), and allele-level distance alone let
anchor-carrying peptides into the decoy set (e.g. `QRSRFIVVV`, P2-Arg, as a B\*27:05
"decoy"). Rebuilt with canonical binders and anchor-rejected decoys, the anchor-PAE signal
appears clearly. The confidence (`iptm`) and geometry negatives survive the correction.

**Caveats.** 6 binders + 6 decoys per allele — the pooled figure and the consistency of
direction across five alleles are more reliable than any single per-allele AUROC. 0.783 is
well below the sequence model's ~0.97, so structure alone does **not** outperform sequence
(RQ1 remains negative); the interest is in complementarity (RQ2). The two AUROCs are also
not strictly comparable: the sequence figure is over a large held-out set with pooled
negatives, this one over 60 designed complexes with motif-mismatched decoys.

---

## Structure phase (ESMFold2) — RUNS ON BETA

ESMFold2 (Biohub, built on the ESMC 6B backbone) runs **locally on Beta's RTX 4090** —
no API, no rate limit, no cost. Model load is ~13.7 GB of the 24 GB card, leaving room for
a ~383-residue complex.

Environment (separate from `pmhcpresent`, which is on Python 3.13 — the `esm` package
requires >=3.12,<3.13):
```
conda create -n esmfold2 python=3.12 -y
conda activate esmfold2
pip install esm@git+https://github.com/Biohub/esm.git@main
```
Weights download from HuggingFace on first use (~25 GB across 6 shards, cached thereafter).
Optional speedups, not required: `pip install xformers flash-attn`.

Allele sequences must be on Beta at `data/sequences/` (`hla_{a,b,c}.json`, `human_b2m.json`
— the same files Chris ships with the Boltz code).

**Fold the same 60 complexes:**
```
conda activate esmfold2
python scripts/fold_esmfold2.py \
  --csv fold_sets/fold_set_60.csv \
  --sequences data/sequences \
  --out esmfold2-experiments
```
Chains match the Boltz setup (A = MHC, B = β2m, C = peptide) and the output layout is
identical (`outputs/files/prediction/{metrics.json, sample_0_pae.npz}`), so the analysis
scripts run unchanged. `--num-diffusion-samples N` folds each complex N times and records
per-sample metrics under `all_sample_results` (ESMFold2 applies fresh LM dropout per fold,
so repeats are genuinely diverse) — this is the cheap route to the confidence *distribution*.

No `.cif` is written, so `extract_geometry.py` does not apply; pass only `--pae` to the
AUROC script.

```
python scripts/analyse_pae.py esmfold2-experiments --out pae_esmfold2.csv
python scripts/auroc_structure.py --pae pae_esmfold2.csv --out auroc_esmfold2.csv
```

Beyond PAE, the result object also exposes `pair_chains_iptm` (per-chain-pair interface
confidence — the MHC↔peptide element is more targeted than global iptm) and
`output_embedding_sequence` / `output_embedding_pair_pooled`, which `fold_esmfold2.py`
saves to `embeddings.npz`. Those embeddings are the raw material for the RQ2
combined-representation work.

---

## Two-model comparison — the headline structural result

Identical 60 complexes (5 alleles × 6 canonical binders + 6 decoys), identical analysis.
`pae_anchors` = mean PAE of peptide P2 and C-terminus vs the MHC.

**Decoy difficulty matters, and should always be stated alongside the number.** Two
classes were used:
- **motif-mismatched** (`select_decoys_clean.py`) — anchor-carrying candidates *rejected*,
  so rejectable on anchors alone;
- **anchor-matched** (`select_decoys_hard.py`) — candidates *required* to carry the
  target's anchors at P2 and C-terminus while scoring low overall, removing the anchor
  shortcut.

| Pooled AUROC (n=60) | Boltz (mismatched) | ESMFold2 (mismatched) | ESMFold2 (anchor-matched) |
|---|---|---|---|
| `pae_anchors` | 0.783 | **0.911** | **0.700** |
| `pae_anchor2` | 0.737 | 0.921 | 0.759 |
| `pae_pep_mhc` | 0.694 | 0.863 | 0.672 |

Per-allele `pae_anchors`:

| Allele | Boltz (mismatched) | ESMFold2 (mismatched) | ESMFold2 (anchor-matched) |
|---|---|---|---|
| HLA-A\*02:01 | 0.639 | 0.944 | 0.639 |
| HLA-B\*07:02 | 0.917 | 1.000 | 0.972 |
| HLA-B\*27:05 | 0.722 | 1.000 | 0.889 |
| HLA-C\*15:05 | 0.972 | 1.000 | 0.639 |
| HLA-C\*16:02 | 0.750 | 0.972 | 0.667 |

**Anchor-localised PAE carries binding signal in two independent architectures** — so
this is a property of structure prediction on pMHC, not an artifact of one model. Signal
increases as the metric localises to the anchors (whole-interface 0.863 → anchors 0.911
under ESMFold2), while global confidence (`iptm`) and contact counts carry nothing.

**But roughly two-thirds of that signal is anchor recognition.** Against anchor-matched
decoys, ESMFold2 falls from 0.911 to 0.700. It stays above chance and all five alleles
keep the expected direction, so there is residual sensitivity to groove fit beyond the
anchors — but the headline figure is highly sensitive to how negatives are built.

Note also that `pae_anchor2` (P2 only, 0.759) *beats* the P2+C-term average under the
harder test, and `pae_anchorC` is the weakest component (0.630) — most of the residual
signal sits at P2.

**The equity claim does not survive the harder test.** Against mismatched decoys,
C\*15:05 — the sequence model's worst allele (per-allele AUROC 0.889) — scored 1.000,
suggesting structure helps most where sequence fails. Against anchor-matched decoys it
drops to 0.639, while B\*07:02 (0.972) and B\*27:05 (0.889) hold up. So the apparent
inverse relationship between sequence and structure performance was largely an artifact
of easy negatives. Whether structure genuinely complements sequence in the orphan-allele
regime (RQ2) is still open.

**Caveats.**
- 6 binders + 6 decoys per allele. Per-allele AUROCs move by ~0.03 per swapped pair;
  treat pooled figures and direction-consistency as the reliable signal.
- **Not directly comparable to the sequence model's ~0.97.** That is over a large
  held-out set with pooled negatives; these are 60 designed complexes. A like-for-like
  comparison needs both models scored on the *same* positives and negatives — still
  outstanding.
- The anchor-matched decoys are conservative: "not observed on this allele" is weaker
  than "does not bind this allele", so a few may be genuine binders, depressing the
  measured AUROC.
- Some mismatched-decoy PAE values are very large (14.5, 15.9) — worth checking whether
  those folds are pathological rather than merely low-confidence.

- **Code:** all in `~/pmhc-present` (committed). Structure analysis scripts in `scripts/`.
- **Data:** `data/processed/atlas_labelled.csv`, `data/pseudoseq/*.json`,
  `data/sequences/*.json` (Beta; gitignored).
- **Models:** `models/*.pt` (Beta; gitignored).
- **Sequence results:** `results/*.csv` (Beta; gitignored — BACK UP separately).
- **Boltz outputs:** `boltz-experiments/` + `boltz-clean/` (Mac only — BACK UP to Drive;
  they cost credits). `boltz-clean/` holds just the current 60-complex set.
- **ESMFold2 outputs:** `esmfold2-experiments/` (Beta; free to regenerate, so lower
  backup priority — but the embeddings are worth keeping).
- **Fold sets:** `fold_sets/` (Beta; gitignored).

---

## Known caveats / TODO

- Fold sets are 6 binders + 6 decoys per allele across 5 alleles — enough to establish
  direction, not enough for tight per-allele estimates. Scaling to more peptides per
  allele (and more alleles) is the obvious next step, and is now cheap: ESMFold2 runs
  locally on Beta with no per-fold cost.
- Boltz folds used `num_samples: 1`. ESMFold2 supports `--num-diffusion-samples N` for the
  confidence *distribution* the meeting asked for — not yet run.
- Binder/decoy selection dominates the result. The first pilot's null came from
  motif-atypical binders and anchor-carrying decoys; the corrected easy-decoy set then
  gave 0.911; anchor-matched decoys bring it to 0.700. **Always report which decoy class
  a structural AUROC refers to.** Any change to the selection scripts warrants re-running
  both comparisons.
- The obvious next probe: if most of the signal is at P2, does discrimination survive
  decoys matched at P2 *and* with similar overall composition? That would isolate whatever
  groove-fit sensitivity remains.
- For a fair RQ1 comparison, structure & sequence should use the **same** positives/negatives.
  They currently do not — this is the most important outstanding methodological gap.
- Remaining structural avenues:
    - **AF2 / HISTOFold** (Chris's tuned MSAs, github.com/drchristhorpe/HISTOFold) — needs
      Docker + NVIDIA Container Toolkit on Beta (sudo available; neither Docker nor Podman
      currently installed). Currently 9mer-only; Chris is extending it to 8-11mers.
      He is free to help from Wednesday afternoon.
    - **ESMFold2-Fast** — Chris's benchmarking found the fast variant better on pMHC,
      especially peptide confidence for unseen peptides. Only the standard model has been
      run so far; `--model biohub/ESMFold2-Fast` (repo name unverified).
    - **Structure embeddings** (King et al. lead) — `fold_esmfold2.py` already saves
      ESMFold2's sequence and pooled-pair embeddings per fold. Do learned representations
      carry *complementary* signal to sequence in the weak/orphan regime (RQ2)?
- `data/` and `models/` are gitignored, so a fresh clone can't reproduce without them —
  document where to obtain/regenerate the Atlas download, pseudoseq JSONs, and
  `data/sequences/`.