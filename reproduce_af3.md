
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

**And none of these models is fine-tuned.** The claim RQ1 supports is that
*off-the-shelf structural confidence* does not outperform sequence. Motmaen et al.
show that fine-tuning closes most of the gap, reaching 0.97 on Class I. A fine-tuned
comparison has been requested from a colleague who has one; if it materialises it
would bound the result properly, and if not this limitation must be stated
explicitly wherever RQ1 is reported. Note also that AF3's weights terms of use
prohibit using its output to train models intended for commercial application, so
AF3 stays evaluation-only regardless.
