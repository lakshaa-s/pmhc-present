#!/usr/bin/env bash
# Fold peptide–MHC complexes on Beta (RTX 4090, CUDA 13). Placeholder — wire to your
# AlphaFold/Boltz/Protenix install. Emits model PDB + PAE JSON consumed by
# p11.structure.features. For RQ3 saturation mutagenesis, only refold-required
# features (pLDDT/PAE/ipSAE) need a fresh fold per mutant.
set -euo pipefail
echo "TODO: invoke folding engine; output model.pdb + pae.json to data/fold_outputs/"
