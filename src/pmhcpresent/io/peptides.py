"""Peptide and HLA-pseudosequence encoding.

Amino acids are mapped to integer indices for learnable embeddings. Index 0 is the
pad token, so real residues start at 1. ``X`` (unknown) gets its own index. This
module is pure-Python/numpy so it runs anywhere; the tensors it feeds are built in
``pmhcpresent.models.nn``.
"""
from __future__ import annotations

import numpy as np

# 20 standard amino acids + X (unknown). PAD is index 0.
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
PAD_TOKEN = "-"
UNK_TOKEN = "X"

_VOCAB = [PAD_TOKEN] + list(AMINO_ACIDS) + [UNK_TOKEN]
AA_TO_IDX = {aa: i for i, aa in enumerate(_VOCAB)}
IDX_TO_AA = {i: aa for aa, i in AA_TO_IDX.items()}
PAD_IDX = AA_TO_IDX[PAD_TOKEN]
UNK_IDX = AA_TO_IDX[UNK_TOKEN]
VOCAB_SIZE = len(_VOCAB)


def encode_sequence(seq: str, max_len: int) -> np.ndarray:
    """Encode an amino-acid string to a length-``max_len`` int array (right-padded).

    Unknown characters map to ``X``. Sequences longer than ``max_len`` are truncated
    with a warning-free hard cut (callers should validate lengths upstream).
    """
    seq = seq.strip().upper()[:max_len]
    idx = [AA_TO_IDX.get(c, UNK_IDX) for c in seq]
    if len(idx) < max_len:
        idx.extend([PAD_IDX] * (max_len - len(idx)))
    return np.asarray(idx, dtype=np.int64)


def encode_batch(seqs: list[str], max_len: int) -> np.ndarray:
    """Encode a list of sequences to an (N, max_len) int array."""
    return np.stack([encode_sequence(s, max_len) for s in seqs], axis=0)


def length_mask(seqs: list[str], max_len: int) -> np.ndarray:
    """Boolean (N, max_len) mask: True where a real residue sits, False on padding."""
    mask = np.zeros((len(seqs), max_len), dtype=bool)
    for i, s in enumerate(seqs):
        mask[i, : min(len(s), max_len)] = True
    return mask
