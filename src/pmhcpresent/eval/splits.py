"""Cluster-based splitting to control leakage.

Random peptide splits leak: near-identical peptides land in both train and test and
inflate scores. The literature-review spec calls for cluster-based splitting — cluster
similar peptides, then assign whole clusters to a fold so no cluster straddles
train/test. This module provides greedy single-linkage clustering on equal-length
peptides plus a grouped split over precomputed cluster ids.
"""
from __future__ import annotations

import numpy as np


def hamming_identity(a: str, b: str) -> float:
    """Fractional identity for equal-length peptides (1.0 = identical)."""
    if len(a) != len(b):
        return 0.0
    if not a:
        return 1.0
    same = sum(x == y for x, y in zip(a, b))
    return same / len(a)


def greedy_cluster(
    peptides: list[str], identity_threshold: float = 0.8
) -> np.ndarray:
    """Greedy single-linkage clustering by length then identity.

    Peptides of different lengths never share a cluster. Within a length, a peptide
    joins the first existing cluster whose representative it matches at or above the
    threshold, else it seeds a new cluster. Returns an int cluster id per peptide.

    O(n·k) in the number of clusters k — fine for tens of thousands of peptides; swap
    in MMseqs2/CD-HIT for larger sets.
    """
    cluster_ids = np.full(len(peptides), -1, dtype=int)
    reps: list[tuple[int, str]] = []   # (cluster_id, representative peptide)
    next_id = 0

    for i, pep in enumerate(peptides):
        assigned = False
        for cid, rep in reps:
            if len(rep) == len(pep) and hamming_identity(pep, rep) >= identity_threshold:
                cluster_ids[i] = cid
                assigned = True
                break
        if not assigned:
            cluster_ids[i] = next_id
            reps.append((next_id, pep))
            next_id += 1
    return cluster_ids


def grouped_kfold(cluster_ids: np.ndarray, n_splits: int = 5, seed: int = 42):
    """Yield (train_idx, test_idx) keeping whole clusters within a single fold."""
    from sklearn.model_selection import GroupKFold

    n = len(cluster_ids)
    gkf = GroupKFold(n_splits=n_splits)
    dummy_X = np.zeros((n, 1))
    dummy_y = np.zeros(n)
    yield from gkf.split(dummy_X, dummy_y, groups=cluster_ids)

def exact_dedup_cluster(peptides, alleles=None):
    """Fast leakage-control split helper: identical peptides share a cluster id.

    A pragmatic stand-in for full identity clustering. It guarantees that exact
    duplicate peptides never straddle a train/test split (the most severe leakage),
    and runs in O(n) via hashing rather than O(n^2) pairwise comparison. It does
    NOT catch near-duplicates (peptides differing by a residue or two) -- swap in
    MMseqs2/CD-HIT clustering for that. If `alleles` is given, identity is keyed on
    (allele, peptide) so the same peptide under different alleles is treated as
    distinct prediction problems.
    """
    peptides = list(peptides)
    if alleles is not None:
        alleles = list(alleles)
        if len(alleles) != len(peptides):
            raise ValueError("peptides and alleles must be the same length")
        keys = list(zip(alleles, peptides))
    else:
        keys = peptides
    id_of = {}
    cluster_ids = np.empty(len(keys), dtype=int)
    nxt = 0
    for i, k in enumerate(keys):
        cid = id_of.get(k)
        if cid is None:
            cid = nxt
            id_of[k] = cid
            nxt += 1
        cluster_ids[i] = cid
    return cluster_ids
