"""Cluster-based splitting to control leakage.

Random peptide splits leak: near-identical peptides land in both train and test and
inflate scores. The literature-review spec calls for cluster-based splitting — cluster
similar peptides, then assign whole clusters to a fold so no cluster straddles
train/test. This module provides greedy single-linkage clustering on equal-length
peptides plus a grouped split over precomputed cluster ids.
"""
from __future__ import annotations

import numpy as np
import os
import subprocess
import tempfile


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

def mmseqs_cluster(peptides, alleles=None, identity_threshold=0.8,
                   mmseqs_bin="mmseqs", per_allele=True):
    """Cluster peptides with MMseqs2 (near-duplicate-aware), globally-unique ids.
 
    Near-duplicate-aware alternative to exact_dedup_cluster: catches peptides that
    differ by a residue or two, which exact dedup misses. Uses short-sequence
    parameters. With alleles + per_allele, clusters within each allele and offsets
    ids so none collide across alleles.
    """
    peptides = list(peptides)
    if alleles is None or not per_allele:
        return _mmseqs_one(peptides, identity_threshold, mmseqs_bin)
 
    alleles = list(alleles)
    if len(alleles) != len(peptides):
        raise ValueError("peptides and alleles must be the same length")
    cluster_ids = np.full(len(peptides), -1, dtype=int)
    by_allele = {}
    for i, a in enumerate(alleles):
        by_allele.setdefault(a, []).append(i)
    offset = 0
    for allele, idxs in by_allele.items():
        sub = [peptides[i] for i in idxs]
        sub_ids = _mmseqs_one(sub, identity_threshold, mmseqs_bin)
        for j, i in enumerate(idxs):
            cluster_ids[i] = int(sub_ids[j]) + offset
        offset += int(sub_ids.max()) + 1 if len(sub_ids) else 0
    return cluster_ids
 
 
# --- add these functions at the END of splits.py (numpy already imported) ---


def _hamming_le_threshold(a, b, max_diffs):
    """True if equal-length a, b differ in <= max_diffs positions."""
    diffs = 0
    for x, y in zip(a, b):
        if x != y:
            diffs += 1
            if diffs > max_diffs:
                return False
    return True


def _cluster_unique_peptides(uniq_peps, identity_threshold):
    """Greedy single-linkage on UNIQUE peptides, bucketed by length.

    Returns {peptide: local_cluster_id}. Only compares within a length bucket and
    against existing reps, so cost stays low once exact duplicates are removed.
    """
    by_len = {}
    for p in uniq_peps:
        by_len.setdefault(len(p), []).append(p)

    cid_of = {}
    next_id = 0
    for length, peps in by_len.items():
        max_diffs = int(round((1.0 - identity_threshold) * length))
        reps = []
        for p in peps:
            hit = None
            for rep_pep, rep_cid in reps:
                if _hamming_le_threshold(p, rep_pep, max_diffs):
                    hit = rep_cid
                    break
            if hit is None:
                hit = next_id
                reps.append((p, hit))
                next_id += 1
            cid_of[p] = hit
    return cid_of

def hamming_cluster(peptides, alleles=None, identity_threshold=0.8):
    """Near-duplicate-aware clustering for short peptides, globally-unique ids.

    The peptide-appropriate near-duplicate method. Protein clustering tools
    (MMseqs2/CD-HIT) rely on k-mer prefilters that don't transfer to 8-11mers
    (they barely cluster short peptides), so this uses Hamming identity directly.
    Per allele: dedup exact repeats, then greedy single-linkage by Hamming
    identity within each length bucket. Near-duplicates must share length and
    allele, keeping comparisons cheap. Cluster ids are offset per allele so none
    collide. Stricter than exact_dedup_cluster (catches 1-2 residue near-dups);
    use for the final leakage-controlled split.
    """
    peptides = list(peptides)
    n = len(peptides)
    cluster_ids = np.full(n, -1, dtype=int)

    if alleles is None:
        alleles = ["_"] * n
    else:
        alleles = list(alleles)
        if len(alleles) != n:
            raise ValueError("peptides and alleles must be the same length")

    by_allele = {}
    for i, a in enumerate(alleles):
        by_allele.setdefault(a, []).append(i)

    offset = 0
    for allele, idxs in by_allele.items():
        uniq = list({peptides[i] for i in idxs})
        cid_of = _cluster_unique_peptides(uniq, identity_threshold)
        local_max = max(cid_of.values()) if cid_of else -1
        for i in idxs:
            cluster_ids[i] = cid_of[peptides[i]] + offset
        offset += local_max + 1
    return cluster_ids