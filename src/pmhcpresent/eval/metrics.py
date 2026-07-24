"""Core presentation-prediction metrics."""
from __future__ import annotations

import numpy as np


def auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return float("nan")        # undefined with one class present
    return float(roc_auc_score(y_true, y_score))


def average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score

    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def ppv_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """Positive predictive value among the top-k scored items."""
    y_true = np.asarray(y_true)
    order = np.argsort(y_score)[::-1][:k]
    if len(order) == 0:
        return float("nan")
    return float(y_true[order].mean())


def summary(y_true: np.ndarray, y_score: np.ndarray, top_k: int = 100) -> dict:
    return {
        "n": len(y_true),
        "n_pos": int(np.sum(y_true)),
        "auroc": auroc(y_true, y_score),
        "ap": average_precision(y_true, y_score),
        f"ppv@{top_k}": ppv_at_k(y_true, y_score, top_k),
    }


def per_allele(
    alleles: np.ndarray, y_true: np.ndarray, y_score: np.ndarray
) -> dict[str, dict]:
    """Metric summary computed independently for each allele."""
    alleles = np.asarray(alleles)
    out: dict[str, dict] = {}
    for a in np.unique(alleles):
        m = alleles == a
        out[str(a)] = summary(y_true[m], y_score[m])
    return out
