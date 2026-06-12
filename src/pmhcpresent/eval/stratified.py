"""Equity-stratified evaluation — the project's core contribution.

The equity question is whether a model performs *as well* on underrepresented HLA
alleles as on common ones. This module stratifies metrics by allele frequency bin
(a proxy for representation in training data) and/or ancestry group, then reports an
explicit **equity gap** between the best- and worst-served strata.

Framing caution (from the RQ1 design): a structure model is not *assumed* to be more
equitable — AlphaFold itself trains on an ancestry-skewed PDB, so any equity benefit
is something this harness *tests*, not something the architecture guarantees.
"""
from __future__ import annotations

import numpy as np

from pmhcpresent.eval.metrics import summary


def assign_frequency_bins(
    frequencies: np.ndarray, bins: list[float], labels: list[str]
) -> np.ndarray:
    """Map per-item allele frequencies to ordinal representation bins."""
    frequencies = np.asarray(frequencies, dtype=float)
    idx = np.digitize(frequencies, bins[1:-1], right=False)
    idx = np.clip(idx, 0, len(labels) - 1)
    return np.array([labels[i] for i in idx])


def stratified_metrics(
    strata: np.ndarray,
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric: str = "auroc",
) -> dict:
    """Per-stratum metric summaries plus an equity gap.

    Returns ``{"per_stratum": {...}, "gap": float, "best": str, "worst": str}`` where
    ``gap`` = best metric − worst metric across strata (lower is more equitable).
    """
    strata = np.asarray(strata)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    per_stratum: dict[str, dict] = {}
    for s in np.unique(strata):
        m = strata == s
        per_stratum[str(s)] = summary(y_true[m], y_score[m])

    scored = {
        s: d[metric] for s, d in per_stratum.items()
        if d.get(metric) is not None and not _isnan(d[metric])
    }
    if len(scored) < 2:
        return {"per_stratum": per_stratum, "gap": float("nan"), "best": None, "worst": None}

    best = max(scored, key=scored.get)
    worst = min(scored, key=scored.get)
    return {
        "per_stratum": per_stratum,
        "metric": metric,
        "gap": float(scored[best] - scored[worst]),
        "best": best,
        "worst": worst,
    }


def compare_models_equity(
    strata: np.ndarray,
    y_true: np.ndarray,
    model_scores: dict[str, np.ndarray],
    metric: str = "auroc",
) -> dict:
    """Compare equity gaps across models (e.g. sequence vs structure vs ensemble).

    Directly answers RQ1's equity comparison: does the structure model shrink the gap
    on underrepresented alleles relative to the sequence model?
    """
    return {
        name: stratified_metrics(strata, y_true, scores, metric)
        for name, scores in model_scores.items()
    }


def _isnan(x) -> bool:
    try:
        return np.isnan(x)
    except TypeError:
        return False
