"""Ensemble of the sequence model and the structure features (RQ2).

RQ2 asks whether sequence and structure combine synergistically. This module provides
a simple, honest baseline stacker: a logistic-regression meta-model over
[sequence_score, structure_features...]. Compare it against each component alone to
test for synergy (ensemble AUROC/AP minus the best single model).

OPEN DESIGN QUESTION (flagged, not resolved): how structural features become a
binding *score* and what training label drives the structure side. Options to weigh:
  - train the meta-model directly on the presentation label (treat structure features
    as extra inputs alongside the sequence score) — simplest, used below;
  - train a separate structure-only classifier first, then stack its score — cleaner
    for the RQ1 "structure vs sequence" comparison but needs its own labels/split.
Pick deliberately; it changes what RQ1 vs RQ2 actually measure.
"""
from __future__ import annotations

import numpy as np


class EnsembleStacker:
    """Logistic-regression stacker over sequence score + structure features."""

    def __init__(self, C: float = 1.0):
        from sklearn.linear_model import LogisticRegression

        self.model = LogisticRegression(C=C, max_iter=1000)
        self.feature_names_: list[str] | None = None

    def fit(self, seq_scores: np.ndarray, struct_feats: np.ndarray, y: np.ndarray):
        X = self._stack(seq_scores, struct_feats)
        self.model.fit(X, y)
        return self

    def predict_proba(self, seq_scores: np.ndarray, struct_feats: np.ndarray) -> np.ndarray:
        X = self._stack(seq_scores, struct_feats)
        return self.model.predict_proba(X)[:, 1]

    @staticmethod
    def _stack(seq_scores: np.ndarray, struct_feats: np.ndarray) -> np.ndarray:
        seq_scores = np.asarray(seq_scores).reshape(-1, 1)
        struct_feats = np.asarray(struct_feats)
        if struct_feats.ndim == 1:
            struct_feats = struct_feats.reshape(-1, 1)
        return np.hstack([seq_scores, struct_feats])


def rank_average(*score_arrays: np.ndarray) -> np.ndarray:
    """Parameter-free baseline: average of rank-normalised scores.

    Useful as a no-training synergy check before fitting a stacker.
    """
    from scipy.stats import rankdata

    ranks = [rankdata(s) / len(s) for s in score_arrays]
    return np.mean(ranks, axis=0)
