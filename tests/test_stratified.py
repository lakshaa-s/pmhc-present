import numpy as np

from pmhcpresent.eval.stratified import assign_frequency_bins, stratified_metrics


def test_frequency_binning():
    freqs = np.array([0.001, 0.03, 0.1, 0.5])
    bins = [0.0, 0.01, 0.05, 0.2, 1.0]
    labels = ["very_rare", "rare", "uncommon", "common"]
    out = assign_frequency_bins(freqs, bins, labels)
    assert out.tolist() == ["very_rare", "rare", "uncommon", "common"]


def test_equity_gap_computed():
    strata = np.array(["common"] * 50 + ["rare"] * 50)
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 100)
    # near-perfect on common, near-random on rare
    score = np.where(strata == "common", y + rng.normal(0, 0.1, 100),
                     rng.normal(0, 1, 100))
    res = stratified_metrics(strata, y, score, metric="auroc")
    assert res["best"] == "common"
    assert res["gap"] >= 0
