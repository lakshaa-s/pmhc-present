import numpy as np
from pmhcpresent.eval.splits import hamming_identity, greedy_cluster, grouped_kfold


def test_hamming_identity():
    assert hamming_identity("AAAA", "AAAA") == 1.0
    assert hamming_identity("AAAA", "AAAB") == 0.75
    assert hamming_identity("AAA", "AAAA") == 0.0  # different length


def test_clustering_groups_similar():
    peptides = ["SIINFEKL", "SIINFEKL", "SIINFEKM", "GILGFVFTL"]
    cids = greedy_cluster(peptides, identity_threshold=0.8)
    # first three are ≥7/8 identical → same cluster; last is alone
    assert cids[0] == cids[1] == cids[2]
    assert cids[3] != cids[0]


def test_grouped_kfold_no_cluster_straddle():
    cids = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    for train, test in grouped_kfold(cids, n_splits=5):
        train_clusters = set(cids[train])
        test_clusters = set(cids[test])
        assert train_clusters.isdisjoint(test_clusters)
