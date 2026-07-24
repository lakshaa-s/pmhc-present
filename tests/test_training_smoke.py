import pytest

pytest.importorskip("torch")

import numpy as np

from pmhcpresent.io.peptides import AMINO_ACIDS
from pmhcpresent.models.nn import NetConfig, PresentationNet
from pmhcpresent.train import PeptideMHCDataset, TrainConfig, evaluate, train_model


def _make_learnable_dataset(n=400, seed=0):
    rng = np.random.default_rng(seed)
    aas = list(AMINO_ACIDS)
    peps, pseuds, labels, strata = [], [], [], []
    pseudo = "Y" * 34
    for _ in range(n):
        L = int(rng.integers(8, 11))
        pep = "".join(rng.choice(aas, size=L))
        # learnable rule: anchor 'L' at position 2 → more likely presented
        prob = 0.8 if pep[1] == "L" else 0.2
        peps.append(pep)
        pseuds.append(pseudo)
        labels.append(int(rng.random() < prob))
        strata.append("common" if rng.random() < 0.7 else "rare")
    return PeptideMHCDataset(peps, pseuds, labels,
                             alleles=["HLA-A*02:01"] * n, strata=strata)


def test_training_runs_and_learns():
    train_ds = _make_learnable_dataset(400, seed=0)
    val_ds = _make_learnable_dataset(150, seed=1)
    model = PresentationNet(NetConfig())
    cfg = TrainConfig(epochs=8, batch_size=64, patience=8, device="cpu")
    model, history = train_model(model, train_ds, val_ds, cfg)

    assert len(history["train_loss"]) >= 1
    # loss should generally fall from first to last recorded epoch
    assert history["train_loss"][-1] <= history["train_loss"][0] + 1e-3


def test_evaluate_returns_equity():
    ds = _make_learnable_dataset(200, seed=2)
    model = PresentationNet(NetConfig())
    res = evaluate(model, ds, TrainConfig(device="cpu"))
    assert "overall" in res and "auroc" in res["overall"]
    assert "equity" in res
    assert set(res["equity"]["per_stratum"]) <= {"common", "rare"}
