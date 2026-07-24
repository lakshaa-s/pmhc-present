import pytest

torch = pytest.importorskip("torch")  # skip cleanly if torch isn't installed here

from pmhcpresent.io.peptides import VOCAB_SIZE
from pmhcpresent.models.nn import NetConfig, PresentationNet, count_parameters


def test_forward_shape():
    cfg = NetConfig()
    net = PresentationNet(cfg)
    pep = torch.randint(0, VOCAB_SIZE, (8, cfg.max_pep_len))
    mhc = torch.randint(0, VOCAB_SIZE, (8, cfg.pseudoseq_len))
    out = net(pep, mhc)
    assert out.shape == (8,)


def test_predict_proba_range():
    net = PresentationNet()
    cfg = net.cfg
    pep = torch.randint(0, VOCAB_SIZE, (4, cfg.max_pep_len))
    mhc = torch.randint(0, VOCAB_SIZE, (4, cfg.pseudoseq_len))
    p = net.predict_proba(pep, mhc)
    assert p.shape == (4,)
    assert ((p >= 0) & (p <= 1)).all()


def test_backprop_runs():
    net = PresentationNet()
    cfg = net.cfg
    pep = torch.randint(0, VOCAB_SIZE, (16, cfg.max_pep_len))
    mhc = torch.randint(0, VOCAB_SIZE, (16, cfg.pseudoseq_len))
    y = torch.randint(0, 2, (16,)).float()
    logits = net(pep, mhc)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
    loss.backward()
    assert net.pep_enc.embed.weight.grad is not None


def test_param_count_positive():
    assert count_parameters(PresentationNet()) > 0
