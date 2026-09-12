"""Custom sequence model for peptide–HLA-I presentation.

Architecture (a deliberate, simple starting point — tune later):

    peptide indices  ─embed─► 1D conv ─► global max-pool ┐
                                                          ├─ concat ─► MLP ─► logit
    HLA pseudoseq    ─embed─► 1D conv ─► global mean-pool ┘

The output is a single presentation logit (apply a sigmoid for probability). The
peptide path uses max-pooling (a binding motif is a local, position-flexible signal);
the allele path uses mean-pooling over the fixed 34-residue pseudosequence.

Validate the forward pass on dummy tensors (CPU or Mac MPS) before sending anything
to Beta:

    >>> import torch
    >>> from pmhcpresent.models.nn import PresentationNet, NetConfig
    >>> net = PresentationNet(NetConfig())
    >>> pep = torch.randint(0, 22, (8, 15))
    >>> mhc = torch.randint(0, 22, (8, 34))
    >>> net(pep, mhc).shape
    torch.Size([8])
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from pmhcpresent.io.peptides import PAD_IDX, VOCAB_SIZE


@dataclass
class NetConfig:
    """Architecture hyperparameters for `PresentationNet`. No ablation flags —
    those live in `scripts/ablation_*.py`, which build separate configs."""

    vocab_size: int = VOCAB_SIZE
    pad_idx: int = PAD_IDX
    max_pep_len: int = 15
    pseudoseq_len: int = 34
    embed_dim: int = 32
    conv_channels: int = 64
    kernel_size: int = 3
    hidden_dim: int = 128
    dropout: float = 0.3


class _SeqEncoder(nn.Module):
    """Embed → conv → pooled vector. Pooling is 'max' or 'mean'."""

    def __init__(self, cfg: NetConfig, pool: str):
        super().__init__()
        assert pool in ("max", "mean")
        self.pool = pool
        self.embed = nn.Embedding(cfg.vocab_size, cfg.embed_dim, padding_idx=cfg.pad_idx)
        self.conv = nn.Conv1d(
            cfg.embed_dim,
            cfg.conv_channels,
            kernel_size=cfg.kernel_size,
            padding=cfg.kernel_size // 2,
        )
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x, integer residue indices, shape (batch, length).
        Returns: pooled feature vector, shape (batch, conv_channels)."""
        h = self.embed(x)                 # (B, L, E)
        h = h.transpose(1, 2)             # (B, E, L)
        h = self.act(self.conv(h))        # (B, C, L)
        if self.pool == "max":
            return h.max(dim=-1).values   # (B, C)
        return h.mean(dim=-1)             # (B, C)


class PresentationNet(nn.Module):
    """Sequence baseline NN producing a presentation logit."""

    def __init__(self, cfg: NetConfig | None = None):
        super().__init__()
        self.cfg = cfg or NetConfig()
        self.pep_enc = _SeqEncoder(self.cfg, pool="max")
        self.mhc_enc = _SeqEncoder(self.cfg, pool="mean")
        feat_dim = 2 * self.cfg.conv_channels
        self.head = nn.Sequential(
            nn.Linear(feat_dim, self.cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(self.cfg.hidden_dim, 1),
        )

    def forward(self, peptide: torch.Tensor, mhc: torch.Tensor) -> torch.Tensor:
        """Args: peptide indices (batch, max_pep_len), mhc pseudoseq indices
        (batch, pseudoseq_len). Returns: raw presentation logit, shape (batch,)
        — apply sigmoid for a probability, or use `predict_proba` below."""
        p = self.pep_enc(peptide)         # (B, C)
        m = self.mhc_enc(mhc)             # (B, C)
        feats = torch.cat([p, m], dim=-1)
        return self.head(feats).squeeze(-1)   # (B,)

    @torch.no_grad()
    def predict_proba(self, peptide: torch.Tensor, mhc: torch.Tensor) -> torch.Tensor:
        """Same inputs as `forward`. Returns: presentation probability in [0, 1]."""
        return torch.sigmoid(self(peptide, mhc))


def count_parameters(model: nn.Module) -> int:
    """Total trainable parameters in `model` (used to report the 30,465 figure)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
