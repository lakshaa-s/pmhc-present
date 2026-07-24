"""Training loop and evaluation for the sequence presentation model.

Device order: CUDA (Beta's 4090) → MPS (your Mac) → CPU. Training uses
``BCEWithLogitsLoss`` on the presentation label, Adam, and early-stops on validation
AUROC (not loss — AUROC is what the RQs care about and is robust to class imbalance).
``evaluate`` reports overall metrics, per-allele metrics, and — when strata are
provided — the equity gap across allele-frequency / ancestry bins that the whole
project is about.

This is a deliberately small, honest training loop, not a framework. Swap in a
scheduler, class weighting, or mixed precision on Beta as needed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from pmhcpresent.eval.metrics import per_allele, summary
from pmhcpresent.eval.stratified import stratified_metrics


@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 0.0
    patience: int = 8          # early-stop after this many epochs without val-AUROC gain
    device: str | None = None  # None → auto-select
    num_workers: int = 0
    seed: int = 42


def select_device(prefer: str | None = None) -> torch.device:
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def _predict_proba(model, dataset, device, batch_size: int) -> np.ndarray:
    """Ordered presentation probabilities over a dataset (no shuffling)."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    out = []
    for pep, mhc, _ in loader:
        logits = model(pep.to(device), mhc.to(device))
        out.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(out) if out else np.array([])


def train_model(model, train_ds, val_ds, cfg: TrainConfig | None = None):
    """Train ``model`` and return (best_model, history).

    Restores the weights from the epoch with the best validation AUROC.
    """
    cfg = cfg or TrainConfig()
    torch.manual_seed(cfg.seed)
    device = select_device(cfg.device)
    model.to(device)

    loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers
    )
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    history = {"train_loss": [], "val_auroc": []}
    best_auroc, best_state, epochs_no_gain = -np.inf, None, 0

    for epoch in range(cfg.epochs):
        model.train()
        running = 0.0
        for pep, mhc, y in loader:
            pep, mhc, y = pep.to(device), mhc.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(pep, mhc), y)
            loss.backward()
            opt.step()
            running += loss.item() * len(y)
        train_loss = running / len(train_ds)

        val_probs = _predict_proba(model, val_ds, device, cfg.batch_size)
        val_auroc = summary(val_ds.labels, val_probs)["auroc"]
        history["train_loss"].append(train_loss)
        history["val_auroc"].append(val_auroc)

        improved = np.isnan(best_auroc) or (val_auroc > best_auroc)
        if not np.isnan(val_auroc) and improved:
            best_auroc = val_auroc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_gain = 0
        else:
            epochs_no_gain += 1

        print(f"epoch {epoch + 1:3d}  train_loss={train_loss:.4f}  "
              f"val_auroc={val_auroc:.4f}  best={best_auroc:.4f}")

        if epochs_no_gain >= cfg.patience:
            print(f"early stop at epoch {epoch + 1} (no val-AUROC gain in {cfg.patience})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def evaluate(model, dataset, cfg: TrainConfig | None = None, *, equity_metric: str = "auroc") -> dict:
    """Overall + per-allele metrics, plus the equity gap if the dataset has strata."""
    cfg = cfg or TrainConfig()
    device = select_device(cfg.device)
    model.to(device)
    probs = _predict_proba(model, dataset, device, cfg.batch_size)

    result = {"overall": summary(dataset.labels, probs)}
    if dataset.alleles is not None:
        result["per_allele"] = per_allele(
            np.asarray(dataset.alleles), dataset.labels, probs
        )
    if dataset.strata is not None:
        result["equity"] = stratified_metrics(
            dataset.strata, dataset.labels, probs, metric=equity_metric
        )
    return result
