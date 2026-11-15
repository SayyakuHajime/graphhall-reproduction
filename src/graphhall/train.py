from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch import Tensor
from torch.nn import functional as F

from graphhall.models import create_model, parameter_count


def _validate_masks(
    labels: Tensor, train_mask: Tensor, val_mask: Tensor, test_mask: Tensor
) -> None:
    masks = [mask.to(dtype=torch.bool, device="cpu") for mask in (train_mask, val_mask, test_mask)]
    if any(mask.ndim != 1 or mask.shape != labels.shape for mask in masks):
        raise ValueError("each split mask must be one-dimensional with one value per label")
    if any(not bool(mask.any()) for mask in masks):
        raise ValueError("train, validation, and test masks must all be non-empty")
    if (masks[0] & masks[1]).any() or (masks[0] & masks[2]).any() or (masks[1] & masks[2]).any():
        raise ValueError("train, validation, and test masks must be disjoint")


def _test_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def train_model(
    learner: str,
    features: Tensor,
    edge_index: np.ndarray,
    edge_weight: np.ndarray,
    labels: Tensor,
    train_mask: Tensor,
    val_mask: Tensor,
    test_mask: Tensor,
    *,
    checkpoint_path: str | Path,
    learning_rate: float = 0.005,
    weight_decay: float = 0.0005,
    max_epochs: int = 400,
    validation_interval: int = 20,
    patience: int = 3,
    seed: int = 42,
) -> dict[str, Any]:
    """Fit one transductive learner; test labels are used only for final metrics."""
    if max_epochs < 1 or validation_interval < 1 or patience < 1:
        raise ValueError("max_epochs, validation_interval, and patience must be positive")
    x = features.detach().to(dtype=torch.float32, device="cpu")
    y = labels.detach().to(dtype=torch.long, device="cpu")
    _validate_masks(y, train_mask, val_mask, test_mask)
    if x.ndim != 2 or x.shape[0] != y.shape[0] or not torch.isfinite(x).all():
        raise ValueError("features must be finite, two-dimensional, and aligned with labels")
    if torch.any((y < 0) | (y > 1)):
        raise ValueError("labels must be binary values 0 or 1")
    train_mask = train_mask.to(dtype=torch.bool, device="cpu")
    val_mask = val_mask.to(dtype=torch.bool, device="cpu")
    test_mask = test_mask.to(dtype=torch.bool, device="cpu")
    edges = torch.as_tensor(edge_index, dtype=torch.long)
    weights = torch.as_tensor(edge_weight, dtype=torch.float32)
    if edges.shape != (2, weights.numel()):
        raise ValueError("edge_index and edge_weight shapes are inconsistent")

    torch.manual_seed(seed)
    model = create_model(learner, input_dim=x.shape[1])
    train_counts = torch.bincount(y[train_mask], minlength=2).to(torch.float32).clamp_min(1)
    class_weights = train_counts.sum() / (2.0 * train_counts)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    target = Path(checkpoint_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    best_state: dict[str, Tensor] | None = None
    best_epoch = 0
    checks_without_improvement = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(x, edges, weights)
        train_loss = F.cross_entropy(logits[train_mask], y[train_mask], weight=class_weights)
        train_loss.backward()
        optimizer.step()

        should_validate = epoch % validation_interval == 0 or epoch == max_epochs
        if not should_validate:
            continue
        model.eval()
        with torch.no_grad():
            val_logits = model(x, edges, weights)
            val_loss = F.cross_entropy(
                val_logits[val_mask], y[val_mask], weight=class_weights
            ).item()
        history.append(
            {"epoch": epoch, "train_loss": float(train_loss.item()), "val_loss": val_loss}
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {
                name: tensor.detach().clone() for name, tensor in model.state_dict().items()
            }
            torch.save(
                {
                    "state_dict": best_state,
                    "learner": learner,
                    "best_epoch": best_epoch,
                    "best_validation_loss": best_val_loss,
                    "parameter_count": parameter_count(model),
                    "seed": seed,
                },
                target,
            )
            checks_without_improvement = 0
        else:
            checks_without_improvement += 1
            if checks_without_improvement >= patience:
                break

    if best_state is None:
        raise RuntimeError("training did not produce a validation checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        predictions = model(x, edges, weights).argmax(dim=1)
    test_indices = test_mask.numpy()
    metrics = _test_metrics(y.numpy()[test_indices], predictions.numpy()[test_indices])
    return {
        "learner": learner,
        "model": model,
        "best_epoch": best_epoch,
        "best_validation_loss": best_val_loss,
        "parameter_count": parameter_count(model),
        "metrics": metrics,
        "predictions": predictions.numpy(),
        "history": history,
        "checkpoint_path": str(target),
    }
