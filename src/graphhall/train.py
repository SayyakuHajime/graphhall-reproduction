from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from graphhall.metrics import classification_metrics
from graphhall.models import create_model, parameter_count
from graphhall.runtime import resolve_device


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


def _finite_logits(logits: Tensor, expected_rows: int, *, phase: str) -> None:
    if logits.shape != (expected_rows, 2):
        raise ValueError(f"{phase} model output must have shape ({expected_rows}, 2)")
    if not torch.isfinite(logits).all():
        raise FloatingPointError(f"{phase} logits contain non-finite values")


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
    device: str | torch.device = "cpu",
    model_profile: str = "weighted",
) -> dict[str, Any]:
    """Fit one transductive learner; test labels are used only for final metrics."""
    if max_epochs < 1 or validation_interval < 1 or patience < 1:
        raise ValueError("max_epochs, validation_interval, and patience must be positive")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("weight_decay must be finite and non-negative")
    resolved_device = resolve_device(device)
    x = features.detach().to(dtype=torch.float32, device="cpu")
    raw_y = labels.detach().to(device="cpu")
    if raw_y.is_floating_point() and (
        not torch.isfinite(raw_y).all() or not torch.equal(raw_y, raw_y.round())
    ):
        raise ValueError("labels must be finite binary integer values")
    y = raw_y.to(dtype=torch.long)
    _validate_masks(y, train_mask, val_mask, test_mask)
    if x.ndim != 2 or x.shape[0] != y.shape[0] or x.shape[1] < 1 or not torch.isfinite(x).all():
        raise ValueError("features must be finite, two-dimensional, and aligned with labels")
    if torch.any((y < 0) | (y > 1)):
        raise ValueError("labels must be binary values 0 or 1")
    cpu_masks = tuple(
        mask.to(dtype=torch.bool, device="cpu") for mask in (train_mask, val_mask, test_mask)
    )
    edges = torch.as_tensor(edge_index, dtype=torch.long, device="cpu")
    weights = torch.as_tensor(edge_weight, dtype=torch.float32, device="cpu")
    if edges.ndim != 2 or edges.shape != (2, weights.numel()):
        raise ValueError("edge_index and edge_weight shapes are inconsistent")
    if edges.numel() and (edges.min() < 0 or edges.max() >= x.shape[0]):
        raise ValueError("edge_index contains an invalid node index")
    if not torch.isfinite(weights).all():
        raise ValueError("edge_weight must contain only finite values")

    x = x.to(resolved_device)
    y = y.to(resolved_device)
    train_mask, val_mask, test_mask = (mask.to(resolved_device) for mask in cpu_masks)
    edges = edges.to(resolved_device)
    weights = weights.to(resolved_device)
    torch.manual_seed(seed)
    if resolved_device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = create_model(learner, input_dim=x.shape[1], model_profile=model_profile).to(
        resolved_device
    )
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
        _finite_logits(logits, x.shape[0], phase="training")
        train_loss_tensor = F.cross_entropy(logits[train_mask], y[train_mask], weight=class_weights)
        if not torch.isfinite(train_loss_tensor):
            raise FloatingPointError("training loss is non-finite")
        train_loss_tensor.backward()
        optimizer.step()

        should_validate = epoch % validation_interval == 0 or epoch == max_epochs
        if not should_validate:
            continue
        model.eval()
        with torch.no_grad():
            val_logits = model(x, edges, weights)
            _finite_logits(val_logits, x.shape[0], phase="validation")
            val_loss_tensor = F.cross_entropy(
                val_logits[val_mask], y[val_mask], weight=class_weights
            )
            if not torch.isfinite(val_loss_tensor):
                raise FloatingPointError("validation loss is non-finite")
            val_loss = float(val_loss_tensor.item())
        history.append(
            {"epoch": epoch, "train_loss": float(train_loss_tensor.item()), "val_loss": val_loss}
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {
                name: tensor.detach().to("cpu").clone()
                for name, tensor in model.state_dict().items()
            }
            torch.save(
                {
                    "state_dict": best_state,
                    "learner": learner,
                    "input_dimension": int(x.shape[1]),
                    "model_profile": model_profile,
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
        logits = model(x, edges, weights)
        _finite_logits(logits, x.shape[0], phase="final")
        probabilities = torch.softmax(logits, dim=1)
        if not torch.isfinite(probabilities).all():
            raise FloatingPointError("final probabilities contain non-finite values")
    probability_array = probabilities.to("cpu").numpy()
    prediction_array = probability_array.argmax(axis=1)
    y_array = y.to("cpu").numpy()
    val_indices = val_mask.to("cpu").numpy()
    test_indices = test_mask.to("cpu").numpy()
    validation_metrics = classification_metrics(
        y_array[val_indices], probability_array[val_indices]
    )
    metrics = classification_metrics(y_array[test_indices], probability_array[test_indices])
    return {
        "learner": learner,
        "model": model,
        "best_epoch": best_epoch,
        "best_validation_loss": best_val_loss,
        "parameter_count": parameter_count(model),
        "metrics": metrics,
        "validation_metrics": validation_metrics,
        "predictions": prediction_array,
        "probabilities": probability_array,
        "history": history,
        "checkpoint_path": str(target),
    }
