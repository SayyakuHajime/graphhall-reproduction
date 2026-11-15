from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def classification_metrics(
    y_true: Any, probabilities: Any, *, bins: int = 10
) -> dict[str, float | None]:
    """Compute hallucination-positive classification and factual calibration metrics."""
    if isinstance(bins, bool) or not isinstance(bins, (int, np.integer)) or bins < 1:
        raise ValueError("bins must be a positive integer")
    try:
        labels = np.asarray(y_true, dtype=np.float64)
        probs = np.asarray(probabilities, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("labels and probabilities must be numeric") from exc
    if labels.ndim != 1 or labels.size == 0:
        raise ValueError("y_true must be a non-empty one-dimensional array")
    if not np.isfinite(labels).all() or not np.isin(labels, (0.0, 1.0)).all():
        raise ValueError("labels must contain only finite binary values 0 or 1")
    labels = labels.astype(np.int64)
    if probs.shape != (labels.size, 2):
        raise ValueError("probabilities must have shape (N, 2) aligned with y_true")
    if not np.isfinite(probs).all():
        raise ValueError("probabilities must contain only finite values")
    if np.any((probs < 0.0) | (probs > 1.0)):
        raise ValueError("probabilities must be within [0, 1]")
    if not np.allclose(probs.sum(axis=1), 1.0, rtol=1e-6, atol=1e-6):
        raise ValueError("each probability row must sum to 1")

    predictions = probs.argmax(axis=1)
    true_positive = int(np.count_nonzero((labels == 1) & (predictions == 1)))
    false_positive = int(np.count_nonzero((labels == 0) & (predictions == 1)))
    false_negative = int(np.count_nonzero((labels == 1) & (predictions == 0)))
    accuracy = float(np.mean(predictions == labels))
    precision = (
        true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    )
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0

    factual_probability = probs[:, 0]
    factual_truth = 1 - labels
    bin_index = np.minimum((factual_probability * bins).astype(np.int64), bins - 1)
    ece = 0.0
    for index in range(bins):
        in_bin = bin_index == index
        if in_bin.any():
            ece += float(in_bin.mean()) * abs(
                float(factual_probability[in_bin].mean()) - float(factual_truth[in_bin].mean())
            )
    brier = float(np.mean((factual_probability - factual_truth) ** 2))
    true_probability = np.where(labels == 1, probs[:, 1], probs[:, 0])
    nll = float(-np.log(np.clip(true_probability, np.finfo(np.float64).tiny, 1.0)).mean())

    if np.unique(labels).size == 1:
        auroc = None
        auprc = None
    else:
        auroc = float(roc_auc_score(labels, probs[:, 1]))
        auprc = float(average_precision_score(labels, probs[:, 1]))
    return {
        "accuracy": accuracy,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auroc": auroc,
        "auprc": auprc,
        "ece": float(ece),
        "brier": brier,
        "nll": nll,
    }
