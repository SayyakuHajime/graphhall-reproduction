from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from graphhall.metrics import classification_metrics
from graphhall.models import parameter_count
from graphhall.runtime import resolve_device


def _validated_baseline_inputs(
    features: Any,
    labels: Any,
    train_mask: Any,
    val_mask: Any,
    test_mask: Any,
) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    try:
        x = np.asarray(features, dtype=np.float64)
        y_raw = np.asarray(labels)
        masks = tuple(np.asarray(mask, dtype=bool) for mask in (train_mask, val_mask, test_mask))
    except (TypeError, ValueError) as exc:
        raise ValueError("features, labels, and masks must be array-like") from exc
    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError("features must have shape (N>=1, D>=1)")
    if not np.isfinite(x).all():
        raise ValueError("features must contain only finite values")
    if y_raw.ndim != 1 or y_raw.shape[0] != x.shape[0]:
        raise ValueError("labels must be one-dimensional and aligned with features")
    try:
        y = y_raw.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("labels must be binary numeric values") from exc
    if not np.isfinite(y).all() or not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("labels must contain only finite binary values 0 or 1")
    y = y.astype(np.int64)
    if any(mask.ndim != 1 or mask.shape[0] != x.shape[0] for mask in masks):
        raise ValueError("each split mask must have one value per feature row")
    if any(not mask.any() for mask in masks):
        raise ValueError("train, validation, and test masks must all be non-empty")
    if (masks[0] & masks[1]).any() or (masks[0] & masks[2]).any() or (masks[1] & masks[2]).any():
        raise ValueError("train, validation, and test masks must be disjoint")
    return x, y, masks  # type: ignore[return-value]


def run_knn_baseline(
    features: Any,
    labels: Any,
    train_mask: Any,
    val_mask: Any,
    test_mask: Any,
    *,
    k: int = 5,
    chunk_size: int = 128,
) -> dict[str, Any]:
    """Predict with bounded cosine batches and votes from training rows only."""
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError("k must be a positive integer")
    x, y, (train, validation, test) = _validated_baseline_inputs(
        features, labels, train_mask, val_mask, test_mask
    )
    train_indices = np.flatnonzero(train)
    if train_indices.size < 2:
        raise ValueError("at least two training rows are required to exclude self-neighbors")
    norms = np.linalg.norm(x, axis=1)
    if np.any(norms == 0):
        raise ValueError("features must not contain zero-norm rows for cosine kNN")
    normalized = x / norms[:, None]
    probabilities = np.zeros((x.shape[0], 2), dtype=np.float64)

    reference = normalized[train_indices]
    reference_labels = y[train_indices]
    if (
        isinstance(chunk_size, bool)
        or not isinstance(chunk_size, (int, np.integer))
        or chunk_size < 1
    ):
        raise ValueError("chunk_size must be a positive integer")
    width = min(int(k), train_indices.size)
    for start in range(0, x.shape[0], int(chunk_size)):
        stop = min(start + int(chunk_size), x.shape[0])
        similarities = normalized[start:stop] @ reference.T
        query_indices = np.arange(start, stop)
        train_queries = train[query_indices]
        if train_queries.any():
            reference_offsets = np.searchsorted(train_indices, query_indices[train_queries])
            similarities[np.flatnonzero(train_queries), reference_offsets] = -np.inf
        # Reference IDs are ascending; stable sort preserves lower-ID tie breaks.
        neighbors = np.argsort(-similarities, axis=1, kind="stable")[:, :width]
        counts = np.minimum(int(k), train_indices.size - train_queries.astype(np.int64))
        valid_votes = np.arange(width)[None, :] < counts[:, None]
        positive_votes = (reference_labels[neighbors] * valid_votes).sum(axis=1)
        probabilities[start:stop, 1] = positive_votes / counts
        probabilities[start:stop, 0] = 1.0 - probabilities[start:stop, 1]

    predictions = probabilities.argmax(axis=1).astype(np.int64)
    validation_metrics = classification_metrics(y[validation], probabilities[validation])
    test_metrics = classification_metrics(y[test], probabilities[test])
    validation_loss = validation_metrics["nll"]
    assert validation_loss is not None
    return {
        "probabilities": probabilities,
        "predictions": predictions,
        "metrics": test_metrics,
        "validation_metrics": validation_metrics,
        "parameter_count": 0,
        "best_epoch": 0,
        "best_validation_loss": float(validation_loss),
    }


class _EncoderClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, hidden_size: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, 2)

    def forward(self, inputs: Mapping[str, Tensor]) -> Tensor:
        output = self.encoder(**inputs)
        hidden = getattr(output, "last_hidden_state", None)
        if not isinstance(hidden, Tensor) or hidden.ndim != 3 or hidden.shape[1] < 1:
            raise ValueError("encoder must return last_hidden_state with shape (N, L>=1, H)")
        return self.classifier(self.dropout(hidden[:, 0, :]))


def _load_prepared_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"prepared data file does not exist: {path}")
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"prepared data file cannot be read: {path}") from exc
    if not lines:
        raise ValueError("prepared data file must contain at least one row")
    for line_number, line in enumerate(lines, start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"prepared data line {line_number} is not valid JSON") from exc
        if not isinstance(row, dict):
            raise ValueError(f"prepared data line {line_number} must be a JSON object")
        record_id = row.get("record_id")
        if not isinstance(record_id, str) or not record_id:
            raise ValueError(f"prepared data line {line_number} needs a non-empty record_id")
        if record_id in seen_ids:
            raise ValueError(f"prepared data contains duplicate record_id {record_id!r}")
        seen_ids.add(record_id)
        if row.get("split") not in {"train", "val", "test"}:
            raise ValueError(f"prepared data line {line_number} has invalid split")
        for field in ("question", "answer"):
            if not isinstance(row.get(field), str):
                raise ValueError(f"prepared data line {line_number} needs string field {field!r}")
        label = row.get("label")
        if isinstance(label, bool) or not isinstance(label, int) or label not in (0, 1):
            raise ValueError(f"prepared data line {line_number} label must be integer 0 or 1")
        rows.append(row)
    for split in ("train", "val", "test"):
        if not any(row["split"] == split for row in rows):
            raise ValueError(f"prepared data must include a non-empty {split!r} split")
    return rows


def _tokenize_rows(
    tokenizer: Any, rows: list[dict[str, Any]], max_length: int, batch_size: int
) -> dict[str, Tensor]:
    accumulated: dict[str, list[Tensor]] = {}
    expected_keys: set[str] | None = None
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        encoded = tokenizer(
            [row["question"] for row in batch],
            [row["answer"] for row in batch],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        if not isinstance(encoded, Mapping) or "input_ids" not in encoded:
            raise ValueError("tokenizer must return a mapping containing input_ids")
        keys = set(encoded)
        if expected_keys is None:
            expected_keys = keys
        elif keys != expected_keys:
            raise ValueError("tokenizer returned inconsistent fields across batches")
        for name, value in encoded.items():
            if not isinstance(value, Tensor) or value.ndim < 2 or value.shape[0] != len(batch):
                raise ValueError(f"tokenizer field {name!r} must be a batched tensor")
            accumulated.setdefault(name, []).append(value.detach().to("cpu"))
    if not accumulated:
        raise ValueError("tokenizer returned no model inputs")
    return {name: torch.cat(chunks, dim=0) for name, chunks in accumulated.items()}


def _encoder_config(encoder: nn.Module) -> tuple[int, dict[str, Any], str | None]:
    config = getattr(encoder, "config", None)
    if config is None:
        raise ValueError("encoder must expose a config with hidden_size")
    hidden_size = getattr(config, "hidden_size", getattr(config, "d_model", None))
    if isinstance(hidden_size, bool) or not isinstance(hidden_size, int) or hidden_size < 1:
        raise ValueError("encoder config must expose a positive hidden_size or d_model")
    to_dict = getattr(config, "to_dict", None)
    config_dict = to_dict() if callable(to_dict) else vars(config)
    if not isinstance(config_dict, dict):
        raise ValueError("encoder config must be serializable as a mapping")
    return hidden_size, config_dict, getattr(config, "_commit_hash", None)


def _model_inputs(
    inputs: Mapping[str, Tensor], indices: Tensor, device: torch.device
) -> dict[str, Tensor]:
    return {name: value[indices].to(device) for name, value in inputs.items()}


def _validate_logits(logits: Tensor, rows: int, *, phase: str) -> None:
    if logits.shape != (rows, 2):
        raise ValueError(f"{phase} classifier output must have shape ({rows}, 2)")
    if not torch.isfinite(logits).all():
        raise FloatingPointError(f"{phase} classifier logits contain non-finite values")


def finetune_classifier(
    prepared_path: str | Path,
    output_dir: str | Path,
    *,
    model_name: str,
    revision: str,
    device: str | torch.device = "cpu",
    batch_size: int = 1,
    max_length: int = 512,
    epochs: int = 3,
    learning_rate: float = 2e-5,
    seed: int = 42,
    gradient_accumulation: int = 8,
    max_steps: int | None = None,
    cache_dir: str | Path | None = None,
    tokenizer: Any | None = None,
    model: nn.Module | None = None,
) -> dict[str, Any]:
    """Fine-tune an encoder with a dropout/linear CLS head.

    ``max_steps`` caps optimizer updates after gradient accumulation and marks the
    saved run as not-full-training.
    """
    prepared = Path(prepared_path)
    rows = _load_prepared_rows(prepared)
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model_name must be a non-empty string")
    if not isinstance(revision, str) or not revision.strip():
        raise ValueError("revision must be a non-empty pinned revision")
    for name, value in (
        ("batch_size", batch_size),
        ("max_length", max_length),
        ("epochs", epochs),
        ("gradient_accumulation", gradient_accumulation),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if max_length > 512:
        raise ValueError("max_length must be in [1, 512]")
    if max_steps is not None and (
        isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1
    ):
        raise ValueError("max_steps must be a positive integer when provided")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    output = Path(output_dir)
    if output.exists() and not output.is_dir():
        raise ValueError("output_dir exists and is not a directory")
    resolved_device = resolve_device(device)

    from transformers import AutoModel, AutoTokenizer

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, revision=revision, cache_dir=cache_dir, use_fast=False
        )
    if model is None:
        model = AutoModel.from_pretrained(model_name, revision=revision, cache_dir=cache_dir)
    if not isinstance(model, nn.Module):
        raise ValueError("model must be a PyTorch encoder module")
    hidden_size, encoder_config, resolved_revision = _encoder_config(model)
    torch.manual_seed(seed)
    use_amp = resolved_device.type == "cuda"
    if use_amp:
        torch.cuda.manual_seed_all(seed)
    model.requires_grad_(True)
    classifier = _EncoderClassifier(model, hidden_size).to(resolved_device)
    tokenized = _tokenize_rows(tokenizer, rows, max_length, batch_size)
    labels = torch.tensor([row["label"] for row in rows], dtype=torch.long)
    split_indices = {
        split: torch.tensor(
            [index for index, row in enumerate(rows) if row["split"] == split], dtype=torch.long
        )
        for split in ("train", "val", "test")
    }
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=True) if use_amp else None
    generator = torch.Generator(device="cpu").manual_seed(seed)
    checkpoint_path = output / "checkpoint.pt"
    best_state: dict[str, Tensor] | None = None
    best_validation_loss = float("inf")
    best_epoch = 0
    global_step = 0
    history: list[dict[str, float | int]] = []
    epochs_completed = 0

    def autocast_context():
        return torch.autocast(device_type="cuda", dtype=torch.float16) if use_amp else nullcontext()

    for epoch in range(1, epochs + 1):
        classifier.train()
        permutation = torch.randperm(len(split_indices["train"]), generator=generator)
        shuffled = split_indices["train"][permutation]
        batches = [
            shuffled[start : start + batch_size] for start in range(0, len(shuffled), batch_size)
        ]
        epoch_loss_sum = 0.0
        epoch_example_count = 0
        for window_start in range(0, len(batches), gradient_accumulation):
            window = batches[window_start : window_start + gradient_accumulation]
            window_examples = sum(len(indices) for indices in window)
            optimizer.zero_grad(set_to_none=True)
            for row_indices in window:
                batch_inputs = _model_inputs(tokenized, row_indices, resolved_device)
                batch_labels = labels[row_indices].to(resolved_device)
                with autocast_context():
                    logits = classifier(batch_inputs)
                    _validate_logits(logits, len(row_indices), phase="training")
                    loss = F.cross_entropy(logits, batch_labels)
                if not torch.isfinite(loss):
                    raise FloatingPointError("training loss is non-finite")
                scaled_loss = loss * (len(row_indices) / window_examples)
                if scaler is not None:
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
                epoch_loss_sum += float(loss.detach().item()) * len(row_indices)
                epoch_example_count += len(row_indices)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            global_step += 1
            if max_steps is not None and global_step >= max_steps:
                break

        classifier.eval()
        val_indices = split_indices["val"]
        val_loss_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for start in range(0, len(val_indices), batch_size):
                row_indices = val_indices[start : start + batch_size]
                with autocast_context():
                    logits = classifier(_model_inputs(tokenized, row_indices, resolved_device))
                _validate_logits(logits, len(row_indices), phase="validation")
                loss = F.cross_entropy(logits, labels[row_indices].to(resolved_device))
                if not torch.isfinite(loss):
                    raise FloatingPointError("validation loss is non-finite")
                val_loss_sum += float(loss.item()) * len(row_indices)
                val_count += len(row_indices)
        val_loss = val_loss_sum / val_count
        train_loss = epoch_loss_sum / epoch_example_count
        epochs_completed = epoch
        history.append(
            {
                "epoch": epoch,
                "global_step": global_step,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
        )
        if val_loss < best_validation_loss:
            best_validation_loss = val_loss
            best_epoch = epoch
            best_state = {
                name: tensor.detach().to("cpu").clone()
                for name, tensor in classifier.state_dict().items()
            }
            output.mkdir(parents=True, exist_ok=True)
            architecture = {
                "encoder_class": f"{type(model).__module__}.{type(model).__qualname__}",
                "hidden_size": hidden_size,
                "dropout": 0.1,
                "classifier": "linear",
                "num_labels": 2,
                "pooling": "last_hidden_state[:,0,:]",
            }
            config_record = {
                "model_name": model_name,
                "requested_revision": revision,
                "resolved_revision": resolved_revision or revision,
                "encoder_config": encoder_config,
                "architecture": architecture,
                "batch_size": batch_size,
                "max_length": max_length,
                "epochs": epochs,
                "learning_rate": learning_rate,
                "gradient_accumulation": gradient_accumulation,
                "max_steps": max_steps,
                "max_steps_unit": "optimizer_update_steps" if max_steps is not None else None,
                "seed": seed,
                "device": str(resolved_device),
                "mixed_precision": use_amp,
                "full_training": max_steps is None,
                "not_full_training": max_steps is not None,
            }
            torch.save(
                {
                    "state_dict": best_state,
                    "architecture": architecture,
                    "config": config_record,
                    "model_name": model_name,
                    "requested_revision": revision,
                    "resolved_revision": resolved_revision or revision,
                    "best_epoch": best_epoch,
                    "best_validation_loss": best_validation_loss,
                    "parameter_count": parameter_count(classifier),
                    "history": history,
                },
                checkpoint_path,
            )
        if max_steps is not None and global_step >= max_steps:
            break

    if best_state is None:
        raise RuntimeError("fine-tuning did not produce a validation-selected checkpoint")
    classifier.load_state_dict(best_state)
    classifier.eval()
    probabilities = np.empty((len(rows), 2), dtype=np.float64)
    all_indices = torch.arange(len(rows), dtype=torch.long)
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            row_indices = all_indices[start : start + batch_size]
            with autocast_context():
                logits = classifier(_model_inputs(tokenized, row_indices, resolved_device))
            _validate_logits(logits, len(row_indices), phase="prediction")
            batch_probabilities = torch.softmax(logits.float(), dim=1)
            if not torch.isfinite(batch_probabilities).all():
                raise FloatingPointError("prediction probabilities contain non-finite values")
            probabilities[row_indices.numpy()] = batch_probabilities.to("cpu").numpy()
    predictions = probabilities.argmax(axis=1).astype(np.int64)
    y_array = labels.numpy()
    val_numpy = split_indices["val"].numpy()
    test_numpy = split_indices["test"].numpy()
    validation_metrics = classification_metrics(y_array[val_numpy], probabilities[val_numpy])
    metrics = classification_metrics(y_array[test_numpy], probabilities[test_numpy])
    parameter_total = parameter_count(classifier)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=(
                "id",
                "split",
                "gold_label",
                "predicted_label",
                "prob_factual",
                "prob_hallucinated",
            ),
        )
        writer.writeheader()
        for index, row in enumerate(rows):
            writer.writerow(
                {
                    "id": row["record_id"],
                    "split": row["split"],
                    "gold_label": row["label"],
                    "predicted_label": int(predictions[index]),
                    "prob_factual": float(probabilities[index, 0]),
                    "prob_hallucinated": float(probabilities[index, 1]),
                }
            )
    output_config = {
        "model_name": model_name,
        "requested_revision": revision,
        "resolved_revision": resolved_revision or revision,
        "device": str(resolved_device),
        "batch_size": batch_size,
        "max_length": max_length,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "gradient_accumulation": gradient_accumulation,
        "max_steps": max_steps,
        "max_steps_unit": "optimizer_update_steps" if max_steps is not None else None,
        "seed": seed,
        "full_training": max_steps is None,
        "not_full_training": max_steps is not None,
        "epochs_completed": epochs_completed,
        "global_step": global_step,
    }
    (output / "history.json").write_text(
        json.dumps(history, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (output / "metrics.json").write_text(
        json.dumps(
            {
                "metrics": metrics,
                "validation_metrics": validation_metrics,
                "best_epoch": best_epoch,
                "best_validation_loss": best_validation_loss,
                "parameter_count": parameter_total,
                "config": output_config,
            },
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "probabilities": probabilities,
        "predictions": predictions,
        "metrics": metrics,
        "validation_metrics": validation_metrics,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "parameter_count": parameter_total,
        "checkpoint_path": str(checkpoint_path),
        "history": history,
        "global_step": global_step,
        "max_steps_unit": "optimizer_update_steps" if max_steps is not None else None,
        "not_full_training": max_steps is not None,
    }
