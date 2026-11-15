from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch

from graphhall.runtime import resolve_device


def encode_text_pairs(
    contexts: Sequence[str],
    answers: Sequence[str],
    tokenizer: Any,
    model: Any,
    *,
    batch_size: int,
    max_length: int,
    device: str | torch.device = "cpu",
    mixed_precision: bool = False,
) -> np.ndarray:
    """Encode text pairs as frozen last-layer [CLS] vectors using bounded batches."""
    if len(contexts) != len(answers) or not contexts:
        raise ValueError("contexts and answers must have the same nonzero length")
    if batch_size < 1 or not 1 <= max_length <= 512:
        raise ValueError("batch_size must be positive and max_length must be in [1, 512]")
    resolved_device = resolve_device(device)
    model.to(resolved_device)
    model.eval()
    model.requires_grad_(False)
    batches: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(contexts), batch_size):
            stop = min(start + batch_size, len(contexts))
            encoded = tokenizer(
                list(contexts[start:stop]),
                list(answers[start:stop]),
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            if isinstance(encoded, Mapping):
                encoded = {
                    key: value.to(resolved_device) if isinstance(value, torch.Tensor) else value
                    for key, value in encoded.items()
                }
            elif callable(getattr(encoded, "to", None)):
                encoded = encoded.to(resolved_device)
            autocast = (
                torch.autocast(device_type=resolved_device.type)
                if mixed_precision
                else nullcontext()
            )
            with autocast:
                output = model(**encoded)
            cls_vectors = output.last_hidden_state[:, 0, :]
            batches.append(cls_vectors.detach().to(device="cpu", dtype=torch.float32).numpy())
    return np.concatenate(batches, axis=0)


def _model_state_sha256(model: Any) -> str | None:
    state_dict = getattr(model, "state_dict", None)
    if not callable(state_dict):
        return None
    state = state_dict()
    if not isinstance(state, Mapping):
        return None
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().to("cpu").contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def encode_prepared(
    prepared_path: str | Path,
    output_path: str | Path,
    *,
    model_name: str,
    revision: str,
    batch_size: int,
    max_length: int,
    include_knowledge: bool = False,
    cache_dir: str | Path | None = None,
    tokenizer: Any | None = None,
    model: Any | None = None,
    device: str | torch.device = "cpu",
    mixed_precision: bool = False,
) -> dict[str, Any]:
    """Encode prepared QA rows and persist features plus model/source provenance."""
    from transformers import AutoModel, AutoTokenizer

    if batch_size < 1 or not 1 <= max_length <= 512:
        raise ValueError("batch_size must be positive and max_length must be in [1, 512]")
    resolved_device = resolve_device(device)
    source = Path(prepared_path)
    if not source.is_file():
        raise ValueError(f"prepared data file does not exist: {source}")
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError("prepared data file must contain at least one row")
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, revision=revision, cache_dir=cache_dir, use_fast=False
        )
    if model is None:
        model = AutoModel.from_pretrained(model_name, revision=revision, cache_dir=cache_dir)
    contexts = [
        f"{row['question']} Knowledge: {row['knowledge']}" if include_knowledge else row["question"]
        for row in rows
    ]
    answers = [row["answer"] for row in rows]
    embeddings = encode_text_pairs(
        contexts,
        answers,
        tokenizer,
        model,
        batch_size=batch_size,
        max_length=max_length,
        device=resolved_device,
        mixed_precision=mixed_precision,
    )
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        embeddings=embeddings,
        record_ids=np.asarray([row["record_id"] for row in rows]),
    )
    commit = getattr(getattr(model, "config", None), "_commit_hash", None)
    config = {
        "model": model_name,
        "requested_revision": revision,
        "resolved_revision": commit or revision,
        "model_state_sha256": _model_state_sha256(model),
        "prepared_jsonl_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "row_count": len(rows),
        "hidden_size": int(embeddings.shape[1]),
        "max_length": max_length,
        "batch_size": batch_size,
        "include_knowledge": include_knowledge,
        "context_policy": "question_plus_knowledge_deviation"
        if include_knowledge
        else "question_only",
        "pooling": "last_hidden_state[:,0,:]",
        "encoder_frozen": True,
        "device": str(resolved_device),
        "mixed_precision": mixed_precision,
    }
    config["config_sha256"] = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode("utf-8")
    ).hexdigest()
    target.with_suffix(".meta.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return config
