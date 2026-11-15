from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

_SPLITS = ("train", "val", "test")


def split_candidate_records(
    records: Sequence[dict[str, Any]], *, max_records: int | None, seed: int
) -> list[dict[str, Any]]:
    """Split original QA rows, then choose exactly one answer from each pair."""
    if max_records is not None and max_records < 10:
        raise ValueError("at least 10 original rows are required for train/val/test")
    count = len(records) if max_records is None else min(len(records), max_records)
    if count < 10:
        raise ValueError("at least 10 original rows are required for train/val/test")

    rng = np.random.default_rng(seed)
    selected = rng.permutation(len(records))[:count].tolist()
    train_count = int(count * 0.8)
    val_count = (count - train_count) // 2
    original_splits = {
        "train": selected[:train_count],
        "val": selected[train_count : train_count + val_count],
        "test": selected[train_count + val_count :],
    }

    prepared: list[dict[str, Any]] = []
    for split_index, split_name in enumerate(_SPLITS):
        split_rng = np.random.default_rng(seed + 1_000_003 * (split_index + 1))
        row_ids = split_rng.permutation(original_splits[split_name]).tolist()
        start_label = int(split_rng.integers(0, 2)) if len(row_ids) % 2 else 0
        for position, original_index in enumerate(row_ids):
            raw = records[original_index]
            for key in ("question", "knowledge", "right_answer", "hallucinated_answer"):
                if not isinstance(raw.get(key), str):
                    raise ValueError(f"record {original_index} is missing string field {key!r}")
            label = (position + start_label) % 2
            prepared.append(
                {
                    "record_id": f"qa-{original_index:06d}",
                    "split": split_name,
                    "question": raw["question"],
                    "knowledge": raw["knowledge"],
                    "answer": raw["hallucinated_answer"] if label else raw["right_answer"],
                    "label": label,
                }
            )
    return prepared
