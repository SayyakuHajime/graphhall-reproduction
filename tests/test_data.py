from graphhall.data import split_candidate_records


def _records(count: int = 30) -> list[dict[str, str]]:
    return [
        {
            "question": f"q-{index}",
            "knowledge": f"context-{index}",
            "right_answer": f"right-{index}",
            "hallucinated_answer": f"wrong-{index}",
        }
        for index in range(count)
    ]


def test_prepare_splits_original_rows_then_selects_one_balanced_candidate() -> None:
    prepared = split_candidate_records(_records(), max_records=30, seed=42)

    assert len(prepared) == 30
    ids = [row["record_id"] for row in prepared]
    assert len(set(ids)) == 30
    assert {row["split"] for row in prepared} == {"train", "val", "test"}
    for split in ("train", "val", "test"):
        rows = [row for row in prepared if row["split"] == split]
        assert len({row["record_id"] for row in rows}) == len(rows)
        assert abs(sum(row["label"] for row in rows) - len(rows) / 2) <= 0.5
        for row in rows:
            expected = "wrong-" if row["label"] == 1 else "right-"
            assert row["answer"].startswith(expected)


def test_prepare_is_seed_deterministic_and_rejects_too_small_samples() -> None:
    first = split_candidate_records(_records(), max_records=30, seed=42)
    second = split_candidate_records(_records(), max_records=30, seed=42)
    assert first == second

    try:
        split_candidate_records(_records(), max_records=8, seed=42)
    except ValueError as exc:
        assert "at least 10" in str(exc)
    else:
        raise AssertionError("fewer than ten original rows must be rejected")
