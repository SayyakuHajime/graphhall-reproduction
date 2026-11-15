import csv
import json
from pathlib import Path

import numpy as np
import pytest


def test_run_cli_trains_all_models_and_writes_metrics_predictions_and_checkpoints(
    tmp_path: Path,
) -> None:
    source = tmp_path / "qa.jsonl"
    raw = [
        {
            "question": f"question {index}",
            "knowledge": f"knowledge {index}",
            "right_answer": f"correct {index}",
            "hallucinated_answer": f"incorrect {index}",
        }
        for index in range(30)
    ]
    source.write_text("\n".join(json.dumps(row) for row in raw) + "\n", encoding="utf-8")
    prepared = tmp_path / "prepared.jsonl"
    from graphhall.cli import main

    main(
        ["prepare", "--source-file", str(source), "--output", str(prepared), "--max-records", "30"]
    )
    rows = [json.loads(line) for line in prepared.read_text(encoding="utf-8").splitlines()]
    embeddings_path = tmp_path / "features.npz"
    np.savez_compressed(
        embeddings_path,
        embeddings=np.random.default_rng(17).normal(size=(30, 12)).astype(np.float32),
        record_ids=np.asarray([row["record_id"] for row in rows]),
    )
    output_dir = tmp_path / "run"

    assert (
        main(
            [
                "run",
                "--input",
                str(prepared),
                "--embeddings",
                str(embeddings_path),
                "--output-dir",
                str(output_dir),
                "--learners",
                "gcn,gat,gt",
                "--k",
                "3",
                "--threshold",
                "-1",
                "--max-epochs",
                "2",
                "--validation-interval",
                "1",
                "--patience",
                "2",
            ]
        )
        == 0
    )

    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    predictions = list(csv.DictReader((output_dir / "predictions.csv").open(encoding="utf-8")))
    assert metrics["run_type"] == "research"
    assert {row["learner"] for row in metrics["results"]} == {"gcn", "gat", "gt"}
    assert len(predictions) == 30 * 3
    assert all(
        (output_dir / "checkpoints" / f"{name}-k3.pt").exists() for name in ("gcn", "gat", "gt")
    )
    for name in ("config.json", "provenance.json"):
        assert (output_dir / name).exists()


def test_encode_cli_dispatches_pinned_model_options(monkeypatch, tmp_path: Path) -> None:
    from graphhall import cli

    prepared = tmp_path / "prepared.jsonl"
    prepared.write_text(json.dumps({}), encoding="utf-8")
    output = tmp_path / "features.npz"
    calls = {}

    def fake_encode_prepared(prepared_path, output_path, **kwargs):
        calls.update(prepared_path=prepared_path, output_path=output_path, **kwargs)
        np.savez_compressed(
            output_path,
            embeddings=np.ones((1, 2), dtype=np.float32),
            record_ids=np.asarray(["qa-000001"]),
        )
        metadata = {
            "resolved_revision": kwargs["revision"],
            "row_count": 1,
            "hidden_size": 2,
        }
        Path(output_path).with_suffix(".meta.json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )
        return metadata

    monkeypatch.setattr(cli, "encode_prepared", fake_encode_prepared, raising=False)

    assert (
        cli.main(
            [
                "encode",
                "--input",
                str(prepared),
                "--output",
                str(output),
                "--revision",
                "test-revision",
                "--batch-size",
                "4",
                "--max-length",
                "96",
                "--include-knowledge",
            ]
        )
        == 0
    )

    assert calls["prepared_path"] == str(prepared)
    assert calls["output_path"] == str(output)
    assert calls["revision"] == "test-revision"
    assert calls["batch_size"] == 4
    assert calls["max_length"] == 96
    assert calls["include_knowledge"] is True
    assert output.exists()
    assert output.with_suffix(".meta.json").exists()


def test_smoke_cli_runs_prepare_encode_and_all_learners(monkeypatch, tmp_path: Path) -> None:
    import hashlib

    from graphhall import cli

    source = tmp_path / "source.json"
    raw = [
        {
            "question": f"question {index}",
            "knowledge": f"knowledge {index}",
            "right_answer": f"right {index}",
            "hallucinated_answer": f"hallucinated {index}",
        }
        for index in range(30)
    ]
    source.write_text(json.dumps(raw), encoding="utf-8")
    captured = {}

    def fake_encode_prepared(prepared_path, output_path, **kwargs):
        prepared_file = Path(prepared_path)
        rows = [json.loads(line) for line in prepared_file.read_text(encoding="utf-8").splitlines()]
        embeddings = (np.random.default_rng(9).random(size=(len(rows), 8)) + 0.1).astype(np.float32)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            embeddings=embeddings,
            record_ids=np.asarray([row["record_id"] for row in rows]),
        )
        metadata = {
            "model": kwargs["model_name"],
            "resolved_revision": kwargs["revision"],
            "prepared_jsonl_sha256": hashlib.sha256(prepared_file.read_bytes()).hexdigest(),
            "row_count": len(rows),
            "hidden_size": 8,
        }
        Path(output_path).with_suffix(".meta.json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )
        captured.update(kwargs)
        return metadata

    monkeypatch.setattr(cli, "encode_prepared", fake_encode_prepared, raising=False)
    output_dir = tmp_path / "smoke"

    assert (
        cli.main(
            [
                "smoke",
                "--source-file",
                str(source),
                "--output-dir",
                str(output_dir),
                "--max-records",
                "30",
                "--max-epochs",
                "2",
                "--validation-interval",
                "1",
                "--threshold",
                "-1",
            ]
        )
        == 0
    )

    run_dir = output_dir / "run"
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    predictions = list(csv.DictReader((run_dir / "predictions.csv").open(encoding="utf-8")))
    assert metrics["run_type"] == "smoke"
    assert metrics["sample_count"] == 30
    assert metrics["test_count"] == 3
    assert {row["learner"] for row in metrics["results"]} == {"gcn", "gat", "gt"}
    assert len(predictions) == 90
    assert captured["batch_size"] == 4
    assert captured["max_length"] == 96
    assert config["max_epochs"] == 2
    assert (run_dir / "histories.json").exists()
    assert all(
        (run_dir / "checkpoints" / f"{learner}-k5.pt").exists() for learner in ("gcn", "gat", "gt")
    )


def test_smoke_cli_rejects_benchmark_sized_limits(tmp_path: Path) -> None:
    from graphhall.cli import main

    with pytest.raises(ValueError, match="max_epochs must be in \\[1, 20\\]"):
        main(["smoke", "--max-epochs", "21"])


def test_run_rejects_embeddings_from_a_different_prepared_file(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    raw = [
        {
            "question": f"question {index}",
            "knowledge": f"knowledge {index}",
            "right_answer": f"right {index}",
            "hallucinated_answer": f"hallucinated {index}",
        }
        for index in range(30)
    ]
    source.write_text(json.dumps(raw), encoding="utf-8")
    prepared = tmp_path / "prepared.jsonl"
    from graphhall.cli import main

    main(
        ["prepare", "--source-file", str(source), "--output", str(prepared), "--max-records", "30"]
    )
    rows = [json.loads(line) for line in prepared.read_text(encoding="utf-8").splitlines()]
    embeddings = tmp_path / "features.npz"
    np.savez_compressed(
        embeddings,
        embeddings=np.ones((30, 8), dtype=np.float32),
        record_ids=np.asarray([row["record_id"] for row in rows]),
    )
    embeddings.with_suffix(".meta.json").write_text(
        json.dumps({"prepared_jsonl_sha256": "not-the-input-hash"}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="prepared JSONL hash"):
        main(
            [
                "run",
                "--input",
                str(prepared),
                "--embeddings",
                str(embeddings),
                "--output-dir",
                str(tmp_path / "run"),
            ]
        )
