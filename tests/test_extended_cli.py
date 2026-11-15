from __future__ import annotations

import json
from pathlib import Path


def test_benchmark_and_report_commands_dispatch_real_service_functions(
    monkeypatch, tmp_path: Path
) -> None:
    from graphhall import cli

    calls = []
    monkeypatch.setattr(
        cli,
        "run_benchmark",
        lambda config, *, resume=False: (
            calls.append(("benchmark", config, resume))
            or {"name": "qa-full", "completed_count": 2, "pending_count": 0, "output_dir": tmp_path}
        ),
        raising=False,
    )
    monkeypatch.setattr(
        cli,
        "generate_report",
        lambda input_dir, output_dir: (
            calls.append(("report", input_dir, output_dir))
            or {
                "completed_count": 2,
                "pending_count": 1,
                "csv_path": tmp_path / "results.csv",
                "markdown_path": tmp_path / "results.md",
            }
        ),
        raising=False,
    )

    assert cli.main(["benchmark", "--config", "configs/qa-full.yaml", "--resume"]) == 0
    assert (
        cli.main(["report", "--input-dir", "results/qa-full", "--output-dir", str(tmp_path)]) == 0
    )
    assert calls == [
        ("benchmark", "configs/qa-full.yaml", True),
        ("report", "results/qa-full", str(tmp_path)),
    ]


def test_predict_and_finetune_commands_dispatch_actual_apis(monkeypatch, tmp_path: Path) -> None:
    from graphhall import cli

    calls = []
    monkeypatch.setattr(
        cli,
        "replay_predictions",
        lambda run_dir, *, learner=None: (
            calls.append(("predict", run_dir, learner))
            or {
                "verified": True,
                "experiment_id": "gcn-seed42-k2",
                "row_count": 30,
                "max_absolute_error": 0.0,
            }
        ),
        raising=False,
    )
    monkeypatch.setattr(
        cli,
        "finetune_classifier",
        lambda input_path, output_dir, **kwargs: (
            calls.append(("finetune", input_path, output_dir, kwargs))
            or {
                "best_epoch": 2,
                "not_full_training": False,
                "checkpoint_path": tmp_path / "ft/checkpoint.pt",
            }
        ),
        raising=False,
    )

    assert (
        cli.main(["predict", "--run-dir", "results/qa-full/runs/gcn-seed42-k2", "--learner", "gcn"])
        == 0
    )
    assert (
        cli.main(
            [
                "finetune",
                "--input",
                "data/processed/halueval-qa-full.jsonl",
                "--output-dir",
                str(tmp_path / "ft"),
                "--model-name",
                "microsoft/deberta-v3-base",
                "--revision",
                "immutable-revision",
                "--device",
                "cpu",
                "--batch-size",
                "2",
                "--max-length",
                "512",
                "--epochs",
                "2",
                "--learning-rate",
                "0.00002",
                "--seed",
                "43",
                "--gradient-accumulation",
                "4",
                "--max-steps",
                "1",
            ]
        )
        == 0
    )
    assert calls[0] == ("predict", "results/qa-full/runs/gcn-seed42-k2", "gcn")
    assert calls[1][0:3] == (
        "finetune",
        "data/processed/halueval-qa-full.jsonl",
        str(tmp_path / "ft"),
    )
    assert calls[1][3] == {
        "model_name": "microsoft/deberta-v3-base",
        "revision": "immutable-revision",
        "device": "cpu",
        "batch_size": 2,
        "max_length": 512,
        "epochs": 2,
        "learning_rate": 0.00002,
        "seed": 43,
        "gradient_accumulation": 4,
        "max_steps": 1,
        "cache_dir": None,
    }


def test_predict_accepts_finetuned_deberta_checkpoint_replay(monkeypatch) -> None:
    from graphhall import cli

    calls = []
    monkeypatch.setattr(
        cli,
        "replay_predictions",
        lambda run_dir, *, learner=None: (
            calls.append((run_dir, learner))
            or {
                "verified": True,
                "experiment_id": "finetuned-deberta-seed42",
                "row_count": 30,
                "max_absolute_error": 0.0,
            }
        ),
    )

    assert (
        cli.main(
            [
                "predict",
                "--run-dir",
                "results/qa-full/runs/finetuned-deberta-seed42",
                "--learner",
                "finetuned_deberta",
            ]
        )
        == 0
    )
    assert calls == [("results/qa-full/runs/finetuned-deberta-seed42", "finetuned_deberta")]


def test_run_command_writes_probabilities_and_accepts_device_profile(tmp_path: Path) -> None:
    import numpy as np

    from graphhall.cli import main
    from graphhall.data import split_candidate_records

    raw = [
        {
            "question": f"question {index}",
            "knowledge": f"knowledge {index}",
            "right_answer": f"factual {index}",
            "hallucinated_answer": f"hallucinated {index}",
        }
        for index in range(30)
    ]
    rows = split_candidate_records(raw, max_records=30, seed=42)
    prepared = tmp_path / "prepared.jsonl"
    prepared.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    embeddings = tmp_path / "features.npz"
    np.savez_compressed(
        embeddings,
        embeddings=np.random.default_rng(27).normal(size=(30, 8)).astype(np.float32),
        record_ids=np.asarray([row["record_id"] for row in rows]),
    )
    out = tmp_path / "cli-run"

    assert (
        main(
            [
                "run",
                "--input",
                str(prepared),
                "--embeddings",
                str(embeddings),
                "--output-dir",
                str(out),
                "--learners",
                "linear",
                "--device",
                "cpu",
                "--model-profile",
                "weighted",
                "--max-epochs",
                "2",
                "--validation-interval",
                "1",
            ]
        )
        == 0
    )
    with (out / "probabilities.csv").open(encoding="utf-8") as stream:
        content = stream.read()
    assert "probability_0,probability_1" in content
    assert content.count("\n") == 31
