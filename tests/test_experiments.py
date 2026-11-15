from __future__ import annotations

from pathlib import Path

import yaml


def _write_config(root: Path, **training_overrides) -> Path:
    payload = {
        "schema_version": 1,
        "name": "tiny-matrix",
        "run_type": "integration",
        "paths": {
            "prepared": "data/prepared.jsonl",
            "embeddings": "data/features.npz",
            "output_dir": "results/tiny-matrix",
        },
        "dataset": {
            "name": "HaluEval-QA",
            "source_revision": "fixture-revision",
            "source_sha256": None,
            "source_rows": 30,
            "selected_records": 30,
            "split_seed": 42,
            "split_counts": {"train": 24, "val": 3, "test": 3},
        },
        "encoder": {
            "model": "fixture/model",
            "revision": "fixture-revision",
            "device": "cpu",
            "frozen": True,
            "context_policy": "question_only",
            "include_knowledge": False,
            "max_length": 32,
            "batch_size": 4,
            "pooling": "last_hidden_state[:,0,:]",
            "hidden_size": 8,
        },
        "graph": {"k_values": [2, 3], "cosine_threshold": -1.0, "lpe_dimensions": 4},
        "training": {
            "learners": ["gcn", "gat", "gt"],
            "seeds": [42, 43],
            "learning_rate": 0.005,
            "weight_decay": 0.0005,
            "max_epochs": 2,
            "validation_interval": 1,
            "patience": 2,
            "threads": 1,
            "model_profile": "weighted",
            "selection_metric": "f1",
            **training_overrides,
        },
        "baselines": ["linear", "knn"],
        "finetuned_deberta": {
            "enabled": False,
            "model_name": "fixture/model",
            "revision": "fixture-revision",
            "batch_size": 1,
            "max_length": 32,
            "epochs": 1,
            "learning_rate": 0.00002,
            "seed_from_matrix": True,
            "gradient_accumulation": 1,
            "max_steps": None,
        },
        "evaluation": {
            "positive_label": 1,
            "protocol": "transductive_full_graph",
            "test_labels_used_for_training": False,
        },
    }
    config_path = root / "configs" / "tiny.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return config_path


def test_matrix_expands_graph_baseline_and_seed_combinations(tmp_path: Path) -> None:
    from graphhall.config import load_config
    from graphhall.experiments import build_experiment_matrix

    config = load_config(_write_config(tmp_path), project_root=tmp_path)
    jobs = build_experiment_matrix(config)

    graph_jobs = [job for job in jobs if job.learner in {"gcn", "gat", "gt"}]
    linear_jobs = [job for job in jobs if job.learner == "linear"]
    knn_jobs = [job for job in jobs if job.learner == "knn"]
    assert len(jobs) == 18
    assert len(graph_jobs) == 12
    assert len(linear_jobs) == 2
    assert len(knn_jobs) == 4
    assert {(job.learner, job.seed, job.k) for job in graph_jobs} == {
        (learner, seed, k) for learner in ("gcn", "gat", "gt") for seed in (42, 43) for k in (2, 3)
    }
    assert {(job.seed, job.k) for job in linear_jobs} == {(42, None), (43, None)}
    assert {(job.seed, job.k) for job in knn_jobs} == {
        (seed, k) for seed in (42, 43) for k in (2, 3)
    }
    assert len({job.experiment_id for job in jobs}) == len(jobs)


def _materialize_tiny_inputs(root: Path, config_path: Path) -> None:
    import hashlib
    import json

    import numpy as np

    from graphhall.data import split_candidate_records

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["graph"]["k_values"] = [2]
    payload["training"]["seeds"] = [42]
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    raw = [
        {
            "question": f"question {index}",
            "knowledge": f"knowledge {index}",
            "right_answer": f"factual answer {index}",
            "hallucinated_answer": f"hallucinated answer {index}",
        }
        for index in range(30)
    ]
    rows = split_candidate_records(raw, max_records=30, seed=42)
    prepared_path = root / "data/prepared.jsonl"
    prepared_path.parent.mkdir(parents=True, exist_ok=True)
    prepared_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    prepared_hash = hashlib.sha256(prepared_path.read_bytes()).hexdigest()
    (root / "data/prepared.meta.json").write_text(
        json.dumps(
            {
                "source_sha256": None,
                "source_revision": "fixture-revision",
                "source_row_count": 30,
                "original_records": 30,
                "seed": 42,
                "split_counts": {"train": 24, "val": 3, "test": 3},
                "label_mapping": {"right_answer": 0, "hallucinated_answer": 1},
            }
        ),
        encoding="utf-8",
    )
    embeddings_path = root / "data/features.npz"
    features = np.random.default_rng(17).normal(size=(30, 8)).astype(np.float32)
    np.savez_compressed(
        embeddings_path,
        embeddings=features,
        record_ids=np.asarray([row["record_id"] for row in rows]),
    )
    (root / "data/features.meta.json").write_text(
        json.dumps(
            {
                "model": "fixture/model",
                "resolved_revision": "fixture-revision",
                "model_state_sha256": "c" * 64,
                "prepared_jsonl_sha256": prepared_hash,
                "row_count": 30,
                "hidden_size": 8,
                "max_length": 32,
                "batch_size": 4,
                "include_knowledge": False,
                "context_policy": "question_only",
                "pooling": "last_hidden_state[:,0,:]",
                "encoder_frozen": True,
            }
        ),
        encoding="utf-8",
    )


def test_source_file_reconstructs_missing_prepared_sidecar_without_reencoding(
    tmp_path: Path,
) -> None:
    import hashlib
    import json

    import yaml

    from graphhall.config import load_config
    from graphhall.experiments import _load_inputs

    config_path = _write_config(tmp_path)
    _materialize_tiny_inputs(tmp_path, config_path)
    raw = [
        {
            "question": f"question {index}",
            "knowledge": f"knowledge {index}",
            "right_answer": f"factual answer {index}",
            "hallucinated_answer": f"hallucinated answer {index}",
        }
        for index in range(30)
    ]
    source_path = tmp_path / "data/raw.jsonl"
    source_path.write_text(json.dumps(raw), encoding="utf-8")
    prepared_path = tmp_path / "data/shared.jsonl"
    embeddings_path = tmp_path / "data/shared.npz"
    (tmp_path / "data/prepared.jsonl").rename(prepared_path)
    (tmp_path / "data/features.npz").rename(embeddings_path)
    (tmp_path / "data/prepared.meta.json").unlink()
    (tmp_path / "data/features.meta.json").rename(tmp_path / "data/shared.meta.json")

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["paths"].update(
        {
            "prepared": "data/shared.jsonl",
            "embeddings": "data/shared.npz",
            "source": "data/raw.jsonl",
        }
    )
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    config = load_config(config_path, project_root=tmp_path, require_inputs=True)
    inputs = _load_inputs(
        config.paths.prepared,
        config.paths.embeddings,
        config.dataset,
        config.encoder,
        source_path=config.paths.source,
    )

    assert len(inputs.rows) == 30
    assert (
        inputs.hashes["source_file_sha256"] == hashlib.sha256(source_path.read_bytes()).hexdigest()
    )
    assert inputs.prepared_metadata["reconstructed_from_source"] is True


def test_finetuned_checkpoint_replays_on_the_same_local_pool(tmp_path: Path) -> None:
    import torch
    import yaml
    from transformers import BertConfig, BertModel, BertTokenizer

    from graphhall.experiments import replay_predictions, run_benchmark

    config_path = _write_config(tmp_path)
    _materialize_tiny_inputs(tmp_path, config_path)
    model_dir = tmp_path / "local-bert"
    model_dir.mkdir()
    vocab_path = model_dir / "vocab.txt"
    vocab_path.write_text(
        "[PAD]\\n[UNK]\\n[CLS]\\n[SEP]\\n[MASK]\\nquestion\\nanswer\\n"
        "factual\\nhallucinated\\n0\\n1\\n",
        encoding="utf-8",
    )
    tokenizer = BertTokenizer(vocab_file=str(vocab_path))
    tokenizer.save_pretrained(model_dir)
    model = BertModel(
        BertConfig(
            vocab_size=len(tokenizer),
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
            max_position_embeddings=32,
        )
    )
    model.save_pretrained(model_dir)

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["finetuned_deberta"].update(
        {
            "enabled": True,
            "model_name": str(model_dir),
            "revision": "local-revision",
            "batch_size": 4,
            "max_length": 16,
            "epochs": 1,
            "gradient_accumulation": 1,
        }
    )
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    result = run_benchmark(config_path, project_root=tmp_path)
    run_dir = tmp_path / "results/tiny-matrix/runs/finetuned-deberta-seed42"
    checkpoint = torch.load(
        run_dir / "finetuned-model/checkpoint.pt", map_location="cpu", weights_only=True
    )
    assert checkpoint["config"]["full_training"] is True
    replayed = replay_predictions(run_dir, learner="finetuned_deberta")

    assert result["completed_count"] == 6
    assert replayed["verified"] is True
    assert replayed["transductive"] is True
    assert replayed["max_absolute_error"] <= 1e-6


def test_embedding_metadata_rejects_malformed_model_state_hash(tmp_path: Path) -> None:
    import json

    from graphhall.config import load_config
    from graphhall.experiments import _load_inputs

    config_path = _write_config(tmp_path)
    _materialize_tiny_inputs(tmp_path, config_path)
    metadata_path = tmp_path / "data/features.meta.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["model_state_sha256"] = "z" * 64
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    config = load_config(config_path, project_root=tmp_path, require_inputs=True)

    try:
        _load_inputs(config.paths.prepared, config.paths.embeddings, config.dataset, config.encoder)
    except ValueError as error:
        assert "model_state_sha256" in str(error)
    else:
        raise AssertionError("model state identity must be a valid hexadecimal SHA-256")


def test_embedding_metadata_must_match_pooling_config(tmp_path: Path) -> None:
    import json

    from graphhall.config import load_config
    from graphhall.experiments import _load_inputs

    config_path = _write_config(tmp_path)
    _materialize_tiny_inputs(tmp_path, config_path)
    metadata_path = tmp_path / "data/features.meta.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["pooling"] = "mean"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    config = load_config(config_path, project_root=tmp_path, require_inputs=True)

    try:
        _load_inputs(config.paths.prepared, config.paths.embeddings, config.dataset, config.encoder)
    except ValueError as error:
        assert "pooling" in str(error).lower()
    else:
        raise AssertionError("embeddings with the wrong pooling rule must be refused")


def test_prepared_metadata_must_match_configured_source_and_split(tmp_path: Path) -> None:
    import json

    from graphhall.config import load_config
    from graphhall.experiments import _load_inputs

    config_path = _write_config(tmp_path)
    _materialize_tiny_inputs(tmp_path, config_path)
    metadata_path = tmp_path / "data/prepared.meta.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["seed"] = 43
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    config = load_config(config_path, project_root=tmp_path, require_inputs=True)

    try:
        _load_inputs(config.paths.prepared, config.paths.embeddings, config.dataset, config.encoder)
    except ValueError as error:
        assert "split seed" in str(error).lower()
    else:
        raise AssertionError("benchmark inputs with the wrong prepared split seed must be refused")


def test_changed_implementation_is_rejected_by_resume_and_replay(tmp_path, monkeypatch) -> None:
    import pytest

    from graphhall import experiments

    config_path = _write_config(tmp_path)
    _materialize_tiny_inputs(tmp_path, config_path)
    monkeypatch.setattr(experiments, "_implementation_sha256", lambda: "a" * 64, raising=False)
    experiments.run_benchmark(config_path, project_root=tmp_path)
    monkeypatch.setattr(experiments, "_implementation_sha256", lambda: "b" * 64)

    with pytest.raises(ValueError, match="stale|provenance"):
        experiments.run_benchmark(config_path, resume=True, project_root=tmp_path)
    with pytest.raises(ValueError, match="provenance"):
        experiments.replay_predictions(tmp_path / "results/tiny-matrix/runs/gcn-seed42-k2")


def test_benchmark_runs_all_tiny_learners_and_resume_verifies_artifacts(tmp_path: Path) -> None:
    import csv
    import json

    from graphhall.experiments import run_benchmark

    config_path = _write_config(tmp_path)
    _materialize_tiny_inputs(tmp_path, config_path)

    result = run_benchmark(config_path, project_root=tmp_path)
    assert result["completed_count"] == 5
    runs_dir = tmp_path / "results/tiny-matrix/runs"
    run_ids = {path.name for path in runs_dir.iterdir() if path.is_dir()}
    assert run_ids == {
        "gcn-seed42-k2",
        "gat-seed42-k2",
        "gt-seed42-k2",
        "linear-seed42",
        "knn-seed42-k2",
    }
    for run_dir in runs_dir.iterdir():
        if not run_dir.is_dir():
            continue
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        assert len(metrics["test_metrics"]) == 9
        with (run_dir / "predictions.csv").open(encoding="utf-8", newline="") as stream:
            predictions = list(csv.DictReader(stream))
        assert len(predictions) == 30
        assert {row["probability_0"] for row in predictions}
        assert {row["probability_1"] for row in predictions}
        assert (run_dir / "completion.json").is_file()
        assert (run_dir / "best.pt").is_file() is (metrics["learner"] != "knn")

    from graphhall.reporting import generate_report

    report = generate_report(runs_dir.parent, tmp_path / "reports/tiny-matrix")
    assert report["completed_count"] == 5
    assert report["pending_count"] == 0
    assert report["selected_count"] == 1
    with report["csv_path"].open(encoding="utf-8", newline="") as stream:
        report_rows = list(csv.DictReader(stream))
    assert sum(row["row_type"] == "per_seed" for row in report_rows) == 5
    assert sum(row["row_type"] == "aggregate" for row in report_rows) == 5
    markdown = report["markdown_path"].read_text(encoding="utf-8")
    assert "Fidelity caveats" in markdown
    assert "test score is not used for selection" in markdown.lower()

    gcn_manifest_path = runs_dir / "gcn-seed42-k2/completion.json"
    before = gcn_manifest_path.read_bytes()
    resumed = run_benchmark(config_path, resume=True, project_root=tmp_path)
    assert resumed["completed_count"] == 5
    assert gcn_manifest_path.read_bytes() == before

    from graphhall.experiments import replay_predictions

    replayed = replay_predictions(runs_dir / "gcn-seed42-k2", learner="gcn")
    assert replayed["verified"] is True
    assert replayed["transductive"] is True
    assert replayed["max_absolute_error"] <= 1e-6
    knn_replayed = replay_predictions(runs_dir / "knn-seed42-k2", learner="knn")
    assert knn_replayed["verified"] is True

    checkpoint = runs_dir / "gcn-seed42-k2/best.pt"
    checkpoint.write_bytes(checkpoint.read_bytes() + b"corruption")
    try:
        run_benchmark(config_path, resume=True, project_root=tmp_path)
    except ValueError as error:
        assert "artifact hash mismatch" in str(error)
    else:
        raise AssertionError("resume must reject a corrupted checkpoint")

    changed_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    changed_config["training"]["learning_rate"] = 0.004
    config_path.write_text(yaml.safe_dump(changed_config), encoding="utf-8")
    try:
        run_benchmark(config_path, resume=True, project_root=tmp_path)
    except ValueError as error:
        assert "refusing stale results" in str(error)
    else:
        raise AssertionError("resume must refuse a changed effective config")
