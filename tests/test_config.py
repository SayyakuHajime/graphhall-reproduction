from __future__ import annotations

from pathlib import Path

import yaml


def _valid_config() -> dict:
    return {
        "schema_version": 1,
        "name": "tiny",
        "run_type": "integration",
        "paths": {
            "prepared": "data/prepared.jsonl",
            "embeddings": "data/features.npz",
            "output_dir": "results/tiny",
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
        "graph": {"k_values": [2, 3], "cosine_threshold": 0.8, "lpe_dimensions": 4},
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


def test_load_config_resolves_paths_and_keeps_matrix_options(tmp_path: Path) -> None:
    from graphhall.config import load_config

    payload = _valid_config()
    config_path = tmp_path / "configs" / "tiny.yaml"
    config_path.parent.mkdir()
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    config = load_config(config_path, project_root=tmp_path)

    assert config.name == "tiny"
    assert config.paths.prepared == tmp_path / "data/prepared.jsonl"
    assert config.paths.embeddings == tmp_path / "data/features.npz"
    assert config.paths.output_dir == tmp_path / "results/tiny"
    assert config.training["seeds"] == [42, 43]
    assert config.graph["k_values"] == [2, 3]
    assert len(config.config_sha256) == 64


def test_config_rejects_unknown_keys(tmp_path: Path) -> None:
    from graphhall.config import load_config

    payload = _valid_config()
    payload["training"]["wheight_decay"] = 0.9
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    try:
        load_config(config_path, project_root=tmp_path)
    except ValueError as error:
        assert "unknown" in str(error).lower()
        assert "wheight_decay" in str(error)
    else:
        raise AssertionError("unknown config keys must be rejected")


def test_config_rejects_duplicate_seeds_and_k_values(tmp_path: Path) -> None:
    from graphhall.config import load_config

    payload = _valid_config()
    payload["training"]["seeds"] = [42, 42]
    config_path = tmp_path / "duplicate.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    try:
        load_config(config_path, project_root=tmp_path)
    except ValueError as error:
        assert "training.seeds" in str(error)
        assert "duplicates" in str(error)
    else:
        raise AssertionError("duplicate seeds must be rejected")


def test_config_rejects_nonfinite_parameters(tmp_path: Path) -> None:
    from graphhall.config import load_config

    payload = _valid_config()
    payload["training"]["learning_rate"] = float("nan")
    config_path = tmp_path / "nonfinite.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    try:
        load_config(config_path, project_root=tmp_path)
    except ValueError as error:
        assert "learning_rate" in str(error)
        assert "finite" in str(error)
    else:
        raise AssertionError("non-finite training rates must be rejected")


def test_config_can_require_real_inputs(tmp_path: Path) -> None:
    from graphhall.config import load_config

    config_path = tmp_path / "missing-inputs.yaml"
    config_path.write_text(yaml.safe_dump(_valid_config()), encoding="utf-8")

    try:
        load_config(config_path, project_root=tmp_path, require_inputs=True)
    except FileNotFoundError as error:
        assert "data/prepared.jsonl" in str(error)
    else:
        raise AssertionError("benchmark execution must reject missing inputs")


def test_published_matrix_configs_use_full_parent_encoded_inputs() -> None:
    from graphhall.config import load_config
    from graphhall.experiments import build_experiment_matrix

    root = Path(__file__).resolve().parents[1]
    full = load_config(root / "configs/qa-full.yaml")
    multiseed = load_config(root / "configs/qa-multiseed.yaml")

    expected_prepared = root / "data/processed/halueval-qa-full.jsonl"
    expected_embeddings = root / "data/processed/halueval-qa-full.npz"
    for config in (full, multiseed):
        assert config.paths.prepared == expected_prepared
        assert config.paths.embeddings == expected_embeddings
        assert config.paths.source == root / "data/raw/qa_data.jsonl"
        assert config.dataset["split_counts"] == {"train": 8000, "val": 1000, "test": 1000}
        assert config.encoder["max_length"] == 512
        assert config.encoder["batch_size"] == 4
    assert full.training["seeds"] == [42]
    assert multiseed.training["seeds"] == [42, 43, 44]
    full_jobs = build_experiment_matrix(full)
    multiseed_jobs = build_experiment_matrix(multiseed)
    assert len(full_jobs) == 13
    assert sum(job.learner in {"gcn", "gat", "gt"} for job in full_jobs) == 9
    assert sum(job.learner == "linear" for job in full_jobs) == 1
    assert sum(job.learner == "knn" for job in full_jobs) == 3
    assert len(multiseed_jobs) == 39
    assert sum(job.learner in {"gcn", "gat", "gt"} for job in multiseed_jobs) == 27
    assert sum(job.learner == "linear" for job in multiseed_jobs) == 3
    assert sum(job.learner == "knn" for job in multiseed_jobs) == 9


def test_config_rejects_noninteger_schema_version(tmp_path: Path) -> None:
    from graphhall.config import load_config

    payload = _valid_config()
    payload["schema_version"] = 1.0
    config_path = tmp_path / "float-schema.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    try:
        load_config(config_path, project_root=tmp_path)
    except ValueError as error:
        assert "schema_version" in str(error)
    else:
        raise AssertionError("schema_version must be an integer, not a float")


def test_config_rejects_duplicate_yaml_keys(tmp_path: Path) -> None:
    from graphhall.config import load_config

    text = yaml.safe_dump(_valid_config(), sort_keys=False) + "name: shadow-config\n"
    config_path = tmp_path / "duplicate-yaml-key.yaml"
    config_path.write_text(text, encoding="utf-8")

    try:
        load_config(config_path, project_root=tmp_path)
    except ValueError as error:
        assert "duplicate" in str(error).lower()
        assert "name" in str(error)
    else:
        raise AssertionError("duplicate YAML keys must be rejected")
