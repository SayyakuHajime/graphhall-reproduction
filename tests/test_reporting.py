from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
from pathlib import Path

import pytest


def _write_completed_run(root: Path, experiment_id: str, metrics: dict) -> Path:
    run_dir = root / "runs" / experiment_id
    run_dir.mkdir(parents=True)
    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, sort_keys=True) + "\n", encoding="utf-8")
    digest = hashlib.sha256(metrics_path.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "experiment_id": experiment_id,
        "identity": {"config_sha256": "a" * 64, "input_hashes": {"prepared": "b" * 64}},
        "artifacts": {"metrics.json": digest},
    }
    (run_dir / "completion.json").write_text(json.dumps(manifest), encoding="utf-8")
    return run_dir


def test_report_deduplicates_runs_aggregates_seeds_and_lists_pending(tmp_path: Path) -> None:
    from graphhall.reporting import generate_report

    input_dir = tmp_path / "results"
    metrics = {
        "experiment_id": "gcn-seed42-k2",
        "learner": "gcn",
        "seed": 42,
        "k": 2,
        "selection_metric": "f1",
        "best_epoch": 2,
        "best_validation_loss": 0.4,
        "validation_metrics": {"accuracy": 0.75, "f1": 0.8},
        "test_metrics": {"accuracy": 0.5, "f1": 0.4},
    }
    run_dir = _write_completed_run(input_dir, metrics["experiment_id"], metrics)
    duplicate_dir = input_dir / "duplicate" / "runs" / metrics["experiment_id"]
    duplicate_dir.parent.mkdir(parents=True)
    shutil.copytree(run_dir, duplicate_dir)
    seed43 = {
        **metrics,
        "experiment_id": "gcn-seed43-k2",
        "seed": 43,
        "validation_metrics": {"accuracy": 0.7, "f1": 0.7},
        "test_metrics": {"accuracy": 0.8, "f1": 0.8},
    }
    _write_completed_run(input_dir, seed43["experiment_id"], seed43)
    validation_winner = {
        **metrics,
        "experiment_id": "gat-seed42-k2",
        "learner": "gat",
        "validation_metrics": {"accuracy": 0.9, "f1": 0.9},
        "test_metrics": {"accuracy": 0.2, "f1": 0.1},
    }
    _write_completed_run(input_dir, validation_winner["experiment_id"], validation_winner)
    pending_metrics = {
        **metrics,
        "experiment_id": "gt-seed42-k2",
        "learner": "gt",
    }
    interrupted_dir = _write_completed_run(
        input_dir, pending_metrics["experiment_id"], pending_metrics
    )
    interrupted_dir.rename(interrupted_dir.with_name(".gt-seed42-k2.partial"))
    input_dir.mkdir(exist_ok=True)
    (input_dir / "matrix.json").write_text(
        json.dumps(
            {
                "jobs": [
                    {"experiment_id": "gcn-seed42-k2", "learner": "gcn", "seed": 42, "k": 2},
                    {"experiment_id": "gcn-seed43-k2", "learner": "gcn", "seed": 43, "k": 2},
                    {"experiment_id": "gat-seed42-k2", "learner": "gat", "seed": 42, "k": 2},
                    {"experiment_id": "gt-seed42-k2", "learner": "gt", "seed": 42, "k": 2},
                ]
            }
        ),
        encoding="utf-8",
    )

    summary = generate_report(input_dir, tmp_path / "report")

    assert summary["completed_count"] == 3
    assert summary["pending_count"] == 1
    with (tmp_path / "report" / "results.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    aggregate = next(
        row for row in rows if row["row_type"] == "aggregate" and row["learner"] == "gcn"
    )
    assert aggregate["seed_count"] == "2"
    assert math.isclose(float(aggregate["test_f1_mean"]), 0.6)
    assert float(aggregate["test_f1_std"]) > 0.0
    single_seed_aggregate = next(
        row for row in rows if row["row_type"] == "aggregate" and row["learner"] == "gat"
    )
    assert single_seed_aggregate["seed_count"] == "1"
    assert single_seed_aggregate["test_f1_std"] == ""
    selected_seed42 = next(
        row for row in rows if row["row_type"] == "validation_selected" and row["seed"] == "42"
    )
    assert selected_seed42["experiment_id"] == "gat-seed42-k2"
    assert float(selected_seed42["validation_f1"]) == 0.9
    assert float(selected_seed42["test_f1"]) == 0.1
    pending = next(row for row in rows if row["status"] == "pending")
    assert pending["learner"] == "gt"
    markdown = (tmp_path / "report" / "results.md").read_text(encoding="utf-8")
    assert "Validation-selected" in markdown
    assert "test score is not used" in markdown.lower()
    assert "Fidelity caveats" in markdown
    assert "parameter counts differ" in markdown
    assert "SD n/a" in markdown


@pytest.mark.parametrize("mode", ["mixed_runs", "matrix_mismatch"])
def test_report_rejects_incomparable_experiment_identities(tmp_path: Path, mode: str) -> None:
    from graphhall.reporting import generate_report

    root = tmp_path / "results"
    metrics = {
        "experiment_id": "gcn-seed42-k2",
        "learner": "gcn",
        "seed": 42,
        "k": 2,
        "validation_metrics": {"f1": 0.8},
        "test_metrics": {"f1": 0.7},
    }
    first = _write_completed_run(root, metrics["experiment_id"], metrics)
    identity = json.loads((first / "completion.json").read_text())["identity"]
    other_identity = {**identity, "config_sha256": "c" * 64}
    if mode == "mixed_runs":
        second = _write_completed_run(
            root,
            "gcn-seed43-k2",
            {**metrics, "experiment_id": "gcn-seed43-k2", "seed": 43},
        )
        manifest_path = second / "completion.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["identity"] = other_identity
        manifest_path.write_text(json.dumps(manifest))
    else:
        (root / "matrix.json").write_text(json.dumps({"identity": other_identity, "jobs": []}))
    with pytest.raises(ValueError, match="identity|provenance"):
        generate_report(root, tmp_path / "report")
    assert not (tmp_path / "report/results.csv").exists()


def test_report_rejects_corrupted_completed_artifact(tmp_path: Path) -> None:
    from graphhall.reporting import generate_report

    input_dir = tmp_path / "results"
    run_dir = _write_completed_run(
        input_dir,
        "linear-seed42",
        {
            "experiment_id": "linear-seed42",
            "learner": "linear",
            "seed": 42,
            "k": None,
            "selection_metric": "f1",
            "best_epoch": 1,
            "best_validation_loss": 0.5,
            "validation_metrics": {"f1": 0.5},
            "test_metrics": {"f1": 0.5},
        },
    )
    (run_dir / "metrics.json").write_text("{}", encoding="utf-8")

    try:
        generate_report(input_dir, tmp_path / "report")
    except ValueError as error:
        assert "hash" in str(error).lower()
    else:
        raise AssertionError("reporting must refuse corrupted completed artifacts")
