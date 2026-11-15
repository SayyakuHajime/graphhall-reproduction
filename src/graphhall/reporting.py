from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

_METRICS = ("accuracy", "precision", "recall", "f1", "auroc", "auprc", "ece", "brier", "nll")
_CSV_FIELDS = [
    "row_type",
    "status",
    "experiment_id",
    "learner",
    "seed",
    "k",
    "selection_metric",
    "best_epoch",
    "best_validation_loss",
    "seed_count",
    *[f"validation_{metric}" for metric in _METRICS],
    *[f"test_{metric}" for metric in _METRICS],
    *[f"test_{metric}_mean" for metric in _METRICS],
    *[f"test_{metric}_std" for metric in _METRICS],
]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return value


def _verify_completion(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = _read_json(path)
    if manifest.get("schema_version") != 1 or manifest.get("status") != "complete":
        raise ValueError(f"invalid completion manifest: {path}")
    experiment_id = manifest.get("experiment_id")
    if not isinstance(experiment_id, str) or not experiment_id:
        raise ValueError(f"completion manifest has no experiment_id: {path}")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError(f"completion manifest has no artifacts: {path}")
    run_dir = path.parent.resolve()
    for relative, expected_hash in artifacts.items():
        if not isinstance(relative, str) or not isinstance(expected_hash, str):
            raise ValueError(f"invalid artifact entry in {path}")
        artifact_path = (run_dir / relative).resolve()
        try:
            artifact_path.relative_to(run_dir)
        except ValueError as exc:
            raise ValueError(f"artifact path escapes experiment directory: {relative}") from exc
        if not artifact_path.is_file():
            raise ValueError(f"completed artifact is missing: {artifact_path}")
        actual_hash = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise ValueError(f"artifact hash mismatch: {artifact_path}")
    metrics_path = run_dir / "metrics.json"
    if "metrics.json" not in artifacts:
        raise ValueError(f"completion manifest omits metrics.json: {path}")
    metrics = _read_json(metrics_path)
    if metrics.get("experiment_id") != experiment_id:
        raise ValueError(f"metrics experiment_id disagrees with completion manifest: {path}")
    for key in ("learner", "seed", "validation_metrics", "test_metrics"):
        if key not in metrics:
            raise ValueError(f"metrics artifact missing {key}: {metrics_path}")
    if not isinstance(metrics["validation_metrics"], dict) or not isinstance(
        metrics["test_metrics"], dict
    ):
        raise ValueError(f"metrics maps are invalid: {metrics_path}")
    return manifest, metrics


def _as_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _base_row(metrics: dict[str, Any], *, row_type: str, status: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        "row_type": row_type,
        "status": status,
        "experiment_id": metrics.get("experiment_id", ""),
        "learner": metrics.get("learner", ""),
        "seed": metrics.get("seed", ""),
        "k": metrics.get("k", ""),
        "selection_metric": metrics.get("selection_metric", ""),
        "best_epoch": metrics.get("best_epoch", ""),
        "best_validation_loss": metrics.get("best_validation_loss", ""),
        "seed_count": "",
    }
    for key, value in metrics.get("validation_metrics", {}).items():
        if key in _METRICS:
            row[f"validation_{key}"] = value
    for key, value in metrics.get("test_metrics", {}).items():
        if key in _METRICS:
            row[f"test_{key}"] = value
    return row


def _metric_pair(row: dict[str, Any], metric: str) -> str:
    mean = row.get(f"test_{metric}_mean", "")
    deviation = row.get(f"test_{metric}_std", "")
    if mean == "":
        return "—"
    if deviation == "":
        return f"{mean} (SD n/a)"
    return f"{mean} ± {deviation}"


def _report_markdown(
    per_seed: list[dict[str, Any]],
    aggregates: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    pending: list[dict[str, Any]],
) -> str:
    lines = [
        "# Generated experiment report",
        "",
        "Rows are read from hash-verified completed artifacts; "
        "pending matrix jobs are shown separately.",
        "Validation-selected winners use validation metrics only. "
        "The test score is not used for selection.",
        "Test metrics are exposed only after that validation-only selection.",
        "",
        "## Per-seed completed experiments",
        "",
        "| Experiment | Learner | Seed | k | Val F1 | Test F1 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in per_seed:
        lines.append(
            f"| {row['experiment_id']} | {row['learner']} | {row['seed']} | {row.get('k', '')} "
            f"| {row.get('validation_f1', '')} | {row.get('test_f1', '')} |"
        )
    lines.extend(["", "## Aggregate test metrics (mean ± sample standard deviation)", ""])
    lines.append("| Learner | k | Seed count | Accuracy | F1 | AUROC | AUPRC |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in aggregates:
        lines.append(
            f"| {row['learner']} | {row.get('k', '')} | {row['seed_count']} "
            f"| {_metric_pair(row, 'accuracy')} | {_metric_pair(row, 'f1')} "
            f"| {_metric_pair(row, 'auroc')} | {_metric_pair(row, 'auprc')} |"
        )
    lines.extend(
        [
            "",
            "## Validation-selected results",
            "",
            "| Seed | Winner | Validation metric/value | Test F1 (post-selection) |",
            "|---:|---|---:|---:|",
        ]
    )
    for row in selected:
        metric = row.get("selection_metric", "f1")
        value = row.get(f"validation_{metric}", "")
        lines.append(
            f"| {row.get('seed', '')} | {row.get('experiment_id', '')} | {metric}={value} "
            f"| {row.get('test_f1', '')} |"
        )
    lines.extend(
        [
            "",
            "## Fidelity caveats",
            "This is an executable local comparison, not proof that paper scores were reproduced.",
            "The frozen encoder pairs the question context with the candidate answer; "
            "knowledge is excluded.",
            "Evaluation is transductive over the fixed pool.",
            "Sampling, attention-edge semantics, and some GT layer details are underspecified.",
            "Audited vanilla PyG GAT/GT parameter counts differ from the manuscript counts; "
            "no dummy parameters are added.",
            "The project docs/paper-fidelity.md records the measured variants and sources.",
        ]
    )
    lines.extend(
        [
            "",
            "## Pending experiments",
            "",
            "| Experiment | Learner | Seed | k |",
            "|---|---|---:|---:|",
        ]
    )
    for row in pending:
        lines.append(
            f"| {row['experiment_id']} | {row['learner']} | {row['seed']} | {row.get('k', '')} |"
        )
    lines.extend(["", "No pending rows means the recorded matrix has no unfinished jobs.", ""])
    return "\n".join(lines)


def generate_report(input_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Aggregate verified run artifacts into per-seed and validation-selected reports."""
    root = Path(input_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"experiment input directory does not exist: {root}")
    by_id: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for completion_path in sorted(root.rglob("completion.json")):
        relative_parts = completion_path.relative_to(root).parts[:-1]
        if any(part.startswith(".") or part.endswith(".partial") for part in relative_parts):
            continue
        manifest, metrics = _verify_completion(completion_path)
        key = manifest["experiment_id"]
        if key in by_id:
            old_manifest, old_metrics = by_id[key]
            if old_manifest.get("identity") != manifest.get("identity") or old_metrics != metrics:
                raise ValueError(f"conflicting completed artifacts for experiment_id {key}")
            continue
        by_id[key] = (manifest, metrics)

    completed = [pair[1] for pair in by_id.values()]
    planned: dict[str, dict[str, Any]] = {}
    matrix_path = root / "matrix.json"
    expected_identity = None
    if matrix_path.is_file():
        matrix = _read_json(matrix_path)
        expected_identity = matrix.get("identity")
        jobs = matrix.get("jobs")
        if not isinstance(jobs, list):
            raise ValueError("matrix.json jobs must be a list")
        for job in jobs:
            if not isinstance(job, dict) or not isinstance(job.get("experiment_id"), str):
                raise ValueError("matrix.json contains an invalid job")
            job_id = job["experiment_id"]
            if job_id in planned and planned[job_id] != job:
                raise ValueError(f"matrix.json contains conflicting duplicate job {job_id}")
            planned[job_id] = job

    for manifest, _metrics in by_id.values():
        identity = manifest.get("identity")
        if not isinstance(identity, dict) or not identity:
            raise ValueError("completed run is missing experiment identity provenance")
        if expected_identity is None:
            expected_identity = identity
        if identity != expected_identity:
            raise ValueError("report contains mixed or matrix-mismatched experiment identity")

    pending: list[dict[str, Any]] = []
    for experiment_id, job in sorted(planned.items()):
        if experiment_id not in by_id:
            pending.append(
                {
                    "experiment_id": experiment_id,
                    "learner": job.get("learner", ""),
                    "seed": job.get("seed", ""),
                    "k": job.get("k", ""),
                    "validation_metrics": {},
                    "test_metrics": {},
                }
            )

    per_seed = sorted(
        (_base_row(metrics, row_type="per_seed", status="completed") for metrics in completed),
        key=lambda row: (
            str(row["learner"]),
            str(row["seed"]),
            str(row["k"]),
            str(row["experiment_id"]),
        ),
    )
    grouped: dict[tuple[str, Any], list[dict[str, Any]]] = defaultdict(list)
    for metrics in completed:
        grouped[(str(metrics["learner"]), metrics.get("k"))].append(metrics)
    aggregates: list[dict[str, Any]] = []
    for (learner, k), group in sorted(
        grouped.items(), key=lambda item: (item[0][0], str(item[0][1]))
    ):
        seeds = {str(item["seed"]) for item in group}
        row: dict[str, Any] = {
            "row_type": "aggregate",
            "status": "completed",
            "experiment_id": "",
            "learner": learner,
            "seed": "",
            "k": "" if k is None else k,
            "selection_metric": "",
            "best_epoch": "",
            "best_validation_loss": "",
            "seed_count": len(seeds),
        }
        for metric in _METRICS:
            values = [
                value
                for item in group
                if (value := _as_number(item.get("test_metrics", {}).get(metric))) is not None
            ]
            if values:
                row[f"test_{metric}_mean"] = statistics.fmean(values)
                if len(values) > 1:
                    row[f"test_{metric}_std"] = statistics.stdev(values)
        aggregates.append(row)

    validation_candidates: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for metrics in completed:
        selection_metric = metrics.get("selection_metric", "f1")
        if selection_metric not in _METRICS:
            raise ValueError(f"unsupported selection metric in completed run: {selection_metric}")
        value = _as_number(metrics.get("validation_metrics", {}).get(selection_metric))
        if value is not None:
            validation_candidates[metrics.get("seed")].append(metrics)
    selected: list[dict[str, Any]] = []
    for _seed, candidates in sorted(validation_candidates.items(), key=lambda item: str(item[0])):
        winner = sorted(
            candidates,
            key=lambda item: (
                -float(item["validation_metrics"][item.get("selection_metric", "f1")]),
                _as_number(item.get("best_validation_loss"))
                if _as_number(item.get("best_validation_loss")) is not None
                else math.inf,
                str(item["experiment_id"]),
            ),
        )[0]
        selected.append(_base_row(winner, row_type="validation_selected", status="completed"))

    all_rows = [*per_seed, *aggregates]
    all_rows.extend(selected)
    all_rows.extend(_base_row(item, row_type="pending", status="pending") for item in pending)
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    with (out / "results.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)
    (out / "results.md").write_text(
        _report_markdown(per_seed, aggregates, selected, pending), encoding="utf-8"
    )
    return {
        "completed_count": len(completed),
        "pending_count": len(pending),
        "selected_count": len(selected),
        "csv_path": out / "results.csv",
        "markdown_path": out / "results.md",
    }
