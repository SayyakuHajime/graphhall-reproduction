from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import uuid
from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from graphhall.config import ExperimentConfig, load_config
from graphhall.data import split_candidate_records
from graphhall.graph import build_knn_graph, laplacian_positional_encoding


@dataclass(frozen=True)
class ExperimentJob:
    learner: str
    seed: int
    k: int | None

    @property
    def experiment_id(self) -> str:
        learner = self.learner.replace("_", "-")
        suffix = f"-k{self.k}" if self.k is not None else ""
        return f"{learner}-seed{self.seed}{suffix}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "learner": self.learner,
            "seed": self.seed,
            "k": self.k,
        }


@dataclass(frozen=True)
class _Inputs:
    rows: list[dict[str, Any]]
    features: np.ndarray
    labels: np.ndarray
    train_mask: np.ndarray
    val_mask: np.ndarray
    test_mask: np.ndarray
    embedding_metadata: dict[str, Any]
    prepared_metadata: dict[str, Any]
    hashes: dict[str, str | None]
    model_identity_sha256: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _implementation_sha256() -> str:
    """Bind reuse to actual package code and numerical dependency versions."""
    from importlib.metadata import version

    digest = hashlib.sha256()
    for source in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(source.name.encode("utf-8"))
        digest.update(source.read_bytes())
    for dependency in (
        "torch",
        "torch-geometric",
        "transformers",
        "numpy",
        "scipy",
        "scikit-learn",
    ):
        digest.update(f"{dependency}=={version(dependency)}".encode())
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
        )
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_object(path: Path) -> dict[str, Any]:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(result, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return result


def build_experiment_matrix(config: ExperimentConfig) -> list[ExperimentJob]:
    """Expand the configured models, seeds, and graph/voting neighborhood sizes."""
    jobs = [
        ExperimentJob(learner, seed, k)
        for learner in config.training["learners"]
        for seed in config.training["seeds"]
        for k in config.graph["k_values"]
    ]
    for baseline in config.baselines:
        if baseline == "linear":
            jobs.extend(ExperimentJob(baseline, seed, None) for seed in config.training["seeds"])
        elif baseline == "knn":
            jobs.extend(
                ExperimentJob(baseline, seed, k)
                for seed in config.training["seeds"]
                for k in config.graph["k_values"]
            )
    if config.finetuned_deberta["enabled"]:
        seeds = (
            config.training["seeds"]
            if config.finetuned_deberta["seed_from_matrix"]
            else config.training["seeds"][:1]
        )
        jobs.extend(ExperimentJob("finetuned_deberta", seed, None) for seed in seeds)
    ids = [job.experiment_id for job in jobs]
    if len(ids) != len(set(ids)):
        raise ValueError("experiment matrix contains duplicate experiment IDs")
    return jobs


def _read_source_records(path: Path) -> list[dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"cannot read source dataset {path}: {exc}") from exc
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        try:
            parsed = [json.loads(line) for line in text.splitlines() if line.strip()]
        except json.JSONDecodeError as exc:
            raise ValueError(f"source dataset is not valid JSON or JSONL: {path}") from exc
    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list) or any(not isinstance(row, dict) for row in parsed):
        raise ValueError(f"source dataset must contain JSON objects: {path}")
    return parsed


def _load_inputs(
    prepared_path: Path,
    embeddings_path: Path,
    dataset_config: dict[str, Any],
    encoder_config: dict[str, Any],
    *,
    source_path: Path | None = None,
) -> _Inputs:
    if not prepared_path.is_file() or not embeddings_path.is_file():
        raise FileNotFoundError("prepared JSONL and embedding NPZ inputs must both exist")
    prepared_hash = _sha256_file(prepared_path)
    embedding_hash = _sha256_file(embeddings_path)
    prepared_meta_path = prepared_path.with_suffix(".meta.json")
    embedding_meta_path = embeddings_path.with_suffix(".meta.json")
    if not embedding_meta_path.is_file():
        raise ValueError("embedding metadata sidecar is required for model provenance checks")
    embedding_metadata = _read_object(embedding_meta_path)
    embedding_meta_hash = _sha256_file(embedding_meta_path)
    prepared_meta_hash = _sha256_file(prepared_meta_path) if prepared_meta_path.is_file() else None
    if prepared_meta_path == embedding_meta_path:
        prepared_metadata = embedding_metadata if "source_revision" in embedding_metadata else {}
    elif prepared_meta_path.is_file():
        prepared_metadata = _read_object(prepared_meta_path)
    else:
        prepared_metadata = {}
    if not prepared_metadata and source_path is None:
        raise ValueError("prepared metadata sidecar or a hash-verified raw source path is required")

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(prepared_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"prepared JSONL line {line_number} is not valid JSON") from exc
        if not isinstance(row, dict):
            raise ValueError(f"prepared JSONL line {line_number} must be an object")
        record_id = row.get("record_id")
        if not isinstance(record_id, str) or not record_id or record_id in seen:
            raise ValueError(
                f"prepared JSONL line {line_number} has a missing or duplicate record_id"
            )
        seen.add(record_id)
        if row.get("split") not in {"train", "val", "test"}:
            raise ValueError(f"prepared JSONL line {line_number} has an invalid split")
        for key in ("question", "answer"):
            if not isinstance(row.get(key), str):
                raise ValueError(f"prepared JSONL line {line_number} needs a string {key}")
        label = row.get("label")
        if isinstance(label, bool) or not isinstance(label, int) or label not in (0, 1):
            raise ValueError(f"prepared JSONL line {line_number} has a non-binary label")
        rows.append(row)
    if len(rows) != dataset_config["selected_records"]:
        raise ValueError(
            f"prepared row count {len(rows)} does not match configured "
            f"selected_records={dataset_config['selected_records']}"
        )
    split_counts = dict(Counter(row["split"] for row in rows))
    if split_counts != dataset_config["split_counts"]:
        raise ValueError(f"prepared split counts mismatch: {split_counts}")
    for split in ("train", "val", "test"):
        label_counts = Counter(row["label"] for row in rows if row["split"] == split)
        if abs(label_counts[0] - label_counts[1]) > 1:
            raise ValueError(f"prepared {split} split is not label-balanced")
    expected_source_hash = dataset_config["source_sha256"]
    source_file_hash: str | None = None
    if source_path is not None:
        if not source_path.is_file():
            raise FileNotFoundError(f"configured raw source data does not exist: {source_path}")
        source_file_hash = _sha256_file(source_path)
        if expected_source_hash is not None and source_file_hash != expected_source_hash:
            raise ValueError("raw source hash does not match config.dataset.source_sha256")
        source_records = _read_source_records(source_path)
        if len(source_records) != dataset_config["source_rows"]:
            raise ValueError("raw source row count does not match config.dataset.source_rows")
        expected_rows = split_candidate_records(
            source_records,
            max_records=dataset_config["selected_records"],
            seed=dataset_config["split_seed"],
        )
        if expected_rows != rows:
            raise ValueError(
                "prepared rows do not match the configured source revision, selection, "
                "and split seed"
            )
        if prepared_metadata:
            if prepared_metadata.get("source_sha256") not in (None, source_file_hash):
                raise ValueError("prepared metadata source hash disagrees with raw source data")
            expected_metadata = {
                "source_revision": dataset_config["source_revision"],
                "source_row_count": dataset_config["source_rows"],
                "seed": dataset_config["split_seed"],
                "split_counts": dataset_config["split_counts"],
                "label_mapping": {"right_answer": 0, "hallucinated_answer": 1},
            }
            for key, expected in expected_metadata.items():
                if key in prepared_metadata and prepared_metadata[key] != expected:
                    raise ValueError(
                        f"prepared metadata {key} does not match configured provenance"
                    )
            if prepared_metadata.get("original_records", len(rows)) != len(rows):
                raise ValueError("prepared metadata original_records does not match row count")
        else:
            prepared_metadata = {
                "source_revision": dataset_config["source_revision"],
                "source_sha256": source_file_hash,
                "source_row_count": len(source_records),
                "original_records": len(rows),
                "seed": dataset_config["split_seed"],
                "split_counts": split_counts,
                "label_mapping": {"right_answer": 0, "hallucinated_answer": 1},
                "reconstructed_from_source": True,
            }
    else:
        source_hash = prepared_metadata.get("source_sha256")
        if expected_source_hash is not None and source_hash != expected_source_hash:
            raise ValueError("prepared source hash does not match config.dataset.source_sha256")
        if prepared_metadata.get("source_revision") != dataset_config["source_revision"]:
            raise ValueError(
                "prepared source revision does not match config.dataset.source_revision"
            )
        if prepared_metadata.get("source_row_count") != dataset_config["source_rows"]:
            raise ValueError("prepared source row count does not match config.dataset.source_rows")
        if prepared_metadata.get("seed") != dataset_config["split_seed"]:
            raise ValueError("prepared split seed does not match config.dataset.split_seed")
        if prepared_metadata.get("split_counts") != dataset_config["split_counts"]:
            raise ValueError(
                "prepared metadata split counts do not match config.dataset.split_counts"
            )
        if prepared_metadata.get("label_mapping") != {
            "right_answer": 0,
            "hallucinated_answer": 1,
        }:
            raise ValueError("prepared metadata label mapping is missing or inconsistent")
        if prepared_metadata.get("original_records", len(rows)) != len(rows):
            raise ValueError("prepared metadata original_records does not match row count")

    if embedding_metadata.get("prepared_jsonl_sha256") != prepared_hash:
        raise ValueError("embedding metadata prepared_jsonl_sha256 does not match prepared JSONL")
    if embedding_metadata.get("row_count") != len(rows):
        raise ValueError("embedding metadata row_count does not match prepared JSONL")
    if embedding_metadata.get("model") != encoder_config["model"]:
        raise ValueError("embedding model identity does not match configured encoder.model")
    if embedding_metadata.get("resolved_revision") != encoder_config["revision"]:
        raise ValueError("embedding revision does not match configured encoder.revision")
    model_state_hash = embedding_metadata.get("model_state_sha256")
    if not isinstance(model_state_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", model_state_hash):
        raise ValueError("embedding metadata must contain a valid model_state_sha256")
    if embedding_metadata.get("max_length") != encoder_config["max_length"]:
        raise ValueError(
            "embedding metadata max_length does not match configured encoder.max_length"
        )
    if embedding_metadata.get("include_knowledge") is not False:
        raise ValueError("benchmark embeddings must exclude the knowledge field")
    if embedding_metadata.get("context_policy") != encoder_config["context_policy"]:
        raise ValueError(
            "embedding context policy does not match configured encoder.context_policy"
        )
    if embedding_metadata.get("pooling") != encoder_config["pooling"]:
        raise ValueError("embedding pooling does not match configured encoder.pooling")
    if embedding_metadata.get("batch_size") != encoder_config["batch_size"]:
        raise ValueError("embedding batch_size does not match configured encoder.batch_size")
    if embedding_metadata.get("encoder_frozen") is not True:
        raise ValueError("benchmark embeddings must come from a frozen encoder")
    if embedding_metadata.get("hidden_size") != encoder_config["hidden_size"]:
        raise ValueError("embedding hidden_size does not match configured encoder.hidden_size")

    with np.load(embeddings_path, allow_pickle=False) as archive:
        if "embeddings" not in archive or "record_ids" not in archive:
            raise ValueError("embedding NPZ must contain embeddings and record_ids")
        features = np.asarray(archive["embeddings"], dtype=np.float32)
        record_ids = [str(value) for value in archive["record_ids"].tolist()]
    if record_ids != [row["record_id"] for row in rows]:
        raise ValueError("embedding record IDs/order do not match prepared JSONL")
    if features.shape != (len(rows), encoder_config["hidden_size"]):
        raise ValueError(
            "embedding array shape does not match row count and configured hidden_size"
        )
    if not np.isfinite(features).all() or np.any(np.linalg.norm(features, axis=1) == 0):
        raise ValueError("embeddings must be finite and non-zero")

    labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
    splits = [row["split"] for row in rows]
    train_mask = np.asarray([split == "train" for split in splits], dtype=bool)
    val_mask = np.asarray([split == "val" for split in splits], dtype=bool)
    test_mask = np.asarray([split == "test" for split in splits], dtype=bool)
    model_identity = {
        "model": embedding_metadata["model"],
        "resolved_revision": embedding_metadata["resolved_revision"],
        "model_state_sha256": model_state_hash,
        "encoder_config": encoder_config,
        "embedding_meta_sha256": embedding_meta_hash,
        "source_file_sha256": source_file_hash,
    }
    model_identity_sha256 = _canonical_sha256(model_identity)
    hashes: dict[str, str | None] = {
        "prepared_jsonl_sha256": prepared_hash,
        "embeddings_npz_sha256": embedding_hash,
        "prepared_meta_sha256": prepared_meta_hash,
        "embedding_meta_sha256": embedding_meta_hash,
        "source_file_sha256": source_file_hash,
        "model_identity_sha256": model_identity_sha256,
    }
    return _Inputs(
        rows=rows,
        features=features,
        labels=labels,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        embedding_metadata=embedding_metadata,
        prepared_metadata=prepared_metadata,
        hashes=hashes,
        model_identity_sha256=model_identity_sha256,
    )


def _graph_statistics(
    edge_index: np.ndarray, edge_weight: np.ndarray, nodes: int
) -> dict[str, Any]:
    incoming = np.bincount(edge_index[1], minlength=nodes)
    return {
        "num_nodes": nodes,
        "directed_knn_edges": int(edge_index.shape[1]),
        "explicit_self_loops": nodes,
        "effective_edges_with_self_loops": int(edge_index.shape[1] + nodes),
        "query_nodes_without_knn_neighbors": int(np.count_nonzero(incoming == 0)),
        "edge_weight_min": float(edge_weight.min()) if edge_weight.size else None,
        "edge_weight_max": float(edge_weight.max()) if edge_weight.size else None,
    }


def _identity(config: ExperimentConfig, inputs: _Inputs) -> dict[str, Any]:
    return {
        "config_sha256": config.config_sha256,
        "implementation_sha256": _implementation_sha256(),
        "prepared_jsonl_sha256": inputs.hashes["prepared_jsonl_sha256"],
        "embeddings_npz_sha256": inputs.hashes["embeddings_npz_sha256"],
        "prepared_meta_sha256": inputs.hashes["prepared_meta_sha256"],
        "embedding_meta_sha256": inputs.hashes["embedding_meta_sha256"],
        "source_file_sha256": inputs.hashes["source_file_sha256"],
        "model_identity_sha256": inputs.model_identity_sha256,
    }


def _artifact_hashes(run_dir: Path) -> dict[str, str]:
    artifacts: dict[str, str] = {}
    for path in sorted(item for item in run_dir.rglob("*") if item.is_file()):
        relative = path.relative_to(run_dir).as_posix()
        if relative != "completion.json":
            artifacts[relative] = _sha256_file(path)
    if not artifacts:
        raise RuntimeError(f"experiment produced no artifacts: {run_dir}")
    return artifacts


def _verify_complete_run(
    run_dir: Path,
    *,
    identity: dict[str, Any] | None = None,
    job: ExperimentJob | None = None,
) -> dict[str, Any]:
    manifest_path = run_dir / "completion.json"
    manifest = _read_object(manifest_path)
    if manifest.get("schema_version") != 1 or manifest.get("status") != "complete":
        raise ValueError(f"invalid completion manifest: {manifest_path}")
    if identity is not None and manifest.get("identity") != identity:
        raise ValueError(f"stale experiment identity; refusing to reuse {run_dir}")
    if job is not None and manifest.get("job") != job.as_dict():
        raise ValueError(f"experiment job mismatch; refusing to reuse {run_dir}")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError(f"completion manifest has no artifact hashes: {manifest_path}")
    base = run_dir.resolve()
    for relative, expected in artifacts.items():
        artifact = (base / relative).resolve()
        try:
            artifact.relative_to(base)
        except ValueError as exc:
            raise ValueError(f"artifact path escapes run directory: {relative}") from exc
        if not artifact.is_file() or _sha256_file(artifact) != expected:
            raise ValueError(f"artifact hash mismatch or missing file: {artifact}")
    return manifest


def _save_predictions(path: Path, rows: list[dict[str, Any]], probabilities: np.ndarray) -> None:
    if probabilities.shape != (len(rows), 2) or not np.isfinite(probabilities).all():
        raise ValueError("model predictions must be finite N x 2 probabilities")
    if (
        np.any(probabilities < 0)
        or np.any(probabilities > 1)
        or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6, rtol=1e-6)
    ):
        raise ValueError("model predictions must be normalized probabilities")
    predictions = probabilities.argmax(axis=1)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "record_id",
                "split",
                "gold_label",
                "predicted_label",
                "probability_0",
                "probability_1",
            ),
        )
        writer.writeheader()
        for index, row in enumerate(rows):
            writer.writerow(
                {
                    "record_id": row["record_id"],
                    "split": row["split"],
                    "gold_label": row["label"],
                    "predicted_label": int(predictions[index]),
                    "probability_0": float(probabilities[index, 0]),
                    "probability_1": float(probabilities[index, 1]),
                }
            )


def _write_run(
    config: ExperimentConfig,
    job: ExperimentJob,
    inputs: _Inputs,
    identity: dict[str, Any],
    run_dir: Path,
    graph_cache: dict[int, tuple[np.ndarray, np.ndarray, dict[str, Any], np.ndarray | None]],
) -> None:
    run_dir.mkdir(parents=True, exist_ok=False)
    effective = {
        "schema_version": 1,
        "name": config.name,
        "run_type": config.run_type,
        "config_sha256": config.config_sha256,
        "config": config.raw,
        "config_source": str(config.source_path),
        "project_root": str(config.project_root),
        "input_paths": {
            "prepared": str(config.paths.prepared),
            "embeddings": str(config.paths.embeddings),
            "source": str(config.paths.source) if config.paths.source is not None else None,
        },
    }
    _atomic_json(run_dir / "effective_config.json", effective)
    provenance = {
        "identity": identity,
        "input_hashes": inputs.hashes,
        "encoder_metadata": inputs.embedding_metadata,
        "prepared_metadata": inputs.prepared_metadata,
        "input_paths": effective["input_paths"],
        "fixed_split_seed": config.dataset["split_seed"],
        "training_seed": job.seed,
        "test_labels_used_for_training": False,
        "transductive_full_graph": True,
    }
    _atomic_json(run_dir / "provenance.json", provenance)

    if job.k is None or job.learner == "knn":
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_weight = np.empty((0,), dtype=np.float32)
        graph_stats = None
        positional = None
    else:
        edge_index, edge_weight, graph_stats, positional = graph_cache[job.k]
    if job.learner in {"gcn", "gat", "gt"}:
        np.savez_compressed(run_dir / "graph.npz", edge_index=edge_index, edge_weight=edge_weight)
        if job.learner == "gt":
            if positional is None:
                raise RuntimeError("GT requested without configured Laplacian positional encodings")
            np.save(run_dir / "lpe.npy", positional, allow_pickle=False)
        features = inputs.features
        if job.learner == "gt":
            features = np.concatenate((features, positional), axis=1)
        from graphhall.train import train_model

        fitted = train_model(
            job.learner,
            torch.from_numpy(np.asarray(features, dtype=np.float32)),
            edge_index,
            edge_weight,
            torch.from_numpy(inputs.labels),
            torch.from_numpy(inputs.train_mask),
            torch.from_numpy(inputs.val_mask),
            torch.from_numpy(inputs.test_mask),
            checkpoint_path=run_dir / "best.pt",
            learning_rate=config.training["learning_rate"],
            weight_decay=config.training["weight_decay"],
            max_epochs=config.training["max_epochs"],
            validation_interval=config.training["validation_interval"],
            patience=config.training["patience"],
            seed=job.seed,
            device=config.encoder["device"],
            model_profile=config.training["model_profile"],
        )
        probabilities = np.asarray(fitted["probabilities"], dtype=np.float64)
        validation_metrics = fitted["validation_metrics"]
        test_metrics = fitted["metrics"]
        history = fitted["history"]
        best_epoch = fitted["best_epoch"]
        best_validation_loss = fitted["best_validation_loss"]
        parameter_count = fitted["parameter_count"]
    elif job.learner == "linear":
        from graphhall.train import train_model

        fitted = train_model(
            "linear",
            torch.from_numpy(inputs.features),
            np.empty((2, 0), dtype=np.int64),
            np.empty((0,), dtype=np.float32),
            torch.from_numpy(inputs.labels),
            torch.from_numpy(inputs.train_mask),
            torch.from_numpy(inputs.val_mask),
            torch.from_numpy(inputs.test_mask),
            checkpoint_path=run_dir / "best.pt",
            learning_rate=config.training["learning_rate"],
            weight_decay=config.training["weight_decay"],
            max_epochs=config.training["max_epochs"],
            validation_interval=config.training["validation_interval"],
            patience=config.training["patience"],
            seed=job.seed,
            device=config.encoder["device"],
            model_profile=config.training["model_profile"],
        )
        probabilities = np.asarray(fitted["probabilities"], dtype=np.float64)
        validation_metrics = fitted["validation_metrics"]
        test_metrics = fitted["metrics"]
        history = fitted["history"]
        best_epoch = fitted["best_epoch"]
        best_validation_loss = fitted["best_validation_loss"]
        parameter_count = fitted["parameter_count"]
    elif job.learner == "knn":
        from graphhall.baselines import run_knn_baseline

        fitted = run_knn_baseline(
            inputs.features,
            inputs.labels,
            inputs.train_mask,
            inputs.val_mask,
            inputs.test_mask,
            k=job.k,
        )
        probabilities = np.asarray(fitted["probabilities"], dtype=np.float64)
        validation_metrics = fitted["validation_metrics"]
        test_metrics = fitted["metrics"]
        history = []
        best_epoch = 0
        best_validation_loss = fitted["best_validation_loss"]
        parameter_count = 0
    elif job.learner == "finetuned_deberta":
        from graphhall.baselines import finetune_classifier

        tuning = config.finetuned_deberta
        fitted = finetune_classifier(
            config.paths.prepared,
            run_dir / "finetuned-model",
            model_name=tuning["model_name"],
            revision=tuning["revision"],
            device=config.encoder["device"],
            batch_size=tuning["batch_size"],
            max_length=tuning["max_length"],
            epochs=tuning["epochs"],
            learning_rate=tuning["learning_rate"],
            seed=job.seed,
            gradient_accumulation=tuning["gradient_accumulation"],
            max_steps=tuning["max_steps"],
        )
        probabilities = np.asarray(fitted["probabilities"], dtype=np.float64)
        validation_metrics = fitted["validation_metrics"]
        test_metrics = fitted["metrics"]
        history = fitted["history"]
        best_epoch = fitted["best_epoch"]
        best_validation_loss = fitted["best_validation_loss"]
        parameter_count = fitted["parameter_count"]
    else:
        raise ValueError(f"unsupported experiment learner {job.learner!r}")

    _save_predictions(run_dir / "predictions.csv", inputs.rows, probabilities)
    _atomic_json(run_dir / "history.json", {"history": history})
    metrics = {
        "experiment_id": job.experiment_id,
        "learner": job.learner,
        "seed": job.seed,
        "k": job.k,
        "selection_metric": config.training["selection_metric"],
        "best_epoch": int(best_epoch),
        "best_validation_loss": float(best_validation_loss),
        "parameter_count": int(parameter_count),
        "sample_count": len(inputs.rows),
        "split_counts": dict(Counter(row["split"] for row in inputs.rows)),
        "graph_stats": graph_stats,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "note": "Test metrics are reported after validation-based checkpoint selection.",
    }
    _atomic_json(run_dir / "metrics.json", metrics)
    completion = {
        "schema_version": 1,
        "status": "complete",
        "experiment_id": job.experiment_id,
        "job": job.as_dict(),
        "identity": identity,
        "artifacts": _artifact_hashes(run_dir),
    }
    _atomic_json(run_dir / "completion.json", completion)


def _matrix_document(
    config: ExperimentConfig,
    identity: dict[str, Any],
    jobs: list[ExperimentJob],
    complete_ids: set[str],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "config_sha256": config.config_sha256,
        "identity": identity,
        "jobs": [
            {
                **job.as_dict(),
                "status": "completed" if job.experiment_id in complete_ids else "pending",
            }
            for job in jobs
        ],
    }


def run_benchmark(
    config_path: str | Path,
    *,
    resume: bool = False,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Run every configured experiment, safely resuming only exact verified matches."""
    config = load_config(config_path, project_root=project_root, require_inputs=True)
    inputs = _load_inputs(
        config.paths.prepared,
        config.paths.embeddings,
        config.dataset,
        config.encoder,
        source_path=config.paths.source,
    )
    identity = _identity(config, inputs)
    jobs = build_experiment_matrix(config)
    output_dir = config.paths.output_dir
    runs_dir = output_dir / "runs"
    matrix_path = output_dir / "matrix.json"
    if matrix_path.exists():
        existing = _read_object(matrix_path)
        expected_jobs = [job.as_dict() for job in jobs]
        previous_jobs = [
            {key: value for key, value in item.items() if key != "status"}
            for item in existing.get("jobs", [])
            if isinstance(item, dict)
        ]
        if existing.get("identity") != identity or previous_jobs != expected_jobs:
            raise ValueError("existing matrix provenance/config differs; refusing stale results")
        if not resume:
            raise FileExistsError(
                "matrix output already exists; pass --resume to verify and reuse it"
            )
    elif runs_dir.exists() and any(runs_dir.iterdir()):
        raise ValueError("runs exist without a matrix manifest; refusing unknown/stale artifacts")
    elif output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("output directory is not empty and has no matrix manifest")

    output_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)
    if not matrix_path.exists():
        _atomic_json(matrix_path, _matrix_document(config, identity, jobs, set()))

    graph_cache: dict[int, tuple[np.ndarray, np.ndarray, dict[str, Any], np.ndarray | None]] = {}
    if config.training["learners"]:
        for k in config.graph["k_values"]:
            edge_index, edge_weight = build_knn_graph(
                inputs.features, k=k, threshold=config.graph["cosine_threshold"]
            )
            stats = _graph_statistics(edge_index, edge_weight, len(inputs.rows))
            positional = None
            if "gt" in config.training["learners"]:
                positional = laplacian_positional_encoding(
                    edge_index,
                    num_nodes=len(inputs.rows),
                    dimensions=config.graph["lpe_dimensions"],
                    edge_weight=edge_weight,
                )
            graph_cache[k] = (edge_index, edge_weight, stats, positional)

    torch.set_num_threads(config.training["threads"])
    complete_ids: set[str] = set()
    for job in jobs:
        final_dir = runs_dir / job.experiment_id
        partial_dir = runs_dir / f".{job.experiment_id}.partial"
        if partial_dir.exists():
            shutil.rmtree(partial_dir)
        if final_dir.exists():
            if not resume:
                raise FileExistsError(f"experiment output exists: {final_dir}")
            if not (final_dir / "completion.json").is_file():
                shutil.rmtree(final_dir)
            else:
                _verify_complete_run(final_dir, identity=identity, job=job)
                complete_ids.add(job.experiment_id)
                _atomic_json(matrix_path, _matrix_document(config, identity, jobs, complete_ids))
                continue
        try:
            _write_run(config, job, inputs, identity, partial_dir, graph_cache)
            _verify_complete_run(partial_dir, identity=identity, job=job)
            os.replace(partial_dir, final_dir)
        except BaseException:
            if partial_dir.exists():
                shutil.rmtree(partial_dir)
            raise
        _verify_complete_run(final_dir, identity=identity, job=job)
        complete_ids.add(job.experiment_id)
        _atomic_json(matrix_path, _matrix_document(config, identity, jobs, complete_ids))

    return {
        "name": config.name,
        "output_dir": output_dir,
        "completed_count": len(complete_ids),
        "pending_count": len(jobs) - len(complete_ids),
        "experiment_ids": [job.experiment_id for job in jobs],
    }


def _replay_finetuned_probabilities(
    checkpoint_path: Path,
    rows: list[dict[str, Any]],
    raw_config: dict[str, Any],
    seed: int,
) -> np.ndarray:
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    from graphhall.baselines import _EncoderClassifier, _model_inputs, _tokenize_rows
    from graphhall.runtime import resolve_device

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    tuning = raw_config["finetuned_deberta"]
    checkpoint_config = checkpoint.get("config")
    architecture = checkpoint.get("architecture")
    if not isinstance(checkpoint_config, dict) or not isinstance(architecture, dict):
        raise ValueError("fine-tuned checkpoint is missing its config or architecture")
    if (
        checkpoint.get("model_name") != tuning["model_name"]
        or checkpoint.get("requested_revision") != tuning["revision"]
        or checkpoint_config.get("model_name") != tuning["model_name"]
        or checkpoint_config.get("requested_revision") != tuning["revision"]
    ):
        raise ValueError("fine-tuned checkpoint model identity does not match effective config")
    resolved_revision = checkpoint.get("resolved_revision")
    if (
        not isinstance(resolved_revision, str)
        or not resolved_revision
        or checkpoint_config.get("resolved_revision") != resolved_revision
    ):
        raise ValueError("fine-tuned checkpoint resolved revision is inconsistent")
    expected_settings = {
        "batch_size": tuning["batch_size"],
        "max_length": tuning["max_length"],
        "epochs": tuning["epochs"],
        "learning_rate": tuning["learning_rate"],
        "gradient_accumulation": tuning["gradient_accumulation"],
        "max_steps": tuning["max_steps"],
        "seed": seed,
    }
    if any(checkpoint_config.get(key) != value for key, value in expected_settings.items()):
        raise ValueError("fine-tuned checkpoint training config does not match experiment config")
    full_training = tuning["max_steps"] is None
    if checkpoint_config.get("full_training") is not full_training or checkpoint_config.get(
        "not_full_training"
    ) is not (not full_training):
        raise ValueError("fine-tuned checkpoint training completeness flags are inconsistent")
    if (
        architecture.get("pooling") != "last_hidden_state[:,0,:]"
        or architecture.get("classifier") != "linear"
        or architecture.get("num_labels") != 2
        or architecture.get("dropout") != 0.1
    ):
        raise ValueError(
            "fine-tuned checkpoint architecture does not match the supported classifier"
        )

    encoder_config_data = checkpoint_config.get("encoder_config")
    if not isinstance(encoder_config_data, dict) or not isinstance(
        encoder_config_data.get("model_type"), str
    ):
        raise ValueError("fine-tuned checkpoint has no reconstructable encoder config")
    model_type = encoder_config_data["model_type"]
    encoder_config = AutoConfig.for_model(
        model_type,
        **{key: value for key, value in encoder_config_data.items() if key != "model_type"},
    )
    encoder = AutoModel.from_config(encoder_config)
    encoder_class = f"{type(encoder).__module__}.{type(encoder).__qualname__}"
    if architecture.get("encoder_class") != encoder_class:
        raise ValueError("fine-tuned checkpoint encoder class does not match saved architecture")
    hidden_size = architecture.get("hidden_size")
    if isinstance(hidden_size, bool) or not isinstance(hidden_size, int) or hidden_size < 1:
        raise ValueError("fine-tuned checkpoint hidden_size is invalid")
    model = _EncoderClassifier(encoder, hidden_size)
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("fine-tuned checkpoint has no classifier state_dict")
    model.load_state_dict(state_dict, strict=True)

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tuning["model_name"],
            revision=resolved_revision,
            use_fast=False,
            local_files_only=True,
        )
    except OSError as exc:
        raise ValueError(
            "fine-tuned checkpoint replay requires the pinned tokenizer in local cache"
        ) from exc
    tokenized = _tokenize_rows(tokenizer, rows, tuning["max_length"], tuning["batch_size"])
    device = resolve_device(raw_config["encoder"]["device"])
    model = model.to(device).eval()
    torch.set_num_threads(raw_config["training"]["threads"])
    use_amp = device.type == "cuda" and checkpoint_config.get("mixed_precision") is True
    probabilities = np.empty((len(rows), 2), dtype=np.float64)
    all_indices = torch.arange(len(rows), dtype=torch.long)
    with torch.no_grad():
        for start in range(0, len(rows), tuning["batch_size"]):
            row_indices = all_indices[start : start + tuning["batch_size"]]
            context = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if use_amp
                else nullcontext()
            )
            with context:
                logits = model(_model_inputs(tokenized, row_indices, device))
            if logits.shape != (len(row_indices), 2) or not torch.isfinite(logits).all():
                raise ValueError("fine-tuned checkpoint produced invalid replay logits")
            batch_probabilities = torch.softmax(logits.float(), dim=1)
            probabilities[row_indices.numpy()] = batch_probabilities.to("cpu").numpy()
    return probabilities


def replay_predictions(
    run_dir: str | Path,
    *,
    learner: str | None = None,
) -> dict[str, Any]:
    """Restore a completed transductive experiment and compare saved probabilities."""
    from graphhall.baselines import run_knn_baseline
    from graphhall.models import create_model
    from graphhall.runtime import resolve_device

    directory = Path(run_dir).resolve()
    manifest = _verify_complete_run(directory)
    job_data = manifest.get("job")
    if not isinstance(job_data, dict):
        raise ValueError("completion manifest does not include the experiment job")
    job = ExperimentJob(job_data.get("learner"), job_data.get("seed"), job_data.get("k"))
    if job.experiment_id != manifest.get("experiment_id"):
        raise ValueError("completion manifest job ID is inconsistent")
    if learner is not None and learner != job.learner:
        raise ValueError(
            f"requested learner {learner!r} does not match run learner {job.learner!r}"
        )
    effective = _read_object(directory / "effective_config.json")
    raw = effective.get("config")
    if not isinstance(raw, dict) or _canonical_sha256(raw) != manifest["identity"].get(
        "config_sha256"
    ):
        raise ValueError("effective config hash does not match completion provenance")
    input_paths = effective.get("input_paths")
    if not isinstance(input_paths, dict):
        raise ValueError("effective config is missing input paths")
    prepared_path = Path(input_paths["prepared"])
    embeddings_path = Path(input_paths["embeddings"])
    source_path = Path(input_paths["source"]) if input_paths.get("source") else None
    inputs = _load_inputs(
        prepared_path,
        embeddings_path,
        raw["dataset"],
        raw["encoder"],
        source_path=source_path,
    )
    identity = _identity_from_raw(raw, inputs)
    if identity != manifest.get("identity"):
        raise ValueError("current input/config/model hashes do not match replay provenance")
    metrics = _read_object(directory / "metrics.json")
    if metrics.get("experiment_id") != job.experiment_id:
        raise ValueError("metrics experiment_id does not match completion manifest")
    saved_rows = _read_prediction_rows(directory / "predictions.csv", inputs.rows)
    if job.learner == "finetuned_deberta":
        checkpoint_path = directory / "finetuned-model" / "checkpoint.pt"
        if not checkpoint_path.is_file():
            raise ValueError("fine-tuned checkpoint is missing from the completed experiment")
        probabilities = _replay_finetuned_probabilities(checkpoint_path, inputs.rows, raw, job.seed)
    elif job.learner == "knn":
        if job.k is None:
            raise ValueError("kNN replay requires the recorded voting k")
        fitted = run_knn_baseline(
            inputs.features,
            inputs.labels,
            inputs.train_mask,
            inputs.val_mask,
            inputs.test_mask,
            k=job.k,
        )
        probabilities = np.asarray(fitted["probabilities"], dtype=np.float64)
    else:
        checkpoint_path = directory / "best.pt"
        if not checkpoint_path.is_file():
            raise ValueError("checkpoint is missing for a trainable experiment")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        profile = raw["training"]["model_profile"]
        model_features = inputs.features
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_weight = np.empty((0,), dtype=np.float32)
        if job.learner in {"gcn", "gat", "gt"}:
            if job.k is None:
                raise ValueError("graph learner replay requires recorded k")
            with np.load(directory / "graph.npz", allow_pickle=False) as graph_archive:
                edge_index = np.asarray(graph_archive["edge_index"], dtype=np.int64)
                edge_weight = np.asarray(graph_archive["edge_weight"], dtype=np.float32)
            if job.learner == "gt":
                positional = np.load(directory / "lpe.npy", allow_pickle=False)
                if positional.shape != (len(inputs.rows), raw["graph"]["lpe_dimensions"]):
                    raise ValueError("saved GT positional features have an invalid shape")
                model_features = np.concatenate((model_features, positional), axis=1)
        expected_dimension = int(model_features.shape[1])
        if (
            checkpoint.get("learner") != job.learner
            or checkpoint.get("seed") != job.seed
            or checkpoint.get("input_dimension") != expected_dimension
            or checkpoint.get("model_profile") != profile
        ):
            raise ValueError("checkpoint learner, seed, profile, or input dimension mismatch")
        model = create_model(job.learner, input_dim=expected_dimension, model_profile=profile)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        device = resolve_device(raw["encoder"]["device"])
        model = model.to(device).eval()
        torch.set_num_threads(raw["training"]["threads"])
        with torch.no_grad():
            logits = model(
                torch.from_numpy(np.asarray(model_features, dtype=np.float32)).to(device),
                torch.as_tensor(edge_index, dtype=torch.long, device=device),
                torch.as_tensor(edge_weight, dtype=torch.float32, device=device),
            )
            probabilities = torch.softmax(logits, dim=1).to("cpu").numpy()
    expected_probabilities = np.asarray(
        [[float(row["probability_0"]), float(row["probability_1"])] for row in saved_rows],
        dtype=np.float64,
    )
    if probabilities.shape != expected_probabilities.shape or not np.isfinite(probabilities).all():
        raise ValueError("replayed probabilities have invalid shape or non-finite values")
    max_error = float(np.max(np.abs(probabilities - expected_probabilities)))
    if not np.allclose(probabilities, expected_probabilities, rtol=0, atol=1e-6):
        raise ValueError(
            f"checkpoint replay probabilities differ from saved predictions (max error {max_error})"
        )
    return {
        "experiment_id": job.experiment_id,
        "learner": job.learner,
        "row_count": len(inputs.rows),
        "verified": True,
        "max_absolute_error": max_error,
        "transductive": True,
    }


def _identity_from_raw(raw: dict[str, Any], inputs: _Inputs) -> dict[str, Any]:
    return {
        "config_sha256": _canonical_sha256(raw),
        "implementation_sha256": _implementation_sha256(),
        "prepared_jsonl_sha256": inputs.hashes["prepared_jsonl_sha256"],
        "embeddings_npz_sha256": inputs.hashes["embeddings_npz_sha256"],
        "prepared_meta_sha256": inputs.hashes["prepared_meta_sha256"],
        "embedding_meta_sha256": inputs.hashes["embedding_meta_sha256"],
        "source_file_sha256": inputs.hashes["source_file_sha256"],
        "model_identity_sha256": inputs.model_identity_sha256,
    }


def _read_prediction_rows(path: Path, rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        saved = list(csv.DictReader(stream))
    if len(saved) != len(rows):
        raise ValueError("saved prediction row count does not match prepared input")
    for actual, expected in zip(saved, rows, strict=True):
        if actual.get("record_id") != expected["record_id"] or actual.get("gold_label") != str(
            expected["label"]
        ):
            raise ValueError("saved prediction IDs/order/labels do not match prepared input")
    return saved
