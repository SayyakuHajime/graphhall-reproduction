from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import numpy as np
import torch

from graphhall.baselines import finetune_classifier
from graphhall.data import split_candidate_records
from graphhall.encoder import encode_prepared
from graphhall.experiments import replay_predictions, run_benchmark
from graphhall.graph import build_knn_graph, laplacian_positional_encoding
from graphhall.reporting import generate_report
from graphhall.train import train_model

HALUEVAL_QA_REVISION = "b7253db3cdaa0ab2c382f92b26b390109174f77e"
HALUEVAL_QA_URL = (
    f"https://raw.githubusercontent.com/RUCAIBox/HaluEval/{HALUEVAL_QA_REVISION}/data/qa_data.json"
)
DEFAULT_MODEL = "microsoft/deberta-v3-base"
DEFAULT_MODEL_REVISION = "8ccc9b6f36199bec6961081d44eb72fb3f7353f3"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    text = Path(path).read_text(encoding="utf-8")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list) or any(not isinstance(row, dict) for row in parsed):
        raise ValueError("dataset must be a JSON array or JSONL objects")
    return parsed


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )


def _download_dataset(cache_path: Path) -> Path:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    request = Request(HALUEVAL_QA_URL, headers={"User-Agent": "graphhall-reproduction/0.1"})
    with urlopen(request, timeout=120) as response, cache_path.open("wb") as output:
        while block := response.read(1024 * 1024):
            output.write(block)
    return cache_path


def prepare_command(args: argparse.Namespace) -> int:
    source = (
        Path(args.source_file) if args.source_file else _download_dataset(Path(args.cache_path))
    )
    records = read_jsonl(source)
    prepared = split_candidate_records(records, max_records=args.max_records, seed=args.seed)
    output = Path(args.output)
    write_jsonl(output, prepared)
    split_counts = Counter(row["split"] for row in prepared)
    label_counts = Counter(row["label"] for row in prepared)
    metadata = {
        "schema_version": 1,
        "dataset": "HaluEval-QA",
        "source_url": None if args.source_file else HALUEVAL_QA_URL,
        "source_revision": None if args.source_file else HALUEVAL_QA_REVISION,
        "source_sha256": sha256_file(source),
        "source_row_count": len(records),
        "original_records": len(prepared),
        "prepared_candidates": len(prepared),
        "seed": args.seed,
        "split_by": "original_record_before_candidate_selection",
        "candidate_policy": (
            "one seeded balanced choice from right_answer/hallucinated_answer per original row"
        ),
        "label_mapping": {"right_answer": 0, "hallucinated_answer": 1},
        "split_counts": dict(split_counts),
        "label_counts": {str(key): value for key, value in label_counts.items()},
    }
    output.with_suffix(".meta.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Prepared {len(prepared)} candidates from {len(prepared)} original QA rows -> {output}")
    return 0


def encode_command(args: argparse.Namespace) -> int:
    metadata = encode_prepared(
        args.input,
        args.output,
        model_name=args.model_name,
        revision=args.revision,
        batch_size=args.batch_size,
        max_length=args.max_length,
        include_knowledge=args.include_knowledge,
        cache_dir=args.cache_dir,
        device=getattr(args, "device", "cpu"),
    )
    print(
        f"Encoded {metadata['row_count']} candidates as {metadata['hidden_size']}-D features "
        f"-> {args.output} (revision {metadata['resolved_revision']})"
    )
    return 0


def run_command(args: argparse.Namespace) -> int:
    rows = read_jsonl(args.input)
    with np.load(args.embeddings, allow_pickle=False) as archive:
        if "embeddings" not in archive or "record_ids" not in archive:
            raise ValueError("embedding NPZ must contain embeddings and record_ids arrays")
        embeddings = np.asarray(archive["embeddings"], dtype=np.float32)
        record_ids = [str(item) for item in archive["record_ids"].tolist()]
    if embeddings.ndim != 2 or embeddings.shape[0] != len(rows):
        raise ValueError("embeddings must be an N x D matrix aligned with prepared input")
    expected_ids = [str(row["record_id"]) for row in rows]
    if record_ids != expected_ids:
        raise ValueError("embedding record_ids/order do not match prepared input")
    if not np.isfinite(embeddings).all():
        raise ValueError("embeddings must be finite")
    embedding_meta_path = Path(args.embeddings).with_suffix(".meta.json")
    embedding_meta = (
        json.loads(embedding_meta_path.read_text(encoding="utf-8"))
        if embedding_meta_path.exists()
        else {"model": "unknown (feature fixture or no metadata sidecar)"}
    )
    prepared_hash = sha256_file(args.input)
    declared_prepared_hash = embedding_meta.get("prepared_jsonl_sha256")
    if declared_prepared_hash is not None and declared_prepared_hash != prepared_hash:
        raise ValueError("embedding metadata prepared JSONL hash does not match --input")
    declared_row_count = embedding_meta.get("row_count")
    if declared_row_count is not None and int(declared_row_count) != len(rows):
        raise ValueError("embedding metadata row_count does not match prepared input")
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError("prepared input must contain at least one JSON object")
    if any(row.get("split") not in {"train", "val", "test"} for row in rows):
        raise ValueError("prepared input splits must be train, val, or test")
    if any(row.get("label") not in {0, 1} for row in rows):
        raise ValueError("prepared input labels must be binary values 0 or 1")
    if args.k < 1 or args.threads < 1:
        raise ValueError("k and threads must be positive")
    if not np.isfinite(args.threshold):
        raise ValueError("threshold must be finite")
    learners = [part.strip().lower() for part in args.learners.split(",") if part.strip()]
    if not learners or set(learners) - {"gcn", "gat", "gt", "linear"}:
        raise ValueError("--learners must be a comma-separated subset of gcn,gat,gt,linear")
    if len(set(learners)) != len(learners):
        raise ValueError("--learners must not contain duplicates")

    torch.set_num_threads(args.threads)
    edge_index, edge_weight = build_knn_graph(embeddings, k=args.k, threshold=args.threshold)
    incoming_counts = np.bincount(edge_index[1], minlength=len(rows))
    graph_stats = {
        "num_nodes": len(rows),
        "directed_knn_edges": int(edge_index.shape[1]),
        "explicit_self_loops": len(rows),
        "effective_edges_with_self_loops": int(edge_index.shape[1] + len(rows)),
        "query_nodes_without_knn_neighbors": int(np.count_nonzero(incoming_counts == 0)),
        "edge_weight_min": float(edge_weight.min()) if edge_weight.size else None,
        "edge_weight_max": float(edge_weight.max()) if edge_weight.size else None,
    }
    labels = torch.tensor([int(row["label"]) for row in rows], dtype=torch.long)
    splits = [row["split"] for row in rows]
    train_mask = torch.tensor([split == "train" for split in splits])
    val_mask = torch.tensor([split == "val" for split in splits])
    test_mask = torch.tensor([split == "test" for split in splits])
    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    result_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    probability_rows: list[dict[str, Any]] = []
    histories: dict[str, list[dict[str, float | int]]] = {}

    for learner in learners:
        features = embeddings
        if learner == "gt":
            positional = laplacian_positional_encoding(
                edge_index,
                num_nodes=len(rows),
                dimensions=10,
                edge_weight=edge_weight,
            )
            features = np.concatenate((embeddings, positional), axis=1)
        checkpoint = checkpoint_dir / f"{learner}-k{args.k}.pt"
        fitted = train_model(
            learner,
            torch.from_numpy(features),
            edge_index,
            edge_weight,
            labels,
            train_mask,
            val_mask,
            test_mask,
            checkpoint_path=checkpoint,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            max_epochs=args.max_epochs,
            validation_interval=args.validation_interval,
            patience=args.patience,
            seed=args.seed,
            device=getattr(args, "device", "cpu"),
            model_profile=getattr(args, "model_profile", "weighted"),
        )
        histories[learner] = fitted["history"]
        result_rows.append(
            {
                "learner": learner,
                "k": args.k,
                "input_dimension": int(features.shape[1]),
                "edge_count": int(edge_index.shape[1]),
                "parameter_count": fitted["parameter_count"],
                "best_epoch": fitted["best_epoch"],
                "best_validation_loss": fitted["best_validation_loss"],
                "validation_metrics": fitted["validation_metrics"],
                "metrics": fitted["metrics"],
            }
        )
        fitted_probabilities = np.asarray(fitted["probabilities"], dtype=np.float64)
        if (
            fitted_probabilities.shape != (len(rows), 2)
            or not np.isfinite(fitted_probabilities).all()
        ):
            raise ValueError("trained model probabilities must be a finite N x 2 array")
        for row, prediction, probability in zip(
            rows, fitted["predictions"], fitted_probabilities, strict=True
        ):
            prediction_rows.append(
                {
                    "record_id": row["record_id"],
                    "split": row["split"],
                    "gold_label": row["label"],
                    "predicted_label": int(prediction),
                    "learner": learner,
                    "k": args.k,
                    "probability_0": float(probability[0]),
                    "probability_1": float(probability[1]),
                }
            )
            probability_rows.append(
                {
                    "record_id": row["record_id"],
                    "split": row["split"],
                    "gold_label": row["label"],
                    "learner": learner,
                    "k": args.k,
                    "probability_0": float(probability[0]),
                    "probability_1": float(probability[1]),
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    prepared_meta_path = Path(args.input).with_suffix(".meta.json")
    prepared_meta = (
        json.loads(prepared_meta_path.read_text(encoding="utf-8"))
        if prepared_meta_path.exists()
        else {}
    )
    config = {
        "dataset": "HaluEval-QA",
        "input_jsonl": str(Path(args.input).resolve()),
        "embeddings_npz": str(Path(args.embeddings).resolve()),
        "prepared_jsonl_sha256": prepared_hash,
        "embeddings_sha256": sha256_file(args.embeddings),
        "encoder_model": embedding_meta.get("model"),
        "encoder_revision": embedding_meta.get("resolved_revision"),
        "encoder_model_state_sha256": embedding_meta.get("model_state_sha256"),
        "source_record_count": prepared_meta.get("source_row_count"),
        "selected_original_records": prepared_meta.get("original_records", len(rows)),
        "graph_stats": graph_stats,
        "learners": learners,
        "k": args.k,
        "threshold": args.threshold,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "max_epochs": args.max_epochs,
        "validation_interval": args.validation_interval,
        "patience": args.patience,
        "seed": args.seed,
        "threads": args.threads,
        "device": getattr(args, "device", "cpu"),
        "model_profile": getattr(args, "model_profile", "weighted"),
        "split_counts": dict(Counter(splits)),
        "directed_edge_orientation": "neighbor_source_to_query_target",
        "train_loss_labels": "train split only",
        "checkpoint_selection": "minimum validation loss",
        "run_type": args.run_type,
    }
    provenance = {
        "dataset_source_sha256": prepared_meta.get("source_sha256"),
        "prepared_jsonl_sha256": prepared_hash,
        "embeddings_sha256": sha256_file(args.embeddings),
        "model_revision": embedding_meta.get("resolved_revision"),
        "model_state_sha256": embedding_meta.get("model_state_sha256"),
        "encoder": embedding_meta,
        "config_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True).encode("utf-8")
        ).hexdigest(),
    }
    metrics = {
        "schema_version": 1,
        "run_type": args.run_type,
        "sample_count": len(rows),
        "test_count": int(test_mask.sum().item()),
        "results": result_rows,
        "note": "Smoke metrics are not benchmark results." if args.run_type == "smoke" else None,
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "record_id",
                "split",
                "gold_label",
                "predicted_label",
                "learner",
                "k",
                "probability_0",
                "probability_1",
            ],
        )
        writer.writeheader()
        writer.writerows(prediction_rows)
    with (output_dir / "probabilities.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "record_id",
                "split",
                "gold_label",
                "learner",
                "k",
                "probability_0",
                "probability_1",
            ],
        )
        writer.writeheader()
        writer.writerows(probability_rows)
    (output_dir / "histories.json").write_text(
        json.dumps(histories, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(result_rows)} learner result(s) to {output_dir} ({args.run_type})")
    return 0


def smoke_command(args: argparse.Namespace) -> int:
    if not 10 <= args.max_records <= 80:
        raise ValueError("max_records must be in [10, 80] for smoke runs")
    if not 1 <= args.max_epochs <= 20:
        raise ValueError("max_epochs must be in [1, 20] for smoke runs")
    if not 1 <= args.max_length <= 96:
        raise ValueError("max_length must be in [1, 96] for smoke runs")
    if args.batch_size < 1 or args.threads < 1 or args.validation_interval < 1:
        raise ValueError("batch_size, threads, and validation_interval must be positive")
    if args.k < 1 or args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("k and learning_rate must be positive; weight_decay cannot be negative")
    # Apply the budget before the encoder, not only when the graph learner starts.
    torch.set_num_threads(args.threads)
    output_dir = Path(args.output_dir)
    prepared_path = output_dir / "prepared" / "halueval-qa.jsonl"
    embeddings_path = output_dir / "embeddings" / "halueval-qa.npz"
    run_path = output_dir / "run"
    prepare_command(
        argparse.Namespace(
            source_file=args.source_file,
            cache_path=args.cache_path,
            output=str(prepared_path),
            max_records=args.max_records,
            seed=args.seed,
        )
    )
    encode_command(
        argparse.Namespace(
            input=str(prepared_path),
            output=str(embeddings_path),
            model_name=args.model_name,
            revision=args.revision,
            batch_size=args.batch_size,
            max_length=args.max_length,
            include_knowledge=args.include_knowledge,
            cache_dir=args.cache_dir,
            device=args.device,
        )
    )
    return run_command(
        argparse.Namespace(
            input=str(prepared_path),
            embeddings=str(embeddings_path),
            output_dir=str(run_path),
            learners=args.learners,
            k=args.k,
            threshold=args.threshold,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            max_epochs=args.max_epochs,
            validation_interval=args.validation_interval,
            patience=args.patience,
            seed=args.seed,
            threads=args.threads,
            device=args.device,
            model_profile=args.model_profile,
            run_type="smoke",
        )
    )


def benchmark_command(args: argparse.Namespace) -> int:
    result = run_benchmark(args.config, resume=args.resume)
    print(
        f"Benchmark {result['name']}: {result['completed_count']} completed, "
        f"{result['pending_count']} pending -> {result['output_dir']}"
    )
    return 0


def report_command(args: argparse.Namespace) -> int:
    result = generate_report(args.input_dir, args.output_dir)
    print(
        f"Report: {result['completed_count']} completed, {result['pending_count']} pending; "
        f"CSV {result['csv_path']}, Markdown {result['markdown_path']}"
    )
    return 0


def predict_command(args: argparse.Namespace) -> int:
    result = replay_predictions(args.run_dir, learner=args.learner)
    print(
        f"Replay verified for {result['experiment_id']} ({result['row_count']} transductive rows); "
        f"max probability error={result['max_absolute_error']:.3g}"
    )
    return 0


def finetune_command(args: argparse.Namespace) -> int:
    result = finetune_classifier(
        args.input,
        args.output_dir,
        model_name=args.model_name,
        revision=args.revision,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
        gradient_accumulation=args.gradient_accumulation,
        max_steps=args.max_steps,
        cache_dir=args.cache_dir,
    )
    run_kind = "capped integration" if result.get("not_full_training") else "full training schedule"
    print(
        f"Fine-tuned DeBERTa completed {run_kind}; best epoch={result['best_epoch']}, "
        f"checkpoint={result['checkpoint_path']}"
    )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="graphhall", description="Local GraphHall research CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", help="download/prepare HaluEval-QA")
    prepare.add_argument(
        "--source-file", help="use a local JSON/JSONL source instead of downloading"
    )
    prepare.add_argument("--cache-path", default="data/raw/qa_data.jsonl")
    prepare.add_argument("--output", default="data/processed/halueval-qa.jsonl")
    prepare.add_argument("--max-records", type=int, default=10_000)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.set_defaults(handler=prepare_command)

    encode = subparsers.add_parser("encode", help="encode prepared QA pairs with frozen DeBERTa")
    encode.add_argument("--input", required=True, help="prepared QA JSONL")
    encode.add_argument("--output", default="data/processed/halueval-qa-embeddings.npz")
    encode.add_argument("--model-name", default=DEFAULT_MODEL)
    encode.add_argument("--revision", default=DEFAULT_MODEL_REVISION)
    encode.add_argument("--batch-size", type=int, default=4)
    encode.add_argument("--max-length", type=int, default=96)
    encode.add_argument("--include-knowledge", action="store_true")
    encode.add_argument("--cache-dir")
    encode.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    encode.set_defaults(handler=encode_command)

    run = subparsers.add_parser(
        "run", help="train and evaluate graph learners from cached embeddings"
    )
    run.add_argument("--input", required=True, help="prepared QA JSONL")
    run.add_argument("--embeddings", required=True, help="encoder embeddings NPZ")
    run.add_argument("--output-dir", default="results/run")
    run.add_argument("--learners", default="gcn,gat,gt")
    run.add_argument("--k", type=int, default=5)
    run.add_argument("--threshold", type=float, default=0.8)
    run.add_argument("--learning-rate", type=float, default=0.005)
    run.add_argument("--weight-decay", type=float, default=0.0005)
    run.add_argument("--max-epochs", type=int, default=400)
    run.add_argument("--validation-interval", type=int, default=20)
    run.add_argument("--patience", type=int, default=3)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--threads", type=int, default=2)
    run.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    run.add_argument("--model-profile", choices=("weighted", "topology"), default="weighted")
    run.add_argument("--run-type", choices=("research", "smoke"), default="research")
    run.set_defaults(handler=run_command)

    smoke = subparsers.add_parser("smoke", help="prepare, encode, and train a bounded QA smoke")
    smoke.add_argument("--source-file", help="use a local JSON/JSONL source instead of downloading")
    smoke.add_argument("--cache-path", default="data/raw/qa_data.jsonl")
    smoke.add_argument("--output-dir", default="results/smoke-qa-60")
    smoke.add_argument("--max-records", type=int, default=60)
    smoke.add_argument("--model-name", default=DEFAULT_MODEL)
    smoke.add_argument("--revision", default=DEFAULT_MODEL_REVISION)
    smoke.add_argument("--cache-dir")
    smoke.add_argument("--batch-size", type=int, default=4)
    smoke.add_argument("--max-length", type=int, default=96)
    smoke.add_argument("--include-knowledge", action="store_true")
    smoke.add_argument("--learners", default="gcn,gat,gt")
    smoke.add_argument("--k", type=int, default=5)
    smoke.add_argument("--threshold", type=float, default=0.8)
    smoke.add_argument("--learning-rate", type=float, default=0.005)
    smoke.add_argument("--weight-decay", type=float, default=0.0005)
    smoke.add_argument("--max-epochs", type=int, default=20)
    smoke.add_argument("--validation-interval", type=int, default=5)
    smoke.add_argument("--patience", type=int, default=3)
    smoke.add_argument("--seed", type=int, default=42)
    smoke.add_argument("--threads", type=int, default=2)
    smoke.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    smoke.add_argument("--model-profile", choices=("weighted", "topology"), default="weighted")
    smoke.set_defaults(handler=smoke_command)

    benchmark = subparsers.add_parser("benchmark", help="run a strict YAML experiment matrix")
    benchmark.add_argument("--config", required=True, help="executable YAML config path")
    benchmark.add_argument(
        "--resume", action="store_true", help="verify and reuse exact completed jobs"
    )
    benchmark.set_defaults(handler=benchmark_command)

    report = subparsers.add_parser("report", help="aggregate hash-verified experiment artifacts")
    report.add_argument("--input-dir", required=True, help="benchmark output directory")
    report.add_argument("--output-dir", required=True, help="generated report directory")
    report.set_defaults(handler=report_command)

    predict = subparsers.add_parser(
        "predict", help="replay a checkpoint on its exact transductive pool"
    )
    predict.add_argument("--run-dir", required=True, help="completed experiment directory")
    predict.add_argument(
        "--learner",
        choices=("gcn", "gat", "gt", "linear", "knn", "finetuned_deberta"),
    )
    predict.set_defaults(handler=predict_command)

    finetune = subparsers.add_parser("finetune", help="fine-tune DeBERTa with a CLS classifier")
    finetune.add_argument("--input", required=True, help="prepared QA JSONL")
    finetune.add_argument("--output-dir", required=True)
    finetune.add_argument("--model-name", default=DEFAULT_MODEL)
    finetune.add_argument("--revision", default=DEFAULT_MODEL_REVISION)
    finetune.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    finetune.add_argument("--batch-size", type=int, default=1)
    finetune.add_argument("--max-length", type=int, default=512)
    finetune.add_argument("--epochs", type=int, default=3)
    finetune.add_argument("--learning-rate", type=float, default=2e-5)
    finetune.add_argument("--seed", type=int, default=42)
    finetune.add_argument("--gradient-accumulation", type=int, default=8)
    finetune.add_argument("--max-steps", type=int)
    finetune.add_argument("--cache-dir")
    finetune.set_defaults(handler=finetune_command)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.handler(args)
