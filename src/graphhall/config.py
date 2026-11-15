from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_TOP_LEVEL = {
    "schema_version",
    "name",
    "run_type",
    "paths",
    "dataset",
    "encoder",
    "graph",
    "training",
    "baselines",
    "finetuned_deberta",
    "evaluation",
}
_SECTIONS = {
    "paths": {"prepared", "embeddings", "output_dir", "source"},
    "dataset": {
        "name",
        "source_revision",
        "source_sha256",
        "source_rows",
        "selected_records",
        "split_seed",
        "split_counts",
    },
    "encoder": {
        "model",
        "revision",
        "device",
        "frozen",
        "context_policy",
        "include_knowledge",
        "max_length",
        "batch_size",
        "pooling",
        "hidden_size",
    },
    "graph": {"k_values", "cosine_threshold", "lpe_dimensions"},
    "training": {
        "learners",
        "seeds",
        "learning_rate",
        "weight_decay",
        "max_epochs",
        "validation_interval",
        "patience",
        "threads",
        "model_profile",
        "selection_metric",
    },
    "finetuned_deberta": {
        "enabled",
        "model_name",
        "revision",
        "batch_size",
        "max_length",
        "epochs",
        "learning_rate",
        "seed_from_matrix",
        "gradient_accumulation",
        "max_steps",
    },
    "evaluation": {"positive_label", "protocol", "test_labels_used_for_training"},
}


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False
) -> dict:
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


@dataclass(frozen=True)
class ConfigPaths:
    prepared: Path
    embeddings: Path
    output_dir: Path
    source: Path | None = None


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    run_type: str
    source_path: Path
    project_root: Path
    paths: ConfigPaths
    dataset: dict[str, Any]
    encoder: dict[str, Any]
    graph: dict[str, Any]
    training: dict[str, Any]
    baselines: tuple[str, ...]
    finetuned_deberta: dict[str, Any]
    evaluation: dict[str, Any]
    raw: dict[str, Any]
    config_sha256: str


def _project_root(config_path: Path) -> Path:
    for parent in (config_path.parent, *config_path.parents):
        if (parent / "pyproject.toml").is_file():
            return parent.resolve()
    return config_path.parent.resolve()


def _mapping(value: Any, name: str, allowed: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} keys must be strings")
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"{name} contains unknown keys: {', '.join(sorted(unknown))}")
    missing = allowed - set(value)
    if missing:
        raise ValueError(f"{name} is missing required keys: {', '.join(sorted(missing))}")
    return value


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _integer(value: Any, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _number(value: Any, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        qualifier = f" >= {minimum}" if minimum is not None else ""
        raise ValueError(f"{name} must be finite{qualifier}")
    return result


def _unique_list(value: Any, name: str, *, allowed: set[Any] | None = None) -> list[Any]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list")
    try:
        values = set(value)
    except TypeError as exc:
        raise ValueError(f"{name} entries must be scalar values") from exc
    if len(values) != len(value):
        raise ValueError(f"{name} must not contain duplicates")
    if allowed is not None and values - allowed:
        raise ValueError(f"{name} contains unsupported values: {sorted(values - allowed)}")
    return value


def _resolve_relative_path(value: Any, key: str, project_root: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"paths.{key} must be a non-empty relative path")
    candidate = Path(value)
    if candidate.is_absolute():
        raise ValueError(f"paths.{key} must be relative to the project root")
    resolved = (project_root / candidate).resolve()
    try:
        resolved.relative_to(project_root)
    except ValueError as exc:
        raise ValueError(f"paths.{key} must stay inside the project root") from exc
    return resolved


def _validate_config(parsed: dict[str, Any]) -> None:
    if set(parsed) - _TOP_LEVEL:
        raise ValueError(
            f"config contains unknown keys: {', '.join(sorted(set(parsed) - _TOP_LEVEL))}"
        )
    missing = _TOP_LEVEL - set(parsed)
    if missing:
        raise ValueError(f"config is missing required keys: {', '.join(sorted(missing))}")
    version = parsed["schema_version"]
    if isinstance(version, bool) or not isinstance(version, int) or version != 1:
        raise ValueError("schema_version must be integer 1")
    _string(parsed["name"], "name")
    run_type = _string(parsed["run_type"], "run_type")
    if run_type not in {"full", "multiseed", "integration"}:
        raise ValueError("run_type must be full, multiseed, or integration")

    paths = parsed["paths"]
    if not isinstance(paths, dict) or any(not isinstance(key, str) for key in paths):
        raise ValueError("paths must be a mapping with string keys")
    unknown_paths = set(paths) - _SECTIONS["paths"]
    if unknown_paths:
        raise ValueError(f"paths contains unknown keys: {', '.join(sorted(unknown_paths))}")
    missing_paths = {"prepared", "embeddings", "output_dir"} - set(paths)
    if missing_paths:
        raise ValueError(f"paths is missing required keys: {', '.join(sorted(missing_paths))}")
    for key, value in paths.items():
        _string(value, f"paths.{key}")

    dataset = _mapping(parsed["dataset"], "dataset", _SECTIONS["dataset"])
    _string(dataset["name"], "dataset.name")
    _string(dataset["source_revision"], "dataset.source_revision")
    digest = dataset["source_sha256"]
    if digest is not None and (
        not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)
    ):
        raise ValueError("dataset.source_sha256 must be a lowercase SHA-256 or null")
    source_rows = _integer(dataset["source_rows"], "dataset.source_rows")
    selected = _integer(dataset["selected_records"], "dataset.selected_records")
    if selected > source_rows:
        raise ValueError("dataset.selected_records cannot exceed source_rows")
    _integer(dataset["split_seed"], "dataset.split_seed", minimum=0)
    counts = _mapping(dataset["split_counts"], "dataset.split_counts", {"train", "val", "test"})
    for split, count in counts.items():
        _integer(count, f"dataset.split_counts.{split}")
    if sum(counts.values()) != selected:
        raise ValueError("dataset.split_counts must sum to selected_records")
    if parsed["run_type"] in {"full", "multiseed"} and (
        selected != 10_000 or counts != {"train": 8_000, "val": 1_000, "test": 1_000}
    ):
        raise ValueError(
            "full and multiseed configs require the fixed 10,000-row 8,000/1,000/1,000 split"
        )

    encoder = _mapping(parsed["encoder"], "encoder", _SECTIONS["encoder"])
    _string(encoder["model"], "encoder.model")
    _string(encoder["revision"], "encoder.revision")
    device = _string(encoder["device"], "encoder.device")
    if device not in {"cpu", "cuda", "auto"}:
        raise ValueError("encoder.device must be cpu, cuda, or auto")
    if not isinstance(encoder["frozen"], bool) or not encoder["frozen"]:
        raise ValueError("encoder.frozen must be true for cached feature experiments")
    if encoder["context_policy"] != "question_only" or encoder["include_knowledge"] is not False:
        raise ValueError("encoder must use question_only context with include_knowledge=false")
    _integer(encoder["max_length"], "encoder.max_length")
    _integer(encoder["batch_size"], "encoder.batch_size")
    _string(encoder["pooling"], "encoder.pooling")
    _integer(encoder["hidden_size"], "encoder.hidden_size")
    if parsed["run_type"] in {"full", "multiseed"} and encoder["max_length"] != 512:
        raise ValueError("full and multiseed configs require encoder.max_length=512")

    graph = _mapping(parsed["graph"], "graph", _SECTIONS["graph"])
    ks = _unique_list(graph["k_values"], "graph.k_values")
    for index, item in enumerate(ks):
        _integer(item, f"graph.k_values[{index}]")
    threshold = _number(graph["cosine_threshold"], "graph.cosine_threshold")
    if not -1.0 <= threshold <= 1.0:
        raise ValueError("graph.cosine_threshold must be in [-1, 1]")
    _integer(graph["lpe_dimensions"], "graph.lpe_dimensions")

    training = _mapping(parsed["training"], "training", _SECTIONS["training"])
    _unique_list(training["learners"], "training.learners", allowed={"gcn", "gat", "gt"})
    seeds = _unique_list(training["seeds"], "training.seeds")
    for index, item in enumerate(seeds):
        _integer(item, f"training.seeds[{index}]", minimum=0)
    learning_rate = _number(training["learning_rate"], "training.learning_rate", minimum=0.0)
    if learning_rate == 0:
        raise ValueError("training.learning_rate must be greater than zero")
    _number(training["weight_decay"], "training.weight_decay", minimum=0.0)
    for name in ("max_epochs", "validation_interval", "patience", "threads"):
        _integer(training[name], f"training.{name}")
    model_profile = _string(training["model_profile"], "training.model_profile")
    if model_profile not in {"weighted", "topology"}:
        raise ValueError("training.model_profile must be weighted or topology")
    selection_metric = _string(training["selection_metric"], "training.selection_metric")
    if selection_metric not in {"f1", "accuracy", "auroc", "auprc"}:
        raise ValueError("training.selection_metric must be f1, accuracy, auroc, or auprc")

    _unique_list(parsed["baselines"], "baselines", allowed={"linear", "knn"})
    finetune = _mapping(
        parsed["finetuned_deberta"], "finetuned_deberta", _SECTIONS["finetuned_deberta"]
    )
    if not isinstance(finetune["enabled"], bool) or not isinstance(
        finetune["seed_from_matrix"], bool
    ):
        raise ValueError("finetuned_deberta.enabled and seed_from_matrix must be booleans")
    _string(finetune["model_name"], "finetuned_deberta.model_name")
    _string(finetune["revision"], "finetuned_deberta.revision")
    for name in ("batch_size", "max_length", "epochs", "gradient_accumulation"):
        _integer(finetune[name], f"finetuned_deberta.{name}")
    finetune_lr = _number(finetune["learning_rate"], "finetuned_deberta.learning_rate", minimum=0.0)
    if finetune_lr == 0:
        raise ValueError("finetuned_deberta.learning_rate must be greater than zero")
    if finetune["max_steps"] is not None:
        _integer(finetune["max_steps"], "finetuned_deberta.max_steps")

    evaluation = _mapping(parsed["evaluation"], "evaluation", _SECTIONS["evaluation"])
    positive_label = evaluation["positive_label"]
    if (
        isinstance(positive_label, bool)
        or not isinstance(positive_label, int)
        or positive_label != 1
    ):
        raise ValueError("evaluation.positive_label must be integer 1 (hallucinated answer)")
    protocol = _string(evaluation["protocol"], "evaluation.protocol")
    if protocol != "transductive_full_graph":
        raise ValueError("evaluation.protocol must be transductive_full_graph")
    if evaluation["test_labels_used_for_training"] is not False:
        raise ValueError("evaluation.test_labels_used_for_training must be false")


def load_config(
    path: str | Path,
    *,
    project_root: str | Path | None = None,
    require_inputs: bool = False,
) -> ExperimentConfig:
    """Load the executable YAML matrix and resolve all configured paths safely."""
    source_path = Path(path).resolve()
    try:
        parsed = yaml.load(source_path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML config {source_path}: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("config root must be a mapping")
    _validate_config(parsed)
    root = Path(project_root).resolve() if project_root is not None else _project_root(source_path)
    paths_data = parsed["paths"]
    paths = ConfigPaths(
        prepared=_resolve_relative_path(paths_data["prepared"], "prepared", root),
        embeddings=_resolve_relative_path(paths_data["embeddings"], "embeddings", root),
        output_dir=_resolve_relative_path(paths_data["output_dir"], "output_dir", root),
        source=(
            _resolve_relative_path(paths_data["source"], "source", root)
            if "source" in paths_data
            else None
        ),
    )
    if require_inputs:
        required = [paths.prepared, paths.embeddings]
        if paths.source is not None:
            required.append(paths.source)
        missing_paths = [item for item in required if not item.is_file()]
        if missing_paths:
            raise FileNotFoundError(
                "configured benchmark inputs do not exist: "
                + ", ".join(str(item) for item in missing_paths)
            )
    canonical = json.dumps(parsed, sort_keys=True, separators=(",", ":"), allow_nan=False)
    config_sha256 = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return ExperimentConfig(
        name=parsed["name"],
        run_type=parsed["run_type"],
        source_path=source_path,
        project_root=root,
        paths=paths,
        dataset=parsed["dataset"],
        encoder=parsed["encoder"],
        graph=parsed["graph"],
        training=parsed["training"],
        baselines=tuple(parsed["baselines"]),
        finetuned_deberta=parsed["finetuned_deberta"],
        evaluation=parsed["evaluation"],
        raw=parsed,
        config_sha256=config_sha256,
    )
