import csv
import json
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from graphhall.baselines import finetune_classifier, run_knn_baseline


def test_knn_test_labels_cannot_change_predictions_or_validation() -> None:
    features = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.9, 0.1], [0.1, 0.9]],
        dtype=np.float32,
    )
    labels = np.asarray([0, 1, 1, 0, 1])
    train = np.asarray([True, True, True, False, False])
    validation = np.asarray([False, False, False, True, False])
    test = np.asarray([False, False, False, False, True])

    first = run_knn_baseline(features, labels, train, validation, test, k=1)
    changed_test_labels = labels.copy()
    changed_test_labels[test] = 1 - changed_test_labels[test]
    second = run_knn_baseline(features, changed_test_labels, train, validation, test, k=1)

    np.testing.assert_array_equal(first["predictions"], second["predictions"])
    np.testing.assert_array_equal(first["probabilities"], second["probabilities"])
    assert first["validation_metrics"] == second["validation_metrics"]
    assert first["parameter_count"] == 0
    assert first["best_epoch"] == 0
    assert first["probabilities"].shape == (5, 2)


def test_knn_excludes_each_training_query_from_its_own_votes() -> None:
    features = np.asarray([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    labels = np.asarray([0, 1, 0, 1])
    train = np.asarray([True, True, False, False])
    validation = np.asarray([False, False, True, False])
    test = np.asarray([False, False, False, True])

    result = run_knn_baseline(features, labels, train, validation, test, k=1)

    np.testing.assert_array_equal(result["probabilities"][0], [0.0, 1.0])
    np.testing.assert_array_equal(result["probabilities"][1], [1.0, 0.0])


def test_knn_chunked_voting_matches_stable_train_only_reference() -> None:
    features = np.asarray([[1.0, 0.0]] * 5)
    labels = np.asarray([0, 1, 1, 0, 1])
    train = np.asarray([True, True, True, False, False])
    validation = np.asarray([False, False, False, True, False])
    test = np.asarray([False, False, False, False, True])
    reference_indices = np.flatnonzero(train)
    for k in (1, 2, 99):
        expected = []
        for query in range(len(features)):
            eligible = reference_indices[reference_indices != query]
            neighbors = eligible[:k]  # All cosine scores tie: lower row ID wins.
            expected.append(np.bincount(labels[neighbors], minlength=2) / len(neighbors))
        for chunk_size in (1, 3):
            result = run_knn_baseline(
                features, labels, train, validation, test, k=k, chunk_size=chunk_size
            )
            np.testing.assert_allclose(result["probabilities"], expected)


class TinyTokenizer:
    def __call__(self, contexts, answers, **kwargs):
        token_ids = [int(context[1:]) % 8 for context in contexts]
        input_ids = torch.tensor([[token_id] for token_id in token_ids], dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}


class TinyEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=4, _commit_hash="tiny-commit")
        self.embeddings = nn.Embedding(8, 4)

    def forward(self, input_ids, attention_mask=None):
        return SimpleNamespace(last_hidden_state=self.embeddings(input_ids))


def test_finetune_updates_encoder_and_restores_best_validation_checkpoint(tmp_path) -> None:
    rows = []
    for index in range(8):
        split = "train" if index < 4 else "val" if index < 6 else "test"
        rows.append(
            {
                "record_id": f"id-{index}",
                "split": split,
                "question": f"q{index}",
                "answer": "candidate answer",
                "label": index % 2,
            }
        )
    prepared_path = tmp_path / "prepared.jsonl"
    prepared_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    encoder = TinyEncoder()
    initial_weights = {
        name: tensor.detach().clone() for name, tensor in encoder.state_dict().items()
    }
    output_dir = tmp_path / "finetuned"

    result = finetune_classifier(
        prepared_path,
        output_dir,
        model_name="tiny-encoder",
        revision="fixed-test-revision",
        tokenizer=TinyTokenizer(),
        model=encoder,
        batch_size=2,
        max_length=8,
        epochs=3,
        learning_rate=0.05,
        gradient_accumulation=1,
        seed=3,
    )

    checkpoint = torch.load(result["checkpoint_path"], map_location="cpu", weights_only=False)
    encoder_state = encoder.state_dict()
    assert any(
        not torch.equal(initial_weights[name], encoder_state[name]) for name in initial_weights
    )
    for name, tensor in encoder_state.items():
        torch.testing.assert_close(tensor, checkpoint["state_dict"][f"encoder.{name}"])
    assert (
        result["best_epoch"] == min(result["history"], key=lambda item: item["val_loss"])["epoch"]
    )
    assert result["probabilities"].shape == (8, 2)
    np.testing.assert_allclose(result["probabilities"].sum(axis=1), 1.0)
    assert result["parameter_count"] > 0
    assert (output_dir / "predictions.csv").is_file()
    assert (output_dir / "metrics.json").is_file()
    assert (output_dir / "history.json").is_file()
    with (output_dir / "predictions.csv").open(newline="", encoding="utf-8") as file:
        predictions = list(csv.DictReader(file))
    assert len(predictions) == len(rows)
    assert predictions[0]["id"] == "id-0"
    assert predictions[-1]["split"] == "test"


def test_finetune_unequal_microbatches_match_one_effective_batch(tmp_path, monkeypatch) -> None:
    import copy

    # Remove dropout noise and Adam's first-step normalization to compare the objective.
    monkeypatch.setattr(nn.Dropout, "forward", lambda self, inputs: inputs)
    monkeypatch.setattr(torch.optim, "AdamW", torch.optim.SGD)
    rows = [
        {
            "record_id": f"id-{index}",
            "split": "train" if index < 5 else "val" if index < 7 else "test",
            "question": f"q{index}",
            "answer": "candidate",
            "label": index % 2,
        }
        for index in range(9)
    ]
    prepared = tmp_path / "prepared.jsonl"
    prepared.write_text("".join(json.dumps(row) + "\n" for row in rows))
    torch.manual_seed(31)
    initial = TinyEncoder()
    states = []
    for batch_size, accumulation in [(5, 1), (2, 3)]:
        result = finetune_classifier(
            prepared,
            tmp_path / f"batch-{batch_size}",
            model_name="tiny",
            revision="fixed",
            tokenizer=TinyTokenizer(),
            model=copy.deepcopy(initial),
            batch_size=batch_size,
            max_length=8,
            epochs=1,
            learning_rate=0.05,
            gradient_accumulation=accumulation,
            seed=3,
        )
        states.append(torch.load(result["checkpoint_path"], weights_only=False)["state_dict"])
    for name in states[0]:
        torch.testing.assert_close(states[0][name], states[1][name], rtol=1e-6, atol=1e-7)


def test_max_steps_counts_optimizer_updates_and_records_capped_training(tmp_path) -> None:
    rows = []
    for index in range(8):
        split = "train" if index < 4 else "val" if index < 6 else "test"
        rows.append(
            {
                "record_id": f"id-{index}",
                "split": split,
                "question": f"q{index}",
                "answer": "candidate answer",
                "label": index % 2,
            }
        )
    prepared_path = tmp_path / "prepared.jsonl"
    prepared_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    result = finetune_classifier(
        prepared_path,
        tmp_path / "capped",
        model_name="tiny-encoder",
        revision="fixed-test-revision",
        tokenizer=TinyTokenizer(),
        model=TinyEncoder(),
        batch_size=1,
        max_length=8,
        epochs=4,
        learning_rate=0.01,
        gradient_accumulation=2,
        max_steps=1,
    )

    checkpoint = torch.load(result["checkpoint_path"], map_location="cpu", weights_only=False)
    assert result["global_step"] == 1
    assert result["not_full_training"] is True
    assert checkpoint["config"]["max_steps_unit"] == "optimizer_update_steps"
    assert checkpoint["config"]["not_full_training"] is True


def test_finetune_test_labels_do_not_change_training_or_validation(tmp_path) -> None:
    rows = []
    for index in range(8):
        split = "train" if index < 4 else "val" if index < 6 else "test"
        rows.append(
            {
                "record_id": f"id-{index}",
                "split": split,
                "question": f"q{index}",
                "answer": "candidate answer",
                "label": index % 2,
            }
        )
    first_path = tmp_path / "first.jsonl"
    second_path = tmp_path / "second.jsonl"
    first_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    changed_rows = [dict(row) for row in rows]
    for row in changed_rows:
        if row["split"] == "test":
            row["label"] = 1 - row["label"]
    second_path.write_text(
        "".join(json.dumps(row) + "\n" for row in changed_rows), encoding="utf-8"
    )
    torch.manual_seed(31)
    first_encoder = TinyEncoder()
    torch.manual_seed(31)
    second_encoder = TinyEncoder()
    options = {
        "model_name": "tiny-encoder",
        "revision": "fixed-test-revision",
        "batch_size": 2,
        "max_length": 8,
        "epochs": 2,
        "learning_rate": 0.02,
        "gradient_accumulation": 1,
        "seed": 7,
    }

    first = finetune_classifier(
        first_path,
        tmp_path / "first-output",
        tokenizer=TinyTokenizer(),
        model=first_encoder,
        **options,
    )
    second = finetune_classifier(
        second_path,
        tmp_path / "second-output",
        tokenizer=TinyTokenizer(),
        model=second_encoder,
        **options,
    )

    assert first["best_epoch"] == second["best_epoch"]
    assert first["best_validation_loss"] == second["best_validation_loss"]
    assert first["validation_metrics"] == second["validation_metrics"]
    np.testing.assert_array_equal(first["predictions"], second["predictions"])
    first_state = torch.load(first["checkpoint_path"], map_location="cpu", weights_only=False)[
        "state_dict"
    ]
    second_state = torch.load(second["checkpoint_path"], map_location="cpu", weights_only=False)[
        "state_dict"
    ]
    for name in first_state:
        torch.testing.assert_close(first_state[name], second_state[name], rtol=0, atol=0)
