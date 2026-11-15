from types import SimpleNamespace

import numpy as np
import pytest
import torch

import graphhall.train as train_module
from graphhall.encoder import encode_prepared, encode_text_pairs
from graphhall.models import create_model
from graphhall.train import train_model


class RecordingTokenizer:
    def __call__(self, contexts, answers, **kwargs):
        return {"input_ids": torch.ones((len(contexts), 2), dtype=torch.long)}


class RecordingEncoder:
    def __init__(self) -> None:
        self.device = None
        self.input_device = None

    def to(self, device):
        self.device = torch.device(device)
        return self

    def eval(self):
        return self

    def requires_grad_(self, enabled):
        return self

    def __call__(self, input_ids):
        self.input_device = input_ids.device
        values = input_ids.float().unsqueeze(-1).repeat(1, 1, 3)
        return SimpleNamespace(last_hidden_state=values)


@pytest.mark.parametrize("learner", ["gat", "gt"])
def test_topology_profile_ignores_scalar_edge_weights(learner: str) -> None:
    torch.manual_seed(17)
    model = create_model(learner, input_dim=4, model_profile="topology")
    model.eval()
    features = torch.randn(3, 4)
    edge_index = torch.tensor([[0, 2], [1, 1]])

    first = model(features, edge_index, torch.tensor([0.0, 0.1]))
    second = model(features, edge_index, torch.tensor([100.0, -50.0]))

    torch.testing.assert_close(first, second)


def test_train_model_returns_probabilities_and_persists_model_shape_metadata(tmp_path) -> None:
    features = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    train = torch.tensor([True, True, True, True, False, False, False, False])
    validation = torch.tensor([False, False, False, False, True, True, False, False])
    test = torch.tensor([False, False, False, False, False, False, True, True])
    checkpoint_path = tmp_path / "linear.pt"

    result = train_model(
        "linear",
        features,
        np.empty((2, 0), dtype=np.int64),
        np.empty(0, dtype=np.float32),
        labels,
        train,
        validation,
        test,
        checkpoint_path=checkpoint_path,
        max_epochs=3,
        validation_interval=1,
        device="cpu",
        model_profile="topology",
    )

    assert result["probabilities"].shape == (8, 2)
    np.testing.assert_allclose(result["probabilities"].sum(axis=1), 1.0)
    assert set(result["validation_metrics"]) == {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "auroc",
        "auprc",
        "ece",
        "brier",
        "nll",
    }
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert saved["input_dimension"] == 3
    assert saved["model_profile"] == "topology"
    assert saved["learner"] == "linear"


def test_training_rejects_nonfinite_model_logits(tmp_path, monkeypatch) -> None:
    class NonfiniteModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))

        def forward(self, features, edge_index, edge_weight):
            return (
                torch.full((features.shape[0], 2), float("nan"), device=features.device)
                + self.weight
            )

    monkeypatch.setattr(train_module, "create_model", lambda *args, **kwargs: NonfiniteModel())
    features = torch.ones((6, 2))
    labels = torch.tensor([0, 1, 0, 1, 0, 1])
    train = torch.tensor([True, True, False, False, False, False])
    validation = torch.tensor([False, False, True, True, False, False])
    test = torch.tensor([False, False, False, False, True, True])

    with pytest.raises(FloatingPointError, match="training logits contain non-finite"):
        train_model(
            "linear",
            features,
            np.empty((2, 0), dtype=np.int64),
            np.empty(0),
            labels,
            train,
            validation,
            test,
            checkpoint_path=tmp_path / "unused.pt",
            max_epochs=1,
        )


def test_linear_learner_is_independent_of_graph_edges() -> None:
    model = create_model("linear", input_dim=3)
    features = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])

    actual = model(features, torch.tensor([[0], [1]]), torch.tensor([0.25]))
    without_edges = model(features, torch.empty((2, 0), dtype=torch.long), torch.empty(0))

    torch.testing.assert_close(actual, without_edges)


def test_text_encoder_moves_model_and_token_batches_to_requested_device() -> None:
    model = RecordingEncoder()

    output = encode_text_pairs(
        ["question"],
        ["answer"],
        RecordingTokenizer(),
        model,
        batch_size=1,
        max_length=16,
        device="cpu",
    )

    assert output.shape == (1, 3)
    assert model.device == torch.device("cpu")
    assert model.input_device == torch.device("cpu")


def test_text_encoder_casts_mixed_precision_cls_vectors_back_to_float32() -> None:
    class BFloatEncoder(RecordingEncoder):
        def __call__(self, input_ids):
            self.input_device = input_ids.device
            values = input_ids.float().unsqueeze(-1).repeat(1, 1, 3).to(torch.bfloat16)
            return SimpleNamespace(last_hidden_state=values)

    output = encode_text_pairs(
        ["question"],
        ["answer"],
        RecordingTokenizer(),
        BFloatEncoder(),
        batch_size=1,
        max_length=16,
        device="cpu",
        mixed_precision=True,
    )

    assert output.dtype == np.float32


def test_prepared_encoder_rejects_unavailable_cuda_before_loading_models(
    tmp_path, monkeypatch
) -> None:
    import json

    import transformers

    prepared = tmp_path / "prepared.jsonl"
    prepared.write_text(
        json.dumps(
            {
                "record_id": "row-1",
                "question": "question",
                "answer": "answer",
                "knowledge": "knowledge",
                "label": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    calls = []

    def forbidden_loader(*args, **kwargs):
        calls.append(True)
        raise AssertionError("model download attempted before device validation")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", forbidden_loader)
    monkeypatch.setattr(transformers.AutoModel, "from_pretrained", forbidden_loader)

    with pytest.raises(RuntimeError, match="CUDA was requested but is unavailable"):
        encode_prepared(
            prepared,
            tmp_path / "encoded.npz",
            model_name="local/fake",
            revision="test-revision",
            batch_size=1,
            max_length=8,
            device="cuda",
        )
    assert calls == []
