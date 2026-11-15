import torch

from graphhall.encoder import encode_prepared, encode_text_pairs


class FakeTokenizer:
    def __init__(self) -> None:
        self.calls = 0
        self.contexts_seen = []

    def __call__(self, contexts, answers, **kwargs):
        self.calls += 1
        self.contexts_seen.extend(contexts)
        values = [len(left) + len(right) for left, right in zip(contexts, answers, strict=True)]
        return {"input_ids": torch.tensor(values, dtype=torch.float32).reshape(-1, 1)}


class FakeModel:
    def __init__(self) -> None:
        self.training = True
        self.requires_grad_state = True
        self.grad_enabled_during_forward = None

    def eval(self):
        self.training = False
        return self

    def to(self, device):
        return self

    def requires_grad_(self, enabled):
        self.requires_grad_state = enabled
        return self

    def __call__(self, input_ids):
        self.grad_enabled_during_forward = torch.is_grad_enabled()
        hidden = torch.cat((input_ids, input_ids + 1, input_ids + 2), dim=1)
        return type("Output", (), {"last_hidden_state": hidden.unsqueeze(1)})()


def test_encoder_returns_frozen_cls_vectors_in_batches() -> None:
    tokenizer = FakeTokenizer()
    model = FakeModel()
    contexts = ["q" * index for index in range(1, 6)]
    answers = ["a"] * 5

    embeddings = encode_text_pairs(contexts, answers, tokenizer, model, batch_size=2, max_length=32)

    assert embeddings.shape == (5, 3)
    assert embeddings.dtype.name == "float32"
    assert tokenizer.calls == 3
    assert not model.training
    assert model.requires_grad_state is False
    assert model.grad_enabled_during_forward is False
    assert embeddings[0, 0] == 2
    assert embeddings[0, 2] == 4


def test_encode_prepared_uses_question_only_by_default_and_saves_metadata(tmp_path) -> None:
    import json

    import numpy as np

    prepared = tmp_path / "prepared.jsonl"
    prepared.write_text(
        json.dumps(
            {
                "record_id": "qa-1",
                "split": "train",
                "question": "Question text",
                "knowledge": "Knowledge payload",
                "answer": "Answer text",
                "label": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "encoded.npz"
    tokenizer = FakeTokenizer()

    metadata = encode_prepared(
        prepared,
        output,
        model_name="local/fake",
        revision="test-revision",
        batch_size=1,
        max_length=64,
        include_knowledge=False,
        tokenizer=tokenizer,
        model=FakeModel(),
    )

    with np.load(output, allow_pickle=False) as archive:
        assert archive["embeddings"].shape == (1, 3)
        assert archive["record_ids"].tolist() == ["qa-1"]
    assert tokenizer.contexts_seen == ["Question text"]
    assert metadata["requested_revision"] == "test-revision"
    assert metadata["pooling"] == "last_hidden_state[:,0,:]"
    assert metadata["include_knowledge"] is False


def test_encode_prepared_requests_sentencepiece_slow_tokenizer(monkeypatch, tmp_path) -> None:
    import json
    from types import SimpleNamespace

    import transformers

    prepared = tmp_path / "prepared.jsonl"
    prepared.write_text(
        json.dumps(
            {
                "record_id": "qa-1",
                "split": "train",
                "question": "Question text",
                "knowledge": "Knowledge payload",
                "answer": "Answer text",
                "label": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    calls = {}
    tokenizer = FakeTokenizer()
    model = FakeModel()
    model.config = SimpleNamespace(_commit_hash="resolved-test")

    def fake_tokenizer_from_pretrained(*args, **kwargs):
        calls["tokenizer"] = kwargs
        return tokenizer

    def fake_model_from_pretrained(*args, **kwargs):
        calls["model"] = kwargs
        return model

    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", fake_tokenizer_from_pretrained
    )
    monkeypatch.setattr(transformers.AutoModel, "from_pretrained", fake_model_from_pretrained)

    encode_prepared(
        prepared,
        tmp_path / "features.npz",
        model_name="microsoft/deberta-v3-base",
        revision="pinned-revision",
        batch_size=1,
        max_length=96,
    )

    assert calls["tokenizer"]["use_fast"] is False
    assert calls["tokenizer"]["revision"] == "pinned-revision"
    assert calls["model"]["revision"] == "pinned-revision"


def test_model_state_sha256_tracks_the_loaded_encoder_weights() -> None:
    from graphhall.encoder import _model_state_sha256

    model = torch.nn.Linear(2, 3, bias=False)
    first = _model_state_sha256(model)
    with torch.no_grad():
        model.weight[0, 0] += 1
    second = _model_state_sha256(model)

    assert first is not None and len(first) == 64
    assert second is not None and second != first
