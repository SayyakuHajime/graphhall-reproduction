import pytest
import torch

from graphhall.runtime import resolve_device


def test_auto_selects_cpu_when_cuda_is_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert resolve_device("auto") == torch.device("cpu")


def test_explicit_cuda_fails_clearly_when_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA was requested but is unavailable"):
        resolve_device("cuda")


def test_auto_selects_cuda_when_available(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert resolve_device("auto") == torch.device("cuda")
