from __future__ import annotations

import torch


def resolve_device(device: str | torch.device = "cpu") -> torch.device:
    """Resolve ``cpu``, ``cuda``, or ``auto`` to an available PyTorch device."""
    requested = str(device).lower()
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return torch.device("cuda")
    raise ValueError(f"unsupported device {requested!r}; expected 'cpu', 'cuda', or 'auto'")
