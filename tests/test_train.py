from pathlib import Path

import numpy as np
import pytest
import torch

from graphhall.train import train_model


@pytest.mark.parametrize("learner", ["gcn", "gat", "gt"])
def test_test_labels_cannot_change_fitted_checkpoint(tmp_path: Path, learner: str) -> None:
    torch.set_num_threads(2)
    rng = np.random.default_rng(7)
    features = torch.tensor(rng.normal(size=(18, 8)), dtype=torch.float32)
    edge_index = np.asarray([[1, 0, 2, 1, 4, 3, 5, 4], [0, 1, 1, 2, 3, 4, 4, 5]], dtype=np.int64)
    edge_weight = np.ones(edge_index.shape[1], dtype=np.float32)
    labels = torch.tensor([index % 2 for index in range(18)], dtype=torch.long)
    train_mask = torch.tensor([index < 12 for index in range(18)])
    val_mask = torch.tensor([12 <= index < 15 for index in range(18)])
    test_mask = torch.tensor([index >= 15 for index in range(18)])

    first = train_model(
        learner,
        features,
        edge_index,
        edge_weight,
        labels,
        train_mask,
        val_mask,
        test_mask,
        checkpoint_path=tmp_path / "first.pt",
        max_epochs=3,
        validation_interval=1,
        patience=3,
        seed=42,
    )
    changed_labels = labels.clone()
    changed_labels[test_mask] = 1 - changed_labels[test_mask]
    second = train_model(
        learner,
        features,
        edge_index,
        edge_weight,
        changed_labels,
        train_mask,
        val_mask,
        test_mask,
        checkpoint_path=tmp_path / "second.pt",
        max_epochs=3,
        validation_interval=1,
        patience=3,
        seed=42,
    )

    assert Path(first["checkpoint_path"]).exists()
    assert first["best_epoch"] == second["best_epoch"]
    first_state = torch.load(first["checkpoint_path"], weights_only=False)["state_dict"]
    second_state = torch.load(second["checkpoint_path"], weights_only=False)["state_dict"]
    assert first_state.keys() == second_state.keys()
    for name in first_state:
        torch.testing.assert_close(first_state[name], second_state[name], rtol=0, atol=0)
