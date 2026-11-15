import numpy as np
import torch

from graphhall.models import create_model, parameter_count


def test_all_learners_produce_binary_node_logits_with_explicit_self_loops() -> None:
    features = torch.randn(5, 16)
    empty_edges = torch.empty((2, 0), dtype=torch.long)
    empty_weights = torch.empty((0,), dtype=torch.float32)

    for learner in ("gcn", "gat", "gt"):
        model = create_model(learner, input_dim=16)
        logits = model(features, empty_edges, empty_weights)
        assert logits.shape == (5, 2)
        assert torch.isfinite(logits).all()
        assert parameter_count(model) > 0


def test_paper_parameter_counts_match_where_architecture_is_exact() -> None:
    assert parameter_count(create_model("gcn", input_dim=768)) == 1538
    gat_count = parameter_count(create_model("gat", input_dim=768))
    gt_count = parameter_count(create_model("gt", input_dim=778))
    assert gat_count > 0 and np.isfinite(gat_count)
    assert gt_count > 0 and np.isfinite(gt_count)
