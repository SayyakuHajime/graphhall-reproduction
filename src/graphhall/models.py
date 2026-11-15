from __future__ import annotations

import torch
from torch import Tensor, nn
from torch_geometric.nn import GATConv, GCNConv, TransformerConv


class GraphClassifier(nn.Module):
    def __init__(self, learner: str, input_dim: int) -> None:
        super().__init__()
        self.learner = learner
        if learner == "gcn":
            self.layer = GCNConv(input_dim, 2, add_self_loops=False)
        elif learner == "gat":
            self.layer = GATConv(
                input_dim,
                2,
                heads=4,
                concat=False,
                dropout=0.0,
                add_self_loops=False,
                edge_dim=1,
            )
        elif learner == "gt":
            self.layer = TransformerConv(
                input_dim,
                2,
                heads=4,
                concat=False,
                beta=True,
                dropout=0.1,
                edge_dim=1,
                root_weight=True,
            )
        else:
            raise ValueError(f"unknown learner {learner!r}; expected gcn, gat, or gt")

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor) -> Tensor:
        loop_nodes = torch.arange(x.shape[0], device=x.device)
        self_loops = torch.stack((loop_nodes, loop_nodes))
        edge_index = torch.cat((edge_index, self_loops), dim=1)
        edge_weight = torch.cat((edge_weight, torch.ones(x.shape[0], device=x.device)))
        if self.learner == "gcn":
            return self.layer(x, edge_index, edge_weight)
        return self.layer(x, edge_index, edge_weight.unsqueeze(-1))


def create_model(learner: str, *, input_dim: int) -> GraphClassifier:
    return GraphClassifier(learner, input_dim)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
