"""Shared GNN encoder helpers."""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import HGTConv, Linear, RGCNConv, SAGEConv, to_hetero

from kgtp.models.hgt import HGTEncoder, Metadata

__all__ = [
    "HGTConv",
    "HGTEncoder",
    "InputProjection",
    "Linear",
    "Metadata",
    "RGCNConv",
    "SAGEConv",
    "to_hetero",
]


class InputProjection(nn.Module):
    """Per-node-type lazy input projection."""

    def __init__(self, hidden_channels: int, node_types: list[str]) -> None:
        super().__init__()
        self.projections = nn.ModuleDict(
            {node_type: Linear(-1, hidden_channels) for node_type in node_types}
        )

    def forward(self, x_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Project each node type into the shared hidden dimension."""

        return {
            node_type: self.projections[node_type](features).tanh()
            for node_type, features in x_dict.items()
        }
