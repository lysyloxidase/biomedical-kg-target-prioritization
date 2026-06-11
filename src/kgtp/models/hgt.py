"""HGT hero model for heterogeneous link prediction."""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import HGTConv, Linear

Metadata = tuple[list[str], list[tuple[str, str, str]]]


class HGTEncoder(nn.Module):
    """Per-type projection followed by stacked ``HGTConv`` layers."""

    def __init__(
        self,
        hidden: int,
        num_heads: int,
        num_layers: int,
        metadata: Metadata,
    ) -> None:
        super().__init__()
        if hidden % num_heads != 0:
            msg = "HGT hidden channels must be divisible by num_heads"
            raise ValueError(msg)
        self.metadata = metadata
        node_types, _ = metadata
        self.lin_dict = nn.ModuleDict(
            {node_type: Linear(-1, hidden) for node_type in node_types}
        )
        self.convs = nn.ModuleList(
            [
                HGTConv(hidden, hidden, metadata, heads=num_heads)
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple[str, str, str], torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Encode all node types into hidden embeddings."""

        z_dict = {
            node_type: self.lin_dict[node_type](features).tanh()
            for node_type, features in x_dict.items()
        }
        for conv in self.convs:
            z_next = conv(z_dict, edge_index_dict)
            z_dict = {
                node_type: z_next.get(node_type, z_dict[node_type])
                for node_type in z_dict
            }
        return z_dict
