"""R-GCN ablation encoder with relation-specific message weights."""

from __future__ import annotations

from typing import cast

import torch
from torch import nn
from torch_geometric.nn import Linear

from kgtp.models.hgt import Metadata


class RGCNEncoder(nn.Module):
    """Heterogeneous R-GCN-style encoder with per-relation transformations."""

    def __init__(
        self,
        hidden_channels: int,
        num_layers: int,
        metadata: Metadata,
        *,
        num_bases: int | None = None,
    ) -> None:
        super().__init__()
        del num_bases
        node_types, edge_types = metadata
        self.node_types = node_types
        self.edge_types = edge_types
        self.lin_dict = nn.ModuleDict(
            {node_type: Linear(-1, hidden_channels) for node_type in node_types}
        )
        self.self_lins = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        node_type: nn.Linear(hidden_channels, hidden_channels)
                        for node_type in node_types
                    }
                )
                for _ in range(num_layers)
            ]
        )
        self.rel_lins = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        _edge_key(edge_type): nn.Linear(
                            hidden_channels, hidden_channels, bias=False
                        )
                        for edge_type in edge_types
                    }
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple[str, str, str], torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Encode with relation-specific message passing."""

        z_dict = {
            node_type: self.lin_dict[node_type](features).relu()
            for node_type, features in x_dict.items()
        }
        for self_lins, rel_lins in zip(self.self_lins, self.rel_lins, strict=True):
            self_lin_dict = cast(nn.ModuleDict, self_lins)
            rel_lin_dict = cast(nn.ModuleDict, rel_lins)
            out = {
                node_type: self_lin_dict[node_type](z_dict[node_type])
                for node_type in self.node_types
            }
            counts = {
                node_type: torch.ones(
                    z_dict[node_type].size(0),
                    1,
                    device=z_dict[node_type].device,
                )
                for node_type in self.node_types
            }
            for edge_type in self.edge_types:
                if edge_type not in edge_index_dict:
                    continue
                src_type, _, dst_type = edge_type
                edge_index = edge_index_dict[edge_type]
                if edge_index.numel() == 0:
                    continue
                messages = rel_lin_dict[_edge_key(edge_type)](
                    z_dict[src_type][edge_index[0]]
                )
                out[dst_type].index_add_(0, edge_index[1], messages)
                ones = torch.ones(messages.size(0), 1, device=messages.device)
                counts[dst_type].index_add_(0, edge_index[1], ones)
            z_dict = {
                node_type: (out[node_type] / counts[node_type].clamp_min(1.0)).relu()
                for node_type in self.node_types
            }
        return z_dict


def _edge_key(edge_type: tuple[str, str, str]) -> str:
    return "__".join(edge_type)
