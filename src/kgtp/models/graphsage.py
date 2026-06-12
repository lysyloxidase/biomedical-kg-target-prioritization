"""GraphSAGE ablation encoders."""

from __future__ import annotations

from typing import cast

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import HeteroConv, Linear, SAGEConv

from kgtp.models.hgt import Metadata


class GraphSAGEEncoder(nn.Module):
    """Heterogeneous GraphSAGE encoder using one ``SAGEConv`` per edge type."""

    def __init__(
        self,
        hidden_channels: int,
        num_layers: int,
        metadata: Metadata,
        *,
        dropout: float = 0.2,
        residual: bool = True,
        normalization: bool = True,
    ) -> None:
        super().__init__()
        node_types, edge_types = metadata
        self.dropout = dropout
        self.residual = residual
        self.lin_dict = nn.ModuleDict(
            {node_type: Linear(-1, hidden_channels) for node_type in node_types}
        )
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                HeteroConv(
                    {
                        edge_type: SAGEConv((-1, -1), hidden_channels)
                        for edge_type in edge_types
                    },
                    aggr="sum",
                )
            )
        self.norms = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        node_type: (
                            nn.LayerNorm(hidden_channels)
                            if normalization
                            else nn.Identity()
                        )
                        for node_type in node_types
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
        """Encode node features with relation-wise GraphSAGE layers."""

        z_dict = {
            node_type: self.lin_dict[node_type](features).relu()
            for node_type, features in x_dict.items()
        }
        for conv, norms in zip(self.convs, self.norms, strict=True):
            norm_dict = cast(nn.ModuleDict, norms)
            z_next = conv(z_dict, edge_index_dict)
            z_dict = {
                node_type: norm_dict[node_type](
                    z_dict[node_type]
                    + F.dropout(
                        F.relu(z_next.get(node_type, z_dict[node_type])),
                        p=self.dropout,
                        training=self.training,
                    )
                    if self.residual
                    else F.dropout(
                        F.relu(z_next.get(node_type, z_dict[node_type])),
                        p=self.dropout,
                        training=self.training,
                    )
                )
                for node_type in z_dict
            }
        return z_dict


class HomogeneousGraphSAGEEncoder(nn.Module):
    """Homogeneous-control GraphSAGE that ignores node/edge type labels."""

    def __init__(
        self,
        hidden_channels: int,
        num_layers: int,
        node_types: list[str],
        *,
        dropout: float = 0.2,
        residual: bool = True,
        normalization: bool = True,
    ) -> None:
        super().__init__()
        self.node_types = node_types
        self.dropout = dropout
        self.residual = residual
        self.lin_dict = nn.ModuleDict(
            {node_type: Linear(-1, hidden_channels) for node_type in node_types}
        )
        self.convs = nn.ModuleList(
            [
                SAGEConv((hidden_channels, hidden_channels), hidden_channels)
                for _ in range(num_layers)
            ]
        )
        self.norms = nn.ModuleList(
            [
                nn.LayerNorm(hidden_channels) if normalization else nn.Identity()
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple[str, str, str], torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Encode by collapsing all nodes and edges into one homogeneous graph."""

        projected = {
            node_type: self.lin_dict[node_type](features).relu()
            for node_type, features in x_dict.items()
        }
        offsets: dict[str, int] = {}
        tensors: list[torch.Tensor] = []
        offset = 0
        for node_type in self.node_types:
            offsets[node_type] = offset
            tensor = projected[node_type]
            tensors.append(tensor)
            offset += tensor.size(0)
        x = torch.cat(tensors, dim=0)
        edge_indices: list[torch.Tensor] = []
        for (src_type, _, dst_type), edge_index in edge_index_dict.items():
            shifted = edge_index.clone()
            shifted[0] += offsets[src_type]
            shifted[1] += offsets[dst_type]
            edge_indices.append(shifted)
        if edge_indices:
            homogeneous_edge_index = torch.cat(edge_indices, dim=1)
        else:
            homogeneous_edge_index = torch.empty(
                (2, 0), dtype=torch.long, device=x.device
            )
        for conv, norm in zip(self.convs, self.norms, strict=True):
            updated = F.dropout(
                F.relu(conv(x, homogeneous_edge_index)),
                p=self.dropout,
                training=self.training,
            )
            x = norm(x + updated if self.residual else updated)

        out: dict[str, torch.Tensor] = {}
        for node_type in self.node_types:
            start = offsets[node_type]
            end = start + projected[node_type].size(0)
            out[node_type] = x[start:end]
        return out
