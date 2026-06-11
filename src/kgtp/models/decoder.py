"""Link decoders for encoder-decoder link prediction."""

from __future__ import annotations

from typing import Literal, cast

import torch
from torch import nn

EdgeType = tuple[str, str, str]
DecoderName = Literal["dot", "distmult", "mlp"]


class DotProductDecoder(nn.Module):
    """Endpoint dot-product decoder."""

    def forward(
        self,
        z_src: torch.Tensor,
        z_dst: torch.Tensor,
        edge_label_index: torch.Tensor,
        edge_type: EdgeType | None = None,
    ) -> torch.Tensor:
        del edge_type
        src = z_src[edge_label_index[0]]
        dst = z_dst[edge_label_index[1]]
        return (src * dst).sum(dim=-1)


class DistMultDecoder(nn.Module):
    """DistMult decoder with a diagonal parameter per relation."""

    def __init__(
        self, hidden_channels: int, edge_types: list[EdgeType] | tuple[EdgeType, ...]
    ) -> None:
        super().__init__()
        self.relation_index = {
            edge_type: index for index, edge_type in enumerate(edge_types)
        }
        self.weight = nn.Parameter(torch.empty(len(edge_types), hidden_channels))
        nn.init.xavier_uniform_(self.weight)

    def forward(
        self,
        z_src: torch.Tensor,
        z_dst: torch.Tensor,
        edge_label_index: torch.Tensor,
        edge_type: EdgeType | None = None,
    ) -> torch.Tensor:
        if edge_type is None:
            msg = "DistMultDecoder requires edge_type"
            raise ValueError(msg)
        relation = self.weight[self.relation_index[edge_type]]
        src = z_src[edge_label_index[0]]
        dst = z_dst[edge_label_index[1]]
        return (src * relation * dst).sum(dim=-1)


class MLPDecoder(nn.Module):
    """Two-layer MLP decoder over concatenated endpoint embeddings."""

    def __init__(self, hidden_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1),
        )

    def forward(
        self,
        z_src: torch.Tensor,
        z_dst: torch.Tensor,
        edge_label_index: torch.Tensor,
        edge_type: EdgeType | None = None,
    ) -> torch.Tensor:
        del edge_type
        src = z_src[edge_label_index[0]]
        dst = z_dst[edge_label_index[1]]
        return cast(torch.Tensor, self.net(torch.cat([src, dst], dim=-1)).view(-1))


def make_decoder(
    name: DecoderName,
    *,
    hidden_channels: int,
    edge_types: list[EdgeType] | tuple[EdgeType, ...],
) -> nn.Module:
    """Construct a selectable link decoder."""

    if name == "dot":
        return DotProductDecoder()
    if name == "distmult":
        return DistMultDecoder(hidden_channels, edge_types)
    if name == "mlp":
        return MLPDecoder(hidden_channels)
    msg = f"Unknown decoder: {name}"
    raise ValueError(msg)


def decode_edge_type(
    decoder: nn.Module,
    z_dict: dict[str, torch.Tensor],
    edge_type: EdgeType,
    edge_label_index: torch.Tensor,
) -> torch.Tensor:
    """Decode scores for one typed edge label tensor."""

    src_type, _, dst_type = edge_type
    return cast(
        torch.Tensor,
        decoder(z_dict[src_type], z_dict[dst_type], edge_label_index, edge_type),
    )
