"""Shared encoder with separate decoder heads per prediction task."""

from __future__ import annotations

from typing import cast

import torch
from torch import nn

from kgtp.models.decoder import DecoderName, EdgeType, decode_edge_type, make_decoder

PREDICTED_EDGE_TYPES: tuple[EdgeType, ...] = (
    ("disease", "associated_with", "gene"),
    ("drug", "targets", "gene"),
    ("gene", "participates_in", "pathway"),
)


class MultiTaskLinkPredictor(nn.Module):
    """Shared GNN encoder with one decoder head per target edge type."""

    def __init__(
        self,
        encoder: nn.Module,
        *,
        hidden_channels: int,
        edge_types: tuple[EdgeType, ...] = PREDICTED_EDGE_TYPES,
        decoder_name: DecoderName = "dot",
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.edge_types = edge_types
        self.decoders = nn.ModuleDict(
            {
                _edge_key(edge_type): make_decoder(
                    decoder_name,
                    hidden_channels=hidden_channels,
                    edge_types=[edge_type],
                )
                for edge_type in edge_types
            }
        )

    def encode(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[EdgeType, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Return per-node-type embeddings."""

        return cast(dict[str, torch.Tensor], self.encoder(x_dict, edge_index_dict))

    def decode(
        self,
        z_dict: dict[str, torch.Tensor],
        edge_type: EdgeType,
        edge_label_index: torch.Tensor,
    ) -> torch.Tensor:
        """Decode one task-specific edge type."""

        return decode_edge_type(
            self.decoders[_edge_key(edge_type)],
            z_dict,
            edge_type,
            edge_label_index,
        )

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[EdgeType, torch.Tensor],
        edge_label_index_dict: dict[EdgeType, torch.Tensor],
    ) -> dict[EdgeType, torch.Tensor]:
        """Return logits for all requested multitask edge heads."""

        z_dict = self.encode(x_dict, edge_index_dict)
        return {
            edge_type: self.decode(z_dict, edge_type, edge_label_index)
            for edge_type, edge_label_index in edge_label_index_dict.items()
        }


def _edge_key(edge_type: EdgeType) -> str:
    return "__".join(edge_type)
