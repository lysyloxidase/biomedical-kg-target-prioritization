"""Disjoint link-prediction supervision splits for PyG ``HeteroData``."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch
    from torch_geometric.data import HeteroData

from kgtp.data.common import PathLike
from kgtp.hetero.negatives import (
    EdgePair,
    edge_index_to_pairs,
    sample_negative_edges,
)

EdgeType = tuple[str, str, str]

DEFAULT_EDGE_TYPES: tuple[EdgeType, ...] = (
    ("disease", "associated_with", "gene"),
    ("drug", "targets", "gene"),
    ("gene", "participates_in", "pathway"),
)

DEFAULT_REV_EDGE_TYPES: tuple[EdgeType, ...] = (
    ("gene", "rev_associated_with", "disease"),
    ("gene", "rev_targets", "drug"),
    ("pathway", "rev_participates_in", "gene"),
)


@dataclass(frozen=True)
class SplitBundle:
    """Train/validation/test split payload with reproducibility metadata."""

    train_data: HeteroData
    val_data: HeteroData
    test_data: HeteroData
    metadata: dict[str, Any]


def disjoint_random_link_split(
    data: HeteroData,
    *,
    seed: int = 13,
    num_val: float | int = 0.1,
    num_test: float | int = 0.1,
    disjoint_train_ratio: float | int = 0.3,
    edge_types: tuple[EdgeType, ...] = DEFAULT_EDGE_TYPES,
    rev_edge_types: tuple[EdgeType, ...] = DEFAULT_REV_EDGE_TYPES,
    negative_strategy: str = "random",
    train_neg_sampling_ratio: float = 1.0,
    eval_neg_sampling_ratio: float = 10.0,
) -> SplitBundle:
    """Create supervision splits disjoint from message-passing target edges."""

    if len(edge_types) != len(rev_edge_types):
        msg = "edge_types and rev_edge_types must have the same length"
        raise ValueError(msg)

    split_parts: dict[EdgeType, dict[str, torch.Tensor]] = {}
    counts: dict[str, dict[str, int]] = {}
    for offset, edge_type in enumerate(edge_types):
        edge_index = data[edge_type].edge_index
        parts = _partition_positive_edges(
            edge_index,
            seed=seed + offset,
            num_val=num_val,
            num_test=num_test,
            disjoint_train_ratio=disjoint_train_ratio,
        )
        split_parts[edge_type] = parts
        counts[_edge_type_key(edge_type)] = {
            name: int(value.size(1)) for name, value in parts.items()
        }

    train_data = _message_graph(data, split_parts, edge_types, rev_edge_types, "train")
    val_data = _message_graph(data, split_parts, edge_types, rev_edge_types, "val")
    test_data = _message_graph(data, split_parts, edge_types, rev_edge_types, "test")

    used_negatives: dict[EdgeType, set[EdgePair]] = {
        edge_type: set() for edge_type in edge_types
    }
    for split_index, (split_name, split_data, ratio) in enumerate(
        (
            ("train", train_data, train_neg_sampling_ratio),
            ("val", val_data, eval_neg_sampling_ratio),
            ("test", test_data, eval_neg_sampling_ratio),
        )
    ):
        for edge_offset, edge_type in enumerate(edge_types):
            positive_edges = split_parts[edge_type][_positive_part_name(split_name)]
            full_positive_edges = data[edge_type].edge_index
            negative_edges = _negative_edges_for_split(
                data,
                edge_type,
                positive_edges,
                full_positive_edges,
                ratio=ratio,
                seed=seed + 10_000 * split_index + edge_offset,
                strategy=negative_strategy,
                used_edges=used_negatives[edge_type],
                restrict_to_positive_nodes=split_name in {"val", "test"},
            )
            used_negatives[edge_type].update(edge_index_to_pairs(negative_edges))
            _attach_edge_labels(split_data, edge_type, positive_edges, negative_edges)

        split_data.split_seed = seed
        split_data.split_name = split_name

    metadata = {
        "seed": seed,
        "num_val": num_val,
        "num_test": num_test,
        "disjoint_train_ratio": disjoint_train_ratio,
        "negative_strategy": negative_strategy,
        "train_neg_sampling_ratio": train_neg_sampling_ratio,
        "eval_neg_sampling_ratio": eval_neg_sampling_ratio,
        "edge_types": [list(edge_type) for edge_type in edge_types],
        "rev_edge_types": [list(edge_type) for edge_type in rev_edge_types],
        "counts": counts,
    }
    return SplitBundle(train_data, val_data, test_data, metadata)


def save_splits(
    bundle: SplitBundle,
    output_dir: PathLike,
    *,
    write_metadata: bool = True,
) -> None:
    """Save split tensors and JSON metadata to disk."""

    import torch

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "train_data": bundle.train_data,
            "val_data": bundle.val_data,
            "test_data": bundle.test_data,
            "metadata": bundle.metadata,
        },
        output / "splits.pt",
    )
    if write_metadata:
        (output / "split_metadata.json").write_text(
            json.dumps(bundle.metadata, indent=2, sort_keys=True),
            encoding="utf-8",
        )


def load_splits(input_dir: PathLike) -> SplitBundle:
    """Reload a split bundle produced by :func:`save_splits`."""

    import torch

    payload = torch.load(Path(input_dir) / "splits.pt", weights_only=False)
    return SplitBundle(
        train_data=payload["train_data"],
        val_data=payload["val_data"],
        test_data=payload["test_data"],
        metadata=payload["metadata"],
    )


def supervision_edge_pairs(
    data: HeteroData, edge_type: EdgeType, *, label: int
) -> set[EdgePair]:
    """Return positive or negative supervision pairs for test assertions."""

    labels = data[edge_type].edge_label
    edge_label_index = data[edge_type].edge_label_index
    mask = labels == float(label)
    return edge_index_to_pairs(edge_label_index[:, mask])


def message_edge_pairs(data: HeteroData, edge_type: EdgeType) -> set[EdgePair]:
    """Return message-passing edge pairs for an edge type."""

    return edge_index_to_pairs(data[edge_type].edge_index)


def _partition_positive_edges(
    edge_index: torch.Tensor,
    *,
    seed: int,
    num_val: float | int,
    num_test: float | int,
    disjoint_train_ratio: float | int,
) -> dict[str, torch.Tensor]:
    import torch

    edge_count = int(edge_index.size(1))
    if edge_count == 0:
        empty = torch.empty((2, 0), dtype=torch.long)
        return {
            "message": empty,
            "train_pos": empty,
            "val_pos": empty,
            "test_pos": empty,
        }

    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(edge_count, generator=generator)
    test_count = _count_from_ratio(num_test, edge_count)
    val_count = _count_from_ratio(num_val, edge_count - test_count)
    test_idx = permutation[:test_count]
    val_idx = permutation[test_count : test_count + val_count]
    train_idx = permutation[test_count + val_count :]

    supervision_count = _count_from_ratio(disjoint_train_ratio, int(train_idx.numel()))
    if (
        supervision_count == 0
        and train_idx.numel() > 0
        and float(disjoint_train_ratio) > 0
    ):
        supervision_count = 1
    if supervision_count >= train_idx.numel() and train_idx.numel() > 1:
        supervision_count = int(train_idx.numel()) - 1

    train_supervision_idx = train_idx[:supervision_count]
    train_message_idx = train_idx[supervision_count:]

    return {
        "message": edge_index[:, train_message_idx].contiguous(),
        "train_pos": edge_index[:, train_supervision_idx].contiguous(),
        "val_pos": edge_index[:, val_idx].contiguous(),
        "test_pos": edge_index[:, test_idx].contiguous(),
    }


def _message_graph(
    data: HeteroData,
    split_parts: dict[EdgeType, dict[str, torch.Tensor]],
    edge_types: tuple[EdgeType, ...],
    rev_edge_types: tuple[EdgeType, ...],
    split_name: str,
) -> HeteroData:
    del split_name
    out = copy.deepcopy(data)
    for edge_type, rev_edge_type in zip(edge_types, rev_edge_types, strict=True):
        message_edges = split_parts[edge_type]["message"]
        out[edge_type].edge_index = message_edges
        if rev_edge_type in out.edge_types:
            out[rev_edge_type].edge_index = message_edges.flip(0)
    return out


def _negative_edges_for_split(
    data: HeteroData,
    edge_type: EdgeType,
    positive_edges: torch.Tensor,
    full_positive_edges: torch.Tensor,
    *,
    ratio: float,
    seed: int,
    strategy: str,
    used_edges: set[EdgePair],
    restrict_to_positive_nodes: bool,
) -> torch.Tensor:
    import torch

    source_type, _, target_type = edge_type
    positive_count = int(positive_edges.size(1))
    negative_count = round(positive_count * ratio)
    if positive_count == 0 or negative_count == 0:
        return torch.empty((2, 0), dtype=torch.long)

    allowed_sources: list[int] | None = None
    allowed_targets: list[int] | None = None
    if restrict_to_positive_nodes:
        allowed_sources = sorted({int(value) for value in positive_edges[0].tolist()})
        allowed_targets = sorted({int(value) for value in positive_edges[1].tolist()})

    try:
        negatives = sample_negative_edges(
            num_src_nodes=int(data[source_type].num_nodes),
            num_dst_nodes=int(data[target_type].num_nodes),
            positive_edge_index=full_positive_edges,
            num_samples=negative_count,
            seed=seed,
            strategy=strategy,
            used_edges=used_edges,
            allowed_src_nodes=allowed_sources,
            allowed_dst_nodes=allowed_targets,
        )
    except ValueError:
        if not restrict_to_positive_nodes:
            raise
        negatives = sample_negative_edges(
            num_src_nodes=int(data[source_type].num_nodes),
            num_dst_nodes=int(data[target_type].num_nodes),
            positive_edge_index=full_positive_edges,
            num_samples=negative_count,
            seed=seed,
            strategy=strategy,
            used_edges=used_edges,
        )
    return negatives


def _attach_edge_labels(
    data: HeteroData,
    edge_type: EdgeType,
    positive_edges: torch.Tensor,
    negative_edges: torch.Tensor,
) -> None:
    import torch

    edge_label_index = torch.cat([positive_edges, negative_edges], dim=1)
    positive_labels = torch.ones(positive_edges.size(1), dtype=torch.float32)
    negative_labels = torch.zeros(negative_edges.size(1), dtype=torch.float32)
    data[edge_type].edge_label_index = edge_label_index
    data[edge_type].edge_label = torch.cat([positive_labels, negative_labels])


def _positive_part_name(split_name: str) -> str:
    return f"{split_name}_pos"


def _count_from_ratio(value: float | int, total: int) -> int:
    if total <= 0:
        return 0
    if isinstance(value, int):
        return max(0, min(value, total))
    return max(0, min(round(total * value), total))


def _edge_type_key(edge_type: EdgeType) -> str:
    return "__".join(edge_type)
