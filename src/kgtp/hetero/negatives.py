"""Negative sampling strategies for heterogeneous link prediction."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

EdgePair = tuple[int, int]


def edge_index_to_pairs(edge_index: torch.Tensor) -> set[EdgePair]:
    """Convert a ``[2, n]`` edge index tensor to a set of integer pairs."""

    if edge_index.numel() == 0:
        return set()
    return {
        (int(source), int(target))
        for source, target in edge_index.t().detach().cpu().tolist()
    }


def pairs_to_edge_index(pairs: Sequence[EdgePair]) -> torch.Tensor:
    """Convert pairs to a deterministic ``[2, n]`` edge-index tensor."""

    import torch

    if not pairs:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(list(pairs), dtype=torch.long).t().contiguous()


def sample_negative_edges(
    *,
    num_src_nodes: int,
    num_dst_nodes: int,
    positive_edge_index: torch.Tensor,
    num_samples: int,
    seed: int,
    strategy: str = "random",
    forbidden_edges: set[EdgePair] | None = None,
    used_edges: set[EdgePair] | None = None,
    source_degrees: torch.Tensor | None = None,
    target_degrees: torch.Tensor | None = None,
    hard_candidates_by_source: Mapping[int, Sequence[int]] | None = None,
    allowed_src_nodes: Sequence[int] | None = None,
    allowed_dst_nodes: Sequence[int] | None = None,
) -> torch.Tensor:
    """Sample negatives while excluding positives and previously used negatives."""

    import torch

    if num_samples <= 0:
        return torch.empty((2, 0), dtype=torch.long)

    forbidden = set(forbidden_edges or set())
    forbidden.update(edge_index_to_pairs(positive_edge_index))
    forbidden.update(used_edges or set())

    source_nodes = (
        list(allowed_src_nodes)
        if allowed_src_nodes is not None
        else list(range(num_src_nodes))
    )
    target_nodes = (
        list(allowed_dst_nodes)
        if allowed_dst_nodes is not None
        else list(range(num_dst_nodes))
    )

    if strategy == "hard" and hard_candidates_by_source:
        candidates = _hard_candidates(
            source_nodes, hard_candidates_by_source, forbidden
        )
        if len(candidates) < num_samples:
            candidates.extend(
                candidate
                for candidate in _all_candidates(source_nodes, target_nodes, forbidden)
                if candidate not in set(candidates)
            )
    else:
        candidates = _all_candidates(source_nodes, target_nodes, forbidden)

    rng = random.Random(seed)
    rng.shuffle(candidates)

    if strategy == "degree_matched":
        candidates.sort(
            key=lambda pair: _degree_score(pair, source_degrees, target_degrees),
            reverse=True,
        )

    selected = candidates[:num_samples]
    return pairs_to_edge_index(selected)


def _all_candidates(
    source_nodes: Sequence[int],
    target_nodes: Sequence[int],
    forbidden: set[EdgePair],
) -> list[EdgePair]:
    return [
        (source, target)
        for source in source_nodes
        for target in target_nodes
        if (source, target) not in forbidden
    ]


def _hard_candidates(
    source_nodes: Sequence[int],
    hard_candidates_by_source: Mapping[int, Sequence[int]],
    forbidden: set[EdgePair],
) -> list[EdgePair]:
    candidates: list[EdgePair] = []
    for source in source_nodes:
        for target in hard_candidates_by_source.get(source, ()):
            pair = (source, int(target))
            if pair not in forbidden:
                candidates.append(pair)
    return candidates


def _degree_score(
    pair: EdgePair,
    source_degrees: torch.Tensor | None,
    target_degrees: torch.Tensor | None,
) -> float:
    source, target = pair
    source_score = float(source_degrees[source]) if source_degrees is not None else 0.0
    target_score = float(target_degrees[target]) if target_degrees is not None else 0.0
    return source_score + target_score
