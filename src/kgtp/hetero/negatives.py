"""Negative sampling strategies for heterogeneous link prediction."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from heapq import heappush, heapreplace
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

    rng = random.Random(seed)
    if strategy == "random":
        selected = _random_candidates(
            source_nodes,
            target_nodes,
            forbidden,
            num_samples=num_samples,
            rng=rng,
        )
    elif strategy == "degree_matched":
        selected = _degree_matched_candidates(
            source_nodes,
            target_nodes,
            forbidden,
            positive_edge_index=positive_edge_index,
            num_samples=num_samples,
            source_degrees=source_degrees,
            target_degrees=target_degrees,
            rng=rng,
        )
    elif strategy == "hard":
        if hard_candidates_by_source is None:
            msg = "Hard sampling requires train-derived hard_candidates_by_source"
            raise ValueError(msg)
        candidates = _hard_candidates(
            source_nodes, hard_candidates_by_source, forbidden
        )
        if len(candidates) < num_samples:
            msg = (
                f"Hard pool has {len(candidates)} eligible pairs; "
                f"{num_samples} requested"
            )
            raise ValueError(msg)
        rng.shuffle(candidates)
        selected = candidates[:num_samples]
    else:
        msg = f"Unknown negative-sampling strategy: {strategy}"
        raise ValueError(msg)
    return pairs_to_edge_index(selected)


def _random_candidates(
    source_nodes: Sequence[int],
    target_nodes: Sequence[int],
    forbidden: set[EdgePair],
    *,
    num_samples: int,
    rng: random.Random,
) -> list[EdgePair]:
    total = len(source_nodes) * len(target_nodes)
    source_set = set(source_nodes)
    target_set = set(target_nodes)
    eligible = total - sum(
        source in source_set and target in target_set for source, target in forbidden
    )
    if eligible < num_samples:
        msg = f"Requested {num_samples} negatives but only {eligible} are eligible"
        raise ValueError(msg)
    selected: set[EdgePair] = set()
    max_attempts = max(100, num_samples * 50)
    attempts = 0
    while len(selected) < num_samples and attempts < max_attempts:
        pair = (rng.choice(source_nodes), rng.choice(target_nodes))
        attempts += 1
        if pair not in forbidden:
            selected.add(pair)
    if len(selected) < num_samples:
        start = rng.randrange(total)
        for offset in range(total):
            index = (start + offset) % total
            pair = (
                source_nodes[index // len(target_nodes)],
                target_nodes[index % len(target_nodes)],
            )
            if pair not in forbidden:
                selected.add(pair)
                if len(selected) == num_samples:
                    break
    return sorted(selected)


def _degree_matched_candidates(
    source_nodes: Sequence[int],
    target_nodes: Sequence[int],
    forbidden: set[EdgePair],
    *,
    positive_edge_index: torch.Tensor,
    num_samples: int,
    source_degrees: torch.Tensor | None,
    target_degrees: torch.Tensor | None,
    rng: random.Random,
) -> list[EdgePair]:
    if source_degrees is None or target_degrees is None:
        msg = "Degree-matched sampling requires source and target train degrees"
        raise ValueError(msg)
    positive_sources = positive_edge_index[0].tolist()
    positive_targets = positive_edge_index[1].tolist()
    source_mean = sum(float(source_degrees[index]) for index in positive_sources) / len(
        positive_sources
    )
    target_mean = sum(float(target_degrees[index]) for index in positive_targets) / len(
        positive_targets
    )
    heap: list[tuple[float, float, EdgePair]] = []
    for source in source_nodes:
        for target in target_nodes:
            pair = (source, target)
            if pair in forbidden:
                continue
            distance = abs(float(source_degrees[source]) - source_mean) + abs(
                float(target_degrees[target]) - target_mean
            )
            item = (-distance, rng.random(), pair)
            if len(heap) < num_samples:
                heappush(heap, item)
            elif item > heap[0]:
                heapreplace(heap, item)
    if len(heap) < num_samples:
        msg = f"Degree-matched pool has only {len(heap)} eligible pairs"
        raise ValueError(msg)
    return sorted(item[2] for item in heap)


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
