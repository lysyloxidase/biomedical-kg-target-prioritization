"""Shared utilities for non-graph baseline scorers."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from typing import Protocol

import numpy as np

from kgtp.eval.metrics import Query, Triple, evaluate_binary_and_ranking


class TripleScorerProtocol(Protocol):
    """Minimal scorer interface consumed by the evaluation protocol."""

    def score(self, triple: Triple) -> float:
        """Return a real-valued compatibility score."""
        ...


def evaluate_model(
    model: TripleScorerProtocol,
    positives: Sequence[Triple],
    *,
    all_known: Sequence[Triple] | set[Triple],
    tail_candidates: Mapping[Query, Sequence[str]],
    negatives_per_positive: int = 1_000,
    seed: int = 13,
) -> dict[str, object]:
    """Evaluate a fitted baseline under the shared OGB-style protocol."""

    return evaluate_binary_and_ranking(
        model.score,
        positives,
        all_known=all_known,
        tail_candidates=tail_candidates,
        negatives_per_positive=negatives_per_positive,
        seed=seed,
    )


def cosine(first: np.ndarray, second: np.ndarray) -> float:
    """Return cosine similarity with zero-vector protection."""

    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(first, second) / denominator)


def hashed_vector(text: str, *, dim: int = 64, signed: bool = True) -> np.ndarray:
    """Return a deterministic lightweight embedding for text or IDs."""

    vector = np.zeros(dim, dtype=float)
    tokens = [token for token in text.lower().replace("_", " ").split() if token]
    if not tokens:
        tokens = [text or "missing"]
    for token in tokens:
        digest = hashlib.sha256(token.encode()).digest()
        index = int.from_bytes(digest[:8], byteorder="big") % dim
        sign = -1.0 if signed and digest[8] % 2 else 1.0
        vector[index] += sign
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector


def node_feature(
    node_id: str,
    node_features: Mapping[str, Sequence[float]] | None,
    *,
    dim: int = 16,
) -> np.ndarray:
    """Return explicit node features or a deterministic ID embedding fallback."""

    if node_features is not None and node_id in node_features:
        return np.asarray(node_features[node_id], dtype=float)
    return hashed_vector(node_id, dim=dim)


def sigmoid(value: float) -> float:
    """Numerically stable sigmoid."""

    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def pair_feature_vector(
    triple: Triple,
    node_features: Mapping[str, Sequence[float]] | None,
    *,
    dim: int = 16,
) -> np.ndarray:
    """Concatenate source, tail, absolute-difference and product features."""

    head, _, tail = triple
    head_features = node_feature(head, node_features, dim=dim)
    tail_features = node_feature(tail, node_features, dim=dim)
    width = min(len(head_features), len(tail_features))
    head_features = head_features[:width]
    tail_features = tail_features[:width]
    return np.concatenate(
        [
            head_features,
            tail_features,
            np.abs(head_features - tail_features),
            head_features * tail_features,
        ]
    )
