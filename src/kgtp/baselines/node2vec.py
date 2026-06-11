"""Node2Vec / DeepWalk-style shallow structure baseline."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from kgtp.baselines.common import cosine, hashed_vector
from kgtp.eval.metrics import Triple


class Node2VecBaseline:
    """Structure-only shallow embeddings via deterministic adjacency SVD."""

    def __init__(self, *, dimension: int = 16) -> None:
        self.dimension = dimension
        self.node_index: dict[str, int] = {}
        self.embeddings: dict[str, np.ndarray] = {}

    def fit(self, triples: Sequence[Triple]) -> Node2VecBaseline:
        """Fit shallow structural embeddings from an undirected KG projection."""

        nodes = sorted({node for head, _, tail in triples for node in (head, tail)})
        self.node_index = {node: index for index, node in enumerate(nodes)}
        if not nodes:
            return self
        adjacency = np.zeros((len(nodes), len(nodes)), dtype=float)
        for head, _, tail in triples:
            i = self.node_index[head]
            j = self.node_index[tail]
            adjacency[i, j] = 1.0
            adjacency[j, i] = 1.0
        augmented = adjacency + np.eye(len(nodes), dtype=float)
        u, singular_values, _ = np.linalg.svd(augmented, full_matrices=False)
        width = min(self.dimension, u.shape[1])
        matrix = u[:, :width] * np.sqrt(singular_values[:width])
        self.embeddings = {
            node: matrix[index] for node, index in self.node_index.items()
        }
        return self

    def score(self, triple: Triple) -> float:
        """Return cosine similarity of shallow structural embeddings."""

        head, _, tail = triple
        return cosine(self._embedding(head), self._embedding(tail))

    def _embedding(self, node_id: str) -> np.ndarray:
        if node_id in self.embeddings:
            return self.embeddings[node_id]
        return hashed_vector(node_id, dim=self.dimension)
