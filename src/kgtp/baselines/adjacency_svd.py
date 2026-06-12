"""Adjacency-SVD shallow graph baseline."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from kgtp.baselines.common import cosine
from kgtp.eval.metrics import Triple


class AdjacencySVDBaseline:
    """Embed an undirected train graph with truncated adjacency SVD."""

    def __init__(self, *, dimension: int = 16) -> None:
        self.dimension = dimension
        self.node_index: dict[str, int] = {}
        self.embeddings: dict[str, np.ndarray] = {}

    def fit(self, triples: Sequence[Triple]) -> AdjacencySVDBaseline:
        """Fit deterministic embeddings on train-graph triples."""

        nodes = sorted({node for head, _, tail in triples for node in (head, tail)})
        self.node_index = {node: index for index, node in enumerate(nodes)}
        if not nodes:
            return self
        adjacency = np.zeros((len(nodes), len(nodes)), dtype=float)
        for head, _, tail in triples:
            source = self.node_index[head]
            target = self.node_index[tail]
            adjacency[source, target] = 1.0
            adjacency[target, source] = 1.0
        augmented = adjacency + np.eye(len(nodes), dtype=float)
        left, singular_values, _ = np.linalg.svd(augmented, full_matrices=False)
        width = min(self.dimension, left.shape[1])
        matrix = left[:, :width] * np.sqrt(singular_values[:width])
        self.embeddings = {
            node: matrix[index] for node, index in self.node_index.items()
        }
        return self

    def score(self, triple: Triple) -> float:
        """Return cosine similarity, or zero for an unseen endpoint."""

        head, _, tail = triple
        if head not in self.embeddings or tail not in self.embeddings:
            return 0.0
        return cosine(self.embeddings[head], self.embeddings[tail])
