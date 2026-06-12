"""True Node2Vec with biased walks and skip-gram negative sampling."""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from collections.abc import Sequence

import numpy as np

from kgtp.baselines.common import cosine, sigmoid
from kgtp.eval.metrics import Triple


class Node2VecBaseline:
    """Learn train-graph embeddings from second-order biased random walks."""

    def __init__(
        self,
        *,
        dimension: int = 16,
        walk_length: int = 10,
        walks_per_node: int = 5,
        context_window: int = 3,
        negative_samples: int = 3,
        p: float = 1.0,
        q: float = 1.0,
        learning_rate: float = 0.025,
        epochs: int = 3,
        seed: int = 13,
    ) -> None:
        if p <= 0 or q <= 0:
            msg = "Node2Vec p and q must be positive"
            raise ValueError(msg)
        self.dimension = dimension
        self.walk_length = walk_length
        self.walks_per_node = walks_per_node
        self.context_window = context_window
        self.negative_samples = negative_samples
        self.p = p
        self.q = q
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.seed = seed
        self.embeddings: dict[str, np.ndarray] = {}
        self.walks: list[list[str]] = []

    def fit(self, triples: Sequence[Triple]) -> Node2VecBaseline:
        """Generate biased walks and optimize a skip-gram objective."""

        adjacency: defaultdict[str, set[str]] = defaultdict(set)
        for head, _, tail in triples:
            adjacency[head].add(tail)
            adjacency[tail].add(head)
        nodes = sorted(adjacency)
        if not nodes:
            return self
        self.walks = self._generate_walks(nodes, adjacency)
        node_index = {node: index for index, node in enumerate(nodes)}
        rng = np.random.default_rng(self.seed)
        input_embeddings = rng.normal(
            0.0, 1.0 / max(1, self.dimension), (len(nodes), self.dimension)
        )
        output_embeddings = np.zeros((len(nodes), self.dimension), dtype=float)
        frequencies = Counter(node for walk in self.walks for node in walk)
        negative_nodes = np.asarray(nodes)
        weights = np.asarray([frequencies[node] ** 0.75 for node in nodes], dtype=float)
        probabilities = weights / weights.sum()
        examples = self._context_examples(self.walks)
        for _ in range(self.epochs):
            rng.shuffle(examples)
            for center, context in examples:
                self._update_pair(
                    input_embeddings,
                    output_embeddings,
                    node_index[center],
                    node_index[context],
                    label=1.0,
                )
                sampled = rng.choice(
                    negative_nodes,
                    size=self.negative_samples,
                    replace=True,
                    p=probabilities,
                )
                for negative in sampled:
                    if str(negative) == context:
                        continue
                    self._update_pair(
                        input_embeddings,
                        output_embeddings,
                        node_index[center],
                        node_index[str(negative)],
                        label=0.0,
                    )
        self.embeddings = {
            node: input_embeddings[index] for node, index in node_index.items()
        }
        return self

    def score(self, triple: Triple) -> float:
        """Return cosine similarity of learned Node2Vec embeddings."""

        head, _, tail = triple
        if head not in self.embeddings or tail not in self.embeddings:
            return 0.0
        return cosine(self.embeddings[head], self.embeddings[tail])

    def hyperparameters(self) -> dict[str, float | int]:
        """Return stored walk and optimization settings."""

        return {
            "dimension": self.dimension,
            "walk_length": self.walk_length,
            "walks_per_node": self.walks_per_node,
            "context_window": self.context_window,
            "negative_samples": self.negative_samples,
            "p": self.p,
            "q": self.q,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "seed": self.seed,
        }

    def _generate_walks(
        self,
        nodes: Sequence[str],
        adjacency: dict[str, set[str]] | defaultdict[str, set[str]],
    ) -> list[list[str]]:
        rng = random.Random(self.seed)
        walks: list[list[str]] = []
        for _ in range(self.walks_per_node):
            shuffled = list(nodes)
            rng.shuffle(shuffled)
            for start in shuffled:
                walk = [start]
                while len(walk) < self.walk_length:
                    current = walk[-1]
                    neighbors = sorted(adjacency[current])
                    if not neighbors:
                        break
                    if len(walk) == 1:
                        walk.append(rng.choice(neighbors))
                        continue
                    previous = walk[-2]
                    weights = [
                        self._transition_weight(previous, current, candidate, adjacency)
                        for candidate in neighbors
                    ]
                    walk.append(rng.choices(neighbors, weights=weights, k=1)[0])
                walks.append(walk)
        return walks

    def _transition_weight(
        self,
        previous: str,
        current: str,
        candidate: str,
        adjacency: dict[str, set[str]] | defaultdict[str, set[str]],
    ) -> float:
        del current
        if candidate == previous:
            return 1.0 / self.p
        if candidate in adjacency[previous]:
            return 1.0
        return 1.0 / self.q

    def _context_examples(
        self,
        walks: Sequence[Sequence[str]],
    ) -> list[tuple[str, str]]:
        examples: list[tuple[str, str]] = []
        for walk in walks:
            for index, center in enumerate(walk):
                lower = max(0, index - self.context_window)
                upper = min(len(walk), index + self.context_window + 1)
                examples.extend(
                    (center, walk[context_index])
                    for context_index in range(lower, upper)
                    if context_index != index
                )
        return examples

    def _update_pair(
        self,
        input_embeddings: np.ndarray,
        output_embeddings: np.ndarray,
        center: int,
        context: int,
        *,
        label: float,
    ) -> None:
        center_vector = input_embeddings[center].copy()
        context_vector = output_embeddings[context].copy()
        logit = float(center_vector @ context_vector)
        if not math.isfinite(logit):
            return
        error = sigmoid(logit) - label
        input_embeddings[center] -= self.learning_rate * error * context_vector
        output_embeddings[context] -= self.learning_rate * error * center_vector
